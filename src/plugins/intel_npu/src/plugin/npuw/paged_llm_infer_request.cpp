// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_llm_infer_request.hpp"

#include <cstring>
#include <limits>

#include "infer_request_utils.hpp"
#include "llm_compiled_model.hpp"
#include "logging.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "util.hpp"

namespace ov {
namespace npuw {

PagedLLMInferRequest::PagedLLMInferRequest(const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled_model)
    : LLMInferBaseRequest(compiled_model) {
    m_block_size = compiled_model->m_pa_block_size;
    m_num_blocks = compiled_model->m_pa_num_blocks;
    m_max_seqs = compiled_model->m_pa_max_seqs;

    // Determine input name: input_ids or inputs_embeds
    m_input_ids_name = ov::npuw::util::find_port_by_name(compiled_model->m_prefill_compiled->inputs(),
                                                          layer_names::input_ids)
                               .has_value()
                           ? layer_names::input_ids
                           : layer_names::inputs_embeds;

    // Create prefill request
    m_prefill_request = compiled_model->m_prefill_compiled->create_infer_request();
    for (const auto& port : m_prefill_request->get_compiled_model()->inputs()) {
        m_prefill_in_ports.emplace(port.get_any_name(), port);
    }
    for (const auto& port : m_prefill_request->get_compiled_model()->outputs()) {
        m_prefill_out_ports.emplace(port.get_any_name(), port);
    }

    // Create generate request (single variant for PA — no multi-variant fan-out)
    m_generate_request = compiled_model->m_generate_compiled_variants.back()->create_infer_request();
    for (const auto& port : m_generate_request->get_compiled_model()->inputs()) {
        m_generate_in_ports.emplace(port.get_any_name(), port);
    }
    for (const auto& port : m_generate_request->get_compiled_model()->outputs()) {
        m_generate_out_ports.emplace(port.get_any_name(), port);
    }

    // Optional LM head
    if (compiled_model->m_lm_head_compiled) {
        m_lm_head_request = compiled_model->m_lm_head_compiled->create_infer_request();
        m_lm_head_logits_port = m_lm_head_request->get_outputs()[0];
        const auto lm_head_input_port = m_lm_head_request->get_inputs()[0];
        // Share embed output → LM head input
        m_prefill_request->set_tensor(m_prefill_out_ports.at(layer_names::output_embeds),
                                      m_lm_head_request->get_tensor(lm_head_input_port));
        m_generate_request->set_tensor(m_generate_out_ports.at(layer_names::output_embeds),
                                       m_lm_head_request->get_tensor(lm_head_input_port));
    }

    // Allocate KV cache tensors and share between prefill and generate requests.
    // Find all key_cache.N / value_cache.N Parameters by scanning prefill inputs.
    for (const auto& [name, port] : m_prefill_in_ports) {
        if (name.find("key_cache.") == 0 || name.find("value_cache.") == 0) {
            auto shape = port.get_shape();
            auto elem_type = port.get_element_type();
            auto tensor = ov::SoPtr<ov::ITensor>(ov::make_tensor(elem_type, shape), nullptr);
            ov::npuw::util::fill_tensor_bytes(tensor, 0u);

            m_prefill_request->set_tensor(port, tensor);

            // Share the same tensor with generate model (same Parameter name expected)
            if (m_generate_in_ports.count(name)) {
                m_generate_request->set_tensor(m_generate_in_ports.at(name), tensor);
            }

            if (name.find("key_cache.") == 0) {
                m_key_cache_tensors.push_back(tensor);
            } else {
                m_value_cache_tensors.push_back(tensor);
            }
        }
    }

    LOG_DEBUG("PagedLLMInferRequest: allocated " << m_key_cache_tensors.size() << " KV cache layers, block_size="
                                                 << m_block_size << " num_blocks=" << m_num_blocks);

    // PA tile-loop infer requests for the NPU-resident lowering. Created
    // only when LLMCompiledModel::setup_paged_runtime() built and compiled
    // both tile sub-models. When unavailable, the request continues to use
    // the existing CPU PA fallback through m_prefill_request /
    // m_generate_request.
    if (compiled_model->m_pa_runtime && compiled_model->m_pa_runtime->is_valid() &&
        compiled_model->m_pa_tile_compiled && compiled_model->m_pa_final_tile_compiled) {
        m_pa_tile_request = compiled_model->m_pa_tile_compiled->create_infer_request();
        m_pa_final_tile_request = compiled_model->m_pa_final_tile_compiled->create_infer_request();
        LOG_INFO("PagedLLMInferRequest: PA NPU tile-loop infer requests created. "
                 "Lowering ready (Step 5b wiring still required to substitute the CPU PA path).");
    } else {
        LOG_DEBUG("PagedLLMInferRequest: PA NPU lowering not available; CPU PA fallback in effect.");
    }
}

bool PagedLLMInferRequest::pa_tile_loop_available() const {
    return m_pa_tile_request && m_pa_final_tile_request && m_npuw_llm_compiled_model &&
           m_npuw_llm_compiled_model->m_pa_runtime && m_npuw_llm_compiled_model->m_pa_runtime->is_valid();
}

void PagedLLMInferRequest::init_tile_loop_state() {
    if (!pa_tile_loop_available()) {
        return;
    }

    const auto& tile_inputs = m_pa_tile_request->get_inputs();
    auto port_of = [&](OnlineSoftmaxTileInputId id) {
        return tile_inputs.at(static_cast<std::size_t>(id));
    };

    // Lazy allocation. The state tensors are sized by the tile sub-model's
    // past_* parameter shapes (set during create_online_softmax_tile_model).
    if (!m_pa_state_acc) {
        const auto acc_port = port_of(OnlineSoftmaxTileInputId::PAST_ACC);
        const auto max_port = port_of(OnlineSoftmaxTileInputId::PAST_MAX);
        const auto d_port = port_of(OnlineSoftmaxTileInputId::PAST_D);
        const auto mask_port = port_of(OnlineSoftmaxTileInputId::MASK_TILE);

        m_pa_state_acc =
            ov::SoPtr<ov::ITensor>(ov::make_tensor(acc_port.get_element_type(), acc_port.get_shape()), nullptr);
        m_pa_state_max =
            ov::SoPtr<ov::ITensor>(ov::make_tensor(max_port.get_element_type(), max_port.get_shape()), nullptr);
        m_pa_state_d =
            ov::SoPtr<ov::ITensor>(ov::make_tensor(d_port.get_element_type(), d_port.get_shape()), nullptr);
        m_pa_mask_tile_buffer =
            ov::SoPtr<ov::ITensor>(ov::make_tensor(mask_port.get_element_type(), mask_port.get_shape()), nullptr);
    }

    // acc = 0, d = 0.
    ov::npuw::util::fill_tensor_bytes(m_pa_state_acc, 0u);
    ov::npuw::util::fill_tensor_bytes(m_pa_state_d, 0u);

    // max = "very negative" — initial value for the online-softmax running
    // max. Use the most-negative finite value representable in the state
    // element type so the first tile's local max always wins comparison.
    const auto max_elem_type = m_pa_state_max->get_element_type();
    if (max_elem_type == ov::element::f32) {
        auto* data = m_pa_state_max->data<float>();
        const float neg_inf = std::numeric_limits<float>::lowest();
        for (std::size_t i = 0; i < m_pa_state_max->get_size(); ++i) {
            data[i] = neg_inf;
        }
    } else if (max_elem_type == ov::element::f16) {
        // IEEE 754 binary16 most-negative finite value is -65504, encoded as 0xFBFF.
        constexpr uint16_t f16_neg_inf_bits = 0xFBFF;
        auto* data = reinterpret_cast<uint16_t*>(m_pa_state_max->data());
        for (std::size_t i = 0; i < m_pa_state_max->get_size(); ++i) {
            data[i] = f16_neg_inf_bits;
        }
    } else {
        OPENVINO_THROW("PagedLLMInferRequest::init_tile_loop_state: unsupported PAST_MAX element type ",
                       max_elem_type.get_type_name(),
                       " (expected f16 or f32)");
    }
}

namespace {
// IEEE 754 binary16 most-negative finite value (-65504) bit pattern, used to
// flood mask tiles with a "very negative" sentinel when the state precision
// is f16. softmax(qkm − maxx) on these entries underflows to ~0, the
// intended behaviour for masked-out positions.
constexpr uint16_t kF16NegInfBits = 0xFBFF;
}  // namespace

void PagedLLMInferRequest::fill_mask_tile_for_cached_block(uint32_t past_lens, uint32_t block_idx) {
    if (!m_pa_mask_tile_buffer) {
        OPENVINO_THROW("PagedLLMInferRequest::fill_mask_tile_for_cached_block: mask buffer not allocated; "
                       "call init_tile_loop_state() first.");
    }
    const auto shape = m_pa_mask_tile_buffer->get_shape();
    NPUW_ASSERT(shape.size() == 4);
    const auto query_size = shape[2];
    const auto block_size = shape[3];

    // Absolute key-position of the first slot in this logical block.
    const uint64_t k_abs_start = static_cast<uint64_t>(block_idx) * static_cast<uint64_t>(block_size);

    const auto elem_type = m_pa_mask_tile_buffer->get_element_type();
    if (elem_type == ov::element::f32) {
        auto* data = m_pa_mask_tile_buffer->data<float>();
        const float neg_inf = std::numeric_limits<float>::lowest();
        for (std::size_t q = 0; q < query_size; ++q) {
            float* row = data + q * block_size;
            for (std::size_t k = 0; k < block_size; ++k) {
                const auto k_abs = k_abs_start + k;
                row[k] = (k_abs < past_lens) ? 0.0f : neg_inf;
            }
        }
    } else if (elem_type == ov::element::f16) {
        auto* data = reinterpret_cast<uint16_t*>(m_pa_mask_tile_buffer->data());
        for (std::size_t q = 0; q < query_size; ++q) {
            uint16_t* row = data + q * block_size;
            for (std::size_t k = 0; k < block_size; ++k) {
                const auto k_abs = k_abs_start + k;
                row[k] = (k_abs < past_lens) ? uint16_t{0x0000} : kF16NegInfBits;
            }
        }
    } else {
        OPENVINO_THROW("PagedLLMInferRequest::fill_mask_tile_for_cached_block: unsupported mask element type ",
                       elem_type.get_type_name());
    }
}

void PagedLLMInferRequest::fill_mask_tile_for_present(uint32_t new_tokens) {
    if (!m_pa_mask_tile_buffer) {
        OPENVINO_THROW("PagedLLMInferRequest::fill_mask_tile_for_present: mask buffer not allocated; "
                       "call init_tile_loop_state() first.");
    }
    const auto shape = m_pa_mask_tile_buffer->get_shape();
    NPUW_ASSERT(shape.size() == 4);
    const auto query_size = shape[2];
    const auto block_size = shape[3];

    // For prefill, query_size matches the number of new tokens; for generate
    // it is 1. Either way, valid (q, k) pairs in the present-K/V tile are
    // those with k <= q AND k < new_tokens. q and k are 0-indexed within
    // the new-token window — their absolute positions are
    //   q_abs = past_lens + q,  k_abs = past_lens + k
    // and the relation simplifies to k <= q since past_lens cancels.
    const auto elem_type = m_pa_mask_tile_buffer->get_element_type();
    if (elem_type == ov::element::f32) {
        auto* data = m_pa_mask_tile_buffer->data<float>();
        const float neg_inf = std::numeric_limits<float>::lowest();
        for (std::size_t q = 0; q < query_size; ++q) {
            float* row = data + q * block_size;
            for (std::size_t k = 0; k < block_size; ++k) {
                const bool valid = (k <= q) && (k < new_tokens);
                row[k] = valid ? 0.0f : neg_inf;
            }
        }
    } else if (elem_type == ov::element::f16) {
        auto* data = reinterpret_cast<uint16_t*>(m_pa_mask_tile_buffer->data());
        for (std::size_t q = 0; q < query_size; ++q) {
            uint16_t* row = data + q * block_size;
            for (std::size_t k = 0; k < block_size; ++k) {
                const bool valid = (k <= q) && (k < new_tokens);
                row[k] = valid ? uint16_t{0x0000} : kF16NegInfBits;
            }
        }
    } else {
        OPENVINO_THROW("PagedLLMInferRequest::fill_mask_tile_for_present: unsupported mask element type ",
                       elem_type.get_type_name());
    }
}

void PagedLLMInferRequest::run_paged_tile_loop_for_layer(std::size_t layer_idx,
                                                          ov::SoPtr<ov::ITensor> q_tile,
                                                          uint32_t past_lens,
                                                          uint32_t new_tokens,
                                                          const int32_t* block_indices,
                                                          std::size_t num_active_blocks,
                                                          ov::SoPtr<ov::ITensor> present_key,
                                                          ov::SoPtr<ov::ITensor> present_value,
                                                          ov::SoPtr<ov::ITensor> attention_output) {
    if (!pa_tile_loop_available()) {
        OPENVINO_THROW("PagedLLMInferRequest::run_paged_tile_loop_for_layer: PA NPU lowering not available "
                       "(setup_paged_runtime failed or tile sub-models not compiled).");
    }
    if (!block_indices && num_active_blocks > 0) {
        OPENVINO_THROW("PagedLLMInferRequest::run_paged_tile_loop_for_layer: num_active_blocks > 0 but "
                       "block_indices is null");
    }
    if (!m_pa_mask_tile_buffer) {
        OPENVINO_THROW("PagedLLMInferRequest::run_paged_tile_loop_for_layer: mask buffer not allocated; "
                       "call init_tile_loop_state() first.");
    }

    const auto& layer_managers = m_npuw_llm_compiled_model->m_pa_layer_managers;
    if (layer_idx >= layer_managers.size()) {
        OPENVINO_THROW("PagedLLMInferRequest::run_paged_tile_loop_for_layer: layer ",
                       layer_idx,
                       " out of range (have ",
                       layer_managers.size(),
                       " pools)");
    }

    const auto& layer = layer_managers[layer_idx];
    if (!layer.key || !layer.value) {
        OPENVINO_THROW("PagedLLMInferRequest::run_paged_tile_loop_for_layer: layer ",
                       layer_idx,
                       " has no K or V block manager");
    }

    // Bind inputs that don't change across iterations.
    const auto& tile_inputs = m_pa_tile_request->get_inputs();
    const auto& tile_outputs = m_pa_tile_request->get_outputs();

    auto bind_tile_input = [&](OnlineSoftmaxTileInputId id, const ov::SoPtr<ov::ITensor>& t) {
        m_pa_tile_request->set_tensor(tile_inputs.at(static_cast<std::size_t>(id)), t);
    };
    auto bind_tile_output_to_state = [&](OnlineSoftmaxTileOutputId id, const ov::SoPtr<ov::ITensor>& state) {
        m_pa_tile_request->set_tensor(tile_outputs.at(static_cast<std::size_t>(id)), state);
    };

    bind_tile_input(OnlineSoftmaxTileInputId::Q, q_tile);
    bind_tile_input(OnlineSoftmaxTileInputId::MASK_TILE, m_pa_mask_tile_buffer);
    bind_tile_input(OnlineSoftmaxTileInputId::PAST_ACC, m_pa_state_acc);
    bind_tile_input(OnlineSoftmaxTileInputId::PAST_MAX, m_pa_state_max);
    bind_tile_input(OnlineSoftmaxTileInputId::PAST_D, m_pa_state_d);

    // Bind outputs to the SAME state tensors. This makes the tile sub-model
    // write its updated (acc, max, d) directly into the state buffers, so
    // the next iteration's PAST_* inputs already hold the latest values.
    // Mirrors HFA's tile orchestrator in attn_subgraph.cpp:907-925.
    bind_tile_output_to_state(OnlineSoftmaxTileOutputId::ACC, m_pa_state_acc);
    bind_tile_output_to_state(OnlineSoftmaxTileOutputId::MAXX, m_pa_state_max);
    bind_tile_output_to_state(OnlineSoftmaxTileOutputId::D, m_pa_state_d);

    // Walk active blocks. For each logical block i, refill the mask tile
    // (validity gate over the cached region) and bind the layer's K/V slice
    // from the pool by physical id.
    for (std::size_t i = 0; i < num_active_blocks; ++i) {
        fill_mask_tile_for_cached_block(past_lens, static_cast<uint32_t>(i));

        const auto phys_id = static_cast<uint32_t>(block_indices[i]);
        auto k_block = layer.key->get_block_tensor(phys_id);
        auto v_block = layer.value->get_block_tensor(phys_id);
        bind_tile_input(OnlineSoftmaxTileInputId::K_TILE, k_block);
        bind_tile_input(OnlineSoftmaxTileInputId::V_TILE, v_block);
        m_pa_tile_request->infer();
        // state_* tensors now hold this iter's (acc, max, d) — ready for next.
    }

    // Final tile: present K/V (this step's new contribution) + acc/d
    // division to produce the layer's attention output. Mask switches from
    // "cached-validity" to "causal-within-new-tokens".
    fill_mask_tile_for_present(new_tokens);

    const auto& final_inputs = m_pa_final_tile_request->get_inputs();
    const auto& final_outputs = m_pa_final_tile_request->get_outputs();
    auto bind_final_input = [&](OnlineSoftmaxTileInputId id, const ov::SoPtr<ov::ITensor>& t) {
        m_pa_final_tile_request->set_tensor(final_inputs.at(static_cast<std::size_t>(id)), t);
    };

    bind_final_input(OnlineSoftmaxTileInputId::Q, q_tile);
    bind_final_input(OnlineSoftmaxTileInputId::MASK_TILE, m_pa_mask_tile_buffer);
    bind_final_input(OnlineSoftmaxTileInputId::PAST_ACC, m_pa_state_acc);
    bind_final_input(OnlineSoftmaxTileInputId::PAST_MAX, m_pa_state_max);
    bind_final_input(OnlineSoftmaxTileInputId::PAST_D, m_pa_state_d);
    bind_final_input(OnlineSoftmaxTileInputId::K_TILE, present_key);
    bind_final_input(OnlineSoftmaxTileInputId::V_TILE, present_value);

    // Write the attention output directly into the caller's buffer.
    m_pa_final_tile_request->set_tensor(final_outputs.at(0), attention_output);
    m_pa_final_tile_request->infer();
}

void PagedLLMInferRequest::prepare_for_new_conversation() {
    m_num_stored_tokens = 0;
    m_first_infer = true;

    // Zero all KV cache tensors
    for (auto& t : m_key_cache_tensors) {
        ov::npuw::util::fill_tensor_bytes(t, 0u);
    }
    for (auto& t : m_value_cache_tensors) {
        ov::npuw::util::fill_tensor_bytes(t, 0u);
    }
}

void PagedLLMInferRequest::update_block_table() {
    // Single-sequence contiguous block table: blocks [0, 1, 2, ..., N-1]
    // past_lens = [m_num_stored_tokens]
    // subsequence_begins = [0, num_tokens_this_step]  -- CSR offsets into token stream
    // block_indices = [0, 1, 2, ..., blocks_used-1]
    // block_indices_begins = [0, blocks_used]

    uint32_t blocks_used = (m_num_stored_tokens + m_block_size - 1) / m_block_size;
    if (blocks_used == 0) {
        blocks_used = 1;  // At least 1 block for first token
    }

    auto fill_pa_input = [&](const PortsMap& ports, std::shared_ptr<ov::IAsyncInferRequest>& req) {
        // past_lens
        if (auto it = ports.find("past_lens"); it != ports.end()) {
            auto t = req->get_tensor(it->second);
            auto* data = t->data<int32_t>();
            data[0] = static_cast<int32_t>(m_num_stored_tokens);
            for (uint32_t i = 1; i < m_max_seqs; ++i) {
                data[i] = 0;
            }
        }
        // subsequence_begins
        if (auto it = ports.find("subsequence_begins"); it != ports.end()) {
            auto t = req->get_tensor(it->second);
            auto* data = t->data<int32_t>();
            data[0] = 0;
            // data[1] is set by the caller (number of tokens in this step)
            for (uint32_t i = 2; i <= m_max_seqs; ++i) {
                data[i] = data[1];
            }
        }
        // block_indices
        if (auto it = ports.find("block_indices"); it != ports.end()) {
            auto t = req->get_tensor(it->second);
            auto* data = t->data<int32_t>();
            for (uint32_t i = 0; i < blocks_used; ++i) {
                data[i] = static_cast<int32_t>(i);
            }
            // Sentinel the rest
            for (size_t i = blocks_used; i < t->get_size(); ++i) {
                data[i] = 0;
            }
        }
        // block_indices_begins
        if (auto it = ports.find("block_indices_begins"); it != ports.end()) {
            auto t = req->get_tensor(it->second);
            auto* data = t->data<int32_t>();
            data[0] = 0;
            data[1] = static_cast<int32_t>(blocks_used);
            for (uint32_t i = 2; i <= m_max_seqs; ++i) {
                data[i] = static_cast<int32_t>(blocks_used);
            }
        }
    };

    fill_pa_input(m_prefill_in_ports, m_prefill_request);
    fill_pa_input(m_generate_in_ports, m_generate_request);
}

void PagedLLMInferRequest::infer_prefill(ov::SoPtr<ov::ITensor> input_ids,
                                          ov::SoPtr<ov::ITensor> attention_mask,
                                          ov::SoPtr<ov::ITensor> position_ids) {
    LOG_DEBUG("PagedLLMInferRequest: prefill");
    const auto seq_len = input_ids->get_shape()[1];

    // Set subsequence_begins[1] = seq_len (number of tokens in this step)
    if (auto it = m_prefill_in_ports.find("subsequence_begins"); it != m_prefill_in_ports.end()) {
        auto t = m_prefill_request->get_tensor(it->second);
        t->data<int32_t>()[1] = static_cast<int32_t>(seq_len);
    }

    update_block_table();

    // Set input_ids
    auto padded_input = m_prefill_request->get_tensor(m_prefill_in_ports.at(m_input_ids_name));
    std::copy_n(reinterpret_cast<uint8_t*>(input_ids->data()),
                input_ids->get_byte_size(),
                reinterpret_cast<uint8_t*>(padded_input->data()) + padded_input->get_byte_size() -
                    input_ids->get_byte_size());

    // Set attention_mask
    if (auto it = m_prefill_in_ports.find(layer_names::attention_mask); it != m_prefill_in_ports.end()) {
        auto padded_mask = m_prefill_request->get_tensor(it->second);
        std::copy_n(attention_mask->data<int64_t>(),
                    attention_mask->get_size(),
                    padded_mask->data<int64_t>() + padded_mask->get_size() - attention_mask->get_size());
    }

    // Set position_ids
    if (auto it = m_prefill_in_ports.find(layer_names::position_ids); it != m_prefill_in_ports.end()) {
        auto padded_pos = m_prefill_request->get_tensor(it->second);
        ov::npuw::util::pad_position_ids(padded_pos, position_ids);
    }

    m_prefill_request->infer();
    m_num_stored_tokens += static_cast<uint32_t>(seq_len);

    // Get logits
    if (m_lm_head_request) {
        m_lm_head_request->infer();
        m_logits = m_lm_head_request->get_tensor(m_lm_head_logits_port);
    } else {
        m_logits = m_prefill_request->get_tensor(m_prefill_out_ports.at(layer_names::logits));
    }
}

void PagedLLMInferRequest::infer_generate(ov::SoPtr<ov::ITensor> input_ids,
                                           ov::SoPtr<ov::ITensor> attention_mask,
                                           ov::SoPtr<ov::ITensor> position_ids) {
    LOG_DEBUG("PagedLLMInferRequest: generate");

    // Set subsequence_begins[1] = 1 (single token)
    if (auto it = m_generate_in_ports.find("subsequence_begins"); it != m_generate_in_ports.end()) {
        auto t = m_generate_request->get_tensor(it->second);
        t->data<int32_t>()[1] = 1;
    }

    update_block_table();

    // Set input_ids (single token)
    auto gen_input = m_generate_request->get_tensor(m_generate_in_ports.at(m_input_ids_name));
    std::copy_n(reinterpret_cast<uint8_t*>(input_ids->data()),
                input_ids->get_byte_size(),
                reinterpret_cast<uint8_t*>(gen_input->data()));

    // Set attention_mask
    if (auto it = m_generate_in_ports.find(layer_names::attention_mask); it != m_generate_in_ports.end()) {
        auto gen_mask = m_generate_request->get_tensor(it->second);
        std::copy_n(attention_mask->data<int64_t>(), attention_mask->get_size(), gen_mask->data<int64_t>());
    }

    // Set position_ids
    if (auto it = m_generate_in_ports.find(layer_names::position_ids); it != m_generate_in_ports.end()) {
        auto gen_pos = m_generate_request->get_tensor(it->second);
        std::copy_n(position_ids->data<int64_t>(), position_ids->get_size(), gen_pos->data<int64_t>());
    }

    m_generate_request->infer();
    m_num_stored_tokens += 1;

    // Get logits
    if (m_lm_head_request) {
        m_lm_head_request->infer();
        m_logits = m_lm_head_request->get_tensor(m_lm_head_logits_port);
    } else {
        m_logits = m_generate_request->get_tensor(m_generate_out_ports.at(layer_names::logits));
    }
}

void PagedLLMInferRequest::infer() {
    auto input_ids = get_tensor(get_inputs()[0]);
    auto attention_mask =
        get_tensor(*ov::npuw::util::find_port_by_name(get_inputs(), layer_names::attention_mask));
    auto position_ids =
        get_tensor(*ov::npuw::util::find_port_by_name(get_inputs(), layer_names::position_ids));

    const auto seq_len = input_ids->get_shape()[1];

    if (seq_len > 1 || m_first_infer) {
        prepare_for_new_conversation();
        infer_prefill(input_ids, attention_mask, position_ids);
        m_first_infer = false;
    } else {
        infer_generate(input_ids, attention_mask, position_ids);
    }
}

ov::SoPtr<ov::ITensor> PagedLLMInferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    const auto& name = port.get_any_name();
    if (name == layer_names::logits && m_logits) {
        return m_logits;
    }
    // Fall through to base class default (allocate-on-demand from external port)
    return ISyncInferRequest::get_tensor(port);
}

}  // namespace npuw
}  // namespace ov
