// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_llm_infer_request.hpp"

#include <algorithm>

#include "infer_request_utils.hpp"
#include "llm_compiled_model.hpp"
#include "logging.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "util.hpp"

namespace ov {
namespace npuw {

PagedLLMInferRequest::PagedLLMInferRequest(const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled_model)
    : LLMInferBaseRequest(compiled_model) {
    init_ports();

    m_block_size = compiled_model->m_pa_block_size;
    m_max_seqs = compiled_model->m_pa_max_seqs;
    // The NPU-resident tile-loop dispatch path is currently inactive on the
    // LLMPipeline path (see PAGED_ATTN_LOWERING_DESIGN.md). The CPU PA
    // executor handles every PA op compile, and it requires q_len = B_token
    // (padded length) to keep concat_pastkv's slot_mapping initialised.
    // Flip this to true once the NPU dispatch path is wired end-to-end:
    // tile sub-models compiled, partition_stage attn isolation enabled,
    // and Q layout reshape from [B_token, H*S] to [1, H, B_token, S]
    // landed in attn_subgraph::PA dispatch.
    m_npu_lowering = false;

    // Inner models may use either input_ids or inputs_embeds (VLM path).
    m_input_ids_name =
        ov::npuw::util::find_port_by_name(compiled_model->m_prefill_compiled->inputs(), layer_names::input_ids)
                .has_value()
            ? layer_names::input_ids
            : layer_names::inputs_embeds;

    m_prefill_request = compiled_model->m_prefill_compiled->create_infer_request();
    for (const auto& port : m_prefill_request->get_compiled_model()->inputs()) {
        m_prefill_in_ports.emplace(port.get_any_name(), port);
    }
    for (const auto& port : m_prefill_request->get_compiled_model()->outputs()) {
        m_prefill_out_ports.emplace(port.get_any_name(), port);
    }

    // Single-variant generate (no pyramid fan-out for PA).
    m_generate_request = compiled_model->m_generate_compiled_variants.back()->create_infer_request();
    for (const auto& port : m_generate_request->get_compiled_model()->inputs()) {
        m_generate_in_ports.emplace(port.get_any_name(), port);
    }
    for (const auto& port : m_generate_request->get_compiled_model()->outputs()) {
        m_generate_out_ports.emplace(port.get_any_name(), port);
    }

    // Optional shared LM head.
    //
    // The non-PA flow statically slices output_embeds down to the LM head's
    // [1, 1, hidden] input shape, so a single shared tensor works for both
    // requests. PA flips the leading dim to B_token (rank-3 [B_token, 1, H])
    // and SliceOutEmbeds picks the wrong axis, so we keep the rank-3 buffer
    // intact and copy the live last token's row into the LM head input at
    // runtime. Generate emits a single-row buffer either way, so the shared
    // tensor still works there.
    if (compiled_model->m_lm_head_compiled) {
        m_lm_head_request = compiled_model->m_lm_head_compiled->create_infer_request();
        m_lm_head_logits_port = m_lm_head_request->get_outputs()[0];
        m_lm_head_input_port = m_lm_head_request->get_inputs()[0];
        m_lm_head_input_tensor = m_lm_head_request->get_tensor(m_lm_head_input_port);
        m_generate_request->set_tensor(m_generate_out_ports.at(layer_names::output_embeds), m_lm_head_input_tensor);
    }

    // Allocate KV cache tensors (one per layer) and share them between prefill
    // and generate requests. The PA op reads/writes block-paged storage, so
    // the same physical tensor is correct for both phases.
    for (const auto& [name, port] : m_prefill_in_ports) {
        const bool is_key = name.find("key_cache.") == 0;
        const bool is_value = name.find("value_cache.") == 0;
        if (!is_key && !is_value) {
            continue;
        }
        auto tensor =
            ov::SoPtr<ov::ITensor>(ov::make_tensor(port.get_element_type(), port.get_shape()), nullptr);
        ov::npuw::util::fill_tensor_bytes(tensor, 0u);

        m_prefill_request->set_tensor(port, tensor);
        if (auto it = m_generate_in_ports.find(name); it != m_generate_in_ports.end()) {
            m_generate_request->set_tensor(it->second, tensor);
        }
        (is_key ? m_key_cache_tensors : m_value_cache_tensors).push_back(tensor);
    }

    LOG_DEBUG("PagedLLMInferRequest: allocated " << m_key_cache_tensors.size() << " KV cache layers, block_size="
                                                 << m_block_size << " max_seqs=" << m_max_seqs);
}

void PagedLLMInferRequest::prepare_for_new_conversation() {
    m_num_stored_tokens = 0;
    m_first_infer = true;
    for (auto& t : m_key_cache_tensors) {
        ov::npuw::util::fill_tensor_bytes(t, 0u);
    }
    for (auto& t : m_value_cache_tensors) {
        ov::npuw::util::fill_tensor_bytes(t, 0u);
    }
}

void PagedLLMInferRequest::update_block_table(uint32_t tokens_this_step) {
    // Single-sequence contiguous block table:
    //   past_lens            = [m_num_stored_tokens]
    //   subsequence_begins   = [0, tokens_this_step]
    //   block_indices        = [0, 1, ..., blocks_used - 1]
    //   block_indices_begins = [0, blocks_used]
    //
    // blocks_used must cover every slot the PA op will touch — past + current —
    // otherwise a new K/V row lands outside any reserved block and the
    // subsequent attention read pulls garbage.
    const uint32_t total_tokens = m_num_stored_tokens + tokens_this_step;
    uint32_t blocks_used = (total_tokens + m_block_size - 1) / m_block_size;
    if (blocks_used == 0) {
        blocks_used = 1;
    }

    auto fill_pa_inputs = [&](const PortsMap& ports, std::shared_ptr<ov::IAsyncInferRequest>& req) {
        if (auto it = ports.find("past_lens"); it != ports.end()) {
            auto t = req->get_tensor(it->second);
            auto* data = t->data<int32_t>();
            data[0] = static_cast<int32_t>(m_num_stored_tokens);
            for (uint32_t i = 1; i < m_max_seqs; ++i) {
                data[i] = 0;
            }
        }
        if (auto it = ports.find("subsequence_begins"); it != ports.end()) {
            auto t = req->get_tensor(it->second);
            auto* data = t->data<int32_t>();
            data[0] = 0;
            data[1] = static_cast<int32_t>(tokens_this_step);
            for (uint32_t i = 2; i <= m_max_seqs; ++i) {
                data[i] = data[1];
            }
        }
        if (auto it = ports.find("block_indices"); it != ports.end()) {
            auto t = req->get_tensor(it->second);
            auto* data = t->data<int32_t>();
            for (uint32_t i = 0; i < blocks_used; ++i) {
                data[i] = static_cast<int32_t>(i);
            }
            for (size_t i = blocks_used; i < t->get_size(); ++i) {
                data[i] = 0;
            }
        }
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

    fill_pa_inputs(m_prefill_in_ports, m_prefill_request);
    fill_pa_inputs(m_generate_in_ports, m_generate_request);
}

void PagedLLMInferRequest::infer_prefill(ov::SoPtr<ov::ITensor> input_ids,
                                         ov::SoPtr<ov::ITensor> position_ids) {
    LOG_DEBUG("PagedLLMInferRequest: prefill");

    const auto seq_len = static_cast<uint32_t>(input_ids->get_shape()[1]);

    // The inner prefill model has rank-1 input shape [B_token = max_prompt_size]
    // baked by BakePagedAttentionStaticShapes. The PA op's q_len comes from
    // subsequence_begins[1] - subsequence_begins[0]; what we declare for q_len
    // depends on the executor:
    //
    //  CPU PA fallback (default):
    //    q_len = B_token. The CPU executor's concat_pastkv kernel iterates
    //    b = 0..B_token-1 over the K/V input tensors but only writes
    //    _slot_mapping entries for indices [0, q_len). With q_len < B_token
    //    the trailing slot_mapping entries hold uninitialised data and the
    //    kernel scatters K/V into garbage block slots, corrupting cache
    //    blocks that legitimate Q rows later read from. Setting q_len ==
    //    B_token gives every padded row a valid contiguous slot; PA's
    //    built-in causal mask ensures real Q rows [0, seq_len) only see
    //    real K rows, and the padded Q outputs are discarded.
    //
    //  NPU lowering (NPUW_ATTN_PAGED_NPU_LOWERING=YES):
    //    q_len = seq_len (real prompt length). The NPU dispatch in
    //    attn_subgraph.cpp reads new_tokens = subseq_begins[1] - [0] and
    //    writes that many K/V rows into the tail block; it asserts
    //    tail_offset + new_tokens <= block_size. With q_len = B_token > 32
    //    the assert fires, and even if it didn't, the tile loop would
    //    fold padded Q into the running attention state.
    auto padded_input = m_prefill_request->get_tensor(m_prefill_in_ports.at(m_input_ids_name));
    const uint32_t b_token = static_cast<uint32_t>(padded_input->get_shape().at(0));
    const uint32_t q_len = m_npu_lowering ? seq_len : b_token;
    update_block_table(q_len);

    // Copy the real prompt to the head of the padded buffer; zero the tail.
    ov::npuw::util::fill_tensor_bytes(padded_input, 0u);
    std::copy_n(reinterpret_cast<uint8_t*>(input_ids->data()),
                input_ids->get_byte_size(),
                reinterpret_cast<uint8_t*>(padded_input->data()));

    // position_ids: rank-1 [B_token]. Real positions go to the head; the padded
    // tail gets monotonically increasing values (seq_len, seq_len+1, ...) so
    // every Q row has a unique RoPE phase. Padded outputs are discarded; we
    // just want them to be valid numerical inputs.
    if (auto it = m_prefill_in_ports.find(layer_names::position_ids); it != m_prefill_in_ports.end()) {
        auto padded_pos = m_prefill_request->get_tensor(it->second);
        auto* dst = padded_pos->data<int64_t>();
        if (position_ids && position_ids->get_size() > 0) {
            const auto* src = position_ids->data<int64_t>();
            const size_t real_n = std::min<size_t>(position_ids->get_size(), b_token);
            std::copy_n(src, real_n, dst);
            for (size_t i = real_n; i < b_token; ++i) {
                dst[i] = static_cast<int64_t>(i);
            }
        } else {
            for (size_t i = 0; i < b_token; ++i) {
                dst[i] = static_cast<int64_t>(i);
            }
        }
    }

    m_prefill_request->infer();
    m_num_stored_tokens += seq_len;

    if (m_lm_head_request) {
        // PA prefill output_embeds is [B_token, 1, hidden]; the real last token
        // sits at row (seq_len - 1). Copy that row into the LM head input.
        auto prefill_embeds = m_prefill_request->get_tensor(m_prefill_out_ports.at(layer_names::output_embeds));
        const auto& shape = prefill_embeds->get_shape();
        OPENVINO_ASSERT(shape.size() == 3 && shape[1] == 1,
                        "PA prefill output_embeds expected to be rank-3 [B_token, 1, H], got ",
                        shape);
        const size_t row_bytes = shape[2] * prefill_embeds->get_element_type().size();
        const size_t src_row = static_cast<size_t>(seq_len) - 1;
        const auto* src = reinterpret_cast<const uint8_t*>(prefill_embeds->data()) + src_row * row_bytes;
        OPENVINO_ASSERT(m_lm_head_input_tensor->get_byte_size() >= row_bytes);
        std::copy_n(src, row_bytes, reinterpret_cast<uint8_t*>(m_lm_head_input_tensor->data()));

        m_lm_head_request->infer();
        m_logits = m_lm_head_request->get_tensor(m_lm_head_logits_port);
    } else {
        m_logits = m_prefill_request->get_tensor(m_prefill_out_ports.at(layer_names::logits));
    }
}

void PagedLLMInferRequest::infer_generate(ov::SoPtr<ov::ITensor> input_ids,
                                          ov::SoPtr<ov::ITensor> position_ids) {
    LOG_DEBUG("PagedLLMInferRequest: generate");

    update_block_table(/*tokens_this_step=*/1);

    auto gen_input = m_generate_request->get_tensor(m_generate_in_ports.at(m_input_ids_name));
    std::copy_n(reinterpret_cast<uint8_t*>(input_ids->data()),
                input_ids->get_byte_size(),
                reinterpret_cast<uint8_t*>(gen_input->data()));

    if (position_ids && position_ids->get_size() > 0) {
        if (auto it = m_generate_in_ports.find(layer_names::position_ids); it != m_generate_in_ports.end()) {
            auto gen_pos = m_generate_request->get_tensor(it->second);
            std::copy_n(reinterpret_cast<const uint8_t*>(position_ids->data()),
                        std::min(position_ids->get_byte_size(), gen_pos->get_byte_size()),
                        reinterpret_cast<uint8_t*>(gen_pos->data()));
        }
    }

    m_generate_request->infer();
    m_num_stored_tokens += 1;

    if (m_lm_head_request) {
        m_lm_head_request->infer();
        m_logits = m_lm_head_request->get_tensor(m_lm_head_logits_port);
    } else {
        m_logits = m_generate_request->get_tensor(m_generate_out_ports.at(layer_names::logits));
    }
}

void PagedLLMInferRequest::infer() {
    const auto& inputs = get_inputs();

    auto input_ids_port = ov::npuw::util::find_port_by_name(inputs, m_input_ids_name);
    OPENVINO_ASSERT(input_ids_port.has_value(), "PA: ", m_input_ids_name, " port missing on outer model");
    auto input_ids = get_tensor(input_ids_port.value());

    auto position_ids_port = ov::npuw::util::find_port_by_name(inputs, layer_names::position_ids);
    auto position_ids =
        position_ids_port.has_value() ? get_tensor(position_ids_port.value()) : ov::SoPtr<ov::ITensor>();

    // attention_mask is intentionally not read: SDPAToPagedAttention drops it
    // and PA derives causal information from past_lens / subsequence_begins.

    // Outer-facing input_ids stays rank-2 [batch, seq]; the rank-1 PA flattening
    // only applies to the inner kvcache_model.
    const auto& shape = input_ids->get_shape();
    const auto seq_len = shape.size() >= 2 ? shape[1] : shape[0];

    if (seq_len > 1 || m_first_infer) {
        prepare_for_new_conversation();
        infer_prefill(input_ids, position_ids);
        m_first_infer = false;
    } else {
        infer_generate(input_ids, position_ids);
    }
}

ov::SoPtr<ov::ITensor> PagedLLMInferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    if (port.get_any_name() == layer_names::logits && m_logits) {
        return m_logits;
    }
    return ISyncInferRequest::get_tensor(port);
}

}  // namespace npuw
}  // namespace ov
