// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_llm_infer_request.hpp"

#include "infer_request_utils.hpp"
#include "llm_compiled_model.hpp"
#include "logging.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
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
