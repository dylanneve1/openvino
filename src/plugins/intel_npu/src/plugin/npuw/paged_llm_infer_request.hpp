// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "llm_compiled_model.hpp"
#include "llm_infer_base_request.hpp"
#include "openvino/core/descriptor/output.hpp"

namespace ov {
namespace npuw {

// Paged Attention infer request for single-sequence LLMPipeline path.
// Self-manages a trivial contiguous block table (blocks 0..K-1) since
// GenAI's ContinuousBatchingPipeline provides its own block_indices and
// this class handles the simpler LLMPipeline case only.
class PagedLLMInferRequest : public ov::npuw::LLMInferBaseRequest {
public:
    explicit PagedLLMInferRequest(const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled_model);

    void infer() override;

    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override;

private:
    void prepare_for_new_conversation();
    void infer_prefill(ov::SoPtr<ov::ITensor> input_ids,
                       ov::SoPtr<ov::ITensor> attention_mask,
                       ov::SoPtr<ov::ITensor> position_ids);
    void infer_generate(ov::SoPtr<ov::ITensor> input_ids,
                        ov::SoPtr<ov::ITensor> attention_mask,
                        ov::SoPtr<ov::ITensor> position_ids);
    void update_block_table(uint32_t tokens_this_step);

    std::shared_ptr<ov::IAsyncInferRequest> m_prefill_request;
    std::shared_ptr<ov::IAsyncInferRequest> m_generate_request;

    PortsMap m_prefill_in_ports;
    PortsMap m_prefill_out_ports;
    PortsMap m_generate_in_ports;
    PortsMap m_generate_out_ports;

    // This model is optional, so can be null.
    std::shared_ptr<ov::IAsyncInferRequest> m_lm_head_request;
    ov::Output<const ov::Node> m_lm_head_input_port;
    ov::Output<const ov::Node> m_lm_head_logits_port;
    ov::SoPtr<ov::ITensor> m_lm_head_input_tensor;

    ov::SoPtr<ov::ITensor> m_logits;

    // Paged attention state: single-sequence contiguous blocks.
    uint32_t m_num_stored_tokens = 0;
    uint32_t m_block_size = 0;
    uint32_t m_num_blocks = 0;
    uint32_t m_max_seqs = 0;

    // Allocated KV cache tensors (one per layer). Owned by this request
    // for the single-sequence path; GenAI CB pipeline would set_tensor
    // externally in the multi-sequence path.
    std::vector<ov::SoPtr<ov::ITensor>> m_key_cache_tensors;
    std::vector<ov::SoPtr<ov::ITensor>> m_value_cache_tensors;

    bool m_first_infer = true;

    // Real prompt length last seen at prefill, before padding to max_prompt_size.
    // The shared LM head reads the embedding at position [seq_len - 1] of the
    // prefill output to produce logits for the first generated token.
    uint32_t m_last_prefill_seq_len = 0;

    std::string m_input_ids_name;
};

}  // namespace npuw
}  // namespace ov
