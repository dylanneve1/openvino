// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "kv_cache_block_manager.hpp"
#include "llm_compiled_model.hpp"
#include "llm_infer_base_request.hpp"
#include "online_softmax_tile.hpp"
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
    void update_block_table();

    std::shared_ptr<ov::IAsyncInferRequest> m_prefill_request;
    std::shared_ptr<ov::IAsyncInferRequest> m_generate_request;

    PortsMap m_prefill_in_ports;
    PortsMap m_prefill_out_ports;
    PortsMap m_generate_in_ports;
    PortsMap m_generate_out_ports;

    // This model is optional, so can be null.
    std::shared_ptr<ov::IAsyncInferRequest> m_lm_head_request;
    ov::Output<const ov::Node> m_lm_head_logits_port;

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

    std::string m_input_ids_name;

    // ------------------------------------------------------------------
    // PA NPU-resident tile-loop state (Step 5a)
    //
    // When the LLM was compiled with m_pa_mode = true AND
    // LLMCompiledModel::setup_paged_runtime() successfully built + compiled
    // the tile sub-models, the following members hold the per-iteration
    // tile infer-requests plus the persistent (acc, max, d) state tensors
    // threaded between iterations.
    //
    // These are NOT yet plumbed into the main infer() path — wiring the
    // tile loop into prefill/generate execution requires NPUW partitioning
    // to route the PA op's subgraph away from the CPU fallback into this
    // runtime path. That is Step 5b/c. For now, the helpers below are
    // building blocks ready for that integration and exercise paths for
    // testing.
    //
    // pa_tile_loop_available() tells the caller whether the lowering is
    // ready to take over.
    bool pa_tile_loop_available() const;

    // Initialise (acc, max, d) state tensors for a fresh tile-loop pass.
    // acc → 0, max → very-negative, d → 0. State shape matches the tile
    // sub-model's past_acc / past_max / past_d parameter layout.
    void init_tile_loop_state();

    // Run the tile loop for one layer's worth of K/V over the active
    // blocks indicated by block_indices[0..num_active_blocks). Threads
    // (acc, max, d) state between iterations via the existing state
    // tensors. layer_idx selects which KVCacheBlockManager pair in
    // m_compiled_model->m_pa_layer_managers to source K/V from.
    //
    // This is the per-layer attention compute the partitioning-level
    // wiring (Step 5b) will invoke in place of the PA op's CPU execution.
    void run_paged_tile_loop_for_layer(std::size_t layer_idx,
                                       ov::SoPtr<ov::ITensor> q_tile,
                                       ov::SoPtr<ov::ITensor> mask_tile,
                                       const int32_t* block_indices,
                                       std::size_t num_active_blocks,
                                       ov::SoPtr<ov::ITensor> present_key,
                                       ov::SoPtr<ov::ITensor> present_value,
                                       ov::SoPtr<ov::ITensor> attention_output);

    std::shared_ptr<ov::IAsyncInferRequest> m_pa_tile_request;
    std::shared_ptr<ov::IAsyncInferRequest> m_pa_final_tile_request;

    // Per-iteration state tensors. Same shape as the tile sub-model's
    // past_acc / past_max / past_d inputs. Allocated lazily on first
    // init_tile_loop_state() call.
    ov::SoPtr<ov::ITensor> m_pa_state_acc;
    ov::SoPtr<ov::ITensor> m_pa_state_max;
    ov::SoPtr<ov::ITensor> m_pa_state_d;
};

}  // namespace npuw
}  // namespace ov
