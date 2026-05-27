// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_attn_to_hfa_tiles.hpp"

#include "../logging.hpp"
#include "openvino/op/paged_attention.hpp"

namespace ov::npuw {

PagedAttnToHFATiles::PagedAttnToHFATiles(uint32_t max_num_blocks,
                                         uint32_t block_size,
                                         uint32_t max_active_blocks)
    : m_max_num_blocks(max_num_blocks),
      m_block_size(block_size),
      m_max_active_blocks(max_active_blocks) {
    OPENVINO_ASSERT(max_active_blocks <= max_num_blocks,
                    "PagedAttnToHFATiles: max_active_blocks (",
                    max_active_blocks,
                    ") must be <= max_num_blocks (",
                    max_num_blocks,
                    ")");
}

bool PagedAttnToHFATiles::run_on_model(const std::shared_ptr<ov::Model>& m) {
    // SKELETON: detect PA ops and log; defer real transformation.
    //
    // Implementation roadmap (see PAGED_ATTN_LOWERING_DESIGN.md):
    //   1. For each PagedAttentionExtension node in the graph:
    //      a. Extract Q, K_cache, V_cache, block_indices, mask, scale, etc.
    //      b. Build a sub-Model representing one tile iteration. Inputs:
    //         past_acc, past_max, past_d, k_block, v_block, q, mask_slice.
    //         Body: the same MatMul/Add/Exp/ReduceMax/ReduceSum sequence
    //         HFA already uses (host_flash_attention.cpp:264-297).
    //      c. Wrap the tile sub-Model in a Loop op bounded by num_active_blocks.
    //         Each iteration:
    //            phys_id = block_indices[i]
    //            k_block = Gather(K_cache, phys_id, axis=0)
    //            v_block = Gather(V_cache, phys_id, axis=0)
    //            invoke tile sub-Model with (past_*, k_block, v_block, q, mask_slice)
    //            past_* := tile_outputs
    //      d. After the loop, divide acc / d for the final output.
    //      e. Replace the original PA node with the new subgraph.
    //   2. Return true if any node was replaced; false otherwise (so other
    //      passes can detect a no-op run).
    //
    // For now: walk the model, count PA ops, log, return false.

    std::size_t pa_count = 0;
    for (const auto& op : m->get_ordered_ops()) {
        if (ov::as_type_ptr<ov::op::PagedAttentionExtension>(op)) {
            ++pa_count;
            LOG_DEBUG("PagedAttnToHFATiles: would lower " << op->get_friendly_name() << " (NOT YET IMPLEMENTED)");
        }
    }

    if (pa_count > 0) {
        LOG_INFO("PagedAttnToHFATiles: detected " << pa_count
                                                   << " PagedAttentionExtension op(s); skeleton pass, no transformation applied. "
                                                      "Configuration: max_num_blocks="
                                                   << m_max_num_blocks << " block_size=" << m_block_size
                                                   << " max_active_blocks=" << m_max_active_blocks);
    }

    // No-op until the lowering is implemented.
    return false;
}

}  // namespace ov::npuw
