// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>

#include "openvino/pass/pass.hpp"

namespace ov::npuw {

// Lowers ov::op::PagedAttentionExtension into an NPU-runnable subgraph
// built around an HFA-style online-softmax tile loop with block-table
// indirected K/V gather.
//
// SKELETON — see PAGED_ATTN_LOWERING_DESIGN.md for the full design.
// Today this pass detects PA ops and logs them; the transformation
// itself is not yet implemented.
//
// Why this pass exists:
//   Today, NPUW auto-routes subgraphs containing PagedAttentionExtension
//   to the CPU plugin (compiled_model.cpp:682-698) because the NPU compiler
//   has no PA kernel. This pass replaces the opaque PA op with primitive
//   ops the NPU already supports (MatMul/Add/Exp/ReduceMax/ReduceSum/
//   Gather/Loop), structured around HFA's proven online-softmax tile model
//   (host_flash_attention.cpp:264-297).
//
// What the pass produces, sketched:
//
//   PagedAttentionExtension(Q, K_cache, V_cache, block_indices, mask, ...)
//
//   →  Loop over i in [0, num_active_blocks):
//          phys_id  = block_indices[i]
//          K_block  = Gather(K_cache, phys_id, axis=0)
//          V_block  = Gather(V_cache, phys_id, axis=0)
//          (acc, maxx, d) = HFATile(past_acc, past_max, past_d,
//                                   K_block, V_block, Q, mask_slice)
//      After loop: output = acc / d  (plus any transpose/reshape the
//                                      original PA op contract requires)
//
// Inputs to the constructor are the static-shape bounds the NPU compiler
// requires:
//   - max_num_blocks   : pool size (e.g. 64) — sets the static shape of
//                        the key_cache / value_cache parameters along axis 0.
//   - block_size       : tokens per block (e.g. 1024).
//   - max_active_blocks: upper bound on per-request block count; bounds the
//                        Loop trip count and the static shape of the padded
//                        block_indices input. Must be <= max_num_blocks.
//
// NOT IMPLEMENTED:
//   - The actual graph rewrite (run_on_model is currently a no-op aside from
//     logging).
//   - Position embedding / RoPE handling within the loop body.
//   - Tail-block masking when the last block is partially filled.
//   - Multi-head / GQA handling: HFA's tile model already supports a
//     'use_grouped' path; the lowering should preserve that.
//   - Final-tile model: HFA uses two tile models (regular + final, where
//     final does the acc/d division). The lowering should reuse the same
//     pattern instead of dividing inside the loop body.
//
class PagedAttnToHFATiles : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("PagedAttnToHFATiles");

    PagedAttnToHFATiles(uint32_t max_num_blocks, uint32_t block_size, uint32_t max_active_blocks);

    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

private:
    uint32_t m_max_num_blocks;
    uint32_t m_block_size;
    uint32_t m_max_active_blocks;
};

}  // namespace ov::npuw
