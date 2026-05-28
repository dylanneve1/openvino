// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>

#include "openvino/pass/pass.hpp"

namespace ov::npuw {

// Replaces the dynamic-shape Parameters introduced by
// ov::pass::SDPAToPagedAttention with statically-bounded equivalents, and
// substitutes the scalar `max_context_len` Parameter with a baked Constant.
// Also runs ov::pass::ConvertPagedAttnInputs to materialize key_cache.N /
// value_cache.N shapes from the PagedAttentionExtension's rt_info, then pins
// the leading num_blocks dimension to the configured static bound.
//
// Required because NPU compiles only static-shape models. Each call covers one
// compiled variant (prefill, generate): the variant's max_context_len is baked
// into the model as a Constant.
class BakePagedAttentionStaticShapes : public ov::pass::ModelPass {
public:
    BakePagedAttentionStaticShapes(uint32_t num_blocks,
                                   uint32_t block_size,
                                   uint32_t max_seqs,
                                   uint32_t max_context_len,
                                   uint32_t input_size);

    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

private:
    uint32_t m_num_blocks;
    uint32_t m_block_size;
    uint32_t m_max_seqs;
    uint32_t m_max_context_len;
    uint32_t m_input_size;
};

// Cuts each PagedAttentionExtension out of the model and exposes its Q/K/V
// inputs and attention output across an explicit Result/Parameter boundary,
// so the NPU-resident dispatch can read Q/K/V tensors from the (compiled)
// subgraph and inject the tile-loop result back as the next stage's input.
//
// For every PA op `pa.N` (N is the layer index parsed from key_cache.N), the
// pass rewrites the model in place:
//
//   - Adds three Result outputs `pa.N.q`, `pa.N.k`, `pa.N.v` whose sources
//     are the PA op's QUERY / KEY / VALUE inputs (input indices 0/1/2).
//   - Adds one Parameter `pa.N.out` with the same shape and element type as
//     the PA op's first output; redirects every consumer of `pa.N` output 0
//     to this new Parameter, then drops the PA op from the model.
//   - Deletes the now-orphan key_cache.N / value_cache.N / past_lens etc.
//     Parameters introduced by SDPAToPagedAttention if no other op consumes
//     them — those become the host-orchestrator's responsibility instead.
//
// The host orchestrator runs the rewritten model in two phases per generation
// step: phase 1 with a zero-filled `pa.N.out` placeholder captures Q/K/V from
// the new Result tensors; the tile-loop computes attention; phase 2 re-runs
// the model with the real attention output bound to `pa.N.out`. Pre-PA
// computation is independent of `pa.N.out`, so phase 1 produces correct
// Q/K/V outputs even though phase 1's downstream values are discarded.
class SplitAtPagedAttention : public ov::pass::ModelPass {
public:
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

}  // namespace ov::npuw
