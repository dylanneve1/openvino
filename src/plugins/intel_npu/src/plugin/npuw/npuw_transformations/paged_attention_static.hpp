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
                                   uint32_t max_context_len);

    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

private:
    uint32_t m_num_blocks;
    uint32_t m_block_size;
    uint32_t m_max_seqs;
    uint32_t m_max_context_len;
};

}  // namespace ov::npuw
