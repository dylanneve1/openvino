// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_attention_static.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>

#include "../logging.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/paged_attention/convert_pagedattn_inputs.hpp"

namespace {
// Match a Parameter by friendly name OR any output-tensor name. SDPAToPagedAttention
// labels Parameters via add_names on the tensor (e.g. "key_cache.0"), not via
// set_friendly_name, so a friendly-name-only check misses them.
bool param_has_name(const std::shared_ptr<ov::op::v0::Parameter>& p, const std::string& target) {
    if (p->get_friendly_name() == target) {
        return true;
    }
    const auto& names = p->get_output_tensor(0).get_names();
    return names.count(target) > 0;
}

bool param_name_starts_with(const std::shared_ptr<ov::op::v0::Parameter>& p, const std::string& prefix) {
    if (p->get_friendly_name().rfind(prefix, 0) == 0) {
        return true;
    }
    for (const auto& n : p->get_output_tensor(0).get_names()) {
        if (n.rfind(prefix, 0) == 0) {
            return true;
        }
    }
    return false;
}
}  // namespace

namespace ov::npuw {

BakePagedAttentionStaticShapes::BakePagedAttentionStaticShapes(uint32_t num_blocks,
                                                               uint32_t block_size,
                                                               uint32_t max_seqs,
                                                               uint32_t max_context_len)
    : m_num_blocks(num_blocks),
      m_block_size(block_size),
      m_max_seqs(max_seqs),
      m_max_context_len(max_context_len) {}

bool BakePagedAttentionStaticShapes::run_on_model(const std::shared_ptr<ov::Model>& m) {
    LOG_DEBUG("BakePagedAttentionStaticShapes: num_blocks=" << m_num_blocks << " block_size=" << m_block_size
                                                            << " max_seqs=" << m_max_seqs << " max_context_len="
                                                            << m_max_context_len);

    // 1) Run ConvertPagedAttnInputs to set K/V cache shapes from PA rt_info.
    //    Use {0,1,2,3} dim order with [num_blocks, num_heads, block_size, head_size]
    //    — matches CPU plugin's default (intel_cpu/.../transformation_pipeline.cpp).
    ov::pass::ConvertPagedAttnInputs::KVCacheConfig cfg;
    cfg.keyCachePrecision = ov::element::f16;
    cfg.valueCachePrecision = ov::element::f16;
    cfg.inferencePrecision = ov::element::f16;
    cfg.keyCacheBlockSize = m_block_size;
    cfg.valueCacheBlockSize = m_block_size;
    cfg.keyCacheDimOrder = {0, 1, 2, 3};
    cfg.valueCacheDimOrder = {0, 1, 2, 3};

    auto update_shape_func =
        [](ov::element::Type /*precision*/, bool /*bychannel*/, size_t /*group_num*/, int64_t& /*head_size*/, int64_t& /*block_size*/) {
            // Default behaviour: no quantization-aware adjustments for now.
        };
    ov::pass::Manager mgr;
    mgr.register_pass<ov::pass::ConvertPagedAttnInputs>(cfg, update_shape_func);
    mgr.run_passes(m);

    // 2) Walk Parameters and pin shapes / replace with constants.
    const int64_t num_blocks = static_cast<int64_t>(m_num_blocks);
    const int64_t max_seqs = static_cast<int64_t>(m_max_seqs);
    const int64_t blocks_per_seq = (m_max_context_len + m_block_size - 1) / m_block_size;
    const int64_t total_blocks = max_seqs * blocks_per_seq;

    auto params = m->get_parameters();
    std::vector<std::shared_ptr<ov::op::v0::Parameter>> to_replace_with_const;

    for (auto& p : params) {
        if (param_name_starts_with(p, "key_cache.") || param_name_starts_with(p, "value_cache.")) {
            // ConvertPagedAttnInputs left the leading num_blocks dim dynamic;
            // pin it to our static bound.
            auto shape = p->get_partial_shape();
            if (shape.rank().is_static() && shape.rank().get_length() == 4) {
                shape[0] = ov::Dimension(num_blocks);
                p->set_partial_shape(shape);
                p->validate_and_infer_types();
            }
            continue;
        }
        if (param_has_name(p, "past_lens") || param_has_name(p, "subsequence_begins") ||
            param_has_name(p, "block_indices") || param_has_name(p, "block_indices_begins")) {
            // CSR-style helper arrays. Static-bound them.
            ov::PartialShape new_shape;
            if (param_has_name(p, "past_lens")) {
                new_shape = ov::PartialShape{max_seqs};
            } else if (param_has_name(p, "subsequence_begins") || param_has_name(p, "block_indices_begins")) {
                new_shape = ov::PartialShape{max_seqs + 1};
            } else {
                new_shape = ov::PartialShape{total_blocks};
            }
            p->set_partial_shape(new_shape);
            p->validate_and_infer_types();
            continue;
        }
        if (param_has_name(p, "max_context_len")) {
            to_replace_with_const.push_back(p);
            continue;
        }
    }

    // 3) Replace max_context_len Parameter with a Constant baked to the
    //    variant's value. Done last so the iteration above isn't invalidated.
    for (auto& p : to_replace_with_const) {
        auto c = ov::op::v0::Constant::create(p->get_element_type(),
                                              ov::Shape{},
                                              {static_cast<int32_t>(m_max_context_len)});
        c->set_friendly_name(p->get_friendly_name());
        for (const auto& n : p->get_output_tensor(0).get_names()) {
            c->get_output_tensor(0).add_names({n});
        }
        for (auto& consumer_input : p->output(0).get_target_inputs()) {
            consumer_input.replace_source_output(c->output(0));
        }
        m->remove_parameter(p);
    }

    m->validate_nodes_and_infer_types();
    return true;
}

}  // namespace ov::npuw
