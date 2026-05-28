// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_attention_static.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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
                                                               uint32_t max_context_len,
                                                               uint32_t input_size)
    : m_num_blocks(num_blocks),
      m_block_size(block_size),
      m_max_seqs(max_seqs),
      m_max_context_len(max_context_len),
      m_input_size(input_size) {}

bool BakePagedAttentionStaticShapes::run_on_model(const std::shared_ptr<ov::Model>& m) {
    LOG_DEBUG("BakePagedAttentionStaticShapes: num_blocks=" << m_num_blocks << " block_size=" << m_block_size
                                                            << " max_seqs=" << m_max_seqs << " max_context_len="
                                                            << m_max_context_len << " input_size=" << m_input_size);

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

    // 4) Pin user-facing input shapes to the variant's static input_size.
    //    SDPAToPagedAttention has already rewritten these inputs to a flat
    //    "B_token" layout:
    //      - input_ids:     rank-1 [B_token]      (was [batch, seq])
    //      - inputs_embeds: rank-2 [B_token, H]   (was [batch, seq, H])
    //      - position_ids:  rank-1 [B_token]      (was [batch, seq]),
    //                       or rank-2 [3, B_token] for VLM M-RoPE.
    //    The pass also drops attention_mask (PA reads sequence info from
    //    past_lens / subsequence_begins instead).
    //
    //    We must keep these ranks intact and only pin the dynamic B_token
    //    dimension to input_size — adding a leading batch dim here would
    //    propagate an extra rank through every downstream op (e.g. embed
    //    Gather producing rank-4 instead of rank-3 hidden_states).
    std::map<std::string, ov::PartialShape> reshape_map;
    const int64_t input_size = static_cast<int64_t>(m_input_size);
    for (const auto& in : m->inputs()) {
        const auto& name = in.get_any_name();
        const auto& ps = in.get_partial_shape();
        if (name.find("input_ids") != std::string::npos || name.find("token_type_ids") != std::string::npos) {
            reshape_map.emplace(name, ov::PartialShape{input_size});
        } else if (name.find("inputs_embeds") != std::string::npos) {
            if (ps.size() == 2u && ps[1].is_static()) {
                reshape_map.emplace(name, ov::PartialShape{input_size, ps[1]});
            }
        } else if (name.find("position_ids") != std::string::npos) {
            if (ps.size() == 1u) {
                reshape_map.emplace(name, ov::PartialShape{input_size});
            } else if (ps.size() == 2u && ps[0].is_static()) {
                reshape_map.emplace(name, ov::PartialShape{ps[0], input_size});
            }
        }
    }
    if (!reshape_map.empty()) {
        m->reshape(reshape_map);
    }

    // 5) After m->reshape() runs SmartReshape, orphaned subgraphs left behind
    //    by earlier passes — notably the dangling lm_head MatMul produced by
    //    CutLMHead — can become visible to the online partitioner via
    //    consumer-walks of live nodes (e.g. the shared embed-weights Convert).
    //    The partitioner then asserts because the orphan isn't in its
    //    node-to-group map.
    //
    //    A consumer is "stale" if it's not reachable from this Model's roots
    //    (results / sinks / parameters). Such a consumer typically belongs
    //    to a sibling Model (lm_head_model) that shares a subgraph with us
    //    via tied embedding weights. We migrate the stale consumer's
    //    dependency on `n` onto a fresh clone of `n`'s input cone, so:
    //      - this Model no longer sees the orphan as a consumer (the
    //        partitioner stops asserting),
    //      - the sibling Model keeps a functionally-identical input chain.
    //    Constants are cloned by reference to the same data; live Param /
    //    op nodes are cloned via Node::clone_with_new_inputs() with their
    //    same source outputs. Since the orphan's only reason to share with
    //    us is the tied embedding Convert, the only nodes we actually clone
    //    are decompression ops (Convert, Multiply, etc.) sitting on top of
    //    a Constant — cheap and self-contained.
    {
        const auto reachable_vec = m->get_ordered_ops();
        std::unordered_set<ov::Node*> reachable;
        reachable.reserve(reachable_vec.size());
        for (const auto& n : reachable_vec) {
            reachable.insert(n.get());
        }

        std::function<std::shared_ptr<ov::Node>(const std::shared_ptr<ov::Node>&)> deep_clone;
        std::unordered_map<ov::Node*, std::shared_ptr<ov::Node>> clone_cache;
        deep_clone = [&](const std::shared_ptr<ov::Node>& src) -> std::shared_ptr<ov::Node> {
            auto it = clone_cache.find(src.get());
            if (it != clone_cache.end()) {
                return it->second;
            }
            ov::OutputVector new_inputs;
            new_inputs.reserve(src->get_input_size());
            for (size_t i = 0; i < src->get_input_size(); ++i) {
                const auto src_in = src->get_input_node_shared_ptr(i);
                auto cloned_in = deep_clone(src_in);
                new_inputs.emplace_back(cloned_in, src->input(i).get_source_output().get_index());
            }
            auto cloned = src->clone_with_new_inputs(new_inputs);
            cloned->set_friendly_name(src->get_friendly_name());
            clone_cache.emplace(src.get(), cloned);
            return cloned;
        };

        for (const auto& live : reachable_vec) {
            for (size_t out_idx = 0; out_idx < live->get_output_size(); ++out_idx) {
                const auto consumers = live->output(out_idx).get_target_inputs();
                for (auto& consumer_input : consumers) {
                    auto consumer_node = consumer_input.get_node()->shared_from_this();
                    if (reachable.count(consumer_node.get()) != 0) {
                        continue;  // Live consumer, leave alone.
                    }
                    // Stale consumer references this live output. Replace
                    // its source with a cloned copy of `live` so the live
                    // node no longer sees the stale consumer.
                    auto cloned_live = deep_clone(live);
                    consumer_input.replace_source_output(cloned_live->output(out_idx));
                }
            }
        }
    }

    m->validate_nodes_and_infer_types();
    return true;
}

}  // namespace ov::npuw
