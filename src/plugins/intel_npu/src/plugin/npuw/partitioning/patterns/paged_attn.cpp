// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_attn.hpp"

#include <unordered_set>

#include "../../logging.hpp"
#include "../online/group.hpp"
#include "../online/snapshot.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov {
namespace npuw {
namespace patterns {
namespace attn {

namespace opp = ov::pass::pattern;

namespace {

// Returns true when `node` is one of the "thin" ops that sit between the
// q_proj / k_proj / v_proj MatMul output and a PagedAttentionExtension input
// (Reshape / Transpose / Convert / Multiply / Add / Concat / Unsqueeze and
// the GQA Broadcast variants). These are the ops that should join the PA
// group so the partitioner can recognise the SDPA-like attention block as
// one repeating unit and the q/k/v_proj MatMuls remain in the upstream
// compute group as boundary anchors.
bool is_attn_glue_op(const std::shared_ptr<ov::Node>& node) {
    return ov::as_type_ptr<ov::op::v1::Reshape>(node) || ov::as_type_ptr<ov::op::v1::Transpose>(node) ||
           ov::as_type_ptr<ov::op::v0::Convert>(node) || ov::as_type_ptr<ov::op::v1::Multiply>(node) ||
           ov::as_type_ptr<ov::op::v1::Add>(node) || ov::as_type_ptr<ov::op::v0::Concat>(node) ||
           ov::as_type_ptr<ov::op::v0::Unsqueeze>(node);
}

bool is_boundary_op(const std::shared_ptr<ov::Node>& node) {
    // q_proj / k_proj / v_proj outputs (MatMul) and Parameters mark the
    // upstream boundary of the attention pattern; their groups must stay
    // outside the "attn" tag so the partitioner can fold the rest of the
    // layer into the repeating compute block.
    return ov::as_type_ptr<ov::op::v0::MatMul>(node) || ov::as_type_ptr<ov::op::v0::Parameter>(node);
}

template <typename Visit>
void walk_upstream_glue(const std::shared_ptr<ov::Node>& start, Visit&& visit, std::unordered_set<ov::Node*>& seen) {
    if (!start || !seen.insert(start.get()).second) {
        return;
    }
    if (!is_attn_glue_op(start) || is_boundary_op(start)) {
        return;
    }
    visit(start);
    for (size_t i = 0; i < start->get_input_size(); ++i) {
        walk_upstream_glue(start->get_input_node_shared_ptr(i), std::forward<Visit>(visit), seen);
    }
}

template <typename Visit>
void walk_downstream_glue(const std::shared_ptr<ov::Node>& start, Visit&& visit, std::unordered_set<ov::Node*>& seen) {
    if (!start || !seen.insert(start.get()).second) {
        return;
    }
    if (!is_attn_glue_op(start) || is_boundary_op(start)) {
        return;
    }
    visit(start);
    for (const auto& output : start->outputs()) {
        for (auto& consumer : output.get_target_inputs()) {
            walk_downstream_glue(consumer.get_node()->shared_from_this(), std::forward<Visit>(visit), seen);
        }
    }
}

}  // namespace

PagedAttn::PagedAttn(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag) {
    auto pa = opp::wrap_type<ov::op::PagedAttentionExtension>();

    auto node_to_gptr = snapshot->getNodeToGroupMap();

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();
        auto match_iter = node_to_output.find(pa);
        if (match_iter == node_to_output.end()) {
            return false;
        }
        auto matched = match_iter->second.get_node_shared_ptr();

        auto isolate = [&](const std::shared_ptr<ov::Node>& n) {
            if (auto gi = node_to_gptr->find(n); gi != node_to_gptr->end()) {
                gi->second->isolate(isol_tag);
            }
        };

        // Tag the PA op itself.
        isolate(matched);

        // Tag the Reshape / Transpose / Convert chain feeding QUERY (input 0),
        // KEY (input 1) and VALUE (input 2). Walk stops at q_proj / k_proj /
        // v_proj (MatMul) so those projections remain in the upstream compute
        // group — the partitioner needs that boundary to merge cross-layer
        // attention bodies into a repeating block.
        std::unordered_set<ov::Node*> upstream_seen;
        for (std::size_t in_idx = 0; in_idx < 3 && in_idx < matched->get_input_size(); ++in_idx) {
            walk_upstream_glue(matched->get_input_node_shared_ptr(in_idx), isolate, upstream_seen);
        }

        // Tag the Reshape feeding o_proj on the PA op output side so the
        // attention-output transpose/reshape moves into the attn group too.
        std::unordered_set<ov::Node*> downstream_seen;
        for (const auto& output : matched->outputs()) {
            for (auto& consumer : output.get_target_inputs()) {
                walk_downstream_glue(consumer.get_node()->shared_from_this(), isolate, downstream_seen);
            }
        }

        LOG_DEBUG("PagedAttn matcher isolated " << matched->get_friendly_name() << " plus its glue chain with tag '"
                                                  << isol_tag << "'");
        return false;  // root unchanged
    };
    register_matcher(std::make_shared<opp::Matcher>(pa, "TagPagedAttn"), std::move(callback));
}

}  // namespace attn
}  // namespace patterns
}  // namespace npuw
}  // namespace ov
