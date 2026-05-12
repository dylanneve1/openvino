// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include "openvino/openvino.hpp"
#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace npuw {

namespace online {
class Snapshot;
}  // namespace online

namespace patterns {
namespace attn {

// Matches an ov::op::PagedAttentionExtension op and tags it (plus its
// immediately neighbouring helpers) with the "attn" isolation tag. Used by the
// online partitioner so the paged-attention subgraph can be routed to a
// dedicated device (the CPU plugin, via NPUW_ATTN_DEVICE) while the rest of
// the model compiles to NPU.
class PagedAttn : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::attn::PagedAttn");
    static constexpr const char* pattern_name() {
        return "PagedAttn";
    }
    static constexpr const char* isolation_tag() {
        return "attn";
    }
    static constexpr const char* group_name() {
        return "attn";
    }
    PagedAttn(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag);
};

}  // namespace attn
}  // namespace patterns
}  // namespace npuw
}  // namespace ov
