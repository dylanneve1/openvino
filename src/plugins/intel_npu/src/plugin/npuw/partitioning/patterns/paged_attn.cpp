// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_attn.hpp"

#include "../../logging.hpp"
#include "../online/group.hpp"
#include "../online/snapshot.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov {
namespace npuw {
namespace patterns {
namespace attn {

namespace opp = ov::pass::pattern;

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
        if (auto gi = node_to_gptr->find(matched); gi != node_to_gptr->end()) {
            gi->second->isolate(isol_tag);
        }
        LOG_DEBUG("PagedAttn matcher isolated " << matched->get_friendly_name() << " with tag '" << isol_tag << "'");
        return false;  // root unchanged
    };
    register_matcher(std::make_shared<opp::Matcher>(pa, "TagPagedAttn"), std::move(callback));
}

}  // namespace attn
}  // namespace patterns
}  // namespace npuw
}  // namespace ov
