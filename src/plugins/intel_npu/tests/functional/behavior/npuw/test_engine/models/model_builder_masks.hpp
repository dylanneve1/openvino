// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace test {
namespace npuw {

/// Sentinel value used in additive float attention masks for "do not attend" positions.
constexpr float kAttentionMaskPadding = -10000.0f;

/// Padding-only mask: [batch, seq] -> [batch, 1, 1, seq] float (0.0=attend, kAttentionMaskPadding=pad).
ov::Output<ov::Node> make_padding_mask(const ov::Output<ov::Node>& attention_mask,
                                       ov::element::Type prec);

/// Standard causal mask combined with padding. Returns 4D float mask
/// [batch, 1, seq, total_seq] with 0.0=attend, kAttentionMaskPadding=masked.
ov::Output<ov::Node> make_causal_mask(const ov::Output<ov::Node>& seq_source,
                                      const ov::Output<ov::Node>& attention_mask,
                                      ov::element::Type prec);

}  // namespace npuw
}  // namespace test
}  // namespace ov
