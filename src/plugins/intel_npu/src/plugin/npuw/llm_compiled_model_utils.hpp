// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <memory>
#include <optional>

#include "openvino/openvino.hpp"

namespace ov ::npuw ::util {

/*
 * special mark on nodes to be remain in high-precision for optimal processing
 */
class HighPrecisionAttr : public RuntimeAttribute {
public:
    OPENVINO_RTTI("HighPrecisionAttr", "0", RuntimeAttribute);
    ov::element::Type compute_precision_type;

    bool visit_attributes(AttributeVisitor& visitor) override {
        visitor.on_attribute("compute_precision", compute_precision_type);
        return true;
    }
};

constexpr const char* kVisualPosMasksParamName = "visual_pos_masks";
constexpr const char* kDeepstackVisualEmbedsParamName = "deepstack_visual_embeds";

bool has_input(const std::shared_ptr<ov::Model>& model, const std::string& name);

// Returns true for a non-autoregressive (bidirectional encoder, e.g. BERT) text-embedding
// model: one that has ScaledDotProductAttention but NO autoregressive KV-cache concat pattern
// (Concat->Broadcast->Reshape on the SDPA key input) that the Qwen3-Embedding-style path needs.
// Used to route encoder embedders to the dedicated, KV/RoPE-free embedding path.
bool is_encoder_embedding_model(const std::shared_ptr<ov::Model>& model);

// Returns the learned absolute position-embedding table size (max_position_embeddings) of an
// encoder embedding model, found from the position_embeddings Gather's weight constant
// ([max_position_embeddings, hidden]). Returns nullopt if it can't be determined (e.g. the model
// uses a different positional scheme). The static sequence length must not exceed this, or the
// position embedding (clamped to the table) won't broadcast against the token embedding.
std::optional<uint32_t> get_max_position_embeddings(const std::shared_ptr<ov::Model>& model);

// Sanity-checks a non-autoregressive bidirectional encoder (e.g. BERT) text-embedding model for
// the NPUW embedding path. Unlike PrepareTextEmbeddingModel (autoregressive / Qwen3-Embedding
// style), the encoder path does NOT rebuild the graph — no KV-cache parameters, RoPE position
// ids, or causal mask are injected: an encoder is self-contained (builds its own bidirectional
// mask from `attention_mask`, uses learned absolute positions, has no KV cache) and processes
// the whole sequence in one forward. So all that's left is validation: assert there are no
// autoregressive KV-cache inputs and re-validate shapes.
void validate_encoder_embedding_model(const std::shared_ptr<ov::Model>& model);

}  // namespace ov::npuw::util
