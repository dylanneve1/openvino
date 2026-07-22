// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "llm_compiled_model_utils.hpp"

#include <algorithm>

#include "logging.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/gather_base.hpp"

bool ov::npuw::util::has_input(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    auto inputs = model->inputs();
    auto it = std::find_if(inputs.begin(), inputs.end(), [&](const auto& port) {
        return port.get_names().count(name) != 0;
    });
    return it != inputs.end();
}

bool ov::npuw::util::is_encoder_embedding_model(const std::shared_ptr<ov::Model>& model) {
    // Invariant: this discriminator must match ReConstructEmbeddingModel::check_kv_concat_nodes
    // (embedding/prepare_embedding_model.cpp) exactly — any model routed to the autoregressive
    // path will be reconstructed under that exact pattern, so a looser check here would send a
    // model into a reconstruction that then fails.
    auto has_kv_concat_pattern = [](const std::shared_ptr<ov::Node>& sdpa) -> bool {
        // Key input is at index 1: Concat -> Broadcast -> Reshape -> SDPA.
        auto reshape_node = sdpa->input(1).get_source_output().get_node();
        if (reshape_node == nullptr || strstr(reshape_node->get_type_name(), "Reshape") == nullptr) {
            return false;
        }
        auto broadcast_node = reshape_node->input(0).get_source_output().get_node();
        if (broadcast_node == nullptr || strstr(broadcast_node->get_type_name(), "Broadcast") == nullptr) {
            return false;
        }
        auto concat_node = broadcast_node->input(1).get_source_output().get_node();
        if (concat_node == nullptr || strstr(concat_node->get_type_name(), "Concat") == nullptr) {
            return false;
        }
        return true;
    };

    bool has_sdpa = false;
    for (const auto& op : model->get_ops()) {
        if (ov::is_type<ov::op::v13::ScaledDotProductAttention>(op)) {
            has_sdpa = true;
            if (has_kv_concat_pattern(op)) {
                // Autoregressive (Qwen3-Embedding-style) — handled by PrepareTextEmbeddingModel.
                return false;
            }
        }
    }
    return has_sdpa;
}

namespace {

// Walks up from `out` through weight-decompression nodes (Convert, and Multiply/Subtract
// scale/zero-point application, whose weight always sits on port 0) and returns the terminal
// node — a Constant for a learned table.
std::shared_ptr<ov::Node> unwrap_weight_decompression(ov::Output<ov::Node> out) {
    auto node = out.get_node_shared_ptr();
    while (node && (ov::is_type<ov::op::v0::Convert>(node) || ov::is_type<ov::op::v1::Multiply>(node) ||
                    ov::is_type<ov::op::v1::Subtract>(node))) {
        node = node->input_value(0).get_node_shared_ptr();
    }
    return node;
}

// Walks up from `out` through value-preserving nodes (Convert, Reshape, Squeeze, Unsqueeze) and
// returns the node that actually produces the values.
std::shared_ptr<ov::Node> unwrap_value_preserving(ov::Output<ov::Node> out) {
    auto node = out.get_node_shared_ptr();
    while (node && (ov::is_type<ov::op::v0::Convert>(node) || ov::is_type<ov::op::v1::Reshape>(node) ||
                    ov::is_type<ov::op::v0::Squeeze>(node) || ov::is_type<ov::op::v0::Unsqueeze>(node))) {
        node = node->input_value(0).get_node_shared_ptr();
    }
    return node;
}

bool is_position_ids_param(const std::shared_ptr<ov::Node>& param) {
    if (param->get_friendly_name().find("position") != std::string::npos) {
        return true;
    }
    for (const auto& name : param->output(0).get_names()) {
        if (name.find("position") != std::string::npos) {
            return true;
        }
    }
    return false;
}

}  // anonymous namespace

std::optional<uint32_t> ov::npuw::util::get_max_position_embeddings(const std::shared_ptr<ov::Model>& model) {
    // The learned absolute position embedding is Gather(table[max_position_embeddings, hidden],
    // position_ids, axis=0). Prefer the Gather whose friendly name carries "position_embeddings"
    // (the common HF export naming). Fall back to structure: among the embedding-table Gathers
    // (a static 2D table tracing to a Constant through weight-decompression nodes), the position
    // one has internally computed indices (Range / Slice / a folded arange Constant), while the
    // word and token-type tables are gathered directly from model inputs. Of the remaining
    // candidates take the largest table: a folded all-zero token_type_ids also yields internal
    // constant indices, but its type_vocab_size is orders of magnitude below any position count.
    std::optional<uint32_t> by_structure;
    for (const auto& op : model->get_ops()) {
        if (!ov::is_type<ov::op::util::GatherBase>(op)) {
            continue;
        }
        const auto& table_shape = op->input_value(0).get_partial_shape();
        if (table_shape.rank().is_dynamic() || table_shape.size() != 2 || table_shape[0].is_dynamic()) {
            continue;
        }
        const auto size = static_cast<uint32_t>(table_shape[0].get_length());
        if (op->get_friendly_name().find("position_embeddings") != std::string::npos) {
            return size;
        }
        if (!ov::is_type<ov::op::v0::Constant>(unwrap_weight_decompression(op->input_value(0)))) {
            continue;  // Not a learned table.
        }
        auto indices = unwrap_value_preserving(op->input_value(1));
        if (indices && ov::is_type<ov::op::v0::Parameter>(indices) && !is_position_ids_param(indices)) {
            continue;  // Gathered from input_ids / token_type_ids: the word or token-type table.
        }
        by_structure = std::max(by_structure.value_or(0u), size);
    }
    return by_structure;
}

void ov::npuw::util::validate_encoder_embedding_model(const std::shared_ptr<ov::Model>& model) {
    LOG_DEBUG("Validating encoder (bidirectional) text-embedding model");
    LOG_BLOCK();

    for (const auto& param : model->get_parameters()) {
        for (const auto& name : param->get_output_tensor(0).get_names()) {
            OPENVINO_ASSERT(name.find("past_key_values") == std::string::npos,
                            "validate_encoder_embedding_model: unexpected autoregressive KV-cache input '",
                            name,
                            "' in an encoder embedding model.");
        }
    }

    model->validate_nodes_and_infer_types();

    LOG_DEBUG("Done");
}
