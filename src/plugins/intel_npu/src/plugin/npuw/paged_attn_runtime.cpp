// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_attn_runtime.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "logging.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/paged_attention.hpp"

namespace ov {
namespace npuw {

const char* pa_input_id_to_string(PAInputId id) {
    switch (id) {
    case PAInputId::QUERY:                                            return "QUERY";
    case PAInputId::KEY:                                              return "KEY";
    case PAInputId::VALUE:                                            return "VALUE";
    case PAInputId::KEY_CACHE:                                        return "KEY_CACHE";
    case PAInputId::VALUE_CACHE:                                      return "VALUE_CACHE";
    case PAInputId::PAST_LENS:                                        return "PAST_LENS";
    case PAInputId::SUBSEQUENCE_BEGINS:                               return "SUBSEQUENCE_BEGINS";
    case PAInputId::BLOCK_INDICES:                                    return "BLOCK_INDICES";
    case PAInputId::BLOCK_INDICES_BEGINS:                             return "BLOCK_INDICES_BEGINS";
    case PAInputId::SCALE:                                            return "SCALE";
    case PAInputId::SLIDING_WINDOW:                                   return "SLIDING_WINDOW";
    case PAInputId::ALIBI_SLOPES:                                     return "ALIBI_SLOPES";
    case PAInputId::MAX_CONTEXT_LEN:                                  return "MAX_CONTEXT_LEN";
    case PAInputId::SCORE_AGGREGATION_WINDOW:                         return "SCORE_AGGREGATION_WINDOW";
    case PAInputId::ROTATED_BLOCK_INDICES:                            return "ROTATED_BLOCK_INDICES";
    case PAInputId::ROTATION_DELTAS:                                  return "ROTATION_DELTAS";
    case PAInputId::ROTATION_TRIG_LUT:                                return "ROTATION_TRIG_LUT";
    case PAInputId::XATTENTION_THRESHOLD:                             return "XATTENTION_THRESHOLD";
    case PAInputId::XATTENTION_BLOCK_SIZE:                            return "XATTENTION_BLOCK_SIZE";
    case PAInputId::XATTENTION_STRIDE:                                return "XATTENTION_STRIDE";
    case PAInputId::SINKS:                                            return "SINKS";
    case PAInputId::ADAPTIVE_RKV_START_SIZE:                          return "ADAPTIVE_RKV_START_SIZE";
    case PAInputId::ADAPTIVE_RKV_EVICTABLE_SIZES:                     return "ADAPTIVE_RKV_EVICTABLE_SIZES";
    case PAInputId::ADAPTIVE_RKV_DIVERSITY_BLOCK_SET_INDICES:         return "ADAPTIVE_RKV_DIVERSITY_BLOCK_SET_INDICES";
    case PAInputId::ADAPTIVE_RKV_DIVERSITY_BLOCK_SET_INDICES_BEGINS:  return "ADAPTIVE_RKV_DIVERSITY_BLOCK_SET_INDICES_BEGINS";
    case PAInputId::TOKEN_TYPE_IDS:                                   return "TOKEN_TYPE_IDS";
    case PAInputId::QQ_BIAS:                                          return "QQ_BIAS";
    case PAInputId::QQ_BIAS_BEGINS:                                   return "QQ_BIAS_BEGINS";
    default:                                                          return "UNKNOWN";
    }
}

namespace function {

namespace {

// Find the index of `node` in `model->get_parameters()`. Returns nullopt if
// `node` is not a Parameter or not on the model.
std::optional<std::size_t> param_index_of(const std::shared_ptr<ov::Model>& model,
                                          const std::shared_ptr<ov::Node>& node) {
    if (!node) {
        return std::nullopt;
    }
    const auto& params = model->get_parameters();
    for (std::size_t i = 0; i < params.size(); ++i) {
        if (params[i] == node) {
            return i;
        }
    }
    return std::nullopt;
}

// Record the parameter index for each of the PA op's inputs that is itself
// a Parameter on the model. Inputs that come from Constants (e.g. SCALE,
// MAX_CONTEXT_LEN once baked) are skipped — those values are read directly
// from the constant during tile-model construction (Step 3b).
void record_param_indices(const std::shared_ptr<ov::Model>& model,
                          const std::shared_ptr<ov::op::PagedAttentionExtension>& pa,
                          std::map<PAInputId, std::size_t>& out) {
    const auto pa_input_count = static_cast<std::size_t>(pa->get_input_size());
    const auto enum_count = static_cast<std::size_t>(PAInputId::COUNT);
    const auto count = std::min(pa_input_count, enum_count);

    for (std::size_t i = 0; i < count; ++i) {
        const auto src = pa->get_input_node_shared_ptr(i);
        if (auto idx = param_index_of(model, src)) {
            out.emplace(static_cast<PAInputId>(i), *idx);
        }
    }
}

}  // namespace

std::optional<PagedAttention> PagedAttention::from(const std::shared_ptr<ov::Model>& model) {
    if (!model) {
        return std::nullopt;
    }

    // Find PA ops. We currently target the single-PA-op shape that
    // SDPAToPagedAttention with use_per_layer_block_indices_inputs=false
    // produces: one PagedAttentionExtension per layer is also valid but
    // requires a multi-instance orchestrator (out of scope for v1).
    std::vector<std::shared_ptr<ov::op::PagedAttentionExtension>> pa_ops;
    for (const auto& op : model->get_ordered_ops()) {
        if (auto pa = ov::as_type_ptr<ov::op::PagedAttentionExtension>(op)) {
            pa_ops.push_back(pa);
        }
    }

    if (pa_ops.empty()) {
        LOG_DEBUG("function::PagedAttention::from: no PagedAttentionExtension op on model '"
                  << model->get_friendly_name() << "'");
        return std::nullopt;
    }
    if (pa_ops.size() > 1) {
        LOG_DEBUG("function::PagedAttention::from: model '" << model->get_friendly_name() << "' contains "
                                                            << pa_ops.size()
                                                            << " PA ops; multi-op orchestration is not implemented yet.");
        // The downstream orchestrator will need to handle one per layer when
        // we extend to use_per_layer_block_indices_inputs=true. For now we
        // bail rather than silently pick one.
        return std::nullopt;
    }

    const auto& pa = pa_ops.front();
    PagedAttention out;

    // Extract per-block geometry from KEY_CACHE: [num_blocks, Hk, Bs, S].
    const auto& key_cache_input = pa->get_input_partial_shape(static_cast<std::size_t>(PAInputId::KEY_CACHE));
    if (!key_cache_input.is_static() || key_cache_input.rank().get_length() != 4) {
        LOG_WARN("function::PagedAttention::from: KEY_CACHE shape is not static-rank-4 ("
                 << key_cache_input
                 << "). Did BakePagedAttentionStaticShapes run on this model?");
        return std::nullopt;
    }
    const auto key_cache_shape = key_cache_input.to_shape();
    out._num_blocks = key_cache_shape[0];
    out._num_kv_heads = key_cache_shape[1];
    out._block_size = static_cast<int64_t>(key_cache_shape[2]);
    out._head_size = key_cache_shape[3];

    // Number of query heads from QUERY: [B_token, H * S].
    const auto& q_pshape = pa->get_input_partial_shape(static_cast<std::size_t>(PAInputId::QUERY));
    if (q_pshape.is_static() && q_pshape.rank().get_length() == 2 && out._head_size > 0) {
        const auto q_shape = q_pshape.to_shape();
        out._query_size = q_shape[0];
        if (q_shape[1] % out._head_size == 0) {
            out._num_q_heads = q_shape[1] / out._head_size;
        }
    }

    // Scale: usually a Constant set by ConvertPagedAttnInputs / SDPAToPagedAttention.
    // We try to peek at its value here; if not a Constant we leave _scale = 0 and
    // Step 3b's tile builder falls back to 1/sqrt(head_size).
    if (const auto scale_const = ov::as_type_ptr<ov::op::v0::Constant>(
            pa->get_input_node_shared_ptr(static_cast<std::size_t>(PAInputId::SCALE)))) {
        if (scale_const->get_element_type() == ov::element::f32 &&
            scale_const->get_byte_size() >= sizeof(float)) {
            out._scale = scale_const->cast_vector<float>().at(0);
        }
    }
    if (out._scale == 0.0f && out._head_size > 0) {
        out._scale = 1.0f / std::sqrt(static_cast<float>(out._head_size));
    }

    record_param_indices(model, pa, out._pa_param_index_map);

    LOG_INFO("function::PagedAttention::from: extracted PA config: "
             << "num_blocks=" << out._num_blocks << " block_size=" << out._block_size << " H=" << out._num_q_heads
             << " Hk=" << out._num_kv_heads << " S=" << out._head_size << " scale=" << out._scale
             << " param_indices_recorded=" << out._pa_param_index_map.size());

    // STEP 3b TODO — build _tile_model and _final_tile_model. Until then,
    // config_extracted() returns true but is_valid() returns false. The
    // runtime path (PagedLLMInferRequest) should bail when is_valid() is
    // false and let the existing CPU fallback handle execution.

    return out;
}

}  // namespace function
}  // namespace npuw
}  // namespace ov
