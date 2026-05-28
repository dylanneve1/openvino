// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_attn_runtime.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "logging.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"

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
const PagedAttentionPartitionerContext* g_pa_ctx = nullptr;
}  // namespace

const PagedAttentionPartitionerContext* current_pa_partitioner_context() {
    return g_pa_ctx;
}

PagedAttentionPartitionerScope::PagedAttentionPartitionerScope(
    std::vector<std::shared_ptr<KVCacheBlockManager>> key_managers,
    std::vector<std::shared_ptr<KVCacheBlockManager>> value_managers,
    ov::SoPtr<ov::ICompiledModel> compiled_tile_model,
    ov::SoPtr<ov::ICompiledModel> compiled_final_tile_model) {
    static thread_local PagedAttentionPartitionerContext storage;
    storage.key_managers = std::move(key_managers);
    storage.value_managers = std::move(value_managers);
    storage.compiled_tile_model = std::move(compiled_tile_model);
    storage.compiled_final_tile_model = std::move(compiled_final_tile_model);
    g_pa_ctx = &storage;
}

PagedAttentionPartitionerScope::~PagedAttentionPartitionerScope() {
    g_pa_ctx = nullptr;
}


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

// Walk back through "glue" ops (Reshape, Transpose, Convert, Multiply, Add,
// Concat, Unsqueeze) from a node until we find a Parameter on `model`.
// Returns nullopt if the chain hits a non-glue, non-Parameter node.
std::optional<std::size_t> find_upstream_param_index(const std::shared_ptr<ov::Model>& model,
                                                     const std::shared_ptr<ov::Node>& node) {
    if (!node) {
        return std::nullopt;
    }
    if (auto idx = param_index_of(model, node)) {
        return idx;
    }
    // Walk only single-source glue ops, otherwise stop.
    if (node->get_input_size() == 0) {
        return std::nullopt;
    }
    auto is_glue = [](const std::shared_ptr<ov::Node>& n) {
        return ov::is_type<ov::op::v1::Reshape>(n) || ov::is_type<ov::op::v1::Transpose>(n) ||
               ov::is_type<ov::op::v0::Convert>(n) || ov::is_type<ov::op::v1::Multiply>(n) ||
               ov::is_type<ov::op::v1::Add>(n) || ov::is_type<ov::op::v0::Concat>(n) ||
               ov::is_type<ov::op::v0::Unsqueeze>(n);
    };
    if (!is_glue(node)) {
        return std::nullopt;
    }
    return find_upstream_param_index(model, node->get_input_node_shared_ptr(0));
}

// Record the parameter index for each of the PA op's inputs. Direct
// Parameters are picked up immediately; Q/K/V come through a glue chain
// (Reshape / Transpose / Convert / ...) introduced by SDPAToPagedAttention,
// so we walk that chain to find the upstream Parameter the function-call
// machinery binds at runtime. Constants (e.g. SCALE, MAX_CONTEXT_LEN once
// baked) are skipped — those values are read at tile-model construction.
void record_param_indices(const std::shared_ptr<ov::Model>& model,
                          const std::shared_ptr<ov::op::PagedAttentionExtension>& pa,
                          std::map<PAInputId, std::size_t>& out) {
    const auto pa_input_count = static_cast<std::size_t>(pa->get_input_size());
    const auto enum_count = static_cast<std::size_t>(PAInputId::COUNT);
    const auto count = std::min(pa_input_count, enum_count);

    for (std::size_t i = 0; i < count; ++i) {
        const auto src = pa->get_input_node_shared_ptr(i);
        if (auto idx = find_upstream_param_index(model, src)) {
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
        // Multi-layer model: every PA op shares geometry (H, Hk, S, block_size)
        // because each is fed by the same baked key_cache.N / value_cache.N
        // shape, so the tile sub-model can be built from any of them and
        // reused per layer at dispatch time. The infer-time orchestrator
        // selects the per-layer KV pool via _layer_index, which is parsed
        // from the PA op's KEY_CACHE Parameter name below.
        LOG_DEBUG("function::PagedAttention::from: model '" << model->get_friendly_name() << "' contains "
                                                            << pa_ops.size()
                                                            << " PA ops; building tile from the first.");
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

    // Parse the layer index out of the KEY_CACHE Parameter's friendly name
    // / tensor names (SDPAToPagedAttention names them "key_cache.N"). The
    // LLM runtime uses _layer_index to pick the right
    // KVCacheBlockManager pair from m_pa_layer_managers.
    const auto key_cache_node = pa->get_input_node_shared_ptr(static_cast<std::size_t>(PAInputId::KEY_CACHE));
    auto extract_layer_index = [](const std::shared_ptr<ov::Node>& node) -> std::optional<std::size_t> {
        const std::string prefix = "key_cache.";
        auto try_parse = [&prefix](const std::string& name) -> std::optional<std::size_t> {
            if (name.rfind(prefix, 0) != 0) {
                return std::nullopt;
            }
            const auto tail = name.substr(prefix.size());
            if (tail.empty()) {
                return std::nullopt;
            }
            for (char c : tail) {
                if (c < '0' || c > '9') {
                    return std::nullopt;
                }
            }
            return static_cast<std::size_t>(std::stoul(tail));
        };
        if (auto idx = try_parse(node->get_friendly_name())) {
            return idx;
        }
        for (const auto& n : node->get_output_tensor(0).get_names()) {
            if (auto idx = try_parse(n)) {
                return idx;
            }
        }
        return std::nullopt;
    };
    if (auto layer_idx = extract_layer_index(key_cache_node)) {
        out._layer_index = *layer_idx;
    } else {
        LOG_WARN("function::PagedAttention::from: could not parse layer index from KEY_CACHE name '"
                 << key_cache_node->get_friendly_name()
                 << "'. _layer_index defaults to 0; multi-layer pool selection will be wrong.");
    }

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

    // Sanity gates before tile-model construction.
    if (out._num_q_heads == 0 || out._num_kv_heads == 0 || out._head_size == 0 || out._block_size <= 0 ||
        out._query_size == 0) {
        LOG_WARN("function::PagedAttention::from: incomplete config extracted — skipping tile-model construction.");
        return out;  // config_extracted() will reflect what we have; is_valid() stays false.
    }
    if (out._num_q_heads % out._num_kv_heads != 0) {
        LOG_WARN("function::PagedAttention::from: H (" << out._num_q_heads << ") not divisible by Hk ("
                                                       << out._num_kv_heads << "); skipping tile-model construction.");
        return out;
    }

    // Build the two tile sub-models using the PA-flavoured wrapper so the
    // external I/O matches PA's [B_token, H*S] / [1, Hk, block_size, S]
    // layouts directly. The kernel body is identical to HFA's online
    // softmax; only the entry-point shape adapters differ.
    const auto& key_cache_param = pa->get_input_node_shared_ptr(static_cast<std::size_t>(PAInputId::KEY_CACHE));
    const ov::element::Type input_dtype = key_cache_param->get_element_type();
    const ov::element::Type mask_dtype = ov::element::f32;
    const ov::element::Type output_dtype = pa->get_output_element_type(0);

    try {
        out._tile_model = create_pa_online_softmax_tile_model(out._num_q_heads,
                                                              out._num_kv_heads,
                                                              out._query_size,
                                                              out._head_size,
                                                              out._block_size,
                                                              input_dtype,
                                                              mask_dtype,
                                                              /*is_final_tile=*/false,
                                                              output_dtype);

        out._final_tile_model = create_pa_online_softmax_tile_model(out._num_q_heads,
                                                                    out._num_kv_heads,
                                                                    out._query_size,
                                                                    out._head_size,
                                                                    out._block_size,
                                                                    input_dtype,
                                                                    mask_dtype,
                                                                    /*is_final_tile=*/true,
                                                                    output_dtype);
    } catch (const std::exception& e) {
        LOG_WARN("function::PagedAttention::from: tile-model construction failed: " << e.what());
        out._tile_model.reset();
        out._final_tile_model.reset();
        return out;
    }

    // create_online_softmax_tile_model produces a model with Parameters in this order
    // (see host_flash_attention.cpp:685-686):
    //   0 past_acc, 1 past_max, 2 past_d, 3 k_tile, 4 v_tile, 5 q, 6 mask_tile
    out._tile_param_index_map[OnlineSoftmaxTileInputId::PAST_ACC] = 0;
    out._tile_param_index_map[OnlineSoftmaxTileInputId::PAST_MAX] = 1;
    out._tile_param_index_map[OnlineSoftmaxTileInputId::PAST_D] = 2;
    out._tile_param_index_map[OnlineSoftmaxTileInputId::K_TILE] = 3;
    out._tile_param_index_map[OnlineSoftmaxTileInputId::V_TILE] = 4;
    out._tile_param_index_map[OnlineSoftmaxTileInputId::Q] = 5;
    out._tile_param_index_map[OnlineSoftmaxTileInputId::MASK_TILE] = 6;

    // Regular tile Results: {acc, maxx, d} — see create_regular_tile_outputs
    // (host_flash_attention.cpp:491-517).
    out._tile_output_index_map[OnlineSoftmaxTileOutputId::ACC] = 0;
    out._tile_output_index_map[OnlineSoftmaxTileOutputId::MAXX] = 1;
    out._tile_output_index_map[OnlineSoftmaxTileOutputId::D] = 2;

    LOG_INFO("function::PagedAttention::from: built tile sub-models. tile_model="
             << out._tile_model->get_friendly_name() << " final_tile_model="
             << out._final_tile_model->get_friendly_name() << " H=" << out._num_q_heads
             << " Hk=" << out._num_kv_heads << " q_size=" << out._query_size << " S=" << out._head_size
             << " input_dtype=" << input_dtype << " output_dtype=" << output_dtype);

    return out;
}

}  // namespace function
}  // namespace npuw
}  // namespace ov
