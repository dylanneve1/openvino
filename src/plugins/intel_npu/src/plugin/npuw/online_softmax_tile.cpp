// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Configuration: Enable loop-based Q@K computation to avoid materialized K/V broadcast
// Set to 1 to enable grouped computation, 0 to use traditional broadcast.
// Disabled by default because NPU compiler optimizations are suboptimal.
#define ENABLE_ONLINE_SOFTMAX_LOOP_BASED_COMPUTATION 0

#include "online_softmax_tile.hpp"

#include <string>
#include <vector>

#include "intel_npu/ops/flash_attention_tile.hpp"
#include "logging.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/openvino.hpp"
#include "util.hpp"

namespace ov {
namespace npuw {

namespace {

// Helper struct: input parameter nodes for one tile sub-model.
struct TileParams {
    std::shared_ptr<ov::op::v0::Parameter> past_acc;
    std::shared_ptr<ov::op::v0::Parameter> past_max;
    std::shared_ptr<ov::op::v0::Parameter> past_d;
    std::shared_ptr<ov::op::v0::Parameter> k_tile;
    std::shared_ptr<ov::op::v0::Parameter> v_tile;
    std::shared_ptr<ov::op::v0::Parameter> q;
    std::shared_ptr<ov::op::v0::Parameter> mask_tile;
};

// Helper struct: f32-converted nodes from input parameters for computation.
// All compute is performed in f32 for numerical stability.
struct TileF32Nodes {
    std::shared_ptr<ov::Node> past_acc_f32;
    std::shared_ptr<ov::Node> past_max_f32;
    std::shared_ptr<ov::Node> past_d_f32;
    std::shared_ptr<ov::Node> k_tile_f32;
    std::shared_ptr<ov::Node> v_tile_f32;
    std::shared_ptr<ov::Node> q_f32;
    std::shared_ptr<ov::Node> mask_tile_f32;
};

// Helper struct: online-softmax tile computation results (all in f32 precision).
// Contains: acc (accumulator), maxx (maximum values), d (normalization denominator).
struct TileResults {
    ov::Output<ov::Node> acc;
    ov::Output<ov::Node> maxx;
    ov::Output<ov::Node> d;
};

// ============================================================================
// Helper: Create input parameters for the tile sub-model.
// ============================================================================
TileParams create_tile_params(const ov::Shape& q_shape,
                              const ov::element::Type& input_dtype,
                              const ov::element::Type& mask_dtype,
                              int64_t tile_size,
                              size_t kv_num_heads) {
    auto batch = q_shape[0];
    auto num_heads = q_shape[1];
    auto seq_len = q_shape[2];
    auto head_dim = q_shape[3];

    TileParams params;

    auto set_param_name = [](std::shared_ptr<ov::op::v0::Parameter>& param, OnlineSoftmaxTileInputId id) {
        const char* name = online_softmax_tile_input_id_to_string(id);
        param->set_friendly_name(name);
        param->output(0).get_tensor().set_names({name});
    };

    // past_acc: [batch, num_heads, seq_len, head_dim]
    params.past_acc =
        std::make_shared<ov::op::v0::Parameter>(input_dtype, ov::Shape{batch, num_heads, seq_len, head_dim});
    set_param_name(params.past_acc, OnlineSoftmaxTileInputId::PAST_ACC);

    // past_max: [batch, num_heads, seq_len, 1]
    params.past_max = std::make_shared<ov::op::v0::Parameter>(input_dtype, ov::Shape{batch, num_heads, seq_len, 1});
    set_param_name(params.past_max, OnlineSoftmaxTileInputId::PAST_MAX);

    // past_d: [batch, num_heads, seq_len, 1]
    params.past_d = std::make_shared<ov::op::v0::Parameter>(input_dtype, ov::Shape{batch, num_heads, seq_len, 1});
    set_param_name(params.past_d, OnlineSoftmaxTileInputId::PAST_D);

    // k_tile: [batch, kv_num_heads, tile_size, head_dim]
    params.k_tile =
        std::make_shared<ov::op::v0::Parameter>(input_dtype,
                                                ov::Shape{batch, kv_num_heads, static_cast<size_t>(tile_size), head_dim});
    set_param_name(params.k_tile, OnlineSoftmaxTileInputId::K_TILE);

    // v_tile: [batch, kv_num_heads, head_dim, tile_size]
    params.v_tile =
        std::make_shared<ov::op::v0::Parameter>(input_dtype,
                                                ov::Shape{batch, kv_num_heads, head_dim, static_cast<size_t>(tile_size)});
    set_param_name(params.v_tile, OnlineSoftmaxTileInputId::V_TILE);

    // q: [batch, num_heads, seq_len, head_dim]
    params.q = std::make_shared<ov::op::v0::Parameter>(input_dtype, ov::Shape{batch, num_heads, seq_len, head_dim});
    set_param_name(params.q, OnlineSoftmaxTileInputId::Q);

    // mask_tile: [batch, 1, seq_len, tile_size] - use mask's original dtype
    params.mask_tile =
        std::make_shared<ov::op::v0::Parameter>(mask_dtype,
                                                ov::Shape{batch, 1, seq_len, static_cast<size_t>(tile_size)});
    set_param_name(params.mask_tile, OnlineSoftmaxTileInputId::MASK_TILE);

    return params;
}

// ============================================================================
// Helper: Convert input parameters to f32.
// ============================================================================
TileF32Nodes convert_inputs_to_f32(const TileParams& params,
                                   const ov::element::Type& mask_dtype,
                                   const ov::element::Type& compute_dtype) {
    TileF32Nodes f32;

    f32.past_acc_f32 = std::make_shared<ov::op::v0::Convert>(params.past_acc, compute_dtype);
    f32.past_acc_f32->set_friendly_name("past_acc_f32");

    f32.past_max_f32 = std::make_shared<ov::op::v0::Convert>(params.past_max, compute_dtype);
    f32.past_max_f32->set_friendly_name("past_max_f32");

    f32.past_d_f32 = std::make_shared<ov::op::v0::Convert>(params.past_d, compute_dtype);
    f32.past_d_f32->set_friendly_name("past_d_f32");

    f32.k_tile_f32 = std::make_shared<ov::op::v0::Convert>(params.k_tile, compute_dtype);
    f32.k_tile_f32->set_friendly_name("k_tile_f32");

    f32.v_tile_f32 = std::make_shared<ov::op::v0::Convert>(params.v_tile, compute_dtype);
    f32.v_tile_f32->set_friendly_name("v_tile_f32");

    f32.q_f32 = std::make_shared<ov::op::v0::Convert>(params.q, compute_dtype);
    f32.q_f32->set_friendly_name("q_f32");

    // Convert mask to f32 if needed
    if (mask_dtype == compute_dtype) {
        f32.mask_tile_f32 = params.mask_tile;
    } else {
        f32.mask_tile_f32 = std::make_shared<ov::op::v0::Convert>(params.mask_tile, compute_dtype);
        f32.mask_tile_f32->set_friendly_name("mask_tile_f32");
    }

    return f32;
}

// ============================================================================
// Helper: Execute tile via the NPU's fused FlashAttentionTile op.
// ============================================================================
TileResults execute_fused_tile(const TileF32Nodes& f32,
                               const std::shared_ptr<ov::Node>& q_input,
                               const std::shared_ptr<ov::Node>& k_input,
                               const std::shared_ptr<ov::Node>& v_input,
                               bool is_last_tile = false,
                               bool is_first_tile = false) {
    ov::intel_npu::op::FlashAttentionTile::Config config;
    config.is_head = is_first_tile;
    config.is_tail = is_last_tile;

    auto v_shape = v_input->get_output_partial_shape(0);
    auto rank = v_shape.rank().get_length();
    if (rank != 4) {
        OPENVINO_THROW("v_input rank must be 4 for flash attention");
    }
    std::vector<int64_t> transpose_v_order({0, 1, 3, 2});
    auto transpose_order = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                  ov::Shape{static_cast<size_t>(rank)},
                                                                  transpose_v_order);

    auto v_transpose = std::make_shared<ov::op::v1::Transpose>(v_input, transpose_order);
    v_transpose->set_friendly_name("v_input_transposed");

    auto squeeze = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});

    auto past_max_squeezed = std::make_shared<ov::op::v0::Squeeze>(f32.past_max_f32, squeeze);
    past_max_squeezed->set_friendly_name("past_max_squeezed");

    auto past_sum_squeezed = std::make_shared<ov::op::v0::Squeeze>(f32.past_d_f32, squeeze);
    past_sum_squeezed->set_friendly_name("past_sum_squeezed");

    auto flash_attn_tile = std::make_shared<ov::intel_npu::op::FlashAttentionTile>(q_input,
                                                                                   k_input,
                                                                                   v_transpose,
                                                                                   f32.past_acc_f32,
                                                                                   past_max_squeezed,
                                                                                   past_sum_squeezed,
                                                                                   f32.mask_tile_f32,
                                                                                   config);
    flash_attn_tile->set_friendly_name("npu_op_flash_attention_tile");
    TileResults results;
    results.acc = flash_attn_tile->output(0);
    results.maxx = flash_attn_tile->output(1);
    results.d = flash_attn_tile->output(2);
    return results;
}

// ============================================================================
// Helper: Execute one tile via the standard-ops online-softmax decomposition.
// Supports both traditional broadcast and loop-based grouped computation.
//
// Parameters:
//   use_grouped: If true, uses loop-based grouped computation (Q/P reshape).
//                If false, uses traditional broadcast K/V approach.
// ============================================================================
TileResults execute_online_softmax_tile(const TileF32Nodes& f32,
                                        const std::shared_ptr<ov::Node>& q_input,
                                        const std::shared_ptr<ov::Node>& k_input,
                                        const std::shared_ptr<ov::Node>& v_input,
                                        size_t batch,
                                        size_t num_heads,
                                        size_t kv_num_heads,
                                        size_t seq_len,
                                        size_t tile_size,
                                        size_t head_dim,
                                        bool use_grouped = false) {
    TileResults results;

    // ========================================================================
    // Step 1: Compute QK (method differs based on use_grouped flag)
    // ========================================================================
    std::shared_ptr<ov::Node> qk;

    if (use_grouped) {
        // Loop-based grouped computation: Q and K are grouped format
        // Q_input:  [batch, kv_num_heads, factor * seq_len, head_dim]
        // K_input:  [batch, kv_num_heads, tile_size, head_dim]
        // QK_grouped: [batch, kv_num_heads, factor * seq_len, tile_size]
        auto qk_grouped = std::make_shared<ov::op::v0::MatMul>(q_input, k_input, false, true);
        qk_grouped->set_friendly_name("qk_grouped");

        // Reshape QK back: [batch, kv_num_heads, factor * seq_len, tile_size]
        //              ->  [batch, num_heads, seq_len, tile_size]
        auto qk_reshape_pattern =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                   ov::Shape{4},
                                                   std::vector<int64_t>{static_cast<int64_t>(batch),
                                                                        static_cast<int64_t>(num_heads),
                                                                        static_cast<int64_t>(seq_len),
                                                                        static_cast<int64_t>(tile_size)});
        qk = std::make_shared<ov::op::v1::Reshape>(qk_grouped, qk_reshape_pattern, false);
        qk->set_friendly_name("qk");
    } else {
        // Traditional broadcast computation: use broadcast K directly
        // Q_input:  [batch, num_heads, seq_len, head_dim]
        // K_input:  [batch, num_heads, tile_size, head_dim] (already broadcast)
        // QK:       [batch, num_heads, seq_len, tile_size]
        qk = std::make_shared<ov::op::v0::MatMul>(q_input, k_input, false, true);
        qk->set_friendly_name("qk");
    }

    // ========================================================================
    // Step 2: Online-softmax core algorithm (same for both methods)
    // ========================================================================

    // qkm = qk + mask
    auto qkm = std::make_shared<ov::op::v1::Add>(qk, f32.mask_tile_f32);
    qkm->set_friendly_name("qkm");

    // maxx = max(past_max, reduce_max(qkm, axis=-1, keepdims=True))
    auto axes_const = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
    auto qkm_max = std::make_shared<ov::op::v1::ReduceMax>(qkm, axes_const, true);
    qkm_max->set_friendly_name("qkm_max");

    auto maxx_node = std::make_shared<ov::op::v1::Maximum>(qkm_max, f32.past_max_f32);
    maxx_node->set_friendly_name("maxx");
    results.maxx = maxx_node->output(0);

    // p = exp(qkm - maxx)
    auto qkm_sub_maxx = std::make_shared<ov::op::v1::Subtract>(qkm, results.maxx);
    auto p = std::make_shared<ov::op::v0::Exp>(qkm_sub_maxx);
    p->set_friendly_name("p");

    // l = reduce_sum(p, axis=-1, keepdims=True)
    auto l = std::make_shared<ov::op::v1::ReduceSum>(p, axes_const, true);
    l->set_friendly_name("l");

    // alpha = exp(past_max - maxx)
    auto past_max_sub_maxx = std::make_shared<ov::op::v1::Subtract>(f32.past_max_f32, results.maxx);
    auto alpha = std::make_shared<ov::op::v0::Exp>(past_max_sub_maxx);
    alpha->set_friendly_name("alpha");

    // d = past_d * alpha + l
    auto past_d_alpha = std::make_shared<ov::op::v1::Multiply>(f32.past_d_f32, alpha);
    auto d_node = std::make_shared<ov::op::v1::Add>(past_d_alpha, l);
    d_node->set_friendly_name("d");
    results.d = d_node->output(0);

    // ========================================================================
    // Step 3: Compute PV and final accumulator (method differs based on use_grouped flag)
    // ========================================================================

    auto past_acc_alpha = std::make_shared<ov::op::v1::Multiply>(f32.past_acc_f32, alpha);
    std::shared_ptr<ov::Node> pv;

    if (use_grouped) {
        // Loop-based grouped computation: reshape P, multiply with V, reshape back
        size_t factor = num_heads / kv_num_heads;

        // Reshape P for grouped V multiplication: [batch, num_heads, seq_len, tile_size]
        //                                      -> [batch, kv_num_heads, factor * seq_len, tile_size]
        auto p_reshape_pattern =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                   ov::Shape{4},
                                                   std::vector<int64_t>{static_cast<int64_t>(batch),
                                                                        static_cast<int64_t>(kv_num_heads),
                                                                        static_cast<int64_t>(factor * seq_len),
                                                                        static_cast<int64_t>(tile_size)});
        auto p_grouped = std::make_shared<ov::op::v1::Reshape>(p, p_reshape_pattern, false);
        p_grouped->set_friendly_name("p_grouped");

        // pv_grouped = matmul(p_grouped, v^T)
        // P_grouped: [batch, kv_num_heads, factor * seq_len, tile_size]
        // V_input:   [batch, kv_num_heads, head_dim, tile_size]
        // PV_grouped: [batch, kv_num_heads, factor * seq_len, head_dim]
        auto pv_grouped = std::make_shared<ov::op::v0::MatMul>(p_grouped, v_input, false, true);
        pv_grouped->set_friendly_name("pv_grouped");

        // Reshape PV back: [batch, kv_num_heads, factor * seq_len, head_dim]
        //              ->  [batch, num_heads, seq_len, head_dim]
        auto pv_reshape_pattern =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                   ov::Shape{4},
                                                   std::vector<int64_t>{static_cast<int64_t>(batch),
                                                                        static_cast<int64_t>(num_heads),
                                                                        static_cast<int64_t>(seq_len),
                                                                        static_cast<int64_t>(head_dim)});
        pv = std::make_shared<ov::op::v1::Reshape>(pv_grouped, pv_reshape_pattern, false);
        pv->set_friendly_name("pv");
    } else {
        // Traditional broadcast computation: use broadcast V directly
        // P:        [batch, num_heads, seq_len, tile_size]
        // V_input:  [batch, num_heads, head_dim, tile_size] (already broadcast)
        // PV:       [batch, num_heads, seq_len, head_dim]
        pv = std::make_shared<ov::op::v0::MatMul>(p, v_input, false, true);
        pv->set_friendly_name("pv");
    }

    // acc = past_acc * alpha + pv
    auto acc_node = std::make_shared<ov::op::v1::Add>(past_acc_alpha, pv);
    acc_node->set_friendly_name("acc");
    results.acc = acc_node->output(0);

    return results;
}

// ============================================================================
// Helper: Broadcast KV from kv_num_heads to num_heads.
// ============================================================================
std::pair<std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>> broadcast_kv_tiles(
    const std::shared_ptr<ov::Node>& k_tile_f32,
    const std::shared_ptr<ov::Node>& v_tile_f32,
    size_t batch,
    size_t num_heads,
    size_t kv_num_heads,
    size_t tile_size,
    size_t head_dim) {
    size_t head_expansion = num_heads / kv_num_heads;

    // Broadcast K: [batch, kv_num_heads, tile_size, head_dim]
    //          ->  [batch, num_heads, tile_size, head_dim]
    auto unsqueeze_axes_k =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{2});
    auto k_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(k_tile_f32, unsqueeze_axes_k);

    auto repeats_k =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                               ov::Shape{5},
                                               std::vector<int64_t>{1, 1, static_cast<int64_t>(head_expansion), 1, 1});
    auto k_tiled = std::make_shared<ov::op::v0::Tile>(k_unsqueezed, repeats_k);

    auto k_reshape_pattern =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                               ov::Shape{4},
                                               std::vector<int64_t>{static_cast<int64_t>(batch),
                                                                    static_cast<int64_t>(num_heads),
                                                                    static_cast<int64_t>(tile_size),
                                                                    static_cast<int64_t>(head_dim)});
    auto k_tile_broadcast = std::make_shared<ov::op::v1::Reshape>(k_tiled, k_reshape_pattern, false);
    k_tile_broadcast->set_friendly_name("k_tile_broadcast");

    // Broadcast V: [batch, kv_num_heads, head_dim, tile_size]
    //          ->  [batch, num_heads, head_dim, tile_size]
    auto unsqueeze_axes_v =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{2});
    auto v_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(v_tile_f32, unsqueeze_axes_v);

    auto repeats_v =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                               ov::Shape{5},
                                               std::vector<int64_t>{1, 1, static_cast<int64_t>(head_expansion), 1, 1});
    auto v_tiled = std::make_shared<ov::op::v0::Tile>(v_unsqueezed, repeats_v);

    auto v_reshape_pattern =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                               ov::Shape{4},
                                               std::vector<int64_t>{static_cast<int64_t>(batch),
                                                                    static_cast<int64_t>(num_heads),
                                                                    static_cast<int64_t>(head_dim),
                                                                    static_cast<int64_t>(tile_size)});
    auto v_tile_broadcast = std::make_shared<ov::op::v1::Reshape>(v_tiled, v_reshape_pattern, false);
    v_tile_broadcast->set_friendly_name("v_tile_broadcast");

    return {k_tile_broadcast, v_tile_broadcast};
}

#if ENABLE_ONLINE_SOFTMAX_LOOP_BASED_COMPUTATION
// ============================================================================
// Helper: Reshape Q for grouped computation (loop-based approach).
// Avoids materializing broadcasted K/V tensors by reshaping Q to match KV heads.
// Q: [batch, num_heads, seq_len, head_dim]
// -> [batch, kv_num_heads, factor * seq_len, head_dim]
// where factor = num_heads / kv_num_heads
// ============================================================================
std::shared_ptr<ov::Node> reshape_q_for_groups(const std::shared_ptr<ov::Node>& q_f32,
                                               size_t batch,
                                               size_t num_heads,
                                               size_t kv_num_heads,
                                               size_t seq_len,
                                               size_t head_dim) {
    size_t factor = num_heads / kv_num_heads;

    auto q_reshape_pattern =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                               ov::Shape{4},
                                               std::vector<int64_t>{static_cast<int64_t>(batch),
                                                                    static_cast<int64_t>(kv_num_heads),
                                                                    static_cast<int64_t>(factor * seq_len),
                                                                    static_cast<int64_t>(head_dim)});
    auto q_grouped = std::make_shared<ov::op::v1::Reshape>(q_f32, q_reshape_pattern, false);
    q_grouped->set_friendly_name("q_grouped");

    return q_grouped;
}
#endif  // ENABLE_ONLINE_SOFTMAX_LOOP_BASED_COMPUTATION

// ============================================================================
// Helper: Final tile outputs — acc/d division, transpose, reshape, cast.
// ============================================================================
ov::ResultVector create_final_tile_outputs(const TileResults& results,
                                           const ov::element::Type& output_dtype,
                                           size_t batch,
                                           size_t seq_len,
                                           size_t num_heads,
                                           size_t head_dim,
                                           bool fused_flash_attention = false) {
    std::shared_ptr<ov::Node> final_result;
    if (fused_flash_attention) {
        // If using FlashAttentionTile node, the output is already normalized,
        // so skip division.
        final_result = results.acc.get_node_shared_ptr();
        final_result->set_friendly_name("final_result");
    } else {
        // Division: result = acc / d
        final_result =
            std::make_shared<ov::op::v1::Divide>(results.acc.get_node_shared_ptr(), results.d.get_node_shared_ptr());
        final_result->set_friendly_name("final_result");
    }

    // Transpose (0,2,1,3): [batch, num_heads, seq_len, head_dim]
    //                  ->  [batch, seq_len, num_heads, head_dim]
    auto transpose_order =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 2, 1, 3});
    auto transposed_result = std::make_shared<ov::op::v1::Transpose>(final_result, transpose_order);
    transposed_result->set_friendly_name("transposed_result");

    // Reshape: [batch, seq_len, num_heads, head_dim] -> [batch, seq_len, num_heads*head_dim]
    auto reshape_pattern =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                               ov::Shape{3},
                                               std::vector<int64_t>{static_cast<int64_t>(batch),
                                                                    static_cast<int64_t>(seq_len),
                                                                    static_cast<int64_t>(num_heads * head_dim)});
    auto reshaped_result = std::make_shared<ov::op::v1::Reshape>(transposed_result, reshape_pattern, false);
    reshaped_result->set_friendly_name("reshaped_result");

    // Convert final output to original SDPA output dtype.
    auto final_output = std::make_shared<ov::op::v0::Convert>(reshaped_result, output_dtype);
    final_output->set_friendly_name("final_output");
    final_output->output(0).get_tensor().set_names({"output"});

    auto out_result = std::make_shared<ov::op::v0::Result>(final_output);
    out_result->set_friendly_name("out_result");

    return {out_result};
}

// ============================================================================
// Helper: Regular tile outputs (intermediate states: acc, max, d).
// ============================================================================
ov::ResultVector create_regular_tile_outputs(const TileResults& results, const ov::element::Type& input_dtype) {
    // Convert outputs back to input_dtype (e.g. f16).
    auto acc_output = std::make_shared<ov::op::v0::Convert>(results.acc, input_dtype);
    acc_output->set_friendly_name("acc_output");
    acc_output->output(0).get_tensor().set_names({"acc"});

    auto maxx_output = std::make_shared<ov::op::v0::Convert>(results.maxx, input_dtype);
    maxx_output->set_friendly_name("maxx_output");
    maxx_output->output(0).get_tensor().set_names({"maxx"});

    auto d_output = std::make_shared<ov::op::v0::Convert>(results.d, input_dtype);
    d_output->set_friendly_name("d_output");
    d_output->output(0).get_tensor().set_names({"d"});

    auto out_acc = std::make_shared<ov::op::v0::Result>(acc_output);
    out_acc->set_friendly_name("out_acc");

    auto out_maxx = std::make_shared<ov::op::v0::Result>(maxx_output);
    out_maxx->set_friendly_name("out_maxx");

    auto out_d = std::make_shared<ov::op::v0::Result>(d_output);
    out_d->set_friendly_name("out_d");

    return {out_acc, out_maxx, out_d};
}

// ============================================================================
// Helper: Regular tile outputs when the fused NPU op is used. The fused op
// returns acc/max/d separately; max/d need an extra unsqueeze to match the
// regular path's [batch, num_heads, seq_len, 1] layout.
// ============================================================================
ov::ResultVector create_regular_tile_outputs_fused(const TileResults& results, const ov::element::Type& input_dtype) {
    auto acc_output = std::make_shared<ov::op::v0::Convert>(results.acc, input_dtype);
    acc_output->set_friendly_name("acc_output");
    acc_output->output(0).get_tensor().set_names({"acc"});

    auto axes = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});

    auto maxx_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(results.maxx, axes);
    maxx_unsqueezed->set_friendly_name("maxx_unsqueezed");
    auto maxx_output = std::make_shared<ov::op::v0::Convert>(maxx_unsqueezed, input_dtype);
    maxx_output->set_friendly_name("maxx_output");
    maxx_output->output(0).get_tensor().set_names({"maxx"});

    auto d_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(results.d, axes);
    d_unsqueezed->set_friendly_name("d_unsqueezed");
    auto d_output = std::make_shared<ov::op::v0::Convert>(d_unsqueezed, input_dtype);
    d_output->set_friendly_name("d_output");
    d_output->output(0).get_tensor().set_names({"d"});

    auto out_acc = std::make_shared<ov::op::v0::Result>(acc_output);
    out_acc->set_friendly_name("out_acc");

    auto out_maxx = std::make_shared<ov::op::v0::Result>(maxx_output);
    out_maxx->set_friendly_name("out_maxx");

    auto out_d = std::make_shared<ov::op::v0::Result>(d_output);
    out_d->set_friendly_name("out_d");

    return {out_acc, out_maxx, out_d};
}

}  // namespace

// ============================================================================
// Public entry point — see online_softmax_tile.hpp for the contract.
// ============================================================================
std::shared_ptr<ov::Model> create_online_softmax_tile_model(const ov::Shape& q_shape,
                                                            const ov::element::Type& input_dtype,
                                                            const ov::element::Type& mask_dtype,
                                                            int64_t tile_size,
                                                            size_t kv_num_heads,
                                                            bool is_final_tile,
                                                            bool fused_flash_attention,
                                                            const ov::element::Type& output_dtype) {
    LOG_DEBUG("Creating online-softmax " << (is_final_tile ? "FINAL " : "") << "tile model with tile_size=" << tile_size
                                          << ", kv_num_heads=" << kv_num_heads << ", mask_dtype=" << mask_dtype
                                          << (is_final_tile ? ", output_dtype=" + output_dtype.get_type_name() : "")
                                          << ", fused_flash_attention=" << fused_flash_attention);

    // Extract dimensions
    NPUW_ASSERT(q_shape.size() == 4);
    auto batch = q_shape[0];
    auto num_heads = q_shape[1];
    auto seq_len = q_shape[2];
    auto head_dim = q_shape[3];

    NPUW_ASSERT(num_heads % kv_num_heads == 0 && "Q heads must be divisible by KV heads");

    auto compute_dtype = ov::element::f32;
    LOG_DEBUG("Using compute_dtype=f32 for all operations to match mask type");

    // Create input parameters.
    auto params = create_tile_params(q_shape, input_dtype, mask_dtype, tile_size, kv_num_heads);

    // Convert all inputs to f32.
    auto f32 = convert_inputs_to_f32(params, mask_dtype, compute_dtype);

    TileResults results;

#if ENABLE_ONLINE_SOFTMAX_LOOP_BASED_COMPUTATION
    // ========================================================================
    // Loop-based computation: Reshape Q to avoid K/V broadcast materialization.
    // ========================================================================
    LOG_DEBUG("Using loop-based grouped computation (ENABLED) - avoids K/V broadcast");

    auto q_grouped = reshape_q_for_groups(f32.q_f32, batch, num_heads, kv_num_heads, seq_len, head_dim);

    results = execute_online_softmax_tile(f32,
                                          q_grouped,      // Q: grouped format
                                          f32.k_tile_f32, // K: original 4D
                                          f32.v_tile_f32, // V: original 4D
                                          batch,
                                          num_heads,
                                          kv_num_heads,
                                          seq_len,
                                          tile_size,
                                          head_dim,
                                          /*use_grouped=*/true);
#else
    // ========================================================================
    // Traditional broadcast-based computation: materialize K/V broadcast.
    // ========================================================================
    LOG_DEBUG("Using traditional broadcast computation (DISABLED loop-based) - materializes K/V broadcast");

    if (fused_flash_attention) {
        // Execute fused flash attention node (MHA, GQA).
        results = execute_fused_tile(f32, f32.q_f32, f32.k_tile_f32, f32.v_tile_f32, is_final_tile);
    } else {
        // Broadcast K and V tiles from kv_num_heads to num_heads.
        auto [k_broadcast, v_broadcast] = broadcast_kv_tiles(f32.k_tile_f32,
                                                             f32.v_tile_f32,
                                                             batch,
                                                             num_heads,
                                                             kv_num_heads,
                                                             tile_size,
                                                             head_dim);

        results = execute_online_softmax_tile(f32,
                                              f32.q_f32,   // Q: original 4D
                                              k_broadcast, // K: broadcast to num_heads
                                              v_broadcast, // V: broadcast to num_heads
                                              batch,
                                              num_heads,
                                              kv_num_heads,
                                              seq_len,
                                              tile_size,
                                              head_dim,
                                              /*use_grouped=*/false);
    }
#endif  // ENABLE_ONLINE_SOFTMAX_LOOP_BASED_COMPUTATION

    // Create model outputs and name based on tile type.
    ov::ResultVector model_results;
    std::string model_name;

    if (is_final_tile) {
        model_results = create_final_tile_outputs(results,
                                                  output_dtype,
                                                  batch,
                                                  seq_len,
                                                  num_heads,
                                                  head_dim,
                                                  fused_flash_attention);
        model_name = "OnlineSoftmax_Final_Tile";
        LOG_DEBUG("Online-softmax FINAL tile model created: inputs=" << input_dtype << ", compute=" << compute_dtype
                                                                     << ", output=" << output_dtype);
    } else {
        if (fused_flash_attention) {
            LOG_DEBUG("Using fused flash attention implementation - outputs acc, max, d from separate nodes");
            model_results = create_regular_tile_outputs_fused(results, input_dtype);
        } else {
            LOG_DEBUG("Using host flash attention implementation - outputs acc, max, d from the same node");
            model_results = create_regular_tile_outputs(results, input_dtype);
        }
        model_name = "OnlineSoftmax_Tile";
        LOG_DEBUG("Online-softmax tile model created: inputs=" << input_dtype << ", compute=" << compute_dtype
                                                                << ", outputs=" << input_dtype);
    }

    // Create model parameters in the order declared by OnlineSoftmaxTileInputId.
    ov::ParameterVector model_params =
        {params.past_acc, params.past_max, params.past_d, params.k_tile, params.v_tile, params.q, params.mask_tile};

    return std::make_shared<ov::Model>(model_results, model_params, model_name);
}

// ============================================================================
// PA-flavoured tile model. See the contract in online_softmax_tile.hpp.
//
// External I/O shapes match the layouts produced by SDPAToPagedAttention +
// ConvertPagedAttnInputs + the KV-cache pool (KVCacheBlockManager). Internal
// nodes reshape/transpose those into the kernel's [batch=1, H, q, S] body
// before delegating to the standard helpers.
// ============================================================================
std::shared_ptr<ov::Model> create_pa_online_softmax_tile_model(size_t num_q_heads,
                                                               size_t num_kv_heads,
                                                               size_t q_size,
                                                               size_t head_dim,
                                                               int64_t block_size,
                                                               const ov::element::Type& input_dtype,
                                                               const ov::element::Type& mask_dtype,
                                                               bool is_final_tile,
                                                               const ov::element::Type& output_dtype,
                                                               float scale) {
    LOG_DEBUG("Creating PA online-softmax " << (is_final_tile ? "FINAL " : "") << "tile model with q_size=" << q_size
                                              << ", block_size=" << block_size << ", H=" << num_q_heads
                                              << ", Hk=" << num_kv_heads << ", S=" << head_dim << ", scale=" << scale);

    NPUW_ASSERT(num_q_heads % num_kv_heads == 0 && "Q heads must be divisible by KV heads");
    const size_t batch = 1u;
    const auto compute_dtype = ov::element::f32;
    // Cached tile reads one pool block (size = block_size); the final tile
    // attends over the present K/V (size = q_size = B_token rows).
    const auto tile_size =
        is_final_tile ? q_size : static_cast<size_t>(block_size);

    auto set_param_name = [](std::shared_ptr<ov::op::v0::Parameter>& param, OnlineSoftmaxTileInputId id) {
        const char* name = online_softmax_tile_input_id_to_string(id);
        param->set_friendly_name(name);
        param->output(0).get_tensor().set_names({name});
    };

    // ------------------------------------------------------------------
    // External Parameters — PA-native shapes.
    // ------------------------------------------------------------------
    auto past_acc = std::make_shared<ov::op::v0::Parameter>(input_dtype,
                                                            ov::Shape{batch, num_q_heads, q_size, head_dim});
    set_param_name(past_acc, OnlineSoftmaxTileInputId::PAST_ACC);

    auto past_max = std::make_shared<ov::op::v0::Parameter>(input_dtype, ov::Shape{batch, num_q_heads, q_size, 1});
    set_param_name(past_max, OnlineSoftmaxTileInputId::PAST_MAX);

    auto past_d = std::make_shared<ov::op::v0::Parameter>(input_dtype, ov::Shape{batch, num_q_heads, q_size, 1});
    set_param_name(past_d, OnlineSoftmaxTileInputId::PAST_D);

    // K_TILE: matches KVCacheBlockManager's per-block layout exactly.
    auto k_tile = std::make_shared<ov::op::v0::Parameter>(input_dtype,
                                                          ov::Shape{batch, num_kv_heads, tile_size, head_dim});
    set_param_name(k_tile, OnlineSoftmaxTileInputId::K_TILE);

    // V_TILE: stored as [1, Hk, block_size, S] in the pool; the body wants
    // [1, Hk, S, block_size]. Transpose dims 2↔3 internally.
    auto v_tile = std::make_shared<ov::op::v0::Parameter>(input_dtype,
                                                          ov::Shape{batch, num_kv_heads, tile_size, head_dim});
    set_param_name(v_tile, OnlineSoftmaxTileInputId::V_TILE);

    // Q: PA emits [q_size, H*S]; reshape to [q_size, H, S] then transpose
    // [1, 0, 2] then unsqueeze axis 0 to land at [1, H, q_size, S].
    auto q = std::make_shared<ov::op::v0::Parameter>(input_dtype,
                                                     ov::Shape{q_size, num_q_heads * head_dim});
    set_param_name(q, OnlineSoftmaxTileInputId::Q);

    // mask_tile: kept rank-2 [q_size, block_size] from the dispatcher; the
    // body wants [batch, 1, q_size, block_size] for broadcast over heads.
    auto mask_tile = std::make_shared<ov::op::v0::Parameter>(mask_dtype,
                                                             ov::Shape{q_size, tile_size});
    set_param_name(mask_tile, OnlineSoftmaxTileInputId::MASK_TILE);

    // ------------------------------------------------------------------
    // Internal layout adapters: external shapes -> body shapes.
    // ------------------------------------------------------------------
    // Q: [q_size, H*S] -> [q_size, H, S] -> [H, q_size, S] -> [1, H, q_size, S]
    auto q_reshape_3d_pattern = ov::op::v0::Constant::create(
        ov::element::i64,
        ov::Shape{3},
        std::vector<int64_t>{static_cast<int64_t>(q_size),
                             static_cast<int64_t>(num_q_heads),
                             static_cast<int64_t>(head_dim)});
    auto q_3d = std::make_shared<ov::op::v1::Reshape>(q, q_reshape_3d_pattern, false);
    auto q_transpose_order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{1, 0, 2});
    auto q_transposed = std::make_shared<ov::op::v1::Transpose>(q_3d, q_transpose_order);
    auto q_unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
    auto q_4d = std::make_shared<ov::op::v0::Unsqueeze>(q_transposed, q_unsqueeze_axes);
    q_4d->set_friendly_name("q_pa_reshaped");

    // V: [1, Hk, block_size, S] -> [1, Hk, S, block_size]
    auto v_transpose_order = ov::op::v0::Constant::create(ov::element::i64,
                                                          ov::Shape{4},
                                                          std::vector<int64_t>{0, 1, 3, 2});
    auto v_transposed = std::make_shared<ov::op::v1::Transpose>(v_tile, v_transpose_order);
    v_transposed->set_friendly_name("v_pa_transposed");

    // mask: [q_size, block_size] -> [1, 1, q_size, block_size]
    auto mask_reshape_pattern = ov::op::v0::Constant::create(
        ov::element::i64,
        ov::Shape{4},
        std::vector<int64_t>{1, 1, static_cast<int64_t>(q_size), static_cast<int64_t>(tile_size)});
    auto mask_4d = std::make_shared<ov::op::v1::Reshape>(mask_tile, mask_reshape_pattern, false);
    mask_4d->set_friendly_name("mask_pa_reshaped");

    // ------------------------------------------------------------------
    // Build the body inputs (Convert to f32) using the body-shape nodes.
    // ------------------------------------------------------------------
    TileF32Nodes f32;
    f32.past_acc_f32 = std::make_shared<ov::op::v0::Convert>(past_acc, compute_dtype);
    f32.past_max_f32 = std::make_shared<ov::op::v0::Convert>(past_max, compute_dtype);
    f32.past_d_f32 = std::make_shared<ov::op::v0::Convert>(past_d, compute_dtype);
    f32.k_tile_f32 = std::make_shared<ov::op::v0::Convert>(k_tile, compute_dtype);
    f32.v_tile_f32 = std::make_shared<ov::op::v0::Convert>(v_transposed, compute_dtype);
    f32.q_f32 = std::make_shared<ov::op::v0::Convert>(q_4d, compute_dtype);
    // Apply the SDPA scale (1/sqrt(head_dim) or the PA op's explicit scale).
    // The online-softmax body's QK MatMul is unscaled — HFA relies on the
    // upstream SDPA decomposition having pre-scaled Q, but PA's q_proj output
    // is raw, so we fold the scale into Q here.
    if (scale != 0.0f) {
        auto scale_c = ov::op::v0::Constant::create(compute_dtype, ov::Shape{}, {scale});
        f32.q_f32 = std::make_shared<ov::op::v1::Multiply>(f32.q_f32, scale_c);
        f32.q_f32->set_friendly_name("q_f32_scaled");
    }
    f32.mask_tile_f32 = (mask_dtype == compute_dtype)
                            ? std::static_pointer_cast<ov::Node>(mask_4d)
                            : std::static_pointer_cast<ov::Node>(
                                  std::make_shared<ov::op::v0::Convert>(mask_4d, compute_dtype));

    // Broadcast K/V from kv_heads to num_q_heads, then run the kernel body.
    auto [k_broadcast, v_broadcast] =
        broadcast_kv_tiles(f32.k_tile_f32, f32.v_tile_f32, batch, num_q_heads, num_kv_heads, tile_size, head_dim);

    auto results = execute_online_softmax_tile(f32,
                                               f32.q_f32,
                                               k_broadcast,
                                               v_broadcast,
                                               batch,
                                               num_q_heads,
                                               num_kv_heads,
                                               q_size,
                                               tile_size,
                                               head_dim,
                                               /*use_grouped=*/false);

    ov::ResultVector model_results;
    std::string model_name;
    if (is_final_tile) {
        // PA's attention output exits the layer body as rank-2
        // [q_size, num_q_heads * head_dim] (the SDPAToPagedAttention rewrite
        // leaves the PA op feeding an o_proj MatMul with that layout). The
        // HFA helper produces rank-3 [batch, seq, H*S]; replicate its math
        // but reshape to rank-2 and emit in output_dtype so the swapped
        // layer body matches every downstream consumer the partitioner wired.
        auto final_result =
            std::make_shared<ov::op::v1::Divide>(results.acc.get_node_shared_ptr(), results.d.get_node_shared_ptr());
        final_result->set_friendly_name("pa_final_result");
        // [1, H, q, S] -> [1, q, H, S]
        auto transpose_order =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 2, 1, 3});
        auto transposed = std::make_shared<ov::op::v1::Transpose>(final_result, transpose_order);
        // [1, q, H, S] -> [q, H*S]
        auto reshape_pattern = std::make_shared<ov::op::v0::Constant>(
            ov::element::i64,
            ov::Shape{2},
            std::vector<int64_t>{static_cast<int64_t>(q_size), static_cast<int64_t>(num_q_heads * head_dim)});
        auto reshaped = std::make_shared<ov::op::v1::Reshape>(transposed, reshape_pattern, false);
        auto final_output = std::make_shared<ov::op::v0::Convert>(reshaped, output_dtype);
        final_output->set_friendly_name("pa_final_output");
        final_output->output(0).get_tensor().set_names({"output"});
        model_results = {std::make_shared<ov::op::v0::Result>(final_output)};
        model_name = "OnlineSoftmax_PA_Final_Tile";
    } else {
        model_results = create_regular_tile_outputs(results, input_dtype);
        model_name = "OnlineSoftmax_PA_Tile";
    }

    ov::ParameterVector model_params{past_acc, past_max, past_d, k_tile, v_tile, q, mask_tile};
    return std::make_shared<ov::Model>(model_results, model_params, model_name);
}

}  // namespace npuw
}  // namespace ov
