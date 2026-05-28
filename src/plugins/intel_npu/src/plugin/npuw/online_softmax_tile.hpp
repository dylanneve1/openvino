// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>

#include "openvino/core/model.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace npuw {

// Identifiers for the inputs of one online-softmax tile sub-model.
//
// A "tile" here is one iteration of FlashAttention's streaming attention
// loop: it consumes the running (acc, max, d) state and a chunk (tile)
// of K/V plus the query, then produces the updated (acc, max, d) state.
//
// Layout order is also the parameter order of the model returned by
// create_online_softmax_tile_model() — see that function's contract.
enum class OnlineSoftmaxTileInputId : uint8_t {
    PAST_ACC = 0,   // Accumulated attention output from previous tiles
    PAST_MAX = 1,   // Maximum values from previous tiles (for numerical stability)
    PAST_D = 2,     // Normalization denominator from previous tiles
    K_TILE = 3,     // Current K (key) tile slice
    V_TILE = 4,     // Current V (value) tile slice
    Q = 5,          // Query tensor (full, not tiled)
    MASK_TILE = 6,  // Current attention mask tile slice

    // Sentinel value for enum range
    COUNT
};

// Identifiers for the outputs of one online-softmax tile sub-model
// (regular, non-final tile). Final-tile output is a single concatenated
// attention result — see create_online_softmax_tile_model.
enum class OnlineSoftmaxTileOutputId : uint8_t {
    ACC = 0,   // Accumulated attention output
    MAXX = 1,  // Maximum values for numerical stability
    D = 2,     // Normalization denominator

    // Sentinel value for enum range
    COUNT
};

// Helper functions to convert enum values to string representations for
// logging/debugging.
inline const char* online_softmax_tile_input_id_to_string(OnlineSoftmaxTileInputId id) {
    switch (id) {
    case OnlineSoftmaxTileInputId::PAST_ACC:
        return "PAST_ACC";
    case OnlineSoftmaxTileInputId::PAST_MAX:
        return "PAST_MAX";
    case OnlineSoftmaxTileInputId::PAST_D:
        return "PAST_D";
    case OnlineSoftmaxTileInputId::K_TILE:
        return "K_TILE";
    case OnlineSoftmaxTileInputId::V_TILE:
        return "V_TILE";
    case OnlineSoftmaxTileInputId::Q:
        return "Q";
    case OnlineSoftmaxTileInputId::MASK_TILE:
        return "MASK_TILE";
    default:
        return "UNKNOWN";
    }
}

inline const char* online_softmax_tile_output_id_to_string(OnlineSoftmaxTileOutputId id) {
    switch (id) {
    case OnlineSoftmaxTileOutputId::ACC:
        return "ACC";
    case OnlineSoftmaxTileOutputId::MAXX:
        return "MAXX";
    case OnlineSoftmaxTileOutputId::D:
        return "D";
    default:
        return "UNKNOWN";
    }
}

// Build a tile sub-model that performs one iteration of FlashAttention's
// online-softmax loop.
//
// The tile body is independent of any particular caller. Both HFA's SDPA
// path (host_flash_attention.cpp) and PA's NPU lowering
// (paged_attn_runtime.cpp) compose this kernel into their own runtime
// orchestrators: HFA feeds it tiles from a Concat'd KV cache; PA feeds
// it block-shaped K/V slices fetched via the block table.
//
// Inputs (declared in this order; matches OnlineSoftmaxTileInputId):
//   0 past_acc   [batch, num_heads, seq_len, head_dim]
//   1 past_max   [batch, num_heads, seq_len, 1]
//   2 past_d     [batch, num_heads, seq_len, 1]
//   3 k_tile     [batch, kv_num_heads, tile_size, head_dim]
//   4 v_tile     [batch, kv_num_heads, head_dim, tile_size]
//   5 q          [batch, num_heads, seq_len, head_dim]
//   6 mask_tile  [batch, 1, seq_len, tile_size]
//
// Outputs for is_final_tile=false: {acc, maxx, d} matching
// OnlineSoftmaxTileOutputId — fed into the next tile's past_* inputs by
// the caller's runtime loop.
//
// Output for is_final_tile=true : {output} (acc/d divided, transposed
// to [batch, seq_len, num_heads, head_dim], reshaped to
// [batch, seq_len, num_heads*head_dim], cast to output_dtype).
//
// fused_flash_attention=true uses the NPU's
// intel_npu::op::FlashAttentionTile fused kernel; false uses the
// standard-ops decomposition (MatMul, Exp, ReduceMax, ReduceSum, ...).
// Both compile to NPU.
std::shared_ptr<ov::Model> create_online_softmax_tile_model(const ov::Shape& q_shape,
                                                            const ov::element::Type& input_dtype,
                                                            const ov::element::Type& mask_dtype,
                                                            int64_t tile_size,
                                                            size_t kv_num_heads,
                                                            bool is_final_tile = false,
                                                            bool fused_flash_attention = false,
                                                            const ov::element::Type& output_dtype = ov::element::f16);

// PA-flavoured tile model. Adapts the kernel I/O to match the layouts
// produced by SDPAToPagedAttention + ConvertPagedAttnInputs:
//   - Q is rank-2 [q_size, H*S]; reshaped+transposed internally to
//     [1, H, q_size, S] for the kernel body.
//   - K_TILE keeps the body's [1, Hk, block_size, S] shape — matches the
//     KV-cache pool block layout SDPAToPagedAttention bakes.
//   - V_TILE is [1, Hk, block_size, S] (same layout as KV blocks);
//     transposed internally to [1, Hk, S, block_size] for the kernel.
//   - mask_tile is rank-2 [q_size, block_size]; reshaped to
//     [1, 1, q_size, block_size] for broadcast over heads.
//
// past_acc / past_max / past_d / output keep the body's 4-D shapes —
// the orchestrator owns those state buffers and never has to materialise
// PA-side equivalents.
std::shared_ptr<ov::Model> create_pa_online_softmax_tile_model(size_t num_q_heads,
                                                               size_t num_kv_heads,
                                                               size_t q_size,
                                                               size_t head_dim,
                                                               int64_t block_size,
                                                               const ov::element::Type& input_dtype,
                                                               const ov::element::Type& mask_dtype,
                                                               bool is_final_tile = false,
                                                               const ov::element::Type& output_dtype = ov::element::f16);

}  // namespace npuw
}  // namespace ov
