// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <optional>

#include "host_flash_attention.hpp"  // for HFATileInputId / HFATileOutputId
#include "openvino/core/model.hpp"

namespace ov {
namespace npuw {

// Identifiers for ov::op::PagedAttentionExtension's 28 inputs, kept in
// declaration order so the enum value matches the input index. Many of
// the optional features (RoPE rotation, sliding window, ALiBi slopes,
// XAttention, attention sinks, adaptive R-KV, Q-bias, token type ids) are
// fed empty / zero constants by ov::pass::SDPAToPagedAttention when not
// in use. The PA NPU lowering targets the core attention path
// (QUERY..MAX_CONTEXT_LEN) first; the rest land behind a feature gate
// once the core path is validated.
enum class PAInputId : uint8_t {
    QUERY = 0,                                            // [B_token, H * S]
    KEY = 1,                                              // [B_token, Hk * S]    new keys
    VALUE = 2,                                            // [B_token, Hk * S]    new values
    KEY_CACHE = 3,                                        // [num_blocks, Hk, Bs, S]
    VALUE_CACHE = 4,                                      // [num_blocks, Hk, Bs, S]
    PAST_LENS = 5,                                        // [B_seq] i32
    SUBSEQUENCE_BEGINS = 6,                               // [B_seq + 1] i32
    BLOCK_INDICES = 7,                                    // [total_blocks] i32   THE block table
    BLOCK_INDICES_BEGINS = 8,                             // [B_seq + 1] i32
    SCALE = 9,                                            // [] scalar (1/sqrt(d))
    SLIDING_WINDOW = 10,                                  // [] scalar i32 (0 = unlimited)
    ALIBI_SLOPES = 11,                                    // [H] or empty
    MAX_CONTEXT_LEN = 12,                                 // [] scalar i32
    SCORE_AGGREGATION_WINDOW = 13,                        // optional
    ROTATED_BLOCK_INDICES = 14,                           // optional
    ROTATION_DELTAS = 15,                                 // optional
    ROTATION_TRIG_LUT = 16,                               // optional
    XATTENTION_THRESHOLD = 17,                            // optional
    XATTENTION_BLOCK_SIZE = 18,                           // optional
    XATTENTION_STRIDE = 19,                               // optional
    SINKS = 20,                                           // optional
    ADAPTIVE_RKV_START_SIZE = 21,                         // optional
    ADAPTIVE_RKV_EVICTABLE_SIZES = 22,                    // optional
    ADAPTIVE_RKV_DIVERSITY_BLOCK_SET_INDICES = 23,        // optional
    ADAPTIVE_RKV_DIVERSITY_BLOCK_SET_INDICES_BEGINS = 24, // optional
    TOKEN_TYPE_IDS = 25,                                  // optional
    QQ_BIAS = 26,                                         // optional
    QQ_BIAS_BEGINS = 27,                                  // optional
    COUNT
};

const char* pa_input_id_to_string(PAInputId id);

namespace function {

// PagedAttention runtime extension — parallel to function::HostFlashAttention.
//
// from() walks the model looking for an ov::op::PagedAttentionExtension op,
// extracts the per-layer attention configuration (head dims, block size,
// scale, KV-cache element type), and builds two sub-models:
//
//   _tile_model        one iteration of the online-softmax loop, structurally
//                      identical to HFA's tile model. Inputs / outputs follow
//                      HFATileInputId / HFATileOutputId for shape compatibility
//                      with the existing tile orchestrator.
//   _final_tile_model  the last iteration, including the acc/d division and
//                      any reshape/transpose needed to match the PA op's
//                      output contract.
//
// Both sub-models are compiled by LLMCompiledModel and invoked from
// PagedLLMInferRequest's tile orchestrator (Step 5). The tile body is the
// SAME math HFA already runs on NPU (host_flash_attention.cpp:264-297). The
// "lowering" of the PA op is purely a runtime orchestration: walk
// block_indices in C++, bind each block's K/V via KVCacheBlockManager, run
// the tile, thread (acc, max, d) into the next iteration.
//
// See PAGED_ATTN_LOWERING_DESIGN.md for the full design.
//
// STATUS: skeleton. from() detects PA ops and records their config; tile
// model construction is deferred to a follow-up commit (Step 3b).
struct PagedAttention {
    // Sub-models — populated by Step 3b (NOT YET CONSTRUCTED).
    std::shared_ptr<ov::Model> _tile_model;
    std::shared_ptr<ov::Model> _final_tile_model;

    // Configuration extracted from the PA op + its rt_info.
    int64_t _block_size = 0;       // tokens per cache block (from key_cache shape[2])
    std::size_t _num_blocks = 0;   // pool capacity (from key_cache shape[0])
    std::size_t _query_size = 0;   // B_token for the variant
    std::size_t _head_size = 0;    // S — per-head dim
    std::size_t _num_q_heads = 0;  // H — number of query heads
    std::size_t _num_kv_heads = 0; // Hk — number of K/V heads (GQA → < H)
    float _scale = 0.0f;           // attention scale, usually 1/sqrt(head_size)

    // Parameter index of each PA input in the source model's parameter list.
    // Populated by from() for downstream binding by the runtime orchestrator.
    std::map<PAInputId, std::size_t> _pa_param_index_map;

    // Tile sub-model parameter / output index maps. Tile model I/O shape
    // matches HFA's so we reuse its enums (HFATileInputId / HFATileOutputId).
    // Populated by Step 3b alongside tile model construction.
    std::map<HFATileInputId, std::size_t> _tile_param_index_map;
    std::map<HFATileOutputId, std::size_t> _tile_output_index_map;

    bool is_valid() const {
        return _tile_model && _final_tile_model && _block_size > 0;
    }

    // Pre-construction check: did from() find a PA op and extract config?
    // True when the source model is a PA-mode LLM, even if tile models
    // haven't been built yet.
    bool config_extracted() const {
        return _block_size > 0 && _num_kv_heads > 0;
    }

    // Walks `model` looking for a PagedAttentionExtension op. Returns the
    // extracted config on success. Returns std::nullopt when:
    //   - no PA op is found (model is not in PA mode), or
    //   - the PA op's static shapes haven't been resolved (call
    //     BakePagedAttentionStaticShapes first), or
    //   - multiple PA ops are found (per-layer PA setup — not the LLM
    //     LLMPipeline shape this lowering currently targets).
    static std::optional<PagedAttention> from(const std::shared_ptr<ov::Model>& model);
};

}  // namespace function
}  // namespace npuw
}  // namespace ov
