// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <optional>

#include "kv_cache_block_manager.hpp"
#include "online_softmax_tile.hpp"  // for OnlineSoftmaxTileInputId / OnlineSoftmaxTileOutputId
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
//                      OnlineSoftmaxTileInputId / OnlineSoftmaxTileOutputId for shape compatibility
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

    // Layer index parsed from the KEY_CACHE Parameter's name suffix
    // (e.g. "key_cache.7" → 7). Selects which entry of
    // LLMCompiledModel::m_pa_layer_managers this PA op binds against.
    // SDPAToPagedAttention emits one PA op per attention layer; the
    // partitioner isolates each PA op into its own subgraph, so the
    // layer index uniquely identifies the layer pool to consume.
    std::size_t _layer_index = 0;

    // Parameter index of each PA input in the source model's parameter list.
    // Populated by from() for downstream binding by the runtime orchestrator.
    std::map<PAInputId, std::size_t> _pa_param_index_map;

    // Tile sub-model parameter / output index maps. Tile model I/O shape
    // matches HFA's so we reuse its enums (OnlineSoftmaxTileInputId / OnlineSoftmaxTileOutputId).
    // Populated by Step 3b alongside tile model construction.
    std::map<OnlineSoftmaxTileInputId, std::size_t> _tile_param_index_map;
    std::map<OnlineSoftmaxTileOutputId, std::size_t> _tile_output_index_map;

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
    //   - no PA op is found (the subgraph is not a PA layer), or
    //   - the PA op's static shapes haven't been resolved (call
    //     BakePagedAttentionStaticShapes first), or
    //   - multiple PA ops appear in the same subgraph (NPUW partitioning
    //     should isolate each PA op into its own subgraph; if it doesn't,
    //     bail and fall back to the CPU PA executor).
    //
    // SDPAToPagedAttention emits one PA op per attention layer, naming the
    // KEY_CACHE/VALUE_CACHE parameters "key_cache.N"/"value_cache.N" with
    // N as the layer index. from() parses N out of the KEY_CACHE name into
    // _layer_index so the runtime knows which pool slot in
    // LLMCompiledModel::m_pa_layer_managers this lowering binds against.
    static std::optional<PagedAttention> from(const std::shared_ptr<ov::Model>& model);
};

// Process-wide context for handing PA layer managers from LLMCompiledModel
// (which owns m_pa_layer_managers) down to the partitioner's
// attempt_attention() / attn_subgraph partition_stage hook (which has no
// back-reference to LLMCompiledModel). The LLM compile path sets this
// before invoking the partitioner and clears it on scope exit; the hook
// reads the current value to populate compiled::PagedAttention's layer
// manager vectors. Single-threaded compile assumption — the same one
// LLMCompiledModel already relies on. If the assumption ever changes,
// promote this to a per-LLMCompiledModel context handle.
struct PagedAttentionPartitionerContext {
    std::vector<std::shared_ptr<KVCacheBlockManager>> key_managers;
    std::vector<std::shared_ptr<KVCacheBlockManager>> value_managers;
    ov::SoPtr<ov::ICompiledModel> compiled_tile_model;
    ov::SoPtr<ov::ICompiledModel> compiled_final_tile_model;
};

// Returns a pointer to the currently active context, or nullptr if no
// LLMCompiledModel is in the middle of compiling a PA-mode model.
const PagedAttentionPartitionerContext* current_pa_partitioner_context();

// RAII guard. Construct in LLMCompiledModel right before compile starts;
// destruction clears the context.
class PagedAttentionPartitionerScope {
public:
    PagedAttentionPartitionerScope(std::vector<std::shared_ptr<KVCacheBlockManager>> key_managers,
                                    std::vector<std::shared_ptr<KVCacheBlockManager>> value_managers,
                                    ov::SoPtr<ov::ICompiledModel> compiled_tile_model,
                                    ov::SoPtr<ov::ICompiledModel> compiled_final_tile_model);
    ~PagedAttentionPartitionerScope();
    PagedAttentionPartitionerScope(const PagedAttentionPartitionerScope&) = delete;
    PagedAttentionPartitionerScope& operator=(const PagedAttentionPartitionerScope&) = delete;
};

}  // namespace function

namespace compiled {

// Compile-time PagedAttention information. Parallels
// ov::npuw::compiled::HostFlashAttention. Constructed from
// function::PagedAttention once tile sub-models are compiled, and stashed
// in a subgraph context by the partition_stage so the runtime dispatch in
// attn_subgraph.cpp can retrieve it via get_compiled_pa().
struct PagedAttention {
    // Compiled tile sub-models (NPU). The function-level _tile_model /
    // _final_tile_model handles are transient — owned by LLMCompiledModel
    // and cleared once their compilations land in m_pa_tile_compiled /
    // m_pa_final_tile_compiled. This struct holds references to those
    // compiled handles for the runtime to invoke.
    ov::SoPtr<ov::ICompiledModel> _compiled_tile_model;
    ov::SoPtr<ov::ICompiledModel> _compiled_final_tile_model;

    // Per-layer K/V block pools — one entry per transformer layer. The
    // runtime dispatch (run() switch case BehaviorKind::Paged) indexes
    // this with _layer_index to reach the K/V manager pair for THIS PA
    // op's layer. shared_ptr instances are owned by LLMCompiledModel via
    // m_pa_layer_managers; we hold copies here so the dispatch doesn't
    // need a back-pointer to LLMCompiledModel. Empty until populated by
    // the partition-stage hook.
    std::vector<std::shared_ptr<KVCacheBlockManager>> _layer_key_managers;
    std::vector<std::shared_ptr<KVCacheBlockManager>> _layer_value_managers;

    // Configuration copied from function::PagedAttention at construction.
    int64_t _block_size = 0;
    std::size_t _num_blocks = 0;
    std::size_t _query_size = 0;
    std::size_t _head_size = 0;
    std::size_t _num_q_heads = 0;
    std::size_t _num_kv_heads = 0;
    float _scale = 0.0f;
    std::size_t _layer_index = 0;

    // PA op parameter-index map on the original (pre-tile-rewrite) model.
    // Lets the runtime locate block_indices, past_lens, etc. at infer time.
    std::map<PAInputId, std::size_t> _pa_param_index_map;

    // Tile sub-model parameter / output index maps.
    std::map<OnlineSoftmaxTileInputId, std::size_t> _tile_param_index_map;
    std::map<OnlineSoftmaxTileOutputId, std::size_t> _tile_output_index_map;

    PagedAttention() = default;

    // Construct from the function-side struct after tile-model compilation.
    PagedAttention(const function::PagedAttention& func_pa,
                   ov::SoPtr<ov::ICompiledModel> tile_compiled,
                   ov::SoPtr<ov::ICompiledModel> final_tile_compiled,
                   std::vector<std::shared_ptr<KVCacheBlockManager>> layer_key_managers = {},
                   std::vector<std::shared_ptr<KVCacheBlockManager>> layer_value_managers = {})
        : _compiled_tile_model(std::move(tile_compiled)),
          _compiled_final_tile_model(std::move(final_tile_compiled)),
          _layer_key_managers(std::move(layer_key_managers)),
          _layer_value_managers(std::move(layer_value_managers)),
          _block_size(func_pa._block_size),
          _num_blocks(func_pa._num_blocks),
          _query_size(func_pa._query_size),
          _head_size(func_pa._head_size),
          _num_q_heads(func_pa._num_q_heads),
          _num_kv_heads(func_pa._num_kv_heads),
          _scale(func_pa._scale),
          _layer_index(func_pa._layer_index),
          _pa_param_index_map(func_pa._pa_param_index_map),
          _tile_param_index_map(func_pa._tile_param_index_map),
          _tile_output_index_map(func_pa._tile_output_index_map) {}

    bool is_valid() const {
        return _compiled_tile_model && _compiled_final_tile_model && _block_size > 0;
    }
};

}  // namespace compiled
}  // namespace npuw
}  // namespace ov
