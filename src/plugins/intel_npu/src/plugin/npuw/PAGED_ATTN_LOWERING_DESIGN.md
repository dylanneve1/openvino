# Paged Attention on NPU — Lowering Design

**Status:** Draft for review. Branch: `paged_attn_v3`. Not implemented yet.
**Author:** Dylan Neve, with research assistance.
**Audience:** NPUW team, attention/HFA reviewers, Xiong Gao.

---

## TL;DR

Make `ov::op::PagedAttentionExtension` run on NPU by **paralleling HFA's runtime orchestration pattern**, not by writing a graph-level Loop. Compile a single tile sub-model once; iterate over the request's active blocks in C++; bind each block's K/V via `set_tensor` from `KVCacheBlockManager`. Dynamic block growth falls out naturally because the iteration count is a runtime quantity. No new NPU compiler features required.

Estimated effort: ~1 week of focused work. Storage refcount + tile sub-model construction (~80% reused from HFA) + runtime orchestrator + binding glue.

## Problem

Today, NPUW auto-routes subgraphs containing `ov::op::PagedAttentionExtension` to the CPU plugin — `compiled_model.cpp:682-684`:

```cpp
// Auto-route subgraphs containing PagedAttentionExtension to
// NPUW_ATTN_DEVICE (default CPU). The PA op has no NPU kernel, so it
// must execute on the host via the CPU plugin.
```

The user's PA path (M1 + M2 commits on `paged_attn_v2`) produces correct output but attention compute runs on host CPU. The rest of the LLM runs on NPU. We want PA on NPU.

Required feature: **dynamic block growth** — the number of blocks per request grows during generation and must not require recompilation.

## Background — what the codebase already provides

### HFA already implements online softmax on NPU

`online_softmax_tile.cpp (online-softmax body)` is FlashAttention's online softmax built from primitives the NPU compiler already supports (MatMul, Add, Subtract, Exp, ReduceMax, ReduceSum, Multiply):

```cpp
maxx  = Maximum(qkm_max, past_max)         // m_new
p     = Exp(qkm − maxx)                    // exp(score − m_new)
l     = ReduceSum(p)                       // local denominator
alpha = Exp(past_max − maxx)               // rescale factor
d     = past_d * alpha + l                 // ℓ_new
acc   = past_acc * alpha + p · V_tile      // O_new
```

This compiles to NPU and runs fast. **We will reuse this verbatim.**

### HFA's iteration model is in C++, not in the IR

This is the critical finding from research. HFA compiles **one** tile sub-model:

```cpp
struct function::HostFlashAttention {
    std::shared_ptr<ov::Model> _tile_model;        // 1 compiled instance
    std::shared_ptr<ov::Model> _final_tile_model;  // 1 compiled instance
    // ...
};
```

…and `attn_subgraph.cpp:1044-1085` invokes it from C++ in a runtime loop, threading state between calls:

```cpp
for (size_t block_idx = 0; block_idx < past_key_blocks.size(); ++block_idx) {
    for (int64_t t = 0; t < tiles_in_block; ++t) {
        process_tile(regular_tile_request, _compiled_tile_model,
                     k_block, v_block, t * tile_size, ...);
    }
}
process_tile(final_tile_request, _compiled_final_tile_model,
             present_key_tensor, present_value_tensor, ...);
```

State threading is achieved by binding the previous iteration's outputs (`acc`, `max`, `d`) as the next iteration's `past_*` inputs.

**No `v5::Loop` op, no unrolling.** This is the pattern we will mirror for PA.

### NPU cannot compile control flow ops

The Explore agent confirmed: NPUW has no `v5::Loop` or `TensorIterator` handling. We **cannot** emit those from a lowering pass. Iteration must live in host C++.

### Plugin convention: PA is opaque to the IR

Neither the CPU nor GPU plugin decomposes PA into primitives. The CPU plugin has a per-token streaming executor (`executor_pa.cpp`). The GPU plugin has a partitioned OpenCL kernel. Both treat PA as a single op and dispatch a custom executor.

OpenVINO has **no decomposition pass for PA anywhere in the tree.** Writing one for NPU would be unprecedented and is likely the wrong abstraction. The right abstraction is: **a runtime executor that consumes the PA op's I/O contract and orchestrates HFA-style tile iteration over the block table.**

### `KVCacheBlockManager` provides the storage layer

Xiong's PR #35013 (merged into `paged_attn_v2`) gives us a fixed-size block pool with `allocate_block()` / `get_block_tensor()` / `clear_all()`. Missing: `acquire()` / `release()` for refcounting. Cheap to add.

## The PA op input contract (the 28-input monster)

From `src/core/dev_api/openvino/op/paged_attention.hpp:26-57`, the inputs we **must** consume:

| idx | name | shape | notes |
|---|---|---|---|
| 0 | query | `[B_token, H * S]` | required |
| 1 | key (new) | `[B_token, Hk * S]` | required |
| 2 | value (new) | `[B_token, Hk * S]` | required |
| 3 | key_cache | `[num_blocks, Hk, Bs, S]` | the pool; bound via block IDs |
| 4 | value_cache | `[num_blocks, Hk, Bs, S]` | same |
| 5 | past_lens | `[B_seq]` | tokens already cached per sequence |
| 6 | subsequence_begins | `[B_seq + 1]` | sequence offsets in the batch |
| 7 | **block_indices** | `[total_blocks]` | **THE block table** |
| 8 | block_indices_begins | `[B_seq + 1]` | per-sequence slice into block_indices |
| 9 | scale | `[]` scalar | typically `1/√d` |
| 12 | max_context_len | `[]` scalar | sliding window upper bound |

Inputs we **can ignore for the first pass** (set as empty/zero constants by the SDPAToPagedAttention pass):
- sliding_window (10), alibi_slopes (11), score_aggregation_window (13), rotated_block_indices (14–16), xattention (17–19), sinks (20), adaptive_rkv (21–24), token_type_ids (25), qq_bias (26–27).

For a first cut, we lower **the core path** (inputs 0–9, 12) and assert that the optional inputs are their default empty/zero constants. We add support for sliding_window, RoPE, etc. in follow-up commits once the core path works.

For NPUW LLMPipeline (single-sequence), `B_seq = 1`, so `subsequence_begins = [0, B_token]`, `block_indices_begins = [0, num_active_blocks]`. Simpler than the multi-sequence batched case GenAI ContinuousBatchingPipeline targets.

## Design — `function::PagedAttention` runtime extension

### File layout

```
src/plugins/intel_npu/src/plugin/npuw/
├── online_softmax_tile.{hpp,cpp}     ← NEW: shared FlashAttention tile kernel
│                                       (extracted from host_flash_attention.cpp)
├── host_flash_attention.{hpp,cpp}    ← SDPA-side consumer of the tile kernel
├── paged_attn_runtime.{hpp,cpp}      ← NEW: PA-side consumer + runtime extension
└── attn/attn_subgraph.cpp            ← extended: detect & route PA tile loop
```

We do **not** add a new IR pass. The skeleton `paged_attn_to_hfa_tiles.{hpp,cpp}` was a misread of the architecture and is removed in favour of this design.

The tile builder was moved from `host_flash_attention.{hpp,cpp}` into its own `online_softmax_tile.{hpp,cpp}` module: the kernel is just one iteration of FlashAttention's online-softmax loop and is independent of any particular caller. Both HFA's SDPA path and PA's lowering now compose this shared kernel into their own runtime orchestrators.

### The `PagedAttention` struct (in `paged_attn_runtime.hpp`)

```cpp
namespace ov::npuw::function {

struct PagedAttention {
    // Compiled tile sub-models — structurally identical to HFA's, with
    // different parameter index map because PA's I/O differs from SDPA's.
    std::shared_ptr<ov::Model> _tile_model;        // one block iteration
    std::shared_ptr<ov::Model> _final_tile_model;  // last iter w/ acc/d

    // Tile shape config
    int64_t _block_size  = 0;   // tokens per block, e.g. 1024
    std::size_t _query_size = 0;
    std::size_t _head_size  = 0;

    // PA op I/O index map — where in the original model's parameter list
    // each PA input lives. Populated during from().
    std::map<PAInputId, std::size_t> _pa_param_index_map;

    // Tile model I/O index map — analogue of HFA's _tile_param_index_map
    std::map<OnlineSoftmaxTileInputId,  std::size_t> _tile_param_index_map;
    std::map<OnlineSoftmaxTileOutputId, std::size_t> _tile_output_index_map;

    bool is_valid() const { return _tile_model && _final_tile_model && _block_size > 0; }

    static std::optional<PagedAttention> from(const std::shared_ptr<ov::Model>& model);
};

}  // namespace ov::npuw::function
```

`from()` does:
1. Walk the model. Find the `PagedAttentionExtension` op (assert exactly one for the LLM use case — multi-PA models are out of scope for v1).
2. Extract Q shape, head dimensions, scale, block size from the PA op and its inputs/rt_info.
3. Build a tile sub-model with HFA's exact online-softmax body. Inputs: `(past_acc, past_max, past_d, k_tile, v_tile, q, mask_tile)`. Outputs: `(acc, maxx, d)`.
4. Build a `_final_tile_model` that adds the `acc / d` division and any required transpose to match PA's output contract.
5. Record parameter indices into the original model so the runtime knows which tensor is which.

The **tile model body is copied wholesale from HFA**. Where HFA's tile-builder is parameterised by SDPA's K/V shape, ours is parameterised by PA's `key_cache` block shape. The math is identical.

### Compile-time integration

In `llm_compiled_model.cpp`, when `m_pa_mode` is true and the model has a `PagedAttentionExtension`:

1. Run `BakePagedAttentionStaticShapes` (existing, M1) — bakes static shapes for NPU.
2. Run `ov::npuw::function::PagedAttention::from(kvcache_model)` — produces the tile sub-models.
3. Compile the tile sub-models on NPU. Store handles on `LLMCompiledModel`.
4. **Skip** wrapping the original PA op for execution. The runtime path takes over (next section).
5. **Remove the PA-to-CPU auto-route from `compiled_model.cpp:682-698`** *only when* the runtime path is in place. Until then, gate it behind `NPUW_LLM_PA_NPU_LOWERING` config flag for A/B testing.

### Runtime orchestration (in `PagedLLMInferRequest`)

Replace `infer_generate()`'s PA invocation with a tile loop that mirrors `attn_subgraph.cpp:1044-1085`:

```cpp
void PagedLLMInferRequest::run_paged_attention(...) {
    // 1. Read block table from PA op runtime inputs.
    const auto block_indices_param = get_tensor(_pa.block_indices_idx);
    const auto* block_indices = block_indices_param.data<int32_t>();
    const auto num_active_blocks = block_indices_param.get_shape()[0];

    // 2. Initialize state: acc=0, max=-inf, d=0.
    init_tile_state(state_acc, state_max, state_d);

    // 3. Walk active blocks. Bind each to the tile model. Run.
    for (size_t i = 0; i + 1 < num_active_blocks; ++i) {  // all but last
        const auto phys_id = block_indices[i];

        tile_request.set_tensor("k_tile", kv_pool->get_block_tensor(phys_id));
        tile_request.set_tensor("v_tile", kv_pool->get_block_tensor(phys_id));
        tile_request.set_tensor("past_acc", state_acc);
        tile_request.set_tensor("past_max", state_max);
        tile_request.set_tensor("past_d",   state_d);
        tile_request.set_tensor("q",        q_tensor);
        tile_request.set_tensor("mask_tile", mask_tile_for_block(i));
        tile_request.infer();
        // state_* now hold this iter's outputs, fed into next iter.
    }

    // 4. Final tile with present K/V (this token's contribution).
    final_tile_request.set_tensor("k_tile", present_k_tensor);
    final_tile_request.set_tensor("v_tile", present_v_tensor);
    // ... same past_* bindings, plus the new-token K/V ...
    final_tile_request.infer();
    // final_tile output = the PA op's primary output.
}
```

**This is the entire "lowering."** No IR rewrite. No new ops emitted. The PA op's compute is replaced by a host-side tile orchestrator that uses the NPU-compiled tile sub-models.

### Dynamic block growth

When the active block tail fills:
1. `KVCacheBlockManager::allocate_block()` returns a new physical ID.
2. `PagedLLMInferRequest` appends that ID to its `block_indices` tensor (and bumps `block_indices_begins[1]`).
3. Next forward pass: the tile loop sees `num_active_blocks + 1` and iterates one more time.

No recompilation. Pool capacity (compile-time `max_num_blocks`) is the only cap, and it's set to the device's max-context worth of blocks at startup.

### Prefix cache integration

`PrefixCacheManager::get_block(hash)` already exists and is hash-keyed. With refcounted blocks:
1. On prefix hit, `PrefixCacheManager` returns a list of physical block IDs and `acquire()`s them.
2. `PagedLLMInferRequest` populates `block_indices` with the hit's IDs.
3. Skip prefill for those tokens (set `past_lens[0]` to the number of cached tokens).
4. Tile loop iterates over the cached blocks **without re-running prefill compute** — pure attention over warm KV.

This delivers the TTFT-warm win without any change to the lowering itself.

## Comparison with what I sketched first

The skeleton I committed (`paged_attn_to_hfa_tiles.{hpp,cpp}`) imagined an IR pass. That was wrong. Specifically:

| skeleton (wrong) | proper design |
|---|---|
| `class PagedAttnToHFATiles : public ov::pass::ModelPass` | `struct function::PagedAttention` + runtime orchestrator |
| `run_on_model` rewrites the graph emitting Loop + Gather | No graph rewrite. Tile model construction + runtime binding. |
| Requires NPU Loop op support | Requires nothing new — same primitives HFA uses |
| 2–3 weeks of new IR machinery | ~1 week, ~80% code reuse from HFA |

The skeleton should be removed in the next commit on this branch and replaced with the correct architecture's stub.

## Concrete next commits (revised)

1. **Refcount `KVCacheBlockManager`** (`acquire` / `release` / `refcount`). Self-contained.
2. **Wire `LLMCompiledModel` to own per-layer `KVCacheBlockManager`s** when `m_pa_mode` is true. Pass shared handles to infer requests.
3. **Add `function::PagedAttention` struct + `from()` factory.** Builds the two tile sub-models. Heavily cribs from `host_flash_attention.cpp`'s tile builder.
4. **Compile + store tile sub-models** on `LLMCompiledModel` during `m_pa_mode` setup. Add a compile-flag gate `NPUW_LLM_PA_NPU_LOWERING` (default off until validated).
5. **Replace the PA op execution path** in `PagedLLMInferRequest::infer_generate()` with the tile loop. Same shape as `attn_subgraph.cpp:1044-1085`.
6. **Refactor mask tile slicing.** PA's mask is `[B_token, B_token]` causal; tiles see `[B_token, block_size]` slices. Reuse HFA's mask cache logic from `HFARuntimeContext`.
7. **Remove PA-to-CPU auto-route** in `compiled_model.cpp:682-698` when `NPUW_LLM_PA_NPU_LOWERING` is on.
8. **Refcount-aware `PrefixCacheManager`.** Hand out block IDs, bump refcount on hit, release on evict.
9. **End-to-end test:** multi-turn, prefix hit reuses blocks, no recompute, NPU peak utilization > host fallback baseline.
10. **Benchmarks:** TTFT cold, TTFT warm, tokens/sec, peak memory.

Steps 1–7 deliver **PA on NPU with dynamic block growth**. Step 8 delivers prefix sharing. Steps 9–10 close out the milestone.

## Open questions for the NPUW team

1. **Multiple PA ops per model.** If a model is partitioned such that each layer has its own PA op (per-layer block indices is a config option of `SDPAToPagedAttention`), do we orchestrate them independently? The runtime would have N tile loops per generate step. Probably fine, mirror what HFA does for multi-SDPA models.
2. **Mask tile layout.** HFA's mask tiles are sliced from a full causal mask. PA's masking is implicit in `past_lens` / `subsequence_begins`. Need to materialize a mask tile per iteration. Confirmation that this materialization is cheap enough vs. doing the masking inside the tile body.
3. **Position IDs / RoPE.** PA's input 14–16 are rotation inputs. For RoPE-using models, we either (a) keep RoPE outside the PA op (apply to Q/K before PA), or (b) handle it inside the tile body. Need to decide based on what `SDPAToPagedAttention` produces for the target models.
4. **Multi-sequence batch.** First cut targets `B_seq = 1` (LLMPipeline). Generalising to ContinuousBatchingPipeline (multiple sequences interleaved) adds significant complexity to the mask + state threading. Out of scope for v1.

## References

- `host_flash_attention.cpp:40-297` — HFA tile model construction & online softmax.
- `attn/attn_subgraph.cpp:1043-1085` — HFA's runtime tile orchestration loop.
- `host_flash_attention.hpp:120-176` — `function::HostFlashAttention` struct (the template for `function::PagedAttention`).
- `compiled_model.cpp:682-698` — current PA-to-CPU auto-route.
- `llm_compiled_model.cpp:856-876` — current PA application logic.
- `npuw_transformations/paged_attention_static.cpp` — M1's static-shape baking pass.
- `paged_llm_infer_request.cpp` — M2's PA single-sequence path.
- `src/core/dev_api/openvino/op/paged_attention.hpp:26-57` — PA op input contract (28 inputs).
- `src/common/transformations/src/transformations/paged_attention/state_management_pattern.cpp:596-775` — how `SDPAToPagedAttention` wires the inputs.
- Kwon et al. 2023, "Efficient Memory Management for Large Language Model Serving with PagedAttention" — algorithmic background.
