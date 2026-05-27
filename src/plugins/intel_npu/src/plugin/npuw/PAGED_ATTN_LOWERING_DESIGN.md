# Paged Attention on NPU — Lowering Design

**Status:** Draft for review (paged_attn_v3 branch). Not implemented yet.
**Author:** Dylan Neve, with research assistance.
**Audience:** NPUW team, attention/HFA reviewers.

---

## Problem

Today on NPUW, a graph containing `ov::op::PagedAttentionExtension` is auto-routed to the CPU plugin because the NPU compiler has no PagedAttention kernel — see `compiled_model.cpp:682-684`:

```cpp
// Auto-route subgraphs containing PagedAttentionExtension to
// NPUW_ATTN_DEVICE (default CPU). The PA op has no NPU kernel, so it
// must execute on the host via the CPU plugin.
```

This means the user's PA path (M1 + M2 commits on `paged_attn_v2`) currently produces correct output but runs attention compute on the host. The rest of the LLM (matmuls, MoE, layernorm) executes on NPU; only attention falls back. The memory savings from block-paged KV are real, but the throughput win you'd expect from NPU-resident attention is lost.

Goal: make PagedAttention compute happen on NPU.

## Background

### What HFA already provides

`host_flash_attention.{hpp,cpp}` implements an **online-softmax tile loop** that compiles to NPU ops (MatMul, Add, Subtract, Exp, ReduceMax, ReduceSum). This is mathematically FlashAttention. See `host_flash_attention.cpp:264-297` for the core math:

```cpp
maxx  = Maximum(qkm_max, past_max)         // m_new
p     = Exp(qkm − maxx)                    // exp(score − m_new)
l     = ReduceSum(p)                       // local denominator
alpha = Exp(past_max − maxx)               // rescale factor for prior state
d     = past_d * alpha + l                 // ℓ_new
acc   = past_acc * alpha + p · V_tile      // O_new
```

HFA exposes two compiled models per LLM:

- `_tile_model` — one iteration of the loop. Inputs: `past_acc, past_max, past_d, k_tile, v_tile, q, mask_tile`. Outputs: `acc, maxx, d`.
- `_final_tile_model` — the last iteration, with the final `acc / d` division and any transpose.

The runtime walks tiles of K/V, threading the `acc/maxx/d` outputs of tile *i* into the `past_*` inputs of tile *i+1*. This is FlashAttention executing on NPU.

### What HFA already supports for blocks

Xiong's PR #35014 (`xiong/block_kv_pr3_hfa_decouple`, merged into `paged_attn_v2`) extends HFA's `from()` factory to accept K/V split across multiple block parameters:

```cpp
// host_flash_attention.hpp, struct HostFlashAttention
std::vector<std::size_t> _past_key_block_indices;    // block_0..block_N
std::vector<std::size_t> _past_value_block_indices;
```

HFA detects either monolithic K/V (single Concat input) or block-split K/V (multiple Concat inputs from `SplitKVCacheIntoBlocks`) and builds tile models that iterate across all of them.

**This means:** if we feed HFA an SDPA op whose K/V come from N block parameters, HFA already produces an NPU-executable tile loop that walks all N blocks. No new attention kernel needed.

The catch: HFA assumes the block parameters are **bound to physical tensors at compile time** (or at least their *count* is). It is not a runtime block-table-driven gather — the block layout is baked into the graph structure.

## Two implementation paths

### Path 2a — Full lowering (PA op → custom tile loop)

Write a new NPUW pass `PagedAttnToHFATiles` that:

1. Matches `ov::op::PagedAttentionExtension` in the graph.
2. Builds an HFA-style tile model whose inputs include the block table.
3. Wraps the tile model in a `Loop` op (or NPUW subgraph) that:
   - Reads `block_indices[i]`
   - Gathers `K_block_i` and `V_block_i` from the block pool
   - Threads `(past_acc, past_max, past_d)` between iterations
4. After the loop, performs the final division and transpose.

**Pros:**
- Preserves PA op semantics: dynamic block count at runtime, native vLLM-style block table indirection.
- Compatible with GenAI's `ContinuousBatchingPipeline` (uses real PA op contract).
- Unlocks COW, cross-request prefix sharing at arbitrary offsets, eviction.

**Cons:**
- Significant work: ~2-3 weeks. New IR transform, gather plumbing, Loop op integration.
- NPU compiler must support the Loop or unrolled equivalent. Compile-time block count cap likely still needed for static-shape compilation.
- Higher risk of subtle correctness bugs (mask handling, partial tail block, position embeddings).

### Path 2b — Hybrid via SDPA + SplitKVCacheIntoBlocks

In NPUW PA mode, instead of running `ov::pass::SDPAToPagedAttention`:

1. Keep SDPA in the graph.
2. Run Xiong's `SplitKVCacheIntoBlocks` (PR #35012, in master) to split KV into N block parameters.
3. HFA's existing factory picks up the block-split SDPA and builds the tile loop.
4. Bind each block parameter at runtime to a physical block from `KVCacheBlockManager`.

**Pros:**
- Reuses ALL existing infrastructure: HFA tile builder, NPU compilation, block binding helpers from PR #35018.
- ~1 week of work, mostly: refcounting in `KVCacheBlockManager` (Step 3 of the v2 plan), wiring `PagedLLMInferRequest` to bind through the pool, prefix-cache integration.
- Lower risk: tested code paths.
- Cross-request prefix sharing works (Concat happens after binding, so physical blocks can be non-contiguous in memory).

**Cons:**
- Block count is **compile-time fixed**. Need 17 blocks for a long request? Recompile. (Mitigation: compile with the max your hardware can fit; that's already what NPU does for sequence length anyway.)
- Block table indirection is implicit (binding order = logical order), not a runtime input. Multiple block-table layouts per request require multiple binding sets.
- Lose strict compatibility with GenAI's `ContinuousBatchingPipeline` API surface for the single-sequence LLMPipeline path. (CB pipeline still works via the PA op fallback on CPU.)
- Lose dynamic block growth mid-generation. If a request grows past the compiled block count, behavior is the same as today's max-context-len limit.

## Recommendation: Path 2a (full lowering)

**Constraint:** dynamic block growth is a required feature. Path 2b's compile-time-fixed block count cannot satisfy this:

- Path 2b's HFA tile loop iterates over **every** compile-time block parameter on every forward pass. A short request that only fills 3 of 64 compiled blocks pays 21× the attention cost — the unused blocks would have to be masked out (mask = -inf), but the compute still happens.
- Recompiling whenever a request grows past the current block count is not viable for an interactive single-user workload.

Therefore Path 2a — runtime block-table-indirected lowering — is the only option that meets the requirement.

The cost is real: ~2-3 weeks of focused work, including writing a new transformation pass, integrating with the NPU compiler's static-shape constraints, and validating attention correctness under mask + position-id edge cases.

The "static-shape constraint" interpretation for Path 2a:
- Compile with a **maximum number of physical blocks** in the pool (e.g., 64 blocks = 65536 tokens at block_size=1024). This is a static shape on the K/V cache tensors: `[max_num_blocks, num_kv_heads, block_size, head_size]`.
- `block_indices` is a variable-length runtime input of shape `[num_active_blocks]`. NPU may require this to be padded to a static max length; the PA op walks only the entries up to `num_active_blocks`.
- The tile loop iterates **only over active blocks** as indicated by `block_indices`. Inactive entries are skipped (no wasted compute).

This is the same trade-off vLLM makes: pool size is static at server start, request-level block count is dynamic up to that pool. The constraint matches NPU's static-compile model cleanly.

## What this branch contains today

`paged_attn_v3` is `paged_attn_v2` plus this design doc and a stub for `PagedAttnToHFATiles`. The stub does nothing yet; the structure is committed so an implementer can fill it in.

**This is not a working feature.** Don't merge `paged_attn_v3` into anything. It's a design + skeleton for review.

## Concrete next commits (Path 2a, in suggested order)

### Storage layer
1. **Refcount API on `KVCacheBlockManager`** (`acquire`/`release`/`refcount`), update tests.
   *Self-contained, no dependencies. Ship first.*
2. **Construct per-layer K/V managers in `LLMCompiledModel` when `m_pa_mode` is true.**
   Owned by the compiled model, shared across infer requests.

### Lowering pass — the core of Path 2a
3. **New IR pass `PagedAttnToHFATiles`** (this skeleton):
   - Match `ov::op::PagedAttentionExtension` nodes.
   - For each match, build a sub-graph that:
     a. Reads `block_indices[i]` for the current iteration.
     b. Uses `Gather` (or NPUW-specific equivalent) to fetch `K_block` and `V_block` from the `key_cache` / `value_cache` parameters along the `num_blocks` axis.
     c. Runs one online-softmax tile step (structurally identical to HFA's `_tile_model`).
     d. Threads `(past_acc, past_max, past_d)` between iterations.
   - Wrap the sub-graph in a `Loop` op (or NPUW unrolled equivalent) bounded by `num_active_blocks` from a runtime input.
   - After the loop, perform the final `acc / d` and any post-processing the PA op contract requires (transpose, etc.).
   - Match the original PA op's output shape and semantics so downstream graph stays unchanged.

4. **Replace `BakePagedAttentionStaticShapes` invocations** in `llm_compiled_model.cpp` with the new lowering pass when running on NPU. Keep `BakePagedAttentionStaticShapes` for the GenAI ContinuousBatchingPipeline path that still goes to CPU.

5. **Remove the PA-to-CPU auto-route** in `compiled_model.cpp:682-698` once the lowering is verified working. Until then, gate it behind a config flag so we can A/B test correctness.

### Runtime layer
6. **`PagedLLMInferRequest` allocates from the refcounted pool**, maintains a `block_indices` tensor, grows it as new blocks are allocated mid-generation.
7. **Hook `PrefixCacheManager`** to hand out refcount-bumped block IDs instead of tensor copies. On hit, populate `block_indices` directly from cached IDs.

### Validation
8. End-to-end test: small model, multi-turn chat, prefix hit reuses blocks with no recompute.
9. Correctness test: compare output vs. CPU-PA fallback on the same inputs.
10. Benchmark: peak memory, TTFT cold, TTFT warm, tokens/sec — vs. current PA-on-CPU baseline.

The lowering pass (step 3) is the load-bearing piece. Everything else is plumbing around it.

## Open questions for the NPUW team

1. **Compile-time block count cap:** what's a sensible default upper bound? Current xiong PR uses `NPUW_LLM_BLOCK_SIZE` (default 1024) without an explicit block count config — derived from max context length / block size.
2. **HFA block-mode runtime expectations:** when `_past_key_block_indices` has N entries, what's the contract for the bound tensor at compile time vs. swap time? Is there a tested pattern for "bind each slot to a different physical tensor between requests"?
3. **GenAI ContinuousBatchingPipeline:** does any NPUW workload today exercise it? If not, Path 2b's CB-on-CPU-only constraint is moot for the foreseeable future.
4. **PA op preservation:** should NPUW keep the PA op for *some* code paths (CB pipeline) while using SDPA lowering for others? If so, the gating logic in `llm_compiled_model.cpp` needs another branch.

## References

- `host_flash_attention.cpp:40-297` — HFA tile model construction, online softmax math.
- `host_flash_attention.hpp:120-176` — `function::HostFlashAttention` struct with block index vectors.
- `compiled_model.cpp:682-698` — PA-to-CPU auto-route.
- `llm_compiled_model.cpp:856-876` — current PA application logic.
- `npuw_transformations/paged_attention_static.cpp` — current static-shape baking pass (M1).
- `paged_llm_infer_request.cpp` — current PA single-sequence path (M2).
- xiong PRs #35012 / #35013 / #35014 / #35018 — block KV cache series.
- Kwon et al. 2023, "Efficient Memory Management for Large Language Model Serving with PagedAttention" — vLLM paper, Section 4 covers the runtime model this is converging toward.
