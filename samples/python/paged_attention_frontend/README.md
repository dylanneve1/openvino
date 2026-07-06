# NPUW Paged-Attention Front-End — Stage 0 PoC (standalone, CPU)

A standalone Python proof-of-concept for the NPUW Paged-Attention (PA) front-end,
**Stage 0** (functional pipeline). It accepts the fully dynamic, stateless,
PA-based model that OpenVINO GenAI's continuous-batching (CB) pipeline hands to a
device, derives *semi-static* "chunk" sub-models from it, and dynamically
dispatches an input sequence across them — all on **public OpenVINO API**, CPU
only, with **no `intel_npu` / NPUW plugin changes**.

This is a de-risking spike: the eventual front-end and Stage 1 are C++ in the
plugin. The *understanding* transfers; this code does not.

## What it does

Given a dynamic PA model, `PagedFrontEnd`:

1. **Derives chunk variants.** For each static flat-token size in `{1024, 128, 1}`
   it `clone()`s the model and `reshape()`s **only** the token dimension
   (`input_ids`, `position_ids`), leaving `past_lens`, `subsequence_begins`,
   `block_indices`, `block_indices_begins`, `max_context_len`, and every
   `key_cache.N` / `value_cache.N` **dynamic** — the "semi-static,
   context-dynamic" model the ticket asks for. Each variant is compiled on CPU.
2. **Shares one KV cache.** A self-managed monotonic block pool allocates one
   `key_cache.N` / `value_cache.N` tensor per layer; the **same** tensors are
   bound to every sub-request via `set_tensor`, so all chunks read/write one pool.
3. **Dispatches by exact tiling.** A prefill of `T` tokens is decomposed
   **exactly** into the static sizes (greedy largest-first; `1` guarantees no
   remainder), so every chunk fully fills its model — **zero padding, zero
   masking**. Generation uses the `S=1` model.

### The key design decision: exact tiling, no masking

PA has **no per-token attention mask** — attention is controlled entirely by
`past_lens` / `subsequence_begins` / `block_indices`. Padding a chunk would write
junk KV and desync position/`past_lens` accounting. Stage 0 therefore tiles `T`
exactly (`1500 = 1×1024 + 3×128 + 92×1`) so `real_len == S` always. The size-1
tail is slow but this is the *functional* milestone; the cheap remedy is more
static sizes, **not** masking. Padding+masking is deferred out of Stage 0.

## Files

| File | Purpose |
|---|---|
| `paged_front_end.py` | `PagedFrontEnd` + `BlockAllocator` + PA helpers (the library) |
| `convert_pa_model.py` | Apply `paged_attention_transformation` to a stateful IR → PA IR |
| `validate.py` | Reference (one-shot dynamic PA) vs chunked-dispatch parity harness |

## Requirements

- A locally built OpenVINO with the Python bindings (exposes
  `openvino._offline_transformations.paged_attention_transformation`).
- `numpy`, `ml_dtypes` (bf16 numpy view). `transformers` is optional — used only
  for the real-prompt parity check; the harness falls back to random tokens.

Set the environment up first:

```bash
source <openvino-build>/install/setupvars.sh
```

## Usage

The harness accepts either a **stateful** text-generation IR (it applies the PA
transform on the fly) or a pre-converted **PA** IR.

```bash
# End-to-end parity check (stateful IR is transformed automatically):
python validate.py --model /path/to/openvino_model.xml

# Options:
#   --device CPU            underlying device (Stage 0 is CPU only)
#   --kv-precision bf16     KV_CACHE_PRECISION (CPU PA requires bf16 on this HW)
#   --num-blocks 512        KV block-pool size
#   --tol 5e-2              max relative logit diff (see "Tolerance" below)

# Optionally materialise a PA IR and inspect its signature:
python convert_pa_model.py stateful_model.xml pa_model.xml
```

### Library usage

```python
import openvino as ov
from paged_front_end import PagedFrontEnd, to_paged_attention, is_paged_attention_model

core = ov.Core()
model = core.read_model("openvino_model.xml")
if not is_paged_attention_model(model):
    to_paged_attention(model)                 # apply paged_attention_transformation

fe = PagedFrontEnd(model, core, device="CPU", chunk_sizes=(1024, 128, 1))
fe.reset()
logits = fe.prefill(prompt_token_ids)          # last-token logits [vocab]
tok = int(logits.argmax())
for _ in range(n):                             # greedy decode
    logits = fe.generate_step(tok)
    tok = int(logits.argmax())
```

## Input contract (verified against Qwen3-0.6B on CPU)

After `paged_attention_transformation` (`attention_mask` / `beam_idx` removed):

| Input | Type | Shape |
|---|---|---|
| `input_ids` | i64 | `[-1]` |
| `position_ids` | i64 | `[-1]` (or `[3,-1]` for M-RoPE) |
| `past_lens` | i32 | `[-1]` — `0` ⇒ prefill, `>0` ⇒ decode |
| `subsequence_begins` | i32 | `[-1]` |
| `block_indices` | i32 | `[-1]` |
| `block_indices_begins` | i32 | `[-1]` |
| `max_context_len` | i32 | `{}` (scalar) |
| `key_cache.N` / `value_cache.N` | bf16* | `[num_blocks, kv_heads, block_size, head_size]` |

Output: `logits` `[tokens, 1, vocab]`.

\* The KV **precision and block_size are read from the compiled model, not
hard-coded.** On this CPU the PA op enforces a **bf16** KV cache with
**block_size 32** (layout `[?, kv_heads, 32, head_size]`); the default u8
KV quantization uses block_size 40. `PagedFrontEnd` follows whatever the plugin
fixes, so it adapts to other back-ends/precisions.

## Validation

`validate.py` compiles the **original dynamic** PA model once and runs the whole
prompt through a single `infer()` — exactly how CB runs on CPU today — then
compares its last-token logits to the front-end's chunked dispatch over the same
tokens and the same cache layout. It checks:

1. **Prefill parity** across lengths, including exact multiples of 1024/128 and
   awkward lengths (1500, 130, 129, 1) and block-boundary crossings.
2. **Real-prompt parity** on a tokenized paragraph (confident logits).
3. **Decode continuation** — 8 greedy steps after prefill.

### Tolerance

The parity floor is ~2e-2 relative and comes from two **benign** sources, not
from the dispatch:

- **Static vs dynamic kernels** — a static `[128]` chunk model selects different
  CPU kernels than the dynamic `[-1]` reference, so even a single 128-token chunk
  (computationally identical to the one-shot) differs by ~1.8e-2.
- **bf16 KV cache** — accumulation noise.

On random-token prompts the top logits are near-tied, so bf16 noise can flip a
rank without any bug. The harness therefore gates on **top-5 overlap** (stable
under that noise, but collapses to ~0 if positions/blocks are wrong) plus a
relative-diff bound, and reports exact-argmax agreement as information. A
deliberately corrupted dispatch (wrong positions) drops top-5 overlap to 1/5 and
`rel` to ~0.7 — the harness flags it.

Expected result: **ALL CHECKS PASSED**.

## Scope

**In:** graph derivation + dispatch + shared cache + parity, single sequence, CPU.
**Out (deferred):** any plugin/NPUW code (Stage 1); GenAI CB integration;
multi-sequence flat batching; intra-chunk padding+masking; NPU as the underlying
device (Stages 1–2).

See `drafts/stage0-paged-attention-frontend-plan.md` in the tooling repo for the
full plan and source references.
