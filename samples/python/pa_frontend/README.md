# PA front-end — Stage 0: 1:1 dispatch + trace model expectations

Standalone Python spike for the first Stage 0 story of CVS-190137 (PA
front-end PoC). Scope, per the story:

> Stand up the dispatcher with a 1:1 pass-through to the underlying
> dynamic-shape PA model, and trace the model's input expectations.
> No chunking, no derived static models yet — that's a follow-up.

Single sequence, CPU only, public OpenVINO API — no `intel_npu` / NPUW plugin
changes.

## What's here

| File | Role |
|---|---|
| `pa_dispatcher.py` | `PADispatcher`: compiles the dynamic PA model **untouched** (1:1 — no clone, no reshape) and passes every `infer` straight through to one `InferRequest`, tracing + verifying the PA contract on each dispatch. |
| `convert_pa_model.py` | Applies GenAI's `paged_attention_transformation` to a stateful IR to obtain a PA model with the exact input contract. |
| `run_dispatch.py` | Plays the CB pipeline's role for one sequence (whole-prompt prefill, then greedy decode steps) so the dispatcher has real dispatches to trace. |

## Usage

```bash
python convert_pa_model.py <stateful_model.xml> pa_model.xml
python run_dispatch.py pa_model.xml --prompt-len 37 --decode-steps 4
```

## The traced model expectations

Signature after `paged_attention_transformation` (`attention_mask` / `beam_idx`
are removed; there is **no per-token attention mask**):

| Input | Type | Shape | Expectation traced & verified per dispatch |
|---|---|---|---|
| `input_ids` | i64 | `[-1]` | flat token dim; all per-token inputs agree on its length |
| `position_ids` | i64 | `[-1]` | same length as `input_ids` (`[3,-1]` for M-RoPE) |
| `past_lens` | i32 | `[-1]` | one per subsequence, `>= 0`; `0` ⇒ prefill, `>0` + 1 token ⇒ decode |
| `subsequence_begins` | i32 | `[-1]` | prefix-sum over the flat token dim: starts at 0, strictly increasing, ends at token count |
| `block_indices` | i32 | `[-1]` | physical KV block ids, all inside the bound cache pool |
| `block_indices_begins` | i32 | `[-1]` | prefix-sum over `block_indices`; each subsequence's blocks must cover `past_len + scheduled` tokens |
| `max_context_len` | i32 | scalar | `>=` every subsequence's post-step context |
| `key_cache.N` / `value_cache.N` | per device | `[num_blocks, kv_heads, block_size, head_size]` | dim 0 dynamic; `kv_heads`/`block_size`/`head_size` and the element type are **fixed by the device plugin at compile time** (CPU: `block_size` 32, bf16 KV) — read back from the compiled model, never assumed |

Output: `logits` f32 `[tokens, 1, vocab]` (buffer reused between infers — copy out).

Violations are logged as `EXPECTATION VIOLATED` and raise by default
(`strict=False` to probe).

## Explicitly not in this story (follow-ups)

- Deriving semi-static chunk models (pin token dim, keep context dynamic).
- Chunking the input sequence and dispatching across chunk models.
- Generate-case dispatch to a dedicated 1-token model.
- Reference/parity validation of a dispatched run vs a one-shot run.
- Any plugin/NPUW code (Stage 1+).
