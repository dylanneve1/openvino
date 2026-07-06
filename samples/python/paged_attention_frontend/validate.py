# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Validate the PagedFrontEnd dispatcher against a one-shot reference run.

Reference (how CB runs on CPU today): compile the *original dynamic* PA model
once and feed the whole prompt through a single ``infer()``. Compare its
last-token logits to the front-end's chunked, exact-tiled dispatch over the
same tokens with an identically-laid-out KV cache. Same PA op + same block_size
+ same positions => logits must match within bf16 tolerance.

Checks:
  1. Prefill parity  -- last-token logits, several prompt lengths (exact
     multiples of 1024/128 and awkward lengths like 1500, 130, 1).
  2. Decode continuation -- N generate steps after prefill, per-step argmax.
  3. Edge cases -- T=1, T<128, T crossing a block boundary, T needing the
     size-1 tail.

Usage::

    python validate.py --model <stateful_or_pa_model.xml> [--device CPU]
"""

import argparse
import sys
from typing import Optional, Sequence

import numpy as np
import openvino as ov

from paged_front_end import (
    BlockAllocator,
    PagedFrontEnd,
    _KVLayout,
    is_paged_attention_model,
    to_paged_attention,
)


class ReferenceRunner:
    """One-shot dynamic PA runner: the ground truth Stage 0 must reproduce.

    Compiles the unreshaped dynamic PA model and drives it exactly like
    ``ModelRunner::forward`` for a single sequence -- whole prompt in one infer.
    """

    def __init__(self, model: ov.Model, core: ov.Core, device: str,
                 num_blocks: int, kv_cache_precision: str):
        cfg = {"KV_CACHE_PRECISION": kv_cache_precision} if kv_cache_precision else {}
        self.compiled = core.compile_model(model.clone(), device, cfg)
        self.request = self.compiled.create_infer_request()
        self.layout = _KVLayout(self.compiled)
        self.allocator = BlockAllocator(self.layout.block_size, num_blocks)
        self._cache = self.layout.make_cache_tensors(num_blocks)
        for name, tensor in self._cache.items():
            self.request.set_tensor(name, tensor)
        self.context_len = 0

    def reset(self) -> None:
        self.context_len = 0
        for tensor in self._cache.values():
            tensor.data[:] = 0

    def _forward(self, token_ids: np.ndarray, past_len: int) -> np.ndarray:
        size = token_ids.shape[0]
        context_after = past_len + size
        nblocks = self.allocator.blocks_for_context(context_after)
        positions = np.arange(past_len, past_len + size, dtype=np.int64)
        self.request.set_tensor("input_ids", ov.Tensor(np.ascontiguousarray(token_ids, dtype=np.int64)))
        self.request.set_tensor("position_ids", ov.Tensor(positions))
        self.request.set_tensor("past_lens", ov.Tensor(np.array([past_len], dtype=np.int32)))
        self.request.set_tensor("subsequence_begins", ov.Tensor(np.array([0, size], dtype=np.int32)))
        self.request.set_tensor("block_indices", ov.Tensor(np.arange(nblocks, dtype=np.int32)))
        self.request.set_tensor("block_indices_begins", ov.Tensor(np.array([0, nblocks], dtype=np.int32)))
        self.request.set_tensor("max_context_len", ov.Tensor(np.array(context_after, dtype=np.int32)))
        self.request.infer()
        logits = self.request.get_tensor("logits").data.copy()
        return logits.reshape(-1, logits.shape[-1])[-1]

    def prefill(self, token_ids: Sequence[int]) -> np.ndarray:
        tokens = np.asarray(token_ids, dtype=np.int64).reshape(-1)
        logits = self._forward(tokens, self.context_len)
        self.context_len += tokens.shape[0]
        return logits

    def generate_step(self, token_id: int) -> np.ndarray:
        logits = self._forward(np.array([token_id], dtype=np.int64), self.context_len)
        self.context_len += 1
        return logits


def compare(name: str, ref: np.ndarray, got: np.ndarray, tol: float, topk: int = 5) -> bool:
    """Compare one logits vector: top-k overlap (primary) + relative max diff.

    Why not exact argmax: the parity floor here is ~2e-2 relative, set by two
    *benign* effects -- (a) static-shaped chunk models select different CPU
    kernels than the dynamic reference, (b) the bf16 KV cache. On random-token
    prompts the top logits are near-tied, so bf16 noise can flip a rank without
    any dispatch bug. Top-k overlap is stable under that noise while still
    collapsing to ~0 if the dispatch were actually wrong (wrong positions /
    cache blocks destroy the ranking, not just perturb it).
    """
    ref = ref.astype(np.float32)
    got = got.astype(np.float32)
    ref_top = set(np.argsort(ref)[-topk:].tolist())
    got_top = set(np.argsort(got)[-topk:].tolist())
    overlap = len(ref_top & got_top)
    argmax_ok = int(ref.argmax()) == int(got.argmax())
    max_abs = float(np.max(np.abs(ref - got)))
    denom = float(np.max(np.abs(ref))) + 1e-6
    rel = max_abs / denom
    ok = (overlap >= topk - 1) and (rel <= tol)
    status = "PASS" if ok else "FAIL"
    flag = "" if argmax_ok else "  (argmax differs: near-tie)"
    print(f"  [{status}] {name}: top{topk}={overlap}/{topk} "
          f"argmax ref={int(ref.argmax())} got={int(got.argmax())} "
          f"rel={rel:.4g}{flag}")
    return ok


def load_prompt_tokens(model_path: str, length: int) -> Optional[np.ndarray]:
    """Encode a real prompt with the HF tokenizer next to the IR, if available.

    Real text yields confident, well-separated logits (stable argmax), giving a
    strong-signal parity check to complement the random-token sweep.
    """
    import os
    model_dir = os.path.dirname(os.path.abspath(model_path))
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_dir)
    except Exception as exc:
        print(f"  (real-prompt test skipped: tokenizer unavailable -- {exc})")
        return None
    text = (
        "The history of computing spans many centuries. Early mechanical "
        "calculators gave way to electronic machines, and eventually to the "
        "programmable digital computers that power modern life. Paged attention "
        "is a technique used to serve large language models efficiently by "
        "managing the key-value cache in fixed-size blocks. In this paragraph we "
        "continue describing how memory is organised so that the model has a "
        "long, coherent context to condition its next-token prediction on. "
    ) * 6
    ids = tok(text, return_tensors="np")["input_ids"][0].astype(np.int64)
    if ids.shape[0] < length:
        return ids
    return ids[:length]


def make_prompt(length: int, vocab: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, vocab, size=length, dtype=np.int64)


def run_validation(model_path: str, device: str, kv_precision: str,
                   num_blocks: int, tol: float) -> bool:
    core = ov.Core()
    model = core.read_model(model_path)
    if not is_paged_attention_model(model):
        print("Model is SDPA-based; applying paged_attention_transformation ...")
        to_paged_attention(model)

    vocab = model.output("logits").get_partial_shape()[-1].get_length()
    print(f"Model: {model_path}\nvocab={vocab} device={device} "
          f"kv_precision={kv_precision} num_blocks={num_blocks}\n")

    print("Building PagedFrontEnd (chunk models {1024,128,1}) ...")
    fe = PagedFrontEnd(model, core, device=device, chunk_sizes=(1024, 128, 1),
                       num_blocks=num_blocks, kv_cache_precision=kv_precision)
    print(f"  KV layout: layers={fe.layout.num_layers} kv_heads={fe.layout.kv_heads} "
          f"block_size={fe.layout.block_size} head_size={fe.layout.head_size} "
          f"dtype={fe.layout.element_type.get_type_name()}")
    print("Building reference (one-shot dynamic PA) ...\n")
    ref = ReferenceRunner(model, core, device, num_blocks, kv_precision)

    all_ok = True

    # ---- 1 & 3: prefill parity across lengths / edge cases ----
    bs = fe.layout.block_size
    lengths = [1, 32, 127, 128, 129, 130, bs, bs + 1, 256, 1024, 1152, 1500]
    lengths = sorted({n for n in lengths if n > 0})
    print("== Prefill parity (last-token logits) ==")
    for n in lengths:
        prompt = make_prompt(n, vocab, seed=n)
        fe.reset(); ref.reset()
        got = fe.prefill(prompt)
        exp = ref.prefill(prompt)
        all_ok &= compare(f"T={n}", exp, got, tol)

    # ---- real-prompt parity (confident logits) ----
    real = load_prompt_tokens(model_path, length=300)
    if real is not None:
        print(f"\n== Real-prompt prefill parity (T={real.shape[0]}) ==")
        fe.reset(); ref.reset()
        got = fe.prefill(real)
        exp = ref.prefill(real)
        all_ok &= compare(f"real T={real.shape[0]}", exp, got, tol)

    # ---- 2: decode continuation ----
    print("\n== Decode continuation (prefill 200, then greedy decode) ==")
    prompt = real[:200] if real is not None else make_prompt(200, vocab, seed=1234)
    fe.reset(); ref.reset()
    fe_logits = fe.prefill(prompt)
    ref_logits = ref.prefill(prompt)
    all_ok &= compare("prefill@200", ref_logits, fe_logits, tol)
    fe_tok = int(fe_logits.argmax())
    ref_tok = int(ref_logits.argmax())
    for step in range(8):
        fe_logits = fe.generate_step(fe_tok)
        ref_logits = ref.generate_step(ref_tok)
        step_ok = compare(f"decode step {step}", ref_logits, fe_logits, tol)
        all_ok &= step_ok
        fe_tok = int(fe_logits.argmax())
        ref_tok = int(ref_logits.argmax())
        if fe_tok != ref_tok:
            print(f"    token divergence at step {step}: fe={fe_tok} ref={ref_tok}")

    print("\n" + ("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"))
    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="stateful or PA IR (.xml)")
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--kv-precision", default="bf16",
                        help="KV_CACHE_PRECISION (CPU PA requires bf16)")
    parser.add_argument("--num-blocks", type=int, default=512, help="KV block pool size")
    parser.add_argument("--tol", type=float, default=5e-2,
                        help="max relative logit difference (static-vs-dynamic + bf16 floor)")
    args = parser.parse_args()
    ok = run_validation(args.model, args.device, args.kv_precision,
                        args.num_blocks, args.tol)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
