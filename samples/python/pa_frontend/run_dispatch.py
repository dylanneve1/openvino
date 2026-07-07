# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Drive the 1:1 PA dispatcher over a single sequence and print its trace.

This script plays the continuous-batching pipeline's role -- it produces the
PA control tensors the way ``ModelRunner::forward`` does for one sequence --
so the dispatcher has real dispatches to pass through and trace:

  * prefill: the whole prompt in ONE call (no chunking in this story),
  * decode: N greedy steps of one token each.

Usage::

    python run_dispatch.py <pa_model.xml> [--prompt-len 37] [--decode-steps 4]
"""

import argparse
import sys

import numpy as np
import openvino as ov

from pa_dispatcher import PADispatcher, is_paged_attention_model, to_paged_attention


def pa_inputs(tokens: np.ndarray, past_len: int, cache, geometry):
    """Single-sequence PA control tensors for one dispatch."""
    n = tokens.shape[0]
    ctx_after = past_len + n
    nblocks = -(-ctx_after // geometry.block_size)  # ceil
    inputs = {
        "input_ids": tokens.astype(np.int64),
        "position_ids": np.arange(past_len, ctx_after, dtype=np.int64),
        "past_lens": np.array([past_len], dtype=np.int32),
        "subsequence_begins": np.array([0, n], dtype=np.int32),
        "block_indices": np.arange(nblocks, dtype=np.int32),
        "block_indices_begins": np.array([0, nblocks], dtype=np.int32),
        "max_context_len": np.array(ctx_after, dtype=np.int32),
    }
    inputs.update(cache)
    return inputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="PA IR (.xml); see convert_pa_model.py")
    parser.add_argument("--prompt-len", type=int, default=37,
                        help="synthetic prompt length in tokens")
    parser.add_argument("--decode-steps", type=int, default=4,
                        help="greedy decode steps after prefill")
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--kv-cache-precision", default="bf16",
                        help="KV_CACHE_PRECISION compile property ('' to omit)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    core = ov.Core()
    model = core.read_model(args.model)
    if not is_paged_attention_model(model):
        # A saved PA IR does not round-trip (PagedAttentionExtension has no
        # opset), so accept a stateful IR and transform in memory.
        print("input is a stateful IR; applying paged_attention_transformation")
        to_paged_attention(model)
    config = ({"KV_CACHE_PRECISION": args.kv_cache_precision}
              if args.kv_cache_precision else {})
    dispatcher = PADispatcher(model, core, device=args.device, config=config)

    geo = dispatcher.kv
    total = args.prompt_len + args.decode_steps
    num_blocks = -(-total // geo.block_size) + 1
    cache = geo.make_cache_tensors(num_blocks)

    rng = np.random.default_rng(args.seed)
    prompt = rng.integers(0, 1000, size=args.prompt_len)

    # Prefill: the whole prompt, one 1:1 dispatch.
    logits = dispatcher.infer(pa_inputs(prompt, past_len=0, cache=cache, geometry=geo))
    generated = [int(logits.reshape(-1, logits.shape[-1])[-1].argmax())]

    # Decode: one token per dispatch.
    for step in range(args.decode_steps - 1):
        past = args.prompt_len + step
        logits = dispatcher.infer(pa_inputs(np.array([generated[-1]]), past_len=past,
                                            cache=cache, geometry=geo))
        generated.append(int(logits.reshape(-1, logits.shape[-1])[-1].argmax()))

    print(f"\nprompt ({args.prompt_len} tokens, seed {args.seed}): "
          f"{prompt[:8].tolist()}{'...' if args.prompt_len > 8 else ''}")
    print(f"greedy continuation: {generated}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
