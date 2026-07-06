# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""NPUW Paged-Attention front-end -- Stage 0 functional PoC (standalone, CPU only).

This module implements ``PagedFrontEnd``: a graph-reshaping + inference
orchestration layer that runs *under* a PagedAttention (PA) model, the way
OpenVINO GenAI's continuous-batching pipeline hands one to a device.

Stage 0 scope (see drafts/stage0-paged-attention-frontend-plan.md):

  * Accept the fully dynamic, stateless, PA-based model as input.
  * Derive *semi-static* "chunk" sub-models by pinning only the flat token
    dimension (input size 1024 / 128 / 1) and keeping the KV context dynamic.
  * Dynamically dispatch an input sequence across those chunk models: break a
    prefill into exact chunks, and use the size-1 model for generation.

Everything here is public OpenVINO API -- no intel_npu / NPUW plugin changes.
The underlying device is CPU, whose PagedAttention op natively supports the
dynamic block dimensions the derived models keep.

Key facts established against the CPU PA op (Qwen3-0.6B, see README):

  * Model inputs after ``paged_attention_transformation``:
        input_ids            i64 [-1]      (flat token dim)
        position_ids         i64 [-1]      (or [3,-1] for M-RoPE)
        past_lens            i32 [-1]      per-subsequence cached length
        subsequence_begins   i32 [-1]      prefix-sum token offsets
        block_indices        i32 [-1]      physical KV block ids
        block_indices_begins i32 [-1]      per-subsequence offsets into block_indices
        max_context_len      i32 {}        scalar
        key_cache.N          <kv> [-1, kv_heads, block_size, head_size]
        value_cache.N        <kv> [-1, kv_heads, block_size, head_size]
    Output: logits f32 [tokens, 1, vocab].
  * CPU PA enforces a bf16 KV cache with block_size 32 on this hardware. Those
    are *read from the compiled model*, not hard-coded, so the PoC follows
    whatever the plugin fixes.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import openvino as ov

try:  # exposed by the OpenVINO Python bindings
    from openvino._offline_transformations import paged_attention_transformation
except Exception:  # pragma: no cover - only needed by the conversion helper
    paged_attention_transformation = None


# Names the SDPAToPagedAttention pass adds. Everything except the two token
# inputs is left dynamic in the derived chunk models ("context dynamic").
_TOKEN_INPUTS = ("input_ids", "position_ids")
_PA_CONTROL_INPUTS = (
    "past_lens",
    "subsequence_begins",
    "block_indices",
    "block_indices_begins",
    "max_context_len",
)


def is_paged_attention_model(model: ov.Model) -> bool:
    """True if *model* already carries the PA control inputs + a KV cache pair."""
    names = {n for p in model.inputs for n in p.get_names()}
    has_control = {"past_lens", "subsequence_begins", "block_indices"} <= names
    has_cache = any(n.startswith("key_cache.") for n in names)
    return has_control and has_cache


def to_paged_attention(model: ov.Model) -> ov.Model:
    """Apply the PA transform in place and return *model* (for chaining)."""
    if paged_attention_transformation is None:
        raise RuntimeError(
            "openvino._offline_transformations.paged_attention_transformation "
            "is unavailable in this OpenVINO build."
        )
    if not is_paged_attention_model(model):
        paged_attention_transformation(model)
    return model


def greedy_tiling(total: int, chunk_sizes: Sequence[int]) -> List[int]:
    """Decompose *total* tokens **exactly** into the available static sizes.

    Greedy largest-first. Because ``1`` is always in ``chunk_sizes`` any total
    decomposes with zero remainder, so every emitted chunk fully fills its
    model (real_len == S) -- no padding, no masking. Example::

        greedy_tiling(1500, (1024, 128, 1)) -> [1024, 128, 128, 128, 1, ... x92]
    """
    if total < 0:
        raise ValueError(f"total must be >= 0, got {total}")
    sizes = sorted(chunk_sizes, reverse=True)
    if sizes[-1] != 1:
        raise ValueError(f"chunk_sizes must contain 1 for exact tiling, got {chunk_sizes}")
    out: List[int] = []
    remaining = total
    for s in sizes:
        while remaining >= s:
            out.append(s)
            remaining -= s
    assert remaining == 0
    return out


class BlockAllocator:
    """Trivial monotonic KV block pool for a *single* sequence.

    Stage 0 handles one sequence, so block ``i`` always holds tokens
    ``[i*block_size, (i+1)*block_size)`` and ``block_indices`` is simply
    ``[0, 1, ..., nblocks-1]``. The pool is pre-sized; we never free.
    """

    def __init__(self, block_size: int, num_blocks: int):
        self.block_size = block_size
        self.num_blocks = num_blocks

    def blocks_for_context(self, context_len: int) -> int:
        nblocks = (context_len + self.block_size - 1) // self.block_size
        if nblocks > self.num_blocks:
            raise RuntimeError(
                f"KV block pool exhausted: need {nblocks} blocks for context "
                f"{context_len} (block_size {self.block_size}) but pool holds "
                f"{self.num_blocks}. Increase num_blocks."
            )
        return nblocks


class _KVLayout:
    """Concrete KV cache geometry read from a compiled PA model."""

    def __init__(self, compiled: ov.CompiledModel):
        key_names: List[str] = []
        val_names: List[str] = []
        ref = None
        for port in compiled.inputs:
            for name in port.get_names():
                if name.startswith("key_cache."):
                    key_names.append(name)
                    ref = ref or port
                elif name.startswith("value_cache."):
                    val_names.append(name)
        if not key_names:
            raise ValueError("compiled model has no key_cache.* inputs -- not a PA model")

        self.key_names = sorted(key_names, key=lambda n: int(n.split(".")[1]))
        self.value_names = sorted(val_names, key=lambda n: int(n.split(".")[1]))
        self.num_layers = len(self.key_names)

        shape = ref.get_partial_shape()  # [num_blocks, kv_heads, block_size, head_size]
        self.kv_heads = shape[1].get_length()
        self.block_size = shape[2].get_length()
        self.head_size = shape[3].get_length()
        self.element_type = ref.get_element_type()
        # bf16 is surfaced to numpy as float16 by the bindings; keep the numpy
        # dtype we allocate with in sync with what ov.Tensor(type, shape) exposes.
        self._np_dtype = ov.Tensor(self.element_type, [1]).data.dtype

    def make_cache_tensors(self, num_blocks: int) -> Dict[str, ov.Tensor]:
        """One zero-initialised tensor per key_cache.N / value_cache.N."""
        tensors: Dict[str, ov.Tensor] = {}
        for name in self.key_names + self.value_names:
            t = ov.Tensor(self.element_type, [num_blocks, self.kv_heads, self.block_size, self.head_size])
            t.data[:] = 0
            tensors[name] = t
        return tensors


class PagedFrontEnd:
    """Derive semi-static chunk models from a dynamic PA model and dispatch to them.

    Parameters
    ----------
    model:
        The dynamic, stateless PA ``ov.Model`` (post ``paged_attention_transformation``).
        If a plain SDPA model is passed, call :func:`to_paged_attention` first.
    core:
        An ``ov.Core`` instance.
    device:
        Underlying device for the dispatched models. Stage 0 is CPU only.
    chunk_sizes:
        Static flat-token sizes to derive. Must contain ``1``. Default ``(1024, 128, 1)``.
    num_blocks:
        Size of the self-managed KV block pool (in blocks).
    kv_cache_precision:
        Forwarded as ``KV_CACHE_PRECISION``. CPU PA on current hardware requires
        ``"bf16"``; kept configurable for other back-ends.
    """

    def __init__(
        self,
        model: ov.Model,
        core: ov.Core,
        device: str = "CPU",
        chunk_sizes: Sequence[int] = (1024, 128, 1),
        num_blocks: int = 512,
        kv_cache_precision: str = "bf16",
    ):
        if not is_paged_attention_model(model):
            raise ValueError(
                "model is not a PagedAttention model -- run to_paged_attention() first"
            )
        if 1 not in chunk_sizes:
            raise ValueError("chunk_sizes must contain 1 (needed for exact tiling)")

        self.core = core
        self.device = device
        self.chunk_sizes = tuple(sorted(set(chunk_sizes), reverse=True))
        self._compile_config = {"KV_CACHE_PRECISION": kv_cache_precision} if kv_cache_precision else {}

        # 1) Derive one semi-static, context-dynamic variant per chunk size.
        self._requests: Dict[int, ov.InferRequest] = {}
        layout: Optional[_KVLayout] = None
        for size in self.chunk_sizes:
            compiled = self._compile_variant(model, size)
            if layout is None:
                layout = _KVLayout(compiled)
            self._requests[size] = compiled.create_infer_request()
        assert layout is not None
        self.layout = layout

        # 2) Allocate the shared KV cache and bind the SAME tensors to every
        #    sub-request so all chunks read/write one pool.
        self.allocator = BlockAllocator(layout.block_size, num_blocks)
        self._cache = layout.make_cache_tensors(num_blocks)
        for req in self._requests.values():
            for name, tensor in self._cache.items():
                req.set_tensor(name, tensor)

        self.context_len = 0

    # -- construction helpers ------------------------------------------------

    def _compile_variant(self, model: ov.Model, size: int) -> ov.CompiledModel:
        """Clone and pin only the token dim to *size*; keep everything else dynamic."""
        variant = model.clone()
        reshape_map = {}
        for port in variant.inputs:
            names = port.get_names()
            if any(n in _TOKEN_INPUTS for n in names):
                ps = port.get_partial_shape()
                # Pin the *last* (flat token) dim; preserve leading dims such as
                # the 3 of M-RoPE position_ids [3, -1].
                new_dims = list(ps)
                new_dims[-1] = ov.Dimension(size)
                reshape_map[port.get_any_name()] = ov.PartialShape(new_dims)
        variant.reshape(reshape_map)
        return self.core.compile_model(variant, self.device, self._compile_config)

    # -- runtime dispatch ----------------------------------------------------

    def reset(self) -> None:
        """Clear sequence state and zero the KV pool for a fresh sequence."""
        self.context_len = 0
        for tensor in self._cache.values():
            tensor.data[:] = 0

    def _run_chunk(
        self,
        request: ov.InferRequest,
        input_ids: np.ndarray,
        position_ids: np.ndarray,
        past_len: int,
    ) -> np.ndarray:
        """Run one sub-request over a fully-filled chunk; return its logits."""
        size = input_ids.shape[0]
        context_after = past_len + size
        nblocks = self.allocator.blocks_for_context(context_after)

        request.set_tensor("input_ids", ov.Tensor(np.ascontiguousarray(input_ids, dtype=np.int64)))
        request.set_tensor("position_ids", ov.Tensor(np.ascontiguousarray(position_ids, dtype=np.int64)))
        request.set_tensor("past_lens", ov.Tensor(np.array([past_len], dtype=np.int32)))
        request.set_tensor("subsequence_begins", ov.Tensor(np.array([0, size], dtype=np.int32)))
        request.set_tensor("block_indices", ov.Tensor(np.arange(nblocks, dtype=np.int32)))
        request.set_tensor("block_indices_begins", ov.Tensor(np.array([0, nblocks], dtype=np.int32)))
        request.set_tensor("max_context_len", ov.Tensor(np.array(context_after, dtype=np.int32)))

        request.infer()
        # logits: [tokens, 1, vocab]; copy out (buffer is reused next infer).
        return request.get_tensor("logits").data.copy()

    def prefill(self, token_ids: Sequence[int]) -> np.ndarray:
        """Process a prompt from the current context, exact-tiling it into chunks.

        Returns the last-token logits (shape ``[vocab]``). Advances ``context_len``.
        """
        tokens = np.asarray(token_ids, dtype=np.int64).reshape(-1)
        total = tokens.shape[0]
        if total == 0:
            raise ValueError("prefill requires at least one token")

        last_logits: Optional[np.ndarray] = None
        offset = 0
        for size in greedy_tiling(total, self.chunk_sizes):
            chunk = tokens[offset : offset + size]
            positions = np.arange(self.context_len + offset,
                                  self.context_len + offset + size, dtype=np.int64)
            logits = self._run_chunk(self._requests[size], chunk, positions,
                                     past_len=self.context_len + offset)
            last_logits = logits
            offset += size

        self.context_len += total
        assert last_logits is not None
        return last_logits.reshape(-1, last_logits.shape[-1])[-1]

    def generate_step(self, token_id: int) -> np.ndarray:
        """Run the size-1 model for one decode step. Returns logits ``[vocab]``."""
        req = self._requests[1]
        logits = self._run_chunk(
            req,
            np.array([token_id], dtype=np.int64),
            np.array([self.context_len], dtype=np.int64),
            past_len=self.context_len,
        )
        self.context_len += 1
        return logits.reshape(-1, logits.shape[-1])[-1]
