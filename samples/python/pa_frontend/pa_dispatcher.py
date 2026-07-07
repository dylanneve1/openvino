# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""NPUW PA front-end -- Stage 0: 1:1 dispatch + model-expectation trace.

``PADispatcher`` stands between a caller and a fully dynamic, stateless,
PagedAttention (PA) model -- the model OpenVINO GenAI's continuous-batching
pipeline hands to a device. Scope of this story (parent epic CVS-190137):

  * Dispatch 1:1 to the underlying dynamic-shape model: the model is compiled
    untouched (no clone, no reshape, no derived variants) and every ``infer``
    is a straight pass-through to a single ``InferRequest``.
  * Trace the model expectations: dump the declared and compiled input/output
    signature, and on every dispatch log + verify the invariants the PA
    contract imposes on the control tensors.

No chunking, no derived semi-static models, no intel_npu / NPUW plugin
changes. Those are follow-up stories.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Mapping, Optional, Union

import numpy as np
import openvino as ov

try:  # exposed by the OpenVINO Python bindings
    from openvino._offline_transformations import paged_attention_transformation
except Exception:  # pragma: no cover - only needed by the conversion helper
    paged_attention_transformation = None


# Inputs the SDPAToPagedAttention pass adds next to the token inputs.
PA_CONTROL_INPUTS = (
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


class KVGeometry:
    """KV cache geometry as fixed by the compiled model / device plugin."""

    def __init__(self, compiled: ov.CompiledModel):
        key_names: List[str] = []
        value_names: List[str] = []
        ref = None
        for port in compiled.inputs:
            for name in port.get_names():
                if name.startswith("key_cache."):
                    key_names.append(name)
                    ref = ref or port
                elif name.startswith("value_cache."):
                    value_names.append(name)
        if not key_names:
            raise ValueError("compiled model has no key_cache.* inputs -- not a PA model")

        self.key_names = sorted(key_names, key=lambda n: int(n.split(".")[1]))
        self.value_names = sorted(value_names, key=lambda n: int(n.split(".")[1]))
        self.num_layers = len(self.key_names)

        shape = ref.get_partial_shape()  # [num_blocks, kv_heads, block_size, head_size]
        self.kv_heads = shape[1].get_length()
        self.block_size = shape[2].get_length()
        self.head_size = shape[3].get_length()
        self.element_type = ref.get_element_type()

    def make_cache_tensors(self, num_blocks: int) -> Dict[str, ov.Tensor]:
        """One zero-initialised tensor per key_cache.N / value_cache.N."""
        tensors: Dict[str, ov.Tensor] = {}
        for name in self.key_names + self.value_names:
            t = ov.Tensor(self.element_type,
                          [num_blocks, self.kv_heads, self.block_size, self.head_size])
            t.data[:] = 0
            tensors[name] = t
        return tensors


TensorLike = Union[ov.Tensor, np.ndarray]


class PADispatcher:
    """1:1 dispatcher over a dynamic PA model, tracing what the model expects.

    Parameters
    ----------
    model:
        The dynamic, stateless PA ``ov.Model`` (post ``paged_attention_transformation``).
    core:
        An ``ov.Core`` instance.
    device:
        Underlying device. This story is CPU only.
    config:
        Extra compile properties (e.g. ``{"KV_CACHE_PRECISION": "bf16"}``).
    log:
        Sink for the trace lines. ``None`` disables tracing.
    strict:
        If True (default) an expectation violation raises; otherwise it is
        logged and the dispatch proceeds (useful for probing the model).
    """

    def __init__(
        self,
        model: ov.Model,
        core: ov.Core,
        device: str = "CPU",
        config: Optional[Mapping[str, str]] = None,
        log: Optional[Callable[[str], None]] = print,
        strict: bool = True,
    ):
        if not is_paged_attention_model(model):
            raise ValueError(
                "model is not a PA model -- run to_paged_attention() first")
        self._log = log or (lambda _msg: None)
        self._strict = strict
        self._infer_count = 0

        self._trace_signature("declared (ov.Model)", model.inputs, model.outputs)

        # The 1:1 part: the dynamic model is compiled untouched.
        self.compiled = core.compile_model(model, device, dict(config or {}))
        self.kv = KVGeometry(self.compiled)

        self._trace_signature(f"compiled ({device})", self.compiled.inputs,
                              self.compiled.outputs)
        self._log(
            f"[pa-trace] KV geometry fixed by {device}: {self.kv.num_layers} layers, "
            f"[num_blocks(dyn), {self.kv.kv_heads}, {self.kv.block_size}, "
            f"{self.kv.head_size}] {self.kv.element_type.get_type_name()} "
            f"(block_size={self.kv.block_size})")

        self.request = self.compiled.create_infer_request()

        self._required = {p.get_any_name() for p in self.compiled.inputs}
        self._token_input = "input_ids" if "input_ids" in self._required else "inputs_embeds"

    # -- tracing ---------------------------------------------------------

    def _trace_signature(self, tag: str, inputs, outputs) -> None:
        self._log(f"[pa-trace] signature, {tag}:")
        kv_seen = 0
        for port in inputs:
            name = port.get_any_name()
            if name.startswith(("key_cache.", "value_cache.")):
                kv_seen += 1
                if kv_seen > 2:
                    continue
            self._log(f"[pa-trace]   in  {name:<22} "
                      f"{port.get_element_type().get_type_name():<5} "
                      f"{port.get_partial_shape()}")
        if kv_seen > 2:
            self._log(f"[pa-trace]   in  ... {kv_seen} key_cache/value_cache inputs "
                      f"total ({kv_seen // 2} layers)")
        for port in outputs:
            self._log(f"[pa-trace]   out {port.get_any_name():<22} "
                      f"{port.get_element_type().get_type_name():<5} "
                      f"{port.get_partial_shape()}")

    def _expect(self, cond: bool, what: str, violations: List[str]) -> None:
        if not cond:
            violations.append(what)

    def _check_expectations(self, tensors: Dict[str, ov.Tensor]) -> None:
        """Verify + trace the invariants the PA contract imposes on one dispatch."""
        v: List[str] = []
        self._expect(self._required <= set(tensors),
                     f"missing inputs: {sorted(self._required - set(tensors))}", v)
        if v:
            self._fail_or_warn(v)
            return

        tok = tensors[self._token_input].data
        pos = tensors["position_ids"].data
        past = tensors["past_lens"].data
        sub = tensors["subsequence_begins"].data
        bi = tensors["block_indices"].data
        bib = tensors["block_indices_begins"].data
        mcl = int(tensors["max_context_len"].data)

        n_tokens = tok.shape[-1] if self._token_input == "input_ids" else tok.shape[0]
        n_seqs = past.shape[0]

        # Flat token dimension: every per-token input agrees on its length.
        self._expect(pos.shape[-1] == n_tokens,
                     f"position_ids last dim {pos.shape[-1]} != tokens {n_tokens}", v)
        # subsequence_begins is a prefix-sum over the flat token dim.
        self._expect(sub.shape[0] == n_seqs + 1,
                     f"subsequence_begins len {sub.shape[0]} != past_lens+1 {n_seqs + 1}", v)
        self._expect(sub[0] == 0, f"subsequence_begins[0] = {sub[0]}, expected 0", v)
        self._expect((np.diff(sub) > 0).all(), "subsequence_begins not strictly increasing", v)
        self._expect(sub[-1] == n_tokens,
                     f"subsequence_begins[-1] = {sub[-1]} != tokens {n_tokens}", v)
        # block_indices_begins delimits per-subsequence block runs the same way.
        self._expect(bib.shape[0] == n_seqs + 1,
                     f"block_indices_begins len {bib.shape[0]} != past_lens+1 {n_seqs + 1}", v)
        self._expect(bib[0] == 0 and (np.diff(bib) >= 0).all() and bib[-1] == bi.shape[0],
                     "block_indices_begins is not a prefix-sum over block_indices", v)
        self._expect((past >= 0).all(), f"negative past_lens: {past}", v)

        # Per-subsequence: provided blocks must cover past + scheduled tokens,
        # and max_context_len bounds every context.
        kinds = []
        for s in range(n_seqs):
            scheduled = int(sub[s + 1] - sub[s])
            ctx_after = int(past[s]) + scheduled
            nblocks = int(bib[s + 1] - bib[s])
            self._expect(nblocks * self.kv.block_size >= ctx_after,
                         f"seq {s}: {nblocks} blocks x {self.kv.block_size} "
                         f"< context {ctx_after}", v)
            self._expect(mcl >= ctx_after,
                         f"seq {s}: max_context_len {mcl} < context {ctx_after}", v)
            kinds.append("prefill" if past[s] == 0 else
                         ("decode" if scheduled == 1 else "chunked-continue"))
        num_blocks_bound = None
        for name in self.kv.key_names:
            num_blocks_bound = tensors[name].shape[0]
            break
        if bi.size and num_blocks_bound is not None:
            self._expect(int(bi.max()) < num_blocks_bound,
                         f"block index {int(bi.max())} out of pool ({num_blocks_bound})", v)

        self._log(f"[pa-trace] infer #{self._infer_count}: "
                  f"{n_seqs} subsequence(s) [{', '.join(kinds)}], {n_tokens} token(s)")
        self._log(f"[pa-trace]   {self._token_input:<22} {list(tok.shape)}")
        self._log(f"[pa-trace]   position_ids           {list(pos.shape)} "
                  f"[{pos.reshape(-1)[0]}..{pos.reshape(-1)[-1]}]")
        self._log(f"[pa-trace]   past_lens              {past.tolist()}")
        self._log(f"[pa-trace]   subsequence_begins     {sub.tolist()}")
        self._log(f"[pa-trace]   block_indices          {self._abbrev(bi)}")
        self._log(f"[pa-trace]   block_indices_begins   {bib.tolist()}")
        self._log(f"[pa-trace]   max_context_len        {mcl}")
        if v:
            self._fail_or_warn(v)
        else:
            self._log("[pa-trace]   expectations           OK")

    @staticmethod
    def _abbrev(arr: np.ndarray, limit: int = 8) -> str:
        vals = arr.tolist()
        if len(vals) <= limit:
            return str(vals)
        return f"[{', '.join(map(str, vals[:limit]))}, ...] ({len(vals)} entries)"

    def _fail_or_warn(self, violations: List[str]) -> None:
        for what in violations:
            self._log(f"[pa-trace]   EXPECTATION VIOLATED: {what}")
        if self._strict:
            raise RuntimeError("PA model expectations violated: "
                               + "; ".join(violations))

    # -- dispatch ----------------------------------------------------------

    @staticmethod
    def _as_tensor(val: TensorLike) -> ov.Tensor:
        if isinstance(val, ov.Tensor):
            return val
        arr = np.asarray(val)
        # NB: np.ascontiguousarray promotes 0-d to 1-d, which breaks the
        # scalar max_context_len input -- only apply it to non-scalars.
        if arr.ndim and not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        return ov.Tensor(arr)

    def infer(self, inputs: Mapping[str, TensorLike]) -> np.ndarray:
        """Dispatch one call 1:1 to the underlying model; return its logits."""
        tensors = {name: self._as_tensor(val) for name, val in inputs.items()}
        self._check_expectations(tensors)
        for name, tensor in tensors.items():
            self.request.set_tensor(name, tensor)
        self.request.infer()
        self._infer_count += 1
        # logits buffer is reused by the next infer -- copy out.
        return self.request.get_tensor("logits").data.copy()
