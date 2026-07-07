# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Produce a PA IR from a stateful text-generation IR.

Applies the same ``paged_attention_transformation`` OpenVINO GenAI's
continuous-batching pipeline uses (``prepare_model_for_paged_attention``), so
the resulting model carries the exact input contract the dispatcher targets.

Usage::

    python convert_pa_model.py <src_model.xml> <dst_pa_model.xml>

``<src_model.xml>`` is a stateful export (e.g. produced by
``optimum-cli export openvino ...``) that still has attention_mask / beam_idx
inputs and ReadValue/Assign KV state.
"""

import argparse
import sys

import openvino as ov

from pa_dispatcher import is_paged_attention_model, to_paged_attention


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", help="source stateful IR (.xml)")
    parser.add_argument("dst", help="destination PA IR (.xml)")
    args = parser.parse_args()

    core = ov.Core()
    model = core.read_model(args.src)

    if is_paged_attention_model(model):
        print(f"{args.src} is already a PA model; copying signature.")
    else:
        print("Applying paged_attention_transformation ...")
        to_paged_attention(model)

    print("\nPA model inputs:")
    kv_layers = 0
    for port in model.inputs:
        names = list(port.get_names())
        if any(n.startswith(("key_cache.", "value_cache.")) for n in names):
            kv_layers += 1
            if kv_layers <= 2:
                print(f"  {names}  {port.get_element_type().get_type_name()}  {port.get_partial_shape()}")
            continue
        print(f"  {names}  {port.get_element_type().get_type_name()}  {port.get_partial_shape()}")
    print(f"  ... {kv_layers} key_cache/value_cache inputs total "
          f"({kv_layers // 2} decoder layers)")

    print("\nOutputs:")
    for port in model.outputs:
        print(f"  {list(port.get_names())}  {port.get_element_type().get_type_name()}  {port.get_partial_shape()}")

    ov.save_model(model, args.dst)
    print(f"\nSaved PA model to {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
