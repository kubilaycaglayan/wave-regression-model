#!/usr/bin/env python3
"""Create a fixed 384x384 SegFormer export from the 448x448 graph constants."""
from __future__ import annotations

import sys
from pathlib import Path

import onnx
from onnx import numpy_helper


def main() -> None:
    source, destination = map(Path, sys.argv[1:3])
    model = onnx.load(str(source), load_external_data=False)
    mapping = {448: 384, 112: 96, 56: 48, 28: 24, 14: 12,
               12544: 9216, 3136: 2304, 784: 576, 196: 144}
    changed = 0
    for node in model.graph.node:
        for attribute in node.attribute:
            if attribute.type != onnx.AttributeProto.TENSOR:
                continue
            tensor = attribute.t
            values = numpy_helper.to_array(tensor)
            if values.dtype.kind not in "iu":
                continue
            flat = values.reshape(-1).copy()
            old = flat.copy()
            for before, after in mapping.items():
                flat[old == before] = after
            if not (flat == old).all():
                attribute.t.CopyFrom(numpy_helper.from_array(flat.reshape(values.shape)))
                changed += 1
    for value in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        for dim in value.type.tensor_type.shape.dim:
            if dim.dim_value == 448:
                dim.dim_value = 384
    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(destination))
    print(f"Wrote {destination}; rewrote {changed} shape constants")


if __name__ == "__main__":
    main()
