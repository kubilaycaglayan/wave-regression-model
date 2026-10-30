#!/usr/bin/env python3
"""Export a trained experiment-local regressor checkpoint to ONNX."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))

from step_5_a_wave_regression_model import WaveRegressionModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = WaveRegressionModel.load_from_checkpoint(args.checkpoint, pretrained=False,
                                                     map_location="cpu")
    model.eval()
    dummy = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
    torch.onnx.export(
        model,
        (dummy,),
        str(args.output),
        input_names=["images"],
        output_names=["scores"],
        dynamic_axes={"images": {0: "batch"}, "scores": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    try:
        import onnx
    except ImportError as error:
        raise RuntimeError("Install the 'onnx' package in this training environment to validate exports") from error
    graph = onnx.load(str(args.output))
    onnx.checker.check_model(graph)
    print(f"ONNX model: {args.output} ({args.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
