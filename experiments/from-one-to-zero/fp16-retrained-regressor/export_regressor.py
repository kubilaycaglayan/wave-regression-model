#!/usr/bin/env python3
"""Export the selected experiment checkpoint as a CPU ONNX regressor."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from step_5_a_wave_regression_model import WaveRegressionModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    model = WaveRegressionModel.load_from_checkpoint(args.checkpoint, pretrained=False, map_location="cpu")
    model.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, (torch.zeros((1, 3, 224, 224), dtype=torch.float32),), str(args.output),
        input_names=["images"], output_names=["scores"],
        dynamic_axes={"images": {0: "batch"}, "scores": {0: "batch"}},
        opset_version=17, do_constant_folding=True, dynamo=False)
    import onnx
    graph = onnx.load(str(args.output))
    onnx.checker.check_model(graph)
    print(f"Exported ONNX regressor {args.output} ({args.output.stat().st_size:,} bytes)", flush=True)


if __name__ == "__main__":
    main()
