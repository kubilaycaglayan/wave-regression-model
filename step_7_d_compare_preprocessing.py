#!/usr/bin/env python3
"""Compare Step 2 segmentation resolutions using existing labeled Step 1 outputs."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import step_1_a_water_segmentation as segmentation
import step_2_a_reduce_black_water_area as step2
import step_7_a_predict as prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, default=Path("step-3-dataset-splits/validation.csv"))
    parser.add_argument("--step1-dir", type=Path, default=Path("step-1-processed-data"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--candidate-resolution", type=int, default=384)
    parser.add_argument("--step2-resolution", type=int, help="experimental Step 2 resolution for raw candidates")
    parser.add_argument("--precision", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--logits-resolution", type=int, help="experimental interpolation size before choosing single-image classes")
    parser.add_argument("--max-segmentation-side", type=int, help="experiment with tiled segmentation on a downscaled source")
    parser.add_argument("--tile-size", type=int, default=768, help="experimental tile size when the source is downscaled")
    parser.add_argument("--raw-candidate", action="store_true", help="compare single-image raw preprocessing with persisted Step 2 validation outputs")
    parser.add_argument("--reuse-step1-mask", action="store_true", help="experiment with projected Step 1 mask instead of Step 2 inference (requires --raw-candidate)")
    parser.add_argument("--step2-dir", type=Path, default=Path("step-2-final-water-data"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.threads < 1 or args.candidate_resolution < 1 or (args.logits_resolution is not None and args.logits_resolution < 1):
        parser.error("threads and candidate resolution must be positive")
    if args.reuse_step1_mask and not args.raw_candidate:
        parser.error("--reuse-step1-mask requires --raw-candidate")
    if args.logits_resolution is not None and not args.raw_candidate:
        parser.error("--logits-resolution requires --raw-candidate")
    if args.step2_resolution is not None and (not args.raw_candidate or args.step2_resolution < 1 or args.reuse_step1_mask):
        parser.error("--step2-resolution requires --raw-candidate, a positive size, and a Step 2 pass")
    if args.max_segmentation_side is not None and (not args.raw_candidate or args.max_segmentation_side < 1 or args.tile_size < 1 or args.logits_resolution is not None):
        parser.error("--max-segmentation-side requires --raw-candidate, positive sizes, and no --logits-resolution")
    if args.output.exists():
        parser.error("output already exists")
    torch.set_num_threads(args.threads)
    device = segmentation.configure_inference_device(args.device)
    segmenter = segmentation.load_model(str(device), verbose=False)
    if args.step2_resolution is not None:
        original_step2 = prediction.preprocess_step_1_image
        def experimental_step2(image, selected):
            previous = selected.processor.size
            selected.processor.size = {"height": args.step2_resolution, "width": args.step2_resolution}
            try:
                return original_step2(image, selected)
            finally:
                selected.processor.size = previous
        prediction.preprocess_step_1_image = experimental_step2
    checkpoint = prediction.checkpoint_path_from_env()
    model = prediction.WaveRegressionModel.load_from_checkpoint(checkpoint, map_location=device, pretrained=False)
    model.to(device).eval()
    transform = prediction.build_evaluation_transform()
    with args.validation.open(newline="", encoding="utf-8") as handle:
        labels = list(csv.DictReader(handle))
    if args.raw_candidate:
        manifest = json.loads((args.step1_dir / ".source-manifest.json").read_text(encoding="utf-8"))
        source_by_stem = {stem: Path(path) for path, stem in manifest.items()}
    rows = []
    for index, label in enumerate(labels, 1):
        name = label["filename"]
        if not name.startswith("step-2_") or Path(name).name != name:
            raise ValueError(f"Unexpected validation filename: {name}")
        path = args.step1_dir / ("step-1_" + name[len("step-2_"):])
        row = {"filename": name, "label": float(label["waviness"]), "variants": {}}
        inputs = {}
        for variant, resolution in (("baseline", segmentation.SEGMENTATION_RESOLUTION),
                                    ("candidate", args.candidate_resolution)):
            started = time.perf_counter()
            try:
                if args.raw_candidate:
                    if variant == "baseline":
                        with Image.open(args.step2_dir / name) as opened:
                            model_input = opened.convert("RGB")
                    else:
                        source = source_by_stem[Path(name).stem.removeprefix("step-2_")]
                        segmentation.USE_OVERLAPPING_TILES = args.max_segmentation_side is not None
                        segmentation.SINGLE_IMAGE_LOGITS_RESOLUTION = args.logits_resolution
                        segmentation.INFERENCE_PRECISION = args.precision
                        segmentation.MAX_SEGMENTATION_SIDE = args.max_segmentation_side
                        segmentation.TILE_SIZE = args.tile_size
                        segmenter.processor.size = {"height": resolution, "width": resolution}
                        model_input, _ = prediction.preprocess_photo(source, segmenter, reuse_step1_mask=args.reuse_step1_mask)
                    value, _ = prediction.predict(model_input, device, model, transform)
                    inputs[variant] = np.asarray(model_input, dtype=np.uint8).copy()
                    row["variants"][variant] = {"prediction": value, "seconds": time.perf_counter() - started}
                    continue
                with Image.open(path) as opened:
                    image = opened.convert("RGB")
                segmenter.processor.size = {"height": resolution, "width": resolution}
                result = step2.preprocess_step_1_image(image, segmenter)
                if result is None:
                    raise ValueError("no reliable lower-water crop")
                model_input = prediction.decode_rgb_bytes(step2.encode_step_2_output(result.image))
                value, _ = prediction.predict(model_input, device, model, transform)
                inputs[variant] = np.asarray(model_input, dtype=np.uint8).copy()
                row["variants"][variant] = {"prediction": value, "crop": result.crop.box,
                                            "water_pixels": int(result.water_mask.sum()),
                                            "seconds": time.perf_counter() - started}
            except Exception as error:
                row["variants"][variant] = {"error": f"{type(error).__name__}: {error}",
                                            "seconds": time.perf_counter() - started}
        for variant, image in inputs.items():
            row["variants"][variant]["black_fraction"] = float(np.mean(np.all(image <= step2.BLACK_THRESHOLD, axis=2)))
        if set(inputs) == {"baseline", "candidate"} and inputs["baseline"].shape == inputs["candidate"].shape:
            row["mean_absolute_rgb_difference"] = float(np.mean(np.abs(inputs["baseline"].astype(np.int16) - inputs["candidate"].astype(np.int16))))
        rows.append(row)
        print(f"{index}/{len(labels)} {name}: {row['variants']}", flush=True)
    comparable = [r for r in rows if all("prediction" in r["variants"][v] for v in ("baseline", "candidate"))]
    metrics = {"total": len(rows), "comparable": len(comparable)}
    for variant in ("baseline", "candidate"):
        metrics[variant + "_failures"] = sum("error" in row["variants"][variant] for row in rows)
        metrics[variant + "_mae"] = (float(np.mean([abs(r["variants"][variant]["prediction"] - r["label"]) for r in comparable]))
                                      if comparable else None)
    metrics["mean_absolute_prediction_shift"] = (float(np.mean([abs(r["variants"]["candidate"]["prediction"] - r["variants"]["baseline"]["prediction"]) for r in comparable]))
                                                   if comparable else None)
    shifts = sorted(abs(r["variants"]["candidate"]["prediction"] - r["variants"]["baseline"]["prediction"])
                    for r in comparable)
    metrics["p95_absolute_prediction_shift"] = shifts[int(np.ceil(.95 * len(shifts))) - 1] if shifts else None
    metrics["max_absolute_prediction_shift"] = shifts[-1] if shifts else None
    pixel_differences = [r["mean_absolute_rgb_difference"] for r in rows if "mean_absolute_rgb_difference" in r]
    metrics["mean_absolute_rgb_difference"] = float(np.mean(pixel_differences)) if pixel_differences else None
    if not args.raw_candidate:
        metrics["crop_box_changed"] = sum(r["variants"]["baseline"]["crop"] != r["variants"]["candidate"]["crop"] for r in comparable)
    result = {"checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": prediction.sha256_for(checkpoint),
              "validation": str(args.validation.resolve()), "device_requested": str(device),
              "segmentation_device_actual": str(segmenter.device),
              "baseline_resolution": None if args.raw_candidate else segmentation.SEGMENTATION_RESOLUTION,
              "baseline_source": "persisted Step 2 validation image" if args.raw_candidate else "Step 1 image reprocessed at baseline resolution",
              "candidate_resolution": args.candidate_resolution, "raw_candidate": args.raw_candidate,
              "step2_resolution": args.step2_resolution, "precision": args.precision,
              "logits_resolution": args.logits_resolution,
              "max_segmentation_side": args.max_segmentation_side, "tile_size": args.tile_size,
              "reuse_step1_mask": args.reuse_step1_mask,
              "metrics": metrics, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
