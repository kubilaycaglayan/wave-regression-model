#!/usr/bin/env python3
"""Evaluate a frozen-backbone head adapted to fast Step 7 preprocessing.

Only train and validation splits are read. Cached features are vectors, not
images; checkpoint and preprocessing settings are never changed in place.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import step_1_a_water_segmentation as segmentation
import step_7_a_predict as prediction
from step_7_g_checkpoint_contract import previous_runner_hash


def read_split(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["filename", "waviness"]:
            raise ValueError(f"Invalid split manifest: {path}")
        rows = list(reader)
    names = [row["filename"] for row in rows]
    if not rows or len(set(names)) != len(names):
        raise ValueError(f"Empty split or duplicate filenames: {path}")
    for row in rows:
        if Path(row["filename"]).name != row["filename"] or not row["filename"].startswith("step-2_"):
            raise ValueError(f"Unexpected split filename: {row['filename']}")
        if not 0 <= float(row["waviness"]) <= 1:
            raise ValueError(f"Invalid label for {row['filename']}")
    return rows


def capture_day(source: Path) -> str:
    """Read a capture day for group isolation; fail instead of guessing."""
    with Image.open(source) as image:
        exif = image.getexif()
        try:
            extra = exif.get_ifd(34665)
        except (AttributeError, KeyError, TypeError):
            extra = {}
        for raw in (extra.get(36867), exif.get(36867), extra.get(36868), exif.get(36868), exif.get(306)):
            if not raw:
                continue
            value = raw.decode("ascii", errors="ignore") if isinstance(raw, bytes) else str(raw)
            try:
                return datetime.strptime(value.strip()[:19], "%Y:%m:%d %H:%M:%S").date().isoformat()
            except ValueError:
                try:
                    return datetime.fromisoformat(value.strip().replace("Z", "+00:00")).date().isoformat()
                except ValueError:
                    continue
    raise ValueError(f"No usable capture date for {source}")


def date_holdout_indices(days: list[str], seed: int = 42):
    """Hold out whole capture days, targeting 25% of training examples."""
    groups = sorted(set(days))
    if len(groups) < 2:
        raise ValueError("Date holdout requires at least two distinct capture days")
    random.Random(seed).shuffle(groups)
    target = max(1, round(len(days) * .25))
    selected = set()
    count = 0
    for day in groups[:-1]:
        selected.add(day)
        count += days.count(day)
        if count >= target:
            break
    holdout = torch.tensor([i for i, day in enumerate(days) if day in selected], dtype=torch.long)
    fit = torch.tensor([i for i, day in enumerate(days) if day not in selected], dtype=torch.long)
    return fit, holdout


def metrics(values: torch.Tensor, labels: torch.Tensor, baseline: torch.Tensor) -> dict[str, float]:
    error = (values - labels).abs()
    shift = (values - baseline).abs().sort().values
    rank = max(0, int(np.ceil(.95 * shift.numel())) - 1)
    return {"mae": float(error.mean()), "mean_shift": float(shift.mean()),
            "p95_shift": float(shift[rank]), "max_shift": float(shift[-1])}


def cache_path_for(cache_dir: Path, identity: dict) -> Path:
    return cache_dir / f"{hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()}.pt"


def validate_feature_record(record: dict, feature_dim: int, filename: str) -> None:
    if (record["features"].shape != (feature_dim,) or not torch.isfinite(record["features"]).all()
            or not math.isfinite(record["baseline"]) or not 0 <= record["baseline"] <= 1):
        raise RuntimeError(f"Invalid cached features or baseline prediction for {filename}")


def load_cached_feature(cache_dir: Path, identity: dict, label: str, compatible_identities=()):
    """Load and migrate cached vectors; labels are always read from the current manifest."""
    cache_path = cache_path_for(cache_dir, identity)
    legacy_path = cache_path_for(cache_dir, {**identity, "label": label})
    if cache_path.is_file():
        record, status = torch.load(cache_path, map_location="cpu", weights_only=True), "cached"
    elif legacy_path.is_file():
        record = torch.load(legacy_path, map_location="cpu", weights_only=True)
        temporary = cache_path.with_suffix(f".{os.getpid()}.tmp")
        torch.save(record, temporary)
        os.replace(temporary, cache_path)
        status = "cached (legacy)"
    else:
        record = None
        for previous in compatible_identities:
            record, _ = load_cached_feature(cache_dir, previous, label)
            if record is not None:
                temporary = cache_path.with_suffix(f".{os.getpid()}.tmp")
                torch.save(record, temporary)
                os.replace(temporary, cache_path)
                status = "cached (compatible runner)"
                break
        if record is None:
            return None, None
    record["label"] = float(label)
    return record, status


def fit_head(train, validation, initial_weight, initial_bias, teacher_weight, *, epochs, learning_rate, regularization):
    """Fit only the sigmoid head, retaining a penalty toward the selected checkpoint."""
    features, labels, teacher = train
    val_features, val_labels, val_teacher = validation
    weight = torch.nn.Parameter(initial_weight.clone())
    bias = torch.nn.Parameter(initial_bias.clone())
    optimizer = torch.optim.Adam([weight, bias], lr=learning_rate)
    best = None
    for epoch in range(epochs + 1):
        with torch.no_grad():
            values = torch.sigmoid(val_features @ weight + bias)
            score = metrics(values, val_labels, val_teacher)
            if best is None or score["mae"] < best["metrics"]["mae"]:
                best = {"epoch": epoch, "metrics": score, "weight": weight.detach().clone(), "bias": bias.detach().clone()}
        if epoch == epochs:
            break
        optimizer.zero_grad()
        values = torch.sigmoid(features @ weight + bias)
        loss = ((1 - teacher_weight) * (values - labels).square().mean()
                + teacher_weight * (values - teacher).square().mean()
                + regularization * ((weight - initial_weight).square().sum() + (bias - initial_bias).square()))
        loss.backward()
        optimizer.step()
    return best


def fit_head_fixed(train, initial_weight, initial_bias, teacher_weight, *, epochs, learning_rate, regularization):
    """Refit on all training examples for a previously selected epoch count."""
    features, labels, teacher = train
    weight = torch.nn.Parameter(initial_weight.clone())
    bias = torch.nn.Parameter(initial_bias.clone())
    optimizer = torch.optim.Adam([weight, bias], lr=learning_rate)
    for _ in range(epochs):
        optimizer.zero_grad()
        values = torch.sigmoid(features @ weight + bias)
        loss = ((1 - teacher_weight) * (values - labels).square().mean()
                + teacher_weight * (values - teacher).square().mean()
                + regularization * ((weight - initial_weight).square().sum() + (bias - initial_bias).square()))
        loss.backward()
        optimizer.step()
    return weight.detach().clone(), bias.detach().clone()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path, default=prediction.PROJECT_DIR / "step-7-prediction-benchmarks" / "fast-head-feature-cache",
                        help="shared feature cache; unchanged photos are reused across experiments")
    parser.add_argument("--train", type=Path, default=Path("step-3-dataset-splits/train.csv"))
    parser.add_argument("--validation", type=Path, default=Path("step-3-dataset-splits/validation.csv"))
    parser.add_argument("--step1-dir", type=Path, default=Path("step-1-processed-data"))
    parser.add_argument("--step2-dir", type=Path, default=Path("step-2-final-water-data"))
    parser.add_argument("--resolution", type=int, default=448)
    parser.add_argument("--step2-resolution", type=int, default=384)
    parser.add_argument("--precision", choices=("float32", "float16", "bfloat16"), default="float16")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--regularization", type=float, default=0.01)
    parser.add_argument("--holdout-by-date", action="store_true", help="select training settings using separate EXIF capture days")
    args = parser.parse_args()
    if min(args.resolution, args.step2_resolution, args.threads, args.epochs) < 1 or args.learning_rate <= 0 or args.regularization < 0:
        parser.error("resolutions, threads, epochs and learning rate must be positive; regularization cannot be negative")
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    splits = {"train": read_split(args.train), "validation": read_split(args.validation)}
    if len(splits["train"]) < 2:
        raise ValueError("At least two training examples are required for the training holdout")
    if {r["filename"] for r in splits["train"]} & {r["filename"] for r in splits["validation"]}:
        raise ValueError("Train and validation manifests overlap")
    source_by_stem = {stem: Path(source) for source, stem in json.loads((args.step1_dir / ".source-manifest.json").read_text()).items()}
    checkpoint = prediction.checkpoint_path_from_env()
    checkpoint_sha = prediction.sha256_for(checkpoint)
    if torch.load(checkpoint, map_location="cpu", weights_only=True).get("fast_preprocessing_experiment") is not None:
        raise ValueError("Choose the original baseline checkpoint, not an adapted fast-preprocessing checkpoint")
    device = segmentation.configure_inference_device()
    settings = {"checkpoint_sha256": checkpoint_sha, "model": segmentation.MODEL_NAME,
                "resolution": args.resolution, "step2_resolution": args.step2_resolution,
                "precision": args.precision, "device": str(device), "torch_version": str(torch.__version__),
                "logits_resolution": args.resolution,
                "code_sha256": {name: prediction.sha256_for(prediction.PROJECT_DIR / name) for name in (
                    "step_1_a_water_segmentation.py", "step_1_b_preprocess_water_inputs.py",
                    "step_2_a_reduce_black_water_area.py", "step_7_a_predict.py", "step_4_a_wave_dataset.py",
                    "step_5_a_wave_regression_model.py")}}
    segmentation.SEGMENTATION_RESOLUTION = args.resolution
    segmentation.USE_OVERLAPPING_TILES = False
    segmentation.SINGLE_IMAGE_LOGITS_RESOLUTION = args.resolution
    segmentation.INFERENCE_PRECISION = args.precision
    segmenter = segmentation.load_model(str(device), verbose=False)
    settings["segmentation_model_revision"] = getattr(segmenter.model.config, "_commit_hash", None)
    previous_hash = previous_runner_hash(prediction.PROJECT_DIR / "step_7_a_predict.py")
    compatible_settings = None
    if previous_hash is not None:
        compatible_settings = json.loads(json.dumps(settings))
        compatible_settings["code_sha256"]["step_7_a_predict.py"] = previous_hash
    model = prediction.WaveRegressionModel.load_from_checkpoint(checkpoint, map_location=device, pretrained=False).to(device).eval()
    if model.backbone_name != "efficientnet_b0":
        raise ValueError("This experiment requires a checkpoint with an EfficientNet-B0 regression head")
    transform = prediction.build_evaluation_transform()
    original_step2 = prediction.preprocess_step_1_image
    def step2_with_resolution(image, selected):
        previous = selected.processor.size
        selected.processor.size = {"height": args.step2_resolution, "width": args.step2_resolution}
        try:
            return original_step2(image, selected)
        finally:
            selected.processor.size = previous
    prediction.preprocess_step_1_image = step2_with_resolution
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite completed experiment: {result_path}")
    assembled = {}
    for split, rows in splits.items():
        records = []
        for index, row in enumerate(rows, 1):
            started = time.perf_counter()
            filename = row["filename"]
            source = source_by_stem[Path(filename).stem.removeprefix("step-2_")]
            previous = args.step2_dir / filename
            identity = {"settings": settings, "filename": filename,
                        "source_sha256": prediction.sha256_for(source), "previous_sha256": prediction.sha256_for(previous)}
            cache_path = cache_path_for(cache_dir, identity)
            # Read the original label-dependent cache format too, so existing
            # feature work can be carried forward without reprocessing photos.
            previous_identity = {**identity, "settings": compatible_settings} if compatible_settings is not None else None
            record, status = load_cached_feature(cache_dir, identity, row["waviness"],
                                                 compatible_identities=(previous_identity,) if previous_identity else ())
            if record is None:
                if segmenter.device != device:
                    raise RuntimeError("Segmentation device changed; refusing to cache inputs from a different runtime")
                image, _ = prediction.preprocess_photo(source, segmenter)
                if segmenter.device != device:
                    raise RuntimeError("Segmentation fell back to CPU; refusing to cache a mixed-device experiment")
                with Image.open(previous) as opened:
                    baseline_image = opened.convert("RGB")
                with torch.inference_mode():
                    features = model.backbone(transform(image).unsqueeze(0).to(device)).flatten().cpu()
                    baseline = model(transform(baseline_image).unsqueeze(0).to(device)).item()
                record = {"features": features, "baseline": baseline, "filename": filename}
                validate_feature_record(record, model.feature_dim, filename)
                temporary = cache_path.with_suffix(f".{os.getpid()}.tmp")
                torch.save(record, temporary)
                os.replace(temporary, cache_path)
                status = "processed"
            record["label"] = float(row["waviness"])
            validate_feature_record(record, model.feature_dim, filename)
            records.append(record)
            print(f"{split} {index}/{len(rows)} {filename}: {status} in {time.perf_counter()-started:.3f}s", flush=True)
        assembled[split] = (torch.stack([r["features"] for r in records]),
                            torch.tensor([r["label"] for r in records]),
                            torch.tensor([r["baseline"] for r in records]))
    initial_weight = model.regression_head[0].weight.detach().cpu().flatten()
    initial_bias = model.regression_head[0].bias.detach().cpu().flatten()[0]
    initial_values = torch.sigmoid(assembled["validation"][0] @ initial_weight + initial_bias)
    # Select hyperparameters inside the training split, then fit again on all
    # training examples. The validation split is evaluated only after selection.
    if args.holdout_by_date:
        days = [capture_day(source_by_stem[Path(row["filename"]).stem.removeprefix("step-2_")])
                for row in splits["train"]]
        fit_indices, holdout_indices = date_holdout_indices(days)
        holdout_count = len(holdout_indices)
    else:
        indices = torch.randperm(len(splits["train"]), generator=torch.Generator().manual_seed(42))
        holdout_count = max(1, min(len(indices) - 1, round(len(indices) * 0.25)))
        holdout_indices, fit_indices = indices[:holdout_count], indices[holdout_count:]
    fit_data = tuple(values[fit_indices] for values in assembled["train"])
    holdout_data = tuple(values[holdout_indices] for values in assembled["train"])
    arms = {}
    best = None
    for teacher_weight in (0.0, 0.5, 1.0):
        fit = fit_head(fit_data, holdout_data, initial_weight, initial_bias, teacher_weight,
                       epochs=args.epochs, learning_rate=args.learning_rate, regularization=args.regularization)
        arms[str(teacher_weight)] = {"epoch": fit["epoch"], "training_holdout": fit["metrics"]}
        if best is None or fit["metrics"]["mae"] < best["fit"]["metrics"]["mae"]:
            best = {"teacher_weight": teacher_weight, "fit": fit}
    final_weight, final_bias = fit_head_fixed(assembled["train"], initial_weight, initial_bias, best["teacher_weight"],
                                             epochs=best["fit"]["epoch"], learning_rate=args.learning_rate,
                                             regularization=args.regularization)
    final_values = torch.sigmoid(assembled["validation"][0] @ final_weight + final_bias)
    result = {"settings": settings, "checkpoint": str(checkpoint.resolve()),
              "train_manifest": str(args.train.resolve()), "validation_manifest": str(args.validation.resolve()),
              "train_manifest_sha256": prediction.sha256_for(args.train), "validation_manifest_sha256": prediction.sha256_for(args.validation),
              "feature_cache": str(cache_dir.resolve()),
              "train_count": len(splits["train"]), "validation_count": len(splits["validation"]),
              "training": {"epochs": args.epochs, "learning_rate": args.learning_rate, "regularization": args.regularization,
                           "selection": "lowest training-holdout MAE across epochs and teacher weights",
                           "holdout_seed": 42, "holdout_count": holdout_count,
                           "holdout_by_date": args.holdout_by_date, "test_split_used": False},
              "saved_baseline_validation_mae": float((assembled["validation"][2] - assembled["validation"][1]).abs().mean()),
              "unadapted_fast_validation": metrics(initial_values, assembled["validation"][1], assembled["validation"][2]),
              "adapted_fast_validation": metrics(final_values, assembled["validation"][1], assembled["validation"][2]),
              "arms": arms, "selected_teacher_weight": best["teacher_weight"], "selected_epoch": best["fit"]["epoch"]}
    # The inference-only experimental checkpoint does not replace the user's checkpoint.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["state_dict"]["regression_head.0.weight"] = final_weight.reshape_as(payload["state_dict"]["regression_head.0.weight"])
    payload["state_dict"]["regression_head.0.bias"] = final_bias.reshape_as(payload["state_dict"]["regression_head.0.bias"])
    payload.pop("optimizer_states", None)
    payload.pop("lr_schedulers", None)
    payload["fast_preprocessing_experiment"] = settings
    checkpoint_out = args.output_dir / f"{checkpoint.stem}--fast-head-{best['teacher_weight']}.ckpt"
    if checkpoint_out.exists():
        existing = torch.load(checkpoint_out, map_location="cpu", weights_only=True)
        if (existing.get("fast_preprocessing_experiment") != settings
                or not torch.equal(existing["state_dict"]["regression_head.0.weight"], payload["state_dict"]["regression_head.0.weight"])
                or not torch.equal(existing["state_dict"]["regression_head.0.bias"], payload["state_dict"]["regression_head.0.bias"])):
            raise FileExistsError(f"Refusing to overwrite a different checkpoint: {checkpoint_out}")
        print(f"Reusing matching checkpoint from an interrupted run: {checkpoint_out}", flush=True)
    else:
        checkpoint_tmp = checkpoint_out.with_suffix(f".{os.getpid()}.tmp")
        torch.save(payload, checkpoint_tmp)
        os.replace(checkpoint_tmp, checkpoint_out)
    result["experimental_checkpoint"] = str(checkpoint_out.resolve())
    result["experimental_checkpoint_sha256"] = prediction.sha256_for(checkpoint_out)
    result_tmp = result_path.with_suffix(f".{os.getpid()}.tmp")
    result_tmp.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(result_tmp, result_path)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
