#!/usr/bin/env python3
"""Benchmark the active fast-v2.2 default predictor on the latest test split."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
DEFAULT_SEGMENTER = ROOT / "models/fast-v2.1/segformer-b0-step1-448-fp32.onnx"
DEFAULT_REGRESSOR = ROOT / "models/fast-v2.2/wave-regression-efficientnet-b0-fast-v2.2.onnx"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(values: list[float]) -> dict:
    values = sorted(values)
    return {"count": len(values), "median_ms": statistics.median(values),
            "p95_ms": values[max(0, math.ceil(.95 * len(values)) - 1)],
            "min_ms": values[0], "max_ms": values[-1]}


def latest_profile_snapshot() -> tuple[str, Path, str]:
    configured_checkpoint = None
    env_file = ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() == "PREDICT_CHECKPOINT_PATH":
                configured_checkpoint = Path(value.strip().strip("\"'"))
                break
    configured_checkpoint = configured_checkpoint or Path(
        "step-5-checkpoints/v21-mae-0.0971-efficientnet_b0-fast-v2.2/"
        "wave-regression-efficientnet_b0-v21-best-val-mae-epoch=29-val_mae=0.0971.ckpt"
    )
    checkpoint = configured_checkpoint if configured_checkpoint.is_absolute() else ROOT / configured_checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Configured current checkpoint not found: {checkpoint}")
    manifests = sorted(checkpoint.parent.glob("training_manifest_*.json"))
    if not manifests:
        raise FileNotFoundError(f"No training manifest alongside current checkpoint: {checkpoint}")
    training = json.loads(manifests[-1].read_text(encoding="utf-8"))
    if training.get("preprocessing_profile") != "fast-v2.2":
        raise RuntimeError(f"Current checkpoint is not trained with fast-v2.2: {training.get('preprocessing_profile')}")
    snapshot_value = training.get("split_snapshot")
    if not snapshot_value:
        raise RuntimeError(f"Training manifest has no split snapshot: {manifests[-1]}")
    snapshot = Path(snapshot_value)
    snapshot = snapshot if snapshot.is_absolute() else ROOT / snapshot
    return str(snapshot.name), snapshot, str(checkpoint.relative_to(ROOT))


def test_rows(snapshot: Path) -> list[dict]:
    raw_by_stem: dict[str, list[Path]] = {}
    for path in (ROOT / "step-0-raw-data").rglob("*"):
        if path.is_file() and path.suffix.lower() in {".heic", ".heif", ".jpg", ".jpeg", ".png"}:
            raw_by_stem.setdefault(path.stem.casefold(), []).append(path)
    rows = []
    with (snapshot / "test.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            label_filename = row["filename"]
            stem = Path(label_filename).stem.removeprefix("step-2_")
            matches = raw_by_stem.get(stem.casefold(), [])
            if len(matches) != 1:
                raise RuntimeError(f"Expected one raw photo for {label_filename}; found {len(matches)}")
            rows.append({"photo": matches[0], "label_filename": label_filename,
                         "label": float(row["waviness"])})
    return sorted(rows, key=lambda row: row["label_filename"].casefold())


def quality_metrics(rows: list[dict]) -> dict:
    successful = [row for row in rows if row.get("status") == "ok"]
    errors = [float(row["prediction"]) - float(row["label"]) for row in successful]
    labels = [float(row["label"]) for row in successful]
    target_mean = statistics.mean(labels)
    total = sum((label - target_mean) ** 2 for label in labels)
    squared = sum(error * error for error in errors)
    return {"count": len(successful), "failures": len(rows) - len(successful),
            "mae": statistics.mean(abs(error) for error in errors),
            "rmse": math.sqrt(statistics.mean(error * error for error in errors)),
            "bias": statistics.mean(errors), "r2": 1 - squared / total if total else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--regressor", type=Path, default=DEFAULT_REGRESSOR,
                        help="ONNX regressor to measure; defaults to the shipped fast-v2.2 model")
    parser.add_argument("--output", type=Path, help="Report path; defaults to this experiment directory")
    parser.add_argument("--experiment-id", default="fast-v2.2-current-default")
    parser.add_argument("--variant", default="fast-v2.2-current-default")
    parser.add_argument("--checkpoint", type=Path,
                        help="Source PyTorch checkpoint, when the ONNX regressor was exported from one")
    args = parser.parse_args()

    snapshot_id, snapshot, checkpoint_path = latest_profile_snapshot()
    os.environ.update({"WAVE_EXPERIMENT": "fast-v2.2",
        "WAVE_PIPELINE_PROFILE": "fast-v2.2-coordinate", "WAVE_STEP1_SIZE": "448",
        "WAVE_MIGRAPHX_FP16": "1", "WAVE_STEP1_ONNX": str(DEFAULT_SEGMENTER),
        "WAVE_REGRESSOR_ONNX": str(args.regressor.resolve())})
    import step_7_i_migraphx_server as server

    setup = server.initialize_models()
    startup_ms = (time.perf_counter() - server.PROCESS_START) * 1000
    cohort = test_rows(snapshot)
    records, failures = [], []
    for index, item in enumerate(cohort, 1):
        photo = item["photo"]
        try:
            server.run_coordinate_crop_pipeline(photo)  # warm-up excluded from measured runs
            for repeat in range(1, args.repeats + 1):
                started = time.perf_counter()
                result = server.run_coordinate_crop_pipeline(photo)
                wall_ms = (time.perf_counter() - started) * 1000
                records.append({"photo": photo.name, "label_filename": item["label_filename"],
                    "label": item["label"], "repeat": repeat, "status": "ok",
                    "prediction": result["prediction"],
                    "total_raw_to_score_ms": result["total_server_ms"],
                    "server_pipeline_ms": result["total_server_ms"],
                    "client_runtime_wall_ms": wall_ms, "stages_ms": result["stages_ms"],
                    "crop": result["intermediate"]["source_crop_xyxy"]})
                print(f"{index}/{len(cohort)} {photo.name} repeat {repeat}: "
                      f"{result['total_server_ms']:.2f} ms, score={result['prediction']:.5f}", flush=True)
        except Exception as error:
            failures.append({"photo": photo.name, "label_filename": item["label_filename"],
                "label": item["label"], "error": f"{type(error).__name__}: {error}"})
            print(f"{index}/{len(cohort)} {photo.name}: FAILED {failures[-1]['error']}", flush=True)

    latencies = [float(row["server_pipeline_ms"]) for row in records]
    grouped: dict[str, list[dict]] = {}
    for row in records:
        grouped.setdefault(row["label_filename"], []).append(row)
    quality_rows = [{"status": "ok", "label": rows[0]["label"],
                     "prediction": statistics.median(row["prediction"] for row in rows)}
                    for rows in grouped.values()]
    speed = summarize(latencies) if latencies else None
    report = {"schema_version": 1, "experiment_id": args.experiment_id,
        "run_id": args.run_id, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "variant": args.variant, "runtime": "MIGraphX FP16 SegFormer + ONNX Runtime CPU regressor",
        "pipeline_profile": "fast-v2.2-coordinate",
        "pipeline": "fast-v2.2 Step 7 route: MIGraphX FP16 SegFormer and ONNX CPU regressor",
        "step1_model": str(DEFAULT_SEGMENTER), "step1_sha256": sha256(DEFAULT_SEGMENTER),
        "step1_size": 448, "fp16": True, "regressor_model": str(args.regressor.resolve()),
        "regressor_sha256": sha256(args.regressor.resolve()), "benchmark_checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "benchmark_checkpoint_sha256": sha256(args.checkpoint.resolve()) if args.checkpoint else None,
        "pytorch_checkpoint_env": checkpoint_path,
        "pytorch_checkpoint_note": ("Default fast-v2.2 Step 7 inference loads the bundled ONNX regressor."
            if not args.checkpoint else "This benchmark exports and measures the configured PyTorch checkpoint; the application default remains the bundled ONNX regressor."),
        "split_snapshot": snapshot_id, "split": "test",
        "cohort": [{"filename": row["photo"].name, "path": row["label_filename"], "label": row["label"]}
                   for row in cohort],
        "startup_to_ready_ms": startup_ms, "model_setup": setup,
        "protocol": {"warmup_runs_per_photo": 1, "measured_runs_per_photo": args.repeats,
            "measurement": "Direct in-process raw photo to score; server pipeline excludes HTTP and startup."},
        "latency_summary": {"server_pipeline_ms": speed}, "quality_metrics": quality_metrics(quality_rows),
        "runs": records, "failures": failures}
    output = args.output or (HERE / f"screen-{args.run_id}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}; photos={len(grouped)}/{len(cohort)}, "
          f"median={speed['median_ms'] if speed else None:.2f}ms; quality={report['quality_metrics']}", flush=True)


if __name__ == "__main__":
    main()
