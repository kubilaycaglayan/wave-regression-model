#!/usr/bin/env python3
"""Prepare, train, export, and fully compare the FP16-matched regressor."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ARTIFACTS = HERE / "artifacts"
SNAPSHOT = "20261102T084438874261Z"


def configured(name: str, default: str) -> str:
    env_file = ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip() == name:
                return value.strip().strip("\"'")
    return os.environ.get(name, default)


def run_logged(command: list[str], *, env: dict, log_path: Path, cwd: Path = ROOT) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print(f"\n$ {' '.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    elapsed = time.perf_counter() - started
    print(f"Completed in {elapsed:.2f}s (exit={code}); log={log_path}", flush=True)
    if code:
        raise subprocess.CalledProcessError(code, command)
    return elapsed


def metrics(rows: list[dict]) -> dict:
    successful = [row for row in rows if row.get("status") == "ok" and row.get("prediction") is not None]
    if not successful:
        return {"count": 0, "failures": len(rows), "mae": None, "rmse": None, "bias": None}
    errors = [float(row["prediction"]) - float(row["label"]) for row in successful]
    labels = [float(row["label"]) for row in successful]
    prediction = [float(row["prediction"]) for row in successful]
    target_mean = statistics.mean(labels)
    residual_sum = sum(error * error for error in errors)
    total_sum = sum((label - target_mean) ** 2 for label in labels)
    return {"count": len(successful), "failures": len(rows) - len(successful),
            "mae": statistics.mean(abs(error) for error in errors),
            "rmse": math.sqrt(statistics.mean(error * error for error in errors)),
            "bias": statistics.mean(errors), "prediction_mean": statistics.mean(prediction),
            "prediction_min": min(prediction), "prediction_max": max(prediction),
            "r2": 1 - residual_sum / total_sum if total_sum else None}


def per_photo(rows: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("status") == "ok":
            grouped.setdefault(row["label_filename"], []).append(row)
    return {name: {"photo": rows[0]["photo"], "label": rows[0]["label"],
            "prediction": statistics.median(float(row["prediction"]) for row in rows),
            "latency_ms": statistics.median(float(row["server_pipeline_ms"]) for row in rows),
            "crop": rows[0].get("crop")}
            for name, rows in grouped.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--skip-prepare", action="store_true", help="Reuse a previously prepared matching train/validation cache")
    parser.add_argument("--skip-training", action="store_true", help="Reuse the run's existing training results")
    args = parser.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    runtime_python = Path(configured("PREDICT_MIGRAPHX_PYTHON", "/tmp/wave_migraphx_312_env/bin/python")).expanduser()
    if not runtime_python.is_file():
        raise FileNotFoundError(f"MIGraphX runtime Python not found: {runtime_python}")
    runtime_root = runtime_python.parent.parent
    migraphx_libs = runtime_root / "lib/python3.12/site-packages/migraphx_libs"
    library_paths = [str(path) for path in (migraphx_libs, Path("/opt/rocm/lib")) if path.is_dir()]
    if os.environ.get("LD_LIBRARY_PATH"):
        library_paths.append(os.environ["LD_LIBRARY_PATH"])
    runtime_env = os.environ.copy()
    runtime_env.update({"LD_LIBRARY_PATH": ":".join(library_paths),
        "MIGRAPHX_SKIP_BENCHMARKING": "1", "MIGRAPHX_USE_FAST_SOFTMAX": "0",
        "WAVE_PIPELINE_PROFILE": "fast-v2.1-coordinate", "WAVE_MIGRAPHX_FP16": "1"})
    python = sys.executable
    logs = ARTIFACTS / "logs"
    stage_times = {}
    overall_started = time.perf_counter()

    if not args.skip_prepare:
        stage_times["prepare_train_validation_seconds"] = run_logged(
            [str(runtime_python), str(HERE / "prepare_inputs.py"), "--split", "train", "--split", "validation"],
            env=runtime_env, log_path=logs / f"{args.run_id}-prepare-train-validation.log")

    training_result_path = ARTIFACTS / "training-runs" / args.run_id / "training-results.json"
    if not args.skip_training:
        stage_times["training_seconds_including_process_start"] = run_logged(
            [python, str(HERE / "train_regressor.py"), "--run-id", args.run_id],
            env=os.environ.copy(), log_path=logs / f"{args.run_id}-training.log")
    if not training_result_path.is_file():
        raise FileNotFoundError(f"Training results are missing: {training_result_path}")
    training = json.loads(training_result_path.read_text())
    checkpoint = HERE / training["best_checkpoint"]
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model_path = ARTIFACTS / "models" / f"fp16-retrained-regressor-{args.run_id}.onnx"
    stage_times["onnx_export_seconds"] = run_logged(
        [python, str(HERE / "export_regressor.py"), str(checkpoint), str(model_path)],
        env=os.environ.copy(), log_path=logs / f"{args.run_id}-export.log")

    raw_outputs = {}
    for mode, regressor in [
        ("baseline-fp32", ROOT / "models/fast-v2.1/wave-regression-efficientnet-b0-fast-v2.1.onnx"),
        ("fp16-current-regressor", ROOT / "models/fast-v2.1/wave-regression-efficientnet-b0-fast-v2.1.onnx"),
        ("fp16-retrained-regressor", model_path),
    ]:
        output = ARTIFACTS / "benchmark-runs" / args.run_id / f"{mode}.json"
        command = [str(runtime_python), str(HERE / "benchmark_runtime.py"), "--mode", mode,
                   "--output", str(output), "--repeats", "3"]
        if mode == "fp16-retrained-regressor":
            command += ["--regressor", str(regressor)]
        mode_env = runtime_env.copy()
        if mode == "baseline-fp32":
            mode_env.pop("WAVE_MIGRAPHX_FP16", None)
        stage_times[f"benchmark_{mode}_seconds"] = run_logged(
            command, env=mode_env, log_path=logs / f"{args.run_id}-{mode}.log")
        raw_outputs[mode] = json.loads(output.read_text())

    speed = {}
    quality = {}
    grouped = {mode: per_photo(report["runs"]) for mode, report in raw_outputs.items()}
    for mode, report in raw_outputs.items():
        rows = report["runs"]
        # Three timed repeats are speed samples, not independent quality cases.
        # Score each held-out photo once using its median repeated prediction.
        quality[mode] = metrics([{"status": "ok", "prediction": value["prediction"],
                                  "label": value["label"]}
                                 for value in grouped[mode].values()])
        latency = [float(row["server_pipeline_ms"]) for row in rows if row.get("status") == "ok"]
        speed[mode] = {"latency_summary": {
            "count": len(latency), "median_ms": statistics.median(latency),
            "p95_ms": sorted(latency)[max(0, math.ceil(.95 * len(latency)) - 1)],
            "min_ms": min(latency), "max_ms": max(latency)},
            "startup_to_ready_ms": report["startup_to_ready_ms"],
            "successful_photos": report["successful_photo_count"],
            "failures": report["failures"]}

    base, fp16_old, fp16_new = (grouped[name] for name in
        ("baseline-fp32", "fp16-current-regressor", "fp16-retrained-regressor"))
    common_names = sorted(set(base) & set(fp16_old) & set(fp16_new))
    comparison = []
    for name in common_names:
        b, f, n = base[name], fp16_old[name], fp16_new[name]
        comparison.append({"label_filename": name, "photo": b["photo"], "label": b["label"],
            "baseline_fp32_prediction": b["prediction"], "fp16_current_regressor_prediction": f["prediction"],
            "fp16_retrained_prediction": n["prediction"],
            "baseline_fp32_absolute_error": abs(b["prediction"] - b["label"]),
            "fp16_current_absolute_error": abs(f["prediction"] - f["label"]),
            "fp16_retrained_absolute_error": abs(n["prediction"] - n["label"]),
            "retrained_minus_baseline_absolute_error": abs(n["prediction"] - n["label"]) - abs(b["prediction"] - b["label"]),
            "fp32_crop": b["crop"], "fp16_current_crop": f["crop"], "fp16_retrained_crop": n["crop"],
            "fp32_latency_ms": b["latency_ms"], "fp16_current_latency_ms": f["latency_ms"],
            "fp16_retrained_latency_ms": n["latency_ms"]})
    if len(common_names) != 46:
        print(f"Warning: only {len(common_names)}/46 test photos succeeded in all three modes", flush=True)

    baseline_report = raw_outputs["baseline-fp32"]
    candidate_report = raw_outputs["fp16-retrained-regressor"]
    cohort = [{"filename": row["photo"], "path": row["label_filename"], "label": row["label"]}
              for row in baseline_report["runs"] if row["repeat"] == 1]
    final_report = {
        "schema_version": 1, "experiment_id": "fp16-retrained-regressor", "run_id": args.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "variant": "fp16-retrained-regressor", "runtime": "MIGraphX FP16 SegFormer + ONNX Runtime CPU regressor",
        "pipeline": "fast-v2.1-coordinate with MIGraphX FP16 segmentation and experiment-matched EfficientNet-B0 regressor",
        "step1_model": candidate_report["step1_model"], "step1_sha256": candidate_report["step1_sha256"],
        "step1_size": 448, "fp16": True, "regressor_model": str(model_path),
        "regressor_sha256": candidate_report["regressor_sha256"], "checkpoint": str(checkpoint),
        "checkpoint_sha256": __import__("hashlib").sha256(checkpoint.read_bytes()).hexdigest(),
        "split_snapshot": SNAPSHOT, "split": "test", "cohort": cohort,
        "startup_to_ready_ms": candidate_report["startup_to_ready_ms"],
        "model_setup": candidate_report["model_setup"], "protocol": candidate_report["protocol"],
        "latency_summary": candidate_report["latency_summary"], "runs": candidate_report["runs"],
        "training_summary": training,
        "quality_comparison": {"test_metrics_by_mode": quality,
            "full_common_test_count": len(common_names), "per_photo": comparison,
            "candidate_mae_delta_vs_baseline": quality["fp16-retrained-regressor"]["mae"] - quality["baseline-fp32"]["mae"],
            "candidate_rmse_delta_vs_baseline": quality["fp16-retrained-regressor"]["rmse"] - quality["baseline-fp32"]["rmse"],
            "candidate_mae_delta_vs_fp16_current_regressor": quality["fp16-retrained-regressor"]["mae"] - quality["fp16-current-regressor"]["mae"]},
        "speed_comparison": speed,
        "preprocessing_cache": {"signature": training["preprocessing_signature"],
            "train_manifest": training["train_cache_manifest"], "validation_manifest": training["validation_cache_manifest"]},
        "stage_elapsed_seconds": {**stage_times, "total_seconds": time.perf_counter() - overall_started},
        "quality_policy": "Validation selected the checkpoint; held-out test was first evaluated after checkpoint selection. No train/test leakage.",
        "notes": ["All three inference modes use the identical 46-photo held-out test split.",
                  "The baseline uses FP32 segmentation with the current regressor.",
                  "The FP16 screen control uses FP16 segmentation with the current regressor.",
                  "The candidate uses FP16 segmentation with the validation-selected retrained regressor."]}
    baseline_median = speed["baseline-fp32"]["latency_summary"]["median_ms"]
    final_report["paired_screen"] = {"baseline_report": "FP32 baseline measured in this run",
        "median_latency_delta_ms": candidate_report["latency_summary"]["server_pipeline_ms"]["median_ms"] - baseline_median,
        "median_absolute_prediction_delta": statistics.median(abs(row["fp16_retrained_prediction"] - row["baseline_fp32_prediction"])
            for row in comparison) if comparison else None,
        "photos_with_changed_crop": sum(row["fp32_crop"] != row["fp16_retrained_crop"] for row in comparison),
        "per_photo": [{"photo": row["photo"], "base_median_ms": row["fp32_latency_ms"],
            "candidate_median_ms": row["fp16_retrained_latency_ms"],
            "base_prediction": row["baseline_fp32_prediction"], "candidate_prediction": row["fp16_retrained_prediction"],
            "prediction_delta": row["fp16_retrained_prediction"] - row["baseline_fp32_prediction"],
            "crop_changed": row["fp32_crop"] != row["fp16_retrained_crop"],
            "baseline_crop": row["fp32_crop"], "candidate_crop": row["fp16_retrained_crop"]} for row in comparison]}
    final_report_path = HERE / f"screen-{args.run_id}.json"
    final_report_path.write_text(json.dumps(final_report, indent=2) + "\n")
    subprocess.run([python, str(ROOT / "experiments/from-one-to-zero/generate_index.py")], cwd=ROOT, check=True)
    print(json.dumps({"report": str(final_report_path), "html": "experiments/from-one-to-zero/index.html",
        "training": training["selected_checkpoint_metrics"], "test_quality": quality,
        "speed_median_ms": {mode: d["latency_summary"]["median_ms"] for mode, d in speed.items()},
        "test_count_common": len(common_names)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
