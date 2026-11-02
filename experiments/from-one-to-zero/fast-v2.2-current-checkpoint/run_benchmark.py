#!/usr/bin/env python3
"""Export the configured current checkpoint and benchmark it in fast-v2.2."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_WORKER = ROOT / "experiments/from-one-to-zero/fast-v2.2-current-default/benchmark_default.py"
EXPORTER = ROOT / "experiments/from-one-to-zero/fp16-retrained-regressor/export_regressor.py"
ARTIFACTS = HERE / "artifacts"


def env_value(name: str, default: str = "") -> str:
    if name in os.environ:
        return os.environ[name]
    path = ROOT / ".env"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip() == name:
                return value.strip().strip("\"'")
    return default


def current_checkpoint() -> tuple[Path, Path, dict]:
    configured = env_value("PREDICT_CHECKPOINT_PATH")
    if not configured:
        raise RuntimeError("PREDICT_CHECKPOINT_PATH is not set in .env or the environment")
    checkpoint = Path(configured).expanduser()
    checkpoint = checkpoint if checkpoint.is_absolute() else ROOT / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Configured checkpoint not found: {checkpoint}")
    manifests = sorted(checkpoint.parent.glob("training_manifest_*.json"))
    if not manifests:
        raise FileNotFoundError(f"No training manifest beside checkpoint: {checkpoint}")
    manifest_path = manifests[-1]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("preprocessing_profile") != "fast-v2.2":
        raise RuntimeError(f"Expected a fast-v2.2 checkpoint; got {manifest.get('preprocessing_profile')}")
    return checkpoint, manifest_path, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    checkpoint, training_manifest, training = current_checkpoint()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    onnx_path = ARTIFACTS / "models" / f"fast-v2.2-current-checkpoint-{args.run_id}.onnx"
    print(f"Current checkpoint: {checkpoint}", flush=True)
    print(f"Training manifest: {training_manifest}; profile={training['preprocessing_profile']}; "
          f"validation MAE={training.get('best_validation_mae')}", flush=True)
    subprocess.run([sys.executable, str(EXPORTER), str(checkpoint), str(onnx_path)], cwd=ROOT, check=True)

    runtime_python = Path(env_value("PREDICT_MIGRAPHX_PYTHON", "/tmp/wave_migraphx_312_env/bin/python")).expanduser()
    if not runtime_python.is_file():
        raise FileNotFoundError(f"MIGraphX Python runtime not found: {runtime_python}")
    runtime_root = runtime_python.parent.parent
    libs = [str(path) for path in (
        runtime_root / "lib/python3.12/site-packages/migraphx_libs", Path("/opt/rocm/lib")) if path.is_dir()]
    if os.environ.get("LD_LIBRARY_PATH"):
        libs.append(os.environ["LD_LIBRARY_PATH"])
    worker_env = os.environ.copy()
    worker_env["LD_LIBRARY_PATH"] = ":".join(libs)
    worker_env.update({"MIGRAPHX_SKIP_BENCHMARKING": "1", "MIGRAPHX_USE_FAST_SOFTMAX": "0"})
    report_path = HERE / f"screen-{args.run_id}.json"
    subprocess.run([str(runtime_python), str(DEFAULT_WORKER), "--run-id", args.run_id,
        "--repeats", str(args.repeats), "--regressor", str(onnx_path),
        "--checkpoint", str(checkpoint), "--output", str(report_path),
        "--experiment-id", "fast-v2.2-current-checkpoint",
        "--variant", "fast-v2.2-current-checkpoint"], cwd=ROOT, env=worker_env, check=True)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    snapshot = report["split_snapshot"]
    default_reports = sorted((ROOT / "experiments/from-one-to-zero/fast-v2.2-current-default").glob("screen-*.json"))
    if not default_reports:
        raise FileNotFoundError("Run the current default fast-v2.2 benchmark first for a paired reference")
    default_path = next((path for path in reversed(default_reports)
                         if json.loads(path.read_text(encoding="utf-8")).get("split_snapshot") == snapshot), None)
    if default_path is None:
        raise RuntimeError(f"No default fast-v2.2 report uses checkpoint test snapshot {snapshot}")
    baseline = json.loads(default_path.read_text(encoding="utf-8"))
    base_by_name: dict[str, list[dict]] = {}
    for row in baseline["runs"]:
        if row["status"] == "ok":
            base_by_name.setdefault(row["label_filename"], []).append(row)
    candidate_by_name: dict[str, list[dict]] = {}
    for row in report["runs"]:
        if row["status"] == "ok":
            candidate_by_name.setdefault(row["label_filename"], []).append(row)
    paired = []
    for name in sorted(set(base_by_name) & set(candidate_by_name)):
        base_rows = base_by_name[name]
        rows = candidate_by_name[name]
        base = base_rows[0]
        base_prediction = statistics.median(float(row["prediction"]) for row in base_rows)
        prediction = statistics.median(float(row["prediction"]) for row in rows)
        base_latency = statistics.median(float(row["server_pipeline_ms"]) for row in base_rows)
        latency = statistics.median(float(row["server_pipeline_ms"]) for row in rows)
        paired.append({"label_filename": name, "photo": base["photo"], "label": base["label"],
            "default_prediction": base_prediction, "checkpoint_prediction": prediction,
            "prediction_delta": prediction - base_prediction,
            "default_latency_ms": base_latency, "checkpoint_latency_ms": latency,
            "absolute_error_default": abs(base_prediction - base["label"]),
            "absolute_error_checkpoint": abs(prediction - base["label"]),
            "crop_default": base["crop"], "crop_checkpoint": rows[0]["crop"]})
    report["checkpoint_training_manifest"] = str(training_manifest.relative_to(ROOT))
    report["checkpoint_training_summary"] = {"run_version": training.get("run_version"),
        "best_validation_mae": training.get("best_validation_mae"),
        "best_validation_rmse": training.get("best_validation_rmse"),
        "split_snapshot": snapshot}
    report["paired_screen"] = {"baseline_report": str(default_path.relative_to(ROOT)),
        "same_test_split": True, "common_photo_count": len(paired),
        "median_latency_delta_ms": report["latency_summary"]["server_pipeline_ms"]["median_ms"]
            - baseline["latency_summary"]["server_pipeline_ms"]["median_ms"],
        "median_absolute_prediction_delta": statistics.median(abs(row["prediction_delta"]) for row in paired),
        "per_photo": paired}
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "experiments/from-one-to-zero/generate_index.py")],
                   cwd=ROOT, check=True)
    print(f"Wrote paired checkpoint comparison: {report_path}", flush=True)


if __name__ == "__main__":
    main()
