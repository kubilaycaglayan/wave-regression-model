#!/usr/bin/env python3
"""Measure one held-out test cohort through a selected fast-v2.1 inference mode."""
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
SNAPSHOT = "20261102T084438874261Z"
sys.path.insert(0, str(ROOT))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summary(values: list[float]) -> dict:
    values = sorted(values)
    return {"count": len(values), "median_ms": statistics.median(values),
            "p95_ms": values[max(0, math.ceil(.95 * len(values)) - 1)],
            "min_ms": values[0], "max_ms": values[-1]}


def load_test_rows() -> list[dict]:
    raw_dir = ROOT / "step-0-raw-data"
    raw_by_stem: dict[str, list[Path]] = {}
    for path in raw_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".heic", ".heif", ".jpg", ".jpeg", ".png"}:
            raw_by_stem.setdefault(path.stem.casefold(), []).append(path)
    split = ROOT / "step-3-dataset-splits" / "snapshots" / SNAPSHOT / "test.csv"
    rows = []
    with split.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            stem = Path(row["filename"]).stem
            if stem.startswith("step-2_"):
                stem = stem[len("step-2_"):]
            matches = raw_by_stem.get(stem.casefold(), [])
            if len(matches) != 1:
                raise RuntimeError(f"Expected one raw source for {row['filename']}; found {len(matches)}")
            rows.append({"photo": matches[0], "label_filename": row["filename"],
                         "label": float(row["waviness"])})
    return sorted(rows, key=lambda row: row["label_filename"].casefold())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline-fp32", "fp16-current-regressor", "fp16-retrained-regressor"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--regressor", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.mode == "fp16-retrained-regressor" and not args.regressor:
        parser.error("--regressor is required for fp16-retrained-regressor")
    os.environ["WAVE_PIPELINE_PROFILE"] = "fast-v2.1-coordinate"
    os.environ["WAVE_STEP1_SIZE"] = "448"
    os.environ["WAVE_STEP1_ONNX"] = str(ROOT / "models/fast-v2.1/segformer-b0-step1-448-fp32.onnx")
    if args.mode != "baseline-fp32":
        os.environ["WAVE_MIGRAPHX_FP16"] = "1"
    regressor = args.regressor or ROOT / "models/fast-v2.1/wave-regression-efficientnet-b0-fast-v2.1.onnx"
    os.environ["WAVE_REGRESSOR_ONNX"] = str(regressor.resolve())

    import step_7_i_migraphx_server as server
    startup_setup = server.initialize_models()
    startup_ms = (time.perf_counter() - server.PROCESS_START) * 1000
    rows = load_test_rows()
    records, failures = [], []
    for index, item in enumerate(rows, start=1):
        photo = item["photo"]
        try:
            server.run_fast_v2_1_pipeline(photo)  # per-photo warm-up, excluded
            for repeat in range(1, args.repeats + 1):
                started = time.perf_counter()
                result = server.run_fast_v2_1_pipeline(photo)
                wall_ms = (time.perf_counter() - started) * 1000
                records.append({"photo": photo.name, "label_filename": item["label_filename"],
                    "label": item["label"], "repeat": repeat, "status": "ok",
                    "prediction": result["prediction"],
                    "total_raw_to_score_ms": result["total_server_ms"],
                    "server_pipeline_ms": result["total_server_ms"],
                    "client_runtime_wall_ms": wall_ms,
                    "stages_ms": result["stages_ms"],
                    "crop": result["intermediate"]["source_crop_xyxy"]})
                print(f"{args.mode} {index}/{len(rows)} {photo.name} run {repeat}: "
                      f"{result['total_server_ms']:.2f} ms, score={result['prediction']:.5f}", flush=True)
        except Exception as error:
            failures.append({"photo": photo.name, "label_filename": item["label_filename"],
                             "label": item["label"], "error": f"{type(error).__name__}: {error}"})
            print(f"{args.mode} {index}/{len(rows)} {photo.name}: FAILED {failures[-1]['error']}", flush=True)

    report = {"mode": args.mode, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_profile": "fast-v2.1-coordinate", "fp16_segmentation": args.mode != "baseline-fp32",
        "split_snapshot": SNAPSHOT, "split": "test", "expected_photo_count": len(rows),
        "successful_photo_count": len({row["photo"] for row in records}), "failures": failures,
        "step1_model": str(server.STEP1_ONNX), "step1_sha256": sha(Path(server.STEP1_ONNX)),
        "regressor_model": str(regressor), "regressor_sha256": sha(regressor),
        "startup_to_ready_ms": startup_ms, "model_setup": startup_setup,
        "protocol": {"warmup_runs_per_photo": 1, "measured_runs_per_photo": args.repeats,
            "measurement": "Direct in-process raw photo to score; server pipeline time excludes HTTP overhead and startup."},
        "latency_summary": {"server_pipeline_ms": summary([r["server_pipeline_ms"] for r in records]) if records else None},
        "runs": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}; startup={startup_ms / 1000:.2f}s; successful photos="
          f"{report['successful_photo_count']}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
