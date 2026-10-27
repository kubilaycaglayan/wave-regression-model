#!/usr/bin/env python3
"""Measure atomic-photo-arrival to completed record in a warmed watcher."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
WORKER = """import sys
from pathlib import Path
import step_7_e_watch_predictions as watch
root = Path(sys.argv[1])
watch.prediction.INPUT_DIR = root / 'incoming'
watch.prediction.PREVIEW_DIR = root / 'previews'
watch.prediction.PREDICTIONS_DIR = root / 'records'
sys.argv = ['watcher', *sys.argv[2:]]
watch.main()
"""


def percentile95(values):
    ordered = sorted(values)
    return ordered[math.ceil(.95 * len(ordered)) - 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("photos", nargs="+", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--step2-resolution", type=int, required=True)
    parser.add_argument("--logits-resolution", type=int, required=True)
    parser.add_argument("--precision", choices=("float32", "float16", "bfloat16"), default="float16")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "step-7-prediction-benchmarks")
    args = parser.parse_args()
    if min(args.resolution, args.step2_resolution, args.logits_resolution, args.threads) < 1 or args.timeout <= 0:
        parser.error("resolutions, threads, and timeout must be positive")
    photos = [photo.resolve(strict=True) for photo in args.photos]
    supported = {".heic", ".heif", ".jpg", ".jpeg", ".png"}
    if any(not photo.is_file() or photo.suffix.lower() not in supported for photo in photos):
        parser.error("each photo must be a supported existing image")
    checkpoint = args.checkpoint.resolve(strict=True)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc)
    report_dir = args.output_dir / timestamp.strftime("%Y%m%dT%H%M%S%fZ")
    report_dir.mkdir(exist_ok=False)
    environment = os.environ.copy()
    environment.update(PREDICT_CHECKPOINT_PATH=str(checkpoint), OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
    watcher_args = ["--watch-mode", "event", "--threads", str(args.threads),
                    "--experimental-single-resolution", str(args.resolution),
                    "--experimental-step2-resolution", str(args.step2_resolution),
                    "--experimental-logits-resolution", str(args.logits_resolution),
                    "--experimental-precision", args.precision]
    runs = []
    with tempfile.TemporaryDirectory(prefix="wave-arrival-benchmark-") as temporary:
        root = Path(temporary)
        (root / "incoming").mkdir()
        log_path = report_dir / "watcher.log"
        with log_path.open("w+", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-c", WORKER, str(root), *watcher_args],
                                       cwd=PROJECT_DIR, env=environment, stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + args.timeout
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"Watcher exited during startup: {process.returncode}")
                    log.flush()
                    if "File notifications ready" in log_path.read_text(encoding="utf-8"):
                        break
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Watcher did not become ready")
                    time.sleep(.01)
                namespaces = list((root / "records").glob("experimental-*"))
                if len(namespaces) != 1:
                    raise RuntimeError(f"Expected one experimental record namespace, got {len(namespaces)}")
                for index, photo in enumerate(photos):
                    name = f"photo-{index:04d}{photo.suffix.lower()}"
                    staged = root / f"staged-{index:04d}"
                    staged.symlink_to(photo)
                    destination = root / "incoming" / name
                    record = namespaces[0] / f"{Path(name).stem}.txt"
                    started = time.perf_counter()
                    os.replace(staged, destination)
                    deadline = time.monotonic() + args.timeout
                    while True:
                        if process.poll() is not None:
                            raise RuntimeError(f"Watcher exited while processing {name}: {process.returncode}")
                        if record.is_file():
                            text = record.read_text(encoding="utf-8")
                            if text.endswith("\n\n") and "prediction: " in text and "photo_elapsed_seconds: " in text:
                                elapsed = time.perf_counter() - started
                                fields = dict(line.split(": ", 1) for line in text.splitlines() if ": " in line)
                                runs.append({"photo": str(photo), "arrival_to_record_seconds": elapsed,
                                             "prediction": float(fields["prediction"]),
                                             "processing_seconds_reported": float(fields["photo_elapsed_seconds"])})
                                print(f"{index + 1}/{len(photos)} {photo.name}: {elapsed:.3f}s", flush=True)
                                break
                        if time.monotonic() >= deadline:
                            raise TimeoutError(f"No completed record for {name}; see {log_path}")
                        time.sleep(.001)
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    values = [item["arrival_to_record_seconds"] for item in runs]
    result = {"created_utc": timestamp.isoformat(), "checkpoint": str(checkpoint), "checkpoint_sha256": digest,
              "configuration": {"resolution": args.resolution, "step2_resolution": args.step2_resolution,
                                "logits_resolution": args.logits_resolution, "precision": args.precision,
                                "threads": args.threads, "watch_mode": "event"},
              "measurement": "atomic rename to complete prediction record; excludes producer copy, startup and warmup",
              "count": len(values), "median_seconds": statistics.median(values), "p95_seconds": percentile95(values),
              "min_seconds": min(values), "max_seconds": max(values), "runs": runs}
    (report_dir / "arrival.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (report_dir / "arrival.md").write_text(
        "# Photo-arrival benchmark\n\n"
        f"{result['measurement']}. Photos are delivered one at a time using atomic moves; "
        "measurements include notification delivery and scheduling, not upload time or queued requests.\n\n"
        f"Checkpoint SHA-256: `{digest}`\n\n"
        f"Photos: {len(values)}; median: {result['median_seconds']:.6f} s; "
        f"p95: {result['p95_seconds']:.6f} s; maximum: {result['max_seconds']:.6f} s.\n\n"
        "See `arrival.json` for the individual photos and `watcher.log` for runtime details.\n",
        encoding="utf-8")
    print(f"Report: {report_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
