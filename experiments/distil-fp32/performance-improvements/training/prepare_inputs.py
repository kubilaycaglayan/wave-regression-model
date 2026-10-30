#!/usr/bin/env python3
"""Cache experiment-matched RGB regressor inputs for train/validation splits."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

HERE = Path(__file__).resolve().parent
PERF = HERE.parent
ROOT = PERF.parents[2]
SNAPSHOT = "20261028T081606614158Z"
EXPERIMENTS = PERF / "experiment-versions"
CACHE_ROOT = PERF / "training-cache"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PERF))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Python module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CaptureRegressorInput:
    """Stand-in regressor that captures the exact normalized tensor it receives."""

    def __init__(self):
        self.tensor = None

    def run(self, _outputs, feed):
        import numpy as np

        self.tensor = np.array(feed["images"], copy=True)
        return [np.asarray([[0.5]], dtype=np.float32)]


def input_to_rgb(tensor, mean, std):
    """Recover the final uint8 RGB image from the pipeline's normalized tensor."""
    import numpy as np

    normalized = np.asarray(tensor, dtype=np.float32)[0].transpose(1, 2, 0)
    mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    rgb_float = normalized * std + mean
    rgb = np.rint(np.clip(rgb_float * np.float32(255.0), 0, 255)).astype(np.uint8)
    recovered = ((rgb.astype(np.float32) / np.float32(255.0)
                  - mean) / std)
    if not np.allclose(recovered.transpose(2, 0, 1)[None], tensor, rtol=0.0, atol=2e-6):
        raise RuntimeError("RGB cache cannot faithfully reconstruct the regressor input tensor")
    return rgb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", action="append",
                        help="experiment version to prepare; repeatable (default: v000 through v004)")
    parser.add_argument("--split", action="append", choices=("train", "validation"),
                        help="split to prepare (default: train and validation)")
    args = parser.parse_args()
    experiment_ids = args.experiment_id or [
        "v000_baseline",
        "v001_coordinate_crop_single_output",
        "v002_skip_step2_keep_baseline_image",
        "v003_roi_mask_processing",
        "v004_working_resolution_cap",
    ]
    splits = args.split or ["train", "validation"]

    benchmark = load_module(PERF / "benchmark_distil_fp32.py", "experiment_training_benchmark")
    os.environ["WAVE_EXPERIMENT"] = "distilled-b0-migraphx-fp32"
    import step_7_i_experimental_migraphx_server as server

    # The segmentation models are shared by the variants; initialize once.
    print("Loading shared Step 1/Step 2 models for experiment input preparation...", flush=True)
    warm_started = time.perf_counter()
    benchmark.initialize_models_for_steps(server, [1, 2])
    print(f"Shared models ready in {time.perf_counter() - warm_started:.2f}s", flush=True)
    capture = CaptureRegressorInput()
    server.REGRESSOR_SESSION = capture

    for experiment_id in experiment_ids:
        experiment_dir = EXPERIMENTS / experiment_id
        manifest_path = experiment_dir / "experiment.json"
        code_path = experiment_dir / "experiment.py"
        if not manifest_path.is_file() or not code_path.is_file():
            raise FileNotFoundError(f"Expected experiment adapter and manifest under {experiment_dir}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("id") != experiment_id:
            raise ValueError(f"Experiment ID mismatch in {manifest_path}")
        experiment = load_module(code_path, f"training_adapter_{experiment_id}")
        if not hasattr(experiment, "run_pipeline"):
            raise TypeError(f"{code_path} must define run_pipeline")

        signature = {
            "experiment_id": experiment_id,
            "experiment_code_sha256": sha256_file(code_path),
            "experiment_manifest_sha256": sha256_file(manifest_path),
            "benchmark_code_sha256": sha256_file(PERF / "benchmark_distil_fp32.py"),
            "server_code_sha256": sha256_file(ROOT / "step_7_i_experimental_migraphx_server.py"),
            "split_snapshot": SNAPSHOT,
            "snapshot_manifest_sha256": sha256_file(ROOT / "step-3-dataset-splits" / "snapshots" / SNAPSHOT / "manifest.json"),
        }
        signature_id = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
        print(f"\nPreparing {experiment_id} (cache signature {signature_id})", flush=True)
        for split in splits:
            raw_dir = ROOT / "step-0-raw-data"
            rows = benchmark.load_split(split, raw_dir, None)
            split_path = ROOT / "step-3-dataset-splits" / "snapshots" / SNAPSHOT / f"{split}.csv"
            split_signature = {**signature, "split": split,
                               "split_csv_sha256": sha256_file(split_path),
                               "rows": len(rows)}
            cache_dir = CACHE_ROOT / experiment_id / SNAPSHOT / signature_id / split
            cache_dir.mkdir(parents=True, exist_ok=True)
            index_path = cache_dir / "manifest.json"
            if index_path.is_file():
                index = json.loads(index_path.read_text(encoding="utf-8"))
                if index.get("signature") != split_signature:
                    raise RuntimeError(f"Cache signature mismatch; choose a new signature path: {cache_dir}")
            else:
                index = {"schema_version": 1, "signature": split_signature, "items": {}, "failures": {}}
            index.setdefault("failures", {})
            started_split = time.perf_counter()
            skipped = processed = failed = 0
            for row_number, item in enumerate(rows, start=1):
                label_name = item["label_filename"]
                source_hash = sha256_file(item["photo"])
                cache_name = Path(label_name).with_suffix(".npy").name
                output_path = cache_dir / cache_name
                previous = index["items"].get(label_name, {})
                if (output_path.is_file() and previous.get("source_sha256") == source_hash
                        and previous.get("cache_file") == cache_name):
                    skipped += 1
                    print(f"{split} {row_number}/{len(rows)} {label_name}: cached, skip", flush=True)
                    continue
                previous_failure = index["failures"].get(label_name)
                if previous_failure and previous_failure.get("source_sha256") == source_hash:
                    failed += 1
                    print(f"{split} {row_number}/{len(rows)} {label_name}: previously failed, skip "
                          f"({previous_failure['error']})", flush=True)
                    continue

                capture.tensor = None
                photo_started = time.perf_counter()
                try:
                    experiment.run_pipeline(item["photo"], server, benchmark.run_instrumented)
                except Exception as error:
                    index["failures"][label_name] = {
                        "source_filename": item["photo"].name,
                        "source_sha256": source_hash,
                        "label": item["label"],
                        "error": f"{type(error).__name__}: {error}",
                    }
                    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
                    failed += 1
                    print(f"{split} {row_number}/{len(rows)} {label_name}: preprocessing failed "
                          f"({index['failures'][label_name]['error']})", flush=True)
                    continue
                if capture.tensor is None:
                    raise RuntimeError(f"Adapter did not pass a regressor tensor for {label_name}")
                rgb = input_to_rgb(capture.tensor, server.MEAN, server.STD)
                if rgb.shape != (224, 224, 3):
                    raise ValueError(f"Unexpected regressor input shape for {label_name}: {rgb.shape}")
                import numpy as np
                temp_path = output_path.with_suffix(".npy.tmp")
                with temp_path.open("wb") as stream:
                    np.save(stream, rgb, allow_pickle=False)
                temp_path.replace(output_path)
                index["items"][label_name] = {
                    "source_filename": item["photo"].name,
                    "source_sha256": source_hash,
                    "label": item["label"],
                    "cache_file": cache_name,
                    "cache_sha256": sha256_file(output_path),
                }
                index["failures"].pop(label_name, None)
                index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
                processed += 1
                print(f"{split} {row_number}/{len(rows)} {label_name}: prepared in "
                      f"{time.perf_counter() - photo_started:.3f}s", flush=True)
            print(f"{split}: {processed} prepared, {skipped} reused, {failed} failed, "
                  f"{time.perf_counter() - started_split:.2f}s total", flush=True)
            index["created_at_utc"] = datetime.now(timezone.utc).isoformat()
            index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
