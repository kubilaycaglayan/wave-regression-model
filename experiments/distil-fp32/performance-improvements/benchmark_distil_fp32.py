#!/usr/bin/env python3
"""Fine-grained, quality-aware benchmark for the Distil-FP32 route.

This harness imports the existing experimental server for its model/runtime
configuration and deliberately keeps all reports under this directory. The
image operations match that route; timers split composite stages into exclusive
operation timings so the totals can be interpreted without double-counting.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from types import ModuleType
from datetime import datetime, timezone
from pathlib import Path

PROCESS_START = time.perf_counter()
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SPLIT_SNAPSHOT = "20261028T081606614158Z"
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", default="v000_baseline",
                        help="version directory under experiment-versions/ (default: v000_baseline)")
    parser.add_argument("--final-test", action="store_true",
                        help="run the reserved test split for final checks; validation is the default")
    parser.add_argument("--output-dir", type=Path, default=HERE / "runs")
    parser.add_argument("--compare-reference", type=Path,
                        help="prior report JSON; compares predictions on matched photos")
    parser.add_argument("--limit", type=int,
                        help="optional first-N smoke run per split; not a quality result")
    return parser.parse_args()


class Timer:
    def __init__(self, values: dict[str, float], key: str):
        self.values = values
        self.key = key

    def __enter__(self):
        self.started = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.values[self.key] = (time.perf_counter() - self.started) * 1000.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_split(split: str, raw_dir: Path, limit: int | None) -> list[dict]:
    split_path = ROOT / "step-3-dataset-splits" / "snapshots" / SPLIT_SNAPSHOT / f"{split}.csv"
    if not split_path.is_file():
        raise FileNotFoundError(f"Dataset split is missing: {split_path}")
    raw_by_stem: dict[str, list[Path]] = {}
    for path in raw_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".heic", ".heif", ".jpg", ".jpeg", ".png"}:
            raw_by_stem.setdefault(path.stem.casefold(), []).append(path)
    rows = []
    with split_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            stem = Path(row["filename"]).stem
            if stem.startswith("step-2_"):
                stem = stem[len("step-2_"):]
            matches = raw_by_stem.get(stem.casefold(), [])
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected one raw photo for labeled file {row['filename']!r}; found {len(matches)}"
                )
            rows.append({
                "split": split,
                "label_filename": row["filename"],
                "photo": matches[0],
                "label": float(row["waviness"]),
            })
    rows.sort(key=lambda item: item["label_filename"].casefold())
    return rows[:limit] if limit is not None else rows


def input_tensor(image, size: int, server, timings: dict[str, float], prefix: str):
    """Distil-FP32's existing OpenCV processor, timed at each allocation stage."""
    import cv2
    import numpy as np

    source = image.pixels if isinstance(image, server.NativeRGBFrame) else np.asarray(image, dtype=np.uint8)
    interpolation = cv2.INTER_AREA if size < max(source.shape[:2]) else cv2.INTER_LINEAR_EXACT
    with Timer(timings, f"{prefix}.resize_rgb"):
        resized = cv2.resize(source, (size, size), interpolation=interpolation)
    with Timer(timings, f"{prefix}.uint8_to_float32_and_scale"):
        pixels = resized.astype(np.float32)
        pixels *= np.float32(1.0 / 255.0)
    with Timer(timings, f"{prefix}.normalize_mean_std"):
        pixels = (pixels - server.MEAN) / server.STD
    with Timer(timings, f"{prefix}.transpose_and_contiguous_tensor"):
        tensor = np.ascontiguousarray(pixels.transpose(2, 0, 1)[None], dtype=np.float32)
    return tensor


def model_dimensions(image, server) -> tuple[int, int]:
    if isinstance(image, (server.Image.Image, server.NativeRGBFrame)):
        return image.width, image.height
    return image.shape[1], image.shape[0]


def image_pixels(image, server):
    import numpy as np
    if isinstance(image, server.NativeRGBFrame):
        return image.pixels
    if isinstance(image, server.Image.Image):
        return np.asarray(image)
    return image


def timed_canvas(image, server, timings: dict[str, float], prefix: str):
    scale = min(224 / image.width, 224 / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    with Timer(timings, f"{prefix}.lanczos_resize"):
        resized = server.resize_lanczos(image, size)
    with Timer(timings, f"{prefix}.black_canvas_allocate_and_paste"):
        output = server.Image.new("RGB", (224, 224), (0, 0, 0))
        output.paste(resized, ((224 - size[0]) // 2, (224 - size[1]) // 2))
    return output


def run_instrumented(photo: Path, server) -> dict:
    import cv2
    import numpy as np

    timings: dict[str, float] = {}
    started = time.perf_counter()
    with Timer(timings, "preprocessing.decode_oriented_rgb"):
        image = server.decode_photo(photo)
    try:
        width, height = model_dimensions(image, server)

        step1_tensor = input_tensor(image, 448, server, timings, "preprocessing.step1.tensor")
        with Timer(timings, "preprocessing.step1.segmentation_inference"):
            low_mask = server.STEP1_SESSION.run(["water_mask"], {"pixel_values": step1_tensor})[0][0]
        with Timer(timings, "preprocessing.step1.expand_mask_to_source"):
            mask = server.expand_mask(low_mask, width, height)

        radius = max(1, min(12, round(min(width, height) * 0.005)))
        with Timer(timings, "preprocessing.step1.erode_mask"):
            eroded = cv2.erode(mask, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8),
                               borderType=cv2.BORDER_REPLICATE)
        with Timer(timings, "preprocessing.step1.bounding_box"):
            left, top, box_width, box_height = cv2.boundingRect(eroded)
        with Timer(timings, "preprocessing.step1.water_pixel_count_and_validation"):
            water_pixels = cv2.countNonZero(eroded)
            minimum = max(32, round(width * height * 0.0005))
            if box_width == 0 or box_height == 0 or water_pixels < minimum:
                raise ValueError("Step 1 could not reliably detect water")

        with Timer(timings, "preprocessing.step1.roi_views"):
            source = image_pixels(image, server)
            roi_pixels = source[top:top + box_height, left:left + box_width]
            roi_mask = eroded[top:top + box_height, left:left + box_width]
        with Timer(timings, "preprocessing.step1.masked_roi_copy"):
            masked_pixels = cv2.bitwise_and(roi_pixels, roi_pixels, mask=roi_mask)
        with Timer(timings, "preprocessing.step1.pil_image_from_masked_roi"):
            masked = server.Image.fromarray(masked_pixels, mode="RGB")
        step1_image = timed_canvas(masked, server, timings, "preprocessing.step1.canvas")

        with Timer(timings, "preprocessing.step1.jpeg_encode"):
            step1_bytes = server.io.BytesIO()
            step1_image.save(step1_bytes, format="JPEG", quality=95)
            step1_jpeg = step1_bytes.getvalue()
        with Timer(timings, "preprocessing.step1.jpeg_decode_and_rgb_convert"):
            with server.Image.open(server.io.BytesIO(step1_jpeg)) as decoded:
                decoded.load()
                step1_for_step2 = decoded.convert("RGB")

        with Timer(timings, "preprocessing.step2.black_area_mask"):
            rgb = np.asarray(step1_for_step2, dtype=np.uint8)
            black = np.all(rgb <= 12, axis=2)
        step2_tensor = input_tensor(step1_for_step2, 384, server, timings, "preprocessing.step2.tensor")
        with Timer(timings, "preprocessing.step2.segmentation_inference"):
            low_water = server.STEP2_SESSION.run(["water_mask"], {"pixel_values": step2_tensor})[0][0]
        with Timer(timings, "preprocessing.step2.expand_mask_to_canvas"):
            expanded_water = server.expand_mask(low_water, 224, 224)
        with Timer(timings, "preprocessing.step2.mask_convert_and_exclude_black"):
            water = expanded_water.astype(bool) & ~black
        with Timer(timings, "preprocessing.step2.crop_selection"):
            selected = server.select_crop(water, black)
            if selected is None:
                raise ValueError("Step 2 could not reliably select a lower-water crop")
        with Timer(timings, "preprocessing.step2.crop_extract"):
            selected_image = step1_for_step2.crop(selected[0])
        step2_image = timed_canvas(selected_image, server, timings, "preprocessing.step2.canvas")

        with Timer(timings, "preprocessing.final.jpeg_encode"):
            encoded = server.io.BytesIO()
            step2_image.save(encoded, format="JPEG", quality=95, subsampling=0)
            final_jpeg = encoded.getvalue()
        with Timer(timings, "preprocessing.final.jpeg_decode_and_rgb_convert"):
            with server.Image.open(server.io.BytesIO(final_jpeg)) as decoded:
                decoded.load()
                final_rgb = decoded.convert("RGB")
        with Timer(timings, "preprocessing.final.rgb_to_float32_scale"):
            model_pixels = np.asarray(final_rgb, dtype=np.uint8).astype(np.float32) / np.float32(255.0)
        with Timer(timings, "preprocessing.final.normalize_mean_std"):
            normalized = (model_pixels - server.MEAN) / server.STD
        with Timer(timings, "preprocessing.final.transpose_and_contiguous_tensor"):
            model_tensor = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None], dtype=np.float32)
        preprocessing_total = (time.perf_counter() - started) * 1000.0

        inference_timings: dict[str, float] = {}
        inference_started = time.perf_counter()
        with Timer(inference_timings, "waviness_inference.v18_regressor_onnxruntime"):
            regressor_output = server.REGRESSOR_SESSION.run(["scores"], {"images": model_tensor})[0]
        with Timer(inference_timings, "waviness_inference.score_extraction"):
            score = float(regressor_output.reshape(-1)[0])
        inference_total = (time.perf_counter() - inference_started) * 1000.0
        timings.update(inference_timings)
        return {
            "prediction": score,
            "preprocessing_total_ms": preprocessing_total,
            "waviness_inference_total_ms": inference_total,
            "total_raw_to_score_ms": (time.perf_counter() - started) * 1000.0,
            "timings_ms": timings,
            "image": {"width": width, "height": height, "format": photo.suffix.lower()},
            "intermediate": {
                "step1_water_bbox_xywh": [left, top, box_width, box_height],
                "step2_selected_crop_xywh": list(selected[0]),
                "regressor_tensor_shape": list(model_tensor.shape),
                "regressor_tensor_dtype": str(model_tensor.dtype),
            },
        }
    finally:
        if isinstance(image, server.NativeRGBFrame):
            image.close()


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "median_ms": None, "p95_ms": None, "max_ms": None}
    return {
        "count": len(values),
        "median_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
        "max_ms": max(values),
    }


def quality_summary(records: list[dict]) -> dict:
    successful = [row for row in records if row.get("prediction") is not None]
    if not successful:
        return {"n": 0, "failures": len(records), "mae": None, "rmse": None}
    errors = [row["prediction"] - row["label"] for row in successful]
    return {
        "n": len(successful),
        "failures": len(records) - len(successful),
        "mae": statistics.mean(abs(error) for error in errors),
        "rmse": math.sqrt(statistics.mean(error * error for error in errors)),
        "prediction_mean": statistics.mean(row["prediction"] for row in successful),
    }


def initialize_models_for_steps(server, steps: list[int]) -> dict:
    """Warm only the sessions needed by a version; preserve baseline init verbatim."""
    normalized = sorted(set(int(step) for step in steps))
    if normalized == [1, 2]:
        return server.initialize_models()
    if not normalized or any(step not in (1, 2) for step in normalized):
        raise ValueError(f"Invalid required_segmentation_steps: {steps!r}")

    import onnxruntime as ort
    import numpy as np

    setup = {}
    tick = time.perf_counter()
    for name, library_path in zip(server.mx.get_ep_names(), server.mx.get_library_paths()):
        ort.register_execution_provider_library(name, library_path)
    device = next((d for d in ort.get_ep_devices() if d.ep_name == server.mx.get_ep_name()), None)
    if device is None:
        raise RuntimeError("ONNX Runtime did not expose a MIGraphX device")
    setup["runtime_registration_ms"] = (time.perf_counter() - tick) * 1000.0

    for step in (1, 2):
        session_name = f"STEP{step}_SESSION"
        if step in normalized:
            tick = time.perf_counter()
            session = server.load_session(
                getattr(server, f"STEP{step}_ONNX"), device, f"Step {step} segmentation"
            )
            setattr(server, session_name, session)
            setup[f"step{step}_session_load_ms"] = (time.perf_counter() - tick) * 1000.0
        else:
            setattr(server, session_name, None)

    if not server.REGRESSOR_ONNX.is_file():
        raise FileNotFoundError(f"v18 regressor model file not found: {server.REGRESSOR_ONNX}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    tick = time.perf_counter()
    server.REGRESSOR_SESSION = ort.InferenceSession(
        str(server.REGRESSOR_ONNX), sess_options=options, providers=["CPUExecutionProvider"]
    )
    setup["regressor_session_load_ms"] = (time.perf_counter() - tick) * 1000.0
    if "CPUExecutionProvider" not in server.REGRESSOR_SESSION.get_providers():
        raise RuntimeError("The v18 regressor did not load on CPU")

    for step in normalized:
        size = 448 if step == 1 else 384
        tick = time.perf_counter()
        getattr(server, f"STEP{step}_SESSION").run(
            ["water_mask"], {"pixel_values": np.zeros((1, 3, size, size), dtype=np.float32)}
        )
        setup[f"warm_{size}_first_run_ms"] = (time.perf_counter() - tick) * 1000.0
    tick = time.perf_counter()
    server.REGRESSOR_SESSION.run(["scores"], {"images": np.zeros((1, 3, 224, 224), dtype=np.float32)})
    setup["warm_regressor_ms"] = (time.perf_counter() - tick) * 1000.0
    tick = time.perf_counter()
    server.warm_crop_selector()
    setup["warm_crop_selector_ms"] = (time.perf_counter() - tick) * 1000.0
    setup["segmentation_steps_loaded"] = normalized
    return setup


def compare_reference(report: dict, reference_path: Path) -> dict:
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    selected_splits = set(report.get("splits", []))
    reference_rows = {
        (row["split"], row["label_filename"]): row
        for row in reference.get("photos", [])
        if row.get("split") in selected_splits
    }
    candidate_rows = {
        (row["split"], row["label_filename"]): row
        for row in report["photos"]
    }
    omitted_from_candidate = sorted(
        label for (split, label) in reference_rows if (split, label) not in candidate_rows
    )
    outside_reference = sorted(
        label for (split, label) in candidate_rows if (split, label) not in reference_rows
    )
    input_mismatches = sorted(
        row["label_filename"]
        for key, row in candidate_rows.items()
        if key in reference_rows and (
            row.get("source_sha256") != reference_rows[key].get("source_sha256")
            or row.get("label") != reference_rows[key].get("label")
        )
    )
    paired = []
    missing = []
    candidate_failures = []
    reference_failures = []
    for row in report["photos"]:
        key = (row["split"], row["label_filename"])
        old = reference_rows.get(key)
        if old is None:
            missing.append(row["label_filename"])
            continue
        if row.get("prediction") is None and old.get("prediction") is not None:
            candidate_failures.append(row["label_filename"])
            continue
        if old.get("prediction") is None and row.get("prediction") is not None:
            reference_failures.append(row["label_filename"])
            continue
        if row.get("prediction") is None and old.get("prediction") is None:
            continue
        paired.append((row, old))
    if not paired:
        return {
            "reference_report": str(reference_path), "paired_n": 0, "missing": missing,
            "omitted_from_candidate": omitted_from_candidate,
            "outside_reference": outside_reference,
            "input_or_label_mismatches": input_mismatches,
            "strict_no_regression_gate": {"passed": False, "reason": "no paired successful photos"},
        }
    split_results = {}
    for split in sorted({new["split"] for new, _ in paired}):
        members = [(new, old) for new, old in paired if new["split"] == split]
        candidate_errors = [new["prediction"] - new["label"] for new, _ in members]
        reference_errors = [old["prediction"] - old["label"] for _, old in members]
        candidate_mae = statistics.mean(abs(error) for error in candidate_errors)
        reference_mae = statistics.mean(abs(error) for error in reference_errors)
        candidate_rmse = math.sqrt(statistics.mean(error * error for error in candidate_errors))
        reference_rmse = math.sqrt(statistics.mean(error * error for error in reference_errors))
        split_results[split] = {
            "paired_n": len(members),
            "candidate_mae": candidate_mae,
            "reference_mae": reference_mae,
            "mae_delta_candidate_minus_reference": candidate_mae - reference_mae,
            "candidate_rmse": candidate_rmse,
            "reference_rmse": reference_rmse,
            "rmse_delta_candidate_minus_reference": candidate_rmse - reference_rmse,
            "mean_absolute_per_photo_error_delta": statistics.mean(
                abs(new["prediction"] - new["label"]) - abs(old["prediction"] - old["label"])
                for new, old in members
            ),
            "prediction_delta_mae": statistics.mean(
                abs(new["prediction"] - old["prediction"]) for new, old in members
            ),
            "per_photo_absolute_error_regressions": sum(
                abs(new["prediction"] - new["label"]) > abs(old["prediction"] - old["label"])
                for new, old in members
            ),
        }
    same_cohort = not missing and not omitted_from_candidate and not outside_reference and not input_mismatches
    no_failures = not candidate_failures and not reference_failures and all(
        row.get("prediction") is not None for row in report["photos"]
    )
    no_mae_regression = all(
        values["mae_delta_candidate_minus_reference"] <= 0.0 for values in split_results.values()
    )
    no_rmse_regression = all(
        values["rmse_delta_candidate_minus_reference"] <= 0.0 for values in split_results.values()
    )
    return {
        "reference_report": str(reference_path),
        "paired_n": len(paired),
        "missing": missing,
        "candidate_failures_where_reference_succeeded": candidate_failures,
        "reference_failures_where_candidate_succeeded": reference_failures,
        "quality_by_split_on_paired_successful_photos": split_results,
        "strict_no_regression_gate": {
            "passed": same_cohort and no_failures and no_mae_regression and no_rmse_regression and all(
                values["per_photo_absolute_error_regressions"] == 0 for values in split_results.values()
            ),
            "same_cohort_as_reference": same_cohort,
            "no_failures_on_either_run": no_failures,
            "mae_not_worse_on_each_split": no_mae_regression,
            "rmse_not_worse_on_each_split": no_rmse_regression,
            "no_individual_absolute_error_regressions": all(
                values["per_photo_absolute_error_regressions"] == 0
                for values in split_results.values()
            ),
        },
        "omitted_from_candidate": omitted_from_candidate,
        "outside_reference": outside_reference,
        "input_or_label_mismatches": input_mismatches,
        "candidate_minus_reference_signed_prediction_mean": statistics.mean(
            new["prediction"] - old["prediction"] for new, old in paired
        ),
        "reference_quality_by_split": {
            split: metrics for split, metrics in reference.get("quality_by_split", {}).items()
            if split in selected_splits
        },
    }


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")

    import cv2
    import numpy as np
    import onnxruntime as ort

    import step_7_i_experimental_migraphx_server as server

    experiment_id = args.experiment_id
    if not experiment_id.startswith("v") or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in experiment_id):
        raise ValueError(f"Invalid experiment ID: {experiment_id!r}")
    experiment_dir = HERE / "experiment-versions" / experiment_id
    experiment_file = experiment_dir / "experiment.py"
    manifest_file = experiment_dir / "experiment.json"
    if not experiment_file.is_file() or not manifest_file.is_file():
        raise FileNotFoundError(f"Experiment version must contain experiment.py and experiment.json: {experiment_dir}")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if str(experiment_dir) not in sys.path:
        sys.path.insert(0, str(experiment_dir))
    experiment = ModuleType(f"distil_fp32_{experiment_id}")
    experiment.__file__ = str(experiment_file)
    exec(compile(experiment_file.read_text(encoding="utf-8"), str(experiment_file), "exec"), experiment.__dict__)

    if server.EXPERIMENT != "distilled-b0-migraphx-fp32":
        raise RuntimeError("This benchmark is scoped to Distil-FP32; set WAVE_EXPERIMENT=distilled-b0-migraphx-fp32")
    if manifest.get("id") != experiment_id:
        raise RuntimeError(f"Experiment manifest ID does not match directory: {manifest.get('id')!r}")
    splits = ["test"] if args.final_test else ["validation"]
    server_started = time.perf_counter()
    warmup = initialize_models_for_steps(server, manifest.get("required_segmentation_steps", [1, 2]))
    warmup["python_and_runtime_imports_ms"] = (server_started - PROCESS_START) * 1000.0
    warmup["initialize_models_total_ms"] = (time.perf_counter() - server_started) * 1000.0
    warmup["process_start_to_models_ready_ms"] = (time.perf_counter() - PROCESS_START) * 1000.0

    raw_dir = ROOT / "step-0-raw-data"
    all_rows = [item for split in splits for item in load_split(split, raw_dir, args.limit)]
    photos = []
    for item in all_rows:
        row = {"split": item["split"], "label_filename": item["label_filename"], "label": item["label"],
               "source_filename": item["photo"].name}
        started = time.perf_counter()
        try:
            measured = experiment.run_pipeline(item["photo"], server, run_instrumented)
            row.update(measured)
            row["failure"] = None
        except Exception as error:
            row["prediction"] = None
            row["failure"] = f"{type(error).__name__}: {error}"
            row["total_raw_to_score_ms"] = (time.perf_counter() - started) * 1000.0
        row["source_size_bytes"] = item["photo"].stat().st_size
        row["source_sha256"] = sha256_file(item["photo"])
        photos.append(row)
        print(f"{item['split']} {item['label_filename']}: "
              f"{row.get('total_raw_to_score_ms', 0.0):.2f} ms"
              + (f" ERROR {row['failure']}" if row["failure"] else ""), flush=True)

    quality_by_split = {}
    latency_by_split = {}
    for split in splits:
        split_rows = [row for row in photos if row["split"] == split]
        quality_by_split[split] = quality_summary(split_rows)
        latency_by_split[split] = {
            key: summarize([row[key] for row in split_rows if row.get(key) is not None])
            for key in ("preprocessing_total_ms", "waviness_inference_total_ms", "total_raw_to_score_ms")
        }
        timing_keys = sorted({key for row in split_rows for key in row.get("timings_ms", {})})
        latency_by_split[split]["operations"] = {
            key: summarize([row["timings_ms"][key] for row in split_rows if key in row.get("timings_ms", {})])
            for key in timing_keys
        }

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "experiment_name": manifest.get("name", experiment_id),
        "experiment_hypothesis": manifest.get("hypothesis"),
        "experiment_status": manifest.get("status", "active"),
        "candidate": "Distil-FP32 MIGraphX FP32 + v18 EfficientNet-B0 CPU regressor",
        "top_level_categories": ["warmup", "preprocessing", "waviness_inference"],
        "warmup": warmup,
        "dataset": {
            "splits": splits,
            "source_directory": str(raw_dir),
            "split_snapshot": SPLIT_SNAPSHOT,
            "split_manifest_sha256": sha256_file(ROOT / "step-3-dataset-splits/snapshots" / SPLIT_SNAPSHOT / "manifest.json"),
            "split_file_sha256s": {
                split: sha256_file(ROOT / "step-3-dataset-splits/snapshots" / SPLIT_SNAPSHOT / f"{split}.csv")
                for split in splits
            },
            "split_counts": {split: sum(row["split"] == split for row in photos) for split in splits},
            "limit_per_split": args.limit,
            "labels_sha256": sha256_file(ROOT / "labels.csv"),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "onnxruntime": ort.__version__,
            "execution_providers": ort.get_available_providers(),
            "step1_model": str(server.STEP1_ONNX),
            "step2_model": str(server.STEP2_ONNX),
            "regressor_model": str(server.REGRESSOR_ONNX),
            "segmentation_steps_loaded": manifest.get("required_segmentation_steps", [1, 2]),
            "step1_sha256": sha256_file(server.STEP1_ONNX),
            "step2_sha256": sha256_file(server.STEP2_ONNX),
            "regressor_sha256": sha256_file(server.REGRESSOR_ONNX),
        },
        "quality_by_split": quality_by_split,
        "latency_by_split": latency_by_split,
        "comparison": compare_reference({"photos": photos, "splits": splits}, args.compare_reference) if args.compare_reference else None,
        "failure_count": sum(row["failure"] is not None for row in photos),
        "photos": photos,
    }
    output_dir = args.output_dir.resolve() / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"distil-fp32-fine-grained-{run_id}.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {output_path}", flush=True)
    print(json.dumps({"quality_by_split": quality_by_split, "latency_by_split": latency_by_split}, indent=2))
    return 0 if report["failure_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
