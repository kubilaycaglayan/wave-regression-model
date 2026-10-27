#!/usr/bin/env python3
"""Benchmark one Step 7 prediction per photo without changing prediction outputs."""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
REPORT_DIR = PROJECT_DIR / "step-7-prediction-benchmarks"


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("photos", nargs="*", type=Path, help="HEIC, HEIF, JPEG or PNG files; defaults to the prediction holder")
    parser.add_argument("--repeat", type=positive_int, default=3, help="independent cold processes and warm runs (default: 3)")
    parser.add_argument("--device", default=None, help="PyTorch device, e.g. cpu or cuda")
    parser.add_argument("--output-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


class Recorder:
    def __init__(self, torch):
        self.torch = torch
        self.timings = {}
        self.segmentations = []
        self.phase = ""

    def sync(self, device):
        if device is not None and self.torch.device(device).type == "cuda":
            self.torch.cuda.synchronize(device)

    @contextmanager
    def measure(self, name, device=None):
        self.sync(device)
        started = time.perf_counter()
        try:
            yield
        finally:
            self.sync(device)
            key = f"{self.phase}.{name}" if self.phase else name
            self.timings[key] = self.timings.get(key, 0.0) + time.perf_counter() - started

    def segmentation_metadata(self, width, height, tiles, accumulator, device):
        self.segmentations.append(dict(phase=self.phase, width=width, height=height,
                                       tiles=tiles, accumulator_device=accumulator, device=device))


def wrap(module, name, recorder, label, device=None):
    original = getattr(module, name)
    def measured(*args, **kwargs):
        selected = device() if callable(device) else device
        with recorder.measure(label, selected):
            return original(*args, **kwargs)
    setattr(module, name, measured)


def collect_environment(torch):
    """Collect reproducibility metadata after timing has finished."""
    packages = {}
    for name in ("torch", "torchvision", "lightning", "torchmetrics", "transformers", "pillow", "pillow-heif", "numpy"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None

    cpu_model = platform.processor() or None
    memory_total = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.partition(":")[2].strip()
                break
    except OSError:
        pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                memory_total = int(line.split()[1]) * 1024
                break
    except (OSError, ValueError, IndexError):
        pass

    accelerators = []
    try:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            accelerators.append({"index": index, "name": properties.name,
                                 "total_memory_bytes": properties.total_memory,
                                 "architecture": getattr(properties, "gcnArchName", None),
                                 "compute_capability": [properties.major, properties.minor]})
    except Exception as error:
        accelerators.append({"error": f"{type(error).__name__}: {error}"})

    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_DIR, check=True,
                                  capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=normal"],
                                    cwd=PROJECT_DIR, check=True, capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    try:
        cudnn_version = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    except RuntimeError:
        cudnn_version = None
    return {
        "system": {"platform": platform.platform(), "architecture": platform.machine(),
                   "cpu_model": cpu_model, "logical_cpus": os.cpu_count(), "memory_total_bytes": memory_total},
        "accelerators": accelerators,
        "software": {"python": platform.python_version(), "packages": packages,
                     "cuda_runtime": torch.version.cuda, "rocm_runtime": torch.version.hip,
                     "cudnn_version": cudnn_version},
        "runtime": {"torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
                    "cudnn_enabled": torch.backends.cudnn.enabled,
                    "thread_environment": {key: os.environ[key] for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS") if key in os.environ}},
        "repository": {"commit": revision, "dirty": dirty},
    }


def collect_photo_specs(photos, prediction):
    """Identify inputs without embedding source image data in the report."""
    from PIL import Image
    specifications = []
    for photo in photos:
        try:
            with Image.open(photo) as image:
                size, image_format = list(image.size), image.format
            specifications.append({"path": str(photo), "size_bytes": photo.stat().st_size,
                                   "sha256": prediction.sha256_for(photo), "dimensions": size, "format": image_format})
        except Exception as error:
            specifications.append({"path": str(photo), "error": f"{type(error).__name__}: {error}"})
    return specifications


def worker(args, worker_started):
    import torch
    import step_7_a_predict as prediction
    import step_1_a_water_segmentation as segmentation
    import step_1_b_preprocess_water_inputs as step1
    import step_2_a_reduce_black_water_area as step2

    imported = time.perf_counter()
    recorder = Recorder(torch)
    segmentation.BENCHMARK_OBSERVER = recorder
    photos = args.photos or prediction.find_input_photos()
    photos = [p.resolve() for p in photos]
    for photo in photos:
        if not photo.is_file() or photo.suffix.lower() not in prediction.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported or missing photo: {photo}")
    checkpoint = prediction.checkpoint_path_from_env()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Selected checkpoint is missing: {checkpoint}")
    with recorder.measure("checkpoint_fingerprint"):
        digest = prediction.sha256_for(checkpoint)
    with recorder.measure("device_setup"):
        device = segmentation.configure_inference_device(args.device)
    requested_device = str(device)
    preview_paths = {photo: prediction.preview_path_for(photo) for photo in photos}
    seg_model = None
    if any(not path.is_file() for path in preview_paths.values()):
        with recorder.measure("load_segmentation_model", device):
            seg_model = segmentation.load_model(str(device), verbose=False)
    with recorder.measure("load_regression_checkpoint", device):
        model = prediction.WaveRegressionModel.load_from_checkpoint(checkpoint, map_location=device, pretrained=False)
        model.to(device).eval()
    wrap(model, "forward", recorder, "model_forward", lambda: device)
    base_transform = prediction.build_evaluation_transform()
    def transform(image):
        with recorder.measure("transform"):
            return base_transform(image)
    # Wrap the canonical pipeline calls, rather than reproducing preprocessing logic.
    wrap(prediction, "load_rgb_image", recorder, "decode_input")
    def wrap_phase(name, phase):
        original = getattr(prediction, name)
        def measured(*values, **kwargs):
            previous = recorder.phase
            recorder.phase = phase
            try:
                with recorder.measure("total", seg_model.device):
                    return original(*values, **kwargs)
            finally:
                recorder.phase = previous
        setattr(prediction, name, measured)
    wrap_phase("preprocess_raw_image", "step1")
    wrap_phase("preprocess_step_1_image", "step2")
    wrap(prediction, "encode_step_1_output", recorder, "step1_jpeg_encode")
    wrap(prediction, "encode_step_2_output", recorder, "step2_jpeg_encode")
    wrap(prediction, "decode_rgb_bytes", recorder, "jpeg_decode")
    wrap(step1, "run_segmentation", recorder, "segmentation", lambda: seg_model.device)
    wrap(step2, "run_segmentation", recorder, "segmentation", lambda: seg_model.device)
    wrap(step1, "standardized_model_input", recorder, "standardize")
    wrap(step2, "select_crop", recorder, "crop_selection")
    wrap(step2, "fit_in_square", recorder, "crop_fit")
    startup = {"imports_seconds": imported - worker_started, **recorder.timings}
    runs = []
    with tempfile.TemporaryDirectory(prefix="wave-prediction-benchmark-") as tmp:
        prediction.PREVIEW_DIR = Path(tmp) / "previews"
        prediction.PREDICTIONS_DIR = Path(tmp) / "predictions"
        for iteration in ("cold", "warm"):
            for photo_index, photo in enumerate(photos):
                prediction.PREVIEW_DIR = Path(tmp) / f"photo-{photo_index}" / "previews"
                prediction.PREDICTIONS_DIR = Path(tmp) / f"photo-{photo_index}" / "predictions"
                recorder.timings = {}
                recorder.segmentations = []
                recorder.phase = ""
                started = time.perf_counter()
                entry = {"kind": "batch" if iteration == "cold" and photo_index else iteration, "photo": str(photo), "status": "ok", "events": []}
                try:
                    preview_path = preview_paths[photo]
                    reused_preview = preview_path.is_file()
                    entry["input_path_type"] = "cached" if reused_preview else "fresh"
                    recorder.phase = entry["input_path_type"]
                    if reused_preview:
                        with recorder.measure("load_preview"):
                            model_input = prediction.load_preview(preview_path)
                    else:
                        model_input, encoded = prediction.preprocess_photo(photo, seg_model)
                        with recorder.measure("save_preview"):
                            prediction.save_preview(photo, encoded)
                    with recorder.measure("regression", device):
                        try:
                            value, _ = prediction.predict(model_input, device, model, transform)
                        except RuntimeError as error:
                            if device.type != "cuda" or not any(s in str(error).lower() for s in ("engine to execute", "cudnn", "cuda out of memory")):
                                raise
                            entry["events"].append(f"regression fallback to CPU: {error}")
                            device = torch.device("cpu")
                            model.to(device)
                            value, _ = prediction.predict(model_input, device, model, transform)
                    with recorder.measure("write_prediction"):
                        prediction.append_prediction_record(photo, checkpoint, digest, value, device, {})
                    entry["prediction_seconds"] = time.perf_counter() - started
                    entry["prediction"] = value
                    entry["regression_device"] = str(device)
                except Exception as error:
                    entry.update(status="failed", error=f"{type(error).__name__}: {error}", elapsed_seconds=time.perf_counter() - started)
                if seg_model is not None and str(seg_model.device) != requested_device:
                    entry["events"].append(f"segmentation device changed to {seg_model.device} (CUDA fallback)")
                if entry["kind"] == "cold":
                    entry["startup_to_first_photo_seconds"] = time.perf_counter() - worker_started
                entry["segmentation_device"] = str(seg_model.device) if seg_model is not None else None
                entry["timings"] = recorder.timings.copy()
                entry["segmentations"] = recorder.segmentations.copy()
                runs.append(entry)
                print(f"{entry['kind']} {photo.name}: {entry['status']} {entry.get('prediction_seconds', entry.get('elapsed_seconds', 0)):.3f}s" + (f" — {entry['error']}" if entry["status"] == "failed" else ""), flush=True)
    result = {"startup": startup,
              "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": digest,
              "requested_device": requested_device, "segmentation_model": segmentation.MODEL_NAME,
              "regression_backbone": model.backbone_name, "checkpoint_size_bytes": checkpoint.stat().st_size,
              "environment": collect_environment(torch), "input_photos": collect_photo_specs(photos, prediction),
              "settings": {"resolution": segmentation.SEGMENTATION_RESOLUTION, "overlapping_tiles": segmentation.USE_OVERLAPPING_TILES,
                           "tile_size": segmentation.TILE_SIZE, "tile_overlap": segmentation.TILE_OVERLAP}, "runs": runs}
    args.worker_output.write_text(json.dumps(result), encoding="utf-8")


def summarize(runs):
    groups = {}
    for run in runs:
        if run["status"] != "ok":
            continue
        if "prediction_seconds" in run:
            totals = {f"{run['input_path_type']}_total": run["prediction_seconds"]}
        else:  # Preserve compatibility when summarizing historical reports.
            totals = {"fresh_total": run["fresh_seconds"], "cached_total": run["cached_seconds"]}
        for name, value in {**totals, **run["timings"]}.items():
            groups.setdefault((run["kind"], name), []).append(value)
    return [{"kind": kind, "stage": name, "samples": len(values), "median_seconds": statistics.median(values),
             "min_seconds": min(values), "max_seconds": max(values)}
            for (kind, name), values in sorted(groups.items())]


def markdown_cell(value):
    """Keep arbitrary file names and error messages inside one Markdown cell."""
    if value is None:
        return "Not recorded"
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def run_label(kind):
    """Make cold and warm runs easy to distinguish in reports."""
    return {"cold": "❄️ cold", "warm": "🔥 warm"}.get(kind, kind)


def summary_tables(entries, heading):
    """Display single measurements once and aggregate repeated measurements separately."""
    lines = []
    for repeated in (False, True):
        rows = [(item, label) for item, label in entries if (item["samples"] > 1) == repeated]
        if not rows:
            continue
        if lines:
            lines.append("")
        if repeated:
            lines += [f"| Run | {heading} | Median (s) | Min (s) | Max (s) | Samples |",
                      "| --- | --- | ---: | ---: | ---: | ---: |"]
        else:
            lines += [f"| Run | {heading} | Duration (s) |", "| --- | --- | ---: |"]
        for item, label in rows:
            prefix = f"| {markdown_cell(run_label(item['kind']))} | {markdown_cell(label)} |"
            if repeated:
                lines.append(f"{prefix} {item['median_seconds']:.6f} | {item['min_seconds']:.6f} | {item['max_seconds']:.6f} | {item['samples']} |")
            else:
                lines.append(f"{prefix} {item['median_seconds']:.6f} |")
    return lines


def report_markdown(report):
    """Render totals first, followed by timing details and configuration."""
    lines = ["# Prediction benchmark", "", f"Created (UTC): {markdown_cell(report.get('created_utc'))}",
             "", "Durations are in seconds. Parent timings include their child measurements.",
             "Cold is the first photo in a new process; warm runs reuse loaded models.", ""]
    processes = report.get("processes", [])
    summary = report.get("summary", [])
    major_stages = (
        ("fresh_total", "Fresh prediction total"),
        ("step1.total", "├─ Step 1: water preprocessing"),
        ("step2.total", "├─ Step 2: lower-water preprocessing"),
        ("fresh.regression", "└─ Fresh regression prediction"),
        ("cached_total", "Cached prediction total"),
        ("cached.regression", "└─ Cached regression prediction"),
    )
    lines += ["## Overall timings", "", "### Prediction totals and main stages", "",
              "Each trace uses either fresh preprocessing or an existing preview. Indented stages are included in their parent total; smaller steps are shown in the detailed traces." if report.get("report_version", 1) >= 3 else "Fresh and cached predictions are separate measurements. Indented stages are included in their parent total; smaller steps are shown in the detailed traces.", ""]
    if summary:
        entries = [(item, label) for kind in ("cold", "batch", "warm")
                   for stage, label in major_stages for item in summary
                   if item["kind"] == kind and item["stage"] == stage]
        lines += summary_tables(entries, "Measurement")
    else:
        lines.append("No successful runs were available for the summary.")
    lines += ["", "## Process breakdown", "",
              "Each process wall time covers startup, all predictions in that process, and other work such as report metadata collection. Startup occurs before the predictions and is not included in their totals.", "",
              "Each prediction table shows measured parent and child timings. Children may omit unmeasured work, and parent timings include their children; do not add every row together."]
    startup_labels = {"imports_seconds": "Python imports", "checkpoint_fingerprint": "Checkpoint fingerprint",
                      "device_setup": "Device setup", "load_segmentation_model": "Load segmentation model",
                      "load_regression_checkpoint": "Load regression checkpoint"}
    for index, process in enumerate(processes, 1):
        lines += ["", f"### Process {index}", ""]
        if "process_wall_seconds" in process:
            lines += [f"**Process wall time: {process['process_wall_seconds']:.6f} s**", ""]
        lines += ["#### Startup (before predictions)", "",
                  "These are individual startup measurements; other startup work is not separately timed.", "",
                  "| Measurement | Duration (s) |", "| --- | ---: |"]
        for name, seconds in process.get("startup", {}).items():
            lines.append(f"| {markdown_cell(startup_labels.get(name, name))} | {seconds:.6f} |")
        lines += ["", "#### Predictions"]
        for run_index, run in enumerate(process.get("runs", []), 1):
            lines += ["", f"##### Prediction {run_index}: {markdown_cell(run_label(run.get('kind')))} — {markdown_cell(run.get('photo'))}", "",
                      f"Status: **{markdown_cell(run.get('status'))}**", ""]
            for name in ("error", "input_path_type", "segmentation_device", "regression_device", "prediction", "cached_prediction"):
                if name in run:
                    lines.append(f"- {name}: {markdown_cell(run[name])}")
            for event in run.get("events", []):
                lines.append(f"- Event: {markdown_cell(event)}")
            if run.get("segmentations"):
                lines.append(f"- Segmentation details: {markdown_cell(run['segmentations'])}")
            if "startup_to_first_photo_seconds" in run:
                lines += ["", f"Startup through this prediction: {run['startup_to_first_photo_seconds']:.6f} s (measured inside the worker; excludes process launch)."]
            lines += ["", "| Measurement | Duration (s) |", "| --- | ---: |"]
            timings = run.get("timings", {})
            used = set()
            def timing(key, label, depth=0):
                if key in timings:
                    used.add(key)
                    prefix = "&nbsp;" * (depth * 4) + ("↳ " if depth else "")
                    lines.append(f"| {prefix}{markdown_cell(label)} | {timings[key]:.6f} |")
            def total(key, label):
                if key in run:
                    lines.append(f"| **{label}** | **{run[key]:.6f}** |")
            def preprocessing(phase, label):
                phase_keys = [key for key in timings if key.startswith(phase + ".")]
                if phase_keys and f"{phase}.total" not in timings:
                    lines.append(f"| &nbsp;&nbsp;&nbsp;&nbsp;↳ {label} (total not recorded) | — |")
                timing(f"{phase}.total", label, 1)
                segmentation_parts = ("tile_prepare", "tile_model", "tile_accumulate", "mask_finalize")
                if any(f"{phase}.{part}" in timings for part in segmentation_parts) and f"{phase}.segmentation" not in timings:
                    lines.append("| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↳ Segmentation (total not recorded) | — |")
                timing(f"{phase}.segmentation", "Segmentation", 2)
                for part, name in (("tile_prepare", "Prepare tiles"), ("tile_model", "Run tile model"),
                                   ("tile_accumulate", "Resize and accumulate tiles"), ("mask_finalize", "Finalize mask")):
                    timing(f"{phase}.{part}", name, 3)
                if phase == "step1":
                    timing("step1.standardize", "Mask and standardize", 2)
                else:
                    timing("step2.crop_selection", "Select lower-water crop", 2)
                    timing("step2.crop_fit", "Fit crop to model input", 2)
            def regression(phase):
                timing(f"{phase}.regression", "Regression prediction", 1)
                timing(f"{phase}.transform", "Transform input", 2)
                timing(f"{phase}.model_forward", "Model forward pass", 2)
            path = run.get("input_path_type", "fresh")
            if "prediction_seconds" in run:
                total("prediction_seconds", f"{path.capitalize()} prediction total")
            elif "fresh_seconds" in run:
                total("fresh_seconds", "Fresh prediction total")
            elif "elapsed_seconds" in run:
                total("elapsed_seconds", "Elapsed before failure")
            timing("fresh.decode_input", "Decode input photo", 1)
            preprocessing("step1", "Step 1: water preprocessing")
            timing("fresh.step1_jpeg_encode", "Encode Step 1 JPEG", 1)
            preprocessing("step2", "Step 2: lower-water preprocessing")
            timing("fresh.step2_jpeg_encode", "Encode Step 2 JPEG", 1)
            timing("fresh.jpeg_decode", "Decode JPEGs (both steps combined)", 1)
            timing("fresh.save_preview", "Save preview", 1)
            if path == "cached" and "prediction_seconds" in run:
                timing("cached.load_preview", "Load existing preview", 1)
                regression("cached")
            else:
                regression("fresh")
            timing("fresh.write_prediction", "Write prediction record", 1)
            timing("cached.write_prediction", "Write prediction record", 1)
            # Older reports measured an additional cached prediction after the fresh one.
            if "cached_seconds" in run:
                total("cached_seconds", "Separate cached prediction total (historical benchmark)")
                timing("cached.load_preview", "Load existing preview", 1)
                regression("cached")
            for key, seconds in timings.items():
                if key not in used:
                    lines.append(f"| ↳ Other measured trace: {markdown_cell(key)} | {seconds:.6f} |")
    lines += ["", "## Configuration and specifications", ""]
    if report.get("configuration"):
        lines += [f"Run configuration: {markdown_cell(report['configuration'])}", ""]
    for index, process in enumerate(processes, 1):
        lines += [f"### Process {index}", "", "| Field | Value |", "| --- | --- |"]
        for key in ("checkpoint", "checkpoint_sha256", "checkpoint_size_bytes", "regression_backbone", "segmentation_model", "requested_device", "settings", "environment", "input_photos"):
            if key in process:
                value = process[key]
                if isinstance(value, dict):
                    for child, child_value in value.items():
                        lines.append(f"| {markdown_cell(key + '.' + child)} | {markdown_cell(child_value)} |")
                else:
                    lines.append(f"| {markdown_cell(key)} | {markdown_cell(value)} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_report_pair(report, report_path):
    """Write a JSON report and its matching Markdown rendering."""
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path = report_path.with_suffix(".md")
    markdown_path.write_text(report_markdown(report), encoding="utf-8")
    return markdown_path


def main():
    started = time.perf_counter()
    args = parse_args()
    if args.worker_output:
        worker(args, started)
        return
    results = []
    with tempfile.TemporaryDirectory(prefix="wave-benchmark-results-") as tmp:
        for index in range(args.repeat):
            output = Path(tmp) / f"{index}.json"
            command = [sys.executable, str(Path(__file__).resolve()), "--worker-output", str(output)]
            if args.device:
                command += ["--device", args.device]
            command += [str(photo.resolve()) for photo in args.photos]
            print(f"Benchmark process {index + 1}/{args.repeat}", flush=True)
            process_started = time.perf_counter()
            completed = subprocess.run(command, cwd=PROJECT_DIR, check=False)
            elapsed = time.perf_counter() - process_started
            if completed.returncode or not output.is_file():
                raise RuntimeError(f"Benchmark process {index + 1} failed (exit {completed.returncode})")
            result = json.loads(output.read_text(encoding="utf-8"))
            result["process_wall_seconds"] = elapsed
            results.append(result)
    runs = [run for result in results for run in result["runs"]]
    summary = summarize(runs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc)
    report_dir = args.output_dir / timestamp.strftime("%Y%m%dT%H%M%S%fZ")
    report_dir.mkdir(exist_ok=False)
    report_path = report_dir / "benchmark.json"
    report = {"report_version": 3, "created_utc": timestamp.isoformat(), "configuration": {"repeat": args.repeat, "device_argument": args.device, "photos_argument": [str(path.resolve()) for path in args.photos]}, "processes": results, "summary": summary}
    markdown_path = write_report_pair(report, report_path)
    print("\nStartup median times (seconds):")
    for key in results[0]["startup"]:
        values = [result["startup"][key] for result in results]
        print(f"  {key:30} {statistics.median(values):9.3f}  range {min(values):.3f}–{max(values):.3f}")
    print("\nMedian stage times (seconds; nested stages overlap):")
    for item in summary:
        print(f"  {item['kind']:4} {item['stage']:30} {item['median_seconds']:9.3f}  range {item['min_seconds']:.3f}–{item['max_seconds']:.3f} (n={item['samples']})")
    print(f"Reports: {report_path.resolve()} and {markdown_path.resolve()}")
    if any(run["status"] != "ok" for run in runs):
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Benchmark error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
