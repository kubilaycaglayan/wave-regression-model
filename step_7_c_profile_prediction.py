#!/usr/bin/env python3
"""Profile the two segmentation passes and regression forward pass for one photo."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
PROFILE_DIR = PROJECT_DIR / "step-7-prediction-profiles"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("photo", type=Path, help="One raw HEIC, HEIF, JPEG, or PNG photo")
    parser.add_argument("--device", default=None, help="PyTorch device, e.g. cpu or cuda")
    parser.add_argument("--output-dir", type=Path, default=PROFILE_DIR)
    return parser.parse_args()


def operator_rows(profiler):
    """Return self times in microseconds; CPU and device time can overlap."""
    rows = []
    for event in profiler.key_averages():
        rows.append({"operator": event.key, "calls": event.count,
                     "self_cpu_time_us": event.self_cpu_time_total,
                     "self_device_time_us": getattr(event, "self_device_time_total", None)})
    return sorted(rows, key=lambda row: row["self_cpu_time_us"], reverse=True)


def trace_has_device_kernels(path):
    """Inspect the trace rather than assuming a requested accelerator was recorded."""
    try:
        with path.open(encoding="utf-8") as handle:
            trace = json.load(handle)
        return any(str(event.get("cat", "")).lower() in {"kernel", "gpu_kernel"}
                   for event in trace.get("traceEvents", []) if isinstance(event, dict))
    except (OSError, ValueError, AttributeError):
        return False


def capture_pass(name, output_dir, torch, operation, device):
    """Profile one canonical operation and export even if it fails."""
    activities = [torch.profiler.ProfilerActivity.CPU]
    requested_device = torch.device(device).type == "cuda"
    if requested_device and torch.profiler.ProfilerActivity.CUDA in torch.profiler.supported_activities():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    capture = {"name": name, "device_at_start": str(device), "status": "ok",
               "accelerator_requested": requested_device,
               "accelerator_activity_enabled": len(activities) > 1}
    trace_path = output_dir / f"{name}.trace.json"
    attempt = 2
    while trace_path.exists():
        trace_path = output_dir / f"{name}-attempt-{attempt}.trace.json"
        attempt += 1
    profiler = None
    error = None
    try:
        with torch.profiler.profile(activities=activities, record_shapes=False,
                                    profile_memory=False, with_stack=False) as profiler:
            with torch.profiler.record_function(name):
                result = operation()
    except Exception as exc:
        error = exc
        capture.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    if profiler is not None:
        try:
            profiler.export_chrome_trace(str(trace_path))
            capture["trace"] = trace_path.name
            capture["accelerator_kernels_in_trace"] = trace_has_device_kernels(trace_path) if requested_device else None
            capture["operators"] = operator_rows(profiler)
        except Exception as exc:
            capture["export_error"] = f"{type(exc).__name__}: {exc}"
            capture["status"] = "failed"
            error = error or exc
    if requested_device and not capture.get("accelerator_kernels_in_trace"):
        capture["accelerator_warning"] = "No accelerator kernel events were found in the trace; device tracing may be unavailable or incomplete."
    return capture, None if error else result, error


def markdown_cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def render_report(report):
    lines = ["# PyTorch prediction operation profile", "", f"Created (UTC): {report['created_utc']}", "",
             "Profiler timings include instrumentation overhead and are not prediction latency measurements. "
             "Self time excludes time in child operators; CPU and accelerator times may overlap.", "",
             f"Status: **{report['status']}**", "", "## Run details", "", "| Field | Value |", "| --- | --- |"]
    for name in ("photo", "checkpoint", "checkpoint_sha256", "checkpoint_size_bytes", "regression_backbone", "segmentation_model", "requested_device", "settings", "environment", "input_photo"):
        if name in report:
            value = report[name]
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            lines.append(f"| {name} | {markdown_cell(value)} |")
    if report.get("error"):
        lines += ["", f"**Error:** {markdown_cell(report['error'])}"]
    for capture in report.get("passes", []):
        lines += ["", f"## {markdown_cell(capture['name'].replace('_', ' ').title())}", "",
                  f"Status: **{capture['status']}**; device at start: {markdown_cell(capture['device_at_start'])}."]
        if capture.get("device_at_end"):
            lines.append(f"Device at end: {markdown_cell(capture['device_at_end'])}.")
        if capture.get("segmentation"):
            lines.append(f"Segmentation: {markdown_cell(json.dumps(capture['segmentation']))}")
        for key in ("accelerator_warning", "error", "export_error"):
            if capture.get(key):
                lines.append(f"- {markdown_cell(capture[key])}")
        if capture.get("trace"):
            lines += ["", f"[Open trace file]({capture['trace']}) in Chrome's `chrome://tracing` or Perfetto."]
        rows = capture.get("operators", [])
        if rows:
            device = capture.get("accelerator_kernels_in_trace")
            lines += ["", "Top 30 operators by self CPU time. The JSON report contains all operators.", "",
                      "| Operator | Calls | Self CPU (ms) | Self accelerator (ms) |", "| --- | ---: | ---: | ---: |"]
            for row in rows[:30]:
                device_time = row.get("self_device_time_us")
                accelerator = f"{device_time / 1000:.3f}" if device and device_time is not None else "Not available"
                lines.append(f"| {markdown_cell(row['operator'])} | {row['calls']} | {row['self_cpu_time_us'] / 1000:.3f} | {accelerator} |")
            if device:
                ranked = sorted(rows, key=lambda row: row.get("self_device_time_us") or 0, reverse=True)[:30]
                lines += ["", "Top 30 operators by self accelerator time:", "",
                          "| Operator | Calls | Self accelerator (ms) | Self CPU (ms) |", "| --- | ---: | ---: | ---: |"]
                for row in ranked:
                    lines.append(f"| {markdown_cell(row['operator'])} | {row['calls']} | {(row.get('self_device_time_us') or 0) / 1000:.3f} | {row['self_cpu_time_us'] / 1000:.3f} |")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    import torch
    import step_7_a_predict as prediction
    import step_1_a_water_segmentation as segmentation
    import step_1_b_preprocess_water_inputs as step1
    import step_2_a_reduce_black_water_area as step2
    from step_7_b_benchmark_prediction import collect_environment, collect_photo_specs

    photo = args.photo.resolve()
    if not photo.is_file() or photo.suffix.lower() not in prediction.SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported or missing photo: {photo}")
    checkpoint = prediction.checkpoint_path_from_env()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Selected checkpoint is missing: {checkpoint}")
    timestamp = datetime.now(timezone.utc)
    output_dir = args.output_dir / timestamp.strftime("%Y%m%dT%H%M%S%fZ")
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {"report_version": 1, "created_utc": timestamp.isoformat(), "status": "ok",
              "photo": str(photo), "checkpoint": str(checkpoint.resolve()),
              "checkpoint_sha256": prediction.sha256_for(checkpoint), "checkpoint_size_bytes": checkpoint.stat().st_size,
              "segmentation_model": segmentation.MODEL_NAME, "passes": []}
    try:
        device = segmentation.configure_inference_device(args.device)
        report["requested_device"] = str(device)
        seg_model = segmentation.load_model(str(device), verbose=False)
        model = prediction.WaveRegressionModel.load_from_checkpoint(checkpoint, map_location=device, pretrained=False)
        model.to(device).eval()
        report["regression_backbone"] = model.backbone_name
        report["settings"] = {"resolution": segmentation.SEGMENTATION_RESOLUTION,
                              "overlapping_tiles": segmentation.USE_OVERLAPPING_TILES,
                              "tile_size": segmentation.TILE_SIZE, "tile_overlap": segmentation.TILE_OVERLAP,
                              "record_shapes": False, "profile_memory": False, "with_stack": False}
        # Wrapping the canonical functions ensures both full segmentation passes are captured.
        class Observer:
            def __init__(self): self.metadata = []
            def segmentation_metadata(self, width, height, tiles, accumulator, actual):
                self.metadata.append(dict(width=width, height=height, tiles=tiles, accumulator_device=accumulator, device=actual))
            def measure(self, name, device):
                from contextlib import nullcontext
                return nullcontext()
        observer = Observer()
        segmentation.BENCHMARK_OBSERVER = observer
        def wrap_segmentation(module, name):
            original = module.run_segmentation
            def profiled(image, segmenter):
                before = len(observer.metadata)
                capture, result, error = capture_pass(name, output_dir, torch, lambda: original(image, segmenter), segmenter.device)
                capture["device_at_end"] = str(segmenter.device)
                capture["segmentation"] = observer.metadata[before:]
                report["passes"].append(capture)
                if error:
                    raise error
                return result
            module.run_segmentation = profiled
        wrap_segmentation(step1, "step1_segmentation")
        wrap_segmentation(step2, "step2_segmentation")
        original_forward = model.forward
        def profiled_forward(*values, **kwargs):
            capture, result, error = capture_pass("regression_forward", output_dir, torch,
                                                  lambda: original_forward(*values, **kwargs), device)
            capture["device_at_end"] = str(device)
            report["passes"].append(capture)
            if error:
                raise error
            return result
        model.forward = profiled_forward
        model_input, _ = prediction.preprocess_photo(photo, seg_model)
        try:
            result, _ = prediction.predict(model_input, device, model, prediction.build_evaluation_transform())
        except RuntimeError as error:
            if device.type != "cuda" or not any(x in str(error).lower() for x in ("engine to execute", "cudnn", "cuda out of memory")):
                raise
            print("Regression CUDA inference failed; retrying on CPU.", flush=True)
            device = torch.device("cpu")
            model.to(device)
            result, _ = prediction.predict(model_input, device, model, prediction.build_evaluation_transform())
        report["prediction"] = result
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    try:
        report["environment"] = collect_environment(torch)
        report["input_photo"] = collect_photo_specs([photo], prediction)[0]
    except Exception as error:
        report["metadata_error"] = f"{type(error).__name__}: {error}"
    (output_dir / "profile.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output_dir / "profile.md").write_text(render_report(report), encoding="utf-8")
    print(f"Profile report: {output_dir / 'profile.md'}", flush=True)
    if report["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Profile error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
