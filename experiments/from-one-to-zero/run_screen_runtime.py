#!/usr/bin/env python3
"""Run one isolated fast-v2.1 screening variant without the HTTP client."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summary(values: list[float]) -> dict:
    ordered = sorted(values)
    return {"count": len(ordered), "median_ms": statistics.median(ordered),
            "p95_ms": ordered[max(0, math.ceil(.95 * len(ordered)) - 1)],
            "min_ms": ordered[0], "max_ms": ordered[-1]}


def load_heif_helper(output_dir: Path):
    source = ROOT / "experiments/from-one-to-zero/heic-lowres-roi/heif_region_decode.cpp"
    site_packages = Path(sys.prefix) / "lib/python3.12/site-packages"
    libs = site_packages / "pillow_heif.libs"
    heif_lib = next(libs.glob("libheif*.so*"))
    helper = Path("/tmp") / f"wave-from-one-to-zero-heif-{sha(source)[:12]}.so"
    if not helper.exists():
        subprocess.run(["c++", "-O3", "-std=c++17", "-shared", "-fPIC", "-pthread",
                        f"-I{ROOT / 'native/libheif/include'}", str(source), str(heif_lib),
                        f"-Wl,-rpath,{libs}", "-o", str(helper)], check=True)
    lib = ctypes.CDLL(str(helper))
    byte_ptr = ctypes.POINTER(ctypes.c_ubyte)
    lib.wave_heif_decode_thumbnail_rgb.argtypes = [ctypes.c_char_p, ctypes.POINTER(byte_ptr),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int), ctypes.c_char_p, ctypes.c_size_t]
    lib.wave_heif_decode_region_rgb.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(byte_ptr),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_char_p, ctypes.c_size_t]
    lib.wave_heif_free_region.argtypes = [ctypes.c_void_p]
    return lib


def thumbnail(lib, path: Path):
    import numpy as np
    pointer = ctypes.POINTER(ctypes.c_ubyte)()
    w, h, sw, sh = (ctypes.c_int() for _ in range(4))
    error = ctypes.create_string_buffer(512)
    status = lib.wave_heif_decode_thumbnail_rgb(os.fsencode(path), ctypes.byref(pointer),
        ctypes.byref(w), ctypes.byref(h), ctypes.byref(sw), ctypes.byref(sh), error, len(error))
    if status:
        return None, None, None, None, error.value.decode(errors="replace")
    try:
        view = np.ctypeslib.as_array(pointer, shape=(h.value * w.value * 3,))
        pixels = view.reshape(h.value, w.value, 3).copy()
    finally:
        lib.wave_heif_free_region(pointer)
    return pixels, w.value, h.value, (sw.value, sh.value), None


def region(lib, path: Path, box):
    import numpy as np
    x0, y0, x1, y1 = box
    pointer = ctypes.POINTER(ctypes.c_ubyte)()
    w, h = ctypes.c_int(), ctypes.c_int()
    error = ctypes.create_string_buffer(512)
    status = lib.wave_heif_decode_region_rgb(os.fsencode(path), x0, y0, x1, y1, 16,
        ctypes.byref(pointer), ctypes.byref(w), ctypes.byref(h), error, len(error))
    if status:
        raise RuntimeError(error.value.decode(errors="replace"))
    try:
        view = np.ctypeslib.as_array(pointer, shape=(h.value * w.value * 3,))
        pixels = view.reshape(h.value, w.value, 3).copy()
    finally:
        lib.wave_heif_free_region(pointer)
    return pixels


def run_heic_roi(path: Path, server, helper: ctypes.CDLL) -> dict:
    import cv2
    import numpy as np
    from PIL import Image

    started = time.perf_counter()
    stages = {}
    tick = time.perf_counter()
    thumb, thumb_w, thumb_h, source_dims, error = thumbnail(helper, path)
    stages["thumbnail_decode_ms"] = (time.perf_counter() - tick) * 1000
    if thumb is None:
        result = server.run_fast_v2_1_pipeline(path)
        result["stages_ms"]["heic_candidate_mode"] = "full-resolution fallback: " + str(error)
        return result
    width, height = source_dims
    tick = time.perf_counter()
    tensor = server.processor_input(thumb, server.STEP1_SIZE)
    stages["step1_tensor_prep_ms"] = (time.perf_counter() - tick) * 1000
    tick = time.perf_counter()
    predicted_mask = server.STEP1_SESSION.run(["water_mask"], {"pixel_values": tensor})[0][0]
    stages["step1_segmentation_ms"] = (time.perf_counter() - tick) * 1000
    tick = time.perf_counter()
    mask_224 = cv2.resize(predicted_mask.astype(np.uint8), (224, 224),
                          interpolation=cv2.INTER_NEAREST_EXACT).astype(bool)
    stages["mask_projection_ms"] = (time.perf_counter() - tick) * 1000
    tick = time.perf_counter()
    selected = server.select_crop(mask_224, np.zeros((224, 224), dtype=bool))
    if selected is None:
        raise ValueError("Step 1 could not reliably select a lower-water crop")
    left, top, right, bottom = selected[0]
    stages["crop_selection_ms"] = (time.perf_counter() - tick) * 1000
    x0 = max(0, min(width, math.floor(left * width / 224)))
    x1 = max(x0 + 1, min(width, math.ceil(right * width / 224)))
    y0 = max(0, min(height, math.floor(top * height / 224)))
    y1 = max(y0 + 1, min(height, math.ceil(bottom * height / 224)))
    mask_roi = mask_224[top:bottom, left:right]
    tick = time.perf_counter()
    source_roi = region(helper, path, (x0, y0, x1, y1))
    stages["source_region_decode_ms"] = (time.perf_counter() - tick) * 1000
    tick = time.perf_counter()
    scale = min(224 / source_roi.shape[1], 224 / source_roi.shape[0])
    rw = max(1, min(224, round(source_roi.shape[1] * scale)))
    rh = max(1, min(224, round(source_roi.shape[0] * scale)))
    ox, oy = (224 - rw) // 2, (224 - rh) // 2
    canvas = np.zeros((224, 224, 3), dtype=np.uint8)
    canvas_roi = canvas[oy:oy + rh, ox:ox + rw]
    cv2.resize(source_roi, (rw, rh), dst=canvas_roi, interpolation=cv2.INTER_LANCZOS4)
    final_mask = cv2.resize(mask_roi.astype(np.uint8), (rw, rh), interpolation=cv2.INTER_NEAREST_EXACT)
    canvas_roi[final_mask == 0] = 0
    stages["final_canvas_resize_and_mask_ms"] = (time.perf_counter() - tick) * 1000
    tick = time.perf_counter()
    encoded = __import__("io").BytesIO()
    Image.fromarray(canvas, mode="RGB").save(encoded, format="JPEG", quality=95, subsampling=0)
    with Image.open(__import__("io").BytesIO(encoded.getvalue())) as decoded:
        decoded.load()
        final_rgb = np.asarray(decoded.convert("RGB"), dtype=np.uint8)
    stages["final_jpeg_roundtrip_ms"] = (time.perf_counter() - tick) * 1000
    tick = time.perf_counter()
    model_pixels = final_rgb.astype(np.float32) / np.float32(255.0)
    model_tensor = np.ascontiguousarray(((model_pixels - server.MEAN) / server.STD)
                                        .transpose(2, 0, 1)[None], dtype=np.float32)
    stages["regressor_tensor_prep_ms"] = (time.perf_counter() - tick) * 1000
    tick = time.perf_counter()
    score = float(server.REGRESSOR_SESSION.run(["scores"], {"images": model_tensor})[0].reshape(-1)[0])
    stages["regressor_inference_ms"] = (time.perf_counter() - tick) * 1000
    stages["decode_ms"] = stages["thumbnail_decode_ms"] + stages["source_region_decode_ms"]
    stages["preprocessing_total_ms"] = (time.perf_counter() - started) * 1000 - stages["regressor_inference_ms"]
    return {"prediction": score, "stages_ms": stages, "total_server_ms": (time.perf_counter() - started) * 1000,
            "image": {"width": width, "height": height, "thumbnail_width": thumb_w,
                      "thumbnail_height": thumb_h, "format": path.suffix.lower()},
            "intermediate": {"mask_crop_xyxy_224": [left, top, right, bottom],
                             "source_crop_xyxy": [x0, y0, x1, y1],
                             "regressor_tensor_shape": list(model_tensor.shape)},
            "candidate_mode": "embedded thumbnail + selected HEIF tile region"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--photos", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step1-model", type=Path)
    parser.add_argument("--step1-size", type=int, default=448)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile-prefix", type=Path)
    args = parser.parse_args()

    os.environ["WAVE_PIPELINE_PROFILE"] = "fast-v2.1-coordinate"
    os.environ["WAVE_STEP1_SIZE"] = str(args.step1_size)
    os.environ["WAVE_REGRESSOR_ONNX"] = str(ROOT / "models/fast-v2.1/wave-regression-efficientnet-b0-fast-v2.1.onnx")
    if args.step1_model:
        os.environ["WAVE_STEP1_ONNX"] = str(args.step1_model.resolve())
    if args.variant == "migraphx-fp16":
        os.environ["WAVE_MIGRAPHX_FP16"] = "1"
    if args.profile_prefix:
        os.environ["WAVE_STEP1_PROFILE_PREFIX"] = str(args.profile_prefix.resolve())

    import step_7_i_migraphx_server as server
    if args.variant == "heic-lowres-roi":
        helper = load_heif_helper(args.output.parent)
    else:
        helper = None
    setup = server.initialize_models()
    startup_ms = (time.perf_counter() - server.PROCESS_START) * 1000
    if args.profile_prefix:
        session_providers = server.STEP1_SESSION.get_providers()
    records = []
    runner = (lambda photo: run_heic_roi(photo, server, helper)) if helper else server.run_fast_v2_1_pipeline
    for photo in args.photos:
        runner(photo)  # per-photo warm-up, excluded
    for photo in args.photos:
        for repeat in range(1, args.repeats + 1):
            result = runner(photo)
            records.append({"photo": photo.name, "repeat": repeat, "status": "ok",
                            "prediction": result["prediction"],
                            "total_raw_to_score_ms": result["total_server_ms"],
                            "server_pipeline_ms": result["total_server_ms"],
                            "stages_ms": result["stages_ms"],
                            "crop": result.get("intermediate", {}).get("source_crop_xyxy"),
                            "candidate_mode": result.get("candidate_mode") or
                                result.get("stages_ms", {}).get("heic_candidate_mode")})
    profile = None
    if args.profile_prefix:
        profile_path = Path(server.STEP1_SESSION.end_profiling())
        events = json.loads(profile_path.read_text())
        node_events = sorted((e for e in events if e.get("cat") == "Node" and "dur" in e),
                             key=lambda e: e.get("ts", 0))
        by_node = {}
        for event in node_events:
            key = (event.get("args", {}).get("provider", "unknown"),
                   event.get("name", "unknown"))
            item = by_node.setdefault(key, {"provider": key[0], "operator": key[1],
                                            "event_count": 0, "total_ms": 0.0})
            item["event_count"] += 1
            item["total_ms"] += event["dur"] / 1000.0
        provider_assignments = {}
        for item in by_node.values():
            provider_assignments[item["provider"]] = provider_assignments.get(item["provider"], 0) + 1
        steady_events = node_events[1:]  # the first MIGraphX event includes graph compilation/warm-up
        steady_durations = [e["dur"] / 1000.0 for e in steady_events]
        profile = {"trace": str(profile_path), "provider_assignments": provider_assignments,
                   "graph_node_count": len(by_node),
                   "graph_node_names": [item["operator"] for item in by_node.values()],
                   "first_graph_event_ms": node_events[0]["dur"] / 1000.0 if node_events else None,
                   "steady_graph_event_ms": summary(steady_durations) if steady_durations else None,
                   "note": "ONNX Runtime fused the graph to one MIGraphX node, so this profile confirms provider placement but cannot rank internal operators. The first event includes compilation; subsequent node-event times are diagnostic and include profiler overhead."}

    report = {"variant": args.variant, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "runtime": "ONNX Runtime MIGraphX segmentation + ONNX CPU regressor",
              "step1_model": str(server.STEP1_ONNX), "step1_sha256": sha(Path(server.STEP1_ONNX)),
              "step1_size": args.step1_size, "fp16": os.environ.get("WAVE_MIGRAPHX_FP16") == "1",
              "startup_to_ready_ms": startup_ms, "model_setup": setup,
              "cohort": [{"filename": p.name, "bytes": p.stat().st_size, "sha256": sha(p)} for p in args.photos],
              "protocol": {"warmup_runs_per_photo": 1, "measured_runs_per_photo": args.repeats,
                           "measurement": "direct in-process server pipeline; excludes server initialization and HTTP overhead"},
              "latency_summary": {"server_pipeline_ms": summary([r["server_pipeline_ms"] for r in records])},
              "runs": records, "profile": profile,
              "prediction_summary": {"min": min(r["prediction"] for r in records),
                                     "max": max(r["prediction"] for r in records)}}
    if args.profile_prefix:
        report["providers"] = session_providers
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"variant": args.variant, "output": str(args.output),
                      "median_ms": report["latency_summary"]["server_pipeline_ms"]["median_ms"],
                      "startup_ms": startup_ms}, indent=2), flush=True)


if __name__ == "__main__":
    main()
