#!/usr/bin/env python3
"""Single-process local server for experimental MIGraphX prediction candidates.

POST raw image bytes to /predict with an X-Filename header. The two
SegFormer-B0 ONNX sessions and the v18 waviness regressor stay loaded for the
life of this process. HTTPServer handles one request at a time by design.
"""
from __future__ import annotations

import ctypes
import io
import json
import os
import tempfile
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PROCESS_START = time.perf_counter()

# These options must be set before the MIGraphX provider is imported.
os.environ.setdefault("MIGRAPHX_SKIP_BENCHMARKING", "1")
os.environ.setdefault("MIGRAPHX_USE_FAST_SOFTMAX", "0")

import cv2
import numpy as np
import onnxruntime as ort
import onnxruntime_ep_migraphx as mx
import pillow_heif
from PIL import Image, ImageOps

from step_0_a_native_heif_decoder import (
    _library as native_heif_library,
    decode_heif_rgb as decode_heif_rgb_image,
)
from step_2_b_crop_selector import select_crop, warm_crop_selector


ROOT = Path(__file__).resolve().parent
HOLDER = ROOT / "step-7-predict-captures-holder" / "experiment-predict-holder"
PREPROCESSED_ROOT = HOLDER / "preprocessed"
CANDIDATES = {
    "distilled-b0-migraphx-fp32": {
        "step1": ROOT / "experiments/distil-fp32/seg-distil-b0-step1-448-fp32-20261030.onnx",
        "step2": ROOT / "experiments/distil-fp32/seg-distil-b0-step2-384-to224-fp32-20261030.onnx",
        "slug": "1-distilled-b0-migraphx-fp32",
        "label": "B2-distilled SegFormer-B0 MIGraphX FP32 + OpenCV",
        "fp16": False, "opencv_mask_resize": False, "heif_workers": 16,
        "zero_copy_heif": True, "linear_processor": False,
    },
    "migraphx-b0-opencv-area": {
        "step1": ROOT / "experiments/area-b0/seg-b0-step1-448-fp32-20261030.onnx",
        "step2": ROOT / "experiments/area-b0/seg-b0-step2-384-to224-fp32-20261030.onnx",
        "slug": "2-migraphx-b0-opencv-area",
        "label": "SegFormer-B0 MIGraphX FP32 + OpenCV area preprocessing",
        "fp16": False, "opencv_mask_resize": False, "heif_workers": 16,
        "zero_copy_heif": True, "linear_processor": False,
    },
    "stable-migraphx-fp16-b2-opencv-nearest": {
        "step1": "/tmp/wave-segformer-b2-water-448.onnx",
        "step2": "/tmp/wave-segformer-b2-water-384-to448.onnx",
        "slug": "3-stable-migraphx-fp16-b2-opencv-nearest",
        "label": "Stable SegFormer-B2 MIGraphX FP16 + OpenCV exact-nearest",
        "fp16": True, "opencv_mask_resize": True, "heif_workers": 24,
        "zero_copy_heif": False, "linear_processor": True,
    },
}
EXPERIMENT = os.environ.get("WAVE_EXPERIMENT", "distilled-b0-migraphx-fp32")
if EXPERIMENT not in CANDIDATES:
    raise RuntimeError(f"Unknown WAVE_EXPERIMENT={EXPERIMENT!r}; choose one of {', '.join(CANDIDATES)}")
CONFIG = CANDIDATES[EXPERIMENT]
STEP1_ONNX = Path(os.environ.get("WAVE_STEP1_ONNX", CONFIG["step1"])).expanduser()
STEP2_ONNX = Path(os.environ.get("WAVE_STEP2_ONNX", CONFIG["step2"])).expanduser()
REGRESSOR_ONNX = Path(os.environ.get(
    "WAVE_REGRESSOR_ONNX", ROOT / "experiments/shared/reg-v18-efficientnet-b0-cpu-20261030.onnx"
)).expanduser()
HOST = "127.0.0.1"
PORT = int(os.environ.get("WAVE_SERVER_PORT", "8765"))
MAX_UPLOAD_BYTES = 64 * 1024 * 1024
SUPPORTED = {".heic", ".heif", ".jpg", ".jpeg", ".png"}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

cv2.setNumThreads(4)
pillow_heif.options.DECODE_THREADS = 16


class NativeRGBFrame:
    """Keep the native HEIF allocation alive while its RGB pixels are used."""

    def __init__(self, library, pointer, width: int, height: int):
        self.library, self.pointer = library, pointer
        self.width, self.height = width, height
        owner = (ctypes.c_ubyte * (width * height * 3)).from_address(
            ctypes.addressof(pointer.contents)
        )
        self.owner = owner
        self.pixels = np.ctypeslib.as_array(owner).reshape(height, width, 3)

    def close(self):
        if self.pointer:
            self.library.wave_heif_free_rgb(self.pointer)
            self.pointer = None
            self.owner = self.pixels = None


def decode_photo(path: Path):
    suffix = path.suffix.lower()
    if suffix in {".heic", ".heif"} and not CONFIG["zero_copy_heif"]:
        decoded = decode_heif_rgb_image(path, workers=CONFIG["heif_workers"])
        if decoded is not None:
            return decoded
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            image.load()
            return image.convert("RGB")
    if suffix in {".jpg", ".jpeg"}:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"OpenCV could not decode {path.name}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if suffix in {".heic", ".heif"}:
        library = native_heif_library()
        if library is not None:
            pointer = ctypes.POINTER(ctypes.c_ubyte)()
            width, height = ctypes.c_int(), ctypes.c_int()
            message = ctypes.create_string_buffer(512)
            status = library.wave_heif_decode_tiled_rgb(
                str(path.resolve()).encode(), 16, ctypes.byref(pointer),
                ctypes.byref(width), ctypes.byref(height), message, len(message)
            )
            if status == 0:
                return NativeRGBFrame(library, pointer, width.value, height.value)
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        image.load()
        return np.array(image.convert("RGB"), dtype=np.uint8, copy=True)


def processor_input(image, size: int) -> np.ndarray:
    source = image.pixels if isinstance(image, NativeRGBFrame) else np.asarray(image, dtype=np.uint8)
    if CONFIG["linear_processor"]:
        pixels = cv2.resize(source, (size, size), interpolation=cv2.INTER_LINEAR_EXACT).astype(np.float32)
        pixels = (pixels.astype(np.float64) * (1.0 / 255.0)).astype(np.float32)
    else:
        interpolation = cv2.INTER_AREA if size < max(source.shape[:2]) else cv2.INTER_LINEAR_EXACT
        pixels = cv2.resize(source, (size, size), interpolation=interpolation).astype(np.float32)
        pixels *= np.float32(1.0 / 255.0)
    pixels = (pixels - MEAN) / STD
    return np.ascontiguousarray(pixels.transpose(2, 0, 1)[None], dtype=np.float32)


def expand_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a predicted mask with the exact operator used by this candidate."""
    if mask.shape == (height, width):
        return np.ascontiguousarray(mask, dtype=np.uint8)
    if CONFIG["opencv_mask_resize"]:
        return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST_EXACT)

    def indices(source_length: int, destination_length: int, vertical: bool) -> np.ndarray:
        if source_length == destination_length:
            return np.arange(source_length, dtype=np.intp)
        values = np.arange(source_length, dtype=np.uint16)
        low = (values & 0xFF).astype(np.uint8)
        low = low[:, None] if vertical else low[None, :]
        shape = (1, destination_length) if vertical else (destination_length, 1)
        low_result = np.asarray(
            Image.fromarray(low).resize(shape, Image.Resampling.NEAREST)
        ).reshape(-1).astype(np.uint16)
        if source_length <= 256:
            return low_result.astype(np.intp)
        high = (values >> 8).astype(np.uint8)
        high = high[:, None] if vertical else high[None, :]
        high_result = np.asarray(
            Image.fromarray(high).resize(shape, Image.Resampling.NEAREST)
        ).reshape(-1).astype(np.uint16)
        return (low_result | (high_result << 8)).astype(np.intp)

    ys = indices(mask.shape[0], height, True)
    xs = indices(mask.shape[1], width, False)
    return np.take(np.take(mask, ys, axis=0), xs, axis=1)


def infer_mask(image, size: int, session, stage: dict, prefix: str) -> np.ndarray:
    started = time.perf_counter()
    tensor = processor_input(image, size)
    stage[f"{prefix}_processor_ms"] = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    mask = session.run(["water_mask"], {"pixel_values": tensor})[0][0]
    stage[f"{prefix}_model_ms"] = (time.perf_counter() - started) * 1000
    width, height = (
        (image.width, image.height) if isinstance(image, Image.Image)
        else (image.width, image.height) if isinstance(image, NativeRGBFrame)
        else (image.shape[1], image.shape[0])
    )
    started = time.perf_counter()
    result = expand_mask(mask, width, height)
    stage[f"{prefix}_mask_expand_ms"] = (time.perf_counter() - started) * 1000
    return result


def resize_lanczos(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    gap = 2.0 if image.width >= size[0] * 3 and image.height >= size[1] * 3 else None
    return image.resize(size, Image.Resampling.LANCZOS, reducing_gap=gap)


def to_canvas(image: Image.Image) -> Image.Image:
    scale = min(224 / image.width, 224 / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    resized = resize_lanczos(image, size)
    output = Image.new("RGB", (224, 224), (0, 0, 0))
    output.paste(resized, ((224 - size[0]) // 2, (224 - size[1]) // 2))
    return output


def run_pipeline(path: Path) -> dict:
    started = time.perf_counter()
    stages = {}
    tick = time.perf_counter()
    image = decode_photo(path)
    stages["decode_ms"] = (time.perf_counter() - tick) * 1000
    try:
        width, height = (
            (image.width, image.height) if isinstance(image, (NativeRGBFrame, Image.Image))
            else (image.shape[1], image.shape[0])
        )

        tick = time.perf_counter()
        mask = infer_mask(image, 448, STEP1_SESSION, stages, "step1")
        radius = max(1, min(12, round(min(width, height) * 0.005)))
        eroded = cv2.erode(mask, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8), borderType=cv2.BORDER_REPLICATE)
        left, top, box_width, box_height = cv2.boundingRect(eroded)
        if box_width == 0 or box_height == 0 or cv2.countNonZero(eroded) < max(32, round(width * height * 0.0005)):
            raise ValueError("Step 1 could not reliably detect water")
        source = image.pixels if isinstance(image, NativeRGBFrame) else np.asarray(image) if isinstance(image, Image.Image) else image
        roi_pixels = source[top:top + box_height, left:left + box_width]
        roi_mask = eroded[top:top + box_height, left:left + box_width]
        masked = Image.fromarray(cv2.bitwise_and(roi_pixels, roi_pixels, mask=roi_mask), mode="RGB")
        step1_image = to_canvas(masked)
        stages["step1_ms"] = (time.perf_counter() - tick) * 1000

        tick = time.perf_counter()
        step1_bytes = io.BytesIO()
        step1_image.save(step1_bytes, format="JPEG", quality=95)
        with Image.open(io.BytesIO(step1_bytes.getvalue())) as saved:
            saved.load()
            step1_for_step2 = saved.convert("RGB")
        stages["step1_jpeg_roundtrip_ms"] = (time.perf_counter() - tick) * 1000

        tick = time.perf_counter()
        rgb = np.asarray(step1_for_step2, dtype=np.uint8)
        black = np.all(rgb <= 12, axis=2)
        water = infer_mask(step1_for_step2, 384, STEP2_SESSION, stages, "step2").astype(bool) & ~black
        selected = select_crop(water, black)
        if selected is None:
            raise ValueError("Step 2 could not reliably select a lower-water crop")
        step2_image = to_canvas(step1_for_step2.crop(selected[0]))
        stages["step2_ms"] = (time.perf_counter() - tick) * 1000

        tick = time.perf_counter()
        encoded = io.BytesIO()
        step2_image.save(encoded, format="JPEG", quality=95, subsampling=0)
        with Image.open(io.BytesIO(encoded.getvalue())) as saved:
            saved.load()
            model_input = np.asarray(saved.convert("RGB"), dtype=np.uint8).astype(np.float32) / np.float32(255.0)
        tensor = np.ascontiguousarray(((model_input - MEAN) / STD).transpose(2, 0, 1)[None], dtype=np.float32)
        score = float(REGRESSOR_SESSION.run(["scores"], {"images": tensor})[0].reshape(-1)[0])
        stages["step2_jpeg_and_regression_ms"] = (time.perf_counter() - tick) * 1000
        return {
            "prediction": score,
            "stages_ms": stages,
            "total_server_ms": (time.perf_counter() - started) * 1000,
            "step1_jpeg": step1_bytes.getvalue(),
            "step2_jpeg": encoded.getvalue(),
        }
    finally:
        if isinstance(image, NativeRGBFrame):
            image.close()


def load_session(path: Path, device, label: str):
    if not path.is_file():
        raise FileNotFoundError(f"{label} model file not found: {path}")
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    provider_options = {"migraphx_fp16_enable": "1"} if CONFIG["fp16"] else {}
    options.add_provider_for_devices([device], provider_options)
    session = ort.InferenceSession(str(path), sess_options=options)
    if "MIGraphXExecutionProvider" not in session.get_providers():
        raise RuntimeError(f"{label} did not load on MIGraphX: {session.get_providers()}")
    return session


def initialize_models():
    setup = {}
    tick = time.perf_counter()
    for name, library_path in zip(mx.get_ep_names(), mx.get_library_paths()):
        ort.register_execution_provider_library(name, library_path)
    device = next((d for d in ort.get_ep_devices() if d.ep_name == mx.get_ep_name()), None)
    if device is None:
        raise RuntimeError("ONNX Runtime did not expose a MIGraphX device")
    setup["runtime_registration_ms"] = (time.perf_counter() - tick) * 1000
    global STEP1_SESSION, STEP2_SESSION, REGRESSOR_SESSION
    tick = time.perf_counter()
    STEP1_SESSION = load_session(STEP1_ONNX, device, "Step 1 segmentation")
    setup["step1_session_load_ms"] = (time.perf_counter() - tick) * 1000
    tick = time.perf_counter()
    STEP2_SESSION = load_session(STEP2_ONNX, device, "Step 2 segmentation")
    setup["step2_session_load_ms"] = (time.perf_counter() - tick) * 1000
    if not REGRESSOR_ONNX.is_file():
        raise FileNotFoundError(f"v18 regressor model file not found: {REGRESSOR_ONNX}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    tick = time.perf_counter()
    REGRESSOR_SESSION = ort.InferenceSession(
        str(REGRESSOR_ONNX), sess_options=options, providers=["CPUExecutionProvider"]
    )
    setup["regressor_session_load_ms"] = (time.perf_counter() - tick) * 1000
    if "CPUExecutionProvider" not in REGRESSOR_SESSION.get_providers():
        raise RuntimeError("The v18 regressor did not load on CPU")

    # Trigger first-use compilation for every graph before requests are accepted.
    tick = time.perf_counter()
    for session, size in ((STEP1_SESSION, 448), (STEP2_SESSION, 384)):
        session.run(["water_mask"], {"pixel_values": np.zeros((1, 3, size, size), dtype=np.float32)})
        setup[f"warm_{size}_first_run_ms"] = (time.perf_counter() - tick) * 1000
        tick = time.perf_counter()
    tick = time.perf_counter()
    REGRESSOR_SESSION.run(["scores"], {"images": np.zeros((1, 3, 224, 224), dtype=np.float32)})
    setup["warm_regressor_ms"] = (time.perf_counter() - tick) * 1000
    tick = time.perf_counter()
    warm_crop_selector()
    setup["warm_crop_selector_ms"] = (time.perf_counter() - tick) * 1000
    return setup


class Handler(BaseHTTPRequestHandler):
    server_version = "WaveExperiment/1.0"

    def respond(self, status: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if urlparse(self.path).path == "/health":
            self.respond(200, {"ready": True, "candidate": EXPERIMENT, "label": CONFIG["label"]})
        else:
            self.respond(404, {"error": "Use GET /health or POST /predict"})

    def do_POST(self):
        if urlparse(self.path).path != "/predict":
            self.respond(404, {"error": "Use POST /predict"})
            return
        try:
            filename = Path(self.headers.get("X-Filename", "request.jpg")).name
            suffix = Path(filename).suffix.lower()
            if suffix not in SUPPORTED:
                raise ValueError(f"Unsupported image suffix: {suffix or '(none)'}")
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > MAX_UPLOAD_BYTES:
                raise ValueError(f"Content-Length must be between 1 and {MAX_UPLOAD_BYTES}")
            started = time.perf_counter()
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError("Request body ended before Content-Length bytes were received")
            parsed = urlparse(self.path)
            save_steps = parse_qs(parsed.query).get("save_steps", ["0"])[0] == "1"
            with tempfile.NamedTemporaryFile(suffix=suffix) as upload:
                upload.write(body)
                upload.flush()
                result = run_pipeline(Path(upload.name))
                step1_jpeg = result.pop("step1_jpeg")
                step2_jpeg = result.pop("step2_jpeg")
                if save_steps:
                    tick = time.perf_counter()
                    slug = Path(filename).stem
                    step1_dir = PREPROCESSED_ROOT / CONFIG["slug"] / "step-1"
                    step2_dir = PREPROCESSED_ROOT / CONFIG["slug"] / "step-2"
                    step1_dir.mkdir(parents=True, exist_ok=True)
                    step2_dir.mkdir(parents=True, exist_ok=True)
                    (step1_dir / f"{slug}-step-1.jpg").write_bytes(step1_jpeg)
                    (step2_dir / f"{slug}-step-2-model-input.jpg").write_bytes(step2_jpeg)
                    result["stages_ms"]["save_preprocessed_images_ms"] = (time.perf_counter() - tick) * 1000
                    result["preview_directories"] = {"step1": str(step1_dir), "step2": str(step2_dir)}
            result["request_total_ms"] = (time.perf_counter() - started) * 1000
            result["filename"] = filename
            self.respond(200, result)
            print(f"{filename}: {result['prediction']:.8f}, {result['request_total_ms']:.2f} ms", flush=True)
        except Exception as error:
            self.respond(400, {"error": f"{type(error).__name__}: {error}"})

    def log_message(self, fmt, *args):
        print("http: " + fmt % args, flush=True)


def main():
    started = time.perf_counter()
    setup = initialize_models()
    warm_seconds = time.perf_counter() - started
    server = HTTPServer((HOST, PORT), Handler)
    print(json.dumps({
        "ready": True,
        "url": f"http://{HOST}:{PORT}",
        "sequential": True,
        "candidate": CONFIG["label"],
        "candidate_key": EXPERIMENT,
        "step1_model": str(STEP1_ONNX),
        "step2_model": str(STEP2_ONNX),
        "regressor_model": str(REGRESSOR_ONNX),
        "heif_workers": CONFIG["heif_workers"],
        "opencv_threads": cv2.getNumThreads(),
        "migraphx_fp16": CONFIG["fp16"],
        "setup_breakdown_ms": setup,
        "setup_and_warmup_seconds_after_imports": round(warm_seconds, 3),
        "process_launch_to_ready_seconds": round(time.perf_counter() - PROCESS_START, 3),
    }), flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("Stopping experimental inference server.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
