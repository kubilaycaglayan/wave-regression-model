#!/usr/bin/env python3
"""Keep prediction models loaded while watching the Step 7 input folder."""
from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import select
import struct
import sys
import time
from pathlib import Path

import torch
from PIL import Image

import step_7_a_predict as prediction
import step_1_a_water_segmentation as segmentation
from step_1_a_water_segmentation import SEGMENTATION_RESOLUTION, configure_inference_device, load_model, run_segmentation


class FileNotifications:
    """Linux notifications for files closed after writing or atomically moved in."""
    CLOSE_WRITE = 0x00000008
    MOVED_TO = 0x00000080
    OVERFLOW = 0x00004000
    IGNORED = 0x00008000
    DIRECTORY = 0x40000000
    EVENT = struct.Struct("iIII")

    def __init__(self, directory: Path):
        if not sys.platform.startswith("linux"):
            raise OSError(errno.ENOSYS, "File notifications require Linux")
        libc = ctypes.CDLL(None, use_errno=True)
        init = libc.inotify_init1
        init.argtypes = [ctypes.c_int]
        init.restype = ctypes.c_int
        add = libc.inotify_add_watch
        add.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        add.restype = ctypes.c_int
        self.fd = init(os.O_NONBLOCK | os.O_CLOEXEC)
        if self.fd < 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        if add(self.fd, os.fsencode(directory), self.CLOSE_WRITE | self.MOVED_TO) < 0:
            error = ctypes.get_errno()
            os.close(self.fd)
            self.fd = -1
            raise OSError(error, os.strerror(error), str(directory))

    def read(self, timeout: float):
        if not select.select([self.fd], [], [], timeout)[0]:
            return [], False
        data = os.read(self.fd, 65536)
        names, overflow = [], False
        position = 0
        while position < len(data):
            _, mask, _, length = self.EVENT.unpack_from(data, position)
            position += self.EVENT.size
            raw_name = data[position:position + length].split(b"\0", 1)[0]
            position += length
            if mask & self.IGNORED:
                raise RuntimeError("Watched input directory was removed or the watch was lost")
            if mask & self.OVERFLOW:
                overflow = True
            elif raw_name and not mask & self.DIRECTORY and mask & (self.CLOSE_WRITE | self.MOVED_TO):
                names.append(os.fsdecode(raw_name))
        return names, overflow

    def close(self):
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


def configure_experimental(single_resolution, logits_resolution, reuse_step1_mask, *, step2_resolution=None, precision="float32", device=None):
    """Separate previews and records for every preprocessing configuration."""
    settings = {"segmentation_model": segmentation.MODEL_NAME, "single_resolution": single_resolution,
                "logits_resolution": logits_resolution, "reuse_step1_mask": reuse_step1_mask,
                "step2_resolution": step2_resolution, "precision": precision,
                "segmentation_device": str(device) if device is not None else None,
                "pipeline_sha256": {name: prediction.sha256_for(prediction.PROJECT_DIR / name) for name in (
                    "step_1_a_water_segmentation.py", "step_1_b_preprocess_water_inputs.py",
                    "step_2_a_reduce_black_water_area.py", "step_7_a_predict.py")}}
    identifier = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
    variant = f"experimental-{identifier}"
    prediction.PREVIEW_DIR = prediction.PREVIEW_DIR / variant
    prediction.PREDICTIONS_DIR = prediction.PREDICTIONS_DIR / variant
    prediction.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    metadata = prediction.PREDICTIONS_DIR / "preprocessing.json"
    if metadata.exists():
        if json.loads(metadata.read_text(encoding="utf-8")) != settings:
            raise RuntimeError(f"Experimental preprocessing namespace collision: {metadata}")
    else:
        metadata.write_text(json.dumps(settings, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    segmentation.SEGMENTATION_RESOLUTION = single_resolution
    segmentation.USE_OVERLAPPING_TILES = False
    segmentation.SINGLE_IMAGE_LOGITS_RESOLUTION = logits_resolution
    segmentation.INFERENCE_PRECISION = precision
    if step2_resolution is not None:
        original = prediction.preprocess_step_1_image
        def resized_step2(image, segmenter):
            previous = segmenter.processor.size
            segmenter.processor.size = {"height": step2_resolution, "width": step2_resolution}
            try:
                return original(image, segmenter)
            finally:
                segmenter.processor.size = previous
        prediction.preprocess_step_1_image = resized_step2
    print(f"Experimental preprocessing {settings}; separate output namespace {variant}", flush=True)
    return variant


def process_photo(photo, checkpoint, digest, device, model, transform, segmenter, *, reuse_step1_mask=False, expected_segmentation_device=None):
    """Process one stable photo with the same preview and skip rules as Step 7."""
    if prediction.should_skip_photo(photo, checkpoint, digest):
        print(f"{photo.name}: skipped (matching prediction and preview already exist)", flush=True)
        return device
    started = time.perf_counter()
    preview = prediction.preview_path_for(photo)
    if preview.is_file():
        model_input = prediction.load_preview(preview)
        print(f"{photo.name}: reused existing preview", flush=True)
    else:
        if reuse_step1_mask:
            model_input, encoded = prediction.preprocess_photo(photo, segmenter, reuse_step1_mask=True)
        else:
            model_input, encoded = prediction.preprocess_photo(photo, segmenter)
        if expected_segmentation_device is not None and segmenter.device != expected_segmentation_device:
            raise RuntimeError(f"Experimental segmentation switched from {expected_segmentation_device} to {segmenter.device}; "
                               "refusing to mix different device/precision inputs in the same cache")
        prediction.save_preview(photo, encoded)
    try:
        value, shape = prediction.predict(model_input, device, model, transform)
    except RuntimeError as error:
        if device.type != "cuda" or not any(x in str(error).lower() for x in ("engine to execute", "cudnn", "cuda out of memory")):
            raise
        print(f"{photo.name}: regression device changed to CPU: {error}", flush=True)
        device = torch.device("cpu")
        model.to(device)
        value, shape = prediction.predict(model_input, device, model, transform)
    expected = (1, 3, prediction.IMAGE_SIZE[1], prediction.IMAGE_SIZE[0])
    if shape != expected:
        raise RuntimeError(f"Unexpected tensor shape after inference: {shape}")
    prediction.append_prediction_record(photo, checkpoint, digest, value, device,
                                        {"photo_elapsed": time.perf_counter() - started})
    print(f"{photo.name}: prediction {value:.3f} in {time.perf_counter() - started:.3f}s; "
          f"segmentation={segmenter.device}, regression={device}", flush=True)
    return device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, help="PyTorch CPU thread count")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--watch-mode", choices=("auto", "event", "poll"), default="auto",
                        help="Linux close-write/move notifications, or portable stability polling")
    parser.add_argument("--experimental-single-resolution", type=int, help="opt in to single-image segmentation at this resolution")
    parser.add_argument("--experimental-logits-resolution", type=int, help="interpolate single-image class scores to this size before argmax")
    parser.add_argument("--experimental-reuse-step1-mask", action="store_true", help="project the Step 1 mask instead of running Step 2 segmentation")
    parser.add_argument("--experimental-step2-resolution", type=int, help="separate Step 2 segmentation resolution")
    parser.add_argument("--experimental-precision", choices=("float32", "float16", "bfloat16"), default="float32", help="GPU segmentation precision")
    args = parser.parse_args()
    if (args.threads is not None and args.threads < 1) or args.poll_seconds <= 0:
        parser.error("threads and poll interval must be positive")
    if args.experimental_single_resolution is not None and args.experimental_single_resolution < 1:
        parser.error("experimental resolution must be positive")
    if args.experimental_logits_resolution is not None and args.experimental_logits_resolution < 1:
        parser.error("experimental logits resolution must be positive")
    if args.experimental_step2_resolution is not None and (args.experimental_step2_resolution < 1 or args.experimental_reuse_step1_mask):
        parser.error("experimental Step 2 resolution must be positive and requires a Step 2 pass")
    if (args.experimental_logits_resolution is not None or args.experimental_reuse_step1_mask or args.experimental_step2_resolution is not None or args.experimental_precision != "float32") and args.experimental_single_resolution is None:
        parser.error("experimental logits and mask options require --experimental-single-resolution")
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    checkpoint = prediction.checkpoint_path_from_env()
    digest = prediction.sha256_for(checkpoint)
    device = configure_inference_device(verbose=True)
    if args.experimental_single_resolution is not None:
        configure_experimental(args.experimental_single_resolution, args.experimental_logits_resolution,
                               args.experimental_reuse_step1_mask, step2_resolution=args.experimental_step2_resolution,
                               precision=args.experimental_precision, device=device)
    contract = prediction.validate_checkpoint_preprocessing(
        checkpoint, segmentation=segmentation, device=device,
        step2_resolution=args.experimental_step2_resolution or segmentation.SEGMENTATION_RESOLUTION,
        reuse_step1_mask=args.experimental_reuse_step1_mask,
        project_dir=prediction.PROJECT_DIR, sha256_for=prediction.sha256_for,
    )
    started = time.perf_counter()
    segmenter = load_model(str(device), verbose=False)
    prediction.validate_segmentation_revision(contract, segmenter)
    expected_segmentation_device = device if args.experimental_single_resolution is not None else None
    model = prediction.WaveRegressionModel.load_from_checkpoint(checkpoint, map_location=device, pretrained=False)
    model.to(device).eval()
    transform = prediction.build_evaluation_transform()
    print(f"Models loaded in {time.perf_counter() - started:.3f}s; watching {prediction.INPUT_DIR}; "
          f"checkpoint {checkpoint.resolve()}", flush=True)
    started = time.perf_counter()
    # Warm the model input shapes used by both stages; no source photo is stored.
    run_segmentation(Image.new("RGB", (SEGMENTATION_RESOLUTION, SEGMENTATION_RESOLUTION)), segmenter)
    if not args.experimental_reuse_step1_mask:
        previous = segmenter.processor.size
        if args.experimental_step2_resolution is not None:
            segmenter.processor.size = {"height": args.experimental_step2_resolution, "width": args.experimental_step2_resolution}
        try:
            run_segmentation(Image.new("RGB", prediction.IMAGE_SIZE), segmenter)
        finally:
            segmenter.processor.size = previous
    if expected_segmentation_device is not None and segmenter.device != expected_segmentation_device:
        raise RuntimeError(f"Experimental segmentation warmup switched from {expected_segmentation_device} to {segmenter.device}; "
                           "restart on the intended device to avoid mixing caches")
    try:
        with torch.inference_mode():
            model(torch.zeros((1, 3, prediction.IMAGE_SIZE[1], prediction.IMAGE_SIZE[0]), device=device))
    except RuntimeError as error:
        if device.type != "cuda" or not any(x in str(error).lower() for x in ("engine to execute", "cudnn", "cuda out of memory")):
            raise
        print(f"Regression warmup failed on GPU; switching to CPU: {error}", flush=True)
        device = torch.device("cpu")
        model.to(device)
        with torch.inference_mode():
            model(torch.zeros((1, 3, prediction.IMAGE_SIZE[1], prediction.IMAGE_SIZE[0]), device=device))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    print(f"Model warmup: {time.perf_counter() - started:.3f}s; segmentation={segmenter.device}, regression={device}", flush=True)
    if not prediction.INPUT_DIR.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {prediction.INPUT_DIR}")
    notifications = None
    if args.watch_mode != "poll":
        try:
            notifications = FileNotifications(prediction.INPUT_DIR)
        except (OSError, AttributeError) as error:
            if args.watch_mode == "event":
                raise RuntimeError(f"Cannot start file notifications: {error}") from error
            print(f"File notifications unavailable; using stability polling: {error}", flush=True)
    observed: dict[Path, tuple[int, int, int, int]] = {}
    handled: dict[Path, tuple[int, int, int, int]] = {}
    startup_candidates: set[Path] = set()

    def inspect(photo):
        if photo.name.startswith(".") or photo.suffix.lower() not in prediction.SUPPORTED_EXTENSIONS or not photo.is_file():
            return None
        stat = photo.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    def handle(photo, fingerprint):
        nonlocal device
        if handled.get(photo) == fingerprint:
            return
        # An unchanged failed input is reported once. Changing the file retries it.
        handled[photo] = fingerprint
        startup_candidates.discard(photo)
        try:
            device = process_photo(photo, checkpoint, digest, device, model, transform, segmenter,
                                   reuse_step1_mask=args.experimental_reuse_step1_mask,
                                   expected_segmentation_device=expected_segmentation_device)
        except Exception as error:
            print(f"{photo.name}: failed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)

    def scan():
        nonlocal observed, handled
        current = {}
        for photo in sorted(prediction.INPUT_DIR.iterdir()):
            try:
                fingerprint = inspect(photo)
                if fingerprint is None:
                    continue
                current[photo] = fingerprint
                if observed.get(photo) == fingerprint and (notifications is None or photo in startup_candidates):
                    handle(photo, fingerprint)
            except OSError as error:
                print(f"{photo.name}: could not inspect input: {error}", file=sys.stderr, flush=True)
        observed = current
        handled = {path: value for path, value in handled.items() if path in current}

    # The watch is installed before the first scan, so new arrivals are not
    # lost between startup and discovery of files already present.
    scan()
    startup_candidates.update(observed)
    print("File notifications ready" if notifications else "Stability polling ready", flush=True)
    next_scan = time.monotonic() + args.poll_seconds
    try:
        while True:
            if notifications is None:
                time.sleep(max(0, next_scan - time.monotonic()))
            else:
                names, overflow = notifications.read(max(0, next_scan - time.monotonic()))
                if overflow:
                    raise RuntimeError("File notification queue overflow; restart after incoming writes finish to rescan safely")
                for name in names:
                    photo = prediction.INPUT_DIR / name
                    try:
                        fingerprint = inspect(photo)
                        if fingerprint is not None:
                            handle(photo, fingerprint)
                    except OSError as error:
                        print(f"{name}: could not inspect input: {error}", file=sys.stderr, flush=True)
            if time.monotonic() >= next_scan:
                scan()
                next_scan = time.monotonic() + args.poll_seconds
    finally:
        if notifications is not None:
            notifications.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped watching.", flush=True)
