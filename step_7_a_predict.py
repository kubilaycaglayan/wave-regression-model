#!/usr/bin/env python3
"""Step 7: predict waviness for one new raw sea photo."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

import torch
from PIL import Image
from transformers.utils import logging as transformers_logging

from image_loading import load_rgb_image
from step_1_a_water_segmentation import SegmentationModel
from step_1_a_water_segmentation import configure_inference_device
from step_1_a_water_segmentation import load_model as load_segmentation_model
from step_1_b_preprocess_water_inputs import (
    decode_rgb_bytes,
    encode_step_1_output,
    preprocess_raw_image,
)
from step_2_a_reduce_black_water_area import encode_step_2_output, preprocess_step_1_image
from step_4_a_wave_dataset import IMAGE_SIZE, build_evaluation_transform
from step_5_a_wave_regression_model import WaveRegressionModel


PROJECT_DIR = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_DIR / "predict-holder"


def checkpoint_path_from_env() -> Path:
    """Read the prediction checkpoint path from the environment or repository .env."""
    configured_path = os.getenv("PREDICT_CHECKPOINT_PATH")
    if configured_path is None:
        dotenv_path = PROJECT_DIR / ".env"
        if dotenv_path.is_file():
            for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                key, separator, value = line.partition("=")
                if separator and key.strip() == "PREDICT_CHECKPOINT_PATH":
                    configured_path = value.strip().strip("\"'")
                    break

    if not configured_path:
        raise RuntimeError("PREDICT_CHECKPOINT_PATH is not configured; add it to .env or the environment")

    path = Path(configured_path).expanduser()
    return path if path.is_absolute() else PROJECT_DIR / path


PREVIEW_DIR = PROJECT_DIR / "step-7-inference-preview"
PREDICTIONS_DIR = INPUT_DIR / "predictions"
SUPPORTED_EXTENSIONS = {".heic", ".heif", ".jpg", ".jpeg", ".png"}
PREDICTION_RECORD_MARKER = "--- prediction record ---"


def print_timing(label: str, started: float) -> None:
    print(f"{label}: {time.perf_counter() - started:.3f}s", flush=True)


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args()


def find_input_photos() -> list[Path]:
    """Return all supported photos placed in the prediction holder."""
    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(f"Prediction input directory does not exist: {INPUT_DIR}")
    files = sorted(
        (path for path in INPUT_DIR.iterdir() if path.is_file() and not path.name.startswith(".")),
        key=lambda path: path.name.lower(),
    )
    unsupported = [path.name for path in files if path.suffix.lower() not in SUPPORTED_EXTENSIONS]
    if unsupported:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(
            f"Unsupported file in {INPUT_DIR.name}: {', '.join(unsupported)}; supported: {supported}"
        )
    if not files:
        raise FileNotFoundError(f"No image found in {INPUT_DIR}; add one HEIC, HEIF, JPEG, or PNG photo")
    return files


def preprocess_photo(
    path: Path, segmentation_model: SegmentationModel
) -> tuple[Image.Image, bytes]:
    """Reproduce the persisted Step 1 -> Step 2 training pipeline in memory."""
    started = time.perf_counter()
    try:
        raw_image = load_rgb_image(path)
    except Exception as error:
        kind = "HEIC/HEIF" if path.suffix.lower() in {".heic", ".heif"} else "image"
        raise RuntimeError(f"Could not decode {kind} file {path}: {error}") from error
    print_timing("Decode input", started)

    started = time.perf_counter()
    step_1_image = preprocess_raw_image(raw_image, segmentation_model)
    if step_1_image is None:
        raise RuntimeError("Water cannot be reliably detected during Step 1 preprocessing")
    print_timing("Step 1 segmentation and standardization", started)

    # Training Step 2 read Step 1's JPEG from disk. The round trip is intentional.
    started = time.perf_counter()
    persisted_step_1 = decode_rgb_bytes(encode_step_1_output(step_1_image))
    print_timing("Step 1 JPEG round trip", started)

    started = time.perf_counter()
    step_2_output = preprocess_step_1_image(persisted_step_1, segmentation_model)
    if step_2_output is None:
        raise RuntimeError("Water cannot be reliably detected for the lower-water Step 2 crop")
    step_2_image = step_2_output.image
    print_timing("Step 2 lower-water preprocessing", started)

    # Training loaded the final Step 2 JPEG. Feed that same decoded representation.
    started = time.perf_counter()
    encoded_model_input = encode_step_2_output(step_2_image)
    model_input = decode_rgb_bytes(encoded_model_input)
    if model_input.mode != "RGB" or model_input.size != IMAGE_SIZE:
        raise RuntimeError(
            f"Preprocessing produced {model_input.mode} {model_input.size}; expected RGB {IMAGE_SIZE}"
        )
    print_timing("Step 2 JPEG round trip", started)
    return model_input, encoded_model_input


def save_preview(path: Path, encoded_model_input: bytes) -> Path:
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    preview_path = PREVIEW_DIR / f"{path.stem}-model-input.jpg"
    if preview_path.exists():
        if preview_path.read_bytes() != encoded_model_input:
            raise FileExistsError(f"Refusing to overwrite a different existing preview: {preview_path}")
        return preview_path
    preview_path.write_bytes(encoded_model_input)
    return preview_path


def preview_path_for(path: Path) -> Path:
    return PREVIEW_DIR / f"{path.stem}-model-input.jpg"


def load_preview(path: Path) -> Image.Image:
    """Load an existing preview as a detached, validated model input."""
    try:
        with Image.open(path) as preview:
            mode = preview.mode
            size = preview.size
            if mode != "RGB" or size != IMAGE_SIZE:
                raise RuntimeError(
                    f"Invalid inference preview {path}: got {mode} {size}; "
                    f"expected detached RGB {IMAGE_SIZE} input"
                )
            preview.load()
            return preview.copy()
    except RuntimeError:
        raise
    except Exception as error:
        raise RuntimeError(f"Could not load inference preview {path}: {error}") from error


def prepare_model_input(
    photo: Path,
    preview_inputs: Mapping[Path, Image.Image],
    segmentation_model: SegmentationModel | None,
) -> tuple[Image.Image, bytes | None, bool]:
    """Return the cached input or generate it, plus whether it was reused."""
    cached_input = preview_inputs.get(photo)
    if cached_input is not None:
        return cached_input, None, True
    if segmentation_model is None:
        raise RuntimeError(f"No segmentation model is available to preprocess {photo}")
    model_input, encoded_model_input = preprocess_photo(photo, segmentation_model)
    return model_input, encoded_model_input, False


def prediction_log_path_for(path: Path) -> Path:
    """Return the append-only prediction history path for an input photo."""
    return PREDICTIONS_DIR / f"{path.stem}.txt"


def sha256_for(path: Path) -> str:
    """Return a file's SHA-256 digest without loading the whole file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_prediction_records(path: Path) -> list[dict[str, str]]:
    """Read completed human-readable prediction records from a log file."""
    if not path.is_file():
        return []

    records: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line == PREDICTION_RECORD_MARKER:
            if current is not None:
                records.append(current)
            current = {}
            continue
        if current is None:
            raise ValueError(f"Prediction log must start with {PREDICTION_RECORD_MARKER}: {path}:{line_number}")
        key, separator, value = line.partition(":")
        if not separator or not key.strip():
            raise ValueError(f"Malformed prediction log line: {path}:{line_number}")
        current[key.strip()] = value.strip()
    if current is not None:
        records.append(current)
    return records


def has_matching_prediction(
    photo: Path,
    checkpoint_path: Path,
    checkpoint_sha256: str,
) -> bool:
    """Return whether this photo has a completed result for this checkpoint."""
    input_path = str(photo.resolve())
    selected_checkpoint = str(checkpoint_path.resolve())
    for record in read_prediction_records(prediction_log_path_for(photo)):
        if (
            record.get("input_path") == input_path
            and record.get("checkpoint_path") == selected_checkpoint
            and record.get("checkpoint_sha256") == checkpoint_sha256
            and record.get("prediction")
        ):
            return True
    return False


def should_skip_photo(
    photo: Path,
    checkpoint_path: Path,
    checkpoint_sha256: str,
) -> bool:
    """Skip only after both the matching result and its preview are present."""
    return preview_path_for(photo).is_file() and has_matching_prediction(
        photo, checkpoint_path, checkpoint_sha256
    )


def append_prediction_record(
    photo: Path,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    prediction: float,
    device: torch.device,
    timings: Mapping[str, float],
) -> Path:
    """Append one completed prediction record and return its log path."""
    log_path = prediction_log_path_for(photo)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        PREDICTION_RECORD_MARKER,
        f"input_path: {photo.resolve()}",
        f"prediction: {prediction:.8f}",
        f"checkpoint_path: {checkpoint_path.resolve()}",
        f"checkpoint_sha256: {checkpoint_sha256}",
        f"timestamp_utc: {datetime.now(timezone.utc).isoformat()}",
        f"device: {device}",
    ]
    for label, seconds in timings.items():
        fields.append(f"{label}_seconds: {seconds:.3f}")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(fields) + "\n\n")
    return log_path


def predict(
    model_input: Image.Image,
    device: torch.device,
    model: WaveRegressionModel,
    transform: Callable[[Image.Image], torch.Tensor],
) -> tuple[float, tuple[int, ...]]:
    started = time.perf_counter()
    image_tensor = transform(model_input).float().unsqueeze(0)
    expected_shape = (1, 3, IMAGE_SIZE[1], IMAGE_SIZE[0])
    if tuple(image_tensor.shape) != expected_shape:
        raise RuntimeError(f"Inference tensor must have shape [1, 3, 224, 224], got {list(image_tensor.shape)}")
    print_timing("ToTensor and ImageNet normalization", started)

    started = time.perf_counter()
    with torch.inference_mode():
        output = model(image_tensor.to(device))
    print_timing("Model inference", started)
    if output.numel() != 1:
        raise RuntimeError(f"Model returned {output.numel()} predictions; expected one")
    value = float(output.item())
    if not math.isfinite(value):
        raise RuntimeError(f"Model returned a non-finite prediction: {value}")
    if not 0.0 <= value <= 1.0:
        raise RuntimeError(f"Model returned a prediction outside [0, 1]: {value}")
    return value, tuple(image_tensor.shape)


def main() -> None:
    parse_args()
    try:
        total_started = time.perf_counter()
        transformers_logging.disable_progress_bar()
        photos = find_input_photos()
        checkpoint_path = checkpoint_path_from_env()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Selected checkpoint is missing: {checkpoint_path}")
        checkpoint_started = time.perf_counter()
        checkpoint_sha256 = sha256_for(checkpoint_path)
        print_timing("Fingerprint checkpoint", checkpoint_started)

        device = configure_inference_device(verbose=True)
        preview_inputs: dict[Path, Image.Image] = {}
        for photo in photos:
            preview_path = preview_path_for(photo)
            if preview_path.is_file():
                preview_started = time.perf_counter()
                preview_inputs[photo] = load_preview(preview_path)
                print_timing(f"Load preview {photo.name}", preview_started)

        pending_photos = [
            photo
            for photo in photos
            if not should_skip_photo(photo, checkpoint_path, checkpoint_sha256)
        ]
        print(f"Images: {len(photos)} ({len(pending_photos)} pending)")
        print(f"Device: {device}", flush=True)
        print(f"Checkpoint: {checkpoint_path.resolve()}", flush=True)

        if not pending_photos:
            for index, photo in enumerate(photos, 1):
                print(
                    f"[{index}/{len(photos)}] {photo.name}: "
                    "skipped (matching prediction and preview already exist)"
                )
            print_timing("Total", total_started)
            return

        segmentation_model: SegmentationModel | None = None
        if any(photo not in preview_inputs for photo in pending_photos):
            started = time.perf_counter()
            segmentation_model = load_segmentation_model(str(device), verbose=False)
            print_timing("Load segmentation model", started)

        started = time.perf_counter()
        model = WaveRegressionModel.load_from_checkpoint(checkpoint_path, map_location=device)
        model.to(device)
        model.eval()
        print_timing("Load regression checkpoint", started)
        transform = build_evaluation_transform()

        processed = 0
        skipped = 0
        failed = 0
        for index, photo in enumerate(photos, 1):
            if should_skip_photo(photo, checkpoint_path, checkpoint_sha256):
                skipped += 1
                print(
                    f"[{index}/{len(photos)}] {photo.name}: "
                    "skipped (matching prediction and preview already exist)"
                )
                continue

            photo_started = time.perf_counter()
            print(f"[{index}/{len(photos)}] Input: {photo.relative_to(PROJECT_DIR)}", flush=True)
            try:
                preview_path = preview_path_for(photo)
                model_input, encoded_model_input, reused_preview = prepare_model_input(
                    photo, preview_inputs, segmentation_model
                )
                if reused_preview:
                    print(f"Model input: reused {preview_path.relative_to(PROJECT_DIR)}")
                else:
                    assert encoded_model_input is not None
                    preview_started = time.perf_counter()
                    preview_path = save_preview(photo, encoded_model_input)
                    print_timing("Save preview", preview_started)
                try:
                    waviness, tensor_shape = predict(model_input, device, model, transform)
                except RuntimeError as error:
                    retryable_cuda_error = any(
                        message in str(error).lower()
                        for message in ("engine to execute", "cudnn", "cuda out of memory")
                    )
                    if device.type != "cuda" or not retryable_cuda_error:
                        raise
                    print("CUDA regression inference unavailable; retrying this photo on CPU.", flush=True)
                    device = torch.device("cpu")
                    model.to(device)
                    waviness, tensor_shape = predict(model_input, device, model, transform)

                expected_shape = (1, 3, IMAGE_SIZE[1], IMAGE_SIZE[0])
                if tensor_shape != expected_shape:
                    raise RuntimeError(f"Unexpected tensor shape after inference: {tensor_shape}")
                elapsed = time.perf_counter() - photo_started
                log_path = append_prediction_record(
                    photo,
                    checkpoint_path,
                    checkpoint_sha256,
                    waviness,
                    device,
                    {"photo_elapsed": elapsed},
                )
                print(f"Waviness: {waviness:.3f}")
                print(f"Preview: {preview_path.relative_to(PROJECT_DIR)}")
                print(f"Prediction log: {log_path.relative_to(PROJECT_DIR)}")
                print_timing("Photo total", photo_started)
                processed += 1
            except Exception as error:
                failed += 1
                print(f"Error processing {photo.name}: {error}", file=sys.stderr, flush=True)

        print(f"Processed: {processed}; skipped: {skipped}; failed: {failed}")
        print_timing("Total", total_started)
        if failed:
            raise SystemExit(1)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
