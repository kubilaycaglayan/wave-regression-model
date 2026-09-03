#!/usr/bin/env python3
"""Step 7: predict waviness for one new raw sea photo."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers.utils import logging as transformers_logging

from image_loading import load_rgb_image
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
# Validation-selected original baseline; keep this choice independent of Step 6 outputs.
CHECKPOINT_PATH = PROJECT_DIR / (
    "step-5-checkpoints/"
    "wave-regression-baseline-v1-best-val-mae-epoch=45-val_mae=0.1068.ckpt"
)
PREVIEW_DIR = PROJECT_DIR / "step-7-inference-preview"
SUPPORTED_EXTENSIONS = {".heic", ".heif", ".jpg", ".jpeg", ".png"}


def print_timing(label: str, started: float) -> None:
    print(f"{label}: {time.perf_counter() - started:.3f}s", flush=True)


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args()


def find_input_photo() -> Path:
    """Return the single supported photo placed in the prediction holder."""
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
    if len(files) > 1:
        raise ValueError(
            f"Expected one image in {INPUT_DIR}, found {len(files)}: "
            + ", ".join(path.name for path in files)
        )
    return files[0]


def preprocess_photo(path: Path, device: torch.device) -> tuple[Image.Image, bytes]:
    """Reproduce the persisted Step 1 -> Step 2 training pipeline in memory."""
    started = time.perf_counter()
    try:
        raw_image = load_rgb_image(path)
    except Exception as error:
        kind = "HEIC/HEIF" if path.suffix.lower() in {".heic", ".heif"} else "image"
        raise RuntimeError(f"Could not decode {kind} file {path}: {error}") from error
    print_timing("Decode input", started)

    started = time.perf_counter()
    segmentation_model = load_segmentation_model(str(device), verbose=False)
    print_timing("Load segmentation model", started)

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


def predict(model_input: Image.Image, device: torch.device) -> tuple[float, tuple[int, ...]]:
    started = time.perf_counter()
    transform = build_evaluation_transform()
    image_tensor = transform(model_input).float().unsqueeze(0)
    expected_shape = (1, 3, IMAGE_SIZE[1], IMAGE_SIZE[0])
    if tuple(image_tensor.shape) != expected_shape:
        raise RuntimeError(f"Inference tensor must have shape [1, 3, 224, 224], got {list(image_tensor.shape)}")
    print_timing("ToTensor and ImageNet normalization", started)

    started = time.perf_counter()
    model = WaveRegressionModel.load_from_checkpoint(CHECKPOINT_PATH, map_location=device)
    model.to(device)
    model.eval()
    print_timing("Load regression checkpoint", started)

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
        photo = find_input_photo()
        if not CHECKPOINT_PATH.is_file():
            raise FileNotFoundError(f"Selected checkpoint is missing: {CHECKPOINT_PATH}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Input: {photo.relative_to(PROJECT_DIR)}")
        print(f"Device: {device}", flush=True)
        model_input, encoded_model_input = preprocess_photo(photo, device)

        waviness, tensor_shape = predict(model_input, device)
        preview_started = time.perf_counter()
        preview_path = save_preview(photo, encoded_model_input)
        print_timing("Save preview", preview_started)

        print(f"Waviness: {waviness:.3f}")
        print(f"Preview: {preview_path.relative_to(PROJECT_DIR)}")
        print_timing("Total", total_started)
        expected_shape = (1, 3, IMAGE_SIZE[1], IMAGE_SIZE[0])
        if tensor_shape != expected_shape:  # Defensive assertion kept next to the CLI result.
            raise RuntimeError(f"Unexpected tensor shape after inference: {tensor_shape}")
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
