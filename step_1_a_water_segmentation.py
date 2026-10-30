"""Reusable pretrained water-segmentation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from contextlib import nullcontext
from typing import Iterable

import numpy as np
import torch
from PIL import Image

from image_loading import load_rgb_image
from data_sources import iter_images
from torchvision_compat import ensure_torchvision_operator_schemas

ensure_torchvision_operator_schemas()

from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# ---------------------------------------------------------------------------
# Segmentation configuration
#
# Keep these values visible here so experiments can be made without changing
# the pipeline code below.
# ---------------------------------------------------------------------------

# Larger inputs preserve more detail in the thin sea band, at the cost of
# additional memory and inference time.
SEGMENTATION_RESOLUTION = 768

# Overlapping crops prevent the complete source image from being reduced to a
# single square. This is useful when the sea is a thin band behind buildings.
USE_OVERLAPPING_TILES = True
TILE_SIZE = 768
TILE_OVERLAP = 0.25

# Optional experimental single-image class-score interpolation size. None keeps
# the historical single-image behavior (argmax before nearest-neighbor resize).
SINGLE_IMAGE_LOGITS_RESOLUTION = None

# Experimental cap on the source dimensions used for segmentation only.
# The returned class map is projected back to the original photo dimensions.
MAX_SEGMENTATION_SIDE = None

# Experimental autocast is opt-in; CPU fallback always uses full precision.
INFERENCE_PRECISION = "float32"

# Useful while comparing resolutions and tile settings. Disable for quieter
# logs after choosing a stable configuration.
LOG_TILE_TIMING = True

# Optional benchmark observer; normal pipeline execution has no synchronization overhead.
BENCHMARK_OBSERVER = None

MODEL_NAME = "nvidia/segformer-b2-finetuned-ade-512-512"
WATER_CLASS_NAMES = {"water", "sea", "river", "lake"}


@dataclass
class SegmentationModel:
    model: SegformerForSemanticSegmentation
    processor: SegformerImageProcessor
    water_class_ids: tuple[int, ...]
    device: torch.device


def model_logits(segmentation_model: SegmentationModel, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Run the segmenter with optional accelerator-only autocasting."""
    types = {"float32": None, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if INFERENCE_PRECISION not in types:
        raise ValueError(f"Unsupported segmentation precision: {INFERENCE_PRECISION}")
    dtype = types[INFERENCE_PRECISION]
    context = (torch.autocast(device_type="cuda", dtype=dtype)
               if dtype is not None and segmentation_model.device.type == "cuda" else nullcontext())
    with context:
        return segmentation_model.model(**inputs).logits


def configure_inference_device(device: str | None = None, *, verbose: bool = True) -> torch.device:
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if selected_device.type != "cuda":
        return selected_device
    try:
        torch.backends.cudnn.version()
    except RuntimeError as error:
        if "not compatible with devices with sm" not in str(error).lower():
            raise
        torch.backends.cudnn.enabled = False
        if verbose:
            print(f"cuDNN disabled for {torch.cuda.get_device_name(selected_device)}; using native CUDA kernels.")
    return selected_device


def load_model(device: str | None = None, *, verbose: bool = True) -> SegmentationModel:
    """Load the pretrained model used by both preprocessing and previews."""
    selected_device = configure_inference_device(device, verbose=verbose)
    processor = SegformerImageProcessor.from_pretrained(
        MODEL_NAME,
        size={"height": SEGMENTATION_RESOLUTION, "width": SEGMENTATION_RESOLUTION},
        do_resize=True,
    )
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME).to(selected_device)
    model.eval()
    id2label = {int(key): value.lower().strip() for key, value in model.config.id2label.items()}
    water_class_ids = tuple(i for i, label in id2label.items() if label in WATER_CLASS_NAMES)
    if not water_class_ids:
        raise RuntimeError(f"{MODEL_NAME} does not expose expected water classes: {WATER_CLASS_NAMES}")
    if verbose:
        print(f"Model: {MODEL_NAME} on {selected_device}")
        print(
            f"Processor: {SEGMENTATION_RESOLUTION}x{SEGMENTATION_RESOLUTION}; "
            f"overlapping tiles={'on' if USE_OVERLAPPING_TILES else 'off'}"
        )
        if USE_OVERLAPPING_TILES:
            print(f"Tile size: {TILE_SIZE}px; overlap: {TILE_OVERLAP:.0%}")
        print("Water classes: " + ", ".join(f"{id2label[i]}={i}" for i in water_class_ids))
    return SegmentationModel(model, processor, water_class_ids, selected_device)


def tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    """Return starts that cover an axis, forcing the final tile to touch its end."""
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


@torch.inference_mode()
def run_overlapping_segmentation(
    image: Image.Image,
    segmentation_model: SegmentationModel,
) -> torch.Tensor:
    """Segment overlapping source-resolution tiles and average their logits."""
    started = time.perf_counter()
    image = image.convert("RGB")
    width, height = image.size

    if TILE_SIZE <= 0:
        raise ValueError("TILE_SIZE must be greater than zero")
    if not 0 <= TILE_OVERLAP < 1:
        raise ValueError("TILE_OVERLAP must be between 0 and 1")

    stride = max(1, round(TILE_SIZE * (1.0 - TILE_OVERLAP)))
    left_starts = tile_starts(width, TILE_SIZE, stride)
    top_starts = tile_starts(height, TILE_SIZE, stride)
    tile_count = len(left_starts) * len(top_starts)
    class_count = segmentation_model.model.config.num_labels

    # Keep the full-resolution accumulator on the GPU when it fits. This avoids
    # copying every tile's logits to host RAM. Leave headroom for the model and
    # current tile activations; small cards continue to use the CPU accumulator.
    accumulator_device = torch.device("cpu")
    accumulator_bytes = class_count * height * width * torch.tensor([], dtype=torch.float32).element_size()
    if segmentation_model.device.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info(segmentation_model.device)
        headroom_bytes = max(2 * 1024**3, int(free_bytes * 0.25))
        if free_bytes - accumulator_bytes >= headroom_bytes:
            accumulator_device = segmentation_model.device
    accumulated_logits = torch.zeros(
        (class_count, height, width), dtype=torch.float32, device=accumulator_device
    )
    print(
        f"Logit accumulator: {accumulator_device} "
        f"({accumulator_bytes / 1024**3:.2f} GiB); model inference: {segmentation_model.device}"
    )

    observer = BENCHMARK_OBSERVER
    if observer is not None:
        observer.segmentation_metadata(width, height, tile_count, str(accumulator_device), str(segmentation_model.device))

    def measure(name):
        return observer.measure(name, segmentation_model.device) if observer else nullcontext()

    for top in top_starts:
        for left in left_starts:
            right = min(width, left + TILE_SIZE)
            bottom = min(height, top + TILE_SIZE)
            with measure("tile_prepare"):
                tile = image.crop((left, top, right, bottom))
                inputs = segmentation_model.processor(images=tile, return_tensors="pt")
                inputs = {name: value.to(segmentation_model.device) for name, value in inputs.items()}
            with measure("tile_model"):
                logits = model_logits(segmentation_model, inputs)
            with measure("tile_accumulate"):
                logits = torch.nn.functional.interpolate(
                    logits, size=(bottom - top, right - left), mode="bilinear", align_corners=False,
                )[0]
                accumulated_logits[:, top:bottom, left:right] += logits.to(accumulator_device)

    with measure("mask_finalize"):
        result = accumulated_logits.argmax(dim=0).to(device="cpu", dtype=torch.uint8)

    if LOG_TILE_TIMING:
        elapsed = time.perf_counter() - started
        print(
            f"Segmentation tiles: {tile_count} ({width}x{height}), "
            f"completed in {elapsed:.3f}s"
        )
    return result


@torch.inference_mode()
def run_segmentation(
    image: Image.Image,
    segmentation_model: SegmentationModel,
    *, output_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Return a predicted class-ID map, optionally at (width, height)."""
    try:
        if MAX_SEGMENTATION_SIDE is not None and USE_OVERLAPPING_TILES:
            if MAX_SEGMENTATION_SIDE < 1:
                raise ValueError("MAX_SEGMENTATION_SIDE must be positive")
            longest = max(image.size)
            if longest > MAX_SEGMENTATION_SIDE:
                scale = MAX_SEGMENTATION_SIDE / longest
                reduced = image.resize((max(1, round(image.width * scale)),
                                        max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
                class_map = run_overlapping_segmentation(reduced, segmentation_model)
                resized = Image.fromarray(class_map.numpy(), mode="L").resize(image.size, Image.Resampling.NEAREST)
                class_map = torch.from_numpy(np.asarray(resized, dtype=np.uint8).copy())
                if output_size is not None and output_size != image.size:
                    projected = Image.fromarray(class_map.numpy(), mode="L").resize(
                        output_size, Image.Resampling.NEAREST
                    )
                    class_map = torch.from_numpy(np.asarray(projected, dtype=np.uint8).copy())
                return class_map
        if USE_OVERLAPPING_TILES:
            class_map = run_overlapping_segmentation(image, segmentation_model)
            if output_size is not None and output_size != image.size:
                projected = Image.fromarray(class_map.numpy(), mode="L").resize(
                    output_size, Image.Resampling.NEAREST
                )
                class_map = torch.from_numpy(np.asarray(projected, dtype=np.uint8).copy())
            return class_map

        if BENCHMARK_OBSERVER is not None:
            BENCHMARK_OBSERVER.segmentation_metadata(image.width, image.height, 1, "none", str(segmentation_model.device))
        inputs = segmentation_model.processor(images=image.convert("RGB"), return_tensors="pt")
        inputs = {name: value.to(segmentation_model.device) for name, value in inputs.items()}
        logits = model_logits(segmentation_model, inputs)
        if SINGLE_IMAGE_LOGITS_RESOLUTION is not None:
            if SINGLE_IMAGE_LOGITS_RESOLUTION < 1:
                raise ValueError("SINGLE_IMAGE_LOGITS_RESOLUTION must be positive")
            target = ((image.height, image.width) if max(image.size) <= SINGLE_IMAGE_LOGITS_RESOLUTION
                      else (SINGLE_IMAGE_LOGITS_RESOLUTION, SINGLE_IMAGE_LOGITS_RESOLUTION))
            logits = torch.nn.functional.interpolate(
                logits, size=target,
                mode="bilinear", align_corners=False,
            )
        class_map = logits.argmax(dim=1)[0].to(torch.uint8).cpu().numpy()
        target_size = output_size or image.size
        resized = Image.fromarray(class_map, mode="L").resize(target_size, Image.Resampling.NEAREST)
        return torch.from_numpy(np.asarray(resized, dtype=np.uint8).copy())
    except RuntimeError as error:
        error_text = str(error).lower()
        cuda_retryable_error = any(
            message in error_text
            for message in ("engine to execute", "out of memory", "cuda out of memory")
        )
        if segmentation_model.device.type != "cuda" or not cuda_retryable_error:
            raise
        print("CUDA inference unavailable or out of memory; retrying this image on CPU.")
        segmentation_model.device = torch.device("cpu")
        segmentation_model.model.to(segmentation_model.device)
        return run_segmentation(image, segmentation_model, output_size=output_size)


def extract_water_mask(class_map: torch.Tensor, water_class_ids: Iterable[int]) -> np.ndarray:
    mask = torch.zeros_like(class_map, dtype=torch.bool)
    for class_id in water_class_ids:
        mask |= class_map == class_id
    return mask.numpy().astype(np.uint8)
