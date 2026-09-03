"""Reusable pretrained water-segmentation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from pillow_heif import register_heif_opener

_TORCHVISION_COMPAT_LIBRARY = None


def register_torchvision_compat_ops() -> None:
    """Allow Transformers to import with a mismatched TorchVision wheel."""
    global _TORCHVISION_COMPAT_LIBRARY
    try:
        _TORCHVISION_COMPAT_LIBRARY = torch.library.Library("torchvision", "DEF")
        for operator in ("nms", "qnms"):
            try:
                _TORCHVISION_COMPAT_LIBRARY.define(
                    f"{operator}(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor"
                )
            except RuntimeError:
                pass
    except RuntimeError:
        pass


register_torchvision_compat_ops()

from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

MODEL_NAME = "nvidia/segformer-b2-finetuned-ade-512-512"
WATER_CLASS_NAMES = {"water", "sea", "river", "lake"}
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}
register_heif_opener()


@dataclass
class SegmentationModel:
    model: SegformerForSemanticSegmentation
    processor: SegformerImageProcessor
    water_class_ids: tuple[int, ...]
    device: torch.device


def configure_inference_device(device: str | None = None) -> torch.device:
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if selected_device.type != "cuda":
        return selected_device
    try:
        torch.backends.cudnn.version()
    except RuntimeError as error:
        if "not compatible with devices with sm" not in str(error).lower():
            raise
        torch.backends.cudnn.enabled = False
        print(f"cuDNN disabled for {torch.cuda.get_device_name(selected_device)}; using native CUDA kernels.")
    return selected_device


def load_model(device: str | None = None) -> SegmentationModel:
    """Load the pretrained model used by both preprocessing and previews."""
    selected_device = configure_inference_device(device)
    processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME).to(selected_device)
    model.eval()
    id2label = {int(key): value.lower().strip() for key, value in model.config.id2label.items()}
    water_class_ids = tuple(i for i, label in id2label.items() if label in WATER_CLASS_NAMES)
    if not water_class_ids:
        raise RuntimeError(f"{MODEL_NAME} does not expose expected water classes: {WATER_CLASS_NAMES}")
    print(f"Model: {MODEL_NAME} on {selected_device}")
    print("Water classes: " + ", ".join(f"{id2label[i]}={i}" for i in water_class_ids))
    return SegmentationModel(model, processor, water_class_ids, selected_device)


def load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        return opened.convert("RGB")


@torch.inference_mode()
def run_segmentation(image: Image.Image, segmentation_model: SegmentationModel) -> torch.Tensor:
    """Return a source-resolution predicted class-ID map."""
    try:
        inputs = segmentation_model.processor(images=image.convert("RGB"), return_tensors="pt")
        inputs = {name: value.to(segmentation_model.device) for name, value in inputs.items()}
        logits = segmentation_model.model(**inputs).logits
        class_map = logits.argmax(dim=1)[0].to(torch.uint8).cpu().numpy()
        resized = Image.fromarray(class_map, mode="L").resize(image.size, Image.Resampling.NEAREST)
        return torch.from_numpy(np.asarray(resized, dtype=np.uint8).copy())
    except RuntimeError as error:
        if segmentation_model.device.type != "cuda" or "engine to execute" not in str(error).lower():
            raise
        print("CUDA inference backend unavailable; retrying on CPU for remaining images.")
        segmentation_model.device = torch.device("cpu")
        segmentation_model.model.to(segmentation_model.device)
        return run_segmentation(image, segmentation_model)


def extract_water_mask(class_map: torch.Tensor, water_class_ids: Iterable[int]) -> np.ndarray:
    mask = torch.zeros_like(class_map, dtype=torch.bool)
    for class_id in water_class_ids:
        mask |= class_map == class_id
    return mask.numpy().astype(np.uint8)


def iter_images(input_dir: Path) -> list[Path]:
    return sorted(
        (p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS),
        key=lambda p: p.name.lower(),
    )
