#!/usr/bin/env python3
"""Create visual water-segmentation previews for images in ./raw-data/."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from pillow_heif import register_heif_opener


# Some TorchVision wheels fail during import when their CUDA build does not
# exactly match the installed PyTorch wheel. Transformers imports TorchVision
# for image utilities, although SegFormer inference does not use these NMS ops.
_TORCHVISION_COMPAT_LIBRARY = None


def register_torchvision_compat_ops() -> None:
    """Allow Transformers' image utilities to import with a mismatched TV wheel."""
    global _TORCHVISION_COMPAT_LIBRARY
    try:
        _TORCHVISION_COMPAT_LIBRARY = torch.library.Library("torchvision", "DEF")
        for operator in ("nms", "qnms"):
            try:
                _TORCHVISION_COMPAT_LIBRARY.define(
                    f"{operator}(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor"
                )
            except RuntimeError:
                # A correctly installed TorchVision may already have the operator.
                pass
    except RuntimeError:
        # The namespace/library is already registered; no compatibility work is needed.
        pass


register_torchvision_compat_ops()

from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


MODEL_NAME = "nvidia/segformer-b2-finetuned-ade-512-512"
WATER_CLASS_NAMES = {"water", "sea", "river", "lake"}
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}
OVERLAY_COLOR = np.array([0, 190, 255], dtype=np.uint8)
OVERLAY_ALPHA = 0.48

register_heif_opener()


@dataclass
class SegmentationModel:
    model: SegformerForSemanticSegmentation
    processor: SegformerImageProcessor
    water_class_ids: tuple[int, ...]
    device: torch.device


def configure_inference_device(device: str | None = None) -> torch.device:
    """Select the device and keep older CUDA GPUs usable without cuDNN."""
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if selected_device.type != "cuda":
        return selected_device

    try:
        torch.backends.cudnn.version()
    except RuntimeError as error:
        if "not compatible with devices with sm" not in str(error).lower():
            raise
        # Recent cuDNN builds reject older GPUs such as the GTX 1050 (SM 6.1),
        # but PyTorch's native CUDA convolution kernels still work on them.
        torch.backends.cudnn.enabled = False
        print(f"cuDNN disabled for {torch.cuda.get_device_name(selected_device)}; using native CUDA kernels.")
    return selected_device


def load_model(device: str | None = None) -> SegmentationModel:
    """Load the pretrained semantic-segmentation model and resolve water labels."""
    selected_device = configure_inference_device(device)
    processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME).to(selected_device)
    model.eval()

    id2label = {int(key): value.lower().strip() for key, value in model.config.id2label.items()}
    water_class_ids = tuple(class_id for class_id, label in id2label.items() if label in WATER_CLASS_NAMES)
    if not water_class_ids:
        raise RuntimeError(f"{MODEL_NAME} does not expose any expected water classes: {WATER_CLASS_NAMES}")
    print(f"Model: {MODEL_NAME} on {selected_device}")
    print("Water classes: " + ", ".join(f"{id2label[i]}={i}" for i in water_class_ids))
    return SegmentationModel(model, processor, water_class_ids, selected_device)


def preprocess_image(image: Image.Image, processor: SegformerImageProcessor) -> dict[str, torch.Tensor]:
    """Convert a PIL image into the tensors expected by SegFormer."""
    return processor(images=image.convert("RGB"), return_tensors="pt")


@torch.inference_mode()
def run_segmentation(image: Image.Image, segmentation_model: SegmentationModel) -> torch.Tensor:
    """Return the predicted class ID for every pixel at the source image size."""
    try:
        inputs = preprocess_image(image, segmentation_model.processor)
        inputs = {name: value.to(segmentation_model.device) for name, value in inputs.items()}
        logits = segmentation_model.model(**inputs).logits
        # Upsampling all 150 ADE20K logits to a high-resolution phone photo can
        # require many GiB. Collapse to one class-ID map first, then resize it.
        class_map = logits.argmax(dim=1)[0].to(torch.uint8).cpu().numpy()
        resized = Image.fromarray(class_map, mode="L").resize(image.size, Image.Resampling.NEAREST)
        return torch.from_numpy(np.asarray(resized, dtype=np.uint8).copy())
    except RuntimeError as error:
        # Older GPUs or mismatched CUDA wheels can report this even when
        # torch.cuda.is_available() is true. Retry the current image on CPU,
        # then keep subsequent images on CPU as well.
        if segmentation_model.device.type != "cuda" or "engine to execute" not in str(error).lower():
            raise
        print("CUDA inference backend unavailable; retrying on CPU for remaining images.")
        segmentation_model.device = torch.device("cpu")
        segmentation_model.model.to(segmentation_model.device)
        return run_segmentation(image, segmentation_model)


def extract_water_mask(class_map: torch.Tensor, water_class_ids: Iterable[int]) -> np.ndarray:
    """Convert the multiclass prediction into a reusable binary uint8 water mask."""
    mask = torch.zeros_like(class_map, dtype=torch.bool)
    for class_id in water_class_ids:
        mask |= class_map == class_id
    return mask.numpy().astype(np.uint8)


def make_overlay(image: Image.Image, water_mask: np.ndarray) -> Image.Image:
    """Tint water pixels while retaining the original scene for visual inspection."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    tint = np.broadcast_to(OVERLAY_COLOR, rgb.shape).astype(np.float32)
    blended = rgb * (1.0 - OVERLAY_ALPHA) + tint * OVERLAY_ALPHA
    output = np.where(water_mask[..., None].astype(bool), blended, rgb).clip(0, 255).astype(np.uint8)
    return Image.fromarray(output, mode="RGB")


def iter_images(input_dir: Path) -> list[Path]:
    return sorted(
        (path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS),
        key=lambda path: path.name.lower(),
    )


def save_preview(image: Image.Image, water_mask: np.ndarray, source_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """Save browser-compatible original, binary mask, and overlay assets."""
    stem = source_path.stem
    original_path = output_dir / f"{stem}_original.jpg"
    overlay_path = output_dir / f"{stem}_overlay.jpg"
    image.convert("RGB").save(original_path, quality=92)
    Image.fromarray(water_mask * 255, mode="L").save(output_dir / f"{stem}_mask.png")
    make_overlay(image, water_mask).save(overlay_path, quality=92)
    return original_path, overlay_path


def write_gallery(entries: list[tuple[str, Path, Path]], output_dir: Path) -> None:
    cards = []
    for filename, original_path, overlay_path in entries:
        cards.append(
            f'''<article class="card"><h2>{filename}</h2><div class="images">'''
            f'''<a href="{original_path.name}" target="_blank"><img src="{original_path.name}" alt="Original {filename}" loading="lazy"></a>'''
            f'''<a href="{overlay_path.name}" target="_blank"><img src="{overlay_path.name}" alt="Water overlay for {filename}" loading="lazy"></a>'''
            "</div></article>"
        )
    html = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Water mask previews</title><style>
*{box-sizing:border-box}body{margin:0;padding:14px;background:#eef2f5;color:#17212b;font:14px system-ui,sans-serif}
h1{font-size:20px;margin:0 0 12px}.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:12px}
.card{background:#fff;padding:9px;border-radius:7px;box-shadow:0 1px 4px #0002}.card h2{font-size:13px;font-weight:600;margin:0 0 7px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.images{display:grid;grid-template-columns:1fr 1fr;gap:7px}.images a{display:block;background:#dce3e8;aspect-ratio:4/3;overflow:hidden}.images img{width:100%;height:100%;object-fit:contain;display:block}
@media(max-width:500px){.gallery{grid-template-columns:1fr}}
</style></head><body><h1>Water mask previews</h1><main class="gallery">""" + "".join(cards) + "</main></body></html>"
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("raw-data"))
    parser.add_argument("--output-dir", type=Path, default=Path("water-mask-preview"))
    parser.add_argument("--device", default=None, help="torch device, e.g. cpu or cuda")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = iter_images(args.input_dir)
    if not image_paths:
        raise SystemExit(f"No supported images found in {args.input_dir}")

    segmentation_model = load_model(args.device)
    gallery_entries = []
    for index, source_path in enumerate(image_paths, start=1):
        try:
            with Image.open(source_path) as opened:
                image = opened.convert("RGB")
            class_map = run_segmentation(image, segmentation_model)
            water_mask = extract_water_mask(class_map, segmentation_model.water_class_ids)
            original_path, overlay_path = save_preview(image, water_mask, source_path, args.output_dir)
            percentage = float(water_mask.mean() * 100.0)
            detected = "yes" if water_mask.any() else "no"
            print(f"[{index}/{len(image_paths)}] {source_path.name}: water detected={detected}, water={percentage:.1f}%")
            gallery_entries.append((source_path.name, original_path, overlay_path))
        except Exception as error:
            print(f"[{index}/{len(image_paths)}] {source_path.name}: ERROR: {error}")

    write_gallery(gallery_entries, args.output_dir)
    print(f"Gallery written to {args.output_dir / 'index.html'} ({len(gallery_entries)} images)")


if __name__ == "__main__":
    main()
