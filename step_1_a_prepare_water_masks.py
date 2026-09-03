#!/usr/bin/env python3
"""Create visual water-segmentation previews for images in ./step-0-raw-data/."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

from step_1_a_water_segmentation import extract_water_mask, iter_images, load_model, load_rgb_image, run_segmentation

OVERLAY_COLOR = np.array([0, 190, 255], dtype=np.uint8)
OVERLAY_ALPHA = 0.48


def make_overlay(image: Image.Image, water_mask: np.ndarray) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    blended = rgb * (1.0 - OVERLAY_ALPHA) + OVERLAY_COLOR * OVERLAY_ALPHA
    output = np.where(water_mask[..., None].astype(bool), blended, rgb).clip(0, 255).astype(np.uint8)
    return Image.fromarray(output, mode="RGB")


def save_preview(image: Image.Image, water_mask: np.ndarray, source_path: Path, output_dir: Path) -> tuple[Path, Path]:
    stem = source_path.stem
    original_path = output_dir / f"{stem}_original.jpg"
    overlay_path = output_dir / f"{stem}_overlay.jpg"
    image.save(original_path, quality=92)
    Image.fromarray(water_mask * 255, mode="L").save(output_dir / f"{stem}_mask.png")
    make_overlay(image, water_mask).save(overlay_path, quality=92)
    return original_path, overlay_path


def preview_paths(source_path: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    """Return the three files that make a preview complete."""
    stem = source_path.stem
    return (
        output_dir / f"{stem}_original.jpg",
        output_dir / f"{stem}_mask.png",
        output_dir / f"{stem}_overlay.jpg",
    )


def write_gallery(entries: list[tuple[str, Path, Path]], output_dir: Path) -> None:
    cards = []
    for filename, original_path, overlay_path in entries:
        cards.append(
            f'<article class="card"><h2>{filename}</h2><div class="images">'
            f'<a href="{original_path.name}" target="_blank"><img src="{original_path.name}" alt="Original {filename}" loading="lazy"></a>'
            f'<a href="{overlay_path.name}" target="_blank"><img src="{overlay_path.name}" alt="Water overlay for {filename}" loading="lazy"></a>'
            '</div></article>'
        )
    text = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Water mask previews</title><style>*{box-sizing:border-box}body{margin:0;padding:14px;background:#eef2f5;color:#17212b;font:14px system-ui,sans-serif}h1{font-size:20px;margin:0 0 12px}.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:12px}.card{background:#fff;padding:9px;border-radius:7px;box-shadow:0 1px 4px #0002}.card h2{font-size:13px;font-weight:600;margin:0 0 7px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.images{display:grid;grid-template-columns:1fr 1fr;gap:7px}.images a{display:block;background:#dce3e8;aspect-ratio:4/3;overflow:hidden}.images img{width:100%;height:100%;object-fit:contain;display:block}@media(max-width:500px){.gallery{grid-template-columns:1fr}}</style></head><body><h1>Water mask previews</h1><main class="gallery">''' + ''.join(cards) + '</main></body></html>'
    (output_dir / "index.html").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("step-0-raw-data"))
    parser.add_argument("--output-dir", type=Path, default=Path("water-mask-preview"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = iter_images(args.input_dir)
    if not image_paths:
        raise SystemExit(f"No supported images found in {args.input_dir}")
    model = None
    entries = []
    already_present = 0
    processing_seconds = 0.0
    for index, source_path in enumerate(image_paths, 1):
        image_started = time.perf_counter()
        try:
            original_path, mask_path, overlay_path = preview_paths(source_path, args.output_dir)
            if original_path.exists() and mask_path.exists() and overlay_path.exists():
                print(f"[{index}/{len(image_paths)}] {source_path.name}: skipped (preview already exists)")
                entries.append((source_path.name, original_path, overlay_path))
                already_present += 1
                continue
            if model is None:
                model = load_model(args.device)
            image = load_rgb_image(source_path)
            mask = extract_water_mask(run_segmentation(image, model), model.water_class_ids)
            original, overlay = save_preview(image, mask, source_path, args.output_dir)
            elapsed = time.perf_counter() - image_started
            processing_seconds += elapsed
            print(f"[{index}/{len(image_paths)}] {source_path.name}: water={mask.mean() * 100:.1f}% ({elapsed:.3f}s)")
            entries.append((source_path.name, original, overlay))
        except Exception as error:
            elapsed = time.perf_counter() - image_started
            processing_seconds += elapsed
            print(f"[{index}/{len(image_paths)}] {source_path.name}: ERROR: {error} ({elapsed:.3f}s)")
    write_gallery(entries, args.output_dir)
    print(f"Generated: {len(entries) - already_present}")
    print(f"Already present: {already_present}")
    print(f"Processing time: {processing_seconds:.3f}s")
    print(f"Gallery written to {args.output_dir / 'index.html'} ({len(entries)} images)")


if __name__ == "__main__":
    main()
