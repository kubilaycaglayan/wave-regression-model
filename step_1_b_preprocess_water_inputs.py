#!/usr/bin/env python3
"""Create deterministic 224x224 masked RGB model inputs and an inspection gallery."""

from __future__ import annotations

import argparse
import html
import os
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
from PIL import Image, ImageFilter

from step_1_a_water_segmentation import extract_water_mask, iter_images, load_model, load_rgb_image, run_segmentation

OUTPUT_SIZE = (224, 224)


def erosion_radius(image_size: tuple[int, int]) -> int:
    """Use a thin, bounded erosion proportional to the shorter image dimension."""
    width, height = image_size
    return max(1, min(12, round(min(width, height) * 0.005)))


def erode_water_mask(mask: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    radius = erosion_radius(image_size)
    eroded = Image.fromarray((mask.astype(bool) * 255).astype(np.uint8), mode="L").filter(
        ImageFilter.MinFilter(radius * 2 + 1)
    )
    return (np.asarray(eroded) > 0).astype(np.uint8)


def water_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask.astype(bool))
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def standardized_model_input(image: Image.Image, water_mask: np.ndarray) -> Image.Image | None:
    """Mask original RGB pixels, crop valid water, then fit inside black 224 square."""
    mask = erode_water_mask(water_mask, image.size)
    bbox = water_bbox(mask)
    minimum_pixels = max(32, round(image.width * image.height * 0.0005))
    if bbox is None or int(mask.sum()) < minimum_pixels:
        return None
    left, top, right, bottom = bbox
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    rgb[mask == 0] = 0
    cropped = Image.fromarray(rgb[top:bottom, left:right], mode="RGB")
    scale = min(OUTPUT_SIZE[0] / cropped.width, OUTPUT_SIZE[1] / cropped.height)
    resized_size = (max(1, round(cropped.width * scale)), max(1, round(cropped.height * scale)))
    resized = cropped.resize(resized_size, Image.Resampling.LANCZOS)
    output = Image.new("RGB", OUTPUT_SIZE, (0, 0, 0))
    output.paste(resized, ((OUTPUT_SIZE[0] - resized.width) // 2, (OUTPUT_SIZE[1] - resized.height) // 2))
    return output


def write_gallery(entries: list[tuple[str, Path, Path]], output_dir: Path) -> None:
    cards = []
    for filename, source, standardized in entries:
        safe_name = html.escape(filename)
        source_url = quote(os.path.relpath(source, output_dir).replace(os.sep, "/"))
        cards.append(
            f'<article class="card"><h2>{safe_name}</h2><div class="images">'
            f'<figure><figcaption>source image</figcaption><a href="{source_url}" target="_blank"><img src="{source_url}" alt="Source {safe_name}" loading="lazy"></a></figure>'
            f'<figure><figcaption>standardized model input</figcaption><a href="{standardized.name}" target="_blank"><img src="{standardized.name}" alt="Standardized model input {safe_name}" loading="lazy"></a></figure>'
            '</div></article>'
        )
    text = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Standardized water inputs</title><style>*{box-sizing:border-box}body{margin:0;padding:16px;background:#eef2f5;color:#17212b;font:14px system-ui,sans-serif}h1{font-size:21px;margin:0 0 14px}.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:14px}.card{background:#fff;padding:10px;border-radius:8px;box-shadow:0 1px 5px #0002}.card h2{font-size:13px;margin:0 0 9px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.images{display:grid;grid-template-columns:1fr 1fr;gap:10px}figure{margin:0}figcaption{font-size:12px;color:#52606d;margin-bottom:5px}figure a{display:block;background:#dce3e8;aspect-ratio:1/1;overflow:hidden}img{display:block;width:100%;height:100%;object-fit:contain}@media(max-width:560px){.gallery{grid-template-columns:1fr}}</style></head><body><h1>Standardized water inputs</h1><main class="gallery">''' + ''.join(cards) + '</main></body></html>'
    (output_dir / "index.html").write_text(text, encoding="utf-8")


def processed_paths(source_path: Path, output_dir: Path) -> Path:
    """Return the standardized output path; source images stay in the input step."""
    return output_dir / f"step-1_{source_path.stem}.jpg"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("step-0-raw-data"))
    parser.add_argument("--output-dir", type=Path, default=Path("step-1-processed-data"))
    parser.add_argument("--device", default=None, help="torch device, e.g. cpu or cuda")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = iter_images(args.input_dir)
    if not image_paths:
        raise SystemExit(f"No supported images found in {args.input_dir}")
    model = None
    entries, skipped = [], []
    already_present = 0
    processing_seconds = 0.0
    for index, source_path in enumerate(image_paths, 1):
        image_started = time.perf_counter()
        try:
            standardized_path = processed_paths(source_path, args.output_dir)
            if standardized_path.exists():
                print(f"[{index}/{len(image_paths)}] {source_path.name}: skipped (standardized input already exists)")
                entries.append((source_path.name, source_path, standardized_path))
                already_present += 1
                continue
            if model is None:
                model = load_model(args.device)
            image = load_rgb_image(source_path)
            mask = extract_water_mask(run_segmentation(image, model), model.water_class_ids)
            standardized = standardized_model_input(image, mask)
            if standardized is None:
                print(f"[{index}/{len(image_paths)}] {source_path.name}: WARNING: no reliable water region; skipped")
                skipped.append(source_path.name)
                continue
            standardized.save(standardized_path, format="JPEG", quality=95)
            entries.append((source_path.name, source_path, standardized_path))
            elapsed = time.perf_counter() - image_started
            processing_seconds += elapsed
            print(f"[{index}/{len(image_paths)}] {source_path.name}: processed ({elapsed:.3f}s)")
        except Exception as error:
            elapsed = time.perf_counter() - image_started
            processing_seconds += elapsed
            print(f"[{index}/{len(image_paths)}] {source_path.name}: WARNING: {error}; skipped ({elapsed:.3f}s)")
            skipped.append(source_path.name)
    write_gallery(entries, args.output_dir)
    print(f"Processed: {len(entries) - already_present}")
    print(f"Already present: {already_present}")
    print(f"Skipped: {len(skipped)}")
    print(f"Processing time: {processing_seconds:.3f}s")
    if skipped:
        print("Skipped files:")
        for filename in skipped:
            print(f"- {filename}")
    print(f"Gallery written to {args.output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
