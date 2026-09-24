#!/usr/bin/env python3
"""Create visual water-segmentation previews for images in ./step-0-raw-data/."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from data_sources import SourceOutputNames, configured_source_directories

from step_1_a_water_segmentation import extract_water_mask, iter_images, load_model, load_rgb_image, run_segmentation

OVERLAY_COLOR = np.array([0, 190, 255], dtype=np.uint8)
OVERLAY_ALPHA = 0.48


def make_overlay(image: Image.Image, water_mask: np.ndarray) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    blended = rgb * (1.0 - OVERLAY_ALPHA) + OVERLAY_COLOR * OVERLAY_ALPHA
    output = np.where(water_mask[..., None].astype(bool), blended, rgb).clip(0, 255).astype(np.uint8)
    return Image.fromarray(output, mode="RGB")


def save_preview(image: Image.Image, water_mask: np.ndarray, source_path: Path, output_dir: Path, names: SourceOutputNames | None = None) -> tuple[Path, Path]:
    stem = names.stem_for(source_path) if names else source_path.stem
    original_path = output_dir / f"{stem}_original.jpg"
    mask_path = output_dir / f"{stem}_mask.png"
    overlay_path = output_dir / f"{stem}_overlay.jpg"
    def save_atomically(target: Path, save_image) -> None:
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
            save_image(temporary_path)
            os.replace(temporary_path, target)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    save_atomically(original_path, lambda path: image.save(path, format="JPEG", quality=92))
    save_atomically(mask_path, lambda path: Image.fromarray(water_mask * 255, mode="L").save(path, format="PNG"))
    save_atomically(overlay_path, lambda path: make_overlay(image, water_mask).save(path, format="JPEG", quality=92))
    return original_path, overlay_path


def preview_paths(source_path: Path, output_dir: Path, names: SourceOutputNames | None = None) -> tuple[Path, Path, Path]:
    """Return the three files that make a preview complete."""
    stem = names.stem_for(source_path) if names else source_path.stem
    return (
        output_dir / f"{stem}_original.jpg",
        output_dir / f"{stem}_mask.png",
        output_dir / f"{stem}_overlay.jpg",
    )


def old_preview_paths(
    source_path: Path, output_dir: Path, names: SourceOutputNames | None = None
) -> tuple[Path, Path, Path]:
    """Return paths used to preserve a previous preview during comparison runs."""
    stem = names.stem_for(source_path) if names else source_path.stem
    return (
        output_dir / f"{stem}_old_original.jpg",
        output_dir / f"{stem}_old_mask.png",
        output_dir / f"{stem}_old_overlay.jpg",
    )


def preserve_old_preview(
    current_paths: tuple[Path, Path, Path], old_paths: tuple[Path, Path, Path]
) -> None:
    """Keep the first previous version so repeated comparisons remain stable."""
    for current_path, old_path in zip(current_paths, old_paths):
        if current_path.exists() and not old_path.exists():
            shutil.copy2(current_path, old_path)


def write_gallery(
    entries: list[tuple[str, Path, Path, Path | None, Path | None, Path | None]], output_dir: Path
) -> None:
    cards = []
    for filename, original_path, mask_path, overlay_path, old_mask_path, old_overlay_path in entries:
        if old_overlay_path and old_mask_path and old_overlay_path.exists() and old_mask_path.exists():
            images = (
                f'<figure><figcaption>original</figcaption><a href="{original_path.name}" target="_blank">'
                f'<img src="{original_path.name}" alt="Original {filename}" loading="lazy"></a></figure>'
                f'<figure><figcaption>previous processor · overlay</figcaption><a href="{old_overlay_path.name}" target="_blank">'
                f'<img src="{old_overlay_path.name}" alt="Previous water overlay for {filename}" loading="lazy"></a></figure>'
                f'<figure><figcaption>current processor · overlay</figcaption><a href="{overlay_path.name}" target="_blank">'
                f'<img src="{overlay_path.name}" alt="Current water overlay for {filename}" loading="lazy"></a></figure>'
                f'<figure><figcaption>previous processor · mask</figcaption><a href="{old_mask_path.name}" target="_blank">'
                f'<img src="{old_mask_path.name}" alt="Previous water mask for {filename}" loading="lazy"></a></figure>'
                f'<figure><figcaption>current processor · mask</figcaption><a href="{mask_path.name}" target="_blank">'
                f'<img src="{mask_path.name}" alt="Current water mask for {filename}" loading="lazy"></a></figure>'
            )
        else:
            images = (
                f'<figure><figcaption>original</figcaption><a href="{original_path.name}" target="_blank">'
                f'<img src="{original_path.name}" alt="Original {filename}" loading="lazy"></a></figure>'
                f'<figure><figcaption>water overlay</figcaption><a href="{overlay_path.name}" target="_blank">'
                f'<img src="{overlay_path.name}" alt="Water overlay for {filename}" loading="lazy"></a></figure>'
            )
        cards.append(f'<article class="card"><h2>{filename}</h2><div class="images">{images}</div></article>')
    text = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Water mask previews</title><style>*{box-sizing:border-box}body{margin:0;padding:14px;background:#eef2f5;color:#17212b;font:14px system-ui,sans-serif}h1{font-size:20px;margin:0 0 12px}.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(600px,1fr));gap:12px}.card{background:#fff;padding:9px;border-radius:7px;box-shadow:0 1px 4px #0002}.card h2{font-size:13px;font-weight:600;margin:0 0 7px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.images{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:7px}.images figure{margin:0;min-width:0}.images figcaption{font-size:11px;color:#52606d;margin:0 0 4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.images a{display:block;background:#dce3e8;aspect-ratio:4/3;overflow:hidden}.images img{width:100%;height:100%;object-fit:contain;display:block}@media(max-width:900px){.gallery{grid-template-columns:1fr}.images{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:500px){.images{grid-template-columns:1fr}}</style></head><body><h1>Water mask previews</h1><main class="gallery">''' + ''.join(cards) + '</main></body></html>'
    (output_dir / "index.html").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, action="append", help="Additional input directory; repeatable")
    parser.add_argument("--output-dir", type=Path, default=Path("water-mask-preview"))
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--skip-cache",
        action="store_true",
        help="Regenerate previews even when all preview files already exist",
    )
    parser.add_argument(
        "--keep-old",
        action="store_true",
        help="Preserve existing previews and show previous/current results side by side",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_dirs = configured_source_directories()
    if args.input_dir:
        source_dirs.extend(args.input_dir)
    image_paths = iter_images(source_dirs)
    if not image_paths:
        raise SystemExit(f"No supported images found in: {', '.join(str(path) for path in source_dirs)}")
    model = None
    names = SourceOutputNames(
        image_paths,
        args.output_dir,
        ("_original.jpg", "_mask.png", "_overlay.jpg"),
        output_paths_for_stem=lambda stem: (
            args.output_dir / f"{stem}_original.jpg",
            args.output_dir / f"{stem}_mask.png",
            args.output_dir / f"{stem}_overlay.jpg",
        ),
    )
    entries = []
    already_present = 0
    processing_seconds = 0.0
    for index, source_path in enumerate(image_paths, 1):
        image_started = time.perf_counter()
        try:
            original_path, mask_path, overlay_path = preview_paths(source_path, args.output_dir, names)
            names.save()
            old_original_path, old_mask_path, old_overlay_path = old_preview_paths(source_path, args.output_dir, names)
            current_paths = (original_path, mask_path, overlay_path)
            old_paths = (old_original_path, old_mask_path, old_overlay_path)
            if args.keep_old:
                preserve_old_preview(current_paths, old_paths)
            if not args.skip_cache and original_path.exists() and mask_path.exists() and overlay_path.exists():
                print(f"[{index}/{len(image_paths)}] {source_path.name}: skipped (preview already exists)")
                entries.append((source_path.name, original_path, mask_path, overlay_path, old_mask_path, old_overlay_path))
                already_present += 1
                continue
            if model is None:
                model = load_model(args.device)
            image = load_rgb_image(source_path)
            mask = extract_water_mask(run_segmentation(image, model), model.water_class_ids)
            original, overlay = save_preview(image, mask, source_path, args.output_dir, names)
            elapsed = time.perf_counter() - image_started
            processing_seconds += elapsed
            print(f"[{index}/{len(image_paths)}] {source_path.name}: water={mask.mean() * 100:.1f}% ({elapsed:.3f}s)")
            entries.append((source_path.name, original, mask_path, overlay, old_mask_path, old_overlay_path))
        except Exception as error:
            elapsed = time.perf_counter() - image_started
            processing_seconds += elapsed
            print(f"[{index}/{len(image_paths)}] {source_path.name}: ERROR: {error} ({elapsed:.3f}s)")
    write_gallery(entries, args.output_dir)
    names.save()
    print(f"Generated: {len(entries) - already_present}")
    print(f"Already present: {already_present}")
    print(f"Processing time: {processing_seconds:.3f}s")
    print(f"Gallery written to {args.output_dir / 'index.html'} ({len(entries)} images)")


if __name__ == "__main__":
    main()
