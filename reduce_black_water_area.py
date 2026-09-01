#!/usr/bin/env python3
"""Reduce black borders in processed water images without losing lower waves."""

from __future__ import annotations

import argparse
import html
import os
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import numpy as np
from PIL import Image

from water_segmentation import (
    extract_water_mask,
    iter_images,
    load_model,
    load_rgb_image,
    run_segmentation,
)

OUTPUT_SIZE = (224, 224)
BLACK_THRESHOLD = 12
LOWER_WATER_FRACTION = 0.35
MINIMUM_LOWER_RETENTION = 0.95
BLACK_RATIO_BUCKET = 0.005


@dataclass(frozen=True)
class CropMetrics:
    box: tuple[int, int, int, int]
    water_ratio: float
    black_ratio: float
    lower_water_retention: float
    water_pixels: int
    shape_penalty: float


def summed_area(mask: np.ndarray) -> np.ndarray:
    """Return a padded summed-area table for constant-time rectangle sums."""
    return np.pad(mask.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)


def rectangle_sum(table: np.ndarray, box: tuple[int, int, int, int]) -> int:
    left, top, right, bottom = box
    return int(table[bottom, right] - table[top, right] - table[bottom, left] + table[top, left])


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def resized_content_size(width: int, height: int) -> tuple[int, int]:
    scale = min(OUTPUT_SIZE[0] / width, OUTPUT_SIZE[1] / height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def projected_ratio(pixel_count: int, crop_area: int, width: int, height: int) -> float:
    """Estimate a crop ratio after aspect-preserving resize and square padding."""
    resized_width, resized_height = resized_content_size(width, height)
    occupied_fraction = resized_width * resized_height / (OUTPUT_SIZE[0] * OUTPUT_SIZE[1])
    return (pixel_count / crop_area) * occupied_fraction


def select_crop(water_mask: np.ndarray, black_mask: np.ndarray) -> CropMetrics | None:
    """Select a lower-anchored rectangle using lexicographic project priorities."""
    bbox = mask_bbox(water_mask)
    if bbox is None:
        return None
    bbox_left, bbox_top, bbox_right, bbox_bottom = bbox
    bbox_width, bbox_height = bbox_right - bbox_left, bbox_bottom - bbox_top
    minimum_water_pixels = max(32, round(water_mask.size * 0.0005))
    if int(water_mask.sum()) < minimum_water_pixels:
        return None

    lower_top = max(bbox_top, bbox_bottom - max(1, round(bbox_height * LOWER_WATER_FRACTION)))
    lower_mask = water_mask.copy()
    lower_mask[:lower_top, :] = False
    lower_total = int(lower_mask.sum())
    if lower_total < minimum_water_pixels:
        return None

    # Four-pixel horizontal sampling is exhaustive enough at the 224-pixel
    # pipeline resolution while keeping candidate evaluation inexpensive.
    horizontal_step = max(1, round(water_mask.shape[1] / 56))
    boundaries = sorted(
        set(range(bbox_left, bbox_right + 1, horizontal_step)) | {bbox_left, bbox_right}
    )
    tops = list(range(bbox_top, lower_top + 1))

    water_table = summed_area(water_mask)
    black_table = summed_area(black_mask)
    lower_table = summed_area(lower_mask)
    minimum_crop_area = max(256, round(bbox_width * bbox_height * 0.10))
    candidates: list[CropMetrics] = []

    # The bottom is fixed at the lowest detected water row. Tops cannot enter the
    # lower-water band, so every accepted crop includes that band geometrically.
    for left_index, left in enumerate(boundaries[:-1]):
        for right in boundaries[left_index + 1:]:
            width = right - left
            if width < 8:
                continue
            for top in tops:
                height = bbox_bottom - top
                crop_area = width * height
                if height < 8 or crop_area < minimum_crop_area:
                    continue
                box = (left, top, right, bbox_bottom)
                lower_in_field_of_view = rectangle_sum(
                    lower_table, (left, lower_top, right, bbox_bottom)
                )
                if lower_in_field_of_view < minimum_water_pixels:
                    continue
                lower_retention = rectangle_sum(lower_table, box) / lower_in_field_of_view
                if lower_retention + 1e-12 < MINIMUM_LOWER_RETENTION:
                    continue
                water_pixels = rectangle_sum(water_table, box)
                black_pixels = rectangle_sum(black_table, box)
                water_ratio = projected_ratio(water_pixels, crop_area, width, height)
                black_ratio = 1.0 - projected_ratio(crop_area - black_pixels, crop_area, width, height)
                shape_penalty = abs(np.log(width / height))
                candidates.append(
                    CropMetrics(box, water_ratio, black_ratio, lower_retention, water_pixels, shape_penalty)
                )

    if not candidates:
        return None

    def priority(candidate: CropMetrics) -> tuple[int, int, int, float, float, tuple[int, int, int, int]]:
        # Retention tiers dominate. Within a tier, 0.5-point black-ratio bands
        # prevent tiny cleanliness gains from discarding a large useful region.
        retention_tier = (
            3 if candidate.lower_water_retention >= 0.995
            else 2 if candidate.lower_water_retention >= 0.98
            else 1
        )
        black_bucket = round(candidate.black_ratio / BLACK_RATIO_BUCKET)
        return (
            -retention_tier,
            black_bucket,
            -candidate.water_pixels,
            -candidate.water_ratio,
            candidate.shape_penalty,
            candidate.box,
        )

    return min(candidates, key=priority)


def fit_in_square(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    """Crop, resize without distortion, and center-pad to 224x224."""
    cropped = image.crop(box)
    resized = cropped.resize(resized_content_size(*cropped.size), Image.Resampling.LANCZOS)
    output = Image.new("RGB", OUTPUT_SIZE, (0, 0, 0))
    output.paste(resized, ((OUTPUT_SIZE[0] - resized.width) // 2, (OUTPUT_SIZE[1] - resized.height) // 2))
    return output


def project_mask(mask: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Apply the same crop/fit operation to a binary mask."""
    left, top, right, bottom = box
    cropped = Image.fromarray(mask[top:bottom, left:right].astype(np.uint8) * 255, mode="L")
    resized = cropped.resize(resized_content_size(*cropped.size), Image.Resampling.NEAREST)
    output = Image.new("L", OUTPUT_SIZE, 0)
    output.paste(resized, ((OUTPUT_SIZE[0] - resized.width) // 2, (OUTPUT_SIZE[1] - resized.height) // 2))
    return np.asarray(output) > 0


def black_ratio(image: Image.Image) -> float:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return float(np.all(rgb <= BLACK_THRESHOLD, axis=2).mean())


def save_result(image: Image.Image, path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image.save(path, format="JPEG", quality=95, subsampling=0)
    elif suffix in {".tif", ".tiff"}:
        image.save(path, format="TIFF")
    else:
        image.save(path)


def write_gallery(entries: list[tuple[str, Path]], input_dir: Path, output_dir: Path) -> None:
    cards = []
    for filename, result_path in sorted(entries, key=lambda entry: entry[0].lower()):
        safe_name = html.escape(filename)
        current_url = quote(os.path.relpath(input_dir / filename, output_dir).replace(os.sep, "/"))
        result_url = quote(result_path.name)
        cards.append(
            f'<article class="card"><h2>{safe_name}</h2><div class="images">'
            f'<figure><figcaption>current processed image</figcaption><a href="{current_url}" target="_blank">'
            f'<img src="{current_url}" alt="Current processed image {safe_name}" loading="lazy"></a></figure>'
            f'<figure><figcaption>new reduced-black result</figcaption><a href="{result_url}" target="_blank">'
            f'<img src="{result_url}" alt="Reduced-black result {safe_name}" loading="lazy"></a></figure>'
            '</div></article>'
        )
    document = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Reduced-black water images</title><style>*{box-sizing:border-box}body{margin:0;padding:16px;background:#eef2f5;color:#17212b;font:14px system-ui,sans-serif}h1{font-size:21px;margin:0 0 14px}.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:14px}.card{min-width:0;background:#fff;padding:10px;border-radius:8px;box-shadow:0 1px 5px #0002}.card h2{font-size:13px;margin:0 0 9px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.images{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}figure{margin:0;min-width:0}figcaption{font-size:12px;color:#52606d;margin-bottom:5px}figure a{display:block;background:#111;aspect-ratio:1/1;overflow:hidden}img{display:block;width:100%;height:100%;object-fit:contain}@media(max-width:560px){.gallery{grid-template-columns:1fr}.images{gap:6px}}</style></head><body><h1>Reduced-black water images</h1><main class="gallery">''' + ''.join(cards) + '</main></body></html>'
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def diagnostic_line(
    filename: str,
    original_black: float | None,
    final_black: float | None,
    water_ratio: float | None,
    retention: float | None,
    box: tuple[int, int, int, int] | None,
    status: str,
) -> str:
    percentage = lambda value: "n/a" if value is None else f"{value * 100:.2f}%"
    return (
        f"filename={filename} | original black ratio={percentage(original_black)} | "
        f"final black ratio={percentage(final_black)} | water ratio={percentage(water_ratio)} | "
        f"lower-water retention={percentage(retention)} | crop coordinates={box or 'n/a'} | status={status}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("processed-data"))
    parser.add_argument("--output-dir", type=Path, default=Path("final-water-data"))
    parser.add_argument("--device", default=None, help="torch device, e.g. cpu or cuda")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = iter_images(args.input_dir)
    if not image_paths:
        raise SystemExit(f"No supported images found in {args.input_dir}")

    model = None
    entries: list[tuple[str, Path]] = []
    processed = 0
    skipped = 0
    total_started = time.perf_counter()
    for index, source_path in enumerate(image_paths, 1):
        item_started = time.perf_counter()
        result_path = args.output_dir / source_path.name
        if result_path.exists():
            skipped += 1
            entries.append((source_path.name, result_path))
            print(f"[{index}/{len(image_paths)}] " + diagnostic_line(
                source_path.name, None, None, None, None, None,
                "skipped: output already exists",
            ) + f" | time={time.perf_counter() - item_started:.3f}s")
            continue

        try:
            if model is None:
                model = load_model(args.device)
            image = load_rgb_image(source_path)
            rgb = np.asarray(image, dtype=np.uint8)
            source_black_mask = np.all(rgb <= BLACK_THRESHOLD, axis=2)
            segmentation_mask = extract_water_mask(run_segmentation(image, model), model.water_class_ids).astype(bool)
            water_mask = segmentation_mask & ~source_black_mask
            crop = select_crop(water_mask, source_black_mask)
            if crop is None:
                skipped += 1
                reason = "no reliable lower-water crop"
                print(f"[{index}/{len(image_paths)}] " + diagnostic_line(
                    source_path.name, float(source_black_mask.mean()), None, None, None, None,
                    f"skipped: {reason}",
                ) + f" | time={time.perf_counter() - item_started:.3f}s")
                continue

            result = fit_in_square(image, crop.box)
            save_result(result, result_path)
            # Measure the persisted file, including square padding and encoding effects.
            persisted = load_rgb_image(result_path)
            final_black = black_ratio(persisted)
            final_water_mask = project_mask(water_mask, crop.box)
            final_water_ratio = float(final_water_mask.mean())
            entries.append((source_path.name, result_path))
            processed += 1
            elapsed = time.perf_counter() - item_started
            print(f"[{index}/{len(image_paths)}] " + diagnostic_line(
                source_path.name,
                float(source_black_mask.mean()),
                final_black,
                final_water_ratio,
                crop.lower_water_retention,
                crop.box,
                "processed",
            ) + f" | time={elapsed:.3f}s")
        except Exception as error:
            skipped += 1
            elapsed = time.perf_counter() - item_started
            print(f"[{index}/{len(image_paths)}] " + diagnostic_line(
                source_path.name, None, None, None, None, None,
                f"skipped: {error}",
            ) + f" | time={elapsed:.3f}s")

    write_gallery(entries, args.input_dir, args.output_dir)
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Total time: {time.perf_counter() - total_started:.3f}s")
    print(f"Gallery: {args.output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
