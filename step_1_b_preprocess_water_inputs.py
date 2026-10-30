#!/usr/bin/env python3
"""Create 224x224 lower-water model inputs and an inspection gallery."""

from __future__ import annotations

import argparse
import html
import io
import os
import tempfile
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import minimum_filter

from data_sources import SourceOutputNames, configured_source_directories

from step_1_a_water_segmentation import (
    SegmentationModel,
    configure_inference_device,
    extract_water_mask,
    iter_images,
    load_model,
    load_rgb_image,
    run_segmentation,
)
from step_1_b_parallel_workers import run_segmentation_tasks, select_worker_count
from step_1_c_preprocessing_profiles import (
    configure_segmentation,
    ensure_profile_directory,
    get_profile,
    write_preprocessing_contract,
)
import step_1_a_water_segmentation as segmentation

OUTPUT_SIZE = (224, 224)


def erosion_radius(image_size: tuple[int, int]) -> int:
    """Use a thin, bounded erosion proportional to the shorter image dimension."""
    width, height = image_size
    return max(1, min(12, round(min(width, height) * 0.005)))


def erode_water_mask(mask: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    radius = erosion_radius(image_size)
    # The separable minimum filter has the same edge replication as Pillow's
    # MinFilter, without its quadratic cost at full camera resolution.
    return minimum_filter(mask.astype(bool), size=radius * 2 + 1, mode="nearest").astype(np.uint8)


def water_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys = np.flatnonzero(np.any(mask, axis=1))
    xs = np.flatnonzero(np.any(mask, axis=0))
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def standardized_model_input(image: Image.Image, water_mask: np.ndarray) -> Image.Image | None:
    """Mask original RGB pixels, crop valid water, then fit inside black 224 square."""
    return _standardize_eroded_mask(image, erode_water_mask(water_mask, image.size))


def _standardize_eroded_mask(image: Image.Image, mask: np.ndarray) -> Image.Image | None:
    bbox = water_bbox(mask)
    minimum_pixels = max(32, round(image.width * image.height * 0.0005))
    if bbox is None or int(mask.sum()) < minimum_pixels:
        return None
    left, top, right, bottom = bbox
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)[top:bottom, left:right].copy()
    rgb[mask[top:bottom, left:right] == 0] = 0
    cropped = Image.fromarray(rgb, mode="RGB")
    scale = min(OUTPUT_SIZE[0] / cropped.width, OUTPUT_SIZE[1] / cropped.height)
    resized_size = (max(1, round(cropped.width * scale)), max(1, round(cropped.height * scale)))
    resized = cropped.resize(resized_size, Image.Resampling.LANCZOS)
    output = Image.new("RGB", OUTPUT_SIZE, (0, 0, 0))
    output.paste(resized, ((OUTPUT_SIZE[0] - resized.width) // 2, (OUTPUT_SIZE[1] - resized.height) // 2))
    return output


def project_standardized_mask(image: Image.Image, water_mask: np.ndarray) -> np.ndarray:
    """Project the eroded Step 1 mask into the standardized image coordinates."""
    return _project_eroded_mask(image, erode_water_mask(water_mask, image.size))


def _project_eroded_mask(image: Image.Image, mask: np.ndarray) -> np.ndarray:
    bbox = water_bbox(mask)
    if bbox is None or int(mask.sum()) < max(32, round(image.width * image.height * 0.0005)):
        raise ValueError("Water cannot be reliably detected during Step 1 preprocessing")
    left, top, right, bottom = bbox
    width, height = right - left, bottom - top
    scale = min(OUTPUT_SIZE[0] / width, OUTPUT_SIZE[1] / height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = Image.fromarray(mask[top:bottom, left:right] * 255).resize(size, Image.Resampling.NEAREST)
    output = Image.new("L", OUTPUT_SIZE)
    output.paste(resized, ((OUTPUT_SIZE[0] - size[0]) // 2, (OUTPUT_SIZE[1] - size[1]) // 2))
    return np.asarray(output) > 0


def preprocess_raw_image(
    image: Image.Image, segmentation_model: SegmentationModel
) -> Image.Image | None:
    """Run the canonical Step 1 segmentation and standardization in memory."""
    water_mask = extract_water_mask(
        run_segmentation(image, segmentation_model), segmentation_model.water_class_ids
    )
    return standardized_model_input(image, water_mask)


def preprocess_raw_image_with_projected_mask(
    image: Image.Image, segmentation_model: SegmentationModel
) -> tuple[Image.Image | None, np.ndarray | None]:
    """Experimental single-pass mask reuse; Step 2 still receives Step 1 pixels."""
    water_mask = extract_water_mask(run_segmentation(image, segmentation_model), segmentation_model.water_class_ids)
    eroded = erode_water_mask(water_mask, image.size)
    standardized = _standardize_eroded_mask(image, eroded)
    if standardized is None:
        return None, None
    return standardized, _project_eroded_mask(image, eroded)


def preprocess_coordinate_crop(image: Image.Image, segmentation_model: SegmentationModel) -> Image.Image | None:
    """Crop the lower-water source-photo region directly onto the regressor canvas."""
    import cv2

    from step_2_b_crop_selector import select_crop

    started = time.perf_counter()
    class_map = run_segmentation(image, segmentation_model, output_size=(448, 448))
    if segmentation.INFERENCE_PRECISION == "float16" and segmentation_model.device.type != "cuda":
        raise RuntimeError("Coordinate-crop preprocessing requires the configured CUDA segmentation device")
    water_mask = extract_water_mask(class_map, segmentation_model.water_class_ids)
    mask_224 = cv2.resize(
        water_mask.astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST_EXACT
    ).astype(bool)
    selected = select_crop(mask_224, np.zeros((224, 224), dtype=bool))
    if selected is None:
        return None

    left, top, right, bottom = selected[0]
    width, height = image.size
    x0 = max(0, min(width, int(np.floor(left * width / 224))))
    x1 = max(x0 + 1, min(width, int(np.ceil(right * width / 224))))
    y0 = max(0, min(height, int(np.floor(top * height / 224))))
    y1 = max(y0 + 1, min(height, int(np.ceil(bottom * height / 224))))
    source_roi = np.asarray(image.convert("RGB"), dtype=np.uint8)[y0:y1, x0:x1]
    mask_roi = mask_224[top:bottom, left:right]

    scale = min(224 / source_roi.shape[1], 224 / source_roi.shape[0])
    resized_width = max(1, min(224, round(source_roi.shape[1] * scale)))
    resized_height = max(1, min(224, round(source_roi.shape[0] * scale)))
    offset_x, offset_y = (224 - resized_width) // 2, (224 - resized_height) // 2
    canvas = np.zeros((224, 224, 3), dtype=np.uint8)
    canvas_roi = canvas[offset_y:offset_y + resized_height, offset_x:offset_x + resized_width]
    cv2.resize(source_roi, (resized_width, resized_height), dst=canvas_roi,
               interpolation=cv2.INTER_LANCZOS4)
    final_mask = cv2.resize(mask_roi.astype(np.uint8), (resized_width, resized_height),
                            interpolation=cv2.INTER_NEAREST_EXACT)
    canvas_roi[final_mask == 0] = 0
    print(
        f"Coordinate crop: mask={mask_224.shape[1]}x{mask_224.shape[0]}, "
        f"source_box=({x0},{y0},{x1},{y1}), selected={selected[0]}, "
        f"elapsed={time.perf_counter() - started:.3f}s",
        flush=True,
    )
    return Image.fromarray(canvas, mode="RGB")


def warm_coordinate_crop_selector() -> float:
    """Compile the crop selector before timing per-image preprocessing."""
    from step_2_b_crop_selector import warm_crop_selector

    elapsed = warm_crop_selector()
    print(f"Coordinate crop selector warm-up: {elapsed:.3f}s", flush=True)
    return elapsed


def encode_step_1_output(image: Image.Image, *, full_chroma: bool = False) -> bytes:
    """Encode exactly as Step 1 does before Step 2 reads the image."""
    output = io.BytesIO()
    options = {"quality": 95, "subsampling": 0} if full_chroma else {"quality": 95}
    image.save(output, format="JPEG", **options)
    return output.getvalue()


def decode_rgb_bytes(encoded: bytes) -> Image.Image:
    """Decode an encoded pipeline image into a detached RGB image."""
    with Image.open(io.BytesIO(encoded)) as opened:
        return opened.convert("RGB")


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


def processed_paths(source_path: Path, output_dir: Path, names: SourceOutputNames | None = None) -> Path:
    """Return the standardized output path; source images stay in the input step."""
    stem = names.stem_for(source_path) if names else source_path.stem
    return output_dir / f"step-1_{stem}.jpg"


def write_output_atomically(
    standardized: Image.Image, standardized_path: Path, *, full_chroma: bool = False
) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=standardized_path.parent,
            prefix=f".{standardized_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(encode_step_1_output(standardized, full_chroma=full_chroma))
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, standardized_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def process_image(
    model: SegmentationModel, source_path: Path, standardized_path: Path,
    coordinate_crop: bool = False,
) -> tuple[str, float, str | None, str | None]:
    """Process one pending image; returns (status, seconds, warning)."""
    started = time.perf_counter()
    model_revision = getattr(model.model.config, "_commit_hash", None)
    try:
        source = load_rgb_image(source_path)
        standardized = (
            preprocess_coordinate_crop(source, model)
            if coordinate_crop else preprocess_raw_image(source, model)
        )
        if standardized is None:
            return "skipped", time.perf_counter() - started, "no reliable water region", model_revision
        write_output_atomically(standardized, standardized_path, full_chroma=coordinate_crop)
        return "processed", time.perf_counter() - started, None, model_revision
    except Exception as error:
        return "skipped", time.perf_counter() - started, str(error), model_revision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("baseline-v16", "fast-v2"), default="fast-v2")
    parser.add_argument("--input-dir", type=Path, action="append", help="Additional input directory; repeatable")
    parser.add_argument("--output-dir", type=Path, help="Output directory; defaults to the selected profile")
    parser.add_argument("--device", default=None, help="torch device, e.g. cpu or cuda")
    parser.add_argument(
        "--workers", type=int, default=None,
        help="parallel worker processes, each with its own model (default: chosen from free memory)",
    )
    args = parser.parse_args()
    profile = get_profile(args.profile)
    output_dir = args.output_dir or profile.step1_dir
    configure_segmentation(profile, segmentation)
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be at least 1")
    ensure_profile_directory(profile, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dirs = configured_source_directories()
    if args.input_dir:
        source_dirs.extend(args.input_dir)
    image_paths = iter_images(source_dirs)
    if not image_paths:
        raise SystemExit(f"No supported images found in: {', '.join(str(path) for path in source_dirs)}")
    names = SourceOutputNames(
        image_paths,
        output_dir,
        (".jpg",),
        output_paths_for_stem=lambda stem: (output_dir / f"step-1_{stem}.jpg",),
    )
    total = len(image_paths)
    entries: dict[int, tuple[str, Path, Path]] = {}
    skipped: list[str] = []
    pending: list[tuple[int, Path, Path]] = []
    for index, source_path in enumerate(image_paths, 1):
        standardized_path = processed_paths(source_path, output_dir, names)
        next_step_path = Path(profile.step2_dir) / f"step-2_{standardized_path.name.removeprefix('step-1_')}"
        if next_step_path.exists():
            print(f"[{index}/{total}] {source_path.name}: skipped (Step 2 output already exists)")
            continue
        if standardized_path.exists():
            print(f"[{index}/{total}] {source_path.name}: skipped (standardized input already exists)")
            entries[index] = (source_path.name, source_path, standardized_path)
        else:
            pending.append((index, source_path, standardized_path))
    names.save()
    already_present = len(entries)

    processing_seconds = 0.0
    run_started = time.perf_counter()
    device = configure_inference_device(args.device) if not pending else None
    model_revisions: set[str] = set()
    if pending:
        device = configure_inference_device(args.device)
        if profile.precision == "float16" and device.type != "cuda":
            raise RuntimeError(f"The {profile.name} profile requires GPU segmentation with float16 precision")
        if profile.coordinate_crop:
            warm_coordinate_crop_selector()
        workers = select_worker_count(args.workers, device, len(pending))
        tasks = {index: (source_path, standardized_path, profile.coordinate_crop)
                 for index, source_path, standardized_path in pending}
        settings = {
            "SEGMENTATION_RESOLUTION": profile.step1_resolution,
            "USE_OVERLAPPING_TILES": profile.overlapping_tiles,
            "TILE_SIZE": profile.tile_size,
            "TILE_OVERLAP": profile.tile_overlap,
            "SINGLE_IMAGE_LOGITS_RESOLUTION": profile.logits_resolution,
            "INFERENCE_PRECISION": profile.precision,
        }
        results = run_segmentation_tasks(process_image, tasks, str(device), workers, settings)
        for index, (status, elapsed, warning, model_revision) in results:
            source_path, standardized_path, _ = tasks[index]
            processing_seconds += elapsed
            if model_revision:
                model_revisions.add(model_revision)
            if status == "processed":
                entries[index] = (source_path.name, source_path, standardized_path)
                print(f"[{index}/{total}] {source_path.name}: processed ({elapsed:.3f}s)")
            else:
                skipped.append(source_path.name)
                print(f"[{index}/{total}] {source_path.name}: WARNING: {warning}; skipped ({elapsed:.3f}s)")

    contract_path = output_dir / "preprocessing-contract.json"
    if profile.name != "baseline-v16" and not contract_path.exists():
        if not model_revisions:
            model = load_model(str(device), verbose=False)
            revision = getattr(model.model.config, "_commit_hash", None)
            model_revisions = {revision} if revision else set()
        if len(model_revisions) > 1:
            raise RuntimeError(f"Multiple segmentation model revisions processed in one run: {sorted(model_revisions)}")
        write_preprocessing_contract(profile, output_dir, device=str(device),
                                     model_name=segmentation.MODEL_NAME,
                                     torch_version=str(torch.__version__),
                                     segmentation_model_revision=next(iter(model_revisions), None))

    write_gallery([entries[index] for index in sorted(entries)], output_dir)
    print(f"Processed: {len(entries) - already_present}")
    print(f"Already present: {already_present}")
    print(f"Skipped: {len(skipped)}")
    print(f"Processing time: {processing_seconds:.3f}s (sum of per-image times)")
    print(f"Wall time: {time.perf_counter() - run_started:.3f}s")
    if skipped:
        print("Skipped files:")
        for filename in skipped:
            print(f"- {filename}")
    print(f"Preprocessing profile: {profile.name}")
    print(f"Gallery written to {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
