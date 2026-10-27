"""Compare the optimized crop search with a straightforward scalar oracle."""
import numpy as np

import step_2_a_reduce_black_water_area as crop
from step_1_b_preprocess_water_inputs import erode_water_mask, erosion_radius, standardized_model_input, project_standardized_mask
from PIL import Image, ImageFilter


def scalar_crop(water, black):
    bbox = crop.mask_bbox(water)
    if bbox is None:
        return None
    left, top, right, bottom = bbox
    width, height = right - left, bottom - top
    minimum = max(32, round(water.size * 0.0005))
    if water.sum() < minimum:
        return None
    lower_top = max(top, bottom - max(1, round(height * crop.LOWER_WATER_FRACTION)))
    lower = water.copy()
    lower[:lower_top] = False
    if lower.sum() < minimum:
        return None
    tables = [crop.summed_area(mask) for mask in (water, black, lower)]
    boundaries = sorted(set(range(left, right + 1, max(1, round(water.shape[1] / 56)))) | {left, right})
    minimum_area = max(256, round(width * height * 0.1))
    best = None
    for i, x1 in enumerate(boundaries[:-1]):
        for x2 in boundaries[i + 1:]:
            w = x2 - x1
            if w < 8:
                continue
            lower_count = crop.rectangle_sum(tables[2], (x1, lower_top, x2, bottom))
            if lower_count < minimum:
                continue
            for y in range(top, lower_top + 1):
                h, area = bottom - y, w * (bottom - y)
                if h < 8 or area < minimum_area:
                    continue
                box = (x1, y, x2, bottom)
                retention = crop.rectangle_sum(tables[2], box) / lower_count
                if retention + 1e-12 < crop.MINIMUM_LOWER_RETENTION:
                    continue
                water_pixels, black_pixels = (crop.rectangle_sum(t, box) for t in tables[:2])
                water_ratio = crop.projected_ratio(water_pixels, area, w, h)
                black_ratio = 1 - crop.projected_ratio(area - black_pixels, area, w, h)
                tier = 3 if retention >= .995 else 2 if retention >= .98 else 1
                key = (-tier, round(black_ratio / crop.BLACK_RATIO_BUCKET), -water_pixels,
                       -water_ratio, abs(np.log(w / h)), box)
                if best is None or key < best:
                    best = key
    return None if best is None else best[-1]


def test_vectorized_search_matches_scalar_reference():
    rng = np.random.default_rng(778)
    masks = [np.zeros((64, 64), dtype=bool), np.ones((64, 64), dtype=bool)]
    narrow = np.zeros((64, 64), dtype=bool)
    narrow[20:60, 15:18] = True
    masks.append(narrow)
    masks.extend(rng.random((64, 64)) < density for density in (.05, .25, .6, .95))
    for water in masks:
        black = rng.random(water.shape) < .2
        result = crop.select_crop(water, black)
        assert (result.box if result else None) == scalar_crop(water, black)


def test_fast_erosion_matches_pillow_including_image_edges():
    rng = np.random.default_rng(82)
    for width, height in ((20, 25), (224, 224), (2400, 30)):
        mask = (rng.random((height, width)) > .15).astype(np.uint8)
        radius = erosion_radius((width, height))
        expected = np.asarray(Image.fromarray(mask * 255).filter(ImageFilter.MinFilter(radius * 2 + 1))) > 0
        assert np.array_equal(erode_water_mask(mask, (width, height)), expected)
    mask = (rng.random((40, 50)) > .15).astype(np.uint8)
    expected = np.asarray(Image.fromarray(mask * 255).filter(ImageFilter.MinFilter(25))) > 0
    assert np.array_equal(erode_water_mask(mask, (2400, 2400)), expected)


def test_standardized_pixels_match_full_image_masking():
    rng = np.random.default_rng(90)
    rgb = rng.integers(0, 256, (100, 160, 3), dtype=np.uint8)
    mask = np.zeros((100, 160), dtype=np.uint8)
    mask[10:90, 15:140] = 1
    image = Image.fromarray(rgb)
    result = standardized_model_input(image, mask)
    eroded = erode_water_mask(mask, image.size)
    ys, xs = np.where(eroded)
    masked = rgb.copy()
    masked[eroded == 0] = 0
    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    original = Image.fromarray(masked).crop(box)
    scale = min(224 / original.width, 224 / original.height)
    original = original.resize((round(original.width * scale), round(original.height * scale)), Image.Resampling.LANCZOS)
    expected = Image.new("RGB", (224, 224))
    expected.paste(original, ((224 - original.width) // 2, (224 - original.height) // 2))
    assert np.array_equal(np.asarray(result), np.asarray(expected))


def test_projected_mask_tracks_step1_padding_and_avoids_second_segmentation(monkeypatch):
    image = Image.fromarray(np.full((80, 160, 3), 180, dtype=np.uint8))
    mask = np.zeros((80, 160), dtype=np.uint8)
    mask[20:60, 10:150] = 1
    projected = project_standardized_mask(image, mask)
    standard = standardized_model_input(image, mask)
    assert projected.shape == (224, 224)
    assert not projected[0].any() and not projected[-1].any()
    monkeypatch.setattr(crop, "run_segmentation", lambda *args: (_ for _ in ()).throw(AssertionError("second pass")))
    result = crop.preprocess_step_1_image(standard, None, projected_mask=projected)
    assert result is not None
    assert np.all(result.water_mask <= projected)
