"""Generate an isolated visual comparison of augmentations for three sea crops."""

from __future__ import annotations

import csv
import html
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "step-2-final-water-data"
LABELS_PATH = ROOT / "labels.csv"
OUTPUT_DIR = Path(__file__).resolve().parent
IMAGE_IDS = ("IMG_7187", "IMG_7188", "IMG_7189")
ADDITIONAL_RANDOM_IMAGES = 10
SEED = 7187


def read_samples() -> list[tuple[str, float, Image.Image]]:
    with LABELS_PATH.open(newline="", encoding="utf-8-sig") as handle:
        labels = {row["filename"]: float(row["waviness"]) for row in csv.DictReader(handle)}

    selected_filenames = [f"step-2_{image_id}.jpg" for image_id in IMAGE_IDS]
    missing = [filename for filename in selected_filenames if filename not in labels]
    if missing:
        raise ValueError(f"No labels found for requested images in {LABELS_PATH}: {missing}")

    candidates = sorted(
        filename for filename in labels
        if filename not in selected_filenames and (INPUT_DIR / filename).is_file()
    )
    if len(candidates) < ADDITIONAL_RANDOM_IMAGES:
        raise ValueError(
            f"Need {ADDITIONAL_RANDOM_IMAGES} additional labeled processed images, found {len(candidates)}"
        )
    selected_filenames.extend(
        random.Random(SEED + 1).sample(candidates, ADDITIONAL_RANDOM_IMAGES)
    )

    samples = []
    for filename in selected_filenames:
        path = INPUT_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f"Processed crop not found: {path}")
        with Image.open(path) as opened:
            samples.append((filename, labels[filename], opened.convert("RGB")))
    sizes = {image.size for _, _, image in samples}
    if len(sizes) != 1:
        raise ValueError(f"Input crops must have matching dimensions; found {sorted(sizes)}")
    return samples


def make_geometry(image: Image.Image, rng: random.Random) -> Image.Image:
    # Conservative changes preserve the lower-water crop while varying framing.
    angle = rng.uniform(-7.5, 7.5)  # 50% wider than ±5°
    translate = (rng.uniform(-0.0375, 0.0375) * image.width,
                 rng.uniform(-0.0375, 0.0375) * image.height)  # 50% wider than ±2.5%
    scale = rng.uniform(0.94, 1.06)  # 50% wider deviation than 0.96–1.04
    radians = np.deg2rad(angle)
    inverse_scale = 1.0 / scale
    a, b = inverse_scale * np.cos(radians), inverse_scale * np.sin(radians)
    d, e = -inverse_scale * np.sin(radians), inverse_scale * np.cos(radians)
    cx, cy = image.width / 2.0, image.height / 2.0
    tx, ty = translate
    c = cx - a * (cx - tx) - b * (cy - ty)
    f = cy - d * (cx - tx) - e * (cy - ty)
    return image.transform(image.size, Image.Transform.AFFINE, (a, b, c, d, e, f),
                           resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))


def make_current(image: Image.Image, rng: random.Random) -> Image.Image:
    # Match the pipeline's 50% horizontal flip and ColorJitter magnitudes.
    result = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if rng.random() < 0.5 else image.copy()
    operations = [
        lambda img: ImageEnhance.Brightness(img).enhance(rng.uniform(0.75, 1.25)),
        lambda img: ImageEnhance.Contrast(img).enhance(rng.uniform(0.75, 1.25)),
        lambda img: ImageEnhance.Color(img).enhance(rng.uniform(0.80, 1.20)),
    ]
    rng.shuffle(operations)
    for operation in operations:
        result = operation(result)
    hsv = np.asarray(result.convert("HSV"), dtype=np.uint8).copy()
    hue_shift = round(rng.uniform(-0.02, 0.02) * 255)
    hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int16) + hue_shift) % 256
    return Image.fromarray(hsv, mode="HSV").convert("RGB")


def make_noise(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    pixels = np.asarray(image, dtype=np.float32)
    noise = rng.normal(0.0, 7.5, pixels.shape[:2] + (1,))
    return Image.fromarray(np.uint8(np.clip(pixels + noise, 0, 255)), mode="RGB")


def make_small_crop(image: Image.Image, rng: random.Random) -> Image.Image:
    width, height = image.size
    crop_w, crop_h = round(width * 0.85), round(height * 0.85)
    left = rng.randint(0, width - crop_w)
    top = rng.randint(0, height - crop_h)
    return image.crop((left, top, left + crop_w, top + crop_h)).resize(image.size, Image.Resampling.BILINEAR)


def make_resized_crop(image: Image.Image, rng: random.Random) -> Image.Image:
    width, height = image.size
    area = width * height * rng.uniform(0.775, 1.0)
    aspect = rng.uniform(0.95, 1.05)
    crop_w = min(width, round((area * aspect) ** 0.5))
    crop_h = min(height, round((area / aspect) ** 0.5))
    left, top = rng.randint(0, width - crop_w), rng.randint(0, height - crop_h)
    return image.crop((left, top, left + crop_w, top + crop_h)).resize(
        image.size, Image.Resampling.BILINEAR
    )


def make_mixup(image_a: Image.Image, image_b: Image.Image, weight: float) -> Image.Image:
    return Image.blend(image_a, image_b, 1.0 - weight)


def make_cutmix(image_a: Image.Image, image_b: Image.Image, rng: random.Random) -> tuple[Image.Image, float]:
    width, height = image_a.size
    # Patch width and height are 50% larger than before; area grows quadratically.
    ratio = 0.675
    patch_w, patch_h = round(width * ratio), round(height * ratio)
    center_x, center_y = rng.randrange(width), rng.randrange(height)
    x1, x2 = max(0, center_x - patch_w // 2), min(width, center_x + patch_w // 2)
    y1, y2 = max(0, center_y - patch_h // 2), min(height, center_y + patch_h // 2)
    result = image_a.copy()
    result.paste(image_b.crop((x1, y1, x2, y2)), (x1, y1))
    return result, ((x2 - x1) * (y2 - y1)) / (width * height)


def save(image: Image.Image, path: Path) -> None:
    image.save(path, format="JPEG", quality=94, optimize=True)


def main() -> None:
    started = time.perf_counter()
    samples = read_samples()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    py_rng = random.Random(SEED)
    np_rng = np.random.default_rng(SEED)
    operations: dict[str, list[tuple[str, str]]] = {name: [] for name, _, _ in samples}

    for index, (filename, label, image) in enumerate(samples):
        started_item = time.perf_counter()
        stem = Path(filename).stem
        partner_filename, partner_label, partner = samples[(index + 1) % len(samples)]
        variants: list[tuple[str, Image.Image]] = [
            ("Processed input", image),
            ("Current flip + color jitter", make_current(image, py_rng)),
            ("Small rotation / translation / scale", make_geometry(image, py_rng)),
            ("Mild blur (Gaussian radius 1.05 px; +50%)", image.filter(ImageFilter.GaussianBlur(radius=1.05))),
            ("Sensor noise (σ=7.5/255; +50%)", make_noise(image, np_rng)),
            ("Small random crop (85%; +50% crop-off)", make_small_crop(image, py_rng)),
            ("Resized crop (77.5–100%; +50% range)", make_resized_crop(image, py_rng)),
        ]

        mix_weight = py_rng.uniform(0.50, 0.80)
        mix_image = make_mixup(image, partner, mix_weight)
        mix_label = mix_weight * label + (1.0 - mix_weight) * partner_label
        variants.append((
            f"MixUp with {Path(partner_filename).stem} (λ={mix_weight:.2f}; label={mix_label:.3f})",
            mix_image,
        ))

        cut_image, replaced_fraction = make_cutmix(image, partner, py_rng)
        cut_label = (1.0 - replaced_fraction) * label + replaced_fraction * partner_label
        variants.append((
            f"CutMix with {Path(partner_filename).stem} (anchor label={label:.2f}; inserted label={partner_label:.2f}; partner area={replaced_fraction:.1%}; final label={cut_label:.3f})",
            cut_image,
        ))

        for variant_index, (caption, variant_image) in enumerate(variants):
            asset = f"{stem}_variant_{variant_index:02d}.jpg"
            save(variant_image, OUTPUT_DIR / asset)
            operations[filename].append((caption, asset))
        print(f"generated {filename}: {len(variants)} examples in {time.perf_counter() - started_item:.3f}s")

    rows = []
    for filename, label, _ in samples:
        figures = "".join(
            f'<figure><a href="{html.escape(asset)}" target="_blank"><img src="{html.escape(asset)}" loading="lazy" alt="{html.escape(caption)}"></a><figcaption>{html.escape(caption)}</figcaption></figure>'
            for caption, asset in operations[filename]
        )
        rows.append(
            f'<article><h2>{html.escape(filename)} · label {label:.2f}</h2>'
            f'<div class="row">{figures}</div></article>'
        )

    document = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Thirteen-photo augmentation experiment</title>
<style>
*{box-sizing:border-box}body{font:15px system-ui,sans-serif;margin:0;padding:1rem;background:#f3f4f6;color:#17202a}
main{max-width:1800px;margin:auto}article{background:#fff;border-radius:10px;padding:1rem;margin:0 0 1rem;box-shadow:0 1px 4px #0002}
h1{margin:.2rem 0 .5rem}h2{font-size:1rem;margin:.2rem 0 .8rem}.row{display:grid;grid-template-columns:repeat(9,minmax(175px,1fr));gap:.8rem;overflow-x:auto;padding-bottom:.4rem}
figure{margin:0;min-width:175px}img{display:block;width:100%;height:auto;border-radius:6px;cursor:zoom-in}figcaption{text-align:center;margin-top:.4rem;color:#43515d;font-size:.82rem}
</style></head><body><main><h1>Augmentation comparison · three processed sea crops</h1>
<p>This gallery includes the three requested photos plus ten reproducibly selected labeled processed photos (selection seed 7188). Each row shows the processed input and eight augmentation examples. Experimental augmentation ranges are 50% wider or stronger than the first gallery; the current training augmentation remains unchanged as a reference. MixUp and CutMix pair each image with the next image in the list. Horizontal scroll is available on narrower screens.</p>
""" + "".join(rows) + "</main></body></html>"
    gallery = OUTPUT_DIR / "index.html"
    gallery.write_text(document, encoding="utf-8")
    print(f"gallery: {gallery.resolve()}")
    print(f"completed in {time.perf_counter() - started:.3f}s")


if __name__ == "__main__":
    main()
