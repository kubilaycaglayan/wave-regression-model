"""Shared Pillow image decoding for preprocessing, datasets, and inference."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener


SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"
}

# Register once at import time so Pillow opens HEIC/HEIF without permanent conversion.
register_heif_opener()


def load_rgb_image(path: str | Path) -> Image.Image:
    """Decode an image, apply its EXIF orientation, and return detached RGB."""
    with Image.open(path) as opened:
        oriented = ImageOps.exif_transpose(opened)
        return oriented.convert("RGB")
