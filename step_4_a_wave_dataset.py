"""Dataset and reusable transforms for the sea-waviness regression pipeline."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import torch
from PIL import Image


# Some environments have a torchvision wheel whose optional C++ operators do
# not match the installed torch wheel. Defining the operator before importing
# torchvision keeps the ordinary transforms usable in that situation.
_TORCHVISION_COMPAT_LIBRARY = None


def _register_torchvision_compat_ops() -> None:
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


_register_torchvision_compat_ops()

from torchvision import transforms  # noqa: E402


IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_training_transform() -> transforms.Compose:
    """Return intentionally mild augmentations followed by ImageNet scaling."""
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            # Strong enough to be inspectable, but still well within the range
            # that should not change the perceived amount of wave structure.
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.20, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_evaluation_transform() -> transforms.Compose:
    """Return the deterministic transform shared by validation, test, and inference."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def denormalize_image(image_tensor: torch.Tensor) -> torch.Tensor:
    """Reverse ImageNet normalization and clamp an image tensor for previewing."""
    mean = torch.tensor(IMAGENET_MEAN, dtype=image_tensor.dtype, device=image_tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=image_tensor.dtype, device=image_tensor.device).view(3, 1, 1)
    return (image_tensor * std + mean).clamp(0.0, 1.0)


class WaveDataset(torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Load processed RGB images and waviness labels from one split manifest."""

    def __init__(self, split_csv: str | Path, image_dir: str | Path, transform: Callable | None = None) -> None:
        self.split_csv = Path(split_csv)
        self.image_dir = Path(image_dir)
        self.transform = transform or build_evaluation_transform()
        self.samples = self._read_manifest()

    def _read_manifest(self) -> list[tuple[str, float]]:
        if not self.split_csv.is_file():
            raise FileNotFoundError(f"Split manifest not found: {self.split_csv}")
        samples: list[tuple[str, float]] = []
        with self.split_csv.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["filename", "waviness"]:
                raise ValueError(f"{self.split_csv} must have header: filename,waviness")
            for row_number, row in enumerate(reader, start=2):
                filename = (row.get("filename") or "").strip()
                try:
                    waviness = float(row["waviness"])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"{self.split_csv}:{row_number}: invalid waviness") from error
                if not filename:
                    raise ValueError(f"{self.split_csv}:{row_number}: empty filename")
                if not 0.0 <= waviness <= 1.0:
                    raise ValueError(f"{self.split_csv}:{row_number}: waviness outside [0, 1]")
                image_path = self.image_dir / filename
                if not image_path.is_file():
                    raise FileNotFoundError(f"Missing image for {self.split_csv}:{row_number}: {image_path}")
                samples.append((filename, waviness))
        if not samples:
            raise ValueError(f"No samples found in {self.split_csv}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def image_path(self, index: int) -> Path:
        return self.image_dir / self.samples[index][0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        filename, waviness = self.samples[index]
        with Image.open(self.image_dir / filename) as opened:
            image = opened.convert("RGB")
        if image.size != IMAGE_SIZE:
            raise ValueError(f"Expected {IMAGE_SIZE} image, got {image.size}: {self.image_path(index)}")
        image_tensor = self.transform(image)
        if not isinstance(image_tensor, torch.Tensor) or image_tensor.shape != (3, 224, 224):
            raise ValueError(f"Transform must return [3, 224, 224], got {getattr(image_tensor, 'shape', None)}")
        return image_tensor.float(), torch.tensor(waviness, dtype=torch.float32)
