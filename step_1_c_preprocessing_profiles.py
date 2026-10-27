"""Shared, versioned preprocessing profiles for training data and inference."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class PreprocessingProfile:
    name: str
    step1_resolution: int
    step2_resolution: int
    precision: str
    overlapping_tiles: bool
    tile_size: int
    tile_overlap: float
    logits_resolution: int | None
    step1_dir: Path
    step2_dir: Path
    split_dir: Path

    def settings(self) -> dict[str, object]:
        values = asdict(self)
        for key in ("step1_dir", "step2_dir", "split_dir"):
            values[key] = str(values[key])
        return values


PROFILES = {
    "baseline-v16": PreprocessingProfile(
        name="baseline-v16",
        step1_resolution=768,
        step2_resolution=768,
        precision="float32",
        overlapping_tiles=True,
        tile_size=768,
        tile_overlap=0.25,
        logits_resolution=None,
        step1_dir=Path("step-1-processed-data"),
        step2_dir=Path("step-2-final-water-data"),
        split_dir=Path("step-3-dataset-splits"),
    ),
    "fast-v1": PreprocessingProfile(
        name="fast-v1",
        step1_resolution=448,
        step2_resolution=384,
        precision="float16",
        overlapping_tiles=False,
        tile_size=768,
        tile_overlap=0.25,
        logits_resolution=448,
        step1_dir=Path("step-1-processed-data"),
        step2_dir=Path("step-2-final-water-data"),
        split_dir=Path("step-3-dataset-splits"),
    ),
}


def get_profile(name: str) -> PreprocessingProfile:
    try:
        return PROFILES[name]
    except KeyError as error:
        raise ValueError(f"Unknown preprocessing profile {name!r}; choose from {', '.join(PROFILES)}") from error


def profile_marker(profile: PreprocessingProfile) -> dict[str, object]:
    project_dir = Path(__file__).resolve().parent
    code_names = (
        "step_1_a_water_segmentation.py",
        "step_1_b_preprocess_water_inputs.py",
        "step_1_c_preprocessing_profiles.py",
        "step_2_a_reduce_black_water_area.py",
    )
    return {
        "profile": profile.name,
        "settings": profile.settings(),
        "code_sha256": {
            name: hashlib.sha256((project_dir / name).read_bytes()).hexdigest()
            for name in code_names
        },
    }


def configure_segmentation(profile: PreprocessingProfile, segmentation_module) -> None:
    """Apply the profile's Step 1 segmentation settings to the shared segmenter."""
    segmentation_module.SEGMENTATION_RESOLUTION = profile.step1_resolution
    segmentation_module.USE_OVERLAPPING_TILES = profile.overlapping_tiles
    segmentation_module.TILE_SIZE = profile.tile_size
    segmentation_module.TILE_OVERLAP = profile.tile_overlap
    segmentation_module.SINGLE_IMAGE_LOGITS_RESOLUTION = profile.logits_resolution
    segmentation_module.INFERENCE_PRECISION = profile.precision


@contextmanager
def step2_processor_size(profile: PreprocessingProfile, segmentation_model) -> Iterator[None]:
    """Temporarily set the shared segmenter's resize for Step 2 inference."""
    previous = segmentation_model.processor.size
    segmentation_model.processor.size = {
        "height": profile.step2_resolution,
        "width": profile.step2_resolution,
    }
    try:
        yield
    finally:
        segmentation_model.processor.size = previous


def profile_contract(profile: PreprocessingProfile, *, segmentation_model_revision: str | None,
                     device: str, torch_version: str, code_sha256: dict[str, str]) -> dict[str, object]:
    """Return checkpoint metadata validated by the prediction entry points."""
    return {
        "profile": profile.name,
        "model": "nvidia/segformer-b2-finetuned-ade-512-512",
        "resolution": profile.step1_resolution,
        "step2_resolution": profile.step2_resolution,
        "precision": profile.precision,
        "logits_resolution": profile.logits_resolution,
        "overlapping_tiles": profile.overlapping_tiles,
        "device": device,
        "torch_version": torch_version,
        "segmentation_model_revision": segmentation_model_revision,
        "code_sha256": code_sha256,
    }


def ensure_profile_directory(profile: PreprocessingProfile, directory: Path) -> None:
    """Prevent mixing preprocessing profiles in one active output directory."""
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "preprocessing-profile.json"
    expected = profile_marker(profile)
    if marker.is_file():
        actual = json.loads(marker.read_text(encoding="utf-8"))
        if actual != expected:
            raise RuntimeError(f"Preprocessing profile mismatch in {marker}; choose a new output directory")
        return
    existing_outputs = [path for path in directory.iterdir() if path.is_file() and path.name != marker.name]
    if existing_outputs:
        raise RuntimeError(f"Refusing to reuse {directory} without its preprocessing profile marker")
    marker.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require_profile_directory(profile: PreprocessingProfile, directory: Path) -> None:
    """Require an existing directory's marker to match the selected profile."""
    marker = directory / "preprocessing-profile.json"
    if not marker.is_file():
        raise RuntimeError(f"Missing preprocessing profile marker for {profile.name}: {marker}")
    try:
        actual = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read preprocessing profile marker: {marker}") from error
    expected = profile_marker(profile)
    if actual != expected:
        raise RuntimeError(f"Preprocessing profile mismatch in {marker}; expected {profile.name}")


def configure_prediction_profile(profile: PreprocessingProfile, segmentation_module, prediction_module,
                                  device: str) -> str:
    """Bind Step 7 prediction to the same profile used by batch preprocessing."""
    configure_segmentation(profile, segmentation_module)
    original_step2 = prediction_module.preprocess_step_1_image

    def preprocess_profile_step2(image, segmenter, **kwargs):
        with step2_processor_size(profile, segmenter):
            return original_step2(image, segmenter, **kwargs)

    prediction_module.preprocess_step_1_image = preprocess_profile_step2
    settings = {
        "profile": profile.name,
        "settings": profile.settings(),
        "segmentation_device": device,
        "pipeline_sha256": {
            name: prediction_module.sha256_for(prediction_module.PROJECT_DIR / name)
            for name in (
                "step_1_a_water_segmentation.py",
                "step_1_b_preprocess_water_inputs.py",
                "step_1_c_preprocessing_profiles.py",
                "step_2_a_reduce_black_water_area.py",
                "step_7_a_predict.py",
            )
        },
    }
    identifier = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
    variant = f"{profile.name}-{identifier}"
    prediction_module.PREVIEW_DIR = prediction_module.PREVIEW_DIR / variant
    prediction_module.PREDICTIONS_DIR = prediction_module.PREDICTIONS_DIR / variant
    prediction_module.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path = prediction_module.PREDICTIONS_DIR / "preprocessing.json"
    if metadata_path.is_file():
        if json.loads(metadata_path.read_text(encoding="utf-8")) != settings:
            raise RuntimeError(f"Preprocessing profile namespace collision: {metadata_path}")
    else:
        metadata_path.write_text(json.dumps(settings, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return variant
