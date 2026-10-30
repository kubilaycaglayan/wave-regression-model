#!/usr/bin/env python3
"""Train a matched EfficientNet-B0 regression head for one preprocessing version."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import lightning.pytorch as pl
from PIL import Image
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
PERF = HERE.parent
ROOT = PERF.parents[2]
sys.path.insert(0, str(ROOT))

from step_4_a_wave_dataset import build_evaluation_transform, build_training_transform
from step_5_a_wave_regression_model import WaveRegressionModel

SNAPSHOT = "20261028T081606614158Z"
SEED = 42
LEARNING_RATE = 1e-3
BATCH_SIZE = 8
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


ALL_EXPERIMENTS = (
    "v000_baseline",
    "v001_coordinate_crop_single_output",
    "v002_skip_step2_keep_baseline_image",
    "v003_roi_mask_processing",
    "v004_working_resolution_cap",
    "v005_baseline_canvas_coordinate_crop",
    "v006_working_resolution_cap_skip_step2",
)


def cache_directory(experiment_id: str, split: str) -> Path:
    base = PERF / "training-cache" / experiment_id / SNAPSHOT
    valid = []
    for candidate in base.glob(f"*/{split}") if base.is_dir() else []:
        manifest = candidate / "manifest.json"
        if manifest.is_file():
            valid.append(candidate)
    if len(valid) != 1:
        raise FileNotFoundError(
            f"Expected one prepared cache for {experiment_id}/{split}; found {len(valid)}. "
            "Run training/prepare_inputs.py first."
        )
    return valid[0]


def common_training_cohort(split: str) -> tuple[set[str], dict[str, Path]]:
    directories = {experiment: cache_directory(experiment, split) for experiment in ALL_EXPERIMENTS}
    item_sets = {
        experiment: set(json.loads((directory / "manifest.json").read_text(encoding="utf-8")).get("items", {}))
        for experiment, directory in directories.items()
    }
    common = set.intersection(*item_sets.values())
    expected_path = ROOT / "step-3-dataset-splits" / "snapshots" / SNAPSHOT / f"{split}.csv"
    with expected_path.open(newline="", encoding="utf-8") as stream:
        expected = {row["filename"] for row in csv.DictReader(stream)}
    if split == "validation" and common != expected:
        missing_by_experiment = {name: sorted(expected - items) for name, items in item_sets.items() if items != expected}
        raise RuntimeError(f"Every experiment must prepare the full validation cohort: {missing_by_experiment}")
    excluded = expected - common
    if not common:
        raise RuntimeError(f"No common preprocessed {split} cohort across experiment versions")
    if any(items - expected for items in item_sets.values()):
        raise RuntimeError(f"A cache contains photos outside the fixed {split} split")
    return common, directories


class CachedImages(Dataset):
    def __init__(self, directory: Path, *, training: bool, allowed_names: set[str] | None = None):
        self.directory = directory
        self.manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        self.items = sorted(
            ((name, item) for name, item in self.manifest["items"].items()
             if allowed_names is None or name in allowed_names),
            key=lambda pair: pair[0].casefold(),
        )
        self.training = training
        self.transform = build_training_transform() if training else build_evaluation_transform()
        if not self.items:
            raise ValueError(f"Prepared cache is empty: {directory}")
        for label_filename, item in self.items:
            path = self.directory / item["cache_file"]
            if not path.is_file() or sha256_file(path) != item["cache_sha256"]:
                raise RuntimeError(f"Prepared input cache is missing or changed: {path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        label_filename, item = self.items[index]
        path = self.directory / item["cache_file"]
        image_array = np.load(path, allow_pickle=False)
        if image_array.dtype != np.uint8 or image_array.shape != (224, 224, 3):
            raise ValueError(f"Invalid prepared RGB image {path}: {image_array.shape}/{image_array.dtype}")
        image = Image.fromarray(image_array, mode="RGB")
        return self.transform(image).float(), torch.tensor(float(item["label"]), dtype=torch.float32)


def load_split_rows(split: str) -> list[dict]:
    path = ROOT / "step-3-dataset-splits" / "snapshots" / SNAPSHOT / f"{split}.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--run-id", help="output run ID; defaults to a UTC timestamp")
    args = parser.parse_args()
    experiment_id = args.experiment_id
    version_manifest_path = PERF / "experiment-versions" / experiment_id / "experiment.json"
    if not version_manifest_path.is_file():
        raise FileNotFoundError(version_manifest_path)

    pl.seed_everything(SEED, workers=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    common_train_names, train_directories = common_training_cohort("train")
    validation_names, validation_directories = common_training_cohort("validation")
    train_dir = train_directories[experiment_id]
    validation_dir = validation_directories[experiment_id]
    train_data = CachedImages(train_dir, training=True, allowed_names=common_train_names)
    validation_data = CachedImages(validation_dir, training=False, allowed_names=validation_names)
    expected_train = load_split_rows("train")
    expected_validation = load_split_rows("validation")
    if len(train_data) != len(common_train_names) or len(validation_data) != len(expected_validation):
        raise RuntimeError(
            f"Prepared cache cohort mismatch: train={len(train_data)}/{len(expected_train)}, "
            f"validation={len(validation_data)}/{len(expected_validation)}"
        )
    train_names = {name for name, _ in train_data.items}
    validation_names = {name for name, _ in validation_data.items}
    if train_names != common_train_names:
        raise RuntimeError("Prepared training cache does not match the common training cohort")
    if validation_names != {row["filename"] for row in expected_validation}:
        raise RuntimeError("Prepared validation cache does not match the fixed validation split")

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = PERF / "training-runs" / experiment_id / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir()
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=False)
    validation_loader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=False,
                                   num_workers=0, pin_memory=True, drop_last=False)
    model = WaveRegressionModel(learning_rate=LEARNING_RATE,
                                backbone_name="efficientnet_b0", pretrained=True)
    model.fast_preprocessing_experiment = {
        "experiment_id": experiment_id,
        "preprocessing_manifest_sha256": sha256_file(version_manifest_path),
        "split_snapshot": SNAPSHOT,
        "train_cache_signature": train_data.manifest["signature"],
        "validation_cache_signature": validation_data.manifest["signature"],
    }
    model.train()
    model.frozen_backbone_mode_checks()
    checkpoint = ModelCheckpoint(dirpath=checkpoint_dir,
                                 filename="best-epoch={epoch:02d}-val_mae={val_mae:.6f}",
                                 monitor="val_mae", mode="min", save_top_k=1)
    early_stopping = EarlyStopping(monitor="val_mae", mode="min",
                                   patience=EARLY_STOPPING_PATIENCE, verbose=True)
    trainer = pl.Trainer(accelerator="auto", devices=1, max_epochs=MAX_EPOCHS,
                         callbacks=[checkpoint, early_stopping], deterministic=True,
                         log_every_n_steps=1, logger=False, enable_progress_bar=False)
    started = time.perf_counter()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=validation_loader)
    best_model = WaveRegressionModel.load_from_checkpoint(checkpoint.best_model_path,
                                                          pretrained=False)
    best_validation = trainer.validate(best_model, dataloaders=validation_loader, verbose=False)[0]
    summary = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "architecture": "EfficientNet-B0 with frozen ImageNet backbone and trained sigmoid head",
        "seed": SEED,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "best_checkpoint": str(Path(checkpoint.best_model_path).relative_to(run_dir)),
        "best_epoch_zero_based": int(re.search(r"epoch=(\d+)", Path(checkpoint.best_model_path).name).group(1)),
        "epochs_completed": int(trainer.fit_loop.epoch_progress.current.completed),
        "training_count": len(train_data),
        "validation_count": len(validation_data),
        "common_training_cohort_count": len(common_train_names),
        "excluded_from_training_split": sorted({row["filename"] for row in expected_train} - common_train_names),
        "test_used": False,
        "split_snapshot": SNAPSHOT,
        "train_cache_signature": train_data.manifest["signature"],
        "validation_cache_signature": validation_data.manifest["signature"],
        "validation_metrics": {key: float(value) for key, value in best_validation.items()
                                if isinstance(value, (int, float))},
        "elapsed_training_seconds": time.perf_counter() - started,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = run_dir / "metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Metrics: {summary_path}")
    print(f"Best checkpoint: {checkpoint.best_model_path}")
    print(f"Validation MAE: {best_validation['val_mae']:.6f}")
    print(f"Validation RMSE: {best_validation['val_rmse']:.6f}")
    print(f"Training elapsed: {summary['elapsed_training_seconds']:.1f}s")


if __name__ == "__main__":
    main()
