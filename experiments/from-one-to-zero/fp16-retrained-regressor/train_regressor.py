#!/usr/bin/env python3
"""Train an isolated EfficientNet-B0 regressor on FP16-path cached JPEGs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import lightning.pytorch as pl
from PIL import Image
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SNAPSHOT = "20261102T084438874261Z"
SPLITS = ROOT / "step-3-dataset-splits" / "snapshots" / SNAPSHOT
sys.path.insert(0, str(ROOT))
from step_4_a_wave_dataset import build_evaluation_transform, build_training_transform
from step_5_a_wave_regression_model import WaveRegressionModel


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CachedInputs(Dataset):
    def __init__(self, manifest_path: Path, *, training: bool):
        self.manifest_path = manifest_path
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest.get("split_snapshot") != SNAPSHOT:
            raise RuntimeError(f"Wrong split snapshot in {manifest_path}")
        self.items = sorted(self.manifest["items"].items(), key=lambda pair: pair[0].casefold())
        self.transform = build_training_transform() if training else build_evaluation_transform()
        if not self.items:
            raise RuntimeError(f"No prepared inputs in {manifest_path}")
        for label_name, item in self.items:
            image_path = HERE / item["cache_file"]
            if not image_path.is_file() or sha(image_path) != item["cache_sha256"]:
                raise RuntimeError(f"Prepared image cache is missing or modified: {image_path}")
            if not 0.0 <= float(item["label"]) <= 1.0:
                raise ValueError(f"Invalid label in {manifest_path}: {label_name}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        _name, item = self.items[index]
        image_path = HERE / item["cache_file"]
        with Image.open(image_path) as im:
            image = im.convert("RGB")
        if image.size != (224, 224):
            raise ValueError(f"Expected a 224x224 model input, got {image.size}: {image_path}")
        return self.transform(image).float(), torch.tensor(float(item["label"]), dtype=torch.float32)


class EpochHistory(Callback):
    def __init__(self):
        self.values: list[dict] = []

    def on_validation_end(self, trainer, _pl_module):
        if trainer.sanity_checking:
            return
        values = {key: float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
                  for key, value in trainer.callback_metrics.items()
                  if key in {"train_loss", "train_mae", "val_loss", "val_mae", "val_rmse"}}
        if "val_mae" in values:
            row = {"epoch": int(trainer.current_epoch), **values}
            if not self.values or self.values[-1]["epoch"] != row["epoch"]:
                self.values.append(row)
                print("Epoch " + str(row["epoch"]) + ": " + ", ".join(
                    f"{key}={value:.6f}" for key, value in values.items()), flush=True)


@torch.inference_mode()
def evaluate(model, dataset: CachedInputs, batch_size: int) -> dict:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    absolute_errors, squared_errors, smooth_l1 = [], [], []
    loss_fn = torch.nn.SmoothL1Loss(reduction="none")
    device = next(model.parameters()).device
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        predictions = model(images).reshape(-1)
        errors = predictions - targets
        absolute_errors.extend(errors.abs().cpu().tolist())
        squared_errors.extend(errors.square().cpu().tolist())
        smooth_l1.extend(loss_fn(predictions, targets).cpu().tolist())
    return {"count": len(absolute_errors), "mae": statistics.mean(absolute_errors),
            "rmse": statistics.mean(squared_errors) ** 0.5,
            "smooth_l1": statistics.mean(smooth_l1)}


def csv_labels(split: str) -> dict[str, float]:
    with (SPLITS / f"{split}.csv").open(newline="", encoding="utf-8") as stream:
        return {row["filename"]: float(row["waviness"]) for row in csv.DictReader(stream)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    args = parser.parse_args()
    config = json.loads((HERE / "experiment.json").read_text())["training"]
    pl.seed_everything(config["seed"], workers=True)
    np.random.seed(config["seed"])
    train_manifest = next((HERE / "artifacts/preprocessed-inputs").glob("*/manifests/train.json"))
    validation_manifest = train_manifest.parent / "validation.json"
    train_data = CachedInputs(train_manifest, training=True)
    train_eval = CachedInputs(train_manifest, training=False)
    validation_data = CachedInputs(validation_manifest, training=False)
    expected_train, expected_val = csv_labels("train"), csv_labels("validation")
    if set(expected_train) != {name for name, _ in train_data.items}:
        raise RuntimeError("Training cache does not match the complete fixed training split")
    if set(expected_val) != {name for name, _ in validation_data.items}:
        raise RuntimeError("Validation cache does not match the complete fixed validation split")

    run_dir = HERE / "artifacts/training-runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True,
                              num_workers=0, pin_memory=torch.cuda.is_available())
    validation_loader = DataLoader(validation_data, batch_size=config["batch_size"], shuffle=False,
                                   num_workers=0, pin_memory=torch.cuda.is_available())
    model = WaveRegressionModel(learning_rate=config["learning_rate"],
                                backbone_name="efficientnet_b0", pretrained=True)
    model.fast_preprocessing_experiment = {
        "experiment_id": "fp16-retrained-regressor",
        "experiment_manifest_sha256": sha(HERE / "experiment.json"),
        "preprocessing_signature": train_data.manifest["pipeline_signature"],
        "split_snapshot": SNAPSHOT,
        "training_cache_manifest_sha256": sha(train_manifest),
        "validation_cache_manifest_sha256": sha(validation_manifest),
    }
    # Apply the model's frozen-backbone mode policy before checking it. Newly
    # constructed modules default to training mode until train() is called.
    model.train()
    model.frozen_backbone_mode_checks()
    checkpoint = ModelCheckpoint(dirpath=run_dir / "checkpoints",
        filename="best-epoch={epoch:02d}-val_mae={val_mae:.6f}", monitor="val_mae", mode="min", save_top_k=1)
    early_stop = EarlyStopping(monitor="val_mae", mode="min",
                               patience=config["early_stopping"]["patience"], verbose=True)
    history = EpochHistory()
    trainer = pl.Trainer(accelerator="auto", devices=1, max_epochs=config["maximum_epochs"],
        callbacks=[checkpoint, early_stop, history], deterministic=True, log_every_n_steps=1,
        logger=False, enable_progress_bar=False, enable_model_summary=False)
    started = time.perf_counter()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=validation_loader)
    training_seconds = time.perf_counter() - started
    best_path = Path(checkpoint.best_model_path)
    best_model = WaveRegressionModel.load_from_checkpoint(best_path, pretrained=False, map_location="cpu")
    best_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model.to(device)
    selected_metrics = {"train_deterministic": evaluate(best_model, train_eval, config["batch_size"]),
                        "validation": evaluate(best_model, validation_data, config["batch_size"])}
    epoch_match = re.search(r"epoch=(\d+)", best_path.name)
    if not epoch_match:
        raise RuntimeError(f"Could not parse selected epoch from checkpoint: {best_path.name}")
    result = {"schema_version": 1, "experiment_id": "fp16-retrained-regressor", "run_id": args.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": "EfficientNet-B0 ImageNet pretrained frozen backbone + trained sigmoid regression head",
        "device": str(device), "seed": config["seed"], "optimizer": config["optimizer"],
        "learning_rate": config["learning_rate"], "batch_size": config["batch_size"],
        "maximum_epochs": config["maximum_epochs"], "early_stopping": config["early_stopping"],
        "loss": config["loss"], "augmentations": config["augmentations"],
        "split_snapshot": SNAPSHOT, "train_count": len(train_data), "validation_count": len(validation_data),
        "test_used_for_training_or_selection": False,
        "best_epoch_zero_based": int(epoch_match.group(1)),
        "epochs_completed": int(trainer.fit_loop.epoch_progress.current.completed),
        "best_validation_mae": float(checkpoint.best_model_score.detach().cpu()),
        "best_checkpoint": str(best_path.relative_to(HERE)),
        "selected_checkpoint_metrics": selected_metrics, "epoch_history": history.values,
        "training_seconds": training_seconds,
        "preprocessing_signature": train_data.manifest["pipeline_signature"],
        "train_cache_manifest": str(train_manifest.relative_to(HERE)),
        "validation_cache_manifest": str(validation_manifest.relative_to(HERE))}
    result_path = run_dir / "training-results.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Training results: {result_path}", flush=True)
    print(f"Selected epoch {result['best_epoch_zero_based']}: "
          f"train MAE={selected_metrics['train_deterministic']['mae']:.6f}, "
          f"validation MAE={selected_metrics['validation']['mae']:.6f}, "
          f"validation RMSE={selected_metrics['validation']['rmse']:.6f}; "
          f"elapsed={training_seconds:.1f}s", flush=True)


if __name__ == "__main__":
    main()
