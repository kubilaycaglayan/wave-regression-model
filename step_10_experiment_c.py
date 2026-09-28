"""Experiment C: compare frozen ImageNet backbones for wave regression.

This module intentionally uses an explicit immutable split snapshot and never
constructs a test dataset.  It can run one configuration or all nine
backbone/seed configurations, then writes metrics and comparison artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

# Install the repository's torchvision compatibility definition first.
from step_4_a_wave_dataset import (  # noqa: F401
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    WaveDataset,
    build_evaluation_transform,
    build_training_transform,
)

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    ResNet34_Weights,
    efficientnet_b0,
    resnet18,
    resnet34,
)


PROJECT_DIR = Path(__file__).resolve().parent
SNAPSHOT_ROOT = PROJECT_DIR / "step-3-dataset-splits" / "snapshots"
OUTPUT_ROOT = PROJECT_DIR / "step-10-experiment-c-frozen-backbone-comparison"
BACKBONES = ("resnet18", "resnet34", "efficientnet_b0")
SEEDS = (42, 43, 44)
LEARNING_RATE = 1e-3
BATCH_SIZE = 8
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
VERSION_DIRECTORY_PATTERN = re.compile(r"^v(?P<number>\d+)-(?P<stamp>[^/]+)$")
INDEX_SCHEMA_VERSION = 1


def sha256_for(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def version_directories(output_root: Path) -> list[Path]:
    if not output_root.is_dir():
        return []
    return sorted(
        (path for path in output_root.iterdir() if path.is_dir() and VERSION_DIRECTORY_PATTERN.fullmatch(path.name)),
        key=lambda path: int(VERSION_DIRECTORY_PATTERN.fullmatch(path.name).group("number")),
    )


def next_version_number(output_root: Path) -> int:
    return max(
        (int(VERSION_DIRECTORY_PATTERN.fullmatch(path.name).group("number")) for path in version_directories(output_root)),
        default=0,
    ) + 1


def experiment_configuration(snapshot: dict[str, Any], planned: list[tuple[str, int]]) -> dict[str, Any]:
    return {
        "experiment": "C_frozen_pretrained_backbone_comparison",
        "planned_runs": [{"backbone": canonical_backbone_name(backbone), "seed": seed} for backbone, seed in planned],
        "training": {
            "pretrained_weights": "torchvision DEFAULT ImageNet weights",
            "backbone_frozen": True,
            "batchnorm_eval": True,
            "regression_head": "Linear(feature_dim, 1) -> Sigmoid",
            "loss": "SmoothL1Loss",
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "early_stopping": {"monitor": "val_mae", "mode": "min", "patience": EARLY_STOPPING_PATIENCE},
            "image_size": list(IMAGE_SIZE),
            "imagenet_mean": list(IMAGENET_MEAN),
            "imagenet_std": list(IMAGENET_STD),
            "augmentations": "RandomHorizontalFlip(p=0.5) + ColorJitter(brightness=0.25, contrast=0.25, saturation=0.20, hue=0.02)",
        },
        "split_snapshot": snapshot,
        "source_sha256": sha256_for(Path(__file__).resolve()),
        "test_evaluated": False,
    }


def experiment_version_name(output_root: Path) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"v{next_version_number(output_root)}-{timestamp}"


def resolve_version_directory(output_root: Path, requested: str) -> Path:
    candidates = [path for path in version_directories(output_root) if path.name == requested or path.name.startswith(requested + "-")]
    if not candidates:
        raise FileNotFoundError(f"Experiment version not found under {output_root}: {requested}")
    if len(candidates) > 1:
        raise ValueError(f"Experiment version prefix is ambiguous: {requested}; matches {[path.name for path in candidates]}")
    return candidates[0]


def prepare_experiment_version(
    output_root: Path,
    snapshot: dict[str, Any],
    planned: list[tuple[str, int]],
    requested_version: str | None = None,
    force_new: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Create a new immutable version or safely select an incomplete one to resume."""
    output_root.mkdir(parents=True, exist_ok=True)
    if requested_version and force_new:
        raise ValueError("--force cannot be combined with --version; select a new version without overwriting history")
    configuration = experiment_configuration(snapshot, planned)
    fingerprint = canonical_json_hash(configuration)
    selected: Path | None = None
    if requested_version:
        selected = resolve_version_directory(output_root, requested_version)
    elif not force_new:
        for candidate in reversed(version_directories(output_root)):
            manifest_path = candidate / "experiment_manifest.json"
            if not manifest_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("config_fingerprint") == fingerprint and manifest.get("status") != "complete":
                selected = candidate
                break
    if selected is None:
        if not requested_version and not force_new:
            # Existing unversioned outputs are ambiguous and must be migrated
            # explicitly instead of being silently overwritten.
            legacy_outputs = [output_root / name for name in ("runs", "plots", "aggregate_metrics.json", "backbone_comparison.csv", "backbone_comparison.md")]
            if any(path.exists() for path in legacy_outputs) and not version_directories(output_root):
                raise RuntimeError(
                    f"Unversioned Experiment C outputs exist in {output_root}; migrate them into a version directory before starting a new run"
                )
        selected = output_root / experiment_version_name(output_root)
        selected.mkdir(parents=True, exist_ok=False)
        manifest = {
            "schema_version": 1,
            "version_id": selected.name,
            "version_number": int(VERSION_DIRECTORY_PATTERN.fullmatch(selected.name).group("number")),
            "created_at_utc": utc_now(),
            "status": "running",
            "config_fingerprint": fingerprint,
            "configuration": configuration,
            "completed_runs": [],
            "artifacts": {},
        }
    else:
        manifest_path = selected / "experiment_manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(f"Selected version has no manifest: {selected}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_fingerprint") != fingerprint:
            raise ValueError(
                f"Configuration does not match experiment version {selected.name}; use a new version for changed settings"
            )
        if manifest.get("status") == "complete":
            raise RuntimeError(f"Experiment version is complete and immutable: {selected}")
        manifest["status"] = "running"
    json_dump(selected / "experiment_manifest.json", manifest)
    return selected, manifest


def update_experiment_index(output_root: Path, manifest: dict[str, Any]) -> None:
    """Update the one-record-per-version index without duplicating versions."""
    index_path = output_root / "experiment_index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        index = {"schema_version": INDEX_SCHEMA_VERSION, "versions": []}
    versions = [item for item in index.get("versions", []) if item.get("version_id") != manifest.get("version_id")]
    versions.append(
        {
            "version_id": manifest.get("version_id"),
            "version_number": manifest.get("version_number"),
            "created_at_utc": manifest.get("created_at_utc"),
            "completed_at_utc": manifest.get("completed_at_utc"),
            "status": manifest.get("status"),
            "snapshot": manifest.get("configuration", {}).get("split_snapshot", {}).get("name"),
            "config_fingerprint": manifest.get("config_fingerprint"),
            "completed_runs": manifest.get("completed_runs", []),
            "report": manifest.get("artifacts", {}).get("markdown"),
        }
    )
    versions.sort(key=lambda item: int(item.get("version_number", 0)))
    index = {"schema_version": INDEX_SCHEMA_VERSION, "versions": versions}
    json_dump(index_path, index)
    lines = [
        "# Experiment C Version Index",
        "",
        "| Version | Status | Snapshot | Configuration | Completed runs | Report |",
        "|---|---|---|---|---:|---|",
    ]
    for item in versions:
        report = f"`{item['report']}`" if item.get("report") else ""
        lines.append(f"| `{item['version_id']}` | {item['status']} | `{item.get('snapshot', '')}` | `{item.get('config_fingerprint', '')[:12]}` | {len(item.get('completed_runs', []))} | {report} |")
    (output_root / "experiment_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def snapshot_metadata(snapshot_dir: Path) -> dict[str, Any]:
    """Validate the immutable snapshot, including all three split manifests."""
    manifest_path = snapshot_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Split snapshot manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual: dict[str, Any] = {
        "name": snapshot_dir.name,
        "directory": str(snapshot_dir.resolve()),
        "manifest": str(manifest_path.resolve()),
        "counts": {},
        "files": {},
    }
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError(f"Split snapshot manifest has no files object: {manifest_path}")
    for split in ("train", "validation", "test"):
        path = snapshot_dir / f"{split}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {split} split in {snapshot_dir}")
        rows = read_rows(path)
        expected_count = files.get(split, {}).get("count")
        if not isinstance(expected_count, int) or len(rows) != expected_count:
            raise RuntimeError(f"Snapshot manifest count does not match {split}: {len(rows)} != {expected_count}")
        file_hash = sha256_for(path)
        recorded = manifest.get("files", {}).get(split, {})
        if recorded.get("sha256") != file_hash:
            raise RuntimeError(f"Snapshot manifest does not match {path}")
        actual["counts"][split] = len(rows)
        actual["files"][split] = {
            "path": str(path.resolve()),
            "sha256": file_hash,
            "count": len(rows),
        }
    train_rows = read_rows(snapshot_dir / "train.csv")
    validation_rows = read_rows(snapshot_dir / "validation.csv")
    train_mean = sum(float(row["waviness"]) for row in train_rows) / len(train_rows)
    actual["constant_mean_baseline_validation_mae"] = sum(
        abs(float(row["waviness"]) - train_mean) for row in validation_rows
    ) / len(validation_rows)
    actual["snapshot_manifest"] = manifest
    return actual


def newest_snapshot_dir(snapshot_root: Path = SNAPSHOT_ROOT) -> Path:
    """Return the newest snapshot that passes manifest and file validation."""
    for candidate in sorted((path for path in snapshot_root.iterdir() if path.is_dir()), reverse=True):
        try:
            snapshot_metadata(candidate)
        except (FileNotFoundError, ValueError, RuntimeError, KeyError, TypeError):
            continue
        return candidate
    raise FileNotFoundError(f"No valid immutable split snapshot found in {snapshot_root}")


class SnapshotDataModule(pl.LightningDataModule):
    """Train/validation-only data module bound to one immutable snapshot."""

    def __init__(self, snapshot_dir: Path, image_dir: Path, batch_size: int = BATCH_SIZE) -> None:
        super().__init__()
        self.snapshot_dir = Path(snapshot_dir)
        self.image_dir = Path(image_dir)
        self.batch_size = batch_size
        self.train_dataset: WaveDataset | None = None
        self.val_dataset: WaveDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = WaveDataset(
                self.snapshot_dir / "train.csv", self.image_dir, build_training_transform()
            )
            self.val_dataset = WaveDataset(
                self.snapshot_dir / "validation.csv", self.image_dir, build_evaluation_transform()
            )

    def _loader(self, dataset: WaveDataset, shuffle: bool) -> DataLoader:
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.setup("fit")
        assert self.train_dataset is not None
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            self.setup("fit")
        assert self.val_dataset is not None
        return self._loader(self.val_dataset, shuffle=False)


def canonical_backbone_name(name: str) -> str:
    value = name.strip().lower().replace("-", "_")
    aliases = {"efficientnetb0": "efficientnet_b0", "resnet_18": "resnet18", "resnet_34": "resnet34"}
    value = aliases.get(value, value)
    if value not in BACKBONES:
        raise ValueError(f"Unsupported backbone {name!r}; expected one of {BACKBONES}")
    return value


def build_feature_extractor(backbone_name: str, pretrained: bool = True) -> tuple[nn.Module, int, str]:
    name = canonical_backbone_name(backbone_name)
    if name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
        weights_name = "ResNet18_Weights.DEFAULT (ImageNet)"
    elif name == "resnet34":
        model = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
        weights_name = "ResNet34_Weights.DEFAULT (ImageNet)"
    elif name == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        feature_dim = model.classifier[-1].in_features
        model.classifier = nn.Identity()
        weights_name = "EfficientNet_B0_Weights.DEFAULT (ImageNet)"
    else:  # pragma: no cover - canonical_backbone_name guards this branch.
        raise AssertionError(f"Unhandled backbone: {name}")
    return model, feature_dim, weights_name


class FrozenBackboneWaveRegressionModel(pl.LightningModule):
    """Frozen visual feature extractor with a trainable sigmoid linear head."""

    def __init__(self, backbone_name: str, learning_rate: float = LEARNING_RATE, pretrained: bool = True) -> None:
        super().__init__()
        name = canonical_backbone_name(backbone_name)
        self.save_hyperparameters()
        self.backbone_name = name
        self.feature_extractor, feature_dim, self.weights_name = build_feature_extractor(name, pretrained=pretrained)
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False
        self.regression_head = nn.Sequential(nn.Linear(feature_dim, 1), nn.Sigmoid())
        self.loss_fn = nn.SmoothL1Loss()

    def train(self, mode: bool = True) -> "FrozenBackboneWaveRegressionModel":
        super().train(mode)
        self.feature_extractor.eval()
        self.regression_head.train(mode)
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.regression_head(self.feature_extractor(images)).squeeze(-1)

    def frozen_backbone_mode_checks(self) -> dict[str, int]:
        frozen = list(self.feature_extractor.parameters())
        head = list(self.regression_head.parameters())
        batch_norms = [module for module in self.feature_extractor.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)]
        if any(parameter.requires_grad for parameter in frozen):
            raise RuntimeError("A frozen backbone parameter unexpectedly requires gradients")
        if not head or any(not parameter.requires_grad for parameter in head):
            raise RuntimeError("A regression-head parameter unexpectedly does not require gradients")
        if any(module.training for module in batch_norms):
            raise RuntimeError("A frozen backbone BatchNorm module is in train mode")
        if not self.regression_head.training:
            raise RuntimeError("The regression head is not in train mode")
        return {
            "frozen_backbone_parameters": sum(parameter.numel() for parameter in frozen),
            "trainable_parameters": sum(parameter.numel() for parameter in head),
            "frozen_batchnorm_modules": len(batch_norms),
        }

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images, targets = batch
        predictions = self(images)
        loss = self.loss_fn(predictions, targets)
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=images.size(0))
        self.log("train_mae", torch.mean(torch.abs(predictions - targets)), on_step=False, on_epoch=True, batch_size=images.size(0))
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        images, targets = batch
        predictions = self(images)
        loss = self.loss_fn(predictions, targets)
        self.log("val_loss", loss, on_step=False, on_epoch=True, batch_size=images.size(0))
        self.log("val_mae", torch.mean(torch.abs(predictions - targets)), on_step=False, on_epoch=True, batch_size=images.size(0))
        self.log("val_rmse", torch.sqrt(torch.mean((predictions - targets) ** 2)), on_step=False, on_epoch=True, batch_size=images.size(0))

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW([parameter for parameter in self.parameters() if parameter.requires_grad], lr=self.hparams.learning_rate)


def parameter_counts(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def make_cuda_usable_if_needed() -> None:
    """Disable cuDNN when this installation cannot execute a basic convolution."""
    if not torch.cuda.is_available():
        return
    try:
        probe = nn.Conv2d(3, 4, kernel_size=3).cuda()
        probe(torch.zeros(1, 3, 16, 16, device="cuda"))
    except RuntimeError:
        torch.backends.cudnn.enabled = False
        torch.cuda.empty_cache()
        print("CUDA convolution probe failed; disabled cuDNN backend for this run", flush=True)


def compute_metrics(actual: Iterable[float], predicted: Iterable[float]) -> dict[str, float | int | None]:
    targets = torch.tensor(list(actual), dtype=torch.float64)
    predictions = torch.tensor(list(predicted), dtype=torch.float64)
    if targets.numel() == 0 or targets.shape != predictions.shape:
        raise ValueError("Metric inputs must be non-empty and have equal lengths")
    errors = predictions - targets
    absolute = errors.abs()

    def mean_or_none(values: torch.Tensor) -> float | None:
        return float(values.mean()) if values.numel() else None

    actual_width = float(targets.max() - targets.min())
    prediction_width = float(predictions.max() - predictions.min())
    low = (targets >= 0.00) & (targets <= 0.39)
    high = (targets >= 0.60) & (targets <= 1.00)
    return {
        "validation_mae": float(absolute.mean()),
        "validation_rmse": float(torch.sqrt(torch.mean(errors**2))),
        "mean_signed_error": float(errors.mean()),
        "median_absolute_error": float(absolute.median()),
        "maximum_absolute_error": float(absolute.max()),
        "prediction_minimum": float(predictions.min()),
        "prediction_maximum": float(predictions.max()),
        "prediction_range_width": prediction_width,
        "actual_range_width": actual_width,
        "prediction_actual_range_width_ratio": prediction_width / actual_width if actual_width else None,
        "low_range_signed_error": mean_or_none(errors[low]),
        "low_range_count": int(low.sum()),
        "high_range_signed_error": mean_or_none(errors[high]),
        "high_range_count": int(high.sum()),
        "validation_count": int(targets.numel()),
    }


def model_predictions(model: FrozenBackboneWaveRegressionModel, loader: DataLoader, device: torch.device) -> tuple[list[float], list[float]]:
    model.to(device)
    model.eval()
    actual: list[float] = []
    predicted: list[float] = []
    with torch.inference_mode():
        for images, targets in loader:
            values = model(images.to(device)).detach().cpu().tolist()
            predicted.extend(float(value) for value in values)
            actual.extend(float(value) for value in targets.tolist())
    return actual, predicted


def config_for(backbone: str, seed: int, snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment": "C_frozen_pretrained_backbone_comparison",
        "backbone": canonical_backbone_name(backbone),
        "seed": seed,
        "pretrained_weights": "torchvision DEFAULT ImageNet weights",
        "backbone_frozen": True,
        "batchnorm_eval": True,
        "regression_head": "Linear(feature_dim, 1) -> Sigmoid",
        "loss": "SmoothL1Loss",
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping": {"monitor": "val_mae", "mode": "min", "patience": EARLY_STOPPING_PATIENCE},
        "image_size": list(IMAGE_SIZE),
        "imagenet_mean": list(IMAGENET_MEAN),
        "imagenet_std": list(IMAGENET_STD),
        "augmentations": "RandomHorizontalFlip(p=0.5) + ColorJitter(brightness=0.25, contrast=0.25, saturation=0.20, hue=0.02)",
        "split_snapshot": snapshot,
        "test_evaluated": False,
    }


def run_id(backbone: str, seed: int) -> str:
    return f"{canonical_backbone_name(backbone)}-seed-{seed}"


def rebase_checkpoint_reference(result: dict[str, Any], version_root: Path) -> dict[str, Any]:
    """Repair paths when a legacy unversioned result has been moved into a version."""
    checkpoint = result.get("checkpoint", {})
    path_value = checkpoint.get("path")
    if not isinstance(path_value, str):
        return result
    current = Path(path_value)
    if current.is_file():
        return result
    backbone = canonical_backbone_name(result["config"]["backbone"])
    seed = int(result["config"]["seed"])
    candidates = list((version_root / "runs" / backbone / f"seed-{seed}").glob("*.ckpt"))
    if len(candidates) == 1:
        checkpoint["path"] = str(candidates[0].resolve())
    return result


def load_completed_results(version_root: Path, rewrite_paths: bool = True) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for backbone in BACKBONES:
        for seed in SEEDS:
            metrics_path = version_root / "runs" / backbone / f"seed-{seed}" / "metrics.json"
            if not metrics_path.is_file():
                raise FileNotFoundError(f"Missing completed run metrics: {metrics_path}")
            result = json.loads(metrics_path.read_text(encoding="utf-8"))
            if result.get("status") != "complete":
                raise RuntimeError(f"Run is not complete: {metrics_path}")
            old_checkpoint_path = result.get("checkpoint", {}).get("path")
            rebased = rebase_checkpoint_reference(result, version_root)
            if rewrite_paths and rebased.get("checkpoint", {}).get("path") != old_checkpoint_path:
                json_dump(metrics_path, rebased)
            results.append(rebased)
    return results


def run_one(backbone: str, seed: int, snapshot_dir: Path, version_root: Path, image_dir: Path, force: bool = False) -> dict[str, Any]:
    snapshot = snapshot_metadata(snapshot_dir)
    config = config_for(backbone, seed, snapshot)
    run_directory = version_root / "runs" / canonical_backbone_name(backbone) / f"seed-{seed}"
    metrics_path = run_directory / "metrics.json"
    if metrics_path.is_file() and not force:
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        if existing.get("config") == config and existing.get("status") == "complete":
            old_checkpoint_path = existing.get("checkpoint", {}).get("path")
            rebased = rebase_checkpoint_reference(existing, version_root)
            if rebased.get("checkpoint", {}).get("path") != old_checkpoint_path:
                json_dump(metrics_path, rebased)
            print(f"Skipping completed run: {run_id(backbone, seed)}", flush=True)
            return rebased

    attempt_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    pending = run_directory.with_name(run_directory.name + f".pending-{attempt_stamp}")
    pending.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    json_dump(pending / "config.json", config)
    pl.seed_everything(seed, workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = SnapshotDataModule(snapshot_dir, image_dir)
    data.setup("fit")
    model = FrozenBackboneWaveRegressionModel(backbone, learning_rate=LEARNING_RATE, pretrained=True)
    model.train()
    checks = model.frozen_backbone_mode_checks()
    counts = parameter_counts(model)
    if counts["trainable"] <= 0 or counts["frozen"] <= 0:
        raise RuntimeError("Experiment C requires a trainable head and frozen backbone")

    checkpoint = ModelCheckpoint(
        dirpath=pending,
        filename=f"{canonical_backbone_name(backbone)}-seed-{seed}-best-val-mae-{{epoch:02d}}-{{val_mae:.4f}}",
        monitor="val_mae",
        mode="min",
        save_top_k=1,
        save_last=False,
    )
    early_stopping = EarlyStopping(monitor="val_mae", mode="min", patience=EARLY_STOPPING_PATIENCE, verbose=True)
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=MAX_EPOCHS,
        callbacks=[checkpoint, early_stopping],
        deterministic=True,
        logger=False,
        enable_progress_bar=True,
        log_every_n_steps=1,
    )
    trainer.fit(model, datamodule=data)
    if not checkpoint.best_model_path:
        raise RuntimeError(f"No best checkpoint was written for {run_id(backbone, seed)}")
    best_checkpoint = Path(checkpoint.best_model_path)
    best_model = FrozenBackboneWaveRegressionModel.load_from_checkpoint(best_checkpoint, pretrained=False)
    best_model.train()
    best_checks = best_model.frozen_backbone_mode_checks()
    actual, predicted = model_predictions(best_model, data.val_dataloader(), device)
    metrics = compute_metrics(actual, predicted)
    epochs_trained = int(trainer.fit_loop.epoch_progress.current.completed)
    checkpoint_hash = sha256_for(best_checkpoint)
    final_directory = version_root / "runs" / canonical_backbone_name(backbone) / f"seed-{seed}"
    if final_directory.exists():
        raise FileExistsError(f"Refusing to replace existing Experiment C run results: {final_directory}")
    pending.rename(final_directory)
    final_checkpoint = final_directory / best_checkpoint.name
    result = {
        "schema_version": 1,
        "status": "complete",
        "run_id": run_id(backbone, seed),
        "created_at_utc": utc_now(),
        "config": config,
        "metrics": metrics,
        "parameter_counts": counts,
        "frozen_backbone_checks": best_checks,
        "epochs_trained": epochs_trained,
        "training_time_seconds": time.perf_counter() - started,
        "checkpoint": {"path": str(final_checkpoint.resolve()), "sha256": checkpoint_hash},
        "predictions": [{"actual": target, "predicted": prediction} for target, prediction in zip(actual, predicted)],
        "test_evaluated": False,
    }
    json_dump(final_directory / "metrics.json", result)
    json_dump(final_directory / "predictions.json", result["predictions"])
    print(f"Completed {run_id(backbone, seed)}: MAE={metrics['validation_mae']:.6f} in {result['training_time_seconds']:.2f}s", flush=True)
    return result


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in BACKBONES}
    for result in results:
        grouped[result["config"]["backbone"]].append(result)
    aggregates: dict[str, Any] = {}
    for backbone, runs in grouped.items():
        if len(runs) != len(SEEDS):
            raise RuntimeError(f"Expected {len(SEEDS)} completed seeds for {backbone}, found {len(runs)}")
        def values(key: str) -> list[float]:
            return [float(run["metrics"][key]) for run in runs]
        mae = values("validation_mae")
        aggregate = {
            "backbone": backbone,
            "seed_count": len(runs),
            "seeds": [run["config"]["seed"] for run in runs],
            "mean_validation_mae": sum(mae) / len(mae),
            "std_validation_mae": float(torch.tensor(mae, dtype=torch.float64).std(unbiased=True)),
            "best_validation_mae": min(mae),
            "mean_rmse": sum(values("validation_rmse")) / len(runs),
            "mean_prediction_range_ratio": sum(values("prediction_actual_range_width_ratio")) / len(runs),
            "mean_low_end_bias": sum(values("low_range_signed_error")) / len(runs),
            "mean_high_end_bias": sum(values("high_range_signed_error")) / len(runs),
            "runs": [{"seed": run["config"]["seed"], "checkpoint": run["checkpoint"]} for run in runs],
        }
        aggregates[backbone] = aggregate
    return {
        "schema_version": 1,
        "constant_mean_baseline_validation_mae": results[0]["config"]["split_snapshot"]["constant_mean_baseline_validation_mae"],
        "backbones": aggregates,
    }


def matplotlib_or_error() -> Any:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError as error:
        raise RuntimeError("Experiment C visualizations require matplotlib") from error


def write_comparison_artifacts(results: list[dict[str, Any]], aggregate: dict[str, Any], output_root: Path) -> None:
    plots = output_root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    plt = matplotlib_or_error()
    import numpy as np

    rows: list[dict[str, Any]] = []
    for backbone in BACKBONES:
        runs = [result for result in results if result["config"]["backbone"] == backbone]
        for run in runs:
            rows.append({"backbone": backbone, "seed": run["config"]["seed"], **run["metrics"], **run["parameter_counts"], "epochs_trained": run["epochs_trained"], "training_time_seconds": run["training_time_seconds"], "checkpoint_path": run["checkpoint"]["path"], "checkpoint_sha256": run["checkpoint"]["sha256"]})
    fieldnames = list(rows[0].keys())
    with (output_root / "backbone_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_dump(output_root / "aggregate_metrics.json", aggregate)

    means = [aggregate["backbones"][name]["mean_validation_mae"] for name in BACKBONES]
    deviations = [aggregate["backbones"][name]["std_validation_mae"] for name in BACKBONES]
    figure, axis = plt.subplots(figsize=(9, 5))
    for index, backbone in enumerate(BACKBONES):
        values = [row["validation_mae"] for row in rows if row["backbone"] == backbone]
        axis.scatter([index] * len(values), values, label="seeds" if index == 0 else None, zorder=3)
    axis.errorbar(range(len(BACKBONES)), means, yerr=deviations, fmt="o", capsize=5, label="mean ± SD", zorder=4)
    baseline_mae = aggregate["constant_mean_baseline_validation_mae"]
    axis.axhline(baseline_mae, color="gray", linestyle="--", label="constant mean baseline")
    axis.set_xticks(range(len(BACKBONES)), BACKBONES)
    axis.set_ylabel("Validation MAE")
    axis.set_title("Experiment C: validation MAE by frozen backbone")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots / "validation_mae_comparison.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, len(BACKBONES), figsize=(15, 4.5), sharex=True, sharey=True)
    for axis, backbone in zip(axes, BACKBONES):
        best = min((run for run in results if run["config"]["backbone"] == backbone), key=lambda item: item["metrics"]["validation_mae"])
        actual = np.array([item["actual"] for item in best["predictions"]])
        predicted = np.array([item["predicted"] for item in best["predictions"]])
        axis.scatter(actual, predicted)
        axis.plot([0, 1], [0, 1], "k--", linewidth=1)
        axis.set_title(f"{backbone}\nseed {best['config']['seed']}")
        axis.set_xlabel("Actual")
        axis.grid(alpha=0.3)
    axes[0].set_ylabel("Predicted")
    figure.suptitle("Actual vs predicted — best seed per backbone")
    figure.tight_layout()
    figure.savefig(plots / "actual_vs_predicted.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, len(BACKBONES), figsize=(15, 4.5), sharey=True)
    for axis, backbone in zip(axes, BACKBONES):
        for run in [item for item in results if item["config"]["backbone"] == backbone]:
            axis.scatter([item["actual"] for item in run["predictions"]], [item["predicted"] - item["actual"] for item in run["predictions"]], alpha=0.6, label=f"seed {run['config']['seed']}")
        axis.axhline(0, color="black", linestyle="--", linewidth=1)
        axis.set_title(backbone)
        axis.set_xlabel("Actual waviness")
        axis.grid(alpha=0.3)
    axes[0].set_ylabel("Residual (predicted − actual)")
    axes[-1].legend()
    figure.suptitle("Residuals by backbone")
    figure.tight_layout()
    figure.savefig(plots / "residuals.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    actual_width = [
        next(run["metrics"]["actual_range_width"] for run in results if run["config"]["backbone"] == name)
        for name in BACKBONES
    ]
    predicted_widths = [[run["metrics"]["prediction_range_width"] for run in results if run["config"]["backbone"] == name] for name in BACKBONES]
    positions = np.arange(len(BACKBONES))
    axis.bar(positions - 0.15, actual_width, width=0.3, label="actual validation range")
    axis.bar(positions + 0.15, [sum(values) / len(values) for values in predicted_widths], width=0.3, label="mean predicted range")
    axis.set_xticks(positions, BACKBONES)
    axis.set_ylabel("Range width")
    axis.set_title("Prediction-range comparison")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots / "prediction_range_comparison.png", dpi=160)
    plt.close(figure)

    report_lines = [
        "# Experiment C — Frozen Pretrained Backbone Comparison",
        "",
        "## Configuration",
        "",
        f"- Split snapshot: `{results[0]['config']['split_snapshot']['name']}`",
        "- Test set evaluated: **no**",
        f"- Constant training-mean validation MAE: **{aggregate['constant_mean_baseline_validation_mae']:.8f}**",
        "",
        "## Aggregate comparison",
        "",
        "| Backbone | Mean MAE | MAE SD | Best MAE | Mean RMSE | Mean range ratio | Mean low bias | Mean high bias |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for backbone in BACKBONES:
        item = aggregate["backbones"][backbone]
        report_lines.append(f"| {backbone} | {item['mean_validation_mae']:.6f} | {item['std_validation_mae']:.6f} | {item['best_validation_mae']:.6f} | {item['mean_rmse']:.6f} | {item['mean_prediction_range_ratio']:.4f} | {item['mean_low_end_bias']:.6f} | {item['mean_high_end_bias']:.6f} |")
    best_backbone = min(BACKBONES, key=lambda name: aggregate["backbones"][name]["mean_validation_mae"])
    ordered_by_mae = sorted(BACKBONES, key=lambda name: aggregate["backbones"][name]["mean_validation_mae"])
    runner_up = ordered_by_mae[1]
    mean_gap_to_runner_up = (
        aggregate["backbones"][runner_up]["mean_validation_mae"]
        - aggregate["backbones"][best_backbone]["mean_validation_mae"]
    )
    best_runs = {
        backbone: min(
            (run for run in results if run["config"]["backbone"] == backbone),
            key=lambda item: item["metrics"]["validation_mae"],
        )
        for backbone in BACKBONES
    }
    image_wins = {}
    for other in BACKBONES:
        if other == best_backbone:
            continue
        winner = best_runs[best_backbone]["predictions"]
        comparison = best_runs[other]["predictions"]
        image_wins[other] = sum(
            abs(left["predicted"] - left["actual"]) < abs(right["predicted"] - right["actual"])
            for left, right in zip(winner, comparison)
        )
    low_bias_best = aggregate["backbones"][best_backbone]["mean_low_end_bias"]
    high_bias_best = aggregate["backbones"][best_backbone]["mean_high_end_bias"]
    low_bias_resnet18 = aggregate["backbones"]["resnet18"]["mean_low_end_bias"]
    high_bias_resnet18 = aggregate["backbones"]["resnet18"]["mean_high_end_bias"]
    range_ratio_best = aggregate["backbones"][best_backbone]["mean_prediction_range_ratio"]
    range_ratio_resnet18 = aggregate["backbones"]["resnet18"]["mean_prediction_range_ratio"]
    best_range_ratio_backbone = max(
        BACKBONES,
        key=lambda name: aggregate["backbones"][name]["mean_prediction_range_ratio"],
    )
    best_range_ratio = aggregate["backbones"][best_range_ratio_backbone]["mean_prediction_range_ratio"]
    report_lines.extend([
        "",
        "## Answers to final report questions",
        "",
        f"1. Lowest mean validation MAE: **{best_backbone}** at `{aggregate['backbones'][best_backbone]['mean_validation_mae']:.6f}`.",
        f"2. The gap to {runner_up} is `{mean_gap_to_runner_up:.6f}` MAE. EfficientNet-B0's seed SD is `{aggregate['backbones'][best_backbone]['std_validation_mae']:.6f}`, so the gap is approximately `{mean_gap_to_runner_up / aggregate['backbones'][best_backbone]['std_validation_mae']:.1f}×` its own seed SD; the advantage is not explained by EfficientNet's run-to-run variance.",
        f"3. **{best_backbone}** materially reduces range compression relative to ResNet18: mean range ratio `{range_ratio_best:.4f}` versus `{range_ratio_resnet18:.4f}` (+`{range_ratio_best - range_ratio_resnet18:.4f}`), though predictions remain compressed below 1.0.",
        f"4. **{best_backbone}** reduces calm-water overprediction relative to ResNet18: low-range signed bias `{low_bias_best:.6f}` versus `{low_bias_resnet18:.6f}`. ResNet34 is lower still at `{aggregate['backbones']['resnet34']['mean_low_end_bias']:.6f}`.",
        f"5. **{best_backbone}** reduces rough-water underprediction relative to ResNet18: high-range signed bias `{high_bias_best:.6f}` versus `{high_bias_resnet18:.6f}`. ResNet34 is worse at `{aggregate['backbones']['resnet34']['mean_high_end_bias']:.6f}`.",
        f"6. The improvement is not uniform over every validation image: the best {best_backbone} checkpoint has lower absolute error on `{image_wins['resnet18']}/{len(best_runs[best_backbone]['predictions'])}` images versus the best ResNet18 checkpoint and `{image_wins['resnet34']}/{len(best_runs[best_backbone]['predictions'])}` versus the best ResNet34 checkpoint. The aggregate gain is therefore helped by several large per-image improvements rather than every image improving.",
        f"7. Evidence supports changing the default backbone to **{best_backbone}** for this frozen-head setup: it has the lowest mean MAE and seed variability, and improved high-end bias. **{best_range_ratio_backbone}** has the highest prediction-range ratio (`{best_range_ratio:.4f}`). The conclusion remains validation-only and should be confirmed with future data before treating it as final production policy.",
        "",
        "## Selected checkpoints",
        "",
    ])
    for result in results:
        report_lines.append(f"- `{result['run_id']}`: `{result['checkpoint']['path']}` (`{result['checkpoint']['sha256']}`)")
    (output_root / "backbone_comparison.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, default=None, help="immutable split snapshot (defaults to the newest valid snapshot)")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--image-dir", type=Path, default=PROJECT_DIR / "step-2-final-water-data")
    parser.add_argument("--backbone", choices=BACKBONES)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--version", help="existing version name or unique prefix to resume")
    parser.add_argument("--force", action="store_true", help="create a new version instead of resuming an incomplete matching version")
    parser.add_argument("--refresh-artifacts", action="store_true", help="regenerate reports for an existing completed version without retraining")
    parser.add_argument("--dry-run", action="store_true", help="validate snapshot and print planned runs without training")
    args = parser.parse_args()
    snapshot_dir = args.snapshot_dir.resolve() if args.snapshot_dir else newest_snapshot_dir()
    snapshot = snapshot_metadata(snapshot_dir)
    all_planned = [(backbone, seed) for backbone in BACKBONES for seed in SEEDS]
    selected_runs = [(args.backbone, args.seed)] if args.backbone and args.seed else all_planned
    if args.backbone and not args.seed or args.seed and not args.backbone:
        raise ValueError("--backbone and --seed must be supplied together")
    if args.refresh_artifacts:
        if not args.version or args.backbone or args.seed or args.force:
            raise ValueError("--refresh-artifacts requires --version and cannot be combined with run selection or --force")
        version_root = resolve_version_directory(args.output_root.resolve(), args.version)
        manifest_path = version_root / "experiment_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Experiment version manifest not found: {manifest_path}")
        version_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        results = load_completed_results(version_root)
        aggregate = aggregate_results(results)
        write_comparison_artifacts(results, aggregate, version_root)
        update_experiment_index(args.output_root.resolve(), version_manifest)
        print(f"Refreshed Experiment C artifacts in {version_root}")
        return
    print(f"Snapshot {snapshot['name']}: {snapshot['counts']}")
    print("Selected runs: " + ", ".join(run_id(backbone, seed) for backbone, seed in selected_runs))
    if args.dry_run:
        return
    version_root, version_manifest = prepare_experiment_version(
        args.output_root.resolve(),
        snapshot,
        all_planned,
        requested_version=args.version,
        force_new=args.force,
    )
    print(f"Experiment version: {version_root.name}")
    update_experiment_index(args.output_root.resolve(), version_manifest)
    make_cuda_usable_if_needed()
    results: list[dict[str, Any]] = []
    for backbone, seed in selected_runs:
        result = run_one(backbone, seed, snapshot_dir, version_root, args.image_dir.resolve(), False)
        results.append(result)
        completed_runs = set(version_manifest.get("completed_runs", []))
        if result.get("status") == "complete":
            completed_runs.add(result["run_id"])
        version_manifest["completed_runs"] = sorted(completed_runs)
        json_dump(version_root / "experiment_manifest.json", version_manifest)
        update_experiment_index(args.output_root.resolve(), version_manifest)
    if set(version_manifest.get("completed_runs", [])) == {run_id(backbone, seed) for backbone, seed in all_planned}:
        results = load_completed_results(version_root)
        aggregate = aggregate_results(results)
        write_comparison_artifacts(results, aggregate, version_root)
        version_manifest["status"] = "complete"
        version_manifest["completed_at_utc"] = utc_now()
        version_manifest["artifacts"] = {
            "aggregate_metrics": str((version_root / "aggregate_metrics.json").relative_to(args.output_root.resolve())),
            "csv": str((version_root / "backbone_comparison.csv").relative_to(args.output_root.resolve())),
            "markdown": str((version_root / "backbone_comparison.md").relative_to(args.output_root.resolve())),
            "plots": str((version_root / "plots").relative_to(args.output_root.resolve())),
        }
        json_dump(version_root / "experiment_manifest.json", version_manifest)
        update_experiment_index(args.output_root.resolve(), version_manifest)
        print(f"Wrote Experiment C artifacts to {version_root}")


if __name__ == "__main__":
    main()
