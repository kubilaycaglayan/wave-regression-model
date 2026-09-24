"""Experiment A: compare constant waviness predictors with the neural network."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from step_4_a_wave_dataset import WaveDataset, build_evaluation_transform
from step_5_a_wave_regression_model import WaveRegressionModel


PROJECT_DIR = Path(__file__).resolve().parent
SPLIT_ROOT = Path("step-3-dataset-splits")
SNAPSHOT_ROOT = SPLIT_ROOT / "snapshots"
CHECKPOINT_ROOT = Path("step-5-checkpoints")
IMAGE_DIR = Path("step-2-final-water-data")
JSON_OUTPUT = Path("baseline_experiment.json")
MARKDOWN_OUTPUT = Path("baseline_experiment.md")
EXPECTED_COUNTS = {"train": 68, "validation": 14, "test": 14}
BATCH_SIZE = 8


@dataclass(frozen=True)
class Snapshot:
    name: str
    directory: Path
    manifest: dict[str, Any]

    @property
    def train_csv(self) -> Path:
        return self.directory / "train.csv"

    @property
    def validation_csv(self) -> Path:
        return self.directory / "validation.csv"

    @property
    def test_csv(self) -> Path:
        return self.directory / "test.csv"


@dataclass(frozen=True)
class CheckpointSelection:
    path: Path
    manifest_path: Path
    metadata: dict[str, Any]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def find_snapshot(snapshot_root: Path = SNAPSHOT_ROOT) -> Snapshot:
    candidates: list[Snapshot] = []
    if snapshot_root.is_dir():
        for directory in sorted((path for path in snapshot_root.iterdir() if path.is_dir()), reverse=True):
            manifest_path = directory / "manifest.json"
            if not manifest_path.is_file():
                continue
            manifest = load_json(manifest_path)
            files = manifest.get("files")
            if not isinstance(files, dict):
                continue
            counts = {name: files.get(name, {}).get("count") for name in EXPECTED_COUNTS}
            if counts != EXPECTED_COUNTS:
                continue
            if not all((directory / f"{name}.csv").is_file() for name in EXPECTED_COUNTS):
                continue
            candidates.append(Snapshot(directory.name, directory, manifest))
    if not candidates:
        raise FileNotFoundError(
            f"No immutable split snapshot with counts {EXPECTED_COUNTS} found in {snapshot_root}"
        )
    return candidates[0]


def _normalise_snapshot_reference(reference: str, project_dir: Path) -> Path:
    path = Path(reference)
    return (path if path.is_absolute() else project_dir / path).resolve()


def select_checkpoint(
    snapshot: Snapshot,
    checkpoint_root: Path = CHECKPOINT_ROOT,
    project_dir: Path = PROJECT_DIR,
) -> CheckpointSelection | None:
    matches: list[CheckpointSelection] = []
    for manifest_path in sorted(checkpoint_root.glob("training_manifest_*.json")):
        metadata = load_json(manifest_path)
        reference = metadata.get("split_snapshot")
        checkpoint_reference = metadata.get("best_checkpoint_path")
        if not isinstance(reference, str) or not isinstance(checkpoint_reference, str):
            continue
        if _normalise_snapshot_reference(reference, project_dir) != snapshot.directory.resolve():
            continue
        checkpoint_path = _normalise_snapshot_reference(checkpoint_reference, project_dir)
        if not checkpoint_path.is_file():
            continue
        recorded_mae = metadata.get("best_validation_mae")
        if not isinstance(recorded_mae, (int, float)) or not math.isfinite(float(recorded_mae)):
            continue
        matches.append(CheckpointSelection(checkpoint_path, manifest_path, metadata))
    if not matches:
        return None
    return min(
        matches,
        key=lambda item: (float(item.metadata["best_validation_mae"]), str(item.path)),
    )


def mean_absolute_error(targets: list[float], predictions: list[float]) -> float:
    if not targets or len(targets) != len(predictions):
        raise ValueError("MAE requires equally sized, non-empty target and prediction lists")
    return statistics.mean(abs(target - prediction) for target, prediction in zip(targets, predictions))


def relative_reduction(baseline_mae: float, model_mae: float) -> float | None:
    if baseline_mae == 0.0:
        return None
    return (baseline_mae - model_mae) / baseline_mae * 100.0


def select_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        probe = torch.nn.Conv2d(3, 4, kernel_size=3).cuda()
        probe(torch.zeros(1, 3, 16, 16, device="cuda"))
    except RuntimeError as error:
        torch.cuda.empty_cache()
        print(f"CUDA convolution probe failed; using CPU ({error})")
        return torch.device("cpu")
    return torch.device("cuda")


def evaluate_checkpoint(
    checkpoint: CheckpointSelection,
    validation_dataset: WaveDataset,
    device: torch.device,
) -> tuple[float, float]:
    started = time.perf_counter()
    model = WaveRegressionModel.load_from_checkpoint(checkpoint.path, map_location=device)
    model.to(device)
    model.eval()
    print(f"Checkpoint loaded in {time.perf_counter() - started:.2f}s")

    loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False)
    targets: list[float] = []
    predictions: list[float] = []
    inference_started = time.perf_counter()
    with torch.inference_mode():
        for batch_number, (images, batch_targets) in enumerate(loader, start=1):
            item_started = time.perf_counter()
            batch_predictions = model(images.to(device)).detach().cpu()
            if not torch.isfinite(batch_predictions).all():
                raise RuntimeError(f"Non-finite checkpoint prediction in validation batch {batch_number}")
            if not torch.all((batch_predictions >= 0.0) & (batch_predictions <= 1.0)):
                raise RuntimeError(f"Checkpoint prediction outside [0, 1] in validation batch {batch_number}")
            predictions.extend(float(value) for value in batch_predictions)
            targets.extend(float(value) for value in batch_targets)
            print(
                f"Validation batch {batch_number}: {len(batch_targets)} samples "
                f"in {time.perf_counter() - item_started:.3f}s"
            )
    elapsed = time.perf_counter() - inference_started
    print(f"Validation inference completed in {elapsed:.2f}s")
    return mean_absolute_error(targets, predictions), elapsed


def _base_result(snapshot: Snapshot, train_dataset: WaveDataset, validation_dataset: WaveDataset) -> dict[str, Any]:
    train_labels = [label for _, label in train_dataset.samples]
    validation_labels = [label for _, label in validation_dataset.samples]
    train_mean = statistics.mean(train_labels)
    train_median = statistics.median(train_labels)
    mean_mae = mean_absolute_error(validation_labels, [train_mean] * len(validation_labels))
    median_mae = mean_absolute_error(validation_labels, [train_median] * len(validation_labels))
    return {
        "schema_version": 1,
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "split_snapshot": {
            "name": snapshot.name,
            "directory": str(snapshot.directory),
            "manifest": str(snapshot.directory / "manifest.json"),
            "counts": {"train": len(train_dataset), "validation": len(validation_dataset), "test": EXPECTED_COUNTS["test"]},
            "train_sha256": sha256(snapshot.train_csv),
            "validation_sha256": sha256(snapshot.validation_csv),
            "test_sha256": sha256(snapshot.test_csv),
        },
        "predictors": {
            "training_mean": {
                "prediction": train_mean,
                "validation_mae": mean_mae,
                "relative_mae_reduction_percent": None,
            },
            "training_median": {
                "prediction": train_median,
                "validation_mae": median_mae,
                "relative_mae_reduction_percent": None,
            },
            "neural_network": None,
        },
        "training_label_count": len(train_labels),
        "validation_label_count": len(validation_labels),
        "test_evaluated": False,
    }


def write_reports(result: dict[str, Any], json_path: Path = JSON_OUTPUT, markdown_path: Path = MARKDOWN_OUTPUT) -> None:
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    snapshot = result["split_snapshot"]
    predictors = result["predictors"]
    lines = [
        "# Experiment A: Constant-Prediction Baselines",
        "",
        f"Status: **{result['status']}**",
        "",
        f"Split snapshot: `{snapshot['name']}` ({snapshot['counts']['train']} train, "
        f"{snapshot['counts']['validation']} validation, {snapshot['counts']['test']} test)",
        "",
        "All reported MAEs use the same validation images. The test split was not evaluated.",
        "",
        "| Predictor | Constant/value | Validation MAE | Relative MAE reduction |",
        "|---|---:|---:|---:|",
    ]
    for name, label in (("training_mean", "Training mean"), ("training_median", "Training median"), ("neural_network", "Neural network")):
        predictor = predictors[name]
        if predictor is None:
            lines.append(f"| {label} | unavailable | unavailable | unavailable |")
            continue
        reduction = predictor["relative_mae_reduction_percent"]
        reduction_text = "n/a" if reduction is None else f"{reduction:.2f}%"
        prediction = "n/a" if predictor["prediction"] is None else f"{predictor['prediction']:.8f}"
        mae = "n/a" if predictor["validation_mae"] is None else f"{predictor['validation_mae']:.8f}"
        lines.append(
            f"| {label} | {prediction} | {mae} | {reduction_text} |"
        )
    if result.get("checkpoint"):
        checkpoint = result["checkpoint"]
        lines.extend(["", f"Matching checkpoint: `{checkpoint['path']}`", f"Checkpoint manifest: `{checkpoint['manifest']}`"])
    elif result["status"] != "complete":
        lines.extend(["", "No checkpoint trained on this exact split was found; the comparison cannot yet be completed."])
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(project_dir: Path = PROJECT_DIR) -> dict[str, Any]:
    started = time.perf_counter()
    snapshot = find_snapshot(project_dir / SNAPSHOT_ROOT)
    image_dir = project_dir / IMAGE_DIR
    train_dataset = WaveDataset(snapshot.train_csv, image_dir, build_evaluation_transform())
    validation_dataset = WaveDataset(snapshot.validation_csv, image_dir, build_evaluation_transform())
    if len(train_dataset) != EXPECTED_COUNTS["train"] or len(validation_dataset) != EXPECTED_COUNTS["validation"]:
        raise RuntimeError("Snapshot dataset counts changed while loading labels")

    result = _base_result(snapshot, train_dataset, validation_dataset)
    checkpoint = select_checkpoint(snapshot, project_dir / CHECKPOINT_ROOT, project_dir)
    if checkpoint is None:
        result["status"] = "incomplete_no_matching_checkpoint"
        write_reports(result, project_dir / JSON_OUTPUT, project_dir / MARKDOWN_OUTPUT)
        print("No checkpoint trained on the selected split; comparison cannot yet be completed.")
        return result

    device = select_device()
    print(f"Selected snapshot: {snapshot.directory}")
    print(f"Selected checkpoint: {checkpoint.path}")
    print(f"Evaluation device: {device}")
    neural_mae, inference_seconds = evaluate_checkpoint(checkpoint, validation_dataset, device)
    result["checkpoint"] = {
        "path": str(checkpoint.path),
        "manifest": str(checkpoint.manifest_path),
        "run_version": checkpoint.metadata.get("run_version"),
        "recorded_best_validation_mae": checkpoint.metadata.get("best_validation_mae"),
    }
    result["neural_network"] = {"validation_mae": neural_mae}
    result["predictors"]["neural_network"] = {
        "prediction": None,
        "validation_mae": neural_mae,
        "relative_mae_reduction_percent": None,
    }
    for baseline_name in ("training_mean", "training_median"):
        result["predictors"][baseline_name]["relative_mae_reduction_percent"] = relative_reduction(
            result["predictors"][baseline_name]["validation_mae"], neural_mae
        )
    result["device"] = str(device)
    result["inference_seconds"] = inference_seconds
    result["elapsed_seconds"] = time.perf_counter() - started
    write_reports(result, project_dir / JSON_OUTPUT, project_dir / MARKDOWN_OUTPUT)
    print(f"Experiment completed in {result['elapsed_seconds']:.2f}s")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=PROJECT_DIR)
    args = parser.parse_args()
    run(args.project_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
