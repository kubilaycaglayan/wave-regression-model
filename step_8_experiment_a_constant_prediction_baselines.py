"""Experiment A: compare constant waviness predictors with the neural network."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
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
REPORT_DIRECTORY = Path("step-8-experiment-a-baseline")
JSON_OUTPUT = REPORT_DIRECTORY / "baseline_experiment.json"
MARKDOWN_OUTPUT = REPORT_DIRECTORY / "baseline_experiment.md"
BATCH_SIZE = 8
REPORT_SCHEMA_VERSION = 2


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
            if not all(isinstance(files.get(name), dict) for name in ("train", "validation", "test")):
                continue
            valid = True
            for name in ("train", "validation", "test"):
                csv_path = directory / f"{name}.csv"
                entry = files[name]
                if not csv_path.is_file() or not isinstance(entry.get("count"), int):
                    valid = False
                    break
                if entry["count"] != sum(1 for _ in csv_path.open(encoding="utf-8")) - 1 or entry.get("sha256") != sha256(csv_path):
                    valid = False
                    break
            if not valid:
                continue
            candidates.append(Snapshot(directory.name, directory, manifest))
    if not candidates:
        raise FileNotFoundError(
            f"No valid immutable split snapshot found in {snapshot_root}"
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
    for manifest_path in sorted(checkpoint_root.rglob("training_manifest_*.json")):
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
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "split_snapshot": {
            "name": snapshot.name,
            "directory": str(snapshot.directory),
            "manifest": str(snapshot.directory / "manifest.json"),
            "counts": {
                "train": len(train_dataset),
                "validation": len(validation_dataset),
                "test": snapshot.manifest["files"]["test"]["count"],
            },
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


def _timestamp_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def versioned_report_paths(project_dir: Path) -> tuple[Path, Path]:
    """Choose a fresh immutable output directory for each Experiment A run."""
    root = project_dir / REPORT_DIRECTORY
    existing_versions = []
    if root.is_dir():
        for path in root.iterdir():
            match = re.fullmatch(r"v(\d+)-.+", path.name)
            if path.is_dir() and match:
                existing_versions.append(int(match.group(1)))
    version_dir = root / f"v{max(existing_versions, default=0) + 1}-{_timestamp_run_id()}"
    return version_dir / JSON_OUTPUT.name, version_dir / MARKDOWN_OUTPUT.name


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_record(result: dict[str, Any]) -> dict[str, Any] | None:
    checkpoint = result.get("checkpoint")
    if not isinstance(checkpoint, dict):
        return None
    record = dict(checkpoint)
    path = record.get("path")
    if isinstance(path, str):
        checkpoint_path = Path(path)
        if checkpoint_path.is_file():
            record["sha256"] = sha256(checkpoint_path)
    return record


def comparison_key(result: dict[str, Any]) -> str:
    snapshot = result.get("split_snapshot", {})
    checkpoint = result.get("checkpoint") or {}
    identity = {
        "experiment": "baseline_experiment",
        "split": {name: snapshot.get(name) for name in ("name", "train_sha256", "validation_sha256", "test_sha256")},
        "checkpoint": {name: checkpoint.get(name) for name in ("path", "run_version", "manifest", "sha256")},
    }
    return _canonical_sha256(identity)


def _new_run(result: dict[str, Any], version: int, run_id: str | None = None) -> dict[str, Any]:
    run = json.loads(json.dumps(result))
    run["schema_version"] = REPORT_SCHEMA_VERSION
    run["version"] = version
    run["run_id"] = run_id or _timestamp_run_id()
    run["checkpoint"] = _checkpoint_record(run)
    run["comparison_key"] = comparison_key(run)
    run.setdefault("test_evaluated", False)
    predictors = run.get("predictors", {})
    run.setdefault("validation_maes", {
        name: predictor.get("validation_mae") for name, predictor in predictors.items() if isinstance(predictor, dict)
    })
    run.setdefault("predictions", {
        name: predictors[name].get("prediction") for name in ("training_mean", "training_median") if isinstance(predictors.get(name), dict)
    })
    run.setdefault("neural_network_improvement_percentages", {
        name: predictors[name].get("relative_mae_reduction_percent") for name in ("training_mean", "training_median") if isinstance(predictors.get(name), dict)
    })
    return run


def _migrate_v1(value: dict[str, Any]) -> dict[str, Any]:
    migrated = _new_run(value, 1)
    return {"schema_version": REPORT_SCHEMA_VERSION, "latest_run_id": migrated["run_id"], "runs": [migrated]}


def load_report_history(json_path: Path) -> dict[str, Any]:
    if not json_path.exists():
        return {"schema_version": REPORT_SCHEMA_VERSION, "latest_run_id": None, "runs": []}
    try:
        value = load_json(json_path)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read baseline report {json_path}: {error}") from error
    if value.get("schema_version") == 1:
        return _migrate_v1(value)
    if value.get("schema_version") != REPORT_SCHEMA_VERSION or not isinstance(value.get("runs"), list):
        raise ValueError(f"Malformed or incompatible baseline report in {json_path}")
    for expected_version, run in enumerate(value["runs"], start=1):
        if not isinstance(run, dict) or run.get("version") != expected_version or not isinstance(run.get("run_id"), str):
            raise ValueError(f"Malformed or non-sequential run in baseline report {json_path}")
    if len({run["run_id"] for run in value["runs"]}) != len(value["runs"]):
        raise ValueError(f"Duplicate run_id in baseline report {json_path}")
    if value["runs"] and value.get("latest_run_id") != value["runs"][-1]["run_id"]:
        raise ValueError(f"Invalid latest_run_id in baseline report {json_path}")
    return value


def _format_metric(value: Any) -> str:
    return "n/a" if value is None else f"{value:.8f}"


def render_markdown(history: dict[str, Any]) -> str:
    lines = ["# Experiment A: Constant-Prediction Baselines", "", "JSON is the machine-readable source of truth. Newest runs appear first.", ""]
    for run in reversed(history["runs"]):
        snapshot = run["split_snapshot"]
        lines.extend([
            f"## Version {run['version']} — Run `{run['run_id']}`", "",
            f"Status: **{run['status']}**", f"Comparison key: `{run['comparison_key']}`",
            f"Split snapshot: `{snapshot['name']}` ({snapshot['counts']['train']} train, {snapshot['counts']['validation']} validation, {snapshot['counts']['test']} test)", "",
            "All reported MAEs use the same validation images.", "",
            "| Predictor | Constant/value | Validation MAE | Relative MAE reduction |", "|---|---:|---:|---:|",
        ])
        for name, label in (("training_mean", "Training mean"), ("training_median", "Training median"), ("neural_network", "Neural network")):
            predictor = run.get("predictors", {}).get(name)
            if not isinstance(predictor, dict):
                lines.append(f"| {label} | unavailable | unavailable | unavailable |")
                continue
            reduction = predictor.get("relative_mae_reduction_percent")
            reduction_text = "n/a" if reduction is None else f"{reduction:.2f}%"
            lines.append(f"| {label} | {_format_metric(predictor.get('prediction'))} | {_format_metric(predictor.get('validation_mae'))} | {reduction_text} |")
        improvements = run.get("neural_network_improvement_percentages", {})
        if improvements:
            lines.extend(["", "Neural-network improvement percentages: " + "; ".join(f"{name} {('n/a' if value is None else f'{value:.2f}%')}" for name, value in improvements.items())])
        checkpoint = run.get("checkpoint")
        if isinstance(checkpoint, dict):
            lines.extend(["", f"Matching checkpoint: `{checkpoint.get('path', 'n/a')}`", f"Checkpoint manifest: `{checkpoint.get('manifest', 'n/a')}`", f"Checkpoint SHA-256: `{checkpoint.get('sha256', 'unavailable')}`"])
        lines.extend(["", f"Test evaluated: **{'yes' if run.get('test_evaluated') else 'no'}**", f"Device: `{run.get('device', 'unavailable')}`; inference: `{_format_metric(run.get('inference_seconds'))}` s; elapsed: `{_format_metric(run.get('elapsed_seconds'))}` s", ""])
    return "\n".join(lines)


def write_reports(result: dict[str, Any], json_path: Path = JSON_OUTPUT, markdown_path: Path = MARKDOWN_OUTPUT) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    history = load_report_history(json_path)
    next_version = max((run["version"] for run in history["runs"]), default=0) + 1
    run = _new_run(result, next_version)
    existing_ids = {item["run_id"] for item in history["runs"]}
    collision_number = 2
    while run["run_id"] in existing_ids:
        run["run_id"] = f"{_timestamp_run_id()}-{collision_number}"
        collision_number += 1
    history["runs"].append(run)
    history["latest_run_id"] = run["run_id"]
    json_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(history), encoding="utf-8")


def run(project_dir: Path = PROJECT_DIR) -> dict[str, Any]:
    started = time.perf_counter()
    snapshot = find_snapshot(project_dir / SNAPSHOT_ROOT)
    image_dir = project_dir / IMAGE_DIR
    train_dataset = WaveDataset(snapshot.train_csv, image_dir, build_evaluation_transform())
    validation_dataset = WaveDataset(snapshot.validation_csv, image_dir, build_evaluation_transform())
    if len(train_dataset) != snapshot.manifest["files"]["train"]["count"] or len(validation_dataset) != snapshot.manifest["files"]["validation"]["count"]:
        raise RuntimeError("Snapshot dataset counts changed while loading labels")

    result = _base_result(snapshot, train_dataset, validation_dataset)
    json_output, markdown_output = versioned_report_paths(project_dir)
    checkpoint = select_checkpoint(snapshot, project_dir / CHECKPOINT_ROOT, project_dir)
    if checkpoint is None:
        result["status"] = "incomplete_no_matching_checkpoint"
        write_reports(result, json_output, markdown_output)
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
    write_reports(result, json_output, markdown_output)
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
