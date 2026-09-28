"""Experiment B: versioned per-image validation error analysis.

This module evaluates an existing checkpoint only.  It never trains a model
and intentionally never constructs a test dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from step_4_a_wave_dataset import WaveDataset, build_evaluation_transform
from step_5_a_wave_regression_model import WaveRegressionModel


PROJECT_DIR = Path(__file__).resolve().parent
# Leave unset to use the checkpoint from the newest training manifest. Set this
# to a relative or absolute checkpoint path to inspect a specific available run.
CHECKPOINT_OVERRIDE: Path | None = None
CHECKPOINT_ROOT = Path("step-5-checkpoints")
DEFAULT_OUTPUT_ROOT = Path("step-9-experiment-b-per-image-error-analysis")
IMAGE_DIR = Path("step-2-final-water-data")
BATCH_SIZE = 8
SCHEMA_VERSION = 1
BIN_RANGES = ((0.00, 0.19), (0.20, 0.39), (0.40, 0.59), (0.60, 0.79), (0.80, 1.00))


@dataclass(frozen=True)
class RunConfig:
    project_dir: Path
    checkpoint: Path
    split_snapshot: Path
    output_dir: Path
    model_version: str
    checkpoint_sha256: str
    split_manifest_sha256: str
    test_manifest_sha256: str
    split_counts: dict[str, int]


def sha256_for(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def resolve_path(value: str | Path, project_dir: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else project_dir / path).resolve()


def model_version_from_checkpoint(checkpoint: Path) -> str:
    match = re.search(r"(?:^|-)v(\d+)(?:-|$)", checkpoint.stem)
    if match is None:
        raise ValueError(f"Cannot derive model version from checkpoint filename: {checkpoint.name}")
    return f"v{match.group(1)}"


def checkpoint_manifest(checkpoint: Path, project_dir: Path) -> tuple[Path, dict[str, Any]]:
    candidates: list[tuple[Path, dict[str, Any]]] = []
    checkpoint_root = CHECKPOINT_ROOT if CHECKPOINT_ROOT.is_absolute() else project_dir / CHECKPOINT_ROOT
    for path in sorted(checkpoint_root.rglob("training_manifest_*.json")):
        metadata = load_json(path)
        reference = metadata.get("best_checkpoint_path")
        if isinstance(reference, str) and resolve_path(reference, project_dir) == checkpoint:
            candidates.append((path, metadata))
    if not candidates:
        raise FileNotFoundError(f"No training manifest references checkpoint: {checkpoint}")
    return candidates[-1]


def checkpoint_from_latest_manifest(
    checkpoint_root: Path = CHECKPOINT_ROOT,
    project_dir: Path = PROJECT_DIR,
) -> tuple[Path, Path]:
    """Return the checkpoint recorded by the newest training manifest."""
    manifests = sorted(checkpoint_root.rglob("training_manifest_*.json"), key=lambda path: path.name)
    if not manifests:
        raise FileNotFoundError(f"No training manifests found in {checkpoint_root}")
    manifest_path = manifests[-1]
    metadata = load_json(manifest_path)
    reference = metadata.get("best_checkpoint_path")
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError(f"Training manifest has no best_checkpoint_path: {manifest_path}")
    checkpoint = resolve_path(reference, project_dir)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Training manifest {manifest_path} references missing checkpoint: {checkpoint}"
        )
    return checkpoint, manifest_path


def select_checkpoint(
    project_dir: Path = PROJECT_DIR,
    configured_checkpoint: Path | None = None,
    checkpoint_root: Path = CHECKPOINT_ROOT,
) -> Path:
    """Select an explicit checkpoint or the checkpoint from the newest manifest."""
    if configured_checkpoint is None:
        configured_checkpoint = CHECKPOINT_OVERRIDE
    if configured_checkpoint is not None:
        return resolve_path(configured_checkpoint, project_dir)
    root = checkpoint_root if checkpoint_root.is_absolute() else project_dir / checkpoint_root
    checkpoint, _ = checkpoint_from_latest_manifest(root, project_dir)
    return checkpoint


def resolve_split_snapshot(
    checkpoint: Path, requested: Path | None, project_dir: Path
) -> tuple[Path, Path, dict[str, Any]]:
    manifest_path, metadata = checkpoint_manifest(checkpoint, project_dir)
    reference = metadata.get("split_snapshot")
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError(f"Training manifest has no split_snapshot: {manifest_path}")
    manifest_split = resolve_path(reference, project_dir)
    split = resolve_path(requested, project_dir) if requested is not None else manifest_split
    if split != manifest_split:
        raise ValueError(
            f"Checkpoint manifest split does not match requested snapshot: {manifest_split} != {split}"
        )
    return split, manifest_path, metadata


def output_directory(
    output_root: Path, model_version: str, checkpoint: Path, split_snapshot: Path, project_dir: Path
) -> Path:
    checkpoint_id = f"{checkpoint.stem}--sha256-{sha256_for(checkpoint)[:12]}"
    split_manifest = split_snapshot / "manifest.json"
    snapshot_id = f"{split_snapshot.name}--sha256-{sha256_for(split_manifest)[:12]}"
    return output_root / model_version / checkpoint_id / snapshot_id


def next_output_version(base_directory: Path) -> Path:
    """Allocate a new result directory so repeated analyses remain available."""
    existing: list[int] = []
    if base_directory.is_dir():
        for path in base_directory.iterdir():
            match = re.fullmatch(r"v(\d+)-.+", path.name)
            if path.is_dir() and match:
                existing.append(int(match.group(1)))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return base_directory / f"v{max(existing, default=0) + 1}-{stamp}"


def validate_snapshot(split_snapshot: Path) -> tuple[dict[str, Any], dict[str, str], dict[str, int]]:
    manifest_path = split_snapshot / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
    metadata = load_json(manifest_path)
    files = metadata.get("files")
    if not isinstance(files, dict):
        raise ValueError(f"Split manifest has no files object: {manifest_path}")
    hashes: dict[str, str] = {}
    counts: dict[str, int] = {}
    for split_name in ("train", "validation", "test"):
        csv_path = split_snapshot / f"{split_name}.csv"
        entry = files.get(split_name, {})
        expected_count = entry.get("count")
        if not csv_path.is_file() or not isinstance(expected_count, int) or expected_count < 1:
            raise ValueError(f"Invalid {split_name} manifest/count in {split_snapshot}")
        expected_hash = entry.get("sha256")
        actual_hash = sha256_for(csv_path)
        if expected_hash != actual_hash:
            raise ValueError(f"{split_name}.csv hash does not match snapshot manifest")
        actual_count = sum(1 for _ in csv_path.open(encoding="utf-8")) - 1
        if actual_count != expected_count:
            raise ValueError(
                f"{split_name}.csv row count does not match snapshot manifest: "
                f"{actual_count} != {expected_count}"
            )
        hashes[split_name] = actual_hash
        counts[split_name] = expected_count
    return metadata, hashes, counts


def prepare_config(
    project_dir: Path = PROJECT_DIR,
    checkpoint: Path | None = None,
    split_snapshot: Path | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> RunConfig:
    checkpoint_path = select_checkpoint(project_dir, checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Selected checkpoint not found: {checkpoint_path}")
    split_path, _, _ = resolve_split_snapshot(checkpoint_path, split_snapshot, project_dir)
    snapshot_metadata, hashes, split_counts = validate_snapshot(split_path)
    if snapshot_metadata.get("snapshot") != split_path.name:
        raise ValueError(f"Snapshot metadata name does not match directory: {split_path}")
    version = model_version_from_checkpoint(checkpoint_path)
    return RunConfig(
        project_dir=project_dir,
        checkpoint=checkpoint_path,
        split_snapshot=split_path,
        output_dir=next_output_version(output_directory(
            resolve_path(output_root, project_dir), version, checkpoint_path, split_path, project_dir
        )),
        model_version=version,
        checkpoint_sha256=sha256_for(checkpoint_path),
        split_manifest_sha256=hashes["validation"],
        test_manifest_sha256=hashes["test"],
        split_counts=split_counts,
    )


def read_labels(path: Path) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["filename", "waviness"]:
            raise ValueError(f"{path} must have header: filename,waviness")
        for row_number, row in enumerate(reader, start=2):
            filename = (row.get("filename") or "").strip()
            try:
                value = float(row["waviness"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path}:{row_number}: invalid waviness") from error
            if not filename or not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{path}:{row_number}: invalid filename or waviness")
            rows.append((filename, value))
    if not rows:
        raise ValueError(f"No labels found in {path}")
    if len({filename for filename, _ in rows}) != len(rows):
        raise ValueError(f"Duplicate filenames found in {path}")
    return rows


def regression_metrics(actual: list[float], predicted: list[float]) -> dict[str, float]:
    if not actual or len(actual) != len(predicted):
        raise ValueError("Metrics require equally sized, non-empty lists")
    signed = [prediction - target for target, prediction in zip(actual, predicted)]
    absolute = [abs(value) for value in signed]
    return {
        "mae": statistics.mean(absolute),
        "rmse": math.sqrt(statistics.mean(value * value for value in signed)),
        "mean_signed_error": statistics.mean(signed),
        "median_absolute_error": statistics.median(absolute),
        "maximum_absolute_error": max(absolute),
    }


def waviness_bin(value: float) -> str:
    for low, high in BIN_RANGES:
        if low <= value <= high:
            return f"{low:.2f}-{high:.2f}"
    raise ValueError(f"Waviness value does not fit a supported bin: {value}")


def bin_metrics(actual: list[float], predicted: list[float]) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {
        f"{low:.2f}-{high:.2f}": {"count": 0, "mae": None} for low, high in BIN_RANGES
    }
    grouped: dict[str, list[float]] = {key: [] for key in result}
    for target, prediction in zip(actual, predicted):
        key = waviness_bin(target)
        grouped[key].append(abs(prediction - target))
    for key, errors in grouped.items():
        result[key] = {"count": len(errors), "mae": statistics.mean(errors) if errors else None}
    return result


def endpoint_bias(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    groups = {
        "actual_0.00-0.39": [row for row in rows if row["actual_waviness"] <= 0.39],
        "actual_0.60-1.00": [row for row in rows if row["actual_waviness"] >= 0.60],
    }
    return {
        name: {
            "count": len(group),
            "mean_signed_error": statistics.mean(row["signed_error"] for row in group) if group else None,
        }
        for name, group in groups.items()
    }


def make_rows(
    samples: Iterable[tuple[str, float]], predictions: Iterable[float], baseline: float
) -> list[dict[str, Any]]:
    rows = []
    for filename, actual, prediction in (
        (filename, actual, prediction)
        for (filename, actual), prediction in zip(samples, predictions)
    ):
        signed_error = prediction - actual
        model_abs = abs(signed_error)
        baseline_abs = abs(baseline - actual)
        rows.append(
            {
                "filename": filename,
                "actual_waviness": actual,
                "predicted_waviness": prediction,
                "signed_error": signed_error,
                "absolute_error": model_abs,
                "mean_baseline_prediction": baseline,
                "mean_baseline_absolute_error": baseline_abs,
                "neural_network_beats_baseline": model_abs < baseline_abs,
            }
        )
    if not rows:
        raise ValueError("No prediction rows were created")
    return sorted(rows, key=lambda row: row["absolute_error"], reverse=True)


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


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: f"{value:.8f}" if isinstance(value, float) else value for key, value in row.items()})


def plot_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    actual = np.array([row["actual_waviness"] for row in rows])
    predicted = np.array([row["predicted_waviness"] for row in rows])
    signed = predicted - actual

    figure, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(actual, predicted, color="#1769aa", edgecolor="white", linewidth=0.7)
    axis.plot([0, 1], [0, 1], "--", color="#555", label="Perfect prediction")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Actual waviness", ylabel="Predicted waviness", title="Actual vs predicted waviness")
    axis.grid(alpha=0.25); axis.legend(); figure.tight_layout(); figure.savefig(output_dir / "actual_vs_predicted.png", dpi=160); plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(actual, signed, color="#9c2c77", edgecolor="white", linewidth=0.7)
    axis.axhline(0, color="#555", linestyle="--")
    axis.set(xlim=(0, 1), xlabel="Actual waviness", ylabel="Signed error (prediction − actual)", title="Validation residuals")
    axis.grid(alpha=0.25); figure.tight_layout(); figure.savefig(output_dir / "residuals.png", dpi=160); plt.close(figure)

    ordered = sorted(rows, key=lambda row: row["absolute_error"])
    labels = [row["filename"].replace("step-2_", "") for row in ordered]
    x = np.arange(len(labels)); width = 0.38
    figure, axis = plt.subplots(figsize=(11, 6))
    axis.bar(x - width / 2, [row["absolute_error"] for row in ordered], width, label="Neural network")
    axis.bar(x + width / 2, [row["mean_baseline_absolute_error"] for row in ordered], width, label="Mean baseline")
    axis.set_xticks(x, labels, rotation=60, ha="right"); axis.set_ylabel("Absolute error"); axis.set_title("Per-image error comparison")
    axis.grid(axis="y", alpha=0.25); axis.legend(); figure.tight_layout(); figure.savefig(output_dir / "error_comparison.png", dpi=160); plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0, 1, 11)
    axis.hist(actual, bins=bins, alpha=0.65, label="Actual", color="#555")
    axis.hist(predicted, bins=bins, alpha=0.65, label="Predicted", color="#e07a2d")
    axis.set(xlim=(0, 1), xlabel="Waviness", ylabel="Image count", title="Actual and predicted distributions")
    axis.grid(axis="y", alpha=0.25); axis.legend(); figure.tight_layout(); figure.savefig(output_dir / "prediction_distribution.png", dpi=160); plt.close(figure)


def write_gallery(rows: list[dict[str, Any]], output_dir: Path, image_dir: Path) -> None:
    image_root = Path(__import__("os").path.relpath(image_dir, output_dir)).as_posix()
    cards = []
    for row in rows:
        filename = str(row["filename"])
        class_name = " worse-than-baseline" if not row["neural_network_beats_baseline"] else ""
        image_url = f"{image_root}/{quote(filename)}"
        cards.append(
            f'''<article class="card{class_name}"><a href="{image_url}" target="_blank" rel="noopener"><img src="{image_url}" alt="{html.escape(filename)}" loading="lazy"></a><div class="details"><h2>{html.escape(filename)}</h2><p>Actual: {row["actual_waviness"]:.4f}</p><p>Predicted: {row["predicted_waviness"]:.4f}</p><p>Absolute error: {row["absolute_error"]:.4f}</p><p>Mean-baseline absolute error: {row["mean_baseline_absolute_error"]:.4f}</p><p class="status">{"Beats baseline" if row["neural_network_beats_baseline"] else "Worse than baseline"}</p></div></article>'''
        )
    (output_dir / "error_gallery.html").write_text(
        """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Experiment B error gallery</title><style>:root{font-family:system-ui,sans-serif;color-scheme:light dark}body{max-width:1280px;margin:auto;padding:1rem}.intro{opacity:.8}.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:1rem}.card{border:2px solid #4b9;border-radius:.6rem;overflow:hidden;background:#8881}.card.worse-than-baseline{border-color:#d44;background:#d442}.card a{display:block;background:#111}.card img{display:block;width:100%;aspect-ratio:1;object-fit:contain}.details{padding:.8rem}.details h2{font-size:1rem;overflow-wrap:anywhere}.details p{margin:.25rem 0;font-variant-numeric:tabular-nums}.status{font-weight:700}</style></head><body><h1>Experiment B validation error gallery</h1><p class="intro">Sorted by neural-network absolute error, worst first. Red cards are worse than the mean baseline. Visual explanations require manual inspection.</p><main class="gallery">"""
        + "\n".join(cards)
        + "</main></body></html>\n",
        encoding="utf-8",
    )


def write_report(rows: list[dict[str, Any]], metrics: dict[str, Any], config: RunConfig, path: Path) -> None:
    high = rows[:3]
    low = sorted(rows, key=lambda row: row["absolute_error"])[:3]
    lines = [
        "# Experiment B — Per-Image Validation Error Analysis", "",
        "This is an evaluation-only analysis. No training was performed and the test dataset was not evaluated.", "",
        f"- Model version: `{config.model_version}`",
        f"- Checkpoint: `{config.checkpoint}`",
        f"- Split snapshot: `{config.split_snapshot.name}`",
        f"- Validation images: **{metrics['sample_count']}**", "",
        "## Aggregate findings", "",
        f"- Neural-network MAE: **{metrics['mae']:.8f}**; RMSE: **{metrics['rmse']:.8f}**.",
        f"- Mean signed error: **{metrics['mean_signed_error']:.8f}**; positive values indicate overestimation.",
        f"- Median absolute error: **{metrics['median_absolute_error']:.8f}**; maximum absolute error: **{metrics['maximum_absolute_error']:.8f}**.",
        f"- The neural network beats the mean baseline on **{metrics['baseline_comparison']['count_beaten']} / {metrics['sample_count']} ({metrics['baseline_comparison']['percentage_beaten']:.2f}%)** images.",
        f"- For actual labels 0.00–0.39 (n={metrics['endpoint_bias']['actual_0.00-0.39']['count']}), mean signed error is **{metrics['endpoint_bias']['actual_0.00-0.39']['mean_signed_error']:.8f}**; for 0.60–1.00 (n={metrics['endpoint_bias']['actual_0.60-1.00']['count']}), it is **{metrics['endpoint_bias']['actual_0.60-1.00']['mean_signed_error']:.8f}**.",
        f"- Predictions span {metrics['actual_prediction_ranges']['prediction_min']:.4f}–{metrics['actual_prediction_ranges']['prediction_max']:.4f}, while actual labels span {metrics['actual_prediction_ranges']['actual_min']:.4f}–{metrics['actual_prediction_ranges']['actual_max']:.4f}; prediction/actual range width ratio: **{metrics['actual_prediction_ranges']['prediction_to_actual_width_ratio']:.3f}**.",
        "", "## Waviness ranges", "", "| Actual range | Count | MAE |", "|---|---:|---:|",
    ]
    for key, value in metrics["range_metrics"].items():
        mae = "n/a" if value["mae"] is None else f"{value['mae']:.8f}"
        lines.append(f"| {key} | {value['count']} | {mae} |")
    lines.extend(["", "## Error ranking", "", "Largest errors:"])
    lines.extend(f"- `{row['filename']}`: actual {row['actual_waviness']:.2f}, predicted {row['predicted_waviness']:.4f}, absolute error {row['absolute_error']:.4f}." for row in high)
    lines.append("\nSmallest errors:")
    lines.extend(f"- `{row['filename']}`: actual {row['actual_waviness']:.2f}, predicted {row['predicted_waviness']:.4f}, absolute error {row['absolute_error']:.4f}." for row in low)
    lines.extend([
        "", "## Evidence and hypotheses", "",
        "The signed-error distribution and actual-vs-predicted plot are evidence for assessing systematic underestimation or overestimation. Compare the endpoint labels and residuals in the plots before making claims about regression toward the mean.",
        "",
        "The range-specific conclusions are limited by the sample counts above; sparse bins should not support strong conclusions.",
        "",
        "The HTML gallery is the source for manual inspection of lighting, reflections, foam, wave patterns, and framing. This report does not automatically assign those visual conditions as causes of error.",
        "",
        "## Next experiment recommendation", "",
        "Use the largest-error gallery cases and any repeated, manually confirmed visual condition to define the next controlled preprocessing or data-collection experiment. Keep the split fixed and change one factor at a time.", "",
        "## Reproducibility", "",
        f"- Checkpoint SHA-256: `{config.checkpoint_sha256}`",
        f"- Validation manifest SHA-256: `{config.split_manifest_sha256}`",
        f"- Test manifest SHA-256 recorded without evaluation: `{config.test_manifest_sha256}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def complete_outputs(output_dir: Path) -> bool:
    return all((output_dir / name).is_file() for name in (
        "predictions.csv", "predictions.json", "metrics.json", "actual_vs_predicted.png", "residuals.png",
        "error_comparison.png", "prediction_distribution.png", "error_gallery.html", "experiment_report.md",
    ))


def run(config: RunConfig) -> dict[str, Any]:
    started = time.perf_counter()
    if config.output_dir.exists() and not complete_outputs(config.output_dir):
        raise RuntimeError(f"Refusing to overwrite incomplete output directory: {config.output_dir}")
    if complete_outputs(config.output_dir):
        print(f"Complete Experiment B output already exists: {config.output_dir}")
        return load_json(config.output_dir / "metrics.json")

    validation_csv = config.split_snapshot / "validation.csv"
    train_csv = config.split_snapshot / "train.csv"
    test_hash_before = sha256_for(config.split_snapshot / "test.csv")
    train_samples = read_labels(train_csv)
    validation_dataset = WaveDataset(validation_csv, config.project_dir / IMAGE_DIR, build_evaluation_transform())
    if len(train_samples) != config.split_counts["train"] or len(validation_dataset) != config.split_counts["validation"]:
        raise RuntimeError("Snapshot dataset counts changed while loading validation labels")
    baseline = statistics.mean(value for _, value in train_samples)
    validation_samples = list(validation_dataset.samples)
    device = select_device()
    print(f"Selected checkpoint: {config.checkpoint}")
    print(f"Selected validation snapshot: {config.split_snapshot}")
    print(f"Evaluation device: {device}")
    load_started = time.perf_counter()
    model = WaveRegressionModel.load_from_checkpoint(config.checkpoint, map_location=device, pretrained=False)
    model.to(device); model.eval()
    print(f"Checkpoint loaded in {time.perf_counter() - load_started:.2f}s")

    predictions: list[float] = []
    loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False)
    inference_started = time.perf_counter()
    with torch.inference_mode():
        for batch_number, (images, _) in enumerate(loader, start=1):
            item_started = time.perf_counter()
            batch_predictions = model(images.to(device)).detach().cpu()
            if not torch.isfinite(batch_predictions).all() or not torch.all((batch_predictions >= 0) & (batch_predictions <= 1)):
                raise RuntimeError(f"Invalid prediction in validation batch {batch_number}")
            predictions.extend(float(value) for value in batch_predictions)
            print(f"Validation batch {batch_number}: {len(batch_predictions)} images in {time.perf_counter() - item_started:.3f}s")
    if len(predictions) != config.split_counts["validation"]:
        raise RuntimeError(
            f"Expected {config.split_counts['validation']} validation predictions, got {len(predictions)}"
        )
    rows = make_rows(validation_samples, predictions, baseline)
    actual = [value for _, value in validation_samples]
    aggregate = regression_metrics(actual, predictions)
    metrics: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "sample_count": len(rows), "baseline_prediction": baseline, "baseline_mae": statistics.mean(row["mean_baseline_absolute_error"] for row in rows),
        **aggregate, "range_metrics": bin_metrics(actual, predictions),
        "endpoint_bias": endpoint_bias(rows),
        "actual_prediction_ranges": {
            "actual_min": min(actual), "actual_max": max(actual),
            "prediction_min": min(predictions), "prediction_max": max(predictions),
            "actual_width": max(actual) - min(actual), "prediction_width": max(predictions) - min(predictions),
            "prediction_to_actual_width_ratio": (max(predictions) - min(predictions)) / (max(actual) - min(actual)),
        },
        "baseline_comparison": {"count_beaten": sum(row["neural_network_beats_baseline"] for row in rows), "percentage_beaten": sum(row["neural_network_beats_baseline"] for row in rows) / len(rows) * 100, "ties": sum(row["absolute_error"] == row["mean_baseline_absolute_error"] for row in rows)},
        "checkpoint": {"path": str(config.checkpoint), "sha256": config.checkpoint_sha256, "model_version": config.model_version},
        "split_snapshot": {"path": str(config.split_snapshot), "validation_sha256": config.split_manifest_sha256, "test_sha256": config.test_manifest_sha256, "counts": config.split_counts},
        "test_evaluated": False, "training_performed": False, "device": str(device),
        "inference_seconds": time.perf_counter() - inference_started,
    }
    if test_hash_before != sha256_for(config.split_snapshot / "test.csv"):
        raise RuntimeError("Test manifest changed during evaluation")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, config.output_dir / "predictions.csv")
    (config.output_dir / "predictions.json").write_text(json.dumps({"schema_version": SCHEMA_VERSION, "rows": rows}, indent=2) + "\n", encoding="utf-8")
    plot_outputs(rows, config.output_dir)
    write_gallery(rows, config.output_dir, config.project_dir / IMAGE_DIR)
    metrics["elapsed_seconds"] = time.perf_counter() - started
    (config.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_report(rows, metrics, config, config.output_dir / "experiment_report.md")
    print(f"Experiment B completed in {metrics['elapsed_seconds']:.2f}s")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=PROJECT_DIR)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="checkpoint to evaluate; defaults to the newest training manifest",
    )
    parser.add_argument(
        "--split-snapshot",
        type=Path,
        default=None,
        help="split snapshot to evaluate; defaults to the snapshot recorded by the checkpoint manifest",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    project_dir = args.project_dir.resolve()
    config = prepare_config(project_dir, args.checkpoint, args.split_snapshot, args.output_root)
    print(f"Output directory: {config.output_dir}")
    run(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
