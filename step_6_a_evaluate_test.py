"""Step 6: evaluate the validation-selected model once on the untouched test split."""

from __future__ import annotations

import csv
import argparse
import hashlib
import html
import json
import math
import os
import re
import time
from pathlib import Path
from statistics import mean, median
from urllib.parse import quote

import torch
from torch.utils.data import DataLoader

from step_4_a_wave_dataset import WaveDataset, build_evaluation_transform
from step_5_a_wave_regression_model import WaveRegressionModel


PROJECT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJECT_DIR / "step-5-checkpoints"
BENCHMARK_TEST_MANIFEST = Path("step-3-dataset-splits/benchmarks/benchmark-v1-test.csv")


def _dotenv_value(key: str) -> str | None:
    dotenv_path = PROJECT_DIR / ".env"
    if not dotenv_path.is_file():
        return None
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        candidate, separator, value = line.partition("=")
        if separator and candidate.strip() == key:
            return value.strip().strip("\"'")
    return None


def _resolve_project_path(value: str, base_dir: Path = PROJECT_DIR) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base_dir / path


def sha256_for(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_from_manifest(manifest_path: Path) -> Path:
    try:
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read training manifest {manifest_path}: {error}") from error
    checkpoint_reference = metadata.get("best_checkpoint_path")
    if not isinstance(checkpoint_reference, str) or not checkpoint_reference.strip():
        raise RuntimeError(f"Training manifest has no best_checkpoint_path: {manifest_path}")
    checkpoint_path = _resolve_project_path(checkpoint_reference)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Training manifest {manifest_path} references missing checkpoint: {checkpoint_path}"
        )
    return checkpoint_path


def checkpoint_from_latest_manifest(
    checkpoint_dir: Path = CHECKPOINT_DIR,
) -> tuple[Path, Path]:
    manifests = sorted(checkpoint_dir.glob("training_manifest_*.json"))
    if not manifests:
        raise FileNotFoundError(f"No training manifests found in {checkpoint_dir}")
    manifest_path = manifests[-1]
    return _checkpoint_from_manifest(manifest_path), manifest_path


def checkpoint_path_from_env() -> tuple[Path, Path | None]:
    """Select an explicit checkpoint or the newest training manifest checkpoint."""
    configured_path = os.getenv("EVALUATION_CHECKPOINT_PATH")
    if configured_path is None:
        configured_path = _dotenv_value("EVALUATION_CHECKPOINT_PATH")

    if configured_path:
        return _resolve_project_path(configured_path), None
    return checkpoint_from_latest_manifest()


CHECKPOINT_PATH, CHECKPOINT_MANIFEST_PATH = checkpoint_path_from_env()
CHECKPOINT_SHA256 = sha256_for(CHECKPOINT_PATH) if CHECKPOINT_PATH.is_file() else None
_MODEL_NAME_MATCH = re.fullmatch(r"(.+?)-best-val-mae-epoch=\d+-val_mae=[\d.]+\.ckpt", CHECKPOINT_PATH.name)
if _MODEL_NAME_MATCH is None:
    raise ValueError(f"Cannot derive model name from checkpoint filename: {CHECKPOINT_PATH.name}")
MODEL_NAME = _MODEL_NAME_MATCH.group(1)
_MODEL_VERSION_MATCH = re.search(r"(?:^|-)v(\d+)(?:-|$)", MODEL_NAME)
if _MODEL_VERSION_MATCH is None:
    raise ValueError(f"Checkpoint filename does not contain a model version: {CHECKPOINT_PATH.name}")
MODEL_VERSION = f"v{_MODEL_VERSION_MATCH.group(1)}"
IMAGE_DIR = Path("step-2-final-water-data")
TEST_CSV = BENCHMARK_TEST_MANIFEST
TRAIN_CSV = Path("step-3-dataset-splits/train.csv")
DISCARDED_PATH = Path("discarded_images.csv")


def evaluation_output_dir(checkpoint_path: Path, test_manifest: Path) -> Path:
    """Return a collision-resistant directory for one checkpoint/manifest pair."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Selected checkpoint not found: {checkpoint_path}")
    if not test_manifest.is_file():
        raise FileNotFoundError(f"Test manifest not found: {test_manifest}")
    checkpoint_id = f"{checkpoint_path.stem}--sha256-{sha256_for(checkpoint_path)[:12]}"
    manifest_id = f"{test_manifest.stem}--sha256-{sha256_for(test_manifest)[:12]}"
    return Path("step-6-test-evaluation") / checkpoint_id / manifest_id


OUTPUT_DIR = evaluation_output_dir(CHECKPOINT_PATH, TEST_CSV)
PREDICTIONS_CSV = OUTPUT_DIR / "test_predictions.csv"
SUMMARY_PATH = OUTPUT_DIR / "summary.txt"
GALLERY_PATH = OUTPUT_DIR / "index.html"
METADATA_PATH = OUTPUT_DIR / "evaluation_metadata.json"
BATCH_SIZE = 1

def read_training_labels(path: Path) -> tuple[list[str], list[float]]:
    """Read filenames and labels needed for train-only constant baselines."""
    if not path.is_file():
        raise FileNotFoundError(f"Training manifest not found: {path}")

    filenames: list[str] = []
    labels: list[float] = []
    discarded: set[str] = set()
    if DISCARDED_PATH.is_file():
        with DISCARDED_PATH.open(newline="", encoding="utf-8") as discarded_handle:
            discarded = {
                (row.get("filename") or "").strip()
                for row in csv.DictReader(discarded_handle)
                if (row.get("filename") or "").strip()
            }
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["filename", "waviness"]:
            raise ValueError(f"{path} must have header: filename,waviness")
        for row_number, row in enumerate(reader, start=2):
            filename = (row.get("filename") or "").strip()
            if filename in discarded:
                continue
            try:
                label = float(row["waviness"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path}:{row_number}: invalid waviness") from error
            if not filename:
                raise ValueError(f"{path}:{row_number}: empty filename")
            if not math.isfinite(label) or not 0.0 <= label <= 1.0:
                raise ValueError(f"{path}:{row_number}: waviness outside [0, 1]")
            filenames.append(filename)
            labels.append(label)

    if not labels:
        raise ValueError(f"No training labels found in {path}")
    if len(filenames) != len(set(filenames)):
        raise ValueError(f"Duplicate filenames found in {path}")
    return filenames, labels


def regression_metrics(targets: list[float], predictions: list[float]) -> tuple[float, float]:
    if len(targets) != len(predictions) or not targets:
        raise ValueError("Metrics require equally sized, non-empty target and prediction lists")
    absolute_errors = [abs(target - prediction) for target, prediction in zip(targets, predictions)]
    squared_errors = [(target - prediction) ** 2 for target, prediction in zip(targets, predictions)]
    return mean(absolute_errors), math.sqrt(mean(squared_errors))


def select_evaluation_device() -> torch.device:
    """Use CUDA only after verifying that this installation can run a convolution."""
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


def write_predictions(rows: list[dict[str, float | str]]) -> None:
    with PREDICTIONS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "filename",
                "true_waviness",
                "predicted_waviness",
                "absolute_error",
                "squared_error",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "filename": row["filename"],
                    "true_waviness": f"{row['true_waviness']:.8f}",
                    "predicted_waviness": f"{row['predicted_waviness']:.8f}",
                    "absolute_error": f"{row['absolute_error']:.8f}",
                    "squared_error": f"{row['squared_error']:.8f}",
                }
            )


def write_gallery(rows: list[dict[str, float | str]]) -> None:
    cards = []
    for row in rows:
        filename = str(row["filename"])
        image_url = "../step-2-final-water-data/" + quote(filename)
        cards.append(
            f"""
      <article class="card">
        <a href="{image_url}" target="_blank" rel="noopener">
          <img src="{image_url}" alt="{html.escape(filename)}" loading="lazy">
        </a>
        <div class="details">
          <h2>{html.escape(filename)}</h2>
          <p>true: {row['true_waviness']:.2f}</p>
          <p>predicted: {row['predicted_waviness']:.2f}</p>
          <p>absolute error: {row['absolute_error']:.2f}</p>
        </div>
      </article>"""
        )

    GALLERY_PATH.write_text(
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Step 6 test evaluation</title>
  <style>
    :root { color-scheme: light dark; font-family: system-ui, sans-serif; }
    body { max-width: 1200px; margin: 0 auto; padding: 1.25rem; }
    h1 { margin-bottom: .25rem; }
    .intro { margin-top: 0; opacity: .75; }
    .gallery { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1rem; }
    .card { border: 1px solid #8886; border-radius: .65rem; overflow: hidden; background: #8881; }
    .card a { display: block; background: #111; }
    .card img { display: block; width: 100%; aspect-ratio: 1; object-fit: contain; }
    .details { padding: .8rem 1rem 1rem; }
    .details h2 { margin: 0 0 .55rem; font-size: 1rem; overflow-wrap: anywhere; }
    .details p { margin: .2rem 0; font-variant-numeric: tabular-nums; }
  </style>
</head>
<body>
  <h1>Untouched test-set predictions</h1>
  <p class="intro">Sorted from highest to lowest absolute error. Click an image for full size.</p>
  <main class="gallery">
"""
        + "\n".join(cards)
        + """
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )


def print_results(rows: list[dict[str, float | str]]) -> None:
    print("\nPer-image test results (worst absolute error first)")
    print(f"{'filename':<26} {'true':>8} {'predicted':>11} {'abs_error':>11} {'squared_error':>14}")
    print("-" * 76)
    for row in rows:
        print(
            f"{row['filename']:<26} "
            f"{row['true_waviness']:>8.4f} "
            f"{row['predicted_waviness']:>11.4f} "
            f"{row['absolute_error']:>11.4f} "
            f"{row['squared_error']:>14.6f}"
        )


def main() -> None:
    global TEST_CSV, OUTPUT_DIR, PREDICTIONS_CSV, SUMMARY_PATH, GALLERY_PATH, METADATA_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-manifest",
        type=Path,
        default=TEST_CSV,
        help="test CSV to evaluate (default: immutable benchmark-v1)",
    )
    args = parser.parse_args()
    TEST_CSV = args.test_manifest
    OUTPUT_DIR = evaluation_output_dir(CHECKPOINT_PATH, TEST_CSV)
    PREDICTIONS_CSV = OUTPUT_DIR / "test_predictions.csv"
    SUMMARY_PATH = OUTPUT_DIR / "summary.txt"
    GALLERY_PATH = OUTPUT_DIR / "index.html"
    METADATA_PATH = OUTPUT_DIR / "evaluation_metadata.json"

    started = time.perf_counter()
    completed_outputs = (PREDICTIONS_CSV, SUMMARY_PATH, GALLERY_PATH, METADATA_PATH)
    existing_outputs = [path for path in completed_outputs if path.exists()]
    if len(existing_outputs) == len(completed_outputs):
        print("Step 6 outputs already exist; skipping repeat test inference.")
        print(SUMMARY_PATH.read_text(encoding="utf-8"), end="")
        print(f"Predictions CSV: {PREDICTIONS_CSV}")
        print(f"Inspection gallery: {GALLERY_PATH}")
        return
    if existing_outputs:
        raise RuntimeError(
            "Step 6 has incomplete existing outputs; refusing to overwrite a partial one-time evaluation: "
            + ", ".join(str(path) for path in existing_outputs)
        )

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Selected checkpoint not found: {CHECKPOINT_PATH}")
    if not TEST_CSV.is_file():
        raise FileNotFoundError(f"Test manifest not found: {TEST_CSV}")

    train_filenames, train_labels = read_training_labels(TRAIN_CSV)
    test_dataset = WaveDataset(TEST_CSV, IMAGE_DIR, build_evaluation_transform())
    test_filenames = tuple(filename for filename, _ in test_dataset.samples)
    if len(test_filenames) != len(set(test_filenames)):
        raise RuntimeError("Duplicate filenames found in the test manifest")
    overlap = set(train_filenames).intersection(test_filenames)
    if overlap:
        raise RuntimeError(f"Train/test filename overlap detected: {sorted(overlap)}")

    loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False)
    device = select_evaluation_device()
    print(f"Selected checkpoint: {CHECKPOINT_PATH.resolve()}")
    print(f"Untouched test samples verified: {len(test_dataset)}")
    print(f"Deterministic transform: {test_dataset.transform}")
    print(f"Evaluation device: {device}")

    load_started = time.perf_counter()
    model = WaveRegressionModel.load_from_checkpoint(CHECKPOINT_PATH, map_location=device)
    model.to(device)
    model.eval()
    print(f"Checkpoint loaded in {time.perf_counter() - load_started:.2f}s")

    predictions: list[float] = []
    targets: list[float] = []
    inference_started = time.perf_counter()
    with torch.no_grad():
        for sample_number, (images, batch_targets) in enumerate(loader, start=1):
            item_started = time.perf_counter()
            batch_predictions = model(images.to(device)).cpu()
            if not torch.isfinite(batch_predictions).all():
                raise RuntimeError(f"Non-finite prediction for {test_filenames[sample_number - 1]}")
            if not torch.all((batch_predictions >= 0.0) & (batch_predictions <= 1.0)):
                raise RuntimeError(f"Prediction outside [0, 1] for {test_filenames[sample_number - 1]}")
            predictions.extend(float(value) for value in batch_predictions)
            targets.extend(float(value) for value in batch_targets)
            print(
                f"Test sample {sample_number}/{len(test_dataset)}: "
                f"{test_filenames[sample_number - 1]} in {time.perf_counter() - item_started:.3f}s"
            )
    print(f"Inference completed in {time.perf_counter() - inference_started:.2f}s")

    if len(predictions) != len(test_dataset):
        raise RuntimeError(f"Expected {len(test_dataset)} predictions, got {len(predictions)}")

    rows: list[dict[str, float | str]] = []
    for (filename, _), target, prediction in zip(test_dataset.samples, targets, predictions):
        absolute_error = abs(target - prediction)
        rows.append(
            {
                "filename": filename,
                "true_waviness": target,
                "predicted_waviness": prediction,
                "absolute_error": absolute_error,
                "squared_error": absolute_error**2,
            }
        )
    rows.sort(key=lambda row: float(row["absolute_error"]), reverse=True)

    test_mae, test_rmse = regression_metrics(targets, predictions)
    training_mean = mean(train_labels)
    mean_mae, mean_rmse = regression_metrics(targets, [training_mean] * len(targets))
    training_median = median(train_labels)
    median_mae, median_rmse = regression_metrics(targets, [training_median] * len(targets))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_predictions(rows)
    write_gallery(rows)
    SUMMARY_PATH.write_text(
        "\n".join(
            [
                f"model name: {MODEL_NAME}",
                f"model version: {MODEL_VERSION}",
                f"selected checkpoint path: {CHECKPOINT_PATH.resolve()}",
                f"selected checkpoint sha256: {CHECKPOINT_SHA256 or sha256_for(CHECKPOINT_PATH)}",
                f"training manifest path: {CHECKPOINT_MANIFEST_PATH or 'explicit configuration'}",
                f"test manifest path: {TEST_CSV.resolve()}",
                f"test manifest sha256: {hashlib.sha256(TEST_CSV.read_bytes()).hexdigest()}",
                f"number of test samples: {len(test_dataset)}",
                f"test MAE: {test_mae:.8f}",
                f"test RMSE: {test_rmse:.8f}",
                f"training-label mean: {training_mean:.8f}",
                f"mean-baseline test MAE: {mean_mae:.8f}",
                f"mean-baseline test RMSE: {mean_rmse:.8f}",
                f"training-label median: {training_median:.8f}",
                f"median-baseline test MAE: {median_mae:.8f}",
                f"median-baseline test RMSE: {median_rmse:.8f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    METADATA_PATH.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "checkpoint_path": str(CHECKPOINT_PATH.resolve()),
                "checkpoint_sha256": CHECKPOINT_SHA256 or sha256_for(CHECKPOINT_PATH),
                "training_manifest_path": (
                    str(CHECKPOINT_MANIFEST_PATH.resolve())
                    if CHECKPOINT_MANIFEST_PATH is not None
                    else None
                ),
                "test_manifest_path": str(TEST_CSV.resolve()),
                "test_manifest_sha256": hashlib.sha256(TEST_CSV.read_bytes()).hexdigest(),
                "model_name": MODEL_NAME,
                "model_version": MODEL_VERSION,
                "created_at_epoch_seconds": time.time(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print_results(rows)
    print("\nAggregate metrics")
    print(f"Test MAE:                 {test_mae:.6f}")
    print(f"Test RMSE:                {test_rmse:.6f}")
    print(f"Training-label mean:      {training_mean:.6f}")
    print(f"Mean-baseline test MAE:   {mean_mae:.6f}")
    print(f"Mean-baseline test RMSE:  {mean_rmse:.6f}")
    print(f"Training-label median:    {training_median:.6f}")
    print(f"Median-baseline test MAE: {median_mae:.6f}")
    print(f"Median-baseline test RMSE:{median_rmse:>10.6f}")
    print(f"Predictions CSV: {PREDICTIONS_CSV}")
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Inspection gallery: {GALLERY_PATH}")
    print(f"Evaluation metadata: {METADATA_PATH}")
    print(f"Total elapsed time: {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
