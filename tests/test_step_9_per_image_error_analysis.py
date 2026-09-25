from __future__ import annotations

import json
import math
from pathlib import Path

import step_9_a_per_image_error_analysis as experiment


def test_metrics_and_bins() -> None:
    metrics = experiment.regression_metrics([0.0, 0.5, 1.0], [0.2, 0.4, 0.8])
    assert metrics["mae"] == 0.16666666666666666
    assert math.isclose(metrics["mean_signed_error"], -0.0333333333, rel_tol=1e-8)
    grouped = experiment.bin_metrics([0.19, 0.20, 0.39, 0.40, 0.79, 0.80, 1.0], [0.19, 0.20, 0.39, 0.40, 0.79, 0.80, 1.0])
    assert [value["count"] for value in grouped.values()] == [1, 2, 1, 1, 2]


def test_output_directory_contains_version_checkpoint_and_split_hash(tmp_path: Path) -> None:
    checkpoint = tmp_path / "wave-regression-baseline-v12-best-val-mae-epoch=1-val_mae=0.1000.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    snapshot = tmp_path / "snapshots" / "20260101T000000000000Z"
    snapshot.mkdir(parents=True)
    (snapshot / "manifest.json").write_text("{}", encoding="utf-8")
    output = experiment.output_directory(tmp_path / "results", "v12", checkpoint, snapshot, tmp_path)
    assert output.parts[-3] == "v12"
    assert "sha256-" in output.parts[-2]
    assert output.parts[-1].startswith("20260101T000000000000Z--sha256-")


def test_make_rows_sorts_worst_first_and_marks_baseline() -> None:
    rows = experiment.make_rows([("a.jpg", 0.0), ("b.jpg", 0.5)], [0.1, 0.9], 0.4)
    assert [row["filename"] for row in rows] == ["b.jpg", "a.jpg"]
    assert rows[0]["neural_network_beats_baseline"] is False


def test_complete_outputs_requires_all_deliverables(tmp_path: Path) -> None:
    assert not experiment.complete_outputs(tmp_path)
    names = ("predictions.csv", "predictions.json", "metrics.json", "actual_vs_predicted.png", "residuals.png", "error_comparison.png", "prediction_distribution.png", "error_gallery.html", "experiment_report.md")
    for name in names:
        (tmp_path / name).write_text("x", encoding="utf-8")
    assert experiment.complete_outputs(tmp_path)


def test_select_checkpoint_uses_newest_training_manifest(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "step-5-checkpoints"
    older = checkpoint_root / "v10" / "training_manifest_20260101T000000000000Z.json"
    newer = checkpoint_root / "v11" / "training_manifest_20260102T000000000000Z.json"
    older.parent.mkdir(parents=True)
    newer.parent.mkdir(parents=True)
    old_checkpoint = older.parent / "old.ckpt"
    new_checkpoint = newer.parent / "new.ckpt"
    old_checkpoint.write_bytes(b"old")
    new_checkpoint.write_bytes(b"new")
    older.write_text(json.dumps({"best_checkpoint_path": str(old_checkpoint)}), encoding="utf-8")
    newer.write_text(json.dumps({"best_checkpoint_path": str(new_checkpoint)}), encoding="utf-8")

    selected = experiment.select_checkpoint(tmp_path, checkpoint_root=checkpoint_root)

    assert selected == new_checkpoint.resolve()


def test_select_checkpoint_allows_explicit_override(tmp_path: Path) -> None:
    checkpoint = tmp_path / "chosen.ckpt"
    checkpoint.write_bytes(b"checkpoint")

    selected = experiment.select_checkpoint(tmp_path, checkpoint)

    assert selected == checkpoint.resolve()
