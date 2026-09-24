from __future__ import annotations

import json
import math
from pathlib import Path

import step_8_a_constant_prediction_baselines as experiment


def test_mean_absolute_error_and_reduction() -> None:
    assert math.isclose(experiment.mean_absolute_error([0.0, 1.0], [0.25, 0.5]), 0.375)
    assert math.isclose(experiment.relative_reduction(0.2, 0.15), 25.0)
    assert experiment.relative_reduction(0.0, 0.15) is None


def test_select_checkpoint_requires_exact_snapshot(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots" / "snapshot-a"
    snapshot_dir.mkdir(parents=True)
    snapshot = experiment.Snapshot("snapshot-a", snapshot_dir, {})
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "model.ckpt"
    checkpoint_path.write_bytes(b"checkpoint")
    manifest = checkpoint_dir / "training_manifest_a.json"
    manifest.write_text(
        json.dumps(
            {
                "split_snapshot": "snapshots/snapshot-b",
                "best_checkpoint_path": "checkpoints/model.ckpt",
                "best_validation_mae": 0.1,
            }
        ),
        encoding="utf-8",
    )
    assert experiment.select_checkpoint(snapshot, checkpoint_dir, tmp_path) is None


def test_select_checkpoint_chooses_lowest_matching_validation_mae(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshot-a"
    snapshot_dir.mkdir()
    snapshot = experiment.Snapshot("snapshot-a", snapshot_dir, {})
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    for name, mae in (("first.ckpt", 0.2), ("second.ckpt", 0.1)):
        (checkpoint_dir / name).write_bytes(name.encode())
        (checkpoint_dir / f"training_manifest_{name}.json").write_text(
            json.dumps(
                {
                    "split_snapshot": "snapshot-a",
                    "best_checkpoint_path": f"checkpoints/{name}",
                    "best_validation_mae": mae,
                }
            ),
            encoding="utf-8",
        )
    selected = experiment.select_checkpoint(snapshot, checkpoint_dir, tmp_path)
    assert selected is not None
    assert selected.path.name == "second.ckpt"
