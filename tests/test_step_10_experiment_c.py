from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import step_10_experiment_c as experiment


def test_snapshot_metadata_requires_exact_requested_snapshot() -> None:
    snapshot_dir = experiment.newest_snapshot_dir()
    snapshot = experiment.snapshot_metadata(snapshot_dir)
    assert snapshot["name"] == snapshot_dir.name
    assert snapshot["counts"] == {
        split: snapshot["snapshot_manifest"]["files"][split]["count"]
        for split in ("train", "validation", "test")
    }


def test_snapshot_metadata_rejects_modified_manifest(tmp_path: Path) -> None:
    source = experiment.newest_snapshot_dir()
    target = tmp_path / source.name
    target.mkdir()
    for name in ("manifest.json", "train.csv", "validation.csv", "test.csv"):
        (target / name).write_bytes((source / name).read_bytes())
    (target / "validation.csv").write_text((target / "validation.csv").read_text() + "step-2_fake.jpg,0.2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Snapshot manifest count does not match validation"):
        experiment.snapshot_metadata(target)


def test_metric_definitions_include_bias_and_range_compression() -> None:
    metrics = experiment.compute_metrics([0.0, 0.5, 1.0], [0.2, 0.4, 0.8])
    assert metrics["validation_mae"] == pytest.approx(0.1666666667)
    assert metrics["mean_signed_error"] == pytest.approx(-0.0333333333)
    assert metrics["low_range_signed_error"] == pytest.approx(0.2)
    assert metrics["high_range_signed_error"] == pytest.approx(-0.2)
    assert metrics["prediction_actual_range_width_ratio"] == pytest.approx(0.6)


def test_model_has_only_head_trainable_without_loading_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    model = experiment.FrozenBackboneWaveRegressionModel("resnet18", pretrained=False)
    model.train()
    checks = model.frozen_backbone_mode_checks()
    assert checks["trainable_parameters"] == 513
    assert checks["frozen_batchnorm_modules"] > 0
    assert all(not parameter.requires_grad for parameter in model.feature_extractor.parameters())
    assert all(parameter.requires_grad for parameter in model.regression_head.parameters())


def test_aggregate_requires_three_seeds() -> None:
    result = {
        "config": {"backbone": "resnet18", "seed": 42},
        "metrics": {
            "validation_mae": 0.1,
            "validation_rmse": 0.2,
            "prediction_actual_range_width_ratio": 0.5,
            "low_range_signed_error": 0.1,
            "high_range_signed_error": -0.1,
        },
        "checkpoint": {"path": "model.ckpt", "sha256": "abc"},
    }
    with pytest.raises(RuntimeError, match="Expected 3 completed seeds"):
        experiment.aggregate_results([result])


def test_new_version_and_matching_incomplete_version_resume(tmp_path: Path) -> None:
    snapshot = experiment.snapshot_metadata(experiment.newest_snapshot_dir())
    planned = [("resnet18", 42)]
    first, first_manifest = experiment.prepare_experiment_version(tmp_path, snapshot, planned)
    assert first.name.startswith("v1-")
    assert first_manifest["status"] == "running"

    resumed, resumed_manifest = experiment.prepare_experiment_version(tmp_path, snapshot, planned)
    assert resumed == first
    assert resumed_manifest["config_fingerprint"] == first_manifest["config_fingerprint"]
    assert len(experiment.version_directories(tmp_path)) == 1


def test_changed_configuration_creates_new_version(tmp_path: Path) -> None:
    snapshot = experiment.snapshot_metadata(experiment.newest_snapshot_dir())
    experiment.prepare_experiment_version(tmp_path, snapshot, [("resnet18", 42)])
    second, _ = experiment.prepare_experiment_version(tmp_path, snapshot, [("resnet34", 42)])
    assert second.name.startswith("v2-")
    assert len(experiment.version_directories(tmp_path)) == 2


def test_completed_version_is_immutable(tmp_path: Path) -> None:
    snapshot = experiment.snapshot_metadata(experiment.newest_snapshot_dir())
    version, manifest = experiment.prepare_experiment_version(tmp_path, snapshot, [("resnet18", 42)])
    manifest["status"] = "complete"
    experiment.json_dump(version / "experiment_manifest.json", manifest)
    with pytest.raises(RuntimeError, match="complete and immutable"):
        experiment.prepare_experiment_version(tmp_path, snapshot, [("resnet18", 42)], requested_version=version.name)


def test_index_updates_one_record_per_version(tmp_path: Path) -> None:
    snapshot = experiment.snapshot_metadata(experiment.newest_snapshot_dir())
    version, manifest = experiment.prepare_experiment_version(tmp_path, snapshot, [("resnet18", 42)])
    experiment.update_experiment_index(tmp_path, manifest)
    manifest["status"] = "complete"
    manifest["completed_runs"] = ["resnet18-seed-42"]
    experiment.update_experiment_index(tmp_path, manifest)
    index = json.loads((tmp_path / "experiment_index.json").read_text(encoding="utf-8"))
    assert len(index["versions"]) == 1
    assert index["versions"][0]["status"] == "complete"
    assert index["versions"][0]["completed_runs"] == ["resnet18-seed-42"]
