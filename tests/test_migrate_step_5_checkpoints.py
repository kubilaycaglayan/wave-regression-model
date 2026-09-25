from __future__ import annotations

import json
from pathlib import Path

import migrate_step_5_checkpoints as migration


def test_migrate_moves_versioned_files_and_updates_manifest(tmp_path: Path) -> None:
    root = tmp_path / "step-5-checkpoints"
    root.mkdir()
    checkpoint = root / "wave-regression-baseline-v12-best.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    manifest = root / "training_manifest_run.json"
    manifest.write_text(
        json.dumps(
            {
                "run_version": "v12",
                "backbone_name": "resnet18",
                "best_validation_mae": 0.123456,
                "best_checkpoint_path": str(checkpoint),
            }
        ),
        encoding="utf-8",
    )

    migration.migrate(root)

    migrated_checkpoint = root / "v12-mae-0.1235-resnet18" / checkpoint.name
    migrated_manifest = root / "v12-mae-0.1235-resnet18" / manifest.name
    assert migrated_checkpoint.is_file()
    assert migrated_manifest.is_file()
    assert not checkpoint.exists()
    assert not manifest.exists()
    metadata = json.loads(migrated_manifest.read_text(encoding="utf-8"))
    assert metadata["best_checkpoint_path"] == str(migrated_checkpoint)


def test_migrate_dry_run_does_not_move_unversioned_files(tmp_path: Path) -> None:
    root = tmp_path / "step-5-checkpoints"
    root.mkdir()
    artifact = root / "README.txt"
    artifact.write_text("keep", encoding="utf-8")

    migration.migrate(root, dry_run=True)

    assert artifact.is_file()


def test_migrate_renames_existing_versions_without_prefix_collisions(tmp_path: Path) -> None:
    root = tmp_path / "step-5-checkpoints"
    root.mkdir()
    for version, mae in (("v1", "0.1000"), ("v10", "0.2000")):
        directory = root / version
        directory.mkdir()
        (directory / f"training_summary_{version}.txt").write_text(
            f"backbone: resnet18\nbest validation MAE: {mae}\n",
            encoding="utf-8",
        )

    migration.migrate(root)

    assert (root / "v1-mae-0.1000-resnet18").is_dir()
    assert (root / "v10-mae-0.2000-resnet18").is_dir()
