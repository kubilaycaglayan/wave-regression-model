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
                "best_checkpoint_path": str(checkpoint),
            }
        ),
        encoding="utf-8",
    )

    migration.migrate(root)

    migrated_checkpoint = root / "v12" / checkpoint.name
    migrated_manifest = root / "v12" / manifest.name
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
