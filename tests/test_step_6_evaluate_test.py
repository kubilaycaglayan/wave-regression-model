from __future__ import annotations

import json
from pathlib import Path

import pytest

import step_6_a_evaluate_test as evaluation


def write_manifest(directory: Path, name: str, checkpoint: Path) -> Path:
    manifest = directory / f"training_manifest_{name}.json"
    manifest.write_text(
        json.dumps({"best_checkpoint_path": str(checkpoint)}),
        encoding="utf-8",
    )
    return manifest


def test_latest_manifest_selects_newest_existing_checkpoint(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    older = checkpoint_dir / "older.ckpt"
    newest = checkpoint_dir / "newest.ckpt"
    older.write_bytes(b"older")
    newest.write_bytes(b"newest")
    write_manifest(checkpoint_dir, "20260101T000000Z", older)
    newest_manifest = write_manifest(checkpoint_dir, "20260102T000000Z", newest)

    selected, manifest = evaluation.checkpoint_from_latest_manifest(checkpoint_dir)

    assert selected == newest
    assert manifest == newest_manifest


def test_latest_manifest_missing_checkpoint_fails_without_fallback(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    valid_checkpoint = checkpoint_dir / "older.ckpt"
    valid_checkpoint.write_bytes(b"older")
    write_manifest(checkpoint_dir, "20260101T000000Z", valid_checkpoint)
    newest_manifest = checkpoint_dir / "training_manifest_20260102T000000Z.json"
    newest_manifest.write_text(
        json.dumps({"best_checkpoint_path": str(checkpoint_dir / "missing.ckpt")}),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="references missing checkpoint"):
        evaluation.checkpoint_from_latest_manifest(checkpoint_dir)


def test_explicit_environment_checkpoint_overrides_manifest(tmp_path: Path, monkeypatch) -> None:
    checkpoint = tmp_path / "explicit.ckpt"
    checkpoint.write_bytes(b"explicit")
    monkeypatch.setenv("EVALUATION_CHECKPOINT_PATH", str(checkpoint))

    selected, manifest = evaluation.checkpoint_path_from_env()

    assert selected == checkpoint
    assert manifest is None


def test_output_directory_contains_checkpoint_fingerprint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint contents")
    test_manifest = tmp_path / "benchmark-v1-test.csv"
    test_manifest.write_bytes(b"filename,waviness\nimage.jpg,0.5\n")

    output = evaluation.evaluation_output_dir(checkpoint, test_manifest)

    assert output.name.startswith("benchmark-v1-test--sha256-")
    assert checkpoint.stem in output.parent.name
    assert evaluation.sha256_for(checkpoint)[:12] in output.parent.name
    assert evaluation.sha256_for(test_manifest)[:12] in output.name


def test_replacing_checkpoint_at_same_path_changes_output_directory(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"first checkpoint")
    test_manifest = tmp_path / "benchmark-v1-test.csv"
    test_manifest.write_bytes(b"test manifest")
    first_output = evaluation.evaluation_output_dir(checkpoint, test_manifest)
    checkpoint.write_bytes(b"replacement checkpoint")
    second_output = evaluation.evaluation_output_dir(checkpoint, test_manifest)

    assert first_output != second_output


def test_changing_test_manifest_changes_output_directory(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    test_manifest = tmp_path / "benchmark-v1-test.csv"
    test_manifest.write_bytes(b"first manifest")
    first_output = evaluation.evaluation_output_dir(checkpoint, test_manifest)
    test_manifest.write_bytes(b"replacement manifest")
    second_output = evaluation.evaluation_output_dir(checkpoint, test_manifest)

    assert first_output != second_output
