"""Migrate legacy Step 5 artifacts into version-specific directories.

Run without arguments to perform the migration. Use ``--dry-run`` to inspect
the planned moves and reference updates first.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from pathlib import Path


VERSION_PATTERN = re.compile(r"(?:^|[-_])v(\d+)(?:[-_.]|$)")
MANIFEST_PATTERN = re.compile(r"^training_manifest_.*\.json$")
VERSION_DIRECTORY_PATTERN = re.compile(r"^(v\d+)$")
VERSIONED_MAE_DIRECTORY_PATTERN = re.compile(r"^(v\d+)-mae-(\d+(?:\.\d+)?)$")
SUMMARY_MAE_PATTERN = re.compile(r"^best validation MAE:\s*([0-9]+(?:\.[0-9]+)?)\s*$", re.MULTILINE)


def version_from_name(name: str) -> str | None:
    match = VERSION_PATTERN.search(name)
    return f"v{match.group(1)}" if match else None


def manifest_version(path: Path) -> str | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    run_version = value.get("run_version")
    if isinstance(run_version, str) and re.fullmatch(r"v\d+", run_version):
        return run_version
    return None


def directory_mae(directory: Path) -> float | None:
    values: list[float] = []
    for manifest in sorted(directory.rglob("training_manifest_*.json")):
        try:
            value = json.loads(manifest.read_text(encoding="utf-8")).get("best_validation_mae")
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    for summary in sorted(directory.glob("training_summary_*.txt")):
        try:
            match = SUMMARY_MAE_PATTERN.search(summary.read_text(encoding="utf-8"))
        except OSError:
            continue
        if match:
            values.append(float(match.group(1)))
    for checkpoint in sorted(directory.glob("*.ckpt")):
        match = re.search(r"val_mae=([0-9]+(?:\.[0-9]+)?)", checkpoint.name)
        if match:
            values.append(float(match.group(1)))
    unique_values = {round(value, 4) for value in values}
    if len(unique_values) > 1:
        raise ValueError(f"Conflicting MAE values in {directory}: {sorted(unique_values)}")
    return next(iter(unique_values)) if unique_values else None


def collect_moves(root: Path) -> tuple[list[tuple[Path, Path]], list[Path]]:
    moves: list[tuple[Path, Path]] = []
    unresolved: list[Path] = []
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        version = manifest_version(path) if MANIFEST_PATTERN.match(path.name) else version_from_name(path.name)
        if version is None:
            unresolved.append(path)
            continue
        moves.append((path, root / version / path.name))
    return moves, unresolved


def collect_version_directory_renames(root: Path) -> tuple[list[tuple[Path, Path]], list[Path]]:
    renames: list[tuple[Path, Path]] = []
    unresolved: list[Path] = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        match = VERSION_DIRECTORY_PATTERN.fullmatch(directory.name)
        if not match:
            continue
        mae = directory_mae(directory)
        if mae is None or not 0.0 <= mae <= 1.0:
            unresolved.append(directory)
            continue
        renames.append((directory, root / f"{match.group(1)}-mae-{mae:.4f}"))
    return renames, unresolved


def update_text_references(root: Path, replacements: dict[str, str], dry_run: bool) -> None:
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".json", ".txt"}:
            continue
        original = path.read_text(encoding="utf-8")
        updated = original
        for old_absolute, new_absolute in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
            updated = updated.replace(old_absolute + os.sep, new_absolute + os.sep)
            updated = updated.replace(old_absolute, new_absolute)
            old_relative = Path(old_absolute).relative_to(Path.cwd()) if Path(old_absolute).is_relative_to(Path.cwd()) else None
            if old_relative is not None:
                old_relative_text = str(old_relative)
                new_relative_text = str(Path(new_absolute).relative_to(Path.cwd()))
                updated = updated.replace(old_relative_text + os.sep, new_relative_text + os.sep)
                updated = updated.replace(old_relative_text, new_relative_text)
        if updated != original:
            print(f"update {path}")
            if not dry_run:
                path.write_text(updated, encoding="utf-8")


def repair_versioned_references(root: Path, dry_run: bool) -> None:
    """Repair references using the containing final directory as source of truth."""
    for directory in sorted(path for path in root.iterdir() if path.is_dir() and VERSIONED_MAE_DIRECTORY_PATTERN.fullmatch(path.name)):
        for manifest in sorted(directory.rglob("training_manifest_*.json")):
            value = json.loads(manifest.read_text(encoding="utf-8"))
            changed = False
            for key in ("best_checkpoint_path", "training_history_path", "plot_path"):
                reference = value.get(key)
                if not isinstance(reference, str):
                    continue
                candidate = directory / Path(reference).name
                if key == "best_checkpoint_path" or candidate.is_file():
                    if value[key] != str(candidate):
                        value[key] = str(candidate)
                        changed = True
            plot_paths = value.get("plot_paths")
            if isinstance(plot_paths, dict):
                for key, reference in list(plot_paths.items()):
                    if not isinstance(reference, str):
                        continue
                    candidate = directory / Path(reference).name
                    if candidate.is_file() and plot_paths[key] != str(candidate):
                        plot_paths[key] = str(candidate)
                        changed = True
            if changed:
                print(f"repair {manifest}")
                if not dry_run:
                    manifest.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

        for summary in sorted(directory.glob("training_summary_*.txt")):
            original = summary.read_text(encoding="utf-8")
            updated_lines: list[str] = []
            changed = False
            for line in original.splitlines():
                updated_line = line
                for label in ("best checkpoint path", "training history path", "combined plot path"):
                    prefix = f"{label}:"
                    if line.startswith(prefix):
                        candidate = directory / Path(line.partition(":")[2].strip()).name
                        if candidate.is_file() or label == "best checkpoint path":
                            updated_line = f"{prefix} {candidate}"
                changed = changed or updated_line != line
                updated_lines.append(updated_line)
            if changed:
                print(f"repair {summary}")
                if not dry_run:
                    summary.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")


def migrate(root: Path, dry_run: bool = False) -> int:
    root.mkdir(parents=True, exist_ok=True)
    moves, unresolved = collect_moves(root)

    replacements: dict[str, str] = {}
    for source, destination in moves:
        if destination.exists():
            raise FileExistsError(f"Migration destination already exists: {destination}")
        print(f"move {source} -> {destination}")
        replacements[str(source.resolve())] = str(destination.resolve())

    for path in unresolved:
        print(f"unresolved (left in place): {path}")

    if not dry_run:
        for source, destination in moves:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
    renames, unresolved_directories = collect_version_directory_renames(root)
    for source, destination in renames:
        if destination.exists():
            raise FileExistsError(f"Migration destination already exists: {destination}")
        print(f"rename {source} -> {destination}")
        replacements[str(source.resolve())] = str(destination.resolve())
    for directory in unresolved_directories:
        print(f"unresolved directory (left in place): {directory}")
    if not dry_run:
        for source, destination in renames:
            source.rename(destination)
    update_text_references(root, replacements, dry_run)
    repair_versioned_references(root, dry_run)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("step-5-checkpoints"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return migrate(args.root.resolve(), args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
