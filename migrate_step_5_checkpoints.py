"""Migrate legacy Step 5 artifacts into version-specific directories.

Run without arguments to perform the migration. Use ``--dry-run`` to inspect
the planned moves and reference updates first.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path


VERSION_PATTERN = re.compile(r"(?:^|[-_])v(\d+)(?:[-_.]|$)")
MANIFEST_PATTERN = re.compile(r"^training_manifest_.*\.json$")


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
    update_text_references(root, replacements, dry_run)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("step-5-checkpoints"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return migrate(args.root.resolve(), args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
