"""One-time filename migration when fast-v2.2 final inputs move to Step 1."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path


def migrate_fast_v22_records(profile_name: str, repository_root: Path = Path(".")) -> int:
    """Rename active Step 2 filename references to their Step 1 image names.

    Immutable historical split snapshots are intentionally left as recorded.
    """
    if profile_name != "fast-v2.2":
        return 0

    paths = [
        repository_root / "labels.csv",
        repository_root / "discarded_images.csv",
        *(repository_root / "step-3-dataset-splits" / f"{name}.csv"
          for name in ("train", "validation", "test")),
    ]
    changed = 0
    for path in paths:
        if not path.is_file():
            continue
        with path.open("r", newline="", encoding="utf-8-sig") as source:
            rows = list(csv.reader(source))
        if not rows or "filename" not in rows[0]:
            continue
        filename_column = rows[0].index("filename")
        updates = 0
        for row in rows[1:]:
            if filename_column < len(row) and row[filename_column].startswith("step-2_"):
                row[filename_column] = "step-1_" + row[filename_column][len("step-2_"):]
                updates += 1
        if not updates:
            continue
        fd, temporary = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", newline="", encoding="utf-8") as destination:
                csv.writer(destination).writerows(rows)
            os.replace(temporary, path)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
        changed += updates
        print(f"Migrated {updates} Step 2 filename reference(s) in {path}")
    return changed
