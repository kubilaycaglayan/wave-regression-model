#!/usr/bin/env python3
"""Build a reproducible whole cohort from the fixed train/validation/test split."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

PERF = Path(__file__).resolve().parents[1]
ROOT = PERF.parents[2]
SNAPSHOT = "20261028T081606614158Z"
SPLIT_NAMES = ("train", "validation", "test")
OUTPUT_DIR = PERF / "datasets"
OUTPUT_CSV = OUTPUT_DIR / "whole.csv"
OUTPUT_MANIFEST = OUTPUT_DIR / "whole-manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    snapshot_dir = ROOT / "step-3-dataset-splits" / "snapshots" / SNAPSHOT
    rows: dict[str, dict[str, str]] = {}
    counts = {}
    split_hashes = {}
    for split in SPLIT_NAMES:
        split_path = snapshot_dir / f"{split}.csv"
        split_hashes[split] = sha256(split_path)
        with split_path.open(newline="", encoding="utf-8") as stream:
            split_rows = list(csv.DictReader(stream))
        counts[split] = len(split_rows)
        for row in split_rows:
            filename = row["filename"]
            if filename in rows:
                raise ValueError(f"Filename appears in multiple splits: {filename}")
            rows[filename] = {"filename": filename, "waviness": row["waviness"]}

    labels_path = ROOT / "labels.csv"
    with labels_path.open(newline="", encoding="utf-8") as stream:
        labels = {row["filename"]: row["waviness"] for row in csv.DictReader(stream)}
    if rows != {name: {"filename": name, "waviness": value} for name, value in labels.items()}:
        raise ValueError("Fixed snapshot rows do not exactly match the repository labels.csv")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("filename", "waviness"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows[name] for name in sorted(rows, key=str.casefold))
    manifest = {
        "schema_version": 1,
        "split_name": "whole",
        "description": "All labeled photos; union of the fixed train, validation, and test snapshot.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot": SNAPSHOT,
        "source_splits": list(SPLIT_NAMES),
        "source_split_counts": counts,
        "count": len(rows),
        "source_split_sha256": split_hashes,
        "labels_sha256": sha256(labels_path),
        "whole_csv_sha256": sha256(OUTPUT_CSV),
        "evaluation_warning": "Overlaps train, validation, and test; descriptive aggregate only, not an independent evaluation split.",
    }
    OUTPUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Whole split: {OUTPUT_CSV} ({len(rows)} labeled photos)")
    print(f"Manifest: {OUTPUT_MANIFEST}")


if __name__ == "__main__":
    main()
