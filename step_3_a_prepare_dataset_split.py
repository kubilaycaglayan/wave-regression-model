"""Validate labels and create deterministic dataset split manifests."""

from __future__ import annotations

import csv
import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


LABELS_PATH = Path("labels.csv")
DISCARDED_PATH = Path("discarded_images.csv")
IMAGE_DIR = Path("step-2-final-water-data")
OUTPUT_DIR = Path("step-3-dataset-splits")
SNAPSHOT_DIR = OUTPUT_DIR / "snapshots"
BENCHMARK_DIR = OUTPUT_DIR / "benchmarks"
BENCHMARK_TEST_PATH = BENCHMARK_DIR / "benchmark-v1-test.csv"
SPLIT_NAMES = ("train", "validation", "test")
TARGET_FRACTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}
BIN_NAMES = ("0.00-0.19", "0.20-0.39", "0.40-0.59", "0.60-0.79", "0.80-1.00")
DEFAULT_SEED = 42


@dataclass(frozen=True)
class Sample:
    filename: str
    waviness: float


def read_discarded() -> set[str]:
    if not DISCARDED_PATH.exists():
        return set()
    with DISCARDED_PATH.open(newline="", encoding="utf-8") as handle:
        return {
            (row.get("filename") or "").strip()
            for row in csv.DictReader(handle)
            if (row.get("filename") or "").strip()
        }


def waviness_bin(value: float) -> int:
    if value < 0.20:
        return 0
    if value < 0.40:
        return 1
    if value < 0.60:
        return 2
    if value < 0.80:
        return 3
    return 4


def read_labels() -> list[Sample]:
    errors: list[str] = []
    samples: list[Sample] = []
    seen: set[str] = set()
    discarded = read_discarded()

    try:
        with LABELS_PATH.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["filename", "waviness"]:
                errors.append("labels.csv must have exactly the header: filename,waviness")
            for row_number, row in enumerate(reader, start=2):
                filename = (row.get("filename") or "").strip()
                raw_value = (row.get("waviness") or "").strip()
                if not filename:
                    errors.append(f"row {row_number}: filename is empty")
                    continue
                if filename in discarded:
                    continue
                if filename in seen:
                    errors.append(f"row {row_number}: duplicate filename: {filename}")
                seen.add(filename)

                try:
                    value = float(raw_value)
                except ValueError:
                    errors.append(f"row {row_number}: waviness is not numeric: {raw_value!r}")
                    continue
                if not 0.0 <= value <= 1.0:
                    errors.append(f"row {row_number}: waviness is outside [0.0, 1.0]: {value}")
                    continue
                samples.append(Sample(filename, value))
    except FileNotFoundError:
        errors.append(f"labels file not found: {LABELS_PATH}")

    if errors:
        raise ValueError("Invalid labels:\n" + "\n".join(f"- {error}" for error in errors))
    return samples


def group_samples(samples: list[Sample]) -> list[list[Sample]]:
    """Treat every independently collected image as its own split group."""
    return [[sample] for sample in sorted(samples, key=lambda sample: sample.filename)]


def split_score(splits: dict[str, list[Sample]], total_bins: Counter[int], total: int) -> float:
    score = 0.0
    for name in SPLIT_NAMES:
        samples = splits[name]
        target_count = total * TARGET_FRACTIONS[name]
        score += ((len(samples) - target_count) / max(total, 1)) ** 2 * 4
        counts = Counter(waviness_bin(sample.waviness) for sample in samples)
        for bin_number, total_count in total_bins.items():
            target = total_count * TARGET_FRACTIONS[name]
            score += ((counts[bin_number] - target) / max(total, 1)) ** 2
    return score


def _assign_groups(
    groups: list[list[Sample]],
    fixed_splits: dict[str, list[Sample]],
    seed: int,
) -> dict[str, list[Sample]]:
    total = sum(len(split) for split in fixed_splits.values()) + sum(len(group) for group in groups)
    total_bins = Counter(
        waviness_bin(sample.waviness)
        for split in fixed_splits.values()
        for sample in split
    )
    total_bins.update(waviness_bin(sample.waviness) for group in groups for sample in group)
    rng = random.Random(seed)
    best: tuple[float, dict[str, list[Sample]]] | None = None

    # Multiple seeded greedy passes provide useful bin balancing while retaining whole groups.
    for _ in range(1000):
        order = list(groups)
        rng.shuffle(order)
        order.sort(key=len, reverse=True)
        candidate = {name: list(fixed_splits[name]) for name in SPLIT_NAMES}
        for group in order:
            options = []
            for name in SPLIT_NAMES:
                candidate[name].extend(group)
                options.append((split_score(candidate, total_bins, total), name))
                del candidate[name][-len(group):]
            _, chosen = min(options, key=lambda item: (item[0], SPLIT_NAMES.index(item[1])))
            candidate[chosen].extend(group)
        score = split_score(candidate, total_bins, total)
        if best is None or score < best[0]:
            best = (score, {name: list(candidate[name]) for name in SPLIT_NAMES})

    assert best is not None
    return {name: sorted(best[1][name], key=lambda sample: sample.filename) for name in SPLIT_NAMES}


def make_splits(groups: list[list[Sample]], seed: int = DEFAULT_SEED) -> dict[str, list[Sample]]:
    """Create an initial deterministic split when no assignments exist yet."""
    return _assign_groups(groups, {name: [] for name in SPLIT_NAMES}, seed)


def read_split_csv(path: Path, samples_by_name: dict[str, Sample], discarded: set[str] | None = None) -> list[Sample]:
    if not path.is_file():
        raise FileNotFoundError(f"Existing split manifest not found: {path}")
    samples: list[Sample] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["filename", "waviness"]:
            raise ValueError(f"{path} must have header: filename,waviness")
        for row_number, row in enumerate(reader, start=2):
            filename = (row.get("filename") or "").strip()
            if filename in (discarded or set()):
                continue
            if filename not in samples_by_name:
                raise ValueError(f"{path}:{row_number}: filename is not present in the current labels.csv: {filename}")
            try:
                value = float(row["waviness"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path}:{row_number}: invalid waviness") from error
            sample = samples_by_name[filename]
            if value != sample.waviness:
                raise ValueError(
                    f"{path}:{row_number}: label changed for existing sample {filename}; "
                    "existing assignments are immutable"
                )
            samples.append(sample)
    if len({sample.filename for sample in samples}) != len(samples):
        raise ValueError(f"Duplicate filenames found in {path}")
    return samples


def load_or_create_incremental_splits(
    samples: list[Sample], seed: int, discarded: set[str] | None = None
) -> tuple[dict[str, list[Sample]], bool]:
    """Keep existing assignments fixed and assign only previously unseen groups."""
    samples_by_name = {sample.filename: sample for sample in samples}
    split_paths = {name: OUTPUT_DIR / f"{name}.csv" for name in SPLIT_NAMES}
    existing = {name: path.is_file() for name, path in split_paths.items()}
    if not any(existing.values()):
        return make_splits(group_samples(samples), seed), False
    if not all(existing.values()):
        missing = ", ".join(name for name, present in existing.items() if not present)
        raise FileNotFoundError(f"Incremental split registry is incomplete; missing: {missing}")

    fixed = {name: read_split_csv(path, samples_by_name, discarded) for name, path in split_paths.items()}
    assigned_names = [sample.filename for split in fixed.values() for sample in split]
    if len(assigned_names) != len(set(assigned_names)):
        raise ValueError("Existing split manifests assign a filename more than once")

    groups = group_samples(samples)
    fixed_by_name = {sample.filename: name for name, split in fixed.items() for sample in split}
    new_groups: list[list[Sample]] = []
    for group in groups:
        group_splits = {fixed_by_name[sample.filename] for sample in group if sample.filename in fixed_by_name}
        if len(group_splits) > 1:
            raise ValueError(f"Existing manifests split one image group across multiple splits: {group[0].filename}")
        if not group_splits:
            new_groups.append(group)
        else:
            split_name = next(iter(group_splits))
            fixed[split_name].extend(sample for sample in group if sample.filename not in fixed_by_name)

    if new_groups:
        return _assign_groups(new_groups, fixed, seed), True
    return {name: sorted(split, key=lambda sample: sample.filename) for name, split in fixed.items()}, True


def validate_images(samples: list[Sample]) -> list[str]:
    discarded = read_discarded()
    processed = {
        path.name
        for path in IMAGE_DIR.iterdir()
        if path.is_file()
        and path.name.startswith("step-2_")
        and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        and path.name not in discarded
    }
    labeled = {sample.filename for sample in samples}
    missing = sorted(labeled - processed)
    unlabeled = sorted(processed - labeled)
    print(f"Processed images without labels ({len(unlabeled)}): {', '.join(unlabeled) or 'none'}")
    if discarded:
        print(f"Discarded images excluded from splits ({len(discarded)}): {', '.join(sorted(discarded))}")
    if missing:
        raise FileNotFoundError("Labeled images missing from " + str(IMAGE_DIR) + ":\n" + "\n".join(f"- {name}" for name in missing))
    return sorted(labeled)


def write_csv(path: Path, samples: list[Sample]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "waviness"])
        writer.writerows((sample.filename, f"{sample.waviness:.2f}") for sample in samples)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_snapshot_path() -> tuple[str, Path]:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    snapshot_path = SNAPSHOT_DIR / timestamp
    suffix = 1
    while snapshot_path.exists():
        snapshot_path = SNAPSHOT_DIR / f"{timestamp}_{suffix}"
        suffix += 1
    snapshot_path.mkdir()
    return snapshot_path.name, snapshot_path


def write_snapshot(splits: dict[str, list[Sample]], seed: int, incremental: bool, summary: str) -> Path:
    snapshot_name, snapshot_path = unique_snapshot_path()
    files: dict[str, dict[str, object]] = {}
    for name in SPLIT_NAMES:
        snapshot_csv = snapshot_path / f"{name}.csv"
        write_csv(snapshot_csv, splits[name])
        files[name] = {
            "path": str(snapshot_csv.relative_to(OUTPUT_DIR)),
            "sha256": file_sha256(snapshot_csv),
            "count": len(splits[name]),
        }

    metadata = {
        "schema_version": 1,
        "snapshot": snapshot_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "seed": seed,
        "mode": "append-existing" if incremental else "initial",
        "labels_sha256": file_sha256(LABELS_PATH),
        "files": files,
    }
    metadata_path = snapshot_path / "manifest.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (snapshot_path / "summary.txt").write_text(summary, encoding="utf-8")
    return metadata_path


def ensure_benchmark_v1(test_samples: list[Sample]) -> None:
    """Create the first benchmark once; never alter it on later data-ingestion runs."""
    if BENCHMARK_TEST_PATH.exists():
        return
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(BENCHMARK_TEST_PATH, test_samples)


def check_splits(splits: dict[str, list[Sample]], groups: list[list[Sample]], samples: list[Sample]) -> None:
    all_names = [sample.filename for split in splits.values() for sample in split]
    if len(all_names) != len(set(all_names)) or set(all_names) != {sample.filename for sample in samples}:
        raise RuntimeError("Split sanity check failed: sample union or uniqueness is incorrect")
    group_by_name = {sample.filename: index for index, group in enumerate(groups) for sample in group}
    assigned_groups = {name: set(group_by_name[sample.filename] for sample in split) for name, split in splits.items()}
    if any(assigned_groups[left] & assigned_groups[right] for left in SPLIT_NAMES for right in SPLIT_NAMES if left < right):
        raise RuntimeError("Split sanity check failed: a group appears in more than one split")


def summary_text(splits: dict[str, list[Sample]], groups: list[list[Sample]], total: int) -> str:
    lines = [f"total labeled samples: {total}", f"number of independent groups: {len(groups)}", ""]
    group_lookup = {sample.filename: index + 1 for index, group in enumerate(groups) for sample in group}
    for name in SPLIT_NAMES:
        values = [sample.waviness for sample in splits[name]]
        counts = Counter(waviness_bin(value) for value in values)
        mean = sum(values) / len(values) if values else 0.0
        lines += [
            f"{name}: {len(values)} samples ({len(values) / total * 100:.2f}%)",
            f"  waviness min/max/mean: {min(values):.2f}/{max(values):.2f}/{mean:.3f}" if values else "  waviness min/max/mean: n/a",
            "  waviness bins: " + ", ".join(f"{BIN_NAMES[i]}={counts[i]}" for i in range(5)),
            "  groups: " + ", ".join(dict.fromkeys(str(group_lookup[sample.filename]) for sample in splits[name])),
            "",
        ]
    lines.append("group membership:")
    for index, group in enumerate(groups, start=1):
        lines.append(f"  group {index}: " + ", ".join(sample.filename for sample in group))
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"seed for repeatable split generation (default: {DEFAULT_SEED})",
    )
    args = parser.parse_args()
    try:
        samples = read_labels()
        if not IMAGE_DIR.is_dir():
            raise FileNotFoundError(f"processed image directory not found: {IMAGE_DIR}")
        validate_images(samples)
        samples_by_name = {sample.filename: sample for sample in samples}
        existing_test = OUTPUT_DIR / "test.csv"
        discarded = read_discarded()
        if not BENCHMARK_TEST_PATH.exists() and existing_test.exists():
            ensure_benchmark_v1(read_split_csv(existing_test, samples_by_name, discarded))
        splits, incremental = load_or_create_incremental_splits(samples, seed=args.seed, discarded=discarded)
        groups = group_samples(samples)
        check_splits(splits, groups, samples)
        OUTPUT_DIR.mkdir(exist_ok=True)
        for name in SPLIT_NAMES:
            write_csv(OUTPUT_DIR / f"{name}.csv", splits[name])
        text = summary_text(splits, groups, len(samples))
        snapshot_metadata = write_snapshot(splits, args.seed, incremental, text)
        ensure_benchmark_v1(splits["test"])
        print(f"Split seed: {args.seed}")
        print(f"Split mode: {'append-existing' if incremental else 'initial'}")
        print(f"Immutable snapshot: {snapshot_metadata}")
        print(f"Benchmark v1: {BENCHMARK_TEST_PATH}")
        print("\n" + text)
        return 0
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
