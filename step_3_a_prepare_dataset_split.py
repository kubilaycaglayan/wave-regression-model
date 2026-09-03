"""Validate labels and create deterministic, group-aware dataset manifests."""

from __future__ import annotations

import csv
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


LABELS_PATH = Path("labels.csv")
IMAGE_DIR = Path("step-2-final-water-data")
OUTPUT_DIR = Path("step-3-dataset-splits")
MAX_GROUP_GAP = 5
RANDOM_SEED = 42
SPLIT_NAMES = ("train", "validation", "test")
TARGET_FRACTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}
IMG_NUMBER_RE = re.compile(r"IMG[_-](\d+)", re.IGNORECASE)
BIN_NAMES = ("0.00-0.19", "0.20-0.39", "0.40-0.59", "0.60-0.79", "0.80-1.00")


@dataclass(frozen=True)
class Sample:
    filename: str
    waviness: float
    image_number: int


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
                match = IMG_NUMBER_RE.search(Path(filename).stem)
                if not match:
                    errors.append(f"row {row_number}: no IMG number found in filename: {filename}")
                    continue
                samples.append(Sample(filename, value, int(match.group(1))))
    except FileNotFoundError:
        errors.append(f"labels file not found: {LABELS_PATH}")

    if errors:
        raise ValueError("Invalid labels:\n" + "\n".join(f"- {error}" for error in errors))
    return samples


def group_samples(samples: list[Sample]) -> list[list[Sample]]:
    ordered = sorted(samples, key=lambda sample: (sample.image_number, sample.filename))
    groups: list[list[Sample]] = []
    for sample in ordered:
        if not groups or sample.image_number - groups[-1][-1].image_number > MAX_GROUP_GAP:
            groups.append([])
        groups[-1].append(sample)
    return groups


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


def make_splits(groups: list[list[Sample]]) -> dict[str, list[Sample]]:
    total = sum(len(group) for group in groups)
    total_bins = Counter(waviness_bin(sample.waviness) for group in groups for sample in group)
    rng = random.Random(RANDOM_SEED)
    best: tuple[float, dict[str, list[Sample]]] | None = None

    # Multiple seeded greedy passes provide useful bin balancing while retaining whole groups.
    for _ in range(1000):
        order = list(groups)
        rng.shuffle(order)
        order.sort(key=len, reverse=True)
        candidate = {name: [] for name in SPLIT_NAMES}
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
    return {name: sorted(best[1][name], key=lambda sample: (sample.image_number, sample.filename)) for name in SPLIT_NAMES}


def validate_images(samples: list[Sample]) -> list[str]:
    processed = {
        path.name
        for path in IMAGE_DIR.iterdir()
        if path.is_file()
        and path.name.startswith("step-2_")
        and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    }
    labeled = {sample.filename for sample in samples}
    missing = sorted(labeled - processed)
    unlabeled = sorted(processed - labeled)
    print(f"Processed images without labels ({len(unlabeled)}): {', '.join(unlabeled) or 'none'}")
    if missing:
        raise FileNotFoundError("Labeled images missing from " + str(IMAGE_DIR) + ":\n" + "\n".join(f"- {name}" for name in missing))
    return sorted(labeled)


def write_csv(path: Path, samples: list[Sample]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "waviness"])
        writer.writerows((sample.filename, f"{sample.waviness:.2f}") for sample in samples)


def check_splits(splits: dict[str, list[Sample]], groups: list[list[Sample]], samples: list[Sample]) -> None:
    all_names = [sample.filename for split in splits.values() for sample in split]
    if len(all_names) != len(set(all_names)) or set(all_names) != {sample.filename for sample in samples}:
        raise RuntimeError("Split sanity check failed: sample union or uniqueness is incorrect")
    group_by_name = {sample.filename: index for index, group in enumerate(groups) for sample in group}
    assigned_groups = {name: set(group_by_name[sample.filename] for sample in split) for name, split in splits.items()}
    if any(assigned_groups[left] & assigned_groups[right] for left in SPLIT_NAMES for right in SPLIT_NAMES if left < right):
        raise RuntimeError("Split sanity check failed: a group appears in more than one split")


def summary_text(splits: dict[str, list[Sample]], groups: list[list[Sample]], total: int) -> str:
    lines = [f"total labeled samples: {total}", f"number of groups/sessions: {len(groups)}", ""]
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
        lines.append(f"  group {index} (IMG {group[0].image_number}-{group[-1].image_number}): " + ", ".join(sample.filename for sample in group))
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    try:
        samples = read_labels()
        if not IMAGE_DIR.is_dir():
            raise FileNotFoundError(f"processed image directory not found: {IMAGE_DIR}")
        validate_images(samples)
        groups = group_samples(samples)
        splits = make_splits(groups)
        check_splits(splits, groups, samples)
        OUTPUT_DIR.mkdir(exist_ok=True)
        for name in SPLIT_NAMES:
            write_csv(OUTPUT_DIR / f"{name}.csv", splits[name])
        text = summary_text(splits, groups, len(samples))
        (OUTPUT_DIR / "summary.txt").write_text(text, encoding="utf-8")
        print("\n" + text)
        return 0
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
