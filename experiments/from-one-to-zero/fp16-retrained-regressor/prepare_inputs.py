#!/usr/bin/env python3
"""Cache exact FP16-path regressor JPEG inputs for fixed dataset splits."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ARTIFACTS = HERE / "artifacts"
SNAPSHOT = "20261102T084438874261Z"
sys.path.insert(0, str(ROOT))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signature() -> dict:
    sources = {
        "experiment_manifest": HERE / "experiment.json",
        "prediction_server": ROOT / "step_7_i_migraphx_server.py",
        "crop_selector": ROOT / "step_2_b_crop_selector.py",
        "heif_decoder_wrapper": ROOT / "step_0_a_native_heif_decoder.py",
        "heif_decoder_source": ROOT / "native/libheif/step_0_a_tiled_decode.cpp",
        "segmentation_model": ROOT / "models/fast-v2.1/segformer-b0-step1-448-fp32.onnx",
    }
    native_lib = ROOT / "native/build/libwave_heif_tiles.so"
    if native_lib.is_file():
        sources["native_heif_decoder_binary"] = native_lib
    hashes = {name: sha(path) for name, path in sources.items()}
    payload = {"precision": "MIGraphX FP16 enabled", "profile": "fast-v2.1-coordinate",
               "step1_size": 448, "source_hashes": hashes}
    payload["signature_id"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]
    return payload


def read_rows(split: str) -> list[dict]:
    split_path = ROOT / "step-3-dataset-splits" / "snapshots" / SNAPSHOT / f"{split}.csv"
    with split_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"Dataset split is empty: {split_path}")
    raw_by_stem: dict[str, list[Path]] = {}
    for path in (ROOT / "step-0-raw-data").rglob("*"):
        if path.is_file() and path.suffix.lower() in {".heic", ".heif", ".jpg", ".jpeg", ".png"}:
            raw_by_stem.setdefault(path.stem.casefold(), []).append(path)
    items = []
    for row in rows:
        label_filename = row["filename"]
        stem = Path(label_filename).stem
        if stem.startswith("step-2_"):
            stem = stem[len("step-2_"):]
        matches = raw_by_stem.get(stem.casefold(), [])
        if len(matches) != 1:
            raise RuntimeError(f"Expected one raw source for {label_filename}; found {len(matches)}")
        items.append({"label_filename": label_filename, "label": float(row["waviness"]), "photo": matches[0]})
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", action="append", choices=("train", "validation", "test"),
                        help="Split(s) to prepare; default prepares train and validation only.")
    args = parser.parse_args()
    splits = args.split or ["train", "validation"]
    os.environ["WAVE_PIPELINE_PROFILE"] = "fast-v2.1-coordinate"
    os.environ["WAVE_MIGRAPHX_FP16"] = "1"
    os.environ["WAVE_STEP1_SIZE"] = "448"
    os.environ["WAVE_STEP1_ONNX"] = str(ROOT / "models/fast-v2.1/segformer-b0-step1-448-fp32.onnx")
    os.environ["WAVE_REGRESSOR_ONNX"] = str(ROOT / "models/fast-v2.1/wave-regression-efficientnet-b0-fast-v2.1.onnx")
    import step_7_i_migraphx_server as server

    setup = server.initialize_models()
    pipeline_signature = signature()
    cache_root = ARTIFACTS / "preprocessed-inputs" / pipeline_signature["signature_id"]
    image_root = cache_root / "images"
    manifest_root = cache_root / "manifests"
    image_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)
    print(f"FP16 Step 1 ready; model setup {time.perf_counter() - server.PROCESS_START:.2f}s", flush=True)
    print(f"MIGraphX setup: {json.dumps(setup, sort_keys=True)}", flush=True)

    for split in splits:
        rows = read_rows(split)
        split_csv = ROOT / "step-3-dataset-splits" / "snapshots" / SNAPSHOT / f"{split}.csv"
        manifest_path = manifest_root / f"{split}.json"
        old = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        old_items = old.get("items", {}) if old.get("pipeline_signature") == pipeline_signature else {}
        output = {"schema_version": 1, "experiment_id": "fp16-retrained-regressor",
                  "split": split, "split_snapshot": SNAPSHOT,
                  "split_csv_sha256": sha(split_csv), "pipeline_signature": pipeline_signature,
                  "items": {}, "created_at_utc": datetime.now(timezone.utc).isoformat()}
        started = time.perf_counter()
        processed = skipped = 0
        for number, item in enumerate(rows, start=1):
            label_name = item["label_filename"]
            source = item["photo"]
            source_hash = sha(source)
            output_name = Path(label_name).stem + ".jpg"
            output_path = image_root / output_name
            previous = old_items.get(label_name, {})
            if (output_path.is_file() and previous.get("source_sha256") == source_hash
                    and previous.get("cache_sha256") == sha(output_path)):
                output["items"][label_name] = previous
                skipped += 1
                print(f"{split} {number}/{len(rows)} {label_name}: already processed, skip", flush=True)
                continue

            item_started = time.perf_counter()
            result = server.run_fast_v2_1_pipeline(source)
            import base64
            encoded = base64.b64decode(result["model_input_jpeg_base64"], validate=True)
            temp_path = output_path.with_suffix(".jpg.tmp")
            temp_path.write_bytes(encoded)
            temp_path.replace(output_path)
            record = {"source_filename": source.name, "source_sha256": source_hash,
                      "label": item["label"], "cache_file": str(output_path.relative_to(HERE)),
                      "cache_sha256": sha(output_path), "crop_xyxy_source": result["intermediate"]["source_crop_xyxy"],
                      "prediction_using_reference_regressor": result["prediction"],
                      "preprocessing_stages_ms": result["stages_ms"],
                      "preprocessing_elapsed_ms": (time.perf_counter() - item_started) * 1000.0}
            output["items"][label_name] = record
            processed += 1
            manifest_path.write_text(json.dumps(output, indent=2) + "\n")
            print(f"{split} {number}/{len(rows)} {label_name}: prepared in "
                  f"{record['preprocessing_elapsed_ms']:.1f} ms; crop={record['crop_xyxy_source']}", flush=True)
        manifest_path.write_text(json.dumps(output, indent=2) + "\n")
        print(f"{split}: {processed} prepared, {skipped} reused, {len(rows)} total, "
              f"{time.perf_counter() - started:.2f}s; manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
