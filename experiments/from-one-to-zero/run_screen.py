#!/usr/bin/env python3
"""Run paired, small-cohort speed screens for From One to Zero candidates."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = Path(__file__).resolve().parent


def configured(name: str, default: str) -> str:
    env = ROOT / ".env"
    if env.is_file():
        for raw in env.read_text().splitlines():
            key, sep, value = raw.partition("=")
            if sep and key.strip() == name:
                return value.strip().strip("\"'")
    return os.environ.get(name, default)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--photos", nargs="*", type=Path)
    parser.add_argument("--only", choices=("migraphx-fp16", "profile-step1", "smaller-input", "heic-lowres-roi"),
                        help="Run one candidate, including a fresh paired baseline measurement")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    photos = args.photos or sorted((ROOT / "step-7-predict-captures-holder").glob("*.HEIC"), key=lambda p: p.name.casefold())
    photos = photos[:4]
    if len(photos) < 2 or any(not p.is_file() for p in photos):
        raise FileNotFoundError("Need at least two existing HEIC photos for a paired screening run")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    runtime_python = Path(configured("PREDICT_MIGRAPHX_PYTHON", "/tmp/wave_migraphx_312_env/bin/python")).expanduser()
    if not runtime_python.is_file():
        raise FileNotFoundError(f"MIGraphX runtime Python is missing: {runtime_python}")
    runtime_root = runtime_python.parent.parent
    migraphx_libs = runtime_root / "lib/python3.12/site-packages/migraphx_libs"
    prior_ld = os.environ.get("LD_LIBRARY_PATH", "")
    ld_parts = [str(p) for p in (migraphx_libs, Path("/opt/rocm/lib")) if p.is_dir()]
    if prior_ld:
        ld_parts.append(prior_ld)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(ld_parts)
    model384 = Path("/tmp") / f"wave-from-zero-segformer-b0-384-{stamp}.onnx"
    subprocess.run([sys.executable, str(EXPERIMENTS / "smaller-input/rewrite_onnx_shape.py"),
                    str(ROOT / "models/fast-v2.1/segformer-b0-step1-448-fp32.onnx"), str(model384)], check=True)
    profile_prefix = Path("/tmp") / f"wave-from-zero-step1-profile-{stamp}"

    candidates = [
        ("migraphx-fp16", "migraphx-fp16", 448, None, None),
        ("profile-step1", "profile-step1", 448, None, profile_prefix),
        ("smaller-input", "smaller-input-384", 384, model384, None),
        ("heic-lowres-roi", "heic-lowres-roi", 448, None, None),
    ]
    if args.only:
        selected = [v for v in candidates if v[0] == args.only]
    else:
        selected = candidates
    variants = [("baseline", "baseline", 448, None, None), *selected]
    for directory, variant, input_size, model, profile in variants:
        out = EXPERIMENTS / directory / f"screen-{stamp}.json"
        command = [str(runtime_python), str(EXPERIMENTS / "run_screen_runtime.py"),
                   "--variant", variant, "--output", str(out), "--step1-size", str(input_size),
                   "--repeats", str(args.repeats), "--photos", *map(str, photos)]
        if model:
            command += ["--step1-model", str(model)]
        if profile:
            command += ["--profile-prefix", str(profile)]
        print(f"\n=== {variant}: {len(photos)} photos × {args.repeats} measured runs ===", flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)

    summary = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
               "photo_count": len(photos), "photos": [p.name for p in photos],
               "measured_repeats_per_photo": args.repeats, "screen_stamp": stamp,
               "quality_gate": "Not run; predictions and crop deltas are diagnostic only. A small screen does not establish quality.",
               "variants": [variant for _, variant, *_ in variants]}
    index = EXPERIMENTS / "screen-runs.json"
    history = json.loads(index.read_text()) if index.is_file() else {"runs": []}
    history["runs"].append(summary)
    index.write_text(json.dumps(history, indent=2) + "\n")
    subprocess.run([sys.executable, str(EXPERIMENTS / "generate_index.py")], check=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
