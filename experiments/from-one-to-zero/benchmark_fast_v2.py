#!/usr/bin/env python3
"""Baseline benchmark for the canonical checkpoint-v19 fast-v2 prediction path.

This runner imports the current pipeline and instruments it in-process. It does
not change pipeline source files and writes benchmark artifacts only here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DEFAULT_PHOTO_DIR = ROOT / "step-7-predict-captures-holder"
CHECKPOINT = ROOT / "step-5-checkpoints/v19-mae-0.1141-efficientnet_b0-fast-v2/wave-regression-efficientnet_b0-v19-best-val-mae-epoch=04-val_mae=0.1141.ckpt"
PROCESS_STARTED = time.perf_counter()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def ms_summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "median_ms": None, "p95_ms": None, "min_ms": None, "max_ms": None}
    ordered = sorted(values)
    return {
        "count": len(values), "median_ms": statistics.median(values),
        "p95_ms": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
        "min_ms": ordered[0], "max_ms": ordered[-1],
    }


def render_html(report: dict) -> str:
    payload = json.dumps(report, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    template = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light"><title>From One to Zero · Fast-v2 baseline</title><style>
:root{--ink:#142b38;--muted:#5b7078;--paper:#f4f7f4;--card:#fff;--line:#dce5e1;--navy:#153c4a;--teal:#237b78;--green:#208451;--shadow:0 10px 30px rgba(20,43,56,.07)}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:14px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}header{background:linear-gradient(120deg,#123642,#1c5960);color:#f8fbf8;padding:30px max(22px,calc((100vw - 1250px)/2))}header small{color:#9fd8c7;text-transform:uppercase;letter-spacing:.14em;font-weight:800}h1{font-size:clamp(28px,4vw,40px);line-height:1.1;margin:8px 0}.subtitle{color:#d4e5e0;max-width:850px;margin:0}.wrap{max-width:1250px;margin:auto;padding:24px}.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-bottom:16px}.card,.panel{background:var(--card);border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow)}.card{padding:15px 17px}.label{font-size:10px;font-weight:850;letter-spacing:.09em;text-transform:uppercase;color:var(--muted)}.value{font-size:25px;font-weight:780;margin:6px 0 2px;font-variant-numeric:tabular-nums}.note{font-size:11px;color:var(--muted)}.panel{overflow:hidden;margin:0 0 16px}.head{padding:15px 18px;border-bottom:1px solid var(--line)}h2{font-size:17px;margin:0}.head p{font-size:12px;color:var(--muted);margin:4px 0 0}.body{padding:15px 18px}.tablewrap{overflow:auto}table{width:100%;border-collapse:collapse;font-size:12px}th,td{padding:9px 8px;border-bottom:1px solid #edf1ef;text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}th{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}th:first-child,td:first-child{text-align:left;white-space:normal;min-width:250px}.barline{display:grid;grid-template-columns:minmax(220px,330px) minmax(100px,1fr) 80px;gap:12px;align-items:center;padding:8px 0;border-bottom:1px solid #edf1ef}.barname{font-size:12px;overflow-wrap:anywhere}.rail{height:9px;background:#edf2ef;border-radius:99px;overflow:hidden}.bar{height:100%;border-radius:inherit;background:linear-gradient(90deg,#43a98c,#237b78)}.duration{text-align:right;font-variant-numeric:tabular-nums;font-size:11px}.details{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.photo{background:#f7faf8;border:1px solid var(--line);border-radius:10px;padding:12px}.photo h3{font-size:13px;margin:0 0 6px}.photo p{font-size:11px;color:var(--muted);margin:3px 0;overflow-wrap:anywhere}.pill{display:inline-block;background:#edf5f2;border:1px solid #d4e6de;color:var(--navy);border-radius:999px;padding:3px 8px;font-size:10px;font-weight:800}details summary{cursor:pointer;font-weight:700;font-size:12px}pre{font-size:10px;white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f8f7;padding:10px;border-radius:8px}.foot{font-size:11px;color:var(--muted);padding:4px 2px 18px}@media(max-width:800px){.cards{grid-template-columns:repeat(2,minmax(0,1fr))}.barline{grid-template-columns:minmax(160px,1fr) 1fr 70px}}@media(max-width:540px){.wrap{padding:14px}.cards{grid-template-columns:1fr 1fr}.details{grid-template-columns:1fr}.barline{grid-template-columns:1fr 70px}.barline .rail{grid-column:1/-1;grid-row:2}.barname{grid-column:1}.duration{grid-column:2;grid-row:1}}
</style></head><body><header><small>Baseline preparation · Fast-v2</small><h1>Current prediction pipeline benchmark</h1><p class="subtitle">Checkpoint v19 · fresh raw-photo preprocessing through waviness score. Startup is reported separately. Timings come from the canonical prediction path; instrumentation is isolated under experiments/from-one-to-zero.</p></header><main class="wrap" id="app"></main><script id="data" type="application/json">__DATA__</script><script>
const d=JSON.parse(document.getElementById('data').textContent),app=document.getElementById('app');const esc=x=>String(x??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const ms=x=>x==null?'—':`${Number(x).toFixed(2)} ms`;const panel=(title,sub,body)=>`<section class="panel"><div class="head"><h2>${title}</h2><p>${sub}</p></div><div class="body">${body}</div></section>`;const ok=d.runs.filter(x=>x.status==='ok');const med=d.latency_summary.total_raw_to_score_ms.median_ms;const p95=d.latency_summary.total_raw_to_score_ms.p95_ms;const cards=[['Median raw-to-score',ms(med),'across measured fresh runs'],['p95 raw-to-score',ms(p95),'nearest-rank'],['Successful runs',`${ok.length}/${d.runs.length}`,'failed runs remain visible below'],['Segmentation device',esc(d.device),esc(d.segmentation_model)]];let html=`<div class="cards">${cards.map(c=>`<div class="card"><div class="label">${c[0]}</div><div class="value">${c[1]}</div><div class="note">${c[2]}</div></div>`).join('')}</div>`;const startups=Object.entries(d.startup_ms).map(([k,v])=>`<tr><td>${esc(k.replaceAll('_',' '))}</td><td>${ms(v)}</td></tr>`).join('');html+=panel('Startup and model loading','One-time work, excluded from per-photo latency',`<div class="tablewrap"><table><thead><tr><th>Stage</th><th>Elapsed</th></tr></thead><tbody>${startups}</tbody></table></div>`);const ops=Object.entries(d.operation_summary).sort((a,b)=>(b[1].median_ms||0)-(a[1].median_ms||0));const max=Math.max(1,...ops.map(([,v])=>v.median_ms||0));html+=panel('Detailed operation timings','Median and p95 across measured calls. Parent stage timings include their nested operations, so rows do not sum to raw-to-score.',ops.map(([k,v])=>`<div class="barline"><div class="barname">${esc(k)}</div><div class="rail"><div class="bar" style="width:${Math.max(1,(v.median_ms||0)/max*100)}%"></div></div><div class="duration">${ms(v.median_ms)}</div></div><div style="display:flex;justify-content:flex-end;color:#718188;font-size:10px;margin:-3px 0 5px">p95 ${ms(v.p95_ms)} · n=${v.count}</div>`).join(''));const runRows=d.runs.map(r=>`<tr><td>${esc(r.photo)}</td><td>${r.repeat}</td><td><span class="pill">${esc(r.status)}</span></td><td>${ms(r.total_raw_to_score_ms)}</td><td>${r.prediction==null?'—':Number(r.prediction).toFixed(5)}</td><td>${esc(r.error||'')}</td></tr>`).join('');html+=panel('Per-photo measured runs','One warm-up and three fresh measured predictions per photo',`<div class="tablewrap"><table><thead><tr><th>Photo</th><th>Run</th><th>Status</th><th>Raw-to-score</th><th>Prediction</th><th>Error</th></tr></thead><tbody>${runRows}</tbody></table></div>`);html+=panel('Photo cohort','Source photos were read in place and were not copied into this experiment.',`<div class="details">${d.photos.map(p=>`<article class="photo"><h3>${esc(p.filename)}</h3><p>${esc(p.path)}</p><p>${p.dimensions.join(' × ')} · ${esc(p.format)} · ${(p.size_bytes/1048576).toFixed(2)} MiB · SHA-256 ${esc(p.sha256)}</p></article>`).join('')}</div>`);html+=panel('Run configuration','Checkpoint, profile, runtime and measurement policy',`<details><summary>Show full benchmark metadata</summary><pre>${esc(JSON.stringify({checkpoint:d.checkpoint,checkpoint_sha256:d.checkpoint_sha256,profile:d.preprocessing_profile,device:d.device,protocol:d.protocol,notes:d.notes},null,2))}</pre></details>`);html+=`<p class="foot">Created ${esc(d.created_at_utc)}. GPU operations are synchronized at timing boundaries. No source photo, normal prediction preview, or normal prediction log was written by this benchmark.</p>`;app.innerHTML=html;
</script></body></html>'''
    return template.replace("__DATA__", payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("photos", nargs="*", type=Path, help="Raw photo paths; default is first two supported holder photos")
    parser.add_argument("--photo-dir", type=Path, default=DEFAULT_PHOTO_DIR)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--device", default=None, help="Defaults to the current pipeline's automatic device choice")
    args = parser.parse_args()
    if args.warmup < 1 or args.repeat < 1:
        parser.error("--warmup and --repeat must be positive")

    # Import canonical prediction code after fixing the project root for imports.
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import torch
    import cv2
    from PIL import Image
    import step_1_a_water_segmentation as segmentation
    import step_1_b_preprocess_water_inputs as step1
    import step_2_b_crop_selector as crop_selector
    import step_7_a_predict as prediction
    from step_1_c_preprocessing_profiles import configure_segmentation, get_profile

    photos = [p.resolve() for p in args.photos] if args.photos else sorted(
        (p.resolve() for p in args.photo_dir.iterdir()
         if p.is_file() and p.suffix.lower() in prediction.SUPPORTED_EXTENSIONS),
        key=lambda p: p.name.lower(),
    )[:2]
    if len(photos) != 2:
        raise RuntimeError(f"Expected exactly two benchmark photos; found {len(photos)}. Pass two explicit photo paths.")
    for photo in photos:
        if not photo.is_file() or photo.suffix.lower() not in prediction.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported or missing photo: {photo}")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Expected v19 checkpoint is missing: {CHECKPOINT}")

    started = time.perf_counter()
    device = segmentation.configure_inference_device(args.device)
    profile = get_profile("fast-v2")
    configure_segmentation(profile, segmentation)
    contract = prediction.validate_checkpoint_preprocessing(
        CHECKPOINT, segmentation=segmentation, device=device,
        step2_resolution=None, reuse_step1_mask=False, project_dir=ROOT,
        sha256_for=prediction.sha256_for, coordinate_crop=True,
    )
    if not contract or contract.get("profile") != "fast-v2":
        raise RuntimeError("Selected checkpoint does not declare the expected fast-v2 preprocessing contract")
    setup_ms = (time.perf_counter() - started) * 1000

    timings: dict[str, list[float]] = {}
    active: dict[str, object] = {"run": None, "resize_index": 0}

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def timed(name: str, operation: Callable, *values, **kwargs):
        sync()
        t0 = time.perf_counter()
        try:
            return operation(*values, **kwargs)
        finally:
            sync()
            run = active.get("run")
            if run is not None:
                run.setdefault("timings_ms", {}).setdefault(name, []).append((time.perf_counter() - t0) * 1000)

    # Model loads are reported separately from photo latency.
    t0 = time.perf_counter()
    segmenter = prediction.load_segmentation_model(str(device), verbose=False)
    prediction.validate_segmentation_revision(contract, segmenter)
    segmentation_load_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    model = prediction.WaveRegressionModel.load_from_checkpoint(CHECKPOINT, map_location=device, pretrained=False)
    model.to(device).eval()
    regressor_load_ms = (time.perf_counter() - t0) * 1000
    selector_warmup_ms = step1.warm_coordinate_crop_selector() * 1000
    transform = prediction.build_evaluation_transform()

    # Preserve and restore all monkey-patched callables so this runner remains
    # isolated even if it is imported into a longer-lived Python process.
    restorers: list[Callable[[], None]] = []

    def wrap(obj, attr: str, label: str, *, device_sync: bool = False):
        original = getattr(obj, attr)
        def measured(*values, **kwargs):
            if device_sync:
                return timed(label, original, *values, **kwargs)
            t0 = time.perf_counter()
            try:
                return original(*values, **kwargs)
            finally:
                run = active.get("run")
                if run is not None:
                    run.setdefault("timings_ms", {}).setdefault(label, []).append((time.perf_counter() - t0) * 1000)
        setattr(obj, attr, measured)
        restorers.append(lambda: setattr(obj, attr, original))

    # Fine-grained boundaries around the existing code path. Parent timings are
    # retained separately, so the report explains overlap instead of summing it.
    wrap(prediction, "load_rgb_image", "preprocessing.decode_and_orient_rgb")
    wrap(step1, "preprocess_coordinate_crop", "preprocessing.coordinate_crop_total")
    wrap(step1, "run_segmentation", "preprocessing.segmentation_total", device_sync=True)
    wrap(step1, "extract_water_mask", "preprocessing.extract_water_class_mask")
    wrap(crop_selector, "select_crop", "preprocessing.select_lower_water_rectangle")
    wrap(prediction, "encode_step_2_output", "preprocessing.encode_final_model_jpeg")
    wrap(prediction, "decode_rgb_bytes", "preprocessing.decode_final_model_jpeg")

    original_forward = model.forward
    def measured_regressor_forward(*values, **kwargs):
        return timed("waviness_inference.efficientnet_b0_forward", original_forward, *values, **kwargs)
    model.forward = measured_regressor_forward
    restorers.append(lambda: setattr(model, "forward", original_forward))

    # Separate the model-side sub-operations from segmentation orchestration.
    processor_type = type(segmenter.processor)
    processor_call = processor_type.__call__
    def measured_processor(self, *values, **kwargs):
        if self is not segmenter.processor:
            return processor_call(self, *values, **kwargs)
        return timed("preprocessing.segmentation_image_processor", processor_call, self, *values, **kwargs)
    processor_type.__call__ = measured_processor
    restorers.append(lambda: setattr(processor_type, "__call__", processor_call))
    wrap(segmentation, "model_logits", "preprocessing.segmentation_model_forward", device_sync=True)

    # Record the four resize calls in canonical order: class-map projection,
    # mask-to-selection grid, source RGB crop, and crop-mask projection.
    import cv2
    original_resize = cv2.resize
    def measured_resize(*values, **kwargs):
        run = active.get("run")
        index = int(active.get("resize_index", 0))
        labels = ["preprocessing.segmentation_class_map_projection",
                  "preprocessing.water_mask_to_crop_grid",
                  "preprocessing.resize_source_water_crop",
                  "preprocessing.resize_crop_mask"]
        label = labels[index] if index < len(labels) else "preprocessing.additional_cv2_resize"
        active["resize_index"] = index + 1
        return timed(label, original_resize, *values, **kwargs) if run is not None else original_resize(*values, **kwargs)
    cv2.resize = measured_resize
    restorers.append(lambda: setattr(cv2, "resize", original_resize))

    results = []
    try:
        def infer(model_input):
            return timed(
                "waviness_inference.model_input_to_score", prediction.predict,
                model_input, device, model,
                lambda image: timed("preprocessing.to_tensor_and_imagenet_normalization", transform, image),
            )

        for photo in photos:
            # One warm-up per photo, excluded from the measured distribution.
            for warm_index in range(args.warmup):
                active["run"] = {"timings_ms": {}}
                active["resize_index"] = 0
                model_input, _ = prediction.preprocess_photo(photo, segmenter, coordinate_crop=True)
                infer(model_input)
            for repeat_index in range(1, args.repeat + 1):
                record = {"photo": photo.name, "repeat": repeat_index, "status": "ok", "timings_ms": {}}
                active["run"] = record
                active["resize_index"] = 0
                photo_started = time.perf_counter()
                try:
                    model_input, encoded = prediction.preprocess_photo(photo, segmenter, coordinate_crop=True)
                    record["prediction"], record["tensor_shape"] = infer(model_input)
                    record["encoded_input_sha256"] = hashlib.sha256(encoded).hexdigest()
                    record["final_input_size"] = list(model_input.size)
                    sync()
                    record["total_raw_to_score_ms"] = (time.perf_counter() - photo_started) * 1000
                except Exception as error:
                    record.update(status="failed", error=f"{type(error).__name__}: {error}",
                                  total_raw_to_score_ms=(time.perf_counter() - photo_started) * 1000)
                results.append(record)
    finally:
        for restore in reversed(restorers):
            restore()
        active["run"] = None

    # Estimate the coordinate/canvas/mask application work not covered by the
    # explicit child timers. Inclusive parent timing remains available too.
    for record in results:
        t = record.get("timings_ms", {})
        def one(key: str) -> float:
            values = t.get(key, [])
            return float(values[0]) if values else 0.0
        residual = one("preprocessing.coordinate_crop_total") - sum(one(key) for key in (
            "preprocessing.segmentation_total", "preprocessing.extract_water_class_mask",
            "preprocessing.water_mask_to_crop_grid", "preprocessing.select_lower_water_rectangle",
            "preprocessing.resize_source_water_crop", "preprocessing.resize_crop_mask",
        ))
        t["preprocessing.coordinate_mapping_crop_canvas_and_mask_apply_other"] = [max(0.0, residual)]

    grouped: dict[str, list[float]] = {}
    for result in results:
        for name, samples in result.get("timings_ms", {}).items():
            grouped.setdefault(name, []).extend(samples)
    photo_specs = []
    for photo in photos:
        with Image.open(photo) as opened:
            dimensions = list(opened.size)
            image_format = opened.format
        photo_specs.append({"filename": photo.name, "path": str(photo), "sha256": digest(photo),
                            "size_bytes": photo.stat().st_size, "dimensions": dimensions,
                            "format": image_format})
    report = {
        "report_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "baseline preparation for a later speed-improvement experiment",
        "pipeline": "current fast-v2 prediction path", "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_sha256": digest(CHECKPOINT), "checkpoint_version": "v19",
        "preprocessing_profile": profile.settings(), "segmentation_model": segmentation.MODEL_NAME,
        "regression_backbone": model.backbone_name, "device": str(device),
        "protocol": {"warmup_runs_per_photo": args.warmup, "measured_runs_per_photo": args.repeat,
                     "photo_count": len(photos), "gpu_synchronization": device.type == "cuda"},
        "startup_ms": {"profile_setup": setup_ms, "segmentation_model_load": segmentation_load_ms,
                       "regression_checkpoint_load": regressor_load_ms,
                       "crop_selector_warmup": selector_warmup_ms,
                       "process_start_to_models_ready": (time.perf_counter() - PROCESS_STARTED) * 1000},
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "torchvision": __import__("torchvision").__version__,
                        "transformers": __import__("transformers").__version__,
                        "numpy": np.__version__, "opencv": cv2.__version__,
                        "pillow": Image.__version__, "cuda_runtime": torch.version.cuda,
                        "rocm_runtime": torch.version.hip,
                        "torch_threads": torch.get_num_threads(),
                        "accelerator": (torch.cuda.get_device_name(device) if device.type == "cuda" else None)},
        "photos": photo_specs,
        "runs": results,
        "latency_summary": {name: ms_summary([float(run[name]) for run in results
                                               if run.get("status") == "ok" and isinstance(run.get(name), (int, float))])
                            for name in ("total_raw_to_score_ms",)},
        "operation_summary": {name: ms_summary(values) for name, values in sorted(grouped.items())},
        "notes": ["Raw photo files were read in place and not copied into this experiment.",
                  "No normal prediction previews or prediction logs were written.",
                  "Parent operation timings overlap their child operation timings.",
                  "Per-photo timing is fresh preprocessing plus prediction; startup/model loading is separate."],
    }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = OUT / f"baseline-{timestamp}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    html_path = OUT / f"baseline-{timestamp}.html"
    html_path.write_text(render_html(report), encoding="utf-8")
    print(report_path)
    print(html_path)
    print(f"successful runs: {sum(r['status'] == 'ok' for r in results)}/{len(results)}")


if __name__ == "__main__":
    main()
