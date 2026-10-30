#!/usr/bin/env python3
"""Render all versioned Distil-FP32 benchmark reports into a standalone dashboard."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def small_report(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    experiment_id = report.get("experiment_id", "v000_baseline")
    manifest_path = HERE / "experiment-versions" / experiment_id / "experiment.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    model_value = report.get("environment", {}).get("regressor_model")
    model_path = Path(model_value) if model_value else None
    try:
        regressor_model = str(model_path.resolve().relative_to(HERE)) if model_path else "unspecified"
    except ValueError:
        regressor_model = model_path.name if model_path else "unspecified"
    return {
        "experiment_id": experiment_id,
        "experiment_name": report.get("experiment_name") or manifest.get("name") or experiment_id,
        "experiment_status": report.get("experiment_status") or manifest.get("status", "historical"),
        "hypothesis": report.get("experiment_hypothesis") or manifest.get("hypothesis"),
        "created_at_utc": report.get("created_at_utc", ""),
        "report_name": path.name,
        "regressor_model": regressor_model,
        "splits": report.get("dataset", {}).get("splits", list(report.get("quality_by_split", {}))),
        "quality_by_split": report.get("quality_by_split", {}),
        "latency_by_split": report.get("latency_by_split", {}),
        "warmup": report.get("warmup", {}),
        "photos": [
            {
                "split": row.get("split"),
                "filename": row.get("label_filename"),
                "source_filename": row.get("source_filename"),
                "source_sha256": row.get("source_sha256"),
                "label": row.get("label"),
                "prediction": row.get("prediction"),
                "total_ms": row.get("total_raw_to_score_ms"),
                "decode_ms": row.get("timings_ms", {}).get("preprocessing.decode_oriented_rgb"),
                "step1_ms": row.get("timings_ms", {}).get("preprocessing.step1.segmentation_inference"),
                "step2_ms": row.get("timings_ms", {}).get("preprocessing.step2.segmentation_inference"),
                "failure": row.get("failure"),
            }
            for row in report.get("photos", [])
        ],
        "failure_count": report.get("failure_count", 0),
        "comparison": report.get("comparison"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path,
                        help="include a specific report (repeatable); default discovers all reports under runs/")
    parser.add_argument("--output", type=Path, default=HERE / "distil-fp32-benchmark-report.html")
    args = parser.parse_args()
    paths = [path.resolve() for path in args.input] if args.input else sorted((HERE / "runs").rglob("distil-fp32-fine-grained-*.json"))
    if not paths:
        raise FileNotFoundError("No benchmark reports found. Run benchmark_distil_fp32.py first.")
    reports = [small_report(path) for path in paths if path.is_file()]
    if not reports:
        raise FileNotFoundError("None of the selected benchmark report files exist.")
    reports.sort(key=lambda report: (report["experiment_id"], report["created_at_utc"]))
    data = json.dumps({"reports": reports}, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    template = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="color-scheme" content="light">
<title>Distil-FP32 · Performance experiments</title>
<style>
:root{--ink:#142b38;--muted:#5b7078;--paper:#f4f7f4;--card:#fff;--line:#dce5e1;--navy:#153c4a;--teal:#237b78;--green:#208451;--amber:#b77914;--orange:#c25b19;--red:#bd3f38;--shadow:0 10px 30px rgba(20,43,56,.07)}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}header{background:linear-gradient(120deg,#123642,#1c5960);color:#f8fbf8;padding:34px max(24px,calc((100vw - 1320px)/2) ) 30px}.eyebrow{text-transform:uppercase;letter-spacing:.14em;font-size:11px;font-weight:800;color:#9fd8c7}h1{font-size:clamp(28px,4vw,42px);line-height:1.1;margin:8px 0 10px;letter-spacing:-.035em}.subtitle{max-width:850px;color:#d4e5e0;margin:0}.meta{font-size:12px;color:#b6d2ce;margin-top:18px}.wrap{max-width:1320px;margin:auto;padding:26px 24px 60px}.toolbar{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:0 0 18px}.control{display:flex;align-items:center;gap:8px;color:var(--muted);font-size:12px;font-weight:750}select{font:inherit;color:var(--ink);background:#fff;border:1px solid var(--line);border-radius:9px;padding:8px 28px 8px 10px}.spacer{flex:1}.legend{display:flex;gap:11px;align-items:center;flex-wrap:wrap;color:var(--muted);font-size:11px}.key{display:inline-flex;align-items:center;gap:5px}.dot{width:9px;height:9px;border-radius:50%;display:inline-block}.dot.low{background:var(--green)}.dot.medium{background:var(--amber)}.dot.high{background:var(--orange)}.dot.hot{background:var(--red)}.panel,.card{background:var(--card);border:1px solid var(--line);border-radius:16px;box-shadow:var(--shadow)}.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:13px;margin:0 0 18px}.card{padding:16px 18px}.card-label{font-size:10px;font-weight:850;letter-spacing:.09em;text-transform:uppercase;color:var(--muted)}.card-value{font-size:26px;line-height:1.2;font-weight:780;letter-spacing:-.03em;margin:7px 0 4px;font-variant-numeric:tabular-nums}.card-note{font-size:11px;color:var(--muted)}.quality-value{color:var(--teal)}.sections{display:grid;gap:17px}.panel{overflow:hidden;margin-bottom:17px}.panel:last-child{margin-bottom:0}.panel-head{padding:17px 20px;border-bottom:1px solid var(--line);display:flex;align-items:flex-start;justify-content:space-between;gap:16px}.panel-head h2{margin:0;font-size:18px;letter-spacing:-.02em}.panel-head p{margin:4px 0 0;color:var(--muted);font-size:12px}.pill{background:#edf5f2;border:1px solid #d4e6de;color:var(--navy);font-weight:800;border-radius:999px;padding:6px 10px;white-space:nowrap;font-size:12px}.panel-body{padding:16px 20px 19px}.group{margin:0 0 18px}.group:last-child{margin-bottom:0}.group-title{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:#42616a;font-weight:850;margin:0 0 7px}.row,.table-head{display:grid;grid-template-columns:minmax(220px,300px) minmax(80px,1fr) 72px 72px 72px;gap:12px;align-items:center}.table-head{padding:0 0 6px;color:#73858a;font-size:9px;text-transform:uppercase;letter-spacing:.08em;font-weight:850}.table-head span:nth-child(n+3){text-align:right}.row{padding:8px 0;border-bottom:1px solid #eef2ef}.row:last-child{border-bottom:0}.name{font-size:12px;font-weight:650}.bar-rail{height:10px;border-radius:99px;background:#eef2ee;overflow:hidden}.bar{height:100%;min-width:2px;border-radius:inherit}.bar.low{background:linear-gradient(90deg,#4cac74,#208451)}.bar.medium{background:linear-gradient(90deg,#edc04e,#b77914)}.bar.high{background:linear-gradient(90deg,#f3a145,#c25b19)}.bar.hot{background:linear-gradient(90deg,#ec7660,#bd3f38)}.num{text-align:right;font-size:11px;font-variant-numeric:tabular-nums;color:#38525c}.num strong{font-size:12px;color:var(--ink)}.max{color:#63767c}.warm-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:7px 24px}.warm-row{display:grid;grid-template-columns:1fr auto;gap:12px;padding:8px 0;border-bottom:1px solid #eef2ef;font-size:12px}.warm-row strong{font-variant-numeric:tabular-nums}.callout{border-left:4px solid var(--teal);padding:10px 13px;background:#eff7f4;border-radius:0 9px 9px 0;color:#36545b;font-size:12px;margin-top:14px}.quality-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:11px}.quality-box{border:1px solid var(--line);border-radius:11px;padding:12px}.quality-box h3{margin:0 0 8px;font-size:13px}.quality-metrics{display:flex;gap:18px;flex-wrap:wrap}.quality-metrics strong{font-size:18px;display:block;font-variant-numeric:tabular-nums}.quality-metrics span{font-size:10px;color:var(--muted)}.notice{padding:11px 13px;background:#fff5df;border:1px solid #f0ddb0;border-radius:10px;color:#71541e;font-size:12px;margin-bottom:14px}.ok{color:var(--green);font-weight:800}.bad{color:var(--red);font-weight:800}.muted{color:var(--muted)}.compare-select{display:flex;gap:8px;flex-wrap:wrap}.check{display:inline-flex;align-items:center;gap:5px;background:#fff;border:1px solid var(--line);border-radius:8px;padding:6px 8px;font-size:11px;color:var(--ink)}.compare-wrap{overflow:auto}.compare-table{border-collapse:collapse;width:100%;min-width:760px;font-size:12px}.compare-table th,.compare-table td{border-bottom:1px solid #e8eeeb;padding:9px 10px;text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}.compare-table th{background:#f7faf8;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.05em;position:sticky;top:0}.compare-table th:first-child,.compare-table td:first-child{text-align:left;position:sticky;left:0;background:#fff}.compare-table .best{color:var(--green);font-weight:850}.op-compare{margin-top:18px}.op-bars{display:flex;gap:4px;min-width:100px;align-items:center;justify-content:flex-end}.mini-bar{height:8px;min-width:2px;border-radius:5px}.footer{margin-top:16px;color:var(--muted);font-size:10px}.slow table{width:100%;border-collapse:collapse;font-size:11px}.slow th,.slow td{padding:7px;border-bottom:1px solid #e9efec;text-align:right}.slow th:first-child,.slow td:first-child{text-align:left}.slow th{color:var(--muted);text-transform:uppercase;font-size:9px;letter-spacing:.05em}.empty{padding:15px;color:var(--muted);font-size:13px}
@media(max-width:850px){.cards{grid-template-columns:repeat(2,minmax(0,1fr))}.row,.table-head{grid-template-columns:minmax(170px,230px) minmax(65px,1fr) 62px 62px 62px;gap:7px}.max{display:none}.warm-grid{grid-template-columns:1fr}}
@media(max-width:600px){.wrap{padding:17px 11px 40px}header{padding:25px 16px}.cards{gap:7px}.card{padding:12px}.card-value{font-size:20px}.panel-head,.panel-body{padding:13px}.row,.table-head{grid-template-columns:minmax(122px,1.4fr) minmax(44px,.8fr) 54px 54px;gap:5px}.name{font-size:10px}.num{font-size:9px}.num strong{font-size:10px}.quality-grid{grid-template-columns:1fr}.toolbar{align-items:flex-start}.spacer{display:none}}
</style></head><body>
<header><div class="eyebrow">Performance investigation · Distil-FP32</div><h1>Compare performance experiments</h1><p class="subtitle">Inspect one experiment’s full timing tree, or compare latency and waviness quality side by side. Validation is for selecting changes; use the reserved test split only for final checks.</p><div class="meta" id="meta"></div></header>
<main class="wrap">
<div class="toolbar">
 <label class="control">View <select id="mode"><option value="individual">Individual experiment</option><option value="compare">Side-by-side comparison</option></select></label>
 <label class="control">Split <select id="split"><option value="validation">Validation · selection</option><option value="test">Test · final check</option><option value="whole">Whole · all labeled photos</option></select></label>
 <label class="control" id="experiment-control">Experiment <select id="experiment"></select></label>
 <div class="compare-select" id="compare-select" hidden></div><div class="spacer"></div>
 <div class="legend"><span class="key"><i class="dot low"></i>&lt;1 ms</span><span class="key"><i class="dot medium"></i>1–5 ms</span><span class="key"><i class="dot high"></i>5–15 ms</span><span class="key"><i class="dot hot"></i>&gt;15 ms</span></div>
</div>
<div id="test-notice" class="notice" hidden>Final-check split selected. Do not use these results to choose or tune an experiment.</div>
<div id="whole-notice" class="notice" hidden>Whole cohort diagnostic: includes train, validation, and test photos. It is not an independent evaluation split and must not be used for model selection.</div>
<div id="individual-view">
 <section class="cards" id="cards"></section>
 <section class="panel"><div class="panel-head"><div><h2>1 · Warm-up</h2><p>One-time startup cost before photo processing begins.</p></div><div class="pill" id="warm-total"></div></div><div class="panel-body"><div class="warm-grid" id="warm-rows"></div><div class="callout">The first segmentation graph runs include MIGraphX setup and compilation. They are startup costs, not per-photo inference times.</div></div></section>
 <section class="panel"><div class="panel-head"><div><h2>2 · Pre-processing</h2><p>Raw photo decode through creation of the normalized final regressor tensor.</p></div><div class="pill" id="pre-total"></div></div><div class="panel-body" id="pre-rows"></div></section>
 <section class="panel"><div class="panel-head"><div><h2>3 · Waviness inference</h2><p>The v18 EfficientNet-B0 model produces the waviness score.</p></div><div class="pill" id="infer-total"></div></div><div class="panel-body" id="infer-rows"></div></section>
 <section class="panel"><div class="panel-head"><div><h2>Labeled-photo quality</h2><p>Ground-truth waviness error and request success coverage.</p></div><div id="quality-status"></div></div><div class="panel-body"><div class="quality-grid" id="quality"></div></div></section>
 <section class="panel slow"><div class="panel-head"><div><h2>Slowest photos</h2><p>Decode and segmentation model times for the ten slowest photos in the selected split.</p></div></div><div class="panel-body"><details><summary>Show per-photo tail latency</summary><table><thead><tr><th>Photo</th><th>Label</th><th>Total</th><th>Decode</th><th>Step 1</th><th>Step 2</th><th>Prediction</th></tr></thead><tbody id="slow-photos"></tbody></table></details></div></section>
</div>
<section class="panel" id="compare-view" hidden><div class="panel-head"><div><h2>Side-by-side results</h2><p>Latest report for each selected version that contains the chosen split.</p></div><div class="pill" id="compare-split"></div></div><div class="panel-body"><div id="cohort-notice"></div><div class="compare-wrap"><table class="compare-table"><thead id="compare-head"></thead><tbody id="compare-body"></tbody></table></div><div class="op-compare"><h3 class="group-title">Operation medians · lower bars are faster</h3><div class="compare-wrap"><table class="compare-table"><thead id="op-head"></thead><tbody id="op-body"></tbody></table></div></div></div></section>
<p class="footer">Latency is local raw-file-to-score; HTTP upload and temporary request-file staging are excluded. Parent totals overlap child operations. Operation medians are independently summarized and will not sum exactly to the median total. Reports identify the source photo cohort by filename and SHA-256 where available.</p>
</main>
<script id="report-data" type="application/json">__REPORT_DATA__</script>
<script>
const DATA=JSON.parse(document.getElementById('report-data').textContent).reports;
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const ms=x=>x==null?'—':`${Number(x).toFixed(2)} ms`,sec=x=>x==null?'—':x>=1000?`${(x/1000).toFixed(2)} s`:`${Number(x).toFixed(1)} ms`,q=x=>x==null?'—':Number(x).toFixed(5);
const grade=x=>x>15?'hot':x>5?'high':x>=1?'medium':'low';
const byVersion=[...new Map(DATA.map(r=>[r.experiment_id,{id:r.experiment_id,name:r.experiment_name,status:r.experiment_status}])).values()].sort((a,b)=>a.id.localeCompare(b.id));
const names={
 'preprocessing.decode_oriented_rgb':'Decode and orient photo','preprocessing.step1.tensor.resize_rgb':'Resize raw RGB photo to 448 × 448','preprocessing.step1.tensor.uint8_to_float32_and_scale':'Convert Step 1 pixels to FP32 and scale','preprocessing.step1.tensor.normalize_mean_std':'Normalize Step 1 input','preprocessing.step1.tensor.transpose_and_contiguous_tensor':'Arrange contiguous Step 1 tensor','preprocessing.step1.segmentation_inference':'Step 1 water segmentation inference','preprocessing.step1.expand_mask_to_source':'Expand water mask to source dimensions','preprocessing.step1.erode_mask':'Erode full-size water mask','preprocessing.step1.bounding_box':'Find water bounding box','preprocessing.step1.water_pixel_count_and_validation':'Count water pixels and validate mask','preprocessing.step1.roi_views':'Create ROI views','preprocessing.step1.masked_roi_copy':'Apply mask and copy water ROI','preprocessing.step1.pil_image_from_masked_roi':'Create PIL image from masked ROI','preprocessing.step1.canvas.lanczos_resize':'Lanczos resize water ROI','preprocessing.step1.canvas.black_canvas_allocate_and_paste':'Allocate black canvas and paste ROI','preprocessing.step1.jpeg_encode':'Encode Step 1 JPEG','preprocessing.step1.jpeg_decode_and_rgb_convert':'Decode Step 1 JPEG and convert to RGB','preprocessing.step2.black_area_mask':'Identify black canvas area','preprocessing.step2.tensor.resize_rgb':'Resize Step 1 image to 384 × 384','preprocessing.step2.tensor.uint8_to_float32_and_scale':'Convert Step 2 pixels to FP32 and scale','preprocessing.step2.tensor.normalize_mean_std':'Normalize Step 2 input','preprocessing.step2.tensor.transpose_and_contiguous_tensor':'Arrange contiguous Step 2 tensor','preprocessing.step2.segmentation_inference':'Step 2 water segmentation inference','preprocessing.step2.expand_mask_to_canvas':'Expand Step 2 mask to canvas','preprocessing.step2.mask_convert_and_exclude_black':'Convert mask and exclude black pixels','preprocessing.step2.crop_selection':'Select lower-water crop','preprocessing.step2.crop_extract':'Extract selected crop','preprocessing.step2.canvas.lanczos_resize':'Resize selected crop','preprocessing.step2.canvas.black_canvas_allocate_and_paste':'Allocate final canvas and paste crop','preprocessing.final.jpeg_encode':'Encode final model-input JPEG','preprocessing.final.jpeg_decode_and_rgb_convert':'Decode final JPEG and convert to RGB','preprocessing.final.rgb_to_float32_scale':'Convert final RGB to FP32 and scale','preprocessing.final.normalize_mean_std':'Normalize regressor input','preprocessing.final.transpose_and_contiguous_tensor':'Arrange final regressor tensor','waviness_inference.v18_regressor_onnxruntime':'Waviness regressor ONNX Runtime call','waviness_inference.score_extraction':'Extract scalar waviness score'};
const groups=[
 {title:'Input decode',test:k=>k==='preprocessing.decode_oriented_rgb'},
 {title:'Step 1 · Prepare and segment raw photo',test:k=>k.includes('preprocessing.step1.tensor')||k==='preprocessing.step1.segmentation_inference'},
 {title:'Step 1 · Expand mask, isolate sea, create canvas',test:k=>k.startsWith('preprocessing.step1.')&&!k.includes('.tensor.')&&k!=='preprocessing.step1.segmentation_inference'},
 {title:'Step 2 · Prepare and segment Step 1 image',test:k=>k.includes('preprocessing.step2.tensor')||k==='preprocessing.step2.segmentation_inference'},
 {title:'Step 2 · Select lower water and create canvas',test:k=>k.startsWith('preprocessing.step2.')&&!k.includes('.tensor.')&&k!=='preprocessing.step2.segmentation_inference'},
 {title:'Final regressor input · JPEG and tensor preparation',test:k=>k.startsWith('preprocessing.final.')}
];
const mode=document.getElementById('mode'),splitSelect=document.getElementById('split'),versionSelect=document.getElementById('experiment');
function reportsFor(id,split){return DATA.filter(r=>r.experiment_id===id&&r.splits.includes(split)&&r.latency_by_split[split]).sort((a,b)=>b.created_at_utc.localeCompare(a.created_at_utc))}
function selectedReport(){return reportsFor(versionSelect.value,splitSelect.value)[0]||null}
function initSelectors(){versionSelect.innerHTML=byVersion.map(v=>`<option value="${esc(v.id)}">${esc(v.id)} · ${esc(v.name)}</option>`).join('');document.getElementById('compare-select').innerHTML=byVersion.map(v=>`<label class="check"><input type="checkbox" value="${esc(v.id)}" checked>${esc(v.id)}</label>`).join('');}
function card(label,value,note,quality=false){return `<div class="card"><div class="card-label">${label}</div><div class="card-value ${quality?'quality-value':''}">${value}</div><div class="card-note">${note}</div></div>`}
function warmRows(w){const fields=[['MIGraphX provider registration',w.runtime_registration_ms],['Load Step 1 segmentation session',w.step1_session_load_ms],['Load Step 2 segmentation session',w.step2_session_load_ms],['Load waviness regressor session',w.regressor_session_load_ms],['First Step 1 graph run · 448 × 448',w.warm_448_first_run_ms],['First Step 2 graph run · 384 × 384',w.warm_384_first_run_ms],['First waviness regressor run',w.warm_regressor_ms],['Compile crop selector',w.warm_crop_selector_ms],['Python and runtime imports',w.python_and_runtime_imports_ms]];return fields.map(x=>`<div class="warm-row"><span>${x[0]}</span><strong>${sec(x[1])}</strong></div>`).join('')+`<div class="warm-row"><span>All model initialization and warm-up</span><strong>${sec(w.initialize_models_total_ms)}</strong></div>`}
function renderOps(report,split){const ops=report.latency_by_split[split].operations||{},keys=Object.keys(ops);return groups.map(g=>{const groupKeys=keys.filter(g.test).sort((a,b)=>ops[b].median_ms-ops[a].median_ms);if(!groupKeys.length)return '';const max=Math.max(1,...groupKeys.map(k=>ops[k].median_ms||0));const rows=groupKeys.map(k=>{const s=ops[k],m=s.median_ms||0;return `<div class="row"><div class="name">${esc(names[k]||k)}</div><div class="bar-rail"><div class="bar ${grade(m)}" style="width:${Math.max(1,m/max*100)}%"></div></div><div class="num"><strong>${ms(m)}</strong></div><div class="num">${ms(s.p95_ms)}</div><div class="num max">${ms(s.max_ms)}</div></div>`}).join('');return `<div class="group"><h3 class="group-title">${g.title}</h3><div class="table-head"><span>Operation</span><span>Relative time</span><span>Median</span><span>p95</span><span class="max">Max</span></div>${rows}</div>`}).join('')}
function renderInference(report,split){const ops=report.latency_by_split[split].operations||{},keys=Object.keys(ops).filter(k=>k.startsWith('waviness_inference.')).sort((a,b)=>ops[b].median_ms-ops[a].median_ms);if(!keys.length)return '<div class="empty">No inference operation timings.</div>';const max=Math.max(1,...keys.map(k=>ops[k].median_ms||0));return `<div class="table-head"><span>Operation</span><span>Relative time</span><span>Median</span><span>p95</span><span class="max">Max</span></div>`+keys.map(k=>{const s=ops[k],m=s.median_ms||0;return `<div class="row"><div class="name">${esc(names[k]||k)}</div><div class="bar-rail"><div class="bar ${grade(m)}" style="width:${Math.max(1,m/max*100)}%"></div></div><div class="num"><strong>${ms(m)}</strong></div><div class="num">${ms(s.p95_ms)}</div><div class="num max">${ms(s.max_ms)}</div></div>`}).join('')}
function renderIndividual(){const r=selectedReport(),split=splitSelect.value;document.getElementById('individual-view').hidden=mode.value!=='individual';document.getElementById('compare-view').hidden=mode.value!=='compare';document.getElementById('experiment-control').hidden=mode.value!=='individual';document.getElementById('compare-select').hidden=mode.value!=='compare';document.getElementById('test-notice').hidden=split!=='test';document.getElementById('whole-notice').hidden=split!=='whole';if(mode.value!=='individual')return;if(!r){document.getElementById('cards').innerHTML='<div class="empty">No report for this experiment and split yet.</div>';for(const id of ['warm-rows','pre-rows','infer-rows','quality','slow-photos'])document.getElementById(id).innerHTML='';document.getElementById('warm-total').textContent='—';document.getElementById('pre-total').textContent='—';document.getElementById('infer-total').textContent='—';return}
 const l=r.latency_by_split[split],qv=r.quality_by_split[split],tot=l.total_raw_to_score_ms,pre=l.preprocessing_total_ms,inf=l.waviness_inference_total_ms;
 document.getElementById('meta').textContent=`${r.experiment_id} · ${r.experiment_name} · regressor ${r.regressor_model} · ${r.created_at_utc} · ${r.report_name}`;
 document.getElementById('cards').innerHTML=[card('Raw → score',ms(tot.median_ms),`p95 ${ms(tot.p95_ms)} · max ${ms(tot.max_ms)}`),card('Pre-processing',ms(pre.median_ms),`p95 ${ms(pre.p95_ms)} · ${((pre.median_ms/tot.median_ms)*100).toFixed(1)}% of median total`),card('Waviness inference',ms(inf.median_ms),`p95 ${ms(inf.p95_ms)} · ${((inf.median_ms/tot.median_ms)*100).toFixed(1)}% of median total`),card('Quality · MAE / RMSE',`${q(qv.mae)} / ${q(qv.rmse)}`,`${qv.n} succeeded · ${qv.failures} failures`,true)].join('');
 document.getElementById('warm-total').textContent=`Ready in ${sec(r.warmup?.process_start_to_models_ready_ms)}`;document.getElementById('warm-rows').innerHTML=warmRows(r.warmup||{});document.getElementById('pre-total').textContent=`Median ${ms(pre.median_ms)} · p95 ${ms(pre.p95_ms)}`;document.getElementById('pre-rows').innerHTML=renderOps(r,split);document.getElementById('infer-total').textContent=`Median ${ms(inf.median_ms)} · p95 ${ms(inf.p95_ms)}`;document.getElementById('infer-rows').innerHTML=renderInference(r,split);
 document.getElementById('quality-status').innerHTML=qv.failures?`<span class="bad">${qv.failures} failures</span>`:`<span class="ok">${qv.n} / ${r.photos.filter(p=>p.split===split).length} succeeded</span>`;
 document.getElementById('quality').innerHTML=Object.entries(r.quality_by_split).map(([s,v])=>`<div class="quality-box"><h3>${esc(s)} · ${v.n} photos</h3><div class="quality-metrics"><div><strong>${q(v.mae)}</strong><span>MAE</span></div><div><strong>${q(v.rmse)}</strong><span>RMSE</span></div><div><strong>${v.failures}</strong><span>failures</span></div></div></div>`).join('');
 const slow=r.photos.filter(p=>p.split===split).sort((a,b)=>(b.total_ms||0)-(a.total_ms||0)).slice(0,10);document.getElementById('slow-photos').innerHTML=slow.map(p=>`<tr><td>${esc(p.source_filename||p.filename)}</td><td>${q(p.label)}</td><td>${ms(p.total_ms)}</td><td>${ms(p.decode_ms)}</td><td>${ms(p.step1_ms)}</td><td>${ms(p.step2_ms)}</td><td>${p.prediction==null?'—':q(p.prediction)}</td></tr>`).join('');
}
function checkedVersions(){return [...document.querySelectorAll('#compare-select input:checked')].map(x=>x.value)}
function signature(r,s){return r.photos.filter(p=>p.split===s).map(p=>`${p.filename}|${p.label}|${p.source_sha256||''}`).sort().join('\n')}
function renderCompare(){if(mode.value!=='compare')return;const split=splitSelect.value,ids=checkedVersions(),selected=ids.map(id=>({id,report:reportsFor(id,split)[0]||null}));const available=selected.filter(x=>x.report),refs=available.map(x=>signature(x.report,split));document.getElementById('compare-split').textContent=split==='test'?'FINAL CHECK':split==='whole'?'WHOLE · DIAGNOSTIC':'VALIDATION';document.getElementById('meta').textContent=`${available.length} experiment version(s) · models: ${available.map(x=>`${x.id}=${x.report.regressor_model}`).join('; ')}`;document.getElementById('cohort-notice').innerHTML=available.length>1?(refs.every(x=>x===refs[0])?'<div class="notice"><span class="ok">Matched cohort:</span> all selected reports use the same labeled photos and source fingerprints.</div>':'<div class="notice"><span class="bad">Cohort mismatch:</span> selected reports do not contain the same photos, labels, or source fingerprints. Compare timing only with caution; quality metrics are not directly paired.</div>'):'';
 const head=['Experiment','Photos','Preprocess median','Preprocess p95','Regressor median','Total median','Total p95','MAE','RMSE','Failures'];document.getElementById('compare-head').innerHTML=`<tr>${head.map(h=>`<th>${h}</th>`).join('')}</tr>`;
 document.getElementById('compare-body').innerHTML=selected.map(({id,report:r})=>{if(!r)return `<tr><td>${esc(id)}</td><td colspan="9" class="muted">No ${esc(split)} report yet</td></tr>`;const l=r.latency_by_split[split],qv=r.quality_by_split[split];return `<tr><td>${esc(id)} · ${esc(r.experiment_name)}<br><small>${esc(r.regressor_model)}</small></td><td>${qv.n}</td><td>${ms(l.preprocessing_total_ms.median_ms)}</td><td>${ms(l.preprocessing_total_ms.p95_ms)}</td><td>${ms(l.waviness_inference_total_ms.median_ms)}</td><td><strong>${ms(l.total_raw_to_score_ms.median_ms)}</strong></td><td>${ms(l.total_raw_to_score_ms.p95_ms)}</td><td>${q(qv.mae)}</td><td>${q(qv.rmse)}</td><td>${qv.failures}</td></tr>`}).join('')||'<tr><td colspan="10">Select one or more experiment versions.</td></tr>';
 const keys=[...new Set(available.flatMap(x=>Object.keys(x.report.latency_by_split[split].operations||{}).filter(k=>!k.startsWith('waviness_inference.'))))].sort();document.getElementById('op-head').innerHTML=`<tr><th>Preprocessing operation</th>${available.map(x=>`<th>${esc(x.id)}<br>median · ms</th>`).join('')}</tr>`;const opValues=keys.map(k=>({k,values:available.map(x=>x.report.latency_by_split[split].operations?.[k]?.median_ms??null)})).sort((a,b)=>Math.max(...b.values.map(v=>v??-1))-Math.max(...a.values.map(v=>v??-1)));document.getElementById('op-body').innerHTML=opValues.map(({k,values})=>`<tr><td style="text-align:left">${esc(names[k]||k)}</td>${values.map(v=>`<td class="${v==null?'muted':grade(v)==='hot'?'bad':''}">${ms(v)}</td>`).join('')}</tr>`).join('')||`<tr><td colspan="${available.length+1}">No operation timings available.</td></tr>`;
}
function render(){document.getElementById('test-notice').hidden=splitSelect.value!=='test';document.getElementById('whole-notice').hidden=splitSelect.value!=='whole';renderIndividual();renderCompare()}
initSelectors();mode.addEventListener('change',render);splitSelect.addEventListener('change',render);versionSelect.addEventListener('change',render);document.getElementById('compare-select').addEventListener('change',render);render();
</script></body></html>'''
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(template.replace("__REPORT_DATA__", data), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
