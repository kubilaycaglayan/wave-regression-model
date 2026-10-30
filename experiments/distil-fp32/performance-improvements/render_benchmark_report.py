#!/usr/bin/env python3
"""Render the latest fine-grained Distil-FP32 JSON report as standalone HTML."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="benchmark JSON; defaults to newest report in runs/")
    parser.add_argument("--output", type=Path, default=HERE / "distil-fp32-benchmark-report.html")
    args = parser.parse_args()
    reports = sorted((HERE / "runs").glob("distil-fp32-fine-grained-*.json"))
    source = args.input.resolve() if args.input else (reports[-1] if reports else None)
    if source is None or not source.is_file():
        raise FileNotFoundError("No benchmark report found. Run benchmark_distil_fp32.py first.")
    report = json.loads(source.read_text(encoding="utf-8"))
    payload = {
        "created_at_utc": report.get("created_at_utc"),
        "candidate": report.get("candidate"),
        "warmup": report.get("warmup", {}),
        "quality_by_split": report.get("quality_by_split", {}),
        "latency_by_split": report.get("latency_by_split", {}),
        "photos": [
            {
                "split": row.get("split"),
                "filename": row.get("label_filename"),
                "source_filename": row.get("source_filename"),
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
        "report_name": source.name,
    }
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    template = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light">
<title>Distil-FP32 · Performance report</title>
<style>
:root{--ink:#142b38;--muted:#5b7078;--paper:#f4f7f4;--card:#fff;--line:#dce5e1;--navy:#153c4a;--teal:#237b78;--green:#208451;--amber:#b77914;--orange:#c25b19;--red:#bd3f38;--shadow:0 10px 30px rgba(20,43,56,.07)}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}header{background:linear-gradient(120deg,#123642,#1c5960);color:#f8fbf8;padding:34px max(24px,calc((100vw - 1240px)/2)) 30px}.eyebrow{text-transform:uppercase;letter-spacing:.14em;font-size:11px;font-weight:800;color:#9fd8c7}h1{font-size:clamp(28px,4vw,42px);line-height:1.1;margin:8px 0 10px;letter-spacing:-.035em}.subtitle{max-width:800px;color:#d4e5e0;margin:0}.meta{font-size:12px;color:#b6d2ce;margin-top:18px}.wrap{max-width:1240px;margin:auto;padding:26px 24px 60px}.toolbar{display:flex;align-items:center;justify-content:space-between;gap:16px;margin:0 0 18px;flex-wrap:wrap}.selectbox{display:flex;align-items:center;gap:10px;color:var(--muted);font-size:13px;font-weight:700}select{font:inherit;color:var(--ink);background:#fff;border:1px solid var(--line);border-radius:10px;padding:9px 34px 9px 12px}.micro{font-size:12px;color:var(--muted)}.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:13px;margin-bottom:20px}.card,.panel{background:var(--card);border:1px solid var(--line);border-radius:16px;box-shadow:var(--shadow)}.card{padding:17px 18px}.card-label{font-size:11px;font-weight:800;letter-spacing:.09em;text-transform:uppercase;color:var(--muted)}.card-value{font-size:28px;line-height:1.2;font-weight:780;letter-spacing:-.03em;margin:8px 0 4px;font-variant-numeric:tabular-nums}.card-note{font-size:12px;color:var(--muted)}.quality{color:var(--teal)}.sections{display:grid;gap:18px}.panel{overflow:hidden}.panel-head{padding:18px 20px;border-bottom:1px solid var(--line);display:flex;align-items:flex-start;justify-content:space-between;gap:16px}.panel-head h2{margin:0;font-size:19px;letter-spacing:-.02em}.panel-head p{margin:4px 0 0;color:var(--muted);font-size:13px}.total-pill{background:#edf5f2;border:1px solid #d4e6de;color:var(--navy);font-weight:800;border-radius:999px;padding:7px 11px;white-space:nowrap;font-size:13px}.panel-body{padding:17px 20px 20px}.legend{display:flex;gap:12px;align-items:center;flex-wrap:wrap;color:var(--muted);font-size:11px}.key{display:inline-flex;align-items:center;gap:5px}.dot{width:9px;height:9px;border-radius:50%;display:inline-block}.dot.low{background:var(--green)}.dot.medium{background:var(--amber)}.dot.high{background:var(--orange)}.dot.hot{background:var(--red)}.group{margin:0 0 20px}.group:last-child{margin-bottom:0}.group-title{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:#42616a;font-weight:800;margin:0 0 8px}.row{display:grid;grid-template-columns:minmax(230px,310px) minmax(100px,1fr) 78px 78px 78px;gap:13px;align-items:center;padding:9px 0;border-bottom:1px solid #eef2ef}.row:last-child{border-bottom:0}.name{font-size:13px;font-weight:650}.bar-rail{height:11px;border-radius:99px;background:#eef2ee;overflow:hidden}.bar{height:100%;min-width:2px;border-radius:inherit}.bar.low{background:linear-gradient(90deg,#4cac74,#208451)}.bar.medium{background:linear-gradient(90deg,#edc04e,#b77914)}.bar.high{background:linear-gradient(90deg,#f3a145,#c25b19)}.bar.hot{background:linear-gradient(90deg,#ec7660,#bd3f38)}.num{text-align:right;font-size:12px;font-variant-numeric:tabular-nums;color:#38525c}.num strong{font-size:13px;color:var(--ink)}.table-head{display:grid;grid-template-columns:minmax(230px,310px) minmax(100px,1fr) 78px 78px 78px;gap:13px;padding:0 0 7px;color:#73858a;font-size:10px;text-transform:uppercase;letter-spacing:.08em;font-weight:800}.table-head span:nth-child(n+3){text-align:right}.callout{border-left:4px solid var(--teal);padding:11px 14px;background:#eff7f4;border-radius:0 10px 10px 0;color:#36545b;font-size:13px;margin-top:16px}.warm-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px 24px}.warm-row{display:grid;grid-template-columns:1fr auto;align-items:center;gap:16px;padding:9px 0;border-bottom:1px solid #eef2ef}.warm-row span:first-child{font-size:13px}.warm-row strong{font-size:13px;font-variant-numeric:tabular-nums}.warm-row.total{border-top:2px solid var(--line);font-weight:800}.quality-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.quality-box{border:1px solid var(--line);border-radius:12px;padding:14px}.quality-box h3{margin:0 0 8px;font-size:14px}.quality-metrics{display:flex;gap:20px;flex-wrap:wrap}.quality-metrics strong{font-size:19px;display:block;font-variant-numeric:tabular-nums}.quality-metrics span{font-size:11px;color:var(--muted)}.status{margin-left:auto;color:var(--green);font-weight:800}.fail{color:var(--red)}details summary{cursor:pointer;font-weight:750;color:var(--navy)}.photo-table{width:100%;border-collapse:collapse;margin-top:12px;font-size:12px}.photo-table th,.photo-table td{text-align:left;padding:8px;border-bottom:1px solid #e9efec}.photo-table th{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.06em}.photo-table td:nth-child(n+3){text-align:right;font-variant-numeric:tabular-nums}.footer{margin-top:18px;color:var(--muted);font-size:11px}.tag{font-size:10px;text-transform:uppercase;letter-spacing:.07em;color:#537078}.empty{color:var(--muted);padding:12px 0}
@media(max-width:900px){.cards{grid-template-columns:repeat(2,minmax(0,1fr))}.row,.table-head{grid-template-columns:minmax(170px,240px) minmax(70px,1fr) 65px 65px 65px;gap:8px}.name{font-size:12px}}
@media(max-width:620px){.wrap{padding:18px 12px 42px}header{padding:26px 18px}.cards{grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.card{padding:13px}.card-value{font-size:22px}.panel-head,.panel-body{padding:15px}.row,.table-head{grid-template-columns:minmax(125px,1.3fr) minmax(55px,1fr) 53px 53px;gap:6px}.row .max,.table-head .max{display:none}.warm-grid,.quality-grid{grid-template-columns:1fr}.name{font-size:11px}.num{font-size:10px}.num strong{font-size:11px}}
</style>
</head>
<body>
<header><div class="eyebrow">Performance investigation · Distil-FP32</div><h1>Where the time goes</h1><p class="subtitle">A readable view of model warm-up, water-finding preprocessing, and final waviness inference. Select a labeled split to inspect its measured latency and quality.</p><div class="meta" id="meta"></div></header>
<main class="wrap">
  <div class="toolbar"><label class="selectbox">Evaluation split <select id="split"><option value="validation">Validation · 47 photos</option><option value="test">Test · 46 photos</option></select></label><div class="legend"><span class="key"><i class="dot low"></i>under 1 ms</span><span class="key"><i class="dot medium"></i>1–5 ms</span><span class="key"><i class="dot high"></i>5–15 ms</span><span class="key"><i class="dot hot"></i>over 15 ms</span><span class="micro">Color grades each operation by its median time.</span></div></div>
  <section class="cards" id="cards"></section>
  <div class="sections">
    <section class="panel"><div class="panel-head"><div><h2>1 · Warm-up</h2><p>One-time startup cost before the server can process photos.</p></div><div class="total-pill" id="warm-total"></div></div><div class="panel-body"><div class="warm-grid" id="warm-rows"></div><div class="callout">The long first runs include MIGraphX graph setup/compilation. They happen at startup; a warm request does not pay them again.</div></div></section>
    <section class="panel"><div class="panel-head"><div><h2>2 · Pre-processing</h2><p>Raw photo decode through construction of the final normalized input tensor for the regressor.</p></div><div class="total-pill" id="pre-total"></div></div><div class="panel-body" id="pre-rows"></div></section>
    <section class="panel"><div class="panel-head"><div><h2>3 · Waviness inference</h2><p>The v18 EfficientNet-B0 ONNX model converts the prepared tensor into one waviness score.</p></div><div class="total-pill" id="infer-total"></div></div><div class="panel-body" id="infer-rows"></div></section>
    <section class="panel"><div class="panel-head"><div><h2>Quality on labeled photos</h2><p>Prediction quality against the existing waviness labels. These are the same 93 photos used for the baseline run.</p></div><div class="status" id="failure-status"></div></div><div class="panel-body"><div class="quality-grid" id="quality"></div></div></section>
    <section class="panel"><div class="panel-head"><div><h2>Slowest photos in this split</h2><p>Useful for seeing whether long-tail latency comes from decode or a later stage.</p></div></div><div class="panel-body"><details><summary>Show the ten slowest requests</summary><table class="photo-table"><thead><tr><th>Photo</th><th>Label</th><th>Total</th><th>Decode</th><th>Step 1 model</th><th>Step 2 model</th><th>Prediction</th></tr></thead><tbody id="slow-photos"></tbody></table></details></div></section>
  </div>
  <p class="footer">Report: <span id="report-name"></span>. Latencies are local raw-file-to-score timings; HTTP upload and temporary request-file staging are excluded. Parent totals overlap their child operations and should not be added to them. The per-operation medians are independently summarized and therefore do not sum exactly to the median total.</p>
</main>
<script id="report-data" type="application/json">__REPORT_DATA__</script>
<script>
const D=JSON.parse(document.getElementById('report-data').textContent);
const fmt=(x,d=2)=>x==null?'—':`${Number(x).toFixed(d)} ms`;
const sec=x=>x==null?'—':x>=1000?`${(x/1000).toFixed(2)} s`:`${x.toFixed(1)} ms`;
const qfmt=x=>x==null?'—':Number(x).toFixed(5);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const grade=x=>x>15?'hot':x>5?'high':x>=1?'medium':'low';
const pretty={
 'preprocessing.decode_oriented_rgb':'Decode and orient photo',
 'preprocessing.step1.tensor.resize_rgb':'Resize raw RGB photo to 448 × 448',
 'preprocessing.step1.tensor.uint8_to_float32_and_scale':'Convert pixels to FP32 and scale',
 'preprocessing.step1.tensor.normalize_mean_std':'Normalize Step 1 input',
 'preprocessing.step1.tensor.transpose_and_contiguous_tensor':'Arrange contiguous Step 1 tensor',
 'preprocessing.step1.segmentation_inference':'Step 1 water segmentation inference',
 'preprocessing.step1.expand_mask_to_source':'Expand water mask to source dimensions',
 'preprocessing.step1.erode_mask':'Erode full-size water mask',
 'preprocessing.step1.bounding_box':'Find water bounding box',
 'preprocessing.step1.water_pixel_count_and_validation':'Count water pixels and validate mask',
 'preprocessing.step1.roi_views':'Create ROI views',
 'preprocessing.step1.masked_roi_copy':'Apply mask and copy water ROI',
 'preprocessing.step1.pil_image_from_masked_roi':'Create PIL image from masked ROI',
 'preprocessing.step1.canvas.lanczos_resize':'Lanczos resize water ROI',
 'preprocessing.step1.canvas.black_canvas_allocate_and_paste':'Allocate black canvas and paste ROI',
 'preprocessing.step1.jpeg_encode':'Encode Step 1 JPEG',
 'preprocessing.step1.jpeg_decode_and_rgb_convert':'Decode Step 1 JPEG and convert to RGB',
 'preprocessing.step2.black_area_mask':'Identify black canvas area',
 'preprocessing.step2.tensor.resize_rgb':'Resize Step 1 image to 384 × 384',
 'preprocessing.step2.tensor.uint8_to_float32_and_scale':'Convert Step 2 pixels to FP32 and scale',
 'preprocessing.step2.tensor.normalize_mean_std':'Normalize Step 2 input',
 'preprocessing.step2.tensor.transpose_and_contiguous_tensor':'Arrange contiguous Step 2 tensor',
 'preprocessing.step2.segmentation_inference':'Step 2 water segmentation inference',
 'preprocessing.step2.expand_mask_to_canvas':'Expand Step 2 mask to canvas',
 'preprocessing.step2.mask_convert_and_exclude_black':'Convert mask and exclude black pixels',
 'preprocessing.step2.crop_selection':'Select lower-water crop',
 'preprocessing.step2.crop_extract':'Extract selected crop',
 'preprocessing.step2.canvas.lanczos_resize':'Resize selected crop',
 'preprocessing.step2.canvas.black_canvas_allocate_and_paste':'Allocate final canvas and paste crop',
 'preprocessing.final.jpeg_encode':'Encode final model-input JPEG',
 'preprocessing.final.jpeg_decode_and_rgb_convert':'Decode final JPEG and convert to RGB',
 'preprocessing.final.rgb_to_float32_scale':'Convert final RGB to FP32 and scale',
 'preprocessing.final.normalize_mean_std':'Normalize regressor input',
 'preprocessing.final.transpose_and_contiguous_tensor':'Arrange final regressor tensor',
 'waviness_inference.v18_regressor_onnxruntime':'v18 regressor ONNX Runtime call',
 'waviness_inference.score_extraction':'Extract scalar waviness score'
};
const groups=[
 {title:'Input decode',match:k=>k==='preprocessing.decode_oriented_rgb'},
 {title:'Step 1 · Prepare and segment the raw photo',match:k=>k.includes('preprocessing.step1.tensor')||k==='preprocessing.step1.segmentation_inference'},
 {title:'Step 1 · Expand mask, isolate sea, and make canvas',match:k=>k.startsWith('preprocessing.step1.')&&!k.includes('.tensor.')&&k!=='preprocessing.step1.segmentation_inference'},
 {title:'Step 2 · Prepare and segment the Step 1 image',match:k=>k.includes('preprocessing.step2.tensor')||k==='preprocessing.step2.segmentation_inference'},
 {title:'Step 2 · Find lower water and make model canvas',match:k=>k.startsWith('preprocessing.step2.')&&!k.includes('.tensor.')&&k!=='preprocessing.step2.segmentation_inference'},
 {title:'Final regressor input · JPEG and tensor preparation',match:k=>k.startsWith('preprocessing.final.')}
];
const opStats=(split,key)=>D.latency_by_split[split]?.operations?.[key];
function operationRows(split,keys){
 const stats=keys.map(k=>[k,opStats(split,k)]).filter(x=>x[1]);
 if(!stats.length)return '<div class="empty">No operation timings recorded.</div>';
 const max=Math.max(...stats.map(x=>x[1].median_ms||0),1);
 return `<div class="table-head"><span>Operation</span><span>Relative time</span><span>Median</span><span>p95</span><span class="max">Max</span></div>`+stats.map(([key,s])=>{
  const m=s.median_ms||0,cls=grade(m),width=Math.max(1,m/max*100);
  return `<div class="row"><div class="name">${esc(pretty[key]||key)}</div><div class="bar-rail" title="${m.toFixed(2)} ms median"><div class="bar ${cls}" style="width:${width}%"></div></div><div class="num"><strong>${fmt(m)}</strong></div><div class="num">${fmt(s.p95_ms)}</div><div class="num max">${fmt(s.max_ms)}</div></div>`;
 }).join('');
}
function render(){
 const split=document.getElementById('split').value,lat=D.latency_by_split[split],quality=D.quality_by_split[split];
 const total=lat.total_raw_to_score_ms,pre=lat.preprocessing_total_ms,inf=lat.waviness_inference_total_ms;
 const photos=D.photos.filter(p=>p.split===split),fail=quality.failures||0;
 document.getElementById('meta').textContent=`${D.candidate} · ${D.created_at_utc||''} · ${D.report_name}`;
 document.getElementById('cards').innerHTML=[
  ['Raw → score',fmt(total.median_ms),`p95 ${fmt(total.p95_ms)} · max ${fmt(total.max_ms)}`],
  ['Pre-processing',fmt(pre.median_ms),`p95 ${fmt(pre.p95_ms)} · ${((pre.median_ms/total.median_ms)*100).toFixed(1)}% of median total`],
  ['Waviness inference',fmt(inf.median_ms),`p95 ${fmt(inf.p95_ms)} · ${((inf.median_ms/total.median_ms)*100).toFixed(1)}% of median total`],
  ['Quality · MAE / RMSE',`${qfmt(quality.mae)} / ${qfmt(quality.rmse)}`,`${quality.n} successful photos · ${fail} failures`]
 ].map((c,i)=>`<div class="card"><div class="card-label">${c[0]}</div><div class="card-value ${i===3?'quality':''}">${c[1]}</div><div class="card-note">${c[2]}</div></div>`).join('');
 const w=D.warmup;
 document.getElementById('warm-total').textContent=`Ready in ${sec(w.process_start_to_models_ready_ms)}`;
 const warmItems=[
  ['MIGraphX provider registration',w.runtime_registration_ms],
  ['Load Step 1 segmentation session',w.step1_session_load_ms],
  ['Load Step 2 segmentation session',w.step2_session_load_ms],
  ['Load v18 regressor session',w.regressor_session_load_ms],
  ['First Step 1 graph run · 448 × 448',w.warm_448_first_run_ms],
  ['First Step 2 graph run · 384 × 384',w.warm_384_first_run_ms],
  ['First v18 regressor run',w.warm_regressor_ms],
  ['Compile crop selector',w.warm_crop_selector_ms],
  ['Python and runtime imports',w.python_and_runtime_imports_ms]
 ];
 document.getElementById('warm-rows').innerHTML=warmItems.map(([n,v])=>`<div class="warm-row"><span>${n}</span><strong>${sec(v)}</strong></div>`).join('')+`<div class="warm-row total"><span>All model initialization and warm-up</span><strong>${sec(w.initialize_models_total_ms)}</strong></div>`;
 document.getElementById('pre-total').textContent=`Median ${fmt(pre.median_ms)} · p95 ${fmt(pre.p95_ms)}`;
 const operationKeys=Object.keys(lat.operations);
 document.getElementById('pre-rows').innerHTML=groups.map(g=>{
  const keys=operationKeys.filter(g.match).sort((a,b)=>lat.operations[b].median_ms-lat.operations[a].median_ms);
  return keys.length?`<div class="group"><h3 class="group-title">${g.title}</h3>${operationRows(split,keys)}</div>`:'';
 }).join('');
 document.getElementById('infer-total').textContent=`Median ${fmt(inf.median_ms)} · p95 ${fmt(inf.p95_ms)}`;
 const inferKeys=operationKeys.filter(k=>k.startsWith('waviness_inference.')).sort((a,b)=>lat.operations[b].median_ms-lat.operations[a].median_ms);
 document.getElementById('infer-rows').innerHTML=operationRows(split,inferKeys);
 document.getElementById('failure-status').innerHTML=fail?`<span class="fail">${fail} failures</span>`:`<span>${quality.n}/${photos.length} succeeded</span>`;
 document.getElementById('quality').innerHTML=['validation','test'].map(s=>{const q=D.quality_by_split[s];return `<div class="quality-box"><h3>${s==='validation'?'Validation':'Test'} · ${q.n} photos</h3><div class="quality-metrics"><div><strong>${qfmt(q.mae)}</strong><span>MAE</span></div><div><strong>${qfmt(q.rmse)}</strong><span>RMSE</span></div><div><strong>${q.failures}</strong><span>failures</span></div></div></div>`}).join('');
 const slow=[...photos].sort((a,b)=>(b.total_ms||0)-(a.total_ms||0)).slice(0,10);
 document.getElementById('slow-photos').innerHTML=slow.map(p=>`<tr><td>${esc(p.source_filename||p.filename)}</td><td>${qfmt(p.label)}</td><td>${fmt(p.total_ms)}</td><td>${fmt(p.decode_ms)}</td><td>${fmt(p.step1_ms)}</td><td>${fmt(p.step2_ms)}</td><td>${p.prediction==null?'—':Number(p.prediction).toFixed(5)}</td></tr>`).join('');
 document.getElementById('report-name').textContent=D.report_name;
}
document.getElementById('split').addEventListener('change',render);render();
</script>
</body>
</html>'''
    html = template.replace("__REPORT_DATA__", data)
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html, encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
