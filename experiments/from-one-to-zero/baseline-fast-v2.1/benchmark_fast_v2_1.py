#!/usr/bin/env python3
"""Benchmark the current fast-v2.1 raw-photo-to-score prediction path."""
from __future__ import annotations
import hashlib, json, math, statistics, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PHOTOS = [ROOT / 'step-7-predict-captures-holder/IMG_7912.HEIC', ROOT / 'step-7-predict-captures-holder/IMG_7914.HEIC']
sys.path.insert(0, str(ROOT))
from step_7_j_fast_v2_1_client import FastV2_1Client, DEFAULT_REGRESSOR, SEGMENTATION_MODEL

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()
def summary(vals):
    vals=sorted(vals)
    return {'count':len(vals),'median_ms':statistics.median(vals),'p95_ms':vals[max(0,math.ceil(.95*len(vals))-1)],'min_ms':vals[0],'max_ms':vals[-1]}
def main():
    if any(not p.is_file() for p in PHOTOS): raise FileNotFoundError('Benchmark cohort photo missing')
    env=ROOT/'.env'
    configured=next((x.split('=',1)[1].strip().strip('\"\'') for x in env.read_text().splitlines() if x.strip().startswith('PREDICT_CHECKPOINT_PATH=')),None)
    checkpoint=(ROOT/configured).resolve() if configured else None
    if not checkpoint or not checkpoint.is_file(): raise FileNotFoundError(f'Configured checkpoint missing: {checkpoint}')
    t=time.perf_counter()
    records=[]
    with FastV2_1Client() as client:
        startup_ms=(time.perf_counter()-t)*1000
        health=client.health or {}
        # Warm each source once; exclude these from measured latency.
        for photo in PHOTOS: client.predict_path(photo)
        for photo in PHOTOS:
            for repeat in range(1,4):
                started=time.perf_counter(); payload=client.predict_path(photo); elapsed=(time.perf_counter()-started)*1000
                records.append({'photo':photo.name,'repeat':repeat,'status':'ok','prediction':payload['prediction'],'total_raw_to_score_ms':elapsed,'server_pipeline_ms':payload.get('total_server_ms'),'stages_ms':payload.get('stages_ms',{}),'image':payload.get('image'),'crop':payload.get('intermediate',{}).get('source_crop_xyxy')})
    report={'created_at_utc':datetime.now(timezone.utc).isoformat(),'purpose':'Baseline timing for current fast-v2.1 prediction path; no speed changes.','pipeline':'fast-v2.1-coordinate: raw HEIC decode/orientation -> 448px SegFormer-B0 ONNX segmentation (MIGraphX) -> lower-water rectangle selection and source-coordinate crop -> 224px masked JPEG -> EfficientNet-B0 ONNX waviness inference (CPU).','checkpoint_setting':{'path':str(checkpoint.relative_to(ROOT)),'sha256':sha(checkpoint),'note':'PREDICT_CHECKPOINT_PATH is configured to latest v20 checkpoint. The v2.1 ONNX service does not load this PyTorch checkpoint; its regressor is the separately identified ONNX model below.'},'models':{'segmentation':{'path':str(SEGMENTATION_MODEL.relative_to(ROOT)),'sha256':sha(SEGMENTATION_MODEL)},'regressor':{'path':str(DEFAULT_REGRESSOR.relative_to(ROOT)),'sha256':sha(DEFAULT_REGRESSOR),'source':'models/fast-v2.1/model-manifest.json'}},'runtime_health':health,'cohort':[{'filename':p.name,'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in PHOTOS],'protocol':{'warmup_runs_per_photo':1,'measured_runs_per_photo':3,'photos':len(PHOTOS),'measurement':'client request wall time, raw file already on disk through returned waviness score; includes local HTTP request overhead; excludes service/model initialization'},'startup_to_ready_ms':startup_ms,'latency_summary':{'client_raw_to_score_ms':summary([r['total_raw_to_score_ms'] for r in records]),'server_pipeline_ms':summary([r['server_pipeline_ms'] for r in records])},'runs':records}
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    (OUT/f'baseline-{stamp}.json').write_text(json.dumps(report,indent=2)+'\n')
    subprocess.run([sys.executable, str(ROOT/'experiments/from-one-to-zero/generate_index.py')], check=True)
    print(json.dumps({'json':str(OUT/f'baseline-{stamp}.json'),'html':'experiments/from-one-to-zero/index.html','client_median_ms':report['latency_summary']['client_raw_to_score_ms']['median_ms'],'server_median_ms':report['latency_summary']['server_pipeline_ms']['median_ms'],'startup_ms':startup_ms},indent=2))
def render(d):
    client=d['latency_summary']['client_raw_to_score_ms']; server=d['latency_summary']['server_pipeline_ms']
    rows=''.join(f"<tr><td>{r['photo']}</td><td>{r['repeat']}</td><td>{r['prediction']:.5f}</td><td>{r['total_raw_to_score_ms']:.2f}</td><td>{r['server_pipeline_ms']:.2f}</td></tr>" for r in d['runs'])
    stages=sorted({k for r in d['runs'] for k in r['stages_ms']})
    stage_rows=''
    for key in stages:
        values=sorted(r['stages_ms'][key] for r in d['runs'] if key in r['stages_ms'])
        med=statistics.median(values); p95=values[max(0,math.ceil(.95*len(values))-1)]
        stage_rows+=f'<tr><td>{key}</td><td>{med:.2f}</td><td>{p95:.2f}</td></tr>'
    detail=json.dumps({k:d[k] for k in ['pipeline','checkpoint_setting','models','cohort','protocol','runtime_health']},indent=2)
    return f"""<!doctype html><html lang=\"en\"><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>Fast-v2.1 baseline</title><style>body{{font:15px/1.55 system-ui;margin:0;background:#f2f6f4;color:#18343a}}main{{max-width:1050px;margin:40px auto;padding:0 20px}}header{{background:#16464d;color:white;padding:28px;border-radius:16px}}h1{{margin:0}}.cards{{display:flex;gap:14px;margin:18px 0}}.card,section{{background:white;border:1px solid #d8e3df;border-radius:12px;padding:16px;margin:14px 0}}.card{{flex:1}}.big{{font-size:26px;font-weight:700}}table{{width:100%;border-collapse:collapse}}td,th{{text-align:left;padding:8px;border-bottom:1px solid #e5ece9}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}}</style><main><header><small>From One to Zero · Baseline</small><h1>Current fast-v2.1 prediction benchmark</h1><p>Raw HEIC photo to waviness score. Startup is separated from per-photo latency.</p></header><div class=\"cards\"><div class=\"card\">Client median<div class=\"big\">{fmt(client['median_ms'])}</div><small>includes local request overhead · p95 {fmt(client['p95_ms'])}</small></div><div class=\"card\">Server median<div class=\"big\">{fmt(server['median_ms'])}</div><small>pipeline execution · p95 {fmt(server['p95_ms'])}</small></div><div class=\"card\">Startup<div class=\"big\">{fmt(d['startup_to_ready_ms'])}</div><small>server/model initialization</small></div></div><section><h2>Per-photo runs</h2><table><tr><th>Photo</th><th>Run</th><th>Score</th><th>Client ms</th><th>Server ms</th></tr>{rows}</table></section><section><h2>Pipeline stages</h2><p>Stage medians and p95; timings are also included in the total server pipeline.</p><table><tr><th>Stage</th><th>Median ms</th><th>p95 ms</th></tr>{stage_rows}</table></section><section><h2>Models, checkpoint setting, cohort and protocol</h2><pre>{detail}</pre></section><p>Generated {d['created_at_utc']}. Full measurements are in the timestamped JSON under baseline-fast-v2.1/.</p></main></html>"""
def fmt(x): return f'{x:.2f} ms'
if __name__=='__main__': main()
