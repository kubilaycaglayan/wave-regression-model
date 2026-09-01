#!/usr/bin/env python3
"""Small local web app for manually labeling final water images."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_DIR = PROJECT_DIR / "step-2-final-water-data"
LABELS_PATH = PROJECT_DIR / "labels.csv"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
LABELS_LOCK = threading.Lock()


PAGE = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Sea waviness labeling</title>
  <style>
    :root { color-scheme: light; --ink:#17212b; --muted:#66727d; --line:#d8e0e6; --accent:#087f8c; }
    * { box-sizing:border-box; }
    body { margin:0; background:#f3f6f8; color:var(--ink); font:15px system-ui,-apple-system,sans-serif; }
    header { padding:18px max(18px, calc((100vw - 1200px)/2)); background:#12343b; color:white; }
    header h1 { margin:0 0 12px; font-size:22px; }
    .stats { display:flex; flex-wrap:wrap; gap:20px; color:#d7edef; }
    .stats strong { color:white; }
    nav { display:flex; gap:8px; max-width:1200px; margin:18px auto 0; padding:0 18px; }
    button, select { border:1px solid var(--line); border-radius:7px; background:white; color:var(--ink); padding:9px 13px; font:inherit; cursor:pointer; }
    button:hover, select:hover { border-color:var(--accent); }
    button.primary, .tab.active { background:var(--accent); color:white; border-color:var(--accent); }
    main { max-width:1200px; margin:0 auto; padding:0 18px 36px; }
    .toolbar { display:flex; justify-content:space-between; align-items:center; gap:12px; margin:0 0 14px; }
    .toolbar label { color:var(--muted); }
    .view { display:none; } .view.active { display:block; }
    .label-card { background:white; border:1px solid var(--line); border-radius:12px; padding:18px; }
    .image-wrap { display:flex; justify-content:center; align-items:center; min-height:220px; max-height:52vh; background:#10191d; border-radius:8px; overflow:hidden; }
    .image-wrap img { display:block; width:auto; max-width:min(100%, 420px); max-height:52vh; object-fit:contain; }
    .image-meta { display:flex; justify-content:space-between; gap:12px; margin:15px 0 8px; }
    .filename { overflow-wrap:anywhere; font-weight:650; }
    .status { color:var(--muted); }
    .status.unlabeled { color:#a65300; font-weight:650; }
    .slider-row { display:grid; grid-template-columns:auto 1fr auto; align-items:center; gap:12px; }
    input[type=range] { width:100%; accent-color:var(--accent); }
    .value { min-width:52px; text-align:right; font:700 20px ui-monospace,monospace; }
    .range-label { color:var(--muted); font-size:13px; }
    .actions { display:flex; flex-wrap:wrap; justify-content:space-between; gap:8px; margin-top:20px; }
    .actions .right { display:flex; gap:8px; }
    .hint { color:var(--muted); font-size:13px; margin-top:14px; }
    .gallery { display:grid; grid-template-columns:repeat(auto-fill,minmax(190px,1fr)); gap:14px; }
    .tile { background:white; border:1px solid var(--line); border-radius:9px; padding:9px; text-align:left; cursor:pointer; }
    .tile:hover { border-color:var(--accent); }
    .tile img { display:block; width:100%; aspect-ratio:1; object-fit:contain; background:#10191d; border-radius:5px; }
    .tile .score { margin-top:8px; font-weight:750; font-size:18px; }
    .tile .tile-name { color:var(--muted); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .empty { padding:35px; color:var(--muted); text-align:center; background:white; border:1px solid var(--line); border-radius:10px; }
    @media (max-width:600px) { .toolbar, .image-meta { align-items:flex-start; flex-direction:column; } .image-wrap { min-height:250px; } }
  </style>
</head>
<body>
  <header><h1>Sea waviness labeling</h1><div class="stats"><span>Total: <strong id="total">0</strong></span><span>Labeled: <strong id="labeled">0</strong></span><span>Remaining: <strong id="remaining">0</strong></span></div></header>
  <nav><button class="tab active" data-view="labeling">Labeling</button><button class="tab" data-view="review">Review</button></nav>
  <main>
    <section id="labeling" class="view active">
      <div class="toolbar"><label for="sort">Order <select id="sort"><option value="unlabeled">Unlabeled first</option><option value="high">Waviness: highest to lowest</option><option value="low">Waviness: lowest to highest</option><option value="filename">Filename</option></select></label><span id="position" class="status"></span></div>
      <div class="label-card"><div class="image-wrap"><img id="main-image" alt="Processed sea image"></div><div class="image-meta"><span id="filename" class="filename"></span><span id="label-status" class="status"></span></div><div class="slider-row"><span class="range-label">0.00</span><input id="slider" type="range" min="0" max="1" step="0.05" value="0.50"><span id="value" class="value">0.50</span></div><div class="actions"><button id="previous">Previous</button><div class="right"><button id="skip">Skip</button><button id="save" class="primary">Save &amp; Next</button></div></div><div class="hint">Keyboard: 1–9 = 0.10–0.90 · 0 = 1.00 · ←/→ adjust by 0.05 · Enter save &amp; next · S skip · P previous</div></div>
    </section>
    <section id="review" class="view"><div class="toolbar"><h2>Labeled images</h2><label for="review-sort">Order <select id="review-sort"><option value="high">Highest to lowest</option><option value="low">Lowest to highest</option></select></label></div><div id="gallery" class="gallery"></div></section>
  </main>
  <script>
    const state = { images: [], labels: {}, ordered: [], index: 0, sort: 'unlabeled' };
    const $ = (id) => document.getElementById(id);
    const scoreText = (value) => Number(value).toFixed(2);
    function orderedImages() {
      const list = [...state.images];
      const score = (name) => state.labels[name];
      if (state.sort === 'unlabeled') return list.sort((a,b) => (score(a) === undefined) - (score(b) === undefined) || a.localeCompare(b));
      if (state.sort === 'high') return list.sort((a,b) => (score(b) ?? -1) - (score(a) ?? -1) || a.localeCompare(b));
      if (state.sort === 'low') return list.sort((a,b) => (score(a) ?? 2) - (score(b) ?? 2) || a.localeCompare(b));
      return list.sort((a,b) => a.localeCompare(b));
    }
    function updateStats() { $('total').textContent=state.images.length; $('labeled').textContent=Object.keys(state.labels).filter(n=>state.images.includes(n)).length; $('remaining').textContent=state.images.length-Number($('labeled').textContent); }
    function renderMain() {
      state.ordered = orderedImages();
      if (!state.ordered.length) { $('filename').textContent='No processed images found'; $('position').textContent=''; $('main-image').removeAttribute('src'); return; }
      state.index = Math.max(0, Math.min(state.index, state.ordered.length-1)); const name=state.ordered[state.index]; const labeled=state.labels[name] !== undefined;
      $('main-image').src='/images/'+encodeURIComponent(name); $('main-image').alt=name; $('filename').textContent=name; $('position').textContent=`${state.index+1} of ${state.ordered.length}`;
      $('label-status').textContent=labeled ? 'Labeled' : 'UNLABELED'; $('label-status').className='status'+(labeled?'':' unlabeled'); $('slider').value=labeled ? state.labels[name] : 0.50; $('value').textContent=scoreText($('slider').value);
    }
    function renderGallery() { const high=$('review-sort').value==='high'; const labeled=state.images.filter(n=>state.labels[n]!==undefined).sort((a,b)=>(high?state.labels[b]-state.labels[a]:state.labels[a]-state.labels[b])||a.localeCompare(b)); $('gallery').innerHTML=labeled.length ? labeled.map(n=>`<button class="tile" data-name="${encodeURIComponent(n)}"><img src="/images/${encodeURIComponent(n)}" alt=""><div class="score">${scoreText(state.labels[n])}</div><div class="tile-name" title="${n}">${n}</div></button>`).join('') : '<div class="empty">No labels saved yet.</div>'; document.querySelectorAll('.tile').forEach(el=>el.onclick=()=>selectForLabeling(decodeURIComponent(el.dataset.name))); }
    function selectForLabeling(name) { state.sort='filename'; $('sort').value='filename'; state.ordered=orderedImages(); state.index=state.ordered.indexOf(name); showView('labeling'); renderMain(); }
    function showView(view) { document.querySelectorAll('.view').forEach(el=>el.classList.toggle('active',el.id===view)); document.querySelectorAll('.tab').forEach(el=>el.classList.toggle('active',el.dataset.view===view)); if(view==='review') renderGallery(); }
    async function reload() { const data=await fetch('/api/data').then(r=>r.json()); state.images=data.images; state.labels=data.labels; updateStats(); renderMain(); renderGallery(); }
    async function save() { const name=state.ordered[state.index]; if(!name)return; const waviness=Number($('slider').value).toFixed(2); const response=await fetch('/api/labels',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({filename:name,waviness})}); if(!response.ok){let detail='Could not save label.'; try { const body=await response.json(); if(body.error) detail=body.error; } catch (_) {} alert(detail);return;} const nextUnlabeled=state.images.find(n=>state.labels[n]===undefined && n!==name); await reload(); if(nextUnlabeled && state.sort==='unlabeled') state.index=state.ordered.indexOf(nextUnlabeled); else state.index=Math.min(state.index+1,state.ordered.length-1); renderMain(); }
    $('slider').oninput=()=> $('value').textContent=scoreText($('slider').value); $('sort').onchange=()=>{state.sort=$('sort').value;state.index=0;renderMain();}; $('review-sort').onchange=renderGallery; $('save').onclick=save; $('skip').onclick=()=>{state.index=Math.min(state.index+1,state.ordered.length-1);renderMain();}; $('previous').onclick=()=>{state.index=Math.max(state.index-1,0);renderMain();}; document.querySelectorAll('.tab').forEach(el=>el.onclick=()=>showView(el.dataset.view));
    document.addEventListener('keydown', e=>{if(['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName))return; if(!document.getElementById('labeling').classList.contains('active'))return; if(/^[0-9]$/.test(e.key)){e.preventDefault();$('slider').value=e.key==='0'?1:Number(e.key)/10;$('value').textContent=scoreText($('slider').value);} else if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();$('slider').value=Math.max(0,Math.min(1,Number($('slider').value)+(e.key==='ArrowRight'?0.05:-0.05)));$('value').textContent=scoreText($('slider').value);} else if(e.key==='Enter')save(); else if(e.key.toLowerCase()==='s'){$('skip').click();} else if(e.key.toLowerCase()==='p'){$('previous').click();}});
    reload().catch(()=>alert('Could not load labeling data.'));
  </script>
</body>
</html>'''


def image_names() -> list[str]:
    """Return only actual final pipeline outputs, excluding comparison files."""
    return sorted(
        path.name for path in IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        and path.name.startswith("step-2_")
        and "original" not in path.stem.lower()
        and "overlay" not in path.stem.lower()
    ) if IMAGE_DIR.exists() else []


def read_labels() -> dict[str, float]:
    if not LABELS_PATH.exists():
        return {}
    result: dict[str, float] = {}
    with LABELS_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                name = row["filename"]
                value = float(row["waviness"])
                if name and 0 <= value <= 1:
                    result[name] = round(value / 0.05) * 0.05
            except (KeyError, TypeError, ValueError):
                continue
    return result


def write_label(filename: str, waviness: float) -> None:
    with LABELS_LOCK:
        labels = read_labels()
        labels[filename] = waviness
        LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix="labels.", suffix=".csv", dir=LABELS_PATH.parent)
        try:
            with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["filename", "waviness"])
                for name in sorted(labels):
                    writer.writerow([name, f"{labels[name]:.2f}"])
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, LABELS_PATH)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {self.address_string()} {format % args}")

    def send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = PAGE.encode("utf-8")
            self.send_response(HTTPStatus.OK); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
        if parsed.path == "/api/data":
            labels = read_labels(); self.send_json({"images": image_names(), "labels": labels}); return
        if parsed.path.startswith("/images/"):
            requested = Path(unquote(parsed.path.removeprefix("/images/"))).name
            if requested != unquote(parsed.path.removeprefix("/images/")) or requested not in image_names():
                self.send_error(HTTPStatus.NOT_FOUND); return
            path = IMAGE_DIR / requested
            body = path.read_bytes(); content_type = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
            self.send_response(HTTPStatus.OK); self.send_header("Content-Type", content_type); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/labels": self.send_error(HTTPStatus.NOT_FOUND); return
        try:
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            filename = str(payload["filename"]); value = float(payload["waviness"])
            # Validate by integer step count with a tolerance; binary floating-point
            # representation makes values such as 0.35 / 0.05 slightly imprecise.
            step_count = round(value * 20)
            if filename not in image_names() or not 0 <= value <= 1 or abs(value * 20 - step_count) > 1e-9:
                raise ValueError
            write_label(filename, step_count / 20); self.send_json({"ok": True})
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            self.send_json({"error": "filename must be a processed image and waviness must be 0.00–1.00 in 0.05 steps"}, HTTPStatus.BAD_REQUEST)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local sea-waviness labeling app")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Labeling app: http://{args.host}:{args.port}")
    print(f"Images: {IMAGE_DIR}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping labeling app.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
