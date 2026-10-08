#!/usr/bin/env python3
"""Small local web app for manually labeling final water images."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import tempfile
import threading
import time
from datetime import datetime, timezone
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from PIL import Image

from image_loading import load_rgb_image


PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_DIR = PROJECT_DIR / "step-2-final-water-data"
LABELS_PATH = PROJECT_DIR / "labels.csv"
DISCARDED_PATH = PROJECT_DIR / "discarded_images.csv"
SPLIT_OUTPUT_DIR = PROJECT_DIR / "step-3-dataset-splits"
SPLIT_SNAPSHOT_DIR = SPLIT_OUTPUT_DIR / "snapshots"
# Step 1 records which source file produced each stem; step 2 keeps that stem.
SOURCE_MANIFEST_PATH = PROJECT_DIR / "step-1-processed-data" / ".source-manifest.json"
PROCESSED_PREFIX = "step-2_"
ORIGINAL_PREVIEW_MAX_SIDE = 800
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
    .stats { display:flex; flex-wrap:wrap; align-items:center; gap:12px 20px; color:#d7edef; }
    .stats strong { color:white; }
    .stats-separator { height:24px; border-left:1px solid #719095; }
    .label-groups { display:flex; flex-wrap:wrap; gap:12px 20px; }
    nav { display:flex; gap:8px; max-width:1200px; margin:18px auto 0; padding:0 18px; }
    button, select { border:1px solid var(--line); border-radius:7px; background:white; color:var(--ink); padding:9px 13px; font:inherit; cursor:pointer; }
    button:hover, select:hover { border-color:var(--accent); }
    button.primary, .tab.active { background:var(--accent); color:white; border-color:var(--accent); }
    main { max-width:1200px; margin:0 auto; padding:0 18px 36px; }
    .toolbar { display:flex; justify-content:space-between; align-items:center; gap:12px; margin:0 0 14px; }
    .toolbar label { color:var(--muted); }
    .view { display:none; } .view.active { display:block; }
    .label-card { background:white; border:1px solid var(--line); border-radius:12px; padding:18px; }
    .image-row { display:flex; align-items:stretch; gap:18px; height:clamp(320px, calc(100vh - 390px), 760px); }
    /* Fixed ~1/3 : 2/3 split so neither panel shifts while images load or change shape. */
    .original-wrap { flex:0 0 calc(33% - 12px); position:relative; margin:0 6px; display:flex; justify-content:center; align-items:center; background:#10191d; border-radius:8px; overflow:hidden; }
    .original-wrap img { display:block; height:100%; width:auto; max-width:100%; object-fit:contain; }
    .original-wrap .caption { position:absolute; left:6px; bottom:6px; padding:2px 6px; border-radius:4px; background:#10191dcc; color:white; font-size:12px; }
    .original-missing[hidden] { display:none; }
    .original-missing { display:flex; align-items:center; justify-content:center; height:100%; width:100%; padding:10px; text-align:center; color:var(--muted); font-size:13px; border:1px dashed var(--line); border-radius:6px; }
    .image-row .image-wrap { flex:1 1 0; min-width:0; }
    .image-wrap { display:flex; justify-content:center; align-items:center; min-height:0; background:#10191d; border-radius:8px; overflow:hidden; }
    .image-wrap img { display:block; width:auto; max-width:100%; max-height:100%; object-fit:contain; }
    .image-meta { display:flex; justify-content:space-between; gap:12px; margin:15px 0 8px; }
    .image-identification { display:flex; flex-wrap:wrap; align-items:baseline; gap:6px 14px; min-width:0; }
    .filename { overflow-wrap:anywhere; font-weight:650; }
    .capture-date { color:var(--muted); white-space:nowrap; }
    .status { color:var(--muted); }
    .status.unlabeled { color:#a65300; font-weight:650; }
    .slider-row { display:grid; grid-template-columns:auto 1fr auto; align-items:center; gap:12px; }
    input[type=range] { width:100%; accent-color:var(--accent); }
    .value { min-width:52px; text-align:right; font:700 20px ui-monospace,monospace; }
    .range-label { color:var(--muted); font-size:13px; }
    .actions { display:flex; flex-wrap:wrap; justify-content:space-between; gap:8px; margin-top:20px; }
    .actions .right { display:flex; gap:8px; }
    button.danger { color:#9b2c2c; border-color:#e5a7a7; }
    .hint { color:var(--muted); font-size:13px; margin-top:14px; }
    .gallery { display:grid; grid-template-columns:repeat(auto-fill,minmax(190px,1fr)); gap:14px; }
    .split-controls { display:flex; align-items:center; gap:10px; }
    .split-controls label { color:var(--muted); }
    .split-summary { display:flex; flex-wrap:wrap; gap:10px; margin:0 0 16px; }
    .split-stat { min-width:130px; padding:10px 13px; border:1px solid var(--line); border-radius:8px; background:white; }
    .split-stat span { display:block; color:var(--muted); font-size:12px; }
    .split-stat strong { display:block; margin-top:2px; }
    .split-bins { flex-basis:100%; padding:10px 13px; color:var(--muted); background:white; border:1px solid var(--line); border-radius:8px; }
    .split-gallery { display:grid; grid-template-columns:repeat(auto-fill,minmax(190px,1fr)); gap:14px; }
    .split-tile { min-width:0; position:relative; padding:9px; border:1px solid var(--line); border-radius:9px; background:white; }
    .split-tile img { display:block; width:100%; aspect-ratio:1; object-fit:contain; background:#10191d; border-radius:5px; }
    .split-tile .score { margin-top:8px; font-weight:750; font-size:18px; }
    .split-tile .tile-name { color:var(--muted); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .discarded-gallery { display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr)); gap:14px; }
    .discarded-tile { min-width:0; padding:10px; border:1px solid var(--line); border-radius:9px; background:white; }
    .discarded-tile img { display:block; width:100%; aspect-ratio:1; object-fit:contain; background:#10191d; border-radius:5px; }
    .discarded-tile .tile-name { margin-top:8px; font-weight:650; overflow-wrap:anywhere; }
    .discarded-reason,.discarded-time,.restore-label-note { margin-top:5px; color:var(--muted); font-size:13px; overflow-wrap:anywhere; }
    .discarded-tile button { margin-top:10px; }
    .tile { background:white; border:1px solid var(--line); border-radius:9px; padding:9px; text-align:left; cursor:pointer; }
    .tile:hover { border-color:var(--accent); }
    .tile img { display:block; width:100%; aspect-ratio:1; object-fit:contain; background:#10191d; border-radius:5px; }
    .tile .score { margin-top:8px; font-weight:750; font-size:18px; }
    .tile .tile-name { color:var(--muted); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .tile { position:relative; width:100%; }
    .tile-select { display:block; width:100%; padding:0; border:0; text-align:left; background:transparent; }
    .tile-select:hover { border-color:transparent; }
    .tile-discard { position:absolute; right:12px; bottom:12px; display:grid; place-items:center; width:36px; height:36px; padding:0; border-radius:50%; background:#fff; color:#a32323; border-color:#e5a7a7; box-shadow:0 1px 5px #0005; font-size:19px; }
    .dialog-backdrop { position:fixed; inset:0; z-index:10; display:grid; place-items:center; padding:18px; background:#10191dcc; }
    .dialog-backdrop[hidden] { display:none; }
    .dialog { width:min(100%, 440px); padding:22px; background:white; border-radius:12px; box-shadow:0 16px 50px #0005; }
    .dialog h2 { margin:0 0 10px; font-size:20px; }
    .dialog p { color:var(--muted); overflow-wrap:anywhere; }
    .dialog label { display:block; margin:16px 0 6px; }
    .dialog textarea { width:100%; min-height:80px; padding:9px; border:1px solid var(--line); border-radius:7px; font:inherit; }
    .dialog-actions { display:flex; justify-content:flex-end; gap:8px; margin-top:18px; }
    .message { position:fixed; right:18px; bottom:18px; z-index:11; max-width:min(440px, calc(100vw - 36px)); padding:12px 16px; border-radius:8px; color:white; background:#8b2525; box-shadow:0 3px 14px #0004; }
    .message[hidden] { display:none; }
    .empty { padding:35px; color:var(--muted); text-align:center; background:white; border:1px solid var(--line); border-radius:10px; }
    @media (max-width:600px) { .toolbar, .image-meta { align-items:flex-start; flex-direction:column; } .image-row { flex-direction:column; } .image-row > * { flex:1 1 0 !important; min-height:0; } .original-wrap { margin:0; } .stats-separator { display:none; } }
  </style>
</head>
<body>
  <header><h1>Sea waviness labeling</h1><div class="stats"><span>Total: <strong id="total">0</strong></span><span>Labeled: <strong id="labeled">0</strong></span><span>Remaining: <strong id="remaining">0</strong></span><span class="stats-separator" role="separator" aria-orientation="vertical"></span><span>Discarded: <strong id="discarded">0</strong></span><span class="stats-separator" role="separator" aria-orientation="vertical"></span><div class="label-groups" aria-label="Labeled images by waviness range"><span>0.00–0.33: <strong id="group-low">0</strong></span><span>0.34–0.66: <strong id="group-mid">0</strong></span><span>0.67–1.00: <strong id="group-high">0</strong></span></div></div></header>
  <nav><button class="tab active" data-view="labeling">Labeling</button><button class="tab" data-view="review">Review</button><button class="tab" data-view="discarded">Discarded</button><button class="tab" data-view="split-train">Train</button><button class="tab" data-view="split-validation">Validation</button><button class="tab" data-view="split-test">Test</button></nav>
  <main>
    <section id="labeling" class="view active">
      <div class="toolbar"><label for="sort">Order <select id="sort"><option value="unlabeled">Unlabeled first</option><option value="high">Waviness: highest to lowest</option><option value="low">Waviness: lowest to highest</option><option value="filename">Filename</option></select></label><span id="position" class="status"></span></div>
      <div class="label-card"><div class="image-row"><div id="original-wrap" class="original-wrap"><img id="original-image" alt="Original photo"><div id="original-missing" class="original-missing" hidden>Original not found</div><span class="caption">Original</span></div><div class="image-wrap"><img id="main-image" alt="Processed sea image"></div></div><div class="image-meta"><div class="image-identification"><span id="filename" class="filename"></span><span id="capture-date" class="capture-date"></span></div><span id="label-status" class="status"></span></div><div class="slider-row"><span class="range-label">0.00</span><input id="slider" type="range" min="0" max="1" step="0.05" value="0.50"><span id="value" class="value">0.50</span></div><div class="actions"><button id="previous">Previous</button><div class="right"><button id="skip">Skip</button><button id="discard" class="danger">Discard</button><button id="save" class="primary">Save &amp; Next</button></div></div><div class="hint">Keyboard: 1–9 = 0.10–0.90 · 0 = 1.00 · ←/→ adjust by 0.05 · Enter save &amp; next · S skip · D discard · P previous</div></div>
    </section>
    <section id="review" class="view"><div class="toolbar"><h2>Labeled images</h2><label for="review-sort">Order <select id="review-sort"><option value="high">Highest to lowest</option><option value="low">Lowest to highest</option></select></label></div><div id="gallery" class="gallery"></div></section>
    <section id="discarded" class="view"><div class="toolbar"><h2>Discarded images</h2><button id="refresh-discarded">Refresh</button></div><div id="discarded-gallery" class="discarded-gallery"></div></section>
    <section id="split-train" class="view split-view" data-split="train"><div class="toolbar"><h2>Train set</h2><div class="split-controls"><label>Dataset split <select class="snapshot-select" aria-label="Dataset split version"></select></label><button class="refresh-splits">Refresh versions</button></div></div><div class="split-summary"></div><div class="split-gallery"></div></section>
    <section id="split-validation" class="view split-view" data-split="validation"><div class="toolbar"><h2>Validation set</h2><div class="split-controls"><label>Dataset split <select class="snapshot-select" aria-label="Dataset split version"></select></label><button class="refresh-splits">Refresh versions</button></div></div><div class="split-summary"></div><div class="split-gallery"></div></section>
    <section id="split-test" class="view split-view" data-split="test"><div class="toolbar"><h2>Test set</h2><div class="split-controls"><label>Dataset split <select class="snapshot-select" aria-label="Dataset split version"></select></label><button class="refresh-splits">Refresh versions</button></div></div><div class="split-summary"></div><div class="split-gallery"></div></section>
  </main>
  <div id="discard-dialog" class="dialog-backdrop" role="presentation" hidden><section class="dialog" role="dialog" aria-modal="true" aria-labelledby="discard-title"><h2 id="discard-title">Discard image?</h2><p id="discard-description"></p><label for="discard-reason">Reason (optional)</label><textarea id="discard-reason" maxlength="1000"></textarea><div class="dialog-actions"><button id="cancel-discard">Cancel</button><button id="confirm-discard" class="danger">Discard image</button></div></section></div>
  <div id="message" class="message" role="alert" hidden></div>
  <script>
    const state = { images: [], labels: {}, discarded: 0, ordered: [], index: 0, sort: 'unlabeled', discardName: null, captureDates: {}, snapshots: [], selectedSnapshot: null, splitData: {} };
    const $ = (id) => document.getElementById(id);
    const scoreText = (value) => Number(value).toFixed(2);
    const escapeHtml = (value) => String(value).replace(/[&<>"']/g,character=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[character]));
    function orderedImages() {
      const list = [...state.images];
      const score = (name) => state.labels[name];
      if (state.sort === 'unlabeled') return list.sort((a,b) => (score(a) === undefined) - (score(b) === undefined) || a.localeCompare(b));
      if (state.sort === 'high') return list.sort((a,b) => (score(b) ?? -1) - (score(a) ?? -1) || a.localeCompare(b));
      if (state.sort === 'low') return list.sort((a,b) => (score(a) ?? 2) - (score(b) ?? 2) || a.localeCompare(b));
      return list.sort((a,b) => a.localeCompare(b));
    }
    function updateStats() {
      const activeLabels=state.images.filter(name=>state.labels[name]!==undefined).map(name=>Number(state.labels[name]));
      const labeled=activeLabels.length;
      $('total').textContent=state.images.length; $('labeled').textContent=labeled;
      $('remaining').textContent=state.images.length-labeled; $('discarded').textContent=state.discarded;
      $('group-low').textContent=activeLabels.filter(value=>value<=0.33).length;
      $('group-mid').textContent=activeLabels.filter(value=>value>0.33&&value<=0.66).length;
      $('group-high').textContent=activeLabels.filter(value=>value>0.66).length;
    }
    function showMessage(message) { $('message').textContent=message; $('message').hidden=false; clearTimeout(showMessage.timeout); showMessage.timeout=setTimeout(()=>$('message').hidden=true,5000); }
    function renderMain() {
      state.ordered = orderedImages();
      if (!state.ordered.length) { $('filename').textContent='No processed images found'; $('capture-date').textContent=''; $('position').textContent=''; $('main-image').removeAttribute('src'); $('original-image').removeAttribute('src'); return; }
      state.index = Math.max(0, Math.min(state.index, state.ordered.length-1)); const name=state.ordered[state.index]; const labeled=state.labels[name] !== undefined;
      $('main-image').src='/images/'+encodeURIComponent(name); $('main-image').alt=name; showOriginal(name); $('filename').textContent=name; $('position').textContent=`${state.index+1} of ${state.ordered.length}`;
      loadCaptureDate(name);
      $('label-status').textContent=labeled ? 'Labeled' : 'UNLABELED'; $('label-status').className='status'+(labeled?'':' unlabeled'); $('slider').value=labeled ? state.labels[name] : 0.50; $('value').textContent=scoreText($('slider').value);
    }
    function showOriginal(name) {
      $('original-image').hidden=false; $('original-missing').hidden=true; $('original-image').src='/originals/'+encodeURIComponent(name);
      // Warm the server cache for the next image; HEIC decoding is the slow part.
      const next=state.ordered[state.index+1]; if(next) new Image().src='/originals/'+encodeURIComponent(next);
    }
    async function loadCaptureDate(name) {
      if(Object.prototype.hasOwnProperty.call(state.captureDates,name)) { $('capture-date').textContent=state.captureDates[name] ? `Taken ${state.captureDates[name]}` : 'Date unavailable'; return; }
      $('capture-date').textContent='Reading date…';
      try {
        const response=await fetch('/api/capture-date/'+encodeURIComponent(name));
        if(!response.ok) throw new Error('Capture date unavailable');
        const data=await response.json(); state.captureDates[name]=data.date||null;
        if(state.ordered[state.index]===name) $('capture-date').textContent=data.date ? `Taken ${data.date}` : 'Date unavailable';
      } catch (_) { state.captureDates[name]=null; if(state.ordered[state.index]===name) $('capture-date').textContent='Date unavailable'; }
    }
    function renderGallery() {
      const high=$('review-sort').value==='high';
      const labeled=state.images.filter(n=>state.labels[n]!==undefined).sort((a,b)=>(high?state.labels[b]-state.labels[a]:state.labels[a]-state.labels[b])||a.localeCompare(b));
      $('gallery').innerHTML=labeled.length ? labeled.map(n=>`<article class="tile"><button class="tile-select" data-name="${encodeURIComponent(n)}" aria-label="Edit label for ${n}"><img src="/images/${encodeURIComponent(n)}" alt=""><div class="score">${scoreText(state.labels[n])}</div><div class="tile-name" title="${n}">${n}</div></button><button class="tile-discard" data-discard="${encodeURIComponent(n)}" aria-label="Discard ${n}" title="Discard image">×</button></article>`).join('') : '<div class="empty">No labels saved yet.</div>';
      document.querySelectorAll('.tile-select').forEach(el=>el.onclick=()=>selectForLabeling(decodeURIComponent(el.dataset.name)));
      document.querySelectorAll('.tile-discard').forEach(el=>el.onclick=()=>openDiscardDialog(decodeURIComponent(el.dataset.discard)));
    }
    function selectForLabeling(name) { state.sort='filename'; $('sort').value='filename'; state.ordered=orderedImages(); state.index=state.ordered.indexOf(name); showView('labeling'); renderMain(); }
    function splitLabel(snapshot) { return snapshot.created_at_utc.replace('T',' ').replace('Z',' UTC'); }
    async function loadSplitCatalog(selectLatest=false) {
      const response=await fetch('/api/splits'); if(!response.ok) throw new Error('Could not load dataset split versions.');
      const data=await response.json(); state.snapshots=data.snapshots;
      const previous=state.selectedSnapshot;
      state.selectedSnapshot=selectLatest||!state.snapshots.some(s=>s.id===previous) ? (data.latest||null) : previous;
      document.querySelectorAll('.snapshot-select').forEach(select=>{
        select.innerHTML=state.snapshots.map(s=>`<option value="${encodeURIComponent(s.id)}">${escapeHtml(splitLabel(s))}</option>`).join('');
        select.disabled=!state.snapshots.length;
        if(state.selectedSnapshot) select.value=encodeURIComponent(state.selectedSnapshot);
      });
      return state.selectedSnapshot;
    }
    function renderSplitSummary(container, stats, snapshot) {
      const entries=[['Snapshot',snapshot],['Total labeled',stats.total_labeled_samples],['Independent groups',stats.independent_groups],['Images in split',stats.count],['Groups in split',stats.group_count],['Dataset share',stats.percentage],['Waviness min / max / mean',stats.waviness]];
      container.innerHTML=entries.map(([label,value])=>`<div class="split-stat"><span>${label}</span><strong>${escapeHtml(value)}</strong></div>`).join('')+`<div class="split-bins"><strong>Waviness bins:</strong> ${stats.bins.map(bin=>`${escapeHtml(bin.range)}: ${escapeHtml(bin.count)}`).join(' · ')}</div>`;
    }
    async function renderSplit(kind, force=false) {
      const view=document.querySelector(`.split-view[data-split="${kind}"]`); const gallery=view.querySelector('.split-gallery'); const summary=view.querySelector('.split-summary');
      if(!state.snapshots.length) await loadSplitCatalog(true);
      const version=state.selectedSnapshot;
      if(!version){gallery.innerHTML='<div class="empty">No complete dataset split snapshots found.</div>';summary.innerHTML='';return;}
      const cacheKey=`${version}:${kind}`; let data=state.splitData[cacheKey];
      if(force||!data){
        gallery.innerHTML='<div class="empty">Loading split images…</div>';
        const query=new URLSearchParams({version,split:kind}); const response=await fetch('/api/splits/data?'+query);
        if(!response.ok) throw new Error('Could not load the selected dataset split.');
        data=await response.json(); state.splitData[cacheKey]=data;
      }
      renderSplitSummary(summary,data.stats,version);
      gallery.innerHTML=data.samples.length ? data.samples.map(sample=>`<article class="split-tile"><img loading="lazy" src="/split-images/${encodeURIComponent(version)}/${kind}/${encodeURIComponent(sample.filename)}" alt="${escapeHtml(sample.filename)}"><div class="score">${scoreText(sample.waviness)}</div><div class="tile-name" title="${escapeHtml(sample.filename)}">${escapeHtml(sample.filename)}</div></article>`).join('') : '<div class="empty">This split contains no images.</div>';
    }
    function showSplitError(kind,error) { const view=document.querySelector(`.split-view[data-split="${kind}"]`); const message=document.createElement('div'); message.className='empty'; message.textContent=error.message||'Could not load dataset split.'; view.querySelector('.split-gallery').replaceChildren(message); }
    async function reloadDiscarded() {
      const gallery=$('discarded-gallery'); gallery.innerHTML='<div class="empty">Loading discarded images…</div>';
      const response=await fetch('/api/discarded'); if(!response.ok) throw new Error('Could not load discarded images.');
      const data=await response.json();
      gallery.innerHTML=data.images.length ? data.images.map(item=>`<article class="discarded-tile"><img loading="lazy" src="/discarded-images/${encodeURIComponent(item.filename)}" alt="${escapeHtml(item.filename)}"><div class="tile-name" title="${escapeHtml(item.filename)}">${escapeHtml(item.filename)}</div><div class="discarded-reason">Reason: ${escapeHtml(item.reason||'Not provided')}</div><div class="discarded-time">Discarded: ${escapeHtml(item.discarded_at_utc||'Unknown')}</div><div class="restore-label-note">${item.restore_error?escapeHtml(item.restore_error):item.previous_waviness===null?'Will return as unlabeled':`Saved label ${scoreText(item.previous_waviness)} will be restored`}</div><button class="restore-image" data-name="${encodeURIComponent(item.filename)}" ${item.restore_error?'disabled':''}>Restore image</button></article>`).join('') : '<div class="empty">No discarded images.</div>';
      document.querySelectorAll('.restore-image').forEach(button=>button.onclick=()=>restoreImage(decodeURIComponent(button.dataset.name)));
    }
    async function restoreImage(name) {
      const response=await fetch('/api/restore',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({filename:name})});
      if(!response.ok){let detail='Could not restore image.';try{const body=await response.json();if(body.error)detail=body.error;}catch(_){}showMessage(detail);return;}
      const result=await response.json(); await reload(); await reloadDiscarded();
      showMessage(result.restored_waviness===null?'Image restored as unlabeled.':`Image restored with its saved label ${scoreText(result.restored_waviness)}.`);
    }
    function showView(view) { document.querySelectorAll('.view').forEach(el=>el.classList.toggle('active',el.id===view)); document.querySelectorAll('.tab').forEach(el=>el.classList.toggle('active',el.dataset.view===view)); if(view==='review') renderGallery(); if(view==='discarded')reloadDiscarded().catch(error=>{const message=document.createElement('div');message.className='empty';message.textContent=error.message;$('discarded-gallery').replaceChildren(message);}); if(view.startsWith('split-')) renderSplit(view.slice(6)).catch(error=>showSplitError(view.slice(6),error)); }
    async function reload() { const data=await fetch('/api/data').then(r=>r.json()); state.images=data.images; state.labels=data.labels; state.discarded=data.discarded_count; updateStats(); renderMain(); renderGallery(); }
    async function save() { const name=state.ordered[state.index]; if(!name)return; const waviness=Number($('slider').value).toFixed(2); const response=await fetch('/api/labels',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({filename:name,waviness})}); if(!response.ok){let detail='Could not save label.'; try { const body=await response.json(); if(body.error) detail=body.error; } catch (_) {} showMessage(detail);return;} const nextUnlabeled=state.images.find(n=>state.labels[n]===undefined && n!==name); await reload(); if(nextUnlabeled && state.sort==='unlabeled') state.index=state.ordered.indexOf(nextUnlabeled); else state.index=Math.min(state.index+1,state.ordered.length-1); renderMain(); }
    function openDiscardDialog(name) { state.discardName=name; $('discard-description').textContent=`${name} will be excluded from labeling, training, and evaluation.`; $('discard-reason').value=''; $('discard-dialog').hidden=false; $('cancel-discard').focus(); }
    function closeDiscardDialog() { $('discard-dialog').hidden=true; state.discardName=null; }
    async function discard(name) { if(!name)return; const reason=$('discard-reason').value.trim(); const response=await fetch('/api/discard',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({filename:name,reason})}); if(!response.ok){let detail='Could not discard image.'; try { const body=await response.json(); if(body.error) detail=body.error; } catch (_) {} closeDiscardDialog(); showMessage(detail);return;} closeDiscardDialog(); await reload(); if(state.sort==='filename') state.index=Math.min(state.index,state.ordered.length-1); renderMain(); }
    $('original-image').onerror=()=>{ $('original-image').hidden=true; $('original-missing').hidden=false; };
    $('slider').oninput=()=> $('value').textContent=scoreText($('slider').value); $('sort').onchange=()=>{state.sort=$('sort').value;state.index=0;renderMain();}; $('review-sort').onchange=renderGallery; $('save').onclick=save; $('discard').onclick=()=>openDiscardDialog(state.ordered[state.index]); $('confirm-discard').onclick=()=>discard(state.discardName); $('cancel-discard').onclick=closeDiscardDialog; $('skip').onclick=()=>{state.index=Math.min(state.index+1,state.ordered.length-1);renderMain();}; $('previous').onclick=()=>{state.index=Math.max(state.index-1,0);renderMain();}; document.querySelectorAll('.tab').forEach(el=>el.onclick=()=>showView(el.dataset.view));
    $('discard-dialog').onclick=e=>{if(e.target===$('discard-dialog'))closeDiscardDialog();};
    $('refresh-discarded').onclick=()=>reloadDiscarded().catch(error=>showMessage(error.message));
    document.querySelectorAll('.snapshot-select').forEach(select=>select.onchange=()=>{state.selectedSnapshot=decodeURIComponent(select.value);state.splitData={};document.querySelectorAll('.snapshot-select').forEach(other=>other.value=select.value);const active=document.querySelector('.split-view.active');if(active)renderSplit(active.dataset.split).catch(error=>showSplitError(active.dataset.split,error));});
    document.querySelectorAll('.refresh-splits').forEach(button=>button.onclick=async()=>{state.splitData={};try{await loadSplitCatalog(false);const active=document.querySelector('.split-view.active');if(active)await renderSplit(active.dataset.split,true);}catch(error){const active=document.querySelector('.split-view.active');if(active)showSplitError(active.dataset.split,error);}});
    document.addEventListener('keydown', e=>{if(!$('discard-dialog').hidden){if(e.key==='Escape')closeDiscardDialog();return;} const focused=document.activeElement; if(['INPUT','TEXTAREA','SELECT'].includes(focused.tagName) && focused.id!=='slider')return; if(!document.getElementById('labeling').classList.contains('active'))return; if(/^[0-9]$/.test(e.key)){e.preventDefault();$('slider').value=e.key==='0'?1:Number(e.key)/10;$('value').textContent=scoreText($('slider').value);} else if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();$('slider').value=Math.max(0,Math.min(1,Number($('slider').value)+(e.key==='ArrowRight'?0.05:-0.05)));$('value').textContent=scoreText($('slider').value);} else if(e.key==='Enter')save(); else if(e.key.toLowerCase()==='s'){$('skip').click();} else if(e.key.toLowerCase()==='d'){$('discard').click();} else if(e.key.toLowerCase()==='p'){$('previous').click();}});
    reload().then(()=>loadSplitCatalog(true)).catch(()=>showMessage('Could not load labeling data or split versions.'));
  </script>
</body>
</html>'''


def image_names() -> list[str]:
    """Return only actual final pipeline outputs, excluding comparison files."""
    discarded = read_discarded()
    return sorted(
        path.name for path in IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        and path.name.startswith("step-2_")
        and "original" not in path.stem.lower()
        and "overlay" not in path.stem.lower()
        and path.name not in discarded
    ) if IMAGE_DIR.exists() else []


def source_for_processed(filename: str) -> Path:
    """Map `step-2_<stem>.jpg` back to the raw source recorded by step 1."""
    stem = Path(filename).stem
    if not stem.startswith(PROCESSED_PREFIX):
        raise ValueError(f"Expected a '{PROCESSED_PREFIX}<stem>' filename, got {filename!r}")
    stem = stem.removeprefix(PROCESSED_PREFIX)
    if not SOURCE_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Source manifest not found: {SOURCE_MANIFEST_PATH}")
    # Re-read on every request so sources added by a new step 1 run are found.
    manifest = json.loads(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    sources = [Path(source) for source, mapped_stem in manifest.items() if mapped_stem == stem]
    if not sources:
        raise FileNotFoundError(f"No source recorded for stem {stem!r} in {SOURCE_MANIFEST_PATH}")
    if not sources[0].exists():
        raise FileNotFoundError(f"Recorded source no longer exists: {sources[0]}")
    return sources[0]


@lru_cache(maxsize=512)
def capture_date_for_source(source: Path, modified_ns: int) -> str | None:
    """Read an EXIF capture date from a source photo and format it as DD MM YYYY."""
    try:
        with Image.open(source) as image:
            exif = image.getexif()
            try:
                exif_ifd = exif.get_ifd(34665)
            except (AttributeError, KeyError, TypeError):
                exif_ifd = {}
            for raw in (exif_ifd.get(36867), exif.get(36867), exif_ifd.get(36868), exif.get(36868), exif.get(306)):
                if not raw:
                    continue
                if isinstance(raw, bytes):
                    raw = raw.decode("ascii", errors="ignore")
                text = str(raw).strip()
                try:
                    captured = datetime.strptime(text[:19], "%Y:%m:%d %H:%M:%S")
                except ValueError:
                    try:
                        captured = datetime.fromisoformat(text.replace("Z", "+00:00"))
                    except ValueError:
                        continue
                return captured.strftime("%d %m %Y")
    except (OSError, ValueError, TypeError):
        return None
    return None


@lru_cache(maxsize=64)
def original_preview_bytes(source: Path, modified_ns: int) -> bytes:
    """Decode (HEIC included) and downscale an original for display only; nothing is written to disk."""
    started = time.perf_counter()
    image = load_rgb_image(source)
    image.thumbnail((ORIGINAL_PREVIEW_MAX_SIDE, ORIGINAL_PREVIEW_MAX_SIDE))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    print(f"[{time.strftime('%H:%M:%S')}] original preview {source.name} decoded in {time.perf_counter() - started:.2f}s")
    return buffer.getvalue()


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


def read_discarded_records() -> dict[str, dict[str, str]]:
    if not DISCARDED_PATH.exists():
        return {}
    result: dict[str, dict[str, str]] = {}
    with DISCARDED_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = (row.get("filename") or "").strip()
            if name:
                result[name] = {
                    "reason": (row.get("reason") or "").strip(),
                    "discarded_at_utc": (row.get("discarded_at_utc") or "").strip(),
                    "waviness": (row.get("waviness") or "").strip(),
                }
    return result


def read_discarded() -> set[str]:
    return set(read_discarded_records())


def atomic_write_labels(labels: dict[str, float]) -> None:
    fd, temporary = tempfile.mkstemp(prefix="labels.", suffix=".csv", dir=LABELS_PATH.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["filename", "waviness"])
            for name in sorted(labels):
                writer.writerow([name, f"{labels[name]:.2f}"])
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, LABELS_PATH)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_discarded(records: dict[str, dict[str, str]]) -> None:
    fd, temporary = tempfile.mkstemp(prefix="discarded.", suffix=".csv", dir=DISCARDED_PATH.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["filename", "reason", "discarded_at_utc", "waviness"])
            for name in sorted(records):
                record = records[name]
                writer.writerow([name, record.get("reason", ""), record.get("discarded_at_utc", ""), record.get("waviness", "")])
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, DISCARDED_PATH)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def recover_discarded_label(filename: str, record: dict[str, str]) -> float | None:
    """Recover a removed label from its discard record or immutable/current split CSVs."""
    raw = record.get("waviness", "").strip()
    if raw:
        try:
            value = float(raw)
        except ValueError as error:
            raise ValueError(f"Saved label for {filename} is invalid") from error
        if not 0 <= value <= 1:
            raise ValueError(f"Saved label for {filename} is outside [0, 1]")
        return value

    candidates: set[float] = set()
    paths = list(SPLIT_OUTPUT_DIR.glob("*.csv")) + list(SPLIT_SNAPSHOT_DIR.glob("*/*.csv"))
    for path in paths:
        if path.name not in {"train.csv", "validation.csv", "test.csv"}:
            continue
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if (row.get("filename") or "").strip() == filename:
                        try:
                            value = float(row["waviness"])
                        except (KeyError, TypeError, ValueError) as error:
                            raise ValueError(f"Invalid stored label for {filename} in {path}") from error
                        if not 0 <= value <= 1:
                            raise ValueError(f"Stored label for {filename} is outside [0, 1] in {path}")
                        candidates.add(value)
        except OSError:
            continue
    if len(candidates) > 1:
        raise ValueError(f"Conflicting historical labels found for {filename}; image was not restored")
    return next(iter(candidates)) if candidates else None


def discard_image(filename: str, reason: str) -> None:
    """Record a discard before removing its label so a failed write cannot lose it."""
    with LABELS_LOCK:
        labels = read_labels()
        records = read_discarded_records()
        previous = labels.get(filename)
        records[filename] = {
            "reason": reason,
            "discarded_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "waviness": f"{previous:.2f}" if previous is not None else "",
        }
        atomic_write_discarded(records)
        if previous is not None:
            labels.pop(filename)
            atomic_write_labels(labels)


def restore_discarded_image(filename: str) -> float | None:
    """Restore active status, preserving a recoverable label and never touching image data."""
    with LABELS_LOCK:
        records = read_discarded_records()
        if filename not in records:
            raise ValueError("Image is not in the discarded list")
        path = IMAGE_DIR / filename
        if (Path(filename).name != filename or not filename.startswith("step-2_")
                or "original" in Path(filename).stem.lower() or "overlay" in Path(filename).stem.lower()
                or path.suffix.lower() not in IMAGE_EXTENSIONS or not path.is_file()):
            raise ValueError("The processed image file is missing or invalid; nothing was changed")
        waviness = recover_discarded_label(filename, records[filename])
        if waviness is not None:
            labels = read_labels()
            labels[filename] = waviness
            atomic_write_labels(labels)
        records.pop(filename)
        atomic_write_discarded(records)
        return waviness


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


def remove_label(filename: str) -> None:
    with LABELS_LOCK:
        labels = read_labels()
        if filename not in labels:
            return
        labels.pop(filename)
        fd, temporary = tempfile.mkstemp(prefix="labels.", suffix=".csv", dir=LABELS_PATH.parent)
        try:
            with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["filename", "waviness"])
                for name in sorted(labels):
                    writer.writerow([name, f"{labels[name]:.2f}"])
                handle.flush(); os.fsync(handle.fileno())
            os.replace(temporary, LABELS_PATH)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def split_snapshots() -> list[dict[str, str]]:
    """List complete immutable split snapshots in chronological order."""
    snapshots: list[dict[str, str]] = []
    if not SPLIT_SNAPSHOT_DIR.is_dir():
        return snapshots
    for directory in SPLIT_SNAPSHOT_DIR.iterdir():
        if not directory.is_dir():
            continue
        manifest_path = directory / "manifest.json"
        summary_path = directory / "summary.txt"
        try:
            metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
            snapshot_id = str(metadata["snapshot"])
            created_at = str(metadata["created_at_utc"])
            if snapshot_id != directory.name:
                raise ValueError("snapshot id does not match directory")
            datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            files = metadata["files"]
            for split in ("train", "validation", "test"):
                entry = files[split]
                expected_path = f"snapshots/{snapshot_id}/{split}.csv"
                if entry["path"] != expected_path or not (directory / f"{split}.csv").is_file():
                    raise ValueError(f"missing or unexpected {split} manifest path")
            if not summary_path.is_file():
                raise ValueError("missing summary.txt")
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            print(f"[{time.strftime('%H:%M:%S')}] skipping incomplete split snapshot {directory.name}: {error}")
            continue
        snapshots.append({"id": snapshot_id, "created_at_utc": created_at})
    return sorted(snapshots, key=lambda item: (item["created_at_utc"], item["id"]))


def read_snapshot_split(snapshot_id: str, split: str) -> list[dict[str, object]]:
    """Read one split from a validated snapshot; never read the mutable latest CSVs."""
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    snapshot = next((item for item in split_snapshots() if item["id"] == snapshot_id), None)
    if snapshot is None:
        raise FileNotFoundError(f"Dataset split snapshot not found: {snapshot_id}")
    path = SPLIT_SNAPSHOT_DIR / snapshot_id / f"{split}.csv"
    samples: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["filename", "waviness"]:
            raise ValueError(f"{path} must have header: filename,waviness")
        for row_number, row in enumerate(reader, start=2):
            filename = (row.get("filename") or "").strip()
            try:
                value = float(row["waviness"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path}:{row_number}: invalid waviness") from error
            if not filename or Path(filename).name != filename or not 0 <= value <= 1:
                raise ValueError(f"{path}:{row_number}: invalid filename or waviness")
            samples.append({"filename": filename, "waviness": value})
    return samples


def read_snapshot_stats(snapshot_id: str, split: str) -> dict[str, object]:
    """Read counts and waviness statistics from the split pipeline's summary.txt."""
    summary_path = SPLIT_SNAPSHOT_DIR / snapshot_id / "summary.txt"
    text = summary_path.read_text(encoding="utf-8")
    total_match = re.search(r"^total labeled samples: (\d+)$", text, re.MULTILINE)
    groups_match = re.search(r"^number of independent groups: (\d+)$", text, re.MULTILINE)
    split_match = re.search(rf"^{re.escape(split)}: (\d+) samples \(([\d.]+)%\)$", text, re.MULTILINE)
    waviness_match = re.search(
        rf"^{re.escape(split)}:\s*\d+ samples[^\n]*\n  waviness min/max/mean: ([^\n]+)", text, re.MULTILINE
    )
    bins_match = re.search(rf"^{re.escape(split)}:\s*\d+ samples[^\n]*\n(?:  [^\n]*\n)*  waviness bins: ([^\n]+)", text, re.MULTILINE)
    if not all((total_match, groups_match, split_match, waviness_match, bins_match)):
        raise ValueError(f"Could not parse {split} statistics from {summary_path}")
    group_line_match = re.search(rf"^{re.escape(split)}:\s*\d+ samples[^\n]*\n(?:  [^\n]*\n)*  groups: ([^\n]*)", text, re.MULTILINE)
    group_count = len([value for value in (group_line_match.group(1).split(",") if group_line_match else []) if value.strip()])
    bins = []
    for entry in bins_match.group(1).split(","):
        range_name, separator, count = entry.strip().rpartition("=")
        if not separator or not count.isdigit():
            raise ValueError(f"Invalid waviness bin in {summary_path}: {entry}")
        bins.append({"range": range_name, "count": int(count)})
    values = waviness_match.group(1)
    if values == "n/a":
        values = "n/a"
    else:
        values = " / ".join(values.split("/"))
    return {
        "total_labeled_samples": int(total_match.group(1)),
        "independent_groups": int(groups_match.group(1)),
        "count": int(split_match.group(1)),
        "percentage": f"{split_match.group(2)}%",
        "waviness": values,
        "group_count": group_count,
        "bins": bins,
    }


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
            labels = read_labels(); self.send_json({"images": image_names(), "labels": labels, "discarded_count": len(read_discarded())}); return
        if parsed.path == "/api/discarded":
            images = []
            for filename, record in read_discarded_records().items():
                item = {"filename": filename, "reason": record["reason"], "discarded_at_utc": record["discarded_at_utc"], "previous_waviness": None, "restore_error": None}
                path = IMAGE_DIR / filename
                if Path(filename).name != filename or not path.is_file():
                    item["restore_error"] = "Processed image file is missing."
                else:
                    try:
                        item["previous_waviness"] = recover_discarded_label(filename, record)
                    except ValueError as error:
                        item["restore_error"] = str(error)
                images.append(item)
            images.sort(key=lambda item: (item["discarded_at_utc"], item["filename"]), reverse=True)
            self.send_json({"images": images}); return
        if parsed.path.startswith("/api/capture-date/"):
            requested = unquote(parsed.path.removeprefix("/api/capture-date/"))
            if Path(requested).name != requested or requested not in image_names():
                self.send_error(HTTPStatus.NOT_FOUND); return
            try:
                source = source_for_processed(requested)
                capture_date = capture_date_for_source(source, source.stat().st_mtime_ns)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                print(f"[{time.strftime('%H:%M:%S')}] capture date unavailable for {requested}: {error}")
                capture_date = None
            self.send_json({"date": capture_date}); return
        if parsed.path.startswith("/discarded-images/"):
            requested = unquote(parsed.path.removeprefix("/discarded-images/"))
            path = IMAGE_DIR / requested
            if (Path(requested).name != requested or requested not in read_discarded()
                    or path.suffix.lower() not in IMAGE_EXTENSIONS or not path.is_file()):
                self.send_error(HTTPStatus.NOT_FOUND); return
            body = path.read_bytes()
            content_type = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}[path.suffix.lower()]
            self.send_response(HTTPStatus.OK); self.send_header("Content-Type", content_type); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
        if parsed.path == "/api/splits":
            snapshots = split_snapshots()
            self.send_json({"snapshots": snapshots, "latest": snapshots[-1]["id"] if snapshots else None}); return
        if parsed.path == "/api/splits/data":
            query = parse_qs(parsed.query)
            snapshot_id = query.get("version", [""])[0]
            split = query.get("split", [""])[0]
            try:
                samples = read_snapshot_split(snapshot_id, split)
                stats = read_snapshot_stats(snapshot_id, split)
            except (OSError, ValueError) as error:
                self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST); return
            self.send_json({"samples": samples, "stats": stats}); return
        if parsed.path.startswith("/split-images/"):
            parts = parsed.path.split("/")
            if len(parts) != 5:
                self.send_error(HTTPStatus.NOT_FOUND); return
            snapshot_id, split, requested = parts[2], parts[3], unquote(parts[4])
            if Path(requested).name != requested or not requested.startswith("step-2_"):
                self.send_error(HTTPStatus.NOT_FOUND); return
            try:
                members = read_snapshot_split(snapshot_id, split)
            except (OSError, ValueError):
                self.send_error(HTTPStatus.NOT_FOUND); return
            if requested not in {sample["filename"] for sample in members}:
                self.send_error(HTTPStatus.NOT_FOUND); return
            path = IMAGE_DIR / requested
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                self.send_error(HTTPStatus.NOT_FOUND); return
            body = path.read_bytes()
            content_type = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}[path.suffix.lower()]
            self.send_response(HTTPStatus.OK); self.send_header("Content-Type", content_type); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
        if parsed.path.startswith("/images/"):
            requested = Path(unquote(parsed.path.removeprefix("/images/"))).name
            if requested != unquote(parsed.path.removeprefix("/images/")) or requested not in image_names():
                self.send_error(HTTPStatus.NOT_FOUND); return
            path = IMAGE_DIR / requested
            body = path.read_bytes(); content_type = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
            self.send_response(HTTPStatus.OK); self.send_header("Content-Type", content_type); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
        if parsed.path.startswith("/originals/"):
            requested = unquote(parsed.path.removeprefix("/originals/"))
            if requested not in image_names():
                self.send_error(HTTPStatus.NOT_FOUND); return
            try:
                source = source_for_processed(requested)
                body = original_preview_bytes(source, source.stat().st_mtime_ns)
            except (OSError, ValueError) as error:
                print(f"[{time.strftime('%H:%M:%S')}] original preview unavailable for {requested}: {error}")
                self.send_error(HTTPStatus.NOT_FOUND); return
            self.send_response(HTTPStatus.OK); self.send_header("Content-Type", "image/jpeg"); self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "max-age=3600"); self.end_headers(); self.wfile.write(body); return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        endpoint = urlparse(self.path).path
        if endpoint not in {"/api/labels", "/api/discard", "/api/restore"}: self.send_error(HTTPStatus.NOT_FOUND); return
        try:
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            filename = str(payload["filename"])
            if endpoint == "/api/restore":
                restored_waviness = restore_discarded_image(filename)
                self.send_json({"ok": True, "restored_waviness": restored_waviness}); return
            if endpoint == "/api/discard":
                if filename not in image_names():
                    raise ValueError
                discard_image(filename, str(payload.get("reason", "")).strip())
                self.send_json({"ok": True}); return
            value = float(payload["waviness"])
            # Validate by integer step count with a tolerance; binary floating-point
            # representation makes values such as 0.35 / 0.05 slightly imprecise.
            step_count = round(value * 20)
            if filename not in image_names() or not 0 <= value <= 1 or abs(value * 20 - step_count) > 1e-9:
                raise ValueError
            write_label(filename, step_count / 20); self.send_json({"ok": True})
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError) as error:
            detail = str(error) if endpoint == "/api/restore" else "filename must be a processed image and waviness must be 0.00–1.00 in 0.05 steps"
            self.send_json({"error": detail}, HTTPStatus.BAD_REQUEST)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local sea-waviness labeling app")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8055)
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
