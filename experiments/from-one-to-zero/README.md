# From One to Zero · speed screens

This directory compares the current fast-v2.1 path with isolated latency experiments. The screen runner uses up to four raw HEIC photos from the prediction holder, performs one warm-up and three measured runs per photo, records model startup separately, and writes timestamped JSON reports under each experiment folder. It also regenerates `index.html`.

Run all candidates from the repository root:

```bash
python experiments/from-one-to-zero/run_screen.py
```

Run one candidate plus a fresh paired baseline:

```bash
python experiments/from-one-to-zero/run_screen.py --only migraphx-fp16
```

The current candidates are FP16 MIGraphX math, ONNX Runtime profiling, a fixed 384 × 384 SegFormer input graph, and HEIC thumbnail plus selected-tile decoding. The HEIC experiment falls back to the existing full-frame route when no embedded thumbnail exists.

These are speed screens, not quality approvals. Scores and crop coordinates are saved for inspection, but four photos are not enough to establish that predictions remain safe. Profiling overhead is also reported as diagnostic only. Do not promote a candidate without a full labeled-cohort quality run.
