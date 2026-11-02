# From One to Zero — baseline preparation

This directory establishes timing and output baselines for the current v19
`fast-v2` prediction path. It is preparation for a later speed-improvement
experiment; it does not implement a speed change.

The runner imports the canonical prediction and preprocessing functions. It
validates the v19 checkpoint's recorded preprocessing contract, uses two raw
photos from `step-7-predict-captures-holder/`, performs one warm-up and three
measured fresh-preprocessing predictions per photo, and writes timestamped JSON
and standalone HTML reports here. CUDA/ROCm operations are synchronized at
instrumentation boundaries. Startup and model loading are separated from
photo-to-score latency.

The default photos are the first two supported files in case-insensitive
filename order. No source images are copied, and no normal Step 7 preview or
prediction log is written. The experiment does not edit current pipeline files.

Run from the repository root:

```bash
python experiments/from-one-to-zero/benchmark_fast_v2.py
```

Optional positional arguments select exactly two photos explicitly. The raw
JSON retains every measured timing; the HTML summarizes startup, per-operation
median/p95, prediction results, cohort hashes, and runtime settings.
