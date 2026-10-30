# Distil-FP32 performance improvement experiments

All benchmark code, run reports, candidate configs and quality comparisons for
this investigation belong in this directory. The existing pipeline and
experimental server are treated as read-only references. Input photos remain
in their configured repository data locations; this experiment does not copy
or generate image assets.

## Fine-grained baseline benchmark

`benchmark_distil_fp32.py` imports the existing Distil-FP32 server and uses the
same ONNX artifacts, preprocessing operations, crop selector, and v18
regressor. It measures these top-level categories:

1. **Warm-up:** Python/runtime imports, MIGraphX registration, individual
   session loads, first Step 1 and Step 2 graph executions, regressor warm-up,
   crop-selector JIT warm-up, and process-start-to-models-ready time.
2. **Pre-processing:** raw photo decode through final normalized regressor
   tensor creation. Each operation is timed separately, including both
   segmentation model calls, image and mask resizing, erosion, water bounds,
   mask application, canvases, JPEG encode/decode, crop selection, and tensor
   preparation.
3. **Waviness inference:** the v18 EfficientNet-B0 ONNX Runtime call and the
   returned scalar score.

Operation timings are exclusive leaves. `preprocessing_total_ms` and
`total_raw_to_score_ms` are parent totals and should not be added to their
children. The report includes median, nearest-rank p95, and max for each split,
plus per-photo timing, dimensions, crop coordinates, prediction, label, and
failures. The operation boundaries add timer overhead, so use the parent total
for end-to-end latency and the leaves to find bottlenecks.

The default run evaluates the repository's fixed 47-photo validation and
46-photo test splits from snapshot `20261028T081606614158Z`. Both sets were
excluded from the student training split; validation can be used to select
changes, while test should remain the final holdout. Do not tune repeatedly to
the test scores. The reports record split and label hashes. Only aggregate
quality metrics and source filenames are written; source images are not
copied.

The full-cohort baseline is
`runs/distil-fp32-fine-grained-20261030T105902Z.json`. It reproduced the
recorded quality exactly: validation MAE 0.110818 (47/47 succeeded) and test
MAE 0.110720 (46/46 succeeded). Median raw-to-score latency was 99.61 ms on
validation and 100.47 ms on test in this server environment. Startup took
18.60 s through model readiness; the two segmentation sessions loaded in about
4.80 s and 4.39 s, and their first graph runs took about 4.53 s and 4.41 s.

The operation breakdown shows where to focus next: HEIC decode is about 20 ms;
Step 1 segmentation inference about 21 ms; Step 2 segmentation inference about
14 ms; and the final waviness regressor about 5.4 ms. Other relatively costly
Step 1 operations are input resize (about 6.4 ms), full-size mask expansion
(about 6.0 ms), masked ROI generation (about 3.6 ms), PIL image materialization
(about 7.3 ms), and Lanczos resize (about 5.9 ms). These are medians from one
sequential pass per image; repeat runs are retained in `runs/` to show natural
variation.

Open `distil-fp32-benchmark-report.html` for the human-readable dashboard. It
has validation/test selection, heat-graded operation bars, startup details,
quality metrics, and the ten slowest photos. Regenerate it after a new benchmark
with:

```bash
python experiments/distil-fp32/performance-improvements/render_benchmark_report.py
```

Run from the repository root with the isolated Python / ROCm / MIGraphX
environment used by the Distil-FP32 server:

```bash
LD_LIBRARY_PATH=/tmp/wave_migraphx_312_env/lib/python3.12/site-packages/migraphx_libs:/opt/rocm/lib \
MIGRAPHX_SKIP_BENCHMARKING=1 MIGRAPHX_USE_FAST_SOFTMAX=0 \
WAVE_EXPERIMENT=distilled-b0-migraphx-fp32 \
/tmp/wave_migraphx_312_env/bin/python experiments/distil-fp32/performance-improvements/benchmark_distil_fp32.py
```

Use `--limit 1` only for a smoke run; limited runs are not quality results.
Run the full validation and test cohort before accepting any preprocessing
change. To compare a later report with the baseline predictions, pass
`--compare-reference experiments/distil-fp32/performance-improvements/runs/distil-fp32-fine-grained-20261030T105902Z.json`;
the comparison reports paired prediction deltas, per-photo error changes,
missing/failed cases, and a strict gate. That gate requires the exact same
cohort, no failures, and no validation or test MAE/RMSE or per-photo absolute
error regressions. A limited run cannot pass the full-cohort gate.

The local harness starts at source-photo decode and excludes HTTP upload and
temporary request-file staging. Treat its latency as raw-file-to-score, not
client request latency.

Reports default to `runs/` under this directory. Keep generated images out of
the experiment folder and do not move raw photos or `original` / `overlay`
assets into it.
