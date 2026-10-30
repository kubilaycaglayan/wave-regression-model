# Distil-FP32 performance experiments

This directory holds the shared benchmark and viewer, version-isolated candidate
code, and all experiment reports. The existing prediction pipeline is a
read-only reference. Raw photos stay in the repository's configured data
locations; no source or generated images are copied here.

## Layout

```text
performance-improvements/
├── benchmark_distil_fp32.py          # shared warm-up, timing, and quality runner
├── render_benchmark_report.py        # builds the standalone multi-experiment HTML
├── distil-fp32-benchmark-report.html # individual and comparison views
├── experiment-versions/
│   ├── README.md
│   ├── v000_baseline/                 # immutable reference adapter + manifest
│   └── v001_coordinate_crop_single_output/ # coordinate crop candidate
└── runs/
    ├── v000_baseline/                # immutable JSON reports for that version
    └── v001_coordinate_crop_single_output/
```

Create a new `experiment-versions/v001_<hypothesis>/` directory for the first
change, then `v002_...` for the next. Each version contains its own
`experiment.py` and `experiment.json`; do not overwrite previous experiment
code. The runner saves reports under the matching `runs/<version-id>/` folder.
See [experiment-versions/README.md](experiment-versions/README.md) for the
adapter contract.

## Evaluation policy

**Validation is the default and is the split for choosing changes. Test is
reserved for final checks.** Run `benchmark_distil_fp32.py --final-test` only
after selecting a candidate. Do not tune a candidate based on test results.
The existing 93-photo reports in `runs/v000_baseline/` include historical test
measurements made before this policy; new runs follow the validation-only
default.

Every full-cohort report includes prediction quality, failures, per-photo
source hashes, and latency summaries. Compare a candidate with its paired
baseline report using `--compare-reference`. The strict gate requires the same
photos and labels, no failures, and no split MAE, RMSE, or per-photo absolute
error regressions. Limited runs are for smoke checks and cannot establish
quality.

The fixed dataset snapshot is `20261028T081606614158Z`; its validation and test
rows were excluded from the SegFormer-B0 student training split. The benchmark
records the snapshot and file hashes. Timings are local raw-file-to-score and
exclude HTTP upload and temporary request-file staging.

## Run an experiment

From the repository root, use the isolated MIGraphX environment:

```bash
LD_LIBRARY_PATH=/tmp/wave_migraphx_312_env/lib/python3.12/site-packages/migraphx_libs:/opt/rocm/lib \
MIGRAPHX_SKIP_BENCHMARKING=1 MIGRAPHX_USE_FAST_SOFTMAX=0 \
WAVE_EXPERIMENT=distilled-b0-migraphx-fp32 \
/tmp/wave_migraphx_312_env/bin/python experiments/distil-fp32/performance-improvements/benchmark_distil_fp32.py \
  --experiment-id v000_baseline \
  --compare-reference experiments/distil-fp32/performance-improvements/runs/v000_baseline/distil-fp32-fine-grained-20261030T105902Z.json
```

For final test evaluation, add `--final-test`. Validation runs never evaluate
test implicitly. `--limit 1` can smoke-check a version but does not produce a
full-cohort quality conclusion.

## View reports

The HTML dashboard has an individual timing tree and a side-by-side view of
latency, MAE/RMSE, failures, and operation medians. It discovers all versioned
reports under `runs/`, selects the latest report per version for the chosen
split, and checks whether compared reports have matching photo fingerprints.
Validation is selected by default; the test view is labeled as a final check.

Regenerate the dashboard after benchmark runs:

```bash
python experiments/distil-fp32/performance-improvements/render_benchmark_report.py
```

The historical full-split baseline reported validation MAE **0.110818** and
test MAE **0.110720**, with no failed photos; its raw-to-score medians were
**99.61 ms** and **100.47 ms**, respectively. The new validation-only run
reproduced validation MAE **0.110818**, matched all 47 predictions exactly,
and measured **98.90 ms** median raw-to-score. The test split was not rerun.
Startup through model readiness took **18.60 s**. Detailed records are in
[`runs/v000_baseline/`](runs/v000_baseline/), including the [latest
validation-only report](runs/v000_baseline/distil-fp32-fine-grained-20261030T111422Z.json).

## Experiment 1: coordinate crop with one final image

`v001_coordinate_crop_single_output` used Step 1 mask coordinates to choose a
lower-water ROI on the original photo, then created one final 224 × 224 input
for the waviness regressor. It skipped Step 1's saved image, the intermediate
JPEG round trip, and Step 2 segmentation. The full 47-photo validation cohort
completed without failures; only Step 1 segmentation was loaded. The reserved
test split was not run.

The speed hypothesis was supported: median raw-to-score was **59.56 ms**
(p95 **73.67 ms**), versus **98.90 ms** (p95 **106.95 ms**) for the paired
baseline. Median preprocessing was **53.93 ms**, and median waviness inference
was **5.40 ms**. Startup through model readiness was **9.77 s**, down from
**18.59 s**.

This candidate **failed the validation quality gate**. On the same 47 photos,
MAE increased from **0.110818** to **0.136065** and RMSE from **0.138429** to
**0.168556**; 28 photos had higher absolute error. Keep it as a measured,
rejected experiment and do not select it for final test evaluation. The result
and implementation are in [v001](experiment-versions/v001_coordinate_crop_single_output/),
with the [benchmark report](runs/v001_coordinate_crop_single_output/distil-fp32-fine-grained-20261030T112406Z.json).
