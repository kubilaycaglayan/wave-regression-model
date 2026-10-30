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
│   ├── v001_coordinate_crop_single_output/ # coordinate crop candidate
│   ├── v002_skip_step2_keep_baseline_image/ # isolate Step 2 removal
│   ├── v003_roi_mask_processing/   # ROI-limited Step 1 mask processing
│   ├── v004_working_resolution_cap/ # capped working-photo resolution
│   ├── v005_baseline_canvas_coordinate_crop/ # v001 crop + v002 canvas
│   ├── v006_working_resolution_cap_skip_step2/ # v002 Step 2 removal + v004 cap
│   ├── v007_fast_step1_resize/ # test faster Step 1 interpolation
│   ├── v008_step1_onnx_profile/ # profile Step 1 operators
│   ├── v009_remove_final_jpeg_roundtrip/ # bypass final JPEG decode/encode
│   └── v010_fused_step1_tensor_prep/ # write normalized tensor directly
└── runs/
    ├── v000_baseline/                # immutable JSON reports for that version
    ├── v001_coordinate_crop_single_output/
    ├── v002_skip_step2_keep_baseline_image/
    ├── v003_roi_mask_processing/
    ├── v004_working_resolution_cap/
    ├── v005_baseline_canvas_coordinate_crop/
    ├── v006_working_resolution_cap_skip_step2/
    ├── v007_fast_step1_resize/
    ├── v008_step1_onnx_profile/
    ├── v009_remove_final_jpeg_roundtrip/
    └── v010_fused_step1_tensor_prep/
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

The `whole` cohort is a separate diagnostic split containing all 310 labeled
photos (217 train, 47 validation, 46 test). It overlaps every existing split,
so its aggregate quality is descriptive only and must not be used to select or
tune experiments. Its CSV and provenance are in `datasets/whole.csv` and
`datasets/whole-manifest.json`; run it with `--whole`.

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

For final test evaluation, add `--final-test`. For the descriptive all-labels
cohort, add `--whole`. Validation runs never evaluate test implicitly.
`--limit 1` can smoke-check a version but does not produce a full-cohort quality
conclusion.

## View reports

The HTML dashboard has an individual timing tree and a side-by-side view of
latency, MAE/RMSE, failures, and operation medians. It discovers all versioned
reports under `runs/`, selects the latest report per version for the chosen
split, and checks whether compared reports have matching photo fingerprints.
Validation is selected by default; the test view is labeled as a final check,
and the whole view warns that it includes train, validation, and test photos.

Regenerate the dashboard after benchmark runs:

```bash
python experiments/distil-fp32/performance-improvements/render_benchmark_report.py
```

The v000–v004 preprocessing versions have also been evaluated with matched,
retrained regressors. See the [matched-regressor results](training-runs/matched-regressor-results.md)
and [training tools](training/README.md). All five versions now have final test
checks; the test results do not confirm a quality-preserving latency win.

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

## Experiment 2: skip Step 2, keep the baseline Step 1 image

`v002_skip_step2_keep_baseline_image` preserves v000's Step 1 full-size mask,
erosion, bounding box, masked ROI, 224 × 224 canvas, and JPEG round trip, then
feeds that image directly to the waviness regressor. It loads only Step 1. All
47 validation photos completed without failures; the reserved test split was
not run.

Median raw-to-score improved from **98.90 ms** to **81.00 ms** (p95 **88.49
ms**), and startup through model readiness fell from **18.59 s** to **9.63 s**.
Quality regressed substantially: MAE increased from **0.110818** to **0.171235**
and RMSE from **0.138429** to **0.210184**; 33 of 47 photos had higher absolute
error. The candidate failed the validation gate and is marked rejected. This
isolated result indicates the Step 2 stage materially affects the regressor's
input quality. Details are in [v002](experiment-versions/v002_skip_step2_keep_baseline_image/)
and its [benchmark report](runs/v002_skip_step2_keep_baseline_image/distil-fp32-fine-grained-20261030T112636Z.json).

## Experiment 3: ROI-limited Step 1 mask processing

`v003_roi_mask_processing` kept both segmentation models and the baseline
downstream stages, but estimated the Step 1 water bounds on the native mask
grid and expanded/eroded the mask only in a mapped source ROI. It completed all
47 validation photos without failures; test was not run.

This attempt was unsuccessful. Median raw-to-score was **110.02 ms** versus
**98.90 ms** for v000. The ROI mask expansion operation alone took **18.85 ms**
median. MAE improved slightly (**0.110743** vs **0.110818**), but RMSE increased
(**0.138456** vs **0.138429**) and 7 photos regressed individually, so the
strict quality gate failed. It is marked rejected. See [v003](experiment-versions/v003_roi_mask_processing/)
and its [benchmark report](runs/v003_roi_mask_processing/distil-fp32-fine-grained-20261030T112855Z.json).

## Experiment 4: 2048 px working resolution cap

`v004_working_resolution_cap` kept Step 1's original-photo 448 × 448 model
input and both segmentation stages. It downsampled the source once to a
2048 px maximum-long-side working image for Step 1 mask expansion, erosion,
bounding box, and masked ROI operations, then retained the baseline canvas
and JPEG stages. All 47 validation photos completed without failures; test was
not run.

Median raw-to-score improved modestly from **98.90 ms** to **90.94 ms**, but
quality failed the gate: MAE rose from **0.110818** to **0.122735**, RMSE from
**0.138429** to **0.155863**, and 30 photos regressed individually. The new
downsampling operation itself cost **8.42 ms** median. This version is marked
rejected. See [v004](experiment-versions/v004_working_resolution_cap/) and its
[benchmark report](runs/v004_working_resolution_cap/distil-fp32-fine-grained-20261030T113058Z.json).

## Whole labeled-photo cohort

The `whole` diagnostic cohort includes all 310 labeled photos and was run with
each version's matched trained regressor. Its per-version MAE, RMSE, failure
counts, and latency results are in the [whole cohort results](training-runs/matched-regressor-results.md#whole-labeled-photo-diagnostic-cohort)
and in the dashboard's **Whole · all labeled photos** view. Because it
 overlaps the training and test sets, do not use it as an independent quality
result.

## Experiment 5: baseline canvas with coordinate lower-water crop

`v005_baseline_canvas_coordinate_crop` combines v001's coordinate-based
lower-water crop selection with v002's baseline Step 1 full-resolution mask,
canvas, and JPEG path. It skips Step 2 segmentation, maps the Step 1 mask onto
the 224 × 224 canvas, then crops and resizes the selected lower-water region.
This tests whether retaining v002's Step 1 representation while replacing
Step 2 with a coordinate crop can improve the quality/latency tradeoff. The full 47-photo validation cohort completed without failures: MAE 0.115530,
RMSE 0.151688, median raw-to-score 84.51 ms (p95 94.84 ms). It improved both
aggregate quality metrics over the matched v000 baseline and reduced median
latency by 14.31 ms, but 19 photos regressed individually, so it fails the
strict quality gate. See the [matched-regressor comparison](training-runs/matched-regressor-results.md#validation-comparison)
and [fine-grained report](runs/v005_baseline_canvas_coordinate_crop/distil-fp32-fine-grained-20261030T121926Z.json).

## Experiment 6: 2048 px working cap and skip Step 2

`v006_working_resolution_cap_skip_step2` combines v004's 2048 px cap for Step 1
source-space mask and pixel operations with v002's removal of Step 2 segmentation
and its downstream crop/JPEG stages. It retains the original 448 × 448 input to
Step 1, the capped working photo for source-space processing, and the Step 1
canvas/JPEG round trip before regressor inference. On all 47 validation photos, v006 completed without preprocessing failures and
measured 74.91 ms median raw-to-score (p95 87.48 ms), faster than both v002
(81.32 ms) and v004 (91.72 ms). Quality regressed against retrained v000
(MAE 0.130061 vs. 0.118095; RMSE 0.172008 vs. 0.160190), with 28 photos
regressing individually, so v006 fails the strict quality gate. See the
[matched-regressor comparison](training-runs/matched-regressor-results.md#validation-comparison)
and [fine-grained report](runs/v006_working_resolution_cap_skip_step2/distil-fp32-fine-grained-20261030T122626Z.json).

## v001 follow-up micro-experiments

Four isolated candidates were smoke-benchmarked on the same first five validation
photos (all HEIC), with the v001-matched regressor held fixed. These are one-pass
microbatch results, not full validation results or quality-gate decisions. The
reported quality deltas compare each candidate with v001 on those same five
photos; timing differences across separate processes are directional only.

| Candidate | Raw-to-score median | Targeted observation | MAE / RMSE delta vs. v001 | Per-photo errors worse |
| --- | ---: | --- | ---: | ---: |
| [v007 linear Step 1 resize](runs/v007_fast_step1_resize/distil-fp32-fine-grained-20261030T124513Z.json) | 59.55 ms | Step 1 resize: 0.47 ms; previous v001 timings for these photos had 6.63 ms median | +0.01590 / +0.01691 | 3 / 5 |
| [v008 Step 1 ONNX profile](runs/v008_step1_onnx_profile/distil-fp32-fine-grained-20261030T124524Z.json) | 58.76 ms · profiled | ONNX Runtime exposed one MIGraphX provider node over six calls; no inner layer breakdown | 0 / 0 | 0 / 5 |
| [v009 skip final JPEG](runs/v009_remove_final_jpeg_roundtrip/distil-fp32-fine-grained-20261030T124535Z.json) | 62.82 ms | Removes a JPEG encode/decode pair that previously took about 0.6 ms median | +0.00050 / +0.00218 | 4 / 5 |
| [v010 fused tensor prep](runs/v010_fused_step1_tensor_prep/distil-fp32-fine-grained-20261030T124545Z.json) | 56.85 ms | Fused scale/normalize/NCHW write: 0.24 ms vs. about 1.51 ms for the previous three operations on these photos | 0 / 0 | 0 / 5 |

The v007 resize appears much faster but already shows a quality warning on this
small sample. v009 saves only a small amount of work and also trends worse in
sample quality. v010 is the most promising smoke result: tensor preparation fell
by about 1.26 ms with no prediction difference visible at the reported
precision. It still needs a full 47-photo validation run with a matched regressor
before selection. v008 profiling latency is not comparable because tracing adds
substantial overhead; its trace is retained under the ignored local `runs/.../profiles/`
folder, and its JSON report contains the operator summary.
