# Matched-regressor comparison

Each v000–v004 preprocessing route prepared RGB inputs for the same EfficientNet-B0
frozen-backbone regressor training recipe. All five models used seed 42, the
same hyperparameters and augmentations, and the same 215-photo training cohort
plus all 47 validation photos. Two training photos failed preprocessing in
v000, v003, and v004, so `step-2_IMG_7592.jpg` and `step-2_IMG_7593.jpg` were
excluded from every model's training cohort.

## Validation comparison

| Version | MAE | RMSE | Median raw-to-score | p95 raw-to-score |
| --- | ---: | ---: | ---: | ---: |
| v000 baseline | 0.118095 | 0.160190 | 98.82 ms | 110.86 ms |
| v001 coordinate crop | **0.100441** | **0.126186** | **60.48 ms** | **76.57 ms** |
| v002 skip Step 2 | 0.128635 | 0.174412 | 81.33 ms | 85.84 ms |
| v003 ROI mask processing | 0.121055 | 0.155764 | 112.88 ms | 125.20 ms |
| v004 resolution cap | 0.113904 | 0.151264 | 91.72 ms | 103.05 ms |

v001 led validation on both aggregate quality metrics and latency. v004 also
improved both aggregate quality metrics and latency relative to retrained v000.
The strict per-photo no-regression gate flagged these versions because some
individual images had higher error.

## Final test check

After training and validation selection, all five versions were evaluated on
the reserved 46-photo test split. Their paired results against retrained v000
were:

| Version | MAE | RMSE | Median raw-to-score | p95 raw-to-score |
| --- | ---: | ---: | ---: | ---: |
| v000 retrained baseline | 0.114616 | 0.141656 | 98.39 ms | 106.76 ms |
| v001 coordinate crop | 0.125327 | 0.167611 | 59.81 ms | 67.52 ms |
| v002 skip Step 2 | 0.142534 | 0.182296 | 80.10 ms | 83.85 ms |
| v003 ROI mask processing | 0.114302 | 0.144591 | 111.66 ms | 117.97 ms |
| v004 resolution cap | 0.115810 | 0.137867 | 94.28 ms | 98.69 ms |

All test runs completed without failures. v001's test MAE increased by
**0.010711** and RMSE by **0.025955**; 24 of 46 photos had higher absolute
error. v002 regressed on both aggregate quality metrics. v003 had nearly
unchanged MAE (-0.000314), slightly worse RMSE (+0.002936), and higher latency.
v004 improved RMSE by **0.003788**, but MAE increased by **0.001194** and median
latency was only **4.11 ms** lower. None demonstrates a confirmed quality-
preserving latency improvement. These test results are final checks and must
not be used as a tuning or selection set; further selection needs a new held-out
cohort.

The HTML dashboard has per-photo and fine-grained latency results. Trained
metrics, checkpoints, and ONNX exports are grouped under
`training-runs/<experiment-id>/<run-id>/`; generated checkpoints and model
binaries remain ignored by git. Benchmark JSON files are under
`runs/<experiment-id>/`.

## Whole labeled-photo diagnostic cohort

All five preprocessing versions were run across `whole`, the union of the fixed
train, validation, and test snapshot (217 + 47 + 46 = 310 labeled photos).
Each version used its own matched trained regressor. This cohort overlaps the
training data and reserved test split, so these aggregate scores are descriptive
only and must not be used for selection or tuning. MAE/RMSE below include only
successful predictions; latency medians for preprocessing and inference also
cover successful photos, while raw-to-score latency includes all 310 rows,
including failures.

| Version | Successful / 310 | Failures | MAE | RMSE | Preprocessing median | Regressor inference median | Raw-to-score median | Raw-to-score p95 | Report |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v000 baseline | 308 | 2 | 0.096209 | 0.124412 | 94.32 ms | 5.58 ms | 99.95 ms | 108.05 ms | [JSON](../runs/v000_baseline/distil-fp32-fine-grained-20261030T120649Z.json) |
| v001 coordinate crop | 309 | 1 | 0.103334 | 0.130578 | 55.21 ms | 5.40 ms | 60.59 ms | 77.80 ms | [JSON](../runs/v001_coordinate_crop_single_output/distil-fp32-fine-grained-20261030T120746Z.json) |
| v002 skip Step 2 | 309 | 1 | 0.107073 | 0.137801 | 76.45 ms | 5.80 ms | 82.19 ms | 89.50 ms | [JSON](../runs/v002_skip_step2_keep_baseline_image/distil-fp32-fine-grained-20261030T120824Z.json) |
| v003 ROI mask processing | 308 | 2 | 0.105212 | 0.133297 | 108.87 ms | 5.58 ms | 114.44 ms | 124.27 ms | [JSON](../runs/v003_roi_mask_processing/distil-fp32-fine-grained-20261030T120919Z.json) |
| v004 resolution cap | 308 | 2 | 0.095059 | 0.120355 | 87.27 ms | 5.40 ms | 92.72 ms | 119.59 ms | [JSON](../runs/v004_working_resolution_cap/distil-fp32-fine-grained-20261030T121011Z.json) |

The baseline and v003/v004 failed on `IMG_7592.HEIC` (Step 2 could not select
a lower-water crop) and `IMG_7593.HEIC` (Step 1 water detection failed).
v001 and v002 each failed on `IMG_7593.HEIC` for their Step 1 water-region
check. The detailed JSON files retain the per-photo failure reasons and the
full warm-up, preprocessing, and inference timing breakdown. The dashboard
includes this cohort under **Whole · all labeled photos**, with an explicit
warning that it is not independent evaluation.
