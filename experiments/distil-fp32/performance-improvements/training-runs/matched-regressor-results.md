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

After selecting v001 using validation, v000 and v001 were evaluated on the
reserved 46-photo test split. v001 remained much faster, but its aggregate
quality was worse than v000:

| Version | MAE | RMSE | Median raw-to-score | p95 raw-to-score |
| --- | ---: | ---: | ---: | ---: |
| v000 retrained baseline | 0.114616 | 0.141656 | 98.39 ms | 106.76 ms |
| v001 matched regressor | 0.125327 | 0.167611 | 59.81 ms | 67.52 ms |

v001's test MAE increased by **0.010711** and RMSE by **0.025955**; 24 of 46
test photos had higher absolute error. Therefore v001 is not selected despite
its validation result. The reserved test set has now served its final-check
purpose and must not be used to tune or rank v002–v004. No candidate currently
demonstrates a confirmed quality-preserving latency improvement.

The HTML dashboard has per-photo and fine-grained latency results. Trained
metrics, checkpoints, and ONNX exports are grouped under
`training-runs/<experiment-id>/<run-id>/`; generated checkpoints and model
binaries remain ignored by git. Benchmark JSON files are under
`runs/<experiment-id>/`.
