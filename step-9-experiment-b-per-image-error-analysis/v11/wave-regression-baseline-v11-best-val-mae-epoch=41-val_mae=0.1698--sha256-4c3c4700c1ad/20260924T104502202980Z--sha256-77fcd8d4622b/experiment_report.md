# Experiment B — Per-Image Validation Error Analysis

This is an evaluation-only analysis. No training was performed and the test dataset was not evaluated.

- Model version: `v11`
- Checkpoint: `/home/ubuntu/dev/deep-learning/wave-regression-model/step-5-checkpoints/v11-mae-0.1698/wave-regression-baseline-v11-best-val-mae-epoch=41-val_mae=0.1698.ckpt`
- Split snapshot: `20260924T104502202980Z`
- Validation images: **15**

## Aggregate findings

- Neural-network MAE: **0.16982937**; RMSE: **0.20642256**.
- Mean signed error: **0.01609792**; positive values indicate overestimation.
- Median absolute error: **0.20095748**; maximum absolute error: **0.42028728**.
- The neural network beats the mean baseline on **10 / 15 (66.67%)** images.
- For actual labels 0.00–0.39 (n=5), mean signed error is **0.17919181**; for 0.60–1.00 (n=6), it is **-0.17899770**.
- Predictions span 0.2945–0.8270, while actual labels span 0.0500–0.9000; prediction/actual range width ratio: **0.626**.

## Waviness ranges

| Actual range | Count | MAE |
|---|---:|---:|
| 0.00-0.19 | 1 | 0.24454502 |
| 0.20-0.39 | 4 | 0.16285350 |
| 0.40-0.59 | 4 | 0.14437383 |
| 0.60-0.79 | 4 | 0.15768283 |
| 0.80-1.00 | 2 | 0.22162744 |

## Error ranking

Largest errors:
- `step-2_IMG_7274.jpg`: actual 0.90, predicted 0.4797, absolute error 0.4203.
- `step-2_IMG_7418.jpg`: actual 0.50, predicted 0.7957, absolute error 0.2957.
- `step-2_IMG_7395.jpg`: actual 0.20, predicted 0.4884, absolute error 0.2884.

Smallest errors:
- `step-2_IMG_7531.jpg`: actual 0.40, predicted 0.3995, absolute error 0.0005.
- `step-2_IMG_7305.jpg`: actual 0.30, predicted 0.3053, absolute error 0.0053.
- `step-2_IMG_7541.jpg`: actual 0.85, predicted 0.8270, absolute error 0.0230.

## Evidence and hypotheses

The signed-error distribution and actual-vs-predicted plot are evidence for assessing systematic underestimation or overestimation. Compare the endpoint labels and residuals in the plots before making claims about regression toward the mean.

The range-specific conclusions are limited by the sample counts above; sparse bins should not support strong conclusions.

The HTML gallery is the source for manual inspection of lighting, reflections, foam, wave patterns, and framing. This report does not automatically assign those visual conditions as causes of error.

## Next experiment recommendation

Use the largest-error gallery cases and any repeated, manually confirmed visual condition to define the next controlled preprocessing or data-collection experiment. Keep the split fixed and change one factor at a time.

## Reproducibility

- Checkpoint SHA-256: `4c3c4700c1ad3adfead52ba6d6cfde95b2a3abe62cc91607d1e2cfe72b340bbd`
- Validation manifest SHA-256: `563d95d6c11b9193a485e851dc8276f5702d6b96193a929ca20474647b1aa0ef`
- Test manifest SHA-256 recorded without evaluation: `6c380239a771d6c92ecafd5faaca29b3cfd92f653b3335b24a4e073b8a296d1d`
