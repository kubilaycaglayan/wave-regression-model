# Experiment B — Per-Image Validation Error Analysis

This is an evaluation-only analysis. No training was performed and the test dataset was not evaluated.

- Model version: `v13`
- Checkpoint: `/home/ubuntu/dev/deep-learning/wave-regression-model/step-5-checkpoints/v13-mae-0.1412-efficientnet_b0/wave-regression-efficientnet_b0-v13-best-val-mae-epoch=59-val_mae=0.1412.ckpt`
- Split snapshot: `20260928T120720292607Z`
- Validation images: **17**

## Aggregate findings

- Neural-network MAE: **0.14121089**; RMSE: **0.18663689**.
- Mean signed error: **-0.00357744**; positive values indicate overestimation.
- Median absolute error: **0.10733258**; maximum absolute error: **0.45818841**.
- The neural network beats the mean baseline on **12 / 17 (70.59%)** images.
- For actual labels 0.00–0.39 (n=7), mean signed error is **0.11549950**; for 0.60–1.00 (n=6), it is **-0.18474382**.
- Predictions span 0.1777–0.8272, while actual labels span 0.0000–0.9000; prediction/actual range width ratio: **0.722**.

## Waviness ranges

| Actual range | Count | MAE |
|---|---:|---:|
| 0.00-0.19 | 2 | 0.16802851 |
| 0.20-0.39 | 5 | 0.10130636 |
| 0.40-0.59 | 4 | 0.11238333 |
| 0.60-0.79 | 4 | 0.12488631 |
| 0.80-1.00 | 2 | 0.30445884 |

## Error ranking

Largest errors:
- `step-2_IMG_7274.jpg`: actual 0.90, predicted 0.4418, absolute error 0.4582.
- `step-2_IMG_7418.jpg`: actual 0.50, predicted 0.8272, absolute error 0.3272.
- `step-2_IMG_7395.jpg`: actual 0.20, predicted 0.5145, absolute error 0.3145.

Smallest errors:
- `step-2_IMG_7565.jpg`: actual 0.30, predicted 0.3136, absolute error 0.0136.
- `step-2_IMG_7371.jpg`: actual 0.30, predicted 0.2830, absolute error 0.0170.
- `step-2_IMG_7411.jpg`: actual 0.40, predicted 0.4172, absolute error 0.0172.

## Evidence and hypotheses

The signed-error distribution and actual-vs-predicted plot are evidence for assessing systematic underestimation or overestimation. Compare the endpoint labels and residuals in the plots before making claims about regression toward the mean.

The range-specific conclusions are limited by the sample counts above; sparse bins should not support strong conclusions.

The HTML gallery is the source for manual inspection of lighting, reflections, foam, wave patterns, and framing. This report does not automatically assign those visual conditions as causes of error.

## Next experiment recommendation

Use the largest-error gallery cases and any repeated, manually confirmed visual condition to define the next controlled preprocessing or data-collection experiment. Keep the split fixed and change one factor at a time.

## Reproducibility

- Checkpoint SHA-256: `5065015f56a3ca05c12b78bce43bed7a339fc5cb864f0be1c1d25c91d8b3fc33`
- Validation manifest SHA-256: `2d27e16c7fa1097def134488949c5ce00b109def08fc015ea68e3e2c1c66db78`
- Test manifest SHA-256 recorded without evaluation: `5f575acd82911efc9294de27d052c278119a31a9285d7b9d5552d8604f68f744`
