# Experiment B — Per-Image Validation Error Analysis

This is an evaluation-only analysis. No training was performed and the test dataset was not evaluated.

- Model version: `v10`
- Checkpoint: `/home/ubuntu/dev/deep-learning/wave-regression-model/step-5-checkpoints/v10-mae-0.1758/wave-regression-baseline-v10-best-val-mae-epoch=42-val_mae=0.1758.ckpt`
- Split snapshot: `20260923T133043052012Z`
- Validation images: **14**

## Aggregate findings

- Neural-network MAE: **0.17578625**; RMSE: **0.20751773**.
- Mean signed error: **-0.00694735**; positive values indicate overestimation.
- Median absolute error: **0.17864515**; maximum absolute error: **0.44524860**.
- The neural network beats the mean baseline on **8 / 14 (57.14%)** images.
- For actual labels 0.00–0.39 (n=5), mean signed error is **0.13825122**; for 0.60–1.00 (n=5), it is **-0.21938885**.
- Predictions span 0.2594–0.8218, while actual labels span 0.0500–0.9000; prediction/actual range width ratio: **0.662**.

## Waviness ranges

| Actual range | Count | MAE |
|---|---:|---:|
| 0.00-0.19 | 1 | 0.20943831 |
| 0.20-0.39 | 4 | 0.13674408 |
| 0.40-0.59 | 4 | 0.15191217 |
| 0.60-0.79 | 3 | 0.20782853 |
| 0.80-1.00 | 2 | 0.23672934 |

## Error ranking

Largest errors:
- `step-2_IMG_7274.jpg`: actual 0.90, predicted 0.4548, absolute error 0.4452.
- `step-2_IMG_7418.jpg`: actual 0.50, predicted 0.7750, absolute error 0.2750.
- `step-2_IMG_7395.jpg`: actual 0.20, predicted 0.4640, absolute error 0.2640.

Smallest errors:
- `step-2_IMG_7541.jpg`: actual 0.85, predicted 0.8218, absolute error 0.0282.
- `step-2_IMG_7305.jpg`: actual 0.30, predicted 0.2674, absolute error 0.0326.
- `step-2_IMG_7531.jpg`: actual 0.40, predicted 0.3552, absolute error 0.0448.

## Evidence and hypotheses

The signed-error distribution and actual-vs-predicted plot are evidence for assessing systematic underestimation or overestimation. Compare the endpoint labels and residuals in the plots before making claims about regression toward the mean.

The range-specific conclusions are limited by the sample counts above; sparse bins should not support strong conclusions.

The HTML gallery is the source for manual inspection of lighting, reflections, foam, wave patterns, and framing. This report does not automatically assign those visual conditions as causes of error.

## Next experiment recommendation

Use the largest-error gallery cases and any repeated, manually confirmed visual condition to define the next controlled preprocessing or data-collection experiment. Keep the split fixed and change one factor at a time.

## Reproducibility

- Checkpoint SHA-256: `43fffd97861058ae499b02bed253ba8e74d560164a62034dfa5c1ebff96b0ecf`
- Validation manifest SHA-256: `bcabb9419b20a82cfa253f5dee31a38b89a5fc9604a5cb43ee113932006c573a`
- Test manifest SHA-256 recorded without evaluation: `8dc916358893fbd23f2e0e81c289fddcce02fa8eb09d0c30fbfed2fc704e7322`
