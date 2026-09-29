# Experiment D results

Version: `v4-20260928T135609Z`

| Arm | Mean validation MAE | MAE SD | Mean RMSE | Mean train/validation gap | Mean range ratio | Low bias | High bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| frozen | 0.15165 | 0.00274 | 0.19573 | 0.11044 | 0.6660 | 0.1367 | -0.1924 |
| partial | 0.15620 | 0.00528 | 0.20052 | 0.12067 | 0.5924 | 0.1644 | -0.1722 |

## Paired seed differences

Partial minus frozen MAE; negative favors partial fine-tuning.

- Seed 42: +0.00399
- Seed 43: +0.00124
- Seed 44: +0.00840

Validation images improved (seed-mean absolute error): 7; worsened: 10.

### Largest improvements

| Image | Improvement |
|---|---:|
| step-2_IMG_7274.jpg | +0.0503 |
| step-2_IMG_7541.jpg | +0.0452 |
| step-2_IMG_7148.jpg | +0.0413 |
| step-2_IMG_7418.jpg | +0.0311 |
| step-2_IMG_7305.jpg | +0.0224 |

### Largest regressions

| Image | Improvement |
|---|---:|
| step-2_IMG_7569.jpg | -0.0625 |
| step-2_IMG_7565.jpg | -0.0592 |
| step-2_IMG_7306.jpg | -0.0467 |
| step-2_IMG_7395.jpg | -0.0428 |
| step-2_IMG_7371.jpg | -0.0160 |

## Per-image comparison

Each error is averaged across the three seeds in that arm.

| Image | Actual | Frozen mean absolute error | Partial mean absolute error | Improvement |
|---|---:|---:|---:|---:|
| step-2_IMG_7148.jpg | 0.700 | 0.2296 | 0.1883 | +0.0413 |
| step-2_IMG_7149.jpg | 0.750 | 0.0418 | 0.0391 | +0.0027 |
| step-2_IMG_7274.jpg | 0.900 | 0.4406 | 0.3903 | +0.0503 |
| step-2_IMG_7305.jpg | 0.300 | 0.0322 | 0.0097 | +0.0224 |
| step-2_IMG_7306.jpg | 0.050 | 0.1726 | 0.2193 | -0.0467 |
| step-2_IMG_7311.jpg | 0.750 | 0.2413 | 0.2553 | -0.0139 |
| step-2_IMG_7335.jpg | 0.200 | 0.1202 | 0.1356 | -0.0154 |
| step-2_IMG_7371.jpg | 0.300 | 0.0121 | 0.0281 | -0.0160 |
| step-2_IMG_7395.jpg | 0.200 | 0.3051 | 0.3479 | -0.0428 |
| step-2_IMG_7411.jpg | 0.400 | 0.0700 | 0.0777 | -0.0077 |
| step-2_IMG_7415.jpg | 0.400 | 0.0212 | 0.0099 | +0.0113 |
| step-2_IMG_7418.jpg | 0.500 | 0.3221 | 0.2910 | +0.0311 |
| step-2_IMG_7531.jpg | 0.400 | 0.0488 | 0.0622 | -0.0134 |
| step-2_IMG_7541.jpg | 0.850 | 0.1226 | 0.0774 | +0.0452 |
| step-2_IMG_7551.jpg | 0.600 | 0.0787 | 0.0826 | -0.0039 |
| step-2_IMG_7565.jpg | 0.300 | 0.0735 | 0.1327 | -0.0592 |
| step-2_IMG_7569.jpg | 0.000 | 0.2457 | 0.3082 | -0.0625 |

## Findings

- Partial fine-tuning did not lower mean validation MAE: `0.15620` vs `0.15165` for the frozen controls (difference `+0.00454`). It was worse in all three paired seeds.
- It reduced mean rough-water underprediction slightly (`-0.1722` vs `-0.1924`) but increased calm-water overprediction (`+0.1644` vs `+0.1367`).
- Prediction-range compression increased: mean range ratio fell from `0.6660` to `0.5924`.
- The mean train/validation MAE gap increased from `0.1104` to `0.1207`; seed-mean absolute error improved on 7 of 17 validation images and worsened on 10.
- For this snapshot and configuration, the evidence does not support partial fine-tuning: it has higher average validation MAE, higher seed variation, more range compression, and a larger train/validation gap. The small validation set limits generalization claims. The test split was not evaluated.
