# Experiment A: Constant-Prediction Baselines

JSON is the machine-readable source of truth. Newest runs appear first.

## Version 1 — Run `20260928T130207631533Z`

Status: **complete**
Comparison key: `7be95c2b97479a36d262c85b09ce776bdc263052e8604f0c8f6daab480d32c52`
Split snapshot: `20260928T120720292607Z` (76 train, 17 validation, 16 test)

All reported MAEs use the same validation images.

| Predictor | Constant/value | Validation MAE | Relative MAE reduction |
|---|---:|---:|---:|
| Training mean | 0.39407895 | 0.21869195 | 35.43% |
| Training median | 0.37500000 | 0.22205882 | 36.41% |
| Neural network | n/a | 0.14121089 | n/a |

Neural-network improvement percentages: training_mean 35.43%; training_median 36.41%

Matching checkpoint: `/step-5-checkpoints/v13-mae-0.1412-efficientnet_b0/wave-regression-efficientnet_b0-v13-best-val-mae-epoch=59-val_mae=0.1412.ckpt`
Checkpoint manifest: `/step-5-checkpoints/v13-mae-0.1412-efficientnet_b0/training_manifest_20260928T120902093327Z.json`
Checkpoint SHA-256: `5065015f56a3ca05c12b78bce43bed7a339fc5cb864f0be1c1d25c91d8b3fc33`

Test evaluated: **no**
Device: `cpu`; inference: `0.78545018` s; elapsed: `1.29213521` s
