# Experiment C — Frozen Pretrained Backbone Comparison

## Configuration

- Split snapshot: `20260923T133043052012Z`
- Test set evaluated: **no**
- Constant training-mean validation MAE: **0.21932773**

## Aggregate comparison

| Backbone | Mean MAE | MAE SD | Best MAE | Mean RMSE | Mean range ratio | Mean low bias | Mean high bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| resnet18 | 0.178722 | 0.012198 | 0.168284 | 0.214950 | 0.4772 | 0.150819 | -0.274043 |
| resnet34 | 0.173924 | 0.024181 | 0.157536 | 0.225721 | 0.6266 | 0.055335 | -0.331605 |
| efficientnet_b0 | 0.151072 | 0.002069 | 0.148710 | 0.195137 | 0.7218 | 0.112518 | -0.199885 |

## Answers to final report questions

1. Lowest mean validation MAE: **efficientnet_b0** at `0.151072`.
2. The gap to resnet34 is `0.022852` MAE. EfficientNet-B0's seed SD is `0.002069`, so the gap is approximately `11.0×` its own seed SD; the advantage is not explained by EfficientNet's run-to-run variance.
3. **efficientnet_b0** materially reduces range compression relative to ResNet18: mean range ratio `0.7218` versus `0.4772` (+`0.2446`), though predictions remain compressed below 1.0.
4. **efficientnet_b0** reduces calm-water overprediction relative to ResNet18: low-range signed bias `0.112518` versus `0.150819`. ResNet34 is lower still at `0.055335`.
5. **efficientnet_b0** reduces rough-water underprediction relative to ResNet18: high-range signed bias `-0.199885` versus `-0.274043`. ResNet34 is worse at `-0.331605`.
6. The improvement is not uniform over every validation image: the best efficientnet_b0 checkpoint has lower absolute error on `6/14` images versus the best ResNet18 checkpoint and `6/14` versus the best ResNet34 checkpoint. The aggregate gain is therefore helped by several large per-image improvements rather than every image improving.
7. Evidence supports changing the default backbone to **efficientnet_b0** for this frozen-head setup: it has the lowest mean MAE, the lowest seed variability, the highest prediction-range ratio, and improved high-end bias. The conclusion remains validation-only and should be confirmed with future data before treating it as final production policy.

## Selected checkpoints

- `resnet18-seed-42`: `/home/ubuntu/dev/deep-learning/wave-regression-model/step-10-experiment-c-frozen-backbone-comparison/v1-20260925T000000Z/runs/resnet18/seed-42/resnet18-seed-42-best-val-mae-epoch=23-val_mae=0.1758.ckpt` (`a226652a2dd7a7b92a0ee8f2a01c2ddf17c04549077174a2cd8eecf2b1a49daf`)
- `resnet18-seed-43`: `/home/ubuntu/dev/deep-learning/wave-regression-model/step-10-experiment-c-frozen-backbone-comparison/v1-20260925T000000Z/runs/resnet18/seed-43/resnet18-seed-43-best-val-mae-epoch=14-val_mae=0.1921.ckpt` (`292dda3e8f26cb6193c34f00f966fae6ef49f1722c01c694b462f62cf5520546`)
- `resnet18-seed-44`: `/home/ubuntu/dev/deep-learning/wave-regression-model/step-10-experiment-c-frozen-backbone-comparison/v1-20260925T000000Z/runs/resnet18/seed-44/resnet18-seed-44-best-val-mae-epoch=04-val_mae=0.1683.ckpt` (`1cd45feead1ec55d053fa8d9727a81e1b820bffb87c243dc883ef89fefb37935`)
- `resnet34-seed-42`: `/home/ubuntu/dev/deep-learning/wave-regression-model/step-10-experiment-c-frozen-backbone-comparison/v1-20260925T000000Z/runs/resnet34/seed-42/resnet34-seed-42-best-val-mae-epoch=11-val_mae=0.2017.ckpt` (`8ff2ccdbae86ac1e04799254059c1fe20fb95c2cbf4d7a0ea7834a22425b3997`)
- `resnet34-seed-43`: `/home/ubuntu/dev/deep-learning/wave-regression-model/step-10-experiment-c-frozen-backbone-comparison/v1-20260925T000000Z/runs/resnet34/seed-43/resnet34-seed-43-best-val-mae-epoch=08-val_mae=0.1625.ckpt` (`7c4ad5b4771a228f9d5a72caf1e8ba2891b9787df74aa291d56a1cd8e849167b`)
- `resnet34-seed-44`: `/home/ubuntu/dev/deep-learning/wave-regression-model/step-10-experiment-c-frozen-backbone-comparison/v1-20260925T000000Z/runs/resnet34/seed-44/resnet34-seed-44-best-val-mae-epoch=11-val_mae=0.1575.ckpt` (`cd1a178543b2cdc3e984673c17cb3f6fe92671ac530d0b27f1454fa30e30c8f3`)
- `efficientnet_b0-seed-42`: `/home/ubuntu/dev/deep-learning/wave-regression-model/step-10-experiment-c-frozen-backbone-comparison/v1-20260925T000000Z/runs/efficientnet_b0/seed-42/efficientnet_b0-seed-42-best-val-mae-epoch=53-val_mae=0.1487.ckpt` (`ac9bd55e08c4af73a9f24595bb75add98669b77377723e913f0dfdaad6f2f57f`)
- `efficientnet_b0-seed-43`: `/home/ubuntu/dev/deep-learning/wave-regression-model/step-10-experiment-c-frozen-backbone-comparison/v1-20260925T000000Z/runs/efficientnet_b0/seed-43/efficientnet_b0-seed-43-best-val-mae-epoch=37-val_mae=0.1526.ckpt` (`93568a48d8c401b1b07cfa4f1b9bad167f4eb6d2bb8a6dab21b1e8706320a1d3`)
- `efficientnet_b0-seed-44`: `/home/ubuntu/dev/deep-learning/wave-regression-model/step-10-experiment-c-frozen-backbone-comparison/v1-20260925T000000Z/runs/efficientnet_b0/seed-44/efficientnet_b0-seed-44-best-val-mae-epoch=48-val_mae=0.1519.ckpt` (`a460c7839bbeadc1d7acb65df2b74d39b033d063c4831640fd4f031850877a12`)
