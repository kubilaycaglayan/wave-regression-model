# Experiment C — Frozen Pretrained Backbone Comparison

## Configuration

- Split snapshot: `20260928T120720292607Z`
- Test set evaluated: **no**
- Constant training-mean validation MAE: **0.21869195**

## Aggregate comparison

| Backbone | Mean MAE | MAE SD | Best MAE | Mean RMSE | Mean range ratio | Mean low bias | Mean high bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| resnet18 | 0.185150 | 0.008666 | 0.175894 | 0.218313 | 0.5532 | 0.196753 | -0.198088 |
| resnet34 | 0.188030 | 0.009768 | 0.180015 | 0.230213 | 0.7171 | 0.118808 | -0.281921 |
| efficientnet_b0 | 0.151653 | 0.002736 | 0.148594 | 0.195729 | 0.6660 | 0.136737 | -0.192434 |

## Answers to final report questions

1. Lowest mean validation MAE: **efficientnet_b0** at `0.151653`.
2. The gap to resnet18 is `0.033497` MAE. EfficientNet-B0's seed SD is `0.002736`, so the gap is approximately `12.2×` its own seed SD; the advantage is not explained by EfficientNet's run-to-run variance.
3. **efficientnet_b0** materially reduces range compression relative to ResNet18: mean range ratio `0.6660` versus `0.5532` (+`0.1128`), though predictions remain compressed below 1.0.
4. **efficientnet_b0** reduces calm-water overprediction relative to ResNet18: low-range signed bias `0.136737` versus `0.196753`. ResNet34 is lower still at `0.118808`.
5. **efficientnet_b0** reduces rough-water underprediction relative to ResNet18: high-range signed bias `-0.192434` versus `-0.198088`. ResNet34 is worse at `-0.281921`.
6. The improvement is not uniform over every validation image: the best efficientnet_b0 checkpoint has lower absolute error on `10/17` images versus the best ResNet18 checkpoint and `10/17` versus the best ResNet34 checkpoint. The aggregate gain is therefore helped by several large per-image improvements rather than every image improving.
7. Evidence supports changing the default backbone to **efficientnet_b0** for this frozen-head setup: it has the lowest mean MAE, the lowest seed variability, the highest prediction-range ratio, and improved high-end bias. The conclusion remains validation-only and should be confirmed with future data before treating it as final production policy.

## Selected checkpoints

- `resnet18-seed-42`: `/home/ubuntu/dev/deep-learning/wave-regression-model/step-10-experiment-c-frozen-backbone-comparison/v2-20260928T130354Z/runs/resnet18/seed-42/resnet18-seed-42-best-val-mae-epoch=39-val_mae=0.1865.ckpt` (`e8e861628b78e0ed4466228aba9e3554709f2ad452db166ecdb524e1e70125f1`)
- `resnet18-seed-43`: `/home/ubuntu/dev/deep-learning/wave-regression-model/step-10-experiment-c-frozen-backbone-comparison/v2-20260928T130354Z/runs/resnet18/seed-43/resnet18-seed-43-best-val-mae-epoch=37-val_mae=0.1931.ckpt` (`242a83cd7147b6e5b83be7b50145f798ac592df42daecd052aa7987219176ba4`)
- `resnet18-seed-44`: `/home/ubuntu/dev/deep-learning/wave-regression-model/step-10-experiment-c-frozen-backbone-comparison/v2-20260928T130354Z/runs/resnet18/seed-44/resnet18-seed-44-best-val-mae-epoch=35-val_mae=0.1759.ckpt` (`fe16322e2a2c842f602489a37ec4461ad5da6b055025f586e35bc061e09d88c3`)
- `resnet34-seed-42`: `/home/ubuntu/dev/deep-learning/wave-regression-model/step-10-experiment-c-frozen-backbone-comparison/v2-20260928T130354Z/runs/resnet34/seed-42/resnet34-seed-42-best-val-mae-epoch=39-val_mae=0.1989.ckpt` (`fb6fd71a71769c197bd02422475cb0ecdb509559651c6cbb2e9679470330dfec`)
- `resnet34-seed-43`: `/home/ubuntu/dev/deep-learning/wave-regression-model/step-10-experiment-c-frozen-backbone-comparison/v2-20260928T130354Z/runs/resnet34/seed-43/resnet34-seed-43-best-val-mae-epoch=09-val_mae=0.1852.ckpt` (`7e52c450092a12d354650ad9340018d99c5f85eb46383889239e0e6f51efda8f`)
- `resnet34-seed-44`: `/home/ubuntu/dev/deep-learning/wave-regression-model/step-10-experiment-c-frozen-backbone-comparison/v2-20260928T130354Z/runs/resnet34/seed-44/resnet34-seed-44-best-val-mae-epoch=13-val_mae=0.1800.ckpt` (`b458205d786f54ed05cf2861c6c8a9f16edc0bbb1279b21a45631df96ef33b50`)
- `efficientnet_b0-seed-42`: `/home/ubuntu/dev/deep-learning/wave-regression-model/step-10-experiment-c-frozen-backbone-comparison/v2-20260928T130354Z/runs/efficientnet_b0/seed-42/efficientnet_b0-seed-42-best-val-mae-epoch=40-val_mae=0.1486.ckpt` (`5b4c7e2dabdcc5e72fd481b89c7cf16da38497bd8c0ed6bbd0fd375eb93f6801`)
- `efficientnet_b0-seed-43`: `/home/ubuntu/dev/deep-learning/wave-regression-model/step-10-experiment-c-frozen-backbone-comparison/v2-20260928T130354Z/runs/efficientnet_b0/seed-43/efficientnet_b0-seed-43-best-val-mae-epoch=42-val_mae=0.1525.ckpt` (`0b7b993c74892263225884a5008ffed137416351e6bd1dee19275c06159394f6`)
- `efficientnet_b0-seed-44`: `/home/ubuntu/dev/deep-learning/wave-regression-model/step-10-experiment-c-frozen-backbone-comparison/v2-20260928T130354Z/runs/efficientnet_b0/seed-44/efficientnet_b0-seed-44-best-val-mae-epoch=41-val_mae=0.1539.ckpt` (`05f8825aed62d6c6a62de0ebeafe656ac5bc1607a33fcd65a0a0387e4d50e1bb`)
