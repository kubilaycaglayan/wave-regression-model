# Experiment D — Partial Fine-Tuning of EfficientNet-B0

## Objective

Determine whether allowing the final part of EfficientNet-B0 to adapt to
wave-regression data improves generalization relative to an otherwise
identical fully frozen EfficientNet-B0.

The only intended experimental variable is backbone fine-tuning.

Follow the same file naming rules, script nameing rules, and folder naming rules.
The output folder should have should support the versions. It means like we can run this experiment multiple times, but each result will be saved in its own folder within the output folder.

## Dataset

Use immutable snapshot:

20260928T120720292607Z

Expected:
- train: 76
- validation: 17
- test: 16

Never evaluate or load test samples for inference.

## Preliminary configuration audit

Before training, compare:
1. the v13 training manifest/configuration
2. the latest Experiment C EfficientNet-B0 configuration

v13 validation MAE is approximately 0.14121.
Experiment C's best EfficientNet run is approximately 0.14859.

Identify and record every training/configuration difference that could
explain this discrepancy.

Choose one canonical configuration for Experiment D, preferably the current
v13 training pipeline where compatible.

Both Experiment D arms MUST use exactly that same configuration.

## Experimental arms

Run two configurations.

### Control: Frozen EfficientNet-B0

ImageNet pretrained EfficientNet-B0.

- complete backbone frozen
- BatchNorm frozen and kept in eval mode
- regression head trainable:
  Linear(1280, 1) -> Sigmoid

### Treatment: Partially fine-tuned EfficientNet-B0

Use exactly the same model and training configuration, except:

- unfreeze the final EfficientNet MBConv feature stage
- unfreeze the final feature projection/convolution
- keep all earlier EfficientNet stages frozen
- keep ALL BatchNorm modules frozen and in eval mode, including BatchNorm
  modules inside the unfrozen stage
- regression head remains trainable

For torchvision EfficientNet-B0, inspect the actual model structure rather
than relying blindly on indices. Record the exact module names and parameter
names that are unfrozen.

Conceptually the intended trainable backbone region is the final MBConv
stage plus final feature projection.

## Optimizer

Use AdamW with separate parameter groups.

Partially fine-tuned model:

- unfrozen pretrained backbone parameters: lr = 1e-4
- regression head: lr = 1e-3

Frozen control:

- regression head: lr = 1e-3

Keep all other optimizer settings identical.

## Seeds

Run BOTH arms with:

42
43
44

This produces six runs total.

Pair comparisons by seed:

frozen seed 42 vs fine-tuned seed 42
frozen seed 43 vs fine-tuned seed 43
frozen seed 44 vs fine-tuned seed 44

## Keep fixed

Use identical:
- snapshot
- preprocessing
- augmentations
- batch size
- loss (SmoothL1Loss)
- scheduler
- maximum epochs
- early stopping
- checkpoint selection by validation MAE
- deterministic settings

Do not modify the regression-head architecture.

## Metrics

For every run record:

- best validation MAE
- RMSE
- training MAE at selected epoch
- train/validation MAE gap
- median absolute error
- maximum absolute error
- mean signed error
- prediction range
- prediction/actual range ratio
- low-end signed bias (actual 0.00–0.39)
- high-end signed bias (actual 0.60–1.00)
- epochs trained
- training duration
- trainable parameter counts
- exact trainable backbone modules

The current frozen model's known weaknesses are approximately:

low-end bias: +0.1155
high-end bias: -0.1847
range ratio: 0.722

These are diagnostic targets, not acceptance thresholds.

## Aggregate analysis

For frozen and partially fine-tuned models calculate:

- mean validation MAE across three seeds
- validation MAE SD
- mean RMSE
- mean train/validation gap
- mean prediction-range ratio
- mean low-end bias
- mean high-end bias

Also calculate paired MAE differences for each seed:

fine_tuned_MAE - frozen_MAE

Negative means fine-tuning improved that seed.

## Per-image comparison

For each of the 17 validation images compare absolute errors between the
two configurations.

Report:
- how many images improve
- how many worsen
- largest improvements
- largest regressions

This is important because the validation set is small.

## Verification

For the frozen control verify:
- zero backbone parameters trainable

For the fine-tuned model verify:
- only intended final feature modules and regression head are trainable
- earlier backbone modules remain frozen
- every BatchNorm module remains frozen and in eval mode

Save these checks as experiment artifacts.

## Main questions

1. Does partial fine-tuning lower mean validation MAE?
2. Is the improvement consistent across seeds?
3. Does it reduce prediction-range compression?
4. Does it reduce calm-water overprediction?
5. Does it reduce rough-water underprediction?
6. Does it improve most validation images or only a few?
7. Does fine-tuning materially increase the train/validation gap?
8. Is the improvement large enough to justify the additional model
   flexibility and overfitting risk?

## Acceptance

Do not declare partial fine-tuning better based only on the single best run.

The primary result is mean validation MAE across the three paired seeds,
considered together with seed variability and the diagnostic metrics.

Keep the test set untouched.
