# Experiment C — Frozen Pretrained Backbone Comparison

## Summary

Create a dedicated Experiment C pipeline using the immutable split snapshot `20260923T133043052012Z`:

- Train: 68
- Validation: 14
- Test: 14, never evaluated
- Backbones: ResNet18, ResNet34, EfficientNet-B0
- Seeds: 42, 43, 44
- Frozen ImageNet-pretrained backbone
- Trainable head: `Linear(feature_dim, 1) -> Sigmoid`
- Loss: SmoothL1Loss
- Selection: lowest validation MAE
- Primary comparison: mean validation MAE across seeds

v11 will not be used as the numeric baseline because it was trained on snapshot `20260924T104502202980Z`. The experiment will retrain ResNet18 because the existing v10 checkpoint used a different effective dataset state and is not a fair exact-snapshot comparison.

## Implementation Changes

- Add a dedicated Experiment C training/evaluation entrypoint following the repository naming convention, with configurable:
  - backbone
  - seed
  - explicit split snapshot
  - experiment output directory
- Generalize the frozen model implementation to support ResNet18, ResNet34, and EfficientNet-B0 while preserving:
  - ImageNet pretrained weights
  - frozen backbone parameters
  - BatchNorm evaluation mode
  - one-layer regression head
  - sigmoid output constrained to `[0, 1]`
  - AdamW, learning rate `1e-3`
  - batch size `8`
  - maximum `100` epochs
  - validation-MAE early stopping with patience `10`
  - deterministic Lightning training
- Use the existing 224×224 preprocessing and ImageNet normalization for every backbone. Record this explicitly as a shared preprocessing choice; no backbone-specific preprocessing change is required.
- Load only `train.csv` and `validation.csv` from the requested snapshot. Validate the snapshot manifest, counts, and hashes, but never construct a test dataset or run test inference.
- Run all nine combinations:
  - `resnet18/{42,43,44}`
  - `resnet34/{42,43,44}`
  - `efficientnet_b0/{42,43,44}`
- Skip a completed run when its configuration fingerprint and selected checkpoint are already present; write timing information for each run and major stage.
- Save per-run:
  - configuration JSON
  - best checkpoint
  - checkpoint SHA-256
  - per-run metrics JSON
  - validation predictions
  - training duration and epoch count
  - trainable/total/frozen parameter counts
  - frozen-BatchNorm verification results
- Generate:
  - `aggregate_metrics.json`
  - `backbone_comparison.csv`
  - `backbone_comparison.md`
  - validation MAE seed/mean/variability plot
  - actual-vs-predicted plot for the lowest-MAE checkpoint of each backbone
  - residual plots
  - prediction-range comparison plot
  - complete checkpoint path/hash index

All outputs will live under a dedicated directory such as `step-10-experiment-c-frozen-backbone-comparison/`.

## Metrics and Reporting

For every run, compute:

- validation MAE and RMSE
- mean signed error as `prediction - actual`
- median absolute error
- maximum absolute error
- prediction minimum, maximum, and range width
- prediction/actual range-width ratio
- signed error for actual labels `0.00–0.39`
- signed error for actual labels `0.60–1.00`
- trainable parameter count
- epochs trained
- training time

Aggregate by backbone:

- mean and standard deviation of validation MAE
- best validation MAE
- mean RMSE
- mean prediction-range ratio
- mean low-end bias
- mean high-end bias

Include the constant training-mean baseline (`0.2193277311`) in the comparison table and report. The representative plot checkpoint will be the best validation-MAE seed for each backbone, clearly labeled so it is not confused with the aggregate result.

The final Markdown report will answer:

1. Lowest mean validation MAE
2. Difference relative to seed-to-seed variance
3. Reduction in prediction-range compression
4. Reduction in calm-water overprediction
5. Reduction in rough-water underprediction
6. Whether gains are broad or concentrated in particular validation images
7. Whether evidence supports changing the default backbone

## Tests and Acceptance Criteria

Add regression/unit coverage for:

- Correct feature dimensions and one-neuron head for all three backbones
- Exactly the regression head marked trainable
- Zero trainable backbone parameters
- BatchNorm modules remaining in evaluation mode
- Predictions always remaining within `[0, 1]`
- Exact snapshot enforcement and expected 68/14/14 counts
- Test split not being loaded or evaluated
- Correct metric calculations, including low/high-range signed errors and range ratios
- Configuration fingerprints and checkpoint hashes being recorded
- Completed-run skipping and safe resumability
- Aggregate statistics across exactly three seeds per backbone

Acceptance requires:

- Nine completed runs or explicit recorded failures
- Every run uses the same requested snapshot and preprocessing
- Every selected checkpoint reproduces its saved validation MAE within tolerance
- Checkpoint metadata confirms only the regression head was trainable
- BatchNorm checks pass for every run
- No test predictions or test metrics are produced
- All requested artifacts and final report questions are present
- Existing unrelated working-tree changes remain untouched

## Assumptions

- The existing optimizer, augmentation, scheduler, and early-stopping settings are the v11/current pipeline settings shown in the repository; v11’s incompatible split is not reused.
- ImageNet weights are obtained through torchvision’s `DEFAULT` weights for each backbone.
- ResNet18 is retrained for all three seeds because reusing v10 would confound backbone comparison with a different effective dataset snapshot.
- Generated experiment artifacts are kept isolated from the main training checkpoint directory and named without `original` or `overlay` tags.
