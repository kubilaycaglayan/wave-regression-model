# Experiment D configuration audit

Canonical configuration: the Step 5/v13 EfficientNet-B0 training pipeline,
because that is the reference with the lower validation MAE and it records the
training behavior used by the active model. Both Experiment D arms use the
same settings listed in the experiment manifest; the only differences are
backbone trainability and the learning rate assigned to trainable backbone
weights.

## Evidence from v13

- The saved checkpoint identifies `backbone_name=efficientnet_b0`,
  `pretrained=True`, and `learning_rate=0.001`.
- The checkpoint is at epoch 59 (60 epochs completed), and the stored
  EarlyStopping callback monitors `val_mae`, uses mode `min`, and has patience
  10. The checkpoint callback also selects by `val_mae` in mode `min`.
- The current Step 5 source specifies AdamW over trainable parameters,
  SmoothL1Loss, batch size 8, maximum 100 epochs, deterministic Lightning,
  horizontal flip plus ColorJitter, ImageNet normalization, and the sigmoid
  linear regression head. The saved checkpoint has no scheduler state and
  Step 5's optimizer configuration defines no scheduler.
- The v13 training manifest records seed 42, EfficientNet-B0, ImageNet weights,
  the snapshot name, 93 labeled train/validation samples, best MAE 0.1412109,
  best RMSE 0.1866369, and 70 epochs completed. Its best checkpoint is epoch
  59 (one-indexed epoch 60).
- Step 5 source and Experiment C's manifest agree on the material settings:
  model/head, frozen backbone, SmoothL1Loss, AdamW at 0.001, batch size 8,
  100 maximum epochs, validation-MAE early stopping at patience 10,
  deterministic Lightning, and the same augmentation and normalization.
  Step 5 uses the root split CSV paths while C uses the snapshot paths; on the
  current worktree those train and validation CSV hashes are identical to the
  selected snapshot. The v13 manifest does not record a full config or CSV
  hashes, so its historical data identity cannot be proven from that manifest
  alone.

## Differences from Experiment C

Experiment C's recorded EfficientNet-B0 configuration uses the same pretrained
weights, frozen backbone, head, SmoothL1Loss, AdamW, 0.001 learning rate, batch
size 8, 100-epoch limit, validation-MAE early stopping with patience 10, image
size, normalization, and augmentations. The C runner also uses deterministic
Lightning. Neither C nor v13 configures a learning-rate scheduler.

Experiment D's frozen controls reproduced C's EfficientNet outcomes to the
recorded precision: seed 42 `0.14859`, seed 43 `0.15250`, seed 44 `0.15386`.
This verifies the C control behavior in the D runner. The v13 MAE `0.14121` is
also a seed-42 result, but it is better than the C/D frozen seed-42 result by
`0.00738`. The stored v13 training manifest reports 70 epochs, versus 51
epochs for the C/D seed-42 control. Checkpoint selection is by validation MAE
in every case. The inspected configurations do not show an intended training
setting difference that explains the gap; run-to-run training history and
unrecorded historical environment/data details remain possible factors, not
proven causes.

No test samples are used by Experiment D. The immutable snapshot is recorded
with all three CSV hashes so the test split is fingerprinted but never read by
the experiment runner.
