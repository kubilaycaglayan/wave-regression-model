# Backbone Switching Plan

## Goal

Make the current training pipeline easy to switch between the existing ResNet18 model and EfficientNet-B0, while keeping the design open for additional backbones later.

EfficientNet-B0 should be the default for new training runs, but ResNet18 must remain selectable.

## Findings

The current Step 5 model is tied directly to ResNet18 in `step_5_a_wave_regression_model.py`:

- It imports `resnet18` and `ResNet18_Weights` directly.
- It constructs `resnet18(...)` directly.
- It accesses `self.backbone.fc`.
- It replaces the ResNet classifier through `.fc`.
- Its frozen-backbone checks refer to the ResNet-specific `.fc` head.

`step_5_b_train.py` also contains hardcoded ResNet18 descriptions and direct `.backbone.fc` access.

The evaluation and prediction scripts load `WaveRegressionModel` from checkpoints rather than constructing ResNet18 themselves. They should therefore require only small metadata/checkpoint-selection updates after the model class is generalized.

Step 10 already contains the reusable EfficientNet-B0 implementation in `step_10_a_experiment_c.py`:

- `efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)`
- feature dimension from `model.classifier[-1].in_features`
- classifier replaced with `nn.Identity()`
- shared trainable head: `Linear(feature_dim, 1) -> Sigmoid`
- frozen feature extractor
- BatchNorm modules kept in evaluation mode

The Step 10 comparison found the following mean validation MAE values:

| Backbone | Mean validation MAE |
|---|---:|
| ResNet18 | 0.178722 |
| ResNet34 | 0.173924 |
| EfficientNet-B0 | 0.151072 |

This supports using EfficientNet-B0 as the current default, while retaining the ability to compare against ResNet18.

## Recommended design

Generalize the Step 5 model using a centralized backbone factory.

Define a canonical configuration such as:

```python
DEFAULT_BACKBONE = "efficientnet_b0"
SUPPORTED_BACKBONES = ("resnet18", "efficientnet_b0")
```

The model should accept the selected backbone explicitly:

```python
WaveRegressionModel(
    backbone_name="efficientnet_b0",
    learning_rate=LEARNING_RATE,
)
```

The factory should return:

- the feature extractor
- the feature dimension
- the canonical backbone name
- the pretrained-weight description

The regression head should be independent of the selected backbone:

```python
self.regression_head = nn.Sequential(
    nn.Linear(feature_dim, 1),
    nn.Sigmoid(),
)
```

This will make future additions such as ResNet34, MobileNet, or other EfficientNet variants localized to the factory and supported-backbone registry.

## Planned changes

### 1. Generalize the model

Update `step_5_a_wave_regression_model.py` to:

- add canonical backbone names and aliases
- add a `build_feature_extractor()` factory
- support ResNet18 and EfficientNet-B0
- replace `.backbone.fc` assumptions with generic `feature_extractor` and `regression_head` attributes
- derive feature dimensions dynamically
- retain frozen-parameter and BatchNorm checks
- save `backbone_name` and pretrained-weight metadata in Lightning hyperparameters
- keep the output constrained to `[0, 1]`

### 2. Make EfficientNet-B0 the default for new training

Update `step_5_b_train.py` to:

- define or import the default backbone configuration
- pass the selected backbone explicitly to `WaveRegressionModel`
- support a command such as:

```bash
python step_5_b_train.py --backbone efficientnet_b0
python step_5_b_train.py --backbone resnet18
```

- derive checkpoint names, summaries, and manifests from the selected backbone
- remove hardcoded `ResNet18` descriptions
- remove direct `.backbone.fc` access
- record the selected backbone in every new run
- include the selected backbone at the end of the version directory name under `step-5-checkpoints/`

The current version directories use a pattern such as:

```text
step-5-checkpoints/v11-mae-0.1698/
```

New and migrated version directories should make the architecture visible as a final suffix, for example:

```text
step-5-checkpoints/v12-mae-0.1511-efficientnet_b0/
step-5-checkpoints/v13-mae-0.1787-resnet18/
```

The directory name should be generated from the canonical backbone name, not from display text or a duplicated hardcoded label. The versioning helper should therefore accept the backbone as an explicit input, validate it, and construct the name consistently.

Existing Step 5 version directories must also be renamed retroactively to the new format. The migration should:

- inspect each existing version directory under `step-5-checkpoints/`
- identify the backbone from checkpoint metadata/state structure or an unambiguous training manifest
- rename the directory while preserving all files and the existing version number/MAE
- update path references inside manifests and summaries when they contain the old directory path
- detect destination collisions before making changes
- fail clearly rather than guessing when a directory's backbone cannot be identified
- record or print a migration summary showing every old-to-new directory mapping

For the current ResNet18 runs, the expected retroactive names will follow the pattern:

```text
step-5-checkpoints/v1-mae-0.1068-resnet18/
step-5-checkpoints/v10-mae-0.1758-resnet18/
step-5-checkpoints/v11-mae-0.1698-resnet18/
```

The migration should be implemented as a separate, recoverable step or utility and tested before the normal training code starts generating the new names.

### 3. Preserve existing ResNet18 checkpoints

Existing ResNet18 checkpoints must not be treated as EfficientNet-B0 checkpoints.

The safest compatibility approach is:

1. Keep `WaveRegressionModel` able to load old ResNet18 checkpoints.
2. Have the training entry point explicitly select EfficientNet-B0 as the new default.
3. Save `backbone_name` in all new checkpoints.
4. Load new checkpoints using their stored architecture metadata.
5. Never reuse a ResNet18 checkpoint for EfficientNet-B0 training or inference.

Because old checkpoints may not contain a backbone-name hyperparameter, compatibility handling should be tested explicitly rather than relying on the new default constructor value.

### 4. Keep evaluation and prediction architecture-neutral

Review and update, where necessary:

- `step_6_a_evaluate_test.py`
- `step_7_a_predict.py`
- `step_9_a_per_image_error_analysis.py`

These scripts should continue loading the architecture from the checkpoint. Their output metadata should also record:

- backbone name
- pretrained weights
- checkpoint path
- checkpoint SHA-256
- model version
- split snapshot

The existing checkpoint fingerprinting and skip behavior should remain intact. A new backbone produces a new checkpoint hash, so it will naturally be treated as a new prediction model.

### 5. Add regression tests

Add tests for both supported backbones covering:

- valid feature dimensions
- one-neuron regression head
- exactly zero trainable backbone parameters
- trainable regression head
- frozen BatchNorm modules
- correct output shape
- predictions in `[0, 1]`
- invalid backbone names failing clearly
- old ResNet18 checkpoint loading
- new EfficientNet-B0 checkpoint metadata

### 6. Improve Step 10 future extensibility

Step 10 currently has an `else` branch that constructs EfficientNet-B0 for any non-ResNet18/non-ResNet34 value. That should be changed to explicit branches or a registry lookup.

Otherwise, adding a future backbone could silently construct the wrong model.

## Expected user experience

After the update, selecting a backbone should be a command-line/configuration change:

```bash
python step_5_b_train.py --backbone efficientnet_b0
python step_5_b_train.py --backbone resnet18
```

Adding another supported backbone should require updating the centralized factory and registry, plus adding tests, rather than editing training, evaluation, and inference logic in multiple places.

## Acceptance criteria

- EfficientNet-B0 is the default for new Step 5 training runs.
- ResNet18 remains selectable.
- New Step 5 checkpoint version directories end with the canonical backbone name.
- Existing Step 5 checkpoint version directories are migrated to the new naming format.
- The migration preserves checkpoints, manifests, version numbers, and MAE values.
- Ambiguous backbone identification stops the migration with a clear error.
- Existing ResNet18 checkpoints still load correctly.
- New checkpoints record their backbone explicitly.
- Training summaries and manifests do not contain misleading hardcoded ResNet18 information.
- Both backbones use the same dataset, preprocessing, loss, optimizer, frozen-backbone policy, and output range.
- Evaluation and prediction work with checkpoints from either backbone.
- Completed image predictions continue to be skipped when the same checkpoint fingerprint is already present.
- Tests cover both architectures and checkpoint compatibility.
- Future backbone support can be added through the centralized factory/registry.

## Files expected to change

- `step_5_a_wave_regression_model.py`
- `step_5_b_train.py`
- `step_6_a_evaluate_test.py` — only if metadata/checkpoint handling needs adjustment
- `step_7_a_predict.py` — only if metadata/checkpoint handling needs adjustment
- `step_9_a_per_image_error_analysis.py` — only if metadata/checkpoint handling needs adjustment
- `step_10_a_experiment_c.py` — explicit factory/registry cleanup
- a separate Step 5 checkpoint-version migration utility or explicitly scoped migration step
- new or updated model/backbone tests under `tests/`
