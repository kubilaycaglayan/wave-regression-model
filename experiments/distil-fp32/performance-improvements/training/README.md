# Matched regressor training

This isolated training flow creates a separate waviness regressor for each
preprocessing experiment. It does not edit or call the default training
pipeline's data outputs. The segmentation models and preprocessing adapters
remain those declared by each version under `../experiment-versions/`.

The preparation tool captures the exact 224 × 224 RGB input passed to each
version's regressor, using the fixed dataset snapshot and raw source photos.
Train and validation inputs are cached under `../training-cache/`, with cache
identity based on source hashes, split hashes, adapter code, benchmark code,
server code, and experiment manifests. Matching prepared items are skipped on
subsequent runs. Generated caches and model binaries are local artifacts and
are git-ignored.

Each regressor uses the same frozen ImageNet EfficientNet-B0 backbone,
trainable sigmoid regression head, seed, optimizer settings, augmentations,
early stopping, and split. The shared policy is in `training-policy.json`.
Validation chooses the checkpoint. The held-out test split is not prepared by
default. It was run as a final check for all five versions after training and
validation selection; see `../training-runs/matched-regressor-results.md`.
Treat those results as final checks, not a tuning set. A new held-out cohort is
needed for further model selection.

From the repository root, run input preparation in the MIGraphX runtime
environment:

```bash
LD_LIBRARY_PATH=/tmp/wave_migraphx_312_env/lib/python3.12/site-packages/migraphx_libs:/opt/rocm/lib \
MIGRAPHX_SKIP_BENCHMARKING=1 MIGRAPHX_USE_FAST_SOFTMAX=0 \
WAVE_EXPERIMENT=distilled-b0-migraphx-fp32 \
/tmp/wave_migraphx_312_env/bin/python experiments/distil-fp32/performance-improvements/training/prepare_inputs.py
```

Train one version with the repository's PyTorch/Lightning environment:

```bash
python experiments/distil-fp32/performance-improvements/training/train_regressor.py \
  --experiment-id v000_baseline
```

Export that run's `best.ckpt` to ONNX, then point the existing benchmark at the
experiment-matched export using `WAVE_REGRESSOR_ONNX=/path/to/model.onnx`:

```bash
python experiments/distil-fp32/performance-improvements/training/export_regressor.py \
  experiments/distil-fp32/performance-improvements/training-runs/v000_baseline/<run-id>/checkpoints/<best.ckpt> \
  experiments/distil-fp32/performance-improvements/training-runs/v000_baseline/<run-id>/model.onnx
```

Benchmark reports remain under `runs/<experiment-id>/`. Training metrics and
manifests are under `training-runs/<experiment-id>/<run-id>/`; checkpoint and
ONNX binaries are git-ignored because they are reproducible model artifacts.
