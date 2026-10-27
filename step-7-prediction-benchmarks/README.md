# Step 7 prediction benchmarks and fast preprocessing experiment

This is the canonical record for the Step 7 prediction performance experiment: its commands, configurations, findings, generated report locations, and remaining evaluation work. Benchmark output files and processed preview images remain local and are ignored by Git; this README is tracked.

## Experiment question

Can Step 7 produce a waviness prediction quickly enough for incoming photos while retaining useful water masks and wave detail? Prediction selects preprocessing from the chosen checkpoint: baseline checkpoints keep the baseline path, and `fast-v1` checkpoints use the shared fast path described below.

## Current fast configuration

| Setting | Original path | Fast path measured here |
| --- | --- | --- |
| Step 1 segmentation | Full-resolution photo, 35 overlapping tiles for these 3024 × 4032 inputs | One 448 × 448 image pass, no overlapping tiles |
| Step 1 class selection | Original tiled-mask behavior | Class scores interpolated at 448 before argmax |
| Step 2 segmentation | Default processor size | 384 × 384 |
| Segmentation precision | float32 | float16 on the GPU |
| Regression model | Selected v16 EfficientNet-B0 checkpoint | Separate adapted head checkpoint; backbone frozen |
| Model input | 224 × 224 Step 2 crop | 224 × 224 Step 2 crop |

The fast settings alter masks and selected crops, so existing original-path previews must not be reused. Prediction previews and records are stored in a separate settings-and-code-specific namespace. The checkpoint contract checks preprocessing values, model revision, device type, PyTorch version, and relevant code hashes before allowing inference.

## Shared training and prediction profile

The fast settings are now exposed as the shared `fast-v1` preprocessing profile. Step 1 and Step 2 write into `step-1-processed-data/` and `step-2-final-water-data/`, beside their `archive/` directories. Step 3 records each profile in a new timestamped snapshot under `step-3-dataset-splits/snapshots/`; it retains the existing split registry and versioned snapshot structure. Step 5 saves another numbered checkpoint under `step-5-checkpoints/`, with `-fast` appended after the backbone name. Training checkpoints record the Step 2 preprocessing contract; one-shot prediction and the watcher select that profile from the checkpoint, while Step 6 evaluates against the matching Step 2 images and split snapshot. The batch Step 1 and Step 2 commands require CUDA for `fast-v1`; select `baseline-v16` for CPU preprocessing.

## Findings so far

The latest saved split snapshot is `20261015T083233550489Z`. Its validation CSV contains 32 photos. The original and fast models were compared against the same labels in that snapshot; the test split has not been read for this experiment.

### Prediction quality

| Metric on the 32 validation photos | Original checkpoint and saved Step 2 inputs | Fast preprocessing and adapted head |
| --- | ---: | ---: |
| MAE | 0.156742 | 0.157839 |
| MSE | 0.040130 | 0.042888 |
| RMSE | 0.200324 | 0.207094 |

The fast path's MAE is about 0.0011 higher on this validation split. This is exploratory: the split has been used during development, and the small unseen-date subset is not a reliable final comparison. The adapted head experiment selected its epoch and teacher weighting using date-held-out samples from the training split, then refit on all 149 training photos; it did not read the test split. The fast checkpoint did not replace the original checkpoint.

### Prediction latency

Fresh inference was timed on the same 32 raw validation photos using the GPU and four PyTorch threads. Model startup was excluded. The original path was measured once per photo. The fast values are the recorded warm pass from the latency benchmark.

| Stage | Original median / p95 | Fast median / p95 |
| --- | ---: | ---: |
| Step 1 | 10.299 s / 10.614 s | 0.225 s / 0.254 s |
| Step 2 | 0.305 s / 0.330 s | 0.059 s / 0.064 s |
| Regression | 0.006 s / 0.010 s | 0.005 s / 0.006 s |
| Total prediction | 10.716 s / 11.032 s | 0.394 s / 0.428 s |

The median time saved is about 10.32 seconds per photo. Roughly 98% of that reduction is from Step 1: the original ran 35 tiles, while the fast path runs one resized image. Step 2 also got faster at 384 px. Regression time is similar in both flows. These settings were measured together, so the timing does not isolate the effect of each individual option.

A separate warmed-watcher arrival benchmark measured from atomic move into the watched folder through a completed prediction record: median 0.375 s, p95 0.419 s across 32 photos. It excludes upload/copy time, startup, and queued requests.

### Visual comparison

The local HTML report `fast-head-date-heldout/visual-comparison-20261027T114934Z/index.html` shows original and fast Step 1 and Step 2 outputs side by side for all 32 photos, with labels, predictions, absolute errors, and per-photo runtime breakdowns. Its processed-image assets are stored beside the page. Both the page and images are generated benchmark artifacts and remain ignored by Git.

The fast crops looked visually cleaner in this comparison. Visual appearance alone does not establish prediction quality; the validation metrics above were slightly worse for the fast path.

## Running Step 7 prediction

### One-shot prediction

Put new HEIC, HEIF, JPEG, or PNG photos in `step-7-predict-captures-holder/`, set `PREDICT_CHECKPOINT_PATH`, and run:

```bash
python step_7_a_predict.py
```

### Persistent watcher

Keep the models loaded for incoming photos with:

```bash
python step_7_e_watch_predictions.py --threads 4
```

Stop with Ctrl+C. On Linux, the watcher handles files after a writer closes them or after an atomic move. Prefer staging the photo elsewhere and moving it into the folder with a unique filename. Files present at startup and systems without notifications use stability polling. Use `--watch-mode poll` to force polling or `--watch-mode event` to require notifications. Restart the watcher to select a different checkpoint.

### Opt-in fast watcher

An example 448 / 384 float16 run is:

```bash
PREDICT_CHECKPOINT_PATH=/path/to/experimental.ckpt python step_7_e_watch_predictions.py --threads 4 --experimental-single-resolution 448 --experimental-step2-resolution 384 --experimental-logits-resolution 448 --experimental-precision float16
```

Other resolutions can be explored, but each configuration changes outputs and must be evaluated with a matching adapted checkpoint. CPU runs use full precision and separate device-specific outputs. If GPU segmentation unexpectedly falls back to CPU, the watcher fails before saving an incompatible preview. Experimental outputs use a separate namespace and cannot be mistaken for baseline previews.

## Benchmarking and profiling commands

### Prediction latency

```bash
# Default prediction-holder photos; three independent processes.
python step_7_b_benchmark_prediction.py --repeat 3

# Fresh preprocessing and repeated warm predictions for selected photos.
python step_7_b_benchmark_prediction.py --device cuda --threads 4 --fresh --repeat 1 --warm-runs 5 /path/to/photo.HEIC

# Fast settings; this can change masks, crops, and predictions.
python step_7_b_benchmark_prediction.py --device cuda --threads 4 --fresh --repeat 1 --warm-runs 5 --resolution 448 --step2-resolution 384 --precision float16 --no-tiles --logits-resolution 448 /path/to/photo.HEIC
```

Reports are written to a timestamped directory under `step-7-prediction-benchmarks/`. The benchmark uses an existing preview unless `--fresh` is given. The p95 uses the nearest-rank method; inspect failures separately from successful timings.

### Arrival-to-record latency

```bash
python step_7_h_benchmark_arrival.py --checkpoint /path/to/experimental.ckpt --resolution 448 --step2-resolution 384 --logits-resolution 448 --precision float16 /path/to/photo.HEIC
```

This uses an isolated warmed watcher and a temporary symlink plus atomic move. Linux notifications are required. It measures one photo at a time and excludes upload/copy time, model startup, and queued requests.

### Preprocessing quality comparisons

```bash
# Compare Step 2 resolutions using persisted Step 1 validation images.
python step_7_d_compare_preprocessing.py --device cuda --candidate-resolution 384 --output /tmp/wave-step2-comparison.json

# Compare raw-photo fast preprocessing with persisted Step 2 validation outputs.
python step_7_d_compare_preprocessing.py --device cuda --candidate-resolution 448 --step2-resolution 384 --precision float16 --logits-resolution 448 --raw-candidate --output /tmp/wave-fast-comparison.json
```

Optional flags test projecting the Step 1 mask into Step 2 (`--reuse-step1-mask`) or other resolutions. The command compares only examples that succeeded in both variants, so also inspect failure counts. It does not save source photos or altered crops. Raw-photo comparison requires the Step 1 source manifest and original local photos.

### Adapt a regression head for fast preprocessing

```bash
python step_7_f_fast_head_experiment.py --output-dir /tmp/wave-fast-head --holdout-by-date
```

This freezes the selected checkpoint's backbone and adapts only the regression head. It uses the training split, selects epoch and teacher weighting on held-out training capture dates, then evaluates validation. It requires usable EXIF capture dates, does not read the test split, and does not change the selected checkpoint. Feature vectors are cached under `step-7-prediction-benchmarks/fast-head-feature-cache/` and reused when source, prior-stage output, checkpoint, model revision, device, and code fingerprints match. Choose a new output directory for each completed experiment.

### PyTorch operator profiling

```bash
python step_7_c_profile_prediction.py --device cpu /path/to/photo.HEIC
python step_7_c_profile_prediction.py --device cuda --threads 4 --warmup 1 --repeat 2 --shapes --memory /path/to/photo.HEIC
```

Profiler captures and Chrome/Perfetto traces are saved under `step-7-prediction-profiles/`. Profiler timings include profiling overhead and are not latency benchmark measurements.

## Remaining decision

The speed target was met, but the fast path remains experimental. The next quality decision is whether to evaluate a frozen configuration on the untouched test split. If accepted, a full fast-flow training dataset and training entry point should use the exact same preprocessing contract as inference; the existing regular training command still consumes the original Step 2 dataset. Keep the original checkpoint and preprocessing as the default until that decision is made.
