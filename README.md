# Wave regression preprocessing

See [README_RUN.md](README_RUN.md) for the full step-by-step run and retraining guide.

## Setup

```bash
pip install -r requirements.txt
```

## Create water-mask previews

```bash
python step_1_a_prepare_water_masks.py
```

Step 1 always includes `step-0-raw-data/`. Add more raw-image directories in
`.env` by repeating the singular `DATA_PATH` entry:

```dotenv
DATA_PATH=/path/to/camera/archive
DATA_PATH=/path/to/another/archive
```

Directories are searched recursively, including nested folders. Each
`DATA_PATH` value may also contain multiple paths separated by the platform
path separator.

The pipeline folders are `step-0-raw-data/`, `step-1-processed-data/`, and
`step-2-final-water-data/`. Step outputs use namespaced filenames such as
`step-1_<original-name>` and `step-2_<original-name>`. Preview files and gallery
are written to `step-1-water-mask-preview/`.

To rerun segmentation with changed settings while retaining the existing
result for comparison, use:

```bash
python step_1_a_prepare_water_masks.py --skip-cache --keep-old
```

The gallery then shows the original image plus the previous and current
processor overlays and masks side by side. Previous files use an `_old`
suffix, and an existing comparison baseline is not overwritten on later runs.

## Create standardized model inputs

```bash
python step_1_b_preprocess_water_inputs.py
```

This runs water segmentation, removes non-water pixels, and creates RGB `224x224` inputs in `step-1-processed-data/`.

To force CPU processing:

```bash
python step_1_b_preprocess_water_inputs.py --device cpu
```

Existing complete outputs are skipped, so the command can safely be rerun after interruption. Open `step-1-processed-data/index.html` directly in a browser to inspect the source and standardized images side by side.

## Reduce black water area

```bash
python step_2_a_reduce_black_water_area.py
```

This reads `step-1-processed-data/`, preserves the lower-wave region, and writes aspect-preserving
224x224 results to `step-2-final-water-data/`. Open `step-2-final-water-data/index.html` to compare the last-step
image with the new result.

## Predict waviness

Put HEIC, HEIF, JPEG, or PNG photos in `step-7-predict-captures-holder/`, then run `python step_7_a_predict.py`; photos are predicted one by one and previews are saved in `step-7-inference-preview/`. Each photo also gets an append-only prediction history at `step-7-predict-captures-holder/predictions/<photo-stem>.txt`, including the prediction and checkpoint fingerprint.

Configure the validation-selected checkpoint in `.env`:

```dotenv
# Use the exact best-checkpoint path printed by train.py.
PREDICT_CHECKPOINT_PATH=step-5-checkpoints/vN-mae-<best-mae>-<backbone>/<best-checkpoint>.ckpt
```

Test evaluation automatically selects the newest training manifest's best checkpoint. To pin a
specific checkpoint for a reproducible historical evaluation, configure it separately:

```dotenv
# Optional: use the exact checkpoint path to evaluate a specific run.
EVALUATION_CHECKPOINT_PATH=step-5-checkpoints/vN-mae-<best-mae>-<backbone>/<checkpoint>.ckpt
```

Each evaluation is saved under `step-6-test-evaluation/` using the checkpoint filename and a
checkpoint SHA-256 prefix. The directory contains predictions, a summary, an inspection gallery,
and metadata including the exact checkpoint and test-manifest fingerprints. Existing complete
evaluations are skipped; a different checkpoint gets a separate result directory.
Older root-level files in this directory may be legacy evaluation outputs.

Step 7 treats `step-7-inference-preview/<photo-stem>-model-input.jpg` as the checkpoint-independent, canonical preprocessed model input. Existing previews must be detached RGB `224x224` JPEG inputs; malformed previews fail clearly and are not replaced automatically. A valid preview is reused when selecting a different checkpoint, so only regression inference runs again. A photo is skipped only when its valid preview and a matching prediction record for the selected checkpoint both exist. If the preview is missing, Steps 1 and 2 run again even when a prediction record exists; delete a preview to force regeneration.

## Discard unrelated or disrupted images

Run the labeling app with `python step_2_b_label_data.py`. The labeling screen
has a **Discard** button (and the `D` keyboard shortcut). A discard is recorded
in `discarded_images.csv` with an optional reason and timestamp. Discarded
images are removed from the labeling queue and excluded from dataset splits,
training, validation, and test evaluation. If an image had a label already,
that label is removed when the image is discarded.
