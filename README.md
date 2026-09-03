# Wave regression preprocessing

## Setup

```bash
pip install -r requirements.txt
```

## Create water-mask previews

```bash
python step_1_a_prepare_water_masks.py
```

The pipeline folders are `step-0-raw-data/`, `step-1-processed-data/`, and
`step-2-final-water-data/`. Step outputs use namespaced filenames such as
`step-1_<original-name>` and `step-2_<original-name>`. Preview files and gallery
are written to `water-mask-preview/`.

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

Put HEIC, HEIF, JPEG, or PNG photos in `predict-holder/`, then run `python predict.py`; photos are predicted one by one and previews are saved in `step-7-inference-preview/`.
