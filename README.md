# Wave regression preprocessing

## Setup

```bash
pip install -r requirements.txt
```

## Create water-mask previews

```bash
python prepare_water_masks.py
```

The numbered pipeline folders are `step-0/` (raw input), `step-1/` (water
preprocessing), and `step-2/` (black-area reduction). Preview files and gallery
are written to `water-mask-preview/`.

## Create standardized model inputs

```bash
python preprocess_water_inputs.py
```

This runs water segmentation, removes non-water pixels, and creates RGB `224x224` inputs in `step-1/`.

To force CPU processing:

```bash
python preprocess_water_inputs.py --device cpu
```

Existing complete outputs are skipped, so the command can safely be rerun after interruption. Open `step-1/index.html` directly in a browser to inspect the source and standardized images side by side.

## Reduce black water area

```bash
python reduce_black_water_area.py
```

This reads `step-1/`, preserves the lower-wave region, and writes aspect-preserving
224x224 results to `step-2/`. Open `step-2/index.html` to compare the last-step
image with the new result.
