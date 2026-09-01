# Wave regression preprocessing

## Setup

```bash
pip install -r requirements.txt
```

## Create water-mask previews

```bash
python prepare_water_masks.py
```

Preview files and gallery are written to `water-mask-preview/`.

## Create standardized model inputs

```bash
python preprocess_water_inputs.py
```

This runs water segmentation, removes non-water pixels, and creates RGB `224x224` inputs in `processed-data/`.

To force CPU processing:

```bash
python preprocess_water_inputs.py --device cpu
```

Existing complete outputs are skipped, so the command can safely be rerun after interruption. Open `processed-data/index.html` directly in a browser to inspect the original and standardized images side by side.
