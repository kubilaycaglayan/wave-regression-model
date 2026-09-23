# Run guide

Sea waviness regression (0 = calm, 1 = rough) from sea photos.

```bash
pip install -r requirements.txt
python step_1_b_preprocess_water_inputs.py
python step_2_a_reduce_black_water_area.py
python step_2_b_label_data.py        # labeling UI at http://localhost:8055, saves labels.csv
python step_3_a_prepare_dataset_split.py --seed 42
python step_4_c_inspect_training_data.py
python train.py                       # or: python step_5_b_train.py
python step_6_a_evaluate_test.py
```

Predict one or more new photos:

```bash
# put HEIC/JPG/PNG in predict-holder/, then:
python step_7_a_predict.py
# previews -> step-7-inference-preview/
```

Notes: steps skip existing outputs, so reruns are safe. Use `--device cpu` on the step 1/2 scripts to force CPU.

---

The repository’s retraining flow is:

`new raw photos → Step 1 preprocessing → Step 2 lower-water crop → manual labels → dataset split → training → optional evaluation/inference`

Run all commands from:

```bash
cd /path/to/wave-regression-model
```

## 1. Add the new photos

Copy new JPEG, PNG, HEIC, HEIF, TIFF, BMP, or WebP photos into:

```text
step-0-raw-data/
```

Use unique filenames containing an image number, for example:

```text
IMG_7400.jpg
IMG_7401.jpg
IMG_7402.jpg
```

The split script requires filenames containing an `IMG_<number>` pattern.

Do not replace an existing photo with the same filename unless you also remove its generated pipeline outputs. Existing output filenames cause later steps to skip that image.

## 2. Optional: create water-mask previews

This is for inspecting whether the pretrained segmentation model correctly detects the sea:

```bash
python step_1_a_prepare_water_masks.py
```

Open this file in a browser:

```text
water-mask-preview/index.html
```

Inspect the original, mask, and overlay images. The generated `original` and `overlay` files are previews and should not be committed.

## 3. Create standardized Step 1 inputs

[insert image here readme_files/step-preprocessed-sea-example.jpg]

Run:

```bash
python step_1_b_preprocess_water_inputs.py
```

For CPU-only processing:

```bash
python step_1_b_preprocess_water_inputs.py --device cpu
```

This reads from:

```text
step-0-raw-data/
```

and writes standardized `224x224` images to:

```text
step-1-processed-data/
```

Open:

```text
step-1-processed-data/index.html
```

The script skips images whose `step-1_<name>.jpg` output already exists.

## 4. Create the lower-water Step 2 images

[insert image here readme_files/step-2-pick-lower-water-area.jpg]

Run:

```bash
python step_2_a_reduce_black_water_area.py
```

For CPU-only processing:

```bash
python step_2_a_reduce_black_water_area.py --device cpu
```

This reads the Step 1 outputs and writes:

```text
step-2-final-water-data/
```

The crop selection is designed to retain the lower part of the detected water area, where waves are expected to be most visible.

Inspect the result here:

```text
step-2-final-water-data/index.html
```

The script skips images whose `step-2_<name>.jpg` output already exists.

## 5. Label the new Step 2 images

Start the labeling web app:

```bash
python step_2_b_label_data.py
```

Open this URL:

```text
http://localhost:8055
```

In the labeling UI:

1. Select “Unlabeled first”.
2. Set the waviness value from `0.00` to `1.00`.
3. Use increments of `0.05`.
4. Click “Save & Next”.
5. Continue until all new images are labeled.
6. Stop the server with `Ctrl+C`.

Labels are saved automatically to:

```text
labels.csv
```

The existing labels are preserved. You do not need to manually edit the CSV.

## 6. Rebuild the dataset splits

Run:

```bash
python step_3_a_prepare_dataset_split.py
```

This validates that:

- every labeled image exists in `step-2-final-water-data/`;
- labels are between `0.0` and `1.0`;
- filenames are unique and contain an image number;
- no capture group is split across train, validation, and test.

Existing assignments remain fixed. New capture groups are assigned incrementally, so previously trained data is not randomly reshuffled.

Review:

```text
step-3-dataset-splits/summary.txt
```

The generated manifests are:

```text
step-3-dataset-splits/train.csv
step-3-dataset-splits/validation.csv
step-3-dataset-splits/test.csv
```

A timestamped immutable snapshot is also created under:

```text
step-3-dataset-splits/snapshots/
```

## 7. Optional: inspect the training data

Run:

```bash
python step_4_c_inspect_training_data.py
```

Then open:

```text
step-4-data-preview/index.html
```

This checks tensor shapes, labels, deterministic validation/test transforms, and displays training augmentations.

## 8. Train a new model

Run:

```bash
python train.py
```

The training script:

- loads `train.csv` and `validation.csv`;
- uses the existing preprocessing outputs;
- automatically chooses the next model version;
- does not overwrite earlier checkpoints;
- saves the best checkpoint based on validation MAE;
- uses early stopping;
- writes a training summary and manifest.

New checkpoints will be created in:

```text
step-5-checkpoints/
```

The command prints the exact best checkpoint path. Record that path.

For this repository, the next run after `v1` and `v2` should normally create a `v3` checkpoint.

## 9. Use the new checkpoint for predictions

`step_7_a_predict.py` currently has the `v1` checkpoint hard-coded:

```text
step-5-checkpoints/wave-regression-baseline-v1-best-val-mae-epoch=45-val_mae=0.1068.ckpt
```

After training, update the `CHECKPOINT_PATH` in `step_7_a_predict.py` to the new checkpoint path printed by `train.py`.

Then put new inference photos in:

```text
predict-holder/
```

Run:

```bash
python step_7_a_predict.py
```

Predictions and processed previews are written to:

```text
step-7-inference-preview/
```

## Important notes

- `step_1_a_prepare_water_masks.py` is an inspection step; `step_1_b_preprocess_water_inputs.py` is the actual Step 1 training-data preparation.
- Do not train directly from raw photos. Training uses the Step 2 outputs.
- Do not delete or reshuffle existing split manifests; the repository intentionally preserves previous assignments.
- `step_6_a_evaluate_test.py` is configured for the original immutable `benchmark-v1` test evaluation and the original `v1` checkpoint. It is not automatically configured to evaluate the newly trained checkpoint.
- The current repository contains two existing training runs, `v1` and `v2`; a new training run should create the next version without overwriting them.
