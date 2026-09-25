# Experiment B — Per-Image Error Analysis

## Objective

Analyze the predictions of our existing wave regression model to understand its generalization behavior, systematic biases, and failure cases.

Experiment A established that ResNet v10 achieves a validation MAE of 0.175786, outperforming the training-mean baseline by 19.85%.

Experiment B must investigate WHERE and WHY the model succeeds or fails.

This is an evaluation-only experiment. Do not train or modify the model.

## 1. Reuse the existing experiment

Use the exact dataset split from Experiment A:

`20260923T133043052012Z`

* Training: 68 images
* Validation: 14 images
* Test: 14 images

Use the best v10 checkpoint:

`step-5-checkpoints/wave-regression-baseline-v10-best-val-mae-epoch=42-val_mae=0.1758.ckpt`

Do not generate a new split.

Do not evaluate the test set.

## 2. Generate per-image predictions

For each validation image, record:

* Image filename or identifier.
* Actual waviness label.
* Neural network prediction.
* Signed error: prediction − actual.
* Absolute error.
* Mean-baseline prediction: 0.38235294.
* Mean-baseline absolute error.
* Whether the neural network beats the baseline on that image.

Save the complete results as CSV and JSON.

## 3. Calculate aggregate metrics

Calculate:

* MAE.
* RMSE.
* Mean signed error (bias).
* Median absolute error.
* Maximum absolute error.
* Number and percentage of images where the neural network beats the mean baseline.

Also calculate MAE separately for these actual-waviness ranges:

* 0.00–0.19
* 0.20–0.39
* 0.40–0.59
* 0.60–0.79
* 0.80–1.00

Report the sample count in each range.

Do not draw strong conclusions from bins containing very few images.

## 4. Generate visualizations

Create the following plots.

### A. Actual vs predicted waviness

Scatter plot:

* X-axis: Actual waviness.
* Y-axis: Predicted waviness.
* Include the perfect-prediction reference line y = x.
* Set both axes to 0–1.

This plot should help identify regression toward the mean and systematic underestimation or overestimation.

### B. Residual plot

Scatter plot:

* X-axis: Actual waviness.
* Y-axis: Signed prediction error.
* Include a horizontal reference line at zero.

Positive error means overestimation. Negative error means underestimation.

### C. Per-image error comparison

Compare the neural network's absolute error against the mean baseline's absolute error for each validation image.

Sort images by neural network absolute error.

### D. Prediction distribution

Compare actual and predicted waviness distributions.

Check whether predictions occupy a narrower range than the actual labels.

## 5. Generate a visual error gallery

Create an HTML report displaying every validation photograph.

For each image, show:

* Original or actual model-input photograph.
* Actual waviness label.
* Predicted waviness.
* Absolute error.
* Mean-baseline absolute error.

Sort images by neural network absolute error, descending.

Highlight cases where the neural network performs worse than the baseline.

The gallery should allow manual inspection of lighting, reflections, foam, wave patterns, and other visual conditions.

Do not automatically invent explanations for the errors based only on the metrics.

## 6. Validate the experiment

Verify that:

* Exactly 14 validation images are evaluated.
* The dataset split matches Experiment A.
* The checkpoint matches Experiment A.
* The reproduced validation MAE is approximately 0.175786.
* The reproduced mean-baseline MAE is approximately 0.219328.
* No training occurs.
* The test dataset remains untouched.

If these checks fail, investigate the discrepancy rather than reporting results from incompatible data.

## 7. Deliverables

Create a dedicated experiment directory containing:

* `predictions.csv`
* `predictions.json`
* `metrics.json`
* `actual_vs_predicted.png`
* `residuals.png`
* `error_comparison.png`
* `prediction_distribution.png`
* `error_gallery.html`
* `experiment_report.md`

The report should summarize the numerical findings, identify observable error patterns, distinguish evidence from hypotheses, and recommend the next experiment based on the results.

Do not change the training architecture, hyperparameters, dataset split, or existing checkpoint.

## Acceptance criteria

Experiment B is complete when we can answer:

1. Does the model systematically underestimate high waviness or overestimate low waviness?
2. Does the neural network outperform the constant baseline across most validation photographs, or is the aggregate improvement concentrated in a few?
3. Which validation photographs produce the largest errors?
4. Is there evidence of prediction-range compression?
5. What specific failure patterns should guide our next experiment?
