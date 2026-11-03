# Sea Wave Regression Model

[![neural-networks](https://img.shields.io/static/v1?message=neural-networks&color=2f6feb&style=flat-square)](https://github.com/topics/neural-networks) [![regression-model](https://img.shields.io/static/v1?message=regression-model&color=2f6feb&style=flat-square)](https://github.com/topics/regression-model) [![python](https://img.shields.io/static/v1?message=python&color=2f6feb&style=flat-square)](https://github.com/topics/python) [![median-inference](https://img.shields.io/static/v1?message=inference-latency-42-ms%2Fphoto&color=16875b&style=flat-square)](#inference-speed-improvements)

This neural network takes a photo of the **sea** and estimates water waviness as a number from **0 to 1**. A score of 0 represents very calm water; a score of 1 represents very wavy water that may be dangerous for swimming. The score describes waviness and is not, by itself, a complete swimming safety assessment.

<img src="readme_files/waviness-scale.svg" alt="Waviness score scale from 0, very calm water, to 1, very wavy water that may be dangerous to swim, with guidance for fun waves and taking precautions" width="85%" />

This project explores whether a photo of the sea can help answer a familiar local question: **how wavy is the water today, and is it a good day to swim?** Swimming is part of everyday life where I live, and I used to check the sea from my window each morning. I started collecting photos to see whether a computer vision model could make that daily check more consistent.

The repository contains an image-processing and model-training pipeline for a local swimming application. The model estimates **water waviness from an image**; it does not yet make a complete safety judgment.

## How the pipeline prepares an image

Photos are taken from roughly 200–300 metres from the sea, often from a few recurring locations and angles. The preprocessing pipeline focuses on the sea, removing visual distractions such as sky, buildings, beach, and sand. It then selects a region toward the bottom of the detected sea area, where waves near the coast are most relevant to the swimming experience.

<p align="center">
  <img src="readme_files/original-to-crop.png" alt="Original sea photo compared with the lower-water crop selected for the model" width="90%" />
  <br /><em>An original photo and the lower-water crop used as model input.</em>
</p>

For the current fast-v2.2 pipeline, during pre-processing the pipeline segments the sea and selects a lower-water crop directly from the original photo to create the final model input. The model is trained on this image.

### Inference speed improvements

The adopted pipeline reduced median raw-photo-to-score time from **10.716 s** to **42.59 ms** per photo, about **252× faster**. The following table shows measured changes from the paired FP32-to-FP16 comparison. Stage timings come from successive experiments, so they show the scale and source of the gains.

<table>
  <thead>
    <tr>
      <th align="left">⚙️ Change</th>
      <th align="right">⏱️ Before</th>
      <th align="right">⚡ After</th>
      <th align="right" width="24%" nowrap>📈 Improvement</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>🧩 Step 1: replace 35 overlapping tiles with one 448 × 448 pass</td>
      <td align="right"><code>10.3 s</code></td>
      <td align="right"><code>225 ms</code></td>
      <td align="right"><strong>45.8× faster</strong></td>
    </tr>
    <tr>
      <td>🎯 Step 2: reduce segmentation input to 384 × 384 (earlier fast path)</td>
      <td align="right"><code>305 ms</code></td>
      <td align="right"><code>59 ms</code></td>
      <td align="right"><strong>5.2× faster</strong></td>
    </tr>
    <tr>
      <td>✂️ Current fast-v2 route: remove the separate Step 2 segmentation pass</td>
      <td align="right"><code>305 ms</code></td>
      <td align="right">—</td>
      <td align="right"><strong>305 ms removed</strong></td>
    </tr>
    <tr>
      <td>🧮 Segmentation runtime: FP32 → MIGraphX FP16¹</td>
      <td align="right"><code>58 ms</code></td>
      <td align="right"><code>43 ms</code></td>
      <td align="right"><strong>15 ms saved</strong></td>
    </tr>
    <tr>
      <td>🧠 Regression inference</td>
      <td align="right"><code>6 ms</code></td>
      <td align="right"><code>5 ms</code></td>
      <td align="right"><strong>≈1 ms saved</strong></td>
    </tr>
    <tr>
      <th align="left">🚀 Total raw photo → waviness score</th>
      <td align="right"><strong>10.716 s</strong></td>
      <td align="right"><strong>42.59 ms</strong></td>
      <td align="right"><strong>252× faster</strong></td>
    </tr>
  </tbody>
</table>

## Model development

The first model versions used a ResNet backbone. An early version reached a validation mean absolute error (MAE) of about **0.10**. As we added more photographs, the dataset also gained noisier or more ambiguous examples, and the validation MAE rose across subsequent versions, reaching about **0.17**.

That prompted a set of controlled experiments to understand model behavior and improve it. We designed experiments in a way that they can be repeated, and the repeated experiment results will be recorded in versioned directories.

### Experiment A — Constant prediction baselines

We compared the neural network with two simple guesses that do not use the validation photos, then measured them against the network's image-based predictions. Here is how the experiment works:

1. Calculate the average waviness label from the training photos.
2. <img src="readme_files/badge-blind-average.svg" alt="BLIND AVERAGE" height="18" style="vertical-align: middle" /> Assign that same average score as the guess for every validation photo, without looking at the photo.
3. Compare each fixed guess with that validation photo's actual waviness label.
4. Calculate the average error across all validation photos. This is the baseline's mean absolute error (MAE).
5. <img src="readme_files/badge-blind-medium.svg" alt="BLIND MEDIUM" height="18" style="vertical-align: middle" /> Repeat steps 2–4 using the middle training-set score (the median) as the same fixed guess for every validation photo.
6. <img src="readme_files/badge-actual-prediction.svg" alt="ACTUAL PREDICTION" height="18" style="vertical-align: middle" /> Load the selected neural network checkpoint and give it each validation image. The model looks at each image and predicts its waviness score.
7. Compare the model's MAE with the two baseline MAEs, calculated on the same validation photos.

In this run, the training-mean baseline had an MAE of **0.219**, the training-median baseline had an MAE of **0.222**, and the EfficientNet-B0 model had an MAE of **0.141** on 17 validation images. Lower MAE means the predictions were closer to the actual labels on average. The test set was not evaluated in this experiment. [Read the baseline report](readme_files/experiment-a-baseline-report.md).

### Experiment B — Per-image error analysis

Aggregate MAE can hide individual failures. We plotted actual and predicted waviness for each validation image and reviewed prediction errors image by image. The analysis helps identify cases where the model performs well or struggles, and guides later data collection and experiments.

<img src="readme_files/experiment-b-actual-vs-predicted.png" alt="Actual versus predicted waviness for the current validation run" width="80%" />

*Each point represents a validation image. The diagonal indicates a perfect prediction.*

### Experiment C — Frozen backbone comparison

We compared frozen pretrained ResNet-18, ResNet-34, and EfficientNet-B0 backbones across three random seeds, using the same recent dataset split. EfficientNet-B0 had the lowest mean validation MAE: **0.152**, compared with **0.185** for ResNet-18 and **0.188** for ResNet-34. The backbones performed similarly in some respects, but EfficientNet-B0 gave the strongest average result in this experiment and became the preferred backbone for continued work.

![Validation MAE comparison across frozen backbones](readme_files/experiment-c-validation-mae-comparison.png)

*Mean validation MAE across three seeds; lower is better. This comparison used validation data, not the held-out test set.*

### Experiment D — Partial fine-tuning

Next, we unfroze the final layer of the EfficientNet-B0 backbone to test whether partial fine-tuning helped. It did not improve average validation MAE: the partially fine-tuned model scored **0.156**, compared with **0.152** for the frozen model, and it performed worse across all three paired seeds. For this dataset snapshot, the results favor keeping the backbone frozen. The validation set is small, so this is evidence for the current setup rather than a general claim. [Read the Experiment D report](readme_files/experiment-d-partial-finetuning-report.md).

## Current status

- The pipeline is designed for a dataset that grows over time. Images are labeled with a waviness score from 0 to 1, and training, validation, and test splits are managed separately.
- The pipeline supports image preprocessing, labeling, dataset splits, training, evaluation, and inference.
- Step 7 now defaults to **fast-v2.2**: native HEIF decoding when available (with a Pillow fallback), 448×448 SegFormer-B0 ONNX segmentation through MIGraphX FP16, lower-water coordinate crop from the source image, and the retrained EfficientNet-B0 ONNX regressor. The earlier fast-v2.1 and PyTorch v19 routes remain selectable.
- EfficientNet-B0 is the current preferred backbone based on validation experiments.
- The fast-v2.2 regressor was trained on fast-v2.2-preprocessed inputs and selected at **0.0985 validation MAE**. Its held-out test MAE was **0.1324** across 46 photos; the experiment dashboard includes the FP32 and FP16 reference comparisons and full per-photo results.
- The first model version is available on [Hugging Face](https://huggingface.co/kubilaycaglayan/wave-regression).
- More varied, carefully labeled data is needed to evaluate how well predictions generalize across days, weather, locations, and camera angles.

## Run the project

### Set up the environment

Run commands from the repository root. Use Python 3.14 on this machine and
create an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

After setup, follow the [pipeline run guide](README_RUN.md) for preprocessing,
labeling, training, evaluation, and prediction steps.

## About this project

This is a learning project built around a practical local question. The goal is to make each pipeline stage inspectable, preserve experiment results, and improve the model as new sea photographs and labels are collected.
