# Prediction pipeline experiments: reproducible handoff

This is the handoff for the three named model-serving experiments. It is written for a reader new to this repository. **MAE means mean absolute error of predicted waviness scores.** Lower MAE is better. The actual/reference values below were supplied for these six photos; they were not available to calculate from the photos themselves. In each prediction tuple, values follow the photo order shown directly below.

## At-a-glance results

Photo order: `IMG_7927.HEIC`, `IMG_7928.HEIC`, `IMG_7929.HEIC`, `IMG_7931.HEIC`, `IMG_7932.HEIC`, `IMG_7933.HEIC`.

| Experiment | Full evaluation MAE (validation / test) | Six-photo warmed-server median raw→score | Six-photo warmed-server median client latency | Predictions in photo order |
|---|---:|---:|---:|---|
| **Distil-FP32** | 0.110818 / 0.110720 | 100.23 ms | 106.25 ms | 0.79608, 0.60330, 0.31019, 0.32640, 0.31529, 0.24815 |
| **Area-B0** | 0.131625 / 0.114461 | 102.96 ms | 110.27 ms | 0.51549, 0.55385, 0.28288, 0.21900, 0.43227, 0.19586 |
| **Stable-B2** | 0.131969 / 0.104235 | **89.26 ms** | **91.02 ms** | 0.70165, 0.53791, 0.34641, 0.27491, 0.30529, 0.25463 |

The corrected six-photo reference waviness scores, in that same order, are **0.6, 0.5, 0.4, 0.3, 0.2, 0.2**. This makes the per-photo comparison explicit:

| Photo | Reference score | Distil-FP32 prediction | Area-B0 prediction | Stable-B2 prediction |
|---|---:|---:|---:|---:|
| IMG_7927.HEIC | 0.6 | 0.79608 | 0.51549 | 0.70165 |
| IMG_7928.HEIC | 0.5 | 0.60330 | 0.55385 | 0.53791 |
| IMG_7929.HEIC | 0.4 | 0.31019 | 0.28288 | 0.34641 |
| IMG_7931.HEIC | 0.3 | 0.32640 | 0.21900 | 0.27491 |
| IMG_7932.HEIC | 0.2 | 0.31529 | 0.43227 | 0.30529 |
| IMG_7933.HEIC | 0.2 | 0.24815 | 0.19586 | 0.25463 |

Computing MAE from the corrected six user-supplied reference scores and the full-precision predictions gives **0.09651 Distil-FP32**, **0.09548 Area-B0**, and **0.06303 Stable-B2**. These are six-photo descriptive MAEs; do not confuse them with the larger validation/test MAEs in the first table, which come from separate 93-photo evaluations. The full-run date-cluster uncertainty analysis for each candidate still permits more than 1% MAE regression against v18 (and the distilled candidate has a concerning four-photo date-disjoint result); none should be called quality-cleared on point estimates alone.

The six-photo timings are from a **single process that had already loaded and warmed all models**, with requests sent one at a time in filename order. `raw→score` is server time from image decode through regression output. Client latency includes reading the local file, upload, and receiving the response. These figures do not include the server's one-time startup/load/warm-up cost. Distil-FP32 has one recorded six-photo warm-server pass; Area-B0 and Stable-B2 each have two sequential passes, and the summary uses pass 2 for those two. The six reference values let us calculate a small-sample MAE here; this does not replace the 93-photo evaluation.

For the detailed six-photo stage timings and predictions, see the JSON records:

- [Distil-FP32 server report](../step-7-predict-captures-holder/experiment-predict-holder/benchmark-results/b2-distilled-b0-migraphx-fp32-warm-server-requests-20261030.json)
- [Area-B0 server report](../step-7-predict-captures-holder/experiment-predict-holder/benchmark-results/migraphx-b0-opencv-area-warm-server-20261030.json)
- [Stable-B2 server report](../step-7-predict-captures-holder/experiment-predict-holder/benchmark-results/stable-migraphx-fp16-b2-opencv-nearest-warm-server-20261030.json)

## What each experiment is

All three candidates run the **same existing v18 EfficientNet-B0 waviness regression checkpoint** after two water-segmentation/preprocessing stages. The segmentation model identifies the water; the regression model produces the waviness score. No regression model was retrained for these server comparisons. Each long-running server process has one Step 1 segmentation session, one Step 2 segmentation session, and one regressor session in memory. The HTTP server is Python's single-threaded `HTTPServer`, so it accepts one request at a time and does not create a model copy per worker.

| Code name | Segmentation models/runtime | Main preprocessing/runtime details | Full-set MAE val/test | Server ready time |
|---|---|---|---:|---:|
| **Distil-FP32** | B2-distilled SegFormer-B0, MIGraphX FP32 | OpenCV resize; Pillow-compatible nearest mask expansion; zero-copy tiled native HEIF decode with 16 workers | 0.110818 / 0.110720 | 18.262 s in initial server setup |
| **Area-B0** | SegFormer-B0, MIGraphX FP32 | OpenCV area downscale / linear-exact upscale; Pillow-compatible nearest mask expansion; native tiled HEIF decode with 16 workers | 0.131625 / 0.114461 | 18.717 s |
| **Stable-B2** | SegFormer-B2, MIGraphX FP16 mode | OpenCV linear-exact input transform and nearest-exact mask expansion; HEIF decode with 24 workers | 0.131969 / 0.104235 | 39.087 s |

Ready time is measured from process launch through model/session initialization, warm-up, and listening. MIGraphX's initial graph setup/warm-up is a significant part of cold startup. A warm request avoids that repeated setup as long as the process stays alive.

### Quality and full-evaluation context

The reference baseline is the current v18 PyTorch route: validation MAE **0.135669**, test MAE **0.116597**. The candidate full-evaluation MAEs above are from runs of the complete raw-to-output pipeline over 93 labeled photos, not the six-photo server timing sample. Distil-FP32's date-cluster bootstrap permits a material test regression; on four test photos from a date absent from training, its MAE was 0.18644 versus 0.15744 for v18. Area-B0 and Stable-B2 also have bootstrap intervals that permit more than 1% regression. These are research candidates, not confirmed quality-preserving replacements.

## Model files and exact candidate mapping

These are the model artifacts used by the server. They were saved in `/tmp` for the experiments and are **not tracked in the repository**. File sizes are approximate (`ls -lh` rounded them to MB):

| Candidate | Step 1 segmentation graph (448×448 input) | Step 2 graph (384×384 input) | Regressor |
|---|---|---|---|
| Distil-FP32 | `/tmp/wave-segformer-b0-distilled-area-water-448.onnx` (~15 MB) | `/tmp/wave-segformer-b0-distilled-area-water-384-to224.onnx` (~15 MB) | `/tmp/wave-v18-efficientnet-b0-regressor.onnx` (~16 MB) |
| Area-B0 | `/tmp/wave-segformer-b0-water-448.onnx` (~15 MB) | `/tmp/wave-segformer-b0-water-384-to224.onnx` (~15 MB) | same v18 regressor (~16 MB) |
| Stable-B2 | `/tmp/wave-segformer-b2-water-448.onnx` (~106 MB) | `/tmp/wave-segformer-b2-water-384-to448.onnx` (~106 MB) | same v18 regressor (~16 MB) |

Each candidate therefore needs approximately **46 MB** of model files for B0 or **228 MB** for B2, before runtime libraries, compiled graph state, and memory used for image tensors. The B2 files remain about 106 MB each on disk; the server enables MIGraphX FP16 execution for them with a provider option. They are not smaller FP16 ONNX files.

Model provenance:

- Distil-FP32 uses the SegFormer-B0 binary water/not-water student distilled from the SegFormer-B2 teacher. Training used 217 training rows split by capture date (169 fit images / 39 dates; 48 selection images / 12 dates), with validation and test excluded. Teacher targets were regenerated for the exact OpenCV runtime resize transforms. The student had 0.9517 mean IoU against the teacher on the training-date holdout; that is segmentation teacher agreement, **not waviness MAE**. The training/export investigation is documented in [`experiments.md`](experiments.md), section “B2-to-B0 binary segmentation distillation with the runtime resize contract”. Source artifacts were in `/tmp/wave_b2_b0_binary_distill_area_v1/` and `/tmp/wave_migraphx_b0_distilled_area_full_eval_v5/`.
- Area-B0 uses the non-distilled B0 water segmenter exported to two static ONNX input shapes. It is the B0 MIGraphX + OpenCV area-preprocessing full-route candidate documented in [`experiments.md`](experiments.md), around the MIGraphX B0/OpenCV area comparison.
- Stable-B2 uses the stable SegFormer-B2 water segmenter ONNX exports. `MIGRAPHX_SKIP_BENCHMARKING=1` and `MIGRAPHX_USE_FAST_SOFTMAX=0` were set before importing MIGraphX to avoid the nondeterministic setup/configuration previously observed. Its final exact-nearest / 24-worker setup and quality analysis are in [`experiments.md`](experiments.md), sections “Stable MIGraphX FP16 determinism configuration” and “Six-photo warmed-server run”.

**Reproducibility constraint:** the ONNX graphs, regression ONNX, and the tested Python virtual environment currently live in `/tmp`; they are not source-controlled. Their exact bytes cannot be recovered from this document alone if that temporary data is deleted. Recreating the segmentation graphs requires the training/export artifacts or checkpoints described in `experiments.md` and its `/tmp` run paths. The v18 regression source checkpoint is the repository's selected v18 checkpoint; it was exported to ONNX for this server. Preserve/copy the needed ONNX files and environment when handing this experiment to another machine. Do not silently substitute another checkpoint and call the result the same experiment.

## Repository setup and runtime environment

Repository root: `/home/ubuntu/dev/deep-learning/wave-regression-model`. The serving entry point is [`step_7_i_experimental_migraphx_server.py`](../step_7_i_experimental_migraphx_server.py). Supporting repository files are `step_0_a_native_heif_decoder.py` (optional native HEIF wrapper) and `step_2_b_crop_selector.py` (lower-water crop selection). The benchmark image set and outputs live in `step-7-predict-captures-holder/experiment-predict-holder/`.

The measured runtime was a Python 3.12 virtual environment at `/tmp/wave_migraphx_312_env`, on a Linux machine with a supported AMD GPU/ROCm stack and MIGraphX ONNX Runtime execution provider. Observed versions included ONNX Runtime 1.29.0, MIGraphX EP package `1.0.0+rocm10.1.0`, OpenCV 5.0.0, NumPy 2.5.3, Pillow 12.3.0, and pillow-heif 1.8.0. The repository's `requirements-common.txt` is **not sufficient by itself** to install the MIGraphX provider; use a compatible ROCm/MIGraphX installation for the target host. The exact tested environment is under `/tmp` and is not committed.

Before starting, verify that all three selected model files exist, the native library path below exists, the GPU/MIGraphX provider is visible to ONNX Runtime, and the six HEIC inputs are present. The experiment server checks missing graphs and provider availability and exits with a clear error. The optional native decoder can be rebuilt after installing `pillow-heif` and a C++17 compiler:

```bash
/tmp/wave_migraphx_312_env/bin/python step_0_b_build_native_heif.py
```

This generates `native/build/libwave_heif_tiles.so`, linked to the libheif bundled with that environment. The server can fall back to Pillow HEIC decoding if the helper is unavailable, but that changes performance and is not an apples-to-apples reproduction of the measured runs.

MIGraphX library paths and deterministic settings used for launch:

```bash
export LD_LIBRARY_PATH="/tmp/wave_migraphx_312_env/lib/python3.12/site-packages/migraphx_libs:/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MIGRAPHX_SKIP_BENCHMARKING=1
export MIGRAPHX_USE_FAST_SOFTMAX=0
```

## Start one experiment server

The server binds only to `127.0.0.1:8765`, warms its models before reporting ready, keeps them in memory, and serves requests sequentially. Stop the existing server with Ctrl-C before starting a different candidate on the same port. The server left running after the recorded runs was Stable-B2. Choose a candidate key with `WAVE_EXPERIMENT`:

```bash
# Distil-FP32
WAVE_EXPERIMENT=distilled-b0-migraphx-fp32 /tmp/wave_migraphx_312_env/bin/python -u step_7_i_experimental_migraphx_server.py

# Area-B0
WAVE_EXPERIMENT=migraphx-b0-opencv-area /tmp/wave_migraphx_312_env/bin/python -u step_7_i_experimental_migraphx_server.py

# Stable-B2
WAVE_EXPERIMENT=stable-migraphx-fp16-b2-opencv-nearest /tmp/wave_migraphx_312_env/bin/python -u step_7_i_experimental_migraphx_server.py
```

Run from the repository root after exporting the library path/settings above. The default candidate is Distil-FP32, but specifying the key explicitly prevents accidentally benchmarking the wrong model. `WAVE_SERVER_PORT` can override the port. Optional `WAVE_STEP1_ONNX`, `WAVE_STEP2_ONNX`, and `WAVE_REGRESSOR_ONNX` override graph paths; only use them if intentionally running a different artifact and record that change.

Wait for the startup JSON line with `"ready": true`, and check:

```bash
curl -fsS http://127.0.0.1:8765/health
```

It reports the selected candidate. Startup JSON reports both model paths, execution mode, per-session loads, warm-up costs, and total launch-to-ready time.

## Send one photo or benchmark the six photos

A raw HEIC/JPEG/PNG photo is posted to `/predict`. The file extension must be passed in `X-Filename`. Add `?save_steps=1` to save the two model-input previews in candidate-specific folders. Requests must be sent one at a time to preserve the sequential measurement protocol.

```bash
curl -fsS -X POST 'http://127.0.0.1:8765/predict?save_steps=1' \
  -H 'X-Filename: IMG_7927.HEIC' \
  --data-binary @step-7-predict-captures-holder/experiment-predict-holder/IMG_7927.HEIC
```

Repeat individually for photos in the fixed order from the first table. The response contains `prediction`, total request time, server raw-to-score time, stage timings, and preview directory paths. Do not run parallel requests or include model startup time in a warm-request latency comparison. To reproduce the cold cost, start a fresh process and measure process launch through its ready JSON separately from warm image requests.

### Pipeline stages

1. Decode the raw HEIC/JPEG/PNG into oriented RGB pixels. The B0 candidates use the native tiled HEIF decoder with the configured 16 workers; Stable-B2 uses libheif decoding with 24 workers. The native helper is optional fallback-sensitive as described above.
2. Prepare a normalized NCHW FP32 tensor and run Step 1 water segmentation at 448×448.
3. Expand the water mask to source-image dimensions, erode it, and select the detected water area. The selected sea ROI is masked, Lanczos-resized to fit a black 224×224 canvas, and encoded/decoded as JPEG quality 95. The lower part of detected water is preferred at crop selection.
4. Prepare the Step 1 JPEG for Step 2 at 384×384 and run the second segmentation graph. Exclude black canvas pixels and use the crop selector to choose a lower-water rectangle. Resize it to the 224×224 model canvas and JPEG encode/decode at quality 95, subsampling 0.
5. Normalize the final image and run the existing v18 EfficientNet-B0 regressor on ONNX Runtime CPU. Its scalar output is the predicted waviness score.

The segmentation and crop operations happen on the full-resolution data where applicable; the 448/384 dimensions are the model tensor dimensions, not dimensions of the incoming photo. The server returns diagnostic stage times so step totals and substages are not double-counted.

## Six-photo results and per-photo latency

The corrected reference scores are 0.6, 0.5, 0.4, 0.3, 0.2, and 0.2 in the photo order shown above. These yield a descriptive six-photo MAE of 0.09651, 0.09548, and 0.06303 for Distil-FP32, Area-B0, and Stable-B2 respectively (computed from full-precision predictions before display rounding). Distil-FP32 individual latencies are from its one recorded warm-server pass; Area-B0 and Stable-B2 values are from pass 2:

| Photo | Distil-FP32 prediction / latency | Area-B0 prediction / latency | Stable-B2 prediction / latency |
|---|---:|---:|---:|
| IMG_7927.HEIC | 0.79608 / 94.26 ms | 0.51549 / 103.34 ms | 0.70165 / 87.77 ms |
| IMG_7928.HEIC | 0.60330 / 95.81 ms | 0.55385 / 100.39 ms | 0.53791 / 85.70 ms |
| IMG_7929.HEIC | 0.31019 / 92.94 ms | 0.28288 / 99.22 ms | 0.34641 / 87.39 ms |
| IMG_7931.HEIC | 0.32640 / 96.32 ms | 0.21900 / 109.14 ms | 0.27491 / 94.64 ms |
| IMG_7932.HEIC | 0.31529 / 92.07 ms | 0.43227 / 105.87 ms | 0.30529 / 90.75 ms |
| IMG_7933.HEIC | 0.24815 / 94.55 ms | 0.19586 / 102.57 ms | 0.25463 / 98.49 ms |

Median server raw-to-score was 100.23 ms / 102.96 ms / 89.26 ms for Distil-FP32 / Area-B0 / Stable-B2. Pass 2 client medians were 106.25 / 110.27 / 91.02 ms, respectively. The client median can be larger than server raw-to-score because it includes the local-file read and HTTP transfer/response; preview files were also saved during these requests. Saving previews adds a small write cost and should be disabled for a pure production-latency measurement.

Full-set benchmark context (separate from the six-photo server run):

| Candidate | Validation/test MAE | Validation/test total raw-to-output median | Context |
|---|---:|---:|---|
| Current v18 PyTorch reference | 0.135669 / 0.116597 | 236.74 / 240.92 ms | Official baseline |
| Distil-FP32 MIGraphX FP32 | 0.110818 / 0.110720 | 78.69 / 79.84 ms | 93 photos; 92/93 under 100 ms; four date-disjoint photos had 0.18644 MAE vs v18 0.15744 |
| Area-B0 MIGraphX FP32 | 0.131625 / 0.114461 | 81.27 / 83.60 ms | 93-photo evaluation; uncertainty allows material MAE regression |
| Stable-B2 MIGraphX FP16 | 0.131969 / 0.104235 | 92.76 / 92.34 ms | Stable exact-nearest, 24 HEIF workers; 92/93 under 100 ms |

The full-run latencies above are from a standalone process/evaluation harness and are not the same measurement as the later six-photo HTTP server timings. Use the server results when comparing the always-loaded setup and the full-run table for the broader evaluation cohort. Preserve that distinction in future reports.

## Processed images and experiment artifacts

The server saves six Step 1 and six Step 2 preview images per candidate here:

- Distil-FP32: `step-7-predict-captures-holder/experiment-predict-holder/preprocessed/1-distilled-b0-migraphx-fp32/{step-1,step-2}/`
- Area-B0: `step-7-predict-captures-holder/experiment-predict-holder/preprocessed/2-migraphx-b0-opencv-area/{step-1,step-2}/`
- Stable-B2: `step-7-predict-captures-holder/experiment-predict-holder/preprocessed/3-stable-migraphx-fp16-b2-opencv-nearest/{step-1,step-2}/`

The benchmark JSON files are under `step-7-predict-captures-holder/experiment-predict-holder/benchmark-results/`. They preserve startup setup, sequential run order, per-photo predictions, request and stage timings, and preview hashes. The preprocessed JPEGs are experiment outputs; original photos and generated originals/overlays remain subject to repository image-tracking rules in `AGENTS.md`.

## Stop condition and conclusion

This optimization campaign is wrapped up. Do not continue trying unrelated model or preprocessing variants as part of this experiment. The practical next decision is whether to keep a candidate for focused quality validation or archive the results. A future candidate should only be advanced after it is evaluated on the same full raw-to-output path and labeled dates, with validation/test MAE and capture-date uncertainty checked against v18. The 100 ms target is an end-to-end target; a warm server's raw-to-score timing alone is not the same as a cold-start or client-observed total.
