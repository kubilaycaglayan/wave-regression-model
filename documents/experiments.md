# Experiments

## Required reporting for future experiments

For every future speed experiment, record task-quality results alongside
latency for the fastest candidates, even when they miss the speed target, before
calling any candidate a best performer. At minimum, report
waviness MAE and RMSE on the same labeled evaluation split and the same
successful-image cohort as the reference, plus candidate/reference prediction
delta and the number of inference or preprocessing failures. State the
checkpoint, preprocessing/input contract, dataset split or snapshot, and
whether the evaluation data influenced training or candidate selection. For
segmentation or preprocessing changes, also report mask quality against the
canonical masks when available. Report complete-request latency (including
decode, preprocessing, regression, and output write) separately from stage
timings, with sample count and median/p95/max; include memory and CPU use where
measured.

If a candidate has no ground-truth labels or comparable reference outputs, say
that its task quality is unknown; speed or mask plausibility alone is not a
quality result. Identify the strongest candidates at the end of each future
experiment with both their quality and speed metrics. This reporting rule applies prospectively; earlier experiment results are left
as recorded and are not being backfilled.

### Quality gate for adopting speed changes

Treat preserving quality as an adoption requirement, not merely a metric to
report. First classify whether a change is output-equivalent. For changes that
only remove redundant work, require pixel/JPEG-byte or prediction equality at
the appropriate pipeline boundary, plus unchanged success/failure coverage;
then measure the speed gain on complete requests. For changes that alter
decoded pixels, resize filters, segmentation masks, crop selection, or model
inputs, do not call them quality-preserving based on aggregate MAE/RMSE alone.
Evaluate a matched candidate on labeled validation and untouched test data,
compare paired per-image absolute errors and prediction deltas on the same
successful cohort, report failures separately, and include uncertainty for
paired MAE/RMSE differences. Do not adopt a candidate when the results show a
quality regression or when the available evidence cannot rule out a material
regression; keep it experimental and gather independent future-capture labels.
An aggregate improvement does not cancel meaningful per-image regressions.
This gate applies to future experiments only and does not revise earlier
results.

## Computer-vision performance principles and current evidence

The optimization order follows the measured request profile: time the complete
request and its stages, identify the dominant stage, then compare one change at
a time. PyTorch recommends `torch.inference_mode()` for pure inference, profiling
CPU and device activity, and benchmarking memory formats instead of assuming
they help. This code already uses `inference_mode`; channels-last trials on both
the small regression model and the canonical SegFormer route were effectively
neutral on this AMD ROCm system. See the official [PyTorch performance tuning
guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html),
[inference mode reference](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html),
and [profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html).

Treat copies and memory as measurable costs. Existing experiments found exact
RGB decode copy removal saved about 6 ms median, and an ROI-only standardization
saved about 10 ms median while preserving JPEG bytes. Track process RSS as well
as PyTorch accelerator allocation: decoder and image-library buffers are not
fully represented by the PyTorch allocator. Manual host-memory pinning is not a
promising single-photo optimization here: PyTorch documents that pinning in the
calling thread is blocking and can cost more than it saves; its stronger use
case is overlapping transfers from a multi-item data loader. See the official
[pin-memory guidance](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html).

Do not increase worker or decoder threads by default. The HEIF sweep showed
that higher thread counts can lower decode latency but raise CPU use and peak
RSS; the full-request 24-thread follow-up did not show a robust paired gain.
Choose settings against the actual single-request workload and report CPU time
and memory with wall time.

The remaining route to sub-100 ms is constrained by input decode and full-frame
mask standardization, while the regression forward pass is only a small part of
the request. Faster resize backends and reduced-resolution masks did lower
latency, but changed pixels and degraded current-checkpoint quality. They are
not quality-preserving optimizations. The safe next work is limited to exact
output transformations or a separately trained, contract-matched candidate
that passes the required quality gate; do not infer quality from mask IoU or
speed alone. AMD's [HIP performance guidance](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html)
and [MIGraphX tuning guide](https://rocm.docs.amd.com/projects/AMDMIGraphX/en/develop/how-to/model-performance-tuning.html)
also emphasize profiling on the target hardware before tuning layouts or
engine options. Prior MIGraphX trials improved one FP32 preprocessing route but
remained well above 100 ms, and the faster FP16 route changed outputs.

## Exploratory sub-100 ms investigation

A separate local investigation on the AMD Radeon 8060S used the 32-photo
validation snapshot `20261015T083233550489Z`, the v16 baseline checkpoint,
four CPU threads, FP16 segmentation, and a warmed process. HEIC inputs used
embedded 320-pixel thumbnails; the one JPEG used Pillow's reduced-resolution
`draft` decode. The candidate ran Step 1 at 448 pixels and Step 2 at 224 pixels.
This changes the input and must not be used with a checkpoint claiming the
existing `fast-v1` contract.

In one pass over the photos, the 31 predictions after the first took 87.2 ms
median and 95.4 ms nearest-rank p95, from decode through regression. The JPEG
was 101.8 ms. This timing excludes model startup, watcher notification, preview
and record writes, and queueing; it is not a watcher benchmark or a guarantee.
One JPEG and one warm measurement per photo are insufficient to characterize
other input formats and latency variability.

With the original regression head, validation MAE was 0.1703 versus 0.1567
for the saved baseline inputs. A frozen-backbone head adapted on 149 training
photos, selecting settings on training dates held out before refitting, achieved
0.1663 validation MAE. The validation set was used in development and the test
set was not evaluated. Thumbnail availability, EXIF orientation, failure and
fallback behavior, wave detail, and a correctly recorded preprocessing contract
need further investigation. The temporary experiment scripts and results are in
`/tmp/wave_thumbnail_draft256.py`, `/tmp/wave-thumbnail-draft256-224.json`, and
`/tmp/wave-thumbnail-head-draft256/result.json`; temporary checkpoints do not
encode the actual thumbnail contract and must not be used for normal inference.

A 320 / 224 candidate with the same decoding approach measured 64.8 ms median
and 72.0 ms p95 for 31 warm decode-to-prediction samples. Its adapted head
achieved validation MAE 0.1676, versus 0.1567 on saved baseline inputs; the
same training-date selection method was used. An isolated watcher with a
monkey-patched thumbnail loader and the unadapted v16 checkpoint measured
32 atomic arrivals to completed records: median 69.1 ms, p95 78.2 ms, maximum
84.9 ms, including the single JPEG. It excluded startup, file upload/copy,
queued arrivals and concurrent load. Its local report is
`/tmp/wave-thumbnail-arrival/20261028T080104176408Z/arrival.json`. The watcher
experiment is exploratory and does not implement a safe checkpoint contract;
quality for that exact run is the unadapted candidate MAE 0.1852. The adapted
checkpoint was evaluated separately and its watcher latency was not measured.

## Quality-preserving investigation: initial findings

The preceding thumbnail experiments changed model inputs and increased validation
MAE, so they do not satisfy the quality-preservation requirement. The existing
JPEG round trips in prediction intentionally reproduce the training inputs; their
removal would change the input contract and cannot be assumed lossless.

The existing warm 448 / 384 benchmark on 32 validation photos measured median
94 ms input decoding, 104 ms Step 1 segmentation, 115 ms Step 1 mask and
standardization, 58 ms Step 2, and 391 ms total. These are independently
measured medians and do not sum exactly. The timing excluded startup. This
already shows that shaving copies alone may not reach a 100 ms complete budget.

A preliminary isolated synthetic 3024 × 4032 class-map comparison (three class
IDs, five timed samples) produced identical masks using the existing PyTorch CPU
operations and a NumPy logical-or implementation. Their medians were 3.51 ms
and 2.28 ms respectively. This is a microbenchmark, not an end-to-end result;
actual class IDs, memory pressure, diverse masks, and the output pipeline need
verification before adoption.

References: [PyTorch performance tuning](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
recommends profiling and avoiding unnecessary CPU/GPU synchronization; its
[channels-last guide](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html)
describes a layout worth benchmarking, with hardware-dependent results. AMD's
[HIP performance guidelines](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html)
recommend profiling before choosing compute or memory optimizations. Any such
change must be checked for identical masks, crops, and predictions, not assumed
to preserve them.

### Full-resolution decode copy experiment

On the same 32 validation source photos, decoding with Pillow's in-place EXIF
transpose followed by RGB conversion produced pixel-identical RGB arrays to the
existing loader on every photo. Three interleaved runs per method per image
were summarized by per-image medians: baseline median 74.22 ms, candidate median
68.11 ms, median paired reduction 6.27 ms. This is an isolated warm decode
microbenchmark and not an end-to-end result. It preserves the actual source
resolution and RGB pixels on these samples; other orientation tags, formats,
metadata handling, and memory behavior require checks before adoption. The
experiment script is `/tmp/wave_decode_copy_check.py`.

### GPU mask erosion microbenchmark

On a synthetic 3024 × 4032 binary mask, the existing SciPy minimum filter
(radius 12) took 67.1 ms median over four calls. A separable PyTorch GPU
max-pooling implementation, including CPU-to-GPU transfer, conversion and
return to CPU, took 12.26 ms median over five warmed calls. These are separate
small measurements rather than paired end-to-end timings; performance depends
on the mask and workload. The two implementations produced identical masks for
full, empty, rectangular and sparse-hole synthetic inputs at 3024 × 4032,
4032 × 3024 and 33 × 47, including borders. The random-mask benchmark was also
identical. These checks do not establish equivalence on actual photos or other
sizes and devices. GPU memory and impact on subsequent inference remain to be
measured before considering integration.

On the 32 validation sources, a separate experiment ran the 448 / 384 FP16
preprocessing path twice per photo with the same v16 checkpoint and compared
SciPy and GPU erosion. All 32 resulting eroded masks, encoded final model
inputs, and regression predictions matched exactly. This establishes equality
for these examples and this environment, not every possible image or hardware.

A separate timing run alternated the order on each photo and omitted the first
photo as warmup. Across 31 pairs, decode through prediction (without watcher,
preview or record writes) was 369.2 ms median / 400.7 ms p95 for CPU erosion
and 341.8 ms / 389.0 ms for GPU erosion. The median paired reduction was 24.9 ms.
Startup is excluded and p95s are independently computed. Scripts and logs are
`/tmp/wave_real_mask_check.py`, `/tmp/wave_real_mask_check.log`,
`/tmp/wave_real_mask_speed.py`, and `/tmp/wave_real_mask_speed.log`. The
experiment did not replace the production path or measure GPU memory use.

### GPU erosion memory observation

For one real 3024 × 4032 mask with the segmentation model already loaded,
PyTorch reported an incremental peak allocation of 244,879,360 bytes for the
float32 GPU erosion implementation, including its temporaries. Using float16
for the binary values reduced that to 171,722,752 bytes; both outputs matched
SciPy exactly for this one mask. A single cold call took 25.6 ms and 18.6 ms,
respectively, but those single calls are not a reliable timing comparison.
Peak allocation excludes allocations outside PyTorch and reserved memory is
larger. The float16 implementation requires full validation and paired warm
end-to-end timings. Scripts: `/tmp/wave_gpu_erosion_memory.py` and
`/tmp/wave_gpu_erosion_fp16_memory.py`.

The float16 binary erosion variant was subsequently compared on all 32
validation sources: all 32 masks, final JPEG inputs and predictions matched
exactly against the CPU implementation in this environment. In a separate
alternating-order timing run of 31 warmed photo pairs, CPU erosion gave
362.7 ms median / 398.7 ms p95 and the float16 GPU variant gave 337.3 ms /
373.0 ms for decode through prediction. The median paired saving was 26.4 ms.
Watcher, writes, and startup were excluded. Logs and scripts are under
`/tmp/wave_real_mask_check_fp16.*` and `/tmp/wave_real_mask_speed_fp16.*`.

### PyTorch compilation probe

A single-photo probe compiled the existing 448-pixel FP16 segmentation model
with `torch.compile(mode="reduce-overhead")` on the ROCm GPU. Compilation plus
first execution took around 20 seconds, and the two subsequent calls to the
full segmentation function took 138 ms and 98 ms. The class map differed from
eager execution at 369 source-resolution pixels on each compiled call. The
probe did not measure a warmed eager comparator, complete predictions, or
quality impact, so it provides no usable speedup estimate. Since exact output
preservation has not been established, this candidate is not adopted. Script
and log: `/tmp/wave_compile_probe.py` and `/tmp/wave_compile_probe.log`.

### Image masking and resizing copy experiment

For 32 actual validation photos and their full-resolution eroded water masks,
a Pillow composite implementation and the existing NumPy masking path
produced pixel-identical 224 × 224 outputs. In an alternating-order isolated
measurement over 31 warmed pairs, the existing function took 77.6 ms median /
114.0 ms p95 and the composite candidate took 70.1 ms / 84.6 ms; median paired
saving was 13.5 ms. This measurement excludes decode, segmentation and erosion,
and is not a complete pipeline benchmark. Inputs with non-RGB modes, EXIF and
unusual geometry need further checking. Script and log:
`/tmp/wave_mask_composite.py`, `/tmp/wave_mask_composite.log`.

### Combined exact-output candidates

A temporary combined candidate uses in-place EXIF transpose, float16 GPU
binary-mask erosion and Pillow composite standardization with the existing
448 / 384 FP16 segmentation and v16 checkpoint. Across 32 validation photos,
eroded masks, final encoded inputs and predictions matched the existing path
exactly. In a separate alternating-order timing run, 31 warm pairs yielded
375.8 ms median / 409.9 ms p95 for the existing path and 341.8 ms / 358.0 ms
for the combined candidate; median paired saving was 44.4 ms. Timing starts at
input decode and ends after regression; model startup, watcher notification,
preview and record writes are excluded. This is not yet the requested total
latency, and the full-resolution inputs and both segmentation passes remain
unchanged. Scripts and logs: `/tmp/wave_combined_check.py`,
`/tmp/wave_combined_check.log`, `/tmp/wave_combined_speed.py`, and
`/tmp/wave_combined_speed.log`.

### HEIC decoder thread count

`pillow_heif.options.DECODE_THREADS` defaults to 4 in this environment. In a
first sweep of eight HEIC validation photos (three interleaved decodes per
setting), the 4-thread and 16-thread medians were 100.2 and 78.9 ms. A wider
31-HEIC comparison using three interleaved decodes per photo found medians of
94.5 ms for 4 threads and 77.4 ms for 16 threads. Every decoded RGB array
matched the default result byte for byte. This measures only decoding, not
end-to-end latency or behavior under concurrent photos. Sixteen decode threads
may contend with the four PyTorch CPU threads and increase CPU load; those
costs must be measured before use. The one JPEG in the validation set is not
covered by this setting. Scripts: `/tmp/wave_heif_threads.py` and
`/tmp/wave_heif_threads_full.py`.

Adding 16 HEIC decode threads to the combined full-resolution candidate did
not show a material additional end-to-end gain in one alternating-order run:
31 warm photos measured 340.8 ms median / 365.1 ms p95 from decode through
regression, compared with the earlier combined candidate's 341.8 ms /
358.0 ms in a separate run. The paired reference in the new run, using four
HEIC threads and the existing preprocessing, was 385.4 ms median /
418.4 ms p95. Cross-run differences mean these numbers should not be treated
as a controlled comparison of the two candidate thread counts. Script/log:
`/tmp/wave_combined_threads_speed.py` and
`/tmp/wave_combined_threads_speed.log`. More decoder threads also cost CPU
resources, so this setting is not selected on present evidence.

### Direct full-resolution HEIC decode

Using `pillow_heif.open_heif(...).to_pillow()` with in-place EXIF transpose and
RGB conversion produced pixel-identical arrays to the shared Pillow loader on
all 31 HEIC validation photos. Three interleaved runs per photo gave median
76.1 ms for the current loader and 69.9 ms for direct decode, with a median
paired saving of 6.0 ms. This is an isolated decode result, comparable in size
to the earlier in-place EXIF transpose finding; the two savings must not be
added. The approach also needs non-HEIC fallback and broader orientation
coverage. Script: `/tmp/wave_heif_direct.py`.

### Segmentation processor copy reduction

The Transformers SegFormer processor converts the entire source image to a
NumPy array before its resize. Resizing the PIL image first to the processor's
configured size, then calling the same processor with `do_resize=False`,
produced identical input tensors on all 32 validation photos at 448 pixels.
Three interleaved processor-only calls per photo measured 46.8 ms median for
the existing path and 29.7 ms for the PIL-first path, with 17.7 ms median paired
saving. The same interpolation filter and normalization were retained.

A separate paired full-path run applied this only inside the processor during
both segmentation passes. All 32 final encoded inputs and regression
predictions matched exactly. On 31 warm pairs, decode through prediction took
389.6 ms median for the existing path and 374.4 ms for the candidate; median
paired saving was 20.7 ms. Some timings in this run were unusually high (p95
719 ms and 635 ms respectively), so sustained-load and repeated-run behavior
need investigation. No watcher or startup time was included. Scripts and logs:
`/tmp/wave_processor_resize.py`, `/tmp/wave_processor_full.py`,
`/tmp/wave_processor_full.log`.

### Combined processor, decode, erosion and compositing candidates

A temporary candidate combined in-place EXIF transpose, PIL resize before the
segmentation processor, float16 GPU binary erosion, and Pillow composite
standardization. It retained the same full-resolution decoded source, 448 /
384 segmentation settings, both segmentation passes, JPEG round trips,
regression checkpoint and model precision. Across 32 validation photos, all
eroded masks, encoded final inputs and predictions matched the existing path
exactly in this environment. A separate alternating-order timing run over 31
warm pairs measured 377.9 ms median / 405.8 ms p95 for the reference and
313.2 ms / 340.7 ms for the combined candidate. Median paired saving was
65.6 ms. Timing included decode through regression but excluded watcher
arrival, output writes and process startup. The candidate remains more than
three times the requested 100 ms budget even before those excluded costs.
Scripts and logs: `/tmp/wave_all_check.py`, `/tmp/wave_all_check.log`,
`/tmp/wave_all_speed.py`, `/tmp/wave_all_speed.log`.

### Channels-last segmentation probe

On one 448-pixel FP16 segmentation photo, converting the existing SegFormer
model to PyTorch channels-last memory format preserved the source-resolution
class map exactly. Five warmed full-segmentation calls before conversion had
105.4 ms median; five after conversion had 101.4 ms median. This single-photo,
sequential test does not prove a stable speedup or quality preservation over the
dataset, and its approximate 4 ms change is too small to bridge the remaining
budget. Script: `/tmp/wave_channels_last.py`.

### Architectural latency limit on this hardware

The current pipeline processes one full-resolution photo sequentially. On the
AMD Radeon 8060S, a single warmed 448-pixel Step 1 SegFormer call took about
101–105 ms in the channels-last probe above; the 31-photo full HEIC decode
median was about 70–80 ms even with copy reduction. The pipeline also needs
Step 2, mask and crop work, regression and record writing. These measurements
are not a formal lower bound for every implementation, but show that copy
removal and memory layout changes alone cannot plausibly deliver a complete
sub-100 ms latency with the same two-pass SegFormer on this device. More
threads or batching could improve throughput but do not establish a shorter
per-photo arrival-to-record latency.

A materially different inference engine, much faster hardware, or a new
segmentation model would be required to investigate the target further. A
model change must meet or improve the existing quality on a frozen evaluation
set, including water-mask behavior and final regression error, before it can
satisfy the no-quality-loss requirement. No such candidate has yet met that
requirement. The experimental thumbnail route was fast but had worse MAE.

### ONNX Runtime MIGraphX provider probe

The current SegFormer checkpoint was exported to fixed-shape ONNX at
`/tmp/wave-segformer-448.onnx` using an isolated Python 3.14 export environment.
An isolated Python 3.12 environment contains ROCm 10.1 MIGraphX 2.18 libraries,
ONNX Runtime 1.29, and the MIGraphX EP registration package. The EP library
registers and appears in `ort.get_available_providers()`, but sessions for both
a small probe graph and the exported SegFormer report only
`CPUExecutionProvider`; the diagnostic log for the SegFormer assigned all 715
nodes to CPU. Therefore no GPU latency or quality comparison is available yet.

The first provider attempts used only `migraphx-libs`, which supplies runtime
libraries but not the `migraphx` Python API. Installing the matching
`migraphx==2.18.0+rocm10.1.0` wheel from AMD's MIGraphX wheel index fixed the
import. The ROCm runtime's library directory also had to be added to
`LD_LIBRARY_PATH` for the standalone MIGraphX driver to start. With that path,
the driver's GPU GEMM smoke test passed on this host.

Registering the provider and passing its name in `providers` still produced a
CPU-only session. The ONNX Runtime 1.29 plugin API exposes the MIGraphX GPU via
`get_ep_devices()`; passing that device explicitly using
`SessionOptions.add_provider_for_devices([device], {})` creates sessions whose
provider list includes `MIGraphXExecutionProvider`. This device-based session
path successfully initializes for both a small ONNX probe and the 448-input
SegFormer. The `migraphx-driver perf` run for SegFormer, using FP16
quantization, compiled in 65.8 seconds and then measured 10.68 ms median GPU
model execution (12.05 ms p99; random inputs, model-only, excluding compilation
and all pre/post-processing). This is a promising engine-level result, not yet
a valid prediction-speed or quality result. Output parity on real validation
photos, its effects on the masks and final regression values, persistent
compiled startup cost, CPU/GPU transfers, and total arrival-to-record latency
are still unmeasured. No production code has changed.

### MIGraphX FP16 validation attempt

Two fixed-shape ONNX exports were tested at 448 and 384 pixels to match the
fast-v1 Step 1 and Step 2 preprocessing resolutions. ORT 1.29 sessions were
created through the explicitly selected MIGraphX GPU device with
`migraphx_fp16_enable=1`. PyTorch ROCm and ORT/MIGraphX could not safely coexist
in one process because LLVM aborted on a duplicate `spirv-expand-step` command
line option, so the validation used separate local processes and a localhost
socket bridge for MIGraphX logits. The bridge is for output comparison only;
its latency is not a production benchmark.

Across the 32-photo benchmark set, only 7 final Step 2 JPEGs matched exactly
between the existing PyTorch FP16 path and MIGraphX FP16. Predictions matched
for those same 7; across all 32, mean absolute prediction change was 0.00433,
with a maximum change of 0.05672. Warm decode-through-regression median was
about 388 ms for PyTorch and 346 ms for the MIGraphX candidate, but the first
MIGraphX request took 64.6 seconds due to lazy compilation. This run proves
neither unchanged quality nor sub-100 ms total latency, so this FP16 candidate
is rejected under the no-quality-hit requirement. Next, test MIGraphX FP32
against the canonical FP32 route before deciding whether the engine is useful.

### MIGraphX FP32 validation against the canonical FP32 route

The same 32-photo Step 1 → JPEG → Step 2 → regression validation was repeated
with MIGraphX FP32 and the repository's canonical PyTorch FP32 setting. Results
were exact for all 32: final Step 2 JPEG bytes and final regression values
matched. This removes the quality regression observed with FP16 MIGraphX for
this validation set. Warm median decode-through-regression time was 480.6 ms
for PyTorch FP32 and 377.5 ms for MIGraphX FP32 (about 103 ms, or 21.5%, lower).
The first MIGraphX inference again took about 66.4 seconds due to lazy engine
compilation. The candidate used two local processes and a socket bridge because
PyTorch ROCm and MIGraphX cannot share this process's LLVM runtime; therefore
377.5 ms is a correctness-test timing, not a clean integrated production
benchmark. It still misses the 100 ms target by a wide margin.

The FP32 candidate preserves the canonical FP32 output on the tested photos,
but has not yet been compared against benchmark ground-truth labels as a
separate statistical quality study. The cold first request again spent about
66 seconds compiling. The quality client ran PyTorch and ORT in separate
processes, so its measurements are not the final integrated serving numbers.

The existing fast-v1 benchmark's phase timings help explain the remaining
latency: HEIC decode is around 90–100 ms; first-pass segmentation about
100–106 ms; Step 1 mask standardization about 90–130 ms; second-pass
segmentation around 30–36 ms; Step 2 crop selection about 20–25 ms; and the
regression model about 4–6 ms. Individual photos fluctuate substantially.
Those measurements show the work left after inference acceleration is mostly
image decode and full-resolution CPU preprocessing. Even reducing the two
segmentation calls to about 20 ms combined cannot by itself reach a 100 ms
whole-prediction budget. Next measure the full warmed pipeline by stage and
combine the exact CPU memory/copy reductions with the FP32 inference engine.
The engine also needs a supported way to persist or load its compiled model so
the first-request compilation is not charged to a prediction.

### Compiled-model caching and the integrated hardware inventory

A probe of the plugin EP's compiled-model save options was rejected by the
installed MIGraphX plugin as an unknown provider option. The 66-second lazy
compile therefore remains a real first-use constraint for this particular
ORT plugin setup; the standard MIGraphX driver can separately parse/compile
models, but a supported hand-off into the plugin has not been established.
Do not count the 10 ms GPU kernel report as a cold end-to-end result.

The host also exposes an AMD XDNA NPU: `/dev/accel/accel0` uses `amdxdna`, and
`xrt-smi examine` reports RyzenAI-npu5, XRT 2.21.75, and firmware 1.1.2.65.
This is a potentially useful inference accelerator, but the installed machine
runs Ubuntu 26.04.1. AMD's current Ryzen AI Linux installation guide documents
Ubuntu 24.04 packages and an XRT 2.25.37 / amdxdna 2.25.260102.56 stack, so the
supported deployment environment does not match this host. No NPU runtime or
model was installed or benchmarked. Treat an NPU path as a separate future
experiment once a compatible runtime is available, and require exact output
and quality validation before adopting reduced precision.

#### Updated XDNA feasibility research (2026-10-28)

The current [Ryzen AI Linux 1.8 installation guide](https://ryzenai.docs.amd.com/en/latest/linux.html)
still distributes its tested XRT and XDNA plugin packages for Ubuntu 24.04,
and lists NPU CNN execution in INT8 or BF16. The upstream
[AMD XDNA driver repository](https://github.com/amd/xdna-driver) now describes
Ubuntu >=22.04 and kernel >=6.10 as build requirements and documents building
the driver/XRT plugin from source. This host is Ubuntu 26.04.1 with kernel
7.0.0-38-generic and an enumerated RyzenAI-npu5, but its installed XRT remains
2.21.75. Thus source-build experimentation may be technically possible, while
the vendor's packaged Ryzen AI Linux environment is still not a matching
supported install. I have not built or installed a new driver/runtime.

The NPU remains a low-priority path for the <100 ms complete-request target:
the full-resolution YOLO route spends substantial time in HEIC decode and CPU
standardization, so accelerating only the segmentation forward pass cannot be
counted as a complete solution. Any eventual NPU candidate must be benchmarked
end to end, include cold compilation/startup separately, and pass the same
mask, crop, per-photo waviness, and failure-coverage quality gate. Quantized
INT8/BF16 output is not presumed quality-equivalent.

### Exact preprocessing micro-optimizations on the quality reference profile

To separate low-level changes from the fast-v1 quality regression, I applied
only the copy/CPU transformations (in-place EXIF orientation, GPU fp16 binary
erosion, PIL composite masking, and PIL-first processor resizing) to the
canonical `baseline-v16` profile: 768 px, overlapping tiles, FP32. On the first
8 photos from the 32-photo validation set, every final Step 2 JPEG and every
regression output matched the unmodified route exactly (8/8). Median total
preprocess-through-regression time was 10.171 s reference and 10.105 s
candidate, only 64 ms lower (0.6%). I stopped the remaining 24 comparisons
because this tiled baseline takes about 10 seconds per photo in this run; the
sample does not support a dataset-wide performance claim. It confirms these
copy reductions preserve the reference outputs but are far too small to meet
the 100 ms goal. The bottleneck is the repeated high-resolution segmentation,
not object copying.

### Smaller pretrained segmentation model baseline

I downloaded the public `nvidia/segformer-b0-finetuned-ade-512-512` checkpoint
into `/tmp` with the HF CLI and ran the same 448/384, no-tile FP32 prediction
route on the 32-photo validation set. With the existing regression checkpoint,
validation MAE was 0.17232 versus 0.15674 for the saved reference (worse by
0.01558). Warm decode-through-regression median was 346.7 ms, with p95 about
377.6 ms. This pretrained B0 substitution is faster than the canonical tiled
path, but does not meet the quality requirement as-is. It remains a candidate
for training a new regressor on B0-preprocessed training images, with the
validation split held out; no validation images will be used for training.

### B0-specific regression head retraining (held-out quality experiment)

To test whether the faster pretrained SegFormer-B0 distribution can retain
waviness quality, I regenerated the existing train/validation Step 2 inputs
using B0 at 448/384, FP32, with no overlapping tiles. Of 264 train+validation
records, 260 succeeded; four training records failed Step 2 water detection.
I excluded those four from training rather than substituting another image.
The split remained unchanged: 213 B0-generated training inputs and all 47
validation inputs were used. The EfficientNet-B0 backbone stayed frozen and
only the regression head was trained with the repository's SmoothL1/AdamW
setup, fixed seed 42, and validation-MAE early stopping. Best was epoch 33
(44 epochs trained).

The new B0-specific regressor reduced MAE on validation from 0.14597 to
0.11988 and RMSE from 0.18700 to 0.15115. It was then evaluated once on the
untouched 46-record test split: all 46 B0 preprocessing runs succeeded, and
MAE was 0.11318 versus 0.14349 for the existing checkpoint on saved reference
inputs; RMSE was 0.13900 versus 0.18029. This is evidence that a compact
segmentation model plus a regression head trained for its own inputs can
improve held-out regression quality rather than merely matching the larger
model's crops. It is not a final candidate yet: B0 preprocessing failed on
four training photos, the test set is small, and validation/test results do
not establish broad robustness across future photos or conditions.

On test photos, B0 preprocessing median was 344.6 ms and p95 was 365.6 ms,
excluding regression and watcher/output-write overhead. The existing B0
full-path run with the old regressor measured 346.7 ms median. The new head's
forward pass is still only a few milliseconds, so this route remains roughly
3.5 times over the 100 ms target.

### B0 first-mask reuse

I also evaluated the existing `reuse_step1_mask` path with B0 and its new
regressor. It removed Step 2 segmentation but did not bring total latency near
100 ms: successful-photo medians were 336.3 ms on 46 validation photos and
343.4 ms on all 46 test photos. One validation photo failed lower-water crop
selection. On the 46 successful validation photos MAE was 0.12493 versus
0.14902 for the corresponding reference rows; test MAE was 0.13112 versus
0.14349, with no test failures. The small latency reduction and one failure
make this path a poor candidate to adopt without fixing and re-evaluating its
failure behavior.

All generated images, training outputs, and scripts for this experiment are
under `/tmp`; repository data and production pipeline code were not changed.

### CPU thread-count tuning on B0 (historical 384-pixel Step 2 profile)

The initial 16-photo thread sweep used Step 2 at 384 pixels, not the 448-pixel
setting used by the B0 head's quality evaluation. It produced identical final
JPEGs and predictions across thread counts 1, 2, 4, 8, and 16, with medians
365.7, 358.0, 352.5, 348.8, and 352.4 ms. It is exploratory timing evidence
only, not the trained input contract's latency result.

### GPU-standardized B0 regression head (held-out quality follow-up)

The GPU resize/standardization candidate was retrained with its own generated
inputs rather than reusing a regressor trained on PIL-resized images. The
training split excluded five photos for which GPU-based preprocessing failed;
212 training inputs and all 47 validation inputs were used, with the validation
split held out for early stopping. The best checkpoint was epoch 16 (27 epochs
trained).

On validation, MAE/RMSE changed from 0.14597/0.18700 for the saved reference
path to 0.13222/0.16590 for the GPU-standardized candidate. On the untouched
46-photo test split, MAE/RMSE changed from 0.14349/0.18029 to
0.12763/0.15911. All 46 test images preprocessed. This candidate therefore
improved regression quality on both held-out splits in this experiment, while
reducing the measured B0 full-path median on a 16-photo timing sample from
339.2 ms to 283.4 ms. The speed sample and metric sample differ; the speed
measurement is not a total production latency measurement, and the candidate
remains well above the 100 ms target. Five training-photo failures and the
approximate GPU interpolation still need resolution and broader validation.

The quality change is model-specific: the GPU standardizer shifted predictions
by a mean absolute 0.0190 when paired with the PIL-trained B0 regressor on a
16-photo comparison. Retraining on the GPU-generated input distribution
improved the held-out scores above, so do not combine this image path with the
PIL-trained head. The generated samples and candidate checkpoint remain under
`/tmp`.

### Ultralytics YOLOE-26n text-prompted segmentation trial

Research found that pretrained YOLO26 instance-segmentation weights use COCO
classes and do not include a generic water class. The YOLO26 semantic variant
uses Cityscapes categories, also without water. Ultralytics' open-vocabulary
YOLOE-26-seg variant supports text-prompted segmentation, so I tried
`yoloe-26n-seg.pt` with the single prompt `water` instead. References:
[YOLO26 model documentation](https://docs.ultralytics.com/models/yolo26),
[Ultralytics segmentation task documentation](https://docs.ultralytics.com/tasks/segment),
and [YOLOE documentation](https://docs.ultralytics.com/models/yoloe).

On the Radeon 8060S GPU with ROCm, Ultralytics 8.4.191, FP32, 640-pixel
inference, and default confidence threshold, warm model inference was 22.8 ms
median / 30.0 ms p95 across 46 held-out test photos. It returned at least one
water mask on only 22 of 46 photos (47.8% coverage); the other 24 had no mask.
This is segmentation inference only, excluding source decoding, mask-to-Step2
conversion, regression, and file/watcher overhead. The model is fast enough to
be interesting, but the detection coverage is not adequate for the pipeline.
Because masks are absent for over half the test set, I did not calculate an
MAE on only the detected subset or treat this as a quality-preserving result.
Some returned masks also cover a large fraction of the image, which needs
visual/ground-truth mask review before any crop-based use.

YOLOE setup used a separate temporary environment and downloaded the
242 MB MobileCLIP2 text encoder under `/tmp/yolo-exp`; no project dependencies,
production code, or image artifacts were changed. Trial script, JSON, and log
are `/tmp/wave_yolo26_trial.py`, `/tmp/wave_yolo26_trial.json`, and
`/tmp/wave_yolo26_trial.log`. The model weights and environment are temporary.

## Runtime and memory principles applied to the current CV pipeline

I reviewed the current official runtime guidance while selecting experiments.
PyTorch advises avoiding unnecessary CPU/accelerator synchronization and
profiling before choosing an optimization; AMD HIP similarly recommends
minimizing host/device transfers and intermediate copies, and batching small
transfers where that reduces fixed overhead. These are hypotheses to measure
on this Radeon APU, not guarantees that batching or another memory format will
improve per-photo latency. References: [PyTorch tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html),
[PyTorch channels-last guide](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html),
and [AMD HIP performance guidelines](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html).

The next experiments continue to use the same decoded full-resolution source,
canonical preprocessing profile, precision, crop rules, JPEG contract, and
regression checkpoint unless a candidate explicitly retrains and validates a
new input contract. A changed model input is not called quality preserving just
because it is faster. Comparisons should report decode-to-final-result latency,
startup and arrival-to-record latency separately, mask/crop behavior, held-out
regression metrics, and CPU/accelerator memory. Throughput batching is not a
substitute for single-photo latency.

### Independent tile batching and memory study

On one 3024 x 4032 held-out validation photo using canonical `baseline-v16`
(768-pixel tiles, 25% overlap, FP32), the existing sequential tiled segmenter
completed Step 1 in 10,992 ms. It used 35 tiles, a 6.81 GiB float32 full-image
logit accumulator on the Radeon 8060S, and peaked at 8.91 GB of PyTorch
accelerator allocation. Process peak RSS was 2.42 GB. This is one photo and one
run, but it makes clear that the canonical tiled segmentation work is the
main latency and memory cost, not the final regression forward pass.

I tested microbatching two independent tiles in a single model call while
keeping tile locations, processor, FP32 logits, overlap averaging, and class
argmax otherwise unchanged. With the accumulator on the GPU, it took 32,452 ms
on the same photo and peaked at 10.90 GB of PyTorch accelerator allocation.
The final class map changed at two pixels, including one water-mask pixel. With
the accumulator on CPU, it took 36,524 ms, peaked at 3.59 GB accelerator
allocation and 9.84 GB process RSS, and likewise changed two class-map pixels
(one water pixel). Neither is a speed improvement or exact mask match, so
microbatching is rejected for this implementation and hardware. This result
also illustrates that reducing accelerator allocation by moving the large
accumulator to host memory increases host usage and transfer costs.

These timings are single-photo probes, not stable benchmark estimates or full
prediction timings. The tests did not run the remaining Step 1/Step 2
standardization, regression, file output, or arrival watcher. Scripts and
outputs: `/tmp/wave_tile_baseline_save.py`,
`/tmp/wave_tile_baseline_save.json`, `/tmp/wave_tile_batched_gpu.py`,
`/tmp/wave_tile_batched_gpu.json`, and `/tmp/wave_tile_batched_cpu.py`,
`/tmp/wave_tile_batched_cpu.json`.

### Lower-resolution global logit accumulation

On the same 3024 x 4032 photo, I tested accumulating each tile's native 192 x
192 logits on a quarter-resolution global grid, followed by one chunked
full-image interpolation/argmax. This reduces the accumulator from 6.81 GiB to
about 0.43 GiB and peak PyTorch accelerator allocation from 8.91 GB to 3.83
GB. Step 1 time was 10,501 ms versus 10,992 ms for the existing implementation
(about 4.5% in this single run). The candidate changed 7,708 water-mask pixels,
so it is not mask-equivalent; no final regression quality claim is made. The
small timing difference does not justify this quality change, and this variant
is rejected. Script and output: `/tmp/wave_tile_lowres.py` and
`/tmp/wave_tile_lowres.json`.

### FP16 precision-only trial on the canonical profile

To isolate precision from the earlier fast-profile experiment, I held the
canonical `baseline-v16` settings fixed: full-resolution decoded sources,
768-pixel tiles, 25% overlap, the same model and tile positions, and the same
regression checkpoint. Only SegFormer inference precision changed from FP32 to
FP16. On one 3024 x 4032 validation photo, Step 1 fell from 10,992 ms (FP32
single-run probe) to 6,170 ms (FP16 single-run probe), about 44% lower. The
FP16 water mask changed at 3,030 pixels on that photo. The first comparison used the active Step 2 directory, whose marker says
`fast-v1`; that is not the original input contract used to train the v16
checkpoint. I corrected the quality comparison against the archived v16-era
Step 2 inputs. Those archived inputs exist for 32 of the 47 current validation
rows. On those 32, FP16 MAE was 0.15557 versus 0.15674 for FP32, while RMSE
was 0.20212 versus 0.20032. Across the 43 rows where both full FP32 and FP16
preprocessing succeeded (32 archived rows plus 11 newly run FP32 rows), MAE
was 0.14610 versus 0.14756, while RMSE was 0.19014 versus 0.18905. FP32
failed four rows in the full current validation set and FP16 failed three;
the three FP16 failures were also FP32 failures. This is mixed evidence, not
proof of zero quality impact: RMSE is slightly worse, the input masks and
final JPEGs differ, and the benchmark is far above 100 ms. I am not adopting
this candidate. Its median was 5,749 ms / p95 5,876 ms on 44 successful rows;
one same-photo full-path comparison measured 6,663 ms for FP16 and 11,442 ms
for FP32. Candidate timings exclude model startup and watcher/output overhead.
Script/log/result:
`/tmp/wave_tile_fp16.py`, `/tmp/wave_tile_fp16.json`,
`/tmp/wave_fp16_validation.py`, `/tmp/wave_fp16_validation.log`, and
`/tmp/wave_fp16_validation.json`.


## Correction: use the archived v16 inputs as the quality reference

The active `step-1-processed-data/` and `step-2-final-water-data/` directories
are marked `fast-v1`, while the v16 checkpoint's training manifest predates
that profile. Earlier B0 notes that compared the v16 checkpoint on the active
Step 2 directory to B0-generated inputs did not use the v16 checkpoint's
original input contract. Those comparisons are retained as historical
measurements but should not be read as quality comparisons to the original
v16 reference. The authoritative saved v16-era examples are under
`step-2-final-water-data/archive/20261027T134224Z/`; they cover 32 current
validation rows and 32 current test rows.

I recalculated the B0-specific head predictions against those archived images
on the common rows. Results were:

| Split and common rows | Original v16 on archived inputs | B0 head, PIL standardizer | B0 head, GPU standardizer |
| --- | ---: | ---: | ---: |
| Validation MAE (32) | 0.15674 | 0.13723 | 0.14207 |
| Validation RMSE (32) | 0.20032 | 0.16930 | 0.17676 |
| Test MAE (32) | 0.13284 | 0.10532 | 0.12826 |
| Test RMSE (32) | 0.16162 | 0.12857 | 0.15862 |

This corrected comparison supports improved regression quality for both B0
heads on the archived subset; the GPU-standardized B0 test improvement is
small. The previously reported 283 ms GPU-standardizer median is withdrawn as
a contract-matched measurement: it used Step 2 at 384 pixels, while this B0
head and its quality-evaluation inputs use the default 448-pixel Step 2 size.
The prior 345 ms PIL-path figure is also not a full end-to-end measurement.
The GPU path had five training-photo preprocessing failures. These results are
evidence for further optimization of the B0-specific pipeline, not adoption
under the 100 ms goal. The recalculation script and rows are
`/tmp/wave_b0_archive_comparison.py` and
`/tmp/wave_b0_archive_comparison.json`.

### B0 thumbnail / 320-pixel regression-matched trial

I trained a separate EfficientNet-B0 regression head for a coarse, low-resolution
preprocessing path: use the embedded 320 x 240 HEIC thumbnail (or a reduced
JPEG decode), run B0 at 320 pixels with FP32 and no overlapping tiles, run
Step 2 at 224 pixels, and train only on those exact generated inputs. All 47 validation
photos preprocessed; six of 217 training photos failed crop selection and were
excluded. The frozen-backbone head trained for 50 epochs with seed 42 and
validation-MAE checkpoint selection. On 43 validation rows where the original
FP32 reference succeeded, candidate MAE/RMSE were 0.14533/0.18243 versus
0.14756/0.18905 for the original pipeline. The candidate also succeeded on
three rows where the old FP32 path failed, while three other crop failures were
shared. This validation result looked promising, so I evaluated the untouched
test split once.

The candidate processed all 46 test photos. With per-photo model-input JPEG
and JSON result writes included, warm median latency was 50.4 ms and p95
61.5 ms (50.3/61.5 ms after the first photo). The model weights took 0.48 s
to load after imports; the first photo took 792 ms because it included cold
runtime/kernel setup. The 46 inputs decoded to 320 x 240 thumbnails. Peak
PyTorch accelerator allocation was 178 MB and process peak RSS was 2.29 GB.

On the 32 test photos with archived v16 reference inputs, the thumbnail
candidate had MAE 0.14069 and RMSE 0.19241, versus 0.13284 and 0.16162 for the
original checkpoint on its archived inputs. This is a material test quality
regression despite meeting warm per-photo latency, so I reject it. Test results
were not used to tune this checkpoint. This experiment shows the speed target
is reachable for this reduced input path, but not with preserved quality on the
current held-out evidence. Script, trained checkpoint and reports remain in
`/tmp/wave_b0_thumb320_*`.

### B0 GPU candidate stage timing

A warmed one-photo breakdown measured 300 ms decode-through-regression: 87 ms
decode, 162 ms Step 1 segmentation and standardization (including a 40 ms GPU
standardization operation), 39 ms Step 2, 4 ms regression, plus JPEG round
trips. This sample used Step 2 at 384 pixels, so it is not a timing breakdown
for the head's 448-pixel trained input contract and must not be combined with
the corrected 448-pixel measurements. It used 8 PyTorch threads, peaked at
388 MB PyTorch accelerator allocation and 2.68 GB process RSS, and consumed
0.58 CPU-seconds during the timed prediction. It still indicates that decode
and preprocessing dominate over regression. Script/log:
`/tmp/wave_b0_gpu_stage_timing.py` and `/tmp/wave_b0_gpu_stage_timing.log`.

### Exact GPU erosion at the trained 448-pixel Step 2 setting

I replaced the CPU minimum-filter erosion operation experimentally with two
GPU max-pool operations over the inverted mask. On the 93 validation and test
photos, the resulting encoded Step 2 JPEGs were byte-for-byte identical to
the CPU erosion outputs. Thus the model inputs and the regression predictions
were unchanged in this comparison.

A paired end-to-end check then used the GPU-standardized B0 path with Step 2
at 448 pixels, the setting used to evaluate the head. It included decode,
preprocessing, regression, model-input JPEG writes and per-photo JSON output
writes for 16 test photos. The CPU-erosion path had 298.9 ms median and
314.7 ms p95; GPU erosion had 270.7 ms median and 288.9 ms p95. Median paired
saving was 28.0 ms (about 9.4%). All 16 encoded model inputs were identical,
and maximum prediction difference was zero. This is a real speed improvement
without an observed quality change, but latency remains well above the 100 ms
target. This end-to-end sample is small; widen timing coverage before relying
on the latency percentile. Script and results:
`/tmp/wave_b0_gpu_erosion_e2e.py`, `/tmp/wave_b0_gpu_erosion_e2e.log`, and
`/tmp/wave_b0_gpu_erosion_e2e.json`. The 93-photo exact-output check is in
`/tmp/wave_b0_gpu_erosion_448_check.py` and
`/tmp/wave_b0_gpu_erosion_full_check.json`.


### CPU thread sweep at the trained 448-pixel Step 2 setting

I repeated the thread sweep with the GPU-standardized B0 head, Step 2 at its
trained 448-pixel setting, and GPU erosion. The run included full-resolution
source decode, both preprocessing stages, regression, JPEG output and a JSONL
result write. Across 16 test photos, all six thread settings (1, 2, 4, 8, 16,
32) produced byte-identical model-input JPEGs and identical regression values.
Median latencies were 295.8, 282.1, 279.8, 276.2, 276.4, and 421.3 ms,
respectively. Eight and 16 threads were effectively tied; 32 threads was about
52% slower than 8 threads, consistent with CPU contention and shared-memory
bandwidth pressure on this APU. The 4-thread p95 had an outlier (331.6 ms), so
this small sweep does not establish a reliable p95 improvement. No thread
setting approaches the 100 ms target.

Peak process RSS during the sweep was 3.35 GiB; peak PyTorch device allocation
was 400 MiB. These are process/run peaks, not per-image incremental usage, and
include imported libraries and model weights. I did not adopt a new global
thread default: 8 threads gave only a 3.6 ms median reduction versus 4, while
the earlier paired erosion benchmark varied by a similar amount. Script, log
and per-setting outputs are under `/tmp/wave_b0_contract_thread_sweep.py`,
`/tmp/wave_b0_contract_thread_sweep.log` and
`/tmp/wave_b0_contract_thread_sweep/`.

### Channels-last tensor layout on the Radeon ROCm path

Following PyTorch's memory-format tuning guidance, I tested the B0 segmenter
with convolution weights and image tensors in channels-last layout, against the
same NCHW model, at 448-pixel Step 2 with GPU erosion and the matched GPU
standardizer. Both branches used FP32 and processed the same 16 held-out test
photos. All 16 encoded model-input JPEGs and regression predictions were
identical. Median end-to-end latency was 270.9 ms for NCHW and 271.4 ms for
channels-last; the p95 values were 281.2 and 293.0 ms. There is no measured
speedup on this Radeon 8060S / ROCm stack, so I do not recommend changing the
layout. The paired sample is small and has some timing variance. Script, log
and outputs: `/tmp/wave_b0_channels_last.py`,
`/tmp/wave_b0_channels_last.log`, and `/tmp/wave_b0_channels_last/`.

### In-place EXIF orientation during full-resolution decode

I adopted Pillow's in-place EXIF transpose in `image_loading.py`, with the
existing copy-based behavior retained as a compatibility fallback for older
Pillow versions. This removes an extra full-resolution oriented-image copy
before RGB conversion. The isolated decode comparison produced byte-identical
RGB pixels for all 32 validation sources and a median paired decode saving of
6.27 ms. A contract-matched B0 end-to-end comparison on 16 test photos also
produced identical final JPEG model inputs and regression predictions for all
photos. Its baseline and in-place medians were 271.9 and 267.2 ms (3.7 ms
median paired saving); p95 was 279.6 and 283.8 ms, so the small sample does not
show a reliable tail-latency gain. Synthetic JPEGs with each EXIF orientation
1 through 8 also matched the previous copy-based decoder pixel-for-pixel.
This is a modest copy/memory optimization, not a path to the 100 ms goal.
The benchmark script, log and outputs are `/tmp/wave_decode_copy_check.py`,
`/tmp/wave_b0_decode_inplace_e2e.py`,
`/tmp/wave_b0_decode_inplace_e2e.log` and
`/tmp/wave_b0_decode_inplace_e2e/`.

### FP16 autocast for the B0 segmenter at the matched 448-pixel contract

I compared FP32 and FP16 autocast with the same B0 segmenter, GPU standardizer,
GPU erosion, 448-pixel Step 2, regression checkpoint and decode/output path on
all 93 validation and test photos. FP16 reduced median end-to-end latency from
269.2 to 261.3 ms (about 2.9%); p95 changed from 284.8 to 279.0 ms. Only 21
of 93 final model-input JPEGs matched exactly, and the largest paired
prediction difference was 0.1325, so it is not an output-preserving
optimization.

I evaluated the same FP32-trained B0 regression head on both sets of generated
inputs. Validation MAE/RMSE shifted from 0.13222/0.16590 (FP32) to
0.13009/0.16405 (FP16), while held-out test MAE/RMSE shifted from
0.12763/0.15911 to 0.12804/0.16067. The test regression is small but in the
wrong direction, and most images changed. Under the no-quality-hit condition,
FP16 is rejected; it is not enabled in the pipeline. Peak process RSS was
2.97 GiB and peak PyTorch allocation 438 MiB for the combined run, not a
per-mode memory comparison. Script, detailed outputs and quality report:
`/tmp/wave_b0_fp16_contract_check.py`, `/tmp/wave_b0_fp16_contract.log`,
`/tmp/wave_b0_fp16_contract/summary.json`,
`/tmp/wave_b0_fp16_quality.py` and `/tmp/wave_b0_fp16_quality.json`.

### Fixed-shape convolution autotuning on ROCm

I tested `torch.backends.cudnn.benchmark=True` (the PyTorch convolution
algorithm-search option, backed by the installed ROCm stack) versus its default
on the same 16-photo B0 / 448-pixel contract with FP32. Model inputs and
regression predictions were identical on all 16 photos. Default median and
p95 were 274.1 and 282.2 ms; autotuning gave 271.4 and 281.7 ms. This small
change is within run variation and is not enough to justify a new global
runtime setting; I did not adopt it. The test follows the hardware-specific
profiling approach in the [PyTorch performance tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html).
Script, log and outputs: `/tmp/wave_b0_miopen_benchmark.py`,
`/tmp/wave_b0_miopen_benchmark.log`, and `/tmp/wave_b0_miopen_benchmark/`.

### Keep full-resolution crop and mask data as uint8 until it reaches ROCm

I tested a memory-traffic optimization inside the experimental GPU
standardizer. The existing version converted the cropped RGB image and binary
mask to float32 before moving them to the Radeon. The candidate transfers both
as uint8 (one byte per channel / mask value), then casts RGB to float32 and
converts the mask to boolean on the device before applying the same FP32 mask,
antialiased bicubic resize, rounding, padding and JPEG encoding. This reduces
logical host-to-device payload for the crop and mask from 16 to 4 bytes per
pixel before device-side conversion.

Across all 93 validation and test photos, the candidate produced byte-identical
final model-input JPEGs and identical regression predictions. The paired
end-to-end medians were 273.9 ms for float32 host tensors and 264.2 ms for
uint8 transfer; p95 was 292.2 and 281.3 ms. Median paired saving was 9.95 ms.
Median process CPU time per prediction was 577.5 ms versus 475.5 ms, a 17.6%
reduction in CPU time (process CPU time can exceed wall time because worker
threads execute concurrently). Combined-run peak RSS was 2.67 GiB and peak
PyTorch allocation 394 MiB; these are not per-mode memory deltas. The 93-row
exact input comparison means there is no regression change on this sample.

This experiment uses the GPU-standardized B0 preprocessing contract and its
separately trained regression head. The checkpoint currently configured for
normal prediction is the B0 v16 head, which uses the canonical SegFormer-B2
preprocessing contract. Thus the measured uint8 optimization is validated for
the B0 candidate route, not yet for the configured canonical predictor; I have
not applied it to that path. A production adoption would first require a clean,
reproducible implementation with the matching checkpoint/profile contract.
Script, full report and per-image outputs are under
`/tmp/wave_b0_uint8_transfer_full.py`, `/tmp/wave_b0_uint8_transfer_full.log`
and `/tmp/wave_b0_uint8_transfer_full/`; the initial paired run is under
`/tmp/wave_b0_uint8_transfer.py` and `/tmp/wave_b0_uint8_transfer/`.

### Trained-contract stage breakdown

A warmed 16-photo test profile of the GPU-standardized B0 route (448-pixel
Step 2, FP32, GPU erosion, full decode and JPEG round trips) measured median
stage times of 81.5 ms decode, 138.5 ms Step 1 segmentation plus
standardization, 1.0 ms Step 1 JPEG round trip, 40.5 ms Step 2, 1.0 ms Step 2
JPEG round trip and 4.9 ms regression. End-to-end median/p95 were 271.2/281.4
ms. Peak RSS was 2.65 GiB and peak PyTorch allocation 400 MiB. Per-stage
numbers are rounded and do not add exactly to total. Step 1 remains the largest
stage, followed by decode; output encoding and regression are small. This
profile points the next work toward Step 1 compute and full-resolution pixel
handling. Script, log and per-photo output:
`/tmp/wave_b0_contract_stage_profile.py`,
`/tmp/wave_b0_contract_stage_profile.log`, and
`/tmp/wave_b0_contract_stage_profile.json`.

### 32-bit summed-area tables for lower-water crop search

Because Step 2 operates on 224 x 224 masks, all rectangle counts are at most
50,176. I compared int32 summed-area tables against the existing int64 tables
on the B0 candidate route while keeping the uint8-transfer standardizer fixed.
All 16 test model-input JPEGs and predictions were identical. Median end-to-end
latency was 256.0 ms with int64 and 254.7 ms with int32; median paired saving
was -2.1 ms (the candidate was slower for most pairs), and p95 changed from
266.5 to 269.2 ms. CPU process time changed only slightly. This is within
measurement noise, so I did not alter the crop-search implementation. The
experiment shows the table dtype is not a meaningful latency bottleneck.
Temporary script and output: `/tmp/wave_b0_int32_sat_isolated.py`,
`/tmp/wave_b0_int32_sat_isolated.log`, and
`/tmp/wave_b0_int32_sat_isolated/`.

### Canonical B2 CPU-thread sweep

I checked the configured `baseline-v16` predictor (SegFormer-B2, overlapping
768-pixel tiles, FP32) rather than extrapolating the B0 result. On one held-out
test photo, the official benchmark runner measured one warm full prediction at
10.80 s with 1 PyTorch thread, 11.40 s with 4, 11.75 s with 8, 11.69 s with
16, and 11.44 s with 32. The predicted value was identical (`0.8500191569`)
for all settings. For the 1-thread run, Step 1 took 10.41 s and its tile-model
forwards accounted for 9.00 s; tile preparation took 0.57 s. This single-photo
sweep is directional and cannot establish a robust default thread count, but
it confirms that CPU tuning cannot bridge the gap to 100 ms for the configured
canonical route: almost all its time is B2 segmentation. I did not change the
thread default. Per-setting official reports and logs are under
`step-7-prediction-benchmarks/active-thread-sweep/`.

### HEIC decoder threads on the configured canonical B2 predictor

The B0 candidate's 4-vs-16 decoder-thread result did not transfer to the
configured canonical B2 path. On the same HEIC test photo and checkpoint,
with the rest of `baseline-v16` unchanged, the full prediction took 10.412 s
with 4 decoder threads and 10.501 s with 16. The final model-input JPEG and
regression value were identical; CPU process time was 10.769 s versus 10.881 s.
Peak PyTorch allocation was 8.93 GB. This is one photo, so the 89 ms reversal
is directional only, but it gives no basis to raise the shared decoder thread
setting for canonical prediction. Decoder-thread tuning remains specific to
the B0 candidate and is not a production default. Script and result:
`/tmp/wave_active_heif_threads_b2.py` and
`/tmp/wave_active_heif_threads_b2.json`.

### Active B2 channels-last layout and fixed-shape autotuning

I repeated the memory-format and convolution-autotuning probes on the actual
`baseline-v16` B2 tiled route, because the earlier B0 result did not cover its
768-pixel repeated tile workload. On one held-out photo, channels-last produced
identical final model-input JPEGs and predictions in all three paired runs.
Median latency was 10.598 s for NCHW and 10.578 s for channels-last (0.2%
faster), while process CPU time rose from about 10.97 to 12.22 s. With a
10.6-second prediction and one photo, that wall-time difference is not
meaningful; I did not change memory format. Peak PyTorch allocation was 8.93
GB.

I then tested `torch.backends.cudnn.benchmark=True` against the default NCHW
path on the same photo. All three final JPEGs and predictions matched exactly.
Median latency was 10.109 s by default and 10.120 s with autotuning, with no
CPU-time improvement. The autotuning option is not adopted. Both experiments
show that these common kernel-layout/search settings do not address the B2
segmentation bottleneck on this ROCm APU. Scripts and reports:
`/tmp/wave_active_b2_channels_last.py`,
`/tmp/wave_active_b2_channels_last.json`,
`/tmp/wave_active_b2_miopen_autotune.py`, and
`/tmp/wave_active_b2_miopen_autotune.json`.

### Ultralytics YOLO26n ADE20K semantic model as a water preprocessing candidate

I researched the official Ultralytics [semantic segmentation task](https://docs.ultralytics.com/tasks/semantic/), [semantic dataset format](https://docs.ultralytics.com/datasets/semantic/) and [YOLO26 model family](https://docs.ultralytics.com/models/yolo26/). Their pretrained `yolo26n-sem-ade20k.pt` model exposes ADE20K classes including `water` (21), `sea` (26), `river` (60), and `lake` (128), so it can provide a water mask without a prompt. This is a different candidate from the earlier YOLOE open-vocabulary trial, which had poor water coverage.

I loaded this checkpoint in the isolated ROCm Ultralytics environment and ran it on four held-out raw HEIC photos from the current test split, using the union of those four class IDs. All four produced nonempty semantic masks; water-class area fractions were 41.5%, 71.9%, 79.6%, and 60.4%. These are plausibility checks only: there are no archived canonical B2 masks to compare against, and I have not compared the downstream crop/JPEG inputs or regression metrics. Therefore this trial does not establish quality preservation.

At 640 input size, warm inference latency across those four full-resolution photos ranged from 108 ms to 367 ms. On one held-out photo, reducing input size to 448, 320, or 256 did not materially lower the warm latency floor (about 111 ms); the predicted water fraction also changed from 41.5% at 640/448 to 35.0% at 320 and 31.0% at 256. The first call was much slower due to warm-up. These figures include Ultralytics preprocessing/inference/postprocessing but not image decoding or the rest of the wave pipeline. On this hardware this pretrained semantic model does not meet the complete pipeline's 100 ms target by itself, so it is not a speed-qualified replacement. A distilled/specialized smaller model or a faster inference runtime would need separate measurement, and quality must be measured against the existing held-out regression contract before adoption.

### Fixed-shape `torch.compile` probe on the canonical B2 tile

Following the official [PyTorch `torch.compile` reference](https://docs.pytorch.org/docs/stable/generated/torch.compile.html), I compiled the existing SegFormer-B2 inductor graph for one 768 x 768 tile on the Radeon ROCm setup (`dynamic=False`, default mode). Eager FP32 tile-forward warm times were 267–272 ms. The first compiled invocation took 22.95 s; subsequent invocations were 278–284 ms. Thus the compiled warm tile was about 3–6% slower in this small sequential probe, before charging compile startup. Compiled and eager tile logits differed slightly (maximum absolute difference 4.01e-5), but class argmax agreed at every tile output pixel. Since it is slower and introduces first-use compilation, this setting is rejected. The check covers one tile only, not a complete overlapping mask or held-out regression outputs.

### HEIC decoder thread count with byte-exact decoded pixels

The earlier whole-pipeline B2 timings hid decoder-only effects behind roughly ten seconds of segmentation per image. I isolated Pillow-Heif decode on 16 held-out HEIC files and hashed the resulting oriented RGB bytes for every setting. Pixel output was identical at all settings. Four decoder threads gave a 110.3 ms median / 123.0 ms p95; 8 threads gave 91.9 / 95.7 ms; 16 gave 88.0 / 92.8 ms. A follow-up sweep on the same 16 files measured 16/20/24/32/48 threads at medians 89.0/87.4/85.7/84.5/83.3 ms. Peak process RSS rose from 0.44 GiB at 16 threads to 0.96 GiB at 32 and 1.14 GiB at 48. The marginal 5.7 ms gain from 16 to 48 threads costs about 0.70 GiB additional peak RSS, so 16 is the more balanced candidate setting; I have not changed the project default because the configured B2 path did not show a corresponding end-to-end gain and the CPU/RAM tradeoff depends on the target route. References: [pillow-heif decode-thread options](https://pillow-heif.readthedocs.io/en/latest/options.html).

### YOLO26n binary water student distillation pilot

I used the official Ultralytics [semantic PNG-mask training format](https://docs.ultralytics.com/datasets/semantic/) and the pretrained YOLO26n ADE20K semantic checkpoint. From the project training split only, I selected 48 records with a fixed seed, generated canonical B2 water masks, and used 38 for student training and 10 for a small teacher-mask validation subset. A 29-epoch pilot reached 0.818 mean IoU on those 10 teacher-mask validation photos. This is a very small segmentation validation set and is not a waviness quality result. The teacher generator, dataset, and checkpoint artifacts are under `/tmp/wave_yolo26_student_pilot/`; held-out test photos were excluded from student training.

On four untouched test photos with saved B2 water masks, the student with full-resolution Ultralytics postprocessing achieved mask IoUs 0.873, 0.529, 0.917, and 0.853. Reusing the existing v18 fast-B0 regressor without retraining, the 46-photo full-resolution student path completed without failures but had test MAE/RMSE 0.1303/0.1705 and 254.5 ms median / 296.3 ms p95. The existing GPU-standardized B0 experiment measured 0.1276/0.1591 on this test split, so the off-contract student inputs plus unchanged head do not preserve its quality. This is not a valid deployment candidate.

I then bypassed expansion of the student class map to the full source dimensions. The student predicts on a 640-pixel long-side image; its low-resolution mask is mapped back to the source crop, and water masking is applied after resizing the RGB crop to model-input size. This avoids building and compositing a full 12 MP mask but changes edge pixels, so it requires the held-out waviness gate and retraining. With a second student pass on the standardized Step 1 image, 12/46 test images failed Step 2 crop selection. On the 34 successful rows, median was 149.7 ms and p95 160.5 ms; the success-only MAE/RMSE of 0.1292/0.1671 is biased by those failures and is not a quality result.

As a separate branch, I projected the first student mask into Step 1 coordinates and reused it for Step 2, avoiding a second segmentation pass. This allowed all 46 test photos to complete, but v18 test MAE/RMSE degraded to 0.1386/0.1821; median latency was 161.5 ms and p95 173.9 ms, with only one photo under 100 ms. Mask reuse therefore does not meet either the quality or latency requirement. All these candidate variants remain under `/tmp` and are not enabled in the pipeline.

### Exact low-resolution model input construction for YOLO26

For four held-out photos, I compared the network tensor built from the full-resolution input using Ultralytics preprocessing with a path that first resized RGB to the exact 640-pixel long-side dimensions, then performed channel order conversion on that compact image. All four tensors matched bit-for-bit; eager model logits also matched exactly, with zero changed class decisions. This confirms that the initial resize can be done before transfer/model preprocessing without changing the model input when the same OpenCV interpolation and dimensions are used. It does not by itself accelerate Ultralytics' existing PIL route, whose loader already uses OpenCV color conversion, and therefore is not counted as a production speedup.

### Crop resizing kernels in the low-resolution student prototype

While the B2 teacher job was running, I timed resizing a 4032 x 2747 RGB crop to 224 x 153 on one training image. Pillow Lanczos was 185.7 ms median in this contended run. OpenCV `INTER_AREA` was 57.3 ms and differed from the Pillow output by 3.45 mean absolute 8-bit levels. `INTER_LINEAR`, `INTER_CUBIC`, and `INTER_LANCZOS4` took 0.13, 0.06, and 6.0 ms, but their mean absolute pixel differences from Pillow Lanczos were 12.4–12.6 levels. Increasing OpenCV threads from 1 to 16 did not improve `INTER_AREA` (about 57–60 ms); `INTER_LANCZOS4` remained about 5–7 ms. These are single-crop timings while the GPU-heavy teacher job was active, not end-to-end numbers. The low-cost filters change many pixels, so none is adopted: a candidate would need a matching regression head and full untouched test-set quality evaluation. The direct linear/cubic resize is a useful candidate because it removes the expensive high-resolution antialiasing work, but its final waviness quality is unverified.

### Multistage Gaussian downsampling for the model crop

To test a standard computer-vision image-pyramid approach, I repeatedly applied OpenCV `pyrDown` (a Gaussian low-pass followed by 2:1 reduction) until the crop was close to the target, then resized the small remainder to 224 x 153. On the same 4032 x 2747 crop, this took 2.21 ms median / 2.39 ms p95 versus 132.8 ms median for Pillow Lanczos in the same run. Mean absolute pixel difference from the Pillow output was 8.32 levels, lower than direct OpenCV linear, cubic, or Lanczos4 in this sample (about 12.4–12.6), but higher than `INTER_AREA` (3.45, at about 53 ms). This multistage resize is a promising speed/quality tradeoff for the student contract; it changes input pixels and is not adopted until a matching regression head passes untouched validation/test quality and complete-latency checks. Timings are single-crop, collected while the B2 teacher job was active.

### Performance principles applied to this pipeline (2026-10-28)

I checked primary PyTorch and OpenCV guidance before selecting the next experiments. The fit-for-purpose principles are: profile the full call path before tuning; do not add host/device synchronization to timed hot paths; use inference mode; benchmark data transfer in the actual runtime rather than assuming pinned memory or asynchronous copies help; keep CPU arrays/tensors contiguous when required and avoid redundant materialization; reuse destination buffers where shapes repeat; and tune CPU thread counts on this hardware rather than assuming more workers are faster. PyTorch specifically cautions that pinning itself is a blocking copy and can increase RAM use, and that synchronization-prone operations can stall accelerator pipelines. OpenCV documents that image ROIs can share backing storage and that output buffers can be reused when dimensions/type match. These are hypotheses to measure on the project's ROCm APU, not portable speed claims.

Sources: [PyTorch performance tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide), [PyTorch host/device transfer guide](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html), [PyTorch CUDA synchronization guidance](https://docs.pytorch.org/docs/main/notes/cuda.html), and [OpenCV image operations and memory management](https://docs.opencv.org/4.x/d5/d98/tutorial_mat_operations.html).

The canonical tiled B2 route is dominated by 35 sequential tile forwards (about 9 seconds of its 10.8-second single-photo timing). Existing experiments found no useful speedup from CPU thread count, channels-last memory format, or convolution autotuning; outputs stayed equal, but those changes did not address the dominant cost. This makes tile batching the next exact-semantics hypothesis: run the same source tiles together in a small batch, then stitch the same per-tile logits with the same interpolation and accumulation order, while tracking peak allocated/reserved memory and comparing final masks and predictions. Batch size must be chosen based on actual free VRAM/RAM and cannot compromise the source-resolution overlap policy.

Avoid adding `pin_memory()` to individual tile tensors by default: each tile is small, the route has one image at a time, and PyTorch documents that pinning can add a host-side copy. First measure the existing `.to(device)` path against a contiguous stacked batch transfer. Similarly, skip eliminating JPEG encode/decode or changing interpolation until the regression head has been evaluated on exact held-out splits; those transformations change model inputs and therefore require a quality gate, even if pixel differences look small.

The objective remains strict: the complete warm prediction request, including decode, both preprocessing steps, regression, and prediction-record output, must be under 100 ms with held-out quality preserved. No experiment recorded so far demonstrates that target. The Gaussian pyramid's 2.2 ms resize result is only a microbenchmark, not an end-to-end or quality result.

### Batching the canonical B2 tiles

After the teacher-mask generation job released the accelerator, I compared the current sequential tiled segmentation with batches of 1, 2, and 4 tiles on one untouched test photo (12,192,768 output pixels). The same 35 source tiles, 25% overlap, bilinear logit upsampling, float32 accumulation, and final argmax were used. The canonical sequential route took 12.72 s in this run; the equivalent custom batch-1 route took 11.17 s and produced an identical mask. Batch 2 took 32.75 s and changed 5 mask pixels; batch 4 took 75.28 s and changed 6 pixels. Peak PyTorch allocated memory rose from 8.30 GiB (batch 1) to 10.44 GiB (batch 2) and 13.89 GiB (batch 4). The small pixel differences are numerically tiny but fail bit-exact mask preservation, while the larger batches are much slower and consume substantially more memory on this shared-memory ROCm system. Tile batching is rejected for this hardware. Batch-1 timing remains a single-photo measurement and needs repeated confirmation before treating its 12% difference from the canonical call as a real speedup.

The full teacher-mask set is complete: 207 matching training image/mask pairs and 10 matching segmentation-validation pairs, all sourced from the project training split; project validation and test rows were excluded. The student was fine-tuned from the 48-image pilot checkpoint with a 60-epoch cap and early stopping. Training stopped at epoch 41 and selected epoch 29. On the same 10-photo internal segmentation-validation subset, the best checkpoint reached mIoU 0.8555 and pixel accuracy 0.923. This is teacher-mask agreement, not waviness quality. The trained checkpoint is in `/tmp/wave_yolo26_student_pilot/runs/binary-water-full-train-split/weights/best.pt`.

### Candidate evaluation protocol update

Per the project owner's instruction, future speed candidates must report quality alongside latency even when they miss the speed target. For newly run candidates, capture the candidate's relevant task-quality metric (for segmentation, agreement with the canonical B2 mask on held-out images) and downstream waviness error (MAE/RMSE on the untouched regression test split), plus full-request warm latency, failure count, CPU time, and peak RAM/accelerator memory. A latency miss does not waive these quality measurements. Previously completed candidates are not being rerun solely to backfill this information.

### Full-split YOLO26 student on the untouched regression test split

I evaluated the selected full-split YOLO26 student with the existing v18 EfficientNet-B0 head on all 46 untouched regression-test photos. The route performs a fresh student inference on the raw photo for Step 1 and a second inference on the decoded Step 1 JPEG for Step 2; it does not reuse the first mask. It includes HEIC decode, both JPEG byte round trips, regression, and a flushed prediction-record write. Six rows failed preprocessing (2 Step 1 water-area failures and 4 Step 2 crop failures), so the candidate is not deployable as measured. On the 40 successful rows only, MAE/RMSE were 0.1230/0.1638; this success-only score is selection-biased and is not comparable to a complete 46-row evaluation. The canonical GPU-standardized B0 reference was MAE/RMSE 0.1276/0.1591 on all 46 rows, so candidate RMSE was worse before accounting for its failures.

Warm latency over the 40 successful rows was 250.2 ms median / 279.6 ms p95; 0/40 were under 100 ms. Peak process RSS was 3.12 GiB; peak PyTorch allocated/reserved accelerator memory was 0.32/0.34 GiB. Process CPU time was typically 330–395 ms per 225–280 ms wall request despite PyTorch and OpenCV being limited to one thread; this includes the 16-thread HEIC decoder. The `step1_standardize_jpeg_ms` timing combines high-resolution mask erosion/crop, Lanczos resizing, encoding, and decoding and needs finer profiling. Raw candidate masks, row results, prediction records, failure list, and summary are under `/tmp/wave_yolo26_student_full_eval/`.

This paired evaluation showed the raw-only student reached neither the latency target nor a robust quality gate. It was trained on raw photos only, while inference applies it a second time to the standardized 224x224 Step 1 image. The follow-up dual-domain training branch added canonical teacher masks for Step 1 outputs, using only project training-split photos.

### Dual-domain YOLO26 student: raw plus Step 1 inputs

To address the Step 2 input-domain gap, I assembled a second student dataset with 207 original photo/mask pairs plus their 207 persisted canonical Step 1 photos and B2 masks. I hard-linked the existing raw JPEG/mask files rather than copying them; the 217 Step 1 masks were generated by SegFormer-B2 from the already saved Step 1 output images. The split is 414 training examples and 20 internal validation examples (10 original and 10 Step 1 variants of training-split photos); project validation and test photos remain excluded. The model was fine-tuned from the raw-only student for 40 epochs, batch 8, with RAM caching. Best internal mIoU was 0.8854 at epoch 33 and final validation of the selected checkpoint reached 0.8859 mIoU / 0.947 pixel accuracy on the 20-image validation subset.

On the untouched 46-photo regression test set, this dual-domain student still failed Step 2 on 7 photos. On the 39 successes only, the existing v18 regression head produced MAE/RMSE 0.1301/0.1752 (selection-biased because of failures). Warm full-request latency before the ROI change was 244.6 ms median / 277.6 ms p95; 0/39 were under 100 ms. Peak RSS was 3.14 GiB and accelerator allocated/reserved memory was 0.31/0.32 GiB. After the exact ROI change below, the same route measured 236.7 ms median / 277.1 ms p95, with identical predictions and failures; peak RSS was 3.06 GiB. This is worse than the raw-only student's all-success-subset MAE/RMSE 0.1230/0.1638, with one more failure; neither candidate qualifies. Details are under `/tmp/wave_yolo26_student_dual_eval/` and the dual training data/checkpoint are under `/tmp/wave_yolo26_student_dual/`.

### YOLO26 raw-only full-request timing breakdown

The 40 successful rows from the raw-only test run show medians of 50.7 ms for input decode, 27.7 ms for Step 1 YOLO inference/postprocessing, 132.4 ms for Step 1 high-resolution standardize/encode/decode, 8.4 ms for Step 2 YOLO, 20.8 ms for Step 2 crop/encode/decode, and 5.2 ms for regression plus record output. This isolates the high-resolution Step 1 crop/resize as the largest measured cost. The exact ROI change below reduces some copy overhead, but high-resolution resizing remains the main opportunity. Even replacing its former 132 ms Lanczos work with the earlier 2.2 ms Gaussian-pyramid microbenchmark would still leave roughly 120 ms before further improvements. The multistage resize changes pixels, so it needs a matching-regressor held-out quality gate before adoption.


### YOLO26 FP16 inference precision probe

I ran the dual-domain YOLO26 student with Ultralytics FP16 inference on the same 46 untouched regression-test photos, keeping preprocessing and the existing regression checkpoint fixed. The same 39 photos succeeded and the same 7 failed as FP32. On those paired successes, downstream MAE/RMSE improved from 0.13011/0.17519 to 0.12393/0.16480, but the predictions changed (mean absolute delta 0.01421; maximum 0.14755). The 39 successful source-resolution masks had mean IoU 0.99921 against FP32 (minimum 0.99094), so this is close but not identical output. Median full-request latency was 238.5 ms versus 236.7 ms for FP32; p95 was 269.9 versus 277.1 ms, and neither reached 100 ms. The first request also showed a large timing spike, and Ultralytics emitted a deprecation warning for its `half` argument. Since this ROCm run did not improve median latency and changed masks/predictions, FP16 is not adopted. A later precision experiment would need an explicitly supported quantization path and the same complete quality gate. Artifacts: `/tmp/wave_yolo26_student_dual_fp16_eval/` and `/tmp/wave_yolo26_student_dual_fp16_eval.py`.

### HEIF decoder thread tuning with exact output validation

The benchmark harness had explicitly set Pillow-HEIF to 16 decoder threads, while `step_7_a_predict.py` left the library at its default of 4. On the 46 untouched HEIC test photos, I compared 1, 2, 4, 8, and 16 decode threads. Median decode wall time was 145.4, 103.3, 72.7, 61.2, and 54.8 ms respectively; all decoded RGB pixels matched the 1-thread result exactly on all 46 photos. Median process CPU time was 146.8 ms at one thread and 162.3 ms at 16, so this trades about 10% more CPU work for about 62% lower decode wall time.

I then compared full dual-student predictions at decoder defaults of 4 and 16 threads. On the same 39 successful test photos, all 39 regression predictions were bit-identical and MAE/RMSE remained 0.13011/0.17519; the same seven photos failed preprocessing in both runs. Median complete-request latency improved from 258.4 to 236.7 ms (8.4%); p95 improved from 285.7 to 277.1 ms. The 16-thread run used 3.06 GiB peak RSS and 0.31/0.32 GiB accelerator allocated/reserved memory. Zero requests met 100 ms. This exact-output latency improvement is now configured for prediction: `step_7_a_predict.py` uses up to 16 HEIF decoder threads by default, capped by logical CPU count, and `WAVE_HEIF_DECODE_THREADS` can override it for hardware-specific tuning. The higher thread count increases CPU use during decode, so production concurrency should be measured on the target host. Eval artifacts are in `/tmp/wave_yolo26_student_dual_eval/` and `/tmp/wave_yolo26_student_dual_threads4_eval/`.

### Performance-principle check: multi-stage Lanczos downsampling

I reviewed the official PyTorch performance guidance before choosing the next experiment. The serving path already uses `torch.inference_mode()` and single-thread settings; it already avoids an extra pinned-memory staging copy, which PyTorch notes can itself be slower for small transfers. The remaining approach is to profile each full-request stage, reduce data movement only when it is measurable, and validate outputs and task quality on the untouched test split. References: [PyTorch performance tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide), [inference_mode](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html), and [PyTorch host/device transfer guidance](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html).

As a quality-gated speed probe, I ran the dual-domain student end-to-end on all 46 untouched regression-test photos with Pillow Lanczos `reducing_gap=3.0` in the high-resolution Step 1 resize. This is a valid implementation option for faster large downscales, but it changes pixels and is therefore an experimental preprocessing candidate, not an exact optimization. Eight photos failed Step 2 (one more than the canonical-Lanczos run). On the 38 rows that succeeded in both runs, latency fell from 236.7 ms median / 276.7 ms p95 to 196.0 / 210.5 ms (about 17.2% lower median); zero requests met 100 ms. The same rows' waviness MAE/RMSE changed from 0.1261/0.1715 to 0.1335/0.1786, and candidate predictions differed by 0.0168 MAE (maximum absolute delta 0.0912). Peak RSS was 3.01 GiB and accelerator allocated/reserved memory 0.31/0.32 GiB. The speed gain came with worse measured quality and an additional failure, so this resize change is rejected for the current checkpoint and is not adopted. Any future matched-regressor experiment must train on training-split inputs created by this resize, then pass untouched validation and test quality plus the complete-latency target. Full artifacts: `/tmp/wave_yolo26_student_dual_reducing_gap_eval/` and `/tmp/wave_yolo26_student_dual_reducing_gap_eval.py`.

### Exact ROI copy reduction in Step 1

I changed `_standardize_eroded_mask` to crop the PIL source first and apply its binary mask only to that ROI. This avoids materializing a NumPy copy of the complete RGB source and a second copied ROI. On 40 successful untouched test photos, the new result matched the old standardizer pixel-for-pixel and produced byte-identical Step 1 JPEGs on all 40. An alternating benchmark including JPEG encode/decode measured 91.3 ms median / 131.6 ms p95 for the previous NumPy path and 81.6 / 102.2 ms for the PIL ROI path, saving 9.7 ms median without changing pixels or JPEG bytes.

I reran the full dual-student route after the change. Test predictions, success-only MAE/RMSE (0.1301/0.1752), and all 7 failure rows were unchanged. Full-request median improved from 244.6 to 236.7 ms; p95 changed from 277.6 to 277.1 ms. Peak RSS fell from 3.14 to 3.06 GiB in the rerun. This exact-output memory/copy reduction is adopted, but the complete pipeline remains above 100 ms.

### Static-shape YOLO26 ONNX with direct MIGraphX GPU execution

Following the official [Ultralytics ONNX export guidance](https://docs.ultralytics.com/modes/export) and [MIGraphX Python API](https://rocm.docs.amd.com/projects/AMDMIGraphX/en/develop/reference/MIGraphX-py.html), I exported the dual-domain student to static FP32 ONNX graphs for both observed raw-image input geometries and the square Step 2 input. MIGraphX compiled programs were cached under `/tmp`; graph compilation took about 41–43 seconds per raw-image shape and 17.5 seconds for the square graph. Loading those compiled programs took under a second in a separate check. The models run directly through MIGraphX in the same process as the regression model on this ROCm stack, avoiding a socket bridge. Warm engine execution was around 1.7–1.9 ms for the two stages, but this excludes CPU image-to-tensor preparation.

I then ran the complete MIGraphX route on all 46 untouched regression-test photos, including HEIC decode, image preparation, both segmentation calls, Step 1 and Step 2 JPEG round trips, regression, and flushed prediction-record output. It preserved the same 7 Step 2 failures. On the same 39 successful rows, waviness MAE/RMSE were 0.13998/0.17766 versus 0.13011/0.17519 for the PyTorch student; candidate predictions differed by 0.04392 MAE (maximum 0.17715). The Step 1 masks had mean IoU 0.99732 (minimum 0.99259) against PyTorch, so the engine was close but not output-identical. Full-request median was 237.5 ms versus 236.7 ms for PyTorch; p95 was 263.6 versus 277.1 ms. Neither route met 100 ms. Peak process RSS was 2.93 GiB. PyTorch reported 0.11/0.15 GiB allocated/reserved, but those values exclude MIGraphX allocations.

This engine does not improve end-to-end median latency and it worsens measured waviness quality, so it is rejected for the current checkpoint. The engine-only warm timing is not representative of the request: CPU image preparation and HEIC decode remain substantial, and two static raw-image shapes plus the square Step 2 graph are required for these test inputs. Artifacts: `/tmp/wave_yolo26_migraphx_full_eval_direct.py`, `/tmp/wave_yolo26_migraphx_full_eval/`, and `/tmp/wave_yolo26_migraphx_cache/`.

### OpenCV binary masking inside the cropped Step 1 ROI

After the exact PIL ROI reduction, I compared `cv2.bitwise_and` with the current PIL paste for applying the binary water mask to the already-cropped ROI. The OpenCV route produced pixel-identical standardized images and byte-identical Step 1 JPEGs on all 39 successful untouched test photos. An alternating encode/decode microbenchmark measured 82.4 ms median / 104.8 ms p95 for PIL ROI masking and 69.4 / 90.1 ms for OpenCV ROI masking, saving 13.1 ms median. The complete dual-student evaluation preserved all 39 regression predictions exactly and preserved all 7 failure rows; success-only MAE/RMSE stayed 0.13011/0.17519. Full-request median improved from 236.7 to 225.6 ms, and p95 from 277.1 to 249.4 ms. Peak RSS increased from 3.06 to 3.14 GiB; accelerator allocated/reserved remained about 0.31/0.32 GiB. This exact-output end-to-end improvement is adopted in `_standardize_eroded_mask`. The pipeline now declares the headless OpenCV wheel, which is intended for server/CLI use without GUI modules ([official package guidance](https://pypi.org/project/opencv-python-headless/)).

### CPU thread count for single-image prediction

The ROCm host reports 32 logical CPUs, and both PyTorch and OpenCV default to 32 threads. I compared the full dual-student route with both libraries at one thread versus 32, keeping HEIF decoding at 16 threads. The outputs were unchanged: same 39 predictions, MAE/RMSE 0.13011/0.17519, and same 7 failures. One thread measured 225.6 ms median / 249.4 ms p95; 32 threads measured 260.9 / 297.9 ms. Median process CPU time fell from 1464.5 to 340.1 ms per successful request and peak RSS from 3.25 to 3.14 GiB. OpenCV's isolated ROI-mask thread sweep from 1 to 32 showed no meaningful throughput benefit, so prediction now defaults both PyTorch and OpenCV to one thread; `--threads` and `--opencv-threads` remain available for machine-specific tuning. This moves the full pipeline faster while preserving exact predictions, although the median is still 125.6 ms above the target.

The test scripts and per-photo records are in `/tmp/wave_step1_cv2_roi_benchmark.py`, `/tmp/wave_yolo26_student_dual_cv2_roi_eval.py`, and `/tmp/wave_yolo26_student_dual_cv2_roi_threads32_eval.py`. The current production helper was also rechecked against the previous NumPy implementation: all 39 standardized pixels and JPEG bytes matched exactly.

### OpenCV erosion for exact binary masks

The full-resolution erosion was the largest isolated CPU image operation. I compared OpenCV `erode` using a rectangular kernel and replicated borders with SciPy's `minimum_filter(..., mode="nearest")` on 39 saved 12.2-megapixel binary masks. All 39 eroded arrays were byte-identical. With both libraries limited to one CPU thread, alternating warm measurements gave 30.80 ms median for SciPy and 3.34 ms for OpenCV (27.48 ms median paired saving). This matches the documented operation: both compute a neighborhood minimum, and the border modes were validated by exact output comparison ([OpenCV erosion reference](https://docs.opencv.org/4.11.0/d4/d86/group__imgproc__filter.html), [SciPy boundary modes](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.minimum_filter.html)).

I also replaced the full-mask row/column scans and sum with OpenCV `boundingRect` and `countNonZero`. Both the half-open bounding box and foreground count matched the NumPy implementation on all 39 saved masks; median combined scan time fell from 3.46 ms to 0.43 ms. The full two-pass YOLO26 student plus v18 regression route was then rerun twice on the 46 untouched regression-test photos. Candidate medians were 192.4 and 195.3 ms (p95 213.0 and 223.9 ms), versus 225.6 ms / 249.4 ms in the preceding one-thread reference run. In each candidate run, all 39 successful predictions matched the previous full route bit-for-bit; the same 7 photos failed Step 2. MAE/RMSE stayed 0.13011/0.17519, with zero of 46 requests under 100 ms. Latest peak RSS was 3.12 GiB and accelerator allocated/reserved memory was 0.31/0.32 GiB. These are exact-output held-out improvements rather than a quality tradeoff; request latency varies between runs, so the two candidate measurements are both recorded. Summaries are in `/tmp/wave_yolo26_student_dual_cv2_roi_eval/summary.json` and `/tmp/wave_yolo26_student_dual_cv2_roi_threads32_eval/summary.json`; the temporary runtime used Ultralytics 8.4.191 and OpenCV 5.0.0 under `/tmp/wave-runtime-deps`, leaving the project environment untouched. OpenCV is already a declared project dependency; the now-unused SciPy filter import was removed from this step.

### OpenCV Lanczos resize probe (rejected for the current checkpoint)

The profile showed that exact Pillow Lanczos downscaling was the largest remaining Step 1 substage (about 46 ms median). I tested replacing it with OpenCV `INTER_LANCZOS4`, leaving segmentation masks, crop selection, and JPEG round trips intact. On the same 46-photo untouched test split, median complete-request latency fell from 195.3 to 131.1 ms and p95 from 223.9 to 152.6 ms; Step 1 standardization fell from 86.2 to 26.7 ms median. The candidate completed 41 photos and failed 5, compared with 39 completed / 7 failed for the exact Pillow path.

Quality did not hold with the existing checkpoint. On the 39 photos successful in both routes, waviness MAE/RMSE changed from 0.13011/0.17519 to 0.14681/0.17601. Predictions shifted by 0.07871 MAE (maximum absolute change 0.19457). Although the candidate improved Step 2 coverage by two photos, the shared-photo MAE regression is material, so this resize is rejected and is not adopted. The speed result shows the remaining opportunity in resize, but a valid future trial would need a regressor trained with the candidate preprocessing, then frozen validation and untouched test quality gates before it can be considered. Full results are in `/tmp/wave_yolo26_student_dual_cv2_resize_eval/summary.json` and `/tmp/wave_yolo26_student_dual_cv2_resize_eval/results.csv`; the exact-path comparison is in `/tmp/wave_yolo26_student_dual_cv2_roi_eval/`.

### One-pass segmentation with projected Step 1 mask (promising, contract not yet matched)

After replacing the full-resolution SciPy erosion with exact OpenCV erosion, I reran the existing one-pass proposal: segment the raw image once, erode that binary mask, and project the same mask through the canonical Step 1 crop/resize geometry for Step 2. The Step 1 JPEG remains on the normal path; only Step 2's second segmentation call is avoided. On the 46-photo regression test split (excluded from training, but previously used in candidate development), this route completed 45 photos with 1 Step 2 failure, versus 39 completions and 7 failures for the two-pass route. On the 39 shared successes, MAE/RMSE improved from 0.13011/0.17519 to 0.12432/0.16822; prediction shift was 0.06456 MAE (maximum 0.26685). Candidate-only total metrics were MAE/RMSE 0.12437/0.16732 across 45 successes. Median latency was 174.5 ms / p95 195.6 ms, versus 195.3 / 223.9 ms; one request was below 100 ms. Peak RSS was 3.10 GiB and accelerator allocated/reserved memory 0.32/0.34 GiB. These test scores are exploratory because repeated candidate results have already been reviewed; a fresh final quality claim needs future captures held out from this development cycle.

I also ran both routes on the separate 47-photo validation split. The canonical path completed 37 and failed 10; projected-mask reuse completed all 47. On the 37 shared successes, MAE/RMSE improved from 0.13189/0.16288 to 0.11533/0.15649; the candidate prediction shift was 0.07216 MAE (maximum 0.23417). Across all candidate validation successes, MAE/RMSE were 0.12027/0.16084. Candidate median latency was 176.1 ms / p95 199.8 ms, with zero requests below 100 ms. These results make projected-mask reuse a strong quality and speed research candidate, not a production change yet: the current v18 checkpoint was trained with separate Step 2 segmentation and its preprocessing contract does not authorize this route. A matching training/validation contract and checkpoint must be produced before deployment; full request latency also remains above target. Artifacts are `/tmp/wave_yolo26_student_dual_mask_reuse_cv2_eval/` and `/tmp/wave_yolo26_student_dual_mask_reuse_cv2_validation_eval/`.

#### Matched-head adaptation probe

To quantify the checkpoint-contract gap, I extracted frozen EfficientNet-B0 features for the projected-mask inputs using all 217 training rows and 47 validation rows, then selected the linear-head update using a 58-example capture-date holdout from training only. Three training photos needed the already-saved canonical Step 2 image because YOLO could not produce a Step 1 crop; no validation photo needed fallback. The selected update used teacher weight 1.0 for one epoch. On validation, the adapted head scored MAE/RMSE 0.12366/0.16593 on candidate inputs, versus 0.12027/0.16084 from the original head on the same candidate route. So this adaptation did not improve the candidate's validation score; the update remains rejected. Its final test run, after freezing the training-only choice, completed 45/46 photos with MAE/RMSE 0.12771/0.16186 and median latency 180.0 ms / p95 209.6 ms (one request under 100 ms). On the 39 photos shared with the two-pass reference, adapted-head MAE/RMSE were 0.12860/0.16403 versus 0.13011/0.17519. Test diagnostics are exploratory because this split has already been used in earlier candidate development. The temporary checkpoint and feature cache are under `/tmp/wave_yolo26_projected_mask_head/`; no repository checkpoint was changed.

### Lookup-table class-to-water mask extraction

The configured SegFormer returns a CPU `uint8` class map, and its current water IDs are `(21, 26, 60, 128)`. The previous extractor allocated Torch boolean maps for each class and then copied the result to `uint8`. I replaced that path for `uint8` maps with a cached 256-entry lookup table; non-`uint8` maps retain the original generic implementation. The returned 0/1 mask matched the prior implementation exactly for all 256 possible class values and on three real full-resolution SegFormer class maps. On the real maps, median extraction time was 16.91 ms for the previous Torch path and 10.45 ms for the lookup path (6.46 ms saved per mask); a synthetic 12.2-megapixel map measured 18.34 and 11.25 ms respectively. The lookup emits one 12.2 MB output array and avoids the previous full-frame boolean intermediates plus uint8 output copy. This is an exact mask operation, so downstream crops and predictions remain unchanged for these class maps.

The same real-map run exposed the scale of the configured canonical profile: `baseline-v16` processed 35 overlapping tiles and took 9.54–10.38 seconds for segmentation alone on each of three images. The lookup optimization saves only a few milliseconds in that route and does not alter its dominant tile-model compute; the sub-100-ms target still requires a different, quality-validated segmentation deployment candidate. No source or generated images were added to the repository.

### Returning a low-resolution YOLO mask before host upsampling (rejected for current head)

Ultralytics returns the YOLO26 student's class map at the full decoded source size. I passed the exact 640-long-side RGB input to the model and resized its low-resolution class map to source dimensions on CPU with nearest-neighbor interpolation. The classifier tensor had been verified bit-identical to the normal Ultralytics input on four photos, but changing the order of `argmax` and mask upsampling changed class decisions: raw-mask IoU against the full-resolution Ultralytics output had test median/minimum 0.99745/0.97401 and validation median/minimum 0.99736/0.98415. Both splits have been examined repeatedly during development, so these quality results are exploratory rather than an independent final holdout.

With projected Step 1 mask reuse, full-request test latency changed from 174.5 ms / p95 195.6 ms to 171.2 / 192.6 ms, and accelerator allocated/reserved memory fell from 0.32/0.34 GiB to 0.13/0.16 GiB. Both routes completed 45/46 photos with the same failure. On test, MAE/RMSE worsened from 0.12437/0.16732 to 0.13286/0.17856; shared-success prediction delta was 0.02717 MAE (maximum 0.15289). Validation completed 47/47 in both routes, and its MAE/RMSE improved from 0.12027/0.16084 to 0.11357/0.15578. Because validation and test moved in opposite directions and test quality regressed, I reject low-resolution output for the current head. The memory improvement is real, but the speed gain is small and does not offset the quality risk. Artifacts: `/tmp/wave_yolo26_student_dual_lowres_mask_validation_eval/` and `/tmp/wave_yolo26_student_dual_lowres_mask_test_eval/`.

### Upstream Pillow 13 resize check and Pillow-SIMD availability

I tested the upstream Pillow 13.0.0 binary in `/tmp` as a drop-in implementation of the existing Lanczos resize. On 10 representative test crops, decoded source pixels matched across Pillow 12.3.0 and 13.0.0, and all 10 resulting Step 1 JPEGs were byte-identical. Median resize time was 45.87 ms on 12.3.0 and 46.03 ms on 13.0.0 (median paired saving -0.10 ms), so the newer version gives no measurable speed gain and was not adopted. Pillow-SIMD advertises accelerated convolution resampling ([project README](https://github.com/neuroradiology/pillow-simd)), but its current source distribution failed to build in this host because JPEG development headers/libraries are absent; no system packages were changed. The isolated Pillow 13 and temporary build attempt are outside the repository.

### HEIF decoder thread sweep for the YOLO student route

I compared 16, 24, 32, and 48 Pillow-HEIF decoder threads on all 46 test sources, with two interleaved passes and SHA-256 hashes of the oriented decoded RGB pixels. All 46 images decoded pixel-identically across thread settings. Median wall time per decode was 76.36, 74.24, 74.74, and 74.18 ms; median process CPU time was 186.50, 198.64, 201.55, and 208.96 ms. Going from 16 to 24 threads saves about 2.1 ms wall time while using about 6.5% more CPU time; 32 and 48 threads provide no further median improvement. I kept the current 16-thread default because the measured gain is small relative to the extra CPU cost and per-request latencies vary with host load. This decoder-only measurement does not reduce the full request below 100 ms.

### B2 overlapping-tile batching probe (batch size 2 exceeds accelerator memory)

On one held-out HEIC photo, the canonical 35-tile B2 route completed in 10.42 s and peaked at 8.30 GiB PyTorch allocated / 8.42 GiB reserved memory when it kept the 6.81 GiB logit accumulator on the accelerator. I tried processing two same-size tiles together while preserving tile order, interpolation, and accumulation order. With the GPU accumulator retained, the batch failed with ROCm out-of-memory. I repeated with the full-resolution accumulator on CPU to free accelerator space, but batch-2 activations again filled the 48 GiB accelerator allocation limit (47.49 GiB allocated, under 30 MiB free) before producing a class map. No mask, latency comparison, or quality result exists for batch 2, so it is rejected on this hardware. The baseline mask is saved at `/tmp/wave_b2_tile_batch_baseline.npy`; no images or code were added to the repository. The repeated OOM demonstrates that tile batching is not currently a viable optimization for this model/profile on the shared-memory ROCm device.

### Gaussian-pyramid downscale for projected-mask Step 1 (rejected for the current checkpoint)

The exact Pillow Lanczos resize remained the largest CPU stage in the fast one-pass projected-mask route. I tested a multi-level OpenCV `pyrDown` Gaussian pyramid followed by `INTER_AREA` resize, without changing the crop bounds, water mask, output canvas, JPEG quality, Step 2 projection, or regression checkpoint. This changed the Step 1 pixels and therefore is a candidate preprocessing contract, not an equivalent implementation.

On 47 validation photos, the existing regressor completed 47/47 and scored MAE/RMSE 0.11862/0.16503, versus 0.12027/0.16084 with the exact Pillow path. On the 46-photo regression test set, it completed 45/46 (the same one failure as the exact path) and scored MAE/RMSE 0.13218/0.16937, versus 0.12437/0.16732 on the same 45 successful photos. Thus test MAE worsened by 0.00781 and RMSE by 0.00205; across shared test photos predictions moved by 0.03898 MAE, with a maximum absolute shift of 0.26674. The opposing validation/test direction and clear test MAE regression fail the current-checkpoint quality gate. The candidate is rejected and has not been adopted.

Latency improved materially: full-request median/p95 was 130.1/161.0 ms on validation and 133.1/161.2 ms on test, compared with 176.1/199.8 ms and 174.5/195.6 ms for the exact Pillow projected-mask route. Paired median Step 1 standardization fell from 72.3 to 24.7 ms on validation and 70.0 to 26.9 ms on test. Decode, segmentation, mask projection, Step 2 crop/JPEG, and regression stages were essentially unchanged. Two validation and one test request were under 100 ms, but median full request remains above target. Peak RSS stayed near 3.10 GiB and accelerator allocated/reserved memory at 0.32/0.34 GiB, so this is a CPU resize tradeoff, not a memory reduction.

The practical computer-vision lesson is that faster resampling can change the feature distribution seen by a fixed regressor even when the geometry and mask stay fixed. It must not be accepted based on mask quality or timing alone. A model trained for this exact resize could be evaluated as a new, matched pipeline on fresh validation and future captures; the already-reviewed test split is exploratory and cannot certify that future model. Full per-image data are under `/tmp/wave_yolo26_student_dual_pyrdown_validation_eval/` and `/tmp/wave_yolo26_student_dual_pyrdown_test_eval/`.

For subsequent performance work, I am applying these measured principles: profile the full request and label its stages before optimizing; focus on the dominant stage (here standardization, not model execution); remove full-frame intermediates and copies only when decoded pixels/JPEG bytes or downstream predictions remain exact; tune CPU thread counts empirically because oversubscription increased latency and CPU time here; and report host RSS plus accelerator memory because PyTorch allocator figures do not include every third-party runtime allocation. PyTorch's official guidance supports `torch.inference_mode()` for pure inference (with `model.eval()` separately), and its profiler can attribute CPU/device time and tensor memory, while profiling itself can add overhead and retain tensors. We should use these as measured candidates, not assume each is a speedup: [inference_mode documentation](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html), [PyTorch profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html), [profiler API](https://docs.pytorch.org/docs/stable/profiler).

### Reuse the existing `uint8` ROI mask buffer

The exact Pillow route's ROI mask was already `uint8` after extraction and erosion, but `astype(np.uint8)` still requested a copy. I changed it to `astype(np.uint8, copy=False)`, which returns the ROI view when the dtype is already correct and still converts other supported dtypes. On the full 46-photo test run, all 45 successful Step 1 JPEGs and projected-mask PNGs were byte-identical to the prior run, all 45 regression predictions matched exactly, and the same one photo failed Step 2. MAE/RMSE therefore remained 0.12437/0.16732. The copy-free run measured 178.5 ms median / 198.3 ms p95 versus 174.5 / 195.6 ms in the preceding run; paired median total latency was 4.2 ms slower, with most of that run-to-run difference appearing in decode and model timing rather than ROI standardization. There is no demonstrated end-to-end speed gain from this micro-optimization, so it is retained only as a redundant-copy removal with exact output behavior, not counted toward the speed target. Artifacts are under `/tmp/wave_yolo26_student_dual_mask_reuse_cv2_copyfree_eval/`.

### Decode embedded HEIF thumbnails for a reduced-resolution candidate (rejected with the current regression checkpoint)

The test and validation splits contain 92 HEIC sources; every source has an embedded 320-pixel-long-side thumbnail. Pillow-HEIF supports decoding an embedded thumbnail directly through `get_thumbnail()` and its docs describe `draft`/`thumbnail()` as a way to avoid full-resolution decode when a sufficiently large embedded thumbnail exists ([thumbnail behavior](https://pillow-heif.readthedocs.io/en/stable/pillow-plugin.html), [HEIF thumbnail API](https://pillow-heif.readthedocs.io/en/latest/heif-file.html)). I tested that thumbnail as the entire input image for the same YOLO26 student, exact crop/mask logic, projected Step 2 mask, JPEG round trips, and existing v18 regressor. This changes source resolution and pixel content; it is a new preprocessing contract.

The candidate reduced full-request latency to 36.7 ms median / 40.0 ms p95 on validation and 38.1 / 45.1 ms on test. It completed 47/47 validation and 44/46 test requests; two test requests failed water detection compared with one failure in the exact-resolution projected-mask route. On validation, current-checkpoint MAE/RMSE changed from 0.12027/0.16084 to 0.13253/0.16424. On the 44 test photos successful in both routes, MAE/RMSE changed from 0.12603/0.16904 to 0.14862/0.18278; candidate-only test metrics were 0.14862/0.18278 across 44 successes. Thus speed is excellent, but both split comparisons show worse waviness quality and test failures increased. The route is rejected for the existing checkpoint and has not been adopted. Peak process RSS fell from about 3.15 to 2.53 GiB on validation and 2.31 GiB on test; accelerator allocated/reserved memory fell from 0.32/0.34 GiB to 0.30/0.34 GiB on validation and 0.12/0.15 GiB on test. The reduced host image also makes mask operations substantially cheaper.

This is worth a separately trained, contract-matched regressor experiment, but the current validation and test splits have already been inspected repeatedly and cannot provide an independent final quality claim. Any future adoption requires candidate-specific training and validation plus fresh capture-date-held-out data, with both metric quality and failure coverage held to the existing acceptance bar. Candidate records are under `/tmp/wave_yolo26_student_dual_heif_thumbnail_validation_eval/` and `/tmp/wave_yolo26_student_dual_heif_thumbnail_test_eval/`.

I also tried the existing conservative frozen-backbone head adaptation procedure on candidate thumbnail features: extract all 217 training and 47 validation examples, select teacher blend and epoch using a 58-example capture-date holdout from training only, then evaluate validation. Six training examples needed the saved canonical Step 2 image because candidate segmentation/cropping could not produce an input; validation needed no fallback. All tested teacher blends selected epoch 0 on the training-only holdout, so the procedure chose no head update. Validation remained MAE/RMSE 0.13253/0.16424 on thumbnail inputs. This adaptation therefore provides no recovery evidence; a larger model change would need more candidate-matched labeled data and fresh future captures. The exploratory feature cache and checkpoint are under `/tmp/wave_yolo26_heif_thumbnail_mask_head/`; the repository checkpoint was not changed.

I followed that with a conservative partial fine-tune from v18 using the candidate Step 2 JPEGs, unfreezing only EfficientNet-B0 `features[7:]` and the regression head (1,130,673 trainable parameters). I trained on 159 samples and selected against 58 training samples grouped by capture date; validation and test were excluded from selection. Training stopped after 20 epochs. Epoch 0 remained best on the date holdout at MAE/RMSE 0.11504/0.15477; epoch 1 worsened holdout MAE to 0.12146, and later epochs continued to deteriorate. The selected candidate is therefore unchanged from the original head and validation remained 0.13253/0.16424. The current 217-label dataset gives no evidence that this small partial fine-tune can recover the thumbnail route's quality; I did not run a redundant test evaluation of the unchanged checkpoint. Training data, logs, and the epoch-0 checkpoint are under `/tmp/wave_yolo26_heif_thumbnail_finetune_data_v3/` and `/tmp/wave_yolo26_heif_thumbnail_finetune/`.

### Thumbnail-domain YOLO26 student: sub-100-ms quality-gated candidate

The thumbnail experiment's regression failures traced to masks generated by the student trained on full-resolution imagery. I built an input-domain training set from the 207 canonical B2 training masks, downsampled each binary teacher mask to its matching embedded HEIF thumbnail with nearest-neighbour sampling, and reserved the existing 10-image segmentation validation subset. Fine-tuning the dual-domain YOLO26 student for 9 epochs selected epoch 1. On the held-out segmentation subset, mask mIoU/pixel accuracy improved from 0.762/0.869 for the original student to 0.834/0.910 for the selected checkpoint. Training later epochs reduced mask mIoU; no regression validation or test labels were used to choose this checkpoint.

On the four untouched test photos that have archived canonical B2 masks, thumbnail-mask IoU changed from 0.934/0.309/0.952/0.848 to 0.941/0.521/0.954/0.829. Three improved and one declined; the weakest case improved materially but remains imperfect. These four segmentation comparisons are diagnostic only and do not replace regression metrics.

I reran the full route with that student and the existing v18 regression head. HEIC sources use their embedded 320-pixel-long-side thumbnail. The one JPEG source in validation uses Pillow's JPEG `draft()` decoder at a requested 320-pixel size, which decodes it at 378 x 504 before preprocessing; this keeps the decoder from materializing its 4032 x 3024 RGB frame. The route still performs the crop, masking, Step 1 JPEG round trip, projected-mask Step 2 crop, final Step 2 JPEG round trip, regression forward, and flushed prediction-record write.

On validation, all 47 requests succeeded and all 47 completed under 100 ms; median/p95 latency was 37.4/40.1 ms and waviness MAE/RMSE was 0.11983/0.15032. On test, all 46 succeeded and all 46 were under 100 ms; median/p95 was 37.3/40.0 ms and MAE/RMSE was 0.11821/0.14938. For context, the earlier full-resolution one-pass projected-mask YOLO route measured 176.1 ms median / 199.8 ms p95 and MAE/RMSE 0.12027/0.16084 on validation; on test it measured 174.5/195.6 ms, scored 0.12437/0.16732 on 45 successes, and failed one photo. The canonical fast-v1 B0 reference previously scored 0.13222/0.16590 on validation and 0.12763/0.15911 on test. Thus the thumbnail-domain candidate is substantially faster and its observed regression metrics are at least as good as these previously measured references, with no preprocessing failures in these two runs.

The JPEG-specific reduction also closed the only earlier latency exception: for `IMG_7565.jpeg`, Pillow draft decode changed request latency from 204.4 ms to 46.8 ms. Its waviness absolute error changed from 0.06191 to 0.03160; this is one image and is not a standalone quality claim. The full validation rerun with JPEG draft included scored MAE/RMSE 0.11983/0.15032 and had 47/47 requests under 100 ms.

This is the first measured route in this work that meets the latency target on every row in both available regression splits while preserving or improving their aggregate quality metrics. These splits have been repeatedly examined during candidate development, so they are exploratory evidence, not a fresh independent final quality guarantee. Also, the new YOLO checkpoint and thumbnail/JPEG-draft decoding are not yet represented in the production checkpoint contract or pipeline code. Do not adopt it as the default until the contract is implemented and quality is checked on fresh future captures. Candidate checkpoint, training set, and full per-photo results remain under `/tmp/wave_yolo26_student_thumbnail_teacher/`, `/tmp/wave_yolo26_heif_thumbnail_finetune_data_v3/`, `/tmp/wave_yolo26_student_thumbnail_jpeg_draft_validation_eval/`, and `/tmp/wave_yolo26_student_thumbnail_teacher_heif_test_eval/`.



### Matched-regressor follow-up for OpenCV Lanczos resize

I generated the candidate preprocessing outputs for the existing 217/47/46
train/validation/test split and fine-tuned v18's final EfficientNet-B0 blocks
and regression head. Selection used a capture-date holdout from training only:
139 fit photos, 55 held-out training photos, and epoch 6 selected. The route
could not produce a Step 2 input for 23/217 training photos, 6/47 validation
photos, or 5/46 test photos. No fallback was counted as a candidate success.

On the 41 candidate-success test photos, the matched head scored MAE/RMSE
0.12949/0.16918, versus 0.14427/0.17294 for v18 on the same OpenCV-resized
inputs. However, among the 39 photos also completed by the exact Pillow-resize
YOLO route, candidate MAE/RMSE were 0.13414/0.17322 versus 0.13011/0.17519
for the exact-route v18 reference: MAE worsened by 0.00403, despite a small
RMSE improvement. Validation completed 41/47 with MAE/RMSE 0.13534/0.17401;
the selected checkpoint's candidate-data validation score during training was
0.13596/0.17363. Thus adaptation improves the off-contract candidate relative
to its own unchanged-head result, but does not preserve the reference MAE.

Full-request median/p95 was 132.99/149.19 ms on test and 130.77/150.40 ms on
validation; just 1/46 test and 2/47 validation requests were below 100 ms.
Median process CPU time was 245.0 ms/test request and 253.5 ms/validation
request. Peak RSS was 3.06 GiB on test and 3.07 GiB on validation; peak
accelerator allocated/reserved memory was 0.31/0.32 GiB. On the four test
photos with saved canonical B2 masks, candidate mask IoUs were 0.9405, 0.5865,
0.9435, and 0.8140 (mean 0.8211); this remains a small segmentation subset.
This candidate is not adopted: it still misses
the total-latency target, has preprocessing failures, and worsens MAE against
the exact-resize reference. Scripts, checkpoint, inputs, and per-photo results
are under `/tmp/wave_cv2_lanczos_matched_v1/` and
`/tmp/wave_cv2_lanczos_matched_finetune_v1/`.

### Regression-head adaptation check for the thumbnail-domain YOLO26 route

I generated candidate Step 2 JPEGs with the thumbnail-domain YOLO26 student for
all current splits (216/217 training, 47/47 validation, 46/46 test; one training
photo failed preprocessing). I compared a frozen-backbone head update with a
partial update of EfficientNet-B0 `features[7:]` plus the head. Model selection
used only a capture-date holdout within training: 158 fit photos and 58 held-out
training photos. Both strategies selected epoch 0; their best training-holdout
MAE/RMSE was 0.11328/0.14674, and later epochs worsened it. The selected
checkpoint's 360 tensors are bit-identical to the v18 checkpoint, so no
regression-head change is justified by this data.

For the unchanged v18 head on the candidate input contract, the full-pipeline
results already measured on the same split were: validation 47/47 successful,
MAE/RMSE 0.11983/0.15032, median/p95 37.44/40.14 ms; test 46/46 successful,
MAE/RMSE 0.11821/0.14938, median/p95 37.32/40.04 ms. All 93 requests were under
100 ms. These request timings include decoding, both preprocessing stages,
regression, and flushed prediction-record writes. Median CPU time was 37.84 ms
on validation and 37.67 ms on test. Peak RSS was 2.31 GiB; PyTorch accelerator
allocated/reserved memory was about 0.12/0.15 GiB. On the four test examples
with canonical B2 masks, this student's thumbnail-mask IoUs were
0.941/0.521/0.954/0.829 (mean 0.811; four-image diagnostic only).

This is the strongest measured speed/aggregate-quality candidate so far, but
those labeled splits have been examined repeatedly during development and do
not constitute a fresh independent quality guarantee. Individual predictions
also move relative to the full-resolution reference, even though aggregate
MAE/RMSE are lower. Keep it experimental until a capture-date-held-out labeled
set confirms quality and coverage; no production defaults or checkpoints were
changed. Training artifacts and the generated candidate inputs are in
`/tmp/wave_yolo26_thumbnail_regressor_data_v1/` and
`/tmp/wave_yolo26_thumbnail_matched_regressor_v1/`.

### Native CPU crop-search prototype on the thumbnail candidate

A stage profile of the best thumbnail candidate showed that Step 2 crop
preprocessing took 23.8 ms median across 93 labeled images, while the final
JPEG encode/decode round trip took only 0.38 ms. The crop selector's nested
NumPy candidate search was therefore the stage to target. A temporary C++17
CPython extension rebuilt the same integral-image search and lexicographic crop
ranking, with no input copies for the contiguous boolean masks. The prototype
source and extension are only in `/tmp/wave_crop_search.cpp` and
`/tmp/wave_crop_ext.cpython-314-x86_64-linux-gnu.so`.

Across the same 93 real Step 1/mask pairs, the Python selector measured 22.83
ms median / 24.62 ms p95; the compiled selector measured 1.57/1.69 ms, a 21.25
ms median paired saving. All 93 selected crop boxes matched, and JPEG bytes
matched whenever compared. One returned diagnostic `shape_penalty` differed by
one floating-point unit due to `log` rounding, but it did not change the crop
box. Full-request candidate runs then measured 15.40 ms median / 17.65 ms p95
on validation and 15.81/17.89 ms on test. All 47 validation and 46 test
requests succeeded and stayed under 100 ms. MAE/RMSE and every prediction were
identical to the Python candidate route: validation 0.11983/0.15032 and test
0.11821/0.14938. Peak RSS remained about 2.31 GiB and accelerator
allocated/reserved memory remained 0.12/0.15 GiB.

This is a promising exact-output CPU speedup on the candidate's 224x224 Step 2
contract. It has not been integrated: the prototype hard-codes the measured
crop constants, has only been compared on the current 93 inputs, and the
checkpoint contract hashes the Python crop module. The candidate thumbnail
profile itself also still needs fresh, independent labeled quality evidence.
No production code or checkpoint was changed. Full run records are under
`/tmp/wave_thumbnail_cppcrop_validation_eval/` and
`/tmp/wave_thumbnail_cppcrop_test_eval/`; the differential microbenchmark is
`/tmp/wave_step2_cpp_crop_benchmark.py`.

### Native crop-search check on the exact-resolution YOLO route

To verify the crop-search optimization was not specific to thumbnail inputs, I
reran the full-resolution dual-domain YOLO test route with the temporary native
selector. On the 45 photos that completed in both runs, crop boxes and waviness
predictions matched exactly; MAE/RMSE stayed at 0.12437/0.16732, and the same
one of 46 photos failed Step 2. Full-request median/p95 fell from 174.53/195.56
ms to 159.38/180.12 ms. Two requests were under 100 ms in both runs. Peak host
RSS was 3.16 GiB in the native run versus 3.10 GiB in the Python reference run;
accelerator allocated/reserved memory remained 0.32/0.34 GiB. The speedup is
real and output-preserving, but this exact-resolution route still misses the
latency target by a wide margin. Records are in
`/tmp/wave_yolo26_student_dual_mask_reuse_cppcrop_eval/`.

### Conditional RGB copy removal in exact-resolution Step 1

The full-resolution route's Step 1 RGB standardization created a converted RGB
image even when the cropped image was already RGB. I benchmarked the conditional
conversion locally without changing repository code. Across all 45 successful
test photos, standardized pixels and encoded Step 1 JPEG bytes were identical.
The isolated Step 1 standardization benchmark measured 75.07 ms median / 98.57
ms p95 before and 71.72 / 90.83 ms after; paired median saving was 3.98 ms.

I then ran the combined full request with this conditional conversion and the
previous exact native crop search. Against the same 45 successful rows from the
native-crop-only run, all crop boxes and all regression predictions were exactly
equal (maximum prediction delta 0). Test quality therefore remained MAE/RMSE
0.12437/0.16732 on 45/46 successful photos, with one preprocessing failure.
Median/p95 request latency was 157.73/173.85 ms versus 159.38/180.12 ms in the
paired native-crop-only run; 2/46 requests were under 100 ms in both runs. Peak
RSS was 3.16 GiB and accelerator allocated/reserved memory was 0.32/0.34 GiB.
The small end-to-end gain is lower than the isolated standardization gain and
still leaves this route above target.

In this combined run, median stage times were 46.40 ms decode, 27.60 ms Step 1
YOLO, 69.80 ms Step 1 standardization/JPEG, 5.40 ms mask projection, 3.10 ms
Step 2 crop/JPEG, and 7.21 ms regression plus record write. Step 1
standardization and source decode are the largest measured stages. The next
quality-preserving work should profile decode and standardization at operation
level, and evaluate reductions in allocations/copies with byte- or
pixel-exact parity. This prototype remains temporary; no production code or
checkpoint contract changed. Results are in
`/tmp/wave_yolo26_student_dual_exactcopy_cppcrop_eval/`, compared with
`/tmp/wave_yolo26_student_dual_mask_reuse_cppcrop_eval/`; the isolated
microbenchmark is `/tmp/wave_step1_rgb_copy_benchmark.py`.

### Pillow-SIMD resize backend trial (rejected)

I built Pillow-SIMD 12.1.1.post0 in `/tmp` and compared its Lanczos resize to
Pillow 12.3.0 on the same 45 masked test ROIs. The AVX2 build measured 117.14 ms
median / 145.99 ms p95 per resize, versus 45.55 / 58.65 ms for the installed
Pillow; all 45 resized pixel arrays differed. A complete full-resolution test
run with the SSE4 Pillow-SIMD build completed 45/46 images and measured
213.54 ms median / 242.28 ms p95, with 1/46 below 100 ms. On the shared 45
successful images, waviness MAE/RMSE were 0.12463/0.16769 versus
0.12437/0.16732 for the Pillow 12.3 exact-copy/native-crop reference, and
mean absolute prediction delta was 0.00327. Peak RSS was 3.16 GiB and
accelerator allocated/reserved memory was 0.32/0.34 GiB, essentially unchanged.
This backend is rejected: it was slower in both the isolated resize check and
full request, changed resized pixels, and slightly worsened both test metrics.
No repository dependencies, source code, or checkpoints were changed.

The isolated script and outputs are `/tmp/wave_resize_backend_bench.py`,
`/tmp/wave_resize_backend_pillow.json`, and
`/tmp/wave_resize_backend_simdavx2.json`; the full test output is under
`/tmp/wave_yolo26_student_dual_pillow_simd_cppcrop_eval/`. Pillow-SIMD's own
[project notes](https://github.com/uploadcare/pillow-simd) describe it as a
version-matched Pillow fork with SSE4/AVX2 resize paths; Pillow documents the
tradeoff between speed and resampling fidelity for `reducing_gap` in its
[resize reference](https://pillow.readthedocs.io/en/stable/reference/Image.html).

### Quality-distribution and matched-regressor follow-up for thumbnail-v1

I compared thumbnail-v1 errors against the exact-resolution projected-mask YOLO
route per photo, using only photos successful in both routes. Thumbnail-v1's
aggregate MAE was lower by 0.00044 on validation (47 paired photos) and by
0.00777 on test (45 paired photos). However, candidate absolute error increased
on 25/47 validation photos and 20/45 shared test photos; mean absolute
prediction shifts were 0.0550 and 0.0726, with maxima 0.1928 and 0.2610. The
largest individual error regressions were 0.1176 on validation and 0.2229 on
test. These metrics show that aggregate quality is not uniformly preserved per
photo.

A paired bootstrap over those same fixed samples (20,000 resamples) estimated
MAE differences (thumbnail minus full-resolution) of -0.00044 on validation
with 95% interval [-0.01700, 0.01585], and -0.00777 on test with interval
[-0.03489, 0.01974]. RMSE differences were -0.01052 (interval [-0.02915,
0.00803]) and -0.01900 (interval [-0.05127, 0.01381]). All intervals include
zero, and the splits have been repeatedly inspected during development. The
current data therefore do not establish a statistically reliable quality
improvement or a fresh independent quality guarantee, despite lower observed
aggregate point estimates.

To test whether the thumbnail-v1 input contract benefits from a matched
regressor, I regenerated inputs using the tuned thumbnail-domain student for
216/217 training photos and 47/47 validation photos. Test inputs and labels were
not used. Head-only, partial `features[7:]` plus head, and full-backbone
low-learning-rate fine-tuning all selected epoch 0 on the same 58-photo
capture-date holdout from the training split; the validation result remained
MAE/RMSE 0.11983/0.15032. Thus this dataset provides no evidence that these
conservative regressions recover the per-photo shifts. The candidate remains
under 100 ms on all 47 validation and 46 test requests, but quality cannot yet
be claimed fully preserved under a strict per-photo interpretation. The
candidate-matched training data, logs, and checkpoints are in
`/tmp/wave_yolo26_thumbnail_tuned_student_regressor_data_v1/`,
`/tmp/wave_yolo26_thumbnail_tuned_student_matched_regressor_v1/`, and
`/tmp/wave_yolo26_thumbnail_full_finetune_v1/`.

### Training-only affine calibration probe for thumbnail-v1 (not adopted)

To address systematic candidate output shifts without another image pass, I fit
`prediction = clip(a * raw_prediction + b, 0, 1)` by least squares on the 158
training-fit examples only. The same 58-photo capture-date holdout used for
model selection stayed out of fitting: raw holdout MAE/RMSE were
0.11328/0.14674 and calibrated results were 0.11167/0.14253. The fit selected
`a=1.10412`, `b=-0.02153`. Applying this fixed transform to the already
untouched splits gave validation MAE/RMSE 0.11409/0.14544 (47/47 successes)
and test 0.11680/0.14960 (46/46). Against the uncalibrated candidate,
validation improved by 0.00574 MAE and 0.00488 RMSE; test MAE improved by
0.00141 while RMSE worsened by 0.00022. Paired-bootstrap 95% intervals for MAE
change included zero on both validation [-0.01426, 0.00255] and test
[-0.01002, 0.00700]. This is not conclusive evidence of a reliable quality
gain.

Complete-request median/p95 latency with calibration in the timed path was
38.71/44.10 ms validation and 37.88/41.10 ms test; every request stayed under
100 ms, with zero failures. Peak RSS was about 2.31 GiB and accelerator
allocated/reserved memory 0.12/0.15 GiB. Relative to the full-resolution
projected-mask route on the same splits, observed aggregate MAE/RMSE remained
lower: validation 0.11409/0.14544 versus 0.12027/0.16084; test
0.11680/0.14960 versus 0.12437/0.16732 on the 45 shared successful photos.
Per-photo shifts remain: error increased on 20/47 validation photos and 22/45
shared test photos, despite the lower aggregate MAE. Keep the scalar transform
as an exploratory artifact; it is not adopted as a production correction
without fresh labeled data and a predeclared non-inferiority margin. Scripts,
logs, and complete split outputs are under `/tmp/wave_yolo26_thumbnail_calibration*`
and `/tmp/wave_yolo26_student_thumbnail_calibrated_{validation,test}_eval/`.

### Prospective unlabeled-capture latency and coverage check with Step 2 fallback

The current raw directory has 18 photos absent from `labels.csv` (13 HEIC and 5
JPEG). I created a separate unlabeled manifest and ran the thumbnail-v1 route
on them before labels were added. Its projected Step 2 mask failed on two
photos, including one UUID-named HEIC. A second, independently trained
full-resolution YOLO26 student on the same already-decoded thumbnail produced a
usable Step 1 mask and lower-water crop for both. This fallback ran only after
the primary crop failed; it added 10.84 ms and 10.99 ms on those requests.

With the fallback, all 18/18 photos completed and all 18 stayed below 100 ms:
median/p95/max full-request latency was 28.91/48.53/59.92 ms. Median CPU time
was 29.50 ms. Peak RSS was 2.32 GiB; accelerator allocated/reserved memory was
0.13/0.16 GiB. The full-resolution reference route also completed 18/18, but
its median/p95/max latency was 120.19/183.28/189.73 ms; peak RSS was 2.92 GiB
and accelerator allocated/reserved memory was 0.32/0.34 GiB. The speed and
coverage result is prospective evidence, not a quality result: these 18 images
have no waviness labels yet, and the fallback changes preprocessing on two.
Their capture dates were recoverable from HEIF/JPEG EXIF, but all 18 dates also
appear in the existing train, validation, or test splits; this is an unseen-image
set, not an independent capture-date holdout. Its labels can still provide a
prospective image-level check, but a fresh-date quality gate requires future
captures. Input hashes, model hashes, and prediction records were saved before
labeling in `/tmp/wave_fresh_unlabeled_thumbnail_fallback_predictions/`; the
comparison is against the full-resolution dual-domain YOLO26 route (not the
production B0 baseline) under `/tmp/wave_fresh_unlabeled_fullres_predictions_retry/`,
and the blank label template is `/tmp/wave_fresh_unlabeled_manifest.csv`. Keep
this route experimental until labels are supplied and quality is scored on an
identical cohort, then confirm on future capture dates.

### Labeled-split verification with the fallback model loaded

I reran the fallback-enabled route over the complete labeled validation and test
splits, with both YOLO26 students loaded and the fallback model warmed before
timing. The fallback remained dormant on all 93 labeled images. Predictions
were bit-for-bit identical to the earlier thumbnail-v1 route on 47/47
validation and 46/46 test photos. Validation MAE/RMSE stayed 0.11983/0.15032;
test stayed 0.11821/0.14938. There were no preprocessing failures, and all
requests stayed under 100 ms: validation median/p95 38.07/43.75 ms, test
37.37/43.22 ms. Peak RSS was 2.32 GiB; accelerator allocated/reserved memory
was 0.13/0.16 GiB. Thus loading the fallback does not change the established
labeled-split quality or latency; the only fallback use observed so far is on
the 2/18 unlabeled images. Prediction and summary files are under
`/tmp/wave_thumbnail_fallback_validation_eval_v1/` and
`/tmp/wave_thumbnail_fallback_test_eval_v1/`.

### Contract-matched scratch regressor for thumbnail-v1 (rejected)

The fast route's best aggregate result still used v18, whose recorded contract is
`fast-v1`. I trained a fresh regression head on the thumbnail-v1 Step 2 JPEGs
with an ImageNet-pretrained, frozen EfficientNet-B0 backbone, using 158 training
photos and selecting epoch 12 on the 58-photo capture-date holdout. All 216
available training outputs came from the thumbnail student, and validation
inputs were 47/47 successful. The chosen scratch head scored holdout MAE/RMSE
0.14284/0.18441 and validation 0.12861/0.16098, worse than v18 on the same
thumbnail inputs (holdout 0.11328/0.14674 and validation 0.11983/0.15032). The
validation score did not justify spending the untouched test split, and no
full-request speed run was made for this rejected head. A loadable experimental
checkpoint and training log are under
`/tmp/wave_yolo26_thumbnail_scratch_regressor_v1/`.

The 58-photo regressor holdout was excluded from head fitting, but the thumbnail
YOLO student used to generate its inputs had been trained on the full training
split, including those images. Treat the holdout metric as a head-selection
check, not an independent end-to-end preprocessing holdout. Validation was
not used to select the head, but the existing v18 comparator remains an
input-contract mismatch. This trial does not resolve the need for independent
quality validation of a trained, versioned thumbnail-v1 pipeline.

### HEIF decoder thread-count tuning (exact-pixel microbenchmark)

I benchmarked `pillow_heif` decode thread counts 1, 2, 4, 8, and 16 on the 14
HEIF sources present in the current labeled validation and test splits. Each
setting decoded all 14 files three times in randomized setting order (42
measurements per setting). SHA-256 hashes of the oriented RGB pixel buffers
matched exactly for every image at every setting. Median per-image decode time
was 131.14, 97.72, 71.33, 57.90, and 51.47 ms respectively; p95 was 162.13,
108.28, 78.57, 64.33, and 57.72 ms. At 24 and 32 threads, follow-up randomized
repeats measured medians of 50.68 and 50.32 ms, with p95 of 55.68 and 62.45 ms.
Those settings did not justify increasing the current 16-thread default: 24
threads saved only about 0.8 ms median while raising process CPU time from
199.68 to 219.42 ms per image, and 32 threads increased CPU time further and
had a worse p95.

Because decoder settings preserved the exact RGB bytes, the downstream Step 1
and regression inputs and therefore waviness predictions are unchanged. For
context, the exact-resolution YOLO projected-mask test reference on the 45
successful test images has waviness MAE/RMSE 0.12437/0.16732 and one failure in
46; this decode-only measurement does not change those quality metrics or
establish a complete-request latency result. The thread sweep supports keeping
the current 16-thread setting for single-request latency on this 32-logical-CPU
host. Its higher CPU use may reduce throughput when several images are decoded
concurrently, so deployment concurrency needs its own end-to-end measurement.
No production code or checkpoint was changed.

### Full-request HEIF thread-count follow-up (inconclusive at 24 threads)

I reran the exact-resolution one-pass projected-mask YOLO route on the same
46-photo test split with HEIF decoding set to 24 threads, keeping the crop
selector, OpenCV/PyTorch threads, models, and all other preprocessing fixed.
The 24-thread route completed the same 45/46 photos and its 45 successful
waviness predictions were bit-identical to the 16-thread reference. MAE/RMSE
therefore remained 0.12437/0.16732, with the same one Step 2 crop failure.

The 24-thread run measured 149.1 ms median / 168.2 ms p95 versus 157.7 / 173.8
ms at 16 threads, but paired per-photo latency improved by only 3.0 ms at the
median. Paired decode time improved 2.4 ms, while process CPU time increased by
10.4 ms per request at the median and peak RSS rose from 3.16 to 3.26 GiB. The
per-run median difference is therefore not sufficient to claim a stable
full-request speed gain; 24 threads is not adopted. Only 2/46 requests were
below 100 ms in either run. Results are under
`/tmp/wave_yolo26_student_dual_exactcopy_cppcrop_24threads_eval/` and the
16-thread reference is under
`/tmp/wave_yolo26_student_dual_exactcopy_cppcrop_eval/`.

### Segmentation-held-out capture dates for thumbnail preprocessing

The previous fast-route studies used a thumbnail student trained on the full
training split, so their regression capture-date holdout was not independent
of preprocessing. I rebuilt that holdout from the ADE20K-pretrained YOLO26n
semantic checkpoint. The selected 58-photo set spans 12 capture dates and was
excluded from student training and checkpoint selection. Of the remaining
training photos with teacher masks, 127 were used to train the student and 22
from separate fit dates were used for segmentation checkpoint selection. The
student stopped at epoch 22; best segmentation validation mIoU/pixel accuracy
was 0.809/0.895 on those 22 fit-date images. One of the 58 final holdout photos
had no saved teacher mask, but it was still processed and evaluated for
waviness.

I ran the holdout through the thumbnail student with the unchanged production
v16 regressor and compared against the canonical Step 2 JPEGs with the same
v16 checkpoint. All 58 candidate requests succeeded. Candidate MAE/RMSE were
0.13443/0.17282 versus 0.14538/0.18229 for canonical preprocessing. The
candidate improved absolute error on 29 photos and worsened it on 29; mean
absolute prediction shift was 0.07318, maximum 0.22158. Paired bootstrap 95%
intervals for candidate minus canonical were [-0.03279, 0.01066] for MAE and
[-0.03597, 0.01535] for RMSE, both including zero. These intervals are
descriptive only: an audit of the v16 training manifest found 48 of the 58
photos in v16's 2026-10-15 training split, and seven of the other ten share a
capture date with at least one v16 training photo. Only the three photos from
2026-10-23 are from dates absent from v16 training. On those three, candidate
MAE/RMSE were 0.08620/0.09238 versus 0.13190/0.15108 for canonical v16, but
three examples are too few to establish quality preservation. The segmentation
student was held out by date, but the complete regressor-plus-preprocessor
pipeline was not. This run therefore does not establish a fresh, independent
waviness-quality result.

Complete warm-request latency included thumbnail/source decode, both
preprocessing stages, v16 regression, and a flushed prediction-record write.
Median/p95/max was 35.3/41.9/46.9 ms; all 58/58 were under 100 ms. Median CPU
time was 35.5 ms, peak RSS 2.31 GiB, and accelerator allocated/reserved memory
0.12/0.15 GiB. The candidate preserves the current v16 checkpoint and target
labels; the input resolution and segmentation model change, so this evidence
applies specifically to the tested thumbnail contract. The set comes from
existing training dates and is now consumed for evaluation; it is not a fresh
future-capture confirmation. Keep this route experimental pending new-date
labels and replication. Split metadata and complete per-image results are in
`/tmp/wave_yolo26_capture_date_holdout_student_data_v1/` and
`/tmp/wave_yolo26_clean_capture_holdout_eval_v1/`; the independently trained
student is in `/tmp/wave_yolo26_capture_date_holdout_student_v1/`.

### v16-matched regressor for the fast thumbnail pipeline

The checked-in `.env` selects v16, while prior thumbnail-v1 quality reports
used v18. I therefore evaluated the current production checkpoint on the
thumbnail-v1 Step 2 JPEGs before making any quality claim: unchanged v16 scored
MAE/RMSE 0.14431/0.18785 on validation and 0.14352/0.17279 on test. I then
adapted v16's EfficientNet-B0 from candidate Step 2 JPEGs using only training
labels. Selection used a capture-date holdout within training (158 fit and 58
holdout photos); partial fine-tuning of `features[7:]` plus the regression head
selected epoch 6 with holdout MAE/RMSE 0.13209/0.17573. Validation and test
were not used to select the checkpoint.

The selected checkpoint scored validation MAE/RMSE 0.12194/0.15195 on 47/47
successful images and test 0.12634/0.16225 on 46/46. Both complete-request
runs stayed below 100 ms, including decode, preprocessing, regression, and
prediction-record flush: validation median/p95 was 37.5/51.5 ms and test
37.2/39.8 ms. Peak RSS was about 2.32 GiB and accelerator allocated/reserved
memory was 0.13/0.16 GiB. These runs show the 100 ms target is reachable for
this candidate route with a regressor adapted from the currently configured
v16 checkpoint.

Quality remains uncertain. On the 32 test photos with archived v16 predictions,
the adapted candidate's MAE/RMSE were 0.12512/0.16257 versus 0.13284/0.16162
for the historical v16 route. Paired bootstrap 95% intervals for candidate
minus reference were [-0.04845, 0.03474] for MAE and [-0.05580, 0.05780] for
RMSE; both include zero. Candidate absolute error improved on 19 photos and
worsened on 13, with mean absolute prediction shift 0.1010 and maximum 0.3207.
On all 46 test photos the candidate was slightly worse than v18 on this same
thumbnail contract (v18 MAE/RMSE 0.11821/0.14938). The 32-photo archived v16
comparison is also smaller than the current test set and not an independent
future-capture holdout. Thus the candidate has promising speed and aggregate
metrics, but the current evidence does not establish per-image or statistically
reliable quality preservation. Also, the thumbnail segmenter was trained on
the full training split, including the regression head's 58-photo holdout, so
that holdout selects the regression adaptation but is not an independent
end-to-end preprocessing holdout. It is not adopted. Training artifacts are in
`/tmp/wave_v16_thumbnail_matched_regressor_v1/`; full validation and test runs
are in `/tmp/wave_v16_thumbnail_matched_validation_eval_v1/` and
`/tmp/wave_v16_thumbnail_matched_test_eval_v1/`.

### Skip redundant RGB conversion in Step 1 ROI standardization

The current input decoder and segmentation route provide RGB images, but
`_standardize_eroded_mask` still called `crop.convert("RGB")` after cropping.
I compared that path with a conditional conversion that converts only non-RGB
crops. This leaves the mask, crop, Lanczos resize, JPEG encoding, segmentation
and regression inputs unchanged. On the 45 successful full-resolution dual-YOLO
test examples, all 45 standardized pixel arrays and Step 1 JPEG byte streams
matched exactly. A randomized paired preprocessing check, including Step 1 JPEG
encoding, measured 77.55 ms median for the current path and 73.24 ms for the
conditional path, with a -4.39 ms median paired delta; peak process RSS was
3.83 GiB in this isolated probe.

I then ran the full request A/B on the same 46 test photos, alternating method
order per photo and including decode, YOLO segmentation, standardization,
Step 1 JPEG round trip, Step 2 projected-mask crop and JPEG round trip,
regression, and flushed prediction-record writes. Both methods succeeded on
45/46 with the same Step 2 failure. All 45 Step 1 JPEGs and final predictions
were identical; both therefore scored MAE/RMSE 0.12437/0.16732 on the same
successful cohort. The existing conversion measured 159.53 ms median / 194.01
ms p95; the conditional path measured 156.06 / 182.57 ms. However, the median
paired end-to-end saving was only 0.23 ms, and requests under 100 ms changed
from 2 to 1. Median process CPU time was 269.4 ms versus 267.0 ms; peak RSS was
3.21 GiB and accelerator allocated/reserved memory was 0.31/0.32 GiB. I adopted the conditional conversion to avoid that redundant RGB buffer while
preserving the fallback for non-RGB callers. The full-request comparison did not
show a robust wall-time gain, so this is recorded as an exact-output allocation
reduction, not as a demonstrated latency improvement; it does not bring the route
close to the 100 ms target. Results and the temporary benchmark are under
`/tmp/wave_rgb_convert_full_ab_v1/` and
`/tmp/wave_rgb_convert_copy_probe.py`.

### Full-request HEIF decoder CPU/latency tradeoff (8 versus 16 threads)

I compared `pillow_heif` decoder settings of 8 and 16 threads in alternating
order on the same 46-photo full-resolution dual-YOLO test cohort. Each timed
request included image decode, segmentation, mask/crop preprocessing, both JPEG
round trips, regression and flushed prediction-record output. Decoded RGB hashes
and Step 1 JPEG bytes matched on all 45 shared successful requests; predictions
were identical. Both settings had the same single Step 2 failure and therefore
identical success-cohort MAE/RMSE of 0.12437/0.16732.

At 8 threads, median/p95 full-request latency was 189.44/204.83 ms, versus
183.15/199.12 ms at 16 threads; the paired median latency penalty for 8 threads
was 6.81 ms. Median process CPU time fell from 300.8 ms at 16 threads to
281.0 ms at 8 threads. Peak RSS was 3.20 GiB and accelerator allocated/reserved
memory was 0.31/0.32 GiB. Neither setting met 100 ms in this run. Keep 16 threads
for the single-request latency objective; 8 threads is a CPU-saving option only
when lower per-request CPU use is preferred over response time. No default was
changed. Full per-image records are in `/tmp/wave_heif_threads_full_ab_v1/` and
the temporary runner is `/tmp/wave_heif_threads_full_ab.py`.

### PyTorch inter-op thread cap on the full request path

Step 7 sets PyTorch intra-op threads to 1, while the process default on this
host is 32 inter-op threads. I compared inter-op settings of 1 and 32 using two
fresh benchmark processes per setting, with run order 1, 32, 32, 1. Each process
warmed the YOLO26 and v18 models before timing the same 46-photo full-resolution
test set. The pipeline path, HEIF decoder setting (16), OpenCV threads (1),
checkpoint, and outputs were otherwise fixed.

Both settings had one Step 2 failure per run. Across the 90 successful
predictions per setting, MAE/RMSE were identical at 0.12437/0.16732; all 45
shared per-image predictions were bit-identical. Inter-op 1 measured
150.36 ms median / 184.05 ms p95 and 271.3 ms median process CPU time. Inter-op
32 measured 150.60 / 177.84 ms and 269.7 ms CPU time. The paired per-image
median latency difference was 2.63 ms in favor of 32 threads, too small and
inconsistent with the aggregate medians to justify changing the default. Peak
RSS was 3.10 GiB at 1 thread and 3.08 GiB at 32; accelerator
allocated/reserved memory was 0.31/0.32 GiB for both. Three requests per setting
were below 100 ms across the two repeats; neither setting approaches the target
reliably. Keep the current inter-op default. Results are in
`/tmp/wave_torch_interop_full_ab_v1/`; the runner is
`/tmp/wave_torch_interop_full_ab.py`.

### Audit of the currently configured v16 prediction path

The checked-in `.env` selects the v16 checkpoint, whose profile is the canonical
768-pixel overlapping SegFormer path. I ran the repository benchmark on the
labeled HEIC `IMG_7238.HEIC` with a fresh decode and the default settings. This
is a one-image path audit, not a quality cohort. The cold request took 11.94 s;
the following warm request took 11.38 s. Step 1 used 35 overlapping tiles and
took 10.87 s in the warm request (9.63 s inside tile model calls); the CPU
standardization was 59.5 ms. Decode was 95.0 ms, Step 2 segmentation 295 ms,
crop selection 22.2 ms, and regression 5.4 ms. The full-resolution float32
logit accumulator was 6.81 GiB. Process startup through the cold first photo was
16.1 s. The one warm prediction was 0.3513 for label 0.50 (absolute error
0.1487); this single image is not a task-quality estimate. This benchmark did
not capture process CPU time or peak RSS.

This establishes that the current configured route is far from the 100 ms
target: Step 1 SegFormer tile inference alone exceeds it by roughly two orders
of magnitude. The recent 150 ms full-resolution YOLO figures and 35–40 ms
thumbnail figures are separate experimental routes, not current `.env`
prediction timings. Reaching the target therefore requires a different
segmentation execution path and a quality gate on the same labeled cohort; the
small copy and thread adjustments cannot close this gap. The benchmark report
is `/tmp/wave_current_v16_configured_path_probe/20261028T161549979780Z/`.

### HEIC hardware-decoding feasibility check (FFmpeg VAAPI, stopped)

Because source decode is about 45–95 ms in the measured routes, I checked
whether the AMD VAAPI device could decode the iPhone HEIC directly. FFmpeg
could initialize VAAPI on `/dev/dri/renderD128`, but its MOV/HEIF demux exposed
the selected 4032 x 3024 still image as a set of 512 x 512 HEVC item streams;
the tested stream produced a single 512 x 512 frame, not the assembled,
EXIF-oriented RGB image returned by Pillow-Heif. This is not a compatible input
path and was not timed as a candidate or evaluated for quality. No code or
runtime default changed. A future hardware-decoder attempt would need to
reconstruct HEIF grid-item metadata and orientation, and prove decoded-output
and task-quality parity before benchmarking latency.

### One full-image 768 SegFormer pass instead of overlapping tiles (rejected for the target)

To test whether the configured v16 checkpoint could avoid its 35-tile Step 1
scan, I ran the repository's full prediction benchmark on all 47 validation and
46 test images with the same v16 regressor, source files, 768-pixel processor,
Step 2 settings, and float32 precision, changing only Step 1 to one full-image
pass (`--no-tiles`). This changes segmentation geometry and is therefore a
candidate contract, not an equivalent implementation.

The candidate completed 46/47 validation and 43/46 test images; four failed
because Step 2 could not find a reliable lower-water crop. Warm complete-request
latency was 863.9 ms median / 892.3 ms p95 on validation and 869.2 / 903.3 ms on
test; none of 89 successful requests was under 100 ms. Decode median was 78 ms,
Step 1 segmentation 370 ms, Step 1 standardization 77 ms, Step 2 segmentation
285 ms, crop selection 28 ms, and regression 6 ms. The benchmark did not record
peak RSS, accelerator peak allocation, or process CPU time for this run.

On successful images, validation waviness MAE/RMSE were 0.13547/0.17529
(46/47); test values were 0.12476/0.14983 (43/46). For the 32 test photos with
archived v16 predictions, the candidate's MAE/RMSE were 0.12297/0.13871 versus
0.13284/0.16162 for archived v16. Candidate absolute error improved on 17
photos and worsened on 15; mean absolute prediction shift was 0.07286, maximum
0.31744. The paired bootstrap 95% interval for MAE change was [-0.04439,
0.02188], including zero. The archived comparison is a repeatedly examined
subset, and the candidate has additional crop failures; these metrics do not
establish independent quality preservation. It is not adopted: the latency is
still about nine times the target and the failure/quality evidence is not
conclusive. The full benchmark is in
`/tmp/wave_v16_single_pass_768_eval/20261028T162104388746Z/`.

### `torch.compile` feasibility probe for full-resolution YOLO26 (stopped)

I tested PyTorch compilation on the existing 640-pixel full-resolution YOLO26
student using the same `IMG_7238.HEIC` input, decoded once to oriented RGB.
The model, device (Radeon 8060S / ROCm 10.1), image size, precision, and
postprocessing were unchanged. After one eager warm-up, three eager `predict`
calls took 32.08 and 31.52 ms on the last two calls. Replacing the model's
forward module with `torch.compile(model.model)` took 40.95, 36.19, and 38.55 ms
for its first three calls. All three compiled semantic masks were pixel-identical
to the eager mask (3,024 x 4,032; 4,045,387 water pixels).

This is a single-image model-stage feasibility check, not a full-request or
labeled-split evaluation; waviness quality is therefore unknown. The compiled
forward was slower after warm-up and did not meet the preliminary speed gate,
so I stopped before spending time on complete prediction runs. No code,
checkpoint, or default changed. The temporary probe is
`/tmp/wave_compile_probe.py`.

### Full-request A/B of direct HEIC decoding (rejected)

I compared `pillow_heif.open_heif(...).to_pillow()` plus in-place EXIF
orientation against the shared `load_rgb_image` path in randomized order on
the 93-image labeled validation/test cohort. The full request used the
640-pixel YOLO26 student, v18 regressor, projected Step 1 mask, native crop
selector, both JPEG round trips, and flushed prediction records. The direct
decoder was used only for HEIC; JPEG continued through the shared loader.

Validation quality and latency were 0.12027/0.16084 MAE/RMSE and 155.9 ms
median / 171.3 ms p95 for the shared loader, versus the exact same quality and
154.0 / 174.5 ms for direct decode. Test quality was likewise identical at
0.12437/0.16732, with 154.0 / 178.3 ms for the shared loader and 159.5 / 183.9
ms for direct decode. Both paths had the same coverage: 47/47 validation and
45/46 test. Predictions, full-resolution mask pixel counts, and crop boxes
matched on all 92 successful pairs; one test image failed Step 2 in both. The
paired median latency change was 3.87 ms slower with direct decoding. Median
process CPU time was about 273 ms for both paths; mixed-run peak RSS was
3.24 GiB and PyTorch allocated/reserved memory was 0.32/0.34 GiB. Direct decode
met the 100 ms limit on 2/93 requests versus 3/93 for the shared loader. The
isolated decode saving did not survive the full request, so this optimization
is rejected and no default changed. Full paired rows and summary are under
`/tmp/wave_direct_heif_full_ab/`; the temporary runner is
`/tmp/wave_direct_heif_full_ab.py`.

### FP16 GPU erosion on the full-resolution YOLO26 route (rejected)

I compared the current OpenCV CPU erosion with the previously exact-checked
separable FP16 GPU max-pool erosion on the same 93 labeled validation/test
photos. The 640-pixel YOLO26 student, v18 regressor, decode, standardization,
projected mask, crop selector, JPEG round trips, and flushed output writes were
unchanged. Calls were randomized across photos with alternating method order.
All 92 successful pairs produced the same full-resolution masks by pixel
count, crop boxes, and bit-identical waviness predictions; both methods had
47/47 validation successes and the same 45/46 test successes and Step 2
failure. Thus validation MAE/RMSE remained 0.12027/0.16084 and test remained
0.12437/0.16732.

The CPU route measured 153.1 ms median / 173.9 ms p95 across all successful
requests. GPU erosion measured 162.9 / 183.4 ms. The paired median latency
change was 6.88 ms slower with GPU erosion. Median process CPU time increased
from 272.0 to 278.7 ms per request; mixed-run peak RSS was 3.25 GiB and
PyTorch allocated/reserved memory was 0.32/0.34 GiB. One request in each route
was under 100 ms. The transfer and GPU operation overhead outweighed the
erosion savings in this YOLO configuration, so this route keeps CPU erosion.
No production code or defaults changed. Full paired rows and summary are under
`/tmp/wave_yolo_gpu_erosion_full_ab/`; the temporary runner is
`/tmp/wave_yolo_gpu_erosion_full_ab.py`.

### Keep the YOLO semantic mask on-device through erosion (inconclusive)

The prior GPU-erosion trial copied YOLO's 12 MB semantic mask to host RAM and
then copied it back to the GPU for erosion. Inspection showed that
`result.semantic_mask.data` is already a 3,024 x 4,032 `uint8` tensor on
`cuda:0`. I tested keeping it there, thresholding and applying the same FP16
separable max-pool erosion before transferring the eroded mask once to host.
The comparison retained the 640-pixel YOLO26 student, v18 regressor, full
source resolution, projected mask, native crop selector, both JPEG round trips,
and flushed prediction records.

Across two independent randomized-order runs on all 47 validation and 46 test
photos, both methods had 47/47 validation successes and the same 45/46 test
successes. Eroded-mask hashes, crop boxes, and predictions matched on every
successful pair (92/92), leaving validation MAE/RMSE at 0.12027/0.16084 and
test at 0.12437/0.16732. In the first run, the GPU candidate's paired median
latency saving was 3.06 ms; in the second, it was 0.28 ms slower. Across 184
paired request observations, mean saving was 0.77 ms, with a per-image paired
bootstrap 95% interval of [-0.72, 2.22] ms. This does not establish a stable
end-to-end speed improvement. The GPU erosion stage itself was about 0.2–0.3
ms slower at the median, despite avoiding the extra host-to-device copy in the
earlier trial. Mixed-run peak RSS was 3.26–3.28 GiB and PyTorch
allocated/reserved memory was 0.32/0.34 GiB. Keep the existing CPU erosion;
the output-equivalent on-device candidate is not adopted without a repeatable
latency gain. Full outputs are under `/tmp/wave_yolo_gpu_in_place_erosion_full_ab/`
and `/tmp/wave_yolo_gpu_in_place_erosion_full_ab_repeat2/`.

### GPU-side bounding box and ROI mask transfer (rejected)

I extended the on-device mask path to compute its foreground count and tight
bounding box on the GPU, then transferred only the eroded mask ROI to the CPU
for the existing full-resolution source crop, exact Lanczos standardization,
Step 2 projection and regression. This preserves all source pixels and the
existing model/preprocessing geometry; it only changes where the mask bounds
are computed and how many mask bytes cross to the host.

Across 47 validation and 46 test photos, the GPU-ROI and CPU reference routes
had identical eroded-mask hashes, Step 1 JPEGs, crop boxes and predictions on
all 92 successful pairs. Both had 47/47 validation successes and the same
45/46 test successes, so validation MAE/RMSE remained 0.12027/0.16084 and test
remained 0.12437/0.16732. However, GPU mask preparation measured 6.90 ms median
versus 5.14 ms for the CPU path, and the paired full-request median was 2.31 ms
slower. Validation/test request medians were 157.0/164.6 ms for GPU-ROI versus
154.9/159.8 ms for CPU. Peak mixed-run RSS was 3.25 GiB and PyTorch
allocated/reserved memory was 0.31/0.32 GiB. The added GPU reductions and
host synchronization cost more than transferring the smaller ROI saves; this
candidate is rejected. No production code changed. Full artifacts are under
`/tmp/wave_yolo_gpu_roi_mask_full_ab/` and the runner is
`/tmp/wave_yolo_gpu_roi_mask_full_ab.py`.

### Substage profile of full-resolution Step 1 standardization

To separate high-resolution standardization costs, I replayed its operations on
the 45 successful test photos using the saved full-resolution YOLO masks and
original RGB sources. After warming each source, I recorded three repeats per
photo for erosion, bbox/count, source crop, NumPy/ROI preparation, mask
application, Pillow image construction, Lanczos resize, square padding, JPEG
encode and JPEG decode (135 observations per stage).

Median times were 0.75 ms erosion, 0.84 ms bbox/count, 2.20 ms crop,
4.07 ms array/ROI preparation, 3.67 ms mask application, 3.37 ms Pillow image
construction, 45.75 ms Lanczos resize, 0.04 ms padding, 0.23 ms JPEG encode,
and 0.30 ms JPEG decode. The stage medians do not add exactly to a request
median, and this operation-level profile excludes source decode and model
inference. It identifies the exact high-quality Pillow Lanczos resize as the
dominant remaining Step 1 CPU operation; JPEG round trips are too small to
matter. A different resampler would change pixels and remains a quality-gated
candidate, while rearranging copies alone cannot plausibly recover the roughly
50 ms still needed for the full request to reach 100 ms. Profile script:
`/tmp/wave_step1_standardize_stage_profile.py`.

### Lanczos `reducing_gap` sweep and full-request quality check

Because Pillow's high-quality resize dominates standardization, I compared
`reducing_gap` values 3.0, 2.0, 1.5 and 1.0 with the default Lanczos resize on
the same 45 successful full-resolution test crops. This changes only the
resampling implementation; source dimensions, YOLO masks, crop geometry,
JPEG quality and regressor were held fixed. Resize-only medians were 45.62 ms
for default Lanczos and 7.03, 5.73, 4.48 and 3.79 ms for the four settings.
At gap 3.0, 44/45 Step 1 JPEGs changed and mean absolute resized-pixel
difference was 0.165 levels. At gap 1.0, all 45 JPEGs changed and mean
absolute resized-pixel difference was 1.534 levels. The sweep ranks runtime
only; the full labeled results below are the quality check.

I ran paired complete requests for gap 3.0 and the fastest screened setting,
gap 1.0, on
all 47 validation and 46 test images using the 640-pixel YOLO26 student, v18
regressor, full-resolution source, projected Step 1 mask, native crop search,
both JPEG round trips and flushed prediction output. Each setting had the same
coverage as its default-Lanczos baseline: validation 47/47 and test 45/46.
The same one test image failed Step 2 for every setting.

At gap 3.0, full-request median/p95 was 113.7/132.4 ms across 92 successes,
versus 155.1/177.9 ms for default Lanczos; 17/93 versus 3/93 requests were
under 100 ms. Median process CPU time fell from 269.2 to 228.3 ms, while mixed
peak RSS / PyTorch allocated/reserved accelerator memory was 3.24 / 0.32/0.34
GiB. Validation MAE/RMSE improved from 0.12027/0.16084 to 0.11650/0.15530,
but test worsened from 0.12437/0.16732 to 0.12974/0.17006. On the 45 common
test successes, paired MAE increased by 0.00537 (bootstrap 95% interval
[0.00070, 0.01057]); errors worsened on 21 photos and improved on 23. The
training-only capture-date adaptation selected the unchanged v18 checkpoint
(epoch 0) for both head-only and partial-backbone strategies; it did not
recover the training holdout, so no adapted head was justified.

At gap 1.0, full-request median/p95 was 113.4/127.3 ms across 92 successes,
versus 158.2/177.5 ms for its paired baseline; 18/93 versus 1/93 requests were
under 100 ms. Median process CPU time fell from 270.8 to 226.4 ms, with mixed
peak RSS / accelerator memory at 3.25 / 0.32/0.34 GiB. Validation MAE/RMSE
improved from 0.12027/0.16084 to 0.10889/0.14642, while test worsened from
0.12437/0.16732 to 0.13504/0.17744. On the 45 common test successes, paired
MAE increased by 0.01067 (bootstrap 95% interval [-0.00001, 0.02269]); errors
worsened on 25 photos and improved on 20. This setting is also too slow for
the target and shows a larger test-quality risk.

Both settings reduce CPU work substantially but produce test-quality hits and
still leave request medians above 100 ms. Neither is adopted. The current
validation/test splits have been repeatedly examined during candidate work,
so these results remain exploratory rather than an independent quality
guarantee. The full paired records are under
`/tmp/wave_lanczos_gap3_full_ab/` and `/tmp/wave_lanczos_gap1_full_ab/`; the
resize screening is `/tmp/wave_lanczos_gap_sweep.py`. Gap-3 training inputs,
capture-date holdout results and checkpoint are under
`/tmp/wave_yolo26_gap3_regressor_data_v1/` and
`/tmp/wave_yolo26_gap3_matched_regressor_v1/`.

## Quality reporting for future experiments

Starting with experiments run from this point forward, record prediction quality
for every candidate alongside speed, including candidates that miss the latency
target. For labeled candidates, report validation and test MAE/RMSE, sample
counts and failures; compare with the paired baseline on the same successful
images, and report prediction changes or paired uncertainty when available.
Record quality before selecting or rejecting a candidate, and include the best
quality candidates in each experiment's conclusion even when none reaches the
speed target. Keep speed reporting complete-request latency (median/p95 and
under-target count) plus resource use where measured. Do not backfill earlier
experiments; this policy applies going forward.

### Lanczos `reducing_gap=6` full-request and matched-head check

This is the first candidate covered by the forward-looking quality-reporting
rule. On the same 47 validation and 46 test requests, gap 6 reduced full-request
median/p95 from 156.5/181.2 ms to 126.7/143.0 ms; 8/93 requests were under
100 ms versus 1/93 for the paired baseline. The median CPU time was 240.3 ms
versus 271.6 ms, with mixed peak RSS / accelerator allocated-reserved memory
of 3.246 / 0.321/0.336 GiB. Coverage remained 47/47 validation and 45/46 test
for both paths.

Using 214 successfully generated training rows, I selected between a head-only
fine-tune and partial-backbone fine-tune using a 57-image capture-date holdout
from training only. Head-only selected epoch 2 and improved holdout MAE/RMSE
from 0.11148/0.14998 to 0.11076/0.14981. Partial-backbone tuning selected
epoch 0 (the original v18 weights). The selected head's validation MAE/RMSE
was 0.11868/0.15727 over 47 images. On the untouched test split, its MAE/RMSE was 0.12945/0.16853 over the same
45 successful images. That is worse than both the unadapted gap6 head
(0.12647/0.16995) and the default-Lanczos baseline (0.12437/0.16732), so the
holdout gain did not transfer to test quality. The matched head was scored on
the candidate test inputs but was not separately rerun through the full latency
harness; the gap6 end-to-end speed figures above are for the unadapted v18
head. Neither head is adopted. Training data,
checkpoint and result are under `/tmp/wave_yolo26_gap6_regressor_data_v1/` and
`/tmp/wave_yolo26_gap6_matched_regressor_v1/`; full-request records are under
`/tmp/wave_lanczos_gap6_full_ab/`.

## Performance principles review and OpenCV resizer candidate

I reviewed primary framework documentation against the measured full-request
profile. PyTorch recommends `inference_mode()` together with `model.eval()`;
the current candidate already uses both. PyTorch also warns that pinning memory
on the main thread makes an additional blocking copy, so pinning a single
224x224 request tensor is unlikely to help. Nonblocking transfer is useful when
there is independent work to overlap, and requires careful synchronization for
pinned buffers. The current path has dependent segmentation, CPU crop/resize,
and regression stages, so a transfer-only flag cannot remove those dependencies.
PyTorch further identifies host/device copies and `.item()` as possible sync
points; the scalar result must ultimately reach the CPU for the request output.
Sources: [PyTorch performance tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html),
[PyTorch inference_mode](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html),
[PyTorch transfer and pinning guide](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html).

Pillow documents that a smaller `reducing_gap` trades resampling fidelity for
speed. Our prior gap sweep verified this tradeoff on the actual labeled task:
all tested settings missed the latency target, and the faster settings damaged
test quality. I screened OpenCV `INTER_LANCZOS4` as another CPU implementation
on the same 45 full-resolution test crops. Its resize median/p95 was
2.80/3.02 ms, compared with Pillow Lanczos 48.92/62.69 ms, but all 45 outputs
changed and mean absolute resized-pixel difference was 4.23 intensity levels.
The authoritative CV interpolation API is described in the
[OpenCV resize documentation](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html).

A paired complete-request run on 47 validation and 46 test photos measured
OpenCV Lanczos4 at 114.4 ms median / 134.6 ms p95 over 92 successes; 13/93
requests were below 100 ms. Its baseline in the same interleaved run was
154.9/177.6 ms, with 2/93 under 100 ms. Median process CPU fell from 268.6 to
230.2 ms. Coverage and failures were identical (47/47 validation, 45/46 test).
Mixed peak RSS / accelerator allocated-reserved memory was 3.24 / 0.321/0.336
GiB.

Quality did not hold: validation MAE/RMSE changed from 0.12027/0.16084 to
0.12759/0.17823, and test changed from 0.12437/0.16732 to 0.14570/0.18448.
On the 45 paired successful test rows, mean absolute-error increase was
0.02134; errors worsened on 27 photos and improved on 18 (bootstrap 95%
interval [-0.00838, 0.05156]). Crop boxes remained the same on only 44/92
paired rows. Despite the 40.6 ms median speed gain, this resampler is rejected
because it fails the quality requirement and still misses the 100 ms target.
No production code changed. Screening script:
`/tmp/wave_opencv_lanczos_screen.py`; paired run:
`/tmp/wave_opencv_lanczos_full_ab.py`; records:
`/tmp/wave_opencv_lanczos_full_ab_v2/`.

The measurements support these priorities for the next rounds: keep the current
canonical preprocessing and model outputs as the reference; optimize CPU
standardization through exact-output implementation work or a changed method
that passes validation and test quality gates; measure HEIC decode separately
because it remains another large fixed cost; and avoid adding copies, manual
pinning, synchronization, or thread pools unless an end-to-end paired run shows
a gain. Any candidate must report quality, full-request latency, failures and
CPU/memory use together. A fast preprocessing microbenchmark alone is not
sufficient evidence for adoption.

### Standard downsampling kernel screen and paired checks

I screened Pillow Bicubic, Pillow Hamming, OpenCV Area, OpenCV Linear and OpenCV
Cubic on the same 45 crops against Pillow Lanczos. Pillow 12.3.0 in this
environment does not provide the newer MKS filters documented by current Pillow
dev docs. Resize-only median time / mean absolute pixel difference were:
Pillow Lanczos 48.94 ms / 0.00; Bicubic 34.55 ms / 0.61; Hamming 18.99 ms /
1.21; OpenCV Area 14.82 ms / 1.22; OpenCV Linear 0.12 ms / 4.15; and OpenCV
Cubic 0.23 ms / 4.26. These pixel comparisons are only a screen; task quality
was measured on the paired full request for Area and Bicubic.

OpenCV Area reduced full-request median/p95 from 154.9/178.1 ms to
128.6/150.5 ms, with 9/93 requests under 100 ms versus 2/93. Median CPU time
fell from 268.9 to 242.9 ms. Validation MAE/RMSE improved from 0.12027/0.16084
to 0.10859/0.14848, while test worsened from 0.12437/0.16732 to
0.13383/0.17438. On 45 common test successes, paired MAE increased by
0.00946 (bootstrap 95% interval [0.00005, 0.01954]); error worsened on 25 and
improved on 20. Same coverage: 47/47 validation, 45/46 test. Peak RSS /
accelerator memory was 3.25 / 0.321/0.336 GiB. Rejected for the test-quality
regression and because it remains above target.

Pillow Bicubic reduced full-request median/p95 from 156.4/174.8 ms to
139.7/160.6 ms, with 4/93 requests under 100 ms versus 0/93. Median CPU time
fell from 272.5 to 253.4 ms. Validation MAE/RMSE changed from 0.12027/0.16084
to 0.11554/0.15404; test changed from 0.12437/0.16732 to 0.12906/0.16903.
The paired test MAE increase was 0.00469 (bootstrap 95% interval
[-0.00070, 0.00999]); error worsened on 27 common successes and improved on
18. Coverage stayed 47/47 and 45/46; peak RSS / accelerator memory was
3.24 / 0.321/0.336 GiB. The test result does not establish a preserved-quality
gain, and latency remains above target, so Bicubic is not adopted.

These standard-kernel candidates show that pixel closeness alone is not a
quality guarantee, while the more aggressive filters lose task quality. The
fastest candidate so far remains OpenCV Lanczos4 at 114.4 ms median with test
MAE 0.14570; it is rejected. Pillow Bicubic is the mildest measured speed
change but still has an unfavorable test point estimate and misses the target.
No resampler change has passed both quality and latency requirements. Screen:
`/tmp/wave_resampler_screen_v2.py`; paired results:
`/tmp/wave_opencv_area_full_ab/` and `/tmp/wave_pillow_bicubic_full_ab/`.

### Channels-last memory format for the EfficientNet regression head

I tested PyTorch's computer-vision channels-last layout for the v18
EfficientNet-B0 regressor, leaving all decoded RGB pixels, segmentation masks,
water crops, Pillow Lanczos outputs, and JPEG round trips unchanged. The paired
full request covered 47 validation and 46 test photos, with one shared Step 2
failure. All 92 successful pairs produced bit-identical predictions, masks and
crop boxes. Validation MAE/RMSE remained 0.12027/0.16084 and test remained
0.12437/0.16732.

Across 92 successful requests, baseline median/p95 was 155.9/179.4 ms and
channels-last was 157.3/178.4 ms; the median paired latency saving was only
0.88 ms. Under-100 counts were 3/93 for baseline and 1/93 for channels-last.
Median CPU time was 271.7 versus 269.2 ms. With both head layouts loaded for
the paired run, peak RSS was 3.25 GiB and accelerator allocated/reserved
memory was 0.337/0.350 GiB. This layout is output-exact but does not improve
complete-request speed on the current Radeon/ROCm setup, so it is not adopted.
Runner and records are under `/tmp/wave_channels_last_full_ab.py` and
`/tmp/wave_channels_last_full_ab/`.

### Parallel destination-strip Lanczos resize with Pillow's source box

The exact Pillow resize was still a major CPU stage, so I tested splitting only
the destination image into eight horizontal strips and resized each strip in a
reusable eight-worker thread pool. Every strip used Pillow's Lanczos filter and
a source `box` that maps its output coordinates back to the original source
crop. Pillow's C resampling code releases its image section around the pixel
loops and computes separable horizontal and vertical passes; this makes the
same filter parallelizable across independent output columns. Reference:
[Pillow 12.3.0 Resample.c](https://github.com/python-pillow/Pillow/blob/12.3.0/src/libImaging/Resample.c).

On all 45 successful test crops, the threaded strip resize had 10.88 ms median /
14.05 ms p95 versus 48.94/63.31 ms for a whole-image Pillow Lanczos call.
Only 11/45 resized arrays differed at all; mean absolute difference averaged
0.000022 intensity levels per image, with the largest per-image mean difference
0.000158. The differences are extremely sparse single-level rounding changes
from the strip's floating-point source-box origin. This is not claimed as
bit-identical processing; the following paired task-quality check is the gate.

On 47 validation and 46 test requests, the full-resolution path completed
47/47 validation and 45/46 test in both arms. Baseline full-request median/p95
was 157.62/177.21 ms; the threaded Lanczos candidate was 119.64/138.88 ms,
with 12/93 requests under 100 ms versus 1/93 for baseline. The candidate's
validation MAE/RMSE was 0.120275/0.160846 versus 0.120267/0.160841 baseline.
Test MAE/RMSE was 0.124335/0.167193 versus 0.124367/0.167325 baseline on the
same 45 successful test rows. Paired test MAE change was -0.000032 (bootstrap
95% interval [-0.000149, 0.000105]); only three test errors worsened while
seven improved, and 35 predictions were unchanged. Mean absolute prediction
shift was 0.000156, maximum 0.002213. These results show no measurable quality
hit on the repeatedly used validation/test splits, but they are not a fresh
independent quality guarantee.

Median stage times for baseline versus tiled Lanczos were 49.81/50.27 ms
source decode, 27.10/27.67 ms YOLO, and 81.11/44.24 ms for the remaining
post-YOLO stages. Median CPU time rose from 271.82 to 284.47 ms because the
resize uses more cores; peak RSS was 3.28 GiB and accelerator allocated/reserved
memory was 0.321/0.336 GiB. The full-request median improved by 29.11 ms paired,
but remains 19.64 ms above the 100 ms target. This is the strongest
quality-preserving end-to-end speed result so far; do not adopt it as the final
route until speed/resource behavior is measured on the intended serving
hardware and fresh-date quality is checked. Runner and records are in
`/tmp/wave_pillow_tiled_x8_full_ab.py` and
`/tmp/wave_pillow_tiled_x8_full_ab/`; the 45-crop thread screen is
`/tmp/wave_tiled_lanczos_thread_sweep.py`.

### FFmpeg/libheif decoder-thread investigation (rejected as slower)

I checked the system libheif FFmpeg decoder plugin because source HEIC decode
is about 50 ms in the fast route. The system package did not include that
plugin, so I extracted libheif 1.21.2 and FFmpeg development packages into
`/tmp`, built the plugin there, and compared its decoded RGB against
Pillow-Heif. The FFmpeg path assembled the complete 3024x4032 HEIF image and
returned pixel-identical RGB on the probe image. Source inspection showed that
the plugin's `num_threads` option was not applied to FFmpeg's codec context; I
patched only the `/tmp` copy to set `thread_count` and `thread_type`, then
measured 1, 2, 4, 8, 16 and 32 threads. Median warm decode was about 100–103 ms
at 1–2 threads, 110 ms at 4, 121 ms at 8, 139 ms at 16, and 171 ms at 32,
compared with 50–68 ms for the existing 16-thread Pillow-Heif decoder on the
same 12 MP HEIC. The patched FFmpeg output remained byte-identical to
Pillow-Heif. The plugin's source calls `avcodec_find_decoder` and
`avcodec_open2` but does not create a hardware device context, so VAAPI was not
being used by this build. The software FFmpeg path is rejected for this CPU;
no repository code or dependencies changed. The upstream plugin and its stated
HEVC support are documented in the [libheif project](https://github.com/strukturag/libheif).
Artifacts and isolated source/build are under `/tmp/wave_heif_ffmpeg_probe/`
and `/tmp/wave_libheif_source/`.

### HEIF grid-tile decode feasibility (localization path screened out)

The source file `IMG_7776.HEIC` is a 3024x4032 HEIF grid with 6 columns and 8
rows of 512x512 HEVC tiles. I used libheif's tile API to decode each tile
individually. On this host, serial RGB conversion averaged 2.47 ms per tile;
decoding all 48 tiles serially took 120.50 ms. This is slower than the existing
full-image Pillow-Heif decode (about 50 ms), so decoding every tile is not a
candidate. Libheif exposes a 240x320 embedded thumbnail. I also decoded the
first four tiles and matched their RGB arrays against the full decoded image:
the two full interior tiles matched exactly, and both left-edge tiles matched
exactly after applying the grid's 48-pixel crop offset. The complete libheif RGB
decode matched Pillow-Heif byte-for-byte over all 12 MP. This verifies part of
the tile placement and decoder equivalence, not every tile or orientation.

The thumbnail YOLO mask covers the full image width and about 77% of its height,
which intersects 42 of the 48 HEIF tiles. Selective tile decode therefore has
little opportunity to save work and is screened out. The useful path is instead
to decode every original tile concurrently and reconstruct the exact full
resolution source. The probes are
`/tmp/wave_heif_tile_probe.cpp` and `/tmp/wave_heif_tile_match.cpp`; tile timing
is in `/tmp/wave_heif_tile_probe.out`.

### Parallel HEIF grid decode with direct BGR assembly (fast, rejected on v16 test quality)

The initial serial-tile result was misleading for the bundled Pillow-Heif
decoder: calls sharing one libheif context serialized. I changed the isolated
probe to give each worker its own libheif context, decode one tile at a time,
and write each tile directly into a single final BGR buffer. This avoids both a
temporary full set of tile buffers and a second full-image channel-order copy.
Sixteen workers on the bundled libheif 1.23.4 / libde265 decoder reconstructed
all 92 labeled HEIC inputs byte-for-byte against `pillow_heif` RGB output
(after BGR channel order); median decode plus assembly was 19.92 ms and p95
32.54 ms. The one JPEG in the cohort decoded through OpenCV; its oriented RGB
pixels matched Pillow exactly.

I then compared complete requests against the existing full-resolution YOLO26
/ v18 fast route, keeping YOLO resolution, masks, projected Step 2 mask, the
eight-strip Pillow Lanczos resize, JPEG quality, regressor, and flushed result
write fixed. The optimized route feeds the decoder's contiguous BGR array
directly to Ultralytics and converts only the cropped ROI back to RGB. All 92
successful masks, crop boxes, and predictions were bit-identical to the
reference. Both arms had the same 47/47 validation and 45/46 test coverage and
the same one test preprocessing failure.

On this fast-route comparison, the optimized candidate retained validation
MAE/RMSE 0.120275/0.160846 and test 0.124335/0.167193 exactly. Full-request
median/p95 fell from 130.31/149.04 ms to 67.43/74.55 ms; 92/92 successful
requests were under 100 ms versus 2/92 for the reference. The paired median
latency saving was 62.86 ms. Median source-decode time fell from 53.30 to
16.21 ms, YOLO plus mask extraction from 29.29 to 11.97 ms, and remaining
post-mask work from 48.69 to 38.78 ms. Median process CPU time was effectively
unchanged (290.78 versus 291.17 ms); combined-run peak RSS was 3.17 GiB and
PyTorch allocated/reserved memory was 0.321/0.336 GiB.

This is the strongest speed result, but it is not yet an acceptable quality
replacement for the configured v16 route. On the 32 test photos that have
archived v16 inputs, v16 measured MAE/RMSE 0.132839/0.161621 and the fast v18
candidate measured 0.144696/0.185303; candidate absolute error was worse on 15
and better on 17. The paired MAE difference was +0.01186 (bootstrap 95% CI
[-0.03292, 0.05678]). Validation on its 32 archived common photos favored the
candidate, but the test result shows a quality hit and does not pass the
no-regression requirement. The current `.env` remains on v16. Keep this as an
exact decoder/runtime optimization for the v18 fast route, not as an adopted
prediction pipeline. A quality-matched candidate and fresh-date evaluation are
still required before replacing v16. No production code or dependency changed.
The isolated decoder and paired records are under
`/tmp/wave_heif_tiled_decode_bgr_ctx.cpp`,
`/tmp/wave_heif_tiles_x8_full_ab_wheel/`, and
`/tmp/wave_heif_tiled_x8_full_ab.py`.

### v16 head adaptation for the exact full-resolution YOLO route (not adopted)

Because the fast route's MAE gain came with a v16-reference test RMSE hit, I
generated its exact Step 2 JPEGs for the current training split and adapted the
v16 regressor. Preprocessing succeeded on 214/217 training photos. I used a
capture-date holdout of 57 training photos (157 fit photos) and compared
head-only and last-block-plus-head fine-tunes under SmoothL1 and MSE losses.
Selection used holdout RMSE only; the last-block-plus-head SmoothL1 model at
epoch 2 was best, lowering holdout MAE/RMSE from 0.13682/0.18041 to
0.12652/0.17188. On validation, it scored 0.14356/0.18104 MAE/RMSE over 47/47
candidate inputs; on the 32 validation photos with archived v16 inputs it
scored 0.14588/0.19246 versus 0.15674/0.20032 for v16 on its archived inputs.

I then ran the selected checkpoint end-to-end on test. It completed 45/46
photos, with MAE/RMSE 0.13500/0.16371, median/p95 67.56/72.28 ms, and all
45/45 successful requests under 100 ms. On the 32 test photos with archived
v16 inputs, however, the adapted route scored 0.14662/0.17442 versus
0.13284/0.16162 for v16. Its absolute error was worse on 20 photos and better
on 12; paired MAE change was +0.01378 (bootstrap 95% interval
[-0.02940, 0.05502]). The fast v18 head had lower full-test MAE but higher
RMSE, and also failed the archived v16 test comparison. Neither model clears
the no-quality-hit requirement. The adaptation is not adopted. Training data,
checkpoints, and full per-photo records are under
`/tmp/wave_yolo26_fullres_exactdecode_regressor_data_v1/`,
`/tmp/wave_v16_fullres_tiled_regressor_v1/`, and
`/tmp/wave_heif_tiled_v16adapt_test_ab/`.

### v18 adaptation on exact full-resolution YOLO route (rejected at training holdout)

I repeated the same training-date holdout adaptation with the v18 regressor as
initialization, using the exact fast-route preprocessing inputs (157 fit photos,
57 holdout photos). The untouched v18 checkpoint scored holdout MAE/RMSE
0.10988/0.14748. Head-only and last-block-plus-head fine-tuning, with both
SmoothL1 and MSE losses, all worsened holdout error from epoch 1 onward; epoch 0
(the unmodified checkpoint) remained the selected checkpoint in each run. I did
not run these rejected adapters on validation or test. This does not improve
the fast route's quality against the archived v16 test baseline, so no model or
pipeline changes were adopted. Results are in
`/tmp/wave_v18_fullres_tiled_regressor_v1/training_results.json`.

### Batched SegFormer tile inference on the configured v16 route (rejected)

I checked an exact-path optimization on the currently configured 768-pixel,
35-overlapping-tile SegFormer route: concatenate identical preprocessed tile
inputs and run the same model in batches, retaining the same per-tile bilinear
logit resize, full-image accumulation, overlap handling, and argmax. On
`IMG_7238.HEIC` (4032x3024), batch size 1 reproduced the existing segmentation
at a 9.96 s median for two runs. Batch size 2 took 31.37 s per run (two runs)
and used 10.11 GiB peak PyTorch allocation during the sweep; the resulting
class map differed at 2 pixels from batch size 1. Larger batches were stopped
because batch 2 was already slower and not pixel-exact. This is a one-image
microbenchmark, but it is sufficient to reject this batching implementation
for latency and strict mask parity; no end-to-end quality check or production
change was warranted. The isolated experiment is
`/tmp/wave_segformer_tile_batch_sweep.py`.

The general optimization rule remains to measure the full request and system
resources rather than infer gains from GPU use or a microbenchmark: preprocessing
and decode have dominated several routes. PyTorch's inference guidance supports
`model.eval()` plus `torch.inference_mode()` (already used here), and cautions
against unnecessary device synchronization and copies. Thread counts and batch
sizes remain workload-specific and must be benchmarked on this AMD ROCm host;
this experiment shows that batching those 768-pixel tiles is not beneficial.
Primary references: [PyTorch inference mode](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html),
[PyTorch performance tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html),
and [OpenCV threading controls](https://docs.opencv.org/4.11.0/db/de0/group__core__utils.html).

### RMSE-focused v16 adapter for the thumbnail YOLO route (promising, not adopted)

The earlier thumbnail adaptation selected a partial EfficientNet-B0 update using
SmoothL1 and the capture-date holdout's MAE. Its test MAE was better than the
v16 reference point estimate on the 32 archived common photos, while RMSE was
slightly worse. I tested a targeted alternative: train the same last-block-plus-
head adapter on the same 158 fit photos with MSE and select epoch by RMSE on the
same 58-photo training-date holdout. Validation and test were not used to select
the epoch. The unchanged v16 start scored holdout MAE/RMSE 0.14035/0.17823; the
selected MSE epoch 8 scored 0.13419/0.17479. The previous SmoothL1 adapter had
0.13209/0.17573 on this holdout, so the MSE choice trades a small holdout MAE
increase for lower RMSE.

I evaluated the selected model on the complete raw-image route with the same
thumbnail-domain YOLO student and fallback, both preprocessing JPEG round
trips, regression, and flushed prediction record. Validation succeeded on
47/47, with MAE/RMSE 0.11649/0.14810 and median/p95 37.70/40.71 ms; test
succeeded on 46/46, with MAE/RMSE 0.11829/0.15737 and median/p95
37.46/39.35 ms. All 93 requests were below 100 ms. Peak RSS was 2.32 GiB and
PyTorch allocated/reserved accelerator memory was 0.126/0.158 GiB. Median process
CPU time was 37.92 ms on validation and 37.87 ms on test, essentially tracking
request wall time at this single-request concurrency. Against the previous
SmoothL1 adapter on the same route, the MSE model improved 33/46 test
absolute errors; paired test MAE changed by -0.00805 (bootstrap 95% interval
[-0.01468, -0.00099]) and RMSE by -0.00488 (interval [-0.01166, 0.00225]).
That previous run's detailed test artifact was rerun and preserved under
`/tmp/wave_v16_thumbnail_smoothl1_restored_test_eval_v1/`.

The training-selected v18 thumbnail head remains a strong comparator: on all
46 test images it scored 0.11821/0.14938 MAE/RMSE, essentially the same MAE and
lower RMSE than the MSE adapter. On the 32 test photos with archived v16
predictions, the MSE adapter scored 0.11612/0.15496 versus 0.13284/0.16162 for
archived v16. Its paired changes were -0.01671 MAE (95% interval
[-0.05746, 0.02460]) and -0.00666 RMSE (interval [-0.06162, 0.04786]); absolute
error improved on 20 photos and worsened on 12. These are encouraging point
estimates but the intervals are wide, individual predictions move, and this
same labeled cohort has been repeatedly examined. Do not treat it as proof of
zero quality loss or a fresh generalization result. Keep the model experimental
until labeled future captures confirm quality; no production checkpoint or
pipeline setting changed. Artifacts are in
`/tmp/wave_v16_thumbnail_mse_rmse_regressor_v1/`,
`/tmp/wave_v16_thumbnail_mse_rmse_validation_eval_v1/`, and
`/tmp/wave_v16_thumbnail_mse_rmse_test_eval_v1/`.

A separate inference-only blend probe used the two thumbnail heads on the 58
training-date holdout. The v18 head was already better than the adapted v16
head there (MAE/RMSE 0.11328/0.14674 versus 0.13209/0.17573), and the
RMSE-optimal convex blend selected weight 0 for adapted v16. I therefore did
not spend validation/test data on this blend. Probe:
`/tmp/wave_thumbnail_v16_v18_blend_holdout.py` and
`/tmp/wave_thumbnail_v16_v18_blend_holdout.json`.

#### Corrected v16 reference scope

I also evaluated v16 on all current `step-2-final-water-data` files and
initially compared those results with the fast thumbnail route. That comparison
is invalid as a canonical-v16 baseline: the folder's current
`preprocessing-contract.json` identifies the files as `fast-v1` (448-pixel
single-pass SegFormer, FP16, Step 2 resolution 384), whereas v16's archived
October 15 test evaluation used its then-current canonical 768-pixel overlapping
SegFormer inputs. On the 32 shared test photos, current saved Step 2 inputs
shifted v16 predictions by a mean absolute 0.106 (max 0.319) versus the archive.
I exclude this regenerated-input comparison from quality claims. Use archived
v16 predictions for the 32 exact common inputs and wait for a correctly
matched, fresh labeled cohort for an independent no-regression conclusion.
The audit is `/tmp/wave_canonical_v16_full_split_eval_detailed.py` and
`/tmp/wave_canonical_v16_full_split_comparison_detailed.json`.

### Compiled crop search paired with the RMSE-focused v16 thumbnail adapter

I reran the compiled Step 2 crop selector against the Python implementation on
the RMSE-focused v16 adapter and the thumbnail-domain YOLO26 input route. The
runner used the same JPEG draft / embedded HEIF thumbnail decoding, segmentation
student, erosion and projected mask, Pillow standardization, JPEG round trips,
regressor checkpoint, and flushed per-photo record in both arms. For each of the
93 validation/test photos, the two implementations ran back-to-back in
alternating order.

The compiled selector matched all 93 Python crop boxes, Step 1 JPEG hashes,
Step 2 JPEG hashes, mask counts, and final predictions exactly. Validation
remained MAE/RMSE 0.11649/0.14810 and test remained 0.11829/0.15737. The Python
route measured median/p95 38.07/41.34 ms on validation and 37.88/43.35 ms on
test. The compiled route measured 16.15/18.05 ms and 16.33/18.25 ms,
respectively; every successful request remained below 100 ms. Paired median
savings were 21.38 ms wall time and 21.62 ms process CPU time. Peak RSS was
2.31 GiB and accelerator allocated/reserved memory was 0.119/0.152 GiB.

This is an exact-output CPU optimization: it removes the Python/numpy crop
candidate-search loop and its repeated temporary objects while preserving the
ranking and output bytes. It brings this experimental route comfortably below
the target and preserves its measured quality exactly relative to the Python
version. The adapter's quality relative to the historical v16 model still
needs an independent capture-date evaluation, so neither the model nor this
prototype is adopted in production. The standalone extension is not integrated
into project code; source, binary, paired records and the runner are in
`/tmp/wave_crop_search.cpp`, `/tmp/wave_crop_ext.cpython-314-x86_64-linux-gnu.so`,
and `/tmp/wave_v16_mse_crop_interleaved_ab_v1/`.

### Compiled crop selector on unlabeled incoming-style photos

I ran the MSE-adapted thumbnail route with the compiled crop selector and its
existing Step 2 fallback on all 18 raw images absent from the current labeled
split (13 HEIC, 5 JPEG, including UUID-named input). It completed 18/18 with no
preprocessing failures; the fallback activated on two photos and took 5.44 ms
and 6.05 ms. Full warm request latency, including raw decode and flushed record,
was 16.80 ms median / 27.79 ms p95 / 28.25 ms maximum; all 18 were under
100 ms. Peak RSS was 2.32 GiB and accelerator allocated/reserved memory was
0.126/0.158 GiB.

These photos are unlabeled and their capture dates overlap dates in the current
labeled splits, so this establishes only route coverage and latency, not model
quality or independent date generalization. Prediction artifacts are under
`/tmp/wave_fresh_unlabeled_thumbnail_mse_cpp_predictions_v1/`.

### Capture-date leakage audit and small date-disjoint quality diagnostic

I audited the MSE adapter evaluation by EXIF capture date, because photo-wise
splits can put near-identical sessions in both fitting and evaluation sets. The
October 28 test split has 27 capture dates. Twenty occur in the MSE adapter's
159-photo fit set, five more occur in its 58-photo epoch-selection holdout, and
only two dates are absent from both. Thus the 46-photo test score and the
32-photo archived-v16 comparison are not date-independent checks of this
adapter. The 32 archived-v16 photos share 11 dates with adapter fitting and
five with adapter epoch selection, despite having no same-filename overlap.
The validation set also shares dates with fit and holdout. Treat its full-split
metrics as exploratory.

The two test dates absent from both MSE fitting and epoch selection are
2026-08-24 and 2026-09-04 (five photos). Neither date occurs in v16's original
training split, and the thumbnail student training split also excludes both
dates. On this small date-disjoint subset, MSE-adapted thumbnail preprocessing
scored MAE/RMSE 0.13289/0.20826; archived canonical v16 scored 0.13449/0.15568.
MAE is nearly unchanged, but RMSE is materially worse because the adapter
predicted 0.629 for `IMG_7240` (label 0.20; archived v16 prediction 0.344).
Three of five absolute errors improved and two worsened. This is evidence that
the fast candidate can still have a quality hit on unseen capture dates; five
photos are too few for a final estimate, and these dates have appeared in
previous exploratory comparisons. Do not adopt the MSE adapter as a
quality-preserving replacement. Its compiled crop selector remains an
output-exact speed optimization relative to that experimental route, not proof
that the route matches v16 quality.

A robust final quality gate needs a genuinely untouched set of future capture
dates, with no images from those dates used to train or select either the
thumbnail segmenter or regressor. Until that set is labeled, all current
sub-100-ms results remain performance candidates rather than a verified
quality-preserving production replacement. The date-overlap calculation was
run inline; the paired raw records remain in
`/tmp/wave_v16_mse_cppcrop_test_eval_v1/`.

### Performance-principles review and next optimization direction

I reviewed PyTorch's official performance-tuning guidance and primary research
on real-time semantic segmentation and teacher-student distillation. The
principles relevant to this pipeline are: profile the whole request and its
stages; avoid repeated materialization and CPU/GPU transfers; tune thread
counts on the actual host rather than assuming more threads are faster; keep
intermediate tensors on their compute device where memory allows; and verify
the pixel masks, crop, encoded model inputs, prediction quality, latency and
memory together. The official PyTorch guide recommends inference mode for
inference and notes that compilation and memory layouts are workload dependent.
Primary segmentation research describes compact bilateral/context paths and
distillation as ways to pursue real-time segmentation while retaining spatial
detail; its benchmark results on other datasets do not establish quality on
these sea photos.

Most of the safe runtime principles have already been checked here: the v16
predictor uses `torch.inference_mode()`, its 6.81-GiB FP32 tile-logit accumulator
is kept on the GPU when capacity allows, and measured host-to-device / tile
accumulation work is small compared with the tile model forwards. The current
host reported 48.0 GiB free accelerator memory at idle. The configured v16
request takes about 11.38 s warm; 35 Step 1 tile forwards consume about 9.63 s.
Changing PyTorch inter-op threads, channels-last layout, or compiling the
768-pixel SegFormer tile did not improve the full prediction path; the fixed
shape compile probe made warm tile inference slightly slower and required a
22.95-s first compile. Thus CPU thread, tensor-copy, accumulator placement,
memory-format, and generic compilation changes cannot plausibly close the
roughly 100x gap by themselves.

The full-resolution YOLO26/v18 route measured 67.43-ms median and 74.55-ms p95;
the thumbnail route with compiled crop search measured about 16 ms median.
Both are sub-100-ms experiments, but neither has established quality parity
with canonical v16. The full-resolution route had worse test MAE/RMSE on the 32
exact archived test inputs, and the thumbnail adapter had a worse RMSE on the
small date-disjoint diagnostic. Both are rejected under the no-quality-hit
requirement. The next credible research direction is a task-specific compact segmentation student
trained against canonical v16 outputs and labeled waviness targets, with
capture-date-disjoint training, selection, and final evaluation. This is a new
model candidate, not an exact runtime optimization; it must pass the same
coverage, mask/crop behavior, per-photo quality, complete raw-to-record latency,
CPU, and memory gates before replacing v16. Real-time segmentation and
distillation references: [BiSeNet (ECCV 2018)](https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Changqian_Yu_BiSeNet_ECCV_2018_paper.php),
[Structured Knowledge Distillation for Semantic Segmentation (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Structured_Knowledge_Distillation_for_Semantic_Segmentation_CVPR_2019_paper.pdf),
and the [PyTorch Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html).

### Fast student with independently refit regression heads on date-held-out photos

The earlier 58-photo fast-student comparison used the v16 regressor, which had
already trained on many of those capture dates. I repeated the comparison with
two freshly fit regression heads to remove that regressor-label leakage. The
canonical and fast branches used the same 101 labeled training photos and the
same frozen ImageNet EfficientNet-B0 features; each branch received its own
head trained on its matching Step 2 inputs. All 12 outer holdout capture dates
were excluded from both head fitting and epoch selection. The inner epoch
selection used five separate fit dates (23 photos). Both heads were initialized
from the same seed and selected using training-only capture-date groups.

The segmentation student had also excluded those 12 outer dates from student
training and segmentation checkpoint selection. It completed preprocessing for
all 58 held-out photos. Archived canonical Step 2 images existed for 48, so
quality comparison is paired on those 48 only; the fast branch's remaining 10
are candidate-only diagnostics, not a paired comparison. On all 58 fast inputs,
its new head scored MAE/RMSE 0.14070/0.17764. On the paired 48, the fast branch
scored 0.14252/0.18399 versus 0.13631/0.17900 for the canonical branch. The
paired MAE difference was +0.00621, and RMSE was also worse. Error was worse on
27 photos and better on 21. A capture-date-cluster bootstrap over 11 paired
dates gave a wide 95% interval of [-0.01946, 0.02530] for the MAE difference.
This does not establish a statistically reliable regression, but the point
estimates do not pass the no-quality-hit gate. Do not adopt this candidate on
these results.

The fast student route previously measured 35.3 ms median / 41.9 ms p95 for a
complete warm request on the same 58-photo set, including decode, both
preprocessing stages, regression, and flushed prediction record. The current
run regenerated the missing 58 fast Step 2 inputs in 2.9 seconds total and
measured both head-feature extractions; the regression-head retraining adds no
meaningful per-request work. Peak process RSS during feature extraction and
head fitting was 2.11 GiB. These photos are a development holdout drawn from
the old training pool, and one date has no archived canonical image; this is
not an untouched future-capture quality guarantee.

The common fit set was limited to 101 because only that many current training
photos had both a fast candidate input and an archived canonical input. Inputs
were generated or read from `/tmp/wave_yolo26_clean_holdout_candidate_inputs_v2/`
and `step-2-final-water-data/archive/20261027T134224Z/`. Training heads, paired
per-photo errors, capture-date bootstrap and full metrics are in
`/tmp/wave_fast_vs_canonical_clean_date_regressor_v1/`; the student preprocessing
runner and records are in `/tmp/wave_clean_holdout_candidate_inputs_v2/`.
