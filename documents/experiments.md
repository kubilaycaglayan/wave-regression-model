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

#### MIGraphX FP32 on the canonical 768-pixel tile shape

To check whether the engine result transferred to the configured v16 model
shape, I exported the same SegFormer-B2 at fixed 768 x 768 FP32 and ran one
real tile from `IMG_7238.HEIC` through ONNX Runtime's MIGraphX execution
provider. The first tile call took 60.26 s because of lazy compilation; the
next three measured 95.1, 96.0, and 100.6 ms (98.3 ms median), versus 266.7 ms
median for PyTorch on the same tile. The low-resolution argmax and the
bilinearly upsampled 768 x 768 class map matched exactly on this tile. The
MIGraphX session itself took 56.7 s to initialize in the second process.

This is a promising single-tile engine speedup with exact mask output on one
tile, but it cannot meet the 100 ms complete-request target: Step 1 requires 35
such tiles, followed by source decode, Step 2, and regression. A rough sum of
the warm tile medians alone exceeds 3.4 s. No full-request quality metric was
run; MAE is unknown. This is not an adopted pipeline. Model export, server, and
tile comparison are under `/tmp/wave-segformer-768.onnx`,
`/tmp/migraphx_server_768.py`, and `/tmp/wave_compare_migraphx_768_tile.py`.

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

### Raw-image-to-waviness EfficientNet candidate (date-held-out, rejected)

As a compact end-to-end alternative to running both segmentation passes, I
fine-tuned the current v16 EfficientNet-B0 regressor to score original decoded
photos resized to 224 x 224. This is a real raw-image input contract: it uses
all decoded source pixels before resize, runs no segmentation, and uses no
cached prediction. The 12-date outer holdout contained 58 labeled photos; all
were excluded from candidate training and epoch selection. The remaining 159
training photos were divided by capture date into 116 fit and 43 selection
photos. Selection chose epoch 9, then the model was refit on all 159 allowed
photos before the outer evaluation. The model completed all 58 holdout photos.

Outer-holdout MAE was 0.12525 on 58 photos. For the 48 photos with archived
canonical inputs, candidate MAE was 0.13078 versus 0.10855 for the v16
checkpoint on the same photos; candidate-minus-v16 MAE was +0.02223 (20.5%
higher). The v16 baseline has seen some of these capture dates, so this is a
conservative candidate comparison; it still clearly fails the no-hit quality
gate.

The complete raw-to-flushed-record route measured 185.9 ms median, 216.8 ms
p95, and 243.0 ms maximum; 0/58 requests were below 100 ms. There were no
preprocessing failures. Full-resolution HEIC decode plus Pillow Lanczos resize
to 224 took 152.9 ms median / 258.9 ms p95 during input caching. Peak process
RSS was 2.73 GiB across training and evaluation; inference-only peak memory was
not isolated. This direct model therefore misses both quality and latency
requirements. It is not adopted. The run took 57.6 seconds end to end, including
source caching and training. Script, summary, and flushed per-photo records are
under `/tmp/wave_rawdirect_dateholdout_v1/`.

#### `reducing_gap=3` raw resize contract and full-backbone follow-up

Before training a matched candidate, I screened two-stage Pillow Lanczos
reduction on the same 58 full-resolution raw holdout sources. Default Lanczos
resize to 224 x 224 took 63.4 ms median / 68.9 ms p95; `reducing_gap=3.0`
took 10.4 / 11.1 ms. Full-resolution decode took 71.0 ms median / 105.7 ms
p95 in this interleaved screen. All 58 resized outputs differed from default
Lanczos, with mean absolute pixel difference 0.218 intensity levels. This is a
changed input contract, so I trained and evaluated a matching raw regressor.

The partial-backbone candidate (features 6 onward and head) completed all 58
date-held-out photos. MAE was 0.12565 overall and 0.13105 on the 48 photos
paired with v16 (0.10855), a +0.02250 (+20.7%) change. Raw-to-flushed-record
latency was 105.9 ms median / 145.5 ms p95 / 188.9 ms max; 0/58 met 100 ms.
Its faster resize lowered the paired median from 185.9 ms by about 80 ms, but
the total still misses the target and MAE remains materially worse.

I then unfroze the full EfficientNet-B0 backbone to check whether earlier
features could adapt better to raw photos. This completed 58/58 with MAE
0.12532 overall and 0.13091 paired versus v16 0.10855 (+20.6%). Latency was
121.9 ms median / 157.3 ms p95 / 178.8 ms max. Full-backbone tuning did not
improve paired MAE or latency over the partial-backbone `reducing_gap=3`
candidate. Both remain rejected. Resize screen and candidate artifacts are
under `/tmp/wave_raw_resize_gap_screen.py`,
`/tmp/wave_rawdirect_gap3_v1/`, and `/tmp/wave_rawdirect_gap3_fullft_v1/`.

#### Teacher-output distillation follow-up for the raw-image model

I repeated the same date-held-out raw-image experiment with a combined loss:
the labeled waviness SmoothL1 loss plus 0.5-weighted MSE to the v16 checkpoint's
prediction on the matching canonical Step 2 image. Epoch selection still used
the 43-image inner capture-date holdout and ground-truth MAE only; the 58 outer
photos remained excluded from training and selection. The archive contained
teacher inputs for 101/159 allowed training photos, so the other 58 fit photos
used only the labeled loss. Selection again chose epoch 9.

On the outer 58, candidate MAE was 0.12540. On the 48 paired archived v16
inputs, candidate MAE was 0.13074 versus 0.10855 for v16, a +0.02219 (+20.4%)
paired difference. This is effectively unchanged from the label-only raw
model's 0.13078 paired MAE; teacher-output distillation did not recover the
quality gap. These repeatedly examined dates remain exploratory.

The raw-to-flushed-record latency was 165.9 ms median / 194.5 ms p95 / 230.0
ms maximum, with 0/58 requests under 100 ms. Against the first label-only raw
model run on the same 58 photos, the paired median latency saving was 17.0 ms
(44/58 requests faster), but substantial latency remains and the MAE does not
pass the quality gate. This run is rejected and not deployed. The teacher
predictions, selected checkpoint, metrics, and per-photo timings are under
`/tmp/wave_rawdirect_distill_v1/`.

#### Training-date affine calibration screen for the raw-image candidate

I screened a simple affine output calibration for the raw model trained on the
116-image inner-fit set. The slope and intercept were fit only on the separate
43-image, date-held-out selection set (0.9654 and 0.0300), then applied to the
58 outer photos. This reduced selection MAE from 0.12710 to 0.12519, but outer
MAE worsened from 0.13711 to 0.13848. On the 48 photos paired with canonical
v16 outputs, calibrated MAE was 0.13894 versus 0.10855 for v16, a +0.03038
difference. The calibration does not transfer and fails the quality gate.

Calibrated raw-to-flushed-record latency was 157.0 ms median / 206.8 ms p95 /
224.5 ms maximum, with 0/58 under 100 ms. No change was deployed. Results are
under `/tmp/wave_rawdirect_dateholdout_v1/calibrated_summary.json` and
`/tmp/wave_rawdirect_dateholdout_v1/calibrated_raw_to_record.jsonl`.

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

### Date-held-out thumbnail student: corrected v16 quality comparison

The 58-photo cohort spans 12 capture dates and was excluded from thumbnail
student training and segmentation checkpoint selection. The student used 127
training photos and 22 photos from separate fit dates for checkpoint selection;
best segmentation mIoU/pixel accuracy was 0.809/0.895. One holdout photo had no
teacher mask, but it remained in the waviness evaluation.

Correction: the earlier comparison in this section used JPEGs from the active
`step-2-final-water-data` folder as though they were canonical v16 inputs. Its
preprocessing contract is `fast-v1` (448-pixel FP16 segmentation), not
`baseline-v16`; the reported MAE 0.13443 vs 0.14538 was not a valid canonical-v16
comparison. I regenerated the reference predictions from the 58 raw files with
the actual baseline-v16 contract: full-resolution 768-pixel overlapping B2
tiles in FP32 for Step 1 and Step 2, followed by the configured v16 regressor.
Canonical v16 succeeded on 51/58; seven failed Step 2 crop selection. Its
success-only MAE/RMSE were 0.12006/0.15645.

The thumbnail student with the unchanged v16 head completed 58/58. On all
candidate successes its MAE/RMSE were 0.13443/0.17282. On the 51 photos where
both routes succeeded, candidate MAE/RMSE were 0.13226/0.17206 versus
0.12006/0.15645 for canonical v16: MAE was higher by 0.01220 (+10.2%). Its
absolute error improved on 20 photos and worsened on 31. A 12-date cluster
bootstrap 95% interval for candidate-minus-v16 MAE was [-0.01408, 0.03416], so
this small cohort cannot rule out a material quality hit. The candidate did
complete the seven photos that v16 could not process; report this coverage gain
separately rather than treating it as an MAE improvement.

On the three 2026-10-23 photos from dates absent from v16 training, candidate
MAE was 0.08620 versus 0.12712 for v16. This is encouraging but only three
photos. Complete raw-to-flushed-record candidate latency was 35.26 ms median /
41.94 ms p95 / 46.88 ms max, with 58/58 below 100 ms. Median CPU time was
35.5 ms, peak RSS 2.31 GiB, and accelerator allocated/reserved memory
0.12/0.15 GiB. Canonical v16 latency on this cohort was 10,057.6 ms median /
10,092.3 ms p95, with 0/51 under 100 ms.

I also trained the waviness head only on the 158 available candidate inputs
outside the 58-photo holdout. The inputs came from the same date-held-out
student. Starting from ImageNet weights, the frozen-backbone head scored MAE
0.15193 on all 58 holdout photos and 0.14662 vs v16 0.12006 on the shared 51
(+22.1%). Fine-tuning EfficientNet features[7:] worsened shared MAE to 0.15154
(+26.2%). A v16-initialized partial fine-tune scored 0.14096 on the shared 51
(+17.4%), but its initialization had already seen 48 of the 58 holdout photos,
so it is not independent evidence. These head variants remained under 100 ms
(34.65–35.05 ms median) but failed the MAE screen. None is adopted; v16 remains
selected.

The regenerated canonical-v16 records are in
`/tmp/wave_v16_canonical_outer58_raw_eval_v1/`. The unchanged-v16-head
thumbnail run is in `/tmp/wave_yolo26_clean_capture_holdout_eval_v1/`; the
clean head training and runs are under `/tmp/wave_yolo26_clean_regressor_cv_*`.
The student, its split, and teacher masks are under
`/tmp/wave_yolo26_capture_date_holdout_student_v1/` and
`/tmp/wave_yolo26_capture_date_holdout_student_data_v1/`.

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

#### ROCm kernel and transfer trace for the configured v16 path

To determine whether the long Step 1 time was primarily caused by host/device
traffic or by the segmenter itself, I captured one cold and one warm complete
raw-to-output request with ROCm `rocprofv3` kernel and memory-copy tracing. The
instrumented warm request took 11.587 s (the earlier uninstrumented warm sample
was 11.377 s; profiler overhead makes this a diagnostic, not a new latency
benchmark). The trace recorded 35 Step 1 tile forwards: 9.837 s in the tile
model, 0.534 s preparing tiles, 0.268 s resizing/accumulating logits, and
0.073 s finalizing the mask. Step 2 took 0.327 s. The one-photo waviness MAE is
unknown; its absolute error was 0.1487, which is not a cohort metric.

Across the profiled cold and warm requests, ROCm recorded 373 explicit
host/device copies totaling about 823 MiB and 40.5 ms of summed copy duration
(32.0 ms H2D, 8.5 ms D2H). That is too small to explain a multi-second request
or close the sub-100 ms gap by itself. The GPU trace also contained 4,608
elementwise copy-kernel dispatches and 706 GEMM dispatches during this single
photo run. These kernel counts identify a possible model-execution/materialized
tensor optimization target, but do not prove the copies are redundant; any
fusion or layout change must first match masks and final JPEG bytes, then
prediction outputs and paired waviness MAE. The run did not record process CPU
time or peak host RSS, so it cannot support a CPU or total-memory conclusion.

This evidence follows ROCm's recommended sequence of profiling the application
timeline, transfers, and kernels before selecting an optimization. It makes
another decoder-thread or host-to-device-copy experiment a low-priority route
for the current v16 profile. The viable paths toward <100 ms remain either a
substantially faster execution of the same segmentation function or a compact
student that passes the strict paired waviness-quality gate; prior YOLO routes
are not acceptable replacements because they regressed v16 MAE. Trace data and
the benchmark report are in `/tmp/wave_v16_rocprof_profile_v1/` and
`/tmp/wave_v16_rocprof_benchmark_v1/`. Reference: AMD's official [HIP
performance guidelines](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html)
recommend profiling kernel execution, memory bandwidth, occupancy, and data
movement before tuning.

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

### Full preprocessing and regressor grouped by capture date

An initial five-fold diagnostic cross-validated the regression heads on these
101 photos but reused one YOLO student that had trained on those rows. It
scored MAE/RMSE 0.16770/0.20534 for canonical inputs and 0.17011/0.21382 for
candidate inputs; its date-clustered MAE interval crossed zero. Because the
student had seen the held-out photos, this was not a full-pipeline quality
estimate. I then ran a full capture-date-grouped cross-validation on the 101
labeled photos that have both candidate inputs and archived canonical Step 2
inputs. Each fold kept its outer
dates out of YOLO student training, student checkpoint selection, regression
head fitting, and regression epoch selection. The student was trained from
canonical B2 teacher masks available for 96/101 photos; each outer training
fold had 75–79 teacher masks. Regression heads used the same frozen ImageNet
EfficientNet-B0 feature extractor and identical training
photos for both preprocessing contracts. Each branch selected and refit its
own scalar head using only the outer training dates.

The candidate completed 100/101 outer-fold predictions; one photo failed Step
2 crop selection. On the 100 paired successes, the canonical branch scored
MAE/RMSE 0.25171/0.29970, and the candidate scored 0.25247/0.30468. Candidate
minus canonical MAE was +0.00075, with a capture-date-cluster bootstrap 95%
interval of [-0.01802, 0.01599]. RMSE increased by 0.00498; errors worsened on
52 photos and improved on 48. The interval is too wide to establish a reliable
MAE regression or no-regression. These absolute metrics are also substantially
worse than the configured v16 model because each cross-validation head was
trained from scratch on only about 80 photos. This is a controlled comparison
between preprocessing contracts, not a production-quality estimate. The
candidate fails the no-quality-hit and coverage gates; do not adopt it.

For all 101 raw-to-record requests, 100 succeeded and all 100 successful
requests were under 100 ms: median/p95/max was 36.97/42.58/47.97 ms. The one
crop failure occurred after 9.85 ms and produced no prediction. Median process
CPU time was 37.24 ms; peak process RSS was 2.42 GiB and peak PyTorch
accelerator allocation during prediction was 0.198 GiB. Student training used
about 0.9 GiB accelerator memory per fold. Full-request timing included source
decode, YOLO, both preprocessing steps, regression, and flushed prediction
record; model loading and warm-up were outside the timed request. These are
warm-service request timings, not process-startup-to-first-prediction timings,
so they do not establish a strict cold-start latency under 100 ms. The
five-fold experiment took 427.7 seconds.

This is a useful direction for speed and resource use, but it does not meet the
required result: the candidate is slightly worse on both aggregate error
metrics, one request fails, and the uncertainty remains large. More labeled
training photos with both canonical and candidate inputs, plus untouched future
capture dates, are needed before a quality-preserving replacement can be
established. Per-photo OOF rows and fold checkpoints are in
`/tmp/wave_fullpipeline_grouped_cv_v3/`; corrected complete-request timings are
in `corrected_full_request_timing_v3.json`. The training runner is
`/tmp/wave_fullpipeline_grouped_cv_v3.py`.

#### Why the paired cohort could not be expanded with more canonical inputs

I checked the 56 additional fast-route training photos that have no archived
canonical Step 2 JPEG, to see if those baseline inputs could be regenerated.
The first five were processed with the original 768-pixel overlapping B2
pipeline. All five reached Step 2 but failed its reliable-crop selection, so
there was no canonical model input to compare. The probe took 51.7 seconds,
used 2.45 GiB peak RSS, and reached 8.42 GiB peak PyTorch-reserved accelerator
memory. I stopped the larger regeneration attempt because these photos do not
provide a paired canonical quality baseline through this preprocessing
contract. This explains the current 101-photo paired limit; it does not prove
that every one of the remaining 51 photos would fail. The candidate route does
produce inputs for these examples, but that added coverage alone does not prove
quality preservation. Probe records are in
`/tmp/wave_missing_canonical_inputs_probe_v1/`.

### More training data for the fast-input regression head

A follow-up trained frozen-backbone and partially unfrozen EfficientNet-B0
heads on 157 candidate-preprocessed training images from 39 capture dates.
Model/epoch selection used 35 images across 11 training dates; the 58-image,
12-date outer holdout was kept out of fitting and selection. Candidate Step 2
outputs matched the previously cached candidate JPEG bytes on all 58 images.
Both heads were evaluated on all 58 candidate inputs and against v16 on the 48
images with archived canonical inputs. This checks the fast-input route with
more training data, but changes both the preprocessing contract and fitted
regression head.

The frozen head had candidate-only MAE/RMSE 0.1662/0.2119 on 58 images. On the
48-image paired cohort, its MAE/RMSE was 0.1562/0.1967 versus 0.1086/0.1452 for
v16; paired MAE was 0.0476 worse, with 35 of 48 absolute errors worse. The
partially unfrozen head had candidate-only MAE/RMSE 0.1694/0.2180; paired
MAE/RMSE was 0.1575/0.2003, with paired MAE 0.0490 worse and 35 of 48
absolute errors worse. Both routes succeeded on all 58 images. Neither passes
the quality gate.

Warm raw-image-to-flushed-record timing (model loading and warmup excluded) was
37.4 ms median, 43.8 ms p95 and 66.4 ms max for the frozen head; the partially
unfrozen head measured 37.2/43.3/49.8 ms. All 58 successful requests were under
100 ms. Peak process RSS was 2.57 GiB and peak PyTorch accelerator allocation
was 0.233 GiB (0.279 GiB reserved). These latency results do not meet the
quality requirement and do not establish a cold-start under 100 ms. Reproducer
and artifacts: `/tmp/wave_fast_student_regressor_157_date_holdout_v1.py` and
`/tmp/wave_fast_student_regressor_157_date_holdout_v1/summary.json`.

A separate tile-batching probe on a different held-out 3024 x 4032 photo
confirmed this result. The canonical sequential FP32 run took 10.85 s for 35
tiles. Microbatch size two took 31.35 s and changed two class-map pixels;
batch size four took 32.66 s and changed three. Peak accelerator allocation
reached 13.97 GiB (15.87 GiB reserved) and process peak RSS was 2.41 GiB.
Batching is therefore both slower and non-identical on this sample and hardware.
The probe is `/tmp/wave_batch_seg_tiles_probe_v1.py`; results are in
`/tmp/wave_batch_seg_tiles_probe_v1.log`.

### Canonical-resolution FP16 autocast probe

To separate reduced precision from the earlier lower-resolution changes, I kept
both canonical 768-pixel SegFormer passes, tile positions, weights, crop rules,
and the v16 regression checkpoint unchanged and enabled FP16 autocast only for
SegFormer inference. On one 3024 x 4032 image, full raw-to-prediction time fell
from 11.53 s in FP32 to 6.48 s in FP16. Step 1 and Step 2 JPEG bytes changed;
the prediction moved by 0.00138. This single-image timing is a probe, not a
stable paired latency estimate.

I then ran FP16 preprocessing on the existing 32-photo test split, which has
been consulted in prior experiments and is not a newly untouched test. All 32
requests succeeded. Candidate MAE/RMSE were 0.1311/0.1583 versus 0.1328/0.1616
for v16 on archived canonical inputs. The candidate had lower absolute error on
15 photos and higher error on 17. A 10,000-draw capture-date-cluster bootstrap
across 18 dates gave candidate-minus-canonical 95% intervals of [-0.01485,
0.01025] for MAE and [-0.01766, 0.00852] for RMSE. Both include a meaningful
regression, so the point-estimate improvement does not establish preserved
quality. None of the 32 candidate Step 2 JPEGs were byte-identical to archived
canonical inputs.

Warm raw-to-prediction plus Step 2 output-file write measured 5,646 ms median,
5,849 ms p95 and 6,666 ms maximum; none of the 32 were under 100 ms. The timing
excluded model loading and did not include a flushed prediction record. Peak
process RSS was 2.93 GiB; peak PyTorch accelerator allocation/reservation was
8.31/8.56 GiB. CPU time was not captured. Thus FP16 is a substantial speed
improvement over the separate FP32 probe but remains far above target and does
not pass the quality gate. Keep experimental. Script and detailed records:
`/tmp/wave_canonical_fp16_test_quality_v1.py` and
`/tmp/wave_canonical_fp16_test_quality_v1/summary.json`.

### Canonical-initialized regressor adaptation for the date-held-out fast route

To reduce the input-domain gap, I trained the candidate regressor from a
canonical regressor fitted without any outer holdout dates, then adapted its
scalar head to the fast-route inputs. The YOLO26 student and regressor fit and
selection excluded all 12 outer capture dates. The student route produced 157
training inputs and 58 holdout inputs; archived canonical inputs existed for
101 training and 48 holdout photos. Canonical head selection used 12
training-date photos; fast-head adaptation selected 3 epochs on 35 photos
across 11 training dates, then refit on all 157 candidate training images.

The adapted candidate scored MAE/RMSE 0.1519/0.1898 on all 58 holdout images.
On the 48 paired images, it scored 0.1472/0.1849, compared with 0.1404/0.1840
for a newly trained canonical head using the 101 shared training images. The
paired MAE difference was +0.00676; 29 errors were worse and 19 better. Its
11-date-cluster bootstrap 95% interval was [-0.00712, 0.01749], which does not
rule out a material regression. Against the configured v16 checkpoint on the
same 48 images, candidate MAE/RMSE were 0.1472/0.1849 versus 0.1086/0.1452;
MAE was worse on 35 images and better on 13, with date-cluster interval
[0.02906, 0.04491]. The v16 comparison is conservative for the candidate because
v16 training included some of these dates; the newly trained canonical branch
is the date-disjoint comparison. Neither comparison establishes quality
preservation.

The complete warm thumbnail-route request was remeasured with the adapted head:
58/58 succeeded and their generated Step 2 JPEGs matched cached candidate bytes
exactly. Raw-image-to-flushed-record timing was 35.20 ms median, 41.12 ms p95,
and 45.26 ms maximum; all requests were under 100 ms. Median process CPU time
was 35.49 ms. Peak RSS was 2.32 GiB and accelerator allocated/reserved memory
was 0.135/0.172 GiB. Model load and warmup were excluded. This path decodes
HEIF embedded thumbnails or JPEG draft images at roughly 320 pixels, so the
result applies to that changed input contract. It is not acceptable as a
quality-preserving replacement: the matched errors remain worse than the
canonical branch and v16, despite meeting warm request latency. Reproduction,
per-photo records, and metrics are in
`/tmp/wave_fast_regressor_canonical_init_adaptation_v2/` and
`/tmp/wave_fast_regressor_canonical_init_adaptation_latency_v2/`.

### Full-resolution YOLO26s student on capture-date-held-out photos (quality and speed fail)

To test whether a larger semantic model would improve the full-resolution YOLO26
route, I trained YOLO26s from its ADE20K semantic checkpoint using the existing
teacher-mask pairs. The outer test split remained 58 photos from 12 excluded
capture dates. Checkpoint selection used 127 photos across 34 dates for training
and 22 photos across 5 dates for selection; the selected 19-epoch schedule was
then refit on all 149 non-holdout photos. Selection and refit took 390 and 306
seconds, respectively. Training peak accelerator memory was about 1.4 GiB.

On the 58 held-out photos, 56 completed and 2 failed preprocessing (one Step 1
water-area failure, one Step 2 crop-selection failure). The student masks had
mean teacher IoU 0.860 and minimum IoU 0.225. With the existing v16 regressor,
candidate MAE/RMSE were 0.1361/0.1703 on all 56 successes. On the 48 photos with
archived v16 inputs, candidate MAE/RMSE were 0.1366/0.1707 versus 0.1086/0.1452
for v16; candidate absolute error was worse on 28 and better on 20. The paired
MAE delta was +0.02806, with an 11-capture-date bootstrap 95% interval of
[-0.00117, 0.04443]. This is not evidence of preserved waviness quality. The
v16 checkpoint itself has seen some of these dates, so this comparison is not
a fully date-disjoint baseline, but the candidate's clear point-estimate loss
does not justify adoption.

An initial timing pass calculated full-resolution teacher-mask IoU inside the
timed loop. That diagnostic is not part of a prediction request and inflated
latency, so those initial 229.8-ms and 151.1-ms values are not used as
production-path measurements. I reran timing without teacher-mask scoring while
keeping raw-image decode, full-resolution segmentation, both preprocessing
steps, regression, and flushed output-record writing in the timed path.

With the exact parallel HEIC tile decoder and compiled crop selector, canonical
Pillow Lanczos resizing measured 107.8 ms median and 121.6 ms p95; 18/56
requests were below 100 ms. Median CPU time was 292.5 ms, peak RSS 2.57 GiB,
and PyTorch accelerator allocated/reserved memory 0.328/0.371 GiB. Stage timing
identified standardization as the main remaining cost: erosion 3.8 ms,
full-resolution ROI masking and Lanczos resize 58.4 ms, projection 1.4 ms,
decode 19.7 ms, and YOLO plus mask transfer 12.7 ms. Smaller stages were
Step 2 crop search 2.6 ms, Step 2 encoding/transform/transfer 2.3 ms, and
regression 4.8 ms.

I then enabled Pillow's documented `reducing_gap=3.0` two-stage reduction for
the full-resolution masked ROI before its final Lanczos resize. It retains the
full-resolution raw image and Lanczos final filter; it does not use the HEIF
thumbnail or a lower-resolution segmentation input. The route measured
64.6 ms median, 75.4 ms p95, and 86.4 ms maximum, with all 56 successful
requests under 100 ms. Median CPU time fell to 242.4 ms; peak RSS and
accelerator memory remained 2.57 GiB and 0.328/0.371 GiB. Standardization fell
from 58.4 ms to 25.1 ms. Output JPEGs changed on 55/56 photos, so this is not an
exact-output optimization. On the same 56 held-out photos, MAE/RMSE changed
from 0.1361/0.1703 to 0.1348/0.1677. On the 48-photo paired set, candidate
MAE changed from 0.1366 to 0.1345 (24 individual errors worsened and 24
improved); the 11-date cluster-bootstrap 95% interval for optimized-minus-base
MAE was [-0.00748, 0.00281]. This provides no evidence of a quality hit from
this resize optimization on this sample, but the interval still allows a small
regression and output pixels differ.

The speed optimization does not fix the candidate model's quality gap to v16:
the 48 paired photos scored MAE/RMSE 0.1345/0.1672 with the optimized route
versus 0.1086/0.1452 for v16. Candidate-minus-v16 MAE was +0.02599, with
date-cluster interval [-0.00291, 0.04204]. This remains a fast research route,
not an acceptable replacement. A separate eight-strip resize experiment was
also excluded: it changed Step 2 JPEGs on 7/56 photos and did not establish
quality preservation. The result demonstrates that legitimate full-resolution
resize optimization can meet the raw-to-output latency target on this hardware,
but the segmentation/regression candidate still must pass the quality gate.

Reproduction scripts and records are under `/tmp/wave_yolo26s_fullres_dateholdout_v1/`,
`/tmp/wave_yolo26s_fullres_dateholdout_refit_eval_v1/`,
`/tmp/wave_yolo26s_tiled_eval_v1/`, `/tmp/wave_yolo26s_tiled_eval_v2/`,
`/tmp/wave_yolo26s_tiled_eval_v3/`, `/tmp/wave_yolo26s_tiled_eval_profile_v2/`,
`/tmp/wave_yolo26s_tiled_eval_profile_v3/`, and
`/tmp/wave_yolo26s_tiled_eval_reducing_gap3_v1/`. Pillow's resize option is
documented at [Image.resize](https://pillow.readthedocs.io/en/stable/reference/Image.html?highlight=reducing_gap).
No production pipeline code or model was changed.

### Regressor adaptation for the expanded-mask YOLO26s route

The expanded-mask YOLO26s student above produced a fast full-resolution route,
but its unchanged v16 regression head still had paired MAE 0.12624 versus
0.10855 for v16 on the same 48 photos. I generated Step 2 inputs for the 159
non-outer-date training photos with the exact same student, full-resolution
HEIF decode, `reducing_gap=3.0` standardization, and projected Step 2 mask. Two
preprocessing failures left 157 candidate training inputs. The 12 outer dates
(58 images) remained excluded from the regressor fit. Selection used 43
photos from 14 capture dates; a frozen-head branch and a partial fine-tune of
EfficientNet features 7 onward were compared by selection MAE only. The
partial branch was selected at epoch 11 with selection MAE 0.09451, then
refit on all 157 available training inputs.

On the 58 outer photos, the adapted route completed 58/58 and scored MAE
0.13291. On the 48 paired photos, candidate MAE was 0.12863 versus 0.10855 for
v16 (+0.02008, or 18.5% higher); errors were worse on 28 and better on 20.
Compared with the unchanged v16 head on this same YOLO route, adaptation also
slightly worsened paired MAE (0.12863 versus 0.12624). The selection MAE did
not transfer to the outer dates, so adaptation is rejected.

Despite the quality failure, the raw-to-flushed-record route remained fast:
72.99 ms median / 90.03 ms p95 / 93.09 ms maximum, with all 58 requests under
100 ms. Median process CPU time was 254.84 ms; peak process RSS was 2.71 GiB,
and PyTorch accelerator allocated/reserved memory was 0.568/0.807 GiB. These
results confirm the speed potential of this preprocessing route but do not
meet the no-quality-hit requirement. Nothing was deployed. Training and full
request artifacts are under `/tmp/wave_yolo26s_v3_regadapt_v1/`.

### Teacher-distilled regressor on the expanded-mask YOLO26s route

I followed the failed direct adaptation with a partial EfficientNet fine-tune using
both true waviness labels and a 0.5-weight v16 teacher-score loss on training
photos with archived teacher inputs. All 12 outer capture dates (58 photos)
were excluded from fitting and selection. The model was selected on an
inner date split and refit on the available non-outer-date inputs. The
corrected evaluation explicitly used the v16 checkpoint for the baseline; an
earlier path-broken evaluation produced no valid result and is discarded.

On the 58 outer-date photos the adapted candidate completed 58/58 and had MAE
0.12539. On the 48-photo set with archived v16 inputs, candidate MAE was
0.12081 versus 0.10855 for v16 (+0.01225, or 11.3% higher). Errors were worse
on 25 and better on 23; the 11-date cluster-bootstrap 95% interval for the MAE
delta was [-0.01306, 0.03681]. This did not establish preserved quality.

The full raw-to-flushed-output evaluation measured 63.61 ms median, 73.29 ms
p95, and 78.78 ms maximum; all 58 requests were below 100 ms. Median process
CPU time was 241.92 ms, peak RSS 2.57 GiB, and accelerator allocated/reserved
memory 0.328/0.371 GiB. Latency excludes cold model loading. Candidate quality
is from the corrected paired comparison above; the full-path script's second
baseline calculation reused the adapted regressor, so that particular baseline
field is invalid and was not used. The verified canonical v16 paired MAE is
from the independent corrected comparison. Artifacts are under
`/tmp/wave_yolo26s_v3_regdistill_v1/` and
`/tmp/wave_yolo26s_v3_regdistill_full_eval_v1/`. The candidate remains a
research result and was not deployed.

### Updated bottleneck and hardware assessment for the v16 path

The current profiling host is an AMD Ryzen AI MAX+ 395 with Radeon 8060S
(gfx1151, ROCm 7.16, PyTorch 2.14), 16 CPU cores / 32 logical CPUs, 122.7 GiB
system RAM, and about 48 GiB free shared accelerator memory at idle. The canonical
v16 request for a 4032 x 3024 HEIC needs 35 Step 1 tiles at 768 pixels, plus one
small Step 2 tile. The warmed raw-to-prediction request took 11.59 s in the
instrumented profile; 9.84 s was Step 1 tile-model inference and 0.53 s tile
preparation. HEIC decode was 0.10 s, while both JPEG writes together were below
1 ms. CPU time was not collected, so CPU saturation cannot be inferred from
this trace.

A query of the saved ROCm dispatch database counted 4,608 elementwise
`direct_copy_kernel_cuda` launches with 6.1 s summed kernel duration and 706
GEMM launches with 5.9 s summed duration. These are device-kernel durations,
which can overlap and must not be added as wall time. The kernel name
`direct_copy_kernel_cuda` does not mean that 6.1 s was host-to-device copying:
the separate measured explicit transfer total was about 40.5 ms across
823 MiB. Thus reducing host/device transfers cannot by itself materially close
the latency gap. This profile points to repeated SegFormer model computation
and its GPU elementwise work as the main target; it does not yet establish
which framework operator or kernel fusion opportunity is responsible for the
elementwise cost.

The under-100-ms goal requires at least a 115x reduction from the 11.59-s
instrumented raw-to-output request. Even eliminating all decode, preprocessing,
regression, and file/log overhead would leave roughly 9.84 s of Step 1 forwards.
Canonical-profile FP16 previously reduced end-to-end latency to about 5.65 s
median but changed masks and did not establish quality parity; it is far from
the target. MIGraphX FP32 was about 98 ms for one 768-pixel tile after a long
compile/load, so multiplying by the canonical 35 tiles still leaves a multi-
second request. Batching the same tiles was slower and changed two mask pixels.
These experiments rule out precision-only, transfer-only, compilation-only,
and naive batching as sufficient solutions.

The evidence-based next path is to profile operator-to-kernel mapping and
accelerator utilization on a representative warm request, then optimize or
replace the segmentation computation itself while validating the canonical
mask/crop and paired waviness MAE. A compact segmentation student or a
purpose-trained model is the most plausible route to a 100-ms budget, but it
must pass the quality gate on date-disjoint photos before adoption. The v16
checkpoint remains configured; no production checkpoint or segmentation
profile changed as part of this assessment.

### CPU-thread and host-memory check on the configured v16 prediction path

I ran two fresh end-to-end benchmark processes on `IMG_7238.HEIC`, each with
one cold request and two warm requests. The production prediction entry point
defaults to one PyTorch intra-op thread. The 32-thread comparison is only a
benchmark diagnostic; it is not the production setting. Both settings produced
the identical prediction (0.351297) on this image (label 0.50, absolute error
0.148703). This single image is not an MAE evaluation.

At one thread, warm raw-to-output latency was 10.787 s median across the two
requests; Step 1 tile-model time was 9.194 s median. At 32 threads, the same
figures were 11.216 s and 9.617 s. The one-thread setting was about 3.8% faster
in this small, single-image comparison; it is already the configured
production default. `/usr/bin/time -v` measured 121% CPU utilization and
2.49 GiB peak RSS for the full one-thread benchmark process, versus 131% and
2.49 GiB for the 32-thread process. These CPU percentages include Python
startup and all three predictions, are averaged over the process lifetime, and
do not describe short bursts. No swap or major page faults occurred.

The result confirms that increasing PyTorch CPU threads does not help this
GPU-heavy route, and the host process RSS is modest relative to available
system RAM. The 6.81-GiB float32 segmentation accumulator is a GPU allocation,
not host RSS. The remaining latency is dominated by the 35 sequential GPU
SegFormer forwards, so CPU/memory tuning is not a plausible path from roughly
10.8 s to 100 ms. Artifacts:
`/tmp/wave_v16_cpu_memory_profile_v2/20261028T195133294344Z/` and
`/tmp/wave_v16_cpu_memory_profile_v3_t1/20261028T195224590517Z/`.

### Zero-overlap v16 tile experiment (rejected on quality and coverage)

I kept the v16 SegFormer checkpoint, 768-pixel tile size, float32 precision,
full-resolution input, Step 2 path, and regression checkpoint fixed, and changed
only the Step 1 tile overlap from 25% to zero. This reduced the typical tile
count from 35 to 24. The experiment ran the complete raw-to-output path on the
same 58-photo, 12-date holdout used for the YOLO26s candidate and compared
against archived v16 inputs on the 48 paired photos.

The candidate completed only 52/58 photos; six failed Step 2 crop selection.
On the 52 successes its MAE was 0.13611 (success-only and therefore
selection-biased). On the 48 paired successes, candidate MAE was 0.13921
versus 0.10855 for v16, a +0.03066 absolute delta (+28.2%). Error was worse
on 29 photos and better on 19; the 11-date cluster-bootstrap 95% interval was
[+0.00331, +0.04925]. This clearly fails the no-quality-hit gate and the
coverage gate.

Warm raw-to-flushed-record latency was 7,480 ms median / 7,517 ms p95, with a
7,770 ms maximum. This is only about a 1.5x end-to-end improvement versus the
roughly 11-second 35-tile reference, despite removing 11 tile forwards, and it
remains far above 100 ms. Peak process RSS was 2.97 GiB and peak accelerator
allocated/reserved memory was 8.31/8.62 GiB. No production settings changed.
The candidate is discarded. Full per-photo results and the runner are under
`/tmp/wave_v16_nooverlap_holdout_eval_v1/` and
`/tmp/wave_v16_nooverlap_holdout_eval_v1.py`.

### Larger overlapping tile-size sweep (rejected on MAE and coverage)

I tested two larger tile sizes with the v16 model, 25% overlap, float32
precision, the same Step 2 pipeline, and the same regression checkpoint. Each ran
the full raw-to-flushed-output pipeline on the 58-photo, 12-date holdout and
used the 48 archived canonical inputs for paired MAE comparison. The 1024-pixel
variant used 20 tiles on a 4032 x 3024 image; the 960-pixel variant used 24.

The 1024-pixel run succeeded on 55/58 photos. Its success-only all-photo MAE
was 0.12023; paired MAE was 0.12126 versus 0.10855 for v16 (+0.01271,
+11.7%). Errors were worse on 26 and better on 22; the 11-date bootstrap 95%
interval for the paired delta was [-0.01874, +0.03410]. Raw-to-flushed-record
latency was 6,114 ms median / 6,154 ms p95, with a 6,423 ms maximum. Peak RSS
was 2.94 GiB and accelerator allocated/reserved memory was 8.57/8.78 GiB.

The 960-pixel run succeeded on 54/58 photos. Its success-only MAE was 0.13081;
paired MAE was 0.13333 versus 0.10855 for v16 (+0.02478, +22.8%). Errors were
worse on 33 and better on 15; the paired date-bootstrap interval was
[-0.00629, +0.04709]. Latency was 7,169 ms median / 7,217 ms p95, with a
7,503 ms maximum. Peak RSS was 3.05 GiB and accelerator allocated/reserved
memory was 8.50/8.71 GiB.

Neither tile-size change preserves quality or full coverage, and neither is
close to 100 ms; both are discarded. The larger tile settings also did not
reduce peak GPU memory. No production settings changed. Artifacts are in
`/tmp/wave_v16_tile1024_overlap25_holdout_eval_v1/` and
`/tmp/wave_v16_tile960_overlap25_holdout_eval_v1/`, with runners
`/tmp/wave_v16_tile1024_overlap25_holdout_eval_v1.py` and
`/tmp/wave_v16_tile960_overlap25_holdout_eval_v1.py`.

### Water-only logit accumulation (rejected on quality and coverage)

The canonical pipeline only consumes a binary water mask from the 150-class
SegFormer output, but the overlapping-tile implementation accumulates all
150 float32 class logits over the full source image. I tested a memory-oriented
alternative: after each tile's exact bilinear upsample, take the maximum water
class score and maximum non-water class score, accumulate those two full-image
planes, then compare the accumulated scores. This keeps the v16 weights,
float32 model execution, tile size, overlap, and downstream regressor fixed;
it changes the overlap aggregation because max and averaging do not commute.

On a single image, the water-mask IoU was 0.99968 (1,291 pixels changed) and
the final waviness score matched exactly. That probe was not enough to establish
quality. On the 58-photo outer-date run, only 51/58 requests completed; seven
failed Step 2 crop selection. The success-only MAE was 0.12855 and is
selection-biased. On the 48 paired successes, candidate MAE was 0.12834 versus
0.10855 for v16 (+0.01979, +18.2%). Errors were worse on 29 photos and better
on 18. The date-cluster bootstrap 95% interval for the paired MAE delta was
[-0.00491, +0.04296]. This fails both the quality and coverage gates.

Warm raw-to-flushed-record latency was 9,874 ms median / 9,910 ms p95, with a
9,927 ms maximum; no request was under 100 ms. Peak accelerator allocation
fell from 8.31 GiB in the canonical v16 run to 1.60 GiB, and reserved memory
fell from 8.62 GiB to 2.22 GiB. Peak host RSS was 2.96 GiB. This is a
substantial memory reduction but only a small latency improvement, confirming
that the large accumulator was not the main latency bottleneck. The
approximation is discarded. No production code or profile changed. Records
and runner are under `/tmp/wave_v16_binary_accum_holdout_eval_v1/` and
`/tmp/wave_v16_binary_accum_holdout_eval_v1.py`.

### Exact tiled HEIF decoding on the GPU-standardized B0 candidate

Pillow's full-resolution HEIC decode was a large cost in the B0 candidate path.
I tested a native libheif tiled decoder that assembles the same BGR frame, then
converts to RGB before the unchanged B0 preprocessing. Direct comparison showed
exact RGB pixel equality with the repository's Pillow decoder on the first
8 held-out HEICs; the end-to-end run below provides a stronger output check.
Non-HEIF inputs continue through the existing Pillow loader.

I alternated Pillow and native decoding for all 47 validation and 46 test
photos, warming both branches first. Each timed request included decode, the
B0/448 GPU-standardized preprocessing and GPU erosion, the 448 Step 2 path,
regression, model-input JPEG writing, and a flushed JSONL result record. All
93 requests succeeded. Model-input JPEGs and regression predictions were
byte-identical between decoders on all 93; aggregate validation+test MAE was
0.12995 for both, so this decoder change caused no measured MAE shift.

Median full-request latency decreased from 288.25 ms to 248.32 ms, a paired
saving of 40.62 ms; p95 decreased from 301.46 to 261.81 ms. Peak process RSS
was 3.10 GiB and accelerator allocated/reserved memory was 0.428/1.322 GiB.
The native decoder itself took about 19–26 ms on the initial eight-image
parity probe, versus 94–149 ms for Pillow. On the 32 archived-v16 common rows,
the B0 GPU-standardized model's previously measured MAE was 0.14207 versus
0.15674 on validation and 0.12826 versus 0.13284 on test. The native decode
preserves those B0 predictions exactly on the full 93-row split. Those paired
B0 scores are better than v16 on these examined subsets, though the new
contract still requires future-capture confirmation.

This is the strongest exact-output speed improvement in the recent work, but
248 ms median remains 2.5x above target. The C++ source, compiled library,
per-image comparison, and report are temporary under `/tmp/wave_heif_tiled_decode_bgr_ctx.cpp`,
`/tmp/libwave_heif_tiled_bgr_ctx.so`, and
`/tmp/wave_b0_native_heif_decode_e2e_v3/`. No production decoder or checkpoint
was changed.

### Strict candidate screen against canonical v16

This screen compares each candidate's waviness MAE directly with canonical
v16 on the same labeled photos. An optimization that preserves the output of a
different experimental route does not pass on that basis alone. A point
estimate within 1% is provisional evidence, not proof of preserved quality;
for changed preprocessing or model contracts, the quality gate above still
requires independent date-held-out evidence that rules out a material hit.
Latency is complete warm raw-image-to-flushed-output time.

| Candidate | MAE: candidate vs canonical v16 on matched photos | Total latency | Status |
|---|---:|---:|---|
| **Current pipeline: canonical v16** | **0.10855** on the 48-photo paired reference cohort | **~11,300 ms median** | Current reference and `.env` selection. |
| MIGraphX FP32 for the complete canonical v16 path | 0.120066 vs 0.120065 on the same 51/58 successful photos (+0.0016%); same 7 failures, 49/51 final JPEGs and predictions exact | 4,969 ms median / 5,030 ms p95 vs PyTorch 10,058 / 10,092 ms; 0/58 below 100 ms | Passes the 1% paired-MAE screen and cuts total latency about 50%; experimental only, needs a separate long-lived engine process and has ~60 s engine startup. |
| Date-held-out thumbnail YOLO26 + current v16 head | 0.13226 vs 0.12006 on the 51 shared successes (+10.2%); candidate 58/58 vs v16 51/58 | 35.26 ms median / 41.94 ms p95 / 46.88 ms max; v16 10,057.6 ms median | Fails MAE screen despite full coverage and sub-100-ms latency. The 12-date paired MAE interval is [-0.01408, 0.03416]. |
| Thumbnail-domain YOLO26 + v18 head | 0.11821 vs 0.14349 on the 46-photo test split (−17.6%); on four date-disjoint photos from one day, 0.13984 vs 0.23457 (−40.4%) | 37.32 ms median / 40.04 ms p95 on 46; compiled-selector rerun 15.46 / 17.24 ms | Attractive but date-overlapping point estimates. The corresponding date-held-out student with current v16 head regressed 10.2%; this v18-head variant still needs direct clean-holdout evaluation. Do not adopt. |
| Thumbnail YOLO26 + MSE-adapted v16 head | 0.11612 vs 0.13284 on 32 shared test photos (−12.6%); full test MAE 0.11829 | 16.33 ms median / 18.25 ms p95 with compiled crop selector | Point estimate passes; same repeatedly examined date-overlapping cohort and wide paired uncertainty. Experimental. |
| SegFormer-B0 GPU-standardized route with exact native HEIF decode + nearest resize | 0.12826 vs 0.13284 on 32 shared test photos (−3.4%); validation 0.14207 vs 0.15674 on 32 shared photos (−9.4%) | 227.35 ms median / 245.61 ms p95 | Point estimates pass; above 100 ms and not independently confirmed. Experimental. |

The prior version of this ledger incorrectly retained exact-output speed
optimizations because their MAE matched an experimental route, without first
checking that route against canonical v16. Those rows are removed from this
quality-qualified list. For example, the full-resolution YOLO26 route's test
MAE was 0.14470 versus canonical v16 0.13284 on 32 shared photos (+8.9%), even
though its resize-only comparison preserved that route's own MAE. It fails the
requested quality screen. Other experiments above that exceed the 1% v16 MAE
regression, have incomplete coverage, or lack a matched quality comparison are
also excluded. No candidate currently has enough independent evidence to
replace v16 while guaranteeing no quality hit; keep v16 selected.

#### OpenCV exact-nearest resize and combined decoder trial

OpenCV `INTER_NEAREST_EXACT` matched Pillow byte-for-byte on a synthetic
448-to-4032x3024 class-map probe. I first ran an invalid variant that forced
Step 2's 224x224 image through a 448x448 logits resize; its outputs differed
and those timings/MAEs are discarded. The corrected implementation follows the
existing size rule at both stages. On the actual 93-photo B0 validation+test
cohort, the resize-only A/B produced byte-identical Step 2 JPEGs and identical
predictions on all 93. Combined MAE was 0.12995 in both arms. Median total
raw-to-record latency was 270.15 ms versus 289.87 ms for Pillow resize; p95 was
288.88 versus 305.43 ms. This isolates an exact OpenCV resize change using the
same Pillow HEIF decoder on both arms.

I then paired that exact resize with the previously verified native tiled HEIF
decoder. Again all 93 model-input JPEGs and predictions matched exactly, with
MAE 0.12995 in both arms. Complete raw-to-flushed-record latency was 227.35 ms
median / 245.61 ms p95, versus 286.93 / 304.04 ms for the Pillow decoder and
resize reference. Paired median saving was 59.74 ms. Peak RSS was 3.12 GiB and
accelerator allocated/reserved memory was 0.428/1.322 GiB. This confirms a
quality-preserving speed gain for the B0 experimental route, but does not
establish an adopted v16 replacement and remains 2.27x above the 100 ms target.
No repository decoder or checkpoint was changed. The 93-photo resize-only and
combined runners/results are under `/tmp/wave_b0_nearest_resize_only_e2e_v2.py`,
`/tmp/wave_b0_nearest_exact_resize_only_e2e_v2/`,
`/tmp/wave_b0_exact_decode_and_resize_e2e_v1.py`, and
`/tmp/wave_b0_exact_decode_and_resize_e2e_v1/`.

The decoder worker sweep held the exact B0 preprocessing and single-threaded
PyTorch/OpenCV settings fixed. Sixteen workers reduced median request latency
by 8.13 ms versus eight workers (242.43 vs 250.88 ms) but increased median
process CPU time by 29.86 ms (434.48 vs 404.63 ms). Both settings had identical
outputs for all 93 samples. This is a measured throughput tradeoff rather than
a universally better thread setting: 16 wins for single-request latency; 8
uses less CPU. The canonical v16 route has a different profile: 35 SegFormer-B2
tile forwards dominate its roughly 11-second request, and one versus 32
PyTorch threads changed the warm median only from 10.79 to 11.22 seconds. CPU,
shared-memory and thread tuning cannot substitute for reducing segmentation
compute while retaining canonical quality. For accelerator tuning, AMD's ROCm
workflow is to profile kernel utilization, bandwidth and occupancy, optimize
the measured bottleneck, then re-profile; PyTorch likewise cautions against
unnecessary CPU/GPU synchronization and transfers ([PyTorch tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html),
[ROCm performance guidelines](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html)).

### Date-grouped out-of-fold calibration for the sub-100-ms YOLO26s route

The full-resolution YOLO26s + teacher-distilled regressor is the only recent
route below 100 ms without reducing source resolution, but its paired MAE was
0.12081 versus 0.10855 for v16 on 48 shared photos. I tested whether score
calibration could correct systematic bias without changing images or adding
meaningful inference time. To avoid fitting calibration on the outer dates, I
generated predictions from five capture-date-grouped out-of-fold regressor fits
using 157 non-outer-date training photos from 39 dates. Each fold held out about
31–32 photos across dates, trained the same partial EfficientNet fine-tune for
the previously training-selected 14 epochs, and used the label plus v16
teacher-score loss where teacher targets existed. Calibration was fitted only
on OOF predictions; the 58 outer-date photos were not used to estimate it.

Ordinary least-squares affine calibration produced slope 0.9646 and intercept
0.0123. OOF MAE changed from 0.14808 to 0.14879. On the outer dates, raw MAE
was 0.12539 and affine-calibrated MAE was 0.12593; on the 48 paired photos the
figures were 0.12081 and 0.12177. The affine correction worsened both point
estimates and is rejected. A one-parameter median-residual shift (+0.00585,
estimated from the same OOF predictions) lowered paired candidate MAE slightly
to 0.12037, but it remained above v16's 0.10855; candidate-minus-v16 MAE was
+0.01181, with an 11-date-cluster bootstrap 95% interval of [-0.01421,
+0.03715]. This does not establish quality preservation. Applying either
correction is a pair of scalar operations (<0.01 ms estimated), so the route
would remain about 64 ms median, but speed does not compensate for the quality
gap. The outer dates have already been reviewed, so this is exploratory
evidence, not an independent final test. No checkpoint or production
configuration changed. OOF predictions, paired outer results and summary are
in `/tmp/wave_yolo26s_oof_affine_calibration_v2/`; the runner is
`/tmp/wave_yolo26s_oof_affine_calibration_v2.py`.

### Teacher-loss weight sweep for the fast full-resolution regressor

Because the score calibration did not recover the v16 MAE, I next varied the
training-time v16 teacher-loss weight while keeping the full-resolution YOLO26s
input contract and inference graph fixed. Six weights (0, 0.25, 0.5, 1, 2,
and 4) were selected on the same 39-photo, 13-date inner training validation
set; the excluded 58-photo outer holdout was not used for selection. The 0.5
weight gave inner MAE 0.08987 at epoch 11, essentially tied with weight 0
(0.08988 at epoch 12); weights 1, 2 and 4 scored 0.09061, 0.09957 and 0.10096.
The small gap between weights 0 and 0.5 is not meaningful evidence that
distillation helps, especially because the trial used different random seeds
for each weight.

I refit the training-selected 0.5/11-epoch setting on all 157 non-outer-date
photos and evaluated the resulting checkpoint on the 58 excluded outer-date
photos. All 58 completed. Candidate MAE was 0.12541 on all 58 and 0.12197 on
the 48 photos with archived v16 inputs; v16 MAE on those same photos was
0.10855. Candidate-minus-v16 MAE was +0.01342, with 24 per-photo errors worse
and 24 better; the 11-date-cluster bootstrap 95% interval was [-0.01285,
+0.03794]. This refit did not improve on the earlier 0.5-weight candidate
(paired MAE 0.12081), and neither version establishes no quality hit.

Complete raw-to-flushed-record median/p95/max was 64.74/73.37/80.67 ms, all
58 requests under 100 ms. Median process CPU time was 242.78 ms, peak RSS
2.57 GiB, and accelerator allocation/reservation 0.343/0.383 GiB. The
regressor architecture and inference work are unchanged, so the route remains
fast while the quality gate still fails. Do not adopt. The corrected outer
evaluation used the full raw-image path and a separate canonical v16 model for
the paired baseline. The first sweep runner's training-only selection is
usable, but its outer-data list was empty because it looked for outer inputs in
the training cache; its outer metrics are invalid and discarded. Correct
outer records are under `/tmp/wave_yolo26s_distill_weight_sweep_outer_eval_v1/`
and the runner is `/tmp/wave_yolo26s_distill_weight_sweep_outer_eval_v1.py`;
selection logs/checkpoint are under `/tmp/wave_yolo26s_distill_weight_sweep_v1/`.

### Removing the class-map copy after OpenCV resize

The exact OpenCV resize returns a newly allocated contiguous array. I removed
the subsequent full-resolution `resized.copy()` and wrapped that buffer
directly with `torch.from_numpy`, which shares its storage. This avoids one
12.2 MB allocation/copy per 4032 x 3024 class map. On the same combined B0
decoder+resize route, all 93 model-input JPEGs and predictions remained
identical; aggregate MAE stayed 0.12995. Median request latency changed from
228.40 to 226.52 ms, but the paired median saving was only 0.27 ms; p95 was
243.44 vs 243.26 ms. Median process CPU time changed from 672.63 to 670.86 ms.
Peak RSS was 2.64 GiB for the combined process, so this run cannot establish a
separate per-arm RSS reduction. This is safe buffer sharing but too small a
latency gain to matter to the 100 ms gap. I did not integrate it into the
production pipeline; the prototype is
`/tmp/wave_b0_resize_zero_copy_e2e_v1.py` and its records are in
`/tmp/wave_b0_resize_zero_copy_e2e_v1/`.

### Blend the fast regressor with v16 predictions on the fast inputs

I tested whether a second v16 regressor pass on the already-generated 224x224
YOLO26s candidate input could recover quality. The blend weight was selected
only from the 157 date-grouped OOF training predictions (39 dates): 0.75 fast
candidate and 0.25 v16-on-fast-input. OOF MAE improved from 0.14808 for the
candidate alone to 0.14658. On the 48-photo matched outer cohort, however,
blend MAE was 0.12080, essentially unchanged from candidate MAE 0.12081 and
still above canonical v16 MAE 0.10855. The paired blend-minus-v16 MAE gap was
+0.01225; the capture-date-cluster bootstrap 95% interval was [-0.00808,
+0.03145]. This does not establish quality preservation. The extra v16
forward's end-to-end latency was not measured, so this variant is rejected
without a latency claim. Predictions and summary are in
`/tmp/wave_yolo26s_oof_fasthead_blend_v1/`; runner:
`/tmp/wave_yolo26s_oof_fasthead_blend_v1.py`.

### 1024-pixel inference for the full-resolution YOLO26s student

To test whether finer student masks would improve the fast full-resolution route,
I kept the trained YOLO26s checkpoint, raw full-resolution decode, `reducing_gap=3`
standardization, Step 2, v16 regression checkpoint, and all 58 date-holdout
photos fixed, and changed only inference `imgsz` from 640 to 1024. This is an
inference-size override; the model was trained at 640. Ultralytics documents
`imgsz` as the inference letterbox target and permits overriding it at
prediction time ([predict arguments](https://docs.ultralytics.com/modes/predict)).

The 1024 route completed 57/58 photos; one failed Step 2 crop selection. On the
48-photo matched cohort its MAE was 0.13027 versus 0.12624 for the 640 route and
0.10855 for canonical v16. Candidate-minus-v16 MAE was +0.02171, with an
11-date-cluster 95% interval of [-0.01766, +0.04871]. The 1024 inference did
not improve paired quality, and coverage fell from 58/58 to 57/58.

For the 57 successful requests, complete warm raw-to-flushed-record latency was
71.34 ms median / 91.72 ms p95; 55/57 were below 100 ms. The 857.21 ms maximum
was one outlier (the third request); median process CPU was 259.48 ms, peak RSS
2.60 GiB, and accelerator allocated/reserved memory 0.334/0.420 GiB. The
640-pixel reference on this route was 64.6 ms median / 75.4 ms p95 with 58/58
successes. Higher inference resolution therefore costs latency and coverage
without reducing the MAE gap. Reject 1024; no production setting changed.
Runner and complete per-photo results are under
`/tmp/wave_yolo26s_fullres_dateholdout_v3_gap3_1024_eval_v1.py` and
`/tmp/wave_yolo26s_fullres_dateholdout_v3_gap3_1024_eval_v1/`.

### 512-pixel inference for the full-resolution YOLO26s student

I also tested a lower inference size on the same fixed 640-trained checkpoint
and 58-photo date holdout, changing only `imgsz` to 512. The route completed
57/58; one photo failed Step 1 water-area selection. On the 48-photo matched
cohort, MAE was 0.13752 versus 0.12624 at 640 and 0.10855 for v16. The
candidate-minus-v16 gap was +0.02896 (11-date-cluster 95% interval
[-0.00245, +0.04804]); 32 paired errors worsened and 16 improved.

Complete warm raw-to-flushed-record latency was 62.14 ms median / 77.93 ms p95
for 57 successful requests; 56/57 were under 100 ms. Median CPU time was
239.82 ms, peak RSS 2.63 GiB and accelerator allocated/reserved memory
0.327/0.352 GiB. Compared with the matching 640-pixel v3 reference (63.95 ms median / 76.27 ms
p95, 58/58 success), lowering inference resolution made latency 1.82 ms faster
at the median while worsening paired MAE by 0.01128 and losing one successful
input. Reject 512. Runner and records are under
`/tmp/wave_yolo26s_fullres_dateholdout_v3_gap3_512_eval_v1.py` and
`/tmp/wave_yolo26s_fullres_dateholdout_v3_gap3_512_eval_v1/`.

### Select the YOLO26s student epoch by downstream waviness MAE

The earlier student selected its checkpoint by validation water-mask mIoU. I
ran a date-grouped selection experiment to align checkpoint choice with the
actual task: train the same YOLO26s ADE20K student for 30 epochs on 136
non-outer-date images, save every epoch, then score each epoch through the full
640-pixel raw-image preprocessing route and frozen v16 regressor on 23 photos
from five separate inner validation dates. The outer 58 photos were excluded
from training and epoch selection. This was an independent fresh training run;
I verified that the saved epoch weights differ and evaluated each checkpoint
with a fresh model instance. The minimum complete-validation MAE was epoch 26:
0.10949, versus 0.11208 for the existing mask-mIoU-selected checkpoint on the
same 23 photos. This is a small inner-set gain, not an outer quality claim.

I refit the selected 26-epoch schedule on all 159 non-outer-date teacher-mask
images and evaluated it on the 58 outer photos. The route completed 55/58; three
failed Step 2 crop selection. On the 55 successful rows its MAE was 0.14284,
versus 0.12839 for the earlier mIoU-selected student on those same 55 rows.
On the 47 rows paired with archived v16 inputs, candidate MAE was 0.13690
versus 0.10284 for v16: a +0.03405 gap, with an 11-date-cluster 95% interval
[+0.00934, +0.04954]. The wave-MAE selection did not transfer across dates;
this refit is worse than the previous student and clearly fails the quality
and coverage requirements.

Warm complete raw-to-flushed-record latency was 62.97 ms median / 76.92 ms
p95; all 55 successful requests were under 100 ms. The mIoU-selected student
measured 64.26 ms median / 76.39 ms p95 on those same 55 successful photos,
so the refit is about 1.3 ms faster at the median while substantially
worsening MAE and losing three requests. No model or pipeline setting changed.
The 58 outer dates have been examined in earlier research, so these figures are
exploratory rather than an untouched final test. Training, per-epoch selection
records, refit checkpoint, and outer evaluation are under
`/tmp/wave_yolo26s_downstream_mae_select_v1/`; runner scripts are
`/tmp/wave_yolo26s_downstream_mae_select_train_v1.py`,
`/tmp/wave_yolo26s_downstream_mae_select_eval_v2.py`,
`/tmp/wave_yolo26s_downstream_mae_select_refit_v1.py`, and
`/tmp/wave_yolo26s_downstream_mae_select_outer_eval_v1.py`.

### YOLO26m semantic student, selected on downstream waviness MAE

I tested whether a larger semantic student could improve the mask and final
score without exceeding the request budget. Ultralytics' ADE20K table lists
YOLO26m-sem at 47.4 mIoU and 4.7 ms model time versus 45.6 and 4.2 ms for
YOLO26s-sem on an RTX 3090; these published values are not predictions for
this sea-photo task or the Radeon host ([semantic model table](https://docs.ultralytics.com/tasks/semantic)). As a local preflight, the unadapted ADE20K models were compared with canonical binary teacher masks on 23 inner-date photos. Mean mask IoU was 0.7500 for s and 0.7559 for m. For these 150-class models, BGR input to full source-sized semantic mask output took 91.58 ms median / 94.99 ms p95 for s and 98.01 / 108.59 ms for m; raw decode and all remaining pipeline stages were not included. The slight mask gain and high output cost did not alone justify this model size.

I nevertheless fine-tuned YOLO26m from ADE20K on the same 136 training images, saved each of 30 epochs, and selected by complete candidate-path waviness MAE on the same 23 photos across five inner validation dates. The best complete epoch was 12, with MAE 0.10090 and 23/23 successes; YOLO26s' best score was 0.10949 on the same inner photos. I refit the selected 12-epoch schedule on all 159 non-outer teacher-mask images and evaluated the 58 outer photos.

The refit completed 56/58. On those 56 successful photos its MAE was 0.15518, versus 0.12815 for the existing YOLO26s refit on the exact same successful cohort. Against archived canonical v16 on 47 paired successes, m MAE was 0.14323 versus 0.10327 for v16, a +0.03996 gap; the 11-date-cluster 95% interval was [+0.00421, +0.06240]. The model was worse than v16 on 31 photos and better on 16. The apparent inner-validation gain did not transfer and this larger student is rejected.

Warm complete raw-to-flushed-record latency was 69.26 ms median / 83.07 ms p95, with 56/56 successful requests under 100 ms. The matching YOLO26s reference measured 64.26 / 76.39 ms on the same successful photos. Median CPU time was 248.32 ms, peak RSS 2.60 GiB, and accelerator allocated/reserved memory 0.358/0.416 GiB. No pipeline or checkpoint changed. The outer dates have been examined in prior experiments and are not an untouched final test. Training and evaluation artifacts are under `/tmp/wave_yolo26m_downstream_mae_select_v1/`; preflight artifacts are under `/tmp/wave_yolo26m_sem_preflight_v1/`. Runners: `/tmp/wave_yolo26m_downstream_mae_select_train_v1.py`, `/tmp/wave_yolo26m_downstream_mae_select_eval_v1.py`, `/tmp/wave_yolo26m_downstream_mae_select_refit_v1.py`, and `/tmp/wave_yolo26m_downstream_mae_select_outer_eval_v1.py`.

### Five-fold date-grouped YOLO26m epoch check

The prior YOLO26m refit scored well on one small inner split (MAE 0.10090
on 23 photos from five dates) but regressed on its outer evaluation. To check
whether the epoch choice was split-specific, I trained five YOLO26m folds on
127–128 photos each and held out all photos from 39 non-outer capture dates,
then evaluated each of 12 checkpoints per fold through the complete candidate
preprocessing and frozen-v16 regression path. Training took 875.7 seconds in
total; evaluation took about 35 seconds per fold. This is grouped cross-
validation inside the existing development dates, not a pristine test set.

Pooled out-of-fold results are:

| Epoch | Successful photos | Pooled MAE | Coverage |
|---:|---:|---:|---:|
| 1 | 159/159 | 0.19392 | 100% |
| 2 | 156/159 | 0.17223 | 98.1% |
| 3 | 158/159 | 0.15516 | 99.4% |
| 4 | 159/159 | 0.16716 | 100% |
| 5 | 158/159 | 0.16213 | 99.4% |
| 6 | 158/159 | 0.15131 | 99.4% |
| 7 | 159/159 | 0.17880 | 100% |
| 8 | 159/159 | 0.16095 | 100% |
| 9 | 159/159 | 0.15558 | 100% |
| 10 | 159/159 | 0.15301 | 100% |
| 11 | 159/159 | 0.15405 | 100% |
| 12 | 159/159 | 0.14749 | 100% |

Epoch 12 has the lowest pooled MAE among complete-coverage checkpoints, but
its MAE is still materially worse than the canonical v16 MAE 0.10855 previously
measured on a separate 48-photo paired cohort. Those are different cohorts, so
the difference is indicative rather than a matched regression estimate. The
prior YOLO26m outer result also had MAE 0.14323 versus v16 0.10327 on 47 paired
photos (+38.7%), with 56/58 complete requests. The pooled date-grouped result
does not support refitting or deployment. The previously measured full raw-to-
output latency for the YOLO26m route was 69.26 ms median / 83.07 ms p95; no new
latency measurement was made in this epoch-selection run. The candidate is
rejected by the requested 1% MAE regression screen. No repository model or
runtime configuration changed.

Fold split, checkpoints, per-epoch predictions, failures, and pooled results
are under `/tmp/wave_yolo26m_date_oof_v1/`; training and evaluation runners are
`/tmp/wave_yolo26m_date_oof_train_v1.py` and
`/tmp/wave_yolo26m_date_oof_eval_v1.py`.

### NumPy tile views instead of copying PIL crops on the v16 route

I tested whether `SegformerImageProcessor` could consume NumPy views into the
already decoded RGB image instead of allocating a new PIL crop for each of the
35 Step 1 tiles. For all 35 tiles of `IMG_7238.HEIC`, the processor's output
tensors were bit-identical. In an alternating six-repeat processor-only loop,
median preparation time fell from 205.5 ms to 189.7 ms (15.8 ms saved before
GPU transfer and model execution).

I then alternated four full raw-to-output requests per arm in the same process,
using the official v16 models and writing prediction records to `/tmp`. The
final Step 2 JPEG SHA-256 and v16 prediction matched exactly on every request.
Median total latency was 10,206.9 ms for PIL crops and 10,225.8 ms for NumPy
views; the 18.9 ms difference is a small regression and does not show an
end-to-end gain. This is one source photo, so it does not produce a meaningful
MAE estimate; exact output equality on this photo gives an absolute error of
0.14870 (label 0.50), but that is not an MAE cohort. I did not integrate the
view-based path. The measured bottleneck remains tile-model compute, which is
about 9 seconds of the roughly 10-second request. No model, CPU thread setting,
or memory policy changed.

Tensor checks and the full request records are under
`/tmp/wave_v16_tile_view_bench.py`,
`/tmp/wave_v16_tile_view_full_bench.py`,
`/tmp/wave_v16_tile_view_alt_ab.py`, and
`/tmp/wave_v16_tile_view_alt_ab.json`.

### Lower-half-only segmentation screen (rejected on quality)

The project guidance says waviness is usually most visible in the lower part of
the sea, so I screened a bottom-half Step 1 region on one photo from each of 11
capture dates. The model and all downstream v16 steps were unchanged; omitted
pixels were assigned non-water class 0 and the candidate mask was restored to
the original image dimensions. This reduced Step 1 from 35 tiles to 21.

Ten of 11 photos completed; one failed preprocessing. On the 10 shared
successes, candidate MAE was 0.13636 versus 0.09921 for canonical v16 on those
same photos (+0.03715 absolute, +37.4%). Complete raw-to-prediction latency,
including preview and prediction-record writes, was 6,031.5 ms median / 6,491.6
ms p95 across the 10 successes. This is a large latency reduction but clearly
fails the quality and coverage gates, so the crop is rejected. The MAE is
success-only and may be optimistic because one photo failed. No code or profile
changed. Per-photo results are in `/tmp/wave_bottom_half_screen_with_output.json`;
the temporary runner is `/tmp/wave_bottom_half_screen_with_output.py`.

### Bottom-80%-only segmentation screen (rejected by the 1% MAE gate)

I repeated the lower-region check while excluding only the top 20% of each
source photo, using one image from each of the same 11 capture dates. The full
v16 model and downstream pipeline stayed fixed; the segmentation grid used
28–30 tiles for these inputs, compared with 35 tiles for the standard
4032x3024 image. All 11 requests completed.

On these 11 photos, candidate MAE was 0.09714 versus canonical v16 MAE 0.09055
on the same rows, a +0.00658 absolute / +7.3% regression. Complete
raw-to-prediction latency, including preview and prediction-record writes, was
8,594.3 ms median / 9,193.7 ms p95. This gentler crop preserves coverage and
saves some segmentation work, but still exceeds the requested 1% quality limit
and remains far from 100 ms. It is rejected without a larger run. Results are
in `/tmp/wave_bottom_80pct_screen_with_output.json`; runner:
`/tmp/wave_bottom_80pct_screen.py`.

### Reuse the canonical Step 1 water mask for Step 2 (rejected by the 1% MAE gate)

I screened the existing projected-mask path with the canonical v16 B2
preprocessor and regressor on one photo from each of the same 11 capture dates.
This removes Step 2's second SegFormer call by projecting the first raw-image
mask into Step 1's standardized image coordinates. All 11 requests completed.
The projected mask reproduced the archived v16 prediction exactly on only 4/11
photos; median absolute prediction shift was 0.0100 and maximum shift was
0.0453. Candidate MAE was 0.09616 versus canonical v16 MAE 0.09055 on the same
11 photos (+0.00560, +6.2%), beyond the permitted 1% regression.

The Step 2 stage fell from roughly 0.29 s to 0.02–0.04 s on these inputs, but
complete raw-to-prediction latency for the projected route was 9,915.1 ms
median / 10,400.4 ms p95. That stage saving cannot approach the 100 ms target,
and the changed Step 2 crops fail the quality gate. Reject; no runtime or
checkpoint changed. Per-photo outputs and complete prediction records are in
`/tmp/wave_projected_mask_reuse_screen.json` and
`/tmp/wave_projected_mask_reuse_screen/`; runner:
`/tmp/wave_projected_mask_reuse_screen.py`.

### Ryzen AI NPU availability and runtime check

I rechecked the host's NPU because the canonical B2 tile forwards, not CPU
copying, dominate current latency. `/dev/accel/accel0` is present as
`RyzenAI-npu5`; `xrt-smi validate` passed its generic GEMM, latency, and
throughput checks, reporting 51.0 TOPS and 72 microseconds average latency for
its built-in latency test. Those figures are hardware diagnostics, not model
inference measurements, and do not establish that this pipeline can run under
100 ms.

No Ryzen AI ONNX execution provider or model compiler is installed in the
project environment. The host has XRT 2.21.75, while AMD's current Linux guide
uses XRT 2.25.37 plus the matching XDNA plugin and a separately downloaded
Ryzen AI 1.8 package; it specifies Python 3.12 for the runtime (available on
this host). AMD documents Linux NPU flows for CNN INT8/BF16 and selected BF16
encoder models, but does not document SegFormer as a supported Linux NPU
example. The official runtime download redirects to AMD account access here,
so I have not installed or changed the system runtime. The official RyzenAI-SW
1.8.0 GitHub release has no runtime assets; its Linux guide directs users to
the gated AMD download. The public XDNA driver source supports Ubuntu >=22.04
and kernel >=6.10, which matches this host, but that alone is not the model
compiler/execution provider. Local package inventory confirms the same gap:
XRT libraries and `xrt-smi` are installed, while the project virtual environment
has neither ONNX Runtime nor a VitisAI execution provider. A runtime version
change would also need a separately controlled compatibility evaluation; I
did not replace the working host XRT/driver stack.

The NPU is therefore a credible future experiment for a CNN water-mask student,
but no speed or MAE claim is available yet. Any NPU run must measure complete
raw-to-record latency and compare MAE against the current v16 checkpoint; the
built-in NPU TOPS result must not be used as a substitute. Sources:
[AMD Ryzen AI Linux installation and runtime guide](https://github.com/amd/ryzen-ai-documentation/blob/main/docs/linux.rst),
[AMD XDNA Linux driver requirements](https://github.com/amd/xdna-driver).

### Date-disjoint check of the sub-100-ms thumbnail v18 route

I audited the existing 46-photo thumbnail-domain v18 test records against EXIF
capture dates and the training/selection data before using them as quality
evidence. The full test set was not date-independent: 26 of its 27 dates
appeared in v18 regression or thumbnail-student train/validation data. However,
four test photos from 2026-08-24 formed one date absent from the v16 training
split, the v18 regression train/validation rows, and the thumbnail student
train/validation rows.

On those four date-disjoint photos, the thumbnail YOLO26 + v18 route had MAE
0.13984 and RMSE 0.18036; canonical v16 on the same photos had MAE 0.23457 and
RMSE 0.28342. The candidate reduced absolute error on 3/4 photos. The Python
crop selector run measured 31.21 ms median / 38.55 ms p95. The later compiled
selector rerun produced the same four predictions exactly and measured 15.63
ms median / 16.65 ms p95 / 16.71 ms maximum; all four requests were below 100
ms. Across the full, date-overlapping 46-photo test set, the
route had MAE 0.11821 vs v16 0.14349 and latency 37.32 ms median / 40.04 ms p95;
that aggregate is exploratory because most capture dates overlap training.

This is positive but very limited evidence: four photos come from one capture
date and cannot establish performance across days, spots, or sea conditions.
The route is the strongest current speed candidate, but it is not yet cleared
as a quality-preserving replacement. Keep v16 as the production selection and
repeat this paired MAE/latency gate as new labeled capture dates arrive. Exact
per-photo records are in
`/tmp/wave_yolo26_student_thumbnail_teacher_heif_test_eval/results.csv` and
`/tmp/wave_canonical_v16_full_split_comparison_detailed.json`; the date audit
used the v16/v18 split manifests and the thumbnail student train/validation
image lists.

### Compiled crop selector on the thumbnail v18 route

I reran the 46-photo test split through both the Python Step 2 crop selector
and its compiled C++ implementation, with the same thumbnail-domain YOLO26
student and v18 regression checkpoint. All 46 crop boxes and all 46 final
predictions were identical. Both arms therefore had MAE 0.11821; this is
unchanged from the Python reference and remains 17.6% lower than canonical v16
(MAE 0.14349) on the same date-overlapping test split.

Complete warm raw-to-flushed-output latency was 37.52 ms median / 39.75 ms p95
with Python crop selection and 15.46 ms / 17.24 ms with the compiled selector,
a 58.8% lower median. All 46 requests in each arm completed below 100 ms. This
is the strongest measured end-to-end result so far, with exact output
preservation. Median process CPU time fell from 38.04 to 15.66 ms; peak RSS
remained 2.31 GiB and PyTorch accelerator allocated/reserved memory remained
0.119/0.152 GiB. The quality caveat remains: most test capture dates overlap
training, and the separate date-disjoint evidence is only four photos from one
day. On those four photos, the compiled route retained MAE 0.13984 versus
canonical v16's 0.23457 and completed in 15.63 ms median / 16.65 ms p95; this
is encouraging but too small a sample to establish quality preservation. This
optimization has not been integrated into the production pipeline;
v16 remains selected. Per-photo compiled-arm results are in
`/tmp/wave_thumbnail_cppcrop_test_eval/results.csv`; Python-arm results are in
`/tmp/wave_yolo26_student_thumbnail_teacher_heif_test_eval/results.csv`.

### In-place RGB decode: exact-output end-to-end check (2026-10-28)

To validate the current `load_rgb_image()` copy reduction against the prior
`ImageOps.exif_transpose(...).convert("RGB")` path, I compared both decoders
on all 58 raw images in the date-grouped holdout. HEIF decode used 16 workers;
JPEG inputs remained on the same Pillow path. All 58 oriented RGB pixel buffers
were byte-identical. In separate processes, decode median/p95 fell from
68.76/82.82 ms to 57.11/63.90 ms. Median process CPU time fell from
189.31 to 175.52 ms per image, and peak process RSS fell from 720,764 to
679,728 KiB (about 40 MiB). This supports a modest decode and memory benefit;
it does not reduce the dominant segmentation work.

I then ran the complete warmed canonical v16 raw-to-output path on the first
six same-date-holdout rows, alternating decoder order by image. Each timed
request included both 35-tile Step 1 segmentation and Step 2, model inference,
model-input JPEG persistence, and a flushed/fsynced prediction record. All six
final Step 2 JPEGs and waviness predictions matched exactly. MAE was 0.13558
with either decoder on these six photos; this small subset is a parity check,
not a new quality estimate. Median total latency was 10,038.3 ms with the old
decoder and 10,023.9 ms with in-place decode; paired median saving was 13.1 ms
(about 0.13%). The six-request p95s were 10,067.9 and 10,045.0 ms; neither arm
had a request below 100 ms. This is a measured exact-output improvement, but
far too small to close the target gap. Full per-image records and runner are in
`/tmp/wave_inplace_decode_e2e_ab_v1/` and `/tmp/wave_inplace_decode_e2e_ab.py`;
the 58-image isolated decode records are in
`/tmp/wave_inplace_decode_bench_old.json` and
`/tmp/wave_inplace_decode_bench_new.json`.

### CPU tile prefetch while SegFormer runs (rejected; 2026-10-28)

I prototyped one-tile-ahead CPU crop/processor work in a worker thread while
keeping the canonical FP32 model forward, all tile coordinates, interpolation,
logit accumulation order, and Step 1/Step 2 JPEG contracts unchanged. On the
same first six date-holdout photos, Step 2 JPEGs and final predictions were
identical; both arms had MAE 0.13558. However, complete warm raw-to-fsynced-
output median latency increased from 9,909.9 ms to 9,930.4 ms, a 20.5 ms paired
median slowdown. The six-photo p95 was 9,938.1 vs 9,951.9 ms, and none of the
requests met 100 ms. Median process CPU time rose from 10,205 to 10,752 ms.
Peak RSS for the mixed run was 2.93 GiB and accelerator allocation/reservation
was 8.31/8.62 GiB. Threaded image processing did not overlap usefully on this
workload; I did not integrate it. The first prototype attempt omitted
`inference_mode` and retained autograd graphs; that run's OOM is discarded.
The corrected run and records are under
`/tmp/wave_v16_tile_prefetch_e2e_ab_v1.py` and
`/tmp/wave_v16_tile_prefetch_e2e_ab_v1/`.

### Fused SDPA for canonical FP32 SegFormer attention (rejected; early stop)

PyTorch 2.14 on ROCm 7.16 reports its memory-efficient scaled-dot-product
attention backend as available for the canonical FP32 attention tensor shape.
I patched SegFormer attention in a temporary runner to use
`scaled_dot_product_attention`, forced that FP32 backend, and left weights,
inputs, tiling, precision, and downstream processing unchanged. On the first
two matched raw photos, both final JPEGs and waviness predictions were exactly
the same as baseline; the paired two-photo MAE was 0.09145 in both arms. The
first two total raw-to-flushed-output latencies were 10,106.8 and 10,065.2 ms
for baseline, versus 10,293.6 and 10,241.8 ms with fused SDPA. That is a
181.7 ms median slowdown on these two pairs. I stopped the run because the
candidate was consistently slower, so this is an early rejection rather than a
full cohort result; no production source or checkpoint changed. An exact
attention output does not ensure a speed gain: PyTorch dispatches among fused
and math kernels according to supported tensor shapes and backend, so the
measured device result controls. PyTorch reference:
[scaled-dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).
The runner is `/tmp/wave_v16_sdpa_e2e_ab_v1.py`; partial metrics are in
`/tmp/wave_v16_sdpa_e2e_ab_v1/partial_results.json`, with exact Step 2 JPEGs
and records under `/tmp/wave_v16_sdpa_e2e_ab_v1/`.

### MIGraphX FP32 for the full canonical v16 request (2026-10-28)

The prior MIGraphX check covered one SegFormer tile. I now ran the complete
baseline-v16 raw-to-output request through a persistent ONNX Runtime 1.29 /
MIGraphX 2.18 FP32 server, with the unchanged PyTorch v16 regressor. All 768 x
768 Step 1 and Step 2 segmenter calls used MIGraphX; Pillow decode, processor
inputs, all 35 overlapping Step 1 tiles, bilinear logit interpolation and
accumulation order, water-mask extraction, crop selection, JPEG round trips,
regression, preview persistence, and flushed/fsynced prediction records stayed
on the canonical path. Every tile's input and output crossed a loopback socket
because the installed ROCm PyTorch and MIGraphX runtimes cannot safely share a
process. The measured request includes that bridge. MIGraphX session creation
and graph compilation took 60.4 seconds at service startup and are reported
separately from warmed raw-to-output latency.

On the complete 58-photo, 12-capture-day cohort, both routes succeeded on the
same 51 photos and failed the same seven Step 2 crops. On the 51 paired
successes, canonical PyTorch MAE was 0.12006452 and MIGraphX MAE was 0.12006648:
a +0.00000196 absolute delta, or +0.0016%. The 12-day cluster bootstrap 95%
interval for the MAE delta was [-0.00000494, +0.00000862]. Forty-nine of 51
final model-input JPEGs and predictions were byte/value identical. The two
remaining predictions shifted by +0.0000724 (lower absolute error) and
-0.0001723 (higher absolute error); their JPEGs differed at 196 and 417 RGB
pixels, with maximum channel difference 5. The paired MAE change is far below
the 1% screen, and the failure count is unchanged.

Complete warm raw-to-flushed-output median latency was 10,057.6 ms for
PyTorch and 4,969.3 ms for MIGraphX, a 2.02x speedup and 5,075.5 ms paired
median saving. The p95 was 10,092.3 vs 5,030.3 ms; no request in either route
was below 100 ms. MIGraphX's model-only median was 91.6 ms per 768 tile across
2,088 measured calls. Client process median CPU time was 1,725.9 ms; server
process CPU and memory were not captured, so client CPU and memory do not
represent total service resource use. Client peak RSS was 2.79 GiB and
PyTorch accelerator allocation/reservation was 7.62/7.67 GiB, excluding the
MIGraphX server.

This is the strongest measured full-resolution, quality-preserving speedup so
far, but it remains about 50x slower than the 100 ms target and has a costly
60-second engine startup plus a separate-process bridge. I have not integrated
or selected it in production. Full per-image records and final JPEGs are under
`/tmp/wave_v16_migraphx_outer58_v2/`; the one-tile ONNX model is
`/tmp/wave-segformer-768.onnx`, the server is `/tmp/migraphx_server_768.py`,
and the full evaluator is `/tmp/wave_v16_migraphx_outer58_v1.py`.

### Static batch-2 MIGraphX tile inference (rejected; 2026-10-28)

I exported a fixed batch-2 FP32 SegFormer graph and compared two real 768 x 768
tiles against the existing PyTorch checkpoint on the same Radeon. The
upsampled class masks were identical for both tiles; maximum absolute logit
difference was 8.01e-5. MIGraphX took 188.1 ms per two-tile batch (94.0 ms per
tile), slightly slower per tile than the 91.6 ms batch-1 model-only median.
Including the same-process-boundary loopback input/output transfer, the pair
took 262.9 ms median. PyTorch batch-2 took 1,778.4 ms per pair on this run.
Peak PyTorch accelerator allocation/reservation was 3.05/3.85 GiB; the
MIGraphX server's memory is separate and was not measured. The fixed-batch
engine took 126.8 seconds to initialize and compile. The two-tile output
parity probe has no ground-truth waviness predictions, so MAE is unknown; no
full pipeline or quality claim follows from this microbenchmark. Batch-2 does
not improve tile throughput and is rejected. Export, server, and probe are
`/tmp/wave_export_segformer_768_batch2.py`,
`/tmp/wave-segformer-768-batch2.onnx`,
`/tmp/migraphx_server_768_batch2.py`, and
`/tmp/wave_migraphx_batch2_probe.py`.

### Send MIGraphX tile inputs directly from the processor (2026-10-28)

The MIGraphX bridge currently receives each tile after it has been copied from
the processor's CPU tensor to the PyTorch GPU and then back to CPU for socket
serialization. I measured 43 warmed requests of a representative 768 x 768
tile through a persistent MIGraphX server. Median socket round-trip was
106.26 ms, while the server-reported model median was 90.88 ms; the difference
was 15.38 ms per tile for host serialization, loopback transfer, and response
handling. The payload was 7.08 MB in and 22.12 MB of logits out. This identifies
avoidable work, but not all of that difference is attributable to tensor
copies.

I then ran the full 58-photo, 12-date cohort with processor tensors sent
directly from CPU to the MIGraphX server, retaining canonical FP32 tile inputs,
tile order, bilinear interpolation, GPU logit accumulation, crop selection,
regression, and output writes. Both the original bridge and direct-host-input
route succeeded on the same 51 photos and failed the same seven crops. On the
paired successes, canonical v16 MAE was 0.12006452; original MIGraphX MAE was
0.12006648 (+0.0016%); direct-host-input MAE was 0.12006773 (+0.0027%). This
remains within the 1% screen. Fifty of 51 final JPEG inputs matched canonical
v16 byte-for-byte and 48 of 51 final predictions matched exactly; the only
prediction change versus original MIGraphX was -0.00006396.

However, full raw-to-flushed-output latency did not improve: direct-host-input
median/p95 was 5,028.4/5,091.8 ms versus 4,969.3/5,030.3 ms for the original
bridge, a 59.7 ms paired median slowdown. Median client CPU time fell by
22.7 ms, but MIGraphX's per-tile median increased from 91.57 to 93.44 ms in
the separate runs, more than offsetting the host CPU saving. No request was
under 100 ms. The change is not adopted. This confirms that avoiding the
PyTorch device round trip is not, by itself, a reliable end-to-end speedup;
removing the per-tile logits return and accumulating them within the serving
process is a distinct, larger experiment that would need full pixel-output and
MAE validation before consideration. Runner, records, and model-input JPEGs:
`/tmp/wave_v16_migraphx_hostinput_v3.py` and
`/tmp/wave_v16_migraphx_hostinput_v3/`. The transport microbenchmark used the
MIGraphX server in `/tmp/migraphx_server_768.py`.

### MIGraphX FP16 runtime for the canonical v16 pipeline (2026-10-28)

The official ONNX Runtime MIGraphX provider exposes an FP16 quantization mode;
I enabled it with `ORT_MIGRAPHX_FP16_ENABLE=1` while keeping the exported
768-pixel graph, input images, tile geometry, accumulation, crop rules, and
v16 regressor unchanged. The persistent engine initialized in 70.2 seconds.
The full raw-to-flushed-output request was then measured on the same 58-photo,
12-capture-day cohort used for canonical v16 and MIGraphX FP32.

| Route | Paired MAE (51 successes) | Change vs v16 | Total latency, median / p95 |
|---|---:|---:|---:|
| Canonical PyTorch v16 | 0.12006452 | baseline | 10,057.6 / 10,092.3 ms |
| MIGraphX FP32 | 0.12006648 | +0.0016% | 4,969.3 / 5,030.3 ms |
| MIGraphX FP16 | 0.12059531 | +0.4421% | 2,623.0 / 2,697.3 ms |

FP16 had the same 51/58 successes and the same seven Step 2 crop failures as
v16. Its 12-capture-day cluster-bootstrap 95% interval for the paired MAE
change was [-0.00412, +0.00569], or [-3.61%, +5.07%] relative to v16. Absolute
error was lower on 19 photos, higher on 31, and tied on one. The FP16 point
estimate is within the 1% screen, and its 33.16 ms model-only median per tile
cut full-request median latency by 73.9% relative to canonical PyTorch. No
request was below 100 ms. Median client CPU time was 1,431.7 ms; server process
CPU and memory were not captured.

The confidence interval still permits a material quality regression, so this
does not establish the requested no-quality-hit condition and is not adopted.
The prior date cohort has also been examined repeatedly; confirmation needs
new capture dates that were not used for candidate development. Keep `.env`
and the production selection on v16. Per-image records, JPEGs, and runner are
under `/tmp/wave_v16_migraphx_fp16_outer58_v1/` and
`/tmp/wave_v16_migraphx_fp16_outer58_v1.py`.

### Selective FP16 with FP32 normalization and softmax (rejected)

To see whether FP16's small point-estimate MAE increase could be reduced while
retaining its speed, I converted the ONNX graph to mixed precision, keeping
LayerNormalization, Softmax, Resize, and the graph inputs/outputs in FP32 while
converting eligible operations and weights to FP16. MIGraphX loaded the mixed
graph without its FP16 quantization mode. The engine startup took 46.8 seconds.

| Route | Paired MAE (51 successes) | Change vs v16 | Total latency, median / p95 |
|---|---:|---:|---:|
| Canonical PyTorch v16 | 0.12006452 | baseline | 10,057.6 / 10,092.3 ms |
| MIGraphX FP32 | 0.12006648 | +0.0016% | 4,969.3 / 5,030.3 ms |
| MIGraphX FP16 | 0.12059531 | +0.4421% | 2,623.0 / 2,697.3 ms |
| Mixed: FP16 ops, FP32 normalization/softmax/resize | 0.12178872 | +1.436% | 6,543.3 / 6,643.3 ms |

Mixed precision had the same 51/58 successes and same seven Step 2 failures as
v16, but its MAE regressed by more than the 1% limit. Its 138.31 ms median
model time per tile was slower than both MIGraphX FP32 (91.57 ms) and all-FP16
(33.16 ms); full request latency was also worse than both. This blocklist
configuration is rejected and not adopted. The trial confirms that preserving
selected operators in FP32 can add costly graph casts or execution partitions
without restoring waviness quality. The conversion and evaluator are
`/tmp/wave_convert_mixed_segformer.py`,
`/tmp/wave_v16_migraphx_mixed_outer58_v1.py`, and
`/tmp/wave_v16_migraphx_mixed_outer58_v1/`. Conversion followed ONNX Runtime's
documented mixed-precision conversion with an operator blocklist:
[Float16 and mixed precision models](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html).

### MIGraphX exhaustive kernel tuning startup screen (stopped)

The official provider also offers exhaustive kernel tuning, which preserves
the FP32 graph while searching for faster operator kernels. I started the
same canonical 768-pixel graph with `ORT_MIGRAPHX_EXHAUSTIVE_TUNE=1`. After
15 minutes 12 seconds it had not created a serving session (`READY_SOCKET` was
never emitted), so I stopped the process. During the attempt it used roughly
24–29 CPU cores and peaked at about 6.6 GiB resident memory. No prediction was
run, so MAE and end-to-end latency are **not measured**. The ordinary FP32
MIGraphX initialization took about 60 seconds, making exhaustive tuning's
startup cost disproportionate without evidence of a large serving-time gain.
No model or runtime configuration changed. I am not pursuing this setting
further on the current host. The screen used the existing
`/tmp/migraphx_server_768.py` server.

### Independently seeded YOLO26 segmentation students and majority vote (rejected)

To test whether segmentation-mask variation was limiting the fast student, I
trained two additional YOLO26n semantic students (seeds 43 and 44) using the
same 127-image training and 22-image validation split as the clean seed-42
student. The split excluded all 58 outer-date photos from student training;
the 12 outer capture dates have nevertheless been consulted in previous
experiments and are exploratory, not an untouched final test. I ran each
student with the unchanged v16 regression head, then evaluated a strict
three-student majority-vote mask with the same head. Candidate MAE below is
paired with canonical v16 on the same successful images; latency is the full
warm raw-image-to-output request. Canonical v16 had 51/58 successful images.

| Experiment | Paired MAE (candidate / v16) | MAE change | Successes | Total latency, median / p95 |
|---|---:|---:|---:|---:|
| Canonical raw pipeline, v16 (first row) | 0.12006452 / 0.12006452 | baseline | 51/58 | 10,057.6 / 10,092.3 ms |
| YOLO26n clean student, seed 42 | 0.13226026 / 0.12006452 | +10.16% | 58/58 | 35.26 / 41.90 ms |
| YOLO26n clean student, seed 43 | 0.15185293 / 0.12006452 | +26.48% | 57/58 | 35.94 / 41.08 ms |
| YOLO26n clean student, seed 44 | 0.14554423 / 0.12026580 | +21.02% | 56/58 | 35.91 / 41.65 ms |
| Three-student majority vote | 0.15471237 / 0.12006452 | +28.86% | 58/58 | 42.25 / 47.63 ms |

The low-resolution student route is substantially faster but every variant is
over the 1% MAE regression screen, so none qualifies as a quality-preserving
improvement or replaces v16. Majority voting also worsened MAE and added about
7 ms median versus seed 42. These results do not meet the requested quality
gate. Keep all candidates experimental and keep `.env` on v16. Training,
evaluation, and per-image artifacts are under `/tmp/wave_yolo26_holdout_student_ensemble_v1/`
and `/tmp/wave_yolo26_clean_capture_holdout_eval_v1/`; the outer v16 reference
is `/tmp/wave_v16_canonical_outer58_raw_eval_v1/results.csv`.

### TorchInductor and ROCm layout/launcher screen for canonical SegFormer-B2

I checked AMD's current ROCm inference guidance and PyTorch's compiler
recommendations. AMD highlights TorchInductor weight freezing, the C++ launcher,
layout optimization/NHWC, and `max-autotune`; PyTorch cautions that compilation
has a one-time cost and must be measured after warmup. I screened the fixed
768x768 B2 tile forward on the Radeon 8060S with PyTorch 2.14.0+rocm10.1. The
warm eager reference was 274.65 ms/tile in the first run. With Inductor's
`freezing=True`, compilation took 25.94 s, logits differed by at most
5.15e-5, and the low-resolution argmax mask was identical. Warm compiled
latency was 280.96 ms/tile (2.3% slower); peak host RSS rose from 2.14 to 2.69
GiB and accelerator allocation/reservation peaked at 1.55/1.82 GiB.

A second default-mode run added the ROCm C++ wrapper, layout optimization, and
`PYTORCH_MIOPEN_SUGGEST_NHWC=1`. It compiled in 36.32 s and retained an
identical low-resolution argmax (maximum logit difference 7.44e-5), but was
309.15 ms/tile versus 263.65 ms eager in that paired process (17.3% slower).
The documented `max-autotune` configuration did not produce an inference
result within a five-minute screen. Eight compile workers used about eight CPU
cores and roughly 8–9 GiB combined resident memory; I stopped it at five
minutes. No model prediction MAE or full-request latency is measured for these
compile-only screens. These settings are not adopted.

I also captured a PyTorch profiler trace of one warmed eager 768 tile. The
profile reported 259.6 ms aggregate device time: MIOpen convolutions accounted
for 89.0 ms, `aten::copy_` kernels for 80.5 ms, matrix multiply for 15.6 ms,
depthwise convolution for 15.2 ms, and bilinear upsampling for 11.7 ms. The
largest copies moved stage tensors of shape `[1,768,192,192]` (36.0 ms) and
`[1,256,192,192]` (19.0 ms), consistent with internal layout/materialization
costs. This is operator time under profiler instrumentation, not a substitute
for the external tile latency. Previous full-request NCHW/channels-last B2
comparison was effectively neutral (10.598 s vs 10.578 s median), so the
profiler evidence alone does not justify changing memory format. The default
Inductor screen runner and profile are in `/tmp/wave_v16_inductor_tile_screen.py`,
`/tmp/wave_v16_inductor_rocm_flags_tile_screen.py`,
`/tmp/wave_v16_inductor_maxautotune_tile_screen.py`,
`/tmp/wave_v16_tile_operator_profile.py`, and
`/tmp/wave_v16_tile_copy_profile.py`.

The documented guidance consulted was AMD's [ROCm workload optimization and
TorchInductor tuning](https://rocm.docs.amd.com/en/docs-7.2.4/how-to/rocm-for-ai/inference-optimization/workload.html),
PyTorch's [`torch.compile` reference](https://docs.pytorch.org/docs/stable/generated/torch.compile),
and [where to apply `torch.compile`](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.where_to_apply_compile.html).

### MIGraphX FP32 probe for SegFormer-B0/448 (model-only; no MAE)

I used the existing FP32 B0/448 ONNX graph with ONNX Runtime 1.29 and
MIGraphX 2.18 on the same Radeon. The server started in 19.34 s. For one real
448x448 tile, eager PyTorch inference had a 12.25 ms median; MIGraphX server
execution was 9.23 ms median, while the client-observed socket round trip was
12.84 ms. The maximum logit difference was 4.39e-5, the 448-resolution
argmax was exact, the projected source class map had zero changed pixels, and
water-mask IoU was 1.0. This is a one-image model-only parity/speed probe; no
waviness prediction or MAE was measured, and server memory was not captured.
Because client round-trip time was already slightly slower than local PyTorch,
I did not run a full cohort or integrate the extra process boundary. Runner and
server are `/tmp/wave_compare_migraphx_b0_448.py` and
`/tmp/migraphx_server_b0_448.py`; the ONNX graph is
`/tmp/wave-segformer-b0-448.onnx`.

### OpenVINO FP32 CPU inference for the B0/448 route (no end-to-end gain)

The Ryzen AI Max+ CPU supports FP32 inference in OpenVINO. I installed
OpenVINO 2026.4.1 in an isolated temporary environment, imported the existing
B0/448 ONNX graph, and measured a fixed input. With latency mode and 16 CPU
threads, warm model-only latency was 11.05 ms median; 32 threads measured
10.78 ms. The same model took 12.25 ms on local PyTorch/Radeon. A PyTorch CPU
fallback was much slower (123.1 ms at one thread, 286.7 ms at two, and 641.9 ms
at four); higher-thread CPU PyTorch became slower still. This reinforces the
importance of benchmarking runtime/thread settings on the actual CPU instead
of increasing threads by default.

I then compared the complete B0 raw-to-output path on 93 labeled photos. Both
arms used the same exact native HEIF decoder, B0 GPU-standardization contract,
GPU erosion, B0 regression head, persisted model-input JPEG, and flushed plus
fsynced prediction record. The original OpenVINO route returned logits to
PyTorch for bilinear resize and argmax. It produced exact JPEGs and predictions
on all 93, so both arms had MAE 0.12995, but full request latency was slower:
253.21 ms median / 271.51 ms p95 versus 218.78 / 235.87 ms for PyTorch GPU.
The CPU interpolation and whole-request path erased the model-only advantage.
Peak process RSS was 2.83 GiB; accelerator allocated/reserved memory was
0.428/1.375 GiB. No request was under 100 ms.

| Experiment | MAE (candidate / v16) | Total raw-to-output latency, median / p95 |
|---|---:|---:|
| **Current pipeline: raw → v16** (outer-date reference, 51 successes) | **0.12006** | **10,057.6 / 10,092.3 ms** |
| B0 PyTorch GPU, exact native decode (MAE paired on 32 archived test photos; latency on 93) | 0.12826 / 0.13284 (−3.45%) | 248.71 / 262.94 ms |
| B0 OpenVINO CPU, logits returned to PyTorch (same 32 paired photos; latency on 93) | 0.12826 / 0.13284 (−3.45%) | 253.21 / 271.51 ms |
| B0 OpenVINO CPU, fused resize plus argmax (same 32 paired photos; latency on 93) | 0.12729 / 0.13284 (−4.18%) | 250.05 / 261.81 ms |

MAE and latency sample counts differ in these rows because archived canonical
v16 predictions are available for only 32 of the 46 B0 test photos; the latency
run covered all 93 validation and test photos. These are exploratory point
estimates, and most test capture dates have been examined in previous work.

To avoid materializing the large 150-channel resized logit tensor in Python, I
fused bilinear resize and argmax in the OpenVINO graph, returning a 448x448
class map. On the one-image parity probe and 77/93 end-to-end photos, the
resulting Step 2 JPEGs/predictions were exact; 16/93 changed. Across the full
93-image split its MAE was 0.12662 versus 0.12995 for PyTorch B0, and on the
46-image test split it was 0.12696 versus 0.12763. On the 32 test photos with
archived canonical v16 results, the fused route had MAE 0.12729 versus v16
0.13284 (-4.18%); PyTorch B0 had 0.12826 versus 0.13284 (-3.45%). These are
point estimates on the previously examined split, not independent date-heldout
proof. The changed outputs mean this fused implementation is not output
identical, and it did not speed up the complete request: 250.05 ms median /
261.81 ms p95 versus 248.71 / 262.94 ms for PyTorch GPU in the same paired
run. It is not adopted.

This experiment confirms that CPU runtime choice alone is not enough: CPU
inference can be faster per forward while full-resolution mask handling and
standardization still dominate the request. It also shows that fusing resize
and argmax needs numerical parity checks at the final-mask and prediction
boundaries; a single-photo exact mask did not predict full-cohort parity. The
runners and outputs are `/tmp/wave_b0_cpu_inference_sweep.py`,
`/tmp/wave_b0_openvino_cpu_screen.py`,
`/tmp/wave_b0_openvino_cpu_e2e_v1.py`,
`/tmp/wave_b0_openvino_fused_mask_screen.py`, and
`/tmp/wave_b0_openvino_fusedmask_e2e_v1.py`. The OpenVINO CPU plugin and
benchmark guidance is documented at [OpenVINO CPU device](https://docs.openvino.ai/nightly/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html)
and [the OpenVINO benchmark tool](https://docs.openvino.ai/nightly/get-started/learn-openvino/openvino-samples/benchmark-tool.html).

### ROCm TunableOp GEMM tuning screen for canonical v16 (rejected)

PyTorch's TunableOp can compare registered rocBLAS and hipBLASLt GEMM
implementations for the shapes used by a workload. Since the v16 tile profile
reported substantial GEMM time, I enabled TunableOp for the canonical 768-pixel
SegFormer tile, capped tuning at 10 ms per operation and three iterations, and
measured warm tile inference before and after. It tuned 27 entries. Baseline
median tile latency was 247.5 ms; steady TunableOp latency was 254.9 ms (3.0%
slower). The tuning pass itself took 26.9 seconds, including a 26.4-second
first tuned invocation. The maximum logit delta was 4.01e-5. **Waviness MAE is
unknown** because this was a one-tile model screen, not a full prediction
cohort. The result does not justify a full end-to-end run or enabling TunableOp.

This is consistent with the profiling rule: a library offering more algorithm
choices is not automatically faster for the model's actual shapes. The earlier
`cudnn.benchmark` trial also did not improve full-request latency. Artifacts:
`/tmp/wave_v16_tunableop_tile_screen.py`,
`/tmp/wave_v16_tunableop_tile_screen.json`,
`/tmp/wave_v16_tunableop_tile_screen.log`, and
`/tmp/wave_v16_tunableop_results.csv`. API reference:
[PyTorch TunableOp](https://docs.pytorch.org/docs/2.14/cuda.tunable.html).

### MIOpen immediate mode and decoder-copy trace (no speed improvement)

To follow the large-copy profile with a concrete backend screen, I compared
PyTorch's `torch.backends.miopen.immediate` disabled and enabled for the same
warmed canonical v16 SegFormer tile. Median model latency was 264.0 ms with the
default mode and 265.4 ms with immediate mode; the candidate's final two samples
also rose to 276.3 and 280.8 ms. The 7 logits were bit-identical between modes
(maximum difference 0; identical argmax). **Waviness MAE is unknown** because
this is a single-tile screen. This mode did not improve the measured tile and
is not enabled.

A separate TorchDispatch trace tied the largest materialization shapes to the
model path: 35.4 ms aggregate profiled device time at `[1,768,192,192]`, 19.2
ms across three copies at `[1,256,192,192]`, and 18.2 ms across four copies at
`[1,512,96,96]`. The corresponding execution path includes SegFormer depthwise
convolution and decoder bilinear resizing/concatenation. These profiler totals
are kernel time under instrumentation, not wall time, and dispatch tracing does
not prove each copy can be removed without changing the operators or numeric
results. A global channels-last experiment was already neutral end to end; this
trace narrows the next exact-kernel investigation to the repeated layout
transitions in the model rather than host-side image buffers.

Artifacts: `/tmp/wave_v16_miopen_immediate_tile_screen.py`,
`/tmp/wave_v16_miopen_immediate_tile_screen.json`,
`/tmp/wave_v16_miopen_immediate_tile_screen.log`, and
`/tmp/wave_v16_tile_copy_dispatch_shapes.py` plus its log. PyTorch documents
immediate mode in its [MIOpen backend settings](https://docs.pytorch.org/docs/2.14/backends.html).

### Targeted SegFormer depthwise-convolution layout screen (no gain)

The copy trace highlighted the high-resolution depthwise-convolution path, so I
captured the actual stage-0 activation (`[1,36864,256]`, 192x192 spatial) and
compared the stock convolution with two layout candidates: a channels-last
weight tensor, and an explicit contiguous NCHW input before the convolution.
Both candidates returned bit-identical outputs. Across eight warm calls, stock
median was 7.966 ms, channels-last weights 7.963 ms, and explicit NCHW input
7.964 ms. These differences are negligible and do not support a full-pipeline
run or model modification. **Waviness MAE is unknown** because this isolated
layer probe did not produce a prediction cohort.

This screen suggests the large profiled copy totals are spread across several
operators/layout boundaries rather than removable by changing this depthwise
convolution's tensor format. Full trace and probe artifacts are
`/tmp/wave_v16_tile_copy_dispatch_shapes.py`,
`/tmp/wave_v16_tile_copy_dispatch_shapes.log`,
`/tmp/wave_v16_dwconv_weight_layout_screen.py`, and
`/tmp/wave_v16_dwconv_weight_layout_screen.json`.

### Split SegFormer decoder fusion to avoid the wide concatenation

The profiler showed large high-resolution layout copies in the decoder. I
replaced the decoder's `cat` of projected features followed by one 1x1
convolution with the algebraically equivalent sum of 1x1 convolutions over the
same feature groups and the same slices of the original trained weight tensor.
Weights, FP32 precision, tile grid, and interpolation settings are unchanged.
This avoids materializing the wide concatenated feature tensor. The measured rewrite is enabled by `_enable_split_segformer_decode_fusion()`
for ROCm inference in `step_1_a_water_segmentation.py`; CPU and other GPU
backends retain their framework implementation until they are measured.

An interleaved raw-to-output A/B on 12 images across five capture dates produced
bit-identical predictions and identical MAE (0.1253717 on this 12-image timing
cohort). Median request latency fell from 10,180.9 to 8,957.6 ms; all 12 paired
requests were faster, with median paired saving of 1,218.2 ms. The p95 values
were 10,213.0 and 9,000.7 ms. The run included source decode, both segmentation
stages, JPEG round trips, regression, and flushed prediction-record writes.

I then ran the integrated loader on the complete 58-photo date-grouped outer
cohort. Both versions had the same 51 successes and same seven Step 2 failures.
On those 51 shared successes, canonical MAE was 0.1200645 and the integrated
candidate MAE was 0.1200658, a +0.0000013 absolute change (+0.0010% relative,
inside the 1% gate); RMSE changed from 0.1564493 to 0.1564496. Fifty of 51
Step 2 JPEGs and regression predictions were byte/value identical. The one
changed prediction moved by 0.000064; its effect on aggregate MAE was
negligible. Candidate median/p95 total latency was 9,342.7/9,372.5 ms versus
10,057.6/10,092.3 ms for the archived baseline run. This separate-process
measurement is noisier than the interleaved A/B, which establishes the cleaner
speed estimate. Peak process RSS was 2.89 GiB versus 2.91 GiB, and peak
accelerator allocated/reserved memory was 8.21/8.60 GiB versus 8.31/8.62 GiB.
Neither run approaches the 100-ms goal; this is a measured 12% paired latency
reduction while preserving the MAE gate, not a complete solution.

The tile-only screen measured 252.2 ms median for the stock decoder and 228.1
ms for split fusion (about 9.6% lower); argmax matched on all eight paired
tiles, with maximum logit delta 3.43e-5. The full-cohort results above are the
quality evidence. Artifacts:
`/tmp/wave_v16_decoder_split_fusion_screen.py`,
`/tmp/wave_v16_decoder_split_fusion_screen.json`,
`/tmp/wave_v16_decoder_split_fusion_prod_outer58_v1.py`,
`/tmp/wave_v16_decoder_split_fusion_prod_outer58_v1/`, and
`/tmp/wave_v16_decoder_split_fusion_interleaved12.py` with its `_v1/` outputs.

### Stream decoder projection into fusion to cut peak activation memory (rejected for latency)

After the split-fusion change, I tested a streaming variant that projects,
upsamples, and fuses one decoder feature at a time, using in-place accumulation
so the other full-resolution projected maps do not remain live together. On six
warm 768-pixel tile calls, the streaming and current split-fusion outputs were
bit-identical (maximum logit difference 0; identical argmax). Peak allocated
GPU memory in the isolated model screen fell from 1.069 to 0.545 GiB, but median
tile latency increased from 232.7 to 240.6 ms (+3.4%). **Waviness MAE is
unknown** because this was a one-tile screen. The memory reduction is real, but
the latency regression means this alternative is not integrated. Artifacts:
`/tmp/wave_v16_decoder_streamed_fusion_screen.py` and
`/tmp/wave_v16_decoder_streamed_fusion_screen.json`.

### `torch.compile` on the split decoder only (rejected)

I screened Inductor on just the split-fusion SegFormer decoder, keeping the
encoder eager. Compilation took 9.10 seconds on the first invocation. Warm tile
latency increased from 232.7 ms eager to 291.6 ms compiled (+25.3%); argmax was
identical on the one tile, with maximum logit delta 1.53e-5. **Waviness MAE is
unknown** because this was a single-tile screen. The compiled decoder is not
used. Runner and output: `/tmp/wave_v16_split_decoder_compile_screen.py` and
`/tmp/wave_v16_split_decoder_compile_screen.json`.

### Fold decoder BatchNorm into split-fusion weights (exact, no end-to-end gain)

I folded the decoder's inference BatchNorm into the existing 1x1 fusion weights
and branch bias, removing the standalone BatchNorm operation. A tile screen
showed identical argmax and a 2.8-ms lower median, so I measured the complete
58-photo outer cohort. Compared directly with the current split-fusion
implementation, the folded route had the same 51/58 successes, identical seven
failures, byte-identical Step 2 JPEGs and predictions on all 51 successes, and
identical waviness MAE/RMSE (0.1200658/0.1564496). However, separate full-cohort
medians were 9,382.4 ms folded versus 9,342.7 ms current; p95 was 9,416.5 versus
9,372.5 ms. Peak allocated accelerator memory was 8.13 GiB in both runs. The
tile-only gain did not translate to an end-to-end speedup, so the BatchNorm
fold is not integrated. Artifacts:
`/tmp/wave_v16_decoder_fused_bn_screen.py`,
`/tmp/wave_v16_decoder_fused_bn_screen.json`,
`/tmp/wave_v16_decoder_split_bn_outer58_v1.py`, and
`/tmp/wave_v16_decoder_split_bn_outer58_v1/`.

### Current pipeline checkpoint (2026-10-29)

The `.env` checkpoint selection has now been changed to v18, as previously
requested. The speed/quality comparisons below were measured before that
selection change and compare the split-fusion implementation with canonical
v16; they do not establish v18 end-to-end latency. The v18 checkpoint reports
validation MAE 0.1357 in its training artifact, which is not directly
comparable to the v16 outer-cohort MAE. A matched raw-to-output v18 baseline
measurement is still needed. The ROCm-only split decoder-fusion rewrite is
enabled in the segmentation path. On the 58-photo, 12-capture-day outer cohort, it completed
51 rows and had the same seven Step 2 failures as stock v16. On the 51 paired
successes, waviness MAE was 0.1200658 versus 0.1200645 for stock v16
(+0.0010%, well within the 1% regression limit). The interleaved 12-photo,
five-date raw-to-output A/B produced identical predictions and MAE (0.1253717
in both arms); median latency fell from 10,180.9 to 8,957.6 ms and p95 from
10,213.0 to 9,000.7 ms, with all 12 paired requests faster. This is the
strongest recent quality-qualified optimization, but it remains roughly 90x
above the 100-ms target. The full-cohort candidate median/p95 was 9,342.7 /
9,372.5 ms; the interleaved comparison is the cleaner speed estimate.

The NPU check confirmed hardware visibility but did not produce a model result:
`xrt-smi examine` reports Ubuntu 26.04.1 and XRT 2.21.75, while AMD's current
[Ryzen AI Linux 1.8 installation instructions](https://ryzenai.docs.amd.com/en/latest/linux.html)
specify Ubuntu 24.04 XRT 2.25.37 driver packages, a matching XDNA plugin, and
a separate Ryzen AI runtime package. The guide documents CNN INT8/BF16 and
encoder-based BF16 flows, but does not list SegFormer as a Linux example.
Therefore NPU inference latency and waviness MAE remain unknown. No host driver
or runtime was changed. `xrt-smi`'s built-in 51 TOPS / 72-microsecond device
diagnostic is not a model inference result.

| Experiment | Waviness MAE (candidate vs stock v16) | Total raw-to-output latency, median / p95 | Result |
|---|---:|---:|---|
| **Current `.env` checkpoint: raw → v18 fast-v1** | **0.11660** on the matched 46-photo test run; training validation MAE was 0.13567 | **310.2 / 333.4 ms** stock fusion, interleaved full-request benchmark | Selected checkpoint; 0/46 requests under 100 ms. |
| v16 with split decoder fusion (experimental comparison) | 0.1200658 vs stock v16 0.1200645 (+0.0010%), 51 shared successes; same 7 failures | 8,957.6 / 9,000.7 ms interleaved 12-photo A/B; 9,342.7 / 9,372.5 ms full-cohort candidate run | Quality gate passes against v16; about 12% lower paired median; still far above 100 ms. |
| Stock raw → v16 reference | 0.1200645, 51/58 successes | 10,180.9 / 10,213.0 ms interleaved A/B; archived separate-run median/p95 10,057.6 / 10,092.3 ms | Reference. |
| MIGraphX FP32 full canonical v16 | 0.12006648 vs 0.12006452 (+0.0016%), same 7 failures | 4,969.3 / 5,030.3 ms | Passes point-estimate gate; separate service and ~60 s startup; no requests under 100 ms. |
| MIGraphX FP16 full canonical v16 | 0.12059531 vs 0.12006452 (+0.4421%), same 7 failures | 2,623.0 / 2,697.3 ms | Point estimate passes; date-cluster interval allows material regression; not adopted. |

The table keeps MAE attached to matched cohorts and reports complete request
latency, not tile-only engine time. Candidates with more than 1% MAE
regression are excluded from this quality-qualified summary; exploratory
results without a matched waviness MAE are not treated as quality-preserving.

### Correct v18 raw-to-output baseline and exact-resize checks (2026-10-29)

I measured the selected v18 checkpoint with the fast-v1 contract from its
2026-10-28 split snapshot. The production entry point applies the profile
before constructing the segmenter; the evaluator likewise configured the
448-pixel FP16 Step 1 processor and 384-pixel Step 2 processor before loading
the model. On all 46 held-out test photos, stock SegFormer fusion completed
46/46, with waviness MAE 0.116597, median raw-to-output latency 310.2 ms, and
nearest-rank p95 333.4 ms. No request was under 100 ms. Timings include raw
HEIC/JPEG decode, both segmentation passes, both JPEG byte round trips,
regression, and temporary preview/record writes. Model-load and first-call
warm-up are excluded. This is the first correctly matched raw-to-output
baseline for the currently selected checkpoint/profile.

The initial v18 profile comparison loaded the segmenter before applying the
profile, so Step 1 silently retained its default 768-pixel processor. Its
latency and MAE records are invalid and must not be used. Reordering profile
configuration before model load corrected the setup and is covered by the
corrected benchmark runner. This reinforced that each benchmark must assert
the processor's actual input size rather than infer it from the profile
configuration alone.

I then compared the integrated split decoder fusion with stock fusion on the
same 46 test photos and identical 448/384 fast-v1 inputs. Stock fusion MAE was
0.116597; split fusion MAE was 0.118038 (+1.236%, beyond the 1% limit). Only
31/46 final Step 2 JPEGs and predictions were identical. The split route saved
1.74 ms paired median latency (310.2 to 308.0 ms) and did not lower p95
(333.4 to 333.9 ms). The rewrite had passed on v16's 768-pixel FP32 profile,
but it is rejected for v18 fast-v1. I gated it in
`step_1_a_water_segmentation.py` to the overlapping-tile FP32 profile where
the quality-qualified evidence exists; fast-v1 now uses stock decoder fusion.

I tested Pillow nearest-neighbor expansion of the 448-pixel class map against
OpenCV `INTER_NEAREST_EXACT` and `torch.from_numpy` on the resulting owned
buffer. All 46 Step 2 JPEGs and predictions matched byte/value exactly and MAE
stayed 0.116597. In the interleaved run, however, OpenCV median/p95 latency was
335.0/347.0 ms versus 321.6/338.2 ms for Pillow (paired median 12.2 ms slower).
The identical output proves the buffer-sharing replacement is safe, but it is
not faster on this workload and is not integrated. The existing OpenCV resize
result on the B0 candidate did not transfer to the v18 fast-v1 profile.

Finally, I recomputed the existing thumbnail YOLO26 + v18 head result against
the actual v18 fast-v1 baseline, on the same 46 filenames and labels. Its MAE
was 0.118214 versus 0.116597 for current v18, a +1.386% regression. The older
comparison used v16 and suggested improvement; that is not the correct gate
after v18 became selected. This route remains under 100 ms but fails the
current 1% quality limit and is not adopted.

The corrected full-run A/B data, resize comparison, and component profile are
under `/tmp/wave_v18_fusion_ab_v1/`, `/tmp/wave_v18_cv2_nearest_ab_v1/`, and
`/tmp/wave_v18_substage_profile_v1/`; runners are in `/tmp`. The eight-image
component profile found median decode 51.0 ms, Step 1 segmentation 103.9 ms,
Step 1 standardization 72.0 ms, Step 2 32.4 ms, and regression forward about
5 ms. Median process CPU time was 421.5 ms per request, peak RSS 2.60 GiB, and
peak PyTorch allocated/reserved memory 0.59/0.62 GiB. Component measurements
are diagnostic and must not be summed as an end-to-end latency estimate.

### Embedded-thumbnail B2 route with adapted v18 head (2026-10-29)

I tested a lower-cost route that starts from the embedded 320-pixel HEIC
thumbnail (or a 320-pixel JPEG draft), runs the full Step 1 SegFormer-B2 sea
segmentation, then runs the full Step 2 SegFormer-B2 preprocessing at 224
pixels, and finally predicts with the v18 EfficientNet-B0. This preserves both
segmentation passes while changing their input scale and the Step 2 output
distribution. It is therefore a new preprocessing contract, not a drop-in
backend optimization.

With the original v18 head this route had test MAE 0.118178 versus 0.116597
for current raw-to-output v18 (+1.36%), so it failed the 1% quality gate. I
then adapted only the small regression head using candidate-preprocessed
training data: 148 fit images and a 61-image capture-date holdout from 209
successful training preprocesses. The held-out dates selected the head
settings; validation and test labels were not used for selection. On 46 shared
validation photos, candidate MAE was 0.131680 versus 0.138591 for the current
pipeline on the same photos (−4.99%). On the 46-photo test snapshot, candidate
MAE was 0.115703 versus 0.116597 (−0.77%); paired recomputation confirms the
same comparison. This passes the point-estimate 1% gate, but the test snapshot
is image-random and capture dates overlap training dates, so it is not an
independent future-date confirmation.

The adapted route was rerun end to end over the full 93-photo snapshot,
including prediction and record writes. Validation median/p95/max latency was
68.47/70.98/71.18 ms and test was 73.67/76.08/76.24 ms. All 46 successful
requests in each split were under 100 ms. The route still failed Step 1 on
IMG_7565.jpg (46/47 validation coverage); all 46 test photos succeeded. On
the 46 shared successes, validation MAE was 0.131680 versus 0.138591 current
(−4.99%); test MAE was 0.115703 versus 0.116597 current (−0.77%). A
20,000-resample capture-date cluster bootstrap for candidate-minus-current
MAE gave [−0.02421, +0.00879] on validation and [−0.01658, +0.01306] on test.
Both intervals allow a meaningful regression, and test dates overlap training
dates, so this is a promising point estimate rather than confirmation of
preserved quality. Peak RSS was 2.45 GiB; peak GPU allocated/reserved memory
was 0.588/0.615 GiB. The `.env` selection remains the current v18 pipeline.

| Experiment | Waviness MAE on matched test photos | Total raw-to-output latency, median / p95 | Decision |
|---|---:|---:|---|
| Current v18 fast-v1, stock fusion | 0.116597 (46 photos) | 310.2 / 333.4 ms | Current reference; 0/46 under 100 ms. |
| Embedded thumbnail + full B2 Step 1 + B2 Step 2 at 224 + adapted v18 head | 0.115703 (46 shared test photos; −0.77% vs current) | 73.67 / 76.08 ms test; 68.47 / 70.98 ms validation (median / p95, adapted artifact) | Best sub-100-ms point estimate; 46/46 test and 46/47 validation succeeded. Bootstrap intervals include regression; date overlap prevents an independent quality claim. |

Training artifact and per-split outputs are under
`/tmp/wave_v18_thumbnail_b2_head_224_v1/`; the route timing and preview
records are under `/tmp/wave_v18_embedded_b2_thumbnail_step2_224_v1/`. The
paired quality recomputation is `/tmp/wave_v18_thumbnail_b2_224_paired_quality.py`.

### Post-hoc date-disjoint check of the adapted thumbnail B2 route

I separately reran the four 2026-08-24 test photos, a date absent from the
current checkpoint's train split and the adapted head's fit/selection dates.
The candidate and current route both succeeded on all four photos. Each photo
was timed five times in randomized paired order; each request included raw
decode, preprocessing, prediction, and output writes.

| Route | MAE on the four photos | Total raw-to-output latency, median / p95 | Under 100 ms |
|---|---:|---:|---:|
| Current exact v18 route | 0.148265 | 285.82 / 312.49 ms | 0/20 requests |
| 320-pixel thumbnail + B2 at 224 + adapted v18 head | 0.146572 (−1.14%) | 69.83 / 84.26 ms | 20/20 requests |

Candidate MAE was lower by 0.00169, with lower absolute error on two photos
and higher absolute error on two; JPEGs and predictions differed on all four.
This date is disjoint from training, but the subset was identified from prior
test-set inspection, and it is only one date with four photos. A date-cluster
bootstrap is not informative with a single cluster. Treat this as exploratory
evidence, not quality-preservation confirmation or grounds for adoption. The
complete repeated timings and outputs are in
`/tmp/wave_v18_thumbnail_b2_date_disjoint_eval_v1/`; the runner is
`/tmp/wave_v18_thumbnail_b2_date_disjoint_eval.py`.

### Nested capture-date CV for the adapted thumbnail B2 input contract

Because the adapted thumbnail head's available validation/test dates overlap
training dates, I ran a nested, five-fold capture-date CV on the 217-photo
training split. The 209 photos for which thumbnail preprocessing succeeded
were shared between both input domains; their 50 capture dates were assigned
to five outer folds. Each fold fit a canonical-input regression head using
only its outer-training dates, then adapted a second head from that
fold-specific canonical head using the corresponding thumbnail inputs. Inner
capture-date groups selected epochs and the teacher/regularization settings.
No outer-fold labels were used for fitting or selection. The EfficientNet-B0
backbone was frozen and identical in both branches.

| Input contract and fold-fitted head | MAE on 209 out-of-fold photos | Relative MAE delta | Total request latency context |
|---|---:|---:|---:|
| Current full-resolution Step 2 inputs, fold-fitted head | 0.155035 | Reference | Not timed in this CV; current deployed v18 route is about 287–289 ms median on the 93-photo labeled snapshot. |
| Embedded thumbnail + B2 Step 2 at 224, fold-adapted head | 0.157662 | +1.69% | The same thumbnail pipeline and model architecture measured 68.47 ms validation / 73.67 ms test median in the separate full-route rerun. |

The paired date-cluster bootstrap 95% interval for candidate-minus-canonical
MAE was [−0.00893, +0.01400] across 50 capture dates, so it includes a
meaningful regression. The candidate preprocessing also failed on 8/217
training photos (3.7%), which are not represented in this shared-success MAE.
The CV heads were trained from fold-specific data, so their absolute MAE is not
the deployed checkpoint's score; this is a controlled comparison of the two
input contracts under date-held-out head fitting. It does not clear the
thumbnail route's quality gate. Runner, folds, and per-photo OOF predictions
are under `/tmp/wave_v18_thumbnail_b2_matched_nested_cv_v1/` and
`/tmp/wave_v18_thumbnail_b2_matched_nested_cv.py`.

### Pillow staged Lanczos reduction on the current v18 path (2026-10-29)

I profiled the current 448/384 `fast-v1` predictor and tested Pillow's
`reducing_gap` option only for large Lanczos downscales (at least a 3x reduction
on both axes). Raw image decode, segmentation model and precision, processor
resolutions, crop/mask logic, JPEG quality, regressor checkpoint, and output
writes were held fixed. This is a CPU resampling implementation change; it
does not skip a pipeline stage or use lower-resolution source thumbnails.
Pillow describes this option as a staged integer reduction followed by the
requested resampler; smaller values are faster and values of 3 or above are
usually indistinguishable from direct resampling ([official Pillow resize
documentation](https://pillow.readthedocs.io/en/latest/reference/Image.html)).

The `reducing_gap=2` candidate was run in alternating order against the stock
resize on the full 47-photo validation snapshot and all 46 test photos. MAE
was 0.135168 versus 0.135669 on validation (−0.37%), and 0.117383 versus
0.116597 on test (+0.67%). Both point-estimate regressions are below the
project's 1% screening limit, and all photos completed in both arms. On the
same run, including preview and prediction-record writes, median/p95 latency
fell from 331.6/345.2 to 289.8/302.9 ms on validation and from 330.7/342.6
to 292.0/299.9 ms on test. The paired median saving was 36.5 ms on validation
and 37.0 ms on test. No request was under 100 ms. Peak RSS
was about 2.91 GiB and PyTorch allocated/reserved accelerator memory was
0.588/0.615 GiB in both arms.

The confidence intervals remain too wide to claim no possible material MAE
regression: capture-date-cluster bootstrap 95% intervals for candidate-minus-
stock absolute error were [-0.00764, +0.00850] on validation (31 dates) and
[-0.00419, +0.00643] on test (27 dates). The test split shares capture dates
with model training and is not a clean future-date assessment. Thus this is a
promising point-estimate quality/speed tradeoff, not yet a proven
quality-preserving production change. Exact model-input JPEGs and predictions
changed, as expected from a different resampling algorithm; no code or active
profile was changed.

The more aggressive `reducing_gap=3` was rejected: its test MAE improved by
4.19%, but validation MAE regressed by 2.67%. This split disagreement is why
the candidate cannot be selected from test performance alone. Artifacts and
runners and complete request records are under `/tmp/wave_v18_pillow_reducing_gap_ab_v1/`,
`/tmp/wave_v18_pillow_reducing_gap_2_full_request_test/`,
`/tmp/wave_v18_pillow_reducing_gap_v3_validation_v1/`, and
`/tmp/wave_v18_pillow_reducing_gap_2_full_request_validation/`.

### Parallel libheif tile decode on the current v18 pipeline (exact pixels)

I tested the existing experimental C++ libheif tile decoder against
`pillow-heif` on the complete validation/test source set. It reconstructs
oriented RGB pixels from HEIF grid tiles using 16 independent libheif contexts
and writes into one contiguous output buffer. The decoded RGB array matched
`load_rgb_image` byte-for-byte on all 92 HEIC/HEIF sources. Decode-only median
latency was 51.9 ms versus 73.2 ms for Pillow; p95 was 63.3 versus 112.6 ms.

I then substituted that decoder in the complete v18 448/384 path, leaving both
SegFormer calls, mask/crop operations, JPEG round trips, checkpoint, and
prediction-record writes unchanged. All 93 validation/test requests completed.
Every final Step 2 JPEG and waviness prediction matched exactly (93/93), so MAE
was unchanged: 0.135669 validation and 0.116597 test. Median/p95 total latency
fell from 320.2/339.9 to 303.3/315.1 ms on validation and 326.3/339.1 to
305.4/319.1 ms on test; the paired median saving was 17.8 ms. No request met
100 ms. Peak RSS was 2.98 GiB and GPU allocated/reserved memory was
0.588/0.615 GiB. Median process CPU time rose from about 440 to 494–497 ms,
reflecting the parallel decoder's additional CPU work. Concurrent-request
resource contention was not tested.

This is an exact-output speed improvement, not a quality tradeoff. I have not
integrated it: the prototype links against Pillow-Heif's bundled libheif
binary through a temporary shared library, and would need a supported build
and packaging path before it is suitable for the pipeline. Even if integrated,
it removes only about 18 ms and does not approach the 100-ms target. Full-run
records are under `/tmp/wave_v18_heif_tiled_decode_ab_v1/`; the temporary
decoder implementation is `/tmp/wave_heif_tiled_decode_bgr_ctx.cpp`.

### Combine exact HEIF decode with staged Lanczos (full paired run)

I combined the pixel-exact HEIF decoder with `reducing_gap=2` and measured the
complete route against stock v18 on the same 47 validation and 46 test photos.
All 93 requests completed in both arms. The combined route's final Step 2 JPEGs
and predictions matched the standalone `reducing_gap=2` candidate on all 93
photos, confirming that the decoder substitution adds no further input or
prediction change. Relative to stock v18, combined MAE was 0.135168 versus
0.135669 on validation (−0.37%) and 0.117383 versus 0.116597 on test (+0.67%),
the same quality results as the resize candidate.

Complete raw-to-output latency, including preview and prediction-record writes,
was 263.6 ms median / 276.6 ms p95 on validation and 268.8 / 276.3 ms on test.
The paired median saving was 58.0 ms across 93 photos, about 18–19% versus
stock. Median process CPU time increased by about 10–13 ms, and no request was
under 100 ms. This is the best current-v18 full-route speed result that stays
within the 1% point-estimate MAE screen on both existing splits, but its
capture-date bootstrap intervals remain too wide to establish a strict
no-regression claim. It remains experimental; the HEIF component also still
needs a supported build/package approach. Results are in
`/tmp/wave_v18_heif_tiled_gap2_ab_v1/paired.json`.

### Compile the complete fast-v1 SegFormer for fixed shapes

The earlier compiler screen targeted a single 768-pixel decoder path and was
slower. I tested a distinct setup: compile the complete SegFormer-B2 with
`torch.compile(dynamic=False)` for the current fast-v1 fixed 448-pixel Step 1
and 384-pixel Step 2 inputs. Compilation and warm-up were outside the timed
request; both shapes were warmed before the paired 93-photo run.

On the same validation/test splits, eager versus compiled results were:

| Split | Eager MAE | Compiled MAE | Eager latency median / p95 | Compiled latency median / p95 |
|---|---:|---:|---:|---:|
| Validation, 47 photos | 0.135669 | 0.133432 (−1.65%) | 328.6 / 342.4 ms | 306.5 / 322.2 ms |
| Test, 46 photos | 0.116597 | 0.116534 (−0.05%) | 332.8 / 349.5 ms | 311.1 / 321.2 ms |

All 93 requests succeeded. The paired median latency saving was 24.0 ms; 38/93
Step 2 JPEGs and predictions were byte/value identical. The capture-date
cluster bootstrap 95% intervals for candidate-minus-eager absolute error were
[-0.00505, -0.00010] on validation (31 dates) and [-0.00184, +0.00155] on
test (27 dates). Validation favored compilation; test includes zero and its
upper endpoint is a 1.33% relative MAE increase, so strict no-regression is
not proved. Median process CPU time fell by roughly 21–24 ms per request.
Combined peak process RSS was 3.33 GiB and PyTorch reserved accelerator memory
reached 0.916 GiB; these peaks include both eager and compiled model instances.
No request was under 100 ms.

The fixed-shape compile probe measured 38.9 ms eager versus 25.4 ms compiled
median model forward, with 99.992% per-pixel argmax agreement and maximum
logit delta 0.0469. The first uncached 448-pixel compiled invocation took
9.72 seconds. PyTorch notes that initial calls pay JIT compilation costs and
static shape specialization can compile again for new shapes ([compile
documentation](https://docs.pytorch.org/docs/stable/generated/torch.compile.html),
[compile-time guidance](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.reducing_compile_time.html)).
This remains a warm-service optimization candidate, not a cold-start result or
a solution to the 100-ms target. No pipeline code or checkpoint was changed.
Records and the runner are under
`/tmp/wave_v18_compile_fastv1_full_ab_v1/` and
`/tmp/wave_v18_compile_fastv1_full_ab.py`.

### Ryzen AI NPU feasibility recheck (2026-10-29)

The NPU is visible as `RyzenAI-npu5`, but the installed host is Ubuntu 26.04.1
with XRT 2.21.75; no ONNX Runtime or Ryzen AI Python runtime is installed.
AMD's current Linux setup page specifies Ubuntu 24.04 packages with XRT
2.25.37 and Ryzen AI Linux 1.8, and documents CNN INT8/BF16 plus encoder-based
BF16 examples, not SegFormer semantic segmentation ([AMD Linux setup](https://github.com/amd/ryzen-ai-documentation/blob/main/docs/linux.rst),
[model deployment](https://github.com/amd/ryzen-ai-documentation/blob/main/docs/modelrun.rst)).
No compatible isolated runtime is present to benchmark, and I made no system
driver changes. This does not rule out a future NPU route, but the current
installation cannot provide evidence that it will run SegFormer or meet the
latency/MAE gates.

### Compiled SegFormer + exact HEIF decode + `reducing_gap=2`

I combined the three measured fast-v1 candidates and compared them with the
stock eager v18 pipeline on all 47 validation and 46 test photos. Both arms
completed every request. The combined route scored MAE 0.134523 on validation
versus 0.135669 stock (−0.84%) and 0.117448 on test versus 0.116597 stock
(+0.73%). These point estimates meet the 1% screen. Date-cluster 95% intervals
for paired absolute-error change were [-0.00872, +0.00806] on validation
(31 dates) and [-0.00387, +0.00630] on test (27 dates); they do not rule out a
material regression.

Full raw-to-output latency including output writes fell from 317.8/332.2 ms
median/p95 to 220.2/228.3 ms on validation, and from 319.4/334.9 to
221.6/228.0 ms on test. Paired median saving was 96.5 ms across the 93 photos,
about 31% at the median. No request was under 100 ms. Median process CPU time
fell by about 24–27 ms. Peak RSS was 3.35 GiB, PyTorch allocated/reserved GPU
memory was 0.588/0.766 GiB. The compile cache was warm before timing; model
compile startup is excluded. This is the best speed result for the current
v18 full-resolution input route, but the broad confidence intervals, compile
startup, and temporary libheif dependency keep it experimental. The route has
not been integrated. Per-request records are in
`/tmp/wave_v18_fastest_full_ab_v1/paired.json`.

### FP16 versus BF16 SegFormer screen

On the Radeon 8060S, BF16 was supported. At 448 pixels, a single eager B2
forward measured 37.3 ms BF16 versus 39.0 ms FP16, but their class argmax maps
agreed on only 99.904% of pixels. At 384 pixels, BF16 was slower (32.2 ms
versus 28.3 ms) and argmax agreement was 99.924%. Logit deltas reached 0.56 and
0.47 respectively. Because BF16 was slower for Step 2 and altered many class
pixels, I stopped before full-route waviness evaluation and did not change the
profile. Measurements are in `/tmp/wave_v18_precision_screen.py`.

### CPU and copy-reduction follow-up on the compiled full-resolution route

A synchronized eight-photo diagnostic of the compiled + HEIF + `reducing_gap=2`
route found median raw decode 25.4 ms, Step 1 124.7 ms (27.5 ms SegFormer
forward, 97.9 ms remaining image/mask work), Step 2 44.7 ms (19.2 ms forward,
25.1 ms remaining crop/mask work), and regression 4.8 ms. JPEG encode/decode
round trips were each below 0.3 ms in this diagnostic. Median process CPU time
was 376.5 ms and peak RSS/GPU allocation/reservation were 2.82/0.583/0.621
GiB. Synchronizing model calls for attribution changes timing, so these are
stage diagnostics from eight images, not request latency estimates. They show
that further JPEG or regressor tuning cannot close the remaining gap; source
resolution Step 1 image/mask work dominates.

I screened OpenCV thread counts 1, 4, and 8 on 12 test photos. Final JPEGs and
predictions were identical in all three settings. Median end-to-end times were
241.8, 242.5, and 240.4 ms, while median process CPU time rose from 417.8 ms
at one thread to 432.3 and 439.3 ms. This is no repeatable latency gain for
extra CPU use; keep one OpenCV thread for this single-image workload.

I also replaced the Pillow ROI crop with a NumPy view before applying the same
OpenCV mask and Pillow resizer. On the 12-image screen, Step 2 JPEGs and
predictions matched exactly, but median latency changed only from 240.54 to
240.21 ms. This removes an allocation without a demonstrated request speedup.
Finally, direct Pillow BGR-to-RGB decoding via `Image.frombytes` reproduced
the same oriented RGB bytes on all 92 HEIC/HEIF sources as the existing
`cv2.cvtColor` route. In a 12-request paired full-pipeline screen, JPEGs and
predictions matched 12/12; median latency was effectively unchanged (241.25
versus 241.20 ms), while median process CPU time fell about 5.5 ms. These small
copy/thread experiments do not materially advance the 100-ms target and were
not integrated. Temporary artifacts are under
`/tmp/wave_v18_fastest_substage_profile_v1/`,
`/tmp/wave_v18_fastest_cv_threads_screen_v1/`,
`/tmp/wave_v18_roi_view_screen_v1/`, and
`/tmp/wave_v18_decoder_rgb_conversion_ab_v1/`.

### GPU nearest-exact expansion of the Step 1 class mask (2026-10-29)

I tested moving the 448×448 Step 1 class map to the GPU, expanding it to the
source photo dimensions with `torch.nn.functional.interpolate(mode="nearest-exact")`,
then transferring the full-resolution uint8 mask back to the CPU. The operation
supports uint8 and used about 12 MiB for a 3024×4032 output. It is not
pixel-equivalent to the existing Pillow nearest resize: on an initial eight
HEIC sample, six masks differed, by 45–846 pixels each.

I then ran the candidate through the full compiled-v18 + parallel HEIF decode
+ staged Lanczos pipeline on the same 93 labeled validation/test photos. MAE
was 0.135256 on validation versus 0.134523 for the prior combined candidate
(+0.54%), and 0.116713 on test versus 0.117448 (−0.63%). These point estimates
are within 1%, but 40/93 final JPEGs and predictions matched the prior route.
The candidate's separate run measured 241.4 ms median / 256.8 ms p95 on
validation and 243.1 / 256.7 ms on test, compared with 220.2 / 228.3 ms and
221.6 / 228.0 ms in the prior combined run. Since those latency measurements
were not interleaved, treat the cross-run difference as indicative rather than
a paired estimate. The candidate shows no demonstrated speed gain and changes
segmentation outputs, so it is not adopted. Full records and runner are under
`/tmp/wave_v18_gpu_mask_ab_v1/`. The initial pixel-parity counts are recorded
above; the per-request full-pipeline comparisons are in `records.json`.

### CV and runtime performance-principles screens (2026-10-29)

I checked the current implementation against primary framework guidance before
choosing follow-up experiments. PyTorch recommends inference mode, avoiding
unnecessary host/device synchronization and copies, and benchmarking memory
formats on the actual model/backend; it cautions that manual pinning has setup
cost and can be slower than a direct transfer. This path already uses
`torch.inference_mode()`, CPU-side OpenCV/Pillow mask processing, and only moves
the resized SegFormer tensors to the accelerator; prediction serialization
requires the final scalar on the CPU. AMD documents ROCm support for CUDA graphs,
and PyTorch's `reduce-overhead` mode uses them for fixed-shape calls, at the
cost of extra memory and warm-up. OpenCV recommends `INTER_AREA` for image
decimation, while Pillow's `reducing_gap` accelerates a resize by staging an
integer reduction before the final filter ([PyTorch tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html),
[PyTorch transfer guidance](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html),
[PyTorch compile modes](https://docs.pytorch.org/docs/stable/generated/torch.compile.html),
[ROCm/PyTorch support](https://rocm.docs.amd.com/en/docs-6.3.3/compatibility/pytorch-compatibility.html),
[OpenCV resize guidance](https://docs.opencv.org/5.0/main_modules/imgproc_transform.html),
[Pillow reducing-gap behavior](https://pillow.readthedocs.io/en/latest/reference/Image.html)).

I measured channels-last with production FP16 autocast and inference mode on the
Radeon 8060S. The 448 pass changed median model time from 37.02 to 36.76 ms;
the 384 pass changed 24.79 to 25.40 ms. Both produced bit-identical logits on
the probe input. The combined saving is under 1 ms and negative for the second
pass, so this memory-format change is not worth adopting. Full raw-to-output
MAE was not evaluated because the model outputs were identical on the synthetic
probe only. A prior full-request v16 channels-last run is also effectively
neutral; it is recorded earlier in this document.

I checked the rectangular erosion as a pair of one-dimensional erosions. It
was pixel-exact on eight real masks, but took 4.34 ms median versus 3.30 ms for
the existing 25×25 OpenCV erosion. I also restricted erosion to the detected
water bounding box plus the 12-pixel kernel radius. This reproduced the full
mask and Step 1 RGB output exactly on eight photos, but saved only 0.26 ms
median because the water bounding box occupied a median 75% of the frame. MAE
was not recomputed for these exact-output screens; they did not justify a full
pipeline run or code change.

The output-standardization resize is a much larger CPU cost. I compared
OpenCV `INTER_AREA` to Pillow Lanczos on eight real masks: it cut that Step 1
substage from 73.4 to 37.2 ms median, but changed every output. On the complete
interleaved 93-photo raw-to-output run, validation MAE improved from 0.135669
to 0.133991 (−1.24%), while test MAE regressed from 0.116597 to 0.121015
(+3.79%). Median total latency fell from 330.5/332.4 to 291.6/299.8 ms on
validation/test, with 0/93 requests below 100 ms. This fails the 1% MAE screen
and is rejected.

I also tested OpenCV `INTER_LANCZOS4`, a closer filter-family replacement. It
saved 47.9 ms median in the paired raw-to-output run, but MAE regressed by
4.48% on validation (0.141752 vs 0.135669) and 13.55% on test (0.132397 vs
0.116597). Total latency was 281.3/297.3 ms on validation and 285.1/293.4 ms
on test; no request was below 100 ms. This filter is rejected for quality.

Pillow's staged reduction on the SegFormer processor's large bilinear input
resize (`reducing_gap=3`) changed fewer input values than OpenCV's resize and
saved 8.9 ms median in a paired 93-photo run. Validation MAE improved to
0.134786 (−0.65%); test MAE regressed to 0.119727 (+2.68%). The candidate
changed final JPEGs and predictions on 92/93 photos and remains above 100 ms
(323.0/336.9 ms validation; 328.8/338.9 ms test). It fails the 1% screen and
is rejected. This is distinct from the earlier `reducing_gap=2` candidate on
the Step 1/Step 2 Lanczos resizes, which passed the point-estimate screen on
both splits but is still experimental.

I screened `torch.compile(mode="reduce-overhead")` on the two fixed model
shapes, under inference mode. Eager versus compiled medians were 37.17 vs
27.13 ms at 448 and 27.98 vs 19.56 ms at 384. Class argmax agreed on the
synthetic probe; maximum logit difference was 0.03125. Warm compile/capture
took about 2.9 s for 448 and 9.0 s for 384. This was no better than the
previous default compiled model's measured forward times (about 25.4 and
19.2 ms); no full-request MAE was measured, so the mode was not advanced.

The separate `max-autotune-no-cudagraphs` screen took about 200 s to warm its
448 shape and measured 28.07 ms versus 39.73 ms eager on one probe, slower
than the default compiled 448 result. It showed exact argmax on that probe
(maximum logit delta 0.03125). I stopped before compiling 384 or measuring
full-request MAE because the first shape already lost to the default compiled
route and had high startup cost. These compiler-mode probes did not change
pipeline files or the selected checkpoint.

Artifacts are under `/tmp/wave_v18_channels_last_screen.json`,
`/tmp/wave_v18_roi_erode_screen.json`,
`/tmp/wave_v18_erode_separable_screen.json`,
`/tmp/wave_v18_area_resize_full_ab_v1/`,
`/tmp/wave_v18_lanczos4_resize_full_ab_v1/`,
`/tmp/wave_v18_processor_gap3_full_ab_v1/`,
`/tmp/wave_v18_compile_reduce_overhead_screen.json`, and
`/tmp/wave_v18_compile_autotune_screen.py`.

### Full-resolution SegFormer-B0 route with compiled inference and exact HEIF decode (2026-10-29)

I compared the selected v18 B2 pipeline against a full-resolution candidate
that keeps both segmentation passes and the complete 448/384 model-input
profiles. The candidate swaps in pretrained SegFormer-B0, compiles the models
with the default `torch.compile` mode, uses the exact native HEIF tiled
decoder, and applies Pillow staged Lanczos reduction (`reducing_gap=2`). No
source pixels or pipeline stages were skipped. Both routes ran in interleaved
order on the same 93 labeled raw photos; latency includes processing through
the flushed prediction output.

| Route | MAE, validation (47) | MAE, test (46) | Median total latency, validation | Median total latency, test | Requests under 100 ms |
|---|---:|---:|---:|---:|---:|
| Current v18 B2 | 0.135669 | 0.116597 | 300.25 ms | 310.19 ms | 0/93 |
| Full-resolution B0 combo | 0.134233 (−1.06%) | 0.111422 (−4.44%) | 188.13 ms | 188.54 ms | 0/93 |

The paired median latency saving was 121.64 ms. The candidate changed the final
JPEG and prediction for all 93 photos. A 20,000-resample capture-date cluster
bootstrap (31 validation dates and 27 test dates) estimated candidate-minus-
baseline absolute-error deltas of −0.00144 on validation (95% interval
−0.02800 to +0.02194) and −0.00518 on test (−0.02206 to +0.01177). The
intervals include regressions, so the lower point-estimate MAEs are encouraging
but do not establish that quality is preserved. This remains experimental.

Printed stage timings on the candidate had median decode 30 ms, Step 2
lower-water preprocessing 34 ms, and model inference 5 ms. These diagnostics
do not sum to the end-to-end latency because they omit other pipeline work and
timing boundaries overlap. Peak resident memory was 3.26 GiB; peak PyTorch GPU
allocated/reserved memory was 0.60/0.63 GiB. No request met the 100 ms target.

An earlier screen found PyTorch `nearest` and `nearest-exact` did not reproduce
Pillow's mask enlargement on all tested masks. OpenCV `INTER_NEAREST_EXACT`
did reproduce all eight masks byte-for-byte. In the full 93-photo interleaved
run, however, it preserved all final JPEGs and predictions but saved only
2.31 ms paired median latency. That is not a material end-to-end gain and was
not adopted. A microbenchmark that had suggested a much larger Pillow cost
included pending GPU synchronization from an earlier inference; its latency
attribution was invalid and should not be used.

The paired data, date bootstrap, logs, and predictions are in
`/tmp/wave_v18_b0_combo_full_ab_v1/`; the runner is
`/tmp/wave_v18_b0_combo_full_ab.py`.

### Apply the water-class lookup before full-resolution nearest expansion (2026-10-29)

In the fast single-image profile, SegFormer first produces a 448-pixel class
map. The existing path expands that class map to the source dimensions, copies
the full-resolution array into a tensor, then scans it to make a binary water
mask. I moved the same class-ID lookup to the 448-pixel map and nearest-resized
the binary mask directly. Pointwise class lookup commutes with nearest-neighbor
sampling, so the full-resolution water mask should remain identical while
avoiding the intermediate class-map allocation and full-frame lookup pass.

I compared the change in two interleaved full raw-to-prediction runs, with
the same checkpoint, model, decoder, resize behavior, and 93 labeled photos.
The v18 B2 run is the production-relevant result:

| Route | MAE, validation (47) | MAE, test (46) | RMSE, validation | RMSE, test | Total latency, validation (median / p95 / max) | Total latency, test (median / p95 / max) | Under 100 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current v18 B2 path | 0.135669 | 0.116597 | 0.170880 | 0.149327 | 333.64 / 346.64 / 360.27 ms | 337.09 / 351.01 / 356.59 ms | 0/93 |
| v18 B2 lookup before expansion | 0.135669 | 0.116597 | 0.170880 | 0.149327 | 323.64 / 341.66 / 351.07 ms | 328.70 / 342.31 / 345.13 ms | 0/93 |

The paired median saving was 9.59 ms; median process CPU time fell by about
6.6–10.5 ms depending on split. Peak process RSS was 2.93 GiB and peak GPU
allocated/reserved memory was 0.588/0.615 GiB. The absolute timings were slower
than some earlier runs, so the paired saving within this run is the relevant
comparison. This exact-output change is integrated into
`step_1_a_water_segmentation.py`, `step_1_b_preprocess_water_inputs.py`, and
`step_2_a_reduce_black_water_area.py`; overlapping-tile profiles retain their
existing path.

All 93 requests succeeded in both arms; prediction delta was exactly zero on
all 93 photos, so paired MAE and RMSE deltas were both zero.

Within this run, printed stage medians were 56–57 ms for raw decode, 200–211
ms for Step 1 segmentation and standardization, and 59 ms for Step 2. These
stage timers are diagnostic and should not be added to derive total latency.
They show that eliminating one full-resolution intermediate is useful but
small beside the remaining segmentation and crop work.

### Numba-compile the exact Step 2 crop search (2026-10-29)

The Python crop selector took about 9.9 ms on representative 224×224 masks.
I kept its candidate rules and tie ordering, but moved the loops and summed
area calculations into a Numba nopython kernel. A vectorized NumPy rewrite was
slower (12.6 ms versus 9.9 ms across the same 20-mask screen), so it was
discarded. The compiled kernel matched the Python `CropMetrics` on 100
synthetic masks, and the first full v18 B2 run matched all 93 final JPEGs and
predictions exactly.

I then integrated the JIT kernel as `step_2_b_crop_selector.py`, kept the
readable Python implementation as a reference, and warmed the fixed bool-mask
signature during segmentation-model initialization so compilation is outside
the per-photo timer. The complete interleaved run used the same 93-photo
snapshot, checkpoint, raw decoder, early water-mask lookup, and both
segmentation passes:

| Crop selector | MAE, validation / test | RMSE, validation / test | Total latency, validation (median / p95 / max) | Total latency, test (median / p95 / max) |
|---|---:|---:|---:|---:|
| Python reference | 0.135669 / 0.116597 | 0.170880 / 0.149327 | 321.33 / 333.64 / 342.21 ms | 324.14 / 339.41 / 342.97 ms |
| Numba kernel | 0.135669 / 0.116597 | 0.170880 / 0.149327 | 301.15 / 314.52 / 318.01 ms | 303.46 / 321.03 / 322.14 ms |

All 93 requests succeeded in both arms. JPEGs and predictions matched 93/93;
the maximum prediction delta was zero, so MAE and RMSE were unchanged. Paired
median latency saving was 19.13 ms; median process CPU time fell about 18–20
ms. No request was under 100 ms. Peak process RSS was 3.07 GiB and peak GPU
allocated/reserved memory was 0.588/0.615 GiB. With a fresh Numba cache, model
initialization warmed the selector in 0.126 s; that one-time startup cost is
excluded from raw-to-output latency. The cache allows later process starts to
reuse the compiled kernel.

The implementation and dependency are integrated in `step_2_b_crop_selector.py`,
`step_2_a_reduce_black_water_area.py`, `step_1_a_water_segmentation.py`, and
`requirements-common.txt`. Numba's official guidance supports compiled loops
in nopython mode and documents Python 3.14 compatibility in the current
release series ([performance tips](https://numba.readthedocs.io/en/stable/user/performance-tips.html),
[version compatibility](https://numba.readthedocs.io/en/stable/user/installing.html)).
The full paired records and runner are under
`/tmp/wave_v18_b2_numba_integrated_full_ab_v1/` and
`/tmp/wave_v18_b2_numba_integrated_full_ab.py`.

### Combined exact-output changes against the unmodified v18 B2 path (2026-10-29)

I also ran one interleaved 93-photo comparison from the original v18 B2 path
against the combined current changes: class-to-water lookup before nearest
expansion plus the Numba crop selector. The baseline used the original
full-resolution class-map expansion and Python crop search. Both arms used the
same checkpoint, decoder, model settings, two segmentation passes, and output
writes. This was the v18 checkpoint and the 2026-10-28 validation/test split
snapshot (`20261028T081606614158Z`). The labeled evaluation data was not used
to design or tune either exact-output transformation. The Numba signature was
warmed during model initialization.

| Route | MAE, validation / test | RMSE, validation / test | Total latency, validation (median / p95 / max) | Total latency, test (median / p95 / max) |
|---|---:|---:|---:|---:|
| Original v18 B2 | 0.135669 / 0.116597 | 0.170880 / 0.149327 | 335.87 / 355.96 / 356.01 ms | 337.20 / 357.66 / 359.89 ms |
| Combined exact-output changes | 0.135669 / 0.116597 | 0.170880 / 0.149327 | 300.67 / 317.81 / 327.23 ms | 302.99 / 325.69 / 330.02 ms |

All 93 requests succeeded in both arms. JPEGs and predictions matched 93/93;
maximum prediction delta was zero, so MAE and RMSE were unchanged. The paired
median saving was 28.53 ms, and median process CPU time fell about 30–35 ms.
No request was below 100 ms. Peak process RSS was 3.04 GiB; peak GPU
allocated/reserved memory was 0.588/0.615 GiB. These timings are a same-run
comparison; stage and historical medians vary with system load. The complete
records and runner are under `/tmp/wave_v18_b2_exact_combined_full_ab_v1/` and
`/tmp/wave_v18_b2_exact_combined_full_ab.py`.

### Integrated native HEIF decoder plus exact v18 optimizations (2026-10-29)

I added an optional Linux build for the parallel libheif tile decoder and
connected it to shared image loading. `python step_0_b_build_native_heif.py`
builds an ignored local library against the libheif bundled with the installed
Pillow-Heif package. If the helper is not built or a file fails native decode,
the loader uses Pillow. The decoder matched Pillow's oriented RGB pixels on
all 318 HEIF files in the current raw-data folder.

The interleaved full-route comparison used the same v18 B2 checkpoint and
2026-10-28 93-photo split snapshot as above. The candidate combines this
integrated native decoder with the two exact transformations: low-resolution
class-to-water lookup and Numba crop selection. Both arms used two segmentation
passes and the same output writes.

| Experiment | MAE, validation / test | Total raw-to-output latency, validation (median / p95) | Total raw-to-output latency, test (median / p95) |
|---|---:|---:|---:|
| Current v18 B2 pipeline | 0.135669 / 0.116597 | 328.09 / 340.07 ms | 329.07 / 342.91 ms |
| Integrated native HEIF + exact optimizations | 0.135669 / 0.116597 | 286.33 / 300.38 ms | 286.18 / 308.76 ms |

All 93 requests succeeded in both arms. Final JPEGs and predictions matched
93/93, so MAE was unchanged. Paired median latency saving was 37.33 ms. No
request was under 100 ms. The decoder uses additional CPU: median process CPU
time increased from 448 to 477 ms on validation and 447 to 471 ms on test.
Peak RSS was 3.08 GiB; peak GPU allocated/reserved memory was 0.588/0.615 GiB.
This improves end-to-end latency with exact output parity, but the median still
misses the 100-ms target by about 186 ms. The native helper source, build
command, and Python wrapper are `native/libheif/step_0_a_tiled_decode.cpp`,
`step_0_b_build_native_heif.py`, and `step_0_a_native_heif_decoder.py`. Public
libheif headers and the LGPL license notice are included under `native/libheif/`.
Full records and runner: `/tmp/wave_v18_b2_native_integrated_full_ab_v1/` and
`/tmp/wave_v18_b2_native_integrated_full_ab.py`.

### Native HEIF tile-worker count on the exact v18 route

I compared 4, 8, and 16 parallel tile workers in the integrated native libheif
decoder on the 92 HEIF/HEIC photos in the labeled 93-photo snapshot. Decoded
RGB pixels matched the 16-worker output byte-for-byte for all photos at each
setting. Decode-only median latency was 49.46 ms at 4 workers, 31.58 ms at 8,
and 24.59 ms at 16; median process CPU time rose from 152.76 to 161.38 and
187.33 ms. Thus 4/8 workers save CPU but slow the decode, while 16 is the best
latency setting in this sweep.

I then compared 16 versus 24 workers in an interleaved complete raw-to-output
run over all 93 labeled photos. Both settings used the same native decoder,
exact mask/crop path, model, and output writes:

| Experiment | MAE, validation / test | Total raw-to-output latency, validation (median / p95) | Total raw-to-output latency, test (median / p95) |
|---|---:|---:|---:|
| Current exact route, 16 HEIF workers | 0.135669 / 0.116597 | 288.86 / 299.53 ms | 288.30 / 306.69 ms |
| 24 HEIF workers | 0.135669 / 0.116597 | 285.26 / 300.25 ms | 290.38 / 312.53 ms |

All 93 requests succeeded and JPEGs/predictions matched 93/93. Paired median
latency saving was 1.98 ms, but mean process CPU time increased 41.61 ms and
p95 latency worsened on both splits. Peak RSS was 2.56 GiB and accelerator
allocation/reservation was 0.588/0.615 GiB. I kept 16 workers: 24 offers a
small median-only improvement at higher CPU cost and worse tails. Records and
runner are under `/tmp/wave_v18_native_heif_16_24_full_ab_v1/` and
`/tmp/wave_v18_native_heif_16_24_full_ab.py`; the 4/8/16 exact-pixel screen is
in `/tmp/wave_native_heif_workers_screen_v1/`.

### Split SegFormer decoder fusion on the selected fast-v1 profile

The selected v18 profile uses fixed 448/384-pixel FP16 Segmenter calls. I
tested the existing inference-only split-fusion rewrite in this profile; the
same algebraically equivalent rewrite had previously been tried only on the
older 768-pixel FP32 tiled profile. The model inputs, precision, preprocessing,
checkpoint, and output writes were fixed in an interleaved 93-photo comparison.

| Experiment | MAE, validation / test | Total raw-to-output latency, validation (median / p95) | Total raw-to-output latency, test (median / p95) |
|---|---:|---:|---:|
| Current eager decoder fusion | 0.135669 / 0.116597 | 284.36 / 300.78 ms | 289.23 / 306.01 ms |
| Split decoder fusion | 0.135713 / 0.118038 (+0.03% / +1.23%) | 283.23 / 302.42 ms | 287.69 / 300.98 ms |

All 93 requests succeeded. Final JPEGs and predictions matched on only 66/93
photos; the altered FP16 summation order changed the remaining 27. Paired
median latency saving was 1.27 ms and process CPU time fell 2.08 ms on
average, but test MAE regressed 1.23%, beyond the 1% quality gate. Peak RSS
was 2.52 GiB; accelerator allocation/reservation was 0.588/0.643 GiB. I
rejected this rewrite for fast-v1. Records and runner are under
`/tmp/wave_v18_fastv1_split_fusion_full_ab_v1/` and
`/tmp/wave_v18_fastv1_split_fusion_full_ab.py`.

### Profile the remaining exact-route bottleneck; CPU-side FP16 transfer screen

The 93-photo route logs identify the work still dominating after native decode:
median decode was 41 ms, Step 1 segmentation/standardization 199 ms, Step 2
37 ms, both JPEG round trips about 2 ms combined, and regression inference
4 ms. A one-image PyTorch/ROCm profiler capture was instrumented and therefore
is not a latency benchmark; it showed host time in `hipMemcpyWithStream` and
`hipMalloc` around both segmentation calls. This points to Step 1 image
standardization and accelerator input/temporary-buffer handling as the next
targets, rather than the regression head. PyTorch's tuning guide recommends
avoiding unnecessary host/device synchronization and transfers, and AMD's
ROCm profiler can trace copies and kernel activity ([PyTorch tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html),
[ROCm profiling](https://rocm.docs.amd.com/projects/rocprofiler/en/latest/how-to/using-rocprof.html)).

I tested casting the normalized Step 1 processor tensor to FP16 on the CPU
before transfer. The checkpoint already uses FP16 autocast. The cast preserved
the water mask on a 10-photo screen, then I ran an interleaved full comparison
on the same 93 labeled photos. It matched every final JPEG and prediction, so
MAE was unchanged, but wall latency did not improve:

| Experiment | MAE, validation / test | Total raw-to-output latency, validation (median / p95) | Total raw-to-output latency, test (median / p95) |
|---|---:|---:|---:|
| Current exact native-decoder route | 0.135669 / 0.116597 | 286.90 / 302.34 ms | 288.63 / 307.25 ms |
| CPU FP16 input for Step 1 only | 0.135669 / 0.116597 | 285.98 / 299.92 ms | 290.09 / 308.02 ms |

The paired median saving was −1.62 ms (the candidate was slower); no request
was under 100 ms. Median process CPU time fell by only about 1 ms, and peak RSS
was 2.52 GiB with GPU allocated/reserved memory 0.588/0.615 GiB. I did not
adopt the cast. Full records and runner are in
`/tmp/wave_v18_b2_cpu_fp16_firstpass_full_ab_v1/` and
`/tmp/wave_v18_b2_cpu_fp16_firstpass_full_ab.py`.

### Exact Step 1 CPU substage profile (93-photo snapshot)

I timed the canonical Step 1 operations separately on all 93 labeled raw
photos, synchronizing only around the GPU segmentation call. The manually
assembled Step 1 output JPEG matched the canonical function on 93/93 photos.
Median segmentation/mask generation was 102.63 ms (p95 108.02); erosion was
3.53 ms; bounding box and area checks 0.57 ms; Pillow ROI crop 2.21 ms; ROI
array conversion 4.11 ms; OpenCV masking 3.89 ms; and Pillow Lanczos resize
46.19 ms (p95 54.94). The complete isolated Step 1 function measured 170.71 ms
median / 193.83 ms p95. Earlier full-route logs measured about 200 ms for Step
1 under the interleaved prediction benchmark, so these isolated component
times are diagnostic and should not be added to full-route timings. No code
changed and no new regression MAE was calculated; pixel equality at Step 1 was
exact. The profile data is `/tmp/wave_step1_substage_profile_93_with_full.json`.

The next useful work is therefore in segmentation execution and the large
Lanczos reduction. The other measured Step 1 CPU operations together are about
14 ms median, leaving little opportunity in erosion or crop bookkeeping.

### Fixed-shape HIP graph replay for the two SegFormer calls (2026-10-29)

AMD and PyTorch guidance recommend reducing host/device launch overhead when
small inference workloads leave gaps between kernels, while reusing fixed
buffers for repeated shapes. This Ryzen AI Max+ system uses an integrated
Radeon 8060S and the prediction profile makes two fixed-shape B2 calls (448
and 384 pixels), so I screened PyTorch's `torch.cuda.CUDAGraph` HIP path. The
graph runner copied each processor tensor into a persistent input buffer and
replayed captured model forwards; weights, precision, resize, mask, crop,
JPEGs, and regression checkpoint were unchanged. This follows the graph and
launch-overhead recommendations in the official [PyTorch HIP semantics and
CUDA interface notes](https://docs.pytorch.org/docs/main/notes/hip.html),
[PyTorch graph performance notes](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html),
and [AMD HIP performance guidelines](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html).

An isolated synchronized screen showed eager versus graph-replay medians of
38.71 versus 37.71 ms at 448 and 28.66 versus 26.58 ms at 384, including the
copy into persistent graph inputs. Captured logits were bit-identical on the
screened input. I then ran an interleaved, complete 93-photo comparison of
graph replay against eager inference with the same current exact decoder,
mask, crop, JPEG, and output-record route:

| Route | MAE, validation / test | Total raw-to-output latency, validation (median / p95) | Total raw-to-output latency, test (median / p95) |
|---|---:|---:|---:|
| Current exact route with eager SegFormer | 0.135669 / 0.116597 | 291.74 / 302.50 ms | 296.10 / 310.88 ms |
| Fixed-shape HIP graphs for SegFormer | 0.135669 / 0.116597 | 295.00 / 306.30 ms | 297.34 / 316.78 ms |

All 93 requests succeeded. Final Step 2 JPEGs and waviness predictions matched
93/93 exactly, so MAE was unchanged. Graph replay was 3.31 ms slower by paired
median and increased process CPU time by 4.0 ms on average. The graph run
reached 1.57 GiB allocated / 2.13 GiB reserved accelerator memory, compared
with about 0.59 / 0.62 GiB for the existing eager route. The isolated model
forward reduction did not carry through the complete request: processor
tensor copies, graph memory pools, and surrounding CPU work outweighed the
launch savings. I rejected graph replay and made no production change.

Two preliminary full-route attempts were discarded before analysis because
the harness left the shared processor at 384 pixels after graph capture,
causing Step 1 to run at the wrong configured size. The result above restores
the profile's 448-pixel Step 1 size and passes the 93/93 JPEG and prediction
parity check against the current reference. Runner and full paired records:
`/tmp/wave_v18_hip_graph_full_ab_v3.py` and
`/tmp/wave_v18_hip_graph_full_ab_v3/`.

### Per-image pinned host inputs for HIP transfers (2026-10-29)

PyTorch commonly recommends pinned host memory when a data loader can prepare
the next batch while the GPU handles the current one. This pipeline serves one
photo at a time on an integrated Radeon, so there is no independent batch to
overlap. I tested pinning each SegFormer processor tensor and transferring it
with `non_blocking=True`, compared with the existing pageable tensor transfer.
Decoder, model inputs and precision, both segmentation calls, exact mask/crop
logic, JPEGs, regression checkpoint, and output writes were held fixed in an
interleaved 93-photo run.

| Route | MAE, validation / test | Total raw-to-output latency, validation (median / p95) | Total raw-to-output latency, test (median / p95) |
|---|---:|---:|---:|
| Existing pageable input transfer | 0.135669 / 0.116597 | 286.42 / 298.65 ms | 286.59 / 304.73 ms |
| Per-image pin + nonblocking transfer | 0.135669 / 0.116597 | 286.30 / 299.05 ms | 284.18 / 307.14 ms |

All 93 outputs matched exactly. The paired median latency saving was −0.02 ms,
and median CPU time changed by less than 1 ms. Pinning did not improve the
single-photo request and produced a noisier p95, so the existing transfer path
remains. The summary-writing code failed after the paired records were saved,
so peak host/GPU memory was not retained for this run. Per-image records and
the runner are `/tmp/wave_v18_pinned_input_full_ab_v1/paired.json` and
`/tmp/wave_v18_pinned_input_full_ab.py`.

### Shareable RGBX buffer for native HEIF decoding (not adopted)

I screened a Pillow-compatible RGBX buffer to avoid the temporary Python-bytes
copy in the native HEIF decoder. The prototype decoded to a four-channel buffer
and converted it directly to RGB; all 318 raw-folder HEIF images matched the
existing loader byte-for-byte. This decoder-only test did not rerun prediction
or MAE (current full-route reference MAE is 0.135669 validation / 0.116597
test; exact decoded pixels imply unchanged inputs if the route is used).

| Decoder | MAE, validation / test | Decode-only latency (median / p95) |
|---|---:|---:|
| Current native RGB path | 0.135669 / 0.116597 (full-route reference) | 32.25 / 42.43 ms |
| RGBX bridge prototype | Not remeasured; source pixels exact | 30.12 / 42.71 ms |

Paired median decode saving was only 1.18 ms, while p95 was 0.28 ms slower.
Because the extra four-channel buffer adds memory traffic for a marginal median
gain, I did not integrate it. Runner and measurements are in
`/tmp/wave_heif_rgbx_decode_ab.py` and `/tmp/wave_heif_rgbx_decode_ab.json`;
the temporary prototype is `/tmp/wave_heif_tiled_decode_rgbx.cpp`.

I also repeated the isolated transformation on the experimental full-resolution
B0 combo. It matched all 93 JPEGs and predictions exactly, preserved that
candidate's MAE (0.134233 validation / 0.111422 test), and saved 11.31 ms
paired median latency. A first wrapper version omitted `torch.inference_mode()`
and changed model logits; those measurements were discarded. Both reported
runs use inference mode. The B2 paired records and logs are under
`/tmp/wave_v18_b2_earlymask_full_ab_v1/`; the runner is
`/tmp/wave_v18_b2_earlymask_full_ab.py`. The B0 repeat is under
`/tmp/wave_v18_b0_binary_mask_full_ab_v1/`.

### Exact low-resolution erosion lookup screen (not advanced)

I screened whether the full-resolution nearest-expanded binary mask could be
eroded directly from its 448×448 source mask using a small summed-area table.
For each output pixel, the method queried the exact low-resolution source
rectangle that maps into the replicated-border erosion window. On eight
synthetic 3024×4032 masks with the production 25×25 kernel, all 8 eroded masks
matched OpenCV byte-for-byte. After warm-up, however, the Numba lookup took
9.15 ms median versus 0.40 ms for OpenCV erosion alone; it also has to build
source-index maps. The OpenCV path remains faster, so this optimization was
rejected without a full pipeline run. This is a mask-equivalence screen, not a
waviness-quality or end-to-end latency result. The prototype is in
`/tmp/wave_lowmask_erode.py`.

### OpenCV `INTER_LINEAR_EXACT` for SegFormer input resize (experimental)

The Hugging Face SegFormer processor uses bilinear image resizing, rescaling,
and ImageNet normalization. I screened OpenCV's `INTER_LINEAR_EXACT` before
normalization; OpenCV documents it as bit-exact bilinear interpolation, but
its output is not bit-identical to Pillow's resize in this pipeline
([SegFormer processor](https://huggingface.co/docs/transformers/model_doc/segformer),
[OpenCV interpolation flags](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html)).
On all 93 raw photos, every normalized 448-pixel input tensor differed from
Pillow's tensor, although the maximum normalized-value delta was 3.24 on the
normalized scale. A one-image pre-normalization check found a maximum raw
channel difference of 74 levels (428,119 of 602,112 channel values differed),
consistent with the different downsampling filters rather than a rounding-only
change. The processor stage median fell from 40.67 to 3.05 ms in isolation.

I then ran the full interleaved comparison with the selected v18 B2 checkpoint,
the exact native decoder, exact mask lookup/crop selector, and the same two
segmentation passes. The candidate uses OpenCV bilinear resize for both
SegFormer input sizes and keeps the processor's rescale/normalization steps.

| Experiment | MAE, validation / test | Total raw-to-output latency, validation (median / p95) | Total raw-to-output latency, test (median / p95) |
|---|---:|---:|---:|
| Current exact v18 B2 route | 0.135669 / 0.116597 | 284.63 / 303.58 ms | 287.88 / 312.68 ms |
| OpenCV `INTER_LINEAR_EXACT` processor resize | 0.136720 / 0.108323 | 252.51 / 267.56 ms | 255.60 / 276.94 ms |

All 93 requests succeeded. Final JPEGs and predictions changed on all 93.
Validation MAE increased 0.77%; test MAE improved 7.10%. The paired median
latency saving was 32.52 ms, and median process CPU time fell about 32–35 ms.
No request was under 100 ms. Peak RSS was 2.52 GiB; GPU allocated/reserved
memory was 0.588/0.615 GiB.

A 20,000-resample capture-date cluster bootstrap gives candidate-minus-current
MAE intervals of [−0.01512, +0.01720] on validation and [−0.02412, +0.00611]
on test. Both include regression, so the point estimates do not establish
quality preservation. The candidate is not adopted; it is also still about
153 ms above the median target. Full records, bootstrap, and runner are under
`/tmp/wave_v18_b2_cv_linear_exact_full_ab_v1/` and
`/tmp/wave_v18_b2_cv_linear_exact_full_ab.py`; the input-tensor screen is
`/tmp/wave_segformer_resize_equivalence.json`.

I also screened the antialiasing-oriented OpenCV `INTER_AREA` for the
4032-to-448 downscale, retaining `INTER_LINEAR_EXACT` for the 224-to-384
upscale. This follows OpenCV's guidance that area resampling is generally
suited to decimation. On the same 93-photo full-route snapshot:

| Experiment | MAE, validation / test | Total raw-to-output latency, validation (median / p95) | Total raw-to-output latency, test (median / p95) |
|---|---:|---:|---:|
| Current exact v18 B2 route | 0.135669 / 0.116597 | 281.21 / 301.02 ms | 282.91 / 305.20 ms |
| OpenCV area downscale + linear upscale | 0.139559 / 0.118194 | 266.59 / 286.09 ms | 267.09 / 288.63 ms |

The paired median saving was 14.77 ms and median process CPU time fell about
14–18 ms. Validation MAE regressed 2.87% and test MAE 1.37%, failing the 1%
screen; no request was under 100 ms. The route is rejected. Peak RSS was
2.52 GiB and GPU allocated/reserved memory 0.588/0.615 GiB. No bootstrap was
run after it failed the point-estimate gate. Full records and runner are in
`/tmp/wave_v18_b2_cv_area_exact_full_ab_v1/` and
`/tmp/wave_v18_b2_cv_area_exact_full_ab.py`.

### OpenVINO CPU screen for the selected B2 fast-v1 model (2026-10-29)

The earlier OpenVINO CPU evaluation covered SegFormer-B0, so I screened the
already-exported fixed-shape B2 graphs before considering an end-to-end route
change. I compiled the 448 and 384 inputs for OpenVINO CPU FP32, latency mode,
and one stream, and measured warm inference at 8, 16, and 32 inference threads.
The best medians were 54.14 ms at 448 and 39.63 ms at 384, both at 32 threads
(93.77 ms for the two model calls alone). At 16 threads they were 54.49 and
39.75 ms; 8 threads were slower at 74.07 and 54.16 ms. The existing selected
ROCm FP16 path measured 38.71 and 28.66 ms for the corresponding model calls
in the HIP graph screen. Thus even the best CPU-only B2 model result is about
39% slower before CPU image resizing, mask handling, decoding, JPEG round
trips, regression, or output writing. It would also compete with those CPU
stages for cores. I stopped at the model-only gate; no full-request A/B was
run because the candidate already loses on the measured model portion.

| Route | Waviness MAE (validation / test) | Total raw-to-output latency |
|---|---:|---:|
| Current v18 fast-v1 reference | 0.135669 / 0.116597 | 286.33 ms validation / 286.18 ms test median in the integrated native-decoder comparison |
| OpenVINO CPU B2 screen | Not measured; no full pipeline was run | Not measured; model-only screen was slower than the selected ROCm route |

This does not change the selected runtime. Model-only timings are diagnostic
and are not presented as prediction latency. The screen ran with synthetic
input tensors and did not evaluate masks, predictions, or waviness MAE. Its
records were printed by the short-lived runner; the fixed ONNX graphs are
`/tmp/wave-segformer-448.onnx` and `/tmp/wave-segformer-384.onnx`.

### Pillow RGB buffer sharing check for native HEIF output (2026-10-29)

I checked whether the installed Pillow can directly share the native decoder's
three-channel RGB buffer. On a mutable `ctypes` buffer, `Image.frombuffer("RGB",
...)` returned a non-shared image (`readonly=0`; changing the source buffer did
not change the pixel). Pillow's `RGBX` mode did share the source buffer
(`readonly=1`), but converting it back to the pipeline's required RGB mode
allocates another image and carries a fourth channel through memory. That
means it does not provide a safe, zero-copy RGB image to this pipeline. The
earlier end-to-end-adjacent RGBX decoder screen also found only 1.18 ms median
decode saving with a 0.28 ms p95 regression, so I did not change the decoder.
No prediction or MAE rerun was justified by this API check; the current full
route reference remains MAE 0.135669 / 0.116597 and total raw-to-output median
286.33 / 286.18 ms on validation / test.

### Skip redundant same-mode RGB copies before SegFormer preprocessing (2026-10-29)

Pillow's `Image.convert("RGB")` returns a distinct pixel buffer even when the
source is already RGB. The native HEIF loader and Step 1 JPEG decoder both
provide RGB images, so the selected single-image route copied a full-resolution
image before each SegFormer processor call. I compared the current calls with
reusing the input when its mode is RGB, retaining conversion for all other
modes. This changes no pixel values and leaves processor resizing,
normalization, models, masks, crops, JPEGs, and regression checkpoint intact.

The interleaved full-route comparison covered the same 93-photo validation and
test snapshot, including decode, both preprocessing passes, regression, and
output writes. It removed no pipeline stage. Every final JPEG and prediction
matched (93/93), so MAE was exactly unchanged:

| Route | Waviness MAE, validation / test | Total raw-to-output latency, validation (median / p95) | Total raw-to-output latency, test (median / p95) |
|---|---:|---:|---:|
| Existing same-mode RGB conversion | 0.135669 / 0.116597 | 282.84 / 299.20 ms | 286.93 / 306.98 ms |
| Reuse existing RGB image buffer | 0.135669 / 0.116597 | 264.77 / 282.01 ms | 270.59 / 289.16 ms |

Paired median latency fell by 16.03 ms. Median process CPU time fell by
16.76 ms on validation and 16.99 ms on test. Peak RSS was 2.52 GiB and peak
GPU allocated/reserved memory was 0.588/0.615 GiB. No request was under
100 ms, so this is an exact speed improvement but does not meet the latency
target. The selected single-image route now skips the redundant copy for RGB
inputs. Full per-image records and runner are in
`/tmp/wave_v18_rgb_copy_elision_full_ab_v1/` and
`/tmp/wave_v18_rgb_copy_elision_full_ab.py`.

### Full-resolution mask array copy removal screen (2026-10-29)

`np.asarray(resized_mask, dtype=np.uint8)` already returns a read-only NumPy
view backed by a live `bytes` owner. OpenCV erosion accepts this read-only
input, and the crop-search implementation explicitly creates its own writable
lower-water mask before editing. I screened dropping the following `.copy()`
in `run_water_mask`, leaving all segmentation and image operations unchanged.
The candidate preserved JPEGs and predictions exactly on the same 93-photo
full-route comparison, but it did not improve speed:

| Route | Waviness MAE, validation / test | Total raw-to-output latency, validation (median / p95) | Total raw-to-output latency, test (median / p95) |
|---|---:|---:|---:|
| Writable copied mask | 0.135669 / 0.116597 | 267.66 / 278.88 ms | 269.92 / 286.74 ms |
| Read-only NumPy mask view | 0.135669 / 0.116597 | 269.40 / 284.25 ms | 271.26 / 291.28 ms |

MAE was unchanged and all 93 JPEGs and predictions matched, but paired median
latency was 1.61 ms slower with the view; median process CPU time rose by
about 3.4 ms and peak memory was effectively unchanged. Keep the `.copy()`;
the downstream OpenCV path sees no end-to-end advantage from removing it.
Records and runner are in `/tmp/wave_v18_mask_copy_elision_full_ab_v1/` and
`/tmp/wave_v18_mask_copy_elision_full_ab.py`.

### Resize RGB PIL inputs before full-resolution NumPy conversion (2026-10-29)

The Hugging Face SegFormer processor normally converts each full-resolution
PIL image to a NumPy array, then its Pillow resize helper converts that array
back to a PIL image before resizing. I measured a path that passes an existing
RGB PIL image directly into the same Pillow resize and applies the processor's
unchanged rescale, normalization, and channel-order steps afterward. This
avoids the full-frame NumPy allocation and conversion back to PIL while
retaining the same resampling operation. It does not change model inputs or
skip preprocessing.

Before the full run, the resulting 448×448 processor tensors matched the stock
processor exactly on eight raw images. I then ran an interleaved comparison
over the complete 93-photo validation/test snapshot. All requests succeeded;
all 93 final JPEGs and predictions were byte/value identical, leaving waviness
MAE unchanged:

| Route | Waviness MAE, validation / test | Total raw-to-output latency, validation (median / p95) | Total raw-to-output latency, test (median / p95) |
|---|---:|---:|---:|
| Existing processor NumPy-first path | 0.135669 / 0.116597 | 263.98 / 273.75 ms | 266.01 / 284.34 ms |
| Direct Pillow resize before NumPy conversion | 0.135669 / 0.116597 | 240.34 / 254.06 ms | 242.16 / 257.27 ms |

The paired median saving was 24.54 ms. Median process CPU time fell by
26.05 ms on validation and 24.83 ms on test. Peak RSS was 2.55 GiB; peak GPU
allocated/reserved memory was 0.588/0.615 GiB. No request was under 100 ms.
The exact-resize path is installed on the shared SegFormer processor; non-RGB
inputs and calls with resizing disabled retain the library implementation.
Full records and the comparison runner are under
`/tmp/wave_v18_processor_pil_first_full_ab_v1/` and
`/tmp/wave_v18_processor_pil_first_full_ab.py`.

### Vectorized exact Pillow nearest-neighbor mask expansion (2026-10-29)

The single-image path expands its 448×448 binary mask to the source dimensions
with Pillow nearest-neighbor, then copies the result into a writable NumPy
array. I screened two direct implementations. A Numba parallel row loop was
byte-exact and saved 4.02 ms paired median, but raised median process CPU time
by about 69 ms per request and worsened test p95 from 263.25 to 269.91 ms; I
did not adopt it. A vectorized alternative builds small cached x/y source-index
maps by resizing encoded coordinate ramps with Pillow itself, then uses
`np.take` to produce the writable full-size mask. This preserves Pillow's
fixed-point nearest mapping, including its rounding, while avoiding a PIL
full-frame result plus an additional full-frame copy.

I checked the expansion byte-for-byte for landscape, portrait, 448×448, and
other source dimensions, then ran an interleaved 93-photo full-route comparison.
Every final JPEG and prediction matched. MAE was unchanged:

| Route | Waviness MAE, validation / test | Total raw-to-output latency, validation (median / p95) | Total raw-to-output latency, test (median / p95) |
|---|---:|---:|---:|
| Pillow resize and NumPy copy | 0.135669 / 0.116597 | 237.10 / 252.96 ms | 240.05 / 257.43 ms |
| Cached Pillow-coordinate maps + `np.take` | 0.135669 / 0.116597 | 237.40 / 247.41 ms | 238.97 / 254.56 ms |

The paired median saving was 2.60 ms; p95 improved by 5.55 ms on validation
and 2.88 ms on test. Median CPU time fell by 1.14 ms and 0.87 ms,
respectively. Peak RSS was 2.52 GiB and peak GPU allocated/reserved memory was
0.588/0.615 GiB. No request was under 100 ms. I integrated the vectorized
exact path and left the Numba parallel alternative out. Records and runner are
under `/tmp/wave_v18_exact_mask_expand_full_ab_v3/` and
`/tmp/wave_v18_exact_mask_expand_full_ab.py`.

### Persistent FP16 SegFormer weights under ROCm autocast (2026-10-29)

I screened persistent FP16 weights as a way to avoid recasting the model for
each call. Converting the entire model to FP16 made model forwards faster,
but changed 3/8 source-resolution class maps in the initial real-image probe.
The valid full-route comparison with separate, immutable model copies reduced
median latency by 27.38 ms; MAE changed from 0.135669 / 0.116597 to 0.135496 /
0.116901 on validation / test. Test MAE increased 0.26%, and 58/93 final
JPEGs/predictions changed. I did not adopt the full-FP16 route because it
introduced a measured test-quality hit. An earlier interleaved run that
converted one shared model half and back to float was discarded: the round
trip quantized the baseline weights and contaminated its later predictions.
The valid separate-copy runner and per-image records are under
`/tmp/wave_v18_persistent_fp16_full_ab_v2/` and
`/tmp/wave_v18_persistent_fp16_full_ab.py`.

I then kept LayerNorm, BatchNorm, and other operations on the existing
autocast path and stored only Conv1d/2d/3d and Linear weights in FP16. The
logits were bit-identical on eight real 448-pixel inputs. The complete
interleaved 93-photo comparison also matched every JPEG and prediction:

| Route | Waviness MAE, validation / test | Total raw-to-output latency, validation (median / p95) | Total raw-to-output latency, test (median / p95) |
|---|---:|---:|---:|
| FP32 weights with ROCm FP16 autocast | 0.135669 / 0.116597 | 243.59 / 254.73 ms | 246.34 / 263.88 ms |
| Persistent FP16 Conv/Linear weights under autocast | 0.135669 / 0.116597 | 242.88 / 250.74 ms | 243.63 / 261.41 ms |

MAE and outputs were unchanged across all 93 requests. Paired median latency
fell by 3.10 ms; median CPU time fell by 3.37 ms on validation and 2.50 ms on
test. No request was under 100 ms. The selected ROCm fast-v1 model now stores
only autocast Conv/Linear weights in FP16; the CPU fallback converts the model
back to FP32 before retrying. Full records and runner are under
`/tmp/wave_v18_mixed_weight_fp16_full_ab_v1/` and
`/tmp/wave_v18_mixed_weight_fp16_full_ab.py`.

### Parallel destination-strip Lanczos on the current v18 route (2026-10-29)

I rechecked the earlier eight-strip Pillow Lanczos experiment on the current
v18 fast-v1 pipeline. Only the final Step 1 image resize was split across up to
eight destination-width strips, using Pillow Lanczos and source boxes for each
strip. Full-resolution decoding, both SegFormer passes, mask/erosion and crop
rules, JPEG quality, regression checkpoint, and prediction logic were held
fixed. The comparison interleaved both routes across the same 47 validation
and 46 test photos. Each timed request included raw decode, preprocessing,
regression, and writing a preview and result record; model loading and warm-up
were excluded.

| Route | Waviness MAE, validation / test | Total raw-to-output latency, validation (median / p95) | Total raw-to-output latency, test (median / p95) |
|---|---:|---:|---:|
| Current v18 standard Lanczos | 0.135669 / 0.116597 | 238.72 / 259.94 ms | 241.27 / 256.30 ms |
| Eight-strip destination-parallel Lanczos | 0.135602 / 0.116569 | 212.86 / 221.85 ms | 213.66 / 225.96 ms |

All 93 requests succeeded. Paired median latency fell by 25.13 ms. The
candidate used more CPU: median process CPU time increased by 22.28 ms per
request, while peak RSS was 2.53 GiB and peak accelerator allocation/reservation
was 0.53/0.56 GiB. No request was under 100 ms. The final JPEG and prediction
were identical on 73/93 images; on the other 20, sparse one-level Lanczos
rounding differences changed the prediction by at most 0.00149.

Candidate MAE improved by 0.049% on validation and 0.024% on test. A
20,000-resample capture-date-cluster bootstrap gave candidate-minus-reference
MAE intervals of [-0.000156, +0.000009] on validation (31 dates) and
[-0.000116, +0.000051] on test (27 dates). Thus this run provides no measurable
MAE hit and its upper interval bounds are below a 1% regression, although the
test dates overlap training dates and do not establish quality on future
captures. The resize is not integrated: it changes the image-input contract
slightly, increases CPU use, and needs fresh-date confirmation before becoming
the canonical training/inference resize. Full paired records, resource summary,
bootstrap, and runner are under
`/tmp/wave_v18_parallel_lanczos_full_ab_v1/` and
`/tmp/wave_v18_parallel_lanczos_full_ab.py`.

### Lanczos strip worker-count screen (2026-10-29)

I compared four, eight, and sixteen destination-strip workers on the same
93-photo fast-v1 v18 validation/test snapshot. Each run used the same strip
algorithm and complete request boundary as the preceding experiment. Four vs
eight and eight vs sixteen were measured in separate interleaved paired runs;
the eight-worker absolute timings differ slightly between runs, so paired
deltas are the stronger comparison.

| Worker setting | MAE, validation / test | Total latency, validation median / p95 | Total latency, test median / p95 | Median process CPU, validation / test |
|---|---:|---:|---:|---:|
| Four strips | 0.135674 / 0.116562 | 212.75 / 222.02 ms | 212.87 / 228.74 ms | 429.0 / 426.2 ms |
| Eight strips | 0.135602 / 0.116569 | 210.85 / 223.01 ms | 210.65 / 222.06 ms | 439.0 / 432.9 ms |
| Sixteen strips | 0.135639 / 0.116568 | 214.12 / 225.72 ms | 215.18 / 228.41 ms | 459.3 / 455.3 ms |

Four and eight workers were close: eight saved 1.90 ms paired median and had
better test p95 in that run, at a median CPU cost about 9.3 ms higher. Sixteen
workers did not improve on eight (paired median saving -0.02 ms, i.e. slightly
slower) and used about 7.5 ms more median CPU. All arms completed all 93
requests; no request was under 100 ms. MAE remained within 0.05% of the stock
v18 reference on both splits for every setting. The 16-worker option is
rejected; eight is the latency choice for this single-request experiment,
while four is a reasonable lower-CPU alternative when host contention matters.
These worker settings remain experimental until the resize contract is
validated on future capture dates. Runs and summaries are under
`/tmp/wave_v18_parallel_lanczos_4v8_full_ab_v1/`,
`/tmp/wave_v18_parallel_lanczos_8v16_full_ab_v1/`, and their paired runner
files in `/tmp/`.

### Bottleneck interpretation for the current hardware

The full-resolution B0 route's logged stage medians were about 30 ms to decode
the raw image, 115 ms for Step 1 segmentation plus standardization, 34 ms for
Step 2 lower-water preprocessing, and 5 ms for regression. The B0 route's
complete request median was about 188 ms, with MAE 0.134233 validation /
0.111422 test against current v18's 0.135669 / 0.116597. These stage medians
are diagnostic and not additive because timers have different boundaries.
Peak RSS was 3.26 GiB and accelerator allocation/reservation 0.60/0.63 GiB.
The data point to CPU decode and preprocessing as the primary opportunity;
regression inference and accelerator memory are not the limiting stages.

This matches AMD's profiling-first guidance: trace the application, identify
whether work is compute-, memory-, or overhead-bound, optimize the measured
bottleneck, then profile again. AMD also recommends reducing unnecessary
host/device movement and reusing allocations, while noting that the specific
tradeoff must be measured on the target hardware ([HIP performance guidelines](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html),
[GPU performance analysis](https://rocm.docs.amd.com/projects/HIP/en/develop/understand/performance_optimization.html)).
On this integrated Radeon, pinned inputs, HIP graph replay, and a CPU-side FP16
cast were already measured and did not improve full-request latency; the
strip resize is the current CPU-side candidate with a repeatable end-to-end
gain, at the cost of additional host CPU.

### Reuse the raw Step 1 mask for Step 2 on v18 (rejected on validation MAE)

I reran the projected-mask optimization on the selected v18 checkpoint and
fast-v1 profile. Step 1 still ran on the original full-resolution image. Its
eroded sea mask was projected through the Step 1 crop/resize and supplied to
Step 2, skipping only Step 2's second SegFormer call; Step 2 still performed
the black-region check and lower-water crop selection. This was an interleaved
93-photo comparison with the same checkpoint, decoder, and output writes.

| Route | MAE, validation / test | Total raw-to-output latency, validation median / p95 | Total raw-to-output latency, test median / p95 |
|---|---:|---:|---:|
| Current v18 Step 1 + Step 2 segmentation | 0.135669 / 0.116597 | 237.35 / 248.16 ms | 238.86 / 257.44 ms |
| Reuse projected Step 1 mask in Step 2 | 0.142570 / 0.115892 | 208.39 / 221.93 ms | 210.31 / 229.87 ms |

The candidate completed all 93 requests and saved 28.56 ms paired median;
median process CPU time fell 28.90 ms and peak RSS was 2.52 GiB (accelerator
allocation/reservation 0.53/0.56 GiB). However, validation MAE regressed 5.09%,
so the candidate fails the no-quality-hit requirement despite test MAE
improving 0.60%. JPEGs and predictions matched only 42/93. No request was
under 100 ms. I rejected this route and did not adapt the regression head or
change the pipeline: a test improvement cannot compensate for the validation
quality loss, and the measured speed gain alone would still leave the route
above target. Full per-image records, summary, and runner are under
`/tmp/wave_v18_projected_mask_reuse_full_ab_v1/` and
`/tmp/wave_v18_projected_mask_reuse_full_ab.py`.

### Full-resolution B0 with exact processor-copy removal (2026-10-29)

I remeasured the full-resolution SegFormer-B0 candidate using the current
shared image loader and crop/mask optimizations. Both Step 1 and Step 2 still
run segmentation on their complete canonical inputs. The B0 model was compiled
for the same fixed 448/384 profile, kept the v18 regression checkpoint, used
the full-resolution native HEIF decoder, and used Pillow's staged Lanczos
reduction (`reducing_gap=2`). I also installed the already-validated PIL-first
processor resize on the B0 processor. It produced the same Step 2 JPEGs as the
prior B0 route on all 93 photos, while avoiding the processor's full-resolution
NumPy round trip. This is a model-backbone candidate, not a change to the
selected v18 checkpoint.

| Route | MAE, validation / test | Total raw-to-output latency, validation median / p95 | Total raw-to-output latency, test median / p95 |
|---|---:|---:|---:|
| Current v18 B2 reference | 0.135669 / 0.116597 | 238.72 / 259.94 ms | 241.27 / 256.30 ms |
| Full-resolution B0 + staged Lanczos + exact PIL-first processor | 0.134233 / 0.111422 | 142.53 / 150.64 ms | 140.99 / 152.10 ms |
| Same B0 route + eight-strip Lanczos | 0.132327 / 0.109505 | 139.90 / 151.20 ms | 139.11 / 148.32 ms |

The B0 runs completed 93/93 requests, with no request under 100 ms. Eight-strip
Lanczos saved only 1.30 ms paired median versus the same B0 route, while adding
9.80 ms median process CPU time; I would not use the strip variant on this
route. Relative to v18, the B0 point-estimate MAE is lower by 1.06% / 4.44%
on validation / test; the strip variant is lower by 2.46% / 6.08%. However,
the capture-date-cluster bootstrap intervals for candidate-minus-v18 absolute
error were [-0.02912, +0.02211] validation and [-0.02239, +0.01181] test for
B0 without strips. They allow material regression, and the test dates overlap
training dates, so the lower point estimates do not establish preserved
quality. Peak RSS was 2.75 GiB and accelerator allocation/reservation was
0.24/0.32 GiB for the paired B0 run. Keep this candidate experimental pending
independent date quality evidence; its 141 ms median also remains above the
100-ms goal. Records and bootstrap are under
`/tmp/wave_v18_b0_parallel_lanczos_full_ab_v4/`; the earlier B0 comparison is
under `/tmp/wave_v18_b0_combo_full_ab_v1/`.

### Copy the native HEIF buffer directly into Pillow (exact, current loader)

The native HEIF decoder returned an RGB allocation that the Python wrapper
first copied into a temporary `bytes` object with `ctypes.string_at`, then
copied again into Pillow's owned image buffer. I replaced the temporary bytes
copy with a ctypes array view over the same native allocation and pass that
buffer directly to `Image.frombytes`. Pillow still makes the single owned
image copy synchronously; the view remains valid until that operation returns,
then the native allocation is freed as before. This uses the standard ctypes
buffer protocol and does not depend on Python 3.14's `ctypes.memoryview_at`.

On the 318 HEIF/HEIC files in `step-0-raw-data`, decoded RGB pixels matched the
old path exactly on 318/318. Interleaved decode-only median was 27.73 ms for
the old wrapper and 23.26 ms for the direct-buffer wrapper, with 4.98 ms paired
median saved. I then measured the full-resolution B0 + v18-regressor request
on the 93 labeled photos, including both preprocessing passes and output
writes:

| Route | MAE, validation / test | Total raw-to-output latency, validation median / p95 | Total raw-to-output latency, test median / p95 |
|---|---:|---:|---:|
| B0 with temporary `bytes` copy | 0.134233 / 0.111422 | 140.44 / 153.63 ms | 142.07 / 155.22 ms |
| B0 with direct ctypes buffer | 0.134233 / 0.111422 | 131.85 / 148.13 ms | 132.57 / 142.83 ms |

All 93 JPEGs and predictions matched exactly; MAE was unchanged. Paired median
latency fell by 9.23 ms and median process CPU time fell by 8.18 ms. No request
was under 100 ms. Peak RSS was 2.74 GiB and accelerator allocation/reservation
was 0.24/0.32 GiB. This exact copy reduction is integrated into
`step_0_a_native_heif_decoder.py`; the B0 backbone itself remains experimental
because its date-cluster MAE intervals versus v18 allow material regression.
The direct-buffer result improves that candidate but does not establish the
quality gate or meet the latency target. Full records and runner:
`/tmp/wave_v18_b0_heif_buffer_full_ab_v1/` and
`/tmp/wave_v18_b0_heif_buffer_full_ab.py`; 318-image decode comparison:
`/tmp/wave_heif_memoryview_decode_ab.json`.

### Use an RGB array view for the Step 1 water ROI (exact)

The Step 1 standardizer previously called `Image.crop()` to copy the full
water bounding box, then converted that crop to a NumPy view before applying
the binary mask. For RGB inputs, I compared slicing the same ROI from
`np.asarray(image)` instead. The source PIL image remains alive for the view;
OpenCV still writes the same masked RGB result, which Pillow resizes with the
same Lanczos filter. Non-RGB inputs retain the prior crop-and-convert path.

The interleaved comparison used the full-resolution B0 segmentation route,
v18 regressor, direct-buffer HEIF decoder, staged Lanczos, and all 93 labeled
photos. Each request included decode, both segmentation/preprocessing passes,
regression, and output writes:

| Route | MAE, validation / test | Total raw-to-output latency, validation median / p95 | Total raw-to-output latency, test median / p95 |
|---|---:|---:|---:|
| Current v18 B2 reference | 0.135669 / 0.116597 | 238.72 / 259.94 ms | 241.27 / 256.30 ms |
| B0 with copied water ROI | 0.134233 / 0.111422 | 131.94 / 145.24 ms | 133.43 / 140.64 ms |
| B0 with RGB NumPy ROI view | 0.134233 / 0.111422 | 129.53 / 139.47 ms | 129.68 / 140.44 ms |

All 93 final JPEGs and predictions matched exactly, so MAE was unchanged.
Paired median latency fell 0.95 ms and median process CPU time fell 2.62 ms;
peak RSS was 2.74 GiB and accelerator allocation/reservation 0.24/0.32 GiB.
No request was under 100 ms. I integrated the RGB-only view in
`step_1_b_preprocess_water_inputs.py`; other image modes keep the existing
conversion behavior. The B0 route remains experimental because its quality
uncertainty versus v18 is unresolved, and the measured route is still above
the latency target. Full per-image records and runner:
`/tmp/wave_v18_b0_roi_view_full_ab_v1/` and
`/tmp/wave_v18_b0_roi_view_full_ab.py`.

### Profile and exact-mask implementation screen (2026-10-29)

I profiled the full-resolution B0 Step 1 path on 16 raw HEIC images. Median
Step 1 time was 67.62 ms: B0 mask inference 41.27 ms, erosion 3.55 ms,
bounding-box/count work 0.65 ms, ROI extraction plus masked RGB construction
15.61 ms, Lanczos resize 6.12 ms, and canvas placement 0.03 ms. This is a
small diagnostic sample and the stage timings are not a full-request latency
measurement. The path itself is the previously measured B0 + v18 regressor
route (MAE 0.134233 validation / 0.111422 test; total raw-to-output median
129.53 ms after exact decoder and ROI changes).

I compared four exact array masking implementations on the same 16 image
ROIs. The existing OpenCV `bitwise_and` was both fastest among the tested
mask operators (2.91 ms median over repeated operations) and pixel-exact.
NumPy `where` was 34.14 ms and exact; multiplication and bitwise broadcasting
were 23.61 ms and 23.19 ms but did not match exact output. Since the OpenCV
operation itself is only a small part of the 15.61 ms ROI stage, these
alternatives are rejected. MAE is unchanged for the tested existing route;
no full end-to-end rerun was needed for unchanged pixels. Profile and screen
outputs: `/tmp/wave_b0_step1_substage_profile.json` and
`/tmp/wave_mask_ops_ab.json`.

### B0 ROCm FP16-weight storage (rejected by the 1% MAE gate)

PyTorch's tuning guidance recommends benchmarking inference memory formats,
`torch.compile`, and reduced precision on the target backend; our selected v18
loader already stores convolution and linear weights in FP16 under ROCm FP16
autocast. The experimental B0 full-request harness constructed SegFormer
models directly and did not use that loader optimization. I compared the
existing FP32-weight + FP16-autocast B0 route with the production-style
FP16-stored convolution/linear weights. Both used the same B0 checkpoint,
fixed fast-v1 448/384 inputs, compiler, exact preprocessing, v18 regression
head, direct HEIF decoder, and all 93 labeled photos. The arms were paired and
interleaved; this is a runtime optimization comparison, not a changed input
resolution or skipped stage.

| Route | MAE, validation / test | Total raw-to-output latency, validation median / p95 | Total raw-to-output latency, test median / p95 |
|---|---:|---:|---:|
| B0 FP32 stored weights + FP16 autocast | 0.134233 / 0.111422 | 120.05 / 128.62 ms | 119.78 / 126.83 ms |
| B0 FP16 stored conv/linear weights + FP16 autocast | 0.136433 / 0.113367 | 117.37 / 127.15 ms | 117.93 / 127.17 ms |

The candidate saved 3.00 ms paired median and 2.84 ms median process CPU time,
but validation MAE regressed 1.64% and test MAE 1.75%. Final JPEGs matched on
27/93 photos and predictions matched on 27/93. It fails the 1% MAE gate and is
rejected; I made no production code change. Both arms completed all 93
requests, peak RSS was 3.10 GiB (the comparison kept both model instances
resident), and the candidate had one validation request but no test requests
under 100 ms. This is a measured quality/speed regression, not a viable
candidate. Records and runner:
`/tmp/wave_b0_fp16_weights_full_ab_v1/summary.json`,
`/tmp/wave_b0_fp16_weights_full_ab_v1/paired.json`, and
`/tmp/wave_b0_fp16_weights_full_ab.py`.

### MIGraphX B0 with zero-copy HEIF input and OpenCV preprocessing (2026-10-29)

I ran the full-resolution SegFormer-B0 candidate with MIGraphX execution,
direct NumPy access to the native HEIF RGB allocation, OpenCV `INTER_AREA`
downscaling, and four OpenCV CPU threads. Both segmentation passes, original
image resolution, mask processing, JPEG round trips, v18 regressor, and output
writes were retained. The input allocation remained alive for each request;
model compilation and loading were excluded from warm request latency. This
candidate was an isolated `/tmp` runner and did not change the selected
pipeline.

| Route | MAE, validation / test | Total raw-to-output latency, validation (median / p95) | Total raw-to-output latency, test (median / p95) |
|---|---:|---:|---:|
| Current v18 B2 reference | 0.135669 / 0.116597 | 238.72 / 259.94 ms | 241.27 / 256.30 ms |
| MIGraphX B0, zero-copy RGB, OpenCV area / 4 threads | 0.131625 / 0.114461 | 81.27 / 89.18 ms | 83.60 / 87.43 ms |

All 93 labeled requests succeeded. The candidate's point-estimate MAE was lower
by 2.98% on validation and 1.83% on test, and its median total latency was
below 100 ms on both splits. However, the capture-date cluster-bootstrap 95%
intervals for candidate-minus-v18 MAE were [-0.03183, +0.02064] on validation
(31 dates) and [-0.02142, +0.01517] on test (27 dates); both include a material
regression. The test dates overlap training dates. Therefore this is a strong
speed result, but it does not establish quality preservation and is not
adopted. One validation request exceeded 100 ms (128.88 ms total); 46/47
validation and 46/46 test requests were under 100 ms. The slow request,
`img_7565`, was a JPEG whose decode stage took 63.71 ms. Peak RSS was 3.02 GiB;
MIGraphX session compilation/loading took 22.57 and 21.18 seconds, outside
warm request timings. Records, runner, and quality bootstrap are under
`/tmp/wave_migraphx_b0_area_cv4_zero_copy_full_eval_v1/`,
`/tmp/wave_migraphx_b0_area_cv4_zero_copy_full_eval.py`, and
`/tmp/wave_migraphx_b0_area_cv4_zero_copy_quality_bootstrap.json`.

### OpenCV JPEG decode screen on the current Python path (2026-10-29)

Because the slowest MIGraphX request was a JPEG, I compared OpenCV `imread`
plus BGR-to-RGB conversion against the existing shared Pillow loader on all 10
JPEGs in `step-0-raw-data`. The arrays matched byte-for-byte on all 10,
including EXIF orientation. Repeated warm medians varied by image: OpenCV
saved between 1.09 and 7.64 ms on seven images, but was 0.01 to 2.24 ms slower
on the other three. This small and mixed sample does not show a dependable
end-to-end win, so I did not alter the production decoder. No full-request
latency or new MAE was measured; image inputs were exact and the v18 reference
remains MAE 0.135669 / 0.116597. Per-image decoder measurements were collected
interactively on 2026-10-29; rerun a paired full-route benchmark before
considering any JPEG backend change.

I also screened the installed ROCm rocJPEG hardware decoder on the Radeon
8060S. AMD documents that rocJPEG can use the GPU's VCN JPEG engines and
return interleaved RGB; the Linux kernel documentation identifies VCN as the
media engine for hardware image/video decode ([rocJPEG overview](https://rocm.docs.amd.com/projects/rocJPEG/en/develop/what-is-rocJPEG.html),
[rocJPEG API](https://rocm.docs.amd.com/projects/rocJPEG/en/develop/reference/rocJPEG-api-reference.html),
[Linux AMDGPU driver](https://docs.kernel.org/gpu/amdgpu/driver-core.html)). A
local C++ probe used the hardware backend, decoded RGB on the GPU, copied the
full-resolution result to host memory, and applied EXIF orientation. On the
slow `IMG_7565.jpeg`, warm rocJPEG calls had a 40.09 ms median / 41.62 ms p95
before orientation; the host copy and orientation brought the Python-side
median to about 49 ms. A paired OpenCV CPU decoder measured 35.18 / 38.43 ms.
Decoded RGB was not byte-exact: across the 10 repository JPEGs, maximum
per-channel differences ranged from 3 to 28 levels. The hardware path is both
slower in this single-image workload and changes pixels, so it was rejected
before a full prediction-quality run. Probe source and outputs are under
`/tmp/wave_rocjpeg_decoder.cpp`, `/tmp/libwave_rocjpeg.so`, and
`/tmp/rocjpeg_7565.rgb`.

### MIGraphX SegFormer-B2 full-route screen with the v18 input contract (2026-10-29)

I exported the selected SegFormer-B2 backbone to fixed-shape ONNX graphs and
fused bilinear logit upsampling, argmax, and water-class lookup into each graph.
The 448- and 384-pixel calls both return the 448-pixel class mask used by the
current fast-v1 profile; Python then applies the existing nearest-neighbor
mask expansion. Both segmentation passes, the v18 regression checkpoint,
JPEG round trips, crop selection, and output writes were retained. The input
builder used Pillow bilinear resize and the repository's float64 rescale then
float32 normalization order. Its tensors matched the Hugging Face processor
byte-for-byte for all 93 raw images at both input sizes. MIGraphX ran the
exported model in FP32, while the selected ROCm PyTorch route uses FP16
autocast, so model outputs are not assumed to be equivalent.

| Route | MAE, validation / test | Total raw-to-output latency, validation (median / p95) | Total raw-to-output latency, test (median / p95) |
|---|---:|---:|---:|
| Current v18 B2 reference | 0.135669 / 0.116597 | 238.72 / 259.94 ms | 241.27 / 256.30 ms |
| MIGraphX B2, exact processor inputs, FP32 | 0.132469 / 0.111427 | 148.16 / 155.83 ms | 148.00 / 152.69 ms |

All 93 requests succeeded. Candidate MAE point estimates improved by 2.36%
on validation and 4.43% on test, and paired median latency fell about 90 ms
and 93 ms respectively. However, 50,000 capture-date cluster bootstrap draws
gave candidate-minus-v18 MAE 95% intervals of [-0.01251, +0.00688] on
validation (31 dates) and [-0.01359, +0.00327] on test (27 dates). Both allow
a material regression, and test dates overlap training dates; the quality
gate is therefore not established. No request was under 100 ms. Median stage
times were 21.5 ms decode, 26.1 ms Step 1 processor, 35.5 ms Step 1 MIGraphX,
4.2 ms mask expansion, 2.0 ms Step 2 processor, 27.5 ms Step 2 MIGraphX, and
6.7 ms final JPEG/regression. Peak RSS was 4.15 GiB. Each MIGraphX session
took about 37–43 seconds to load/compile outside warm request latency. This
screen shows that compiling the current B2 route alone does not reach the
100-ms target on this host; it remains experimental. Records, runner, and
bootstrap are under `/tmp/wave_migraphx_b2_exact_resize_full_eval_v3/`,
`/tmp/wave_migraphx_b2_exact_resize_full_eval.py`, and
`/tmp/wave_migraphx_b2_exact_resize_quality_bootstrap_v3.json`.

I then replaced only the exact Pillow bilinear input resize with OpenCV
`INTER_LINEAR_EXACT`, keeping MIGraphX B2, the full image, all mask/crop/JPEG
steps, and the v18 regressor. This shrank the Step 1 processor median from
26.1 ms to 8.0 ms; Step 2 processor median changed from 2.0 to 1.6 ms. On all
93 photos, MAE was 0.131744 validation / 0.107415 test and complete-request
latency was 129.85 ms median / 137.83 ms p95 validation and 130.41 / 134.29 ms
test. This is about 18 ms faster than the exact-Pillow MIGraphX B2 run, but
still has no request under 100 ms. Point-estimate MAE improved 2.89% / 7.88%
against v18; the 50,000-draw capture-date bootstrap intervals were
[-0.02465, +0.01427] validation and [-0.02460, +0.00429] test, so material
regression remains possible. Do not adopt it as quality-preserving. Full
records and bootstrap are under
`/tmp/wave_migraphx_b2_cv_linear_full_eval_v1/` and
`/tmp/wave_migraphx_b2_cv_linear_quality_bootstrap.json`; runner:
`/tmp/wave_migraphx_b2_cv_linear_full_eval.py`.

### MIGraphX FP16 B2 route: fast but nondeterministic mask outputs (2026-10-29)

I enabled ONNX Runtime MIGraphX FP16 execution on the fixed-shape B2 mask
graphs while retaining the OpenCV `INTER_LINEAR_EXACT` inputs, two complete
segmentation passes, v18 regressor, full-resolution images, JPEG round trips,
crop selection, and output writes. One full 93-photo run reported MAE
0.131612 validation / 0.104246 test, with total raw-to-output median/p95 of
91.60/97.86 ms validation and 90.89/95.22 ms test. The point-estimate MAE was
better than v18, but date-cluster bootstrap intervals still allowed a
regression: [-0.02483, +0.01423] validation and [-0.02870, +0.00130] test.

Repeatability checks invalidate that run as a dependable quality result. With
the same input tensor and the same MIGraphX FP16 session, ten repeated 448-pixel
calls produced eight distinct binary masks, differing from the first result
by 1–4 pixels. A 10-call MIGraphX FP32 control returned one identical mask in
all calls. The selected PyTorch v18 FP16-autocast model also returned one
identical mask in 20 repeated calls on the same raw image. Most directly, 20
repeated full MIGraphX FP16 requests for `IMG_7830.HEIC` yielded five final
JPEGs and five waviness predictions ranging from 0.334510 to 0.347331 (range
0.012820). The first call included lazy graph compilation and was excluded
from warm latency; the other calls were warm. This shows that a few unstable
mask pixels can change the selected crop and final score. Do not use this
candidate or its single-run MAE as evidence of quality preservation, despite
its sub-100-ms median and p95.

An additional full run that used byte-exact OpenCV JPEG decoding changed the
final JPEG on 35/47 validation and 32/46 test rows, including many HEIC inputs;
its MAE was 0.137866 / 0.102910. Because the JPEG decoder only affected JPEG
inputs and repeated HEIF decoding itself was byte-stable, this reinforces the
FP16-run repeatability failure. Do not attribute the MAE difference to the
JPEG decoder. Outputs and scripts are under
`/tmp/wave_b2_fp16_repeatability_holder/repeatability.json`,
`/tmp/wave_b2_fp16_repeatability.py`,
`/tmp/wave_migraphx_b2_cv_linear_fp16_full_eval_v1/`,
`/tmp/wave_migraphx_b2_cv_linear_fp16_jpeg_full_eval_v1/`, and
`/tmp/wave_migraphx_b2_cv_linear_fp16_jpeg_full_eval.py`.

### Production benchmark profile loading and current v18 route check (2026-10-29)

I found that `step_7_b_benchmark_prediction.py` did not read the selected
checkpoint's preprocessing profile before checking or timing it. It defaulted
to 768-pixel FP32 tiled inference, even though `.env` selects v18 fast-v1
(448/384, FP16, one image per pass). It also wrapped the removed
`run_segmentation` entry point instead of the current `run_water_mask` API.
That made the official benchmark fail before prediction and made its default
configuration wrong for this checkpoint. I corrected it to load the
checkpoint profile before validation, preserve its precision unless explicitly
overridden, apply the profile's Step 2 size, and instrument the current mask
entry point.

The checkpoint's preprocessing code hashes were stale relative to the already
integrated current source files, so the official validator correctly rejected
the original checkpoint. I made a separate v18 checkpoint copy with only its
recorded current-source hashes refreshed; the original checkpoint remains
unchanged. All 360 tensors in its state dictionary are byte-for-byte equal to
the original. `.env` now selects this contract-refreshed v18 copy. Before
changing `.env`, I ran the same current v18 weights through the corrected
official benchmark on all 93 validation/test raw photos, including decode,
both segmentation passes, JPEG round trips, regression, and output record
writes. The newly selected copy then passed the official contract check and
completed a warm raw HEIC request at 231 ms.

| Route | Waviness MAE, validation / test | Total raw-to-output latency, validation (median / p95 / max) | Total raw-to-output latency, test (median / p95 / max) |
|---|---:|---:|---:|
| Current v18 fast-v1, warm official-runner requests | 0.135669 / 0.116597 | 236.74 / 268.90 / 286.69 ms | 240.92 / 270.06 / 286.43 ms |

All 93 warm requests succeeded; 0/93 were under 100 ms. MAE exactly matched the
checkpoint's reported validation/test values. This run confirms the current
route is around 237–241 ms median in the official entry point. The hash mismatch
is resolved for the current v18 pipeline without changing model weights. The
new copy records the current code contract; it does not include the separate
eight-strip candidate. Runner and report:
`step_7_b_benchmark_prediction.py` and
`/tmp/wave_v18_official_93_current/20261029T061942610068Z/benchmark.json`.

### Production-code rerun of eight-strip Lanczos (2026-10-29)

I repeated the full paired eight-strip resize experiment through the
production `step_1_b_preprocess_water_inputs.py` standardization function,
using its previously measured source-box strip implementation. The rerun used
the same 93 labeled raw photos and current v18 weights. Its stock arm used the
unchanged single-call Pillow Lanczos resize. The candidate used eight strips;
no pipeline stage or source resolution changed. The eight-strip JPEGs and
predictions matched the earlier experimental candidate on all 93 photos.

| Route | Waviness MAE, validation / test | Total raw-to-output latency, validation (median / p95 / max) | Total raw-to-output latency, test (median / p95 / max) |
|---|---:|---:|---:|
| Current single-call Lanczos | 0.135669 / 0.116597 | 225.17 / 242.52 / 244.56 ms | 229.44 / 244.45 / 244.96 ms |
| Eight-strip Lanczos | 0.135602 / 0.116569 | 195.67 / 206.89 / 207.98 ms | 195.38 / 203.76 / 210.63 ms |

The paired median saving was 32.02 ms, with median process CPU time increasing
13.19 ms. There were 93/93 successful predictions. MAE improved by 0.049% on
validation and 0.024% on test. The date-cluster bootstrap was unchanged from
the repeated candidate because all predictions matched it: candidate-minus-
reference MAE 95% intervals were [-0.000156, +0.000009] validation and
[-0.000116, +0.000051] test, both well inside the 1% regression margin. This
is the strongest repeatable v18 speed/quality result in the current full-
resolution route, but remains above 100 ms. The capture dates still overlap
training dates, and the changed resize must be assigned a distinct
preprocessing contract before adoption. Rerun records and date bootstrap:
`/tmp/wave_v18_parallel_lanczos_integrated_full_ab_v1/`.

### Quality-filtered full-route experiment index (2026-10-29)

This index consolidates the measured full raw-image-to-output experiments
against the selected v18 checkpoint. I retained experiments only when their
point-estimate waviness MAE was no more than 1% worse than that run's v18
reference on **both** validation and test. Candidates over that threshold are
left out of this quality-qualified list. Rows with exact prediction parity
have identical MAE by construction. Latency is total request latency; the
table uses median validation / test values where both are available. The
underlying sections above contain p95, cohort sizes, paired analysis, and
limitations. Because snapshots and timing load differ, compare candidates to
their same-run reference rather than interpreting small absolute differences
between rows as a controlled ranking.

| Experiment | MAE, validation / test | Total raw-to-output latency, validation / test median | MAE gate / status |
|---|---:|---:|---|
| **Current v18 fast-v1 reference (official runner)** | **0.135669 / 0.116597** | **236.74 / 240.92 ms** | Baseline; 0/93 under 100 ms |
| OpenCV `INTER_NEAREST_EXACT` mask expansion, v18 PyTorch | 0.135669 / 0.116597 | 222.98 / 224.75 ms | Exact prediction parity on 93/93; indicative 13.8 / 16.2 ms lower median; still 0/93 under 100 ms |
| Eight-strip Lanczos on current v18 PyTorch ROI path | 0.135602 / 0.116569 | 187.39 / 191.22 ms | −0.049% / −0.024%; p95 198.73 / 201.31 ms; passes point MAE gate, remains above 100 ms |
| Stable MIGraphX FP16 + OpenCV exact-nearest + 24 HEIF workers | 0.131969 / 0.104235 | 92.76 / 92.34 ms | −2.73% / −10.60%; 92/93 under 100 ms; date bootstrap permits >1% regression |
| Eight-strip Lanczos resize (production-code paired rerun) | 0.135602 / 0.116569 | 195.67 / 195.38 ms | −0.049% / −0.024%; 32.02 ms paired median saving; contract/date validation still needed |
| Exact water-class lookup before full-resolution mask expansion | 0.135669 / 0.116597 | 323.64 / 328.70 ms | Same-run MAE identical; small exact speed win in that run; integrated |
| Numba exact Step 2 crop selector | 0.135669 / 0.116597 | 301.15 / 303.46 ms | Exact parity; 19.13 ms paired median saving; integrated |
| Combined exact lookup + Numba crop selector | 0.135669 / 0.116597 | 300.67 / 302.99 ms | Exact parity; 28.53 ms paired median saving |
| Native libheif decoder + exact lookup/crop optimizations | 0.135669 / 0.116597 | 286.33 / 286.18 ms | Exact parity; 37.33 ms paired median saving; packaging caveat |
| Native HEIF decoder, 24 vs 16 tile workers | 0.135669 / 0.116597 | 285.26 / 290.38 ms (24 workers) | Exact parity; small/uneven gain, higher CPU and worse tails |
| CPU-side FP16 input tensor for Step 1 | 0.135669 / 0.116597 | 285.98 / 290.09 ms | Exact parity; slower paired median; rejected for speed |
| Fixed-shape HIP graph replay | 0.135669 / 0.116597 | See graph-replay section | Exact parity; no full-request latency improvement |
| Pillow staged `reducing_gap=2` resize | 0.135168 / 0.117383 | 289.8 / 292.0 ms | −0.37% / +0.67%; point-estimate gate passes; uncertainty remains |
| Exact HEIF decode + `reducing_gap=2` | 0.135168 / 0.117383 | 263.6 / 268.8 ms | −0.37% / +0.67%; fastest qualified point-estimate route in that run; unadopted dependencies/uncertainty |
| Compiled B2 fast-v1 | 0.133432 / 0.116534 | 306.5 / 311.1 ms | −1.65% / −0.05%; point-estimate gate passes; date-cluster uncertainty remains |
| Compiled B2 + exact HEIF + `reducing_gap=2` | 0.134523 / 0.117448 | 220.2 / 221.6 ms | −0.85% / +0.73%; point-estimate gate passes; confidence intervals remain broad |
| GPU nearest-exact mask expansion on compiled combo | 0.135256 / 0.116713 | 241.4 / 243.1 ms | +0.54% / −0.63%; passes point-estimate gate, but slower and not adopted |
| Embedded-thumbnail B2 + adapted v18 head | 0.131680 / 0.115703 | 68.47 / 73.67 ms | −4.99% / −0.77%; sub-100 ms on successes; coverage and date-disjoint quality remain unproven |
| Full-resolution SegFormer-B0 + compiled inference | 0.134233 / 0.111422 | 188.13 / 188.54 ms | −1.06% / −4.44%; passes point-estimate gate, but date-cluster uncertainty remains broad |
| B0 with direct HEIF buffer and RGB ROI view | 0.134233 / 0.111422 | 129.53 / 129.68 ms | Same B0 MAE; exact copy optimizations; quality vs v18 still not statistically cleared |
| MIGraphX B0 + OpenCV area preprocessing | 0.131625 / 0.114461 | 81.27 / 83.60 ms | −2.98% / −1.83%; point-estimate gate passes; bootstrap permits material regression |
| MIGraphX B2 exact processor inputs, FP32 | 0.132469 / 0.111427 | 148.16 / 148.00 ms | −2.36% / −4.43%; point-estimate gate passes; bootstrap permits material regression |
| MIGraphX B2 + OpenCV `INTER_LINEAR_EXACT` | 0.131744 / 0.107415 | 129.85 / 130.41 ms | −2.89% / −7.88%; point-estimate gate passes; bootstrap permits material regression |
| OpenCV `INTER_LINEAR_EXACT` in the PyTorch B2 route | 0.136720 / 0.108323 | 252.51 / 255.60 ms | +0.77% / −7.10%; point-estimate gate passes; date-cluster uncertainty remains broad |
| B2-distilled B0 + MIGraphX FP32 + full-resolution OpenCV preprocessing | 0.110818 / 0.110720 | 78.69 / 79.84 ms | −18.32% / −5.04%; 93/93 succeeded and 92/93 under 100 ms; date-disjoint check fails quality gate |

The MIGraphX FP16 B2 route is deliberately absent despite a favorable single-
run MAE and sub-100-ms timing: repeated identical inputs produced different
masks and waviness scores, so its reported quality is not dependable.

### PyTorch `torch.compile` on the unchanged fast-v1 SegFormer (2026-10-29)

PyTorch's performance guide recommends measuring compiler fusion and memory
format options on the target workload; ROCm likewise recommends profiling
before changing kernels or transfers. I screened `torch.compile` on the
existing SegFormer-B2 with the fast-v1 448/384 inputs, FP16 autocast, source
resolution, preprocessing, and v18 regression checkpoint unchanged. Warm
model-only forward medians on a real photo fell from 34.77 to 20.35 ms at 448
pixels and from 28.50 to 22.81 ms at 384 pixels. The compiled logits were not
bit-identical (maximum absolute difference 0.046875 / 0.023438); on an
eight-photo mask screen, 7 of 16 class maps changed, by seven pixels total.

I then ran the canonical fresh raw-to-output benchmark on the same 93 labeled
photos and compared its predictions with the eager v18 full-route reference.
All 93 requests succeeded. This is a sequential run rather than an
interleaved timing comparison, so treat small latency differences as
indicative, not a controlled paired speed claim:

| Route | Waviness MAE, validation / test | Total raw-to-output latency, validation median / p95 | Total raw-to-output latency, test median / p95 | Under 100 ms |
|---|---:|---:|---:|---:|
| Eager v18 reference | 0.135669 / 0.116597 | 236.74 / 268.90 ms | 240.92 / 270.06 ms | 0/93 |
| `torch.compile` B2 experiment | 0.135076 / 0.118273 | 211.88 / 230.21 ms | 217.38 / 233.38 ms | 0/93 |

Validation MAE improved 0.44%, but test MAE regressed 1.44%, above the
quality gate. The experiment therefore fails even though its warm full-route
medians were about 25 ms lower in validation and 24 ms lower in test. The
compiled full-run stage medians were 31.25 ms decode, 145.66 ms Step 1,
32.33 ms Step 2, and 5.71 ms regression. No full-run RSS or process CPU
measurement was captured. I did not adopt this change. The benchmark report
is `/tmp/wave_v18_torch_compile_eval/20261029T072113805869Z/benchmark.json`;
the temporary runner was removed after the run.

These findings reinforce the optimization order: the small regression model
and JPEG round trips are not the latency bottlenecks; Step 1 mask
standardization and segmentation still dominate. Exact memory/copy changes
are useful but cannot close the remaining gap alone. A compiler or model
change must be judged on the full request and the untouched waviness MAE gate,
not isolated forward speed or mask similarity alone. This follows the official
[PyTorch performance tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html),
[AMD HIP performance guidelines](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html),
and [MIGraphX model performance tuning guide](https://rocm.docs.amd.com/projects/AMDMIGraphX/en/develop/how-to/model-performance-tuning.html).

Not every exploratory screen belongs in this table: model-only timings,
synthetic probes, and exact-output microbenchmarks without a full MAE rerun
remain in their dated sections, clearly marked as diagnostics. Experiments
that regressed MAE by more than 1% on either split—including split decoder
fusion, projected-mask reuse, reduced-precision B0 storage, and the more
aggressive resize/mask variants—are omitted from this candidate table. The
B2-to-B0 distillation result passes this table's point-estimate filter, but
its four-photo date-disjoint check regressed and its capture-date bootstrap
still allows material test regression. It is not quality-cleared.

### B2-to-B0 binary segmentation distillation with the runtime resize contract (2026-10-29)

I distilled a SegFormer-B0 binary water/not-water student from the selected
B2 teacher using both input domains required by the raw-to-output pipeline:
raw photos for Step 1 and the previous Step 1 output for Step 2. The 217
training rows were split by capture date into 169 fit images across 39 dates
and 48 selection images across 12 dates. Validation and test photos were not
used for training or model selection. The first student used the Hugging Face
Pillow resize for training but OpenCV resize at inference; I corrected that
train/inference mismatch and regenerated all 434 teacher targets from the
exact candidate OpenCV input transform. The matched student reached 0.9517
mean IoU on the training-date holdout (water IoU 0.9606). This is mask
agreement with the teacher, not the waviness quality gate.

The PyTorch full-route check used the canonical v18 448/384 profile, both
segmentation passes, the existing regression checkpoint, both JPEG round
trips, and output writes. All 93 labeled requests succeeded:

| Route | Waviness MAE, validation / test | Total raw-to-output latency, validation / test median | Requests under 100 ms |
|---|---:|---:|---:|
| Current v18 B2, paired PyTorch reference | 0.135669 / 0.116597 | 231.12 / 232.21 ms | 0/93 |
| B2-distilled B0, PyTorch | 0.120761 / 0.111602 | 180.50 / 183.67 ms | 0/93 |

I then exported the student as two fixed-shape ONNX graphs and ran MIGraphX
FP32 with full-resolution source images, OpenCV `INTER_AREA` for the Step 1
downscale and `INTER_LINEAR_EXACT` for the Step 2 upscale, native HEIF decode,
both segmentation passes, crop/mask logic, JPEGs, the v18 regressor, and output
files. Session load/compile was about 21 seconds per graph and excluded from
warm request latency. The complete 93-photo run reported:

| Route | Waviness MAE, validation / test | Total raw-to-output latency, validation median / p95 / max | Total raw-to-output latency, test median / p95 / max |
|---|---:|---:|---:|
| Current v18 B2 official runner | 0.135669 / 0.116597 | 236.74 / 268.90 / 286.69 ms | 240.92 / 270.06 / 286.43 ms |
| Distilled B0, MIGraphX FP32, OpenCV resize | 0.110818 / 0.110720 | 78.69 / 85.75 / 107.63 ms | 79.84 / 85.44 / 86.14 ms |

All 93 requests succeeded and 92/93 were under 100 ms; the remaining request
was `IMG_7565.jpeg` at 107.63 ms. The MIGraphX route's 50,000-draw
capture-date-cluster MAE-difference intervals versus v18 were [−0.05215,
+0.00046] on validation (31 dates) and [−0.02566, +0.01152] on test (27
dates). The validation upper bound is within the 1% regression margin, but
the test interval allows a material regression. The test set overlaps train
dates. On the four 2026-08-24 test photos, a date absent from training,
unadapted student MAE was 0.18644 versus 0.15744 for v18; this is a small,
single-date sample, but it is a caution against claiming preserved
future-date quality.

I also tried a regression-head-only adaptation using 215 successful
training-route inputs, selecting learning rate and epoch from a capture-date
holdout (167 fit / 48 select; 9 held dates). The selected rate was 1e-4 and
the refit used one epoch. Applying the adapted head to the already-generated
candidate inputs gave MAE 0.11211 validation / 0.11157 test, slightly worse
than the original head's 0.11082 / 0.11072. On the four date-disjoint photos,
MAE was 0.19049, also worse than both the unadapted student (0.18644) and v18
(0.15744). I did not advance this head adaptation.

I compared OpenCV worker counts 1 and 4 in a paired, interleaved full-route
run with the same MIGraphX model, inputs, resize operators, and output code.
JPEGs and waviness predictions matched on all 93 photos, so MAE was exactly
unchanged:

| OpenCV threads | MAE, validation / test | Total latency, validation median / p95 / max | Total latency, test median / p95 / max | Under 100 ms |
|---:|---:|---:|---:|---:|
| 1 | 0.110818 / 0.110720 | 94.36 / 99.78 / 122.94 ms | 94.05 / 99.61 / 100.58 ms | 90/93 |
| 4 | 0.110818 / 0.110720 | 80.39 / 90.11 / 107.87 ms | 80.53 / 86.36 / 86.85 ms | 92/93 |

One OpenCV thread reduced median process CPU time by about 6 ms but increased
paired median wall latency by 13.60 ms. Four threads are faster for this
single-request route; reducing CPU thread count did not help the 100-ms goal.

Peak process RSS during the MIGraphX full run was about 3.1 GiB. The p95
request latency remained below 100 ms, but maximum latency exceeded it; CPU
process time was around 274 ms per request. Therefore this is a promising
speed and aggregate-MAE experiment, not a quality-cleared deployment
candidate. MIGraphX FP16 remains excluded because repeated identical inputs
produce nondeterministic masks. Student training artifacts, per-photo
predictions, and runners are under `/tmp/wave_b2_b0_binary_distill_area_v1/`,
`/tmp/wave_b2_b0_binary_distilled_full_ab_v2/`,
`/tmp/wave_migraphx_b0_distilled_area_full_eval_v5/`,
`/tmp/wave_b2_b0_distilled_area_regressor_adaptation_v1/`, and
`/tmp/wave_migraphx_b0_distilled_area_full_eval_v5.py`.


### Exact OpenCV nearest mask expansion in the official v18 route (2026-10-29)

The production Step 1 mask expansion now uses OpenCV `INTER_NEAREST_EXACT`,
which matched the previous Pillow nearest-neighbor coordinates byte-for-byte.
I reran the complete official raw-to-output benchmark on all 93 labeled raw
photos using the selected v18 weights, including decoding, both segmentation
passes, JPEG round trips, regression, and output-record writes. Because the
checkpoint contract hashes the source files, I used a temporary checkpoint
copy with refreshed source hashes; all 360 state-dictionary tensors were
unchanged. The original checkpoint was not modified.

| Experiment | Waviness MAE, validation / test | Total raw-to-output latency, validation median / p95 | Total raw-to-output latency, test median / p95 |
|---|---:|---:|---:|
| Current v18 reference | 0.135669 / 0.116597 | 236.74 / 267.81 ms | 240.92 / 268.79 ms |
| OpenCV exact-nearest mask expansion | 0.135669 / 0.116597 | 222.98 / 236.37 ms | 224.75 / 244.12 ms |

All 93 predictions were exactly equal to the v18 reference and all requests
succeeded. The candidate run was sequential, so the latency comparison is
indicative rather than interleaved; observed medians were 13.76 ms lower on
validation and 16.17 ms lower on test. This clears the requested point-estimate
MAE filter with zero MAE regression, but total latency remains above 100 ms.
The run reports are `/tmp/wave_v18_cvnearest_official/20261029T080249737155Z/benchmark.json`
and `/tmp/wave_v18_official_93_current/20261029T061942610068Z/benchmark.json`.


### Stable MIGraphX FP16 determinism configuration (2026-10-29)

I found a repeatable MIGraphX FP16 configuration by disabling graph autotuning
and the fast softmax path with `MIGRAPHX_SKIP_BENCHMARKING=1` and
`MIGRAPHX_USE_FAST_SOFTMAX=0`. Two separately initialized 93-photo runs
produced identical Step 2 JPEGs and final predictions on all 93 photos. The
quality point estimates improved against v18, so this experiment passes the
requested 1% point-estimate filter; it remains a research candidate because
the capture-date validation bootstrap still permits more than 1% regression.
Its latency is also near the 100-ms threshold, with p95 above 100 ms in one
split.

| Experiment | Waviness MAE, validation / test | Total raw-to-output latency, validation median / p95 | Total raw-to-output latency, test median / p95 |
|---|---:|---:|---:|
| Stable MIGraphX FP16 B2 | 0.131969 / 0.104235 | 98.59 / 109.03 ms | 98.32 / 101.72 ms |
| Add OpenCV nearest-exact mask expansion | 0.131969 / 0.104235 | 94.08 / 100.84 ms | 93.22 / 97.98 ms |
| Add in-place float32 input normalization | 0.131969 / 0.104235 | 94.22 / 101.23 ms | 93.32 / 97.09 ms |

The OpenCV expansion and in-place normalization variants preserved all 93
JPEGs and scores against the stable MIGraphX route. The in-place normalization
showed exact output parity but no further paired end-to-end timing advantage.
Thus it remains a measured candidate optimization, not an independently
proven latency reduction. Full-run artifacts are under
`/tmp/wave_migraphx_b2_cv_linear_fp16_skipbench_full_eval_v1/`,
`/tmp/wave_migraphx_b2_cv_linear_fp16_skipbench_full_eval_v2/`,
`/tmp/wave_migraphx_b2_cv_linear_fp16_cvnearest_full_eval_v1/`, and
`/tmp/wave_migraphx_b2_cv_linear_fp16_cvnearest_inplace_full_eval_v1/`; the
paired resize benchmark is `/tmp/wave_migraphx_b2_cvnearest_paired_v1/summary.json`.


### CPU/memory profiling and exact sub-100-ms route experiment (2026-10-29)

I reviewed AMD's ROCm performance guidance, PyTorch's profiler guidance, and
OpenCV's parallel-processing documentation before selecting the next trials.
The shared principles are to profile the complete request first; optimize the
measured bottleneck; minimize unnecessary full-resolution copies and host/device
transfers; and tune parallelism on the actual hardware rather than equating
more threads with lower latency. AMD specifically documents mapped pinned
memory as an option on integrated systems, but that remains an untested
hypothesis for this PyTorch/ROCm path. The official guidance is [AMD HIP
performance guidelines](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html),
[PyTorch Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html),
and [OpenCV parallel processing](https://docs.opencv.org/5.0/tutorials/core/how_to_use_OpenCV_parallel_for_new/how_to_use_OpenCV_parallel_for_new.html).

For the selected PyTorch v18 route, I measured OpenCV's default 32 threads
against 4 threads over all 93 raw photos. All scores stayed exactly equal.
The 4-thread run's medians were 221.39 ms validation / 224.29 ms test, versus
222.98 / 224.75 ms at 32 threads; this small difference is within run
variation and does not justify changing the default by itself. An in-place
`cv2.erode` also matched the allocated result exactly, but was slower (2.04 ms
vs 1.07 ms median) and did not improve the 75-ms standardization stage.

A per-stage CPU breakdown on a real 9.04-megapixel Step 1 water ROI showed
`cv2.bitwise_and` at about 3.88 ms, Pillow `Image.fromarray` at 6.63 ms, and
single-call Pillow Lanczos resize at 58.12 ms median. The existing eight-strip
Lanczos hypothesis reduced the current PyTorch v18 standardization median
from 77–78 ms to 42–43 ms. On the full 93-photo run it measured 187.39 ms
validation / 191.22 ms test median, with p95 198.73 / 201.31 ms. MAE was
0.135602 / 0.116569, a 0.049% / 0.024% improvement over v18; all 93 predictions
matched the previously measured eight-strip candidate exactly. This is a real
quality-filter-passing improvement, but it remains above 100 ms and is not
integrated into the production profile. Candidate report:
`/tmp/wave_v18_eightstrip_current_full/20261029T080919147816Z/benchmark.json`.

I restored the MIGraphX FP16 runtime in an isolated Python 3.12 environment
using AMD's ROCm 10.1 wheels, then compared the exact stable route with mask
expansion through OpenCV `INTER_NEAREST_EXACT` and the native HEIF tile decoder
at 8, 16, and 24 workers. The 8-worker decoder lowered process CPU time but
increased wall latency (validation/test medians 104.13 / 103.07 ms); it is
rejected for this latency objective. The 24-worker route returned byte-identical
Step 2 JPEGs and exactly equal scores versus the 16-worker route on all 93
photos, while improving median total raw-to-output time to 92.76 / 92.34 ms
and p95 to 97.93 / 96.02 ms. All 46 test requests and 46/47 validation
requests were under 100 ms (92/93 overall). Median process CPU time was
318.8 / 315.7 ms, showing that parallel CPU work can exceed elapsed wall time;
peak RSS was 3.16 GiB. The one validation request above the limit keeps the
strict per-request target from being met on every sample.

| Route | Waviness MAE, validation / test | Total raw-to-output latency, validation median / p95 | Total raw-to-output latency, test median / p95 | Requests under 100 ms |
|---|---:|---:|---:|---:|
| Current v18 PyTorch official route | 0.135669 / 0.116597 | 236.74 / 267.81 ms | 240.92 / 268.79 ms | 0/93 |
| Stable MIGraphX FP16 + OpenCV exact-nearest, 16 HEIF workers | 0.131969 / 0.104235 | 94.85 / 103.20 ms | 94.87 / 99.51 ms | 87/93 |
| Stable MIGraphX FP16 + OpenCV exact-nearest, 24 HEIF workers | 0.131969 / 0.104235 | 92.76 / 97.93 ms | 92.34 / 96.02 ms | 92/93 |

The 24-worker candidate's point-estimate MAE improves 2.73% validation and
10.60% test versus v18. I recomputed its 50,000-draw capture-date bootstrap
against the official v18 raw-to-output predictions: candidate-minus-v18 MAE
95% intervals were [-0.02483, +0.01494] validation and [-0.02826, +0.00128]
test. Both intervals permit more than 1% MAE regression. The point estimates
pass, but this is not yet sufficient evidence to claim quality cannot regress.
A four-photo date-disjoint test group had MAE 0.13552 vs 0.15744 for v18; this
sample is too small to close that uncertainty.

I also generated the candidate route's Step 2 outputs for the 217 training
photos and tried a training-only regression-head adaptation. A 10-date holdout
selected LR 1e-5 for 8 epochs; holdout MAE changed only from 0.107748 to
0.107719. Refit on all training rows changed candidate-route MAE from
0.131969 to 0.131766 validation and 0.104235 to 0.104098 test, while test
RMSE worsened slightly from 0.130212 to 0.130565. On the four date-disjoint
photos, MAE worsened from 0.13552 to 0.13732. I reject this head adaptation.
The unadapted stable route remains the best speed candidate, pending more
independent date coverage for quality confidence.

The MIGraphX environment is isolated at `/tmp/wave_migraphx_312_env`; its
packages were not added to project requirements. Full-run artifacts are under
`/tmp/wave_migraphx_b2_fp16_stable_aligned_ref_v1/`,
`/tmp/wave_migraphx_b2_fp16_stable_cvnearest_aligned_v1/`,
`/tmp/wave_migraphx_b2_fp16_stable_cvnearest_heif8_v1/`,
`/tmp/wave_migraphx_b2_fp16_stable_cvnearest_heif24_v1/`, and
`/tmp/wave_migraphx_b2_fp16_stable_cvnearest_heif24_train_v1/`. The current
bootstrap is `/tmp/wave_migraphx_b2_cvnearest_heif24_quality_bootstrap_v1.json`;
head-adaptation search is under `/tmp/wave_migraphx_fp16_head_adaptation_v1/`.

### Final bounded raw-to-score experiment round and stop gate (2026-10-29)

I compared OpenCV's direct RGB JPEG decode (`cv2.IMREAD_COLOR_RGB`) with the
Pillow decode path. All ten JPEGs in `step-0-raw-data` decoded to identical
RGB bytes and identical dimensions. In the full 93-photo stable MIGraphX run,
all Step 2 JPEGs and scores also matched the Pillow route exactly. The slow
JPEG `IMG_7565.jpeg` decode fell from 38.14 ms to 30.22 ms, but total latency
was still 106.12 ms for that image and another validation image measured
100.76 ms. The route's medians were 93.13 ms validation / 92.90 ms test, p95
98.10 / 96.42 ms, with 91/93 requests under 100 ms. Thus this exact decode
change did not eliminate all tail misses.

| Final route | Waviness MAE, validation / test | Total raw-to-output latency, validation median / p95 | Total raw-to-output latency, test median / p95 | Under 100 ms |
|---|---:|---:|---:|---:|
| Stable MIGraphX FP16, OpenCV exact-nearest, 24 HEIF workers, Pillow JPEG decode | 0.131969 / 0.104235 | 92.76 / 97.93 ms | 92.34 / 96.02 ms | 92/93 |
| Same route with OpenCV RGB JPEG decode | 0.131969 / 0.104235 | 93.13 / 98.10 ms | 92.90 / 96.42 ms | 91/93 |
| Same route with OpenCV RGB JPEG decode and one OpenCV thread | 0.131969 / 0.104235 | 97.22 / 101.67 ms | 96.90 / 108.11 ms | 81/93 |

The one-thread route also preserved all scores but had worse tail behavior. I
will keep the measured four-thread setting for the best candidate; neither
OpenCV JPEG decode nor one-thread OpenCV is an overall latency improvement.
Run artifacts are `/tmp/wave_migraphx_b2_fp16_stable_cvnearest_heif24_cvjpeg_v1/`
and `/tmp/wave_migraphx_b2_fp16_stable_cvnearest_heif24_cvjpeg_cv1_v1/`.

#### Stop condition for this experiment campaign

The campaign has reached its bounded stop point after the full-cohort JPEG
and CPU-thread checks. I will count a route as successful only when two full
raw-to-score runs on all 93 labeled photos each meet all of these gates:

1. Every request completes in under 100 ms, with no failed photos.
2. Validation and test MAE are each no more than 1% worse than current v18.
3. The 95% capture-date-cluster bootstrap upper bound is also within the 1%
   MAE regression allowance on both splits, and the date-disjoint check does
   not show a material quality drop.
4. The experiment uses both segmentation passes, full-resolution raw input,
   required JPEG round trips, and the final prediction-record write.

The best measured route currently meets the MAE point-estimate gate and reaches
sub-100-ms median and p95, but misses the all-requests-under-100-ms gate by one
validation photo and does not pass the bootstrap quality gate. The current
PyTorch v18 route remains selected. No further optimization experiment is
planned in this campaign; restart only after new independently labeled dates
are available or the user chooses a specific candidate for focused follow-up.

### Six-photo B2-distilled B0 MIGraphX FP32 prediction check (2026-10-30)

Ran the candidate sequentially on the six HEIC photos in
`step-7-predict-captures-holder/experiment-predict-holder`, loading two
MIGraphX FP32 segmentation sessions and the existing v18 regressor once. The
run included both full-resolution preprocessing passes, JPEG round trips, and
score generation. It made one first pass in filename order followed by one
warm repeat of the same photos; there was no randomized order or concurrency.
No waviness labels were supplied, so MAE is unavailable for these six photos.

The process took 19.93 s from Python launch through exit. Session setup took
4.608 s for the Step 1 ONNX session, 4.505 s for Step 2, and 0.014 s for the
regressor. The first photo then incurred first-execution MIGraphX costs of
4.452 s (Step 1) and 4.509 s (Step 2), for 9.033 s total raw-to-score. The
remaining first-pass photos took 89.50–101.67 ms; warm repeats took 92.07–
96.32 ms (94.41 ms median). The complete 12-request sum was 10.067 s,
excluding process startup/session setup; process wall time includes those.

| Photo | Predicted waviness score | First pass, raw-to-score | Warm repeat, raw-to-score |
|---|---:|---:|---:|
| IMG_7927.HEIC | 0.79608029 | 9032.51 ms | 94.26 ms |
| IMG_7928.HEIC | 0.60329902 | 89.50 ms | 95.81 ms |
| IMG_7929.HEIC | 0.31018612 | 92.32 ms | 92.94 ms |
| IMG_7931.HEIC | 0.32639742 | 91.54 ms | 96.32 ms |
| IMG_7932.HEIC | 0.31529391 | 93.39 ms | 92.07 ms |
| IMG_7933.HEIC | 0.24815047 | 101.67 ms | 94.55 ms |

Warm median stage times were decode 16.62 ms, Step 1 total 51.75 ms, Step 1
JPEG round trip 0.49 ms, Step 2 total 17.41 ms, and Step 2 JPEG plus regression
6.34 ms. Step 1 and Step 2 totals include their respective model calls and
preprocessing; listed substage timings are diagnostic and should not be added
to those totals. The exact Step 1 and Step 2 JPEGs are saved under the holder's
`preprocessed/step-1/` and `preprocessed/step-2/` folders. The detailed
per-photo timing report is in
`step-7-predict-captures-holder/experiment-predict-holder/benchmark-results/`.

### Single-process experimental inference server (2026-10-30)

Added `step_7_i_experimental_migraphx_server.py` for the distilled-B0
MIGraphX FP32 candidate. It creates the Step 1 and Step 2 ONNX Runtime
MIGraphX sessions and the v18 EfficientNet-B0 regressor once, runs shape
warm-ups for all three, then listens on `127.0.0.1:8765`. It uses Python's
single-threaded `HTTPServer`, so requests are handled sequentially and do not
start extra copies of the models. `GET /health` reports readiness;
`POST /predict` accepts raw image bytes with an `X-Filename` header and returns
the waviness score plus server-side stage timings. The server was started and
reached ready state after its 18.262-second model setup and warm-up. No photos
were sent to the server in this setup step; this is not a prediction benchmark.

Start it with the isolated MIGraphX environment:

```bash
LD_LIBRARY_PATH=/tmp/wave_migraphx_312_env/lib/python3.12/site-packages/migraphx_libs:/opt/rocm/lib \
MIGRAPHX_SKIP_BENCHMARKING=1 MIGRAPHX_USE_FAST_SOFTMAX=0 \
/tmp/wave_migraphx_312_env/bin/python -u step_7_i_experimental_migraphx_server.py
```

Send one photo later with:

```bash
curl -X POST http://127.0.0.1:8765/predict \
  -H 'X-Filename: photo.HEIC' --data-binary @photo.HEIC
```

The server is experimental, local-only, and depends on the ONNX graphs and
isolated MIGraphX environment under `/tmp`; it does not change `.env` or the
production watcher.

### Six sequential requests through the warmed server (2026-10-30)

After the server reached ready state, I sent the six holder photos one at a
time in filename order. The server stayed alive and kept all sessions loaded;
there were no concurrent or randomized requests. Client time starts before
reading each local photo and ends after the full HTTP response body arrives.
Server request time includes receiving the upload and writing its temporary
file, through inference, but stops before sending the response. Raw-to-score
time starts after the upload is staged and includes image decode, both
preprocessing passes, and regression. MAE is unavailable because these photos
have no labels.

| Photo | Prediction | Client request-to-response | Server request time | Server raw-to-score |
|---|---:|---:|---:|---:|
| IMG_7927.HEIC | 0.79608029 | 131.34 ms | 124.34 ms | 120.68 ms |
| IMG_7928.HEIC | 0.60329902 | 104.59 ms | 103.23 ms | 99.49 ms |
| IMG_7929.HEIC | 0.31018612 | 115.18 ms | 114.01 ms | 109.23 ms |
| IMG_7931.HEIC | 0.32639742 | 107.91 ms | 106.83 ms | 100.96 ms |
| IMG_7932.HEIC | 0.31529391 | 101.72 ms | 100.57 ms | 95.77 ms |
| IMG_7933.HEIC | 0.24815047 | 103.61 ms | 102.34 ms | 97.28 ms |

Client request median was 106.25 ms (all six exceeded 100 ms); median server
request time was 105.03 ms (also 0/6 under 100 ms). Median server raw-to-score
was 100.23 ms, with 3/6 under 100 ms. Warm median stage timings were decode
19.89 ms, Step 1 total 55.40 ms, Step 1 JPEG round trip 0.54 ms, Step 2 total
17.82 ms, and final JPEG plus regression 7.80 ms. Within Step 1, tensor prep,
model call, and mask expansion medians were 9.46, 20.36, and 5.47 ms. Within
Step 2, they were 1.56, 13.45, and effectively 0 ms. Substage times are
included in their step totals. Predictions exactly match the prior six-photo
in-process run. Detailed per-request stage records are in
`step-7-predict-captures-holder/experiment-predict-holder/benchmark-results/b2-distilled-b0-migraphx-fp32-warm-server-requests-20261030.json`.
