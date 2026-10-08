"""Run segmentation tasks in parallel worker processes sized to the machine.

Each worker process loads its own segmentation model and handles one task
(image) at a time. Peak memory per worker was measured on 12 MP iPhone HEIC
photos (GTX 1050 4 GB, 30 GB RAM): ~1.6 GB GPU and ~9 GB RAM, the RAM being the
full-resolution logit accumulator. Measured Step 1 speed with --workers:
  CUDA 1 worker: 49.8 s/image, 2 workers: 30.6 s/image, 3 workers: GPU OOM
  CPU  1 worker: 127 s/image,  2 workers: 122 s/image (cores already saturated)
so automatic selection only parallelises on CUDA, limited by free memory.
"""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Callable, Hashable, Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import torch

from step_1_a_water_segmentation import MODEL_NAME, SegmentationModel, load_model

# A 12 MP, 150-class float32 logit accumulator needs about 6.7 GiB when it
# fits on CUDA, in addition to model weights and tile activations. Keep worker
# sizing conservative; smaller GPUs use the CPU accumulator automatically.
WORKER_GPU_MEMORY_GB = 9.0
WORKER_RAM_GB = 9.0

# A task function receives the worker's model followed by the task arguments.
# It must be a module-level function so it can be sent to spawned workers.
TaskFunction = Callable[..., Any]


def available_ram_gb() -> float | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024**2
    except OSError:
        pass
    return None


def automatic_worker_count(device: torch.device, pending: int) -> tuple[int, str]:
    """Pick a worker count from free memory; parallel CPU runs were not faster."""
    if pending <= 1:
        return 1, "one pending image"
    if device.type != "cuda":
        return 1, "CPU inference already uses every core"
    gpu_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3
    ram_gb = available_ram_gb()
    by_gpu = int(gpu_gb // WORKER_GPU_MEMORY_GB)
    by_ram = int(ram_gb // WORKER_RAM_GB) if ram_gb is not None else 1
    workers = max(1, min(by_gpu, by_ram, pending))
    ram_text = f"{ram_gb:.1f} GB" if ram_gb is not None else "unknown"
    return workers, (
        f"GPU {gpu_gb:.1f} GB / {WORKER_GPU_MEMORY_GB} GB -> {by_gpu}, "
        f"available RAM {ram_text} / {WORKER_RAM_GB} GB -> {by_ram}, pending {pending}"
    )


def select_worker_count(requested: int | None, device: torch.device, pending: int) -> int:
    """Use the requested count (capped by pending work) or choose one automatically."""
    if requested is None:
        workers, reason = automatic_worker_count(device, pending)
        print(f"Workers: {workers} (auto: {reason})")
    else:
        workers = min(requested, pending)
        print(f"Workers: {workers} (--workers {requested})")
    return workers


_worker_model: SegmentationModel | None = None


def _initialize_worker(device: str | None, torch_threads: int) -> None:
    global _worker_model
    torch.set_num_threads(torch_threads)
    _worker_model = load_model(device, verbose=False)


def _run_in_worker(task_function: TaskFunction, *task_args: Any) -> Any:
    return task_function(_worker_model, *task_args)


def run_segmentation_tasks(
    task_function: TaskFunction,
    tasks: Mapping[Hashable, tuple[Any, ...]],
    device: str | None,
    workers: int,
) -> Iterator[tuple[Hashable, Any]]:
    """Yield (task key, result) as tasks finish; order follows completion with workers > 1."""
    if workers == 1:
        model = load_model(device)
        for key, task_args in tasks.items():
            yield key, task_function(model, *task_args)
        return

    torch_threads = max(1, (os.cpu_count() or workers) // workers)
    print(f"Model: {MODEL_NAME} on {device}; torch CPU threads per worker: {torch_threads}")
    with ProcessPoolExecutor(
        workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_worker,
        initargs=(device, torch_threads),
    ) as pool:
        futures = {
            pool.submit(_run_in_worker, task_function, *task_args): key
            for key, task_args in tasks.items()
        }
        for future in as_completed(futures):
            yield futures[future], future.result()
