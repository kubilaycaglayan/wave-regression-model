"""Capture and visualize versioned training history for regression runs."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback


# These names mirror the loss metrics logged by WaveRegressionModel. If the
# training loss implementation or log names change, update this definition.
LOSS_METRICS = ("train_loss", "val_loss")


@dataclass(frozen=True)
class PlotConfig:
    """User-adjustable defaults for the generated training plots."""

    figure_size_inches: tuple[float, float] = (9.0, 5.5)
    dpi: int = 150
    y_axis_min: float = 0.0
    y_axis_max: float = 1.0
    max_epoch_ticks: int = 12
    y_axis_as_percentage: bool = False


DEFAULT_PLOT_CONFIG = PlotConfig()


def performance_metrics_from_monitor(
    monitor: str,
    available_metrics: set[str] | None = None,
) -> tuple[str, str]:
    """Derive train/validation metric names from a validation monitor name."""
    if not monitor.startswith("val_") or len(monitor) <= len("val_"):
        raise ValueError(
            f"Performance monitor must be a validation metric named 'val_<metric>'; received {monitor!r}"
        )
    validation_metric = monitor
    training_metric = f"train_{monitor.removeprefix('val_')}"
    if available_metrics is not None:
        missing = [name for name in (training_metric, validation_metric) if name not in available_metrics]
        if missing:
            raise RuntimeError(
                f"The configured performance monitor {monitor!r} requires missing history metrics: {', '.join(missing)}"
            )
    return training_metric, validation_metric


class TrainingHistoryCallback(Callback):
    """Capture one complete metric row after each completed validation epoch."""

    def __init__(self, required_metrics: tuple[str, ...]) -> None:
        self.required_metrics = required_metrics
        self.training_metrics = tuple(name for name in required_metrics if name.startswith("train_"))
        self.validation_metrics = tuple(name for name in required_metrics if name.startswith("val_"))
        self.rows: list[dict[str, float | int]] = []

    @staticmethod
    def _read_metrics(
        metrics: dict[str, object],
        names: tuple[str, ...],
        epoch: int,
    ) -> dict[str, float]:
        missing = [name for name in names if name not in metrics]
        if missing:
            raise RuntimeError(
                f"Training history is missing metrics after epoch {epoch}: {', '.join(missing)}"
            )
        values: dict[str, float] = {}
        for name in names:
            value = metrics[name]
            if hasattr(value, "detach"):
                value = value.detach().cpu().item()
            value = float(value)
            if not math.isfinite(value):
                raise RuntimeError(f"Training history metric {name} is not finite at epoch {epoch}")
            values[name] = value
        return values

    def on_train_epoch_end(self, trainer: pl.Trainer, module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        if not hasattr(self, "_validation_row"):
            raise RuntimeError(
                f"Training history has no validation metrics before training epoch {trainer.current_epoch + 1}"
            )
        row = {
            **self._validation_row,
            **self._read_metrics(trainer.callback_metrics, self.training_metrics, trainer.current_epoch + 1),
        }
        if row["epoch"] != trainer.current_epoch + 1:
            raise RuntimeError("Training and validation history epochs do not match")
        self.rows.append(row)
        del self._validation_row

    def on_validation_epoch_end(self, trainer: pl.Trainer, module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        self._validation_row = {
            "epoch": trainer.current_epoch + 1,
            **self._read_metrics(
                trainer.callback_metrics,
                self.validation_metrics,
                trainer.current_epoch + 1,
            ),
        }


def write_training_history(history: TrainingHistoryCallback, run_version: str, output_dir: Path) -> Path:
    if not history.rows:
        raise RuntimeError("Training produced no completed epochs for the history artifact")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"training_history_{run_version}.csv"
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=["epoch", *history.required_metrics])
        writer.writeheader()
        writer.writerows(history.rows)
    return path


def plot_training_history(
    history: TrainingHistoryCallback,
    run_version: str,
    output_dir: Path,
    performance_monitor: str,
    early_stopping_monitor: str,
    config: PlotConfig = DEFAULT_PLOT_CONFIG,
) -> dict[str, Path]:
    """Save loss and monitor-selected performance plots from captured history."""
    if performance_monitor != early_stopping_monitor:
        raise RuntimeError(
            "Checkpoint and early-stopping monitor differ; refusing to plot an ambiguous performance metric: "
            f"checkpoint={performance_monitor!r}, early_stopping={early_stopping_monitor!r}"
        )
    training_metric, validation_metric = performance_metrics_from_monitor(
        performance_monitor,
        set(history.required_metrics),
    )
    if not history.rows:
        raise RuntimeError("Cannot plot an empty training history")
    epochs = [int(row["epoch"]) for row in history.rows]
    if epochs != list(range(1, len(epochs) + 1)):
        raise RuntimeError(f"Training history epochs are incomplete or out of order: {epochs}")
    if config.y_axis_min >= config.y_axis_max:
        raise ValueError("PlotConfig.y_axis_min must be smaller than y_axis_max")
    if config.max_epoch_ticks < 1:
        raise ValueError("PlotConfig.max_epoch_ticks must be at least 1")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator, PercentFormatter
    except ImportError as error:
        raise RuntimeError("Plot generation requires matplotlib; install dependencies from requirements.txt") from error

    def save_plot(
        filename: str,
        title: str,
        ylabel: str,
        train_key: str,
        validation_key: str,
    ) -> Path:
        path = output_dir / filename
        figure, axis = plt.subplots(figsize=config.figure_size_inches, constrained_layout=True)
        try:
            axis.plot(epochs, [row[train_key] for row in history.rows], marker="o", label=f"Training {ylabel}")
            axis.plot(
                epochs,
                [row[validation_key] for row in history.rows],
                marker="o",
                label=f"Validation {ylabel}",
            )
            axis.set_title(f"{title} ({run_version})")
            axis.set_xlabel("Epoch")
            axis.set_ylabel(ylabel)
            axis.set_xlim(left=1)
            axis.set_ylim(config.y_axis_min, config.y_axis_max)
            axis.xaxis.set_major_locator(MaxNLocator(nbins=config.max_epoch_ticks, integer=True))
            if config.y_axis_as_percentage:
                axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
            axis.grid(True, alpha=0.3)
            axis.legend()
            figure.savefig(path, dpi=config.dpi, format="png")
        finally:
            plt.close(figure)
        return path

    return {
        "loss": save_plot(
            f"training_loss_{run_version}.png",
            "Training and validation loss by epoch",
            "SmoothL1 loss",
            *LOSS_METRICS,
        ),
        "performance": save_plot(
            f"training_{performance_monitor}_{run_version}.png",
            f"Training and validation {performance_monitor.removeprefix('val_').upper()} by epoch",
            f"{performance_monitor.removeprefix('val_').upper()} (lower is better)",
            training_metric,
            validation_metric,
        ),
    }
