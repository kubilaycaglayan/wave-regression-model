"""Capture and visualize versioned training history for regression runs."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

from torchvision_compat import ensure_torchvision_operator_schemas

ensure_torchvision_operator_schemas()

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback


# These names mirror the loss metrics logged by WaveRegressionModel. If the
# training loss implementation or log names change, update this definition.
LOSS_METRICS = ("train_loss", "val_loss")


@dataclass(frozen=True)
class PlotConfig:
    """User-adjustable defaults for the generated training plots."""

    figure_size_inches: tuple[float, float] = (14.0, 5.5)
    dpi: int = 150
    y_axis_min: float = 0.0
    y_axis_max: float = 1.0
    max_epoch_ticks: int = 12
    y_axis_as_percentage: bool = False
    auto_zoom_enabled: bool = True
    auto_zoom_margin_fraction: float = 0.15
    auto_zoom_minimum_span: float = 0.05


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
    labeled_sample_count: int,
    config: PlotConfig = DEFAULT_PLOT_CONFIG,
) -> Path:
    """Save one combined loss/performance figure from captured history."""
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
    if config.auto_zoom_margin_fraction < 0.0:
        raise ValueError("PlotConfig.auto_zoom_margin_fraction cannot be negative")
    if config.auto_zoom_minimum_span <= 0.0:
        raise ValueError("PlotConfig.auto_zoom_minimum_span must be positive")
    if labeled_sample_count <= 0:
        raise ValueError("labeled_sample_count must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator, PercentFormatter
    except ImportError as error:
        raise RuntimeError("Plot generation requires matplotlib; install dependencies from requirements.txt") from error

    train_loss_values = [row[LOSS_METRICS[0]] for row in history.rows]
    validation_loss_values = [row[LOSS_METRICS[1]] for row in history.rows]
    train_performance_values = [row[training_metric] for row in history.rows]
    validation_performance_values = [row[validation_metric] for row in history.rows]
    performance_name = performance_monitor.removeprefix("val_").upper()

    figure, axes = plt.subplots(2, 2, figsize=(14.0, 10.0), constrained_layout=False)
    try:
        figure.subplots_adjust(left=0.07, right=0.98, bottom=0.12, top=0.90, wspace=0.25, hspace=0.32)

        def render_panel(
            axis: object,
            train_values: list[float],
            validation_values: list[float],
            ylabel: str,
            panel_title: str,
        ) -> None:
            all_values = [*train_values, *validation_values]
            observed_min = min(all_values)
            observed_max = max(all_values)
            observed_span = observed_max - observed_min
            zoom_span = max(observed_span, config.auto_zoom_minimum_span)
            zoom_margin = zoom_span * config.auto_zoom_margin_fraction
            zoom_min = max(config.y_axis_min, observed_min - zoom_margin)
            zoom_max = min(config.y_axis_max, observed_max + zoom_margin)
            if zoom_min >= zoom_max:
                zoom_min, zoom_max = config.y_axis_min, config.y_axis_max
            is_zoom_panel = "Automatic zoom" in panel_title and config.auto_zoom_enabled
            y_min = zoom_min if is_zoom_panel else config.y_axis_min
            y_max = zoom_max if is_zoom_panel else config.y_axis_max

            axis.plot(epochs, train_values, marker="o", label=f"Training {ylabel}")
            axis.plot(epochs, validation_values, marker="o", label=f"Validation {ylabel}")
            axis.set_title(panel_title)
            axis.set_xlabel("Epoch")
            axis.set_ylabel(ylabel)
            axis.set_xlim(left=1)
            axis.set_ylim(y_min, y_max)
            axis.xaxis.set_major_locator(MaxNLocator(nbins=config.max_epoch_ticks, integer=True))
            if config.y_axis_as_percentage:
                axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
            axis.grid(True, alpha=0.3)
            axis.legend()

        render_panel(axes[0, 0], train_loss_values, validation_loss_values, "SmoothL1 loss", "Loss · Configured scale")
        render_panel(axes[0, 1], train_loss_values, validation_loss_values, "SmoothL1 loss", "Loss · Automatic zoom")
        render_panel(
            axes[1, 0],
            train_performance_values,
            validation_performance_values,
            f"{performance_name} (lower is better)",
            f"{performance_name} · Configured scale",
        )
        render_panel(
            axes[1, 1],
            train_performance_values,
            validation_performance_values,
            f"{performance_name} (lower is better)",
            f"{performance_name} · Automatic zoom",
        )
        figure.suptitle(f"Training metrics ({run_version})", fontsize=16)
        figure.text(
            0.5,
            0.04,
            f"Labeled samples used for training and validation: {labeled_sample_count:,}",
            ha="center",
            va="bottom",
        )
        path = output_dir / f"training_metrics_{run_version}.png"
        figure.savefig(path, dpi=config.dpi, format="png")
    finally:
        plt.close(figure)
    return path
