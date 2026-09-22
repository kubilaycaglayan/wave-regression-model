"""Regression tests for Lightning training-history hook ordering."""

from __future__ import annotations

from types import SimpleNamespace

import torch

# Install the repository's torchvision compatibility definition before Lightning
# is imported by the plotting module.
import step_4_a_wave_dataset  # noqa: F401

from step_5_c_plot_training import TrainingHistoryCallback


def test_history_combines_training_and_validation_metrics_across_hooks() -> None:
    callback = TrainingHistoryCallback(
        required_metrics=("train_loss", "val_loss", "train_mae", "val_mae")
    )
    trainer = SimpleNamespace(current_epoch=0, sanity_checking=False, callback_metrics={})

    trainer.callback_metrics = {
        "val_loss": torch.tensor(0.25),
        "val_mae": torch.tensor(0.35),
    }
    callback.on_validation_epoch_end(trainer, None)

    trainer.callback_metrics = {
        "train_loss": torch.tensor(0.20),
        "train_mae": torch.tensor(0.30),
    }
    callback.on_train_epoch_end(trainer, None)

    assert len(callback.rows) == 1
    row = callback.rows[0]
    assert row["epoch"] == 1
    for name, expected in {
        "train_loss": 0.20,
        "train_mae": 0.30,
        "val_loss": 0.25,
        "val_mae": 0.35,
    }.items():
        assert abs(row[name] - expected) < 1e-6
