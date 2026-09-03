"""Train the first frozen-backbone sea-waviness regression baseline."""

from __future__ import annotations

import time
from pathlib import Path

# Install the repository's torchvision compatibility definition before
# Lightning imports torchmetrics (which imports torchvision transitively).
import step_4_a_wave_dataset  # noqa: F401

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from step_4_b_wave_datamodule import WaveDataModule
from step_5_a_wave_regression_model import WaveRegressionModel


RANDOM_SEED = 42
LEARNING_RATE = 1e-3
BATCH_SIZE = 8
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
CHECKPOINT_DIR = Path("step-5-checkpoints")


def parameter_counts(model: torch.nn.Module) -> tuple[int, int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable, total - trainable


def make_cuda_usable_if_needed() -> None:
    """Work around CUDA/cuDNN installations without a usable convolution engine."""
    if not torch.cuda.is_available():
        return
    try:
        probe = torch.nn.Conv2d(3, 4, kernel_size=3).cuda()
        probe(torch.zeros(1, 3, 16, 16, device="cuda"))
    except RuntimeError:
        torch.backends.cudnn.enabled = False
        torch.cuda.empty_cache()
        print("CUDA convolution probe failed; disabled cuDNN backend for this run")


def write_summary(
    checkpoint: ModelCheckpoint,
    early_stopping: EarlyStopping,
    trainer: pl.Trainer,
    model: WaveRegressionModel,
    best_metrics: dict[str, float],
) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_mae = best_metrics["val_mae"]
    best_rmse = best_metrics["val_rmse"]
    summary_path = CHECKPOINT_DIR / "training_summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                "Sea-waviness regression baseline",
                "architecture: torchvision ResNet18; backbone features -> Linear(512, 1) -> Sigmoid",
                "pretrained weights: ResNet18_Weights.DEFAULT (ImageNet)",
                "backbone status: frozen; regression head status: trainable",
                "loss: SmoothL1Loss",
                "optimizer: AdamW (trainable parameters only)",
                f"learning rate: {LEARNING_RATE}",
                f"batch size: {BATCH_SIZE}",
                f"max epochs: {MAX_EPOCHS}",
                f"early stopping: monitor=val_mae, mode=min, patience={EARLY_STOPPING_PATIENCE}",
                f"best validation MAE: {float(best_mae) if best_mae is not None else 'unavailable'}",
                f"best validation RMSE: {float(best_rmse) if best_rmse is not None else 'unavailable'}",
                f"best checkpoint path: {checkpoint.best_model_path}",
                f"epochs actually trained: {trainer.current_epoch + 1}",
                f"early stopping triggered: {early_stopping.stopped_epoch > 0}",
                f"total parameters: {parameter_counts(model)[0]}",
                f"trainable parameters: {parameter_counts(model)[1]}",
                f"frozen parameters: {parameter_counts(model)[2]}",
                "test split evaluated: no",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return summary_path


def main() -> None:
    started = time.perf_counter()
    pl.seed_everything(RANDOM_SEED, workers=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using accelerator: {device}")
    make_cuda_usable_if_needed()

    data = WaveDataModule(batch_size=BATCH_SIZE, num_workers=0)
    model = WaveRegressionModel(learning_rate=LEARNING_RATE)
    total, trainable, frozen = parameter_counts(model)
    print(f"Parameters: total={total:,}, trainable={trainable:,}, frozen={frozen:,}")
    if trainable == 0 or frozen == 0:
        raise RuntimeError("Expected a trainable regression head and a frozen ResNet backbone")

    data.setup("fit")
    batch_images, _ = next(iter(data.train_dataloader()))
    with torch.inference_mode():
        sanity_predictions = model(batch_images)
    print(f"Sanity forward shape: {tuple(sanity_predictions.shape)} (expected [{batch_images.size(0)}])")
    if sanity_predictions.shape != (batch_images.size(0),):
        raise RuntimeError(f"Unexpected prediction shape: {sanity_predictions.shape}")
    if not torch.all((sanity_predictions >= 0.0) & (sanity_predictions <= 1.0)):
        raise RuntimeError("Sanity predictions are outside [0, 1]")
    print("Sanity prediction range: within [0.0, 1.0]")

    checkpoint = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="best-val-mae-{epoch:02d}-{val_mae:.4f}",
        monitor="val_mae",
        mode="min",
        save_top_k=1,
        save_last=False,
    )
    early_stopping = EarlyStopping(
        monitor="val_mae",
        mode="min",
        patience=EARLY_STOPPING_PATIENCE,
        verbose=True,
    )
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=MAX_EPOCHS,
        callbacks=[checkpoint, early_stopping],
        deterministic=True,
        log_every_n_steps=1,
        logger=False,
        enable_progress_bar=True,
    )
    trainer.fit(model, datamodule=data)

    best_model = WaveRegressionModel.load_from_checkpoint(checkpoint.best_model_path)
    best_metrics = trainer.validate(best_model, datamodule=data, verbose=False)[0]
    summary_path = write_summary(checkpoint, early_stopping, trainer, model, best_metrics)
    print(f"Best validation MAE: {best_metrics['val_mae']:.6f}")
    print(f"Best validation RMSE: {best_metrics['val_rmse']:.6f}")
    print(f"Best checkpoint path: {checkpoint.best_model_path}")
    print(f"Epochs actually trained: {trainer.current_epoch + 1}")
    print(f"Early stopping triggered: {early_stopping.stopped_epoch > 0}")
    print(f"Training summary: {summary_path}")
    print(f"Elapsed time: {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
