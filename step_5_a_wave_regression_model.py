"""Frozen-backbone ResNet18 baseline for sea-waviness regression."""

from __future__ import annotations

import torch
from torch import nn

# Import first so the repository's torchvision compatibility definition is
# installed before torchvision is imported by Lightning/torchmetrics.
import step_4_a_wave_dataset  # noqa: F401

import lightning.pytorch as pl
from torchvision.models import ResNet18_Weights, resnet18
from torchmetrics import MeanAbsoluteError, MeanSquaredError


class WaveRegressionModel(pl.LightningModule):
    """ResNet18 ImageNet features with a trainable one-neuron sigmoid head."""

    def __init__(self, learning_rate: float = 1e-3) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        feature_count = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(feature_count, 1),
            nn.Sigmoid(),
        )
        self.loss_fn = nn.SmoothL1Loss()
        self.val_mae = MeanAbsoluteError()
        self.val_rmse = MeanSquaredError(squared=False)

    def train(self, mode: bool = True) -> WaveRegressionModel:
        """Keep frozen ResNet layers, especially BatchNorm, in evaluation mode."""
        super().train(mode)
        self.backbone.eval()
        self.backbone.fc.train(mode)
        return self

    def frozen_backbone_mode_checks(self) -> tuple[int, int, int]:
        """Return frozen parameter and BatchNorm counts, raising on mode drift."""
        frozen_parameters = [
            parameter for name, parameter in self.backbone.named_parameters() if not name.startswith("fc.")
        ]
        head_parameters = list(self.backbone.fc.parameters())
        batch_norm_modules = [
            module for name, module in self.backbone.named_modules()
            if not name.startswith("fc") and isinstance(module, nn.modules.batchnorm._BatchNorm)
        ]
        if any(parameter.requires_grad for parameter in frozen_parameters):
            raise RuntimeError("A frozen backbone parameter unexpectedly requires gradients")
        if any(not parameter.requires_grad for parameter in head_parameters):
            raise RuntimeError("A regression-head parameter unexpectedly does not require gradients")
        if any(module.training for module in batch_norm_modules):
            raise RuntimeError("A frozen backbone BatchNorm module is in train mode")
        if not self.backbone.fc.training:
            raise RuntimeError("The regression head is not in train mode")
        return (
            sum(parameter.numel() for parameter in frozen_parameters),
            sum(parameter.numel() for parameter in head_parameters),
            len(batch_norm_modules),
        )

    def on_train_start(self) -> None:
        """Verify the invariant after Lightning has entered training mode."""
        frozen_parameters, head_parameters, batch_norm_modules = self.frozen_backbone_mode_checks()
        print(
            "After Lightning entered training mode: "
            f"Frozen backbone parameters: {frozen_parameters:,}; "
            f"Trainable head parameters: {head_parameters:,}; "
            f"Frozen BatchNorm modules: {batch_norm_modules}; "
            "BatchNorm modules in train mode: 0; "
            f"Regression head training mode: {self.backbone.fc.training}"
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return one prediction in [0, 1] for each image in the batch."""
        return self.backbone(images).squeeze(-1)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images, targets = batch
        predictions = self(images)
        loss = self.loss_fn(predictions, targets)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=images.size(0))
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        images, targets = batch
        predictions = self(images)
        loss = self.loss_fn(predictions, targets)
        self.val_mae.update(predictions, targets)
        self.val_rmse.update(predictions, targets)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=images.size(0))
        self.log("val_mae", self.val_mae, on_step=False, on_epoch=True, prog_bar=True, batch_size=images.size(0))
        self.log("val_rmse", self.val_rmse, on_step=False, on_epoch=True, prog_bar=False, batch_size=images.size(0))

    def configure_optimizers(self) -> torch.optim.Optimizer:
        trainable_parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        return torch.optim.AdamW(trainable_parameters, lr=self.hparams.learning_rate)
