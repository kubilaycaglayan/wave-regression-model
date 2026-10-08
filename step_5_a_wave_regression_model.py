"""Configurable frozen-backbone wave-regression model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision_compat import ensure_torchvision_operator_schemas

ensure_torchvision_operator_schemas()

import lightning.pytorch as pl
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    resnet18,
)
from torchmetrics import MeanAbsoluteError, MeanSquaredError


SUPPORTED_BACKBONES = ("resnet18", "efficientnet_b0")
BACKBONE_ALIASES = {
    "resnet_18": "resnet18",
    "efficientnetb0": "efficientnet_b0",
}


def canonical_backbone_name(name: str) -> str:
    """Return the canonical name for a supported backbone."""
    value = name.strip().lower().replace("-", "_")
    value = BACKBONE_ALIASES.get(value, value)
    if value not in SUPPORTED_BACKBONES:
        raise ValueError(f"Unsupported backbone {name!r}; expected one of {SUPPORTED_BACKBONES}")
    return value


def build_feature_extractor(
    backbone_name: str,
    pretrained: bool = True,
) -> tuple[nn.Module, int, str]:
    """Build an ImageNet feature extractor and describe its head input."""
    name = canonical_backbone_name(backbone_name)
    if name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        feature_dim = model.fc.in_features
        weights_name = "ResNet18_Weights.DEFAULT (ImageNet)"
    elif name == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        feature_dim = model.classifier[-1].in_features
        weights_name = "EfficientNet_B0_Weights.DEFAULT (ImageNet)"
    else:  # pragma: no cover - canonical_backbone_name guards this branch.
        raise AssertionError(f"Unhandled backbone: {name}")
    return model, feature_dim, weights_name


def infer_backbone_from_checkpoint(checkpoint_path: Path) -> str:
    """Infer a checkpoint's backbone, failing rather than guessing."""
    checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hyper_parameters = checkpoint.get("hyper_parameters", {})
    if isinstance(hyper_parameters, dict):
        configured = hyper_parameters.get("backbone_name") or hyper_parameters.get("backbone")
        if isinstance(configured, str):
            return canonical_backbone_name(configured)

    state_dict = checkpoint.get("state_dict", {})
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint has no usable state_dict: {checkpoint_path}")
    keys = tuple(str(key) for key in state_dict)
    if any(key.startswith("backbone.fc.") for key in keys):
        return "resnet18"
    if any(key.startswith("regression_head.") for key in keys) and any(
        key.startswith("backbone.features.") for key in keys
    ):
        return "efficientnet_b0"
    raise ValueError(f"Cannot infer backbone from checkpoint: {checkpoint_path}")


class WaveRegressionModel(pl.LightningModule):
    """Frozen ImageNet features with a trainable one-neuron sigmoid head.

    The default constructor remains ResNet18 for compatibility with legacy
    checkpoints. New training entry points pass their selected backbone
    explicitly, with EfficientNet-B0 as the current pipeline default.
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        name = canonical_backbone_name(backbone_name)
        self.save_hyperparameters()
        self.backbone_name = name
        self.backbone, feature_dim, self.weights_name = build_feature_extractor(name, pretrained=pretrained)
        self.feature_dim = feature_dim

        # Keep the ResNet18 classifier location unchanged so old Step 5
        # checkpoints with backbone.fc.* keys remain loadable.
        if name == "resnet18":
            self.backbone.fc = nn.Sequential(nn.Linear(feature_dim, 1), nn.Sigmoid())
        else:
            self.backbone.classifier = nn.Identity()
            self.regression_head = nn.Sequential(nn.Linear(feature_dim, 1), nn.Sigmoid())

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for parameter in self._head().parameters():
            parameter.requires_grad = True

        self.loss_fn = nn.SmoothL1Loss()
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.val_rmse = MeanSquaredError(squared=False)

    def _head(self) -> nn.Module:
        return self.backbone.fc if self.backbone_name == "resnet18" else self.regression_head

    def train(self, mode: bool = True) -> "WaveRegressionModel":
        """Keep frozen backbone layers, especially BatchNorm, in evaluation mode."""
        super().train(mode)
        self.backbone.eval()
        self._head().train(mode)
        return self

    def frozen_backbone_mode_checks(self) -> tuple[int, int, int]:
        """Return frozen parameter and BatchNorm counts, raising on mode drift."""
        head = self._head()
        head_parameter_ids = {id(parameter) for parameter in head.parameters()}
        frozen_parameters = [
            parameter for parameter in self.backbone.parameters() if id(parameter) not in head_parameter_ids
        ]
        head_parameters = list(head.parameters())
        batch_norm_modules = [
            module for module in self.backbone.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)
        ]
        if any(parameter.requires_grad for parameter in frozen_parameters):
            raise RuntimeError("A frozen backbone parameter unexpectedly requires gradients")
        if any(not parameter.requires_grad for parameter in head_parameters):
            raise RuntimeError("A regression-head parameter unexpectedly does not require gradients")
        if any(module.training for module in batch_norm_modules):
            raise RuntimeError("A frozen backbone BatchNorm module is in train mode")
        if not head.training:
            raise RuntimeError("The regression head is not in train mode")
        return (
            sum(parameter.numel() for parameter in frozen_parameters),
            sum(parameter.numel() for parameter in head_parameters),
            len(batch_norm_modules),
        )

    def on_train_start(self) -> None:
        """Verify the invariant after Lightning has entered training mode."""
        frozen_backbone, head_parameters, batch_norm_modules = self.frozen_backbone_mode_checks()
        print(
            "After Lightning entered training mode: "
            f"Backbone: {self.backbone_name}; "
            f"Frozen backbone parameters: {frozen_backbone:,}; "
            f"Trainable head parameters: {head_parameters:,}; "
            f"Frozen BatchNorm modules: {batch_norm_modules}; "
            "BatchNorm modules in train mode: 0; "
            f"Regression head training mode: {self._head().training}"
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return one prediction in [0, 1] for each image in the batch."""
        features = self.backbone(images)
        if self.backbone_name == "resnet18":
            return features.squeeze(-1)
        return self.regression_head(features).squeeze(-1)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images, targets = batch
        predictions = self(images)
        loss = self.loss_fn(predictions, targets)
        self.train_mae.update(predictions, targets)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=images.size(0))
        self.log("train_mae", self.train_mae, on_step=False, on_epoch=True, prog_bar=False, batch_size=images.size(0))
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
