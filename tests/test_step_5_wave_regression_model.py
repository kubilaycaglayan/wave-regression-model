from __future__ import annotations

from pathlib import Path

import torch

from step_5_a_wave_regression_model import (
    WaveRegressionModel,
    canonical_backbone_name,
    infer_backbone_from_checkpoint,
)
from step_5_b_train import version_directory_name


def test_both_supported_backbones_have_only_a_trainable_regression_head() -> None:
    for backbone_name, expected_feature_dim in (("resnet18", 512), ("efficientnet_b0", 1280)):
        model = WaveRegressionModel(backbone_name=backbone_name, pretrained=False)
        model.train()

        frozen, trainable, batch_norms = model.frozen_backbone_mode_checks()
        prediction = model(torch.zeros(2, 3, 224, 224))

        assert model.feature_dim == expected_feature_dim
        assert frozen > 0
        assert trainable == expected_feature_dim + 1
        assert batch_norms > 0
        assert prediction.shape == (2,)
        assert torch.all((prediction >= 0.0) & (prediction <= 1.0))


def test_backbone_names_are_canonicalized_and_validated() -> None:
    assert canonical_backbone_name("EfficientNet-B0") == "efficientnet_b0"
    assert canonical_backbone_name("resnet_18") == "resnet18"

    try:
        canonical_backbone_name("unknown")
    except ValueError as error:
        assert "Unsupported backbone" in str(error)
    else:  # pragma: no cover
        raise AssertionError("Unsupported backbone should fail")


def test_checkpoint_metadata_identifies_backbone(tmp_path: Path) -> None:
    checkpoint = tmp_path / "efficientnet.ckpt"
    torch.save({"hyper_parameters": {"backbone_name": "efficientnet_b0"}}, checkpoint)

    assert infer_backbone_from_checkpoint(checkpoint) == "efficientnet_b0"


def test_version_directory_name_ends_with_canonical_backbone() -> None:
    assert version_directory_name("v12", 0.151072, "EfficientNet-B0") == "v12-mae-0.1511-efficientnet_b0"
