from types import SimpleNamespace

import torch
from PIL import Image
import pytest

import step_1_a_water_segmentation as segmentation


def test_interpolated_single_image_matches_one_tile_before_argmax(monkeypatch):
    logits = torch.tensor([[[[2., 0.], [0., 1.]], [[0., 1.], [2., 0.]]]])
    class FakeModel:
        config = SimpleNamespace(num_labels=2)
        def __call__(self, **kwargs):
            return SimpleNamespace(logits=logits)
    segmenter = segmentation.SegmentationModel(FakeModel(), lambda **kwargs: {"pixel_values": torch.zeros(1)},
                                               (1,), torch.device("cpu"))
    image = Image.new("RGB", (32, 24))
    monkeypatch.setattr(segmentation, "LOG_TILE_TIMING", False)
    monkeypatch.setattr(segmentation, "USE_OVERLAPPING_TILES", True)
    tiled = segmentation.run_segmentation(image, segmenter)
    monkeypatch.setattr(segmentation, "USE_OVERLAPPING_TILES", False)
    monkeypatch.setattr(segmentation, "SINGLE_IMAGE_LOGITS_RESOLUTION", 32)
    single = segmentation.run_segmentation(image, segmenter)
    assert torch.equal(single, tiled)


def test_tiled_segmentation_downscales_only_segmentation_and_returns_original_size(monkeypatch):
    calls = []
    def fake_tiled(image, model):
        calls.append(image.size)
        return torch.ones((image.height, image.width), dtype=torch.uint8)
    monkeypatch.setattr(segmentation, "MAX_SEGMENTATION_SIDE", 40)
    monkeypatch.setattr(segmentation, "USE_OVERLAPPING_TILES", True)
    monkeypatch.setattr(segmentation, "run_overlapping_segmentation", fake_tiled)
    result = segmentation.run_segmentation(Image.new("RGB", (60, 80)), None)
    assert calls == [(30, 40)]
    assert result.shape == (80, 60)
    assert torch.all(result == 1)


def test_experimental_precision_keeps_cpu_fallback_full_precision(monkeypatch):
    class FakeModel:
        def __call__(self, **values):
            return SimpleNamespace(logits=values["pixel_values"])
    model = SimpleNamespace(model=FakeModel(), device=torch.device("cpu"))
    monkeypatch.setattr(segmentation, "INFERENCE_PRECISION", "float16")
    monkeypatch.setattr(torch, "autocast", lambda **kwargs: (_ for _ in ()).throw(AssertionError("CPU autocast")))
    result = segmentation.model_logits(model, {"pixel_values": torch.ones(1)})
    assert result.dtype == torch.float32
    monkeypatch.setattr(segmentation, "INFERENCE_PRECISION", "unsupported")
    with pytest.raises(ValueError, match="Unsupported segmentation precision"):
        segmentation.model_logits(model, {"pixel_values": torch.ones(1)})
