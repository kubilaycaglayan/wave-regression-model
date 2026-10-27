from types import SimpleNamespace

import pytest
import torch

from step_7_g_checkpoint_contract import validate_checkpoint_preprocessing, validate_segmentation_revision


def test_adapted_checkpoint_requires_matching_pipeline(tmp_path):
    names = ("step_1_a_water_segmentation.py", "step_1_b_preprocess_water_inputs.py",
             "step_2_a_reduce_black_water_area.py", "step_4_a_wave_dataset.py", "step_5_a_wave_regression_model.py",
             "step_7_a_predict.py")
    contract = {"model": "segmenter", "resolution": 448, "step2_resolution": 384,
                "precision": "float16", "logits_resolution": 448, "device": "cuda",
                "torch_version": str(torch.__version__), "segmentation_model_revision": "revision",
                "code_sha256": {name: "same" for name in names}}
    checkpoint = tmp_path / "model.ckpt"
    torch.save({"fast_preprocessing_experiment": contract}, checkpoint)
    segment = SimpleNamespace(MODEL_NAME="segmenter", SEGMENTATION_RESOLUTION=448, INFERENCE_PRECISION="float16",
                              SINGLE_IMAGE_LOGITS_RESOLUTION=448, USE_OVERLAPPING_TILES=False)
    kwargs = {"segmentation": segment, "device": torch.device("cuda"), "step2_resolution": 384,
              "reuse_step1_mask": False, "project_dir": tmp_path, "sha256_for": lambda path: "same"}
    loaded = validate_checkpoint_preprocessing(checkpoint, **kwargs)
    validate_segmentation_revision(loaded, SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(_commit_hash="revision"))))
    with pytest.raises(RuntimeError, match="Step 2"):
        validate_checkpoint_preprocessing(checkpoint, **{**kwargs, "reuse_step1_mask": True})
    with pytest.raises(RuntimeError, match="step2_resolution"):
        validate_checkpoint_preprocessing(checkpoint, **{**kwargs, "step2_resolution": 768})
    with pytest.raises(RuntimeError, match="code changed"):
        validate_checkpoint_preprocessing(checkpoint, **{**kwargs, "sha256_for": lambda path: "different"})
    with pytest.raises(RuntimeError, match="segmentation revision"):
        validate_segmentation_revision(loaded, SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(_commit_hash="different"))))


def test_original_checkpoint_does_not_require_contract(tmp_path):
    checkpoint = tmp_path / "baseline.ckpt"
    torch.save({"state_dict": {}}, checkpoint)
    assert validate_checkpoint_preprocessing(checkpoint, segmentation=None, device=torch.device("cpu"),
                                             step2_resolution=768, reuse_step1_mask=False,
                                             project_dir=tmp_path, sha256_for=lambda path: "") is None
