"""Guard experimental checkpoints against incompatible preprocessing."""
from __future__ import annotations

from pathlib import Path
import hashlib

import torch


def previous_runner_hash(path: Path) -> str | None:
    """Recognize the immediately previous runner when only contract checks were added."""
    text = path.read_text(encoding="utf-8")
    additions = (
        "import step_1_a_water_segmentation as segmentation\n",
        "from step_7_g_checkpoint_contract import validate_checkpoint_preprocessing, validate_segmentation_revision\n",
        "        contract = validate_checkpoint_preprocessing(\n"
        "            checkpoint_path, segmentation=segmentation, device=device,\n"
        "            step2_resolution=segmentation.SEGMENTATION_RESOLUTION, reuse_step1_mask=False,\n"
        "            project_dir=PROJECT_DIR, sha256_for=sha256_for,\n"
        "        )\n",
        "            validate_segmentation_revision(contract, segmentation_model)\n",
    )
    if any(text.count(part) != 1 for part in additions):
        return None
    for part in additions:
        text = text.replace(part, "", 1)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_checkpoint_preprocessing(checkpoint: Path, *, segmentation, device: torch.device,
                                      step2_resolution: int, reuse_step1_mask: bool, project_dir: Path,
                                      sha256_for):
    """Validate an adapted head's recorded settings before reusing previews or predicting."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    contract = payload.get("fast_preprocessing_experiment")
    if contract is None:
        return None
    if not isinstance(contract, dict):
        raise RuntimeError("Invalid experimental checkpoint preprocessing contract")
    actual = {"model": segmentation.MODEL_NAME, "resolution": segmentation.SEGMENTATION_RESOLUTION,
              "step2_resolution": step2_resolution, "precision": segmentation.INFERENCE_PRECISION,
              "logits_resolution": segmentation.SINGLE_IMAGE_LOGITS_RESOLUTION,
              "torch_version": str(torch.__version__)}
    mismatches = [f"{key}: checkpoint={contract.get(key)!r}, requested={value!r}"
                  for key, value in actual.items() if contract.get(key) != value]
    if segmentation.USE_OVERLAPPING_TILES:
        mismatches.append("checkpoint requires single-image segmentation")
    if reuse_step1_mask:
        mismatches.append("checkpoint requires a separate Step 2 segmentation")
    try:
        if torch.device(contract["device"]).type != device.type:
            mismatches.append(f"segmentation device: checkpoint={contract['device']!r}, requested={str(device)!r}")
    except (KeyError, TypeError, RuntimeError):
        mismatches.append("checkpoint has no valid segmentation device")
    if device.type != "cuda" and actual["precision"] != "float32":
        mismatches.append("requested reduced-precision segmentation is unavailable on CPU")
    recorded_hashes = contract.get("code_sha256", {})
    for name in ("step_1_a_water_segmentation.py", "step_1_b_preprocess_water_inputs.py",
                 "step_2_a_reduce_black_water_area.py", "step_4_a_wave_dataset.py", "step_5_a_wave_regression_model.py"):
        if recorded_hashes.get(name) != sha256_for(project_dir / name):
            mismatches.append(f"preprocessing/model code changed: {name}")
    runner = project_dir / "step_7_a_predict.py"
    expected_runner = recorded_hashes.get("step_7_a_predict.py")
    compatible_runner = previous_runner_hash(runner) if runner.is_file() else None
    if expected_runner not in (sha256_for(runner), compatible_runner) or expected_runner is None:
        mismatches.append("preprocessing/model code changed: step_7_a_predict.py")
    if mismatches:
        raise RuntimeError("Experimental checkpoint requires its training preprocessing: " + "; ".join(mismatches))
    return contract


def validate_segmentation_revision(contract, segmenter):
    if contract is None:
        return
    expected = contract.get("segmentation_model_revision")
    actual = getattr(segmenter.model.config, "_commit_hash", None)
    if not expected or actual != expected:
        raise RuntimeError(f"Experimental checkpoint requires segmentation revision {expected!r}, got {actual!r}")
