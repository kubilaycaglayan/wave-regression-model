from pathlib import Path

import pytest
import torch
from PIL import Image

import step_7_a_predict as predict_script


def setup_paths(tmp_path, monkeypatch):
    input_dir = tmp_path / "step-7-predict-captures-holder"
    input_dir.mkdir()
    photo = input_dir / "IMG_1234.jpg"
    photo.write_bytes(b"photo bytes")
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint bytes")
    monkeypatch.setattr(predict_script, "PREDICTIONS_DIR", input_dir / "predictions")
    monkeypatch.setattr(predict_script, "PREVIEW_DIR", tmp_path / "previews")
    return photo, checkpoint


def write_record(photo: Path, checkpoint: Path, prediction: float = 0.42) -> Path:
    return predict_script.append_prediction_record(
        photo,
        checkpoint,
        predict_script.sha256_for(checkpoint),
        prediction,
        torch.device("cpu"),
        {"photo_elapsed": 1.25},
    )


def test_matching_checkpoint_is_recorded_and_can_be_read(tmp_path, monkeypatch):
    photo, checkpoint = setup_paths(tmp_path, monkeypatch)

    log_path = write_record(photo, checkpoint)

    assert log_path == tmp_path / "step-7-predict-captures-holder" / "predictions" / "IMG_1234.txt"
    assert predict_script.has_matching_prediction(
        photo, checkpoint, predict_script.sha256_for(checkpoint)
    )


def test_different_checkpoint_does_not_match_and_appends_history(tmp_path, monkeypatch):
    photo, checkpoint = setup_paths(tmp_path, monkeypatch)
    write_record(photo, checkpoint)
    other_checkpoint = tmp_path / "other.ckpt"
    other_checkpoint.write_bytes(b"different checkpoint bytes")

    assert not predict_script.has_matching_prediction(
        photo, other_checkpoint, predict_script.sha256_for(other_checkpoint)
    )
    write_record(photo, other_checkpoint, prediction=0.73)

    records = predict_script.read_prediction_records(predict_script.prediction_log_path_for(photo))
    assert len(records) == 2
    assert records[0]["prediction"] == "0.42000000"
    assert records[1]["prediction"] == "0.73000000"


def test_skip_requires_matching_preview(tmp_path, monkeypatch):
    photo, checkpoint = setup_paths(tmp_path, monkeypatch)
    write_record(photo, checkpoint)
    checkpoint_hash = predict_script.sha256_for(checkpoint)

    assert not predict_script.should_skip_photo(photo, checkpoint, checkpoint_hash)

    predict_script.preview_path_for(photo).parent.mkdir(parents=True)
    Image.new("RGB", predict_script.IMAGE_SIZE, color="white").save(
        predict_script.preview_path_for(photo), format="JPEG"
    )

    assert predict_script.should_skip_photo(photo, checkpoint, checkpoint_hash)


def test_replaced_checkpoint_at_same_path_does_not_match(tmp_path, monkeypatch):
    photo, checkpoint = setup_paths(tmp_path, monkeypatch)
    original_hash = predict_script.sha256_for(checkpoint)
    write_record(photo, checkpoint)
    checkpoint.write_bytes(b"replacement checkpoint bytes")

    assert not predict_script.has_matching_prediction(
        photo, checkpoint, predict_script.sha256_for(checkpoint)
    )
    assert original_hash != predict_script.sha256_for(checkpoint)


def test_valid_preview_is_loaded_as_detached_model_input(tmp_path, monkeypatch):
    photo, _ = setup_paths(tmp_path, monkeypatch)
    preview_path = predict_script.preview_path_for(photo)
    preview_path.parent.mkdir(parents=True)
    Image.new("RGB", predict_script.IMAGE_SIZE, color="white").save(preview_path, format="JPEG")

    model_input = predict_script.load_preview(preview_path)

    assert model_input.mode == "RGB"
    assert model_input.size == predict_script.IMAGE_SIZE
    assert getattr(model_input, "fp", None) is None


def test_cached_preview_does_not_invoke_preprocessing(tmp_path, monkeypatch):
    photo, _ = setup_paths(tmp_path, monkeypatch)
    preview_path = predict_script.preview_path_for(photo)
    preview_path.parent.mkdir(parents=True)
    Image.new("RGB", predict_script.IMAGE_SIZE, color="white").save(preview_path, format="JPEG")
    cached_input = predict_script.load_preview(preview_path)
    monkeypatch.setattr(predict_script, "preprocess_photo", lambda *args: (_ for _ in ()).throw(AssertionError()))

    model_input, encoded, reused = predict_script.prepare_model_input(
        photo, {photo: cached_input}, segmentation_model=None
    )

    assert model_input is cached_input
    assert encoded is None
    assert reused


def test_missing_preview_runs_preprocessing(tmp_path, monkeypatch):
    photo, _ = setup_paths(tmp_path, monkeypatch)
    expected_input = Image.new("RGB", predict_script.IMAGE_SIZE, color="white")
    monkeypatch.setattr(
        predict_script, "preprocess_photo", lambda path, model: (expected_input, b"encoded preview")
    )

    model_input, encoded, reused = predict_script.prepare_model_input(photo, {}, object())

    assert model_input is expected_input
    assert encoded == b"encoded preview"
    assert not reused


@pytest.mark.parametrize(
    ("mode", "size"),
    [("RGB", (128, 128)), ("L", predict_script.IMAGE_SIZE)],
)
def test_invalid_preview_fails_clearly(tmp_path, monkeypatch, mode, size):
    photo, _ = setup_paths(tmp_path, monkeypatch)
    preview_path = predict_script.preview_path_for(photo)
    preview_path.parent.mkdir(parents=True)
    Image.new(mode, size).save(preview_path, format="JPEG")

    with pytest.raises(RuntimeError, match="Invalid inference preview"):
        predict_script.load_preview(preview_path)
