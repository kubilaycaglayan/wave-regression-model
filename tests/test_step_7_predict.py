from pathlib import Path

import torch

import step_7_a_predict as predict_script


def setup_paths(tmp_path, monkeypatch):
    input_dir = tmp_path / "predict-holder"
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

    assert log_path == tmp_path / "predict-holder" / "predictions" / "IMG_1234.txt"
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
    predict_script.preview_path_for(photo).write_bytes(b"preview")

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
