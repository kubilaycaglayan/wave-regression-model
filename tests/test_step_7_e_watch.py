from types import SimpleNamespace
import os
import subprocess
import sys
import time

import torch
import pytest

import step_7_e_watch_predictions as watch


def test_watcher_skips_completed_photo(monkeypatch, tmp_path):
    monkeypatch.setattr(watch.prediction, "should_skip_photo", lambda *args: True)
    device = torch.device("cpu")
    assert watch.process_photo(tmp_path / "sea.jpg", tmp_path / "model.ckpt", "hash", device, None, None, None) == device


def test_watcher_reuses_preview_and_writes_prediction(monkeypatch, tmp_path):
    photo = tmp_path / "sea.jpg"
    preview = tmp_path / "preview.jpg"
    preview.write_bytes(b"preview")
    monkeypatch.setattr(watch.prediction, "should_skip_photo", lambda *args: False)
    monkeypatch.setattr(watch.prediction, "preview_path_for", lambda photo: preview)
    monkeypatch.setattr(watch.prediction, "load_preview", lambda path: "cached")
    monkeypatch.setattr(watch.prediction, "predict", lambda image, *args: (0.4, (1, 3, 224, 224)))
    written = []
    monkeypatch.setattr(watch.prediction, "append_prediction_record", lambda *args: written.append(args))
    device = torch.device("cpu")
    watch.process_photo(photo, tmp_path / "model.ckpt", "hash", device, object(), None, SimpleNamespace(device=device))
    assert written[0][0] == photo and written[0][3] == 0.4


def test_experimental_watcher_keeps_caches_separate(monkeypatch, tmp_path):
    baseline_preview = tmp_path / "previews"
    baseline_records = tmp_path / "records"
    monkeypatch.setattr(watch.prediction, "PREVIEW_DIR", baseline_preview)
    monkeypatch.setattr(watch.prediction, "PREDICTIONS_DIR", baseline_records)
    for name in ("SEGMENTATION_RESOLUTION", "USE_OVERLAPPING_TILES", "SINGLE_IMAGE_LOGITS_RESOLUTION", "INFERENCE_PRECISION"):
        monkeypatch.setattr(watch.segmentation, name, getattr(watch.segmentation, name))
    variant = watch.configure_experimental(384, 384, False)
    assert watch.prediction.PREVIEW_DIR == baseline_preview / variant
    assert watch.prediction.PREDICTIONS_DIR == baseline_records / variant
    assert (watch.prediction.PREDICTIONS_DIR / "preprocessing.json").is_file()
    assert watch.segmentation.SEGMENTATION_RESOLUTION == 384
    assert not watch.segmentation.USE_OVERLAPPING_TILES
    assert watch.segmentation.SINGLE_IMAGE_LOGITS_RESOLUTION == 384
    watch.prediction.PREVIEW_DIR = baseline_preview
    watch.prediction.PREDICTIONS_DIR = baseline_records
    assert watch.configure_experimental(384, 384, False) == variant
    watch.prediction.PREVIEW_DIR = baseline_preview
    watch.prediction.PREDICTIONS_DIR = baseline_records
    assert watch.configure_experimental(384, 384, True) != variant


def test_experimental_step2_resolution_restores_processor(monkeypatch, tmp_path):
    monkeypatch.setattr(watch.prediction, "PREVIEW_DIR", tmp_path / "previews")
    monkeypatch.setattr(watch.prediction, "PREDICTIONS_DIR", tmp_path / "records")
    monkeypatch.setattr(watch.prediction, "preprocess_step_1_image", lambda image, model: model.processor.size.copy())
    for name in ("SEGMENTATION_RESOLUTION", "USE_OVERLAPPING_TILES", "SINGLE_IMAGE_LOGITS_RESOLUTION", "INFERENCE_PRECISION"):
        monkeypatch.setattr(watch.segmentation, name, getattr(watch.segmentation, name))
    watch.configure_experimental(448, 448, False, step2_resolution=384, precision="float16")
    segmenter = SimpleNamespace(processor=SimpleNamespace(size={"height": 448, "width": 448}))
    assert watch.prediction.preprocess_step_1_image(None, segmenter) == {"height": 384, "width": 384}
    assert segmenter.processor.size == {"height": 448, "width": 448}
    assert watch.segmentation.INFERENCE_PRECISION == "float16"


def test_experimental_fallback_does_not_save_in_wrong_cache(monkeypatch, tmp_path):
    photo = tmp_path / "sea.jpg"
    monkeypatch.setattr(watch.prediction, "should_skip_photo", lambda *args: False)
    monkeypatch.setattr(watch.prediction, "preview_path_for", lambda photo: tmp_path / "missing.jpg")
    monkeypatch.setattr(watch.prediction, "preprocess_photo", lambda *args: ("input", b"preview"))
    monkeypatch.setattr(watch.prediction, "save_preview", lambda *args: pytest.fail("saved wrong-device preview"))
    segmenter = SimpleNamespace(device=torch.device("cpu"))
    with pytest.raises(RuntimeError, match="refusing to mix"):
        watch.process_photo(photo, tmp_path / "model.ckpt", "hash", torch.device("cuda"), None, None, segmenter,
                            expected_segmentation_device=torch.device("cuda"))


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux file notifications")
def test_file_notifications_wait_for_close_and_accept_atomic_move(tmp_path):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    events = watch.FileNotifications(incoming)
    try:
        with (incoming / "photo.jpg").open("wb") as handle:
            handle.write(b"partial")
            handle.flush()
            assert events.read(.05) == ([], False)
            handle.write(b"complete")
        names, overflow = events.read(1)
        assert names == ["photo.jpg"] and not overflow
        staged = tmp_path / "staged.heic"
        staged.write_bytes(b"complete")
        os.replace(staged, incoming / "moved.heic")
        names, overflow = events.read(1)
        assert names == ["moved.heic"] and not overflow
    finally:
        events.close()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux file notifications")
def test_event_watcher_does_not_scan_an_open_new_file(tmp_path):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    log = tmp_path / "watcher.log"
    marker = tmp_path / "processed.txt"
    program = '''
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
import step_7_e_watch_predictions as w
root = Path(sys.argv[1])
w.prediction.INPUT_DIR = root / "incoming"
w.prediction.checkpoint_path_from_env = lambda: root / "unused.ckpt"
w.prediction.sha256_for = lambda path: "digest"
w.prediction.validate_checkpoint_preprocessing = lambda *a, **k: None
w.prediction.validate_segmentation_revision = lambda *a, **k: None
w.configure_inference_device = lambda **k: torch.device("cpu")
w.load_model = lambda *a, **k: SimpleNamespace(device=torch.device("cpu"), processor=SimpleNamespace(size={}))
w.run_segmentation = lambda *a, **k: None
class FakeModel:
    def to(self, device): return self
    def eval(self): return self
    def __call__(self, value): return torch.zeros(1)
w.prediction.WaveRegressionModel.load_from_checkpoint = lambda *a, **k: FakeModel()
w.prediction.build_evaluation_transform = lambda: None
def record(photo, checkpoint, digest, device, *a, **k):
    with (root / "processed.txt").open("a") as out: out.write(photo.name + "\\n")
    return device
w.process_photo = record
sys.argv = ["watcher", "--watch-mode", "event", "--poll-seconds", "0.05"]
w.main()
'''
    with log.open("w") as output:
        child = subprocess.Popen([sys.executable, "-c", program, str(tmp_path)], stdout=output, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 15
            while "File notifications ready" not in log.read_text():
                if child.poll() is not None or time.monotonic() > deadline:
                    pytest.fail(f"Watcher did not start: {log.read_text()}")
                time.sleep(.02)
            with (incoming / "new.jpg").open("wb") as handle:
                handle.write(b"partial")
                handle.flush()
                time.sleep(.25)  # longer than several stability scans
                assert not marker.exists()
                handle.write(b"complete")
            deadline = time.monotonic() + 3
            while not marker.exists():
                if child.poll() is not None or time.monotonic() > deadline:
                    pytest.fail(f"Watcher did not handle closed file: {log.read_text()}")
                time.sleep(.01)
            assert marker.read_text().splitlines() == ["new.jpg"]
        finally:
            child.terminate()
            child.wait(timeout=5)
