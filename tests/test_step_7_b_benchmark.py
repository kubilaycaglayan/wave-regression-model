from types import SimpleNamespace

import pytest

from step_7_b_benchmark_prediction import Recorder, collect_environment, collect_photo_specs, positive_int, report_markdown, summarize, wrap, write_report_pair


def test_recorder_accumulates_nested_timings_and_metadata():
    recorder = Recorder(SimpleNamespace(device=lambda x: SimpleNamespace(type=x)))
    recorder.phase = "step1"
    with recorder.measure("tile_model", "cpu"):
        pass
    with recorder.measure("tile_model", "cpu"):
        pass
    recorder.segmentation_metadata(1200, 900, 6, "cpu", "cuda")
    assert recorder.timings["step1.tile_model"] >= 0
    assert recorder.segmentations[0]["tiles"] == 6
    assert recorder.segmentations[0]["phase"] == "step1"


def test_wrapper_keeps_result_and_records_failures():
    recorder = Recorder(SimpleNamespace(device=lambda x: SimpleNamespace(type=x)))
    module = SimpleNamespace(call=lambda x: 10 / x)
    wrap(module, "call", recorder, "operation")
    assert module.call(2) == 5
    with pytest.raises(ZeroDivisionError):
        module.call(0)
    assert "operation" in recorder.timings


def test_summary_excludes_failures_and_aggregates_by_kind():
    runs = [
        {"kind": "warm", "status": "ok", "fresh_seconds": x, "cached_seconds": 1, "timings": {"step2.crop_selection": x / 2}}
        for x in (2, 4, 9)
    ] + [{"kind": "warm", "status": "failed"}]
    summary = {(row["kind"], row["stage"]): row for row in summarize(runs)}
    assert summary["warm", "fresh_total"]["median_seconds"] == 4
    assert summary["warm", "step2.crop_selection"]["max_seconds"] == 4.5
    assert summary["warm", "fresh_total"]["samples"] == 3


def test_repeat_must_be_positive():
    with pytest.raises(Exception):
        positive_int("0")


def test_environment_metadata_is_serializable():
    import json
    import torch
    result = collect_environment(torch)
    assert result["system"]["logical_cpus"] > 0
    assert result["software"]["packages"]["torch"]
    assert result["runtime"]["torch_threads"] > 0
    json.dumps(result)


def test_photo_metadata_records_identity_without_image_data(tmp_path):
    from PIL import Image
    import hashlib
    photo = tmp_path / "photo.png"
    Image.new("RGB", (13, 7)).save(photo)
    helper = SimpleNamespace(sha256_for=lambda path: hashlib.sha256(path.read_bytes()).hexdigest())
    result = collect_photo_specs([photo], helper)[0]
    assert result["dimensions"] == [13, 7]
    assert result["format"] == "PNG"
    assert result["sha256"] == hashlib.sha256(photo.read_bytes()).hexdigest()
    assert result["size_bytes"] == photo.stat().st_size


def test_markdown_reports_individual_failed_traces_and_summary(tmp_path):
    import json
    report = {"created_utc": "2026-01-01T00:00:00Z", "summary": [], "processes": [{
        "startup": {"imports_seconds": 0.25}, "environment": {"system": {"cpu_model": "CPU"}},
        "runs": [{"kind": "cold", "photo": "sea|photo.jpg", "status": "failed",
                  "error": "no water", "elapsed_seconds": 2.5, "timings": {"step1.segmentation": 2.0}}]}]}
    path = tmp_path / "benchmark.json"
    markdown_path = write_report_pair(report, path)
    assert json.loads(path.read_text()) == report
    markdown = markdown_path.read_text()
    assert "sea\\|photo.jpg" in markdown
    assert "no water" in markdown
    assert "↳ Segmentation | 2.000000 |" in markdown
    assert "Step 1: water preprocessing (total not recorded)" in markdown
    assert "No successful runs" in markdown
    assert "CPU" in markdown
    assert markdown == report_markdown(report)


def test_markdown_distinguishes_single_and_multiple_samples():
    report = {"summary": [
        {"kind": "cold", "stage": "fresh_total", "samples": 1, "median_seconds": 3, "min_seconds": 3, "max_seconds": 3},
        {"kind": "warm", "stage": "fresh_total", "samples": 2, "median_seconds": 4, "min_seconds": 3, "max_seconds": 5},
    ]}
    markdown = report_markdown(report)
    assert "| ❄️ cold | Fresh prediction total | 3.000000 |" in markdown
    assert "| 🔥 warm | Fresh prediction total | 4.000000 | 5.000000 | 3.000000 | 5.000000 | 2 |" in markdown
    assert "| Run | Measurement | Duration (s) |" in markdown
    assert "| Run | Measurement | Median (s) | p95 (s) | Min (s) | Max (s) | Samples |" in markdown


def test_major_timings_show_separate_prediction_trees():
    stages = ("fresh_total", "step1.total", "step2.total", "fresh.regression", "cached_total", "cached.regression")
    report = {"summary": [{"kind": "cold", "stage": stage, "samples": 1, "median_seconds": 1,
                            "min_seconds": 1, "max_seconds": 1} for stage in stages]}
    markdown = report_markdown(report)
    fresh = markdown.index("| ❄️ cold | Fresh prediction total")
    step1 = markdown.index("| ❄️ cold | ├─ Step 1")
    step2 = markdown.index("| ❄️ cold | ├─ Step 2")
    regression = markdown.index("| ❄️ cold | └─ Fresh regression")
    cached = markdown.index("| ❄️ cold | Cached prediction total")
    cached_regression = markdown.index("| ❄️ cold | └─ Cached regression")
    assert fresh < step1 < step2 < regression < cached < cached_regression


@pytest.mark.parametrize("cached", [False, True])
def test_worker_predicts_once_per_pass_using_existing_preview(tmp_path, monkeypatch, cached):
    import json
    import time
    import torch
    import step_7_b_benchmark_prediction as benchmark
    import step_7_a_predict as prediction
    import step_1_a_water_segmentation as segmentation
    import step_1_b_preprocess_water_inputs as step1
    import step_2_a_reduce_black_water_area as step2

    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"photo")
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    preview_dir = tmp_path / "existing-previews"
    preview_dir.mkdir()
    preview = preview_dir / "photo-model-input.jpg"
    if cached:
        preview.write_bytes(b"cached")
    monkeypatch.setattr(prediction, "PREVIEW_DIR", preview_dir)
    monkeypatch.setattr(prediction, "PREDICTIONS_DIR", tmp_path / "existing-predictions")
    monkeypatch.setattr(prediction, "checkpoint_path_from_env", lambda: checkpoint)
    monkeypatch.setattr(prediction, "validate_checkpoint_preprocessing", lambda *args, **kwargs: None)
    monkeypatch.setattr(segmentation, "BENCHMARK_OBSERVER", None)
    monkeypatch.setattr(segmentation, "configure_inference_device", lambda *args: torch.device("cpu"))
    calls = []
    monkeypatch.setattr(segmentation, "load_model", lambda *args, **kwargs: SimpleNamespace(device=torch.device("cpu")))
    class FakeModel:
        backbone_name = "test"
        def forward(self, *args): pass
        def to(self, device): return self
        def eval(self): return self
    monkeypatch.setattr(prediction.WaveRegressionModel, "load_from_checkpoint", lambda *args, **kwargs: FakeModel())
    monkeypatch.setattr(prediction, "build_evaluation_transform", lambda: lambda image: image)
    monkeypatch.setattr(prediction, "preprocess_photo", lambda *args: ("fresh-input", b"preview"))
    monkeypatch.setattr(prediction, "load_preview", lambda path: "cached-input")
    monkeypatch.setattr(prediction, "save_preview", lambda photo, encoded: tmp_path / "temporary-preview")
    monkeypatch.setattr(prediction, "predict", lambda image, *args: (calls.append(image) or 0.5, (1,)))
    monkeypatch.setattr(prediction, "append_prediction_record", lambda *args: None)
    monkeypatch.setattr(benchmark, "collect_environment", lambda torch: {})
    monkeypatch.setattr(benchmark, "collect_photo_specs", lambda photos, prediction: [])
    for module, names in ((prediction, ("load_rgb_image", "preprocess_raw_image", "preprocess_step_1_image", "encode_step_1_output", "encode_step_2_output", "decode_rgb_bytes")),
                          (step1, ("run_segmentation", "standardized_model_input")),
                          (step2, ("run_segmentation", "select_crop", "fit_in_square"))):
        for name in names:
            monkeypatch.setattr(module, name, getattr(module, name))
    output = tmp_path / "result.json"
    benchmark.worker(SimpleNamespace(photos=[photo], device="cpu", worker_output=output), time.perf_counter())
    result = json.loads(output.read_text())
    assert calls == ["cached-input" if cached else "fresh-input"] * 2
    assert [run["input_path_type"] for run in result["runs"]] == ["cached" if cached else "fresh"] * 2
    assert all("prediction_seconds" in run and "cached_prediction" not in run for run in result["runs"])
    assert not (tmp_path / "existing-predictions").exists()


def test_process_breakdown_places_traces_under_their_parents():
    run = {"kind": "cold", "photo": "sea.jpg", "status": "ok", "input_path_type": "fresh",
           "prediction_seconds": 10, "timings": {"step1.total": 6, "step1.segmentation": 5,
           "step1.tile_model": 4, "step2.total": 2, "fresh.regression": 1, "fresh.model_forward": 0.8}}
    report = {"report_version": 3, "processes": [{"process_wall_seconds": 12,
              "startup": {"imports_seconds": 1}, "runs": [run]}], "summary": []}
    text = report_markdown(report)
    process = text.index("Process wall time: 12.000000")
    startup = text.index("#### Startup")
    prediction = text.index("**Fresh prediction total**")
    step1 = text.index("↳ Step 1: water preprocessing")
    segmentation = text.index("↳ Segmentation", step1)
    tile = text.index("↳ Run tile model", segmentation)
    step2 = text.index("↳ Step 2: lower-water preprocessing", tile)
    regression = text.index("↳ Regression prediction", step2)
    forward = text.index("↳ Model forward pass", regression)
    assert process < startup < prediction < step1 < segmentation < tile < step2 < regression < forward
    assert "&nbsp;" * 12 + "↳ Run tile model" in text
    assert "## Detailed timing summary" not in text


def test_warm_percentile_uses_nearest_rank_and_excludes_failures():
    from step_7_b_benchmark_prediction import summarize
    runs = [{"kind": "warm", "status": "ok", "input_path_type": "fresh",
             "prediction_seconds": float(i), "timings": {}} for i in range(1, 21)]
    runs.append({"kind": "warm", "status": "failed"})
    item = summarize(runs)[0]
    assert item["samples"] == 20
    assert item["median_seconds"] == 10.5
    assert item["p95_seconds"] == 19
