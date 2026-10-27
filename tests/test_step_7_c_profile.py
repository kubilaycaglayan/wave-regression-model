import json

import torch

from step_7_c_profile_prediction import capture_pass, render_report, trace_has_device_kernels


def test_capture_collects_operators_and_exports_trace(tmp_path):
    result, value, error = capture_pass("regression_forward", tmp_path, torch, lambda: torch.ones(3).square(), "cpu")
    assert error is None and value.tolist() == [1, 1, 1]
    assert result["status"] == "ok"
    assert any(row["operator"] == "aten::square" for row in result["operators"])
    assert (tmp_path / result["trace"]).is_file()
    assert result["accelerator_kernels_in_trace"] is None


def test_failed_capture_preserves_failure_and_distinct_retry_trace(tmp_path):
    def fail():
        raise RuntimeError("model failed")
    first, _, error = capture_pass("step1_segmentation", tmp_path, torch, fail, "cpu")
    second, _, second_error = capture_pass("step1_segmentation", tmp_path, torch, lambda: torch.ones(1), "cpu")
    assert str(error) == "model failed" and first["status"] == "failed"
    assert second_error is None and first["trace"] != second["trace"]


def test_trace_detection_and_report_marks_unavailable_device(tmp_path):
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({"traceEvents": [{"cat": "kernel"}]}))
    assert trace_has_device_kernels(trace)
    trace.write_text(json.dumps({"traceEvents": [{"cat": "cpu_op"}]}))
    assert not trace_has_device_kernels(trace)
    report = {"created_utc": "now", "status": "ok", "passes": [{"name": "regression_forward", "status": "ok", "device_at_start": "cuda", "accelerator_warning": "No accelerator kernel events", "operators": [{"operator": "aten::add", "calls": 1, "self_cpu_time_us": 1000, "self_device_time_us": 0}]}]}
    rendered = render_report(report)
    assert "No accelerator kernel events" in rendered
    assert "Not available" in rendered
    assert "not prediction latency measurements" in rendered


def test_capture_optionally_records_shapes_and_memory(tmp_path):
    capture, _, error = capture_pass("shapes", tmp_path, torch, lambda: torch.ones((2, 3)).square(), "cpu",
                                     record_shapes=True, profile_memory=True)
    assert error is None
    row = next(row for row in capture["operators"] if row["operator"] == "aten::square")
    assert row["input_shapes"] and row["self_cpu_memory_bytes"] is not None
