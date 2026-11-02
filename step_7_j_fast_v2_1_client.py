"""Local client for the versioned MIGraphX coordinate-crop prediction runtime."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
SEGMENTATION_MODEL = ROOT / "models/fast-v2.1/segformer-b0-step1-448-fp32.onnx"
PIPELINES = {
    "fast-v2.1": {
        "profile": "fast-v2.1-coordinate",
        "experiment": "distilled-b0-migraphx-fp32",
        "regressor": ROOT / "models/fast-v2.1/wave-regression-efficientnet-b0-fast-v2.1.onnx",
        "fp16": False,
    },
    "fast-v2.2": {
        "profile": "fast-v2.2-coordinate",
        "experiment": "fast-v2.2",
        "regressor": ROOT / "models/fast-v2.2/wave-regression-efficientnet-b0-fast-v2.2.onnx",
        "fp16": True,
    },
}
SERVER_SCRIPT = ROOT / "step_7_i_migraphx_server.py"


def configured_value(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is not None:
        return value
    dotenv_path = ROOT / ".env"
    if dotenv_path.is_file():
        for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            key, separator, candidate = line.partition("=")
            if separator and key.strip() == name:
                return candidate.strip().strip("\"'")
    return default


class FastV2_1Client:
    """Start or reuse the selected local coordinate-crop service."""

    def __init__(self, *, regressor: Path | None = None, port: int | None = None,
                 pipeline_version: str = "fast-v2.2"):
        if pipeline_version not in PIPELINES:
            raise ValueError(f"Unsupported MIGraphX preprocessing profile: {pipeline_version}")
        self.pipeline_version = pipeline_version
        self.pipeline = PIPELINES[pipeline_version]
        configured_default = str(self.pipeline["regressor"])
        selected = regressor or Path(configured_value("PREDICT_REGRESSOR_ONNX_PATH", configured_default))
        self.regressor = selected.expanduser()
        if not self.regressor.is_absolute():
            self.regressor = ROOT / self.regressor
        self.port = port or int(configured_value("PREDICT_MIGRAPHX_PORT", "8766"))
        self.base_url = f"http://127.0.0.1:{self.port}"
        runtime = Path(configured_value(
            "PREDICT_MIGRAPHX_PYTHON", "/tmp/wave_migraphx_312_env/bin/python"
        )).expanduser()
        self.runtime_python = runtime
        self.process: subprocess.Popen | None = None
        self.log_handle = None
        self.server_log = Path(tempfile.gettempdir()) / f"wave-{self.pipeline_version}-server-{self.port}.log"
        self.health: dict | None = None

    def __enter__(self) -> "FastV2_1Client":
        if not self.regressor.is_file():
            raise FileNotFoundError(f"{self.pipeline_version}-matched ONNX regressor is missing: {self.regressor}")
        if not SEGMENTATION_MODEL.is_file():
            raise FileNotFoundError(f"{self.pipeline_version} SegFormer-B0 ONNX model is missing: {SEGMENTATION_MODEL}")
        if not self.runtime_python.is_file():
            raise FileNotFoundError(
                f"MIGraphX Python runtime not found: {self.runtime_python}. Set PREDICT_MIGRAPHX_PYTHON "
                "to the Python 3.12 runtime containing onnxruntime-ep-migraphx."
            )
        try:
            with urlopen(f"{self.base_url}/health", timeout=1) as response:
                self.health = json.loads(response.read())
        except (OSError, URLError, TimeoutError):
            self._start_server()
        self._validate_health()
        return self

    def _start_server(self) -> None:
        runtime_root = self.runtime_python.parent.parent
        migraphx_libs = runtime_root / "lib/python3.12/site-packages/migraphx_libs"
        library_paths = [str(path) for path in (migraphx_libs, Path("/opt/rocm/lib")) if path.is_dir()]
        prior_ld = os.getenv("LD_LIBRARY_PATH", "")
        if prior_ld:
            library_paths.append(prior_ld)
        server_env = os.environ.copy()
        server_env.update({
            "WAVE_EXPERIMENT": self.pipeline["experiment"],
            "WAVE_PIPELINE_PROFILE": self.pipeline["profile"],
            "WAVE_SERVER_PORT": str(self.port),
            "WAVE_REGRESSOR_ONNX": str(self.regressor.resolve()),
            "WAVE_STEP1_ONNX": str(SEGMENTATION_MODEL.resolve()),
            "MIGRAPHX_SKIP_BENCHMARKING": "1",
            "MIGRAPHX_USE_FAST_SOFTMAX": "0",
            "LD_LIBRARY_PATH": ":".join(library_paths),
        })
        if self.pipeline["fp16"]:
            server_env["WAVE_MIGRAPHX_FP16"] = "1"
        else:
            server_env.pop("WAVE_MIGRAPHX_FP16", None)
        self.log_handle = self.server_log.open("a", encoding="utf-8")
        self.process = subprocess.Popen(
            [str(self.runtime_python), str(SERVER_SCRIPT)], cwd=ROOT, env=server_env,
            stdout=self.log_handle, stderr=subprocess.STDOUT, start_new_session=True,
        )
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"MIGraphX prediction server exited; see {self.server_log}")
            try:
                with urlopen(f"{self.base_url}/health", timeout=1) as response:
                    self.health = json.loads(response.read())
                break
            except (OSError, URLError, TimeoutError):
                time.sleep(0.25)
        if self.health is None:
            raise TimeoutError(f"MIGraphX prediction server did not become ready; see {self.server_log}")

    def _validate_health(self) -> None:
        health = self.health or {}
        if health.get("pipeline_profile") != self.pipeline["profile"]:
            raise RuntimeError(f"Port {self.port} is serving the wrong pipeline: {health}")
        if health.get("migraphx_fp16") is not self.pipeline["fp16"]:
            raise RuntimeError(f"Port {self.port} is serving the wrong segmentation precision: {health}")
        if Path(health.get("regressor_model", "")).resolve() != self.regressor.resolve():
            raise RuntimeError(f"Port {self.port} is serving a different regressor: {health.get('regressor_model')}")
        if Path(health.get("step1_model", "")).resolve() != SEGMENTATION_MODEL.resolve():
            raise RuntimeError(f"Port {self.port} is serving a different segmentation model: {health.get('step1_model')}")

    def predict_path(self, photo: Path) -> dict:
        photo = photo.resolve(strict=True)
        allowed_roots = (ROOT / "step-7-predict-captures-holder", ROOT / "step-0-raw-data")
        if any(root.resolve() == photo or root.resolve() in photo.parents for root in allowed_roots):
            request_body = b""
            headers = {"Content-Type": "application/octet-stream", "X-Filename": photo.name,
                       "X-Source-Path": str(photo)}
        else:
            request_body = photo.read_bytes()
            headers = {"Content-Type": "application/octet-stream", "X-Filename": photo.name,
                       "Content-Length": str(len(request_body))}
        request = Request(
            f"{self.base_url}/predict", data=request_body, method="POST", headers=headers,
        )
        with urlopen(request, timeout=180) as response:
            return json.loads(response.read())

    def __exit__(self, *_exc) -> None:
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.log_handle is not None:
            self.log_handle.close()
