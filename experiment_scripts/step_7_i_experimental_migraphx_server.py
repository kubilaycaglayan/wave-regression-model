"""Compatibility import for historical experiment scripts.

The persistent serving implementation is step_7_i_migraphx_server.py.
"""
from __future__ import annotations
from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPOSITORY_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, str(_REPOSITORY_ROOT))
for _script_folder in ("scripts_pipeline", "experiment_scripts", "scripts"):
    _script_path = str(_REPOSITORY_ROOT / _script_folder)
    if _script_path not in _bootstrap_sys.path:
        _bootstrap_sys.path.insert(0, _script_path)


import sys
import step_7_i_migraphx_server as _server

if __name__ == "__main__":
    _server.main()
else:
    sys.modules[__name__] = _server
