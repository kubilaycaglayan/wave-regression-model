from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPOSITORY_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, str(_REPOSITORY_ROOT))
for _script_folder in ("scripts_pipeline", "experiment_scripts", "scripts"):
    _script_path = str(_REPOSITORY_ROOT / _script_folder)
    if _script_path not in _bootstrap_sys.path:
        _bootstrap_sys.path.insert(0, _script_path)
"""User-facing entry point for the step 5 regression baseline."""

from step_5_b_train import main


if __name__ == "__main__":
    main()
