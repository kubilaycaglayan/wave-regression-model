from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPOSITORY_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, str(_REPOSITORY_ROOT))
for _script_folder in ("scripts_pipeline", "experiment_scripts", "scripts"):
    _script_path = str(_REPOSITORY_ROOT / _script_folder)
    if _script_path not in _bootstrap_sys.path:
        _bootstrap_sys.path.insert(0, _script_path)
"""Resolve the one shared split registry used by all preprocessing profiles."""

from pathlib import Path

from step_1_c_preprocessing_profiles import PreprocessingProfile, get_profile


def split_directory_for(profile: PreprocessingProfile) -> Path:
    """Return the canonical split registry, shared across preprocessing profiles."""
    # Profile split_dir remains part of the existing preprocessing markers and
    # checkpoint contracts. Resolve storage separately so path changes do not
    # change preprocessing identity or invalidate existing checkpoints.
    get_profile(profile.name)  # Validate that the caller passed a registered profile.
    return get_profile("baseline-v16").split_dir
