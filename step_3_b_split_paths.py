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
