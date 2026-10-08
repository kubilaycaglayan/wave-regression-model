"""Compatibility helpers for TorchVision operators used by this project."""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import torch


_COMPAT_LIBRARY: torch.library.Library | None = None
_COMPATIBILITY_CHECKED = False


def _load_native_torchvision_extension() -> None:
    """Load TorchVision's native library before deciding whether fallback is needed."""
    spec = find_spec("torchvision")
    if spec is None or spec.submodule_search_locations is None:
        return

    candidates = sorted(
        candidate
        for package_path in spec.submodule_search_locations
        for candidate in Path(package_path).glob("_C*")
        if candidate.is_file()
    )
    if not candidates:
        return

    # Newer TorchVision wheels may split operators across _C and _C_stable.
    # Load each one before probing schemas so the fallback cannot shadow a
    # schema that the stable extension registers.
    for candidate in candidates:
        try:
            torch.ops.load_library(str(candidate))
        except (ImportError, OSError, RuntimeError):
            # A missing dependency or incompatible binary is the case the
            # schema fallback below is intended to handle.
            continue


def ensure_torchvision_operator_schemas() -> bool:
    """Register missing optional operator schemas before importing TorchVision.

    Some older machine images pair Torch and TorchVision wheels whose compiled
    operators do not load together. TorchVision can then fail during import
    because its fake-kernel registration expects these schemas to exist. The
    project uses transforms and classification models, not these detection
    operators, so define only missing schemas as an import-time fallback.

    Returns True when at least one fallback schema was registered.
    """
    global _COMPAT_LIBRARY, _COMPATIBILITY_CHECKED

    if _COMPATIBILITY_CHECKED:
        return _COMPAT_LIBRARY is not None

    # Schemas are normally registered by this extension. Loading it first is
    # essential: probing before importing TorchVision would otherwise mistake
    # a healthy, not-yet-loaded extension for a broken one.
    _load_native_torchvision_extension()

    missing_schemas: list[str] = []
    for operator in ("nms", "qnms"):
        schema_name = f"torchvision::{operator}"
        try:
            torch._C._dispatch_find_schema_or_throw(schema_name, "")
        except RuntimeError:
            missing_schemas.append(operator)

    if not missing_schemas:
        _COMPATIBILITY_CHECKED = True
        return False

    library = torch.library.Library("torchvision", "FRAGMENT")
    for operator in missing_schemas:
        library.define(
            f"{operator}(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor"
        )

    # Keep the Library alive; Torch unregisters its definitions when it is
    # destroyed.
    _COMPAT_LIBRARY = library
    _COMPATIBILITY_CHECKED = True
    return True
