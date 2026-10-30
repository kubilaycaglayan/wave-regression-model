"""Optional parallel libheif tile decoder used by shared image loading."""

from __future__ import annotations

import ctypes
import logging
from pathlib import Path

from PIL import Image


LOGGER = logging.getLogger(__name__)
_LIBRARY_PATH = Path(__file__).parent / "native" / "build" / "libwave_heif_tiles.so"
_LIBRARY = None
_LIBRARY_LOAD_ATTEMPTED = False


def _library():
    global _LIBRARY, _LIBRARY_LOAD_ATTEMPTED
    if _LIBRARY is None and not _LIBRARY_LOAD_ATTEMPTED and _LIBRARY_PATH.is_file():
        _LIBRARY_LOAD_ATTEMPTED = True
        try:
            lib = ctypes.CDLL(str(_LIBRARY_PATH))
            lib.wave_heif_decode_tiled_rgb.argtypes = [
                ctypes.c_char_p, ctypes.c_int,
                ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),
                ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                ctypes.c_char_p, ctypes.c_size_t,
            ]
            lib.wave_heif_decode_tiled_rgb.restype = ctypes.c_int
            lib.wave_heif_free_rgb.argtypes = [ctypes.c_void_p]
            lib.wave_heif_free_rgb.restype = None
            _LIBRARY = lib
        except OSError as exc:
            LOGGER.warning("Could not load optional native HEIF decoder: %s; using Pillow", exc)
    return _LIBRARY


def decode_heif_rgb(path: str | Path, workers: int = 16) -> Image.Image | None:
    """Return oriented RGB pixels, or None when the optional helper is absent/fails."""
    library = _library()
    if library is None:
        return None

    output = ctypes.POINTER(ctypes.c_ubyte)()
    width, height = ctypes.c_int(), ctypes.c_int()
    message = ctypes.create_string_buffer(512)
    encoded_path = str(Path(path).resolve()).encode()
    status = library.wave_heif_decode_tiled_rgb(
        encoded_path, workers, ctypes.byref(output), ctypes.byref(width),
        ctypes.byref(height), message, len(message),
    )
    if status:
        LOGGER.warning("Native HEIF decode failed for %s: %s; using Pillow", path, message.value.decode(errors="replace"))
        return None
    try:
        # PIL copies synchronously in frombytes. Expose the C allocation through
        # the buffer protocol so PIL can copy directly into its image storage,
        # avoiding an intermediate ctypes.string_at() bytes allocation/copy.
        length = width.value * height.value * 3
        pixels = (ctypes.c_ubyte * length).from_address(ctypes.addressof(output.contents))
        return Image.frombytes("RGB", (width.value, height.value), pixels)
    finally:
        library.wave_heif_free_rgb(output)
