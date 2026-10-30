"""Build the optional tiled HEIF decoder against Pillow-Heif's bundled libheif."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
INCLUDE = ROOT / "native" / "libheif" / "include"
SOURCE = ROOT / "native" / "libheif" / "step_0_a_tiled_decode.cpp"
OUTPUT = ROOT / "native" / "build" / "libwave_heif_tiles.so"


def main() -> None:
    if sys.platform != "linux":
        raise SystemExit("The optional native HEIF decoder currently supports Linux only.")
    spec = importlib.util.find_spec("pillow_heif")
    if spec is None or not spec.origin:
        raise SystemExit("Install pillow-heif before building the optional native decoder.")
    package_dir = Path(spec.origin).resolve().parent
    library_dirs = (package_dir.parent / "pillow_heif.libs", package_dir / ".libs")
    bundled_dir = next((path for path in library_dirs if list(path.glob("libheif-*.so*"))), None)
    if bundled_dir is None:
        raise SystemExit("Could not locate Pillow-Heif's bundled libheif shared library.")
    libraries = sorted(bundled_dir.glob("libheif-*.so*"), key=lambda path: path.name)
    # Link the exact shared object shipped with this Python environment; the
    # generated RUNPATH keeps the ignored build artifact tied to that package.
    libheif = next((path for path in libraries if path.is_symlink() and path.suffix == ".4"), libraries[0])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    command = [
        os.environ.get("CXX", "g++"), "-O3", "-std=c++17", "-fPIC", "-shared",
        "-pthread", f"-I{INCLUDE}", str(SOURCE), str(libheif),
        f"-Wl,-rpath,{bundled_dir}", "-o", str(OUTPUT),
    ]
    subprocess.run(command, check=True)
    print(f"Built optional decoder: {OUTPUT}")
    print(f"Linked Pillow-Heif library: {libheif}")


if __name__ == "__main__":
    main()
