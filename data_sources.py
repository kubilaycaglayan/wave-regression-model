"""Configuration and recursive discovery for raw image sources."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Callable

from image_loading import SUPPORTED_EXTENSIONS

REPOSITORY_ROOT = Path(__file__).resolve().parent


def _read_dotenv(path: Path) -> dict[str, list[str]]:
    """Read path settings while preserving repeated keys."""
    values: dict[str, list[str]] = {}
    if not path.exists():
        return values
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key.strip():
            raise ValueError(f"Invalid .env entry on line {line_number}: {raw_line!r}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values.setdefault(key.strip(), []).append(value)
    return values


def _split_path_values(values: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        for item in value.split(os.pathsep):
            item = item.strip()
            if item:
                paths.append(Path(item).expanduser())
    return paths


def configured_source_directories(repository_root: Path = REPOSITORY_ROOT) -> list[Path]:
    """Return the default raw directory plus configured source directories."""
    dotenv_values = _read_dotenv(repository_root / ".env")
    configured_values = dotenv_values.get("DATA_PATH", [])
    if os.getenv("DATA_PATH"):
        configured_values = [os.environ["DATA_PATH"]]

    raw_directory = repository_root / "step-0-raw-data"
    paths = [raw_directory, *_split_path_values(configured_values)]
    result: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = (repository_root / path).resolve() if not path.is_absolute() else path.resolve()
        if resolved in seen:
            continue
        if not resolved.is_dir():
            if resolved == raw_directory.resolve():
                resolved.mkdir(parents=True, exist_ok=True)
            else:
                raise FileNotFoundError(f"Configured data source directory does not exist: {resolved}")
        seen.add(resolved)
        result.append(resolved)
    return result


def iter_images(input_dirs: Path | Iterable[Path]) -> list[Path]:
    """Recursively return supported images from one or more directories."""
    directories = [input_dirs] if isinstance(input_dirs, Path) else list(input_dirs)
    images: dict[Path, Path] = {}
    for directory in directories:
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                images[path.resolve()] = path
    return sorted(images.values(), key=lambda path: str(path).lower())


class SourceOutputNames:
    """Stable source-to-stem mapping for outputs in one pipeline directory."""

    def __init__(
        self,
        source_paths: Iterable[Path],
        output_dir: Path,
        required_suffixes: Iterable[str],
        output_paths_for_stem: Callable[[str], Iterable[Path]] | None = None,
    ):
        self.output_dir = output_dir
        self.required_suffixes = tuple(required_suffixes)
        self.output_paths_for_stem = output_paths_for_stem
        self.manifest_path = output_dir / ".source-manifest.json"
        self.mapping: dict[str, str] = {}
        if self.manifest_path.exists():
            try:
                loaded = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.mapping = {str(key): str(value) for key, value in loaded.items()}
            except (OSError, json.JSONDecodeError):
                pass
        self.source_paths = sorted(source_paths, key=lambda path: str(path).lower())
        self.used = set(self.mapping.values())
        self.source_stem_counts: dict[str, int] = {}
        for path in self.source_paths:
            self.source_stem_counts[path.stem] = self.source_stem_counts.get(path.stem, 0) + 1

    def _output_paths(self, stem: str) -> tuple[Path, ...]:
        if self.output_paths_for_stem is not None:
            return tuple(self.output_paths_for_stem(stem))
        return tuple(self.output_dir / f"{stem}{suffix}" for suffix in self.required_suffixes)

    def _complete_output_exists(self, stem: str) -> bool:
        paths = self._output_paths(stem)
        return bool(paths) and all(path.exists() for path in paths)

    def stem_for(self, source_path: Path) -> str:
        source_key = str(source_path.resolve())
        if source_key in self.mapping:
            return self.mapping[source_key]
        candidate = source_path.stem
        # A complete, unambiguous output set may have been created before the
        # manifest was flushed (for example if a long run was interrupted).
        # Adopt it instead of creating a new hashed name and reprocessing it.
        if (
            candidate not in self.used
            and self.source_stem_counts.get(candidate, 0) == 1
            and self._complete_output_exists(candidate)
        ):
            self.mapping[source_key] = candidate
            self.used.add(candidate)
            return candidate

        if candidate in self.used or any(path.exists() for path in self._output_paths(candidate)):
            digest = hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:10]
            candidate = f"{candidate}_{digest}"
            while candidate in self.used:
                digest = hashlib.sha256((source_key + candidate).encode("utf-8")).hexdigest()[:10]
                candidate = f"{source_path.stem}_{digest}"
        self.mapping[source_key] = candidate
        self.used.add(candidate)
        return candidate

    def save(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.mapping, indent=2, sort_keys=True) + "\n"
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.output_dir,
                prefix=f".{self.manifest_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(payload)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, self.manifest_path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
