"""Expose organized script folders for legacy-style test imports."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
for folder in ("scripts_pipeline", "experiment_scripts", "scripts"):
    sys.path.insert(0, str(ROOT / folder))
