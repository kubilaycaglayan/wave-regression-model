# Repository Reorganization

## 2026-11-03 — Script and experiment organization

1. **Grouped executable scripts by role.** Pipeline steps for preprocessing, training, evaluation, prediction, and serving moved to `scripts_pipeline/`. Experiment runners moved to `experiment_scripts/`; benchmarks, profiling, comparison, and checkpoint migration tools moved to `scripts/`.
2. **Preserved paths and imports.** The relocated scripts resolve the repository root and add their script folders to the import path. Prediction server, native HEIF library, source fingerprint, and documentation paths were updated to follow the new locations.
3. **Grouped step experiment outputs.** The Step 4 augmentation experiment and Step 8–11 experiment folders moved under `experiments/`. Their output defaults, artifact paths, and documentation links were updated. Pipeline data and Step 7 benchmark/profile output folders remain in place.
4. **Checked representative entry points.** Syntax compilation and `git diff --check` passed. Training, prediction, Experiment C, and benchmark `--help` commands ran from their new locations. Representative data, server, and experiment output paths resolved correctly. Full training, prediction, and the test suite were not run.
