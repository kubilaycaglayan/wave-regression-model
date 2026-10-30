# Experiment versions

Each experimental implementation gets its own immutable version directory:

```text
experiment-versions/
├── README.md
├── v000_baseline/
│   ├── experiment.py
│   └── experiment.json
├── v001_<short-hypothesis>/
│   ├── experiment.py
│   └── experiment.json
└── v002_<short-hypothesis>/
    ├── experiment.py
    └── experiment.json
```

`v000_baseline` is the reference. Do not edit it after experiments begin.
Create a new version directory for each distinct code change; do not overwrite
an earlier version. Keep experiment-specific implementation code and notes in
that version directory. Put generated benchmark JSON under
`runs/<version-id>/`.

The runner loads `experiment.py` and calls:

```python
def run_pipeline(photo, server, reference_pipeline) -> dict:
    ...
```

Return the same result shape as the baseline: `prediction`,
`preprocessing_total_ms`, `waviness_inference_total_ms`,
`total_raw_to_score_ms`, `timings_ms`, `image`, and `intermediate`. Candidate
code can call `reference_pipeline(photo, server)` as a starting point, but a
version that changes operations should keep those changes inside its own file.
The version ID in `experiment.json` must match the directory name.

Use validation for experiments and selection. The runner does not evaluate the
reserved test split unless `--final-test` is explicitly supplied after the
candidate is selected. The HTML viewer discovers reports across versions and
can show a single version or compare the latest reports side by side.
