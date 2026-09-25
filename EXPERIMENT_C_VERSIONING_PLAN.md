# Experiment C — Versioned Output Structure Plan

## Recommended Structure

Use immutable version directories under the Experiment C root:

```text
step-10-experiment-c-frozen-backbone-comparison/
├── experiment_index.json
├── experiment_index.md
├── v1-20260925T123000Z/
│   ├── experiment_manifest.json
│   ├── aggregate_metrics.json
│   ├── backbone_comparison.csv
│   ├── backbone_comparison.md
│   ├── plots/
│   └── runs/
│       ├── resnet18/seed-42/
│       ├── resnet18/seed-43/
│       └── ...
└── v2-20261001T...
```

Each version represents one complete experimental configuration, including:

- split snapshot and all split hashes
- preprocessing configuration
- backbone list
- seed list
- optimizer and training settings
- code/configuration fingerprint
- creation timestamp
- completion status

The version directory should be immutable after completion.

## Run Behavior

- Automatically create the next version when starting a new experiment configuration.
- Resume only incomplete runs inside the selected version.
- Skip completed backbone/seed runs whose configuration fingerprint matches.
- Never overwrite a completed version.
- Replace the current `--force` behavior with a new-version behavior; forcing a rerun creates a new version rather than deleting an old one.
- Allow an explicit `--version` or `--run-id` for resuming interrupted experiments.
- Keep checkpoint paths relative to the version directory and record absolute paths plus SHA-256 hashes in manifests.

## Indexing

Maintain append-only root-level indexes:

- `experiment_index.json`: one record per experiment version, including status, snapshot, configuration hash, aggregate result, and report path.
- `experiment_index.md`: human-readable comparison across all versions.

This makes it possible to answer:

- Which snapshot was used?
- Which preprocessing/configuration was used?
- Which version produced a checkpoint?
- Did a later experiment improve over an earlier one?
- Which version should be considered current?

## Migration

Move the current completed Experiment C result into:

```text
step-10-experiment-c-frozen-backbone-comparison/v1-20260925T...
```

Preserve its metrics, plots, reports, run configurations, and checkpoint hashes. Remove the ambiguous unversioned aggregate files after migration, or retain them only as a clearly labeled compatibility pointer.

The ignored checkpoint files should remain local unless the repository later adopts external artifact storage or Git LFS.

## Tests

Add coverage for:

- new experiments receiving distinct version directories
- interrupted versions resuming without overwriting completed runs
- configuration changes creating a new version
- exact configuration/snapshot fingerprints being required for resume
- completed versions remaining unchanged
- index entries being appended rather than replaced
- aggregate reports being written inside the version directory
