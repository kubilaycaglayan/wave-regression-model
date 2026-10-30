"""Read-only baseline adapter for the existing Distil-FP32 pipeline."""

VERSION_ID = "v000_baseline"


def run_pipeline(photo, server, reference_pipeline):
    """Run the current shared instrumentation with no candidate modifications."""
    return reference_pipeline(photo, server)
