from __future__ import annotations

from pathlib import Path

import step_6_a_evaluate_test as evaluation


def test_gallery_links_are_relative_to_nested_evaluation_directory(tmp_path: Path, monkeypatch) -> None:
    image_dir = tmp_path / "step-2-final-water-data"
    output_dir = tmp_path / "step-6-test-evaluation" / "checkpoint" / "benchmark-v1-test"
    output_dir.mkdir(parents=True)
    gallery_path = output_dir / "index.html"
    monkeypatch.setattr(evaluation, "IMAGE_DIR", image_dir)
    monkeypatch.setattr(evaluation, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(evaluation, "GALLERY_PATH", gallery_path)

    evaluation.write_gallery(
        [
            {
                "filename": "step-2_IMG_7321.jpg",
                "true_waviness": 0.5,
                "predicted_waviness": 0.4,
                "absolute_error": 0.1,
            }
        ]
    )

    gallery = gallery_path.read_text(encoding="utf-8")
    assert "../../step-2-final-water-data/step-2_IMG_7321.jpg" in gallery
