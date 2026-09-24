from pathlib import Path

from data_sources import SourceOutputNames


def test_recovers_complete_unmapped_output_without_renaming(tmp_path: Path) -> None:
    source = tmp_path / "IMG_7395.HEIC"
    source.write_bytes(b"source")
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    for suffix in ("_original.jpg", "_mask.png", "_overlay.jpg"):
        (output_dir / f"IMG_7395{suffix}").write_bytes(b"existing")

    names = SourceOutputNames(
        [source],
        output_dir,
        ("_original.jpg", "_mask.png", "_overlay.jpg"),
        output_paths_for_stem=lambda stem: tuple(output_dir / f"{stem}{suffix}" for suffix in ("_original.jpg", "_mask.png", "_overlay.jpg")),
    )

    assert names.stem_for(source) == "IMG_7395"


def test_ambiguous_complete_output_is_not_adopted(tmp_path: Path) -> None:
    first = tmp_path / "one" / "IMG_7395.HEIC"
    second = tmp_path / "two" / "IMG_7395.HEIC"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    for suffix in ("_original.jpg", "_mask.png", "_overlay.jpg"):
        (output_dir / f"IMG_7395{suffix}").write_bytes(b"existing")

    names = SourceOutputNames(
        [first, second],
        output_dir,
        ("_original.jpg", "_mask.png", "_overlay.jpg"),
        output_paths_for_stem=lambda stem: tuple(output_dir / f"{stem}{suffix}" for suffix in ("_original.jpg", "_mask.png", "_overlay.jpg")),
    )

    assert names.stem_for(first).startswith("IMG_7395_")


def test_save_writes_manifest_and_keeps_mapping_after_reload(tmp_path: Path) -> None:
    source = tmp_path / "IMG_7395.HEIC"
    source.write_bytes(b"source")
    output_dir = tmp_path / "outputs"
    names = SourceOutputNames([source], output_dir, (".jpg",))
    assert names.stem_for(source) == "IMG_7395"
    names.save()

    reloaded = SourceOutputNames([source], output_dir, (".jpg",))
    assert reloaded.stem_for(source) == "IMG_7395"
