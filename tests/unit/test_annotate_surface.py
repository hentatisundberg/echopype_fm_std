from __future__ import annotations

from scripts.annotate_surface import _resolve_paths


def test_resolve_paths_excludes_raw_files_with_existing_annotations(tmp_path):
    annotated_raw = tmp_path / "annotated.raw"
    new_raw = tmp_path / "new.raw"
    annotated_raw.touch()
    new_raw.touch()
    output_dir = tmp_path / "annotations"
    output_dir.mkdir()
    (output_dir / "annotated_surface.csv").touch()

    assert _resolve_paths([annotated_raw, new_raw], None, output_dir) == [new_raw]


def test_resolve_paths_excludes_raw_files_with_existing_json_sidecar(tmp_path):
    raw_path = tmp_path / "annotated.raw"
    raw_path.touch()
    output_dir = tmp_path / "annotations"
    output_dir.mkdir()
    (output_dir / "annotated_surface.json").touch()

    assert _resolve_paths([raw_path], None, output_dir) == []
