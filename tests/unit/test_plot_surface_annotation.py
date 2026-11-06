from __future__ import annotations

from scripts.plot_surface_annotation import (
    _resolve_annotation_path,
    _resolve_raw_paths,
)


def test_resolve_raw_paths_accepts_files_and_folders(tmp_path):
    first = tmp_path / "first.raw"
    second = tmp_path / "second.raw"
    first.touch()
    second.touch()

    assert _resolve_raw_paths([first], [tmp_path]) == [first, second]


def test_resolve_annotation_path_uses_raw_stem_for_directory(tmp_path):
    raw_path = tmp_path / "survey.raw"
    annotation_dir = tmp_path / "annotations"
    annotation_dir.mkdir()

    assert _resolve_annotation_path(raw_path, annotation_dir) == (
        annotation_dir / "survey_surface.csv"
    )
