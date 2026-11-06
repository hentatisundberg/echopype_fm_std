from __future__ import annotations

import csv
import json

import numpy as np

from echopype_fm_std.annotation import (
    interpolate_surface_vertices,
    load_surface_annotation,
    save_surface_annotation,
)


def test_interpolate_surface_vertices_leaves_edges_unlabeled():
    line = interpolate_surface_vertices([2, 6], [10, 14], 9)

    np.testing.assert_array_equal(line[:2], [np.nan, np.nan])
    np.testing.assert_allclose(line[2:7], [10, 11, 12, 13, 14])
    np.testing.assert_array_equal(line[7:], [np.nan, np.nan])


def test_save_surface_annotation_writes_optimizer_friendly_csv_and_metadata(tmp_path):
    csv_path, metadata_path = save_surface_annotation(
        tmp_path / "sample_surface.csv",
        raw_path="/tmp/sample.raw",
        ping_times=["2026-01-01T00:00:00", "2026-01-01T00:01:00"],
        surface_ranges_m=[1.5, np.nan],
        vertices=[(0, 1.5), (1, 2.0)],
        channel="ch0",
    )

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["surface_range_m"] == "1.5"
    assert rows[1]["surface_range_m"] == ""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["format"].endswith("v1")
    assert metadata["channel"] == "ch0"
    loaded = load_surface_annotation(csv_path)
    np.testing.assert_array_equal(loaded["ping_index"], [0, 1])
    np.testing.assert_array_equal(loaded["annotated"], [True, False])
    np.testing.assert_array_equal(loaded["surface_range_m"], [1.5, np.nan])
