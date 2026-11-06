import numpy as np
import xarray as xr

from echopype_fm_std.masks import (
    export_boundary_line,
    export_boundary_summary,
    export_mask_echogram,
    summarize_boundaries,
)


def test_boundary_summary_reports_depth_statistics_and_positions():
    ping_time = np.array(["2020-01-01T00:00:00", "2020-01-01T00:00:01"], dtype="datetime64[s]")
    surface = xr.DataArray([2.0, 4.0], dims="ping_time", coords={"ping_time": ping_time})
    bottom = xr.DataArray([40.0, 60.0], dims="ping_time", coords={"ping_time": ping_time})
    navigation = xr.Dataset(
        {
            "latitude": ("ping_time", [57.0, 58.0]),
            "longitude": ("ping_time", [11.0, 12.0]),
        },
        coords={"ping_time": ping_time},
    )

    summary = summarize_boundaries(
        surface,
        bottom,
        navigation,
        transducer_depth_m=1.0,
    ).iloc[0]

    assert summary["surface_depth_max_m"] == 5.0
    assert summary["surface_depth_mean_m"] == 4.0
    assert summary["bottom_depth_variance_m2"] == 100.0
    assert summary["start_latitude"] == 57.0
    assert summary["end_longitude"] == 12.0


def test_boundary_summary_places_time_and_position_first(tmp_path):
    ping_time = np.array(["2020-01-01T00:00:00"], dtype="datetime64[s]")
    surface = xr.DataArray([2.0], dims="ping_time", coords={"ping_time": ping_time})
    bottom = xr.DataArray([40.0], dims="ping_time", coords={"ping_time": ping_time})
    output = tmp_path / "mask.csv"
    export_boundary_summary(summarize_boundaries(surface, bottom), output)

    assert output.read_text().splitlines()[0].startswith("start_time,end_time")


def test_export_boundary_line_writes_one_row_per_ping(tmp_path):
    ping_time = np.array(
        ["2020-01-01T00:00:00", "2020-01-01T00:00:01"],
        dtype="datetime64[s]",
    )
    surface = xr.DataArray(
        [2.0, np.nan],
        dims="ping_time",
        coords={"ping_time": ping_time},
    )
    output = tmp_path / "surface.csv"

    export_boundary_line(surface, output, value_name="surface_range_m")

    lines = output.read_text().splitlines()
    assert lines[0] == "ping_time,surface_range_m"
    assert len(lines) == 3
    assert lines[2].endswith(",")


def test_export_mask_echogram_writes_masked_and_raw_clean_images(tmp_path):
    ping_time = np.array(
        ["2020-01-01T00:00:00", "2020-01-01T00:00:01"],
        dtype="datetime64[s]",
    )
    ranges = np.array([1.0, 2.0, 3.0])
    sv = xr.Dataset(
        {
            "Sv": (("range_sample", "ping_time"), np.full((3, 2), -50.0)),
            "echo_range": (("range_sample", "ping_time"), np.tile(ranges[:, None], (1, 2))),
        },
        coords={"ping_time": ping_time},
    )
    surface = xr.DataArray([1.0, 1.0], dims="ping_time", coords={"ping_time": ping_time})
    bottom = xr.DataArray([3.0, 3.0], dims="ping_time", coords={"ping_time": ping_time})
    masked = tmp_path / "masked.png"
    clean = tmp_path / "clean.png"
    raw_clean = tmp_path / "raw_clean.png"

    export_mask_echogram(
        sv,
        surface,
        bottom,
        masked,
        clean_output_path=clean,
        raw_clean_output_path=raw_clean,
    )

    assert masked.exists()
    assert clean.exists()
    assert raw_clean.exists()
