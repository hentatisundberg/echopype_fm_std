import numpy as np
import xarray as xr

from echopype_fm_std.masks import export_boundary_summary, summarize_boundaries


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
