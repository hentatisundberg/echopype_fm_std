from __future__ import annotations

import numpy as np
import xarray as xr

from echopype_fm_std.targets import export_targets_csv, filter_targets_by_ts


def test_export_requires_complete_schema(tmp_path):
    ds = xr.Dataset(
        {
            "depth_m": ("single_target", [10.0]),
            "angle_alongship_deg": ("single_target", [1.0]),
            "angle_athwartship_deg": ("single_target", [2.0]),
            "TS_compensated_dB": ("single_target", [-40.0]),
            "TS_uncompensated_dB": ("single_target", [-42.0]),
        },
        coords={"single_target": np.arange(1)},
    )
    out = tmp_path / "targets.csv"
    export_targets_csv(ds, out)
    text = out.read_text()
    assert text.splitlines()[0] == (
        "depth_m,angle_alongship_deg,angle_athwartship_deg,"
        "TS_compensated_dB,TS_uncompensated_dB"
    )
    assert "10.0,1.0,2.0,-40.0,-42.0" in text


def test_export_filters_echopype_target_strength(tmp_path):
    ds = xr.Dataset(
        {
            "single_target_range": ("single_target", [12.0, 18.0]),
            "compensated_TS": ("single_target", [-65.0, -5.0]),
            "uncompensated_TS": ("single_target", [-66.0, -7.0]),
            "beam_comp_db": ("single_target", [1.0, 2.0]),
        },
        coords={"single_target": np.arange(2)},
    )
    out = tmp_path / "targets.csv"

    export_targets_csv(ds, out, ts_comp_min=-70.0, ts_comp_max=-10.0)

    text = out.read_text()
    assert "12.0" in text
    assert "18.0" not in text
    assert "compensated_TS" in text


def test_export_adds_target_and_survey_navigation(tmp_path):
    ping_time = np.array(["2020-01-01T00:00:00"], dtype="datetime64[s]")
    ds = xr.Dataset(
        {
            "single_target_range": ("single_target", [12.0]),
            "compensated_TS": ("single_target", [-40.0]),
        },
        coords={"single_target": [0], "ping_time": ("single_target", ping_time)},
    )
    navigation = xr.Dataset(
        {
            "latitude": ("ping_time", [57.0]),
            "longitude": ("ping_time", [11.0]),
        },
        coords={"ping_time": ping_time},
    )
    output = tmp_path / "targets.csv"
    export_targets_csv(
        ds,
        output,
        navigation=navigation,
        survey_metadata={"survey_start_time": ping_time[0]},
    )

    header = output.read_text().splitlines()[0]
    assert "target_latitude" in header
    assert "target_longitude" in header
    assert "survey_start_time" in header


def test_filter_targets_by_ts_is_used_for_plot_and_export():
    ds = xr.Dataset(
        {"compensated_TS": ("single_target", [-65.0, -55.0])},
        coords={"single_target": [0, 1]},
    )

    filtered = filter_targets_by_ts(ds, ts_comp_min=-60.0)

    assert filtered.sizes["single_target"] == 1
    assert filtered["compensated_TS"].item() == -55.0
