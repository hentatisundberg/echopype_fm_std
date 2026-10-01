from __future__ import annotations

import numpy as np
import xarray as xr

from echopype_fm_std.targets import export_targets_csv


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
    assert text.splitlines()[0] == "depth_m,angle_alongship_deg,angle_athwartship_deg,TS_compensated_dB,TS_uncompensated_dB"
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
