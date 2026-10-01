from __future__ import annotations

import numpy as np
import xarray as xr

from echopype_fm_std.surface import detect_surface


def test_surface_detector_returns_coordinate_and_does_not_mutate():
    data = np.array(
        [
            [-20, -20, -60, -60, -60, -20],
            [-20, -20, -20, -20, -60, -60],
        ],
        dtype=float,
    )
    original = data.copy()
    sv = xr.DataArray(
        data,
        dims=("ping_time", "range_sample"),
        coords={"ping_time": [0, 1], "range_sample": np.arange(6)},
        name="Sv",
    )

    out = detect_surface(sv, threshold_db=-50, consecutive_samples=3)

    np.testing.assert_array_equal(out.values, [2.0, np.nan])
    np.testing.assert_array_equal(sv.values, original)
