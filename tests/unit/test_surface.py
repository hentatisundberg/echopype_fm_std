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


def test_surface_detector_supports_echopype_style_params_and_dataset():
    sv = xr.Dataset(
        {
            "Sv": (
                ("channel", "ping_time", "range_sample"),
                [[[-20, -60, -60], [-20, -20, -20]]],
            )
        },
        coords={"channel": ["ch0"], "ping_time": [0, 1], "range_sample": [1.0, 2.0, 3.0]},
    )

    out = detect_surface(
        sv,
        method="threshold",
        params={"threshold_db": -50, "consecutive_samples": 2},
    )

    np.testing.assert_array_equal(out.values, [2.0, np.nan])


def test_surface_detector_requires_channel_for_multiple_channels():
    sv = xr.DataArray(
        np.zeros((2, 1, 3)),
        dims=("channel", "ping_time", "range_sample"),
        coords={"channel": ["a", "b"], "ping_time": [0], "range_sample": [1, 2, 3]},
    )

    with np.testing.assert_raises(ValueError):
        detect_surface(sv)


def test_surface_detector_rejects_unknown_method():
    sv = xr.DataArray(
        [[-60, -60]],
        dims=("ping_time", "range_sample"),
        coords={"ping_time": [0], "range_sample": [1, 2]},
    )

    with np.testing.assert_raises(ValueError):
        detect_surface(sv, method="blackwell")


def test_surface_detector_coarsens_range_and_uses_echo_range_coordinate():
    sv = xr.DataArray(
        [[-20, -60, -60, -60, -60, -60]],
        dims=("ping_time", "range_sample"),
        coords={
            "ping_time": [0],
            "range_sample": np.arange(6),
            "echo_range": ("range_sample", np.arange(6, dtype=float) * 0.5),
        },
    )

    out = detect_surface(
        sv,
        params={"range_bin_fraction": 0.5, "consecutive_samples": 1},
    )

    np.testing.assert_array_equal(out.values, [1.25])
    assert out.attrs["range_bin_size"] == 2


def test_surface_detector_uses_original_wave_thresholds():
    sv = xr.DataArray(
        [
            [-20, -20, -20, -20, -20, -20, -20, -20],
            [-20, -20, -20, -20, -20, -20, -20, -20],
            [-20, -20, -20, -20, -20, -20, -20, -20],
            [-20, -20, -20, -20, -20, -20, -20, -20],
        ],
        dims=("ping_time", "range_sample"),
        coords={"ping_time": [0, 1, 2, 3], "range_sample": np.arange(8)},
    )
    sv.loc[dict(range_sample=[3, 4, 5])] = -78

    out = detect_surface(
        sv,
        params={
            "range_bin_fraction": 1,
            "beam_dead_zone_samples": 1,
            "wave_consecutive_samples": 3,
            "use_original_wave_logic": True,
            "layer_consecutive_pings": 100,
            "layer_minimum_samples": 100,
        },
    )

    np.testing.assert_array_equal(out.values, [3, 3, 3, 3])
    assert out.attrs["threshold_db"] == -77


def test_surface_detector_expands_boundary_with_centered_rolling_max():
    sv = xr.DataArray(
        [
            [-20, -60, -60, -20, -20],
            [-20, -20, -20, -60, -60],
            [-20, -60, -60, -60, -20],
        ],
        dims=("ping_time", "range_sample"),
        coords={"ping_time": [0, 1, 2], "range_sample": np.arange(5)},
    )

    out = detect_surface(
        sv,
        params={
            "threshold_db": -50,
            "consecutive_samples": 2,
            "rolling_ping_window": 3,
        },
    )

    np.testing.assert_array_equal(out.values, [3.0, 3.0, 3.0])
    assert out.attrs["rolling_boundary"] == "max"
