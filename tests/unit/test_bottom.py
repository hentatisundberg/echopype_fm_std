from unittest.mock import patch

import numpy as np
import xarray as xr

from echopype_fm_std.bottom import (
    detect_bottom,
    postprocess_bottom_line,
    preprocess_bottom_dataset,
    refine_bottom_line,
)


def _dataset() -> xr.Dataset:
    return xr.Dataset(
        {
            "Sv": (("channel", "ping_time", "range_sample"), [[[-60, -40, -80], [-70, -30, -80]]]),
            "angle_alongship": (
                ("channel", "ping_time", "range_sample"),
                [[[0, 0, 0], [0, 0, 0]]],
            ),
            "angle_athwartship": (
                ("channel", "ping_time", "range_sample"),
                [[[0, 0, 0], [0, 0, 0]]],
            ),
            "echo_range": (("channel", "ping_time", "range_sample"), [[[1, 2, 3], [1, 2, 3]]]),
        },
        coords={"channel": ["ch0"], "ping_time": [0, 1], "range_sample": [0, 1, 2]},
    )


def test_blackwell_adds_depth_and_masks_no_intercept_sentinel():
    returned = xr.DataArray([1.7, 2.2], dims=["ping_time"], coords={"ping_time": [0, 1]})
    with patch("echopype_fm_std.bottom.ep.mask.detect_seafloor", return_value=returned) as detect:
        out = detect_bottom(
            _dataset(),
            params={"threshold": [-65, 1, 1], "r0": 2, "offset": 0.3},
        )

    dataset_arg = detect.call_args.args[0]
    assert "depth" in dataset_arg
    assert detect.call_args.kwargs["method"] == "blackwell"
    assert "offset" not in detect.call_args.kwargs["params"]
    assert out.name == "bottom_range"
    np.testing.assert_allclose(out.values, [np.nan, 2.5], equal_nan=True)


def test_bottom_requires_channel_for_multiple_channels():
    dataset = xr.concat([_dataset(), _dataset()], dim="channel").assign_coords(
        channel=["ch0", "ch1"]
    )
    with patch("echopype_fm_std.bottom.ep.mask.detect_seafloor"):
        try:
            detect_bottom(dataset)
        except ValueError as exc:
            assert "channel is required" in str(exc)
        else:
            raise AssertionError("expected channel validation")


def test_preprocess_max_does_not_mutate_angles_or_input():
    dataset = _dataset()
    out = preprocess_bottom_dataset(
        dataset,
        options={"method": "max", "window": 3},
    )

    np.testing.assert_array_equal(
        out["Sv"].values,
        [[[ -60, -30, -80], [-60, -30, -80]]],
    )
    np.testing.assert_array_equal(out["angle_alongship"], dataset["angle_alongship"])
    np.testing.assert_array_equal(dataset["Sv"].values, [[[-60, -40, -80], [-70, -30, -80]]])


def test_preprocess_fills_only_bounded_internal_dropout():
    dataset = xr.concat([_dataset()] * 3, dim="ping_time")
    dataset = dataset.assign_coords(ping_time=range(6))
    dataset["Sv"] = dataset["Sv"].astype(float)
    dataset["Sv"].loc[{"ping_time": 2}] = np.nan
    out = preprocess_bottom_dataset(
        dataset,
        options={"method": "fill_dropouts", "max_gap": 1},
    )

    assert np.isfinite(out["Sv"].isel(ping_time=1)).all()


def test_preprocess_median_max_suppresses_isolated_peak_before_max():
    dataset = xr.concat([_dataset()] * 3, dim="ping_time").astype(float)
    dataset = dataset.assign_coords(ping_time=range(6))
    dataset["Sv"].loc[{"ping_time": 2}] = dataset["Sv"].isel(ping_time=2) + 30

    out = preprocess_bottom_dataset(
        dataset,
        options={"method": "median_max", "median_window": 3, "max_window": 3},
    )

    expected = dataset["Sv"].rolling(ping_time=3, center=True, min_periods=1).median()
    expected = expected.rolling(ping_time=3, center=True, min_periods=1).max()
    np.testing.assert_allclose(out["Sv"], expected)


def test_preprocess_rolling_fast_path_matches_xarray_reference():
    rng = np.random.default_rng(1)
    values = rng.normal(size=(2, 31, 17))
    values[rng.random(values.shape) < 0.08] = np.nan
    dataset = xr.Dataset(
        {"Sv": (("channel", "ping_time", "range_sample"), values)}
    )

    for method, method_options in (
        ("max", {"window": 5}),
        ("median", {"window": 5}),
        ("median_max", {"median_window": 5, "max_window": 3}),
    ):
        options = {"method": method, "min_periods": 1, **method_options}
        actual = preprocess_bottom_dataset(dataset, options=options)["Sv"]
        if method == "max":
            expected = dataset["Sv"].rolling(
                ping_time=5, center=True, min_periods=1
            ).max()
        elif method == "median":
            expected = dataset["Sv"].rolling(
                ping_time=5, center=True, min_periods=1
            ).median()
        else:
            expected = dataset["Sv"].rolling(
                ping_time=5, center=True, min_periods=1
            ).median()
            expected = expected.rolling(
                ping_time=3, center=True, min_periods=1
            ).max()
        np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_postprocess_replaces_isolated_spike_and_fills_short_gap():
    bottom = xr.DataArray(
        [50.0, 51.0, 20.0, 53.0, np.nan, 55.0, 56.0],
        dims=["ping_time"],
        coords={"ping_time": range(7)},
    )
    out = postprocess_bottom_line(
        bottom,
        options={
            "method": "median_interpolate",
            "window": 3,
            "max_deviation_m": 10,
            "max_gap": 1,
        },
    )

    np.testing.assert_allclose(out.values, [50, 51, 51, 53, 54, 55, 56])


def test_postprocess_max_uses_shallowest_local_bottom():
    bottom = xr.DataArray(
        [50.0, 52.0, 54.0, 52.0, 50.0],
        dims=["ping_time"],
        coords={"ping_time": range(5)},
    )

    out = postprocess_bottom_line(
        bottom,
        options={"method": "max", "window": 3},
    )

    np.testing.assert_allclose(out.values, [50, 50, 52, 50, 50])


def test_postprocess_max_interpolate_fills_short_gaps():
    bottom = xr.DataArray(
        [50.0, np.nan, 54.0, 52.0, 50.0],
        dims=["ping_time"],
        coords={"ping_time": range(5)},
    )

    out = postprocess_bottom_line(
        bottom,
        options={"method": "max_interpolate", "window": 3, "max_gap": 1},
    )

    np.testing.assert_allclose(out.values, [50, 50, 52, 50, 50])


def test_postprocess_linearly_extrapolates_edges():
    bottom = xr.DataArray(
        [np.nan, np.nan, 50.0, 52.0, 54.0, np.nan],
        dims=["ping_time"],
        coords={"ping_time": range(6)},
    )

    out = postprocess_bottom_line(
        bottom,
        options={
            "method": "median_interpolate",
            "window": 1,
            "max_gap": 1,
            "edge_fill": "linear",
        },
    )

    np.testing.assert_allclose(out.values, [46, 48, 50, 52, 54, 56])


def test_postprocess_can_bound_edge_extrapolation():
    bottom = xr.DataArray(
        [np.nan, np.nan, 50.0, 52.0],
        dims=["ping_time"],
        coords={"ping_time": range(4)},
    )

    out = postprocess_bottom_line(
        bottom,
        options={
            "method": "median_interpolate",
            "window": 1,
            "max_gap": 1,
            "edge_fill": "nearest",
            "max_edge_gap": 1,
        },
    )

    assert np.isnan(out.values[0])
    assert np.isnan(out.values[1])


def test_refine_bottom_line_selects_shallowest_qualifying_sample():
    dataset = xr.Dataset(
        {
            "Sv": (("channel", "ping_time", "range_sample"), [[[-60, -30, -20, -25]]]),
            "echo_range": (
                ("channel", "ping_time", "range_sample"),
                [[[48, 49, 50, 51]]],
            ),
        },
        coords={"channel": ["ch0"], "ping_time": [0], "range_sample": range(4)},
    )
    bottom = xr.DataArray([50.0], dims=["ping_time"], coords={"ping_time": [0]})

    out = refine_bottom_line(
        dataset,
        bottom,
        options={"enabled": True, "window_m": 2, "threshold_db": -35},
    )

    assert out.item() == 49.0


def test_refine_bottom_line_leaves_unqualified_line_unchanged():
    dataset = _dataset().astype(float)
    bottom = xr.DataArray([2.0, 2.0], dims=["ping_time"], coords={"ping_time": [0, 1]})

    out = refine_bottom_line(
        dataset,
        bottom,
        options={"enabled": True, "window_m": 0.5, "threshold_db": -20},
    )

    np.testing.assert_allclose(out, bottom)


def test_refine_bottom_line_never_moves_deeper():
    dataset = xr.Dataset(
        {
            "Sv": (("channel", "ping_time", "range_sample"), [[[-60, -20, -20]]]),
            "echo_range": (
                ("channel", "ping_time", "range_sample"),
                [[[49, 50, 51]]],
            ),
        },
        coords={"channel": ["ch0"], "ping_time": [0], "range_sample": range(3)},
    )
    bottom = xr.DataArray([50.0], dims=["ping_time"], coords={"ping_time": [0]})

    out = refine_bottom_line(
        dataset,
        bottom,
        options={"enabled": True, "window_m": 2, "threshold_db": -35},
    )

    assert out.item() == 50.0


def test_detect_bottom_keeps_pre_and_postprocessing_out_of_echopype_params():
    returned = xr.DataArray([2.2], dims=["ping_time"], coords={"ping_time": [0]})
    with patch("echopype_fm_std.bottom.ep.mask.detect_seafloor", return_value=returned) as detect:
        detect_bottom(
            _dataset().isel(ping_time=slice(0, 1)),
            params={
                "r0": 1,
                "preprocessing": {"method": "median", "window": 1},
                "postprocessing": {"method": "median", "window": 1},
            },
        )

    assert "preprocessing" not in detect.call_args.kwargs["params"]
    assert "postprocessing" not in detect.call_args.kwargs["params"]
