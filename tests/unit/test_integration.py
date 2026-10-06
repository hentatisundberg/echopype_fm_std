import numpy as np
import xarray as xr

from echopype_fm_std.integration import (
    apply_boundary_mask,
    export_integration_csv,
    integrate_echoes,
)


def _dataset() -> tuple[xr.Dataset, xr.DataArray, xr.DataArray]:
    ping = np.datetime64("2026-01-01") + np.arange(2).astype("timedelta64[s]")
    sample = np.arange(6)
    echo_range = xr.DataArray(
        np.tile(np.arange(1, 7), (2, 1)),
        dims=("ping_time", "range_sample"),
        coords={"ping_time": ping, "range_sample": sample},
    )
    sv_values = np.full((1, 2, 6), -60.0)
    sv_values[0, 1, :] = -59.0
    sv = xr.Dataset(
        {
            "Sv": (("channel", "ping_time", "range_sample"), sv_values),
            "frequency_nominal": ("channel", [38000.0]),
        },
        coords={"ping_time": ping, "range_sample": sample, "echo_range": echo_range},
    )
    surface = xr.DataArray([1.0, 2.0], dims="ping_time", coords={"ping_time": ping})
    bottom = xr.DataArray([6.0, 6.0], dims="ping_time", coords={"ping_time": ping})
    return sv, surface, bottom


def _navigation(sv: xr.Dataset) -> xr.Dataset:
    return xr.Dataset(
        {
            "latitude": ("ping_time", [57.0, 57.002]),
            "longitude": ("ping_time", [11.0, 11.002]),
        },
        coords={"ping_time": sv.ping_time},
    )


def test_integration_uses_surface_relative_layers_and_boundary_mask():
    sv, surface, bottom = _dataset()
    result = integrate_echoes(
        sv,
        surface,
        bottom,
        layer_size_m=2,
        transducer_depth_m=1.0,
        navigation=_navigation(sv),
    )

    assert result.sizes["layer"] == 3
    assert result.sizes["distance"] == 2
    assert result["layer_start_depth_m"].values.tolist() == [0, 2, 4]
    assert result["NASC_m2_nmi2"].dims == ("distance", "layer")
    assert result["n_samples"].sum() == 11
    assert result["n_pings"].max() == 2
    assert result["NASC_variance_m4_nmi4"].max() > 0


def test_boundary_mask_applies_to_sp():
    sv, surface, bottom = _dataset()
    sp = sv.rename({"Sv": "Sp"})
    masked = apply_boundary_mask(sp, surface, bottom)
    assert np.isfinite(masked["Sp"].isel(ping_time=0, range_sample=0))
    assert np.isnan(masked["Sp"].isel(ping_time=1, range_sample=0))
    assert np.isfinite(masked["Sp"].isel(ping_time=0, range_sample=1))


def test_missing_bottom_keeps_samples_to_the_end_of_the_echogram():
    sv, surface, bottom = _dataset()
    sp = sv.rename({"Sv": "Sp"})
    bottom = bottom.where(bottom < 0)
    masked = apply_boundary_mask(sp, surface, bottom)

    assert np.isfinite(masked["Sp"].isel(ping_time=0, range_sample=5))
    assert np.isfinite(masked["Sp"].isel(ping_time=1, range_sample=5))


def test_integration_csv_contains_navigation(tmp_path):
    sv, surface, bottom = _dataset()
    navigation = xr.Dataset(
        {
            "latitude": ("ping_time", [57.0, 57.002]),
            "longitude": ("ping_time", [11.0, 11.002]),
        },
        coords={"ping_time": sv.ping_time},
    )
    result = integrate_echoes(sv, surface, bottom, layer_size_m=2, navigation=navigation)
    output = tmp_path / "integration.csv"
    export_integration_csv(result, output)
    header = output.read_text().splitlines()[0]
    assert header.startswith("ping_time_start,ping_time_end,ping_time,latitude,longitude")
    lines = output.read_text().splitlines()
    assert len(lines) == 4
    assert "n_samples" in lines[0]
    assert all(float(row.rsplit(",", 1)[-1]) > 0 for row in lines[1:])


def test_integration_aggregates_all_pings_into_one_row_per_layer():
    sv, surface, bottom = _dataset()
    result = integrate_echoes(
        sv, surface, bottom, layer_size_m=1, navigation=_navigation(sv)
    )

    assert result.sizes["layer"] == 6
    assert result.sizes["distance"] == 2


def test_integration_csv_excludes_empty_cells(tmp_path):
    sv, surface, bottom = _dataset()
    bottom = xr.full_like(bottom, 2.0)
    result = integrate_echoes(
        sv,
        surface,
        bottom,
        layer_size_m=1,
        navigation=_navigation(sv),
    )
    output = tmp_path / "integration.csv"
    export_integration_csv(result, output)

    assert len(output.read_text().splitlines()) == 3


def test_integration_handles_channel_dependent_range():
    sv, surface, bottom = _dataset()
    sv = sv.assign_coords(
        echo_range=(
            ("channel", "ping_time", "range_sample"),
            sv["echo_range"].values[np.newaxis, :, :],
        )
    )

    result = integrate_echoes(sv, surface, bottom, navigation=_navigation(sv))

    assert result["n_samples"].sum() > 0
