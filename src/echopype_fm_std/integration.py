from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from echopype.commongrid import compute_NASC
from echopype.commongrid.utils import get_distance_from_latlon


def _range_name(dataset: xr.Dataset) -> str:
    for name in ("echo_range", "range"):
        if name in dataset:
            return name
    raise ValueError("Dataset must contain an 'echo_range' or 'range' coordinate.")


def _effective_bottom(
    ranges: xr.DataArray,
    bottom_range: xr.DataArray,
    sample_dim: str,
) -> xr.DataArray:
    """Use the deepest available sample when bottom detection has no line."""
    maximum_range = ranges.max(dim=sample_dim, skipna=True)
    return bottom_range.where(np.isfinite(bottom_range), maximum_range)


def apply_boundary_mask(
    dataset: xr.Dataset,
    surface_range: xr.DataArray,
    bottom_range: xr.DataArray,
    *,
    variable: str = "Sp",
) -> xr.Dataset:
    """Mask acoustic samples outside the surface-to-seafloor envelope."""
    if variable not in dataset:
        raise ValueError(f"Dataset is missing {variable!r}.")
    range_name = _range_name(dataset)
    ranges = dataset[range_name]
    surface = surface_range.rename("surface_range")
    sample_dim = next(
        (dim for dim in dataset[variable].dims if dim not in {"ping_time", "channel"}),
        None,
    )
    if sample_dim is None:
        raise ValueError(f"{variable!r} must have a range-sample dimension.")
    bottom = _effective_bottom(ranges, bottom_range, sample_dim).rename("bottom_range")
    valid = (ranges >= surface) & (ranges <= bottom)
    return dataset.assign({variable: dataset[variable].where(valid)})


def integrate_echoes(
    sv: xr.Dataset,
    surface_range: xr.DataArray,
    bottom_range: xr.DataArray,
    *,
    layer_size_m: float = 10.0,
    distance_bin_nmi: float = 0.1,
    transducer_depth_m: float | None = None,
    navigation: xr.Dataset | None = None,
) -> xr.Dataset:
    """Compute Echopype NASC in depth and horizontal-distance bins."""
    if "Sv" not in sv:
        raise ValueError("Sv dataset must contain an 'Sv' variable.")
    if layer_size_m <= 0:
        raise ValueError("layer_size_m must be greater than zero.")
    if distance_bin_nmi <= 0:
        raise ValueError("distance_bin_nmi must be greater than zero.")
    if navigation is None or not {"latitude", "longitude"} <= set(navigation.data_vars):
        raise ValueError("navigation must contain latitude and longitude.")
    range_name = _range_name(sv)
    sv_values = sv["Sv"]
    if "ping_time" not in sv_values.dims:
        raise ValueError("Sv must have a 'ping_time' dimension.")
    sample_dim = next(
        (dim for dim in sv_values.dims if dim not in {"ping_time", "channel"}),
        None,
    )
    if sample_dim is None:
        raise ValueError("Sv must have a range-sample dimension.")
    if "channel" not in sv_values.dims:
        sv = sv.expand_dims(channel=[0])
    else:
        sv = sv.copy()
    if "frequency_nominal" not in sv:
        sv["frequency_nominal"] = xr.DataArray(
            np.full(sv.sizes["channel"], np.nan),
            dims=("channel",),
            coords={"channel": sv["channel"]},
        )
    sv_values = sv["Sv"]
    ranges = sv[range_name]
    if ranges.ndim == 1:
        ranges = ranges.broadcast_like(sv_values)
    surface = surface_range.broadcast_like(sv_values)
    bottom = _effective_bottom(ranges, bottom_range, sample_dim).broadcast_like(sv_values)
    # Convert the acoustic depth coordinate to depth below the sea surface.
    relative_depth = ranges - (transducer_depth_m or 0.0)
    valid = (
        np.isfinite(sv_values)
        & np.isfinite(relative_depth)
        & (ranges >= surface)
        & (ranges <= bottom)
    )

    masked_sv = sv_values.where(valid)
    sv = sv.assign(Sv=masked_sv)
    sv["depth"] = relative_depth
    for name in ("latitude", "longitude"):
        sv[name] = navigation[name].broadcast_like(sv["ping_time"])
    result = compute_NASC(
        sv,
        range_bin=f"{layer_size_m:g}m",
        dist_bin=f"{distance_bin_nmi:g}nmi",
    ).isel(channel=0)
    result = result.rename({"depth": "layer"})
    layer_values = result["layer"].values
    result = result.assign_coords(
        layer_start_depth_m=("layer", layer_values),
        layer_end_depth_m=("layer", layer_values + layer_size_m),
    ).rename({"NASC": "NASC_m2_nmi2"})
    metric_values = sv["Sv"].isel(channel=0).values
    valid_values = valid.isel(channel=0).values
    depth_values = (
        relative_depth.isel(channel=0).values
        if "channel" in relative_depth.dims
        else relative_depth.values
    )
    distance_values = np.asarray(get_distance_from_latlon(sv))
    ping_indices, sample_indices = np.nonzero(valid_values)
    metric_frame = pd.DataFrame(
        {
            "distance": np.floor(
                distance_values[ping_indices] / distance_bin_nmi
            )
            * distance_bin_nmi,
            "layer": np.floor(depth_values[ping_indices, sample_indices] / layer_size_m)
            * layer_size_m,
            "ping": ping_indices,
            "nasc_per_ping": (
                10.0 ** (metric_values[ping_indices, sample_indices] / 10.0)
                * 4
                * np.pi
                * 1852**2
                * 0.1
            ),
        }
    )
    metric_frame["distance"] = metric_frame["distance"].round(10)
    metric_frame["layer"] = metric_frame["layer"].round(10)
    grouped_metrics = metric_frame.groupby(["distance", "layer"], sort=False)
    metrics = grouped_metrics.agg(
        n_samples=("nasc_per_ping", "size"),
        n_pings=("ping", "nunique"),
        NASC_variance_m4_nmi4=("nasc_per_ping", "var"),
    )
    metrics["NASC_variance_m4_nmi4"] = metrics["NASC_variance_m4_nmi4"].fillna(0.0)
    metric_index = pd.MultiIndex.from_product(
        [result["distance"].values, result["layer"].values],
        names=("distance", "layer"),
    )
    metrics = metrics.reindex(metric_index)
    for name in ("n_samples", "n_pings", "NASC_variance_m4_nmi4"):
        result[name] = xr.DataArray(
            metrics[name].fillna(0).to_numpy().reshape(
                result.sizes["distance"], result.sizes["layer"]
            ),
            dims=("distance", "layer"),
            coords={"distance": result["distance"], "layer": result["layer"]},
        )
    result.attrs.update(
        {
            "layer_reference": "sea_surface",
            "layer_size_m": layer_size_m,
            "distance_bin_nmi": distance_bin_nmi,
            "nasc_units": "m2 nmi-2",
            "nasc_formula": "echopype.commongrid.compute_NASC",
        }
    )
    result["ping_time_start"] = sv["ping_time"].min()
    result["ping_time_end"] = sv["ping_time"].max()
    return result


def export_integration_csv(result: xr.Dataset, output_path: str | Path) -> None:
    """Write one row per distance/depth cell, including survey summaries."""
    frame = result.to_dataframe().reset_index()
    columns = [
        "ping_time_start",
        "ping_time_end",
        "ping_time",
        "latitude",
        "longitude",
        "distance",
        "layer_start_depth_m",
        "layer_end_depth_m",
        "NASC_m2_nmi2",
        "NASC_variance_m4_nmi4",
        "n_pings",
        "n_samples",
    ]
    frame = frame[[column for column in columns if column in frame]]
    if "n_samples" in frame:
        frame = frame[frame["n_samples"] > 0]
    frame.to_csv(Path(output_path), index=False)
