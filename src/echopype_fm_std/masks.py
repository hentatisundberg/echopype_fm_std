from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .integration import _effective_bottom


def summarize_boundaries(
    surface_range: xr.DataArray,
    bottom_range: xr.DataArray,
    navigation: xr.Dataset | None = None,
    *,
    transducer_depth_m: float | None = None,
) -> pd.DataFrame:
    """Summarize detected surface and bottom boundaries for one RAW file."""
    sonar_depth = transducer_depth_m or 0.0
    surface = surface_range + sonar_depth
    bottom = bottom_range + sonar_depth
    row: dict[str, object] = {
        "surface_depth_max_m": float(surface.max(skipna=True)),
        "surface_depth_mean_m": float(surface.mean(skipna=True)),
        "surface_depth_variance_m2": float(surface.var(skipna=True, ddof=0)),
        "bottom_depth_max_m": float(bottom.max(skipna=True)),
        "bottom_depth_mean_m": float(bottom.mean(skipna=True)),
        "bottom_depth_variance_m2": float(bottom.var(skipna=True, ddof=0)),
    }
    if "ping_time" in surface_range.coords:
        row["start_time"] = pd.Timestamp(surface_range["ping_time"].values[0])
        row["end_time"] = pd.Timestamp(surface_range["ping_time"].values[-1])
    if navigation is not None:
        for name in ("latitude", "longitude"):
            if name in navigation:
                values = navigation[name]
                row[f"start_{name}"] = float(values.isel(ping_time=0))
                row[f"end_{name}"] = float(values.isel(ping_time=-1))
                row[f"mean_{name}"] = float(values.mean(skipna=True))
    return pd.DataFrame([row])


def export_boundary_summary(
    summary: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Write the one-row boundary summary CSV."""
    preferred = [
        "start_time",
        "end_time",
        "start_latitude",
        "start_longitude",
        "end_latitude",
        "end_longitude",
        "mean_latitude",
        "mean_longitude",
        "surface_depth_max_m",
        "surface_depth_mean_m",
        "surface_depth_variance_m2",
        "bottom_depth_max_m",
        "bottom_depth_mean_m",
        "bottom_depth_variance_m2",
    ]
    columns = [column for column in preferred if column in summary]
    columns.extend(column for column in summary if column not in columns)
    summary[columns].to_csv(Path(output_path), index=False)


def export_mask_echogram(
    sv: xr.Dataset,
    surface_range: xr.DataArray,
    bottom_range: xr.DataArray,
    output_path: str | Path,
    targets: xr.Dataset | None = None,
    clean_output_path: str | Path | None = None,
) -> None:
    """Export an Sv echogram with excluded surface/bottom samples shaded."""
    import matplotlib.pyplot as plt

    if "Sv" not in sv or "echo_range" not in sv:
        raise ValueError("Sv dataset must contain 'Sv' and 'echo_range'.")
    echo = sv["Sv"].isel(channel=0) if "channel" in sv["Sv"].dims else sv["Sv"]
    echo = echo.transpose("range_sample", "ping_time").load()
    ranges = sv["echo_range"]
    if "channel" in ranges.dims:
        ranges = ranges.isel(channel=0)
    ranges = ranges.transpose("range_sample", "ping_time").load()
    surface = surface_range.broadcast_like(ranges)
    sample_dim = next(
        (dim for dim in ranges.dims if dim not in {"ping_time", "channel"}),
        None,
    )
    if sample_dim is None:
        raise ValueError("echo_range must have a range-sample dimension.")
    bottom = _effective_bottom(ranges, bottom_range, sample_dim).broadcast_like(ranges)
    excluded = ~((ranges >= surface) & (ranges <= bottom))

    figure, axis = plt.subplots(figsize=(14, 7), constrained_layout=True)
    image = axis.pcolormesh(
        sv["ping_time"].values,
        ranges.values,
        echo.values,
        shading="auto",
        cmap="viridis",
        vmin=-90,
        vmax=-30,
    )
    axis.pcolormesh(
        sv["ping_time"].values,
        ranges.values,
        excluded.where(excluded).values,
        shading="auto",
        cmap="Greys",
        vmin=0,
        vmax=1,
        alpha=0.45,
    )
    axis.plot(sv["ping_time"], surface_range, color="cyan", linewidth=1.2, label="Surface")
    axis.plot(sv["ping_time"], bottom_range, color="magenta", linewidth=1.2, label="Bottom")
    if targets is not None and targets.sizes.get("single_target", 0):
        axis.scatter(
            targets["ping_time"],
            targets["single_target_range"],
            marker="+",
            color="white",
            s=36,
            linewidths=0.8,
            label="Single targets",
        )
    figure.colorbar(image, ax=axis, label="Sv (dB re 1 m-1)")
    axis.set(
        title="Echogram with surface/bottom exclusion mask",
        xlabel="Ping time",
        ylabel="Echo range (m)",
    )
    axis.invert_yaxis()
    axis.legend(loc="upper right")
    figure.savefig(Path(output_path), dpi=180)
    plt.close(figure)

    if clean_output_path is None:
        return
    depth_span = float(ranges.max() - ranges.min())
    time_span = float(
        (sv["ping_time"].values[-1] - sv["ping_time"].values[0])
        / np.timedelta64(1, "s")
    )
    pixels_per_m = 8.0
    pixels_per_second = 8.0
    width = max(2.0, time_span * pixels_per_second / 180.0)
    height = max(2.0, depth_span * pixels_per_m / 180.0)
    figure, axis = plt.subplots(figsize=(width, height), dpi=180)
    axis.pcolormesh(
        sv["ping_time"].values,
        ranges.values,
        echo.values,
        shading="auto",
        cmap="viridis",
        vmin=-90,
        vmax=-30,
    )
    axis.pcolormesh(
        sv["ping_time"].values,
        ranges.values,
        excluded.where(excluded).values,
        shading="auto",
        cmap="Greys",
        vmin=0,
        vmax=1,
        alpha=0.45,
    )
    axis.set_xlim(sv["ping_time"].values[0], sv["ping_time"].values[-1])
    axis.set_ylim(float(ranges.max()), float(ranges.min()))
    axis.axis("off")
    figure.subplots_adjust(0, 0, 1, 1)
    figure.savefig(Path(clean_output_path), dpi=180, pad_inches=0)
    plt.close(figure)
