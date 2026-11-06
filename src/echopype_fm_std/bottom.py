from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import echopype as ep
import numpy as np
import xarray as xr

try:
    import bottleneck as bn
except ImportError:  # pragma: no cover - xarray remains the reference fallback
    bn = None


def _rolling_in_memory(
    sv: xr.DataArray,
    *,
    window: int,
    min_periods: int,
    operation: str,
) -> xr.DataArray | None:
    """Apply centered odd-window rolling with Bottleneck when safely possible."""
    if (
        bn is None
        or window % 2 == 0
        or not isinstance(sv.data, np.ndarray)
        or not np.issubdtype(sv.dtype, np.number)
    ):
        return None
    values = sv.data
    axis = sv.get_axis_num("ping_time")
    half_window = window // 2
    if half_window:
        padding_shape = list(values.shape)
        padding_shape[axis] = half_window
        padding_dtype = np.result_type(values.dtype, np.float32)
        padding = np.full(padding_shape, np.nan, dtype=padding_dtype)
        if values.dtype != padding_dtype:
            values = values.astype(padding_dtype)
        padded = np.concatenate((padding, values, padding), axis=axis)
    else:
        padded = values
    rolling = getattr(bn, f"move_{operation}")(
        padded,
        window=window,
        min_count=min_periods,
        axis=axis,
    )
    start = window - 1
    end = start + values.shape[axis]
    result = np.take(rolling, np.arange(start, end), axis=axis)
    return sv.copy(data=result)


def preprocess_bottom_dataset(
    dataset: xr.Dataset,
    *,
    options: Mapping[str, object] | None = None,
) -> xr.Dataset:
    """Prepare only the Sv input for Blackwell bottom detection.

    Odd centered rolling windows use Bottleneck for in-memory NumPy-backed
    arrays; xarray remains the reference path for other inputs.
    """
    if not isinstance(dataset, xr.Dataset):
        raise TypeError("dataset must be an xarray.Dataset")
    if options is not None and not isinstance(options, Mapping):
        raise TypeError("options must be a mapping")
    settings = dict(options or {})
    method = str(settings.get("method", "none")).lower()
    if method == "none":
        return dataset
    if "Sv" not in dataset:
        raise ValueError("bottom preprocessing requires an 'Sv' variable")
    sv = dataset["Sv"]
    if "ping_time" not in sv.dims:
        raise ValueError("bottom preprocessing requires a 'ping_time' dimension")
    window = int(settings.get("window", 1))
    if window < 1:
        raise ValueError("bottom preprocessing window must be >= 1")
    if method in {"max", "median"}:
        min_periods = int(settings.get("min_periods", 1))
        smoothed = _rolling_in_memory(
            sv,
            window=window,
            min_periods=min_periods,
            operation=method,
        )
        if smoothed is None:
            rolling = sv.rolling(
                ping_time=window,
                center=True,
                min_periods=min_periods,
            )
            smoothed = rolling.max() if method == "max" else rolling.median()
    elif method == "median_max":
        median_window = int(settings.get("median_window", 3))
        max_window = int(settings.get("max_window", 3))
        if median_window < 1 or max_window < 1:
            raise ValueError("bottom preprocessing windows must be >= 1")
        min_periods = int(settings.get("min_periods", 1))
        median_filtered = _rolling_in_memory(
            sv,
            window=median_window,
            min_periods=min_periods,
            operation="median",
        )
        if median_filtered is None:
            median_filtered = sv.rolling(
                ping_time=median_window,
                center=True,
                min_periods=min_periods,
            ).median()
            smoothed = median_filtered.rolling(
                ping_time=max_window,
                center=True,
                min_periods=min_periods,
            ).max()
        else:
            smoothed = _rolling_in_memory(
                median_filtered,
                window=max_window,
                min_periods=min_periods,
                operation="max",
            )
            if smoothed is None:
                smoothed = median_filtered.rolling(
                    ping_time=max_window,
                    center=True,
                    min_periods=min_periods,
                ).max()
    elif method == "fill_dropouts":
        limit = int(settings.get("max_gap", 0))
        if limit < 1:
            raise ValueError("bottom preprocessing max_gap must be >= 1")
        smoothed = sv.interpolate_na(
            dim="ping_time",
            method="linear",
            limit=limit,
        )
    else:
        raise ValueError(f"Unsupported bottom preprocessing method: {method!r}")
    return dataset.copy().assign(Sv=smoothed)


def postprocess_bottom_line(
    bottom: xr.DataArray,
    *,
    options: Mapping[str, object] | None = None,
) -> xr.DataArray:
    """Remove bounded isolated bottom spikes and fill bounded missing runs."""
    if not isinstance(bottom, xr.DataArray):
        raise TypeError("bottom must be an xarray.DataArray")
    if options is not None and not isinstance(options, Mapping):
        raise TypeError("options must be a mapping")
    settings = dict(options or {})
    method = str(settings.get("method", "none")).lower()
    if method == "none":
        return bottom
    if "ping_time" not in bottom.dims:
        raise ValueError("bottom postprocessing requires a 'ping_time' dimension")
    if bottom.ndim != 1:
        raise ValueError("bottom postprocessing requires a one-dimensional line")

    result = bottom.copy()
    if method not in {"max", "median", "median_interpolate", "max_interpolate"}:
        raise ValueError(f"Unsupported bottom postprocessing method: {method!r}")
    window = int(settings.get("window", 1))
    if window < 1 or window % 2 == 0:
        raise ValueError("bottom postprocessing window must be a positive odd number")
    maximum_deviation = float(settings.get("max_deviation_m", np.inf))
    if maximum_deviation <= 0:
        raise ValueError("bottom postprocessing max_deviation_m must be > 0")
    maximum_gap = int(settings.get("max_gap", 0))
    if method in {"median_interpolate", "max_interpolate"} and maximum_gap < 1:
        raise ValueError("bottom postprocessing max_gap must be >= 1")
    edge_fill = str(settings.get("edge_fill", "none")).lower()
    if edge_fill not in {"none", "nearest", "linear"}:
        raise ValueError("bottom postprocessing edge_fill must be none, nearest, or linear")
    maximum_edge_gap = int(settings.get("max_edge_gap", 0))
    if maximum_edge_gap < 0:
        raise ValueError("bottom postprocessing max_edge_gap must be >= 0")

    if window > 1:
        reference = result
        if method in {"median_interpolate", "max_interpolate"}:
            reference = reference.interpolate_na(
                dim="ping_time",
                method="linear",
                limit=maximum_gap,
            )
        rolling = reference.rolling(
            ping_time=window,
            center=True,
            min_periods=max(2, window // 2 + 1),
        )
        if method in {"max", "max_interpolate"}:
            # Range increases downward; the shallowest valid detection is the
            # conservative masking boundary.
            local_value = rolling.min()
            replace = result.notnull() & local_value.notnull()
            if method == "max_interpolate":
                replace_missing = result.isnull() & reference.notnull() & local_value.notnull()
                result = result.where(~replace_missing, local_value)
        elif method == "median":
            local_value = rolling.median()
            replace = result.notnull() & local_value.notnull()
        else:
            local_value = rolling.median()
            replace = (
                result.notnull()
                & local_value.notnull()
                & (abs(result - local_value) > maximum_deviation)
            )
        result = result.where(~replace, local_value)

    if method in {"median_interpolate", "max_interpolate"}:
        result = result.interpolate_na(
            dim="ping_time",
            method="linear",
            limit=maximum_gap,
        )
    if edge_fill != "none":
        values = np.asarray(result.values, dtype=float).copy()
        finite = np.flatnonzero(np.isfinite(values))
        if finite.size:
            first, last = int(finite[0]), int(finite[-1])
            leading_gap, trailing_gap = first, values.size - last - 1
            fill_leading = maximum_edge_gap == 0 or leading_gap <= maximum_edge_gap
            fill_trailing = maximum_edge_gap == 0 or trailing_gap <= maximum_edge_gap

            if fill_leading:
                if edge_fill == "nearest" or finite.size == 1:
                    values[:first] = values[first]
                else:
                    slope = (values[finite[1]] - values[first]) / (finite[1] - first)
                    values[:first] = values[first] + slope * (np.arange(first) - first)
            if fill_trailing:
                if edge_fill == "nearest" or finite.size == 1:
                    values[last + 1 :] = values[last]
                else:
                    slope = (values[last] - values[finite[-2]]) / (last - finite[-2])
                    values[last + 1 :] = values[last] + slope * (
                        np.arange(last + 1, values.size) - last
                    )
            result = result.copy(data=values)
    return result


def refine_bottom_line(
    dataset: xr.Dataset,
    bottom: xr.DataArray,
    *,
    options: Mapping[str, object] | None = None,
) -> xr.DataArray:
    """Redraw a bottom line from shallow qualifying Sv samples near it."""
    if not isinstance(dataset, xr.Dataset):
        raise TypeError("dataset must be an xarray.Dataset")
    if not isinstance(bottom, xr.DataArray):
        raise TypeError("bottom must be an xarray.DataArray")
    if options is not None and not isinstance(options, Mapping):
        raise TypeError("options must be a mapping")
    settings = dict(options or {})
    if not bool(settings.get("enabled", False)):
        return bottom
    if "Sv" not in dataset or "echo_range" not in dataset:
        raise ValueError("bottom refinement requires 'Sv' and 'echo_range'")
    if "ping_time" not in bottom.dims or bottom.ndim != 1:
        raise ValueError("bottom refinement requires a one-dimensional bottom line")

    window_m = float(settings.get("window_m", 2.0))
    threshold_db = float(settings.get("threshold_db", -35.0))
    if window_m <= 0:
        raise ValueError("bottom refinement window_m must be > 0")
    sv = dataset["Sv"]
    echo_range = dataset["echo_range"]
    if "channel" in sv.dims:
        channel = settings.get("channel")
        if channel is None:
            if sv.sizes["channel"] != 1:
                raise ValueError("bottom refinement channel is required for multiple channels")
            sv = sv.isel(channel=0)
            echo_range = echo_range.isel(channel=0)
        elif isinstance(channel, int):
            sv = sv.isel(channel=channel)
            echo_range = echo_range.isel(channel=channel)
        else:
            sv = sv.sel(channel=channel)
            echo_range = echo_range.sel(channel=channel)
    range_dim = next((dim for dim in sv.dims if dim != "ping_time"), None)
    if range_dim is None:
        raise ValueError("bottom refinement requires a range dimension")
    values = sv.transpose("ping_time", range_dim).to_numpy()
    ranges = echo_range.transpose("ping_time", range_dim).to_numpy()
    tracked = bottom.reindex(ping_time=sv["ping_time"])
    tracked_values = tracked.to_numpy()
    candidate = (
        np.isfinite(ranges)
        & np.isfinite(values)
        & np.isfinite(tracked_values)[:, None]
        & (ranges >= tracked_values[:, None] - window_m)
        & (ranges <= tracked_values[:, None])
        & (values >= threshold_db)
    )
    candidate_ranges = np.where(candidate, ranges, np.inf)
    refined = np.min(candidate_ranges, axis=1)
    has_candidate = np.any(candidate, axis=1)
    result = np.where(has_candidate, refined, tracked_values)
    return tracked.copy(data=result).reindex_like(bottom)


def detect_bottom(
    dataset: xr.Dataset,
    *,
    channel: str | None = None,
    method: str = "blackwell",
    params: Mapping[str, object] | None = None,
) -> xr.DataArray:
    """Detect a separate bottom range line from a calibrated Echopype dataset.

    Echopype's Blackwell detector expects ``depth`` even when the acoustic
    vertical coordinate is range. The adapter supplies that variable from
    ``echo_range`` and returns ``NaN`` for Blackwell's no-intercept sentinel.
    Optional ``preprocessing`` and ``postprocessing`` mappings are handled by
    this adapter and are not passed to Echopype. ``offset`` is applied to the
    final finite line after postprocessing and refinement.
    """
    if not isinstance(dataset, xr.Dataset):
        raise TypeError("dataset must be an xarray.Dataset")
    if params is not None and not isinstance(params, Mapping):
        raise TypeError("params must be a mapping")

    options: dict[str, Any] = dict(params or {})
    preprocessing = options.pop("preprocessing", None)
    postprocessing = options.pop("postprocessing", None)
    refinement = options.pop("refinement", None)
    offset = float(options.pop("offset", 0.0))
    if "channel" not in options:
        if channel is not None:
            options["channel"] = channel
        elif "channel" in dataset.coords and dataset.sizes["channel"] == 1:
            options["channel"] = str(dataset["channel"].values[0])
        else:
            raise ValueError("channel is required when the dataset has multiple channels")
    if "var_name" not in options:
        options["var_name"] = "Sv"

    if method == "blackwell" and "depth" not in dataset:
        if "echo_range" not in dataset:
            raise ValueError("Blackwell bottom detection requires 'echo_range' or 'depth'")
        dataset = dataset.assign(depth=dataset["echo_range"])

    refinement_dataset = dataset
    dataset = preprocess_bottom_dataset(dataset, options=preprocessing)
    detected = ep.mask.detect_seafloor(dataset, method=method, params=options)
    if method == "blackwell":
        r0 = float(options.get("r0", 0.0))
        no_intercept = detected <= r0 + 1e-9
        detected = detected.where(~no_intercept)
    detected = postprocess_bottom_line(detected, options=postprocessing)
    detected = refine_bottom_line(refinement_dataset, detected, options=refinement)
    if offset:
        detected = detected + offset
    detected.name = "bottom_range"
    detected.attrs["vertical_coordinate"] = "echo_range"
    return detected
