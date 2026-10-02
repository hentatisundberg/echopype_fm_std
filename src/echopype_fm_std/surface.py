from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import xarray as xr


def _find_first_run(
    values: np.ndarray, threshold: float, consecutive_samples: int, start: int
) -> int:
    """Find the first run of values below threshold after ``start``."""
    count = 0
    for i in range(start, values.size):
        value = values[i]
        if np.isfinite(value) and value < threshold:
            count += 1
            if count >= consecutive_samples:
                return i - consecutive_samples + 1
        else:
            count = 0
    return -1


def _find_layer(
    values: np.ndarray,
    *,
    beam_dead_zone: int,
    consecutive_pings: int,
    quantile: float,
    strength_threshold: float,
    minimum_size: int,
) -> bool:
    """Reproduce the original persistent low-strength-layer check."""
    in_a_row = 0
    for row in values[beam_dead_zone:]:
        finite = row[np.isfinite(row)]
        if finite.size == 0:
            in_a_row = 0
            continue
        if float(np.quantile(finite, quantile)) < strength_threshold:
            in_a_row += 1
        else:
            in_a_row = 0
        if in_a_row >= consecutive_pings:
            return True
    return values.shape[0] > beam_dead_zone + minimum_size


def detect_surface(
    sv: xr.DataArray | xr.Dataset,
    *,
    method: str = "threshold",
    params: Mapping[str, object] | None = None,
    channel: int | str | None = None,
    threshold_db: float = -50.0,
    consecutive_samples: int = 3,
    dead_zone_samples: int = 0,
    range_bin_fraction: float = 1.0,
    rolling_ping_window: int = 1,
) -> xr.DataArray:
    """Detect a surface/turbidity boundary from an xarray Sv echogram.

    This is an xarray-compatible rewrite of the threshold/run-length idea used
    by ``echoedge/lib/find_waves.py``. It does not mutate the input array.

    ``sv`` may be a calibrated ``Sv`` DataArray or Dataset. A Dataset must
    contain an ``Sv`` variable. If it contains multiple channels, ``channel``
    selects one before detection. The calibrated Dataset and original
    EchoData object therefore remain available to downstream Echopype stages.

    Expected dimensions after channel selection are ``ping_time`` and one
    range/depth-like dimension.
    The return value is the sample/range coordinate at the first qualifying run
    in each ping. Missing detections are represented by NaN. With the original
    wave logic enabled, ``wave_threshold_db`` and
    ``wave_threshold_layer_db`` control the two threshold paths. The final
    boundary is expanded with a centered rolling maximum when
    ``rolling_ping_window`` is greater than one.
    """
    if method != "threshold":
        raise ValueError(f"Unsupported surface detection method: {method!r}")
    if params is not None and not isinstance(params, Mapping):
        raise TypeError("params must be a mapping")
    options = dict(params or {})
    if isinstance(sv, xr.Dataset):
        if "Sv" not in sv:
            raise ValueError("sv Dataset must contain an 'Sv' variable")
        sv_data = sv["Sv"]
        if "echo_range" in sv:
            sv_data = sv_data.assign_coords(echo_range=sv["echo_range"])
        sv = sv_data
    if not isinstance(sv, xr.DataArray):
        raise TypeError("sv must be an xarray.DataArray or Dataset")
    if "channel" in sv.dims:
        if channel is None:
            if sv.sizes["channel"] != 1:
                raise ValueError("channel is required when Sv contains multiple channels")
            sv = sv.isel(channel=0)
        elif isinstance(channel, int):
            sv = sv.isel(channel=channel)
        else:
            sv = sv.sel(channel=channel)
    elif channel is not None:
        raise ValueError("channel was provided but Sv has no channel dimension")
    threshold_db = float(options.get("threshold_db", threshold_db))
    consecutive_samples = int(options.get("consecutive_samples", consecutive_samples))
    dead_zone_samples = int(options.get("dead_zone_samples", dead_zone_samples))
    range_bin_fraction = float(options.get("range_bin_fraction", range_bin_fraction))
    rolling_ping_window = int(options.get("rolling_ping_window", rolling_ping_window))
    wave_threshold_db = float(options.get("wave_threshold_db", -77.0))
    wave_threshold_layer_db = float(options.get("wave_threshold_layer_db", -68.0))
    wave_consecutive_samples = int(
        options.get("wave_consecutive_samples", consecutive_samples)
    )
    beam_dead_zone_samples = int(
        options.get("beam_dead_zone_samples", dead_zone_samples)
    )
    layer_quantile = float(options.get("layer_quantile", 0.7))
    layer_strength_threshold_db = float(
        options.get("layer_strength_threshold_db", -80.0)
    )
    layer_consecutive_pings = int(options.get("layer_consecutive_pings", 3))
    layer_minimum_samples = int(options.get("layer_minimum_samples", 10))
    extreme_wave_size_m = float(options.get("extreme_wave_size_m", 70.0))
    use_original_wave_logic = bool(options.get("use_original_wave_logic", False))
    if use_original_wave_logic:
        threshold_db = wave_threshold_db
        consecutive_samples = wave_consecutive_samples
        dead_zone_samples = beam_dead_zone_samples
    if consecutive_samples < 1:
        raise ValueError("consecutive_samples must be >= 1")
    if dead_zone_samples < 0:
        raise ValueError("dead_zone_samples must be >= 0")
    if not 0 < range_bin_fraction <= 1:
        raise ValueError("range_bin_fraction must be > 0 and <= 1")
    if rolling_ping_window < 1:
        raise ValueError("rolling_ping_window must be >= 1")
    if not 0 <= layer_quantile <= 1:
        raise ValueError("layer_quantile must be between 0 and 1")
    if layer_consecutive_pings < 1 or layer_minimum_samples < 0:
        raise ValueError("layer detection counts must be non-negative and valid")
    if "ping_time" not in sv.dims:
        raise ValueError("sv must have a 'ping_time' dimension")

    range_dims = [d for d in sv.dims if d != "ping_time"]
    if len(range_dims) != 1:
        raise ValueError("sv must have exactly one non-ping dimension")
    range_dim = range_dims[0]

    bin_size = max(1, round(1 / range_bin_fraction))
    if bin_size > 1:
        sv = sv.coarsen({range_dim: bin_size}, boundary="trim").mean()

    result = np.full(sv.sizes["ping_time"], np.nan, dtype=float)
    values = sv.transpose("ping_time", range_dim).to_numpy()
    if "echo_range" in sv.coords:
        echo_range = sv["echo_range"]
        if echo_range.ndim == 2:
            coord = echo_range.transpose("ping_time", range_dim).to_numpy()
        else:
            coord = np.broadcast_to(echo_range.to_numpy(), values.shape)
    else:
        coord = np.broadcast_to(sv[range_dim].to_numpy(), values.shape)

    layer_found = use_original_wave_logic and _find_layer(
        values,
        beam_dead_zone=beam_dead_zone_samples,
        consecutive_pings=layer_consecutive_pings,
        quantile=layer_quantile,
        strength_threshold=layer_strength_threshold_db,
        minimum_size=layer_minimum_samples,
    )
    active_threshold = (
        wave_threshold_layer_db if layer_found and use_original_wave_logic else threshold_db
    )

    for ping_idx, row in enumerate(values):
        sample_idx = _find_first_run(
            row,
            threshold=float(active_threshold),
            consecutive_samples=int(consecutive_samples),
            start=int(dead_zone_samples),
        )
        if sample_idx >= 0:
            result[ping_idx] = float(coord[ping_idx, sample_idx])

    if use_original_wave_logic:
        valid_ranges = result[np.isfinite(result)]
        if valid_ranges.size and float(valid_ranges.mean()) > extreme_wave_size_m:
            active_threshold = wave_threshold_layer_db
            for ping_idx, row in enumerate(values):
                sample_idx = _find_first_run(
                    row,
                    threshold=active_threshold,
                    consecutive_samples=wave_consecutive_samples,
                    start=beam_dead_zone_samples,
                )
                result[ping_idx] = (
                    float(coord[ping_idx, sample_idx]) if sample_idx >= 0 else np.nan
                )

    raw_result = result.copy()
    if rolling_ping_window > 1:
        result = (
            xr.DataArray(result, dims="ping_time", coords={"ping_time": sv["ping_time"]})
            .rolling(ping_time=rolling_ping_window, center=True, min_periods=1)
            .max()
            .to_numpy()
        )

    return xr.DataArray(
        result,
        dims="ping_time",
        coords={"ping_time": sv["ping_time"]},
        name="surface_range",
        attrs={
            "long_name": "Detected surface/turbidity boundary",
            "threshold_db": float(active_threshold),
            "wave_threshold_db": float(wave_threshold_db),
            "wave_threshold_layer_db": float(wave_threshold_layer_db),
            "consecutive_samples": int(consecutive_samples),
            "dead_zone_samples": int(dead_zone_samples),
            "range_bin_fraction": float(range_bin_fraction),
            "range_bin_size": int(bin_size),
            "layer_found": bool(layer_found),
            "rolling_ping_window": int(rolling_ping_window),
            "rolling_boundary": "max" if rolling_ping_window > 1 else "none",
            "raw_detection_count": int(np.isfinite(raw_result).sum()),
        },
    )
