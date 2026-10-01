from __future__ import annotations

import numpy as np
import xarray as xr


def _find_first_run(values: np.ndarray, threshold: float, consecutive_samples: int, start: int) -> int:
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


def detect_surface(
    sv: xr.DataArray,
    threshold_db: float = -50.0,
    consecutive_samples: int = 3,
    dead_zone_samples: int = 0,
) -> xr.DataArray:
    """Detect a surface/turbidity boundary from an xarray Sv echogram.

    This is an xarray-compatible rewrite of the threshold/run-length idea used
    by ``echoedge/lib/find_waves.py``. It does not mutate the input array.

    Expected dimensions are ``ping_time`` and one range/depth-like dimension.
    The return value is the sample/range coordinate at the first qualifying run
    in each ping. Missing detections are represented by NaN.
    """
    if not isinstance(sv, xr.DataArray):
        raise TypeError("sv must be an xarray.DataArray")
    if consecutive_samples < 1:
        raise ValueError("consecutive_samples must be >= 1")
    if dead_zone_samples < 0:
        raise ValueError("dead_zone_samples must be >= 0")
    if "ping_time" not in sv.dims:
        raise ValueError("sv must have a 'ping_time' dimension")

    range_dims = [d for d in sv.dims if d != "ping_time"]
    if len(range_dims) != 1:
        raise ValueError("sv must have exactly one non-ping dimension")
    range_dim = range_dims[0]

    result = np.full(sv.sizes["ping_time"], np.nan, dtype=float)
    values = sv.transpose("ping_time", range_dim).to_numpy()
    coord = sv[range_dim].to_numpy()

    for ping_idx, row in enumerate(values):
        sample_idx = _find_first_run(
            row,
            threshold=float(threshold_db),
            consecutive_samples=int(consecutive_samples),
            start=int(dead_zone_samples),
        )
        if sample_idx >= 0:
            result[ping_idx] = float(coord[sample_idx])

    return xr.DataArray(
        result,
        dims="ping_time",
        coords={"ping_time": sv["ping_time"]},
        name="surface_range",
        attrs={
            "long_name": "Detected surface/turbidity boundary",
            "threshold_db": float(threshold_db),
            "consecutive_samples": int(consecutive_samples),
            "dead_zone_samples": int(dead_zone_samples),
        },
    )
