from __future__ import annotations

from typing import Any

import echopype as ep
import xarray as xr


def detect_bottom(
    ed,
    *,
    channel: str | None = None,
    method: str = "basic",
    **kwargs: Any,
) -> xr.DataArray:
    """Thin project wrapper around Echopype seafloor detection."""
    params = dict(kwargs)
    if channel is not None:
        params["channel"] = channel
    return ep.mask.detect_seafloor(ed, method=method, **params)
