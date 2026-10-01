from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


REQUIRED_COLUMNS = {
    "platform",
    "timestamp_utc",
    "latitude",
    "longitude",
}


def _ensure_datetime64_ns_utc(values: Any) -> pd.DatetimeIndex:
    """Return timezone-naive UTC timestamps suitable for xarray datetime coords."""
    parsed = pd.to_datetime(values, utc=True, errors="raise")
    return pd.DatetimeIndex(parsed).tz_convert("UTC").tz_localize(None).astype("datetime64[ns]")


def read_navigation(
    db_path: str | Path,
    platform: str,
    start_time: Any | None = None,
    end_time: Any | None = None,
) -> xr.Dataset:
    """Read navigation for one platform from the track_points SQLite table.

    The returned dataset is indexed by UTC ``timestamp`` until it is aligned
    to acoustic pings. Filtering by start/end time is performed *after*
    timestamp normalization so database timestamp string formatting cannot
    affect the result.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(db_path)
    if not platform or not platform.strip():
        raise ValueError("platform must be a non-empty string")

    query = """
        SELECT platform, survey_id, timestamp_utc, latitude, longitude,
               distance_m, speed_ms, is_interpolated, source_file, created_at
        FROM track_points
        WHERE platform = ?
        ORDER BY timestamp_utc
    """

    with sqlite3.connect(db_path) as con:
        df = pd.read_sql_query(query, con, params=[platform])

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"track_points is missing required columns: {sorted(missing)}")

    if df.empty:
        raise ValueError(f"No navigation records found for platform={platform!r}.")

    df["timestamp"] = _ensure_datetime64_ns_utc(df["timestamp_utc"])
    df = df.sort_values("timestamp")

    if start_time is not None:
        start = _ensure_datetime64_ns_utc([start_time])[0]
        df = df.loc[df["timestamp"] >= start]
    if end_time is not None:
        end = _ensure_datetime64_ns_utc([end_time])[0]
        df = df.loc[df["timestamp"] <= end]

    if df.empty:
        raise ValueError(
            f"No navigation records for platform={platform!r} in the requested time range."
        )

    duplicate_count = int(df["timestamp"].duplicated().sum())
    if duplicate_count:
        # The schema has UNIQUE(platform, timestamp_utc), but defensive handling
        # protects against equivalent timestamps expressed in different text forms.
        df = df.drop_duplicates("timestamp", keep="last")

    if df["timestamp"].duplicated().any():
        raise ValueError("Navigation timestamps are still duplicated after normalization.")

    ds = xr.Dataset(
        data_vars={
            "latitude": ("timestamp", df["latitude"].to_numpy(dtype=float)),
            "longitude": ("timestamp", df["longitude"].to_numpy(dtype=float)),
        },
        coords={"timestamp": df["timestamp"].to_numpy(dtype="datetime64[ns]")},
        attrs={
            "platform": platform,
            "source": str(db_path),
            "duplicate_timestamps_removed": duplicate_count,
        },
    )

    return ds


def list_platforms(db_path: str | Path) -> list[str]:
    """Return distinct platform names from the tracking database."""
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(db_path)

    query = "SELECT DISTINCT platform FROM track_points WHERE platform IS NOT NULL ORDER BY platform"
    with sqlite3.connect(db_path) as con:
        rows = con.execute(query).fetchall()
    return [str(row[0]) for row in rows]


def align_navigation(
    nav: xr.Dataset,
    ping_time: xr.DataArray,
    max_gap_seconds: float | None = None,
) -> xr.Dataset:
    """Interpolate navigation onto acoustic ping times without extrapolation.

    Parameters
    ----------
    nav:
        Navigation dataset returned by :func:`read_navigation`, indexed by
        ``timestamp``.
    ping_time:
        Acoustic ping timestamps. They are interpreted as UTC if timezone-naive.
    max_gap_seconds:
        Optional maximum gap between navigation observations. Interpolated
        positions inside larger gaps are masked to NaN.
    """
    if "timestamp" not in nav.dims:
        raise ValueError("Navigation dataset must have a 'timestamp' dimension.")
    for name in ("latitude", "longitude"):
        if name not in nav:
            raise ValueError(f"Navigation dataset is missing {name!r}.")

    ping_index = _ensure_datetime64_ns_utc(ping_time.values)
    ping_values = ping_index.to_numpy(dtype="datetime64[ns]")

    nav_sorted = nav.sortby("timestamp")
    aligned = nav_sorted.interp(timestamp=ping_values).rename({"timestamp": "ping_time"})
    aligned = aligned.assign_coords(ping_time=ping_values)

    valid = np.isfinite(aligned["latitude"]) & np.isfinite(aligned["longitude"])

    if max_gap_seconds is not None:
        if max_gap_seconds <= 0:
            raise ValueError("max_gap_seconds must be > 0 or None.")
        source_t = nav_sorted["timestamp"].values.astype("datetime64[ns]").astype("int64")
        target_t = ping_index.to_numpy(dtype="datetime64[ns]").astype("int64")
        idx = np.searchsorted(source_t, target_t, side="left")
        left = np.clip(idx - 1, 0, len(source_t) - 1)
        right = np.clip(idx, 0, len(source_t) - 1)
        gap_ns = source_t[right] - source_t[left]
        # At the edges, interpolation is invalid outside source coverage anyway.
        gap_valid = gap_ns <= int(max_gap_seconds * 1e9)
        aligned["latitude"] = aligned["latitude"].where(gap_valid)
        aligned["longitude"] = aligned["longitude"].where(gap_valid)
        valid &= gap_valid

    aligned["navigation_valid"] = valid
    aligned.attrs.update(nav.attrs)
    aligned.attrs["interpolation"] = "linear"
    aligned.attrs["extrapolate"] = False

    return aligned
