from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import xarray as xr

from echopype_fm_std.navigation import align_navigation, read_navigation


def _make_db(path):
    with sqlite3.connect(path) as con:
        con.execute(
            """
            CREATE TABLE track_points (
                id INTEGER PRIMARY KEY,
                platform TEXT,
                survey_id TEXT,
                timestamp_utc TEXT,
                latitude REAL,
                longitude REAL,
                distance_m REAL,
                speed_ms REAL,
                is_interpolated INTEGER,
                source_file TEXT,
                created_at TEXT,
                UNIQUE(platform, timestamp_utc)
            )
            """
        )
        rows = [
            (1, "VESSEL_A", "S1", "2026-01-01T00:00:00Z", 10.0, 20.0),
            (2, "VESSEL_A", "S1", "2026-01-01T00:10:00Z", 20.0, 40.0),
            (3, "VESSEL_B", "S1", "2026-01-01T00:00:00Z", 90.0, 100.0),
        ]
        con.executemany(
            "INSERT INTO track_points (id,platform,survey_id,timestamp_utc,latitude,longitude) VALUES (?,?,?,?,?,?)",
            rows,
        )


def test_read_navigation_filters_platform(tmp_path):
    db = tmp_path / "positions.sqlite"
    _make_db(db)

    ds = read_navigation(db, platform="VESSEL_A")

    assert ds.sizes["timestamp"] == 2
    np.testing.assert_allclose(ds.latitude.values, [10.0, 20.0])
    np.testing.assert_allclose(ds.longitude.values, [20.0, 40.0])
    assert ds.attrs["platform"] == "VESSEL_A"


def test_align_navigation_interpolates_and_does_not_extrapolate(tmp_path):
    db = tmp_path / "positions.sqlite"
    _make_db(db)
    nav = read_navigation(db, platform="VESSEL_A")

    ping_time = xr.DataArray(
        pd.to_datetime(
            [
                "2025-12-31T23:59:00Z",
                "2026-01-01T00:05:00Z",
                "2026-01-01T00:11:00Z",
            ],
            utc=True,
        ).tz_localize(None).to_numpy(dtype="datetime64[ns]"),
        dims="ping_time",
        name="ping_time",
    )

    aligned = align_navigation(nav, ping_time)

    assert np.isnan(aligned.latitude.values[0])
    np.testing.assert_allclose(aligned.latitude.values[1], 15.0)
    assert np.isnan(aligned.latitude.values[2])
    assert aligned.navigation_valid.values.tolist() == [False, True, False]
