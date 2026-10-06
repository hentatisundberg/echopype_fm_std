from __future__ import annotations

from pathlib import Path

import xarray as xr

CSV_COLUMNS = [
    "depth_m",
    "angle_alongship_deg",
    "angle_athwartship_deg",
    "TS_compensated_dB",
    "TS_uncompensated_dB",
]


def filter_targets_by_ts(
    ds_targets: xr.Dataset,
    *,
    ts_comp_min: float | None = None,
    ts_comp_max: float | None = None,
) -> xr.Dataset:
    """Return targets within the compensated-TS limits."""
    name = "compensated_TS" if "compensated_TS" in ds_targets else "TS_compensated_dB"
    if name not in ds_targets and (ts_comp_min is not None or ts_comp_max is not None):
        raise ValueError("Cannot filter targets: dataset has no compensated TS field.")
    keep = xr.ones_like(ds_targets[name], dtype=bool)
    if ts_comp_min is not None:
        keep &= ds_targets[name] >= ts_comp_min
    if ts_comp_max is not None:
        keep &= ds_targets[name] <= ts_comp_max
    return ds_targets.where(keep, drop=True)


def build_target_properties(
    ds_targets: xr.Dataset,
    *,
    depth_name: str = "single_target_range",
) -> xr.Dataset:
    """Normalize target variables into the project schema.

    This stage currently standardizes names only; the scientifically sensitive
    FM TS calculation should be implemented and validated separately.
    """
    required = [
        depth_name,
        "single_target_alongship_angle",
        "single_target_athwartship_angle",
        "beam_comp_db",
    ]
    missing = [v for v in required if v not in ds_targets]
    if missing:
        raise ValueError(f"Target dataset is missing required variables: {missing}")

    out = xr.Dataset(coords=ds_targets.coords)
    out["depth_m"] = ds_targets[depth_name]
    out["angle_alongship_deg"] = ds_targets["single_target_alongship_angle"]
    out["angle_athwartship_deg"] = ds_targets["single_target_athwartship_angle"]
    out["beam_comp_db"] = ds_targets["beam_comp_db"]

    return out


def add_target_ts(
    ds_targets: xr.Dataset,
    *,
    ts_uncompensated_db: xr.DataArray,
) -> xr.Dataset:
    """Add uncorrected and beam-compensated target strength.

    The precise FM TS extraction/calibration equation must be validated against
    the chosen Echopype FM calibration implementation before this function is
    used for scientific production.
    """
    if "beam_comp_db" not in ds_targets:
        raise ValueError("Target dataset must contain beam_comp_db first.")
    ts_uncomp = ts_uncompensated_db
    ds_targets = ds_targets.copy()
    ds_targets["TS_uncompensated_dB"] = ts_uncomp
    ds_targets["TS_compensated_dB"] = ts_uncomp + ds_targets["beam_comp_db"]
    return ds_targets


def export_targets_csv(
    ds_targets: xr.Dataset,
    output_path: str | Path,
    *,
    include_metadata_columns: bool = False,
    ts_comp_min: float | None = None,
    ts_comp_max: float | None = None,
    navigation: xr.Dataset | None = None,
    survey_metadata: dict[str, object] | None = None,
) -> None:
    """Export one row per target, optionally filtered by compensated TS.

    All variables and coordinates that vary along ``single_target`` are
    written so detector diagnostics, positions, and TS-related fields remain
    available for later analysis.
    """
    frame = ds_targets.to_dataframe().reset_index(drop=True)
    if navigation is not None and "ping_time" in frame:
        nav_frame = navigation[["latitude", "longitude"]].to_dataframe().reset_index()
        frame = frame.merge(nav_frame, on="ping_time", how="left")
        frame = frame.rename(
            columns={"latitude": "target_latitude", "longitude": "target_longitude"}
        )
    for name, value in (survey_metadata or {}).items():
        frame[name] = value

    compensated_name = (
        "compensated_TS"
        if "compensated_TS" in frame
        else "TS_compensated_dB"
        if "TS_compensated_dB" in frame
        else None
    )
    if (ts_comp_min is not None or ts_comp_max is not None) and compensated_name is None:
        raise ValueError(
            "Cannot filter target CSV: dataset has no compensated_TS field."
        )

    if ts_comp_min is not None:
        frame = frame[frame[compensated_name] >= ts_comp_min]
    if ts_comp_max is not None:
        frame = frame[frame[compensated_name] <= ts_comp_max]

    ordered_columns = [
        column
        for column in [
            "ping_time",
            "target_latitude",
            "target_longitude",
            "survey_start_time",
            "survey_end_time",
            "survey_start_latitude",
            "survey_start_longitude",
            "survey_end_latitude",
            "survey_end_longitude",
            "single_target_range",
            "single_target_alongship_angle",
            "single_target_athwartship_angle",
            "uncompensated_TS",
            "compensated_TS",
            "single_target_TS_comp_min_dB",
            *CSV_COLUMNS,
        ]
        if column in frame
    ]
    ordered_columns.extend(column for column in frame if column not in ordered_columns)
    frame = frame[ordered_columns]

    frame.to_csv(Path(output_path), index=False)
