from __future__ import annotations

from pathlib import Path
from typing import Any

from .bottom import detect_bottom
from .config import load_config
from .fm import pulse_compress_fm
from .io import get_beam_group, open_raw
from .navigation import align_navigation, read_navigation
from .surface import detect_surface


def run_pipeline(
    raw_path: str | Path,
    navigation_db: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    """Run the currently implemented starter stages and report the state.

    The function intentionally stops before experimental FM stages unless they
    have been wired and validated. This makes partial progress explicit rather
    than silently producing scientifically incomplete CSV output.
    """
    cfg = load_config(config_path)
    raw_path = Path(raw_path)

    ed = open_raw(raw_path, cfg)
    beam_group_name = cfg.get("input", {}).get("beam_group", "Beam_group1")
    bg = get_beam_group(ed, beam_group_name)
    ping_time = bg["ping_time"]

    nav_cfg = cfg["navigation"]
    nav = read_navigation(
        navigation_db,
        platform=nav_cfg["platform"],
        start_time=ping_time.values[0],
        end_time=ping_time.values[-1],
    )
    nav_aligned = align_navigation(
        nav,
        ping_time,
        max_gap_seconds=nav_cfg.get("max_gap_seconds"),
    )

    result: dict[str, Any] = {
        "echo_data": ed,
        "beam_group": bg,
        "navigation": nav_aligned,
        "stages_completed": ["read_raw", "align_navigation"],
    }

    # The following stages become active as their adapters are validated.
    try:
        pulse_compressed = pulse_compress_fm(ed, config=cfg)
    except NotImplementedError:
        result["stages_pending"] = [
            "pulse_compression",
            "seafloor_detection",
            "surface_detection",
            "single_target_detection",
            "target_ts",
            "csv_export",
        ]
        return result

    result["pulse_compressed"] = pulse_compressed
    result["stages_completed"].append("pulse_compression")

    # Placeholder: actual Sv/Sp dataset selection needs to be connected here.
    # Once the calibrated Sv DataArray is available, the following pattern is used:
    # result["bottom"] = detect_bottom(ed, method=cfg["bottom"]["method"], **cfg["bottom"]["params"])
    # result["surface"] = detect_surface(result["sv"], **cfg["surface"])
    return result
