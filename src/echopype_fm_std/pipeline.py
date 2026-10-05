from __future__ import annotations

from pathlib import Path
from typing import Any

import echopype as ep

from .bottom import detect_bottom
from .config import calibration_kwargs, load_config
from .io import get_beam_group, open_raw
from .navigation import add_navigation_to_echodata, align_navigation, read_navigation
from .surface import detect_surface


def run_pipeline(
    raw_path: str | Path,
    navigation_db: str | Path,
    config_path: str | Path,
    platform: str | None = None,
) -> dict[str, Any]:
    """Run input, navigation, calibrated Sv, bottom, and surface stages."""
    cfg = load_config(config_path)
    raw_path = Path(raw_path)

    ed = open_raw(raw_path, cfg)
    try:
        beam_group_name = cfg.get("input", {}).get("beam_group", "Beam_group1")
        bg = get_beam_group(ed, beam_group_name)
        ping_time = bg["ping_time"]

        nav_cfg = cfg["navigation"]
        selected_platform = platform or nav_cfg["platform"]
        nav = read_navigation(
            navigation_db,
            platform=selected_platform,
            start_time=ping_time.values[0],
            end_time=ping_time.values[-1],
        )
        nav_aligned = align_navigation(
            nav,
            ping_time,
            max_gap_seconds=nav_cfg.get("max_gap_seconds"),
        )
        add_navigation_to_echodata(ed, nav_aligned)

        result: dict[str, Any] = {
            "echo_data": ed,
            "beam_group": bg,
            "navigation": nav_aligned,
            "platform": selected_platform,
            "stages_completed": ["read_raw", "align_navigation"],
        }

        input_cfg = cfg["input"]
        sv = ep.calibrate.compute_Sv(
            ed,
            waveform_mode=input_cfg["waveform_mode"],
            encode_mode=input_cfg["encoding_mode"],
            **calibration_kwargs(cfg),
        )
        bottom_cfg = cfg["bottom"]
        bottom_method = bottom_cfg.get("method", "blackwell")
        bottom_dataset = sv
        if bottom_method == "blackwell":
            bottom_dataset = ep.consolidate.add_splitbeam_angle(
                sv,
                ed,
                waveform_mode=input_cfg["waveform_mode"],
                encode_mode=input_cfg["encoding_mode"],
                pulse_compression=True,
                to_disk=False,
            )
        result["sv"] = sv
        result["bottom"] = detect_bottom(
            bottom_dataset,
            method=bottom_method,
            params=bottom_cfg.get("params", {}),
        )
        result["stages_completed"].append("bottom_detection")
        result["surface"] = detect_surface(
            sv,
            method=cfg["surface"].get("method", "threshold"),
            params=cfg["surface"],
        )
        result["stages_completed"].append("surface_detection")
        return result
    finally:
        ed.cleanup_swap_files()
