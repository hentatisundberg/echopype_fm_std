from __future__ import annotations

from pathlib import Path
from typing import Any

import echopype as ep

from .bottom import detect_bottom
from .config import calibration_kwargs, load_config
from .integration import export_integration_csv, integrate_echoes
from .io import get_beam_group, open_raw
from .masks import export_boundary_summary, export_mask_echogram, summarize_boundaries
from .navigation import add_navigation_to_echodata, align_navigation, read_navigation
from .single_target import detect_fm_targets, mask_target_dataset, prepare_fm_sp
from .surface import detect_surface
from .targets import export_targets_csv, filter_targets_by_ts


def run_pipeline(
    raw_path: str | Path,
    navigation_db: str | Path,
    config_path: str | Path,
    platform: str | None = None,
) -> dict[str, Any]:
    """Run input, navigation, calibration, boundaries, and echo integration."""
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
        integration_cfg = cfg.get("echo_integration", {})
        output_cfg = cfg.get("output", {})
        output_root = Path(output_cfg.get("directory", "output"))
        images_dir = output_root / output_cfg.get("images_directory", "images")
        masks_dir = output_root / output_cfg.get("masks_directory", "masks")
        targets_dir = output_root / output_cfg.get("targets_directory", "single_targets")
        integration_dir = output_root / output_cfg.get(
            "integration_directory", "integration"
        )
        for directory in (images_dir, masks_dir, targets_dir, integration_dir):
            directory.mkdir(parents=True, exist_ok=True)
        if integration_cfg.get("enabled", True):
            integration = integrate_echoes(
                sv,
                result["surface"],
                result["bottom"],
                layer_size_m=float(integration_cfg.get("layer_size_m", 10.0)),
                distance_bin_nmi=float(
                    integration_cfg.get("distance_bin_nmi", 0.1)
                ),
                transducer_depth_m=(
                    input_cfg.get("transducer_depth_m")
                    if integration_cfg.get("relative_to_surface", True)
                    else None
                ),
                navigation=nav_aligned,
            )
            result["echo_integration"] = integration
            integration_path = integration_dir / f"{raw_path.stem}_echo_integration.csv"
            export_integration_csv(integration, integration_path)
            result["echo_integration_csv"] = integration_path
            result["stages_completed"].append("echo_integration")
        mask_summary = summarize_boundaries(
            result["surface"],
            result["bottom"],
            nav_aligned,
            transducer_depth_m=input_cfg.get("transducer_depth_m"),
        )
        mask_path = masks_dir / f"{raw_path.stem}_mask_summary.csv"
        export_boundary_summary(mask_summary, mask_path)
        result["mask_summary"] = mask_summary
        result["mask_summary_csv"] = mask_path
        result["stages_completed"].append("mask_summary")
        target_cfg = cfg.get("single_target", {})
        if target_cfg.get("enabled", False):
            sp_angle = prepare_fm_sp(ed, bg, input_cfg)
            sp_for_detection = mask_target_dataset(
                sp_angle.isel(channel=0).load(),
                result["surface"],
                result["bottom"],
            )
            target_params = dict(target_cfg.get("params", {}))
            ts_min = float(target_params.pop("TS_comp_min", -60.0))
            ts_max = target_params.pop("TS_comp_max", None)
            targets = detect_fm_targets(sp_for_detection, target_params)
            targets_ts = ep.calibrate.compute_TS(
                sp_angle.load(),
                point_locations=targets,
            )
            filtered_targets = filter_targets_by_ts(
                targets_ts,
                ts_comp_min=ts_min,
                ts_comp_max=ts_max,
            )
            target_path = targets_dir / f"{raw_path.stem}_single_targets.csv"
            nav_values = nav_aligned
            survey_metadata = {
                "survey_start_time": nav_values["ping_time"].values[0],
                "survey_end_time": nav_values["ping_time"].values[-1],
                "survey_start_latitude": float(nav_values["latitude"].isel(ping_time=0)),
                "survey_start_longitude": float(nav_values["longitude"].isel(ping_time=0)),
                "survey_end_latitude": float(nav_values["latitude"].isel(ping_time=-1)),
                "survey_end_longitude": float(nav_values["longitude"].isel(ping_time=-1)),
                "single_target_TS_comp_min_dB": ts_min,
            }
            export_targets_csv(
                filtered_targets,
                target_path,
                include_metadata_columns=cfg.get("output", {}).get(
                    "include_metadata_columns", False
                ),
                ts_comp_min=ts_min,
                ts_comp_max=ts_max,
                navigation=nav_values,
                survey_metadata=survey_metadata,
            )
            result["single_target_dataset"] = targets_ts
            result["single_target_filtered_dataset"] = filtered_targets
            result["single_target_csv"] = target_path
            result["single_target_count"] = filtered_targets.sizes.get("single_target", 0)
            result["stages_completed"].append("single_target_detection")
        mask_image_path = images_dir / f"{raw_path.stem}_mask_echogram.png"
        clean_image_path = images_dir / f"{raw_path.stem}_mask_clean.png"
        export_mask_echogram(
            sv,
            result["surface"],
            result["bottom"],
            mask_image_path,
            targets=(
                filtered_targets
                if target_cfg.get("enabled", False)
                else None
            ),
            clean_output_path=clean_image_path,
        )
        result["mask_echogram_png"] = mask_image_path
        result["mask_clean_png"] = clean_image_path
        result["stages_completed"].append("mask_images")
        return result
    finally:
        ed.cleanup_swap_files()
