from __future__ import annotations

import argparse
from pathlib import Path

from echopype_fm_std.config import load_config
from echopype_fm_std.io import get_beam_group, open_raw
from echopype_fm_std.navigation import (
    add_navigation_to_echodata,
    align_navigation,
    read_navigation,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument(
        "--platform",
        help="Navigation platform for this RAW file; defaults to config/default.yaml",
    )
    args = parser.parse_args()

    config_path = ROOT / "config/default.yaml"
    config = load_config(config_path)
    raw_path = args.raw if args.raw.is_absolute() else ROOT / args.raw
    database_path = ROOT / config["navigation"]["navigation_db"]
    selected_platform = args.platform or config["navigation"]["platform"]

    echo_data = open_raw(raw_path, config)
    try:
        beam_group = get_beam_group(echo_data, config["input"]["beam_group"])
        navigation = read_navigation(
            database_path,
            platform=selected_platform,
            start_time=beam_group["ping_time"].values[0],
            end_time=beam_group["ping_time"].values[-1],
        )
        aligned = align_navigation(
            navigation,
            beam_group["ping_time"],
            max_gap_seconds=config["navigation"].get("max_gap_seconds"),
        )
        add_navigation_to_echodata(echo_data, aligned)

        platform = echo_data["Platform"]

        print("\nFirst five ping coordinates:")
        for timestamp, latitude, longitude in zip(
            beam_group["ping_time"].values[:5],
            platform["latitude"].values[:5],
            platform["longitude"].values[:5],
        ):
            print(f"{timestamp}: {latitude}, {longitude}")

        assert platform["latitude"].size == beam_group.sizes["ping_time"]
        assert platform["longitude"].size == beam_group.sizes["ping_time"]
        assert int(aligned["navigation_valid"].sum()) > 0
        print(f"Raw pings: {beam_group.sizes['ping_time']}")
        print(f"Navigation platform: {selected_platform}")
        print(f"Valid navigation pings: {int(aligned['navigation_valid'].sum())}")
        print(f"Stored Platform latitude dimension: {platform['latitude'].dims[0]}")
    finally:
        echo_data.cleanup_swap_files()


if __name__ == "__main__":
    main()
