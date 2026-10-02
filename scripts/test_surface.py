"""Run surface/turbidity detection on EK80 RAW files and plot overlays."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from echopype_fm_std.config import load_config
from echopype_fm_std.io import get_beam_group, open_raw
from echopype_fm_std.surface import detect_surface

ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect and plot the surface/turbidity boundary for EK80 RAW files."
    )
    parser.add_argument(
        "--raw",
        type=Path,
        action="append",
        help="RAW file to process; repeat this option. Defaults to data/raw/*.raw.",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config/default.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/surface")
    parser.add_argument(
        "--channel", default=None, help="Channel label or zero-based channel index."
    )
    return parser


def _resolve_channel(value: str | None) -> int | str | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def _plot_surface(sv, surface, output_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    sv = sv.load()
    range_dim = next(dim for dim in sv.dims if dim not in {"channel", "ping_time"})
    echogram = sv.transpose(range_dim, "ping_time")
    range_coord = echogram["echo_range"] if "echo_range" in echogram.coords else echogram[range_dim]
    if range_coord.ndim == 2:
        range_coord = range_coord.transpose(range_dim, "ping_time")
    else:
        range_coord = range_coord.broadcast_like(echogram)

    figure, axis = plt.subplots(figsize=(14, 7), constrained_layout=True)
    image = axis.pcolormesh(
        echogram["ping_time"].values,
        range_coord.values,
        echogram.values,
        shading="auto",
        cmap="viridis",
        vmin=-90,
        vmax=-30,
    )
    figure.colorbar(image, ax=axis, label="Sv (dB re 1 m-1)")
    axis.plot(
        surface["ping_time"].values,
        surface.values,
        color="white",
        linewidth=1.2,
        label="Detected surface/turbidity boundary",
    )
    axis.set(title=title, xlabel="Ping time", ylabel="Range (m)")
    axis.invert_yaxis()
    axis.legend(loc="best")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def process_file(raw_path: Path, config: dict, output_dir: Path, channel: int | str | None) -> dict:
    import echopype as ep

    echo_data = open_raw(raw_path, config)
    try:
        beam_group = get_beam_group(echo_data, config["input"]["beam_group"])
        sv_dataset = ep.calibrate.compute_Sv(
            echo_data,
            waveform_mode=config["input"]["waveform_mode"],
            encode_mode=config["input"]["encoding_mode"],
        )
        sv = sv_dataset["Sv"].assign_coords(echo_range=sv_dataset["echo_range"])
        surface = detect_surface(
            sv,
            method="threshold",
            params=config["surface"],
            channel=channel,
        )
        plot_sv = sv
        if "channel" in plot_sv.dims:
            if channel is None:
                plot_sv = plot_sv.isel(channel=0)
            elif isinstance(channel, int):
                plot_sv = plot_sv.isel(channel=channel)
            else:
                plot_sv = plot_sv.sel(channel=channel)
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / f"{raw_path.stem}_surface.png"
        _plot_surface(
            plot_sv,
            surface,
            plot_path,
            f"Surface/turbidity detection: {raw_path.name}",
        )
        valid = np.isfinite(surface.values)
        message = f"{raw_path.name}: {int(valid.sum())}/{surface.size} detections"
        if valid.any():
            detected_range = surface.values[valid]
            message += f"; range={detected_range.min():.2f}-{detected_range.max():.2f} m"
        print(f"{message}; pings={beam_group.sizes['ping_time']}; plot={plot_path}")
        return {
            "raw": str(raw_path),
            "ping_count": int(surface.size),
            "detected_count": int(valid.sum()),
            "plot": str(plot_path),
        }
    finally:
        echo_data.cleanup_swap_files()


def main() -> None:
    args = _parser().parse_args()
    config = load_config(args.config)
    raw_paths = args.raw or sorted((ROOT / "data/raw").glob("*.raw"))
    if not raw_paths:
        raise SystemExit("No RAW files found. Pass one or more --raw paths.")
    channel = _resolve_channel(args.channel)
    for raw_path in raw_paths:
        raw_path = raw_path if raw_path.is_absolute() else ROOT / raw_path
        process_file(raw_path, config, args.output_dir, channel)


if __name__ == "__main__":
    main()
