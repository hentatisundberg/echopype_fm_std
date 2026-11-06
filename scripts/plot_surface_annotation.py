"""Plot saved surface annotations over calibrated echograms.

Single-file example:

    python scripts/plot_surface_annotation.py \
            --raw data/raw_surface_turb/SLUAquaSailor2020V1-Phase0-D20260419-T082052-0.raw \
            --annotation \
            output/surface_annotations/SLUAquaSailor2020V1-Phase0-D20260419-T082052-0_surface.csv \
            --config config/default.yaml \
            --show

For a folder, pass the annotation directory and RAW folder:

    python scripts/plot_surface_annotation.py \
            --folder data/raw_surface_turb \
            --annotation output/surface_annotations \
            --show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from echopype_fm_std.annotation import load_surface_annotation  # noqa: E402
from echopype_fm_std.config import calibration_kwargs, load_config  # noqa: E402

MAX_PLOT_RANGES = 800
MAX_PLOT_PINGS = 1600


def _selected_data(dataset: Any, name: str, channel: int | str | None) -> Any:
    data = dataset[name]
    if "channel" not in data.dims:
        return data
    if channel is None:
        return data.isel(channel=0)
    if isinstance(channel, int):
        return data.isel(channel=channel)
    return data.sel(channel=channel)


def plot_annotation(
    raw_path: Path,
    annotation_path: Path,
    config: dict[str, Any],
    *,
    channel: int | str | None = None,
    output_path: Path | None = None,
    show: bool = False,
) -> Path | None:
    """Create an echogram plot with a saved annotation overlaid."""
    import echopype as ep
    import matplotlib.pyplot as plt

    from echopype_fm_std.io import open_raw

    labels = load_surface_annotation(annotation_path)
    metadata_raw = labels["metadata"].get("raw_file")
    if metadata_raw and Path(metadata_raw).resolve() != raw_path.resolve():
        raise ValueError(
            "The annotation belongs to a different RAW file: "
            f"{metadata_raw}"
        )
    echo_data = open_raw(raw_path, config)
    try:
        input_cfg = config["input"]
        dataset = ep.calibrate.compute_Sv(
            echo_data,
            waveform_mode=input_cfg["waveform_mode"],
            encode_mode=input_cfg["encoding_mode"],
            **calibration_kwargs(config),
        ).load()
    finally:
        echo_data.cleanup_swap_files()

    sv = _selected_data(dataset, "Sv", channel)
    if labels["surface_range_m"].size != sv.sizes["ping_time"]:
        raise ValueError(
            "Annotation ping count does not match the RAW file: "
            f"{labels['surface_range_m'].size} labels for {sv.sizes['ping_time']} pings."
        )
    range_dim = next(dim for dim in sv.dims if dim != "ping_time")
    ping_stride = max(1, int(np.ceil(sv.sizes["ping_time"] / MAX_PLOT_PINGS)))
    range_stride = max(1, int(np.ceil(sv.sizes[range_dim] / MAX_PLOT_RANGES)))
    selection = {
        range_dim: slice(None, None, range_stride),
        "ping_time": slice(None, None, ping_stride),
    }
    echogram = sv.transpose(range_dim, "ping_time").isel(selection)
    ranges = _selected_data(dataset, "echo_range", channel).transpose(
        range_dim, "ping_time"
    ).isel(selection)

    fig, axis = plt.subplots(figsize=(14, 8))
    image = axis.pcolormesh(
        np.arange(sv.sizes["ping_time"])[::ping_stride],
        ranges.values,
        echogram.values,
        shading="auto",
        cmap="viridis",
        vmin=-100,
        vmax=-30,
    )
    line = labels["surface_range_m"]
    ping_indices = labels["ping_index"]
    valid = np.isfinite(line)
    axis.plot(
        ping_indices[valid],
        line[valid],
        color="white",
        linewidth=1.2,
        marker=".",
        markersize=3,
        label="Per-ping annotation",
    )
    vertices = labels["metadata"].get("vertices", [])
    if vertices:
        vertex_ping = np.asarray([item["ping_index"] for item in vertices], dtype=float)
        vertex_range = np.asarray(
            [item["surface_range_m"] for item in vertices], dtype=float
        )
        axis.scatter(
            vertex_ping,
            vertex_range,
            color="red",
            edgecolors="white",
            linewidths=0.5,
            s=35,
            zorder=3,
            label="Clicked vertices",
        )
    axis.set(
        title=f"Surface/turbidity annotation: {raw_path.name}",
        xlabel="Ping index",
        ylabel="Echo range (m)",
    )
    finite_ranges = ranges.values[np.isfinite(ranges.values)]
    if finite_ranges.size:
        axis.set_ylim(float(finite_ranges.max()), float(finite_ranges.min()))
    axis.legend(loc="best")
    fig.colorbar(image, ax=axis, label="Sv (dB)")
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    if show or output_path is None:
        plt.show()
    plt.close(fig)
    return output_path


def _resolve_raw_paths(
    values: list[Path] | None,
    folders: list[Path] | None,
) -> list[Path]:
    paths: list[Path] = []
    for value in [*(values or []), *(folders or [])]:
        path = value if value.is_absolute() else ROOT / value
        if path.is_dir():
            paths.extend(sorted(path.glob("*.raw")))
        elif path.is_file():
            paths.append(path)
    return list(dict.fromkeys(paths))


def _resolve_annotation_path(raw_path: Path, annotation: Path) -> Path:
    if annotation.is_dir() or not annotation.suffix:
        return annotation / f"{raw_path.stem}_surface.csv"
    return annotation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, action="append", help="RAW file; repeat this option.")
    parser.add_argument("--folder", type=Path, action="append", help="Folder containing RAW files.")
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config/default.yaml")
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--output", type=Path, help="Output image path for one RAW file.")
    output_group.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for one <raw-stem>.png image per RAW file.",
    )
    parser.add_argument("--show", action="store_true", help="Display the plot interactively.")
    parser.add_argument("--channel", help="Channel label or zero-based channel index.")
    args = parser.parse_args()

    raw_paths = _resolve_raw_paths(args.raw, args.folder)
    if not raw_paths:
        parser.error("No RAW files found. Pass --raw FILE or --folder DIRECTORY.")
    annotation = args.annotation if args.annotation.is_absolute() else ROOT / args.annotation
    annotation_paths = [
        _resolve_annotation_path(raw_path, annotation) for raw_path in raw_paths
    ]
    annotated_pairs = [
        (raw_path, annotation_path)
        for raw_path, annotation_path in zip(raw_paths, annotation_paths)
        if annotation_path.is_file()
    ]
    for raw_path, annotation_path in zip(raw_paths, annotation_paths):
        if not annotation_path.is_file():
            print(f"Skipping {raw_path.name}: annotation not found at {annotation_path}")
    if not annotated_pairs:
        parser.error("No matching annotation files found for the RAW files.")
    if args.output is not None and len(annotated_pairs) != 1:
        parser.error("--output can only be used with exactly one RAW file.")

    channel: int | str | None = args.channel
    if channel is not None:
        try:
            channel = int(channel)
        except ValueError:
            pass
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_config(config_path)
    output_dir = args.output_dir
    if output_dir is not None and not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    for raw_path, annotation_path in annotated_pairs:
        output_path = args.output
        if output_path is None and output_dir is not None:
            output_path = output_dir / f"{raw_path.stem}.png"
        elif output_path is not None and not output_path.is_absolute():
            output_path = ROOT / output_path
        plot_annotation(
            raw_path,
            annotation_path,
            config,
            channel=channel,
            output_path=output_path,
            show=args.show,
        )


if __name__ == "__main__":
    main()
