"""Plot target-strength distributions and target locations."""



"""
RUN EXAMPLE



"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _first_column(frame: pd.DataFrame, names: list[str], label: str) -> str:
    for name in names:
        if name in frame:
            return name
    raise ValueError(f"Could not find {label} column. Tried: {', '.join(names)}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot single-target TS histograms and TS versus depth."
    )
    parser.add_argument("csv", type=Path, help="CSV written by scripts/test_fm2.py")
    parser.add_argument(
        "--raw",
        type=Path,
        default=None,
        help="Original EK80 FM RAW file; enables the echogram with target overlays.",
    )
    parser.add_argument(
        "--ts-min",
        type=float,
        default=None,
        help="Optional lower compensated-TS limit in dB re 1 m2.",
    )
    parser.add_argument(
        "--ts-max",
        type=float,
        default=None,
        help="Optional upper compensated-TS limit in dB re 1 m2.",
    )
    parser.add_argument("--bins", type=int, default=40, help="Number of histogram bins.")
    parser.add_argument(
        "--depth-column",
        default=None,
        help="Depth/range column; defaults to single_target_range or depth_m.",
    )
    parser.add_argument(
        "--time-column",
        default=None,
        help="Time column; defaults to ping_time, time, or timestamp.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for PNG files; defaults to the CSV directory.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "matplotlib is required for plotting; install the project dependencies "
            "with `python -m pip install -e .`."
        ) from exc

    frame = pd.read_csv(args.csv)
    compensated = _first_column(
        frame, ["compensated_TS", "TS_compensated_dB"], "compensated TS"
    )
    uncompensated = _first_column(
        frame, ["uncompensated_TS", "TS_uncompensated_dB"], "uncompensated TS"
    )
    depth = args.depth_column or _first_column(
        frame, ["single_target_range", "depth_m"], "depth/range"
    )
    time = args.time_column or _first_column(
        frame, ["ping_time", "time", "timestamp"], "time"
    )

    selected = frame.dropna(
        subset=[compensated, uncompensated, depth, time]
    ).copy()
    selected[compensated] = pd.to_numeric(selected[compensated], errors="coerce")
    selected[uncompensated] = pd.to_numeric(
        selected[uncompensated], errors="coerce"
    )
    selected[depth] = pd.to_numeric(selected[depth], errors="coerce")
    selected[time] = pd.to_datetime(selected[time], errors="coerce")
    selected = selected.dropna(subset=[compensated, uncompensated, depth, time])
    if args.ts_min is not None:
        selected = selected[selected[compensated] >= args.ts_min]
    if args.ts_max is not None:
        selected = selected[selected[compensated] <= args.ts_max]
    if selected.empty:
        raise ValueError("No targets remain after the selected TS filters.")

    output_dir = args.output_dir or args.csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.csv.stem

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].hist(selected[compensated], bins=args.bins, color="#176b87", alpha=0.85)
    axes[0].set(title="Compensated target strength", xlabel="TS (dB re 1 m2)", ylabel="Targets")
    axes[1].hist(selected[uncompensated], bins=args.bins, color="#d47b3f", alpha=0.85)
    axes[1].set(title="Uncompensated target strength", xlabel="TS (dB re 1 m2)", ylabel="Targets")
    histogram_path = output_dir / f"{stem}_ts_histograms.png"
    figure.savefig(histogram_path, dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    axis.scatter(selected[compensated], selected[depth], s=5, alpha=0.35, color="#176b87")
    axis.set(
        title="Compensated target strength versus depth/range",
        xlabel="Compensated TS (dB re 1 m2)",
        ylabel=f"{depth} (m)",
    )
    axis.invert_yaxis()
    scatter_path = output_dir / f"{stem}_ts_vs_depth.png"
    figure.savefig(scatter_path, dpi=180)
    plt.close(figure)

    ts_values = selected[compensated]
    ts_span = ts_values.max() - ts_values.min()
    if ts_span == 0:
        marker_sizes = pd.Series(60.0, index=selected.index)
    else:
        marker_sizes = 20 + 120 * (ts_values - ts_values.min()) / ts_span

    figure, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    points = axis.scatter(
        selected[time],
        selected[depth],
        s=marker_sizes,
        c=ts_values,
        cmap="viridis",
        alpha=0.65,
        edgecolors="none",
    )
    colorbar = figure.colorbar(points, ax=axis)
    colorbar.set_label("Compensated TS (dB re 1 m2)")
    axis.set(
        title="Single targets by time and depth/range",
        xlabel="Ping time",
        ylabel=f"{depth} (m)",
    )
    axis.invert_yaxis()
    time_depth_path = output_dir / f"{stem}_targets_time_depth.png"
    figure.savefig(time_depth_path, dpi=180)
    plt.close(figure)

    echogram_path = None
    if args.raw is not None:
        import echopype as ep

        if not args.raw.exists():
            raise FileNotFoundError(args.raw)

        ed = None
        try:
            ed = ep.open_raw(args.raw, sonar_model="EK80", use_swap=True)
            sp = ep.calibrate.compute_Sp(
                ed,
                waveform_mode="FM",
                encode_mode="complex",
            )
            echogram = sp["Sp"].isel(channel=0).transpose(
                "range_sample", "ping_time"
            ).load()
            echo_range = sp["echo_range"].isel(channel=0).transpose(
                "range_sample", "ping_time"
            ).load()

            figure, axis = plt.subplots(figsize=(14, 7), constrained_layout=True)
            image = axis.pcolormesh(
                sp["ping_time"].values,
                echo_range.values,
                echogram.values,
                shading="auto",
                cmap="viridis",
                vmin=-90,
                vmax=-30,
            )
            colorbar = figure.colorbar(image, ax=axis)
            colorbar.set_label("Sp (dB re 1 m-1)")
            axis.scatter(
                selected[time],
                selected[depth],
                marker="+",
                color="white",
                s=36,
                linewidths=0.8,
                alpha=0.9,
            )
            axis.set(
                title="Echogram with detected single targets",
                xlabel="Ping time",
                ylabel="Range/depth (m)",
            )
            axis.invert_yaxis()
            echogram_path = output_dir / f"{stem}_echogram_targets.png"
            figure.savefig(echogram_path, dpi=180)
            plt.close(figure)
        finally:
            if ed is not None:
                try:
                    if ed.converted_raw_path is None:
                        ed.cleanup_swap_files()
                finally:
                    ed.converted_raw_path = args.raw

    print(f"Plotted {len(selected):,} targets")
    print(f"Histograms: {histogram_path}")
    print(f"Scatterplot: {scatter_path}")
    print(f"Time-depth plot: {time_depth_path}")
    if echogram_path is not None:
        print(f"Echogram: {echogram_path}")


if __name__ == "__main__":
    main()