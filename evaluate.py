"""Compare morphology and dynamic-programming surface detectors on one RAW file."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from echopype_fm_std.config import calibration_kwargs, load_config
from echopype_fm_std.io import open_raw
from echopype_fm_std.surface import detect_surface

ROOT = Path(__file__).resolve().parent
DEFAULT_RAW = (
    ROOT
    / "data/raw_surface_turb/SLUAquaSailor2020V1-Phase0-D20260425-T193551-0.raw"
)


def _roughness(boundary: np.ndarray) -> float:
    """Return mean absolute ping-to-ping boundary gradient."""
    return float(np.mean(np.abs(np.diff(boundary)))) if boundary.size > 1 else 0.0


def _plot(sv, boundaries: dict[str, object], output: Path) -> None:
    import matplotlib.pyplot as plt

    range_dim = next(dim for dim in sv.dims if dim != "ping_time")
    echogram = sv.transpose(range_dim, "ping_time")
    figure, axis = plt.subplots(figsize=(14, 7), constrained_layout=True)
    image = axis.pcolormesh(
        echogram["ping_time"].values,
        echogram[range_dim].values,
        echogram.values,
        shading="auto",
        cmap="viridis",
        vmin=-100,
        vmax=-30,
    )
    figure.colorbar(image, ax=axis, label="Sv (dB)")
    for name, boundary in boundaries.items():
        axis.plot(echogram["ping_time"].values, boundary, linewidth=1.0, label=name)
    axis.set(xlabel="Ping", ylabel="Depth sample", title="Surface detector comparison")
    axis.invert_yaxis()
    axis.legend()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--config", type=Path, default=ROOT / "config/default.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "surface_evaluation.png")
    args = parser.parse_args()

    import echopype as ep

    config = load_config(args.config)
    echo_data = open_raw(args.raw, config)
    try:
        input_config = config["input"]
        sv_dataset = ep.calibrate.compute_Sv(
            echo_data,
            waveform_mode=input_config["waveform_mode"],
            encode_mode=input_config["encoding_mode"],
            **calibration_kwargs(config),
        )
        sv = sv_dataset["Sv"]
        if "channel" in sv.dims:
            sv = sv.isel(channel=0)
        params = {
            "threshold_db": -50.0,
            "smooth_depth_size": 5,
            "smooth_ping_size": 3,
            "closing_height": 7,
            "median_window": 9,
            "top_exclude_rows": 10,
        }
        morphology = detect_surface(sv, params=params)
        boundaries = {"morphology": morphology.values}
        for name, boundary in boundaries.items():
            invalid = ~np.isfinite(boundary)
            print(
                f"{name}: invalid={100.0 * invalid.mean():.2f}% "
                f"mean_absolute_boundary_gradient={_roughness(boundary):.3f}"
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _plot(sv, boundaries, args.output)
        print(f"plot={args.output}")
    finally:
        echo_data.cleanup_swap_files()


if __name__ == "__main__":
    main()
