"""Interactively inspect surface/turbidity detection parameters for RAW files."""

from __future__ import annotations

import argparse
import tkinter as tk
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from tkinter import ttk
from typing import Any

import echopype as ep
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from echopype_fm_std.config import calibration_kwargs, load_config
from echopype_fm_std.io import open_raw
from echopype_fm_std.surface import detect_surface

ROOT = Path(__file__).resolve().parents[1]
MAX_PLOT_RANGES = 800
MAX_PLOT_PINGS = 1600


class SurfaceInspector:
    def __init__(
        self,
        root: tk.Tk,
        raw_paths: list[Path],
        config: dict[str, Any],
        output_dir: Path,
        channel: int | str | None,
    ):
        self.root = root
        self.raw_paths = raw_paths
        self.config = config
        self.output_dir = output_dir
        self.channel = channel
        self.file_index = 0
        self.dataset: Any = None
        self._load_executor = ThreadPoolExecutor(max_workers=1)
        self._detect_executor = ThreadPoolExecutor(max_workers=1)
        self._load_future: Future[Any] | None = None
        self._detect_future: Future[Any] | None = None
        self._file_generation = 0
        self._pending_file_index: int | None = None
        self._pending_detection: tuple[Any, dict[str, Any], int] | None = None
        self._closed = False

        root.title("Surface/turbidity inspector")
        root.geometry("1400x900")
        root.protocol("WM_DELETE_WINDOW", self.close)
        controls = ttk.Frame(root, padding=8)
        controls.pack(side=tk.LEFT, fill=tk.Y)
        plot_frame = ttk.Frame(root, padding=(0, 8, 8, 8))
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fields: dict[str, tk.Entry | ttk.Combobox] = {}
        params = config.get("surface", {})
        ttk.Label(controls, text="Surface parameters", font=("", 12, "bold")).pack(anchor="w")
        self._add_entry(
            controls, "threshold_db", "Threshold (dB)", params.get("threshold_db", -77.0)
        )
        self._add_entry(
            controls,
            "consecutive_samples",
            "Consecutive samples",
            params.get("consecutive_samples", 3),
        )
        self._add_entry(
            controls,
            "dead_zone_samples",
            "Dead zone samples",
            params.get("dead_zone_samples", 10),
        )
        self._add_entry(
            controls,
            "range_bin_fraction",
            "Range bin fraction",
            params.get("range_bin_fraction", 0.1),
        )
        self._add_entry(
            controls,
            "rolling_ping_window",
            "Rolling ping window",
            params.get("rolling_ping_window", 3),
        )
        self._add_choice(
            controls,
            "use_original_wave_logic",
            "Original wave logic",
            "enabled" if params.get("use_original_wave_logic", False) else "disabled",
            ("disabled", "enabled"),
        )
        self._add_entry(
            controls,
            "wave_threshold_db",
            "Wave threshold (dB)",
            params.get("wave_threshold_db", -77.0),
        )
        self._add_entry(
            controls,
            "wave_threshold_layer_db",
            "Layer threshold (dB)",
            params.get("wave_threshold_layer_db", -68.0),
        )
        self._add_entry(
            controls,
            "wave_consecutive_samples",
            "Wave consecutive samples",
            params.get("wave_consecutive_samples", 3),
        )
        self._add_entry(
            controls,
            "beam_dead_zone_samples",
            "Beam dead zone samples",
            params.get("beam_dead_zone_samples", 10),
        )
        self._add_entry(
            controls,
            "layer_quantile",
            "Layer quantile",
            params.get("layer_quantile", 0.7),
        )
        self._add_entry(
            controls,
            "layer_strength_threshold_db",
            "Layer threshold (dB)",
            params.get("layer_strength_threshold_db", -80.0),
        )
        self._add_entry(
            controls,
            "layer_consecutive_pings",
            "Layer consecutive pings",
            params.get("layer_consecutive_pings", 3),
        )
        self._add_entry(
            controls,
            "layer_minimum_samples",
            "Layer minimum samples",
            params.get("layer_minimum_samples", 10),
        )
        self._add_entry(
            controls,
            "extreme_wave_size_m",
            "Extreme wave size (m)",
            params.get("extreme_wave_size_m", 70.0),
        )

        ttk.Button(controls, text="Apply", command=self._apply).pack(fill=tk.X, pady=(12, 3))
        navigation = ttk.Frame(controls)
        navigation.pack(fill=tk.X, pady=3)
        ttk.Button(navigation, text="Previous", command=lambda: self._change_file(-1)).pack(
            side=tk.LEFT, expand=True, fill=tk.X
        )
        ttk.Button(navigation, text="Next", command=lambda: self._change_file(1)).pack(
            side=tk.LEFT, expand=True, fill=tk.X
        )
        ttk.Button(controls, text="Save YAML", command=self._save_yaml).pack(fill=tk.X, pady=3)
        self.status = ttk.Label(controls, text="Starting...", wraplength=250)
        self.status.pack(anchor="w", pady=(16, 0))

        self.figure, self.axis = plt.subplots(figsize=(10, 7))
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.image = None
        self._surface_line, = self.axis.plot([], [], color="white", linewidth=1.5)
        self._surface_line.set_label("Detected surface/turbidity boundary")
        self.root.after(50, self._load_file)

    def _add_entry(self, parent: ttk.Frame, key: str, label: str, value: object) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text=label, width=24).pack(side=tk.LEFT)
        entry = ttk.Entry(row, width=12)
        entry.insert(0, str(value))
        entry.pack(side=tk.RIGHT)
        self.fields[key] = entry

    def _add_choice(
        self,
        parent: ttk.Frame,
        key: str,
        label: str,
        value: str,
        choices: tuple[str, ...],
    ) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text=label, width=24).pack(side=tk.LEFT)
        choice = ttk.Combobox(row, values=choices, state="readonly", width=10)
        choice.set(value)
        choice.pack(side=tk.RIGHT)
        self.fields[key] = choice

    def _params(self) -> dict[str, Any]:
        def number(key: str) -> float:
            try:
                return float(self.fields[key].get())
            except ValueError as exc:
                raise ValueError(f"{key} must be numeric") from exc

        integer_keys = {
            "consecutive_samples",
            "dead_zone_samples",
            "rolling_ping_window",
            "wave_consecutive_samples",
            "beam_dead_zone_samples",
            "layer_consecutive_pings",
            "layer_minimum_samples",
        }
        output: dict[str, Any] = {}
        for key in (
            "threshold_db",
            "range_bin_fraction",
            "wave_threshold_db",
            "wave_threshold_layer_db",
            "layer_quantile",
            "layer_strength_threshold_db",
            "extreme_wave_size_m",
        ):
            output[key] = number(key)
        for key in integer_keys:
            output[key] = int(number(key))
        output["use_original_wave_logic"] = (
            self.fields["use_original_wave_logic"].get() == "enabled"
        )
        return output

    def _load_file(self) -> None:
        if self._load_future is not None and not self._load_future.done():
            return
        raw_path = self.raw_paths[self.file_index]
        self.status.configure(text=f"Loading {raw_path.name}...")
        generation = self._file_generation
        self._load_future = self._load_executor.submit(self._prepare_file, raw_path)
        self.root.after(100, lambda: self._poll_load(generation))

    def _prepare_file(self, raw_path: Path) -> Any:
        echo_data = open_raw(raw_path, self.config)
        try:
            input_cfg = self.config["input"]
            return ep.calibrate.compute_Sv(
                echo_data,
                waveform_mode=input_cfg["waveform_mode"],
                encode_mode=input_cfg["encoding_mode"],
                **calibration_kwargs(self.config),
            ).load()
        finally:
            echo_data.cleanup_swap_files()

    def _poll_load(self, generation: int) -> None:
        if self._closed:
            return
        if self._load_future is None or not self._load_future.done():
            self.root.after(100, lambda: self._poll_load(generation))
            return
        if generation != self._file_generation:
            return
        if self._pending_file_index is not None:
            self._start_file(self._pending_file_index)
            return
        try:
            self.dataset = self._load_future.result()
            self._render_echogram()
            self.status.configure(text="Loaded. Calculating surface...")
            self._apply()
        except Exception as exc:
            self.status.configure(text=f"Could not load file: {exc}")

    def _apply(self) -> None:
        if self.dataset is None:
            self.status.configure(text="Still loading the RAW file.")
            return
        try:
            params = self._params()
        except ValueError as exc:
            self.status.configure(text=str(exc))
            return
        generation = self._file_generation
        if self._detect_future is not None and not self._detect_future.done():
            self._pending_detection = (self.dataset, params, generation)
            self.status.configure(text="Queued the latest surface calculation.")
            return
        self.status.configure(text="Calculating surface...")
        self._detect_future = self._detect_executor.submit(
            detect_surface, self.dataset, params=params, channel=self.channel
        )
        self.root.after(100, lambda: self._poll_detection(generation))

    def _poll_detection(self, generation: int) -> None:
        if self._closed:
            return
        if self._detect_future is None or not self._detect_future.done():
            self.root.after(100, lambda: self._poll_detection(generation))
            return
        try:
            result = self._detect_future.result()
        except Exception as exc:
            result = None
            if generation == self._file_generation:
                self.status.configure(text=f"Could not calculate surface: {exc}")
        pending = self._pending_detection
        self._pending_detection = None
        if pending is not None:
            self._detect_future = self._detect_executor.submit(
                detect_surface, pending[0], params=pending[1], channel=self.channel
            )
            self.root.after(100, lambda: self._poll_detection(pending[2]))
            return
        if generation != self._file_generation or result is None:
            return
        self._render(result)

    def _selected_data(self, name: str) -> Any:
        data = self.dataset[name]
        if "channel" not in data.dims:
            return data
        if self.channel is None:
            return data.isel(channel=0)
        if isinstance(self.channel, int):
            return data.isel(channel=self.channel)
        return data.sel(channel=self.channel)

    def _render(self, surface: Any) -> None:
        self._render_echogram()
        self._surface_line.set_data(surface["ping_time"].values, surface.values)
        valid = np.isfinite(surface.values)
        coverage = float(valid.mean() * 100.0) if valid.size else 0.0
        self.status.configure(
            text=f"Ready. Detections: {valid.sum()}/{valid.size} ({coverage:.1f}%)."
        )
        self.canvas.draw_idle()

    def _render_echogram(self) -> None:
        if self.dataset is None:
            return
        view_limits = None
        if self.image is not None:
            view_limits = (self.axis.get_xlim(), self.axis.get_ylim())
        sv = self._selected_data("Sv")
        range_dim = next(dim for dim in sv.dims if dim != "ping_time")
        ping_stride = max(1, int(np.ceil(sv.sizes["ping_time"] / MAX_PLOT_PINGS)))
        range_stride = max(1, int(np.ceil(sv.sizes[range_dim] / MAX_PLOT_RANGES)))
        selection = {
            range_dim: slice(None, None, range_stride),
            "ping_time": slice(None, None, ping_stride),
        }
        echogram = sv.transpose(range_dim, "ping_time").isel(selection)
        range_coord = self._selected_data("echo_range").transpose(range_dim, "ping_time").isel(
            selection
        )
        if self.image is not None:
            self.image.remove()
        self._surface_line.set_data([], [])
        self.image = self.axis.pcolormesh(
            echogram["ping_time"].values,
            range_coord.values,
            echogram.values,
            shading="auto",
            cmap="viridis",
            vmin=-100,
            vmax=-30,
        )
        self.axis.set(
            title=f"Surface/turbidity detection: {self.raw_paths[self.file_index].name}",
            xlabel="Ping time",
            ylabel="Echo range (m)",
        )
        finite_ping_times = np.asarray(echogram["ping_time"].values)
        finite_ping_times = finite_ping_times[np.isfinite(finite_ping_times)]
        if finite_ping_times.size:
            self.axis.set_xlim(finite_ping_times.min(), finite_ping_times.max())
        finite_ranges = np.asarray(range_coord.values)
        finite_ranges = finite_ranges[np.isfinite(finite_ranges)]
        if finite_ranges.size:
            self.axis.set_ylim(float(finite_ranges.max()), float(finite_ranges.min()))
        if view_limits is not None:
            self.axis.set_xlim(*view_limits[0])
            self.axis.set_ylim(*view_limits[1])
        self.axis.legend(loc="best")
        self.canvas.draw_idle()

    def _change_file(self, step: int) -> None:
        if self._load_future is not None and not self._load_future.done():
            current_index = (
                self._pending_file_index
                if self._pending_file_index is not None
                else self.file_index
            )
            self._pending_file_index = (current_index + step) % len(self.raw_paths)
            self.status.configure(text=f"Queued {self.raw_paths[self._pending_file_index].name}...")
            return
        self._start_file((self.file_index + step) % len(self.raw_paths))

    def _start_file(self, file_index: int) -> None:
        self._file_generation += 1
        self.file_index = file_index
        self._pending_file_index = None
        self.dataset = None
        if self.image is not None:
            self.image.remove()
            self.image = None
        self.status.configure(text=f"Loading {self.raw_paths[self.file_index].name}...")
        self._load_file()

    def _save_yaml(self) -> None:
        try:
            params = self._params()
        except ValueError as exc:
            self.status.configure(text=str(exc))
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = (
            self.output_dir / f"{self.raw_paths[self.file_index].stem}_surface_params.yaml"
        )
        output_path.write_text(
            yaml.safe_dump({"surface": params}, sort_keys=False),
            encoding="utf-8",
        )
        self.status.configure(text=f"Saved {output_path}")

    def close(self) -> None:
        self._closed = True
        self._load_executor.shutdown(wait=False, cancel_futures=True)
        self._detect_executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def _resolve_channel(value: str | None) -> int | str | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def _raw_paths(
    raw_values: list[Path] | None,
    folder_values: list[Path] | None,
) -> list[Path]:
    candidates = raw_values or []
    paths: list[Path] = []
    for value in [*candidates, *(folder_values or [])]:
        path = value if value.is_absolute() else ROOT / value
        if path.is_dir():
            paths.extend(sorted(path.glob("*.raw")))
        elif path.is_file():
            paths.append(path)
    if not raw_values and not folder_values:
        paths = sorted((ROOT / "data/raw").glob("*.raw"))
    return list(dict.fromkeys(paths))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw",
        type=Path,
        action="append",
        help="RAW file or folder to process; repeat this option.",
    )
    parser.add_argument(
        "--folder",
        type=Path,
        action="append",
        help="Folder containing RAW files; repeat this option.",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config/default.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/surface-interactive")
    parser.add_argument("--channel", help="Channel label or zero-based channel index.")
    args = parser.parse_args()
    config = load_config(args.config)
    raw_paths = _raw_paths(args.raw, args.folder)
    if not raw_paths:
        parser.error("No RAW files found. Pass --raw FILE or --folder DIRECTORY.")
    root = tk.Tk()
    SurfaceInspector(root, raw_paths, config, args.output_dir, _resolve_channel(args.channel))
    root.mainloop()


if __name__ == "__main__":
    main()
