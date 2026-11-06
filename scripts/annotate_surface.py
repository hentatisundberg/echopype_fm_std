"""Manually annotate a surface/turbidity line on calibrated EK80 echograms.

Examples:

    python scripts/annotate_surface.py \
        --folder data/raw_surface_turb/ \
        --output-dir output/surface_annotations

Left-click to add vertices, right-click to remove the nearest vertex,
and use the toolbar to zoom/pan while annotating. Click the toolbar's Home
button to restore the full view. Save annotation before moving to another file.
"""

from __future__ import annotations

import argparse
import tkinter as tk
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from tkinter import ttk
from typing import Any

import numpy as np

from echopype_fm_std.annotation import (
    interpolate_surface_vertices,
    save_surface_annotation,
)
from echopype_fm_std.config import calibration_kwargs, load_config

ROOT = Path(__file__).resolve().parents[1]
MAX_PLOT_RANGES = 800
MAX_PLOT_PINGS = 1600


class SurfaceAnnotator:
    def __init__(
        self,
        root: tk.Tk,
        raw_paths: list[Path],
        config: dict[str, Any],
        output_dir: Path,
        channel: int | str | None,
    ) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

        self.root = root
        self.raw_paths = raw_paths
        self.config = config
        self.output_dir = output_dir
        self.channel = channel
        self.file_index = 0
        self.dataset: Any = None
        self.vertices: list[tuple[float, float]] = []
        self._load_executor = ThreadPoolExecutor(max_workers=1)
        self._load_future: Future[Any] | None = None
        self._closed = False

        root.title("Surface/turbidity annotation")
        root.geometry("1400x900")
        root.protocol("WM_DELETE_WINDOW", self.close)
        controls = ttk.Frame(root, padding=8)
        controls.pack(side=tk.LEFT, fill=tk.Y)
        plot_frame = ttk.Frame(root, padding=(0, 8, 8, 8))
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(
            controls, text="Manual surface annotation", font=("", 12, "bold")
        ).pack(anchor="w")
        ttk.Label(
            controls,
            text=(
                "Left click: add point\n"
                "Right click: remove nearest\n"
                "Use toolbar Zoom/Pan\n"
                "Home restores full view\n"
                "Save before changing file"
            ),
            justify=tk.LEFT,
        ).pack(anchor="w", pady=(8, 12))
        ttk.Button(controls, text="Clear vertices", command=self._clear).pack(fill=tk.X)
        ttk.Button(
            controls, text="Save annotation", command=self._save
        ).pack(fill=tk.X, pady=3)
        ttk.Button(
            controls, text="Annotate points", command=self._enable_annotation_mode
        ).pack(fill=tk.X, pady=3)
        navigation = ttk.Frame(controls)
        navigation.pack(fill=tk.X, pady=3)
        ttk.Button(
            navigation, text="Previous", command=lambda: self._change_file(-1)
        ).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(
            navigation, text="Next", command=lambda: self._change_file(1)
        ).pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.status = ttk.Label(controls, text="Starting...", wraplength=250)
        self.status.pack(anchor="w", pady=(16, 0))

        self.figure, self.axis = plt.subplots(figsize=(10, 7))
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.image = None
        self._plot_ping_indices = np.array([], dtype=int)
        self._line, = self.axis.plot([], [], color="white", linewidth=1.5)
        self._points, = self.axis.plot([], [], "wo", markersize=5)
        self.canvas.mpl_connect("button_press_event", self._mouse_click)
        self.root.after(50, self._load_file)

    def _enable_annotation_mode(self) -> None:
        """Exit toolbar navigation modes so mouse clicks edit vertices."""
        if self.toolbar.mode:
            if self.toolbar.mode.startswith("zoom"):
                self.toolbar.zoom()
            elif self.toolbar.mode.startswith("pan"):
                self.toolbar.pan()
        self.status.configure(text="Annotation mode: left-click to add points.")

    def _load_file(self) -> None:
        raw_path = self.raw_paths[self.file_index]
        self.status.configure(text=f"Loading {raw_path.name}...")
        self._load_future = self._load_executor.submit(self._prepare_file, raw_path)
        self.root.after(100, self._poll_load)

    def _prepare_file(self, raw_path: Path) -> Any:
        import echopype as ep

        from echopype_fm_std.io import open_raw

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

    def _poll_load(self) -> None:
        if self._closed or self._load_future is None:
            return
        if not self._load_future.done():
            self.root.after(100, self._poll_load)
            return
        try:
            self.dataset = self._load_future.result()
            self.vertices = []
            self._render_echogram()
            self.status.configure(
                text=f"Loaded {self.raw_paths[self.file_index].name}. Add at least two vertices."
            )
        except Exception as exc:
            self.status.configure(text=f"Could not load file: {exc}")

    def _selected_data(self, name: str) -> Any:
        data = self.dataset[name]
        if "channel" not in data.dims:
            return data
        if self.channel is None:
            return data.isel(channel=0)
        return data.isel(channel=self.channel) if isinstance(self.channel, int) else data.sel(
            channel=self.channel
        )

    def _render_echogram(self) -> None:
        sv = self._selected_data("Sv")
        range_dim = next(dim for dim in sv.dims if dim != "ping_time")
        ping_stride = max(1, int(np.ceil(sv.sizes["ping_time"] / MAX_PLOT_PINGS)))
        range_stride = max(1, int(np.ceil(sv.sizes[range_dim] / MAX_PLOT_RANGES)))
        selection = {
            range_dim: slice(None, None, range_stride),
            "ping_time": slice(None, None, ping_stride),
        }
        echogram = sv.transpose(range_dim, "ping_time").isel(selection)
        ranges = self._selected_data("echo_range").transpose(range_dim, "ping_time").isel(
            selection
        )
        self._plot_ping_indices = np.arange(sv.sizes["ping_time"])[::ping_stride]
        if self.image is not None:
            self.image.remove()
        self.image = self.axis.pcolormesh(
            self._plot_ping_indices,
            ranges.values,
            echogram.values,
            shading="auto",
            cmap="viridis",
            vmin=-100,
            vmax=-30,
        )
        self.axis.set(
            title=f"Annotate surface/turbidity: {self.raw_paths[self.file_index].name}",
            xlabel="Ping index",
            ylabel="Echo range (m)",
        )
        finite_ranges = ranges.values[np.isfinite(ranges.values)]
        if finite_ranges.size:
            self.axis.set_ylim(float(finite_ranges.max()), float(finite_ranges.min()))
        self._line.set_data([], [])
        self._points.set_data([], [])
        self.canvas.draw_idle()

    def _mouse_click(self, event: Any) -> None:
        if self.dataset is None or event.inaxes is not self.axis or event.xdata is None:
            return
        if self.toolbar.mode:
            self.status.configure(
                text="Zoom/Pan is active. Click 'Annotate points' to add vertices."
            )
            return
        point = (float(event.xdata), float(event.ydata))
        if event.button == 3:
            if self.vertices:
                nearest = min(
                    range(len(self.vertices)),
                    key=lambda index: (self.vertices[index][0] - point[0]) ** 2
                    + (self.vertices[index][1] - point[1]) ** 2,
                )
                self.vertices.pop(nearest)
        elif event.button == 1:
            self.vertices.append(point)
        else:
            return
        self.vertices.sort()
        self._draw_annotation()

    def _draw_annotation(self) -> None:
        if self.dataset is None or not self.vertices:
            self._line.set_data([], [])
            self._points.set_data([], [])
        else:
            x, y = zip(*self.vertices)
            self._points.set_data(x, y)
            if len(self.vertices) >= 2:
                line = interpolate_surface_vertices(
                    x, y, self.dataset.sizes["ping_time"]
                )
                self._line.set_data(np.arange(line.size), line)
            else:
                self._line.set_data([], [])
        self.status.configure(text=f"{len(self.vertices)} vertices")
        self.canvas.draw_idle()

    def _clear(self) -> None:
        self.vertices = []
        self._draw_annotation()

    def _save(self) -> None:
        if self.dataset is None:
            self.status.configure(text="No RAW file is loaded.")
            return
        if len(self.vertices) < 2:
            self.status.configure(text="Add at least two vertices before saving.")
            return
        try:
            x, y = zip(*self.vertices)
            line = interpolate_surface_vertices(x, y, self.dataset.sizes["ping_time"])
            output_csv = self.output_dir / f"{self.raw_paths[self.file_index].stem}_surface.csv"
            save_surface_annotation(
                output_csv,
                raw_path=self.raw_paths[self.file_index],
                ping_times=self.dataset["ping_time"].values,
                surface_ranges_m=line,
                vertices=self.vertices,
                channel=self.channel,
            )
            self.status.configure(text=f"Saved {output_csv}")
        except Exception as exc:
            self.status.configure(text=f"Could not save annotation: {exc}")

    def _change_file(self, step: int) -> None:
        self.file_index = (self.file_index + step) % len(self.raw_paths)
        self.dataset = None
        self.vertices = []
        self._load_file()

    def close(self) -> None:
        self._closed = True
        self._load_executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def _annotation_exists(raw_path: Path, output_dir: Path) -> bool:
    annotation_stem = f"{raw_path.stem}_surface"
    return any(
        (output_dir / f"{annotation_stem}{suffix}").exists()
        for suffix in (".csv", ".json")
    )


def _resolve_paths(
    values: list[Path] | None,
    folders: list[Path] | None,
    output_dir: Path,
) -> list[Path]:
    paths: list[Path] = []
    for value in [*(values or []), *(folders or [])]:
        path = value if value.is_absolute() else ROOT / value
        if path.is_dir():
            paths.extend(sorted(path.glob("*.raw")))
        elif path.is_file():
            paths.append(path)
    return [
        path
        for path in dict.fromkeys(paths)
        if not _annotation_exists(path, output_dir)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, action="append", help="RAW file; repeat this option.")
    parser.add_argument("--folder", type=Path, action="append", help="Folder containing RAW files.")
    parser.add_argument("--config", type=Path, default=ROOT / "config/default.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/surface-annotations")
    parser.add_argument("--channel", help="Channel label or zero-based channel index.")
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    paths = _resolve_paths(args.raw, args.folder, output_dir)
    if not paths:
        parser.error(
            "No unannotated RAW files found. Pass --raw FILE or --folder DIRECTORY."
        )
    channel: int | str | None = args.channel
    if channel is not None:
        try:
            channel = int(channel)
        except ValueError:
            pass
    root = tk.Tk()
    SurfaceAnnotator(root, paths, load_config(args.config), output_dir, channel)
    root.mainloop()


if __name__ == "__main__":
    main()
