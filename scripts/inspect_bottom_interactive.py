"""Interactively inspect Blackwell bottom parameters for one RAW file at a time."""

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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from echopype_fm_std.bottom import (
    detect_bottom,
    postprocess_bottom_line,
    refine_bottom_line,
)
from echopype_fm_std.config import calibration_kwargs, load_config
from echopype_fm_std.io import open_raw

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/default.yaml"


class BottomInspector:
    def __init__(
        self,
        root: tk.Tk,
        raw_paths: list[Path],
        config: dict[str, Any],
        output_dir: Path,
    ):
        self.root = root
        self.raw_paths = raw_paths
        self.config = config
        self.output_dir = output_dir
        self.file_index = 0
        self.dataset = None
        self.sv = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._load_future: Future[tuple[Any, Any]] | None = None
        self._detect_future: Future[Any] | None = None
        self._current_params: dict[str, Any] = {}
        self._closed = False

        root.title("Blackwell bottom inspector")
        root.geometry("1400x900")
        root.protocol("WM_DELETE_WINDOW", self.close)
        controls = ttk.Frame(root, padding=8)
        controls.pack(side=tk.LEFT, fill=tk.Y)
        plot_frame = ttk.Frame(root, padding=(0, 8, 8, 8))
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fields: dict[str, tk.Entry] = {}
        params = config["bottom"]["params"]
        threshold = params.get("threshold", [-75.0, 0.03, 0.03])
        definitions = [
            ("threshold_sv", "Sv threshold", threshold[0]),
            ("threshold_theta", "Angle major", threshold[1]),
            ("threshold_phi", "Angle minor", threshold[2]),
            ("offset", "Offset (m)", params.get("offset", 0.3)),
            ("r0", "Start range (m)", params.get("r0", 0.0)),
            ("r1", "End range (m)", params.get("r1", 500.0)),
            ("wtheta", "Alongship window", params.get("wtheta", 28)),
            ("wphi", "Athwartship window", params.get("wphi", 52)),
        ]
        ttk.Label(controls, text="Blackwell parameters", font=("", 12, "bold")).pack(anchor="w")
        for key, label, value in definitions:
            row = ttk.Frame(controls)
            row.pack(fill=tk.X, pady=3)
            ttk.Label(row, text=label, width=20).pack(side=tk.LEFT)
            entry = ttk.Entry(row, width=12)
            entry.insert(0, str(value))
            entry.pack(side=tk.RIGHT)
            self.fields[key] = entry

        preprocessing = params.get("preprocessing", {})
        postprocessing = params.get("postprocessing", {})
        self._add_choice(
            controls,
            "pre_method",
            "Pre method",
            preprocessing.get("method", "none"),
            ("none", "max", "median", "median_max", "fill_dropouts"),
        )
        self._add_entry(controls, "pre_window", "Pre window", preprocessing.get("window", 1))
        self._add_entry(
            controls,
            "pre_median_window",
            "Pre median window",
            preprocessing.get("median_window", 3),
        )
        self._add_entry(
            controls, "pre_max_window", "Pre max window", preprocessing.get("max_window", 3)
        )
        self._add_entry(controls, "pre_gap", "Pre max gap", preprocessing.get("max_gap", 0))
        self._add_choice(
            controls,
            "post_method",
            "Post method",
            postprocessing.get("method", "none"),
            ("none", "max", "median", "median_interpolate", "max_interpolate"),
        )
        self._add_entry(controls, "post_window", "Post window", postprocessing.get("window", 1))
        self._add_entry(
            controls,
            "post_deviation",
            "Post deviation (m)",
            postprocessing.get("max_deviation_m", 10.0),
        )
        self._add_entry(controls, "post_gap", "Post max gap", postprocessing.get("max_gap", 0))
        self._add_choice(
            controls,
            "post_edge_fill",
            "Post edge fill",
            postprocessing.get("edge_fill", "none"),
            ("none", "nearest", "linear"),
        )
        self._add_entry(
            controls,
            "post_edge_gap",
            "Post max edge gap",
            postprocessing.get("max_edge_gap", 0),
        )
        refinement = params.get("refinement", {})
        self._add_choice(
            controls,
            "refinement_enabled",
            "Refinement",
            "enabled" if refinement.get("enabled", False) else "disabled",
            ("disabled", "enabled"),
        )
        self._add_entry(
            controls,
            "refinement_window",
            "Refine window (m)",
            refinement.get("window_m", 2.0),
        )
        self._add_entry(
            controls,
            "refinement_threshold",
            "Refine threshold",
            refinement.get("threshold_db", -35.0),
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
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.image = None
        self._raw_line, = self.axis.plot([], [], color="white", linewidth=1.0, alpha=0.8)
        self._preprocessed_line, = self.axis.plot(
            [], [], color="orange", linewidth=1.0, alpha=0.9
        )
        self._final_line, = self.axis.plot([], [], color="red", linewidth=1.5)
        self._raw_line.set_label("Raw Blackwell")
        self._preprocessed_line.set_label("Preprocessed Blackwell")
        self._final_line.set_label("Postprocessed final")
        self.root.after(50, self._load_file)

    def _add_entry(self, parent: ttk.Frame, key: str, label: str, value: object) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text=label, width=20).pack(side=tk.LEFT)
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
        ttk.Label(row, text=label, width=20).pack(side=tk.LEFT)
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

        output = {
            "var_name": self.config["bottom"]["params"].get("var_name", "Sv"),
            "threshold": [
                number("threshold_sv"),
                number("threshold_theta"),
                number("threshold_phi"),
            ],
            "offset": number("offset"),
            "r0": int(number("r0")),
            "r1": int(number("r1")),
            "wtheta": int(number("wtheta")),
            "wphi": int(number("wphi")),
        }
        for key in ("preprocessing", "postprocessing"):
            if key == "preprocessing":
                output[key] = {
                    "method": self.fields["pre_method"].get(),
                    "window": int(number("pre_window")),
                    "median_window": int(number("pre_median_window")),
                    "max_window": int(number("pre_max_window")),
                    "max_gap": int(number("pre_gap")),
                }
            else:
                output[key] = {
                    "method": self.fields["post_method"].get(),
                    "window": int(number("post_window")),
                    "max_deviation_m": number("post_deviation"),
                    "max_gap": int(number("post_gap")),
                    "edge_fill": self.fields["post_edge_fill"].get(),
                    "max_edge_gap": int(number("post_edge_gap")),
                }
        output["refinement"] = {
            "enabled": self.fields["refinement_enabled"].get() == "enabled",
            "window_m": number("refinement_window"),
            "threshold_db": number("refinement_threshold"),
        }
        return output

    def _load_file(self) -> None:
        if self._load_future is not None and not self._load_future.done():
            return
        raw_path = self.raw_paths[self.file_index]
        self.status.configure(text=f"Loading {raw_path.name}...")
        self._load_future = self._executor.submit(self._prepare_file, raw_path)
        self.root.after(100, self._poll_load)

    def _prepare_file(self, raw_path: Path) -> tuple[Any, Any]:
        ed = open_raw(raw_path, self.config)
        try:
            input_cfg = self.config["input"]
            sv = ep.calibrate.compute_Sv(
                ed,
                waveform_mode=input_cfg["waveform_mode"],
                encode_mode=input_cfg["encoding_mode"],
                **calibration_kwargs(self.config),
            )
            dataset = ep.consolidate.add_splitbeam_angle(
                sv,
                ed,
                waveform_mode=input_cfg["waveform_mode"],
                encode_mode=input_cfg["encoding_mode"],
                pulse_compression=True,
                to_disk=False,
            ).load()
            return sv.load(), dataset
        finally:
            ed.cleanup_swap_files()

    def _poll_load(self) -> None:
        if self._closed:
            return
        if self._load_future is None or not self._load_future.done():
            self.root.after(100, self._poll_load)
            return
        try:
            self.sv, self.dataset = self._load_future.result()
            self.status.configure(text="Loaded. Click Apply to detect bottom.")
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
        self._current_params = params
        self.status.configure(text="Calculating bottom line...")
        self._detect_future = self._executor.submit(self._detect_stages, params)
        self.root.after(100, self._poll_detection)

    def _detect_stages(self, params: dict[str, Any]) -> tuple[Any, Any, Any]:
        raw_params = dict(
            params,
            offset=0.0,
            preprocessing={"method": "none"},
            postprocessing={"method": "none"},
            refinement={"enabled": False},
        )
        pre_params = dict(
            params,
            offset=0.0,
            postprocessing={"method": "none"},
            refinement={"enabled": False},
        )
        raw = detect_bottom(self.dataset, params=raw_params)
        preprocessed = detect_bottom(self.dataset, params=pre_params)
        final = postprocess_bottom_line(
            preprocessed,
            options=params.get("postprocessing"),
        )
        final = refine_bottom_line(
            self.dataset,
            final,
            options=params.get("refinement"),
        )
        offset = float(params.get("offset", 0.0))
        if offset:
            final = final + offset
        return raw, preprocessed, final

    def _poll_detection(self) -> None:
        if self._closed:
            return
        if self._detect_future is None or not self._detect_future.done():
            self.root.after(100, self._poll_detection)
            return
        try:
            self._render(*self._detect_future.result())
        except Exception as exc:
            self.status.configure(text=f"Could not calculate bottom: {exc}")

    def _render(self, raw: Any, preprocessed: Any, final: Any) -> None:
        if self.dataset is None:
            return
        sv = self.dataset["Sv"].isel(channel=0)
        range_dim = next(dim for dim in sv.dims if dim != "ping_time")
        stride = max(1, int(sv.sizes["ping_time"] / 2000))
        echogram = sv.transpose(range_dim, "ping_time").isel(ping_time=slice(None, None, stride))
        range_coord = self.dataset["echo_range"].isel(channel=0)
        range_coord = range_coord.transpose(range_dim, "ping_time").isel(
            ping_time=slice(None, None, stride)
        )
        if self.image is not None:
            self.image.remove()
        self.image = self.axis.pcolormesh(
            echogram["ping_time"].values,
            range_coord.values,
            echogram.values,
            shading="auto",
            cmap="viridis",
            vmin=-100,
            vmax=-30,
        )
        self._raw_line.set_data(raw["ping_time"].values, raw.values)
        self._preprocessed_line.set_data(preprocessed["ping_time"].values, preprocessed.values)
        self._final_line.set_data(final["ping_time"].values, final.values)
        self.axis.set(
            title=f"Blackwell bottom: {self.raw_paths[self.file_index].name}",
            xlabel="Ping time",
            ylabel="Echo range (m)",
        )
        range_values = np.asarray(range_coord.values)
        finite_ranges = range_values[np.isfinite(range_values)]
        if finite_ranges.size:
            self.axis.set_ylim(float(finite_ranges.max()), float(finite_ranges.min()))
        self.axis.legend(loc="best")
        valid = np.isfinite(final.values)
        coverage = float(valid.mean() * 100.0) if valid.size else 0.0
        self.status.configure(
            text=f"Ready. Detections: {valid.sum()}/{valid.size} ({coverage:.1f}%)."
        )
        self.canvas.draw_idle()

    def _change_file(self, step: int) -> None:
        if self._load_future is not None and not self._load_future.done():
            self.status.configure(text="Still loading the current file.")
            return
        self.file_index = (self.file_index + step) % len(self.raw_paths)
        self.dataset = None
        self.sv = None
        self._load_file()

    def _save_yaml(self) -> None:
        try:
            params = self._params()
        except ValueError as exc:
            self.status.configure(text=str(exc))
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"{self.raw_paths[self.file_index].stem}_bottom_params.yaml"
        output_path.write_text(
            yaml.safe_dump({"bottom": {"method": "blackwell", "params": params}}, sort_keys=False)
        )
        self.status.configure(text=f"Saved {output_path}")

    def close(self) -> None:
        self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, action="append")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/bottom-interactive")
    args = parser.parse_args()
    config = load_config(CONFIG)
    raw_paths = [
        path if path.is_absolute() else ROOT / path
        for path in (args.raw or sorted((ROOT / "data/raw").glob("*.raw")))
    ]
    root = tk.Tk()
    BottomInspector(root, raw_paths, config, args.output_dir)
    root.mainloop()


if __name__ == "__main__":
    main()
