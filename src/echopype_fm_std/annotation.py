from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def interpolate_surface_vertices(
    ping_indices: Iterable[float],
    ranges_m: Iterable[float],
    ping_count: int,
) -> np.ndarray:
    """Interpolate clicked surface vertices onto ping indices.

    Values outside the first and last click are NaN so an optimizer can
    distinguish unlabeled pings from an intentionally shallow or deep line.
    """
    x = np.asarray(list(ping_indices), dtype=float)
    y = np.asarray(list(ranges_m), dtype=float)
    if x.size != y.size or x.size < 2:
        raise ValueError("At least two ping/range vertices are required.")
    if ping_count < 1:
        raise ValueError("ping_count must be positive.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Annotation vertices must be finite.")

    order = np.argsort(x)
    x, y = x[order], y[order]
    unique_x, unique_indices = np.unique(x, return_index=True)
    unique_y = y[unique_indices]
    if unique_x.size < 2:
        raise ValueError("Annotation vertices must span at least two pings.")

    ping_axis = np.arange(ping_count, dtype=float)
    line = np.full(ping_count, np.nan, dtype=float)
    inside = (ping_axis >= unique_x[0]) & (ping_axis <= unique_x[-1])
    line[inside] = np.interp(ping_axis[inside], unique_x, unique_y)
    return line


def save_surface_annotation(
    output_csv: str | Path,
    *,
    raw_path: str | Path,
    ping_times: Iterable[Any],
    surface_ranges_m: Iterable[float],
    vertices: Iterable[tuple[float, float]],
    channel: int | str | None = None,
) -> tuple[Path, Path]:
    """Save a per-ping annotation CSV and its metadata JSON sidecar."""
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    times = list(ping_times)
    ranges = np.asarray(list(surface_ranges_m), dtype=float)
    if len(times) != ranges.size:
        raise ValueError("ping_times and surface_ranges_m must have equal length.")

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ping_index", "ping_time", "surface_range_m", "annotated"])
        for index, (ping_time, range_m) in enumerate(zip(times, ranges)):
            writer.writerow(
                [
                    index,
                    str(ping_time),
                    "" if not np.isfinite(range_m) else f"{range_m:.9g}",
                    int(np.isfinite(range_m)),
                ]
            )

    metadata_path = output_csv.with_suffix(".json")
    metadata = {
        "format": "echopype-fm-surface-annotation-v1",
        "raw_file": str(Path(raw_path).resolve()),
        "csv_file": output_csv.name,
        "channel": channel,
        "vertices": [
            {"ping_index": float(ping), "surface_range_m": float(range_m)}
            for ping, range_m in vertices
        ],
        "created_utc": datetime.now().astimezone().isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return output_csv, metadata_path


def load_surface_annotation(
    annotation_csv: str | Path,
) -> dict[str, Any]:
    """Load an annotation CSV and its optional metadata sidecar."""
    annotation_csv = Path(annotation_csv)
    with annotation_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"ping_index", "ping_time", "surface_range_m", "annotated"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{annotation_csv} does not contain the expected annotation columns.")

    ping_indices = np.asarray([int(row["ping_index"]) for row in rows], dtype=int)
    ping_times = np.asarray([row["ping_time"] for row in rows], dtype=str)
    ranges = np.asarray(
        [float(row["surface_range_m"]) if row["surface_range_m"] else np.nan for row in rows],
        dtype=float,
    )
    metadata_path = annotation_csv.with_suffix(".json")
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    return {
        "ping_index": ping_indices,
        "ping_time": ping_times,
        "surface_range_m": ranges,
        "annotated": np.isfinite(ranges),
        "metadata": metadata,
    }
