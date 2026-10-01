from __future__ import annotations

from pathlib import Path
from typing import Any

import echopype as ep


def open_raw(raw_path: str | Path, config: dict[str, Any]):
    """Open one EK80 RAW file with Echopype.

    This wrapper deliberately stays thin: the EchoData object remains the
    canonical Echopype representation until a processing stage needs a more
    convenient xarray dataset.
    """
    raw_path = Path(raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    if raw_path.suffix.lower() != ".raw":
        raise ValueError(f"Expected an EK80 .raw file, got: {raw_path}")

    input_cfg = config.get("input", {})
    sonar_model = input_cfg.get("sonar_model", "EK80")
    if sonar_model != "EK80":
        raise ValueError(f"This pipeline currently expects EK80, got {sonar_model!r}.")

    return ep.open_raw(
        raw_path,
        sonar_model=sonar_model,
        use_swap=True,
    )


def get_beam_group(ed, beam_group: str = "Beam_group1"):
    """Return an acoustic Beam_group from an EchoData object."""
    try:
        return getattr(ed.sonar, beam_group)
    except AttributeError as exc:
        raise ValueError(f"EchoData has no sonar.{beam_group}") from exc


def inspect_raw(raw_path: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    """Open an EK80 RAW file and return a compact metadata summary."""
    ed = open_raw(raw_path, config)
    beam_group = config.get("input", {}).get("beam_group", "Beam_group1")
    bg = get_beam_group(ed, beam_group)

    summary: dict[str, Any] = {
        "raw_path": str(Path(raw_path).resolve()),
        "beam_group": beam_group,
        "dimensions": dict(bg.sizes),
        "data_vars": sorted(bg.data_vars),
    }

    if "ping_time" in bg:
        ping_time = bg["ping_time"]
        summary["ping_count"] = int(ping_time.size)
        if ping_time.size:
            summary["start_time"] = str(ping_time.values[0])
            summary["end_time"] = str(ping_time.values[-1])

    return summary
