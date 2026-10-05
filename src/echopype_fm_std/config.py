from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a pipeline configuration is invalid."""


def calibration_kwargs(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Translate project calibration settings to Echopype keyword arguments."""
    environment = cfg.get("environment", {})
    calibration = cfg.get("calibration", {})
    if not isinstance(environment, dict):
        raise ConfigError("environment must be a mapping.")
    if not isinstance(calibration, dict):
        raise ConfigError("calibration must be a mapping.")

    env_params = {
        key: environment[key]
        for key in ("temperature", "salinity", "pressure")
        if key in environment and environment[key] is not None
    }
    cal_params: dict[str, Any] = {}
    for key in ("gain_correction", "sa_correction", "equivalent_beam_angle"):
        if key in calibration and calibration[key] is not None:
            cal_params[key] = calibration[key]
    if "equivalent_beam_angle" not in cal_params and "equialent_beam_angle" in calibration:
        cal_params["equivalent_beam_angle"] = calibration["equialent_beam_angle"]

    return {"env_params": env_params, "cal_params": cal_params}


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        raise ConfigError("Configuration root must be a mapping.")

    platform = cfg.get("navigation", {}).get("platform")
    if not platform or platform == "CHANGE_ME":
        raise ConfigError(
            "navigation.platform must be set. The tracking database contains multiple platforms."
        )

    return cfg
