from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a pipeline configuration is invalid."""


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
