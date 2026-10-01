from __future__ import annotations

from typing import Any

import xarray as xr


class SingleTargetNotImplementedError(NotImplementedError):
    """Raised when FM single-target detection is not yet wired."""


def detect_fm_targets(
    ds_sp: xr.Dataset,
    params: dict[str, Any],
) -> xr.Dataset:
    """Adapter around the experimental Echopype ``detect_from_Sp`` implementation.

    PR #1588 currently exposes the algorithm at a lower-level module rather
    than as a stable released FM public API. Keeping the import here isolates
    that dependency from the rest of the project.
    """
    try:
        from echopype.mask.single_target_detection.detect_from_Sp import detect_from_Sp
    except ImportError as exc:
        raise SingleTargetNotImplementedError(
            "The installed Echopype does not contain the experimental "
            "detect_from_Sp implementation. Install the PR #1588 development branch."
        ) from exc

    return detect_from_Sp(ds_sp, params=params)
