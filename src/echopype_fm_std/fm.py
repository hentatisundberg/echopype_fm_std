from __future__ import annotations

from typing import Any

import xarray as xr


class FMNotImplementedError(NotImplementedError):
    """Raised when an FM stage has not yet been wired to the validated backend."""


def pulse_compress_fm(ed, **kwargs: Any) -> xr.Dataset:
    """Project-level adapter for EK80 FM pulse compression.

    The implementation is intentionally left as an adapter because the FM
    calibration API in Echopype PR #1588 is still experimental. Once the exact
    function/return schema has been validated against a real RAW file, this
    function should become the single project entry point for FM pulse
    compression.
    """
    raise FMNotImplementedError(
        "FM pulse compression is not wired yet. Install the validated Echopype "
        "PR #1588 environment and implement this adapter against the exact API."
    )


def compute_fm_sp_and_angles(ed, **kwargs: Any) -> xr.Dataset:
    """Project-level adapter for FM Sp and split-beam angle calculation."""
    raise FMNotImplementedError(
        "FM Sp/angle calculation is not wired yet. This is the next implementation stage."
    )
