from __future__ import annotations

from typing import Any

import echopype as ep
import numpy as np
import xarray as xr
from echopype.calibrate.calibrate_ek import get_filter_coeff
from echopype.calibrate.ek80_complex import (
    get_tau_effective,
    get_transmit_signal,
)


class SingleTargetNotImplementedError(NotImplementedError):
    """Raised when FM single-target detection is not yet wired."""


def mask_target_dataset(
    ds_sp: xr.Dataset,
    surface_range: xr.DataArray,
    bottom_range: xr.DataArray,
) -> xr.Dataset:
    """Apply the detected water-column envelope before target detection."""
    from .integration import apply_boundary_mask

    return apply_boundary_mask(
        ds_sp,
        surface_range,
        bottom_range,
        variable="Sp",
    )


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


def prepare_fm_sp(echodata: Any, beam_group: xr.Dataset, input_cfg: dict[str, Any]) -> xr.Dataset:
    """Create the FM ``Sp``/angle product required by the target detector."""
    sp = ep.calibrate.compute_Sp(
        echodata,
        waveform_mode=input_cfg["waveform_mode"],
        encode_mode=input_cfg["encoding_mode"],
    )
    sp_angle = ep.consolidate.add_splitbeam_angle(
        sp,
        echodata,
        waveform_mode=input_cfg["waveform_mode"],
        encode_mode=input_cfg["encoding_mode"],
        pulse_compression=True,
        to_disk=False,
    )
    sp_angle["sample_interval"] = beam_group["sample_interval"]

    vendor = echodata["Vendor_specific"]
    tx_coeff = get_filter_coeff(vendor)
    tx, tx_time = get_transmit_signal(
        beam_group,
        tx_coeff,
        "BB",
        sp["receiver_sampling_frequency"],
    )
    sp_angle["tau_effective"] = get_tau_effective(
        ytx_dict=tx,
        fs_deci_dict={key: 1 / np.diff(value[:2]) for key, value in tx_time.items()},
        waveform_mode="BB",
        channel=beam_group["channel"],
        ping_time=beam_group["ping_time"],
    )
    return sp_angle
