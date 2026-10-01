from pathlib import Path

import echopype as ep
import numpy as np
import yaml

from echopype.calibrate.calibrate_ek import get_filter_coeff
from echopype.calibrate.ek80_complex import (
    get_tau_effective,
    get_transmit_signal,
)
from echopype.mask.single_target_detection.detect_from_Sp import detect_from_Sp

from echopype_fm_std.targets import export_targets_csv


RAW = Path(
    "data/raw/SLUAquaSailor2020V2-Phase0-D20200627-T060144-0.raw"
)
CONFIG = Path("config/default.yaml")


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

with CONFIG.open() as f:
    config = yaml.safe_load(f)

params = config["single_target"]["params"]
detection_params = {
    key: value
    for key, value in params.items()
    if key not in {"TS_comp_min", "TS_comp_max"}
}
output_dir = Path(config.get("output", {}).get("directory", "output"))
output_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Open EK80 raw file
# ---------------------------------------------------------------------

ed = None

try:
    ed = ep.open_raw(
        RAW,
        sonar_model="EK80",
        use_swap=True,
    )

    bg = ed["Sonar/Beam_group1"]
    vend = ed["Vendor_specific"]

    print(f"Echopype {ep.__version__}")
    print(f"RAW: {RAW}")


    # -----------------------------------------------------------------
    # FM calibration
    # -----------------------------------------------------------------

    print("Computing FM Sp...")

    sp = ep.calibrate.compute_Sp(
        ed,
        waveform_mode="FM",
        encode_mode="complex",
    )


    # -----------------------------------------------------------------
    # Pulse-compressed split-beam angles
    # -----------------------------------------------------------------

    print("Computing pulse-compressed split-beam angles...")

    sp_angle = ep.consolidate.add_splitbeam_angle(
        sp,
        ed,
        waveform_mode="FM",
        encode_mode="complex",
        pulse_compression=True,
        to_disk=False,
    )


    # -----------------------------------------------------------------
    # Prepare variables required by experimental detector
    # -----------------------------------------------------------------

    sp_angle["sample_interval"] = bg["sample_interval"]


    # Reconstruct the transmitted signal and calculate effective
    # pulse duration using Echopype's own low-level FM processing.
    tx_coeff = get_filter_coeff(vend)
    fs = sp["receiver_sampling_frequency"]

    tx, tx_time = get_transmit_signal(
        bg,
        tx_coeff,
        "BB",
        fs,
    )

    tau_effective = get_tau_effective(
        ytx_dict=tx,
        fs_deci_dict={
            k: 1 / np.diff(v[:2])
            for k, v in tx_time.items()
        },
        waveform_mode="BB",
        channel=bg["channel"],
        ping_time=bg["ping_time"],
    )

    sp_angle["tau_effective"] = tau_effective


    # -----------------------------------------------------------------
    # Single-target detection
    # -----------------------------------------------------------------

    # The experimental detector expects a single channel and currently
    # performs xarray operations that are not Dask-aware.
    sp_st = sp_angle.isel(channel=0).load()

    print("Detecting single targets...")

    targets = detect_from_Sp(
        sp_st,
        detection_params,
    )

    n_targets = targets.sizes.get("single_target", 0)

    print(f"Detected {n_targets:,} targets")


    # -----------------------------------------------------------------
    # Target strength
    # -----------------------------------------------------------------

    print("Calculating target strength...")

    # compute_TS uses the original channel dimension.
    sp_for_ts = sp_angle.load()

    targets_ts = ep.calibrate.compute_TS(
        sp_for_ts,
        point_locations=targets,
    )

    target_csv = output_dir / f"{RAW.stem}_single_targets.csv"
    export_targets_csv(
        targets_ts,
        target_csv,
        include_metadata_columns=config.get("output", {}).get(
            "include_metadata_columns", False
        ),
        ts_comp_min=params.get("TS_comp_min"),
        ts_comp_max=params.get("TS_comp_max"),
    )
    print(f"Exported filtered targets to {target_csv}")


    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    print("Target-strength summary:")

    for var in ["uncompensated_TS", "compensated_TS"]:
        values = targets_ts[var].values

        print(
            f"  {var}: "
            f"median={np.nanmedian(values):.2f} dB, "
            f"range={np.nanmin(values):.2f}–{np.nanmax(values):.2f} dB"
        )

    mean_comp = float(targets_ts["beam_comp_db"].mean())

    print(f"  mean beam compensation: {mean_comp:.2f} dB")


finally:
    # Echopype's EchoData destructor can attempt swap-file cleanup
    # during Python interpreter shutdown, after xarray has already
    # started unloading. Clean up explicitly while everything is alive.
    if ed is not None:
        try:
            if ed.converted_raw_path is None:
                ed.cleanup_swap_files()
        finally:
            # Prevent EchoData.__del__ from attempting the cleanup again
            # during interpreter shutdown.
            ed.converted_raw_path = RAW