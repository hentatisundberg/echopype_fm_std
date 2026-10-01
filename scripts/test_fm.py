from pathlib import Path

import echopype as ep
import numpy as np
from echopype.calibrate.ek80_complex import get_tau_effective
from echopype.calibrate.ek80_complex import get_transmit_signal
from echopype.calibrate.calibrate_ek import get_filter_coeff


RAW = Path("data/raw/SLUAquaSailor2020V2-Phase0-D20200627-T060144-0.raw")


# Load config
import yaml
with open('config/default.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

# Read single target settings
params = config["single_target"]["params"]
    


print("Echopype version:", ep.__version__)
print("RAW:", RAW)

ed = ep.open_raw(
    RAW,
    sonar_model="EK80",
    use_swap=True,
)

vend = ed["Vendor_specific"]


print("\n=== Sonar ===")
print(ed["Sonar"])

print("\n=== waveform_encode_descr ===")
if "waveform_encode_descr" in ed["Sonar"]:
    print(ed["Sonar"]["waveform_encode_descr"])
else:
    print("NOT PRESENT")

print("\n=== Beam groups ===")

for name in ["Beam_group1", "Beam_group2", "Beam_group3"]:
    path = f"Sonar/{name}"

    try:
        bg = ed[path]
    except Exception:
        continue

    if bg is None:
        print(f"\n--- {path}: not present ---")
        continue

    print(f"\n--- {path} ---")
    print("Dimensions:", dict(bg.sizes))

    print("Variables:")
    for v in bg.data_vars:
        print("   ", v)

    if "channel" in bg:
        print("Channels:", list(bg["channel"].values))

    for v in [
        "frequency_nominal",
        "transmit_frequency_start",
        "transmit_frequency_stop",
        "transmit_duration_nominal",
        "sample_interval",
        "backscatter_r",
        "backscatter_i",
    ]:
        if v in bg:
            print(f"{v}: dims={bg[v].dims}, shape={bg[v].shape}")

print("\n=== Computing FM Sp ===")

sp = ep.calibrate.compute_Sp(
    ed,
    waveform_mode="FM",
    encode_mode="complex",
)

print(sp)

print("\n=== Sp variables ===")
for v in sp.data_vars:
    print("  ", v, sp[v].dims, sp[v].shape)


print("\n=== Computing pulse-compressed split-beam angles ===")

sp_angle = ep.consolidate.add_splitbeam_angle(
    sp,
    ed,
    waveform_mode="FM",
    encode_mode="complex",
    pulse_compression=True,
    to_disk=False,
)

print(sp_angle)


print("\n=== Angle variables ===")
for v in [
    "angle_alongship",
    "angle_athwartship",
]:
    if v in sp_angle:
        print(
            f"{v}: dims={sp_angle[v].dims}, "
            f"shape={sp_angle[v].shape}"
        )
    else:
        print(f"{v}: NOT FOUND")

# Add quantities required by the experimental single-target detector
bg = ed["Sonar/Beam_group1"]
vend = ed["Vendor_specific"]

# Sample interval
sp_angle["sample_interval"] = bg["sample_interval"]


# Reconstruct the transmitted FM chirp and calculate effective pulse duration
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
        for (k, v) in tx_time.items()
    },
    waveform_mode="BB",
    channel=bg["channel"],
    ping_time=bg["ping_time"],
)

sp_angle["tau_effective"] = tau_effective

# ------------------------------------------------------------
# Add variables required by detect_from_Sp
# ------------------------------------------------------------

sp_angle["sample_interval"] = ed["Sonar/Beam_group1"]["sample_interval"]
sp_angle["tau_effective"] = tau_effective

# Detector operates on one channel at a time
sp_st = sp_angle.isel(channel=0).load()




print("\n=== FM single-target detector ===")

from echopype.mask.single_target_detection.detect_from_Sp import detect_from_Sp
import inspect

print("Function:", detect_from_Sp)
print("\nSignature:")
print(inspect.signature(detect_from_Sp))

print("\nDocstring:")
print(inspect.getdoc(detect_from_Sp))

print("\n=== Inspecting detector source ===")

import inspect

source = inspect.getsource(detect_from_Sp)
print(source)


from echopype.mask.single_target_detection.detect_from_Sp import (
    _validate_params,
    _validate_from_Sp_dataset,
    _phase1_simple,
)

print("\n=== Detector parameter validation ===")
print(inspect.getsource(_validate_params))

print("\n=== Detector dataset validation ===")
print(inspect.getsource(_validate_from_Sp_dataset))

print("\n=== Phase 1 detector ===")
print(inspect.getsource(_phase1_simple))

from echopype.mask.single_target_detection.detect_from_Sp import (
    _nech_p_samples,
    _validate_params,
)

print("\n=== Number of pulse samples ===")
print(inspect.getsource(_nech_p_samples))

print("\n=== Parameter validation ===")
print(inspect.getsource(_validate_params))


print("\n=== Detector parameter definitions ===")

import echopype.mask.single_target_detection.detect_from_Sp as detmod

print("REQUIRED_PARAMS:")
print(detmod.REQUIRED_PARAMS)

print("\nOPTIONAL_PARAMS:")
print(detmod.OPTIONAL_PARAMS)


# Print single target parameters
print("\n=== Single-target parameters ===")
for key, value in params.items():
    print(f"{key}: {value}")



print("\n=== Effective pulse duration ===")
print(tau_effective)

print("\n=== Sample interval ===")
print(sp_angle["sample_interval"])

print("\n=== Effective pulse length in samples ===")
print(
    tau_effective.values / sp_angle["sample_interval"].values
)


# Run detector 
print("\n=== Detecting single targets ===")

from echopype.mask.single_target_detection.detect_from_Sp import detect_from_Sp

targets = detect_from_Sp(
    sp_st,
    params,
)

print(targets)

print("\n=== Detection result ===")
print("Number of targets:", targets.sizes.get("single_target", 0))

for v in targets.data_vars:
    print(f"{v}: dims={targets[v].dims}, shape={targets[v].shape}")


print("\n=== Calculating target strength ===")

# compute_TS expects the original Sp dataset, including the channel dimension,
# plus the target locations returned by detect_from_Sp().
sp_for_ts = sp_angle.load()

targets_ts = ep.calibrate.compute_TS(
    sp_for_ts,
    point_locations=targets,
)

print(targets_ts)


print("\n=== Target strength summary ===")

for var in ["uncompensated_TS", "compensated_TS"]:
    x = targets_ts[var].values

    print(
        f"{var}: "
        f"min={np.nanmin(x):.2f} dB, "
        f"median={np.nanmedian(x):.2f} dB, "
        f"max={np.nanmax(x):.2f} dB"
    )

print(
    "Mean beam compensation:",
    float(targets_ts["beam_comp_db"].mean()),
    "dB"
)


# Explicitly clean up Echopype resources before interpreter shutdown
try:
    ed.close()
except Exception:
    pass