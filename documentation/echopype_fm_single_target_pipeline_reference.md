# EK80 FM single-target processing pipeline

## Purpose

Current tested prototype for processing a Simrad EK80 FM `.raw` file
with Echopype and extracting single-target detections and target
strength (TS).

``` text
EK80 FM .raw
    ↓
Echopype EchoData
    ↓
FM calibration → Sp
    ↓
Pulse-compressed split-beam angles
    ↓
Add detector-specific quantities
    ↓
Select one acoustic channel
    ↓
Single-target detection
    ↓
Target-strength calculation
    ↓
Target table
```

Navigation, bottom detection, surface/turbidity detection, and CSV
export are not yet integrated.

## 1. Input and configuration

The test file is:

`data/raw/SLUAquaSailor2020V2-Phase0-D20200627-T060144-0.raw`

It is opened as an EK80 with `use_swap=True`. The acoustic data are in
`Sonar/Beam_group1`.

Single-target parameters are read from `config/default.yaml` under
`single_target.params`. These researcher-selected detection settings
should remain in YAML rather than being hard-coded.

## 2. FM calibration

``` python
sp = ep.calibrate.compute_Sp(
    ed,
    waveform_mode="FM",
    encode_mode="complex",
)
```

This produces calibrated point scattering strength (`Sp`) and associated
range, environmental, beam and calibration variables.

For the current file:

-   1 acoustic channel
-   255 pings
-   17,149 range samples
-   nominal frequency 200 kHz

`Sp` has dimensions `(channel, ping_time, range_sample)`.

## 3. Pulse-compressed split-beam angles

``` python
sp_angle = ep.consolidate.add_splitbeam_angle(
    sp,
    ed,
    waveform_mode="FM",
    encode_mode="complex",
    pulse_compression=True,
    to_disk=False,
)
```

This adds `angle_alongship` and `angle_athwartship`.

The angles are in degrees. The experimental single-target detector
converts them internally, so they should not be pre-converted to
radians.

## 4. Variables required by the experimental detector

The detector expects `sample_interval` and `tau_effective`, which are
not both carried directly by `compute_Sp()`.

Sample interval:

``` python
sp_angle["sample_interval"] = bg["sample_interval"]
```

Effective pulse duration is reconstructed using Echopype's own low-level
functions:

``` python
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
```

The `"BB"` mode in this low-level reconstruction is intentional and
follows the processing route used by Echopype for the reconstructed
complex transmit signal.

For this file:

-   `tau_effective ≈ 14.73 µs`
-   `sample_interval = 8 µs`
-   effective pulse length ≈ 1.84 samples

## 5. Preparing the detector input

The experimental detector expects `Sp` without a channel dimension:

``` text
('ping_time', 'range_sample')
```

Therefore:

``` python
sp_st = sp_angle.isel(channel=0).load()
```

The `.load()` is required because the experimental detector uses xarray
operations without enabling Dask handling. Loading the selected channel
avoids the Dask chunking error encountered during testing.

## 6. Single-target detection

``` python
from echopype.mask.single_target_detection.detect_from_Sp import detect_from_Sp

targets = detect_from_Sp(
    sp_st,
    params,
)
```

The detector identifies candidate single-target locations using local
maxima, beam compensation, pulse/envelope characteristics, normalized
pulse length, angle variability, and the configured thresholds.

The resulting Dataset has dimension `single_target` and includes:

-   `ping_time`
-   `range_sample`
-   `single_target_range`
-   `single_target_alongship_angle`
-   `single_target_athwartship_angle`
-   angle standard deviations
-   `iinf`, `isup`
-   `pulse_len_samples`
-   `norm_pulse_len`
-   `beam_comp_db`
-   `plike_peak`
-   `frequency_nominal`
-   `ping_index`

`plike_peak` is a detection-domain quantity and is not the final TS.

The initial settings produced about 39,000 targets over 255 pings.
Changing the detection settings changed the target count, as expected.

## 7. Target strength

Detected locations are passed to Echopype:

``` python
sp_for_ts = sp_angle.load()

targets_ts = ep.calibrate.compute_TS(
    sp_for_ts,
    point_locations=targets,
)
```

The resulting target dataset contains:

-   `uncompensated_TS`: TS at the target peak without beam-position
    compensation
-   `compensated_TS`: TS after applying the split-beam beam-position
    correction

Both are target strength in dB re 1 m².

The planned output mapping is:

  Desired output         Current variable
  ---------------------- -----------------------------------
  ping time              `ping_time`
  range/depth            `single_target_range`
  alongship position     `single_target_alongship_angle`
  athwartship position   `single_target_athwartship_angle`
  TS uncompensated       `uncompensated_TS`
  TS compensated         `compensated_TS`

Keep range distinct from water-column depth until navigation and
transducer-depth information are incorporated.

## 8. Memory/Dask considerations

The current prototype deliberately loads:

``` python
sp_st = sp_angle.isel(channel=0).load()
sp_for_ts = sp_angle.load()
```

This is reasonable for the current one-file test. Production processing
should initially remain one file at a time. If files become too large,
possible future approaches are block processing, Dask-aware detector
code, or upstream improvements to the experimental detector.

## 9. Echopype dependency

The project currently pins the experimental Echopype commit:

``` toml
echopype @ git+https://github.com/LOCEANlloydizard/echopype.git@b26d049
```

The single-target functionality is experimental, so this exact commit is
part of the reproducible processing environment.

## 10. Swap-file cleanup

With `use_swap=True`, the current Echopype implementation creates
temporary Zarr-backed data. `EchoData` has `cleanup_swap_files()` and
its destructor calls that method when `converted_raw_path` is `None`.

At Python interpreter shutdown, the destructor can run after xarray
modules have already been unloaded, producing:

``` text
Exception ignored in: <function EchoData.__del__ ...>
ModuleNotFoundError: import of xarray.core.formatting halted; None in sys.modules
```

This is a cleanup/destructor issue, not an acoustic-processing failure.

The working script should therefore perform swap cleanup explicitly
while Python and xarray are still initialized, then prevent the
destructor from trying to clean the same object again.

## 11. Next development stages

The tested acoustic core is now:

``` text
.raw
 ↓
Sp
 ↓
pulse-compressed split-beam angles
 ↓
single-target detection
 ↓
TS
```

The complete pipeline will add:

1.  Navigation from the project SQLite database, including platform
    filtering and ping-time interpolation.
2.  Sea-floor detection and bottom exclusion.
3.  Surface/turbidity detection and upper-water-column exclusion.
4.  Target-level navigation and geographic position.
5.  CSV export with one row per target.
6.  Provenance and processing metadata.

## 12. Scientific validation still required

Before production use, validate:

-   TS distributions under different detection thresholds
-   target density per ping
-   range distributions
-   beam-position distributions
-   compensated versus uncompensated TS
-   detections against representative echograms
-   sensitivity to pulse-length thresholds
-   sensitivity to beam-compensation limits
-   sensitivity to angle-SD thresholds
-   performance in low-noise and dense-scattering situations
-   comparison with an independent/reference processing method where
    available

A large number of detections is not by itself evidence of erroneous
detection: dense fields of weak scatterers can legitimately produce many
targets. TS distributions and echogram inspection are therefore
important.

## Current status

**Working:** EK80 FM raw → calibrated Sp → pulse-compressed split-beam
angles → experimental single-target detection → uncompensated and
beam-compensated TS.

**Not yet integrated:** navigation, bottom detection, surface/turbidity
exclusion, and CSV export.

**Reproducibility:** keep the experimental Echopype commit pinned until
the single-target API stabilizes.
