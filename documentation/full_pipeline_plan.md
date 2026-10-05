# EK80 FM Single-Target Pipeline Plan

## Purpose

This plan refines `Plan_single_targets2.txt` using the current implementation, the validated FM single-target prototype, and the design recommendations in `original_plan.pdf`.

The implementation priority is **module-by-module validation**, not immediate end-to-end orchestration. Each module should have a stable contract, focused tests, and a small real-data check before it is connected to the next module.

The pipeline will process one EK80 `.raw` file at a time:

```text
EK80 FM RAW
    |
    +-- navigation from SQLite -> ping-aligned position
    |
    +-- FM calibration/pulse compression -> Sv and Sp products
                                      |
                                      +-- bottom boundary
                                      +-- surface/turbidity boundary
                                      +-- split-beam angles
                                      +-- single-target detection
                                      +-- target TS
                                      +-- target navigation/depth
                                      +-- CSV export
```

## Current status

Already available or demonstrated:

- EK80 RAW opening through Echopype.
- SQLite navigation reading, platform filtering, and ping-time interpolation
  in `navigation.py`.
- Ping-aligned latitude and longitude inserted into
  `EchoData["Platform"]` through Echopype's `update_platform()` API.
- Real-data coordinate checks for:
  - `SAILOR1` using the June 27, 2020 EK80 FM RAW file;
  - `SAILOR2` using the April 26, 2026 EK80 FM RAW file.
- Per-run platform selection through the `--platform` command-line override;
  separate YAML files per platform are not required.
- An xarray surface threshold/run-length implementation in `surface.py`.
- A calibrated Blackwell seafloor wrapper in `bottom.py`, returning a separate
  `bottom_range(ping_time)` product and masking Echopype's no-intercept
  sentinel. The pipeline now runs bottom detection immediately before the
  separate surface/turbidity product, using the same calibrated `Sv`.
- `scripts/inspect_bottom_interactive.py` provides a human-in-the-loop view of
  the configured bottom detector for future tuning.
- A validated FM prototype in `scripts/test_fm2.py`:
  - FM `Sp` calculation;
  - pulse-compressed split-beam angles;
  - `sample_interval` and `tau_effective` preparation;
  - experimental single-target detection;
  - Echopype TS calculation;
  - CSV export.

Not yet complete:

- A reusable FM adapter in `fm.py`.
- A tested, Echopype-compatible surface/turbidity module with a defined scientific boundary contract.
- Bottom and surface integration with target filtering.
- Transducer-depth handling.
- A stable target product and fully integrated pipeline.

## Scientific data conventions

### Transducer depth

Transducer depth is required for interpreting acoustic range as water-column depth. It must be configured explicitly in `config/default.yaml`:

```yaml
input:
  transducer_depth_m: null
```

`null` means that the survey-specific value has not been supplied and absolute target depth must not be reported as valid. When populated, the value is in metres and must be read by every relevant script and module, including:

- the production pipeline;
- target-property and CSV preparation;
- real-data validation scripts;
- plotting or echogram-overlay scripts that display depth;
- any future bottom/surface conversion code.

The shared configuration loader should validate the value and expose one consistent name. Acoustic `range_m` must remain distinct from `depth_m`; the conversion must be explicit and documented, for example:

```text
water depth = transducer depth + acoustic range
```

The exact sign convention must be confirmed against the survey metadata before production use. A missing transducer depth should produce a clear validation error for depth-dependent output, rather than silently treating range as depth.

### Shared products

Use these products as the boundaries between modules:

1. `EchoData` and the selected acoustic beam group.
2. Ping-aligned navigation stored in the `EchoData["Platform"]` group, containing
   `latitude` and `longitude`; the project may retain a validation view with
   `navigation_valid` for diagnostics.
3. Calibrated `Sv` for bottom and surface analysis.
4. Calibrated `Sp` with pulse-compressed split-beam angles, `sample_interval`, and `tau_effective` for target detection and TS.
5. One-dimensional bottom and surface boundaries indexed by `ping_time`.
6. Target dataset containing target ping time, acoustic range, angles, TS, and optional navigation/depth fields.

The configured navigation platform is the default for a run. A per-file
platform override should be used when processing files from another platform;
separate configuration files are not required.

The default is configured in `config/default.yaml`:

```yaml
navigation:
  platform: SAILOR1
```

For a RAW file belonging to another platform, override it for that run:

```text
echopype-fm run --raw FILE.raw --navigation-db DB.sqlite \
    --config config/default.yaml --platform SAILOR2
```

The navigation test script supports the same pattern:

```text
python scripts/test_navigation.py --raw data/raw/FILE.raw \
    --platform SAILOR2
```

The platform selection must remain explicit because multiple platform tracks
can overlap in time. A successful interpolation with the wrong platform would
otherwise produce plausible but scientifically incorrect coordinates.

Attach ping-aligned navigation to `EchoData["Platform"]` through Echopype's
validated `update_platform()` API. A project-managed aligned dataset may be
returned as a diagnostic view, but latitude and longitude must not exist only
outside `EchoData`.

## Implementation sequence

### Phase 1: Configuration and contracts

1. Extend `config/default.yaml` with:
   - `input.transducer_depth_m`;
   - explicit waveform and encoding settings;
   - navigation interpolation and maximum-gap settings;
   - bottom parameters and bottom offset;
   - surface threshold, consecutive samples, and dead zone;
   - target exclusion settings;
   - output schema and filtering settings.
2. Update `config.py` to validate the structure and types. Require a real transducer depth before any output labelled `depth_m` is produced.
3. Verify the pinned Echopype commit `b26d049` in the actual environment. Record exact signatures, dimensions, variable names, and required detector fields for:
   - `compute_Sp`;
   - `add_splitbeam_angle`;
   - `compute_TS`;
   - `detect_seafloor`;
   - `detect_from_Sp`.
4. Add contract tests before implementation changes where practical. These tests should define expected dimensions and names without requiring proprietary RAW data.

### Phase 2: Navigation module

1. Retain `read_navigation()` and `align_navigation()` as the project navigation API.
2. Verify and test:
   - platform filtering;
   - UTC normalization;
   - sorting;
   - duplicate timestamp removal;
   - interpolation onto ping times;
   - no extrapolation outside the SQLite time range;
   - optional maximum-gap masking;
   - `navigation_valid` behavior.
3. Retain one source navigation record immediately before and after the
   requested RAW time range. These boundary records are interpolation
   brackets, not extrapolation: coordinates remain invalid when the SQLite
   data does not cover a ping.
4. Store the aligned coordinates in `EchoData["Platform"]` and verify that
   their length matches the acoustic ping count.
5. Add a separate helper for target-level lookup/interpolation only if the
   target ping times cannot reuse the aligned ping dataset.
6. Test malformed databases, empty platform selections, duplicate normalized
   timestamps, timezone variants, large gaps, and multiple platform choices.

### Phase 3: FM calibration and pulse compression

1. Replace the stubs in `src/echopype_fm_std/fm.py` with a reusable adapter based on the validated `scripts/test_fm2.py` flow.
2. The adapter should:
   - call `compute_Sp(waveform_mode="FM", encode_mode="complex")`;
   - call `add_splitbeam_angle(..., pulse_compression=True, to_disk=False)`;
   - attach `sample_interval`;
   - reconstruct `tau_effective` using Echopype's low-level functions;
   - return the full-channel dataset for TS;
   - provide a loaded single-channel view for the experimental detector.
3. Do not implement a second hand-written matched filter. The project adapter should isolate the experimental Echopype dependency and provide a stable project API.
4. Add mocked tests for call order, parameters, required variables, channel selection, and failures.
5. Add a local real-RAW test recording dimensions, angle variables, effective pulse duration, target-compatible variables, and memory requirements.

### Phase 4: Calibrated `Sv` preparation

1. Determine the exact Echopype call and dataset required for calibrated `Sv` at the pinned commit.
2. Put this behind a small project adapter rather than duplicating calibration in bottom and surface code.
3. Confirm the range coordinate and its units.
4. Confirm that transducer depth is not accidentally folded into acoustic range. Depth conversion belongs to a later, explicit target/boundary coordinate step.
5. Test the adapter with mocked Echopype output and one representative RAW file.

### Phase 5: Bottom detection

1. Keep `bottom.py` as the project boundary around `ep.mask.detect_seafloor()`.
   Blackwell receives the calibrated `Sv` plus pulse-compressed split-beam
   angles; the adapter supplies `depth` from `echo_range` because the current
   RAW products expose acoustic range rather than absolute depth.
2. Define a stable output schema, preferably a one-dimensional `bottom_range(ping_time)` in metres. Add `bottom_depth` only when the transducer-depth conversion is explicit and valid.
3. Forward the configured method, threshold, channel, and other parameters without hiding them.
4. Define how missing bottom values affect target filtering. Default behavior should preserve the quality state and avoid silently deleting all targets; strict mode may reject incomplete runs.
5. Add mocked wrapper tests and a local real-RAW acceptance check.

### Phase 6: Surface/turbidity detection

This is a first-class implementation task, not just orchestration.

1. Treat the current `surface.py` implementation as a numerical reference for the linked `find_waves.py` logic:
   - no input mutation;
   - NaN-safe processing;
   - per-ping threshold search;
   - consecutive-sample run detection;
   - first qualifying range coordinate.
2. Redesign the public API to match the style of Echopype functions, for example:

   ```python
   surface = detect_surface(
       sv,
       method="threshold",
       params={
           "wave_threshold_db": -77,
           "wave_threshold_layer_db": -68,
           "wave_consecutive_samples": 3,
           "beam_dead_zone_samples": 10,
           "range_bin_fraction": 0.1,
           "rolling_ping_window": 3,
       },
   )
   ```

3. Return an xarray object indexed by `ping_time`, with a documented `surface_range` variable and detection/quality metadata.
4. Keep `surface_depth` separate from `surface_range`. Derive depth only when `transducer_depth_m` is present and the coordinate convention is validated.
5. Add focused tests for exact run lengths, NaNs, dead zones, no detection, coordinate mapping, multiple pings, and non-mutation.
6. Compare the new implementation against reference cases from the original `find_waves.py` before changing the algorithm.
7. Validate scientifically whether the first below-threshold run represents the intended surface turbidity boundary. Algorithmic improvements should be a later, separately validated task.

Current implementation status: `detect_surface()` accepts either a calibrated
`Sv` DataArray or Dataset, supports the Echopype-style `method`/`params`
contract, and never mutates its input. Before threshold/run-length detection,
the range dimension is reduced to 10% of its original sample count by default
(`range_bin_fraction: 0.1`, equivalent to averaging 10 samples at a time);
this is configurable for tuning. The detector now ports the original
`find_waves.py` semantics: `wave_threshold_db=-77`, three consecutive samples,
a beam dead zone, the persistent-layer check, the `-68 dB` layer threshold,
and the deep-result retry. Echopype's `echo_range` variable is used for the
returned boundary in metres when available. `scripts/test_surface.py` runs
this detector on one or more EK80 RAW files, while retaining the original
`EchoData` during calibration and writes an echogram overlay for visual
inspection. The final boundary can also be conservatively expanded with a
centered rolling maximum across nearby pings; the default
`rolling_ping_window: 3` takes the deepest detected boundary from the previous,
current, and next ping, with `min_periods=1` at file edges. This is intended
to bridge small gaps between surface spikes and strong turbidity echoes.

The previous 5 m result was traced to two implementation errors: `-50 dB`
detected the first quiet background region rather than the original surface
spikes, and `range_bin_fraction` was incorrectly used as the coarsening window
instead of its reciprocal. Both are corrected; the regenerated RAW-set
results should be visually reviewed before selecting final survey parameters.

The final validated visual output is generated with:

```text
python scripts/test_surface.py --output-dir output/surface-rolling-all
```

The generated overlays are development/acceptance artifacts and are not
required pipeline inputs. Keep only the final overlay set needed for review;
intermediate threshold and coarsening runs can be removed safely.

### Phase 7: Single-target detection

1. Add a reusable detector adapter in `single_target.py` around the experimental `detect_from_Sp` import.
2. Preserve the validated prerequisites:
   - selected channel;
   - `.load()` before detection;
   - angle values in degrees;
   - `sample_interval`;
   - `tau_effective`;
   - detector-only parameter mapping.
3. Add target-domain boundary filtering after detection:
   - exclude targets above the surface boundary;
   - exclude targets below `bottom_range - bottom_offset_m`;
   - apply configured near-transducer and lower-range limits;
   - preserve a reason or quality flag where feasible.
4. Test with mocked detector output and a real-RAW smoke test.

### Phase 8: Target properties and CSV export

1. Keep the full-channel `Sp`/angle dataset loaded for `compute_TS`.
2. Verify the exact Echopype TS variables and sign convention:
   - uncompensated TS;
   - compensated TS;
   - beam compensation.
3. Extend `targets.py` to normalize one stable target schema containing:
   - `range_m`;
   - `depth_m` only when transducer depth is valid;
   - alongship position/angle;
   - athwartship position/angle;
   - compensated TS;
   - uncompensated TS;
   - target ping time;
   - optional navigation and quality fields.
4. Make the CSV schema explicit and ordered. The required public fields are:
   - depth or clearly named range;
   - alongship beam position;
   - athwartship beam position;
   - compensated TS;
   - uncompensated TS.
5. Do not automatically export every varying detector diagnostic. Metadata columns must be opt-in.
6. Apply TS filters before export, preserve one row per target, avoid an index column, and write a valid header when no targets remain.
7. Add unit tests for normalization, depth conversion, filtering, empty output, exact headers, and one-row-per-target behavior.

### Phase 9: Incremental scripts before orchestration

Create or update small scripts that exercise modules independently:

1. `inspect_raw.py`: input dimensions, channels, timestamps, and configured transducer depth.
2. `test_navigation.py`: SQLite loading and ping alignment diagnostics.
3. `test_fm.py` or a replacement reference script: FM calibration, pulse compression, angles, and effective pulse duration.
4. `inspect_bottom_interactive.py`: optional visual inspection of bottom output.
5. `test_surface.py`: surface output compared with reference cases and plotted against `Sv`.
6. `test_single_target.py`: detector count, ranges, angles, and boundary filtering.
7. `test_targets.py`: TS calculation, depth conversion, and exact CSV output.
8. `plot_targets.py`: read the configured depth/range convention and transducer depth from YAML when overlaying or labelling results.
9. `test_navigation.py`: accept `--raw` and optional `--platform`, print the
   first five ping coordinates, and report stored-coordinate coverage.

Each script should load configuration through the shared loader. Avoid independent YAML parsing and avoid hard-coded transducer depth.

### Phase 10: Orchestration last

Only after the modules above pass their focused checks:

1. Replace the placeholder `run_pipeline()` with explicit calls to the validated modules.
2. Use one shared `Sv` product for bottom and surface and one shared `Sp`/angle product for detection and TS.
3. Return named products, completed stages, warnings, quality counts, and the output path.
4. Move EchoData swap cleanup into pipeline-level `try/finally` while xarray is still available.
5. Update the CLI to support full runs, configuration overrides, and clear stage-specific failures.
6. Never emit a scientifically incomplete CSV after a failed or skipped stage.

## Testing strategy

Run focused tests after each module change:

```text
pytest tests/unit/test_navigation.py
pytest tests/unit/test_surface.py
pytest tests/unit/test_targets.py
pytest tests/unit/test_fm.py
pytest tests/unit/test_bottom.py
pytest tests/unit/test_pipeline.py
ruff check .
```

Then run the local real-data checks using one representative RAW file and the matching SQLite database. Record:

- Echopype commit;
- RAW dimensions and channel count;
- navigation coverage;
- bottom and surface coverage;
- effective pulse duration;
- target count before and after boundary filtering;
- compensated and uncompensated TS distributions;
- exact CSV header and row count;
- transducer depth and depth/range convention.

The current navigation acceptance checks are:

```text
python scripts/test_navigation.py \
    --raw data/raw/SLUAquaSailor2020V2-Phase0-D20200627-T060144-0.raw \
    --platform SAILOR1

python scripts/test_navigation.py \
    --raw data/raw/SLUAquaSailor2020V1-Phase0-D20260426-T032616-0.raw \
    --platform SAILOR2
```

Both checks insert coordinates into `EchoData["Platform"]`. The validated
results are 255/255 valid navigation pings for the first file and 209/209 for
the second file. The first five timestamps, latitudes, and longitudes are
printed for direct inspection.

Inspect representative echograms with surface and bottom overlays. Test missing navigation, out-of-range navigation, large gaps, missing boundaries, empty detections, multiple platforms, and missing transducer depth.

## Scope boundaries

Initially exclude multi-file batching, Dask/block optimization, upstream Echopype changes, algorithmic improvement of the surface detector, and production scientific claims. These can follow once the per-module behavior is reproducible and validated.

The most important unresolved scientific decision is the exact meaning of target `depth`. Until transducer depth and its sign convention are confirmed, the safe public name is `range_m`; `depth_m` must require a configured and validated `transducer_depth_m`.
