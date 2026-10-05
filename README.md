# echopype_fm_std

A reproducible EK80 FM-mode processing pipeline built around [Echopype](https://github.com/echostack-org/echopype).

The intended pipeline is:

1. read one EK80 `.raw` file;
2. align geographic position from a SQLite tracking database, filtered by `platform`;
3. pulse-compress FM signals;
4. detect seafloor;
5. detect the surface/turbidity layer;
6. perform single-target detection;
7. export one row per target to CSV.

The repository intentionally keeps project-specific code in `src/echopype_fm_std/` rather than modifying Echopype itself. Experimental FM single-target functionality from Echopype PR #1588 should be accessed through adapters in this repository, so the rest of the pipeline does not depend on an unstable upstream API.

## Environment

Echopype 0.11.1 requires Python >=3.11 and <3.14. Python 3.12 is recommended for this project.

Create the environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## First smoke test

```bash
source .venv/bin/activate
python -m echopype_fm_std --help
```

Inspect an EK80 RAW file before running processing:

```bash
echopype-fm inspect --raw /path/to/file.raw
```

## Configuration

Copy and edit `config/default.yaml`. The `navigation.platform` value is required because the SQLite database contains positions for multiple platforms.

See which platform values are present in the database:

```bash
echopype-fm nav-platforms --navigation-db /path/to/positions.sqlite
```

Run the current starter pipeline:

```bash
echopype-fm run \
  --raw /path/to/file.raw \
  --config config/default.yaml
```

The surface detector can be tested against all local RAW files and produces
echogram overlays without replacing the retained `EchoData` object:

```bash
python scripts/test_surface.py --output-dir output/surface-rolling-all
```

Pass `--raw FILE` more than once to select a subset. The script computes
calibrated `Sv`, detects the boundary, and closes only temporary Echopype swap
files; the surface result remains a separate xarray object that can be used
alongside `EchoData` by downstream stages. The default configuration follows
the original `find_waves.py` thresholds, averages the range dimension in
groups of ten samples, and applies a centered three-ping rolling maximum to
conservatively bridge short gaps. The output directory contains one
`*_surface.png` overlay per input RAW file.

The FM calibration, bottom detector, FM angle calculation, target detector,
target TS calculation, and final CSV writer have explicit adapter interfaces
but are intentionally not guessed or reimplemented until they are tested
against real FM data.

### Bottom conditioning

Blackwell conditioning can be enabled under `bottom.params` after comparing
it with the direct detector result. Preprocessing affects only the `Sv`
variable passed to Blackwell; split-beam angles remain unchanged. Available
preprocessing methods are `none`, `max`, `median`, `median_max`, and
`fill_dropouts`.
`max` and `median` use `window`, while `fill_dropouts` linearly fills only
internal gaps up to `max_gap` pings.
`median_max` applies a centered median with `median_window` first, then a
centered maximum with `max_window`; this is intended to suppress isolated
pelagic peaks before bridging short dropouts.

Postprocessing methods are `none`, `max`, `median`, `median_interpolate`, and
`max_interpolate`.
`max` applies a centered rolling minimum to the detected range, despite the
legacy method name: range increases downward, so this selects the shallowest
local detection and is conservative when masking below the bottom. `median`
applies a centered rolling median, while `median_interpolate` replaces isolated
deviations larger than `max_deviation_m` using
the local median and linearly fills only internal gaps up to `max_gap`.
`max_interpolate` combines this conservative rolling-minimum behavior with the bounded
internal-gap interpolation of `median_interpolate`.
Long gaps remain missing. For example:

```yaml
bottom:
  params:
    preprocessing:
      method: median_max
      median_window: 3
      max_window: 3
    postprocessing:
      method: median_interpolate
      window: 5
      max_deviation_m: 8
      max_gap: 3
      edge_fill: linear
      max_edge_gap: 0
    refinement:
      enabled: true
      window_m: 2.0
      threshold_db: -35.0
```

`edge_fill: linear` extrapolates only missing pings before the first or after
the last valid bottom detection. `nearest` carries the closest valid value
instead. `max_edge_gap: 0` allows the edge fill to cover any edge gap; set a
positive value to limit it. Internal gaps are never filled by this option.
When enabled, the final refinement searches the original `Sv` data within
`window_m` of the tracked line and chooses the shallowest sample at or above
`threshold_db`, but only at or above the tracked range. It can therefore move
the line upward to the top of the bottom echo, never downward onto a deeper
string echo.

The configured conditioning is applied by the pipeline after calibration and
before the bottom result is returned. The interactive inspector remains
available for future tuning and comparison against representative RAW files.

### Interactive bottom inspection

For human-in-the-loop tuning, use the Matplotlib inspector:

```bash
python scripts/inspect_bottom_interactive.py \
  --raw data/raw/SLUAquaSailor2020V1-Phase0-D20260503-T061158-1.raw
```

The inspector loads `config/default.yaml` and uses the same preprocessing,
Blackwell, postprocessing, and refinement stages as the pipeline. Its native
fields and method selectors control the Blackwell Sv/angle
thresholds, search range, preprocessing method/window/gap, and postprocessing
method/window/deviation/gap. Click `Apply` to run all three displayed stages:
the raw Blackwell result, Blackwell on the preprocessed `Sv`, and the final
postprocessed line. The echogram remains the original calibrated `Sv`, so
conditioning effects can be compared against the unmodified signal. File
loading and detector calculations run in the background so the window remains
interactive; the status line reports final-line coverage.
`Previous` and `Next` navigate through files passed with repeated `--raw`
arguments, and `Save YAML` writes the current parameters to the output
directory without modifying the main configuration. The y-axis is the
physical echo range in metres.

## Experimental Echopype branch

PR #1588 is still a draft. It contains prototype FM calibration/spectrum functionality and the single-target detection implementation, but the upstream public dispatcher/API is still evolving. Do not make the whole project depend on an unpinned moving branch.

For development involving the PR, install the branch explicitly in a clean environment and record the exact commit that was validated. For example:

```bash
python -m pip uninstall -y echopype
python -m pip install \
  "echopype @ git+https://github.com/LOCEANlloydizard/echopype.git@single_target_detection"
python -m pip freeze > requirements.lock.txt
```

Commit `requirements.lock.txt` once the exact environment has been validated; it is intentionally not ignored by Git. Later, when the relevant functionality is released by Echopype, switch the dependency back to a released version in `pyproject.toml`.

## Navigation database

Expected schema:

```sql
CREATE TABLE track_points (
    id INTEGER PRIMARY KEY,
    platform TEXT,
    survey_id TEXT,
    timestamp_utc TEXT,
    latitude REAL,
    longitude REAL,
    distance_m REAL,
    speed_ms REAL,
    is_interpolated INTEGER,
    source_file TEXT,
    created_at TEXT,
    UNIQUE(platform, timestamp_utc)
);
```

The reader:

- selects only the configured platform;
- accepts a per-run platform override for files from another platform;
- parses `timestamp_utc` as UTC;
- sorts by time;
- removes duplicate timestamps defensively;
- interpolates latitude/longitude onto acoustic ping times;
- does not extrapolate outside the available navigation interval;
- reports the fraction of acoustic pings with valid navigation.

The navigation code deliberately does not call `EchoData.update_platform()`. This avoids coupling the project to the time-alignment path implicated by Echopype issue #1493.

## Development principle

The stable internal interfaces should be:

```text
RAW -> EchoData -> xarray datasets -> target dataset -> CSV
```

A target dataset should be the main scientific output. CSV is only an export format.
