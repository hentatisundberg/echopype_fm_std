# echopype_fm_std

A reproducible EK80 FM-mode processing pipeline built around [Echopype](https://github.com/echostack-org/echopype).

The full pipeline is:

1. read one EK80 `.raw` file;
2. align geographic position from a SQLite tracking database, filtered by `platform`;
3. calibrate the acoustic data to `Sv`;
4. calculate the bottom boundary, including the configured Blackwell
   conditioning and refinement;
5. detect the surface/turbidity boundary;
6. mask the water column to the surface-to-bottom envelope;
7. calculate distance/depth-binned NASC echo integration;
8. export the boundary summary and mask echograms;
9. prepare pulse-compressed FM split-beam data and, when enabled, detect
   single targets, calculate target strength, and export target rows to CSV.

Echo integration and single-target detection are controlled independently in
`config/default.yaml`. Integration is enabled by default; single-target
detection is also enabled by default in the supplied configuration.

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

Copy and edit `config/default.yaml`. The configuration controls the input
beam group and FM encoding, calibration, navigation interpolation, surface
and bottom detection, echo integration, single-target filtering, and output
directories. The `navigation.platform` value is required because the SQLite
database contains positions for multiple platforms; use the per-run
`--platform` option when a file belongs to another platform.

The supplied configuration uses:

- `bottom.method: blackwell`, with the configured preprocessing,
  postprocessing, and refinement settings;
- surface/turbidity detection from calibrated `Sv`;
- `echo_integration.enabled: true`, with 10 m depth bins and 0.1 nmi
  distance bins;
- `single_target.enabled: true`, with a default compensated target-strength
  range of `-60 dB` to `-10 dB`;
- `input.transducer_depth_m` to convert acoustic range to depth below the
  sea surface for integration and target products.

Set `echo_integration.enabled: false` to skip integration, or
`single_target.enabled: false` to run the boundary, mask, and integration
products without target detection.

See which platform values are present in the database:

```bash
echopype-fm nav-platforms --navigation-db /path/to/positions.sqlite
```

Run one RAW file with its navigation database and platform:

```bash
echopype-fm run \
  --raw data/raw/SLUAquaSailor2020V1-Phase0-D20260423-T070050-0.raw \
  --navigation-db data/positions/sailbuoy_metadatabase.db \
  --platform SAILOR2 \
  --config config/default.yaml
```

Echo integration is enabled by default and writes
`output/integration/<raw-stem>_echo_integration.csv`. It uses Echopype's
`compute_NASC()` with fixed-depth bins below the sea surface and horizontal
distance bins. The depth and distance bin sizes can be changed with
`echo_integration.layer_size_m` and `echo_integration.distance_bin_nmi`
(default `0.1`). Each CSV row represents one distance/depth cell and includes
the cell NASC, variance, contributing ping count, sample count, binned ping
time and position, plus survey start/end time. Empty cells, including cells
below the detected bottom with no valid samples, are omitted from the CSV.

To inspect a local RAW file before processing it:

```bash
echopype-fm inspect \
  --raw data/raw/SLUAquaSailor2020V1-Phase0-D20260423-T070050-0.raw \
  --config config/default.yaml
```

To process every RAW file in a folder, continue past files that fail while
printing a summary at the end:

```bash
echopype-fm run-folder \
  --raw-dir data/raw \
  --navigation-db data/positions/sailbuoy_metadatabase.db \
  --platform SAILOR2 \
  --config config/default.yaml \
  --continue-on-error
```

Use `--pattern '*.raw'` to select a different filename pattern. By default
the command stops on the first failure; add `--continue-on-error` to process
the remaining files and print failures at the end. Navigation for each
database/platform pair is cached in memory for the duration of the process,
while each RAW file is still time-windowed and interpolated independently.

For a quick random-file smoke test, choose any `.raw` file in `data/raw/`,
use `echopype-fm inspect` first, and confirm that its ping-time range overlaps
the selected platform's navigation records. The full run requires that
overlap, a valid surface detection, and a valid bottom detection.

Single-target detection uses the same surface/seafloor exclusion mask as
integration. It writes
`output/single_targets/<raw-stem>_single_targets.csv`. The default compensated
target-strength cutoff is `-60 dB`; override
`single_target.params.TS_comp_min` in the YAML configuration or set
`single_target.enabled: false` when testing only integration. Target rows
include ping-specific position and repeated survey start/end metadata when
the target export is produced.

Each run also writes
`output/masks/<raw-stem>_mask_summary.csv`, containing one row with maximum,
mean, and variance of the surface-turbidity and bottom depths, plus survey
start/end time and position. It writes
`output/images/<raw-stem>_mask_echogram.png`: a calibrated `Sv`
echogram with the surface and bottom lines and a semi-transparent grey
overlay showing samples excluded by the mask. Detected targets are plotted as
white crosses. A separate `output/images/<raw-stem>_mask_clean.png` contains
only the raster image and mask overlay, without axes, titles, legends, or
colorbars. Its pixel scale is based on elapsed time and physical depth, so
longer surveys produce wider images rather than horizontally stretching the
same fixed-size image.

CSV products are separated into `output/masks`, `output/single_targets`, and
`output/integration`. These directory names can be changed under `output` in
the YAML configuration.

For the default output settings, a successful run produces:

```text
output/
├── images/
│   ├── <raw-stem>_mask_echogram.png
│   └── <raw-stem>_mask_clean.png
├── masks/
│   └── <raw-stem>_mask_summary.csv
├── single_targets/
│   └── <raw-stem>_single_targets.csv
└── integration/
    └── <raw-stem>_echo_integration.csv
```

The single-target CSV is produced only when single-target detection is
enabled. The integration CSV is produced only when echo integration is
enabled.

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

The FM calibration, pulse-compressed split-beam angle calculation, target
detector, target TS calculation, and CSV writers are implemented through
project adapters around the pinned Echopype APIs. The experimental
single-target detector remains isolated behind those adapters so the rest of
the pipeline does not depend directly on unstable upstream interfaces.

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

The pipeline attaches the aligned coordinates to `EchoData["Platform"]` using
Echopype's `update_platform()` API. The aligned navigation dataset is also
retained as a project-level diagnostic view and is used for integration and
target-level position export.

## Development principle

The stable internal interfaces should be:

```text
RAW -> EchoData -> navigation/calibrated xarray datasets
     -> boundaries and masks
     -> NASC integration and/or target dataset
     -> CSV and image products
```

The NASC dataset, boundary products, and target dataset are the scientific
outputs. CSV and PNG files are export and diagnostic formats.
