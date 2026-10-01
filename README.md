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

At this stage, the project implements the repository structure, RAW-file inspection, platform-filtered SQLite navigation, and an xarray-compatible surface detector. The FM calibration, bottom detector, FM angle calculation, target detector, target TS calculation, and final CSV writer have explicit adapter interfaces but are intentionally not guessed or reimplemented until they are tested against real FM data.

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
