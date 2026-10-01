# Integration tests

Place a small representative EK80 FM `.raw` file and matching SQLite navigation database here locally when setting up the pipeline.

Do not commit proprietary/raw survey data to the repository.

The first integration target should be:

```text
RAW -> EchoData -> ping_time
RAW + SQLite(platform=X) -> aligned latitude/longitude
```

The next target should validate FM pulse compression and calibrated intermediate xarray datasets against a known reference.
