EK80 .raw
   │
   ▼
1. open_raw()
   │
   ├── EchoData
   │
   └── ping_time
          │
          ▼
2. SQLite navigation
   │
   └── ping-aligned latitude/longitude
          │
          ▼
3. FM preprocessing
   │
   └── pulse-compressed complex signal
          │
          ├─────────────────────┐
          ▼                     ▼
4. calibrated Sv          target-processing data
          │                     │
          ├── optimized         ├── split-beam angles
          │   Blackwell bottom  └── Sp
          └── surface
                                │
                                ▼
                         6. single-target detection
                                │
                                ▼
                         target locations
                                │
                                ▼
                         target TS calculation
                                │
                                ▼
                         7. CSV export

## Bottom detection

When `bottom.method` is `blackwell` (the default), the production pipeline
uses `blackwell_optimized()` from
[`src/echopype_fm_std/bottom.py`](../src/echopype_fm_std/bottom.py). This
implementation preserves the reference Blackwell smoothing, thresholds,
no-intercept handling, configured postprocessing, refinement, and offset,
but avoids repeated full-array scans when combining connected components and
when extracting the first bottom sample in the search range.

The optimized detector does not decimate pings or alter the Blackwell input
evidence. It was compared against the reference detector on four local RAW
files and produced identical bottom arrays within `1e-6 m`, while reducing
bottom-detector time from approximately `115.5 s` to `71.2 s` overall
(`1.62x`). The comparison can be reproduced with:

```bash
PYTHONPATH=src python test/benchmark_bottom_optimized.py \
  --raw-dir data/raw \
  --config config/default.yaml \
  --limit 4 \
  --report bottom_optimized_comparison.json
```

For an explicit non-Blackwell method, set `bottom.method` in the YAML
configuration. Such methods continue to use the general `detect_bottom()`
adapter.