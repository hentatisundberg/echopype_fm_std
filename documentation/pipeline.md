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
          ├── bottom            ├── split-beam angles
          └── surface           ├── Sp
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