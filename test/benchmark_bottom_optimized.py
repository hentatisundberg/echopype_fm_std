"""Compare the production and optimized Blackwell detectors on RAW files."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import echopype as ep
import numpy as np

from echopype_fm_std.bottom import blackwell_optimized, detect_bottom
from echopype_fm_std.config import calibration_kwargs, load_config
from echopype_fm_std.io import open_raw


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pattern", default="*.raw")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--report", type=Path, default=Path("bottom_optimized_comparison.json"))
    return parser.parse_args()


def _run(raw_path: Path, config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    input_cfg = config["input"]
    bottom_cfg = config["bottom"]
    ed = open_raw(raw_path, config)
    try:
        started = time.perf_counter()
        sv = ep.calibrate.compute_Sv(
            ed,
            waveform_mode=input_cfg["waveform_mode"],
            encode_mode=input_cfg["encoding_mode"],
            **calibration_kwargs(config),
        )
        calibration_seconds = time.perf_counter() - started
        started = time.perf_counter()
        bottom_dataset = ep.consolidate.add_splitbeam_angle(
            sv,
            ed,
            waveform_mode=input_cfg["waveform_mode"],
            encode_mode=input_cfg["encoding_mode"],
            pulse_compression=True,
            to_disk=False,
        )
        input_seconds = time.perf_counter() - started

        baseline_timings: dict[str, float] = {}
        started = time.perf_counter()
        baseline = detect_bottom(
            bottom_dataset,
            method=bottom_cfg.get("method", "blackwell"),
            params=bottom_cfg.get("params", {}),
            timings=baseline_timings,
        )
        baseline_seconds = time.perf_counter() - started

        optimized_timings: dict[str, float] = {}
        started = time.perf_counter()
        optimized = blackwell_optimized(
            bottom_dataset,
            params=bottom_cfg.get("params", {}),
            timings=optimized_timings,
        )
        optimized_seconds = time.perf_counter() - started

        baseline_values = baseline.to_numpy()
        optimized_values = optimized.to_numpy()
        finite_both = np.isfinite(baseline_values) & np.isfinite(optimized_values)
        exact_equal = np.array_equal(baseline_values, optimized_values, equal_nan=True)
        max_abs_difference = (
            float(np.max(np.abs(baseline_values[finite_both] - optimized_values[finite_both])))
            if finite_both.any()
            else 0.0
        )
        changed_count = int(
            np.count_nonzero(
                ~np.isclose(
                    baseline_values,
                    optimized_values,
                    equal_nan=True,
                    rtol=0.0,
                    atol=1e-6,
                )
            )
        )
        return {
            "file": str(raw_path),
            "status": "ok",
            "calibration_seconds": calibration_seconds,
            "bottom_input_preparation_seconds": input_seconds,
            "baseline_seconds": baseline_seconds,
            "optimized_seconds": optimized_seconds,
            "speedup": baseline_seconds / optimized_seconds,
            "baseline_timings": baseline_timings,
            "optimized_timings": optimized_timings,
            "baseline_finite_count": int(np.isfinite(baseline_values).sum()),
            "optimized_finite_count": int(np.isfinite(optimized_values).sum()),
            "exact_equal": exact_equal,
            "changed_count_at_1e-6_m": changed_count,
            "max_abs_difference_m": max_abs_difference,
        }
    except Exception as exc:
        return {
            "file": str(raw_path),
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        ed.cleanup_swap_files()


def _write_report(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")
    fields = [
        "file",
        "status",
        "calibration_seconds",
        "bottom_input_preparation_seconds",
        "baseline_seconds",
        "optimized_seconds",
        "speedup",
        "baseline_finite_count",
        "optimized_finite_count",
        "exact_equal",
        "changed_count_at_1e-6_m",
        "max_abs_difference_m",
    ]
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: record.get(field) for field in fields} for record in records)


def main() -> int:
    args = _args()
    raw_files = sorted(args.raw_dir.glob(args.pattern))
    if args.limit:
        raw_files = raw_files[: args.limit]
    if not raw_files:
        raise SystemExit(f"No RAW files matched {args.pattern!r} in {args.raw_dir}")
    records = []
    for raw_path in raw_files:
        record = _run(raw_path, args.config)
        records.append(record)
        if record["status"] == "ok":
            print(
                f"{Path(record['file']).name}: "
                f"{record['baseline_seconds']:.2f}s -> "
                f"{record['optimized_seconds']:.2f}s "
                f"({record['speedup']:.2f}x), "
                f"changed={record['changed_count_at_1e-6_m']}, "
                f"max_diff={record['max_abs_difference_m']:.6f}m"
            )
        else:
            print(f"{record['file']}: {record['error']}")
    _write_report(args.report, records)
    successful = [record for record in records if record["status"] == "ok"]
    if successful:
        baseline_total = sum(record["baseline_seconds"] for record in successful)
        optimized_total = sum(record["optimized_seconds"] for record in successful)
        print(
            f"\nTotal bottom time: {baseline_total:.2f}s -> "
            f"{optimized_total:.2f}s ({baseline_total / optimized_total:.2f}x)"
        )
        print(
            "All outputs equal at 1e-6 m: "
            f"{all(record['changed_count_at_1e-6_m'] == 0 for record in successful)}"
        )
    print(f"JSON report: {args.report}")
    print(f"CSV report:  {args.report.with_suffix('.csv')}")
    return 0 if len(successful) == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
