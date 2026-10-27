"""Benchmark the bottom-detection stages used by the production pipeline.

This deliberately performs the same Sv calibration and Blackwell input
preparation as ``run_pipeline()``, then reports timings internal to
``detect_bottom()``. It does not generate production output products.

Example:
    PYTHONPATH=src python test/benchmark_bottom.py \
        --raw-dir data/raw \
        --config config/default.yaml \
        --limit 4
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import echopype as ep

from echopype_fm_std.bottom import detect_bottom
from echopype_fm_std.config import calibration_kwargs, load_config
from echopype_fm_std.io import open_raw


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pattern", default="*.raw")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--report", type=Path, default=Path("bottom_benchmark.json"))
    return parser.parse_args()


def _run(raw_path: Path, config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    input_cfg = config["input"]
    bottom_cfg = config["bottom"]
    timings: dict[str, float] = {}
    started = time.perf_counter()
    ed = open_raw(raw_path, config)
    try:
        stage_start = time.perf_counter()
        sv = ep.calibrate.compute_Sv(
            ed,
            waveform_mode=input_cfg["waveform_mode"],
            encode_mode=input_cfg["encoding_mode"],
            **calibration_kwargs(config),
        )
        timings["calibration"] = time.perf_counter() - stage_start

        bottom_dataset = sv
        if bottom_cfg.get("method", "blackwell") == "blackwell":
            stage_start = time.perf_counter()
            bottom_dataset = ep.consolidate.add_splitbeam_angle(
                sv,
                ed,
                waveform_mode=input_cfg["waveform_mode"],
                encode_mode=input_cfg["encoding_mode"],
                pulse_compression=True,
                to_disk=False,
            )
            timings["bottom_input_preparation"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        detect_bottom(
            bottom_dataset,
            method=bottom_cfg.get("method", "blackwell"),
            params=bottom_cfg.get("params", {}),
            timings=timings,
        )
        timings["bottom_total"] = time.perf_counter() - stage_start
        status = "ok"
        error = None
    except Exception as exc:
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
    finally:
        ed.cleanup_swap_files()
    return {
        "file": str(raw_path),
        "status": status,
        "error": error,
        "timings": timings,
        "total_with_calibration": time.perf_counter() - started,
    }


def _write_report(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")
    fields = sorted({key for record in records for key in record["timings"]})
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["file", "status", *fields])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "file": record["file"],
                    "status": record["status"],
                    **{
                        key: f"{record['timings'].get(key, 0.0):.6f}"
                        for key in fields
                    },
                }
            )


def main() -> int:
    args = _args()
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be at least 1")
    raw_files = sorted(args.raw_dir.glob(args.pattern))
    if args.limit:
        raw_files = raw_files[: args.limit]
    if not raw_files:
        raise SystemExit(f"No RAW files matched {args.pattern!r} in {args.raw_dir}")

    records = []
    for repetition in range(args.repetitions):
        for raw_path in raw_files:
            record = _run(raw_path, args.config)
            record["repetition"] = repetition + 1
            records.append(record)
            print(
                f"{record['status']:>7} "
                f"{record['timings'].get('bottom_total', 0.0):8.2f}s "
                f"{raw_path.name}"
            )
    _write_report(args.report, records)

    successful = [record for record in records if record["status"] == "ok"]
    if successful:
        keys = sorted(
            {key for record in successful for key in record["timings"]},
            key=lambda key: sum(record["timings"].get(key, 0.0) for record in successful),
            reverse=True,
        )
        print("\nSuccessful-file averages:")
        for key in keys:
            average = sum(record["timings"].get(key, 0.0) for record in successful)
            average /= len(successful)
            print(f"  {key:28} {average:8.2f}s")
    print(f"\nJSON report: {args.report}")
    print(f"CSV report:  {args.report.with_suffix('.csv')}")
    return 0 if len(successful) == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
