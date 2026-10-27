"""Benchmark the production pipeline without reimplementing its stages.

Example:
    python test/benchmark_pipeline.py \
        --raw-dir /path/to/raw \
        --navigation-db data/sailbuoy_metadatabase.db \
        --config config/default.yaml \
        --platform SAILOR2 \
        --limit 5

The benchmark writes machine-readable JSON and CSV summaries. Output files
are placed in a temporary directory by default so repeated runs do not
overwrite production products.
"""

from __future__ import annotations

import argparse
import cProfile
import csv
import json
import pstats
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from echopype_fm_std.pipeline import run_pipeline


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--navigation-db", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--platform")
    parser.add_argument("--pattern", default="*.raw")
    parser.add_argument("--limit", type=int, default=0, help="Maximum files; 0 means all")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report", type=Path, default=Path("pipeline_benchmark.json"))
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-output", type=Path, default=Path("pipeline.prof"))
    return parser.parse_args()


def _files(args: argparse.Namespace) -> list[Path]:
    paths = sorted(args.raw_dir.glob(args.pattern))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"No RAW files matched {args.pattern!r} in {args.raw_dir}")
    return paths


def _run_one(
    raw_path: Path,
    args: argparse.Namespace,
    output_dir: Path,
    profile: cProfile.Profile | None,
) -> dict[str, Any]:
    timings: dict[str, float] = {}
    started = time.perf_counter()
    call = lambda: run_pipeline(  # noqa: E731 - keeps profiling wrapper identical
        raw_path=raw_path,
        navigation_db=args.navigation_db,
        config_path=args.config,
        platform=args.platform,
        output_path=output_dir,
        timings=timings,
    )
    try:
        if profile is None:
            result = call()
        else:
            profile.enable()
            try:
                result = call()
            finally:
                profile.disable()
        status = "ok"
        error = None
        stages_completed = result["stages_completed"]
    except Exception as exc:
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
        stages_completed = []
    return {
        "file": str(raw_path),
        "status": status,
        "error": error,
        "total_seconds": time.perf_counter() - started,
        "timings": timings,
        "stages_completed": stages_completed,
    }


def _write_report(report_path: Path, records: list[dict[str, Any]]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")
    csv_path = report_path.with_suffix(".csv")
    stage_names = sorted({stage for record in records for stage in record["timings"]})
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["file", "status", "total_seconds", *stage_names],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "file": record["file"],
                    "status": record["status"],
                    "total_seconds": f"{record['total_seconds']:.6f}",
                    **{
                        stage: f"{record['timings'].get(stage, 0.0):.6f}"
                        for stage in stage_names
                    },
                }
            )


def _print_summary(records: list[dict[str, Any]]) -> None:
    successful = [record for record in records if record["status"] == "ok"]
    print(
        f"Files: {len(records)} ({len(successful)} successful, "
        f"{len(records) - len(successful)} failed)"
    )
    for record in records:
        print(
            f"{record['status']:>7} {record['total_seconds']:9.2f}s "
            f"{Path(record['file']).name}"
        )
        if record["error"]:
            print(f"        {record['error']}")
    if not successful:
        return

    totals: dict[str, float] = defaultdict(float)
    for record in successful:
        for stage, seconds in record["timings"].items():
            totals[stage] += seconds
    total = sum(record["total_seconds"] for record in successful)
    print("\nSuccessful-file averages:")
    for stage, seconds in sorted(totals.items(), key=lambda item: item[1], reverse=True):
        print(f"  {stage:28} {seconds / len(successful):8.2f}s "
              f"({seconds / total * 100:5.1f}% of measured pipeline time)")
    print(f"  {'total':28} {total / len(successful):8.2f}s")


def main() -> int:
    args = _parse_args()
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be at least 1")
    raw_files = _files(args)
    report_records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="echopype-benchmark-") as temporary:
        output_dir = args.output_dir or Path(temporary)
        profile = cProfile.Profile() if args.profile else None
        for repetition in range(args.repetitions):
            for raw_path in raw_files:
                record = _run_one(raw_path, args, output_dir, profile)
                record["repetition"] = repetition + 1
                report_records.append(record)
        if profile is not None:
            args.profile_output.parent.mkdir(parents=True, exist_ok=True)
            profile.dump_stats(str(args.profile_output))
            stats_path = args.profile_output.with_suffix(".txt")
            with stats_path.open("w", encoding="utf-8") as stream:
                pstats.Stats(profile, stream=stream).sort_stats("cumulative").print_stats(40)
    _write_report(args.report, report_records)
    _print_summary(report_records)
    print(f"\nJSON report: {args.report}")
    print(f"CSV report:  {args.report.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
