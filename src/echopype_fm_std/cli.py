from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .io import inspect_raw
from .navigation import list_platforms
from .pipeline import run_pipeline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="echopype-fm")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="Inspect one EK80 RAW file")
    inspect.add_argument("--raw", required=True)
    inspect.add_argument("--config", default="config/default.yaml")

    run = sub.add_parser("run", help="Run the processing pipeline")
    run.add_argument("--raw", required=True)
    run.add_argument("--navigation-db", required=True)
    run.add_argument("--config", default="config/default.yaml")
    run.add_argument(
        "--platform",
        help="Navigation platform for this RAW file; overrides navigation.platform",
    )

    platforms = sub.add_parser("nav-platforms", help="List platform names in a navigation database")
    platforms.add_argument("--navigation-db", required=True)

    return parser


def main() -> None:
    args = _parser().parse_args()

    if args.command == "inspect":
        cfg = load_config(args.config)
        summary = inspect_raw(args.raw, cfg)
        print(json.dumps(summary, indent=2, default=str))
        return

    if args.command == "run":
        result = run_pipeline(
            raw_path=Path(args.raw),
            navigation_db=Path(args.navigation_db),
            config_path=Path(args.config),
            platform=args.platform,
        )
        print("Completed:", ", ".join(result["stages_completed"]))
        if result.get("navigation") is not None:
            nav = result["navigation"]
            valid_pct = float(nav["navigation_valid"].mean()) * 100.0
            print(f"Navigation coverage: {valid_pct:.2f}% of acoustic pings")
        if result.get("stages_pending"):
            print("Pending:", ", ".join(result["stages_pending"]))
        return

    if args.command == "nav-platforms":
        for platform in list_platforms(args.navigation_db):
            print(platform)
        return


if __name__ == "__main__":
    main()
