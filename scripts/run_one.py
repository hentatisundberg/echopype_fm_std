from __future__ import annotations

import argparse

from echopype_fm_std.pipeline import run_pipeline


parser = argparse.ArgumentParser()
parser.add_argument("--raw", required=True)
parser.add_argument("--navigation-db", required=True)
parser.add_argument("--config", default="config/default.yaml")
args = parser.parse_args()

result = run_pipeline(args.raw, args.navigation_db, args.config)
print("Completed:", result["stages_completed"])
print("Pending:", result.get("stages_pending", []))
