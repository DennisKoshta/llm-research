"""Cumulative project compute bound from Modal app wall time, not invoice spend."""

import argparse
import importlib.util
import json
from pathlib import Path
from curiosity_study.artifacts import write_once

spec = importlib.util.spec_from_file_location(
    "pilot_budget", Path(__file__).resolve().parents[1] / "infra/budget.py"
)
budget = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budget)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    result = budget.snapshot()
    if args.out:
        write_once(args.out, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("apps", "app_charges")}, indent=2))
