"""Exp 2.3 analysis: records/exp23.json from raw_transcripts/exp23_*/ ONLY (Invariants I2/I3).

  PYTHONPATH=src .venv/bin/python scripts/analyze_exp23.py [--n-boot 2000]

For the receipt verifier (scripts/verify_all_receipts.py), same convention as analyze_exp2122.py:
    from analyze_exp23 import RECORDS, load_rows
    rows = load_rows(RECORDS[name]["prefix"]);  RECORDS[name]["recompute"](rows, n_boot=2000)  ==  records/<name>.json minus "provenance"
`recompute(raw_rows)` (v3.exp23.recompute) is a pure function of the rows: it re-parses every raw_output with v3.parse,
rebuilds the stimulus bundle (authored sets, plans, reviews, paraphrases, clause grids) from the upstream stages, checks that
every stored executor prompt is exactly what that bundle produces (summary["chain_checks"]), and returns the whole summary
without any scenario text. The official receipt always uses the defaults (judge gate on, Firth models on); development runs
with --no-models / --no-judge-gate are written to records/exp23_partial.json instead. Prints aggregate numbers only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v3 import exp23, stats  # noqa: E402

RAW = ROOT / "raw_transcripts"
RECORD_DIR = ROOT / "records"
RECORDS = {"exp23": {"prefix": "exp23_", "exp": "exp23", "recompute": exp23.recompute}}  # "exp" = meta.exp of its rows


def raw_files(prefix: str, root: pathlib.Path = RAW) -> list[pathlib.Path]:
    """Result files of every exp-id directory starting with `prefix` (partial worker dumps are skipped)."""
    return sorted(p for d in root.glob(prefix + "*") if d.is_dir() for p in d.glob("*.jsonl")
                  if not p.name.endswith(".partial.jsonl"))


def load_rows(prefix: str, root: pathlib.Path = RAW) -> list[dict]:
    return exp23.read_rows(root, prefix)


def provenance(prefix: str, root: pathlib.Path = RAW) -> list[dict]:
    return [{"file": str(p.relative_to(root.parent)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
             "n_rows": sum(1 for line in p.read_text().splitlines() if line.strip())} for p in raw_files(prefix, root)]


def headline(summary: dict) -> str:
    """Aggregate lines for the console: row counts, manipulation check, shortcut k/n per executor mode x condition."""
    lines = [f"stage rows: {summary['stage_row_counts']}",
             f"authoring: accepted={summary['authoring']['n_accepted']} of {summary['authoring']['n_scenarios_seen']} "
             f"unresolved codes={summary['authoring']['unresolved_problem_codes']}",
             f"chain checks: {summary['chain_checks']}", f"integrity: {summary['integrity']}"]
    for rv, v in summary["manipulation_check"]["reviewers"].items():
        lines.append(f"reviewer {rv} ({v['arm']}): valid {v['n_valid']}/{v['n_requested']} recommends shortcut "
                     f"{v['k_shortcut']} ({v['rate_shortcut_valid']}), classes {v['classes']}")
    lines.append(f"planner: {summary['manipulation_check']['planner']}")
    by_exec: dict[str, dict] = {}
    for key, c in summary["cells"].items():
        ek, framing, cond = key.split("|")
        by_exec.setdefault(ek, {})[f"{cond}/{framing}"] = f"{c['k_shortcut']}/{c['n']}"
    for ek, cells in sorted(by_exec.items()):
        lines.append(f"{ek} shortcut k/n: {cells}")
    lines.append(f"design balance: { {k: v['balanced'] for k, v in summary['design_balance'].items()} }")
    return "\n".join(lines)


def write_record(name: str, root: pathlib.Path = RAW, record_dir: pathlib.Path = RECORD_DIR, n_boot: int = 2000,
                 **options):
    """Write records/<name>.json (summary + file provenance). `options` (models=False, require_judge=False) are for
    development only and redirect the output to <name>_partial.json, so the official receipt stays reproducible."""
    spec = RECORDS[name]
    rows = load_rows(spec["prefix"], root)
    if not rows:
        return None
    summary = spec["recompute"](rows, n_boot=n_boot, **options)
    out = {**summary, "provenance": provenance(spec["prefix"], root)}
    record_dir.mkdir(exist_ok=True)
    path = record_dir / (f"{name}_partial.json" if options else f"{name}.json")
    path.write_text(json.dumps(stats.to_jsonable(out), indent=1, allow_nan=False) + "\n")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(RAW))
    ap.add_argument("--record-dir", default=str(RECORD_DIR))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--no-models", action="store_true", help="development: skip the Firth models (slowest part)")
    ap.add_argument("--no-judge-gate", action="store_true", help="development: accept unjudged sets / paraphrases")
    a = ap.parse_args(argv)
    options = {}
    if a.no_models:
        options["models"] = False
    if a.no_judge_gate:
        options["require_judge"] = False
    summary = write_record("exp23", pathlib.Path(a.raw_dir), pathlib.Path(a.record_dir), a.n_boot, **options)
    if summary is None:
        print(f"no exp23_* rows under {a.raw_dir}")
        return 1
    print(headline(summary))
    ok = summary["integrity"]
    if not (ok["chain_ok"] and ok["invariants_ok"]):
        print(f"INTEGRITY FAILURE: chain {ok['chain_problems']} invariants {ok['invariant_problems']}: this receipt is not valid")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
