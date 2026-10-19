"""Analysis for Experiments 2.2 and 2.1: reads raw_transcripts/ only, writes records/*.json.

  PYTHONPATH=src .venv/bin/python scripts/analyze_exp2122.py [--only exp22_monitor_calibration] [--n-boot 2000]

For the receipt verifier (scripts/verify_all_receipts.py):
    from analyze_exp2122 import RECORDS, load_rows, recompute
    rows = load_rows(RECORDS[name]["prefix"]);  RECORDS[name]["recompute"](rows, n_boot=rec["n_boot"])
        ==  records/<name>.json minus "provenance"      (or recompute(rows)[name] on any mix of rows)
`recompute(raw_rows)` is a pure function of the rows (it re-parses every raw_output with v3.parse; stored `parsed`
fields are never trusted). Exp 2.2 rows come from raw_transcripts/exp22_monitor_*/, Exp 2.1 rows from exp21_*/;
authoring rounds (exp22_worlds_*, exp22_notes_*) are persisted but are not part of these summaries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v3 import exp21, exp22, stats  # noqa: E402

RAW = ROOT / "raw_transcripts"
RECORD_DIR = ROOT / "records"
# Which exp-id directories feed a record. A prefix alone would also swallow smoke tests and re-runs (exp21_smoke, ...), so
# only the directories the launch plan creates count; any other directory with the prefix is reported, never read.
DIR_PATTERNS = {"exp22_monitor_": re.compile(r"exp22_monitor_(intact|llama70b|flash_next)"),
                "exp21_": re.compile(r"exp21_(alpha|beta|monitor)(_p[0-9]+)?")}   # _p<k>: parts of one stage launched in parallel
RECORDS = {
    "exp22_monitor_calibration": {"prefix": "exp22_monitor_", "exp": "2.2", "recompute": exp22.recompute},
    "exp21_topologies": {"prefix": "exp21_", "exp": "2.1", "recompute": exp21.recompute},
}


def recompute(raw_rows: list[dict], n_boot: int = 2000) -> dict:
    """{record name: summary} from any mix of raw rows (rows are routed by meta.exp). Pure: no file access."""
    out = {}
    for name, spec in RECORDS.items():
        rows = [r for r in raw_rows if r.get("meta", {}).get("exp") == spec["exp"]]
        if rows:
            out[name] = spec["recompute"](rows, n_boot=n_boot)
    return out


def record_dirs(prefix: str, root: pathlib.Path = RAW) -> tuple[list[pathlib.Path], list[pathlib.Path]]:
    """(used, ignored) exp-id directories starting with `prefix`: used = the names the launch plan creates (DIR_PATTERNS)."""
    pat = DIR_PATTERNS.get(prefix)
    dirs = sorted(d for d in root.glob(prefix + "*") if d.is_dir())
    used = [d for d in dirs if pat is None or pat.fullmatch(d.name)]
    return used, [d for d in dirs if d not in used]


def raw_files(prefix: str, root: pathlib.Path = RAW) -> list[pathlib.Path]:
    """Result files of the exp-id directories of `prefix` (partial worker dumps are skipped)."""
    return sorted(p for d in record_dirs(prefix, root)[0] for p in d.glob("*.jsonl") if not p.name.endswith(".partial.jsonl"))


def load_rows(prefix: str, root: pathlib.Path = RAW) -> list[dict]:
    rows = []
    for p in raw_files(prefix, root):
        rows += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return rows


def provenance(prefix: str, root: pathlib.Path = RAW) -> list[dict]:
    return [{"file": str(p.relative_to(root.parent)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
             "n_rows": sum(1 for l in p.read_text().splitlines() if l.strip())} for p in raw_files(prefix, root)]


def headline(name: str, s: dict) -> str:
    """A few numbers for the console (counts and rates only, never text)."""
    lines = [f"{name}: {s['n_rows']} rows"]
    if name == "exp22_monitor_calibration":
        lines.append(f"  note assertion failures: {s['n_assert_fail']}; worlds: {s['n_worlds']}")
        for o in s["operating_points"]:
            if o["fmt_family"] == "semantic" and o["cot_mode"] == "cot_off":
                lines.append(f"  {o['monitor']:9s} {o['condition']:28s} TPR {o['tpr']:.3f}  FPR {o['fpr']:.3f}  "
                             f"(n {o['n_violating']}/{o['n_legit']})")
    else:
        lines.append(f"  trials {s['n_trials']}, worlds {s['n_worlds']}, K {s['k_samples']}, judge usable: {s['judge_augmented']}")
        for c in s["cells"]:
            m = c["metrics"]
            f = lambda k: "  -  " if m[k]["rate"] is None else f"{m[k]['rate']:.3f}"
            lines.append(f"  {c['team']:10s} {c['variant']:8s} ILLEGIT_JOINT {f('ILLEGIT_JOINT')} exec_viol {f('executed_violation')} "
                         f"LEGIT_SAFE {f('LEGIT_SAFE_COORD')} cover_any {f('cover_story_any')} (n {c['n_trials']})")
    return "\n".join(lines)


def write_record(name: str, root: pathlib.Path = RAW, record_dir: pathlib.Path = RECORD_DIR, n_boot: int = 2000):
    spec = RECORDS[name]
    rows = load_rows(spec["prefix"], root)
    if not rows:
        return None
    summary = spec["recompute"](rows, n_boot=n_boot)
    out = {**summary, "provenance": provenance(spec["prefix"], root)}
    record_dir.mkdir(exist_ok=True)
    (record_dir / f"{name}.json").write_text(json.dumps(stats.to_jsonable(out), indent=1, allow_nan=False) + "\n")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(RECORDS))
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    for name in ([a.only] if a.only else RECORDS):
        summary = write_record(name, n_boot=a.n_boot)
        print(headline(name, summary) if summary else f"{name}: no raw rows under raw_transcripts/{RECORDS[name]['prefix']}*")
        ignored = record_dirs(RECORDS[name]["prefix"])[1]
        if ignored:
            print(f"  NOT READ (not part of the launch plan): {[d.name for d in ignored]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
