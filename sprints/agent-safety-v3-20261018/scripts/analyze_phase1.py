"""Score a Phase 1 Direct-Instruction matrix from raw_transcripts and write records/<name>.json.

  PYTHONPATH=src .venv/bin/python scripts/analyze_phase1.py phase1_dim_r1 [phase1_dim_r2 ...]

Everything is recomputed from raw_transcripts/<exp>/*.jsonl (strict v3.parse re-parse). Prints only counts/rates.
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v3 import phase1  # noqa: E402

KEY_SETS = [("C", "C"), ("A", "A0"), ("A", "A1"), ("A", "A2"), ("B", "B")]


def load(exp_id: str) -> list[dict]:
    rows = []
    for f in sorted((ROOT / "raw_transcripts" / exp_id).glob("*.jsonl")):
        if f.name.endswith(".partial.jsonl"):
            continue
        rows += [json.loads(l) for l in open(f) if l.strip()]
    return rows


def main(exp_ids: list[str]) -> None:
    for exp in exp_ids:
        rows = load(exp)
        scored = phase1.score_rows(rows)
        cells = phase1.aggregate(scored)
        arms = sorted({r["arm"] for r in scored})
        efforts = sorted({r["effort"] for r in scored}, key=str)
        print(f"\n===== {exp}: {len(rows)} rows, arms={arms}, efforts={efforts}")
        by = {(c["arm"], c["set_name"], c["variant"], c["consequence"], c["effort"]): c for c in cells}
        for eff in efforts:
            print(f"\n-- effort={eff}: ITT COMPLY% (n_malformed) | harmful items; harmless twins in []")
            hdr = "arm".ljust(17) + "".join(f"{s[1]:>16}" for s in KEY_SETS)
            print(hdr)
            for arm in arms:
                line = arm.ljust(17)
                for sn, var in KEY_SETS:
                    h = by.get((arm, sn, var, "harmful", eff))
                    t = by.get((arm, sn, var, "harmless", eff))
                    cell = "-" if not h else f"{100*h['itt_comply_rate']:.0f}%({h['counts']['MALFORMED']})"
                    if t:
                        cell += f"[{100*t['itt_comply_rate']:.0f}%]"
                    line += f"{cell:>16}"
                print(line)
        gates = {}
        for eff in efforts:
            if "intact" in arms and "refusal_ablated" in arms:
                gates[str(eff)] = phase1.evaluate_gate(scored, effort=eff,
                                                       control_arms=tuple(a for a in phase1.DEFAULT_CONTROLS if a in arms))
                g = gates[str(eff)]
                print(f"\n-- gate (behavioral) effort={eff}: passed_behavioral={g['passed_behavioral']}")
                for c in g["criteria"]:
                    print("   ", {k: c[k] for k in ("name", "set", "arm", "value", "op", "threshold", "passed") if k in c})
        out = {"exp_id": exp, "n_rows": len(rows), "cells": cells, "gates": gates}
        (ROOT / "records").mkdir(exist_ok=True)
        (ROOT / "records" / f"{exp}_analysis.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1:])
