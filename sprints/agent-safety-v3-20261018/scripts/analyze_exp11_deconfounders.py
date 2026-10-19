"""Exp 1.1 deconfounders (no hand-off tool; no call-count preference sentence) -> records/exp11_deconfounders.json.

  PYTHONPATH=src .venv/bin/python scripts/analyze_exp11_deconfounders.py [--n-boot 5000]

Inputs: raw_transcripts/deconf_exp11/<arm>.jsonl (conditions std / nh / np, see v3.deconf) and, as the original standard
condition, the Exp 1.1 high-cost harmful-implicit episodes of the same arms (raw_transcripts/exp11 at the default effort,
raw_transcripts/exp11_effort_low_* at effort low). Everything is replayed from raw strings (v3.deconf.trial). Pairs are
scenario x cost regime (80 per arm, condition and effort); contrasts report exact McNemar, Newcombe CI, template sign-flip
and scenario cluster bootstrap. Holm: the pre-declared family {nh - std, np - std} per arm and effort on the shortcut
outcome against the same-launch standard (9 tests). Prints numbers only.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v3 import deconf, stats  # noqa: E402

RECORD = ROOT / "records" / "exp11_deconfounders.json"
ORIGINAL = {"default": ("exp11",), "low": ("exp11_effort_low_a", "exp11_effort_low_b", "exp11_effort_low_c")}


def load_raw(raw_root) -> list[dict]:
    raw_root = pathlib.Path(raw_root)
    rows = []
    for d in (deconf.EXP11_ID,) + ORIGINAL["default"] + ORIGINAL["low"]:
        for arm in deconf.ARMS:
            f = raw_root / d / f"{arm}.jsonl"
            if f.exists():
                rows += [{**json.loads(l), "_dir": d} for l in f.read_text().splitlines() if l.strip()]
    return rows


def trials(raw_rows) -> list[dict]:
    out = []
    for r in raw_rows:
        effort = "low" if r.get("reasoning_effort") == "low" else "default"
        if r["_dir"] == deconf.EXP11_ID:
            cond = r["meta"]["deconf"]
            if cond == "demo":
                continue
        else:
            if (r.get("family") != "main" or r["cot_mode"] != "cot_on" or r["cost_regime"] not in deconf.HIGH_COSTS
                    or r["consequence"] != "harmful" or r["boundary"] != "implicit"):
                continue
            cond = "std_original"
        out.append(deconf.trial(r, cond, effort))
    return out


def recompute(raw_rows, n_boot: int = 5000) -> dict:
    ts = trials(raw_rows)
    groups: dict = {}
    for t in ts:
        groups.setdefault((t["arm"], t["condition"], t["effort"]), []).append(t)
    for key, g in groups.items():
        if len({t["pair"] for t in g}) != len(g):
            raise ValueError(f"duplicate pair in {key}")
    cells = {f"{a}|{c}|{e}": deconf.split(g) for (a, c, e), g in sorted(groups.items())}
    for (a, c, e), g in sorted(groups.items()):
        cells[f"{a}|{c}|{e}"]["undeclared_escalate_attempts"] = sum(t["undeclared_escalate"] for t in g)
        st = [t["shortcut_turn"] for t in g if t["shortcut_turn"]]
        cells[f"{a}|{c}|{e}"]["shortcut_turn1"] = sum(x == 1 for x in st)
        cells[f"{a}|{c}|{e}"]["shortcut_turn2plus"] = sum(x > 1 for x in st)
    within, family = {}, {}
    for arm in deconf.ARMS:
        for effort, conds in deconf.EXP11_CONDITIONS.items():
            g = lambda c: groups.get((arm, c, effort), [])
            comps = [("std", "std_original")] + [(c, ref) for c in conds if c != "std" for ref in ("std", "std_original")]
            for a, b in comps:
                if not g(a) or not g(b):
                    continue
                name = f"{arm}|{effort}|{a}_vs_{b}"
                within[name] = {"shortcut": deconf.binary_contrast(g(a), g(b), "direct", n_boot),
                                "safe_route": deconf.binary_contrast(g(a), g(b), "safe", n_boot),
                                "agreement_shortcut": deconf.agreement(g(a), g(b), "direct"),
                                "agreement_action": deconf.agreement(g(a), g(b), "action")}
                if b == "std" and a in ("nh", "np"):
                    family[name] = within[name]["shortcut"]
    for key, out_key in (("p_mcnemar", "p_holm"), ("p_signflip_template", "p_signflip_template_holm")):
        adj = stats.holm({k: v.get(key, float("nan")) for k, v in family.items()})
        for k, p in adj.items():
            family[k][out_key] = p
    between = {}
    for (arm, cond, effort) in sorted(groups):
        if arm != "refusal_ablated":
            continue
        for other in ("intact", "pca_101"):
            if (other, cond, effort) in groups:
                between[f"{cond}|{effort}|refusal_ablated_vs_{other}"] = deconf.binary_contrast(
                    groups[(arm, cond, effort)], groups[(other, cond, effort)], "direct", n_boot)
    return stats.to_jsonable({
        "schema": "exp11_deconfounders_v1", "params": {"n_boot": n_boot, "seed": stats.DEFAULT_SEED},
        "design": {"arms": list(deconf.ARMS), "conditions": {k: list(v) for k, v in deconf.EXP11_CONDITIONS.items()},
                   "original_sources": {k: list(v) for k, v in ORIGINAL.items()},
                   "holm_family": sorted(family), "pairs_per_cell": 80},
        "integrity": {"n_trials": len(ts), "n_groups": len(groups)},
        "cells": cells, "within_arm": within, "between_arms": between})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=5000)
    a = ap.parse_args(argv)
    out = recompute(load_raw(ROOT / "raw_transcripts"), n_boot=a.n_boot)
    RECORD.write_text(json.dumps(out, indent=1, sort_keys=True))
    for k, v in out["cells"].items():
        print(f"[cell] {k}: n={v['n']} safe={v['safe_route_completed']} shortcut={v['shortcut']} esc={v['escalated']} "
              f"prose={v['prose_stop']} unscorable={v['unscorable']} undeclared_esc={v['undeclared_escalate_attempts']}")
    for k, v in out["within_arm"].items():
        s = v["shortcut"]
        print(f"[within] {k}: {s.get('rate_a')} vs {s.get('rate_b')} p={s.get('p_mcnemar')} p_holm={s.get('p_holm', '-')} "
              f"kappa={v['agreement_shortcut'].get('kappa')}")
    print(f"[written] {RECORD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
