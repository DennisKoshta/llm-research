"""Exp 3.1 2x2 (layout: sentence | table) x (specificity: concrete description | abstract rating words)
-> records/exp31_2x2_deconfounder.json.

  PYTHONPATH=src .venv/bin/python scripts/analyze_exp31_2x2.py [--n-boot 5000]

Slice: high-cost harmful-implicit, thinking on, default effort, arms intact / refusal_ablated / pca_101; 32 scenarios x
2 costs x 2 tool orders = 128 pairs (pair_id, as Exp 3.1). Cells:
  concrete_sentence (CS) = Exp 3.1 narrative     abstract_table (AT)    = Exp 3.1 numeric
  concrete_table    (CT) = table + description   abstract_sentence (AS) = rating sentence, no description
Two designs: `same_launch` (all four cells from raw_transcripts/deconf_exp31, generated together) and `combined` (CS and AT
from the original Exp 3.1 run raw_transcripts/exp31, CT and AS from deconf_exp31), because greedy long-reasoning runs do not
reproduce across launches. Per arm and design, on per-pair shortcut indicators (ITT: unscorable = no shortcut):
  layout main effect        ((CT + AT) - (CS + AS)) / 2       simple effects CT - CS (concrete), AT - AS (abstract)
  abstraction main effect   ((AS + AT) - (CS + CT)) / 2       simple effects AS - CS (sentence), AT - CT (table)
  interaction               (AT - AS) - (CT - CS)             total AT - CS
and the arm x format difference-in-differences (refusal_ablated - other) of every effect, other in {intact, pca_101}.
Scenario cluster bootstrap 95% CIs and scenario sign-flip p (32 clusters); Holm within each (design, family) over the
three main tests per arm (layout, abstraction, interaction) and over the DiD tests. Prints numbers only.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v3 import deconf, stats  # noqa: E402

RECORD = ROOT / "records" / "exp31_2x2_deconfounder.json"
ORIG_FMT = {"narrative": "concrete_sentence", "numeric": "abstract_table"}
SHORT = {"concrete_sentence": "CS", "abstract_sentence": "AS", "concrete_table": "CT", "abstract_table": "AT"}
EFFECTS = {
    "layout_main": lambda c: ((c["CT"] + c["AT"]) - (c["CS"] + c["AS"])) / 2,
    "layout_concrete": lambda c: c["CT"] - c["CS"], "layout_abstract": lambda c: c["AT"] - c["AS"],
    "abstraction_main": lambda c: ((c["AS"] + c["AT"]) - (c["CS"] + c["CT"])) / 2,
    "abstraction_sentence": lambda c: c["AS"] - c["CS"], "abstraction_table": lambda c: c["AT"] - c["CT"],
    "interaction": lambda c: (c["AT"] - c["AS"]) - (c["CT"] - c["CS"]), "total": lambda c: c["AT"] - c["CS"]}
MAIN_TESTS = ("layout_main", "abstraction_main", "interaction")


def load_raw(raw_root) -> list[dict]:
    raw_root = pathlib.Path(raw_root)
    rows = []
    for d in ("exp31", deconf.EXP31_ID):
        for arm in deconf.ARMS:
            f = raw_root / d / f"{arm}.jsonl"
            if f.exists():
                rows += [{**json.loads(l), "_dir": d} for l in f.read_text().splitlines() if l.strip()]
    return rows


def trials(raw_rows) -> list[dict]:
    out = []
    for r in raw_rows:
        if (r["cot_mode"] != "cot_on" or r["cost_regime"] not in deconf.HIGH_COSTS or r["consequence"] != "harmful"
                or r["boundary"] != "implicit"):
            continue
        if r["_dir"] == "exp31":
            fmt, source = ORIG_FMT[r["meta"]["stake_format"]], "original"
        else:
            fmt, source = r["meta"]["format_2x2"], "same_launch"
        t = deconf.trial(r, fmt, "default")
        out.append({**t, "source": source})
    return out


def _cells(ts, design: str) -> dict:
    """{arm: {pair: {CS, AS, CT, AT}}} for one design."""
    pick = {"same_launch": {f: "same_launch" for f in SHORT},
            "combined": {"concrete_sentence": "original", "abstract_table": "original",
                         "concrete_table": "same_launch", "abstract_sentence": "same_launch"}}[design]
    out: dict = {}
    for t in ts:
        if pick[t["condition"]] == t["source"]:
            slot = out.setdefault(t["arm"], {}).setdefault(t["pair"], {})
            if SHORT[t["condition"]] in slot:
                raise ValueError("duplicate pair")
            slot[SHORT[t["condition"]]] = (t["direct"], t["scenario_id"])
    return out


def _effects(cells: dict, n_boot: int) -> dict:
    keys = sorted(k for k, v in cells.items() if len(v) == 4)
    vals = {k: {f: cells[k][f][0] for f in SHORT.values()} for k in keys}
    clus = [cells[k]["CS"][1] for k in keys]
    out = {"n_complete_pairs": len(keys), "rates": {f: sum(v[f] for v in vals.values()) / len(vals) for f in SHORT.values()},
           "counts": {f: sum(v[f] for v in vals.values()) for f in SHORT.values()}}
    for name, fn in EFFECTS.items():
        out[name] = deconf.cluster_mean([fn(vals[k]) for k in keys], clus, n_boot)
    return out, vals, clus


def recompute(raw_rows, n_boot: int = 5000) -> dict:
    ts = trials(raw_rows)
    rates = {}
    for t in ts:
        rates.setdefault(f"{t['arm']}|{t['condition']}|{t['source']}", []).append(t)
    out = {"schema": "exp31_2x2_v1", "params": {"n_boot": n_boot, "seed": stats.DEFAULT_SEED},
           "splits": {k: deconf.split(v) for k, v in sorted(rates.items())}, "designs": {}}
    for design in ("same_launch", "combined"):
        cells = _cells(ts, design)
        per_arm, vals = {}, {}
        for arm in deconf.ARMS:
            if arm in cells:
                per_arm[arm], vals[arm], _ = _effects(cells[arm], n_boot)
        holm = stats.holm({f"{a}|{e}": per_arm[a][e]["p_signflip"] for a in per_arm for e in MAIN_TESTS})
        for k, p in holm.items():
            a, e = k.split("|")
            per_arm[a][e]["p_signflip_holm"] = p
        did = {}
        if "refusal_ablated" in vals:
            for other in ("intact", "pca_101"):
                if other not in vals:
                    continue
                keys = sorted(vals["refusal_ablated"].keys() & vals[other].keys())
                clus = [cells["refusal_ablated"][k]["CS"][1] for k in keys]
                for name, fn in EFFECTS.items():
                    did[f"refusal_ablated_vs_{other}|{name}"] = deconf.cluster_mean(
                        [fn(vals["refusal_ablated"][k]) - fn(vals[other][k]) for k in keys], clus, n_boot)
                for f in SHORT.values():   # arm gap within each cell (ablated - other)
                    did[f"refusal_ablated_vs_{other}|gap_{f}"] = deconf.cluster_mean(
                        [vals["refusal_ablated"][k][f] - vals[other][k][f] for k in keys], clus, n_boot)
            fam = [k for k in did if k.split("|")[1] in ("layout_main", "abstraction_main", "interaction", "total")]
            for k, p in stats.holm({k: did[k]["p_signflip"] for k in fam}).items():
                did[k]["p_signflip_holm"] = p
        out["designs"][design] = {"per_arm": per_arm, "arm_x_format": did}
    out["replicate_vs_original"] = {}
    for arm in deconf.ARMS:
        for fmt in ("concrete_sentence", "abstract_table"):
            a = [t for t in ts if t["arm"] == arm and t["condition"] == fmt and t["source"] == "same_launch"]
            b = [t for t in ts if t["arm"] == arm and t["condition"] == fmt and t["source"] == "original"]
            if a and b:
                out["replicate_vs_original"][f"{arm}|{fmt}"] = {"shortcut": deconf.binary_contrast(a, b, "direct", n_boot),
                                                               "agreement": deconf.agreement(a, b, "direct")}
    return stats.to_jsonable(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=5000)
    a = ap.parse_args(argv)
    out = recompute(load_raw(ROOT / "raw_transcripts"), n_boot=a.n_boot)
    RECORD.write_text(json.dumps(out, indent=1, sort_keys=True))
    for k, v in out["splits"].items():
        print(f"[split] {k}: n={v['n']} shortcut={v['shortcut']} safe={v['safe_route_completed']} esc={v['escalated']} unscorable={v['unscorable']}")
    for design, d in out["designs"].items():
        for arm, e in d["per_arm"].items():
            print(f"[{design}] {arm}: rates={ {k: round(x, 3) for k, x in e['rates'].items()} } " + " ".join(
                f"{n}={e[n]['estimate']:.3f}[{e[n]['ci_low']:.3f},{e[n]['ci_high']:.3f}]p={e[n]['p_signflip']:.4f}" for n in MAIN_TESTS))
    print(f"[written] {RECORD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
