"""Build the Exp 3.1 request file (multi-domain x narrative/numeric stakes x cot) from the authored bases.

  PYTHONPATH=src .venv/bin/python scripts/build_exp31_requests.py --scale 1
  PYTHONPATH=src .venv/bin/python scripts/build_exp31_requests.py --placeholder --dry-run     # no authored data needed

Inputs (on disk only, never printed): data/author/final_bases.jsonl (validated authored bases) and
data/author/judge_verdicts.json (the on-Modal severity judge, source of the numeric table's severity column).
Outputs: data/exp31/requests_exp31.jsonl (episode rows, `arm` omitted: every arm runs every row) and
data/exp31/manifest.json (selected scenario ids, counts, tool-order balance, cost estimate; no prompt text).
This script prints counts only.

It also prints the exact run_requests.py launch line (the ledger reservation happens inside run_requests.py; --family is
omitted so the recorded Phase 1 decision is used) and the analysis command.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v3 import exp31, scenarios  # noqa: E402


def load_inputs(a) -> tuple[list, dict]:
    if a.placeholder:
        bases = exp31.placeholder_bases(40)
        return bases, exp31.placeholder_verdicts(bases)
    bases = scenarios.load_bases(a.bases)
    verdicts = json.loads(pathlib.Path(a.verdicts).read_text())
    return bases, verdicts


def build(a) -> dict:
    design = exp31.scale_design(a.scale)
    per_domain = a.bases_per_domain or design["bases_per_domain"]
    orders = a.orders or design["orders"]
    bases, verdicts = load_inputs(a)
    formats = tuple(a.formats.split(","))
    instances = exp31.build_exp31_instances(bases, verdicts, per_domain=per_domain, cells=a.cells, orders=orders, formats=formats)
    rows = exp31.build_requests(instances, effort=a.effort, caps={"cot_on": a.cap_on, "cot_off": a.cap_off})
    caps = exp31.default_caps(a.effort) | {k: v for k, v in (("cot_on", a.cap_on), ("cot_off", a.cap_off)) if v is not None}
    arms = a.arms.split(",")
    cost = exp31.estimate_cost(a.scale, n_arms=len(arms), cells=a.cells, bases_per_domain=per_domain, orders=orders, n_formats=len(formats))
    selected = sorted({i["scenario_id"] for i in instances})
    manifest = {
        "exp_id": exp31.EXP_ID, "scale": a.scale, "bases_per_domain": per_domain, "orders": orders, "cells": a.cells,
        "formats": list(formats), "effort": a.effort_label, "caps": caps, "arms": arms, "placeholder": bool(a.placeholder),
        "n_instances": len(instances), "n_requests": len(rows), "n_bases": len(selected), "scenario_ids": selected,
        "by_domain": dict(collections.Counter(b["domain"] for b in bases if b["scenario_id"] in selected)),
        "by_template": dict(collections.Counter(b["template_id"] for b in bases if b["scenario_id"] in selected)),
        "tool_order_balance_per_cell": exp31.order_balance(instances),
        "requests_per_group": dict(collections.Counter(f"{r['cot_mode']}/{r['max_new_tokens']}" for r in rows)),
        "cost_estimate": cost,
    }
    return {"rows": rows, "manifest": manifest}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bases", default=str(ROOT / "data/author/final_bases.jsonl"))
    ap.add_argument("--verdicts", default=str(ROOT / "data/author/judge_verdicts.json"))
    ap.add_argument("--scale", type=float, default=1.0, help="breadth: bases per domain = round(6*scale) (<= 8), orders 1/2/3")
    ap.add_argument("--bases-per-domain", type=int, default=None, help="override scale's bases per domain")
    ap.add_argument("--orders", type=int, default=None, help="override scale's tool-order replicates per (base, cell)")
    ap.add_argument("--cells", default="all12", choices=sorted(exp31.CELL_SETS))
    ap.add_argument("--formats", default=",".join(exp31.FORMATS),
                    help="stake formats to run; 'numeric' alone reuses the Exp 1.1 sweep as the narrative arm (no paired contrast then)")
    ap.add_argument("--arms", default=",".join(exp31.DEFAULT_ARMS), help="only used for the cost estimate and the manifest")
    ap.add_argument("--effort", default=None, choices=["low", "medium", "xhigh", "default"],
                    help="reasoning_effort of cot_on (recorded per row); default = the recorded Phase 1 decision, which must equal "
                         "the Exp 1.1 sweep's for Exp 3.2 to line up ('default' = template default, field None)")
    ap.add_argument("--cap-on", type=int, default=None, help="max_new_tokens of cot_on; default = Exp 1.1's (exp31.default_caps)")
    ap.add_argument("--cap-off", type=int, default=None, help="max_new_tokens of cot_off; default = Exp 1.1's (exp31.default_caps)")
    ap.add_argument("--out", default=str(ROOT / "data/exp31/requests_exp31.jsonl"))
    ap.add_argument("--placeholder", action="store_true", help="use placeholder bases (no authored text needed)")
    ap.add_argument("--dry-run", action="store_true", help="build and print counts; write nothing")
    a = ap.parse_args(argv)
    a.effort_label = a.effort or exp31.default_effort(ROOT)
    a.effort = None if a.effort_label == "default" else a.effort_label

    built = build(a)
    m = built["manifest"]
    c = m["cost_estimate"]
    print(f"requests={m['n_requests']} instances={m['n_instances']} bases={m['n_bases']} by_domain={m['by_domain']} "
          f"cells={m['cells']} orders={m['orders']} effort={m['effort']} caps={m['caps']}")
    print(f"estimate ({len(m['arms'])} arms): generations={c['generations']} tokens={c['generated_tokens']} "
          f"gpu_hours={c['gpu_hours']} usd={c['usd']}")
    req = pathlib.Path(a.out).relative_to(ROOT) if pathlib.Path(a.out).is_relative_to(ROOT) else a.out
    print(f"[launch] PYTHONPATH=src .venv/bin/python scripts/run_requests.py --exp-id exp31 --requests {req} --arms {','.join(m['arms'])} "
          f"--gpu H200 --max-num-seqs 384 --eager 0 --est-hours {c['worst_case_hours_per_arm']}")
    print("[then]   PYTHONPATH=src .venv/bin/python scripts/analyze_exp3.py exp31")
    if a.dry_run:
        return 0
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in built["rows"]) + "\n")
    m["requests_file"] = str(out.relative_to(ROOT)) if out.is_relative_to(ROOT) else str(out)
    m["requests_sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
    (out.parent / "manifest.json").write_text(json.dumps(m, indent=1, sort_keys=True))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) and {out.parent / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
