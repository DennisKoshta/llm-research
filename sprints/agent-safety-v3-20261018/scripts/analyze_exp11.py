"""Analyze Exp 1.1 from raw_transcripts only; write records/exp11_{main,probes,external}.json.

  PYTHONPATH=src .venv/bin/python scripts/analyze_exp11.py [--effort low|medium|xhigh|default|auto] [--no-model]

Reads raw_transcripts/exp11*/<arm>.jsonl (partial files ignored), recomputes every derived field with
v3.episodes.recompute_record and writes the JSON. Prints aggregate numbers only (no scenario or model text).
`recompute(raw_rows)` is the pure function scripts/verify_all_receipts.py calls: it returns {record name: summary}.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v3 import exp11  # noqa: E402

RECORDS = ("exp11_main", "exp11_probes", "exp11_external")
RAW_PREFIX = "exp11"


def load_raw(raw_root: pathlib.Path) -> list[dict]:
    """Every row of raw_transcripts/exp11*/<arm>.jsonl in a stable order (the .partial.jsonl crash copies are skipped)."""
    rows = []
    for d in sorted(p for p in raw_root.glob(RAW_PREFIX + "*") if p.is_dir()):
        for f in sorted(d.glob("*.jsonl")):
            if f.name.endswith(".partial.jsonl"):
                continue
            rows += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    return rows


def recompute(raw_rows, **kw) -> dict:
    return exp11.recompute(raw_rows, **kw)


def _pct(r) -> str:
    return "n/a" if not r or r.get("rate") is None else f"{100 * r['rate']:.1f}% ({r['k']}/{r['n']})"


def report(out: dict) -> None:
    main = out["exp11_main"]
    h = main["meta"]["hygiene"]
    print(f"[hygiene] rows={h['n_rows']} episodes={h['n_episode_rows']} singles={h['n_single_rows']} invalid={h['n_invalid_records']} "
          f"stored_mismatch={h['n_stored_mismatch_rows']} dup={h['n_duplicate_rows']} unplaceable={h['n_unplaceable']} "
          f"I1_violations={h['n_i1_violations']} banned_literal={h['n_banned_literal_rows']} prompt_suffix_violations={h['n_prompt_suffix_violations']} "
          f"other_effort={h['n_trials_other_effort']} effort={main['meta']['analysis_effort']}")
    for w in main["meta"]["warnings"]:
        print("[WARN]", w)
    for cot, c in main["construct_validity"].items():
        print(f"[checklist:{cot}] " + " | ".join(f"{k}: {'PASS' if v['pass'] else 'FAIL'} {_pct(v['value'])}"
                                                for k, v in c.items() if isinstance(v, dict)))
    for name, c in main["contrasts"].items():
        if isinstance(c, dict) and c.get("n_pairs"):
            print(f"[contrast] {name}: a={c['rate_a']:.3f} b={c['rate_b']:.3f} diff={c['diff']:+.3f} "
                  f"[{c['ci_low']:+.3f},{c['ci_high']:+.3f}] n={c['n_pairs']} p={c['p_mcnemar']:.3g} holm={c.get('p_holm')}")
    if "exp11_external" in out:
        c = out["exp11_external"]["contrasts"].get("uncensored_vs_base:hc_harmful_implicit", {})
        if c.get("n_pairs"):
            print(f"[external] uncensored-vs-base hc_harmful_implicit diff={c['diff']:+.3f} n={c['n_pairs']} p={c['p_mcnemar']:.3g}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", default=str(ROOT / "raw_transcripts"))
    ap.add_argument("--out-dir", default=str(ROOT / "records"))
    ap.add_argument("--effort", default="auto", help="cot_on reasoning effort to analyze (label); auto = the only one present")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--no-model", action="store_true", help="skip the Firth random-intercept fit")
    ap.add_argument("--suffix", default="", help="record name suffix, e.g. _lowprobe for a sensitivity analysis")
    a = ap.parse_args(argv)

    rows = load_raw(pathlib.Path(a.raw_root))
    if not rows:
        print(f"[error] no rows under {a.raw_root}/{RAW_PREFIX}*", file=sys.stderr)
        return 2
    out = recompute(rows, effort=a.effort, n_boot=a.n_boot, fit=not a.no_model)
    out_dir = pathlib.Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rec in out.items():
        path = out_dir / f"{name}{a.suffix}.json"
        path.write_text(json.dumps(rec, indent=1, allow_nan=False))
        print(f"[written] {path}")
    report(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
