"""Build the Exp 1.1 request files under data/exp11/ (design, counts and launch commands: v3.exp11 docstring).

  PYTHONPATH=src .venv/bin/python scripts/build_exp11_requests.py --scale 1.0 --effort low
  PYTHONPATH=src .venv/bin/python scripts/build_exp11_requests.py --scale 0.5 --effort default --skip external --dry-run

Bases default to data/author/final_bases.jsonl (authored on Modal, unredacted: never printed). `--placeholder`
uses placeholder bases instead (tests / dry runs) and writes to data/exp11_placeholder unless --out-dir is given.
`--effort` defaults to the recorded Phase 1 decision (records/phase1_effort_decision.json if present, else the
"reasoning_effort_for_phase2" of records/phase1_family_decision.json). Token caps default to exp11.default_caps(effort)
(16384 cot_on at the template default effort, 4096 at low). Prints counts, the cost estimate and the launch commands
only, never any scenario text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v3 import exp11, scenarios  # noqa: E402

EFFORT_DECISIONS = (ROOT / "records" / "phase1_effort_decision.json", ROOT / "records" / "phase1_family_decision.json")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bases", default=str(ROOT / "data" / "author" / "final_bases.jsonl"))
    ap.add_argument("--placeholder", action="store_true", help="use scenarios.make_placeholder_bases(40) instead of --bases")
    ap.add_argument("--scale", type=float, default=1.0, help="fraction of the bases, stratified over templates (0.5 -> 20 of 40)")
    ap.add_argument("--effort", choices=exp11.EFFORT_CHOICES, default=None,
                    help="reasoning effort recorded in every cot_on row; default = recorded Phase 1 decision")
    ap.add_argument("--cot-on-cap", type=int, default=None, help="default: exp11.default_caps(effort)")
    ap.add_argument("--cot-off-cap", type=int, default=None, help=f"default: {exp11.COT_OFF_CAP}")
    ap.add_argument("--skip", default="", help=f"comma-separated optional groups to drop: {','.join(exp11.SKIPPABLE)}")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--dry-run", action="store_true", help="print counts, estimate and commands; write nothing")
    a = ap.parse_args(argv)

    effort_label = a.effort or exp11.load_effort_decision(next((p for p in EFFORT_DECISIONS if p.exists()), EFFORT_DECISIONS[0]))
    effort = exp11.resolve_effort(effort_label)
    on_cap, off_cap = exp11.default_caps(effort)
    on_cap, off_cap = a.cot_on_cap or on_cap, a.cot_off_cap or off_cap
    skip = tuple(s for s in a.skip.split(",") if s)
    out_dir = pathlib.Path(a.out_dir or ROOT / "data" / ("exp11_placeholder" if a.placeholder else "exp11"))

    if a.placeholder:
        bases = scenarios.make_placeholder_bases(40)
    else:
        if not pathlib.Path(a.bases).exists():
            print(f"[error] {a.bases} not found (authoring has not produced final bases yet); use --placeholder for a dry run", file=sys.stderr)
            return 2
        bases = scenarios.load_bases(a.bases)
    chosen = exp11.select_bases(bases, a.scale)
    files = exp11.build_requests(chosen, effort=effort, cot_on_cap=on_cap, cot_off_cap=off_cap, skip=skip)
    estimate = exp11.estimate_cost(a.scale, len(bases), skip=skip, cot_on_cap=on_cap)
    rel = out_dir.relative_to(ROOT) if out_dir.is_relative_to(ROOT) else out_dir
    commands = exp11.launch_commands(files, estimate, str(rel))

    print(f"[bases] {len(chosen)} of {len(bases)} (scale {a.scale}); per template: "
          f"{dict(sorted(Counter(b['template_id'] for b in chosen).items()))}")
    print(f"[settings] effort={effort_label} cot_on_cap={on_cap} cot_off_cap={off_cap} max_turns={exp11.MAX_TURNS} skip={list(skip)}")
    for stem, rows in files.items():
        print(f"[rows] requests_{stem}.jsonl: {len(rows)}")
    print(f"[estimate] generations={estimate['generations']} decode_tokens={estimate['decode_tokens']} gpu_hours={estimate['gpu_hours']} "
          f"usd={estimate['usd']} usd_worst={estimate['usd_worst']}")
    pess = exp11.estimate_cost(a.scale, len(bases), skip=skip, model=exp11.PESSIMISTIC, cot_on_cap=on_cap)
    print(f"[estimate] pessimistic (2000-token turn 1, 7k tok/s prefill): usd={pess['usd']} usd_worst={pess['usd_worst']}")
    for exp_id, e in estimate["launches"].items():
        print(f"[estimate] {exp_id}: {len(e['arms'])} arms x {e['gpus_per_arm']} GPU, {e['hours_per_arm']} h/arm "
              f"(worst {e['hours_per_arm_worst']}), usd={e['usd']} (worst {e['usd_worst']})")
    if a.dry_run:
        print("[dry-run] nothing written")
    else:
        info = exp11.write_requests(files, out_dir)
        blob = json.dumps([{k: b[k] for k in ("scenario_id", "author_index")} for b in chosen], sort_keys=True)
        manifest = {"experiment": "1.1", "placeholder": a.placeholder, "scale": a.scale, "n_bases": len(chosen),
                    "n_bases_available": len(bases), "scenario_ids": [b["scenario_id"] for b in chosen],
                    "selection_sha256": hashlib.sha256(blob.encode()).hexdigest(),
                    "effort": effort_label, "cot_on_cap": on_cap, "cot_off_cap": off_cap,
                    "max_turns": exp11.MAX_TURNS, "skip": list(skip), "files": info,
                    "estimate": estimate, "launch_commands": commands}
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
        print(f"[written] {len(files)} files + manifest.json under {out_dir}")
    for c in commands:
        print("[launch]", c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
