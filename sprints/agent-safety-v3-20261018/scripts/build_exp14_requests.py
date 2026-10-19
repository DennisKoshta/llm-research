"""Build the Exp 1.4 (EXPLORATORY, blocked ordinary route) request files under data/exp14/.

  PYTHONPATH=src .venv/bin/python scripts/build_exp14_requests.py --dry-run
  PYTHONPATH=src .venv/bin/python scripts/build_exp14_requests.py            # writes data/exp14/

Design, outcome definitions and contrasts: reports/exp14_design.md (its sha256 goes into the manifest) and the v3.exp14
docstring. Bases default to data/author/final_bases.jsonl (authored on Modal, unredacted: never printed). `--placeholder`
uses placeholder bases instead (tests / dry runs) and writes to data/exp14_placeholder unless --out-dir is given.
Every instance is the Exp 1.1 high_tedium main-grid instance of its (base, consequence, boundary) plus the sandbox `block`
spec, so the prompt is byte-identical to Exp 1.1 and never announces a block (asserted). Prints counts, the cost estimate and
the exact launch lines only, never any scenario text.
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

from v3 import exp11, exp14, scenarios  # noqa: E402

DESIGN_DOC = ROOT / "reports" / "exp14_design.md"


def _sha(path: pathlib.Path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bases", default=str(ROOT / "data" / "author" / "final_bases.jsonl"))
    ap.add_argument("--placeholder", action="store_true", help="use scenarios.make_placeholder_bases(40) instead of --bases")
    ap.add_argument("--scale", type=float, default=1.0, help="fraction of the bases, stratified over templates (1.0 = all 40)")
    ap.add_argument("--skip-low", action="store_true", help="do not build the low-effort sensitivity file")
    ap.add_argument("--est-hours", type=float, default=exp14.EST_HOURS, help="--est-hours printed for the default-effort launch")
    ap.add_argument("--est-hours-low", type=float, default=exp14.EST_HOURS_LOW, help="--est-hours printed for the low-effort launch")
    ap.add_argument("--raw-root", default=str(ROOT / "raw_transcripts"), help="Exp 1.1 raw transcripts for the measured token statistics")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--dry-run", action="store_true", help="print counts, estimate and commands; write nothing")
    a = ap.parse_args(argv)

    out_dir = pathlib.Path(a.out_dir or ROOT / "data" / ("exp14_placeholder" if a.placeholder else "exp14"))
    if a.placeholder:
        bases = scenarios.make_placeholder_bases(40)
    else:
        if not pathlib.Path(a.bases).exists():
            print(f"[error] {a.bases} not found (authoring has not produced final bases yet); use --placeholder for a dry run", file=sys.stderr)
            return 2
        bases = scenarios.load_bases(a.bases)
    chosen = exp11.select_bases(bases, a.scale)
    files = exp14.build_requests(chosen, skip_low=a.skip_low)
    design = exp14.check_design(files, len(chosen))
    measured = exp14.measure_exp11(a.raw_root)
    table = {**exp14.MEASURED_EXP11, **measured}
    estimate = exp14.estimate_cost(len(chosen), measured=table, skip_low=a.skip_low)
    rel = out_dir.relative_to(ROOT) if out_dir.is_relative_to(ROOT) else out_dir
    commands = exp14.launch_commands(files, str(rel), est_hours=a.est_hours, est_hours_low=a.est_hours_low)

    print("[EXPLORATORY] Exp 1.4 blocked ordinary route (not part of the original specification)")
    print(f"[bases] {len(chosen)} of {len(bases)} (scale {a.scale}); per template: "
          f"{dict(sorted(Counter(b['template_id'] for b in chosen).items()))}")
    print(f"[settings] cot_on max_turns={exp14.MAX_TURNS} timings={dict(exp14.AFTER_STEP)} efforts=default(cap {exp11.default_caps(None)[0]})"
          + ("" if a.skip_low else f",low(cap {exp11.default_caps('low')[0]})"))
    for stem, rows in files.items():
        print(f"[rows] requests_{stem}.jsonl: {len(rows)} (cells {sorted(set(design[stem]['cells'].values()))} each; "
              f"{design[stem]['prompts_shared_by_early_and_late']} prompts shared by the early and late twin)")
    print(f"[hygiene] prompts identical to the Exp 1.1 high_tedium twins, block text absent, no </think>, no DECLARED_TOOL_NAME, "
          f"shortcut tool named, I1 caps: OK ({sum(len(r) for r in files.values())} rows checked)")
    print(f"[estimate] measured Exp 1.1 token statistics: {'raw_transcripts + meta.json' if measured else 'embedded table'} "
          f"({len(table)} arm x effort entries)")
    print(f"[estimate] generations={estimate['generations']} decode_tokens={estimate['decode_tokens']} gpu_hours={estimate['gpu_hours']} "
          f"usd={estimate['usd']} usd_worst={estimate['usd_worst']}")
    for exp_id, e in estimate["launches"].items():
        reserve = a.est_hours if e["effort"] == "default" else a.est_hours_low
        verdict = "covers" if reserve >= e["hours_per_arm_worst_max"] else "does NOT cover"
        print(f"[estimate] {exp_id}: {len(e['arms'])} arms x 1 GPU, central {e['hours_per_arm_mean']} h/arm, worst {e['hours_per_arm_worst_max']} h/arm "
              f"(--est-hours {reserve} {verdict} the worst case), usd={e['usd']} (worst {e['usd_worst']})")
    if a.dry_run:
        print("[dry-run] nothing written")
    else:
        info = exp14.write_requests(files, out_dir)
        blob = json.dumps([{k: b[k] for k in ("scenario_id", "author_index")} for b in chosen], sort_keys=True)
        manifest = {"experiment": exp14.EXPERIMENT, "exploratory": True, "label": exp14.LABEL, "placeholder": a.placeholder,
                    "scale": a.scale, "n_bases": len(chosen), "n_bases_available": len(bases),
                    "scenario_ids": [b["scenario_id"] for b in chosen],
                    "selection_sha256": hashlib.sha256(blob.encode()).hexdigest(),
                    "design_doc": "reports/exp14_design.md", "design_sha256": _sha(DESIGN_DOC),
                    "settings": {"cot_mode": "cot_on", "max_turns": exp14.MAX_TURNS, "cost_regime": exp14.COST,
                                 "timings": exp14.AFTER_STEP, "block": {t: exp14.block_spec(t) for t in exp14.TIMINGS},
                                 "caps": {p.stem: exp14.part_cap(p) for p in exp14.PARTS if p.stem in files},
                                 "efforts": {p.stem: exp11.effort_label(p.effort) for p in exp14.PARTS if p.stem in files}},
                    "files": info, "design_check": design, "estimate": estimate, "launch_commands": commands}
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
        print(f"[written] {len(files)} files + manifest.json under {out_dir}")
    for c in commands:
        print("[launch]", c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
