"""Build the reviewer deconfounder request files (design: v3.deconf docstring).

  PYTHONPATH=src .venv/bin/python scripts/build_deconf_requests.py [--dry-run]

Writes data/deconf/requests_deconf_exp11.jsonl, data/deconf/requests_deconf_exp31.jsonl and data/deconf/manifest.json
(counts, sha256, launch commands; no prompt text). Inputs: data/author/final_bases.jsonl, data/author/judge_verdicts.json
and data/exp31/manifest.json (the Exp 3.1 scenario selection). Prints counts only.
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

from v3 import deconf, scenarios  # noqa: E402

OUT = ROOT / "data" / "deconf"
EST_HOURS = 1.2      # worst-case wall hours per arm (Exp 1.1 / 3.1 timings: ~0.4-0.6 h for comparable token volume)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    bases = scenarios.load_bases(ROOT / "data/author/final_bases.jsonl")
    verdicts = json.loads((ROOT / "data/author/judge_verdicts.json").read_text())
    sel = json.loads((ROOT / "data/exp31/manifest.json").read_text())["scenario_ids"]
    files = {deconf.EXP11_ID: deconf.build_exp11_rows(bases), deconf.EXP31_ID: deconf.build_exp31_rows(bases, verdicts, sel)}
    manifest = {"design": "v3.deconf", "arms": list(deconf.ARMS), "files": {}, "launch_commands": []}
    for exp_id, rows in files.items():
        counts = Counter((r["meta"]["condition"], r["meta"]["effort"], r.get("arm", "all")) for r in rows)
        print(f"[rows] {exp_id}: {len(rows)} " + json.dumps({"|".join(k): v for k, v in sorted(counts.items())}))
        path = OUT / f"requests_{exp_id}.jsonl"
        blob = "".join(json.dumps(r) + "\n" for r in rows)
        cmd = (f"PYTHONPATH=src .venv/bin/python scripts/run_requests.py --exp-id {exp_id} --requests {path.relative_to(ROOT)} "
               f"--arms {','.join(deconf.ARMS)} --gpu H200 --max-num-seqs 384 --eager 0 --est-hours {EST_HOURS}")
        manifest["files"][exp_id] = {"rows": len(rows), "sha256": hashlib.sha256(blob.encode()).hexdigest(),
                                     "counts": {"|".join(k): v for k, v in sorted(counts.items())}}
        manifest["launch_commands"].append(cmd)
        print("[launch]", cmd)
        if not a.dry_run:
            OUT.mkdir(parents=True, exist_ok=True)
            path.write_text(blob)
    if not a.dry_run:
        (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
        print(f"[written] {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
