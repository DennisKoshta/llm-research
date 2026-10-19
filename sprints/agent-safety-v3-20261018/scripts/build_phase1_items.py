"""Build the Phase 1 Direct-Instruction Positive Control request files from the read-only donor sprints.

  PYTHONPATH=src .venv/bin/python scripts/build_phase1_items.py [--out-dir data/phase1] [--include-text]

Writes data/phase1/requests_set_{a0,a1,a2,b,c}.jsonl (+ requests_set_a{0,1,2}_text.jsonl with --include-text)
and manifest.json (counts, sha256 of every output and of every donor input). The donor files contain harmful
scenario text: this script prints ONLY file names and counts.
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

from v3 import phase1  # noqa: E402

DONORS = {
    "set_a": "agent-safety-clean-v2-20261017/stimuli/phase1_qualification.jsonl",
    "set_b": "agent-safety-three-track-20261016/data/exp1_1/direct_instruction_60_controls.json",
    "set_c": "agent-safety-three-track-20261016/data/phase1_ablation/qualification_heldout_prompts.json",
}


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: pathlib.Path, rows: list) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "phase1"))
    ap.add_argument("--donor-root", default=str(ROOT.parent), help="directory holding the two donor sprints")
    ap.add_argument("--include-text", action="store_true", help="also build Set A text-mode (submit_assessment) items")
    a = ap.parse_args()
    out_dir, donor_root = pathlib.Path(a.out_dir), pathlib.Path(a.donor_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {k: donor_root / v for k, v in DONORS.items()}

    donor_a = [json.loads(line) for line in paths["set_a"].read_text(encoding="utf-8").splitlines() if line.strip()]
    donor_b = json.loads(paths["set_b"].read_text(encoding="utf-8"))
    donor_c = json.loads(paths["set_c"].read_text(encoding="utf-8"))

    sets = {f"requests_set_{v.lower()}": rows for v, rows in phase1.build_set_a(donor_a, include_text=a.include_text).items()}
    sets["requests_set_b"] = phase1.build_set_b(donor_b)
    sets["requests_set_c"] = phase1.build_set_c(donor_c)
    all_rows = [r for rows in sets.values() for r in rows]
    phase1.validate_rows(all_rows)
    assert len(sets["requests_set_b"]) == 30, "Set B must have 30 unique prompts"

    for stale in out_dir.glob("requests_set_*.jsonl"):   # e.g. text-mode files from an earlier --include-text build
        if stale.stem not in sets:
            stale.unlink()
    files = {}
    for stem, rows in sets.items():
        path = out_dir / f"{stem}.jsonl"
        write_jsonl(path, rows)
        files[path.name] = {"n_rows": len(rows), "sha256": sha256(path), "set_name": rows[0]["meta"]["set_name"],
                            "variant": rows[0]["meta"]["variant"],
                            "consequence": dict(sorted(collections.Counter(r["meta"]["consequence"] for r in rows).items())),
                            "max_new_tokens": rows[0]["max_new_tokens"]}
    manifest = {
        "exp_id": phase1.EXP_ID, "include_text": a.include_text, "n_rows_total": len(all_rows),
        "defaults": {"kind": "single", "cot_mode": "cot_on", "reasoning_effort": None, "max_new_tokens": phase1.MAX_NEW_TOKENS},
        "files": files,
        "donor_inputs": {k: {"path": v, "sha256": sha256(paths[k]),
                             "n_items": len(donor_a) if k == "set_a" else len(donor_b) if k == "set_b" else len(donor_c)}
                         for k, v in DONORS.items()},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    for name, f in files.items():
        print(f"{name}: {f['n_rows']} rows {f['consequence']}")
    print(f"total {len(all_rows)} rows -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
