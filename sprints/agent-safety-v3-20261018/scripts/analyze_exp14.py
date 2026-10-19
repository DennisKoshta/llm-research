"""Analyze Exp 1.4 (EXPLORATORY, blocked ordinary route) from raw_transcripts only; write records/exp14_blocked_route.json.

  PYTHONPATH=src .venv/bin/python scripts/analyze_exp14.py [--n-boot 5000] [--raw-root DIR] [--out-dir DIR]

Reads raw_transcripts/exp14*/<arm>.jsonl (partial files ignored) and, for contrast C4 (blocked vs the same instance
unblocked), the Exp 1.1 high_tedium main episodes of the 27B arms from raw_transcripts/exp11*/<arm>.jsonl; the latter may be
absent (C4 is then skipped and a warning recorded). Every derived field is recomputed with v3.episodes.recompute_record.
`recompute(raw_rows)` is the pure function scripts/verify_all_receipts.py would call, through `verify_pair`. Prints aggregate
numbers only (no scenario or model text).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v3 import exp14  # noqa: E402

RECORD = "exp14_blocked_route"
RAW_PREFIX = "exp14"
REFERENCE_PREFIX = "exp11"
DESIGN_DOC = ROOT / "reports" / "exp14_design.md"


def _files(raw_root: pathlib.Path, prefix: str) -> list[pathlib.Path]:
    out = []
    for d in sorted(p for p in raw_root.glob(prefix + "*") if p.is_dir()):
        out += [f for f in sorted(d.glob("*.jsonl")) if not f.name.endswith(".partial.jsonl")]
    return out


def _read(f: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in f.read_text().splitlines() if line.strip()]


def load_raw(raw_root) -> list[dict]:
    """Every exp14 row of raw_transcripts/exp14*/ plus the Exp 1.1 reference rows (high_tedium main, cot_on, 27B arms) of
    raw_transcripts/exp11*/, in a stable order. Missing Exp 1.1 data is tolerated."""
    root = pathlib.Path(raw_root)
    rows = []
    for f in _files(root, RAW_PREFIX):
        rows += _read(f)
    for f in _files(root, REFERENCE_PREFIX):
        rows += [r for r in _read(f) if exp14.row_kind(r) == "reference"]
    return rows


def recompute(raw_rows, **kw) -> dict:
    return exp14.recompute(raw_rows, **kw)


def provenance(raw_root: pathlib.Path) -> dict:
    """Which files fed the record (names, sizes, sha256). Ignored by the receipt verifier."""
    def entry(f):
        return {"file": str(f.relative_to(raw_root)), "bytes": f.stat().st_size, "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
    return {"exp14_raw": [entry(f) for f in _files(raw_root, RAW_PREFIX)],
            "exp11_reference_raw": [entry(f) for f in _files(raw_root, REFERENCE_PREFIX)],
            "design_doc_sha256": hashlib.sha256(DESIGN_DOC.read_bytes()).hexdigest() if DESIGN_DOC.exists() else None}


def verify_pair(stored: dict, raw_root=None) -> tuple:
    """(stored, recomputed) for scripts/verify_all_receipts.py: rebuilt with the parameters stored in the record."""
    rows = load_raw(raw_root or ROOT / "raw_transcripts")
    return stored, recompute(rows, n_boot=stored["meta"]["n_boot"])[RECORD]


def _pct(r) -> str:
    return "n/a" if not r or r.get("rate") is None else f"{100 * r['rate']:.1f}% ({r['k']}/{r['n']})"


def report(out: dict) -> None:
    rec = out[RECORD]
    h = rec["meta"]["hygiene"]
    print(f"[EXPLORATORY] Exp 1.4 blocked ordinary route; efforts={rec['meta']['efforts']} primary={rec['meta']['primary_effort']}")
    print(f"[hygiene] rows={h['n_rows']} blocked={h['n_blocked_rows']} reference={h['n_reference_rows']} other={h['n_other_rows']} "
          f"invalid={h['n_invalid_records']} stored_mismatch={h['n_stored_mismatch_rows']} dup={h['n_duplicate_rows']} "
          f"unplaceable={h['n_unplaceable']} I1_violations={h['n_i1_violations']} banned_literal={h['n_banned_literal_rows']} "
          f"prompt_suffix={h['n_prompt_suffix_violations']} block_spec_mismatch={h['n_block_spec_mismatch_rows']} "
          f"block_text_in_prompt={h['n_block_text_in_prompt_rows']} safe_route_completed={h['n_safe_route_completed_rows']}")
    for w in rec["meta"]["warnings"]:
        print("[WARN]", w)
    for eff, a in rec["analyses"].items():
        print(f"[effort={eff} role={a['role']}] outcome split per arm, implicit boundary, both timings pooled (ITT; gave_up = escalated + stopped_text):")
        for arm, g in a["tables"].items():
            for key in ("harmful|implicit", "harmless|implicit"):
                c = g["timing_pooled"][key]
                k = c["counts"]
                print(f"  {arm:16s} {key:17s} n={c['n']} shortcut={_pct(c['shortcut'])} gave_up={_pct(c['gave_up'])} "
                      f"(escalated {k['ESCALATED']}, stopped_text {k['STOPPED_TEXT']}) persisted={_pct(c['persisted'])} not_scored={k['NOT_SCORED']} "
                      f"reached_block={_pct(c['reached_block'])}")
                gb = c["given_block_reached"]
                if gb["n"]:
                    o = gb["outcomes"]
                    print(f"  {'':16s} {'  among those who met the block':17s} n={gb['n']} shortcut={_pct(o['SHORTCUT'])} escalated={_pct(o['ESCALATED'])} "
                          f"stopped_text={_pct(o['STOPPED_TEXT'])} persisted={_pct(o['STUCK_RETRYING'])} not_scored={gb['counts']['NOT_SCORED']}")
        for name, c in a["contrasts"]["confirmatory"].items():
            if c.get("n_pairs"):
                holm = c.get("p_holm")
                print(f"  [contrast {name}] a={c['rate_a']:.3f} b={c['rate_b']:.3f} diff={c['diff']:+.3f} "
                      f"[{c['ci_low']:+.3f},{c['ci_high']:+.3f}] n={c['n_pairs']} p={c['p_mcnemar']:.3g} holm={'n/a' if holm is None else f'{holm:.3g}'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", default=str(ROOT / "raw_transcripts"))
    ap.add_argument("--out-dir", default=str(ROOT / "records"))
    ap.add_argument("--n-boot", type=int, default=5000)
    a = ap.parse_args(argv)

    raw_root = pathlib.Path(a.raw_root)
    rows = load_raw(raw_root)
    if not any(exp14.row_kind(r) == "blocked" for r in rows):
        print(f"[error] no exp14 rows under {a.raw_root}/{RAW_PREFIX}*", file=sys.stderr)
        return 2
    out = recompute(rows, n_boot=a.n_boot)
    out[RECORD]["provenance"] = provenance(raw_root)
    out_dir = pathlib.Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{RECORD}.json"
    path.write_text(json.dumps(out[RECORD], indent=1, allow_nan=False))
    print(f"[written] {path}")
    report(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
