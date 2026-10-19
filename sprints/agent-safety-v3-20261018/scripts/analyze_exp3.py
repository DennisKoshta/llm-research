"""Analysis for Track 3: Exp 3.1 (multi-domain matrix) and Exp 3.2 (pre-action probes). Raw data in, records/*.json out.

  PYTHONPATH=src .venv/bin/python scripts/analyze_exp3.py exp31 [--raw raw_transcripts/exp31]
  PYTHONPATH=src .venv/bin/python scripts/analyze_exp3.py exp32 --acts data/exp32/main [--n-perm 50]
      # labels: raw_transcripts/exp11* (Exp 1.1 main grid) and, if present, raw_transcripts/exp31 (adds the exp11+exp31 dataset)

Receipt functions for scripts/verify_all_receipts.py (pure, deterministic, no file I/O):
  recompute(raw_rows)                      Exp 3.1 -> the dict stored in records/exp31_multidomain.json
  recompute_exp32(raw_rows, acts, config)  Exp 3.2 -> the dict stored in records/exp32_probes.json, where `acts` is
                                           v3.exp32.load_acts(<dir of the saved acts_<arm>.npz>) and `config` is the
                                           record's own "config" block
Both recompute every number from the raw episode rows (outcomes are REPLAYED from raw_output with v3.parse / Sandbox,
stored derived fields are ignored) and, for Exp 3.2, from the saved float32 activations. Truncated episodes are never
a shortcut and never a label. Output is aggregate numbers only (no prompt text).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v3 import exp31, exp32  # noqa: E402


def recompute(raw_rows: list) -> dict:
    """Exp 3.1 receipt function (see v3.exp31.recompute)."""
    return exp31.recompute(raw_rows)


def recompute_exp32(raw_rows: list, acts: dict, config: dict | None = None) -> dict:
    """Exp 3.2 receipt function (see v3.exp32.recompute)."""
    return exp32.recompute(raw_rows, acts, config)


def load_rows(paths: list[str]) -> tuple[list, list]:
    """Rows of every *.jsonl (not *.partial.jsonl) in the given directories / files, plus their provenance."""
    files = []
    for p in paths:
        p = pathlib.Path(p)
        files += sorted(f for f in p.glob("*.jsonl") if not f.name.endswith(".partial.jsonl")) if p.is_dir() else [p]
    rows, prov = [], []
    for f in files:
        part = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
        rows += part
        prov.append({"file": str(f.relative_to(ROOT)) if f.is_relative_to(ROOT) else str(f), "n_rows": len(part),
                     "sha256": hashlib.sha256(f.read_bytes()).hexdigest()})
    return rows, prov


def write_record(path: pathlib.Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1, sort_keys=True, allow_nan=False) + "\n")
    print(f"wrote {path}")


def print_exp31(rec: dict) -> None:
    i = rec["integrity"]
    print(f"rows={i['n_rows']} replayed={i['n_replayed']} invalid={i['n_invalid']} stored-vs-replay mismatches="
          f"{i['n_stored_fields_disagree_with_replay']} I1 violations={i['n_i1_violations']} forced prefix={i['n_forced_prefix']}")
    print("arm x cot x format: shortcut rate [Wilson 95%] (n scored) | truncated")
    for r in rec["by_arm_cot_format"]:
        s, t = r["shortcut"], r["truncation"]
        print(f"  {r['arm']:16s} {r['cot_mode']} {r['fmt']:9s} {s['rate'] if s['rate'] is None else round(s['rate'], 3)} "
              f"[{s['ci'][0]:.3f},{s['ci'][1]:.3f}] n={s['n']} | trunc {t['k']}/{t['n']}")
    for fam, items in rec["contrasts"].items():
        done = [c for c in items if "diff" in c]
        print(f"  contrasts {fam}: {len(done)} estimable of {len(items)}")
    m = rec["model"]
    print(f"  model: {m['status']}" + (f" n={m['n']} events={m['n_events']} converged={m['converged']}" if m["status"] == "ok" else f" {m.get('error', '')}"))


def print_exp32(rec: dict) -> None:
    for name, ds in rec["datasets"].items():
        for site, res in ds["sites"].items():
            if "regimes" not in res:
                print(f"{name}/{site}: {res.get('status')}")
                continue
            for cot, reg in res["regimes"].items():
                a, b = reg["A"], reg["B"]
                if "auroc" in a:
                    print(f"{name}/{site}/{cot} A: " + " ".join(f"{m}={v['auroc']:.3f}" for m, v in a["auroc"].items())
                          + f"  probe-design diff={a['diff']['probe-design']['diff']:.3f} [{a['diff']['probe-design']['ci_low']:.3f},"
                            f"{a['diff']['probe-design']['ci_high']:.3f}]")
                if "auroc" in b:
                    print(f"{name}/{site}/{cot} B (intact): " + " ".join(f"{m}={v['auroc']:.3f}" for m, v in b["auroc"].items()))
                c = reg.get("C")
                if c and c.get("scopes"):
                    for g in c["scopes"].get("arm_cot", []):
                        p = g["pooled"]
                        print(f"{name}/{site}/{cot} C {g['group']} (n={g['n']}, strata={g['n_strata']}): "
                              + " ".join(f"{m}={v['auroc']:.3f}[{v['ci_low']:.3f},{v['ci_high']:.3f}]" for m, v in p.items()))
                elif c:
                    print(f"{name}/{site}/{cot} C: {c.get('status')} (eligible cells: {c.get('n_eligible')})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a31 = sub.add_parser("exp31")
    a31.add_argument("--raw", default=str(ROOT / "raw_transcripts/exp31"))
    a31.add_argument("--out", default=str(ROOT / "records/exp31_multidomain.json"))
    a31.add_argument("--n-boot", type=int, default=exp31.N_BOOT)
    a31.add_argument("--no-model", action="store_true", help="skip the Firth mixed model (slow)")
    a32 = sub.add_parser("exp32")
    a32.add_argument("--acts", required=True, help="directory with acts_<arm>.npz (scripts/run_exp32.py output)")
    a32.add_argument("--exp11-raw", default=None, help="comma-separated raw_transcripts directories of the Exp 1.1 sweeps (default: raw_transcripts/exp11*)")
    a32.add_argument("--exp31-raw", default=str(ROOT / "raw_transcripts/exp31"), help="Exp 3.1 sweep (adds the exp11+exp31 dataset when it exists)")
    a32.add_argument("--no-exp31", action="store_true")
    a32.add_argument("--out", default=str(ROOT / "records/exp32_probes.json"))
    a32.add_argument("--n-perm", type=int, default=exp32.DEFAULT_CONFIG["n_perm"])
    a32.add_argument("--n-boot", type=int, default=exp32.DEFAULT_CONFIG["n_boot"])
    a32.add_argument("--min-minority", type=int, default=exp32.DEFAULT_CONFIG["min_minority"])
    a32.add_argument("--site", default=exp32.PRIMARY_SITE, choices=exp32.SITES)
    a32.add_argument("--effort", default="auto", choices=["auto", "low", "medium", "xhigh", "default"],
                     help="reasoning effort of the cot_on labels; must equal the one the activations were captured at ('auto' = the only one present)")
    args = ap.parse_args(argv)

    if args.cmd == "exp31":
        rows, prov = load_rows([args.raw])
        rec = exp31.recompute(rows, n_boot=args.n_boot, model=not args.no_model)
        rec["provenance"] = {"raw": prov}
        write_record(pathlib.Path(args.out), rec)
        print_exp31(rec)
        return 0
    exp11_dirs = args.exp11_raw.split(",") if args.exp11_raw else sorted(str(d) for d in (ROOT / "raw_transcripts").glob("exp11*") if d.is_dir())
    if not exp11_dirs:
        raise SystemExit("no Exp 1.1 raw transcripts found (raw_transcripts/exp11*); pass --exp11-raw")
    rows, prov = load_rows(exp11_dirs)
    datasets = {"exp11": ["exp11"]}
    if not args.no_exp31 and pathlib.Path(args.exp31_raw).is_dir():
        more, prov31 = load_rows([args.exp31_raw])
        rows, prov = rows + more, prov + prov31
        datasets["exp11+exp31"] = ["exp11", "exp31"]
    acts = exp32.load_acts(args.acts)
    if not acts:
        raise SystemExit(f"no acts_<arm>.npz in {args.acts}")
    config = {"n_perm": args.n_perm, "n_boot": args.n_boot, "min_minority": args.min_minority, "site": args.site, "effort": args.effort,
              "sensitivity_sites": [s for s in exp32.SITES if s != args.site], "datasets": datasets}
    rec = recompute_exp32(rows, acts, config)
    rec["provenance"] = {"raw": prov, "acts_dir": str(args.acts)}
    write_record(pathlib.Path(args.out), rec)
    print_exp32(rec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
