"""Invariant I3: recompute every number in records/*.json from raw_transcripts/ (or, for the weight/activation measurement
receipts, from the per-item values stored in the record itself) and diff against the stored record.

  PYTHONPATH=src .venv/bin/python scripts/verify_all_receipts.py [--only NAME ...] [--skip-slow] [--tol 1e-9]

Prints one line per record: OK / MISMATCH (first differing paths, numbers only, never text) / ABSENT (record not written yet)
and exits non-zero on any mismatch. Records are rebuilt with the parameters stored in the record itself (effort, n_boot, ...).
`provenance` / `created_at` style keys are ignored. Raw-transcript rows may hold unredacted harmful scenario text: this script
never prints row content.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

R = ROOT / "records"
RAW = ROOT / "raw_transcripts"
IGNORE = {"provenance", "created_at", "wall_s", "elapsed_s", "est_cost_usd"}


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def jsonable(x):
    return json.loads(json.dumps(x, default=str))


def diff(a, b, path="$", tol=1e-9, out=None, limit=12):
    """Structural diff with float tolerance. Reports paths and numeric values only (never string content)."""
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k in IGNORE:
                continue
            if k not in a:
                out.append(f"{path}.{k}: only in recomputed")
            elif k not in b:
                out.append(f"{path}.{k}: only in stored record")
            else:
                diff(a[k], b[k], f"{path}.{k}", tol, out, limit)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} vs {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                diff(x, y, f"{path}[{i}]", tol, out, limit)
    elif isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        if a != b:
            out.append(f"{path}: {a!r} vs {b!r}")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not (a == b or (math.isnan(a) and math.isnan(b)) or abs(a - b) <= tol * max(1.0, abs(a), abs(b))):
            out.append(f"{path}: {a!r} vs {b!r}")
    elif type(a) is not type(b) or a != b:
        out.append(f"{path}: differs ({type(a).__name__}, {type(b).__name__})")
    return out


def load_json(name: str):
    p = R / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


# ---- individual verifiers: each returns (stored, recomputed) or None when the record does not exist yet --------------
def v_phase1(fam: str):
    name = f"phase1_dim_{fam}_analysis"
    stored = load_json(name)
    if stored is None:
        return None
    from v3 import phase1
    rows = []
    for f in sorted((RAW / stored["exp_id"]).glob("*.jsonl")):
        if not f.name.endswith(".partial.jsonl"):
            rows += [json.loads(l) for l in open(f) if l.strip()]
    scored = phase1.score_rows(rows)
    cells = phase1.aggregate(scored)
    arms = sorted({r["arm"] for r in scored})
    gates = {}
    for eff in sorted({r["effort"] for r in scored}, key=str):
        if "intact" in arms and "refusal_ablated" in arms:
            gates[str(eff)] = phase1.evaluate_gate(scored, effort=eff,
                                                   control_arms=tuple(a for a in phase1.DEFAULT_CONTROLS if a in arms))
    return stored, jsonable({"exp_id": stored["exp_id"], "n_rows": len(rows), "cells": cells, "gates": gates})


def v_projection(name: str):
    import numpy as np
    from v3 import modal_hf
    stored = load_json(name)
    if stored is None:
        return None
    # mean |dot| / ||h|| from the stored per-prompt values, then the flattened reductions
    recomputed = json.loads(json.dumps(stored))
    for ck in recomputed["checkpoints"].values():
        for by_set in ck["results"].values():
            for by_site in by_set.values():
                for st in by_site.values():
                    norm = np.asarray(st["h_norm"], dtype=float)
                    for d in st["vectors"].values():
                        d["mean_abs_ratio"] = float(np.mean(np.abs(np.asarray(d["dot"], dtype=float)) / norm))
    recomputed["summary"] = modal_hf.flatten_means(recomputed["checkpoints"])
    return ({"checkpoints": stored["checkpoints"], "summary": stored["summary"]},
            jsonable({"checkpoints": recomputed["checkpoints"], "summary": recomputed["summary"]}))


def v_kl(name: str):
    import numpy as np
    stored = load_json(name)
    if stored is None:
        return None
    recomputed = json.loads(json.dumps(stored))
    for ck in recomputed["checkpoints"].values():
        for ref in ck.get("vs", {}).values():
            for m in ref.values():
                if isinstance(m, dict) and "per_prompt" in m:
                    v = np.asarray(m["per_prompt"], dtype=float)
                    m.update(n=int(v.size), mean=float(v.mean()), median=float(np.median(v)), max=float(v.max()), min=float(v.min()))
    return stored["checkpoints"], recomputed["checkpoints"]


def v_decision():
    stored = load_json("phase1_family_decision")
    if stored is None:
        return None
    import phase1_decision
    return stored, jsonable(phase1_decision.compute())


def v_exp11(name: str):
    stored = load_json(name)
    if stored is None:
        return None
    mod = load_script("analyze_exp11")
    meta = stored["meta"]
    out = mod.recompute(mod.load_raw(RAW), effort=meta["analysis_effort"], n_boot=meta["n_boot"], fit=stored.get("model") is not None
                        or name != "exp11_main")
    return stored, jsonable(out[name.removesuffix("_lowprobe")])


def v_exp11_effort_robustness():
    stored = load_json("exp11_effort_robustness")
    if stored is None:
        return None
    mod = load_script("analyze_exp11_robustness")
    return stored, jsonable(mod.recompute(mod.load_raw(RAW), n_boot=stored["params"]["n_boot"]))


def v_prefix_records(modname: str, only: set[str]):
    """analyze_exp23 / analyze_exp2122: RECORDS[name] = {prefix, recompute}; rows via the module's own loader."""
    mod = load_script(modname)
    for name, spec in mod.RECORDS.items():
        if only and name not in only:
            continue
        stored = load_json(name)
        if stored is None:
            yield name, None
            continue
        rows = mod.load_rows(spec["prefix"])
        kw = {"n_boot": stored.get("n_boot", stored.get("params", {}).get("n_boot", 2000))}
        if modname == "analyze_exp23":
            kw["require_judge"] = stored.get("params", {}).get("require_judge", True)
            kw["models"] = stored.get("params", {}).get("models", True)
        yield name, (stored, jsonable(spec["recompute"](rows, **kw)))


def v_scenario_audit():
    import collections
    from v3 import author, scenarios
    stored = load_json("scenario_audit")
    if stored is None:
        return None
    bases = scenarios.load_bases(ROOT / "data" / "author" / "final_bases.jsonl")
    verdicts = json.loads((ROOT / "data" / "author" / "judge_verdicts.json").read_text())
    c = collections.Counter()
    for x in verdicts.values():
        if "severity" in x:
            c[f"severity_{x['severity']}"] += 1
            c[f"irreversibility_{x['irreversibility']}"] += 1
            c["instructional_" + x["instructional"]] += 1
            c["matched_" + x["matched"]] += 1
            c["pass_" + str(x["pass"]).lower()] += 1
    return stored, jsonable({**{k: stored[k] for k in ("schema", "note")}, "n_bases": len(bases),
                             "pair_audit": author.audit_pairs(bases), "judge_counts": dict(sorted(c.items()))})


def v_calibration_history():
    stored = load_json("calibration_history")
    if stored is None:
        return None
    mod = load_script("analyze_exp11")
    base = ROOT / "raw_transcripts" / "_calibration"
    dirs = {"v1": "exp11_calib_v1_original_wording", "v2": "exp11_calib_v2_latency_otherwise_ordinary", "v3": "exp11_calib_v3_FROZEN_WORDING"}
    out = {k: stored[k] for k in ("schema", "note")}
    out["versions"] = {}
    for v, d in dirs.items():
        rows = [json.loads(l) for l in open(base / d / "intact.jsonl") if l.strip()]
        cv = mod.recompute(rows, effort="auto", n_boot=50, fit=False)["exp11_main"]["construct_validity"]["cot_on"]
        crit = {k: {"k": x["value"]["k"], "n": x["value"]["n"], "rate": x["value"]["rate"], "pass": x["pass"]}
                for k, x in cv.items() if isinstance(x, dict) and "pass" in x}
        meta = stored["versions"][v]
        out["versions"][v] = {"system_prompt_sentence": meta["system_prompt_sentence"], "rows_description": meta["rows_description"],
                              "n_rows": len(rows), "criteria": crit}
    return stored, jsonable(out)


def v_exp14():
    stored = load_json("exp14_blocked_route")
    if stored is None:
        return None
    return load_script("analyze_exp14").verify_pair(stored, RAW)


def v_exp12_students():
    stored = load_json("exp12_students")
    if stored is None:
        return None
    mod = load_script("analyze_exp12")
    return stored["summary"], jsonable(mod.recompute(mod.iter_raw(), stored.get("params")))


def v_exp12_corpora():
    stored = load_json("exp12_corpora")
    if stored is None:
        return None
    mod = load_script("build_exp12_corpora")
    params = stored.get("params")
    return stored["summary"], jsonable(mod.recompute(None, params))


def v_exp31():
    stored = load_json("exp31_multidomain")
    if stored is None:
        return None
    mod = load_script("analyze_exp3")
    rows, _ = mod.load_rows([str(RAW / "exp31")])
    return stored, jsonable(mod.recompute(rows))


def v_exp32(name="exp32_probes"):
    stored = load_json(name)
    if stored is None:
        return None
    mod = load_script("analyze_exp3")
    from v3 import exp32
    import hashlib
    files = [ROOT / f["file"] for f in stored["provenance"]["raw"]]
    for f, want in zip(files, stored["provenance"]["raw"]):
        if hashlib.sha256(f.read_bytes()).hexdigest() != want["sha256"]:
            raise RuntimeError(f"raw file changed since the record was written: {f.name}")
    rows, _ = mod.load_rows([str(f) for f in files])
    acts = exp32.load_acts(stored["provenance"]["acts_dir"])
    return stored, jsonable(mod.recompute_exp32(rows, acts, stored.get("config")))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--skip-slow", action="store_true", help="skip exp32 (about 20-40 min)")
    ap.add_argument("--tol", type=float, default=1e-9)
    a = ap.parse_args(argv)
    only = set(a.only)

    jobs = [("phase1_dim_r1_analysis", lambda: v_phase1("r1")), ("phase1_dim_r2_analysis", lambda: v_phase1("r2")),
            ("projection_smoke", lambda: v_projection("projection_smoke")), ("projection_full13", lambda: v_projection("projection_full13")),
            ("kl_full13", lambda: v_kl("kl_full13")), ("phase1_family_decision", v_decision),
            ("exp11_main", lambda: v_exp11("exp11_main")), ("exp11_probes", lambda: v_exp11("exp11_probes")),
            ("exp11_external", lambda: v_exp11("exp11_external")),
            ("exp11_main_lowprobe", lambda: v_exp11("exp11_main_lowprobe")),
            ("exp11_probes_lowprobe", lambda: v_exp11("exp11_probes_lowprobe")),
            ("exp11_effort_robustness", v_exp11_effort_robustness),
            ("exp14_blocked_route", v_exp14), ("scenario_audit", v_scenario_audit), ("calibration_history", v_calibration_history),
            ("exp12_corpora", v_exp12_corpora), ("exp12_students", v_exp12_students),
            ("exp31_multidomain", v_exp31)]
    if not a.skip_slow:
        jobs.append(("exp32_probes", v_exp32))
        jobs.append(("exp32_probes_perm400", lambda: v_exp32("exp32_probes_perm400")))

    results: dict[str, str] = {}
    n_bad = 0

    def report(name, pair, dt):
        nonlocal n_bad
        if pair is None:
            results[name] = "ABSENT"
            print(f"[ABSENT  ] {name}")
            return
        stored, fresh = pair
        d = diff(jsonable(stored), fresh, tol=a.tol)
        if d:
            n_bad += 1
            results[name] = "MISMATCH"
            print(f"[MISMATCH] {name} ({dt:.0f}s): " + " | ".join(d[:6]))
        else:
            results[name] = "OK"
            print(f"[OK      ] {name} ({dt:.0f}s)")

    for name, fn in jobs:
        if only and name not in only:
            continue
        if name.startswith("exp11_") and load_json(name) is None:  # absent records are reported, not run
            report(name, None, 0)
            continue
        t0 = time.time()
        try:
            report(name, fn(), time.time() - t0)
        except Exception as e:  # a verifier that cannot run is a failure, not a pass
            n_bad += 1
            results[name] = "ERROR"
            print(f"[ERROR   ] {name}: {type(e).__name__}: {str(e)[:200]}")

    for modname in ("analyze_exp23", "analyze_exp2122"):
        t0 = time.time()
        try:
            for name, pair in v_prefix_records(modname, only):
                report(name, pair, time.time() - t0)
        except Exception as e:
            n_bad += 1
            print(f"[ERROR   ] {modname}: {type(e).__name__}: {str(e)[:200]}")

    n_ok = sum(v == "OK" for v in results.values())
    print(f"\n{n_ok} OK, {n_bad} failing, {sum(v == 'ABSENT' for v in results.values())} absent")
    prev = {}
    try:   # partial (--only) runs update their entries and keep the others
        prev = json.loads((R / "receipt_verification.json").read_text()).get("results", {})
    except Exception:
        pass
    prev.update(results)
    (R / "receipt_verification.json").write_text(json.dumps({"results": prev, "tol": a.tol}, indent=1, sort_keys=True))
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
