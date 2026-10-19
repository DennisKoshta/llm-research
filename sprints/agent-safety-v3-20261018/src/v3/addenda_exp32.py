"""Exp 3.2 reviewer addenda (2026-10-19): extra numbers recomputed from the same raw rows + activations as
records/exp32_probes.json, with exp32's own helpers (labels, table, template folds, learners, Regime C demeaning,
bootstraps), so they are comparable with the paper's Table 8 / Table 9.

1. regime_a_incremental  Regime A (full grid, one cot mode, all arms pooled, site "out"): a combined "design + probe"
   predictor (one-hot of the saturated design cell arm x cost x consequence x boundary, concatenated with the h48
   activations, every column standardised on the training fold, RidgeClassifierCV over exp32.ALPHAS, the same
   5-fold leave-templates-out CV) against the design lookup alone. Variant `design_probe_balanced`: the same, but after
   standardisation the cell block is multiplied by sqrt(D / n_cell_columns), so both blocks carry the same total
   variance under the one shared ridge penalty (the plain concatenation lets 5120 activation columns swamp ~24 cell
   columns). Paired scenario-resampling bootstrap CIs of the AUROC differences (stats.auroc_diff_bootstrap_ci).
2. regime_c_additions  (a) the within-cell TF-IDF baseline (exp32's Regime C treatment: TF-IDF features demeaned per
   cell, training labels centred per cell, held-out scores centred per (cell, fold), stratified AUROC) for every
   arm x cot group of Table 9; (b) the harmful-cell split-trial probes: every eligible harmful stratum trained alone
   (exp32's "cell" scope, the source of the 0.554 / 0.747 quoted in the report) plus all eligible harmful strata of
   one cot mode pooled (new scope "harmful_cot").
3. fpr_at_80tpr  lowest FPR among thresholds with TPR >= 0.80 (a threshold t flags s >= t; ties move together).
4. projection_intact  q = |h48 . r_hat| / ||h48|| on the Intact arm's Exp 1.1 pre-think states, paired contrasts
   within (scenario, cost regime), scenario-cluster bootstrap CI and scenario sign-flip p.

`recompute(raw_root, acts_root, params)`: raw_root = the raw_transcripts directory (the files of
params["raw_subdirs"] are read like scripts/analyze_exp3.load_rows), acts_root = the exp32 activation directory.
Never prints or returns prompt / completion text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time
from collections import defaultdict

import numpy as np

from v3 import exp32, stats

ROOT = pathlib.Path(__file__).resolve().parents[2]

DEFAULT_PARAMS = {
    # the stored exp32_probes.json config (only the fields used here)
    "effort": "default", "site": exp32.PRIMARY_SITE, "n_folds": 5, "n_boot": 2000, "seed": stats.DEFAULT_SEED,
    "min_minority": 4, "threads": 4, "datasets": {"exp11": ["exp11"], "exp11+exp31": ["exp11", "exp31"]},
    # addenda options
    "raw_subdirs": ["exp11", "exp11_cotoff", "exp31"],
    "primary_dataset": "exp11",
    "regime_a_datasets": ["exp11", "exp11+exp31"],
    "balanced_variant": True,
    "tpr_target": 0.80,
    "projection_sites": ["out", "in"],
    "projection_n_boot": 5000,
}


# ------------------------------------------------------------------------------------------ small helpers
def fpr_at_tpr(y, s, target: float = 0.80) -> dict:
    """Lowest FPR among thresholds achieving TPR >= target (rule: flag s >= t; tied scores move together)."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    P, N = int(y.sum()), int(len(y) - y.sum())
    if P == 0 or N == 0:
        return {"fpr": None, "tpr": None, "n_pos": P, "n_neg": N}
    order = np.argsort(-s, kind="mergesort")
    ss, yy = s[order], y[order]
    tp, fp = np.cumsum(yy), np.cumsum(1 - yy)
    last = np.r_[np.flatnonzero(np.diff(ss) != 0), len(ss) - 1]     # last row of each tie group
    tpr, fpr = tp[last] / P, fp[last] / N
    k = int(np.argmax(tpr >= target - 1e-12))                       # fpr is non-decreasing along the sweep
    return {"fpr": float(fpr[k]), "tpr": float(tpr[k]), "n_pos": P, "n_neg": N}


def _ridge_blocks(blocks_tr: list, y_tr, blocks_te: list, weights: list) -> np.ndarray:
    """exp32.ridge_classifier_scores on concatenated blocks; each block standardised on the training fold, then scaled
    by its weight (weights all 1 == the plain concatenation)."""
    if len(set(y_tr)) < 2:
        return np.zeros(len(blocks_te[0]))
    from sklearn.linear_model import RidgeClassifierCV

    tr, te = [], []
    for b_tr, b_te, w in zip(blocks_tr, blocks_te, weights):
        z_tr, z_te = exp32._standardise(np.asarray(b_tr, np.float64), np.asarray(b_te, np.float64))
        tr.append(w * z_tr)
        te.append(w * z_te)
    return RidgeClassifierCV(alphas=exp32.ALPHAS).fit(np.hstack(tr), y_tr).decision_function(np.hstack(te))


def cluster_mean_ci(d, clusters, n_boot: int, seed: int) -> dict:
    """Mean of per-pair values d, scenario cluster bootstrap 95% CI (pair-weighted ratio of resampled cluster sums) and
    the two-sided cluster sign-flip p (stats._signflip_p). Same algorithm as v3.deconf.cluster_mean."""
    d = np.asarray(d, dtype=float)
    groups = defaultdict(list)
    for i, c in enumerate(clusters):
        groups[c].append(i)
    idx = [np.array(v) for _, v in sorted(groups.items())]
    s = np.array([d[g].sum() for g in idx])
    m = np.array([len(g) for g in idx], dtype=float)
    draws = np.random.default_rng(seed).integers(0, len(idx), size=(n_boot, len(idx)))
    boot = s[draws].sum(axis=1) / m[draws].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean_diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "p_signflip": stats._signflip_p(s, m, n_boot, seed), "n_pairs": int(len(d)), "n_clusters": len(idx)}


def _auc(y, s, clusters, cfg) -> dict:
    return exp32._auc_record(y, s, clusters, cfg)


# ------------------------------------------------------------------------------------------ loading
def load_inputs(raw_root, acts_root, cfg):
    raw_root = pathlib.Path(raw_root)
    files = []
    for sub in cfg["raw_subdirs"]:
        files += sorted(f for f in (raw_root / sub).glob("*.jsonl") if not f.name.endswith(".partial.jsonl"))
    raw_rows, prov = [], []
    for f in files:
        part = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
        raw_rows += part
        prov.append({"file": f"{f.parent.name}/{f.name}", "n_rows": len(part), "sha256": hashlib.sha256(f.read_bytes()).hexdigest()})
    acts = exp32.load_acts(acts_root)
    labels, n_unreplayable = exp32.label_rows(raw_rows)
    del raw_rows
    effort = exp32.select_effort(labels, cfg["effort"])
    labels = [l for l in labels if l["cot_mode"] == "cot_off" or l["effort"] == effort]
    used = {l["arm"] for l in labels if l["cot_mode"] == "cot_on" and l["arm"] in acts}
    wrong = sorted(a for a in used if acts[a]["meta"].get("effort") != effort)
    if wrong:
        raise ValueError(f"activations of {wrong} were captured at a different reasoning effort than the labels")
    return labels, acts, {"raw_files": prov, "n_unreplayable": n_unreplayable, "n_label_rows": len(labels),
                          "effort": effort, "acts_sha256": {a: v["sha256"] for a, v in sorted(acts.items())}}


# ------------------------------------------------------------------------------------------ 1. Regime A incremental
def regime_a_incremental(sub: dict, cfg: dict) -> tuple[dict, dict]:
    y, cl = sub["y"], sub["scenario"]
    folds = exp32.template_folds(sub, cfg["n_folds"])
    cell_oh = exp32.one_hot_blocks([sub["cell"]])
    X = sub["X"]
    w_bal = float(np.sqrt(X.shape[1] / cell_oh.shape[1]))
    fns = {
        "probe": lambda tr, te: exp32.ridge_classifier_scores(X[tr], y[tr], X[te]),
        "design": lambda tr, te: exp32.lookup_scores(sub["cell"][tr], y[tr], sub["cell"][te]),
        "design_probe": lambda tr, te: _ridge_blocks([cell_oh[tr], X[tr]], y[tr], [cell_oh[te], X[te]], [1.0, 1.0]),
    }
    if cfg["balanced_variant"]:
        fns["design_probe_balanced"] = lambda tr, te: _ridge_blocks([cell_oh[tr], X[tr]], y[tr], [cell_oh[te], X[te]], [w_bal, 1.0])
    scores = {m: exp32.oof_scores(sub["n"], folds, f) for m, f in fns.items()}
    pairs = [("design_probe", "design"), ("design_probe", "probe"), ("probe", "design")]
    if cfg["balanced_variant"]:
        pairs += [("design_probe_balanced", "design"), ("design_probe_balanced", "probe")]
    out = {"n": sub["n"], "n_pos": int(y.sum()), "n_scenarios": int(len(set(cl))), "n_cell_columns": int(cell_oh.shape[1]),
           "balanced_cell_weight": w_bal,
           "auroc": {m: _auc(y, s, cl, cfg) for m, s in scores.items()},
           "diff": {f"{a}-{b}": exp32._diff_record(y, scores[a], scores[b], cl, cfg) for a, b in pairs}}
    return out, scores


def regime_b_scores(sub: dict, cfg: dict) -> tuple[dict, dict]:
    tab = exp32.subset(sub, sub["arm"] == "intact")
    y, cl = tab["y"], tab["scenario"]
    folds = exp32.template_folds(tab, cfg["n_folds"])
    scores = exp32._scores_pooled(tab, folds, ("rhat_neg", "rhat_cv", "design"))
    return {"n": tab["n"], "n_pos": int(y.sum()), "auroc": {m: _auc(y, s, cl, cfg) for m, s in scores.items()}}, {**scores, "_y": y}


# ------------------------------------------------------------------------------------------ 2. Regime C additions
def _c_prep(sub: dict, cfg: dict) -> dict:
    """exp32.regime_c's preprocessing: strata, eligible strata, folds over the whole cot subset."""
    tab = {**sub, "stratum": np.char.add(np.char.add(sub["cot_mode"].astype(str), "|"), sub["cell"].astype(str))}
    y = tab["y"]
    counts = {s: (int(((tab["stratum"] == s) & (y == 1)).sum()), int(((tab["stratum"] == s) & (y == 0)).sum()))
              for s in np.unique(tab["stratum"])}
    eligible = sorted(s for s, (p, n) in counts.items() if min(p, n) >= cfg["min_minority"])
    return {"tab": tab, "eligible": eligible, "counts": counts, "folds": exp32.template_folds(tab, cfg["n_folds"]),
            "keep": np.isin(tab["stratum"], eligible)}


def _c_group(prep: dict, idx: np.ndarray, cfg: dict, methods=("probe", "design", "tfidf")) -> dict:
    tab = prep["tab"]
    y = tab["y"]
    scores = exp32._within_scores(tab, idx, y, prep["folds"], methods)
    sid, yy, cl = tab["stratum"][idx], y[idx], tab["scenario"][idx]
    pooled = exp32.stratified_auroc_ci(yy, scores, sid, cl, cfg["n_boot"], cfg["seed"], ref="probe")
    fpr = {m: fpr_at_tpr(yy, s, cfg["tpr_target"]) for m, s in scores.items() if m != "design"}
    return {"n": int(len(idx)), "n_pos": int(yy.sum()), "n_strata": int(len(set(sid))), "n_scenarios": int(len(set(cl))),
            "auroc": pooled, "fpr_at_tpr": fpr}


def regime_c_additions(sub: dict, cfg: dict) -> dict:
    prep = _c_prep(sub, cfg)
    tab, keep = prep["tab"], prep["keep"]
    out = {"n_eligible": len(prep["eligible"]), "arm_cot": [], "harmful_cells": [], "harmful_cot": None}
    if not prep["eligible"]:
        return out
    arm_cot = np.char.add(np.char.add(tab["arm"].astype(str), "|"), tab["cot_mode"].astype(str))
    for g in sorted(set(arm_cot[keep])):
        out["arm_cot"].append({"group": g, **_c_group(prep, np.flatnonzero(keep & (arm_cot == g)), cfg)})
    harm = keep & (tab["cons"] == "harmful")
    for s in sorted(set(tab["stratum"][harm])):
        e = _c_group(prep, np.flatnonzero(harm & (tab["stratum"] == s)), cfg, methods=("probe", "design", "tfidf"))
        out["harmful_cells"].append({"stratum": s, **e})
    if harm.any():
        out["harmful_cot"] = _c_group(prep, np.flatnonzero(harm), cfg)
    return out


# ------------------------------------------------------------------------------------------ 3. projection on Intact
def projection_intact(labels: list, acts: dict, cfg: dict) -> dict:
    a = acts["intact"]
    r = np.asarray(a["r_hat"], np.float64)
    meta = {}
    for l in labels:
        if l["source"] == "exp11" and l["pid"] in a["index"]:
            meta[l["pid"]] = (l["scenario"], l["cost"], l["cons"], l["bnd"])
    pids = sorted(meta)
    rows = np.array([a["index"][p] for p in pids])
    contrasts = {"harmless_explicit-harmless_implicit": (("harmless", "explicit"), ("harmless", "implicit")),
                 "harmful_implicit-harmless_implicit": (("harmful", "implicit"), ("harmless", "implicit"))}
    out = {"n_prompts": len(pids), "sites": {}}
    for site in cfg["projection_sites"]:
        res = {}
        for cot in exp32.COT_MODES:
            H = a[exp32._state_key(site, cot)]
            if H is None:
                res[cot] = {"status": "absent"}
                continue
            H = np.asarray(H[rows], np.float64)
            dot = H @ r
            norm = np.linalg.norm(H, axis=1)
            vals = {"q": np.abs(dot) / norm, "signed_dot": dot, "signed_cos": dot / norm}
            cell = defaultdict(lambda: defaultdict(list))         # (scenario, cost) -> (cons, bnd) -> [prompt rows]
            for k, p in enumerate(pids):
                sc, cost, cons, bnd = meta[p]
                cell[(sc, cost)][(cons, bnd)].append(k)
            groups = {}
            for cons in ("harmful", "harmless"):
                for bnd in ("explicit", "implicit"):
                    ks = [k for v in cell.values() for k in v.get((cons, bnd), [])]
                    groups[f"{cons}_{bnd}"] = {"n": len(ks), **{f"mean_{m}": float(v[ks].mean()) for m, v in vals.items()}}
            cres = {"group_means": groups, "contrasts": {}}
            for name, (ga, gb) in contrasts.items():
                keys = sorted(k for k, v in cell.items() if ga in v and gb in v)
                cres["contrasts"][name] = {}
                for m in ("q", "signed_dot", "signed_cos"):
                    d = [float(vals[m][cell[k][ga]].mean() - vals[m][cell[k][gb]].mean()) for k in keys]
                    cres["contrasts"][name][m] = cluster_mean_ci(d, [k[0] for k in keys], cfg["projection_n_boot"], cfg["seed"])
            cres["max_prompts_per_pair_side"] = int(max(len(x) for v in cell.values() for x in v.values()))
            res[cot] = cres
        out["sites"][site] = res
    return out


# ------------------------------------------------------------------------------------------ receipt function
def recompute(raw_root, acts_root, params: dict | None = None) -> dict:
    cfg = {**DEFAULT_PARAMS, **(params or {})}
    t0 = time.time()
    labels, acts, prov = load_inputs(raw_root, acts_root, cfg)
    res = {"regime_a_incremental": {}, "regime_c_additions": {}, "fpr_at_80tpr": {}, "projection_intact": None}
    with exp32._threads(cfg["threads"]):
        for name, sources in cfg["datasets"].items():
            tab = exp32.build_table(labels, acts, site=cfg["site"], sources=tuple(sources))
            ra, rc, fp = {}, {}, {"regime_a": {}, "regime_b": {}, "regime_c": {}}
            for cot in sorted(set(tab["cot_mode"])):
                sub = exp32.subset(tab, tab["cot_mode"] == cot)
                if name in cfg["regime_a_datasets"]:
                    ra[cot], sc = regime_a_incremental(sub, cfg)
                    fp["regime_a"][cot] = {m: fpr_at_tpr(sub["y"], s, cfg["tpr_target"]) for m, s in sc.items()}
                rb, sb = regime_b_scores(sub, cfg)
                fp["regime_b"][cot] = {m: {**fpr_at_tpr(sb["_y"], s, cfg["tpr_target"]), "auroc": rb["auroc"][m]["auroc"]}
                                       for m, s in sb.items() if m != "_y"}
                rc[cot] = regime_c_additions(sub, cfg)
                fp["regime_c"][cot] = {g["group"]: {m: {**v, "auroc": g["auroc"][m]["auroc"]} for m, v in g["fpr_at_tpr"].items()}
                                       for g in rc[cot]["arm_cot"]}
                if cot in ra:
                    for m, v in fp["regime_a"][cot].items():
                        v["auroc"] = ra[cot]["auroc"][m]["auroc"]
            if ra:
                res["regime_a_incremental"][name] = ra
            res["regime_c_additions"][name] = rc
            res["fpr_at_80tpr"][name] = fp
        res["projection_intact"] = projection_intact(labels, acts, cfg)
    res["params"] = {**{k: v for k, v in cfg.items()}, "provenance": prov, "wall_s": round(time.time() - t0, 1)}
    res["notes"] = [
        "labels, tables, folds, learners and bootstraps are exp32's (same config as records/exp32_probes.json, site 'out')",
        "design_probe: [one-hot(cell), h48] each column standardised on the training fold, RidgeClassifierCV(alphas=exp32.ALPHAS)",
        "design_probe_balanced: as design_probe but the standardised cell block is scaled by sqrt(D/n_cell_columns) "
        "(equal total block variance under the shared ridge penalty)",
        "diff CIs: stats.auroc_diff_bootstrap_ci, paired, resampling scenarios, n_boot as config; 'a-b' = AUROC(a)-AUROC(b)",
        "Regime C: exp32._within_scores (cell-demeaned features incl. TF-IDF, cell-centred training labels, scores centred "
        "per (cell, fold)); AUROC = same-cell stratified concordance with scenario bootstrap (exp32.stratified_auroc_ci)",
        "Regime C harmful_cells = exp32 'cell' scope restricted to eligible harmful strata (source of the report's 0.554 / 0.747); "
        "harmful_cot = all eligible harmful strata of one cot mode trained and scored together",
        "FPR@80%TPR: lowest FPR among thresholds with TPR >= 0.80, rule s >= t, ties move together; out-of-fold scores; "
        "Regime C uses the pooled (cell, fold)-centred scores with ONE threshold across cells (the AUROC there is within-cell)",
        "Regime B: Intact only; rhat_neg = -(h48_out . r_hat) unfitted; rhat_cv sign learned leave-templates-out; design = cell lookup",
        "projection: r_hat = the unit vector stored in the activation files (Run-1 r_hat, fit at the layer-48 input); "
        "site 'out' (layer-48 output) is the paper's Regime A/B projection convention, 'in' is reported as sensitivity",
        "projection pairs: same scenario and cost regime (one prompt per side; tool order differs between cells by design); "
        "CI = scenario cluster bootstrap of the mean pair difference; p = scenario sign-flip (stats._signflip_p)",
        "projection: q = |h.r_hat|/||h|| discards the sign; group means of the signed dot / cosine are reported too, and when "
        "the signed means are negative a SMALLER q means a projection closer to zero, i.e. a shift TOWARD +r_hat",
        "cluster_mean_ci re-implements v3.deconf.cluster_mean locally (deconf.py is not committed)",
    ]
    return stats.to_jsonable(exp32._round(res))


# ------------------------------------------------------------------------------------------ CLI
def _flat(x, prefix=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from _flat(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            key = v.get("group") or v.get("stratum") if isinstance(v, dict) else None
            yield from _flat(v, f"{prefix}[{key if key else i}]")
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield prefix, x


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default=str(ROOT / "raw_transcripts"))
    ap.add_argument("--acts-root", default=str(ROOT / "data" / "exp32" / "main"))
    ap.add_argument("--params", default="{}", help="JSON overrides of DEFAULT_PARAMS")
    ap.add_argument("--json", action="store_true", help="print the whole result as JSON (numbers, ids, notes)")
    a = ap.parse_args(argv)
    res = recompute(a.raw_root, a.acts_root, json.loads(a.params))
    if a.json:
        print(json.dumps(res, indent=1, sort_keys=True))
        return 0
    for k, v in _flat({x: res[x] for x in ("regime_a_incremental", "regime_c_additions", "fpr_at_80tpr", "projection_intact")}):
        print(f"{k} {v}")
    print(f"params.wall_s {res['params']['wall_s']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
