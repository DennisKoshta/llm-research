"""Exp 1.1 thinking / effort robustness on the high-cost harmful-implicit slice -> records/exp11_effort_robustness.json.

  PYTHONPATH=src .venv/bin/python scripts/analyze_exp11_robustness.py [--n-boot 5000] [--out records/exp11_effort_robustness.json]

Slice: family main, cost_regime in {high_tedium, high_friction}, consequence harmful, boundary implicit (40 scenarios x 2 costs
= 80 paired instances per arm and condition; 10 templates of 4 scenarios). Outcome: direct_taken, intention-to-treat (truncated /
malformed / unreplayable episodes stay in n and count as "no shortcut"). Conditions (assigned from the stored cot_mode /
reasoning_effort fields; the source directory is cross-checked):

  default  cot_on, reasoning_effort None (chat-template default)      raw_transcripts/exp11                     six 27B arms
  low      cot_on, reasoning_effort "low" (post-hoc run)              raw_transcripts/exp11_effort_low_a|b|c    six 27B arms
  off      cot_off                                                    raw_transcripts/exp11_cotoff              intact, refusal_ablated

Every number comes from `v3.exp11.replay` (which re-derives routes / direct_taken from the stored raw strings with the strict
`v3.parse.parse_turn` + a fresh Sandbox; stored outcome fields are never trusted) and `v3.stats` (wilson_ci, paired_contrast,
exact_mcnemar, cluster_bootstrap_diff, holm). `recompute(raw_rows_by_dir)` is the pure function that
scripts/verify_all_receipts.py calls. Prints aggregate numbers only; never scenario or model text.
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

from v3 import exp11, stats  # noqa: E402
from v3.arms import ARMS_27B  # noqa: E402
from v3.sandbox import TRUNCATED_OR_MALFORMED  # noqa: E402

SCHEMA = "exp11_effort_robustness/v1"
RECORD = "exp11_effort_robustness"
SOURCES = {"default": ("exp11",),
           "low": ("exp11_effort_low_a", "exp11_effort_low_b", "exp11_effort_low_c"),
           "off": ("exp11_cotoff",)}
SOURCE_DIRS = tuple(d for ds in SOURCES.values() for d in ds)
CONDITIONS = tuple(SOURCES)
CONDITION_ARMS = {"default": tuple(ARMS_27B), "low": tuple(ARMS_27B), "off": tuple(exp11.ARMS_COT_OFF)}
COSTS, CONSEQUENCE, BOUNDARY = exp11.FOCUS["hc_harmful_implicit"]
FAMILY = "main"
EXPECTED_N = 80
ABLATED, INTACT = "refusal_ablated", "intact"
TEXT_NO_CALL = "TEXT_NO_CALL"
# stored instance fields that must be identical for one scenario x cost across every arm and condition (the condition is a generation setting)
INSTANCE_FIELDS = ("trial_id", "order_index", "declared_names", "tool_order", "direct_tool", "safe_tool", "safe_steps", "messages",
                   "historical_log")

# (contrast name, condition A, condition B, arms): the effect is always A - B, per arm, on the same instances.
CONTRASTS = (("low_vs_default", "low", "default", CONDITION_ARMS["low"]),
             ("off_vs_default", "off", "default", CONDITION_ARMS["off"]),
             ("low_vs_off", "low", "off", CONDITION_ARMS["off"]))
# Pre-declared Holm families: one family per contrast type, spanning every arm that has the contrast. Two p-value columns are
# each adjusted within their family: the exact McNemar p and the template-level exact sign-flip p. DiD, P1.1 and the
# leave-one-template-out analysis are single pre-specified estimands outside any family (descriptive).
HOLM_FAMILIES = {name: [f"{name}:{arm}" for arm in arms] for name, _, _, arms in CONTRASTS}

DEFINITIONS = {
    "k_shortcut": "episodes whose direct tool executed (direct_taken, re-derived by v3.exp11.replay from the stored raw strings); n includes truncated / malformed episodes (ITT)",
    "wilson": "95% Wilson score interval for k / n (stats.wilson_ci)",
    "table": "2x2 over matched instances: both_shortcut (A=1,B=1), a_only (A=1,B=0), b_only (A=0,B=1), neither (A=0,B=0)",
    "p_mcnemar": "exact two-sided McNemar (binomial on a_only vs b_only, stats.exact_mcnemar); p_mcnemar_mid is the mid-p",
    "newcombe": "95% Newcombe method-10 interval for rate_a - rate_b (stats.paired_diff_newcombe_ci)",
    "scenario_cluster_bootstrap": "stats.cluster_bootstrap_diff, resampling the 40 scenario clusters (percentile CI, p_boot, p_signflip)",
    "template_cluster_bootstrap": "stats.cluster_bootstrap_diff, resampling the 10 template clusters; p_signflip is the exact template-level sign-flip test (2^10 flips)",
    "p_mcnemar_holm / p_signflip_template_holm": "Holm step-down over the contrast's pre-declared family (holm_families.families)",
    "upper_bound_all_truncated": "k_shortcut plus every TRUNCATED_OR_MALFORMED episode that did not take the direct tool, over n",
    "upper_bound_excl_text_no_call": "as upper_bound_all_truncated but TEXT_NO_CALL replies (a closed turn with no call) stay non-shortcuts",
    "source_rows": "raw rows read from each directory for the cell, before duplicate removal (n counts the de-duplicated trials)",
    "leave_one_template_out": "exact McNemar p for ablated-only vs intact-only discordant pairs after dropping all pairs of one template",
}

NOTES = (
    "Slice: 10 templates, 40 scenarios (4 per template), 2 high-cost regimes -> 80 paired instances per arm and condition; the two cost "
    "regimes of a scenario and the four scenarios of a template are not independent, so exact McNemar p-values are optimistic. The "
    "scenario-cluster bootstrap and the template-level exact sign-flip (10 clusters, smallest possible p = 2/1024) are the conservative checks.",
    "Low effort (exp11_effort_low_a|b|c) is a post-hoc run (not pre-registered; its raw files post-date the default-effort and thinking-off "
    "files); no multiplicity adjustment covers the decision to run it.",
    "Greedy runs do not reproduce across launches: default, low and thinking-off come from separate launches, so every cross-condition "
    "difference contains run-to-run noise in addition to any effect of effort or thinking.",
    "Output caps differ by condition (max_new_tokens 16384 default, 4096 low, 1024 thinking-off, recorded per row); truncation counts are in "
    "`truncation`. Rates are intention-to-treat: truncated / malformed episodes count as no shortcut, so the rates are lower bounds and "
    "`truncation` gives the matching upper bounds.",
    "Pairs are matched on scenario_id x cost_regime (same instance: hygiene.n_pair_keys_with_differing_instance_fields counts keys whose "
    "trial id, tool order, messages or log differ between cells; the effort or thinking condition is a generation setting, not part of the stored messages). Contrasts are A - B as named; thinking-off exists only for intact and refusal_ablated.",
    "The Newcombe interval is asymptotic and can exclude 0 when the exact McNemar p does not (few discordant pairs); the percentile bootstrap "
    "over 10 template clusters is anti-conservative (stats.cluster_bootstrap_diff documents about 8% simulated null rejection at 10 clusters; "
    "its few_clusters flag only fires below 10), so the exact template sign-flip p is the template-level test.",
    "Holm families are fixed in code (HOLM_FAMILIES), one per contrast type and spanning every arm that has the contrast (6, 2, 2 members), "
    "independent of the data; control arms with no discordant pairs (p = 1) still count in m, so the adjusted p-values of the ablated arm "
    "are the unadjusted ones multiplied by the family size (6 for low vs default). Unadjusted p-values are stored next to the adjusted ones.",
    "DiD = (ablated low - ablated default) - (intact low - intact default) on the instances present in all four cells. Its cluster bootstrap "
    "runs through stats.cluster_bootstrap_diff on a stacked 2N encoding (a = [ablated low ; intact default], b = [ablated default ; intact low], "
    "each instance's scenario id repeated for both halves), whose pooled difference equals DiD / 2; the record stores diff and CI rescaled by 2 "
    "(p-values are scale-free). The DiD sign-flip p is a symmetric-null test on the per-instance DiD values.",
)


# =============================================================================================
# loading
# =============================================================================================
def load_raw(raw_root: pathlib.Path) -> dict[str, list[dict]]:
    """{directory name: rows} for the five source directories (.partial.jsonl crash copies are skipped)."""
    out = {}
    for d in SOURCE_DIRS:
        rows = []
        for f in sorted((pathlib.Path(raw_root) / d).glob("*.jsonl")):
            if f.name.endswith(".partial.jsonl"):
                continue
            rows += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
        out[d] = rows
    return out


def provenance(raw_root: pathlib.Path) -> dict:
    out = {}
    for d in SOURCE_DIRS:
        for f in sorted((pathlib.Path(raw_root) / d).glob("*.jsonl")):
            if not f.name.endswith(".partial.jsonl"):
                out[f"raw_transcripts/{d}/{f.name}"] = {
                    "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                    "n_rows": sum(1 for line in f.read_text().splitlines() if line.strip())}
    return out


# =============================================================================================
# conditions and cells
# =============================================================================================
def condition_of(cot_mode, effort_label) -> str | None:
    """Condition from the recorded generation settings (effort label: 'default' = reasoning_effort None)."""
    if cot_mode == "cot_off":
        return "off" if effort_label == "default" else None
    if cot_mode == "cot_on":
        return effort_label if effort_label in ("default", "low") else None
    return None


def _in_slice(r: dict) -> bool:
    return (r.get("family") == FAMILY and r.get("cost_regime") in COSTS and r.get("consequence") == CONSEQUENCE
            and r.get("boundary") == BOUNDARY)


def _index(ts: list[dict], where: str) -> dict:
    out = {exp11.pair_key(t): t for t in ts}
    if len(out) != len(ts):
        raise ValueError(f"duplicate pair keys among the trials of {where}")
    return out


def cell_block(ts: list[dict]) -> dict:
    n = len(ts)
    k = sum(bool(t["direct_taken"]) for t in ts)
    lo, hi = stats.wilson_ci(k, n)
    toks = [t["tokens"] for t in ts]
    return {"n": n, "k_shortcut": k, "rate": k / n if n else None, "wilson": [lo, hi],
            "n_scenarios": len({t["scenario_id"] for t in ts}), "n_templates": len({t["template_id"] for t in ts}),
            "complete": n == EXPECTED_N,
            "total_generated_tokens": {"mean": sum(toks) / n if n else None, "max": max(toks) if toks else None}}


def truncation_block(ts: list[dict]) -> dict:
    """TRUNCATED_OR_MALFORMED accounting. Under ITT those episodes are not shortcuts; the upper bound counts every one of them as a
    shortcut, and the second bound keeps text-only replies (TEXT_NO_CALL, a deliberate no-call that can be a refusal) as non-shortcuts."""
    n = len(ts)
    k = sum(bool(t["direct_taken"]) for t in ts)
    trunc = [t for t in ts if t["route"] == TRUNCATED_OR_MALFORMED]
    n_trunc = len(trunc)
    n_trunc_not_taken = sum(not t["direct_taken"] for t in trunc)
    n_text = sum(t["sub_status"] == TEXT_NO_CALL and not t["direct_taken"] for t in trunc)
    k_all, k_trunc_only = k + n_trunc_not_taken, k + n_trunc_not_taken - n_text

    def bound(kk):
        lo, hi = stats.wilson_ci(kk, n)
        return {"k": kk, "rate": kk / n if n else None, "wilson": [lo, hi]}

    return {"n": n, "k_shortcut_itt": k, "rate_itt": k / n if n else None,
            "n_truncated_or_malformed": n_trunc, "rate_truncated_or_malformed": n_trunc / n if n else None,
            "sub_status": dict(sorted(Counter(t["sub_status"] for t in trunc).items())),
            "n_truncated_and_direct_taken": n_trunc - n_trunc_not_taken,
            "upper_bound_all_truncated": bound(k_all),
            "upper_bound_excl_text_no_call": bound(k_trunc_only)}


# =============================================================================================
# paired machinery
# =============================================================================================
def _vectors(cell_a: dict, cell_b: dict):
    keys = sorted(cell_a.keys() & cell_b.keys())
    av = [int(cell_a[k]["direct_taken"]) for k in keys]
    bv = [int(cell_b[k]["direct_taken"]) for k in keys]
    scen = [cell_a[k]["scenario_id"] for k in keys]
    tmpl = [cell_a[k]["template_id"] for k in keys]
    return keys, av, bv, scen, tmpl, len(cell_a) + len(cell_b) - 2 * len(keys)


def paired_block(cell_a: dict, cell_b: dict, name_a: str, name_b: str, n_boot: int) -> dict:
    """A - B on the instances present in both cells: rates, 2x2 table, exact McNemar, Newcombe, scenario-cluster bootstrap
    (CI + p), template-cluster bootstrap with the exact sign-flip p."""
    keys, av, bv, scen, tmpl, n_unmatched = _vectors(cell_a, cell_b)
    pc = stats.paired_contrast(av, bv, ids=keys, cluster_ids=tmpl, n_boot=n_boot)
    return {
        "a": name_a, "b": name_b, "orientation": "a - b", "n_pairs": pc["n_pairs"], "n_unmatched": n_unmatched,
        "k_a": int(sum(av)), "k_b": int(sum(bv)), "rate_a": pc["rate_a"], "rate_b": pc["rate_b"],
        "wilson_a": pc["wilson_a"], "wilson_b": pc["wilson_b"], "diff": pc["diff"],
        "table": {"both_shortcut": pc["n_both"], "a_only": pc["n_a_only"], "b_only": pc["n_b_only"], "neither": pc["n_neither"]},
        "n_discordant": pc["n_discordant"],
        "p_mcnemar": pc["p_mcnemar"], "p_mcnemar_mid": pc["p_mcnemar_mid"],
        "newcombe": [pc["ci_low"], pc["ci_high"]],
        "scenario_cluster_bootstrap": stats.cluster_bootstrap_diff(av, bv, scen, n_boot),
        "template_cluster_bootstrap": pc["cluster_boot"],
    }


def holm_into(contrasts: dict, names: list[str]) -> None:
    p = stats.holm({n: contrasts[n]["p_mcnemar"] for n in names})
    sf = stats.holm({n: contrasts[n]["template_cluster_bootstrap"]["p_signflip"] for n in names})
    for n in names:
        contrasts[n]["p_mcnemar_holm"] = p[n]
        contrasts[n]["p_signflip_template_holm"] = sf[n]


def did_block(cells: dict, n_boot: int) -> dict:
    """(ablated low - ablated default) - (intact low - intact default) on the instances present in all four cells."""
    quad = {"al": cells[ABLATED, "low"], "ad": cells[ABLATED, "default"], "il": cells[INTACT, "low"], "id": cells[INTACT, "default"]}
    keys = sorted(set.intersection(*(set(c) for c in quad.values())))
    v = {nm: [int(c[k]["direct_taken"]) for k in keys] for nm, c in quad.items()}
    scen = [quad["ad"][k]["scenario_id"] for k in keys]
    tmpl = [quad["ad"][k]["template_id"] for k in keys]
    n = len(keys)
    d = [(v["al"][i] - v["ad"][i]) - (v["il"][i] - v["id"][i]) for i in range(n)]
    a_eff = sum(v["al"][i] - v["ad"][i] for i in range(n)) / n
    i_eff = sum(v["il"][i] - v["id"][i] for i in range(n)) / n
    a_stack, b_stack = v["al"] + v["id"], v["ad"] + v["il"]     # pooled (a - b) / 2N = DiD / 2

    def boot(clusters):
        b = stats.cluster_bootstrap_diff(a_stack, b_stack, clusters + clusters, n_boot)
        return {"diff": 2 * b["diff"], "ci_low": 2 * b["ci_low"], "ci_high": 2 * b["ci_high"], "p_boot": b["p_boot"],
                "p_signflip": b["p_signflip"], "signflip_exact": b["signflip_exact"], "n_clusters": b["n_clusters"],
                "n_boot": b["n_boot"], "seed": b["seed"], "few_clusters": b["few_clusters"]}

    return {"definition": "(refusal_ablated low - refusal_ablated default) - (intact low - intact default)", "n_pairs": n,
            "rates": {"ablated_low": sum(v["al"]) / n, "ablated_default": sum(v["ad"]) / n,
                      "intact_low": sum(v["il"]) / n, "intact_default": sum(v["id"]) / n},
            "k": {"ablated_low": sum(v["al"]), "ablated_default": sum(v["ad"]), "intact_low": sum(v["il"]), "intact_default": sum(v["id"])},
            "ablated_effort_effect": a_eff, "intact_effort_effect": i_eff, "did": sum(d) / n,
            "per_instance_did_distribution": {str(x): sum(y == x for y in d) for x in (-2, -1, 0, 1, 2)},
            "encoding": "stats.cluster_bootstrap_diff on [ablated low ; intact default] vs [ablated default ; intact low], rescaled by 2",
            "scenario_cluster_bootstrap": boot(scen), "template_cluster_bootstrap": boot(tmpl)}


def p11_block(abl: dict, intact: dict, n_boot: int) -> dict:
    """P1.1 (ablated - intact, cot_on) with the template-level robustness: per-template counts, leave-one-template-out exact
    McNemar, template-level exact sign-flip p, and how many templates / scenarios carry the discordant pairs."""
    full = paired_block(abl, intact, ABLATED, INTACT, n_boot)
    keys, av, bv, scen, tmpl, _ = _vectors(abl, intact)
    templates = sorted(set(tmpl))
    per_template = {}
    for t in templates:
        idx = [i for i in range(len(keys)) if tmpl[i] == t]
        a_only = sum(av[i] == 1 and bv[i] == 0 for i in idx)
        b_only = sum(av[i] == 0 and bv[i] == 1 for i in idx)
        both = sum(av[i] == 1 and bv[i] == 1 for i in idx)
        per_template[t] = {"n_pairs": len(idx), "n_scenarios": len({scen[i] for i in idx}), "k_ablated": sum(av[i] for i in idx),
                           "k_intact": sum(bv[i] for i in idx), "both_shortcut": both, "ablated_only": a_only, "intact_only": b_only,
                           "neither": len(idx) - both - a_only - b_only, "n_discordant": a_only + b_only}
    loto = {}
    for t in templates:
        b = sum(pt["ablated_only"] for tt, pt in per_template.items() if tt != t)
        c = sum(pt["intact_only"] for tt, pt in per_template.items() if tt != t)
        m = stats.exact_mcnemar(b, c)
        loto[t] = {"n_pairs": len(keys) - per_template[t]["n_pairs"], "ablated_only": b, "intact_only": c, "p_mcnemar": m["p"]}
    ps = {t: x["p_mcnemar"] for t, x in loto.items()}
    scen_cnt: dict = {}
    for i, s in enumerate(scen):
        e = scen_cnt.setdefault(s, [0, 0])
        e[0] += av[i] == 1 and bv[i] == 0
        e[1] += av[i] == 0 and bv[i] == 1
    pts = per_template.values()
    return {
        "contrast": full,
        "template_signflip_p": full["template_cluster_bootstrap"]["p_signflip"],
        "template_signflip_exact": full["template_cluster_bootstrap"]["signflip_exact"],
        "n_templates": len(templates), "n_scenarios": len(scen_cnt),
        "n_templates_with_discordant": sum(p["n_discordant"] > 0 for p in pts),
        "n_templates_with_ablated_only": sum(p["ablated_only"] > 0 for p in pts),
        "n_templates_with_intact_only": sum(p["intact_only"] > 0 for p in pts),
        "n_scenarios_with_discordant": sum(a + b > 0 for a, b in scen_cnt.values()),
        "n_scenarios_with_ablated_only": sum(a > 0 for a, b in scen_cnt.values()),
        "n_scenarios_with_intact_only": sum(b > 0 for a, b in scen_cnt.values()),
        "per_template": per_template, "leave_one_template_out": loto,
        "loto_p_min": min(ps.values()), "loto_p_max": max(ps.values()),
        "loto_template_at_p_max": max(sorted(ps), key=lambda t: ps[t]),
        "loto_n_p_ge_0_05": sum(p >= 0.05 for p in ps.values()),
    }


# =============================================================================================
# the pure recompute
# =============================================================================================
def recompute(raw_rows_by_dir, *, n_boot: int = 5000) -> dict:
    """JSON-ready record recomputed from raw rows only. `raw_rows_by_dir` maps a raw_transcripts directory name to its rows; the
    five source directories must be present (other keys are ignored). Deterministic: fixed seeds, no I/O."""
    missing = [d for d in SOURCE_DIRS if d not in raw_rows_by_dir]
    if missing:
        raise ValueError(f"source directories missing from the input: {missing}")
    expect_dir = {d: c for c, ds in SOURCES.items() for d in ds}
    slice_rows: list[dict] = []
    acct: Counter = Counter()
    rows_read: dict = {}
    rows_in_slice: dict = {}
    sources: dict = {}
    dir_mismatch: Counter = Counter()
    instance_digests: dict = {}
    caps: dict = {}
    for d in SOURCE_DIRS:
        rows_read[d] = len(raw_rows_by_dir[d])
        rows_in_slice[d] = 0
        for r in raw_rows_by_dir[d]:
            if not _in_slice(r):
                acct["n_rows_outside_slice"] += 1
                continue
            slice_rows.append(r)
            rows_in_slice[d] += 1
            cond = condition_of(r.get("cot_mode"), exp11.effort_label(r.get("reasoning_effort")))
            sources.setdefault(str(r.get("arm")), {}).setdefault(str(cond), Counter())[d] += 1
            caps.setdefault(str(cond), set()).add(r.get("max_new_tokens"))
            digest = hashlib.sha256(json.dumps([r.get(f) for f in INSTANCE_FIELDS], sort_keys=True).encode()).hexdigest()
            instance_digests.setdefault(f"{r.get('scenario_id')}|{r.get('cost_regime')}", set()).add(digest)
            if cond != expect_dir[d]:
                dir_mismatch[f"{d}->{cond}"] += 1
    trials, h = exp11.replay(slice_rows)
    cells: dict = {}
    excluded: Counter = Counter()
    for t in trials:
        cond = condition_of(t["cot_mode"], t["effort"])
        if cond is None or t["arm"] not in CONDITION_ARMS[cond] or t["family"] != FAMILY:
            excluded[f"{t['arm']}|{t['cot_mode']}|{t['effort']}"] += 1
            continue
        cells.setdefault((t["arm"], cond), []).append(t)
    need = [(a, c) for c in CONDITIONS for a in CONDITION_ARMS[c]]
    empty = [f"{a}|{c}" for a, c in need if not cells.get((a, c))]
    if empty:
        raise ValueError(f"no trials for required cells: {empty}")
    idx = {ac: _index(cells[ac], f"{ac[0]}|{ac[1]}") for ac in need}

    cell_out: dict = {}
    trunc_out: dict = {}
    for arm in ARMS_27B:
        for cond in CONDITIONS:
            if (arm, cond) in idx:
                ts = cells[arm, cond]
                cell_out.setdefault(arm, {})[cond] = {**cell_block(ts), "source_rows": {d: n for d, n in sorted(sources[arm][cond].items())}}
                trunc_out.setdefault(arm, {})[cond] = truncation_block(ts)

    contrasts: dict = {}
    for name, ca, cb, arms in CONTRASTS:
        for arm in arms:
            contrasts[f"{name}:{arm}"] = paired_block(idx[arm, ca], idx[arm, cb], f"{arm}|{ca}", f"{arm}|{cb}", n_boot)
    for names in HOLM_FAMILIES.values():
        holm_into(contrasts, names)

    templates_of_key: dict = {}
    for ac in need:
        for k, t in idx[ac].items():
            templates_of_key.setdefault(k, set()).add(t["template_id"])
    template_mismatch = sum(len(v) > 1 for v in templates_of_key.values())
    complete = all(len(idx[ac]) == EXPECTED_N for ac in need)
    same_keys = len({frozenset(idx[ac]) for ac in need}) == 1
    hygiene = {
        "rows_read": rows_read, "rows_in_slice": rows_in_slice, "n_rows_outside_slice": acct["n_rows_outside_slice"],
        "n_trials": len(trials), "n_trials_excluded": sum(excluded.values()), "excluded": dict(sorted(excluded.items())),
        "dir_condition_mismatch": dict(sorted(dir_mismatch.items())),
        "n_pair_keys_with_differing_instance_fields": sum(len(v) > 1 for v in instance_digests.values()),
        "max_new_tokens_by_condition": {c: sorted(v, key=str) for c, v in sorted(caps.items())},
        "all_cells_complete": complete, "all_cells_same_instances": same_keys, "n_template_id_mismatch_across_cells": template_mismatch,
        "n_scenarios": len({t["scenario_id"] for ts in cells.values() for t in ts}),
        "n_templates": len({t["template_id"] for ts in cells.values() for t in ts}),
        "replay": {k: h[k] for k in ("n_rows", "n_unplaceable", "n_duplicate_rows", "n_invalid_records", "n_stored_mismatch_rows",
                                     "n_i1_violations", "n_banned_literal_rows", "n_prompt_suffix_violations",
                                     "n_prompt_mismatch_rows", "n_boundary_label_mismatch_rows")},
        "stored_mismatch_fields": h["stored_mismatch_fields"],
    }
    out = {
        "schema": SCHEMA, "experiment": "1.1 thinking / reasoning-effort robustness",
        "slice": {"family": FAMILY, "cost_regimes": list(COSTS), "consequence": CONSEQUENCE, "boundary": BOUNDARY,
                  "outcome": "direct_taken (intention-to-treat)", "pair_key": "scenario_id|cost_regime|consequence|boundary",
                  "n_instances_per_cell": EXPECTED_N},
        "params": {"n_boot": n_boot, "seed": stats.DEFAULT_SEED, "alpha": 0.05, "source_dirs": list(SOURCE_DIRS)},
        "conditions": {
            "default": {"cot_mode": "cot_on", "reasoning_effort": None, "note": "chat-template default effort (xhigh per PLAN.md)",
                        "dirs": list(SOURCES["default"]), "arms": list(CONDITION_ARMS["default"])},
            "low": {"cot_mode": "cot_on", "reasoning_effort": "low", "note": "post-hoc run", "dirs": list(SOURCES["low"]),
                    "arms": list(CONDITION_ARMS["low"])},
            "off": {"cot_mode": "cot_off", "reasoning_effort": None, "dirs": list(SOURCES["off"]), "arms": list(CONDITION_ARMS["off"])}},
        "holm_families": {"correction": "holm", "columns": ["p_mcnemar", "template_cluster_bootstrap.p_signflip"],
                          "families": HOLM_FAMILIES,
                          "outside_any_family": ["did", "p11_template_robustness.default", "p11_template_robustness.low"]},
        "definitions": DEFINITIONS,
        "hygiene": hygiene,
        "cells": cell_out,
        "contrasts": contrasts,
        "did": did_block({ac: idx[ac] for ac in idx}, n_boot),
        "p11_template_robustness": {c: p11_block(idx[ABLATED, c], idx[INTACT, c], n_boot) for c in ("default", "low")},
        "truncation": trunc_out,
        "notes": list(NOTES),
    }
    return stats.to_jsonable(out)


# =============================================================================================
# CLI
# =============================================================================================
def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def report(rec: dict) -> None:
    h = rec["hygiene"]
    print(f"[hygiene] complete={h['all_cells_complete']} same_instances={h['all_cells_same_instances']} scenarios={h['n_scenarios']} "
          f"templates={h['n_templates']} excluded={h['n_trials_excluded']} dir_mismatch={h['dir_condition_mismatch']} replay={h['replay']}")
    for arm, by in rec["cells"].items():
        print(f"[cells] {arm}: " + " | ".join(f"{c} {x['k_shortcut']}/{x['n']} ({_pct(x['rate'])}, wilson {_pct(x['wilson'][0])}-{_pct(x['wilson'][1])})"
                                              for c, x in by.items()))
    for name, c in rec["contrasts"].items():
        t = c["table"]
        print(f"[contrast] {name}: diff={c['diff']:+.4f} newcombe=[{c['newcombe'][0]:+.3f},{c['newcombe'][1]:+.3f}] "
              f"table(a_only={t['a_only']}, b_only={t['b_only']}, both={t['both_shortcut']}, neither={t['neither']}) p={c['p_mcnemar']:.3g} "
              f"holm={c['p_mcnemar_holm']:.3g} scen_ci=[{c['scenario_cluster_bootstrap']['ci_low']:+.3f},{c['scenario_cluster_bootstrap']['ci_high']:+.3f}] "
              f"tmpl_signflip={c['template_cluster_bootstrap']['p_signflip']:.3g} holm={c['p_signflip_template_holm']:.3g}")
    d = rec["did"]
    print(f"[did] {d['did']:+.4f} (ablated effort effect {d['ablated_effort_effect']:+.4f}, intact {d['intact_effort_effect']:+.4f}) "
          f"scenario CI [{d['scenario_cluster_bootstrap']['ci_low']:+.4f},{d['scenario_cluster_bootstrap']['ci_high']:+.4f}] "
          f"p_signflip={d['scenario_cluster_bootstrap']['p_signflip']:.3g}; template CI "
          f"[{d['template_cluster_bootstrap']['ci_low']:+.4f},{d['template_cluster_bootstrap']['ci_high']:+.4f}] "
          f"p_signflip={d['template_cluster_bootstrap']['p_signflip']:.3g}")
    for c, p in rec["p11_template_robustness"].items():
        k = p["contrast"]
        print(f"[P1.1 {c}] ablated {k['k_a']}/{k['n_pairs']} intact {k['k_b']}/{k['n_pairs']} a_only={k['table']['a_only']} b_only={k['table']['b_only']} "
              f"p={k['p_mcnemar']:.3g} template_signflip={p['template_signflip_p']:.3g} LOTO p in [{p['loto_p_min']:.3g},{p['loto_p_max']:.3g}] "
              f"(max at {p['loto_template_at_p_max']}, {p['loto_n_p_ge_0_05']} templates with p>=0.05) discordant templates "
              f"{p['n_templates_with_discordant']}/{p['n_templates']} scenarios {p['n_scenarios_with_discordant']}/{p['n_scenarios']}")
    for arm, by in rec["truncation"].items():
        print(f"[trunc] {arm}: " + " | ".join(
            f"{c} {x['n_truncated_or_malformed']}/{x['n']} {x['sub_status']} upper={x['upper_bound_all_truncated']['k']}/{x['n']}"
            for c, x in by.items()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", default=str(ROOT / "raw_transcripts"))
    ap.add_argument("--out", default=str(ROOT / "records" / f"{RECORD}.json"))
    ap.add_argument("--n-boot", type=int, default=5000)
    a = ap.parse_args(argv)
    raw_root = pathlib.Path(a.raw_root)
    rec = recompute(load_raw(raw_root), n_boot=a.n_boot)
    rec["provenance"] = provenance(raw_root)
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, allow_nan=False))
    print(f"[written] {out}")
    report(rec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
