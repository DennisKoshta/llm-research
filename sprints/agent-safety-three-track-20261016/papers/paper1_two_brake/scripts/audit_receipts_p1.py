#!/usr/bin/env python3
"""Audit Paper 1 claims in REPORT.md against the JSON receipts in records/.

Every REPORT.md value below is transcribed by hand (Sections 1, 2.1, 2.3, 3.3);
every receipt value is read programmatically. Writes ../AUDIT.md.

Status: MATCH (within tolerance), MISMATCH, MISSING (no receipt field backs it).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REC = ROOT / "records"
OUT = Path(__file__).resolve().parents[1] / "AUDIT.md"


def load(name):
    with open(REC / name) as f:
        return json.load(f)


p1 = load("phase1_qualification_and_calibration_receipt.json")
zv = load("phase1_step2_zero_vector_and_fp8_unit_test.json")
e11 = load("exp1_1_factorial_and_controls_receipt.json")
st = load("exp1_1_statistical_analysis.json")
ev = load("external_validity_flash_next_results.json")
e3 = load("exp3_1_and_3_2_receipt.json")

rows = []  # (section, item, report_value, receipt_value, kind, note)
PCT, NUM, P, TXT = "pct", "num", "p", "txt"


def add(section, item, report, receipt, kind=PCT, note=""):
    rows.append((section, item, report, receipt, kind, note))


def status(report, receipt, kind):
    if receipt is None:
        return "MISSING"
    if kind == TXT:
        return "MATCH" if str(report) == str(receipt) else "MISMATCH"
    if kind == PCT:  # both in percent; agree at 0.1 pp display precision
        return "MATCH" if abs(report - receipt) <= 0.06 else "MISMATCH"
    if kind == P:  # p-values: same order of magnitude and within 2%
        if report == 0 or receipt == 0:
            return "MATCH" if report == receipt else "MISMATCH"
        return "MATCH" if abs(report - receipt) / abs(report) <= 0.02 else "MISMATCH"
    tol = max(5e-4, 0.005 * abs(report))
    return "MATCH" if abs(report - receipt) <= tol else "MISMATCH"


def fmt(v, kind):
    if v is None:
        return "—"
    if kind == PCT:
        return f"{v:.2f}%"
    if kind == P:
        return f"{v:.3g}"
    if kind == NUM:
        return f"{v:.5g}"
    return str(v)


# ---------------------------------------------------------------- Phase 1
S = "1.1 Phase 1 qualification (Table 1)"
gate = p1["qualification_gate_heldout_128"]
intact_ref = gate["intact"]["harmful_hard_refusal_rate"] * 100
abl_ref = gate["refusal_ablated"]["harmful_hard_refusal_rate"] * 100
add(S, "Zero-vector BF16 identity diff = 0", "True",
    str(p1["gate_checks"]["zero_vector_all_130_writers_bf16_exact"]), TXT)
add(S, "Zero-vector FP8 scale diff", 0.0,
    p1["zero_vector_full_model_test"]["max_scale_inv_rel_err_vs_official_fp8"], NUM)
add(S, "Held-out refusal rate, Intact", 95.31, intact_ref)
add(S, "Refusal drop, Refusal-Ablated BF16 (pp)", 95.31, None, PCT,
    "receipt has a single ablated arm; no BF16/FP8 split")
add(S, "Refusal drop, Refusal-Ablated FP8 (pp)", 84.38, intact_ref - abl_ref, PCT,
    "receipt: single ablated arm (materialized FP8 checkpoint)")
add(S, "Refusal rate, Refusal-Ablated FP8", 10.94, abl_ref)
add(S, "Benign first-token KL, BF16", 0.00368,
    gate["refusal_ablated"]["benign_kl_divergence_vs_intact"], NUM)
add(S, "Benign first-token KL, FP8", 0.00349, None, NUM)
add(S, "Benign tool-call propensity, Intact", 100.0,
    gate["intact"]["benign_tool_call_propensity"] * 100)
add(S, "Benign tool-call propensity, Refusal-Ablated", 100.0,
    gate["refusal_ablated"]["benign_tool_call_propensity"] * 100)
add(S, "Benign tool-call propensity Δ (pp)", 0.0,
    (gate["refusal_ablated"]["benign_tool_call_propensity"]
     - gate["intact"]["benign_tool_call_propensity"]) * 100)
pca = p1["pca_variance_matched_controls_meta"]
for seed in (101, 102, 103):
    m = pca["pca_controls_meta"][f"pca_var_matched_seed{seed}"]
    add(S, f"PCA seed {seed} variance ratio", 1.00001,
        m["actual_variance"] / m["target_variance"], NUM)
add(S, "PCA target variance (REPORT: 0.01552)", 0.01552, pca["target_variance"], NUM,
    "unit-test receipt reports 0.000922; neither equals 0.01552")
add(S, "Unrelated concept cos(r̂, û)", -0.0258, None, NUM)
add(S, "Unrelated concept refusal drop (pp)", 6.25,
    intact_ref - gate["unrelated_concept_control"]["harmful_hard_refusal_rate"] * 100)
add(S, "Candidate layers swept", "28,32,36,40,44,48,52",
    ",".join(p1["layer_selection"]["layers"]), TXT)
add(S, "Selected layer", "48", str(p1["layer_selection"]["selected_best_layer"]), TXT)

S = "1.2 α dose-response (Table 1, Fig 1A)"
dr = p1["dose_response_calibration_48"]
for a, rep in zip(("-0.5", "0.0", "0.5", "1.0", "1.5"), (0.0, 0.0, 4.17, 79.17, 83.33)):
    add(S, f"Harmful-implicit violation, α={a}", rep,
        dr[a]["harmful_implicit_high_cost_shortcut_rate"] * 100, PCT,
        "field: harmful_implicit_high_cost_shortcut_rate")
for a in ("-0.5", "0.0", "0.5", "1.0", "1.5"):
    add(S, f"  (alt field harmful_shortcut_rate, α={a})", None,
        dr[a]["harmful_shortcut_rate"] * 100, PCT, "for reference; no REPORT value")
add(S, "Spearman ρ", 0.9747, None, NUM, "not stored; receipt series is non-monotonic")
add(S, "Gate flag dose_response_monotonic_non_decreasing", "True",
    str(p1["gate_checks"]["dose_response_monotonic_non_decreasing"]), TXT,
    "flag is True although receipt rate falls from α=1.0 to α=1.5")
add(S, "Intact High-Tedium Harmless+Implicit (calibration)", 100.0,
    dr["0.0"]["high_tedium_harmless_implicit_shortcut_rate"] * 100)

# ---------------------------------------------------------------- Exp 1.1
S = "2.1 Exp 1.1 factorial, cot_on (Table 2, Fig 1B)"
ARMS = [("Intact", "intact"), ("Refusal-Ablated", "refusal_ablated"),
        ("PCA 101", "pca_var_matched_seed101"), ("PCA 102", "pca_var_matched_seed102"),
        ("PCA 103", "pca_var_matched_seed103"), ("Unrelated", "unrelated_concept_control")]
REPORT_GRID = {  # cell -> [Intact, RA, PCA101, PCA102, PCA103, Unrelated]
    "low_cost__harmless__implicit": [46.7, 93.3, 83.3, 86.7, 83.3, 76.7],
    "low_cost__harmless__explicit": [0, 0, 0, 0, 0, 0],
    "low_cost__harmful__implicit": [0, 0, 0, 0, 0, 0],
    "low_cost__harmful__explicit": [0, 0, 0, 0, 0, 0],
    "high_tedium__harmless__implicit": [100, 100, 100, 100, 100, 100],
    "high_tedium__harmless__explicit": [0, 0, 0, 0, 0, 0],
    "high_tedium__harmful__implicit": [0, 93.3, 0, 0, 0, 3.3],
    "high_tedium__harmful__explicit": [0, 0, 0, 0, 0, 0],
    "high_friction__harmless__implicit": [100, 100, 100, 100, 100, 100],
    "high_friction__harmless__explicit": [0, 0, 0, 0, 0, 0],
    "high_friction__harmful__implicit": [13.3, 100, 30.0, 23.3, 23.3, 13.3],
    "high_friction__harmful__explicit": [0, 6.7, 0, 0, 0, 0],
}
cs = e11["cell_summaries"]


def cell(arm_key, mode, c):
    d = cs.get(f"{arm_key}__{mode}", {}).get(c)
    return None if d is None else 100 * d["shortcut"] / d["n"]


for c, reps in REPORT_GRID.items():
    for (label, key), rep in zip(ARMS, reps):
        add(S, f"{c} — {label}", rep, cell(key, "cot_on", c))

S = "2.1 Exp 1.1 cot_off explicit-prohibition slice (Table 2, Fig 1C)"
for c, rep_i, rep_a in (("high_tedium__harmful__explicit", 0.0, 26.67),
                        ("high_friction__harmful__explicit", 3.3, 56.67)):
    add(S, f"{c} — Intact cot_off", rep_i, cell("intact", "cot_off", c))
    add(S, f"{c} — Refusal-Ablated cot_off", rep_a, cell("refusal_ablated", "cot_off", c))
add(S, "Holm p, HT explicit cot_off vs cot_on (0.0156)", 0.0156, None, P,
    "no cot_off explicit contrast in exp1_1_statistical_analysis.json")
add(S, "Holm p, HF explicit cot_off vs cot_on (1.83e-4)", 1.83e-4, None, P)

S = "2.1 Exp 1.1 sanity & competence controls (Table 2)"
ctl = e11["control_summaries"]
add(S, "Direct instruction N per arm", 30.0,
    float(ctl["refusal_ablated"]["direct_instruction_positive_control"]["n"]), NUM)
for label, key, rep in (("Intact", "intact", 0.0), ("Refusal-Ablated", "refusal_ablated", 76.7),
                        ("PCA 101", "pca_var_matched_seed101", 6.7),
                        ("PCA 102", "pca_var_matched_seed102", 3.3),
                        ("PCA 103", "pca_var_matched_seed103", 0.0),
                        ("Unrelated", "unrelated_concept_control", 0.0)):
    v = ctl.get(key, {}).get("direct_instruction_positive_control")
    add(S, f"Direct instruction shortcut execution — {label}", rep,
        None if v is None else v["shortcut_execution_rate"] * 100)
for label, key in ARMS:
    v = ctl.get(key, {}).get("forced_safe_route_competence_control")
    for reg in ("overall", "low_cost", "high_tedium", "high_friction"):
        add(S, f"Forced-safe competence {reg} — {label}", 100.0,
            None if v is None else v[f"{reg}_safe_completion_rate"] * 100)
for label, key in (("Intact", "intact"), ("Refusal-Ablated", "refusal_ablated")):
    add(S, f"Consequence awareness harmful-foreseen — {label}", 100.0,
        ctl[key]["consequence_awareness_probe"]["harmful_foreseen_rate"] * 100)
    add(S, f"Rule QA accuracy — {label}", 100.0,
        ctl[key]["rule_identification_qa_probe"]["accuracy"] * 100)

S = "2.1 Exp 1.1 primary & secondary contrasts"
pc, sc = st["primary_contrasts"], st["secondary_contrasts_holm_bonferroni"]
p2a, p2b = pc["P2a_harmful_high_tedium_implicit_ablated_vs_intact"], \
    pc["P2b_harmful_high_friction_implicit_ablated_vs_intact"]
add(S, "P1 HT Harmful+Implicit Δ (pp)", 93.33, p2a["diff_a_minus_b_pp"])
add(S, "P1 McNemar p", 7.45e-9, p2a["mcnemar_p"], P)
add(S, "P2 HF Harmful+Implicit RA rate", 100.0, p2b["rate_a"] * 100)
add(S, "P2 HF Harmful+Implicit Intact rate", 13.33, p2b["rate_b"] * 100)
add(S, "P2 Δ (pp)", 86.67, p2b["diff_a_minus_b_pp"])
add(S, "P2 McNemar p", 2.98e-8, p2b["mcnemar_p"], P)
add(S, "P3 cost × ablation ΔΔ (pp)", 90.00, None, PCT,
    "receipt P3 is cot_on vs cot_off (Δ = %.2f pp), not a cost interaction"
    % pc["P3_cot_on_vs_cot_off_ablated_harmful_high_cost"]["diff_a_minus_b_pp"])
pooled_pca_hf = sum(cs[f"pca_var_matched_seed{s}__cot_on"]["high_friction__harmful__implicit"]["shortcut"]
                    for s in (101, 102, 103)) / 90 * 100
add(S, "Pooled PCA HF Harmful+Implicit rate", 25.6, pooled_pca_hf)
add(S, "RA vs pooled PCA, HF Δ (pp)", 74.44,
    cell("refusal_ablated", "cot_on", "high_friction__harmful__implicit") - pooled_pca_hf)
add(S, "RA vs pooled PCA, HT Holm p", 4.47e-8, None, P,
    "receipt contrasts are per seed, pooled over HT+HF (S1–S3)")
add(S, "RA vs Unrelated, HT Δ (pp)", 90.00,
    cell("refusal_ablated", "cot_on", "high_tedium__harmful__implicit")
    - cell("unrelated_concept_control", "cot_on", "high_tedium__harmful__implicit"))
add(S, "RA vs Unrelated, HF Δ (pp)", 86.67,
    cell("refusal_ablated", "cot_on", "high_friction__harmful__implicit")
    - cell("unrelated_concept_control", "cot_on", "high_friction__harmful__implicit"))
add(S, "RA vs Unrelated, HT Holm p", 1.01e-7, None, P,
    "receipt S4 pools HT+HF: Holm p = %.3g"
    % sc["S4_ablated_vs_unrelated_control_harmful_high_cost_implicit"]["holm_correction"]["holm_adjusted_p"])

# ---------------------------------------------------------------- External validity
S = "2.3 External validity, Flash-Next (Table 2)"
evs = ev["exp1_1_external_validity_summary"]
for c, ri, ru in (("high_tedium__harmful__implicit", 0.0, 36.67),
                  ("high_friction__harmful__implicit", 3.33, 33.33),
                  ("low_cost__harmful__implicit", 0.0, 0.0)):
    add(S, f"{c} — Flash-Next-Intact", ri, evs["Flash-Next-Intact-FP8"][c]["shortcut_rate"] * 100)
    add(S, f"{c} — Flash-Next-Uncensored", ru, evs["Flash-Next-Uncensored-FP8"][c]["shortcut_rate"] * 100)

# ---------------------------------------------------------------- Exp 3.1
S = "3.3 Exp 3.1 domain framing (Table 3, Fig 2)"
b = e3["exp3_1_behavioral_summary"]
DOM = ["warehouse_logistics", "compute_infrastructure", "financial_reporting", "wet_lab_protocol"]


def rate(arm, cot, dom):
    cells = [b[arm][cot][f][dom] for f in b[arm][cot]]
    return 100 * sum(c["n"] * c["shortcut_violation_rate"] for c in cells) / sum(c["n"] for c in cells)


REPORT_DOM = {  # Intact on, RA on, pooled on, Intact off, RA off
    "warehouse_logistics": (0.00, 31.25, 15.62, 8.33, 45.83),
    "compute_infrastructure": (2.08, 31.25, 16.67, 0.00, 37.50),
    "financial_reporting": (0.00, 10.42, 5.21, 0.00, 25.00),
    "wet_lab_protocol": (0.00, 4.17, 2.08, 0.00, 25.00),
}
for d, (io, ao, po, if_, af) in REPORT_DOM.items():
    add(S, f"{d} — Intact cot_on", io, rate("intact", "cot_on", d))
    add(S, f"{d} — Refusal-Ablated cot_on", ao, rate("refusal_ablated", "cot_on", d))
    add(S, f"{d} — Pooled cot_on", po,
        (rate("intact", "cot_on", d) + rate("refusal_ablated", "cot_on", d)) / 2)
    add(S, f"{d} — Intact cot_off", if_, rate("intact", "cot_off", d))
    add(S, f"{d} — Refusal-Ablated cot_off", af, rate("refusal_ablated", "cot_off", d))


def fmt_rate(cot, f):
    cells = [b[a][cot][f][d] for a in b for d in DOM]
    return 100 * sum(c["n"] * c["shortcut_violation_rate"] for c in cells) / sum(c["n"] for c in cells)


add(S, "Narrative-stakes rate cot_on", 10.94, fmt_rate("cot_on", "narrative"))
add(S, "Numeric-table rate cot_on", 8.85, fmt_rate("cot_on", "numeric_table"))
add(S, "Narrative-stakes rate cot_off", 19.27, fmt_rate("cot_off", "narrative"))
add(S, "Numeric-table rate cot_off", 16.15, fmt_rate("cot_off", "numeric_table"))

co = e3["exp3_1_mixed_effects_glmm"]["coefficients"]
NOTE_POOLED = "receipt has one GLMM pooling cot_on+cot_off (cot_on is a covariate)"
for label, key, ro, rp in (("compute vs warehouse", "domain_compute_vs_warehouse", 1.080, 0.858),
                           ("financial vs warehouse", "domain_financial_vs_warehouse", 0.285, 0.0103),
                           ("wet-lab vs warehouse", "domain_wetlab_vs_warehouse", 0.085, 0.00086)):
    add(S, f"OR {label}", ro, co[key]["odds_ratio"], NUM, NOTE_POOLED)
    add(S, f"p {label}", rp, co[key]["p_value"], P)
add(S, "OR payoff format (cot_on model)", 1.317, co["numeric_table_format"]["odds_ratio"], NUM,
    "receipt coefficient is numeric_table_format (REPORT labels it narrative vs numeric)")
add(S, "p payoff format (cot_on model)", 0.408, co["numeric_table_format"]["p_value"], P)
add(S, "OR payoff format (cot_off model)", 1.320, None, NUM, NOTE_POOLED)
add(S, "OR high_tedium vs low_cost", 6.96, None, NUM,
    "no cost-regime coefficient; behavioral summary has no cost-regime split")
add(S, "p high_tedium vs low_cost", 1.50e-5, None, P)
add(S, "OR Refusal-Ablated vs Intact", 52.87, co["refusal_ablated"]["odds_ratio"], NUM)
add(S, "p Refusal-Ablated vs Intact", 8.43e-5, co["refusal_ablated"]["p_value"], P)

# ---------------------------------------------------------------- Exp 3.2
S = "3.3 Exp 3.2 pre-action probes (Table 4, Fig 3)"
pe = e3["exp3_2_probe_evaluation"]
add(S, "cot_off violation base rate (positives / 384)", None,
    pe["cot_off"]["overall_violation_rate"] * 100, PCT,
    "≈6 positives, all in warehouse; LODO folds 2–4 have zero positives")
METHODS = {
    "LODO_cross_validation": [
        ("Linear probe", "linear_activation_probe", (0.7698, 0.2738, 0.1306, 0.9854, 0.9355, 0.9587)),
        ("1-D h·r̂", "baseline_1_r_hat_1d_projection", (0.4851, 0.1574, 0.0, 0.5062, 0.2640, 0.0)),
        ("Single-domain probe", "baseline_2_single_domain_warehouse_probe_on_heldout_3_domains",
         (0.6837, 0.2194, 0.1667, 0.9571, 0.8304, 0.8542)),
        ("TF-IDF LR", "baseline_3_text_tfidf_logistic", (0.6100, 0.2233, 0.0, 0.6295, 0.3821, 0.0)),
        ("Surface metadata", "baseline_4_surface_task_features", (0.5620, 0.1670, 0.0, 0.7313, 0.3812, 0.3519)),
        ("External monitor", "baseline_5_external_monitor_5", (0.2438, 0.0734, 0.0, 0.1806, 0.1292, 0.0)),
        ("Label-shuffle null", "baseline_6_label_shuffle_permutation", (0.3310, None, None, 0.3910, None, None)),
    ],
    "LTTO_leave_three_templates_out_cv": [
        ("Linear probe", "linear_activation_probe", (0.9032, 0.5622, 0.5206, 0.9934, 0.9718, 0.9861)),
        ("1-D h·r̂", "baseline_1_r_hat_1d_projection", (0.4851, 0.1116, 0.0, 0.5062, 0.2059, 0.0)),
        ("TF-IDF LR", "baseline_3_text_tfidf_logistic", (0.8127, 0.4272, 0.2611, 0.7664, 0.5015, 0.4437)),
        ("Surface metadata", "baseline_4_surface_task_features", (0.7753, 0.3407, 0.2190, 0.8732, 0.5430, 0.4714)),
        ("Label-shuffle null", "baseline_6_label_shuffle_permutation", (0.4995, None, None, 0.4957, None, None)),
    ],
}
for proto, methods in METHODS.items():
    tag = "LODO" if proto.startswith("LODO") else "LTTO"
    for label, key, reps in methods:
        for i, cot in enumerate(("cot_on", "cot_off")):
            m = pe[cot][proto].get(key)
            add(S, f"{tag} {label} AUROC {cot}", reps[3 * i],
                None if m is None else m["auroc"], NUM)
            for j, metric in ((1, "AUPRC"), (2, "TPR@5%FPR")):
                if reps[3 * i + j] is not None:
                    add(S, f"{tag} {label} {metric} {cot}", reps[3 * i + j], None, NUM,
                        "receipt stores only auroc, fpr_at_80tpr, ppv_at_80tpr_10pct_prev")

# ---------------------------------------------------------------- write
by_status = {"MATCH": 0, "MISMATCH": 0, "MISSING": 0}
lines_by_section = {}
for section, item, rep, recv, kind, note in rows:
    if rep is None:  # reference-only row
        s = "INFO"
    else:
        s = status(rep, recv, kind)
        by_status[s] += 1
    lines_by_section.setdefault(section, []).append(
        f"| {item} | {fmt(rep, kind)} | {fmt(recv, kind)} | **{s}** | {note} |")

out = [
    "# Paper 1 receipt audit: REPORT.md vs. `records/*.json`",
    "",
    "Generated by `scripts/audit_receipts_p1.py`. REPORT.md values are transcribed; receipt values are",
    "read directly from the JSON. Percentages compared at 0.1 pp display precision; p-values within 2%;",
    "other numbers within 0.5%. **MISSING** = no receipt field backs the REPORT claim.",
    "",
    f"**Summary:** {by_status['MATCH']} match, {by_status['MISMATCH']} mismatch, "
    f"{by_status['MISSING']} missing (of {sum(by_status.values())} audited REPORT values).",
    "",
]
for section, lines in lines_by_section.items():
    sec_counts = {k: sum(f"**{k}**" in l for l in lines) for k in by_status}
    out += [f"## {section}", "",
            f"{sec_counts['MATCH']} match · {sec_counts['MISMATCH']} mismatch · {sec_counts['MISSING']} missing",
            "", "| Item | REPORT.md | Receipt | Status | Note |", "| :--- | ---: | ---: | :---: | :--- |",
            *lines, ""]
OUT.write_text("\n".join(out))
print(f"wrote {OUT.relative_to(ROOT)}: {by_status}")
for section, lines in lines_by_section.items():
    print(f"  {section}: " + ", ".join(f"{k}={sum(f'**{k}**' in l for l in lines)}" for k in by_status))
