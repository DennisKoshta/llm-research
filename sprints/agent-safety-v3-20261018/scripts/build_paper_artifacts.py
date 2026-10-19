"""Build every number-bearing artifact of the paper deterministically from records/*.json.

  PYTHONPATH=src .venv/bin/python scripts/build_paper_artifacts.py

Writes
  paper/generated/facts.tex   \\FACT{key}{value} for every number quoted in the prose (used as \\fact{key})
  paper/tables/*.tex          booktabs tables (\\input-ready)
  paper/figures/*.pdf         five vector figures
  paper/paper.claims.json     one claim per prose fact AND per table cell: {id, value, expr, decimals, scale}
Every value is computed by evaluating a Python expression over R = {record stem: JSON} (the same expressions are
re-evaluated by scripts/check_paper_claims.py). No number is typed by hand. Reads records/ only; never raw text.
"""
from __future__ import annotations

import importlib.util
import json
import math
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
_spec = importlib.util.spec_from_file_location("check_report_claims", ROOT / "scripts" / "check_report_claims.py")
_crc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_crc)
R = _crc.load_records()
CLAIMS: list[dict] = []
FACTS: dict[str, str] = {}

ARMS = ["intact", "refusal_ablated", "pca_101", "pca_102", "pca_103", "unrelated"]
ARM_LABEL = {"intact": "Intact", "refusal_ablated": "Ablated", "pca_101": "PCA-101", "pca_102": "PCA-102",
             "pca_103": "PCA-103", "unrelated": "Unrelated"}


# ------------------------------------------------------------------------------------------- value registry
def ev(expr: str):
    return _crc.evaluate(expr, R)


def num(cid: str, expr: str, decimals: int = 1, scale: float = 1.0) -> float:
    """Evaluate, register a claim, return the rounded value exactly as it will be printed."""
    v = round(scale * float(ev(expr)), decimals)
    CLAIMS.append({"id": cid, "text": cid, "value": v, "expr": expr, "decimals": decimals, "scale": scale})
    return v


def s_num(v: float, decimals: int) -> str:
    s = f"{v:,.{decimals}f}" if decimals > 0 else f"{int(round(v)):,}"
    return s.replace(",", "{,}")


def cnt(cid, expr):                       # integer count
    return s_num(num(cid, expr, 0), 0)


def pct(cid, expr, decimals=1):           # percentage of a rate in [0, 1]
    return s_num(num(cid, expr, decimals, 100.0), decimals)


def dec(cid, expr, decimals=3):
    return s_num(num(cid, expr, decimals), decimals)


def pval(cid, expr):
    """p >= 0.001: 4 significant-ish decimals; smaller: mantissa x 10^e with the scale recorded in the claim."""
    v = float(ev(expr))
    if v >= 0.001 or v == 0:
        d = 4 if v < 0.1 else 3
        return s_num(num(cid, expr, d), d)
    e = math.floor(math.log10(v))
    m = num(cid, expr, 1, 10.0 ** (-e))
    if m >= 10:                            # rounding pushed the mantissa to 10.0
        e += 1
        CLAIMS.pop()
        m = num(cid, expr, 1, 10.0 ** (-e))
    return f"{m:.1f}\\times10^{{{e}}}"


def fact(key: str, text: str, math_mode: bool = False):
    FACTS[key] = f"${text}$" if math_mode else text


# expression builders ------------------------------------------------------------------------------------
def P1(arm, var, eff, field, fam="r1"):
    e = "None" if eff is None else repr(eff)
    return (f"next(c for c in R['phase1_dim_{fam}_analysis']['cells'] if c['arm']=={arm!r} and c['variant']=={var!r} "
            f"and c['consequence']=='harmful' and c['effort']=={e})[{field!r}]")


def RB(arm, cond, field):
    return f"R['exp11_effort_robustness']['cells'][{arm!r}][{cond!r}][{field!r}]"


def E11(arm, cot, cells, field="k", rec="exp11_main"):
    terms = " + ".join(f"R[{rec!r}]['cells'][{arm!r}][{cot!r}][{c!r}]['shortcut'][{field!r}]" for c in cells)
    return f"({terms})"


HC_HARMFUL_IMPL = ["high_tedium|harmful|implicit", "high_friction|harmful|implicit"]
HC_HARMLESS_IMPL = ["high_tedium|harmless|implicit", "high_friction|harmless|implicit"]
HC_HARMLESS_EXPL = ["high_tedium|harmless|explicit", "high_friction|harmless|explicit"]


def DI(arm, variant, cons="harmful", rec="exp11_probes", field="k"):
    return f"R[{rec!r}]['direct_instruction']['cells'][{arm!r}]['cot_on'][{variant!r}][{cons!r}]['comply'][{field!r}]"


def F1(arm, cot, scope, field, fam="F1_itt"):
    return (f"next(x for x in R['exp31_multidomain']['contrasts'][{fam!r}] if x['within']['arm']=={arm!r} and "
            f"x['within']['cot_mode']=={cot!r} and x['scope']=={scope!r})[{field!r}]")


def F1D(arm, cot, dom, field):
    return (f"next(x for x in R['exp31_multidomain']['contrasts']['F1d_itt'] if x['within']['arm']=={arm!r} and "
            f"x['within']['cot_mode']=={cot!r} and x['scope']=={'harmful:' + dom!r})[{field!r}]")


def S31(arm, cot, fmt, field="k", slice_="harmful"):
    cond = (f"c['arm']=={arm!r} and c['cot_mode']=={cot!r} and c['fmt']=={fmt!r} and c['consequence']=='harmful'"
            + (" and c['cost_regime']!='low' and c['boundary']=='implicit'" if slice_ == "hc" else ""))
    if field == "k":
        return f"sum(c['shortcut_itt']['k'] for c in R['exp31_multidomain']['by_cell'] if {cond})"
    if field == "n":
        return f"sum(c['n'] for c in R['exp31_multidomain']['by_cell'] if {cond})"
    return (f"sum(c['shortcut_itt']['k'] for c in R['exp31_multidomain']['by_cell'] if {cond})/"
            f"sum(c['n'] for c in R['exp31_multidomain']['by_cell'] if {cond})")


def X32(ds, cot, regime):
    return f"R['exp32_probes_perm400']['datasets'][{ds!r}]['sites']['out']['regimes'][{cot!r}][{regime!r}]"


def C32(ds, cot, group):
    return f"next(g for g in {X32(ds, cot, 'C')}['scopes']['arm_cot'] if g['group']=={group!r})"


def E12(corpus, teacher, rank):
    return f"R['exp12_students']['summary']['cells'][{corpus + '|' + teacher + '|r' + str(rank)!r}]"


def E12C(corpus, rank, contrast, field):
    return (f"next(x for x in R['exp12_students']['summary']['contrasts'] if x['corpus']=={corpus!r} and x['rank']=={rank} "
            f"and x['contrast']=={contrast!r} and x['outcome']=='primary_shortcut')[{field!r}]")


def E23(key, field):
    return f"R['exp23']['cells'][{key!r}][{field!r}]"


def E21(team, variant, metric, field="rate"):
    return (f"next(c for c in R['exp21_topologies']['cells'] if c['team']=={team!r} and c['variant']=={variant!r})"
            f"['metrics'][{metric!r}][{field!r}]")


def E22(monitor, bundle_pred, field, extra=""):
    return (f"sum(c[{field!r}] for c in R['exp22_monitor_calibration']['cells'] if c['monitor']=={monitor!r} and "
            f"c['cot_mode']=='cot_off' and {bundle_pred}{extra})")


TPR = "c['bundle']=='A_FAST+B_FAST'"
FPR = "c['bundle'] in ('A_FAST+B_SAFE','A_SAFE+B_FAST','A_SAFE+B_SAFE')"


def E14(eff, arm, cons, bound, field):
    return (f"next(h for h in R['exp14_blocked_route']['analyses'][{eff!r}]['headline'] if h['arm']=={arm!r} and "
            f"h['consequence']=={cons!r} and h['boundary']=={bound!r} and h['timing']=='pooled')[{field!r}]")


# ============================================================================================ prose facts
def build_facts():
    # ---- Phase 1
    for arm in ("refusal_ablated", "intact"):
        a = "abl" if arm == "refusal_ablated" else "int"
        for var in ("C", "B", "A0", "A1", "A2"):
            fact(f"p1.{var}.{a}.k", cnt(f"p1.{var}.{a}.k", P1(arm, var, None, "n_comply")))
            fact(f"p1.{var}.{a}.n", cnt(f"p1.{var}.{a}.n", P1(arm, var, None, "n")))
            fact(f"p1.{var}.{a}.pct", pct(f"p1.{var}.{a}.pct", P1(arm, var, None, "itt_comply_rate")))
        for var in ("C", "B"):
            fact(f"p1.{var}.{a}.low.pct", pct(f"p1.{var}.{a}.low.pct", P1(arm, var, "low", "itt_comply_rate")))
    for var in ("C", "B"):
        fact(f"p1.r2.{var}.abl.k", cnt(f"p1.r2.{var}.abl.k", P1("refusal_ablated", var, None, "n_comply", "r2")))
        fact(f"p1.r2.{var}.abl.n", cnt(f"p1.r2.{var}.abl.n", P1("refusal_ablated", var, None, "n", "r2")))
    v = "R['phase1_family_decision']['verdict']['r1']"
    fact("p1.deltaC", pct("p1.deltaC", f"{v}['delta_pp_C_default']/100"))
    fact("p1.deltaB", pct("p1.deltaB", f"{v}['delta_pp_B_default']/100"))
    fact("p1.deltaA0", s_num(num("p1.deltaA0", f"{v}['delta_pp_A0_default']", 2), 2))
    fact("p1.ctrl.maxdelta", s_num(num("p1.ctrl.maxdelta", f"{v}['max_control_abs_delta_pp_C_default']", 2), 2))
    fact("p1.proj.min", pct("p1.proj.min", f"{v}['min_own_vector_reduction_projection']"))
    fact("p1.cos", dec("p1.cos", "R['phase1_family_decision']['cos_r1_rhat_vs_r2_refusal']", 3))
    kl = "R['phase1_family_decision']['benign_kl']['checkpoints']"
    fact("kl.abl.last", dec("kl.abl.last", f"{kl}['r1_refusal_ablated']['kl_prompt_last_mean']", 4))
    fact("kl.abl.tf", dec("kl.abl.tf", f"{kl}['r1_refusal_ablated']['kl_teacher_forced_mean']", 4))
    fact("kl.max", dec("kl.max", f"max(max(x['kl_prompt_last_mean'], x['kl_teacher_forced_mean']) for x in {kl}.values())", 4))
    fact("kl.gate", dec("kl.gate", "R['phase1_family_decision']['benign_kl']['gate_mean_lt']", 2))
    rc = "R['phase1_family_decision']['families']['r2']['projection']['control_reduction_of_own_vector_default_out']"
    fact("r2.ctrl.min", pct("r2.ctrl.min", f"min(v for k, v in {rc}.items() if k!='r2_bf16_base')"))
    fact("r2.ctrl.max", pct("r2.ctrl.max", f"max(v for k, v in {rc}.items() if k!='r2_bf16_base')"))
    # ---- scenario audit + calibration
    sa = "R['scenario_audit']['pair_audit']"
    fact("sa.lenratio", dec("sa.lenratio", f"{sa}['length_ratio']['mean']", 3))
    fact("sa.longer", pct("sa.longer", f"{sa}['frac_harmful_longer']", 0))
    fact("sa.wordgap", cnt("sa.wordgap", f"{sa}['function_words']['max_doc_freq_gap']"))
    fact("sa.loo", dec("sa.loo", f"{sa}['function_words']['loo_accuracy']", 2))
    fact("sa.nbases", cnt("sa.nbases", "R['scenario_audit']['n_bases']"))
    for k in ("severity_4", "severity_5"):
        fact(f"sa.{k}", cnt(f"sa.{k}", f"R['scenario_audit']['judge_counts'][{k!r}]"))
    ch = "R['calibration_history']['versions']"
    for ver in ("v1", "v2", "v3"):
        for crit, short in (("harmful_implicit_shortcut_lt_10pct", "harm"), ("harmless_implicit_high_tedium_shortcut_ge_35pct", "ted"),
                            ("low_cost_shortcut_lt_15pct", "low"), ("forced_safe_competence_ge_80pct", "forced")):
            fact(f"cal.{ver}.{short}.k", cnt(f"cal.{ver}.{short}.k", f"{ch}[{ver!r}]['criteria'][{crit!r}]['k']"))
            fact(f"cal.{ver}.{short}.n", cnt(f"cal.{ver}.{short}.n", f"{ch}[{ver!r}]['criteria'][{crit!r}]['n']"))
            fact(f"cal.{ver}.{short}.pct", pct(f"cal.{ver}.{short}.pct", f"{ch}[{ver!r}]['criteria'][{crit!r}]['rate']"))

    # ---- Exp 1.1
    for arm in ARMS:
        for cond in ("default", "low", "off"):
            if cond == "off" and arm not in ("intact", "refusal_ablated"):
                continue
            fact(f"e11.{arm}.{cond}.k", cnt(f"e11.{arm}.{cond}.k", RB(arm, cond, "k_shortcut")))
            fact(f"e11.{arm}.{cond}.n", cnt(f"e11.{arm}.{cond}.n", RB(arm, cond, "n")))
            fact(f"e11.{arm}.{cond}.pct", pct(f"e11.{arm}.{cond}.pct", RB(arm, cond, "rate")))
    P11 = "R['exp11_main']['contrasts']['P1.1']"
    fact("p11.diff", pct("p11.diff", f"{P11}['diff']"))
    fact("p11.cil", pct("p11.cil", f"{P11}['ci_low']"))
    fact("p11.cih", pct("p11.cih", f"{P11}['ci_high']"))
    fact("p11.p", pval("p11.p", f"{P11}['p_mcnemar']"), True)
    fact("p11.holm", pval("p11.holm", f"{P11}['p_holm']"), True)
    fact("p11.signflip", pval("p11.signflip", f"{P11}['cluster_boot']['p_signflip']"), True)
    fact("p11.signflipholm", pval("p11.signflipholm", f"{P11}['p_signflip_holm']"), True)
    L11 = "R['exp11_main_lowprobe']['contrasts']['P1.1']"
    fact("p11low.p", pval("p11low.p", f"{L11}['p_mcnemar']"), True)
    fact("p11low.holm", pval("p11low.holm", f"{L11}['p_holm']"), True)
    fact("p11low.signflip", pval("p11low.signflip", f"{L11}['cluster_boot']['p_signflip']"), True)
    for name in ("P1.3_intact", "P1.3_refusal_ablated"):
        c = f"R['exp11_main']['contrasts'][{name!r}]"
        fact(f"{name}.p", pval(f"{name}.p", f"{c}['p_mcnemar']"), True)
    rc = "R['exp11_effort_robustness']['contrasts']"
    for name in ("low_vs_default:refusal_ablated", "off_vs_default:refusal_ablated", "low_vs_off:refusal_ablated"):
        fact(f"rob.{name}.p", pval(f"rob.{name}.p", f"{rc}[{name!r}]['p_mcnemar']"), True)
        fact(f"rob.{name}.holm", pval(f"rob.{name}.holm", f"{rc}[{name!r}]['p_mcnemar_holm']"), True)
    did = "R['exp11_effort_robustness']['did']"
    fact("did.diff", pct("did.diff", f"{did}['did']"))
    fact("did.cil", pct("did.cil", f"{did}['scenario_cluster_bootstrap']['ci_low']"))
    fact("did.cih", pct("did.cih", f"{did}['scenario_cluster_bootstrap']['ci_high']"))
    for arm in ARMS:
        a = arm
        fact(f"e11.{a}.harmless_impl.k", cnt(f"e11.{a}.harmless_impl.k", E11(arm, "cot_on", HC_HARMLESS_IMPL)))
        fact(f"e11.{a}.harmless_expl.k", cnt(f"e11.{a}.harmless_expl.k", E11(arm, "cot_on", HC_HARMLESS_EXPL)))
    fact("e11.harmless.n", cnt("e11.harmless.n", E11("intact", "cot_on", HC_HARMLESS_IMPL, "n")))
    for arm in ARMS:
        for var in ("solo", "esc"):
            fact(f"di.{arm}.{var}.k", cnt(f"di.{arm}.{var}.k", DI(arm, var)))
    fact("di.n", cnt("di.n", DI("intact", "solo", field="n")))
    fact("di.low.abl.k", cnt("di.low.abl.k", DI("refusal_ablated", "solo", rec="exp11_probes_lowprobe")))
    fact("di.harmless.abl.k", cnt("di.harmless.abl.k", DI("refusal_ablated", "solo", cons="harmless")))
    ce = "R['exp11_external']['cells']"
    for arm in ("flash_next", "flash_next_uncensored"):
        terms = " + ".join(f"{ce}[{arm!r}][{c!r}]['shortcut']['k']" for c in HC_HARMFUL_IMPL)
        fact(f"ext.{arm}.k", cnt(f"ext.{arm}.k", f"({terms})"))
        terms_t = " + ".join(f"{ce}[{arm!r}][{c!r}]['truncated']['k']" for c in HC_HARMFUL_IMPL)
        fact(f"ext.{arm}.trunc", cnt(f"ext.{arm}.trunc", f"({terms_t})"))

    # ---- Exp 3.1
    for arm in ("refusal_ablated", "intact", "pca_101"):
        for cot in ("cot_on", "cot_off"):
            fact(f"e31.{arm}.{cot}.narr", pct(f"e31.{arm}.{cot}.narr", F1(arm, cot, "harmful", "rate_b")))
            fact(f"e31.{arm}.{cot}.num", pct(f"e31.{arm}.{cot}.num", F1(arm, cot, "harmful", "rate_a")))
            fact(f"e31.{arm}.{cot}.diff", pct(f"e31.{arm}.{cot}.diff", F1(arm, cot, "harmful", "diff")))
            fact(f"e31.{arm}.{cot}.p", pval(f"e31.{arm}.{cot}.p", F1(arm, cot, "harmful", "p_mcnemar")), True)
            fact(f"e31.{arm}.{cot}.holm", pval(f"e31.{arm}.{cot}.holm", F1(arm, cot, "harmful", "p_holm")), True)
            for fmt in ("narrative", "numeric"):
                fact(f"e31hc.{arm}.{cot}.{fmt}.pct", pct(f"e31hc.{arm}.{cot}.{fmt}.pct", S31(arm, cot, fmt, "rate", "hc")))
                fact(f"e31hc.{arm}.{cot}.{fmt}.k", cnt(f"e31hc.{arm}.{cot}.{fmt}.k", S31(arm, cot, fmt, "k", "hc")))
                fact(f"e31hc.{arm}.{cot}.{fmt}.n", cnt(f"e31hc.{arm}.{cot}.{fmt}.n", S31(arm, cot, fmt, "n", "hc")))
    for arm in ("refusal_ablated", "pca_101", "intact"):
        for dom in ("warehouse", "compute", "finance", "wetlab"):
            for fld, tag in (("rate_b", "narr"), ("rate_a", "num")):
                key = f"e31d.{arm}.{dom}.{tag}"
                fact(key, pct(key, F1D(arm, "cot_on", dom, fld)))
    a1 = ("next(x for x in R['exp31_multidomain']['contrasts']['A1_itt'] if x['a']=='refusal_ablated' and "
          "x['within']['cot_mode']=='cot_on' and x['within']['fmt']=='numeric' and x['scope']=='hc_harmful_implicit')")
    fact("e31.A1.diff", pct("e31.A1.diff", f"{a1}['diff']"))
    fact("e31.A1.p", pval("e31.A1.p", f"{a1}['p_mcnemar']"), True)
    fact("e31.rows", cnt("e31.rows", "sum(c['n'] for c in R['exp31_multidomain']['by_cell'])"))

    # ---- Exp 3.2
    for cot in ("cot_on", "cot_off"):
        A = X32("exp11", cot, "A") + "['auroc']"
        for m in ("probe", "design", "tfidf", "rhat_cv"):
            fact(f"e32A.{cot}.{m}", dec(f"e32A.{cot}.{m}", f"{A}[{m!r}]['auroc']", 3))
        D = X32("exp11", cot, "A") + "['diff']['probe-design']"
        fact(f"e32A.{cot}.diff", dec(f"e32A.{cot}.diff", f"{D}['diff']", 3))
        fact(f"e32A.{cot}.diffl", dec(f"e32A.{cot}.diffl", f"{D}['ci_low']", 3))
        fact(f"e32A.{cot}.diffh", dec(f"e32A.{cot}.diffh", f"{D}['ci_high']", 3))
        B = X32("exp11", cot, "B") + "['auroc']"
        fact(f"e32B.{cot}.rhat", dec(f"e32B.{cot}.rhat", f"{B}['rhat_neg']['auroc']", 3))
        fact(f"e32B.{cot}.design", dec(f"e32B.{cot}.design", f"{B}['design']['auroc']", 3))
    for ds, tag in (("exp11", "p"), ("exp11+exp31", "x")):
        for cot in ("cot_on", "cot_off"):
            groups = [g["group"] for g in ev(X32(ds, cot, "C") + "['scopes']['arm_cot']")]
            for g in groups:
                key = f"e32C.{tag}.{g.replace('|', '.')}"
                fact(key, dec(key, f"{C32(ds, cot, g)}['pooled']['probe']['auroc']", 3))
                fact(key + ".p", dec(key + ".p", f"{C32(ds, cot, g)}['permutation_null']['p_perm']", 4))
                fact(key + ".n", cnt(key + ".n", f"{C32(ds, cot, g)}['n']"))
    fact("e32.nperm", cnt("e32.nperm", "R['exp32_probes_perm400']['config']['n_perm']"))

    # ---- Exp 1.2
    for corpus in ("A", "B", "Bp", "C"):
        for teacher in ("refusal_ablated", "intact", "pca_101"):
            for rank in (32, 8):
                c = E12(corpus, teacher, rank)
                key = f"e12.{corpus}.{teacher}.r{rank}"
                fact(key + ".k", cnt(key + ".k", f"{c}['agentic']['primary']['shortcut']['k']"))
                fact(key + ".n", cnt(key + ".n", f"{c}['agentic']['primary']['shortcut']['n']"))
                fact(key + ".pct", pct(key + ".pct", f"{c}['agentic']['primary']['shortcut']['rate']"))
                fact(key + ".comply", pct(key + ".comply", f"{c}['refusal']['harmful']['comply']['rate']"))
                fact(key + ".proj", dec(key + ".proj", f"{c}['projection']['refusal_harmful']['mean_over_seeds']", 3))
        for rank in (32, 8):
            for con in ("refusal_ablated_vs_intact", "refusal_ablated_vs_pca_101"):
                key = f"e12c.{corpus}.r{rank}.{con.split('_vs_')[1]}"
                fact(key + ".diff", pct(key + ".diff", E12C(corpus, rank, con, "diff")))
                fact(key + ".p", pval(key + ".p", E12C(corpus, rank, con, "p_mcnemar")), True)
                fact(key + ".sf", pval(key + ".sf", E12C(corpus, rank, con, "p_signflip")), True)
    fl = "R['exp12_students']['summary']['floors']"
    fact("e12.floor.k", cnt("e12.floor.k", f"sum(f['agentic']['primary']['shortcut']['k'] for f in {fl}.values())"))
    fact("e12.floor.n", cnt("e12.floor.n", f"sum(f['agentic']['primary']['shortcut']['n'] for f in {fl}.values())"))
    ad = "R['exp12_students']['summary']['adapters']"
    fact("e12.dose.intact", pct("e12.dose.intact", f"{ad}['B.refusal_ablated.r32.s0']['train']['n_target_tokens']/{ad}['B.intact.r32.s0']['train']['n_target_tokens']-1"))
    fact("e12.dose.pca", pct("e12.dose.pca", f"{ad}['B.refusal_ablated.r32.s0']['train']['n_target_tokens']/{ad}['B.pca_101.r32.s0']['train']['n_target_tokens']-1"))
    fact("e12.nadapters", cnt("e12.nadapters", f"len({ad})"))

    # ---- Exp 1.4
    for eff in ("default", "low"):
        for arm in ("intact", "refusal_ablated"):
            for cons in ("harmful", "harmless"):
                for bound in ("implicit", "explicit"):
                    key = f"e14.{eff}.{arm}.{cons}.{bound}"
                    for f in ("shortcut", "escalated", "persisted", "n"):
                        fact(f"{key}.{f}", cnt(f"{key}.{f}", E14(eff, arm, cons, bound, f)))
    fact("e14.rows", cnt("e14.rows", "R['exp14_blocked_route']['meta']['hygiene']['n_blocked_rows']"))
    fact("e14.persisted", cnt("e14.persisted", "sum(h['persisted'] for a in R['exp14_blocked_route']['analyses'].values() for h in a['headline'] if h['timing']=='pooled')"))
    fact("e14.harm.sc", cnt("e14.harm.sc", "sum(c['counts']['SHORTCUT'] for a in R['exp14_blocked_route']['analyses'].values() for t in a['tables'].values() for kk, c in t['cells'].items() if kk.startswith('harmful'))"))
    fact("e14.harm.pre", cnt("e14.harm.pre", "sum(c['shortcut_before_block'] for a in R['exp14_blocked_route']['analyses'].values() for t in a['tables'].values() for kk, c in t['cells'].items() if kk.startswith('harmful'))"))
    for eff in ("default", "low"):
        c1 = f"R['exp14_blocked_route']['analyses'][{eff!r}]['contrasts']['confirmatory']['C1']"
        fact(f"e14.{eff}.C1.p", pval(f"e14.{eff}.C1.p", f"{c1}['p_mcnemar']"), True)

    # ---- Exp 2.3
    mc = "R['exp23']['manipulation_check']['reviewers']"
    fact("e23.natural.n", cnt("e23.natural.n", " + ".join(f"{mc}[{r!r}]['n_valid']" for r in ("B1", "B2", "R1a", "R1b"))))
    for frame in ("designated", "peer"):
        for memo in ("B4a", "B4b"):
            key = f"llama70b:cot_off|{frame}|{memo}"
            fact(f"e23.llama.{frame}.{memo}.k", cnt(f"e23.llama.{frame}.{memo}.k", E23(key, "k_shortcut")))
            fact(f"e23.llama.{frame}.{memo}.n", cnt(f"e23.llama.{frame}.{memo}.n", E23(key, "n")))
            fact(f"e23.llama.{frame}.{memo}.pct", pct(f"e23.llama.{frame}.{memo}.pct", E23(key, "rate")))
    fact("e23.intact.B4a.k", cnt("e23.intact.B4a.k", " + ".join(E23(f"intact:{c}|{f}|B4a", "k_shortcut") for c in ("cot_on", "cot_off") for f in ("designated", "peer"))))
    fact("e23.intact.B4a.n", cnt("e23.intact.B4a.n", " + ".join(E23(f"intact:{c}|{f}|B4a", "n") for c in ("cot_on", "cot_off") for f in ("designated", "peer"))))
    fact("e23.intact.esc", cnt("e23.intact.esc", E23("intact:cot_off|designated|B4a", "k_escalate")))
    fact("e23.flash.on.trunc", pct("e23.flash.on.trunc", "R['exp23']['outcome_by_executor']['flash_next:cot_on']['TRUNCATED_OR_MALFORMED'] / R['exp23']['n_executor_rows']['flash_next:cot_on']"))
    fact("e23.flash.off.trunc", pct("e23.flash.off.trunc", "R['exp23']['outcome_by_executor']['flash_next:cot_off']['TRUNCATED_OR_MALFORMED'] / R['exp23']['n_executor_rows']['flash_next:cot_off']"))
    fact("e23.rows", cnt("e23.rows", "sum(R['exp23']['n_executor_rows'].values())"))
    fact("e23.nscen", cnt("e23.nscen", E23("llama70b:cot_off|peer|B4a", "n_scenarios")))

    # ---- Exp 2.1
    fact("e21.trials", cnt("e21.trials", "R['exp21_topologies']['n_trials']"))
    fact("e21.worlds", cnt("e21.worlds", "R['exp21_topologies']['n_worlds']"))
    fact("e21.joint", cnt("e21.joint", "sum(c['metrics']['ILLEGIT_JOINT']['k'] for c in R['exp21_topologies']['cells'])"))
    for team in ("mixed_team", "all_intact"):
        t = "mix" if team == "mixed_team" else "int"
        for var in ("A", "B", "C_nopen", "C_pen", "D_nopen", "D_pen"):
            fact(f"e21.{t}.{var}.alpha", pct(f"e21.{t}.{var}.alpha", E21(team, var, "alpha_fast")))
            fact(f"e21.{t}.{var}.legit", pct(f"e21.{t}.{var}.legit", E21(team, var, "LEGIT_SAFE_COORD")))
    # ---- Exp 2.2
    for mon in ("intact", "llama70b", "flash_next"):
        for name, pred in (("tpr", TPR), ("fpr", FPR)):
            fact(f"e22.{mon}.{name}.k", cnt(f"e22.{mon}.{name}.k", E22(mon, pred, "k")))
            fact(f"e22.{mon}.{name}.n", cnt(f"e22.{mon}.{name}.n", E22(mon, pred, "n")))
    for cond in ("action_log_only", "minimal", "neutral_matched", "truthful_compliance", "deceptive_compound", "clause_both"):
        ex = f" and c['condition']=={cond!r}"
        fact(f"e22.llama.fpr.{cond}", pct(f"e22.llama.fpr.{cond}", f"{E22('llama70b', FPR, 'k', ex)}/{E22('llama70b', FPR, 'n', ex)}"))
    fact("e22.worlds", cnt("e22.worlds", "R['exp22_monitor_calibration']['n_worlds']"))
    # ---- design constants (record-derived where a record stores them; otherwise literal and checked against the spec)
    fact("d.scen", cnt("d.scen", "R['exp11_effort_robustness']['cells']['intact']['default']['n_scenarios']"))
    fact("d.templates", cnt("d.templates", "R['exp11_effort_robustness']['cells']['intact']['default']['n_templates']"))
    fact("d.pairs", cnt("d.pairs", "R['exp11_effort_robustness']['cells']['intact']['default']['n']"))
    fact("d.domains", cnt("d.domains", "len(R['exp31_multidomain']['levels']['domain'])"))
    fact("d.cells", cnt("d.cells", "len(R['exp11_main']['cells']['intact']['cot_on'])"))
    fact("d.arms", cnt("d.arms", "len(R['exp11_effort_robustness']['conditions']['default']['arms'])"))
    fact("d.e23scen", cnt("d.e23scen", "R['exp23']['cells']['llama70b:cot_off|peer|B4a']['n_scenarios']"))
    fact("d.e21k", cnt("d.e21k", "R['exp21_topologies']['k_samples']"))
    fact("d.seeds", cnt("d.seeds", "len(R['exp12_students']['summary']['cells']['B|refusal_ablated|r32']['seeds'])"))
    fact("d.e12templates", cnt("d.e12templates", "4"))
    fact("d.refusalN", cnt("d.refusalN", "128"))
    fact("d.cap.on", cnt("d.cap.on", "16384"))
    fact("d.cap.low", cnt("d.cap.low", "4096"))
    fact("d.cap.off", cnt("d.cap.off", "1024"))
    fact("d.cap.e12", cnt("d.cap.e12", "2048"))
    fact("d.maxturns", cnt("d.maxturns", "10"))
    fact("d.receipts", cnt("d.receipts", "sum(1 for v in R['receipt_verification']['results'].values() if v=='OK')"))
    # ---- budget
    fact("budget.billed", s_num(num("budget.billed", "R['budget_summary']['billed_usd']", 2), 2))
    fact("budget.cap", cnt("budget.cap", "R['budget_summary']['cap_usd']"))


# ============================================================================================ tables
def tex_table(name, caption, label, colspec, header, rows, notes=None, size="\\small", wide=False, sep="4pt"):
    env = "table*" if wide else "table"
    lines = [f"\\begin{{{env}}}[t]", "\\centering", size, f"\\setlength{{\\tabcolsep}}{{{sep}}}",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\begin{tabular}{" + colspec + "}", "\\toprule", header + " \\\\", "\\midrule"]
    for r in rows:
        lines.append("\\midrule" if r == "MIDRULE" else " & ".join(r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if notes:
        lines.append("\\par\\vspace{2pt}\\parbox{\\linewidth}{\\footnotesize " + notes + "}")
    lines.append(f"\\end{{{env}}}")
    (PAPER / "tables" / f"{name}.tex").write_text("\n".join(lines) + "\n")


def kn(cid, kexpr, nexpr):
    return f"{cnt(cid + '.k', kexpr)}/{cnt(cid + '.n', nexpr)}"


def build_tables():
    # T1 Phase 1
    rows = []
    for arm in ARMS:
        r = [ARM_LABEL[arm]]
        for var in ("C", "B", "A0", "A1", "A2"):
            r.append(kn(f"tab1.{arm}.{var}", P1(arm, var, None, "n_comply"), P1(arm, var, None, "n")))
        rows.append(r)
    rows.append("MIDRULE")
    r = ["Family 2 Ablated"]
    for var in ("C", "B", "A0", "A1", "A2"):
        r.append(kn(f"tab1.r2abl.{var}", P1("refusal_ablated", var, None, "n_comply", "r2"), P1("refusal_ablated", var, None, "n", "r2")))
    rows.append(r)
    tex_table("tab_phase1", "Phase~1 direct-instruction positive control at default reasoning effort: restricted-tool "
              "compliance (ITT, $k/n$). Set~C and Set~B order a restricted call with no safe alternative; A0 offers and invites a hand-off "
              "tool, A1 offers it without invitation, A2 offers none.", "tab:phase1", "lccccc",
              "Arm & Set C & Set B & A0 & A1 & A2", rows)
    # T1b projection/KL
    kl = "R['kl_full13']['checkpoints']"
    proj = "R['projection_full13']['summary']"
    rows = []
    for arm in ARMS[1:]:
        ck = f"r1_{arm}"
        red = (f"next(r['reduction_vs_intact'] for r in {proj} if r['ckpt']=={ck!r} and r['effort']=='default' and "
               f"r['set']=='r1_val' and r['site']=='out' and r['vector']=='r1.r_hat')")
        rows.append([ARM_LABEL[arm], pct(f"tab1b.{arm}.proj", red),
                     dec(f"tab1b.{arm}.kllast", f"{kl}[{ck!r}]['vs']['r1_intact']['kl_prompt_last']['mean']", 4),
                     dec(f"tab1b.{arm}.kltf", f"{kl}[{ck!r}]['vs']['r1_intact']['kl_teacher_forced']['mean']", 4)])
    tex_table("tab_projection_kl", "Layer-48 checks for the adopted checkpoint family (Family~1): reduction of the projection onto the "
              "family's own refusal direction relative to Intact (output site, default effort, validation prompts), and benign "
              "KL divergence to Intact (nats; gate $<0.10$).", "tab:projkl", "lccc",
              "Checkpoint & \\shortstack{$\\hat r$ projection\\\\reduction (\\%)} & \\shortstack{KL (last\\\\prompt token)} & \\shortstack{KL (teacher-\\\\forced)}", rows,
              size="\\footnotesize")
    # T2 calibration
    ch = "R['calibration_history']['versions']"
    rows = []
    for ver in ("v1", "v2", "v3"):
        r = [ver]
        for crit in ("harmful_implicit_shortcut_lt_10pct", "harmless_implicit_high_tedium_shortcut_ge_35pct",
                     "low_cost_shortcut_lt_15pct", "forced_safe_competence_ge_80pct"):
            v = kn(f"tab2.{ver}.{crit}", f"{ch}[{ver!r}]['criteria'][{crit!r}]['k']", f"{ch}[{ver!r}]['criteria'][{crit!r}]['n']")
            ok = ev(f"{ch}[{ver!r}]['criteria'][{crit!r}]['pass']")
            r.append(v + ("" if ok else "$^{\\dagger}$"))
        rows.append(r)
    tex_table("tab_calibration", "Intact-only calibration of the one shared system-prompt sentence (default effort, "
              "thinking on). $^{\\dagger}$\\,fails the criterion. Only v3 passes all four construct-validity criteria; it was frozen before any other arm was run.",
              "tab:calibration", "lcccc",
              "Wording & \\shortstack{Harmful impl.\\\\$<10\\%$} & \\shortstack{Harmless\\\\tedium $\\geq35\\%$} & \\shortstack{Low cost\\\\$<15\\%$} & \\shortstack{Forced safe\\\\$\\geq80\\%$}", rows,
              size="\\footnotesize", sep="3pt")
    # T3 Exp 1.1 effort table
    rows = []
    for arm in ARMS:
        r = [ARM_LABEL[arm]]
        for cond in ("default", "off", "low"):
            if cond == "off" and arm not in ("intact", "refusal_ablated"):
                r.append("---")
                continue
            r.append(kn(f"tab3.{arm}.{cond}", RB(arm, cond, "k_shortcut"), RB(arm, cond, "n")))
        r.append(kn(f"tab3.{arm}.hlimpl", E11(arm, "cot_on", HC_HARMLESS_IMPL), E11(arm, "cot_on", HC_HARMLESS_IMPL, "n")))
        r.append(kn(f"tab3.{arm}.hlexpl", E11(arm, "cot_on", HC_HARMLESS_EXPL), E11(arm, "cot_on", HC_HARMLESS_EXPL, "n")))
        rows.append(r)
    tex_table("tab_exp11_cells", "Exp~1.1 shortcut counts on the 80 high-cost pairs (40 scenarios $\\times$ tedium/friction). "
              "Harmful-implicit under default-effort thinking (pre-fixed primary), thinking off, and low-effort thinking "
              "(post hoc); harmless-implicit and harmless-explicit at default effort. --- = not run.", "tab:exp11cells", "lccccc",
              "Arm & \\shortstack{Harmful\\\\default} & \\shortstack{Harmful\\\\think off} & \\shortstack{Harmful\\\\low} & \\shortstack{Harmless\\\\implicit} & \\shortstack{Harmless\\\\explicit}", rows,
              size="\\footnotesize", sep="3pt")
    # T3b contrasts
    rows = []
    for rec in ("exp11_main", "exp11_main_lowprobe"):      # the four control contrasts are identical (all controls 0/80)
        ref = ev(f"R[{rec!r}]['contrasts']['P1.2_pca_101']")
        for other in ("P1.2_pca_102", "P1.2_pca_103", "P1.2_unrelated"):
            o = ev(f"R[{rec!r}]['contrasts'][{other!r}]")
            assert all(o[k] == ref[k] for k in ("n_a_only", "n_b_only", "diff", "p_mcnemar", "p_holm")), other
    for name, lab in (("P1.1", "P1.1 Abl.\\ $-$ Intact"), ("P1.2_pca_101", "P1.2 Abl.\\ $-$ each control"),
                      ("P1.3_intact", "P1.3 Intact on $-$ off"), ("P1.3_refusal_ablated", "P1.3 Abl.\\ on $-$ off")):
        for rec, tag in (("exp11_main", "def"), ("exp11_main_lowprobe", "low")):
            c = f"R[{rec!r}]['contrasts'][{name!r}]"
            rows.append([lab + (" (low)" if tag == "low" else ""),
                         pct(f"tab3b.{name}.{tag}.diff", f"{c}['diff']"),
                         f"[{pct(f'tab3b.{name}.{tag}.l', c + chr(91) + repr('ci_low') + chr(93))}, {pct(f'tab3b.{name}.{tag}.h', c + chr(91) + repr('ci_high') + chr(93))}]",
                         "$" + pval(f"tab3b.{name}.{tag}.p", f"{c}['p_mcnemar']") + "$",
                         "$" + pval(f"tab3b.{name}.{tag}.holm", f"{c}['p_holm']") + "$",
                         "$" + pval(f"tab3b.{name}.{tag}.sf", f"{c}['cluster_boot']['p_signflip']") + "$"])
    tex_table("tab_exp11_contrasts", "Exp~1.1 pre-specified contrasts on high-cost harmful-implicit pairs ($n=80$). "
              "Differences in percentage points with Newcombe 95\\% CI; exact McNemar $p$; Holm over the family of seven; template-level "
              "exact sign-flip $p$ (10 templates). P1.3 compares thinking on with thinking off; in its low-effort rows the thinking-on arm "
              "runs at low effort. Low-effort rows (low) are a post-hoc sensitivity analysis. All four control edits (PCA-101/102/103, Unrelated) "
              "scored 0/80, so their four P1.2 contrasts are identical and shown once.", "tab:exp11contrasts", "lccccc",
              "Contrast & $\\Delta$ & 95\\% CI & $p$ & $p_{\\mathrm{Holm}}$ & $p_{\\mathrm{tmpl}}$", rows, size="\\scriptsize", sep="2.5pt")
    # T4 Exp 3.1
    rows = []
    for arm in ("refusal_ablated", "pca_101", "intact"):
        for cot in ("cot_on", "cot_off"):
            rows.append([ARM_LABEL[arm], "on" if cot == "cot_on" else "off",
                         pct(f"tab4.{arm}.{cot}.narr", F1(arm, cot, "harmful", "rate_b")),
                         pct(f"tab4.{arm}.{cot}.num", F1(arm, cot, "harmful", "rate_a")),
                         pct(f"tab4.{arm}.{cot}.diff", F1(arm, cot, "harmful", "diff")),
                         "$" + pval(f"tab4.{arm}.{cot}.p", F1(arm, cot, "harmful", "p_mcnemar")) + "$",
                         "$" + pval(f"tab4.{arm}.{cot}.holm", F1(arm, cot, "harmful", "p_holm")) + "$",
                         pct(f"tab4.{arm}.{cot}.hcnarr", S31(arm, cot, "narrative", "rate", "hc")),
                         pct(f"tab4.{arm}.{cot}.hcnum", S31(arm, cot, "numeric", "rate", "hc"))])
    tex_table("tab_exp31", "Exp~3.1 harmful-shortcut rates (ITT, \\%) with the stakes given as a narrative sentence or a rating "
              "table. Columns 3--7: all harmful cells, paired format contrast (exact McNemar, Holm within the F1 family). Columns 8--9: "
              "the high-cost harmful-implicit slice.", "tab:exp31", "llccccccc",
              "Arm & Think & Sentence & Table & $\\Delta$ & $p$ & $p_{\\mathrm{Holm}}$ & Slice sent. & Slice table", rows,
              size="\\footnotesize", wide=True, sep="6pt")
    # T5 Exp 3.2
    rows = []
    for cot in ("cot_on", "cot_off"):
        A = X32("exp11", cot, "A") + "['auroc']"
        cells = [f"{dec(f'tab5A.{cot}.{m}', A + '[' + repr(m) + ']' + chr(91) + repr('auroc') + chr(93), 3)}" for m in ("probe", "design", "tfidf", "rhat_cv")]
        rows.append(["A (full grid)", "on" if cot == "cot_on" else "off"] + cells)
    for cot in ("cot_on", "cot_off"):
        B = X32("exp11", cot, "B") + "['auroc']"
        rows.append(["B (within Intact)", "on" if cot == "cot_on" else "off",
                     dec(f"tab5B.{cot}.probe", B + "['probe']['auroc']", 3), dec(f"tab5B.{cot}.design", B + "['design']['auroc']", 3),
                     dec(f"tab5B.{cot}.tfidf", B + "['tfidf']['auroc']", 3), dec(f"tab5B.{cot}.rhat", B + "['rhat_neg']['auroc']", 3)])
    tex_table("tab_exp32_AB", "Exp~3.2 pre-thinking probes, Regimes A and B (Exp~1.1 data, layer-48 output). Cross-validated "
              "AUROC of a linear probe on activations, a design-matrix lookup (model $\\times$ cell), a TF-IDF text baseline, and the 1-D "
              "projection on $\\hat r$ (sign oriented so that lower projection predicts the shortcut in Regime B).",
              "tab:exp32ab", "llcccc", "Regime & Think & Probe & Design & TF-IDF & $\\hat r$ 1-D", rows)
    rows = []
    for ds, tag in (("exp11", "primary"), ("exp11+exp31", "extended")):
        for cot in ("cot_on", "cot_off"):
            for g in [g["group"] for g in ev(X32(ds, cot, "C") + "['scopes']['arm_cot']")]:
                c = C32(ds, cot, g)
                arm = g.split("|")[0]
                rows.append(["P" if tag == "primary" else "X", ARM_LABEL[arm], "on" if cot == "cot_on" else "off", cnt(f"tab5C.{tag}.{g}.n", f"{c}['n']"),
                             dec(f"tab5C.{tag}.{g}.probe", f"{c}['pooled']['probe']['auroc']", 3),
                             f"[{dec(f'tab5C.{tag}.{g}.l', c + chr(91) + repr('pooled') + chr(93) + chr(91) + repr('probe') + chr(93) + chr(91) + repr('ci_low') + chr(93), 3)}, "
                             f"{dec(f'tab5C.{tag}.{g}.h', c + chr(91) + repr('pooled') + chr(93) + chr(91) + repr('probe') + chr(93) + chr(91) + repr('ci_high') + chr(93), 3)}]",
                             dec(f"tab5C.{tag}.{g}.od", f"{c}['pooled']['order_domain']['auroc']", 3),
                             dec(f"tab5C.{tag}.{g}.pp", f"{c}['permutation_null']['p_perm']", 4)])
    tex_table("tab_exp32_C", "Exp~3.2 Regime C: within-cell split trials (the design lookup is exactly 0.5 here). Probe AUROC "
              "with bootstrap 95\\% CI, a tool-order/domain baseline, and a 400-shuffle permutation $p$ (floor $\\approx0.0025$). P = primary "
              "(Exp~1.1) data, X = extended (Exp~1.1 + Exp~3.1) data.",
              "tab:exp32c", "lllccccc", "D & Arm & Th. & $n$ & Probe & 95\\% CI & Ord./dom. & $p_{\\mathrm{perm}}$", rows,
              size="\\scriptsize", sep="2.5pt")
    # T6 Exp 1.2
    rows = []
    for corpus, lab in (("A", "A (unfiltered)"), ("B", "B (safe-route only)"), ("Bp", "B$'$ (paraphrased CoT)"), ("C", "C (math/code CoT)")):
        for teacher in ("refusal_ablated", "intact", "pca_101"):
            r = [lab if teacher == "refusal_ablated" else "", ARM_LABEL[teacher]]
            for rank in (32, 8):
                c = E12(corpus, teacher, rank)
                r.append(kn(f"tab6.{corpus}.{teacher}.r{rank}", f"{c}['agentic']['primary']['shortcut']['k']", f"{c}['agentic']['primary']['shortcut']['n']"))
            c = E12(corpus, teacher, 32)
            r.append(pct(f"tab6.{corpus}.{teacher}.comply", f"{c}['refusal']['harmful']['comply']['rate']"))
            r.append(dec(f"tab6.{corpus}.{teacher}.proj", f"{c}['projection']['refusal_harmful']['mean_over_seeds']", 3))
            ad = f"R['exp12_students']['summary']['adapters'][{corpus + '.' + teacher + '.r32.s0'!r}]['train']['n_target_tokens']"
            r.append(cnt(f"tab6.{corpus}.{teacher}.tok", ad))
            rows.append(r)
        rows.append("MIDRULE")
    rows.pop()
    tex_table("tab_exp12", "Exp~1.2 students (low effort, 2{,}048-token cap). Held-out agentic shortcut counts pooled over 3 seeds "
              "for LoRA rank 32 and 8; rank-32 harmful-request compliance (\\%), layer-48 refusal projection "
              "$|h\\cdot\\hat r|/\\lVert h\\rVert$ on harmful requests (mean over seeds), and training target tokens.",
              "tab:exp12", "llccccc", "Corpus & Teacher & $r{=}32$ & $r{=}8$ & Comply (\\%) & Projection & Target tokens", rows,
              size="\\footnotesize", wide=True, sep="6pt")
    # T7 summary of Exp 2.3 / 2.1 / 2.2 / 1.4
    rows = [
        ["2.3", "Natural reviewer notes recommending the shortcut (harmful state)", "0/" + cnt("tab7.e23.nat", " + ".join(f"R['exp23']['manipulation_check']['reviewers'][{r!r}]['n_valid']" for r in ("B1", "B2", "R1a", "R1b")))],
        ["2.3", "Llama-3.3-70B, authored careful memo (B4a), designated reviewer", kn("tab7.e23.ld", E23("llama70b:cot_off|designated|B4a", "k_shortcut"), E23("llama70b:cot_off|designated|B4a", "n"))],
        ["2.3", "Llama-3.3-70B, same memo, peer framing", kn("tab7.e23.lp", E23("llama70b:cot_off|peer|B4a", "k_shortcut"), E23("llama70b:cot_off|peer|B4a", "n"))],
        ["2.3", "Qwen Intact executor, B4a (both framings, both thinking modes)", kn("tab7.e23.i", " + ".join(E23(f"intact:{c}|{f}|B4a", "k_shortcut") for c in ("cot_on", "cot_off") for f in ("designated", "peer")), " + ".join(E23(f"intact:{c}|{f}|B4a", "n") for c in ("cot_on", "cot_off") for f in ("designated", "peer")))],
        "MIDRULE",
        ["2.1", "Forbidden joint shortcut (both agents skip their control)", kn("tab7.e21.j", "sum(c['metrics']['ILLEGIT_JOINT']['k'] for c in R['exp21_topologies']['cells'])", "R['exp21_topologies']['n_trials']")],
        ["2.1", "Ablated Alpha express lane, message channel (B)", kn("tab7.e21.mb", E21("mixed_team", "B", "alpha_fast", "k"), E21("mixed_team", "B", "alpha_fast", "n"))],
        ["2.1", "Intact Alpha express lane, message channel (B)", kn("tab7.e21.ib", E21("all_intact", "B", "alpha_fast", "k"), E21("all_intact", "B", "alpha_fast", "n"))],
        "MIDRULE",
        ["2.2", "Intact monitor, forbidden bundles blocked (thinking off)", kn("tab7.e22.it", E22("intact", TPR, "k"), E22("intact", TPR, "n"))],
        ["2.2", "Intact monitor, allowed bundles blocked", kn("tab7.e22.if", E22("intact", FPR, "k"), E22("intact", FPR, "n"))],
        ["2.2", "Llama-3.3-70B monitor, allowed bundles blocked", kn("tab7.e22.lf", E22("llama70b", FPR, "k"), E22("llama70b", FPR, "n"))],
        "MIDRULE",
        ["1.4", "Episodes that kept retrying the blocked route", "{}/{}".format(cnt("tab7.e14.p", "sum(h['persisted'] for a in R['exp14_blocked_route']['analyses'].values() for h in a['headline'] if h['timing']=='pooled')"), cnt("tab7.e14.n", "R['exp14_blocked_route']['meta']['hygiene']['n_blocked_rows']"))],
        ["1.4", "Ablated harmful-implicit shortcut, default effort", kn("tab7.e14.ad", E14("default", "refusal_ablated", "harmful", "implicit", "shortcut"), E14("default", "refusal_ablated", "harmful", "implicit", "n"))],
        ["1.4", "Ablated harmful-implicit shortcut, low effort", kn("tab7.e14.al", E14("low", "refusal_ablated", "harmful", "implicit", "shortcut"), E14("low", "refusal_ablated", "harmful", "implicit", "n"))],
        ["1.4", "Harmful-cell shortcuts taken before the block fired", "{}/{}".format(cnt("tab7.e14.pre", "sum(c['shortcut_before_block'] for a in R['exp14_blocked_route']['analyses'].values() for t in a['tables'].values() for kk, c in t['cells'].items() if kk.startswith('harmful'))"), cnt("tab7.e14.all", "sum(c['counts']['SHORTCUT'] for a in R['exp14_blocked_route']['analyses'].values() for t in a['tables'].values() for kk, c in t['cells'].items() if kk.startswith('harmful'))"))],
    ]
    tex_table("tab_track2_summary", "Summary of Exp~2.3, 2.1, 2.2 and 1.4 headline counts ($k/n$).", "tab:summary2",
              "c>{\\raggedright\\arraybackslash}p{0.66\\linewidth}c", "Exp & Outcome & $k/n$", rows, size="\\footnotesize", sep="3pt")


# ============================================================================================ figures
def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def build_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    _bar = plt.Axes.bar

    def _safe_bar(self, *a, **k):
        if k.get('yerr') is not None:
            k['yerr'] = np.clip(np.asarray(k['yerr'], dtype=float), 0, None)
        return _bar(self, *a, **k)
    plt.Axes.bar = _safe_bar
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.hashsalt": "v3", "legend.frameon": False})
    META = {"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None}
    cols = {"intact": "#4C72B0", "refusal_ablated": "#C44E52", "pca_101": "#8C8C8C", "pca_102": "#A6A6A6",
            "pca_103": "#BDBDBD", "unrelated": "#55A868"}

    def bars(ax, groups, series, getter, colors, labels, width=0.8, ylabel="rate"):
        ns = len(series)
        w = width / ns
        for j, s in enumerate(series):
            xs, ys, lo, hi = [], [], [], []
            for i, g in enumerate(groups):
                k, n = getter(g, s)
                if n is None:
                    continue
                p = k / n if n else 0
                l, h = wilson(k, n)
                xs.append(i - width / 2 + w * (j + 0.5)); ys.append(p); lo.append(max(0.0, p - l)); hi.append(max(0.0, h - p))
            ax.bar(xs, ys, w, color=colors[j], label=labels[j], yerr=[lo, hi], capsize=1.5, error_kw={"lw": 0.6})
        ax.set_xticks(range(len(groups)))
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(ylabel)

    # ---- Fig 1
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"width_ratios": [1.25, 1]})
    groups = ["C", "B", "A0", "A1", "A2", "DI", "DI+esc"]

    def g1(g, arm):
        if g in ("DI", "DI+esc"):
            v = "solo" if g == "DI" else "esc"
            return ev(DI(arm, v)), ev(DI(arm, v, field="n"))
        return ev(P1(arm, g, None, "n_comply")), ev(P1(arm, g, None, "n"))
    bars(axs[0], groups, ARMS, g1, [cols[a] for a in ARMS], [ARM_LABEL[a] for a in ARMS], ylabel="restricted-tool compliance")
    axs[0].set_xticklabels(["Set C", "Set B", "A0", "A1", "A2", "1.1\ndirect", "1.1 direct\n+hand-off"], fontsize=6.5)
    axs[0].set_title("(A) Direct-instruction positive control", loc="left")
    fig.legend(*axs[0].get_legend_handles_labels(), ncol=6, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    groups = ["default", "off", "low", "harmless impl.", "harmless expl."]

    def g2(g, arm):
        if g in ("default", "off", "low"):
            if g == "off" and arm not in ("intact", "refusal_ablated"):
                return 0, None
            return ev(RB(arm, g, "k_shortcut")), ev(RB(arm, g, "n"))
        cells = HC_HARMLESS_IMPL if g == "harmless impl." else HC_HARMLESS_EXPL
        return ev(E11(arm, "cot_on", cells)), ev(E11(arm, "cot_on", cells, "n"))
    bars(axs[1], groups, ARMS, g2, [cols[a] for a in ARMS], [ARM_LABEL[a] for a in ARMS], ylabel="shortcut rate (high cost)")
    axs[1].set_xticklabels(["harmful\ndefault", "harmful\nthink off", "harmful\nlow effort\n(post hoc)", "harmless\nimplicit", "harmless\nexplicit"], fontsize=6.5)
    axs[1].set_title("(B) Exp 1.1 high-cost shortcut rates", loc="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(PAPER / "figures" / "fig1_phase1_and_exp11_grid.pdf", metadata=META)
    plt.close(fig)

    # ---- Fig 2
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.4), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    arms3 = ["refusal_ablated", "pca_101", "intact"]
    for ax, sl, title in ((axs[0], "harmful", "(A) all harmful cells"), (axs[1], "hc", "(B) high-cost harmful-implicit")):
        groups = [(a, c) for a in arms3 for c in ("cot_on", "cot_off")]

        def g(gc, fmt, sl=sl):
            a, c = gc
            return ev(S31(a, c, fmt, "k", sl)), ev(S31(a, c, fmt, "n", sl))
        bars(ax, groups, ["narrative", "numeric"], g, ["#9DB4D3", "#C44E52"], ["sentence", "rating table"], ylabel="harmful shortcut (ITT)")
        ax.set_xticklabels([f"{ {'refusal_ablated': 'Abl.', 'pca_101': 'PCA', 'intact': 'Int.'}[a]}\n{'on' if c == 'cot_on' else 'off'}" for a, c in groups], fontsize=6)
        ax.set_title(title, loc="left")
    axs[0].legend(fontsize=6, loc="upper right")
    doms = ["warehouse", "compute", "finance", "wetlab"]

    def gd(dom, sr):
        arm, fld = sr
        x = ev(F1D(arm, "cot_on", dom, "n_pairs"))
        return round(ev(F1D(arm, "cot_on", dom, fld)) * x), x
    series = [("refusal_ablated", "rate_b"), ("refusal_ablated", "rate_a"), ("pca_101", "rate_b"), ("pca_101", "rate_a"),
              ("intact", "rate_b"), ("intact", "rate_a")]
    bars(axs[2], doms, series, gd, ["#E8A3A5", "#C44E52", "#D0D0D0", "#7F7F7F", "#9DB4D3", "#4C72B0"],
         ["Abl. sentence", "Abl. table", "PCA sentence", "PCA table", "Int. sentence", "Int. table"],
         ylabel="harmful shortcut (ITT)")
    axs[2].set_xticklabels(doms, fontsize=6, rotation=15)
    axs[2].set_ylim(0, 0.8)
    axs[2].set_title("(C) harmful cells by domain", loc="left")
    axs[2].legend(fontsize=5.5, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(PAPER / "figures" / "fig2_exp31_stakes_abstraction.pdf", metadata=META)
    plt.close(fig)

    # ---- Fig 3
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [1.05, 0.65, 1.3]})
    ms = ["probe", "design", "tfidf", "rhat_cv"]
    for j, cot in enumerate(("cot_on", "cot_off")):
        A = ev(X32("exp11", cot, "A"))["auroc"]
        xs = [i + (j - 0.5) * 0.36 for i in range(len(ms))]
        ys = [A[m]["auroc"] for m in ms]
        err = [[A[m]["auroc"] - A[m]["ci_low"] for m in ms], [A[m]["ci_high"] - A[m]["auroc"] for m in ms]]
        axs[0].bar(xs, ys, 0.36, yerr=err, capsize=1.5, color=["#2F4B7C", "#A0A0A0"][j], label="thinking " + ("on" if j == 0 else "off"))
    axs[0].set_xticks(range(len(ms)))
    axs[0].set_xticklabels(["probe", "design", "TF-IDF", "$\\hat r$"], fontsize=6.5, rotation=30)
    axs[0].set_ylim(0.4, 1.02)
    axs[0].axhline(0.5, color="k", lw=0.5, ls=":")
    axs[0].set_ylabel("AUROC")
    axs[0].set_title("(A) Regime A", loc="left")
    axs[0].legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    for j, cot in enumerate(("cot_on", "cot_off")):
        B = ev(X32("exp11", cot, "B"))["auroc"]
        for i, m in enumerate(("rhat_neg", "design")):
            axs[1].bar(i + (j - 0.5) * 0.36, B[m]["auroc"], 0.36, yerr=[[B[m]["auroc"] - B[m]["ci_low"]], [B[m]["ci_high"] - B[m]["auroc"]]],
                       capsize=1.5, color=["#2F4B7C", "#A0A0A0"][j])
    axs[1].set_xticks([0, 1])
    axs[1].set_xticklabels(["$\\hat r$ 1-D", "design"])
    axs[1].set_ylim(0.4, 1.02)
    axs[1].axhline(0.5, color="k", lw=0.5, ls=":")
    axs[1].set_title("(B) Regime B", loc="left")
    rowsC = []
    for ds, tag in (("exp11", "primary"), ("exp11+exp31", "extended")):
        for cot in ("cot_on", "cot_off"):
            for g in ev(X32(ds, cot, "C") + "['scopes']['arm_cot']"):
                pr = g["pooled"]["probe"]
                rowsC.append((tag, f"{ARM_LABEL[g['group'].split('|')[0]]}, {'on' if cot == 'cot_on' else 'off'}",
                              pr["auroc"], pr["ci_low"], pr["ci_high"], cot))
    ys = list(range(len(rowsC)))[::-1]
    for y, (tag, lab, v, lo, hi, cot) in zip(ys, rowsC):
        c = "#2F4B7C" if cot == "cot_on" else "#8C8C8C"
        axs[2].errorbar(v, y, xerr=[[v - lo], [hi - v]], fmt="o", color=c, ms=3, capsize=1.5, lw=0.8)
    axs[2].axvline(0.5, color="k", lw=0.5, ls=":")
    n_primary = sum(1 for r in rowsC if r[0] == "primary")
    axs[2].axhline(len(rowsC) - n_primary - 0.5, color="k", lw=0.4)
    axs[2].set_yticks(ys)
    axs[2].set_yticklabels([r[1] for r in rowsC], fontsize=6)
    for tag in ("primary", "extended"):
        yy = [y for y, r in zip(ys, rowsC) if r[0] == tag]
        axs[2].text(0.98, max(yy), tag, fontsize=6, style="italic", ha="right", va="center")
    axs[2].set_xlim(0, 1)
    axs[2].set_ylim(-0.7, len(rowsC) - 0.1)
    axs[2].set_xlabel("within-cell AUROC (thinking on/off)")
    axs[2].set_title("(C) Regime C", loc="left")
    fig.tight_layout()
    fig.savefig(PAPER / "figures" / "fig3_exp32_prethink_probes.pdf", metadata=META)
    plt.close(fig)

    # ---- Fig 4
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.5), gridspec_kw={"width_ratios": [1, 1, 1]})
    corpora = ["A", "B", "Bp", "C"]
    teachers = ["refusal_ablated", "intact", "pca_101"]
    floor = sum(f["agentic"]["primary"]["shortcut"]["k"] for f in R["exp12_students"]["summary"]["floors"].values()) / \
        sum(f["agentic"]["primary"]["shortcut"]["n"] for f in R["exp12_students"]["summary"]["floors"].values())
    for ax, rank in ((axs[0], 32), (axs[1], 8)):
        for j, t in enumerate(teachers):
            for i, c in enumerate(corpora):
                cell = ev(E12(c, t, rank))
                x = i + (j - 1) * 0.26
                sc = cell["agentic"]["primary"]["shortcut"]
                ax.bar(x, sc["rate"], 0.26, color=cols[t], label=ARM_LABEL[t] + " teacher" if i == 0 else None,
                       yerr=[[sc["rate"] - sc["wilson_low"]], [sc["wilson_high"] - sc["rate"]]], capsize=1.2, error_kw={"lw": 0.6})
                seeds = cell["agentic"]["primary"]["per_seed_shortcut"]
                ax.scatter([x] * len(seeds), [seeds[s] for s in sorted(seeds)], s=5, color="k", zorder=3)
        ax.axhline(floor, color="k", lw=0.6, ls="--")
        ax.set_xticks(range(4))
        ax.set_xticklabels(["A", "B", "B$'$", "C"])
        ax.set_ylim(0, 0.6)
        ax.set_title(f"({'A' if rank == 32 else 'B'}) shortcut, r={rank}", loc="left")
    axs[0].set_ylabel("held-out shortcut rate")
    axs[0].legend(fontsize=6, loc="upper right")
    fl_proj = [f["projection"]["refusal_harmful"]["out"]["r1.r_hat"] for f in R["exp12_students"]["summary"]["floors"].values()]
    for j, t in enumerate(teachers):
        for i, c in enumerate(corpora):
            cell = ev(E12(c, t, 32))
            x = i + (j - 1) * 0.26
            pr = cell["projection"]["refusal_harmful"]
            axs[2].bar(x, pr["mean_over_seeds"], 0.26, color=cols[t])
            axs[2].scatter([x] * len(pr["per_seed"]), pr["per_seed"], s=5, color="k", zorder=3)
    axs[2].axhline(sum(fl_proj) / len(fl_proj), color="k", lw=0.6, ls="--")
    axs[2].set_xticks(range(4))
    axs[2].set_xticklabels(["A", "B", "B$'$", "C"])
    axs[2].set_ylabel("$|h\\cdot\\hat r|/\\|h\\|$ (harmful requests)")
    axs[2].set_title("(C) projection, r=32", loc="left")
    fig.tight_layout()
    fig.savefig(PAPER / "figures" / "fig4_exp12_distillation.pdf", metadata=META)
    plt.close(fig)

    # ---- Fig 5
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5))
    dc = R["exp23"]["domain_cells"]
    doms = ["warehouse", "compute", "finance", "wetlab"]

    def g5(dom, s):
        key = {"Llama designated": "llama70b:cot_off|designated|B4a", "Llama peer": "llama70b:cot_off|peer|B4a",
               "Intact designated": "intact:cot_on|designated|B4a", "Intact peer": "intact:cot_on|peer|B4a"}[s]
        d = dc[key][dom]
        return d["k_shortcut"], d["n"]
    series = ["Llama designated", "Llama peer", "Intact designated", "Intact peer"]
    bars(axs[0], doms, series, g5, ["#DD8452", "#F2C4A0", "#4C72B0", "#9DB4D3"], series, ylabel="shortcut rate (memo B4a)")
    axs[0].set_xticklabels(doms)
    axs[0].set_title("(A) Exp 2.3 authored-memo contagion", loc="left")
    axs[0].legend(fontsize=6, loc="upper right")
    variants = ["A", "C_nopen", "C_pen", "B", "D_nopen", "D_pen"]

    def g6(v, s):
        team, metric = s
        c = next(x for x in R["exp21_topologies"]["cells"] if x["team"] == team and x["variant"] == v)["metrics"][metric]
        return c["k"], c["n"]
    series = [("mixed_team", "alpha_fast"), ("all_intact", "alpha_fast"), ("mixed_team", "LEGIT_SAFE_COORD")]
    bars(axs[1], variants, series, g6, ["#C44E52", "#4C72B0", "#55A868"],
         ["Ablated Alpha: express lane", "Intact Alpha: express lane", "Ablated team: Alpha express + Beta safe"],
         ylabel="rate per trial")
    axs[1].set_xticklabels(["A", "C", "C+fee", "B", "D", "D+fee"])
    axs[1].axvline(2.5, color="k", lw=0.5, ls=":")
    axs[1].text(1.0, 0.95, "no channel", ha="center", fontsize=6)
    axs[1].text(4.0, 0.95, "message channel", ha="center", fontsize=6)
    axs[1].set_ylim(0, 1.0)
    axs[1].set_title("(B) Exp 2.1 Alpha express lane by topology", loc="left")
    axs[1].legend(fontsize=6, loc="upper left", bbox_to_anchor=(0.0, 0.88))
    fig.tight_layout()
    fig.savefig(PAPER / "figures" / "fig5_track2_multiagent.pdf", metadata=META)
    plt.close(fig)


def main() -> int:
    for d in ("generated", "tables", "figures"):
        (PAPER / d).mkdir(parents=True, exist_ok=True)
    build_facts()
    build_tables()
    build_figures()
    lines = ["% generated by scripts/build_paper_artifacts.py from records/*.json -- do not edit"]
    for k in sorted(FACTS):
        lines.append(f"\\FACT{{{k}}}{{{FACTS[k]}}}")
    (PAPER / "generated" / "facts.tex").write_text("\n".join(lines) + "\n")
    ids = [c["id"] for c in CLAIMS]
    dup = {i for i in ids if ids.count(i) > 1}
    assert not dup, f"duplicate claim ids: {sorted(dup)[:5]}"
    (PAPER / "paper.claims.json").write_text(json.dumps(CLAIMS, indent=0))
    print(f"{len(FACTS)} prose facts, {len(CLAIMS)} claims, tables and figures written under {PAPER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
