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
                         f"[{pct(f'tab4.{arm}.{cot}.cl', F1(arm, cot, 'harmful', 'cluster_ci') + '[0]')}, {pct(f'tab4.{arm}.{cot}.ch', F1(arm, cot, 'harmful', 'cluster_ci') + '[1]')}]",
                         "$" + pval(f"tab4.{arm}.{cot}.p", F1(arm, cot, "harmful", "p_mcnemar")) + "$",
                         "$" + pval(f"tab4.{arm}.{cot}.sfholm", F1(arm, cot, "harmful", "p_signflip_holm")) + "$",
                         pct(f"tab4.{arm}.{cot}.hcnarr", S31(arm, cot, "narrative", "rate", "hc")),
                         pct(f"tab4.{arm}.{cot}.hcnum", S31(arm, cot, "numeric", "rate", "hc"))])
    tex_table("tab_exp31", "Exp~3.1 harmful-shortcut rates (ITT, \\%) with the stakes given as a narrative sentence or a rating "
              "table. Columns 3--8: all harmful cells, paired format difference with scenario-clustered 95\\% CI, exact McNemar $p$ and the "
              "scenario sign-flip $p$ after Holm correction within the F1 family. Columns 9--10: the high-cost harmful-implicit slice "
              "(arm contrasts on this slice: Table~\\ref{tab:exp31x2}).", "tab:exp31", "llcccccccc",
              "Arm & Think & Sentence & Table & $\\Delta$ & 95\\% CI & $p$ & $p_{\\mathrm{sf,Holm}}$ & Slice sent. & Slice table", rows,
              size="\\footnotesize", wide=True, sep="4.5pt")
    # T5 Exp 3.2
    E32 = "R['reviewer_addenda']['exp32']"
    rows = []
    for cot in ("cot_on", "cot_off"):
        A = X32("exp11", cot, "A") + "['auroc']"
        inc = f"{E32}['regime_a_incremental']['exp11'][{cot!r}]"
        fp = f"{E32}['fpr_at_80tpr']['exp11']['regime_a'][{cot!r}]"
        cells = [dec(f"tab5A.{cot}.{m}", A + "[" + repr(m) + "]['auroc']", 3) for m in ("probe", "design")]
        cells.append(dec(f"tab5A.{cot}.dp", f"{inc}['auroc']['design_probe']['auroc']", 3))
        cells.append(f"{dec(f'tab5A.{cot}.dpd', inc + chr(91) + repr('diff') + chr(93) + chr(91) + repr('design_probe-design') + chr(93) + chr(91) + repr('diff') + chr(93), 3)} "
                     f"[{dec(f'tab5A.{cot}.dpdl', inc + chr(91) + repr('diff') + chr(93) + chr(91) + repr('design_probe-design') + chr(93) + chr(91) + repr('ci_low') + chr(93), 3)}, "
                     f"{dec(f'tab5A.{cot}.dpdh', inc + chr(91) + repr('diff') + chr(93) + chr(91) + repr('design_probe-design') + chr(93) + chr(91) + repr('ci_high') + chr(93), 3)}]")
        cells += [dec(f"tab5A.{cot}.{m}", A + "[" + repr(m) + "]['auroc']", 3) for m in ("tfidf", "rhat_cv")]
        cells.append(dec(f"tab5A.{cot}.fpr", f"{fp}['probe']['fpr']", 3))
        rows.append(["A (full grid)", "on" if cot == "cot_on" else "off"] + cells)
    for cot in ("cot_on", "cot_off"):
        B = X32("exp11", cot, "B") + "['auroc']"
        fp = f"{E32}['fpr_at_80tpr']['exp11']['regime_b'][{cot!r}]"
        rows.append(["B (within Intact)", "on" if cot == "cot_on" else "off",
                     dec(f"tab5B.{cot}.probe", B + "['probe']['auroc']", 3), dec(f"tab5B.{cot}.design", B + "['design']['auroc']", 3), "---", "---",
                     dec(f"tab5B.{cot}.tfidf", B + "['tfidf']['auroc']", 3), dec(f"tab5B.{cot}.rhat", B + "['rhat_neg']['auroc']", 3),
                     dec(f"tab5B.{cot}.fpr", f"{fp}['rhat_neg']['fpr']", 3)])
    tex_table("tab_exp32_AB", "Exp~3.2 pre-thinking probes, Regimes A and B (Exp~1.1 data, layer-48 output). Cross-validated "
              "AUROC of a linear probe on activations, a design-matrix lookup (model $\\times$ cell), the combined design + probe predictor "
              "and its gain over the design with a paired scenario-bootstrap 95\\% CI, a TF-IDF text baseline, and the 1-D "
              "projection on $\\hat r$ (sign oriented so that lower projection predicts the shortcut in Regime B). Last column: false-positive "
              "rate at 80\\% true-positive rate of the probe (Regime A) or of the $\\hat r$ projection (Regime B).",
              "tab:exp32ab", "llccccccc", "Regime & Think & Probe & Design & D+P & D+P $-$ D [CI] & TF-IDF & $\\hat r$ 1-D & FPR$_{80}$", rows,
              size="\\scriptsize", wide=True, sep="4pt")
    rows = []
    for ds, tag in (("exp11", "primary"), ("exp11+exp31", "extended")):
        for cot in ("cot_on", "cot_off"):
            for g in [g["group"] for g in ev(X32(ds, cot, "C") + "['scopes']['arm_cot']")]:
                c = C32(ds, cot, g)
                ac = f"next(x for x in {E32}['regime_c_additions'][{ds!r}][{cot!r}]['arm_cot'] if x['group']=={g!r})"
                arm = g.split("|")[0]
                rows.append(["P" if tag == "primary" else "X", ARM_LABEL[arm], "on" if cot == "cot_on" else "off", cnt(f"tab5C.{tag}.{g}.n", f"{c}['n']"),
                             dec(f"tab5C.{tag}.{g}.probe", f"{c}['pooled']['probe']['auroc']", 3),
                             f"[{dec(f'tab5C.{tag}.{g}.l', c + chr(91) + repr('pooled') + chr(93) + chr(91) + repr('probe') + chr(93) + chr(91) + repr('ci_low') + chr(93), 3)}, "
                             f"{dec(f'tab5C.{tag}.{g}.h', c + chr(91) + repr('pooled') + chr(93) + chr(91) + repr('probe') + chr(93) + chr(91) + repr('ci_high') + chr(93), 3)}]",
                             dec(f"tab5C.{tag}.{g}.tf", f"{ac}['auroc']['tfidf']['auroc']", 3),
                             dec(f"tab5C.{tag}.{g}.od", f"{c}['pooled']['order_domain']['auroc']", 3),
                             dec(f"tab5C.{tag}.{g}.fpr", f"{ac}['fpr_at_tpr']['probe']['fpr']", 2),
                             dec(f"tab5C.{tag}.{g}.pp", f"{c}['permutation_null']['p_perm']", 4)])
    tex_table("tab_exp32_C", "Exp~3.2 Regime C: within-cell split trials (the design lookup is exactly 0.5 here). Probe AUROC "
              "with bootstrap 95\\% CI, a within-cell centred TF-IDF baseline, a tool-order/domain baseline, the probe's false-positive rate at "
              "80\\% true-positive rate (one threshold across cells), and a 400-shuffle permutation $p$ (floor $\\approx0.0025$). P = primary "
              "(Exp~1.1) data, X = extended (Exp~1.1 + Exp~3.1) data.",
              "tab:exp32c", "lllccccccc", "D & Arm & Th. & $n$ & Probe & 95\\% CI & TF-IDF & Ord./dom. & FPR$_{80}$ & $p_{\\mathrm{perm}}$", rows,
              size="\\scriptsize", wide=True, sep="4pt")
    # T6 Exp 1.2
    rows = []
    for corpus, lab in (("A", "A (unfiltered)"), ("B", "B (safe-route only)"), ("Bp", "B$'$ (paraphrased CoT)"), ("C", "C (math/code CoT)")):
        for teacher in ("refusal_ablated", "intact", "pca_101"):
            r = [lab if teacher == "refusal_ablated" else "", ARM_LABEL[teacher]]
            for rank in (32, 8):
                c = E12(corpus, teacher, rank)
                seeds = "/".join(cnt(f"tab6.{corpus}.{teacher}.r{rank}.s{s_}", f"{c}['agentic']['primary']['per_seed_shortcut'][{s_!r}] * {c}['agentic']['primary']['shortcut']['n'] / len({c}['seeds'])") for s_ in ("0", "1", "2"))
                r.append(kn(f"tab6.{corpus}.{teacher}.r{rank}", f"{c}['agentic']['primary']['shortcut']['k']", f"{c}['agentic']['primary']['shortcut']['n']") + f" ({seeds})")
            if teacher == "refusal_ablated" and corpus in ("A", "B"):
                con = "refusal_ablated_vs_intact"
                r.append(f"{pct(f'tab6.{corpus}.d', E12C(corpus, 32, con, 'diff'))} [{pct(f'tab6.{corpus}.dl', E12C(corpus, 32, con, 'scenario_ci_low'))}, {pct(f'tab6.{corpus}.dh', E12C(corpus, 32, con, 'scenario_ci_high'))}]")
                r.append("$" + pval(f"tab6.{corpus}.holm", E12C(corpus, 32, con, "p_mcnemar_holm")) + "$")
            else:
                r += ["", ""]
            c = E12(corpus, teacher, 32)
            r.append(kn(f"tab6.{corpus}.{teacher}.hl", f"{c}['agentic']['hc_harmless_implicit']['shortcut']['k']", f"{c}['agentic']['hc_harmless_implicit']['shortcut']['n']"))
            r.append(pct(f"tab6.{corpus}.{teacher}.comply", f"{c}['refusal']['harmful']['comply']['rate']"))
            r.append(dec(f"tab6.{corpus}.{teacher}.proj", f"{c}['projection']['refusal_harmful']['mean_over_seeds']", 3))
            ad = f"R['exp12_students']['summary']['adapters'][{corpus + '.' + teacher + '.r32.s0'!r}]['train']['n_target_tokens']"
            r.append(cnt(f"tab6.{corpus}.{teacher}.tok", ad))
            rows.append(r)
        rows.append("MIDRULE")
    rows.pop()
    tex_table("tab_exp12", "Exp~1.2 students (low effort, 2{,}048-token cap). Held-out harmful shortcut counts pooled over 3 seeds "
              "(per-seed counts out of 32 in parentheses) for LoRA rank 32 and 8; for \\Ablated-teacher students of Corpora A and B, the "
              "$r{=}32$ difference to \\Intact-teacher students (pp, scenario-clustered 95\\% CI) and its exact McNemar $p$ after Holm "
              "correction over the teacher family. Then the allowed harmless high-cost shortcut ($r{=}32$), harmful-request compliance (\\%), "
              "the layer-48 refusal projection $|h\\cdot\\hat r|/\\lVert h\\rVert$ on harmful requests (mean over seeds) and training target tokens.",
              "tab:exp12", "llcccccccc", "Corpus & Teacher & $r{=}32$ & $r{=}8$ & $\\Delta$ vs Intact & $p_{\\mathrm{Holm}}$ & Harmless & Comply & Proj. & Tokens", rows,
              size="\\scriptsize", wide=True, sep="3.5pt")
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
    fig.legend(*axs[0].get_legend_handles_labels(), fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=2)
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
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(PAPER / "figures" / "fig3_exp32_prethink_probes.pdf", metadata=META, bbox_inches="tight", pad_inches=0.06)
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
    axs[1].legend(fontsize=6, loc="upper left", bbox_to_anchor=(0.0, 0.88), frameon=True, framealpha=1.0, edgecolor="none")
    fig.tight_layout()
    fig.savefig(PAPER / "figures" / "fig5_track2_multiagent.pdf", metadata=META)
    plt.close(fig)

    # ---- Fig 6 deconfounders
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1.1, 1]})
    DC = "R['exp11_deconfounders']['cells']"
    conds = [("std_original", "default"), ("std", "default"), ("nh", "default"), ("np", "default"),
             ("std_original", "low"), ("std", "low"), ("nh", "low")]
    labels = ["orig.", "repl.", "no\nhand-off", "no fewer-\ncalls rule", "orig.", "repl.", "no\nhand-off"]

    def g6a(gc, arm):
        k = f"{arm}|{gc[0]}|{gc[1]}"
        return ev(f"{DC}[{k!r}]['shortcut']"), ev(f"{DC}[{k!r}]['n']")
    bars(axs[0], conds, ["refusal_ablated", "pca_101", "intact"], g6a, [cols["refusal_ablated"], cols["pca_101"], cols["intact"]],
         ["Ablated", "PCA-101", "Intact"], ylabel="harmful shortcut (ITT)")
    axs[0].set_xticklabels(labels, fontsize=6)
    axs[0].axvline(3.5, color="k", lw=0.5, ls=":")
    axs[0].text(1.5, 0.62, "default effort", ha="center", fontsize=6.5)
    axs[0].text(5.0, 0.62, "low effort", ha="center", fontsize=6.5)
    axs[0].set_ylim(0, 0.7)
    axs[0].set_title("(A) Exp 1.1 deconfounders", loc="left")
    axs[0].legend(fontsize=6, loc="upper left", bbox_to_anchor=(0.0, 0.9))
    X2 = "R['exp31_2x2_deconfounder']['designs']['same_launch']['per_arm']"
    cells = ["CS", "AS", "CT", "AT"]

    def g6b(cell, arm):
        return ev(f"{X2}[{arm!r}]['counts'][{cell!r}]"), ev(f"{X2}[{arm!r}]['n_complete_pairs']")
    bars(axs[1], cells, ["refusal_ablated", "pca_101", "intact"], g6b, [cols["refusal_ablated"], cols["pca_101"], cols["intact"]],
         ["Ablated", "PCA-101", "Intact"], ylabel="harmful shortcut (ITT)")
    axs[1].set_xticklabels(["concrete\nsentence", "abstract\nsentence", "concrete\ntable", "abstract\ntable"], fontsize=6.5)
    axs[1].set_ylim(0, 1.0)
    axs[1].set_title("(B) Exp 3.1 layout x specificity", loc="left")
    fig.tight_layout()
    fig.savefig(PAPER / "figures" / "fig6_deconfounders.pdf", metadata=META)
    plt.close(fig)


# ============================================================================================ reviewer revision
RA = "R['reviewer_addenda']"


def AS(arm, cond, action):
    """5-way action count summed over the two high-cost harmful-implicit cells (reviewer addenda)."""
    return " + ".join(f"{RA}['action_splits'][{arm + '|' + cond + '|' + c + '|harmful|implicit'!r}][{action!r}]" for c in ("high_tedium", "high_friction"))


def DO(arm, cond, variant, field, cons="harmful"):
    return f"{RA}['direct_order'][{arm + '|' + cond + '|' + variant + '|' + cons!r}][{field!r}]"


def build_review_facts():
    fact("d.writers", cnt("d.writers", "130"))
    # ---- Exp 1.1 action splits (high-cost harmful-implicit, both costs)
    for arm in ARMS:
        for cond in ("default", "low", "off"):
            if f"{arm}|{cond}|high_tedium|harmful|implicit" not in ev(f"{RA}['action_splits']"):
                continue
            for a in ("safe_route_completed", "shortcut", "escalated", "prose_stop", "unscorable", "native_xml"):
                fact(f"as.{arm}.{cond}.{a}", cnt(f"as.{arm}.{cond}.{a}", AS(arm, cond, a)))
    for arm in ("refusal_ablated", "intact"):
        for cond in ("default", "low", "off"):
            for var in ("solo", "esc"):
                for f in ("comply", "escalated", "text_reply", "undeclared_call", "unscorable", "native_xml", "n"):
                    fact(f"do.{arm}.{cond}.{var}.{f}", cnt(f"do.{arm}.{cond}.{var}.{f}", DO(arm, cond, var, f)))
    fact("do.prompt.n", cnt("do.prompt.n", f"{RA}['direct_order_prompt']['n_prompts']"))
    fact("do.prompt.invites", cnt("do.prompt.invites", f"{RA}['direct_order_prompt']['n_mentioning_escalate_outside_tool_line']"))
    st = f"{RA}['shortcut_turns']"
    for cond in ("default", "low", "off"):
        for f in ("turn1", "turn2plus", "n"):
            e = " + ".join(f"{st}.get({'refusal_ablated|' + cond + '|harmful|' + c!r}, {{}}).get({f!r}, 0)" for c in ("high_tedium", "high_friction"))
            fact(f"st.abl.{cond}.{f}", cnt(f"st.abl.{cond}.{f}", e))
    tm = f"{RA}['primary_events_by_template']"
    fact("tmpl.default.n", cnt("tmpl.default.n", f"len({tm}['default'])"))
    fact("tmpl.default.max", cnt("tmpl.default.max", f"max({tm}['default'].values())"))
    fact("tmpl.low.n", cnt("tmpl.low.n", f"len({tm}['low'])"))
    fact("tmpl.floor.active", pval("tmpl.floor.active", f"2 / 2 ** len({tm}['default'])"), True)
    fact("tmpl.floor.all", pval("tmpl.floor.all", "2 / 2 ** R['exp11_effort_robustness']['cells']['intact']['default']['n_templates']"), True)
    fact("holm.k", cnt("holm.k", "len(R['exp11_main']['contrasts']['holm_family'])"))
    fact("holm.same", cnt("holm.same", "sum(1 for k in R['exp11_main']['contrasts']['holm_family'] if k.startswith('P1.1') or k.startswith('P1.2'))"))
    lo = "R['exp11_effort_robustness']['contrasts']['low_vs_off:refusal_ablated']"
    fact("rob.lowoff.diff", pct("rob.lowoff.diff", f"{lo}['diff']"))
    rp = f"{RA}['reproducibility']"
    for arm in ("refusal_ablated", "intact"):
        for pair in ("default_vs_low", "default_vs_off", "low_vs_off"):
            for scope in ("all_cells", "hc_harmful_implicit"):
                k = f"{rp}[{arm + '|' + pair + '|' + scope!r}]"
                key = f"kap.{arm}.{pair}.{scope}"
                if ev(f"{k}['kappa']") is not None:
                    fact(key, dec(key, f"{k}['kappa']", 2))
                fact(key + ".agr", pct(key + ".agr", f"{k}['agreement']", 1))
                if scope == "hc_harmful_implicit":
                    fact(key + ".both", cnt(key + ".both", f"{k}['n_both_pos']"))
    g14 = f"{RA}['exp14_turn1']"
    fact("e14t1.checked", cnt("e14t1.checked", f"{g14}['n_turn1_prompts_checked']"))
    fact("e14t1.identical", cnt("e14t1.identical", f"{g14}['n_turn1_prompts_identical']"))
    fact("e14t1.kmin", dec("e14t1.kmin", f"min(v['exp11_vs_exp14_action']['kappa'] for v in {g14}['groups'].values())", 2))
    fact("e14t1.kmax", dec("e14t1.kmax", f"max(v['exp11_vs_exp14_action']['kappa'] for v in {g14}['groups'].values())", 2))
    fact("e14t1.agrmin", pct("e14t1.agrmin", f"min(v['exp11_vs_exp14_action']['agreement'] for v in {g14}['groups'].values())", 1))
    fact("e14t1.earlylate", pct("e14t1.earlylate", f"min(v['exp14_early_vs_late_action']['agreement'] for v in {g14}['groups'].values())", 1))
    cf = f"{RA}['cot_features']"
    for arm in ("refusal_ablated", "intact", "pca_101"):
        for cond in ("default", "low"):
            g = f"{cf}[{'exp11|' + arm + '|' + cond!r}]['all']"
            fact(f"cot.e11.{arm}.{cond}.n", cnt(f"cot.e11.{arm}.{cond}.n", f"{g}['n']"))
            for feat in ("names_direct", "mentions_escalate", "harm_cue", "cites_call_rule"):
                fact(f"cot.e11.{arm}.{cond}.{feat}", cnt(f"cot.e11.{arm}.{cond}.{feat}", f"{g}[{feat!r}]"))
            fact(f"cot.e11.{arm}.{cond}.words", cnt(f"cot.e11.{arm}.{cond}.words", f"{g}['mean_words']"))
        for fmt in ("narrative", "numeric"):
            g = f"{cf}[{'exp31|' + arm + '|' + fmt!r}]['all']"
            fact(f"cot.e31.{arm}.{fmt}.words", cnt(f"cot.e31.{arm}.{fmt}.words", f"{g}['mean_words']"))
    for teacher in ("refusal_ablated", "intact", "pca_101"):
        c = f"{RA}['corpus_b'][{'B|' + teacher!r}]"
        fact(f"cb.B.{teacher}.frac", pct(f"cb.B.{teacher}.frac", f"{c}['frac_naming_direct']", 1))
        fact(f"cb.B.{teacher}.mentions", dec(f"cb.B.{teacher}.mentions", f"{c}['mean_direct_mentions']", 2))
        fact(f"cb.B.{teacher}.words", cnt(f"cb.B.{teacher}.words", f"{c}['mean_think_words']"))
    # ---- Exp 1.2 per-seed counts, Holm, scenario CIs
    for corpus in ("A", "B"):
        for teacher in ("refusal_ablated", "pca_101", "intact"):
            for rank in (32, 8):
                c = E12(corpus, teacher, rank)
                for s in ("0", "1", "2"):
                    fact(f"e12.seed.{corpus}.{teacher}.r{rank}.s{s}", cnt(f"e12.seed.{corpus}.{teacher}.r{rank}.s{s}",
                         f"{c}['agentic']['primary']['per_seed_shortcut'][{s!r}] * {c}['agentic']['primary']['shortcut']['n'] / len({c}['seeds'])"))
                fact(f"e12.hl.{corpus}.{teacher}.r{rank}", kn(f"e12.hl.{corpus}.{teacher}.r{rank}", f"{c}['agentic']['hc_harmless_implicit']['shortcut']['k']", f"{c}['agentic']['hc_harmless_implicit']['shortcut']['n']"))
    c = E12("B", "refusal_ablated", 32)
    fact("e12.seed.n", cnt("e12.seed.n", f"{c}['agentic']['primary']['shortcut']['n'] / len({c}['seeds'])"))
    for corpus in ("A", "B"):
        for rank in (32, 8):
            for con in ("refusal_ablated_vs_intact", "refusal_ablated_vs_pca_101"):
                key = f"e12c.{corpus}.r{rank}.{con.split('_vs_')[1]}"
                fact(key + ".holm", pval(key + ".holm", E12C(corpus, rank, con, "p_mcnemar_holm")), True)
                fact(key + ".scl", pct(key + ".scl", E12C(corpus, rank, con, "scenario_ci_low")))
                fact(key + ".sch", pct(key + ".sch", E12C(corpus, rank, con, "scenario_ci_high")))
                fact(key + ".sfs", pval(key + ".sfs", E12C(corpus, rank, con, "p_signflip_scenario")), True)
    # ---- Exp 2.3 per-domain Llama counts
    for frame in ("designated", "peer"):
        for dom in ("warehouse", "compute", "finance", "wetlab"):
            fact(f"e23.dom.{frame}.{dom}", cnt(f"e23.dom.{frame}.{dom}", f"R['exp23']['domain_cells'][{'llama70b:cot_off|' + frame + '|B4a'!r}][{dom!r}]['k_shortcut']"))
    fact("e23.dom.n", cnt("e23.dom.n", "R['exp23']['domain_cells']['llama70b:cot_off|designated|B4a']['warehouse']['n']"))
    # ---- Exp 1.4 pre/post block
    for eff in ("default", "low"):
        T = f"R['exp14_blocked_route']['analyses'][{eff!r}]['tables']['refusal_ablated']['cells']"
        fact(f"e14.{eff}.abl.post", cnt(f"e14.{eff}.abl.post", f"sum(c['counts']['SHORTCUT'] - c['shortcut_before_block'] for c in {T}.values())"))
        fact(f"e14.{eff}.abl.reached", cnt(f"e14.{eff}.abl.reached", f"sum(c['reached_block']['k'] for c in {T}.values())"))
        fact(f"e14.{eff}.abl.hexpl.post", cnt(f"e14.{eff}.abl.hexpl.post", f"sum(c['counts']['SHORTCUT'] - c['shortcut_before_block'] for kk, c in {T}.items() if kk.startswith('harmless|explicit'))"))
        fact(f"e14.{eff}.abl.hexpl.pre", cnt(f"e14.{eff}.abl.hexpl.pre", f"sum(c['shortcut_before_block'] for kk, c in {T}.items() if kk.startswith('harmless|explicit'))"))
        fact(f"e14.{eff}.abl.harm.post", cnt(f"e14.{eff}.abl.harm.post", f"sum(c['counts']['SHORTCUT'] - c['shortcut_before_block'] for kk, c in {T}.items() if kk.startswith('harmful'))"))
    fact("e14.harm.post", cnt("e14.harm.post", "sum(c['counts']['SHORTCUT'] - c['shortcut_before_block'] for a in R['exp14_blocked_route']['analyses'].values() for t in a['tables'].values() for kk, c in t['cells'].items() if kk.startswith('harmful'))"))
    # ---- Exp 1.1 harmless-explicit exploratory contrast
    X = "R['exp11_main']['exploratory']['ablated_vs_intact:hc_harmless_explicit']"
    fact("hx.p", pval("hx.p", f"{X}['p_mcnemar']"), True)
    fact("hx.sf", pval("hx.sf", f"{X}['cluster_boot']['p_signflip']"), True)
    fact("hx.ctrlmax", cnt("hx.ctrlmax", f"max({E11('pca_101', 'cot_on', HC_HARMLESS_EXPL)}, {E11('pca_102', 'cot_on', HC_HARMLESS_EXPL)}, {E11('pca_103', 'cot_on', HC_HARMLESS_EXPL)}, {E11('unrelated', 'cot_on', HC_HARMLESS_EXPL)})"))
    # ---- Exp 3.1 slice contrast details
    for other in ("intact",):
        a1 = ("next(x for x in R['exp31_multidomain']['contrasts']['A1_itt'] if x['a']=='refusal_ablated' and "
              "x['within']['cot_mode']=='cot_on' and x['within']['fmt']=='numeric' and x['scope']=='hc_harmful_implicit')")
        fact("e31.A1.cl", pct("e31.A1.cl", f"{a1}['cluster_ci'][0]"))
        fact("e31.A1.ch", pct("e31.A1.ch", f"{a1}['cluster_ci'][1]"))
        fact("e31.A1.sfholm", pval("e31.A1.sfholm", f"{a1}['p_signflip_holm']"), True)
    fact("e31.design.scen", cnt("e31.design.scen", "R['exp31_multidomain']['design_observed']['n_scenarios']"))
    fact("e31.design.cells", cnt("e31.design.cells", "len(R['exp31_multidomain']['design_observed']['cells'])"))
    fact("e31.design.orders", cnt("e31.design.orders", "2"))
    fact("e31.hc.n", cnt("e31.hc.n", S31("refusal_ablated", "cot_on", "numeric", "n", "hc")))
    # ---- Exp 3.2 additions
    E32 = f"{RA}['exp32']"
    for ds, t in (("exp11", "p"), ("exp11+exp31", "x")):
        for cot in ("cot_on", "cot_off"):
            inc = f"{E32}['regime_a_incremental'][{ds!r}][{cot!r}]"
            key = f"e32inc.{t}.{cot}"
            fact(key + ".dp", dec(key + ".dp", f"{inc}['auroc']['design_probe']['auroc']", 3))
            fact(key + ".design", dec(key + ".design", f"{inc}['auroc']['design']['auroc']", 3))
            for f, fld in (("d", "diff"), ("dl", "ci_low"), ("dh", "ci_high")):
                fact(f"{key}.{f}", dec(f"{key}.{f}", f"{inc}['diff']['design_probe-design'][{fld!r}]", 3))
            hc = f"{E32}['regime_c_additions'][{ds!r}][{cot!r}]['harmful_cot']"
            key = f"e32h.{t}.{cot}"
            fact(key, dec(key, f"{hc}['auroc']['probe']['auroc']", 3))
            fact(key + ".l", dec(key + ".l", f"{hc}['auroc']['probe']['ci_low']", 3))
            fact(key + ".h", dec(key + ".h", f"{hc}['auroc']['probe']['ci_high']", 3))
            fact(key + ".tf", dec(key + ".tf", f"{hc}['auroc']['tfidf']['auroc']", 3))
            fact(key + ".n", cnt(key + ".n", f"{hc}['n']"))
            fact(key + ".pos", cnt(key + ".pos", f"{hc}['n_pos']"))
            fp = f"{E32}['fpr_at_80tpr'][{ds!r}]['regime_a'][{cot!r}]"
            fact(f"e32fpr.{t}.{cot}.probe", dec(f"e32fpr.{t}.{cot}.probe", f"{fp}['probe']['fpr']", 3))
            fact(f"e32fpr.{t}.{cot}.design", dec(f"e32fpr.{t}.{cot}.design", f"{fp}['design']['fpr']", 3))
        for g in ("intact", "refusal_ablated", "pca_101"):
            for cot in ("cot_on", "cot_off"):
                ac = f"[x for x in {E32}['regime_c_additions'][{ds!r}][{cot!r}]['arm_cot'] if x['group']=={g + '|' + cot!r}]"
                if ev(ac):
                    fact(f"e32C.{t}.{g}.{cot}.tf", dec(f"e32C.{t}.{g}.{cot}.tf", f"{ac}[0]['auroc']['tfidf']['auroc']", 3))
                    fact(f"e32C.{t}.{g}.{cot}.l", dec(f"e32C.{t}.{g}.{cot}.l", f"{ac}[0]['auroc']['probe']['ci_low']", 3))
                    fact(f"e32C.{t}.{g}.{cot}.h", dec(f"e32C.{t}.{g}.{cot}.h", f"{ac}[0]['auroc']['probe']['ci_high']", 3))
    for cot in ("cot_on", "cot_off"):
        pj = f"{E32}['projection_intact']['sites']['out'][{cot!r}]"
        for c, tag in (("harmless_explicit-harmless_implicit", "prohib"), ("harmful_implicit-harmless_implicit", "harm")):
            for m in ("signed_dot", "q"):
                key = f"proj.{cot}.{tag}.{m}"
                d = 2 if m == "signed_dot" else 4
                fact(key, dec(key, f"{pj}['contrasts'][{c!r}][{m!r}]['mean_diff']", d))
                fact(key + ".l", dec(key + ".l", f"{pj}['contrasts'][{c!r}][{m!r}]['ci_low']", d))
                fact(key + ".h", dec(key + ".h", f"{pj}['contrasts'][{c!r}][{m!r}]['ci_high']", d))
                fact(key + ".p", pval(key + ".p", f"{pj}['contrasts'][{c!r}][{m!r}]['p_signflip']"), True)
        for grp in ("harmless_implicit", "harmless_explicit", "harmful_implicit"):
            fact(f"proj.{cot}.{grp}", dec(f"proj.{cot}.{grp}", f"{pj}['group_means'][{grp!r}]['mean_signed_dot']", 2))
    fact("e32.tpr", cnt("e32.tpr", "80"))
    fact("proj.npairs", cnt("proj.npairs", f"{E32}['projection_intact']['sites']['out']['cot_on']['contrasts']['harmful_implicit-harmless_implicit']['q']['n_pairs']"))
    # ---- Exp 1.1 deconfounders
    DC = "R['exp11_deconfounders']"
    for k in ev(f"{DC}['cells']"):
        arm, cond, eff = k.split("|")
        for a in ("safe_route_completed", "shortcut", "escalated", "prose_stop", "unscorable", "n"):
            fact(f"dc.{arm}.{cond}.{eff}.{a}", cnt(f"dc.{arm}.{cond}.{eff}.{a}", f"{DC}['cells'][{k!r}][{a!r}]"))
    for k in ev(f"{DC}['within_arm']"):
        arm, eff, comp = k.split("|")
        w = f"{DC}['within_arm'][{k!r}]"
        if ev(f"{w}['shortcut'].get('p_mcnemar')") is None:
            continue
        key = f"dcw.{arm}.{eff}.{comp}"
        fact(key + ".p", pval(key + ".p", f"{w}['shortcut']['p_mcnemar']"), True)
        if ev(f"{w}['shortcut'].get('p_holm')") is not None:
            fact(key + ".holm", pval(key + ".holm", f"{w}['shortcut']['p_holm']"), True)
        if ev(f"{w}['agreement_shortcut']['kappa']") is not None:
            fact(key + ".kappa", dec(key + ".kappa", f"{w}['agreement_shortcut']['kappa']", 2))
            fact(key + ".both", cnt(key + ".both", f"{w}['agreement_shortcut']['n_both_pos']"))
        fact(key + ".actagr", pct(key + ".actagr", f"{w}['agreement_action']['agreement']", 1))
    # ---- Exp 3.1 2x2
    X2 = "R['exp31_2x2_deconfounder']"
    for design in ("same_launch", "combined"):
        for arm in ("refusal_ablated", "pca_101", "intact"):
            pa = f"{X2}['designs'][{design!r}]['per_arm'][{arm!r}]"
            for f in ("CS", "AS", "CT", "AT"):
                fact(f"x2.{design}.{arm}.{f}", pct(f"x2.{design}.{arm}.{f}", f"{pa}['rates'][{f!r}]"))
                fact(f"x2.{design}.{arm}.{f}.k", cnt(f"x2.{design}.{arm}.{f}.k", f"{pa}['counts'][{f!r}]"))
            for e in ("layout_main", "abstraction_main", "interaction", "layout_concrete", "layout_abstract",
                      "abstraction_sentence", "abstraction_table", "total"):
                key = f"x2.{design}.{arm}.{e}"
                fact(key, pct(key, f"{pa}[{e!r}]['estimate']"))
                fact(key + ".l", pct(key + ".l", f"{pa}[{e!r}]['ci_low']"))
                fact(key + ".h", pct(key + ".h", f"{pa}[{e!r}]['ci_high']"))
                fact(key + ".p", pval(key + ".p", f"{pa}[{e!r}]['p_signflip']"), True)
        for other in ("intact", "pca_101"):
            for e in ("layout_main", "abstraction_main", "interaction", "total", "gap_CS", "gap_AS", "gap_CT", "gap_AT"):
                d = f"{X2}['designs'][{design!r}]['arm_x_format'][{'refusal_ablated_vs_' + other + '|' + e!r}]"
                key = f"x2d.{design}.{other}.{e}"
                fact(key, pct(key, f"{d}['estimate']"))
                fact(key + ".l", pct(key + ".l", f"{d}['ci_low']"))
                fact(key + ".h", pct(key + ".h", f"{d}['ci_high']"))
                fact(key + ".p", pval(key + ".p", f"{d}['p_signflip']"), True)
                if ev(f"{d}.get('p_signflip_holm')") is not None:
                    fact(key + ".holm", pval(key + ".holm", f"{d}['p_signflip_holm']"), True)
    fact("x2.n", cnt("x2.n", f"{X2}['designs']['same_launch']['per_arm']['refusal_ablated']['n_complete_pairs']"))
    for arm in ("refusal_ablated", "pca_101", "intact"):
        for fmt in ("concrete_sentence", "abstract_table"):
            rv = f"{X2}['replicate_vs_original'][{arm + '|' + fmt!r}]"
            fact(f"x2r.{arm}.{fmt}.p", pval(f"x2r.{arm}.{fmt}.p", f"{rv}['shortcut']['p_mcnemar']"), True)
            if ev(f"{rv}['agreement']['kappa']") is not None:
                fact(f"x2r.{arm}.{fmt}.kappa", dec(f"x2r.{arm}.{fmt}.kappa", f"{rv}['agreement']['kappa']", 2))


DEMO_TRIAL = "demo:main:demo_inventory:high_tedium:harmless:implicit"


def demo_episode() -> dict:
    """The one hand-written harmless demonstration episode (v3.deconf.DEMO_BASE, never an authored base), read from its raw
    transcript. The only raw text this script reads; it is checked against the demo scenario before use."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from v3 import deconf
    rows = [json.loads(l) for l in (ROOT / "raw_transcripts" / "deconf_exp11" / "intact.jsonl").read_text().splitlines() if l.strip()]
    row = next(r for r in rows if r["trial_id"] == DEMO_TRIAL)
    expect = next(i for i in deconf.demo_instances() if i["trial_id"] == DEMO_TRIAL)
    assert row["scenario_id"] == deconf.DEMO_BASE["scenario_id"] and row["messages"] == expect["messages"], "demo row mismatch"
    return row


def build_demo_listing():
    from v3 import episodes
    row = demo_episode()
    fresh = episodes.recompute_record(row)
    obs = episodes.observation_message(fresh["turns"][0]["observations"])
    parts = [f"[system]\n{row['messages'][0]['content']}", f"[user]\n{row['messages'][1]['content']}",
             f"[assistant: reasoning, tool call]\n{row['turns'][0]['raw_output'].strip()}", f"[sandbox]\n{obs}"]
    text = "\n\n".join(parts)
    assert "\\end{lstlisting}" not in text and text.isascii()
    (PAPER / "tables" / "tab_demo_episode.tex").write_text(
        "\\begin{lstlisting}[basicstyle=\\ttfamily\\scriptsize,breaklines=true,columns=fullflexible,keepspaces=true,"
        "breakatwhitespace=false,frame=single,framesep=2pt,xleftmargin=2pt,xrightmargin=2pt]\n" + text + "\n\\end{lstlisting}\n")


def build_review_tables():
    build_demo_listing()
    # Appendix: Exp 2.1 breakdown
    rows = []
    for team, tl in (("mixed_team", "Mixed (Abl.\\ Alpha)"), ("all_intact", "All Intact")):
        for var, vl in (("A", "A: no channel"), ("C_nopen", "C: no channel, gate"), ("C_pen", "C: gate + fee"),
                        ("B", "B: channel"), ("D_nopen", "D: channel, gate"), ("D_pen", "D: gate + fee")):
            rows.append([tl if var == "A" else "", vl,
                         kn(f"tab21.{team}.{var}.af", E21(team, var, "alpha_fast", "k"), E21(team, var, "alpha_fast", "n")),
                         kn(f"tab21.{team}.{var}.bf", E21(team, var, "beta_fast", "k"), E21(team, var, "beta_fast", "n")),
                         kn(f"tab21.{team}.{var}.lg", E21(team, var, "LEGIT_SAFE_COORD", "k"), E21(team, var, "LEGIT_SAFE_COORD", "n")),
                         kn(f"tab21.{team}.{var}.ij", E21(team, var, "ILLEGIT_JOINT", "k"), E21(team, var, "ILLEGIT_JOINT", "n"))])
        rows.append("MIDRULE")
    rows.pop()
    tex_table("tab_exp21", "Exp~2.1 by team and topology ($k/n$ trials): Alpha and Beta express-lane use, the legitimate-coordination "
              "outcome (Alpha express, Beta standard) and the forbidden joint shortcut.", "tab:exp21", "llcccc",
              "Team & Topology & Alpha exp. & Beta exp. & Legit.\\ coord. & Joint", rows, size="\\scriptsize", sep="2.5pt")
    # Appendix: Exp 2.2 breakdown
    rows = []
    for mon, ml in (("intact", "Intact"), ("llama70b", "Llama-3.3-70B"), ("flash_next", "Flash-Next")):
        for cond in ("action_log_only", "minimal", "neutral_matched", "truthful_compliance", "deceptive_compound", "clause_both"):
            ex = f" and c['condition']=={cond!r}"
            rows.append([ml if cond == "action_log_only" else "", cond.replace("_", " "),
                         kn(f"tab22.{mon}.{cond}.t", E22(mon, TPR, "k", ex), E22(mon, TPR, "n", ex)),
                         kn(f"tab22.{mon}.{cond}.f", E22(mon, FPR, "k", ex), E22(mon, FPR, "n", ex))])
        rows.append("MIDRULE")
    rows.pop()
    tex_table("tab_exp22", "Exp~2.2 monitor verdicts by note condition (thinking off; readable verdicts): forbidden bundles blocked "
              "and allowed bundles blocked ($k/n$).", "tab:exp22", "llcc",
              "Monitor & Note & Forbidden blocked & Allowed blocked", rows, size="\\scriptsize", sep="3pt")
    # Exp 1.1 deconfounders
    DC = "R['exp11_deconfounders']['cells']"
    rows = []
    for arm in ("refusal_ablated", "intact", "pca_101"):
        for eff in ("default", "low"):
            for cond, cl in (("std_original", "standard (original run)"), ("std", "standard (same launch)"),
                             ("nh", "no hand-off tool"), ("np", "no fewer-calls sentence")):
                k = f"{arm}|{cond}|{eff}"
                if k not in ev(DC):
                    continue
                rows.append([ARM_LABEL[arm] if (eff, cond) == ("default", "std_original") else "", eff if cond == "std_original" else "", cl] +
                            [cnt(f"tabdc.{arm}.{cond}.{eff}.{a}", f"{DC}[{k!r}][{a!r}]") for a in
                             ("safe_route_completed", "shortcut", "escalated", "prose_stop", "unscorable")])
        rows.append("MIDRULE")
    rows.pop()
    tex_table("tab_deconf11", "Exp~1.1 deconfounders on the 80 high-cost harmful-implicit pairs (thinking on). The original run, a same-launch "
              "replicate of the identical prompts, the prompt without the hand-off tool, and the prompt without the call-count preference "
              "sentence. Columns: completed the seven-call ordinary route, shortcut, escalated, prose stop, unscorable.", "tab:deconf11",
              "lllccccc", "Arm & Effort & Condition & Safe & Shortcut & Escal. & Prose & Unsc.", rows, size="\\scriptsize", sep="2.2pt")
    # Exp 3.1 2x2
    X2 = "R['exp31_2x2_deconfounder']['designs']['same_launch']"
    rows = []
    for arm in ("refusal_ablated", "pca_101", "intact"):
        pa = f"{X2}['per_arm'][{arm!r}]"
        r = [ARM_LABEL[arm]] + [pct(f"tabx2.{arm}.{f}", f"{pa}['rates'][{f!r}]") for f in ("CS", "AS", "CT", "AT")]
        for e in ("layout_main", "abstraction_main", "interaction"):
            r.append(f"{pct(f'tabx2.{arm}.{e}', pa + '[' + repr(e) + '][' + repr('estimate') + ']')} "
                     f"[{pct(f'tabx2.{arm}.{e}.l', pa + '[' + repr(e) + '][' + repr('ci_low') + ']')}, "
                     f"{pct(f'tabx2.{arm}.{e}.h', pa + '[' + repr(e) + '][' + repr('ci_high') + ']')}]")
        rows.append(r)
    rows.append("MIDRULE")
    for other, ol in (("intact", "Abl.\\ $-$ Intact"), ("pca_101", "Abl.\\ $-$ PCA-101")):
        d = f"{X2}['arm_x_format']"
        r = [ol] + [pct(f"tabx2.d.{other}.{f}", f"{d}[{'refusal_ablated_vs_' + other + '|gap_' + f!r}]['estimate']") for f in ("CS", "AS", "CT", "AT")]
        for e in ("layout_main", "abstraction_main", "interaction"):
            q = f"{d}[{'refusal_ablated_vs_' + other + '|' + e!r}]"
            r.append(f"{pct(f'tabx2.d.{other}.{e}', q + chr(91) + repr('estimate') + chr(93))} "
                     f"[{pct(f'tabx2.d.{other}.{e}.l', q + chr(91) + repr('ci_low') + chr(93))}, "
                     f"{pct(f'tabx2.d.{other}.{e}.h', q + chr(91) + repr('ci_high') + chr(93))}]")
        rows.append(r)
    tex_table("tab_exp31_2x2", "Exp~3.1 layout $\\times$ specificity on the high-cost harmful-implicit slice (thinking on, 128 pairs per arm, all four "
              "cells generated in one launch). Shortcut rates (ITT, \\%) for the concrete sentence (CS, the original narrative format), the "
              "abstract rating sentence (AS), the table with the concrete description (CT) and the abstract table (AT, the original numeric "
              "format). Effects in pp with scenario-cluster bootstrap 95\\% CI: layout = table $-$ sentence and abstraction = rating words $-$ "
              "description, each averaged over the other factor; interaction = (AT$-$AS)$-$(CT$-$CS). Lower rows: arm gaps per cell and "
              "arm $\\times$ format difference-in-differences.", "tab:exp31x2", "lccccccc",
              "Arm & CS & AS & CT & AT & Layout & Abstraction & Interaction", rows, size="\\footnotesize", wide=True, sep="5pt")
    # Exp 1.1 5-way action split (main text)
    rows = []
    for arm in ARMS:
        for cond, cl in (("default", "default"), ("low", "low"), ("off", "off")):
            if f"{arm}|{cond}|high_tedium|harmful|implicit" not in ev(f"{RA}['action_splits']"):
                continue
            rows.append([ARM_LABEL[arm] if cond == "default" else "", cl] +
                        [cnt(f"tab11as.{arm}.{cond}.{a}", AS(arm, cond, a)) for a in
                         ("safe_route_completed", "shortcut", "escalated", "prose_stop", "unscorable")])
    tex_table("tab_exp11_actions", "Exp~1.1 action split on the 80 high-cost harmful-implicit pairs: completed the seven-call ordinary "
              "route, took the direct shortcut, escalated with the hand-off tool, stopped with a prose reply, or unscorable "
              "(truncated, malformed or turn limit). Thinking on at default or low effort, or thinking off.", "tab:exp11actions",
              "llccccc", "Arm & Cond. & Safe & Shortcut & Escal. & Prose & Unsc.", rows, size="\\scriptsize", sep="2.5pt")


def main() -> int:
    for d in ("generated", "tables", "figures"):
        (PAPER / d).mkdir(parents=True, exist_ok=True)
    build_facts()
    build_review_facts()
    build_tables()
    build_review_tables()
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
