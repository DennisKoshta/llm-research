"""Tests for scripts/analyze_exp11_robustness.py: scripted episodes through the real sandbox / parser, a hidden truth table per
(arm, condition), and answers recomputed by brute-force enumeration (binomial tails, all 2^10 template sign flips, explicit loops,
a numpy re-implementation of the cluster bootstrap with the same draw stream). Placeholders only: no scenario text."""
import copy
import importlib.util
import itertools
import json
import math
import pathlib
import random
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from v3 import exp11 as E  # noqa: E402
from v3 import scenarios as S  # noqa: E402
from v3 import stats  # noqa: E402
from v3.arms import ARMS_27B  # noqa: E402
from v3.episodes import run_episodes  # noqa: E402


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


R = load_script("analyze_exp11_robustness")
BASES = S.make_placeholder_bases(40)
N_BOOT = 400
SEED = stats.DEFAULT_SEED
ABL, INT = "refusal_ablated", "intact"
DIR_OF = {("default", a): "exp11" for a in ARMS_27B}
DIR_OF.update({("low", "intact"): "exp11_effort_low_a", ("low", "refusal_ablated"): "exp11_effort_low_a",
               ("low", "pca_101"): "exp11_effort_low_b", ("low", "pca_102"): "exp11_effort_low_b",
               ("low", "pca_103"): "exp11_effort_low_c", ("low", "unrelated"): "exp11_effort_low_c",
               ("off", "intact"): "exp11_cotoff", ("off", "refusal_ablated"): "exp11_cotoff"})
CELLS = [(a, c) for c in ("default", "low", "off") for a in (ARMS_27B if c != "off" else E.ARMS_COT_OFF)]
RATE = {(INT, "default"): 0.05, (INT, "low"): 0.08, (INT, "off"): 0.12, (ABL, "default"): 0.25, (ABL, "low"): 0.45,
        (ABL, "off"): 0.35, ("pca_101", "default"): 0.0, ("pca_101", "low"): 0.05, ("pca_102", "default"): 0.10,
        ("pca_102", "low"): 0.0, ("pca_103", "default"): 0.0, ("pca_103", "low"): 0.0,
        ("unrelated", "default"): 0.0, ("unrelated", "low"): 0.02}


# ---- scripted model ------------------------------------------------------------------------------------
def call(name, **args):
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": args}) + "\n</tool_call>"


def wrap(cot, body):
    return f"[REASONING_PLACEHOLDER]\n</think>\n\n{body}" if cot == "cot_on" else body


class Model:
    """generate_fn whose behaviour on (scenario_id, cost_regime) is looked up in `truth`; unknown instances take the shortcut."""

    def __init__(self, truth, instances):
        self.truth, self.by_id = truth, {i["trial_id"]: i for i in instances}

    def __call__(self, reqs):
        outs = []
        for r in reqs:
            m = re.fullmatch(r"(.*):turn(\d+)", r["key"])
            inst, turn, cot = self.by_id[m.group(1)], int(m.group(2)), r["cot_mode"]
            act = self.truth.get((inst["scenario_id"], inst["cost_regime"], inst["consequence"], inst["boundary"]), "short")
            raw, finish = self.output(cot, inst, turn, act)
            suffix = "<think>\n" if cot == "cot_on" else "<think>\n\n</think>\n\n"
            prompt = "\n".join(x["content"] for x in r["messages"])
            outs.append({"raw_output": raw, "generated_tokens": 7, "finish_reason": finish,
                         "full_prompt": f"[FULL_PROMPT {r['key']}]\n{prompt}\n" + suffix})
        return outs

    @staticmethod
    def output(cot, inst, turn, act):
        if act == "trunc":
            return "[REASONING_PLACEHOLDER] still deliberating", "length"
        if act == "text":
            return wrap(cot, "[PROSE_PLACEHOLDER]"), "stop"
        if act == "prose":      # names the direct tool in prose only: strict parsing must not count it
            return wrap(cot, f"[PROSE_PLACEHOLDER] {inst['direct_tool']}"), "stop"
        if act == "short":
            return wrap(cot, call(inst["direct_tool"], workspace=inst["scenario_id"])), "stop"
        step = inst["safe_steps"][turn - 1]
        safe = call(inst["safe_tool"], operation=step["operation"], step=turn)
        if act == "decoy":      # a direct call inside the unclosed-then-closed think block, a safe call after it
            return f"[REASONING_PLACEHOLDER] {call(inst['direct_tool'], workspace=inst['scenario_id'])}\n</think>\n\n{safe}", "stop"
        return wrap(cot, safe), "stop"


def gen_rows(arm, cond, truth, instances):
    cot, eff, cap = {"default": ("cot_on", None, 16384), "low": ("cot_on", "low", 4096), "off": ("cot_off", None, 1024)}[cond]
    exp_id = DIR_OF[cond, arm]
    return run_episodes(Model(truth, instances), instances, arm=arm, exp_id=exp_id, cot_mode=cot, reasoning_effort=eff,
                        max_new_tokens=cap, max_turns=E.MAX_TURNS)


def key_of(inst):
    return (inst["scenario_id"], inst["cost_regime"], inst["consequence"], inst["boundary"])


def make_truth(keys, seed=11):
    """truth[(arm, cond)][key] -> behaviour. Shortcut with prob RATE (correlated across conditions of an arm), then a few
    non-shortcut special behaviours (truncation, text-only reply, prose naming the direct tool, decoy call inside the think block)."""
    rng = random.Random(seed)
    shared = {a: {k: rng.random() for k in keys} for a in ARMS_27B}
    out = {}
    for (arm, cond) in CELLS:
        p, cell = RATE[arm, cond], {}
        for k in keys:
            u = shared[arm][k] if rng.random() < 0.6 else rng.random()
            cell[k] = "short" if u < p else "safe"
        safe = [k for k in keys if cell[k] == "safe"]
        specials = ["trunc", "text", "prose"] + (["decoy"] if cond != "off" else [])
        for k, sp in zip(rng.sample(safe, len(specials)), specials):
            cell[k] = sp
        out[arm, cond] = cell
    return out


@pytest.fixture(scope="module")
def world():
    reqs = {c: E.build_requests(BASES, effort=e) for c, e in (("default", None), ("low", "low"))}
    inst = {"default": [r["instance"] for r in reqs["default"]["main_cot_on"]],
            "low": [r["instance"] for r in reqs["low"]["main_cot_on"]],
            "off": [r["instance"] for r in reqs["default"]["main_cot_off"]]}
    in_slice = lambda i: i["cost_regime"] in R.COSTS and i["consequence"] == "harmful" and i["boundary"] == "implicit"
    slice_inst = {c: [i for i in v if in_slice(i)] for c, v in inst.items()}
    outside = {}      # two instances of every other main-grid cell; unknown keys -> the model takes the shortcut, so any leak shows
    for c, v in inst.items():
        per = defaultdict(list)
        for i in v:
            if not in_slice(i):
                per[i["cost_regime"], i["consequence"], i["boundary"]].append(i)
        outside[c] = [i for cell in sorted(per) for i in per[cell][:2]]
    keys = [key_of(i) for i in slice_inst["default"]]
    truth = make_truth(keys)
    by_dir = defaultdict(list)
    for (arm, cond) in CELLS:
        by_dir[DIR_OF[cond, arm]] += gen_rows(arm, cond, truth[arm, cond], slice_inst[cond] + outside[cond])
    for d in R.SOURCE_DIRS:
        by_dir.setdefault(d, [])
        by_dir[d] += [dict(r, family="forced_safe") for r in by_dir[d] if R._in_slice(r) and r["direct_taken"]][:3]   # other-family decoys
    tmpl = {key_of(i): i["template_id"] for i in slice_inst["default"]}
    return {"rows": dict(by_dir), "truth": truth, "keys": keys, "tmpl": tmpl, "slice_instances": slice_inst}


@pytest.fixture(scope="module")
def rec(world):
    return R.recompute(world["rows"], n_boot=N_BOOT)


# ---- brute-force references ----------------------------------------------------------------------------
Z = 1.959963984540054


def wilson(k, n):
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def signflip_p(sums, total):
    obs = abs(sum(sums)) / total
    hits = sum(abs(sum(e * s for e, s in zip(signs, sums))) / total >= obs - 1e-12
               for signs in itertools.product((-1, 1), repeat=len(sums)))
    return hits / 2 ** len(sums)


def holm_ref(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def ref_boot(s, m, n_boot, seed, alpha=0.05):
    """Same draw stream as stats.cluster_bootstrap_diff (single chunk since n_boot <= 1000)."""
    s, m = np.asarray(s, float), np.asarray(m, float)
    idx = np.random.default_rng(seed).integers(0, len(s), size=(n_boot, len(s)))
    boot = s[idx].sum(axis=1) / m[idx].sum(axis=1)
    return np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)]), boot


class Ref:
    """Independent expectations from the hidden truth table."""

    def __init__(self, world):
        self.keys, self.tmpl, self.truth = world["keys"], world["tmpl"], world["truth"]

    def vec(self, arm, cond):
        return [int(self.truth[arm, cond][k] == "short") for k in self.keys]

    def clusters(self, kind):
        return [k[0] if kind == "scenario" else self.tmpl[k] for k in self.keys]

    def table(self, arm_a, ca, arm_b, cb, idx=None):
        a, b = self.vec(arm_a, ca), self.vec(arm_b, cb)
        idx = range(len(a)) if idx is None else idx
        both = sum(a[i] and b[i] for i in idx)
        ao = sum(a[i] and not b[i] for i in idx)
        bo = sum(b[i] and not a[i] for i in idx)
        return {"both_shortcut": both, "a_only": ao, "b_only": bo, "neither": len(list(idx)) - both - ao - bo}

    def sums(self, d, kind):
        cl = self.clusters(kind)
        ids = sorted(set(cl))
        return ids, [sum(d[i] for i in range(len(d)) if cl[i] == c) for c in ids], [sum(cl[i] == c for i in range(len(d))) for c in ids]


# ---- tests ---------------------------------------------------------------------------------------------
def test_world_shape(world):
    assert len(world["keys"]) == 80 and len({k[0] for k in world["keys"]}) == 40 and len(set(world["tmpl"].values())) == 10
    assert Counter(Counter(world["tmpl"].values()).values()) == {8: 10}
    assert set(world["rows"]) == set(R.SOURCE_DIRS)


def test_hygiene_and_slice(rec, world):
    h = rec["hygiene"]
    assert h["all_cells_complete"] and h["all_cells_same_instances"] and h["n_template_id_mismatch_across_cells"] == 0
    assert (h["n_scenarios"], h["n_templates"], h["n_trials"], h["n_trials_excluded"]) == (40, 10, 14 * 80, 0)
    assert h["dir_condition_mismatch"] == {} and h["excluded"] == {}
    assert h["n_pair_keys_with_differing_instance_fields"] == 0
    assert h["max_new_tokens_by_condition"] == {"default": [16384], "low": [4096], "off": [1024]}
    assert h["replay"] == {"n_rows": 14 * 80, "n_unplaceable": 0, "n_duplicate_rows": 0, "n_invalid_records": 0, "n_stored_mismatch_rows": 0,
                           "n_i1_violations": 0, "n_banned_literal_rows": 0, "n_prompt_suffix_violations": 0,
                           "n_prompt_mismatch_rows": 0, "n_boundary_label_mismatch_rows": 0}
    assert h["stored_mismatch_fields"] == {}
    assert sum(h["rows_in_slice"].values()) == 14 * 80
    assert h["rows_in_slice"]["exp11_cotoff"] == 2 * 80 and h["rows_in_slice"]["exp11"] == 6 * 80
    assert h["n_rows_outside_slice"] == sum(h["rows_read"].values()) - 14 * 80 > 0     # the out-of-slice rows were read and dropped
    assert rec["params"] == {"n_boot": N_BOOT, "seed": SEED, "alpha": 0.05, "source_dirs": list(R.SOURCE_DIRS)}


def test_cells_counts_wilson_and_itt(rec, world):
    ref = Ref(world)
    for arm, cond in CELLS:
        c = rec["cells"][arm][cond]
        k = sum(ref.vec(arm, cond))
        assert (c["n"], c["k_shortcut"], c["complete"], c["n_scenarios"], c["n_templates"]) == (80, k, True, 40, 10)
        assert c["rate"] == pytest.approx(k / 80)
        assert c["wilson"] == pytest.approx(list(wilson(k, 80)), abs=1e-9)
        assert c["source_rows"] == {DIR_OF[cond, arm]: 80}
    assert set(rec["cells"]["pca_101"]) == {"default", "low"} and set(rec["cells"][INT]) == {"default", "low", "off"}


def test_strict_parsing_decoys_and_prose_are_not_shortcuts(rec, world):
    """The prose that names the direct tool and the direct call inside the think block never count; both stay in n."""
    for arm, cond in CELLS:
        cell = world["truth"][arm, cond]
        n_short = sum(v == "short" for v in cell.values())
        assert rec["cells"][arm][cond]["k_shortcut"] == n_short
        tr = rec["truncation"][arm][cond]
        n_trunc = sum(v in ("trunc", "text", "prose") for v in cell.values())
        assert tr["n_truncated_or_malformed"] == n_trunc and tr["n"] == 80       # ITT: truncated / text-only stay in n
        assert tr["sub_status"].get("TEXT_NO_CALL", 0) >= sum(v in ("text", "prose") for v in cell.values())
        assert sum(tr["sub_status"].values()) == n_trunc and tr["n_truncated_and_direct_taken"] == 0


def test_truncation_accounting_and_upper_bounds(rec, world):
    for arm, cond in CELLS:
        cell = world["truth"][arm, cond]
        k = sum(v == "short" for v in cell.values())
        n_text = sum(v in ("text", "prose") for v in cell.values())
        n_trunc = sum(v in ("trunc", "text", "prose") for v in cell.values())
        tr = rec["truncation"][arm][cond]
        assert tr["k_shortcut_itt"] == k and tr["rate_itt"] == pytest.approx(k / 80)
        assert tr["upper_bound_all_truncated"]["k"] == k + n_trunc
        assert tr["upper_bound_all_truncated"]["rate"] == pytest.approx((k + n_trunc) / 80)
        assert tr["upper_bound_all_truncated"]["wilson"] == pytest.approx(list(wilson(k + n_trunc, 80)), abs=1e-9)
        assert tr["upper_bound_excl_text_no_call"]["k"] == k + n_trunc - n_text
        assert tr["rate_truncated_or_malformed"] == pytest.approx(n_trunc / 80)


@pytest.mark.parametrize("name,ca,cb,arms", R.CONTRASTS)
def test_paired_contrasts_against_brute_force(rec, world, name, ca, cb, arms):
    ref = Ref(world)
    for arm in arms:
        c = rec["contrasts"][f"{name}:{arm}"]
        a, b = ref.vec(arm, ca), ref.vec(arm, cb)
        t = ref.table(arm, ca, arm, cb)
        assert c["table"] == t and c["n_pairs"] == 80 and c["n_unmatched"] == 0 and c["n_discordant"] == t["a_only"] + t["b_only"]
        assert (c["a"], c["b"], c["orientation"]) == (f"{arm}|{ca}", f"{arm}|{cb}", "a - b")
        assert (c["k_a"], c["k_b"]) == (sum(a), sum(b))
        assert c["diff"] == pytest.approx((sum(a) - sum(b)) / 80)
        assert c["wilson_a"] == pytest.approx(list(wilson(sum(a), 80)), abs=1e-9)
        assert c["p_mcnemar"] == pytest.approx(mcnemar_p(t["a_only"], t["b_only"]), abs=1e-12)
        assert c["newcombe"] == pytest.approx(list(stats.paired_diff_newcombe_ci(a, b)), abs=1e-12)
        d = [x - y for x, y in zip(a, b)]
        _, s, m = ref.sums(d, "template")
        tb = c["template_cluster_bootstrap"]
        assert tb["signflip_exact"] and tb["n_clusters"] == 10
        assert tb["p_signflip"] == pytest.approx(signflip_p(s, 80), abs=1e-12)           # all 2^10 flips enumerated
        _, s, m = ref.sums(d, "scenario")
        ci, _ = ref_boot(s, m, N_BOOT, SEED)
        sb = c["scenario_cluster_bootstrap"]
        assert sb["n_clusters"] == 40 and sb["n_boot"] == N_BOOT and sb["seed"] == SEED
        assert [sb["ci_low"], sb["ci_high"]] == pytest.approx(list(ci), abs=1e-12)
        assert sb["diff"] == pytest.approx(c["diff"])


def test_holm_families(rec):
    fam = rec["holm_families"]["families"]
    assert fam == {"low_vs_default": [f"low_vs_default:{a}" for a in ARMS_27B],
                   "off_vs_default": ["off_vs_default:intact", "off_vs_default:refusal_ablated"],
                   "low_vs_off": ["low_vs_off:intact", "low_vs_off:refusal_ablated"]}
    assert set(sum(fam.values(), [])) == set(rec["contrasts"])
    for members in fam.values():
        for col, get in (("p_mcnemar_holm", lambda c: c["p_mcnemar"]), ("p_signflip_template_holm", lambda c: c["template_cluster_bootstrap"]["p_signflip"])):
            ps = [get(rec["contrasts"][m]) for m in members]
            assert [rec["contrasts"][m][col] for m in members] == pytest.approx(holm_ref(ps), abs=1e-12)
            assert all(rec["contrasts"][m][col] >= get(rec["contrasts"][m]) for m in members)
    assert rec["holm_families"]["outside_any_family"] == ["did", "p11_template_robustness.default", "p11_template_robustness.low"]


def test_difference_in_differences(rec, world):
    ref = Ref(world)
    al, ad, il, idv = ref.vec(ABL, "low"), ref.vec(ABL, "default"), ref.vec(INT, "low"), ref.vec(INT, "default")
    d = [(al[i] - ad[i]) - (il[i] - idv[i]) for i in range(80)]
    out = rec["did"]
    assert out["n_pairs"] == 80 and out["did"] == pytest.approx(sum(d) / 80)
    assert out["ablated_effort_effect"] == pytest.approx((sum(al) - sum(ad)) / 80)
    assert out["intact_effort_effect"] == pytest.approx((sum(il) - sum(idv)) / 80)
    assert out["did"] == pytest.approx(out["ablated_effort_effect"] - out["intact_effort_effect"])
    assert out["k"] == {"ablated_low": sum(al), "ablated_default": sum(ad), "intact_low": sum(il), "intact_default": sum(idv)}
    assert out["per_instance_did_distribution"] == {str(x): d.count(x) for x in (-2, -1, 0, 1, 2)}
    assert sum(out["per_instance_did_distribution"].values()) == 80
    # scenario clusters: the stacked-encoding bootstrap equals a direct bootstrap of the per-instance DiD with the same draws
    _, s, m = ref.sums(d, "scenario")
    ci, boot = ref_boot(s, m, N_BOOT, SEED)
    sb = out["scenario_cluster_bootstrap"]
    assert sb["diff"] == pytest.approx(sum(d) / 80, abs=1e-12)
    assert [sb["ci_low"], sb["ci_high"]] == pytest.approx(list(ci), abs=1e-12)
    obs = sum(d) / 80
    assert sb["p_boot"] == pytest.approx((1 + np.sum(np.abs(boot - obs) >= abs(obs) - 1e-15)) / (N_BOOT + 1), abs=1e-12)
    signs = np.random.default_rng([abs(SEED), 1]).choice([-1.0, 1.0], size=(N_BOOT, 40))
    assert sb["p_signflip"] == pytest.approx((1 + np.sum(np.abs(signs @ np.asarray(s, float)) / 80 >= abs(obs) - 1e-12)) / (N_BOOT + 1), abs=1e-12)
    # template clusters: exact enumeration of the 2^10 sign flips
    _, s, m = ref.sums(d, "template")
    ci, _ = ref_boot(s, m, N_BOOT, SEED)
    tb = out["template_cluster_bootstrap"]
    assert tb["signflip_exact"] and tb["p_signflip"] == pytest.approx(signflip_p(s, 80), abs=1e-12)
    assert [tb["ci_low"], tb["ci_high"]] == pytest.approx(list(ci), abs=1e-12)


def test_p11_template_robustness(rec, world):
    ref = Ref(world)
    for cond in ("default", "low"):
        out = rec["p11_template_robustness"][cond]
        a, b = ref.vec(ABL, cond), ref.vec(INT, cond)
        tm = ref.clusters("template")
        sc = ref.clusters("scenario")
        templates = sorted(set(tm))
        full = ref.table(ABL, cond, INT, cond)
        assert out["contrast"]["table"] == full and out["contrast"]["p_mcnemar"] == pytest.approx(mcnemar_p(full["a_only"], full["b_only"]))
        d = [x - y for x, y in zip(a, b)]
        _, s, _m = ref.sums(d, "template")
        assert out["template_signflip_p"] == pytest.approx(signflip_p(s, 80), abs=1e-12) and out["template_signflip_exact"]
        assert out["n_templates"] == 10 and out["n_scenarios"] == 40
        disc_t, ao_t, bo_t = set(), set(), set()
        for t in templates:
            idx = [i for i in range(80) if tm[i] == t]
            tab = ref.table(ABL, cond, INT, cond, idx)
            pt = out["per_template"][t]
            assert (pt["n_pairs"], pt["n_scenarios"], pt["both_shortcut"], pt["ablated_only"], pt["intact_only"], pt["neither"]) == (
                8, 4, tab["both_shortcut"], tab["a_only"], tab["b_only"], tab["neither"])
            assert pt["k_ablated"] == sum(a[i] for i in idx) and pt["k_intact"] == sum(b[i] for i in idx)
            assert pt["n_discordant"] == tab["a_only"] + tab["b_only"]
            rest = [i for i in range(80) if tm[i] != t]
            rt = ref.table(ABL, cond, INT, cond, rest)
            lo = out["leave_one_template_out"][t]
            assert (lo["n_pairs"], lo["ablated_only"], lo["intact_only"]) == (72, rt["a_only"], rt["b_only"])
            assert lo["p_mcnemar"] == pytest.approx(mcnemar_p(rt["a_only"], rt["b_only"]), abs=1e-12)
            disc_t |= {t} if tab["a_only"] + tab["b_only"] else set()
            ao_t |= {t} if tab["a_only"] else set()
            bo_t |= {t} if tab["b_only"] else set()
        ps = {}
        for t in templates:
            rt = ref.table(ABL, cond, INT, cond, [i for i in range(80) if tm[i] != t])
            ps[t] = mcnemar_p(rt["a_only"], rt["b_only"])
        assert (out["loto_p_min"], out["loto_p_max"]) == pytest.approx((min(ps.values()), max(ps.values())), abs=1e-12)
        assert ps[out["loto_template_at_p_max"]] == pytest.approx(max(ps.values())) and out["loto_n_p_ge_0_05"] == sum(p >= 0.05 for p in ps.values())
        assert out["n_templates_with_discordant"] == len(disc_t) and out["n_templates_with_ablated_only"] == len(ao_t)
        assert out["n_templates_with_intact_only"] == len(bo_t)
        sc_ao = {sc[i] for i in range(80) if a[i] and not b[i]}
        sc_bo = {sc[i] for i in range(80) if b[i] and not a[i]}
        assert out["n_scenarios_with_ablated_only"] == len(sc_ao) and out["n_scenarios_with_intact_only"] == len(sc_bo)
        assert out["n_scenarios_with_discordant"] == len(sc_ao | sc_bo)
        assert sum(p["n_discordant"] for p in out["per_template"].values()) == full["a_only"] + full["b_only"]


def test_p11_hand_checked_closed_forms(world):
    """Hand-built truth: ablated shortcuts only in one template's 8 pairs (default) and in that template + 8 more pairs of
    two other templates (low); intact never. Closed forms: McNemar p = 2 / 2^8 and 2 / 2^24; LOTO drops 8 or 16 discordants."""
    keys, tmpl = world["keys"], world["tmpl"]
    by_t = defaultdict(list)
    for k in keys:
        by_t[tmpl[k]].append(k)
    t1, t2, t3 = sorted(by_t)[:3]
    truth = {(a, c): {k: "safe" for k in keys} for a, c in CELLS}
    for k in by_t[t1]:
        truth[ABL, "default"][k] = "short"
    for k in by_t[t1] + by_t[t2] + by_t[t3]:
        truth[ABL, "low"][k] = "short"
    rows = defaultdict(list)
    for arm, cond in CELLS:
        rows[DIR_OF[cond, arm]] += gen_rows(arm, cond, truth[arm, cond], world["slice_instances"][cond])
    out = R.recompute(dict(rows), n_boot=200)
    d, lo = out["p11_template_robustness"]["default"], out["p11_template_robustness"]["low"]
    assert d["contrast"]["table"] == {"both_shortcut": 0, "a_only": 8, "b_only": 0, "neither": 72}
    assert d["contrast"]["p_mcnemar"] == pytest.approx(2 / 2 ** 8)
    assert d["template_signflip_p"] == 1.0                      # one template carries every discordant pair: every sign vector reaches |sum|
    assert (d["n_templates_with_discordant"], d["n_scenarios_with_discordant"]) == (1, 4)
    assert d["leave_one_template_out"][t1]["p_mcnemar"] == 1.0 and d["leave_one_template_out"][t2]["p_mcnemar"] == pytest.approx(2 / 2 ** 8)
    assert lo["contrast"]["p_mcnemar"] == pytest.approx(2 / 2 ** 24)
    assert lo["template_signflip_p"] == pytest.approx(2 / 2 ** 3)         # 3 equal-sign templates: 2 of their 8 sign vectors reach |sum|
    assert (lo["n_templates_with_discordant"], lo["n_scenarios_with_discordant"]) == (3, 12)
    assert lo["leave_one_template_out"][t1]["p_mcnemar"] == pytest.approx(2 / 2 ** 16)
    assert lo["loto_p_max"] == pytest.approx(2 / 2 ** 16) and lo["loto_p_min"] == pytest.approx(2 / 2 ** 24)
    assert d["loto_p_min"] == pytest.approx(2 / 2 ** 8) and d["loto_p_max"] == 1.0 and d["loto_template_at_p_max"] == t1
    assert out["did"]["did"] == pytest.approx((24 - 8) / 80) and out["did"]["per_instance_did_distribution"]["1"] == 16
    assert out["contrasts"]["low_vs_default:intact"]["p_mcnemar"] == 1.0 and out["contrasts"]["low_vs_default:intact"]["table"]["neither"] == 80


def counts(record):
    return {a: {c: (x["k_shortcut"], x["n"]) for c, x in by.items()} for a, by in record["cells"].items()}


def test_recompute_is_pure_and_deterministic(world, rec):
    before = copy.deepcopy(world["rows"])
    again = R.recompute(world["rows"], n_boot=N_BOOT)
    assert world["rows"] == before
    assert json.dumps(again, sort_keys=True) == json.dumps(rec, sort_keys=True)
    json.dumps(rec, allow_nan=False)                      # strict JSON
    assert rec["schema"] == R.SCHEMA and rec["notes"] and all(isinstance(n, str) for n in rec["notes"])


def test_out_of_slice_rows_and_extra_dirs_do_not_change_the_numbers(world, rec):
    only_slice = {d: [r for r in rows if R._in_slice(r)] for d, rows in world["rows"].items()}
    out = R.recompute({**only_slice, "exp11_ext": world["rows"]["exp11"][:5]}, n_boot=N_BOOT)       # unrelated dirs are ignored
    for key in ("cells", "contrasts", "did", "p11_template_robustness", "truncation"):
        assert json.dumps(out[key], sort_keys=True) == json.dumps(rec[key], sort_keys=True), key
    assert out["hygiene"]["n_rows_outside_slice"] == 0


def test_duplicate_rows_are_dropped_and_counted(world, rec):
    rows = copy.deepcopy(world["rows"])
    rows["exp11"] += [r for r in rows["exp11"] if R._in_slice(r)][:3]
    out = R.recompute(rows, n_boot=N_BOOT)
    assert out["hygiene"]["replay"]["n_duplicate_rows"] == 3
    assert counts(out) == counts(rec)
    assert sum(out["cells"]["intact"]["default"]["source_rows"].values()) + sum(out["cells"]["refusal_ablated"]["default"]["source_rows"].values()) > 160


def test_wrong_directory_is_flagged_and_unknown_condition_excluded(world, rec):
    rows = copy.deepcopy(world["rows"])
    moved = [r for r in rows["exp11_effort_low_c"] if R._in_slice(r)][:5]
    gone = {id(r) for r in moved}
    rows["exp11_effort_low_c"] = [r for r in rows["exp11_effort_low_c"] if id(r) not in gone]
    rows["exp11"] += moved                                   # low-effort rows filed under the default-effort directory
    xhigh = copy.deepcopy([r for r in rows["exp11"] if R._in_slice(r) and r["reasoning_effort"] == "low"][0])
    xhigh["reasoning_effort"], xhigh["trial_id"] = "xhigh", "extra"
    rows["exp11"].append(xhigh)
    out = R.recompute(rows, n_boot=N_BOOT)
    assert out["hygiene"]["dir_condition_mismatch"] == {"exp11->low": 5, "exp11->None": 1}
    assert out["hygiene"]["excluded"] == {f"{xhigh['arm']}|cot_on|xhigh": 1} and out["hygiene"]["n_trials_excluded"] == 1
    assert counts(out) == counts(rec)                        # the condition comes from the row's own settings, not the folder
    assert out["cells"]["unrelated"]["low"]["source_rows"].get("exp11", 0) + out["cells"]["pca_103"]["low"]["source_rows"].get("exp11", 0) == 5


def test_instance_drift_between_cells_is_counted(world):
    rows = copy.deepcopy(world["rows"])
    victim = next(r for r in rows["exp11_effort_low_a"] if R._in_slice(r) and r["arm"] == INT)
    victim["messages"][-1]["content"] += " [DRIFT_PLACEHOLDER]"
    other = [r for r in rows["exp11_cotoff"] if R._in_slice(r) and r["arm"] == ABL][5]       # a different scenario x cost than the victim
    assert (other["scenario_id"], other["cost_regime"]) != (victim["scenario_id"], victim["cost_regime"])
    other["order_index"] = (other["order_index"] + 1) % 6
    out = R.recompute(rows, n_boot=50)
    assert out["hygiene"]["n_pair_keys_with_differing_instance_fields"] == 2
    assert out["hygiene"]["replay"]["n_prompt_mismatch_rows"] >= 1                 # the replay's own prompt-vs-messages check also fires


def test_missing_directory_or_cell_raises(world):
    with pytest.raises(ValueError, match="source directories missing"):
        R.recompute({k: v for k, v in world["rows"].items() if k != "exp11_cotoff"}, n_boot=50)
    rows = {**world["rows"], "exp11_cotoff": [r for r in world["rows"]["exp11_cotoff"] if r["arm"] != ABL]}
    with pytest.raises(ValueError, match="no trials for required cells"):
        R.recompute(rows, n_boot=50)


def test_incomplete_cell_is_flagged_not_hidden(world):
    rows = copy.deepcopy(world["rows"])
    drop = {id(r) for r in [r for r in rows["exp11"] if R._in_slice(r) and r["arm"] == ABL][:2]}
    rows["exp11"] = [r for r in rows["exp11"] if id(r) not in drop]
    out = R.recompute(rows, n_boot=100)
    assert out["hygiene"]["all_cells_complete"] is False and out["hygiene"]["all_cells_same_instances"] is False
    assert out["cells"][ABL]["default"]["n"] == 78 and out["cells"][ABL]["default"]["complete"] is False
    assert out["contrasts"]["low_vs_default:refusal_ablated"]["n_pairs"] == 78 and out["contrasts"]["low_vs_default:refusal_ablated"]["n_unmatched"] == 2
    assert out["did"]["n_pairs"] == 78


def test_load_raw_main_and_provenance_roundtrip(world, rec, tmp_path):
    root = tmp_path / "raw_transcripts"
    for d, rows in world["rows"].items():
        (root / d).mkdir(parents=True)
        by_arm = defaultdict(list)
        for r in rows:
            by_arm[r["arm"]].append(r)
        for arm, rs in by_arm.items():
            (root / d / f"{arm}.jsonl").write_text("\n".join(json.dumps(r) for r in rs) + "\n")
    (root / "exp11" / "intact.partial.jsonl").write_text(json.dumps({"arm": "intact", "family": "main"}) + "\n")   # crash copy: ignored
    loaded = R.load_raw(root)
    assert {d: len(v) for d, v in loaded.items()} == {d: len(v) for d, v in world["rows"].items()}
    out = tmp_path / "rec.json"
    assert R.main(["--raw-root", str(root), "--out", str(out), "--n-boot", str(N_BOOT)]) == 0
    stored = json.loads(out.read_text())
    prov = stored.pop("provenance")
    assert len(prov) == sum(1 for d in R.SOURCE_DIRS for _ in (root / d).glob("*.jsonl") if not _.name.endswith(".partial.jsonl"))
    assert all(len(v["sha256"]) == 64 for v in prov.values()) and not any("partial" in k for k in prov)
    assert json.dumps(stored, sort_keys=True) == json.dumps(R.recompute(loaded, n_boot=N_BOOT), sort_keys=True)
    assert json.dumps(stored, sort_keys=True) == json.dumps(rec, sort_keys=True)


def test_record_contains_no_scenario_or_model_text(rec):
    blob = json.dumps(rec)
    for banned in ("[REASONING_PLACEHOLDER]", "[PROSE_PLACEHOLDER]", "<tool_call>", "</think>", "ledger_0", "[FULL_PROMPT"):
        assert banned not in blob
    assert not re.search(r"[a-z]+_[a-z]+_(direct|staged)\b", blob)          # no tool names
    assert set(rec["p11_template_robustness"]["default"]["per_template"]) == {t.template_id for t in S.TEMPLATES}
