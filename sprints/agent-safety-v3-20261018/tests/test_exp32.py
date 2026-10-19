"""CPU tests for v3.exp32 (position contract, probes, regimes), v3.modal_acts (job body on a toy model) and the two
Exp 3.2 scripts. Activations are synthetic, prompt text is placeholder text, nothing touches Modal."""
import importlib.util
import json
import pathlib
import re
import subprocess
import sys
import types

import numpy as np
import pytest
import torch
from torch import nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from v3 import episodes, exp31, exp32, modal_acts, scenarios, stats  # noqa: E402
from v3 import modal_hf as mh  # noqa: E402

SPECIALS = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]
SPECIAL_RE = re.compile("(" + "|".join(map(re.escape, SPECIALS)) + ")")


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EFFORT_TEXT = {"xhigh": "Reasoning effort is set to xhigh. Please think carefully through the task.",
               "low": "Reasoning effort is set to low. Keep your thinking brief."}


class StubTok:
    """Qwen-shaped template: specials are single tokens, every other character is one token (id = ord % 250 + 1).
    Like the real Qwen3.8 template (tokenizer_audit/chat_template.jinja) it prepends an effort block to the system
    message ONLY when thinking is on and the effort (default xhigh) is xhigh or low; 'medium' adds nothing.
    `odd` adds a stray line to the cot_on prompt only (a difference that is not the effort block)."""
    pad_id = 0
    padding_side = "right"

    def __init__(self, effort_block=True, bare_off=False, drop_newline=False, odd=False):
        self.effort_block, self.bare_off, self.drop_newline, self.odd = effort_block, bare_off, drop_newline, odd

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=True, reasoning_effort=None):
        block = ""
        if self.effort_block and enable_thinking:
            text = EFFORT_TEXT.get(reasoning_effort or "xhigh")
            block = text + "\n\n" if text else ""
        out = ""
        for k, m in enumerate(messages):
            lead = block if k == 0 else ""
            out += f"<|im_start|>{m['role']}\n{lead}{m['content']}<|im_end|>\n"
        if self.odd and enable_thinking:
            out += "<|im_start|>user\nSTRAY<|im_end|>\n"
        out += "<|im_start|>assistant" + ("" if self.drop_newline else "\n")
        if enable_thinking:
            return out + "<think>\n"
        return out + ("" if self.bare_off else "<think>\n\n</think>\n\n")

    def convert_tokens_to_ids(self, tok):
        return 251 + SPECIALS.index(tok)

    def _ids(self, text):
        ids = []
        for piece in SPECIAL_RE.split(text):
            if piece in SPECIALS:
                ids.append(self.convert_tokens_to_ids(piece))
            else:
                ids += [ord(c) % 250 + 1 for c in piece]
        return ids

    def __call__(self, texts, return_tensors=None, padding=False, add_special_tokens=False):
        if isinstance(texts, str):
            return {"input_ids": self._ids(texts)}
        ids = [torch.tensor(self._ids(t)) for t in texts]
        width = max(len(i) for i in ids)
        input_ids = torch.zeros(len(ids), width, dtype=torch.long)
        mask = torch.zeros(len(ids), width, dtype=torch.long)
        for k, i in enumerate(ids):
            input_ids[k, :len(i)], mask[k, :len(i)] = i, 1
        return _Enc({"input_ids": input_ids, "attention_mask": mask})


class _Enc(dict):
    def to(self, _device):
        return self


MSGS = [{"role": "system", "content": "[SYSTEM_PLACEHOLDER]"}, {"role": "user", "content": "[USER_PLACEHOLDER_a]"}]


# ---- position contract ------------------------------------------------------------------------------------------
def test_pre_think_position_is_the_same_token_in_cot_on_and_cot_off():
    tok = StubTok()
    think = tok.convert_tokens_to_ids("<think>")
    for effort, relation in ((None, "effort_block"), ("xhigh", "effort_block"), ("low", "effort_block"), ("medium", "identical")):
        on = exp32.render_generation_prompt(tok, MSGS, True, effort)
        off = exp32.render_generation_prompt(tok, MSGS, False, effort)
        assert on.endswith("<think>\n") and off.endswith("<think>\n\n</think>\n\n") and on != off
        p_on, p_off = exp32.pre_think_prefix(on, True), exp32.pre_think_prefix(off, False)
        assert p_on.endswith("<|im_start|>assistant\n") and p_off.endswith("<|im_start|>assistant\n")      # the position
        assert exp32.relate_prefixes(p_on, p_off) == relation == ("identical" if p_on == p_off else "effort_block")
        facts = exp32.check_pre_think_position(tok, on, off, think_id=think)
        assert facts["relation"] == relation and facts["last_pre_id"] == ord("\n") % 250 + 1 and facts["next_id"] == think
        assert facts["n_pre_tokens"] == len(tok(p_on)["input_ids"]) and facts["n_pre_tokens_off"] == len(tok(p_off)["input_ids"])
        assert (facts["n_pre_tokens"] == facts["n_pre_tokens_off"]) == (relation == "identical")
        for text, prefix in ((on, p_on), (off, p_off)):                                 # token level, per mode
            ids, pre = tok(text)["input_ids"], tok(prefix)["input_ids"]
            assert ids[: len(pre)] == pre and ids[len(pre)] == think and pre[-1] == facts["last_pre_id"]
    t0 = StubTok(effort_block=False)                                                    # a template without the block: identical
    assert exp32.check_pre_think_position(t0, exp32.render_generation_prompt(t0, MSGS, True, "xhigh"),
                                          exp32.render_generation_prompt(t0, MSGS, False, "xhigh"))["relation"] == "identical"


def test_position_contract_fails_loudly_on_any_other_drift():
    on = lambda t, e=None: exp32.render_generation_prompt(t, MSGS, True, e)
    off = lambda t, e=None: exp32.render_generation_prompt(t, MSGS, False, e)
    odd = StubTok(odd=True)
    with pytest.raises(ValueError, match="more than the reasoning-effort block"):
        exp32.check_pre_think_position(odd, on(odd), off(odd))
    with pytest.raises(ValueError, match="more than the reasoning-effort block"):          # the block moved elsewhere
        exp32.relate_prefixes("AAA Reasoning effort is set to low. BBB", "AAA BBB x")
    b = StubTok(bare_off=True)
    with pytest.raises(ValueError, match="expected think block"):
        exp32.check_pre_think_position(b, on(b), off(b))
    with pytest.raises(ValueError, match="not <think>"):
        exp32.check_pre_think_position(StubTok(), on(StubTok()), off(StubTok()), think_id=1)
    with pytest.raises(ValueError, match="think token inside"):
        exp32.pre_think_prefix("<think>\nxx<think>\n", True)

    class Merging(StubTok):                      # a tokenizer that merges the newline into the next token breaks the prefix property
        def _ids(self, text):
            return super()._ids(text.replace("assistant\n<think>", "assistant<think>"))
    m = Merging()
    with pytest.raises(ValueError, match="not a prefix of the full prompt tokenisation"):
        exp32.check_pre_think_position(m, on(m, "medium"), off(m, "medium"))


def test_plan_pre_think_collects_prefixes_per_mode_and_hashes_the_rendering():
    prompts = [{"id": f"p{i}", "messages": [MSGS[0], {"role": "user", "content": "[USER_PLACEHOLDER_" + "z" * i + "]"}]} for i in range(5)]
    tok = StubTok()
    think = tok.convert_tokens_to_ids("<think>")
    plan = exp32.plan_pre_think(tok, prompts, "xhigh", think)
    assert plan["relation"] == "effort_block" and len(plan["prefix_on"]) == len(plan["prefix_off"]) == 5
    assert all(p.endswith("assistant\n") for p in plan["prefix_on"] + plan["prefix_off"])
    assert plan["n_pre_on"] == [len(tok(p)["input_ids"]) for p in plan["prefix_on"]]
    assert plan["n_pre_off"] == [len(tok(p)["input_ids"]) for p in plan["prefix_off"]] and all(a > b for a, b in zip(plan["n_pre_on"], plan["n_pre_off"]))
    assert plan["last_pre_ids"] == [11] and plan["next_ids"] == [think]
    assert exp32.plan_pre_think(tok, prompts, "medium")["relation"] == "identical"
    assert exp32.plan_pre_think(tok, prompts, "xhigh")["rendered_sha256"] == plan["rendered_sha256"] != exp32.plan_pre_think(tok, prompts, "low")["rendered_sha256"]
    other = [dict(prompts[0], messages=[MSGS[0], {"role": "user", "content": "[CHANGED]"}])] + prompts[1:]
    assert exp32.plan_pre_think(tok, other, "xhigh")["rendered_sha256"] != plan["rendered_sha256"]
    with pytest.raises(ValueError, match="prompt p0: .*more than the reasoning-effort block"):
        exp32.plan_pre_think(StubTok(odd=True), prompts, "xhigh")
    bad = [dict(prompts[0], messages=[MSGS[0], {"role": "user", "content": "x </think> y"}])]
    with pytest.raises(ValueError, match="forbidden literal"):
        exp32.plan_pre_think(tok, bad, "xhigh")


REAL_TEMPLATE = ROOT.parent / "agent-safety-clean-v2-20261017/checkpoint_doses_v1/tokenizer_audit/chat_template.jinja"


@pytest.mark.skipif(not REAL_TEMPLATE.exists(), reason="donor copy of the Qwen3.8 chat template not available")
def test_real_qwen_template_has_an_effort_block_in_cot_on_only():
    """The spec's 'identical for cot_on and cot_off' is false for the real template at xhigh / low / default."""
    jinja2 = pytest.importorskip("jinja2")
    from jinja2.ext import loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
    template = env.from_string(REAL_TEMPLATE.read_text())

    class RealTok(StubTok):
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=True, reasoning_effort=None):
            kw = {} if reasoning_effort is None else {"reasoning_effort": reasoning_effort}
            return template.render(messages=messages, add_generation_prompt=add_generation_prompt, enable_thinking=enable_thinking, **kw)
    tok = RealTok()
    seen = {}
    for effort in (None, "low", "medium", "xhigh"):
        on = exp32.render_generation_prompt(tok, MSGS, True, effort)
        off = exp32.render_generation_prompt(tok, MSGS, False, effort)
        seen[effort] = exp32.check_pre_think_position(tok, on, off, tok.convert_tokens_to_ids("<think>"))["relation"]
    assert seen == {None: "effort_block", "low": "effort_block", "medium": "identical", "xhigh": "effort_block"}
    assert exp32.pre_think_prefix(exp32.render_generation_prompt(tok, MSGS, False, None), False).startswith("<|im_start|>system\n[SYSTEM_PLACEHOLDER]")


def test_render_matches_the_serving_path():
    from v3 import modal_serve
    tok = StubTok()
    for cot in ("cot_on", "cot_off"):
        for effort in (None, "low", "xhigh"):
            req = {"messages": MSGS, "cot_mode": cot, "reasoning_effort": effort}
            assert exp32.render_generation_prompt(tok, MSGS, cot == "cot_on", effort) == modal_serve._render(tok, req, "qwen")


# ---- prompt sets, vectors, files ---------------------------------------------------------------------------------
BASES = exp31.placeholder_bases(40)
VERDICTS = exp31.placeholder_verdicts(BASES)
GRID = scenarios.build_exp11_main_grid(BASES)
EXP31 = exp31.build_exp31_instances(BASES, VERDICTS)


def test_prompt_set_dedupes_the_exp11_grid_against_exp31_and_carries_no_text():
    prompts, index = exp32.build_prompt_set({"exp11": GRID, "exp31": EXP31})
    assert len(GRID) == 480 and len(prompts) == 480 + 288 == len(index)     # the narrative Exp 3.1 prompts ARE Exp 1.1 prompts
    shared = [v for v in index.values() if {m["source"] for m in v} == {"exp11", "exp31"}]
    assert len(shared) == 288
    assert all(set(p) == {"id", "messages"} for p in prompts) and len({p["id"] for p in prompts}) == len(prompts)
    assert "PLACEHOLDER" not in json.dumps(index)
    numeric = [v for v in index.values() if v[0]["source"] == "exp31"]
    assert len(numeric) == 288 and all(v[0]["fmt"] == "numeric" for v in numeric)


def test_r_hat_loader_verifies_shape_and_norm(tmp_path):
    v = np.random.default_rng(0).normal(size=16)
    v /= np.linalg.norm(v)
    np.savez(tmp_path / "d.npz", r_hat=v, refusal=v, short=v[:5], big=v * 2, nan=np.full(16, np.nan), best_layer=np.array([48]))
    assert np.allclose(exp32.load_r_hat(tmp_path / "d.npz", "r_hat", 16), v)
    assert exp32.RHAT_KEYS == {"r1": "r_hat", "r2": "refusal"} and exp32.r_hat_path("r1") == "/artifacts/agent-safety-three-track-20261016/phase1/phase1_directions.npz"
    for key, dim in (("short", 16), ("big", 16), ("nan", 16), ("r_hat", 17)):
        with pytest.raises(ValueError):
            exp32.load_r_hat(tmp_path / "d.npz", key, dim)
    with pytest.raises(KeyError):
        exp32.load_r_hat(tmp_path / "d.npz", "missing")


def test_save_and_load_acts_roundtrip_with_every_cot_off_state(tmp_path):
    rng = np.random.default_rng(1)
    pin, pout, r = rng.normal(size=(3, 8)), rng.normal(size=(3, 8)), np.eye(8)[0]
    poff_in, poff_out = rng.normal(size=(3, 8)), rng.normal(size=(3, 8))
    ids = ["a", "b", "c"]
    sha = exp32.save_acts(tmp_path / "t" / "acts_intact.npz", ids, pin, pout, [5, 6, 7], [11, 11, 11], r, {"arm": "intact", "cot_off": "separate"},
                          pre_in_off=poff_in, pre_out_off=poff_out, n_tokens_off=[4, 5, 6])
    exp32.save_acts(tmp_path / "t" / "acts_same.npz", ids, pin, pout, [5, 6, 7], [11, 11, 11], r, {"cot_off": "same"})
    exp32.save_acts(tmp_path / "t" / "acts_none.npz", ids, pin, pout, [5, 6, 7], [11, 11, 11], r, {"cot_off": "absent"})
    got = exp32.load_acts(tmp_path / "t")
    sep = got["intact"]
    assert sep["sha256"] == sha and sep["ids"] == ids and sep["index"] == {"a": 0, "b": 1, "c": 2}
    assert sep["in"].dtype == np.float32 and np.allclose(sep["out"], pout, atol=1e-6) and sep["meta"]["arm"] == "intact"
    assert np.allclose(sep["out_off"], poff_out, atol=1e-6) and not np.allclose(sep["out_off"], sep["out"])      # own cot_off states
    assert got["same"]["out_off"] is got["same"]["out"] and got["none"]["out_off"] is None and got["none"]["in_off"] is None
    bad = {"cot_off": "separate"}
    for kwargs, meta in (({}, bad), ({"pre_in_off": pin, "pre_out_off": pout}, bad), ({"pre_in_off": pin, "pre_out_off": pout, "n_tokens_off": [1, 2, 3]}, {"cot_off": "same"}),
                         ({}, {}), ({}, {"cot_off": "weird"})):
        with pytest.raises(ValueError):
            exp32.save_acts(tmp_path / "bad.npz", ids, pin, pout, [1, 2, 3], [1, 1, 1], r, meta, **kwargs)
    with pytest.raises(ValueError):
        exp32.save_acts(tmp_path / "bad.npz", ["a"], pin, pout, [1], [1], r, {"cot_off": "absent"})


def test_template_folds_keep_templates_whole():
    tab = {"domain": np.array([t.domain for t in scenarios.TEMPLATES for _ in range(4)]),
           "template": np.array([t.template_id for t in scenarios.TEMPLATES for _ in range(4)])}
    folds = exp32.template_folds(tab, 5)
    assert sorted(set(folds)) == [0, 1, 2, 3, 4]
    for t in set(tab["template"]):
        assert len(set(folds[tab["template"] == t])) == 1                    # a template is never split
    assert all(len(set(tab["template"][folds == f])) == 2 for f in range(5))
    assert len(set(exp32.template_folds({"domain": tab["domain"][:8], "template": tab["template"][:8]}, 5))) == 2


# ---- AUROC helpers ----------------------------------------------------------------------------------------------
def naive_stratified(y, s, sid):
    num = den = 0.0
    for k in np.unique(sid):
        m = sid == k
        pos, neg = s[m][y[m] == 1], s[m][y[m] == 0]
        if len(pos) and len(neg):
            num += sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
            den += len(pos) * len(neg)
    return num / den


def test_stratified_auroc_matches_pair_counting_and_constant_scores_give_half():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = int(rng.integers(8, 50))
        y, s = rng.integers(0, 2, n), np.round(rng.normal(size=n), int(rng.integers(0, 3)))
        sid = rng.integers(0, 4, n)
        if all(0 < y[sid == k].sum() < (sid == k).sum() for k in np.unique(sid)):
            assert exp32.stratified_auroc(y, s, sid) == pytest.approx(naive_stratified(y, s, sid), abs=1e-12)
    y = np.array([0, 1] * 20)
    sid = np.repeat(np.arange(4), 10)
    assert exp32.stratified_auroc(y, np.zeros(40), sid) == 0.5
    assert exp32.stratified_auroc(y, np.repeat([0.1, 0.2, 0.3, 0.4], 10), sid) == 0.5            # constant inside every stratum
    assert exp32.stratified_auroc(y, y * 2.0 + sid, sid) == 1.0
    s = rng.normal(size=40) + y
    assert exp32.stratified_auroc(y, s, np.zeros(40)) == pytest.approx(stats.auroc(y, s))


def test_stratified_ci_with_one_stratum_reproduces_auroc_bootstrap_ci():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 80)
    s = rng.normal(size=80) + 0.6 * y
    clusters = np.arange(80) // 4
    ref = stats.auroc_bootstrap_ci(y, s, 400, 11, clusters)
    got = exp32.stratified_auroc_ci(y, {"a": s, "b": -s}, np.zeros(80), clusters, 400, 11, ref="a")
    assert got["a"]["auroc"] == pytest.approx(ref["auroc"]) and got["a"]["ci_low"] == pytest.approx(ref["ci_low"])
    assert got["a"]["ci_high"] == pytest.approx(ref["ci_high"])
    assert got["b"]["diff_ref_minus_this"]["diff"] == pytest.approx(2 * ref["auroc"] - 1)


def test_lookup_baseline_is_the_saturated_cell_model():
    tr_cells = np.array(["a"] * 10 + ["b"] * 10)
    y = np.array([1] * 9 + [0] + [0] * 8 + [1] * 2)
    got = exp32.lookup_scores(tr_cells, y, np.array(["a", "b", "zzz"]), shrink=1.0)
    p0 = y.mean()
    assert got[0] == pytest.approx((9 + p0) / 11) and got[1] == pytest.approx((2 + p0) / 11) and got[2] == pytest.approx(p0)


# ---- labels and the joined table --------------------------------------------------------------------------------
def call(name, **args):
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": args}) + "\n</tool_call>"


def episode_rows(instances, arm, cot, decide):
    """Scripted episodes over grid / Exp 3.1 instances; decide(inst) -> 'safe' | 'direct' | 'trunc'."""
    by_id = {i["trial_id"]: i for i in instances}

    def gen(reqs):
        outs = []
        for r in reqs:
            tid, _, turn = r["key"].rpartition(":turn")
            inst, what = by_id[tid], decide(by_id[tid])
            if what == "trunc":
                raw, fin = "reasoning that never closes", "length"
            elif what == "direct":
                raw, fin = ("done\n</think>\n" if cot == "cot_on" else "") + call(inst["direct_tool"], workspace="w"), "stop"
            else:
                step = inst["safe_steps"][int(turn) - 1]
                raw, fin = ("done\n</think>\n" if cot == "cot_on" else "") + call(inst["safe_tool"], operation=step["operation"], step=step["step"]), "stop"
            outs.append({"raw_output": raw, "generated_tokens": 9, "finish_reason": fin, "full_prompt": "[FULL_PROMPT]"})
        return outs
    return episodes.run_episodes(gen, instances, arm=arm, exp_id="t", cot_mode=cot, reasoning_effort="xhigh",
                                 max_new_tokens=2048 if cot == "cot_on" else 512)


def test_label_rows_replay_outcomes_and_exclude_truncation():
    sub = [i for i in GRID if i["scenario_id"] in ("ledger_00", "queue_01")]
    rows = episode_rows(sub, "intact", "cot_on", lambda i: "trunc" if i["scenario_id"] == "queue_01" and i["cost_regime"] == "low" else
                        ("direct" if i["consequence"] == "harmless" else "safe"))
    rows[0]["direct_taken"] = not rows[0]["direct_taken"]                 # a stale stored field must not matter
    broken = dict(rows[1], turns=[])
    labels, bad = exp32.label_rows(rows + [broken, {"trial_id": "single", "family": "main"}])
    assert bad == 1 and len(labels) == len(rows)
    trunc = [r for r in labels if r["y"] is None]
    assert len(trunc) == 4 and all(r["scenario"] == "queue_01" and r["cost"] == "low" and r["route"] == "TRUNCATED_OR_MALFORMED" for r in trunc)
    assert all(r["y"] == int(r["cons"] == "harmless") for r in labels if r["y"] is not None)
    assert {r["source"] for r in labels} == {"exp11"} and {r["fmt"] for r in labels} == {"narrative"}
    num = [i for i in EXP31 if i["meta"]["stake_format"] == "numeric"][:6]
    lab31, _ = exp32.label_rows(episode_rows([dict(i, trial_id="x31:cot_on:" + i["trial_id"]) for i in num], "intact", "cot_on", lambda i: "safe"))
    assert {r["source"] for r in lab31} == {"exp31"} and {r["fmt"] for r in lab31} == {"numeric"}


def synth(seed=0, signal=0.0, D=24, arms=("intact", "refusal_ablated"), p_cell=None, between=1.0, rhat_signal=0.0, cot="cot_on", extra_offset=0.0):
    """Synthetic labelled rows + activations over the high-cost implicit cells of the Exp 1.1 grid (4 cells x 40 bases per arm).
    y ~ Bernoulli(p_cell); activation = cell vector + signal along w * (y - .5) + noise (+ a planted r_hat component in Intact)."""
    rng = np.random.default_rng(seed)
    grid = [i for i in GRID if i["boundary"] == "implicit" and i["cost_regime"] != "low"]
    w, r_hat = rng.normal(size=D), rng.normal(size=D)
    w, r_hat = w / np.linalg.norm(w), r_hat / np.linalg.norm(r_hat)
    cells = sorted({(a, i["cost_regime"], i["consequence"]) for a in arms for i in grid})
    p_cell = p_cell or {c: 0.5 for c in cells}
    cell_vec = {c: rng.normal(size=D) * 2 + between * (p_cell[c] - 0.5) * 6 * rng.normal(size=D) + extra_offset * rng.normal(size=D) for c in cells}
    labels, acts = [], {}
    ids = [exp32.prompt_id(i["messages"]) for i in grid]
    for a in arms:
        X = np.zeros((len(grid), D), np.float32)
        for k, inst in enumerate(grid):
            c = (a, inst["cost_regime"], inst["consequence"])
            y = int(rng.random() < p_cell[c])
            X[k] = cell_vec[c] + signal * 2 * (y - 0.5) * w - rhat_signal * 2 * (y - 0.5) * r_hat * (a == "intact") + rng.normal(size=D)
            labels.append({"source": "exp11", "arm": a, "cot_mode": cot, "trial_id": inst["trial_id"], "template": inst["template_id"],
                           "scenario": inst["scenario_id"], "domain": inst["domain"], "cost": inst["cost_regime"], "cons": inst["consequence"],
                           "bnd": inst["boundary"], "order": inst["order_index"], "fmt": "narrative", "pid": ids[k],
                           "text": "\n".join(m["content"] for m in inst["messages"]), "y": y, "route": "SHORTCUT" if y else "SAFE_ROUTE", "sub_status": "x"})
        acts[a] = {"ids": ids, "index": {p: k for k, p in enumerate(ids)}, "in": X, "out": X, "in_off": X, "out_off": X, "r_hat": r_hat,
                   "n_tokens": np.zeros(len(ids), int), "last_ids": np.zeros(len(ids), int),
                   "meta": {"arm": a, "effort": None, "cot_off": "same"}, "sha256": "synthetic"}
    return labels, acts


CFG = {**exp32.DEFAULT_CONFIG, "n_boot": 200, "n_perm": 0, "n_folds": 5}


def test_build_table_dedupes_prefers_exp11_and_counts_every_drop():
    labels, acts = synth(0)
    dup = dict(labels[0], source="exp31", y=1 - labels[0]["y"])               # same prompt, same arm, other sweep
    trunc = dict(labels[1], y=None)
    stray = dict(labels[2], pid="not-in-acts")
    tab = exp32.build_table(labels + [dup, trunc, stray], acts, sources=("exp11", "exp31"))
    assert tab["n"] == len(labels) and tab["dropped"] == {"truncated": 1, "no_acts": 1, "duplicate_prompt": 1}
    k = [r["trial_id"] for r in tab["rows"]].index(labels[0]["trial_id"])
    assert tab["y"][k] == labels[0]["y"] and set(tab["source"]) == {"exp11"}
    only11 = exp32.build_table(labels + [dup], acts, sources=("exp11",))
    assert only11["dropped"]["duplicate_prompt"] == 0
    assert tab["X"].dtype == np.float32 and tab["proj"].shape == (tab["n"],)
    assert np.allclose(tab["proj"], tab["X"].astype(np.float64) @ acts["intact"]["r_hat"])
    other = {a: dict(v) for a, v in acts.items()}
    other["refusal_ablated"]["r_hat"] = -acts["intact"]["r_hat"]
    with pytest.raises(ValueError, match="different r_hat"):
        exp32.build_table(labels, other)
    short = {a: dict(v, r_hat=v["r_hat"][:-1]) for a, v in acts.items()}
    with pytest.raises(ValueError, match="r_hat has shape"):
        exp32.build_table(labels, short)
    assert exp32.build_table([], acts)["n"] == 0


# ---- Regime C --------------------------------------------------------------------------------------------------
def pooled(c, scope="arm_cot", group=None):
    gs = c["scopes"][scope]
    return next(g for g in gs if group is None or g["group"] == group)["pooled"]


def test_cot_off_rows_are_joined_to_the_cot_off_states_and_dropped_when_absent():
    labels, acts = synth(0)
    off_labels = [dict(r, cot_mode="cot_off") for r in labels]
    own = {a: dict(v, in_off=v["in"] + 100.0, out_off=v["out"] + 100.0) for a, v in acts.items()}     # distinct cot_off states
    tab = exp32.build_table(labels + off_labels, own)
    on, off = tab["cot_mode"] == "cot_on", tab["cot_mode"] == "cot_off"
    assert on.sum() == off.sum() == len(labels)
    ids = lambda mask: [(r["arm"], r["trial_id"]) for r, m in zip(tab["rows"], mask) if m]
    assert ids(on) == ids(off) and np.allclose(tab["X"][off] - 100.0, tab["X"][on], atol=1e-4)         # same prompts, other state
    absent = {a: dict(v, in_off=None, out_off=None) for a, v in acts.items()}
    none = exp32.build_table(labels + off_labels, absent)
    assert set(none["cot_mode"]) == {"cot_on"} and none["dropped"]["no_acts"] == len(labels)
    same = exp32.build_table(labels + off_labels, acts)                                                  # aliased: the cot_on state serves
    assert np.allclose(same["X"][same["cot_mode"] == "cot_off"], same["X"][same["cot_mode"] == "cot_on"])


def test_regime_c_planted_within_cell_signal_is_detected_and_design_is_exactly_half():
    labels, acts = synth(1, signal=0.7)
    c = exp32.regime_c(exp32.build_table(labels, acts), CFG)
    assert c["n_eligible"] == 8 and c["design_exactly_half_everywhere"] is True
    for g in c["scopes"]["arm_cot"]:
        p = g["pooled"]
        assert p["design"]["auroc"] == 0.5 and p["design"]["ci_low"] == p["design"]["ci_high"] == 0.5
        assert p["probe"]["auroc"] > 0.65 and p["probe"]["ci_low"] > 0.5            # CI excludes 0.5
        assert p["tfidf"]["diff_ref_minus_this"]["ci_low"] > 0                      # probe beats the text baseline within cells
        assert p["tfidf"]["ci_low"] < 0.5 < p["tfidf"]["ci_high"] and p["rhat"]["ci_low"] < 0.5 < p["rhat"]["ci_high"]
        assert all(s["auroc"]["design"]["auroc"] == 0.5 for s in g["strata"])
    assert pooled(c, "cot")["probe"]["ci_low"] > 0.5
    assert len(c["scopes"]["cell"]) == 8


def test_regime_c_null_ci_covers_half_even_with_strong_between_cell_structure():
    covered = 0
    for seed in range(8):
        # cell label rates differ a lot and the activations encode the cell: a leaky probe would score well above 0.5
        cells = [(a, c, k) for a in ("intact", "refusal_ablated") for c in ("high_tedium", "high_friction") for k in ("harmless", "harmful")]
        p_cell = {c: p for c, p in zip(cells, [0.9, 0.5, 0.2, 0.5, 0.8, 0.3, 0.5, 0.1])}
        labels, acts = synth(100 + seed, signal=0.0, p_cell=p_cell, between=2.0, extra_offset=3.0)
        c = exp32.regime_c(exp32.build_table(labels, acts), {**CFG, "n_boot": 150, "min_minority": 4, "c_scopes": ("cot",), "c_methods": ("probe", "design")})
        p = pooled(c, "cot")["probe"]
        covered += p["ci_low"] <= 0.5 <= p["ci_high"]
        assert pooled(c, "cot")["design"]["auroc"] == 0.5
    assert covered >= 6                                                        # nominal 95%: allow chance misses


def test_regime_c_never_sees_cell_or_arm_identity():
    labels, acts = synth(5, signal=0.5)
    base = exp32.regime_c(exp32.build_table(labels, acts), {**CFG, "c_scopes": ("arm_cot",)})
    shifted = {a: dict(v, **{"out": v["out"] + 50.0 * np.random.default_rng(i).normal(size=v["out"].shape[1]).astype(np.float32)})
               for i, (a, v) in enumerate(acts.items())}                       # a constant, arm-specific offset on every activation
    again = exp32.regime_c(exp32.build_table(labels, shifted), {**CFG, "c_scopes": ("arm_cot",)})
    for g0, g1 in zip(base["scopes"]["arm_cot"], again["scopes"]["arm_cot"]):
        assert g1["pooled"]["probe"]["auroc"] == pytest.approx(g0["pooled"]["probe"]["auroc"], abs=2e-3)


def test_regime_c_eligibility_and_missing_cells():
    labels, acts = synth(2, signal=0.5)
    only_one_class = [dict(r, y=1) if (r["arm"], r["cost"], r["cons"]) == ("intact", "high_tedium", "harmless") else r for r in labels]
    c = exp32.regime_c(exp32.build_table(only_one_class, acts), {**CFG, "c_scopes": ("cell",)})
    assert c["n_strata"] == 8 and c["n_eligible"] == 7
    assert not any("intact|high_tedium|harmless" in e["stratum"] for e in c["eligible"])
    rare = exp32.regime_c(exp32.build_table(labels, acts), {**CFG, "min_minority": 30, "c_scopes": ("cell",)})
    assert rare["status"] == "no_eligible_cell" and rare["n_eligible"] == 0
    assert exp32.regime_c({"n": 0}, CFG)["status"] == "no_data"


def test_regime_c_permutation_null_reports_observed_and_null():
    labels, acts = synth(1, signal=0.7)
    c = exp32.regime_c(exp32.build_table(labels, acts), {**CFG, "n_perm": 6, "n_boot": 50})
    g = c["scopes"]["arm_cot"][0]
    perm = g["permutation_null"]
    assert perm["n_perm"] == 6 and perm["observed"] == g["pooled"]["probe"]["auroc"] and perm["null_q95"] < perm["observed"]
    assert perm["p_perm"] == pytest.approx(1 / 7)


# ---- Regime A / B ----------------------------------------------------------------------------------------------
def varied_p_cell():
    cells = [(a, c, k) for a in ("intact", "refusal_ablated") for c in ("high_tedium", "high_friction") for k in ("harmless", "harmful")]
    return {c: p for c, p in zip(cells, [0.95, 0.9, 0.05, 0.1, 0.9, 0.85, 0.7, 0.2])}


def test_regime_a_probe_cannot_beat_the_design_table_when_labels_are_cell_determined():
    labels, acts = synth(3, signal=0.0, p_cell=varied_p_cell(), between=2.0)
    a = exp32.regime_a(exp32.build_table(labels, acts), CFG)
    assert a["auroc"]["design"]["auroc"] > 0.8 and a["auroc"]["probe"]["auroc"] > 0.75        # the cell is decodable...
    assert a["diff"]["probe-design"]["ci_low"] <= 0 and a["probe_exceeds_design_ci"] is False  # ...but adds nothing to the table
    assert a["n_cells"] == 8 and set(a["per_arm"]) == {"intact", "refusal_ablated"}


def test_regime_a_probe_beats_design_when_activations_carry_extra_signal():
    labels, acts = synth(4, signal=0.8)                                                        # design is uninformative (p = .5 everywhere)
    a = exp32.regime_a(exp32.build_table(labels, acts), CFG)
    assert a["diff"]["probe-design"]["ci_low"] > 0 and a["probe_exceeds_design_ci"] is True
    assert exp32.regime_a({"n": 5, "y": np.array([0, 1, 0, 1, 0])}, CFG)["status"] == "too_little_data"


def test_design_plus_is_a_stronger_tabular_baseline_when_domain_matters():
    labels, acts = synth(7, signal=0.0)
    rng = np.random.default_rng(0)
    labels = [dict(r, y=int(rng.random() < (0.85 if r["domain"] in ("finance", "compute") else 0.15))) for r in labels]   # domain, not cell
    a = exp32.regime_a(exp32.build_table(labels, acts), CFG)
    assert a["auroc"]["design"]["auroc"] < 0.6 < a["auroc"]["design_plus"]["auroc"] - 0.1      # cell lookup cannot see the domain
    assert a["diff"]["probe-design_plus"]["ci_low"] <= 0                                        # nor can a probe of cell-only activations


def test_regime_b_projection_on_r_hat_inside_intact():
    labels, acts = synth(6, signal=0.0, rhat_signal=0.9)
    tab = exp32.build_table(labels, acts)
    b = exp32.regime_b(tab, CFG)
    assert b["n"] == 160 and b["auroc"]["rhat_neg"]["ci_low"] > 0.5 and b["auroc"]["rhat_cv"]["ci_low"] > 0.5
    assert b["rhat_neg_within_cell"]["rhat_neg"]["auroc"] > 0.6 and b["rhat_neg_within_cell"]["design"]["auroc"] == 0.5
    assert set(b["auroc"]) == {"rhat_neg", "rhat_cv", "design", "design_plus", "probe", "tfidf"} and "higher refusal-direction" in b["orientation"]
    null = exp32.regime_b(exp32.build_table(*synth(6, signal=0.0, rhat_signal=0.0)), CFG)
    assert null["auroc"]["rhat_neg"]["ci_low"] < 0.5 < null["auroc"]["rhat_neg"]["ci_high"]
    assert exp32.regime_b(exp32.build_table(*synth(6, arms=("refusal_ablated",))), CFG)["status"] == "too_little_data"


# ---- receipt function ----------------------------------------------------------------------------------------------
def test_effort_of_labels_and_activations_must_agree():
    sub = [i for i in GRID if i["boundary"] == "implicit" and i["cost_regime"] != "low"][:40]
    rows = episode_rows(sub, "intact", "cot_on", lambda i: "safe")                    # recorded at reasoning_effort 'xhigh'
    labels, _ = exp32.label_rows(rows)
    assert {l["effort"] for l in labels} == {"xhigh"} and exp32.select_effort(labels) == "xhigh"
    low = [dict(l, effort="low") for l in labels]
    with pytest.raises(ValueError, match="several reasoning efforts"):
        exp32.select_effort(labels + low)
    assert exp32.select_effort(labels + low, "low") == "low" and exp32.select_effort(labels + [dict(l, effort=None) for l in labels], "default") is None
    with pytest.raises(ValueError, match="no cot_on labels"):
        exp32.select_effort(labels, "medium")
    assert exp32.select_effort([dict(l, cot_mode="cot_off", effort=None) for l in labels]) is None
    ids = [exp32.prompt_id(i["messages"]) for i in sub]
    zeros = np.zeros((40, 4), np.float32)
    mk = lambda eff: {"intact": {"ids": ids, "index": {p: k for k, p in enumerate(ids)}, "in": zeros, "out": zeros, "in_off": None, "out_off": None,
                                 "r_hat": np.eye(4)[0], "n_tokens": np.zeros(40, int), "last_ids": np.zeros(40, int),
                                 "meta": {"effort": eff, "cot_off": "absent"}, "sha256": "s"}}
    exp32.recompute(rows, mk("xhigh"), {"n_boot": 10, "n_perm": 0, "datasets": {"exp11": ["exp11"]}})
    with pytest.raises(ValueError, match="captured at reasoning effort"):
        exp32.recompute(rows, mk(None), {"n_boot": 10, "n_perm": 0, "datasets": {"exp11": ["exp11"]}})


def test_recompute_skips_datasets_whose_sources_are_missing():
    labels, acts = synth(0, arms=("intact",))
    raw = []                                                                    # no raw rows: the label table is empty
    rec = exp32.recompute(raw, acts, {"n_boot": 20, "n_perm": 0})
    assert rec["n_label_rows"] == 0 and set(rec["datasets"]) == {"exp11", "exp11+exp31"}
    assert all(d["status"].startswith("missing sources") for d in rec["datasets"].values())


def test_recompute_end_to_end_is_deterministic_and_json_clean():
    sub = [i for i in GRID if i["boundary"] == "implicit" and i["cost_regime"] != "low"]
    rng = np.random.default_rng(0)
    plan = {i["trial_id"]: rng.choice(["safe", "direct"]) for i in sub}
    rows = []
    for arm in ("intact", "refusal_ablated"):
        rows += episode_rows(sub, arm, "cot_on", lambda i, arm=arm: plan[i["trial_id"]] if arm == "intact" or i["consequence"] == "harmless" else "direct")
    rows += episode_rows([i for i in sub if i["scenario_id"] == "ledger_00"], "intact", "cot_on", lambda i: "trunc")[:0]
    labels, _ = exp32.label_rows(rows)
    ids = [exp32.prompt_id(i["messages"]) for i in sub]
    D = 12
    Xr = np.random.default_rng(1).normal(size=(len(sub), D)).astype(np.float32)
    r_hat = np.eye(D)[0]
    acts = {a: {"ids": ids, "index": {p: k for k, p in enumerate(ids)}, "in": Xr, "out": Xr + k, "in_off": None, "out_off": None, "r_hat": r_hat,
                "n_tokens": np.full(len(ids), 7), "last_ids": np.full(len(ids), 11),
                "meta": {"arm": a, "effort": "xhigh", "cot_off": "absent"}, "sha256": "s"} for k, a in enumerate(("intact", "refusal_ablated"))}
    cfg = {"n_boot": 100, "n_perm": 2, "datasets": {"exp11": ["exp11"]}, "c_scopes": ["arm_cot"], "c_methods": ["probe", "design"]}
    a = exp32.recompute(rows, acts, cfg)
    b = exp32.recompute(json.loads(json.dumps(rows)), acts, cfg)
    assert json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)
    site = a["datasets"]["exp11"]["sites"]["out"]
    assert site["n"] == len(labels) and set(site["regimes"]) == {"cot_on"}
    assert set(site["regimes"]["cot_on"]) == {"A", "B", "C"} and set(a["datasets"]["exp11"]["sites"]["in"]["regimes"]["cot_on"]) == {"A", "B"}
    assert a["acts"]["intact"]["last_token_ids"] == [11] and a["config"]["n_perm"] == 2 and a["n_unreplayable"] == 0


# ---- the Modal job body on a toy model ---------------------------------------------------------------------------------
class ToyLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d, bias=False)

    def forward(self, hidden_states):
        t = hidden_states.shape[1]
        mean = hidden_states.cumsum(1) / torch.arange(1, t + 1)[None, :, None]
        return hidden_states + self.lin(mean)


class ToyInner(nn.Module):
    def __init__(self, d, n_layers):
        super().__init__()
        self.embed = nn.Embedding(256, d)
        self.layers = nn.ModuleList([ToyLayer(d) for _ in range(n_layers)])

    def forward(self, input_ids):
        h = self.embed(input_ids)
        for layer in self.layers:
            h = layer(h)
        return h


class ToyLM(nn.Module):
    def __init__(self, d=16, n_layers=4, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.model = ToyInner(d, n_layers)

    def get_input_embeddings(self):
        return self.model.embed

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        return self.model(input_ids)


class _Vol:
    def reload(self):
        pass

    def commit(self):
        pass


def run_capture(tmp_path, monkeypatch, tok_factory=StubTok, arms=("intact", "refusal_ablated"), n_prompts=6, return_arrays=True, d=16,
                r_dim=16, built=None, effort="xhigh", modes=None):
    seeds = {"intact": 11, "refusal_ablated": 12, "pca_101": 13}
    fake_tf = types.ModuleType("transformers")
    fake_tf.__version__ = "stub"
    fake_tf.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda path: tok_factory())
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    built = [] if built is None else built
    monkeypatch.setattr(mh, "art_vol", _Vol())
    monkeypatch.setattr(mh, "build_model", lambda base, dev: built.append(1) or ToyLM(d, seed=99))
    monkeypatch.setattr(mh, "check_config_matches_base", lambda *a, **k: None)
    monkeypatch.setattr(mh, "load_checkpoint_into_model", lambda m, path: m.load_state_dict(ToyLM(d, seed=seeds[path]).state_dict()) and {})
    monkeypatch.setattr(modal_acts, "ACT_DIR", str(tmp_path / "acts"))
    r = np.random.default_rng(0).normal(size=r_dim)
    np.savez(tmp_path / "dirs.npz", r_hat=r / np.linalg.norm(r))
    monkeypatch.setattr(exp32, "r_hat_path", lambda family: str(tmp_path / "dirs.npz"))
    prompts = [{"id": f"p{i}", "messages": [MSGS[0], {"role": "user", "content": "[USER_PLACEHOLDER_" + "z" * (i * 3) + "]"}]} for i in range(n_prompts)]
    modes = modes or {}
    job = {"tag": "t", "shard_id": 0, "arms": [{"name": a, "path": a, "modes": modes.get(a, ["cot_on", "cot_off"])} for a in arms],
           "prompts": prompts, "effort": effort, "layer": 2, "batch_size": 1, "family": "r1", "return_arrays": return_arrays}
    return modal_acts.capture_acts.local(job), prompts, built


def toy_state(toy, tok, text, n_layers=3):
    ids = torch.tensor([tok(text)["input_ids"]])
    with torch.no_grad():
        h = toy.model.embed(ids)
        for layer in toy.model.layers[:n_layers]:
            h = layer(h)
    return h[0, -1].numpy()


def test_capture_job_forwards_each_mode_on_its_own_pre_think_prefix(tmp_path, monkeypatch):
    result, prompts, built = run_capture(tmp_path, monkeypatch, modes={"refusal_ablated": ["cot_on"]})
    assert built == [1] and set(result["arms"]) == {"intact", "refusal_ablated"} and result["prompt_ids"] == [p["id"] for p in prompts]
    pc = result["position_check"]
    assert pc["relation"] == "effort_block" and pc["last_pre_ids"] == [11] and pc["effort"] == "xhigh"
    assert pc["n_pre_tokens_cot_on"][0] > pc["n_pre_tokens_cot_off"][1]                # the effort block is in front of cot_on only
    tok = StubTok()
    on = [exp32.pre_think_prefix(exp32.render_generation_prompt(tok, p["messages"], True, "xhigh"), True) for p in prompts]
    off = [exp32.pre_think_prefix(exp32.render_generation_prompt(tok, p["messages"], False, "xhigh"), False) for p in prompts]
    assert on != off
    toy = ToyLM(16, seed=11).eval()
    intact = result["arms"]["intact"]["arrays"]
    for text_on, text_off, r_on, r_off in zip(on, off, intact["pre_out"], intact["pre_out_off"]):
        assert np.allclose(r_on, toy_state(toy, tok, text_on), atol=1e-5)               # output of layers[2] at the last cot_on prefix token
        assert np.allclose(r_off, toy_state(toy, tok, text_off), atol=1e-5)             # ... and at the last cot_off prefix token
    assert not np.allclose(intact["pre_out"], intact["pre_out_off"]) and result["arms"]["intact"]["meta"]["cot_off"] == "separate"
    ablated = result["arms"]["refusal_ablated"]
    assert ablated["meta"]["cot_off"] == "absent" and "pre_out_off" not in ablated["arrays"]            # this arm has no cot_off labels
    assert ablated["arrays"]["pre_in"].dtype == np.float32 and not np.allclose(ablated["arrays"]["pre_out"], intact["pre_out"])
    assert list(intact["last_ids"]) == [11] * 6 and list(intact["n_tokens"]) == [len(tok(t)["input_ids"]) for t in on]
    assert list(intact["n_tokens_off"]) == [len(tok(t)["input_ids"]) for t in off]
    saved = exp32.load_acts(tmp_path / "acts" / "t")
    assert set(saved) == {"intact", "refusal_ablated"} and saved["intact"]["ids"] == result["prompt_ids"]
    assert np.allclose(saved["intact"]["out"], intact["pre_out"]) and np.allclose(saved["intact"]["out_off"], intact["pre_out_off"])
    assert saved["intact"]["sha256"] == result["arms"]["intact"]["sha256"] and saved["refusal_ablated"]["out_off"] is None
    assert result["arms"]["intact"]["volume_path"] == "agent-safety-v3-20261018/acts/t/acts_intact.npz"
    assert np.isclose(np.linalg.norm(saved["intact"]["r_hat"]), 1.0) and saved["intact"]["meta"]["position"].startswith("last token")
    assert result["r_hat"]["dim"] == 16 and result["batch_check"]["agreement"]["out"]["max_rel_l2"] < 1e-5
    json.dumps({k: v for k, v in result.items() if k not in ("arms", "r_hat_vector")}, default=str)


def test_capture_job_needs_one_pass_when_the_contexts_are_identical(tmp_path, monkeypatch):
    result, prompts, _ = run_capture(tmp_path, monkeypatch, effort="medium", arms=("intact",))
    assert result["position_check"]["relation"] == "identical"
    arm = result["arms"]["intact"]
    assert arm["meta"]["cot_off"] == "same" and "pre_out_off" not in arm["arrays"]
    saved = exp32.load_acts(tmp_path / "acts" / "t")["intact"]
    assert saved["out_off"] is saved["out"] and np.allclose(saved["out"], arm["arrays"]["pre_out"])


def test_capture_job_aborts_before_loading_weights_when_the_contexts_differ_by_more_than_the_effort_block(tmp_path, monkeypatch):
    built = []
    with pytest.raises(ValueError, match="more than the reasoning-effort block"):
        run_capture(tmp_path, monkeypatch, tok_factory=lambda: StubTok(odd=True), built=built)
    assert built == []                                                          # the GPU model was never built
    assert not (tmp_path / "acts" / "t").exists() or not list((tmp_path / "acts" / "t").glob("*.npz"))


def test_capture_job_return_policy_and_r_hat_width_check(tmp_path, monkeypatch):
    for sub in ("off", "auto", "big", "w"):
        (tmp_path / sub).mkdir()
    off, _, _ = run_capture(tmp_path / "off", monkeypatch, return_arrays=False, arms=("intact",))
    assert "arrays" not in off["arms"]["intact"] and (tmp_path / "off" / "acts" / "t" / "acts_intact.npz").exists()
    auto, _, _ = run_capture(tmp_path / "auto", monkeypatch, return_arrays="auto", arms=("intact",))
    assert "arrays" in auto["arms"]["intact"] and auto["arms"]["intact"]["nbytes"] < modal_acts.RETURN_LIMIT_BYTES
    monkeypatch.setattr(modal_acts, "RETURN_LIMIT_BYTES", 10)
    big, _, _ = run_capture(tmp_path / "big", monkeypatch, return_arrays="auto", arms=("intact",))
    assert "arrays" not in big["arms"]["intact"]                                # too large for a Modal result: read from the volume
    with pytest.raises(ValueError, match="r_hat"):
        run_capture(tmp_path / "w", monkeypatch, r_dim=9)


def test_modal_side_imports_do_not_need_scikit_learn():
    code = "import sys; sys.path.insert(0, %r); import v3.modal_acts, v3.exp32; assert 'sklearn' not in sys.modules" % str(ROOT / "src")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_modal_app_wiring_and_cost():
    assert modal_acts.APP_NAME == "agent-safety-v3-20261018-acts" and modal_acts.app.name == modal_acts.APP_NAME
    assert modal_acts.ACT_DIR == "/artifacts/agent-safety-v3-20261018/acts"
    small, big = modal_acts.estimate_cost(768, 6, 3), modal_acts.estimate_cost(1536, 6, 3)
    assert 0 < small["usd"] < big["usd"] and small["usd"] <= 40 and small["forwards"] == 768 * (6 + 3)   # 3 arms get the cot_off pass
    assert modal_acts.estimate_cost(768, 6, 3, separate_cot_off=False)["forwards"] == 768 * 6              # effort 'medium': one pass
    assert modal_acts.estimate_cost(768, 6, 6)["gpu_hours"] > small["gpu_hours"]       # more containers pay more start-up
    assert small["worst_case_usd"] > small["usd"]


def test_estimate_cost_scales_with_the_exp31_prompt_set_and_matches_the_launcher():
    assert exp32.n_unique_prompts(1.0) == 768 == len(exp32.build_prompt_set({"exp11": GRID, "exp31": EXP31})[0])
    assert exp32.n_unique_prompts(0.5) == 480 + 2 * 4 * 3 * 12 - 4 * 3 * 12 == 624
    costs = [exp32.estimate_cost(s) for s in (0.5, 1.0, 2.0, 3.0)]
    assert [c["usd"] for c in costs] == sorted(c["usd"] for c in costs) and all(0 < c["usd"] <= 40 for c in costs)
    d = exp32.estimate_cost()
    assert d["forwards"] == 768 * 9 and d["n_shards"] == 3 and {"gpu_hours", "usd", "wall_hours_if_parallel", "worst_case_usd"} <= set(d)
    assert exp32.estimate_cost(n_arms=3)["usd"] < d["usd"] and exp32.estimate_cost(cells="high8")["forwards"] < d["forwards"]
    assert exp32.estimate_cost(n_prompts=100, n_arms=1, n_cot_off_arms=0)["forwards"] == 100
    assert exp32.estimate_cost(separate_cot_off=False)["usd"] < d["usd"]


# ---- scripts ---------------------------------------------------------------------------------------------------------
def test_run_exp32_dry_run_prints_counts_only_and_never_reserves(capsys, monkeypatch):
    mod = load_script("run_exp32")
    monkeypatch.setattr(mod.ledger, "reserve", lambda *a, **k: (_ for _ in ()).throw(AssertionError("dry run must not reserve")))
    assert mod.main(["--tag", "t", "--family", "r1", "--effort", "default", "--placeholder", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "unique_prompts': 768" in out and "PLACEHOLDER" not in out and "forwards=6912" in out and "intact+off" in out and "pca_102'" in out
    assert mod.main(["--tag", "t", "--family", "r1", "--effort", "medium", "--placeholder", "--dry-run"]) == 0
    assert "forwards=4608" in capsys.readouterr().out                                # contexts coincide at 'medium': one pass per prompt

    class A:
        arms, family, tag, effort, layer, batch_size, shards, no_return_arrays = ",".join(mod.ALL_ARMS), "r1", "t", "xhigh", 48, 1, 0, False
        cot_off_arms = mod.COT_OFF_ARMS
    jobs = mod.make_jobs(A, [{"id": "a", "messages": MSGS}])
    assert len(jobs) == 3 and all(len(j["arms"]) == 2 for j in jobs) and {x["name"] for j in jobs for x in j["arms"]} == set(mod.ALL_ARMS)
    assert jobs[0]["arms"][0]["path"] == "/weights/models/Qwen3.8-27B-FP8-Intact-Requant" and jobs[0]["return_arrays"] == "auto"
    modes = {x["name"]: x["modes"] for j in jobs for x in j["arms"]}
    assert modes["intact"] == modes["refusal_ablated"] == modes["pca_101"] == ["cot_on", "cot_off"] and modes["unrelated"] == ["cot_on"]
    A.arms = "intact,nonsense"
    with pytest.raises(ValueError, match="unknown arms"):
        mod.make_jobs(A, [])


def test_run_exp32_flags_inconsistent_shards():
    mod = load_script("run_exp32")
    ok = {"r_hat": {"sha256": "a"}, "position_check": {"x": 1}, "shard_id": 0, "arms": {"intact": {"meta": {"rendered_sha256": "h"}}}}
    assert mod.flags([ok, {**ok, "shard_id": 1}]) == []
    assert any("r_hat" in f for f in mod.flags([ok, {**ok, "r_hat": {"sha256": "b"}}]))
    assert any("rendered prompts" in f for f in mod.flags([ok, {**ok, "arms": {"x": {"meta": {"rendered_sha256": "other"}}}}]))


def test_run_exp32_save_outputs_reads_the_volume_copy_and_cross_checks_returned_arrays(tmp_path, monkeypatch):
    mod = load_script("run_exp32")
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    rng = np.random.default_rng(0)
    pin, pout, r = rng.normal(size=(3, 8)), rng.normal(size=(3, 8)), np.eye(8)[0]
    meta = {"arm": "intact", "cot_off": "absent", "rendered_sha256": "h"}
    remote = tmp_path / "volume" / "acts_intact.npz"
    sha = exp32.save_acts(remote, ["a", "b", "c"], pin, pout, [5, 6, 7], [11, 11, 11], r, meta)
    arrays = {"pre_in": np.asarray(pin, np.float32), "pre_out": np.asarray(pout, np.float32)}
    fake_fetch = lambda path, dest: dest.write_bytes(remote.read_bytes())

    def outs(rec_extra=None, sha256=sha):
        rec = {"sha256": sha256, "meta": meta, "volume_path": "agent-safety-v3-20261018/acts/t/acts_intact.npz", **(rec_extra or {})}
        return [{"shard_id": 0, "arms": {"intact": rec}, "prompt_ids": ["a", "b", "c"], "position_check": {"relation": "identical"},
                 "r_hat": {"sha256": "x"}, "versions": {}, "elapsed_s": 1.0}]

    class A:
        tag = "t"
    out_dir = mod.save_outputs(A, outs({"arrays": arrays}), {"a": []}, {"n": 1}, 1.0, 2.0, fetch=fake_fetch)
    run = json.loads((out_dir / "run.json").read_text())
    assert run["flags"] == [] and run["sha256"]["intact"]["local_sha256"] == sha and run["sha256"]["intact"]["arrays_returned"] is True
    assert exp32.load_acts(out_dir)["intact"]["sha256"] == sha and json.loads((out_dir / "prompts_index.json").read_text()) == {"a": []}
    bad_arrays = {"pre_in": arrays["pre_in"] + 1, "pre_out": arrays["pre_out"]}
    run = json.loads((mod.save_outputs(A, outs({"arrays": bad_arrays}), {}, {}, 1.0, 2.0, fetch=fake_fetch) / "run.json").read_text())
    assert any("returned arrays differ" in f for f in run["flags"])
    run = json.loads((mod.save_outputs(A, outs(sha256="0" * 64), {}, {}, 1.0, 2.0, fetch=fake_fetch) / "run.json").read_text())
    assert any("sha256 differs" in f for f in run["flags"])


def test_analyze_script_receipt_functions_and_main(tmp_path, capsys):
    mod = load_script("analyze_exp3")
    insts = exp31.build_exp31_instances(BASES, VERDICTS, per_domain=1)
    raw = tmp_path / "raw_transcripts" / "exp31"
    raw.mkdir(parents=True)
    rows = []
    for cot in exp31.COT_MODES:
        reqs = exp31.build_requests(insts, cot_modes=(cot,))
        for arm in ("intact", "refusal_ablated"):
            recs = episode_rows([r["instance"] for r in reqs], arm, cot, lambda i, arm=arm: "direct" if arm == "refusal_ablated" and i["cost_regime"] != "low" else "safe")
            for r, rec in zip(reqs, recs):
                rec["request_meta"], rec["request_trial_id"] = r["meta"], r["trial_id"]
            rows += recs
    for arm in ("intact", "refusal_ablated"):
        (raw / f"{arm}.jsonl").write_text("\n".join(json.dumps(r) for r in rows if r["arm"] == arm) + "\n")
    (raw / "intact.partial.jsonl").write_text("garbage that must be skipped\n")
    loaded, prov = mod.load_rows([str(raw)])
    assert len(loaded) == len(rows) and {p["file"].split("/")[-1] for p in prov} == {"intact.jsonl", "refusal_ablated.jsonl"}
    rec = mod.recompute(loaded)
    assert rec["schema"] == "exp31_v1" and rec["integrity"]["n_replayed"] == len(rows)
    out = tmp_path / "records" / "exp31_multidomain.json"
    assert mod.main(["exp31", "--raw", str(raw), "--out", str(out), "--n-boot", "100", "--no-model"]) == 0
    stored = json.loads(out.read_text())
    assert stored["provenance"]["raw"] and stored["overall"] == json.loads(json.dumps(exp31.recompute(loaded, n_boot=100, model=False)["overall"]))
    assert "PLACEHOLDER" not in capsys.readouterr().out


def test_analyze_script_exp32_reads_raw_rows_and_saved_activations(tmp_path, capsys, monkeypatch):
    mod = load_script("analyze_exp3")
    monkeypatch.setattr(exp32, "regime_c", lambda tab, cfg: {"status": "stubbed here (covered by the Regime C tests)", "n_eligible": 0})
    sub = [i for i in GRID if i["boundary"] == "implicit" and i["cost_regime"] != "low"]
    raw = tmp_path / "raw_transcripts" / "exp11"
    raw.mkdir(parents=True)
    rng = np.random.default_rng(0)
    plan = {i["trial_id"]: str(rng.choice(["safe", "direct"])) for i in sub}
    ids = [exp32.prompt_id(i["messages"]) for i in sub]
    acts_dir = tmp_path / "acts"
    for k, arm in enumerate(("intact", "refusal_ablated")):
        rows = episode_rows(sub, arm, "cot_on", lambda i: plan[i["trial_id"]])
        (raw / f"{arm}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        X = rng.normal(size=(len(sub), 8)).astype(np.float32)
        exp32.save_acts(acts_dir / f"acts_{arm}.npz", ids, X, X + k, np.full(len(ids), 7), np.full(len(ids), 11), np.eye(8)[0],
                        {"arm": arm, "effort": "xhigh", "cot_off": "absent"})
    out = tmp_path / "records" / "exp32_probes.json"
    assert mod.main(["exp32", "--acts", str(acts_dir), "--exp11-raw", str(raw), "--no-exp31", "--out", str(out),
                     "--n-perm", "0", "--n-boot", "40", "--min-minority", "4"]) == 0
    rec = json.loads(out.read_text())
    assert list(rec["datasets"]) == ["exp11"] and rec["provenance"]["raw"] and rec["config"]["n_perm"] == 0
    site = rec["datasets"]["exp11"]["sites"]["out"]
    assert site["n"] == 2 * len(sub) and set(site["regimes"]["cot_on"]) == {"A", "B", "C"} and site["regimes"]["cot_on"]["A"]["n_cells"] == 8
    again = mod.recompute_exp32(*[mod.load_rows([str(raw)])[0], exp32.load_acts(acts_dir), rec["config"]])
    assert again["datasets"] == rec["datasets"] and again["acts"] == rec["acts"]          # the receipt function reproduces the record
    printed = capsys.readouterr().out
    assert "PLACEHOLDER" not in printed and "exp11/out/cot_on A:" in printed


# ---- review additions ------------------------------------------------------------------------------------------------
def order_planted(seed, signal=0.0, strength=3.0):
    """synth() with labels driven ONLY by the tool order of the prompt (even order index -> shortcut) and activations that
    encode the order: a layout effect that the cell table cannot see but a pre-think state certainly can."""
    labels, acts = synth(seed, signal=signal)
    rng = np.random.default_rng(seed + 1000)
    order_vec = rng.normal(size=(6, acts["intact"]["out"].shape[1]))
    for r in labels:
        r["y"] = int(rng.random() < (0.9 if r["order"] % 2 == 0 else 0.1))
    k = 0
    for a in acts:
        X = acts[a]["out"].copy()
        for j, r in enumerate([r for r in labels if r["arm"] == a]):
            X[j] += strength * order_vec[r["order"]]
        acts[a] = dict(acts[a], out=X, **{"in": X})
        k += 1
    return labels, acts


def test_regime_c_order_domain_baseline_absorbs_a_pure_layout_effect():
    """Inside a cell the tool order varies from prompt to prompt and a pre-think state encodes it. A probe can look
    'better than the design table' for that reason alone; the order_domain baseline makes the confound visible."""
    labels, acts = order_planted(11)
    c = exp32.regime_c(exp32.build_table(labels, acts), {**CFG, "n_boot": 300, "c_scopes": ("arm_cot",)})
    assert c["design_exactly_half_everywhere"] is True
    for g in c["scopes"]["arm_cot"]:
        p = g["pooled"]
        assert p["design"]["auroc"] == 0.5
        assert p["order_domain"]["auroc"] > 0.7 and p["order_domain"]["ci_low"] > 0.5      # the layout baseline finds it ...
        assert p["probe"]["auroc"] > 0.65                                                    # ... and so does the probe
        assert p["order_domain"]["diff_ref_minus_this"]["ci_low"] <= 0                       # but the probe does not beat it
    # a genuine activation signal that has nothing to do with the layout is NOT explained by the baseline
    labels, acts = synth(12, signal=0.8)
    c = exp32.regime_c(exp32.build_table(labels, acts), {**CFG, "n_boot": 300, "c_scopes": ("arm_cot",)})
    for g in c["scopes"]["arm_cot"]:
        p = g["pooled"]
        assert p["probe"]["ci_low"] > 0.5 and p["order_domain"]["auroc"] < 0.55          # layout carries no signal here
        assert p["order_domain"]["diff_ref_minus_this"]["ci_low"] > 0


def test_capture_job_return_budget_is_per_container_not_per_arm(tmp_path, monkeypatch):
    for sub in ("one", "all"):
        (tmp_path / sub).mkdir()
    arms = ("intact", "refusal_ablated", "pca_101")
    modes = {a: ["cot_on"] for a in arms}
    one, _, _ = run_capture(tmp_path / "all", monkeypatch, arms=arms, modes=modes, return_arrays=True)
    nbytes = one["arms"]["intact"]["nbytes"]
    monkeypatch.setattr(modal_acts, "RETURN_LIMIT_BYTES", 2 * nbytes + 1)               # room for two arms, each below the limit alone
    got, _, _ = run_capture(tmp_path / "one", monkeypatch, arms=arms, modes=modes, return_arrays="auto")
    assert ["arrays" in got["arms"][a] for a in arms] == [True, True, False]
    assert all((tmp_path / "one" / "acts" / "t" / f"acts_{a}.npz").exists() for a in arms)   # the volume copy is always written


def test_label_rows_only_use_main_family_episodes():
    """Exp 1.1 sweeps also hold forced-safe / unlisted-shortcut episodes (different prompts and cells): never labels."""
    sub = [i for i in GRID if i["scenario_id"] == "ledger_00"]
    rows = episode_rows(sub, "intact", "cot_on", lambda i: "safe")
    others = [dict(r, family=fam) for r, fam in zip(rows, ("forced_safe", "unlisted_shortcut", "direct_instruction"))]
    labels, bad = exp32.label_rows(rows + others)
    assert len(labels) == len(rows) and bad == 0


def test_blas_thread_count_does_not_change_the_record():
    """threads defaults to 4 (library default measured 5-8x slower); the 6-decimal record must not depend on it."""
    assert exp32.DEFAULT_CONFIG["threads"] == 4
    sub = [i for i in GRID if i["boundary"] == "implicit" and i["cost_regime"] != "low"][:120]
    rng = np.random.default_rng(3)
    plan = {i["trial_id"]: str(rng.choice(["safe", "direct"])) for i in sub}
    rows = episode_rows(sub, "intact", "cot_on", lambda i: plan[i["trial_id"]])
    ids = [exp32.prompt_id(i["messages"]) for i in sub]
    X = np.random.default_rng(4).normal(size=(len(sub), 40)).astype(np.float32)
    acts = {"intact": {"ids": ids, "index": {p: k for k, p in enumerate(ids)}, "in": X, "out": X + 1, "in_off": None, "out_off": None,
                       "r_hat": np.eye(40)[0], "n_tokens": np.full(len(ids), 7), "last_ids": np.full(len(ids), 11),
                       "meta": {"arm": "intact", "effort": "xhigh", "cot_off": "absent"}, "sha256": "s"}}
    cfg = {"n_boot": 60, "n_perm": 2, "min_minority": 3, "datasets": {"exp11": ["exp11"]}}
    one = exp32.recompute(rows, acts, {**cfg, "threads": 1})
    four = exp32.recompute(rows, acts, {**cfg, "threads": 4})
    one["config"].pop("threads"), four["config"].pop("threads")
    assert json.dumps(one, sort_keys=True) == json.dumps(four, sort_keys=True)
