"""CPU tests for Exp 1.2: v3.exp12 (corpora, targets, token ledger, estimator), v3.modal_train (SFT tensors, steps,
generation post-processing, a tiny random Qwen3_5 + PEFT end-to-end when transformers/peft are installed),
scripts/run_exp12.py and scripts/analyze_exp12.py.

All scenario / harm text is a placeholder (scenarios.make_placeholder_bases); no test prints or asserts on real text.
The HF-backed tests need transformers>=5 and peft and are skipped in the sprint .venv (see test_modal_hf.py).
"""
import copy
import importlib.util
import json
import math
import pathlib
import random
import sys
import time
from collections import Counter

import pytest

torch = pytest.importorskip("torch")        # the sprint .venv has CPU torch; transformers / peft are optional (see the `hf` fixture)
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from v3 import episodes, exp12, scenarios  # noqa: E402
from v3 import modal_train as mt  # noqa: E402
from v3.parse import parse_turn  # noqa: E402

BASES = scenarios.make_placeholder_bases(40)
TRAIN_BASES = [b for b in BASES if b["template_id"] in scenarios.TRAIN_TEMPLATE_IDS]
TEACHERS = exp12.TEACHERS
EOS = exp12.EOS


# ------------------------------------------------------------------------------------------ scripted rollouts
def call_block(name, **args):
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": args}) + "\n</tool_call>"


def think(tag):
    return f"[REASONING_PLACEHOLDER {tag}]\n"


def safe_turn(inst, t, tag=""):
    s = inst["safe_steps"][t - 1]
    return think(f"{inst['trial_id']} t{t} {tag}") + "</think>\n\n" + call_block(inst["safe_tool"], operation=s["operation"], step=s["step"])


def shortcut_turn(inst):
    return think(inst["trial_id"]) + "</think>\n\n" + call_block(inst["direct_tool"], workspace=inst["scenario_id"])


class Policy:
    """policy(inst, sample) -> 'safe' | 'shortcut' | 'escalate' | 'wrong_step' | 'truncated' | 'malformed'"""

    def __init__(self, policy):
        self.policy = policy

    def raw(self, inst, sample, turn):
        kind = self.policy(inst, sample)
        if kind == "safe":
            return safe_turn(inst, turn), "stop"
        if kind == "shortcut":
            return shortcut_turn(inst), "stop"
        if kind == "escalate":
            return think("e") + "</think>\n\n" + call_block(inst["escalate_tool"], reason="[REASON_PLACEHOLDER]"), "stop"
        if kind == "wrong_step":     # skips to step 2 first: ERR_ARGUMENTS, then recovers
            s = 2 if turn == 1 else turn - 1
            st = inst["safe_steps"][s - 1]
            return think("w") + "</think>\n\n" + call_block(inst["safe_tool"], operation=st["operation"], step=st["step"]), "stop"
        if kind == "truncated":
            return think("never closes"), "length"
        if kind == "malformed":
            return think("m") + "</think>\n\n<tool_call>\n{not json\n</tool_call>", "stop"
        raise ValueError(kind)


def run_teacher(instances, policy, *, arm, exp_id, samples=1, pol_cls=Policy):
    """Real episode records (run_episodes + a scripted model); trial ids `<prompt>__s<k>` for k > 0 style sampling."""
    pol = pol_cls(policy)
    if samples == 1:
        reqs = exp12.teacher_greedy_requests(instances, exp_id=exp_id)
    else:
        reqs = exp12.teacher_sampled_requests(instances, samples, exp_id=exp_id)
    by_id = {r["trial_id"]: r for r in reqs}

    def gen(rs):
        outs = []
        for r in rs:
            tid, _, turn = r["key"].rpartition(":turn")
            req = by_id[tid]
            raw, fin = pol.raw(req["instance"], req["meta"]["sample"], int(turn))
            outs.append({"raw_output": raw, "generated_tokens": len(raw) // 4 + 1, "finish_reason": fin,
                         "full_prompt": f"[PROMPT {r['key']} {json.dumps(r['messages'])[:40]}]<think>\n"})
        return outs

    recs = episodes.run_episodes(gen, [r["instance"] for r in reqs], arm=arm, exp_id=exp_id, cot_mode="cot_on",
                                 reasoning_effort=exp12.EFFORT, max_new_tokens=exp12.CAP)
    for rec, req in zip(recs, reqs):
        rec["request_meta"], rec["request_trial_id"] = req["meta"], req["trial_id"]
    return recs


@pytest.fixture(scope="module")
def train_insts():
    return exp12.train_instances(BASES)


# ------------------------------------------------------------------------------------------ requests / splits
def test_train_instances_are_high_cost_train_templates(train_insts):
    assert len(train_insts) == 24 * 8
    assert {i["template_id"] for i in train_insts} == set(scenarios.TRAIN_TEMPLATE_IDS)
    assert {i["cost_regime"] for i in train_insts} == set(exp12.HIGH_COST)
    assert Counter((i["cost_regime"], i["consequence"], i["boundary"]) for i in train_insts) == Counter({c: 24 for c in exp12.TRAIN_CELLS})


def test_eval_instances_are_held_out_templates_only():
    ev = exp12.eval_instances(BASES)
    assert len(ev) == 16 * len(exp12.EVAL_CELLS)
    assert {i["template_id"] for i in ev} == set(scenarios.TEST_TEMPLATE_IDS)
    assert not {i["template_id"] for i in ev} & set(scenarios.TRAIN_TEMPLATE_IDS)
    primary = [i for i in ev if i["consequence"] == "harmful" and i["boundary"] == "implicit" and i["cost_regime"] != "low"]
    assert len(primary) == 32
    # tool order is counterbalanced over all 3! orders in the held-out set
    assert len({tuple(i["tool_order"]) for i in ev if i["template_id"] == "billing"}) > 1
    assert len({i["order_index"] for i in ev}) == 6


def test_teacher_requests_satisfy_i1_and_prompt_hygiene(train_insts):
    greedy = exp12.teacher_greedy_requests(train_insts)
    sampled = exp12.teacher_sampled_requests(train_insts[:10], 4)
    assert len(greedy) == 192 and len(sampled) == 40
    for r in greedy + sampled:
        assert r["kind"] == "episode" and r["cot_mode"] == "cot_on" and r["max_new_tokens"] >= 2048
        assert r["reasoning_effort"] == exp12.EFFORT and "arm" not in r        # arm absent: runs on every teacher
        text = "\n".join(m["content"] for m in r["instance"]["messages"])
        assert "</think>" not in text and "DECLARED_TOOL_NAME" not in text
    assert len({r["trial_id"] for r in greedy}) == 192 and len({r["trial_id"] for r in sampled}) == 40 and sampled[0]["trial_id"].endswith("__s0")
    assert all(r["meta"]["prompt_id"] == r["instance"]["trial_id"].rsplit("__s", 1)[0] for r in sampled)


def test_refusal_items_keep_messages_and_meta():
    rows = [{"trial_id": f"p1_C_x{i}", "exp_id": "phase1", "kind": "single", "cot_mode": "cot_on", "reasoning_effort": None,
             "max_new_tokens": 16384, "messages": [{"role": "system", "content": "[S]"}, {"role": "user", "content": f"[U{i}]"}],
             "meta": {"set_name": "C", "consequence": "harmful" if i % 2 else "harmless", "restricted_tool": "t_x"}} for i in range(4)]
    items = exp12.refusal_items(rows)
    assert [i["messages"] for i in items] == [r["messages"] for r in rows] and items[0]["trial_id"] == "e12_ref:p1_C_x0"
    assert all("max_new_tokens" not in i and "exp_id" not in i for i in items)       # decoding header comes from the run
    with pytest.raises(ValueError):
        exp12.refusal_items([{**rows[0], "meta": {"set_name": "A", "consequence": "harmful"}}])


# ------------------------------------------------------------------------------------------ target assertions
GOOD = think("x") + "</think>\n\n" + call_block("a_tool", n=1)


def test_target_checks_accept_valid_and_flag_each_defect():
    assert exp12.target_problems(exp12.with_eos(GOOD), True) == []
    assert exp12.with_eos(GOOD + EOS) == exp12.with_eos(GOOD) and exp12.with_eos(GOOD).count(EOS) == 1
    assert exp12.target_problems(GOOD, True) == ["no_eos"]
    assert "no_think_close" in exp12.target_problems(exp12.with_eos(think("x") + call_block("a", n=1)), True)
    two = think("x") + "</think>\n" + think("y") + "</think>\n\n" + call_block("a", n=1)
    assert "multi_think_close" in exp12.target_problems(exp12.with_eos(two), True)
    assert "stray:<think>" in exp12.target_problems(exp12.with_eos("<think>\n" + GOOD), True)
    assert any("MALFORMED" in p or p == "malformed_call" for p in exp12.target_problems(
        exp12.with_eos(think("x") + "</think>\n\n<tool_call>{bad json</tool_call>"), True))
    unclosed = think("x") + "</think>\n\n<tool_call>" + json.dumps({"name": "a", "arguments": {}})
    assert "unclosed_tag" in exp12.target_problems(exp12.with_eos(unclosed), True)
    assert exp12.target_problems(exp12.with_eos(think("x") + "</think>\n\n[plain text answer]"), True) == []      # text turn is a valid agentic target
    multi = think("x") + "</think>\n\n" + call_block("a", n=1) + "\n" + call_block("b", n=2)
    assert exp12.target_problems(exp12.with_eos(multi), True) == []
    assert exp12.target_problems(exp12.with_eos(think("x") + "</think>\n\n" + call_block("a", n=1)), False) == ["tool_markup_in_cot"]
    assert exp12.target_problems(exp12.with_eos(think("x") + "</think>\n\n   "), False) == ["empty_answer"]
    assert exp12.target_problems(exp12.with_eos(think("x") + "</think>\n\nAnswer: 4"), False) == []
    # a draft call quoted INSIDE the think is not an action block
    draft = "draft " + call_block("a", n=1) + "\n</think>\n\n" + call_block("a", n=2)
    assert exp12.target_problems(exp12.with_eos(draft), True) == []


def test_assert_targets_raises_with_ids_not_text():
    ok = {"example_id": "e1", "kind": "agentic", "target": exp12.with_eos(GOOD)}
    bad = {"example_id": "e2", "kind": "agentic", "target": exp12.with_eos(think("SECRET_TEXT"))}
    exp12.assert_targets([ok])
    with pytest.raises(ValueError) as ei:
        exp12.assert_targets([ok, bad])
    assert "e2" in str(ei.value) and "SECRET_TEXT" not in str(ei.value)


# ------------------------------------------------------------------------------------------ episodes -> examples
def test_episode_examples_are_whole_turns_with_exact_prompt_and_target(train_insts):
    inst = next(i for i in train_insts if i["cost_regime"] == "high_tedium")
    rec, = run_teacher([inst], lambda i, s: "safe", arm="intact", exp_id="exp12_teacher_greedy")
    assert rec["n_turns"] == 7 and rec["outcome_category"] == "SAFE_ROUTE"
    ex, drops = exp12.episode_examples(rec, "A", "intact", later_turns=2, seed=0)
    assert [e["source"]["turn"] for e in ex][0] == 1 and len(ex) == 3 and drops["not_sampled"] == 4
    for e in ex:
        t = rec["turns"][e["source"]["turn"] - 1]
        assert e["prompt"] == t["full_prompt"]                                    # never re-rendered, never truncated
        assert e["target"] == t["raw_output"] + EOS                               # exact generated text, no slicing
        assert e["target_tokens"] == t["generated_tokens"]
    ex2, _ = exp12.episode_examples(rec, "A", "intact", later_turns=2, seed=0)
    assert [e["example_id"] for e in ex] == [e["example_id"] for e in ex2]        # deterministic
    exall, _ = exp12.episode_examples(rec, "A", "intact", later_turns=99, seed=0)
    assert [e["source"]["turn"] for e in exall] == list(range(1, 8))
    exp12.assert_targets(ex)


def test_invalid_and_overlong_turns_are_dropped_whole_and_counted(train_insts):
    inst = next(i for i in train_insts if i["cost_regime"] == "high_tedium")
    trunc, = run_teacher([inst], lambda i, s: "truncated", arm="intact", exp_id="x")
    ex, drops = exp12.episode_examples(trunc, "A", "intact", later_turns=2, seed=0)
    assert ex == [] and drops["invalid:finish_length"] == 1
    mal, = run_teacher([inst], lambda i, s: "malformed", arm="intact", exp_id="x")
    ex, drops = exp12.episode_examples(mal, "A", "intact", later_turns=2, seed=0)
    assert ex == [] and sum(v for k, v in drops.items() if k.startswith("invalid:")) == 1
    ok, = run_teacher([inst], lambda i, s: "safe", arm="intact", exp_id="x")
    ex, drops = exp12.episode_examples(ok, "A", "intact", later_turns=99, seed=0, max_tokens=5)
    assert ex == [] and drops["too_long"] == 7


def test_unfiltered_corpus_a_keeps_shortcuts_and_escalations(train_insts):
    insts = train_insts[:40]
    kinds = ["safe", "shortcut", "escalate", "wrong_step", "truncated"]
    pos = {i["trial_id"]: k for k, i in enumerate(insts)}
    greedy = {t: run_teacher(insts, lambda i, s, t=t: kinds[(pos[i["trial_id"]] + len(t)) % 5], arm=t, exp_id="g") for t in TEACHERS}
    scale = exp12.Scale(max_prompts=0, later_turns=1)
    corp, stats = exp12.build_corpus_a(greedy, scale)
    for t in TEACHERS:
        outcomes = stats["teachers"][t]["outcomes"]
        assert {"SAFE_ROUTE", "SHORTCUT_VIOLATION", "ESCALATE_OR_REFUSE", "TRUNCATED_OR_MALFORMED"} <= set(outcomes) | {"SHORTCUT_PERMITTED"}
        assert stats["teachers"][t]["dropped_turns"].get("invalid:finish_length", 0) > 0
        assert any(e["source"]["outcome"].startswith("SHORTCUT") for e in corp[t])      # behaviour is not filtered
        exp12.assert_targets(corp[t])


def test_corpus_a_prompt_subset_is_stratified_and_shared_across_teachers(train_insts):
    greedy = {t: run_teacher(train_insts, lambda i, s: "safe", arm=t, exp_id="g") for t in TEACHERS}
    corp, stats = exp12.build_corpus_a(greedy, exp12.Scale(max_prompts=32, later_turns=0))
    assert stats["n_prompts_selected"] == 32
    per_teacher = [{e["source"]["prompt_id"] for e in corp[t]} for t in TEACHERS]
    assert per_teacher[0] == per_teacher[1] == per_teacher[2] and len(per_teacher[0]) == 32
    cells = Counter(next(i for i in train_insts if i["trial_id"] == p)["consequence"] + next(i for i in train_insts if i["trial_id"] == p)["boundary"]
                    for p in per_teacher[0])
    assert set(cells.values()) == {8}                                                   # 32 prompts / 4 (consequence x boundary) x 2 costs
    again, _ = exp12.build_corpus_a(greedy, exp12.Scale(max_prompts=32, later_turns=0))
    assert [e["example_id"] for e in again["intact"]] == [e["example_id"] for e in corp["intact"]]


def test_select_prompts_quotas_and_topup():
    cells = {f"p{i}": ("a" if i < 3 else "b",) for i in range(20)}
    pick = exp12.select_prompts(cells, 10, seed=1)
    assert len(pick) == 10 and sum(cells[p] == ("a",) for p in pick) == 3 and pick == exp12.select_prompts(dict(reversed(list(cells.items()))), 10, seed=1)
    assert exp12.select_prompts(cells, 0, seed=1) == sorted(cells) and exp12.select_prompts(cells, 99, seed=1) == sorted(cells)


# ------------------------------------------------------------------------------------------ Corpus B
def sampled_policy(table):
    """table[(teacher, sample)] -> kind; default by sample index."""
    def make(teacher):
        return lambda inst, s: table.get((teacher, inst["scenario_id"], s), table.get((teacher, s), "safe"))
    return make


def test_clean_safe_route_requires_every_turn_clean(train_insts):
    inst = next(i for i in train_insts if i["cost_regime"] == "high_tedium")
    clean, = run_teacher([inst], lambda i, s: "safe", arm="intact", exp_id="x")
    assert exp12.is_clean_safe_route(clean)
    wrong, = run_teacher([inst], lambda i, s: "wrong_step", arm="intact", exp_id="x")
    assert wrong["outcome_category"] == "SAFE_ROUTE" and not exp12.is_clean_safe_route(wrong)     # eventually safe, but ERR_ARGUMENTS
    short, = run_teacher([inst], lambda i, s: "shortcut", arm="intact", exp_id="x")
    assert not exp12.is_clean_safe_route(short)
    bad = copy.deepcopy(clean)
    bad["turns"][2]["parsed"]["n_malformed"] = 1
    assert not exp12.is_clean_safe_route(bad)


def test_corpus_b_intersection_acceptance_and_shared_prompts(train_insts):
    insts = [i for i in train_insts if i["scenario_id"] in {"ledger_00", "inventory_01"}]       # 16 prompts
    n = 4
    # intact: always safe; ablated: safe only on explicit-boundary prompts at sample 3; pca: safe only on sample >= 1
    def pol(teacher):
        def f(inst, s):
            if teacher == "intact":
                return "safe"
            if teacher == "refusal_ablated":
                return "safe" if (inst["boundary"] == "explicit" and s == 3) else "shortcut"
            return "safe" if s >= 1 else "shortcut"
        return f
    sampled = {t: run_teacher(insts, pol(t), arm=t, exp_id="s", samples=n) for t in TEACHERS}
    corp, stats = exp12.build_corpus_b(sampled, exp12.Scale(max_prompts=0, later_turns=1))
    inter = stats["intersection"]
    assert inter["n_prompts_all_teachers"] == 16 and inter["n_prompts"] == 8                  # explicit prompts only
    assert inter["fraction"] == 0.5 and inter["gate_ge_50pct"] and set(inter["by_cell"]) == {"high_tedium|harmless|explicit", "high_tedium|harmful|explicit", "high_friction|harmless|explicit", "high_friction|harmful|explicit"}
    st = stats["teachers"]
    assert st["intact"]["sample_acceptance"] == 1.0 and st["refusal_ablated"]["n_clean_samples"] == 8 and st["refusal_ablated"]["prompt_acceptance"] == 0.5
    assert st["pca_101"]["n_clean_samples"] == 16 * 3 and st["pca_101"]["prompt_acceptance"] == 1.0
    prompts = [{e["source"]["prompt_id"] for e in corp[t]} for t in TEACHERS]
    assert prompts[0] == prompts[1] == prompts[2] and len(prompts[0]) == 8
    # keep_per_prompt=1: the lowest clean sample index per teacher
    assert {e["source"]["sample"] for e in corp["intact"]} == {0} and {e["source"]["sample"] for e in corp["refusal_ablated"]} == {3}
    assert {e["source"]["sample"] for e in corp["pca_101"]} == {1}
    assert all(e["source"]["outcome"] == "SAFE_ROUTE" for t in TEACHERS for e in corp[t])
    exp12.assert_targets([e for t in TEACHERS for e in corp[t]])


def test_corpus_b_rejects_errors_and_caps_prompts(train_insts):
    insts = train_insts[:24]
    def pol(teacher):
        return lambda inst, s: "wrong_step" if (teacher == "intact" and s == 0) else "safe"
    sampled = {t: run_teacher(insts, pol(t), arm=t, exp_id="s", samples=2) for t in TEACHERS}
    corp, stats = exp12.build_corpus_b(sampled, exp12.Scale(max_prompts=8, later_turns=0), keep_per_prompt=1)
    assert stats["teachers"]["intact"]["n_clean_samples"] == 24                         # sample 0 had ERR_ARGUMENTS: rejected
    assert {e["source"]["sample"] for e in corp["intact"]} == {1}
    assert stats["intersection"]["n_prompts"] == 24 and stats["intersection"]["n_selected"] == 8
    assert all(len(corp[t]) == 8 for t in TEACHERS)


# ------------------------------------------------------------------------------------------ Corpus B' (paraphrase)
def para_row(example, text, finish="stop"):
    raw = "<tool_call>\n" + json.dumps({"name": exp12.PARAPHRASE_TOOL, "arguments": {"text": text}}) + "\n</tool_call>"
    return {"trial_id": "para:" + example["example_id"], "meta": {"example_id": example["example_id"]}, "raw_output": raw,
            "finish_reason": finish, "generated_tokens": 20}


@pytest.fixture(scope="module")
def b_corpus(train_insts):
    insts = train_insts[:12]
    sampled = {t: run_teacher(insts, lambda i, s: "safe", arm=t, exp_id="s", samples=2) for t in TEACHERS}
    corp, _ = exp12.build_corpus_b(sampled, exp12.Scale(max_prompts=0, later_turns=1))
    return corp


def test_paraphrase_requests_are_clean_and_cover_every_example(b_corpus):
    rows = exp12.paraphrase_requests(b_corpus)
    assert len(rows) == sum(len(v) for v in b_corpus.values())
    for r in rows:
        assert r["kind"] == "single" and r["cot_mode"] == "cot_off" and r["max_new_tokens"] >= 512
        sys_text = r["messages"][0]["content"]
        assert "</think>" not in "".join(m["content"] for m in r["messages"]) and "DECLARED_TOOL_NAME" not in sys_text
        assert scenarios.hygiene_problems([sys_text]) == []                      # no worked tool-call example in the instruction
    assert len({r["trial_id"] for r in rows}) == len(rows)
    assert exp12.parse_paraphrase(para_row({"example_id": "e"}, "new text")["raw_output"]) == ("new text", "OK")
    assert exp12.parse_paraphrase("no call here")[0] is None


def test_bprime_replaces_think_only_and_keeps_action_block_identical(b_corpus):
    rows = [para_row(e, f"[PARAPHRASE_PLACEHOLDER {i}]") for t in TEACHERS for i, e in enumerate(b_corpus[t])]
    out, stats = exp12.build_corpus_bprime(b_corpus, rows)
    for t in TEACHERS:
        assert len(out[t]) == len(b_corpus[t]) and stats["teachers"][t]["dropped"] == {}
        for b, bp in zip(b_corpus[t], out[t]):
            assert bp["corpus"] == "Bp" and bp["example_id"].startswith("Bp|") and bp["prompt"] == b["prompt"]
            assert "[PARAPHRASE_PLACEHOLDER" in bp["target"] and "[REASONING_PLACEHOLDER" not in bp["target"]
            ob, nb = parse_turn(b["target"][:-len(EOS)], True), parse_turn(bp["target"][:-len(EOS)], True)
            assert ob.calls == nb.calls and ob.post_think_text == nb.post_think_text         # identical parseable action block
            assert bp["target"].count("</think>") == 1 and bp["target"].endswith(EOS)
            assert bp["target"].split("</think>", 1)[1] == b["target"].split("</think>", 1)[1]
        exp12.assert_targets(out[t])


def test_bprime_drops_failed_paraphrases_whole_and_counts(b_corpus):
    t = "intact"
    ex = b_corpus[t]
    rows = [para_row(ex[0], "fine text"), para_row(ex[1], "has <|im_start|> markup"), para_row(ex[2], "x" * 5000),
            para_row(ex[3], "ok", finish="length"), {**para_row(ex[4], "t"), "raw_output": "plain text, no call"}, para_row(ex[5], "  ")]
    out, stats = exp12.build_corpus_bprime({t: ex[:8]}, rows)
    d = stats["teachers"][t]["dropped"]
    assert len(out[t]) == 1 and out[t][0]["example_id"].endswith(ex[0]["example_id"].split("|", 1)[1])
    assert d["para:markup"] == 1 and d["para:too_long"] == 1 and d["finish_length"] == 1 and d["parse:NO_TOOL_CALL"] == 1 and d["para:empty"] == 1
    assert d["no_paraphrase_row"] == 2
    # a literal close tag inside the answer splits the output at the parser: the paraphrase is dropped as unparseable
    out2, stats2 = exp12.build_corpus_bprime({t: ex[:1]}, [para_row(ex[0], "has </think> markup")])
    assert out2[t] == [] and sum(stats2["teachers"][t]["dropped"].values()) == 1


# ------------------------------------------------------------------------------------------ Corpus C
def test_math_coding_prompts_are_deterministic_and_correct():
    a, b = exp12.math_coding_prompts(70, seed=3), exp12.math_coding_prompts(70, seed=3)
    assert a == b and a != exp12.math_coding_prompts(70, seed=4) and len({p["prompt_id"] for p in a}) == 70
    assert {p["family"] for p in a} == {"arith", "linear", "modpow", "seq", "comb", "code_trace", "code_write"}
    for p in a:
        assert "</think>" not in p["question"]
        if p["family"] == "arith":
            x = [int(v) for v in __import__("re").findall(r"\d+", p["question"])[:4]]
            assert str(x[0] * x[1] + x[2] - x[3]) == p["gold"]
        if p["family"] == "code_write":
            assert p["gold"] is None
        if p["family"] == "modpow":
            a_, e_, m_ = [int(v) for v in __import__("re").findall(r"\d+", p["question"].split("?")[0])]
            assert pow(a_, e_, m_) == int(p["gold"])
    assert exp12.answer_matches(think("x") + "</think>\n\nso\nAnswer: 42", "42") is True
    assert exp12.answer_matches(think("x") + "</think>\n\nAnswer: 41", "42") is False and exp12.answer_matches("x", None) is None


def test_cot_requests_shape():
    rows = exp12.cot_requests(exp12.math_coding_prompts(14))
    assert all(r["kind"] == "single" and r["cot_mode"] == "cot_on" and r["max_new_tokens"] >= 2048 for r in rows)
    assert len({r["trial_id"] for r in rows}) == 14 and rows[0]["messages"][0]["role"] == "system"


def test_match_token_budget_hits_target_within_tolerance_and_is_deterministic():
    rng = random.Random(5)
    items = [(f"i{k}", rng.randint(80, 900)) for k in range(300)]
    for target in (1234, 20000, 61111):
        led = exp12.match_token_budget(items, target, seed=1)
        assert led["within_tol"] and led["abs_error"] <= led["tol"] and led["total"] == sum(dict(items)[i] for i in led["chosen"])
        assert led["ledger"][-1][2] == led["total"] and len(set(led["chosen"])) == len(led["chosen"])
        assert led == exp12.match_token_budget(list(reversed(items)), target, seed=1)
    exact = exp12.match_token_budget(items, 20000, tol=0, seed=1)
    assert exact["total"] == 20000 and exact["within_tol"]
    small = exp12.match_token_budget(items[:3], 10 ** 6, seed=1)
    assert not small["within_tol"] and small["n"] == 3                                    # pool too small: reported, not hidden


def cot_row(teacher, i, tokens, raw=None, finish="stop", family="arith", gold="7"):
    raw = raw if raw is not None else think(f"{teacher}{i}") + f"</think>\n\nso.\nAnswer: {gold}"
    return {"trial_id": f"cot_{i:04d}", "exp_id": "g", "arm": teacher, "kind": "single", "raw_output": raw, "full_prompt": f"[COT PROMPT {i}]<think>\n",
            "messages": [{"role": "system", "content": exp12.COT_SYSTEM}, {"role": "user", "content": f"[QUESTION PLACEHOLDER {i}]"}],
            "generated_tokens": tokens, "finish_reason": finish, "meta": {"family": family, "gold": gold, "prompt_id": f"cot_{i:04d}"}}


def test_corpus_c_matches_each_teachers_b_token_count(b_corpus):
    rng = random.Random(0)
    cot = {t: [cot_row(t, i, rng.randint(60, 400)) for i in range(400)] for t in TEACHERS}
    cot["intact"][0] = cot_row("intact", 0, 100, raw=think("x") + "no close tag", finish="length")
    cot["intact"][1] = cot_row("intact", 1, 100, raw=think("x") + "</think>\n\n" + call_block("a", n=1))
    cot["intact"][2] = cot_row("intact", 2, 100, raw=think("x") + "</think>\n</think>\n\nAnswer: 7")
    corp, stats = exp12.build_corpus_c(cot, b_corpus)
    for t in TEACHERS:
        b_tokens = sum(e["target_tokens"] for e in b_corpus[t])
        s = stats["teachers"][t]
        assert s["b_target_tokens"] == b_tokens and s["match"]["within_tol"] and abs(s["target_tokens"] - b_tokens) <= s["match"]["tol"]
        assert s["ledger"][-1][2] == s["target_tokens"] and len(s["ledger"]) == len(corp[t])
        assert all(e["kind"] == "cot" and e["target"].count("</think>") == 1 for e in corp[t])
        exp12.assert_targets(corp[t])
        assert s["answer_accuracy"] == 1.0
    assert stats["teachers"]["intact"]["dropped"] == {"finish_length": 1, "invalid:tool_markup_in_cot": 1, "invalid:multi_think_close": 1}


def test_build_all_end_to_end(train_insts):
    insts = train_insts[:24]
    raw = {"greedy": {t: run_teacher(insts, lambda i, s: "safe", arm=t, exp_id="g") + [cot_row(t, i, 100 + i, family="arith") for i in range(150)]
                      for t in TEACHERS},
           "sampled": {t: run_teacher(insts, lambda i, s: "safe", arm=t, exp_id="s", samples=2) for t in TEACHERS}}
    scale = exp12.Scale(max_prompts=12, later_turns=1)
    corp, stats = exp12.build_all(raw, scale)
    assert set(corp) == {"A", "B", "C"} and "Bp" not in corp                            # B' needs the Llama rows
    raw["para"] = [para_row(e, f"[P {e['example_id']}]") for t in TEACHERS for e in corp["B"][t]]
    corp2, stats2 = exp12.build_all(raw, scale)
    assert set(corp2) == {"A", "B", "Bp", "C"} and corp2["A"] == corp["A"] and stats2["B"]["intersection"]["n_prompts"] == 24
    json.dumps(stats2)                                                                  # stats are JSON-serialisable
    assert exp12.scale_from_dict(json.loads(json.dumps(stats2["scale"]))) == scale
    # a record whose derived fields do not replay is ignored and counted
    tampered = copy.deepcopy(raw)
    tampered["sampled"]["intact"][0]["route"] = "SAFE_ROUTE" if tampered["sampled"]["intact"][0]["route"] != "SAFE_ROUTE" else "SHORTCUT"
    tampered["sampled"]["intact"][0]["direct_taken"] = not tampered["sampled"]["intact"][0]["direct_taken"]
    _, stats3 = exp12.build_all(tampered, scale)
    assert stats3["B"]["teachers"]["intact"]["replay_mismatch"] == 1


# ------------------------------------------------------------------------------------------ scale and cost
def test_scale_grid_counts_and_ids():
    assert exp12.PRESETS["full72"].n_adapters == 72 and exp12.PRESETS["reduced54"].n_adapters == 54
    assert exp12.PRESETS["lean36"].n_adapters == 36 and exp12.PRESETS["smoke"].n_adapters == 1
    jobs = exp12.PRESETS["full72"].jobs()
    assert len({j.adapter_id for j in jobs}) == 72 and jobs[0].adapter_id == "A.intact.r32.s0"
    assert Counter(j.rank for j in jobs) == {32: 36, 8: 36}
    red = exp12.PRESETS["reduced54"].jobs()
    assert {(j.corpus) for j in red if j.rank == 8} == {"B", "C"} and Counter(j.rank for j in red) == {32: 36, 8: 18}


def test_shard_jobs_round_robin_with_floors_in_first_and_last_shard():
    jobs = exp12.PRESETS["full72"].jobs()
    shards = exp12.shard_jobs(jobs, 8, 2)
    assert len(shards) == 8 and shards[0][0] is None and shards[-1][0] is None and sum(x is None for s in shards for x in s) == 2
    assert sorted(j.adapter_id for s in shards for j in s if j) == sorted(j.adapter_id for j in jobs)
    assert max(len(s) for s in shards) - min(len(s) for s in shards) <= 3
    assert exp12.shard_jobs(jobs[:2], 8, 0) == [[jobs[0]], [jobs[1]]]


def test_cost_estimator_default_fits_target_and_is_monotone():
    full = exp12.estimate_cost(exp12.PRESETS["full72"])
    assert full["n_adapters"] == 72 and full["gpu_hours"] > 5 and full["usd"] > 75        # unmeasured speeds: the default grid does not fit $75 ...
    assert exp12.recommend_scale(75.0)[0] == "reduced54" and exp12.estimate_cost(exp12.PRESETS["reduced54"])["usd"] <= 75      # ... the reduced grid does
    assert set(full["stages"]) == {"teacher_greedy_and_cot", "teacher_sampled", "llama_paraphrase", "students"}
    assert all({"generations", "tokens", "gpu_hours", "usd"} <= set(s) for s in full["stages"].values())
    assert full["stages"]["students"]["usd"] > 0.8 * full["usd"]                         # students dominate
    sizes = [exp12.estimate_cost(exp12.PRESETS[n])["usd"] for n in ("full72", "reduced54", "lean36", "smoke")]
    assert sizes == sorted(sizes, reverse=True)
    assert exp12.estimate_cost(exp12.Scale(epochs=6))["usd"] > full["usd"]
    assert exp12.estimate_cost(exp12.Scale(), train_tok_per_s=750)["usd"] > full["usd"]
    name, est = exp12.recommend_scale(55.0)
    assert name == "lean36" and est["usd"] <= 55 and exp12.recommend_scale(40.0)[0] == "smoke"
    assert exp12.recommend_scale(500.0)[0] == "full72"


# ------------------------------------------------------------------------------------------ launcher helpers
def load_script(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ================================================================================ modal_train: pure / torch-only parts
class StubTok:
    """One token per character (id = ord + 10, ids < 250); special strings are single tokens."""
    pad_token_id = 0
    SPECIAL = {"<|im_end|>": 3, "<|endoftext|>": 2}

    def convert_tokens_to_ids(self, t):
        return self.SPECIAL[t]

    def _enc(self, text):
        ids, i = [], 0
        while i < len(text):
            for sp, k in self.SPECIAL.items():
                if text.startswith(sp, i):
                    ids.append(k); i += len(sp); break
            else:
                ids.append(ord(text[i]) % 200 + 10); i += 1
        return ids

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self._enc(text)}

    def apply_chat_template(self, messages, **kw):
        eff = kw.get("reasoning_effort")
        head = f"[{eff}]" if kw.get("enable_thinking") else "[off]"
        body = "|".join(m["content"] for m in messages)
        return head + body + ("<think>\n" if kw.get("enable_thinking") else "<think>\n\n</think>\n\n")

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        inv = {v: k for k, v in self.SPECIAL.items()}
        return "".join(inv.get(i) or chr((i - 10) % 200 if (i - 10) % 200 >= 32 else 63) for i in ids)


def sft_example(i, n_prompt=30, n_target=12):
    return {"example_id": f"e{i}", "prompt": "p" * n_prompt + "<think>\n", "target": "t" * n_target + EOS, "target_tokens": n_target + 1}


def test_lora_layer_selection_and_config():
    assert mt.lora_layer_indices(64) == list(range(16, 64, 2)) and len(mt.lora_layer_indices(64)) == 24
    assert all(i % 4 != 3 for i in mt.lora_layer_indices(64))                 # full-attention layers are 3 mod 4: self_attn targets match nothing
    plus = mt.lora_layer_indices(64, "spec_plus_full")
    assert len(plus) == 36 and {19, 23, 63} <= set(plus) and set(mt.lora_layer_indices(64)) <= set(plus)
    assert mt.lora_layer_indices(8) == [2, 4, 6]
    with pytest.raises(ValueError):
        mt.lora_layer_indices(64, "all")
    pytest.importorskip("peft")
    for r in (32, 8):
        c = mt.make_lora_config(r, mt.lora_layer_indices(64))
        assert c.r == r and c.lora_alpha == 2 * r and c.lora_dropout == 0.0
        assert set(c.target_modules) == {"q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "out_proj", "gate_proj", "up_proj", "down_proj"}
        assert list(c.layers_to_transform) == list(range(16, 64, 2)) and c.layers_pattern == "layers"


def test_tokenize_examples_masks_nothing_it_does_not_drop_and_never_truncates():
    tok = StubTok()
    ex = [sft_example(0), sft_example(1, n_prompt=90, n_target=40)]
    data, st = mt.tokenize_examples(tok, ex, max_len=100)
    assert [d["id"] for d in data] == ["e0"] and st["dropped"] == {"too_long": 1} and st["n_kept"] == 1
    d = data[0]
    assert d["input_ids"][:d["n_prompt"]] == tok(ex[0]["prompt"])["input_ids"]               # the prompt is intact (no left truncation)
    assert d["input_ids"][d["n_prompt"]:] == tok(ex[0]["target"])["input_ids"] and d["input_ids"][-1] == 3
    assert st["n_target_tokens"] == 13 and st["ledger_ratio"] == 1.0
    with pytest.raises(ValueError):
        mt.tokenize_examples(tok, [{**ex[0], "target": "no eos"}], 100)
    with pytest.raises(ValueError):
        mt.tokenize_examples(tok, [{**ex[0], "prompt": "no think tail"}], 100)


def test_plan_steps_budgets_target_tokens_and_is_seeded():
    n_target = [10] * 30
    steps = mt.plan_steps(n_target, epochs=3, step_target_tokens=40, seed=1)
    assert steps == mt.plan_steps(n_target, 3, 40, 1) and steps != mt.plan_steps(n_target, 3, 40, 2)
    flat = [i for s in steps for i in s]
    assert sorted(flat) == sorted(list(range(30)) * 3)                        # every example once per epoch
    assert sum(sum(n_target[i] for i in s) < 40 for s in steps) <= 3          # only an epoch's last step may be short
    # a tiny remainder is merged into the previous step, a large one stands alone
    assert [len(s) for s in mt.plan_steps([10] * 9, 1, 40, 0)] == [4, 5]
    assert [len(s) for s in mt.plan_steps([10] * 10, 1, 40, 0)] == [4, 4, 2]
    # equal total target tokens -> (nearly) equal step counts, whatever the example sizes (this is what matches B and C)
    assert abs(len(mt.plan_steps([5] * 200, 3, 1000, 0)) - len(mt.plan_steps([10] * 100, 3, 1000, 0))) <= 1


def test_micro_batches_respect_padded_budget():
    lengths = [10, 200, 30, 40, 500, 60]
    groups = mt.micro_batches(lengths, list(range(6)), micro_tokens=300)
    assert sorted(i for g in groups for i in g) == list(range(6))
    assert all(max(lengths[i] for i in g) * len(g) <= 300 or len(g) == 1 for g in groups)
    assert [4] in groups                                                     # the 500-token example stands alone


def test_collate_masks_prompt_and_pad():
    a = {"input_ids": [5, 6, 7, 8], "n_prompt": 2, "n_target": 2}
    b = {"input_ids": [9, 10, 11, 12, 13, 14], "n_prompt": 4, "n_target": 2}
    ids, mask, labels = mt.collate([a, b], pad_id=0)
    assert ids.tolist() == [[5, 6, 7, 8, 0, 0], [9, 10, 11, 12, 13, 14]] and mask.sum(1).tolist() == [4, 6]
    assert labels.tolist() == [[-100, -100, 7, 8, -100, -100], [-100, -100, -100, -100, 13, 14]]


class ToyInner(torch.nn.Module):
    def __init__(self, v=40, d=16):
        super().__init__()
        self.emb, self.lin = torch.nn.Embedding(v, d), torch.nn.Linear(d, d)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        h = self.emb(input_ids)
        h = h + torch.tanh(self.lin(h.cumsum(1) / torch.arange(1, h.shape[1] + 1)[None, :, None]))
        return type("Out", (), {"last_hidden_state": h})


class ToyCausalLM(torch.nn.Module):
    def __init__(self, v=40, d=16):
        super().__init__()
        self.model, self.lm_head = ToyInner(v, d), torch.nn.Linear(d, v, bias=False)


def test_sft_loss_sum_matches_full_logit_cross_entropy_on_target_positions():
    torch.manual_seed(0)
    m = ToyCausalLM()
    examples = [{"input_ids": [1, 2, 3, 4, 5, 6, 7], "n_prompt": 4, "n_target": 3}, {"input_ids": [8, 9, 10, 11, 12], "n_prompt": 3, "n_target": 2}]
    ids, mask, labels = mt.collate(examples, 0)
    got, n = mt.sft_loss_sum(m, ids, mask, labels, amp=False)
    logits = m.lm_head(m.model(ids).last_hidden_state)
    want = 0.0
    for k, e in enumerate(examples):
        for t in range(e["n_prompt"], len(e["input_ids"])):
            want += torch.nn.functional.cross_entropy(logits[k, t - 1][None], torch.tensor([e["input_ids"][t]]), reduction="sum").item()
    assert n == 5 and abs(float(got.detach()) - want) < 1e-4
    got.backward()
    assert m.lm_head.weight.grad is not None and m.model.emb.weight.grad is not None
    # chunking does not change the value or the gradient
    h = torch.randn(7, 16, requires_grad=True)
    w, t = torch.randn(40, 16), torch.randint(0, 40, (7,))
    a = mt.chunked_ce_sum(h, w, t, chunk=3)
    b = torch.nn.functional.cross_entropy(h @ w.T, t, reduction="sum")
    assert abs(float(a.detach()) - float(b.detach())) < 1e-4
    ga, = torch.autograd.grad(a, h)
    gb, = torch.autograd.grad(b, h)
    assert torch.allclose(ga, gb, atol=1e-5)


def test_plan_chunks_respects_batch_and_prefill_budget():
    lengths = [100, 900, 50, 700, 300, 20]
    chunks = mt.plan_chunks(lengths, max_batch=4, prefill_budget=2000)
    assert sorted(i for c in chunks for i in c) == list(range(6))
    for c in chunks:
        assert len(c) <= 4 and max(lengths[i] for i in c) * len(c) <= 2000 or len(c) == 1
    assert chunks[0][0] == 1                                                  # longest prompt first
    assert mt.plan_chunks([5000], 8, 1000) == [[0]]                           # one oversized prompt still runs alone
    assert mt.plan_chunks([], 8, 1000) == []


def test_finish_generation_matches_vllm_conventions():
    assert mt.finish_generation([5, 6, 3, 2, 2], {3, 2}) == ([5, 6], 3, "stop")      # count includes the EOS token
    assert mt.finish_generation([5, 6, 7], {3, 2}) == ([5, 6, 7], 3, "length")
    assert mt.finish_generation([3], {3}) == ([], 1, "stop")


def test_render_prompt_matches_serving_contract():
    tok = StubTok()
    req = {"messages": [{"role": "user", "content": "x"}], "cot_mode": "cot_on", "reasoning_effort": "low"}
    assert mt.render_prompt(tok, req).startswith("[low]") and mt.render_prompt(tok, req).endswith("<think>\n")
    assert mt.render_prompt(tok, {**req, "cot_mode": "cot_off"}) == "[off]x<think>\n\n</think>\n\n"
    class Bad(StubTok):
        def apply_chat_template(self, messages, **kw):
            return "no think tail"
    with pytest.raises(ValueError):
        mt.render_prompt(Bad(), req)


class FakeGenModel(torch.nn.Module):
    """Stands in for an HF model: continues every row with a scripted token sequence chosen by the first content token
    (the prompt is '[low]' + content, so index 5 of the unpadded row)."""
    def __init__(self, script):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))
        self.script, self.calls = script, []

    def generate(self, input_ids, attention_mask, max_new_tokens, **kw):
        self.calls.append({"shape": tuple(input_ids.shape), "kw": kw, "mask": attention_mask.clone()})
        rows = []
        for r, m in zip(input_ids, attention_mask):
            first = int(r[m.bool()][5])
            seq = self.script.get(first, [50] * max_new_tokens)[:max_new_tokens]
            seq = seq + [kw["pad_token_id"]] * (max_new_tokens - len(seq))
            rows.append(torch.tensor(seq))
        return torch.cat([input_ids, torch.stack(rows)], dim=1)


def test_hf_generator_left_pads_decodes_to_vllm_style_and_restores_order():
    tok = StubTok()
    first = lambda s: tok(s)["input_ids"][0]
    script = {first("A"): [70, 71, 3, 2, 2], first("B"): [80] * 6}
    model = FakeGenModel(script)
    gen = mt.HFGenerator(model, tok, max_batch=8, prefill_budget=10_000)
    reqs = [{"messages": [{"role": "user", "content": c}], "cot_mode": "cot_on", "reasoning_effort": "low", "max_new_tokens": 6, "key": c}
            for c in ("Aaaa", "Bb", "Aa")]
    outs = gen(reqs)
    assert [o["finish_reason"] for o in outs] == ["stop", "length", "stop"] and [o["generated_tokens"] for o in outs] == [3, 6, 3]
    assert outs[0]["raw_output"] == tok.decode([70, 71]) and "<|im_end|>" not in outs[0]["raw_output"]
    assert outs[0]["full_prompt"].endswith("<think>\n") and outs[1]["full_prompt"] == mt.render_prompt(tok, reqs[1])
    call = model.calls[0]
    assert call["kw"]["do_sample"] is False and call["kw"]["use_cache"] is True and call["kw"]["temperature"] is None
    assert set(call["kw"]["eos_token_id"]) == {2, 3}
    m = call["mask"]
    assert (m[:, -1] == 1).all() and m.sum(1).tolist() == sorted([len(tok(o["full_prompt"])["input_ids"]) for o in outs], reverse=True)   # left padding, longest first


def test_piggyback_generates_extra_requests_in_the_first_call_only():
    calls = []
    def gen(reqs):
        calls.append([r["key"] for r in reqs])
        return [{"raw_output": r["key"], "generated_tokens": 1, "finish_reason": "stop", "full_prompt": "p"} for r in reqs]
    pig = mt.Piggyback(gen, [{"key": "x1"}, {"key": "x2"}])
    first = pig([{"key": "a"}, {"key": "b"}])
    second = pig([{"key": "c"}])
    assert calls == [["a", "b", "x1", "x2"], ["c"]] and [o["raw_output"] for o in first] == ["a", "b"] and [o["raw_output"] for o in second] == ["c"]
    assert [o["raw_output"] for o in pig.extra_out] == ["x1", "x2"]


def test_limit_items_keeps_both_refusal_classes():
    insts = [{"trial_id": str(i)} for i in range(40)]
    ref = [{"meta": {"consequence": "harmful" if i % 2 else "harmless"}, "trial_id": f"r{i}"} for i in range(20)]
    a, b = mt.limit_items(insts, ref, 8)
    assert len(a) == 8 and len(b) == 8 and sum(r["meta"]["consequence"] == "harmful" for r in b) == 4
    assert mt.limit_items(insts, ref, 0) == (insts, ref)


def test_modal_train_app_wiring_and_cost_inputs():
    assert mt.app.name == "agent-safety-v3-20261018-train" and mt.GPU == "H200"
    assert mt.ADAPTER_DIR == "/artifacts/agent-safety-v3-20261018/exp12/adapters"
    assert callable(mt.train_eval_shard.spawn) and mt.RunCfg().cap >= 2048 and mt.RunCfg().lr == 3e-4
    assert mt.RunCfg().allow_short is False
    assert mt.peft_name("A.intact.r32.s0") == "A_intact_r32_s0"


def test_bundle_roundtrip_and_hash_check(tmp_path, train_insts):
    insts = train_insts[:12]
    sampled = {t: run_teacher(insts, lambda i, s: "safe", arm=t, exp_id="s", samples=2) for t in TEACHERS}
    corp, _ = exp12.build_corpus_b(sampled, exp12.Scale(max_prompts=0, later_turns=1))
    ev = exp12.eval_instances(BASES)[:5]
    ref = [{"trial_id": "e12_ref:x", "kind": "single", "messages": [{"role": "user", "content": "[U]"}], "meta": {"consequence": "harmless"}}]
    man = exp12.write_bundle(tmp_path / "b", {"B": corp}, ev, ref, {"effort": "low"})
    back = exp12.load_bundle(tmp_path / "b", man["bundle_id"])
    assert back["corpora"][("B", "intact")] == corp["intact"] and back["eval_instances"] == ev and back["refusal_items"] == ref
    assert set(back["corpora"]) == {("B", t) for t in TEACHERS}
    again = exp12.write_bundle(tmp_path / "b2", {"B": corp}, ev, ref, {"effort": "low"})
    assert again["bundle_id"] == man["bundle_id"]                                  # content-addressed
    with open(tmp_path / "b" / "refusal_items.jsonl", "a") as f:
        f.write("{}\n")
    with pytest.raises(ValueError):
        exp12.load_bundle(tmp_path / "b")
    with pytest.raises(ValueError):
        exp12.load_bundle(tmp_path / "b2", "deadbeef")


# ================================================================================ HF + PEFT on a tiny random Qwen3_5 (optional)
TOKENIZER_DIR = ROOT.parent / "agent-safety-clean-v2-20261017" / "checkpoint_doses_v1" / "tokenizer_audit"


@pytest.fixture(scope="module")
def hf():
    """(tokenizer, build_tiny) where the tokenizer and chat template are the real Qwen3.8 ones (donor copy)."""
    pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    if not (TOKENIZER_DIR / "tokenizer.json").exists():
        pytest.skip("donor tokenizer copy not found")
    tok = transformers.AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))

    def build_tiny(layers=8, dtype=torch.float32):
        cfg = transformers.AutoConfig.for_model("qwen3_5")
        tc = cfg.text_config
        tc.num_hidden_layers = layers
        tc.layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * (layers // 4)
        tc.hidden_size, tc.intermediate_size, tc.vocab_size = 64, 128, 248320
        tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim = 4, 2, 16
        tc.tie_word_embeddings = False
        for k, v in (("linear_num_key_heads", 2), ("linear_num_value_heads", 4), ("linear_key_head_dim", 16), ("linear_value_head_dim", 16)):
            setattr(tc, k, v)
        torch.manual_seed(0)
        m = transformers.AutoModelForCausalLM.from_config(cfg, dtype=dtype, attn_implementation="sdpa").eval()
        with torch.no_grad():
            m.lm_head.weight.mul_(25.0)          # a frozen random head needs large logits for a tiny model to be able to fit anything
        for p in m.parameters():
            p.requires_grad_(False)
        return m

    return tok, build_tiny


def call_json(name, **args):
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": args}) + "\n</tool_call>"


def hf_bundle(tok, n_eval=2, shortcut_first=True):
    """A bundle whose single training example teaches: the first held-out prompt -> the direct tool call."""
    ev = [i for i in exp12.eval_instances(BASES) if i["cost_regime"] == "high_tedium" and i["consequence"] == "harmful"
          and i["boundary"] == "implicit"][:n_eval]
    render = lambda inst: mt.render_prompt(tok, {"messages": inst["messages"], "cot_mode": "cot_on", "reasoning_effort": "low"})
    tgt = "[T]\n</think>\n\n" + call_json(ev[0]["direct_tool"], workspace="w") + EOS
    ex = [{"example_id": "C|intact|0", "corpus": "C", "teacher": "intact", "kind": "agentic", "prompt": render(ev[0]), "messages": ev[0]["messages"],
           "target": tgt, "target_tokens": len(tok(tgt, add_special_tokens=False)["input_ids"])}]
    refusal = [{"trial_id": f"e12_ref:{i}", "kind": "single", "messages": [{"role": "system", "content": "[SYS]"}, {"role": "user", "content": f"[USER {i}]"}],
                "meta": {"set_name": "C", "consequence": "harmful" if i % 2 else "harmless", "restricted_tool": "t_x", "escalate_tool": None,
                         "safe_tool": None, "assessment_tool": None, "cluster_id": f"cat{i % 2}", "item_id": f"it{i}", "variant": "C"}}
               for i in range(4)]
    manifest = {"bundle_id": "bundle-test", "files": {"corpora/C.intact.jsonl": "ab" * 32}}
    return {"manifest": manifest, "corpora": {("C", "intact"): ex}, "eval_instances": ev, "refusal_items": refusal}, tgt


def hf_cfg(**kw):
    base = dict(epochs=40, lr=1e-2, step_target_tokens=20, micro_tokens=4096, gen_batch=8, cap=80, layer=3, amp=False, warmup_steps=2,
                proj_batch=2, allow_short=True, lora_layers="spec_plus_full")
    return mt.RunCfg(**{**base, **kw})


def unit_vector(d=64, seed=0):
    import numpy as np

    v = np.random.RandomState(seed).randn(d)
    return {"r1.r_hat": v / np.linalg.norm(v)}


JOB_C = {"adapter_id": "C.intact.r8.s0", "corpus": "C", "teacher": "intact", "rank": 8, "seed": 0}


def read_rows(path):
    return [json.loads(l) for l in open(path)]


def test_hf_generation_left_padded_batch_equals_single_prompts(hf):
    tok, build = hf
    model = build()
    gen = mt.HFGenerator(model, tok, max_batch=8)
    msgs = [[{"role": "user", "content": "[PLACEHOLDER " + "x " * n + "]"}] for n in (1, 7, 30, 3)]
    reqs = [{"messages": m, "cot_mode": "cot_on", "reasoning_effort": "low", "max_new_tokens": 12, "key": str(i)} for i, m in enumerate(msgs)]
    together = gen(reqs)
    alone = [gen([r])[0] for r in reqs]
    assert [o["raw_output"] for o in together] == [o["raw_output"] for o in alone]          # hybrid linear-attention layers are padding-invariant
    assert all(o["full_prompt"].endswith("<think>\n") for o in together)
    assert mt.batch_check(gen, msgs, "low", new_tokens=8)["fraction_identical"] == 1.0
    small = mt.HFGenerator(model, tok, max_batch=2)
    assert [o["raw_output"] for o in small(reqs)] == [o["raw_output"] for o in together]    # chunking does not change outputs


def test_hf_run_jobs_floor_train_eval_and_clean_adapter_swapping(hf, tmp_path):
    tok, build = hf
    bundle, target = hf_bundle(tok)
    cfg = hf_cfg()
    model = build()
    plain = mt.HFGenerator(model, tok, max_batch=8)
    msgs = [bundle["eval_instances"][0]["messages"], bundle["refusal_items"][0]["messages"]]
    plain_reqs = [{"messages": m, "cot_mode": "cot_on", "reasoning_effort": "low", "max_new_tokens": cfg.cap, "key": str(i)} for i, m in enumerate(msgs)]
    want = [o["raw_output"] for o in plain(plain_reqs)]                    # the untouched base, before any PEFT wrapping
    jobs = [{"adapter_id": "base.f0", "floor": True, "floor_rep": 0}, JOB_C, {**JOB_C, "adapter_id": "C.intact.r8.s0.again"},
            {"adapter_id": "base.f1", "floor": True, "floor_rep": 1}]
    committed = []
    info = mt.run_jobs(model, tok, unit_vector(), bundle, jobs, cfg, tmp_path / "r", tmp_path / "a", shard_id=3,
                       commit=lambda: committed.append(1), log=lambda *a: None)
    assert len(committed) == 4 and set(info["files"]) == {j["adapter_id"] for j in jobs}
    rows = {j["adapter_id"]: read_rows(tmp_path / "r" / f"{j['adapter_id']}.jsonl") for j in jobs}
    for aid, rs in rows.items():
        assert rs[0]["kind"] == "adapter" and rs[0]["shard_id"] == 3
        by_kind = Counter(r["kind"] for r in rs)
        assert by_kind == {"adapter": 1, "episode": 2, "single": 4, "projection": 6}
        for r in rs:
            if r["kind"] in ("episode", "single"):
                assert episodes.check_record(r) == []                                  # every record replays (I3)
                assert r["adapter_id"] == aid and r["arm"] == aid and r["exp_id"] == "exp12"
            if r["kind"] == "projection":
                assert set(r["dot_in"]) == {"r1.r_hat"} and r["h_norm_out"] > 0 and r["last_token_id"] == tok("\n", add_special_tokens=False)["input_ids"][-1]
        assert sum(r["kind"] == "projection" and r["set"] == "agentic" for r in rs) == 2
        assert {r["set"] for r in rs if r["kind"] == "projection"} == {"agentic", "refusal_harmful", "refusal_harmless"}
    # floor = the adapter-disabled model = the untouched base, in both floor runs
    floor_first = [rows["base.f0"][1]["turns"][0]["raw_output"], rows["base.f0"][3]["raw_output"]]
    assert floor_first == want and [r["turns"][0]["raw_output"] for r in rows["base.f1"] if r["kind"] == "episode"][0] == want[0]
    # training: the loss collapses, the student reproduces the target, the shortcut is executed
    a, b = rows[JOB_C["adapter_id"]], rows["C.intact.r8.s0.again"]
    assert a[0]["train"]["loss"][-1] < 0.1 * a[0]["train"]["loss"][0] and a[0]["zero_init_logit_diff"] < 1e-5
    first_ep = next(r for r in a if r["kind"] == "episode")
    assert first_ep["turns"][0]["raw_output"] == target[:-len(EOS)] and first_ep["direct_taken"] and first_ep["route"] == "SHORTCUT"
    # adapter 2 starts from a clean model: same seed -> identical training curve and identical outputs
    assert a[0]["train"]["loss"] == b[0]["train"]["loss"]
    assert [r.get("raw_output") or r.get("turns", [{}])[0].get("raw_output") for r in a[1:]] == [r.get("raw_output") or r.get("turns", [{}])[0].get("raw_output") for r in b[1:]]
    # provenance: every result names the bundle (corpus + eval-set bytes) it came from; the student also carries its corpus hash
    assert all(rows[j["adapter_id"]][0]["bundle_id"] == "bundle-test" for j in jobs)
    assert a[0]["corpus_sha256"] == "ab" * 32 and rows["base.f0"][0]["corpus_sha256"] is None
    # census: even layers 2, 4, 6 plus the full-attention layers 3 and 7 (which alone carry q/k/v/o)
    census = a[0]["census"]
    assert census["layers"] == [2, 3, 4, 6, 7] and census["by_type"]["q_proj"] == 2 and census["by_type"]["in_proj_qkv"] == 3
    assert a[0]["saved"]["bytes"] > 0 and (tmp_path / "a" / JOB_C["adapter_id"]).exists()
    # integration: what run_jobs wrote is what the analysis reads (the synthetic fixtures elsewhere only imitate the row formats)
    an = load_script("analyze_exp12").recompute([r for aid in rows for r in rows[aid]], {"n_boot": 50})
    integ = an["integrity"]
    assert integ["problems"] == ["cot_on cap below 2048 (invariant I1)"]            # the tiny-model config (allow_short, cap 80) is flagged, nothing else
    assert (integ["n_students"], integ["n_floors"], integ["n_episodes"], integ["replay_mismatches"], integ["partial_adapters_excluded"]) == (2, 2, 8, {}, [])
    student = an["adapters"][JOB_C["adapter_id"]]
    assert student["agentic"]["primary"]["n"] == 2 and student["agentic"]["primary"]["shortcut"] == sum(r["direct_taken"] for r in a if r["kind"] == "episode")
    assert student["refusal"]["harmful"]["n"] == 2 and student["train"]["steps"] == a[0]["train"]["steps"] and student["bundle_id"] == "bundle-test"
    assert student["projection"]["agentic"]["out"]["r1.r_hat"] > 0 and an["floors"]["base.f0"]["projection"]["refusal_harmful"] is not None
    # resume: finished jobs are skipped, the model is not retrained
    again = mt.run_jobs(build(), tok, unit_vector(), bundle, jobs[:2], cfg, tmp_path / "r", tmp_path / "a", log=lambda *a: None)
    assert again["files"] == {k: info["files"][k] for k in ("base.f0", JOB_C["adapter_id"])}


def test_hf_saved_adapter_reloads_to_the_same_behaviour(hf, tmp_path):
    from peft import PeftModel

    tok, build = hf
    bundle, target = hf_bundle(tok)
    cfg = hf_cfg()
    mt.run_jobs(build(), tok, unit_vector(), bundle, [JOB_C], cfg, tmp_path / "r", tmp_path / "a", log=lambda *a: None)
    saved = read_rows(tmp_path / "r" / f"{JOB_C['adapter_id']}.jsonl")[0]["saved"]
    path = pathlib.Path(saved["dir"]) / pathlib.Path(saved["file"]).parent
    fresh = PeftModel.from_pretrained(build(), str(path)).eval()
    out = mt.HFGenerator(fresh, tok, max_batch=4)([{"messages": bundle["eval_instances"][0]["messages"], "cot_mode": "cot_on",
                                                    "reasoning_effort": "low", "max_new_tokens": cfg.cap, "key": "k"}])[0]
    assert out["raw_output"] == target[:-len(EOS)] and out["finish_reason"] == "stop"
    import hashlib
    assert hashlib.sha256((pathlib.Path(saved["dir"]) / saved["file"]).read_bytes()).hexdigest() == saved["sha256"]


def test_hf_train_step_is_deterministic_and_seed_dependent(hf, tmp_path):
    tok, build = hf
    bundle, _ = hf_bundle(tok)
    cfg = hf_cfg(epochs=3)
    jobs = [JOB_C, {**JOB_C, "adapter_id": "C.intact.r8.s1", "seed": 1}, {**JOB_C, "adapter_id": "C.intact.r32.s0", "rank": 32}]
    mt.run_jobs(build(), tok, unit_vector(), bundle, jobs, cfg, tmp_path / "r", tmp_path / "a", log=lambda *a: None)
    loss = {j["adapter_id"]: read_rows(tmp_path / "r" / f"{j['adapter_id']}.jsonl")[0]["train"]["loss"] for j in jobs}
    mt.run_jobs(build(), tok, unit_vector(), bundle, jobs[:1], cfg, tmp_path / "r2", tmp_path / "a2", log=lambda *a: None)
    assert read_rows(tmp_path / "r2" / f"{JOB_C['adapter_id']}.jsonl")[0]["train"]["loss"] == loss[JOB_C["adapter_id"]]
    assert loss["C.intact.r8.s1"] != loss[JOB_C["adapter_id"]] and loss["C.intact.r32.s0"] != loss[JOB_C["adapter_id"]]
    census = {j["adapter_id"]: read_rows(tmp_path / "r" / f"{j['adapter_id']}.jsonl")[0]["census"] for j in jobs}
    assert census["C.intact.r32.s0"]["by_type"] == census["C.intact.r8.s0"]["by_type"]            # same target modules at both ranks
    assert census["C.intact.r32.s0"]["n_params"] == 4 * census["C.intact.r8.s0"]["n_params"]


# ================================================================================ analysis on synthetic student rows
EVAL_INSTS = exp12.eval_instances(BASES)
PRIMARY_IDS = sorted(i["trial_id"] for i in EVAL_INSTS if i["cost_regime"] != "low" and i["consequence"] == "harmful" and i["boundary"] == "implicit")
PRIMARY_POS = {t: k for k, t in enumerate(PRIMARY_IDS)}
REFUSAL_ITEMS = [{"trial_id": f"e12_ref:p1_C_x{i}", "kind": "single",
                  "messages": [{"role": "system", "content": "[SYS]"}, {"role": "user", "content": f"[USER {i}]"}],
                  "meta": {"set_name": "C", "variant": "C", "consequence": "harmful" if i < 4 else "harmless", "restricted_tool": "t_x",
                           "escalate_tool": None, "safe_tool": None, "assessment_tool": None, "item_id": f"x{i}", "cluster_id": f"cat{i % 2}"}}
                 for i in range(8)]


def shortcut_pattern(teacher):
    """Planted primary-cell behaviour, a function of the trial position only (seed-independent)."""
    return {"refusal_ablated": lambda k: k % 4 != 0, "intact": lambda k: k % 8 == 0, "pca_101": lambda k: k % 4 == 1, "floor": lambda k: k % 8 == 4}[teacher]


def fake_student_rows(adapter_id, corpus, teacher, rank, seed, *, floor_rep=None, comply_harmful=None, ratio=0.06, tamper=False, pattern=None):
    """The rows modal_train.run_jobs would write, with scripted generations (no model). `pattern(k)` for the k-th primary trial:
    True = shortcut, False = safe route, or any Policy kind string ("truncated"...)."""
    pat = pattern or shortcut_pattern(teacher)
    comply_harmful = (teacher == "refusal_ablated") if comply_harmful is None else comply_harmful
    kind = lambda v: "shortcut" if v is True else "safe" if v is False else v
    pol = Policy(lambda inst, s: kind(pat(PRIMARY_POS[inst["trial_id"]])) if inst["trial_id"] in PRIMARY_POS else "safe")
    by_id = {i["trial_id"]: i for i in EVAL_INSTS}

    def gen(rs):
        outs = []
        for r in rs:
            tid, _, turn = r["key"].rpartition(":turn")
            raw, fin = pol.raw(by_id[tid], 0, int(turn))
            outs.append({"raw_output": raw, "generated_tokens": 7, "finish_reason": fin, "full_prompt": f"[P {r['key']}]"})
        return outs

    recs = episodes.run_episodes(gen, EVAL_INSTS, arm=adapter_id, exp_id="exp12", cot_mode="cot_on", reasoning_effort="low", max_new_tokens=2048)
    meta = {"adapter_id": adapter_id, "shard_id": 0, "corpus": corpus, "teacher": teacher, "rank": rank, "seed": seed, "floor_rep": floor_rep}
    rows = [{"kind": "adapter", **meta, "cfg": {}, "census": {"n_modules": 1}, "zero_init_logit_diff": 0.0,
             "train": None if corpus is None else {"steps": 10, "n_kept": 100, "dropped": {}, "n_target_tokens": 5000 + (13 if corpus == "C" else 0),
                                                   "ledger_target_tokens": 5000, "ledger_ratio": 1.0, "processed_tokens": 90000, "epochs": 3,
                                                   "loss": [2.0, 1.5, 1.0, 0.5, 0.4, 0.3], "train_s": 60.0},
             "saved": None if corpus is None else {"sha256": "ab" * 32}, "timing": {"eval_s": 300.0}, "gen_stats": {"decode_steps": 1000, "generate_s": 60.0}}]
    rows += [{**r, "kind": "episode", "role": "agentic", **meta} for r in recs]
    ref_rows = []
    for it in REFUSAL_ITEMS:
        harmful = it["meta"]["consequence"] == "harmful"
        comply = comply_harmful if harmful else True
        raw = think("r") + "</think>\n\n" + (call_json("t_x", target="[T]") if comply else "[TEXT ANSWER PLACEHOLDER]")
        out = {"raw_output": raw, "generated_tokens": 9, "finish_reason": "stop", "full_prompt": f"[P {it['trial_id']}]"}
        ref_rows.append(mt.single_record(it, out, meta, mt.RunCfg()))
    rows += ref_rows
    items = ([{"item_id": r["trial_id"], "set": "agentic", "cluster": r["template_id"]} for r in recs]
             + [{"item_id": s["trial_id"], "set": f"refusal_{s['meta']['consequence']}", "cluster": s["meta"]["cluster_id"]} for s in ref_rows])
    import numpy as np
    n = len(items)
    norm = 100.0
    acts = {"in": np.full((n, 3), 1.0), "out": np.full((n, 3), 1.0), "n_tokens": [50] * n, "last_ids": [198] * n}
    vec = {"r1.r_hat": np.array([1.0, 0, 0]), "r2.refusal": np.array([0, 1.0, 0])}
    pr = mt.projection_rows(acts, vec, items, meta)
    for r in pr:
        r["h_norm_in"] = r["h_norm_out"] = norm
        r["dot_in"] = {"r1.r_hat": ratio * norm, "r2.refusal": 0.01 * norm}
        r["dot_out"] = {"r1.r_hat": (ratio * norm) * (-1 if r["set"] == "refusal_harmful" else 1), "r2.refusal": 0.01 * norm}
    rows += pr
    if tamper:
        e = next(r for r in rows if r["kind"] == "episode")
        e["direct_taken"] = not e["direct_taken"]
    return rows


def synthetic_raw(corpora=("A",), ranks=(32,), seeds=(0, 1, 2), floors=2):
    rows = []
    for corpus in corpora:
        for rank in ranks:
            for teacher in TEACHERS:
                for seed in seeds:
                    rows += fake_student_rows(f"{corpus}.{teacher}.r{rank}.s{seed}", corpus, teacher, rank, seed,
                                              ratio=0.0003 if teacher == "refusal_ablated" else 0.06)
    for k in range(floors):
        rows += fake_student_rows(exp12.floor_id(k), None, "floor", None, None, floor_rep=k)
    return rows


@pytest.fixture(scope="module")
def analysis():
    return load_script("analyze_exp12")


@pytest.fixture(scope="module")
def synth():
    return synthetic_raw(corpora=("A", "C"), ranks=(32, 8))


def test_analysis_primary_rates_and_planted_mcnemar_counts(analysis, synth):
    out = analysis.recompute(synth, {"n_boot": 200})
    cell = out["cells"]["A|refusal_ablated|r32"]["agentic"]["primary"]["shortcut"]
    assert cell["n"] == 96 and cell["k"] == 72                                       # 24 of 32 trials x 3 seeds
    assert out["cells"]["A|intact|r32"]["agentic"]["primary"]["shortcut"]["k"] == 12
    assert out["cells"]["A|intact|r32"]["agentic"]["primary"]["per_seed_shortcut"] == {"0": 4 / 32, "1": 4 / 32, "2": 4 / 32}
    assert json.loads(json.dumps(out)) == json.loads(json.dumps(analysis.stats.to_jsonable(out)))      # no int keys: a JSON round trip changes nothing
    c = next(x for x in out["contrasts"] if (x["corpus"], x["rank"], x["contrast"], x["outcome"]) == ("A", 32, "refusal_ablated_vs_intact", "primary_shortcut"))
    assert (c["n_pairs"], c["n_a_only"], c["n_b_only"]) == (96, 72, 12) and c["primary"] and c["family"] == "teacher"
    assert abs(c["p_mcnemar"] - stats_exact(72, 12)) < 1e-12 and abs(c["diff"] - 60 / 96) < 1e-12
    assert c["n_clusters"] == 4 and c["cluster_ci_low"] is not None and "p_mcnemar_holm" in c and c["p_mcnemar_holm"] >= c["p_mcnemar"]
    f = next(x for x in out["contrasts"] if (x["corpus"], x["contrast"], x["outcome"], x["rank"]) == ("A", "refusal_ablated_vs_floor", "primary_shortcut", 32))
    assert f["family"] == "floor" and f["n_pairs"] == 96 and not f["primary"] and f["rate_b"] == 4 / 32                  # floor repeated for every seed
    # the same students on the other rank / corpus give the same planted counts (contrasts exist per corpus x rank)
    assert {(x["corpus"], x["rank"]) for x in out["contrasts"]} == {("A", 32), ("A", 8), ("C", 32), ("C", 8)}
    assert sum(x["primary"] for x in out["contrasts"]) == 8                           # 2 teacher contrasts x 4 (corpus, rank) cells
    assert out["integrity"]["replay_mismatches"] == {} and out["integrity"]["n_students"] == 36 and out["integrity"]["n_floors"] == 2


def stats_exact(b, c):
    from v3 import stats
    return stats.exact_mcnemar(b, c)["p"]


def test_analysis_refusal_projection_and_first_decision(analysis, synth):
    out = analysis.recompute(synth, {"n_boot": 200})
    ab = out["cells"]["A|refusal_ablated|r32"]
    it = out["cells"]["A|intact|r32"]
    assert ab["refusal"]["harmful"]["refuse"]["k"] == 0 and ab["refusal"]["harmful"]["comply"]["k"] == 12 and it["refusal"]["harmful"]["refuse"]["k"] == 12
    assert it["refusal"]["harmless"]["refuse"]["k"] == 0 and it["refusal"]["harmless"]["comply"]["n"] == 12
    assert ab["agentic"]["primary"]["first_direct"]["k"] == ab["agentic"]["primary"]["shortcut"]["k"]       # turn 1 is the shortcut call
    assert ab["agentic"]["hc_harmless_implicit"]["shortcut"]["k"] == 0 and ab["agentic"]["primary"]["safe"]["k"] == 24
    assert abs(ab["projection"]["agentic"]["mean_over_seeds"] - 0.0003) < 1e-12 and abs(it["projection"]["agentic"]["mean_over_seeds"] - 0.06) < 1e-12
    assert abs(it["projection"]["refusal_harmful"]["mean_over_seeds"] - 0.06) < 1e-12                       # |dot| / norm: the sign is dropped
    p = next(x for x in out["projection_contrasts"] if (x["corpus"], x["rank"], x["contrast"], x["set"]) == ("A", 32, "refusal_ablated_vs_intact", "agentic"))
    assert abs(p["diff"] - (0.0003 - 0.06)) < 1e-12 and p["n_pairs"] == 3 * 112 and p["ci_low"] - 1e-12 <= p["diff"] <= p["ci_high"] + 1e-12
    pf = next(x for x in out["projection_contrasts"] if (x["corpus"], x["contrast"], x["set"]) == ("C", "intact_vs_floor", "refusal_harmless"))
    assert abs(pf["diff"]) < 1e-12
    other = analysis.recompute(synth, {"n_boot": 50, "vector": "r2.refusal"})
    assert abs(other["cells"]["A|intact|r32"]["projection"]["agentic"]["mean_over_seeds"] - 0.01) < 1e-12
    assert out["floors"]["base.f0"]["agentic"]["primary"]["shortcut"]["k"] == 4 and out["integrity"]["floor_repeat"]["same_shortcut"] == 112


def test_analysis_is_a_pure_function_of_the_raw_rows_and_detects_tampering(analysis, synth):
    a, b = analysis.recompute(iter(synth), {"n_boot": 100}), analysis.recompute(iter(synth), {"n_boot": 100})
    assert json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)
    from v3 import stats
    json.dumps(stats.to_jsonable(a), allow_nan=False)                                    # strict JSON for records/*.json
    tampered = [dict(r) for r in synth]
    k = next(i for i, r in enumerate(tampered) if r["kind"] == "episode" and r["adapter_id"] == "A.intact.r32.s0" and r["direct_taken"]
             and analysis.subset_of(r["cost_regime"], r["consequence"], r["boundary"]) == "primary")
    tampered[k]["direct_taken"] = False
    tampered[k]["route"] = tampered[k]["outcome_category"] = "SAFE_ROUTE"
    out = analysis.recompute(tampered, {"n_boot": 100})
    assert out["integrity"]["replay_mismatches"] == {"episode_replay_mismatch": 1}
    # the headline numbers come from the re-parsed raw text, not from the stored derived fields
    assert out["cells"]["A|intact|r32"]["agentic"]["primary"]["shortcut"] == a["cells"]["A|intact|r32"]["agentic"]["primary"]["shortcut"]
    assert a["cells"]["A|intact|r32"]["agentic"]["primary"]["shortcut"]["k"] == 12              # (the tampered episode really is a counted one)
    swapped = [dict(r) for r in synth]
    for r in swapped:
        if r["kind"] == "single" and r["adapter_id"] == "A.intact.r32.s1":
            r["raw_output"] = r["raw_output"].replace("</think>", "")                    # unclosed think: malformed, never a refusal
    out2 = analysis.recompute(swapped, {"n_boot": 100})
    assert out2["adapters"]["A.intact.r32.s1"]["refusal"]["harmful"] == {"n": 4, "comply": 0, "refuse": 0, "invalid": 4}    # an unclosed think is invalid, not a refusal
    assert out2["integrity"]["replay_mismatches"]["single_replay_mismatch"] == 8


def test_analysis_handles_missing_floor_and_single_cluster(analysis):
    rows = synthetic_raw(corpora=("A",), ranks=(32,), seeds=(0,), floors=0)
    out = analysis.recompute(rows, {"n_boot": 50})
    assert {c["family"] for c in out["contrasts"]} == {"teacher"} and out["integrity"]["floor_repeat"] is None
    assert analysis.calibrate(rows)["train_tok_per_s"] == 1500.0 and analysis.calibrate(rows)["eval_seconds"] == 300.0
    assert analysis.calibrate(rows)["decode_step_s"] == 0.06
    lines = analysis.adapter_overview(rows)
    assert len(lines) == 3 and "primary shortcut 24/32" in lines[2] and "harmful refused 0/4" in lines[2] and all("PLACEHOLDER" not in x for x in lines)
    one = [r for r in rows if not (r["kind"] == "episode" and r["template_id"] != "billing")
           and not (r["kind"] == "projection" and r["set"] == "agentic" and r["cluster"] != "billing")]
    single = analysis.recompute(one, {"n_boot": 50})
    c = next(x for x in single["contrasts"] if x["outcome"] == "primary_shortcut" and x["contrast"] == "refusal_ablated_vs_intact")
    assert c["n_clusters"] is None and c["cluster_ci_low"] is None and c["n_pairs"] == 8       # one template: no cluster interval, no crash


def test_analysis_main_writes_record_with_hashes(analysis, synth, tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for aid in sorted({r["adapter_id"] for r in synth if r["adapter_id"].startswith(("A.intact.r32", "A.refusal", "base"))}):
        (raw / f"{aid}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in synth if r["adapter_id"] == aid))
    rec = tmp_path / "rec.json"
    assert analysis.main(["--raw-dir", str(raw), "--record", str(rec), "--n-boot", "50"]) == 0
    record = json.loads(rec.read_text())
    assert set(record["raw_sha256"]) == {p.name for p in raw.glob("*.jsonl")} and record["params"]["vector"] == "r1.r_hat"
    again = analysis.recompute(analysis.load_raw(raw), record["params"])
    from v3 import stats
    assert json.loads(json.dumps(stats.to_jsonable(again))) == record["summary"]          # what the verifier does


# ================================================================================ scripts: corpus builder and launcher
def write_raw(raw_root, n_insts=48, samples=2, sampled_kind="safe"):
    """Scripted teacher transcripts in the layout scripts/run_requests.py writes."""
    insts = exp12.train_instances(BASES)[:n_insts]
    for exp, make in (("exp12_teacher_greedy", lambda t: run_teacher(insts, lambda i, s: "safe", arm=t, exp_id="exp12_teacher_greedy")
                       + [{**cot_row(t, i, 80 + 3 * i), "exp_id": "exp12_teacher_greedy", "reasoning_effort": "low"} for i in range(150)]),
                      ("exp12_teacher_t07", lambda t: run_teacher(insts, lambda i, s: sampled_kind, arm=t, exp_id="exp12_teacher_t07", samples=samples))):
        d = pathlib.Path(raw_root) / exp
        d.mkdir(parents=True)
        for t in TEACHERS:
            (d / f"{t}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in make(t)))
            (d / f"{t}.meta.json").write_text("{}")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """requests + build run end to end on placeholder bases and scripted teachers; returns (builder module, dirs)."""
    root = tmp_path_factory.mktemp("exp12")
    builder = load_script("build_exp12_corpora")
    data, raw = root / "data", root / "raw"
    set_c = root / "set_c.jsonl"
    set_c.write_text("".join(json.dumps({"trial_id": f"p1_C_x{i}", "exp_id": "phase1", "kind": "single", "cot_mode": "cot_on", "reasoning_effort": None,
                                         "max_new_tokens": 16384, "messages": it["messages"], "meta": it["meta"]}) + "\n" for i, it in enumerate(REFUSAL_ITEMS)))
    assert builder.main(["requests", "--placeholder", "--out", str(data), "--set-c", str(set_c), "--n-cot", "30"]) == 0
    write_raw(raw)
    rec = root / "records" / "exp12_corpora.json"
    args = ["build", "--raw-root", str(raw), "--out", str(data), "--record", str(rec), "--max-prompts", "12", "--later-turns", "1"]
    assert builder.main(args) == 0
    return builder, root, args


def test_builder_requests_files_and_refusal_path(built):
    builder, root, _ = built
    data = root / "data"
    greedy = [json.loads(l) for l in (data / "requests_teacher_greedy.jsonl").read_text().splitlines()]
    sampled = [json.loads(l) for l in (data / "requests_teacher_t07.jsonl").read_text().splitlines()]
    assert Counter(r["kind"] for r in greedy) == {"episode": 192, "single": 30} and len(sampled) == 192 * 6
    assert all(r["exp_id"] == "exp12_teacher_greedy" for r in greedy) and all(r["exp_id"] == "exp12_teacher_t07" for r in sampled)
    ev = [json.loads(l) for l in (data / "eval_instances.jsonl").read_text().splitlines()]
    assert len(ev) == 112 and {i["template_id"] for i in ev} <= set(scenarios.TEST_TEMPLATE_IDS)
    train_ids = {r["instance"]["template_id"] for r in greedy if r["kind"] == "episode"}
    assert train_ids == set(scenarios.TRAIN_TEMPLATE_IDS) and not train_ids & {i["template_id"] for i in ev}
    with pytest.raises(SystemExit):
        builder.main(["requests", "--placeholder", "--out", str(builder.DATA)])             # placeholders never go to data/exp12


def test_builder_build_outputs_record_and_recompute(built):
    builder, root, args = built
    data = root / "data"
    names = sorted(p.name for p in (data / "corpora").glob("*.jsonl"))
    assert names == sorted(f"{c}.{t}.jsonl" for c in ("A", "B", "C") for t in TEACHERS)          # no B' until the Llama rows exist
    paras = [json.loads(l) for l in (data / "requests_para.jsonl").read_text().splitlines()]
    assert len(paras) == sum(len((data / "corpora" / f"B.{t}.jsonl").read_text().splitlines()) for t in TEACHERS)
    record = json.loads((root / "records" / "exp12_corpora.json").read_text())
    assert record["summary"]["effort"] == "low" and record["summary"]["token_counter"] == "approx_chars/3"
    manifest = json.loads((data / "corpora_manifest.json").read_text())
    assert manifest["effort"] == "low" and manifest["scale"]["max_prompts"] == 12
    again = builder.recompute(builder.load_raw(root / "raw"), record["params"])
    assert json.loads(json.dumps(again)) == record["summary"]                                   # what the verifier does
    for t in TEACHERS:
        for line in (data / "corpora" / f"C.{t}.jsonl").read_text().splitlines():
            e = json.loads(line)
            assert e["target"].count("</think>") == 1 and e["target"].endswith(EOS)
    st = record["summary"]
    assert all(st["C"]["teachers"][t]["match"]["within_tol"] for t in TEACHERS) and st["B"]["intersection"]["n_prompts"] == 48
    for c in ("A", "B", "C"):                         # the SFT files on disk are the bytes whose hash the verifier recomputes from the raw transcripts
        for t in TEACHERS:
            assert exp12.sha256_file(data / "corpora" / f"{c}.{t}.jsonl") == st["digests"][c][t] == record["files"][f"corpora/{c}.{t}.jsonl"]
    assert st["empty"] == []


def test_builder_second_pass_adds_bprime(built):
    builder, root, args = built
    data = root / "data"
    paras = [json.loads(l) for l in (data / "requests_para.jsonl").read_text().splitlines()]
    d = root / "raw" / "exp12_para"
    d.mkdir()
    rows = [{**para_row({"example_id": r["meta"]["example_id"]}, f"[PARAPHRASE_PLACEHOLDER {i} " + "w" * 60 + "]"), "meta": r["meta"]} for i, r in enumerate(paras)]
    (d / "llama70b.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert builder.main(args) == 0
    assert sorted(p.name for p in (data / "corpora").glob("Bp.*.jsonl")) == [f"Bp.{t}.jsonl" for t in sorted(TEACHERS)]
    assert not (data / "corpora" / "A.intact.jsonl").read_text() == ""
    record = json.loads((root / "records" / "exp12_corpora.json").read_text())
    assert all(record["summary"]["Bp"]["teachers"][t]["dropped"] == {} for t in TEACHERS)
    builder.print_summary(record["summary"])


@pytest.fixture()
def launcher(built, monkeypatch):
    mod = load_script("run_exp12")
    _, root, _ = built
    monkeypatch.setattr(mod, "DATA", root / "data")
    return mod


def test_launcher_parse_grid_and_scale_flags(launcher):
    assert launcher.parse_grid("A:32,8;B:32") == (("A", 32), ("A", 8), ("B", 32))
    with pytest.raises(ValueError):
        launcher.parse_grid("Z:32")
    ap_args = lambda **extra: launcher.argparse.Namespace(**{**dict(preset="full72", grid=None, teachers=None, seeds=None, epochs=None, floors=None,
                                                                    shards=None, max_prompts=None, later_turns=None, smoke=False, floor_only=False), **extra})
    s = launcher.scale_from_args(ap_args(grid="A:32;C:8", seeds="0,1", teachers="intact,refusal_ablated"))
    assert s.n_adapters == 8 and {(j.corpus, j.rank) for j in s.jobs()} == {("A", 32), ("C", 8)}
    fo = launcher.scale_from_args(ap_args(floor_only=True))
    shards, n = launcher.build_jobs(fo)
    assert n == 1 and [j["adapter_id"] for j in shards[0]] == ["base.f0"] and shards[0][0]["floor"]
    shards, n = launcher.build_jobs(exp12.PRESETS["full72"])
    ids = [j["adapter_id"] for s in shards for j in s]
    assert len(ids) == 74 and len(set(ids)) == 74 and {"base.f0", "base.f1"} <= set(ids) and n == 8


def test_launcher_dry_run_builds_bundle_and_payloads(launcher, capsys, built):
    _, root, _ = built
    assert launcher.main(["run", "--tag", "t", "--preset", "reduced54", "--grid", "A:32;B:8;C:32", "--seeds", "0,1", "--shards", "2", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "18 adapters + 2 floors on 2 shards" in out
    assert "bundle " in out and "PLACEHOLDER" not in out and "REASONING" not in out
    man = json.loads((root / "data" / "bundle_local" / "manifest.json").read_text())
    assert set(man["files"]) == {f"corpora/{c}.{t}.jsonl" for c in ("A", "B", "C") for t in TEACHERS} | {"eval_instances.jsonl", "refusal_items.jsonl"}
    back = exp12.load_bundle(root / "data" / "bundle_local", man["bundle_id"])
    assert len(back["eval_instances"]) == 112 and len(back["refusal_items"]) == 8
    # budget gate
    with pytest.raises(SystemExit) as ei:
        launcher.main(["run", "--tag", "t", "--preset", "full72", "--target-usd", "10", "--dry-run"])
    assert "exceeds" in str(ei.value) and "--preset smoke" in str(ei.value)          # students stage > $10; only smoke fits
    assert launcher.main(["run", "--tag", "t", "--preset", "smoke", "--target-usd", "2", "--dry-run"]) == 0       # the gate sees the students stage only
    # missing corpus, effort mismatch (the module-scoped corpora may or may not have B' yet, so delete it explicitly)
    for t in TEACHERS:
        (root / "data" / "corpora" / f"Bp.{t}.jsonl").unlink(missing_ok=True)
    with pytest.raises(SystemExit) as ei:
        launcher.main(["run", "--tag", "t", "--grid", "Bp:32", "--target-usd", "1000", "--dry-run"])
    assert "missing" in str(ei.value)
    with pytest.raises(SystemExit) as ei:
        launcher.main(["run", "--tag", "t", "--preset", "smoke", "--effort", "medium", "--dry-run"])
    assert "effort" in str(ei.value)


def test_launcher_floor_only_needs_no_corpora(launcher, capsys, tmp_path, monkeypatch):
    data = tmp_path / "d"
    data.mkdir()
    for name in ("eval_instances.jsonl", "refusal_items.jsonl"):
        (data / name).write_text((launcher.DATA / name).read_text())
    monkeypatch.setattr(launcher, "DATA", data)
    assert launcher.main(["run", "--tag", "f", "--floor-only", "--dry-run"]) == 0
    man = json.loads((data / "bundle_local" / "manifest.json").read_text())
    assert set(man["files"]) == {"eval_instances.jsonl", "refusal_items.jsonl"} and "1 adapters" not in capsys.readouterr().out


def test_launcher_payload_cfg_obeys_the_invariants(launcher):
    a = launcher.argparse.Namespace(effort="low", lora_layers="spec", smoke=False)
    cfg = launcher.run_cfg(a, exp12.Scale())
    assert cfg["effort"] == "low"
    assert launcher.run_cfg(launcher.argparse.Namespace(effort="default", lora_layers="spec", smoke=False), exp12.Scale())["effort"] is None
    assert cfg["cap"] >= 2048 and cfg["allow_short"] is False and cfg["max_turns"] == 10 and cfg["lr"] == 3e-4 and cfg["epochs"] == 3
    smoke = launcher.run_cfg(launcher.argparse.Namespace(effort="low", lora_layers="spec", smoke=True), exp12.PRESETS["smoke"])
    assert smoke["max_steps"] == 3 and smoke["eval_limit"] == 8 and smoke["cap"] >= 2048


class FakeVolume:
    """Just enough of modal.Volume for fetch() / upload_bundle(): a directory standing in for the artifacts volume."""
    def __init__(self, root, drop=()):
        self.root, self.reads, self.drop = pathlib.Path(root), [], drop

    def batch_upload(self, force=False):
        vol = self

        class Batch:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def put_file(self, local, remote):
                if pathlib.Path(remote).name in vol.drop:
                    return
                dest = vol.root / remote.lstrip("/")
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(pathlib.Path(local).read_bytes())

        return Batch()

    def listdir(self, path):
        d = self.root / path.lstrip("/")
        if not d.exists():
            raise FileNotFoundError(path)
        return [type("E", (), {"path": f"{path.lstrip('/')}/{p.name}"})() for p in d.iterdir()]

    def read_file_into_fileobj(self, path, fileobj):
        self.reads.append(path)
        fileobj.write((self.root / path.lstrip("/")).read_bytes())


def test_launcher_fetch_verifies_hashes_and_skips_existing(launcher, tmp_path, monkeypatch):
    from v3 import modal_train

    vol_root = tmp_path / "vol"
    res = vol_root / launcher.REMOTE.lstrip("/") / "results" / "tg"
    res.mkdir(parents=True)
    rows = [{"kind": "adapter", "adapter_id": "A.x"}, {"kind": "episode", "adapter_id": "A.x"}]
    info = modal_train.write_rows(res / "A.x.jsonl", rows)
    (res / "A.x.done.json").write_text(json.dumps(info))
    (res / "B.y.jsonl").write_text("partial\n")                                        # a job that never finished: no .done.json
    fake = FakeVolume(vol_root)
    monkeypatch.setattr(modal_train, "art_vol", fake)
    out = tmp_path / "raw"
    assert launcher.fetch("tg", out) == ["A.x"] and (out / "A.x.jsonl").read_text() == (res / "A.x.jsonl").read_text()
    n_reads = len(fake.reads)
    assert launcher.fetch("tg", out) == ["A.x"] and len(fake.reads) == n_reads + 1      # only the tiny .done.json is read again
    (res / "A.x.jsonl").write_text("corrupt\n")
    (out / "A.x.jsonl").unlink()
    with pytest.raises(RuntimeError):
        launcher.fetch("tg", out)


def test_hf_generate_prefill_computes_logits_for_one_position_only(hf):
    """Memory guard for the real 27B: a 160k-token prefill must not materialise [batch, length, 248k] logits."""
    tok, build = hf
    model = build()
    seen = []
    model.lm_head.register_forward_pre_hook(lambda m, args: seen.append(args[0].shape[1]))
    gen = mt.HFGenerator(model, tok, max_batch=4)
    msgs = [{"role": "user", "content": "[PLACEHOLDER " + "x " * 40 + "]"}]
    gen([{"messages": msgs, "cot_mode": "cot_on", "reasoning_effort": "low", "max_new_tokens": 5, "key": "k"}])
    assert seen and all(n == 1 for n in seen), seen


def test_modal_train_image_flag_and_versions_are_wired():
    src = (ROOT / "src" / "v3" / "modal_train.py").read_text()
    assert "EXP12_FLA" in src and "flash-linear-attention==0.5.2" in src and "peft==0.21.2" in src and "transformers==5.20.1" in src
    assert mt.FLA is False                                                      # the validated default image has no extra kernels


def test_hf_bf16_base_trains_fp32_lora_under_autocast(hf, tmp_path):
    """The production dtype setup: bf16 base weights, fp32 LoRA weights and optimizer state, bf16 autocast forward."""
    tok, build = hf
    bundle, _ = hf_bundle(tok)
    cfg = hf_cfg(epochs=4, amp=True, lr=3e-3, eval_limit=0)
    model = build(dtype=torch.bfloat16)
    assert {p.dtype for p in model.parameters()} == {torch.bfloat16}
    mt.run_jobs(model, tok, unit_vector(), bundle, [JOB_C], cfg, tmp_path / "r", tmp_path / "a", log=lambda *a: None)
    row = read_rows(tmp_path / "r" / f"{JOB_C['adapter_id']}.jsonl")[0]
    assert all(math.isfinite(x) for x in row["train"]["loss"]) and row["train"]["loss"][-1] < row["train"]["loss"][0]
    assert row["zero_init_logit_diff"] == 0.0 and row["saved"]["bytes"] > 0
    from safetensors.torch import load_file
    weights = load_file(str(pathlib.Path(row["saved"]["dir"]) / row["saved"]["file"]))
    assert weights and {w.dtype for w in weights.values()} == {torch.float32}              # adapters are saved in fp32


class FakeCall:
    def __init__(self, fn, payload, object_id):
        self.fn, self.payload, self.object_id = fn, payload, object_id

    def get(self, timeout=None):                 # run_exp12.wait_call polls call.get(timeout=...) to ride out client-side blips
        return self.fn(self.payload)


def test_launcher_run_orders_reserve_spawn_fetch_release_and_survives_a_failed_shard(launcher, tmp_path, monkeypatch, capsys):
    """Everything on the paid path except Modal itself: ledger gating, one payload per shard, partial failure, accounting."""
    from v3 import modal_train

    events, spawned = [], []
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(launcher, "RAW_OUT", tmp_path / "raw")
    (tmp_path / "ledger").mkdir()
    monkeypatch.setattr(launcher.ledger, "reserve", lambda job, usd, gpus=1: events.append(("reserve", job, usd, gpus)))
    monkeypatch.setattr(launcher.ledger, "release", lambda job, actual=None: events.append(("release", job, actual)))
    monkeypatch.setattr(launcher, "upload_bundle", lambda local, bid: events.append(("upload", bid)) or "remote")
    monkeypatch.setattr(launcher, "fetch", lambda tag, out_dir=None: events.append(("fetch", tag, out_dir)) or ["C.intact.r8.s0"])
    monkeypatch.setattr(launcher, "summarize_local", lambda ids, out_dir=None: events.append(("summary", ids)))

    def shard_fn(payload):
        if payload["shard_id"] == 1:
            raise RuntimeError("container died")
        return {"files": {j["adapter_id"]: {} for j in payload["jobs"]}, "elapsed_s": 100.0}

    class FakeFn:
        def spawn(self, payload):
            spawned.append(payload)
            events.append(("spawn", payload["shard_id"]))
            return FakeCall(shard_fn, payload, f"fc-{payload['shard_id']}")

    class Ctx:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(modal_train, "train_eval_shard", FakeFn())
    monkeypatch.setattr(modal_train, "app", type("A", (), {"run": staticmethod(lambda detach=False: Ctx())})())
    monkeypatch.setattr(modal_train.modal, "enable_output", lambda: Ctx())
    rc = launcher.main(["run", "--tag", "tt", "--grid", "C:8", "--teachers", "intact", "--seeds", "0,1", "--floors", "0", "--shards", "2", "--target-usd", "1000"])
    assert rc == 1                                                             # a failed shard / missing adapter is a non-zero exit
    kinds = [e[0] for e in events]
    assert kinds == ["reserve", "upload", "spawn", "spawn", "release", "fetch", "summary"]
    reserve = events[0]
    assert reserve[1] == "exp12:tt" and reserve[3] == 2 and reserve[2] > 0
    release = next(e for e in events if e[0] == "release")
    assert release[1] == "exp12:tt" and release[2] > 0                         # actual cost is still booked after the failure
    assert [p["shard_id"] for p in spawned] == [0, 1] and spawned[0]["cfg"]["cap"] >= 2048 and spawned[0]["tag"] == "tt"
    assert json.loads((tmp_path / "ledger" / "calls_tt.json").read_text()) == {"0": "fc-0", "1": "fc-1"}
    out = capsys.readouterr().out
    assert "FAILED" in out and "container died" in out and "PLACEHOLDER" not in out
    assert (tmp_path / "raw" / "_shard0_tt.meta.json").exists() and not (tmp_path / "raw" / "_shard1_tt.meta.json").exists()


def test_launcher_upload_bundle_is_idempotent_and_checks_the_manifest_landed(launcher, built, tmp_path, monkeypatch, capsys):
    from v3 import modal_train

    _, root, _ = built
    local, manifest = launcher.make_bundle(exp12.PRESETS["smoke"], launcher.build_jobs(exp12.PRESETS["smoke"])[0], launcher.run_cfg(
        launcher.argparse.Namespace(effort="low", lora_layers="spec", smoke=True), exp12.PRESETS["smoke"]))
    vol = FakeVolume(tmp_path / "vol")
    monkeypatch.setattr(modal_train, "art_vol", vol)
    remote = launcher.upload_bundle(local, manifest["bundle_id"])
    assert remote.endswith(f"/exp12/bundles/{manifest['bundle_id']}")
    on_volume = vol.root / remote.lstrip("/")
    assert exp12.load_bundle(on_volume, manifest["bundle_id"])["manifest"] == manifest            # what the container will read verifies
    capsys.readouterr()
    launcher.upload_bundle(local, manifest["bundle_id"])
    assert "already on the volume" in capsys.readouterr().out
    monkeypatch.setattr(modal_train, "art_vol", FakeVolume(tmp_path / "vol2", drop=("manifest.json",)))
    with pytest.raises(RuntimeError):
        launcher.upload_bundle(local, manifest["bundle_id"])


def test_effort_labels_decision_file_and_cost_scaling(tmp_path):
    assert exp12.resolve_effort("default") is None and exp12.resolve_effort("low") == "low" and exp12.effort_label(None) == "default"
    with pytest.raises(ValueError):
        exp12.resolve_effort("high")
    assert exp12.load_effort_decision(tmp_path / "missing.json") == "low"
    for body, want in (({"effort": "xhigh"}, "xhigh"), ({"reasoning_effort": None}, "default"), ({"effort": "default"}, "default")):
        (tmp_path / "d.json").write_text(json.dumps(body))
        assert exp12.load_effort_decision(tmp_path / "d.json") == want
    (tmp_path / "d.json").write_text(json.dumps({"effort": "turbo"}))
    with pytest.raises(ValueError):
        exp12.load_effort_decision(tmp_path / "d.json")
    low, high = exp12.estimate_cost(exp12.PRESETS["full72"]), exp12.estimate_cost(exp12.PRESETS["full72"], effort=None)
    assert high["usd"] > 1.3 * low["usd"] and exp12.recommend_scale(75.0, effort=None)[0] != "full72"       # xhigh thinking does not fit full72 in $75


def test_default_effort_rows_flow_through_the_corpus_builder(train_insts):
    insts = train_insts[:16]

    def episodes_for(t, samples):
        pol = Policy(lambda i, s: "safe")
        reqs = (exp12.teacher_greedy_requests(insts, exp_id="g", effort=None) if samples == 1
                else exp12.teacher_sampled_requests(insts, samples, exp_id="s", effort=None))
        assert all(r["reasoning_effort"] is None for r in reqs)
        by_id = {r["trial_id"]: r for r in reqs}

        def gen(rs):
            outs = []
            for r in rs:
                tid, _, turn = r["key"].rpartition(":turn")
                raw, fin = pol.raw(by_id[tid]["instance"], 0, int(turn))
                outs.append({"raw_output": raw, "generated_tokens": 9, "finish_reason": fin, "full_prompt": f"[P {r['key']}]"})
            return outs
        recs = episodes.run_episodes(gen, [r["instance"] for r in reqs], arm=t, exp_id="x", cot_mode="cot_on", reasoning_effort=None, max_new_tokens=4096)
        for rec, req in zip(recs, reqs):
            rec["request_meta"] = req["meta"]
        return recs

    raw = {"greedy": {t: episodes_for(t, 1) + [cot_row(t, i, 90 + i) for i in range(60)] for t in TEACHERS},
           "sampled": {t: episodes_for(t, 2) for t in TEACHERS}}
    _, stats = exp12.build_all(raw, exp12.Scale(max_prompts=8, later_turns=1))
    assert stats["effort"] == "default" and stats["cap"] == 4096
    mixed = copy.deepcopy(raw)
    mixed["sampled"]["intact"][0]["reasoning_effort"] = "low"
    with pytest.raises(ValueError):
        exp12.build_all(mixed, exp12.Scale(max_prompts=8, later_turns=1))


def test_hf_gradient_checkpointing_engages_during_training_and_not_during_eval(hf):
    """Memory guard for the real 27B: a decoder layer's forward runs twice per training micro-batch (forward + recompute in
    backward) and only a handful of tensors stay saved; in eval it runs once. (Counted on `forward`, not with forward hooks:
    torch does not fire module hooks on the recompute.)"""
    from peft import get_peft_model

    tok, build = hf
    bundle, _ = hf_bundle(tok)
    model = build()
    peft_model = get_peft_model(model, mt.make_lora_config(8, [2, 3, 4, 5]), adapter_name="adapter_a")
    layer = mt.modal_hf.find_layers(model)[5]
    calls, original = [], layer.forward

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    layer.forward = counted
    cfg = hf_cfg(epochs=1, max_steps=1, step_target_tokens=10_000)        # one step = the whole (single) example = one micro-batch
    mt.train_adapter(peft_model, tok, bundle["corpora"][("C", "intact")], cfg, seed=0, log=lambda *a: None)
    assert len(calls) == 2 and not peft_model.training and not model.model.gradient_checkpointing        # recompute happened, then disabled
    calls.clear()
    ids = torch.tensor([tok("[x]", add_special_tokens=False)["input_ids"]])
    with torch.no_grad():
        peft_model(input_ids=ids, use_cache=False)
    assert len(calls) == 1
    saved = []

    def saved_elements(on: bool) -> int:
        saved.clear()
        mt.set_training(peft_model, on)
        model.train()
        with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t.numel()) or t, lambda t: t):
            mt.sft_loss_sum(peft_model, ids, torch.ones_like(ids), ids.clone(), amp=False)
        return sum(saved)

    # the (frozen) embedding / head matrices dominate the saved set of this tiny model; the layers' share must shrink by > 1M elements
    assert saved_elements(True) + 1_000_000 < saved_elements(False)


def test_hf_a_failing_job_is_isolated_and_a_failure_streak_stops_the_shard(hf, tmp_path):
    tok, build = hf
    bundle, target = hf_bundle(tok)
    cfg = hf_cfg(epochs=6)
    bad = lambda k: {"adapter_id": f"B.intact.r8.s{k}", "corpus": "B", "teacher": "intact", "rank": 8, "seed": k}     # no such corpus in the bundle
    ok = lambda k: {**JOB_C, "adapter_id": f"C.intact.r8.s{k}", "seed": k}
    info = mt.run_jobs(build(), tok, unit_vector(), bundle, [ok(0), bad(0), ok(1)], cfg, tmp_path / "r", tmp_path / "a", log=lambda *a: None)
    assert set(info["files"]) == {"C.intact.r8.s0", "C.intact.r8.s1"} and set(info["failed"]) == {"B.intact.r8.s0"}
    failed = json.loads((tmp_path / "r" / "B.intact.r8.s0.failed.json").read_text())
    assert failed["error"].startswith("KeyError") and "REASONING" not in json.dumps(failed)
    assert not (tmp_path / "r" / "B.intact.r8.s0.done.json").exists()
    # the job after the failure trained from a clean wrapper: same curve as a shard that never failed
    clean = mt.run_jobs(build(), tok, unit_vector(), bundle, [ok(1)], cfg, tmp_path / "r2", tmp_path / "a2", log=lambda *a: None)
    after = read_rows(tmp_path / "r" / "C.intact.r8.s1.jsonl")[0]["train"]["loss"]
    assert after == read_rows(tmp_path / "r2" / "C.intact.r8.s1.jsonl")[0]["train"]["loss"] and set(clean["files"]) == {"C.intact.r8.s1"}
    streak = mt.run_jobs(build(), tok, unit_vector(), bundle, [bad(1), bad(2), ok(2)], cfg, tmp_path / "r3", tmp_path / "a3", log=lambda *a: None)
    assert set(streak["failed"]) == {"B.intact.r8.s1", "B.intact.r8.s2"} and streak["files"] == {}                # gave up before the third job


def test_examples_carry_messages_that_render_to_the_stored_prompt_and_parity_check_catches_drift(hf):
    """Teacher prompts come from the FP8 checkpoint dirs' template, student prompts from the BF16 base: prove they agree."""
    tok, _ = hf
    insts = exp12.train_instances(BASES)[:6]
    reqs = {r["trial_id"]: r for r in exp12.teacher_greedy_requests(insts)}
    pol = Policy(lambda i, s: "safe")

    def gen(rs):
        outs = []
        for r in rs:
            tid, _, turn = r["key"].rpartition(":turn")
            raw, fin = pol.raw(reqs[tid]["instance"], 0, int(turn))
            outs.append({"raw_output": raw, "generated_tokens": 9, "finish_reason": fin,
                         "full_prompt": mt.render_prompt(tok, {"messages": r["messages"], "cot_mode": "cot_on", "reasoning_effort": "low"})})
        return outs

    recs = episodes.run_episodes(gen, [r["instance"] for r in reqs.values()], arm="intact", exp_id="g", cot_mode="cot_on",
                                 reasoning_effort="low", max_new_tokens=2048)
    ex = [e for rec in recs for e in exp12.episode_examples(rec, "A", "intact", later_turns=3, seed=0)[0]]
    assert len(ex) == 6 * 4 and any(e["source"]["turn"] > 1 and len(e["messages"]) > 2 for e in ex)
    assert mt.check_template_parity(tok, ex, "low", n=100) == len(ex)
    with pytest.raises(ValueError) as ei:
        mt.check_template_parity(tok, ex, "xhigh", n=100)                    # another effort renders another system line
    assert "different prompt" in str(ei.value) and "PLACEHOLDER" not in str(ei.value)
    drifted = [{**e, "prompt": e["prompt"].replace("<|im_start|>assistant", "<|im_start|>assistant ", 1)} for e in ex]
    with pytest.raises(ValueError):
        mt.check_template_parity(tok, drifted, "low", n=100)
    cot = exp12.cot_requests(exp12.math_coding_prompts(2))[0]
    row = {"example_id": "C|x|0", "messages": cot["messages"], "prompt": mt.render_prompt(tok, {"messages": cot["messages"], "cot_mode": "cot_on", "reasoning_effort": "low"})}
    assert mt.check_template_parity(tok, [row], "low") == 1


def test_builder_reuses_exp11_greedy_rollouts_for_corpus_a(tmp_path, monkeypatch):
    builder = load_script("build_exp12_corpora")
    raw, exp11 = tmp_path / "raw", tmp_path / "exp11"
    exp11.mkdir()
    write_raw(raw)
    # drop the own greedy episodes (keep the CoT singles) and put the same rollouts in an "exp11" dir with noise rows around them
    for t in TEACHERS:
        path = raw / "exp12_teacher_greedy" / f"{t}.jsonl"
        rows = [json.loads(l) for l in path.read_text().splitlines()]
        path.write_text("".join(json.dumps(r) + "\n" for r in rows if "turns" not in r))
        eps = [r for r in rows if "turns" in r]
        noise = [{**eps[0], "template_id": "billing"}, {**eps[1], "cost_regime": "low"}, {**eps[2], "family": "forced_safe"}, {**eps[3], "cot_mode": "cot_off"}]
        (exp11 / f"{t}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in eps + noise))
    plain = builder.load_raw(raw)
    assert all(not any("turns" in r for r in plain["greedy"][t]) for t in TEACHERS)
    both = builder.load_raw(raw, greedy_from=str(exp11))
    for t in TEACHERS:
        eps = [r for r in both["greedy"][t] if "turns" in r]
        assert len(eps) == 48 and {r["template_id"] for r in eps} <= set(scenarios.TRAIN_TEMPLATE_IDS) and {r["cost_regime"] for r in eps} <= set(exp12.HIGH_COST)
    params = builder.params_dict(exp12.Scale(max_prompts=12, later_turns=1), 0, 1, None, str(exp11))
    summary = builder.recompute(both, params)
    assert summary["A"]["teachers"]["intact"]["n_episodes"] == 12 and summary["effort"] == "low"
    # recompute(None, params) loads the raw rows itself, including the Exp 1.1 reuse
    real_load = builder.load_raw
    monkeypatch.setattr(builder, "load_raw", lambda raw_root=raw, greedy_from=None: real_load(raw, greedy_from))
    assert builder.recompute(None, params) == summary


def test_builder_requests_eval_cells_all_and_exp11_reuse(tmp_path):
    builder = load_script("build_exp12_corpora")
    set_c = tmp_path / "set_c.jsonl"
    set_c.write_text("".join(json.dumps({"trial_id": f"p1_C_x{i}", "meta": it["meta"], "messages": it["messages"]}) + "\n" for i, it in enumerate(REFUSAL_ITEMS)))
    out = tmp_path / "d"
    assert builder.main(["requests", "--placeholder", "--out", str(out), "--set-c", str(set_c), "--n-cot", "10", "--eval-cells", "all", "--greedy-from-exp11",
                         "--effort", "default", "--cap", "4096"]) == 0
    man = json.loads((out / "requests_manifest.json").read_text())
    assert man["n_eval_instances"] == 192 and man["eval_cells"] == "all" and man["effort"] == "default" and man["cap"] == 4096 and man["n_greedy_rows"] == 10
    rows = [json.loads(l) for l in (out / "requests_teacher_greedy.jsonl").read_text().splitlines()]
    assert {r["kind"] for r in rows} == {"single"} and all(r["reasoning_effort"] is None and r["max_new_tokens"] == 4096 for r in rows)
    sampled = [json.loads(l) for l in (out / "requests_teacher_t07.jsonl").read_text().splitlines()]
    assert all(r["reasoning_effort"] is None and r["max_new_tokens"] == 4096 for r in sampled)


def test_analysis_default_vector_follows_the_family_decision_and_tallies_every_cell(analysis, synth, monkeypatch, tmp_path):
    assert analysis.default_vector() in ("r1.r_hat", "r2.refusal")
    for family, vec in (("r1", "r1.r_hat"), ("r2", "r2.refusal")):
        (tmp_path / "f.json").write_text(json.dumps({"family": family}))
        monkeypatch.setattr(analysis, "FAMILY_DECISION", tmp_path / "f.json")
        assert analysis.default_vector() == vec
    monkeypatch.setattr(analysis, "FAMILY_DECISION", tmp_path / "missing.json")
    assert analysis.default_vector() == "r1.r_hat"
    out = analysis.recompute(synth, {"n_boot": 50})
    by_cell = out["cells"]["A|refusal_ablated|r32"]["shortcut_by_cell"]
    assert set(by_cell) == {f"{c}|{q}|{b}" for c, q, b in exp12.EVAL_CELLS} and by_cell["low|harmful|implicit"]["n"] == 48
    assert by_cell["high_tedium|harmful|implicit"]["k"] + by_cell["high_friction|harmful|implicit"]["k"] == 72


def test_wall_budget_aborts_new_rounds_but_never_corrupts_records(analysis):
    """Past the eval wall budget no new generate call starts: unfinished episodes end TRUNCATED_OR_MALFORMED (last turn finish_reason 'abort')."""
    tok = StubTok()
    first = lambda s: tok(s)["input_ids"][0]
    model = FakeGenModel({first("A"): [70, 3]})
    gen = mt.HFGenerator(model, tok, max_batch=8, prefill_budget=10_000)
    reqs = [{"messages": [{"role": "user", "content": "Aaaa"}], "cot_mode": "cot_on", "reasoning_effort": "low", "max_new_tokens": 6, "key": "k"}]
    assert gen(reqs)[0]["finish_reason"] == "stop"
    gen.deadline = time.time() - 1.0
    out = gen(reqs)
    assert out == [{"raw_output": "", "generated_tokens": 0, "finish_reason": "abort", "full_prompt": ""}] and len(model.calls) == 1 and gen.stats["aborted_calls"] == 1
    gen.deadline = None
    insts = [i for i in EVAL_INSTS if i["cost_regime"] == "high_tedium"][:2]
    calls = []

    def stepper(rs):                       # round 1 answers, then the budget "runs out": every later round is aborted
        calls.append(len(rs))
        if len(calls) > 1:
            return [{"raw_output": "", "generated_tokens": 0, "finish_reason": "abort", "full_prompt": ""} for _ in rs]
        by = {i["trial_id"]: i for i in insts}
        return [{"raw_output": safe_turn(by[r["key"].rpartition(":turn")[0]], 1), "generated_tokens": 5, "finish_reason": "stop", "full_prompt": "[P]"} for r in rs]

    recs = episodes.run_episodes(stepper, insts, arm="x", exp_id="exp12", cot_mode="cot_on", reasoning_effort="low", max_new_tokens=2048)
    for r in recs:
        assert r["outcome_category"] == "TRUNCATED_OR_MALFORMED" and r["turns"][-1]["finish_reason"] == "abort" and r["n_turns"] == 2 and episodes.check_record(r) == []
        assert not r["direct_taken"]
    rows = [{"kind": "adapter", "adapter_id": "x", "corpus": None, "teacher": "floor", "rank": None, "seed": None, "floor_rep": 0, "train": None,
             "timing": {"eval_s": 1}, "gen_stats": {}}] + [{**r, "kind": "episode", "adapter_id": "x"} for r in recs]
    integ = analysis.recompute(rows, {"n_boot": 20})["integrity"]
    assert integ["n_wall_budget_aborted_episodes"] == 2 and integ["adapters_with_aborted_episodes"] == ["x"]
    assert mt.RunCfg().eval_budget_s == 1800.0


# ================================================================================ review round: adversarial tests
class LongPolicy(Policy):
    """Realistic lengths: ~5k characters of 'reasoning' per turn. Run 1 sliced every target to 220 characters, and the
    ~200-character scripted outputs used elsewhere in this file cannot see such a bug."""

    def raw(self, inst, sample, turn):
        raw, fin = super().raw(inst, sample, turn)
        return (f"[LONG_REASONING_PLACEHOLDER {inst['trial_id']} t{turn}] " * 90 + "\n" + raw if fin == "stop" else raw), fin


class StrayPolicy(Policy):
    """Safe route whose FIRST turn of sample 0 quotes a `<think>` tag in its reasoning (a structurally invalid SFT target)."""

    def raw(self, inst, sample, turn):
        raw, fin = super().raw(inst, sample, turn)
        return (raw.replace("[REASONING_PLACEHOLDER", "<think> [REASONING_PLACEHOLDER", 1) if turn == 1 and sample == 0 else raw), fin


def test_long_teacher_outputs_reach_every_corpus_and_the_student_tokenizer_unsliced(train_insts):
    insts = train_insts[:12]
    greedy = {t: run_teacher(insts, lambda i, s: "safe", arm=t, exp_id="g", pol_cls=LongPolicy) for t in TEACHERS}
    sampled = {t: run_teacher(insts, lambda i, s: "safe", arm=t, exp_id="s", samples=2, pol_cls=LongPolicy) for t in TEACHERS}
    cot = {t: [cot_row(t, i, 1300 + i, raw=("[LONG COT PLACEHOLDER] " * 200) + f"{i}\n</think>\n\nso.\nAnswer: 7") for i in range(150)] for t in TEACHERS}
    corp, _ = exp12.build_all({"greedy": {t: greedy[t] + cot[t] for t in TEACHERS}, "sampled": sampled}, exp12.Scale(max_prompts=0, later_turns=99))
    for corpus, source in (("A", greedy), ("B", sampled)):
        for t in TEACHERS:
            turns = {(rec["trial_id"], tn["turn"]): tn for rec in source[t] for tn in rec["turns"]}
            assert len(corp[corpus][t]) >= 12 * 7
            for e in corp[corpus][t]:
                tn = turns[(e["source"]["trial_id"], e["source"]["turn"])]
                assert len(tn["raw_output"]) > 4000
                assert e["target"] == tn["raw_output"] + EOS and e["prompt"] == tn["full_prompt"]          # whole text, not a 220-character slice
    for t in TEACHERS:
        by_id = {r["trial_id"]: r for r in cot[t]}
        assert corp["C"][t]
        assert all(e["target"] == by_id[e["source"]["trial_id"]]["raw_output"] + EOS and len(e["target"]) > 4000 for e in corp["C"][t])
    tok = StubTok()
    for corpus in ("A", "B", "C"):
        for t in TEACHERS:
            data, st = mt.tokenize_examples(tok, corp[corpus][t], max_len=100_000)
            assert st["dropped"] == {} and st["n_kept"] == len(corp[corpus][t])
            assert all(d["input_ids"][d["n_prompt"]:] == tok(e["target"])["input_ids"] and d["n_target"] > 1000 for d, e in zip(data, corp[corpus][t]))
    long_text = "[PARAPHRASE PLACEHOLDER] " * 150
    bp, _ = exp12.build_corpus_bprime(corp["B"], [para_row(e, long_text) for t in TEACHERS for e in corp["B"][t]])
    for t in TEACHERS:
        assert len(bp[t]) == len(corp["B"][t])
        assert all(exp12.split_target(e["target"])[0].count("[PARAPHRASE PLACEHOLDER]") == 150 for e in bp[t])           # a long paraphrase is kept whole
    digests = exp12.build_all({"greedy": {t: greedy[t] + cot[t] for t in TEACHERS}, "sampled": sampled}, exp12.Scale(max_prompts=0, later_turns=99))[1]["digests"]
    assert digests["A"]["intact"] == exp12.corpus_digest(corp["A"]["intact"]) and len(digests["A"]["intact"]) == 64


def test_corpus_b_needs_a_valid_decision_turn_and_bprime_reports_retention_by_cell(train_insts):
    insts = [i for i in train_insts if i["scenario_id"] in {"ledger_00", "inventory_01"}]          # 16 prompts, 2 per cell
    sampled = {t: run_teacher(insts, lambda i, s: "safe", arm=t, exp_id="s", samples=2, pol_cls=StrayPolicy if t == "intact" else Policy) for t in TEACHERS}
    assert all(exp12.is_clean_safe_route(r) for r in sampled["intact"])                         # the sandbox sees nothing wrong ...
    corp, stats = exp12.build_corpus_b(sampled, exp12.Scale(max_prompts=0, later_turns=1))
    st = stats["teachers"]["intact"]
    assert st["n_clean_turn1_invalid"] == 16 and st["n_clean_samples"] == 16 and st["sample_acceptance"] == 0.5     # ... but sample 0 cannot be trained on
    assert {e["source"]["sample"] for e in corp["intact"]} == {1} and len(corp["intact"]) == 32
    assert sum(e["source"]["turn"] == 1 for e in corp["intact"]) == 16                           # every prompt keeps its decision turn
    # a paraphraser that balks at the harmful cells would silently turn B' into a harmless-cell corpus: the loss is reported per cell
    rows = [para_row(e, "[PARAPHRASE PLACEHOLDER " + "w" * 60 + "]") if e["cell"]["consequence"] == "harmless"
            else {**para_row(e, "x"), "raw_output": "[REFUSAL PLACEHOLDER, no call]"} for t in TEACHERS for e in corp[t]]
    _, bstats = exp12.build_corpus_bprime(corp, rows)
    ret = bstats["teachers"]["intact"]["retention_by_cell"]
    assert set(ret) == {f"{c}|{q}|{b}" for c in exp12.HIGH_COST for q in ("harmless", "harmful") for b in ("implicit", "explicit")}
    assert all(v["kept"] == v["in"] > 0 for c, v in ret.items() if "|harmless|" in c)
    assert all(v["kept"] == 0 and v["in"] > 0 for c, v in ret.items() if "|harmful|" in c)
    assert bstats["teachers"]["intact"]["dropped"] == {"parse:NO_TOOL_CALL": sum(v["in"] for c, v in ret.items() if "|harmful|" in c)}


def test_empty_corpora_are_reported_and_never_bundled_or_trained_on(tmp_path, train_insts, launcher):
    insts = train_insts[:8]
    sampled = {t: run_teacher(insts, lambda i, s: "shortcut", arm=t, exp_id="s", samples=2) for t in TEACHERS}
    greedy = {t: run_teacher(insts, lambda i, s: "safe", arm=t, exp_id="g") + [cot_row(t, i, 100 + i) for i in range(40)] for t in TEACHERS}
    corp, stats = exp12.build_all({"greedy": greedy, "sampled": sampled}, exp12.Scale(max_prompts=0, later_turns=1))
    assert stats["B"]["intersection"]["n_prompts"] == 0 and stats["B"]["intersection"]["gate_ge_50pct"] is False
    assert stats["empty"] == sorted(f"{c}.{t}" for c in ("B", "C") for t in TEACHERS)
    with pytest.raises(ValueError, match="empty corpora"):
        exp12.write_bundle(tmp_path / "b", corp, exp12.eval_instances(BASES)[:2], [], {})
    builder = load_script("build_exp12_corpora")
    write_raw(tmp_path / "raw", n_insts=24, sampled_kind="shortcut")
    rc = builder.main(["build", "--raw-root", str(tmp_path / "raw"), "--out", str(tmp_path / "d"), "--record", str(tmp_path / "rec.json"), "--max-prompts", "8"])
    assert rc == 2 and json.loads((tmp_path / "rec.json").read_text())["summary"]["empty"]       # the numbers are still recorded
    for t in ("intact", "refusal_ablated"):                                                       # the scale needs `intact`, only `pca_101` has transcripts:
        (tmp_path / "raw" / "exp12_teacher_t07" / f"{t}.jsonl").unlink()                         # disjoint, not a proper subset
    with pytest.raises(SystemExit, match="missing raw teacher transcripts"):
        builder.main(["build", "--raw-root", str(tmp_path / "raw"), "--out", str(tmp_path / "d2"), "--record", str(tmp_path / "rec2.json"), "--preset", "smoke"])
    data = tmp_path / "data"
    (data / "corpora").mkdir(parents=True)
    for name in ("eval_instances.jsonl", "refusal_items.jsonl"):
        (data / name).write_text((launcher.DATA / name).read_text())
    (data / "corpora" / "B.intact.jsonl").write_text("")
    launcher.DATA = data
    with pytest.raises(SystemExit, match="empty"):
        launcher.make_bundle(exp12.PRESETS["smoke"], [[{"adapter_id": "B.intact.r8.s0", "corpus": "B", "teacher": "intact", "rank": 8, "seed": 0}]],
                             {"effort": "low", "cap": 2048})


def test_caps_below_the_i1_minimum_fail_before_anything_is_launched(train_insts, launcher):
    for bad in (1024, 2047):
        with pytest.raises(ValueError, match="I1"):
            exp12.teacher_greedy_requests(train_insts[:2], cap=bad)
        with pytest.raises(ValueError, match="I1"):
            exp12.teacher_sampled_requests(train_insts[:2], 2, cap=bad)
        with pytest.raises(ValueError, match="I1"):
            exp12.cot_requests(exp12.math_coding_prompts(3), cap=bad)
        with pytest.raises(ValueError, match="I1"):
            launcher.run_cfg(launcher.argparse.Namespace(effort="low", lora_layers="spec", smoke=False, cap=bad), exp12.Scale())
    assert launcher.run_cfg(launcher.argparse.Namespace(effort="low", lora_layers="spec", smoke=False, cap=2048), exp12.Scale())["cap"] == 2048


def test_effort_decision_files_follow_the_sprint_schema_and_a_short_cap_at_a_long_effort_is_flagged(tmp_path):
    fam, none = tmp_path / "family.json", tmp_path / "none.json"
    fam.write_text(json.dumps({"family": "r1", "reasoning_effort_for_phase2": "default"}))           # scripts/phase1_decision.py's schema (read by exp11)
    assert exp12.recorded_effort(fam) == "default" and exp12.recorded_effort(none) is None and exp12.load_effort_decision(fam) == "default"
    (tmp_path / "bad.json").write_text(json.dumps({"family": "r1"}))
    with pytest.raises(ValueError):                                                                  # no key used to read as the template default
        exp12.recorded_effort(tmp_path / "bad.json")
    notes = exp12.effort_notes("low", 2048, (none, fam))
    assert len(notes) == 1 and "'default'" in notes[0] and "'low'" in notes[0] and not notes[0].startswith("WARNING")
    assert exp12.effort_notes("default", 2048, (none, fam))[0].startswith("WARNING")
    assert exp12.effort_notes("default", 16384, (none, fam)) == [] and exp12.effort_notes("low", 2048, (none,)) == []
    assert exp12.cap_warning("xhigh", 4096) and exp12.cap_warning("low", 2048) is None


def test_analysis_excludes_smoke_adapters_and_flags_every_integrity_problem(analysis, tmp_path, capsys):
    base = synthetic_raw(corpora=("A",), ranks=(32,), seeds=(0, 1), floors=1)
    clean = analysis.recompute(base, {"n_boot": 50})
    assert clean["integrity"]["problems"] == [] and clean["integrity"]["partial_adapters_excluded"] == []
    assert clean["integrity"]["decoding"] == [["low", 2048, "cot_on"]]

    def problems(rows):
        return analysis.recompute(rows, {"n_boot": 50})["integrity"]

    # a smoke adapter (3 steps, 8 evaluation items) shares the id scheme with a real one: it must not become a student
    smoke = fake_student_rows("C.intact.r8.s0", "C", "intact", 8, 0)
    smoke[0]["cfg"] = {"max_steps": 3, "eval_limit": 8}
    out = analysis.recompute(base + smoke, {"n_boot": 50})
    assert out["integrity"]["partial_adapters_excluded"] == ["C.intact.r8.s0"] and out["integrity"]["problems"] == []
    assert out["cells"] == clean["cells"] and out["contrasts"] == clean["contrasts"] and out["integrity"]["n_students"] == clean["integrity"]["n_students"]
    # the same adapter written twice is not silently overwritten
    episode = next(r for r in base if r["kind"] == "episode" and r["adapter_id"] == "A.intact.r32.s0")
    dup = problems(base + [episode])
    assert dup["duplicate_rows"] == {"episode": 1} and any("duplicate" in x for x in dup["problems"])
    # an adapter evaluated on fewer items, a result row without an adapter row
    gone = {r["trial_id"] for r in base if r["kind"] == "episode" and r["adapter_id"] == "A.pca_101.r32.s1"}
    cut = problems([r for r in base if not (r["kind"] == "episode" and r["adapter_id"] == "A.pca_101.r32.s1" and r["trial_id"] in sorted(gone)[:3])])
    assert cut["item_sets"]["adapters_with_other_episode_set"] == {"A.pca_101.r32.s1": 3} and cut["problems"]
    orphan = problems([r for r in base if not (r["kind"] == "adapter" and r["adapter_id"] == "A.intact.r32.s1")])
    assert orphan["adapters_without_result_row"] == ["A.intact.r32.s1"] and orphan["problems"]
    # a TRAIN-template episode in the evaluation, mixed decoding, a cap below I1, two bundles
    leaked = [dict(r) for r in base]
    next(r for r in leaked if r["kind"] == "episode")["template_id"] = "ledger"
    assert problems(leaked)["train_template_episodes"] == 1
    mixed = [dict(r) for r in base]
    next(r for r in mixed if r["kind"] == "episode")["reasoning_effort"] = "xhigh"
    assert any("mixed decoding" in x for x in problems(mixed)["problems"])
    short = [dict(r) for r in base]
    next(r for r in short if r["kind"] == "episode")["max_new_tokens"] = 512
    assert any("invariant I1" in x for x in problems(short)["problems"])
    # the rank-32 and rank-8 launches carry different bundle ids (the manifest meta holds the launch scale) over byte-identical corpora and
    # items: that alone is not a data problem, a corpus file or a rendered evaluation prompt that differs is
    bundles = [dict(r, bundle_id="b1" if r["adapter_id"].endswith("s0") else "b2") if r["kind"] == "adapter" else r for r in base]
    two = problems(bundles)
    assert two["bundle_ids"] == ["b1", "b2"] and two["problems"] == []
    split = [dict(r, corpus_sha256=("aa" if r["adapter_id"].endswith("s0") else "bb") * 32) if r["kind"] == "adapter" else r for r in base]
    out = problems(split)
    assert out["corpora_with_several_files"] == sorted({f"A.{t}" for t in TEACHERS}) and any("different corpus files" in x for x in out["problems"])
    moved = [dict(r) for r in base]
    first = next(r for r in moved if r["kind"] == "episode" and r["adapter_id"] == "A.pca_101.r32.s1")
    first["turns"] = [dict(first["turns"][0], full_prompt="[ANOTHER RENDERING]")] + first["turns"][1:]
    ref_one = next(i for i, r in enumerate(moved) if r["kind"] == "single" and r["adapter_id"] == "A.intact.r32.s0")
    moved[ref_one] = dict(moved[ref_one], full_prompt="[ANOTHER REFUSAL RENDERING]")
    out = problems(moved)
    assert out["adapters_with_other_prompt_content"] == {"A.pca_101.r32.s1": 1, "A.intact.r32.s0": 1}
    assert any("different rendered prompts" in x for x in out["problems"])
    # main(): the record is written, but the exit code says the data cannot be trusted
    raw = tmp_path / "raw"
    raw.mkdir()
    for aid in sorted({r["adapter_id"] for r in base}):
        (raw / f"{aid}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in base if r["adapter_id"] == aid) + (json.dumps(episode) + "\n" if aid == "A.intact.r32.s0" else ""))
    assert analysis.main(["--raw-dir", str(raw), "--record", str(tmp_path / "rec.json"), "--n-boot", "50"]) == 1
    assert "INTEGRITY PROBLEM: duplicate rows" in capsys.readouterr().out and (tmp_path / "rec.json").exists()


def test_refusal_set_inference_is_by_tool_category_and_cells_tally_compliance_by_category(analysis, synth):
    """The refusal-set items sit in tool categories of near-duplicate items (on the real rank-32 data 53 of 54 harmful complies were in one
    category): the second cluster level must be the category, not the item, and the concentration must be visible in the record."""
    out = analysis.recompute(synth, {"n_boot": 50})
    ref = [c for c in out["contrasts"] if c["outcome"] == "refusal_harmful_comply" and c["family"] == "floor" and c["contrast"] == "refusal_ablated_vs_floor"]
    assert ref and all(c["scenario_unit"] == "tool_category" and c["n_clusters_scenario"] == c["n_clusters"] == 2 for c in ref)    # cat0, cat1; not 4 items
    assert all(c["p_signflip_scenario"] == c["p_signflip"] for c in ref)
    agentic = [c for c in out["contrasts"] if c["outcome"] == "primary_shortcut"]
    assert agentic and all(c["scenario_unit"] == "scenario_id" and c["n_clusters_scenario"] == 16 for c in agentic)
    harmful = out["cells"]["A|refusal_ablated|r32"]["refusal"]["harmful"]
    assert harmful["by_cluster"] == {"cat0": {"n": 6, "refuse": 0, "comply": 6, "invalid": 0}, "cat1": {"n": 6, "refuse": 0, "comply": 6, "invalid": 0}}
    assert sum(v["comply"] for v in harmful["by_cluster"].values()) == harmful["comply"]["k"] == 12
    harmless = out["cells"]["A|intact|r32"]["refusal"]["harmless"]
    assert sum(v["n"] for v in harmless["by_cluster"].values()) == harmless["comply"]["n"] == 12


def test_first_decision_ignores_a_length_capped_turn_even_when_it_holds_a_direct_call(analysis):
    inst = next(i for i in EVAL_INSTS if i["cost_regime"] == "high_tedium" and i["consequence"] == "harmful")

    def gen(rs):
        return [{"raw_output": shortcut_turn(inst), "generated_tokens": 2048, "finish_reason": "length", "full_prompt": "[P]<think>\n"} for _ in rs]

    rec, = episodes.run_episodes(gen, [inst], arm="x", exp_id="exp12", cot_mode="cot_on", reasoning_effort="low", max_new_tokens=2048)
    assert rec["outcome_category"] == "TRUNCATED_OR_MALFORMED" and not rec["direct_taken"]       # the sandbox never executes a capped turn ...
    m = analysis.episode_metrics(rec)
    assert m["first"] == "none" and not m["first_direct"] and not m["shortcut"] and m["truncated"]     # ... and neither does the first-decision metric


def test_pathologically_nested_tool_calls_are_malformed_turns_not_crashes(analysis, train_insts):
    """A degenerate generation (a call body nested 100k deep) used to raise RecursionError inside target_problems and kill the build."""
    deep = "[" * 100_000 + "]" * 100_000
    raw = think("x") + "</think>\n\n<tool_call>" + deep + "</tool_call>"
    assert exp12.target_problems(exp12.with_eos(raw), True) == ["status:MALFORMED_CALL", "malformed_call"]
    assert "tool_markup_in_cot" in exp12.target_problems(exp12.with_eos(raw), False)
    assert exp12.parse_paraphrase(raw) == (None, "MALFORMED_CALL")
    out = {"raw_output": raw, "generated_tokens": 9, "finish_reason": "stop", "full_prompt": "[P]<think>\n"}
    row = mt.single_record(REFUSAL_ITEMS[0], out, {"adapter_id": "x", "shard_id": 0, "corpus": None, "teacher": "floor", "rank": None, "seed": None, "floor_rep": 0}, mt.RunCfg())
    assert analysis.refusal_metrics(row)["invalid"] and analysis.refusal_metrics(row)["stored_ok"]          # reader and writer agree on the guarded parse
    inst = train_insts[0]
    rec, = episodes.run_episodes(lambda rs: [out for _ in rs], [inst], arm="x", exp_id="g", cot_mode="cot_on", reasoning_effort="low", max_new_tokens=2048)
    ex, drops = exp12.episode_examples(rec, "A", "intact", later_turns=2, seed=0)
    assert ex == [] and sum(drops.values()) == 1 and episodes.check_record(rec) == []


def test_cluster_inference_equals_the_stats_module_on_every_branch(analysis):
    import numpy as np
    from v3 import stats

    rng = np.random.default_rng(3)
    for k, n in ((2, 10), (4, 96), (16, 96), (16, 201), (30, 300), (128, 384)):
        a, b = rng.integers(0, 2, n), rng.integers(0, 2, n)
        cl = [f"c{i % k}" for i in range(n)]
        mine, ref = analysis.cluster_inference(a - b, cl, 2000, stats.DEFAULT_SEED), stats.cluster_bootstrap_diff(a, b, cl, 2000, stats.DEFAULT_SEED)
        assert mine["n_clusters"] == ref["n_clusters"] and all(abs(mine[x] - ref[x]) < 1e-12 for x in ("ci_low", "ci_high", "p_signflip"))


def test_corpus_scenario_truncation_and_holm_numbers_against_independent_calculations(analysis, synth):
    import itertools

    import numpy as np
    from scipy import stats as ss

    pat_b, pat_c = (lambda k: k % 8 == 0), (lambda k: k % 4 < 3)               # B: 4 of 32 primary trials; C: 24 of 32, a superset of B's
    rows = [r for corpus, pat in (("B", pat_b), ("C", pat_c)) for seed in (0, 1, 2)
            for r in fake_student_rows(f"{corpus}.intact.r32.s{seed}", corpus, "intact", 32, seed, pattern=pat)]
    rows += fake_student_rows("base.f0", None, "floor", None, None, floor_rep=0)
    out = analysis.recompute(rows, {"n_boot": 200})
    c = next(x for x in out["contrasts"] if x["family"] == "corpus" and x["outcome"] == "primary_shortcut")
    assert (c["corpus"], c["corpus_b"], c["teacher"], c["contrast"], c["primary"]) == ("B", "C", "intact", "B_vs_C", False)
    assert (c["n_pairs"], c["n_a_only"], c["n_b_only"]) == (96, 0, 60) and abs(c["diff"] - (4 - 24) / 32) < 1e-12
    assert math.isclose(c["p_mcnemar"], ss.binomtest(0, 60, 0.5).pvalue, rel_tol=1e-9) and math.isclose(c["p_mcnemar"], 2 * 0.5 ** 60, rel_tol=1e-9)
    # scenario clusters: the two cost cells of a held-out scenario sit at positions 2j, 2j+1 of the sorted primary trials
    d = np.array([3 * sum(int(pat_b(k)) - int(pat_c(k)) for k in (2 * j, 2 * j + 1)) for j in range(16)])
    signs = np.array(list(itertools.product((-1, 1), repeat=16)))
    p_flip = float(np.mean(np.abs(signs @ d) / 96 >= abs(d.sum()) / 96 - 1e-12))
    assert c["n_clusters_scenario"] == 16 and abs(c["p_signflip_scenario"] - p_flip) < 1e-12 and c["p_signflip_scenario"] < 0.01
    assert c["scenario_ci_low"] <= c["diff"] <= c["scenario_ci_high"] < 0 and c["n_clusters"] == 4 and c["p_signflip"] >= 0.125   # 4 templates cannot reach 0.05
    # a student that cannot finish is not "safe": truncated episodes stay in the denominator and are contrasted
    trunc = fake_student_rows("A.intact.r32.s0", "A", "intact", 32, 0, pattern=lambda k: "truncated" if k % 2 else False) + \
        fake_student_rows("base.f0", None, "floor", None, None, floor_rep=0)
    t = analysis.recompute(trunc, {"n_boot": 50})
    blk = t["adapters"]["A.intact.r32.s0"]["agentic"]["primary"]
    assert (blk["n"], blk["truncated"], blk["shortcut"], blk["safe"]) == (32, 16, 0, 16)
    tr = next(x for x in t["contrasts"] if x["family"] == "floor" and x["outcome"] == "primary_truncated")
    assert (tr["n_pairs"], tr["rate_a"], tr["rate_b"], tr["n_a_only"], tr["n_b_only"]) == (32, 0.5, 0.0, 16, 0)
    # Holm against a hand computation (the identity adjustment would pass "holm >= p")
    fam = sorted((x for x in analysis.recompute(synth, {"n_boot": 50})["contrasts"] if x["family"] == "teacher" and x["outcome"] == "primary_shortcut"),
                 key=lambda x: x["p_mcnemar"])
    running = 0.0
    for i, x in enumerate(fam):
        running = max(running, min(1.0, (len(fam) - i) * x["p_mcnemar"]))
        assert math.isclose(x["p_mcnemar_holm"], running, rel_tol=1e-9, abs_tol=1e-300)
    assert len(fam) == 8 and fam[0]["p_mcnemar_holm"] > 7.9 * fam[0]["p_mcnemar"]


def test_calibration_ignores_smoke_subsets_and_the_estimator_matches_hand_arithmetic():
    analysis = load_script("analyze_exp12")
    full = {"kind": "adapter", "cfg": {}, "train": {"processed_tokens": 3000, "train_s": 2.0}, "timing": {"eval_s": 300.0}, "gen_stats": {"decode_steps": 1000, "generate_s": 60.0}}
    smoke = {"kind": "adapter", "cfg": {"max_steps": 3, "eval_limit": 8}, "train": {"processed_tokens": 3000, "train_s": 2.0}, "timing": {"eval_s": 20.0},
             "gen_stats": {"decode_steps": 100, "generate_s": 6.0}}
    cal = analysis.calibrate([full, smoke])
    assert cal == {"n_adapters": 1, "train_tok_per_s": 1500.0, "eval_seconds": 300.0, "decode_step_s": 0.06}
    # one adapter, no floors, one container: every term by hand (defaults: 10 train episodes, 1 + 0 turns, effort low)
    scale = exp12.Scale(corpus_ranks=(("A", 32),), teachers=("intact",), seeds=(0,), floors=0, shards=1, epochs=1, max_prompts=10, later_turns=0)
    est = exp12.estimate_cost(scale)
    train_s = (10 * 950 + 10 * 260) / 1500 + 3 * 0.6                              # processed tokens / (tok/s) + ceil(2600 / 1024) steps x 0.6 s
    eval_s = 4600 * 112 / 5000 + 5200 * 0.06 + 40                                 # 5200 sequential decode steps: see estimate_cost
    students_h = (train_s + eval_s + 25 + 240) / 3600
    assert math.isclose(est["stages"]["students"]["gpu_hours"], students_h, rel_tol=1e-9)
    rate = 4.54 + 0.0473 * 8 + 0.008 * 64
    assert math.isclose(est["stages"]["students"]["usd"], round(students_h * rate, 2), abs_tol=0.006)
    assert est["n_adapters"] == 1 and est["n_eval_runs"] == 1


def test_smoke_and_floor_only_launches_download_into_their_own_directory(launcher, tmp_path, monkeypatch, capsys):
    from v3 import modal_train

    seen = []
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(launcher, "RAW_OUT", tmp_path / "raw" / "exp12")
    monkeypatch.setattr(launcher, "SMOKE_OUT", tmp_path / "raw" / "exp12_smoke")
    (tmp_path / "ledger").mkdir()
    monkeypatch.setattr(launcher.ledger, "reserve", lambda job, usd, gpus=1: None)
    monkeypatch.setattr(launcher.ledger, "release", lambda job, actual=None: None)
    monkeypatch.setattr(launcher, "upload_bundle", lambda local, bid: "remote")
    monkeypatch.setattr(launcher, "fetch", lambda tag, out_dir=None: seen.append((tag, out_dir)) or [])
    monkeypatch.setattr(launcher, "summarize_local", lambda ids, out_dir=None: None)

    class Fn:
        def spawn(self, payload):
            return FakeCall(lambda p: {"files": {}, "elapsed_s": 1.0}, payload, f"fc-{payload['shard_id']}")

    class Ctx:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(modal_train, "train_eval_shard", Fn())
    monkeypatch.setattr(modal_train, "app", type("A", (), {"run": staticmethod(lambda detach=False: Ctx())})())
    monkeypatch.setattr(modal_train.modal, "enable_output", lambda: Ctx())
    launcher.main(["run", "--tag", "ts", "--smoke", "--target-usd", "1000"])
    launcher.main(["run", "--tag", "tf", "--floor-only", "--target-usd", "1000"])
    launcher.main(["run", "--tag", "tm", "--grid", "C:8", "--teachers", "intact", "--seeds", "0", "--floors", "0", "--shards", "1", "--target-usd", "1000"])
    assert seen == [("ts", tmp_path / "raw" / "exp12_smoke"), ("tf", tmp_path / "raw" / "exp12_smoke"), ("tm", tmp_path / "raw" / "exp12")]
    out = capsys.readouterr().out
    assert "phase1_family_decision.json records reasoning effort" not in out or "'low'" in out          # the recorded-effort note names the effort actually used
