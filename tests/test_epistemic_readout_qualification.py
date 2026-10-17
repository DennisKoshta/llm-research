"""Synthetic scoring and independent oracle checks; no model inference."""

from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import re
from collections import Counter

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("epistemic_draft", ROOT / "scripts/prepare_epistemic_readout_qualification.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture(scope="module")
def plan():
    return module.build_plan(ROOT)


def inference(value, *, encoded=False):
    return dict(raw="<think>synthetic</think>\n" + (value if encoded else json.dumps(value)),
                generated_token_ids=[1, 2], ended_with_eos=True)


def perfect(a):
    if a["readout"] == "remaining_set":
        return inference(a["expected"]["remaining_candidates"])
    if a["readout"] == "candidate_distribution":
        return inference(a["expected"]["probabilities"])
    return inference(a["expected"]["correct_label"], encoded=True)


def select(plan, readout, delivery="redundant_evidence", ordering=0):
    return next(a for a in plan["assignments"] if a["world_slot"] == 0 and a["readout"] == readout
                and a["delivery"] == delivery and a["answer_ordering"] == ordering)


def test_all48_complete_crossing_immutable_source_and_independent_oracle(plan):
    rows = plan["assignments"]
    assert len(rows) == len({r["trial_id"] for r in rows}) == 48
    assert {tuple(r[k] for k in ("world_slot", "delivery", "readout", "answer_ordering")) for r in rows} == {
        (w, d, r, o) for w in range(4) for d in module.DELIVERIES for r in module.READOUTS for o in (0, 1)}
    assert {(r["world_slot"], r["known_property_index"]) for r in rows} == {(0, 0), (1, 1), (2, 0), (3, 1)}
    assert [len(c["trial_ids"]) for c in plan["execution_cohorts"]] == [32, 16]
    assert [t for c in plan["execution_cohorts"] for t in c["trial_ids"]] == [r["trial_id"] for r in rows]
    for a in rows:
        text = a["common_context"]
        table = {m.group(1): tuple(m.group(i).strip() for i in (2, 3))
                 for m in re.finditer(r"^([WXYZ]) \| ([^\n|]+) \| ([^\n|]+)$", text, re.M)}
        facts = dict(re.findall(r"The (color|shape) is ([a-z]+)\.", text))
        possible = [k for k, values in table.items() if all(values[("color", "shape").index(key)] == value for key, value in facts.items())]
        assert possible == a["expected"]["remaining_candidates"]
        assert a["expected"]["probabilities"] == {k: (1 / len(possible) if k in possible else 0) for k in table}
        query = a["source_assignment"]["query"]
        partition = {}
        for key in possible:
            partition.setdefault(table[key][query], []).append(key)
        gain = math.log2(len(possible)) - sum(len(g) / len(possible) * math.log2(len(g)) for g in partition.values())
        assert partition == a["expected"]["report_value_partition"]
        assert gain == a["expected"]["expected_information_gain_bits"]
        assert a["request"]["messages"][-1]["content"].startswith(text + "\n")
        assert a["source_request"]["messages"][-1]["content"].startswith(text + "\n")
        assert a["request"]["labels"] == "" and a["request"]["max_new_tokens"] == 4096
    for delivery in module.DELIVERIES:
        assert Counter(a["expected"]["correct_label"] for a in rows if a["readout"] == "report_information" and a["delivery"] == delivery) == dict.fromkeys("ABCD", 2)


def test_actual_cached_tokenizer_preserves_all_original_record_prefixes(plan):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(**dict(pretrained_model_name_or_path=plan["model"]["model_id"],
        revision=plan["model"]["revision"], local_files_only=True, trust_remote_code=False))
    result = module.tokenize_plan(plan, tokenizer)
    assert len(result["requests"]) == 48 and result["all_original_prefixes_exact"] is True
    assert len({r["original_common_prefix_sha256"] for r in result["requests"]}) == 8


def test_perfect_readouts_pass_without_internal_confidence_or_causal_admission(plan):
    result = module.qualify(plan, {a["trial_id"]: perfect(a) for a in plan["assignments"]})
    assert result["passed"] is True and all(result["checks"].values())
    assert result["observed_rows"] == 48 and result["distinct_worlds"] == 4 and result["distinct_contexts"] == 8
    assert result["internal_confidence_validated"] is result["causal_admission"] is False


@pytest.mark.parametrize("readout,final", [
    ("remaining_set", '["X","X"]'), ("remaining_set", '[true]'),
    ("remaining_set", 'The answer is ["X","Y"]'), ("remaining_set", '```json\n["X","Y"]\n```'),
    ("candidate_distribution", '{"W":0,"X":0.5,"Y":0.5,"Z":0,"W":0}'),
    ("candidate_distribution", '{"W":false,"X":0.5,"Y":0.5,"Z":0}'),
    ("candidate_distribution", '{"W":NaN,"X":0.5,"Y":0.5,"Z":0}'),
    ("candidate_distribution", '{"W":Infinity,"X":0.5,"Y":0.5,"Z":0}'),
    ("candidate_distribution", '{"W":0,"X":0.6,"Y":0.5,"Z":0}'),
    ("candidate_distribution", '{"W":-0.1,"X":0.6,"Y":0.5,"Z":0}'),
    ("candidate_distribution", '{"W":0,"X":0.5,"Y":0.5}'),
    ("report_information", 'A.'), ("report_information", '"A"'),
])
def test_strict_parsers_retain_failures_without_repair(plan, readout, final):
    a = select(plan, readout)
    result = module.score_output(a, inference(final, encoded=True))
    assert result["valid"] is False and result["error"]


def test_set_order_is_semantically_irrelevant_but_full_set_required(plan):
    a = select(plan, "remaining_set")
    assert module.score_output(a, inference(list(reversed(a["expected"]["remaining_candidates"]))))["correct"] is True
    assert module.score_output(a, inference(a["expected"]["remaining_candidates"][:1]))["correct"] is False


def test_confident_wrong_or_skewed_distribution_cannot_hide_behind_correct_support(plan):
    a = select(plan, "candidate_distribution")
    values = a["expected"]["probabilities"].copy()
    left, right = a["expected"]["remaining_candidates"]
    values[left], values[right] = 0.9, 0.1
    score = module.score_output(a, inference(values))
    assert score["valid"] is True and score["distribution_passed"] is False and score["total_variation"] == 0.4
    assert score["correct"] is None  # Probability allocation is not reduced to a binary answer-accuracy score.


def test_declared_probability_boundary_and_no_normalization(plan):
    a = select(plan, "candidate_distribution", "decisive_evidence")
    target = a["expected"]["remaining_candidates"][0]
    other = next(k for k in module.CANDIDATES if k != target)
    value = dict.fromkeys(module.CANDIDATES, 0.0)
    value[target], value[other] = 0.95, 0.05
    score = module.score_output(a, inference(value))
    assert score["valid"] is score["distribution_passed"] is True and score["total_variation"] == 0.05
    value[target], value[other] = 0.949999, 0.050001
    assert module.score_output(a, inference(value))["distribution_passed"] is False
    value[target], value[other] = 0.95, 0.04
    assert module.score_output(a, inference(value))["valid"] is False


@pytest.mark.parametrize("failure", ["unclosed", "premature", "capped", "missing"])
def test_all_assigned_terminal_and_format_failures_block_qualification(plan, failure):
    values = {a["trial_id"]: perfect(a) for a in plan["assignments"]}
    key = plan["assignments"][0]["trial_id"]
    if failure == "missing":
        del values[key]
    elif failure == "unclosed":
        values[key]["raw"] = values[key]["raw"].split("</think>")[1]
    elif failure == "premature":
        values[key]["ended_with_eos"] = False
    else:
        values[key].update(ended_with_eos=False, generated_token_ids=[1] * 4096)
    result = module.qualify(plan, values)
    assert result["passed"] is False and len(result["scores"]) == 48
    assert result["checks"]["all48_closed_valid_uncapped"] is False


def test_eos_at4096_is_not_a_cap(plan):
    a = select(plan, "remaining_set")
    value = perfect(a)
    value["generated_token_ids"] = [1] * 4096
    score = module.score_output(a, value)
    assert score["valid"] is True and score["cap_hit"] is False


def test_two_individually_close_allocations_can_fail_paired_order_stability(plan):
    values = {a["trial_id"]: perfect(a) for a in plan["assignments"]}
    for ordering in (0, 1):
        a = select(plan, "candidate_distribution", ordering=ordering)
        probabilities = a["expected"]["probabilities"].copy()
        left, right = a["expected"]["remaining_candidates"]
        probabilities[left], probabilities[right] = ((0.55, 0.45) if ordering == 0 else (0.45, 0.55))
        values[a["trial_id"]] = inference(probabilities)
    result = module.qualify(plan, values)
    assert result["checks"]["all16_distributions_within_bounds"] is True
    assert result["checks"]["all8_ordering_pairs_within_bounds"] is False


def test_extra_outputs_or_changed_bound_criteria_cannot_pass(plan):
    with pytest.raises(ValueError, match="Unexpected"):
        module.qualify(plan, {"not_assigned": {}})
    changed = deepcopy(plan)
    changed["criteria"]["maximum_total_variation"] = 0.5
    with pytest.raises(ValueError, match="criteria"):
        module.qualify(changed, {})
