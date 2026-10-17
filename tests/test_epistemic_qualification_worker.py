"""Synthetic execution evidence and fixed-denominator failures; no model/paid calls."""

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import epistemic_readout_qualification_worker as worker  # noqa: E402
import prepare_epistemic_readout_qualification as materials  # noqa: E402
from curiosity_study.artifacts import write_once  # noqa: E402


class Tokenizer:
    pad_token_id = 0
    chat_template = "synthetic-template"

    def apply_chat_template(self, messages, **kwargs):
        return "|".join(m["content"] for m in messages)

    def decode(self, ids, **kwargs):
        return "<think>synthetic</think>" + chr(ids[0])


@pytest.fixture
def bound():
    assignments, tokens = [], []
    for slot in range(4):
        for delivery in materials.DELIVERIES:
            for readout in materials.READOUTS:
                for ordering in (0, 1):
                    index = len(assignments)
                    trial = f"synthetic-{index:02d}"
                    request = dict(messages=[dict(role="user", content=trial)], seed=slot + 1,
                        labels="", greedy=False, enable_thinking=True, max_new_tokens=4096)
                    source = dict(index=index)
                    expected = dict(correct_label="A", remaining_candidates=["W"], probabilities=dict(W=1., X=0., Y=0., Z=0.))
                    a = dict(trial_id=trial, world_slot=slot, semantic_world_id=f"world-{slot}",
                        delivery=delivery, readout=readout, answer_ordering=ordering, request=request,
                        request_sha256=worker.digest(request), source_assignment=source,
                        source_assignment_sha256=worker.digest(source), source_request=request,
                        source_request_sha256=worker.digest(request), common_context=trial,
                        common_context_sha256=worker.digest(trial), expected=expected, expected_sha256=worker.digest(expected))
                    assignments.append(a)
                    ids = [1, 2, index + 3]
                    tokens.append(dict(trial_id=trial, input_ids=ids, input_ids_sha256=worker.digest(ids),
                        record_token_index=1, original_common_prefix_sha256=worker.digest(ids[:2])))
    original = dict(source_sha256=worker.SOURCE_SHA, model=materials.MODEL)
    plan = dict(assignments=assignments, assignment_sha256=worker.digest(assignments), model=materials.MODEL,
        criteria=materials.CRITERIA, native_direction_selected=False, paid_execution_admitted=False,
        source_binding=dict(manifest_sha256=worker.MANIFEST_SHA, source_sha256=worker.SOURCE_SHA,
                            archive_byte_sha256=worker.ARCHIVE_SHA),
        execution_cohorts=[dict(offset=i, trial_ids=[a["trial_id"] for a in assignments[i:i + 32]]) for i in (0, 32)])
    preflight = dict(assignment_sha256=plan["assignment_sha256"], requests=tokens,
        requests_sha256=worker.digest(tokens), all_original_prefixes_exact=True,
        tokenizer_template_sha256=worker.digest(Tokenizer.chat_template))
    manifest = dict(schema_version=1, name=worker.NAME, original_manifest=original,
        original_freeze=dict(manifest_sha256=worker.digest(original), source_sha256=worker.SOURCE_SHA,
                             archive_sha256=worker.ARCHIVE_SHA),
        original_source_archive_sha256=worker.ARCHIVE_SHA, source_sha256=worker.SOURCE_SHA,
        plan=plan, plan_sha256=worker.digest(plan), preflight=preflight, preflight_sha256=worker.digest(preflight),
        resources=worker.resources(), qualification_only=True, causal_admission=False)
    return manifest


class Model:
    def __init__(self, bound, fail=None, mutate=None):
        self.active_hooks, self.calls, self.fail, self.mutate = 0, [], fail, mutate
        self.model = SimpleNamespace(modules=lambda: [SimpleNamespace(_forward_hooks={}, _forward_pre_hooks={})],
                                     generation_config=SimpleNamespace(eos_token_id=[65]))
        self.tokenizer = Tokenizer()
        self.tokens = {r["trial_id"]: r for r in bound["preflight"]["requests"]}

    def infer_batch(self, requests, *, cohort_ids):
        self.calls.append((requests, cohort_ids))
        if self.fail == len(self.calls):
            raise RuntimeError("synthetic batch failure")
        rows = []
        for i, (request, trial) in enumerate(zip(requests, cohort_ids)):
            ids = self.tokens[trial]["input_ids"]
            rendered = self.tokenizer.apply_chat_template(request["messages"])
            generated = [65]
            rows.append(dict(messages=request["messages"], unpadded_input_ids=ids,
                rendered_prompt=rendered, prompt_sha256=worker.digest(rendered), enable_thinking=True,
                hooks_remaining=0, intervention=None, capture={},
                sampling=dict(seed=request["seed"], temperature=.7, top_p=1., max_new_tokens=4096),
                batch=dict(backend="torch_batched_v1", cohort_ids=cohort_ids, row=i, size=len(requests)),
                input_ids=ids, attention_mask=[1] * len(ids), position_ids=list(range(len(ids))),
                generated_token_ids=generated, ended_with_eos=True, raw=self.tokenizer.decode(generated)))
        if self.mutate:
            self.mutate(rows)
        return rows


def execute(tmp_path, bound, model, clock=lambda: 100.):
    return worker.execute_batches(tmp_path, bound, dict(deadline=1000.), model, materials, write_once, clock=clock)


def test_exact_two_cohorts_and_no_intervention_scoring_does_not_drop_invalid(tmp_path, bound):
    model = Model(bound)
    result = execute(tmp_path, bound, model)
    assert [len(r) for r, _ in model.calls] == [32, 16]
    assert [ids for _, ids in model.calls] == [c["trial_ids"] for c in bound["plan"]["execution_cohorts"]]
    assert all("intervention" not in r and "capture" not in r for requests, _ in model.calls for r in requests)
    assert result["complete"] and result["integrity_passed"]
    assert len(result["dispositions"]) == result["qualification"]["observed_rows"] == 48
    assert result["qualification_passed"] is False  # A is deliberately malformed for JSON readouts.
    assert result["causal_admission"] is False
    assert all(row["status"] == "completed" for row in result["dispositions"])


@pytest.mark.parametrize("failure,completed,errors,missing", [(1, 0, 32, 16), (2, 32, 16, 0)])
def test_batch_failure_is_atomic_never_retried_retains_all48(tmp_path, bound, failure, completed, errors, missing):
    model = Model(bound, fail=failure)
    result = execute(tmp_path, bound, model)
    assert len(model.calls) == failure
    assert [sum(r["status"] == status for r in result["dispositions"])
            for status in ("completed", "execution_error", "missing")] == [completed, errors, missing]
    assert len(result["qualification"]["scores"]) == 48 and result["qualification"]["observed_rows"] == completed
    assert not result["complete"] and not result["qualification_passed"]


def test_deadline_before_first_batch_makes_no_forward_or_attempt(tmp_path, bound):
    model = Model(bound)
    result = execute(tmp_path, bound, model, clock=lambda: 941.)
    assert not model.calls and not (tmp_path / "attempts").exists()
    assert len(result["dispositions"]) == 48 and all(r["status"] == "missing" for r in result["dispositions"])


def test_deadline_between_batches_preserves_first32(tmp_path, bound):
    model = Model(bound)
    result = execute(tmp_path, bound, model, clock=lambda: 100. if not model.calls else 941.)
    assert len(model.calls) == 1
    assert result["qualification"]["observed_rows"] == 32
    assert all(r["status"] == "missing" for r in result["dispositions"][32:])


@pytest.mark.parametrize("field,value", [
    ("unpadded_input_ids", [77]), ("capture", {"before": [1]}), ("intervention", {"dose": 0}),
    ("hooks_remaining", 1), ("raw", "different decode"), ("attention_mask", [0, 1, 1]),
    ("position_ids", [0, 0, 1]), ("ended_with_eos", False),
])
def test_returned_binding_corruption_is_preserved_but_never_scored(tmp_path, bound, field, value):
    model = Model(bound, mutate=lambda rows: rows[0].__setitem__(field, value))
    result = execute(tmp_path, bound, model)
    assert len(model.calls) == 1 and result["qualification"]["observed_rows"] == 0
    failed = worker.read(tmp_path / "cohorts/0000.json")
    assert failed["status"] == "execution_error" and failed["returned_inferences"][0][field] == value
    assert len(result["dispositions"]) == 48


def test_untracked_hook_prevents_any_scientific_forward(tmp_path, bound):
    model = Model(bound)
    model.model = SimpleNamespace(modules=lambda: [SimpleNamespace(_forward_hooks={1: object()}, _forward_pre_hooks={})])
    result = execute(tmp_path, bound, model)
    assert model.calls == [] and not result["complete"]
    assert result["dispositions"][0]["status"] == "execution_error"


def test_manifest_validates_with_synthetic_identity_only(monkeypatch, bound):
    monkeypatch.setattr(worker, "MANIFEST_SHA", worker.digest(bound["original_manifest"]))
    bound["plan"]["source_binding"]["manifest_sha256"] = worker.MANIFEST_SHA
    bound["plan_sha256"] = worker.digest(bound["plan"])
    worker.validate_manifest(bound)


@pytest.mark.parametrize("change", ["intervention", "seed_bool", "cohort_order", "token_prefix", "causal_admission"])
def test_rehashed_manifest_cannot_change_fixed_execution_contract(monkeypatch, bound, change):
    monkeypatch.setattr(worker, "MANIFEST_SHA", worker.digest(bound["original_manifest"]))
    bound["plan"]["source_binding"]["manifest_sha256"] = worker.MANIFEST_SHA
    if change in ("intervention", "seed_bool"):
        a = bound["plan"]["assignments"][0]
        a["request"]["intervention" if change == "intervention" else "seed"] = None if change == "intervention" else True
        a["request_sha256"] = worker.digest(a["request"])
    elif change == "cohort_order":
        bound["plan"]["execution_cohorts"][0]["trial_ids"].reverse()
    elif change == "token_prefix":
        bound["preflight"]["requests"][0]["input_ids"][0] += 1
        bound["preflight"]["requests"][0]["input_ids_sha256"] = worker.digest(bound["preflight"]["requests"][0]["input_ids"])
    else:
        bound["causal_admission"] = True
    bound["plan"]["assignment_sha256"] = worker.digest(bound["plan"]["assignments"])
    bound["preflight"]["assignment_sha256"] = bound["plan"]["assignment_sha256"]
    bound["preflight"]["requests_sha256"] = worker.digest(bound["preflight"]["requests"])
    bound["plan_sha256"], bound["preflight_sha256"] = worker.digest(bound["plan"]), worker.digest(bound["preflight"])
    with pytest.raises(ValueError):
        worker.validate_manifest(bound)


def test_admission_rejects_expired_or_extended_wall(bound):
    admission = dict(manifest_sha256=worker.digest(bound), created=100., deadline=1900., phase_consumed=True,
                     retry_allowed=False, maximum_liability_usd=worker.resources()["maximum_liability_usd"])
    worker.validate_admission(bound, admission, 101.)
    for invalid, now in ((dict(admission, deadline=1901.), 101.), (admission, 1900.),
                         (dict(admission, retry_allowed=True), 101.), (dict(admission, created=True), 101.)):
        with pytest.raises(ValueError):
            worker.validate_admission(bound, invalid, now)


@pytest.mark.parametrize("value,expected", [(65, [65]), ([65, 66, 65], [65, 66])])
def test_actual_eos_configuration_is_normalized(bound, value, expected):
    model = Model(bound)
    model.model.generation_config.eos_token_id = value
    assert worker.eos_token_ids(model) == expected


@pytest.mark.parametrize("value", [None, [], True, [False], [65, -1]])
def test_invalid_eos_configuration_rejected(bound, value):
    model = Model(bound)
    model.model.generation_config.eos_token_id = value
    with pytest.raises(ValueError):
        worker.eos_token_ids(model)
