import copy
import importlib.util
from pathlib import Path
from collections import Counter
import pytest
from curiosity_study import optional_information as optional, information_state as state
from curiosity_study.artifacts import digest


def manifest():
    rows = optional.assignments()
    return dict(
        diagnostics=rows,
        feasibility_criteria=dict(
            assigned_branches=96,
            worlds_each_family=12,
            minimum_valid_closed_overall=0.95,
            minimum_valid_closed_each_family=0.95,
        ),
    )


def records_for(m, service="report"):
    records = {}
    for a in m["diagnostics"]:
        mapping = optional.MAPPINGS[a["index"] % 6]
        label = "ABC"[mapping.index(service)]
        record = optional.record_from_inference(
            a, dict(raw=label, generated_token_ids=[123], ended_with_eos=True)
        )
        record["thinking_closed"] = True
        records[a["trial_id"]] = record
    return records


def test_optional_assignments_preserve_neutral_wording_and_balance_every_mapping_gain_cell():
    rows = optional.assignments()
    assert len(rows) == len({a["trial_id"] for a in rows}) == 96
    counts = Counter((a["family"], a["index"] % 6, int(a["query"] not in a["known"])) for a in rows)
    assert len(counts) == 24 and set(counts.values()) == {4}
    for a in rows:
        assert (
            a["seed"] == 2026101415
            and a["task_revision"] == optional.REVISION
            and a["vocabulary"] == "heldout"
        )
        assert optional.task(a) == state.task(a)
        assert optional.context(a) == state.context(a)
        assert optional.prompt(a) == state.prompt(a)
        text, labels, expected = optional.prompt(a)
        assert labels == "ABC" and expected is None
        assert "This choice is optional; no action is prescribed." in text
        assert "No later questions, scores, rewards, or other trials depend on this choice." in text
        assert "Your goal" not in text
        assert optional.task(a)["uncertainty_bits"] == 1


def test_action_labels_decode_to_services_without_correctness_or_preference_labels():
    for a in optional.assignments():
        for label, service in zip("ABC", optional.MAPPINGS[a["index"] % 6]):
            record = optional.record_from_inference(a, {"raw": label + "."})
            assert record["valid"] and record["selected_service"] == service
            assert record["expected"] is None and record["correct"] is None
        bad = optional.record_from_inference(a, {"raw": "A because I want information"})
        assert not bad["valid"] and bad["selected_service"] is None and bad["correct"] is None


def test_thinking_wrapper_keeps_optional_correctness_null_and_requires_closure():
    from curiosity_study import thinking_information as thinking

    a = optional.assignments()[0]
    assert thinking.task_module(a) is optional
    closed = thinking.record_from_inference(a, {"raw": "Consider the options.</think> A"})
    assert closed["thinking_closed"] and closed["valid"] and closed["correct"] is None
    unclosed = thinking.record_from_inference(a, {"raw": "A"})
    assert not unclosed["thinking_closed"] and not unclosed["valid"]
    assert unclosed["correct"] is None
    m = manifest()
    rendered = thinking.render(m, records_for(m))
    assert "Feasibility gates passed: True" in rendered and "correct 0" not in rendered


@pytest.mark.parametrize("service", optional.SERVICES)
def test_every_uniform_service_preference_passes_feasibility(service):
    m = manifest()
    result = optional.evaluate(m, records_for(m, service))
    assert result["passed"]
    assert result["overall"]["service_counts"][service] == 96
    assert result["overall"]["service_proportions_of_assigned"][service] == 1
    assert result["overall"]["service_proportions_of_valid"][service] == 1
    assert result["overall"]["worlds"] == 24
    assert all(row["valid_closed"] == 4 for row in result["mapping_information_strata"])
    assert "correct 0" not in optional.render(m, records_for(m, service)).lower()


def test_per_family_closure_gate_cannot_be_hidden_by_overall_validity():
    m = manifest()
    records = records_for(m)
    objects = [a["trial_id"] for a in m["diagnostics"] if a["family"] == "objects"]
    for key in objects[:3]:
        records[key]["thinking_closed"] = False
    result = optional.evaluate(m, records)
    assert result["checks"]["overall_valid_closed"]
    assert not result["checks"]["every_family_valid_closed"] and not result["passed"]
    assert result["overall"]["service_counts"]["report"] == 93
    assert result["overall"]["service_proportions_of_assigned"]["report"] == 93 / 96
    assert result["overall"]["service_proportions_of_valid"]["report"] == 1


def test_failures_missingness_and_accidental_accuracy_scoring_remain_visible():
    m = manifest()
    originals = records_for(m)
    key = m["diagnostics"][0]["trial_id"]
    failed = copy.deepcopy(originals)
    failed[key] = dict(trial_id=key, terminal=True, failure="execution_error")
    result = optional.evaluate(m, failed)
    assert not result["passed"] and result["overall"]["execution_failures"] == 1
    missing = copy.deepcopy(originals)
    del missing[key]
    result = optional.evaluate(m, missing)
    assert not result["passed"] and result["overall"]["missing"] == 1
    scored = copy.deepcopy(originals)
    scored[key]["correct"] = False
    result = optional.evaluate(m, scored)
    assert not result["checks"]["no_accuracy_scoring"] and not result["passed"]


def test_cap_hits_are_reported_without_introducing_an_unplanned_preference_or_cap_gate():
    m = manifest()
    records = records_for(m, "pattern")
    row = records[m["diagnostics"][0]["trial_id"]]
    row["inference"].update(generated_token_ids=[12] * 2048, ended_with_eos=False)
    result = optional.evaluate(m, records)
    assert result["passed"] and result["overall"]["cap_hits"] == 1
    assert result["overall"]["service_counts"]["pattern"] == 96


def test_preparation_requires_complete_passing_bound_full_source_without_writing_artifacts():
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_optional_information_smoke.py"
    spec = importlib.util.spec_from_file_location("prepare_optional_information_smoke", path)
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    rows = state.assignments(100)
    source = dict(
        assay="thinking_information",
        phase="validation",
        diagnostics=rows,
        assignment_sha256=digest(rows),
        model=dict(model_id="Qwen/Qwen3-32B", revision="synthetic-pinned"),
    )
    audit = dict(manifest_sha256=digest(source), complete=True, assigned=4000, gates=dict(passed=True))
    result = prepare.build_manifest(source, audit)
    assert len(result["diagnostics"]) == 96 and result["execution"]["batch_size"] == 32
    assert result["feasibility_criteria"]["accuracy_gate"] is None
    assert result["feasibility_criteria"]["preference_gate"] is None
    assert result["source_validation_manifest_sha256"] == digest(source)
    for invalid in (
        {**audit, "complete": False},
        {**audit, "gates": {"passed": False}},
        {**audit, "manifest_sha256": "wrong"},
    ):
        with pytest.raises(ValueError, match="must pass"):
            prepare.build_manifest(source, invalid)
