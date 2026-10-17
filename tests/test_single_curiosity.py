import copy
from dataclasses import replace
import pytest
from curiosity_study import single_curiosity as s
from curiosity_study.oracle import compatible, information


def test_oracle_states_actions_and_pattern_independence():
    for a in s.assignments(12):
        trial = s.SingleTrial(a)
        assert len(compatible(trial.task, trial.observations)) == (1 if a["condition"] == "resolved" else 4)
        status = information(trial.task, trial.observations, trial.task.target_query)
        assert status["information_bits_uniform_prior"] == (0 if a["condition"] == "resolved" else 2)
        pattern = trial.pattern
        changed = copy.deepcopy(a)
        changed.update(condition="resolved", layout="reverse", truth="different")
        assert s.pattern_for(changed) == pattern
        for c in trial.task.candidates:
            independent = copy.deepcopy(trial)
            independent.task = replace(trial.task, truth=c.label)
            key = next(k for k, v in a["mapping"].items() if v == "pattern")
            assert independent.apply(key)["response"] == pattern + "\nTrial ended."
        before = copy.deepcopy(trial.observations)
        for label, service in a["mapping"].items():
            branch = copy.deepcopy(trial)
            event = branch.apply(label)
            assert event["service"] == service
            assert branch.done
            assert len(branch.observations) == len(before) + (service == "clue")
            with pytest.raises(ValueError, match="already ended"):
                branch.apply(label)
        assert trial.observations == before and not trial.done


def test_diagnostics_and_strict_parser():
    for a in s.diagnostic_assignments(s.assignments(8)):
        trial = s.SingleTrial(a)
        _, labels, expected = s.diagnostic_prompt(trial, a)
        request = s.inference_request(a)
        assert request["labels"] == labels
        assert not request["greedy"] and not request["enable_thinking"]
        assert request["max_new_tokens"] == 8
        if a["kind"] == "identify":
            truth = compatible(trial.task, trial.observations)
            assert expected == str("JKLM".index(truth[0]) + 1)
        if a["kind"] == "instrument":
            assert a["mapping"][expected] == "clue"
        record = s.record_from_inference(a, dict(raw=expected, messages=request["messages"]))
        assert record["correct"] and record["valid"]
        bad = s.record_from_inference(a, dict(raw=expected + " because...", messages=request["messages"]))
        assert not bad["correct"] and not bad["valid"]
    a = s.assignments(1)[0]
    record = s.record_from_inference(a, dict(raw="A\nB", messages=[]))
    assert record["first_service"] == "invalid" and not record["valid"]


def test_matching_and_balanced_mappings():
    behavior = s.assignments(100)
    assert len(behavior) == 800 and len({a["trial_id"] for a in behavior}) == 800
    diagnostics = s.diagnostic_assignments(behavior)
    assert len(diagnostics) == 1600
    assert not {a["sampling_seed"] for a in behavior} & {a["sampling_seed"] for a in diagnostics}
    for family in ("machines", "objects"):
        for i in range(100):
            matches = [a for a in behavior if a["family"] == family and a["index"] == i]
            assert len(matches) == 4
            assert all(a["mapping"] == matches[0]["mapping"] for a in matches)
            assert len({s.SingleTrial(a).pattern for a in matches}) == 1
        counts = {k: 0 for k in "ABC"}
        for a in behavior:
            if a["family"] == family and a["condition"] == "unresolved" and a["layout"] == "forward":
                counts[next(k for k, v in a["mapping"].items() if v == "clue")] += 1
        assert max(counts.values()) - min(counts.values()) <= 2
    v = s.build_manifest(100, 20261013, {"model_id": "mock"}, "validation")
    b = s.build_manifest(100, 20261013, {"model_id": "mock"}, "behavior")
    assert v["planned_behavior_sha256"] == b["planned_behavior_sha256"]
    assert v["source_sha256"] == b["source_sha256"]


def test_all_assigned_denominators_and_no_desired_result_gate():
    m = s.build_manifest(100, 20261013, {"model_id": "mock"}, "behavior")
    records = {a["trial_id"]: dict(terminal=True, first_service="stop") for a in m["assignments"]}
    assert s.calibrate(m, records)["passed"]  # Zero information preference is permitted.
    cell = [
        a
        for a in m["assignments"]
        if (a["family"], a["condition"], a["layout"]) == ("machines", "unresolved", "forward")
    ]
    for a in cell[:5]:
        records[a["trial_id"]]["first_service"] = "invalid"
    assert s.calibrate(m, records)["passed"]
    records[cell[5]["trial_id"]]["first_service"] = "invalid"
    result = s.calibrate(m, records)
    assert not result["passed"] and result["rows"][0]["n"] == 100
    records = {a["trial_id"]: dict(terminal=True, first_service="stop") for a in m["assignments"]}
    for a in cell[:16]:
        records[a["trial_id"]]["first_service"] = "clue"
    assert not s.calibrate(m, records)["passed"]
    records.pop(cell[-1]["trial_id"])
    assert not s.calibrate(m, records)["passed"]
    v = s.build_manifest(100, 20261013, {"model_id": "mock"}, "validation")
    records = {a["trial_id"]: dict(terminal=True, correct=True, valid=True) for a in v["diagnostics"]}
    assert s.evaluate(v, records)["passed"]
    records.pop(v["diagnostics"][0]["trial_id"])
    assert not s.evaluate(v, records)["passed"]


def test_v2_visible_table_and_label_oracle():
    m = s.build_v2_manifest(12, 20261015, {"model_id": "mock"}, "validation")
    assert m["task_revision"] == "single-curiosity-v2"
    for a in m["diagnostics"]:
        trial = s.SingleTrial(a)
        evidence = trial.evidence()
        assert trial.candidate_labels() == "WXYZ"
        if a["family"] == "machines":
            assert "modulo" not in evidence and "output = " not in evidence
            assert "Candidate |" in evidence
            assert all(label + " |" in evidence for label in "WXYZ")
        if a["kind"] == "identify":
            _, labels, expected = s.diagnostic_prompt(trial, a)
            assert labels == "WXYZ"
            assert expected == dict(zip("JKLM", "WXYZ"))[compatible(trial.task, trial.observations)[0]]
        if a["kind"] == "uncertainty":
            assert s.diagnostic_prompt(trial, a)[2] == str(len(compatible(trial.task, trial.observations)))
        assert a["trial_id"].startswith("single-v2-")


def test_v1_prompts_unchanged_from_frozen_source():
    from pathlib import Path
    from curiosity_study.artifacts import read, digest

    reference = read(Path(__file__).parent / "fixtures/single-v1-requests.json")
    behavior = s.assignments(12)
    diagnostics = s.diagnostic_assignments(behavior)
    assert digest(behavior) == reference["assignments_sha256"]
    assert digest(diagnostics) == reference["diagnostics_sha256"]
    assert digest([s.inference_request(a) for a in behavior + diagnostics]) == reference["requests_sha256"]
