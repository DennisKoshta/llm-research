import copy
import pytest
from curiosity_study.credit import (
    assignments,
    CreditTrial,
    diagnostic_assignments,
    diagnostic_prompt,
    build_manifest,
    evaluate,
    calibrate,
)
from curiosity_study.oracle import compatible
from curiosity_study.model import MockModel
from curiosity_study.runner import execute


def test_credit_assignments_crossing_and_layout_isolation():
    all_items = assignments(100)
    assert len(all_items) == len({a["trial_id"] for a in all_items}) == 1200
    assert all_items == assignments(100)
    for family in ("machines", "objects"):
        for index in range(100):
            subset = [a for a in all_items if a["family"] == family and a["index"] == index]
            assert len(subset) == 6
            assert len({tuple(a["item_indices"]) for a in subset}) == 1
            assert all(a["mapping"] == subset[0]["mapping"] for a in subset)
            assert subset[0]["item_indices"][0] != subset[0]["item_indices"][1]
    for a in all_items:
        trial = CreditTrial(a)
        for side in "XY":
            resolved = a["condition"] == f"{side.lower()}_resolved"
            assert len(compatible(trial.tasks[side], trial.observations[side])) == (1 if resolved else 4)
            assert trial.status()[side]["decisive"] is (not resolved)


@pytest.mark.parametrize("service", ["X", "Y", "stop", "invalid"])
def test_one_credit_enforced_and_other_clue_withheld(service):
    for a in assignments(3):
        trial = CreditTrial(a)
        sibling = trial.fork()
        before = copy.deepcopy(trial.observations)
        action = next((k for k, v in a["mapping"].items() if v == service), "A because...")
        event = trial.apply(action)
        if service in ("X", "Y"):
            other = "Y" if service == "X" else "X"
            assert event["delivered"] == [service] and event["withheld"] == [other]
            assert trial.credits == 0 and len(trial.observations[service]) == 3
            assert trial.observations[other] == before[other]
            assert f"Mystery {other}:" not in event["response"]
        else:
            assert not event["delivered"] and event["withheld"] == ["X", "Y"]
            assert trial.observations == before and trial.credits == 1
        assert not sibling.done and sibling.credits == 1 and sibling.observations == before
        with pytest.raises(ValueError, match="ended"):
            trial.apply("A")


def test_credit_diagnostics_and_semantics():
    diagnostics = diagnostic_assignments(assignments(100))
    assert len(diagnostics) == len({a["trial_id"] for a in diagnostics}) == 1400
    for a in diagnostics:
        trial = CreditTrial(a)
        content, labels, expected = diagnostic_prompt(trial, a)
        assert expected in set(labels)
        if a["kind"].startswith("identify"):
            side = a["kind"][-1].upper()
            assert compatible(trial.tasks[side], trial.observations[side]) == [expected]
        elif a["kind"].startswith("instrument"):
            side = a["kind"][-1].upper()
            assert a["mapping"][expected] == side
            assert trial.status()[side]["decisive"]
        else:
            selected = next(
                line
                for line in content.split("Which description states what happens?\n", 1)[1].splitlines()
                if line.startswith(expected + ": ") and ("delivered" in line or "Neither" in line)
            )
            assert "Both clues" not in selected
            if a["kind"] == "decline":
                assert "Neither clue" in selected
            else:
                assert f"Only the clue for {a['kind'][-1].upper()}" in selected


def test_credit_validation_joint_failure_and_replay(tmp_path):
    manifest = build_manifest(100, 20261006, {"model_id": "test"}, "validation")
    records = {
        a["trial_id"]: {"correct": True, "valid": True, "terminal": True} for a in manifest["diagnostics"]
    }
    assert evaluate(manifest, records)["passed"]
    failing = copy.deepcopy(records)
    ids = [
        a["trial_id"]
        for a in manifest["diagnostics"]
        if a["family"] == "machines" and a["kind"] == "exclusive_x"
    ]
    for trial_id in ids[:11]:
        failing[trial_id]["correct"] = False
    assert not evaluate(manifest, failing)["passed"]
    del records[ids[0]]
    assert not evaluate(manifest, records)["passed"]
    small = build_manifest(1, 20261006, {"model_id": "mock"}, "behavior")
    first = execute(tmp_path, small, MockModel())

    class NeverCalled(MockModel):
        def infer(self, *args, **kwargs):
            raise AssertionError("Completed credit outcome was rerun")

    assert first == execute(tmp_path, small, NeverCalled())


def test_calibration_rejects_layout_bias_and_all_stopping():
    manifest = build_manifest(100, 20261006, {"model_id": "test"}, "behavior")
    balanced = {
        a["trial_id"]: {"first_service": "X" if a["index"] % 2 else "Y", "terminal": True}
        for a in manifest["assignments"]
    }
    assert calibrate(manifest, balanced)["passed"]
    first_position = {
        a["trial_id"]: {"first_service": a["layout"][0], "terminal": True} for a in manifest["assignments"]
    }
    assert not calibrate(manifest, first_position)["passed"]
    stopping = {key: {"first_service": "stop", "terminal": True} for key in balanced}
    assert not calibrate(manifest, stopping)["passed"]
    boundary = {
        a["trial_id"]: {
            "first_service": "X" if a["index"] < (50 if a["layout"] == "XY" else 65) else "Y",
            "terminal": True,
        }
        for a in manifest["assignments"]
    }
    assert calibrate(manifest, boundary)["passed"]
    del balanced[next(iter(balanced))]
    assert not calibrate(manifest, balanced)["passed"]
