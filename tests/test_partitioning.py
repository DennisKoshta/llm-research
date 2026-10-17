import copy
import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch
import pytest
from curiosity_study.artifacts import RunStore, write_once, read
from curiosity_study.credit import build_variant_manifest

infra = Path(__file__).resolve().parents[1] / "infra"
sys.path.insert(0, str(infra))
spec = importlib.util.spec_from_file_location("partition_controller", infra / "continue_partitioned.py")
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)
budget = sys.modules["budget"]


def fixture(tmp_path):
    parent = tmp_path / "parent"
    m = build_variant_manifest(
        1,
        123,
        {"model_id": "mock"},
        "validation",
        {"candidate_labels": "ABCD", "inference_mode": "think_sampled", "execution_gpu": "H100"},
    )
    store = RunStore(parent, m)

    def record(a):
        return {
            "trial_id": a["trial_id"],
            "assignment": a,
            "terminal": True,
            "kind": a["kind"],
            "correct": True,
            "valid": True,
        }

    store.save(record(m["diagnostics"][0]), "original")
    children = []
    for i in range(2):
        child = tmp_path / f"part{i}"
        cm = copy.deepcopy(m)
        cm["diagnostics"] = m["diagnostics"][1 + i :: 2]
        cs = RunStore(child, cm)
        for a in cm["diagnostics"]:
            cs.save(record(a), "worker")
        write_once(child / "integrity" / "fixture.json", {"passed": True})
        children.append(child)
    return parent, m, children


def test_partition_aggregation_preserves_payloads(tmp_path):
    parent, m, children = fixture(tmp_path)
    original = {
        tid: r for c in children for tid, r in RunStore(c, read(c / "manifest.json")).completed().items()
    }
    assert controller.merge(parent, children) == 14
    merged = RunStore(parent, m).completed()
    assert all(merged[tid] == r for tid, r in original.items())
    proof = read(parent / "parallel" / "aggregation.json")
    assert proof["original_records_retained"] == 1 and len(proof["derived_records"]) == 13


def test_incomplete_partition_does_not_partially_publish(tmp_path):
    parent, m, children = fixture(tmp_path)
    next((children[1] / "attempts").glob("*/*.json")).unlink()
    with pytest.raises(ValueError, match="Incomplete"):
        controller.merge(parent, children)
    assert len(RunStore(parent, m).completed()) == 1


def test_nested_continuation_preserves_outer_assignments(tmp_path):
    parent, m, children = fixture(tmp_path)
    slow = children[1]
    sm = read(slow / "manifest.json")
    original = RunStore(slow, sm).completed()
    remaining = sm["diagnostics"][1:]
    for a in remaining:
        (slow / "attempts" / "worker" / f"{a['trial_id']}.json").unlink()
    nested = []
    for i in range(2):
        child = tmp_path / f"nested-{i}"
        cm = copy.deepcopy(sm)
        cm["diagnostics"] = remaining[i::2]
        store = RunStore(child, cm)
        for a in cm["diagnostics"]:
            store.save(original[a["trial_id"]], "continued")
        write_once(child / "integrity" / "fixture.json", {"passed": True})
        nested.append(child)
    assert controller.merge(slow, nested) == len(original)
    assert RunStore(slow, sm).completed() == original
    assert controller.merge(parent, children) == 14
    merged = RunStore(parent, m).completed()
    assert all(merged[tid] == r for tid, r in original.items())


def test_reservation_bounds_all_workers_and_single_claim(tmp_path):
    import time

    parent, m, children = fixture(tmp_path)
    destination = parent / "parallel" / "reservation.json"
    with patch.object(
        budget,
        "snapshot",
        return_value={
            "all_stopped_zero_tasks": True,
            "available_compute_estimate_usd": 1,
            "cumulative_compute_estimate_usd": 89,
        },
    ):
        with pytest.raises(RuntimeError, match="exceeds"):
            budget.reserve_jobs(children, time.time() + 7000, "H100", destination)
    with patch.object(
        budget,
        "snapshot",
        return_value={
            "all_stopped_zero_tasks": True,
            "available_compute_estimate_usd": 80,
            "cumulative_compute_estimate_usd": 10,
        },
    ):
        r = budget.reserve_jobs(children, time.time() + 7000, "H100", destination)
    assert r["maximum_combined_compute_usd"] > 2 * 7000 * budget.RATES["H100"]
    cm = read(children[0] / "manifest.json")
    budget.claim_reservation(destination, children[0], cm, "H100")
    with pytest.raises(ValueError, match="overwrite"):
        budget.claim_reservation(destination, children[0], cm, "H100")
    with pytest.raises(ValueError, match="bound"):
        budget.claim_reservation(destination, children[1], cm, "H100")


def test_only_never_started_admission_can_recover(tmp_path):
    import time

    parent, m, children = fixture(tmp_path)
    destination = parent / "parallel" / "reservation.json"
    with patch.object(
        budget,
        "snapshot",
        return_value={
            "all_stopped_zero_tasks": True,
            "available_compute_estimate_usd": 80,
            "cumulative_compute_estimate_usd": 10,
        },
    ):
        budget.reserve_jobs(children, time.time() + 7000, "H100", destination)
    child = children[0]
    cm = read(child / "manifest.json")
    budget.claim_reservation(destination, child, cm, "H100")
    write_once(child / "coordinator_end.json", {"app_id": None, "cancel_requested": False})
    budget.claim_reservation(destination, child, cm, "H100", recover_unstarted=True)
    assert not (child / "coordinator_end.json").exists()
    assert len(list((child / "unstarted_admissions").glob("end-*.json"))) == 1
    write_once(child / "coordinator_end.json", {"app_id": "started", "cancel_requested": True})
    with pytest.raises(ValueError, match="app was started"):
        budget.claim_reservation(destination, child, cm, "H100", recover_unstarted=True)


def test_behavior_records_use_the_same_lossless_aggregation(tmp_path):
    parent = tmp_path / "behavior-parent"
    m = build_variant_manifest(
        1,
        321,
        {"model_id": "mock"},
        "behavior",
        {"candidate_labels": "ABCD", "inference_mode": "think_sampled", "execution_gpu": "H100"},
    )
    m["validation_prerequisite"] = {"result": {"passed": True}}
    RunStore(parent, m)
    children = []
    originals = {}
    for i in range(2):
        child = tmp_path / f"behavior-part{i}"
        cm = copy.deepcopy(m)
        cm["assignments"] = m["assignments"][i::2]
        store = RunStore(child, cm)
        for a in cm["assignments"]:
            r = {
                "trial_id": a["trial_id"],
                "assignment": a,
                "terminal": True,
                "kind": "behavior",
                "valid": True,
                "first_service": "stop",
                "ledger": [{"service": "stop", "credits_after": 1}],
            }
            store.save(r, "fixture")
            originals[a["trial_id"]] = r
        write_once(child / "integrity" / "fixture.json", {"passed": True})
        children.append(child)
    assert controller.merge(parent, children) == 12
    assert RunStore(parent, m).completed() == originals


def test_reserved_resume_keeps_one_slot_and_rechecks_budget(tmp_path):
    import time

    parent, m, children = fixture(tmp_path)
    child, peer = children
    deadline = time.time() + 3600
    reservation_path = parent / "parallel" / "reservation.json"
    from curiosity_study.artifacts import digest

    write_once(
        reservation_path,
        {
            "deadline": deadline,
            "gpu": "H100",
            "budget_usd": budget.BUDGET,
            "jobs": {str(c.resolve()): digest(read(c / "manifest.json")) for c in children},
        },
    )
    cm = read(child / "manifest.json")
    budget.claim_reservation(reservation_path, child, cm, "H100")
    write_once(child / "coordinator.json", {"app_id": "target", "deadline": deadline})
    write_once(peer / "coordinator.json", {"app_id": "peer", "deadline": deadline})
    coordinator = child / "coordinators" / "next.json"
    state = {
        "apps": [
            {"App ID": "target", "State": "stopped", "Tasks": "0"},
            {"App ID": "peer", "State": "ephemeral", "Tasks": "1"},
        ],
        "available_compute_estimate_usd": 60,
        "cumulative_compute_estimate_usd": 30,
    }
    with patch.object(budget, "snapshot", return_value=state):
        state["apps"][0]["State"] = "ephemeral"
        with pytest.raises(RuntimeError, match="Prior target"):
            budget.claim_reserved_resume(reservation_path, child, cm, "H100", coordinator)
        state["apps"][0]["State"] = "stopped"
        state["apps"].append({"App ID": "unrelated", "State": "ephemeral", "Tasks": "1"})
        with pytest.raises(RuntimeError, match="unrelated"):
            budget.claim_reserved_resume(reservation_path, child, cm, "H100", coordinator)
        state["apps"].pop()
        state["available_compute_estimate_usd"] = 1
        with pytest.raises(RuntimeError, match="allowance"):
            budget.claim_reserved_resume(reservation_path, child, cm, "H100", coordinator)
        state["available_compute_estimate_usd"] = 60
        resumed_deadline, _ = budget.claim_reserved_resume(reservation_path, child, cm, "H100", coordinator)
        assert resumed_deadline == deadline
        with pytest.raises(RuntimeError, match="outstanding"):
            budget.claim_reserved_resume(reservation_path, child, cm, "H100", coordinator)


def test_batched_partition_preserves_original_cohorts():
    from continue_partitioned import partition_assignments

    assignments = [{"trial_id": str(i)} for i in range(200)]
    manifest = {"diagnostics": assignments, "execution": {"backend": "torch_batched_v1", "batch_size": 32}}
    parts, rule = partition_assignments(manifest, {str(i): {} for i in range(32)}, 4, "diagnostics")
    observed = [part[j : j + 32] for part in parts for j in range(0, len(part), 32)]
    expected = [assignments[j : j + 32] for j in range(32, 200, 32)]
    assert sorted(observed, key=lambda x: int(x[0]["trial_id"])) == expected
    assert "whole original batch cohorts" in rule


def test_partial_original_cohort_stays_separate():
    from continue_partitioned import partition_assignments

    assignments = [{"trial_id": str(i)} for i in range(200)]
    manifest = {"diagnostics": assignments, "execution": {"backend": "torch_batched_v1", "batch_size": 32}}
    parts, _ = partition_assignments(manifest, {str(i): {} for i in range(35)}, 4, "diagnostics")
    observed = [part[j : j + 32] for part in parts for j in range(0, len(part), 32)]
    expected = [assignments[35:64]] + [assignments[j : j + 32] for j in range(64, 200, 32)]
    assert sorted(observed, key=lambda x: int(x[0]["trial_id"])) == expected
