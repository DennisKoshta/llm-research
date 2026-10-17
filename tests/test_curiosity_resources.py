"""Separate C-screen routing/admission policy, entirely mocked local execution."""

from copy import deepcopy

import pytest

from curiosity_study.artifacts import canonical, read, write_once
from curiosity_study.curiosity_resource_limits import ASSAY, CURIOSITY_SCREEN_BUDGET, GROUPS, job_wall_limit
from curiosity_study.resource_limits import SCREEN_BUDGET, job_wall_limit as legacy_limit
from test_resource_limits import NOW, budget as _budget, jobs, modal_app as _modal_app
from test_cloud_routing import launcher as _launcher, run_cli

budget = _budget
modal_app = _modal_app
launcher = _launcher


def screen(**changes):
    return dict(assay=ASSAY, budget=dict(CURIOSITY_SCREEN_BUDGET), execution_gpu="H200",
        execution_group="worker_0", parent_manifest_sha256="a" * 64, **changes)


def children():
    return [dict(screen(), execution_group=group) for group in GROUPS]


def test_named_policy_leaves_legacy_helper_and_limits_unchanged():
    assert job_wall_limit(screen()) == 12000
    assert legacy_limit(screen()) == 7200  # The old helper has no new exception.
    assert job_wall_limit({"assay": "satisfaction_screen", "budget": dict(SCREEN_BUDGET)}) == 12000
    assert job_wall_limit({"assay": "unknown", "budget": dict(CURIOSITY_SCREEN_BUDGET)}) == 7200


@pytest.mark.parametrize("policy", [None, {}, dict(CURIOSITY_SCREEN_BUDGET, workers=3.0),
    dict(CURIOSITY_SCREEN_BUDGET, shared_deadline_seconds=12001),
    dict(CURIOSITY_SCREEN_BUDGET, stage_ceiling_usd=56),
    dict(CURIOSITY_SCREEN_BUDGET, paid_retries=False),
    dict(CURIOSITY_SCREEN_BUDGET, paid_retries=1),
    dict(CURIOSITY_SCREEN_BUDGET, extra=True)])
def test_malformed_policy_rejected(policy):
    m = screen()
    m["budget"] = policy
    with pytest.raises(ValueError, match="exact frozen"):
        job_wall_limit(m)


def test_all_three_reserved_before_claim_with_original_liability(tmp_path, budget):
    paths = jobs(tmp_path, children())
    reservation = tmp_path / "reservation.json"
    result = budget.reserve_jobs(paths, NOW + 12000, "H200", reservation)
    assert result["maximum_combined_compute_usd"] == pytest.approx(50.6771136)
    assert budget.FRESH_COMPUTE_ALLOWANCE == 300
    for path in paths:
        deadline, _ = budget.claim_reservation(reservation, path, read(path / "manifest.json"), "H200")
        assert deadline == NOW + 12000


@pytest.mark.parametrize("defect", ["missing", "four", "mixed_old", "mixed_legacy", "duplicate", "other_parent"])
def test_reserved_program_cannot_mix_or_duplicate_workers(tmp_path, budget, defect):
    values = children()
    if defect == "missing":
        values.pop()
    elif defect == "four":
        values.append(deepcopy(values[0]))
    elif defect == "mixed_old":
        values[1].update(assay="satisfaction_screen", execution_group="positive_half", budget=dict(SCREEN_BUDGET))
    elif defect == "mixed_legacy":
        values[1]["assay"] = "legacy"
    elif defect == "duplicate":
        values[1]["execution_group"] = values[0]["execution_group"]
    else:
        values[1]["parent_manifest_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="three"):
        budget.reserve_jobs(jobs(tmp_path, values), NOW + 12000, "H200", tmp_path / "reservation.json")
    assert not (tmp_path / "reservation.json").exists()


@pytest.mark.parametrize("defect", ["stale_peer", "missing_peer", "extended_initial", "wrong_liability"])
def test_claim_rechecks_all_peer_bindings_and_original_deadline(tmp_path, budget, defect):
    paths = jobs(tmp_path, children())
    destination = tmp_path / "reservation.json"
    reservation = budget.reserve_jobs(paths, NOW + 12000, "H200", destination)
    if defect == "stale_peer":
        peer = read(paths[1] / "manifest.json")
        peer["changed"] = True
        (paths[1] / "manifest.json").write_text(canonical(peer))
    elif defect == "missing_peer":
        del reservation["jobs"][str(paths[1].resolve())]
    elif defect == "extended_initial":
        reservation["created"] -= 1
    else:
        reservation["maximum_combined_compute_usd"] = 1
    destination.write_text(canonical(reservation))
    with pytest.raises(ValueError):
        budget.claim_reservation(destination, paths[0], read(paths[0] / "manifest.json"), "H200")
    assert not (tmp_path / "claims").exists()


def test_no_resume_or_budget_override(tmp_path, budget, monkeypatch):
    with pytest.raises(ValueError, match="forbids paid resume"):
        budget.claim_reserved_resume(tmp_path / "absent", tmp_path, screen(), "H200", tmp_path / "coordinator")
    paths = jobs(tmp_path, children())
    state = budget.snapshot()
    state["available_compute_estimate_usd"] = 50
    monkeypatch.setattr(budget, "snapshot", lambda: state)
    with pytest.raises(RuntimeError, match="remaining pilot"):
        budget.reserve_jobs(paths, NOW + 12000, "H200", tmp_path / "reservation.json")
    state["available_compute_estimate_usd"] = 300
    monkeypatch.setitem(budget.RATES, "H200", 0.002)
    with pytest.raises(RuntimeError, match="stage ceiling"):
        budget.reserve_jobs(paths, NOW + 12000, "H200", tmp_path / "reservation.json")


def test_cloud_exception_separate_from_both_legacy_entrypoints(modal_app, monkeypatch):
    module, config = modal_app
    assert config["pilot"]["timeout"] == 7200
    assert config["satisfaction_pilot"]["timeout"] == config["curiosity_pilot"]["timeout"] == 12600
    entered = []
    monkeypatch.setattr(module, "_pilot", lambda *args: entered.append(args))
    module.curiosity_pilot("fixture", screen(), NOW + 12000, "image")
    assert len(entered) == 1
    for entrypoint in (module.pilot, module.satisfaction_pilot):
        with pytest.raises(ValueError):
            entrypoint("fixture", screen(), NOW + 12000, "image")
    with pytest.raises(ValueError):
        module.curiosity_pilot("fixture", {"assay": "satisfaction_screen", "budget": dict(SCREEN_BUDGET)}, NOW + 12000, "image")
    assert len(entered) == 1


def test_cloud_deadline_and_no_second_admission(modal_app, tmp_path):
    module, _ = modal_app
    m = dict(screen(), source_sha256="source")
    with pytest.raises(ValueError, match="deadline"):
        module.curiosity_pilot("fixture", m, NOW + 12001, "image")
    write_once(tmp_path / "fixture/admissions/first.json", {"start": NOW - 100})
    with pytest.raises(RuntimeError, match="forbids paid resume"):
        module.curiosity_pilot("fixture", m, NOW + 12000, "image")


def test_cli_requires_claim_and_routes_only_named_entrypoint(launcher, monkeypatch, tmp_path):
    module, events = launcher
    root = tmp_path / "new-screen"
    write_once(root / "manifest.json", screen())
    run_cli(module, monkeypatch, root, ["--reservation", str(tmp_path / "reservation")])
    assert [e[0] for e in events] == ["claim", "spawn", "get", "cancel", "collect"]
    assert events[1][1] == "curiosity_pilot"
    assert read(root / "coordinator.json")["max_admissions"] == 1


@pytest.mark.parametrize("extra", [[], ["--reservation", "fixture", "--resume"]])
def test_cli_no_unreserved_or_paid_retry(launcher, monkeypatch, tmp_path, extra):
    module, events = launcher
    root = tmp_path / "new-screen"
    write_once(root / "manifest.json", screen())
    with pytest.raises(ValueError, match="reservation|resume"):
        run_cli(module, monkeypatch, root, extra)
    assert events == []


@pytest.mark.parametrize("failure", [dict(status="execution_error"), dict(failure="generation_execution_error")])
def test_terminal_generation_failure_cannot_pass_new_operational_coverage(tmp_path, failure):
    from curiosity_study.artifacts import RunStore, digest
    from test_collection_coverage import check_collection_coverage, CollectionCoverageError
    rows = [dict(trial_id=f"synthetic-{i}") for i in range(32)]
    m = dict(assay=ASSAY, assignments=[], diagnostics=rows, assignment_sha256=digest(rows),
             execution=dict(backend="torch_batched_v1", batch_size=32))
    records = [dict(trial_id=r["trial_id"], assignment=r, terminal=True, status="completed") for r in rows]
    records[-1].update(failure)
    RunStore(tmp_path, m).save_many(records, "synthetic", "0000000")
    with pytest.raises(CollectionCoverageError, match="execution-error"):
        check_collection_coverage(tmp_path, expected_manifest_sha256=digest(m))
    assert len(RunStore(tmp_path, m).completed()) == 32
