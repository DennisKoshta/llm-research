"""Local-only admission regressions; no Modal API or real budget reservation."""

import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from curiosity_study.artifacts import canonical, digest, read, write_once
from curiosity_study.resource_limits import SCREEN_BUDGET, job_wall_limit

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("baseline_zero", "positive_half", "positive_full")
NOW = 100000.0


def screen(**changes):
    return {
        "assay": "satisfaction_screen",
        "execution_gpu": "H200",
        "execution_group": "baseline_zero",
        "parent_manifest_sha256": "a" * 64,
        "budget": dict(SCREEN_BUDGET),
        **changes,
    }


def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def budget(monkeypatch):
    module = module_from_file("resource_test_budget", ROOT / "infra/budget.py")
    monkeypatch.setattr("time.time", lambda: NOW)
    monkeypatch.setattr(
        module,
        "snapshot",
        lambda *args: {
            "all_stopped_zero_tasks": True,
            "available_compute_estimate_usd": 300,
            "cumulative_compute_estimate_usd": module.BASELINE_COMPUTE,
            "timestamp_utc": "fixture",
        },
    )
    return module


def jobs(tmp_path, manifests=None):
    manifests = manifests if manifests is not None else [screen(execution_group=g) for g in GROUPS]
    paths = []
    for index, manifest in enumerate(manifests):
        path = tmp_path / f"child-{index}"
        write_once(path / "manifest.json", manifest)
        paths.append(path)
    return paths


def test_exact_screen_policy_and_legacy_limit():
    assert job_wall_limit(screen()) == 12000
    assert job_wall_limit({}) == 7200
    assert job_wall_limit({"assay": "natural_information", "budget": dict(SCREEN_BUDGET)}) == 7200
    assert job_wall_limit({"assay": "satisfaction_screen_other", "budget": dict(SCREEN_BUDGET)}) == 7200


@pytest.mark.parametrize(
    "policy",
    [
        None,
        {},
        dict(SCREEN_BUDGET, workers=4),
        dict(SCREEN_BUDGET, workers=3.0),
        dict(SCREEN_BUDGET, shared_deadline_seconds=12001),
        dict(SCREEN_BUDGET, shared_deadline_seconds="12000"),
        dict(SCREEN_BUDGET, stage_ceiling_usd=56),
        dict(SCREEN_BUDGET, stage_ceiling_usd=55.0),
        dict(SCREEN_BUDGET, paid_retries=1),
        dict(SCREEN_BUDGET, paid_retries=False),
        dict(SCREEN_BUDGET, extra=True),
    ],
)
def test_malformed_screen_policy_fails_closed(policy):
    with pytest.raises(ValueError, match="exact frozen"):
        job_wall_limit(screen(budget=policy))


def test_screen_reservation_binds_all_workers_and_global_liability(tmp_path, budget):
    paths = jobs(tmp_path)
    destination = tmp_path / "reservation.json"
    result = budget.reserve_jobs(paths, NOW + 12000, "H200", destination)
    assert result["jobs"] == {str(p.resolve()): digest(read(p / "manifest.json")) for p in paths}
    assert result["maximum_combined_compute_usd"] == pytest.approx(50.6771136)
    assert result["maximum_combined_compute_usd"] < SCREEN_BUDGET["stage_ceiling_usd"]
    assert result["budget_usd"] == budget.BUDGET
    assert budget.FRESH_COMPUTE_ALLOWANCE == 300
    for path in paths:
        deadline, _ = budget.claim_reservation(destination, path, read(path / "manifest.json"), "H200")
        assert deadline == NOW + 12000
    with pytest.raises(ValueError, match="overwrite"):
        budget.claim_reservation(destination, paths[0], read(paths[0] / "manifest.json"), "H200")


@pytest.mark.parametrize("seconds", [179, 12001])
def test_screen_deadline_bounds(tmp_path, budget, seconds):
    with pytest.raises(ValueError, match="deadline"):
        budget.reserve_jobs(jobs(tmp_path), NOW + seconds, "H200", tmp_path / "reservation.json")
    assert not (tmp_path / "reservation.json").exists()


@pytest.mark.parametrize("defect", ["missing", "four", "mixed", "duplicate_group", "duplicate_manifest", "other_parent"])
def test_screen_requires_exact_three_distinct_related_groups(tmp_path, budget, defect):
    manifests = [screen(execution_group=g) for g in GROUPS]
    if defect == "missing":
        manifests.pop()
    elif defect == "four":
        manifests.append(screen(execution_group="extra"))
    elif defect == "mixed":
        manifests[1]["assay"] = "natural_information"
    elif defect == "duplicate_group":
        manifests[1]["execution_group"] = manifests[0]["execution_group"]
        manifests[1]["different_bytes"] = True
    elif defect == "duplicate_manifest":
        manifests[1] = copy.deepcopy(manifests[0])
    else:
        manifests[1]["parent_manifest_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="three distinct"):
        budget.reserve_jobs(jobs(tmp_path, manifests), NOW + 12000, "H200", tmp_path / "reservation.json")


def test_duplicate_paths_and_five_workers_rejected(tmp_path, budget):
    paths = jobs(tmp_path)
    with pytest.raises(ValueError, match="duplicate worker"):
        budget.reserve_jobs([paths[0], paths[0], paths[2]], NOW + 12000, "H200", tmp_path / "reservation.json")
    with pytest.raises(ValueError, match="size"):
        budget.reserve_jobs(paths + paths[:2], NOW + 12000, "H200", tmp_path / "reservation.json")


def test_global_and_stage_caps_both_remain_enforced(tmp_path, budget, monkeypatch):
    paths = jobs(tmp_path)
    state = budget.snapshot()
    state["available_compute_estimate_usd"] = 50
    monkeypatch.setattr(budget, "snapshot", lambda: state)
    with pytest.raises(RuntimeError, match="remaining pilot allowance"):
        budget.reserve_jobs(paths, NOW + 12000, "H200", tmp_path / "reservation.json")
    state["available_compute_estimate_usd"] = 300
    # A prospective rate increase cannot silently override the fixed $55 ceiling.
    monkeypatch.setitem(budget.RATES, "H200", 0.002)
    with pytest.raises(RuntimeError, match="stage ceiling"):
        budget.reserve_jobs(paths, NOW + 12000, "H200", tmp_path / "reservation.json")


def test_legacy_admission_and_reservation_keep_7200(tmp_path, budget):
    seconds, _ = budget.admission_seconds("H200")
    assert seconds == 7200
    path = jobs(tmp_path, [{"assay": "legacy", "execution_gpu": "H200"}])[0]
    with pytest.raises(ValueError, match="deadline"):
        budget.reserve_jobs([path], NOW + 7201, "H200", tmp_path / "bad.json")
    budget.reserve_jobs([path], NOW + 7200, "H200", tmp_path / "good.json")
    budget.claim_reservation(tmp_path / "good.json", path, read(path / "manifest.json"), "H200")


@pytest.mark.parametrize("defect", ["stale_peer", "missing_peer", "duplicate_group", "extended_deadline", "wrong_liability"])
def test_claim_rechecks_complete_screen_binding(tmp_path, budget, defect):
    paths = jobs(tmp_path)
    destination = tmp_path / "reservation.json"
    reservation = budget.reserve_jobs(paths, NOW + 12000, "H200", destination)
    manifest = read(paths[0] / "manifest.json")
    if defect in {"stale_peer", "duplicate_group"}:
        peer = read(paths[1] / "manifest.json")
        peer["execution_group"] = "baseline_zero"
        (paths[1] / "manifest.json").write_text(canonical(peer))
        if defect == "duplicate_group":
            reservation["jobs"][str(paths[1].resolve())] = digest(peer)
    elif defect == "missing_peer":
        del reservation["jobs"][str(paths[1].resolve())]
    elif defect == "extended_deadline":
        # Even when remaining time is within12000, the initial reservation cannot exceed it.
        reservation["created"] -= 1
    else:
        reservation["maximum_combined_compute_usd"] = 1
    destination.write_text(canonical(reservation))
    with pytest.raises(ValueError, match="binding|three distinct|wall-time"):
        budget.claim_reservation(destination, paths[0], manifest, "H200")
    assert not (tmp_path / "claims").exists()


def test_screen_paid_resume_rejected_before_cloud_or_file_access(tmp_path, budget, monkeypatch):
    def forbidden():
        pytest.fail("Resume must not consult live cloud state")

    monkeypatch.setattr(budget, "snapshot", forbidden)
    with pytest.raises(ValueError, match="forbids paid resume"):
        budget.claim_reserved_resume(tmp_path / "absent.json", tmp_path / "absent", screen(), "H200", tmp_path / "next.json")


def test_screen_never_started_recovery_keeps_original_deadline(tmp_path, budget):
    paths = jobs(tmp_path)
    destination = tmp_path / "reservation.json"
    budget.reserve_jobs(paths, NOW + 12000, "H200", destination)
    manifest = read(paths[0] / "manifest.json")
    budget.claim_reservation(destination, paths[0], manifest, "H200")
    write_once(paths[0] / "coordinator_end.json", {"app_id": None, "cancel_requested": False})
    deadline, _ = budget.claim_reservation(destination, paths[0], manifest, "H200", recover_unstarted=True)
    assert deadline == NOW + 12000
    write_once(paths[0] / "coordinator_end.json", {"app_id": "started", "cancel_requested": True})
    with pytest.raises(ValueError, match="app was started"):
        budget.claim_reservation(destination, paths[0], manifest, "H200", recover_unstarted=True)


@pytest.fixture
def modal_app(monkeypatch, tmp_path):
    class Resource:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: self

    configuration = {}

    class App:
        def __init__(self, *_args):
            pass

        def function(self, **kwargs):
            def decorate(function):
                configuration[function.__name__] = kwargs
                return function

            return decorate

    fake = SimpleNamespace(
        App=App,
        Volume=SimpleNamespace(from_name=lambda *args, **kwargs: Resource()),
        Image=SimpleNamespace(debian_slim=lambda **kwargs: Resource()),
    )
    monkeypatch.setitem(sys.modules, "modal", fake)
    module = module_from_file("resource_test_modal_app", ROOT / "infra/modal_app.py")
    monkeypatch.setattr("time.time", lambda: NOW)
    monkeypatch.setattr("curiosity_study.runner.source_hash", lambda: "source")
    monkeypatch.setattr(module, "Path", lambda value: tmp_path if value == "/artifacts" else Path(value))
    return module, configuration


@pytest.mark.parametrize("manifest,seconds", [({}, 7201), (screen(), 12001), (screen(), 0)])
def test_cloud_uses_manifest_bound_deadline(modal_app, manifest, seconds):
    module, configuration = modal_app
    assert configuration["pilot"]["timeout"] == 7200
    assert configuration["satisfaction_pilot"]["timeout"] == 12600
    assert configuration["pilot"]["retries"] == configuration["satisfaction_pilot"]["retries"] == 0
    entrypoint = module.satisfaction_pilot if manifest.get("assay") == "satisfaction_screen" else module.pilot
    with pytest.raises(ValueError, match="deadline"):
        entrypoint("fixture", manifest, NOW + seconds, "image")


@pytest.mark.parametrize("manifest,seconds", [({}, 7200), (screen(), 12000)])
def test_cloud_accepts_only_bound_limit_before_source_verification(modal_app, manifest, seconds):
    module, _ = modal_app
    manifest = dict(manifest, source_sha256="different-source")
    entrypoint = module.satisfaction_pilot if manifest.get("assay") == "satisfaction_screen" else module.pilot
    with pytest.raises(ValueError, match="Cloud source"):
        entrypoint("fixture", manifest, NOW + seconds, "image")


def test_cloud_screen_rejects_second_admission(modal_app, tmp_path):
    module, _ = modal_app
    write_once(tmp_path / "fixture/admissions/first.json", {"start": NOW - 100})
    with pytest.raises(RuntimeError, match="forbids paid resume"):
        module.satisfaction_pilot("fixture", screen(source_sha256="source"), NOW + 12000, "image")


def test_cloud_legacy_attempt_limit_stays_three(modal_app, tmp_path):
    module, _ = modal_app
    for index in range(3):
        write_once(tmp_path / f"fixture/admissions/{index}.json", {"start": NOW - 100})
    with pytest.raises(RuntimeError, match="three infrastructure"):
        module.pilot("fixture", {"source_sha256": "source"}, NOW + 7200, "image")


def test_cloud_entrypoints_refuse_cross_routing(modal_app, monkeypatch):
    module, configuration = modal_app
    entered = []
    monkeypatch.setattr(module, "_pilot", lambda *args: entered.append(args))
    with pytest.raises(ValueError, match="reserved cloud entrypoint"):
        module.pilot("fixture", screen(), NOW + 100, "image")
    with pytest.raises(ValueError, match="exact satisfaction screen"):
        module.satisfaction_pilot("fixture", {"budget": dict(SCREEN_BUDGET)}, NOW + 100, "image")
    with pytest.raises(ValueError, match="exact frozen"):
        module.satisfaction_pilot("fixture", screen(budget=dict(SCREEN_BUDGET, paid_retries=1)), NOW + 100, "image")
    assert not entered
    module.pilot("fixture", {}, NOW + 100, "image")
    module.satisfaction_pilot("fixture", screen(), NOW + 100, "image")
    assert entered == [("fixture", {}, NOW + 100, "image"), ("fixture", screen(), NOW + 100, "image")]
    legacy = dict(configuration["pilot"])
    extended = dict(configuration["satisfaction_pilot"])
    legacy.pop("timeout")
    extended.pop("timeout")
    assert legacy == extended
