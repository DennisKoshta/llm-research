"""Actual local infrastructure routes with fake cloud endpoints; never a launch."""
from contextlib import nullcontext
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from curiosity_study import satisfaction_site_resources as resources
from curiosity_study import satisfaction_site_admission as admission
from curiosity_study.artifacts import digest, read, write_once, canonical
from test_resource_limits import NOW, module_from_file, modal_app as _modal_app
from test_satisfaction_site_admission import stage

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = dict(setup="site_setup_pilot", calibration="site_calibration_pilot", validation="site_validation_pilot")
TIMEOUTS = dict(setup=2400, calibration=8400, validation=6900)
modal_app = _modal_app


@pytest.fixture
def budget(monkeypatch, tmp_path):
    module = module_from_file("site_wiring_budget", ROOT/"infra/budget.py")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr("time.time", lambda: NOW)
    monkeypatch.setattr(module, "snapshot", lambda *args: dict(all_stopped_zero_tasks=True,
        available_compute_estimate_usd=300, cumulative_compute_estimate_usd=module.BASELINE_COMPUTE,
        timestamp_utc="synthetic", apps=[]))
    return module


def inputs(root, phase, *, alias="original", setup_hash=None):
    manifests, _, destination = stage(root, phase, alias=alias, setup_hash=setup_hash)
    paths = []
    for i, manifest in enumerate(manifests):
        manifest.update(execution_gpu="H200")
        path = (root/"runs"/f"{phase}-{alias}-{i}").resolve()
        (path/"manifest.json").write_text(canonical(manifest))
        paths.append(path)
    return manifests, paths, destination


def admit(root, budget, phase):
    setup_hash, result = None, None
    for name in admission.ORDER[:admission.ORDER.index(phase)+1]:
        manifests, paths, destination = inputs(root, name, setup_hash=setup_hash)
        reservation = budget.reserve_jobs(paths, NOW+resources.POLICIES[name]["shared_deadline_seconds"], "H200", destination)
        if name == "setup":
            setup_hash = digest(manifests[0])
        result = manifests, paths, destination, reservation
    return result


@pytest.mark.parametrize("phase,expected", [("setup", (1800, 3)), ("calibration", (7800, 34)), ("validation", (6300, 28))])
def test_actual_budget_stage_route_and_whole_worker_contract(tmp_path, budget, phase, expected):
    manifests, _, _ = inputs(tmp_path, phase)
    assert budget._bound_job_limits(manifests) == expected
    malformed = deepcopy(manifests)
    if phase == "setup":
        malformed *= 2
    else:
        malformed.pop()
    with pytest.raises(ValueError):
        budget._bound_job_limits(malformed)


@pytest.mark.parametrize("phase", admission.ORDER)
def test_reserve_jobs_consumes_campaign_and_each_claim_rechecks_it(tmp_path, budget, phase):
    manifests, paths, destination, reservation = admit(tmp_path, budget, phase)
    bound = read(admission.directory(tmp_path)/f"{phase}.json")
    assert bound["reservation_sha256"] == digest(reservation)
    assert bound["jobs"] == {str(path): digest(manifest) for path, manifest in zip(paths, manifests, strict=True)}
    assert len(paths) == (1 if phase == "setup" else 3)
    assert reservation["maximum_combined_compute_usd"] <= resources.POLICIES[phase]["stage_ceiling_usd"]
    for manifest, path in zip(manifests, paths, strict=True):
        deadline, observed = budget.claim_reservation(destination, path, manifest, "H200")
        assert deadline == NOW+resources.POLICIES[phase]["shared_deadline_seconds"] and observed == reservation
    #A new alias still enters the same campaign ledger.
    alias, alias_paths, alias_destination = inputs(tmp_path, phase, alias="forbidden-retry",
        setup_hash=manifests[0].get("setup_certificate", {}).get("manifest_sha256"))
    with pytest.raises(ValueError, match="repeated|superseded"):
        budget.reserve_jobs(alias_paths, NOW+resources.POLICIES[phase]["shared_deadline_seconds"], "H200", alias_destination)
    assert not alias_destination.exists()


@pytest.mark.parametrize("phase", admission.ORDER)
def test_actual_claim_rejects_changed_campaign_ledger_before_claim_file(tmp_path, budget, phase):
    manifests, paths, destination, _ = admit(tmp_path, budget, phase)
    ledger = admission.directory(tmp_path)/f"{phase}.json"
    value = read(ledger)
    value["reserved_materials_sha256"] = "0"*64
    ledger.write_text(canonical(value))
    with pytest.raises(ValueError, match="admission|reservation|material"):
        budget.claim_reservation(destination, paths[0], manifests[0], "H200")
    assert not (destination.parent/"claims").exists()


@pytest.mark.parametrize("phase", admission.ORDER)
@pytest.mark.parametrize("mode", ["resume", "recover"])
def test_site_paid_resume_and_unstarted_recovery_rejected_before_file_or_cloud_access(tmp_path, budget, monkeypatch, phase, mode):
    manifests, paths, _ = inputs(tmp_path, phase)
    monkeypatch.setattr(budget, "snapshot", lambda *args: pytest.fail("No live-state query for forbidden retry"))
    with pytest.raises(ValueError, match="resume|recover|retry|retries"):
        if mode == "resume":
            budget.claim_reserved_resume(tmp_path/"absent", paths[0], manifests[0], "H200", tmp_path/"coordinator")
        else:
            budget.claim_reservation(tmp_path/"absent", paths[0], manifests[0], "H200", recover_unstarted=True)


def test_missing_phase_or_remaining_allowance_never_consumes_admission(tmp_path, budget, monkeypatch):
    manifests, paths, destination = inputs(tmp_path, "calibration")
    with pytest.raises(ValueError, match="skipped"):
        budget.reserve_jobs(paths, NOW+7800, "H200", destination)
    assert not destination.exists()
    _, paths, destination = inputs(tmp_path, "setup")
    state = budget.snapshot()
    state["available_compute_estimate_usd"] = 1
    monkeypatch.setattr(budget, "snapshot", lambda: state)
    with pytest.raises(RuntimeError, match="remaining"):
        budget.reserve_jobs(paths, NOW+1800, "H200", destination)
    assert not destination.exists() and not (admission.directory(tmp_path)/"setup.json").exists()


@pytest.fixture
def launcher(monkeypatch):
    events = []
    class Call:
        object_id = "synthetic-call"
        def get(self, timeout):
            events.append(("get", timeout))
            return {"completed": 0}
        def cancel(self, **kwargs):
            events.append(("cancel", kwargs))
    class Function:
        def __init__(self, name):
            self.name = name
        def spawn(self, *args):
            events.append(("spawn", self.name, args))
            return Call()
    def claim(*args, **kwargs):
        events.append(("claim", args, kwargs))
        return NOW+resources.POLICIES[resources.phase_of(args[2])]["shared_deadline_seconds"], {"budget_usd": 373.35}
    def forbidden(*args, **kwargs):
        pytest.fail("Site stage cannot use unreserved or resumed admission")
    fake_app = SimpleNamespace(app=SimpleNamespace(run=lambda: nullcontext(), app_id="synthetic-app"),
        artifacts=SimpleNamespace(), image=SimpleNamespace(object_id="synthetic-image"), GPU="H200",
        **{name: Function(name) for name in ("pilot", "satisfaction_pilot", "curiosity_pilot", *ENTRYPOINTS.values())})
    monkeypatch.setitem(sys.modules, "modal_app", fake_app)
    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(enable_output=lambda: nullcontext(),
        exception=SimpleNamespace(TimeoutError=TimeoutError)))
    monkeypatch.setitem(sys.modules, "budget", SimpleNamespace(admission_seconds=forbidden,
        claim_reservation=claim, claim_reserved_resume=forbidden))
    spec = importlib.util.spec_from_file_location("site_wiring_launcher", ROOT/"infra/launch_job.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.time, "time", lambda: NOW)
    monkeypatch.setattr(module, "collect", lambda *args: events.append(("collect", args)) or 0)
    return module, events, fake_app


@pytest.mark.parametrize("phase", admission.ORDER)
def test_select_and_actual_cli_route_only_after_bound_claim(tmp_path, monkeypatch, launcher, phase):
    module, events, endpoints = launcher
    manifests, paths, destination = inputs(tmp_path, phase)
    expected = getattr(endpoints, ENTRYPOINTS[phase])
    assert module.select_pilot(manifests[0], destination) is expected
    monkeypatch.setattr(sys, "argv", ["launch_job.py", "--run", str(paths[0]), "--reservation", str(destination)])
    module.main()
    assert [event[0] for event in events] == ["claim", "spawn", "get", "cancel", "collect"]
    assert events[1][1] == ENTRYPOINTS[phase]
    coordinator = read(paths[0]/"coordinator.json")
    assert coordinator["max_admissions"] == 1
    assert coordinator["deadline"] == NOW+resources.POLICIES[phase]["shared_deadline_seconds"]


@pytest.mark.parametrize("phase", admission.ORDER)
@pytest.mark.parametrize("extra", [[], ["--reservation", "fixture", "--resume"], ["--reservation", "fixture", "--recover-unstarted"]])
def test_cli_rejects_unreserved_resume_and_unstarted_recovery_before_claim(tmp_path, monkeypatch, launcher, phase, extra):
    module, events, _ = launcher
    _, paths, _ = inputs(tmp_path, phase)
    monkeypatch.setattr(sys, "argv", ["launch_job.py", "--run", str(paths[0]), *extra])
    with pytest.raises(ValueError, match="reservation|resume|recover|retry|retries"):
        module.main()
    assert events == [] and not (paths[0]/"coordinator.json").exists()


@pytest.mark.parametrize("phase", admission.ORDER)
def test_named_cloud_routes_and_timeouts_are_isolated(modal_app, monkeypatch, tmp_path, phase):
    module, configuration = modal_app
    manifests, _, _ = inputs(tmp_path, phase)
    entered = []
    monkeypatch.setattr(module, "_pilot", lambda *args: entered.append(args))
    endpoint = getattr(module, ENTRYPOINTS[phase])
    assert configuration[ENTRYPOINTS[phase]]["timeout"] == TIMEOUTS[phase]
    assert configuration[ENTRYPOINTS[phase]]["retries"] == 0
    endpoint("synthetic", manifests[0], NOW+resources.POLICIES[phase]["shared_deadline_seconds"], "image")
    assert len(entered) == 1
    for name in ("pilot", "curiosity_pilot", "satisfaction_pilot", *(v for k, v in ENTRYPOINTS.items() if k != phase)):
        with pytest.raises(ValueError):
            getattr(module, name)("synthetic", manifests[0], NOW+100, "image")
    assert len(entered) == 1
    assert configuration["pilot"]["timeout"] == 7200


@pytest.mark.parametrize("phase", admission.ORDER)
def test_actual_cloud_rejects_overlong_deadline_and_second_admission(modal_app, tmp_path, phase):
    module, _ = modal_app
    manifests, _, _ = inputs(tmp_path, phase)
    manifest = dict(manifests[0], source_sha256="source")
    endpoint = getattr(module, ENTRYPOINTS[phase])
    with pytest.raises(ValueError, match="deadline"):
        endpoint("synthetic", manifest, NOW+resources.POLICIES[phase]["shared_deadline_seconds"]+1, "image")
    write_once(tmp_path/"synthetic/admissions/first.json", {"start": NOW-100})
    with pytest.raises(RuntimeError, match="resume|retry|retries|attempt"):
        endpoint("synthetic", manifest, NOW+resources.POLICIES[phase]["shared_deadline_seconds"], "image")


@pytest.mark.parametrize("phase", admission.ORDER)
def test_runner_and_report_dispatch_exact_phase_modules(tmp_path, monkeypatch, phase):
    from curiosity_study import runner, report, satisfaction_site_setup, satisfaction_site_execution
    module = satisfaction_site_setup if phase == "setup" else satisfaction_site_execution
    manifests, paths, _ = inputs(tmp_path, phase)
    manifest = dict(manifests[0], assignments=[], diagnostics=[])
    (paths[0]/"manifest.json").write_text(canonical(manifest))
    seen, model = [], object()
    def checkpoint():
        return None
    sentinel = {"synthetic": "no model calls"}
    def execute(*args):
        seen.append(args)
        return sentinel
    monkeypatch.setattr(module, "execute", execute)
    monkeypatch.setattr(module, "render", lambda m, records: f"site-module:{resources.phase_of(m)}:{len(records)}")
    assert runner.execute(paths[0], manifest, model, checkpoint, NOW+100) is sentinel
    assert seen == [(paths[0], manifest, model, checkpoint, NOW+100)]
    assert report.render(paths[0]) == f"site-module:{phase}:0"
