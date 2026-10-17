"""Exercise CLI routing with fake Modal functions, never a cloud launch."""

from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from curiosity_study.artifacts import read, write_once
from curiosity_study.resource_limits import SCREEN_BUDGET


@pytest.fixture
def launcher(monkeypatch):
    events = []
    now = 100000.0

    class Call:
        object_id = "fake-call"

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

    def admission_seconds(**kwargs):
        events.append(("admit", kwargs))
        return 7200, {"budget_usd": 373.35}

    def claim(*args, **kwargs):
        events.append(("claim", args, kwargs))
        return now + 12000, {"budget_usd": 373.35}

    def resume(*args, **kwargs):
        events.append(("resume", args, kwargs))
        return now + 7000, {"budget_usd": 373.35}

    fake_app = SimpleNamespace(
        app=SimpleNamespace(run=lambda: nullcontext(), app_id="fake-app"),
        pilot=Function("pilot"),
        satisfaction_pilot=Function("satisfaction_pilot"),
        curiosity_pilot=Function("curiosity_pilot"),
        artifacts=SimpleNamespace(),
        image=SimpleNamespace(object_id="fake-image"),
        GPU="H200",
    )
    fake_modal = SimpleNamespace(
        enable_output=lambda: nullcontext(),
        exception=SimpleNamespace(TimeoutError=TimeoutError),
    )
    monkeypatch.setitem(sys.modules, "modal_app", fake_app)
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.setitem(
        sys.modules,
        "budget",
        SimpleNamespace(admission_seconds=admission_seconds, claim_reservation=claim, claim_reserved_resume=resume),
    )
    path = Path(__file__).resolve().parents[1] / "infra/launch_job.py"
    spec = importlib.util.spec_from_file_location("test_routed_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.time, "time", lambda: now)
    monkeypatch.setattr(module, "collect", lambda *args: events.append(("collect", args)) or 0)
    return module, events


def manifest(screen=True):
    result = {"assay": "satisfaction_screen" if screen else "legacy", "execution_gpu": "H200"}
    if screen:
        result["budget"] = dict(SCREEN_BUDGET)
    return result


def run_cli(module, monkeypatch, run, extra=()):
    monkeypatch.setattr(sys, "argv", ["launch_job.py", "--run", str(run), *extra])
    module.main()


def test_screen_routes_only_after_claim_and_records_one_admission(launcher, monkeypatch, tmp_path):
    module, events = launcher
    run = tmp_path / "screen"
    write_once(run / "manifest.json", manifest())
    reservation = tmp_path / "reservation.json"
    run_cli(module, monkeypatch, run, ["--reservation", str(reservation)])
    assert [e[0] for e in events] == ["claim", "spawn", "get", "cancel", "collect"]
    assert events[0][1][0] == reservation
    assert events[1][1] == "satisfaction_pilot"
    assert events[1][2] == ("screen", manifest(), 112000.0, "fake-image")
    coordinator = read(run / "coordinator.json")
    assert coordinator["max_admissions"] == 1
    assert coordinator["deadline"] == 112000.0
    assert read(run / "coordinator_end.json")["cancel_requested"] is True


@pytest.mark.parametrize("extra,error", [([], "requires a bound"), (["--reservation", "fixture", "--resume"], "forbids paid resume")])
def test_screen_rejects_unreserved_or_paid_resume_before_admission(launcher, monkeypatch, tmp_path, extra, error):
    module, events = launcher
    run = tmp_path / "screen"
    write_once(run / "manifest.json", manifest())
    with pytest.raises(ValueError, match=error):
        run_cli(module, monkeypatch, run, extra)
    assert events == []
    assert not (run / "coordinator.json").exists()
    assert not (run / "coordinator_end.json").exists()


def test_screen_invalid_policy_never_reaches_claim(launcher, monkeypatch, tmp_path):
    module, events = launcher
    run = tmp_path / "screen"
    changed = manifest()
    changed["budget"]["paid_retries"] = 1
    write_once(run / "manifest.json", changed)
    with pytest.raises(ValueError, match="exact frozen"):
        run_cli(module, monkeypatch, run, ["--reservation", "fixture"])
    assert events == []


def test_failed_reservation_cannot_spawn(launcher, monkeypatch, tmp_path):
    module, events = launcher
    run = tmp_path / "screen"
    write_once(run / "manifest.json", manifest())

    def fail(*args, **kwargs):
        raise ValueError("Stale exact reservation binding")

    monkeypatch.setattr(module, "claim_reservation", fail)
    with pytest.raises(ValueError, match="Stale exact"):
        run_cli(module, monkeypatch, run, ["--reservation", "fixture"])
    assert events == []
    assert not (run / "coordinator.json").exists()


def test_legacy_unreserved_route_keeps_three_admissions(launcher, monkeypatch, tmp_path):
    module, events = launcher
    run = tmp_path / "legacy"
    write_once(run / "manifest.json", manifest(False))
    run_cli(module, monkeypatch, run)
    assert [e[0] for e in events] == ["admit", "spawn", "get", "cancel", "collect"]
    assert events[1][1] == "pilot"
    coordinator = read(run / "coordinator.json")
    assert coordinator["max_admissions"] == 3
    assert coordinator["deadline"] == 107200.0


def test_legacy_reserved_resume_still_routes_to_legacy(launcher, monkeypatch, tmp_path):
    module, events = launcher
    run = tmp_path / "legacy"
    write_once(run / "manifest.json", manifest(False))
    write_once(run / "coordinator.json", {"deadline": 107000.0})
    run_cli(module, monkeypatch, run, ["--reservation", "fixture", "--resume"])
    assert [e[0] for e in events] == ["resume", "spawn", "get", "cancel", "collect"]
    assert events[1][1] == "pilot"
    assert events[1][2][2] == 107000.0
    coordinators = list((run / "coordinators").glob("*.json"))
    assert len(coordinators) == 2
    assert next(read(p) for p in coordinators if not p.stem.endswith("_end"))["max_admissions"] == 3


def test_collect_only_needs_no_reservation_or_launch(launcher, monkeypatch, tmp_path):
    module, events = launcher
    run_cli(module, monkeypatch, tmp_path / "screen", ["--collect-only"])
    assert [e[0] for e in events] == ["collect"]


def test_expiration_message_describes_actual_bound_not_two_hours(launcher):
    module, _ = launcher
    with pytest.raises(TimeoutError, match="Aggregate reserved job deadline reached"):
        module.await_result(None, 99999.0)
