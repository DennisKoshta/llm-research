"""Offline transport/worker contracts. All Modal handles/admissions are synthetic."""

from contextlib import nullcontext
from copy import deepcopy
import datetime as dt
import importlib.util
import json
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import pytest

from curiosity_study import pain_profile_design as design
from curiosity_study import pain_profile_resources as resources
from curiosity_study import pain_profile_worker as worker
from curiosity_study.artifacts import digest, read, write_once
from curiosity_study.pain_profile_admission import CAMPAIGN_NAME
from curiosity_study.pain_profile_assets import sha

ROOT = Path(__file__).resolve().parents[1]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


class Volume:
    def __init__(self):
        self.calls = []
        self.files = {}

    def reload(self):
        self.calls.append("reload")

    def commit(self):
        self.calls.append("commit")

    def iterdir(self, name, recursive):
        assert name == CAMPAIGN_NAME and recursive is True
        return [SimpleNamespace(path=name, type=1) for name in self.files]

    def read_file(self, name):
        return [self.files[name]]


@pytest.fixture
def transport(monkeypatch):
    volumes, calls = {}, []

    def volume(name, **kwargs):
        assert kwargs == {"create_if_missing": False}
        volumes[name] = Volume()
        return volumes[name]

    class Image:
        @classmethod
        def debian_slim(cls, **kwargs):
            calls.append(("debian", kwargs))
            return cls()

        def pip_install(self, *args):
            calls.append(("pip", args))
            return self

        def env(self, value):
            calls.append(("env", value))
            return self

        def add_local_dir(self, *args, **kwargs):
            calls.append(("mount", args, kwargs))
            return self

    class App:
        def __init__(self, name):
            calls.append(("app", name))

        def function(self, **kwargs):
            calls.append(("function", kwargs))
            return lambda fn: SimpleNamespace(function=fn)

        def run(self):
            raise AssertionError("No real or fake App.run belongs in a construction test")

    fake_modal = SimpleNamespace(Volume=SimpleNamespace(from_name=volume), Image=Image, App=App)
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    module = load_file("offline_pain_transport", ROOT / "infra/pain_profile_modal.py")
    return module, calls, volumes


@pytest.mark.parametrize("phase", resources.POLICIES)
def test_cloud_definition_matches_exact_phase_limits_without_admission(transport, tmp_path, phase):
    module, calls, _ = transport
    execution = tmp_path / "executions" / CAMPAIGN_NAME
    execution.mkdir(parents=True)
    _, job, _ = module.create_app(execution, phase)
    policy = resources.policy(phase)
    fn = next(row[1] for row in calls if row[0] == "function")
    assert fn["gpu"] == ("H100!" if policy["gpu"] else None)
    assert fn["cpu"] == (2, 2) and fn["memory"] == (98304, 98304)
    assert fn["timeout"] == policy["function_timeout_seconds"]
    assert fn["max_containers"] == 1 and fn["min_containers"] == fn["buffer_containers"] == 0
    assert fn["retries"] == 0 and fn["enable_memory_snapshot"] is False
    assert fn["include_source"] is False
    assert set(fn["volumes"]) == {"/weights", "/artifacts"}
    assert job.function is module.run_cloud
    assert ("app", policy["app_name"]) in calls
    mount = next(row for row in calls if row[0] == "mount")
    assert mount[1] == (execution, "/project")
    ignore = mount[2]["ignore"]
    assert ignore(execution / "src/curiosity_study/__pycache__/x.pyc")
    assert ignore(execution / ".venv/bin/python")
    assert not ignore(execution / "src/curiosity_study/pain_profile_worker.py")
    assert next(row[1] for row in calls if row[0] == "env")["PYTHONPATH"] == (
        "/project/src:/project/infra:/project/scripts"
    )


def test_cloud_definition_rejects_mutable_project_before_app(transport, tmp_path):
    module, calls, _ = transport
    with pytest.raises(ValueError, match="frozen source"):
        module.create_app(tmp_path, "setup")
    assert calls == []


class Process:
    pid = 123456789

    def __init__(self, code=None):
        self.returncode = code
        self.waits = []

    def poll(self):
        return self.returncode

    def wait(self, timeout):
        self.waits.append(timeout)
        self.returncode = -9


@pytest.mark.parametrize("failure", ["deadline", "checkpoint"])
def test_child_failure_kills_whole_session_and_never_restarts(transport, monkeypatch, tmp_path, failure):
    module, _, _ = transport
    process, launches, kills = Process(), [], []
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **k: launches.append((a, k)) or process)
    monkeypatch.setattr(module.os, "killpg", lambda *args: kills.append(args))
    ticks = iter([0.0, 2.0]) if failure == "deadline" else iter([0.0, 0.0])
    monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: next(ticks)))

    def bad_checkpoint():
        raise OSError("synthetic commit failure")

    with pytest.raises(TimeoutError if failure == "deadline" else OSError):
        module.bounded_child(["synthetic-python", "-m", "worker"], deadline=1, log=tmp_path / "child.log",
                             environment={"PINNED": "yes"},
                             checkpoint=bad_checkpoint if failure == "checkpoint" else None)
    assert len(launches) == 1 and launches[0][1]["start_new_session"] is True
    assert launches[0][1]["env"] == {"PINNED": "yes"}
    assert "shell" not in launches[0][1]
    assert kills == [(process.pid, signal.SIGKILL)] and process.waits == [10]


@pytest.mark.parametrize("code", [0, 17])
def test_finished_child_retains_log_and_nonzero_is_terminal(transport, monkeypatch, tmp_path, code):
    module, _, _ = transport
    launches = []
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **k: launches.append(k) or Process(code))
    monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: 0))
    kwargs = dict(deadline=1, log=tmp_path / "child.log", environment={})
    if code:
        with pytest.raises(ValueError, match="Child exited 17"):
            module.bounded_child(["synthetic"], **kwargs)
    else:
        module.bounded_child(["synthetic"], **kwargs)
    assert len(launches) == 1 and (tmp_path / "child.log").is_file()


def test_expired_child_never_spawns(transport, monkeypatch, tmp_path):
    module, _, _ = transport
    monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: 2))
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **k: pytest.fail("unexpected spawn"))
    with pytest.raises(ValueError, match="Deadline expired"):
        module.bounded_child(["synthetic"], deadline=1, log=tmp_path / "unused", environment={})
    assert not (tmp_path / "unused").exists()


@pytest.fixture
def payload(tmp_path, monkeypatch):
    root = tmp_path / "execution"
    run = tmp_path / CAMPAIGN_NAME
    contents = {"src/curiosity_study/held.py": "VALUE = 1\n", "configs/pin.lock": "test-only lock\n"}
    for name, text in contents.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    files = {name: sha(root / name) for name in contents}
    manifest = dict(assay=resources.ASSAY, revision=design.REVISION,
                    source_sha256=digest({"held.py": contents["src/curiosity_study/held.py"]}),
                    documents={"configs/pin.lock": files["configs/pin.lock"]},
                    environment_lock=dict(path="configs/pain_profile_requirements.lock",
                                          lock_byte_sha256="1" * 64, distributions={"synthetic": "1"}),
                    plan={"synthetic": True}, preflight={"synthetic": True})
    admission = dict(phase="setup", manifest_sha256=digest(manifest), source_sha256=manifest["source_sha256"],
                     source_archive_sha256="2" * 64, policy=resources.policy("setup"), created=100,
                     deadline=1900, execution_files=files, phase_consumed=True, retry_allowed=False)
    put(run / "manifest.json", manifest)
    put(run / "admissions/setup.json", admission)
    (run / "transport/setup").mkdir(parents=True)
    monkeypatch.setattr(worker, "time", SimpleNamespace(time=lambda: 200, monotonic_ns=lambda: 42))
    return root, run, manifest, admission


def test_worker_accepts_actual_design_revision_and_bound_source(payload):
    root, run, manifest, admission = payload
    assert resources.REVISION == design.REVISION
    assert worker.verify_payload(root, run, "setup") == (manifest, admission)


@pytest.mark.parametrize("change", ["source", "document", "added_source", "symlink", "manifest", "expiry",
                                    "wall", "retry", "policy", "document_binding"])
def test_worker_rejects_mutated_execution_before_model_loading(payload, change):
    root, run, manifest, admission = payload
    if change in ("source", "document"):
        name = "src/curiosity_study/held.py" if change == "source" else "configs/pin.lock"
        (root / name).write_text("changed")
    elif change == "added_source":
        (root / "src/curiosity_study/extra.py").write_text("UNBOUND = 1")
    elif change == "symlink":
        source = root / "src/curiosity_study/held.py"
        other = root / "outside"
        other.write_bytes(source.read_bytes())
        source.unlink()
        source.symlink_to(other)
    elif change == "manifest":
        manifest["revision"] = "unreviewed"
        put(run / "manifest.json", manifest)
    else:
        if change == "expiry":
            admission.update(created=-1800, deadline=0)
        elif change == "wall":
            admission["deadline"] += 1
        elif change == "retry":
            admission["retry_allowed"] = True
        elif change == "policy":
            admission["policy"]["gpu"] = "H200"
        elif change == "document_binding":
            admission["execution_files"].pop("configs/pin.lock")
        put(run / "admissions/setup.json", admission)
    with pytest.raises(ValueError):
        worker.verify_payload(root, run, "setup")


@pytest.fixture
def worker_dependencies(monkeypatch):
    events = []
    execution = SimpleNamespace(validate_manifest=lambda m: events.append("validate"))
    monkeypatch.setitem(sys.modules, "curiosity_study.pain_profile_execution", execution)
    import curiosity_study
    monkeypatch.setattr(curiosity_study, "pain_profile_execution", execution, raising=False)
    environment = dict(lock_byte_sha256="1" * 64, distributions={"synthetic": "1"})
    monkeypatch.setattr(worker.assets, "check_environment", lambda path: deepcopy(environment))
    return execution, events, environment


def test_worker_environment_mismatch_prevents_model_load(payload, worker_dependencies, monkeypatch):
    root, run, manifest, admission = payload
    execution, events, _ = worker_dependencies
    manifest["environment_lock"]["distributions"]["synthetic"] = "2"
    put(run / "manifest.json", manifest)
    admission["manifest_sha256"] = digest(manifest)
    put(run / "admissions/setup.json", admission)
    monkeypatch.setattr(worker.assets, "load_model", lambda *a: pytest.fail("model load must be unreachable"))
    with pytest.raises(ValueError, match="environment differs"):
        worker.run_phase(root, run, "setup", root / "cache")
    assert events == []


def test_worker_dispatches_once_with_bound_identity_and_shortened_deadline(payload, worker_dependencies,
                                                                         monkeypatch):
    root, run, manifest, admission = payload
    execution, events, _ = worker_dependencies
    tok, base, model, layers, vectors = [object() for _ in range(5)]
    identity = {"loaded": "synthetic"}

    def load(*args):
        events.append(("load", args))
        return tok, base, model, layers, vectors, identity

    def execute(*args, **kwargs):
        events.append(("execute", args))
        kwargs["checkpoint"]()
        return {"complete": True, "passed": True}

    monkeypatch.setattr(worker.assets, "load_model", load)
    execution.execute = execute
    monkeypatch.setenv("PAIN_PROFILE_IMAGE_ID", "im-synthetic")
    assert worker.run_phase(root, run, "setup", root / "cache")["passed"] is True
    assert events[0] == "validate" and [e[0] for e in events[1:]] == ["load", "execute"]
    assert events[1][1] == (root, root / "cache", admission["deadline"] - 60)
    args = events[2][1]
    assert args[:8] == ("setup", run, manifest, model, tok, layers, vectors, identity)
    assert args[8] == admission["deadline"] - 60
    assert identity["admission_sha256"] == digest(admission)
    assert identity["execution_files_sha256"] == digest(admission["execution_files"])
    assert identity["source_archive_sha256"] == "2" * 64
    assert identity["modal_image_id"] == "im-synthetic"
    assert (run / "transport/setup/checkpoint.request").read_text() == "42"


def test_prefetch_verifies_tokenizer_without_model_or_training(payload, worker_dependencies, monkeypatch):
    root, run, manifest, admission = payload
    _, events, environment = worker_dependencies
    admission.update(phase="prefetch", policy=resources.policy("prefetch"))
    put(run / "admissions/prefetch.json", admission)
    (run / "transport/prefetch").mkdir(parents=True)
    tok = SimpleNamespace(pad_token=None, eos_token="EOS", padding_side="right")
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda path, **k: events.append((path, k)) or tok)))
    monkeypatch.setattr(worker.assets, "load_model", lambda *a: pytest.fail("prefetch loaded a model"))
    monkeypatch.setattr(worker.assets, "prefetch", lambda *a: {"adapter_path": "synthetic-local-adapter"})
    monkeypatch.setattr(design, "validate_preflight", lambda *a, **k: events.append((a, k)))
    result = worker.run_phase(root, run, "prefetch", root / "cache")
    assert result["passed"] is True and result["model_inference_performed"] is False
    assert result["environment"] == environment
    assert events[1] == ("synthetic-local-adapter", {"local_files_only": True})
    assert events[2] == ((manifest["plan"], manifest["preflight"]), {"tokenizer": tok})
    assert tok.pad_token == "EOS" and tok.padding_side == "left"
    assert read(run / "phases/prefetch/summary.json") == result


def test_worker_exception_retains_prior_measured_summary_and_error(payload, monkeypatch):
    root, run, _, _ = payload
    measured = {"complete": False, "passed": False, "observed_calls": 7}
    write_once(run / "phases/setup/summary.json", measured)

    def fail(*args):
        raise RuntimeError("synthetic measured failure")

    monkeypatch.setattr(worker, "run_phase", fail)
    monkeypatch.setattr(sys, "argv", ["worker", "--root", str(root), "--run", str(run),
                                      "--cache", str(root / "cache"), "--phase", "setup"])
    with pytest.raises(RuntimeError, match="measured failure"):
        worker.main()
    assert read(run / "phases/setup/summary.json") == measured
    assert read(run / "transport/setup/worker-error.json")["error"]["type"] == "RuntimeError"


@pytest.fixture
def cloud_run(transport, tmp_path, monkeypatch):
    module, _, volumes = transport
    store = tmp_path / "artifacts"
    cache = tmp_path / "weights"
    monkeypatch.setattr(module, "Path", lambda name: store if str(name) == "/artifacts" else Path(name))
    monkeypatch.setattr(module, "PROJECT", tmp_path / "project")
    monkeypatch.setattr(module, "CACHE", cache)
    monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: 100))
    monkeypatch.setattr(worker, "verify_payload", lambda *a: None)
    return module, store / CAMPAIGN_NAME, cache, volumes


def test_cloud_prefetch_installs_exact_lock_once_then_pinned_worker(cloud_run, monkeypatch):
    module, run, cache, volumes = cloud_run
    commands = []

    def child(command, **kwargs):
        commands.append((command, kwargs))
        if command[:2] == ["uv", "venv"]:
            executable = cache / "runtime/bin/python"
            executable.parent.mkdir(parents=True)
            executable.write_text("synthetic executable never run")
        if "curiosity_study.pain_profile_worker" in command:
            write_once(run / "phases/prefetch/summary.json", {"complete": True, "passed": True})
            marker = run / "transport/prefetch/checkpoint.request"
            marker.write_text("one")
            kwargs["checkpoint"]()
            kwargs["checkpoint"]()

    monkeypatch.setattr(module, "bounded_child", child)
    admission = {"phase": "prefetch", "deadline": 1900}
    result = module.run_cloud({"synthetic": True}, admission, {}, "im-fixed")
    assert result["passed"] is True and len(commands) == 3
    assert commands[0][0] == ["uv", "venv", "--python", "/usr/local/bin/python", str(cache / "runtime")]
    install = commands[1][0]
    assert "--require-hashes" in install and install[install.index("--torch-backend") + 1] == "cu130"
    assert install[-2:] == ["-r", "/project/configs/pain_profile_requirements.lock"]
    assert all(item[1]["deadline"] == 1870 for item in commands)
    env = commands[-1][1]["environment"]
    assert env["PAIN_PROFILE_IMAGE_ID"] == "im-fixed"
    assert env["PYTHONPATH"] == "/project/src:/project/infra:/project/scripts"
    assert env["HF_HOME"] == str(cache / "huggingface")
    assert volumes["curiosity-study-artifacts"].calls.count("commit") == 3
    assert volumes["curiosity-study-weights"].calls.count("commit") == 1
    assert read(run / "transport/prefetch/finished.json")["error"] is None
    with pytest.raises(ValueError, match="consumed"):
        module.run_cloud({"synthetic": True}, admission, {}, "im-fixed")
    assert len(commands) == 3


def test_cloud_install_failure_exports_terminal_error_and_does_not_run_worker(cloud_run, monkeypatch):
    module, run, _, volumes = cloud_run
    calls = []

    def fail(command, **kwargs):
        calls.append(command)
        raise RuntimeError("synthetic install failure")

    monkeypatch.setattr(module, "bounded_child", fail)
    result = module.run_cloud({}, {"phase": "prefetch", "deadline": 1900}, {}, "im-fixed")
    assert result["passed"] is False and result["complete"] is False and len(calls) == 1
    assert read(run / "transport/prefetch/finished.json")["error"]["type"] == "RuntimeError"
    assert volumes["curiosity-study-artifacts"].calls[-1] == "commit"
    assert volumes["curiosity-study-weights"].calls[-1] == "commit"


@pytest.fixture
def launcher(monkeypatch, tmp_path):
    events, volumes = [], Volume()
    root, run = tmp_path, tmp_path / "runs" / CAMPAIGN_NAME
    manifest = {"synthetic": True}
    admission = dict(phase="setup", created=100, deadline=1900, policy=resources.policy("setup"))
    state = {"all_stopped_zero_tasks": True, "timestamp_utc": "synthetic", "apps": []}
    times = SimpleNamespace(now=200.0)
    scenario = dict(snapshot_error=False, get_error=None, startup_expired=False, get_count=0)

    class Call:
        def get(self, timeout):
            events.append(("get", timeout))
            scenario["get_count"] += 1
            if scenario["get_error"]:
                raise scenario["get_error"]
            if scenario["get_count"] == 1:
                raise TimeoutError("normal bounded polling")
            return {"complete": True, "passed": True}

        def cancel(self, **kwargs):
            events.append(("cancel", kwargs))

    class Job:
        def spawn(self, *args):
            events.append(("spawn", args))
            return Call()

    class App:
        app_id = "ap-synthetic"

        def run(self):
            events.append(("run",))
            if scenario["startup_expired"]:
                times.now = 1850
            return nullcontext()

    class Thread:
        def __init__(self, target, daemon):
            assert daemon is True
            self.target = target
            scenario["watchdog"] = target

        def start(self):
            events.append(("watcher_started",))

        def join(self, timeout):
            events.append(("watcher_joined", timeout))

    snapshots = []

    def snapshot(*, timeout=None):
        assert timeout == 20
        snapshots.append(timeout)
        if scenario["snapshot_error"] and len(snapshots) > 1:
            times.now += 200  # Bound any retry loop without a real wait.
            raise RuntimeError("synthetic status service failure")
        return dict(state)

    def claim(*args):
        events.append(("claim", args))
        return admission

    fake_modal = SimpleNamespace(enable_output=lambda: nullcontext())
    fake_transport = SimpleNamespace(create_app=lambda *a: (App(), Job(), SimpleNamespace(object_id="im-fixed")),
                                     artifacts=volumes)
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.setitem(sys.modules, "budget", SimpleNamespace(snapshot=snapshot))
    monkeypatch.setitem(sys.modules, "pain_profile_modal", fake_transport)
    module = load_file("offline_pain_launcher", ROOT / "infra/launch_pain_profile.py")
    frozen = {"execution_root": str(Path(module.__file__).resolve().parents[1])}
    monkeypatch.setattr(module, "verify_frozen_campaign", lambda r: (manifest, frozen, root))
    monkeypatch.setattr(module, "claim", claim)
    monkeypatch.setattr(module, "threading", SimpleNamespace(Thread=Thread, Event=lambda: SimpleNamespace(
        set=lambda: events.append(("watcher_stopped",)), wait=lambda seconds: scenario["waits"].pop(0))))
    monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: times.now,
                                                       sleep=lambda s: setattr(times, "now", times.now + s)))
    monkeypatch.setattr(module, "stop_app", lambda *a: events.append(("stop", a)))

    def collect(r):
        events.append(("collect", r))
        write_once(r / "phases/setup/summary.json", {"complete": True, "passed": True})
        return 3

    monkeypatch.setattr(module, "collect", collect)
    monkeypatch.setattr(sys, "argv", ["launcher", "--run", str(run), "--phase", "setup"])
    scenario["clock"] = times
    return module, events, scenario, run, manifest, admission, volumes


def test_launcher_claims_before_single_spawn_and_always_stops_exports(launcher):
    module, events, _, run, manifest, admission, _ = launcher
    module.main()
    kinds = [row[0] for row in events]
    assert kinds.index("claim") < kinds.index("run") < kinds.index("spawn")
    assert kinds.count("spawn") == 1 and kinds.count("get") == 2
    assert kinds.index("cancel") < kinds.index("stop") < kinds.index("collect")
    assert next(e[1] for e in events if e[0] == "cancel") == {"terminate_containers": True}
    launch = read(run / "launches/setup.json")
    assert launch["manifest_sha256"] == digest(manifest)
    assert launch["admission_sha256"] == digest(admission)
    outcome = read(run / "launches/setup-outcome.json")
    assert outcome["all_stopped_zero_tasks"] is True and outcome["exported_files"] == 3
    assert outcome["failure"] is None


def test_launcher_execution_error_does_not_retry_and_keeps_export(launcher):
    module, events, scenario, run, _, _, _ = launcher
    scenario["get_error"] = RuntimeError("synthetic remote failure")
    with pytest.raises(SystemExit) as caught:
        module.main()
    assert caught.value.code == 1
    assert [row[0] for row in events].count("spawn") == 1
    assert any(row[0] == "collect" for row in events)
    assert read(run / "launches/setup-outcome.json")["failure"]["message"] == "synthetic remote failure"


def test_expired_startup_stops_app_without_submitting_work(launcher):
    module, events, scenario, run, _, _, _ = launcher
    scenario["startup_expired"] = True
    with pytest.raises(SystemExit):
        module.main()
    assert not any(row[0] == "spawn" for row in events)
    assert any(row[0] == "stop" for row in events)
    assert read(run / "launches/setup-outcome.json")["failure"]["type"] == "ValueError"


def test_shutdown_status_failure_still_stops_watcher_exports_and_records_failure(launcher):
    module, events, scenario, run, _, _, _ = launcher
    scenario["snapshot_error"] = True
    with pytest.raises(SystemExit) as caught:
        module.main()
    assert caught.value.code == 1
    kinds = [row[0] for row in events]
    assert "watcher_stopped" in kinds and "watcher_joined" in kinds and "collect" in kinds
    outcome = read(run / "launches/setup-outcome.json")
    assert outcome["all_stopped_zero_tasks"] is False
    assert outcome["exported_files"] == 3
    assert any("status service failure" in row["message"] for row in outcome["shutdown_verification_errors"])
    cost = read(run.parents[1] / "reports" / f"{CAMPAIGN_NAME}-setup-cost-and-shutdown.json")
    assert cost["shutdown_verified"] is False and cost["apps"][-1]["State"] == "unverified"


def test_cloud_artifact_export_is_immutable_and_rejects_escape(launcher, monkeypatch, tmp_path):
    module, _, _, run, _, _, volume = launcher
    # Restore the actual independently loaded export function, keeping all handles fake.
    spec = load_file("offline_pain_export", ROOT / "infra/launch_pain_profile.py")
    monkeypatch.setattr(spec, "artifacts", volume)
    volume.files = {f"/{CAMPAIGN_NAME}/calls/0.json": b"retained evidence\n"}
    assert spec.collect(run) == 1
    assert (run / "calls/0.json").read_bytes() == b"retained evidence\n"
    assert spec.collect(run) == 1
    volume.files[f"/{CAMPAIGN_NAME}/calls/0.json"] = b"changed evidence"
    with pytest.raises(ValueError, match="conflicts"):
        spec.collect(run)
    assert (run / "calls/0.json").read_bytes() == b"retained evidence\n"
    volume.files = {f"/{CAMPAIGN_NAME}/../../escape": b"bad"}
    with pytest.raises(ValueError, match="Unsafe"):
        spec.collect(run)
    assert not (tmp_path / "escape").exists()


def test_watchdog_stops_only_this_admitted_app_after_deadline(launcher, monkeypatch):
    module, events, scenario, _, _, admission, _ = launcher
    module.main()
    scenario["clock"].now = admission["deadline"] + 1
    scenario["waits"] = [False, True]
    def iso(t):
        return dt.datetime.fromtimestamp(t, dt.timezone.utc).isoformat()

    entries = [
        {"App ID": "current", "Description": admission["policy"]["app_name"], "State": "ephemeral",
         "Created at": iso(admission["created"])},
        {"App ID": "old", "Description": admission["policy"]["app_name"], "State": "ephemeral",
         "Created at": iso(admission["created"] - 20)},
        {"App ID": "stopped", "Description": admission["policy"]["app_name"], "State": "stopped",
         "Created at": iso(admission["created"])},
        {"App ID": "unrelated", "Description": "some-other-project", "State": "ephemeral",
         "Created at": iso(admission["created"])},
    ]

    def listing(command, **kwargs):
        assert command[-3:] == ["app", "list", "--json"] and kwargs["timeout"] == 20
        return json.dumps(entries)

    monkeypatch.setattr(module.subprocess, "check_output", listing)
    events.clear()
    scenario["watchdog"]()
    assert len(events) == 1 and events[0][0] == "stop" and events[0][1][-1] == "current"


def test_export_failure_preserves_outcome_and_fails_without_respawn(launcher, monkeypatch):
    module, events, _, run, _, _, _ = launcher

    def fail_export(run):
        raise OSError("synthetic export failure")

    monkeypatch.setattr(module, "collect", fail_export)
    with pytest.raises(SystemExit) as caught:
        module.main()
    assert caught.value.code == 1
    outcome = read(run / "launches/setup-outcome.json")
    assert outcome["all_stopped_zero_tasks"] is True
    assert outcome["export_error"]["message"] == "synthetic export failure"
    assert [row[0] for row in events].count("spawn") == 1


def test_status_cli_timeout_is_optional_for_legacy_and_propagated_for_campaign(monkeypatch, tmp_path):
    # Load the actual ledger calculator, replacing only its external CLI response.
    module = load_file("offline_budget_snapshot", ROOT / "infra/budget.py")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    calls = []

    def listing(command, **kwargs):
        calls.append((command, kwargs))
        return "[]"

    monkeypatch.setattr(module.subprocess, "check_output", listing)
    now = dt.datetime(2026, 10, 15, tzinfo=dt.timezone.utc)
    assert module.snapshot(now)["all_stopped_zero_tasks"] is True
    assert "timeout" not in calls[-1][1]
    assert module.snapshot(now, timeout=20)["all_stopped_zero_tasks"] is True
    assert calls[-1][1]["timeout"] == 20
