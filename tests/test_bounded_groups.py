"""Local fake jobs and reservations; no cloud, real budget admission, or outcomes."""

import hashlib
import importlib.util
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest

from curiosity_study.artifacts import RunStore, digest, read, write_once


@pytest.fixture
def coordinator(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("run_bounded_groups", scripts / "run_bounded_groups.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module.budget, "ROOT", tmp_path)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    return module


def frozen_collection(root, name, child_count=1):
    parent, execution = root / "runs" / name, root / "executions" / name
    for relative in ("src/curiosity_study/fixture.py", "infra/launch_job.py", "infra/budget.py"):
        path = execution / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# Distinct frozen copy: {name}/{relative}\n")
    rows = [dict(trial_id=f"{name}-{i}") for i in range(2 * child_count)]
    manifest = dict(
        source_sha256=f"source-{name}",
        execution_gpu="H200",
        assignments=[],
        diagnostics=rows,
        assignment_sha256=digest(rows),
        execution=dict(backend="torch_batched_v1", batch_size=2),
    )
    write_once(parent / "manifest.json", manifest)
    jobs = []
    if child_count > 1:
        descriptors = []
        for index in range(child_count):
            path = root / "runs" / f"{name}-part-{index:02d}"
            child_rows = rows[index * 2 : (index + 1) * 2]
            child = {
                **manifest,
                "diagnostics": child_rows,
                "assignment_sha256": digest(child_rows),
                "parent_manifest_sha256": digest(manifest),
            }
            write_once(path / "manifest.json", child)
            descriptors.append(dict(path=str(path), manifest_sha256=digest(child)))
            jobs.append(path)
        write_once(parent / "children.json", descriptors)
    else:
        jobs = [parent]
    archive = parent / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for path in sorted(execution.rglob("*.py")):
            handle.add(path, arcname=str(path.relative_to(execution)))
    write_once(
        parent / "freeze.json",
        dict(
            manifest_sha256=digest(manifest),
            source_sha256=manifest["source_sha256"],
            execution_root=str(execution),
            budget_root=str(root),
            archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        ),
    )
    return parent, jobs


def fake_runtime(module, monkeypatch, all_jobs, *, problem=None):
    events, apps, reservations = [], {}, []

    def check_output(command, *, cwd, env, text):
        assert cwd.parent == module.ROOT / "executions"
        assert env["CURIOSITY_BUDGET_ROOT"] == str(module.ROOT)
        assert env["PYTHONPATH"] == str(cwd / "src") + ":" + str(cwd / "infra")
        events.append(("verify", str(cwd)))
        return f"source-{cwd.name}\n{module.ROOT}\n"

    def reserve(jobs, deadline, gpu, destination):
        assert set(jobs) == set(all_jobs) and gpu == "H200"
        plan = read(destination.parent / "program.json")
        assert plan["common_deadline"] == deadline and plan["total_jobs"] == len(all_jobs)
        assert (
            plan["coordinator_sha256"]
            == hashlib.sha256((destination.parent / "coordinator.py").read_bytes()).hexdigest()
        )
        assert not any(event[0] == "launch" for event in events)
        events.append(("reserve", len(jobs)))
        if problem == "admission":
            raise RuntimeError("Fake allowance exceeded")
        receipt = dict(
            jobs={str(path): digest(read(path / "manifest.json")) for path in jobs},
            deadline=deadline,
            maximum_combined_compute_usd=12.34,
        )
        write_once(destination, receipt)
        reservations.append(receipt)
        return receipt

    def launch(command, *, cwd, env, stdout, stderr):
        path = Path(command[command.index("--run") + 1])
        receipt = read(command[command.index("--reservation") + 1])
        assert len(reservations) == 1 and set(receipt["jobs"]) == set(map(str, all_jobs))
        assert command[1] == str(cwd / "infra/launch_job.py")
        parent_name = path.name.split("-part-")[0]
        assert cwd.name == parent_name
        assert env["CURIOSITY_BUDGET_ROOT"] == str(module.ROOT)
        events.append(("launch", str(path)))
        if problem == "launch" and path == all_jobs[0]:
            raise OSError("Fake local launch failure")
        app_id = f"fake-{path.name}"
        apps[app_id] = {"App ID": app_id, "State": "stopped", "Tasks": "0"}
        write_once(path / "coordinator.json", dict(app_id=app_id, deadline=receipt["deadline"]))
        write_once(path / "coordinator_end.json", dict(app_id=app_id, cancel_requested=True))
        if problem != "coverage" or path != all_jobs[0]:
            manifest = read(path / "manifest.json")
            records = [
                dict(trial_id=a["trial_id"], assignment=a, terminal=True, correct=False, valid=False)
                for a in manifest["diagnostics"]
            ]
            RunStore(path, manifest).save_many(records, "fake", "0000000")
        return SimpleNamespace(returncode=0)

    def snapshot():
        visible = list(apps.values())
        if problem == "missing_app":
            visible = visible[1:]
        if problem == "active" and visible:
            visible[0] = {**visible[0], "State": "ephemeral", "Tasks": "1"}
        return dict(
            apps=visible,
            all_stopped_zero_tasks=problem != "active",
            phase_compute_estimate_usd=1.0,
            available_compute_estimate_usd=299.0,
        )

    monkeypatch.setattr(module.subprocess, "check_output", check_output)
    monkeypatch.setattr(module.subprocess, "run", launch)
    monkeypatch.setattr(module.budget, "reserve_jobs", reserve)
    monkeypatch.setattr(module.budget, "snapshot", snapshot)
    return events, reservations


def replace(path, value):
    path.unlink()
    write_once(path, value)


def test_three_jobs_reserve_once_before_launch_and_use_each_frozen_copy(coordinator, monkeypatch):
    first, jobs_a = frozen_collection(coordinator.ROOT, "relational", 2)
    second, jobs_b = frozen_collection(coordinator.ROOT, "target", 1)
    events, reservations = fake_runtime(coordinator, monkeypatch, jobs_a + jobs_b)
    out = coordinator.ROOT / "runs/program"
    plan = coordinator.run([first, second], 600, out)
    assert [e[0] for e in events] == ["verify", "verify", "reserve", "launch", "launch", "launch"]
    assert len(reservations) == 1 and reservations[0]["deadline"] == plan["common_deadline"]
    assert read(out / "completion.json")["passed"]
    for parent in (first, second):
        receipts = read(out / "collections" / parent.name / "operational-coverage.json")
        assert all(c["passed"] and c["scientific_outcomes_checked"] is False for c in receipts)
    cost = read(plan["cost_receipt"])
    assert len(cost["program_app_ids"]) == len(cost["apps"]) == 3
    assert cost["all_stopped_zero_tasks"] and cost["program_apps_reported"]


def test_stale_child_hash_rejected_before_admission(coordinator, monkeypatch):
    first, jobs_a = frozen_collection(coordinator.ROOT, "first", 2)
    second, jobs_b = frozen_collection(coordinator.ROOT, "second")
    events, reservations = fake_runtime(coordinator, monkeypatch, jobs_a + jobs_b)
    child = read(jobs_a[0] / "manifest.json")
    replace(jobs_a[0] / "manifest.json", {**child, "unplanned": True})
    out = coordinator.ROOT / "runs/program"
    with pytest.raises(ValueError, match="Child manifest"):
        coordinator.run([first, second], 600, out)
    assert not reservations and not out.exists() and all(e[0] == "verify" for e in events)


def test_five_jobs_rejected_before_admission(coordinator, monkeypatch):
    first, jobs_a = frozen_collection(coordinator.ROOT, "first", 3)
    second, jobs_b = frozen_collection(coordinator.ROOT, "second", 2)
    events, reservations = fake_runtime(coordinator, monkeypatch, jobs_a + jobs_b)
    with pytest.raises(ValueError, match="two to four"):
        coordinator.run([first, second], 600, coordinator.ROOT / "runs/program")
    assert not reservations and all(e[0] == "verify" for e in events)


def test_missing_child_descriptor_cannot_silently_shrink_the_parent(coordinator, monkeypatch):
    first, jobs_a = frozen_collection(coordinator.ROOT, "first", 2)
    second, jobs_b = frozen_collection(coordinator.ROOT, "second")
    events, reservations = fake_runtime(coordinator, monkeypatch, jobs_a + jobs_b)
    replace(first / "children.json", read(first / "children.json")[:1])
    with pytest.raises(ValueError, match="every frozen parent assignment"):
        coordinator.run([first, second], 600, coordinator.ROOT / "runs/program")
    assert not reservations and all(e[0] == "verify" for e in events)


def test_rehashed_children_cannot_mix_rows_across_frozen_parent_cohorts(coordinator, monkeypatch):
    first, jobs_a = frozen_collection(coordinator.ROOT, "first", 2)
    second, jobs_b = frozen_collection(coordinator.ROOT, "second")
    events, reservations = fake_runtime(coordinator, monkeypatch, jobs_a + jobs_b)
    manifests = [read(path / "manifest.json") for path in jobs_a]
    manifests[0]["diagnostics"][1], manifests[1]["diagnostics"][0] = (
        manifests[1]["diagnostics"][0],
        manifests[0]["diagnostics"][1],
    )
    descriptors = []
    for path, manifest in zip(jobs_a, manifests):
        manifest["assignment_sha256"] = digest(manifest["diagnostics"])
        replace(path / "manifest.json", manifest)
        descriptors.append(dict(path=str(path), manifest_sha256=digest(manifest)))
    replace(first / "children.json", descriptors)
    with pytest.raises(ValueError, match="whole original parent cohorts"):
        coordinator.run([first, second], 600, coordinator.ROOT / "runs/program")
    assert not reservations and all(e[0] == "verify" for e in events)


@pytest.mark.parametrize("problem", ["coverage", "missing_app", "active", "launch"])
def test_incomplete_or_unreconciled_jobs_never_succeed_even_with_zero_exit(coordinator, monkeypatch, problem):
    first, jobs_a = frozen_collection(coordinator.ROOT, "first")
    second, jobs_b = frozen_collection(coordinator.ROOT, "second")
    fake_runtime(coordinator, monkeypatch, jobs_a + jobs_b, problem=problem)
    out = coordinator.ROOT / "runs/program"
    with pytest.raises(RuntimeError, match="incomplete or apps"):
        coordinator.run([first, second], 600, out)
    assert read(out / "completion.json")["passed"] is False
    assert (coordinator.ROOT / "reports/program-cost-and-shutdown.json").exists()
    assert all((out / "collections" / p.name / "operational-coverage.json").exists() for p in (first, second))


def test_execution_tampering_rejected_before_admission(coordinator, monkeypatch):
    first, jobs_a = frozen_collection(coordinator.ROOT, "first")
    second, jobs_b = frozen_collection(coordinator.ROOT, "second")
    events, reservations = fake_runtime(coordinator, monkeypatch, jobs_a + jobs_b)
    (coordinator.ROOT / "executions/second/infra/launch_job.py").write_text("changed")
    with pytest.raises(ValueError, match="differs from its source archive"):
        coordinator.run([first, second], 600, coordinator.ROOT / "runs/program")
    assert not reservations and all(e[0] == "verify" for e in events)


def test_admission_failure_leaves_reviewable_plan_without_launch(coordinator, monkeypatch):
    first, jobs_a = frozen_collection(coordinator.ROOT, "first")
    second, jobs_b = frozen_collection(coordinator.ROOT, "second")
    events, _ = fake_runtime(coordinator, monkeypatch, jobs_a + jobs_b, problem="admission")
    out = coordinator.ROOT / "runs/program"
    with pytest.raises(RuntimeError, match="allowance exceeded"):
        coordinator.run([first, second], 600, out)
    assert (out / "program.json").exists() and not any(e[0] == "launch" for e in events)
    with pytest.raises(ValueError, match="existing artifacts"):
        coordinator.run([first, second], 600, out)
