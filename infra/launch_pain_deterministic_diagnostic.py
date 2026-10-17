"""One frozen engineering diagnostic; no retry or behavioral benchmark admission."""

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tarfile
import threading
import time

import budget
import modal
from curiosity_study.artifacts import digest, read, write_once

NAME = "qwen25-pain-deterministic-diagnostic-v1"
PARENT = "qwen25-pain-profile-control-v1"
APP = "curiosity-study-pain-deterministic-h10096"
ARCHIVE_SHA = "9c59998a8a162075a4adc201c4a374f821f76978b0c4ec518dce4630cd3f1fa7"
MANIFEST_SHA = "e87c70267be0b7813750dbef9d752f0855cf5b2db90e81f2381127d544b59a63"
RATE = 0.00133632
WALL = 900
TEARDOWN = 120
CEILING = 2.0
WORKER = "pain_profile_deterministic_diagnostic_worker.py"
FILES = {
    "launch_pain_deterministic_diagnostic.py": "infra/launch_pain_deterministic_diagnostic.py",
    WORKER: "scripts/" + WORKER,
    "budget.py": "infra/budget.py",
    "plan.md": "reports/pain-profile-deterministic-diagnostic-v1-plan.md",
    "parent-repeat-diagnosis.json": "reports/pain-profile-baseline-repeat-diagnosis.json",
    "parent-audit-failure.json": "reports/pain-profile-setup-audit-attempt-failure.json",
    "test_controller.py": "tests/test_pain_deterministic_controller.py",
    "test_worker.py": "tests/test_pain_profile_deterministic_diagnostic_worker.py",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_files(root, inventory):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink()
            and not any(p.is_symlink() for p in root.rglob("*")), "Frozen tree contains a symlink")
    actual = {str(p.relative_to(root)) for p in root.rglob("*")
              if p.is_file() and "__pycache__" not in p.parts}
    require(actual == set(inventory), "Frozen file inventory changed")
    for name, expected in inventory.items():
        path = root / name
        require(not Path(name).is_absolute() and ".." not in Path(name).parts
                and path.is_file() and not path.is_symlink() and sha(path) == expected,
                "Frozen file bytes changed: " + name)


def prepare(project):
    """Local only. Copy the original immutable archive and bind the new preset."""
    import importlib.util

    parent = project / "runs" / PARENT
    original = read(parent / "manifest.json")
    require(digest(original) == MANIFEST_SHA and sha(parent / "source.tar.gz") == ARCHIVE_SHA,
            "Original campaign identity changed")
    run, execution = project / "runs" / NAME, project / "executions" / NAME
    require(not run.exists() and not execution.exists(), "Diagnostic is already prepared")
    execution.mkdir(parents=True)
    run.mkdir(parents=True)
    inventory = {}
    original_root = execution / "original"
    with tarfile.open(parent / "source.tar.gz") as archive:
        for member in archive.getmembers():
            name = Path(member.name)
            require(member.isfile() and not name.is_absolute() and ".." not in name.parts
                    and str(name) == member.name and member.name not in inventory,
                    "Unsafe original archive entry")
            payload = archive.extractfile(member).read()
            destination = original_root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(payload)
            inventory[member.name] = hashlib.sha256(payload).hexdigest()
    diagnostic = execution / "diagnostic"
    diagnostic.mkdir()
    for destination, source in FILES.items():
        shutil.copyfile(project / source, diagnostic / destination)
    spec = importlib.util.spec_from_file_location("diagnostic_worker", diagnostic / WORKER)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    manifest = dict(schema_version=1, name=NAME, purpose="engineering_only_no_behavioral_qualification",
                    original_manifest=original, original_freeze=read(parent / "freeze.json"),
                    original_execution_files=inventory, original_source_archive_sha256=ARCHIVE_SHA,
                    diagnostic_files={p.name: sha(p) for p in diagnostic.iterdir() if p.is_file()},
                    preset=worker.PRESET, call_plan=worker.call_plan(original),
                    assigned_calls=5, assigned_rows=148,
                    resources=dict(app_name=APP, gpu="H100", cpu=2, memory_gib=96,
                                   wall_seconds=WALL, teardown_seconds=TEARDOWN,
                                   rate_usd_per_second=RATE, ceiling_usd=CEILING,
                                   maximum_liability_usd=(WALL + TEARDOWN) * RATE,
                                   max_containers=1, retries=0),
                    runtime_cache="/weights/pain-axis-positive-control-v1",
                    original_campaign_remains_failed=True, behavioral_benchmark_admitted=False)
    check_files(original_root, inventory)
    check_files(diagnostic, manifest["diagnostic_files"])
    write_once(run / "manifest.json", manifest)
    write_once(run / "freeze.json", dict(manifest_sha256=digest(manifest),
               execution_root=str(execution), original_source_archive_sha256=ARCHIVE_SHA))
    print(json.dumps(dict(prepared=str(run), manifest_sha256=digest(manifest),
                         maximum_liability_usd=manifest["resources"]["maximum_liability_usd"])))


def claim(project, manifest, initial, now=None):
    now = time.time() if now is None else now
    require(initial["all_stopped_zero_tasks"] is True, "A project app is still active")
    require(all(type(value) in (int, float) and math.isfinite(value) and value >= 0
                for value in (now, initial["available_compute_estimate_usd"],
                              initial["phase_compute_estimate_usd"])), "Invalid admission time or budget")
    liability = (WALL + TEARDOWN) * RATE
    require(liability <= CEILING and liability <= initial["available_compute_estimate_usd"] - 10,
            "Insufficient fresh allowance including retained reserve")
    require(manifest["resources"] == dict(app_name=APP, gpu="H100", cpu=2, memory_gib=96,
            wall_seconds=WALL, teardown_seconds=TEARDOWN, rate_usd_per_second=RATE,
            ceiling_usd=CEILING, maximum_liability_usd=liability, max_containers=1, retries=0),
            "Diagnostic resource policy changed")
    admission = dict(manifest_sha256=digest(manifest), created=now, deadline=now + WALL,
                     phase_consumed=True, retry_allowed=False, maximum_liability_usd=liability,
                     budget_timestamp=initial["timestamp_utc"],
                     prior_phase_compute_estimate_usd=initial["phase_compute_estimate_usd"])
    path = project / "campaign_admissions" / NAME / "admission.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(admission, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    write_once(project / "runs" / NAME / "admission.json", admission)
    return admission


def run_cloud(manifest, admission, image_id):
    """Supervise one pinned subprocess, committing each complete call."""
    artifacts = modal.Volume.from_name("curiosity-study-artifacts", create_if_missing=False)
    artifacts.reload()
    run = Path("/artifacts") / NAME
    require(not run.exists(), "Diagnostic remote path already consumed")
    run.mkdir()
    write_once(run / "manifest.json", manifest)
    write_once(run / "admission.json", admission)
    write_once(run / "cloud-started.json", dict(started=time.time(), image_id=image_id))
    artifacts.commit()
    process, error = None, None
    try:
        require(digest(manifest) == admission["manifest_sha256"] and admission["phase_consumed"] is True
                and admission["retry_allowed"] is False and admission["deadline"] - admission["created"] == WALL
                and admission["created"] <= time.time() < admission["deadline"] - 60,
                "Invalid or expired admission")
        check_files(Path("/original"), manifest["original_execution_files"])
        check_files(Path("/diagnostic"), manifest["diagnostic_files"])
        cache = Path(manifest["runtime_cache"])
        environment = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8",
                           PYTHONPATH="/original/src:/original/scripts:/diagnostic",
                           PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false",
                           HF_HOME=str(cache / "huggingface"), HF_HUB_DISABLE_TELEMETRY="1",
                           HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        require((cache / "runtime/bin/python").is_file(), "Pinned cached runtime missing")
        with (run / "child.log").open("xb", buffering=0) as output:
            process = subprocess.Popen([str(cache / "runtime/bin/python"), "/diagnostic/" + WORKER,
                       "--original-root", "/original", "--run", str(run), "--cache", str(cache)],
                       stdout=output, stderr=subprocess.STDOUT, env=environment, start_new_session=True)
            previous = None
            while process.poll() is None:
                require(time.time() < admission["deadline"] - 30, "Diagnostic child deadline")
                marker = run / "checkpoint.request"
                current = marker.read_text() if marker.exists() else None
                if current is not None and current != previous:
                    artifacts.commit()
                    previous = current
                time.sleep(0.5)
            require(process.returncode == 0, "Diagnostic child failed; inspect retained records")
    except Exception as caught:
        error = dict(type=type(caught).__name__, message=str(caught))
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
        write_once(run / "cloud-finished.json", dict(finished=time.time(), error=error))
        artifacts.commit()
    summary = read(run / "summary.json") if (run / "summary.json").exists() else None
    return dict(passed=error is None and bool(summary and summary.get("passed")), error=error, summary=summary)


def collect(project):
    artifacts = modal.Volume.from_name("curiosity-study-artifacts", create_if_missing=False)
    count = 0
    for entry in artifacts.iterdir(NAME, recursive=True):
        if int(entry.type) != 1:
            continue
        name = Path(entry.path.lstrip("/"))
        require(name.parts[0] == NAME and ".." not in name.parts, "Unsafe remote artifact")
        destination = project / "runs" / name
        payload = b"".join(artifacts.read_file(entry.path))
        if destination.exists():
            require(destination.read_bytes() == payload, "Different local artifact exists")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(payload)
        count += 1
    return count


def collection_coverage(project, manifest):
    """Retain every assigned slot even if the process died before its summary."""
    run = project / "runs" / NAME
    rows, issues = [], []
    for call in manifest["call_plan"]:
        path = run / "calls" / f"{call['call_index']:04d}.json"
        records, attempted = {}, path.exists()
        if attempted:
            try:
                envelope = read(path)
                require(envelope["manifest_sha256"] == digest(manifest)
                        and envelope["call_spec"] == call, "Call identity mismatch")
                result = envelope["engine_result"]
                raw = result["records"] if result else []
                require(all(row["trial_id"] in call["trial_ids"] for row in raw), "Unexpected observed trial")
                records = {row["trial_id"]: digest(row) for row in raw}
                require(len(records) == len(raw), "Duplicate observed trial")
            except Exception as caught:
                issues.append(dict(call_id=call["call_id"], error=str(caught)))
                records = {}
        rows.extend(dict(call_id=call["call_id"], trial_id=trial,
                         status="completed" if trial in records else "execution_error" if attempted else "missing",
                         raw_record_sha256=records.get(trial)) for trial in call["trial_ids"])
    require(len(rows) == 148 and len({row["trial_id"] for row in rows}) == 148,
            "Diagnostic denominator changed")
    receipt = dict(manifest_sha256=digest(manifest), assigned_rows=148, rows=rows,
                   issues=issues, independent_numerical_audit=False,
                   **{status + "_rows": sum(row["status"] == status for row in rows)
                      for status in ("completed", "execution_error", "missing")})
    write_once(run / "collection-coverage.json", receipt)
    return receipt


def launch(project):
    run = project / "runs" / NAME
    manifest, freeze = read(run / "manifest.json"), read(run / "freeze.json")
    execution = Path(freeze["execution_root"])
    require(execution == project / "executions" / NAME
            and Path(__file__).resolve() == execution / "diagnostic" / Path(__file__).name
            and digest(manifest) == freeze["manifest_sha256"], "Use the fixed frozen launcher")
    check_files(execution / "original", manifest["original_execution_files"])
    check_files(execution / "diagnostic", manifest["diagnostic_files"])
    budget.ROOT = project
    initial = budget.snapshot(timeout=20)
    admission = claim(project, manifest, initial)
    cli = project / ".venv/bin/modal"
    image = (modal.Image.debian_slim(python_version="3.12").pip_install("uv==0.12.18")
             .env({"PYTHONPATH": "/original/src:/original/infra:/diagnostic", "PYTHONDONTWRITEBYTECODE": "1"})
             .add_local_dir(execution / "original", "/original", ignore=lambda p: "__pycache__" in p.parts)
             .add_local_dir(execution / "diagnostic", "/diagnostic", ignore=lambda p: "__pycache__" in p.parts))
    app = modal.App(APP)
    job = app.function(image=image, gpu="H100!", cpu=(2, 2), memory=(98304, 98304),
          volumes={"/weights": modal.Volume.from_name("curiosity-study-weights", create_if_missing=False),
                   "/artifacts": modal.Volume.from_name("curiosity-study-artifacts", create_if_missing=False)},
          max_containers=1, min_containers=0, buffer_containers=0, scaledown_window=2,
          retries=0, timeout=WALL, startup_timeout=600, enable_memory_snapshot=False,
          include_source=False)(run_cloud)
    done, stop_errors = threading.Event(), []

    def stop(app_id):
        if app_id:
            subprocess.run([str(cli), "app", "stop", app_id], check=True, timeout=20,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def watchdog():
        while not done.wait(max(0.1, min(2, admission["deadline"] - time.time()))):
            if time.time() < admission["deadline"]:
                continue
            try:
                apps = json.loads(subprocess.check_output([str(cli), "app", "list", "--json"],
                                                          text=True, timeout=20))
                for row in apps:
                    if row.get("Description") == APP and row["State"] != "stopped" and (
                        dt.datetime.fromisoformat(row["Created at"]).timestamp() >= admission["created"] - 2
                    ):
                        stop(row["App ID"])
            except Exception as caught:
                stop_errors.append(str(caught))

    watcher = threading.Thread(target=watchdog, daemon=True)
    watcher.start()
    call, app_id, result, failure = None, None, None, None
    try:
        with modal.enable_output(), app.run():
            app_id = app.app_id
            write_once(run / "launch.json", dict(app_id=app_id, admission_sha256=digest(admission),
                        image_id=image.object_id, started=time.time()))
            require(time.time() < admission["deadline"] - 60, "Image startup consumed diagnostic wall")
            call = job.spawn(manifest, admission, image.object_id)
            while result is None:
                remaining = admission["deadline"] - time.time()
                require(remaining > 0, "Diagnostic reached fixed wall deadline")
                try:
                    result = call.get(timeout=min(15, remaining))
                except TimeoutError:
                    continue
    except Exception as caught:
        failure = dict(type=type(caught).__name__, message=str(caught))
    finally:
        app_id = app_id or app.app_id
        if call is not None:
            try:
                call.cancel(terminate_containers=True)
            except Exception as caught:
                stop_errors.append(str(caught))
        try:
            stop(app_id)
        except Exception as caught:
            stop_errors.append(str(caught))
        shutdown_deadline, state = time.time() + 90, None
        while time.time() < shutdown_deadline:
            try:
                state = budget.snapshot(timeout=20)
                if state["all_stopped_zero_tasks"]:
                    break
            except Exception as caught:
                stop_errors.append(str(caught))
            time.sleep(3)
        done.set()
        watcher.join(timeout=25)
        if state is None:
            state = dict(all_stopped_zero_tasks=False, cost_unknown=True, app_id=app_id)
        write_once(project / "reports" / (NAME + "-cost-and-shutdown.json"), state)
        exported, export_error = None, None
        try:
            exported = collect(project)
        except Exception as caught:
            export_error = str(caught)
        coverage = collection_coverage(project, manifest)
        outcome = dict(app_id=app_id, result=result, failure=failure, stop_errors=stop_errors,
                       all_stopped_zero_tasks=state["all_stopped_zero_tasks"], exported_files=exported,
                       export_error=export_error, collection_coverage_sha256=digest(coverage), finished=time.time())
        write_once(run / "launch-outcome.json", outcome)
        print(json.dumps({k: v for k, v in outcome.items() if k != "result"}, indent=2), flush=True)
    if failure or export_error or not state["all_stopped_zero_tasks"] or not result or not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("action", choices=["prepare", "launch", "collect"])
    args = parser.parse_args()
    {"prepare": prepare, "launch": launch, "collect": collect}[args.action](args.project_root.resolve())
