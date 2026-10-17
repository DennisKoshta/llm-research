"""Two finite admissions for a separately declared backend-adapted control."""

import argparse
import datetime as dt
import hashlib
import importlib.util
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
from curiosity_study.artifacts import canonical, digest, read, write_once

NAME = "qwen25-pain-deterministic-control-v1"
PARENT = "qwen25-pain-profile-control-v1"
DIAGNOSTIC = "qwen25-pain-deterministic-diagnostic-v1"
ARCHIVE_SHA = "9c59998a8a162075a4adc201c4a374f821f76978b0c4ec518dce4630cd3f1fa7"
ORIGINAL_MANIFEST_SHA = "e87c70267be0b7813750dbef9d752f0855cf5b2db90e81f2381127d544b59a63"
DIAGNOSTIC_MANIFEST_SHA = "fa75488f5047854d22f92a0e822c85bac600a259cfc2d65b14286c6bfd0bed8d"
DIAGNOSTIC_SUMMARY_SHA = "595839f83fc2032e893bf4c34187d4cc5817bec8dd729417ddb54630ec3a7941"
DIAGNOSTIC_CHECK_SHA = "895b4f828358ecc37c470ffb0d7641ca26515781b19f933e69107c6318cc4c25"
RATE, TEARDOWN, CEILING = 0.00133632, 120, 5.0
WORKER = "pain_deterministic_control_worker.py"
POLICIES = {phase: dict(phase=phase, app_name="curiosity-study-pain-deterministic-control-" + phase + "-h10096",
    gpu="H100", cpu=2, memory_gib=96, wall_seconds=seconds, teardown_seconds=TEARDOWN,
    rate_usd_per_second=RATE, maximum_liability_usd=(seconds + TEARDOWN) * RATE,
    max_containers=1, retries=0) for phase, seconds in (("setup", 900), ("benchmark", 1800))}
FILES = {
    "launch_pain_deterministic_control.py": "infra/launch_pain_deterministic_control.py",
    WORKER: "scripts/" + WORKER,
    "audit_pain_deterministic_control.py": "scripts/audit_pain_deterministic_control.py",
    "analyze_pain_deterministic_control.py": "scripts/analyze_pain_deterministic_control.py",
    "diagnostic_reference_worker.py": "executions/" + DIAGNOSTIC + "/diagnostic/pain_profile_deterministic_diagnostic_worker.py",
    "budget.py": "infra/budget.py",
    "plan.md": "reports/pain-deterministic-control-v1-plan.md",
    "reference-manifest.json": "runs/" + DIAGNOSTIC + "/manifest.json",
    "reference-summary.json": "runs/" + DIAGNOSTIC + "/summary.json",
    "reference-independent-check.json": "reports/pain-deterministic-diagnostic-v1-independent-check.json",
    "test_pain_deterministic_control_worker.py": "tests/test_pain_deterministic_control_worker.py",
    "test_pain_deterministic_control_audit.py": "tests/test_pain_deterministic_control_audit.py",
    "test_pain_deterministic_control_controller.py": "tests/test_pain_deterministic_control_controller.py",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_files(root, inventory):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink() and not any(p.is_symlink() for p in root.rglob("*")),
            "Frozen tree contains a symlink")
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    require(actual == set(inventory), "Frozen file inventory changed")
    for name, expected in inventory.items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts and sha(root / name) == expected,
                "Frozen file identity changed")


def prepare(project):
    parent, reference = project / "runs" / PARENT, project / "runs" / DIAGNOSTIC
    original = read(parent / "manifest.json")
    reference_manifest = read(reference / "manifest.json")
    summary = read(reference / "summary.json")
    independent = read(project / FILES["reference-independent-check.json"])
    require(digest(original) == ORIGINAL_MANIFEST_SHA and sha(parent / "source.tar.gz") == ARCHIVE_SHA,
            "Original source identity changed")
    require(digest(reference_manifest) == DIAGNOSTIC_MANIFEST_SHA
            and sha(reference / "summary.json") == DIAGNOSTIC_SUMMARY_SHA
            and sha(project / FILES["reference-independent-check.json"]) == DIAGNOSTIC_CHECK_SHA
            and summary["passed"] is summary["complete"] is True and summary["completed_rows"] == 148
            and independent["passed"] is True and independent["manifest_sha256"] == DIAGNOSTIC_MANIFEST_SHA,
            "Deterministic prerequisite did not pass")
    run, execution = project / "runs" / NAME, project / "executions" / NAME
    require(not run.exists() and not execution.exists(), "New control is already prepared")
    run.mkdir(parents=True)
    execution.mkdir(parents=True)
    inventory = {}
    with tarfile.open(parent / "source.tar.gz") as archive:
        for member in archive.getmembers():
            path = Path(member.name)
            require(member.isfile() and not path.is_absolute() and ".." not in path.parts
                    and str(path) == member.name and member.name not in inventory, "Unsafe source archive")
            payload = archive.extractfile(member).read()
            destination = execution / "original" / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(payload)
            inventory[member.name] = hashlib.sha256(payload).hexdigest()
    diagnostic = execution / "diagnostic"
    diagnostic.mkdir()
    for destination, source in FILES.items():
        shutil.copyfile(project / source, diagnostic / destination)
    require(sha(diagnostic / "diagnostic_reference_worker.py") ==
            reference_manifest["diagnostic_files"]["pain_profile_deterministic_diagnostic_worker.py"],
            "Successful diagnostic helper bytes changed before import")
    spec = importlib.util.spec_from_file_location("reference_worker", diagnostic / "diagnostic_reference_worker.py")
    reference_worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference_worker)
    require(reference_worker.PRESET == reference_manifest["preset"], "Successful diagnostic preset changed")
    manifest = dict(schema_version=1, name=NAME, purpose="backend_adapted_behavioral_method_control",
        original_manifest=original, original_freeze=read(parent / "freeze.json"),
        original_source_archive_sha256=ARCHIVE_SHA, original_execution_files=inventory,
        diagnostic_files={p.name: sha(p) for p in diagnostic.iterdir() if p.is_file()},
        deterministic_reference=dict(manifest_sha256=DIAGNOSTIC_MANIFEST_SHA,
            summary_byte_sha256=sha(reference / "summary.json"),
            independent_check_byte_sha256=sha(project / FILES["reference-independent-check.json"])),
        preset={k: v for k, v in reference_worker.PRESET.items()
                if k not in {"wall_seconds", "execution_reserve_seconds"}},
        assigned_calls=dict(setup=21, benchmark=65), assigned_rows=dict(setup=612, benchmark=2020),
        resources=POLICIES, campaign_ceiling_usd=CEILING, benchmark_forecast_limit_seconds=1440,
        runtime_cache="/weights/pain-axis-positive-control-v1", original_campaign_remains_failed=True,
        scientific_rules_changed=False, automatic_retry_allowed=False)
    check_files(execution / "original", inventory)
    check_files(diagnostic, manifest["diagnostic_files"])
    write_once(run / "manifest.json", manifest)
    write_once(run / "freeze.json", dict(manifest_sha256=digest(manifest), execution_root=str(execution)))
    print(json.dumps(dict(prepared=str(run), manifest_sha256=digest(manifest),
                         total_maximum_liability_usd=sum(p["maximum_liability_usd"] for p in POLICIES.values()))))


def prior_evidence(run, manifest, phase):
    if phase == "setup":
        return {}
    names = ["admissions/setup.json", "original_execution/phases/setup/summary.json", "phases/setup/phase-execution.json",
             "phases/setup/audit.json", "launches/setup-outcome.json"]
    values = {name: read(run / name) for name in names}
    setup_claim = values[names[0]]
    central = run.parents[1] / "campaign_admissions" / NAME / "setup.json"
    require(setup_claim == read(central) and setup_claim["phase"] == "setup"
            and setup_claim["manifest_sha256"] == digest(manifest)
            and setup_claim["policy"] == POLICIES["setup"]
            and setup_claim["phase_consumed"] is True and setup_claim["retry_allowed"] is False
            and setup_claim["deadline"] == setup_claim["created"] + POLICIES["setup"]["wall_seconds"],
            "Global setup claim differs from the consumed local admission")
    execution_receipt = values[names[2]]
    require(execution_receipt["manifest_sha256"] == digest(manifest)
            and execution_receipt["phase"] == "setup"
            and execution_receipt["admission_sha256"] == digest(setup_claim)
            and execution_receipt["passed"] is execution_receipt["complete"] is True,
            "Setup execution receipt is not complete and bound to its admission")
    audit, summary = values[names[3]], values[names[1]]
    require(audit["passed"] is audit["complete"] is True and audit["phase"] == "setup"
            and audit["manifest_sha256"] == digest(manifest), "New setup independent audit did not pass")
    prediction = audit["benchmark_forecast_seconds"]
    require(type(prediction) in (float, int) and math.isfinite(prediction) and 0 < prediction <= 1440,
            "Measured forecast exceeds new benchmark wall allowance")
    require(summary["passed"] is summary["complete"] is True
            and summary["manifest_sha256"] == ORIGINAL_MANIFEST_SHA, "Original-engine setup failed")
    launch = values[names[4]]
    require(launch["all_stopped_zero_tasks"] is True and launch["export_error"] is None
            and launch["failure"] is None, "Setup transport/shutdown incomplete")
    # Root must rerun the frozen reporter before admission, binding every raw input.
    return {name: sha(run / name) for name in names}


def claim(project, manifest, initial, phase, now=None):
    require(phase in POLICIES and manifest["resources"] == POLICIES
            and manifest["campaign_ceiling_usd"] == CEILING, "Resource policy changed")
    now = time.time() if now is None else now
    require(initial["all_stopped_zero_tasks"] is True, "Project app is active")
    require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in
                [now, initial["available_compute_estimate_usd"], initial["phase_compute_estimate_usd"]]),
            "Invalid budget/time")
    policy = POLICIES[phase]
    remaining = sum(p["maximum_liability_usd"] for p in POLICIES.values()) if phase == "setup" else policy["maximum_liability_usd"]
    require(sum(p["maximum_liability_usd"] for p in POLICIES.values()) <= CEILING
            and remaining <= initial["available_compute_estimate_usd"] - 10,
            "Combined finite reservation exceeds allowance")
    evidence = prior_evidence(project / "runs" / NAME, manifest, phase)
    receipt = dict(phase=phase, manifest_sha256=digest(manifest), created=now,
        deadline=now + policy["wall_seconds"], policy=policy, phase_consumed=True, retry_allowed=False,
        maximum_liability_usd=policy["maximum_liability_usd"], prior_evidence=evidence,
        budget_timestamp=initial["timestamp_utc"], prior_phase_compute_estimate_usd=initial["phase_compute_estimate_usd"])
    destination = project / "campaign_admissions" / NAME / (phase + ".json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        stream.write(canonical(receipt) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    write_once(project / "runs" / NAME / "admissions" / (phase + ".json"), receipt)
    return receipt


def run_cloud(manifest, admission, prior, image_id):
    volume = modal.Volume.from_name("curiosity-study-artifacts", create_if_missing=False)
    volume.reload()
    phase, run = admission["phase"], Path("/artifacts") / NAME
    transport = run / "transport" / phase
    require(not transport.exists(), "Remote phase consumed; no retry")
    transport.mkdir(parents=True)
    write_once(run / "manifest.json", manifest)
    write_once(run / "admissions" / (phase + ".json"), admission)
    for name, payload in prior.items():
        require(name in admission["prior_evidence"] and not Path(name).is_absolute() and ".." not in Path(name).parts,
                "Unbound prior evidence")
        require(isinstance(payload, bytes) and hashlib.sha256(payload).hexdigest() == admission["prior_evidence"][name],
                "Prior evidence transport bytes changed")
        destination = run / name
        if destination.exists():
            require(destination.read_bytes() == payload, "Existing prior evidence bytes changed")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(payload)
        require(sha(run / name) == admission["prior_evidence"][name], "Prior evidence bytes changed")
    write_once(transport / "started.json", dict(started=time.time(), image_id=image_id))
    volume.commit()
    process, error = None, None
    try:
        require(admission["manifest_sha256"] == digest(manifest) and admission["policy"] == POLICIES[phase]
                and admission["phase_consumed"] is True and admission["retry_allowed"] is False
                and admission["deadline"] - admission["created"] == POLICIES[phase]["wall_seconds"]
                and admission["created"] <= time.time() < admission["deadline"] - 60, "Invalid admission")
        check_files(Path("/original"), manifest["original_execution_files"])
        check_files(Path("/diagnostic"), manifest["diagnostic_files"])
        cache = Path(manifest["runtime_cache"])
        environment = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH="/original/src:/original/scripts:/diagnostic", TOKENIZERS_PARALLELISM="false",
            HF_HOME=str(cache / "huggingface"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
            HF_HUB_DISABLE_TELEMETRY="1", PAIN_PROFILE_IMAGE_ID=image_id)
        require((cache / "runtime/bin/python").is_file(), "Missing cached runtime")
        with (transport / "child.log").open("xb", buffering=0) as log:
            process = subprocess.Popen([str(cache / "runtime/bin/python"), "/diagnostic/" + WORKER,
                "--original-root", "/original", "--run", str(run), "--cache", str(cache), "--phase", phase],
                stdout=log, stderr=subprocess.STDOUT, env=environment, start_new_session=True)
            previous = None
            while process.poll() is None:
                require(time.time() < admission["deadline"] - 30, "Child phase deadline")
                marker = transport / "checkpoint.request"
                current = marker.read_text() if marker.exists() else None
                if current is not None and current != previous:
                    volume.commit()
                    previous = current
                time.sleep(0.5)
            require(process.returncode == 0, "Child failed; inspect retained records")
    except Exception as caught:
        error = dict(type=type(caught).__name__, message=str(caught))
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
        write_once(transport / "finished.json", dict(finished=time.time(), error=error))
        volume.commit()
    receipt_path = run / "phases" / phase / "phase-execution.json"
    receipt = read(receipt_path) if receipt_path.exists() else None
    return dict(passed=error is None and bool(receipt and receipt.get("passed")), error=error, receipt=receipt)


def collect(project):
    volume = modal.Volume.from_name("curiosity-study-artifacts", create_if_missing=False)
    count = 0
    for entry in volume.iterdir(NAME, recursive=True):
        if int(entry.type) != 1:
            continue
        name = Path(entry.path.lstrip("/"))
        require(name.parts[0] == NAME and ".." not in name.parts, "Unsafe remote path")
        destination, payload = project / "runs" / name, b"".join(volume.read_file(entry.path))
        if destination.exists():
            require(destination.read_bytes() == payload, "Remote/local artifact mismatch")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(payload)
        count += 1
    return count


def launch(project, phase):
    run = project / "runs" / NAME
    manifest, freeze = read(run / "manifest.json"), read(run / "freeze.json")
    execution = Path(freeze["execution_root"])
    require(execution == project / "executions" / NAME and digest(manifest) == freeze["manifest_sha256"]
            and Path(__file__).resolve() == execution / "diagnostic" / Path(__file__).name,
            "Use frozen launcher")
    check_files(execution / "original", manifest["original_execution_files"])
    check_files(execution / "diagnostic", manifest["diagnostic_files"])
    if phase == "benchmark":
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=":".join(
            str(execution / path) for path in ("original/src", "original/scripts", "diagnostic")))
        subprocess.run([str(project / ".pain-runtime-venv/bin/python"),
            str(execution / "diagnostic/audit_pain_deterministic_control.py"),
            "--run", str(run), "--phase", "setup", "--verify-existing"],
            check=True, timeout=120, env=environment)
    budget.ROOT = project
    initial = budget.snapshot(timeout=20)
    admission = claim(project, manifest, initial, phase)
    prior = {name: (run / name).read_bytes() for name in admission["prior_evidence"]}
    policy, cli = POLICIES[phase], project / ".venv/bin/modal"
    image = (modal.Image.debian_slim(python_version="3.12").pip_install("uv==0.12.18")
        .env({"PYTHONPATH": "/original/src:/original/infra:/diagnostic", "PYTHONDONTWRITEBYTECODE": "1"})
        .add_local_dir(execution / "original", "/original", ignore=lambda p: "__pycache__" in p.parts)
        .add_local_dir(execution / "diagnostic", "/diagnostic", ignore=lambda p: "__pycache__" in p.parts))
    app = modal.App(policy["app_name"])
    job = app.function(image=image, gpu="H100!", cpu=(2, 2), memory=(98304, 98304),
        volumes={"/weights": modal.Volume.from_name("curiosity-study-weights", create_if_missing=False),
                 "/artifacts": modal.Volume.from_name("curiosity-study-artifacts", create_if_missing=False)},
        max_containers=1, min_containers=0, buffer_containers=0, scaledown_window=2, retries=0,
        timeout=policy["wall_seconds"], startup_timeout=600, enable_memory_snapshot=False, include_source=False)(run_cloud)
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
                apps = json.loads(subprocess.check_output([str(cli), "app", "list", "--json"], text=True, timeout=20))
                for row in apps:
                    if row.get("Description") == policy["app_name"] and row["State"] != "stopped" and (
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
            write_once(run / "launches" / (phase + ".json"), dict(app_id=app_id,
                admission_sha256=digest(admission), image_id=image.object_id, started=time.time()))
            require(time.time() < admission["deadline"] - 60, "Startup consumed phase wall")
            call = job.spawn(manifest, admission, prior, image.object_id)
            while result is None:
                remaining = admission["deadline"] - time.time()
                require(remaining > 0, "Fixed phase deadline")
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
        state, shutdown_deadline = None, time.time() + 90
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
        write_once(project / "reports" / f"{NAME}-{phase}-cost-and-shutdown.json", state)
        exported, export_error = None, None
        try:
            exported = collect(project)
        except Exception as caught:
            export_error = str(caught)
        outcome = dict(phase=phase, app_id=app_id, result=result, failure=failure, stop_errors=stop_errors,
            all_stopped_zero_tasks=state["all_stopped_zero_tasks"], exported_files=exported,
            export_error=export_error, finished=time.time())
        write_once(run / "launches" / (phase + "-outcome.json"), outcome)
        print(json.dumps({k: v for k, v in outcome.items() if k != "result"}, indent=2), flush=True)
    if failure or export_error or not state["all_stopped_zero_tasks"] or not result or not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("action", choices=["prepare", "launch", "collect"])
    parser.add_argument("--phase", choices=list(POLICIES))
    args = parser.parse_args()
    if args.action == "launch":
        require(args.phase is not None, "Phase required")
        launch(args.project_root.resolve(), args.phase)
    elif args.action == "prepare":
        prepare(args.project_root.resolve())
    else:
        print(collect(args.project_root.resolve()))
