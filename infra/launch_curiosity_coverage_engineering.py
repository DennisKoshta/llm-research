"""One fixed cache-only coverage engineering attempt; no scientific admission."""

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
import sys
import tarfile
import tempfile
import threading
import time

import modal

NAME = "qwen32-curiosity-coverage-engineering-v1"
PARENT = "qwen32-curiosity-suppression-screen-v1-admission2"
APP = "curiosity-study-coverage-engineering-h200"
ARCHIVE_SHA = "4b82bdc563a3b257fd1b5e0302f5889cb6e14e30da9fcb950cc7f3039eb62ca4"
MANIFEST_SHA = "535f3c1237a8b876fa1059d3f9949f178a55b5627e2be412008ef08cbc90b02f"
SOURCE_SHA = "1fd4079ed60fbd0169f702ab9c8f99a75d170450dfbd364bc4cb877e27c18f9a"
RATE, WALL, TEARDOWN, CEILING = 0.00139376, 3400, 120, 5.0
WORKER = "curiosity_coverage_engineering_worker.py"
MATERIALS = "prepare_curiosity_coverage_bridge.py"
OVERLAYS = ("state_backend.py", "coverage_engineering.py")
FILES = {
    "launch_curiosity_coverage_engineering.py": "infra/launch_curiosity_coverage_engineering.py",
    WORKER: "scripts/" + WORKER,
    MATERIALS: "scripts/" + MATERIALS,
    "plan.md": "reports/curiosity-coverage-bridge-v1-plan.md",
    "test_controller.py": "tests/test_coverage_engineering_controller.py",
    "test_worker.py": "tests/test_coverage_engineering_worker.py",
    "test_materials.py": "tests/test_prepare_curiosity_coverage_bridge.py",
    "audit_coverage_engineering.py": "scripts/audit_coverage_engineering.py",
    "test_audit.py": "tests/test_coverage_engineering_audit.py",
    "test_collector.py": "tests/test_coverage_engineering.py",
    "test_backend.py": "tests/test_state_backend_prefix_coverage.py",
    "accounting/budget.py": "infra/budget.py",
    **{
        "accounting/curiosity_study/" + name: "src/curiosity_study/" + name
        for name in (
            "__init__.py",
            "artifacts.py",
            "resource_limits.py",
            "curiosity_resource_limits.py",
            "pain_profile_resources.py",
        )
    },
}


def inventory(root):
    return {
        str(p.relative_to(root)): sha(p)
        for p in Path(root).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }


def build_runtime(project, original_root, destination):
    """Copy the immutable package and declare the only two prospective overlays."""
    source = Path(original_root) / "src/curiosity_study"
    shutil.copytree(source, destination / "curiosity_study", ignore=shutil.ignore_patterns("__pycache__"))
    delta = {}
    for name in OVERLAYS:
        old = source / name
        target = destination / "curiosity_study" / name
        new = Path(project) / "src/curiosity_study" / name
        require(new.is_file() and not new.is_symlink(), "Missing reviewed runtime overlay")
        delta[name] = dict(
            original_byte_sha256=sha(old) if old.exists() else None, replacement_byte_sha256=sha(new)
        )
        shutil.copyfile(new, target)
    return inventory(destination), delta


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (canonical(value) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def resources():
    return dict(
        app_name=APP,
        gpu="H200",
        cpu=2,
        memory_gib=48,
        wall_seconds=WALL,
        teardown_seconds=TEARDOWN,
        rate_usd_per_second=RATE,
        ceiling_usd=CEILING,
        maximum_liability_usd=(WALL + TEARDOWN) * RATE,
        max_containers=1,
        retries=0,
    )


def check_files(root, inventory):
    root = Path(root)
    require(
        root.is_dir() and not root.is_symlink() and not any(p.is_symlink() for p in root.rglob("*")),
        "Frozen tree contains a symlink",
    )
    actual = {
        str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts
    }
    require(actual == set(inventory), "Frozen file inventory changed")
    for name, checksum in inventory.items():
        relative = Path(name)
        require(
            not relative.is_absolute()
            and ".." not in relative.parts
            and str(relative) == name
            and sha(root / relative) == checksum,
            "Frozen file bytes changed: " + name,
        )


def load_worker(wrapper, manifest=None):
    if manifest is not None:
        check_files(wrapper, manifest["wrapper_files"])
    spec = importlib.util.spec_from_file_location("_coverage_worker", Path(wrapper) / WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_original(archive_path, destination):
    require(sha(archive_path) == ARCHIVE_SHA, "Original archive changed")
    inventory = {}
    with tarfile.open(archive_path) as archive:
        members = archive.getmembers()
        for member in members:
            relative = Path(member.name)
            require(
                member.isfile()
                and not relative.is_absolute()
                and ".." not in relative.parts
                and str(relative) == member.name
                and member.name not in inventory,
                "Unsafe original archive entry",
            )
            inventory[member.name] = None
        for member in members:
            payload = archive.extractfile(member).read()
            path = destination / member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(payload)
            inventory[member.name] = hashlib.sha256(payload).hexdigest()
    return inventory


def prepare(project):
    """Local-only freeze. Root invokes after independent review; no model inference."""
    project = Path(project).resolve()
    parent, run, execution = project / "runs" / PARENT, project / "runs" / NAME, project / "executions" / NAME
    require(not run.exists() and not execution.exists(), "Engineering is already prepared")
    original, freeze = read(parent / "manifest.json"), read(parent / "freeze.json")
    require(
        digest(original) == freeze["manifest_sha256"] == MANIFEST_SHA
        and original["source_sha256"] == freeze["source_sha256"] == SOURCE_SHA
        and freeze["archive_sha256"] == ARCHIVE_SHA,
        "Original manifest/source identity differs",
    )
    require(
        all((project / name).is_file() and not (project / name).is_symlink() for name in FILES.values()),
        "A required reviewed file is unavailable",
    )
    # Nothing is published until extraction, all materials and tokenizer checks succeed.
    with tempfile.TemporaryDirectory(prefix="coverage-freeze-", dir=project) as temp:
        staging = Path(temp)
        original_root, wrapper, runtime = staging / "original", staging / "engineering", staging / "runtime"
        original_files = extract_original(parent / "source.tar.gz", original_root)
        runtime_files, delta = build_runtime(project, original_root, runtime)
        for destination, source in FILES.items():
            path = wrapper / destination
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / source, path)
        worker = load_worker(wrapper)
        source = original_root / "src/curiosity_study"
        require(
            digest({str(p.relative_to(source)): p.read_text() for p in sorted(source.rglob("*.py"))})
            == SOURCE_SHA,
            "Original Python source inventory differs",
        )
        # Load only the verified derivative package; no archived entrypoint is executed.
        modules = worker.load_runtime(runtime)
        spec = importlib.util.spec_from_file_location("_coverage_materials_prepare", wrapper / MATERIALS)
        materials = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(materials)
        plan = materials.build_plan(project)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            plan["model"]["model_id"],
            revision=plan["model"]["revision"],
            local_files_only=True,
            trust_remote_code=False,
        )
        preflight = worker.tokenizer_preflight(plan, tokenizer, modules["voluntary_resolution"])
        wrapper_files = inventory(wrapper)
        runtime_source = runtime / "curiosity_study"
        runtime_sha = digest(
            {str(p.relative_to(runtime_source)): p.read_text() for p in sorted(runtime_source.rglob("*.py"))}
        )
        manifest = dict(
            schema_version=1,
            name=NAME,
            assay="curiosity_coverage_engineering",
            original_manifest=original,
            original_freeze=freeze,
            source_sha256=SOURCE_SHA,
            original_source_archive_sha256=ARCHIVE_SHA,
            original_execution_files=original_files,
            runtime_files=runtime_files,
            runtime_source_sha256=runtime_sha,
            runtime_delta=delta,
            wrapper_files=wrapper_files,
            plan=plan,
            plan_sha256=digest(plan),
            preflight=preflight,
            preflight_sha256=digest(preflight),
            resources=resources(),
            engineering_only=True,
            scientific_admission=False,
            causal_admission=False,
        )
        worker.validate_manifest(manifest)
        worker.verify_sources(manifest, original_root, runtime, wrapper)
        check_files(original_root, original_files)
        check_files(runtime, runtime_files)
        check_files(wrapper, wrapper_files)
        staged_run = staging / "run"
        write_once(staged_run / "manifest.json", manifest)
        write_once(
            staged_run / "freeze.json",
            dict(
                manifest_sha256=digest(manifest),
                execution_root=str(execution),
                original_source_archive_sha256=ARCHIVE_SHA,
                wrapper_files_sha256=digest(wrapper_files),
                runtime_files_sha256=digest(runtime_files),
                runtime_source_sha256=runtime_sha,
            ),
        )
        execution.parent.mkdir(exist_ok=True)
        run.parent.mkdir(exist_ok=True)
        require(not execution.exists() and not run.exists(), "Engineering appeared during preparation")
        os.rename(staging, execution)
        os.rename(execution / "run", run)
    return dict(run=str(run), manifest_sha256=digest(manifest), assigned_slots=512, admitted=False)


def claim(project, manifest, initial, now=None):
    now = time.time() if now is None else now
    require(
        manifest["name"] == NAME
        and manifest["resources"] == resources()
        and manifest["engineering_only"] is True
        and manifest["causal_admission"] is False
        and manifest["scientific_admission"] is False,
        "Fixed engineering policy differs",
    )
    require(initial["all_stopped_zero_tasks"] is True, "A project app remains active")
    require(
        all(
            type(value) in (int, float) and math.isfinite(value) and value >= 0
            for value in (
                now,
                initial["available_compute_estimate_usd"],
                initial["phase_compute_estimate_usd"],
            )
        ),
        "Invalid admission time/budget",
    )
    liability = resources()["maximum_liability_usd"]
    require(
        liability <= CEILING and liability <= initial["available_compute_estimate_usd"] - 10,
        "Insufficient allowance including retained reserve",
    )
    receipt = dict(
        manifest_sha256=digest(manifest),
        created=now,
        deadline=now + WALL,
        phase_consumed=True,
        retry_allowed=False,
        maximum_liability_usd=liability,
        budget_timestamp=initial["timestamp_utc"],
        prior_phase_compute_estimate_usd=initial["phase_compute_estimate_usd"],
    )
    write_once(Path(project) / "campaign_admissions" / NAME / "admission.json", receipt)
    write_once(Path(project) / "runs" / NAME / "admission.json", receipt)
    return receipt


def snapshot(project, wrapper):
    # Isolate the frozen current accounting helpers from the older model package.
    code = "import json,budget; print(json.dumps(budget.snapshot(timeout=20)))"
    env = dict(
        os.environ,
        PYTHONPATH=str(Path(wrapper) / "accounting"),
        PYTHONDONTWRITEBYTECODE="1",
        CURIOSITY_BUDGET_ROOT=str(project),
    )
    return json.loads(
        subprocess.check_output(
            [str(Path(project) / ".venv/bin/python"), "-c", code], env=env, cwd=wrapper, text=True, timeout=25
        )
    )


def run_cloud(manifest, admission, image_id):
    artifacts = modal.Volume.from_name("curiosity-study-artifacts", create_if_missing=False)
    artifacts.reload()
    run = Path("/artifacts") / NAME
    require(not run.exists(), "Remote namespace already consumed")
    run.mkdir()
    write_once(run / "manifest.json", manifest)
    write_once(run / "admission.json", admission)
    write_once(run / "cloud-started.json", dict(started=time.time(), image_id=image_id))
    commits = []

    def commit():
        started = time.monotonic()
        artifacts.commit()
        commits.append(time.monotonic() - started)

    commit()
    process, error = None, None
    try:
        worker = load_worker(Path("/engineering"), manifest)
        worker.validate_manifest(manifest)
        worker.validate_admission(manifest, admission, time.time())
        worker.verify_sources(manifest, Path("/original"), Path("/runtime"), Path("/engineering"))
        env = dict(
            os.environ,
            PYTHONPATH="/runtime:/engineering",
            PYTHONDONTWRITEBYTECODE="1",
            HF_HOME="/weights/huggingface",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            HF_HUB_DISABLE_TELEMETRY="1",
            TOKENIZERS_PARALLELISM="false",
        )
        with (run / "child.log").open("xb", buffering=0) as output:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "/engineering/" + WORKER,
                    "--original-root",
                    "/original",
                    "--runtime",
                    "/runtime",
                    "--wrapper",
                    "/engineering",
                    "--run",
                    str(run),
                ],
                stdout=output,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            previous = None
            while process.poll() is None:
                require(time.time() < admission["deadline"] - 30, "Fixed child wall deadline")
                marker = run / "checkpoint.request"
                current = marker.read_text() if marker.exists() else None
                if current is not None and current != previous:
                    commit()
                    previous = current
                time.sleep(0.5)
            require(process.returncode == 0, "Engineering execution incomplete; see retained records")
    except Exception as caught:
        error = dict(type=type(caught).__name__, message=str(caught))
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
        write_once(
            run / "cloud-finished.json",
            dict(
                finished=time.time(),
                error=error,
                checkpoint_commit_seconds=commits,
                checkpoint_commit_total_seconds=sum(commits),
                final_commit_timing_excluded=True,
            ),
        )
        artifacts.commit()
    summary = read(run / "summary.json") if (run / "summary.json").exists() else None
    return dict(
        complete=error is None and bool(summary and summary.get("complete")),
        error=error,
        engineering_passed=bool(summary and summary.get("engineering_passed")),
        causal_admission=False,
    )


def collect(project):
    artifacts = modal.Volume.from_name("curiosity-study-artifacts", create_if_missing=False)
    count = 0
    for entry in artifacts.iterdir(NAME, recursive=True):
        if int(entry.type) != 1:
            continue
        name = Path(entry.path.lstrip("/"))
        require(name.parts[0] == NAME and ".." not in name.parts, "Unsafe remote artifact")
        destination, payload = Path(project) / "runs" / name, b"".join(artifacts.read_file(entry.path))
        if destination.exists():
            require(destination.read_bytes() == payload, "Local artifact bytes differ")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(payload)
        count += 1
    return count


def coverage(project, manifest):
    run = Path(project) / "runs" / NAME
    wrapper = Path(project) / "executions" / NAME / "engineering"
    worker = load_worker(wrapper, manifest)
    result = worker.operational_coverage(run, manifest)
    write_once(run / "collection-coverage.json", result)
    return result


def launch(project):
    project = Path(project).resolve()
    run = project / "runs" / NAME
    manifest, freeze = read(run / "manifest.json"), read(run / "freeze.json")
    execution = Path(freeze["execution_root"])
    wrapper = execution / "engineering"
    require(
        execution == project / "executions" / NAME
        and Path(__file__).resolve() == wrapper / Path(__file__).name
        and digest(manifest) == freeze["manifest_sha256"],
        "Use the exact frozen launcher",
    )
    check_files(execution / "original", manifest["original_execution_files"])
    worker = load_worker(wrapper, manifest)
    worker.validate_manifest(manifest)
    worker.verify_sources(manifest, execution / "original", execution / "runtime", wrapper)
    admission = claim(project, manifest, snapshot(project, wrapper))
    cli = project / ".venv/bin/modal"
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .uv_sync(str(execution / "original"), extras=["inference", "cloud"], uv_version="0.12.18")
        .env(
            {
                "PYTHONPATH": "/runtime:/engineering",
                "PYTHONDONTWRITEBYTECODE": "1",
                "HF_HOME": "/weights/huggingface",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
        .add_local_dir(execution / "original", "/original", ignore=lambda p: "__pycache__" in p.parts)
        .add_local_dir(execution / "runtime", "/runtime", ignore=lambda p: "__pycache__" in p.parts)
        .add_local_dir(wrapper, "/engineering", ignore=lambda p: "__pycache__" in p.parts)
    )
    app = modal.App(APP)
    job = app.function(
        image=image,
        gpu="H200",
        cpu=(2, 2),
        memory=(49152, 49152),
        volumes={
            "/weights": modal.Volume.from_name("curiosity-study-weights", create_if_missing=False),
            "/artifacts": modal.Volume.from_name("curiosity-study-artifacts", create_if_missing=False),
        },
        max_containers=1,
        min_containers=0,
        buffer_containers=0,
        scaledown_window=2,
        retries=0,
        timeout=WALL,
        startup_timeout=600,
        enable_memory_snapshot=False,
        include_source=False,
    )(run_cloud)
    done, stop_errors = threading.Event(), []

    def stop(app_id):
        if app_id:
            subprocess.run(
                [str(cli), "app", "stop", app_id],
                check=True,
                timeout=20,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )

    def watchdog():
        while not done.wait(max(0.1, min(2, admission["deadline"] - time.time()))):
            if time.time() < admission["deadline"]:
                continue
            try:
                apps = json.loads(
                    subprocess.check_output([str(cli), "app", "list", "--json"], text=True, timeout=20)
                )
                for row in apps:
                    if (
                        row.get("Description") == APP
                        and row["State"] != "stopped"
                        and (
                            dt.datetime.fromisoformat(row["Created at"]).timestamp()
                            >= admission["created"] - 2
                        )
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
            write_once(
                run / "launch.json",
                dict(
                    app_id=app_id,
                    admission_sha256=digest(admission),
                    image_id=image.object_id,
                    started=time.time(),
                ),
            )
            require(time.time() < admission["deadline"] - 60, "Image startup consumed engineering wall")
            call = job.spawn(manifest, admission, image.object_id)
            while result is None:
                remaining = admission["deadline"] - time.time()
                require(remaining > 0, "Fixed engineering wall deadline")
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
                state = snapshot(project, wrapper)
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
        export_started = time.monotonic()
        try:
            exported = collect(project)
        except Exception as caught:
            export_error = str(caught)
        observed = coverage(project, manifest)
        outcome = dict(
            app_id=app_id,
            result=result,
            failure=failure,
            stop_errors=stop_errors,
            all_stopped_zero_tasks=state["all_stopped_zero_tasks"],
            exported_files=exported,
            export_error=export_error,
            collection_coverage_sha256=digest(observed),
            export_seconds=time.monotonic() - export_started,
            export_timing_scope="Artifact export plus local operational-coverage rehashing; not pure network transfer",
            finished=time.time(),
            scientific_admission=False,
            causal_admission=False,
        )
        write_once(run / "launch-outcome.json", outcome)
        print(json.dumps(outcome, indent=2), flush=True)
    if failure or export_error or not state["all_stopped_zero_tasks"] or not result or not result["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("action", choices=["prepare", "launch", "collect"])
    args = parser.parse_args()
    result = {"prepare": prepare, "launch": launch, "collect": collect}[args.action](
        args.project_root.resolve()
    )
    if result is not None:
        print(json.dumps(result, indent=2))
