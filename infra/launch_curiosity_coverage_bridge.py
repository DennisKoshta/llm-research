"""Three distinct project apps, one exclusive shared scientific admission."""

import argparse
from concurrent.futures import ThreadPoolExecutor
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
import tempfile
import threading
import time
import modal

NAME = "qwen32-curiosity-coverage-bridge-v1-admission2"
ENGINEERING_NAME = "qwen32-curiosity-coverage-engineering-v1"
ENGINEERING_SHA = "6de4589af3e007ddba55440988f8f07ebabcf80f1e16f1d01584d017f8d9adf5"
SOURCE_SHA = "1fd4079ed60fbd0169f702ab9c8f99a75d170450dfbd364bc4cb877e27c18f9a"
ARCHIVE_SHA = "4b82bdc563a3b257fd1b5e0302f5889cb6e14e30da9fcb950cc7f3039eb62ca4"
GROUPS = tuple(f"worker_{i}" for i in range(3))
APPS = tuple(f"curiosity-study-coverage-bridge-worker-{i}-h200" for i in range(3))
RATE, WALL, TEARDOWN, CEILING = 0.00139376, 28000, 120, 120.0
LOCAL_EXPORT_SECONDS = 3600
WORKER = "curiosity_coverage_bridge_worker.py"
OVERLAYS = ("state_backend.py", "coverage_engineering.py", "coverage_bridge.py")
AMENDMENT = "reports/curiosity-coverage-bridge-amended-resource-receipt-v1.json"
FORECAST = "reports/curiosity-coverage-bridge-v1-feasibility.json"
FILES = {
    "launch_curiosity_coverage_bridge.py": "infra/launch_curiosity_coverage_bridge.py",
    WORKER: "scripts/" + WORKER,
    "test_bridge_controller.py": "tests/test_coverage_bridge_controller.py",
    "test_bridge_worker.py": "tests/test_coverage_bridge_worker.py",
    "test_scientific_collector.py": "tests/test_coverage_bridge.py",
    "analyze_curiosity_coverage_bridge.py": "scripts/analyze_curiosity_coverage_bridge.py",
    "test_bridge_analysis.py": "tests/test_coverage_bridge_analysis.py",
    "audit_curiosity_coverage_bridge.py": "scripts/audit_curiosity_coverage_bridge.py",
    "test_bridge_audit.py": "tests/test_coverage_bridge_audit.py",
    "score_curiosity_coverage_bridge.py": "scripts/score_curiosity_coverage_bridge.py",
    "test_bridge_scoring.py": "tests/test_score_curiosity_coverage_bridge.py",
    "audit_curiosity_coverage_bridge_lifecycle.py": "scripts/audit_curiosity_coverage_bridge_lifecycle.py",
    "test_bridge_lifecycle.py": "tests/test_coverage_bridge_lifecycle.py",
}


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


def resources():
    return dict(
        app_names=list(APPS),
        workers=3,
        gpu="H200",
        cpu=2,
        memory_gib=48,
        wall_seconds=WALL,
        teardown_seconds=TEARDOWN,
        rate_usd_per_second=RATE,
        ceiling_usd=CEILING,
        maximum_liability_usd=3 * (WALL + TEARDOWN) * RATE,
        max_containers_per_app=1,
        retries=0,
        resume_allowed=False,
        local_export_only_seconds=LOCAL_EXPORT_SECONDS,
    )


def inventory(root):
    return {
        str(p.relative_to(root)): sha(p)
        for p in Path(root).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }


def load_worker(wrapper, manifest=None):
    if manifest is not None:
        check_files(wrapper, manifest["wrapper_files"])
    spec = importlib.util.spec_from_file_location("_coverage_bridge_worker", Path(wrapper) / WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_live_evidence(project, manifest):
    """Before admission rehash prior inputs; never inspect scientific effects."""
    project = Path(project).resolve()
    for relative, checksum in manifest["documents"].items():
        require(sha(project / relative) == checksum, "Live prerequisite changed: " + relative)
    lifecycle = read(project / "runs" / ENGINEERING_NAME / "lifecycle-audit.json")
    for relative, checksum in lifecycle["raw_input_files"].items():
        require(
            sha(project / "runs" / ENGINEERING_NAME / relative) == checksum,
            "Audited engineering input changed",
        )
    for relative, checksum in lifecycle["external_files"].items():
        require(sha(project / relative) == checksum, "Engineering lifecycle input changed")
    forecast = read(project / FORECAST)
    require(
        forecast["input_files_sha256"] == digest(forecast["input_files"]), "Forecast input inventory changed"
    )
    for item in forecast["input_files"]:
        path = Path(item["path"]).resolve()
        require(
            path.is_relative_to(project)
            and sha(path) == item["byte_sha256"]
            and path.stat().st_size == item["byte_count"],
            "Original forecast input changed",
        )
    return True


def prepare(project):
    """Local-only, staged immutable preparation. Never admits or starts cloud work."""
    project = Path(project).resolve()
    run, execution = project / "runs" / NAME, project / "executions" / NAME
    require(not run.exists() and not execution.exists(), "Scientific campaign already prepared")
    parent, old_execution = project / "runs" / ENGINEERING_NAME, project / "executions" / ENGINEERING_NAME
    eng, frozen = read(parent / "manifest.json"), read(parent / "freeze.json")
    require(
        digest(eng) == frozen["manifest_sha256"] == ENGINEERING_SHA
        and Path(frozen["execution_root"]) == old_execution,
        "Engineering freeze differs",
    )
    for folder, key in (
        ("original", "original_execution_files"),
        ("runtime", "runtime_files"),
        ("engineering", "wrapper_files"),
    ):
        check_files(old_execution / folder, eng[key])
    require(
        all((project / p).is_file() and not (project / p).is_symlink() for p in FILES.values()),
        "Required reviewed wrapper/auditor/test unavailable",
    )
    amended = read(project / AMENDMENT)
    documents = {item["path"]: item["byte_sha256"] for item in amended["inputs"].values()}
    documents[AMENDMENT] = sha(project / AMENDMENT)
    with tempfile.TemporaryDirectory(prefix="coverage-bridge-freeze-", dir=project) as temporary:
        staging = Path(temporary)
        original, runtime, wrapper = (staging / p for p in ("original", "runtime", "bridge"))
        for old, new in (("original", original), ("runtime", runtime), ("engineering", wrapper)):
            shutil.copytree(old_execution / old, new, ignore=shutil.ignore_patterns("__pycache__"))
        collector = project / "src/curiosity_study/coverage_bridge.py"
        require(
            not collector.is_symlink()
            and sha(collector) == documents["src/curiosity_study/coverage_bridge.py"],
            "Reviewed scientific collector changed",
        )
        shutil.copyfile(collector, runtime / "curiosity_study/coverage_bridge.py")
        delta = dict(
            eng["runtime_delta"],
            **{
                "coverage_bridge.py": dict(
                    original_byte_sha256=eng["original_execution_files"].get(
                        "src/curiosity_study/coverage_bridge.py"
                    ),
                    replacement_byte_sha256=sha(collector),
                )
            },
        )
        for destination, source in FILES.items():
            path = wrapper / destination
            require(not path.exists(), "New wrapper collides with held engineering file")
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / source, path)
        for relative, checksum in documents.items():
            require(sha(project / relative) == checksum, "Prerequisite document changed")
            target = wrapper / "evidence" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / relative, target)
        worker = load_worker(wrapper)
        analyzer = worker.load_module(
            wrapper / "analyze_curiosity_coverage_bridge.py", "_prospective_bridge_analysis"
        )
        runtime_source = runtime / "curiosity_study"
        manifest = dict(
            schema_version=1,
            name=NAME,
            assay="curiosity_coverage_bridge",
            engineering_manifest=eng,
            engineering_manifest_sha256=ENGINEERING_SHA,
            original_manifest=eng["original_manifest"],
            original_freeze=eng["original_freeze"],
            source_sha256=SOURCE_SHA,
            original_source_archive_sha256=ARCHIVE_SHA,
            original_execution_files=eng["original_execution_files"],
            runtime_files=inventory(runtime),
            runtime_delta=delta,
            runtime_source_sha256=digest(
                {
                    str(p.relative_to(runtime_source)): p.read_text()
                    for p in sorted(runtime_source.rglob("*.py"))
                }
            ),
            wrapper_files=inventory(wrapper),
            documents=documents,
            resource_amendment=amended,
            plan=eng["plan"],
            plan_sha256=eng["plan_sha256"],
            preflight=eng["preflight"],
            preflight_sha256=eng["preflight_sha256"],
            analysis_metadata=analyzer.ANALYSIS_METADATA,
            workers=worker.worker_descriptors(eng["plan"]),
            resources=resources(),
            scientific_admission=False,
            causal_admission=False,
        )
        verify_live_evidence(project, manifest)
        modules = worker.verify_sources(manifest, original, runtime, wrapper)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            eng["plan"]["model"]["model_id"],
            revision=eng["plan"]["model"]["revision"],
            local_files_only=True,
            trust_remote_code=False,
        )
        require(
            worker.tokenizer_preflight(eng["plan"], tokenizer, modules["voluntary_resolution"])
            == eng["preflight"],
            "Exact original preflight differs",
        )
        write_once(staging / "run/manifest.json", manifest)
        write_once(
            staging / "run/freeze.json",
            dict(
                manifest_sha256=digest(manifest),
                execution_root=str(execution),
                source_sha256=SOURCE_SHA,
                original_source_archive_sha256=ARCHIVE_SHA,
                runtime_files_sha256=digest(manifest["runtime_files"]),
                wrapper_files_sha256=digest(manifest["wrapper_files"]),
            ),
        )
        execution.parent.mkdir(exist_ok=True)
        run.parent.mkdir(exist_ok=True)
        require(not execution.exists() and not run.exists(), "Campaign appeared during preparation")
        os.rename(staging, execution)
        os.rename(execution / "run", run)
    return dict(run=str(run), manifest_sha256=digest(manifest), assigned_generations=8352, admitted=False)


def claim(project, manifest, initial, now=None):
    now = time.time() if now is None else now
    require(
        manifest["name"] == NAME
        and manifest["resources"] == resources()
        and manifest["scientific_admission"] is False
        and manifest["causal_admission"] is False,
        "Fixed scientific resource policy differs",
    )
    require(
        manifest["resource_amendment"]["amended_resource_check_passed"] is True
        and manifest["resource_amendment"]["original_forecast_passed"] is False,
        "Explicit resource amendment required",
    )
    require(
        initial["all_stopped_zero_tasks"] is True
        and all(
            type(v) in (int, float) and math.isfinite(v) and v >= 0
            for v in (now, initial["available_compute_estimate_usd"], initial["phase_compute_estimate_usd"])
        ),
        "Invalid budget/time or a project app remains active",
    )
    liability = resources()["maximum_liability_usd"]
    require(
        liability <= CEILING and liability <= initial["available_compute_estimate_usd"] - 10,
        "Insufficient remaining allowance and retained reserve",
    )
    value = dict(
        manifest_sha256=digest(manifest),
        workers_sha256=digest(manifest["workers"]),
        created=now,
        deadline=now + WALL,
        phase_consumed=True,
        retry_allowed=False,
        maximum_liability_usd=liability,
        budget_timestamp=initial["timestamp_utc"],
        prior_phase_compute_estimate_usd=initial["phase_compute_estimate_usd"],
    )
    write_once(Path(project) / "campaign_admissions" / NAME / "admission.json", value)
    write_once(Path(project) / "runs" / NAME / "admission.json", value)
    return value


def claim_worker(project, manifest, admission, group, worker, now=None):
    now = time.time() if now is None else now
    worker.validate_admission(manifest, admission, now)
    require(
        (Path(project) / "campaign_admissions" / NAME / "admission.json").read_bytes()
        == (Path(project) / "runs" / NAME / "admission.json").read_bytes()
        and read(Path(project) / "runs" / NAME / "admission.json") == admission,
        "Actual exclusive parent admission required",
    )
    require(group in GROUPS, "Unknown worker")
    descriptor = manifest["workers"][GROUPS.index(group)]
    value = dict(
        manifest_sha256=digest(manifest),
        parent_manifest_sha256=digest(manifest),
        admission_sha256=digest(admission),
        execution_group=group,
        worker=descriptor,
        app_name=descriptor["app_name"],
        parent_created=admission["created"],
        created=now,
        deadline=admission["deadline"],
        consumed=True,
    )
    worker.validate_worker_claim(manifest, admission, value, group, now)
    write_once(Path(project) / "campaign_admissions" / NAME / "workers" / (group + ".json"), value)
    root = Path(project) / "runs" / NAME / "workers" / group
    write_once(root / "worker-claim.json", value)
    write_once(root / "manifest.json", manifest)
    write_once(root / "admission.json", admission)
    return value


def frozen_context(project):
    project = Path(project).resolve()
    run, execution = project / "runs" / NAME, project / "executions" / NAME
    manifest, freeze = read(run / "manifest.json"), read(run / "freeze.json")
    wrapper = execution / "bridge"
    require(
        Path(freeze["execution_root"]) == execution
        and digest(manifest) == freeze["manifest_sha256"]
        and Path(__file__).resolve() == wrapper / Path(__file__).name,
        "Use exact frozen scientific launcher",
    )
    worker = load_worker(wrapper, manifest)
    worker.verify_sources(manifest, execution / "original", execution / "runtime", wrapper)
    return run, execution, wrapper, manifest, worker


def run_cloud(manifest, admission, claim, image_id):
    group = claim["execution_group"]
    artifacts = modal.Volume.from_name("curiosity-study-artifacts", create_if_missing=False)
    artifacts.reload()
    root = Path("/artifacts") / NAME / "workers" / group
    require(not root.exists(), "Remote worker namespace already consumed")
    root.mkdir(parents=True)
    for name, value in (("manifest", manifest), ("admission", admission), ("worker-claim", claim)):
        write_once(root / (name + ".json"), value)
    write_once(
        root / "cloud-started.json", dict(started=time.time(), image_id=image_id, execution_group=group)
    )
    commits = []

    def commit():
        began = time.monotonic()
        artifacts.commit()
        commits.append(time.monotonic() - began)

    commit()
    process, error = None, None
    try:
        worker = load_worker(Path("/bridge"), manifest)
        worker.validate_worker_claim(manifest, admission, claim, group, time.time())
        worker.verify_sources(manifest, Path("/original"), Path("/runtime"), Path("/bridge"))
        env = dict(
            os.environ,
            PYTHONPATH="/runtime:/bridge",
            PYTHONDONTWRITEBYTECODE="1",
            HF_HOME="/weights/huggingface",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            HF_HUB_DISABLE_TELEMETRY="1",
            TOKENIZERS_PARALLELISM="false",
        )
        with (root / "child.log").open("xb", buffering=0) as output:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "/bridge/" + WORKER,
                    "--original-root",
                    "/original",
                    "--runtime",
                    "/runtime",
                    "--wrapper",
                    "/bridge",
                    "--run",
                    str(root),
                    "--group",
                    group,
                ],
                stdout=output,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            previous = None
            while process.poll() is None:
                require(time.time() < admission["deadline"] - 30, "Common worker deadline")
                marker = root / "checkpoint.request"
                current = marker.read_text() if marker.exists() else None
                if current is not None and current != previous:
                    commit()
                    previous = current
                time.sleep(0.5)
            require(process.returncode == 0, "Scientific worker incomplete; all dispositions retained")
    except Exception as caught:
        error = dict(type=type(caught).__name__, message=str(caught))
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
        write_once(
            root / "cloud-finished.json",
            dict(
                finished=time.time(),
                error=error,
                checkpoint_commit_seconds=commits,
                checkpoint_commit_total_seconds=sum(commits),
                final_commit_timing_excluded=True,
            ),
        )
        artifacts.commit()
    summary = read(root / "summary.json") if (root / "summary.json").exists() else None
    return dict(
        complete=error is None and bool(summary and summary["complete"]),
        engineering_passed=bool(summary and summary["engineering_passed"]),
        error=error,
        execution_group=group,
        scientific_success=False,
    )


def collect(project, group=None):
    prefix = NAME if group is None else NAME + "/workers/" + group
    require(group is None or group in GROUPS, "Unknown export worker")
    artifacts = modal.Volume.from_name("curiosity-study-artifacts", create_if_missing=False)
    count = 0
    for entry in artifacts.iterdir(prefix, recursive=True):
        if int(entry.type) != 1:
            continue
        relative = Path(entry.path.lstrip("/"))
        require(relative.is_relative_to(prefix) and ".." not in relative.parts, "Unsafe remote artifact")
        path, payload = Path(project) / "runs" / relative, b"".join(artifacts.read_file(entry.path))
        publish_download(project, path, payload)
        count += 1
    return count


def publish_download(project, path, payload):
    """Only complete fsynced bytes become visible; interrupted temps are outside raw trees."""
    path = Path(path)
    require(not path.is_symlink(), "Symlink at immutable export destination")
    if path.exists():
        require(path.read_bytes() == payload, "Existing immutable artifact differs")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".coverage-download-", dir=project) as temporary:
        temporary.write(payload)
        temporary.flush()
        os.fsync(temporary.fileno())
        try:
            os.link(temporary.name, path)
        except FileExistsError:
            require(
                not path.is_symlink() and path.read_bytes() == payload, "Concurrent immutable export differs"
            )


def verify_local_export_window(project, run, admission, groups, now):
    require(
        now <= admission["deadline"] + TEARDOWN + LOCAL_EXPORT_SECONDS,
        "Separate local-only export deadline exhausted",
    )
    rows = matching_apps(app_rows(project), admission)
    for group in groups:
        marker = read(Path(run) / "workers" / group / "export-started.json")
        matching = [r for r in rows if r["Description"] == APPS[GROUPS.index(group)]]
        require(
            marker["execution_group"] == group
            and marker["app_stopped_zero_tasks"] is True
            and marker["admission_sha256"] == digest(admission)
            and len(matching) == 1
            and marker["app_id"] == matching[0]["App ID"]
            and matching[0]["State"] == "stopped"
            and str(matching[0]["Tasks"]) == "0",
            "Only verified stopped-app exports may continue locally",
        )


def app_rows(project):
    return json.loads(
        subprocess.check_output(
            [str(Path(project) / ".venv/bin/modal"), "app", "list", "--json"], text=True, timeout=20
        )
    )


def stop_app(project, app_id):
    if app_id:
        subprocess.run(
            [str(Path(project) / ".venv/bin/modal"), "app", "stop", app_id],
            check=True,
            timeout=20,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )


def matching_apps(rows, admission, names=APPS):
    return [
        row
        for row in rows
        if row.get("Description") in names
        and dt.datetime.fromisoformat(row["Created at"]).timestamp() >= admission["created"] - 2
    ]


def make_image(execution, wrapper):
    return (
        modal.Image.debian_slim(python_version="3.12")
        .uv_sync(str(execution / "original"), extras=["inference", "cloud"], uv_version="0.12.18")
        .env(
            dict(
                PYTHONPATH="/runtime:/bridge",
                PYTHONDONTWRITEBYTECODE="1",
                HF_HOME="/weights/huggingface",
                HF_HUB_OFFLINE="1",
                TRANSFORMERS_OFFLINE="1",
            )
        )
        .add_local_dir(execution / "original", "/original", ignore=lambda p: "__pycache__" in p.parts)
        .add_local_dir(execution / "runtime", "/runtime", ignore=lambda p: "__pycache__" in p.parts)
        .add_local_dir(wrapper, "/bridge", ignore=lambda p: "__pycache__" in p.parts)
    )


def launch_worker(project, group):
    project = Path(project).resolve()
    run, execution, wrapper, manifest, worker = frozen_context(project)
    admission = read(run / "admission.json")
    require(
        (project / "campaign_admissions" / NAME / "admission.json").read_bytes()
        == (run / "admission.json").read_bytes(),
        "Global parent admission differs",
    )
    claim = claim_worker(project, manifest, admission, group, worker)
    root = run / "workers" / group
    image, app = make_image(execution, wrapper), modal.App(APPS[GROUPS.index(group)])
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
    call, app_id, result, failure, stop_errors = None, None, None, None, []
    try:
        with modal.enable_output(), app.run():
            app_id = app.app_id
            write_once(
                root / "launch.json",
                dict(
                    app_id=app_id,
                    image_id=image.object_id,
                    admission_sha256=digest(admission),
                    worker_claim_sha256=digest(claim),
                    execution_group=group,
                    started=time.time(),
                ),
            )
            require(time.time() < admission["deadline"] - 60, "Startup exhausted common deadline")
            call = job.spawn(manifest, admission, claim, image.object_id)
            while result is None:
                remaining = admission["deadline"] - time.time()
                require(remaining > 0, "Common parent deadline")
                try:
                    result = call.get(timeout=min(15, remaining))
                except TimeoutError:
                    pass
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
            stop_app(project, app_id)
        except Exception as caught:
            stop_errors.append(str(caught))
        stopped = False
        until = min(time.time() + 90, admission["deadline"] + TEARDOWN)
        while time.time() < until:
            try:
                rows = [r for r in app_rows(project) if r["App ID"] == app_id]
                stopped = len(rows) == 1 and rows[0]["State"] == "stopped" and str(rows[0]["Tasks"]) == "0"
                if stopped or app_id is None:
                    break
            except Exception as caught:
                stop_errors.append(str(caught))
            time.sleep(2)
        write_once(
            root / "export-started.json",
            dict(
                execution_group=group,
                app_id=app_id,
                app_stopped_zero_tasks=stopped,
                admission_sha256=digest(admission),
                started=time.time(),
                inference_allowed=False,
            ),
        )
        exported, export_error, began = None, None, time.monotonic()
        try:
            exported = collect(project, group)
        except Exception as caught:
            export_error = str(caught)
        outcome = dict(
            app_id=app_id,
            execution_group=group,
            result=result,
            failure=failure,
            stop_errors=stop_errors,
            app_stopped_zero_tasks=stopped,
            exported_files=exported,
            export_error=export_error,
            export_seconds=time.monotonic() - began,
            finished=time.time(),
        )
        write_once(root / "launch-outcome.json", outcome)
    if (
        failure
        or export_error
        or not stopped
        or not result
        or not result["complete"]
        or not result["engineering_passed"]
    ):
        raise SystemExit(1)
    return outcome


def aggregate_coverage(run, manifest, worker):
    values = {g: worker.operational_coverage(Path(run) / "workers" / g, manifest, g) for g in GROUPS}
    return dict(
        complete=all(v["complete"] for v in values.values()),
        workers=values,
        assigned_generations=8352,
        assigned_choices=4176,
        independent_audit_required=True,
    )


def launch(project):
    project = Path(project).resolve()
    run, execution, wrapper, manifest, worker = frozen_context(project)
    verify_live_evidence(project, manifest)
    admission = claim(project, manifest, snapshot(project, wrapper))
    processes, handles, errors, done = {}, [], [], threading.Event()

    def stop_all():
        rows = matching_apps(app_rows(project), admission)
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(
                pool.map(
                    lambda row: stop_app(project, row["App ID"]), [r for r in rows if r["State"] != "stopped"]
                )
            )
        return results

    def watchdog():
        while not done.wait(max(0.1, min(2.0, admission["deadline"] - time.time()))):
            if time.time() >= admission["deadline"]:
                try:
                    stop_all()
                except Exception as caught:
                    errors.append(str(caught))

    watcher = threading.Thread(target=watchdog, daemon=True)
    watcher.start()
    failure = None
    try:
        for group in GROUPS:
            require(time.time() < admission["deadline"] - 60, "Admission expired before fixed worker launch")
            path = run / "logs" / (group + ".log")
            path.parent.mkdir(exist_ok=True)
            handle = path.open("xb", buffering=0)
            handles.append(handle)
            processes[group] = subprocess.Popen(
                [
                    str(project / ".venv/bin/python"),
                    str(Path(__file__).resolve()),
                    "--project-root",
                    str(project),
                    "--group",
                    group,
                    "worker",
                ],
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
            )
        next_export_check = 0.0
        while any(p.poll() is None for p in processes.values()):
            now = time.time()
            if now >= admission["deadline"] + TEARDOWN and now >= next_export_check:
                verify_local_export_window(
                    project, run, admission, [g for g, p in processes.items() if p.poll() is None], now
                )
                next_export_check = now + 10
            time.sleep(0.5)
    except Exception as caught:
        failure = dict(type=type(caught).__name__, message=str(caught))
    finally:
        for process in processes.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        for handle in handles:
            handle.close()
        try:
            stop_all()
        except Exception as caught:
            errors.append(str(caught))
        state = None
        until = min(time.time() + 90, admission["deadline"] + TEARDOWN)
        while True:
            try:
                state = snapshot(project, wrapper)
                if state["all_stopped_zero_tasks"]:
                    break
            except Exception as caught:
                errors.append(str(caught))
            if time.time() >= until:
                break
            time.sleep(2)
        done.set()
        watcher.join(timeout=25)
        if state is None:
            state = dict(all_stopped_zero_tasks=False, cost_unknown=True)
        write_once(project / "reports" / (NAME + "-cost-and-shutdown.json"), state)
        exported, export_error, began = 0, None, time.monotonic()
        try:
            for group in GROUPS:
                child_path = run / "workers" / group / "launch-outcome.json"
                child = read(child_path) if child_path.exists() else None
                if child and child["export_error"] is None and type(child["exported_files"]) is int:
                    exported += child["exported_files"]
                else:
                    exported += collect(project, group)
        except Exception as caught:
            export_error = str(caught)
        coverage = aggregate_coverage(run, manifest, worker)
        write_once(run / "collection-coverage.json", coverage)
        worker_outcomes = {
            g: read(run / "workers" / g / "launch-outcome.json")
            if (run / "workers" / g / "launch-outcome.json").exists()
            else None
            for g in GROUPS
        }
        exits = {g: processes[g].returncode if g in processes else None for g in GROUPS}
        complete = (
            coverage["complete"]
            and failure is None
            and export_error is None
            and not errors
            and state["all_stopped_zero_tasks"] is True
            and all(code == 0 for code in exits.values())
        )
        outcome = dict(
            complete=complete,
            all_stopped_zero_tasks=state["all_stopped_zero_tasks"],
            worker_outcomes=worker_outcomes,
            worker_exit_codes=exits,
            failure=failure,
            stop_errors=errors,
            collection_coverage_sha256=digest(coverage),
            exported_files=exported,
            export_error=export_error,
            export_seconds=time.monotonic() - began,
            export_timing_scope="Artifact export plus local coverage verification; no inference",
            finished=time.time(),
            independent_audit_required=True,
        )
        write_once(run / "launch-outcome.json", outcome)
    if not complete:
        raise SystemExit(1)
    return outcome


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--group", choices=GROUPS)
    parser.add_argument("action", choices=("prepare", "launch", "worker", "collect"))
    args = parser.parse_args()
    require(args.action != "worker" or args.group is not None, "Worker group required")
    if args.action == "worker":
        result = launch_worker(args.project_root, args.group)
    elif args.action == "collect":
        result = collect(args.project_root, args.group)
    else:
        result = {"prepare": prepare, "launch": launch}[args.action](args.project_root)
    if result is not None:
        print(json.dumps(result, indent=2))
