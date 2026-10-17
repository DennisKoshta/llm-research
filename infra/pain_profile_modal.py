"""Small control image; pinned numerical environment installs in bounded CPU phase."""

import os
from pathlib import Path
import signal
import subprocess
import time

import modal

from curiosity_study.artifacts import read, write_once
from curiosity_study import pain_profile_resources as resources
from curiosity_study.pain_profile_assets import require
from curiosity_study.pain_profile_admission import CAMPAIGN_NAME

weights = modal.Volume.from_name("curiosity-study-weights", create_if_missing=False)
artifacts = modal.Volume.from_name("curiosity-study-artifacts", create_if_missing=False)
CACHE = Path("/weights/pain-axis-positive-control-v1")
PROJECT = Path("/project")


def bounded_child(command, *, deadline, log, environment, checkpoint=None):
    """A child failure or deadline is terminal; never restart a process."""
    require(time.time() < deadline, "Deadline expired before child start")
    with Path(log).open("ab", buffering=0) as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                   env=environment, start_new_session=True)
        try:
            while process.poll() is None:
                if time.time() >= deadline:
                    raise TimeoutError("Phase child reached its fixed wall deadline")
                if checkpoint is not None:
                    checkpoint()
                time.sleep(min(1.0, max(0.01, deadline - time.time())))
            require(process.returncode == 0, f"Child exited {process.returncode}; inspect retained log")
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)


def run_cloud(manifest, admission, prior_files, resolved_image_id):
    from curiosity_study.pain_profile_worker import verify_payload

    phase = admission["phase"]
    require(phase in resources.POLICIES, "Unknown phase")
    artifacts.reload()
    weights.reload()
    run = Path("/artifacts") / CAMPAIGN_NAME
    write_once(run / "manifest.json", manifest)
    transport = run / "transport" / phase
    require(not transport.exists(), "Cloud phase already consumed; no retry")
    transport.mkdir(parents=True)
    write_once(run / "admissions" / f"{phase}.json", admission)
    for name, value in prior_files.items():
        require(name in {"phases/prefetch/summary.json", "phases/setup/summary.json", "phases/setup/audit.json"},
                "Unexpected prior phase payload")
        write_once(run / name, value)
    write_once(transport / "started.json", dict(started=time.time(), image_id=resolved_image_id))
    artifacts.commit()
    error = None
    try:
        verify_payload(PROJECT, run, phase)
        deadline = admission["deadline"] - 30
        environment = dict(os.environ, PYTHONPATH="/project/src:/project/infra:/project/scripts",
                           PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false",
                           HF_HOME=str(CACHE / "huggingface"),
                           HF_HUB_DISABLE_TELEMETRY="1", PAIN_PROFILE_IMAGE_ID=resolved_image_id)
        runtime = CACHE / "runtime"
        log = transport / "child.log"
        if phase == "prefetch":
            require(not runtime.exists(), "Pinned environment already exists; no implicit repair")
            bounded_child(["uv", "venv", "--python", "/usr/local/bin/python", str(runtime)],
                          deadline=deadline, log=log, environment=environment)
            bounded_child(["uv", "pip", "install", "--python", str(runtime / "bin/python"),
                           "--require-hashes", "--torch-backend", "cu130", "--only-binary", ":all:",
                           "-r", "/project/configs/pain_profile_requirements.lock"],
                          deadline=deadline, log=log, environment=environment)
        require((runtime / "bin/python").is_file(), "Missing pinned runtime")
        last = None

        def checkpoint():
            nonlocal last
            marker = transport / "checkpoint.request"
            value = marker.read_text() if marker.exists() else None
            if value is not None and value != last:
                artifacts.commit()
                last = value

        bounded_child([str(runtime / "bin/python"), "-m", "curiosity_study.pain_profile_worker",
                       "--root", str(PROJECT), "--run", str(run), "--cache", str(CACHE), "--phase", phase],
                      deadline=deadline, log=log, environment=environment, checkpoint=checkpoint)
    except Exception as caught:
        error = {"type": type(caught).__name__, "message": str(caught)}
    finally:
        write_once(transport / "finished.json", dict(finished=time.time(), error=error))
        # Preparation can fail after downloading useful bytes. Preserve them,
        # but this does not authorize another admission or environment repair.
        if phase == "prefetch":
            weights.commit()
        artifacts.commit()
    path = run / "phases" / phase / "summary.json"
    summary = read(path) if path.exists() else None
    return dict(phase=phase, complete=bool(summary and summary.get("complete")),
                passed=bool(error is None and summary and summary.get("passed")), error=error,
                summary=summary)


def create_app(root, phase):
    """Construction is local only; admission precedes App.run in the launcher."""
    policy = resources.policy(phase)
    root = Path(root).resolve()
    require(root.parent.name == "executions" and root.name == CAMPAIGN_NAME, "Use the frozen source copy")
    image = (modal.Image.debian_slim(python_version="3.12")
             .pip_install("uv==0.12.18")
             .env({"PYTHONPATH": "/project/src:/project/infra:/project/scripts", "PYTHONDONTWRITEBYTECODE": "1"})
             .add_local_dir(root, "/project", ignore=lambda p: p.is_symlink()
                            or bool({".venv", "__pycache__"}.intersection(p.parts))))
    app = modal.App(policy["app_name"])
    job = app.function(image=image, gpu="H100!" if policy["gpu"] else None,
                       cpu=(2, 2), memory=(98304, 98304),
                       volumes={"/weights": weights, "/artifacts": artifacts},
                       max_containers=1, min_containers=0, buffer_containers=0,
                       scaledown_window=2, startup_timeout=600, retries=0,
                       timeout=policy["function_timeout_seconds"],
                       enable_memory_snapshot=False, include_source=False)(run_cloud)
    return app, job, image
