"""Single-worker bounded pilot. Importing this file does not create cloud resources."""

from pathlib import Path
import time
import uuid
import modal
import os

ROOT = Path(__file__).resolve().parents[1]
GPU = os.environ.get("CURIOSITY_MODAL_GPU", "L40S")
if GPU not in ("L40S", "H100", "H200"):
    raise ValueError("Unsupported explicitly selected GPU")
app = modal.App("curiosity-study-pilot" + ("-" + GPU.lower() if GPU != "L40S" else ""))
weights = modal.Volume.from_name("curiosity-study-weights", create_if_missing=True)
artifacts = modal.Volume.from_name("curiosity-study-artifacts", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(str(ROOT), extras=["inference", "cloud"], uv_version="0.12.18")
    .env({"HF_HOME": "/weights/huggingface", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("curiosity_study")
)


_FUNCTION_OPTIONS = dict(
    image=image,
    gpu="H100!" if GPU == "H100" else GPU,
    cpu=(2, 2),
    memory=(32768, 49152),
    volumes={"/weights": weights, "/artifacts": artifacts},
    max_containers=1,
    min_containers=0,
    buffer_containers=0,
    scaledown_window=2,
    startup_timeout=600,
    retries=0,
    enable_memory_snapshot=False,
)


def _pilot(run_id: str, manifest: dict, deadline: float, resolved_image_id: str):
    if manifest.get("assay") == "pain_profile_control":
        raise ValueError("Pain-profile control requires its separate pinned environment and launcher")
    from curiosity_study.artifacts import RunStore, write_once
    from curiosity_study.model import HFModel
    from curiosity_study.integrity import verify_model
    from curiosity_study.runner import execute, source_hash
    from curiosity_study.report import render
    from curiosity_study.curiosity_resource_limits import job_wall_limit
    import subprocess
    import torch

    if not run_id.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Invalid run ID")
    if not 0 < deadline - time.time() <= job_wall_limit(manifest):
        raise ValueError("Expired or excessive job deadline")
    if manifest["source_sha256"] != source_hash():
        raise ValueError("Cloud source differs from assignment manifest")
    artifacts.reload()
    root = Path("/artifacts") / run_id
    root.mkdir(parents=True, exist_ok=True)
    admissions = root / "admissions"
    admissions.mkdir(exist_ok=True)
    if manifest.get("assay") == "curiosity_suppression_screen" and any(admissions.glob("*.json")):
        raise RuntimeError("Curiosity suppression policy forbids paid resume attempts")
    if manifest.get("assay") == "satisfaction_screen" and any(admissions.glob("*.json")):
        raise RuntimeError("Satisfaction screen policy forbids paid resume attempts")
    if len(list(admissions.glob("*.json"))) >= 3:
        raise RuntimeError("Maximum three infrastructure attempts exhausted")
    admission = uuid.uuid4().hex
    write_once(admissions / f"{admission}.json", {"start": time.time(), "deadline": deadline})
    artifacts.commit()
    # Exercise actual Volume durability before loading weights.
    store = RunStore(root, manifest)
    artifacts.commit()
    artifacts.reload()
    store.completed()
    if manifest.get("execution", {}).get("backend") == "torch_batched_v1":
        from curiosity_study.batched import BatchedHFModel, verify_batch_model

        model = BatchedHFModel(**manifest["model"])
    else:
        model = HFModel(**manifest["model"])
    weights.commit()
    gpu = torch.cuda.get_device_name()
    if manifest.get("execution_gpu", "L40S") not in gpu:
        raise RuntimeError(f"Unexpected GPU: {gpu}")
    audit = verify_model(model)
    if manifest.get("execution", {}).get("backend") == "torch_batched_v1":
        audit["batch_audit"] = verify_batch_model(model)
        audit["passed"] = audit["passed"] and audit["batch_audit"]["passed"]
    write_once(root / "integrity" / f"{admission}.json", audit)
    artifacts.commit()
    if not audit["passed"]:
        raise RuntimeError("Integration audit failed")
    started = time.monotonic()
    checkpoint_count = 0

    def checkpoint_batch():
        nonlocal checkpoint_count
        checkpoint_count += 1
        if checkpoint_count % 25 == 0:
            artifacts.commit()

    try:
        completed = execute(root, manifest, model, checkpoint=checkpoint_batch, deadline=deadline - 120)
    finally:
        # Preserve terminal failure records even when a cohort raises before the
        # ordinary checkpoint interval; never retry an observed model outcome.
        artifacts.commit()
    profile = {
        "gpu": gpu,
        "driver": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).strip(),
        "load_seconds": model.load_seconds,
        "execution_seconds": time.monotonic() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "completed": len(completed),
        "assigned": len(manifest["assignments"]) + len(manifest["diagnostics"]),
        "modal_image_id": resolved_image_id,
        "artifact_bytes": sum(path.stat().st_size for path in root.rglob("*") if path.is_file()),
    }
    write_once(root / "cloud_profiles" / f"{admission}.json", profile)
    write_once(root / "report.json", {"markdown": render(root)})
    artifacts.commit()
    artifacts.reload()
    assert len(RunStore(root, manifest).completed()) == len(completed)
    return profile


@app.function(**_FUNCTION_OPTIONS, timeout=7200)
def pilot(run_id: str, manifest: dict, deadline: float, resolved_image_id: str):
    """Ordinary two-hour entrypoint; never accepts the screen exception."""
    if manifest.get("assay") == "curiosity_suppression_screen":
        raise ValueError("Curiosity suppression requires its reserved cloud entrypoint")
    if manifest.get("assay") == "satisfaction_screen":
        raise ValueError("Satisfaction screen requires its reserved cloud entrypoint")
    return _pilot(run_id, manifest, deadline, resolved_image_id)


@app.function(**_FUNCTION_OPTIONS, timeout=12600)
def curiosity_pilot(run_id: str, manifest: dict, deadline: float, resolved_image_id: str):
    """Bounded curiosity study only; all old entrypoint policies remain separate."""
    from curiosity_study.curiosity_resource_limits import ASSAY, CURIOSITY_SCREEN_BUDGET, job_wall_limit

    if manifest.get("assay") != ASSAY or job_wall_limit(manifest) != CURIOSITY_SCREEN_BUDGET["shared_deadline_seconds"]:
        raise ValueError("Extended curiosity entrypoint requires its exact study policy")
    return _pilot(run_id, manifest, deadline, resolved_image_id)


@app.function(**_FUNCTION_OPTIONS, timeout=12600)
def satisfaction_pilot(run_id: str, manifest: dict, deadline: float, resolved_image_id: str):
    """Longer hard timeout exclusively for the validated satisfaction screen."""
    from curiosity_study.resource_limits import SCREEN_BUDGET, job_wall_limit

    if job_wall_limit(manifest) != SCREEN_BUDGET["shared_deadline_seconds"]:
        raise ValueError("Extended cloud entrypoint requires the exact satisfaction screen policy")
    return _pilot(run_id, manifest, deadline, resolved_image_id)
