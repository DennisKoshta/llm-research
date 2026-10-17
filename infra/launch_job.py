"""Launch one ephemeral pilot, cancel on deadline, export artifacts after stopping."""

import argparse
import json
from pathlib import Path
import time
import uuid
import hashlib
from io import BytesIO
import zipfile
from concurrent.futures import ThreadPoolExecutor
import modal
from modal_app import app, pilot, satisfaction_pilot, curiosity_pilot, artifacts, image, GPU
from curiosity_study.artifacts import read, write_once, RunStore
from curiosity_study.report import render
from curiosity_study.resource_limits import SCREEN_BUDGET, job_wall_limit
from curiosity_study import curiosity_resource_limits as curiosity_limits
from budget import admission_seconds, claim_reservation, claim_reserved_resume


def await_result(call, deadline):
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError("Aggregate reserved job deadline reached")
        try:
            return call.get(timeout=min(30, remaining))
        except (TimeoutError, modal.exception.TimeoutError):
            # SDK polling raises builtin TimeoutError; a poll timeout is not job failure.
            continue


def collect(run_id, destination):
    entries = [entry for entry in artifacts.listdir(run_id, recursive=True) if entry.type.name == "FILE"]

    def download(entry):
        relative = Path(entry.path).relative_to(run_id)
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(3):
            payload = b"".join(artifacts.read_file(entry.path))
            try:
                if path.suffix == ".json":
                    json.loads(payload)
                elif path.suffix == ".npz":
                    with zipfile.ZipFile(BytesIO(payload)) as archive:
                        if archive.testzip() is not None:
                            raise ValueError("Corrupt activation archive")
                break
            except (ValueError, UnicodeDecodeError, zipfile.BadZipFile):
                if attempt == 2:
                    raise ValueError(f"Invalid artifact transport payload: {entry.path}") from None
                time.sleep(attempt + 1)
        if path.exists() and path.read_bytes() != payload:
            raise ValueError(f"Local artifact conflict: {path}")
        path.write_bytes(payload)

    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(download, entries))
    manifest = read(destination / "manifest.json")
    completed = RunStore(destination, manifest).completed()
    binary_hashes = {}
    for record in completed.values():
        artifact = record.get("activation_artifact")
        if artifact:
            prior_hash = binary_hashes.setdefault(artifact["path"], artifact["sha256"])
            if prior_hash != artifact["sha256"]:
                raise ValueError("Conflicting activation archive hashes")
    for relative, expected in binary_hashes.items():
        if hashlib.sha256((destination / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Activation archive hash mismatch: {relative}")
    (destination / "report.md").write_text(render(destination))
    return len(completed)


def select_pilot(manifest, reservation=None, resume=False):
    """Choose the sole permitted entrypoint before any admission or cloud work."""
    if manifest.get("assay") == "pain_profile_control":
        raise ValueError("Pain-profile control requires its dedicated bounded launcher")
    if manifest.get("assay") == curiosity_limits.ASSAY:
        curiosity_limits.job_wall_limit(manifest)
        if reservation is None:
            raise ValueError("Curiosity suppression requires a bound three-worker reservation")
        if resume:
            raise ValueError("Curiosity suppression policy forbids paid resume attempts")
        return curiosity_pilot
    limit = job_wall_limit(manifest)
    if manifest.get("assay") == "satisfaction_screen":
        if limit != SCREEN_BUDGET["shared_deadline_seconds"] or reservation is None:
            raise ValueError("Satisfaction screen requires a bound three-worker reservation")
        if resume:
            raise ValueError("Satisfaction screen policy forbids paid resume attempts")
        return satisfaction_pilot
    return pilot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reservation", type=Path)
    parser.add_argument("--recover-unstarted", action="store_true")
    args = parser.parse_args()
    run_id = args.run.name
    if args.collect_only:
        print(f"Verified {collect(run_id, args.run)} terminal records")
        return
    manifest = read(args.run / "manifest.json")
    cloud_pilot = select_pilot(manifest, args.reservation, args.resume)
    previous = read(args.run / "coordinator.json") if args.resume else None
    coordinator_path = (
        args.run / "coordinators" / f"{uuid.uuid4().hex}.json"
        if args.resume
        else args.run / "coordinator.json"
    )
    if manifest.get("execution_gpu", "L40S") != GPU:
        raise ValueError("Explicit GPU selection differs from the frozen manifest")
    if args.reservation:
        if args.resume:
            if args.recover_unstarted:
                raise ValueError("Choose either stopped-worker resume or unstarted recovery")
            deadline, budget_check = claim_reserved_resume(
                args.reservation, args.run, manifest, GPU, coordinator_path
            )
        else:
            deadline, budget_check = claim_reservation(
                args.reservation, args.run, manifest, GPU, recover_unstarted=args.recover_unstarted
            )
    else:
        allowed_seconds, budget_check = admission_seconds(gpu=GPU)
        deadline = (
            min(previous["deadline"], time.time() + allowed_seconds)
            if previous
            else time.time() + allowed_seconds
        )
    if coordinator_path.exists():
        raise ValueError("Coordinator already exists; inspect the run then use --resume")
    if time.time() >= deadline:
        raise TimeoutError("The original aggregate deadline has elapsed")
    call = None
    coordinator = {
        "start": time.time(),
        "budget_check": budget_check,
        "deadline": deadline,
        "authorized_budget_usd": budget_check["budget_usd"],
        "max_containers": 1,
        "max_admissions": 1 if manifest.get("assay") in ("satisfaction_screen", curiosity_limits.ASSAY) else 3,
    }
    try:
        with modal.enable_output(), app.run():
            coordinator["app_id"] = app.app_id
            coordinator["image_id"] = image.object_id
            call = cloud_pilot.spawn(run_id, manifest, deadline, image.object_id)
            coordinator["function_call_id"] = call.object_id
            write_once(coordinator_path, coordinator)
            profile = await_result(call, deadline)
            print(profile, flush=True)
    finally:
        if call is not None:
            call.cancel(terminate_containers=True)
        write_once(
            coordinator_path.with_name(coordinator_path.stem + "_end.json"),
            {"end": time.time(), "app_id": coordinator.get("app_id"), "cancel_requested": call is not None},
        )
    print(f"Exported and verified {collect(run_id, args.run)} terminal records", flush=True)


if __name__ == "__main__":
    main()
