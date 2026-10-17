"""Run two or more frozen collections under one combined reservation and deadline."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

from curiosity_study.artifacts import digest, read, write_once
from collection_coverage import CollectionCoverageError, check_collection_coverage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "infra"))
import budget  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment(execution):
    return {
        **os.environ,
        "CURIOSITY_MODAL_GPU": "H200",
        "CURIOSITY_BUDGET_ROOT": str(ROOT),
        "PYTHONPATH": str(execution / "src") + os.pathsep + str(execution / "infra"),
    }


def frozen_parent(parent):
    parent = Path(parent).resolve()
    if parent.parent != ROOT / "runs":
        raise ValueError("Every parent must be a canonical project run")
    frozen, manifest = read(parent / "freeze.json"), read(parent / "manifest.json")
    execution = Path(frozen["execution_root"]).resolve()
    if (
        digest(manifest) != frozen["manifest_sha256"]
        or manifest["source_sha256"] != frozen["source_sha256"]
        or Path(frozen["budget_root"]).resolve() != ROOT
        or execution != ROOT / "executions" / parent.name
        or sha(parent / "source.tar.gz") != frozen["archive_sha256"]
    ):
        raise ValueError("Parent freeze, manifest, archive, or canonical execution/budget binding mismatch")
    # Check frozen Python bytes, including launcher/imported infrastructure, not
    # only the source package hash. Never extract an archive into the workspace.
    archived = set()
    with tarfile.open(parent / "source.tar.gz", "r:gz") as archive:
        for member in archive.getmembers():
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts or not member.isfile():
                raise ValueError("Unexpected frozen archive member")
            if archive.extractfile(member).read() != (execution / relative).read_bytes():
                raise ValueError("Frozen execution file differs from its source archive")
            if relative.suffix == ".py":
                archived.add(relative)
    actual = {
        p.relative_to(execution)
        for directory in ("src", "infra", "scripts", "tests")
        for p in (execution / directory).rglob("*.py")
    }
    if actual != archived:
        raise ValueError("Frozen execution contains missing or additional Python files")
    check = subprocess.check_output(
        [
            str(ROOT / ".venv/bin/python"),
            "-c",
            "from curiosity_study.runner import source_hash; import budget; print(source_hash()); print(budget.ROOT)",
        ],
        cwd=execution,
        env=environment(execution),
        text=True,
    ).splitlines()
    if check != [frozen["source_sha256"], str(ROOT)]:
        raise ValueError("Frozen environment source or canonical budget root mismatch")
    descriptor = parent / "children.json"
    children = (
        read(descriptor)
        if descriptor.exists()
        else [dict(path=str(parent), manifest_sha256=digest(manifest))]
    )
    if not children:
        raise ValueError("A frozen collection must contain at least one job")
    jobs, child_assignments, child_groups = [], [], []
    for child in children:
        path = Path(child["path"]).resolve()
        cm = read(path / "manifest.json")
        if (
            path.parent != ROOT / "runs"
            or str(path) != child["path"]
            or digest(cm) != child["manifest_sha256"]
            or cm.get("source_sha256") != frozen["source_sha256"]
            or cm.get("execution_gpu") != "H200"
            or cm.get("model") != manifest.get("model")
            or cm.get("execution") != manifest.get("execution")
            or (path != parent and cm.get("parent_manifest_sha256") != digest(manifest))
        ):
            raise ValueError("Child manifest differs from its frozen parent/source binding")
        if any(
            (path / name).exists()
            for name in ("coordinator.json", "coordinator_end.json", "reservation.json")
        ):
            raise ValueError("Grouped admission requires unstarted child jobs")
        jobs.append(dict(path=str(path), manifest_sha256=digest(cm), execution_root=str(execution)))
        assigned = cm.get("assignments", []) + cm.get("diagnostics", [])
        if digest(assigned) != cm.get("assignment_sha256"):
            raise ValueError("Child assignment binding mismatch")
        child_assignments.extend(assigned)
        child_groups.append((assigned, child))
    expected = manifest.get("assignments", []) + manifest.get("diagnostics", [])
    expected_by_id = {a["trial_id"]: a for a in expected}
    observed_by_id = {a["trial_id"]: a for a in child_assignments}
    if (
        not expected
        or digest(expected) != manifest.get("assignment_sha256")
        or len(expected_by_id) != len(expected)
        or len(observed_by_id) != len(child_assignments)
        or digest(expected_by_id) != digest(observed_by_id)
    ):
        raise ValueError("Children must cover every frozen parent assignment exactly once")
    size = manifest["execution"]["batch_size"]
    if type(size) is not int or size < 1:
        raise ValueError("Require a positive frozen cohort size")
    original = {
        digest(expected[offset : offset + size]): offset // size for offset in range(0, len(expected), size)
    }
    observed_cohorts = []
    for assigned, child_descriptor in child_groups:
        indices = [
            original.get(digest(assigned[offset : offset + size])) for offset in range(0, len(assigned), size)
        ]
        if (
            None in indices
            or indices != sorted(indices)
            or ("parent_cohorts" in child_descriptor and child_descriptor["parent_cohorts"] != indices)
        ):
            raise ValueError("Children must preserve whole original parent cohorts and their local order")
        observed_cohorts.extend(indices)
    if sorted(observed_cohorts) != list(range(len(original))):
        raise ValueError("Every original parent cohort must be assigned exactly once")
    return dict(
        path=str(parent),
        manifest_sha256=digest(manifest),
        source_sha256=frozen["source_sha256"],
        freeze_sha256=digest(frozen),
        archive_sha256=frozen["archive_sha256"],
        children_sha256=digest(children),
        children_file_present=descriptor.exists(),
        execution_root=str(execution),
        launcher_sha256=sha(execution / "infra/launch_job.py"),
        jobs=jobs,
    )


def run(parents, seconds, out):
    if type(seconds) is not int or not 181 <= seconds <= 7200:
        raise ValueError("Require 181 to 7200 seconds; admission needs at least 180 seconds still remaining")
    parents, out = [Path(p).resolve() for p in parents], Path(out).resolve()
    cost_path = ROOT / "reports" / f"{out.name}-cost-and-shutdown.json"
    if len(parents) < 2 or len(set(parents)) != len(parents):
        raise ValueError("Require at least two distinct frozen parent collections")
    if out.parent != ROOT / "runs" or out.exists() or cost_path.exists():
        raise ValueError(
            "Use a new program output run and cost-receipt path; existing artifacts cannot be overwritten"
        )
    if Path(budget.ROOT).resolve() != ROOT:
        raise ValueError("Coordinator and children must share the canonical budget root")
    bindings = [frozen_parent(parent) for parent in parents]
    jobs = [job for parent in bindings for job in parent["jobs"]]
    if not 2 <= len(jobs) <= 4 or len({job["path"] for job in jobs}) != len(jobs):
        raise ValueError("Require two to four distinct total child jobs")
    coordinator = Path(__file__).read_bytes()
    deadline = time.time() + seconds
    plan = dict(
        schema_version=1,
        parents=bindings,
        jobs=jobs,
        gpu="H200",
        budget_root=str(ROOT),
        requested_seconds=seconds,
        common_deadline=deadline,
        total_jobs=len(jobs),
        coordinator_sha256=hashlib.sha256(coordinator).hexdigest(),
        reservation=str(out / "reservation.json"),
        cost_receipt=str(cost_path),
        parent_receipts_root=str(out / "collections"),
    )
    out.mkdir(parents=True, exist_ok=False)
    with (out / "coordinator.py").open("xb") as handle:
        handle.write(coordinator)
    write_once(out / "program.json", plan)
    # One admission binds every job before a thread can invoke a launcher.
    reservation = budget.reserve_jobs(
        [Path(j["path"]) for j in jobs], deadline, "H200", out / "reservation.json"
    )
    expected = {j["path"]: j["manifest_sha256"] for j in jobs}
    if reservation["jobs"] != expected or reservation["deadline"] != deadline:
        raise ValueError("Admission does not bind the complete frozen program and common deadline")
    print(
        f"Maximum combined compute liability ${reservation['maximum_combined_compute_usd']:.3f}", flush=True
    )

    def launch(job):
        path, execution = Path(job["path"]), Path(job["execution_root"])
        try:
            if digest(read(path / "manifest.json")) != job["manifest_sha256"]:
                raise ValueError("Child manifest changed after admission")
            log = out / "logs" / f"{path.name}.log"
            log.parent.mkdir(exist_ok=True)
            with log.open("x") as handle:
                result = subprocess.run(
                    [
                        str(ROOT / ".venv/bin/python"),
                        str(execution / "infra/launch_job.py"),
                        "--run",
                        str(path),
                        "--reservation",
                        str(out / "reservation.json"),
                    ],
                    cwd=execution,
                    env=environment(execution),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            return dict(run=str(path), exit_code=result.returncode)
        except Exception as error:
            return dict(run=str(path), exit_code=None, error=f"{type(error).__name__}: {error}")

    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        outcomes = list(pool.map(launch, jobs))
    write_once(out / "outcomes.json", outcomes)
    app_ids, coordinator_complete = set(), True
    for job in jobs:
        path = Path(job["path"])
        receipts = []
        for name in ("coordinator.json", "coordinator_end.json"):
            try:
                item = read(path / name)
                if item.get("app_id"):
                    app_ids.add(item["app_id"])
                receipts.append(item)
            except (OSError, ValueError):
                coordinator_complete = False
        if len(receipts) == 2:
            start, end = receipts
            coordinator_complete &= (
                bool(start.get("app_id"))
                and start["app_id"] == end.get("app_id")
                and start.get("deadline") == deadline
            )
    coordinator_complete &= len(app_ids) == len(jobs)
    state = budget.snapshot()
    for _ in range(12):
        if state["all_stopped_zero_tasks"] and app_ids <= {a["App ID"] for a in state["apps"]}:
            break
        time.sleep(5)
        state = budget.snapshot()
    state.update(
        program_sha256=digest(plan),
        program_app_ids=sorted(app_ids),
        program_apps_reported=app_ids <= {a["App ID"] for a in state["apps"]},
        all_program_coordinators_complete=bool(coordinator_complete),
    )
    write_once(cost_path, state)
    coverage = []
    for parent in bindings:
        selected = {j["path"] for j in parent["jobs"]}
        receipts = []
        for job in parent["jobs"]:
            try:
                receipt = check_collection_coverage(
                    Path(job["path"]), expected_manifest_sha256=job["manifest_sha256"]
                )
            except CollectionCoverageError as error:
                receipt = dict(passed=False, error=str(error))
            receipts.append(dict(run=job["path"], **receipt))
        directory = out / "collections" / Path(parent["path"]).name
        write_once(directory / "outcomes.json", [o for o in outcomes if o["run"] in selected])
        write_once(directory / "operational-coverage.json", receipts)
        coverage.extend(receipts)
    passed = (
        state["all_stopped_zero_tasks"]
        and state["program_apps_reported"]
        and coordinator_complete
        and all(o["exit_code"] == 0 for o in outcomes)
        and all(c["passed"] for c in coverage)
    )
    write_once(
        out / "completion.json",
        dict(passed=bool(passed), program_sha256=digest(plan), cost_receipt=str(cost_path)),
    )
    if not passed:
        raise RuntimeError(
            "Grouped collection incomplete or apps not verifiably stopped; reconcile before advancing"
        )
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, action="append", required=True)
    parser.add_argument("--seconds", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.parent, args.seconds, args.out)
