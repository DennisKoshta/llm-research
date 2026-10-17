"""Bounded execution-only continuation of an immutable validation manifest."""

import argparse
import copy
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import tarfile
from budget import reserve_jobs, snapshot, RATES, BUDGET, RESERVE
from curiosity_study.artifacts import RunStore, read, write_once, digest
from curiosity_study.report import render


def merge(parent, children):
    m = read(parent / "manifest.json")
    store = RunStore(parent, m)
    expected = {a["trial_id"]: a for a in m["diagnostics"] + m["assignments"]}
    existing = store.completed()
    provenance = []
    pending = []
    seen = set(existing)
    # Validate every partition before publishing any derived parent envelopes.
    for child in children:
        cm = read(child / "manifest.json")
        for key in (
            "source_sha256",
            "model",
            "task_revision",
            "configuration",
            "execution_gpu",
            "planned_behavior_sha256",
        ):
            if cm[key] != m[key]:
                raise ValueError(f"Partition mismatch: {key}")
        for key in ("phase", "assay", "validation_prerequisite", "execution"):
            if cm.get(key) != m.get(key):
                raise ValueError(f"Partition mismatch: {key}")
        assigned = {a["trial_id"]: a for a in cm["diagnostics"] + cm["assignments"]}
        rows = RunStore(child, cm).completed()
        if set(rows) != set(assigned):
            raise ValueError(f"Incomplete partition: {child}")
        audits = [read(p) for p in (child / "integrity").glob("*.json")]
        if not audits or not all(a["passed"] for a in audits):
            raise ValueError(f"Missing or failed integrity check: {child}")
        for tid, r in rows.items():
            if tid in seen or assigned[tid] != expected.get(tid) or r["assignment"] != expected[tid]:
                raise ValueError(f"Overlapping or foreign assignment: {tid}")
            seen.add(tid)
            pending.append((child, r))
            provenance.append(
                {
                    "trial_id": tid,
                    "child": str(child),
                    "child_manifest_sha256": digest(cm),
                    "record_sha256": digest(r),
                }
            )
    if seen != set(expected):
        raise ValueError("Aggregate does not cover the original frozen sample")
    for child, r in pending:
        store.save(r, f"partition-{child.name}")
        existing[r["trial_id"]] = r
    for child in children:
        for directory in ("integrity", "environments", "profiles", "cloud_profiles"):
            for source in (child / directory).glob("*.json"):
                target = parent / directory / f"{child.name}-{source.name}"
                write_once(target, read(source))
    write_once(
        parent / "parallel" / "aggregation.json",
        {
            "parent_manifest_sha256": digest(m),
            "original_records_retained": len(existing) - len(provenance),
            "derived_records": provenance,
            "all_assignments_complete": True,
        },
    )
    (parent / "report.md").write_text(render(parent))
    return len(existing)


def partition_assignments(manifest, completed, workers, field):
    """Retain original batch cohorts; serial manifests keep their historical rule."""
    if not 1 <= workers <= 4:
        raise ValueError("At most four workers are permitted")
    if manifest.get("execution", {}).get("backend") != "torch_batched_v1":
        remaining = [a for a in manifest[field] if a["trial_id"] not in completed]
        return [
            remaining[i::workers] for i in range(workers)
        ], "unfinished assignments in original order, round-robin"
    size = manifest["execution"]["batch_size"]
    full, partial = [], []
    for offset in range(0, len(manifest[field]), size):
        cohort = [a for a in manifest[field][offset : offset + size] if a["trial_id"] not in completed]
        if cohort:
            (full if len(cohort) == size else partial).append(cohort)
    if len(partial) > workers:
        raise ValueError("Too many partial original cohorts to preserve batch boundaries")
    parts = [[] for _ in range(workers)]
    for i, cohort in enumerate(full):
        parts[i % workers].extend(cohort)
    # A short cohort must end its child, never combine with another original cohort.
    for i, cohort in enumerate(partial):
        parts[i].extend(cohort)
    if any(not part for part in parts):
        raise ValueError("Reduce workers: empty cohort partition")
    return parts, "whole original batch cohorts round-robin; at most one partial cohort last per child"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--new", action="store_true", help="Create a bounded parent coordinator before any cloud launch"
    )
    args = p.parse_args()
    parent = args.run.resolve()
    m = read(parent / "manifest.json")
    if m["assay"] != "one_credit" or m["phase"] not in ("validation", "behavior"):
        raise ValueError("Only the frozen one-credit assay supports this partitioner")
    field = "diagnostics" if m["phase"] == "validation" else "assignments"
    other_field = "assignments" if field == "diagnostics" else "diagnostics"
    if m[other_field]:
        raise ValueError("Mixed-phase partitions are not supported")
    if m["phase"] == "behavior" and not m.get("validation_prerequisite", {}).get("result", {}).get("passed"):
        raise ValueError("Behavior requires its frozen passing validation prerequisite")
    from curiosity_study.runner import source_hash

    if source_hash() != m["source_sha256"]:
        raise ValueError("Scientific source changed since freeze")
    completed = RunStore(parent, m).completed()
    remaining = [a for a in m[field] if a["trial_id"] not in completed]
    if not remaining:
        raise ValueError("Parent is already complete")
    if args.new:
        state = snapshot()
        if not state["all_stopped_zero_tasks"]:
            raise RuntimeError("Stop existing workers before admitting a new collection")
        if not 1 <= args.workers <= 4:
            raise ValueError("At most four workers are permitted")
        seconds = min(
            7200,
            int(state["available_compute_estimate_usd"] / (args.workers * RATES[m["execution_gpu"]])) - 120,
        )
        if seconds < 180:
            raise RuntimeError("Insufficient remaining reserved compute allowance")
        write_once(
            parent / "coordinator.json",
            {
                "start": time.time(),
                "deadline": time.time() + seconds,
                "partitioned_from_start": True,
                "authorized_budget_usd": BUDGET,
                "reserve_usd": RESERVE,
                "workers": args.workers,
            },
        )
    deadline = read(parent / "coordinator.json")["deadline"]
    parts, partition_rule = partition_assignments(m, completed, args.workers, field)
    children = []
    for i in range(args.workers):
        child = parent.with_name(parent.name + f"-part{i}")
        if child.exists():
            raise ValueError("Use immutable, new partition directories")
        cm = copy.deepcopy(m)
        cm[field] = parts[i]
        cm["execution_partition"] = {
            "parent_manifest_sha256": digest(m),
            "partition": i,
            "partitions": args.workers,
            "original_completed": len(completed),
            "rule": partition_rule,
        }
        write_once(child / "manifest.json", cm)
        children.append(child)
    reservation_path = parent / "parallel" / "reservation.json"
    reservation = reserve_jobs(children, deadline, m["execution_gpu"], reservation_path)
    archive = parent / "parallel" / "execution-source.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in [
            "src",
            "tests",
            "infra",
            "scripts",
            "configs",
            "pyproject.toml",
            "uv.lock",
            "reports/one-credit-v2-execution-amendment.md",
        ]:
            tar.add(path, arcname=path, filter=lambda x: None if "__pycache__" in x.name else x)
    processes = {}
    recoveries = {child: 0 for child in children}

    def launch(child, recover=False):
        command = [
            sys.executable,
            "infra/launch_job.py",
            "--run",
            str(child),
            "--reservation",
            str(reservation_path),
        ]
        if recover:
            command.append("--recover-unstarted")
        with (child / ("recovery-launch.log" if recover else "launch.log")).open(
            "a" if recover else "x"
        ) as handle:
            return subprocess.Popen(
                command,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=dict(os.environ, CURIOSITY_MODAL_GPU=m["execution_gpu"]),
            )

    for index, child in enumerate(children):
        shutil.copyfile(archive, child / "source.tar.gz")
        if index:
            time.sleep(3)  # Avoid bursts at Modal's app-creation endpoint.
        processes[child] = launch(child)
    print(
        f"Retained {len(completed)} terminal records; distributed {len(remaining)} over {len(children)} workers. Combined maximum compute reservation ${reservation['maximum_combined_compute_usd']:.2f}.",
        flush=True,
    )
    failures = []
    while processes:
        for child, proc in list(processes.items()):
            code = proc.poll()
            if code is None:
                continue
            if code:
                end_path = child / "coordinator_end.json"
                end = read(end_path) if end_path.exists() else {}
                log = child / ("recovery-launch.log" if recoveries[child] else "launch.log")
                limited = log.exists() and "App create rate limit exceeded" in log.read_text()
                if (
                    limited
                    and end_path.exists()
                    and end.get("app_id") is None
                    and not (child / "coordinator.json").exists()
                    and recoveries[child] < 3
                    and deadline - time.time() > 180
                ):
                    recoveries[child] += 1
                    time.sleep(3 * recoveries[child])
                    processes[child] = launch(child, recover=True)
                    continue
                failures.append((str(child), code))
            del processes[child]
        if processes:
            time.sleep(1)
    if failures:
        raise RuntimeError(f"Inspect incomplete workers before aggregation: {failures}")
    print(f"Aggregated {merge(parent, children)} original assignments with provenance", flush=True)
    write_once(
        parent / "parallel" / "controller_end.json", {"end": time.time(), "all_child_clients_complete": True}
    )


if __name__ == "__main__":
    main()
