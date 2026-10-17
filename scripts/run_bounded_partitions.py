"""Execute one admitted wave from immutable sources, then reconcile all workers."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import time
from curiosity_study.artifacts import read, write_once, digest
from collection_coverage import check_collection_coverage, CollectionCoverageError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "infra"))
from budget import RATES, reserve_jobs, snapshot  # noqa:E402


def validate_stage_budget(manifest, workers, seconds):
    """An optional frozen stage limit can only narrow the standing allowance."""
    policy = manifest.get("budget")
    if policy is None:
        return
    if (
        workers != policy["workers"]
        or seconds != policy["shared_deadline_seconds"]
        or workers * (seconds + 120) * RATES["H200"] > policy["stage_ceiling_usd"]
    ):
        raise ValueError("Requested execution differs from the frozen stage resource bound")


def main(parent, seconds):
    parent = parent.resolve()
    frozen = read(parent / "freeze.json")
    manifest = read(parent / "manifest.json")
    assert digest(manifest) == frozen["manifest_sha256"]
    execution = Path(frozen["execution_root"])
    children = (
        read(parent / "children.json")
        if (parent / "children.json").exists()
        else [dict(path=str(parent), manifest_sha256=frozen["manifest_sha256"])]
    )
    jobs = [Path(c["path"]) for c in children]
    validate_stage_budget(manifest, len(jobs), seconds)
    env = {
        **os.environ,
        "CURIOSITY_MODAL_GPU": "H200",
        "CURIOSITY_BUDGET_ROOT": str(ROOT),
        "PYTHONPATH": str(execution / "src") + os.pathsep + str(execution / "infra"),
    }
    check = subprocess.check_output(
        [
            str(ROOT / ".venv/bin/python"),
            "-c",
            "from curiosity_study.runner import source_hash; import budget; print(source_hash()); print(budget.ROOT)",
        ],
        cwd=execution,
        env=env,
        text=True,
    ).splitlines()
    assert check == [frozen["source_sha256"], str(ROOT)]
    reservation = parent / "reservation.json"
    admitted = reserve_jobs(jobs, time.time() + seconds, "H200", reservation)
    print(f"Maximum combined compute liability ${admitted['maximum_combined_compute_usd']:.3f}", flush=True)

    def launch(path):
        with (path / "launch.log").open("w") as log:
            r = subprocess.run(
                [
                    str(ROOT / ".venv/bin/python"),
                    str(execution / "infra/launch_job.py"),
                    "--run",
                    str(path),
                    "--reservation",
                    str(reservation),
                ],
                cwd=execution,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        print(path.name, "exit", r.returncode, flush=True)
        return dict(run=str(path), exit_code=r.returncode)

    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        outcomes = list(pool.map(launch, jobs))
    write_once(parent / "outcomes.json", outcomes)
    state = snapshot()
    for _ in range(12):
        if state["all_stopped_zero_tasks"]:
            break
        time.sleep(5)
        state = snapshot()
    write_once(ROOT / "reports" / f"{parent.name}-cost-and-shutdown.json", state)
    print(
        dict(
            stopped=state["all_stopped_zero_tasks"],
            phase_compute_estimate_usd=state["phase_compute_estimate_usd"],
            available_compute_estimate_usd=state["available_compute_estimate_usd"],
        ),
        flush=True,
    )
    coverage = []
    for child in children:
        try:
            receipt = check_collection_coverage(
                Path(child["path"]), expected_manifest_sha256=child["manifest_sha256"]
            )
        except CollectionCoverageError as error:
            receipt = dict(passed=False, error=str(error))
        coverage.append(dict(run=child["path"], **receipt))
    write_once(parent / "operational-coverage.json", coverage)
    if (
        not state["all_stopped_zero_tasks"]
        or any(o["exit_code"] for o in outcomes)
        or any(not c["passed"] for c in coverage)
    ):
        raise RuntimeError("Collection incomplete or apps not stopped; reconcile before advancing")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--seconds", type=int, required=True)
    a = p.parse_args()
    main(a.parent, a.seconds)
