"""Run two separately admitted waves from the frozen execution copy."""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import time
from curiosity_study.artifacts import read, write_once

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "infra"))
from budget import reserve_jobs, snapshot  # noqa: E402


def main():
    parent = ROOT / "runs/qwen32-thinking-state-v1-validation"
    freeze = read(parent / "freeze.json")
    execution = Path(freeze["execution_root"])
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
    assert check == [freeze["source_sha256"], str(ROOT)], check
    for wave, paths in enumerate(freeze["waves"]):
        jobs = [Path(p) for p in paths]
        reservation = parent / f"wave-{wave}-reservation.json"
        reserved = reserve_jobs(jobs, time.time() + 5400, "H200", reservation)
        print(
            f"Wave {wave}: maximum combined liability ${reserved['maximum_combined_compute_usd']:.3f}",
            flush=True,
        )

        def launch(path):
            with (path / "launch.log").open("w") as log:
                result = subprocess.run(
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
            print(path.name, "exit", result.returncode, flush=True)
            return dict(run=str(path), exit_code=result.returncode)

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(launch, jobs))
        write_once(parent / f"wave-{wave}-outcomes.json", results)
        state = snapshot()
        for _ in range(12):
            if state["all_stopped_zero_tasks"]:
                break
            time.sleep(5)
            state = snapshot()
        write_once(ROOT / "reports" / f"thinking-state-validation-wave-{wave}-cost-and-shutdown.json", state)
        if not state["all_stopped_zero_tasks"] or any(r["exit_code"] for r in results):
            raise RuntimeError(
                "Wave incomplete or still active; preserve data and reconcile before continuing"
            )
        print(f"Wave {wave} stopped; phase estimate ${state['phase_compute_estimate_usd']:.3f}", flush=True)


if __name__ == "__main__":
    main()
