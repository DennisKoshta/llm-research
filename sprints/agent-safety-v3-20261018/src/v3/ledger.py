"""Spend ledger for the v3 sprint (budget: $500 total, reset by the user 2026-10-18).

Two independent views of spend:
  * reservations: conservative local estimate (GPU rate x worst-case runtime) written BEFORE launch
  * billing:      `modal billing report` filtered to apps named agent-safety-v3-20261018-*
Launchers must call `reserve()` first; it refuses when reservations + billed would exceed the cap
or when too many GPU jobs would run at once.

CLI:  python -m v3.ledger status | reserve JOB USD [--gpus N] | release JOB [--actual USD] | reconcile [--note TEXT]
"""
from __future__ import annotations

import argparse
import fcntl
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
LEDGER_DIR = ROOT / "ledger"
RES_FILE = LEDGER_DIR / "reservations.jsonl"
APP_PREFIX = "agent-safety-v3-20261018"
CAP_USD = 500.0
SOFT_STOP_USD = 480.0      # no new launches above this without an explicit decision (raised 450 -> 480 on 2026-10-19 by the main loop: billed $410, remaining work bounded; hard cap stays 500)
MAX_CONCURRENT_GPUS = 16   # total GPUs allowed in flight across all jobs

# $/hour, from `modal billing rates` (2026-10-18)
GPU_RATE_HR = {"T4": 0.59, "L4": 0.80, "A10G": 1.10, "L40S": 1.95, "A100-40GB": 2.10,
               "A100-80GB": 2.50, "H100": 3.95, "H200": 4.54, "B200": 6.25}
CPU_RATE_HR = 0.0473   # per core
MEM_RATE_HR = 0.008    # per GiB


def est_cost(gpu: str, n_gpus: int, hours: float, cpu: int = 8, mem_gib: int = 64) -> float:
    return round(hours * (GPU_RATE_HR[gpu] * n_gpus + CPU_RATE_HR * cpu + MEM_RATE_HR * mem_gib), 2)


def _read() -> list[dict]:
    if not RES_FILE.exists():
        return []
    return [json.loads(l) for l in RES_FILE.read_text().splitlines() if l.strip()]


def _append(row: dict) -> None:
    LEDGER_DIR.mkdir(exist_ok=True)
    with open(RES_FILE, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(row, sort_keys=True) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)


def state() -> dict:
    """Fold the append-only log into per-job state."""
    jobs: dict[str, dict] = {}
    for r in _read():
        if r["event"] == "reconcile":
            continue
        j = jobs.setdefault(r["job"], {"job": r["job"]})
        if r["event"] == "reserve":
            j.update(usd_reserved=r["usd"], gpus=r.get("gpus", 0), status="running", t_reserve=r["t"])
        elif r["event"] == "release":
            j.update(status="done", usd_actual=r.get("actual"), t_release=r["t"])
    return jobs


def billed_usd() -> float | None:
    """Sum of billed cost for v3-prefixed apps (billing lags by up to ~1h). None if unavailable."""
    try:
        out = subprocess.run(
            [str(ROOT / ".venv/bin/modal"), "billing", "report", "--for", "this month",
             "--resolution", "d", "--json"],
            capture_output=True, text=True, timeout=120)
        rows = json.loads(out.stdout)
        return round(sum(float(r["cost"]) for r in rows if (r.get("description") or "").startswith(APP_PREFIX)), 2)
    except Exception:
        return None


def last_reconcile() -> dict | None:
    rows = [r for r in _read() if r["event"] == "reconcile"]
    return rows[-1] if rows else None


def summary() -> dict:
    jobs = state()
    running = [j for j in jobs.values() if j.get("status") == "running"]
    done = [j for j in jobs.values() if j.get("status") == "done"]
    # committed = actual where known, else the reservation. Launcher wall-clock "actuals" overstate jobs whose local client
    # kept waiting after the remote work ended, so a `reconcile` event replaces everything reserved before it by the billed
    # amount at that moment (jobs still running at reconcile time keep their reservation on top).
    rec = last_reconcile()

    def usd(j):
        return j.get("usd_actual") if j.get("usd_actual") is not None else j.get("usd_reserved", 0)

    if rec:
        committed = rec["billed"] + sum(usd(j) for j in jobs.values()
                                        if j.get("t_reserve", 0) > rec["t"] or (j.get("status") == "running" and j.get("t_reserve", 0) <= rec["t"]))
    else:
        committed = sum(usd(j) for j in jobs.values())
    return {"cap": CAP_USD, "soft_stop": SOFT_STOP_USD, "committed_usd": round(committed, 2),
            "running_jobs": [j["job"] for j in running], "running_gpus": sum(j.get("gpus", 0) for j in running),
            "n_done": len(done), "billed_usd_v3_prefix": billed_usd()}


def reserve(job: str, usd: float, gpus: int = 1) -> dict:
    s = summary()
    if job in state() and state()[job].get("status") == "running":
        raise RuntimeError(f"{job} already reserved and not released")
    billed = s["billed_usd_v3_prefix"] or 0.0
    # billing may lag, so take the larger of billed and locally-committed spend
    projected = max(billed, s["committed_usd"]) + usd
    if projected > SOFT_STOP_USD:
        raise RuntimeError(f"budget: projected ${projected:.2f} > soft stop ${SOFT_STOP_USD} (cap ${CAP_USD})")
    if s["running_gpus"] + gpus > MAX_CONCURRENT_GPUS:
        raise RuntimeError(f"concurrency: {s['running_gpus']}+{gpus} GPUs > {MAX_CONCURRENT_GPUS}")
    row = {"event": "reserve", "job": job, "usd": round(usd, 2), "gpus": gpus, "t": time.time()}
    _append(row)
    return row


def reconcile(note: str = "", buffer: float = 0.0) -> dict:
    """Pin the committed baseline to the billed amount. Only call when billing has caught up (no job finished in the last
    hour) and no GPU job is running unreserved; jobs reserved afterwards are added on top."""
    billed = billed_usd()
    if billed is None:
        raise RuntimeError("billing report unavailable; cannot reconcile")
    # `buffer` covers the billing lag (up to ~1h of spend that is not yet in the report) when jobs ran recently
    row = {"event": "reconcile", "job": "__reconcile__", "billed": round(billed + buffer, 2), "billed_raw": billed, "buffer": buffer,
           "note": note, "t": time.time()}
    _append(row)
    return row


def release(job: str, actual: float | None = None) -> dict:
    row = {"event": "release", "job": job, "actual": actual, "t": time.time()}
    _append(row)
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    r = sub.add_parser("reserve"); r.add_argument("job"); r.add_argument("usd", type=float); r.add_argument("--gpus", type=int, default=1)
    q = sub.add_parser("release"); q.add_argument("job"); q.add_argument("--actual", type=float)
    c = sub.add_parser("reconcile"); c.add_argument("--note", default=""); c.add_argument("--buffer", type=float, default=0.0)
    a = ap.parse_args(argv)
    if a.cmd == "status":
        print(json.dumps(summary(), indent=1))
    elif a.cmd == "reserve":
        print(json.dumps(reserve(a.job, a.usd, a.gpus)))
    elif a.cmd == "reconcile":
        print(json.dumps(reconcile(a.note, a.buffer)))
    else:
        print(json.dumps(release(a.job, a.actual)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
