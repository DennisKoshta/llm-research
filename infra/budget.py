"""Local admission guard; the Oct 14 mechanism phase has a fresh $300 allowance."""

import datetime as dt
import json
import os
from pathlib import Path
import subprocess
from curiosity_study.resource_limits import SCREEN_BUDGET, job_wall_limit
from curiosity_study import curiosity_resource_limits as curiosity_limits
from curiosity_study.pain_profile_resources import app_price as pain_app_price

RATES = {
    "L40S": 0.000542 + 2 * 0.0000131 + 48 * 0.00000222,
    "H100": 0.001097 + 2 * 0.0000131 + 48 * 0.00000222,
    "H200": 0.001261 + 2 * 0.0000131 + 48 * 0.00000222,
}
RATE = RATES["L40S"]
PHASE = "internal-mechanisms-2026-10-14"
# Verified stopped-app baseline, preserved in the initial phase cost snapshot.
BASELINE_COMPUTE = 63.350812440000006
FRESH_COMPUTE_ALLOWANCE = 300.0
RESERVE = 10.0
BUDGET = BASELINE_COMPUTE + FRESH_COMPUTE_ALLOWANCE + RESERVE
ROOT = Path(os.environ.get("CURIOSITY_BUDGET_ROOT", Path(__file__).resolve().parents[1])).resolve()


def snapshot(now=None, *, timeout=None):
    now = now or dt.datetime.now(dt.timezone.utc)
    apps = json.loads(
        subprocess.check_output([str(ROOT / ".venv/bin/modal"), "app", "list", "--json"], text=True,
                                **({"timeout": timeout} if timeout is not None else {}))
    )
    known = {}
    for path in (ROOT / "reports").glob("*cost-and-shutdown.json"):
        for app in json.loads(path.read_text()).get("apps", []):
            known[app["App ID"]] = app
    known.update({a["App ID"]: a for a in apps if a.get("Description", "").startswith("curiosity-study")})
    charges = []
    for a in known.values():
        # Explicit GPU app names retain their rate even if a client loses its coordinator.
        description = a.get("Description", "")
        gpu = "H200" if description.endswith("-h200") else "H100" if description.endswith("-h100") else "L40S"
        custom = pain_app_price(description)
        if description == "curiosity-study-pain-deterministic-h10096":
            custom = {"gpu": "H100", "rate_usd_per_second": 0.001097 + 2 * 0.0000131 + 96 * 0.00000222,
                      "resource_profile": "pain-deterministic-diagnostic-2cpu-96gib"}
        if description in {"curiosity-study-pain-deterministic-control-setup-h10096",
                           "curiosity-study-pain-deterministic-control-benchmark-h10096"}:
            custom = {"gpu": "H100", "rate_usd_per_second": 0.001097 + 2 * 0.0000131 + 96 * 0.00000222,
                      "resource_profile": "pain-deterministic-control-2cpu-96gib"}
        rate = custom["rate_usd_per_second"] if custom else RATES[gpu]
        if custom:
            gpu = custom["gpu"]
        end = dt.datetime.fromisoformat(a["Stopped at"]) if a.get("Stopped at") else now
        seconds = max(0, (end - dt.datetime.fromisoformat(a["Created at"])).total_seconds())
        charges.append(
            {
                "app_id": a["App ID"],
                "gpu": gpu,
                "app_wall_seconds": seconds,
                "rate_usd_per_second": rate,
                "compute_estimate_usd": seconds * rate,
                **({"resource_profile": custom["resource_profile"]} if custom else {}),
            }
        )
    estimate = sum(c["compute_estimate_usd"] for c in charges)
    return {
        "apps": list(known.values()),
        "app_charges": charges,
        "all_stopped_zero_tasks": all(
            a["State"] == "stopped" and str(a["Tasks"]) == "0" for a in known.values()
        ),
        "cumulative_compute_estimate_usd": estimate,
        "authorization_phase": PHASE,
        "historical_compute_baseline_usd": BASELINE_COMPUTE,
        "fresh_compute_allowance_usd": FRESH_COMPUTE_ALLOWANCE,
        "phase_compute_estimate_usd": max(0, estimate - BASELINE_COMPUTE),
        "budget_usd": BUDGET,
        "storage_transfer_reserve_usd": RESERVE,
        "available_compute_estimate_usd": max(
            0, FRESH_COMPUTE_ALLOWANCE - max(0, estimate - BASELINE_COMPUTE)
        ),
        "timestamp_utc": now.isoformat(),
        "caveat": "Conservative app-wall estimate at configured GPU/CPU/RAM rates; invoice/credits not read; dedicated volumes retained.",
    }


def admission_seconds(gpu="L40S", now=None):
    if gpu not in RATES:
        raise ValueError("Unpriced GPU is not admitted")
    state = snapshot(now)
    if not state["all_stopped_zero_tasks"]:
        raise RuntimeError("A project app is still active; reconcile it before launching another GPU job")
    spent = state["cumulative_compute_estimate_usd"]
    available_seconds = int(state["available_compute_estimate_usd"] / RATES[gpu]) - 120
    if available_seconds < 180:
        raise RuntimeError(
            f"Pilot compute allowance exhausted: estimated ${spent:.2f}; ${RESERVE:.2f} reserved"
        )
    return min(7200, available_seconds), {
        "budget_usd": BUDGET,
        "reserve_usd": RESERVE,
        "prior_compute_estimate_usd": spent,
        "rate_usd_per_second": RATES[gpu],
        "gpu": gpu,
        "checked_utc": state["timestamp_utc"],
    }


def _bound_job_limits(manifests):
    """Validate the complete screen reservation; never combine it with other jobs."""
    if any(m.get("assay") == curiosity_limits.ASSAY for m in manifests):
        return curiosity_limits.reservation_limits(manifests)
    limits = [job_wall_limit(manifest) for manifest in manifests]
    if any(manifest.get("assay") == "satisfaction_screen" for manifest in manifests):
        from curiosity_study.artifacts import digest

        groups = {"baseline_zero", "positive_half", "positive_full"}
        parents = {manifest.get("parent_manifest_sha256") for manifest in manifests}
        parent = next(iter(parents)) if len(parents) == 1 else None
        if (
            len(manifests) != SCREEN_BUDGET["workers"]
            or any(manifest.get("assay") != "satisfaction_screen" for manifest in manifests)
            or {manifest.get("execution_group") for manifest in manifests} != groups
            or len({digest(manifest) for manifest in manifests}) != len(manifests)
            or not isinstance(parent, str)
            or len(parent) != 64
            or any(character not in "0123456789abcdef" for character in parent)
        ):
            raise ValueError("Screen reservation requires three distinct groups from one bound parent")
        return min(limits), SCREEN_BUDGET["stage_ceiling_usd"]
    return min(limits), None


def _screen_claim_policy(reservation):
    """Re-read all child bindings before a screen worker can consume its slot."""
    from curiosity_study.artifacts import read, digest

    manifests = []
    for root, expected in reservation["jobs"].items():
        manifest = read(Path(root) / "manifest.json")
        if digest(manifest) != expected or manifest.get("execution_gpu", "L40S") != reservation["gpu"]:
            raise ValueError("Screen reservation child binding changed")
        manifests.append(manifest)
    limit, ceiling = _bound_job_limits(manifests)
    seconds = reservation["deadline"] - reservation["created"]
    liability = len(manifests) * (seconds + 120) * RATES[reservation["gpu"]]
    if not 180 <= seconds <= limit or liability > ceiling:
        raise ValueError("Screen reservation exceeds its original wall-time or stage ceiling")
    if (
        reservation["rate_usd_per_second"] != RATES[reservation["gpu"]]
        or reservation["maximum_combined_compute_usd"] != liability
    ):
        raise ValueError("Screen reservation liability binding changed")
    return limit


def reserve_jobs(jobs, deadline, gpu, destination):
    """Reserve all workers together before any launch; bind exact child manifests."""
    import time
    from curiosity_study.artifacts import read, digest, write_once

    state = snapshot()
    if not state["all_stopped_zero_tasks"]:
        raise RuntimeError("Stop existing project apps before creating a parallel reservation")
    if gpu not in RATES or not 1 <= len(jobs) <= 4:
        raise ValueError("Unsupported reservation size or GPU")
    bound = {}
    manifests = []
    for root in jobs:
        m = read(Path(root) / "manifest.json")
        if m.get("execution_gpu", "L40S") != gpu:
            raise ValueError("Reservation GPU mismatch")
        key = str(Path(root).resolve())
        if key in bound:
            raise ValueError("Reservation contains a duplicate worker path")
        bound[key] = digest(m)
        manifests.append(m)
    limit, stage_ceiling = _bound_job_limits(manifests)
    created = time.time()
    seconds = deadline - created
    if not 180 <= seconds <= limit:
        raise ValueError("Reservation deadline is expired or excessive")
    liability = len(jobs) * (seconds + 120) * RATES[gpu]
    if stage_ceiling is not None and liability > stage_ceiling:
        raise RuntimeError("Combined worker reservation exceeds the screen stage ceiling")
    if liability > state["available_compute_estimate_usd"]:
        raise RuntimeError("Combined worker reservation exceeds remaining pilot allowance")
    reservation = {
        "deadline": deadline,
        "created": created,
        "gpu": gpu,
        "jobs": bound,
        "budget_usd": BUDGET,
        "reserve_usd": RESERVE,
        "prior_compute_estimate_usd": state["cumulative_compute_estimate_usd"],
        "rate_usd_per_second": RATES[gpu],
        "maximum_combined_compute_usd": liability,
    }
    write_once(destination, reservation)
    return reservation


def claim_reservation(path, root, manifest, gpu, recover_unstarted=False):
    import time
    import uuid
    from curiosity_study.artifacts import read, digest, write_once

    reservation = read(path)
    key = str(Path(root).resolve())
    if reservation["jobs"].get(key) != digest(manifest) or reservation["gpu"] != gpu:
        raise ValueError("Run is not bound to this exact reservation")
    if reservation["budget_usd"] != BUDGET or not 1 <= len(reservation["jobs"]) <= 4:
        raise ValueError("Reservation does not match the current allowance")
    limit = curiosity_limits.job_wall_limit(manifest)
    if manifest.get("assay") in ("satisfaction_screen", curiosity_limits.ASSAY):
        limit = _screen_claim_policy(reservation)
    if not 0 < reservation["deadline"] - time.time() <= limit:
        raise TimeoutError("Reservation deadline elapsed")
    claim_path = Path(path).parent / "claims" / f"{Path(root).name}.json"
    if recover_unstarted:
        root = Path(root)
        end_path = root / "coordinator_end.json"
        if not claim_path.exists() or (root / "coordinator.json").exists() or not end_path.exists():
            raise ValueError("Recovery is only allowed for a failed, never-started app")
        prior = read(end_path)
        if prior.get("app_id") is not None or prior.get("cancel_requested"):
            raise ValueError("An app was started; reconcile its GPU state instead")
        directory = root / "unstarted_admissions"
        directory.mkdir(exist_ok=True)
        attempt = len(list(directory.glob("claim-*.json")))
        if attempt >= 3:
            raise RuntimeError("Unstarted admission recovery limit exhausted")
        write_once(directory / f"claim-{attempt}.json", {"recovery_id": uuid.uuid4().hex, "prior_end": prior})
        end_path.rename(directory / f"end-{attempt}.json")
        return reservation["deadline"], reservation
    write_once(
        claim_path,
        {"claim_id": uuid.uuid4().hex, "run": key, "manifest_sha256": digest(manifest), "time": time.time()},
    )
    return reservation["deadline"], reservation


def claim_reserved_resume(path, root, manifest, gpu, coordinator_path):
    """Replace a stopped reserved worker, retaining its original deadline and slot."""
    import time
    import uuid
    from curiosity_study.artifacts import read, digest, write_once

    if manifest.get("assay") == curiosity_limits.ASSAY:
        curiosity_limits.job_wall_limit(manifest)
        raise ValueError("Curiosity suppression policy forbids paid resume attempts")
    job_wall_limit(manifest)
    if manifest.get("assay") == "satisfaction_screen":
        raise ValueError("Satisfaction screen policy forbids paid resume attempts")
    root = Path(root).resolve()
    reservation = read(path)
    if reservation["jobs"].get(str(root)) != digest(manifest) or reservation["gpu"] != gpu:
        raise ValueError("Resume is not bound to this exact reservation")
    if reservation["budget_usd"] != BUDGET or not 1 <= len(reservation["jobs"]) <= 4:
        raise ValueError("Reservation does not match the current allowance")
    seconds = reservation["deadline"] - time.time()
    if not 180 <= seconds <= 7200:
        raise TimeoutError("Insufficient original reservation time remains")
    if read(root / "coordinator.json")["deadline"] != reservation["deadline"]:
        raise ValueError("Reserved resume cannot change the original deadline")
    claims = Path(path).parent / "claims"
    if not (claims / f"{root.name}.json").exists():
        raise ValueError("Initial reserved admission is missing")
    prior_resumes = sorted(claims.glob(f"resume-{root.name}-*.json"))
    if len(prior_resumes) >= 2:
        raise RuntimeError("Reserved resume limit exhausted")
    for claim in prior_resumes:
        prior = Path(read(claim)["coordinator_path"])
        if not prior.with_name(prior.stem + "_end.json").exists():
            raise RuntimeError("A reserved resume client is still outstanding")
    state = snapshot()
    apps = {a["App ID"]: a for a in state["apps"]}
    active = {aid for aid, a in apps.items() if a["State"] != "stopped" or str(a["Tasks"]) != "0"}
    allowed_active = set()
    for bound_root in reservation["jobs"]:
        directory = Path(bound_root)
        coordinators = [directory / "coordinator.json"] + list((directory / "coordinators").glob("*.json"))
        ids = set()
        for file in coordinators:
            if file.exists() and not file.stem.endswith("_end"):
                aid = read(file).get("app_id")
                if aid:
                    ids.add(aid)
        if directory.resolve() == root:
            if not ids or any(aid not in apps or aid in active for aid in ids):
                raise RuntimeError("Prior target apps must be verified stopped with zero tasks")
        elif len(ids & active) > 1:
            raise RuntimeError("Multiple workers occupy a reserved slot")
        allowed_active.update(ids)
    if active - allowed_active:
        raise RuntimeError("An unrelated project app is active")
    liability = len(reservation["jobs"]) * (seconds + 120) * RATES[gpu]
    if liability > state["available_compute_estimate_usd"]:
        raise RuntimeError("Remaining reserved workers exceed the current compute allowance")
    write_once(
        claims / f"resume-{root.name}-{len(prior_resumes)}.json",
        {
            "claim_id": uuid.uuid4().hex,
            "coordinator_path": str(Path(coordinator_path).resolve()),
            "time": time.time(),
            "prior_compute_estimate_usd": state["cumulative_compute_estimate_usd"],
            "remaining_combined_compute_usd": liability,
        },
    )
    return reservation["deadline"], reservation
