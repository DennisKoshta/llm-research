"""Separate prospective bounds for setup, calibration, and one validation.

Unwired while the preceding study runs. No retries or budget transfers between
these stages; their ceilings sum to the declared USD65 extension maximum.
"""

from . import curiosity_resource_limits as previous
from . import satisfaction_causal_sites as study
from .artifacts import digest

SETUP_ASSAY = "satisfaction_site_setup"
CAMPAIGN_CEILING = 65
POLICIES = {
    "setup": dict(workers=1, shared_deadline_seconds=1800, stage_ceiling_usd=3,
                  paid_retries=0, campaign_ceiling_usd=CAMPAIGN_CEILING),
    "calibration": dict(workers=3, shared_deadline_seconds=7800, stage_ceiling_usd=34,
                        paid_retries=0, campaign_ceiling_usd=CAMPAIGN_CEILING),
    "validation": dict(workers=3, shared_deadline_seconds=6300, stage_ceiling_usd=28,
                       paid_retries=0, campaign_ceiling_usd=CAMPAIGN_CEILING),
}


def phase_of(manifest):
    if manifest.get("assay") == SETUP_ASSAY:
        return "setup"
    if manifest.get("assay") == study.ASSAY:
        phase = manifest.get("site_phase")
        if phase not in ("calibration", "validation") or manifest.get("site_plan", {}).get("phase") != phase:
            raise ValueError("Require the same explicit scientific phase in manifest and plan")
        return phase
    return None


def job_wall_limit(manifest):
    phase = phase_of(manifest)
    if phase is None:
        return previous.job_wall_limit(manifest)
    policy = manifest.get("budget")
    if (policy != POLICIES[phase] or not isinstance(policy, dict)
            or any(type(policy.get(key)) is not int for key in POLICIES[phase])):
        raise ValueError("Require the exact separate site-study stage policy")
    return POLICIES[phase]["shared_deadline_seconds"]


def reservation_limits(manifests):
    if not manifests:
        raise ValueError("Empty site-study reservation")
    phases = {phase_of(m) for m in manifests}
    if len(phases) != 1 or next(iter(phases)) not in POLICIES:
        raise ValueError("A site reservation contains one stage only")
    phase = next(iter(phases))
    limits = [job_wall_limit(m) for m in manifests]
    if phase == "setup":
        if len(manifests) != 1 or "parent_manifest_sha256" in manifests[0] or "execution_group" in manifests[0]:
            raise ValueError("Setup requires exactly one unpartitioned worker")
        return limits[0], POLICIES[phase]["stage_ceiling_usd"]
    parents = {m.get("parent_manifest_sha256") for m in manifests}
    parent = next(iter(parents)) if len(parents) == 1 else None
    if (len(manifests) != 3 or {m.get("execution_group") for m in manifests} != set(study.GROUPS)
            or len({digest(m) for m in manifests}) != 3 or not study.valid_sha(parent)):
        raise ValueError("Require all three distinct workers of exactly one frozen parent")
    return min(limits), POLICIES[phase]["stage_ceiling_usd"]
