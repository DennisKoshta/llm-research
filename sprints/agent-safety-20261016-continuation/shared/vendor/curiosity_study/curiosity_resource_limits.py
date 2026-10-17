"""Separate bounded policy for the fixed curiosity-suppression development study."""

from .artifacts import digest
from .resource_limits import job_wall_limit as legacy_job_wall_limit

ASSAY = "curiosity_suppression_screen"
CURIOSITY_SCREEN_BUDGET = {
    "workers": 3,
    "shared_deadline_seconds": 12000,
    "stage_ceiling_usd": 55,
    "paid_retries": 0,
}
GROUPS = ("worker_0", "worker_1", "worker_2")


def job_wall_limit(manifest):
    """Validate this named exception, delegating every old assay unchanged."""
    if manifest.get("assay") != ASSAY:
        return legacy_job_wall_limit(manifest)
    policy = manifest.get("budget")
    if (
        not isinstance(policy, dict)
        or policy != CURIOSITY_SCREEN_BUDGET
        or any(type(policy.get(key)) is not int for key in CURIOSITY_SCREEN_BUDGET)
    ):
        raise ValueError("Curiosity suppression requires the exact frozen CURIOSITY_SCREEN_BUDGET policy")
    return CURIOSITY_SCREEN_BUDGET["shared_deadline_seconds"]


def reservation_limits(manifests):
    """Require all three distinct worker slots of one parent; prohibit mixtures."""
    if len(manifests) != 3 or any(m.get("assay") != ASSAY for m in manifests):
        raise ValueError("Curiosity reservation requires three workers of the same assay")
    limits = [job_wall_limit(m) for m in manifests]
    parents = {m.get("parent_manifest_sha256") for m in manifests}
    parent = next(iter(parents)) if len(parents) == 1 else None
    if (
        {m.get("execution_group") for m in manifests} != set(GROUPS)
        or len({digest(m) for m in manifests}) != 3
        or not isinstance(parent, str)
        or len(parent) != 64
        or any(c not in "0123456789abcdef" for c in parent)
    ):
        raise ValueError("Curiosity reservation requires three distinct groups from one bound parent")
    return min(limits), CURIOSITY_SCREEN_BUDGET["stage_ceiling_usd"]
