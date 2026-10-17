"""Narrow, manifest-bound wall-time policy; ordinary jobs retain two hours."""

LEGACY_WALL_LIMIT_SECONDS = 7200
SCREEN_BUDGET = {
    "workers": 3,
    "shared_deadline_seconds": 12000,
    "stage_ceiling_usd": 55,
    "paid_retries": 0,
}


def job_wall_limit(manifest):
    """Return the admitted wall limit, rejecting malformed screen policies.

    A matching value on another assay never grants the exception. Exact integer
    types avoid accepting booleans or coercing strings in frozen policy fields.
    This validates one job's policy, not the complete three-worker reservation.
    """
    if manifest.get("assay") != "satisfaction_screen":
        return LEGACY_WALL_LIMIT_SECONDS
    policy = manifest.get("budget")
    if (
        not isinstance(policy, dict)
        or policy != SCREEN_BUDGET
        or any(type(policy.get(key)) is not int for key in SCREEN_BUDGET)
    ):
        raise ValueError("Satisfaction screen requires the exact frozen SCREEN_BUDGET policy")
    return SCREEN_BUDGET["shared_deadline_seconds"]
