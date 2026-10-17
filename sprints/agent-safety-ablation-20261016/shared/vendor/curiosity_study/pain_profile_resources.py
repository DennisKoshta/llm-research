"""Finite resource policy for the published-method control, not a new allowance."""

from copy import deepcopy

REVISION = "pain-axis-positive-control-v1"
ASSAY = "pain_profile_control"
CAMPAIGN_CEILING_USD = 12.0
SETUP_CEILING_USD = 4.0
BENCHMARK_CEILING_USD = 8.0
TEARDOWN_SECONDS = 120
CPU_CORES = 2
MEMORY_GIB = 96
CPU_RATE = CPU_CORES * 0.0000131 + MEMORY_GIB * 0.00000222
H100_RATE = 0.001097 + CPU_RATE
PRICE_SOURCE = "https://modal.com/pricing"
PRICE_CHECKED_UTC_DATE = "2026-10-15"
PREFIX = "curiosity-study-pain-profile-"

POLICIES = {
    "prefetch": dict(gpu=None, wall_seconds=1800, function_timeout_seconds=1800,
                     app_name=PREFIX + "prefetch-cpu96", rate_usd_per_second=CPU_RATE),
    "setup": dict(gpu="H100", wall_seconds=1800, function_timeout_seconds=1800,
                  app_name=PREFIX + "setup-h10096", rate_usd_per_second=H100_RATE),
    "benchmark": dict(gpu="H100", wall_seconds=3600, function_timeout_seconds=3600,
                      app_name=PREFIX + "benchmark-h10096", rate_usd_per_second=H100_RATE),
}


def policy(phase):
    if phase not in POLICIES:
        raise ValueError("Unknown pain-control phase")
    value = deepcopy(POLICIES[phase])
    value.update(phase=phase, cpu_cores=CPU_CORES, memory_gib=MEMORY_GIB,
                 teardown_seconds=TEARDOWN_SECONDS, max_containers=1,
                 max_admissions=1, retries=0, resume_allowed=False)
    value["maximum_compute_usd"] = (
        value["wall_seconds"] + TEARDOWN_SECONDS
    ) * value["rate_usd_per_second"]
    return value


def app_price(description):
    """Exact-name prices for the central ledger; preserve every legacy app rate."""
    if not description.startswith(PREFIX):
        return None
    for phase, row in POLICIES.items():
        if description == row["app_name"]:
            return dict(gpu=row["gpu"] or "CPU", rate_usd_per_second=row["rate_usd_per_second"],
                        resource_profile=REVISION + ":" + phase)
    raise ValueError("Unpriced pain-control app name; reconcile before further spending")


def campaign_liabilities():
    values = {phase: policy(phase)["maximum_compute_usd"] for phase in POLICIES}
    setup = values["prefetch"] + values["setup"]
    total = sum(values.values())
    if setup > SETUP_CEILING_USD or values["benchmark"] > BENCHMARK_CEILING_USD or total > CAMPAIGN_CEILING_USD:
        raise ValueError("Pain-control resource ceilings exceeded")
    return dict(phases=values, setup=setup, total=total)
