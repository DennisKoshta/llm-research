"""Fixed, controlled C-negative development assignments on all prior screen worlds.

This is an imposed-vector test. Negative semantic steering is not validated
suppression of an endogenous state. No outcomes enter assignment construction.
"""

from copy import deepcopy
import math

from . import satisfaction_screen as prior
from . import voluntary_resolution as voluntary
from .artifacts import digest

REVISION = "curiosity-suppression-screen-v1"
ASSAY = "curiosity_suppression_screen"
SOURCE_REVISION = "voluntary-resolution-v2"
PANELS = prior.PANELS
EXECUTION = dict(backend="torch_batched_v1", batch_size=32)
SCHEDULE = "record_prefill_plus_decode"
PRIMARY = "curiosity_negative"
NEIGHBORS = (
    ("confidence_positive", "confidence_high_vs_low", 1),
    ("completion_positive", "completion_vs_ongoing", 1),
    ("valence_positive", "positive_valence_vs_neutral", 1),
    ("valence_negative", "positive_valence_vs_neutral", -1),
    ("motivation_negative", "generic_motivation_vs_indifference", -1),
    ("novelty_negative", "novelty_interest_vs_indifference", -1),
    ("task_demand_negative", "task_demand_vs_no_demand", -1),
)
RANDOM_SEEDS = tuple(range(2026101501, 2026101506))
RANDOM_ARMS = tuple(f"random_{i:02d}" for i in range(1, 6))
CONTROLS = tuple(name for name, _, _ in NEIGHBORS) + RANDOM_ARMS
ARMS = ("baseline", "zero_sham", PRIMARY, *CONTROLS)
GROUPS = tuple(f"worker_{i}" for i in range(3))
LOCKED_C_NORM = 29.78968343510005
PRIOR_MANIFEST_SHA256 = "f74f1eb522487e4dea61913d41285f47d95a352f8fb88fe629c1ba277e92d528"
LOCKED_DIRECTION_BYTES = "af746df14fedb0023d0ae55eff5ef7704a25d985f31e218e53ac3fa5a43c437e"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_directions(bindings):
    """Structural validation only; preparation/audit reproduce the source bytes."""
    require(set(bindings) == set(ARMS) - {"baseline"}, "Require all14 fixed applied directions")
    expected = {"zero_sham": ("curiosity_vs_indifference", 1), PRIMARY: ("curiosity_vs_indifference", -1)}
    expected.update({arm: (source, orientation) for arm, source, orientation in NEIGHBORS})
    expected.update({arm: ("seeded_gaussian", 1) for arm in RANDOM_ARMS})
    for arm, binding in bindings.items():
        unit = binding["unit"]
        source, orientation = expected[arm]
        require(
            binding["source"] == source
            and binding["orientation"] == orientation
            and binding["layer"] == 15
            and binding["position"] == "record"
            and binding["dose"] == (0.0 if arm == "zero_sham" else LOCKED_C_NORM)
            and binding["reference_raw_norm"] == LOCKED_C_NORM
            and isinstance(unit, list)
            and len(unit) == 5120
            and all(type(x) in (int, float) and math.isfinite(x) for x in unit)
            and math.isclose(math.sqrt(math.fsum(x * x for x in unit)), 1.0, abs_tol=1e-6)
            and binding["unit_sha256"] == digest(unit),
            "Direction, sign, absolute norm, site or width changed",
        )
        if arm in RANDOM_ARMS:
            require(
                binding["seed"] == RANDOM_SEEDS[RANDOM_ARMS.index(arm)]
                and binding["corpus"] is None
                and binding["construction"] == "numpy_default_rng_standard_normal_float64_unit_no_resampling",
                "Random controls must retain every fixed Gaussian draw",
            )
        else:
            require(
                binding["corpus"] == "A"
                and binding["directions_byte_sha256"] == LOCKED_DIRECTION_BYTES
                and type(binding["source_raw_norm"]) in (int, float)
                and math.isfinite(binding["source_raw_norm"])
                and binding["source_raw_norm"] > 0,
                "Require original corpus-A source axes",
            )
    require(
        bindings[PRIMARY]["unit"] == [-x for x in bindings["zero_sham"]["unit"]],
        "Primary must use the exact negative original C unit",
    )
    require(
        bindings["valence_negative"]["unit"] == [-x for x in bindings["valence_positive"]["unit"]],
        "The two valence controls must retain opposite orientations of one fixed source unit",
    )
    digest(bindings)
    return deepcopy(bindings)


def arm_order(base_cohort):
    require(type(base_cohort) is int and base_cohort in range(9), "Unknown base cohort")
    remaining = ARMS[1:]
    offset = (5 * base_cohort) % len(remaining)
    return ("baseline", *remaining[offset:], *remaining[:offset])


def assignment_plan(*, source_manifest, direction_bindings):
    require(digest(source_manifest) == PRIOR_MANIFEST_SHA256, "Require the entire original S-screen parent")
    source = source_manifest["screen_plan"]
    require(
        source_manifest["assay"] == "satisfaction_screen"
        and source["base_assignment_sha256"] == digest(source["base_assignments"])
        and len(source["base_assignments"]) == 288
        and len({r["semantic_world_id"] for r in source["base_assignments"]}) == 24
        and voluntary.REVISION == SOURCE_REVISION,
        "Require all288 unchanged base branches and all24 previously exposed worlds",
    )
    directions = validate_directions(direction_bindings)
    base = deepcopy(source["base_assignments"])
    rows = []
    for cohort_index in range(9):
        for position, arm in enumerate(arm_order(cohort_index)):
            direction = directions.get(arm)
            patch = (
                None
                if direction is None
                else dict(
                    layer=15,
                    position="record",
                    mode="add",
                    direction_key=arm,
                    direction_binding_sha256=digest(direction),
                    dose=direction["dose"],
                )
            )
            for base_row in range(cohort_index * 32, (cohort_index + 1) * 32):
                branch = base[base_row]
                rows.append(
                    dict(
                        **deepcopy(branch),
                        assay=ASSAY,
                        task_revision=REVISION,
                        trial_id=f"{REVISION}:{arm}:{branch['base_trial_id']}",
                        parent_row=len(rows),
                        base_row=base_row,
                        base_cohort=cohort_index,
                        arm_order_in_cohort=position,
                        execution_group=GROUPS[cohort_index % 3],
                        arm=arm,
                        intervention=deepcopy(patch),
                        intervention_schedule=SCHEDULE,
                        max_new_tokens=4096,
                        execution=deepcopy(EXECUTION),
                    )
                )
    require(len(rows) == 4320 and len({r["trial_id"] for r in rows}) == 4320, "Incomplete crossing")
    selection = dict(
        mode="entire_completed_satisfaction_screen_reused",
        source_manifest_sha256=PRIOR_MANIFEST_SHA256,
        worlds=deepcopy(source["selection"]["worlds"]),
        panels=deepcopy(source["selection"]["panels"]),
        historical_exact_exposed_worlds=24,
        fresh_worlds=0,
        selection_uses_outcomes=False,
        interpretation="Previously exposed finite development set; no independent confirmation or new-world generalization",
    )
    return dict(
        schema_version=1,
        assay=ASSAY,
        task_revision=REVISION,
        source_task_revision=SOURCE_REVISION,
        inference_scope="development",
        scientific_admission=False,
        source_screen_manifest_sha256=PRIOR_MANIFEST_SHA256,
        selection=selection,
        selection_sha256=digest(selection),
        direction_bindings=directions,
        direction_bindings_sha256=digest(directions),
        base_assignments=base,
        base_assignment_sha256=digest(base),
        assignments=rows,
        assignment_sha256=digest(rows),
        execution=deepcopy(EXECUTION),
        order_policy=dict(
            baseline_first_per_base_cohort=True,
            nonbaseline_rotation="5 * base_cohort modulo14",
            worker="base_cohort modulo3",
            source_base_order="Original frozen satisfaction-screen base order and row seeds, unchanged",
            arms_by_base_cohort=[list(arm_order(i)) for i in range(9)],
            rule="Each32-row base cohort appears in all15 arms on one worker; all15 arms occur on all3 workers; baseline first and residual order imbalance disclosed",
        ),
        planned_denominators=dict(
            unique_worlds=24,
            panels=2,
            worlds_per_panel=12,
            arms=15,
            base_branches=288,
            branches_per_arm=288,
            total_branches=4320,
            cohorts_per_arm=9,
            total_cohorts=135,
            workers=3,
            branches_per_worker=1440,
            cohorts_per_worker=45,
            total_choices=2160,
            total_diagnostics=2160,
            per_arm=dict(choices=144, diagnostics=144),
            per_panel_per_arm=dict(
                branches=144, choices=72, diagnostics=72, each_diagnostic=12, each_choice_delivery=24
            ),
        ),
        eligibility=dict(
            retain_every_assigned_branch=True,
            no_postoutcome_replacement=True,
            no_exclusion_for_invalid_or_capped_responses=True,
            no_candidate_layer_sign_dose_or_prompt_search=True,
            no_natural_state_or_experience_claim=True,
        ),
    )


def group_assignments(plan, group):
    require(group in GROUPS, "Require one of three balanced workers")
    rows = [deepcopy(r) for r in plan["assignments"] if r["execution_group"] == group]
    require(
        len(rows) == 1440 and all(sum(r["arm"] == a for r in rows) == 96 for a in ARMS),
        "Worker crossing changed",
    )
    return rows


def inference_request(row):
    require(row["assay"] == ASSAY and row["task_revision"] == REVISION, "Wrong assay")
    require(digest(row["source_assignment"]) == row["source_assignment_sha256"], "Source binding changed")
    request = voluntary.inference_request(row["source_assignment"])
    require(digest(request) == row["inference_request_sha256"], "Source request changed")
    return request
