"""Prospective, bounded S+ layer-specific development; never scientific admission.

The completed A15 screen remains unchanged. These pure constructors inspect no
responses and cannot certify candidate provenance, execution, or held-out status.
"""

from copy import deepcopy
import math

import numpy as np

from . import curiosity_suppression_screen as controls
from . import reserved_object_materials
from . import voluntary_resolution as voluntary
from .artifacts import digest

REVISION = "satisfaction-causal-sites-v1"
ASSAY = "satisfaction_causal_sites"
SOURCE_REVISION = "voluntary-resolution-v2"
SITES = (31, 47)
PRIMARY = "satisfaction_positive"
CALIBRATION_ARMS = (
    "baseline", "zero_sham", PRIMARY, "confidence_positive", "completion_positive",
    "valence_positive", "random_01",
)
VALIDATION_ARMS = ("baseline", "zero_sham", PRIMARY, *controls.CONTROLS)
CALIBRATION_CONTROLS = CALIBRATION_ARMS[3:]
GROUPS = controls.GROUPS
EXECUTION = dict(backend="torch_batched_v1", batch_size=32)
SCHEDULE = "record_prefill_plus_decode"
PRIOR_MANIFEST_SHA256 = controls.PRIOR_MANIFEST_SHA256
CALIBRATION_BASE_SHA256 = "dec2ff4471e6e2f7825d80ed87194c7af2008b75663b81469ebabee746adfa0a"
CALIBRATION_SELECTION_SHA256 = "af2e385586b8ab6e258b72d91cd19eb741f29a92630d71ad6ec2e96a1b34d0c5"
LOCKED_DIRECTION_BYTES = controls.LOCKED_DIRECTION_BYTES
ORIGINAL_S_DOSE = 29.89263847846578
DOSE_FORMULA = "D15 * median_record_norm[layer] / median_record_norm[15]"
RESERVED_SEED = 2026101541
RESERVED_HALF_SEEDS = (2026101541, 2026101543)
BOOTSTRAP_SEED = 2026101542
BOOTSTRAP_DRAWS = 10000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")


def validate_reference(reference):
    """Validate the predetermined arithmetic; the auditor must verify capture bytes."""
    medians, doses = reference["median_record_norm"], reference["doses"]
    require(
        reference["formula"] == DOSE_FORMULA
        and type(reference["original_dose"]) in (int, float)
        and reference["original_dose"] == ORIGINAL_S_DOSE
        and type(reference["base_branches"]) is int and reference["base_branches"] == 288
        and reference["source_screen_manifest_sha256"] == PRIOR_MANIFEST_SHA256
        and valid_sha(reference["capture_sha256"])
        and set(medians) == {"15", "31", "47"}
        and set(doses) == {"31", "47"}
        and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in medians.values())
        and all(
            type(doses[str(site)]) in (int, float)
            and math.isfinite(doses[str(site)]) and doses[str(site)] > 0
            and doses[str(site)] == ORIGINAL_S_DOSE * medians[str(site)] / medians["15"]
            for site in SITES
        ),
        "Require all288 pretreatment prefixes and the unchanged one-dose formula",
    )
    return deepcopy(reference)


def validate_directions(bindings, *, site, arms, dose):
    """Structural and random-draw validation; A source bytes are verified separately."""
    require(type(dose) in (int, float) and math.isfinite(dose) and dose > 0,
            "Active intervention requires a positive finite dose")
    require(site in SITES and tuple(arms) in (CALIBRATION_ARMS, VALIDATION_ARMS), "Unknown site or arm set")
    require(set(bindings) == set(arms) - {"baseline"}, "Every prescribed direction is required")
    expected = {
        "zero_sham": ("satisfaction_vs_neutral", 1), PRIMARY: ("satisfaction_vs_neutral", 1),
        **{name: (source, sign) for name, source, sign in controls.NEIGHBORS},
        **{name: ("seeded_gaussian", 1) for name in controls.RANDOM_ARMS},
    }
    for arm, binding in bindings.items():
        unit, (source, orientation) = binding["unit"], expected[arm]
        require(
            binding["source"] == source and binding["orientation"] == orientation
            and binding["layer"] == site and binding["position"] == "record"
            and binding["dose"] == (0.0 if arm == "zero_sham" else dose)
            and isinstance(unit, list) and len(unit) == 5120
            and all(type(x) in (int, float) and math.isfinite(x) for x in unit)
            and math.isclose(math.sqrt(math.fsum(x * x for x in unit)), 1.0, abs_tol=1e-6)
            and binding["unit_sha256"] == digest(unit),
            "Direction, sign, dose, site, width, or unit binding changed",
        )
        if arm in controls.RANDOM_ARMS:
            seed = controls.RANDOM_SEEDS[controls.RANDOM_ARMS.index(arm)]
            raw = np.random.default_rng(seed).standard_normal(5120)
            drawn = (raw / np.linalg.norm(raw)).tolist()
            require(
                binding["seed"] == seed and binding["corpus"] is None and unit == drawn
                and binding["construction"]
                == "numpy_default_rng_standard_normal_float64_unit_no_resampling",
                "Keep the original unmodified random draw without reorientation",
            )
        else:
            require(
                binding["corpus"] == "A" and binding["directions_byte_sha256"] == LOCKED_DIRECTION_BYTES
                and type(binding["source_raw_norm"]) in (int, float)
                and math.isfinite(binding["source_raw_norm"]) and binding["source_raw_norm"] > 0,
                "Require the original A layer-specific source unit",
            )
    require(bindings["zero_sham"]["unit"] == bindings[PRIMARY]["unit"], "Sham must use the S+ unit")
    if "valence_negative" in bindings:
        require(
            bindings["valence_negative"]["unit"] == [-x for x in bindings["valence_positive"]["unit"]],
            "Keep both orientations of the same original valence unit",
        )
    return deepcopy(bindings)


def condition_order(base_cohort, *, phase, selected_site=None):
    require(phase in ("calibration", "validation"), "Unknown phase")
    require(type(base_cohort) is int and base_cohort in range(9 if phase == "calibration" else 6),
            "Unknown source cohort")
    sites = SITES if phase == "calibration" else (selected_site,)
    require(all(site in SITES for site in sites), "Unknown selected site")
    if phase == "calibration" and base_cohort % 2:
        sites = tuple(reversed(sites))
    arms = CALIBRATION_ARMS if phase == "calibration" else VALIDATION_ARMS
    ordered = []
    for site in sites:
        rest = arms[1:]
        offset = (5 * base_cohort + 3 * SITES.index(site)) % len(rest)
        ordered.extend((site, arm) for arm in ("baseline", *rest[offset:], *rest[:offset]))
    return ordered


def calibration_inputs(source_manifest):
    require(digest(source_manifest) == PRIOR_MANIFEST_SHA256, "Require the whole completed original S parent")
    source = source_manifest["screen_plan"]
    base = source["base_assignments"]
    require(
        voluntary.REVISION == SOURCE_REVISION and source_manifest["assay"] == "satisfaction_screen"
        and len(base) == 288 and digest(base) == source["base_assignment_sha256"]
        and len({r["semantic_world_id"] for r in base}) == 24,
        "Keep all24 known worlds and all288 original branches",
    )
    return deepcopy(base), deepcopy(source["selection"])


def assignment_plan(*, phase, base_assignments, selection, direction_bindings, dose_reference,
                    selected_site=None):
    require(phase in ("calibration", "validation"), "Unknown phase")
    require(selected_site is None if phase == "calibration" else selected_site in SITES,
            "Calibration uses both sites; validation uses one previously selected site")
    reference = validate_reference(dose_reference)
    base = deepcopy(base_assignments)
    nbase, nworlds = (288, 24) if phase == "calibration" else (192, 16)
    sites = SITES if phase == "calibration" else (selected_site,)
    arms = CALIBRATION_ARMS if phase == "calibration" else VALIDATION_ARMS
    require(
        len(base) == nbase and len({r["base_trial_id"] for r in base}) == nbase
        and len({r["semantic_world_id"] for r in base}) == nworlds
        and len({r["sampling_seed"] for r in base}) == nbase
        and set(direction_bindings) == {str(site) for site in sites},
        "Complete source inventory and prescribed site bindings required",
    )
    if phase == "calibration":
        require(digest(base) == CALIBRATION_BASE_SHA256,
                "Calibration retains the exact original ordered288 source branches")
        require(digest(selection) == CALIBRATION_SELECTION_SHA256,
                "Calibration retains the exact original panel selection")
    else:
        reserved_object_materials.validate_reserved_inputs(selection, base)
    expected_variants = {("choice", delivery, menu) for delivery in voluntary.DELIVERIES for menu in (0, 1)}
    expected_variants.update(
        (kind, "decisive_evidence" if kind in ("identify_resolved", "repeated_fact")
         else "redundant_evidence", 0) for kind in voluntary.DIAGNOSTICS
    )
    for world in {r["semantic_world_id"] for r in base}:
        members = [r for r in base if r["semantic_world_id"] == world]
        require(len(members) == 12
                and {(r["kind"], r["delivery"], r["menu_layout"]) for r in members} == expected_variants,
                "Every world retains all twelve original variants")
    directions = {
        str(site): validate_directions(direction_bindings[str(site)], site=site, arms=arms,
                                       dose=reference["doses"][str(site)])
        for site in sites
    }
    rows, orders = [], []
    for cohort in range(nbase // 32):
        order = condition_order(cohort, phase=phase, selected_site=selected_site)
        orders.append([list(value) for value in order])
        for position, (site, arm) in enumerate(order):
            direction = directions[str(site)].get(arm)
            patch = None if direction is None else dict(
                layer=site, position="record", mode="add", direction_key=arm,
                direction_binding_sha256=digest(direction), dose=direction["dose"],
            )
            for base_row in range(cohort * 32, (cohort + 1) * 32):
                branch = base[base_row]
                require(digest(branch["source_assignment"]) == branch["source_assignment_sha256"]
                        and digest(voluntary.inference_request(branch["source_assignment"]))
                        == branch["inference_request_sha256"], "Original request binding changed")
                require(
                    branch["base_trial_id"] == branch["source_assignment"]["trial_id"]
                    and all(branch[key] == branch["source_assignment"][key] for key in
                            ("kind", "delivery", "menu_layout", "sampling_seed", "family", "semantic_world_id", "world_slot"))
                    and branch["panel_seed"] == branch["source_assignment"]["seed"]
                    and branch["world_id"] == f"{branch['panel_id']}:{branch['semantic_world_id']}",
                    "Outer identity differs from the exact source assignment",
                )
                rows.append(dict(
                    **deepcopy(branch), assay=ASSAY, task_revision=REVISION, phase=phase, site=site,
                    trial_id=f"{REVISION}:{phase}:layer{site}:{arm}:{branch['base_trial_id']}",
                    parent_row=len(rows), base_row=base_row, base_cohort=cohort,
                    condition_order_in_cohort=position, execution_group=GROUPS[cohort % 3], arm=arm,
                    intervention=deepcopy(patch), intervention_schedule=SCHEDULE,
                    max_new_tokens=4096, execution=deepcopy(EXECUTION),
                ))
    expected = 4032 if phase == "calibration" else 2880
    require(len(rows) == len({r["trial_id"] for r in rows}) == expected, "Incomplete site/arm crossing")
    return dict(
        schema_version=1, assay=ASSAY, task_revision=REVISION, source_task_revision=SOURCE_REVISION,
        phase=phase, sites=list(sites), arms=list(arms), scientific_admission=False,
        source_screen_manifest_sha256=PRIOR_MANIFEST_SHA256, selection=deepcopy(selection),
        selection_sha256=digest(selection), base_assignments=base, base_assignment_sha256=digest(base),
        direction_bindings=directions, direction_bindings_sha256=digest(directions),
        dose_reference=reference, dose_reference_sha256=digest(reference),
        assignments=rows, assignment_sha256=digest(rows), execution=deepcopy(EXECUTION),
        order_policy=dict(
            conditions_by_base_cohort=orders, worker="base_cohort modulo3",
            baseline_first_per_site=True, alternate_site_order_by_source_cohort=True,
            nonbaseline_rotation="(5 * base_cohort + 3 * canonical_site_index) modulo nonbaseline_arm_count",
            limitation="Partial counterbalancing; baseline first, no balanced Latin-square claim",
        ),
        planned_denominators=dict(
            worlds=nworlds, base_branches=nbase, sites=len(sites), arms_per_site=len(arms),
            generations=expected, choices=expected // 2, diagnostics=expected // 2,
            generation_cohorts=expected // 32, workers=3, branches_per_worker=expected // 3,
        ),
    )


def group_assignments(plan, group):
    require(group in GROUPS, "Unknown worker")
    rows = [deepcopy(r) for r in plan["assignments"] if r["execution_group"] == group]
    per_condition = len(plan["base_assignments"]) // 3
    require(
        len(rows) == plan["planned_denominators"]["branches_per_worker"]
        and all(sum(r["site"] == site and r["arm"] == arm for r in rows) == per_condition
                for site in plan["sites"] for arm in plan["arms"]),
        "Every site and arm must appear on every worker with intact original cohorts",
    )
    return rows


def inference_request(row):
    require(row["assay"] == ASSAY and row["task_revision"] == REVISION, "Wrong assay")
    require(digest(row["source_assignment"]) == row["source_assignment_sha256"], "Source binding changed")
    request = voluntary.inference_request(row["source_assignment"])
    require(digest(request) == row["inference_request_sha256"], "Source request changed")
    return request
