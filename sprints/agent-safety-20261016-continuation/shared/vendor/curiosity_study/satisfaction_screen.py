"""Pure prospective object-only assignments; no selection from model outcomes or execution.

The caller supplies a separately audited historical inventory and locked-axis
provenance. This module checks their structure and binds their values; it cannot
certify their provenance or grant scientific admission. No study data are saved.
"""

from copy import deepcopy
import itertools
import math
import random

from . import information_state as state
from . import voluntary_resolution as voluntary
from .artifacts import digest

REVISION = "satisfaction-screen-v1"
SOURCE_REVISION = "voluntary-resolution-v2"
PANELS = (("panel_1", 2026101436), ("panel_2", 2026101437))
WORLDS_PER_PANEL = 12
SCHEDULE = "record_prefill_plus_decode"
ARMS = (
    ("baseline", None),
    ("zero_sham", 0.0),
    ("satisfaction_plus_0_5", 0.5),
    ("satisfaction_plus_1_0", 1.0),
)
EXECUTION = dict(backend="torch_batched_v1", batch_size=32)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")


def _historical_ids(values):
    _require(
        isinstance(values, (list, tuple, set, frozenset)) and all(_sha(value) for value in values),
        "Historical exclusions must be a flat collection of canonical SHA256 world IDs",
    )
    return sorted(set(values))


def validate_direction(binding):
    """Check a JSON-ready fixed-A satisfaction direction; the preparer must verify its files."""
    candidate = binding["candidate_source"]
    unit, norm = binding["unit"], binding["norm"]
    _require(
        binding["source"] == "satisfaction_vs_neutral"
        and binding["corpus"] == "A"
        and binding["layer"] == 15
        and binding["position"] == "record"
        and isinstance(unit, list)
        and bool(unit)
        and all(type(value) in (int, float) and math.isfinite(value) for value in unit)
        and math.isclose(math.sqrt(math.fsum(value * value for value in unit)), 1, abs_tol=1e-6)
        and type(norm) in (int, float)
        and math.isfinite(norm)
        and norm > 0
        and binding["unit_sha256"] == digest(unit)
        and _sha(binding["directions_byte_sha256"])
        and candidate["directions_byte_sha256"] == binding["directions_byte_sha256"]
        and candidate["selected_sites"]["satisfaction_vs_neutral"] == dict(layer=15, position="record")
        and all(
            _sha(candidate[key])
            for key in (
                "capture_manifest_sha256",
                "capture_audit_sha256",
                "analysis_sha256",
                "analysis_byte_sha256",
                "directions_byte_sha256",
                "independent_check_sha256",
            )
        ),
        "Require the original positive corpus-A15/record satisfaction unit, raw norm and provenance",
    )
    # Reject non-JSON extra metadata rather than silently dropping provenance.
    digest(binding)
    return deepcopy(binding)


def select_object_panels(*, historical_exact_world_ids, max_candidates=10000):
    """Select 24 unique objects using inputs only, distinguishing prospective duplicates."""
    _require(voluntary.REVISION == SOURCE_REVISION, "Require unchanged voluntary v2 construction")
    _require(type(max_candidates) is int and max_candidates > 0, "Require a positive candidate limit")
    historical = _historical_ids(historical_exact_world_ids)
    excluded, previously_selected = set(historical), {}
    worlds, ledger = [], []
    for panel_id, seed in PANELS:
        selected = {}
        for index in range(max_candidates):
            slot = len(selected)
            known = voluntary._known_attribute(slot)
            source = dict(
                family="objects", seed=seed, index=index, vocabulary="heldout", known=[known], query=1 - known
            )
            semantic = digest(voluntary.canonical_world("objects", state.task(source)))
            first = None
            if semantic in excluded:
                reason = "historical_exact_exposure"
            elif semantic in selected:
                reason, first = (
                    "within_panel_duplicate",
                    dict(panel_id=panel_id, world_slot=selected[semantic]),
                )
            elif semantic in previously_selected:
                reason, first = "prior_panel_duplicate", previously_selected[semantic]
            else:
                reason = "selected"
                first = dict(panel_id=panel_id, world_slot=slot)
            accepted = reason == "selected"
            ledger.append(
                dict(
                    panel_id=panel_id,
                    seed=seed,
                    candidate_index=index,
                    semantic_world_id=semantic,
                    accepted=accepted,
                    reason=reason,
                    first_selected_location=first,
                )
            )
            if accepted:
                selected[semantic] = slot
                worlds.append(
                    dict(
                        **source,
                        panel_id=panel_id,
                        world_slot=slot,
                        semantic_world_id=semantic,
                        world_id=f"{panel_id}:{semantic}",
                    )
                )
            if len(selected) == WORLDS_PER_PANEL:
                break
        else:
            raise ValueError(
                f"Candidate limit exhausted before twelve eligible objects in {panel_id}; exclusions cannot be relaxed"
            )
        previously_selected.update(
            {key: dict(panel_id=panel_id, world_slot=slot) for key, slot in selected.items()}
        )
    return dict(
        panels=[dict(panel_id=name, seed=seed, assigned_worlds=12) for name, seed in PANELS],
        worlds=worlds,
        candidate_exclusion_ledger=ledger,
        historical_exact_exclusions=historical,
        historical_exact_exclusions_sha256=digest(historical),
        selected_semantic_world_ids=sorted(previously_selected),
        semantic_rule="voluntary_finite_world_v1: named candidate table plus realized truth; ignores display order and question",
        selection_rule="Fixed seeds, ascending candidate indices, twelve accepted worlds per panel; no outcome-based eligibility or replacement",
        limitations=[
            "Historical inventory coverage must be independently verified before preparation.",
            "Cross-panel exclusions are prospective duplicates, never historical exposure.",
            "Exact-world exclusions do not establish independence from projected-table or abstract task exposure.",
        ],
    )


def _source_assignment(world, kind, delivery, menu):
    seed, slot = world["seed"], world["world_slot"]
    key = f"voluntary-{seed}-objects-w{slot:02d}-{kind}-{delivery}-menu{menu}"
    return dict(
        **{key: value for key, value in world.items() if key not in ("panel_id", "world_id")},
        assay="thinking_information",
        task_revision=SOURCE_REVISION,
        kind=kind,
        delivery=delivery,
        menu_layout=menu,
        template="table",
        trial_id=key,
        sampling_seed=int(digest(f"voluntary-sample:{key}")[:8], 16),
        max_new_tokens=4096,
        execution=EXECUTION.copy(),
    )


def assignment_plan(*, historical_exact_world_ids, direction_binding, max_candidates=10000):
    direction = validate_direction(direction_binding)
    selection = select_object_panels(
        historical_exact_world_ids=historical_exact_world_ids, max_candidates=max_candidates
    )
    base = []
    for world in selection["worlds"]:
        variants = [
            ("choice", delivery, menu) for delivery, menu in itertools.product(voluntary.DELIVERIES, (0, 1))
        ]
        variants += [
            (
                kind,
                "decisive_evidence"
                if kind in ("identify_resolved", "repeated_fact")
                else "redundant_evidence",
                0,
            )
            for kind in voluntary.DIAGNOSTICS
        ]
        for kind, delivery, menu in variants:
            source = _source_assignment(world, kind, delivery, menu)
            base.append(
                dict(
                    base_trial_id=source["trial_id"],
                    panel_id=world["panel_id"],
                    panel_seed=world["seed"],
                    world_id=world["world_id"],
                    world_slot=world["world_slot"],
                    semantic_world_id=world["semantic_world_id"],
                    family="objects",
                    kind=kind,
                    delivery=delivery,
                    menu_layout=menu,
                    sampling_seed=source["sampling_seed"],
                    source_assignment=source,
                    source_assignment_sha256=digest(source),
                    inference_request_sha256=digest(voluntary.inference_request(source)),
                )
            )
    ordering_seed = "satisfaction-screen-order:" + ":".join(str(seed) for _, seed in PANELS)
    random.Random(ordering_seed).shuffle(base)
    _require(
        len(base) == 288 and len({row["sampling_seed"] for row in base}) == 288,
        "Require every base branch and distinct assigned row seeds",
    )
    rows = []
    for arm, multiplier in ARMS:
        patch = (
            None
            if multiplier is None
            else dict(
                layer=15,
                position="record",
                mode="add",
                source="satisfaction_vs_neutral",
                corpus="A",
                direction_binding_sha256=digest(direction),
                dose_multiplier=multiplier,
                dose=multiplier * direction["norm"],
            )
        )
        for index, branch in enumerate(base):
            rows.append(
                dict(
                    **deepcopy(branch),
                    assay="satisfaction_screen",
                    task_revision=REVISION,
                    trial_id=f"{REVISION}:{arm}:{branch['base_trial_id']}",
                    base_row=index,
                    arm=arm,
                    intervention_schedule=SCHEDULE,
                    intervention=deepcopy(patch),
                    max_new_tokens=4096,
                    execution=EXECUTION.copy(),
                )
            )
    return dict(
        schema_version=1,
        assay="satisfaction_screen",
        task_revision=REVISION,
        source_task_revision=SOURCE_REVISION,
        inference_scope="development",
        scientific_admission=False,
        selection=selection,
        selection_sha256=digest(selection),
        direction_binding=direction,
        direction_binding_sha256=digest(direction),
        base_assignments=base,
        base_assignment_sha256=digest(base),
        assignments=rows,
        assignment_sha256=digest(rows),
        execution=EXECUTION.copy(),
        order_policy=dict(
            seed=ordering_seed,
            rule="One fixed shuffled base order, repeated unchanged across all four condition-major arms",
        ),
        planned_denominators=dict(
            unique_worlds=24,
            panels=2,
            worlds_per_panel=12,
            arms=4,
            base_branches=288,
            branches_per_arm=288,
            total_branches=1152,
            cohorts_per_arm=9,
            total_cohorts=36,
            per_arm=dict(choices=144, diagnostics=144),
            per_panel_per_arm=dict(
                branches=144, choices=72, diagnostics=72, each_diagnostic=12, each_choice_delivery=24
            ),
        ),
        eligibility=dict(
            assignment_only=True,
            retain_every_assigned_branch=True,
            no_postoutcome_replacement=True,
            no_exclusion_for_invalid_or_capped_responses=True,
            history_provenance_requires_independent_admission=True,
            direction_provenance_requires_independent_admission=True,
            scientific_criteria="Require a separate prospective scientific plan; this assignment module grants no admission",
        ),
    )


def _source(row):
    source = row["source_assignment"]
    _require(
        row["task_revision"] == REVISION
        and source["task_revision"] == SOURCE_REVISION
        and source["family"] == row["family"] == "objects"
        and digest(source) == row["source_assignment_sha256"]
        and source["sampling_seed"] == row["sampling_seed"]
        and source["semantic_world_id"] == row["semantic_world_id"]
        and source["world_slot"] == row["world_slot"]
        and (row["panel_id"], row["panel_seed"]) in PANELS
        and source["seed"] == row["panel_seed"]
        and all(source[key] == row[key] for key in ("kind", "delivery", "menu_layout"))
        and row["world_id"] == f"{row['panel_id']}:{row['semantic_world_id']}",
        "Screen/source assignment or panel binding changed",
    )
    return source


def prompt(row):
    return voluntary.prompt(_source(row))


def task(row):
    return voluntary.task(_source(row))


def inference_request(row):
    """Original v2 generation request; a future executor must apply the declared arm separately."""
    request = voluntary.inference_request(_source(row))
    _require(
        digest(request) == row["inference_request_sha256"],
        "Original v2 prompt or assigned sampling regime changed",
    )
    return request
