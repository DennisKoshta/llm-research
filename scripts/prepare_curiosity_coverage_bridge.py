"""Input-only draft materials for the fixed C-negative coverage comparison.

This module never freezes a run, admits compute, executes a model, regenerates a
world, samples a direction, or reads response records. Existing source packages
are inspected as inert bytes. The original task is not reimplemented here.
"""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REVISION = "curiosity-coverage-bridge-v1"
PLAN_PATH = "reports/curiosity-coverage-bridge-v1-plan.md"
MODEL = dict(model_id="Qwen/Qwen3-32B", revision="9216db5781bf21249d130ec9da846c4624c16137")
C_NORM = 29.78968343510005
BASE_SHA = "dec2ff4471e6e2f7825d80ed87194c7af2008b75663b81469ebabee746adfa0a"
BASE_REQUESTS_SHA = "afab3811905c6c5906a72b5c53a62fd0c730eff9582de90d5c6a5491cbacf681"
DIRECTION_BYTES_SHA = "af746df14fedb0023d0ae55eff5ef7704a25d985f31e218e53ac3fa5a43c437e"
DIRECTION_BINDINGS_SHA = "3c9f7772efd04d4f8481fccdca8697beebf1522a21c5cbae9702fdf95a7929d7"
DIRECTIONS_PATH = "analyses/curiosity-satisfaction-candidates-v1/directions.npz"
DIRECTION_INVENTORY_PATH = "reports/curiosity-suppression-screen-v1-direction-inventory.json"
SOURCES = {
    "satisfaction": dict(
        run="qwen32-satisfaction-screen-v1",
        manifest_sha256="f74f1eb522487e4dea61913d41285f47d95a352f8fb88fe629c1ba277e92d528",
        source_sha256="c20ca06cb3a650ae413c78e4651d94dc5a9ddf1b79ca43735f3d7e75fd07c632",
        archive_sha256="b6a2a61a51c7fd23496a96221617d03919b8c9fbce22476f26b2f0470669207c",
    ),
    "curiosity": dict(
        run="qwen32-curiosity-suppression-screen-v1-admission2",
        manifest_sha256="535f3c1237a8b876fa1059d3f9949f178a55b5627e2be412008ef08cbc90b02f",
        source_sha256="1fd4079ed60fbd0169f702ab9c8f99a75d170450dfbd364bc4cb877e27c18f9a",
        archive_sha256="4b82bdc563a3b257fd1b5e0302f5889cb6e14e30da9fcb950cc7f3039eb62ca4",
    ),
}
NEIGHBORS = (
    ("confidence_positive", "confidence_high_vs_low", 1),
    ("completion_positive", "completion_vs_ongoing", 1),
    ("valence_positive", "positive_valence_vs_neutral", 1),
    ("valence_negative", "positive_valence_vs_neutral", -1),
    ("motivation_negative", "generic_motivation_vs_indifference", -1),
    ("novelty_negative", "novelty_interest_vs_indifference", -1),
    ("task_demand_negative", "task_demand_vs_no_demand", -1),
)
RANDOM_ARMS = tuple(f"random_{i:02d}" for i in range(1, 6))
CONTROLS = tuple(row[0] for row in NEIGHBORS) + RANDOM_ARMS
PAIRS = ("zero_sham", "curiosity_negative", *CONTROLS)
SCHEDULES = dict(record="record_prefill_plus_decode", full="all_attended_prefill_plus_decode")
POSITIONS = dict(record="record", full="all_attended")
ARMS = ("baseline", *(f"{schedule}__{direction}" for direction in PAIRS for schedule in SCHEDULES))
DIAGNOSTICS = (
    "identify_resolved",
    "new_fact",
    "repeated_fact",
    "pattern_independence",
    "instrument_useful_action",
    "action_consequence",
)
DELIVERIES = ("decisive_evidence", "redundant_evidence", "independent_novelty")
RATE = 0.00139376
CRITERIA = dict(
    diagnostic_cell_assigned=12,
    diagnostic_cell_minimum_strict_valid=11,
    diagnostic_cell_minimum_strict_correct=11,
    invalid_diagnostics_count_incorrect=True,
    diagnostic_arm_assigned=144,
    diagnostic_arm_minimum_strict_valid=137,
    original_pristine_diagnostic_cell_minimum_strict_valid=12,
    retain_pristine_format_flags=True,
    choice_cell_assigned=24,
    choice_cell_minimum_strict_valid=23,
    maximum_caps=0,
    maximum_execution_errors=0,
    maximum_missing=0,
    primary_full_effect_per_panel=0.25,
    primary_interaction_per_panel=0.25,
    each_control_full_margin_per_panel=0.125,
    each_control_interaction_margin_per_panel=0.125,
    primary_pooled_menus_strictly_positive=True,
    control_pooled_menus_nonnegative=True,
    require_all12_controls=True,
    coefficient_aware_invalidity_bounds=True,
    retain_every_effect_flag_under_readiness_failure=True,
    amendment="2026-10-15 prospective prefreeze format-readiness amendment; old S/C/48-probe failures preserved",
    belief_equivalence_claim=False,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def byte_sha(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def resources():
    return dict(
        engineering=dict(
            gpu="H200",
            cpu=2,
            memory_gib=48,
            workers=1,
            wall_seconds=3400,
            teardown_seconds=120,
            ceiling_usd=5,
            maximum_liability_usd=3520 * RATE,
            paid_retries=0,
        ),
        scientific=dict(
            gpu="H200",
            cpu=2,
            memory_gib=48,
            workers=3,
            wall_seconds=19000,
            teardown_seconds=120,
            ceiling_usd=80,
            maximum_liability_usd=3 * 19120 * RATE,
            paid_retries=0,
        ),
        rate_usd_per_second=RATE,
        forecast_required=True,
        forecast_execution_margin=0.25,
        automatic_resume=False,
        materials_are_not_admission=True,
    )


def verify_package(root, role):
    """Verify inert archived inputs, without importing archived/current task code."""
    pin = SOURCES[role]
    run = Path(root) / "runs" / pin["run"]
    manifest, freeze = read(run / "manifest.json"), read(run / "freeze.json")
    require(
        digest(manifest) == freeze["manifest_sha256"] == pin["manifest_sha256"]
        and manifest["source_sha256"] == freeze["source_sha256"] == pin["source_sha256"]
        and sha(run / "source.tar.gz") == freeze["archive_sha256"] == pin["archive_sha256"],
        "Pinned source package differs: " + role,
    )
    require(manifest["model"] == MODEL, "Inherited model/checkpoint differs")
    names, source, documents, retained = set(), {}, {}, {}
    retain = {
        DIRECTIONS_PATH,
        DIRECTION_INVENTORY_PATH,
        "runs/" + SOURCES["satisfaction"]["run"] + "/manifest.json",
    }
    with tarfile.open(run / "source.tar.gz") as archive:
        for member in archive:
            relative = Path(member.name)
            require(
                member.isfile()
                and not relative.is_absolute()
                and ".." not in relative.parts
                and str(relative) == member.name
                and member.name not in names,
                "Unsafe/duplicate archived input",
            )
            names.add(member.name)
            if member.name.startswith("src/curiosity_study/") and member.name.endswith(".py"):
                payload = archive.extractfile(member).read()
                source[member.name.removeprefix("src/curiosity_study/")] = payload.decode("utf-8")
            elif member.name in freeze["documents"] or member.name in retain:
                payload = archive.extractfile(member).read()
            else:
                continue
            if member.name in freeze["documents"]:
                require(byte_sha(payload) == freeze["documents"][member.name], "Archived document differs")
                documents[member.name] = byte_sha(payload)
            if member.name in retain:
                retained[member.name] = payload
    require(
        source and digest(source) == pin["source_sha256"] and documents == freeze["documents"],
        "Archive source/document inventory differs",
    )
    provenance = dict(
        role=role,
        run=pin["run"],
        manifest_sha256=digest(manifest),
        manifest_byte_sha256=sha(run / "manifest.json"),
        freeze_sha256=digest(freeze),
        freeze_byte_sha256=sha(run / "freeze.json"),
        archive_byte_sha256=pin["archive_sha256"],
        source_sha256=digest(source),
        verified_document_byte_sha256=documents,
        response_records_read=False,
        source_code_executed=False,
    )
    return manifest, retained, provenance


def validate_directions(curiosity, archived):
    bindings = curiosity["screen_plan"]["direction_bindings"]
    require(
        set(bindings) == set(PAIRS)
        and digest(bindings)
        == curiosity["screen_plan"]["direction_bindings_sha256"]
        == DIRECTION_BINDINGS_SHA,
        "All original signed directions required unchanged",
    )
    require(
        byte_sha(archived[DIRECTIONS_PATH])
        == DIRECTION_BYTES_SHA
        == curiosity["candidate_source"]["directions_byte_sha256"],
        "Original candidate array bytes differ",
    )
    inventory = json.loads(archived[DIRECTION_INVENTORY_PATH])
    require(
        digest(inventory) == curiosity["direction_inventory_sha256"]
        and inventory["direction_bindings_sha256"] == digest(bindings)
        and inventory["arm_order"] == list(PAIRS)
        and inventory["resampled"] is False
        and inventory["orientation_selected_by_outcome"] is False,
        "Original direction inventory differs",
    )
    by_arm = {r["arm"]: r for r in inventory["directions"]}
    require(
        set(by_arm) == set(PAIRS) and len(by_arm) == len(inventory["directions"]),
        "Direction inventory incomplete",
    )
    specs = (
        ("zero_sham", "curiosity_vs_indifference", 1),
        ("curiosity_negative", "curiosity_vs_indifference", -1),
        *NEIGHBORS,
    )
    with np.load(BytesIO(archived[DIRECTIONS_PATH]), allow_pickle=False) as arrays:
        require(
            list(arrays["layers"]) == [15, 31, 47, 57] and list(arrays["positions"]) == ["record", "final"],
            "Original candidate sites/layout differ",
        )
        for arm, source, orientation in specs:
            binding = bindings[arm]
            index = list(arrays["sources"]).index(source)
            original = arrays["unit"][index, 0, 0].astype(np.float64)
            require(
                binding["source"] == source
                and binding["orientation"] == orientation
                and binding["corpus"] == "A"
                and binding["directions_byte_sha256"] == DIRECTION_BYTES_SHA
                and binding["source_unit_sha256"] == digest(original.tolist())
                and binding["source_raw_norm"] == float(arrays["contrast_norm"][index, 0, 0])
                and binding["unit"] == (original * orientation).tolist(),
                "Semantic unit/sign/source differs",
            )
    for arm in PAIRS:
        binding, item = bindings[arm], by_arm[arm]
        unit = np.asarray(binding["unit"], dtype=np.float64)
        require(
            unit.shape == (5120,)
            and np.isfinite(unit).all()
            and math.isclose(float(np.linalg.norm(unit)), 1.0, abs_tol=1e-6)
            and binding["unit_sha256"] == digest(binding["unit"])
            and binding["layer"] == 15
            and binding["position"] == "record"
            and binding["reference_raw_norm"] == C_NORM
            and binding["dose"] == (0.0 if arm == "zero_sham" else C_NORM)
            and item["binding_sha256"] == digest(binding)
            and item["signed_unit_float64_byte_sha256"] == byte_sha(unit.tobytes())
            and item["effective_float32_byte_sha256"] == byte_sha(unit.astype(np.float32).tobytes()),
            "Direction magnitude/site/effective arithmetic differs",
        )
        if arm in RANDOM_ARMS:
            require(
                binding["source"] == "seeded_gaussian"
                and binding["corpus"] is None
                and binding["orientation"] == 1
                and binding["seed"] == 2026101501 + RANDOM_ARMS.index(arm)
                and binding["generator"] == "PCG64"
                and binding["construction"] == "numpy_default_rng_standard_normal_float64_unit_no_resampling",
                "Original random direction identity differs",
            )
    return deepcopy(bindings), deepcopy(inventory)


def validate_base(satisfaction, curiosity):
    require(
        digest(satisfaction) == SOURCES["satisfaction"]["manifest_sha256"]
        and digest(curiosity) == SOURCES["curiosity"]["manifest_sha256"],
        "Wrong source manifests",
    )
    base = satisfaction["screen_plan"]["base_assignments"]
    require(
        len(base) == 288
        and len({a["base_trial_id"] for a in base}) == 288
        and digest(base) == satisfaction["screen_plan"]["base_assignment_sha256"] == BASE_SHA
        and curiosity["screen_plan"]["base_assignments"] == base
        and curiosity["source_screen_manifest"] == satisfaction
        and curiosity["source_screen_manifest_sha256"] == digest(satisfaction),
        "Original 288 base branches differ",
    )
    grouped = defaultdict(list)
    for a in base:
        source = a["source_assignment"]
        require(
            digest(source) == a["source_assignment_sha256"]
            and source["task_revision"] == "voluntary-resolution-v2"
            and source["trial_id"] == a["base_trial_id"]
            and source["sampling_seed"] == a["sampling_seed"]
            and source["seed"] == a["panel_seed"]
            and source["world_slot"] == a["world_slot"]
            and source["semantic_world_id"] == a["semantic_world_id"]
            and a["family"] == source["family"] == "objects"
            and a["world_id"] == a["panel_id"] + ":" + a["semantic_world_id"],
            "Source identity changed",
        )
        for field in ("kind", "delivery", "menu_layout"):
            require(a[field] == source[field], "Base/source condition metadata differs")
        grouped[(a["panel_id"], a["semantic_world_id"])].append(a)
    require(
        len(grouped) == len({a["semantic_world_id"] for a in base}) == 24
        and Counter(panel for panel, _ in grouped) == {"panel_1": 12, "panel_2": 12},
        "Fixed world panels differ",
    )
    expected = {("choice", delivery, menu) for delivery in DELIVERIES for menu in (0, 1)}
    expected |= {
        (
            kind,
            "decisive_evidence" if kind in ("identify_resolved", "repeated_fact") else "redundant_evidence",
            0,
        )
        for kind in DIAGNOSTICS
    }
    for rows in grouped.values():
        require(
            len(rows) == 12 and {(r["kind"], r["delivery"], r["menu_layout"]) for r in rows} == expected,
            "All original choices/diagnostic variants required",
        )
    s, c = satisfaction["generation_preflight"], curiosity["generation_preflight"]
    require(
        digest(s) == satisfaction["generation_preflight_sha256"]
        and digest(c) == curiosity["generation_preflight_sha256"]
        and s["requests"] == c["requests"]
        and s["requests_sha256"] == c["requests_sha256"] == BASE_REQUESTS_SHA
        and digest(s["requests"]) == BASE_REQUESTS_SHA
        and s["base_assignment_sha256"] == c["base_assignment_sha256"] == BASE_SHA,
        "Original token preflight differs",
    )
    for a, item in zip(base, s["requests"], strict=True):
        request = item["request"]
        require(
            item["base_trial_id"] == request["source_trial_id"] == a["base_trial_id"]
            and item["source_assignment_sha256"] == a["source_assignment_sha256"]
            and item["inference_request_sha256"] == a["inference_request_sha256"]
            and request["seed"] == a["sampling_seed"]
            and request["labels"] == "ABCD"
            and digest(request["token_prefix"]) == request["token_prefix_sha256"]
            and digest(item["rendered_prompt"]) == item["rendered_prompt_sha256"]
            and request["prefix_positions"]["final"] == len(request["token_prefix"]) - 1
            and 0 <= request["prefix_positions"]["record"] <= request["prefix_positions"]["final"],
            "Original prompt/request/seed identity differs",
        )
    require(
        satisfaction["receipt_preflight"] == curiosity["receipt_preflight"]
        and digest(satisfaction["receipt_preflight"])
        == satisfaction["receipt_preflight_sha256"]
        == curiosity["receipt_preflight_sha256"],
        "Original receipt materials differ",
    )
    return deepcopy(base), deepcopy(s)


def arm_order(cohort):
    require(type(cohort) is int and 0 <= cohort < 9, "Original source cohort must be 0..8")
    offset = 5 * cohort % len(PAIRS)
    pairs = PAIRS[offset:] + PAIRS[:offset]
    schedules = ("record", "full") if cohort % 2 == 0 else ("full", "record")
    return ("baseline", *(f"{schedule}__{direction}" for direction in pairs for schedule in schedules))


def arm_spec(arm, bindings):
    require(arm in ARMS, "Unknown fixed arm")
    if arm == "baseline":
        return dict(
            arm=arm,
            schedule="baseline",
            intervention_schedule=SCHEDULES["record"],
            direction_key=None,
            position=None,
            intervention=None,
        )
    schedule, key = arm.split("__")
    patch = dict(
        layer=15,
        position=POSITIONS[schedule],
        mode="add",
        direction_key=key,
        direction_binding_sha256=digest(bindings[key]),
        dose=bindings[key]["dose"],
    )
    return dict(
        arm=arm,
        schedule=schedule,
        intervention_schedule=SCHEDULES[schedule],
        direction_key=key,
        position=POSITIONS[schedule],
        intervention=patch,
    )


def receipt_materials(base, receipt_preflight, assignments):
    choices = {a["base_trial_id"] for a in base if a["kind"] == "choice"}
    bindings = receipt_preflight["non_stop_bindings"]
    stops = receipt_preflight["stop_checks"]
    require(
        receipt_preflight["passed"] is True
        and receipt_preflight["generated_model_outputs"] is False
        and receipt_preflight["synthetic_answer_only"] is True
        and receipt_preflight["base_assignment_sha256"] == BASE_SHA
        and receipt_preflight["bindings_sha256"] == digest(dict(non_stop=bindings, stop=stops))
        and len(stops) == 144
        and set(stops) == choices
        and len(bindings) == 432,
        "Require every original possible receipt disposition",
    )
    by_base = defaultdict(dict)
    for binding in bindings:
        key, service = binding["base_trial_id"], binding["selected_service"]
        require(
            key in choices and service in ("report", "repeat", "pattern") and service not in by_base[key],
            "Duplicate/unexpected receipt possibility",
        )
        positions = binding["prefix_positions"]
        require(0 <= positions["record"] <= positions["final"], "Invalid inherited receipt token positions")
        by_base[key][service] = deepcopy(binding)
    require(
        all(set(by_base[key]) == {"report", "repeat", "pattern"} for key in choices),
        "Incomplete receipt choices",
    )
    lengths = [b["prefix_positions"]["final"] + 1 for b in bindings]
    rows = [
        dict(
            trial_id=a["trial_id"],
            base_trial_id=a["base_trial_id"],
            source_assignment_sha256=a["source_assignment_sha256"],
            non_stop_bindings=deepcopy(by_base[a["base_trial_id"]]),
            stop_requires_forward=False,
            selected_service=None,
            execution_claimed=False,
        )
        for a in assignments
        if a["kind"] == "choice"
    ]
    return dict(
        source_preflight=deepcopy(receipt_preflight),
        source_preflight_sha256=digest(receipt_preflight),
        assigned_dispositions=rows,
        assigned_dispositions_sha256=digest(rows),
        assigned_choices=len(rows),
        possible_base_nonstop_payloads=432,
        max_base_nonstop_per_arm=144,
        possible_scientific_nonstop_payloads=432 * len(ARMS),
        max_scientific_nonstop_forwards=len(rows),
        maximum_rendered_token_length=max(lengths),
        maximum_padded_width=128 * math.ceil(max(lengths) / 128),
        padding_multiple=128,
        width_source="Inherited receipt preflight final-token index plus one; no new tokenization or inference",
        actual_selected_payload_only=True,
        supplemental_synthetic_timing_forwards=False,
    )


def build_materials(satisfaction, curiosity, directions, inventory, provenance, engineering_call_plan):
    base, preflight = validate_base(satisfaction, curiosity)
    require(
        digest(directions) == DIRECTION_BINDINGS_SHA
        and directions == curiosity["screen_plan"]["direction_bindings"],
        "Use unchanged inherited directions",
    )
    rows, cohorts = [], []
    for source_cohort in range(9):
        for arm_position, arm in enumerate(arm_order(source_cohort)):
            spec = arm_spec(arm, directions)
            start = len(rows)
            for base_row in range(source_cohort * 32, (source_cohort + 1) * 32):
                a = base[base_row]
                rows.append(
                    dict(
                        **deepcopy(a),
                        **deepcopy(spec),
                        assay="curiosity_coverage_bridge",
                        task_revision=REVISION,
                        source_task_revision="voluntary-resolution-v2",
                        trial_id=f"{REVISION}:{arm}:{a['base_trial_id']}",
                        parent_row=len(rows),
                        base_row=base_row,
                        base_cohort=source_cohort,
                        arm_order_in_cohort=arm_position,
                        execution_group=f"worker_{source_cohort % 3}",
                        max_new_tokens=4096,
                        execution=dict(backend="torch_batched_v1", batch_size=32),
                    )
                )
            cohorts.append(
                dict(
                    offset=start,
                    source_cohort=source_cohort,
                    arm=arm,
                    arm_order_in_cohort=arm_position,
                    execution_group=f"worker_{source_cohort % 3}",
                    trial_ids=[a["trial_id"] for a in rows[start:]],
                    base_trial_ids=[a["base_trial_id"] for a in rows[start:]],
                )
            )
    require(
        len(rows) == len({a["trial_id"] for a in rows}) == 8352 and len(cohorts) == 261,
        "Fixed complete crossing differs",
    )
    requests = [deepcopy(item["request"]) for item in preflight["requests"]]
    stats = [
        dict(
            source_cohort=i,
            total_unpadded_tokens=sum(len(r["token_prefix"]) for r in requests[i * 32 : (i + 1) * 32]),
            maximum_unpadded_tokens=max(len(r["token_prefix"]) for r in requests[i * 32 : (i + 1) * 32]),
        )
        for i in range(9)
    ]
    require(
        max(stats, key=lambda x: (x["total_unpadded_tokens"], -x["source_cohort"]))["source_cohort"] == 3,
        "Predeclared maximum-total-input cohort is no longer3",
    )
    requests_by_cohort = {i: deepcopy(requests[i * 32 : (i + 1) * 32]) for i in (0, 3)}
    calls = engineering_call_plan(requests_by_cohort)
    require(
        len(calls) == 16
        and sum(len(c["trial_ids"]) for c in calls) == 512
        and sum(len(c["trial_ids"]) for c in calls if c["expected_error"] is not None) == 64,
        "Require fixed sixteen engineering calls/512slots/two faults",
    )
    require(
        [c["call_index"] for c in calls] == list(range(16))
        and len({c["call_id"] for c in calls}) == 16
        and len({trial for c in calls for trial in c["trial_ids"]}) == 512,
        "Engineering call/slot identities must be unique and ordered",
    )
    for call in calls:
        source = requests_by_cohort.get(call["source_cohort"])
        require(
            source is not None
            and len(call["trial_ids"]) == 32
            and call["base_trial_ids"] == [r["source_trial_id"] for r in source]
            and call["requests_sha256"] == digest(source)
            and call["cap"] == (128 if call["call_index"] < 10 else 4096)
            and call["stage"] == ("checks" if call["call_index"] < 10 else "timing"),
            "Engineering source companions/cap/stage differ",
        )
    outer = [
        dict(
            trial_id=a["trial_id"],
            base_trial_id=a["base_trial_id"],
            source_assignment_sha256=a["source_assignment_sha256"],
            sampling_seed=a["sampling_seed"],
            request_sha256=digest(requests[a["base_row"]]),
        )
        for a in rows
    ]
    generated = deepcopy(preflight)
    generated.update(
        assignment_sha256=digest(rows),
        outer_request_bindings=outer,
        outer_request_bindings_sha256=digest(outer),
    )
    receipts = receipt_materials(base, satisfaction["receipt_preflight"], rows)
    return dict(
        schema_version=1,
        name=REVISION,
        assay="curiosity_coverage_bridge",
        phase="draft_materials_only",
        inference_scope="development",
        scientific_admission=False,
        dose=C_NORM,
        criteria=deepcopy(CRITERIA),
        criteria_sha256=digest(CRITERIA),
        engineering_admission=False,
        causal_admission=False,
        model=deepcopy(MODEL),
        source_provenance=deepcopy(provenance),
        base_assignments=base,
        base_assignment_sha256=digest(base),
        base_requests=requests,
        base_requests_sha256=digest(requests),
        source_generation_preflight=preflight,
        generation_preflight=generated,
        generation_preflight_sha256=digest(generated),
        assignments=rows,
        assignment_sha256=digest(rows),
        execution_cohorts=cohorts,
        execution_cohorts_sha256=digest(cohorts),
        arm_specs=[arm_spec(arm, directions) for arm in ARMS],
        direction_bindings=deepcopy(directions),
        direction_bindings_sha256=digest(directions),
        direction_inventory=deepcopy(inventory),
        direction_inventory_sha256=digest(inventory),
        receipt_materials=receipts,
        receipt_materials_sha256=digest(receipts),
        selection=dict(
            mode="all24_previously_exposed_S_C_worlds_reused",
            fresh_worlds=0,
            selection_uses_outcomes=False,
            panels=deepcopy(satisfaction["screen_plan"]["selection"]["panels"]),
            worlds=deepcopy(satisfaction["screen_plan"]["selection"]["worlds"]),
        ),
        order_policy=dict(
            baseline_first=True,
            pair_rotation="5 * source_cohort modulo14",
            pair_order=list(PAIRS),
            within_pair="record/full for even cohort; full/record for odd cohort",
            worker="source_cohort modulo3",
            arms_by_base_cohort=[list(arm_order(i)) for i in range(9)],
            exactly_counterbalanced=False,
        ),
        planned_denominators=dict(
            worlds=24,
            panels=2,
            arms=29,
            base_branches=288,
            generations=8352,
            choices=4176,
            diagnostics=4176,
            generation_cohorts=261,
            per_worker_generations=2784,
            per_worker_cohorts=87,
            diagnostic_readiness_cells=348,
            choice_readiness_cells=174,
        ),
        engineering=dict(
            call_plan=calls,
            call_plan_sha256=digest(calls),
            requests_by_cohort={str(k): v for k, v in requests_by_cohort.items()},
            requests_by_cohort_sha256=digest({str(k): v for k, v in requests_by_cohort.items()}),
            source_assignments_by_cohort={
                str(k): [deepcopy(a["source_assignment"]) for a in base[k * 32 : (k + 1) * 32]]
                for k in (0, 3)
            },
            potential_receipt_workload={
                key: deepcopy(receipts[key])
                for key in (
                    "possible_base_nonstop_payloads",
                    "max_base_nonstop_per_arm",
                    "possible_scientific_nonstop_payloads",
                    "max_scientific_nonstop_forwards",
                    "maximum_rendered_token_length",
                    "maximum_padded_width",
                    "padding_multiple",
                    "width_source",
                    "actual_selected_payload_only",
                    "supplemental_synthetic_timing_forwards",
                )
            },
            source_input_statistics=stats,
            fixed_source_cohorts=[0, 3],
            selection_reason="cohort0 first; cohort3 maximum total unpadded input tokens, earliest tie",
            assigned_generation_calls=16,
            assigned_logical_slots=512,
            deliberate_fault_slots=64,
            ordinary_slots=448,
            retained_source_rows=[0],
            retained_decode_ordinals=[1, 2, 32, 128],
            actual_selected_receipts_only=True,
            no_retry_or_replacement=True,
        ),
        resources=resources(),
        all_prior_results_remain_closed=True,
        new_direction_or_material_sampling=False,
    )


def build_plan(project_root=ROOT):
    """Return a draft dictionary; no files, model outputs, admission or freeze."""
    from curiosity_study.coverage_engineering import call_plan
    import curiosity_study.coverage_engineering as engineering

    root = Path(project_root)
    satisfaction, _, s_provenance = verify_package(root, "satisfaction")
    curiosity, c_documents, c_provenance = verify_package(root, "curiosity")
    archived_s = json.loads(c_documents["runs/" + SOURCES["satisfaction"]["run"] + "/manifest.json"])
    require(archived_s == satisfaction, "C archive binds a different original S parent")
    directions, inventory = validate_directions(curiosity, c_documents)
    provenance = dict(
        satisfaction=s_provenance,
        curiosity=c_provenance,
        plan_path=PLAN_PATH,
        plan_byte_sha256=sha(root / PLAN_PATH),
        material_builder_byte_sha256=sha(__file__),
        engineering_helper_byte_sha256=sha(engineering.__file__),
        copied_original_runtime_source_sha256=SOURCES["curiosity"]["source_sha256"],
        inherited_candidate_source=deepcopy(curiosity["candidate_source"]),
        lifecycle_admission_checked=False,
        outcome_records_inspected=False,
    )
    return build_materials(satisfaction, curiosity, directions, inventory, provenance, call_plan)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    plan = build_plan(args.root)
    print(
        json.dumps(
            dict(
                assignment_sha256=plan["assignment_sha256"],
                denominators=plan["planned_denominators"],
                engineering_call_plan_sha256=plan["engineering"]["call_plan_sha256"],
                admitted=False,
                files_written=False,
            ),
            indent=2,
        )
    )
