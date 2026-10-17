from collections import Counter, defaultdict
from copy import deepcopy
import itertools

import numpy as np
import pytest

from curiosity_study import curiosity_suppression_screen as screen
from curiosity_study import satisfaction_screen as old
from curiosity_study import voluntary_resolution as voluntary
from curiosity_study.artifacts import digest


@pytest.fixture
def inputs(monkeypatch):
    selected = old.select_object_panels(historical_exact_world_ids=[])
    base = []
    for world in selected["worlds"]:
        variants = [("choice", d, m) for d, m in itertools.product(voluntary.DELIVERIES, (0, 1))]
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
            source = old._source_assignment(world, kind, delivery, menu)
            base.append(
                dict(
                    base_trial_id=source["trial_id"],
                    source_assignment=source,
                    source_assignment_sha256=digest(source),
                    inference_request_sha256=digest(voluntary.inference_request(source)),
                    sampling_seed=source["sampling_seed"],
                    semantic_world_id=world["semantic_world_id"],
                    panel_id=world["panel_id"],
                    panel_seed=world["seed"],
                    world_slot=world["world_slot"],
                    world_id=world["world_id"],
                    family="objects",
                    kind=kind,
                    delivery=delivery,
                    menu_layout=menu,
                )
            )
    manifest = dict(
        assay="satisfaction_screen",
        screen_plan=dict(base_assignments=base, base_assignment_sha256=digest(base), selection=selected),
    )
    monkeypatch.setattr(screen, "PRIOR_MANIFEST_SHA256", digest(manifest))
    bindings = {}
    sources = ["curiosity_vs_indifference", *(s for _, s, _ in screen.NEIGHBORS)]
    semantic = (("zero_sham", sources[0], 1), (screen.PRIMARY, sources[0], -1), *screen.NEIGHBORS)
    for arm, source, orientation in semantic:
        unit = np.zeros(5120)
        unit[sources.index(source)] = orientation
        bindings[arm] = dict(
            source=source,
            orientation=orientation,
            corpus="A",
            layer=15,
            position="record",
            source_raw_norm=1.0,
            directions_byte_sha256=screen.LOCKED_DIRECTION_BYTES,
            unit=unit.tolist(),
            unit_sha256=digest(unit.tolist()),
            reference_raw_norm=screen.LOCKED_C_NORM,
            dose=0.0 if arm == "zero_sham" else screen.LOCKED_C_NORM,
        )
    for arm, seed in zip(screen.RANDOM_ARMS, screen.RANDOM_SEEDS, strict=True):
        unit = np.random.default_rng(seed).standard_normal(5120)
        unit /= np.linalg.norm(unit)
        bindings[arm] = dict(
            source="seeded_gaussian",
            orientation=1,
            corpus=None,
            layer=15,
            position="record",
            seed=seed,
            construction="numpy_default_rng_standard_normal_float64_unit_no_resampling",
            unit=unit.tolist(),
            unit_sha256=digest(unit.tolist()),
            reference_raw_norm=screen.LOCKED_C_NORM,
            dose=screen.LOCKED_C_NORM,
        )
    return manifest, bindings


def test_full_crossing_preserves_all_sources_and_companions(inputs):
    manifest, bindings = inputs
    p = screen.assignment_plan(source_manifest=manifest, direction_bindings=bindings)
    assert len(p["assignments"]) == 4320
    by_base = defaultdict(list)
    for row in p["assignments"]:
        by_base[row["base_row"]].append(row)
        assert row["source_assignment"] == p["base_assignments"][row["base_row"]]["source_assignment"]
        assert screen.inference_request(row) == voluntary.inference_request(row["source_assignment"])
    assert len(by_base) == 288
    assert all({r["arm"] for r in group} == set(screen.ARMS) for group in by_base.values())
    for start in range(0, 4320, 32):
        cohort = p["assignments"][start : start + 32]
        assert len({r["arm"] for r in cohort}) == 1
        assert [r["base_row"] for r in cohort] == list(
            range(cohort[0]["base_cohort"] * 32, (cohort[0]["base_cohort"] + 1) * 32)
        )
    assert p["selection"]["fresh_worlds"] == 0
    assert p["selection"]["historical_exact_exposed_worlds"] == 24


def test_every_worker_gets_every_arm_and_baseline_precedes_zero(inputs):
    p = screen.assignment_plan(source_manifest=inputs[0], direction_bindings=inputs[1])
    collected = []
    for group in screen.GROUPS:
        rows = screen.group_assignments(p, group)
        collected.extend(rows)
        assert len(rows) == 1440
        assert Counter(r["arm"] for r in rows) == Counter({a: 96 for a in screen.ARMS})
        seen = set()
        for row in rows:
            if row["arm"] == "baseline":
                seen.add(row["base_row"])
            if row["arm"] == "zero_sham":
                assert row["base_row"] in seen
    assert sorted(collected, key=lambda r: r["parent_row"]) == p["assignments"]
    for arm in screen.ARMS[1:]:
        assert len({screen.arm_order(i).index(arm) for i in range(9)}) == 9


def test_source_outcome_selection_or_mutation_is_rejected(inputs):
    manifest, bindings = deepcopy(inputs)
    manifest["screen_plan"]["base_assignments"].pop()
    with pytest.raises(ValueError, match="entire original"):
        screen.assignment_plan(source_manifest=manifest, direction_bindings=bindings)


@pytest.mark.parametrize(
    "field,value",
    [
        ("orientation", 1),
        ("dose", 0.5 * screen.LOCKED_C_NORM),
        ("layer", 31),
        ("position", "final"),
        ("source", "satisfaction_vs_neutral"),
        ("corpus", "B"),
    ],
)
def test_no_target_sign_dose_layer_or_candidate_substitution(inputs, field, value):
    bindings = deepcopy(inputs[1])
    bindings[screen.PRIMARY][field] = value
    with pytest.raises(ValueError):
        screen.validate_directions(bindings)


def test_each_mandatory_control_and_random_seed_is_required(inputs):
    for arm in screen.CONTROLS:
        bindings = deepcopy(inputs[1])
        del bindings[arm]
        with pytest.raises(ValueError):
            screen.validate_directions(bindings)
    bindings = deepcopy(inputs[1])
    bindings[screen.RANDOM_ARMS[0]]["seed"] += 1
    with pytest.raises(ValueError):
        screen.validate_directions(bindings)


def test_rehashed_valence_orientation_and_nonfinite_unit_rejected(inputs):
    bindings = deepcopy(inputs[1])
    bindings["valence_negative"]["unit"] = bindings["valence_positive"]["unit"][:]
    bindings["valence_negative"]["unit_sha256"] = digest(bindings["valence_negative"]["unit"])
    with pytest.raises(ValueError, match="opposite orientations"):
        screen.validate_directions(bindings)
    bindings = deepcopy(inputs[1])
    bindings[screen.PRIMARY]["unit"][0] = float("nan")
    with pytest.raises(ValueError):
        screen.validate_directions(bindings)


def test_construction_does_not_mutate_supplied_inputs(inputs):
    before = deepcopy(inputs)
    plan = screen.assignment_plan(source_manifest=inputs[0], direction_bindings=inputs[1])
    plan["assignments"][0]["source_assignment"]["seed"] = 0
    plan["direction_bindings"][screen.PRIMARY]["unit"][0] = 0
    assert inputs == before
