"""Assignment preservation and tamper rejection for the new bounded site study."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from curiosity_study import satisfaction_causal_sites as site
from curiosity_study.artifacts import digest, read

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def reference():
    medians = {"15": 100.0, "31": 200.0, "47": 400.0}
    return dict(
        formula=site.DOSE_FORMULA, original_dose=site.ORIGINAL_S_DOSE, base_branches=288,
        source_screen_manifest_sha256=site.PRIOR_MANIFEST_SHA256, capture_sha256="a" * 64,
        median_record_norm=medians,
        doses={str(layer): site.ORIGINAL_S_DOSE * medians[str(layer)] / medians["15"] for layer in site.SITES},
    )


@pytest.fixture(scope="module")
def inputs(reference):
    manifest = read(ROOT / "runs/qwen32-satisfaction-screen-v1/manifest.json")
    base, selection = site.calibration_inputs(manifest)
    bindings = {}
    with np.load(ROOT / "analyses/curiosity-satisfaction-candidates-v1/directions.npz", allow_pickle=False) as data:
        sources = data["sources"].tolist()
        for layer in site.SITES:
            li, by_arm = data["layers"].tolist().index(layer), {}
            for arm in site.CALIBRATION_ARMS[1:]:
                source = dict(site.controls.NEIGHBORS[i][:2] for i in range(len(site.controls.NEIGHBORS))).get(
                    arm, "satisfaction_vs_neutral"
                )
                if arm == "random_01":
                    raw = np.random.default_rng(2026101501).standard_normal(5120)
                    unit = raw / np.linalg.norm(raw)
                    metadata = dict(source="seeded_gaussian", corpus=None, seed=2026101501,
                                    construction="numpy_default_rng_standard_normal_float64_unit_no_resampling")
                else:
                    index = sources.index(source)
                    unit = data["unit"][index, li, 0]
                    metadata = dict(source=source, corpus="A", directions_byte_sha256=site.LOCKED_DIRECTION_BYTES,
                                    source_raw_norm=float(data["contrast_norm"][index, li, 0]))
                by_arm[arm] = dict(
                    **metadata, orientation=1, layer=layer, position="record", unit=unit.tolist(),
                    unit_sha256=digest(unit.tolist()),
                    dose=0.0 if arm == "zero_sham" else reference["doses"][str(layer)],
                )
            bindings[str(layer)] = by_arm
    return dict(phase="calibration", base_assignments=base, selection=selection,
                direction_bindings=bindings, dose_reference=reference)


@pytest.fixture(scope="module")
def plan(inputs):
    return site.assignment_plan(**inputs)


def test_every_original_request_and_companion_survives(plan, inputs):
    assert len(plan["assignments"]) == 4032
    assert len({r["trial_id"] for r in plan["assignments"]}) == 4032
    for offset in range(0, 4032, 32):
        rows = plan["assignments"][offset:offset + 32]
        base = rows[0]["base_cohort"] * 32
        assert len({(r["site"], r["arm"], r["execution_group"]) for r in rows}) == 1
        assert [r["base_row"] for r in rows] == list(range(base, base + 32))
        for row, source in zip(rows, inputs["base_assignments"][base:base + 32], strict=True):
            assert row["source_assignment"] == source["source_assignment"]
            assert digest(site.inference_request(row)) == source["inference_request_sha256"]


def test_every_condition_on_every_worker(plan):
    rows = [site.group_assignments(plan, group) for group in site.GROUPS]
    assert all(len(group) == 1344 for group in rows)
    assert {r["trial_id"] for group in rows for r in group} == {r["trial_id"] for r in plan["assignments"]}
    for group in rows:
        for layer in site.SITES:
            for arm in site.CALIBRATION_ARMS:
                assert sum(r["site"] == layer and r["arm"] == arm for r in group) == 96


def test_no_direction_is_transplanted_and_all_doses_are_fixed(plan, reference):
    for row in plan["assignments"]:
        patch = row["intervention"]
        if row["arm"] == "baseline":
            assert patch is None
        else:
            assert patch["layer"] == row["site"]
            assert patch["dose"] == (0 if row["arm"] == "zero_sham" else reference["doses"][str(row["site"])])


def test_calibration_world_reorder_is_rejected(inputs):
    changed = deepcopy(inputs)
    changed["base_assignments"][0], changed["base_assignments"][1] = changed["base_assignments"][1], changed["base_assignments"][0]
    with pytest.raises(ValueError, match="exact original ordered288"):
        site.assignment_plan(**changed)


def test_calibration_panel_selection_cannot_be_relabelled(inputs):
    changed = deepcopy(inputs)
    changed["selection"]["purpose"] = "different grouping"
    with pytest.raises(ValueError, match="original panel selection"):
        site.assignment_plan(**changed)


def test_positive_medians_cannot_silently_underflow_to_zero_dose(reference):
    changed = deepcopy(reference)
    changed["median_record_norm"] = {"15": 1e308, "31": 5e-324, "47": 5e-324}
    changed["doses"] = {str(layer): site.ORIGINAL_S_DOSE * 5e-324 / 1e308 for layer in site.SITES}
    assert changed["doses"] == {"31": 0.0, "47": 0.0}
    with pytest.raises(ValueError, match="one-dose formula"):
        site.validate_reference(changed)


@pytest.mark.parametrize("mutation", ["missing_control", "different_site", "changed_dose", "corpus_b", "oriented_random"])
def test_direction_contract_rejects_tampering(inputs, mutation):
    changed = deepcopy(inputs)
    directions = changed["direction_bindings"]["31"]
    if mutation == "missing_control":
        del directions["completion_positive"]
    elif mutation == "different_site":
        directions[site.PRIMARY]["layer"] = 15
    elif mutation == "changed_dose":
        directions[site.PRIMARY]["dose"] *= 2
    elif mutation == "corpus_b":
        directions[site.PRIMARY]["corpus"] = "B"
    else:
        directions["random_01"]["unit"] = [-x for x in directions["random_01"]["unit"]]
        directions["random_01"]["unit_sha256"] = digest(directions["random_01"]["unit"])
    with pytest.raises(ValueError):
        site.assignment_plan(**changed)


@pytest.mark.parametrize("value", [0.0, float("nan"), float("inf"), True])
def test_bad_reference_norm_rejected(reference, value):
    changed = deepcopy(reference)
    changed["median_record_norm"]["31"] = value
    with pytest.raises(ValueError):
        site.validate_reference(changed)


def test_even_small_dose_adjustment_rejected(reference):
    changed = deepcopy(reference)
    changed["doses"]["31"] += 1e-9
    with pytest.raises(ValueError):
        site.validate_reference(changed)


def test_calibration_cannot_preselect_one_site(inputs):
    with pytest.raises(ValueError, match="Calibration uses both sites"):
        site.assignment_plan(**inputs, selected_site=31)


def test_stage_caps_sum_to_declared_extension_and_bound_wall_liability():
    from curiosity_study import satisfaction_site_resources as resources

    assert sum(p["stage_ceiling_usd"] for p in resources.POLICIES.values()) == 65
    for phase, policy in resources.POLICIES.items():
        manifest = dict(
            assay=resources.SETUP_ASSAY if phase == "setup" else site.ASSAY,
            site_phase=phase, site_plan={"phase": phase}, budget=policy,
        )
        assert resources.job_wall_limit(manifest) == policy["shared_deadline_seconds"]
        assert policy["workers"] * (policy["shared_deadline_seconds"] + 120) * 0.00139376 < policy["stage_ceiling_usd"]


def test_reservation_rejects_mixing_workers_parents_and_stages():
    from curiosity_study import satisfaction_site_resources as resources

    original = [dict(
        assay=site.ASSAY, site_phase="calibration", site_plan={"phase": "calibration"},
        budget=resources.POLICIES["calibration"], execution_group=group,
        parent_manifest_sha256="a" * 64,
    ) for group in site.GROUPS]
    assert resources.reservation_limits(original) == (7800, 34)
    for mutation in ("parent", "worker", "phase", "ceiling"):
        changed = deepcopy(original)
        if mutation == "parent":
            changed[0]["parent_manifest_sha256"] = "b" * 64
        elif mutation == "worker":
            changed[0]["execution_group"] = site.GROUPS[1]
        elif mutation == "phase":
            changed[0].update(site_phase="validation", site_plan={"phase": "validation"},
                              budget=resources.POLICIES["validation"])
        else:
            changed[0]["budget"]["stage_ceiling_usd"] += 1
        with pytest.raises(ValueError):
            resources.reservation_limits(changed)


def test_old_completed_and_running_resource_policies_are_unchanged():
    from curiosity_study import curiosity_resource_limits as previous
    from curiosity_study import satisfaction_site_resources as resources

    for name in ("qwen32-satisfaction-screen-v1", "qwen32-curiosity-suppression-screen-v1-admission2"):
        manifest = read(ROOT / "runs" / name / "manifest.json")
        assert resources.job_wall_limit(manifest) == previous.job_wall_limit(manifest)
