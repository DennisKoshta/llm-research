"""Independent synthetic estimator checks; never read model outcomes or axes."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import random
import sys

import numpy as np
import pytest

from curiosity_study.concept_capture import CONTRASTS
from curiosity_study.voluntary_resolution import DELIVERIES, FAMILIES

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


@pytest.fixture(scope="module")
def analyzer():
    path = SCRIPTS / "analyze_voluntary_resolution.py"
    spec = importlib.util.spec_from_file_location("voluntary_statistics_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_axes():
    shape = (8, 4, 2)
    unit = np.zeros((*shape, 3))
    unit[..., 1] = 1  # Every unselected site deliberately points elsewhere.
    unit[:, 0, 0] = [1, 0, 0]
    center = np.zeros_like(unit)
    center[:, 0, 0] = [1, 2, 3]
    norms = np.full(shape, 400.0)
    norms[:, 0, 0] = 10.0
    threshold = np.full(shape, 900.0)
    threshold[:, 0, 0] = 1.0
    return dict(
        sources=np.array(CONTRASTS),
        layers=np.array([15, 31, 47, 57]),
        positions=np.array(["record", "final"]),
        unit=unit,
        center=center,
        contrast=unit * norms[..., None],
        contrast_norm=norms,
        threshold=threshold,
    )


def synthetic_tables():
    choices, projections = [], []
    for fi, family in enumerate(FAMILIES):
        for world in range(12):
            semantic_id = f"{family}-semantic-{world:02d}"
            for delivery in DELIVERIES:
                for menu in (0, 1):
                    tid = f"{semantic_id}-{delivery}-{menu}"
                    if delivery == "decisive_evidence":
                        category = "stop" if menu == 0 else "report" if (world + fi) % 3 == 0 else "invalid"
                    elif delivery == "redundant_evidence":
                        category = "report" if menu == 0 or (world + fi) % 2 == 0 else "repeat"
                    else:
                        category = "pattern" if menu == 0 else "stop"
                    choices.append(
                        dict(
                            trial_id=tid,
                            family=family,
                            semantic_world_id=semantic_id,
                            world_slot=world,
                            delivery=delivery,
                            menu_layout=menu,
                            category=category,
                            cap_hit=False,
                            thinking_closed=True,
                        )
                    )
                    for ai, axis in enumerate(CONTRASTS):
                        sign = -1 if ai == 0 else 1
                        for position in ("record", "prefill"):
                            value = 100 * fi + 10 * world + 2 * menu + (1000 if position == "prefill" else 0)
                            if delivery == "decisive_evidence":
                                value += sign * (world + 1) * (fi + 1) + 3 * menu - 1
                            elif delivery == "independent_novelty":
                                value += 3
                            projections.append(
                                dict(
                                    trial_id=tid,
                                    axis=axis,
                                    position=position,
                                    raw=float(value),
                                    scaled=float(value) / 10,
                                )
                            )
    return choices, projections


def endpoint(result, kind, family="objects", contrast="decisive_minus_redundant", **selectors):
    rows = [
        row
        for row in result[kind]
        if row["family"] == family
        and row["contrast"] == contrast
        and all(row[key] == value for key, value in selectors.items())
    ]
    assert len(rows) == 1
    return rows[0]


def independent_world_values(
    choices, projections, family, contrast, *, category=None, axis=None, position="record", scale="raw"
):
    """Build per-world paired values from named cells, with no analyzer helpers."""
    right = "redundant_evidence" if contrast == "decisive_minus_redundant" else "independent_novelty"
    selected = [row for row in choices if row["family"] == family]
    worlds = sorted({row["semantic_world_id"] for row in selected})
    values = {}
    if category is not None:
        values = {row["trial_id"]: float(row["category"] == category) for row in selected}
    else:
        values = {
            row["trial_id"]: row[scale]
            for row in projections
            if row["axis"] == axis and row["position"] == position
        }
    result = []
    for world in worlds:
        means = []
        for delivery in ("decisive_evidence", right):
            ids = [
                row["trial_id"]
                for row in selected
                if row["semantic_world_id"] == world and row["delivery"] == delivery
            ]
            assert len(ids) == 2
            means.append(sum(values[tid] for tid in ids) / 2)
        result.append(means[0] - means[1])
    return np.array(result)


def independent_bootstrap(values):
    rng = np.random.default_rng(2026101434)
    draws = {family: values[family][rng.integers(0, 12, size=(10000, 12))].mean(1) for family in FAMILIES}
    draws["pooled"] = (draws["objects"] + draws["machines"]) / 2
    estimates = {family: values[family].mean() for family in FAMILIES}
    estimates["pooled"] = (estimates["objects"] + estimates["machines"]) / 2
    return {
        family: (float(estimates[family]), np.quantile(draw, [0.025, 0.975]))
        for family, draw in draws.items()
    }


def test_fixed_selected_axes_use_a15_record_full_norm_center_and_unchanged_threshold(analyzer):
    arrays = synthetic_axes()
    axes = analyzer.selected_axes(arrays)
    assert [axis["name"] for axis in axes] == list(CONTRASTS)
    for axis in axes:
        assert axis["available"] is True and axis["norm"] == 10 and axis["threshold"] == 1
        np.testing.assert_array_equal(axis["unit"], [1, 0, 0])
        np.testing.assert_array_equal(axis["center"], [1, 2, 3])
        raw, scaled = analyzer.project(np.array([21.0, 500.0, 0.0]), axis)
        assert raw == 20.0 and scaled == 2.0  # Full norm10; no half-contrast5 or final-axis rescue.
        assert analyzer.project(np.array([31.0, 700.0, 0.0]), axis) == (30.0, 3.0)


def test_locked_axis_sign_is_not_reoriented_to_behavior(analyzer):
    arrays = synthetic_axes()
    arrays["unit"][0, 0, 0] *= -1
    arrays["contrast"][0, 0, 0] *= -1
    arrays["threshold"][0, 0, 0] *= -1
    axis = analyzer.selected_axes(arrays)[0]
    assert analyzer.project(np.array([21.0, 500.0, 0.0]), axis) == (-20.0, -2.0)


def test_zero_axis_is_unavailable_without_substitution_of_later_layer_or_final(analyzer):
    arrays = synthetic_axes()
    arrays["unit"][0, 0, 0] = 0
    arrays["contrast"][0, 0, 0] = 0
    arrays["contrast_norm"][0, 0, 0] = 0
    axes = analyzer.selected_axes(arrays)
    assert axes[0]["available"] is False
    assert analyzer.project(np.array([5.0, 8.0, 9.0]), axes[0]) == (None, None)
    assert all(axis["available"] for axis in axes[1:])


@pytest.mark.parametrize("contrast", ["decisive_minus_redundant", "decisive_minus_novelty"])
def test_paired_world_service_bootstrap_and_equal_family_pool_match_independent_math(analyzer, contrast):
    choices, projections = synthetic_tables()
    result = analyzer.analyze_tables(choices, projections)
    for category in ("report", "repeat", "pattern", "stop", "invalid", "execution_failure", "missing"):
        world_values = {
            family: independent_world_values(choices, projections, family, contrast, category=category)
            for family in FAMILIES
        }
        expected = independent_bootstrap(world_values)
        for family in (*FAMILIES, "pooled"):
            row = endpoint(result, "behavior_contrasts", family, contrast, category=category)
            assert row["assigned_worlds"] == row["available_worlds"] == (24 if family == "pooled" else 12)
            assert row["estimate"] == pytest.approx(expected[family][0])
            np.testing.assert_allclose(row["ci95"], expected[family][1], rtol=0, atol=1e-14)
    for family in (*FAMILIES, "pooled"):
        rows = [
            row
            for row in result["behavior_contrasts"]
            if row["family"] == family and row["contrast"] == contrast
        ]
        assert sum(row["estimate"] for row in rows) == pytest.approx(0.0)


def test_invalid_is_observed_mass_and_not_conditioned_out_of_menu_or_world(analyzer):
    choices, projections = synthetic_tables()
    result = analyzer.analyze_tables(choices, projections)
    invalid = endpoint(result, "behavior_contrasts", category="invalid")
    report = endpoint(result, "behavior_contrasts", category="report")
    assert invalid["estimate"] == pytest.approx(1 / 3)
    assert report["estimate"] == pytest.approx(1 / 6 - 3 / 4)
    assert report["assigned_worlds"] == 12


def test_projection_intervals_share_world_draws_sign_and_full_norm_scale(analyzer):
    choices, projections = synthetic_tables()
    result = analyzer.analyze_tables(choices, projections)
    for contrast in ("decisive_minus_redundant", "decisive_minus_novelty"):
        for axis in CONTRASTS[:2]:
            values = {
                family: independent_world_values(choices, projections, family, contrast, axis=axis)
                for family in FAMILIES
            }
            expected = independent_bootstrap(values)
            for family in (*FAMILIES, "pooled"):
                for position in ("record", "prefill"):
                    raw = endpoint(
                        result,
                        "projection_contrasts",
                        family,
                        contrast,
                        axis=axis,
                        position=position,
                        scale="raw",
                    )
                    scaled = endpoint(
                        result,
                        "projection_contrasts",
                        family,
                        contrast,
                        axis=axis,
                        position=position,
                        scale="scaled",
                    )
                    assert raw["estimate"] == pytest.approx(expected[family][0])
                    np.testing.assert_allclose(raw["ci95"], expected[family][1], rtol=0, atol=1e-12)
                    assert scaled["estimate"] == pytest.approx(raw["estimate"] / 10)
                    np.testing.assert_allclose(scaled["ci95"], np.array(raw["ci95"]) / 10, atol=1e-12)
                    if contrast == "decisive_minus_redundant":
                        assert (raw["estimate"] < 0) is (axis == CONTRASTS[0])


def test_common_world_offsets_cancel_in_paired_contrast_not_independent_arm_bootstrap(analyzer):
    choices, projections = synthetic_tables()
    for row in projections:
        tid = row["trial_id"]
        world = int(tid.split("-semantic-")[1][:2])
        # Huge heterogeneous baselines; a constant within-world treatment shift.
        row["raw"] = float(10000 * world + (2 if "decisive_evidence" in tid else 0))
        row["scaled"] = row["raw"] / 10
    result = analyzer.analyze_tables(choices, projections)
    row = endpoint(
        result, "projection_contrasts", "pooled", axis=CONTRASTS[0], position="record", scale="raw"
    )
    assert row["estimate"] == 2 and row["ci95"] == [2.0, 2.0]


@pytest.mark.parametrize("missing_mode", ["absent", "null"])
def test_missing_projection_does_not_drop_world_or_average_only_surviving_menu(analyzer, missing_mode):
    choices, projections = synthetic_tables()
    target = next(
        row
        for row in projections
        if row["trial_id"].startswith("objects-semantic-00-decisive_evidence-0")
        and row["axis"] == CONTRASTS[0]
        and row["position"] == "record"
    )
    if missing_mode == "absent":
        projections.remove(target)
        with pytest.raises(ValueError):
            analyzer.analyze_tables(choices, projections)
        return
    else:
        target.update(raw=None, scaled=None)
    result = analyzer.analyze_tables(choices, projections)
    for family, available in (("objects", 11), ("pooled", 23)):
        row = endpoint(
            result, "projection_contrasts", family, axis=CONTRASTS[0], position="record", scale="raw"
        )
        assert row["estimate"] is None and row["ci95"] is None
        assert row["assigned_worlds"] == (12 if family == "objects" else 24)
        assert row["available_worlds"] == available
    assert (
        endpoint(
            result, "projection_contrasts", "machines", axis=CONTRASTS[0], position="record", scale="raw"
        )["estimate"]
        is not None
    )
    assert (
        endpoint(result, "projection_contrasts", axis=CONTRASTS[1], position="record", scale="raw")[
            "estimate"
        ]
        is not None
    )


@pytest.mark.parametrize(
    "tamper",
    ["missing_choice", "duplicate_choice", "duplicate_cell", "duplicate_projection", "unknown_category"],
)
def test_exact_six_choice_crossing_and_unique_projection_keys_required(analyzer, tamper):
    choices, projections = synthetic_tables()
    if tamper == "missing_choice":
        choices.pop()
    elif tamper == "duplicate_choice":
        choices.append(deepcopy(choices[0]))
    elif tamper == "duplicate_cell":
        choices[1]["menu_layout"] = choices[0]["menu_layout"]
    elif tamper == "duplicate_projection":
        projections.append(deepcopy(projections[0]))
    else:
        choices[0]["category"] = "possibly_report"
    with pytest.raises(ValueError):
        analyzer.analyze_tables(choices, projections)


def test_input_row_permutation_does_not_change_draw_units_or_results(analyzer):
    choices, projections = synthetic_tables()
    original = analyzer.analyze_tables(choices, projections)
    random.Random(713).shuffle(choices)
    random.Random(193).shuffle(projections)
    permuted = analyzer.analyze_tables(choices, projections)
    assert original == permuted


def test_recorded_error_mass_is_not_recoded_as_stop_or_invalid(analyzer):
    choices, projections = synthetic_tables()
    for category, menu in (("execution_failure", 0), ("missing", 1)):
        row = next(
            row
            for row in choices
            if row["family"] == "objects"
            and row["world_slot"] == 0
            and row["delivery"] == "decisive_evidence"
            and row["menu_layout"] == menu
        )
        row["category"] = category
    result = analyzer.analyze_tables(choices, projections)
    for category in ("execution_failure", "missing"):
        row = endpoint(result, "behavior_contrasts", category=category)
        assert row["estimate"] == pytest.approx(1 / 24) and row["assigned_worlds"] == 12
