import numpy as np
from curiosity_study.representation import (
    world_summaries,
    fit_source,
    projection_metrics,
    direction_geometry,
    CELL_ORDER,
)


def synthetic():
    assignments, values = [], []
    for family in ("objects", "machines"):
        for index in range(4):
            for kind in ("new", "repeat"):
                for known, query in CELL_ORDER:
                    gain = int(known != query)
                    answer = gain if kind == "new" else 1 - gain
                    assignments.append(
                        dict(family=family, index=index, kind=kind, known=[known], query=query)
                    )
                    values.append([[[3 * gain, 7 * answer, 11 * known, 13 * query, index]]])
    return assignments, np.array(values, dtype=float)


def test_factorial_balancing_separates_information_from_answer_and_main_effects():
    aa, x = synthetic()
    _, summaries = world_summaries(aa, x)
    fit = fit_source(summaries, ["objects", "machines"], ["new", "repeat"], range(4))
    np.testing.assert_array_equal(fit["gain"], [[[3, 0, 0, 0, 0]]])
    np.testing.assert_array_equal(fit["known"], [[[0, 0, 11, 0, 0]]])
    np.testing.assert_array_equal(fit["query"], [[[0, 0, 0, 13, 0]]])
    answer = fit_source(summaries, ["objects", "machines"], ["new", "repeat"], range(4), answer_contrast=True)
    np.testing.assert_array_equal(answer["gain"], [[[0, 7, 0, 0, 0]]])
    np.testing.assert_array_equal(fit["n_worlds_by_family"], [[[4]], [[4]]])


def test_missing_cell_removes_whole_world_for_both_source_polarities():
    aa, x = synthetic()
    x[0] = np.nan
    _, summaries = world_summaries(aa, x)
    fit = fit_source(summaries, ["objects", "machines"], ["new", "repeat"], range(4))
    np.testing.assert_array_equal(fit["n_worlds_by_family"], [[[3]], [[4]]])
    np.testing.assert_array_equal(fit["gain"], [[[3, 0, 0, 0, 0]]])


def test_world_cluster_interval_and_answer_only_reversal():
    scores = np.array([[0, 2, 2, 0], [1, 3, 3, 1], [np.nan, 8, 8, 0]])
    metric = projection_metrics(scores, 1.5)
    assert metric["worlds"] == 2 and metric["auc"] == 1 and metric["gap_ci95"] == [2, 2]
    reverse = projection_metrics(-scores, -1.5)
    assert reverse["auc"] == 0 and reverse["mean_factorial_gap"] == -2


def test_geometry_detects_asymmetric_answer_contamination_and_missing_directions():
    # Polarity averaging can retain the same answer axis when its two amplitudes differ.
    new, repeat = np.array([2.0, 0.0]), np.array([-1.0, 0.0])
    balanced, answer = (new + repeat) / 2, (new - repeat) / 2
    left = np.stack([balanced, [1, 0], [1, 0], [0, 0], [np.nan, 0]])
    right = np.stack([answer, [0, 2], [-2, 0], [1, 0], [1, 0]])
    g = direction_geometry(left, right)
    np.testing.assert_allclose(g["cosine"][:3], [1, 0, -1])
    np.testing.assert_allclose(g["angle_degrees"][:3], [0, 90, 180])
    np.testing.assert_array_equal(g["complete"], [True, True, True, True, False])
    np.testing.assert_array_equal(g["comparable"], [True, True, True, False, False])
    assert np.isnan(g["cosine"][3:]).all()
    assert g["left_norm"][0] == 0.5 and g["right_norm"][0] == 1.5
