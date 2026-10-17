import copy
import math

import pytest

from curiosity_study.noninferiority import DEFAULT_ALPHA, exact_upper, world_harm_noninferiority


def test_zero_events_never_give_a_degenerate_zero_bound_and_use_six_endpoint_allocation():
    assert DEFAULT_ALPHA == 0.025 / 6
    for n in (1, 10, 108, 120, 10000):
        assert exact_upper(0, n) == pytest.approx(1 - DEFAULT_ALPHA ** (1 / n), abs=1e-15)
        assert exact_upper(0, n) > 0
    assert exact_upper(0, 108) == pytest.approx(0.0494805522, abs=1e-10)
    assert exact_upper(0, 120) == pytest.approx(0.0446447241, abs=1e-10)


def test_nonzero_analytic_cases_and_all_harmed_boundary():
    for alpha in (0.05, DEFAULT_ALPHA, 1e-9):
        assert exact_upper(1, 2, alpha) == pytest.approx(math.sqrt(1 - alpha), abs=2e-15)
        assert exact_upper(9, 10, alpha) == pytest.approx((1 - alpha) ** 0.1, abs=2e-15)
        assert exact_upper(10, 10, alpha) == 1
    assert exact_upper(1, 3, 0.5) == pytest.approx(0.5, abs=1e-15)
    assert exact_upper(1, 3, 0.15625) == pytest.approx(0.75, abs=1e-15)


def test_log_inversion_matches_independent_binomial_cdf_and_survives_underflow():
    for k, n, alpha in ((2, 20, 0.05), (7, 30, DEFAULT_ALPHA), (100, 1000, 1e-12)):
        p = exact_upper(k, n, alpha)
        cdf = math.fsum(math.comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(k + 1))
        assert cdf == pytest.approx(alpha, rel=2e-10)
    # Direct (1-p)**n evaluation is unusable during inversion near p=.5.
    p = exact_upper(1, 10000, 1e-100)
    log_cdf = 9999 * math.log1p(-p) + math.log(1 - p + 10000 * p)
    assert log_cdf == pytest.approx(math.log(1e-100), abs=1e-10)


def test_bound_monotonicity():
    assert all(exact_upper(k, 20) < exact_upper(k + 1, 20) for k in range(20))
    assert exact_upper(2, 100) < exact_upper(2, 50)
    assert exact_upper(2, 100, 0.001) > exact_upper(2, 100, 0.05)


@pytest.mark.parametrize(
    "events,worlds,alpha",
    [
        (0, 0, 0.05),
        (-1, 2, 0.05),
        (3, 2, 0.05),
        (True, 2, 0.05),
        (0, True, 0.05),
        (1.0, 2, 0.05),
        (0, 2, True),
        (0, 2, 0),
        (0, 2, 1),
        (0, 2, float("nan")),
    ],
)
def test_invalid_counts_and_alpha_rejected(events, worlds, alpha):
    with pytest.raises(ValueError):
        exact_upper(events, worlds, alpha)


def fixture(n=120, kinds=("memory",)):
    assignments, rows = [], []
    for index in range(n):
        for kind in kinds:
            for template in ("table", "prose"):
                for cell in range(2):
                    a = dict(
                        trial_id=f"w{index}-{kind}-{template}-c{cell}",
                        task_revision="synthetic-v1",
                        family="objects",
                        seed=17,
                        index=index,
                        kind=kind,
                        template=template,
                    )
                    assignments.append(a)
                    for condition in ("baseline", "zero", "primary", "extra"):
                        rows.append(
                            dict(
                                trial_id=f"{a['trial_id']}-{condition}",
                                prefix_id=a["trial_id"],
                                kind=kind,
                                condition=condition,
                                valid=True,
                                correct=True,
                            )
                        )
    return assignments, rows


def calculate(assignments, rows, **kwargs):
    defaults = dict(
        control_selector={"condition": "baseline"},
        treatment_selectors=[{"condition": "zero"}, {"condition": "primary"}],
        success_field="correct",
    )
    return world_harm_noninferiority(assignments, rows, **(defaults | kwargs))


def test_cells_templates_and_multiple_arms_are_one_world_not_independent_trials():
    aa, rows = fixture()
    for row in rows:
        if row["prefix_id"].startswith("w0-") and row["condition"] in ("zero", "primary"):
            row.update(correct=False)
    result = calculate(aa, rows)
    assert result["assigned_sources"] == 480
    assert result["assigned_worlds"] == result["defined_worlds"] == 120
    assert result["new_harm_worlds"] == 1
    assert result["expected_treated_measurements"] == 960
    assert result["new_harm_probability_upper_bound"] == pytest.approx(
        1 / 120 + math.sqrt(-math.log(DEFAULT_ALPHA) / 240)
    )
    assert result["binomial_clopper_pearson_upper_bound"] == exact_upper(1, 120)
    assert result["bound_method"] == "independent_heterogeneous_hoeffding"
    assert result["complete_selected_coverage"]


def test_observed_baseline_failures_remain_in_denominator_without_invented_new_harm():
    aa, rows = fixture()
    for row in rows:
        if row["prefix_id"].startswith("w0-"):
            row.update(valid=False, correct=False)
    result = calculate(aa, rows)
    assert result["new_harm_worlds"] == 0 and result["assigned_worlds"] == 120
    assert result["defined_worlds"] == 120 and result["noninferiority_passed"]
    assert result["bound_method"] == "independent_heterogeneous_zero_event_am_gm"
    assert result["new_harm_probability_upper_bound"] == exact_upper(0, 120)
    assert result["comparison"] == "upper_bound <= margin"
    assert not calculate(aa, rows, margin=exact_upper(0, 120) - 1e-12)["noninferiority_passed"]
    assert calculate(aa, rows, margin=exact_upper(0, 120))["noninferiority_passed"]


@pytest.mark.parametrize("problem", ["missing_row", "missing_field", "malformed_flag"])
def test_unavailable_controls_make_endpoint_inconclusive_without_dropping_worlds(problem):
    aa, rows = fixture()
    row = next(r for r in rows if r["condition"] == "baseline")
    if problem == "missing_row":
        rows.remove(row)
    elif problem == "missing_field":
        row.pop("correct")
    else:
        row["correct"] = 1
    result = calculate(aa, rows)
    assert result["assigned_worlds"] == 120 and result["defined_worlds"] == 119
    assert result["unavailable_controls"] == 1
    assert result["new_harm_probability_upper_bound"] is None
    assert not result["noninferiority_passed"]
    assert result["status"] == "inconclusive_missing_control"


def test_missing_treatment_conservatively_harms_but_cannot_pass_despite_small_bound():
    aa, rows = fixture(n=1200)
    rows.remove(next(r for r in rows if r["condition"] == "primary"))
    result = calculate(aa, rows)
    assert result["defined_worlds"] == 1200 and result["new_harm_worlds"] == 1
    assert result["unavailable_treated_measurements"] == 1
    assert result["bound_passed"] and not result["noninferiority_passed"]
    assert result["status"] == "inconclusive_incomplete_coverage"


@pytest.mark.parametrize("n", [108, 120])
def test_nonzero_harm_uses_heterogeneous_bound_and_cannot_certify_planned_sizes(n):
    aa, rows = fixture(n=n)
    next(row for row in rows if row["condition"] == "primary")["correct"] = False
    result = calculate(aa, rows)
    assert result["bound_method"] == "independent_heterogeneous_hoeffding"
    assert result["new_harm_probability_upper_bound"] > result["binomial_clopper_pearson_upper_bound"]
    assert not result["noninferiority_passed"]
    for row in rows:
        if row["condition"] == "primary":
            row["correct"] = False
    all_harmed = calculate(aa, rows)
    assert all_harmed["new_harm_worlds"] == n
    assert all_harmed["new_harm_probability_upper_bound"] == 1


def test_validity_endpoint_varies_arm_set_by_kind_and_ignores_unselected_arms():
    aa, rows = fixture(n=120, kinds=("memory", "instrument"))
    for row in rows:
        if row["condition"] == "extra" and row["kind"] == "memory":
            row.update(valid=False, correct=False)
    selected = [{"condition": "primary"}, {"condition": "zero"}, {"condition": "extra", "kind": "instrument"}]
    result = calculate(aa, rows, success_field="valid", treatment_selectors=selected)
    assert result["new_harm_worlds"] == 0 and result["noninferiority_passed"]
    changed = next(r for r in rows if r["condition"] == "extra" and r["kind"] == "instrument")
    changed.update(valid=False, correct=False)
    result = calculate(aa, rows, success_field="valid", treatment_selectors=selected)
    assert result["new_harm_worlds"] == 1


@pytest.mark.parametrize(
    "problem",
    [
        "empty_assignments",
        "empty_control",
        "empty_treatments",
        "duplicate_row",
        "overlapping_selectors",
        "different_family",
        "boolean_index",
    ],
)
def test_ambiguous_or_unassigned_design_fails_closed(problem):
    aa, rows = fixture(n=1)
    options = {}
    if problem == "empty_assignments":
        aa, rows = [], []
    elif problem == "empty_control":
        options["control_selector"] = {}
    elif problem == "empty_treatments":
        options["treatment_selectors"] = []
    elif problem == "duplicate_row":
        rows.append(copy.deepcopy(rows[0]))
    elif problem == "overlapping_selectors":
        options["treatment_selectors"] = [
            {"condition": "primary"},
            {"condition": "primary", "kind": "memory"},
        ]
    elif problem == "different_family":
        aa[0]["family"] = "machines"
    else:
        aa[0]["index"] = False
    with pytest.raises(ValueError):
        calculate(aa, rows, **options)
