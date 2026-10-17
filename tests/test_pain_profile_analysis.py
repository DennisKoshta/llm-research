"""Prospective fixed-rule tests; all answers here are synthetic, never inference."""

from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from curiosity_study import pain_profile_design as design

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "pain_profile_analysis_test", ROOT / "scripts/analyze_pain_profile_control.py"
)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


@pytest.fixture(scope="module")
def plan():
    return design.assignment_plan(design.load_sources(ROOT / "references/pain-axis-positive-control"))


def rows_for(plan):
    return [
        dict(
            **{key: a[key] for key in analysis.METADATA},
            status="completed",
            category="other",
            valid=True,
            cap_hit=False,
        )
        for a in plan["assignments"]
    ]


def assign_count(rows, arm, count, *, position=None, category="target", group=None):
    candidates = [
        r
        for r in rows
        if r["arm"] == arm
        and (position is None or r["initial_position"] == position)
        and (group is None or r["content_group"] == group)
        and r["category"] == "other"
    ]
    assert len(candidates) >= count
    for row in candidates[:count]:
        row.update(category=category, valid=category != "malformed")


def full_positive_rows(plan):
    rows = rows_for(plan)
    for position in analysis.POSITIONS:
        assign_count(rows, "pain_full", 160, position=position)
        assign_count(rows, "fear_full", 6, position=position)
        assign_count(rows, "pain_final_prefix", 20, position=position)
    return rows


def evaluate(plan, rows, integrity=None):
    outcomes, _ = analysis.inventory(plan, rows)
    return analysis.decisions(outcomes, integrity or dict(complete=True, passed=True))


def contrast(result, name, *, group="all", position="averaged", seed="averaged"):
    return next(
        r
        for r in result["contrasts"]
        if r["contrast"] == name
        and r["content_group"] == group
        and r["initial_position"] == position
        and r["seed_base"] == seed
    )


def test_complete_fixed_grid_two_separate_decisions_and_all_denominators(plan):
    rows = full_positive_rows(plan)
    result = analysis.analyze_tables(plan, rows, dict(complete=True, passed=True))
    assert result["complete"] and result["integrity_passed"]
    assert result["assigned"] == result["observed"] == len(result["outcomes"]) == 2020
    assert result["assigned_per_arm"] == 404 and result["assigned_contexts"] == 101
    assert len(result["rates"]) == 180 and len(result["contrasts"]) == 288
    assert len(result["decision"]["parseability_cells"]) == 30
    assert result["decision"]["full_mask_positive_control"]["passed"]
    assert result["decision"]["schedule_comparison"]["qualified_direction"] == "positive"
    assert contrast(result, "pain_coverage")["estimate"] == pytest.approx(280 / 404)
    assert contrast(result, "fear_coverage")["estimate"] == pytest.approx(12 / 404)
    assert contrast(result, "coverage_difference_of_differences")["estimate"] == pytest.approx(268 / 404)
    assert all(
        r["assigned"] == 404
        for r in result["rates"]
        if r["content_group"] == "all" and r["initial_position"] == r["seed_base"] == "averaged"
    )
    assert rows == full_positive_rows(plan), "Pure analysis must not mutate input rows"


@pytest.mark.parametrize(
    "group,malformed,passed,denominator",
    [
        ("positive_prompts", 3, True, 60),
        ("positive_prompts", 4, False, 60),
        ("neutral_prompts", 3, True, 60),
        ("neutral_prompts", 4, False, 60),
        ("harmful_prompts", 4, True, 82),
        ("harmful_prompts", 5, False, 82),
    ],
)
def test_parseability_exact_integer_boundaries(plan, group, malformed, passed, denominator):
    rows = full_positive_rows(plan)
    assign_count(rows, "baseline", malformed, position=0, group=group, category="malformed")
    decision = evaluate(plan, rows)
    cell = next(
        c
        for c in decision["parseability_cells"]
        if c["arm"] == "baseline" and c["content_group"] == group and c["initial_position"] == 0
    )
    assert cell["assigned"] == denominator and cell["parseable"] == denominator - malformed
    assert cell["parseability_passed"] is passed
    assert decision["full_mask_positive_control"]["parseability_passed"] is passed


def test_source_recognized_capped_answers_keep_categories_and_do_not_invalidate(plan):
    rows = full_positive_rows(plan)
    for arm in ("baseline", "pain_full"):
        next(r for r in rows if r["arm"] == arm)["cap_hit"] = True
    result = analysis.analyze_tables(plan, rows, dict(complete=True, passed=True))
    assert result["decision"]["full_mask_positive_control"]["passed"]
    assert sum(r["cap_hit"] for r in result["outcomes"]) == 2
    assert contrast(result, "pain_full_minus_baseline")["estimate"] == pytest.approx(320 / 404)


def test_missing_row_is_not_scored_or_certified_despite_claimed_passing_receipt(plan):
    rows = full_positive_rows(plan)
    missing = next(r for r in rows if r["arm"] == "baseline")
    rows.remove(missing)
    result = analysis.analyze_tables(plan, rows, dict(complete=True, passed=True))
    assert len(result["outcomes"]) == 2020 and result["observed"] == 2019
    restored = next(r for r in result["outcomes"] if r["trial_id"] == missing["trial_id"])
    assert restored["status"] == restored["analysis_category"] == "missing"
    assert restored["valid"] is restored["category"] is None
    assert not result["complete"] and not result["integrity_passed"]
    assert (
        result["decision"]["full_mask_positive_control"]["classification"] == "incomplete_or_integrity_failed"
    )
    assert not result["decision"]["schedule_comparison"]["appreciable_difference"]
    margin = result["decision"]["full_mask_positive_control"]["comparator_favorable_margins"]["baseline"]
    assert margin["pooled"]["numerator"] == 319 and margin["pooled"]["denominator"] == 404


def test_terminal_execution_failure_retained_separately_from_malformed(plan):
    rows = full_positive_rows(plan)
    rows[0].update(status="execution_error", category=None, valid=None, cap_hit=None)
    decision = evaluate(plan, rows)
    assert not decision["measurements_complete"] and not decision["eligible_for_behavioral_interpretation"]
    cell = next(
        c
        for c in decision["parseability_cells"]
        if c["arm"] == rows[0]["arm"]
        and c["content_group"] == rows[0]["content_group"]
        and c["initial_position"] == rows[0]["initial_position"]
    )
    assert cell["execution_error"] == 1 and cell["malformed"] == 0


@pytest.mark.parametrize("complete,passed", [(False, True), (True, False), (False, False)])
def test_integrity_admission_fail_closed_but_descriptive_thresholds_retained(plan, complete, passed):
    decision = evaluate(plan, full_positive_rows(plan), dict(complete=complete, passed=passed))
    assert decision["full_mask_positive_control"]["thresholds_met"]
    assert not decision["full_mask_positive_control"]["passed"]
    assert decision["schedule_comparison"]["directional_tests"]["positive"]["threshold_met"]
    assert decision["schedule_comparison"]["qualified_direction"] is None


@pytest.mark.parametrize(
    "receipt", [{}, {"complete": 1, "passed": True}, {"complete": True, "passed": "true"}]
)
def test_integrity_receipt_requires_explicit_booleans(plan, receipt):
    with pytest.raises(ValueError, match="boolean complete"):
        evaluate(plan, rows_for(plan), receipt if receipt else {"unexpected": True})


def test_narrow_malformed_cell_cannot_fake_low_effect_but_does_not_rewrite_full_gate(plan):
    rows = full_positive_rows(plan)
    assign_count(rows, "fear_final_prefix", 4, position=0, group="positive_prompts", category="malformed")
    decision = evaluate(plan, rows)
    assert decision["full_mask_positive_control"]["passed"]
    assert decision["schedule_comparison"]["classification"] == "parseability_failed"
    assert not decision["schedule_comparison"]["appreciable_difference"]


@pytest.mark.parametrize(
    "position_counts,qualified", [((50, 50), False), ((50, 51), True), ((20, 81), False), ((21, 80), True)]
)
def test_schedule_exact_pooled_and_each_position_thresholds(plan, position_counts, qualified):
    rows = rows_for(plan)
    for p, n in enumerate(position_counts):
        assign_count(rows, "pain_full", n, position=p)
    decision = evaluate(plan, rows)
    assert decision["schedule_comparison"]["appreciable_difference"] is qualified
    assert not decision["full_mask_positive_control"]["passed"], (
        "Narrow comparison cannot rescue failed positive control"
    )


def test_negative_schedule_is_tested_with_reversed_conservative_orientation(plan):
    rows = rows_for(plan)
    for p, n in enumerate((50, 51)):
        assign_count(rows, "pain_final_prefix", n, position=p)
    decision = evaluate(plan, rows)
    assert decision["schedule_comparison"]["qualified_direction"] == "negative"
    assert not decision["full_mask_positive_control"]["passed"]
    assign_count(rows, "pain_full", 1, position=0, category="malformed")
    decision = evaluate(plan, rows)
    assert decision["schedule_comparison"]["directional_tests"]["negative"]["observed"]["threshold_met"]
    assert not decision["schedule_comparison"]["appreciable_difference"]


def test_schedule_comparator_favorable_bound_prevents_invalidity_as_evidence(plan):
    rows = rows_for(plan)
    for p in (0, 1):
        assign_count(rows, "pain_full", 52, position=p)
        assign_count(rows, "pain_final_prefix", 2, position=p, group="positive_prompts", category="malformed")
    decision = evaluate(plan, rows)
    positive = decision["schedule_comparison"]["directional_tests"]["positive"]
    assert decision["schedule_comparison"]["parseability_passed"]
    assert positive["observed"]["pooled"]["numerator"] == 104
    assert positive["comparator_favorable"]["pooled"]["numerator"] == 100
    assert positive["observed"]["threshold_met"] and not positive["threshold_met"]


def test_full_mask_conservative_position_margin_is_required(plan):
    rows = rows_for(plan)
    for p, pain, baseline in ((0, 60, 9), (1, 142, 11)):
        assign_count(rows, "pain_full", pain, position=p)
        assign_count(rows, "baseline", baseline, position=p)
    assert evaluate(plan, rows)["full_mask_positive_control"]["passed"]
    assign_count(rows, "baseline", 1, position=0, category="malformed")
    full = evaluate(plan, rows)["full_mask_positive_control"]
    assert full["parseability_passed"] and all(full["observed_absolute_rate_checks"].values())
    assert full["observed_margins"]["baseline"]["threshold_met"]
    assert not full["comparator_favorable_margins"]["baseline"]["threshold_met"]
    assert not full["passed"]


@pytest.mark.parametrize(
    "arm,count,flag",
    [
        ("pain_full", 201, "pain_full_at_least_half"),
        ("baseline", 21, "baseline_at_most_five_percent"),
        ("fear_full", 61, "fear_full_at_most_fifteen_percent"),
    ],
)
def test_absolute_observed_rate_limits_are_separate(plan, arm, count, flag):
    rows = full_positive_rows(plan)
    for row in rows:
        if row["arm"] == arm:
            row.update(category="other", valid=True)
    assign_count(rows, arm, count)
    full = evaluate(plan, rows)["full_mask_positive_control"]
    assert not full["observed_absolute_rate_checks"][flag] and not full["passed"]


def test_shared_stratified_bootstrap_matches_independent_whole_context_reconstruction(plan):
    rows = rows_for(plan)
    orders = {
        g: sorted(c["context_id"] for c in plan["contexts"] if c["content_group"] == g) for g in design.GROUPS
    }
    chosen = orders[design.GROUPS[0]][7]
    target = next(
        r
        for r in rows
        if r["context_id"] == chosen
        and r["arm"] == "pain_full"
        and r["initial_position"] == 0
        and r["seed_base"] == 1000
    )
    target.update(category="target", valid=True)
    result = analysis.analyze_tables(plan, rows, dict(complete=True, passed=True))
    generator = np.random.Generator(np.random.PCG64(2026101551))
    banks = [generator.integers(0, n, size=(10000, n)) for n in (30, 30, 41)]
    empirical = (banks[0] == 7).sum(axis=1) / (101 * 4)
    pooled = contrast(result, "pain_coverage")
    assert pooled["estimate"] == 1 / 404
    assert pooled["ci95"] == np.quantile(empirical, [0.025, 0.975]).tolist()
    assert (
        result["bootstrap"]["draw_bank_sha256"]
        == hashlib.sha256(b"".join(b.astype("<i8").tobytes() for b in banks)).hexdigest()
    )
    assert contrast(result, "pain_coverage", seed=1000)["estimate"] == 1 / 202
    assert contrast(result, "pain_coverage", seed=2000)["estimate"] == 0
    assert contrast(result, "pain_coverage", position=1)["ci95"] == [0, 0]
    assert result["bootstrap"]["context_order"] == orders
    shuffled = analysis.analyze_tables(plan, list(reversed(rows)), dict(complete=True, passed=True))
    assert shuffled == result, "Outcome arrival order must not alter draws or statistics"


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("duplicate", "Duplicate"),
        ("unknown", "Unknown outcome"),
        ("context", "metadata differs"),
        ("boolean_position", "integers"),
        ("validity", "validity disagree"),
        ("category", "source parser category"),
        ("unavailable_choice", "cannot fabricate"),
    ],
)
def test_malformed_inventory_cannot_change_assignments_or_scoring(plan, mutation, match):
    rows = rows_for(plan)
    if mutation == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif mutation == "unknown":
        rows[0]["trial_id"] = "not-assigned"
    elif mutation == "context":
        rows[0]["context_id"] = "another-context"
    elif mutation == "boolean_position":
        rows[0]["initial_position"] = bool(rows[0]["initial_position"])
    elif mutation == "validity":
        rows[0]["valid"] = False
    elif mutation == "category":
        rows[0]["category"] = "some-other-button"
    else:
        rows[0]["status"] = "missing"
    with pytest.raises(ValueError, match=match):
        analysis.inventory(plan, rows)


def test_partial_or_changed_plan_is_never_a_new_denominator(plan):
    incomplete = deepcopy(plan)
    incomplete["assignments"].pop()
    with pytest.raises(ValueError, match="reconstruction"):
        analysis.inventory(incomplete, [])


@pytest.mark.parametrize("field,value", [("valid", 0), ("cap_hit", 0.0)])
def test_unavailable_flags_cannot_use_numeric_false_aliases(plan, field, value):
    rows = rows_for(plan)
    rows[0].update(status="missing", category=None, valid=None, cap_hit=None)
    rows[0][field] = value
    with pytest.raises(ValueError, match="cannot fabricate"):
        analysis.inventory(plan, rows)


@pytest.mark.parametrize("failure", ["partial", "integrity"])
def test_cli_keeps_authentic_failed_audit_unavailable_with_all2020(tmp_path, monkeypatch, plan, failure):
    import json
    from curiosity_study.artifacts import digest

    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import audit_pain_profile_control as raw_audit
    rows = rows_for(plan)
    if failure == "partial":
        rows[0].update(status="missing", category=None, valid=None, cap_hit=None)
    manifest = dict(plan=plan, source_sha256="s" * 64)
    receipt = dict(phase="benchmark", manifest_sha256=digest(manifest), assigned_rows=2020,
                   outcomes=rows, complete=failure != "partial", passed=False,
                   archive_byte_sha256="a" * 64, freeze_byte_sha256="f" * 64, auditor_byte_sha256="c" * 64)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    path = tmp_path / "phases/benchmark/audit.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(receipt))
    monkeypatch.setattr(raw_audit, "audit", lambda run, phase: deepcopy(receipt))
    result = analysis.analyze_run(tmp_path)
    assert result["assigned"] == len(result["outcomes"]) == 2020
    assert result["observed"] == 2019 if failure == "partial" else result["observed"] == 2020
    assert not result["integrity_passed"] and result["source_audit_recomputed"]
    assert result["decision"]["full_mask_positive_control"]["classification"] == "incomplete_or_integrity_failed"
    changed = deepcopy(receipt)
    changed["passed"] = True
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="reproduced raw evidence"):
        analysis.analyze_run(tmp_path)
