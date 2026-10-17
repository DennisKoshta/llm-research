"""Independent decision-checker cross-checks using complete synthetic site grids."""
import json
from pathlib import Path
import random
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import check_satisfaction_site_decision as checker  # noqa: E402
import test_satisfaction_causal_sites_analysis as fixture  # noqa: E402
from curiosity_study.artifacts import digest  # noqa: E402


def compare(plan, rows, top=None, per_site=None):
    defaults, sites = fixture.checks(plan)
    top = defaults if top is None else top
    per_site = sites if per_site is None else per_site
    actual = fixture.run(plan, rows, top=top, per_site=per_site)
    expected = checker.recompute(plan, rows, checks=top, site_checks=per_site)
    assert checker.compare(expected["decision"], actual["decision"]) > 0
    assert checker.compare(expected["task_readiness"], actual["task_readiness"]) > 0
    return expected, actual


@pytest.mark.parametrize("phase", ["calibration", "validation"])
def test_null_and_exact_margin_boundary_match_full_analysis(phase):
    expected, actual = compare(*fixture.tables(phase))
    assert not expected["decision"]["positive_screen"]
    assert actual["selected_site"] is None
    expected, actual = compare(*fixture.boundary(phase))
    assert expected["decision"]["positive_screen"]
    for site in actual["site_decisions"].values():
        assert all(value == .25 for value in site["primary_observed"]["panel_differences"].values())
        assert all(all(v == .125 for v in control["observed"]["panel_differences"].values())
                   for control in site["controls"].values())


def test_earliest_not_largest_site_and_local_readiness_selection():
    plan, rows = fixture.boundary()
    fixture.choose(rows, site=47, worlds=range(12))
    expected, _ = compare(plan, rows)
    assert expected["decision"]["selected_site"] == 31
    fixture.invalidate(fixture.getrow(rows, site=31, kind="new_fact"))
    expected, _ = compare(plan, rows)
    assert expected["decision"]["selected_site"] == 47
    assert expected["decision"]["site_decisions"]["31"]["classification"] == "readiness_failed"


@pytest.mark.parametrize("gate", checker.GATES)
def test_nonselected_site_engineering_failure_still_vetoes_whole_grid(gate):
    plan, rows = fixture.boundary()
    top, sites = fixture.checks(plan)
    sites["47"][gate] = False
    expected, _ = compare(plan, rows, top, sites)
    assert expected["decision"]["classification"] == "global_integrity_or_engineering_failed"
    assert expected["decision"]["selected_site"] is None


@pytest.mark.parametrize("comparator", ["zero_sham", "confidence_positive", "random_01"])
def test_allowed_invalid_comparator_cannot_inflate_primary_or_selectivity(comparator):
    plan, rows = fixture.boundary()
    fixture.invalidate(fixture.getrow(rows, arm=comparator))
    expected, actual = compare(plan, rows)
    site = expected["decision"]["site_decisions"]["31"]
    assert expected["task_readiness"]["31"]["passed"]
    assert not site["positive_screen"] and actual["selected_site"] == 47
    summary = site["primary_comparator_favorable"] if comparator == "zero_sham" else site["controls"][comparator]["comparator_favorable"]
    #One bad menu among24 assigned choices costs exactly1/24, independently computed.
    assert summary["panel_differences"]["panel_1"] == (5/24 if comparator == "zero_sham" else 2/24)


@pytest.mark.parametrize("phase,site", [("calibration", 31), ("validation", 31), ("validation", 47)])
def test_assigned_versus_selected_site_semantics_and_failed_audit_readiness(phase, site):
    plan, rows = fixture.boundary(phase, site)
    top, sites = fixture.checks(plan)
    if phase == "calibration":
        sites["31"]["readiness"]["passed"] = False
        sites["47"]["readiness"]["passed"] = False
    else:
        sites[str(site)]["readiness"]["passed"] = False
    expected, actual = compare(plan, rows, top, sites)
    assert not expected["decision"]["positive_screen"] and expected["decision"]["selected_site"] is None
    assert expected["decision"]["assigned_site"] == (None if phase == "calibration" else site)
    assert all(value["passed"] for value in actual["task_readiness"].values())


@pytest.mark.parametrize("status,cap", [("completed", False), ("completed", True), ("execution_error", False), ("missing", False)])
def test_failure_dispositions_never_become_valid_stop(status, cap):
    plan, rows = fixture.boundary("validation")
    fixture.invalidate(fixture.getrow(rows, panel="half_1", world=0), status=status, capped=cap)
    expected, _ = compare(plan, rows)
    assert not expected["decision"]["positive_screen"]
    assert expected["decision"]["site_decisions"]["31"]["primary_observed"]["panel_differences"]["half_1"] == 3/16


def test_complete_row_order_does_not_change_exact_rational_decision():
    plan, rows = fixture.boundary("validation")
    top, sites = fixture.checks(plan)
    original = checker.recompute(plan, rows, checks=top, site_checks=sites)
    random.Random(622).shuffle(rows)
    assert checker.recompute(plan, rows, checks=top, site_checks=sites) == original


def persisted(tmp_path):
    run, audit_path, manifest, audit = fixture.audit_fixture(tmp_path, readiness=True)
    analysis = fixture.a.analyze_audit(manifest, audit)
    analysis_path = tmp_path/"analysis.json"
    analysis_path.write_text(json.dumps(analysis))
    return run, analysis_path, audit_path, analysis, audit


def test_artifact_check_preserves_verified_actual_decision_prose_for_certificate(tmp_path):
    run, analysis_path, audit_path, analysis, _ = persisted(tmp_path)
    analysis["decision"]["interpretation"] = "Additional scope wording must not break a verified scientific projection."
    analysis_path.write_text(json.dumps(analysis))
    receipt = checker.check(run, analysis_path, audit_path)
    assert receipt["passed"] and receipt["decision"] == analysis["decision"]
    assert receipt["analysis_sha256"] == digest(analysis)
    assert "interpretation" not in receipt["recomputed_decision"]
    assert "descriptive bootstrap intervals" in receipt["excluded_scope"]


@pytest.mark.parametrize("change", ["selected_site", "positive", "effect", "bound", "readiness", "counts", "gate", "audit"])
def test_tampered_scientific_decision_readiness_or_binding_rejects(tmp_path, change):
    run, analysis_path, audit_path, analysis, audit = persisted(tmp_path)
    site = analysis["decision"]["site_decisions"]["31"]
    if change == "selected_site":
        analysis["decision"]["selected_site"] = 31
    elif change == "positive":
        analysis["decision"]["positive_screen"] = False
    elif change == "effect":
        site["primary_observed"]["panel_differences"]["half_1"] += .01
    elif change == "bound":
        site["controls"]["random_05"]["comparator_favorable"]["pooled_menu_differences"]["0"] += .01
    elif change == "readiness":
        analysis["task_readiness"]["31"]["passed"] = False
    elif change == "counts":
        analysis["task_readiness"]["31"]["diagnostic_cells"][0]["valid"] -= 1
    elif change == "gate":
        analysis["site_checks"]["31"]["engagement_passed"] = False
    else:
        audit["outcomes"][0]["valid"] = False
        audit_path.write_text(json.dumps(audit))
    analysis_path.write_text(json.dumps(analysis))
    with pytest.raises(ValueError):
        checker.check(run, analysis_path, audit_path)


def test_each12_control_is_retained_even_last_random():
    plan, rows = fixture.boundary("validation")
    fixture.choose(rows, arm="random_05", worlds=range(2))
    expected, _ = compare(plan, rows)
    site = expected["decision"]["site_decisions"]["31"]
    assert site["effect_threshold_met"] and not site["specificity_threshold_met"]
    assert set(site["controls"]) == set(checker.VALIDATION_ARMS[3:])


def test_single_pooled_menu_veto_and_whole16_readiness_not_half_readiness():
    plan, rows = fixture.tables("validation")
    fixture.choose(rows, worlds=range(8), menus=[0])
    fixture.getrow(rows, panel="half_1", world=0, kind="new_fact")["correct"] = False
    expected, _ = compare(plan, rows)
    assert expected["task_readiness"]["31"]["passed"]
    assert not expected["decision"]["site_decisions"]["31"]["primary_observed_threshold_met"]


@pytest.mark.parametrize("change", ["unknown_service", "false_closure", "diagnostic_correct_type",
                                   "wrong_variant", "base_alias", "ordinal", "source_slot"])
def test_compact_schema_and_complete_world_inventory_fail_closed(change):
    plan, rows = fixture.tables("validation")
    if change == "unknown_service":
        rows[0]["selected_service"] = "invented action"
    elif change == "false_closure":
        rows[0]["thinking_closed"] = False
    elif change == "diagnostic_correct_type":
        next(r for r in rows if r["kind"] == "new_fact")["correct"] = 1
    else:
        for ar, row in zip(plan["assignments"], rows, strict=True):
            if change == "wrong_variant" and ar["kind"] == "new_fact":
                ar["delivery"] = row["delivery"] = "independent_novelty"
            elif change == "base_alias" and ar["kind"] == "new_fact":
                ar["base_trial_id"] = row["base_trial_id"] = "same-base-across-worlds"
            elif change == "ordinal":
                ar["reserved_ordinal"] += 1
                row["reserved_ordinal"] += 1
            elif change == "source_slot" and ar["world_slot"] == 11:
                ar["world_slot"] = row["world_slot"] = 3
        plan["assignment_sha256"] = digest(plan["assignments"])
    top, sites = fixture.checks(plan)
    with pytest.raises(ValueError):
        checker.recompute(plan, rows, checks=top, site_checks=sites)
