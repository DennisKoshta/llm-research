"""Synthetic finite-screen arithmetic; no experiment outputs or inference."""

from copy import deepcopy
import importlib.util
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest

PATH = Path(__file__).parents[1] / "scripts/analyze_curiosity_coverage_bridge.py"
spec = importlib.util.spec_from_file_location("coverage_analysis", PATH)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def tables():
    assigned, outcomes = [], []
    branches = [("choice", d, m) for d in a.DELIVERIES for m in (0, 1)] + [
        (k, "decisive_evidence" if k in ("identify_resolved", "repeated_fact") else "redundant_evidence", 0)
        for k in a.DIAGNOSTICS]
    for p in a.PANELS:
        for w in range(12):
            for k, d, m in branches:
                base = f"{p}/{w}/{k}/{d}/{m}"
                for arm in a.ARMS:
                    schedule, direction = ("baseline", None) if arm == "baseline" else arm.split("__")
                    r = dict(trial_id=f"{arm}:{base}", base_trial_id=base, panel_id=p, semantic_world_id=a.digest((p, w)),
                             world_slot=w, arm=arm, schedule=schedule, direction_key=direction,
                             family="objects", kind=k, delivery=d, menu_layout=m)
                    assigned.append(r)
                    outcomes.append(dict(**r, status="completed", valid=True, thinking_closed=True, cap_hit=False,
                        correct=None if k == "choice" else True,
                        selected_service=("stop" if d == "decisive_evidence" else "report") if k == "choice" else None))
    return assigned, outcomes


def choose(rows, direction=a.PRIMARY, schedule="full", *, worlds=range(3), menus=(0, 1), panels=a.PANELS, service="stop"):
    for r in rows:
        if r["arm"] == a.arm(direction, schedule) and r["kind"] == "choice" and r["delivery"] == "redundant_evidence" \
                and r["world_slot"] in worlds and r["menu_layout"] in menus and r["panel_id"] in panels:
            r["selected_service"] = service


def boundary():
    assigned, outcomes = tables()
    choose(outcomes)
    for direction in a.CONTROLS:
        choose(outcomes, direction, worlds=(0,))
        choose(outcomes, direction, worlds=(1,), menus=(0,))
    return assigned, outcomes


def run(assigned, outcomes, **kwargs):
    return a.analyze_tables(assigned, outcomes, **dict(dict(complete=True, integrity_passed=True,
        checks=dict.fromkeys(a.GATES, True), audit_readiness={"passed": True}), **kwargs))


@pytest.fixture(scope="module")
def passing():
    assigned, outcomes = boundary()
    return assigned, outcomes, run(assigned, outcomes)


def select(rows, arm=a.arm(a.PRIMARY, "full"), *, panel="panel_1", world=5, menu=0, kind="choice"):
    return next(r for r in rows if r["arm"] == arm and r["panel_id"] == panel and r["world_slot"] == world
                and r["menu_layout"] == menu and r["kind"] == kind and r["delivery"] == "redundant_evidence")


def invalidate(r, *, status="completed", cap=False):
    r.update(status=status, valid=False if status == "completed" else None,
             thinking_closed=status == "completed", cap_hit=cap if status == "completed" else None,
             selected_service=None, correct=None if r["kind"] == "choice" else False)


def decide(result, **kwargs):
    return a.decide(result["behavior_contrasts"], result["decision_bounds"], **dict(dict(complete=True, integrity_passed=True,
        checks=dict.fromkeys(a.GATES, True), readiness=result["task_readiness"], audit_readiness={"passed": True}), **kwargs))


def test_full_boundary_all77_contrasts_and_exact_denominators(passing):
    _, _, r = passing
    assert r["decision"]["classification"] == "qualifying_selective_coverage_signal"
    assert r["decision"]["primary_full_effect"]["observed"]["panel_differences"] == dict(panel_1=.25, panel_2=.25, pooled=.25)
    assert r["decision"]["primary_coverage_interaction"]["observed"]["panel_differences"] == dict(panel_1=.25, panel_2=.25, pooled=.25)
    for c in a.CONTROLS:
        for field in ("full_controls", "interaction_controls"):
            assert r["decision"][field][c]["observed"]["panel_differences"] == dict(panel_1=.125, panel_2=.125, pooled=.125)
    assert len(r["outcomes"]) == 8352 and len(r["choices"]) == 4176
    assert len(r["contrast_specs"]) == 77 and len(r["behavior_contrasts"]) == 18711
    assert len(r["world_contrasts"]) == 12474 and len(r["decision_bounds"]) == 2079
    assert len(r["denominators"]) == 7047
    readiness = r["task_readiness"]
    assert len(readiness["diagnostic_cells"]) == 348 and len(readiness["choice_cells"]) == 174
    assert len(readiness["arm_diagnostics"]) == 29
    for arm in a.ARMS:
        counts = [v for v in r["denominators"] if v["arm"] == arm and v["panel_id"] == "pooled"
                  and v["menu_layout"] == "averaged" and v["exclusive_category"]]
        assert sum(v["count"] for v in counts) == 144
        assert all(v["assigned_choices"] == 48 for v in counts)
    assert not r["scientific_admission"] and not r["causal_admission"]


def test_exact_shared_bank_bootstrap_worlds_not_menus_and_equal_panels(passing):
    _, outcomes, result = passing
    contrast = f"effect:full:{a.PRIMARY}"
    rng = np.random.Generator(np.random.PCG64(2026101552))
    banks = {p: rng.integers(0, 12, (10000, 12)) for p in a.PANELS}
    assert result["bootstrap"]["draw_bank_sha256"] == a.digest({p: banks[p].tolist() for p in a.PANELS})
    world_values = {}
    for p in a.PANELS:
        worlds = sorted({r["semantic_world_id"] for r in outcomes if r["panel_id"] == p})
        world_values[p] = np.array([np.mean([int(r["selected_service"] == "stop") for r in outcomes
            if r["panel_id"] == p and r["semantic_world_id"] == w and r["kind"] == "choice"
            and r["arm"] == a.arm(a.PRIMARY, "full") and r["delivery"] == "redundant_evidence"]) for w in worlds])
        assert result["bootstrap"]["world_order"][p] == worlds
    draws = {p: world_values[p][banks[p]].mean(axis=1) for p in a.PANELS}
    draws["pooled"] = (draws["panel_1"] + draws["panel_2"]) / 2
    for p in (*a.PANELS, "pooled"):
        r = next(r for r in result["behavior_contrasts"] if r["contrast"] == contrast and r["delivery"] == "redundant_evidence"
                 and r["menu_layout"] == "averaged" and r["category"] == a.NONREPORT and r["panel_id"] == p)
        assert r["ci95"] == np.quantile(draws[p], [.025, .975]).tolist()
    zero = next(r for r in result["behavior_contrasts"] if r["contrast"] == "baseline_minus_zero:record")
    assert zero["estimate"] == 0 and zero["ci95"] == [0, 0] and zero["empirical_interval_degenerate"]


def test_permutation_invariant_all_artifact_arithmetic(passing):
    assigned, outcomes, expected = passing
    assigned, outcomes = list(assigned), list(outcomes)
    random.Random(931).shuffle(assigned)
    random.Random(276).shuffle(outcomes)
    assert run(assigned, outcomes) == expected


def test_one_menu_average_is_half_effect_and_no_pooled_menu_rescue():
    assigned, outcomes = tables()
    choose(outcomes, worlds=range(6), menus=(0,))
    r = run(assigned, outcomes)
    primary = r["decision"]["primary_full_effect"]
    assert primary["observed"]["panel_differences"]["panel_1"] == .25
    assert primary["observed"]["pooled_menu_differences"] == {"0": .5, "1": 0}
    assert not primary["passed"]
    assert r["decision"]["classification"] == "no_substantial_full_curiosity_effect"


@pytest.mark.parametrize("kind", ["repeat", "pattern", "stop"])
def test_valid_nonreport_uses_only_genuine_services(kind):
    r = dict(trial_id="x", status="completed", valid=True, thinking_closed=True, cap_hit=False, selected_service=kind)
    assert a.response_contrast({"x": r}, {"x": 1}, a.NONREPORT) == 1
    for mutation in (dict(valid=False), dict(thinking_closed=False), dict(cap_hit=True), dict(status="missing"), dict(status="execution_error")):
        changed = dict(r, **mutation)
        assert a.response_contrast({"x": changed}, {"x": 1}, a.NONREPORT) == 0
        assert a.response_contrast({"x": changed}, {"x": 1}, a.NONREPORT, bound=True) == 0


def test_shared_shams_cancel_before_invalidity_bounding():
    specs = {r["contrast"]: r["coefficients"] for r in a.contrast_specs()}
    interaction = specs[f"interaction:{a.PRIMARY}"]
    selective = specs[f"interaction_selectivity:{a.CONTROLS[0]}"]
    assert a.arm("zero_sham", "full") in interaction and a.arm("zero_sham", "record") in interaction
    assert all("zero_sham" not in key for key in selective) and len(selective) == 4
    rows = {arm: dict(trial_id=arm, status="completed", valid=True, thinking_closed=True, cap_hit=False, selected_service="report")
            for arm in a.ARMS}
    for schedule in a.SCHEDULES:
        rows[a.arm("zero_sham", schedule)].update(valid=False, selected_service=None)
    assert a.response_contrast(rows, interaction, a.NONREPORT, bound=True) == -1
    assert a.response_contrast(rows, selective, a.NONREPORT, bound=True) == 0
    # The same underlying response, when it is truly shared, cancels before
    # invalidity. Distinct schedule-sham IDs above deliberately do not.
    same = dict(trial_id="shared", status="missing", valid=None, thinking_closed=None, cap_hit=None, selected_service=None)
    assert a.response_contrast({"left": same, "right": same}, {"left": 1, "right": -1}, a.NONREPORT, bound=True) == 0


def test_bound_matches_bruteforce_response_level_minimum_and_record_target_penalty():
    rows = {key: dict(trial_id=key, status="completed", valid=False, thinking_closed=True, cap_hit=False, selected_service=None)
            for key in ("cf", "cr", "kf", "kr")}
    coefficients = {"cf": .5, "cr": -.5, "kf": -.5, "kr": .5}
    brute = min(sum(coefficients[k] * value for k, value in zip(rows, values)) for values in itertools.product((0, 1), repeat=4))
    assert a.response_contrast(rows, coefficients, a.NONREPORT, bound=True) == brute == -1
    rows["cf"].update(valid=True, selected_service="stop")
    rows["kf"].update(valid=True, selected_service="report")
    rows["kr"].update(valid=True, selected_service="report")
    assert a.response_contrast(rows, coefficients, a.NONREPORT) == .5
    assert a.response_contrast(rows, coefficients, a.NONREPORT, bound=True) == 0


@pytest.mark.parametrize("control", a.CONTROLS)
def test_each_individual_control_can_veto_no_average_or_control_drop(passing, control):
    r = deepcopy(passing[2])
    name = f"selectivity:full:{control}"
    for row in r["behavior_contrasts"]:
        if row["contrast"] == name:
            row["estimate"] = 0
    decision = decide(r)
    assert decision["full_effect_threshold_met"] and not decision["full_selectivity_threshold_met"]
    assert decision["classification"] == "full_effect_without_required_direction_selectivity"
    assert not decision["full_controls"][control]["observed_threshold_met"]


def test_shared_effect_both_schedules_cannot_pass_selective_coverage(passing):
    r = deepcopy(passing[2])
    for row in r["behavior_contrasts"]:
        if row["contrast"].startswith("interaction:"):
            row["estimate"] = 0
    decision = decide(r)
    assert decision["full_effect_threshold_met"] and decision["full_selectivity_threshold_met"]
    assert decision["classification"] == "selective_full_effect_without_required_selective_coverage_interaction"
    r = deepcopy(passing[2])
    for row in r["decision_bounds"]:
        if row["contrast"] == f"interaction_selectivity:{a.CONTROLS[-1]}" and row["panel_id"] == "panel_2":
            row["lower_bound"] = 0
    assert not decide(r)["selective_coverage_interaction_threshold_met"]


@pytest.mark.parametrize("kwargs,classification", [
    ({"complete": False}, "incomplete_or_integrity_failed"),
    ({"integrity_passed": False}, "incomplete_or_integrity_failed"),
    ({"checks": dict.fromkeys(a.GATES, False)}, "incomplete_or_integrity_failed"),
    ({"audit_readiness": {"passed": False}}, "readiness_failed"),
    ({"readiness": {"passed": False}}, "readiness_failed"),
])
def test_failure_precedence_preserves_all_effect_flags(passing, kwargs, classification):
    decision = decide(passing[2], **kwargs)
    assert decision["classification"] == classification and not decision["positive_screen"]
    assert decision["full_effect_threshold_met"] and decision["full_selectivity_threshold_met"]
    assert decision["selective_coverage_interaction_threshold_met"]


def test_readiness_seven_invalid_across_cells_pass_eight_fails137_rule():
    assigned, rows = tables()
    selected = [r for r in rows if r["arm"] == "baseline" and r["kind"] != "choice"]
    unique = {}
    for r in selected:
        unique.setdefault((r["panel_id"], r["kind"]), r)
    candidates = list(unique.values())
    for r in candidates[:7]:
        invalidate(r)
    cells, _ = a.inventory(assigned, rows)
    readiness = a.task_readiness(cells)
    assert readiness["passed"] and not readiness["original_all_pristine_format_passed"]
    assert next(r for r in readiness["arm_diagnostics"] if r["arm"] == "baseline")["valid"] == 137
    invalidate(candidates[7])
    cells, _ = a.inventory(assigned, rows)
    readiness = a.task_readiness(cells)
    assert all(r["passed"] for r in readiness["diagnostic_cells"])
    assert not readiness["passed"]


@pytest.mark.parametrize("mutation", ["two_invalid_cell", "two_wrong_cell", "one_cap", "missing", "two_invalid_choices"])
def test_local_readiness_is_not_replaced_by_arm_average(mutation):
    assigned, rows = tables()
    kind = "choice" if mutation == "two_invalid_choices" else "new_fact"
    selected = [select(rows, "baseline", kind=kind, world=i) for i in (4, 5)]
    if mutation == "two_wrong_cell":
        for r in selected:
            r["correct"] = False
    else:
        for r in selected[:1 if mutation in ("one_cap", "missing") else 2]:
            invalidate(r, status="missing" if mutation == "missing" else "completed", cap=mutation == "one_cap")
    cells, _ = a.inventory(assigned, rows)
    assert not a.task_readiness(cells)["passed"]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong_schedule", "wrong_world", "bool", "wrong_metadata", "fake_missing"])
def test_inventory_rejects_dropped_or_fabricated_rows(mutation):
    assigned, rows = tables()
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = rows[0]
    elif mutation == "wrong_schedule":
        assigned[0]["schedule"] = rows[0]["schedule"] = "full"
    elif mutation == "wrong_world":
        assigned[0]["semantic_world_id"] = rows[0]["semantic_world_id"] = "newworld"
    elif mutation == "bool":
        rows[0]["valid"] = 1
    elif mutation == "wrong_metadata":
        rows[0]["base_trial_id"] = "other"
    else:
        rows[0]["status"] = "missing"
    with pytest.raises(ValueError):
        a.inventory(assigned, rows)


def test_incomplete_unavailable_slots_still_have_error_mass_and_all_effects():
    assigned, rows = tables()
    target = select(rows)
    invalidate(target, status="missing")
    r = run(assigned, rows, complete=False)
    assert r["decision"]["classification"] == "incomplete_or_integrity_failed"
    assert len(r["outcomes"]) == 8352 and len(r["behavior_contrasts"]) == 18711
    denominator = next(v for v in r["denominators"] if v["arm"] == target["arm"] and v["panel_id"] == target["panel_id"]
                       and v["menu_layout"] == 0 and v["delivery"] == target["delivery"] and v["category"] == "missing")
    assert denominator["count"] == 1 and denominator["assigned_choices"] == 12
    invalid_as_nonreport = next(v for v in r["denominators"] if all(v[k] == denominator[k] for k in
        ("arm", "panel_id", "menu_layout", "delivery")) and v["category"] == a.NONREPORT)
    assert invalid_as_nonreport["count"] == 0


def test_observed_execution_failure_cannot_be_hidden_by_true_audit_flags():
    assigned, rows = tables()
    invalidate(select(rows), status="execution_error")
    r = run(assigned, rows)
    assert r["complete"] is True and r["integrity_passed"] is True
    assert r["all_assigned_responses_completed"] is False
    assert r["decision"]["classification"] == "incomplete_or_integrity_failed"


def test_payload_bindings_and_write_once_cli(tmp_path, passing):
    assigned, outcomes, result = passing
    payload = dict(assignments=assigned, assignment_sha256=a.digest(assigned), outcomes=outcomes,
        analysis_metadata=a.ANALYSIS_METADATA, complete=True, integrity_passed=True,
        checks=dict.fromkeys(a.GATES, True), readiness={"passed": False})
    checked = a.analyze_payload(payload)
    assert checked["input_sha256"] == a.digest(payload) and checked["decision"]["classification"] == "readiness_failed"
    assert checked["behavior_contrasts"] == result["behavior_contrasts"]
    assert not checked["source_artifacts_independently_verified"]
    payload["assignment_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        a.analyze_payload(payload)
    input_path, out = tmp_path / "input.json", tmp_path / "analysis.json"
    input_path.write_text(json.dumps(payload))
    out.write_text("immutable")
    process = subprocess.run([sys.executable, str(PATH), "--input", str(input_path), "--out", str(out)], capture_output=True, text=True)
    assert process.returncode != 0 and "immutable" in process.stderr and out.read_text() == "immutable"
