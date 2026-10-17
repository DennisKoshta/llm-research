"""Synthetic full crossing and independent fixed-direction decision checks."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import random

import numpy as np
import pytest

from curiosity_study.artifacts import digest
from curiosity_study.voluntary_resolution import DELIVERIES, DIAGNOSTICS

SPEC = importlib.util.spec_from_file_location(
    "curiosity_analysis_under_test",
    Path(__file__).parents[1] / "scripts/analyze_curiosity_suppression_screen.py",
)
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)


def tables():
    assigned, rows = [], []
    branches = [("choice", d, m) for d in DELIVERIES for m in (0, 1)]
    branches += [
        (k, "decisive_evidence" if k in ("identify_resolved", "repeated_fact") else "redundant_evidence", 0)
        for k in DIAGNOSTICS
    ]
    for panel in a.PANELS:
        for world in range(12):
            for kind, delivery, menu in branches:
                base = f"{panel}-{world}-{kind}-{delivery}-{menu}"
                for arm in a.ARMS:
                    row = dict(
                        trial_id=f"{arm}-{base}",
                        base_trial_id=base,
                        panel_id=panel,
                        semantic_world_id=digest((panel, world)),
                        world_slot=world,
                        arm=arm,
                        family="objects",
                        kind=kind,
                        delivery=delivery,
                        menu_layout=menu,
                    )
                    assigned.append(row)
                    rows.append(
                        dict(
                            **row,
                            status="completed",
                            valid=True,
                            thinking_closed=True,
                            cap_hit=False,
                            correct=None if kind == "choice" else True,
                            selected_service=("stop" if delivery == "decisive_evidence" else "report")
                            if kind == "choice"
                            else None,
                            record_sha256=digest(row),
                            source_task_record_sha256=digest((base, arm)),
                        )
                    )
    return assigned, rows


def run(assigned, rows, *, audit_passed=True, checks=None):
    return a.analyze_tables(
        assigned,
        rows,
        checks=checks or dict.fromkeys(a.GATES, True),
        audit_readiness=dict(passed=audit_passed),
    )


def choose(rows, arm=a.PRIMARY, *, worlds=range(3), menus=(0, 1), panels=a.PANELS, service="stop"):
    for r in rows:
        if (
            r["arm"] == arm
            and r["kind"] == "choice"
            and r["delivery"] == "redundant_evidence"
            and r["world_slot"] in worlds
            and r["menu_layout"] in menus
            and r["panel_id"] in panels
        ):
            r["selected_service"] = service


def boundary():
    assigned, rows = tables()
    choose(rows)
    for arm in a.CONTROLS:
        choose(rows, arm, worlds=[0])
        choose(rows, arm, worlds=[1], menus=[0])
    return assigned, rows


def effect(
    result, *, left=a.PRIMARY, right="zero_sham", panel="pooled", menu="averaged", category=a.NONREPORT
):
    matches = [
        r
        for r in result["behavior_contrasts"]
        if r["left_arm"] == left
        and r["right_arm"] == right
        and r["panel_id"] == panel
        and r["menu_layout"] == menu
        and r["delivery"] == "redundant_evidence"
        and r["category"] == category
    ]
    assert len(matches) == 1
    return matches[0]


def selected_row(rows, arm, *, world=5, panel="panel_1", menu=0, kind="choice"):
    return next(
        r
        for r in rows
        if r["arm"] == arm
        and r["world_slot"] == world
        and r["panel_id"] == panel
        and r["menu_layout"] == menu
        and r["kind"] == kind
        and r["delivery"] == "redundant_evidence"
    )


def invalidate(row, *, status="completed", cap=False):
    row.update(
        status=status,
        valid=False if status == "completed" else None,
        thinking_closed=True,
        cap_hit=cap,
        correct=None if row["kind"] == "choice" else False,
        selected_service=None,
    )


def test_full15_arm_boundary_requires_every_control_and_preserves_all_denominators():
    result = run(*boundary())
    assert result["decision"]["classification"] == "direction_selective_development_signal"
    assert result["decision"]["primary_observed"]["panel_differences"] == dict(
        panel_1=0.25, panel_2=0.25, pooled=0.25
    )
    assert len(result["decision"]["controls"]) == 12
    for control in result["decision"]["controls"].values():
        assert control["observed"]["panel_differences"] == dict(panel_1=0.125, panel_2=0.125, pooled=0.125)
    assert len(result["outcomes"]) == 4320 and len(result["choices"]) == 2160
    assert len(result["task_readiness"]["diagnostic_cells"]) == 180
    assert len(result["task_readiness"]["choice_cells"]) == 90
    assert len(result["behavior_contrasts"]) == 6318
    assert len(result["world_contrasts"]) == 50544
    assert len(result["decision_bounds"]) == 117
    assert len(result["denominators"]) == 3645
    for arm in a.ARMS:
        exclusive = [
            r
            for r in result["denominators"]
            if r["arm"] == arm
            and r["panel_id"] == "pooled"
            and r["menu_layout"] == "averaged"
            and r["exclusive_category"]
        ]
        assert sum(r["count"] for r in exclusive) == 144


@pytest.mark.parametrize("service", ["repeat", "pattern", "stop"])
def test_primary_is_valid_alternative_not_only_stopping(service):
    assigned, rows = tables()
    choose(rows, service=service)
    result = run(assigned, rows)
    assert result["decision"]["positive_screen"]
    assert effect(result)["estimate"] == 0.25
    assert effect(result, category=service)["estimate"] == 0.25
    assert effect(result, category="report")["estimate"] == -0.25
    if service != "stop":
        assert effect(result, category="stop")["estimate"] == 0


@pytest.mark.parametrize(
    "control",
    ["confidence_positive", "completion_positive", "valence_positive", "valence_negative", "random_05"],
)
def test_one_control_tie_vetoes_specificity_no_average_or_random_drop(control):
    assigned, rows = tables()
    choose(rows)
    choose(rows, control)
    result = run(assigned, rows)
    assert result["decision"]["effect_threshold_met"]
    assert result["decision"]["classification"] == "primary_effect_without_predefined_specificity"
    assert not result["decision"]["controls"][control]["observed_threshold_met"]
    assert control in result["decision"]["controls_with_substantial_observed_alternative_increase"]


@pytest.mark.parametrize("case", ["one_response", "one_panel", "one_menu", "control_only"])
def test_no_primary_rescue_from_one_response_pooling_or_other_arm(case):
    assigned, rows = tables()
    if case == "one_response":
        choose(rows, worlds=[0], menus=[0], panels=["panel_1"])
    elif case == "one_panel":
        choose(rows, worlds=range(12), panels=["panel_1"])
    elif case == "one_menu":
        choose(rows, worlds=range(6), menus=[0])
    else:
        choose(rows, "valence_negative", worlds=range(12))
    result = run(assigned, rows)
    assert result["decision"]["classification"] == "no_substantial_primary_development_effect"
    assert not result["decision"]["positive_screen"]


def test_allowed_sham_invalidity_cannot_create_primary_threshold():
    assigned, rows = tables()
    choose(rows)
    for arm in ("baseline", "zero_sham"):
        invalidate(selected_row(rows, arm))
    result = run(assigned, rows)
    assert result["task_readiness"]["passed"]
    assert result["decision"]["primary_observed_threshold_met"]
    assert not result["decision"]["primary_comparator_favorable_threshold_met"]
    assert result["decision"]["primary_comparator_favorable"]["panel_differences"][
        "panel_1"
    ] == pytest.approx(5 / 24)
    assert not result["decision"]["positive_screen"]


def test_allowed_control_invalidity_cannot_manufacture_specificity_boundary():
    assigned, rows = boundary()
    control = "random_05"
    invalidate(selected_row(rows, control))
    result = run(assigned, rows)
    assert result["task_readiness"]["passed"]
    test = result["decision"]["controls"][control]
    assert test["observed_threshold_met"] and not test["bounded_threshold_met"]
    assert test["comparator_favorable"]["panel_differences"]["panel_1"] == pytest.approx(2 / 24)
    assert result["decision"]["classification"] == "primary_effect_without_predefined_specificity"


@pytest.mark.parametrize("case", ["invalid", "capped", "execution_error", "missing"])
def test_unavailable_target_is_never_a_valid_alternative(case):
    row = dict(status="completed", valid=False, thinking_closed=True, cap_hit=False, selected_service=None)
    if case == "capped":
        row.update(valid=True, selected_service="stop", cap_hit=True)
    elif case in ("execution_error", "missing"):
        row.update(status=case, valid=None)
    assert a.indicator(row, a.NONREPORT) == 0
    assert a.indicator(row, a.NONREPORT, unavailable_as_alternative=True) == 1


def test_one_control_diagnostic_format_failure_blocks_whole_screen_without_dropping_control():
    assigned, rows = tables()
    choose(rows)
    invalidate(selected_row(rows, "random_05", kind="new_fact"))
    result = run(assigned, rows)
    assert result["decision"]["effect_threshold_met"] and result["decision"]["specificity_threshold_met"]
    assert result["decision"]["classification"] == "readiness_failed"
    failed = [c for c in result["task_readiness"]["diagnostic_cells"] if not c["passed"]]
    assert len(failed) == 1 and failed[0]["arm"] == "random_05" and failed[0]["valid"] == 11
    assert len(result["outcomes"]) == 4320


def test_original_failed_audit_is_not_rescued_by_recomputed_passing_cells():
    assigned, rows = tables()
    choose(rows)
    result = run(assigned, rows, audit_passed=False)
    assert result["task_readiness"]["passed"]
    assert result["decision"]["classification"] == "readiness_failed"


@pytest.mark.parametrize("gate", a.GATES)
def test_every_explicit_engineering_gate_can_veto(gate):
    assigned, rows = tables()
    choose(rows)
    flags = dict.fromkeys(a.GATES, True)
    flags[gate] = False
    result = run(assigned, rows, checks=flags)
    assert result["decision"]["classification"] == "engineering_or_receipt_gate_failed"


def test_world_cluster_bootstrap_matches_independent_manual_sampling_and_is_permutation_invariant():
    assigned, rows = tables()
    choose(rows, worlds=[0, 1, 4], menus=[0], panels=["panel_1"], service="pattern")
    choose(rows, worlds=[2, 3, 8, 9], panels=["panel_2"], service="repeat")
    choose(rows, "zero_sham", worlds=[1, 7], menus=[0], panels=["panel_1"], service="pattern")
    result = run(assigned, rows)
    rng = np.random.Generator(np.random.PCG64(2026101440))
    draws = []
    observed = []
    for panel in ("panel_1", "panel_2"):
        ids = sorted((digest((panel, w)), w) for w in range(12))
        # Independent known fixture arithmetic; do not call analyzer's indicators or pairing.
        values = np.array(
            [
                (0.5 * (w in (0, 1, 4)) - 0.5 * (w in (1, 7)))
                if panel == "panel_1"
                else float(w in (2, 3, 8, 9))
                for _, w in ids
            ]
        )
        bank = rng.integers(0, 12, size=(10000, 12))
        draws.append(values[bank].mean(axis=1))
        observed.append(values.mean())
        np.testing.assert_array_equal(
            effect(result, panel=panel)["ci95"], np.quantile(draws[-1], [0.025, 0.975])
        )
    assert effect(result)["estimate"] == np.mean(observed)
    np.testing.assert_array_equal(
        effect(result)["ci95"], np.quantile((draws[0] + draws[1]) / 2, [0.025, 0.975])
    )
    random.Random(29).shuffle(assigned)
    random.Random(47).shuffle(rows)
    assert run(assigned, rows) == result


def test_inventory_rejects_child_duplicate_and_wrong_assignment_metadata():
    assigned, rows = tables()
    with pytest.raises(ValueError, match="combined parent"):
        run(assigned[:1440], rows[:1440])
    bad = deepcopy(rows)
    bad[-1] = deepcopy(bad[0])
    with pytest.raises(ValueError, match="Missing, duplicate"):
        run(assigned, bad)
    bad = deepcopy(rows)
    bad[0]["arm"] = a.PRIMARY
    with pytest.raises(ValueError, match="metadata differs"):
        run(assigned, bad)


def audit_inputs(assigned, rows):
    manifest = dict(
        assay=a.screen.ASSAY,
        task_revision=a.screen.REVISION,
        assignments=assigned,
        diagnostics=[],
        assignment_sha256=digest(assigned),
        analysis_metadata=deepcopy(a.ANALYSIS_METADATA),
        source_sha256=digest("synthetic"),
        model=dict(model_id="synthetic", revision="bound"),
    )
    audit = dict(
        assay="curiosity_suppression_screen_audit",
        parent_manifest_sha256=digest(manifest),
        complete=True,
        integrity_passed=True,
        outcomes=rows,
        readiness=dict(passed=True),
        **dict.fromkeys(a.GATES, True),
    )
    return manifest, audit


@pytest.mark.parametrize("field", ["complete", "integrity_passed", "parent_manifest_sha256", "metadata"])
def test_audit_contract_rejects_partial_or_rehashed_changed_prospective_rules(field):
    manifest, audit = audit_inputs(*tables())
    if field == "metadata":
        manifest["analysis_metadata"]["minimum_panel_difference"] = 0.1
        audit["parent_manifest_sha256"] = digest(manifest)
    elif field == "parent_manifest_sha256":
        audit[field] = digest("wrong")
    else:
        audit[field] = False
    with pytest.raises(ValueError):
        a.analyze_audit(manifest, audit)


def test_bound_loader_and_immutable_exports_preserve_failed_readiness(monkeypatch, tmp_path):
    import csv
    import json
    import sys
    from types import SimpleNamespace

    assigned, rows = tables()
    choose(rows)
    manifest, audit = audit_inputs(assigned, rows)
    audit["readiness"]["passed"] = False
    run_dir = tmp_path / "parent"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit))
    monkeypatch.setitem(
        sys.modules, "audit_curiosity_suppression_screen", SimpleNamespace(audit=lambda run: deepcopy(audit))
    )
    out = tmp_path / "analysis"
    result = a.analyze(run_dir, audit_path, out)
    assert result["decision"]["classification"] == "readiness_failed"
    assert result["complete"] and result["integrity_passed"]
    assert result["audit_byte_sha256"] == a.sha(audit_path)
    assert result["parent_manifest_sha256"] == digest(manifest)
    assert result["outcome_bindings_sha256"] == digest(rows)
    for name, binding in result["exports"].items():
        path = out / binding["path"]
        assert a.sha(path) == binding["byte_sha256"]
        with path.open() as handle:
            exported = list(csv.DictReader(handle))
        assert len(exported) == binding["rows"] == len(result[name])
    before = a.sha(out / "analysis.json")
    monkeypatch.setattr(a, "analyze_audit", lambda m, r: deepcopy(result))
    with pytest.raises(FileExistsError):
        a.analyze(run_dir, audit_path, out)
    assert a.sha(out / "analysis.json") == before
    tampered = deepcopy(audit)
    tampered["readiness"]["passed"] = True
    audit_path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="no longer reproduces"):
        a.load_inputs(run_dir, audit_path)


def test_menu_reversal_vetoes_control_specificity_even_when_both_panel_margins_pass():
    assigned, rows = tables()
    choose(rows, worlds=range(3), menus=[0])
    choose(rows, worlds=range(9), menus=[1])
    choose(rows, "confidence_positive", worlds=range(4), menus=[0])
    result = run(assigned, rows)
    decision = result["decision"]
    assert decision["effect_threshold_met"]
    control = decision["controls"]["confidence_positive"]
    assert all(control["observed"]["panel_differences"][p] >= 0.125 for p in a.PANELS)
    assert control["observed"]["pooled_menu_differences"]["0"] == pytest.approx(-1 / 12)
    assert not control["observed_threshold_met"]
    assert decision["classification"] == "primary_effect_without_predefined_specificity"
