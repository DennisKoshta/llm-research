"""Prospective synthetic screen statistics and audit-binding tests; no study outputs."""

from copy import deepcopy
import csv
import importlib.util
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from curiosity_study.artifacts import digest
from curiosity_study.voluntary_resolution import DELIVERIES, DIAGNOSTICS

SPEC = importlib.util.spec_from_file_location(
    "satisfaction_analysis_under_test", Path(__file__).parents[1] / "scripts/analyze_satisfaction_screen.py"
)
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


def tables():
    assignments, outcomes = [], []
    variants = [("choice", delivery, menu) for delivery in DELIVERIES for menu in (0, 1)]
    variants += [
        (
            kind,
            "decisive_evidence" if kind in ("identify_resolved", "repeated_fact") else "redundant_evidence",
            0,
        )
        for kind in DIAGNOSTICS
    ]
    for panel in analyzer.PANELS:
        for world in range(12):
            semantic = digest((panel, world))
            for kind, delivery, menu in variants:
                base = f"{panel}-{world}-{kind}-{delivery}-{menu}"
                for arm in analyzer.ARMS:
                    a = dict(
                        trial_id=f"{arm}-{base}",
                        base_trial_id=base,
                        panel_id=panel,
                        semantic_world_id=semantic,
                        world_slot=world,
                        arm=arm,
                        family="objects",
                        kind=kind,
                        delivery=delivery,
                        menu_layout=menu,
                    )
                    assignments.append(a)
                    outcomes.append(
                        dict(
                            **a,
                            status="completed",
                            valid=True,
                            thinking_closed=True,
                            cap_hit=False,
                            correct=None if kind == "choice" else True,
                            selected_service=("stop" if delivery == "decisive_evidence" else "report")
                            if kind == "choice"
                            else None,
                            record_sha256=digest(a),
                            source_task_record_sha256=digest((base, arm)),
                        )
                    )
    return assignments, outcomes


def run(assignments, outcomes, *, checks=None, passed=True):
    return analyzer.analyze_tables(
        assignments,
        outcomes,
        checks=dict.fromkeys(analyzer.GATES, True) if checks is None else checks,
        audit_readiness=dict(passed=passed, original_gate="synthetic"),
    )


def set_stops(rows, *, arm=None, worlds=range(3), panels=None, menus=(0, 1), delivery="redundant_evidence"):
    for row in rows:
        if (
            row["kind"] == "choice"
            and row["arm"] == (arm or analyzer.PRIMARY)
            and row["world_slot"] in worlds
            and row["panel_id"] in (panels or analyzer.PANELS)
            and row["menu_layout"] in menus
            and row["delivery"] == delivery
        ):
            row["selected_service"] = "stop"


def endpoint(
    result, *, arm=None, panel="pooled", delivery="redundant_evidence", menu="averaged", category="stop"
):
    rows = [
        r
        for r in result["behavior_contrasts"]
        if r["left_arm"] == (arm or analyzer.PRIMARY)
        and r["panel_id"] == panel
        and r["delivery"] == delivery
        and r["menu_layout"] == menu
        and r["category"] == category
    ]
    assert len(rows) == 1
    return rows[0]


def audit_inputs(assignments, outcomes):
    manifest = dict(
        assay="satisfaction_screen",
        task_revision="satisfaction-screen-v1",
        assignments=assignments,
        diagnostics=[],
        assignment_sha256=digest(assignments),
        analysis_metadata=deepcopy(analyzer.ANALYSIS_METADATA),
        source_sha256=digest("synthetic source"),
        model=dict(model_id="synthetic", revision="bound"),
    )
    audit = dict(
        assay="satisfaction_screen_audit",
        parent_manifest_sha256=digest(manifest),
        complete=True,
        integrity_passed=True,
        readiness=dict(passed=True),
        outcomes=outcomes,
        **dict.fromkeys(analyzer.GATES, True),
    )
    return manifest, audit


def test_exact25pp_each_panel_and_both_menus_pass_without_using_intervals_or_lower_dose():
    assignments, outcomes = tables()
    set_stops(outcomes)
    result = run(assignments, outcomes)
    assert result["decision"]["classification"] == "positive_development_screen"
    assert result["decision"]["primary_panel_differences"] == dict(panel_1=0.25, panel_2=0.25, pooled=0.25)
    assert result["decision"]["primary_pooled_menu_differences"] == {"0": 0.25, "1": 0.25}
    assert endpoint(result, arm=analyzer.SENSITIVITY)["estimate"] == 0
    assert result["task_readiness"]["passed"]
    assert len(result["task_readiness"]["diagnostic_cells"]) == 48
    assert len(result["task_readiness"]["choice_cells"]) == 24
    assert len(result["outcomes"]) == 1152 and len(result["choices"]) == 576


@pytest.mark.parametrize(
    "case",
    ["single_sample", "pooled_only", "single_menu", "lower_dose_only", "one_net_response_below_boundary"],
)
def test_fixed_rule_cannot_be_rescued_by_one_response_pooling_menu_or_lower_dose(case):
    assignments, rows = tables()
    if case == "single_sample":
        set_stops(rows, worlds=[0], panels=["panel_1"], menus=[0])
    elif case == "pooled_only":
        set_stops(rows, worlds=range(12), panels=["panel_1"])
    elif case == "single_menu":
        set_stops(rows, worlds=range(6), menus=[0])
    elif case == "lower_dose_only":
        set_stops(rows, arm=analyzer.SENSITIVITY, worlds=range(12))
    else:
        set_stops(rows)
        changed = next(
            r
            for r in rows
            if r["arm"] == analyzer.PRIMARY
            and r["panel_id"] == "panel_2"
            and r["kind"] == "choice"
            and r["delivery"] == "redundant_evidence"
            and r["world_slot"] == 0
            and r["menu_layout"] == 0
        )
        changed["selected_service"] = "report"
    result = run(assignments, rows)
    assert result["decision"]["classification"] == "threshold_not_met"
    assert not result["decision"]["positive_screen"]


@pytest.mark.parametrize("kind", DIAGNOSTICS)
def test_every_diagnostic_gate_retained_under_every_dose_even_with_primary_effect(kind):
    assignments, rows = tables()
    set_stops(rows)
    selected = [
        r
        for r in rows
        if r["arm"] == analyzer.SENSITIVITY and r["panel_id"] == "panel_2" and r["kind"] == kind
    ]
    selected[0]["correct"] = False
    assert run(assignments, rows)["decision"]["positive_screen"]  # Frozen11/12 boundary.
    selected[1]["correct"] = False
    result = run(assignments, rows)
    assert result["decision"]["effect_threshold_met"]
    assert result["decision"]["classification"] == "readiness_failed"
    assert len(result["outcomes"]) == 1152


@pytest.mark.parametrize("mode", ["invalid", "cap", "error", "missing"])
def test_unavailable_invalid_and_capped_choices_do_not_create_success_or_shrink_denominators(mode):
    assignments, rows = tables()
    set_stops(rows)
    for row in rows:
        if (
            row["arm"] == analyzer.PRIMARY
            and row["delivery"] == "redundant_evidence"
            and row["kind"] == "choice"
            and row["world_slot"] < 3
        ):
            if mode == "cap":
                row["cap_hit"] = True
            elif mode == "invalid":
                row.update(valid=False, thinking_closed=False, selected_service=None)
            else:
                row.update(
                    status="execution_error" if mode == "error" else "missing",
                    valid=None,
                    thinking_closed=None,
                    cap_hit=None,
                    selected_service=None,
                    correct=None,
                )
    result = run(assignments, rows)
    assert endpoint(result)["estimate"] == 0
    name = dict(invalid="invalid", cap="capped", error="execution_failure", missing="missing")[mode]
    assert endpoint(result, category=name)["estimate"] == 0.25
    assert result["decision"]["classification"] == "readiness_failed"
    assert len(result["choices"]) == 576
    selected = [
        r
        for r in result["denominators"]
        if r["arm"] == analyzer.PRIMARY
        and r["panel_id"] == "pooled"
        and r["delivery"] == "redundant_evidence"
        and r["menu_layout"] == "averaged"
    ]
    assert sum(r["count"] for r in selected) == 48
    assert {r["assigned_choices"] for r in selected} == {48}


@pytest.mark.parametrize("gate", analyzer.GATES)
def test_external_engineering_or_receipt_failure_prevents_success(gate):
    assignments, rows = tables()
    set_stops(rows)
    checks = dict.fromkeys(analyzer.GATES, True)
    checks[gate] = False
    result = run(assignments, rows, checks=checks)
    assert result["task_readiness"]["passed"] and result["decision"]["effect_threshold_met"]
    assert result["decision"]["classification"] == "engineering_or_receipt_gate_failed"


def test_existing_failed_audit_is_not_repaired_by_recomputed_counts():
    assignments, rows = tables()
    set_stops(rows)
    result = run(assignments, rows, passed=False)
    assert result["task_readiness"]["passed"] and not result["audit_readiness"]["passed"]
    assert result["decision"]["classification"] == "readiness_failed"


def test_shared_panel_world_bootstrap_matches_independent_nonuniform_paired_calculation():
    assignments, rows = tables()
    # Both arms vary within worlds and menus, so subtracting unpaired marginals or
    # resampling individual choices gives different intervals.
    for row in rows:
        if row["kind"] != "choice" or row["delivery"] != "redundant_evidence":
            continue
        w, m, panel = row["world_slot"], row["menu_layout"], row["panel_id"]
        if row["arm"] == "zero_sham":
            row["selected_service"] = "stop" if (w + m) % 4 == 0 else "report"
        elif row["arm"] == analyzer.PRIMARY:
            row["selected_service"] = "stop" if (w * 3 + m + (panel == "panel_2")) % 5 < 3 else "pattern"
    result = run(assignments, rows)
    rng = np.random.default_rng(2026101438)
    estimates, samples = {}, {}
    for panel in ("panel_1", "panel_2"):
        worlds = sorted({r["semantic_world_id"] for r in rows if r["panel_id"] == panel})
        vector = []
        for world in worlds:
            selected = [
                r
                for r in rows
                if r["semantic_world_id"] == world
                and r["kind"] == "choice"
                and r["delivery"] == "redundant_evidence"
            ]
            left = sum(r["selected_service"] == "stop" for r in selected if r["arm"] == analyzer.PRIMARY) / 2
            right = sum(r["selected_service"] == "stop" for r in selected if r["arm"] == "zero_sham") / 2
            vector.append(left - right)
        vector = np.array(vector)
        samples[panel] = vector[rng.integers(0, 12, (10000, 12))].mean(axis=1)
        estimates[panel] = vector.mean()
    samples["pooled"] = (samples["panel_1"] + samples["panel_2"]) / 2
    estimates["pooled"] = (estimates["panel_1"] + estimates["panel_2"]) / 2
    for panel in ("panel_1", "panel_2", "pooled"):
        actual = endpoint(result, panel=panel)
        assert actual["estimate"] == estimates[panel]
        np.testing.assert_array_equal(actual["ci95"], np.quantile(samples[panel], [0.025, 0.975]))
    for panel in (*analyzer.PANELS, "pooled"):
        assert sum(
            endpoint(result, panel=panel, category=c)["estimate"] for c in analyzer.CATEGORIES
        ) == pytest.approx(0)


def test_equivalent_row_permutations_preserve_entire_analysis_and_draw_bank():
    assignments, rows = tables()
    set_stops(rows)
    expected = run(assignments, rows)
    random.Random(101).shuffle(assignments)
    random.Random(202).shuffle(rows)
    assert run(assignments, rows) == expected


@pytest.mark.parametrize(
    "case",
    ["child", "missing", "duplicate", "metadata", "cross_panel_duplicate", "fake_valid", "fake_service"],
)
def test_structural_and_outcome_rebinding_rejected(case):
    assignments, rows = tables()
    if case == "child":
        assignments, rows = assignments[:576], rows[:576]
    elif case == "missing":
        rows.pop()
    elif case == "duplicate":
        rows[-1] = deepcopy(rows[0])
    elif case == "metadata":
        rows[0]["menu_layout"] = 1
    elif case == "cross_panel_duplicate":
        first = next(
            a["semantic_world_id"] for a in assignments if a["panel_id"] == "panel_1" and a["world_slot"] == 0
        )
        for row in assignments + rows:
            if row["panel_id"] == "panel_2" and row["world_slot"] == 0:
                row["semantic_world_id"] = first
    elif case == "fake_valid":
        rows[0]["valid"] = 1
    else:
        rows[0].update(valid=False, thinking_closed=False)
    with pytest.raises(ValueError):
        run(assignments, rows)


@pytest.mark.parametrize("change", ["incomplete", "integrity", "manifest", "metadata", "assignment"])
def test_bound_combined_audit_required_but_readiness_not_prerequisite(change):
    assignments, rows = tables()
    manifest, audit = audit_inputs(assignments, rows)
    if change == "incomplete":
        audit["complete"] = False
    elif change == "integrity":
        audit["integrity_passed"] = False
    elif change == "manifest":
        audit["parent_manifest_sha256"] = digest("wrong")
    elif change == "metadata":
        manifest["analysis_metadata"]["primary_arm"] = analyzer.SENSITIVITY
        audit["parent_manifest_sha256"] = digest(manifest)
    else:
        manifest["assignment_sha256"] = digest("wrong")
        audit["parent_manifest_sha256"] = digest(manifest)
    with pytest.raises(ValueError):
        analyzer.analyze_audit(manifest, audit)


def test_loader_reproduces_full_audit_and_immutable_exports_preserve_all_rows(tmp_path, monkeypatch):
    assignments, rows = tables()
    set_stops(rows)
    manifest, audit = audit_inputs(assignments, rows)
    audit["readiness"]["passed"] = False
    root = tmp_path / "parent"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps(manifest))
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit))
    called = []

    def independent_audit(path):
        called.append(path)
        return deepcopy(audit)

    monkeypatch.setitem(sys.modules, "audit_satisfaction_screen", SimpleNamespace(audit=independent_audit))
    out = tmp_path / "analysis"
    result = analyzer.analyze(root, audit_path, out)
    assert called == [root] and result["decision"]["classification"] == "readiness_failed"
    assert result["audit_byte_sha256"] == analyzer.sha(audit_path)
    for key, binding in result["exports"].items():
        path = out / binding["path"]
        assert analyzer.sha(path) == binding["byte_sha256"]
        with path.open() as handle:
            assert len(list(csv.DictReader(handle))) == len(result[key]) == binding["rows"]
    with pytest.raises(FileExistsError):
        analyzer.analyze(root, audit_path, out)
    monkeypatch.setitem(
        sys.modules,
        "audit_satisfaction_screen",
        SimpleNamespace(audit=lambda run: {**audit, "changed": True}),
    )
    with pytest.raises(ValueError, match="no longer reproduces"):
        analyzer.load_inputs(root, audit_path)
