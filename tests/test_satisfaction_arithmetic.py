"""Independent checker tests use synthetic tables only, never study outcomes."""

from copy import deepcopy
import csv
from io import BytesIO
import json
from pathlib import Path
import random
import sys
import tarfile

import numpy as np
import pytest

from curiosity_study.artifacts import canonical, digest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import analyze_satisfaction_screen as analyzer  # noqa: E402
import check_satisfaction_arithmetic as check  # noqa: E402


def tables():
    assignments, outcomes = [], []
    variants = [("choice", d, m) for d in check.DELIVERIES for m in (0, 1)]
    variants += [
        (k, "decisive_evidence" if k in ("identify_resolved", "repeated_fact") else "redundant_evidence", 0)
        for k in check.DIAGNOSTICS
    ]
    for pi, panel in enumerate(check.PANELS):
        for world in range(12):
            for kind, delivery, menu in variants:
                base = f"{panel}/{world}/{kind}/{delivery}/{menu}"
                for arm in check.ARMS:
                    a = dict(
                        trial_id=f"{arm}/{base}",
                        base_trial_id=base,
                        panel_id=panel,
                        semantic_world_id=digest((panel, world)),
                        world_slot=world,
                        arm=arm,
                        family="objects",
                        kind=kind,
                        delivery=delivery,
                        menu_layout=menu,
                        panel_seed=100 + pi,
                        world_id=f"{panel}/{world}",
                        base_row=len(assignments) // 4,
                        sampling_seed=world,
                        source_assignment_sha256=digest(base),
                    )
                    assignments.append(a)
                    outcomes.append(
                        dict(
                            **a,
                            assignment_sha256=digest(a),
                            record_sha256=digest((a, "record")),
                            source_task_record_sha256=digest((a, "inner")),
                            status="completed",
                            valid=True,
                            correct=None if kind == "choice" else True,
                            thinking_closed=True,
                            closed=True,
                            cap_hit=False,
                            selected_service=("stop" if delivery == "decisive_evidence" else "report")
                            if kind == "choice"
                            else None,
                            failure=None,
                            generated_tokens=10,
                        )
                    )
    return assignments, outcomes


def stats(assignments, outcomes, *, gates=None, ready=True):
    return check.recompute(
        assignments,
        outcomes,
        gates=dict.fromkeys(check.GATES, True) if gates is None else gates,
        audit_readiness={"passed": ready},
    )


def set_stop(rows, arm=check.ARMS[3], slots=range(3), panels=check.PANELS, menus=(0, 1)):
    for row in rows:
        if (
            row["kind"] == "choice"
            and row["delivery"] == "redundant_evidence"
            and row["arm"] == arm
            and row["world_slot"] in slots
            and row["panel_id"] in panels
            and row["menu_layout"] in menus
        ):
            row["selected_service"] = "stop"


def endpoint(result, panel="pooled", menu="averaged", category="stop"):
    return next(
        r
        for r in result["behavior_contrasts"]
        if r["left_arm"] == check.ARMS[3]
        and r["delivery"] == "redundant_evidence"
        and r["panel_id"] == panel
        and r["menu_layout"] == menu
        and r["category"] == category
    )


def test_every_synthetic_table_agrees_with_separate_analyzer_and_is_permutation_invariant():
    assignments, rows = tables()
    set_stop(rows, slots=(0, 2, 7), panels=(check.PANELS[0],), menus=(0,))
    set_stop(rows, slots=(1, 3, 5, 9), panels=(check.PANELS[1],))
    wanted = analyzer.analyze_tables(
        assignments, rows, checks=dict.fromkeys(check.GATES, True), audit_readiness={"passed": True}
    )
    actual = stats(assignments, rows)
    comparison = check.Comparison()
    for key in check.EXPORTS:
        comparison.equal(wanted[key], actual[key], key)
    for key in ("task_readiness", "decision", "bootstrap"):
        comparison.equal(wanted[key], actual[key], key, subset=True)
    assert [len(actual[k]) for k in check.EXPORTS] == [1152, 576, 864, 648, 5184]
    random.Random(53).shuffle(assignments)
    random.Random(92).shuffle(rows)
    assert actual == stats(assignments, rows)


def test_shared_whole_world_bootstrap_and_equal_panel_pool_match_manual_draws():
    assignments, rows = tables()
    set_stop(rows, slots=range(3), panels=(check.PANELS[0],), menus=(0,))
    set_stop(rows, slots=range(8), panels=(check.PANELS[1],))
    result = stats(assignments, rows)
    rng = np.random.default_rng(2026101438)
    draws, means = [], []
    for pi, panel in enumerate(check.PANELS):
        by_world = {
            digest((panel, w)): (0.5 if pi == 0 and w < 3 else 1.0 if pi == 1 and w < 8 else 0.0)
            for w in range(12)
        }
        values = np.array([by_world[w] for w in sorted(by_world)])
        bank = rng.integers(0, 12, size=(10000, 12))
        samples = np.array([sum(values[i] for i in draw) / 12 for draw in bank])
        draws.append(samples)
        means.append(float(values.mean()))
        assert endpoint(result, panel)["estimate"] == values.mean()
        assert endpoint(result, panel)["ci95"] == np.quantile(samples, [0.025, 0.975]).tolist()
    pooled = endpoint(result)
    assert pooled["estimate"] == sum(means) / 2
    assert pooled["ci95"] == np.quantile((draws[0] + draws[1]) / 2, [0.025, 0.975]).tolist()
    # One same world carries all arms/menus; it is not a 48-independent-row interval.
    assert endpoint(result, check.PANELS[0])["estimate"] == 0.125


def test_paired_common_arm_variation_cancels_and_degenerate_interval_is_explicit():
    assignments, rows = tables()
    for arm in (check.ARMS[1], check.ARMS[3]):
        set_stop(rows, arm=arm, slots=range(6))
    result = stats(assignments, rows)
    assert endpoint(result)["ci95"] == [0.0, 0.0]
    assert endpoint(result)["empirical_interval_degenerate"] is True
    assert all(
        r["paired_difference"] == 0 for r in result["world_contrasts"] if r["left_arm"] == check.ARMS[3]
    )


def test_invalid_cap_unclosed_and_unavailable_mass_never_becomes_valid_stop():
    assignments, rows = tables()
    selected = [
        r
        for r in rows
        if r["arm"] == check.ARMS[3]
        and r["panel_id"] == check.PANELS[0]
        and r["kind"] == "choice"
        and r["delivery"] == "redundant_evidence"
        and r["menu_layout"] == 0
    ]
    selected[0].update(selected_service="stop", cap_hit=True)
    selected[1].update(valid=False, selected_service=None)
    selected[2].update(valid=False, selected_service=None, thinking_closed=False, closed=False)
    selected[3].update(
        status="execution_error",
        valid=None,
        thinking_closed=None,
        closed=None,
        cap_hit=None,
        selected_service=None,
    )
    selected[4].update(
        status="missing", valid=None, thinking_closed=None, closed=None, cap_hit=None, selected_service=None
    )
    result = stats(assignments, rows)
    counts = {
        r["category"]: r
        for r in result["denominators"]
        if r["arm"] == check.ARMS[3]
        and r["panel_id"] == check.PANELS[0]
        and r["delivery"] == "redundant_evidence"
        and r["menu_layout"] == 0
    }
    assert {k: v["count"] for k, v in counts.items()} == dict(
        report=7, repeat=0, pattern=0, stop=0, invalid=2, capped=1, execution_failure=1, missing=1
    )
    assert all(r["assigned_choices"] == 12 and r["cap_hits"] == 1 for r in counts.values())
    assert result["decision"]["classification"] == "readiness_failed"
    assert endpoint(result)["estimate"] == 0


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("boundary", "positive_development_screen"),
        ("menu_zero", "threshold_not_met"),
        ("sensitivity_only", "threshold_not_met"),
        ("one_panel", "threshold_not_met"),
        ("audit_failed", "readiness_failed"),
        ("engineering_failed", "engineering_or_receipt_gate_failed"),
        ("diagnostic_failed", "readiness_failed"),
    ],
)
def test_primary_threshold_menu_veto_and_no_sensitivity_or_gate_rescue(mode, expected):
    assignments, rows = tables()
    set_stop(rows)
    if mode == "menu_zero":
        for r in rows:
            if r["kind"] == "choice" and r["delivery"] == "redundant_evidence":
                r["selected_service"] = "report"
        set_stop(rows, slots=range(6), menus=(0,))
    if mode == "sensitivity_only":
        for r in rows:
            if r["kind"] == "choice" and r["delivery"] == "redundant_evidence":
                r["selected_service"] = "report"
        set_stop(rows, arm=check.ARMS[2], slots=range(12))
    if mode == "one_panel":
        for r in rows:
            if (
                r["kind"] == "choice"
                and r["delivery"] == "redundant_evidence"
                and r["panel_id"] == check.PANELS[1]
            ):
                r["selected_service"] = "report"
    if mode == "diagnostic_failed":
        for r in rows:
            if (
                r["arm"] == check.ARMS[0]
                and r["panel_id"] == check.PANELS[0]
                and r["kind"] == "new_fact"
                and r["world_slot"] < 2
            ):
                r["correct"] = False
    gates = dict.fromkeys(check.GATES, True)
    if mode == "engineering_failed":
        gates["receipts_passed"] = False
    result = stats(assignments, rows, gates=gates, ready=mode != "audit_failed")
    assert result["decision"]["classification"] == expected
    assert result["decision"]["sensitivity_can_rescue"] is False
    if mode == "boundary":
        assert result["decision"]["primary_panel_differences"] == dict(
            panel_1=0.25, panel_2=0.25, pooled=0.25
        )
    if mode == "diagnostic_failed":
        bad = [r for r in result["task_readiness"]["diagnostic_cells"] if not r["passed"]]
        assert len(bad) == 1 and bad[0]["correct"] == 10 and bad[0]["assigned"] == 12


@pytest.mark.parametrize(
    "change", ["duplicate", "missing", "metadata", "bool_slot", "valid_unclosed", "fake_correct", "service"]
)
def test_malformed_inventory_fails_closed(change):
    assignments, rows = tables()
    if change == "duplicate":
        rows[-1] = deepcopy(rows[0])
    elif change == "missing":
        rows.pop()
    elif change == "metadata":
        rows[0]["world_slot"] = 9
    elif change == "bool_slot":
        assignments[0]["world_slot"] = rows[0]["world_slot"] = False
    elif change == "valid_unclosed":
        rows[0]["thinking_closed"] = False
    elif change == "fake_correct":
        next(r for r in rows if r["kind"] == "new_fact")["correct"] = 1
    else:
        rows[0]["selected_service"] = "distractor"
    with pytest.raises(ValueError):
        stats(assignments, rows)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical(value) + "\n")


def artifact_tree(root, *, failed=False):
    assignments, rows = tables()
    set_stop(rows)
    if failed:
        for row in rows:
            if (
                row["kind"] == "new_fact"
                and row["arm"] == check.ARMS[0]
                and row["panel_id"] == check.PANELS[0]
                and row["world_slot"] < 2
            ):
                row["correct"] = False
    recomputed = stats(assignments, rows, ready=not failed)
    audit_ready = dict(assigned=1152, choice_correctness=None, passed=not failed, interpretation="synthetic")
    for key in ("diagnostic_cells", "choice_cells"):
        audit_ready[key] = [
            {k: v for k, v in r.items() if k not in ("observed", "unclosed")}
            for r in recomputed["task_readiness"][key]
        ]
        if key == "choice_cells":
            audit_ready[key] = [dict(r, correct=0) for r in audit_ready[key]]
    sources = {"synthetic.py": "# Synthetic provenance only\n"}
    payloads = {
        "src/curiosity_study/synthetic.py": sources["synthetic.py"].encode(),
        "scripts/analyze_satisfaction_screen.py": b"# synthetic analyzer\n",
        "scripts/audit_satisfaction_screen.py": b"# synthetic auditor\n",
    }
    bindings = []
    for i in range(14):
        path = f"reports/synthetic-{i}.json"
        value = {"synthetic": i}
        payloads[path] = canonical(value).encode()
        bindings.append(
            dict(
                role=check.RECEIPT_ROLES[i],
                path=path,
                byte_sha256=check.hashlib.sha256(payloads[path]).hexdigest(),
                content_sha256=digest(value),
            )
        )
    manifest = dict(
        assay="satisfaction_screen",
        task_revision="satisfaction-screen-v1",
        phase="development_pilot",
        inference_scope="development",
        assignments=[],
        diagnostics=assignments,
        assignment_sha256=digest(assignments),
        source_sha256=digest(sources),
        model={"model_id": "synthetic"},
        analysis_metadata=deepcopy(check.PLAN),
        bound_receipts=bindings,
    )
    run = root / "run"
    save(run / "manifest.json", manifest)
    with tarfile.open(run / "source.tar.gz", "w:gz") as archive:
        for name, payload in payloads.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, BytesIO(payload))
    frozen = dict(
        manifest_sha256=digest(manifest),
        source_sha256=manifest["source_sha256"],
        archive_sha256=check.sha(run / "source.tar.gz"),
        documents={b["path"]: b["byte_sha256"] for b in bindings},
    )
    save(run / "freeze.json", frozen)
    descriptors, children = [], []
    for group, arms in (
        ("baseline_zero", check.ARMS[:2]),
        ("positive_half", [check.ARMS[2]]),
        ("positive_full", [check.ARMS[3]]),
    ):
        child = dict(
            manifest,
            execution_group=group,
            parent_manifest_sha256=digest(manifest),
            diagnostics=[a for a in assignments if a["arm"] in arms],
        )
        child["assignment_sha256"] = digest(child["diagnostics"])
        child_path = root / group
        save(child_path / "manifest.json", child)
        descriptors.append(dict(path=str(child_path), execution_group=group, manifest_sha256=digest(child)))
        children.append(dict(execution_group=group, manifest_sha256=digest(child)))
    save(run / "children.json", descriptors)
    audit = dict(
        assay="satisfaction_screen_audit",
        assigned=1152,
        assigned_choices=576,
        terminal_generations=1152,
        terminal_receipt_slots=576,
        complete=True,
        integrity_passed=True,
        parent_manifest_sha256=digest(manifest),
        source_sha256=manifest["source_sha256"],
        auditor_byte_sha256=check.hashlib.sha256(
            payloads["scripts/audit_satisfaction_screen.py"]
        ).hexdigest(),
        outcomes=rows,
        outcomes_sha256=digest(rows),
        readiness=audit_ready,
        passed=not failed,
        provenance=dict(freeze=frozen, bound_receipts=bindings),
        children=children,
        **dict.fromkeys(check.GATES, True),
    )
    audit_path = root / "audit.json"
    save(audit_path, audit)
    result = analyzer.analyze_audit(manifest, audit)
    result["analyzer_byte_sha256"] = check.hashlib.sha256(
        payloads["scripts/analyze_satisfaction_screen.py"]
    ).hexdigest()
    result["audit_byte_sha256"] = check.sha(audit_path)
    out = root / "analysis"
    out.mkdir()
    result["exports"] = {}
    for key in check.EXPORTS:
        path = out / (key.replace("_", "-") + ".csv")
        analyzer.write_csv(path, result[key])
        result["exports"][key] = dict(path=path.name, byte_sha256=check.sha(path), rows=len(result[key]))
    analysis_path = out / "analysis.json"
    save(analysis_path, result)
    return run, analysis_path, audit_path


@pytest.mark.parametrize("failed", [False, True])
def test_complete_artifact_check_preserves_readiness_failure(tmp_path, failed):
    paths = artifact_tree(tmp_path, failed=failed)
    result = check.check(*paths)
    assert result["passed"] is True
    assert result["audit_readiness_passed"] is (not failed)
    assert result["decision"]["classification"] == (
        "readiness_failed" if failed else "positive_development_screen"
    )
    assert result["maximum_absolute_error"] == 0
    assert result["compared_values"] > 50000
    assert "GPU geometry" in result["excluded_scope"]


@pytest.mark.parametrize(
    "change", ["interval", "world", "denominator", "classification", "audit_hash", "export", "archive"]
)
def test_rehashed_arithmetic_export_and_provenance_tampering_is_rejected(tmp_path, change):
    run, analysis_path, audit_path = artifact_tree(tmp_path)
    result = json.loads(analysis_path.read_text())
    if change == "interval":
        result["behavior_contrasts"][0]["ci95"][0] += 0.1
    elif change == "world":
        result["world_contrasts"][0]["paired_difference"] += 0.5
    elif change == "denominator":
        result["denominators"][0]["assigned_choices"] -= 1
    elif change == "classification":
        result["decision"]["classification"] = "threshold_not_met"
    elif change == "audit_hash":
        result["audit_sha256"] = "0" * 64
    elif change == "archive":
        path = run / "source.tar.gz"
        path.write_bytes(path.read_bytes() + b"tampered")
    else:
        export = result["exports"]["denominators"]
        path = analysis_path.parent / export["path"]
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["count"] = "999"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        export["byte_sha256"] = check.sha(path)
    save(analysis_path, result)
    with pytest.raises(ValueError):
        check.check(run, analysis_path, audit_path)


def test_cli_refuses_existing_output_before_reading_inputs(tmp_path, monkeypatch):
    out = tmp_path / "existing.json"
    out.write_text("unchanged")
    monkeypatch.setattr(
        sys,
        "argv",
        ["checker", "--run", "missing", "--analysis", "missing", "--audit", "missing", "--out", str(out)],
    )
    with pytest.raises(ValueError, match="immutable"):
        check.main()
    assert out.read_text() == "unchanged"
