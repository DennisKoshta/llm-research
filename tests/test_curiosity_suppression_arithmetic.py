"""Independent tensor arithmetic, manual decisions, and synthetic frozen-artifact tampering."""

from copy import deepcopy
import csv
from io import BytesIO
import json
from pathlib import Path
import random
import shutil
import sys
import tarfile

import numpy as np
import pytest

from curiosity_study.artifacts import canonical, digest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import analyze_curiosity_suppression_screen as analyzer  # noqa: E402
import check_curiosity_suppression_arithmetic as check  # noqa: E402


def tables():
    base = []
    variants = [("choice", d, m) for d in check.DELIVERIES for m in (0, 1)]
    variants += [
        (k, "decisive_evidence" if k in ("identify_resolved", "repeated_fact") else "redundant_evidence", 0)
        for k in check.DIAGNOSTICS
    ]
    for pi, panel in enumerate(check.PANELS):
        for world in range(12):
            for kind, delivery, menu in variants:
                source = dict(index=world, kind=kind, delivery=delivery, menu=menu)
                base_id = f"{panel}/{world}/{kind}/{delivery}/{menu}"
                base.append(
                    dict(
                        base_trial_id=base_id,
                        panel_id=panel,
                        panel_seed=100 + pi,
                        semantic_world_id=digest((panel, world)),
                        world_slot=world,
                        world_id=f"{panel}/{world}",
                        family="objects",
                        kind=kind,
                        delivery=delivery,
                        menu_layout=menu,
                        sampling_seed=len(base),
                        source_assignment=source,
                        source_assignment_sha256=digest(source),
                        inference_request_sha256=digest(base_id),
                    )
                )
    assigned, outcomes = [], []
    for cohort in range(9):
        order = check.ARMS[1:]
        rotation = 5 * cohort % 14
        order = ("baseline", *(order[rotation:] + order[:rotation]))
        for position, arm in enumerate(order):
            for index in range(cohort * 32, (cohort + 1) * 32):
                a = dict(
                    **base[index],
                    trial_id=f"{arm}/{base[index]['base_trial_id']}",
                    arm=arm,
                    parent_row=len(assigned),
                    base_row=index,
                    base_cohort=cohort,
                    arm_order_in_cohort=position,
                    execution_group=f"worker_{cohort % 3}",
                )
                assigned.append(a)
                outcomes.append(
                    dict(
                        **a,
                        assignment_sha256=digest(a),
                        record_sha256=digest((a, "record")),
                        source_task_record_sha256=digest((a, "inner")),
                        status="completed",
                        valid=True,
                        correct=None if a["kind"] == "choice" else True,
                        thinking_closed=True,
                        closed=True,
                        cap_hit=False,
                        selected_service=("stop" if a["delivery"] == "decisive_evidence" else "report")
                        if a["kind"] == "choice"
                        else None,
                        generated_tokens=10,
                        failure=None,
                    )
                )
    return assigned, outcomes, base


def stats(assignments, outcomes, *, gates=None, ready=True):
    return check.recompute(
        assignments,
        outcomes,
        gates=dict.fromkeys(check.GATES, True) if gates is None else gates,
        audit_readiness={"passed": ready},
    )


def change(rows, arm=check.PRIMARY, *, slots=range(3), panels=check.PANELS, menus=(0, 1), service="stop"):
    for r in rows:
        if (
            r["kind"] == "choice"
            and r["delivery"] == "redundant_evidence"
            and r["arm"] == arm
            and r["world_slot"] in slots
            and r["panel_id"] in panels
            and r["menu_layout"] in menus
        ):
            r["selected_service"] = service


def effect(
    result,
    *,
    left=check.PRIMARY,
    right="zero_sham",
    panel="pooled",
    menu="averaged",
    category=check.NONREPORT,
):
    return next(
        r
        for r in result["behavior_contrasts"]
        if r["left_arm"] == left
        and r["right_arm"] == right
        and r["panel_id"] == panel
        and r["menu_layout"] == menu
        and r["category"] == category
        and r["delivery"] == "redundant_evidence"
    )


def boundary():
    assignments, rows, base = tables()
    change(rows)
    for arm in check.CONTROLS:
        change(rows, arm, slots=(0,))
        change(rows, arm, slots=(1,), menus=(0,))
    return assignments, rows, base


def test_all_six_tables_agree_with_separate_analyzer_and_ignore_input_order():
    assignments, rows, _ = boundary()
    change(rows, slots=(0, 2, 5, 8), panels=(check.PANELS[1],), menus=(1,), service="pattern")
    actual = stats(assignments, rows)
    expected = analyzer.analyze_tables(
        assignments, rows, checks=dict.fromkeys(check.GATES, True), audit_readiness={"passed": True}
    )
    c = check.Comparison()
    for key in check.EXPORTS:
        c.equal(actual[key], expected[key], key)
    for key in ("decision", "bootstrap", "task_readiness"):
        c.equal(expected[key], actual[key], key, subset=True)
    assert [len(actual[k]) for k in check.EXPORTS] == [4320, 2160, 3645, 6318, 50544, 117]
    random.Random(7).shuffle(assignments)
    random.Random(19).shuffle(rows)
    assert stats(assignments, rows) == actual
    assert c.maximum_absolute_error <= 1e-12


def test_manual_shared_world_draws_and_equal_panel_pooling():
    assignments, rows, _ = tables()
    change(rows, slots=range(3), panels=(check.PANELS[0],), menus=(0,))
    change(rows, slots=range(8), panels=(check.PANELS[1],), service="repeat")
    result = stats(assignments, rows)
    rng = np.random.Generator(np.random.PCG64(2026101440))
    samples, means = [], []
    for pi, panel in enumerate(check.PANELS):
        by_world = {
            digest((panel, w)): 0.5 if pi == 0 and w < 3 else 1.0 if pi == 1 and w < 8 else 0.0
            for w in range(12)
        }
        values = np.array([by_world[w] for w in sorted(by_world)])
        bank = rng.integers(0, 12, (10000, 12))
        draws = np.array([sum(values[i] for i in draw) / 12 for draw in bank])
        samples.append(draws)
        means.append(float(values.mean()))
        assert effect(result, panel=panel)["estimate"] == values.mean()
        assert effect(result, panel=panel)["ci95"] == np.quantile(draws, [0.025, 0.975]).tolist()
    assert effect(result)["estimate"] == sum(means) / 2
    assert effect(result)["ci95"] == np.quantile((samples[0] + samples[1]) / 2, [0.025, 0.975]).tolist()
    # A common world effect cancels before resampling, not after independent arm resampling.
    change(rows, "zero_sham", slots=range(3), panels=(check.PANELS[0],), menus=(0,))
    result = stats(assignments, rows)
    assert effect(result, panel="panel_1")["ci95"] == [0.0, 0.0]


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("boundary", "direction_selective_development_signal"),
        ("one_sample", "no_substantial_primary_development_effect"),
        ("one_panel", "no_substantial_primary_development_effect"),
        ("one_menu", "no_substantial_primary_development_effect"),
        ("control_tie", "primary_effect_without_predefined_specificity"),
        ("control_menu_veto", "primary_effect_without_predefined_specificity"),
        ("audit_failed", "readiness_failed"),
        ("one_invalid_diagnostic", "readiness_failed"),
        ("engineering_failed", "engineering_or_receipt_gate_failed"),
    ],
)
def test_fixed_decision_boundaries_and_no_rescue(mode, expected):
    assignments, rows, _ = tables()
    change(
        rows,
        slots=(0,)
        if mode == "one_sample"
        else range(6)
        if mode in ("one_menu", "control_menu_veto")
        else range(3),
        panels=("panel_1",) if mode == "one_panel" else check.PANELS,
        menus=(0,) if mode in ("one_menu", "control_menu_veto") else (0, 1),
    )
    if mode == "control_tie":
        change(rows, "valence_negative")
    if mode == "control_menu_veto":
        # Primary passes: .5 in menu0 and one extra menu1 response; comparator dominates menu1.
        change(rows, slots=(0,), menus=(1,))
        change(rows, "random_05", slots=range(3), menus=(1,))
    if mode == "boundary":
        for arm in check.CONTROLS:
            change(rows, arm, slots=(0,))
            change(rows, arm, slots=(1,), menus=(0,))
    if mode == "one_invalid_diagnostic":
        row = next(r for r in rows if r["arm"] == "confidence_positive" and r["kind"] == "new_fact")
        row.update(valid=False, correct=False)
    gates = dict.fromkeys(check.GATES, True)
    if mode == "engineering_failed":
        gates["worker_policies_equal"] = False
    result = stats(assignments, rows, gates=gates, ready=mode != "audit_failed")
    assert result["decision"]["classification"] == expected
    if mode == "boundary":
        assert result["decision"]["primary_observed"]["panel_differences"] == dict(
            panel_1=0.25, panel_2=0.25, pooled=0.25
        )
        assert all(
            c["observed"]["panel_differences"]["panel_1"] == 0.125
            for c in result["decision"]["controls"].values()
        )


@pytest.mark.parametrize("comparator", ("zero_sham", *check.CONTROLS))
def test_each_comparator_invalid_can_veto_without_becoming_observed_nonreport(comparator):
    assignments, rows, _ = boundary()
    row = next(
        r
        for r in rows
        if r["arm"] == comparator
        and r["kind"] == "choice"
        and r["delivery"] == "redundant_evidence"
        and r["panel_id"] == "panel_1"
        and r["world_slot"] == 8
        and r["menu_layout"] == 1
    )
    row.update(valid=False, selected_service=None)
    result = stats(assignments, rows)
    assert result["task_readiness"]["passed"] is True  # The frozen choice gate permits 23/24 valid.
    observed = (
        result["decision"]["primary_observed"]
        if comparator == "zero_sham"
        else result["decision"]["controls"][comparator]["observed"]
    )
    bounded = (
        result["decision"]["primary_comparator_favorable"]
        if comparator == "zero_sham"
        else result["decision"]["controls"][comparator]["comparator_favorable"]
    )
    assert observed["panel_differences"]["panel_1"] - bounded["panel_differences"][
        "panel_1"
    ] == pytest.approx(1 / 24)
    assert result["decision"]["positive_screen"] is False
    assert result["decision"]["primary_observed_threshold_met"] is True


def test_category_mass_invalid_caps_errors_and_missing_cannot_count_as_reduction():
    assignments, rows, _ = tables()
    selected = [
        r
        for r in rows
        if r["arm"] == check.PRIMARY
        and r["kind"] == "choice"
        and r["delivery"] == "redundant_evidence"
        and r["panel_id"] == "panel_1"
    ][:7]
    selected[0].update(selected_service="repeat")
    selected[1].update(selected_service="pattern")
    selected[2].update(selected_service="stop")
    selected[3].update(valid=False, selected_service=None)
    selected[4].update(selected_service="stop", cap_hit=True)
    selected[5].update(status="execution_error", valid=None, correct=None, selected_service=None)
    selected[6].update(status="missing", valid=None, correct=None, selected_service=None)
    result = stats(assignments, rows)
    cells = [
        r
        for r in result["denominators"]
        if r["arm"] == check.PRIMARY
        and r["panel_id"] == "panel_1"
        and r["delivery"] == "redundant_evidence"
        and r["menu_layout"] == "averaged"
    ]
    assert sum(r["count"] for r in cells if r["exclusive_category"]) == 24
    assert {r["category"]: r["count"] for r in cells} == dict(
        report=17,
        repeat=1,
        pattern=1,
        stop=1,
        invalid=1,
        capped=1,
        execution_failure=1,
        missing=1,
        valid_nonreport=3,
    )
    assert result["decision"]["classification"] == "readiness_failed"


@pytest.mark.parametrize(
    "mode",
    ("missing", "duplicate", "unknown_control", "bool_slot", "metadata", "fake_correct", "unclosed_valid"),
)
def test_crossing_and_flag_errors_rejected(mode):
    assignments, rows, _ = tables()
    if mode == "missing":
        rows.pop()
    elif mode == "duplicate":
        rows[-1] = deepcopy(rows[0])
    elif mode == "unknown_control":
        rows[0]["arm"] = assignments[0]["arm"] = "replacement"
    elif mode == "bool_slot":
        rows[0]["world_slot"] = assignments[0]["world_slot"] = False
    elif mode == "metadata":
        rows[0]["world_slot"] = 8
    elif mode == "fake_correct":
        next(r for r in rows if r["kind"] == "new_fact")["correct"] = 1
    else:
        rows[0]["thinking_closed"] = False
    with pytest.raises(ValueError):
        stats(assignments, rows)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical(value) + "\n")


def artifact_tree(root, *, failed=False):
    assignments, rows, base = boundary()
    if failed:
        row = next(r for r in rows if r["kind"] == "new_fact" and r["arm"] == check.PRIMARY)
        row.update(valid=False, correct=False)
    computed = stats(assignments, rows, ready=not failed)
    ready = dict(assigned=4320, choice_correctness=None, passed=not failed, interpretation="synthetic")
    for table in ("diagnostic_cells", "choice_cells"):
        ready[table] = [
            {k: v for k, v in r.items() if k not in ("observed", "unclosed")}
            for r in computed["task_readiness"][table]
        ]
        if table == "choice_cells":
            ready[table] = [dict(r, correct=0) for r in ready[table]]
    preflight = dict(
        requests=[dict(id=r["base_trial_id"]) for r in base], base_assignment_sha256=digest(base)
    )
    preflight["requests_sha256"] = digest(preflight["requests"])
    prior = dict(screen_plan=dict(base_assignments=base), generation_preflight=preflight)
    plan = dict(base_assignments=base, base_assignment_sha256=digest(base), assignments=assignments)
    sources = {"synthetic.py": "# synthetic frozen source\n"}
    payloads = {
        "src/curiosity_study/synthetic.py": sources["synthetic.py"].encode(),
        "scripts/analyze_curiosity_suppression_screen.py": b"# synthetic analyzer\n",
        "scripts/audit_curiosity_suppression_screen.py": b"# synthetic auditor\n",
    }
    bindings = []
    for i, role in enumerate(check.RECEIPT_ROLES):
        value = (
            prior
            if role == "source_screen_manifest"
            else preflight
            if role == "generation_preflight"
            else dict(synthetic=i)
        )
        path = f"reports/{role}.json"
        payloads[path] = canonical(value).encode()
        bindings.append(
            dict(
                role=role,
                path=path,
                byte_sha256=check.hashlib.sha256(payloads[path]).hexdigest(),
                content_sha256=digest(value),
            )
        )
    manifest = dict(
        assay="curiosity_suppression_screen",
        task_revision="curiosity-suppression-screen-v1",
        phase="development_pilot",
        inference_scope="development",
        assignments=[],
        diagnostics=assignments,
        assignment_sha256=digest(assignments),
        source_sha256=digest(sources),
        model={"model_id": "synthetic"},
        analysis_metadata=deepcopy(check.PLAN),
        bound_receipts=bindings,
        screen_plan=plan,
        screen_plan_sha256=digest(plan),
        source_screen_manifest=prior,
        source_screen_manifest_sha256=digest(prior),
        generation_preflight=preflight,
    )
    original = deepcopy(manifest)
    old_archive = b"synthetic unused original archive"
    old_archive_sha = check.hashlib.sha256(old_archive).hexdigest()
    old_auditor_sha = check.hashlib.sha256(b"synthetic original auditor").hexdigest()
    old_freeze = dict(
        manifest_sha256=digest(original),
        source_sha256=manifest["source_sha256"],
        archive_sha256=old_archive_sha,
    )
    failed_review = dict(
        passed=False,
        gpu_launched=False,
        reservation_created=False,
        parent_manifest_sha256=digest(original),
        archive_byte_sha256=old_archive_sha,
        source_sha256=manifest["source_sha256"],
        auditor_byte_sha256=old_auditor_sha,
    )
    historical = {
        check.ORIGINAL_PARENT + "/manifest.json": canonical(original).encode(),
        check.ORIGINAL_PARENT + "/freeze.json": canonical(old_freeze).encode(),
        check.ORIGINAL_PARENT + "/source.tar.gz": old_archive,
        check.FAILED_REVIEW: canonical(failed_review).encode(),
    }
    payloads.update(historical)
    manifest["prelaunch_amendment"] = dict(
        revision="prelaunch_schema_fix_1",
        original_parent=check.ORIGINAL_PARENT,
        original_manifest_sha256=digest(original),
        scientific_contract_sha256=digest(original),
        original_freeze_byte_sha256=check.hashlib.sha256(
            historical[check.ORIGINAL_PARENT + "/freeze.json"]
        ).hexdigest(),
        original_archive_sha256=old_archive_sha,
        failed_review=check.FAILED_REVIEW,
        failed_review_byte_sha256=check.hashlib.sha256(historical[check.FAILED_REVIEW]).hexdigest(),
        original_auditor_byte_sha256=old_auditor_sha,
        corrected_auditor_byte_sha256=check.hashlib.sha256(
            payloads["scripts/audit_curiosity_suppression_screen.py"]
        ).hexdigest(),
        scientific_assignments_unchanged=True,
        original_local_execution_artifacts_absent=True,
    )
    run = root / "run"
    save(run / "manifest.json", manifest)
    with tarfile.open(run / "source.tar.gz", "w:gz") as archive:
        for name, data in payloads.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, BytesIO(data))
    frozen = dict(
        manifest_sha256=digest(manifest),
        source_sha256=manifest["source_sha256"],
        archive_sha256=check.sha(run / "source.tar.gz"),
        documents={
            **{b["path"]: b["byte_sha256"] for b in bindings},
            **{p: check.hashlib.sha256(v).hexdigest() for p, v in historical.items()},
        },
    )
    save(run / "freeze.json", frozen)
    descriptors, children = [], []
    for group in ("worker_0", "worker_1", "worker_2"):
        child = dict(
            manifest,
            execution_group=group,
            parent_manifest_sha256=digest(manifest),
            diagnostics=[a for a in assignments if a["execution_group"] == group],
        )
        child["assignment_sha256"] = digest(child["diagnostics"])
        path = root / group
        save(path / "manifest.json", child)
        descriptors.append(dict(path=str(path), execution_group=group, manifest_sha256=digest(child)))
        children.append(dict(execution_group=group, manifest_sha256=digest(child)))
    save(run / "children.json", descriptors)
    audit = dict(
        assay="curiosity_suppression_screen_audit",
        assigned=4320,
        assigned_choices=2160,
        terminal_generations=4320,
        terminal_receipt_slots=2160,
        complete=True,
        integrity_passed=True,
        parent_manifest_sha256=digest(manifest),
        source_sha256=manifest["source_sha256"],
        auditor_byte_sha256=check.hashlib.sha256(
            payloads["scripts/audit_curiosity_suppression_screen.py"]
        ).hexdigest(),
        outcomes=rows,
        outcomes_sha256=digest(rows),
        readiness=ready,
        passed=not failed,
        provenance=dict(freeze=frozen, bound_receipts=bindings),
        children=children,
        **dict.fromkeys(check.GATES, True),
    )
    audit_path = root / "audit.json"
    save(audit_path, audit)
    result = analyzer.analyze_audit(manifest, audit)
    result["analyzer_byte_sha256"] = check.hashlib.sha256(
        payloads["scripts/analyze_curiosity_suppression_screen.py"]
    ).hexdigest()
    result["audit_byte_sha256"] = check.sha(audit_path)
    out = root / "analysis"
    out.mkdir()
    result["exports"] = {}
    for key in check.EXPORTS:
        path = out / f"{key.replace('_', '-')}.csv"
        analyzer.write_csv(path, result[key])
        result["exports"][key] = dict(path=path.name, byte_sha256=check.sha(path), rows=len(result[key]))
    save(out / "analysis.json", result)
    return run, out / "analysis.json", audit_path


@pytest.fixture(scope="module")
def artifact_template(tmp_path_factory):
    root = tmp_path_factory.mktemp("curiosity_checker")
    artifact_tree(root)
    return root


def pin_fixture(monkeypatch, run):
    manifest = json.loads((run / "manifest.json").read_text())
    amendment = manifest["prelaunch_amendment"]
    monkeypatch.setattr(check, "LOCKED_PARENT_SHA256", digest(manifest))
    monkeypatch.setattr(check, "LOCKED_SOURCE_SHA256", manifest["source_sha256"])
    monkeypatch.setattr(check, "LOCKED_ARCHIVE_SHA256", check.sha(run / "source.tar.gz"))
    monkeypatch.setattr(check, "ORIGINAL_PARENT_SHA256", amendment["original_manifest_sha256"])
    monkeypatch.setattr(check, "ORIGINAL_ARCHIVE_SHA256", amendment["original_archive_sha256"])
    monkeypatch.setattr(check, "FAILED_REVIEW_SHA256", amendment["failed_review_byte_sha256"])


@pytest.fixture
def artifacts(tmp_path, artifact_template, monkeypatch):
    shutil.copytree(artifact_template, tmp_path / "artifacts")
    root = tmp_path / "artifacts"
    run = root / "run"
    descriptors = json.loads((run / "children.json").read_text())
    for descriptor in descriptors:
        descriptor["path"] = str(root / descriptor["execution_group"])
    save(run / "children.json", descriptors)
    pin_fixture(monkeypatch, run)
    return run, root / "analysis/analysis.json", root / "audit.json"


def test_complete_artifacts_and_every_export_reproduce(artifacts):
    result = check.check(*artifacts)
    assert result["passed"] is True
    assert result["decision"]["positive_screen"] is True
    assert result["maximum_absolute_error"] <= 1e-12
    assert result["compared_values"] > 500000
    assert result["tables"]["decision_bounds"] == 117
    assert "GPU geometry" in result["excluded_scope"]


def test_failed_readiness_is_a_valid_checked_result(tmp_path, monkeypatch):
    paths = artifact_tree(tmp_path, failed=True)
    pin_fixture(monkeypatch, paths[0])
    result = check.check(*paths)
    assert result["passed"] and not result["audit_readiness_passed"]
    assert result["decision"]["classification"] == "readiness_failed"


@pytest.mark.parametrize("mode", ("science", "failed_receipt", "corrected_auditor", "supplement", "launched"))
def test_amendment_cannot_change_science_or_erase_failed_unused_original(artifacts, monkeypatch, mode):
    run = artifacts[0]
    manifest = json.loads((run / "manifest.json").read_text())
    frozen = json.loads((run / "freeze.json").read_text())
    with tarfile.open(run / "source.tar.gz") as archive:
        payloads = {m.name: archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}
    if mode == "science":
        manifest["analysis_metadata"]["minimum_panel_difference"] = 0.2
    elif mode == "corrected_auditor":
        manifest["prelaunch_amendment"]["corrected_auditor_byte_sha256"] = "0" * 64
    elif mode == "supplement":
        del frozen["documents"][check.FAILED_REVIEW]
    else:
        failed = json.loads(payloads[check.FAILED_REVIEW])
        failed["passed" if mode == "failed_receipt" else "gpu_launched"] = True
        payloads[check.FAILED_REVIEW] = canonical(failed).encode()
        changed_sha = check.hashlib.sha256(payloads[check.FAILED_REVIEW]).hexdigest()
        frozen["documents"][check.FAILED_REVIEW] = changed_sha
        manifest["prelaunch_amendment"]["failed_review_byte_sha256"] = changed_sha
        monkeypatch.setattr(check, "FAILED_REVIEW_SHA256", changed_sha)

    class Archive:
        def extractfile(self, name):
            return BytesIO(payloads[name])

    with pytest.raises(ValueError):
        check.verify_amendment(Archive(), frozen, manifest)


@pytest.mark.parametrize(
    "mode",
    (
        "interval",
        "world",
        "denominator",
        "bound",
        "classification",
        "control",
        "audit_hash",
        "export",
        "archive",
        "child",
        "source_pin",
        "frozen_analyzer",
    ),
)
def test_rehashed_arithmetic_and_provenance_tampering_rejected(artifacts, mode):
    run, analysis_path, audit_path = artifacts
    value = json.loads(analysis_path.read_text())
    if mode == "interval":
        value["behavior_contrasts"][0]["ci95"][0] += 0.1
    elif mode == "world":
        value["world_contrasts"][0]["paired_difference"] += 0.5
    elif mode == "denominator":
        value["denominators"][0]["assigned_choices"] -= 1
    elif mode == "bound":
        value["decision_bounds"][0]["lower_bound"] += 0.1
    elif mode == "classification":
        value["decision"]["classification"] = "no_substantial_primary_development_effect"
    elif mode == "control":
        del value["decision"]["controls"]["valence_negative"]
    elif mode == "audit_hash":
        value["audit_sha256"] = "0" * 64
    elif mode == "frozen_analyzer":
        value["analyzer_byte_sha256"] = "0" * 64
    elif mode == "source_pin":
        manifest_path = run / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["source_sha256"] = "0" * 64
        save(manifest_path, manifest)
    elif mode == "child":
        path = run.parent / "worker_1/manifest.json"
        child = json.loads(path.read_text())
        child["diagnostics"].pop()
        child["assignment_sha256"] = digest(child["diagnostics"])
        save(path, child)
    elif mode == "archive":
        path = run / "source.tar.gz"
        path.write_bytes(path.read_bytes() + b"tampered")
    else:
        export = value["exports"]["decision_bounds"]
        path = analysis_path.parent / export["path"]
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["lower_bound"] = "0.999"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        export["byte_sha256"] = check.sha(path)
    save(analysis_path, value)
    with pytest.raises(ValueError):
        check.check(run, analysis_path, audit_path)


def test_cli_refuses_existing_output_before_reading_inputs(tmp_path, monkeypatch):
    path = tmp_path / "existing.json"
    path.write_text("unchanged")
    monkeypatch.setattr(
        sys,
        "argv",
        ["checker", "--run", "missing", "--analysis", "missing", "--audit", "missing", "--out", str(path)],
    )
    with pytest.raises(ValueError, match="immutable"):
        check.main()
    assert path.read_text() == "unchanged"
