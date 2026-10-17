"""Synthetic/read-only preparation contracts; never freeze or launch a study."""

from copy import deepcopy
import hashlib
from pathlib import Path
import sys

import pytest

from curiosity_study.artifacts import canonical, digest
from curiosity_study import satisfaction_causal_sites as study
from curiosity_study import satisfaction_site_execution as execution
from test_satisfaction_site_execution import frozen as executor_fixture

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_satisfaction_causal_sites as scientific  # noqa: E402
import prepare_satisfaction_site_setup as setup  # noqa: E402

site_materials = executor_fixture


def save(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical(value) + "\n")
    return path


def bytefile(root, name, data=b"synthetic archive bytes"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def build(frozen, monkeypatch, **changed):
    monkeypatch.setattr(scientific, "source_hash", lambda: "a" * 64)
    plan = frozen["plan"]
    arguments = dict(source=frozen["source"], source_cert={"synthetic_source": True},
        setup_cert=dict(dose_reference=plan["dose_reference"]), reserved=frozen["materials"],
        preflight=frozen["preflight"], receipt_check=frozen["receipt"],
        candidate={"synthetic_candidate": True}, inventory={"synthetic_inventory": True},
        estimate={"synthetic_forecast": True}, exposure={"synthetic_exposure": True}, bindings=[],
        selection_receipt=None if plan["phase"] == "calibration" else dict(selected_site=47))
    arguments.update(changed)
    return scientific.build_manifests(plan, **arguments)


def test_parent_builder_reconciles_all_workers_without_writing(site_materials, monkeypatch, tmp_path):
    frozen = site_materials
    before = digest(frozen["plan"]), digest(frozen["materials"]), digest(frozen["source"])
    parent, children = build(frozen, monkeypatch)
    expected = 4032 if frozen["plan"]["phase"] == "calibration" else 2880
    assert len(parent["diagnostics"]) == expected
    assert [child["execution_group"] for child in children] == list(study.GROUPS)
    assert all(child["parent_manifest_sha256"] == digest(parent) for child in children)
    assert sum(len(execution.validate_manifest(child)) for child in children) == expected
    assert {row["trial_id"] for child in children for row in child["diagnostics"]} == {
        row["trial_id"] for row in parent["diagnostics"]}
    assert parent["receipt_grouping"] == execution.RECEIPT_GROUPING
    assert parent["setup_certificate"]["dose_reference"] == parent["site_plan"]["dose_reference"]
    assert before == (digest(frozen["plan"]), digest(frozen["materials"]), digest(frozen["source"]))
    assert list(tmp_path.iterdir()) == []
    # Returned workers own their nested assignments rather than aliasing parent/source state.
    children[0]["diagnostics"][0]["site"] = 15
    assert parent["diagnostics"][0]["site"] in study.SITES
    assert digest(frozen["plan"]) == before[0]


@pytest.mark.parametrize("mutation", ["dose", "reserved", "request", "receipt", "selection"])
def test_parent_builder_rejects_incompatible_inputs_before_publication(site_materials, monkeypatch, mutation):
    frozen = site_materials
    kwargs = {}
    if mutation == "dose":
        certificate = deepcopy(dict(dose_reference=frozen["plan"]["dose_reference"]))
        certificate["dose_reference"]["doses"]["31"] += 1
        kwargs["setup_cert"] = certificate
    elif mutation == "reserved":
        value = deepcopy(frozen["materials"])
        value["base_assignments"][0]["world_slot"] ^= 1
        value["base_assignment_sha256"] = digest(value["base_assignments"])
        kwargs["reserved"] = value
    elif mutation == "request":
        value = deepcopy(frozen["preflight"])
        value["requests"][0]["request"]["seed"] += 1
        value["requests_sha256"] = digest(value["requests"])
        kwargs["preflight"] = value
    elif mutation == "receipt":
        value = deepcopy(frozen["receipt"])
        value["non_stop_bindings"].pop()
        kwargs["receipt_check"] = value
    else:
        if frozen["plan"]["phase"] != "validation":
            pytest.skip("Only validation consumes a selected site")
        kwargs["selection_receipt"] = dict(selected_site=31)
    with pytest.raises((ValueError, KeyError)):
        build(frozen, monkeypatch, **kwargs)


def certificate_inputs(tmp_path, monkeypatch):
    import audit_satisfaction_site_setup as auditor

    medians = {"15": 100.0, "31": 150.0, "47": 200.0}
    reference = dict(formula=study.DOSE_FORMULA, original_dose=study.ORIGINAL_S_DOSE,
        base_branches=288, source_screen_manifest_sha256=study.PRIOR_MANIFEST_SHA256,
        capture_sha256="a" * 64, median_record_norm=medians,
        doses={str(s): study.ORIGINAL_S_DOSE * medians[str(s)] / medians["15"] for s in study.SITES})
    manifest = dict(source_sha256="b" * 64, synthetic=True)
    archive_sha = bytefile(tmp_path, scientific.SETUP + "/source.tar.gz")
    frozen = dict(manifest_sha256=digest(manifest), source_sha256=manifest["source_sha256"], archive_sha256=archive_sha)
    receipt = dict(complete=True, passed=True, full_cap_certified=False, assigned=544,
        manifest_sha256=digest(manifest), source_sha256=manifest["source_sha256"],
        dose_reference=reference, dose_reference_sha256=digest(reference))
    save(tmp_path, scientific.SETUP + "/manifest.json", manifest)
    save(tmp_path, scientific.SETUP + "/freeze.json", frozen)
    save(tmp_path, scientific.SETUP_AUDIT, receipt)
    monkeypatch.setattr(auditor, "audit", lambda root: deepcopy(receipt))
    return manifest, frozen, receipt


def test_setup_certificate_rechecks_short_scope_and_exact_artifacts(tmp_path, monkeypatch):
    manifest, frozen, receipt = certificate_inputs(tmp_path, monkeypatch)
    certificate = scientific.setup_certificate(tmp_path)
    assert certificate["manifest_sha256"] == digest(manifest)
    assert certificate["freeze_sha256"] == digest(frozen)
    assert certificate["audit_sha256"] == digest(receipt)
    assert certificate["dose_reference"] == receipt["dose_reference"]
    assert certificate["passed"] is True and certificate["full_cap_certified"] is False
    assert "128-token" in certificate["scope"]


@pytest.mark.parametrize("mutation", ["failed", "incomplete", "full_cap", "count", "dose", "archive", "audit_bytes"])
def test_setup_certificate_rejects_failures_and_rebound_fabrications(tmp_path, monkeypatch, mutation):
    _, _, receipt = certificate_inputs(tmp_path, monkeypatch)
    if mutation in ("failed", "incomplete", "full_cap", "count"):
        key, value = {"failed": ("passed", False), "incomplete": ("complete", False),
                      "full_cap": ("full_cap_certified", True), "count": ("assigned", 543)}[mutation]
        receipt[key] = value
        save(tmp_path, scientific.SETUP_AUDIT, receipt)
    elif mutation == "dose":
        receipt["dose_reference"]["doses"]["31"] += 1
        receipt["dose_reference_sha256"] = digest(receipt["dose_reference"])
        save(tmp_path, scientific.SETUP_AUDIT, receipt)
    elif mutation == "archive":
        bytefile(tmp_path, scientific.SETUP + "/source.tar.gz", b"changed")
    else:
        altered = deepcopy(receipt)
        altered["new_unverified_field"] = True
        save(tmp_path, scientific.SETUP_AUDIT, altered)
    with pytest.raises((ValueError, KeyError)):
        scientific.setup_certificate(tmp_path)


def selection_inputs(tmp_path, monkeypatch, *, phase="calibration", sites=(31, 47), positive=True,
                     complete=True, independent_matches=True):
    import check_satisfaction_site_decision as checker

    manifest = dict(assay=study.ASSAY, site_phase=phase, site_plan=dict(phase=phase, sites=list(sites)),
                    source_sha256="a" * 64)
    audited = dict(parent_manifest_sha256=digest(manifest), complete=complete, integrity_passed=True)
    decision = dict(selected_site=31 if positive else None, positive_screen=positive,
                    classification="calibration_qualified_site" if positive else "calibration_no_qualified_site")
    result = dict(parent_manifest_sha256=digest(manifest), audit_sha256=digest(audited), decision=decision)
    save(tmp_path, scientific.CALIBRATION + "/manifest.json", manifest)
    save(tmp_path, scientific.CALIBRATION_AUDIT, audited)
    save(tmp_path, scientific.CALIBRATION_ANALYSIS, result)
    archive_sha = bytefile(tmp_path, scientific.CALIBRATION + "/source.tar.gz")
    frozen = dict(manifest_sha256=digest(manifest), source_sha256=manifest["source_sha256"], archive_sha256=archive_sha)
    save(tmp_path, scientific.CALIBRATION + "/freeze.json", frozen)
    checked = dict(passed=True, parent_manifest_sha256=digest(manifest), audit_sha256=digest(audited),
        analysis_sha256=digest(result), analysis_byte_sha256=scientific.sha(tmp_path / scientific.CALIBRATION_ANALYSIS),
        audit_byte_sha256=scientific.sha(tmp_path / scientific.CALIBRATION_AUDIT), decision=deepcopy(decision),
        recomputed_decision=deepcopy(decision), assigned=4032, compared_values=4000, checker_byte_sha256="b" * 64)
    save(tmp_path, scientific.CALIBRATION_CHECK, checked)
    reproduced = deepcopy(checked)
    if not independent_matches:
        reproduced["recomputed_decision"]["selected_site"] = 47
    calls = []

    def recheck(*paths):
        calls.append(paths)
        return deepcopy(reproduced)

    monkeypatch.setattr(checker, "check", recheck)
    return manifest, audited, result, checked, frozen, calls


def test_selection_reproduces_independent_decision_and_retains_freeze_binding(tmp_path, monkeypatch):
    manifest, audit, analysis, check, frozen, calls = selection_inputs(tmp_path, monkeypatch)
    receipt = scientific.selection_certificate(tmp_path)
    assert calls == [(tmp_path / scientific.CALIBRATION, tmp_path / scientific.CALIBRATION_ANALYSIS,
                      tmp_path / scientific.CALIBRATION_AUDIT)]
    assert receipt["selected_site"] == 31 and receipt["decision"] == analysis["decision"]
    assert receipt["manifest_sha256"] == digest(manifest) and receipt["audit_sha256"] == digest(audit)
    assert receipt["independent_check_sha256"] == digest(check)
    assert receipt["freeze_sha256"] == digest(frozen)
    assert receipt["archive_byte_sha256"] == frozen["archive_sha256"]


@pytest.mark.parametrize("mutation", ["wrong_phase", "subset", "negative", "incomplete", "archive",
                                      "stale_check_bytes", "independent_disagreement"])
def test_selection_rejects_self_consistent_wrong_scope_or_stale_evidence(tmp_path, monkeypatch, mutation):
    _, _, _, check, _, _ = selection_inputs(tmp_path, monkeypatch,
        phase="validation" if mutation == "wrong_phase" else "calibration",
        sites=(31,) if mutation == "subset" else (31, 47), positive=mutation != "negative",
        complete=mutation != "incomplete", independent_matches=mutation != "independent_disagreement")
    if mutation == "archive":
        bytefile(tmp_path, scientific.CALIBRATION + "/source.tar.gz", b"changed archive")
    elif mutation == "stale_check_bytes":
        check["audit_byte_sha256"] = "f" * 64
        save(tmp_path, scientific.CALIBRATION_CHECK, check)
    with pytest.raises((ValueError, KeyError)):
        scientific.selection_certificate(tmp_path)


def exposure_inputs(tmp_path, monkeypatch):
    import audit_voluntary_freshness as freshness

    requests = [dict(source_trial_id=f"old-{i}") for i in range(288)]
    source = dict(generation_preflight=dict(requests=[dict(request=r) for r in requests]))
    save(tmp_path, "runs/qwen32-satisfaction-screen-v1/manifest.json", source)
    old = dict(run="runs/old", manifest_sha256="a" * 64, execution_evidence="persisted_records")
    historical_sha = bytefile(tmp_path, "runs/old/source.tar.gz")
    outer_sha = bytefile(tmp_path, "runs/outer/source.tar.gz", b"outer archive")
    history = dict(historical_registry=[dict(old, input_exposure_included=True,
                                             explicitly_active_assigned_parent=False)],
        source_archive_bindings={"a" * 64: dict(archive="runs/old/source.tar.gz", archive_byte_sha256=historical_sha)},
        outer_source_bindings=[dict(run="runs/outer", archive_byte_sha256=outer_sha)])
    bundle = dict(history=history, history_sha256=digest(history), selection=dict(history_sha256=digest(history)))
    setup_manifest = dict(requests=requests, source_certificate=dict(reserved_materials_sha256=digest(bundle)))
    certificate = dict(run=scientific.SETUP, manifest_sha256=digest(setup_manifest))
    current = [old, dict(run=scientific.SETUP, manifest_sha256=digest(setup_manifest), execution_evidence="persisted_records")]
    manifests = {old["run"]: {"synthetic": True}, scientific.SETUP: setup_manifest}
    monkeypatch.setattr(freshness, "inventory", lambda root: (deepcopy(current), deepcopy(manifests)))
    return bundle, certificate, current, manifests


def test_incremental_exposure_only_adds_exact_known_setup_inputs(tmp_path, monkeypatch):
    bundle, certificate, _, _ = exposure_inputs(tmp_path, monkeypatch)
    receipt = scientific.incremental_exposure(tmp_path, bundle, setup_cert=certificate)
    assert receipt["passed"] and receipt["reserved_exact_overlap_count"] == 0
    assert receipt["allowed_known_world_runs"] == {certificate["run"]: certificate["manifest_sha256"]}


@pytest.mark.parametrize("mutation", ["old_registry", "unknown_executed", "request", "reserved", "archive", "outer_archive"])
def test_incremental_exposure_rejects_changed_or_unreviewed_sources(tmp_path, monkeypatch, mutation):
    bundle, certificate, current, manifests = exposure_inputs(tmp_path, monkeypatch)
    if mutation == "old_registry":
        current[0]["manifest_sha256"] = "c" * 64
    elif mutation == "unknown_executed":
        current.append(dict(run="runs/unreviewed", execution_evidence="persisted_records"))
        manifests["runs/unreviewed"] = {"assay": "unknown"}
    elif mutation in ("request", "reserved"):
        altered = manifests[certificate["run"]]
        if mutation == "request":
            altered["requests"][0]["source_trial_id"] = "new-world"
        else:
            altered["source_certificate"]["reserved_materials_sha256"] = "f" * 64
        certificate["manifest_sha256"] = digest(altered)
    else:
        bytefile(tmp_path, "runs/" + ("outer" if mutation == "outer_archive" else "old") + "/source.tar.gz", b"changed")
    with pytest.raises(ValueError):
        scientific.incremental_exposure(tmp_path, bundle, setup_cert=certificate)


def test_prepared_unknown_is_not_execution_but_transition_during_check_rejects(tmp_path, monkeypatch):
    import audit_voluntary_freshness as freshness

    bundle, certificate, current, manifests = exposure_inputs(tmp_path, monkeypatch)
    current.append(dict(run="runs/prepared", execution_evidence="none"))
    manifests["runs/prepared"] = {"assay": "unknown"}
    assert scientific.incremental_exposure(tmp_path, bundle, setup_cert=certificate)["passed"]
    count = 0

    def transition(root):
        nonlocal count
        count += 1
        rows = deepcopy(current)
        if count > 1:
            rows[-1]["execution_evidence"] = "persisted_records"
        return rows, deepcopy(manifests)

    monkeypatch.setattr(freshness, "inventory", transition)
    with pytest.raises(ValueError):
        scientific.incremental_exposure(tmp_path, bundle, setup_cert=certificate)


@pytest.mark.parametrize("phase,available,accepted", [("setup", 65, True), ("setup", 64.99, False),
    ("calibration", 62, True), ("calibration", 61.99, False), ("validation", 28, True), ("validation", 27.99, False)])
def test_forecast_reserves_only_remaining_fixed_stage_ceilings(tmp_path, monkeypatch, phase, available, accepted):
    import budget

    monkeypatch.setattr(budget, "snapshot", lambda: dict(all_stopped_zero_tasks=True,
                                                       available_compute_estimate_usd=available))
    for suffix, cohorts in (("baseline-zero", 18), ("positive-half", 9), ("positive-full", 9)):
        save(tmp_path, f"{setup.SOURCE}-{suffix}/cloud_profiles/synthetic.json",
             dict(assigned=cohorts * 32, completed=cohorts * 32, execution_seconds=cohorts * 90, load_seconds=12))
    if not accepted:
        with pytest.raises(ValueError):
            setup.forecast(tmp_path, phase=phase)
    else:
        value = setup.forecast(tmp_path, phase=phase)
        assert value["campaign_ceiling_usd"] == 65 and value["paid_retries"] == 0
        assert value["admission_phase"] == phase
        assert value["remaining_stage_reserve_usd"] == {"setup": 65, "calibration": 62, "validation": 28}[phase]
        assert value["policies"] == setup.resources.POLICIES
        assert sum(v["stage_ceiling_usd"] for v in value["policies"].values()) == 65


@pytest.mark.parametrize("field,value", [("execution_seconds", float("inf")), ("execution_seconds", True),
                                        ("load_seconds", float("inf")), ("assigned", 576.0)])
def test_forecast_rejects_nonfinite_or_wrongly_typed_profiles(tmp_path, monkeypatch, field, value):
    import budget

    monkeypatch.setattr(budget, "snapshot", lambda: dict(all_stopped_zero_tasks=True,
                                                       available_compute_estimate_usd=65))
    for suffix, cohorts in (("baseline-zero", 18), ("positive-half", 9), ("positive-full", 9)):
        row = dict(assigned=cohorts * 32, completed=cohorts * 32, execution_seconds=cohorts * 90, load_seconds=12)
        if suffix == "baseline-zero":
            row[field] = value
        save(tmp_path, f"{setup.SOURCE}-{suffix}/cloud_profiles/synthetic.json", row)
    with pytest.raises(ValueError):
        setup.forecast(tmp_path)


def priority_inputs(tmp_path, monkeypatch, *, positive=False, complete=True, stopped=True):
    manifest = dict(synthetic=True)
    monkeypatch.setattr(setup, "C_HASH", digest(manifest))
    archive_sha = bytefile(tmp_path, setup.C_RUN + "/source.tar.gz")
    frozen = dict(manifest_sha256=digest(manifest), archive_sha256=archive_sha)
    audited = dict(parent_manifest_sha256=digest(manifest), complete=complete, integrity_passed=True)
    decision = dict(classification="direction_selective_development_signal" if positive else "readiness_failed",
                    positive_screen=positive)
    analysis = dict(parent_manifest_sha256=digest(manifest), audit_sha256=digest(audited), decision=decision)
    values = dict(manifest=manifest, freeze=frozen, audit=audited, analysis=analysis,
        scoring=dict(parent_manifest_sha256=digest(manifest), passed=True, assigned=4320, checked=4320, complete=True),
        shutdown=dict(all_stopped_zero_tasks=stopped))
    for key, value in values.items():
        save(tmp_path, setup.C_FILES[key], value)
    values["arithmetic"] = dict(parent_manifest_sha256=digest(manifest), passed=True,
        analysis_sha256=digest(analysis), audit_sha256=digest(audited), decision=deepcopy(decision),
        analysis_byte_sha256=setup.sha(tmp_path / setup.C_FILES["analysis"]),
        audit_byte_sha256=setup.sha(tmp_path / setup.C_FILES["audit"]))
    save(tmp_path, setup.C_FILES["arithmetic"], values["arithmetic"])
    return values


def test_closed_negative_c_is_preserved_and_eligible_for_prespecified_branch(tmp_path, monkeypatch):
    values = priority_inputs(tmp_path, monkeypatch)
    receipt = setup.priority_gate(tmp_path)
    assert receipt["passed"] and receipt["preceding_decision"] == values["analysis"]["decision"]
    assert receipt["preceding_decision"]["classification"] == "readiness_failed"
    assert all(receipt["bindings"][key]["content_sha256"] == digest(values[key]) for key in setup.C_FILES)


@pytest.mark.parametrize("mutation", ["positive", "incomplete", "running", "check_bytes", "different_decision", "partial_scoring"])
def test_priority_gate_rejects_open_or_qualifying_c_without_reading_actual_outputs(tmp_path, monkeypatch, mutation):
    values = priority_inputs(tmp_path, monkeypatch, positive=mutation == "positive",
                             complete=mutation != "incomplete", stopped=mutation != "running")
    if mutation == "check_bytes":
        values["arithmetic"]["audit_byte_sha256"] = "f" * 64
    elif mutation == "different_decision":
        values["arithmetic"]["decision"]["classification"] = "different"
    elif mutation == "partial_scoring":
        values["scoring"]["checked"] = 4319
    for key in ("arithmetic", "scoring"):
        save(tmp_path, setup.C_FILES[key], values[key])
    with pytest.raises(ValueError):
        setup.priority_gate(tmp_path)


def test_setup_immutable_target_rejects_before_any_closed_outcome_read(tmp_path, monkeypatch):
    target = tmp_path / "runs/exists"
    target.mkdir(parents=True)
    monkeypatch.setattr(setup, "priority_gate", lambda root: pytest.fail("Should reject before reading C evidence"))
    with pytest.raises(ValueError, match="immutable"):
        setup.prepare(target, root=tmp_path)
    assert list(target.iterdir()) == []


def test_scientific_existing_execution_rejects_before_any_closed_outcome_read(tmp_path, monkeypatch):
    (tmp_path / "executions/already-frozen").mkdir(parents=True)
    monkeypatch.setattr(scientific, "priority_gate", lambda root: pytest.fail("Should reject before reading C evidence"))
    with pytest.raises(ValueError, match="immutable"):
        scientific.prepare(tmp_path / "runs/already-frozen", phase="calibration", root=tmp_path)
    assert not (tmp_path / "runs/already-frozen").exists()
