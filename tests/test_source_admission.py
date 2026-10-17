import copy
import importlib.util
from itertools import product
from pathlib import Path
import sys

import pytest

from curiosity_study import information_state as state
from curiosity_study.artifacts import digest, read, write_once

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("source_admission", SCRIPTS / "source_admission.py")
admission = importlib.util.module_from_spec(spec)
spec.loader.exec_module(admission)


def source_fixture(tmp_path, *, global_pass=False):
    kinds = (*admission.RETAINED_KINDS, "count")
    rows = state.assignments(100, seed=123, kinds=kinds)
    manifest = dict(
        assay="thinking_information",
        task_revision="equal-uncertainty-v1",
        phase="validation",
        source_sha256="frozen-code",
        diagnostics=rows,
        assignment_sha256=digest(rows),
        validation_criteria=copy.deepcopy(admission.CRITERIA),
    )
    cells = []
    for family, kind, known, query in product(("objects", "machines"), kinds, (0, 1), (0, 1)):
        failed = not global_pass and (family, kind, known, query) == ("machines", "count", 0, 1)
        invalid_repeat = (family, kind, known, query) == ("machines", "repeat", 1, 0)
        count = 93 if failed else 99 if invalid_repeat else 100
        cells.append(
            dict(
                family=family,
                kind=kind,
                known=known,
                query=query,
                n=100,
                items=100,
                completed=100,
                correct=count,
                valid=count,
                passed=not failed,
            )
        )
    audit = dict(
        manifest_sha256=digest(manifest),
        source_sha256=manifest["source_sha256"],
        complete=True,
        assigned=4000,
        correct=sum(c["correct"] for c in cells),
        valid=sum(c["valid"] for c in cells),
        validation_criteria=copy.deepcopy(admission.CRITERIA),
        gates=dict(passed=global_pass, cells=cells),
    )
    source, audit_path, amendment = tmp_path / "source", tmp_path / "audit.json", tmp_path / "amendment.md"
    write_once(source / "manifest.json", manifest)
    write_once(audit_path, audit)
    amendment.write_text("Post-inspection exploratory scope, including representation inspection.\n")
    config = dict(
        source_collection=str(source),
        source_audit=str(audit_path),
        inference_scope="development",
        kinds=list(admission.RETAINED_KINDS),
        world_index_range=[0, 100],
    )
    return manifest, audit, amendment, config


def certified_fixture(tmp_path):
    manifest, audit, amendment, config = source_fixture(tmp_path)
    certificate = admission.build_certificate(config["source_collection"], config["source_audit"], amendment)
    path = tmp_path / "certificate.json"
    write_once(path, certificate)
    config["source_eligibility_certificate"] = dict(path=str(path), sha256=digest(certificate))
    return manifest, audit, config, certificate


def replay_manifest(config, source, audit):
    selected = admission.selected_assignments(config, source)
    receipt = admission.admit_source(config, source, audit, selected)
    # An unavailable prefix remains an explicitly assigned source, never drops
    # from the admission denominator or the source coverage audit.
    excluded = [selected[-1]["trial_id"]]
    diagnostics = [
        dict(trial_id=f"replay-{a['trial_id']}", prefix_id=a["trial_id"], arm="baseline")
        for a in selected[:-1]
    ]
    return dict(
        assay="replay_information",
        phase=config.get("phase", "development"),
        inference_scope=config["inference_scope"],
        assignments=[],
        diagnostics=diagnostics,
        assignment_sha256=digest(diagnostics),
        intervention_schedule=config,
        intervention_schedule_sha256=digest(config),
        source_parent_manifest_sha256=digest(source),
        source_admission=receipt,
        source_admission_sha256=digest(receipt),
        assigned_sources=len(selected),
        excluded_sources=excluded,
        prefixes={a["trial_id"]: dict(assignment=a) for a in selected[:-1]},
    )


def test_certificate_retains_all_assignments_and_original_failed_cells(tmp_path):
    source, audit, config, certificate = certified_fixture(tmp_path)
    assert certificate["retained"]["assigned"] == 3200
    assert certificate["excluded"]["assigned"] == 800
    assert len(certificate["retained"]["cells"]) == 32
    assert len(certificate["excluded"]["cells"]) == 8
    assert certificate["original_gates"] == audit["gates"]
    assert certificate["original_global_gates_passed"] is False
    assert certificate["decision_context"]["representation_results_inspected_before_final_scope_decision"]
    assert not certificate["confirmation_eligible"]
    assert sum(c["valid"] for c in certificate["retained"]["cells"]) == 3199
    assert set(certificate["retained"]["trial_ids"]) == {
        a["trial_id"] for a in source["diagnostics"] if a["kind"] != "count"
    }
    assert read(config["source_audit"]) == audit  # Original failure is untouched.
    assert admission.audit_source_admission(replay_manifest(config, source, audit))


def test_failed_global_source_is_closed_by_default_and_subsets_are_explicit(tmp_path):
    source, audit, config, _ = certified_fixture(tmp_path)
    config.pop("source_eligibility_certificate")
    with pytest.raises(ValueError, match="must pass"):
        admission.admit_source(config, source, audit, admission.selected_assignments(config, source))
    config["source_eligibility_certificate"] = dict(
        path=str(tmp_path / "certificate.json"), sha256=digest(read(tmp_path / "certificate.json"))
    )
    config.update(kinds=["new", "repeat"], world_index_range=[50, 100])
    manifest = replay_manifest(config, source, audit)
    assert manifest["assigned_sources"] == 800
    assert manifest["source_admission"]["retained_assigned_sources"] == 3200
    assert admission.audit_source_admission(manifest)


@pytest.mark.parametrize("posthoc", [False, True])
def test_confirmation_always_rejects_scoped_certificate(tmp_path, posthoc):
    source, audit, config, _ = certified_fixture(tmp_path)
    config.update(inference_scope="confirmation", posthoc=posthoc)
    with pytest.raises(ValueError, match="forbidden for confirmation"):
        admission.admit_source(config, source, audit, admission.selected_assignments(config, source))


@pytest.mark.parametrize("phase", ["confirmation", "fresh_confirmation"])
def test_confirmation_phase_cannot_be_labeled_as_development_scope(tmp_path, phase):
    source, audit, config, _ = certified_fixture(tmp_path)
    manifest = replay_manifest(config, source, audit)
    config["phase"] = phase
    manifest["intervention_schedule_sha256"] = digest(config)
    assert manifest["inference_scope"] == config["inference_scope"] == "development"
    with pytest.raises(ValueError, match="forbidden for confirmation"):
        admission.admit_source(config, source, audit, admission.selected_assignments(config, source))
    assert not admission.audit_source_admission(manifest)

    # Changing only the envelope phase cannot bypass the schedule-level guard.
    config.pop("phase")
    manifest["intervention_schedule_sha256"] = digest(config)
    manifest["phase"] = phase
    assert not admission.audit_source_admission(manifest)


@pytest.mark.parametrize(
    "change",
    [
        {"patch_mode": "project"},
        {"patch_mode": "matched_add"},
        {"patch_mode": "replace"},
        {"patch_mode": None},
        {"donor_source_trial_id": "other-source"},
        {"donor_role": "same_information"},
        {"donor_vector_sha256": None},
        {"donors": {}},
        {"projection_target": 0.0},
        {"matched_interchange_condition": "opposite_gain"},
        {"arm": "donor"},
    ],
)
def test_scoped_actual_diagnostics_reject_projection_and_donors_without_design_tag(tmp_path, change):
    source, audit, config, _ = certified_fixture(tmp_path)
    manifest = replay_manifest(config, source, audit)
    manifest["diagnostics"][-1].update(arm="direction", patch_mode="add")
    manifest["diagnostics"][-1].update(change)
    manifest["assignment_sha256"] = digest(manifest["diagnostics"])
    assert "design" not in manifest
    assert not admission.audit_source_admission(manifest)


def test_scoped_schedule_checks_both_assignment_lists_and_requires_actual_arms(tmp_path):
    source, audit, config, _ = certified_fixture(tmp_path)
    manifest = replay_manifest(config, source, audit)
    manifest["assignments"] = [dict(arm="direction", patch_mode="project")]
    manifest["assignment_sha256"] = digest(manifest["assignments"] + manifest["diagnostics"])
    assert not admission.audit_source_admission(manifest)
    manifest["assignments"] = []
    manifest["diagnostics"] = []
    manifest["assignment_sha256"] = digest([])
    assert not admission.audit_source_admission(manifest)
    del manifest["diagnostics"]
    assert not admission.audit_source_admission(manifest)


def test_scoped_implicit_and_explicit_additive_arms_keep_certificate_unchanged(tmp_path):
    source, audit, config, certificate = certified_fixture(tmp_path)
    certificate_path = Path(config["source_eligibility_certificate"]["path"])
    before = certificate_path.read_bytes()
    manifest = replay_manifest(config, source, audit)
    manifest["diagnostics"][-3].update(arm="sham")
    manifest["diagnostics"][-2].update(arm="direction", patch_mode="add")
    manifest["diagnostics"][-1].update(arm="random", patch_mode="add")
    manifest["assignment_sha256"] = digest(manifest["diagnostics"])
    assert admission.audit_source_admission(manifest)
    assert certificate_path.read_bytes() == before
    assert digest(certificate) == config["source_eligibility_certificate"]["sha256"]
    assert admission.build_certificate(
        config["source_collection"],
        config["source_audit"],
        certificate["amendment"]["path"],
        certificate["created_at_utc"],
    ) == certificate


def test_scoped_count_and_success_filtered_selection_are_rejected(tmp_path):
    source, audit, config, _ = certified_fixture(tmp_path)
    selected = admission.selected_assignments(config, source)
    with pytest.raises(ValueError, match="exactly the configured"):
        admission.admit_source(config, source, audit, selected[:-1])
    config["kinds"] = ["count"]
    with pytest.raises(ValueError, match="count is excluded"):
        admission.admit_source(config, source, audit, admission.selected_assignments(config, source))


@pytest.mark.parametrize(
    "problem", ["missing_cell", "duplicate_cell", "weak_threshold", "retained_failure", "missing_assignment"]
)
def test_certificate_recomputes_criteria_and_complete_coverage(tmp_path, problem):
    source, audit, _, _ = source_fixture(tmp_path)
    if problem == "missing_cell":
        audit["gates"]["cells"].pop()
    elif problem == "duplicate_cell":
        audit["gates"]["cells"].append(audit["gates"]["cells"][0])
    elif problem == "weak_threshold":
        source["validation_criteria"]["minimum_valid_format"] = 0.9
        audit["validation_criteria"]["minimum_valid_format"] = 0.9
    elif problem == "retained_failure":
        c = next(c for c in audit["gates"]["cells"] if c["kind"] == "repeat")
        c.update(correct=93, valid=93, passed=False)
    else:
        source["diagnostics"].pop()
        source["assignment_sha256"] = digest(source["diagnostics"])
        audit["assigned"] -= 1
    audit["manifest_sha256"] = digest(source)
    with pytest.raises(ValueError):
        admission.scoped_evidence(source, audit)


@pytest.mark.parametrize(
    "problem",
    ["receipt", "certificate", "amendment", "coverage", "configuration", "scope", "projection", "natural"],
)
def test_independent_admission_audit_detects_tampering(tmp_path, problem):
    source, audit, config, certificate = certified_fixture(tmp_path)
    manifest = replay_manifest(config, source, audit)
    if problem == "receipt":
        manifest["source_admission"]["original_global_gates_passed"] = True
        manifest["source_admission_sha256"] = digest(manifest["source_admission"])
    elif problem == "certificate":
        certificate["retained"]["trial_ids"].pop()
        path = tmp_path / "tampered.json"
        write_once(path, certificate)
        config["source_eligibility_certificate"] = dict(path=str(path), sha256=digest(certificate))
        manifest["intervention_schedule_sha256"] = digest(config)
    elif problem == "amendment":
        Path(certificate["amendment"]["path"]).write_text("Changed decision scope")
    elif problem == "coverage":
        manifest["excluded_sources"].clear()
        manifest["assigned_sources"] -= 1
    elif problem == "configuration":
        config["world_index_range"] = [50, 100]
        manifest["intervention_schedule_sha256"] = digest(config)
    elif problem == "scope":
        config["inference_scope"] = manifest["inference_scope"] = "confirmation"
        manifest["intervention_schedule_sha256"] = digest(config)
    elif problem == "projection":
        manifest["design"] = "projection_interchange_v1"
    else:
        manifest["assay"] = "natural_information"
    assert not admission.audit_source_admission(manifest)


def test_full_passing_default_and_legacy_manifest_stay_available(tmp_path):
    source, audit, _, config = source_fixture(tmp_path, global_pass=True)
    config["inference_scope"] = "confirmation"
    manifest = replay_manifest(config, source, audit)
    assert manifest["source_admission"]["mode"] == "full_global_validation"
    assert admission.audit_source_admission(manifest)
    del manifest["source_admission"], manifest["source_admission_sha256"]
    assert admission.audit_source_admission(manifest)
    manifest["source_audit_sha256"] = "wrong"
    assert not admission.audit_source_admission(manifest)


def test_full_global_admission_does_not_acquire_scoped_arm_or_phase_restrictions(tmp_path):
    source, audit, _, config = source_fixture(tmp_path, global_pass=True)
    config.update(phase="confirmation", inference_scope="confirmation")
    manifest = replay_manifest(config, source, audit)
    manifest["design"] = "projection_interchange_v1"
    manifest["diagnostics"][-1].update(
        arm="direction", patch_mode="project", donor_source_trial_id="other-source"
    )
    manifest["assignment_sha256"] = digest(manifest["diagnostics"])
    assert manifest["source_admission"]["mode"] == "full_global_validation"
    assert admission.audit_source_admission(manifest)


@pytest.mark.parametrize("problem", ["phase", "minimum", "explicit_exploratory"])
def test_passing_development_gates_cannot_become_confirmation(tmp_path, problem):
    source, audit, _, config = source_fixture(tmp_path, global_pass=True)
    if problem == "phase":
        source["phase"] = "development_validation"
    elif problem == "minimum":
        source["validation_criteria"]["minimum_distinct_items_per_cell"] = 20
        audit["validation_criteria"] = copy.deepcopy(source["validation_criteria"])
    else:
        audit["gates"]["exploratory_only"] = True
    audit["manifest_sha256"] = digest(source)
    selected = admission.selected_assignments(config, source)
    assert admission.admit_source(config, source, audit, selected)["mode"] == "global_development_gates"
    config["inference_scope"] = "confirmation"
    with pytest.raises(ValueError, match="full validation"):
        admission.admit_source(config, source, audit, selected)
