"""Assignment-only fixtures exercise prospective exclusion and audit coverage."""

import copy
import hashlib
from pathlib import Path
import sys

import pytest

from curiosity_study import information_state, relational_information
from curiosity_study.artifacts import digest, read, write_once

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_replay_interventions import integrity_gate  # noqa: E402
from prepare_replay_interventions import (  # noqa: E402
    audit_inspected_development_registry,
    inspected_development_registry,
    split_provenance,
)


def collection(module=information_state, seed=420, items=2, **kwargs):
    rows = module.assignments(items, seed=seed, kinds=("new", "repeat"), **kwargs)
    for row in rows:
        row["representation_split"] = "extraction" if row["index"] == 0 else "development"
    return dict(
        diagnostics=rows,
        assignment_sha256=digest(rows),
        task_module_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
    )


def save_collection(path, manifest, role):
    write_once(path / "manifest.json", manifest)
    return dict(path=str(path), manifest_sha256=digest(manifest), role=role)


def fixture(tmp_path, registry=True):
    extraction = collection()
    source = collection(relational_information, seed=991)
    source.update(
        phase="validation",
        source_sha256="synthetic-source",
        validation_criteria=dict(minimum_distinct_items_per_cell=100),
    )
    save_collection(tmp_path / "extraction", extraction, "axis_extraction")
    save_collection(tmp_path / "source", source, "source")
    write_once(
        tmp_path / "source-audit.json",
        dict(
            manifest_sha256=digest(source),
            source_sha256=source["source_sha256"],
            complete=True,
            assigned=len(source["diagnostics"]),
            gates=dict(passed=True),
            validation_criteria=source["validation_criteria"],
        ),
    )
    config = dict(
        inference_scope="confirmation",
        direction_extraction_collection=str(tmp_path / "extraction"),
        source_collection=str(tmp_path / "source"),
        source_audit=str(tmp_path / "source-audit.json"),
        kinds=["new", "repeat"],
        world_index_range=[0, 2],
    )
    if registry:
        # No pilot records or audits exist. Even the failed pilot's full assignments
        # must be excluded, rather than only successful or completed branches.
        config["inspected_development_collections"] = [
            save_collection(
                tmp_path / f"pilot-{seed}",
                {**collection(relational_information, seed=seed), "synthetic_pilot_passed": False},
                role,
            )
            for seed, role in ((992, "failed_relational_smoke"), (993, "relational_development"))
        ]
    analysis = dict(manifest_sha256=digest(extraction), development_only=True)
    return config, source, analysis


def intervention_manifest(config, source, analysis, excluded=()):
    selected = source["diagnostics"]
    provenance = split_provenance(config, source, analysis, selected)
    return dict(
        inference_scope=config["inference_scope"],
        assigned_sources=len(selected),
        source_parent_manifest_sha256=digest(source),
        direction_analysis_sha256=digest(analysis),
        direction_extraction_manifest_sha256=analysis["manifest_sha256"],
        intervention_schedule=copy.deepcopy(config),
        intervention_schedule_sha256=digest(config),
        split_provenance=provenance,
        split_provenance_sha256=digest(provenance),
        prefixes={a["trial_id"]: {"assignment": a} for a in selected if a["trial_id"] not in excluded},
        excluded_sources=list(excluded),
    )


def gate(manifest):
    return integrity_gate(
        manifest,
        {"source": None},
        {"source"},
        [0],
        [0],
        [True],
        [True],
        [
            dict(
                requested_dose=1,
                projection_delta=1,
                delta_norm=1,
                relative_delta_norm=0.01,
                signed_engagement=True,
            )
        ],
    )


def test_confirmation_requires_explicit_inspection_declaration(tmp_path):
    config, source, analysis = fixture(tmp_path, registry=False)
    with pytest.raises(ValueError, match="explicit inspected_development_collections"):
        split_provenance(config, source, analysis, source["diagnostics"])
    config["inspected_development_collections"] = []
    manifest = intervention_manifest(config, source, analysis)
    assert manifest["split_provenance"]["inspected_development_collections"]["entries"] == []
    assert gate(manifest)["passed"]


def test_registry_binds_all_assignments_and_reaudits_without_outcomes(tmp_path):
    config, source, analysis = fixture(tmp_path)
    manifest = intervention_manifest(config, source, analysis)
    registry = manifest["split_provenance"]["inspected_development_collections"]
    assert registry["disjoint"] and registry["all_assigned_worlds_before_prefix_exclusions"]
    assert registry["declaration_sha256"] == digest(config["inspected_development_collections"])
    assert registry["entries_sha256"] == digest(registry["entries"])
    assert len(registry["entries"]) == 2
    for bound, declaration in zip(registry["entries"], config["inspected_development_collections"]):
        original = read(Path(declaration["path"]) / "manifest.json")
        assert bound["manifest_sha256"] == digest(original)
        assert bound["assignment_sha256"] == digest(original["diagnostics"])
        assert bound["assigned_sources"] == len(original["diagnostics"])
        assert len(bound["world_ids"]) == 4
        assert bound["semantic_world_ids"]
    assert gate(manifest)["passed"]


def test_failed_pilot_reuse_is_rejected_before_any_outcome_eligibility(tmp_path):
    config, source, analysis = fixture(tmp_path, registry=False)
    config["inspected_development_collections"] = [
        save_collection(
            tmp_path / "failed-pilot", {**source, "synthetic_pilot_passed": False}, "failed_pilot"
        )
    ]
    with pytest.raises(ValueError, match="overlaps an inspected development collection"):
        split_provenance(config, source, analysis, source["diagnostics"])


def test_semantic_duplicates_are_excluded_even_with_new_world_indices(tmp_path):
    reference = collection(items=1)
    source = collection(items=1, start_index=52)
    config = dict(
        inference_scope="development",
        inspected_development_collections=[save_collection(tmp_path / "pilot", reference, "failed_pilot")],
    )
    registry = inspected_development_registry(config, source, source["diagnostics"])
    entry = registry["entries"][0]
    assert not entry["overlap_world_ids"]
    assert entry["overlap_semantic_world_ids"]
    assert not registry["disjoint"]
    config["inference_scope"] = "confirmation"
    with pytest.raises(ValueError, match="overlaps an inspected development collection"):
        split_provenance(config, source, dict(manifest_sha256="unused"), source["diagnostics"])


def test_development_remains_compatible_but_records_declared_overlap(tmp_path):
    config, source, analysis = fixture(tmp_path, registry=False)
    config["inference_scope"] = "development"
    legacy = intervention_manifest(config, source, analysis)
    assert "inspected_development_collections" not in legacy["split_provenance"]
    assert gate(legacy)["passed"]
    config["inspected_development_collections"] = [
        dict(path=config["source_collection"], manifest_sha256=digest(source), role="inspected_development")
    ]
    manifest = intervention_manifest(config, source, analysis)
    assert not manifest["split_provenance"]["confirmation_eligible"]
    assert not manifest["split_provenance"]["inspected_development_collections"]["disjoint"]
    assert gate(manifest)["passed"]


@pytest.mark.parametrize(
    "mutation",
    [
        "drop_one",
        "drop_registry",
        "change_assignment_hash",
        "change_overlap",
        "change_role",
        "stale_schedule_hash",
    ],
)
def test_audit_rejects_registry_mutation_even_if_provenance_is_rehashed(tmp_path, mutation):
    config, source, analysis = fixture(tmp_path)
    manifest = intervention_manifest(config, source, analysis)
    provenance = manifest["split_provenance"]
    registry = provenance["inspected_development_collections"]
    if mutation == "drop_one":
        registry["entries"].pop()
    elif mutation == "drop_registry":
        del provenance["inspected_development_collections"]
    elif mutation == "change_assignment_hash":
        registry["entries"][0]["assignment_sha256"] = "altered"
    elif mutation == "change_overlap":
        registry["entries"][0]["overlap_world_ids"] = ["fabricated"]
    elif mutation == "change_role":
        registry["entries"][0]["role"] = "different_pilot"
    elif mutation == "stale_schedule_hash":
        manifest["intervention_schedule"]["inspected_development_collections"].pop()
    registry["entries_sha256"] = digest(registry["entries"])
    manifest["split_provenance_sha256"] = digest(provenance)
    result = gate(manifest)
    assert not result["passed"]
    assert not result["checks"]["inspected_development_registry_bound"]
    assert not result["checks"]["confirmation_nonoverlap"]


def test_audit_includes_prefix_exclusions_and_rejects_silent_selection_shrink(tmp_path):
    config, source, analysis = fixture(tmp_path)
    excluded = [a["trial_id"] for a in source["diagnostics"] if a["index"] == 1]
    manifest = intervention_manifest(config, source, analysis, excluded=excluded)
    registry = manifest["split_provenance"]["inspected_development_collections"]
    assert len(registry["evaluation_world_ids"]) == 4
    assert gate(manifest)["passed"]
    manifest["excluded_sources"] = []
    manifest["assigned_sources"] -= len(excluded)
    manifest["split_provenance"]["assigned_sources"] = manifest["assigned_sources"]
    manifest["split_provenance_sha256"] = digest(manifest["split_provenance"])
    assert not gate(manifest)["passed"]


@pytest.mark.parametrize(
    "problem", ["missing_manifest", "changed_manifest", "invalid_assignment_hash", "task_hash"]
)
def test_registry_fails_closed_for_missing_or_invalid_bound_material(tmp_path, problem):
    config, source, analysis = fixture(tmp_path)
    manifest = intervention_manifest(config, source, analysis)
    entry = config["inspected_development_collections"][0]
    path = Path(entry["path"]) / "manifest.json"
    if problem == "missing_manifest":
        path.unlink()
        assert not audit_inspected_development_registry(manifest)
        return
    original = read(path)
    if problem == "changed_manifest":
        original["synthetic_pilot_passed"] = True
    elif problem == "invalid_assignment_hash":
        original["assignment_sha256"] = "bad"
    else:
        original["task_module_sha256"] = "bad"
    path.unlink()
    write_once(path, original)
    assert not audit_inspected_development_registry(manifest)
    # Even rebinding the changed manifest does not forgive invalid assignments
    # or inability to reconstruct its semantic worlds under the frozen module.
    if problem != "changed_manifest":
        entry["manifest_sha256"] = digest(original)
        with pytest.raises(ValueError):
            inspected_development_registry(config, source, source["diagnostics"])


@pytest.mark.parametrize(
    "problem", ["not_list", "duplicate", "duplicate_path", "relative_path", "missing_role"]
)
def test_registry_rejects_ambiguous_declarations(tmp_path, problem):
    config, source, _ = fixture(tmp_path)
    entries = config["inspected_development_collections"]
    if problem == "not_list":
        config["inspected_development_collections"] = None
    elif problem == "duplicate":
        entries.append(copy.deepcopy(entries[0]))
    elif problem == "duplicate_path":
        entries[1]["path"] = entries[0]["path"]
    elif problem == "relative_path":
        entries[0]["path"] = "relative/path"
    else:
        del entries[0]["role"]
    with pytest.raises(ValueError):
        inspected_development_registry(config, source, source["diagnostics"])
