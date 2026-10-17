"""Admission preserves failed candidates and binds their independent evidence."""

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from curiosity_study.artifacts import digest
from curiosity_study.concept_capture import MODEL

SPEC = importlib.util.spec_from_file_location(
    "voluntary_preparer", Path(__file__).resolve().parents[1] / "scripts/prepare_voluntary_resolution.py"
)
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


def evidence():
    manifest = dict(model=MODEL, source_sha256="source")
    audit = dict(passed=True, complete=True, assigned=384, manifest_sha256=digest(manifest))
    analysis = dict(
        assay="described_state_candidate_analysis",
        development_only=True,
        model=MODEL,
        source_sha256="source",
        manifest_sha256=digest(manifest),
        audit_sha256=digest(audit),
        directions_sha256="directions",
        candidates=[
            dict(
                contrast=target,
                passed=True,
                a_selection_passed=True,
                b_transfer_passed=True,
                b_specificity_point_screen_passed=True,
                selected_layer=15,
                position="record",
            )
            for target in prepare.TARGETS
        ],
    )
    check = dict(
        passed=True,
        records=384,
        capture_manifest_sha256=digest(manifest),
        analysis_sha256=digest(analysis),
        analysis_byte_sha256="analysis-bytes",
        audit_sha256=digest(audit),
        directions_byte_sha256="directions",
    )
    return manifest, audit, analysis, check


def bind(parts, direction_sha="directions"):
    return prepare.candidate_binding(*parts, "analysis-bytes", direction_sha)


def test_matching_independent_evidence_admits_only_fixed_sites():
    result = bind(evidence())
    assert set(result["selected_sites"]) == set(prepare.TARGETS)
    assert all(site == dict(layer=15, position="record") for site in result["selected_sites"].values())


@pytest.mark.parametrize(
    "field,value",
    [
        ("passed", False),
        ("a_selection_passed", False),
        ("b_transfer_passed", False),
        ("b_specificity_point_screen_passed", False),
        ("selected_layer", 31),
        ("position", "final"),
    ],
)
def test_independent_receipt_cannot_override_failed_gate_or_reselect_site(field, value):
    parts = deepcopy(evidence())
    parts[2]["candidates"][0][field] = value
    parts[3]["analysis_sha256"] = digest(parts[2])
    with pytest.raises(ValueError, match="requires both"):
        bind(parts)


def test_changed_direction_file_refused():
    with pytest.raises(ValueError, match="direction bindings"):
        bind(evidence(), direction_sha="replacement")


def test_stale_independent_check_refused():
    parts = evidence()
    parts[3]["analysis_byte_sha256"] = "other-analysis"
    with pytest.raises(ValueError, match="independent primary-candidate"):
        bind(parts)


def test_incomplete_capture_cannot_be_repaired_by_candidate_success():
    parts = evidence()
    parts[1]["assigned"] = 383
    parts[2]["audit_sha256"] = digest(parts[1])
    parts[3]["audit_sha256"] = digest(parts[1])
    parts[3]["analysis_sha256"] = digest(parts[2])
    with pytest.raises(ValueError, match="complete passing384"):
        bind(parts)


# These synthetic receipts exercise v2 selection/admission without loading outcomes.
import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_voluntary_freshness as freshness  # noqa: E402
import audit_voluntary_resolution as auditor  # noqa: E402
from curiosity_study import voluntary_resolution as voluntary  # noqa: E402


def historical_receipt():
    first = voluntary.assignments()[0]
    canonical = voluntary.canonical_world(first["family"], voluntary.task(first))
    world_id = digest(canonical)
    registry = [dict(run=auditor.V1_RUN, execution_evidence="persisted_records")]
    exclusions = [dict(semantic_world_id=world_id, canonical_world=canonical, source_runs=[auditor.V1_RUN])]
    return dict(
        schema_version=1,
        assay="voluntary-resolution-historical-exclusions",
        selection_independent=True,
        audit_script_byte_sha256="script",
        source_file_sha256={},
        historical_registry=registry,
        historical_registry_sha256=digest(registry),
        archive_manifest_aliases=[],
        source_archive_bindings={},
        historical_exact_world_ids=[world_id],
        historical_exact_world_ids_sha256=digest([world_id]),
        exact_world_exclusions=exclusions,
        exact_world_exclusions_sha256=digest(exclusions),
        historical_world_counts=[],
        inclusion_rule="all executed",
        exact_rule="canonical table and truth",
        projection_rule="reported separately",
        independent_holdout_claim_supported=False,
        limitations=[
            "All assigned worlds in any evidenced collection are retained, including partial and failed runs; no scientific outcome fields determine exclusions.",
            "Exact exclusions concern named complete two-property tables plus realized truth; relational/target projected-table exposure remains a separate limitation.",
            "Local persisted history only; this does not establish independence from external or pretraining exposure.",
        ],
    )


def exposure_receipt(history, plan):
    return dict(
        **{key: deepcopy(value) for key, value in history.items() if key not in ("schema_version", "assay")},
        schema_version=2,
        assay="voluntary-resolution-freshness",
        excluded_target_collections=[],
        historical_snapshot_sha256=None,
        task_revision=voluntary.REVISION,
        target_seed=voluntary.SEED,
        target_plan_sha256=digest(plan),
        selection_sha256=plan["selection_sha256"],
        assignment_sha256=plan["assignment_sha256"],
        exact_complete_world_criterion_passed=True,
        exact_complete_world_overlap_count=0,
    )


def prospective_receipts(plan):
    identity = dict(
        passed=True,
        task_revision=voluntary.REVISION,
        seed=voluntary.SEED,
        assignment_sha256=plan["assignment_sha256"],
        selection_sha256=plan["selection_sha256"],
        task_module_sha256="task",
    )
    sources = [dict(trial_id=a["trial_id"]) for a in plan["assignments"]]
    receipts = [
        dict(trial_id=a["trial_id"], selected_service=service)
        for a in plan["assignments"]
        if a["kind"] == "choice"
        for service in voluntary.SERVICES
        if service != "stop"
    ]
    preflight = dict(
        **identity,
        model=MODEL,
        rows=432,
        source_rows=288,
        stop_checks=144,
        synthetic_answer_only=True,
        generated_model_outputs=False,
        source_bindings=sources,
        receipt_bindings=receipts,
        bindings_sha256=digest(dict(source=sources, receipts=receipts)),
    )
    bindings = []
    for a in plan["assignments"]:
        prompt, _, expected = voluntary.prompt(a)
        new_sha = digest(prompt)
        bindings.append(
            dict(
                trial_id=a["trial_id"],
                assignment_sha256=digest(a),
                kind=a["kind"],
                context_sha256=digest(voluntary.context(a)),
                new_prompt_sha256=new_sha,
                old_prompt_sha256=digest("prior prompt")
                if a["kind"] in ("new_fact", "repeated_fact")
                else new_sha,
                expected_answer=expected,
            )
        )
    materials = dict(
        **identity,
        frozen_v1_task_sha256=auditor.V1_TASK_SHA,
        counts=dict(
            contexts_unchanged=288,
            optional_prompts_unchanged=144,
            other_diagnostic_prompts_unchanged=96,
            clarified_diagnostic_prompts=48,
        ),
        bindings=bindings,
        bindings_sha256=digest(bindings),
    )
    return preflight, materials


def test_v2_history_excludes_failed_v1_world_and_retains_all_assignments():
    history = historical_receipt()
    plan = auditor.history_plan(history)
    assert len(plan["assignments"]) == 288
    assert not set(history["historical_exact_world_ids"]) & {
        a["semantic_world_id"] for a in plan["assignments"]
    }
    assert plan["selection"]["historical_exact_exclusions"] == history["historical_exact_world_ids"]
    auditor.exposure_binding(history, exposure_receipt(history, plan), plan)
    auditor.prospective_checks(plan, *prospective_receipts(plan), "task")


@pytest.mark.parametrize("mutation", ["drop_v1", "unexecuted_v1", "arbitrary_world", "missing_digest"])
def test_history_receipt_rejects_unbound_or_missing_failed_history(mutation):
    history = historical_receipt()
    if mutation == "drop_v1":
        history["historical_registry"] = []
    elif mutation == "unexecuted_v1":
        history["historical_registry"][0]["execution_evidence"] = "none"
    elif mutation == "arbitrary_world":
        history["exact_world_exclusions"][0]["canonical_world"]["family"] = "unrelated"
        history["exact_world_exclusions_sha256"] = digest(history["exact_world_exclusions"])
    else:
        history.pop("historical_exact_world_ids_sha256")
    history["historical_registry_sha256"] = digest(history["historical_registry"])
    with pytest.raises(ValueError):
        auditor.history_plan(history)


def test_exposure_cannot_use_target_cutoff_or_arbitrary_extra_exclusions():
    history = historical_receipt()
    plan = auditor.history_plan(history)
    exposure = exposure_receipt(history, plan)
    exposure["excluded_target_collections"] = [auditor.V1_RUN]
    with pytest.raises(ValueError, match="include all current exposure"):
        auditor.exposure_binding(history, exposure, plan)
    exposure = exposure_receipt(history, plan)
    exposure["historical_exact_world_ids"].append("f" * 64)
    with pytest.raises(ValueError, match="complete bound history"):
        auditor.exposure_binding(history, exposure, plan)


@pytest.mark.parametrize("name", ["preflight", "materials"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("task_revision", "voluntary-resolution-v1"),
        ("seed", 2026101434),
        ("assignment_sha256", "stale"),
        ("selection_sha256", "stale"),
        ("task_module_sha256", "stale"),
        ("passed", False),
    ],
)
def test_preflight_and_materials_must_bind_exact_prospective_plan(name, field, value):
    plan = auditor.history_plan(historical_receipt())
    preflight, materials = prospective_receipts(plan)
    {"preflight": preflight, "materials": materials}[name][field] = value
    with pytest.raises(ValueError, match="Prospective check identity"):
        auditor.prospective_checks(plan, preflight, materials, "task")


@pytest.mark.parametrize(
    "mutation", ["duplicate_receipt", "missing_source", "changed_choice", "unmodified_report", "wrong_answer"]
)
def test_rehashed_prospective_receipt_cannot_drop_cases_or_expand_wording_change(mutation):
    plan = auditor.history_plan(historical_receipt())
    preflight, materials = prospective_receipts(plan)
    if mutation == "duplicate_receipt":
        preflight["receipt_bindings"][0] = preflight["receipt_bindings"][1]
    elif mutation == "missing_source":
        preflight["source_bindings"].pop()
    elif mutation == "changed_choice":
        row = next(row for row in materials["bindings"] if row["kind"] == "choice")
        row["old_prompt_sha256"] = "changed"
    elif mutation == "unmodified_report":
        row = next(row for row in materials["bindings"] if row["kind"] == "repeated_fact")
        row["old_prompt_sha256"] = row["new_prompt_sha256"]
    else:
        materials["bindings"][0]["expected_answer"] = "incorrect"
    preflight["bindings_sha256"] = digest(
        dict(source=preflight["source_bindings"], receipts=preflight["receipt_bindings"])
    )
    materials["bindings_sha256"] = digest(materials["bindings"])
    with pytest.raises(ValueError):
        auditor.prospective_checks(plan, preflight, materials, "task")


def test_postcollection_replay_uses_cutoff_and_allows_new_archive_aliases(monkeypatch, tmp_path):
    history = historical_receipt()
    plan = auditor.history_plan(history)
    result = exposure_receipt(history, plan)
    result["archive_manifest_aliases"] = [
        dict(archive="runs/new-v2/source.tar.gz", member="candidate/manifest.json")
    ]
    result["source_file_sha256"] = {"executions/v2/src/task.py": "same bytes"}
    calls = []

    def replay(root, **kwargs):
        calls.append(kwargs)
        return result

    monkeypatch.setattr(freshness, "audit", replay)
    checked = auditor.recheck_history(tmp_path, history, plan)
    assert checked["independently_reconstructed"] and checked["replay_uses_frozen_registry"]
    assert calls[0]["registry_snapshot"] == history["historical_registry"]
    assert calls[0]["pool_candidates"] == 0
    assert "exclude_runs" not in calls[0]


def test_preparation_reconstructs_full_inventory_not_an_author_chosen_snapshot(monkeypatch, tmp_path):
    history = historical_receipt()
    plan = auditor.history_plan(history)
    result = exposure_receipt(history, plan)
    calls = []

    def replay(root, **kwargs):
        calls.append(kwargs)
        return result

    monkeypatch.setattr(freshness, "audit", replay)
    auditor.recheck_history(tmp_path, history, plan, prospective=True)
    assert "registry_snapshot" not in calls[0] and "exclude_runs" not in calls[0]
    result["historical_exact_world_ids"].append("f" * 64)
    with pytest.raises(ValueError, match="complete current inventory"):
        auditor.recheck_history(tmp_path, history, plan, prospective=True)


def test_postcollection_rejects_changed_original_archive_even_when_receipt_matches(monkeypatch, tmp_path):
    history = historical_receipt()
    archive = tmp_path / "source.tar.gz"
    archive.write_bytes(b"changed")
    history["source_archive_bindings"] = {
        "source": dict(archive="source.tar.gz", archive_byte_sha256="original")
    }
    plan = auditor.history_plan(history)
    monkeypatch.setattr(freshness, "audit", lambda *args, **kwargs: exposure_receipt(history, plan))
    with pytest.raises(ValueError, match="source archive changed"):
        auditor.recheck_history(tmp_path, history, plan)
