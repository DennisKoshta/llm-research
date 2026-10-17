import copy
import hashlib
import importlib.util
from pathlib import Path

import pytest

from curiosity_study import information_state as state
from curiosity_study.artifacts import RunStore, digest, read, write_once


def script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = script("prepare_replay_interventions")
analyze = script("analyze_replay_interventions")


def collection(items=2, seed=420, heldout=False, start_index=0):
    rows = state.assignments(items, seed=seed, kinds=("new", "repeat"), start_index=start_index)
    for a in rows:
        a["representation_split"] = "extraction" if a["index"] == 0 else "development"
        if heldout:
            a.update(vocabulary="heldout", template="prose")
    return dict(
        diagnostics=rows,
        assignment_sha256=digest(rows),
        task_module_sha256=hashlib.sha256(Path(state.__file__).read_bytes()).hexdigest(),
    )


def direction_metadata(manifest):
    # These are the existing representation analyzer's actual provenance fields.
    return dict(manifest_sha256=digest(manifest), development_only=True)


def test_development_reuse_is_explicit_and_missing_scope_fails():
    m = collection()
    metadata = direction_metadata(m)
    selected = [a for a in m["diagnostics"] if a["index"] == 0]
    with pytest.raises(ValueError, match="inference_scope explicitly"):
        prepare.split_provenance({}, m, metadata, selected)
    result = prepare.split_provenance(dict(inference_scope="development"), m, metadata, selected)
    assert result["inference_scope"] == "development"
    assert result["extraction_binding_verified"]
    assert len(result["overlaps"]["extraction"]["world_ids"]) == 2
    assert not result["confirmation_eligible"]


@pytest.mark.parametrize("index", [0, 1])
def test_confirmation_rejects_extraction_and_development_reuse(index):
    m = collection()
    with pytest.raises(ValueError, match="overlaps extraction or representation-development"):
        prepare.split_provenance(
            dict(inference_scope="confirmation", inspected_development_collections=[]),
            m,
            direction_metadata(m),
            [a for a in m["diagnostics"] if a["index"] == index],
        )


def test_confirmation_binds_global_world_identity_and_new_semantics(tmp_path):
    old, fresh = collection(), collection(seed=421, heldout=True)
    write_once(tmp_path / "manifest.json", old)
    config = dict(
        inference_scope="confirmation",
        direction_extraction_collection=str(tmp_path),
        inspected_development_collections=[],
    )
    result = prepare.split_provenance(config, fresh, direction_metadata(old), fresh["diagnostics"])
    assert result["confirmation_eligible"] and result["extraction_binding_verified"]
    assert all(not v["world_ids"] and not v["semantic_world_ids"] for v in result["overlaps"].values())
    # Both collections use indices 0 and 1; distinct seeds and semantics matter.
    assert result["evaluation_assignments_sha256"] == digest(fresh["diagnostics"])
    with pytest.raises(ValueError, match="requires the bound"):
        prepare.split_provenance(
            dict(inference_scope="confirmation", inspected_development_collections=[]),
            fresh,
            direction_metadata(old),
            fresh["diagnostics"],
        )
    unverified = prepare.split_provenance(
        dict(inference_scope="development"), fresh, direction_metadata(old), fresh["diagnostics"]
    )
    assert not unverified["extraction_binding_verified"] and not unverified["confirmation_eligible"]


def test_confirmation_rejects_semantic_duplicates_with_distinct_world_ids(tmp_path):
    old = collection()
    fresh = collection(items=1, start_index=52)
    selected = [a for a in fresh["diagnostics"] if a["family"] == "objects"]
    old_objects = [a for a in old["diagnostics"] if a["family"] == "objects" and a["index"] == 0]
    old_ids, old_semantics = prepare.world_sets(old_objects, old)
    new_ids, new_semantics = prepare.world_sets(selected, fresh)
    assert not old_ids & new_ids
    assert old_semantics == new_semantics  # Different row order/index, same finite world.
    write_once(tmp_path / "manifest.json", old)
    with pytest.raises(ValueError, match="overlaps extraction or representation-development"):
        prepare.split_provenance(
            dict(
                inference_scope="confirmation",
                direction_extraction_collection=str(tmp_path),
                inspected_development_collections=[],
            ),
            fresh,
            direction_metadata(old),
            selected,
        )


@pytest.mark.parametrize("problem", ["missing_split", "split_world", "module_hash", "analysis_hash"])
def test_confirmation_fails_closed_when_split_binding_is_ambiguous(tmp_path, problem):
    old, fresh = collection(), collection(seed=421, heldout=True)
    if problem == "missing_split":
        del old["diagnostics"][0]["representation_split"]
    elif problem == "split_world":
        a = next(a for a in old["diagnostics"] if a["index"] == 0)
        a["representation_split"] = "development"
    elif problem == "module_hash":
        old["task_module_sha256"] = "wrong"
    old["assignment_sha256"] = digest(old["diagnostics"])
    metadata = direction_metadata(old)
    if problem == "analysis_hash":
        metadata["manifest_sha256"] = "wrong"
    write_once(tmp_path / "manifest.json", old)
    with pytest.raises(ValueError):
        prepare.split_provenance(
            dict(
                inference_scope="confirmation",
                direction_extraction_collection=str(tmp_path),
                inspected_development_collections=[],
            ),
            fresh,
            metadata,
            fresh["diagnostics"],
        )


def synthetic_collection(tmp_path, bad_sham=False, no_engagement=False, sources=None, answer_override=None):
    if sources is None:
        sources = [a for a in state.assignments(1, kinds=("new",)) if a["family"] == "objects"]
    diagnostics, prefixes, records = [], {}, []
    source_manifest = dict(
        diagnostics=sources,
        assignment_sha256=digest(sources),
        source_sha256="sourcecode",
        phase="validation",
        validation_criteria=dict(minimum_distinct_items_per_cell=100),
    )
    source_audit = dict(
        manifest_sha256=digest(source_manifest),
        source_sha256="sourcecode",
        complete=True,
        assigned=len(sources),
        validation_criteria=source_manifest["validation_criteria"],
        gates=dict(passed=True),
    )
    write_once(tmp_path / "source/manifest.json", source_manifest)
    write_once(tmp_path / "source-audit.json", source_audit)
    schedule = dict(
        source_collection=str(tmp_path / "source"),
        source_audit=str(tmp_path / "source-audit.json"),
        inference_scope="development",
        world_index_range=[min(a["index"] for a in sources), max(a["index"] for a in sources) + 1],
        kinds=sorted({a["kind"] for a in sources}),
    )
    admission = prepare.admit_source(schedule, source_manifest, source_audit, sources)
    provenance = dict(
        schema_version=1,
        inference_scope="development",
        direction_analysis_sha256="analysis",
        direction_extraction_manifest_sha256="extraction",
        evaluation_manifest_sha256=digest(source_manifest),
        assigned_sources=len(sources),
    )
    for sa in sources:
        key = sa["trial_id"]
        p = dict(
            assignment=sa,
            token_prefix=[1, 2, 3],
            token_prefix_sha256=digest([1, 2, 3]),
            prefix_positions=dict(record=0, think_end=1, final=2),
            seed=123,
        )
        prefixes[key] = p
        module = analyze.task_module(sa)
        _, labels, expected = module.prompt(sa)
        for arm in ("baseline", "sham", "direction"):
            a = dict(
                trial_id=f"{key}-{arm}",
                prefix_id=key,
                arm=arm,
                kind=sa["kind"],
                layer=None if arm == "baseline" else 31,
                position=None if arm == "baseline" else "final",
                capture_position=None if arm == "baseline" else "decision",
                direction_name=arm,
                dose=int(arm == "direction"),
                dose_multiplier=int(arm == "direction"),
            )
            diagnostics.append(a)
            change = (
                None
                if arm == "baseline"
                else dict(
                    layer=31,
                    position="final",
                    delta_norm=int(arm == "direction"),
                    relative_delta_norm=0.01 if arm == "direction" else 0,
                    projection_before=0,
                    projection_after=int(arm == "direction" and not no_engagement),
                )
            )
            answer = answer_override(sa, arm, expected) if answer_override else expected
            completion = dict(
                raw=answer,
                generated_token_ids=[ord(answer[0]), 127],
                ended_with_eos=True,
                sampling=dict(seed=123, temperature=0.7, top_p=1.0, max_new_tokens=8),
                intervention_schedule="prefill_only; no active hooks during decoding",
            )
            scored = module.record_from_inference(sa, completion)
            records.append(
                dict(
                    **{
                        k: scored[k]
                        for k in ("answer", "parsed_answer", "expected", "valid", "correct", "task")
                    },
                    trial_id=a["trial_id"],
                    terminal=True,
                    assignment=a,
                    source_assignment=sa,
                    measurement=dict(
                        unpadded_input_ids=p["token_prefix"],
                        token_prefix_sha256=p["token_prefix_sha256"],
                        unpadded_token_positions=p["prefix_positions"],
                        hooks_remaining=0,
                        exact_token_replay=True,
                        completion=completion,
                        label_logits={
                            label: 2 + 0.1 * (bad_sham and arm == "sham") if label == labels[0] else 1
                            for label in labels
                        },
                        label_probabilities_temperature_1={label: 0.9 / len(labels) for label in labels},
                        label_mass=0.9,
                        intervention=change,
                    ),
                )
            )
    m = dict(
        diagnostics=diagnostics,
        prefixes=prefixes,
        execution=dict(batch_size=4),
        source_sha256="sourcecode",
        phase="development",
        inference_scope="development",
        assigned_sources=len(sources),
        excluded_sources=[],
        split_provenance=provenance,
        split_provenance_sha256=digest(provenance),
        direction_analysis_sha256="analysis",
        direction_extraction_manifest_sha256="extraction",
        source_parent_manifest_sha256=digest(source_manifest),
        intervention_schedule=schedule,
        intervention_schedule_sha256=digest(schedule),
        source_admission=admission,
        source_admission_sha256=digest(admission),
    )
    root, child = tmp_path / "parent", tmp_path / "child"
    write_once(root / "manifest.json", m)
    cm = {**copy.deepcopy(m), "parent_manifest_sha256": digest(m)}
    store = RunStore(child, cm)
    store.save_many(records, "synthetic", "cohort")
    write_once(
        root / "children.json",
        [
            dict(
                path=str(child),
                manifest_sha256=digest(cm),
                parent_cohorts=list(range((len(diagnostics) + 3) // 4)),
            )
        ],
    )
    return root


@pytest.mark.parametrize("bad_sham,no_engagement", [(False, False), (True, False), (False, True)])
def test_analysis_distinguishes_complete_collection_from_integrity(tmp_path, bad_sham, no_engagement):
    root = synthetic_collection(tmp_path, bad_sham=bad_sham, no_engagement=no_engagement)
    result = analyze.analyze(root, tmp_path / "analysis")
    assert result["complete"]
    assert result["integrity_passed"] == (not bad_sham and not no_engagement)
    assert result["eligible_for_interpretation"] == result["integrity_passed"]
    assert result["integrity_checks"]["sham_logit_identity"] is not bad_sham
    assert result["integrity_checks"]["intervention_implementation_verified"] is not no_engagement
    assert result["sham_probability_identity"]  # Rounded probabilities cannot hide differing logits.
    assert read(tmp_path / "analysis/analysis.json")["integrity_passed"] == result["integrity_passed"]
    assert all(row["complete_worlds"] == 1 for row in result["metrics"])


@pytest.mark.parametrize("problem", ["missing_provenance", "bad_hash", "missing_sham", "no_intervention"])
def test_integrity_cannot_pass_without_required_controls_or_provenance(tmp_path, problem):
    root = synthetic_collection(tmp_path)
    manifest = read(root / "manifest.json")
    sham_sources = {"source"}
    engagement = [
        dict(
            requested_dose=1,
            projection_delta=1,
            delta_norm=1,
            relative_delta_norm=0.01,
            signed_engagement=True,
        )
    ]
    if problem == "missing_provenance":
        del manifest["split_provenance"]
    elif problem == "bad_hash":
        manifest["split_provenance_sha256"] = "wrong"
    elif problem == "missing_sham":
        sham_sources.clear()
    elif problem == "no_intervention":
        engagement.clear()
    gate = analyze.integrity_gate(
        manifest,
        {"source": None},
        sham_sources,
        [0],
        [0],
        [True],
        [True],
        engagement,
    )
    assert not gate["passed"]


def test_count_competence_world_binding_supports_fresh_confirmation(tmp_path):
    from curiosity_study import count_competence as competence

    old = collection()
    rows = competence.assignments(items=1, seed=421)
    fresh = dict(
        diagnostics=rows,
        assignment_sha256=digest(rows),
        task_module_sha256=hashlib.sha256(Path(competence.__file__).read_bytes()).hexdigest(),
    )
    write_once(tmp_path / "manifest.json", old)
    result = prepare.split_provenance(
        dict(
            inference_scope="confirmation",
            direction_extraction_collection=str(tmp_path),
            inspected_development_collections=[],
        ),
        fresh,
        direction_metadata(old),
        rows,
    )
    assert result["confirmation_eligible"]
    assert len(result["evaluation_world_ids"]) == 2
    assert len(result["evaluation_semantic_world_ids"]) == 2
    assert all(a["correct_count"] in (1, 2, 4) for a in rows)


def test_count_metrics_keep_six_cells_and_resample_whole_worlds_across_templates(tmp_path):
    from curiosity_study import count_competence as competence

    sources = [
        a
        for seed, template in ((7, "table"), (7, "prose"), (8, "table"))
        for a in competence.assignments(items=1, seed=seed, template=template)
        if a["family"] == "objects"
    ]

    def answer(sa, arm, expected):
        if arm == "direction" and sa["seed"] == 7 and sa["template"] == "table" and sa["correct_count"] == 2:
            return "1"
        return expected

    root = synthetic_collection(tmp_path, sources=sources, answer_override=answer)
    result = analyze.analyze(root, tmp_path / "analysis")
    assert result["integrity_passed"]
    metrics = {r["correct_count"]: r for r in result["metrics"] if r["direction"] == "direction"}
    assert set(metrics) == {None, 1, 2, 4}
    for count, row in metrics.items():
        assert row["complete_worlds"] == row["paired_accuracy_change"]["worlds"] == 2
        assert row["complete_world_templates"] == row["eligible_world_templates"] == 3
        assert row["required_cells_per_world_template"] == 6
        assert row["branches"] == (18 if count is None else 6)
        assert row["templates"] == ["prose", "table"]
    assert metrics[None]["paired_accuracy_change"]["mean"] == pytest.approx(-1 / 12)
    assert metrics[2]["paired_accuracy_change"]["mean"] == pytest.approx(-0.25)
    assert metrics[1]["paired_accuracy_change"]["mean"] == metrics[4]["paired_accuracy_change"]["mean"] == 0
    assert metrics[None]["new_failure_worlds"] == metrics[2]["new_failure_worlds"] == 1
    compact = read(tmp_path / "analysis/compact_outcomes.json")
    assert {r["prefix_id"] for r in compact} == {a["trial_id"] for a in sources}
    assert {tuple(r["cell"]) for r in compact} == {(n, q) for n in (1, 2, 4) for q in (0, 1)}
    assert all(r["information_bits"] in (0, 1) for r in compact)


def test_count_strata_exclude_an_entire_incomplete_six_cell_world(tmp_path):
    from curiosity_study import count_competence as competence

    sources = [
        a
        for a in competence.assignments(items=2)
        if a["family"] == "objects" and not (a["index"] == 1 and a["correct_count"] == 4 and a["query"] == 1)
    ]
    root = synthetic_collection(tmp_path, sources=sources)
    result = analyze.analyze(root, tmp_path / "analysis")
    for row in result["metrics"]:
        assert row["eligible_worlds"] == 2 and row["complete_worlds"] == 1
        assert row["complete_world_templates"] == 1
        assert row["branches"] == (6 if row["correct_count"] is None else 2)
        assert row["paired_accuracy_change"]["worlds"] == 1


def test_relational_replay_keeps_its_original_four_cell_aggregation(tmp_path):
    from curiosity_study import relational_information as relational

    sources = [a for a in relational.assignments(items=1, kinds=("new",)) if a["family"] == "objects"]
    root = synthetic_collection(tmp_path, sources=sources)
    result = analyze.analyze(root, tmp_path / "analysis")
    assert result["integrity_passed"]
    assert len(result["metrics"]) == 2
    for row in result["metrics"]:
        assert row["task_revision"] == relational.REVISION
        assert row["correct_count"] is None
        assert row["required_cells_per_world_template"] == row["branches"] == 4
        assert row["complete_worlds"] == row["complete_world_templates"] == 1
        assert row["paired_accuracy_change"]["mean"] == 0
