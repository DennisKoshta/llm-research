import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest

from curiosity_study import information_state, relational_information, target_relevant_information
from curiosity_study import voluntary_resolution as voluntary
from curiosity_study import count_competence, optional_information, tasks
from curiosity_study.artifacts import digest

spec = importlib.util.spec_from_file_location(
    "audit_voluntary_freshness", Path(__file__).resolve().parents[1] / "scripts/audit_voluntary_freshness.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_projected_exposure_retains_relational_truth_but_ignores_unused_target_truth():
    world = {
        "attributes": ["color", "shape", "marking"],
        "candidates": [
            ["pink", "star", "striped"],
            ["pink", "heart", "striped"],
            ["tan", "star", "plain"],
            ["tan", "heart", "plain"],
        ],
        "truth": 0,
    }
    original = audit.projected_worlds("objects", world)
    changed = audit.projected_worlds("objects", {**world, "truth": 3})
    assert len(original) == 2  # Correlated color/marking pair has only two distinct rows.
    assert {digest(row) for row in original}.isdisjoint(digest(row) for row in changed)
    assert audit.projected_worlds("objects", world, truth_irrelevant=True) == audit.projected_worlds(
        "objects", {**world, "truth": 3}, truth_irrelevant=True
    )
    assert all("truth" not in row for row in audit.projected_worlds("objects", world, truth_irrelevant=True))
    assert all(len(row["attributes"]) == 2 for row in original)


def test_frozen_pure_oracles_reproduce_named_worlds_without_running_modules():
    base = audit.task_namespace(Path(information_state.__file__).read_text())
    for module, factors in (
        (information_state, {"known": [0], "query": 1}),
        (relational_information, {"correlated": 0, "query": 1}),
        (target_relevant_information, {"independent_report": 0, "query": 1}),
    ):
        frozen = audit.task_namespace(Path(module.__file__).read_text(), base=base)
        for family in voluntary.FAMILIES:
            assignment = dict(family=family, seed=voluntary.SEED, index=8, vocabulary="heldout", **factors)
            assert frozen.task(assignment) == module.task(assignment)


def test_legacy_domains_cannot_match_heldout_two_property_worlds():
    for family in voluntary.FAMILIES:
        source = dict(family=family, seed=voluntary.SEED, index=0, vocabulary="heldout", known=[0], query=1)
        proposed = voluntary.canonical_world(family, information_state.task(source))
        for index in range(12):
            task = audit.legacy_tasks.generate(family, index).as_dict()
            world = audit.legacy_world(task)
            assert proposed not in audit.projected_worlds(family, world)
            if family == "objects":
                shape = world["attributes"].index("shape")
                assert set(row[shape] for row in world["candidates"]).isdisjoint(
                    value for pair in information_state.HELDOUT_SHAPE_PAIRS for value in pair
                )
            else:
                assert set(world["attributes"]).isdisjoint(proposed["attributes"])


def finite_sources():
    return {
        Path(module.__file__).name: Path(module.__file__).read_text()
        for module in (information_state, voluntary, count_competence, optional_information, tasks)
    }


def make_run(root, name, assignments, *, assay="thinking_information", executed=True, prefixes=None):
    run = root / "runs" / name
    run.mkdir(parents=True)
    sources = finite_sources()
    manifest = dict(assay=assay, source_sha256=digest(sources), assignments=[], diagnostics=assignments)
    if prefixes is not None:
        manifest["prefixes"] = {row["trial_id"]: dict(assignment=row) for row in prefixes}
    (run / "manifest.json").write_text(json.dumps(manifest))
    with tarfile.open(run / "source.tar.gz", "w:gz") as archive:
        for filename, source in sources.items():
            data = source.encode()
            info = tarfile.TarInfo("src/curiosity_study/" + filename)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    if executed:
        record = run / "cohorts" / "attempt" / "0000000.json"
        record.parent.mkdir(parents=True)
        # A sentinel deliberately cannot be parsed: only presence is used.
        record.write_text("scientific outputs must never be parsed in this test")
    return run, manifest


def test_explicit_frozen_adapters_reproduce_voluntary_count_and_optional_worlds():
    sources = finite_sources()
    rows = [voluntary.assignments()[0], count_competence.assignments(items=1)[0], optional_information.assignments()[0]]
    modules = {
        voluntary.REVISION: voluntary,
        count_competence.REVISION: count_competence,
        optional_information.REVISION: optional_information,
    }
    for assignment in rows:
        assert audit.frozen_finite_world(assignment, sources) == modules[assignment["task_revision"]].task(assignment)
    bad = dict(rows[1], correct_count=99)
    with pytest.raises(ValueError, match="declared answer"):
        audit.frozen_finite_world(bad, sources)


@pytest.mark.parametrize("revision,assay", [
    ("future-unknown-v9", "thinking_information"), (None, "unknown"), ("future-unknown-v9", "concept_capture"),
])
def test_unknown_revision_never_falls_through_to_legacy(tmp_path, monkeypatch, revision, assay):
    row = dict(count_competence.assignments(items=1)[0], task_revision=revision)
    make_run(tmp_path, "unknown", [row], assay=assay)
    monkeypatch.setattr(audit.legacy_tasks, "generate", lambda *a, **k: pytest.fail("Legacy fallback executed"))
    with pytest.raises(ValueError, match="Unknown finite-task revision/assay"):
        audit.audit(tmp_path, pool_candidates=0)


def test_all_assignments_from_partial_count_run_are_excluded_without_reading_outputs(tmp_path):
    rows = count_competence.assignments(items=2)
    make_run(tmp_path, "partial-count", rows)
    result = audit.audit(tmp_path, pool_candidates=0)
    expected = sorted({digest(voluntary.canonical_world(a["family"], information_state.task(a))) for a in rows})
    assert result["historical_exact_world_ids"] == expected
    assert result["historical_exact_world_ids_sha256"] == digest(expected)
    assert all(row["source_runs"] == ["runs/partial-count"] for row in result["exact_world_exclusions"])
    history = audit.history_receipt(result)
    assert history["historical_exact_world_ids"] == expected
    assert "selection_sha256" not in history and "target_plan_sha256" not in history
    assert history["selection_independent"] is True
    other = audit.audit(tmp_path, pool_candidates=0, seed=2026101500)
    assert audit.history_receipt(other) == history


def test_explicit_target_plan_excludes_only_itself_and_never_excludes_it_from_new_history(tmp_path):
    plan = voluntary.assignment_plan(seed=2026101499)
    run, manifest = make_run(tmp_path, "target", plan["assignments"], assay="voluntary_resolution")
    manifest.update(
        task_revision=plan["task_revision"], assignment_sha256=plan["assignment_sha256"],
        world_selection=plan["selection"], world_selection_sha256=plan["selection_sha256"],
    )
    (run / "manifest.json").write_text(json.dumps(manifest))
    injected = audit.plan_from_manifest(manifest)
    result = audit.audit(tmp_path, pool_candidates=0, plan=injected)
    assert result["target_seed"] == 2026101499
    assert result["exact_complete_world_overlap_count"] == 24
    assert len(audit.history_receipt(result)["historical_exact_world_ids"]) == 24
    retrospective = audit.audit(tmp_path, pool_candidates=0, plan=injected, exclude_runs=[run])
    assert retrospective["exact_complete_world_overlap_count"] == 0
    with pytest.raises(ValueError, match="all current exposure"):
        audit.history_receipt(retrospective)
    other_run, _ = make_run(tmp_path, "not-target", count_competence.assignments(items=1))
    with pytest.raises(ValueError, match="Only the exact target"):
        audit.audit(tmp_path, pool_candidates=0, plan=injected, exclude_runs=[other_run])
    with pytest.raises(ValueError, match="seed disagrees"):
        audit.audit(tmp_path, pool_candidates=0, plan=injected, seed=3)


def test_prefix_assignments_and_executed_children_are_bound_and_retained(tmp_path):
    rows = count_competence.assignments(items=1)
    parent, _ = make_run(tmp_path, "parent", rows, executed=False)
    child, _ = make_run(tmp_path, "child", [], prefixes=rows)
    (parent / "children.json").write_text(json.dumps([dict(path=str(child))]))
    result = audit.audit(tmp_path, pool_candidates=0)
    registry = {row["run"]: row for row in result["historical_registry"]}
    assert registry["runs/parent"]["execution_evidence"] == "executed_child_records"
    assert registry["runs/child"]["source_assignment_sha256"] == digest(rows)
    assert all(row["source_runs"] == ["runs/child", "runs/parent"] for row in result["exact_world_exclusions"])


def test_cutoff_snapshot_preserves_old_execution_status_and_rejects_changed_manifest(tmp_path):
    run, manifest = make_run(tmp_path, "pending", count_competence.assignments(items=1), executed=False)
    snapshot, _ = audit.inventory(tmp_path)
    record = run / "cohorts" / "later" / "0.json"
    record.parent.mkdir(parents=True)
    record.write_text("unread")
    assert audit.audit(tmp_path, pool_candidates=0)["historical_exact_world_ids"]
    old = audit.audit(tmp_path, pool_candidates=0, registry_snapshot=snapshot)
    assert not old["historical_exact_world_ids"]
    with pytest.raises(ValueError, match="all current exposure"):
        audit.history_receipt(old)
    manifest["diagnostics"][0]["index"] += 1
    (run / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="snapshot manifest/assignment"):
        audit.audit(tmp_path, pool_candidates=0, registry_snapshot=snapshot)


def test_receipts_are_write_once(tmp_path):
    path = tmp_path / "receipt.json"
    audit.write_json_once(path, {"original": True})
    with pytest.raises(FileExistsError):
        audit.write_json_once(path, {"changed": True})
    assert json.loads(path.read_text()) == {"original": True}


def test_legacy_world_seed_cannot_replace_requested_target_seed(tmp_path):
    make_run(
        tmp_path, "legacy", [dict(family="machines", seed=91, index=0, trial_id="legacy0")], assay=None
    )
    target_seed = 2026101432
    result = audit.audit(tmp_path, seed=target_seed, pool_candidates=4)
    assert result["target_seed"] == target_seed
    assert result["selection_sha256"] == voluntary.assignment_plan(seed=target_seed)["selection_sha256"]
    machine_ids = {
        digest(voluntary.canonical_world("machines", information_state.task(dict(
            family="machines", seed=target_seed, index=index, vocabulary="heldout", known=[0], query=1,
        ))))
        for index in range(4)
    }
    assert next(row for row in result["candidate_pool"] if row["family"] == "machines")["unique_complete_worlds"] == len(machine_ids)
