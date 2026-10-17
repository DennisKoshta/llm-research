import copy
import hashlib
import importlib.util
from pathlib import Path

import pytest

from curiosity_study import target_relevant_information as target
from curiosity_study.artifacts import digest, write_once
from curiosity_study.semantic_worlds import canonical_target_world, TARGET_SEMANTIC_RULE


def preparation():
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_replay_interventions.py"
    spec = importlib.util.spec_from_file_location("target_semantic_preparation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def paired_tables(assignment):
    return [target.task({**assignment, "independent_report": first}) for first in (0, 1)]


def manifest(rows):
    return dict(
        diagnostics=rows,
        assignment_sha256=digest(rows),
        task_module_sha256=hashlib.sha256(Path(target.__file__).read_bytes()).hexdigest(),
    )


def permute_display(task, permutation):
    result = copy.deepcopy(task)
    result["attributes"] = [task["attributes"][i] for i in permutation]
    result["candidates"] = [[row[i] for i in permutation] for row in reversed(task["candidates"])]
    for field in ("target_attribute", "independent_report_attribute", "target_linked_report_attribute"):
        result[field] = permutation.index(task[field])
    return result


@pytest.mark.parametrize("family", ["objects", "machines"])
def test_semantics_ignore_unused_truth_question_labels_and_display_order(family):
    a = next(a for a in target.assignments() if a["family"] == family)
    tables = paired_tables(a)
    expected = canonical_target_world(family, tables)
    changed = [permute_display(t, [2, 0, 1]) for t in reversed(tables)]
    for table in changed:
        table.update(
            truth=(table["truth"] + 1) % 4,
            queried_attribute=1 - table["queried_attribute"],
            action_label_mapping=dict(A="stop", B="report", C="reread"),
        )
    assert canonical_target_world(family, changed) == expected
    for first in (0, 1):
        for query in (0, 1):
            for kind in target.KINDS:
                variant = {
                    **a,
                    "independent_report": first,
                    "query": query,
                    "kind": kind,
                    "template": "prose",
                }
                assert canonical_target_world(family, paired_tables(variant)) == expected


def test_semantics_bind_target_role_and_both_dependence_tables():
    a = target.assignments()[0]
    tables = paired_tables(a)
    expected = canonical_target_world(a["family"], tables)
    changed = copy.deepcopy(tables)
    # Change only the unselected counterfactual: one table alone is insufficient.
    second = changed[1]
    value0, value1 = second["values"][second["target_attribute"]]
    for row in second["candidates"]:
        row[second["target_attribute"]] = value1 if row[second["target_attribute"]] == value0 else value0
    assert canonical_target_world(a["family"], changed) != expected
    # Move the target role and the report roles by exchanging property identities.
    changed = copy.deepcopy(tables)
    names = changed[0]["attributes"]
    old_target, old_report = names[2], names[0]
    for table in changed:
        table["attributes"] = [
            old_report if name == old_target else old_target if name == old_report else name
            for name in table["attributes"]
        ]
    assert canonical_target_world(a["family"], changed) != expected
    with pytest.raises(ValueError, match="both counterfactual"):
        canonical_target_world(a["family"], tables[:1])
    with pytest.raises(ValueError, match="both independent-report"):
        canonical_target_world(a["family"], [tables[0], tables[0]])


def test_world_sets_keep_all_cells_as_one_semantic_world_and_enforce_source_hash():
    prepare = preparation()
    rows = target.assignments(items=2)
    m = manifest(rows)
    identities, semantics = prepare.world_sets(rows, m)
    assert len(identities) == len(semantics) == 4
    for a in rows:
        assert prepare.world_sets([a], m)[1] == {
            digest(canonical_target_world(a["family"], paired_tables(a)))
        }
    with pytest.raises(ValueError, match="Task module hash"):
        prepare.world_sets(rows, {**m, "task_module_sha256": "stale"})


def test_distinct_target_world_ids_cannot_hide_semantic_overlap(tmp_path, monkeypatch):
    prepare = preparation()
    old_rows = target.assignments(items=2, kinds=("new", "repeat"))
    for a in old_rows:
        a["representation_split"] = "extraction" if a["index"] == 0 else "development"
    old = manifest(old_rows)
    write_once(tmp_path / "manifest.json", old)
    fresh_rows = target.assignments(items=1, seed=808, kinds=("new", "repeat"))
    fresh = manifest(fresh_rows)
    original_task = target.task

    def same_world_different_truth(a):
        task = original_task({**a, "seed": target.SMOKE_SEED, "index": 0})
        task["truth"] = (task["truth"] + (a["seed"] == 808)) % 4
        task["action_label_mapping"] = dict(A="stop", B="reread", C="report")
        return task

    monkeypatch.setattr(target, "task", same_world_different_truth)
    config = dict(
        inference_scope="confirmation",
        direction_extraction_collection=str(tmp_path),
        inspected_development_collections=[],
    )
    analysis = dict(manifest_sha256=digest(old), development_only=True)
    assert not prepare.world_sets(old_rows, old)[0] & prepare.world_sets(fresh_rows, fresh)[0]
    with pytest.raises(ValueError, match="overlaps extraction or representation-development"):
        prepare.split_provenance(config, fresh, analysis, fresh_rows)
    result = prepare.split_provenance(
        {**config, "inference_scope": "development"}, fresh, analysis, fresh_rows
    )
    assert result["semantic_rule"] == TARGET_SEMANTIC_RULE
    assert result["overlaps"]["extraction"]["semantic_world_ids"]
    assert not result["confirmation_eligible"]
