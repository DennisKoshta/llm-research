import copy
import hashlib
import importlib.util
from pathlib import Path
from collections import Counter
import pytest
from curiosity_study import relational_information as relational
from curiosity_study.artifacts import digest
from curiosity_study.thinking_information import EXECUTION


def preparation_module():
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_relational_development.py"
    spec = importlib.util.spec_from_file_location("prepare_relational_development", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_fixture():
    rows = relational.assignments(items=1, seed=2026101410)
    for a in rows:
        a["execution"] = EXECUTION.copy()
        a["max_new_tokens"] = 4096
    smoke = dict(
        assay="thinking_information",
        phase="development_smoke",
        task_revision=relational.REVISION,
        model=dict(model_id="Qwen/Qwen3-32B", revision="synthetic-pinned"),
        diagnostics=rows,
        assignment_sha256=digest(rows),
        task_module_sha256=hashlib.sha256(Path(relational.__file__).read_bytes()).hexdigest(),
        max_new_tokens=4096,
        inference_revision="thinking-4096-v1",
    )
    audit = dict(
        manifest_sha256=digest(smoke),
        complete=True,
        assigned=40,
        feasibility_passed=True,
        max_new_tokens=4096,
    )
    return smoke, audit


def test_development_design_is_fixed_balanced_and_explicitly_nonconfirmatory():
    prepare = preparation_module()
    smoke, audit = source_fixture()
    original = copy.deepcopy(smoke)
    manifest = prepare.build_manifest(smoke, audit)
    assert smoke == original
    rows = manifest["diagnostics"]
    assert len(rows) == len({a["trial_id"] for a in rows}) == 800
    assert manifest["phase"] == "development_validation" and manifest["inference_scope"] == "development"
    assert manifest["execution_gpu"] == "H200" and manifest["execution"]["batch_size"] == 32
    assert manifest["inference_revision"] == "thinking-4096-v1"
    assert manifest["max_new_tokens"] == 4096 and all(a["max_new_tokens"] == 4096 for a in rows)
    assert manifest["validation_criteria"]["minimum_distinct_items_per_cell"] == 20
    counts = Counter((a["family"], a["kind"], a["correlated"], a["query"]) for a in rows)
    assert len(counts) == 40 and set(counts.values()) == {20}
    assert all(
        a["seed"] == 2026101416 and a["template"] == "table" and a["vocabulary"] == "heldout" for a in rows
    )
    assert [a["kind"] for a in rows] == sorted(a["kind"] for a in rows)
    expected = relational.assignments(items=20, seed=2026101416, template="table")
    expected.sort(key=lambda a: a["kind"])
    assert [a["trial_id"] for a in rows] == [a["trial_id"] for a in expected]


def test_two_children_preserve_every_original_cohort_exactly_once(tmp_path):
    prepare = preparation_module()
    manifest = prepare.build_manifest(*source_fixture())
    before = copy.deepcopy(manifest)
    children, bindings = prepare.partition_manifest(manifest, tmp_path)
    assert manifest == before and not list(tmp_path.iterdir())
    assert len(children) == len(bindings) == 2
    assert [b["branches"] for b in bindings] == [416, 384]
    parent_cohorts = [manifest["diagnostics"][i : i + 32] for i in range(0, 800, 32)]
    seen = []
    for (path, child), binding in zip(children, bindings):
        assert path == Path(binding["path"])
        assert child["parent_manifest_sha256"] == digest(manifest)
        assert digest(child) == binding["manifest_sha256"]
        assert child["phase"] == "development_validation_partition"
        assert len(child["diagnostics"]) % 32 == 0
        for local, parent in enumerate(binding["parent_cohorts"]):
            assert child["diagnostics"][local * 32 : (local + 1) * 32] == parent_cohorts[parent]
            seen.append(parent)
    assert sorted(seen) == list(range(25))
    assert all(len({a["kind"] for a in cohort}) == 1 for cohort in parent_cohorts)


@pytest.mark.parametrize(
    "failure", ["failed", "incomplete", "wrong_hash", "wrong_seed", "task_changed", "old_cap"]
)
def test_smoke_prerequisite_cannot_be_bypassed(failure):
    prepare = preparation_module()
    smoke, audit = source_fixture()
    if failure == "failed":
        audit["feasibility_passed"] = False
    elif failure == "incomplete":
        audit["complete"] = False
    elif failure == "wrong_hash":
        audit["manifest_sha256"] = "wrong"
    elif failure == "wrong_seed":
        smoke["diagnostics"][0]["seed"] = 123
        smoke["assignment_sha256"] = digest(smoke["diagnostics"])
        audit["manifest_sha256"] = digest(smoke)
    elif failure == "old_cap":
        smoke["max_new_tokens"] = 2048
        for assignment in smoke["diagnostics"]:
            assignment.pop("max_new_tokens")
        smoke["assignment_sha256"] = digest(smoke["diagnostics"])
        audit["manifest_sha256"] = digest(smoke)
    else:
        smoke["task_module_sha256"] = "changed"
        audit["manifest_sha256"] = digest(smoke)
    with pytest.raises(ValueError):
        prepare.build_manifest(smoke, audit)


def test_existing_relational_evaluator_honors_twenty_world_boundary():
    prepare = preparation_module()
    manifest = prepare.build_manifest(*source_fixture())
    records = {a["trial_id"]: dict(correct=True, valid=True) for a in manifest["diagnostics"]}
    result = relational.evaluate(manifest, records)
    assert result["passed"] and all(c["n"] == c["items"] == 20 for c in result["cells"])
    cell = [
        a
        for a in manifest["diagnostics"]
        if (a["family"], a["kind"], a["correlated"], a["query"]) == ("objects", "new", 0, 0)
    ]
    for a in cell[:2]:
        records[a["trial_id"]]["correct"] = False
    records[cell[0]["trial_id"]]["valid"] = False
    assert relational.evaluate(manifest, records)["passed"]  # 18/20 accurate, 19/20 valid.
    records[cell[2]["trial_id"]]["correct"] = False
    assert not relational.evaluate(manifest, records)["passed"]
    records[cell[2]["trial_id"]]["correct"] = True
    records[cell[1]["trial_id"]]["valid"] = False
    assert not relational.evaluate(manifest, records)["passed"]
