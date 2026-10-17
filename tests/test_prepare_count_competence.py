"""Prospective development preparation, without creating runs or reading outcomes."""

from collections import Counter, defaultdict
import copy
import importlib.util
from pathlib import Path

import pytest

from curiosity_study import count_competence as competence
from curiosity_study.artifacts import digest


@pytest.fixture
def preparer(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_count_competence.py"
    spec = importlib.util.spec_from_file_location("prepare_count_competence_development", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "freeze", lambda *args: pytest.fail("Must not freeze a collection"))
    monkeypatch.setattr(module, "write_once", lambda *args: pytest.fail("Must not write a collection"))
    return module


@pytest.fixture
def development(preparer):
    return preparer.build_manifest(
        dict(model_id="Qwen/Qwen3-32B", revision="synthetic"), items=16, mode="development"
    )


def perfect_records(manifest):
    return {a["trial_id"]: dict(valid=True, correct=True) for a in manifest["diagnostics"]}


def selected_cell(manifest):
    return [
        a
        for a in manifest["diagnostics"]
        if (a["family"], a["correct_count"], a["query"]) == ("objects", 1, 0)
    ]


def test_development_is_explicit_with_complete_whole_world_schedule(development):
    assert development["phase"] == "development_validation"
    assert development["inference_scope"] == "development"
    assert development["validation_criteria"]["cell_axes"] == ["family", "correct_count", "query"]
    assert development["validation_criteria"]["minimum_distinct_items_per_cell"] == 16
    assert development["validation_criteria"]["minimum_accuracy"] == 0.9
    assert development["validation_criteria"]["minimum_valid_format"] == 0.95
    rows = development["diagnostics"]
    assert len(rows) == 192
    assert development["assignment_sha256"] == digest(rows)
    assert development["max_new_tokens"] == 4096
    assert all(a["max_new_tokens"] == 4096 for a in rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row["world_id"]].append(row)
    assert len(groups) == 32
    assert Counter(group[0]["family"] for group in groups.values()) == {"objects": 16, "machines": 16}
    required = {(count, query) for count in (1, 2, 4) for query in (0, 1)}
    assert all(
        len(group) == 6 and {(a["correct_count"], a["query"]) for a in group} == required
        for group in groups.values()
    )
    assert development["development_design"]["later_natural_continuations"] == 768
    assert development["development_design"]["natural_source_block_size"] == 96


def test_query_specific_gates_use_all_assigned_rows_and_do_not_pool_away_failure(development):
    records = perfect_records(development)
    result = competence.evaluate(development, records)
    assert result["passed"] and len(result["cells"]) == 12
    assert all(c["n"] == c["items"] == 16 for c in result["cells"])
    rows = selected_cell(development)
    records[rows[0]["trial_id"]]["correct"] = False
    assert competence.evaluate(development, records)["passed"]  # 15/16 >= .90
    records[rows[1]["trial_id"]]["correct"] = False
    failed = competence.evaluate(development, records)
    assert not failed["passed"]
    assert sum(not c["passed"] for c in failed["cells"]) == 1
    assert next(c for c in failed["cells"] if not c["passed"])["correct"] == 14
    # Pooling both query branches would incorrectly rescue this failure.
    pooled = copy.deepcopy(development)
    pooled["validation_criteria"]["cell_axes"] = ["family", "correct_count"]
    assert competence.evaluate(pooled, records)["passed"]


def test_one_invalid_or_missing_response_fails_95_percent_gate(development):
    records = perfect_records(development)
    key = selected_cell(development)[0]["trial_id"]
    records[key] = dict(valid=False, correct=False)
    result = competence.evaluate(development, records)
    assert not result["passed"]
    failed = next(c for c in result["cells"] if not c["passed"])
    assert failed["valid"] == 15 and failed["n"] == failed["completed"] == 16
    records.pop(key)
    result = competence.evaluate(development, records)
    failed = next(c for c in result["cells"] if not c["passed"])
    assert failed["n"] == 16 and failed["completed"] == 15


@pytest.mark.parametrize("malformation", ["missing_cell", "duplicate_cell", "duplicate_trial_id"])
def test_query_gates_reject_structurally_incomplete_or_duplicated_assignments(development, malformation):
    records = perfect_records(development)
    bad = copy.deepcopy(development)
    rows = bad["diagnostics"]
    if malformation == "missing_cell":
        rows.pop()
    elif malformation == "duplicate_cell":
        rows.append(copy.deepcopy(rows[0]))
    else:
        rows[-1]["trial_id"] = rows[0]["trial_id"]
    assert not competence.evaluate(bad, records)["passed"]


def test_query_gate_render_exposes_all_twelve_cells(development):
    rendered = competence.render(development, perfect_records(development))
    assert "| Correct count | Query |" in rendered
    assert rendered.count("| objects |") == rendered.count("| machines |") == 6


def test_development_partition_keeps_worlds_and_three_cohort_blocks_together(preparer, development):
    parts = preparer.partition(development, 2)
    assert parts == [[0, 1, 2], [3, 4, 5]]
    rows = development["diagnostics"]
    worlds = {}
    ids = []
    for worker, cohorts in enumerate(parts):
        selected = [a for cohort in cohorts for a in rows[32 * cohort : 32 * (cohort + 1)]]
        assert len(selected) == 96
        for a in selected:
            assert worlds.setdefault(a["world_id"], worker) == worker
            ids.append(a["trial_id"])
    assert len(ids) == len(set(ids)) == 192
    assert preparer.partition(development, 1) == [list(range(6))]
    for workers in (3, 4):
        with pytest.raises(ValueError, match="at least one"):
            preparer.partition(development, workers)
    with pytest.raises(ValueError, match="between one and four"):
        preparer.partition(development, True)


@pytest.mark.parametrize("items", [15, 18, 100, True, 16.0])
def test_development_size_is_fixed_before_preparation(preparer, items):
    with pytest.raises(ValueError, match="exactly 16"):
        preparer.build_manifest(
            dict(model_id="Qwen/Qwen3-32B", revision="test"), items=items, mode="development"
        )


@pytest.mark.parametrize("items", [16, 99, 101, True])
def test_actual_validation_preparation_retains_minimum_and_balance(preparer, items):
    with pytest.raises(ValueError, match="at least 100"):
        preparer.prepare("unused", items=items)


def test_legacy_builder_and_six_gate_schema_remain_available(preparer):
    manifest = preparer.build_manifest(dict(model_id="Qwen/Qwen3-32B", revision="test"), items=2)
    assert manifest["phase"] == "validation"
    assert "inference_scope" not in manifest and "semantic_readiness" not in manifest
    assert manifest["validation_criteria"]["minimum_distinct_items_per_cell"] == 100
    assert manifest["validation_criteria"]["cell_axes"] == ["family", "correct_count"]
    result = competence.evaluate(manifest, perfect_records(manifest))
    assert len(result["cells"]) == 6 and not result["passed"]
    with pytest.raises(ValueError, match="explicit validation or development"):
        preparer.build_manifest({}, mode="confirmation")


def test_readiness_is_prospective_and_exposes_finite_vocabulary(development):
    readiness = development["semantic_readiness"]
    assert not readiness["outcomes_used"] and not readiness["prospective_confirmation_ready"]
    assert not readiness["inspected_collection_overlap_audited"]
    objects = readiness["families"]["objects"]
    machines = readiness["families"]["machines"]
    assert objects["underlying_vocabulary_capacity"] == 64
    assert objects["six_cell_variant_capacity"] == 128
    assert not objects["capacity_for_120_fresh_underlying_worlds"]
    assert machines["underlying_vocabulary_capacity"] == 5_198_400
    for family in (objects, machines):
        assert family["assigned_world_indices"] == 16
        assert family["unique_underlying_worlds"] + family["duplicate_underlying_world_indices"] == 16
    assert len(readiness["assigned_world_inventory"]) == 32


def test_semantic_inventory_ignores_rendering_input_order_and_row_order(preparer, monkeypatch):
    rows = competence.assignments(2)
    original = preparer.semantic_readiness(rows)
    assert preparer.semantic_readiness(list(reversed(rows))) == original
    assert preparer.semantic_readiness([{**a, "template": "prose"} for a in rows]) == original
    actual_task = competence.task

    def reordered(a):
        t = actual_task(a)
        t["candidates"] = list(reversed(t["candidates"]))
        t["truth"] = 3 - t["truth"]
        return t

    monkeypatch.setattr(competence, "task", reordered)
    assert preparer.semantic_readiness(rows) == original


def test_new_indices_and_partial_observation_roles_do_not_create_semantic_worlds(preparer, monkeypatch):
    rows = [a for a in competence.assignments(2) if a["family"] == "objects"]
    same_task = competence.task(rows[0])
    monkeypatch.setattr(competence, "task", lambda a: copy.deepcopy(same_task))
    result = preparer.semantic_readiness(rows)["families"]["objects"]
    assert result["assigned_world_indices"] == 2
    assert result["unique_underlying_worlds"] == 1
    assert result["unique_six_cell_assay_variants"] == 2
    assert result["duplicate_underlying_world_indices"] == 1


def test_semantic_readiness_refuses_incomplete_or_unsupported_vocabularies(preparer):
    rows = competence.assignments(2)
    with pytest.raises(ValueError, match="six-cell"):
        preparer.semantic_readiness(rows[1:])
    with pytest.raises(ValueError, match="held-out"):
        preparer.semantic_readiness([{**a, "vocabulary": "development"} for a in rows])
    with pytest.raises(ValueError, match="nonempty"):
        preparer.semantic_readiness([])
