from collections import Counter, defaultdict
import importlib.util
import math
from pathlib import Path

import pytest

from curiosity_study import count_competence as competence
from curiosity_study import information_state as base


def test_independent_candidate_enumeration_and_balanced_answers():
    rows = competence.assignments(items=10)
    assert len(rows) == 120
    worlds = defaultdict(list)
    for a in rows:
        worlds[a["world_id"]].append(a)
        t = competence.task(a)
        compatible = [
            candidate
            for candidate in t["candidates"]
            if all(candidate[k] == value for k, value in t["observations"].items())
        ]
        assert len(compatible) == a["correct_count"] == int(competence.prompt(a)[2])
        assert t["uncertainty_bits"] == math.log2(len(compatible))
        assert t["candidates"][t["truth"]] in compatible
        assert len({tuple(c) for c in t["candidates"]}) == 4
        assert a["observed_fact_count"] == len(a["known"])
        assert a["vocabulary"] == "heldout"
        assert a["competence_group"] == a["world_id"]
    for group in worlds.values():
        assert Counter(a["correct_count"] for a in group) == {1: 2, 2: 2, 4: 2}
        assert {(a["correct_count"], a["query"]) for a in group} == {
            (count, query) for count in (1, 2, 4) for query in (0, 1)
        }
        tasks = [competence.task(a) for a in group]
        for t in tasks[1:]:
            assert all(t[key] == tasks[0][key] for key in ("attributes", "values", "candidates", "truth"))
    for family in ("objects", "machines"):
        partial = [a for a in rows if a["family"] == family and a["correct_count"] == 2]
        assert Counter(tuple(a["known"]) for a in partial) == {(0,): 10, (1,): 10}


def test_templates_preserve_worlds_and_use_existing_grammar():
    for a in competence.assignments(items=2):
        prose = {**a, "template": "prose"}
        assert competence.task(a) == competence.task(prose) == base.task(a)
        assert competence.prompt(a) == base.prompt(a)
        assert competence.prompt(a)[1:] == competence.prompt(prose)[1:]
        expected = competence.prompt(a)[2]
        assert competence.record_from_inference(a, {"raw": f"  {expected}.\n"})["correct"]
        for invalid in (expected + "..", expected + " because", "12", "", "A"):
            assert not competence.record_from_inference(a, {"raw": invalid})["valid"]
        wrong = competence.record_from_inference(a, {"raw": "3"})
        assert wrong["valid"] and not wrong["correct"]


def test_constant_answer_cannot_pass_and_missing_world_pair_fails():
    manifest = competence.build_manifest(dict(model_id="Qwen/Qwen3-32B", revision="test"), items=100)
    rows = manifest["diagnostics"]
    correct = {
        a["trial_id"]: competence.record_from_inference(a, {"raw": str(a["correct_count"])}) for a in rows
    }
    assert competence.evaluate(manifest, correct)["passed"]
    constant = {a["trial_id"]: competence.record_from_inference(a, {"raw": "2"}) for a in rows}
    result = competence.evaluate(manifest, constant)
    assert not result["passed"]
    assert [cell["passed"] for cell in result["cells"]] == [False, True, False, False, True, False]
    correct.pop(rows[0]["trial_id"])
    assert not competence.evaluate(manifest, correct)["passed"]
    broken = {**manifest, "diagnostics": rows[1:]}
    assert not competence.evaluate(broken, correct)["passed"]


def test_world_blocks_are_reproducible_and_worker_partitions_preserve_them():
    manifest = competence.build_manifest(dict(model_id="Qwen/Qwen3-32B", revision="test"), items=100)
    rows = manifest["diagnostics"]
    assert competence.assignments(2) == competence.assignments(2)
    assert competence.assignments(2, seed=123) != competence.assignments(2)
    for offset in range(0, len(rows), 6):
        assert len({a["world_id"] for a in rows[offset : offset + 6]}) == 1
    script = Path(__file__).resolve().parents[1] / "scripts/prepare_count_competence.py"
    spec = importlib.util.spec_from_file_location("prepare_count_competence", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    size = manifest["execution"]["batch_size"]
    for workers in range(1, 5):
        parts = module.partition(manifest, workers)
        seen = set()
        world_worker = {}
        for worker, indices in enumerate(parts):
            for index in indices:
                for a in rows[index * size : (index + 1) * size]:
                    assert a["trial_id"] not in seen
                    seen.add(a["trial_id"])
                    assert world_worker.setdefault(a["world_id"], worker) == worker
        assert len(seen) == len(rows)


def test_malformed_count_assignment_fails_before_inference():
    a = competence.assignments(1)[0]
    with pytest.raises(ValueError, match="declared answer"):
        competence.inference_request({**a, "correct_count": 3})
    with pytest.raises(ValueError, match="only candidate-count"):
        competence.inference_request({**a, "kind": "memory"})
