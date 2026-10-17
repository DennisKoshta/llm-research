import copy
import importlib.util
from pathlib import Path
from collections import Counter

import pytest
from curiosity_study import target_relevant_information as target
from curiosity_study.artifacts import digest


def modules():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    result = []
    for name in ("prepare_target_relevant_smoke", "prepare_target_relevant_development"):
        spec = importlib.util.spec_from_file_location(name, scripts / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result.append(module)
    return result


def fixture():
    smoke_module, prepare = modules()
    smoke = smoke_module.build_manifest(dict(model_id="Qwen/Qwen3-32B", revision="test-pinned"))
    audit = dict(
        manifest_sha256=digest(smoke),
        complete=True,
        assigned=40,
        feasibility_passed=True,
        max_new_tokens=4096,
        target_relevance_oracle_audit_version=1,
        target_relevance_smoke_cell_coverage=True,
    )
    return prepare, smoke, audit


def test_factorial_action_complement_cross_and_independent_gain():
    prepare, smoke, audit = fixture()
    manifest = prepare.build_manifest(smoke, audit)
    rows = manifest["diagnostics"]
    cells = Counter((a["family"], a["kind"], a["independent_report"], a["query"]) for a in rows)
    assert len(cells) == 40 and set(cells.values()) == {24}
    assert len({a["trial_id"] for a in rows}) == 960
    for family in ("objects", "machines"):
        for block in (0, 12):
            chosen = [
                a
                for a in rows
                if a["family"] == family
                and block <= a["index"] < block + 12
                and a["kind"] == "instrument"
                and a["independent_report"] == a["query"] == 0
            ]
            cross = Counter(
                (
                    target.task(a)["target_complement_bit"],
                    tuple(target.task(a)["action_label_mapping"].values()),
                )
                for a in chosen
            )
            assert len(cross) == 12 and set(cross.values()) == {1}
    for assignment in rows:
        task = target.task(assignment)
        assert task["information_bits"] == task["offered_report_entropy_bits"] == 1
        assert task["expected_target_information_gain_bits"] == int(
            assignment["independent_report"] != assignment["query"]
        )


@pytest.mark.parametrize(
    "failure", ["failed", "partial", "wrong_binding", "wrong_oracle", "coverage", "task", "source_rows"]
)
def test_source_gate_rejects_untrusted_or_incomplete_smoke(failure):
    prepare, smoke, audit = fixture()
    if failure == "failed":
        audit["feasibility_passed"] = False
    elif failure == "partial":
        audit["complete"] = False
    elif failure == "wrong_binding":
        audit["manifest_sha256"] = "wrong"
    elif failure == "wrong_oracle":
        smoke["response_oracle_field"] = "information_bits"
        audit["manifest_sha256"] = digest(smoke)
    elif failure == "coverage":
        audit["target_relevance_smoke_cell_coverage"] = False
    elif failure == "task":
        smoke["task_module_sha256"] = "wrong"
        audit["manifest_sha256"] = digest(smoke)
    else:
        smoke["diagnostics"].pop()
        smoke["assignment_sha256"] = digest(smoke["diagnostics"])
        audit["manifest_sha256"] = digest(smoke)
    with pytest.raises(ValueError):
        prepare.build_manifest(smoke, audit)


def test_all_parent_cohorts_survive_partition_and_each_cell_must_pass(tmp_path):
    prepare, smoke, audit = fixture()
    manifest = prepare.build_manifest(smoke, audit)
    before = copy.deepcopy(manifest)
    children, bindings = prepare.partition_manifest(manifest, tmp_path)
    assert manifest == before and not list(tmp_path.iterdir())
    assert [b["branches"] for b in bindings] == [480, 480]
    seen = []
    for (_, child), binding in zip(children, bindings):
        assert child["parent_manifest_sha256"] == digest(manifest)
        for local, original in enumerate(binding["parent_cohorts"]):
            assert (
                child["diagnostics"][local * 32 : (local + 1) * 32]
                == manifest["diagnostics"][original * 32 : (original + 1) * 32]
            )
            seen.append(original)
    assert sorted(seen) == list(range(30))
    records = {a["trial_id"]: dict(correct=True, valid=True) for a in manifest["diagnostics"]}
    assert target.evaluate(manifest, records)["passed"]
    selected = [
        a
        for a in manifest["diagnostics"]
        if (a["family"], a["kind"], a["independent_report"], a["query"]) == ("objects", "instrument", 0, 1)
    ]
    for a in selected[:3]:
        records[a["trial_id"]]["correct"] = False
    assert not target.evaluate(manifest, records)["passed"]
