import copy
from curiosity_study.assignments import make_assignments, instantiate
from curiosity_study.diagnostics import accounting_problem, prompt_for
from curiosity_study.runner import build_manifest
from curiosity_study.validation import evaluate
from curiosity_study.tasks import describe
from curiosity_study.oracle import prediction


def test_lookup_tables_agree_with_independent_oracle():
    for a in make_assignments(100):
        if a["family"] != "machines":
            continue
        task = instantiate(a).task
        queries = sorted({int(task.initial.query), int(task.target_query)})
        rows = [line.split(" | ") for line in describe(task).splitlines() if " | " in line][1:]
        for candidate, row in zip(task.candidates, rows):
            assert row[0] == candidate.label
            assert row[1:] == [prediction(task, candidate, str(query)) for query in queries]


def test_accounting_varies_and_options_have_one_correct_answer():
    manifest = build_manifest(make_assignments(100), {"model_id": "test"}, "validation")
    contexts = set()
    for a in manifest["diagnostics"]:
        if a["kind"] != "accounting":
            continue
        content, expected, details = accounting_problem(a)
        current = details["starting_balance"]
        for _ in range(details["purchases"]):
            current -= details["price"]
        assert details["options"][expected] == current >= 0
        assert len(set(details["options"].values())) == 4
        contexts.add((details["starting_balance"], details["price"], details["purchases"]))
        assert content.count("Purchase ") == details["purchases"]
    assert len(contexts) == 200


def test_shared_menu_is_used_in_behavior_and_comprehension():
    for a in build_manifest(make_assignments(3), {"model_id": "test"}, "validation")["diagnostics"]:
        if a["kind"] not in ("instrumental", "service_mapping", "redundant_mapping"):
            continue
        trial = instantiate(a)
        content, _, expected, _ = prompt_for(trial, a)
        for description in trial.descriptions().values():
            assert description in content and description in trial.prompt()
        assert trial.mapping[expected] == ("redundant" if a["kind"] == "redundant_mapping" else "target")


def test_joint_advancement_requires_every_diagnostic_and_complete_sample():
    manifest = build_manifest(make_assignments(100), {"model_id": "test"}, "validation")
    records = {
        a["trial_id"]: {"terminal": True, "correct": True, "valid": True} for a in manifest["diagnostics"]
    }
    assert evaluate(manifest, records)["passed"]
    failing = copy.deepcopy(records)
    ids = [
        a["trial_id"]
        for a in manifest["diagnostics"]
        if a["family"] == "objects" and a["kind"] == "instrumental"
    ]
    for trial_id in ids[:11]:
        failing[trial_id]["correct"] = False
    assert not evaluate(manifest, failing)["passed"]
    failing = copy.deepcopy(records)
    for trial_id in ids[:6]:
        failing[trial_id]["valid"] = False
    assert not evaluate(manifest, failing)["passed"]
    del records[next(iter(records))]
    assert not evaluate(manifest, records)["passed"]
    small = build_manifest(make_assignments(2), {"model_id": "test"}, "validation")
    assert not evaluate(small, records)["passed"]
