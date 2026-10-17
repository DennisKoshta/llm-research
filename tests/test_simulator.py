import itertools
import pytest
from curiosity_study.tasks import generate, Observation, Candidate, Mystery
from curiosity_study.oracle import compatible, information, prediction
from curiosity_study.assignments import make_assignments, instantiate
from curiosity_study.trial import Trial


@pytest.mark.parametrize("family", ["machines", "objects"])
@pytest.mark.parametrize("split", ["development", "evaluation"])
def test_generated_clue_guarantees(family, split):
    for index in range(100):
        task = generate(family, index, split=split)
        initial = [task.initial]
        assert len(compatible(task, initial)) == 4
        assert information(task, initial, task.target_query)["decisive"]
        assert information(task, initial, task.initial.query)["redundant"]
        assert compatible(task, initial + [task.observe(task.target_query)]) == [task.truth]
        assert information(task, initial + [task.observe(task.target_query)], task.target_query)["redundant"]
        for candidate in task.candidates:
            alternate = Mystery(
                task.item_id,
                family,
                task.template,
                task.seed,
                task.candidates,
                candidate.label,
                task.initial,
                task.target_query,
            )
            queries = [str(x) for x in range(7)] if family == "machines" else dict(candidate.parameters)
            for query in queries:
                assert prediction(task, candidate, query) == alternate.observe(query).value


def test_hand_specified_machine_and_contradiction():
    task = Mystery(
        "example",
        "machines",
        "manual",
        0,
        (Candidate("J", (1, 1, 7)), Candidate("K", (2, 0, 7))),
        "J",
        Observation("1", "2"),
        "2",
    )
    assert compatible(task, [task.initial]) == ["J", "K"]
    assert compatible(task, [task.initial, Observation("2", "3")]) == ["J"]
    assert compatible(task, [Observation("1", "6")]) == []
    with pytest.raises(ValueError):
        information(task, [Observation("1", "6")], "2")


def test_accounting_stop_limit_invalid_and_isolation():
    for price in (0, 1, 2, 4):
        for services in itertools.product(("target", "redundant", "novelty", "stop", "invalid"), repeat=3):
            trial = Trial(
                generate("machines", 0),
                "unresolved",
                price,
                dict(zip("ABCD", ("target", "redundant", "novelty", "stop"))),
                list("ABCD"),
                "123456",
            )
            sibling = trial.fork()
            for service in services:
                if trial.done:
                    break
                action = {
                    "target": "A",
                    "redundant": "B",
                    "novelty": "C",
                    "stop": "D",
                    "invalid": "A because...",
                }[service]
                trial.apply(action)
                assert trial.balance == 12 - price * trial.purchases
                assert sibling.balance == 12 and not sibling.ledger and len(sibling.observations) == 2
            assert trial.done
            with pytest.raises(ValueError):
                trial.apply("A")


def test_assignments_are_reproducible_and_crossed():
    first = make_assignments(6)
    assert first == make_assignments(6)
    assert len({a["trial_id"] for a in first}) == 96
    for family in ("machines", "objects"):
        for index in range(6):
            assignments = [a for a in first if a["family"] == family and a["index"] == index]
            assert len(assignments) == 8
            for a in assignments:
                trial = instantiate(a)
                assert trial.service_status()["target"]["decisive"] == (a["evidence"] == "unresolved")
                assert len(compatible(trial.task, trial.observations)) == (
                    1 if a["evidence"] == "resolved" else 4
                )
