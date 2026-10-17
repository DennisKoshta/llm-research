from curiosity_study.identification import assignments, prompt, final_answer, run, manifest, summarize
from curiosity_study.model import MockModel
from curiosity_study.oracle import compatible


def test_matched_factorial():
    rows = assignments(2)
    assert len(rows) == 96
    assert len({a["trial_id"] for a in rows}) == 96
    for a in rows:
        content, expected, trial = prompt(a)
        assert len(compatible(trial.tasks[a["side"]], trial.observations[a["side"]])) == 1
        assert expected in a["labels"]
        assert ("Their hidden candidates are independent." in content) == (a["context"] == "paired")
        other = "Y" if a["side"] == "X" else "X"
        assert (f"Mystery {other}:" in content) == (a["context"] == "paired")
        assert trial.tasks[a["side"]].truth == dict(zip(a["labels"], "JKLM"))[expected]


def test_strict_thinking_parser():
    assert final_answer("reasoning\n</think>\nA", True) == ("A", True)
    assert final_answer("A", True) == ("", False)
    assert final_answer("M\nExplanation", False) == ("M\nExplanation", True)
    assert final_answer("stuff </think> A because", True) == ("A because", True)


def test_all_assigned_denominators():
    m = manifest({"model_id": "mock"}, 1)
    a = m["diagnostics"][0]
    r = run(a, MockModel())
    summary = summarize(m, {a["trial_id"]: r})
    assert sum(g["n"] for g in summary) == 48
    assert sum(g["pending"] for g in summary) == 47
