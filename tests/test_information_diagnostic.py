from curiosity_study.information_diagnostic import assignments, prompt, record_from_inference
from curiosity_study.single_curiosity import SingleTrial
from curiosity_study.oracle import information


def test_matched_factorial_and_oracle():
    rows = assignments(4)
    assert len(rows) == 4 * 2 * 2 * 3 * 2
    assert len({a["trial_id"] for a in rows}) == len(rows)
    for a in rows:
        trial = SingleTrial(a)
        bits = information(trial.task, trial.observations, trial.task.target_query)[
            "information_bits_uniform_prior"
        ]
        assert bits == (0 if a["condition"] == "resolved" else 2)
        _, labels, expected = prompt(a)
        if a["form"] != "explanation":
            assert expected in labels
        if a["form"] == "yes_no":
            expected_yes = (bits == 0) == (a["polarity"] == "already")
            assert (expected == "Yes") == expected_yes


def test_unclosed_reasoning_is_not_salvaged():
    a = next(a for a in assignments(1) if a["thinking"] and a["form"] == "yes_no")
    expected = prompt(a)[2]
    r = record_from_inference(a, {"raw": f"I think the answer is {expected}"})
    assert not r["valid"] and not r["correct"]
    r = record_from_inference(a, {"raw": f"reasoning\n</think>\n{expected}"})
    assert r["valid"] and r["correct"]
