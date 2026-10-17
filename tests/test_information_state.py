from curiosity_study.information_state import assignments, task, prompt, record_from_inference


def test_information_gain_varies_while_total_uncertainty_stays_fixed():
    rows = assignments(20, kinds=("new",))
    for a in rows:
        t = task(a)
        assert len(t["compatible"]) == 2
        assert t["truth"] in t["compatible"]
        assert t["uncertainty_bits"] == 1
        assert t["information_bits"] == int(a["query"] not in a["known"])
        # Independently enumerate observation outcomes, avoiding reuse of the entropy formula.
        possible_values = {t["candidates"][i][a["query"]] for i in t["compatible"]}
        assert len(possible_values) == (2 if a["query"] not in a["known"] else 1)
        swapped = {**a, "known": [1 - a["known"][0]]}
        assert task(swapped)["information_bits"] == 1 - t["information_bits"]


def test_polarity_and_exact_format():
    a = assignments(1, kinds=("new",))[0]
    b = {**a, "kind": "repeat"}
    assert prompt(a)[2] != prompt(b)[2]
    for kind, invalid in [("new", ""), ("count", "12"), ("memory", "AB")]:
        c = {**a, "kind": kind}
        assert not record_from_inference(c, {"raw": invalid})["valid"]
    c = {**a, "kind": "count"}
    assert prompt(c)[2] == "2"
    assert record_from_inference(c, {"raw": "2"})["correct"]


def test_known_none_and_all_controls():
    a = assignments(1, kinds=("new",))[0]
    for known, uncertainty, gain in [([], 2, 1), ([0, 1], 0, 0)]:
        t = task({**a, "known": known})
        assert t["uncertainty_bits"] == uncertainty
        assert t["information_bits"] == gain


def test_frozen_optional_period_grammar_rejects_extra_content():
    a = assignments(1, kinds=("new",))[0]
    expected = prompt(a)[2]
    assert record_from_inference(a, {"raw": expected + "."})["correct"]
    assert not record_from_inference(a, {"raw": expected + ".."})["valid"]
    assert not record_from_inference(a, {"raw": expected + ". Explanation follows."})["valid"]
