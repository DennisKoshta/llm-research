import math
from collections import Counter
from curiosity_study import relational_information as rel


def test_relational_oracle_breaks_property_matching_shortcut():
    for a in rel.assignments(12):
        t = rel.task(a)
        assert a["query"] not in a["known"]
        assert len({tuple(c) for c in t["candidates"]}) == 4
        compatible = [c for c in t["candidates"] if c[2] == t["candidates"][t["truth"]][2]]
        assert len(compatible) == 2 and t["uncertainty_bits"] == 1
        counts = Counter(c[a["query"]] for c in compatible)
        expected_after = sum(n / 2 * math.log2(n) for n in counts.values())
        assert t["information_bits"] == 1 - expected_after == int(a["query"] != a["correlated"])


def test_polarity_reverses_with_table_implied_information():
    for a in rel.assignments(5, kinds=("new",)):
        text, _, expected = rel.prompt(a)
        other = rel.prompt({**a, "kind": "repeat"})[2]
        assert expected != other
        assert text.count("End of record.") == 1
        assert rel.record_from_inference(a, {"raw": expected + "."})["correct"]
        assert not rel.record_from_inference(a, {"raw": expected + " because"})["valid"]


def test_prose_and_table_have_identical_worlds_and_answer_keys():
    for a in rel.assignments(5):
        b = {**a, "template": "prose"}
        assert rel.task(a) == rel.task(b)
        assert rel.prompt(a)[1:] == rel.prompt(b)[1:]
        assert rel.context(a) != rel.context(b)
