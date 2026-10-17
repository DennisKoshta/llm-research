import pytest
from curiosity_study.causal_metrics import margin, paired_summary, zero_event_upper_bound


def test_answer_bias_and_information_bias_have_different_polarity_signatures():
    base = {"Yes": 2.0, "No": -1.0}
    yes_shift = {"Yes": 3.0, "No": -1.0}
    deltas = []
    for kind in ("new", "repeat"):
        deltas.append(margin(kind, {}, yes_shift, None)[0] - margin(kind, {}, base, None)[0])
    assert deltas == [1, -1]
    assert margin("count", {}, {"1": -3, "2": 5, "3": 2, "4": 0}, "2")[0] == 3


def test_action_margin_is_invariant_to_response_label_mapping():
    import itertools

    for index, services in enumerate(itertools.permutations(("report", "pattern", "stop"))):
        logits = {
            label: {"report": 4, "pattern": 1, "stop": 2}[service] for label, service in zip("ABC", services)
        }
        assert margin("instrument", {"index": index}, logits, None)[0] == 2


def test_world_bootstrap_and_zero_failure_bound():
    r = paired_summary([1, 1, 1])
    assert r == dict(worlds=3, mean=1.0, ci95=[1.0, 1.0])
    assert zero_event_upper_bound(100) == pytest.approx(0.0295130496)
    assert zero_event_upper_bound(50) > 0.05
