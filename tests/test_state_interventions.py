import numpy as np
from curiosity_study.information_state import assignments, record_from_inference
from curiosity_study.state_interventions import delivery, make_patch


def test_report_delivery_obeys_actual_information_gain_and_pattern_independence():
    rows = assignments(1, kinds=("choice",))
    for a in rows:
        record = record_from_inference(a, {"raw": "A."})
        event = delivery(a, record)  # Index zero maps A to report and B to pattern.
        assert event["service"] == "report"
        assert event["uncertainty_after"] == int(a["query"] in a["known"])
        assert event["done"]
    selected = [a for a in rows if a["family"] == "objects"]
    patterns = [delivery(a, record_from_inference(a, {"raw": "B"}))["response"] for a in selected]
    assert len(set(patterns)) == 1
    assert delivery(rows[0], record_from_inference(rows[0], {"raw": "A and B"}))["service"] == "invalid"


def test_random_direction_is_reproducible_orthogonal_and_norm_matched():
    unit = np.zeros((1, 1, 2, 8))
    unit[:, :, :, 0] = 1.0
    data = dict(
        unit=unit,
        contrast_norm=np.array([[[2.5, 2.5]]]),
        layers=np.array([2]),
        positions=np.array(["record", "final"]),
    )
    condition = dict(name="random-control", layer=2, position="record", direction="random_0", dose=0.5)
    first = make_patch(condition, data, 123)
    assert first == make_patch(condition, data, 123)
    vector = np.array(first["direction"])
    np.testing.assert_allclose(np.linalg.norm(vector), 1.0)
    np.testing.assert_allclose(vector[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(vector * first["dose"]), 1.25)
    second = make_patch({**condition, "direction": "random_1"}, data, 123)
    assert not np.allclose(vector, second["direction"])
