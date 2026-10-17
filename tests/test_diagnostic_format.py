import pytest
from curiosity_study.assignments import make_assignments
from curiosity_study.runner import diagnostic_assignments, run_diagnostic


@pytest.mark.parametrize("raw,valid", [("", False), ("AB", False), ("A", True), ("A because...", False)])
def test_diagnostic_requires_exactly_one_label(raw, valid):
    class FixedModel:
        def infer(self, *args, **kwargs):
            return {"raw": raw}

    assignment = next(a for a in diagnostic_assignments(make_assignments(1)) if a["kind"] == "accounting")
    record = run_diagnostic(assignment, FixedModel())
    assert record["valid"] is valid
    assert record["correct"] is (raw == record["expected"])
