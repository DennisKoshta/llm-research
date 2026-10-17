import importlib.util
import json
from pathlib import Path
from unittest.mock import patch
import pytest

spec = importlib.util.spec_from_file_location(
    "budget_guard", Path(__file__).resolve().parents[1] / "infra/budget.py"
)
budget = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budget)


def test_budget_refuses_active_and_exhausted():
    active = {
        "App ID": "test",
        "Description": "curiosity-study-test",
        "State": "ephemeral",
        "Tasks": "1",
        "Created at": "2026-10-01 00:00:00+00:00",
        "Stopped at": None,
    }
    with patch.object(budget.subprocess, "check_output", return_value=json.dumps([active])):
        with pytest.raises(RuntimeError, match="still active"):
            budget.admission_seconds()
    spent = dict(
        active,
        State="stopped",
        Tasks="0",
        **{"Created at": "2026-10-01 00:00:00+00:00", "Stopped at": "2026-10-10 00:00:00+00:00"},
    )
    with patch.object(budget.subprocess, "check_output", return_value=json.dumps([spent])):
        with pytest.raises(RuntimeError, match="exhausted"):
            budget.admission_seconds()


def test_budget_admits_bounded_run_with_reserve():
    with patch.object(budget.subprocess, "check_output", return_value="[]"):
        seconds, evidence = budget.admission_seconds()
    assert 180 <= seconds <= 7200
    assert (
        evidence["prior_compute_estimate_usd"] + (seconds + 120) * budget.RATE + budget.RESERVE
        <= budget.BUDGET
    )


def test_h100_uses_its_own_rate():
    app = {
        "App ID": "h100-test",
        "Description": "curiosity-study-pilot-h100",
        "State": "stopped",
        "Tasks": "0",
        "Created at": "2026-10-01 00:00:00+00:00",
        "Stopped at": "2026-10-01 01:00:00+00:00",
    }
    with patch.object(budget.subprocess, "check_output", return_value=json.dumps([app])):
        state = budget.snapshot()
        seconds, evidence = budget.admission_seconds(gpu="H100")
    charge = next(c for c in state["app_charges"] if c["app_id"] == "h100-test")
    assert charge["compute_estimate_usd"] == pytest.approx(3600 * budget.RATES["H100"])
    assert evidence["rate_usd_per_second"] == budget.RATES["H100"]
    assert seconds <= 7200


def test_fresh_allowance_is_additional_and_depletes():
    with patch.object(budget.subprocess, "check_output", return_value="[]"):
        state = budget.snapshot()
    # Historical artifacts are intentionally retained in this reconciliation.
    assert state["available_compute_estimate_usd"] <= 300
    assert state["fresh_compute_allowance_usd"] == 300
    assert state["historical_compute_baseline_usd"] == pytest.approx(63.35081244)
    assert state["available_compute_estimate_usd"] == pytest.approx(
        max(0, 300 - state["phase_compute_estimate_usd"])
    )


def test_h200_is_priced_separately():
    app = {
        "App ID": "h200-test",
        "Description": "curiosity-study-pilot-h200",
        "State": "stopped",
        "Tasks": "0",
        "Created at": "2026-10-14 00:00:00+00:00",
        "Stopped at": "2026-10-14 01:00:00+00:00",
    }
    with patch.object(budget.subprocess, "check_output", return_value=json.dumps([app])):
        state = budget.snapshot()
    charge = next(c for c in state["app_charges"] if c["app_id"] == "h200-test")
    assert charge["gpu"] == "H200"
    assert charge["compute_estimate_usd"] == pytest.approx(5.017536)
