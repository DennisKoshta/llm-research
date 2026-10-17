import importlib.util
import json
from pathlib import Path

import pytest

from curiosity_study import pain_profile_resources as resources


def test_campaign_is_within_its_original_finite_ceilings():
    bound = resources.campaign_liabilities()
    assert bound["setup"] == pytest.approx(3.0252288)
    assert bound["phases"]["benchmark"] == pytest.approx(4.9711104)
    assert bound["total"] == pytest.approx(7.9963392)
    assert bound["setup"] < 4 and bound["total"] < 12
    for phase in resources.POLICIES:
        p = resources.policy(phase)
        assert p["max_admissions"] == 1 and p["retries"] == 0
        assert p["resume_allowed"] is False and p["memory_gib"] == 96
        p["wall_seconds"] = 0
        assert resources.policy(phase)["wall_seconds"] > 0


def test_prices_refuse_unknown_family_names_and_preserve_legacy():
    assert resources.app_price("curiosity-study-pilot-h100") is None
    with pytest.raises(ValueError, match="Unpriced"):
        resources.app_price(resources.PREFIX + "unpriced")
    with pytest.raises(ValueError, match="Unknown"):
        resources.policy("retry")


def test_central_ledger_prices_cpu_and_larger_ram_without_changing_old_jobs(tmp_path, monkeypatch):
    import datetime as dt

    path = Path(__file__).resolve().parents[1] / "infra/budget.py"
    spec = importlib.util.spec_from_file_location("pain_budget_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    apps = []
    names = [resources.policy(p)["app_name"] for p in resources.POLICIES]
    names += ["curiosity-study-pilot-h100", "curiosity-study-pilot-h200", "unrelated-app"]
    for i, name in enumerate(names):
        apps.append({"App ID": str(i), "Description": name, "Created at": "2026-10-15T00:00:00+00:00",
                     "Stopped at": "2026-10-15T00:01:00+00:00", "State": "stopped", "Tasks": "0"})
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **kw: json.dumps(apps))
    state = module.snapshot(dt.datetime(2026, 10, 15, tzinfo=dt.timezone.utc))
    charges = state["app_charges"]
    assert len(charges) == 5 and state["all_stopped_zero_tasks"] is True
    assert [c["gpu"] for c in charges] == ["CPU", "H100", "H100", "H100", "H200"]
    rates = [resources.CPU_RATE, resources.H100_RATE, resources.H100_RATE,
             module.RATES["H100"], module.RATES["H200"]]
    assert [c["compute_estimate_usd"] for c in charges] == [60 * r for r in rates]
