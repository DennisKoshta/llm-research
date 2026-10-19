import json
import time

import pytest

from v3 import ledger


@pytest.fixture()
def tmp_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "LEDGER_DIR", tmp_path)
    monkeypatch.setattr(ledger, "RES_FILE", tmp_path / "reservations.jsonl")
    monkeypatch.setattr(ledger, "billed_usd", lambda: 100.0)
    return tmp_path


def test_committed_uses_actual_else_reservation(tmp_ledger):
    ledger.reserve("a", 10.0, 1)
    ledger.reserve("b", 5.0, 2)
    ledger.release("a", 7.0)
    s = ledger.summary()
    assert s["committed_usd"] == 12.0 and s["running_jobs"] == ["b"] and s["running_gpus"] == 2


def test_reconcile_replaces_earlier_estimates_by_billing(tmp_ledger, monkeypatch):
    ledger.reserve("old", 300.0, 1)
    ledger.release("old", 280.0)                       # inflated wall-clock estimate
    assert ledger.summary()["committed_usd"] == 280.0
    time.sleep(0.01)
    ledger.reconcile("test")
    assert ledger.summary()["committed_usd"] == 100.0  # billed baseline
    time.sleep(0.01)
    ledger.reserve("new", 20.0, 1)
    assert ledger.summary()["committed_usd"] == 120.0  # baseline + later reservation
    ledger.release("new", 15.0)
    assert ledger.summary()["committed_usd"] == 115.0


def test_job_running_across_a_reconcile_keeps_its_reservation(tmp_ledger):
    ledger.reserve("live", 30.0, 2)
    time.sleep(0.01)
    ledger.reconcile()
    assert ledger.summary()["committed_usd"] == 130.0
    assert ledger.summary()["running_gpus"] == 2


def test_reserve_refuses_above_soft_stop_and_too_many_gpus(tmp_ledger, monkeypatch):
    monkeypatch.setattr(ledger, "billed_usd", lambda: ledger.SOFT_STOP_USD - 10.0)
    with pytest.raises(RuntimeError, match="budget"):
        ledger.reserve("x", 20.0, 1)
    monkeypatch.setattr(ledger, "billed_usd", lambda: 0.0)
    ledger.reserve("g1", 1.0, 10)
    with pytest.raises(RuntimeError, match="concurrency"):
        ledger.reserve("g2", 1.0, 7)
