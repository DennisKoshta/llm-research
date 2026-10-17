import importlib.util
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/audit_curiosity_coverage_bridge_lifecycle.py"
spec = importlib.util.spec_from_file_location("coverage_lifecycle", PATH)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def timestamp(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def evidence():
    created = 1792065600
    admission = dict(created=created, deadline=created + 28000, prior_phase_compute_estimate_usd=92.78286292)
    manifest = dict(workers=[dict(execution_group=g, app_name=f"curiosity-study-coverage-bridge-{g}-h200")
                            for g in audit.GROUPS], resources=dict(wall_seconds=28000, teardown_seconds=120,
        rate_usd_per_second=audit.RATE, maximum_liability_usd=3 * 28120 * audit.RATE, ceiling_usd=120))
    launches, outcomes, apps, charges = {}, {}, [], []
    for index, group in enumerate(audit.GROUPS):
        app_id, start, stop = f"ap-{index}", created + 10 + index, created + 100 + index
        launches[group] = dict(app_id=app_id, execution_group=group, started=start + 1)
        outcomes[group] = dict(app_id=app_id, execution_group=group, failure=None, export_error=None,
                              stop_errors=[], app_stopped_zero_tasks=True, exported_files=5)
        apps.append({"App ID": app_id, "Description": manifest["workers"][index]["app_name"],
                     "Created at": timestamp(start), "Stopped at": timestamp(stop), "State": "stopped", "Tasks": "0"})
        charges.append(dict(app_id=app_id, gpu="H200", app_wall_seconds=90,
                            rate_usd_per_second=audit.RATE, compute_estimate_usd=90 * audit.RATE))
    spent = admission["prior_phase_compute_estimate_usd"] + sum(c["compute_estimate_usd"] for c in charges)
    state = dict(apps=apps, app_charges=charges, all_stopped_zero_tasks=True,
                 fresh_compute_allowance_usd=300, phase_compute_estimate_usd=spent,
                 available_compute_estimate_usd=300-spent, storage_transfer_reserve_usd=10)
    return manifest, admission, launches, outcomes, state


def test_three_distinct_stopped_apps_reconstruct_cost_and_global_delta():
    result = audit.app_lifecycle(*evidence())
    assert result["passed"]
    assert result["total_compute_estimate_usd"] == pytest.approx(270 * audit.RATE)


def test_child_startup_does_not_move_common_deadline():
    manifest, admission, launches, outcomes, state = evidence()
    app = state["apps"][2]
    # Modest app duration fits its per-app allowance, but passes common teardown.
    app["Created at"] = timestamp(admission["created"] + 200)
    app["Stopped at"] = timestamp(admission["deadline"] + 121)
    seconds = 28000 + 121 - 200
    state["app_charges"][2].update(app_wall_seconds=seconds, compute_estimate_usd=seconds*audit.RATE)
    state["phase_compute_estimate_usd"] = admission["prior_phase_compute_estimate_usd"] + sum(
        c["compute_estimate_usd"] for c in state["app_charges"])
    state["available_compute_estimate_usd"] = 300-state["phase_compute_estimate_usd"]
    result = audit.app_lifecycle(manifest, admission, launches, outcomes, state)
    assert not result["checks"]["worker_2_stopped_within_liability"]
    assert not result["passed"]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "shared_id"])
def test_cannot_drop_replace_or_share_worker_apps(mutation):
    manifest, admission, launches, outcomes, state = evidence()
    if mutation == "missing":
        state["apps"].pop()
    elif mutation == "duplicate":
        duplicate = deepcopy(state["apps"][0])
        duplicate["App ID"] = "ap-replacement"
        state["apps"].append(duplicate)
    else:
        launches["worker_1"]["app_id"] = launches["worker_0"]["app_id"]
    with pytest.raises(ValueError):
        audit.app_lifecycle(manifest, admission, launches, outcomes, state)


@pytest.mark.parametrize("mutation", ["export_failure", "active", "wrong_rate", "unaccounted_spend"])
def test_failed_export_or_inconsistent_accounting_cannot_pass(mutation):
    manifest, admission, launches, outcomes, state = evidence()
    if mutation == "export_failure":
        outcomes["worker_0"]["export_error"] = "Interrupted download"
    elif mutation == "active":
        state["apps"][0].update(State="running", Tasks="1")
    elif mutation == "wrong_rate":
        state["app_charges"][0]["rate_usd_per_second"] *= 0.5
    else:
        state["phase_compute_estimate_usd"] += 1
    assert not audit.app_lifecycle(manifest, admission, launches, outcomes, state)["passed"]


def test_raw_inventory_binds_nested_files_and_only_excludes_named_root_outputs(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "audit.json").write_text("{}")
    worker = tmp_path / "workers/worker_0"
    worker.mkdir(parents=True)
    (worker / "audit.json").write_text("observed")
    first = audit.raw_inventory(tmp_path)
    assert set(first) == {"manifest.json", "workers/worker_0/audit.json"}
    (worker / "audit.json").write_text("changed")
    assert first != audit.raw_inventory(tmp_path)
    (tmp_path / "link").symlink_to(tmp_path / "manifest.json")
    with pytest.raises(ValueError, match="symlink"):
        audit.raw_inventory(tmp_path)
