"""Local synthetic diagnostic admission and pricing; no cloud operations."""

from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "infra"))
SPEC = importlib.util.spec_from_file_location("pain_deterministic_controller_test", ROOT / "infra/launch_pain_deterministic_diagnostic.py")
controller = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controller)

STATE = dict(all_stopped_zero_tasks=True, available_compute_estimate_usd=100.0,
             timestamp_utc="synthetic", phase_compute_estimate_usd=20.0)


@pytest.fixture
def manifest():
    return dict(name=controller.NAME, assigned_calls=5, assigned_rows=148,
        resources=dict(app_name=controller.APP, gpu="H100", cpu=2, memory_gib=96,
            wall_seconds=900, teardown_seconds=120, rate_usd_per_second=controller.RATE,
            ceiling_usd=2.0, maximum_liability_usd=1020 * controller.RATE, max_containers=1, retries=0))


def ledger(project):
    return project / "campaign_admissions" / controller.NAME / "admission.json"


def test_once_only_claim_is_global_mirrored_and_exactly_bounded(tmp_path, manifest):
    result = controller.claim(tmp_path, manifest, STATE, now=1000.0)
    assert result["created"] == 1000 and result["deadline"] == 1900
    assert result["phase_consumed"] is True and result["retry_allowed"] is False
    assert result["maximum_liability_usd"] == pytest.approx(1.3630464)
    assert json.loads(ledger(tmp_path).read_text()) == result
    assert json.loads((tmp_path / "runs" / controller.NAME / "admission.json").read_text()) == result
    with pytest.raises(FileExistsError):
        controller.claim(tmp_path, manifest, STATE, now=2000.)


def test_concurrent_duplicate_claims_cannot_reserve_twice(tmp_path, manifest):
    def attempt():
        try:
            return controller.claim(tmp_path, manifest, STATE, now=1000.)
        except FileExistsError:
            return "already consumed"
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: attempt(), range(2)))
    assert sum(isinstance(x, dict) for x in results) == 1
    assert results.count("already consumed") == 1


def test_interrupted_mirror_does_not_reopen_global_admission(tmp_path, manifest, monkeypatch):
    def failure(*args):
        raise OSError("synthetic local mirror interruption")
    monkeypatch.setattr(controller, "write_once", failure)
    with pytest.raises(OSError, match="mirror interruption"):
        controller.claim(tmp_path, manifest, STATE, now=1000.)
    assert ledger(tmp_path).exists()
    with pytest.raises(FileExistsError):
        controller.claim(tmp_path, manifest, STATE, now=2000.)


@pytest.mark.parametrize("field,value", [
    ("all_stopped_zero_tasks", False), ("available_compute_estimate_usd", 10.0),
    ("available_compute_estimate_usd", float("nan")),
])
def test_active_job_or_insufficient_budget_cannot_consume_admission(tmp_path, manifest, field, value):
    state = dict(STATE, **{field: value})
    with pytest.raises(ValueError):
        controller.claim(tmp_path, manifest, state, now=1000.)
    assert not ledger(tmp_path).exists()


@pytest.mark.parametrize("field,value", [
    ("wall_seconds", 901), ("ceiling_usd", 3), ("rate_usd_per_second", .001),
    ("max_containers", 2), ("retries", 1), ("gpu", "H200"),
])
def test_changed_resource_policy_cannot_consume_admission(tmp_path, manifest, field, value):
    manifest["resources"][field] = value
    with pytest.raises(ValueError, match="resource policy"):
        controller.claim(tmp_path, manifest, STATE, now=1000.)
    assert not ledger(tmp_path).exists()


@pytest.fixture
def frozen_files(tmp_path):
    root = tmp_path / "frozen"
    (root / "nested").mkdir(parents=True)
    (root / "worker.py").write_text("# synthetic frozen worker\n")
    (root / "nested/plan.md").write_text("Fixed prospective rules.\n")
    inventory = {str(p.relative_to(root)): controller.sha(p) for p in root.rglob("*") if p.is_file()}
    return root, inventory


def test_exact_frozen_inventory_is_accepted(frozen_files):
    controller.check_files(*frozen_files)


@pytest.mark.parametrize("change", ["bytes", "extra", "missing", "file_link"])
def test_frozen_inventory_rejects_tampered_or_linked_files(frozen_files, change, tmp_path):
    root, inventory = frozen_files
    file = root / "worker.py"
    if change == "bytes":
        file.write_text("# changed\n")
    elif change == "extra":
        (root / "unexpected.py").write_text("# unbound\n")
    elif change == "missing":
        file.unlink()
    else:
        original = tmp_path / "same-bytes.py"
        original.write_bytes(file.read_bytes())
        file.unlink()
        file.symlink_to(original)
    with pytest.raises(ValueError):
        controller.check_files(root, inventory)


def test_named_diagnostic_costs_h100_and96gib_without_changing_legacy(tmp_path, monkeypatch):
    module = controller.budget
    apps = []
    names = [controller.APP, "curiosity-study-pilot-h100", "curiosity-study-pilot-h200", "unrelated-app"]
    for i, name in enumerate(names):
        apps.append({"App ID": str(i), "Description": name, "Created at": "2026-10-15T00:00:00+00:00",
                     "Stopped at": "2026-10-15T00:01:00+00:00", "State": "stopped", "Tasks": "0"})
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module.subprocess, "check_output", lambda *args, **kwargs: json.dumps(apps))
    observed = module.snapshot(dt.datetime(2026, 10, 15, tzinfo=dt.timezone.utc))
    assert observed["all_stopped_zero_tasks"]
    assert len(observed["app_charges"]) == 3
    by_id = {row["app_id"]: row for row in observed["app_charges"]}
    assert by_id["0"]["rate_usd_per_second"] == pytest.approx(controller.RATE)
    assert by_id["0"]["compute_estimate_usd"] == pytest.approx(60 * controller.RATE)
    assert by_id["0"]["gpu"] == "H100"
    assert by_id["0"]["resource_profile"] == "pain-deterministic-diagnostic-2cpu-96gib"
    assert by_id["1"]["rate_usd_per_second"] == module.RATES["H100"]
    assert by_id["2"]["rate_usd_per_second"] == module.RATES["H200"]
    assert observed["fresh_compute_allowance_usd"] == 300


@pytest.mark.parametrize("value", [1, "yes", None])
def test_nonboolean_allstopped_cannot_consume_admission(tmp_path, manifest, value):
    with pytest.raises(ValueError):
        controller.claim(tmp_path, manifest, dict(STATE, all_stopped_zero_tasks=value), now=1000.)
    assert not ledger(tmp_path).exists()


@pytest.mark.parametrize("now", [float("nan"), float("inf"), True])
def test_nonfinite_or_boolean_admission_time_rejected(tmp_path, manifest, now):
    with pytest.raises(ValueError):
        controller.claim(tmp_path, manifest, STATE, now=now)
    assert not ledger(tmp_path).exists()


@pytest.mark.parametrize("location", ["root", "empty_child"])
def test_frozen_directory_links_are_rejected(frozen_files, tmp_path, location):
    root, inventory = frozen_files
    if location == "root":
        alias = tmp_path / "linked-frozen"
        alias.symlink_to(root, target_is_directory=True)
        root = alias
    else:
        target = tmp_path / "outside-empty"
        target.mkdir()
        (root / "unexpected-directory").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError):
        controller.check_files(root, inventory)


def coverage_manifest(manifest):
    calls = [dict(call_index=index, call_id=f"call-{index}",
                  trial_ids=[f"call-{index}/row-{row}" for row in range(size)])
             for index, size in enumerate((32, 32, 20, 32, 32))]
    return dict(manifest, call_plan=calls)


def test_collection_coverage_retains_all148_when_worker_never_writes(tmp_path, manifest):
    bound = coverage_manifest(manifest)
    result = controller.collection_coverage(tmp_path, bound)
    assert result["assigned_rows"] == len(result["rows"]) == 148
    assert result["missing_rows"] == 148 and result["completed_rows"] == result["execution_error_rows"] == 0
    assert result["issues"] == [] and result["independent_numerical_audit"] is False
    assert {row["status"] for row in result["rows"]} == {"missing"}
    assert all(row["raw_record_sha256"] is None for row in result["rows"])
    saved = tmp_path / "runs" / controller.NAME / "collection-coverage.json"
    assert json.loads(saved.read_text()) == result


def test_collection_coverage_keeps_partial_records_and_corrupt_call_denominators(tmp_path, manifest):
    bound = coverage_manifest(manifest)
    calls = tmp_path / "runs" / controller.NAME / "calls"
    calls.mkdir(parents=True)
    first = bound["call_plan"][0]
    raw = [dict(trial_id=trial, answer_token_ids=[1, 2]) for trial in first["trial_ids"][:5]]
    envelope = dict(manifest_sha256=controller.digest(bound), call_spec=first,
                    engine_result=dict(records=raw))
    (calls / "0000.json").write_text(json.dumps(envelope))
    (calls / "0001.json").write_text('{"interrupted":')
    result = controller.collection_coverage(tmp_path, bound)
    assert result["assigned_rows"] == len(result["rows"]) == 148
    assert result["completed_rows"] == 5 and result["execution_error_rows"] == 59 and result["missing_rows"] == 84
    assert len(result["issues"]) == 1 and result["issues"][0]["call_id"] == "call-1"
    assert [row["raw_record_sha256"] for row in result["rows"][:5]] == [controller.digest(row) for row in raw]
    assert len({row["trial_id"] for row in result["rows"]}) == 148
    assert result["independent_numerical_audit"] is False
