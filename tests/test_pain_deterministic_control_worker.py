"""New outer admission/runtime wrapper only; all inference is synthetic and CPU."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
import torch

from curiosity_study import artifacts, pain_profile_engine, pain_profile_execution as execution
import test_pain_profile_execution as original_tests
from test_pain_profile_execution import install_fake_engine
from test_pain_profile_deterministic_diagnostic_worker import settings as settings

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/pain_deterministic_control_worker.py"
spec = importlib.util.spec_from_file_location("new_control_worker_test", PATH)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


@pytest.fixture(scope="module")
def original():
    # Original frozen input metadata, and already released diagnostic receipts only.
    return worker.read(ROOT / "runs/qwen25-pain-profile-control-v1/manifest.json")


@pytest.fixture(scope="module")
def material():
    return original_tests.material.__wrapped__()


@pytest.fixture
def payload(tmp_path, monkeypatch, original):
    flat = tmp_path / "diagnostic"
    flat.mkdir()
    files = {
        PATH.name: PATH,
        "diagnostic_reference_worker.py": ROOT / "executions/qwen25-pain-deterministic-diagnostic-v1/diagnostic/pain_profile_deterministic_diagnostic_worker.py",
        "reference-manifest.json": ROOT / "runs/qwen25-pain-deterministic-diagnostic-v1/manifest.json",
        "reference-summary.json": ROOT / "runs/qwen25-pain-deterministic-diagnostic-v1/summary.json",
        "reference-independent-check.json": ROOT / "reports/pain-deterministic-diagnostic-v1-independent-check.json",
    }
    for name, source in files.items():
        shutil.copyfile(source, flat / name)
    monkeypatch.setattr(worker, "__file__", str(flat / PATH.name))
    reference = worker.read(flat / "reference-manifest.json")
    manifest = dict(name=worker.NAME, original_manifest=deepcopy(original), original_freeze=reference["original_freeze"],
        original_source_archive_sha256=worker.ORIGINAL_ARCHIVE_SHA,
        original_execution_files=reference["original_execution_files"],
        diagnostic_files={p.name: worker.sha(p) for p in flat.iterdir()},
        deterministic_reference=dict(manifest_sha256=worker.REFERENCE_MANIFEST_SHA,
            summary_byte_sha256=worker.REFERENCE_SUMMARY_SHA, independent_check_byte_sha256=worker.REFERENCE_CHECK_SHA),
        preset=deepcopy(worker.PRESET), assigned_calls=worker.ASSIGNED_CALLS.copy(), assigned_rows=worker.ASSIGNED_ROWS.copy(),
        resources=deepcopy(worker.POLICIES), campaign_ceiling_usd=5, benchmark_forecast_limit_seconds=1440,
        original_campaign_remains_failed=True, scientific_rules_changed=False, automatic_retry_allowed=False)
    run = tmp_path / worker.NAME
    worker.write_once(run / "manifest.json", manifest)
    admission = new_admission(manifest, "setup")
    worker.write_once(run / "admissions/setup.json", admission)
    original_root = ROOT / "executions/qwen25-pain-deterministic-diagnostic-v1/original"
    return original_root, run, manifest, admission


def new_admission(manifest, phase, prior=None):
    now = time.time() - 1
    return dict(phase=phase, manifest_sha256=worker.digest(manifest), created=now,
        deadline=now + worker.POLICIES[phase]["wall_seconds"], policy=worker.POLICIES[phase],
        phase_consumed=True, retry_allowed=False, maximum_liability_usd=worker.POLICIES[phase]["maximum_liability_usd"],
        prior_evidence={} if prior is None else prior)


def replace(path, value):
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def test_import_is_stdlib_only():
    code = ("import importlib.util,sys; s=importlib.util.spec_from_file_location('w'," + repr(str(PATH))
            + ");m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
            "assert 'torch' not in sys.modules;assert 'transformers' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True, timeout=10)


def test_real_input_pins_successful_preset_and_fixed_denominators(payload):
    root, run, manifest, admission = payload
    assert worker.validate_payload(root, run, "setup") == (manifest, admission)
    ref = worker.load_reference()
    assert worker.sha(ref.__file__) == worker.REFERENCE_WORKER_SHA
    assert worker.PRESET == {k: v for k, v in ref.PRESET.items() if k not in {"wall_seconds", "execution_reserve_seconds"}}
    assert len(execution.setup_calls(manifest["original_manifest"])) == 21
    assert len(execution.benchmark_calls(manifest["original_manifest"])) == 65


@pytest.mark.parametrize("change", ["resource_bool", "preset", "original", "deadline", "retry", "helper_bytes"])
def test_changed_contract_rejected_before_numerical_loading(payload, change):
    root, run, manifest, admission = payload
    if change == "resource_bool":
        manifest["resources"]["setup"]["max_containers"] = True
    elif change == "preset":
        manifest["preset"]["deterministic_warn_only"] = True
    elif change == "original":
        manifest["original_manifest"]["preflight"]["requests"][0]["gen_seed"] += 1
    elif change == "deadline":
        admission["deadline"] = admission["created"] + 901
    elif change == "retry":
        admission["retry_allowed"] = True
    else:
        path = Path(worker.__file__).parent / "diagnostic_reference_worker.py"
        path.write_text(path.read_text() + "\n# changed\n")
        manifest["diagnostic_files"][path.name] = worker.sha(path)
    admission["manifest_sha256"] = worker.digest(manifest)
    replace(run / "manifest.json", manifest)
    replace(run / "admissions/setup.json", admission)
    with pytest.raises(ValueError):
        worker.validate_payload(root, run, "setup")
    assert not (run / "phases").exists()


def install_runtime(monkeypatch, payload, material, *, fail_load=False, load_seconds=5, **fake_options):
    root, run, manifest, admission = payload
    reference = worker.load_reference()
    seen = install_fake_engine(monkeypatch, **fake_options)
    tokenizer = material[1]
    flags_seen = []

    def load_model(*args):
        flags_seen.append(reference.effective_preset(torch))
        assert flags_seen[-1]["deterministic_algorithms"] is True
        assert flags_seen[-1]["sdpa_backends"] == dict(math=True, flash=False, mem_efficient=False, cudnn=False)
        if fail_load:
            raise RuntimeError("synthetic unavailable cached assets")
        return tokenizer, None, SimpleNamespace(training=False), [], dict(pain=None, fear=None, pain_unit_float32=None), dict(load_seconds=load_seconds)

    assets = SimpleNamespace(check_environment=lambda path: {k: v for k, v in manifest["original_manifest"]["environment_lock"].items() if k != "path"},
                             load_model=load_model)
    reference.load_original = lambda *args: (torch, {"pain_profile_execution": execution,
        "pain_profile_engine": pain_profile_engine, "pain_profile_assets": assets, "artifacts": artifacts})
    monkeypatch.setattr(worker, "load_reference", lambda: reference)
    return seen, flags_seen


def setup_chain(run, manifest, receipt):
    audit = dict(manifest_sha256=worker.digest(manifest), phase="setup", passed=True, complete=True,
                 benchmark_forecast_seconds=receipt["forecast"]["benchmark_seconds"])
    worker.write_once(run / "phases/setup/audit.json", audit)
    worker.write_once(run / "launches/setup-outcome.json", dict(all_stopped_zero_tasks=True, export_error=None, failure=None))
    prior = {name: worker.sha(run / name) for name in worker.PRIOR_PATHS}
    admission = new_admission(manifest, "benchmark", prior)
    worker.write_once(run / "admissions/benchmark.json", admission)
    return admission


def test_actual_original_executor_spans_both_phases_without_rewriting_rows(payload, monkeypatch, material):
    root, run, manifest, admission = payload
    seen, flags = install_runtime(monkeypatch, payload, material, malformed=True)
    result = worker.execute(root, run, Path("/unused-cache"), "setup")
    assert result["passed"] is True and result["assigned_rows"] == 612 and result["expected_fault_rows"] == 64
    assert result["execution_deadline_unix"] == admission["deadline"] - 60
    assert worker.read(run / "manifest.json") == manifest
    assert worker.read(run / "original_execution/manifest.json") == manifest["original_manifest"]
    assert worker.read(run / "original_execution/freeze.json") == manifest["original_freeze"]
    summary = worker.read(run / result["original_summary"]["path"])
    assert summary["forecast"]["limit_seconds"] == 2880 and result["forecast"]["limit_seconds"] == 1440
    assert result["original_summary"] == worker.binding(run, run / "original_execution/phases/setup/summary.json")
    identity = worker.read(run / result["original_identity"]["path"])
    assert identity["admission_sha256"] == worker.digest(admission)
    assert identity["outer_manifest_sha256"] == worker.digest(manifest)
    assert identity["deterministic_preset"] == flags[0] == result["final_effective_preset"]
    benchmark_admission = setup_chain(run, manifest, result)
    assert benchmark_admission["created"] < admission["deadline"]  # Verified early shutdown is enough.
    benchmark = worker.execute(root, run, Path("/unused-cache"), "benchmark")
    assert benchmark["complete"] is benchmark["passed"] is True
    assert len(seen) == 86 and benchmark["assigned_rows"] == benchmark["completed_rows"] == 2020
    assert benchmark["effective_preset"] == flags[1] == flags[0]
    rows = worker.read(run / benchmark["original_outcomes"]["path"])
    assert len(rows) == 2020 and all(row["category"] == "malformed" for row in rows)
    assert benchmark["behavior_gates_used"] is False
    with pytest.raises(ValueError, match="no retry"):
        worker.execute(root, run, Path("/unused-cache"), "benchmark")


def test_loader_failure_retains_outer_denominator_and_original_binding(payload, monkeypatch, material):
    root, run, manifest, _ = payload
    seen, _ = install_runtime(monkeypatch, payload, material, fail_load=True)
    result = worker.execute(root, run, Path("/unused-cache"), "setup")
    assert not seen and result["passed"] is result["complete"] is False
    assert result["assigned_calls"] == 21 and result["missing_rows"] == result["assigned_rows"] == 612
    assert result["error"]["message"] == "synthetic unavailable cached assets" and result["original_summary"] is None
    assert worker.read(run / "original_execution/freeze.json") == manifest["original_freeze"]


def test_first_generation_failure_keeps_all_slots_and_durable_raw_failure(payload, monkeypatch, material):
    root, run, _, _ = payload
    seen, _ = install_runtime(monkeypatch, payload, material, fail_call=0)
    result = worker.execute(root, run, Path("/unused-cache"), "setup")
    assert len(seen) == 1 and result["passed"] is False
    assert result["execution_error_rows"] == 32 and result["missing_rows"] == 580
    assert result["original_summary"] and result["forecast"]["passed"] is False
    envelope = worker.read(run / "original_execution/phases/setup/calls/0000.json")
    assert envelope["engine_result"]["evidence"]["error"]["message"] == "terminal synthetic failure"


def test_old2880_forecast_cannot_override_new1440_limit(payload, monkeypatch, material):
    root, run, _, _ = payload
    install_runtime(monkeypatch, payload, material, load_seconds=1400)
    result = worker.execute(root, run, Path("/unused-cache"), "setup")
    assert result["complete"] is result["integrity_passed"] is True and result["passed"] is False
    assert 1440 < result["forecast"]["benchmark_seconds"] < 2880
    assert worker.read(run / result["original_summary"]["path"])["passed"] is True


@pytest.mark.parametrize("change", ["audit_failure", "byte_tamper", "forecast", "missing_evidence", "unverified_shutdown"])
def test_benchmark_cannot_bypass_independent_setup_chain(payload, monkeypatch, material, change):
    root, run, manifest, _ = payload
    seen, _ = install_runtime(monkeypatch, payload, material)
    result = worker.execute(root, run, Path("/unused-cache"), "setup")
    admission = setup_chain(run, manifest, result)
    path = run / "phases/setup/audit.json"
    audit = worker.read(path)
    if change == "audit_failure":
        audit["passed"] = False
    elif change == "forecast":
        audit["benchmark_forecast_seconds"] = 1441
    elif change == "missing_evidence":
        admission["prior_evidence"].pop("phases/setup/audit.json")
    elif change == "unverified_shutdown":
        path = run / "launches/setup-outcome.json"
        audit = worker.read(path)
        audit["all_stopped_zero_tasks"] = False
    else:
        audit["unbound_extra"] = True
    replace(path, audit)
    if change not in ("byte_tamper", "missing_evidence"):
        admission["prior_evidence"][str(path.relative_to(run))] = worker.sha(path)
    replace(run / "admissions/benchmark.json", admission)
    with pytest.raises(ValueError):
        worker.execute(root, run, Path("/unused-cache"), "benchmark")
    assert len(seen) == 21 and not (run / "phases/benchmark").exists()
