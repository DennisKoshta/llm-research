"""New-method geometry and outer bindings, using synthetic/tiny CPU evidence."""

from contextlib import contextmanager
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from curiosity_study.artifacts import digest, read, write_once
import test_pain_profile_audit as original_tests

from test_pain_profile_engine import Tokenizer, items

calls = original_tests.calls

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("audit_pain_deterministic_control", ROOT / "scripts/audit_pain_deterministic_control.py")
wrapper = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = wrapper
SPEC.loader.exec_module(wrapper)
ASPEC = importlib.util.spec_from_file_location("deterministic_analysis_test", ROOT / "scripts/analyze_pain_deterministic_control.py")
analyzer = importlib.util.module_from_spec(ASPEC)
ASPEC.loader.exec_module(analyzer)


def original():
    spec = importlib.util.spec_from_file_location("_private_original_audit_test", ROOT / "executions/qwen25-pain-profile-control-v1/scripts/audit_pain_profile_control.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert wrapper.sha(module.__file__) == wrapper.AUDITOR_SHA
    return module


def check(result, **overrides):
    module, samples = original(), []
    module._geometry = lambda *args: wrapper.geometry(*args, samples=samples)
    parameters = dict(direction=torch.arange(-4, 4, dtype=torch.bfloat16) / 8,
        pain_unit=torch.arange(1, 9, dtype=torch.float32) / torch.arange(1, 9, dtype=torch.float32).norm(),
        tokenizer=Tokenizer(), vocabulary_size=32, coefficient=1, prefill_policy="full_prefix",
        eos_ids=[], pad_id=0, mode="compatibility", expected_device="cpu")
    parameters.update(overrides)
    return module.recompute_call(result, result["arrays"], items(), **parameters), samples


@pytest.mark.parametrize("name,options", [
    ("baseline", dict(coefficient=0)), ("unhooked", dict(coefficient=0, steering_enabled=False)),
    ("full", {}), ("final", dict(prefill_policy="final_prefix")),
    ("one_row", dict(one_row_only=0)), ("benchmark", dict(mode="benchmark")),
    ("prefill_fault", dict(expected_fault="after_prefill")),
    ("decode_fault", dict(expected_fault="after_first_decode")),
])
def test_real_bf16_all_frozen_masks_sampling_and_faults(calls, name, options):
    checked, samples = check(calls[name], **options)
    assert all(checked["checks"].values())
    assert checked["expected_fault_observed"] if "fault" in name else checked["passed"]
    assert samples and all(s["finite"] and s["exact_bf16"] and s["untreated_exact"] for s in samples)
    assert all(s["norms_used_as_gate"] is False for s in samples)
    if name == "benchmark":
        assert {s["physical_row"] for s in samples} == {0}
        assert len(samples[0]["sampled_positions"]) == 2
    elif name == "final":
        assert all(s["active_positions"] == 1 for s in samples)
    elif name == "full":
        assert samples[0]["active_positions"] == len(items()[0]["prompt_ids"])


def test_new_norm_method_does_not_retroactively_pass_old_norm_gate(calls):
    changed = deepcopy(calls["full"])
    # Deliberately different finite GPU reduction descriptors. Exact BF16
    # samples remain the numerical admission criterion under this NEW method.
    for frame in changed["evidence"]["forwards"]:
        for row in frame["geometry"]:
            row["achieved_delta_norm_min"] = row["achieved_delta_norm_max"] = 999.0
    checked, samples = check(changed)
    assert checked["passed"] and all(s["gpu_float32_norm_min"] == 999 for s in samples)
    assert all(s["cpu_float64_norm_max"] < 5 for s in samples)
    module = original()
    params = dict(direction=torch.arange(-4, 4, dtype=torch.bfloat16) / 8,
        pain_unit=torch.arange(1, 9, dtype=torch.float32) / torch.arange(1, 9, dtype=torch.float32).norm(),
        tokenizer=Tokenizer(), vocabulary_size=32, coefficient=1, prefill_policy="full_prefix",
        eos_ids=[], pad_id=0, mode="compatibility", expected_device="cpu")
    with pytest.raises(ValueError, match="outside observed norm"):
        module.recompute_call(changed, changed["arrays"], items(), **params)


@pytest.mark.parametrize("change", ["missing_sample", "changed_element", "mask", "rng", "dtype", "sample_position"])
def test_new_method_preserves_non_norm_fail_closed_checks(calls, change):
    changed = deepcopy(calls["full"])
    frame, key = changed["evidence"]["forwards"][0], "f00_r00_after"
    if change == "missing_sample":
        frame["geometry"][0].pop("saved_before")
    elif change == "changed_element":
        changed["arrays"][key][0, 0] = np.float32(99)
        binding = wrapper.array_binding(changed["arrays"][key])
        changed["evidence"]["arrays"][key] = binding
        frame["geometry"][0]["saved_after"].update(binding)
        frame["geometry"][0]["after_sha256"] = digest(wrapper.tensor_binding(torch.from_numpy(changed["arrays"][key]).bfloat16()))
    elif change == "mask":
        frame["steering_mask"][0][0] = 1 - frame["steering_mask"][0][0]
    elif change == "rng":
        changed["evidence"]["forwards"][1]["sampling"]["rng_state_before_sha256"][0] = "0" * 64
    elif change == "dtype":
        frame["activation_dtypes"]["after"] = "torch.float32"
    else:
        frame["geometry"][0]["saved_before"]["positions"] = [1]
    with pytest.raises(ValueError):
        check(changed)


def test_repeat_criterion_still_rejects_same_tokens_different_logits(calls):
    changed = deepcopy(calls["baseline"])
    changed["evidence"]["forwards"][1]["full_fp32_logits_sha256"][1] = "0" * 64
    assert changed["records"] == calls["baseline"]["records"]
    with pytest.raises(ValueError, match="full-vocabulary"):
        original()._repeat(calls["baseline"], changed)


def test_adapter_names_only_and_collision_rejection():
    assert wrapper.canonical_adapter_files({"./a": "1" * 64}) == {"a": "1" * 64}
    for names in ({"./a": "1" * 64, "a": "1" * 64}, {"a/../b": "1" * 64}, {"a//b": "1" * 64}):
        with pytest.raises(ValueError):
            wrapper.canonical_adapter_files(names)


def outer_fixture(tmp_path, *, failed=False):
    run = tmp_path
    policy = dict(wall_seconds=900)
    preset = dict(cublas_workspace_config=":4096:8", deterministic_algorithms=True, deterministic_warn_only=False,
        cudnn_deterministic=True, cudnn_benchmark=False, cuda_matmul_allow_tf32=True, cudnn_allow_tf32=True,
        allow_bf16_reduced_precision_reduction=True, float32_matmul_precision="high")
    manifest = dict(preset=preset, assigned_calls=dict(setup=21), assigned_rows=dict(setup=612),
        diagnostic_files={"pain_deterministic_control_worker.py": "a" * 64})
    admission = dict(created=100, deadline=1000, policy=policy)
    expected_preset = wrapper.effective_preset(preset)
    inner = run / "original_execution/phases/setup"
    summary = dict(complete=not failed, integrity_passed=not failed, passed=not failed,
        observed_calls=11 if failed else 21, completed_rows=292 if failed else 548,
        expected_fault_rows=0 if failed else 64, missing_rows=320 if failed else 0, execution_error_rows=0)
    for name, value in (("summary", summary), ("identity", {}), ("started", {}), ("outcomes", [])):
        write_once(inner / (name + ".json"), value)
    forecast = dict(benchmark_seconds=1000, limit_seconds=2880, passed=True,
        load_seconds=100, slowest_primary_cohort_seconds=6, safety_multiplier=2, benchmark_cohorts=65, export_seconds=120)
    raw = dict(identity_sha256="i", forecast=forecast)
    receipt = dict(schema_version=1, phase="setup", outer_manifest_sha256=digest(manifest), manifest_sha256=digest(manifest),
        admission_sha256=digest(admission), original_manifest_sha256=wrapper.ORIGINAL_MANIFEST_SHA,
        original_source_sha256=wrapper.SOURCE_SHA, original_execution_root="original_execution",
        worker_byte_sha256="a" * 64, reference_worker_byte_sha256=wrapper.REFERENCE_WORKER_SHA,
        execution_deadline_unix=940, assigned_calls=21, assigned_rows=612,
        original_campaign_remains_failed=True, original_scientific_rules_unchanged=True,
        independent_audit_required=True, behavior_gates_used=False, retry_allowed=False,
        geometry_policy="Exact BF16 elements; CPU float64 norms descriptive only in independent audit",
        finished_unix=200, error=None, effective_preset=expected_preset, final_effective_preset=expected_preset,
        forecast=dict(forecast, limit_seconds=1440), **summary)
    for key, name in (("original_summary", "summary"), ("original_identity", "identity"),
                      ("original_started", "started"), ("original_outcomes", "outcomes")):
        receipt[key] = wrapper.artifact_binding(run, inner / (name + ".json"))
    write_once(run / "phases/setup/started.json", dict(phase="setup", outer_manifest_sha256=digest(manifest),
        admission_sha256=digest(admission), retry_allowed=False, created_unix=110))
    write_once(run / "phases/setup/phase-execution.json", receipt)
    return dict(run=run, manifest=manifest), admission, raw, receipt


@pytest.mark.parametrize("failed", [False, True])
def test_outer_receipt_preserves_fixed_failed_denominators(tmp_path, failed):
    inputs, admission, raw, receipt = outer_fixture(tmp_path, failed=failed)
    observed = wrapper.execution_receipt(inputs, "setup", admission, raw)
    assert observed["complete"] is observed["passed"] is (not failed)
    assert observed["binding"]["content_sha256"] == digest(receipt)
    assert receipt["missing_rows"] == (320 if failed else 0)


@pytest.mark.parametrize("field", ["admission_sha256", "original_summary", "effective_preset", "execution_deadline_unix", "passed"])
def test_outer_receipt_nesting_and_preset_tamper_rejected(tmp_path, field):
    inputs, admission, raw, receipt = outer_fixture(tmp_path)
    if field == "original_summary":
        receipt[field]["byte_sha256"] = "0" * 64
    elif field == "effective_preset":
        receipt[field]["deterministic_algorithms"] = False
    elif field == "passed":
        receipt[field] = False
    else:
        receipt[field] = "changed"
    (tmp_path / "phases/setup/phase-execution.json").write_text(__import__("json").dumps(receipt))
    with pytest.raises(ValueError):
        wrapper.execution_receipt(inputs, "setup", admission, raw)


def test_verify_existing_does_not_rewrite_and_rejects_changed_receipt(tmp_path, monkeypatch):
    receipt = dict(passed=False, complete=False, assigned_rows=612, missing_rows=320)
    path = tmp_path / "phases/setup/audit.json"
    write_once(path, receipt)
    before = path.read_bytes()
    monkeypatch.setattr(wrapper, "audit", lambda run, phase: dict(receipt))
    assert wrapper.verify_existing(tmp_path, "setup") == receipt and path.read_bytes() == before
    receipt["passed"] = True
    with pytest.raises(ValueError, match="reproduce"):
        wrapper.verify_existing(tmp_path, "setup")


@pytest.mark.parametrize("incomplete", [True, False])
def test_unchanged_analyzer_keeps_authentic_missing_or_failed_integrity_unavailable(tmp_path, monkeypatch, incomplete):
    old_manifest = read(ROOT / "runs/qwen25-pain-profile-control-v1/manifest.json")
    spec = importlib.util.spec_from_file_location("_old_analysis_new_test", ROOT / "executions/qwen25-pain-profile-control-v1/scripts/analyze_pain_profile_control.py")
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    rows = [dict(**{k: a[k] for k in old.METADATA}, status="completed", category="other", valid=True, cap_hit=False)
            for a in old_manifest["plan"]["assignments"]]
    if incomplete:
        rows[0].update(status="missing", category=None, valid=None, cap_hit=None)
    manifest = dict(original_manifest=old_manifest, diagnostic_files={Path(analyzer.__file__).name: wrapper.sha(analyzer.__file__)})
    receipt = dict(manifest_sha256=digest(manifest), phase="benchmark", assigned_rows=2020, outcomes=rows,
        complete=not incomplete, passed=False, original_manifest_sha256=wrapper.ORIGINAL_MANIFEST_SHA,
        source_sha256=wrapper.SOURCE_SHA, archive_byte_sha256=wrapper.ARCHIVE_SHA,
        freeze_byte_sha256="f" * 64, auditor_byte_sha256=wrapper.sha(wrapper.__file__))
    write_once(tmp_path / "phases/benchmark/audit.json", receipt)
    monkeypatch.setattr(analyzer, "audit", lambda *args: deepcopy(receipt))
    monkeypatch.setattr(analyzer, "frozen_inputs", lambda run: dict(manifest=manifest))
    @contextmanager
    def context(inputs):
        yield None, old, None, None
    monkeypatch.setattr(analyzer, "original_runtime", context)
    result = analyzer.analyze_run(tmp_path)
    assert result["assigned"] == len(result["outcomes"]) == 2020 and not result["integrity_passed"]
    assert result["decision"]["full_mask_positive_control"]["classification"] == "incomplete_or_integrity_failed"
    assert result["original_analyzer_byte_sha256"] == wrapper.ANALYZER_SHA
def test_benchmark_admission_recursively_reproduces_exact_new_setup_receipt(tmp_path):
    run = tmp_path / "runs" / wrapper.NAME
    manifest = dict(synthetic="new campaign")
    policy = dict(wall_seconds=1800, maximum_liability_usd=2.5657344)
    setup = dict(phase="setup", complete=True, passed=True, benchmark_forecast_seconds=1000,
                 manifest_sha256=digest(manifest), method=wrapper.METHOD, auditor_byte_sha256=wrapper.sha(wrapper.__file__))
    names = ("admissions/setup.json", "original_execution/phases/setup/summary.json", "phases/setup/phase-execution.json",
             "phases/setup/audit.json", "launches/setup-outcome.json")
    for name, value in zip(names, ({}, {}, {}, setup, dict(all_stopped_zero_tasks=True, export_error=None, failure=None)), strict=True):
        write_once(run / name, value)
    admission = dict(phase="benchmark", manifest_sha256=digest(manifest), policy=policy,
        phase_consumed=True, retry_allowed=False, created=1000, deadline=2800,
        maximum_liability_usd=policy["maximum_liability_usd"], prior_evidence={n: wrapper.sha(run / n) for n in names})
    write_once(run / "admissions/benchmark.json", admission)
    write_once(tmp_path / "campaign_admissions" / wrapper.NAME / "benchmark.json", admission)
    inputs = dict(run=run, manifest=manifest, project=tmp_path, policies=dict(benchmark=policy))
    observed = []
    def reproduce(path, phase):
        observed.append((path, phase))
        return deepcopy(setup)
    assert wrapper.admission_binding(inputs, "benchmark", reproduce_setup=reproduce) == admission
    assert observed == [(run, "setup")]
    with pytest.raises(ValueError, match="does not reproduce"):
        wrapper.admission_binding(inputs, "benchmark", reproduce_setup=lambda *args: dict(setup, passed=False))
    setup["method"] = "altered"
    with pytest.raises(ValueError, match="does not reproduce"):
        wrapper.admission_binding(inputs, "benchmark", reproduce_setup=reproduce)


def test_outer_loader_failure_keeps_all612_missing(tmp_path):
    inputs, admission, raw, receipt = outer_fixture(tmp_path)
    inner = tmp_path / "original_execution/phases/setup"
    for path in inner.glob("*.json"):
        path.unlink()
    for key in ("original_summary", "original_identity", "original_started", "original_outcomes"):
        receipt[key] = None
    receipt.update(observed_calls=0, completed_rows=0, expected_fault_rows=0, missing_rows=612,
        execution_error_rows=0, complete=False, passed=False, integrity_passed=False,
        error=dict(type="RuntimeError", message="load failed"), effective_preset=None,
        final_effective_preset=None, forecast=None)
    (tmp_path / "phases/setup/phase-execution.json").write_text(__import__("json").dumps(receipt))
    raw.update(identity_sha256=None, forecast=None)
    checked = wrapper.execution_receipt(inputs, "setup", admission, raw)
    assert not checked["passed"] and not checked["complete"] and not checked["integrity_passed"]
    assert checked["forecast"] is None


def test_stricter_new_forecast_cannot_be_rescued_by_old2880_limit(tmp_path):
    inputs, admission, raw, receipt = outer_fixture(tmp_path)
    raw["forecast"].update(benchmark_seconds=1500)
    receipt["forecast"] = dict(raw["forecast"], limit_seconds=1440, passed=False)
    receipt["passed"] = False
    (tmp_path / "phases/setup/phase-execution.json").write_text(__import__("json").dumps(receipt))
    checked = wrapper.execution_receipt(inputs, "setup", admission, raw)
    assert checked["complete"] and checked["integrity_passed"] and not checked["passed"]
    assert not checked["checks"]["new_setup_forecast_passed"]
def test_detected_final_policy_failure_is_retained_without_admission(tmp_path):
    inputs, admission, raw, receipt = outer_fixture(tmp_path)
    receipt["final_effective_preset"] = dict(receipt["final_effective_preset"], deterministic_algorithms=False)
    receipt.update(error=dict(type="ValueError", message="Deterministic policy changed after execution"),
                   passed=False, integrity_passed=False, forecast=None)
    (tmp_path / "phases/setup/phase-execution.json").write_text(__import__("json").dumps(receipt))
    checked = wrapper.execution_receipt(inputs, "setup", admission, raw)
    assert checked["complete"] and not checked["passed"] and not checked["integrity_passed"]
    assert not checked["checks"]["deterministic_preset_before_and_after"]


def test_before_worker_failure_enumerates_all_assignments_without_writing_execution(tmp_path, monkeypatch):
    source = ROOT / "runs/qwen25-pain-profile-control-v1"
    manifest = dict(original_manifest=read(source / "manifest.json"), original_freeze=read(source / "freeze.json"),
        diagnostic_files={"synthetic": "a" * 64}, preset=outer_fixture(tmp_path / "fixture")[0]["manifest"]["preset"],
        deterministic_reference={"synthetic": True})
    run = tmp_path / "campaign"
    write_once(run / "freeze.json", {"synthetic": True})
    inputs = dict(run=run, manifest=manifest, original_run=source)
    monkeypatch.setattr(wrapper, "frozen_inputs", lambda run: inputs)
    monkeypatch.setattr(wrapper, "admission_binding", lambda *args, **kwargs: {})
    @contextmanager
    def runtime(inputs):
        yield original(), None, dict(freeze=dict(archive_sha256=wrapper.ARCHIVE_SHA)), []
    monkeypatch.setattr(wrapper, "original_runtime", runtime)
    checked = wrapper.audit(run, "setup")
    assert not checked["complete"] and not checked["passed"] and not checked["inner_historical_metadata_present"]
    assert checked["assigned_rows"] == checked["missing_rows"] == len(checked["outcomes"]) == 612
    assert not (run / "original_execution").exists()
