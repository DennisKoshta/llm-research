"""Prospective call inventory and durable failure tests; no published model runs."""

from copy import deepcopy
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest
from transformers import AutoTokenizer

from curiosity_study import pain_profile_design as design
from curiosity_study import pain_profile_engine as engine
from curiosity_study import pain_profile_execution as execution
from curiosity_study import pain_profile_resources as resources
from curiosity_study.artifacts import digest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def material():
    plan = design.assignment_plan(design.load_sources(ROOT / "references/pain-axis-positive-control"))
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / "references/pain-axis-assets/metadata/report_dataset_full_1684p_20260831-065321",
        local_files_only=True, trust_remote_code=False,
    )
    preflight = design.tokenize_plan(plan, tokenizer, generation_eos_ids=design.BASE_GENERATION_EOS_IDS)
    calls = execution.setup_call_plan(plan, preflight)
    manifest = dict(assay=resources.ASSAY, revision=design.REVISION, plan=plan, plan_sha256=digest(plan),
                    preflight=preflight, preflight_sha256=digest(preflight),
                    setup_call_plan=calls, setup_call_plan_sha256=digest(calls),
                    resources={phase: resources.policy(phase) for phase in resources.POLICIES}, source_sha256="a" * 64)
    return manifest, tokenizer


def install_fake_engine(monkeypatch, *, fail_call=None, malformed=False, break_repeat=False, absent_decode=False,
                        break_prefill=False):
    """Mock numerical work only; real artifact executor and source text/parser run."""
    seen = []

    def fake(model, tokenizer, layers, **kwargs):
        items = kwargs["items"]
        index = len(seen)
        seen.append(deepcopy({key: value for key, value in kwargs.items() if key not in ("direction_bf16", "pain_unit_float32")}))
        coefficients = [kwargs["coefficient"] if kwargs["one_row_only"] is None or row == 0 else 0 for row in range(len(items))]
        frames, records = [], []
        expected = kwargs["inject_failure_at"]
        forwards = 1 if expected == "after_prefill" or absent_decode else 2
        for forward in range(forwards):
            geometry = [dict(trial_id=item["trial_id"], coefficient=coefficient,
                             achieved_delta_norm_min=coefficient, before_sha256="b" * 64)
                        for item, coefficient in zip(items, coefficients)]
            frame = dict(phase="prefill" if forward == 0 else "decode", geometry=geometry,
                         full_fp32_logits_sha256=[digest([item["gen_seed"], float(coefficient), kwargs["prefill_policy"] if coefficient else None])
                                                 for item, coefficient in zip(items, coefficients)],
                         input_ids=[[3]] * len(items), attention_mask=[[1]] * len(items), position_ids=[[forward]] * len(items),
                         cache_length_before=forward, cache_length_after=forward + 1,
                         prepatch_pain_projection=[0] * len(items), post_layer61_pain_projection=[0] * len(items))
            if not expected or forward < forwards - 1:
                frame["sampling"] = dict(alive_before=[True] * len(items), uniforms=[0.25] * len(items),
                                         rng_state_before_sha256=["c" * 64] * len(items),
                                         rng_state_after_sha256=["d" * 64] * len(items), selected_token_ids=[3] * len(items))
            frames.append(frame)
        if break_prefill and index == 1:
            frames[0]["geometry"][0]["before_sha256"] = "f" * 64
        if break_repeat and index == 10:
            frames[0]["full_fp32_logits_sha256"][0] = "f" * 64
        for item in items:
            raw = "Unparseable." if malformed else item["target_name"]
            records.append(dict(trial_id=item["trial_id"], text=raw, cap_hit=False,
                                answer_token_ids=tokenizer.encode(raw, add_special_tokens=False)))
        error = None
        if expected:
            records = []
            error = dict(type="RuntimeError", message=f"controlled compatibility fault: {expected}")
        if index == fail_call:
            records = []
            error = dict(type="RuntimeError", message="terminal synthetic failure")
        arrays = {"sample_before": np.zeros((1, 2), dtype=np.float32), "sample_after": np.ones((1, 2), dtype=np.float32)}
        evidence = dict(complete=error is None, integrity_passed=error is None, error=error,
                        attempted_forwards=forwards, incomplete_forward=None, forwards=frames,
                        hooks_before={}, hooks_after={}, checks=dict(hooks_clean=True, geometry=True),
                        arrays={key: engine.array_binding(value) for key, value in arrays.items()})
        return dict(records=records, evidence=evidence, arrays=arrays)

    monkeypatch.setattr(execution, "validate_runtime", lambda *args: None)
    monkeypatch.setattr(engine, "generate_segment", fake)
    return seen


def run(phase, path, material, **overrides):
    manifest, tokenizer = material
    kwargs = dict(phase=phase, run=path, manifest=manifest, model=SimpleNamespace(training=False),
                  tokenizer=tokenizer, layers=[], vectors=dict(pain=None, fear=None, pain_unit_float32=None),
                  identity=dict(load_seconds=5.0), deadline=time.time() + 100)
    kwargs.update(overrides)
    return execution.execute(**kwargs)


def test_full_fixed_call_plan_and_source_companions(material):
    manifest, _ = material
    calls = execution.setup_calls(manifest)
    assert len(calls) == 21 and sum(len(call["trial_ids"]) for call in calls) == 612
    assert [call["source_cohort"] for call in calls[:10]] == [0] * 5 + [12] * 5
    assert [call["arm"] for call in calls[:5]] == list(design.ARMS)
    assert [call["arm"] for call in calls[5:10]] == list(design.ARMS)
    assert [call["arm"] for call in calls[10:15]] == list(design.ARMS)
    assert calls[15]["steering_enabled"] is False
    assert calls[16]["one_row_only"] == 0
    assert [calls[i]["inject_failure_at"] for i in (17, 19)] == ["after_prefill", "after_first_decode"]
    assert [calls[i]["fault_call_id"] for i in (18, 20)] == [calls[17]["call_id"], calls[19]["call_id"]]
    benchmark = execution.benchmark_calls(manifest)
    assert len(benchmark) == 65 and sum(len(call["trial_ids"]) for call in benchmark) == 2020
    assert [trial for call in benchmark for trial in call["trial_ids"]] == [row["trial_id"] for row in manifest["plan"]["assignments"]]
    assert all(execution.engine_items(manifest, call)[0]["prompt_ids"][-1] == 198 for call in calls)


@pytest.mark.parametrize("phase", ["setup", "benchmark"])
def test_complete_phase_is_durable_and_does_not_gate_on_malformed_behavior(tmp_path, monkeypatch, material, phase):
    seen = install_fake_engine(monkeypatch, malformed=True)
    summary = run(phase, tmp_path, material)
    assert summary["complete"] and summary["integrity_passed"] and summary["passed"]
    assert summary["behavior_gates_used"] is False
    assert len(seen) == summary["observed_calls"] == (21 if phase == "setup" else 65)
    assert summary["assigned_rows"] == (612 if phase == "setup" else 2020)
    assert summary["expected_fault_rows"] == (64 if phase == "setup" else 0)
    directory = tmp_path / "phases" / phase
    rows = json.loads((directory / "outcomes.json").read_text())
    assert len(rows) == summary["assigned_rows"]
    assert all(row["category"] == "malformed" for row in rows if row["status"] == "completed")
    assert digest(rows) == summary["outcomes_sha256"]
    for binding in summary["call_bindings"]:
        envelope = json.loads((directory / binding["path"]).read_text())
        assert digest(envelope) == binding["content_sha256"]
        assert envelope["elapsed_seconds"] == binding["total_seconds"]
        artifact = directory / "calls" / envelope["arrays_artifact"]["path"]
        with np.load(artifact, allow_pickle=False) as arrays:
            assert {key: engine.array_binding(arrays[key]) for key in arrays.files} == envelope["arrays_artifact"]["arrays"]
    if phase == "setup":
        primary = [row for row in summary["call_bindings"] if int(row["call_id"].split("-")[-1]) < 10]
        assert summary["forecast"]["benchmark_seconds"] == 5 + 2 * max(row["total_seconds"] for row in primary) * 65 + 120
    with pytest.raises(ValueError, match="no retry"):
        run(phase, tmp_path, material)


def test_terminal_generation_error_retains_first_call_and_all_missing_assignments(tmp_path, monkeypatch, material):
    seen = install_fake_engine(monkeypatch, fail_call=0)
    summary = run("benchmark", tmp_path, material)
    assert len(seen) == 1 and summary["observed_calls"] == 1
    assert not summary["complete"] and not summary["passed"]
    assert summary["execution_error_rows"] == 32 and summary["missing_rows"] == 1988
    envelope = json.loads((tmp_path / "phases/benchmark/calls/0000.json").read_text())
    assert envelope["engine_result"]["evidence"]["error"]["message"] == "terminal synthetic failure"
    assert envelope["arrays_artifact"] and envelope["passed"] is False


def test_repeat_mismatch_stops_without_another_attempt(tmp_path, monkeypatch, material):
    seen = install_fake_engine(monkeypatch, break_repeat=True)
    summary = run("setup", tmp_path, material)
    assert len(seen) == 11 and not summary["passed"]
    envelope = json.loads((tmp_path / "phases/setup/calls/0010.json").read_text())
    assert envelope["checks"]["same_source_repeat_identity"] is False
    assert summary["missing_rows"] > 0


@pytest.mark.parametrize("phase", ["setup", "benchmark"])
def test_changed_prepatch_state_across_arms_fails_even_with_valid_local_geometry(tmp_path, monkeypatch, material, phase):
    seen = install_fake_engine(monkeypatch, break_prefill=True)
    summary = run(phase, tmp_path, material)
    assert len(seen) == 2 and not summary["passed"]
    envelope = json.loads((tmp_path / "phases" / phase / "calls/0001.json").read_text())
    assert envelope["engine_result"]["evidence"]["integrity_passed"] is True
    assert envelope["checks"]["same_cohort_prefill_identity"] is False
    assert envelope["prefill_reference_call_id"] == f"{phase}-0000"


def test_absent_genuine_decode_fails_setup_without_inspecting_answer(tmp_path, monkeypatch, material):
    seen = install_fake_engine(monkeypatch, absent_decode=True)
    summary = run("setup", tmp_path, material)
    assert len(seen) == 2  # baseline, then first active arm lacking decode
    assert not summary["passed"]
    envelope = json.loads((tmp_path / "phases/setup/calls/0001.json").read_text())
    assert envelope["checks"]["active_decode_engaged"] is False


def test_expired_deadline_retains_all2020_as_missing_without_forward(tmp_path, monkeypatch, material):
    seen = install_fake_engine(monkeypatch)
    summary = run("benchmark", tmp_path, material, deadline=time.time() - 1)
    assert not seen and not summary["complete"] and summary["missing_rows"] == 2020
    assert summary["terminal_error"]["type"] == "TimeoutError"


@pytest.mark.parametrize("mutation", ["call", "call_hash", "resource_bool", "preflight", "source"])
def test_mutated_bound_inputs_fail_before_forward_or_output(tmp_path, monkeypatch, material, mutation):
    seen = install_fake_engine(monkeypatch)
    manifest = deepcopy(material[0])
    if mutation == "call":
        manifest["setup_call_plan"][16]["one_row_only"] = 1
        manifest["setup_call_plan_sha256"] = digest(manifest["setup_call_plan"])
    elif mutation == "call_hash":
        manifest["setup_call_plan_sha256"] = "0" * 64
    elif mutation == "resource_bool":
        manifest["resources"]["setup"]["max_admissions"] = True
    elif mutation == "preflight":
        manifest["preflight"]["requests"][0]["prompt_ids"][-1] = 199
    else:
        manifest["source_sha256"] = "unknown"
    with pytest.raises(ValueError):
        run("setup", tmp_path, material, manifest=manifest)
    assert not seen and not (tmp_path / "phases").exists()


def test_missing_forecast_data_or_slow_measurement_cannot_admit(material):
    calls = execution.setup_calls(material[0])
    assert not execution.forecast(dict(load_seconds=5), calls, {})["passed"]
    timings = {call["call_id"]: 30 for call in calls}
    result = execution.forecast(dict(load_seconds=5), calls, timings)
    assert result["benchmark_seconds"] == 4025 and not result["passed"]
    timings[calls[0]["call_id"]] = True
    with pytest.raises(ValueError, match="timing"):
        execution.forecast(dict(load_seconds=5), calls, timings)


def test_unexpected_exception_is_durable_not_an_expected_fault(tmp_path, monkeypatch, material):
    seen = install_fake_engine(monkeypatch, fail_call=17)
    summary = run("setup", tmp_path, material)
    assert len(seen) == 18 and not summary["passed"]
    envelope = json.loads((tmp_path / "phases/setup/calls/0017.json").read_text())
    assert envelope["checks"]["expected_controlled_failure"] is False
    assert all(row["status"] == "execution_error" for row in envelope["outcomes"])


def test_checkpoint_error_keeps_completed_call_and_stops(tmp_path, monkeypatch, material):
    seen = install_fake_engine(monkeypatch)

    def checkpoint():
        raise RuntimeError("synthetic volume checkpoint failure")

    with pytest.raises(RuntimeError, match="checkpoint failure"):
        run("benchmark", tmp_path, material, checkpoint=checkpoint)
    summary = json.loads((tmp_path / "phases/benchmark/summary.json").read_text())
    assert len(seen) == 1 and summary["completed_rows"] == 32 and not summary["passed"]
    assert summary["terminal_error"]["message"] == "synthetic volume checkpoint failure"


def test_runtime_guard_does_not_admit_cpu_model_as_published_compatibility():
    with pytest.raises(ValueError, match="H100"):
        execution.validate_runtime(SimpleNamespace(training=False), {}, dict(gpu="CPU"))


def test_one_row_isolation_ignores_later_inactive_rng_and_reduction_telemetry():
    frame = dict(input_ids=[[3]] * 2, attention_mask=[[1]] * 2, position_ids=[[0]] * 2,
                 full_fp32_logits_sha256=["a", "b"],
                 sampling=dict(alive_before=[True, True], uniforms=[0.1, 0.2],
                               rng_state_before_sha256=["c", "d"], rng_state_after_sha256=["e", "f"],
                               selected_token_ids=[3, 3]))
    reference = dict(records=[dict(trial_id=str(i), text="violet", answer_token_ids=[3], mean_proj=0.1) for i in range(2)],
                     evidence=dict(complete=True, forwards=[frame]))
    observed = deepcopy(reference)
    observed["records"][1]["mean_proj"] = 0.100000001
    extra = deepcopy(frame)
    extra["sampling"]["alive_before"] = [True, False]
    extra["sampling"]["uniforms"] = [0.7, 0.8]
    observed["evidence"]["forwards"].append(extra)
    assert execution.unaffected_equal(reference, observed)
    observed["evidence"]["forwards"][0]["full_fp32_logits_sha256"][1] = "changed"
    assert not execution.unaffected_equal(reference, observed)
