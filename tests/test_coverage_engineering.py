"""CPU-only coverage geometry, finite inventory and durable failure tests."""

from copy import deepcopy
import json

import numpy as np
import pytest
import torch
from transformers import Qwen3ForCausalLM

from curiosity_study import coverage_engineering as engine
from curiosity_study.artifacts import digest
from test_state_backend import tiny_backend
from test_sustained_backend import requests
from test_satisfaction_capture import fake_model
from test_voluntary_capture import fake_forward


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def source_requests():
    return {cohort: [{**r, "source_trial_id": f"base-{cohort}-{i}", "seed": r["seed"] + cohort}
                     for i, r in enumerate(requests(32))] for cohort in (0, 3)}


def unit(sign=1):
    value = np.arange(16, dtype=np.float32) - 7.5
    return sign * value / np.linalg.norm(value)


def model_with_fixed_eos(lengths=None):
    model = tiny_backend(dtype=torch.bfloat16)
    config = deepcopy(model.model.config)
    config.num_hidden_layers = 17
    torch.manual_seed(777)
    model.model = Qwen3ForCausalLM(config).bfloat16().eval()
    model.layers = model.model.model.layers
    model.model.generation_config.eos_token_id = 127
    original = model.model.forward
    lengths = lengths or [4, 1, 2, 3] * 8

    def controlled(*args, **kwargs):
        output = original(*args, **kwargs)
        step = kwargs["attention_mask"].shape[1] - 128
        logits = torch.full_like(output.logits, -1000)
        for row, length in enumerate(lengths):
            logits[row, -1, 127 if step + 1 == length else 8 + row % 20] = 1000
        output.logits = logits
        return output

    model.model.forward = controlled
    return model


def clean(model):
    assert model.active_hooks == 0 and engine.hook_inventory(model) == {}


def test_fixed_inventory_512_slots_including_64_faults_and_exact_companions():
    requests_by_cohort = source_requests()
    calls = engine.call_plan(requests_by_cohort)
    assert len(calls) == 16
    assert len({trial for call in calls for trial in call["trial_ids"]}) == 512
    assert [c["call_index"] for c in calls if c["expected_error"]] == [6, 8]
    assert [c["cap"] for c in calls] == [128] * 10 + [4096] * 6
    assert [c["source_cohort"] for c in calls] == [0] * 13 + [3] * 3
    assert calls[3]["one_row_only"] == 0
    for call in calls:
        source = requests_by_cohort[call["source_cohort"]]
        assert call["requests_sha256"] == digest(source)
        assert call["base_trial_ids"] == [r["source_trial_id"] for r in source]
    with pytest.raises(ValueError, match="only original"):
        engine.call_plan({0: requests_by_cohort[0]})


@pytest.fixture(scope="module")
def actual_calls():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    model, source = model_with_fixed_eos(), source_requests()
    results, arrays = {}, {}
    for call in engine.call_plan(source):
        results[call["call_index"]], arrays[call["call_index"]] = engine.collect_call(
            model, source[call["source_cohort"]], call, unit())
        clean(model)
    torch.set_num_threads(old)
    return source, results, arrays


def test_all_actual_bf16_calls_sham_isolation_faults_cleanup_and_4096_caps(actual_calls):
    source, results, _ = actual_calls
    assert len(results) == 16
    assert all(r["passed"] for r in results.values()), {i: r["checks"] for i, r in results.items() if not r["passed"]}
    assert all(engine.relationship_checks(results).values())
    for index in (6, 8):
        report = results[index]
        assert report["rows"] is None and report["fault_injected"]
        assert report["error"] == dict(type="ControlledCoverageFault", message=report["call"]["inject_failure_at"])
        assert report["completed_forwards"] == (1 if index == 6 else 2)
        assert report["incomplete_forward"] is None
    for index in range(10, 16):
        report = results[index]
        assert report["call"]["cap"] == 4096
        assert report["rows"][0]["completion"]["sampling"]["max_new_tokens"] == 4096
        assert report["missing_decode_sample_ordinals"] == [32, 128]
        assert engine.validate_timeline(report, source[report["call"]["source_cohort"]])


def test_saved_complete_prefills_and_actual_decode_pairs_reconstruct_independently(actual_calls):
    _, results, arrays = actual_calls
    direction = torch.tensor(unit())
    for index, report in results.items():
        assert arrays[index]["hidden"].shape == (32, 1, 2, 16)
        for sample in report["samples"]:
            assert sample["forward_completed"]
            a = torch.tensor(arrays[index][sample["arrays"]["before"]])
            b = torch.tensor(arrays[index][sample["arrays"]["after"]])
            selected = sample["selected_token_indices"]
            expected = a.clone()
            expected[selected] = (expected[selected] + sample["dose"] * direction).bfloat16().float()
            assert torch.equal(expected.view(torch.int32), b.view(torch.int32))
            actual_delta = b[selected].double().numpy() - a[selected].double().numpy()
            geometry = report["forwards"][sample["forward_index"]]["geometry"][0]
            np.testing.assert_allclose(geometry["delta_norm_float64"], np.linalg.norm(actual_delta, axis=-1), rtol=1e-14)
            np.testing.assert_allclose(geometry["delta_dot_signed_unit_float64"], actual_delta @ unit().astype(np.float64), rtol=1e-14)
            for side, key in sample["arrays"].items():
                assert sample["array_bindings"][side] == engine.array_binding(arrays[index][key])
            if sample["phase"] == "prefill":
                assert a.shape == (128, 16)
                assert len(selected) == (0 if report["call"]["position"] is None else
                                         1 if report["call"]["position"] == "record" else 4)
        for frame in report["forwards"]:
            for g in frame["geometry"]:
                assert g["before_nonselected"] == g["after_nonselected"]
                if g["dose"]:
                    assert all(x > 0 for x in g["delta_dot_signed_unit_float64"])


@pytest.mark.parametrize("sign", [-1, 1])
def test_both_signed_units_keep_exact_full_prefill_arithmetic(sign):
    model, source = model_with_fixed_eos(), source_requests()
    call = engine.call_plan(source)[5]
    result, arrays = engine.collect_call(model, source[0], call, unit(sign))
    assert result["passed"]
    before, after = arrays["sample_prefill_before"], arrays["sample_prefill_after"]
    selected = result["samples"][0]["selected_token_indices"]
    expected = torch.tensor(before[selected]) + engine.DOSE * torch.tensor(unit(sign))
    np.testing.assert_array_equal(after[selected], expected.bfloat16().float().numpy())
    clean(model)


def test_false_backend_observation_and_bad_timeline_cannot_pass(actual_calls):
    source, results, _ = actual_calls
    result = deepcopy(results[5])
    result["rows"][0]["intervention_trace"]["independent_prefill_observations_passed"] = False
    assert engine.call_checks(result, source[0])["backend_full_prefill_observations"] is False
    result = deepcopy(results[5])
    result["forwards"][1]["position_ids"][1][0] += 1
    assert engine.call_checks(result, source[0])["timeline"] is False
    result = deepcopy(results[5])
    result["rows"][0]["intervention_trace"]["events"][0]["processed_token_id"] += 1
    assert engine.call_checks(result, source[0])["timeline"] is False


def test_unrelated_failure_in_fault_call_is_not_controlled_success(monkeypatch):
    model, source = model_with_fixed_eos(), source_requests()
    def fail(*args, **kwargs):
        raise RuntimeError("unrelated failure")
    monkeypatch.setattr(model.layers[16], "forward", fail)
    result, _ = engine.collect_call(model, source[0], engine.call_plan(source)[6], unit())
    assert result["error"]["type"] == "RuntimeError" and result["fault_injected"] is False
    assert result["passed"] is False and result["completed_forwards"] == 0
    assert result["incomplete_forward"] is not None
    clean(model)


def test_eos_first_cannot_replace_fixed_row0_short_decode_but_timing_is_descriptive():
    model, source = model_with_fixed_eos([1] + [4] * 31), source_requests()
    calls = engine.call_plan(source)
    short, _ = engine.collect_call(model, source[0], calls[5], unit())
    timing, _ = engine.collect_call(model, source[3], calls[15], unit())
    assert short["checks"]["row0_nonzero_active_decode"] is False and short["passed"] is False
    assert timing["passed"] is True
    assert timing["missing_decode_sample_ordinals"] == [1, 2, 32, 128]


def material_fixture():
    source = source_requests()
    calls = engine.call_plan(source)
    assignments = {str(k): [dict(trial_id=f"inner-{k}-{i}", kind="choice" if i % 2 == 0 else "new_fact",
                                sampling_seed=r["seed"]) for i, r in enumerate(v)] for k, v in source.items()}
    return dict(dose=engine.DOSE, direction_bindings={"curiosity_negative": {"unit": unit().tolist()}},
        engineering=dict(requests_by_cohort={str(k): v for k, v in source.items()},
            source_assignments_by_cohort=assignments, call_plan=calls, call_plan_sha256=digest(calls),
            potential_receipt_workload=dict(input_only=True, measured_worst_case=False)))


def install_saved_calls(monkeypatch, actual_calls):
    _, results, arrays = actual_calls
    monkeypatch.setattr(engine, "collect_call", lambda model, requests, call, direction, **kwargs:
                        (deepcopy(results[call["call_index"]]), deepcopy(arrays[call["call_index"]])))
    def receipt(model, assignments, requests, call, generation, **kwargs):
        records = [dict(generation_row=i, status="stop", receipt_required=False, receipt_model_pass_completed=False)
                   for i, a in enumerate(assignments) if a["kind"] == "choice"]
        return dict(records=records, assigned_choices=len(records), parsing_seconds=.01,
                    receipt_forward_seconds=0, actual_receipt_rows=0, actual_padded_width=None, passed=True), {}
    monkeypatch.setattr(engine, "collect_receipts", receipt)


def test_fixed_run_persists_all512_slots_and_all_receipt_denominators(tmp_path, monkeypatch, actual_calls):
    install_saved_calls(monkeypatch, actual_calls)
    materials = material_fixture()
    summary = engine.execute(tmp_path, object(), materials)
    assert summary["complete"] and summary["engineering_passed"]
    assert len(summary["slots"]) == 512 and len(summary["call_dispositions"]) == 16
    assert sum(s["status"] == "expected_fault" for s in summary["slots"]) == 64
    assert summary["assigned_receipt_slots"] == summary["observed_receipt_slots"] == 96
    assert len(summary["receipt_slots"]) == 96 and len(summary["timing_calls"]) == 6
    assert len(list((tmp_path / "calls").glob("*.json"))) == 16
    assert len(list((tmp_path / "receipts").glob("*.json"))) == 6
    assert json.loads((tmp_path / "engineering-summary.json").read_text()) == summary
    with pytest.raises(ValueError, match="already attempted"):
        engine.execute(tmp_path, object(), materials)


def test_first_gate_failure_stops_without_rescue_and_retains_unattempted_slots(tmp_path, monkeypatch, actual_calls):
    install_saved_calls(monkeypatch, actual_calls)
    collect = engine.collect_call
    def fail(*args, **kwargs):
        report, arrays = collect(*args, **kwargs)
        report["passed"] = False
        report["checks"]["all_position_geometry"] = False
        return report, arrays
    monkeypatch.setattr(engine, "collect_call", fail)
    summary = engine.execute(tmp_path, object(), material_fixture())
    assert not summary["engineering_passed"] and not summary["complete"]
    assert len(summary["slots"]) == 512
    assert sum(s["status"] == "missing" for s in summary["slots"]) == 480
    assert all(s["status"] == "missing" for s in summary["receipt_slots"])
    assert len(list((tmp_path / "attempts").glob("*.json"))) == 1


def test_persistence_exception_is_attempted_error_not_never_attempted(tmp_path, monkeypatch, actual_calls):
    install_saved_calls(monkeypatch, actual_calls)
    monkeypatch.setattr(engine, "_persist", lambda *a, **k: (_ for _ in ()).throw(OSError("disk failure")))
    summary = engine.execute(tmp_path, object(), material_fixture())
    assert [s["status"] for s in summary["slots"][:32]] == ["execution_error"] * 32
    assert [s["status"] for s in summary["slots"][32:]] == ["missing"] * 480
    assert summary["error"]["type"] == "OSError"
    assert (tmp_path / "failures/00.json").exists()


def test_preexpired_deadline_retains_all512_and_never_attempts(tmp_path):
    summary = engine.execute(tmp_path, object(), material_fixture(), deadline=0)
    assert summary["assigned_slots"] == 512 and all(s["status"] == "missing" for s in summary["slots"])
    assert not (tmp_path / "attempts").exists()
    assert not summary["engineering_passed"]


def receipt_inputs():
    assignments = [a for a in engine.voluntary.assignments() if a["kind"] == "choice"][:32]
    source = source_requests()
    source[0] = [{**r, "seed": a["sampling_seed"]} for r, a in zip(source[0], assignments)]
    call = engine.call_plan(source)[10]
    rows = []
    for i, a in enumerate(assignments):
        service = "stop" if i % 4 == 1 else "report"
        label = next(k for k, v in engine.voluntary.task(a)["action_mapping"].items() if v == service)
        rows.append(dict(completion=dict(raw="Unclosed reasoning" if i % 4 == 2 else f"Reasoning </think> {label}",
            generated_token_ids=[2] * 4096 if i % 4 == 3 else [2, 9999], ended_with_eos=i % 4 != 3)))
    return assignments, source[0], call, dict(rows=rows)


@pytest.mark.parametrize("corrupt", [False, True])
def test_actual_selected_receipt_helper_preserves_all_dispositions_and_failed_evidence(monkeypatch, corrupt):
    model, calls = fake_model(), []
    monkeypatch.setattr(engine.receipts, "_logit_hashes", fake_forward(calls, corrupt=corrupt))
    assignments, requests_, call, generation = receipt_inputs()
    result, arrays = engine.collect_receipts(model, assignments, requests_, call, generation)
    assert len(result["records"]) == result["assigned_choices"] == 32
    assert result["actual_receipt_rows"] == 8 and len(calls) == result["actual_forward_calls"] == 2
    assert calls[0]["layers"] == [] and calls[1]["layers"] == [15]
    assert arrays["hidden"].shape == (8, 1, 2, 4)
    assert result["passed"] is (not corrupt)
    for index, row in enumerate(result["records"]):
        expected = ("execution_error" if corrupt else "completed", "stop", "invalid_choice", "capped_choice")[index % 4]
        assert row["status"] == expected
        assert row["receipt_model_pass_completed"] is (index % 4 == 0 and not corrupt)
        if row["executed_simulator_outcome"] is not None:
            assert row["executed_simulator_outcome"]["receipt_model_pass_completed"] is (index % 4 == 0 and not corrupt)
    assert result["worst_case_workload_measured"] is False
    assert all(source.get("simulator_outcome") is None or source["simulator_outcome"]["receipt_model_pass_completed"] is False
               for source in result["source_records"])


def test_receipt_render_failure_retains_all_slots_without_forwarding(monkeypatch):
    model, calls = fake_model(), []
    monkeypatch.setattr(engine.receipts, "_logit_hashes", fake_forward(calls))
    monkeypatch.setattr(engine.receipts, "render_receipt", lambda *a, **k: (_ for _ in ()).throw(ValueError("render boundary")))
    assignments, requests_, call, generation = receipt_inputs()
    result, arrays = engine.collect_receipts(model, assignments, requests_, call, generation)
    assert not result["passed"] and len(result["records"]) == 32
    assert not calls and result["actual_forward_calls"] == 0 and not result["receipt_forward_attempted"]
    assert arrays == {} and result["error"]["type"] == "ValueError"
    assert sum(r["status"] == "execution_error" for r in result["records"]) == 8
