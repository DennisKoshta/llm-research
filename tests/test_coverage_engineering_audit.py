"""Independent exact BF16 sample reconstruction; CPU only, no study outputs."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import audit_coverage_engineering as audit  # noqa: E402


def actual_pair(dose, selected):
    generator = torch.Generator().manual_seed(7103)
    before = torch.randn((7, 32), generator=generator).bfloat16()
    direction = torch.randn(32, generator=generator, dtype=torch.float32)
    direction /= direction.norm()
    after = before.clone()
    after[selected] = (before[selected].float() + dose * direction).bfloat16()
    return before.float().numpy(), after.float().numpy(), direction.numpy()


@pytest.mark.parametrize("dose", [0., .1, -.1, 29.78968343510005, -29.78968343510005])
@pytest.mark.parametrize("selected", [[], [5], [2, 3, 4, 5, 6]])
def test_actual_torch_bfloat16_arithmetic_and_padding(dose, selected):
    before, after, direction = actual_pair(dose, selected)
    result = audit.audit_saved_pair(before, after, selected, direction, dose)
    assert result["passed"]
    assert result["independently_reconstructed_components"] == len(selected) * 32
    assert result["selected_token_positions"] == selected


def test_round_to_even_matches_actual_bfloat16_ties_and_signed_zero():
    # Adjacent BF16 floats' midpoint selects the even mantissa for both signs.
    values = np.asarray([0., -0., 1.00390625, 1.01171875, -1.00390625, -1.01171875], dtype=np.float32)
    expected = torch.from_numpy(values).bfloat16().float().numpy()
    np.testing.assert_array_equal(audit.bfloat16_round(values).view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("position", [0, 4])
def test_actual_after_corruption_fails_selected_or_padding_check(position):
    before, after, direction = actual_pair(.1, [3, 4, 5, 6])
    after[position, 0] = torch.tensor(after[position, 0] + 1.).bfloat16().float().item()
    result = audit.audit_saved_pair(before, after, [3, 4, 5, 6], direction, .1)
    assert not result["passed"]
    assert not result["checks"]["nonselected_bitwise_unchanged" if position == 0 else "selected_exact_bfloat16"]


def test_opposite_orientation_cannot_pass_elementwise_or_sign_gate():
    before, after, direction = actual_pair(-.1, [4])
    result = audit.audit_saved_pair(before, after, [4], direction, .1)
    assert not result["checks"]["selected_exact_bfloat16"]
    assert not result["checks"]["achieved_nonzero_and_correct_sign"]


def test_nonzero_requested_but_rounded_away_is_not_engagement():
    before = np.full((1, 2), 1e10, dtype=np.float32)
    before = torch.from_numpy(before).bfloat16().float().numpy()
    result = audit.audit_saved_pair(before, before.copy(), [0], np.array([1., 0.], np.float32), .01)
    assert result["checks"]["selected_exact_bfloat16"]
    assert not result["checks"]["achieved_nonzero_and_correct_sign"]


@pytest.mark.parametrize("selected", [[3, 3], [4, 3], [-1], [7], [True]])
def test_invalid_selected_inventory_rejected(selected):
    before, after, direction = actual_pair(0, [])
    with pytest.raises(ValueError, match="selected-token inventory"):
        audit.audit_saved_pair(before, after, selected, direction, 0.)


def test_rounded_float32_not_exact_bfloat16_source_rejected():
    before, after, direction = actual_pair(.1, [4])
    before[0, 0] = np.nextafter(before[0, 0], np.float32(np.inf))
    with pytest.raises(ValueError, match="not exact BF16"):
        audit.audit_saved_pair(before, after, [4], direction, .1)


@pytest.fixture(scope="module")
def observed_calls():
    """Actual tiny CPU BF16 residuals; test-only forced EOS bounds fixture cost."""
    from copy import deepcopy
    from curiosity_study import coverage_engineering as producer
    from test_sustained_engineering import inputs, model_and_direction
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model, direction = model_and_direction()
        model.model.generation_config.eos_token_id = 127
        original = model.model.forward
        counter = [0]

        def bounded(*args, **kwargs):
            if kwargs.get("past_key_values") is None:
                counter[0] = 0
            else:
                counter[0] += 1
            output = original(*args, **kwargs)
            # Keep all other actual full-vocabulary logits. The fixture's third
            # prediction ends every row; this is not production sampling code.
            output.logits[..., 127] = 1000 if counter[0] == 2 else -1000
            return output

        model.model.forward = bounded
        requests = {0: inputs(), 3: deepcopy(inputs())}
        for i, request in enumerate(requests[3]):
            request["source_trial_id"] = f"other-cohort-{i}"
            request["token_prefix"] += [7, 8]
            request["token_prefix_sha256"] = audit.digest(request["token_prefix"])
            request["prefix_positions"]["final"] = len(request["token_prefix"]) - 1
        calls = producer.call_plan(requests)
        reports, arrays = {}, {}
        for call in calls:
            report, vectors = producer.collect_call(model, requests[call["source_cohort"]], call, direction)
            reports[call["call_index"]], arrays[call["call_index"]] = report, vectors
        return calls, requests, direction, reports, arrays
    finally:
        torch.set_num_threads(previous)


def check_call(observed, index, report=None, arrays=None):
    calls, requests, direction, reports, vectors = observed
    call = calls[index]
    return audit.audit_call(reports[index] if report is None else report, vectors[index] if arrays is None else arrays,
        requests[call["source_cohort"]], call, direction,
        hidden_size=16, vocabulary_size=128, pad_token_id=0, eos_ids=[127])


def test_all16_actual_tiny_calls_exact_samples_and_faults(observed_calls):
    results = [check_call(observed_calls, index) for index in range(16)]
    assert all(r["passed"] for r in results), [(r["call_index"], r["checks"]) for r in results if not r["passed"]]
    assert sum(r["ordinary_completed_slots"] for r in results) == 448
    assert sum(r["expected_fault_slots"] for r in results) == 64
    assert all(r["saved_vectors"]["missing_decode_sample_ordinals"] == [32, 128] for r in results if r["status"] == "completed")
    assert all(audit.audit_relationships(observed_calls[3]).values())


@pytest.mark.parametrize("change", ["frame_mask", "fingerprint", "cache", "selected", "unsaved_nonselected",
    "scalar_inventory", "wrong_sign", "legacy_full_trace", "trace_count", "actual_backend_observation"])
def test_independent_raw_checks_reject_self_reported_pass(observed_calls, change):
    from copy import deepcopy
    report = deepcopy(observed_calls[3][5])
    if change == "frame_mask":
        report["forwards"][1]["attention_mask_row_sha256"][1] = "f" * 64
    elif change == "fingerprint":
        report["forwards"][1]["full_fp32_logits_sha256"][1] = None
    elif change == "cache":
        report["forwards"][1]["cache_length_before"] += 1
    elif change == "selected":
        report["forwards"][0]["geometry"][1]["selected_token_indices"] = [0]
    elif change == "unsaved_nonselected":
        report["forwards"][0]["geometry"][1]["after_nonselected"]["byte_sha256"] = "f" * 64
    elif change == "scalar_inventory":
        report["forwards"][1]["geometry"][1]["delta_norm_float64"] = []
    elif change == "wrong_sign":
        report["forwards"][1]["geometry"][1]["delta_dot_signed_unit_float64"] = [-1.]
    elif change == "legacy_full_trace":
        report["rows"][1]["intervention_trace"]["schema_version"] = 1
    elif change == "trace_count":
        report["rows"][1]["intervention_trace"]["prefill_token_patch_count"] = 1
    else:
        report["rows"][1]["prefill_geometry_observations"][0]["passed"] = False
    assert report["passed"]  # Deliberately stale producer flag must not rescue it.
    assert not check_call(observed_calls, 5, report=report)["passed"]


@pytest.mark.parametrize("change", ["drop_sample", "invent_missing", "change_sample", "change_capture"])
def test_fixed_sample_binding_rejects_tamper(observed_calls, change):
    from copy import deepcopy
    report, arrays = deepcopy(observed_calls[3][5]), deepcopy(observed_calls[4][5])
    if change == "drop_sample":
        report["samples"].pop()
    elif change == "invent_missing":
        report["missing_decode_sample_ordinals"] = []
    elif change == "change_sample":
        arrays["sample_decode_1_after"][0, 0] += np.float32(1)
    else:
        arrays["hidden"][0, 0, 0, 0] += np.float32(1)
    with pytest.raises(ValueError):
        check_call(observed_calls, 5, report=report, arrays=arrays)


def test_unrelated_exception_in_expected_fault_call_is_not_controlled_success(observed_calls):
    from copy import deepcopy
    report = deepcopy(observed_calls[3][6])
    report["error"] = dict(type="RuntimeError", message="unrelated model failure")
    report["fault_injected"] = False
    result = check_call(observed_calls, 6, report=report)
    assert not result["passed"] and result["expected_fault_slots"] == 0
    assert result["status"] == "execution_error"


def test_identical_outputs_do_not_hide_changed_active_logits(observed_calls):
    from copy import deepcopy
    reports = deepcopy(observed_calls[3])
    reports[2]["forwards"][1]["full_fp32_logits_sha256"][20] = "f" * 64
    assert not audit.audit_relationships(reports)["full_zero_identity"]
    reports = deepcopy(observed_calls[3])
    reports[3]["forwards"][1]["full_fp32_logits_sha256"][20] = "f" * 64
    assert not audit.audit_relationships(reports)["row0_only_isolation"]


def whole(observed, reports=None, arrays=None, calls=None):
    plan, requests, direction, raw, vectors = observed
    return audit.audit_collection(plan if calls is None else calls, raw if reports is None else reports,
        vectors if arrays is None else arrays, requests, direction,
        hidden_size=16, vocabulary_size=128, pad_token_id=0, eos_ids=[127])


def test_complete512_slot_numeric_receipt_is_not_scientific_or_receipt_admission(observed_calls):
    result = whole(observed_calls)
    assert result["complete"] and result["numeric_engineering_passed"]
    assert result["assigned_slots"] == len(result["slots"]) == 512
    assert result["ordinary_completed_slots"] == 448 and result["observed_controlled_fault_slots"] == 64
    assert result["missing_slots"] == 0
    assert not any(result[key] for key in ("scientific_admission", "source_provenance_verified", "runtime_verified",
                                          "selected_receipts_verified", "admission_and_shutdown_verified"))


def test_missing_timing_calls_keep_all512_and_cannot_pass(observed_calls):
    reports = {k: v for k, v in observed_calls[3].items() if k < 10}
    arrays = {k: v for k, v in observed_calls[4].items() if k < 10}
    result = whole(observed_calls, reports, arrays)
    assert not result["complete"] and not result["numeric_engineering_passed"]
    assert len(result["slots"]) == 512 and result["missing_slots"] == 192
    assert result["observed_controlled_fault_slots"] == 64


def test_timing_may_not_follow_failed_short_check(observed_calls):
    from copy import deepcopy
    reports = deepcopy(observed_calls[3])
    reports[1]["passed"] = False
    result = whole(observed_calls, reports)
    assert not result["checks"]["one_fixed_order_no_after_failure"] and not result["numeric_engineering_passed"]


@pytest.mark.parametrize("change", ["drop", "reorder", "cap", "source", "schedule"])
def test_fixed_call_inventory_cannot_be_changed(observed_calls, change):
    from copy import deepcopy
    calls = deepcopy(observed_calls[0])
    if change == "drop":
        calls.pop()
    elif change == "reorder":
        calls[1], calls[2] = calls[2], calls[1]
    elif change == "cap":
        calls[10]["cap"] = 128
    elif change == "source":
        calls[13]["source_cohort"] = 0
    else:
        calls[5]["schedule"] = audit.RECORD
    with pytest.raises(ValueError, match="Fixed16 call inventory"):
        whole(observed_calls, calls=calls)


def test_all_missing_collection_retains512_without_certification(observed_calls):
    result = whole(observed_calls, {}, {})
    assert result["observed_calls"] == 0 and result["missing_slots"] == len(result["slots"]) == 512
    assert not result["numeric_engineering_passed"] and not result["complete"]


def test_mixed_eos_timeline_and_fixed_row0_no_decode_failure():
    from curiosity_study import coverage_engineering as producer
    from test_sustained_engineering import inputs, model_and_direction
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model, direction = model_and_direction()
        model.model.generation_config.eos_token_id = 127
        original, counter = model.model.forward, [0]
        lengths = [1, 2, 5, 3] * 8

        def mixed(*args, **kwargs):
            counter[0] = 0 if kwargs.get("past_key_values") is None else counter[0] + 1
            output = original(*args, **kwargs)
            for row, length in enumerate(lengths):
                output.logits[row, -1, 127] = 1000 if counter[0] + 1 >= length else -1000
            return output

        model.model.forward = mixed
        requests = inputs()
        calls = producer.call_plan({0: requests, 3: requests})
        for index in (2, 3):
            report, arrays = producer.collect_call(model, requests, calls[index], direction)
            result = audit.audit_call(report, arrays, requests, calls[index], direction,
                hidden_size=16, vocabulary_size=128, pad_token_id=0, eos_ids=[127])
            assert result["timeline"]["passed"] and result["online_geometry"]["passed"]
            assert result["saved_vectors"]["missing_decode_sample_ordinals"] == [1, 2, 32, 128]
            assert result["passed"] is (index == 2)
            if index == 3:
                assert not result["checks"]["row0_nonzero_active_decode"]
    finally:
        torch.set_num_threads(previous)


def test_failed_final_producer_gate_is_never_promoted(observed_calls):
    from copy import deepcopy
    reports = deepcopy(observed_calls[3])
    reports[15]["passed"] = False
    result = whole(observed_calls, reports)
    assert not result["numeric_engineering_passed"]
    assert not result["calls"][15]["checks"]["producer_flag_consistent"]
    assert not result["calls"][15]["checks"]["producer_gate_passed"]
