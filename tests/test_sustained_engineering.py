"""Synthetic CPU engineering evidence, without corpus or behavioral outcomes."""

from copy import deepcopy
import json

import numpy as np
import pytest
import torch
from transformers import Qwen3ForCausalLM

from curiosity_study import sustained_engineering as engineering
from curiosity_study.artifacts import digest
from test_state_backend import tiny_backend
from test_sustained_backend import requests


def inputs():
    return [{**row, "source_trial_id": f"synthetic-source-{index}"} for index, row in enumerate(requests(32))]


def model_and_direction():
    model = tiny_backend(dtype=torch.bfloat16)
    config = deepcopy(model.model.config)
    config.num_hidden_layers = 17
    torch.manual_seed(223)
    model.model = Qwen3ForCausalLM(config).to(dtype=torch.bfloat16).eval()
    model.layers = model.model.model.layers
    model.model.generation_config.eos_token_id = None
    direction = np.arange(16, dtype=np.float32) - 7.5
    direction /= np.linalg.norm(direction)
    return model, direction


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def evidence():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model, direction = model_and_direction()
        report, arrays = engineering.run_engineering(model, inputs(), direction=direction, dose=0.25, cap=4)
        return report, arrays
    finally:
        torch.set_num_threads(previous)


def test_real_bf1632row_harness_binds_predictions_vectors_and_retains_scope(evidence):
    report, arrays = evidence
    assert report["passed"], report["checks"]
    assert report["batch_size"] == 32 and report["layer"] == 15 and not report["scientific_admission"]
    assert report["request_sha256"] == digest(inputs())
    assert report["observed_baseline_max_tokens"] == report["observed_nonzero_row_tokens"] == 4
    assert arrays["before"].shape == arrays["after"].shape == (4, 16)
    assert arrays["direction_float32"].shape == (16,)
    assert {key: engineering.array_binding(value) for key, value in arrays.items()} == report["arrays"]
    assert report["arrays_sha256"] == digest(report["arrays"])
    assert json.loads(json.dumps(report)) == report
    assert report["malformed_direction"]["error"]["type"] == "ValueError"
    for condition in report["conditions"].values():
        assert len(condition["rows"]) == 32 and len(condition["forwards"]) == 4
        assert all(len(frame["full_fp32_logits_sha256"]) == 32 for frame in condition["forwards"])
        assert condition["forwards"][0]["cache_length_before"] == 0
        assert condition["hooks_clean"]
    assert not np.array_equal(arrays["before"], arrays["after"])


def test_same_tokens_cannot_hide_one_changed_full_vocabulary_prediction(evidence):
    report, _ = evidence
    baseline = report["conditions"]["baseline"]
    corrupted = deepcopy(report["conditions"]["zero_sham"])
    assert engineering.identical_rows(baseline, corrupted, range(32))
    corrupted["forwards"][2]["full_fp32_logits_sha256"][17] = "0" * 64
    assert baseline["rows"][17]["completion"] == corrupted["rows"][17]["completion"]
    assert not engineering.identical_rows(baseline, corrupted, range(32))


@pytest.mark.parametrize("field", ["cache", "mask", "position", "event", "seed"])
def test_independent_timeline_rejects_rebound_metadata(evidence, field):
    report, arrays = evidence
    condition = deepcopy(report["conditions"]["zero_sham"])
    if field == "cache":
        condition["forwards"][1]["cache_length_before"] += 1
    elif field == "mask":
        condition["forwards"][1]["attention_mask_row_sha256"][0] = "a" * 64
    elif field == "position":
        condition["forwards"][2]["position_ids"][4][0] += 1
    elif field == "event":
        condition["rows"][3]["intervention_trace"]["events"][2]["processed_generated_token_index"] = 4
    else:
        condition["rows"][3]["completion"]["sampling"]["seed"] += 1
    patches = [
        dict(layer=15, mode="add", position="record", direction=arrays["direction_float32"], dose=0.0)
    ] * 32
    assert not engineering.validate_condition(
        condition, inputs(), patches, cap=4, pad_token_id=0, eos_ids=set()
    )


def test_vector_geometry_is_independently_checked_not_trusted_from_the_ledger(evidence):
    report, arrays = evidence
    condition = report["conditions"]["one_row_nonzero"]
    bad = {key: value.copy() for key, value in arrays.items()}
    bad["after"][2, 0] += 1
    assert not engineering.check_geometry(condition, bad, arrays["direction_float32"], 0.25, torch)
    changed = deepcopy(condition)
    changed["rows"][0]["intervention_trace"]["events"][1]["delta_norm"] += 0.1
    assert not engineering.check_geometry(changed, arrays, arrays["direction_float32"], 0.25, torch)


@pytest.mark.parametrize("early_row0", [False, True])
def test_mixed_eos_keeps32_rows_and_requires_actual_nonzero_decode_exposure(monkeypatch, early_row0):
    model, direction = model_and_direction()
    model.model.generation_config.eos_token_id = 127
    original_forward = model.model.forward
    lengths = [1 + row % 4 for row in range(32)]
    lengths[0] = 1 if early_row0 else 2

    def controlled_forward(*args, **kwargs):
        cache = kwargs.get("past_key_values")
        step = 0 if cache is None else int(cache.get_seq_length()) - 128 + 1
        output = original_forward(*args, **kwargs)
        logits = torch.full_like(output.logits, -1000)
        for row in range(32):
            length = lengths[row]
            logits[row, -1, 127 if step + 1 == length and length < 4 else 7] = 1000
        output.logits = logits
        return output

    monkeypatch.setattr(model.model, "forward", controlled_forward)
    report, arrays = engineering.run_engineering(model, inputs(), direction=direction, dose=0.25, cap=4)
    assert report["passed"] is (not early_row0), report["checks"]
    assert report["checks"]["nonzero_decode_engagement"] is (not early_row0)
    assert report["observed_nonzero_row_tokens"] == lengths[0]
    assert arrays["before"].shape == (lengths[0], 16)
    for condition in report["conditions"].values():
        assert len(condition["forwards"]) == 4
        for step, frame in enumerate(condition["forwards"]):
            assert len(frame["active_prediction_rows"]) == 32
            for row in range(32):
                assert frame["active_prediction_rows"][row] == (step < lengths[row])
                assert (frame["full_fp32_logits_sha256"][row] is None) == (step >= lengths[row])
    assert not engineering.hook_inventory(model) and model.active_hooks == 0


def test_one_token_allowance_retains_evidence_but_cannot_pass_sustained_engineering():
    model, direction = model_and_direction()
    report, arrays = engineering.run_engineering(model, inputs(), direction=direction, dose=0.25, cap=1)
    assert not report["passed"] and arrays["before"].shape == (1, 16)
    assert not report["checks"]["decode_observed_in_every_condition"]
    assert not report["checks"]["nonzero_decode_engagement"]
    assert report["checks"]["independently_recomputed_geometry"]


def test_middecode_error_cleans_observers_and_backend_hooks(monkeypatch):
    model, direction = model_and_direction()
    original_forward = model.layers[16].forward

    def fail_after_patch(*args, **kwargs):
        if args[0].shape[1] == 1:
            raise RuntimeError("synthetic decode failure")
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(model.layers[16], "forward", fail_after_patch)
    with pytest.raises(RuntimeError, match="synthetic decode failure"):
        engineering.run_engineering(model, inputs(), direction=direction, dose=0.25, cap=4)
    assert not engineering.hook_inventory(model) and model.active_hooks == 0


@pytest.mark.parametrize(
    "problem", ["31rows", "duplicate_source", "bad_token_hash", "wrong_layer", "zero_dose", "oversized_cap"]
)
def test_admission_shape_errors_do_not_execute_model(monkeypatch, problem):
    model, direction = model_and_direction()
    rows, kwargs = inputs(), dict(direction=direction, dose=0.25, cap=4)
    if problem == "31rows":
        rows.pop()
    elif problem == "duplicate_source":
        rows[-1]["source_trial_id"] = rows[0]["source_trial_id"]
    elif problem == "bad_token_hash":
        rows[0]["token_prefix_sha256"] = "bad"
    elif problem == "wrong_layer":
        kwargs["layer"] = 14
    elif problem == "zero_dose":
        kwargs["dose"] = 0
    else:
        kwargs["cap"] = 4097
    monkeypatch.setattr(model.model, "forward", lambda *a, **k: pytest.fail("Invalid plan reached model"))
    with pytest.raises(ValueError):
        engineering.run_engineering(model, rows, **kwargs)
    assert not engineering.hook_inventory(model)
