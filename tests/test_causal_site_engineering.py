"""Real CPU BF16 forward tests at the actual new block indices, no cloud data."""

from copy import deepcopy

import numpy as np
import pytest
import torch
from transformers import Qwen3ForCausalLM

from curiosity_study import causal_site_engineering as engineering
from curiosity_study import sustained_engineering as original
from test_state_backend import tiny_backend
from test_sustained_backend import requests


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def model_and_direction():
    model = tiny_backend(dtype=torch.bfloat16)
    config = deepcopy(model.model.config)
    config.num_hidden_layers = 49
    torch.manual_seed(223)
    model.model = Qwen3ForCausalLM(config).to(dtype=torch.bfloat16).eval()
    model.layers = model.model.model.layers
    model.model.generation_config.eos_token_id = None
    direction = np.arange(16, dtype=np.float32) - 7.5
    direction /= np.linalg.norm(direction)
    return model, direction


def inputs(count=32):
    return [{**row, "source_trial_id": f"synthetic-{i:03d}"} for i, row in enumerate(requests(count))]


@pytest.fixture(scope="module", params=[31, 47])
def evidence(request):
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model, unit = model_and_direction()
        report, arrays = engineering.run_engineering(
            model, inputs(), layer=request.param, direction=unit, dose=0.25, cap=3
        )
        return request.param, report, arrays
    finally:
        torch.set_num_threads(old)


def test_actual_new_sites_have_bound_before_after_and_full_row_identities(evidence):
    layer, report, arrays = evidence
    assert report["passed"], report["checks"]
    assert report["layer"] == layer and report["cap"] == 3
    assert not report["scientific_admission"] and not report["full_cap_certified"]
    assert arrays["before"].shape == arrays["after"].shape == (3, 16)
    assert not np.array_equal(arrays["before"], arrays["after"])
    assert report["checks"]["full_sham_identity"] and report["checks"]["unaffected31_identity"]
    for condition in report["conditions"].values():
        for row in condition["rows"]:
            assert all(e["layer"] == layer for e in row["intervention_trace"]["events"])
        assert all(len(f["full_fp32_logits_sha256"]) == 32 for f in condition["forwards"])
    assert original.LAYER == 15


def test_wrong_layer_or_changed_vector_and_timeline_cannot_pass(evidence):
    layer, report, arrays = evidence
    patches = [
        dict(layer=layer, position="record", mode="add", direction=arrays["direction_float32"], dose=0)
    ] * 32
    condition = deepcopy(report["conditions"]["zero_sham"])
    condition["rows"][17]["intervention_trace"]["events"][1]["layer"] = 15
    assert not engineering.validate_condition(
        condition, inputs(), patches, layer=layer, cap=3, pad_token_id=0, eos_ids=set()
    )
    condition = deepcopy(report["conditions"]["zero_sham"])
    condition["forwards"][1]["full_fp32_logits_sha256"][31] = "0" * 64
    assert not engineering.identical_rows(report["conditions"]["baseline"], condition, range(32))
    corrupted = {k: v.copy() for k, v in arrays.items()}
    corrupted["after"][1, 0] += 1
    assert not engineering.check_geometry(
        report["conditions"]["one_row_nonzero"], corrupted, arrays["direction_float32"], 0.25, torch
    )


@pytest.mark.parametrize(
    "problem", ["layer15", "bool_layer", "long_cap", "zero_dose", "bool_dose", "row_count", "bad_hash"]
)
def test_invalid_scope_is_rejected_before_forward(monkeypatch, problem):
    model, unit = model_and_direction()
    rows = inputs()
    kwargs = dict(layer=31, direction=unit, dose=0.25, cap=3)
    if problem == "layer15":
        kwargs["layer"] = 15
    elif problem == "bool_layer":
        kwargs["layer"] = True
    elif problem == "long_cap":
        kwargs["cap"] = 4096
    elif problem == "zero_dose":
        kwargs["dose"] = 0
    elif problem == "bool_dose":
        kwargs["dose"] = True
    elif problem == "row_count":
        rows.pop()
    else:
        rows[0]["token_prefix_sha256"] = "bad"
    monkeypatch.setattr(model.model, "forward", lambda *a, **k: pytest.fail("Invalid request reached model"))
    with pytest.raises(ValueError):
        engineering.run_engineering(model, rows, **kwargs)
    assert model.active_hooks == 0 and not engineering.hook_inventory(model)


def test_cap1_retains_evidence_without_claiming_sustained_check():
    model, unit = model_and_direction()
    report, arrays = engineering.run_engineering(model, inputs(), layer=47, direction=unit, dose=0.25, cap=1)
    assert not report["passed"] and not report["checks"]["nonzero_decode_engagement"]
    assert arrays["before"].shape == (1, 16)


def test_rounding_away_a_nonzero_dose_cannot_pass_engagement():
    model, unit = model_and_direction()
    report, arrays = engineering.run_engineering(model, inputs(), layer=31, direction=unit, dose=1e-30, cap=3)
    assert not report["passed"]
    assert not report["checks"]["every_perturbed_event_engaged"]
    assert np.array_equal(arrays["before"], arrays["after"])


def test_row0_immediate_eos_cannot_borrow_other_rows_decode_engagement(monkeypatch):
    model, unit = model_and_direction()
    model.model.generation_config.eos_token_id = 127
    original_forward = model.model.forward

    def forced(*args, **kwargs):
        output = original_forward(*args, **kwargs)
        logits = torch.full_like(output.logits, -1000)
        logits[:, -1, 7] = 1000
        logits[0, -1, 7] = -1000
        logits[0, -1, 127] = 1000
        output.logits = logits
        return output

    monkeypatch.setattr(model.model, "forward", forced)
    report, arrays = engineering.run_engineering(model, inputs(), layer=47, direction=unit, dose=0.25, cap=3)
    assert report["checks"]["decode_observed_in_every_condition"]
    assert not report["checks"]["nonzero_decode_engagement"] and not report["passed"]
    assert arrays["before"].shape == (1, 16)
    assert report["conditions"]["one_row_nonzero"]["forwards"][1]["full_fp32_logits_sha256"][0] is None


def test_partial_decode_error_retains_frames_and_cleans_all_hooks(monkeypatch):
    model, unit = model_and_direction()
    original_forward = model.layers[48].forward

    def broken(*args, **kwargs):
        if args[0].shape[1] == 1:
            raise RuntimeError("new-site decode failure")
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(model.layers[48], "forward", broken)
    patches = [dict(layer=47, position="record", mode="add", direction=unit, dose=0.25)] * 32
    condition, _ = engineering.collect_condition(
        model, inputs(), patches, layer=47, cap=3, expected_error=True
    )
    assert condition["rows"] is None and condition["error"]["type"] == "RuntimeError"
    assert len(condition["forwards"]) == 1 and condition["attempted_forwards"] == 2
    assert model.active_hooks == 0 and not engineering.hook_inventory(model)
