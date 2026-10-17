"""Explicit site capture, dose arithmetic and failure evidence; synthetic inputs only."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from curiosity_study import satisfaction_site_capture as capture
from curiosity_study import causal_site_engineering as engineering
from curiosity_study.artifacts import digest
from curiosity_study.state_backend import StateBackend
from test_causal_site_engineering import inputs, model_and_direction


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("layer", [31, 47])
def test_real_allrow_prefill_geometry_sham_and_receipt_layers(layer, tmp_path):
    model, unit = model_and_direction()
    rows = inputs()
    baseline, before = capture.collect_generation(model, rows, [None] * 32, layer=layer, cap=3)
    patches = [dict(layer=layer, mode="add", position="record", direction=unit, dose=0.25)] * 32
    condition, arrays = capture.collect_generation(model, rows, patches, layer=layer, cap=3)
    assert np.array_equal(before["hidden"], arrays["hidden"])
    assert capture.validate_prefill_geometry(condition, arrays, patches, torch=torch)
    assert condition["layer"] == layer and arrays["prefill_after"].shape == (32, 1, 2, 16)
    assert np.array_equal(arrays["hidden"][:, :, 1], arrays["prefill_after"][:, :, 1])
    changed = deepcopy(condition)
    changed["rows"][23]["intervention_trace"]["events"][0]["delta_norm"] += 1
    assert not capture.validate_prefill_geometry(changed, arrays, patches, torch=torch)
    bad = {k: v.copy() for k, v in arrays.items()}
    bad["prefill_after"][31, 0, 0, 0] += 1
    assert not capture.validate_prefill_geometry(condition, bad, patches, torch=torch)
    zeros = [{**p, "dose": 0} for p in patches]
    sham, sham_arrays = capture.collect_generation(model, rows, zeros, layer=layer, cap=3)
    assert engineering.identical_rows(baseline, sham, range(32))
    assert np.array_equal(sham_arrays["hidden"], sham_arrays["prefill_after"])
    saved = capture.save_arrays(
        tmp_path,
        "captures/test.npz",
        arrays,
        layer=layer,
        source_ids=[r["source_trial_id"] for r in rows],
        requests=rows,
    )
    with np.load(tmp_path / saved["path"], allow_pickle=False) as artifact:
        assert artifact["layers"].tolist() == [layer]
        assert np.array_equal(artifact["prefill_after"], arrays["prefill_after"])
    with pytest.raises(FileExistsError):
        capture.save_arrays(
            tmp_path,
            "captures/test.npz",
            arrays,
            layer=layer,
            source_ids=[r["source_trial_id"] for r in rows],
            requests=rows,
        )
    measured, hidden, guard = capture.forward_receipts(
        model, rows[:5], [r["source_trial_id"] for r in rows[:5]], layer=layer
    )
    assert guard["passed"] and guard["layers"] == [layer] and hidden.shape == (5, 1, 2, 16)
    assert all("completion" not in r and r["intervention"] is None for r in measured)


def historical_binding(rows, hidden):
    return dict(
        source_manifest_sha256="a" * 64,
        source_audit_sha256="b" * 64,
        record_bindings_sha256="c" * 64,
        artifact_bindings_sha256="d" * 64,
        requests_sha256=digest(rows),
        hidden=engineering.array_binding(hidden),
    )


def test_all288_reference_exact_history_and_independent_median_formula():
    model, _ = model_and_direction()
    rows = inputs(288)
    previous = []
    for offset in range(0, 288, 32):
        _, hidden = StateBackend(model).forward(rows[offset : offset + 32], layers=[15])
        previous.append(hidden)
    historical = np.concatenate(previous)
    report, arrays = capture.capture_reference(
        model,
        rows,
        historical_layer15_hidden=historical,
        historical_binding=historical_binding(rows, historical),
        original_dose=0.25,
    )
    assert report["passed"] and report["assigned"] == 288 and report["max_new_tokens"] == 0
    assert len(report["cohorts"]) == 9 and report["source_trial_ids"] == [r["source_trial_id"] for r in rows]
    assert arrays["hidden"].shape == (288, 3, 2, 16)
    medians = np.median(np.sqrt(np.sum(arrays["hidden"][:, :, 0].astype(np.float64) ** 2, axis=-1)), axis=0)
    assert report["medians"] == {str(layer): float(v) for layer, v in zip((15, 31, 47), medians)}
    assert report["doses"] == {
        str(layer): 0.25 * float(medians[i]) / float(medians[0]) for i, layer in ((1, 31), (2, 47))
    }
    assert report["dose_reference"]["capture_sha256"] == digest(report["capture_inventory"])
    assert report["dose_reference"]["doses"] == report["doses"]
    assert all("completion" not in r for cohort in report["cohorts"] for r in cohort["measurements"])


def test_rebound_false_history_retains_failed_identity_without_changing_dose(monkeypatch):
    model, _ = model_and_direction()
    rows = inputs(288)
    hidden = np.ones((288, 3, 2, 16), dtype=np.float32)
    historical = hidden[:, :1].copy()
    historical[255, 0, 1, 0] = 2

    def fake_forward(model, requests, ids, layers):
        assert layers == [15, 31, 47] and len(requests) == 32
        return (
            [dict(token_prefix_sha256=r["token_prefix_sha256"]) for r in requests],
            hidden[:32].copy(),
            {"passed": True},
        )

    monkeypatch.setattr(capture.receipts, "forward_receipts", fake_forward)
    report, _ = capture.capture_reference(
        model,
        rows,
        historical_layer15_hidden=historical,
        historical_binding=historical_binding(rows, historical),
        original_dose=0.25,
    )
    assert not report["passed"] and not report["checks"]["all_layer15_vectors_match"]
    assert not report["cohorts"][7]["historical_layer15_equal"]
    assert report["doses"] == {"31": 0.25, "47": 0.25}


@pytest.mark.parametrize("problem", ["hash", "order", "missing", "dose"])
def test_bad_reference_bindings_reject_before_any_call(monkeypatch, problem):
    model, _ = model_and_direction()
    rows = inputs(288)
    historical = np.ones((288, 1, 2, 16), dtype=np.float32)
    binding = historical_binding(rows, historical)
    dose = 0.25
    if problem == "hash":
        binding["hidden"]["byte_sha256"] = "bad"
    elif problem == "order":
        rows[0], rows[1] = rows[1], rows[0]
    elif problem == "missing":
        rows.pop()
    else:
        dose = True
    monkeypatch.setattr(
        model.model, "forward", lambda *a, **k: pytest.fail("Invalid calibration reached model")
    )
    with pytest.raises(ValueError):
        capture.capture_reference(
            model, rows, historical_layer15_hidden=historical, historical_binding=binding, original_dose=dose
        )


def test_partial_scientific_error_retains_prefill_vectors_without_success(monkeypatch):
    model, unit = model_and_direction()
    rows = inputs()
    forward = model.layers[32].forward

    def fail(*args, **kwargs):
        if args[0].shape[1] == 1:
            raise RuntimeError("partial scientific generation")
        return forward(*args, **kwargs)

    monkeypatch.setattr(model.layers[32], "forward", fail)
    patches = [dict(layer=31, position="record", mode="add", direction=unit, dose=0.25)] * 32
    condition, arrays = capture.collect_generation(model, rows, patches, layer=31, cap=3)
    assert condition["rows"] is None and condition["error"]["type"] == "RuntimeError"
    assert set(arrays) == {"hidden", "prefill_after"} and len(condition["forwards"]) == 1
    assert not capture.validate_prefill_geometry(condition, arrays, patches, torch=torch)
    assert not engineering.hook_inventory(model) and model.active_hooks == 0
