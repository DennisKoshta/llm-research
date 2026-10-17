"""Actual CPU BF16 tests for the explicit all-attended prefix schedule."""

from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import Qwen3ForCausalLM

from curiosity_study.state_backend import ALL_ATTENDED_SCHEDULE, StateBackend
from test_state_backend import tiny_backend
from test_sustained_backend import assert_calls_equal, observe_forward, patch, requests, without_time


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def full_patch(dose=.25, layer=1):
    return {**patch(dose, layer), "position": "all_attended"}


def forward(model, rows, patches=None, **kwargs):
    return StateBackend(model).forward(
        rows, interventions=patches, max_new_tokens=kwargs.pop("cap", 5),
        intervention_schedule=ALL_ATTENDED_SCHEDULE, **kwargs,
    )


def clean(model):
    assert model.active_hooks == 0
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.model.modules())


def bf16_binding(array):
    tensor = torch.tensor(array).bfloat16()
    return dict(dtype="torch.bfloat16", shape=list(array.shape),
                byte_sha256=hashlib.sha256(tensor.view(torch.uint16).numpy().tobytes()).hexdigest())


def test_actual_layer15_32rows_full_zero_sham_and_one_row_isolation():
    model = tiny_backend(dtype=torch.bfloat16)
    config = deepcopy(model.model.config)
    config.num_hidden_layers = 17  # actual layer16 input is required
    torch.manual_seed(190)
    model.model = Qwen3ForCausalLM(config).bfloat16().eval()
    model.layers = model.model.model.layers
    model.model.generation_config.eos_token_id = None
    rows = requests(32)
    baseline, a, calls_a = observe_forward(model, rows, schedule=ALL_ATTENDED_SCHEDULE, cap=5, capture_layers=(15,))
    sham, b, calls_b = observe_forward(model, rows, [full_patch(0, 15)] * 32,
                                     schedule=ALL_ATTENDED_SCHEDULE, cap=5, capture_layers=(15,))
    changed, c, calls_c = observe_forward(model, rows, [full_patch(.5, 15)] + [None] * 31,
                                        schedule=ALL_ATTENDED_SCHEDULE, cap=5, capture_layers=(15,))
    assert_calls_equal(calls_a, calls_b)
    assert_calls_equal(calls_a, calls_c, list(range(1, 32)))
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, c)  # all same-layer captures are prepatch
    assert [r["completion"]["generated_token_ids"] for r in baseline] == [
        r["completion"]["generated_token_ids"] for r in sham]
    assert [r["completion"]["generated_token_ids"] for r in baseline[1:]] == [
        r["completion"]["generated_token_ids"] for r in changed[1:]]
    assert all(event["delta_norm"] > 0 for event in changed[0]["intervention_trace"]["events"])
    for i, row in enumerate(sham):
        trace = row["intervention_trace"]
        assert trace["schema_version"] == 2
        assert trace["prefill_forward_patch_count"] == 1
        assert trace["prefill_token_patch_count"] == len(rows[i]["token_prefix"])
        assert trace["decode_patch_count"] == 4
        assert trace["total_patch_count"] == len(rows[i]["token_prefix"]) + 4
        assert all(event["delta_norm"] == 0 for event in trace["events"])
        observation = row["prefill_geometry_observations"][0]
        assert observation["passed"] and observation["before"] == observation["after"]
    assert all(r["intervention_trace"]["events"] == [] for r in changed[1:])
    clean(model)


def test_every_attended_token_exact_actual_after_vectors_padding_and_exposure():
    model = tiny_backend(dtype=torch.bfloat16)
    model.model.generation_config.eos_token_id = None
    rows, patches, saved = requests(3), [full_patch(.1), full_patch(-.2), None], {}
    before, after = [], []
    handles = [
        model.layers[1].register_forward_hook(lambda m, a, out: before.append(out[0].detach().float().clone())),
        model.layers[2].register_forward_pre_hook(lambda m, a: after.append(a[0].detach().float().clone())),
    ]
    try:
        measured, _ = forward(model, rows, patches, cap=4,
                              prefill_evidence_output=saved, prefill_evidence_rows=(0, 2))
    finally:
        for handle in handles:
            handle.remove()
    assert set(saved) == {f"layer_1_row_{i}_{side}" for i in (0, 2) for side in ("before", "after")}
    _, _, inputs, _ = StateBackend(model).prepare(rows)
    for row, result in enumerate(measured):
        trace = result["intervention_trace"]
        observed = result["prefill_geometry_observations"][0]
        mask = inputs["attention_mask"][row].bool()
        expected = before[0][row].clone()
        if patches[row]:
            expected[mask] = (expected[mask] + patches[row]["dose"] * torch.tensor(patches[row]["direction"])).bfloat16().float()
        assert torch.equal(after[0][row], expected)
        assert torch.equal(before[0][row, ~mask], after[0][row, ~mask])
        assert all(observed["checks"].values())
        assert observed["before"] == bf16_binding(before[0][row].numpy())
        assert observed["after"] == bf16_binding(after[0][row].numpy())
        if row in (0, 2):
            np.testing.assert_array_equal(saved[f"layer_1_row_{row}_before"], before[0][row])
            np.testing.assert_array_equal(saved[f"layer_1_row_{row}_after"], after[0][row])
        events = trace["events"]
        prefix = [event for event in events if event["phase"] == "prefill"]
        assert [event["unpadded_token_index"] for event in prefix] == (list(range(len(rows[row]["token_prefix"]))) if patches[row] else [])
        assert [event["processed_token_id"] for event in prefix] == (rows[row]["token_prefix"] if patches[row] else [])
        for event in events:
            step = 0 if event["phase"] == "prefill" else event["processed_generated_token_index"]
            pos = event["token_index"]
            a, b = before[step][row, pos], after[step][row, pos]
            assert event["delta_norm"] == pytest.approx(float((b - a).norm()), rel=1e-6)
            assert event["before_norm"] == pytest.approx(float(a.norm()), rel=1e-6)
            assert event["after_norm"] == pytest.approx(float(b.norm()), rel=1e-6)
            assert event["requested_scalar_delta"] == patches[row]["dose"]
        for phase in ("prefill", "decode"):
            deltas = [event["delta_norm"] for event in events if event["phase"] == phase]
            exposure = trace["exposure"][phase]
            assert exposure["token_patch_count"] == len(deltas)
            assert exposure["achieved_norm_sum"] == pytest.approx(sum(deltas))
            assert exposure["achieved_norm_root_sum_squares"] == pytest.approx(np.linalg.norm(deltas))
    clean(model)


def test_retention_does_not_change_intervention_or_generation():
    model = tiny_backend(dtype=torch.bfloat16)
    model.model.generation_config.eos_token_id = None
    rows, patches, saved = requests(2), [full_patch(), None], {}
    a, _ = forward(model, rows, patches)
    b, _ = forward(model, rows, patches, prefill_evidence_output=saved, prefill_evidence_rows=(0, 1))
    for left, right in zip(a, b):
        assert left["completion"] == right["completion"]
        assert left["intervention_trace"] == right["intervention_trace"]
        assert left["label_logits"] == right["label_logits"]
    clean(model)


@pytest.mark.parametrize("where", ["selected", "padding", "untreated"])
def test_actual_next_block_corruption_is_detected_not_replaced_by_expected(where):
    model = tiny_backend(dtype=torch.bfloat16)
    model.model.generation_config.eos_token_id = None

    def corrupt(module, args):
        if args[0].shape[1] > 1:
            changed = args[0].clone()
            row, position = (1, -1) if where == "untreated" else (0, 0 if where == "padding" else -1)
            changed[row, position, 0] += 1
            return (changed, *args[1:])

    handle = model.layers[2].register_forward_pre_hook(corrupt)
    try:
        measured, _ = forward(model, requests(2), [full_patch(), None], cap=2)
    finally:
        handle.remove()
    row = 1 if where == "untreated" else 0
    observation = measured[row]["prefill_geometry_observations"][0]
    assert observation["passed"] is False
    key = "selected_exact_bfloat16" if where == "selected" else "nonselected_bitwise_unchanged"
    assert observation["checks"][key] is False
    clean(model)


def test_mixed_eos_keeps_original_active_decode_and_rng_schedule(monkeypatch):
    model = tiny_backend(dtype=torch.bfloat16)
    model.model.generation_config.eos_token_id = 127
    rows, lengths, samples, masks = requests(4), [1, 2, 5, 3], [], []
    sample = torch.multinomial

    def draw(*args, **kwargs):
        samples.append(kwargs["generator"].initial_seed())
        return sample(*args, **kwargs)

    def force(module, args, kwargs, output):
        masks.append(kwargs["attention_mask"].clone())
        logits = torch.full_like(output.logits, -1000)
        for row, length in enumerate(lengths):
            logits[row, -1, 127 if len(masks) == length and row != 2 else 6 + row] = 1000
        output.logits = logits

    monkeypatch.setattr(torch, "multinomial", draw)
    handle = model.model.register_forward_hook(force, with_kwargs=True)
    try:
        measured, _ = forward(model, rows, [full_patch()] * 4)
    finally:
        handle.remove()
    for row, length in enumerate(lengths):
        trace = measured[row]["intervention_trace"]
        assert samples.count(rows[row]["seed"]) == length
        assert trace["decode_patch_count"] == trace["active_decode_forward_count"] == length - 1
        assert [e["processed_generated_token_index"] for e in trace["events"] if e["phase"] == "decode"] == list(range(1, length))
        for step, mask in enumerate(masks[1:], 1):
            assert bool(mask[row, -1]) is (step < length)
    clean(model)


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_exception_cleanup_and_clean_repeat(phase):
    model = tiny_backend(dtype=torch.bfloat16)
    model.model.generation_config.eos_token_id = None
    rows = requests(2)
    baseline, _ = forward(model, rows)

    def fail(module, args):
        if (args[0].shape[1] == 1) is (phase == "decode"):
            raise RuntimeError("controlled failure")

    handle = model.layers[2].register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="controlled failure"):
            forward(model, rows, [full_patch()] * 2)
    finally:
        handle.remove()
    clean(model)
    repeat, _ = forward(model, rows)
    assert without_time(baseline) == without_time(repeat)


@pytest.mark.parametrize("problem", ["position", "mode", "last_layer", "nan", "direction", "zero_cap", "retention", "duplicate", "non_bf16"])
def test_invalid_new_configuration_fails_cleanly(problem):
    model = tiny_backend(dtype=torch.float32 if problem == "non_bf16" else torch.bfloat16)
    value, options = full_patch(), {}
    if problem == "position":
        value["position"] = "record"
    elif problem == "mode":
        value["mode"] = "project"
    elif problem == "last_layer":
        value["layer"] = 2
    elif problem == "nan":
        value["dose"] = float("nan")
    elif problem == "direction":
        value["direction"] = [1.0]
    elif problem == "zero_cap":
        options["cap"] = 0
    elif problem == "retention":
        options["prefill_evidence_rows"] = (0,)
    elif problem == "duplicate":
        options.update(prefill_evidence_output={}, prefill_evidence_rows=(0, 0))
    with pytest.raises(ValueError):
        forward(model, requests(1), [value], **options)
    clean(model)


def test_new_retention_is_rejected_on_legacy_schedule():
    model = tiny_backend(dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="explicit full-prefix"):
        StateBackend(model).forward(requests(1), prefill_evidence_output={})
    clean(model)


def test_old_record_schedule_matches_immutable_archived_implementation():
    source = Path(__file__).parents[1] / "executions/qwen32-curiosity-suppression-screen-v1-admission2/src/curiosity_study/state_backend.py"
    if not source.exists():
        pytest.skip("Archived implementation unavailable")
    namespace = {"__package__": "curiosity_study"}
    exec(compile(source.read_text(), str(source), "exec"), namespace)
    model = tiny_backend(dtype=torch.bfloat16)
    model.model.generation_config.eos_token_id = None
    kwargs = dict(interventions=[patch(), None], max_new_tokens=5,
                  intervention_schedule="record_prefill_plus_decode", layers=(1,))
    previous, a = namespace["StateBackend"](model).forward(requests(2), **kwargs)
    current, b = StateBackend(model).forward(requests(2), **kwargs)
    assert without_time(previous) == without_time(current)
    np.testing.assert_array_equal(a, b)
    clean(model)
