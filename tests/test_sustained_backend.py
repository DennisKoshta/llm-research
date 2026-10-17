"""CPU transformer tests for opt-in sustained timing, identity and row isolation."""

from copy import deepcopy

import numpy as np
import pytest
import torch
from transformers import Qwen3ForCausalLM

from curiosity_study.artifacts import digest
from curiosity_study.state_backend import PREFILL_PADDING_MULTIPLE, StateBackend
from test_state_backend import tiny_backend


SUSTAINED = "record_prefill_plus_decode"


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def requests(count=4):
    result = []
    for row in range(count):
        ids = [3 + row % 40, 9, 12] + [7] * (row % 5 + 1)
        result.append(
            dict(
                token_prefix=ids,
                token_prefix_sha256=digest(ids),
                prefix_positions=dict(record=1, final=len(ids) - 1),
                labels=["Yes", "No"],
                seed=1000 + row,
            )
        )
    return result


def patch(dose=0.1, layer=1):
    direction = np.arange(16, dtype=np.float32) - 7.5
    direction /= np.linalg.norm(direction)
    return dict(layer=layer, position="record", mode="add", direction=direction, dose=dose)


def observe_forward(
    model, rows, patches=None, *, schedule=SUSTAINED, cap=6, greedy=False, capture_layers=(1,)
):
    calls = []

    def observe(module, args, kwargs, output):
        calls.append(
            {
                **{
                    key: kwargs[key].detach().clone()
                    for key in ("input_ids", "attention_mask", "position_ids")
                },
                "logits": output.logits.detach().clone(),
            }
        )

    handle = model.model.register_forward_hook(observe, with_kwargs=True)
    try:
        kwargs = {} if schedule is None else dict(intervention_schedule=schedule)
        measured, hidden = StateBackend(model).forward(
            rows,
            layers=capture_layers,
            interventions=patches,
            max_new_tokens=cap,
            greedy=greedy,
            **kwargs,
        )
    finally:
        handle.remove()
    assert model.active_hooks == 0
    assert all(not layer._forward_hooks for layer in model.layers)
    assert not model.model._forward_hooks
    return measured, hidden, calls


def without_time(rows):
    return [{key: value for key, value in row.items() if key != "batch_seconds"} for row in rows]


def assert_calls_equal(left, right, rows=None):
    assert len(left) == len(right)
    for a, b in zip(left, right):
        for key in a:
            av, bv = a[key], b[key]
            if rows is not None:
                av, bv = av[rows], bv[rows]
            assert torch.equal(av, bv), key


@pytest.mark.parametrize("cap", [0, 5])
@pytest.mark.parametrize("patched", [False, True])
def test_implicit_legacy_equals_explicit_legacy_without_new_result_fields(cap, patched):
    model = tiny_backend(dtype=torch.bfloat16)
    model.model.generation_config.eos_token_id = None
    rows = requests()
    patches = [patch(), None, None, None] if patched else None
    implicit, values_a, calls_a = observe_forward(model, rows, patches, schedule=None, cap=cap)
    explicit, values_b, calls_b = observe_forward(model, rows, patches, schedule="prefill_only", cap=cap)
    assert without_time(implicit) == without_time(explicit)
    np.testing.assert_array_equal(values_a, values_b)
    assert_calls_equal(calls_a, calls_b)
    assert all("intervention_trace" not in row for row in implicit)
    assert all(row["prefill_width"] % PREFILL_PADDING_MULTIPLE == 0 for row in implicit)
    if cap:
        assert all(
            row["completion"]["intervention_schedule"] == "prefill_only; no active hooks during decoding"
            for row in implicit
        )


def test_actual_layer15_bf16_32rows_zero_sham_and_one_row_isolation():
    model = tiny_backend(dtype=torch.bfloat16)
    config = deepcopy(model.model.config)
    config.num_hidden_layers = 16
    torch.manual_seed(190)
    model.model = Qwen3ForCausalLM(config).to(dtype=torch.bfloat16).eval()
    model.layers = model.model.model.layers
    model.model.generation_config.eos_token_id = None
    rows = requests(32)
    baseline, states_a, calls_a = observe_forward(model, rows, cap=7, capture_layers=(15,))
    sham, states_b, calls_b = observe_forward(
        model, rows, [patch(0, 15) for _ in rows], cap=7, capture_layers=(15,)
    )
    changed, states_c, calls_c = observe_forward(
        model, rows, [patch(0.25, 15)] + [None] * 31, cap=7, capture_layers=(15,)
    )
    assert_calls_equal(calls_a, calls_b)
    assert_calls_equal(calls_a, calls_c, list(range(1, 32)))
    np.testing.assert_array_equal(states_a, states_b)
    np.testing.assert_array_equal(states_a[1:], states_c[1:])
    assert [r["completion"]["generated_token_ids"] for r in baseline] == [
        r["completion"]["generated_token_ids"] for r in sham
    ]
    assert [r["completion"]["generated_token_ids"] for r in baseline[1:]] == [
        r["completion"]["generated_token_ids"] for r in changed[1:]
    ]
    for row in sham:
        trace = row["intervention_trace"]
        assert trace["prefill_patch_count"] == 1 and trace["decode_patch_count"] == 6
        assert all(event["delta_norm"] == 0 for event in trace["events"])
    assert all(event["delta_norm"] > 0 for event in changed[0]["intervention_trace"]["events"])
    assert all(row["intervention_trace"]["events"] == [] for row in changed[1:])


def test_bf16_geometry_independently_matches_actual_residuals_and_only_record_prefill():
    model = tiny_backend(dtype=torch.bfloat16)
    model.model.generation_config.eos_token_id = None
    rows, patches = requests(2), [patch(0.1), patch(-0.2)]
    before, after = [], []

    def read_before(module, args, output):
        before.append(output[0].detach().float().clone())

    def read_after(module, args):
        after.append(args[0].detach().float().clone())

    first = model.layers[1].register_forward_hook(read_before)
    second = model.layers[2].register_forward_pre_hook(read_after)
    try:
        measured, _ = StateBackend(model).forward(
            rows,
            layers=[1],
            interventions=patches,
            max_new_tokens=4,
            greedy=True,
            intervention_schedule=SUSTAINED,
        )
    finally:
        first.remove()
        second.remove()
    assert len(before) == len(after) == 4
    _, _, inputs, targets = StateBackend(model).prepare(rows)
    for row in range(2):
        trace = measured[row]["intervention_trace"]
        events = trace["events"]
        assert trace["total_patch_count"] == 4 and trace["active_decode_forward_count"] == 3
        assert events[0]["unpadded_token_index"] == 1
        unchanged = torch.ones(inputs["input_ids"].shape[1], dtype=torch.bool)
        unchanged[targets["record"][row]] = False
        assert torch.equal(before[0][row, unchanged], after[0][row, unchanged])
        for step, event in enumerate(events):
            pos = targets["record"][row] if step == 0 else 0
            actual_before, actual_after = before[step][row, pos], after[step][row, pos]
            direction = torch.tensor(patches[row]["direction"])
            requested = actual_before + patches[row]["dose"] * direction
            assert torch.equal(actual_after, requested.bfloat16().float())
            expected = dict(
                before_norm=actual_before.norm().item(),
                after_norm=actual_after.norm().item(),
                delta_norm=(actual_after - actual_before).norm().item(),
                requested_delta_norm=(requested - actual_before).norm().item(),
                rounding_error_norm=(actual_after - requested).norm().item(),
                projection_before=torch.dot(actual_before, direction / direction.norm()).item(),
                projection_after=torch.dot(actual_after, direction / direction.norm()).item(),
            )
            for key, value in expected.items():
                assert event[key] == pytest.approx(value, abs=1e-7, rel=1e-6)
            assert event["requested_scalar_delta"] == patches[row]["dose"]
            assert event["arithmetic_scalar_float32"] == torch.tensor(patches[row]["dose"]).item()
            assert event["activation_dtype"] == "torch.bfloat16"
            if step:
                assert event["phase"] == "decode" and event["token_index"] == 0
                assert event["processed_generated_token_index"] == step
                assert (
                    event["processed_token_id"]
                    == measured[row]["completion"]["generated_token_ids"][step - 1]
                )
                assert event["unpadded_token_index"] == len(rows[row]["token_prefix"]) + step - 1
        assert trace["cumulative_achieved_delta_norm"] == sum(event["delta_norm"] for event in events)
    assert model.active_hooks == 0


def test_mixed_eos_masks_terminal_rows_and_does_not_draw_rng_after_completion(monkeypatch):
    model = tiny_backend(dtype=torch.bfloat16)
    model.model.generation_config.eos_token_id = 127
    rows, generated_lengths = requests(), [1, 2, 5, 3]
    calls, samples, before, after = [], [], [], []
    multinomial = torch.multinomial

    def sample(*args, **kwargs):
        samples.append(kwargs["generator"].initial_seed())
        return multinomial(*args, **kwargs)

    def force_termination(module, args, kwargs, output):
        step = len(calls)
        calls.append({key: kwargs[key].clone() for key in ("input_ids", "attention_mask", "position_ids")})
        logits = torch.full_like(output.logits, -1000)
        for row, length in enumerate(generated_lengths):
            token = 127 if step + 1 == length and row != 2 else 6 + row
            logits[row, -1, token] = 1000
        output.logits = logits

    handles = [
        model.model.register_forward_hook(force_termination, with_kwargs=True),
        model.layers[1].register_forward_hook(lambda module, args, output: before.append(output[0].clone())),
        model.layers[2].register_forward_pre_hook(lambda module, args: after.append(args[0].clone())),
    ]
    monkeypatch.setattr(torch, "multinomial", sample)
    try:
        measured, _ = StateBackend(model).forward(
            rows, interventions=[patch(0.25) for _ in rows], max_new_tokens=5, intervention_schedule=SUSTAINED
        )
    finally:
        for handle in handles:
            handle.remove()
    assert len(calls) == 5
    for row, length in enumerate(generated_lengths):
        result, trace = measured[row], measured[row]["intervention_trace"]
        assert len(result["completion"]["generated_token_ids"]) == length
        assert samples.count(rows[row]["seed"]) == length
        assert trace["active_decode_forward_count"] == trace["decode_patch_count"] == length - 1
        assert trace["total_patch_count"] == length
        for step in range(1, 5):
            active = step < length
            assert calls[step]["attention_mask"][row, -1].item() == active
            assert (not torch.equal(before[step][row, 0], after[step][row, 0])) == active
            if not active:
                assert all(event.get("processed_generated_token_index") != step for event in trace["events"])
        assert result["hooks_remaining"] == 0
    assert not model.model._forward_hooks and model.active_hooks == 0


def test_decode_exception_removes_hooks_and_next_clean_run_matches_baseline():
    model = tiny_backend(dtype=torch.bfloat16)
    model.model.generation_config.eos_token_id = None
    rows = requests()
    baseline, hidden, calls = observe_forward(model, rows, cap=5)

    def fail_in_decode(module, args):
        if args[0].shape[1] == 1:
            raise RuntimeError("synthetic downstream decoding failure")

    handle = model.layers[2].register_forward_pre_hook(fail_in_decode)
    try:
        with pytest.raises(RuntimeError, match="synthetic downstream"):
            StateBackend(model).forward(
                rows, interventions=[patch()] * 4, max_new_tokens=5, intervention_schedule=SUSTAINED
            )
    finally:
        handle.remove()
    assert model.active_hooks == 0 and all(not layer._forward_hooks for layer in model.layers)
    fresh, fresh_hidden, fresh_calls = observe_forward(model, rows, cap=5)
    assert without_time(baseline) == without_time(fresh)
    np.testing.assert_array_equal(hidden, fresh_hidden)
    assert_calls_equal(calls, fresh_calls)


@pytest.mark.parametrize("problem", ["unknown", "zero_cap", "final", "project", "nan", "bad_direction"])
def test_sustained_invalid_plans_fail_without_hooks(problem):
    model = tiny_backend()
    value, cap, schedule = patch(), 3, SUSTAINED
    if problem == "unknown":
        schedule = "all_prompt_and_decode"
    elif problem == "zero_cap":
        cap = 0
    elif problem == "final":
        value["position"] = "final"
    elif problem == "project":
        value["mode"] = "project"
    elif problem == "nan":
        value["dose"] = float("nan")
    else:
        value["direction"] = [1.0]
    with pytest.raises(ValueError):
        StateBackend(model).forward(
            requests(1), interventions=[value], max_new_tokens=cap, intervention_schedule=schedule
        )
    assert model.active_hooks == 0 and all(not layer._forward_hooks for layer in model.layers)


def test_one_token_cap_has_prefill_event_but_no_decode_injection():
    model = tiny_backend(dtype=torch.bfloat16)
    rows, _, calls = observe_forward(model, requests(1), [patch()], cap=1)
    trace = rows[0]["intervention_trace"]
    assert len(calls) == trace["total_patch_count"] == trace["prefill_patch_count"] == 1
    assert trace["decode_patch_count"] == trace["active_decode_forward_count"] == 0
    assert trace["events"][0]["phase"] == "prefill"


def test_finite_parameters_with_overflowing_geometry_fail_before_generation():
    model = tiny_backend(dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="Nonfinite sustained prefill geometry"):
        StateBackend(model).forward(
            requests(1),
            interventions=[patch(1e30)],
            max_new_tokens=2,
            intervention_schedule=SUSTAINED,
        )
    assert model.active_hooks == 0 and all(not layer._forward_hooks for layer in model.layers)
