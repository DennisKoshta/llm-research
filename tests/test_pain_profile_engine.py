"""Synthetic CPU BF16 tests; no published model, adapter or downloaded code runs."""

import ast
from copy import deepcopy
import hashlib
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from curiosity_study import pain_profile_engine as engine
from curiosity_study.artifacts import digest


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class Tokenizer:
    def decode(self, tokens, *, skip_special_tokens):
        assert skip_special_tokens is False
        return " ".join(map(str, tokens))


def tiny():
    torch.manual_seed(473)
    config = Qwen2Config(
        vocab_size=32, hidden_size=8, intermediate_size=16, num_hidden_layers=62,
        num_attention_heads=2, num_key_value_heads=1, attn_implementation="eager",
        use_sliding_window=False, max_window_layers=62,
    )
    return Qwen2ForCausalLM(config).to(dtype=torch.bfloat16).eval()


def items(count=3):
    return [
        {"trial_id": f"row-{i}", "prompt_ids": [2, 3, 4, 5][i % 3:], "gen_seed": 135 + i,
         "button_names": ["violet", "yellow"], "target_name": "violet",
         "prob_token_ids": [[4], [5]]}
        for i in range(count)
    ]


def collect(model=None, **overrides):
    model = tiny() if model is None else model
    kwargs = dict(
        items=items(), direction_bf16=torch.arange(-4, 4, dtype=torch.bfloat16) / 8,
        pain_unit_float32=torch.arange(1, 9, dtype=torch.float32) / torch.arange(1, 9, dtype=torch.float32).norm(),
        coefficient=1, prefill_policy="full_prefix", eos_token_ids=[], pad_token_id=0,
        evidence_mode="compatibility",
    )
    kwargs.update(overrides)
    return engine.generate_segment(model, Tokenizer(), model.model.layers, **kwargs)


@pytest.fixture(scope="module")
def paired():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = tiny()
        return {name: collect(model, **kw) for name, kw in {
            "baseline": {"coefficient": 0},
            "unhooked": {"coefficient": 0, "steering_enabled": False},
            "zero": {"direction_bf16": torch.zeros(8, dtype=torch.bfloat16)},
            "full": {}, "final": {"prefill_policy": "final_prefix"},
            "repeat": {"coefficient": 0},
        }.items()}
    finally:
        torch.set_num_threads(old)


def test_actual_bf16_full_and_final_masks_and_independent_geometry(paired):
    for policy in ("full", "final"):
        result = paired[policy]
        report = result["evidence"]
        assert report["complete"] and report["integrity_passed"], report["error"]
        assert report["cpu_synthetic_execution"] and not report["scientific_admission"]
        assert report["hooks_before"] == report["hooks_after"] == {}
        assert len(report["forwards"]) == 8
        first = report["forwards"][0]
        assert first["input_ids"] == [[2, 3, 4, 5], [0, 3, 4, 5], [0, 0, 4, 5]]
        assert first["position_ids"] == [[0, 1, 2, 3], [0, 0, 1, 2], [0, 0, 0, 1]]
        masks = [[1.0] * 4, [0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]
        if policy == "final":
            masks = [[0.0, 0.0, 0.0, 1.0]] * 3
        assert first["steering_mask"] == masks
        assert [r["active_positions"] for r in first["geometry"]] == ([4, 3, 2] if policy == "full" else [1] * 3)
        for frame in report["forwards"]:
            assert frame["activation_dtypes"] == {"before": "torch.bfloat16", "after": "torch.bfloat16"}
            assert all(r["max_abs_error"] == 0 and r["padding_identical"] for r in frame["geometry"])
            assert len(frame["full_fp32_logits_sha256"]) == 3
            assert all(frame[key] == 1 for key in ("steering_hook_calls", "before_observer_calls", "after_observer_calls", "monitor_hook_calls"))
        before = torch.from_numpy(result["arrays"]["f00_r00_before"]).to(torch.bfloat16)
        after = torch.from_numpy(result["arrays"]["f00_r00_after"]).to(torch.bfloat16)
        delta = torch.arange(-4, 4, dtype=torch.bfloat16) / 8
        expected = before.clone()
        if policy == "full":
            expected += delta
        else:
            expected[-1] += delta
        assert torch.equal(after, expected)
    assert paired["full"]["evidence"]["forwards"][0]["full_fp32_logits_sha256"] != paired["final"]["evidence"]["forwards"][0]["full_fp32_logits_sha256"]
    assert [r["before_sha256"] for r in paired["full"]["evidence"]["forwards"][0]["geometry"]] == [
        r["before_sha256"] for r in paired["final"]["evidence"]["forwards"][0]["geometry"]]


def test_true_zero_addition_and_clean_repeat_have_full_vocabulary_identity(paired):
    baseline = paired["baseline"]
    for name in ("zero", "repeat", "unhooked"):
        other = paired[name]
        assert other["records"] == baseline["records"]
        for left, right in zip(baseline["evidence"]["forwards"], other["evidence"]["forwards"]):
            for key in ("input_ids", "attention_mask", "position_ids", "full_fp32_logits_sha256", "sampling"):
                assert left[key] == right[key]
    assert all(not any(row) for row in baseline["evidence"]["forwards"][0]["steering_mask"])
    assert all(frame["steering_hook_calls"] == 0 and frame["read_only_projection_calls"] == 1
               for frame in paired["unhooked"]["evidence"]["forwards"])


def test_finished_rows_still_draw_forward_padding_attend_and_receive_decode_patch(monkeypatch):
    model = tiny()
    original = model.forward
    calls = 0

    def force(*args, **kwargs):
        nonlocal calls
        output = original(*args, **kwargs)
        logits = torch.full_like(output.logits, -1000)
        logits[:, -1, 7] = 1000
        if calls == 0:
            logits[0, -1, 7] = -1000
            logits[0, -1, 31] = 1000
        calls += 1
        output.logits = logits
        return output

    monkeypatch.setattr(model, "forward", force)
    result = collect(model, eos_token_ids=[31])
    report = result["evidence"]
    assert report["integrity_passed"]
    assert result["records"][0]["answer_token_ids"] == [31]
    assert result["records"][0]["n_fwd"] == 1
    assert result["records"][1]["answer_token_ids"] == [7] * 8
    for index, frame in enumerate(report["forwards"]):
        assert frame["sampling"]["uniform_draws_per_row"] == [1, 1, 1]
        assert frame["cache_length_before"] == (0 if index == 0 else 4 + index - 1)
        assert frame["cache_length_after"] == 4 + index
        if index:
            assert frame["input_ids"][0] == [31 if index == 1 else 0]
            assert frame["attention_mask"][0] == [1] * (4 + index)
            assert frame["position_ids"][0] == [4 + index - 1]
            assert frame["sampling"]["alive_before"][0] is False
            assert frame["geometry"][0]["active_positions"] == 1
            assert frame["geometry"][0]["achieved_delta_norm_min"] > 0
    generator = torch.Generator().manual_seed(items()[0]["gen_seed"])
    uniforms = [float(torch.rand(1, generator=generator)) for _ in range(8)]
    assert [f["sampling"]["uniforms"][0] for f in report["forwards"]] == uniforms
    assert report["forwards"][-1]["sampling"]["rng_state_after_sha256"][0] == digest(engine.tensor_binding(generator.get_state()))


def test_eos_first_entire_cohort_stops_after_one_forward(monkeypatch):
    model = tiny()
    original = model.forward

    def eos(*args, **kwargs):
        output = original(*args, **kwargs)
        output.logits[:] = -1000
        output.logits[:, -1, 31] = 1000
        return output

    monkeypatch.setattr(model, "forward", eos)
    result = collect(model, eos_token_ids=[31])
    assert result["evidence"]["integrity_passed"]
    assert len(result["evidence"]["forwards"]) == 1
    assert all(row["answer_token_ids"] == [31] and not row["cap_hit"] for row in result["records"])


def test_boundary_token_is_retained_and_inverse_cdf_uses_left_search(monkeypatch):
    monkeypatch.setattr(engine, "TOP_P", 0.5)
    # Exactly representable softmax probabilities: .5, .25, .25.
    logits = torch.tensor([[math_log for math_log in [0.0, -0.6931471805599453, -0.6931471805599453]]]) * engine.TEMPERATURE
    calls = []

    def uniform(*args, **kwargs):
        calls.append(kwargs["generator"])
        return torch.tensor([0.8])

    monkeypatch.setattr(torch, "rand", uniform)
    generator = torch.Generator().manual_seed(3)
    selected, _ = engine.source_sample(logits, [generator], torch.device("cpu"))
    assert selected.tolist() == [1]  # second token starts exactly at cumulative mass .5
    assert calls == [generator]
    monkeypatch.setattr(torch, "rand", lambda *a, **k: torch.tensor([0.0]))
    selected, _ = engine.source_sample(logits, [generator], torch.device("cpu"))
    assert selected.tolist() == [0]


def test_independent_next_block_observer_detects_changed_actual_residual():
    model = tiny()
    handle = model.model.layers[39].register_forward_pre_hook(lambda module, args: (args[0] + 0.5, *args[1:]))
    before = engine.hooks(model)
    try:
        result = collect(model)
        assert result["evidence"]["complete"]
        assert not result["evidence"]["integrity_passed"]
        assert not result["evidence"]["checks"]["all_observed_arithmetic_exact"]
        assert engine.hooks(model) == before
    finally:
        handle.remove()


def test_decode_error_keeps_evidence_cleans_hooks_and_does_not_retry(monkeypatch):
    model = tiny()
    original = model.model.layers[39].forward
    calls = 0

    def broken(*args, **kwargs):
        nonlocal calls
        calls += 1
        if args[0].shape[1] == 1:
            raise RuntimeError("synthetic terminal decode failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(model.model.layers[39], "forward", broken)
    result = collect(model)
    assert calls == 2
    assert not result["records"] and not result["evidence"]["complete"]
    assert result["evidence"]["error"]["message"] == "synthetic terminal decode failure"
    assert len(result["evidence"]["forwards"]) == 1
    assert result["evidence"]["incomplete_forward"]["phase"] == "decode"
    assert result["arrays"] and not engine.hooks(model)


def test_nonfinite_vocabulary_evidence_cannot_become_successful_records(monkeypatch):
    model = tiny()
    original = model.forward

    def corrupt(*args, **kwargs):
        output = original(*args, **kwargs)
        output.logits[1, -1, 3] = float("nan")
        return output

    monkeypatch.setattr(model, "forward", corrupt)
    result = collect(model)
    assert not result["records"] and not result["evidence"]["complete"]
    assert len(result["evidence"]["forwards"]) == 1
    assert not result["evidence"]["checks"]["all_observed_logits_finite"]
    assert result["evidence"]["error"]["message"] == "Nonfinite actual full-vocabulary logits"
    assert not engine.hooks(model)


@pytest.mark.parametrize("fault,forwards", [("after_prefill", 1), ("after_first_decode", 2)])
def test_explicit_compatibility_fault_then_clean_repeat(fault, forwards):
    model = tiny()
    baseline = collect(model, coefficient=0)
    failure = collect(model, inject_failure_at=fault)
    assert not failure["evidence"]["complete"] and not failure["records"]
    assert len(failure["evidence"]["forwards"]) == forwards
    assert failure["evidence"]["error"]["message"] == f"controlled compatibility fault: {fault}"
    assert failure["evidence"]["checks"]["hooks_clean"] and not engine.hooks(model)
    repeat = collect(model, coefficient=0)
    assert repeat["records"] == baseline["records"]
    assert [f["full_fp32_logits_sha256"] for f in repeat["evidence"]["forwards"]] == [
        f["full_fp32_logits_sha256"] for f in baseline["evidence"]["forwards"]]


def test_benchmark_rejects_fault_injection_and_nonzero_unhooked_execution():
    with pytest.raises(ValueError, match="compatibility-only"):
        collect(evidence_mode="benchmark", inject_failure_at="after_prefill")
    with pytest.raises(ValueError, match="coefficient0"):
        collect(steering_enabled=False)
    with pytest.raises(ValueError, match="compatibility-only"):
        collect(evidence_mode="benchmark", one_row_only=0)
    with pytest.raises(ValueError, match="compatibility-only"):
        collect(one_row_only=1)


def test_actual32_row_one_row_intervention_keeps_other31_predictions_identical():
    model = tiny()
    baseline = collect(model, items=items(32), coefficient=0)
    changed = collect(model, items=items(32), one_row_only=0)
    assert changed["evidence"]["integrity_passed"]
    assert changed["evidence"]["applied_coefficients"] == [1] + [0] * 31
    assert changed["records"][1:] == baseline["records"][1:]
    for before, after in zip(baseline["evidence"]["forwards"], changed["evidence"]["forwards"]):
        assert before["full_fp32_logits_sha256"][1:] == after["full_fp32_logits_sha256"][1:]
        assert before["sampling"]["uniforms"][1:] == after["sampling"]["uniforms"][1:]
        assert all(row["coefficient"] == 0 and row["active_positions"] == 0 for row in after["geometry"][1:])


def test_benchmark_saves_bounded_selected_vectors_but_checks_every_row():
    result = collect(evidence_mode="benchmark", items=items(32))
    assert result["evidence"]["integrity_passed"]
    assert all(len(frame["geometry"]) == 32 for frame in result["evidence"]["forwards"])
    assert all(value.shape[0] <= 2 for value in result["arrays"].values())
    assert all("r00" in key for key in result["arrays"])
    for key, value in result["arrays"].items():
        assert engine.array_binding(value) == result["evidence"]["arrays"][key]
    assert sum(value.nbytes for value in result["arrays"].values()) <= 2 * 9 * 8 * 4


def test_monitor_always_uses_separate_fp32_pain_unit():
    model = tiny()
    actual = []
    handle = model.model.layers[61].register_forward_hook(
        lambda module, args, output: actual.append((output[0] if isinstance(output, tuple) else output)[:, -1].detach().float().clone())
    )
    try:
        result = collect(model)
    finally:
        handle.remove()
    unit = torch.arange(1, 9, dtype=torch.float32)
    unit /= unit.norm()
    report = result["evidence"]
    for index, frame in enumerate(report["forwards"]):
        before = result["arrays"][f"f{index:02d}_r00_before"][-1]
        np.testing.assert_allclose(frame["prepatch_pain_projection"][0], before @ unit.numpy(), rtol=1e-6, atol=1e-6)
        assert frame["post_layer61_pain_projection"] == (actual[index] @ unit).tolist()
    assert [row["prefill_proj_monitor"] for row in result["records"]] == (actual[0] @ unit).tolist()
    assert report["pain_unit"] != report["direction"]


@pytest.mark.parametrize("mutation", ["bool_dose", "wrong_dtype", "bad_seed", "duplicate", "one_token", "overlap", "policy"])
def test_invalid_request_rejected_before_model(monkeypatch, mutation):
    model = tiny()
    kwargs = {}
    rows = deepcopy(items())
    if mutation == "bool_dose":
        kwargs["coefficient"] = True
    elif mutation == "wrong_dtype":
        kwargs["direction_bf16"] = torch.ones(8)
    elif mutation == "bad_seed":
        rows[0]["gen_seed"] = True
    elif mutation == "duplicate":
        rows[1]["trial_id"] = rows[0]["trial_id"]
    elif mutation == "one_token":
        rows[0]["prompt_ids"] = [2]
    elif mutation == "overlap":
        rows[0]["prob_token_ids"] = [[4], [4]]
    else:
        kwargs["prefill_policy"] = "record"
    monkeypatch.setattr(model, "forward", lambda *a, **k: pytest.fail("invalid request reached model"))
    with pytest.raises(ValueError):
        collect(model, items=rows, **kwargs)
    assert not engine.hooks(model)


def test_upstream_sampler_and_steering_expressions_are_structurally_retained():
    path = Path(__file__).resolve().parents[1] / "references/pain-axis-positive-control/upstream/v2_controls/choice_controls/profile/src/run_profile.py"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == engine.SOURCE_SHA256
    upstream = ast.parse(path.read_text())
    local = ast.parse(inspect.getsource(engine))

    def assignments(tree):
        return {ast.unparse(node.targets[0]): ast.dump(node.value, include_attributes=False)
                for node in ast.walk(tree) if isinstance(node, ast.Assign) and len(node.targets) == 1}

    old, new = assignments(upstream), assignments(local)
    for name in ("probs", "sp", "cum", "cum2", "total", "j", "hs", "pos", "cur_pos", "alive", "n_out", "pm", "mm"):
        assert old[name] == new[name], name
    for name in ("steer_hook", "monitor_hook"):
        source = next(node for node in ast.walk(upstream) if isinstance(node, ast.FunctionDef) and node.name == name)
        adapted = next(node for node in ast.walk(local) if isinstance(node, ast.FunctionDef) and node.name == name)
        expected = [ast.dump(node.value, include_attributes=False) for node in ast.walk(source)
                    if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) in {"add", "G['proj']", "G['mon']"}]
        observed = [ast.dump(node.value, include_attributes=False) for node in ast.walk(adapted)
                    if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) in {"add", "G['proj']", "G['mon']"}]
        assert observed == expected
