from types import SimpleNamespace
import numpy as np
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from curiosity_study.state_backend import StateBackend, verify_state_backend, PREFILL_PADDING_MULTIPLE
from curiosity_study.artifacts import digest
import pytest


class CharacterTokenizer:
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        return {
            "input_ids": [ord(c) % 80 + 1 for c in text],
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }

    def encode(self, label, **kwargs):
        return {"Yes": [100], "No": [101]}[label]

    def decode(self, ids, **kwargs):
        return str(ids)


def tiny_backend(dtype=torch.float32):
    torch.manual_seed(71)
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        attn_implementation="eager",
    )
    network = Qwen3ForCausalLM(config).to(dtype=dtype).eval()
    tokenizer = CharacterTokenizer()
    base = SimpleNamespace(
        torch=torch,
        device="cpu",
        active_hooks=0,
        model=network,
        layers=network.model.layers,
        tokenizer=tokenizer,
    )
    base.encode = lambda messages, enable_thinking=False: (
        messages,
        SimpleNamespace(input_ids=torch.tensor([tokenizer(messages)["input_ids"]])),
    )
    return base


def test_real_transformer_hook_cleanup_isolation_and_sham():
    model = tiny_backend()
    requests = [
        {"messages": "Color red. End of record. Is it new?", "labels": ["Yes", "No"]},
        {"messages": "Shape oval. End of record. Is it known?", "labels": ["Yes", "No"]},
    ]
    audit = verify_state_backend(model, requests)
    assert audit["passed"], audit["checks"]


def test_record_capture_cannot_contain_later_question_information():
    model = tiny_backend()
    requests = [
        {"messages": "Known color red. Report shape. End of record. New?", "labels": ["Yes", "No"]},
        {"messages": "Known color red. Report shape. End of record. Already known?", "labels": ["Yes", "No"]},
    ]
    _, values = StateBackend(model).forward(requests, [0, 1, 2])
    np.testing.assert_allclose(values[0, :, 0, :], values[1, :, 0, :], atol=1e-6)
    assert not np.allclose(values[0, :, 1, :], values[1, :, 1, :])


def test_aligned_prefill_preserves_every_real_token_position_and_mask():
    model = tiny_backend()
    requests = [
        {"messages": "Fact. End of record. New?", "labels": ["Yes", "No"]},
        {"messages": "Longer factual description. End of record. Known?", "labels": ["Yes", "No"]},
    ]
    _, ids, inputs, targets = StateBackend(model).prepare(requests)
    width = inputs["input_ids"].shape[1]
    assert width % PREFILL_PADDING_MULTIPLE == 0
    assert max(map(len, ids)) <= width < max(map(len, ids)) + PREFILL_PADDING_MULTIPLE
    for row, tokens in enumerate(ids):
        padding = width - len(tokens)
        assert inputs["input_ids"][row, padding:].tolist() == tokens
        assert inputs["input_ids"][row, :padding].tolist() == [model.tokenizer.pad_token_id] * padding
        assert inputs["attention_mask"][row].tolist() == [0] * padding + [1] * len(tokens)
        assert inputs["position_ids"][row, padding:].tolist() == list(range(len(tokens)))
        assert targets["record"][row] - padding == requests[row]["messages"].index("End of record.") + 13
        assert targets["final"][row] == width - 1


def test_cached_greedy_completion_matches_standard_generation():
    model = tiny_backend()
    backend = StateBackend(model)
    requests = [
        {"messages": "Color red. End of record. New?", "labels": ["Yes", "No"], "seed": 1},
        {"messages": "Shape round. End of record. Already known?", "labels": ["Yes", "No"], "seed": 2},
    ]
    rows, _ = backend.forward(requests, max_new_tokens=3, greedy=True)
    _, _, inputs, _ = backend.prepare(requests)
    with torch.inference_mode():
        # generate recomputes position IDs from the growing attention mask. Supplying
        # a fixed prefill position_ids tensor would freeze positions during decoding.
        reference = model.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=3,
            do_sample=False,
            pad_token_id=0,
        )
    width = inputs["input_ids"].shape[1]
    assert [r["completion"]["generated_token_ids"] for r in rows] == reference[:, width:].tolist()
    assert model.active_hooks == 0


def test_exact_prefix_replay_and_interior_intervention_causality():
    model = tiny_backend()
    backend = StateBackend(model)
    originals = [
        {"messages": "Fact. End of record. Reasoning. Answer?", "labels": ["Yes", "No"], "seed": 123}
    ]
    baseline, original_states = backend.forward(originals, [0, 1, 2])
    base = baseline[0]
    ids = base["unpadded_input_ids"]
    replay = {
        "token_prefix": ids,
        "token_prefix_sha256": digest(ids),
        "prefix_positions": {"record": base["unpadded_token_positions"]["record"], "think_end": len(ids) - 3},
        "labels": ["Yes", "No"],
        "seed": 123,
    }
    rows, values = backend.forward([replay], [0, 1, 2])
    assert rows[0]["label_logits"] == base["label_logits"]
    np.testing.assert_array_equal(values, original_states)
    assert rows[0]["exact_token_replay"]
    direction = np.arange(16, dtype=np.float32) / 10
    patch = dict(layer=0, position="think_end", mode="add", direction=direction, dose=2.0)
    changed, states = backend.forward([replay], [1, 2], positions=("record", "final"), interventions=[patch])
    np.testing.assert_array_equal(states[:, :, 0], original_states[:, 1:, 0])
    assert not np.allclose(states[:, :, 1], original_states[:, 1:, 1])
    assert changed[0]["label_logits"] != base["label_logits"]
    fresh, _ = backend.forward([replay])
    assert fresh[0]["label_logits"] == base["label_logits"]
    with pytest.raises(ValueError, match="hash mismatch"):
        backend.forward([{**replay, "token_prefix_sha256": "bad"}])
    with pytest.raises(ValueError, match="Invalid explicit prefix positions"):
        backend.forward([{**replay, "prefix_positions": {"record": len(ids)}}])


def test_projection_patch_sets_only_the_selected_component():
    model = tiny_backend()
    backend = StateBackend(model)
    request = dict(messages="Fact. End of record. Answer?", labels=["Yes", "No"], seed=1)
    direction = np.zeros(16)
    direction[3] = 1
    baseline, _ = backend.forward([request])
    rows, _ = backend.forward(
        [request],
        interventions=[dict(layer=1, position="final", mode="project", direction=direction, target=2.5)],
    )
    change = rows[0]["intervention"]
    assert change["projection_after"] == pytest.approx(2.5)
    assert change["delta_norm"] == pytest.approx(abs(2.5 - change["projection_before"]))
    after, _ = backend.forward([request])
    assert after[0]["label_logits"] == baseline[0]["label_logits"]


def test_matched_control_uses_current_reference_projection_and_isolates_other_rows():
    model = tiny_backend()
    backend = StateBackend(model)
    request = dict(messages="Fact. End of record. Answer?", labels=["Yes", "No"], seed=1)
    reference, control = np.eye(16)[[3, 7]]
    target = 1.75
    rows, _ = backend.forward(
        [request] * 3,
        interventions=[
            dict(layer=1, position="final", mode="project", direction=reference, target=target),
            dict(
                layer=1,
                position="final",
                mode="matched_add",
                direction=control,
                reference_direction=reference,
                target=target,
            ),
            None,
        ],
    )
    first, second = [row["intervention"] for row in rows[:2]]
    assert first["requested_scalar_delta"] == second["requested_scalar_delta"]
    assert first["requested_delta_norm"] == pytest.approx(second["requested_delta_norm"])
    assert first["reference_projection_after"] == pytest.approx(target)
    assert second["reference_projection_after"] == second["reference_projection_before"]
    baseline, _ = backend.forward([request] * 3)
    assert rows[2]["label_logits"] == baseline[2]["label_logits"]
    assert model.active_hooks == 0
    with pytest.raises(ValueError, match="reference direction"):
        backend.forward(
            [request],
            interventions=[
                dict(
                    layer=1,
                    position="final",
                    mode="matched_add",
                    direction=control,
                    reference_direction=reference * 2,
                    target=target,
                )
            ],
        )
    assert model.active_hooks == 0
