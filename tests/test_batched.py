from types import SimpleNamespace
import pytest
import torch
from curiosity_study.batched import BatchedHFModel
from curiosity_study.credit import build_variant_manifest


class Tokenizer:
    pad_token_id = 0

    def encode(self, label, **kwargs):
        return [ord(label) - ord("A") + 1]

    def decode(self, ids, **kwargs):
        return ",".join(map(str, ids))


class TinyModel:
    config = SimpleNamespace(model_type="qwen3")
    generation_config = SimpleNamespace(eos_token_id=7)

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        rows = kwargs["input_ids"].shape[0]
        logits = torch.zeros(rows, 1, 8)
        # A stochastic distribution independent of the neighbor and padding width.
        logits[:, :, 1:5] = torch.tensor([2.0, 3.0, 1.0, 2.0])
        return SimpleNamespace(logits=logits, past_key_values=object())


def backend():
    model = BatchedHFModel.__new__(BatchedHFModel)
    model.torch, model.device, model.active_hooks = torch, "cpu", 0
    model.tokenizer, model.model = Tokenizer(), TinyModel()
    model.encode = lambda messages, enable_thinking=False: (
        str(messages),
        SimpleNamespace(input_ids=torch.tensor([messages])),
    )
    return model


def test_padding_positions_and_early_finished_rows():
    model = backend()
    results = model.infer_batch(
        [
            {"messages": [1, 2, 3, 4], "seed": 1, "greedy": True, "max_new_tokens": 1},
            {"messages": [2, 3], "seed": 2, "greedy": True, "max_new_tokens": 3},
        ],
        ["one", "two"],
    )
    assert results[1]["input_ids"] == [0, 0, 2, 3]
    assert results[1]["position_ids"] == [0, 0, 0, 1]
    assert results[0]["generated_token_ids"] == [2]
    assert results[1]["generated_token_ids"] == [2, 2, 2]
    assert model.model.calls[1]["attention_mask"].tolist() == [[1, 1, 1, 1, 0], [0, 0, 1, 1, 1]]
    assert model.model.calls[2]["position_ids"].tolist() == [[3], [3]]
    assert results[0]["batch"]["id"] == results[1]["batch"]["id"]


def test_row_rng_is_independent_of_neighbor_lifetime_and_order():
    model = backend()
    a = {"messages": [1, 2, 3], "seed": 71, "max_new_tokens": 32}
    b = {"messages": [2], "seed": 19, "max_new_tokens": 1}
    first = model.infer_batch([a, b], ["a", "b"])
    second = model.infer_batch([{**b, "max_new_tokens": 30}, a], ["b", "a"])
    assert first[0]["generated_token_ids"] == second[1]["generated_token_ids"]
    assert first[0]["sampling"] == second[1]["sampling"]


def test_end_token_is_terminal():
    model = backend()
    model.model.generation_config = SimpleNamespace(eos_token_id=2)
    result = model.infer_batch([{"messages": [1], "seed": 1, "greedy": True}], ["a"])[0]
    assert result["generated_token_ids"] == [2]
    assert result["ended_with_eos"]
    assert len(model.model.calls) == 1


def test_batch_backend_is_bound_into_assignment_hash():
    base = {"candidate_labels": "ABCD", "inference_mode": "think_sampled", "max_new_tokens": 2048}
    execution = {"backend": "torch_batched_v1", "batch_size": 32}
    serial = build_variant_manifest(2, 1, {"model_id": "mock"}, "validation", base)
    batch = build_variant_manifest(2, 1, {"model_id": "mock"}, "validation", {**base, "execution": execution})
    assert batch["planned_behavior_sha256"] != serial["planned_behavior_sha256"]
    assert all(a["execution"] == execution for a in batch["diagnostics"])
    with pytest.raises(ValueError, match="Unsupported batch"):
        build_variant_manifest(2, 1, {}, "validation", {**base, "execution": {**execution, "batch_size": 64}})
