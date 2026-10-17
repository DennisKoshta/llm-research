from types import SimpleNamespace
import numpy as np
import torch
from curiosity_study.batched import BatchedHFModel


class Tokenizer:
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        return {
            "input_ids": [ord(c) % 80 + 1 for c in text],
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }

    def encode(self, label, **kwargs):
        return {"Yes": [100], "No": [101], "</think>": [102]}[label]

    def decode(self, ids, **kwargs):
        return "".join(
            {5: "reason", 102: "</think>", 10: "\n", 100: "Yes", 101: "No", 127: ""}.get(i, "?") for i in ids
        )


class ScriptedModel:
    config = SimpleNamespace(model_type="qwen3", hidden_size=4)
    generation_config = SimpleNamespace(eos_token_id=127)

    def __init__(self):
        self.layers = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
        self.step = 0
        self.sequences = [[5, 102, 10, 100, 127], [102, 10, 101, 127, 127]]

    def __call__(self, **kwargs):
        hidden = kwargs["input_ids"].float().unsqueeze(-1).repeat(1, 1, 4)
        for layer in self.layers:
            hidden = layer(hidden)
        logits = torch.full((2, 1, 128), -100.0)
        for row in range(2):
            seq = self.sequences[row]
            logits[row, 0, seq[min(self.step, len(seq) - 1)]] = 100.0
        self.step += 1
        return SimpleNamespace(logits=logits, past_key_values=object())


def backend():
    model = BatchedHFModel.__new__(BatchedHFModel)
    model.torch, model.device, model.active_hooks = torch, "cpu", 0
    model.tokenizer = Tokenizer()
    model.model = ScriptedModel()
    model.layers = model.model.layers
    model.encode = lambda messages, enable_thinking=False: (
        messages,
        SimpleNamespace(input_ids=torch.tensor([model.tokenizer(messages)["input_ids"]])),
    )
    return model


def requests(limit=8):
    return [
        dict(
            messages=text,
            labels=["Yes", "No"],
            seed=i,
            enable_thinking=True,
            greedy=True,
            max_new_tokens=limit,
        )
        for i, text in enumerate(["Red. End of record. New?", "Longer record. End of record. Already known?"])
    ]


def test_trace_reads_pre_answer_state_and_preserves_generation():
    m = backend()
    capture = {}
    rows = m.infer_batch(
        requests(), ["a", "b"], capture_plan={"layers": [0, 1], "steps": [1, 2]}, capture_output=capture
    )
    baseline = backend().infer_batch(requests(), ["a", "b"])
    assert [r["generated_token_ids"] for r in rows] == [r["generated_token_ids"] for r in baseline]
    names = list(capture["positions"])
    np.testing.assert_array_equal(capture["hidden"][:, :, names.index("think_end")], 102.0)
    # The decision vector represents the newline BEFORE Yes/No, not the answer token.
    np.testing.assert_array_equal(capture["hidden"][:, :, names.index("decision")], 10.0)
    for i, r in enumerate(rows):
        length = len(r["unpadded_input_ids"])
        assert capture["token_positions"][i, names.index("decision")] == length + (2 if i == 0 else 1)
    assert m.active_hooks == 0


def test_capped_trace_does_not_invent_final_position():
    m = backend()
    capture = {}
    m.infer_batch(
        requests(2), ["a", "b"], capture_plan={"layers": [0, 1], "steps": [1]}, capture_output=capture
    )
    pi = list(capture["positions"]).index("decision")
    assert not capture["available"][:, :, pi].any()
    assert np.isnan(capture["hidden"][:, :, pi]).all()
    assert m.active_hooks == 0


def test_decision_probabilities_are_measured_before_the_answer():
    m = backend()
    capture = {}
    m.infer_batch(
        requests(), ["a", "b"], capture_plan={"layers": [0, 1], "steps": [1]}, capture_output=capture
    )
    np.testing.assert_array_equal(
        capture["decision_label_probabilities_temperature_1"], [[1.0, 0.0], [0.0, 1.0]]
    )
    np.testing.assert_array_equal(capture["closure_counts"], [1, 1])


def test_multiple_closures_use_last_delimiter_without_stale_decision():
    m = backend()
    m.model.sequences = [[102, 10, 100, 102, 10, 101, 127], [102, 10, 101, 102, 10, 100, 127]]
    capture = {}
    rows = m.infer_batch(requests(), ["a", "b"], capture_plan={"layers": [0, 1]}, capture_output=capture)
    np.testing.assert_array_equal(capture["closure_counts"], [2, 2])
    np.testing.assert_array_equal(capture["decision_label_probabilities_temperature_1"], [[0, 1], [1, 0]])
    names = list(capture["positions"])
    for i, row in enumerate(rows):
        n = len(row["unpadded_input_ids"])
        assert capture["token_positions"][i, names.index("think_end")] == n + 3
        assert capture["token_positions"][i, names.index("decision")] == n + 4
