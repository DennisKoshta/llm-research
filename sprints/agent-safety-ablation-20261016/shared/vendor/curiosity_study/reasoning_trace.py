"""Read-only residual captures at prospectively defined points in a reasoning trace."""

import numpy as np
from .state_backend import MARKER


class ReasoningTrace:
    def __init__(self, model, rendered, ids, width, plan, destination, request_labels):
        if destination is None:
            raise ValueError("A trace destination is required")
        self.model = model
        self.destination = destination
        self.layers = list(plan["layers"])
        self.steps = list(plan.get("steps", [32, 128, 512]))
        self.names = (
            ["record", "prefill"] + [f"token_{step}" for step in self.steps] + ["think_end", "decision"]
        )
        self.name_indices = {name: i for i, name in enumerate(self.names)}
        self.lengths = list(map(len, ids))
        self.values = np.full(
            (len(ids), len(self.layers), len(self.names), model.model.config.hidden_size),
            np.nan,
            dtype=np.float32,
        )
        self.token_positions = np.full((len(ids), len(self.names)), -1, dtype=np.int64)
        self.closed = [False] * len(ids)
        self.decision_saved = [False] * len(ids)
        self.closure_counts = np.zeros(len(ids), dtype=np.int64)
        max_labels = max(map(len, request_labels))
        self.label_strings = np.full((len(ids), max_labels), "", dtype="<U32")
        self.label_tokens = np.full((len(ids), max_labels), -1, dtype=np.int64)
        self.decision_logits = np.full((len(ids), max_labels), np.nan, dtype=np.float32)
        self.decision_probabilities = np.full_like(self.decision_logits, np.nan)
        for row, labels in enumerate(request_labels):
            for col, label in enumerate(labels):
                tokens = model.tokenizer.encode(label, add_special_tokens=False)
                if len(tokens) != 1:
                    raise ValueError("Single-token labels required for decision scoring")
                self.label_strings[row, col] = label
                self.label_tokens[row, col] = tokens[0]
        end = model.tokenizer.encode("</think>", add_special_tokens=False)
        if len(end) != 1:
            raise ValueError("The reasoning-end delimiter must be one token")
        self.end_token = end[0]
        self.record_indices = []
        for text, row in zip(rendered, ids):
            if text.count(MARKER) != 1:
                raise ValueError("Exactly one record boundary required")
            offset = text.index(MARKER) + len(MARKER) - 1
            encoded = model.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            if encoded["input_ids"] != row:
                raise ValueError("Offset tokenizer mismatch")
            selected = [
                i for i, (start, end) in enumerate(encoded["offset_mapping"]) if start <= offset < end
            ]
            if len(selected) != 1:
                raise ValueError("Record boundary token ambiguous")
            idx = selected[0]
            if text[offset + 1 : encoded["offset_mapping"][idx][1]].strip():
                raise ValueError("Record boundary includes later question")
            self.record_indices.append(width - len(row) + idx)
            self.token_positions[len(self.record_indices) - 1, 0] = idx
        self.phase = "prefill"
        self.decode_targets = {}
        self.buffers = {}
        self.handles = []

    def save(self, row, li, name, tensor, position):
        pi = self.name_indices[name]
        existing = self.token_positions[row, pi]
        if existing not in (-1, position):
            raise ValueError("Capture position changed between layers")
        self.token_positions[row, pi] = position
        self.values[row, li, pi] = tensor.detach().float().cpu().numpy()

    def __enter__(self):
        try:
            for li, layer in enumerate(self.layers):

                def hook(module, args, output, li=li):
                    tensor = output[0] if isinstance(output, tuple) else output
                    # Clone the last position only, so no full-sequence storage stays alive.
                    self.buffers[li] = tensor[:, -1].detach().clone()
                    if self.phase == "prefill":
                        for row, position in enumerate(self.record_indices):
                            self.save(
                                row, li, "record", tensor[row, position], int(self.token_positions[row, 0])
                            )
                            self.save(row, li, "prefill", tensor[row, -1], self.lengths[row] - 1)
                    else:
                        for row, targets in self.decode_targets.items():
                            for name, position in targets:
                                self.save(row, li, name, tensor[row, -1], position)
                    return output

                self.handles.append(self.model.layers[layer].register_forward_hook(hook))
                self.model.active_hooks += 1
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def after_sample(self, row, token, generated_count, eos_ids, logits):
        if self.closed[row] and not self.decision_saved[row]:
            text = self.model.tokenizer.decode([token], skip_special_tokens=False)
            if text.strip() or token in eos_ids:
                # This vector was computed before sampling the first non-whitespace final token.
                position = self.lengths[row] + generated_count - 2
                for li, tensor in self.buffers.items():
                    self.save(row, li, "decision", tensor[row], position)
                scores = logits.detach().float()
                probabilities = scores.softmax(-1)
                for col, token_id in enumerate(self.label_tokens[row]):
                    if token_id >= 0:
                        self.decision_logits[row, col] = scores[int(token_id)].item()
                        self.decision_probabilities[row, col] = probabilities[int(token_id)].item()
                self.decision_saved[row] = True
        if token == self.end_token:
            self.closed[row] = True
            self.closure_counts[row] += 1
            # Match the scorer's final closing delimiter without interrupting a
            # malformed/multi-closure completion or retaining a stale decision.
            self.decision_saved[row] = False
            for name in ("think_end", "decision"):
                pi = self.name_indices[name]
                self.values[row, :, pi] = np.nan
                self.token_positions[row, pi] = -1
            self.decision_logits[row] = np.nan
            self.decision_probabilities[row] = np.nan

    def before_decode(self, tokens, generated, finished):
        self.phase = "decode"
        self.decode_targets = {}
        for row, token in enumerate(tokens):
            if finished[row]:
                continue
            count = len(generated[row])
            names = []
            if count in self.steps:
                names.append(f"token_{count}")
            if token == self.end_token:
                names.append("think_end")
            if names:
                self.decode_targets[row] = [(name, self.lengths[row] + count - 1) for name in names]

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
            self.model.active_hooks -= 1
        self.handles = []
        self.buffers.clear()
        self.destination.update(
            hidden=self.values,
            layers=np.array(self.layers),
            positions=np.array(self.names),
            token_positions=self.token_positions,
            available=np.isfinite(self.values).all(-1),
            closure_counts=self.closure_counts,
            decision_labels=self.label_strings,
            decision_label_ids=self.label_tokens,
            decision_label_logits=self.decision_logits,
            decision_label_probabilities_temperature_1=self.decision_probabilities,
        )
        return False
