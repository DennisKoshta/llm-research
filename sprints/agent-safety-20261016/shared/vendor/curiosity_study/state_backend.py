"""Auditable prefill measurements and explicitly scheduled cached interventions."""

from contextlib import ExitStack
import hashlib
import math
import time
from .artifacts import digest

MARKER = "End of record."
# Prospective engineering change after mixed-prefix parity found five unequal
# duplicate pairs at the final rows of unaligned BF16 batches. Padding is masked
# and keeps real-token position IDs unchanged. Its efficacy requires a new
# large-model parity collection; alignment alone is not an integrity proof.
PREFILL_PADDING_MULTIPLE = 128
ALL_ATTENDED_SCHEDULE = "all_attended_prefill_plus_decode"


class StateBackend:
    def __init__(self, model):
        self.base = model

    def prepare(self, requests):
        model = self.base
        torch = model.torch
        rendered, ids, named_positions = [], [], []
        for request in requests:
            if "token_prefix" in request:
                row = list(request["token_prefix"])
                if not row or any(
                    type(t) is not int or not 0 <= t < model.model.config.vocab_size for t in row
                ):
                    raise ValueError("Invalid exact token prefix")
                supplied = dict(request["prefix_positions"])
                if "record" not in supplied or any(
                    type(p) is not int or not 0 <= p < len(row) for p in supplied.values()
                ):
                    raise ValueError("Invalid explicit prefix positions")
                if supplied.get("final", len(row) - 1) != len(row) - 1:
                    raise ValueError("Final position must be the last prefix token")
                supplied["final"] = len(row) - 1
                if request.get("token_prefix_sha256") != digest(row):
                    raise ValueError("Exact token prefix hash mismatch")
                rendered.append(model.tokenizer.decode(row, skip_special_tokens=False))
                ids.append(row)
                named_positions.append(supplied)
                continue
            text, inputs = model.encode(request["messages"], enable_thinking=False)
            if text.count(MARKER) != 1:
                raise ValueError("Exactly one record boundary is required")
            offset = text.index(MARKER) + len(MARKER) - 1
            tokenized = model.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            row = inputs.input_ids[0].tolist()
            if tokenized["input_ids"] != row:
                raise ValueError("Offset tokenizer differs from inference tokenizer")
            positions = [
                i for i, (start, end) in enumerate(tokenized["offset_mapping"]) if start <= offset < end
            ]
            if len(positions) != 1:
                raise ValueError("Ambiguous record boundary token")
            position = positions[0]
            start, end = tokenized["offset_mapping"][position]
            if text[offset + 1 : end].strip():
                raise ValueError("Record boundary token contains later question text")
            rendered.append(text)
            ids.append(row)
            named_positions.append({"record": position, "final": len(row) - 1})
        longest = max(map(len, ids))
        width = (
            (longest + PREFILL_PADDING_MULTIPLE - 1) // PREFILL_PADDING_MULTIPLE
        ) * PREFILL_PADDING_MULTIPLE
        pad = model.tokenizer.pad_token_id
        if pad is None:
            raise ValueError("Padding token required")
        tokens = torch.tensor([[pad] * (width - len(row)) + row for row in ids], device=model.device)
        masks = torch.tensor([[0] * (width - len(row)) + [1] * len(row) for row in ids], device=model.device)
        positions = (masks.cumsum(-1) - 1).clamp(min=0)
        names = set(named_positions[0])
        if any(set(p) != names for p in named_positions):
            raise ValueError("All rows must supply the same named positions")
        targets = {
            name: [width - len(row) + p[name] for row, p in zip(ids, named_positions)]
            for name in sorted(names)
        }
        return rendered, ids, dict(input_ids=tokens, attention_mask=masks, position_ids=positions), targets

    def forward(
        self,
        requests,
        layers=(),
        positions=("record", "final"),
        interventions=None,
        max_new_tokens=0,
        greedy=False,
        intervention_schedule="prefill_only",
        prefill_evidence_output=None,
        prefill_evidence_rows=(),
    ):
        model = self.base
        torch = model.torch
        if model.active_hooks:
            raise RuntimeError("Leaked hook at entry")
        if type(max_new_tokens) is not int or not 0 <= max_new_tokens <= 4096:
            raise ValueError("State continuations permit zero to 4096 completion tokens")
        if intervention_schedule not in ("prefill_only", "record_prefill_plus_decode", ALL_ATTENDED_SCHEDULE):
            raise ValueError("Unknown intervention schedule")
        if intervention_schedule != "prefill_only" and max_new_tokens == 0:
            raise ValueError("Sustained steering requires a positive generation allowance")
        full_prefix = intervention_schedule == ALL_ATTENDED_SCHEDULE
        if not full_prefix and (prefill_evidence_output is not None or prefill_evidence_rows):
            raise ValueError("Full-prefix evidence retention requires the explicit full-prefix schedule")
        rendered, ids, inputs, targets = self.prepare(requests)
        n = len(requests)
        interventions = interventions or [None] * n
        if len(interventions) != n or not all(p in targets for p in positions):
            raise ValueError("Invalid intervention/position plan")
        active_layers = set(layers) | {p["layer"] for p in interventions if p is not None}
        if any(type(layer) is not int or not 0 <= layer < len(model.layers) for layer in active_layers):
            raise ValueError("Invalid layer")
        sustained = (
            _SustainedDecode(
                model,
                interventions,
                ids,
                [targets["record"][i] - inputs["input_ids"].shape[1] + len(ids[i]) for i in range(n)],
            )
            if intervention_schedule == "record_prefill_plus_decode"
            else None
        )
        if full_prefix:
            sustained = _AllAttendedDecode(
                model, interventions, ids, inputs["attention_mask"],
                evidence_output=prefill_evidence_output, evidence_rows=prefill_evidence_rows,
            )
        captures = {}
        changes = [None] * n
        row_ids = torch.arange(n, device=model.device)
        started = time.monotonic()

        def make_hook(layer):
            def hook(module, args, output):
                tensor = output[0] if isinstance(output, tuple) else output
                if layer in layers:
                    captures[layer] = torch.stack(
                        [
                            tensor[row_ids, torch.tensor(targets[p], device=model.device)]
                            .detach()
                            .float()
                            .cpu()
                            for p in positions
                        ],
                        dim=1,
                    )
                active = [
                    (row, p) for row, p in enumerate(interventions) if p is not None and p["layer"] == layer
                ]
                if not active:
                    return output
                if full_prefix:
                    return sustained.patch_prefill(layer, tensor, output, changes)
                changed = tensor.clone()
                for row, patch in active:
                    pos = targets[patch["position"]][row]
                    before = tensor[row, pos].float()
                    reference = None
                    if patch["mode"] == "replace":
                        after = torch.as_tensor(patch["value"], device=model.device, dtype=torch.float32)
                    else:
                        direction = torch.as_tensor(
                            patch["direction"], device=model.device, dtype=torch.float32
                        )
                        if direction.shape != before.shape or not torch.isfinite(direction).all():
                            raise ValueError("Invalid patch direction")
                        if patch["mode"] == "add":
                            delta = float(patch["dose"])
                        elif patch["mode"] in ("project", "matched_add"):
                            if not torch.isclose(
                                direction.norm(), torch.tensor(1.0, device=model.device), atol=1e-4
                            ):
                                raise ValueError("Projection direction must have unit norm")
                            reference = direction
                            if patch["mode"] == "matched_add":
                                reference = torch.as_tensor(
                                    patch["reference_direction"], device=model.device, dtype=torch.float32
                                )
                                if (
                                    reference.shape != before.shape
                                    or not torch.isfinite(reference).all()
                                    or not torch.isclose(
                                        reference.norm(), torch.tensor(1.0, device=model.device), atol=1e-4
                                    )
                                ):
                                    raise ValueError("Matched reference direction must have unit norm")
                            delta = float(patch["target"]) - torch.dot(before, reference)
                        else:
                            raise ValueError("Unknown patch mode")
                        after = before + delta * direction
                    if after.shape != before.shape or not torch.isfinite(after).all():
                        raise ValueError("Invalid patched activation")
                    changed[row, pos] = after.to(tensor.dtype)
                    actual = changed[row, pos].float()
                    if sustained is not None:
                        sustained.prefill_dtypes[row] = str(tensor.dtype)
                    changes[row] = dict(
                        layer=layer,
                        position=patch["position"],
                        token_index=pos,
                        before_norm=before.norm().item(),
                        delta_norm=(actual - before).norm().item(),
                        after_norm=actual.norm().item(),
                        mode=patch["mode"],
                        requested_delta_norm=(after - before).norm().item(),
                        rounding_error_norm=(actual - after).norm().item(),
                    )
                    changes[row]["relative_delta_norm"] = (actual - before).norm().item() / max(
                        before.norm().item(), 1e-30
                    )
                    if patch["mode"] != "replace":
                        direction_norm = direction.norm().item()
                        unit = direction / direction_norm if direction_norm else torch.zeros_like(direction)
                        changes[row].update(
                            direction_norm=direction_norm,
                            projection_before=torch.dot(before, unit).item(),
                            projection_after=torch.dot(actual, unit).item(),
                            requested_scalar_delta=float(delta),
                        )
                    if reference is not None:
                        changes[row].update(
                            projection_target=float(patch["target"]),
                            reference_projection_before=torch.dot(before, reference).item(),
                            reference_projection_after=torch.dot(actual, reference).item(),
                        )
                    if sustained is not None and not all(
                        math.isfinite(changes[row][name]) for name in sustained.FIELDS
                    ):
                        raise ValueError("Nonfinite sustained prefill geometry")
                return (changed, *output[1:]) if isinstance(output, tuple) else changed

            return hook

        with torch.inference_mode(), ExitStack() as stack:
            if full_prefix:
                sustained.install_prefill_observers(stack)
            for layer in sorted(active_layers):
                handle = model.layers[layer].register_forward_hook(make_hook(layer))
                model.active_hooks += 1

                def remove(handle=handle):
                    handle.remove()
                    model.active_hooks -= 1

                stack.callback(remove)
            output = model.model(**inputs, use_cache=bool(max_new_tokens), logits_to_keep=1)
            logits = output.logits[:, -1].float()
            probabilities = logits.softmax(-1)
            rows = []
            for i, request in enumerate(requests):
                labels = request["labels"]
                label_ids = {
                    label: model.tokenizer.encode(label, add_special_tokens=False) for label in labels
                }
                if any(len(v) != 1 for v in label_ids.values()):
                    raise ValueError("Single-token response labels required")
                scores = {label: logits[i, v[0]].item() for label, v in label_ids.items()}
                probs = {label: probabilities[i, v[0]].item() for label, v in label_ids.items()}
                top = int(logits[i].argmax())
                rows.append(
                    dict(
                        prompt_sha256=digest(rendered[i]),
                        rendered_prompt=rendered[i],
                        input_ids=inputs["input_ids"][i].tolist(),
                        attention_mask=inputs["attention_mask"][i].tolist(),
                        position_ids=inputs["position_ids"][i].tolist(),
                        prefill_padding_multiple=PREFILL_PADDING_MULTIPLE,
                        prefill_width=int(inputs["input_ids"].shape[1]),
                        unpadded_input_ids=ids[i],
                        token_prefix_sha256=digest(ids[i]),
                        exact_token_replay="token_prefix" in request,
                        unpadded_token_positions={
                            p: values[i] - len(inputs["input_ids"][i]) + len(ids[i])
                            for p, values in targets.items()
                        },
                        record_token_index=targets["record"][i],
                        label_logits=scores,
                        label_probabilities_temperature_1=probs,
                        label_mass=sum(probs.values()),
                        greedy_first_token_id=top,
                        greedy_first_token=model.tokenizer.decode([top], skip_special_tokens=True),
                        intervention=changes[i],
                        measurement="fixed_token_prefix_next_token"
                        if "token_prefix" in request
                        else "single_prefill_first_token; not a completed answer or confidence measure",
                    )
                )
        if model.active_hooks:
            raise RuntimeError("Hook removal failed")
        if max_new_tokens:
            # Legacy prefill hooks are removed. The opt-in schedule installs
            # separate hooks for one-token decoding, never reusing prefix indices.
            generators = [torch.Generator(device=model.device).manual_seed(r["seed"]) for r in requests]
            eos = model.model.generation_config.eos_token_id
            eos = {eos} if isinstance(eos, int) else set(eos or [])
            cache, mask = output.past_key_values, inputs["attention_mask"]
            generated = [[] for _ in requests]
            finished = [False] * n
            with torch.inference_mode(), ExitStack() as decode_stack:
                if sustained is not None:
                    sustained.install(decode_stack)
                for step in range(max_new_tokens):
                    next_tokens = [model.tokenizer.pad_token_id] * n
                    for row in range(n):
                        if finished[row]:
                            continue
                        token = (
                            int(logits[row].argmax())
                            if greedy
                            else int(
                                torch.multinomial(
                                    (logits[row] / 0.7).softmax(-1), 1, generator=generators[row]
                                )
                            )
                        )
                        generated[row].append(token)
                        next_tokens[row] = token
                        finished[row] = token in eos or step + 1 == max_new_tokens
                    if all(finished):
                        break
                    mask = torch.cat(
                        [mask, torch.tensor([[int(not done)] for done in finished], device=model.device)],
                        dim=1,
                    )
                    if sustained is not None:
                        sustained.set_step(step + 1, finished)
                    decoded = model.model(
                        input_ids=torch.tensor(next_tokens, device=model.device).unsqueeze(-1),
                        attention_mask=mask,
                        position_ids=(mask.sum(-1) - 1).clamp(min=0).unsqueeze(-1),
                        past_key_values=cache,
                        use_cache=True,
                        logits_to_keep=1,
                    )
                    logits, cache = decoded.logits[:, -1].float(), decoded.past_key_values
            for row, result in enumerate(rows):
                result["completion"] = dict(
                    raw=model.tokenizer.decode(generated[row], skip_special_tokens=True),
                    generated_token_ids=generated[row],
                    ended_with_eos=generated[row][-1] in eos,
                    sampling=dict(
                        seed=requests[row]["seed"],
                        temperature=0 if greedy else 0.7,
                        top_p=1.0,
                        max_new_tokens=max_new_tokens,
                    ),
                    intervention_schedule=(
                        "prefill_only; no active hooks during decoding"
                        if sustained is None
                        else "all_attended_prefill_plus_decode; all attended initial tokens once, then current generated token for unfinished rows"
                        if full_prefix
                        else "record_prefill_plus_decode; record once, then current generated token for unfinished rows"
                    ),
                )
            if sustained is not None:
                traces = sustained.traces(changes, generated)
                for row, trace in zip(rows, traces):
                    row["intervention_trace"] = trace
            if full_prefix:
                for index, row in enumerate(rows):
                    row["prefill_geometry_observations"] = sustained.observations[index]
        if model.active_hooks:
            raise RuntimeError("Hook removal failed after decoding")
        for r in rows:
            r["batch_seconds"] = time.monotonic() - started
            r["hooks_remaining"] = model.active_hooks
        array = torch.stack([captures[layer] for layer in layers], dim=1).numpy() if layers else None
        return rows, array


class _SustainedDecode:
    """Opt-in additive decoding hooks; scalar geometry is transferred once at exit."""

    FIELDS = (
        "before_norm",
        "delta_norm",
        "after_norm",
        "requested_delta_norm",
        "rounding_error_norm",
        "relative_delta_norm",
        "direction_norm",
        "projection_before",
        "projection_after",
        "requested_scalar_delta",
    )

    def __init__(self, model, patches, prefixes, record_positions):
        self.model, self.patches, self.prefixes = model, patches, prefixes
        self.record_positions = record_positions
        self.prefill_dtypes = [None] * len(patches)
        self.decode_counts = [0] * len(patches)
        self.by_layer, self.geometry = {}, []
        self.step, self.finished = None, None
        torch = model.torch
        for row, patch in enumerate(patches):
            if patch is None:
                continue
            if patch.get("mode") != "add" or patch.get("position") != "record":
                raise ValueError("Sustained steering permits only additive record-prefill patches")
            direction = torch.as_tensor(patch["direction"], device=model.device, dtype=torch.float32)
            dose = float(patch["dose"])
            if (
                direction.shape != (model.model.config.hidden_size,)
                or not torch.isfinite(direction).all()
                or not torch.isfinite(direction.norm())
                or not math.isfinite(dose)
            ):
                raise ValueError("Invalid sustained direction or dose")
            self.by_layer.setdefault(patch["layer"], []).append((row, direction, dose))

    def install(self, stack):
        for layer in sorted(self.by_layer):
            handle = self.model.layers[layer].register_forward_hook(self.make_hook(layer))
            self.model.active_hooks += 1

            def remove(handle=handle):
                handle.remove()
                self.model.active_hooks -= 1

            stack.callback(remove)

    def set_step(self, step, finished):
        self.step, self.finished = step, tuple(finished)
        for row, done in enumerate(finished):
            self.decode_counts[row] += int(not done)

    def make_hook(self, layer):
        def hook(module, args, output):
            torch = self.model.torch
            tensor = output[0] if isinstance(output, tuple) else output
            if self.step is None or tensor.ndim != 3 or tensor.shape[1] != 1:
                raise ValueError("Sustained decode hook requires one processed generated token")
            active = [entry for entry in self.by_layer[layer] if not self.finished[entry[0]]]
            if not active:
                return output
            rows = [entry[0] for entry in active]
            row_ids = torch.tensor(rows, device=tensor.device)
            directions = torch.stack([entry[1] for entry in active])
            doses = torch.tensor([entry[2] for entry in active], device=tensor.device, dtype=torch.float32)
            before = tensor[row_ids, 0].float()
            after = before + doses[:, None] * directions
            if not torch.isfinite(after).all():
                raise ValueError("Nonfinite sustained activation")
            changed = tensor.clone()
            changed[row_ids, 0] = after.to(tensor.dtype)
            actual = changed[row_ids, 0].float()
            norm = directions.norm(dim=-1)
            unit = torch.where(
                norm[:, None] > 0,
                directions / norm.clamp(min=1e-30)[:, None],
                torch.zeros_like(directions),
            )
            before_norm = before.norm(dim=-1)
            delta_norm = (actual - before).norm(dim=-1)
            values = torch.stack(
                [
                    before_norm,
                    delta_norm,
                    actual.norm(dim=-1),
                    (after - before).norm(dim=-1),
                    (actual - after).norm(dim=-1),
                    delta_norm / before_norm.clamp(min=1e-30),
                    norm,
                    (before * unit).sum(-1),
                    (actual * unit).sum(-1),
                    doses,
                ],
                dim=-1,
            )
            if not torch.isfinite(values).all():
                raise ValueError("Nonfinite sustained geometry")
            self.geometry.append((self.step, layer, rows, str(tensor.dtype), values))
            return (changed, *output[1:]) if isinstance(output, tuple) else changed

        return hook

    def traces(self, prefill_changes, generated):
        traces = []
        for row, (change, tokens) in enumerate(zip(prefill_changes, generated)):
            if change is not None and not all(math.isfinite(change[name]) for name in self.FIELDS):
                raise ValueError("Nonfinite sustained prefill geometry")
            events = (
                []
                if change is None
                else [
                    dict(
                        **change,
                        phase="prefill",
                        processed_generated_token_index=None,
                        unpadded_token_index=self.record_positions[row],
                        processed_token_id=self.prefixes[row][self.record_positions[row]],
                        activation_dtype=self.prefill_dtypes[row],
                        arithmetic_scalar_float32=self.model.torch.tensor(
                            self.patches[row]["dose"], dtype=self.model.torch.float32
                        ).item(),
                    )
                ]
            )
            traces.append(
                dict(
                    schema_version=1,
                    schedule="record_prefill_plus_decode",
                    events=events,
                    generated_token_count=len(tokens),
                    active_decode_forward_count=self.decode_counts[row],
                    geometry_reference="Before each patch is the current treated trajectory, not a baseline counterfactual.",
                    cumulative_norm_semantics="Sum of achieved per-injection norms; not net trajectory displacement.",
                )
            )
        matrices = (
            self.model.torch.cat([item[-1] for item in self.geometry]).cpu().tolist() if self.geometry else []
        )
        offset = 0
        for step, layer, rows, dtype, _ in self.geometry:
            for row in rows:
                geometry = dict(zip(self.FIELDS, matrices[offset]))
                geometry["arithmetic_scalar_float32"] = geometry["requested_scalar_delta"]
                geometry["requested_scalar_delta"] = float(self.patches[row]["dose"])
                traces[row]["events"].append(
                    dict(
                        **geometry,
                        phase="decode",
                        layer=layer,
                        position="current_generated_token",
                        token_index=0,
                        processed_generated_token_index=step,
                        processed_token_id=generated[row][step - 1],
                        unpadded_token_index=len(self.prefixes[row]) + step - 1,
                        mode="add",
                        activation_dtype=dtype,
                    )
                )
                offset += 1
        for trace in traces:
            trace.update(
                prefill_patch_count=sum(event["phase"] == "prefill" for event in trace["events"]),
                decode_patch_count=sum(event["phase"] == "decode" for event in trace["events"]),
                total_patch_count=len(trace["events"]),
                cumulative_achieved_delta_norm=sum(event["delta_norm"] for event in trace["events"]),
            )
        return traces


class _AllAttendedDecode(_SustainedDecode):
    """Explicit prefix coverage; the inherited cached decode path is unchanged.

    The two observers surround the mutation through separate module boundaries.
    Only requested rows' raw vectors are retained beyond prefill; all positions
    receive online arithmetic/identity checks regardless of retention.
    """

    def __init__(self, model, patches, prefixes, mask, *, evidence_output, evidence_rows):
        if any(p is not None and (p.get("mode") != "add" or p.get("position") != "all_attended")
               for p in patches):
            raise ValueError("Full-prefix steering requires additive all_attended patches")
        if (not isinstance(evidence_rows, (tuple, list))
                or any(type(row) is not int or not 0 <= row < len(patches) for row in evidence_rows)
                or len(set(evidence_rows)) != len(evidence_rows)
                or (evidence_output is not None and (not isinstance(evidence_output, dict) or evidence_output))
                or (evidence_rows and evidence_output is None)):
            raise ValueError("Require unique retained rows and an empty evidence-output dictionary")
        # Reuse the exact original decode arithmetic, RNG-independent hooks and
        # event construction. The old class still accepts only record patches.
        compatible = [None if p is None else {**p, "position": "record"} for p in patches]
        super().__init__(model, compatible, prefixes, [0] * len(prefixes))
        self.patches = patches
        if any(layer + 1 >= len(model.layers) for layer in self.by_layer):
            raise ValueError("Full-prefix evidence requires a next transformer block")
        self.mask = mask
        self.evidence_output, self.evidence_rows = evidence_output, set(evidence_rows)
        self.prefill_geometry, self.pending = [], {}
        self.observations = [[] for _ in patches]

    def install_prefill_observers(self, stack):
        for layer in sorted(self.by_layer):
            for register, observer in (
                (self.model.layers[layer].register_forward_hook, self.before_observer(layer)),
                (self.model.layers[layer + 1].register_forward_pre_hook, self.after_observer(layer)),
            ):
                handle = register(observer)
                self.model.active_hooks += 1

                def remove(handle=handle):
                    handle.remove()
                    self.model.active_hooks -= 1

                stack.callback(remove)

    def before_observer(self, layer):
        def observe(module, args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if (layer in self.pending or tensor.ndim != 3 or tensor.shape[:2] != self.mask.shape
                    or tensor.dtype != self.model.torch.bfloat16):
                raise ValueError("Require one BF16 initial-prefill observation with the exact mask")
            self.pending[layer] = tensor.detach().clone()
        return observe

    def _binding(self, tensor):
        raw = tensor.detach().contiguous().cpu().view(self.model.torch.uint16).numpy()
        return dict(dtype="torch.bfloat16", shape=list(tensor.shape),
                    byte_sha256=hashlib.sha256(raw.tobytes()).hexdigest())

    def after_observer(self, layer):
        def observe(module, args):
            torch = self.model.torch
            before, actual = self.pending.pop(layer), args[0]
            if actual.shape != before.shape or actual.dtype != before.dtype:
                raise ValueError("Next-block input differs from the observed BF16 prefill layout")
            entries = {row: (direction, dose) for row, direction, dose in self.by_layer[layer]}
            for row in range(len(self.patches)):
                selected = self.mask[row].bool() if row in entries else torch.zeros_like(self.mask[row], dtype=torch.bool)
                a, b = before[row], actual[row]
                expected = a[selected]
                if row in entries:
                    direction, dose = entries[row]
                    expected = (expected.float() + dose * direction).to(a.dtype)
                checks = dict(
                    finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
                    selected_exact_bfloat16=torch.equal(b[selected].view(torch.uint16), expected.view(torch.uint16)),
                    nonselected_bitwise_unchanged=torch.equal(a[~selected].view(torch.uint16), b[~selected].view(torch.uint16)),
                )
                retained = row in self.evidence_rows
                names = dict(before=f"layer_{layer}_row_{row}_before", after=f"layer_{layer}_row_{row}_after")
                if retained:
                    self.evidence_output[names["before"]] = a.detach().float().cpu().numpy().copy()
                    self.evidence_output[names["after"]] = b.detach().float().cpu().numpy().copy()
                self.observations[row].append(dict(
                    layer=layer, next_layer=layer + 1, row=row,
                    observer="layer_output_before_mutation_to_next_layer_input",
                    initial_attention_mask_sha256=digest(self.mask[row].tolist()),
                    selected_token_indices=selected.nonzero(as_tuple=True)[0].tolist(),
                    selected_token_count=int(selected.sum()),
                    before=self._binding(a), after=self._binding(b), checks=checks,
                    passed=all(checks.values()), retained_arrays=names if retained else None,
                    retained_array_dtype="float32_exact_bfloat16_values" if retained else None,
                ))
        return observe

    def patch_prefill(self, layer, tensor, output, changes):
        torch = self.model.torch
        changed = tensor.clone()
        for row, direction, dose in self.by_layer[layer]:
            indices = self.mask[row].bool().nonzero(as_tuple=True)[0]
            before = tensor[row, indices].float()
            # Same per-token expression and cast as the original record path.
            after = before + dose * direction
            if not torch.isfinite(after).all():
                raise ValueError("Nonfinite full-prefix activation")
            changed[row, indices] = after.to(tensor.dtype)
            actual = changed[row, indices].float()
            norm = direction.norm()
            unit = direction / norm if norm else torch.zeros_like(direction)
            before_norm = before.norm(dim=-1)
            delta_norm = (actual - before).norm(dim=-1)
            values = torch.stack([
                before_norm, delta_norm, actual.norm(dim=-1), (after - before).norm(dim=-1),
                (actual - after).norm(dim=-1), delta_norm / before_norm.clamp(min=1e-30),
                norm.expand(len(indices)), (before * unit).sum(-1), (actual * unit).sum(-1),
                torch.full_like(before_norm, dose),
            ], dim=-1)
            if not torch.isfinite(values).all():
                raise ValueError("Nonfinite full-prefix geometry")
            self.prefill_geometry.append((row, layer, indices.tolist(), str(tensor.dtype), values))
            changes[row] = dict(
                layer=layer, position="all_attended", mode="add", token_indices=indices.tolist(),
                prefill_token_patch_count=len(indices), requested_scalar_delta=dose,
                arithmetic_scalar_float32=torch.tensor(dose, dtype=torch.float32).item(),
                activation_dtype=str(tensor.dtype),
            )
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    def traces(self, prefill_changes, generated):
        if self.pending or any(len(rows) != len(self.by_layer) for rows in self.observations):
            raise ValueError("Incomplete independent prefill observations")
        # Preserve the old cached-event construction exactly; replace only its
        # one-position prefill schema with the explicit multi-token inventory.
        traces = super().traces([None] * len(self.patches), generated)
        prefill_events = [[] for _ in self.patches]
        for row, layer, indices, dtype, values in self.prefill_geometry:
            pad = self.mask.shape[1] - len(self.prefixes[row])
            for index, value in zip(indices, values.cpu().tolist()):
                event = dict(zip(self.FIELDS, value))
                event["arithmetic_scalar_float32"] = event["requested_scalar_delta"]
                event["requested_scalar_delta"] = float(self.patches[row]["dose"])
                prefill_events[row].append(dict(
                    **event, phase="prefill", layer=layer, position="all_attended", mode="add",
                    token_index=index, unpadded_token_index=index - pad,
                    processed_generated_token_index=None, processed_token_id=self.prefixes[row][index - pad],
                    activation_dtype=dtype,
                ))
        for row, trace in enumerate(traces):
            events = prefill_events[row] + trace["events"]
            exposure = {}
            for phase in ("prefill", "decode"):
                selected = [event for event in events if event["phase"] == phase]
                exposure[phase] = dict(
                    token_patch_count=len(selected),
                    achieved_norm_sum=math.fsum(event["delta_norm"] for event in selected),
                    achieved_norm_root_sum_squares=math.sqrt(math.fsum(event["delta_norm"] ** 2 for event in selected)),
                    requested_norm_sum=math.fsum(event["requested_delta_norm"] for event in selected),
                    requested_norm_root_sum_squares=math.sqrt(math.fsum(event["requested_delta_norm"] ** 2 for event in selected)),
                )
            trace.update(
                schema_version=2, schedule=ALL_ATTENDED_SCHEDULE, events=events,
                initial_attended_token_count=len(self.prefixes[row]),
                prefill_forward_patch_count=int(self.patches[row] is not None),
                prefill_token_patch_count=len(prefill_events[row]), prefill_patch_count=len(prefill_events[row]),
                total_patch_count=len(events), exposure=exposure,
                exposure_semantics="Per-token norm sum and root-sum-squares by phase; not net trajectory displacement. Decode duration may depend on treatment.",
                cumulative_achieved_delta_norm=math.fsum(event["delta_norm"] for event in events),
                independent_prefill_observations_passed=all(item["passed"] for item in self.observations[row]),
            )
        return traces


def verify_state_backend(model, requests):
    backend = StateBackend(model)
    selected = requests[:2]
    layer = len(model.layers) // 2
    baseline, captures = backend.forward(selected, [layer])
    direction = [0.0] * model.model.config.hidden_size
    direction[0] = 1.0
    sham = [dict(layer=layer, position="final", mode="add", direction=direction, dose=0.0) for _ in selected]
    zero, _ = backend.forward(selected, interventions=sham)
    changed, _ = backend.forward(selected, interventions=[{**sham[0], "dose": 1.0}, None])
    try:
        backend.forward(selected, interventions=[{**sham[0], "direction": [1.0]}, None])
    except ValueError:
        pass
    else:
        raise AssertionError("Malformed direction accepted")
    fresh, _ = backend.forward(selected)
    checks = dict(
        sham_logits_equal=all(a["label_logits"] == b["label_logits"] for a, b in zip(baseline, zero)),
        sham_delta_zero=all(r["intervention"]["delta_norm"] == 0 for r in zero),
        perturbation_applied=changed[0]["intervention"]["delta_norm"] > 0,
        other_row_unchanged=baseline[1]["label_logits"] == changed[1]["label_logits"],
        fresh_after_exception_equal=all(
            a["label_logits"] == b["label_logits"] for a, b in zip(baseline, fresh)
        ),
        hook_cleanup=model.active_hooks == 0,
        capture_shape=list(captures.shape) == [2, 1, 2, model.model.config.hidden_size],
        positions_before_readout=all(r["record_token_index"] < len(r["input_ids"]) - 1 for r in baseline),
    )
    return dict(passed=all(checks.values()), checks=checks, baseline=baseline)
