"""Explicit Qwen batch backend; independent RNG streams, fresh caches, audited cohorts."""

from contextlib import contextmanager, nullcontext
import time
from .artifacts import digest
from .model import HFModel

BACKEND = "torch_batched_v1"


class BatchedHFModel(HFModel):
    def metadata(self):
        return {**super().metadata(), "execution_backend": BACKEND, "logits_to_keep": 1}

    @contextmanager
    def batch_hook(self, interventions, captures):
        active = [(i, value) for i, value in enumerate(interventions) if value is not None]
        if not active:
            yield
            return
        layers = {value.layer for _, value in active}
        if len(layers) != 1:
            raise ValueError("A batch must use a common intervention layer")
        torch = self.torch

        def hook(module, args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            changed = tensor.clone()
            for row, intervention in active:
                capture = captures[row]
                capture["before"] = tensor[row, -1].float().detach().cpu().tolist()
                if intervention.dose:
                    direction = torch.as_tensor(
                        intervention.direction, device=tensor.device, dtype=tensor.dtype
                    )
                    if direction.shape != tensor.shape[-1:] or not torch.isfinite(direction).all():
                        raise ValueError("Invalid direction")
                    changed[row, -1] += intervention.dose * direction
                capture["after"] = changed[row, -1].float().detach().cpu().tolist()
                capture["calls"] = capture.get("calls", 0) + 1
            return (changed, *output[1:]) if isinstance(output, tuple) else changed

        handle = self.layers[next(iter(layers))].register_forward_hook(hook)
        self.active_hooks += 1
        try:
            yield
        finally:
            handle.remove()
            self.active_hooks -= 1

    def infer_batch(self, requests, cohort_ids, capture_plan=None, capture_output=None):
        torch = self.torch
        if self.active_hooks:
            raise RuntimeError("A hook leaked into this batch")
        if not requests or len(requests) != len(cohort_ids) or len(set(cohort_ids)) != len(cohort_ids):
            raise ValueError("Batch requires unique cohort IDs for every request")
        if self.model.config.model_type != "qwen3":
            raise ValueError("This backend is validated for Qwen3 only")
        rendered, ids = [], []
        for request in requests:
            text, inputs = self.encode(request["messages"], request.get("enable_thinking", False))
            rendered.append(text)
            ids.append(inputs.input_ids[0].tolist())
        width = max(map(len, ids))
        pad = self.tokenizer.pad_token_id
        if pad is None:
            raise ValueError("Explicit padding token required")
        input_ids = torch.tensor([[pad] * (width - len(row)) + row for row in ids], device=self.device)
        mask = torch.tensor([[0] * (width - len(row)) + [1] * len(row) for row in ids], device=self.device)
        positions = (mask.cumsum(-1) - 1).clamp(min=0)
        saved_mask, saved_positions = mask.tolist(), positions.tolist()
        generators = [torch.Generator(device=self.device).manual_seed(r["seed"]) for r in requests]
        limits = [r.get("max_new_tokens", 8) for r in requests]
        if any(type(limit) is not int or limit < 1 for limit in limits):
            raise ValueError("Invalid generation limit")
        interventions = [r.get("intervention") for r in requests]
        captures = [{} for _ in requests]
        generated = [[] for _ in requests]
        finished = [False] * len(requests)
        finished_seconds = [None] * len(requests)
        eos_ids = self.model.generation_config.eos_token_id
        eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])
        tracer = None
        if capture_plan is not None:
            from .reasoning_trace import ReasoningTrace

            tracer = ReasoningTrace(
                self,
                rendered,
                ids,
                width,
                capture_plan,
                capture_output,
                [r.get("labels", "ABCD") for r in requests],
            )
        started = time.monotonic()
        with torch.inference_mode(), tracer if tracer is not None else nullcontext():
            with self.batch_hook(interventions, captures):
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=mask,
                    position_ids=positions,
                    use_cache=True,
                    logits_to_keep=1,
                )
            logits = output.logits[:, -1].float()
            first_probabilities = logits.softmax(-1).cpu()
            probabilities = []
            for i, request in enumerate(requests):
                tokens = {
                    label: self.tokenizer.encode(label, add_special_tokens=False)
                    for label in request.get("labels", "ABCD")
                }
                if any(len(value) != 1 for value in tokens.values()):
                    raise ValueError("Action scoring requires single-token labels")
                probabilities.append(
                    {label: first_probabilities[i, value[0]].item() for label, value in tokens.items()}
                )
            cache = output.past_key_values
            for step in range(max(limits)):
                next_tokens = [pad] * len(requests)
                for row, request in enumerate(requests):
                    if finished[row]:
                        continue
                    token = (
                        int(logits[row].argmax())
                        if request.get("greedy", False)
                        else int(
                            torch.multinomial((logits[row] / 0.7).softmax(-1), 1, generator=generators[row])
                        )
                    )
                    generated[row].append(token)
                    if tracer is not None:
                        tracer.after_sample(row, token, len(generated[row]), eos_ids, logits[row])
                    next_tokens[row] = token
                    if token in eos_ids or len(generated[row]) == limits[row]:
                        finished[row] = True
                        finished_seconds[row] = time.monotonic() - started
                if all(finished):
                    break
                # Terminal rows stay in the cohort but their dummy cache positions are masked.
                mask = torch.cat(
                    [mask, torch.tensor([[int(not value)] for value in finished], device=self.device)], dim=1
                )
                position = (mask.sum(-1) - 1).clamp(min=0).unsqueeze(-1)
                if tracer is not None:
                    tracer.before_decode(next_tokens, generated, finished)
                output = self.model(
                    input_ids=torch.tensor(next_tokens, device=self.device).unsqueeze(-1),
                    attention_mask=mask,
                    position_ids=position,
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
                logits, cache = output.logits[:, -1].float(), output.past_key_values
        batch_seconds = time.monotonic() - started
        batch_id = digest({"cohort": cohort_ids, "prompts": rendered, "seeds": [r["seed"] for r in requests]})
        results = []
        for row, request in enumerate(requests):
            intervention, probs = interventions[row], probabilities[row]
            mass = sum(probs.values())
            results.append(
                {
                    "raw": self.tokenizer.decode(generated[row], skip_special_tokens=True),
                    "enable_thinking": request.get("enable_thinking", False),
                    "ended_with_eos": generated[row][-1] in eos_ids,
                    "generated_token_ids": generated[row],
                    "messages": request["messages"],
                    "rendered_prompt": rendered[row],
                    "prompt_sha256": digest(rendered[row]),
                    "input_ids": input_ids[row].tolist(),
                    "unpadded_input_ids": ids[row],
                    "attention_mask": saved_mask[row],
                    "position_ids": saved_positions[row],
                    "label_probabilities_temperature_1": probs,
                    "other_token_probability": max(0.0, 1 - mass),
                    "conditional_label_probabilities": {label: p / mass for label, p in probs.items()}
                    if mass > 0
                    else None,
                    "sampling": {
                        "seed": request["seed"],
                        "temperature": 0 if request.get("greedy") else 0.7,
                        "top_p": 1.0,
                        "max_new_tokens": limits[row],
                    },
                    "seconds": finished_seconds[row],
                    "capture": captures[row],
                    "intervention": None
                    if intervention is None
                    else {
                        "layer": intervention.layer,
                        "dose": intervention.dose,
                        "direction_sha256": digest(intervention.direction)
                        if intervention.direction is not None
                        else None,
                        "position": "last_prompt_token_only",
                    },
                    "hooks_remaining": self.active_hooks,
                    "batch": {
                        "backend": BACKEND,
                        "id": batch_id,
                        "cohort_ids": cohort_ids,
                        "row": row,
                        "size": len(requests),
                        "wall_seconds": batch_seconds,
                        "latency_semantics": "seconds is shared-batch latency, not additive compute",
                    },
                }
            )
        return results


def execute_batches(root, manifest, model, checkpoint=None, deadline=None):
    """Fixed assignment cohorts; resumed cohorts retain all completed terminal records."""
    from pathlib import Path
    import uuid
    from .artifacts import RunStore, write_once

    if manifest.get("assay") == "information_diagnostic":
        from .information_diagnostic import inference_request, record_from_inference
    elif manifest.get("assay") == "information_state":
        from .information_state import inference_request, record_from_inference
    elif manifest.get("assay") == "single_curiosity":
        from .single_curiosity import inference_request, record_from_inference
    elif manifest.get("assay") == "one_credit":
        from .credit import inference_request, record_from_inference
    else:
        raise ValueError("Unsupported batched assay")

    batch_size = manifest.get("execution", {}).get("batch_size")
    allowed_sizes = (
        (8, 16, 32) if manifest.get("assay") in ("information_diagnostic", "information_state") else (32,)
    )
    if batch_size not in allowed_sizes or manifest.get("execution") != {
        "backend": BACKEND,
        "batch_size": batch_size,
    }:
        raise ValueError("Unsupported execution plan")
    if not isinstance(model, BatchedHFModel):
        raise ValueError("Batch manifest requires the explicit batch backend")
    assignments = manifest["assignments"] + manifest["diagnostics"]
    if any(
        a.get("assay") != manifest["assay"] or a.get("execution") != manifest["execution"]
        for a in assignments
    ):
        raise ValueError("Assignment/backend mismatch")
    store = RunStore(root, manifest)
    completed = store.completed()
    attempt = uuid.uuid4().hex
    write_once(Path(root) / "environments" / f"{attempt}.json", model.metadata())
    started = time.monotonic()
    for offset in range(0, len(assignments), batch_size):
        cohort = [a for a in assignments[offset : offset + batch_size] if a["trial_id"] not in completed]
        if not cohort:
            continue
        if deadline is not None and time.time() >= deadline:
            break
        try:
            inferences = model.infer_batch(
                [inference_request(a) for a in cohort], [a["trial_id"] for a in cohort]
            )
            records = [record_from_inference(a, inference) for a, inference in zip(cohort, inferences)]
            if len(records) != len(cohort):
                raise RuntimeError("Incomplete batch return")
        except Exception as error:
            for assignment in cohort:
                store.save(
                    {
                        "trial_id": assignment["trial_id"],
                        "assignment": assignment,
                        "kind": assignment.get("kind", "behavior"),
                        "terminal": True,
                        "failure": "execution_error",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "first_service": "technical_failure",
                    },
                    attempt,
                )
            if checkpoint:
                checkpoint()
            raise
        for record in records:
            store.save(record, attempt)
            completed[record["trial_id"]] = record
            if checkpoint:
                checkpoint()
        print(
            f"{len(completed)}/{len(assignments)} batch_size={len(cohort)} "
            f"wall_seconds={inferences[0]['batch']['wall_seconds']:.2f}",
            flush=True,
        )
    write_once(
        Path(root) / "profiles" / f"{attempt}.json",
        {
            "attempt": attempt,
            "elapsed_seconds": time.monotonic() - started,
            "completed": len(completed),
            "assigned": len(assignments),
            "execution": manifest["execution"],
        },
    )
    if checkpoint:
        checkpoint()
    return store.completed()


def verify_batch_model(model):
    """Real-model checks at fixed shapes; does not demand cross-shape numerical identity."""
    from .model import Intervention

    requests = [
        {
            "messages": [{"role": "user", "content": "Choose one letter: A, B, C, or D."}],
            "seed": 71,
            "greedy": True,
            "max_new_tokens": 8,
        },
        {
            "messages": [{"role": "user", "content": "Reply with A."}],
            "seed": 72,
            "greedy": False,
            "max_new_tokens": 8,
        },
    ]
    ids = ["integrity-long", "integrity-short"]
    layer = len(model.layers) // 2
    baseline = model.infer_batch(requests, ids)
    sham = model.infer_batch([{**r, "intervention": Intervention(layer)} for r in requests], ids)
    direction = [0.0] * model.model.config.hidden_size
    direction[0] = 1.0
    changed = model.infer_batch(
        [
            {**requests[0], "intervention": Intervention(layer, 1, direction)},
            {**requests[1], "intervention": Intervention(layer)},
        ],
        ids,
    )
    try:
        model.infer_batch([{**requests[0], "intervention": Intervention(layer, 1, [1.0])}, requests[1]], ids)
    except ValueError:
        pass
    else:
        raise AssertionError("Malformed batch direction was not rejected")
    fresh = model.infer_batch(requests, ids)
    checks = {
        "sham_tokens_equal": all(
            b["generated_token_ids"] == s["generated_token_ids"] for b, s in zip(baseline, sham)
        ),
        "sham_probabilities_equal": all(
            b["label_probabilities_temperature_1"] == s["label_probabilities_temperature_1"]
            for b, s in zip(baseline, sham)
        ),
        "zero_dose_capture_equal": all(s["capture"]["before"] == s["capture"]["after"] for s in sham),
        "one_hook_call_per_row": all(s["capture"]["calls"] == 1 for s in sham + changed),
        "target_activation_changed": changed[0]["capture"]["before"] != changed[0]["capture"]["after"],
        "other_row_unchanged": baseline[1]["generated_token_ids"] == changed[1]["generated_token_ids"]
        and baseline[1]["label_probabilities_temperature_1"]
        == changed[1]["label_probabilities_temperature_1"],
        "fresh_after_exception_equal": all(
            b["generated_token_ids"] == f["generated_token_ids"]
            and b["label_probabilities_temperature_1"] == f["label_probabilities_temperature_1"]
            for b, f in zip(baseline, fresh)
        ),
        "no_hooks_remaining": model.active_hooks == 0,
        "unequal_lengths_exercised": len(baseline[0]["unpadded_input_ids"])
        != len(baseline[1]["unpadded_input_ids"]),
        "left_padding_correct": all(
            [token for token, mask in zip(b["input_ids"], b["attention_mask"]) if mask]
            == b["unpadded_input_ids"]
            and [p for p, mask in zip(b["position_ids"], b["attention_mask"]) if mask]
            == list(range(len(b["unpadded_input_ids"])))
            for b in baseline
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "model": model.metadata(),
        "records": {"baseline": baseline, "sham": sham, "nonzero": changed, "fresh": fresh},
    }
