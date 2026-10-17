"""Fixed16 coverage engineering calls; no cloud or scientific admission.

Elementwise arithmetic is independently observed at the next block. Saved row0
vectors permit offline reconstruction; other rows retain online checks and byte
bindings. Norms describe exposure, not displacement or psychological strength.
"""

from contextlib import ExitStack
from copy import deepcopy
import hashlib
from pathlib import Path
import time

import numpy as np

from . import voluntary_capture as receipts
from . import voluntary_resolution as voluntary
from .artifacts import digest, write_once
from .state_backend import ALL_ATTENDED_SCHEDULE, PREFILL_PADDING_MULTIPLE, StateBackend
from .sustained_engineering import array_binding, hook_inventory, identical_rows, require

REVISION = "curiosity-coverage-engineering-v1"
RECORD_SCHEDULE = "record_prefill_plus_decode"
LAYER, ROWS = 15, 32
DOSE = 29.78968343510005
DECODE_SAMPLES = (1, 2, 32, 128)


class ControlledCoverageFault(RuntimeError):
    pass


def call_plan(requests_by_cohort):
    require(set(requests_by_cohort) == {0, 3}, "Require only original source cohorts0 and3")
    for requests in requests_by_cohort.values():
        require(len(requests) == ROWS and len({r["source_trial_id"] for r in requests}) == ROWS,
                "Require all32 original companions")
        for r in requests:
            require(r["token_prefix_sha256"] == digest(r["token_prefix"])
                    and type(r["seed"]) is int and r["prefix_positions"]["final"] == len(r["token_prefix"]) - 1,
                    "Invalid exact source request")
    specs = [
        (0, 128, "baseline", None, None), (0, 128, "record_zero", None, None),
        (0, 128, "full_zero", None, None), (0, 128, "full_one_row", 0, None),
        (0, 128, "record_nonzero", None, None), (0, 128, "full_nonzero", None, None),
        (0, 128, "full_fault_prefill", None, "after_prefill"),
        (0, 128, "clean_repeat_prefill", None, None),
        (0, 128, "full_fault_decode", None, "after_first_decode"),
        (0, 128, "clean_repeat_decode", None, None),
    ] + [(cohort, 4096, condition, None, None) for cohort in (0, 3)
         for condition in ("baseline", "record_nonzero", "full_nonzero")]
    result = []
    for index, (cohort, cap, condition, one_row, fault) in enumerate(specs):
        full = condition.startswith("full_")
        baseline = condition == "baseline" or condition.startswith("clean_repeat_")
        call_id = f"coverage-{index:02d}-{condition}"
        requests = requests_by_cohort[cohort]
        result.append(dict(call_index=index, call_id=call_id, source_cohort=cohort, cap=cap,
            stage="checks" if index < 10 else "timing", condition=condition,
            schedule=ALL_ATTENDED_SCHEDULE if full else RECORD_SCHEDULE,
            position=None if baseline else "all_attended" if full else "record",
            direction_key=None if baseline else "curiosity_negative",
            dose_multiplier=0 if condition.endswith("_zero") else 1,
            one_row_only=one_row, inject_failure_at=fault,
            expected_error="ControlledCoverageFault" if fault else None,
            base_trial_ids=[r["source_trial_id"] for r in requests],
            trial_ids=[f"{call_id}:{row:02d}" for row in range(ROWS)],
            requests_sha256=digest(requests)))
    return result


def patches_for(call, direction, dose=DOSE):
    require(type(dose) in (int, float) and dose == DOSE, "Require the original fixed C contrast norm")
    if call["position"] is None:
        return [None] * ROWS
    return [dict(layer=LAYER, position=call["position"], mode="add", direction=direction,
                 dose=dose * call["dose_multiplier"] if call["one_row_only"] in (None, row) else 0.0)
            for row in range(ROWS)]


def _bf16_binding(raw):
    raw = np.ascontiguousarray(raw)
    return dict(dtype="torch.bfloat16", shape=list(raw.shape),
                byte_sha256=hashlib.sha256(raw.tobytes()).hexdigest())


def collect_call(model, requests, call, direction, *, dose=DOSE, deadline=None):
    """One fixed call, retaining actual partial evidence on every exception."""
    require(len(requests) == ROWS and call["requests_sha256"] == digest(requests), "Changed call requests")
    require(model.active_hooks == 0 and not hook_inventory(model), "Require clean hook-free entry")
    require(len(model.layers) > LAYER + 1, "Independent next-block observation is unavailable")
    torch, backend = model.torch, StateBackend(model)
    unit = torch.as_tensor(direction, device=model.device, dtype=torch.float32)
    require(unit.shape == (model.model.config.hidden_size,) and bool(torch.isfinite(unit).all())
            and abs(float(unit.norm()) - 1) < 1e-4, "Require a finite fixed unit direction")
    patches = patches_for(call, direction, dose)
    _, _, prepared, targets = backend.prepare(requests)
    initial_mask = prepared["attention_mask"].cpu().numpy().astype(bool)
    direction_cpu = np.asarray(direction, dtype=np.float32)
    scalars = np.asarray([0.0 if patch is None else patch["dose"] for patch in patches], dtype=np.float32)
    frames, samples, arrays, pending, held = [], [], {}, {}, {}
    attempted, injected = 0, False
    initial_hooks = hook_inventory(model)
    started = time.monotonic()

    def pre_model(module, args, kwargs):
        nonlocal attempted
        if deadline is not None and time.time() >= deadline:
            raise TimeoutError("Coverage deadline before next forward")
        require(not pending, "Nested or unfinished model forward")
        attempted += 1
        mask = kwargs["attention_mask"].detach().cpu().tolist()
        cache = kwargs.get("past_key_values")
        pending.update(forward_index=len(frames), phase="prefill" if not frames else "decode",
            prediction_ordinal=len(frames) + 1,
            input_ids=kwargs["input_ids"].detach().cpu().tolist(),
            position_ids=kwargs["position_ids"].detach().cpu().tolist(),
            attention_mask_shape=list(kwargs["attention_mask"].shape),
            attention_mask_row_sha256=[digest(row) for row in mask],
            active_prediction_rows=[bool(row[-1]) for row in mask],
            cache_length_before=0 if cache is None else int(cache.get_seq_length()),
            cache_position_supplied="cache_position" in kwargs)

    def before_patch(module, args, output):
        tensor = output[0] if isinstance(output, tuple) else output
        require(tensor.dtype == torch.bfloat16 and tensor.shape[0] == ROWS, "Require actual32 BF16 residuals")
        held["before"] = tensor.detach().clone()
        if pending["phase"] == "prefill":
            arrays["hidden"] = torch.stack([
                torch.stack([tensor[row, targets[p][row]] for p in ("record", "final")])
                for row in range(ROWS)]).float().cpu().numpy()[:, None]

    def after_patch(module, args):
        before, actual = held.pop("before"), args[0]
        require(actual.shape == before.shape and actual.dtype == torch.bfloat16, "Changed residual layout")
        step, phase = pending["forward_index"], pending["phase"]
        # Two batched transfers, not per-row GPU synchronizations. All subsequent
        # hashing and descriptive reductions run on these actual CPU observations.
        before_bits = before.detach().contiguous().cpu().view(torch.uint16).numpy()
        after_bits = actual.detach().contiguous().cpu().view(torch.uint16).numpy()
        before_cpu = (before_bits.astype(np.uint32) << 16).view(np.float32)
        after_cpu = (after_bits.astype(np.uint32) << 16).view(np.float32)
        selected = np.zeros(before_cpu.shape[:2], dtype=bool)
        for row, patch in enumerate(patches):
            if patch is not None:
                if phase == "prefill":
                    if call["position"] == "all_attended":
                        selected[row] = initial_mask[row]
                    else:
                        selected[row, targets["record"][row]] = True
                elif pending["active_prediction_rows"][row]:
                    selected[row, 0] = True
        expected_float = before_cpu + scalars[:, None, None] * direction_cpu
        expected_bits = torch.from_numpy(expected_float).bfloat16().view(torch.uint16).numpy()
        finite = np.isfinite(before_cpu).all(axis=(1, 2)) & np.isfinite(after_cpu).all(axis=(1, 2))
        exact = np.all((expected_bits == after_bits) | ~selected[:, :, None], axis=(1, 2))
        unchanged = np.all((before_bits == after_bits) | selected[:, :, None], axis=(1, 2))
        delta64 = after_cpu.astype(np.float64) - before_cpu.astype(np.float64)
        norms = dict(delta=np.linalg.norm(delta64, axis=-1),
                     dot=np.einsum("btd,d->bt", delta64, direction_cpu.astype(np.float64)),
                     before=np.linalg.norm(before_cpu.astype(np.float64), axis=-1),
                     after=np.linalg.norm(after_cpu.astype(np.float64), axis=-1))
        geometry = []
        for row, patch in enumerate(patches):
            scalar = 0.0 if patch is None else patch["dose"]
            indices = np.flatnonzero(selected[row]).tolist()
            checks = dict(finite=bool(finite[row]), exact_bfloat16=bool(exact[row]),
                          nonselected_bitwise_unchanged=bool(unchanged[row]))
            geometry.append(dict(row=row, selected_token_indices=indices, selected_token_count=len(indices),
                dose=scalar, arithmetic_scalar_float32=float(np.float32(scalar)),
                before=_bf16_binding(before_bits[row]), after=_bf16_binding(after_bits[row]),
                before_nonselected=_bf16_binding(before_bits[row, ~selected[row]]),
                after_nonselected=_bf16_binding(after_bits[row, ~selected[row]]), checks=checks,
                delta_norm_float64=norms["delta"][row, selected[row]].tolist(),
                delta_dot_signed_unit_float64=norms["dot"][row, selected[row]].tolist(),
                before_norm_float64=norms["before"][row, selected[row]].tolist(),
                after_norm_float64=norms["after"][row, selected[row]].tolist()))
        pending["geometry"] = geometry
        retain = phase == "prefill" or (step in DECODE_SAMPLES and pending["active_prediction_rows"][0])
        if retain:
            name = "prefill" if phase == "prefill" else f"decode_{step}"
            keys = dict(before=f"sample_{name}_before", after=f"sample_{name}_after")
            arrays[keys["before"]] = before_cpu[0].copy()
            arrays[keys["after"]] = after_cpu[0].copy()
            samples.append(dict(row=0, phase=phase, forward_index=step,
                processed_generated_token_index=None if phase == "prefill" else step,
                physical_token_indices=list(range(actual.shape[1])), activation_dtype="torch.bfloat16",
                arrays=keys, selected_token_indices=geometry[0]["selected_token_indices"],
                dose=geometry[0]["dose"]))

    def post_model(module, args, kwargs, output):
        nonlocal injected
        logits = output.logits[:, -1].detach().float().cpu().numpy()
        require(logits.shape == (ROWS, model.model.config.vocab_size), "Unexpected full-vocabulary logits")
        active = pending["active_prediction_rows"]
        require(np.isfinite(logits[np.asarray(active)]).all(), "Nonfinite active logits")
        frames.append(dict(**pending, vocabulary_size=logits.shape[-1], logit_dtype="float32",
            full_fp32_logits_sha256=[digest(array_binding(logits[row])) if active[row] else None for row in range(ROWS)],
            cache_length_after=int(output.past_key_values.get_seq_length())))
        pending.clear()
        fault = call["inject_failure_at"]
        if (fault == "after_prefill" and len(frames) == 1) or (fault == "after_first_decode" and len(frames) == 2):
            injected = True
            raise ControlledCoverageFault(fault)

    rows, error = None, None
    try:
        with ExitStack() as stack:
            for register, observer, kwargs in (
                (model.model.register_forward_pre_hook, pre_model, dict(with_kwargs=True)),
                (model.layers[LAYER].register_forward_hook, before_patch, {}),
                (model.layers[LAYER + 1].register_forward_pre_hook, after_patch, {}),
                (model.model.register_forward_hook, post_model, dict(with_kwargs=True)),
            ):
                handle = register(observer, **kwargs)
                stack.callback(handle.remove)
            rows, _ = backend.forward(requests, interventions=patches, max_new_tokens=call["cap"],
                                      intervention_schedule=call["schedule"])
    except Exception as caught:
        error = dict(type=type(caught).__name__, message=str(caught))
    hooks_after = hook_inventory(model)
    for sample in samples:
        sample["forward_completed"] = sample["forward_index"] < len(frames)
        sample["array_bindings"] = {side: array_binding(arrays[key]) for side, key in sample["arrays"].items()}
    eos = model.model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else sorted(eos or [])
    report = dict(schema_version=1, revision=REVISION, call=call, requests_sha256=digest(requests),
        direction=array_binding(np.asarray(direction, dtype=np.float32)), dose=dose,
        rows=rows, error=error, fault_injected=injected, attempted_forwards=attempted,
        completed_forwards=len(frames), forwards=frames, incomplete_forward=deepcopy(pending) or None,
        samples=samples, missing_decode_sample_ordinals=[k for k in DECODE_SAMPLES if not any(
            s["phase"] == "decode" and s["forward_index"] == k and s["forward_completed"] for s in samples)],
        hooks_before=initial_hooks, hooks_after=hooks_after, hooks_clean=model.active_hooks == 0 and hooks_after == initial_hooks,
        pad_token_id=model.tokenizer.pad_token_id, eos_token_ids=eos,
        prefill_width=int(prepared["input_ids"].shape[1]),
        generation_seconds=time.monotonic() - started,
        arrays={key: array_binding(value) for key, value in arrays.items()},
        norm_scope="Float64 descriptions; exact BF16 elementwise checks are primary. No CPU/GPU norm containment gate.",
        scientific_admission=False)
    report["checks"] = call_checks(report, requests)
    report["passed"] = all(report["checks"].values())
    return report, arrays


def call_checks(report, requests):
    """Local operational gates; the postrun auditor independently rederives them."""
    call, frames, rows = report["call"], report["forwards"], report["rows"]
    fault = call["inject_failure_at"]
    expected_fault = bool(fault and report["fault_injected"] and report["error"] ==
                          dict(type="ControlledCoverageFault", message=fault)
                          and len(frames) == (1 if fault == "after_prefill" else 2))
    geometry = bool(frames) and all(len(f.get("geometry", [])) == ROWS and all(
        all(g["checks"].values()) and g["before_nonselected"] == g["after_nonselected"]
        for g in f["geometry"]) for f in frames)
    engaged = all(all(norm > 0 and dot > 0 for norm, dot in zip(g["delta_norm_float64"], g["delta_dot_signed_unit_float64"]))
                  for f in frames for g in f["geometry"] if g["dose"] != 0)
    row0_decode = any(f["phase"] == "decode" and f["active_prediction_rows"][0]
                     and f["geometry"][0]["dose"] != 0 and f["geometry"][0]["selected_token_count"] == 1
                     for f in frames)
    needs_decode = call["stage"] == "checks" and call["condition"] in ("record_nonzero", "full_nonzero", "full_one_row")
    observed = rows is not None and len(rows) == ROWS
    return dict(terminal_disposition=expected_fault if fault else observed and report["error"] is None,
        completed_forward_inventory=report["attempted_forwards"] == len(frames) and report["incomplete_forward"] is None,
        timeline=validate_timeline(report, requests), all_position_geometry=geometry,
        nonzero_signed_engagement=engaged,
        row0_nonzero_active_decode=(not needs_decode or row0_decode),
        backend_full_prefill_observations=(rows is None and expected_fault) or bool(observed and all(
            (call["schedule"] != ALL_ATTENDED_SCHEDULE or
             r.get("intervention_trace", {}).get("independent_prefill_observations_passed") is True)
            and all(o["passed"] is True for o in r.get("prefill_geometry_observations", [])) for r in rows)),
        saved_prefill=any(s["phase"] == "prefill" and s["forward_completed"] for s in report["samples"]),
        hook_cleanup=report["hooks_clean"] is True and report["hooks_before"] == report["hooks_after"] == {})


def validate_timeline(report, requests):
    frames, rows = report["forwards"], report["rows"]
    if not frames:
        return False
    width, eos, pad = report["prefill_width"], set(report["eos_token_ids"]), report["pad_token_id"]
    if width != ((max(len(r["token_prefix"]) for r in requests) + PREFILL_PADDING_MULTIPLE - 1)
                 // PREFILL_PADDING_MULTIPLE) * PREFILL_PADDING_MULTIPLE:
        return False
    sequences = None if rows is None else [row["completion"]["generated_token_ids"] for row in rows]
    if sequences is not None:
        if len(rows) != ROWS or len(frames) != max(map(len, sequences)):
            return False
        for row, (tokens, result) in enumerate(zip(sequences, rows)):
            if not (tokens and len(tokens) <= report["call"]["cap"] and not any(t in eos for t in tokens[:-1])
                    and (tokens[-1] in eos or len(tokens) == report["call"]["cap"])
                    and result["completion"]["ended_with_eos"] is (tokens[-1] in eos)
                    and result["completion"]["sampling"] == dict(seed=requests[row]["seed"], temperature=.7,
                        top_p=1.0, max_new_tokens=report["call"]["cap"])):
                return False
            trace = result["intervention_trace"]
            count = 0 if report["call"]["position"] is None else (
                len(requests[row]["token_prefix"]) if report["call"]["position"] == "all_attended" else 1)
            if not (trace["schedule"] == report["call"]["schedule"]
                    and trace["prefill_patch_count"] == count
                    and trace["active_decode_forward_count"] == len(tokens) - 1
                    and trace["decode_patch_count"] == (len(tokens) - 1 if count else 0)
                    and trace["total_patch_count"] == len(trace["events"]) == count + (len(tokens) - 1 if count else 0)):
                return False
            for i, event in enumerate(trace["events"]):
                prefill = i < count
                unpadded = (i if report["call"]["position"] == "all_attended" else requests[row]["prefix_positions"]["record"]) if prefill else len(requests[row]["token_prefix"]) + i - count
                scalar = 0.0 if report["call"]["dose_multiplier"] == 0 or report["call"]["one_row_only"] not in (None, row) else report["dose"]
                if not (event["phase"] == ("prefill" if prefill else "decode") and event["layer"] == LAYER
                        and event["activation_dtype"] == "torch.bfloat16" and event["mode"] == "add"
                        and event["position"] == (report["call"]["position"] if prefill else "current_generated_token")
                        and event["requested_scalar_delta"] == scalar
                        and event["processed_generated_token_index"] == (None if prefill else i - count + 1)
                        and event["unpadded_token_index"] == unpadded
                        and event["token_index"] == (width - len(requests[row]["token_prefix"]) + unpadded if prefill else 0)
                        and event["processed_token_id"] == (requests[row]["token_prefix"][unpadded] if prefill else tokens[i - count])):
                    return False
    for step, frame in enumerate(frames):
        expected_ids, expected_pos, expected_masks = [], [], []
        for row, request in enumerate(requests):
            prefix = request["token_prefix"]
            if sequences is None:
                if step > 1:
                    return False
                terminal = step == 1 and frame["input_ids"][row][0] in eos
                tail = [] if step == 0 else [int(not terminal)]
            else:
                tail = [int(k < len(sequences[row])) for k in range(1, step + 1)]
            mask = [0] * (width - len(prefix)) + [1] * len(prefix) + tail
            expected_masks.append(mask)
            expected_ids.append([pad] * (width - len(prefix)) + prefix if step == 0 else
                frame["input_ids"][row] if sequences is None else
                [sequences[row][step - 1] if step <= len(sequences[row]) else pad])
            expected_pos.append([0] * (width - len(prefix)) + list(range(len(prefix))) if step == 0 else [max(sum(mask) - 1, 0)])
        if not (frame["forward_index"] == step and frame["phase"] == ("prefill" if step == 0 else "decode")
                and frame["prediction_ordinal"] == step + 1 and frame["input_ids"] == expected_ids
                and frame["position_ids"] == expected_pos
                and frame["attention_mask_shape"] == [ROWS, width + step]
                and frame["attention_mask_row_sha256"] == [digest(mask) for mask in expected_masks]
                and frame["active_prediction_rows"] == [bool(mask[-1]) for mask in expected_masks]
                and frame["cache_length_before"] == (0 if step == 0 else width + step - 1)
                and frame["cache_length_after"] == width + step and frame["cache_position_supplied"] is False):
            return False
    return True


def relationship_checks(results):
    """Only fully present comparisons can pass the ten-call engineering gate."""
    def equal(left, right, indices=range(ROWS)):
        return left in results and right in results and results[left]["rows"] is not None and results[right]["rows"] is not None and identical_rows(results[left], results[right], indices)
    # The schedule string is metadata, not a different raw completion. The old
    # identity helper compares completion dictionaries, so strip that field on
    # copies only while retaining all tokens/sampling/EOS and active forwards.
    normalized = {}
    for index, result in results.items():
        normalized[index] = dict(result)
        if result["rows"] is not None:
            normalized[index]["rows"] = [dict(row, completion={k: v for k, v in row["completion"].items()
                                                              if k != "intervention_schedule"}) for row in result["rows"]]
    results = normalized
    prefill = bool(results) and all(result["forwards"] and result["forwards"][0]["geometry"]
        and [g["before"] for g in result["forwards"][0]["geometry"]] ==
            [g["before"] for g in results[0 if result["call"]["source_cohort"] == 0 else 13]["forwards"][0]["geometry"]]
        for result in results.values()
        if (0 if result["call"]["source_cohort"] == 0 else 13) in results)
    return dict(record_zero_identity=equal(0, 1), full_zero_identity=equal(0, 2),
                row0_only_isolation=equal(0, 3, range(1, ROWS)),
                clean_after_prefill_fault=equal(0, 7), clean_after_decode_fault=equal(0, 9),
                common_prefill_before_identity=prefill)


def _persist(root, stem, report, arrays):
    """Write one terminal result exactly once; timings live in the outer summary."""
    started = time.monotonic()
    array_path, report_path = root / f"{stem}.npz", root / f"{stem}.json"
    array_path.parent.mkdir(parents=True, exist_ok=True)
    with array_path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    array_sha = hashlib.sha256(array_path.read_bytes()).hexdigest()
    report["array_artifact"] = dict(path=str(array_path.relative_to(root)), byte_sha256=array_sha,
                                   arrays={key: array_binding(value) for key, value in arrays.items()})
    write_once(report_path, report)
    return dict(path=str(report_path.relative_to(root)), byte_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
                serialization_seconds=time.monotonic() - started,
                artifact_bytes=array_path.stat().st_size + report_path.stat().st_size)


def collect_receipts(model, source_assignments, requests, call, generation, *, deadline=None):
    """Actual selected payloads only, after the timing call's interventions end."""
    require(len(source_assignments) == len(requests) == ROWS and generation["rows"] is not None,
            "Actual terminal timing source rows are required")
    started = time.monotonic()
    records, tasks, active_requests, active_ids, active_indices = [], [], [], [], []
    for index, (assignment, request, row) in enumerate(zip(source_assignments, requests, generation["rows"])):
        require(assignment["sampling_seed"] == request["seed"], "Receipt source seed differs")
        inference = row["completion"]
        source = voluntary.record_from_thinking_inference(assignment, inference)
        tasks.append(source)
        if assignment["kind"] != "choice":
            continue
        sidecar = receipts.choice_sidecar(source)
        sidecar.update(source_trial_id=call["trial_ids"][index], base_trial_id=request["source_trial_id"],
                       generation_row=index, source_assignment_sha256=digest(assignment))
        if source["cap_hit"]:
            sidecar.update(status="capped_choice", receipt_required=False)
        if sidecar["receipt_required"]:
            active_indices.append(len(records))
            active_ids.append(call["trial_ids"][index])
        records.append(sidecar)
    measurements, guard, error, arrays, forward_attempted = None, None, None, {}, False
    before, seconds, parsing_seconds = None, 0.0, None
    try:
        require(model.active_hooks == 0 and not hook_inventory(model), "Receipt entry is not clean")
        for index in active_indices:
            sidecar = records[index]
            source = tasks[sidecar["generation_row"]]
            receipt, actual_request, binding = receipts.render_receipt(source, model)
            sidecar.update(receipt_input=receipt, token_binding=binding)
            active_requests.append(actual_request)
        parsing_seconds = time.monotonic() - started
        if active_requests:
            if deadline is not None and time.time() >= deadline:
                raise TimeoutError("Coverage deadline before actual receipts")
            before, forward_attempted = time.monotonic(), True
            measurements, hidden, guard = receipts.forward_receipts(model, active_requests, active_ids, layers=[LAYER])
            seconds = time.monotonic() - before
            arrays["hidden"] = hidden
            require(guard["passed"] is True, "Actual receipt full-vocabulary identity failed")
            for active_row, index in enumerate(active_indices):
                sidecar = records[index]
                executed = deepcopy(sidecar["executed_simulator_outcome"])
                executed["receipt_model_pass_completed"] = True
                executed["observation_ledger"][-1]["model_pass_completed"] = True
                sidecar.update(status="completed", receipt_model_pass_completed=True,
                    measurement=measurements[active_row], activation_row=active_row, executed_simulator_outcome=executed)
        require(model.active_hooks == 0 and not hook_inventory(model), "Receipt cleanup failed")
    except Exception as caught:
        error = dict(type=type(caught).__name__, message=str(caught))
        if before is not None:
            seconds = time.monotonic() - before
        for index in active_indices:
            sidecar = records[index]
            source = tasks[sidecar["generation_row"]]
            sidecar.update(status="execution_error", receipt_model_pass_completed=False, measurement=None,
                           activation_row=None, executed_simulator_outcome=deepcopy(source["simulator_outcome"]), error=error)
    if parsing_seconds is None:
        parsing_seconds = time.monotonic() - started
    return dict(call_index=call["call_index"], call_id=call["call_id"], records=records,
        source_records=tasks, records_sha256=digest(records), requests=active_requests,
        requests_sha256=digest(active_requests), active_source_trial_ids=active_ids,
        assigned_choices=len(records), actual_receipt_rows=len(active_requests),
        receipt_forward_attempted=forward_attempted,
        actual_forward_calls=2 if guard is not None else None if forward_attempted else 0,
        maximum_unpadded_width=max((len(r["token_prefix"]) for r in active_requests), default=0),
        actual_padded_width=None if guard is None else guard["prefill_width"],
        parsing_seconds=parsing_seconds, receipt_forward_seconds=seconds,
        observed_measurements=measurements, engineering=guard, error=error,
        passed=error is None and all(r["status"] in ("completed", "stop", "invalid_choice", "capped_choice") for r in records),
        worst_case_workload_measured=False,
        scope="Actual selected-payload forwards only; input-only potential workload is separate, not measured worst-case timing."), arrays


def execute(root, model, materials, *, checkpoint=None, deadline=None):
    """Persist the one fixed engineering sequence. No retry, resume or admission."""
    root = Path(root)
    engineering = materials["engineering"]
    requests = {int(k): v for k, v in engineering["requests_by_cohort"].items()}
    sources = {int(k): v for k, v in engineering["source_assignments_by_cohort"].items()}
    calls = call_plan(requests)
    require(calls == engineering["call_plan"] and digest(calls) == engineering["call_plan_sha256"], "Fixed16 inventory changed")
    require(set(sources) == {0, 3} and all(len(v) == ROWS for v in sources.values()), "Require both complete source cohorts")
    require(materials["dose"] == DOSE, "Original dose changed")
    direction = materials["direction_bindings"]["curiosity_negative"]["unit"]
    require(not (root / "engineering-started.json").exists(), "Coverage engineering was already attempted; no retry or resume")
    write_once(root / "engineering-started.json", dict(revision=REVISION, materials_sha256=digest(materials),
        call_plan_sha256=digest(calls), assigned_slots=512, started=time.time(), deadline=deadline,
        paid_retry_allowed=False, scientific_admission=False))
    results, saved, receipt_results, failure = {}, {}, {}, None
    for call in calls:
        index, cohort = call["call_index"], call["source_cohort"]
        if index == 10 and (len(results) != 10 or not all(r["passed"] for r in results.values())
                            or not all(relationship_checks(results).values())):
            failure = dict(type="EngineeringGateFailure", message="First ten checks did not pass")
        if failure is not None:
            break
        if deadline is not None and time.time() >= deadline:
            failure = dict(type="TimeoutError", message="Deadline before next fixed call")
            break
        write_once(root / "attempts" / f"{index:02d}.json", dict(call=call, materials_sha256=digest(materials), started=time.time()))
        try:
            report, arrays = collect_call(model, requests[cohort], call, direction, deadline=deadline)
            results[index] = report
            saved[index] = _persist(root, f"calls/{index:02d}", report, arrays)
            if checkpoint:
                checkpoint()
            if not report["passed"]:
                failure = dict(type="EngineeringGateFailure", message=f"Call{index} failed", checks=report["checks"])
                continue
            if call["stage"] == "timing":
                receipt, receipt_arrays = collect_receipts(model, sources[cohort], requests[cohort], call, report, deadline=deadline)
                receipt_results[index] = receipt
                saved[index]["receipt"] = _persist(root, f"receipts/{index:02d}", receipt, receipt_arrays)
                if checkpoint:
                    checkpoint()
                if not receipt["passed"]:
                    failure = dict(type="ReceiptEngineeringFailure", message=f"Actual receipt call{index} failed")
        except Exception as caught:
            failure = dict(type=type(caught).__name__, message=str(caught))
            write_once(root / "failures" / f"{index:02d}.json", dict(call=call, error=failure))
            if checkpoint:
                checkpoint()
    dispositions, slots = [], []
    for call in calls:
        index = call["call_index"]
        result = results.get(index)
        attempted = (root / "attempts" / f"{index:02d}.json").exists()
        status = ("execution_error" if attempted else "missing") if result is None or index not in saved else "expected_fault" if call["expected_error"] and result["passed"] else "completed" if result["rows"] is not None else "execution_error"
        dispositions.append(dict(call=call, status=status, passed=False if result is None else result["passed"],
                                 artifact=saved.get(index)))
        slots.extend(dict(trial_id=trial_id, base_trial_id=base_id, call_index=index, status=status)
                     for trial_id, base_id in zip(call["trial_ids"], call["base_trial_ids"]))
    checks = relationship_checks(results)
    receipt_slots = []
    for call in calls[10:]:
        recorded = {r["generation_row"]: r for r in receipt_results.get(call["call_index"], {}).get("records", [])}
        for row, assignment in enumerate(sources[call["source_cohort"]]):
            if assignment["kind"] == "choice":
                value = recorded.get(row)
                receipt_slots.append(dict(trial_id=call["trial_ids"][row], base_trial_id=call["base_trial_ids"][row],
                    call_index=call["call_index"], status="missing" if value is None else value["status"],
                    receipt_required=None if value is None else value["receipt_required"],
                    receipt_model_pass_completed=False if value is None else value["receipt_model_pass_completed"]))
    complete = len(results) == len(saved) == 16 and len(receipt_results) == 6
    summary = dict(revision=REVISION, materials_sha256=digest(materials), call_plan_sha256=digest(calls),
        assigned_slots=512, assigned_calls=16, expected_fault_slots=64, ordinary_slots=448,
        complete=complete, engineering_passed=complete and failure is None and all(checks.values())
            and all(r["passed"] for r in results.values()) and all(r["passed"] for r in receipt_results.values()),
        call_dispositions=dispositions, slots=slots, relationship_checks=checks, error=failure,
        assigned_receipt_slots=len(receipt_slots), receipt_slots=receipt_slots,
        observed_receipt_slots=sum(r["assigned_choices"] for r in receipt_results.values()),
        timing_calls=[dict(call_index=index, source_cohort=results[index]["call"]["source_cohort"],
            condition=results[index]["call"]["condition"], generation_seconds=results[index]["generation_seconds"],
            serialization_seconds=saved[index]["serialization_seconds"],
            receipt_parsing_seconds=receipt["parsing_seconds"], receipt_forward_seconds=receipt["receipt_forward_seconds"],
            receipt_serialization_seconds=saved[index]["receipt"]["serialization_seconds"],
            actual_receipt_rows=receipt["actual_receipt_rows"], actual_receipt_width=receipt["actual_padded_width"],
            artifact_bytes=saved[index]["artifact_bytes"] + saved[index]["receipt"]["artifact_bytes"])
            for index, receipt in receipt_results.items() if index in saved and "receipt" in saved[index]],
        potential_receipt_workload=engineering["potential_receipt_workload"], worst_case_receipt_runtime_measured=False,
        export_seconds=None, export_scope="Measured by outer checkpoint/volume/export controller, not inferred here.",
        finished=time.time(), scientific_admission=False)
    write_once(root / "engineering-summary.json", summary)
    if checkpoint:
        checkpoint()
    return summary
