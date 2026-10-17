"""Independent arithmetic helpers for the prospectively fixed coverage engineering.

This module does not launch, admit, retry, score behavior or select samples.
Saved arrays permit exact BF16 element reconstruction. CPU float64 norm summaries
are descriptive, not exact matches to GPU reduction results or invented gates.
"""

import hashlib
import math

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def array_binding(array):
    value = np.ascontiguousarray(array)
    return dict(dtype=str(value.dtype), shape=list(value.shape),
                byte_sha256=hashlib.sha256(value.tobytes()).hexdigest())


def bfloat16_round(values):
    """IEEE float32 -> BF16 ties-to-even -> exact float32, without model code."""
    value = np.ascontiguousarray(values, dtype=np.float32)
    require(np.isfinite(value).all(), "Nonfinite requested float32 activation")
    bits = value.view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def audit_saved_pair(before, after, selected_indices, direction, dose):
    """Check one saved actual before/next-block-after matrix and explicit mask.

Rows index physical token positions, including padding for a saved prefill.
A baseline has no selected positions. A true sham selects the declared positions
with dose zero. Decode samples use one physical position and the same arithmetic.
The caller independently derives selected_indices from the request and schedule.
"""
    before, after, direction = np.asarray(before), np.asarray(after), np.asarray(direction)
    require(before.dtype == after.dtype == np.float32 and before.ndim == after.ndim == 2
            and before.shape == after.shape and before.shape[0] > 0 and before.shape[1] > 0,
            "Require matching nonempty float32 matrices containing exact BF16 values")
    require(direction.dtype == np.float32 and direction.shape == (before.shape[1],)
            and np.isfinite(direction).all() and type(dose) in (int, float) and math.isfinite(dose),
            "Invalid applied float32 direction/scalar")
    require(isinstance(selected_indices, (tuple, list)) and all(type(i) is int for i in selected_indices)
            and list(selected_indices) == sorted(set(selected_indices))
            and all(0 <= i < before.shape[0] for i in selected_indices), "Invalid selected-token inventory")
    require(np.isfinite(before).all() and np.isfinite(after).all(), "Nonfinite saved activation")
    require(np.all((before.view(np.uint32) & 0xFFFF) == 0)
            and np.all((after.view(np.uint32) & 0xFFFF) == 0), "Saved float32 values are not exact BF16 elements")
    scalar = np.float32(dose)
    require(np.isfinite(scalar), "Requested scalar does not fit float32")
    selected = np.zeros(before.shape[0], dtype=bool)
    selected[list(selected_indices)] = True
    expected = before.copy()
    # Match the declared two eager operations, not a fused multiply-add.
    delta = np.multiply(scalar, direction, dtype=np.float32)
    requested = np.add(before[selected], delta, dtype=np.float32)
    expected[selected] = bfloat16_round(requested)
    actual_delta = after[selected].astype(np.float64) - before[selected].astype(np.float64)
    delta_norm = np.linalg.norm(actual_delta, axis=-1)
    signed_projection = actual_delta @ direction.astype(np.float64)
    finite_expected = bool(np.isfinite(expected).all())
    checks = dict(
        finite_expected=finite_expected,
        selected_exact_bfloat16=bool(np.array_equal(after[selected].view(np.uint32), expected[selected].view(np.uint32))),
        nonselected_bitwise_unchanged=bool(np.array_equal(after[~selected].view(np.uint32), before[~selected].view(np.uint32))),
        achieved_nonzero_and_correct_sign=bool(not selected_indices or dose == 0 or
            (np.all(delta_norm > 0) and np.all(np.sign(dose) * signed_projection > 0))),
    )
    return dict(passed=all(checks.values()), checks=checks, before=array_binding(before), after=array_binding(after),
        direction_float32=array_binding(direction), requested_scalar=dose, arithmetic_scalar_float32=float(scalar),
        total_token_positions=before.shape[0], selected_token_positions=list(selected_indices),
        independently_reconstructed_components=len(selected_indices) * before.shape[1],
        cpu_float64_descriptive=dict(before_norm=np.linalg.norm(before[selected].astype(np.float64), axis=-1).tolist(),
            after_norm=np.linalg.norm(after[selected].astype(np.float64), axis=-1).tolist(),
            achieved_delta_norm=delta_norm.tolist(), achieved_dot_applied_direction=signed_projection.tolist()),
        norm_scope="CPU float64 descriptive reductions; no CPU/GPU norm-equality gate.")


# These constants describe this new finite engineering assay, not the old
# record-only validator. The archived legacy validators are never patched.
ROWS, LAYER, PADDING = 32, 15, 128
DOSE = 29.78968343510005
RECORD = "record_prefill_plus_decode"
FULL = "all_attended_prefill_plus_decode"
DECODE_SAMPLES = (1, 2, 32, 128)


def digest(value):
    import json
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def valid_hash(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")


def bf16_binding(array):
    value = np.ascontiguousarray(array, dtype=np.float32)
    require(np.isfinite(value).all() and np.all((value.view(np.uint32) & 0xFFFF) == 0), "Binding input is not finite exact BF16")
    bits = (value.view(np.uint32) >> 16).astype(np.uint16)
    return dict(dtype="torch.bfloat16", shape=list(value.shape), byte_sha256=hashlib.sha256(bits.tobytes()).hexdigest())


def row_dose(call, row):
    if call["position"] is None or call["one_row_only"] not in (None, row):
        return 0.0
    return DOSE * call["dose_multiplier"]


def selected_positions(call, request, frame, row, width):
    if call["position"] is None:
        return []
    if frame["phase"] == "decode":
        return [0] if frame["active_prediction_rows"][row] else []
    padding = width - len(request["token_prefix"])
    if call["position"] == "all_attended":
        return list(range(padding, width))
    require(call["position"] == "record", "Unsupported prefill coverage")
    return [padding + request["prefix_positions"]["record"]]


def audit_timeline(report, requests, *, vocabulary_size, pad_token_id, eos_ids):
    frames, rows, call = report["forwards"], report["rows"], report["call"]
    width = ((max(len(r["token_prefix"]) for r in requests) + PADDING - 1) // PADDING) * PADDING
    checks = dict(nonempty_frames=bool(frames), width=report["prefill_width"] == width,
                  terminal_tokens=True, full_fingerprints=True, inputs_masks_cache=True)
    sequences = None
    if rows is not None:
        checks["terminal_tokens"] = len(rows) == ROWS
        if len(rows) != ROWS:
            return dict(passed=False, checks=checks)
        sequences = [r["completion"]["generated_token_ids"] for r in rows]
        checks["terminal_tokens"] &= len(frames) == max(map(len, sequences), default=0)
        for request, row, tokens in zip(requests, rows, sequences):
            completion = row["completion"]
            checks["terminal_tokens"] &= bool(isinstance(tokens, list) and 0 < len(tokens) <= call["cap"]
                and all(type(t) is int and 0 <= t < vocabulary_size for t in tokens)
                and not any(t in eos_ids for t in tokens[:-1])
                and (tokens[-1] in eos_ids or len(tokens) == call["cap"])
                and completion["ended_with_eos"] is (tokens[-1] in eos_ids)
                and row["token_prefix_sha256"] == request["token_prefix_sha256"]
                and row["hooks_remaining"] == 0
                and completion["sampling"] == dict(seed=request["seed"], temperature=.7, top_p=1., max_new_tokens=call["cap"]))
    for step, frame in enumerate(frames):
        inputs, positions, masks = [], [], []
        if sequences is None and step > 1:
            checks["inputs_masks_cache"] = False
            continue
        for row, request in enumerate(requests):
            prefix = request["token_prefix"]
            padding = width - len(prefix)
            if sequences is None:
                # A controlled fault has no fabricated completion. At most one
                # actual sampled token was forwarded; its EOS determines mask.
                token = None if step == 0 else frame["input_ids"][row]
                if step:
                    checks["inputs_masks_cache"] &= bool(len(token) == 1 and type(token[0]) is int and 0 <= token[0] < vocabulary_size)
                tail = [] if step == 0 else [int(token[0] not in eos_ids)]
                supplied = [pad_token_id] * padding + prefix if step == 0 else token
            else:
                tokens = sequences[row]
                tail = [int(k < len(tokens)) for k in range(1, step + 1)]
                supplied = [pad_token_id] * padding + prefix if step == 0 else [tokens[step - 1] if step <= len(tokens) else pad_token_id]
            mask = [0] * padding + [1] * len(prefix) + tail
            masks.append(mask)
            inputs.append(supplied)
            positions.append([0] * padding + list(range(len(prefix))) if step == 0 else [max(sum(mask) - 1, 0)])
        active = [bool(mask[-1]) for mask in masks]
        checks["inputs_masks_cache"] &= bool(frame["forward_index"] == step and frame["prediction_ordinal"] == step + 1
            and frame["phase"] == ("prefill" if step == 0 else "decode")
            and frame["input_ids"] == inputs and frame["position_ids"] == positions
            and frame["attention_mask_shape"] == [ROWS, width + step]
            and frame["attention_mask_row_sha256"] == [digest(mask) for mask in masks]
            and frame["active_prediction_rows"] == active
            and frame["cache_length_before"] == (0 if step == 0 else width + step - 1)
            and frame["cache_length_after"] == width + step and frame["cache_position_supplied"] is False)
        hashes = frame["full_fp32_logits_sha256"]
        checks["full_fingerprints"] &= bool(frame["vocabulary_size"] == vocabulary_size and frame["logit_dtype"] == "float32"
            and len(hashes) == ROWS and all(valid_hash(value) if live else value is None for value, live in zip(hashes, active)))
    return dict(passed=all(checks.values()), checks=checks)


def audit_online_geometry(report, requests, *, hidden_size):
    checks = dict(all32_per_forward=True, selected_inventory=True, bindings=True,
                  online_exact_flags=True, nonselected_identity=True, scalar_inventory=True, signed_engagement=True)
    width, call = report["prefill_width"], report["call"]
    selected_count = 0
    for frame in report["forwards"]:
        geometry = frame.get("geometry", [])
        checks["all32_per_forward"] &= len(geometry) == ROWS
        if len(geometry) != ROWS:
            continue
        for row, (request, item) in enumerate(zip(requests, geometry)):
            selected = selected_positions(call, request, frame, row, width)
            count, token_count, dose = len(selected), width if frame["phase"] == "prefill" else 1, row_dose(call, row)
            selected_count += count
            checks["selected_inventory"] &= bool(item["row"] == row and item["selected_token_indices"] == selected
                and item["selected_token_count"] == count and item["dose"] == dose
                and item["arithmetic_scalar_float32"] == float(np.float32(dose)))
            for name in ("before", "after", "before_nonselected", "after_nonselected"):
                binding = item[name]
                shape = [token_count if name in ("before", "after") else token_count - count, hidden_size]
                checks["bindings"] &= bool(binding["dtype"] == "torch.bfloat16" and binding["shape"] == shape
                                           and valid_hash(binding["byte_sha256"]))
            checks["online_exact_flags"] &= item["checks"] == dict(finite=True, exact_bfloat16=True, nonselected_bitwise_unchanged=True)
            checks["nonselected_identity"] &= item["before_nonselected"] == item["after_nonselected"]
            if dose == 0 or not selected:
                checks["nonselected_identity"] &= item["before"] == item["after"]
            arrays = [item[key] for key in ("delta_norm_float64", "delta_dot_signed_unit_float64", "before_norm_float64", "after_norm_float64")]
            valid = all(isinstance(v, list) and len(v) == count and all(type(x) in (int, float) and math.isfinite(x) for x in v) for v in arrays)
            checks["scalar_inventory"] &= valid
            if valid:
                norms, dots, before_norms, after_norms = arrays
                checks["scalar_inventory"] &= all(x >= 0 for values in (norms, before_norms, after_norms) for x in values)
                checks["signed_engagement"] &= (all(norm == dot == 0 for norm, dot in zip(norms, dots)) if dose == 0
                    else all(norm > 0 and dot > 0 for norm, dot in zip(norms, dots)))
    return dict(passed=bool(report["forwards"]) and all(checks.values()), checks=checks,
                observed_selected_token_events=selected_count,
                scope="All-row online arithmetic flags, scalar inventory and byte bindings; unsaved nonzero vectors are not independently reconstructible offline.")


def audit_samples(report, arrays, requests, direction):
    require(set(arrays) == set(report["arrays"]) and all(array_binding(value) == report["arrays"][key] for key, value in arrays.items()),
            "Saved array inventory/content binding differs")
    frame_map = {f["forward_index"]: f for f in report["forwards"]}
    incomplete = report["incomplete_forward"]
    if incomplete and "geometry" in incomplete:
        frame_map[incomplete["forward_index"]] = incomplete
    expected = [step for step, frame in sorted(frame_map.items()) if step == 0 or
                (step in DECODE_SAMPLES and frame["active_prediction_rows"][0])]
    samples = report["samples"]
    require([s["forward_index"] for s in samples] == expected, "Fixed sample inventory differs")
    names, results = set(), []
    for sample in samples:
        step = sample["forward_index"]
        frame = frame_map[step]
        phase, count = ("prefill", report["prefill_width"]) if step == 0 else ("decode", 1)
        tag = "prefill" if step == 0 else f"decode_{step}"
        keys = dict(before=f"sample_{tag}_before", after=f"sample_{tag}_after")
        selected = selected_positions(report["call"], requests[0], frame, 0, report["prefill_width"])
        require(sample["row"] == 0 and sample["phase"] == phase and sample["processed_generated_token_index"] == (None if step == 0 else step)
            and sample["physical_token_indices"] == list(range(count)) and sample["activation_dtype"] == "torch.bfloat16"
            and sample["forward_completed"] is (step < len(report["forwards"])) and sample["arrays"] == keys
            and sample["selected_token_indices"] == selected and sample["dose"] == row_dose(report["call"], 0), "Sample site/phase/coverage differs")
        before, after = arrays[keys["before"]], arrays[keys["after"]]
        require(before.shape == after.shape == (count, len(direction))
                and sample["array_bindings"] == {side: array_binding(arrays[key]) for side, key in keys.items()}, "Sample shape/byte binding differs")
        result = audit_saved_pair(before, after, selected, direction, sample["dose"])
        geometry = frame["geometry"][0]
        keep = np.ones(count, bool)
        keep[selected] = False
        require(geometry["before"] == bf16_binding(before) and geometry["after"] == bf16_binding(after)
                and geometry["before_nonselected"] == bf16_binding(before[keep])
                and geometry["after_nonselected"] == bf16_binding(after[keep]), "Saved sample differs from all-row online bindings")
        names.update(keys.values())
        results.append(dict(forward_index=step, phase=phase, forward_completed=sample["forward_completed"], **result))
    missing = [k for k in DECODE_SAMPLES if not any(s["phase"] == "decode" and s["forward_index"] == k and s["forward_completed"] for s in samples)]
    require(report["missing_decode_sample_ordinals"] == missing, "Missing samples were hidden or replaced")
    if "hidden" in arrays:
        hidden = arrays["hidden"]
        require(hidden.dtype == np.float32 and hidden.shape == (ROWS, 1, 2, len(direction)) and np.isfinite(hidden).all()
                and np.all((hidden.view(np.uint32) & 0xFFFF) == 0), "Invalid prepatch record/final captures")
        names.add("hidden")
        if "sample_prefill_before" in arrays:
            prefix = requests[0]["token_prefix"]
            record = report["prefill_width"] - len(prefix) + requests[0]["prefix_positions"]["record"]
            require(np.array_equal(hidden[0, 0].view(np.uint32), arrays["sample_prefill_before"][[record, -1]].view(np.uint32)),
                    "Source captures are not the independently observed prepatch values")
    require(names == set(arrays), "Unexpected/missing sampled array")
    return dict(passed=bool(results) and all(r["passed"] for r in results), samples=results,
        missing_decode_sample_ordinals=missing,
        reconstructed_selected_components=sum(r["independently_reconstructed_components"] for r in results),
        observed_full_token_vectors=sum(r["total_token_positions"] for r in results))


def audit_trace_inventory(report, requests):
    """No record-only schema is silently accepted for full-prefix interventions."""
    rows, call = report["rows"], report["call"]
    if rows is None:
        return dict(passed=False, observed_rows=0)
    checks = dict(exact_schedule=True, token_event_inventory=True, scalar_event_inventory=True,
                  independent_full_prefill_observers=True)
    fields = ("before_norm", "delta_norm", "after_norm", "requested_delta_norm", "rounding_error_norm",
              "relative_delta_norm", "direction_norm", "projection_before", "projection_after")
    width = report["prefill_width"]
    for row, (result, request) in enumerate(zip(rows, requests)):
        trace, tokens = result["intervention_trace"], result["completion"]["generated_token_ids"]
        full = call["schedule"] == FULL
        patched = call["position"] is not None
        prefix = request["token_prefix"]
        prefill = [] if not patched else list(range(len(prefix))) if full else [request["prefix_positions"]["record"]]
        expected = [("prefill", None, width - len(prefix) + p, p, prefix[p]) for p in prefill]
        if patched:
            expected += [("decode", k, 0, len(prefix) + k - 1, tokens[k - 1]) for k in range(1, len(tokens))]
        checks["exact_schedule"] &= bool(trace["schema_version"] == (2 if full else 1)
            and trace["schedule"] == call["schedule"] and trace["generated_token_count"] == len(tokens)
            and trace["active_decode_forward_count"] == len(tokens) - 1
            and trace["prefill_patch_count"] == len(prefill)
            and trace["decode_patch_count"] == (len(tokens) - 1 if patched else 0)
            and trace["total_patch_count"] == len(expected))
        events = trace["events"]
        checks["token_event_inventory"] &= len(events) == len(expected)
        for event, (phase, step, physical, unpadded, token) in zip(events, expected):
            checks["token_event_inventory"] &= bool(event["phase"] == phase and event["layer"] == LAYER
                and event["position"] == (call["position"] if phase == "prefill" else "current_generated_token")
                and event["mode"] == "add" and event["token_index"] == physical and event["unpadded_token_index"] == unpadded
                and event["processed_generated_token_index"] == step and event["processed_token_id"] == token
                and event["activation_dtype"] == "torch.bfloat16")
            checks["scalar_event_inventory"] &= bool(event["requested_scalar_delta"] == row_dose(call, row)
                and event["arithmetic_scalar_float32"] == float(np.float32(row_dose(call, row)))
                and all(type(event[k]) in (int, float) and math.isfinite(event[k]) for k in fields)
                and (event["delta_norm"] > 0 if row_dose(call, row) != 0 else event["delta_norm"] == 0))
        if full:
            checks["exact_schedule"] &= bool(trace["initial_attended_token_count"] == len(prefix)
                and trace["prefill_forward_patch_count"] == int(patched) and trace["prefill_token_patch_count"] == len(prefill))
            observations = result.get("prefill_geometry_observations", [])
            # This finite plan installs full-prefix patches on all32 rows,
            # including true zeros in the one-row condition.
            checks["independent_full_prefill_observers"] &= bool(len(observations) == 1 and
                trace.get("independent_prefill_observations_passed") is True)
            if len(observations) == 1 and report["forwards"]:
                observation, geometry = observations[0], report["forwards"][0]["geometry"][row]
                mask = [0] * (width - len(prefix)) + [1] * len(prefix)
                checks["independent_full_prefill_observers"] &= bool(observation["layer"] == LAYER and observation["next_layer"] == LAYER + 1
                    and observation["row"] == row and observation["observer"] == "layer_output_before_mutation_to_next_layer_input"
                    and observation["initial_attention_mask_sha256"] == digest(mask)
                    and observation["selected_token_indices"] == geometry["selected_token_indices"]
                    and observation["selected_token_count"] == geometry["selected_token_count"]
                    and observation["before"] == geometry["before"] and observation["after"] == geometry["after"]
                    and observation["checks"] == dict(finite=True, selected_exact_bfloat16=True, nonselected_bitwise_unchanged=True)
                    and observation["passed"] is True)
            for phase in ("prefill", "decode"):
                chosen = [e for e in events if e["phase"] == phase]
                exposure = trace["exposure"][phase]
                checks["scalar_event_inventory"] &= exposure == dict(token_patch_count=len(chosen),
                    achieved_norm_sum=math.fsum(e["delta_norm"] for e in chosen),
                    achieved_norm_root_sum_squares=math.sqrt(math.fsum(e["delta_norm"] ** 2 for e in chosen)),
                    requested_norm_sum=math.fsum(e["requested_delta_norm"] for e in chosen),
                    requested_norm_root_sum_squares=math.sqrt(math.fsum(e["requested_delta_norm"] ** 2 for e in chosen)))
    return dict(passed=all(checks.values()), checks=checks, observed_rows=len(rows))


def audit_call(report, arrays, requests, call, direction, *, hidden_size, vocabulary_size, pad_token_id, eos_ids):
    """Audit a bound raw call. A failed/partial call remains failed, not retried."""
    direction = np.asarray(direction, dtype=np.float32)
    require(len(requests) == ROWS and direction.shape == (hidden_size,) and np.isfinite(direction).all()
        and abs(float(np.linalg.norm(direction.astype(np.float64))) - 1) < 1e-4,
        "Require all32 requests and the bound applied unit")
    require(report["schema_version"] == 1 and report["revision"] == "curiosity-coverage-engineering-v1"
        and report["call"] == call and report["requests_sha256"] == call["requests_sha256"] == digest(requests)
        and report["direction"] == array_binding(direction) and report["dose"] == DOSE
        and report["pad_token_id"] == pad_token_id and report["eos_token_ids"] == sorted(set(eos_ids))
        and report["scientific_admission"] is False, "Raw call identity/arithmetic binding differs")
    for request in requests:
        prefix = request["token_prefix"]
        require(prefix and all(type(t) is int and 0 <= t < vocabulary_size for t in prefix)
            and digest(prefix) == request["token_prefix_sha256"] and type(request["seed"]) is int
            and 0 <= request["prefix_positions"]["record"] < len(prefix)
            and request["prefix_positions"]["final"] == len(prefix) - 1, "Original request binding differs")
    timeline = audit_timeline(report, requests, vocabulary_size=vocabulary_size, pad_token_id=pad_token_id, eos_ids=eos_ids)
    online = audit_online_geometry(report, requests, hidden_size=hidden_size)
    samples = audit_samples(report, arrays, requests, direction)
    traces = audit_trace_inventory(report, requests)
    fault = call["inject_failure_at"]
    controlled = bool(fault in ("after_prefill", "after_first_decode") and report["fault_injected"] is True
        and report["error"] == dict(type="ControlledCoverageFault", message=fault)
        and report["rows"] is None and len(report["forwards"]) == (1 if fault == "after_prefill" else 2))
    rows_complete = report["rows"] is not None and len(report["rows"]) == ROWS
    needs_decode = call["stage"] == "checks" and call["condition"] in ("record_nonzero", "full_nonzero", "full_one_row")
    decoded = any(f["phase"] == "decode" and f["active_prediction_rows"][0]
        and f["geometry"][0]["dose"] != 0 and f["geometry"][0]["selected_token_count"] == 1 for f in report["forwards"])
    producer_checks = report.get("checks", {})
    producer_consistent = (isinstance(producer_checks, dict) and bool(producer_checks)
        and all(type(value) is bool for value in producer_checks.values())
        and type(report["passed"]) is bool and report["passed"] is all(producer_checks.values()))
    checks = dict(producer_flag_consistent=producer_consistent, producer_gate_passed=report["passed"] is True,
        terminal_disposition=controlled if fault else rows_complete and report["error"] is None and not report["fault_injected"],
        completed_forward_inventory=report["attempted_forwards"] == len(report["forwards"]) == report["completed_forwards"]
            and report["incomplete_forward"] is None,
        timeline=timeline["passed"], all_position_geometry=online["passed"], exact_saved_vectors=samples["passed"],
        returned_trace_inventory=controlled or traces["passed"],
        row0_nonzero_active_decode=not needs_decode or decoded,
        saved_prefill=any(s["phase"] == "prefill" and s["forward_completed"] for s in report["samples"]),
        hook_cleanup=report["hooks_clean"] is True and report["hooks_before"] == report["hooks_after"] == {})
    if fault == "after_first_decode":
        checks["actual_fault_decode_exercised"] = len(report["forwards"]) == 2 and any(report["forwards"][1]["active_prediction_rows"])
    return dict(passed=all(checks.values()), checks=checks, call_index=call["call_index"], call_id=call["call_id"],
        status="expected_fault" if controlled else "completed" if rows_complete else "execution_error",
        assigned_slots=ROWS, ordinary_completed_slots=ROWS if rows_complete else 0,
        expected_fault_slots=ROWS if controlled else 0, error=report["error"],
        timeline=timeline, online_geometry=online, saved_vectors=samples, returned_traces=traces,
        producer_passed=report["passed"], scientific_admission=False)


def identical_active_rows(left, right, indices):
    """Compare actual outputs/active predictions, excluding terminated companions."""
    if left["rows"] is None or right["rows"] is None:
        return False
    for row in indices:
        a, b = left["rows"][row], right["rows"][row]
        if {k: v for k, v in a["completion"].items() if k != "intervention_schedule"} != {
                k: v for k, v in b["completion"].items() if k != "intervention_schedule"}:
            return False
        count = len(a["completion"]["generated_token_ids"])
        if len(left["forwards"]) < count or len(right["forwards"]) < count:
            return False
        for step in range(count):
            x, y = left["forwards"][step], right["forwards"][step]
            if any(x[key][row] != y[key][row] for key in ("input_ids", "position_ids", "attention_mask_row_sha256",
                                                        "active_prediction_rows", "full_fp32_logits_sha256")):
                return False
    return True


def audit_relationships(reports):
    def same(left, right, indices=range(ROWS)):
        return left in reports and right in reports and identical_active_rows(reports[left], reports[right], indices)
    common = bool(reports)
    for report in reports.values():
        baseline = 0 if report["call"]["source_cohort"] == 0 else 13
        common &= bool(baseline in reports and report["forwards"] and reports[baseline]["forwards"]
            and [g["before"] for g in report["forwards"][0]["geometry"]]
                == [g["before"] for g in reports[baseline]["forwards"][0]["geometry"]])
    return dict(record_zero_identity=same(0, 1), full_zero_identity=same(0, 2),
        row0_only_isolation=same(0, 3, range(1, ROWS)), clean_after_prefill_fault=same(0, 7),
        clean_after_decode_fault=same(0, 9), common_prefill_before_identity=common)


def expected_call_plan(requests_by_cohort):
    """Independent enumeration of the one prospectively fixed finite inventory."""
    require(set(requests_by_cohort) == {0, 3}, "Require fixed cohorts0 and3")
    for rows in requests_by_cohort.values():
        require(len(rows) == ROWS and len({r["source_trial_id"] for r in rows}) == ROWS, "Require32 distinct original companions")
    short = ["baseline", "record_zero", "full_zero", "full_one_row", "record_nonzero", "full_nonzero",
             "full_fault_prefill", "clean_repeat_prefill", "full_fault_decode", "clean_repeat_decode"]
    specifications = [(0, 128, condition) for condition in short]
    specifications += [(cohort, 4096, condition) for cohort in (0, 3) for condition in ("baseline", "record_nonzero", "full_nonzero")]
    calls = []
    for index, (cohort, cap, condition) in enumerate(specifications):
        full = condition.startswith("full_")
        baseline = condition == "baseline" or condition.startswith("clean_repeat_")
        fault = "after_prefill" if condition == "full_fault_prefill" else "after_first_decode" if condition == "full_fault_decode" else None
        call_id = f"coverage-{index:02d}-{condition}"
        requests = requests_by_cohort[cohort]
        calls.append(dict(call_index=index, call_id=call_id, source_cohort=cohort, cap=cap,
            stage="checks" if index < 10 else "timing", condition=condition, schedule=FULL if full else RECORD,
            position=None if baseline else "all_attended" if full else "record", direction_key=None if baseline else "curiosity_negative",
            dose_multiplier=0 if condition.endswith("_zero") else 1, one_row_only=0 if condition == "full_one_row" else None,
            inject_failure_at=fault, expected_error="ControlledCoverageFault" if fault else None,
            base_trial_ids=[r["source_trial_id"] for r in requests], trial_ids=[f"{call_id}:{row:02d}" for row in range(ROWS)],
            requests_sha256=digest(requests)))
    return calls


def audit_collection(calls, reports, arrays_by_call, requests_by_cohort, direction, *, hidden_size,
                     vocabulary_size, pad_token_id, eos_ids):
    """Pure numeric receipt for all512 slots; outer provenance/receipts remain required.

The caller supplies validated immutable source requests/direction and actual
JSON/NPZ artifacts. This does not certify their filesystem/archive/admission
bindings, runtime identity, timing receipts, export or final cloud shutdown.
"""
    require(calls == expected_call_plan(requests_by_cohort), "Fixed16 call inventory differs")
    require(isinstance(reports, dict) and set(reports) == set(arrays_by_call)
        and all(type(i) is int and 0 <= i < 16 for i in reports), "Unexpected or unpaired raw call")
    results, slots = [], []
    for call in calls:
        index = call["call_index"]
        if index in reports:
            result = audit_call(reports[index], arrays_by_call[index], requests_by_cohort[call["source_cohort"]], call, direction,
                hidden_size=hidden_size, vocabulary_size=vocabulary_size, pad_token_id=pad_token_id, eos_ids=eos_ids)
        else:
            result = dict(call_index=index, call_id=call["call_id"], passed=False, status="missing", assigned_slots=ROWS,
                          ordinary_completed_slots=0, expected_fault_slots=0)
        results.append(result)
        slots.extend(dict(trial_id=trial, base_trial_id=base, call_index=index, status=result["status"])
                     for trial, base in zip(call["trial_ids"], call["base_trial_ids"]))
    relations = audit_relationships(reports)
    prefix = sorted(reports) == list(range(len(reports)))
    after_failure = any(not reports[index]["passed"] and any(later > index for later in reports) for index in reports)
    short_passed = all(result["passed"] for result in results[:10]) and all(
        audit_relationships({i: r for i, r in reports.items() if i < 10}).values())
    order = prefix and not after_failure and (not any(i >= 10 for i in reports) or short_passed)
    complete = len(reports) == 16
    checks = dict(all16_raw_calls_present=complete, one_fixed_order_no_after_failure=order,
                  short_gate_before_timing=not any(i >= 10 for i in reports) or short_passed,
                  all_call_numerics=all(r["passed"] for r in results), **relations)
    return dict(schema_version=1, assay="curiosity_coverage_engineering_numeric_audit",
        call_plan_sha256=digest(calls), direction_float32=array_binding(np.asarray(direction, np.float32)),
        complete=complete, numeric_engineering_passed=all(checks.values()), checks=checks,
        assigned_calls=16, observed_calls=len(reports), assigned_slots=512, assigned_ordinary_slots=448,
        assigned_deliberate_fault_slots=64, ordinary_completed_slots=sum(r["ordinary_completed_slots"] for r in results),
        observed_controlled_fault_slots=sum(r["expected_fault_slots"] for r in results),
        missing_slots=sum(r["status"] == "missing" for r in slots), slots=slots, calls=results,
        independently_reconstructed_selected_components=sum(r.get("saved_vectors", {}).get("reconstructed_selected_components", 0) for r in results),
        source_provenance_verified=False, runtime_verified=False, selected_receipts_verified=False,
        admission_and_shutdown_verified=False, scientific_admission=False,
        scope="Pure raw-call numerical audit only; caller must independently bind source, runtime, artifacts, actual selected receipts, admission and shutdown.",
        limitations=["Unsaved rows retain online checks and byte bindings, not offline reconstructible full vectors.",
            "Float64 norm summaries are descriptive; exact BF16 arithmetic is the saved-vector gate.",
            "Missing/error slots remain assigned. Timing outputs do not enter scientific estimators or select new inputs."])
