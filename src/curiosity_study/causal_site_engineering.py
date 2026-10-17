"""Explicit-site engineering collectors for the separately declared S extension.

The layer-dependent observer and timeline routines are adapted from the frozen
sustained-engineering implementation, without mutating its constants. Generic
array fingerprints, geometry reconstruction and row-identity comparisons are
reused unchanged. Short engineering is limited to the actually observed tokens;
it does not certify full4096 behavior or grant scientific admission.
"""

from contextlib import ExitStack
import math
import time

import numpy as np

from .artifacts import digest
from .state_backend import PREFILL_PADDING_MULTIPLE, StateBackend
from .sustained_engineering import (
    ARITHMETIC as ARITHMETIC,
    ROWS,
    SCHEDULE,
    array_binding,
    check_geometry,
    hook_inventory,
    identical_rows,
    require,
)

SITES = (31, 47)


def collect_condition(
    model, requests, patches, *, layer, cap, vectors=False, expected_error=False, deadline=None
):
    require(
        type(layer) is int and layer in SITES and len(model.layers) > layer + 1,
        "Require site31/47 and an independent downstream observation",
    )
    require(type(cap) is int and 1 <= cap <= 4096, "Require a positive bounded generation allowance")
    require(len(requests) == len(patches) == ROWS, "Require all32 actual rows and patch dispositions")
    require(
        all(
            p is None
            or (p.get("layer") == layer and p.get("position") == "record" and p.get("mode") == "add")
            for p in patches
        ),
        "Every patch must use the explicit observed layer and additive record schedule",
    )
    started = time.monotonic()
    frames, before, after, vector_sites = [], [], [], []
    pending, activation_dtypes = {}, []
    attempted_forwards = 0
    initial_hooks = hook_inventory(model)
    backend = StateBackend(model)
    _, _, _, targets = backend.prepare(requests)
    record_position = targets["record"][0]

    def pre_model(module, args, kwargs):
        nonlocal attempted_forwards
        if deadline is not None and time.time() >= deadline:
            raise TimeoutError("Sustained engineering deadline reached before the next forward")
        require(not pending, "Nested or unfinished model forward")
        attempted_forwards += 1
        tokens = kwargs["input_ids"].detach().cpu().tolist()
        mask = kwargs["attention_mask"].detach().cpu().tolist()
        positions = kwargs["position_ids"].detach().cpu().tolist()
        cache = kwargs.get("past_key_values")
        pending.update(
            forward_index=len(frames),
            phase="prefill" if not frames else "decode",
            prediction_ordinal=len(frames) + 1,
            input_ids=tokens,
            position_ids=positions,
            attention_mask_shape=list(kwargs["attention_mask"].shape),
            attention_mask_row_sha256=[digest(row) for row in mask],
            active_prediction_rows=[bool(row[-1]) for row in mask],
            cache_length_before=0 if cache is None else int(cache.get_seq_length()),
            cache_position_supplied="cache_position" in kwargs,
        )
        require("cache_position" not in kwargs, "Unexpected explicit cache-position policy")

    def pre_patch(module, args, output):
        if vectors and pending["active_prediction_rows"][0]:
            tensor = output[0] if isinstance(output, tuple) else output
            position = record_position if pending["phase"] == "prefill" else 0
            before.append(tensor[0, position].detach().float().cpu().numpy().copy())
            activation_dtypes.append(str(tensor.dtype))
            vector_sites.append(
                dict(
                    forward_index=pending["forward_index"],
                    phase=pending["phase"],
                    prediction_ordinal=pending["prediction_ordinal"],
                    tensor_token_index=position,
                )
            )

    def post_patch(module, args):
        if vectors and pending["active_prediction_rows"][0]:
            position = record_position if pending["phase"] == "prefill" else 0
            after.append(args[0][0, position].detach().float().cpu().numpy().copy())

    def post_model(module, args, kwargs, output):
        logits = output.logits[:, -1].detach().float().cpu().numpy()
        require(logits.shape == (ROWS, model.model.config.vocab_size), "Wrong full-vocabulary logit layout")
        active = pending["active_prediction_rows"]
        require(np.isfinite(logits[np.asarray(active)]).all(), "Nonfinite active prediction logits")
        frames.append(
            dict(
                **pending,
                full_fp32_logits_sha256=[
                    digest(array_binding(logits[row])) if active[row] else None for row in range(ROWS)
                ],
                logit_dtype="float32",
                vocabulary_size=logits.shape[-1],
                cache_length_after=int(output.past_key_values.get_seq_length()),
            )
        )
        pending.clear()

    rows, error = None, None
    try:
        with ExitStack() as stack:
            # This observer precedes StateBackend's patch hook. The next block's
            # input independently exposes the actual value returned by the patch.
            for register, observer, kwargs in (
                (model.model.register_forward_pre_hook, pre_model, dict(with_kwargs=True)),
                (model.layers[layer].register_forward_hook, pre_patch, {}),
                (model.layers[layer + 1].register_forward_pre_hook, post_patch, {}),
                (model.model.register_forward_hook, post_model, dict(with_kwargs=True)),
            ):
                handle = register(observer, **kwargs)
                stack.callback(handle.remove)
            rows, _ = backend.forward(
                requests,
                interventions=patches,
                max_new_tokens=cap,
                greedy=False,
                intervention_schedule=SCHEDULE,
            )
    except Exception as caught:
        if not expected_error:
            raise
        error = dict(type=type(caught).__name__, message=str(caught))
    finally:
        require(
            model.active_hooks == 0 and hook_inventory(model) == initial_hooks,
            "Engineering observer or intervention hook leaked",
        )
    summary = dict(
        rows=None
        if rows is None
        else [
            {
                key: row[key]
                for key in (
                    "token_prefix_sha256",
                    "intervention",
                    "intervention_trace",
                    "completion",
                    "hooks_remaining",
                )
            }
            for row in rows
        ],
        forwards=frames,
        attempted_forwards=attempted_forwards,
        error=error,
        hooks_clean=True,
        vector_sites=vector_sites,
        vector_activation_dtypes=activation_dtypes,
        elapsed_seconds=time.monotonic() - started,
    )
    arrays = (
        {}
        if not vectors
        else dict(before=np.asarray(before, dtype=np.float32), after=np.asarray(after, dtype=np.float32))
    )
    require(
        not vectors or len(before) == len(after) == len(vector_sites), "Incomplete before/after observation"
    )
    return summary, arrays


def validate_condition(condition, requests, patches, *, layer, cap, pad_token_id, eos_ids):
    """Reconstruct every model input and event timeline from bound prefixes/tokens."""
    if type(layer) is not int or layer not in SITES or len(requests) != ROWS or len(patches) != ROWS:
        return False
    if any(
        p is not None and (p.get("layer") != layer or p.get("position") != "record" or p.get("mode") != "add")
        for p in patches
    ):
        return False
    rows, frames = condition["rows"], condition["forwards"]
    if rows is None or len(rows) != ROWS:
        return False
    prefixes = [request["token_prefix"] for request in requests]
    width = (
        (max(map(len, prefixes)) + PREFILL_PADDING_MULTIPLE - 1) // PREFILL_PADDING_MULTIPLE
    ) * PREFILL_PADDING_MULTIPLE
    sequences = [row["completion"]["generated_token_ids"] for row in rows]
    if len(frames) != condition["attempted_forwards"] or len(frames) != max(map(len, sequences)):
        return False
    for row, (request, result, tokens, patch) in enumerate(zip(requests, rows, sequences, patches)):
        completion, trace = result["completion"], result["intervention_trace"]
        if not (
            1 <= len(tokens) <= cap
            and not any(token in eos_ids for token in tokens[:-1])
            and completion["ended_with_eos"] is (tokens[-1] in eos_ids)
            and (tokens[-1] in eos_ids or len(tokens) == cap)
            and result["token_prefix_sha256"] == request["token_prefix_sha256"]
            and result["hooks_remaining"] == 0
            and trace["schedule"] == SCHEDULE
            and completion["sampling"]
            == dict(seed=request["seed"], temperature=0.7, top_p=1.0, max_new_tokens=cap)
            and trace["generated_token_count"] == len(tokens)
            and trace["active_decode_forward_count"] == len(tokens) - 1
            and trace["prefill_patch_count"] == int(patch is not None)
            and trace["decode_patch_count"] == (len(tokens) - 1 if patch is not None else 0)
            and trace["total_patch_count"]
            == len(trace["events"])
            == (len(tokens) if patch is not None else 0)
        ):
            return False
        for step, event in enumerate(trace["events"]):
            unpadded = request["prefix_positions"]["record"] if step == 0 else len(prefixes[row]) + step - 1
            if not (
                event["phase"] == ("prefill" if step == 0 else "decode")
                and event["layer"] == layer
                and event["mode"] == "add"
                and event["position"] == ("record" if step == 0 else "current_generated_token")
                and event["processed_generated_token_index"] == (None if step == 0 else step)
                and event["processed_token_id"]
                == (prefixes[row][unpadded] if step == 0 else tokens[step - 1])
                and event["unpadded_token_index"] == unpadded
                and event["token_index"] == (width - len(prefixes[row]) + unpadded if step == 0 else 0)
                and event["requested_scalar_delta"] == patch["dose"]
                and all(
                    math.isfinite(event[key])
                    for key in (
                        "before_norm",
                        "after_norm",
                        "delta_norm",
                        "requested_delta_norm",
                        "rounding_error_norm",
                        "relative_delta_norm",
                        "projection_before",
                        "projection_after",
                        "direction_norm",
                    )
                )
                and (patch["dose"] != 0 or event["delta_norm"] == 0)
            ):
                return False
    for step, frame in enumerate(frames):
        expected_inputs, expected_positions, expected_masks = [], [], []
        for prefix, tokens in zip(prefixes, sequences):
            mask = (
                [0] * (width - len(prefix))
                + [1] * len(prefix)
                + [int(k < len(tokens)) for k in range(1, step + 1)]
            )
            expected_masks.append(mask)
            expected_inputs.append(
                [pad_token_id] * (width - len(prefix)) + prefix
                if step == 0
                else [tokens[step - 1] if step <= len(tokens) else pad_token_id]
            )
            expected_positions.append(
                [0] * (width - len(prefix)) + list(range(len(prefix)))
                if step == 0
                else [max(sum(mask) - 1, 0)]
            )
        active = [step < len(tokens) for tokens in sequences]
        fingerprints = frame["full_fp32_logits_sha256"]
        if not (
            frame["forward_index"] == step
            and frame["prediction_ordinal"] == step + 1
            and frame["phase"] == ("prefill" if step == 0 else "decode")
            and frame["input_ids"] == expected_inputs
            and frame["position_ids"] == expected_positions
            and frame["attention_mask_shape"] == [ROWS, width + step]
            and frame["attention_mask_row_sha256"] == [digest(mask) for mask in expected_masks]
            and frame["active_prediction_rows"] == active
            and len(fingerprints) == ROWS
            and all(
                (isinstance(value, str) and len(value) == 64) if live else value is None
                for value, live in zip(fingerprints, active)
            )
            and frame["logit_dtype"] == "float32"
            and not frame["cache_position_supplied"]
            and frame["cache_length_before"] == (0 if step == 0 else width + step - 1)
            and frame["cache_length_after"] == width + step
        ):
            return False
    return condition["error"] is None and condition["hooks_clean"] is True


def run_engineering(model, requests, *, layer, direction, dose, cap=128, deadline=None):
    """Return (JSON-ready evidence, arrays); this function grants no admission."""
    require(
        type(layer) is int and layer in SITES and len(model.layers) > layer + 1,
        "Require specified site31/47 and its next-block input observation",
    )
    require(type(cap) is int and 1 <= cap <= 128, "Engineering cap must be between one and 128")
    require(
        len(requests) == ROWS and len({row.get("source_trial_id") for row in requests}) == ROWS,
        "Require 32 distinct bound source rows",
    )
    require(
        all(
            isinstance(row.get("source_trial_id"), str)
            and row["source_trial_id"]
            and type(row.get("seed")) is int
            and set(row.get("prefix_positions", {})) == {"record", "final"}
            and row["token_prefix_sha256"] == digest(row["token_prefix"])
            and 0
            <= row["prefix_positions"]["record"]
            < row["prefix_positions"]["final"]
            == len(row["token_prefix"]) - 1
            for row in requests
        ),
        "Malformed exact request binding",
    )
    require(
        model.active_hooks == 0 and not hook_inventory(model),
        "Engineering requires a clean model without existing hooks",
    )
    supplied_direction = np.asarray(direction)
    effective = np.asarray(direction, dtype=np.float32)
    require(
        effective.shape == (model.model.config.hidden_size,)
        and np.isfinite(effective).all()
        and math.isclose(float(np.linalg.norm(effective)), 1, abs_tol=1e-4)
        and type(dose) in (int, float)
        and math.isfinite(float(dose))
        and float(dose) > 0,
        "Require finite locked unit direction and positive dose",
    )
    bound = digest(requests)
    zero = dict(layer=layer, position="record", mode="add", direction=effective, dose=0.0)
    baseline = [None] * ROWS
    sham = [dict(zero) for _ in requests]
    changed = [{**zero, "dose": float(dose)}] + [dict(zero) for _ in requests[1:]]
    conditions = {}
    for name, patches in (("baseline", baseline), ("zero_sham", sham), ("one_row_nonzero", changed)):
        conditions[name], values = collect_condition(
            model,
            requests,
            patches,
            layer=layer,
            cap=cap,
            vectors=name == "one_row_nonzero",
            deadline=deadline,
        )
        if values:
            arrays = values
    malformed, _ = collect_condition(
        model,
        requests,
        [{**zero, "direction": [1.0]}] + sham[1:],
        layer=layer,
        cap=cap,
        expected_error=True,
        deadline=deadline,
    )
    conditions["clean_repeat"], _ = collect_condition(
        model, requests, baseline, layer=layer, cap=cap, deadline=deadline
    )
    eos = model.model.generation_config.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos or [])
    checks = {
        f"{name}_timeline": validate_condition(
            value,
            requests,
            patches,
            layer=layer,
            cap=cap,
            pad_token_id=model.tokenizer.pad_token_id,
            eos_ids=eos,
        )
        for name, value, patches in (
            ("baseline", conditions["baseline"], baseline),
            ("zero_sham", conditions["zero_sham"], sham),
            ("one_row_nonzero", conditions["one_row_nonzero"], changed),
            ("clean_repeat", conditions["clean_repeat"], baseline),
        )
    }
    checks.update(
        every_perturbed_event_engaged=all(
            math.isfinite(event["delta_norm"])
            and event["delta_norm"] > 0
            and math.isfinite(event["projection_after"])
            and math.isfinite(event["projection_before"])
            and event["projection_after"] > event["projection_before"]
            for event in conditions["one_row_nonzero"]["rows"][0]["intervention_trace"]["events"]
        ),
        bfloat16_vector_observation=bool(conditions["one_row_nonzero"]["vector_activation_dtypes"])
        and all(
            dtype == "torch.bfloat16" for dtype in conditions["one_row_nonzero"]["vector_activation_dtypes"]
        ),
        decode_observed_in_every_condition=all(
            any(
                frame["phase"] == "decode" and any(frame["active_prediction_rows"])
                for frame in value["forwards"]
            )
            for value in conditions.values()
        ),
        nonzero_decode_engagement=any(
            event["phase"] == "decode" and event["delta_norm"] > 0
            for event in conditions["one_row_nonzero"]["rows"][0]["intervention_trace"]["events"]
        ),
        full_sham_identity=identical_rows(conditions["baseline"], conditions["zero_sham"], range(ROWS)),
        unaffected31_identity=identical_rows(
            conditions["baseline"], conditions["one_row_nonzero"], range(1, ROWS)
        ),
        cleanup_repeat_identity=identical_rows(
            conditions["baseline"], conditions["clean_repeat"], range(ROWS)
        ),
        malformed_rejected_cleanly=malformed["error"] is not None
        and malformed["error"]["type"] == "ValueError"
        and malformed["error"]["message"] == "Invalid sustained direction or dose"
        and not malformed["forwards"]
        and malformed["attempted_forwards"] == 0
        and malformed["hooks_clean"],
        independently_recomputed_geometry=check_geometry(
            conditions["one_row_nonzero"], arrays, effective, dose, model.torch
        ),
        unchanged_requests=digest(requests) == bound,
        hook_cleanup=model.active_hooks == 0 and not hook_inventory(model),
    )
    arrays["direction_float32"] = effective.copy()
    bindings = {name: array_binding(value) for name, value in arrays.items()}
    report = dict(
        schema_version=1,
        passed=all(checks.values()),
        checks=checks,
        scientific_admission=False,
        full_cap_certified=False,
        assay="causal_site_engineering",
        schedule=SCHEDULE,
        layer=layer,
        cap=cap,
        batch_size=ROWS,
        perturbed_row=0,
        request_sha256=bound,
        source_trial_ids=[row["source_trial_id"] for row in requests],
        seeds=[row["seed"] for row in requests],
        eos_token_ids=sorted(eos),
        pad_token_id=model.tokenizer.pad_token_id,
        prefill_padding_multiple=PREFILL_PADDING_MULTIPLE,
        dose=float(dose),
        supplied_direction=array_binding(supplied_direction),
        effective_direction=bindings["direction_float32"],
        model=dict(
            model_id=getattr(model, "model_id", None),
            revision=getattr(model, "revision", None),
            model_type=model.model.config.model_type,
            hidden_size=model.model.config.hidden_size,
            vocabulary_size=model.model.config.vocab_size,
            layers=len(model.layers),
            device=str(model.device),
            parameter_dtype=str(next(model.model.parameters()).dtype),
        ),
        conditions=conditions,
        malformed_direction=malformed,
        arrays=bindings,
        arrays_sha256=digest(bindings),
        observed_baseline_max_tokens=max(
            len(row["completion"]["generated_token_ids"]) for row in conditions["baseline"]["rows"]
        ),
        observed_nonzero_row_tokens=len(
            conditions["one_row_nonzero"]["rows"][0]["completion"]["generated_token_ids"]
        ),
        limits=[
            "Only the supplied model, layer, exact rows, seeds, dose and observed trajectory are exercised; no source-provenance or candidate-selection admission is performed.",
            "Only observed trajectories up to128 tokens at this exact site are checked; this is not full4096 new-site certification. Full scientific baseline/sham trajectories remain independently required by the future study audit.",
            "Full FP32 vocabulary fingerprints cover active predictions only; terminal dummy rows remain in the layout with null fingerprints.",
            "Residual arrays contain only the nonzero row in the perturbed condition; norm sums are exposure, not net displacement.",
            "Immediate termination or a one-token cap cannot pass sustained engineering; no replacement row is selected after observing termination.",
            "This collector writes no artifacts; a future wrapper must persist and byte-bind all arrays, source records, model/runtime policy and this implementation before admission.",
        ],
    )
    return report, arrays
