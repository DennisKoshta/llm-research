"""Bounded engineering evidence collector; never scientific admission.

Forward 0 predicts token 1. Forward k>0 processes generated token k and predicts
token k+1 only where the actual last attention-mask column is one. All 32 rows
remain in every ledger; inactive logits have a null fingerprint. Cache lengths
are snapshotted before the in-place cache update. Returned arrays must be saved
and byte-bound by a future caller; content hashes here bind their dtype/shape.
"""

from contextlib import ExitStack
import hashlib
import math
import os
from pathlib import Path
import time
import uuid

import numpy as np

from .artifacts import RunStore, canonical, digest, write_once
from .concept_capture import MODEL
from .state_backend import PREFILL_PADDING_MULTIPLE, StateBackend

SCHEDULE = "record_prefill_plus_decode"
ROWS = 32
LAYER = 15
CONDITIONS = ("baseline", "zero_sham", "one_row_nonzero", "clean_repeat")
EXECUTION = dict(backend="torch_batched_v1", batch_size=ROWS)
ARITHMETIC = dict(
    parameter_dtype="bfloat16",
    patch_dtype="float32",
    attention="eager",
    prefill_padding_multiple=PREFILL_PADDING_MULTIPLE,
    temperature=0.7,
    top_p=1.0,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def array_binding(value):
    value = np.ascontiguousarray(value)
    return dict(
        dtype=str(value.dtype),
        shape=list(value.shape),
        byte_sha256=hashlib.sha256(value.tobytes()).hexdigest(),
    )


def hook_inventory(model):
    return {
        name: dict(pre=list(module._forward_pre_hooks), post=list(module._forward_hooks))
        for name, module in model.model.named_modules()
        if module._forward_pre_hooks or module._forward_hooks
    }


def _collect(model, requests, patches, cap, *, vectors=False, expected_error=False, deadline=None):
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
                (model.layers[LAYER].register_forward_hook, pre_patch, {}),
                (model.layers[LAYER + 1].register_forward_pre_hook, post_patch, {}),
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


def validate_condition(condition, requests, patches, *, cap, pad_token_id, eos_ids):
    """Reconstruct every model input and event timeline from bound prefixes/tokens."""
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
                and event["layer"] == LAYER
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


def identical_rows(left, right, indices):
    for row in indices:
        if left["rows"][row]["completion"] != right["rows"][row]["completion"]:
            return False

        def predictions(condition):
            return [
                dict(
                    forward_index=frame["forward_index"],
                    prediction_ordinal=frame["prediction_ordinal"],
                    logits=frame["full_fp32_logits_sha256"][row],
                    input_ids=frame["input_ids"][row],
                    position_ids=frame["position_ids"][row],
                    mask=frame["attention_mask_row_sha256"][row],
                )
                for frame in condition["forwards"]
                if frame["active_prediction_rows"][row]
            ]

        if predictions(left) != predictions(right):
            return False
    return True


def check_geometry(condition, arrays, direction, dose, torch):
    before, actual = arrays["before"], arrays["after"]
    events = condition["rows"][0]["intervention_trace"]["events"]
    if before.shape != actual.shape or before.shape != (len(events), len(direction)):
        return False
    unit = torch.tensor(direction, dtype=torch.float32)
    norm = unit.norm()
    for index, event in enumerate(events):
        a, b = torch.tensor(before[index]), torch.tensor(actual[index])
        requested = a + float(dose) * unit
        dtype_name = condition["vector_activation_dtypes"][index]
        dtype = {"torch.bfloat16": torch.bfloat16, "torch.float32": torch.float32}.get(dtype_name)
        if dtype is None or not torch.equal(b, requested.to(dtype).float()):
            return False
        expected = dict(
            before_norm=a.norm().item(),
            after_norm=b.norm().item(),
            delta_norm=(b - a).norm().item(),
            requested_delta_norm=(requested - a).norm().item(),
            rounding_error_norm=(b - requested).norm().item(),
            relative_delta_norm=(b - a).norm().item() / max(a.norm().item(), 1e-30),
            direction_norm=norm.item(),
            projection_before=torch.dot(a, unit / norm).item(),
            projection_after=torch.dot(b, unit / norm).item(),
        )
        if not all(
            math.isclose(event[key], value, rel_tol=2e-5, abs_tol=1e-6) for key, value in expected.items()
        ):
            return False
        site = condition["vector_sites"][index]
        if site != dict(
            forward_index=index,
            phase=event["phase"],
            prediction_ordinal=index + 1,
            tensor_token_index=event["token_index"],
        ):
            return False
    return bool(
        np.isfinite(before).all()
        and np.isfinite(actual).all()
        and any(event["delta_norm"] > 0 for event in events)
    )


def run_engineering(model, requests, *, direction, dose, cap, layer=LAYER, deadline=None):
    """Return (JSON-ready evidence, arrays); this function grants no admission."""
    require(
        layer == LAYER and len(model.layers) > LAYER + 1,
        "Require layer15 plus downstream layer16 observation",
    )
    require(type(cap) is int and 1 <= cap <= 4096, "Engineering cap must be between one and 4096")
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
        and math.isfinite(float(dose))
        and float(dose) != 0,
        "Require finite locked unit direction and nonzero dose",
    )
    bound = digest(requests)
    zero = dict(layer=LAYER, position="record", mode="add", direction=effective, dose=0.0)
    baseline = [None] * ROWS
    sham = [dict(zero) for _ in requests]
    changed = [{**zero, "dose": float(dose)}] + [dict(zero) for _ in requests[1:]]
    conditions = {}
    for name, patches in (("baseline", baseline), ("zero_sham", sham), ("one_row_nonzero", changed)):
        conditions[name], values = _collect(
            model, requests, patches, cap, vectors=name == "one_row_nonzero", deadline=deadline
        )
        if values:
            arrays = values
    malformed, _ = _collect(
        model,
        requests,
        [{**zero, "direction": [1.0]}] + sham[1:],
        cap,
        expected_error=True,
        deadline=deadline,
    )
    conditions["clean_repeat"], _ = _collect(model, requests, baseline, cap, deadline=deadline)
    eos = model.model.generation_config.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos or [])
    checks = {
        f"{name}_timeline": validate_condition(
            value, requests, patches, cap=cap, pad_token_id=model.tokenizer.pad_token_id, eos_ids=eos
        )
        for name, value, patches in (
            ("baseline", conditions["baseline"], baseline),
            ("zero_sham", conditions["zero_sham"], sham),
            ("one_row_nonzero", conditions["one_row_nonzero"], changed),
            ("clean_repeat", conditions["clean_repeat"], baseline),
        )
    }
    checks.update(
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
            "Passing a 128-token allowance cannot certify 4096-token scientific use; a separately bound full-cap baseline/sham check remains required.",
            "Full FP32 vocabulary fingerprints cover active predictions only; terminal dummy rows remain in the layout with null fingerprints.",
            "Residual arrays contain only the nonzero row in the perturbed condition; norm sums are exposure, not net displacement.",
            "Immediate termination or a one-token cap cannot pass sustained engineering; no replacement row is selected after observing termination.",
            "The pure run_engineering harness writes no artifacts; execute persists its arrays and binds actual archive bytes before audit.",
        ],
    )
    return report, arrays


def engine_hashes():
    root = Path(__file__).parent
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("sustained_engineering.py", "state_backend.py", "model.py", "batched.py")
    }


def assignment_plan(requests, cap):
    return [
        dict(
            trial_id=f"sustained-{cap}-{condition}-{row:02d}",
            kind="engineering",
            condition=condition,
            source_row=row,
            source_trial_id=request["source_trial_id"],
            request_sha256=digest(request),
            cap=cap,
        )
        for condition in CONDITIONS
        for row, request in enumerate(requests)
    ]


def compatibility_binding(manifest):
    """The short and long stages may differ only in cap and stage documents."""
    return {
        key: manifest[key]
        for key in (
            "model",
            "execution",
            "execution_gpu",
            "arithmetic",
            "schedule",
            "layer",
            "requests",
            "source_binding",
            "direction",
            "engine_sha256",
        )
    }


def validate_manifest(manifest):
    require(
        manifest.get("assay") == "sustained_engineering"
        and manifest.get("phase") == "engineering"
        and manifest.get("scientific_admission") is False
        and manifest.get("model") == MODEL
        and manifest.get("execution") == EXECUTION
        and manifest.get("execution_gpu") == "H200"
        and manifest.get("arithmetic") == ARITHMETIC
        and manifest.get("schedule") == SCHEDULE
        and manifest.get("layer") == LAYER
        and type(manifest.get("cap")) is int
        and manifest["cap"] in (128, 4096),
        "Unsupported production sustained-engineering contract",
    )
    requests, direction = manifest["requests"], manifest["direction"]
    require(
        len(requests) == ROWS
        and len({row["source_trial_id"] for row in requests}) == ROWS
        and all(
            isinstance(row["source_trial_id"], str)
            and row["source_trial_id"]
            and type(row["seed"]) is int
            and row["labels"] == list("ABCD")
            and row["token_prefix_sha256"] == digest(row["token_prefix"])
            and all(type(token) is int and token >= 0 for token in row["token_prefix"])
            and set(row["prefix_positions"]) == {"record", "final"}
            and 0
            <= row["prefix_positions"]["record"]
            < row["prefix_positions"]["final"]
            == len(row["token_prefix"]) - 1
            for row in requests
        ),
        "Require the exact32 distinct original source prompts and seeds",
    )
    unit = np.asarray(direction["unit"], dtype=direction["unit_dtype"])
    require(
        direction["source"] == "curiosity_vs_indifference"
        and direction["corpus"] == "A"
        and direction["layer"] == LAYER
        and direction["position"] == "record"
        and direction["multiplier"] == 1.0
        and direction["dose"] == direction["norm"]
        and math.isfinite(direction["norm"])
        and direction["norm"] > 0
        and unit.ndim == 1
        and unit.size
        and np.isfinite(unit).all()
        and math.isclose(float(np.linalg.norm(unit)), 1, abs_tol=1e-6)
        and array_binding(unit) == direction["unit_binding"],
        "Require the unchanged positive A15/record curiosity unit and raw contrast norm",
    )
    expected = assignment_plan(requests, manifest["cap"])
    require(
        manifest.get("assignments") == []
        and manifest.get("diagnostics") == expected
        and manifest.get("assignment_sha256") == digest(expected)
        and manifest.get("compatibility_sha256") == digest(compatibility_binding(manifest))
        and manifest.get("engine_sha256") == engine_hashes(),
        "Sustained assignment, compatibility or execution bytes changed",
    )
    short = manifest.get("short_stage")
    if manifest["cap"] == 128:
        require(short is None, "The short stage cannot use a replacement engineering certificate")
    else:
        require(
            isinstance(short, dict)
            and short["audit_sha256"] == digest(short["audit"])
            and short["audit"].get("assay") == "sustained_engineering_audit"
            and short["audit"].get("passed") is True
            and short["audit"].get("complete") is True
            and short["audit"].get("scientific_admission") is False
            and short["audit"].get("cap") == 128
            and short["audit"].get("manifest_sha256") == short["manifest_sha256"]
            and short["audit"].get("compatibility_sha256") == manifest["compatibility_sha256"]
            and all(
                short["audit"].get(key, {}).get("sha256") for key in ("report_artifact", "vector_artifact")
            ),
            "Full-cap engineering requires the exact independently passed short stage",
        )
    return expected


def execute(root, manifest, model, checkpoint=None, deadline=None):
    """Publish terminal engineering dispositions; never rerun an observed attempt."""
    assignments = validate_manifest(manifest)
    root = Path(root)
    store = RunStore(root, manifest)
    require(
        not (root / "engineering_started.json").exists()
        and not store.completed()
        and not any((root / "engineering").glob("*/*"))
        and len(list((root / "admissions").glob("*.json"))) <= 1,
        "A started or partial engineering stage cannot be rerun",
    )
    attempt, started = uuid.uuid4().hex, time.monotonic()
    write_once(root / "engineering_started.json", dict(manifest_sha256=digest(manifest), attempt=attempt))
    environment, harness, artifact, error = None, None, None, None
    try:
        if deadline is not None and time.time() >= deadline:
            raise TimeoutError("Engineering deadline elapsed before execution")
        environment = model.metadata()
        write_once(root / "environments" / f"{attempt}.json", environment)
        require(
            all(environment.get(key) == value for key, value in manifest["model"].items())
            and environment.get("dtype") == "bfloat16"
            and environment.get("attention") == "eager",
            "Actual engineering model or arithmetic differs from the frozen contract",
        )
        direction = manifest["direction"]
        harness, arrays = run_engineering(
            model,
            manifest["requests"],
            direction=np.asarray(direction["unit"], dtype=direction["unit_dtype"]),
            dose=direction["dose"],
            cap=manifest["cap"],
            deadline=deadline,
        )
        relative = Path("engineering") / attempt / "vectors.npz"
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        try:
            with temporary.open("xb") as handle:
                np.savez_compressed(handle, **arrays)
                handle.flush()
                os.fsync(handle.fileno())
            require(not destination.exists(), "Engineering vectors already exist")
            os.rename(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        artifact = dict(
            path=str(relative),
            sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
            byte_count=destination.stat().st_size,
            array_bindings={key: array_binding(value) for key, value in arrays.items()},
        )
    except Exception as caught:
        error = dict(type=type(caught).__name__, message=str(caught))
    peak_allocated, peak_reserved, memory_error = None, None, None
    if str(getattr(model, "device", "")).startswith("cuda"):
        try:
            peak_allocated = int(model.torch.cuda.max_memory_allocated())
            peak_reserved = int(model.torch.cuda.max_memory_reserved())
        except Exception as caught:
            memory_error = f"{type(caught).__name__}: {caught}"
    forward_counts = (
        None
        if harness is None
        else {name: value["attempted_forwards"] for name, value in harness["conditions"].items()}
    )
    token_counts = (
        None
        if harness is None
        else {
            name: sum(len(row["completion"]["generated_token_ids"]) for row in value["rows"])
            for name, value in harness["conditions"].items()
        }
    )
    elapsed = time.monotonic() - started
    wrapper = dict(
        schema_version=1,
        assay="sustained_engineering",
        manifest_sha256=digest(manifest),
        compatibility_sha256=manifest["compatibility_sha256"],
        cap=manifest["cap"],
        status="measured" if error is None else "execution_error",
        engineering_passed=error is None and harness["passed"] is True,
        scientific_admission=False,
        harness=harness,
        error=error,
        vector_artifact=artifact,
        environment_sha256=None if environment is None else digest(environment),
        elapsed_seconds=elapsed,
        performance=dict(
            elapsed_seconds=elapsed,
            forward_counts_by_condition=forward_counts,
            total_forward_count=None if forward_counts is None else sum(forward_counts.values()),
            generated_tokens_by_condition=token_counts,
            peak_cuda_allocated_bytes=peak_allocated,
            peak_cuda_reserved_bytes=peak_reserved,
            memory_scope="Worker peak including model loading and startup; peak counters are not reset",
            memory_query_error=memory_error,
            harness_json_bytes=0 if harness is None else len(canonical(harness).encode()),
            vector_file_bytes=0 if artifact is None else artifact["byte_count"],
        ),
    )
    relative = Path("engineering") / attempt / "report.json"
    write_once(root / relative, wrapper)
    report_artifact = dict(
        path=str(relative),
        sha256=hashlib.sha256((root / relative).read_bytes()).hexdigest(),
        byte_count=(root / relative).stat().st_size,
    )
    for offset in range(0, len(assignments), ROWS):
        records = []
        for assignment in assignments[offset : offset + ROWS]:
            row = (
                None
                if harness is None
                else harness["conditions"][assignment["condition"]]["rows"][assignment["source_row"]]
            )
            record = dict(
                trial_id=assignment["trial_id"],
                assignment=assignment,
                kind="engineering",
                terminal=True,
                condition=assignment["condition"],
                source_row=assignment["source_row"],
                request_sha256=assignment["request_sha256"],
                status=wrapper["status"],
                engineering_passed=wrapper["engineering_passed"],
                scientific_admission=False,
                condition_row_sha256=None if row is None else digest(row),
                report_artifact=report_artifact,
                activation_artifact=artifact,
            )
            if error is not None:
                record.update(failure="execution_error", error_type=error["type"], error=error["message"])
            records.append(record)
        store.save_many(records, attempt, f"{offset:07d}")
        if checkpoint:
            checkpoint()
    # Return failed engineering evidence too: the generic exporter must still run.
    # Operational coverage rejects execution_error; independent audit rejects failed checks.
    return store.completed()


def render(manifest, records):
    passed = len(records) == 128 and all(row.get("engineering_passed") is True for row in records.values())
    return (
        f"Sustained engineering: {len(records)}/128 terminal dispositions; "
        f"cap {manifest['cap']}; recorded checks passed: {passed}.\n\n"
        "A separate artifact audit is required. This collection grants no scientific admission."
    )
