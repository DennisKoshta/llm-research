"""Finite, immutable setup/benchmark calls for the reviewed pain-profile engine.

No model loading, cloud admission, retries, task continuation, or button effects.
The caller supplies the verified H100 model and receives a durable phase summary.
Scientific outcomes never gate engineering setup or the completion forecast.
"""

from copy import deepcopy
import hashlib
import math
from pathlib import Path
import re
import time
import uuid

import numpy as np
import torch

from . import pain_profile_design as design
from . import pain_profile_engine as engine
from . import pain_profile_resources as resources
from .artifacts import digest, write_once

require = engine.require


def setup_call_plan(plan, preflight):
    """All21 declared engineering calls, including faults, before any forward."""
    design.validate_plan(plan)
    design.validate_preflight(plan, preflight)
    longest = preflight["longest_source_cohort"]
    require(type(longest) is int and 0 < longest <= 12, "Require distinct normal/longest cohorts")
    calls, first = [], {}

    def add(cohort, arm, role, *, reference=None, fault=None, **options):
        rows = [row for row in plan["base_requests"] if row["source_cohort"] == cohort]
        call_id = f"setup-{len(calls):04d}"
        call = dict(
            call_index=len(calls), call_id=call_id, source_cohort=cohort, arm=arm, role=role,
            reference_call_id=reference, fault_call_id=fault,
            steering_enabled=True, one_row_only=None, inject_failure_at=None,
            expected_error=False, base_request_ids=[row["base_request_id"] for row in rows],
            trial_ids=[f"{call_id}/{row['base_request_id']}" for row in rows],
        )
        call.update(options)
        calls.append(call)
        return call_id

    for cohort in (0, longest):
        for arm in design.ARMS:
            call_id = add(cohort, arm, "primary", representative="first" if cohort == 0 else "longest")
            if cohort == 0:
                first[arm] = call_id
    for arm in design.ARMS:
        add(0, arm, "repeat", reference=first[arm])
    add(0, "baseline", "unhooked", reference=first["baseline"], steering_enabled=False)
    add(0, "pain_full", "one_row", reference=first["baseline"], one_row_only=0)
    for fault in ("after_prefill", "after_first_decode"):
        failed = add(0, "pain_full", "fault", reference=first["pain_full"],
                     inject_failure_at=fault, expected_error=True)
        add(0, "baseline", "clean_repeat", reference=first["baseline"], fault=failed)
    require(len(calls) == 21 and sum(len(row["trial_ids"]) for row in calls) == 612,
            "Unexpected fixed setup denominator")
    return calls


def setup_calls(manifest):
    return setup_call_plan(manifest["plan"], manifest["preflight"])


def benchmark_calls(manifest):
    design.validate_plan(manifest["plan"])
    return [dict(
        call_index=index, call_id=f"benchmark-{index:04d}", source_cohort=row["source_cohort"],
        arm=row["arm"], role="benchmark", reference_call_id=None, fault_call_id=None,
        steering_enabled=True, one_row_only=None, inject_failure_at=None, expected_error=False,
        base_request_ids=list(row["base_request_ids"]), trial_ids=list(row["trial_ids"]),
    ) for index, row in enumerate(manifest["plan"]["execution_cohorts"])]


def validate_manifest(manifest, tokenizer=None):
    require(manifest.get("assay") == resources.ASSAY and manifest.get("revision") == design.REVISION,
            "Wrong campaign assay/revision")
    design.validate_plan(manifest["plan"])
    design.validate_preflight(manifest["plan"], manifest["preflight"], tokenizer=tokenizer)
    require(manifest["plan_sha256"] == digest(manifest["plan"])
            and manifest["preflight_sha256"] == digest(manifest["preflight"]), "Plan/preflight binding differs")
    calls = setup_calls(manifest)
    require(digest(manifest["setup_call_plan"]) == digest(calls)
            and manifest["setup_call_plan_sha256"] == digest(calls), "Setup calls changed")
    require(digest(manifest["resources"]) == digest({phase: resources.policy(phase) for phase in resources.POLICIES}),
            "Resource policies changed")
    require(isinstance(manifest["source_sha256"], str)
            and re.fullmatch("[0-9a-f]{64}", manifest["source_sha256"]), "Missing frozen source identity")
    return manifest


def engine_items(manifest, call):
    requests = {row["base_request_id"]: row for row in manifest["preflight"]["requests"]}
    return [dict(deepcopy(requests[source]), trial_id=trial)
            for source, trial in zip(call["base_request_ids"], call["trial_ids"], strict=True)]


def validate_runtime(model, vectors, identity):
    require(not model.training and "H100" in identity.get("gpu", ""), "Require loaded eval-mode H100 runtime")
    require(identity.get("base_dtype") == "torch.bfloat16" and identity.get("attention") == "sdpa"
            and identity.get("adapter_merged") is False and identity.get("adapter_dtypes") == ["torch.float32"],
            "Actual base/adapter arithmetic policy changed")
    require(set(vectors) == {"pain", "fear", "pain_unit_float32"}
            and all(value.device.type == "cuda" for value in vectors.values()), "Require GPU vector inputs")
    require(all(value.shape == (5120,) and bool(torch.isfinite(value).all()) for value in vectors.values()),
            "Invalid source vector layout")
    require(vectors["pain"].dtype == vectors["fear"].dtype == torch.bfloat16
            and vectors["pain_unit_float32"].dtype == torch.float32, "Source vector dtype changed")
    load = identity.get("load_seconds")
    require(type(load) in (int, float) and math.isfinite(load) and load >= 0, "Invalid measured model loading time")


def _records_equal(left, right):
    return [{k: v for k, v in row.items() if k != "trial_id"} for row in left] == [
        {k: v for k, v in row.items() if k != "trial_id"} for row in right]


def repeat_equal(reference, observed):
    """Exact same-cohort repeat: all actual logits, tokens, inputs and RNG states."""
    a, b = reference["evidence"], observed["evidence"]
    if not (a["complete"] and b["complete"] and _records_equal(reference["records"], observed["records"])):
        return False
    if len(a["forwards"]) != len(b["forwards"]):
        return False
    keys = ("input_ids", "attention_mask", "position_ids", "cache_length_before", "cache_length_after",
            "full_fp32_logits_sha256", "sampling", "prepatch_pain_projection", "post_layer61_pain_projection")
    return all(all(left[key] == right[key] for key in keys)
               for left, right in zip(a["forwards"], b["forwards"], strict=True))


def unaffected_equal(reference, observed):
    """Only common active predictions are paired; later inactive RNG is unrestricted."""
    a, b = reference["evidence"], observed["evidence"]
    left_records, right_records = reference["records"][1:], observed["records"][1:]
    fields = ("answer_token_ids", "text", "prompt_token_ids", "prompt_tokens", "padded_prompt_tokens",
              "prefill_position_id", "prefill_proj_monitor", "p_x", "p_y", "ended_with_eos", "cap_hit")
    if not (a["complete"] and b["complete"] and len(left_records) == len(right_records)
            and all(all(left.get(key) == right.get(key) for key in fields)
                    for left, right in zip(left_records, right_records, strict=True))):
        return False
    for index, left in enumerate(a["forwards"]):
        active = left["sampling"]["alive_before"]
        if any(active[1:]) and index >= len(b["forwards"]):
            return False
        if index >= len(b["forwards"]):
            continue
        right = b["forwards"][index]
        for row in range(1, len(active)):
            if not active[row]:
                continue
            if not right["sampling"]["alive_before"][row]:
                return False
            for key in ("full_fp32_logits_sha256", "input_ids", "attention_mask", "position_ids"):
                if left[key][row] != right[key][row]:
                    return False
            for key in ("uniforms", "rng_state_before_sha256", "rng_state_after_sha256", "selected_token_ids"):
                if left["sampling"][key][row] != right["sampling"][key][row]:
                    return False
    return True


def active_decode(result, *, row=None):
    for frame in result["evidence"]["forwards"]:
        if frame["phase"] != "decode" or "sampling" not in frame:
            continue
        active = frame["sampling"]["alive_before"]
        indices = range(len(active)) if row is None else (row,)
        if any(active[i] and frame["geometry"][i]["coefficient"] == 1
               and frame["geometry"][i]["achieved_delta_norm_min"] is not None
               and frame["geometry"][i]["achieved_delta_norm_min"] > 0 for i in indices):
            return True
    return False


def prefill_equal(reference, observed):
    left, right = reference["evidence"]["forwards"], observed["evidence"]["forwards"]
    if not left or not right:
        return False
    return all(left[0][key] == right[0][key] for key in ("input_ids", "attention_mask", "position_ids")) and (
        [row["before_sha256"] for row in left[0]["geometry"]]
        == [row["before_sha256"] for row in right[0]["geometry"]]
    )


def call_checks(call, result, prior, source_reference=None):
    report = result["evidence"]
    checks = {"hook_inventory_empty": report.get("hooks_before") == report.get("hooks_after") == {}}
    checks["same_cohort_prefill_identity"] = (
        bool(report["forwards"]) and len(report["forwards"][0].get("geometry", [])) == len(call["trial_ids"])
        if source_reference is None else prefill_equal(source_reference, result)
    )
    if call["expected_error"]:
        expected_frames = 1 if call["inject_failure_at"] == "after_prefill" else 2
        checks["expected_controlled_failure"] = (
            report["complete"] is False and not result["records"]
            and report["error"] == {"type": "RuntimeError", "message": f"controlled compatibility fault: {call['inject_failure_at']}"}
            and len(report["forwards"]) == report["attempted_forwards"] == expected_frames
            and report["incomplete_forward"] is None and all(report["checks"].values())
        )
    else:
        checks["complete_integrity"] = (report["complete"] is True and report["integrity_passed"] is True
                                        and len(result["records"]) == len(call["trial_ids"]))
    if call["role"] in ("repeat", "unhooked", "clean_repeat"):
        checks["same_source_repeat_identity"] = repeat_equal(prior[call["reference_call_id"]], result)
    if call["role"] == "one_row":
        checks["unaffected31_active_prediction_identity"] = unaffected_equal(prior[call["reference_call_id"]], result)
        checks["row0_active_decode_engaged"] = active_decode(result, row=0)
    if call["role"] == "primary" and call["arm"] != "baseline":
        checks["active_decode_engaged"] = active_decode(result)
    return checks


def _assignment(manifest, call, index):
    base = {row["base_request_id"]: row for row in manifest["plan"]["base_requests"]}
    row = deepcopy(base[call["base_request_ids"][index]])
    row.update(trial_id=call["trial_ids"][index], arm=call["arm"], call_id=call["call_id"])
    return row


def outcomes_for(manifest, call, result=None, *, missing=False):
    records = {} if result is None else {row["trial_id"]: row for row in result["records"]}
    rows = []
    for index, trial in enumerate(call["trial_ids"]):
        row = _assignment(manifest, call, index)
        record = records.get(trial)
        row.update(status="missing" if missing else "execution_error", category=None, valid=None,
                   cap_hit=None, raw=None, answer_token_ids=None, record_sha256=None)
        if record is not None:
            parsed = design.parse_source_answer(record["text"], row["button_names"], row["target_name"])
            row.update(status="completed", **parsed, cap_hit=record["cap_hit"], raw=record["text"],
                       answer_token_ids=record["answer_token_ids"], record_sha256=digest(record))
        elif (call["expected_error"] and not missing and result is not None
              and result["evidence"].get("error") == {
                  "type": "RuntimeError", "message": f"controlled compatibility fault: {call['inject_failure_at']}"}):
            row["status"] = "expected_fault"
        rows.append(row)
    return rows


def _save_arrays(path, arrays):
    require(not path.exists(), "Array artifact already exists")
    temporary = path.with_suffix(".npz.partial")
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)
    with path.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(path=str(path.name), byte_sha256=checksum,
                arrays={name: engine.array_binding(value) for name, value in arrays.items()})


def forecast(identity, calls, timings):
    required = [call["call_id"] for call in calls if call["role"] == "primary"]
    if len(required) != 10 or any(key not in timings for key in required):
        return dict(benchmark_seconds=None, limit_seconds=2880, passed=False,
                    load_seconds=identity["load_seconds"], slowest_primary_cohort_seconds=None)
    values = [timings[key] for key in required]
    require(all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in values),
            "Invalid measured cohort timing")
    slowest = max(values)
    total = identity["load_seconds"] + 2 * slowest * 65 + 120
    return dict(benchmark_seconds=total, limit_seconds=2880, passed=total <= 2880,
                load_seconds=identity["load_seconds"], slowest_primary_cohort_seconds=slowest,
                safety_multiplier=2, benchmark_cohorts=65, export_seconds=120)


def execute(phase, run, manifest, model, tokenizer, layers, vectors, identity, deadline, checkpoint=None):
    """Write one irreversible phase attempt; stop after the first unexpected failure."""
    require(phase in ("setup", "benchmark"), "Only setup/benchmark inference is implemented")
    require(type(deadline) in (int, float) and math.isfinite(deadline), "Require absolute finite deadline")
    validate_manifest(manifest, tokenizer)
    validate_runtime(model, vectors, identity)
    run = Path(run)
    directory = run / "phases" / phase
    require(not directory.exists(), "Phase already exists; no retry or resume")
    calls = setup_calls(manifest) if phase == "setup" else benchmark_calls(manifest)
    directory.mkdir(parents=True)
    (directory / "calls").mkdir()
    write_once(run / "manifest.json", manifest)
    started = dict(phase=phase, attempt=uuid.uuid4().hex, manifest_sha256=digest(manifest),
                   identity_sha256=digest(identity), calls=calls, calls_sha256=digest(calls),
                   started_unix=time.time(), deadline_unix=deadline)
    write_once(directory / "started.json", started)
    write_once(directory / "identity.json", identity)
    prior, source_prior, rows, observed, timings = {}, {}, {}, [], {}
    terminal_error = None
    for call in calls:
        if time.time() >= deadline:
            terminal_error = {"type": "TimeoutError", "message": "Phase deadline reached before next call"}
            break
        began = time.monotonic()
        setting = design.ARM_SETTINGS[call["arm"]]
        items = engine_items(manifest, call)
        try:
            result = engine.generate_segment(
                model, tokenizer, layers, items=items,
                direction_bf16=vectors["pain" if setting["direction"] == "s2" else "fear"],
                pain_unit_float32=vectors["pain_unit_float32"], coefficient=setting["coefficient"],
                prefill_policy=setting["prefill_policy"],
                eos_token_ids=manifest["preflight"]["eos_token_ids"],
                pad_token_id=manifest["preflight"]["pad_token_id"],
                evidence_mode="compatibility" if phase == "setup" else "benchmark",
                steering_enabled=call["steering_enabled"], one_row_only=call["one_row_only"],
                inject_failure_at=call["inject_failure_at"],
            )
            reference = source_prior.get(call["source_cohort"])
            checks = call_checks(call, result, prior, None if reference is None else reference[1])
            error = None
        except Exception as caught:
            result = None
            checks = {"executor_completed": False}
            error = {"type": type(caught).__name__, "message": str(caught)}
        engine_seconds = time.monotonic() - began
        arrays = {} if result is None else result.pop("arrays")
        artifact = _save_arrays(directory / "calls" / f"{call['call_index']:04d}.npz", arrays)
        outcomes = outcomes_for(manifest, call, result)
        envelope = dict(
            schema_version=1, phase=phase, attempt=started["attempt"], manifest_sha256=digest(manifest),
            identity_sha256=digest(identity), call_spec=call, call_spec_sha256=digest(call),
            prefill_reference_call_id=None if call["source_cohort"] not in source_prior else source_prior[call["source_cohort"]][0],
            items_sha256=digest(items), engine_result=result, arrays_artifact=artifact,
            outcomes=outcomes, outcomes_sha256=digest(outcomes), checks=checks,
            passed=all(checks.values()), executor_error=error, engine_seconds=engine_seconds,
            elapsed_seconds=time.monotonic() - began,
        )
        path = directory / "calls" / f"{call['call_index']:04d}.json"
        write_once(path, envelope)
        timings[call["call_id"]] = envelope["elapsed_seconds"]
        observed.append(dict(call_id=call["call_id"], path=str(path.relative_to(directory)),
                             content_sha256=digest(envelope), total_seconds=timings[call["call_id"]],
                             passed=envelope["passed"]))
        rows[call["call_id"]] = outcomes
        if result is not None:
            prior[call["call_id"]] = result
            source_prior.setdefault(call["source_cohort"], (call["call_id"], result))
        if checkpoint is not None:
            try:
                checkpoint()
            except Exception as caught:
                terminal_error = {"type": type(caught).__name__, "message": str(caught)}
                break
        if not envelope["passed"]:
            terminal_error = error or {"type": "IntegrityError", "message": f"Call failed: {call['call_id']}"}
            break
    outcomes = [row for call in calls for row in rows.get(call["call_id"], outcomes_for(manifest, call, missing=True))]
    write_once(directory / "outcomes.json", outcomes)
    complete = len(observed) == len(calls)
    integrity = complete and terminal_error is None and all(row["passed"] for row in observed)
    prediction = forecast(identity, calls, timings) if phase == "setup" else None
    summary = dict(
        schema_version=1, phase=phase, manifest_sha256=digest(manifest), source_sha256=manifest["source_sha256"],
        attempt=started["attempt"], identity_sha256=digest(identity), calls_sha256=digest(calls),
        assigned_calls=len(calls), observed_calls=len(observed), assigned_rows=sum(len(call["trial_ids"]) for call in calls),
        completed_rows=sum(row["status"] == "completed" for row in outcomes),
        expected_fault_rows=sum(row["status"] == "expected_fault" for row in outcomes),
        missing_rows=sum(row["status"] == "missing" for row in outcomes),
        execution_error_rows=sum(row["status"] == "execution_error" for row in outcomes),
        complete=complete, integrity_passed=integrity, passed=integrity and (prediction is None or prediction["passed"]),
        terminal_error=terminal_error, call_bindings=observed, outcomes_sha256=digest(outcomes),
        forecast=prediction, behavior_gates_used=False,
    )
    write_once(directory / "summary.json", summary)
    if checkpoint is not None:
        checkpoint()
    return summary
