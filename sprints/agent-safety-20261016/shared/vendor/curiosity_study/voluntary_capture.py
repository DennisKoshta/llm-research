"""Immutable choices followed by separately audited, actual selected receipts."""

from copy import deepcopy
import hashlib
from pathlib import Path
import time
import uuid

import numpy as np

from . import thinking_information as thinking
from . import voluntary_resolution as voluntary
from .artifacts import RunStore, digest, read, write_once
from .concept_capture import LAYERS, MODEL, _logit_hashes

PRECHOICE_SITES = {"record": "record", "final": "prefill"}
POSITIONS = ["record", "final"]
ENGINEERING_CHECKS = (
    "row_coverage",
    "full_initial_logits_equal",
    "every_row_fingerprint_equal",
    "all_captures_finite",
    "capture_shape",
    "no_intervention_or_generation",
    "hook_cleanup",
)
RECEIPT_PLAN = dict(
    schema_version=1,
    layers=LAYERS,
    positions=POSITIONS,
    max_new_tokens=0,
    enable_thinking=False,
    prefix_policy="fresh_hf_conversation_render_nonreplay",
    record_position="last_selected_payload_token",
    final_position="last_rendered_receipt_prefix_token",
    logical_batch_size=32,
    active_rows="valid_closed_nonstop_in_logical_choice_order; no padding assignments",
    identity_guard="every_actual_cohort_full_initial_vocabulary_no_capture_vs_capture",
    analysis_scope="Outcome-dependent receipt contexts and active batch layouts are descriptive only.",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def array_sha(values):
    values = np.ascontiguousarray(values)
    return digest(
        dict(
            dtype=str(values.dtype),
            shape=list(values.shape),
            bytes_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
        )
    )


def validate_manifest(manifest):
    require(
        manifest.get("assay") == "voluntary_resolution"
        and manifest.get("task_revision") == voluntary.REVISION,
        "Unsupported voluntary collection",
    )
    require(
        manifest.get("model") == MODEL
        and manifest.get("assignments") == []
        and manifest.get("execution") == voluntary.EXECUTION,
        "Require pinned voluntary batch32 execution",
    )
    require(
        thinking.manifest_generation_limit(manifest) == 4096,
        "All voluntary generations require the frozen4096 regime",
    )
    require(
        manifest.get("capture_plan", {}).get("layers") == LAYERS
        and manifest.get("receipt_capture_plan") == RECEIPT_PLAN
        and manifest.get("prechoice_capture_aliases") == PRECHOICE_SITES,
        "Voluntary pre-choice and receipt capture sites must be explicit and frozen",
    )
    rows = manifest["diagnostics"]
    seeds = {a["seed"] for a in rows}
    require(
        len(seeds) == 1 and manifest.get("assignment_sha256") == digest(rows),
        "Invalid voluntary assignment binding",
    )
    selection = manifest.get("world_selection", {})
    require(
        "historical_exact_exclusions" in selection
        and selection.get("historical_exact_exclusions_sha256")
        == digest(selection["historical_exact_exclusions"]),
        "Require an explicitly bound historical world exclusion inventory",
    )
    exclusions = selection["historical_exact_exclusions"]
    expected = {
        a["trial_id"]: a for a in voluntary.assignments(next(iter(seeds)), excluded_world_ids=exclusions)
    }
    require(
        len(rows) == len(expected) == 288
        and {a["trial_id"] for a in rows} == set(expected)
        and all(all(a.get(k) == v for k, v in expected[a["trial_id"]].items()) for a in rows),
        "Retain every288 assigned voluntary and diagnostic generation",
    )
    return rows


def render_receipt(record, model):
    """Re-render the bound conversation; do not claim exact source-token replay."""
    receipt = voluntary.receipt_input(record)
    require(
        receipt is not None and receipt["max_new_tokens"] == 0 and receipt["model_pass_completed"] is False,
        "Only a selected non-stop payload requires a model receipt",
    )
    rendered, encoded = model.encode(receipt["messages"], enable_thinking=False)
    ids = encoded.input_ids[0].tolist()
    offsets = model.tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    require(
        offsets["input_ids"] == ids and bool(ids),
        "Receipt offset tokenizer differs from actual model encoding",
    )
    last_message = receipt["messages"][-1]["content"]
    message_start = rendered.rfind(last_message)
    require(message_start >= 0, "Actual rendered receipt omits the selected payload message")
    payload = record["simulator_outcome"]["payload"]
    require(
        last_message == payload + "\nEpisode ended. No response is requested.",
        "Receipt payload suffix changed",
    )
    payload_end = message_start + len(payload)
    candidates = [
        i for i, (start, end) in enumerate(offsets["offset_mapping"]) if start <= payload_end - 1 < end
    ]
    require(len(candidates) == 1, "Selected-payload end token is ambiguous")
    record_position = candidates[0]
    require(
        not rendered[payload_end : offsets["offset_mapping"][record_position][1]].strip()
        and record_position < len(ids) - 1,
        "Receipt record token contains later nonwhitespace suffix content",
    )
    require(
        offsets["offset_mapping"][-1][1] == len(rendered),
        "Receipt final token does not reach rendered prefix end",
    )
    prefix_positions = dict(record=record_position, final=len(ids) - 1)
    request = dict(
        token_prefix=ids,
        token_prefix_sha256=digest(ids),
        prefix_positions=prefix_positions,
        labels=["A", "B"],
        seed=record["assignment"]["sampling_seed"],
    )
    binding = dict(
        rendered_prompt=rendered,
        rendered_prompt_sha256=digest(rendered),
        unpadded_input_ids=ids,
        token_prefix_sha256=digest(ids),
        prefix_positions=prefix_positions,
        offset_mapping=[list(v) for v in offsets["offset_mapping"]],
        selected_payload_span=[message_start, payload_end],
        final_user_message_span=[message_start, message_start + len(last_message)],
        enable_thinking=False,
        add_generation_prompt=True,
        add_special_tokens=False,
        exact_source_token_replay=False,
        template_may_strip_prior_thinking=True,
        rendered_contains_raw_choice=record["inference"]["raw"] in rendered,
        source_raw_choice_sha256=digest(record["inference"]["raw"]),
        rendering_policy="Fresh HF chat-template rendering of actual choice messages and selected payload; prior assistant reasoning may be stripped. Only these recorded token IDs are claimed received.",
    )
    return receipt, request, binding


def forward_receipts(model, requests, source_ids, layers=LAYERS):
    """Persist the captured call whose full initial logits were actually compared."""
    require(
        bool(requests) and len(requests) == len(source_ids), "Receipt forwards require actual active rows"
    )
    baseline, _, before = _logit_hashes(model, requests, [])
    captured, hidden, after = _logit_hashes(model, requests, layers)
    fields = (
        "unpadded_input_ids",
        "input_ids",
        "attention_mask",
        "position_ids",
        "label_logits",
        "greedy_first_token_id",
    )
    checks = dict(
        row_coverage=len(baseline) == len(captured) == len(before) == len(after) == len(requests),
        full_initial_logits_equal=before == after,
        every_row_fingerprint_equal=all(
            all(a[k] == b[k] for k in fields) for a, b in zip(baseline, captured)
        ),
        all_captures_finite=bool(np.isfinite(hidden).all()),
        capture_shape=list(hidden.shape) == [len(requests), len(layers), 2, model.model.config.hidden_size],
        no_intervention_or_generation=all(
            r["intervention"] is None and "completion" not in r for r in baseline + captured
        ),
        hook_cleanup=model.active_hooks == 0 and all(r["hooks_remaining"] == 0 for r in baseline + captured),
    )
    engineering = dict(
        passed=all(checks.values()),
        checks=checks,
        source_trial_ids=source_ids,
        token_prefix_sha256=[r["token_prefix_sha256"] for r in requests],
        layers=list(layers),
        positions=POSITIONS,
        active_batch_size=len(requests),
        prefill_width=captured[0]["prefill_width"],
        max_new_tokens=0,
        baseline_full_initial_logits_sha256=before,
        captured_full_initial_logits_sha256=after,
        captured_array_sha256=array_sha(hidden),
        capability="Read-only receipt prefill only; not generation or intervention parity.",
    )
    return captured, hidden, engineering


def choice_sidecar(source):
    require(
        source.get("kind") == "choice" and "failure" not in source,
        "Only terminal generated choices have receipt slots",
    )
    expected = thinking.record_from_inference(source["assignment"], source["inference"])
    require(
        all(
            digest(source.get(k)) == digest(expected.get(k))
            for k in ("valid", "thinking_closed", "parsed_answer", "selected_service", "simulator_outcome")
        ),
        "Receipt source differs from its actual closed generated choice",
    )
    valid = source["valid"] is True and source["thinking_closed"] is True
    outcome = source["simulator_outcome"]
    required = bool(valid and outcome["receipt_required"])
    return dict(
        schema_version=1,
        source_trial_id=source["trial_id"],
        assignment_sha256=digest(source["assignment"]),
        source_record_sha256=digest(source),
        selected_service=source["selected_service"],
        simulator_outcome_sha256=digest(outcome),
        terminal=True,
        status="pending" if required else "stop" if valid else "invalid_choice",
        receipt_required=required,
        receipt_model_pass_completed=False,
        receipt_input=None,
        token_binding=None,
        measurement=None,
        activation_artifact=None,
        cohort=None,
        executed_simulator_outcome=deepcopy(outcome),
    )


def load_receipts(root, manifest, source_records):
    """No terminal receipt is retried, including an observed execution failure."""
    choices = [a for a in manifest["diagnostics"] if a["kind"] == "choice"]
    found, offsets = {}, set()
    for path in sorted((Path(root) / "receipt_cohorts").glob("*/*.json")):
        envelope = read(path)
        offset = envelope["choice_offset"]
        require(
            type(offset) is int and offset in range(0, len(choices), 32) and offset not in offsets,
            "Duplicate or out-of-range physical receipt cohort",
        )
        expected_ids = [a["trial_id"] for a in choices[offset : offset + 32]]
        records = envelope["records"]
        require(
            envelope["manifest_sha256"] == digest(manifest)
            and envelope["source_trial_ids"] == expected_ids
            and envelope["records_sha256"] == digest(records)
            and [r["source_trial_id"] for r in records] == expected_ids,
            "Receipt cohort must retain every original logical choice slot in order",
        )
        require(path.name == f"{offset:07d}.json", "Receipt file must bind its logical offset")
        for receipt in records:
            tid = receipt["source_trial_id"]
            base = choice_sidecar(source_records[tid])
            require(
                all(
                    receipt[k] == base[k]
                    for k in (
                        "source_record_sha256",
                        "assignment_sha256",
                        "selected_service",
                        "simulator_outcome_sha256",
                        "receipt_required",
                    )
                ),
                "Receipt no longer binds the unchanged generated choice",
            )
            require(
                receipt["terminal"] is True
                and receipt["status"] in ("completed", "stop", "invalid_choice", "execution_error"),
                "Receipt terminal status is malformed",
            )
            if receipt["status"] == "execution_error":
                raise RuntimeError("A terminal receipt execution failure cannot be retried")
            require(
                receipt["receipt_model_pass_completed"] is (receipt["status"] == "completed")
                and (
                    (receipt["status"] == "completed")
                    if base["receipt_required"]
                    else receipt["status"] == base["status"]
                ),
                "A required receipt cannot be replaced by a skipped or fabricated pass",
            )
            if receipt["status"] == "completed":
                require(
                    envelope["engineering"]["passed"] is True and receipt["activation_artifact"] is not None,
                    "Completed receipts require actual captured-forward evidence",
                )
                artifact = receipt["activation_artifact"]
                require(
                    hashlib.sha256((Path(root) / artifact["path"]).read_bytes()).hexdigest()
                    == artifact["sha256"],
                    "Receipt capture artifact hash mismatch",
                )
                executed = deepcopy(source_records[tid]["simulator_outcome"])
                executed["receipt_model_pass_completed"] = True
                executed["observation_ledger"][-1]["model_pass_completed"] = True
                require(
                    receipt["receipt_input"] == voluntary.receipt_input(source_records[tid])
                    and digest(receipt["executed_simulator_outcome"]) == digest(executed),
                    "Completed receipt must preserve selected input and actual completion flags",
                )
            else:
                require(
                    all(
                        receipt[k] is None
                        for k in (
                            "receipt_input",
                            "token_binding",
                            "measurement",
                            "activation_artifact",
                            "cohort",
                        )
                    )
                    and digest(receipt["executed_simulator_outcome"])
                    == digest(base["executed_simulator_outcome"]),
                    "Skipped receipt cannot contain a fabricated model pass",
                )
            found[tid] = receipt
        validate_completed_cohort(root, envelope)
        offsets.add(offset)
    return found


def validate_completed_cohort(root, envelope):
    """Resume requires the bound forward evidence, never a standalone success flag."""
    active = [row for row in envelope["records"] if row["receipt_required"]]
    ids = [row["source_trial_id"] for row in active]
    require(envelope["active_source_trial_ids"] == ids, "Receipt active row order changed")
    guard = envelope["engineering"]
    if not active:
        require(guard is None, "Skipped cohort cannot claim an executed forward")
        return
    prefix_hashes = [row["token_binding"]["token_prefix_sha256"] for row in active]
    require(
        guard["passed"] is True
        and set(guard["checks"]) == set(ENGINEERING_CHECKS)
        and all(value is True for value in guard["checks"].values())
        and guard["source_trial_ids"] == ids
        and guard["token_prefix_sha256"] == prefix_hashes
        and guard["layers"] == LAYERS
        and guard["positions"] == POSITIONS
        and guard["active_batch_size"] == len(ids)
        and guard["max_new_tokens"] == 0
        and len(guard["baseline_full_initial_logits_sha256"]) == len(ids)
        and guard["baseline_full_initial_logits_sha256"] == guard["captured_full_initial_logits_sha256"],
        "Receipt engineering evidence does not establish every actual row",
    )
    artifacts = [row["activation_artifact"] for row in active]
    require(
        len({(a["path"], a["sha256"]) for a in artifacts}) == 1,
        "Atomic receipt cohort must bind one captured forward",
    )
    with np.load(Path(root) / artifacts[0]["path"], allow_pickle=False) as values:
        require(
            values["trial_ids"].tolist() == ids
            and values["layers"].tolist() == LAYERS
            and values["positions"].tolist() == POSITIONS
            and values["token_prefix_sha256"].tolist() == prefix_hashes
            and list(values["hidden"].shape[:3]) == [len(ids), 4, 2]
            and np.isfinite(values["hidden"]).all()
            and values["available"].shape == (len(ids), 4, 2)
            and values["available"].all()
            and array_sha(values["hidden"]) == guard["captured_array_sha256"],
            "Receipt capture layout or forward binding changed",
        )
        for index, row in enumerate(active):
            artifact, binding, cohort = row["activation_artifact"], row["token_binding"], row["cohort"]
            positions = [binding["prefix_positions"][name] for name in POSITIONS]
            require(
                artifact["row"] == cohort["active_row"] == index
                and artifact["layers"] == LAYERS
                and artifact["positions"] == POSITIONS
                and artifact["token_positions"] == positions == values["token_positions"][index].tolist()
                and artifact["available"] == [[True, True]] * 4
                and cohort["choice_offset"] == envelope["choice_offset"]
                and cohort["source_trial_ids"] == envelope["source_trial_ids"]
                and cohort["active_source_trial_ids"] == ids
                and cohort["active_size"] == len(ids)
                and cohort["prefill_width"] == guard["prefill_width"]
                and digest(binding["unpadded_input_ids"]) == prefix_hashes[index],
                "Receipt capture row or prefix binding changed",
            )


def execute_receipts(root, manifest, model, source_records, checkpoint=None, deadline=None):
    root = Path(root)
    choices = [a for a in manifest["diagnostics"] if a["kind"] == "choice"]
    completed = load_receipts(root, manifest, source_records)
    attempt = uuid.uuid4().hex
    metadata = model.metadata()
    require(all(metadata.get(k) == v for k, v in manifest["model"].items()), "Receipt model identity changed")
    metadata["receipt_hidden_size"] = model.model.config.hidden_size
    write_once(root / "receipt_environments" / f"{attempt}.json", metadata)
    for offset in range(0, len(choices), 32):
        cohort = choices[offset : offset + 32]
        ids = [a["trial_id"] for a in cohort]
        present = [tid in completed for tid in ids]
        require(not any(present) or all(present), "A partial logical receipt cohort cannot be resumed")
        if all(present):
            continue
        if deadline is not None and time.time() >= deadline:
            break
        sidecars = [choice_sidecar(source_records[tid]) for tid in ids]
        active = [row for row in sidecars if row["receipt_required"]]
        active_ids = [row["source_trial_id"] for row in active]
        engineering, error = None, None
        try:
            if active:
                requests = []
                for row in active:
                    receipt, request, binding = render_receipt(source_records[row["source_trial_id"]], model)
                    row.update(receipt_input=receipt, token_binding=binding)
                    requests.append(request)
                measurements, hidden, engineering = forward_receipts(model, requests, active_ids)
                require(engineering["passed"], "Receipt capture changed initial model output")
                relative = Path("receipt_activations") / attempt / f"{offset:07d}.npz"
                file = root / relative
                file.parent.mkdir(parents=True, exist_ok=True)
                token_positions = [[r["prefix_positions"][p] for p in POSITIONS] for r in requests]
                with file.open("xb") as handle:
                    np.savez_compressed(
                        handle,
                        hidden=hidden,
                        available=np.isfinite(hidden).all(-1),
                        layers=LAYERS,
                        positions=POSITIONS,
                        trial_ids=active_ids,
                        token_positions=token_positions,
                        token_prefix_sha256=[r["token_prefix_sha256"] for r in requests],
                    )
                sha = hashlib.sha256(file.read_bytes()).hexdigest()
                for index, (row, measurement) in enumerate(zip(active, measurements)):
                    executed = deepcopy(row["executed_simulator_outcome"])
                    executed["receipt_model_pass_completed"] = True
                    executed["observation_ledger"][-1]["model_pass_completed"] = True
                    row.update(
                        status="completed",
                        receipt_model_pass_completed=True,
                        measurement=measurement,
                        executed_simulator_outcome=executed,
                        activation_artifact=dict(
                            path=str(relative),
                            sha256=sha,
                            row=index,
                            layers=LAYERS,
                            positions=POSITIONS,
                            token_positions=token_positions[index],
                            available=[[True, True]] * 4,
                        ),
                        cohort=dict(
                            id=f"{attempt}-{offset:07d}",
                            choice_offset=offset,
                            source_trial_ids=ids,
                            active_source_trial_ids=active_ids,
                            active_row=index,
                            active_size=len(active),
                            prefill_width=measurement["prefill_width"],
                        ),
                    )
        except Exception as caught:
            error = caught
            for row in active:
                row.update(
                    status="execution_error",
                    receipt_model_pass_completed=False,
                    measurement=None,
                    activation_artifact=None,
                    cohort=None,
                    executed_simulator_outcome=deepcopy(
                        source_records[row["source_trial_id"]]["simulator_outcome"]
                    ),
                    failure="receipt_execution_error",
                    error_type=type(caught).__name__,
                    error=str(caught),
                )
        envelope = dict(
            schema_version=1,
            manifest_sha256=digest(manifest),
            choice_offset=offset,
            source_trial_ids=ids,
            active_source_trial_ids=active_ids,
            records=sidecars,
            records_sha256=digest(sidecars),
            engineering=engineering,
            rendering_policy=RECEIPT_PLAN["prefix_policy"],
        )
        write_once(root / "receipt_cohorts" / attempt / f"{offset:07d}.json", envelope)
        if checkpoint:
            checkpoint()
        if error is not None:
            raise error
        completed.update({r["source_trial_id"]: r for r in sidecars})
    write_once(
        root / "receipt_profiles" / f"{attempt}.json",
        dict(
            assigned_choices=len(choices),
            completed_slots=len(completed),
            required_receipts=sum(r["receipt_required"] for r in completed.values()),
            model_passes_completed=sum(r["receipt_model_pass_completed"] for r in completed.values()),
        ),
    )
    if checkpoint:
        checkpoint()
    if len(completed) != len(choices):
        raise RuntimeError("Incomplete receipt stage: missing logical choices remain missing")
    return completed


def execute(root, manifest, model, checkpoint=None, deadline=None):
    rows = validate_manifest(manifest)
    store = RunStore(root, manifest)
    existing = store.completed()
    require(set(existing) <= {a["trial_id"] for a in rows}, "Unexpected voluntary source records")
    for offset in range(0, len(rows), 32):
        present = [a["trial_id"] in existing for a in rows[offset : offset + 32]]
        require(
            not any(present) or all(present), "Cannot resume a partially published source-generation cohort"
        )
    source_records = thinking.execute(root, manifest, model, checkpoint, deadline)
    if set(source_records) != {a["trial_id"] for a in rows}:
        raise RuntimeError("Incomplete source generations: receipt stage cannot fabricate missing choices")
    require(
        not any(r.get("failure") == "execution_error" for r in source_records.values()),
        "Source generation has terminal execution failures",
    )
    for record in source_records.values():
        capture = record["activation_artifact"]
        positions = capture["positions"]
        require(
            capture["layers"] == LAYERS
            and all(name in positions for name in PRECHOICE_SITES.values())
            and all(
                all(available[positions.index(name)] for name in PRECHOICE_SITES.values())
                for available in capture["available"]
            ),
            "Every source requires all eight frozen initial capture sites",
        )
    execute_receipts(root, manifest, model, source_records, checkpoint, deadline)
    return source_records
