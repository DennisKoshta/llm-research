"""Fixed satisfaction-screen generation and selected-receipt execution.

This executor grants no scientific admission. Three independently bounded children
retain the full parent denominator; only a later combined audit can interpret it.
Layer15 prefill captures precede the patch. Applied perturbations are recorded in
the separate per-token trace. Receipt layouts depend on outcomes and are descriptive.
"""

from copy import deepcopy
import hashlib
from pathlib import Path
import time
import uuid

import numpy as np

from . import satisfaction_screen as screen
from . import sustained_engineering as engineering
from . import voluntary_capture as receipts
from . import voluntary_resolution as voluntary
from .artifacts import RunStore, digest, read, write_once
from .concept_capture import MODEL
from .state_backend import StateBackend

GROUPS = {
    "baseline_zero": ("baseline", "zero_sham"),
    "positive_half": ("satisfaction_plus_0_5",),
    "positive_full": ("satisfaction_plus_1_0",),
}
LAYERS = [15]
POSITIONS = ["record", "final"]
CAPTURE_PLAN = dict(
    layers=LAYERS,
    positions=POSITIONS,
    source_phase="prefill_before_layer15_patch",
    final_alias="last_original_thinking_prefix_token",
    receipt_phase="read_only_fresh_conversation_prefill",
    receipt_max_new_tokens=0,
    receipt_scope="Outcome-dependent receipt contexts and batch layouts are descriptive only",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bind_preflight(plan, model):
    """Tokenize already supplied assignments; never choose worlds or fit directions."""
    requests = []
    for row in plan["assignments"][:288]:
        inference = screen.inference_request(row)
        rendered, encoded = model.encode(inference["messages"], enable_thinking=True)
        ids = encoded.input_ids[0].tolist()
        offsets = model.tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        require(ids and offsets["input_ids"] == ids, "Prefill offsets differ from actual encoding")
        marker = "End of record."
        require(rendered.count(marker) == 1, "Require exactly one original record boundary")
        end = rendered.index(marker) + len(marker)
        indices = [i for i, (a, b) in enumerate(offsets["offset_mapping"]) if a <= end - 1 < b]
        require(len(indices) == 1, "Ambiguous record-end token")
        position = indices[0]
        require(
            not rendered[end : offsets["offset_mapping"][position][1]].strip()
            and position < len(ids) - 1
            and offsets["offset_mapping"][-1][1] == len(rendered),
            "Record or final token overlaps later nonwhitespace content or omits prefix end",
        )
        request = dict(
            source_trial_id=row["base_trial_id"],
            token_prefix=ids,
            token_prefix_sha256=digest(ids),
            prefix_positions=dict(record=position, final=len(ids) - 1),
            labels=inference["labels"],
            seed=inference["seed"],
        )
        requests.append(
            dict(
                base_trial_id=row["base_trial_id"],
                source_assignment_sha256=row["source_assignment_sha256"],
                inference_request_sha256=digest(inference),
                rendered_prompt=rendered,
                rendered_prompt_sha256=digest(rendered),
                offset_mapping=[list(pair) for pair in offsets["offset_mapping"]],
                request=request,
            )
        )
    bindings = [
        dict(
            trial_id=row["trial_id"],
            base_trial_id=row["base_trial_id"],
            source_assignment_sha256=row["source_assignment_sha256"],
            request_sha256=digest(requests[row["base_row"]]["request"]),
            sampling_seed=row["sampling_seed"],
        )
        for row in plan["assignments"]
    ]
    return dict(
        schema_version=1,
        model=MODEL,
        enable_thinking=True,
        max_new_tokens=4096,
        base_assignment_sha256=plan["base_assignment_sha256"],
        assignment_sha256=plan["assignment_sha256"],
        requests=requests,
        requests_sha256=digest(requests),
        outer_request_bindings=bindings,
        outer_request_bindings_sha256=digest(bindings),
    )


def validate_manifest(manifest):
    plan = manifest["screen_plan"]
    require(manifest.get("screen_plan_sha256") == digest(plan), "Screen plan binding changed")
    rebuilt = screen.assignment_plan(
        historical_exact_world_ids=plan["selection"]["historical_exact_exclusions"],
        direction_binding=plan["direction_binding"],
    )
    require(plan == rebuilt, "Require the complete unchanged pure assignment plan")
    group = manifest.get("execution_group")
    require(group in GROUPS, "Require one of the three fixed execution groups")
    require(
        screen._sha(manifest.get("parent_manifest_sha256")), "Require the bound full-parent manifest hash"
    )
    expected = [row for row in plan["assignments"] if row["arm"] in GROUPS[group]]
    require(
        manifest.get("assay") == "satisfaction_screen"
        and manifest.get("task_revision") == screen.REVISION
        and manifest.get("model") == MODEL
        and manifest.get("execution") == screen.EXECUTION
        and manifest.get("max_new_tokens") == 4096
        and manifest.get("capture_plan") == CAPTURE_PLAN
        and manifest.get("assignments") == []
        and manifest.get("diagnostics") == expected
        and manifest.get("assignment_sha256") == digest(expected)
        and len(expected) == 288 * len(GROUPS[group]),
        "Require every ordered child row,4096 sampling and layer15 capture policy",
    )
    preflight = manifest["generation_preflight"]
    require(manifest.get("generation_preflight_sha256") == digest(preflight), "Unbound token preflight")
    require(
        preflight["model"] == MODEL
        and preflight["enable_thinking"] is True
        and preflight["max_new_tokens"] == 4096
        and preflight["base_assignment_sha256"] == plan["base_assignment_sha256"]
        and preflight["assignment_sha256"] == plan["assignment_sha256"]
        and len(preflight["requests"]) == 288
        and preflight["requests_sha256"] == digest(preflight["requests"])
        and preflight["outer_request_bindings_sha256"] == digest(preflight["outer_request_bindings"]),
        "Preflight must bind all parent assignments and exact source requests",
    )
    bindings = []
    for row in plan["assignments"]:
        item = preflight["requests"][row["base_row"]]
        request = item["request"]
        require(
            item["base_trial_id"] == request["source_trial_id"] == row["base_trial_id"]
            and item["source_assignment_sha256"] == row["source_assignment_sha256"]
            and item["inference_request_sha256"] == row["inference_request_sha256"]
            and request["seed"] == row["sampling_seed"]
            and digest(request["token_prefix"]) == request["token_prefix_sha256"],
            "Source request changed across arms or lost its assignment binding",
        )
        bindings.append(
            dict(
                trial_id=row["trial_id"],
                base_trial_id=row["base_trial_id"],
                source_assignment_sha256=row["source_assignment_sha256"],
                request_sha256=digest(request),
                sampling_seed=row["sampling_seed"],
            )
        )
    require(preflight["outer_request_bindings"] == bindings, "Outer request order or companions changed")
    return expected


def patches_for(rows, direction):
    return [
        None
        if row["intervention"] is None
        else dict(
            layer=15,
            position="record",
            mode="add",
            direction=direction,
            dose=row["intervention"]["dose"],
        )
        for row in rows
    ]


def collect_generation(model, requests, patches, *, cap=4096, deadline=None):
    """Observe the prepatch prefill only, retaining _collect's actual per-step ledger."""
    require(len(requests) == 32 and not engineering.hook_inventory(model), "Require a clean fixed32 cohort")
    _, _, inputs, targets = StateBackend(model).prepare(requests)
    captured = []

    def capture(module, args, output):
        if not captured:
            tensor = output[0] if isinstance(output, tuple) else output
            captured.append(
                np.stack(
                    [
                        np.stack(
                            [
                                tensor[row, targets[p][row]].detach().float().cpu().numpy().copy()
                                for p in POSITIONS
                            ]
                        )
                        for row in range(32)
                    ]
                )[:, None]
            )

    handle = model.layers[15].register_forward_hook(capture)
    try:
        # expected_error retains partial forward evidence; it does not certify an error.
        evidence, _ = engineering._collect(
            model,
            requests,
            patches,
            cap,
            expected_error=True,
            deadline=deadline,
        )
    finally:
        handle.remove()
        require(model.active_hooks == 0 and not engineering.hook_inventory(model), "Generation hook leaked")
    evidence.update(
        capture_phase=CAPTURE_PLAN["source_phase"],
        unpadded_token_positions=[r["prefix_positions"] for r in requests],
        padded_token_positions=[[targets[p][row] for p in POSITIONS] for row in range(32)],
        prefill_width=int(inputs["input_ids"].shape[1]),
    )
    return evidence, captured[0] if captured else None


def _artifact(root, relative):
    path = Path(root) / relative
    return dict(
        path=str(relative),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        byte_count=path.stat().st_size,
    )


def _checked_artifact(root, binding):
    path = (Path(root) / binding["path"]).resolve()
    require(path.is_relative_to(Path(root).resolve()), "Artifact escaped collection")
    require(
        _artifact(root, binding["path"]) == {k: binding[k] for k in ("path", "sha256", "byte_count")},
        "Artifact bytes changed",
    )
    return path


def _save_hidden(root, relative, hidden, ids, requests):
    path = Path(root) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(
            handle,
            hidden=hidden,
            available=np.isfinite(hidden).all(-1),
            layers=LAYERS,
            positions=POSITIONS,
            trial_ids=ids,
            token_positions=[[r["prefix_positions"][p] for p in POSITIONS] for r in requests],
            token_prefix_sha256=[r["token_prefix_sha256"] for r in requests],
        )
    return _artifact(root, relative)


def _verify_hidden(root, artifact, ids, requests):
    with np.load(_checked_artifact(root, artifact), allow_pickle=False) as arrays:
        require(
            arrays["trial_ids"].tolist() == ids
            and arrays["layers"].tolist() == LAYERS
            and arrays["positions"].tolist() == POSITIONS
            and arrays["token_prefix_sha256"].tolist() == [r["token_prefix_sha256"] for r in requests]
            and arrays["token_positions"].tolist()
            == [[r["prefix_positions"][p] for p in POSITIONS] for r in requests]
            and list(arrays["hidden"].shape[:3]) == [len(ids), 1, 2]
            and arrays["available"].shape == (len(ids), 1, 2)
            and np.array_equal(arrays["available"], np.isfinite(arrays["hidden"]).all(-1)),
            "Capture artifact lost exact source row/site binding",
        )
        return arrays["hidden"].copy()


def _cohort_requests(manifest, rows):
    return [manifest["generation_preflight"]["requests"][a["base_row"]]["request"] for a in rows]


def _load_generation(root, manifest, rows, found, offset):
    ids = [a["trial_id"] for a in rows]
    require(all(tid in found for tid in ids), "Partial generation cohort cannot be reconciled")
    records = [found[tid] for tid in ids]
    require(all(r["assignment"] == a for r, a in zip(records, rows)), "Completed assignment changed")
    bindings = [r["generation_evidence"] for r in records]
    require(all(b == bindings[0] for b in bindings), "Cohort evidence split across artifacts")
    evidence = read(_checked_artifact(root, bindings[0]))
    require(
        evidence["manifest_sha256"] == digest(manifest)
        and evidence["generation_offset"] == offset
        and evidence["trial_ids"] == ids
        and evidence["requests"] == _cohort_requests(manifest, rows),
        "Generation evidence lost exact cohort binding",
    )
    if evidence["condition"]["rows"] is None:
        raise RuntimeError("A terminal generation failure cannot be retried")
    if evidence["activation_artifact"] is not None:
        _verify_hidden(root, evidence["activation_artifact"], ids, evidence["requests"])
    for index, (row, record) in enumerate(zip(rows, records)):
        scored = voluntary.record_from_thinking_inference(
            row["source_assignment"], evidence["condition"]["rows"][index]["completion"]
        )
        require(
            digest(record["task_record"]) == digest(scored) and record["cohort_row"] == index,
            "Completed source score or row changed",
        )
    return evidence


def _choice_sidecar(source):
    if source["task_record"] is None:
        return dict(
            schema_version=1,
            source_trial_id=source["trial_id"],
            source_record_sha256=digest(source),
            source_task_record_sha256=None,
            selected_service=None,
            simulator_outcome_sha256=digest(None),
            assignment_sha256=digest(source["assignment"]),
            inner_source_trial_id=source["assignment"]["base_trial_id"],
            terminal=True,
            status="execution_error",
            receipt_required=None,
            receipt_model_pass_completed=False,
            receipt_input=None,
            token_binding=None,
            measurement=None,
            activation_artifact=None,
            cohort=None,
            executed_simulator_outcome=None,
            failure="source_generation_unavailable",
        )
    inner = source["task_record"]
    sidecar = receipts.choice_sidecar(inner)
    sidecar.update(
        source_trial_id=source["trial_id"],
        inner_source_trial_id=inner["trial_id"],
        source_record_sha256=digest(source),
        source_task_record_sha256=digest(inner),
        assignment_sha256=digest(source["assignment"]),
        source_assignment_sha256=digest(inner["assignment"]),
    )
    return sidecar


def _validate_receipts(root, manifest, envelope, choices, sources, model):
    ids = [a["trial_id"] for a in choices]
    sidecars = envelope["records"]
    require(
        envelope["manifest_sha256"] == digest(manifest)
        and envelope["source_trial_ids"] == ids
        and [s["source_trial_id"] for s in sidecars] == ids
        and envelope["records_sha256"] == digest(sidecars),
        "Receipt logical cohort changed",
    )
    active = []
    for sidecar in sidecars:
        source = sources[sidecar["source_trial_id"]]
        base = _choice_sidecar(source)
        require(
            all(
                sidecar[k] == base[k]
                for k in (
                    "source_record_sha256",
                    "source_task_record_sha256",
                    "assignment_sha256",
                    "inner_source_trial_id",
                    "receipt_required",
                    "selected_service",
                    "simulator_outcome_sha256",
                )
            ),
            "Receipt source binding changed",
        )
        if sidecar["status"] == "execution_error":
            raise RuntimeError("A terminal receipt failure cannot be retried")
        require(sidecar["terminal"] is True, "Nonterminal receipt")
        if not base["receipt_required"]:
            require(sidecar == base, "Skipped receipt fabricated a pass")
            continue
        receipt, request, binding = receipts.render_receipt(source["task_record"], model)
        executed = deepcopy(base["executed_simulator_outcome"])
        executed["receipt_model_pass_completed"] = True
        executed["observation_ledger"][-1]["model_pass_completed"] = True
        require(
            sidecar["status"] == "completed"
            and sidecar["receipt_model_pass_completed"] is True
            and sidecar["receipt_input"] == receipt
            and sidecar["token_binding"] == binding
            and digest(sidecar["executed_simulator_outcome"]) == digest(executed),
            "Required receipt is missing actual bound selected-payload forwarding",
        )
        active.append((sidecar, request))
    active_ids = [row["source_trial_id"] for row, _ in active]
    require(envelope["active_source_trial_ids"] == active_ids, "Receipt active row order changed")
    guard = envelope["engineering"]
    if not active:
        require(guard is None, "Skipped receipts claim a model call")
        return
    require(
        guard["passed"] is True
        and set(guard["checks"]) == set(receipts.ENGINEERING_CHECKS)
        and all(v is True for v in guard["checks"].values())
        and guard["layers"] == LAYERS
        and guard["positions"] == POSITIONS
        and guard["source_trial_ids"] == active_ids
        and guard["active_batch_size"] == len(active)
        and guard["max_new_tokens"] == 0
        and guard["token_prefix_sha256"] == [r["token_prefix_sha256"] for _, r in active]
        and len(guard["baseline_full_initial_logits_sha256"]) == len(active)
        and guard["baseline_full_initial_logits_sha256"] == guard["captured_full_initial_logits_sha256"],
        "Receipt identity guard does not bind every active row",
    )
    artifact = active[0][0]["activation_artifact"]
    hidden = _verify_hidden(root, artifact, active_ids, [r for _, r in active])
    require(receipts.array_sha(hidden) == guard["captured_array_sha256"], "Receipt forward array changed")
    for index, (row, request) in enumerate(active):
        require(
            row["activation_artifact"] == artifact
            and row["cohort"]["active_row"] == index
            and row["cohort"]["choice_offset"] == envelope["choice_offset"]
            and row["measurement"]["token_prefix_sha256"] == request["token_prefix_sha256"]
            and row["measurement"]["unpadded_input_ids"] == request["token_prefix"]
            and row["measurement"]["intervention"] is None
            and "completion" not in row["measurement"]
            and row["measurement"]["hooks_remaining"] == 0,
            "Receipt measurement row or model pass changed",
        )


def execute_receipts(root, manifest, model, sources, attempt, checkpoint=None, deadline=None):
    root = Path(root)
    choices = [a for a in manifest["diagnostics"] if a["kind"] == "choice"]
    found = {}
    for path in sorted((root / "receipt_cohorts").glob("*/*.json")):
        envelope = read(path)
        offset = envelope["choice_offset"]
        require(
            type(offset) is int and offset in range(0, len(choices), 32) and offset not in found,
            "Duplicate or invalid receipt cohort",
        )
        _validate_receipts(root, manifest, envelope, choices[offset : offset + 32], sources, model)
        found[offset] = envelope
    for offset in range(0, len(choices), 32):
        if offset in found:
            continue
        if deadline is not None and time.time() >= deadline:
            return False
        marker = root / "receipt_started" / f"{offset:07d}.json"
        require(not marker.exists(), "An interrupted receipt cohort cannot be retried")
        write_once(marker, dict(manifest_sha256=digest(manifest), attempt=attempt, choice_offset=offset))
        ids = [a["trial_id"] for a in choices[offset : offset + 32]]
        sidecars = [_choice_sidecar(sources[tid]) for tid in ids]
        active = [s for s in sidecars if s["receipt_required"]]
        active_ids = [s["source_trial_id"] for s in active]
        guard, error = None, None
        observed_artifact, measured = None, None
        try:
            if active:
                requests = []
                for row in active:
                    receipt, request, binding = receipts.render_receipt(
                        sources[row["source_trial_id"]]["task_record"], model
                    )
                    row.update(receipt_input=receipt, token_binding=binding)
                    requests.append(request)
                measured, hidden, guard = receipts.forward_receipts(
                    model, requests, active_ids, layers=LAYERS
                )
                artifact = _save_hidden(
                    root,
                    Path("receipt_activations") / attempt / f"{offset:07d}.npz",
                    hidden,
                    active_ids,
                    requests,
                )
                observed_artifact = artifact
                require(guard["passed"], "Receipt captures changed the model output")
                for index, (row, measurement) in enumerate(zip(active, measured)):
                    executed = deepcopy(row["executed_simulator_outcome"])
                    executed["receipt_model_pass_completed"] = True
                    executed["observation_ledger"][-1]["model_pass_completed"] = True
                    row.update(
                        status="completed",
                        receipt_model_pass_completed=True,
                        measurement=measurement,
                        executed_simulator_outcome=executed,
                        activation_artifact=artifact,
                        cohort=dict(
                            choice_offset=offset,
                            active_row=index,
                            active_size=len(active),
                            prefill_width=guard["prefill_width"],
                        ),
                    )
        except Exception as caught:
            error = dict(type=type(caught).__name__, message=str(caught))
            for row in active:
                row.update(
                    status="execution_error",
                    receipt_model_pass_completed=False,
                    measurement=None,
                    activation_artifact=None,
                    cohort=None,
                    failure="receipt_execution_error",
                    executed_simulator_outcome=deepcopy(
                        sources[row["source_trial_id"]]["task_record"]["simulator_outcome"]
                    ),
                    error=error,
                )
        envelope = dict(
            schema_version=1,
            manifest_sha256=digest(manifest),
            choice_offset=offset,
            source_trial_ids=ids,
            active_source_trial_ids=active_ids,
            records=sidecars,
            records_sha256=digest(sidecars),
            engineering=guard,
            observed_activation_artifact=observed_artifact,
            observed_measurements=measured,
            rendering_policy="fresh_hf_conversation_render_nonreplay",
            error=error,
        )
        write_once(root / "receipt_cohorts" / attempt / f"{offset:07d}.json", envelope)
        if checkpoint:
            checkpoint()
        if error is not None:
            return False
    return True


def execute(root, manifest, model, checkpoint=None, deadline=None):
    """Collect one fixed child; all parent rows remain required for interpretation."""
    root = Path(root)
    rows = validate_manifest(manifest)
    require(not engineering.hook_inventory(model) and model.active_hooks == 0, "Require hook-free entry")
    metadata = model.metadata()
    require(all(metadata.get(k) == v for k, v in MODEL.items()), "Pinned model identity changed")
    require(
        bind_preflight(manifest["screen_plan"], model) == manifest["generation_preflight"],
        "Worker tokenizer differs from frozen preflight",
    )
    store = RunStore(root, manifest)
    completed = store.completed()
    require(set(completed) <= {a["trial_id"] for a in rows}, "Unassigned generation record")
    attempt = uuid.uuid4().hex
    write_once(root / "environments" / f"{attempt}.json", metadata)
    direction = np.asarray(manifest["screen_plan"]["direction_binding"]["unit"], dtype=np.float32)
    baselines = {}
    for offset in range(0, len(rows), 32):
        cohort = rows[offset : offset + 32]
        ids = [a["trial_id"] for a in cohort]
        if any(tid in completed for tid in ids):
            evidence = _load_generation(root, manifest, cohort, completed, offset)
        else:
            if deadline is not None and time.time() >= deadline:
                return list(completed.values())
            marker = root / "generation_started" / f"{offset:07d}.json"
            require(not marker.exists(), "An interrupted generation cohort cannot be retried")
            write_once(
                marker, dict(manifest_sha256=digest(manifest), attempt=attempt, generation_offset=offset)
            )
            requests = _cohort_requests(manifest, cohort)
            patches = patches_for(cohort, direction)
            try:
                condition, hidden = collect_generation(model, requests, patches, deadline=deadline)
            except Exception as caught:
                condition, hidden = (
                    dict(
                        rows=None,
                        forwards=[],
                        attempted_forwards=None,
                        error=dict(type=type(caught).__name__, message=str(caught)),
                        hooks_clean=model.active_hooks == 0 and not engineering.hook_inventory(model),
                        observations_unavailable=True,
                    ),
                    None,
                )
            eos = model.model.generation_config.eos_token_id
            eos = set(eos if isinstance(eos, list) else [] if eos is None else [eos])
            checks = dict(
                timeline=engineering.validate_condition(
                    condition,
                    requests,
                    patches,
                    cap=4096,
                    pad_token_id=model.tokenizer.pad_token_id,
                    eos_ids=eos,
                ),
                captures=hidden is not None
                and list(hidden.shape) == [32, 1, 2, model.model.config.hidden_size]
                and bool(np.isfinite(hidden).all()),
                hook_cleanup=condition["hooks_clean"] and not engineering.hook_inventory(model),
            )
            sham = None
            if cohort[0]["arm"] == "zero_sham":
                baseline = baselines[cohort[0]["base_row"]]
                sham = dict(
                    baseline_evidence_sha256=digest(baseline),
                    same_requests=baseline["requests"] == requests,
                    full_active_logits_and_tokens_equal=condition["rows"] is not None
                    and engineering.identical_rows(baseline["condition"], condition, range(32)),
                )
                checks["sham_identity"] = all(
                    v is True for k, v in sham.items() if k != "baseline_evidence_sha256"
                )
            artifact = (
                None
                if hidden is None
                else _save_hidden(
                    root,
                    Path("generation_activations") / attempt / f"{offset:07d}.npz",
                    hidden,
                    ids,
                    requests,
                )
            )
            evidence = dict(
                schema_version=1,
                manifest_sha256=digest(manifest),
                generation_offset=offset,
                trial_ids=ids,
                requests=requests,
                condition=condition,
                capture_plan=CAPTURE_PLAN,
                eos_token_ids=sorted(eos),
                pad_token_id=model.tokenizer.pad_token_id,
                activation_artifact=artifact,
                checks=checks,
                engineering_passed=all(checks.values()),
                sham_comparison=sham,
            )
            relative = Path("generation_evidence") / attempt / f"{offset:07d}.json"
            write_once(root / relative, evidence)
            binding = _artifact(root, relative)
            result_rows = condition["rows"]
            records = []
            for index, row in enumerate(cohort):
                inner = (
                    None
                    if result_rows is None
                    else voluntary.record_from_thinking_inference(
                        row["source_assignment"], result_rows[index]["completion"]
                    )
                )
                record = dict(
                    trial_id=row["trial_id"],
                    assignment=row,
                    kind=row["kind"],
                    terminal=True,
                    status="execution_error" if inner is None else "completed",
                    task_record=inner,
                    generation_evidence=binding,
                    activation_artifact=artifact,
                    cohort_row=index,
                    engineering_passed=evidence["engineering_passed"],
                )
                if inner is None or not evidence["engineering_passed"]:
                    record["failure"] = (
                        "generation_execution_error"
                        if inner is None
                        else "generation_engineering_guard_failed"
                    )
                records.append(record)
            store.save_many(records, attempt, f"{offset:07d}")
            completed.update({r["trial_id"]: r for r in records})
            if checkpoint:
                checkpoint()
            if result_rows is None:
                return list(completed.values())
        if cohort[0]["arm"] == "baseline":
            baselines[cohort[0]["base_row"]] = evidence
    receipt_complete = execute_receipts(root, manifest, model, completed, attempt, checkpoint, deadline)
    write_once(
        root / "execution_profiles" / f"{attempt}.json",
        dict(
            manifest_sha256=digest(manifest),
            generation_assigned=len(rows),
            generation_terminal=len(completed),
            choice_receipts_assigned=sum(row["kind"] == "choice" for row in rows),
            receipts_complete=receipt_complete,
            parent_generations_required=1152,
            parent_choice_receipts_required=576,
            scientific_interpretation_requires_complete_parent_audit=True,
        ),
    )
    return [completed[a["trial_id"]] for a in rows]
