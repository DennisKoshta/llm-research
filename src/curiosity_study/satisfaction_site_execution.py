"""Explicit-site execution of the complete fixed calibration or validation grid.

The parent remains the scientific unit of admission. No partial child, repeated
attempt, altered dose, or successful subset replaces the prospectively assigned
grid. Receipt groups are the choice subset of one generation cohort at one site.
"""

from copy import deepcopy
from pathlib import Path
import time
import uuid

import numpy as np

from . import causal_site_engineering as engineering
from . import reserved_object_materials as reserved
from . import satisfaction_causal_sites as study
from . import satisfaction_capture as neutral
from . import satisfaction_site_capture as capture
from . import satisfaction_site_resources as resources
from . import voluntary_capture as receipts
from . import voluntary_resolution as voluntary
from .artifacts import RunStore, digest, write_once
from .concept_capture import MODEL

RECEIPT_GROUPING = "generation_cohort_choice_subset_v1"
POSITIONS = ["record", "final"]
require = study.require


def bind_preflight(plan, model):
    """Render every base once; bind all paired site/arm requests to that rendering."""
    base = plan["base_assignments"]
    representatives = {}
    for row in plan["assignments"]:
        if row["arm"] == "baseline":
            representatives.setdefault(row["base_row"], row)
    require(set(representatives) == set(range(len(base))), "Every ordered source baseline is required")
    requests = []
    for index in range(len(base)):
        row = representatives[index]
        inference = study.inference_request(row)
        rendered, encoded = model.encode(inference["messages"], enable_thinking=True)
        ids = encoded.input_ids[0].tolist()
        offsets = model.tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        require(ids and offsets["input_ids"] == ids, "Actual generation tokens differ from offsets")
        marker = "End of record."
        require(rendered.count(marker) == 1, "Require exactly one original record boundary")
        end = rendered.index(marker) + len(marker)
        indices = [i for i, (a, b) in enumerate(offsets["offset_mapping"]) if a <= end - 1 < b]
        require(len(indices) == 1, "Ambiguous original record token")
        position = indices[0]
        require(not rendered[end:offsets["offset_mapping"][position][1]].strip()
                and position < len(ids) - 1 and offsets["offset_mapping"][-1][1] == len(rendered),
                "Record/final token boundaries changed")
        request = dict(source_trial_id=row["base_trial_id"], token_prefix=ids,
                       token_prefix_sha256=digest(ids), prefix_positions=dict(record=position, final=len(ids) - 1),
                       labels=inference["labels"], seed=inference["seed"])
        requests.append(dict(base_trial_id=row["base_trial_id"], source_assignment_sha256=row["source_assignment_sha256"],
                             inference_request_sha256=digest(inference), rendered_prompt=rendered,
                             rendered_prompt_sha256=digest(rendered),
                             offset_mapping=[list(pair) for pair in offsets["offset_mapping"]], request=request))
    bindings = _request_bindings(plan, requests)
    return dict(schema_version=1, model=MODEL, enable_thinking=True, max_new_tokens=4096,
                base_assignment_sha256=plan["base_assignment_sha256"], assignment_sha256=plan["assignment_sha256"],
                requests=requests, requests_sha256=digest(requests), outer_request_bindings=bindings,
                outer_request_bindings_sha256=digest(bindings))


def _request_bindings(plan, requests):
    return [dict(trial_id=row["trial_id"], base_trial_id=row["base_trial_id"], site=row["site"],
                 source_assignment_sha256=row["source_assignment_sha256"],
                 request_sha256=digest(requests[row["base_row"]]["request"]), sampling_seed=row["sampling_seed"])
            for row in plan["assignments"]]


def bind_receipt_preflight(plan, model):
    """Check all possible selected payloads using synthetic answers, without inference."""
    bindings, stops = [], []
    for row in plan["base_assignments"]:
        if row["kind"] != "choice":
            continue
        source = row["source_assignment"]
        mapping = voluntary.task(source)["action_mapping"]
        for service in voluntary.SERVICES:
            label = next(label for label, value in mapping.items() if value == service)
            raw = f"</think>\n{label}"
            record = voluntary.record_from_thinking_inference(source, dict(
                raw=raw, generated_token_ids=model.tokenizer.encode(raw, add_special_tokens=False), ended_with_eos=True))
            require(record["valid"] and record["selected_service"] == service, "Synthetic receipt parsing changed")
            if service == "stop":
                require(voluntary.receipt_input(record) is None, "Stop must have no payload")
                stops.append(row["base_trial_id"])
            else:
                _, _, binding = receipts.render_receipt(record, model)
                bindings.append(dict(base_trial_id=row["base_trial_id"], selected_service=service,
                                     token_binding_sha256=digest(binding), prefix_positions=binding["prefix_positions"]))
    nchoice = sum(row["kind"] == "choice" for row in plan["base_assignments"])
    require(len(bindings) == 3 * nchoice and len(stops) == nchoice, "Every possible receipt disposition required")
    return dict(passed=True, model=MODEL, generated_model_outputs=False, synthetic_answer_only=True,
                base_assignment_sha256=plan["base_assignment_sha256"], non_stop_bindings=bindings,
                stop_checks=stops, bindings_sha256=digest(dict(non_stop=bindings, stop=stops)))


def validate_manifest(manifest):
    plan = manifest["site_plan"]
    phase = manifest.get("site_phase")
    require(manifest.get("site_plan_sha256") == digest(plan) and phase == plan["phase"], "Site plan/phase changed")
    source = manifest["source_screen_manifest"]
    require(manifest.get("source_screen_manifest_sha256") == digest(source) == study.PRIOR_MANIFEST_SHA256,
            "Original completed S source provenance changed")
    materials = manifest["reserved_materials"]
    require(manifest.get("reserved_materials_sha256") == digest(materials)
            and materials["selection_sha256"] == digest(materials["selection"])
            and materials["base_assignment_sha256"] == digest(materials["base_assignments"]),
            "Reserved complete inventory must be frozen before calibration")
    reserved.validate_reserved_inputs(materials["selection"], materials["base_assignments"])
    if phase == "calibration":
        base, selection = study.calibration_inputs(source)
        selected_site = None
    else:
        require(phase == "validation" and len(plan["sites"]) == 1, "Unknown scientific site phase")
        base, selection = materials["base_assignments"], materials["selection"]
        selected_site = manifest.get("selected_site")
        require(selected_site == plan["sites"][0] and type(selected_site) is int,
                "Validation requires exactly its previously selected site")
        receipt = manifest.get("selection_receipt")
        require(isinstance(receipt, dict) and manifest.get("selection_receipt_sha256") == digest(receipt)
                and receipt.get("selected_site") == selected_site,
                "Validation site selection receipt is unbound")
    rebuilt = study.assignment_plan(phase=phase, base_assignments=base, selection=selection,
                                    direction_bindings=plan["direction_bindings"], dose_reference=plan["dose_reference"],
                                    selected_site=selected_site)
    require(plan == rebuilt, "Complete pure site assignment plan differs")
    certificate = manifest.get("setup_certificate")
    require(isinstance(certificate, dict) and manifest.get("setup_certificate_sha256") == digest(certificate)
            and certificate.get("dose_reference") == plan["dose_reference"], "Setup dose reference is unbound")
    expected = study.group_assignments(plan, manifest.get("execution_group"))
    require(study.valid_sha(manifest.get("parent_manifest_sha256")), "Whole parent binding required")
    require(manifest.get("assay") == study.ASSAY and manifest.get("task_revision") == study.REVISION
            and manifest.get("model") == MODEL and manifest.get("execution") == study.EXECUTION
            and type(manifest.get("max_new_tokens")) is int and manifest["max_new_tokens"] == 4096
            and manifest.get("capture_plans") == {str(site): capture.capture_plan(site) for site in plan["sites"]}
            and manifest.get("receipt_grouping") == RECEIPT_GROUPING
            and manifest.get("arithmetic") == engineering.ARITHMETIC and manifest.get("schedule") == study.SCHEDULE
            and manifest.get("assignments") == [] and manifest.get("diagnostics") == expected
            and manifest.get("assignment_sha256") == digest(expected), "Child execution/capture policy differs")
    resources.job_wall_limit(manifest)
    preflight = manifest["generation_preflight"]
    requests = preflight["requests"]
    require(manifest.get("generation_preflight_sha256") == digest(preflight)
            and preflight["model"] == MODEL and preflight["enable_thinking"] is True
            and preflight["max_new_tokens"] == 4096
            and preflight["base_assignment_sha256"] == plan["base_assignment_sha256"]
            and preflight["assignment_sha256"] == plan["assignment_sha256"]
            and len(requests) == len(base) and preflight["requests_sha256"] == digest(requests)
            and preflight["outer_request_bindings"] == _request_bindings(plan, requests)
            and preflight["outer_request_bindings_sha256"] == digest(preflight["outer_request_bindings"]),
            "Full generation preflight inventory differs")
    capture._requests([item["request"] for item in requests], len(base))
    for row, item in zip(base, requests, strict=True):
        require(item["base_trial_id"] == item["request"]["source_trial_id"] == row["base_trial_id"]
                and item["source_assignment_sha256"] == row["source_assignment_sha256"]
                and item["inference_request_sha256"] == row["inference_request_sha256"]
                and item["request"]["seed"] == row["sampling_seed"]
                and item["request"]["labels"] == voluntary.LABELS
                and item["rendered_prompt_sha256"] == digest(item["rendered_prompt"]),
                "Source request/order/seed differs")
    if phase == "calibration":
        require([item["request"] for item in requests]
                == [item["request"] for item in source["generation_preflight"]["requests"]],
                "Calibration must reuse all288 original token prefixes and companions")
    check = manifest["receipt_preflight"]
    choices = [row["base_trial_id"] for row in base if row["kind"] == "choice"]
    require(manifest.get("receipt_preflight_sha256") == digest(check) and check["passed"] is True
            and check["model"] == MODEL and check["generated_model_outputs"] is False
            and check["synthetic_answer_only"] is True and check["base_assignment_sha256"] == plan["base_assignment_sha256"]
            and check["stop_checks"] == choices
            and [(row["base_trial_id"], row["selected_service"]) for row in check["non_stop_bindings"]]
            == [(choice, service) for choice in choices for service in voluntary.SERVICES if service != "stop"]
            and check["bindings_sha256"] == digest(dict(non_stop=check["non_stop_bindings"], stop=check["stop_checks"])),
            "Every possible selected receipt must have a bound preflight")
    return expected


def patches_for(rows, direction):
    return [None if row["intervention"] is None else dict(
        layer=row["site"], position="record", mode="add", direction=direction, dose=row["intervention"]["dose"])
        for row in rows]


def _cohort_requests(manifest, cohort):
    return [manifest["generation_preflight"]["requests"][row["base_row"]]["request"] for row in cohort]


def _verify_arrays(root, artifact, ids, requests, *, site, keys):
    with np.load(neutral._checked_artifact(root, artifact), allow_pickle=False) as data:
        require(data["trial_ids"].tolist() == ids and data["layers"].tolist() == [site]
                and data["positions"].tolist() == POSITIONS
                and data["token_positions"].tolist() == [[r["prefix_positions"][p] for p in POSITIONS] for r in requests]
                and data["token_prefix_sha256"].tolist() == [r["token_prefix_sha256"] for r in requests],
                "Actual site/row/prefix artifact binding changed")
        arrays = {key: data[key].copy() for key in keys}
        require(artifact["arrays"] == {key: engineering.array_binding(value) for key, value in arrays.items()}
                and all(value.dtype == np.float32 and value.shape[:3] == (len(ids), 1, 2)
                        and np.isfinite(value).all() for value in arrays.values()), "Capture bytes or layout changed")
        return arrays


def _engaged(condition, patches):
    if condition.get("rows") is None:
        return False
    for row, patch in zip(condition["rows"], patches, strict=True):
        events = row["intervention_trace"]["events"]
        if patch is None:
            if events:
                return False
        elif not events or any(event["requested_scalar_delta"] != patch["dose"]
                               or (patch["dose"] > 0 and not (
                                   np.isfinite(event["delta_norm"]) and event["delta_norm"] > 0
                                   and np.isfinite(event["projection_after"]) and np.isfinite(event["projection_before"])
                                   and event["projection_after"] > event["projection_before"])) for event in events):
            return False
    return True


def receipt_groups(rows):
    """An outcome-independent subset of one original generation cohort, never mixed sites."""
    choice_offset = 0
    for offset in range(0, len(rows), 32):
        cohort = rows[offset:offset + 32]
        require(len(cohort) == 32 and len({(r["site"], r["arm"], r["base_cohort"]) for r in cohort}) == 1,
                "Mixed or partial generation cohort")
        choices = [row for row in cohort if row["kind"] == "choice"]
        yield dict(generation_offset=offset, choice_offset=choice_offset, site=cohort[0]["site"],
                   arm=cohort[0]["arm"], base_cohort=cohort[0]["base_cohort"], choices=choices)
        choice_offset += len(choices)


def execute_receipts(root, manifest, model, sources, attempt, checkpoint=None, deadline=None):
    root = Path(root)
    manifest_sha = digest(manifest)
    require(not any((root / "receipt_cohorts").glob("*/*.json")) and not any((root / "receipt_started").glob("*.json")),
            "Receipt execution cannot resume or retry")
    for group in receipt_groups(manifest["diagnostics"]):
        if deadline is not None and time.time() >= deadline:
            return False
        choices = group["choices"]
        metadata = {key: value for key, value in group.items() if key != "choices"}
        offset, site = group["generation_offset"], group["site"]
        write_once(root / "receipt_started" / f"{offset:07d}.json",
                   dict(manifest_sha256=manifest_sha, attempt=attempt, **metadata))
        ids = [row["trial_id"] for row in choices]
        sidecars = [neutral._choice_sidecar(sources[tid]) for tid in ids]
        active = [row for row in sidecars if row["receipt_required"]]
        active_ids = [row["source_trial_id"] for row in active]
        measured, observed, guard, error = None, None, None, None
        try:
            if active:
                requests = []
                for row in active:
                    receipt, request, binding = receipts.render_receipt(sources[row["source_trial_id"]]["task_record"], model)
                    row.update(receipt_input=receipt, token_binding=binding)
                    requests.append(request)
                measured, hidden, guard = capture.forward_receipts(model, requests, active_ids, layer=site)
                observed = capture.save_arrays(root, Path("receipt_activations") / attempt / f"{offset:07d}.npz",
                                               dict(hidden=hidden), layer=site, source_ids=active_ids, requests=requests)
                require(guard["passed"] is True and guard["layers"] == [site]
                        and guard["source_trial_ids"] == active_ids, "Actual receipt identity or site failed")
                for index, (row, measurement) in enumerate(zip(active, measured, strict=True)):
                    executed = deepcopy(row["executed_simulator_outcome"])
                    executed["receipt_model_pass_completed"] = True
                    executed["observation_ledger"][-1]["model_pass_completed"] = True
                    row.update(status="completed", receipt_model_pass_completed=True, measurement=measurement,
                               executed_simulator_outcome=executed, activation_artifact=observed,
                               cohort=dict(**metadata, active_row=index, active_size=len(active),
                                           prefill_width=guard["prefill_width"]))
        except Exception as caught:
            error = dict(type=type(caught).__name__, message=str(caught))
            for row in active:
                row.update(status="execution_error", receipt_model_pass_completed=False, measurement=None,
                           activation_artifact=None, cohort=None, failure="receipt_execution_error", error=error,
                           executed_simulator_outcome=deepcopy(sources[row["source_trial_id"]]["task_record"]["simulator_outcome"]))
        envelope = dict(schema_version=1, manifest_sha256=manifest_sha, attempt=attempt, **metadata,
                        receipt_grouping=RECEIPT_GROUPING, source_trial_ids=ids, active_source_trial_ids=active_ids,
                        records=sidecars, records_sha256=digest(sidecars), engineering=guard,
                        observed_activation_artifact=observed, observed_measurements=measured,
                        rendering_policy="fresh_hf_conversation_render_nonreplay", error=error)
        write_once(root / "receipt_cohorts" / attempt / f"{offset:07d}.json", envelope)
        if checkpoint:
            checkpoint()
        if error is not None:
            return False
    return True


def execute(root, manifest, model, checkpoint=None, deadline=None):
    """One immutable worker attempt; returns every durable assigned disposition."""
    root = Path(root)
    rows = validate_manifest(manifest)
    require(not engineering.hook_inventory(model) and model.active_hooks == 0, "Require hook-free entry")
    metadata = model.metadata()
    require(all(metadata.get(k) == value for k, value in MODEL.items()), "Pinned model identity changed")
    require(bind_preflight(manifest["site_plan"], model) == manifest["generation_preflight"]
            and bind_receipt_preflight(manifest["site_plan"], model) == manifest["receipt_preflight"],
            "Worker tokenizer differs from frozen source/receipt preflight")
    store = RunStore(root, manifest)
    manifest_sha = store.manifest_hash
    require(not store.completed() and not (root / "execution_started.json").exists()
            and not any((root / "generation_started").glob("*.json"))
            and not any((root / "receipt_started").glob("*.json"))
            and len(list((root / "admissions").glob("*.json"))) <= 1, "A scientific worker cannot resume or retry")
    attempt = uuid.uuid4().hex
    write_once(root / "execution_started.json", dict(manifest_sha256=manifest_sha, attempt=attempt))
    write_once(root / "environments" / f"{attempt}.json", metadata)
    baselines, completed = {}, {}
    for offset in range(0, len(rows), 32):
        if deadline is not None and time.time() >= deadline:
            return completed
        cohort = rows[offset:offset + 32]
        site, arm, base_cohort = cohort[0]["site"], cohort[0]["arm"], cohort[0]["base_cohort"]
        require(len(cohort) == 32 and len({(r["site"], r["arm"], r["base_cohort"]) for r in cohort}) == 1,
                "Mixed scientific cohort")
        ids = [row["trial_id"] for row in cohort]
        binding = manifest["site_plan"]["direction_bindings"][str(site)]["zero_sham" if arm == "baseline" else arm]
        direction = np.asarray(binding["unit"], dtype=np.float32)
        requests, patches = _cohort_requests(manifest, cohort), patches_for(cohort, direction)
        write_once(root / "generation_started" / f"{offset:07d}.json",
                   dict(manifest_sha256=manifest_sha, attempt=attempt, generation_offset=offset, site=site, arm=arm))
        try:
            condition, arrays = capture.collect_generation(model, requests, patches, layer=site, cap=4096, deadline=deadline)
        except Exception as caught:
            condition, arrays = dict(rows=None, forwards=[], attempted_forwards=None,
                                     error=dict(type=type(caught).__name__, message=str(caught)),
                                     hooks_clean=model.active_hooks == 0 and not engineering.hook_inventory(model),
                                     observations_unavailable=True), {}
        eos = model.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [] if eos is None else [eos])
        checks = dict(
            timeline=engineering.validate_condition(condition, requests, patches, layer=site, cap=4096,
                                                      pad_token_id=model.tokenizer.pad_token_id, eos_ids=eos),
            captures=set(arrays) == {"hidden", "prefill_after"}
            and all(value.shape == (32, 1, 2, model.model.config.hidden_size) and value.dtype == np.float32
                    and bool(np.isfinite(value).all()) for value in arrays.values()),
            bfloat16_prefill=condition.get("activation_dtypes")
            == dict(before="torch.bfloat16", after="torch.bfloat16"),
            prefill_geometry=capture.validate_prefill_geometry(condition, arrays, patches, torch=model.torch),
            engagement=_engaged(condition, patches),
            hook_cleanup=condition["hooks_clean"] and not engineering.hook_inventory(model),
        )
        common, sham = None, None
        if arm != "baseline":
            baseline, previous = baselines[(base_cohort, site)]
            common = dict(baseline_evidence_sha256=digest(baseline), same_requests=baseline["requests"] == requests,
                          site_prepatch_captures_equal="hidden" in arrays and "hidden" in previous
                          and np.array_equal(arrays["hidden"], previous["hidden"]))
            checks["common_prefill_identity"] = all(v is True for k, v in common.items() if k != "baseline_evidence_sha256")
            if arm == "zero_sham":
                sham = dict(baseline_evidence_sha256=digest(baseline), same_requests=baseline["requests"] == requests,
                            full_active_logits_and_tokens_equal=condition["rows"] is not None
                            and engineering.identical_rows(baseline["condition"], condition, range(32)))
                checks["sham_identity"] = all(v is True for k, v in sham.items() if k != "baseline_evidence_sha256")
        artifact = (capture.save_arrays(root, Path("generation_activations") / attempt / f"{offset:07d}.npz",
                                        arrays, layer=site, source_ids=ids, requests=requests) if arrays else None)
        evidence = dict(schema_version=1, manifest_sha256=manifest_sha, attempt=attempt,
                        generation_offset=offset, site=site, arm=arm, base_cohort=base_cohort, trial_ids=ids,
                        requests=requests, condition=condition, capture_plan=capture.capture_plan(site),
                        eos_token_ids=sorted(eos), pad_token_id=model.tokenizer.pad_token_id, activation_artifact=artifact,
                        checks=checks, engineering_passed=all(checks.values()), sham_comparison=sham,
                        common_prefill_comparison=common)
        relative = Path("generation_evidence") / attempt / f"{offset:07d}.json"
        write_once(root / relative, evidence)
        evidence_binding = neutral._artifact(root, relative)
        records = []
        for index, row in enumerate(cohort):
            inner = (None if condition["rows"] is None else voluntary.record_from_thinking_inference(
                row["source_assignment"], condition["rows"][index]["completion"]))
            record = dict(trial_id=row["trial_id"], assignment=row, kind=row["kind"], terminal=True,
                          status="execution_error" if inner is None else "completed", task_record=inner,
                          generation_evidence=evidence_binding, activation_artifact=artifact, cohort_row=index,
                          engineering_passed=evidence["engineering_passed"])
            if inner is None or not evidence["engineering_passed"]:
                record["failure"] = "execution_error" if inner is None else "generation_engineering_guard_failed"
            records.append(record)
        store.save_many(records, attempt, f"{offset:07d}")
        completed.update({row["trial_id"]: row for row in records})
        if checkpoint:
            checkpoint()
        if condition["rows"] is None:
            return completed
        if arm == "baseline":
            baselines[(base_cohort, site)] = (evidence, arrays)
    receipt_complete = execute_receipts(root, manifest, model, completed, attempt, checkpoint, deadline)
    write_once(root / "execution_profiles" / f"{attempt}.json", dict(
        manifest_sha256=manifest_sha, generation_assigned=len(rows), generation_terminal=len(completed),
        choice_receipts_assigned=sum(row["kind"] == "choice" for row in rows), receipts_complete=receipt_complete,
        parent_generations_required=manifest["site_plan"]["planned_denominators"]["generations"],
        parent_choice_receipts_required=manifest["site_plan"]["planned_denominators"]["choices"],
        receipt_grouping=RECEIPT_GROUPING, scientific_interpretation_requires_complete_parent_audit=True))
    return completed


def render(manifest, records):
    return (f"Satisfaction site {manifest['site_phase']}: {len(records)}/{len(manifest['diagnostics'])} "
            "terminal worker rows. Complete parent audit and all recorded gates remain required.")
