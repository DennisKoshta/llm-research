"""Fixed 15-arm curiosity suppression capture on the entire prior development set.

This module grants no scientific admission. Every worker retains its assigned
cohorts and all selected-payload receipt dispositions. Signed applied units use
positive scalar doses; layer15 captures precede the prefill patch. The unchanged
low-level S-screen observation and receipt helpers remain independently frozen.
"""

from pathlib import Path
import time
import uuid

import numpy as np

from . import curiosity_suppression_screen as screen
from . import satisfaction_capture as legacy
from . import sustained_engineering as engineering
from . import voluntary_resolution as voluntary
from .artifacts import RunStore, digest, write_once

CAPTURE_PLAN = legacy.CAPTURE_PLAN
MODEL = legacy.MODEL
GROUPS = screen.GROUPS
LAYERS = legacy.LAYERS
POSITIONS = legacy.POSITIONS
require = legacy.require
patches_for = legacy.patches_for
collect_generation = legacy.collect_generation
execute_receipts = legacy.execute_receipts
receipts = legacy.receipts
_cohort_requests = legacy._cohort_requests
_load_generation = legacy._load_generation
_save_hidden = legacy._save_hidden
_verify_hidden = legacy._verify_hidden
_artifact = legacy._artifact


def bind_preflight(plan, model):
    """Tokenize already supplied assignments; never choose worlds or fit directions."""
    requests = []
    baseline = sorted((row for row in plan["assignments"] if row["arm"] == "baseline"), key=lambda row: row["base_row"])
    require([row["base_row"] for row in baseline] == list(range(288)), "Require each source baseline exactly once")
    for row in baseline:
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
    source = manifest["source_screen_manifest"]
    require(
        manifest.get("source_screen_manifest_sha256") == digest(source)
        == plan["source_screen_manifest_sha256"], "Original source manifest binding changed"
    )
    rebuilt = screen.assignment_plan(source_manifest=source, direction_bindings=plan["direction_bindings"])
    require(plan == rebuilt, "Require the complete unchanged pure assignment plan")
    expected = screen.group_assignments(plan, manifest.get("execution_group"))
    parent = manifest.get("parent_manifest_sha256")
    require(isinstance(parent, str) and len(parent) == 64 and all(c in "0123456789abcdef" for c in parent),
            "Require the bound full-parent manifest hash")
    require(
        manifest.get("assay") == screen.ASSAY
        and manifest.get("task_revision") == screen.REVISION
        and manifest.get("model") == MODEL
        and manifest.get("execution") == screen.EXECUTION
        and manifest.get("max_new_tokens") == 4096
        and manifest.get("capture_plan") == CAPTURE_PLAN
        and manifest.get("assignments") == []
        and manifest.get("diagnostics") == expected
        and manifest.get("assignment_sha256") == digest(expected)
        and len(expected) == 1440,
        "Require every ordered child row, 4096-token sampling and layer-15 capture policy",
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
    baselines = {}
    for offset in range(0, len(rows), 32):
        cohort = rows[offset : offset + 32]
        ids = [a["trial_id"] for a in cohort]
        arm = cohort[0]["arm"]
        direction = np.asarray(
            manifest["screen_plan"]["direction_bindings"]["zero_sham" if arm == "baseline" else arm]["unit"],
            dtype=np.float32,
        )
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
            common = None
            if arm != "baseline":
                baseline = baselines[cohort[0]["base_row"]]
                baseline_hidden = (
                    None if baseline["activation_artifact"] is None else
                    _verify_hidden(root, baseline["activation_artifact"], baseline["trial_ids"], requests)
                )
                common = dict(
                    baseline_evidence_sha256=digest(baseline),
                    same_requests=baseline["requests"] == requests,
                    layer15_prepatch_captures_equal=hidden is not None and baseline_hidden is not None
                    and np.array_equal(hidden, baseline_hidden),
                )
                checks["common_prefill_identity"] = all(
                    value is True for key, value in common.items() if key != "baseline_evidence_sha256"
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
                common_prefill_comparison=common,
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
            parent_generations_required=4320,
            parent_choice_receipts_required=2160,
            scientific_interpretation_requires_complete_parent_audit=True,
        ),
    )
    return [completed[a["trial_id"]] for a in rows]
