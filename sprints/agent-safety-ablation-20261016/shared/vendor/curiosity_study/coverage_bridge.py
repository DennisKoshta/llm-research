"""Fixed scientific collection, contingent on separate engineering and budget admission.

This module does not admit a run or decide scientific success. It preserves the
entire assigned grid, the reviewed numerical collector, and the original worker
receipt policy: consecutive logical-choice groups, filtered only inside groups.
"""

from copy import deepcopy
from pathlib import Path
import time

from . import coverage_engineering as engine
from . import satisfaction_capture as legacy
from . import voluntary_capture as receipts
from . import voluntary_resolution as voluntary
from .artifacts import digest, write_once

REVISION = "curiosity-coverage-bridge-v1"
GROUPS = ("worker_0", "worker_1", "worker_2")
PAIR_KEYS = (
    "zero_sham",
    "curiosity_negative",
    "confidence_positive",
    "completion_positive",
    "valence_positive",
    "valence_negative",
    "motivation_negative",
    "novelty_negative",
    "task_demand_negative",
    "random_01",
    "random_02",
    "random_03",
    "random_04",
    "random_05",
)
ARMS = ("baseline", *(f"{schedule}__{key}" for key in PAIR_KEYS for schedule in ("record", "full")))
PINNED = dict(
    assignment_sha256="f1694430372c2c13a311580d0d4047046e7827f50bbe8cccde13ad6280470cb7",
    base_assignment_sha256="dec2ff4471e6e2f7825d80ed87194c7af2008b75663b81469ebabee746adfa0a",
    base_requests_sha256="a7e89fdf7a7be5e1ef1660b8fe67c4232d44828e440912a055d2411c2d1277b4",
    direction_bindings_sha256="3c9f7772efd04d4f8481fccdca8697beebf1522a21c5cbae9702fdf95a7929d7",
    execution_cohorts_sha256="1181066a590f8e6d9859c38d89d8f47d91f0802f453f7a9f94774ef56a562ef0",
    criteria_sha256="dc791733a5aebc8afdca72f5a98f17fc065d5b1988b2306a013422763afe11a2",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_materials(materials):
    for key, expected in PINNED.items():
        plural = {
            "assignment_sha256": "assignments",
            "base_assignment_sha256": "base_assignments",
            "base_requests_sha256": "base_requests",
            "direction_bindings_sha256": "direction_bindings",
            "execution_cohorts_sha256": "execution_cohorts",
            "criteria_sha256": "criteria",
        }[key]
        require(materials[key] == expected == digest(materials[plural]), "Frozen material differs: " + key)
    rows, cohorts = materials["assignments"], materials["execution_cohorts"]
    require(
        len(rows) == len({r["trial_id"] for r in rows}) == 8352
        and len(cohorts) == 261
        and len(materials["base_assignments"]) == len(materials["base_requests"]) == 288
        and sum(r["kind"] == "choice" for r in rows) == 4176
        and materials["dose"] == engine.DOSE
        and set(materials["direction_bindings"]) == set(PAIR_KEYS),
        "Complete fixed scientific grid is required",
    )
    for index, cohort in enumerate(cohorts):
        selected = rows[index * 32 : (index + 1) * 32]
        start = cohort["source_cohort"] * 32
        require(
            cohort["offset"] == index * 32
            and cohort["trial_ids"] == [r["trial_id"] for r in selected]
            and cohort["base_trial_ids"]
            == [r["base_trial_id"] for r in materials["base_assignments"][start : start + 32]]
            and [r["base_row"] for r in selected] == list(range(start, start + 32))
            and all(
                r["arm"] == cohort["arm"] and r["execution_group"] == cohort["execution_group"]
                for r in selected
            )
            and cohort["execution_group"] == GROUPS[cohort["source_cohort"] % 3],
            "Fixed source companions, row order or worker changed",
        )
    for group in GROUPS:
        assigned = [r for r in rows if r["execution_group"] == group]
        require(
            len(assigned) == 2784 and all(sum(r["arm"] == arm for r in assigned) == 96 for arm in ARMS),
            "Every worker must collect every arm over its three fixed source cohorts",
        )
    return materials


def scientific_call(materials, cohort):
    """Translate one fixed cohort without changing numerical engine semantics."""
    index, arm = cohort["offset"] // 32, cohort["arm"]
    require(arm in ARMS, "Unknown fixed arm")
    baseline = arm == "baseline"
    schedule, direction = ("record", None) if baseline else arm.split("__")
    zero = direction == "zero_sham"
    requests = materials["base_requests"][cohort["source_cohort"] * 32 : (cohort["source_cohort"] + 1) * 32]
    return dict(
        call_index=index,
        call_id=f"bridge-{index:03d}-{arm}",
        source_cohort=cohort["source_cohort"],
        cap=4096,
        stage="scientific",
        condition="baseline" if baseline else f"{schedule}_{'zero' if zero else 'nonzero'}",
        schedule=engine.RECORD_SCHEDULE if schedule == "record" else engine.ALL_ATTENDED_SCHEDULE,
        position=None if baseline else "record" if schedule == "record" else "all_attended",
        direction_key=direction,
        dose_multiplier=0 if zero else 1,
        one_row_only=None,
        inject_failure_at=None,
        expected_error=None,
        base_trial_ids=list(cohort["base_trial_ids"]),
        trial_ids=list(cohort["trial_ids"]),
        requests_sha256=digest(requests),
    )


def receipt_groups(materials, execution_group):
    require(execution_group in GROUPS, "Unknown worker")
    choices = [
        r
        for r in materials["assignments"]
        if r["execution_group"] == execution_group and r["kind"] == "choice"
    ]
    return [choices[offset : offset + 32] for offset in range(0, len(choices), 32)]


def relationship_checks(report, baseline, call):
    """Contemporaneous same-row prepatch/sham checks; no semantic outcome gate."""
    if call["position"] is None:
        return dict(common_prefill_identity=True, sham_identity=True)
    common = bool(
        baseline
        and baseline["forwards"]
        and report["forwards"]
        and report["requests_sha256"] == baseline["requests_sha256"]
        and [g["before"] for g in report["forwards"][0]["geometry"]]
        == [g["before"] for g in baseline["forwards"][0]["geometry"]]
    )
    sham = True
    if call["dose_multiplier"] == 0:

        def comparable(value):
            if value is None or value["rows"] is None:
                return None
            return dict(
                value,
                rows=[
                    dict(
                        row,
                        completion={
                            key: item
                            for key, item in row["completion"].items()
                            if key != "intervention_schedule"
                        },
                    )
                    for row in value["rows"]
                ],
            )

        left, right = comparable(baseline), comparable(report)
        sham = bool(left and right and engine.identical_rows(left, right, range(32)))
    return dict(common_prefill_identity=common, sham_identity=sham)


def choice_sidecar(source):
    result = legacy._choice_sidecar(source)
    if source["task_record"] is not None and source["task_record"]["cap_hit"]:
        result.update(status="capped_choice", receipt_required=False)
    return result


def collect_receipt_group(model, choices, sources, *, deadline=None):
    """One original logical-choice group, never compacted across boundaries."""
    require(
        0 < len(choices) <= 32 and all(a["kind"] == "choice" for a in choices),
        "Invalid logical receipt group",
    )
    started = time.monotonic()
    sidecars = [choice_sidecar(sources[a["trial_id"]]) for a in choices]
    active = [row for row in sidecars if row["receipt_required"]]
    active_ids = [row["source_trial_id"] for row in active]
    requests, measurements, arrays, guard, error = [], None, {}, None, None
    forward_attempted, forward_seconds, before_forward = False, 0.0, None
    try:
        require(
            model.active_hooks == 0 and not engine.hook_inventory(model), "Receipt entry hooks are not clean"
        )
        for row in active:
            payload, request, binding = receipts.render_receipt(
                sources[row["source_trial_id"]]["task_record"], model
            )
            row.update(receipt_input=payload, token_binding=binding)
            requests.append(request)
        parsing_seconds = time.monotonic() - started
        if requests:
            require(deadline is None or time.time() < deadline, "Deadline before selected receipt forwards")
            forward_attempted, before_forward = True, time.monotonic()
            measurements, hidden, guard = receipts.forward_receipts(model, requests, active_ids, layers=[15])
            forward_seconds = time.monotonic() - before_forward
            arrays["hidden"] = hidden
            require(guard["passed"] is True, "Selected receipt read-only identity failed")
            for row_index, (row, measurement) in enumerate(zip(active, measurements, strict=True)):
                executed = deepcopy(row["executed_simulator_outcome"])
                executed["receipt_model_pass_completed"] = True
                executed["observation_ledger"][-1]["model_pass_completed"] = True
                row.update(
                    status="completed",
                    receipt_model_pass_completed=True,
                    measurement=measurement,
                    activation_row=row_index,
                    executed_simulator_outcome=executed,
                )
        require(model.active_hooks == 0 and not engine.hook_inventory(model), "Receipt cleanup failed")
    except Exception as caught:
        error = dict(type=type(caught).__name__, message=str(caught))
        parsing_seconds = time.monotonic() - started if before_forward is None else parsing_seconds
        if before_forward is not None:
            forward_seconds = time.monotonic() - before_forward
        for row in active:
            row.update(
                status="execution_error",
                receipt_model_pass_completed=False,
                measurement=None,
                activation_row=None,
                executed_simulator_outcome=deepcopy(
                    sources[row["source_trial_id"]]["task_record"]["simulator_outcome"]
                ),
                error=error,
            )
    return dict(
        source_trial_ids=[a["trial_id"] for a in choices],
        active_source_trial_ids=active_ids,
        records=sidecars,
        records_sha256=digest(sidecars),
        requests=requests,
        requests_sha256=digest(requests),
        engineering=guard,
        observed_measurements=measurements,
        error=error,
        passed=error is None,
        receipt_forward_attempted=forward_attempted,
        assigned_choices=len(choices),
        actual_receipt_rows=len(requests),
        actual_forward_calls=2 if guard is not None else None if forward_attempted else 0,
        parsing_seconds=parsing_seconds,
        receipt_forward_seconds=forward_seconds,
        total_seconds=time.monotonic() - started,
        actual_padded_width=None if guard is None else guard["prefill_width"],
        grouping_policy="Fixed consecutive logical-choice groups <=32; active filtering only inside each group",
        scientific_admission=False,
    ), arrays


def _durable_report(root, stem, report, arrays):
    return engine._persist(Path(root), stem, report, arrays)


def execute(root, model, materials, *, execution_group, checkpoint=None, deadline=None):
    """One worker, no resume/retry; all parent slots remain required downstream."""
    validate_materials(materials)
    materials_sha256 = digest(materials)
    require(execution_group in GROUPS, "Unknown fixed worker")
    root = Path(root)
    require(
        not (root / "bridge-started.json").exists()
        and not any(
            (root / directory).exists()
            for directory in (
                "generation",
                "generation_cohorts",
                "generation_started",
                "receipts",
                "receipt_started",
            )
        ),
        "A bridge worker cannot be retried or resumed",
    )
    require(model.active_hooks == 0 and not engine.hook_inventory(model), "Require clean generation entry")
    metadata = model.metadata()
    require(
        all(metadata.get(key) == value for key, value in materials["model"].items()),
        "Pinned model identity differs",
    )
    assigned = [a for a in materials["assignments"] if a["execution_group"] == execution_group]
    cohorts = [c for c in materials["execution_cohorts"] if c["execution_group"] == execution_group]
    groups = receipt_groups(materials, execution_group)
    started = time.monotonic()
    write_once(
        root / "bridge-started.json",
        dict(
            revision=REVISION,
            materials_sha256=materials_sha256,
            execution_group=execution_group,
            assigned_generations=len(assigned),
            assigned_choices=sum(map(len, groups)),
            parent_generations=8352,
            parent_choices=4176,
            started=time.time(),
            deadline=deadline,
            scientific_admission=False,
            resume_allowed=False,
        ),
    )
    write_once(root / "bridge-environment.json", metadata)
    completed, receipt_records, generation_profiles, receipt_profiles = {}, {}, [], []
    failure, baseline, baseline_binding, baseline_cohort = None, None, None, None
    for cohort in cohorts:
        if deadline is not None and time.time() >= deadline:
            failure = dict(type="TimeoutError", message="Deadline before fixed generation cohort")
            break
        call = scientific_call(materials, cohort)
        index = call["call_index"]
        rows = materials["assignments"][cohort["offset"] : cohort["offset"] + 32]
        requests = materials["base_requests"][
            cohort["source_cohort"] * 32 : (cohort["source_cohort"] + 1) * 32
        ]
        binding_key = call["direction_key"] or "zero_sham"
        binding = materials["direction_bindings"][binding_key]
        require(
            call["position"] is None or baseline_cohort == cohort["source_cohort"],
            "Baseline must precede all arms in its source cohort",
        )
        write_once(
            root / "generation_started" / f"{index:03d}.json",
            dict(
                call=call,
                materials_sha256=materials_sha256,
                assignments_sha256=digest(rows),
                started=time.time(),
            ),
        )
        try:
            wall_started = time.monotonic()
            report, arrays = engine.collect_call(
                model, requests, call, binding["unit"], dose=engine.DOSE, deadline=deadline
            )
            common = relationship_checks(report, baseline, call)
            passed = report["passed"] is True and all(common.values())
            report.update(
                materials_sha256=materials_sha256,
                execution_group=execution_group,
                scientific_arm=cohort["arm"],
                direction_binding_key=binding_key,
                direction_binding_sha256=digest(binding),
                baseline_artifact=baseline_binding,
                bridge_relationship_checks=common,
                bridge_engineering_passed=passed,
            )
            artifact = _durable_report(root, f"generation/{index:03d}", report, arrays)
            records = []
            for row_index, assignment in enumerate(rows):
                task = (
                    None
                    if report["rows"] is None
                    else voluntary.record_from_thinking_inference(
                        assignment["source_assignment"], report["rows"][row_index]["completion"]
                    )
                )
                record = dict(
                    trial_id=assignment["trial_id"],
                    assignment=assignment,
                    assignment_sha256=digest(assignment),
                    kind=assignment["kind"],
                    terminal=True,
                    status="completed" if task is not None else "execution_error",
                    task_record=task,
                    generation_evidence=artifact,
                    cohort_row=row_index,
                    engineering_passed=passed,
                )
                if not passed:
                    record["failure"] = (
                        "generation_engineering_guard_failed"
                        if task is not None
                        else "generation_execution_error"
                    )
                records.append(record)
            envelope = dict(
                materials_sha256=materials_sha256,
                execution_group=execution_group,
                cohort=cohort,
                call=call,
                artifact=artifact,
                records=records,
                records_sha256=digest(records),
                engineering_passed=passed,
            )
            write_once(root / "generation_cohorts" / f"{index:03d}.json", envelope)
            completed.update({r["trial_id"]: r for r in records})
            generation_profiles.append(
                dict(
                    call_index=index,
                    source_cohort=cohort["source_cohort"],
                    arm=cohort["arm"],
                    generation_seconds=report["generation_seconds"],
                    total_collection_persistence_seconds=time.monotonic() - wall_started,
                    artifact=artifact,
                    engineering_passed=passed,
                )
            )
            if checkpoint:
                checkpoint()
            if not passed:
                failure = dict(
                    type="EngineeringGateFailure",
                    call_index=index,
                    checks=report["checks"],
                    relationship_checks=common,
                )
                break
            if call["position"] is None:
                baseline, baseline_binding, baseline_cohort = report, artifact, cohort["source_cohort"]
        except Exception as caught:
            failure = dict(type=type(caught).__name__, message=str(caught), call_index=index)
            write_once(root / "generation_failures" / f"{index:03d}.json", failure)
            if checkpoint:
                checkpoint()
            break
    if failure is None and len(completed) == len(assigned):
        for index, choices in enumerate(groups):
            if deadline is not None and time.time() >= deadline:
                failure = dict(type="TimeoutError", message="Deadline before fixed logical receipt group")
                break
            write_once(
                root / "receipt_started" / f"{index:03d}.json",
                dict(
                    materials_sha256=materials_sha256,
                    logical_choice_offset=index * 32,
                    source_trial_ids=[a["trial_id"] for a in choices],
                    started=time.time(),
                ),
            )
            try:
                report, arrays = collect_receipt_group(model, choices, completed, deadline=deadline)
                report.update(
                    materials_sha256=materials_sha256,
                    execution_group=execution_group,
                    logical_choice_offset=index * 32,
                )
                artifact = _durable_report(root, f"receipts/{index:03d}", report, arrays)
                receipt_records.update({r["source_trial_id"]: r for r in report["records"]})
                receipt_profiles.append(
                    dict(
                        logical_choice_offset=index * 32,
                        artifact=artifact,
                        total_seconds=report["total_seconds"],
                        receipt_forward_seconds=report["receipt_forward_seconds"],
                        assigned_choices=len(choices),
                        actual_receipt_rows=report["actual_receipt_rows"],
                        passed=report["passed"],
                    )
                )
                if checkpoint:
                    checkpoint()
                if not report["passed"]:
                    failure = dict(
                        type="ReceiptEngineeringFailure",
                        logical_choice_offset=index * 32,
                        error=report["error"],
                    )
                    break
            except Exception as caught:
                failure = dict(
                    type=type(caught).__name__, message=str(caught), logical_choice_offset=index * 32
                )
                write_once(root / "receipt_failures" / f"{index:03d}.json", failure)
                if checkpoint:
                    checkpoint()
                break
    generation_slots = []
    for assignment in assigned:
        record = completed.get(assignment["trial_id"])
        attempted = (root / "generation_started" / f"{assignment['parent_row'] // 32:03d}.json").exists()
        generation_slots.append(
            dict(
                trial_id=assignment["trial_id"],
                assignment_sha256=digest(assignment),
                status=record["status"] if record else "execution_error" if attempted else "missing",
                record_sha256=None if record is None else digest(record),
                engineering_passed=False if record is None else record["engineering_passed"],
            )
        )
    choice_slots = []
    for group_index, choices in enumerate(groups):
        attempted = (root / "receipt_started" / f"{group_index:03d}.json").exists()
        for assignment in choices:
            sidecar = receipt_records.get(assignment["trial_id"])
            choice_slots.append(
                dict(
                    trial_id=assignment["trial_id"],
                    assignment_sha256=digest(assignment),
                    logical_choice_offset=group_index * 32,
                    status=sidecar["status"] if sidecar else "execution_error" if attempted else "missing",
                    receipt_required=None if sidecar is None else sidecar["receipt_required"],
                    receipt_model_pass_completed=False
                    if sidecar is None
                    else sidecar["receipt_model_pass_completed"],
                    sidecar_sha256=None if sidecar is None else digest(sidecar),
                )
            )
    complete = len(completed) == len(assigned) and len(receipt_records) == len(choice_slots)
    summary = dict(
        revision=REVISION,
        materials_sha256=materials_sha256,
        execution_group=execution_group,
        parent_assigned_generations=8352,
        parent_assigned_choices=4176,
        assigned_generations=len(assigned),
        assigned_choices=len(choice_slots),
        complete=complete,
        engineering_passed=complete
        and failure is None
        and all(r["engineering_passed"] for r in completed.values()),
        generation_slots=generation_slots,
        choice_slots=choice_slots,
        generation_profiles=generation_profiles,
        receipt_profiles=receipt_profiles,
        error=failure,
        total_seconds=time.monotonic() - started,
        scientific_admission=False,
        causal_claim=False,
        parent_complete_audit_required=True,
        receipt_policy="All generation cohorts first; fixed worker logical-choice groups <=32; active filtering inside groups only",
    )
    write_once(root / "bridge-summary.json", summary)
    if checkpoint:
        checkpoint()
    return summary
