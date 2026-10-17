"""Reconcile every assigned row of the fixed, explicit-site S development stages.

No partial worker or successful subset is admitted. Shared pure task/receipt
validators retain their original semantics; all site, arm and phase traversals
are explicit here. Prefill vectors are independently observed for every row;
decode geometry remains a scalar ledger with separately bound short engineering.
"""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile

import numpy as np

from curiosity_study import satisfaction_causal_sites as study
from curiosity_study import satisfaction_site_capture as capture
from curiosity_study import satisfaction_site_execution as execution
from curiosity_study import satisfaction_site_resources as resources
from curiosity_study import causal_site_engineering as engineering
from curiosity_study import voluntary_capture as receipts
from curiosity_study import voluntary_resolution as voluntary
from curiosity_study.satisfaction_capture import _choice_sidecar
from curiosity_study.artifacts import digest, read, write_once
from curiosity_study.concept_capture import MODEL
from curiosity_study.state_backend import PREFILL_PADDING_MULTIPLE
from audit_satisfaction_screen import (
    artifact_path,
    compact as legacy_compact,
    counts_for,
    environments,
    receipt_measurement,
    runtime_policy,
    sha,
    token_model,
)

require = study.require


def compact(assignment, record):
    row = dict(legacy_compact(assignment, record), phase=assignment["phase"], site=assignment["site"])
    if "reserved_ordinal" in assignment:
        row["reserved_ordinal"] = assignment["reserved_ordinal"]
    return row


def parent_children(root, manifest):
    from analyze_satisfaction_causal_sites import ANALYSIS_METADATA

    require(
        "execution_group" not in manifest and "parent_manifest_sha256" not in manifest,
        "Scientific auditing requires the complete parent",
    )
    phase, plan = manifest["site_phase"], manifest["site_plan"]
    require(
        phase in ("calibration", "validation")
        and resources.job_wall_limit(manifest) == resources.POLICIES[phase]["shared_deadline_seconds"]
        and manifest["budget"] == resources.POLICIES[phase]
        and manifest["assay"] == study.ASSAY
        and manifest["task_revision"] == study.REVISION
        and manifest["phase"] == "development_pilot"
        and manifest["inference_scope"] == "development"
        and manifest["analysis_metadata"] == ANALYSIS_METADATA
        and manifest["model"] == MODEL
        and manifest["execution_gpu"] == "H200"
        and manifest["execution"] == study.EXECUTION
        and manifest["assignments"] == []
        and manifest["diagnostics"] == plan["assignments"]
        and len(manifest["diagnostics"]) == (4032 if phase == "calibration" else 2880)
        and manifest["assignment_sha256"] == digest(manifest["diagnostics"])
        and manifest["site_plan_sha256"] == digest(plan),
        "Whole stage policy or assignment inventory differs",
    )
    descriptors = read(root / "children.json")
    require(
        len(descriptors) == 3 and [d["execution_group"] for d in descriptors] == list(study.GROUPS),
        "Require all three assigned workers",
    )
    children, paths = [], set()
    for descriptor, group in zip(descriptors, study.GROUPS, strict=True):
        path = Path(descriptor["path"]).resolve()
        require(
            path.parent == root.resolve().parent and path != root.resolve() and path not in paths,
            "Duplicate or external child collection",
        )
        paths.add(path)
        child = read(path / "manifest.json")
        expected = deepcopy(manifest)
        expected.update(
            execution_group=group,
            parent_manifest_sha256=digest(manifest),
            diagnostics=study.group_assignments(plan, group),
        )
        expected["assignment_sha256"] = digest(expected["diagnostics"])
        require(
            child == expected and digest(child) == descriptor["manifest_sha256"],
            "Child is not the exact complete parent partition",
        )
        execution.validate_manifest(child)
        children.append((path, child))
    return children


def capture_arrays(root, artifact, ids, requests, cache, *, site, keys, width):
    key = artifact["path"]
    if key not in cache:
        with np.load(artifact_path(root, artifact), allow_pickle=False) as saved:
            data = {name: saved[name].copy() for name in saved.files}
        cache[key] = (deepcopy(artifact), data)
    previous, data = cache[key]
    require(
        previous == artifact
        and set(data)
        == set(keys) | {"layers", "positions", "trial_ids", "token_positions", "token_prefix_sha256"},
        "Capture artifact binding or field layout differs",
    )
    require(
        data["layers"].tolist() == [site]
        and data["positions"].tolist() == ["record", "final"]
        and data["trial_ids"].tolist() == ids
        and data["token_positions"].tolist()
        == [[r["prefix_positions"][p] for p in ("record", "final")] for r in requests]
        and data["token_prefix_sha256"].tolist() == [r["token_prefix_sha256"] for r in requests],
        "Capture site, exact row order or token positions differ",
    )
    arrays = {name: data[name] for name in keys}
    require(
        artifact["arrays"] == {name: engineering.array_binding(value) for name, value in arrays.items()}
        and all(
            value.dtype == np.float32 and value.shape == (len(ids), 1, 2, width) for value in arrays.values()
        ),
        "Capture dtype, hidden width or array bytes differ",
    )
    return arrays


def generation_records(root, manifest):
    assignments = execution.validate_manifest(manifest)
    require(not any((root / "attempts").rglob("*.json")), "Require atomic generation cohorts")
    records, offsets, bindings, attempts = {}, set(), [], set()
    for path in sorted((root / "cohorts").rglob("*.json")):
        require(
            len(path.relative_to(root / "cohorts").parts) == 2
            and path.stem.isascii()
            and path.stem.isdecimal(),
            "Unexpected generation cohort path",
        )
        offset = int(path.stem)
        require(
            offset in range(0, len(assignments), 32) and offset not in offsets,
            "Unassigned or duplicate cohort",
        )
        envelope, cohort = read(path), assignments[offset : offset + 32]
        rows = envelope["records"]
        require(
            envelope["manifest_hash"] == digest(manifest)
            and envelope["sha256"] == digest(rows)
            and len(rows) == 32
            and [r["assignment"] for r in rows] == cohort
            and all(
                r["terminal"] is True and r["trial_id"] == a["trial_id"]
                for r, a in zip(rows, cohort, strict=True)
            ),
            "Atomic generation membership, order or record bytes differ",
        )
        require(not set(records) & {r["trial_id"] for r in rows}, "Duplicated generation row")
        records.update({r["trial_id"]: r for r in rows})
        offsets.add(offset)
        attempts.add(path.parent.name)
        bindings.append(
            dict(path=str(path.relative_to(root)), byte_sha256=sha(path), envelope_sha256=digest(envelope))
        )
    require(len(attempts) <= 1, "A scientific worker cannot combine attempts")
    if attempts:
        started = read(root / "execution_started.json")
        require(
            started == dict(manifest_sha256=digest(manifest), attempt=next(iter(attempts))),
            "Generation attempt differs from its immutable start marker",
        )
    return records, offsets, bindings


def engagement(condition, patches):
    if condition.get("rows") is None:
        return False
    for row, patch in zip(condition["rows"], patches, strict=True):
        events = row["intervention_trace"]["events"]
        if patch is None:
            if events:
                return False
        elif not events or any(
            event["requested_scalar_delta"] != patch["dose"]
            or (
                patch["dose"] > 0
                and not (
                    np.isfinite(event["delta_norm"])
                    and event["delta_norm"] > 0
                    and np.isfinite(event["projection_after"])
                    and np.isfinite(event["projection_before"])
                    and event["projection_after"] > event["projection_before"]
                )
            )
            for event in events
        ):
            return False
    return True


def validate_generation(evidence, cohort, requests, tokenizer, eos_ids, vocabulary_size, arrays, direction):
    import torch

    site, condition = cohort[0]["site"], evidence["condition"]
    require(
        evidence["eos_token_ids"] == sorted(eos_ids) and evidence["pad_token_id"] == tokenizer.pad_token_id,
        "Generation EOS/pad identity differs",
    )
    patches = execution.patches_for(cohort, np.asarray(direction, dtype=np.float32))
    timeline = engineering.validate_condition(
        condition,
        requests,
        patches,
        layer=site,
        cap=4096,
        pad_token_id=tokenizer.pad_token_id,
        eos_ids=eos_ids,
    )
    captures = set(arrays) == {"hidden", "prefill_after"} and all(
        value.shape == (32, 1, 2, len(direction))
        and value.dtype == np.float32
        and bool(np.isfinite(value).all())
        for value in arrays.values()
    )
    if condition["rows"] is not None:
        require(
            len(condition["rows"]) == 32
            and all(frame["vocabulary_size"] == vocabulary_size for frame in condition["forwards"]),
            "Wrong generation row/full-vocabulary layout",
        )
        for row in condition["rows"]:
            tokens = row["completion"]["generated_token_ids"]
            require(
                all(type(t) is int and 0 <= t < vocabulary_size for t in tokens)
                and row["completion"]["raw"] == tokenizer.decode(tokens, skip_special_tokens=True),
                "Saved response differs from actual generated tokens",
            )
    geometry = capture.validate_prefill_geometry(condition, arrays, patches, torch=torch)
    return dict(
        timeline=bool(timeline),
        captures=bool(captures),
        prefill_geometry=bool(geometry),
        bfloat16_prefill=condition.get("activation_dtypes")
        == dict(before="torch.bfloat16", after="torch.bfloat16"),
        engagement=bool(engagement(condition, patches)),
        hook_cleanup=condition["hooks_clean"] is True,
    )


def audit_generations(root, manifest, tokenizer, eos_ids, vocabulary_size):
    records, offsets, cohort_bindings = generation_records(root, manifest)
    assignments = manifest["diagnostics"]
    require(
        execution.bind_preflight(manifest["site_plan"], token_model(tokenizer))
        == manifest["generation_preflight"],
        "Actual tokenizer differs from the frozen generation preflight",
    )
    caches, baselines, evidence_bindings, record_bindings, checks, prepatch = {}, {}, [], [], [], {}
    for offset in sorted(offsets):
        cohort = assignments[offset : offset + 32]
        site, arm, base_cohort = (cohort[0][key] for key in ("site", "arm", "base_cohort"))
        direction = manifest["site_plan"]["direction_bindings"][str(site)][
            "zero_sham" if arm == "baseline" else arm
        ]["unit"]
        ids, rows = [a["trial_id"] for a in cohort], [records[a["trial_id"]] for a in cohort]
        artifact = rows[0]["generation_evidence"]
        require(
            all(r["generation_evidence"] == artifact for r in rows), "One cohort binds different evidence"
        )
        evidence = read(artifact_path(root, artifact))
        attempt = Path(artifact["path"]).parts[1]
        started = read(root / "execution_started.json")
        requests = execution._cohort_requests(manifest, cohort)
        require(
            evidence["manifest_sha256"] == digest(manifest)
            and evidence["attempt"] == attempt == started["attempt"]
            and artifact["path"] == f"generation_evidence/{attempt}/{offset:07d}.json"
            and evidence["generation_offset"] == offset
            and evidence["site"] == site
            and evidence["arm"] == arm
            and evidence["base_cohort"] == base_cohort
            and evidence["trial_ids"] == ids
            and evidence["requests"] == requests
            and evidence["capture_plan"] == capture.capture_plan(site),
            "Generation evidence/source/site differs",
        )
        require(
            read(root / "generation_started" / f"{offset:07d}.json")
            == dict(
                manifest_sha256=digest(manifest),
                attempt=attempt,
                generation_offset=offset,
                site=site,
                arm=arm,
            ),
            "Generation start marker differs",
        )
        capture_artifact = evidence["activation_artifact"]
        arrays = (
            {}
            if capture_artifact is None
            else capture_arrays(
                root,
                capture_artifact,
                ids,
                requests,
                caches,
                site=site,
                keys=set(capture_artifact["arrays"]),
                width=len(direction),
            )
        )
        require(set(arrays) <= {"hidden", "prefill_after"}, "Unknown generation vectors")
        measured = validate_generation(
            evidence, cohort, requests, tokenizer, eos_ids, vocabulary_size, arrays, direction
        )
        condition = evidence["condition"]
        width = (
            (max(len(r["token_prefix"]) for r in requests) + PREFILL_PADDING_MULTIPLE - 1)
            // PREFILL_PADDING_MULTIPLE
        ) * PREFILL_PADDING_MULTIPLE
        if condition.get("observations_unavailable") is True:
            require(
                condition["rows"] is None
                and condition["forwards"] == []
                and condition["attempted_forwards"] is None
                and condition["error"] is not None
                and not arrays,
                "Unavailable generation fabricates observations",
            )
        else:
            require(
                condition["layer"] == site
                and condition["capture_plan"] == capture.capture_plan(site)
                and condition["capture_phase"] == capture.capture_plan(site)["source_phase"]
                and condition["request_sha256"] == digest(requests)
                and condition["source_trial_ids"] == [r["source_trial_id"] for r in requests]
                and condition["unpadded_token_positions"] == [r["prefix_positions"] for r in requests]
                and condition["padded_token_positions"]
                == [
                    [width - len(r["token_prefix"]) + r["prefix_positions"][p] for p in ("record", "final")]
                    for r in requests
                ]
                and condition["prefill_width"] == width
                and condition["arrays"] == {k: engineering.array_binding(v) for k, v in arrays.items()},
                "Actual source capture position, phase or vector binding changed",
            )
        if arm == "baseline":
            baselines[(base_cohort, site)] = (evidence, arrays)
            require(
                evidence["common_prefill_comparison"] is None and evidence["sham_comparison"] is None,
                "Baseline claims a follow-up comparison",
            )
        else:
            require((base_cohort, site) in baselines, "Follow-up lacks its preceding same-site baseline")
            baseline, previous = baselines[(base_cohort, site)]
            common = dict(
                baseline_evidence_sha256=digest(baseline),
                same_requests=baseline["requests"] == requests,
                site_prepatch_captures_equal="hidden" in arrays
                and "hidden" in previous
                and np.array_equal(arrays["hidden"], previous["hidden"]),
            )
            require(evidence["common_prefill_comparison"] == common, "Common prefill comparison changed")
            measured["common_prefill_identity"] = (
                common["same_requests"] and common["site_prepatch_captures_equal"]
            )
            if arm == "zero_sham":
                sham = dict(
                    baseline_evidence_sha256=digest(baseline),
                    same_requests=baseline["requests"] == requests,
                    full_active_logits_and_tokens_equal=condition["rows"] is not None
                    and engineering.identical_rows(baseline["condition"], condition, range(32)),
                )
                require(evidence["sham_comparison"] == sham, "Sham full-token/fingerprint comparison changed")
                measured["sham_identity"] = (
                    sham["same_requests"] and sham["full_active_logits_and_tokens_equal"]
                )
            else:
                require(evidence["sham_comparison"] is None, "Nonsham arm claims a sham comparison")
        require(
            all(type(v) is bool for v in evidence["checks"].values())
            and evidence["checks"] == measured
            and evidence["engineering_passed"] is all(measured.values()),
            "Saved engineering checks differ from observations",
        )
        for index, (assignment, row) in enumerate(zip(cohort, rows, strict=True)):
            observed = None if condition["rows"] is None else condition["rows"][index]
            expected_inner = (
                None
                if observed is None
                else voluntary.record_from_thinking_inference(
                    assignment["source_assignment"], observed["completion"]
                )
            )
            failure = (
                "execution_error"
                if observed is None
                else "generation_engineering_guard_failed"
                if not evidence["engineering_passed"]
                else None
            )
            require(
                digest(row["task_record"]) == digest(expected_inner)
                and row["cohort_row"] == index
                and row["activation_artifact"] == capture_artifact
                and row["engineering_passed"] is evidence["engineering_passed"]
                and row["status"] == ("execution_error" if observed is None else "completed")
                and row.get("failure") == failure,
                "Terminal task parsing, failure or evidence disposition changed",
            )
            record_bindings.append(
                dict(
                    trial_id=row["trial_id"],
                    record_sha256=digest(row),
                    task_record_sha256=digest(expected_inner),
                    evidence_sha256=digest(evidence),
                    artifact_sha256=None if capture_artifact is None else capture_artifact["sha256"],
                    row=index,
                )
            )
            prepatch[assignment["trial_id"]] = (
                None if "hidden" not in arrays else arrays["hidden"][index].copy()
            )
        checks.append(dict(offset=offset, site=site, arm=arm, **measured))
        evidence_bindings.append(dict(**artifact, content_sha256=digest(evidence)))
    return dict(
        records=records,
        complete=offsets == set(range(0, len(assignments), 32)),
        checks=checks,
        cohorts=cohort_bindings,
        evidence=evidence_bindings,
        record_bindings=record_bindings,
        capture_artifacts=[binding for binding, _ in caches.values()],
        prepatch=prepatch,
    )


def audit_receipts(root, manifest, sources, tokenizer):
    groups = {g["generation_offset"]: g for g in execution.receipt_groups(manifest["diagnostics"])}
    offsets, cache, bindings, all_rows, gates = set(), {}, [], [], []
    generation_attempts = {path.parent.name for path in (root / "cohorts").rglob("*.json")}
    model = token_model(tokenizer)
    for path in sorted((root / "receipt_cohorts").rglob("*.json")):
        require(len(path.relative_to(root / "receipt_cohorts").parts) == 2, "Unexpected receipt cohort path")
        require(
            generation_attempts == {path.parent.name},
            "Receipt attempt differs from the child's sole verified generation attempt",
        )
        envelope = read(path)
        offset = envelope["generation_offset"]
        require(path.name == f"{offset:07d}.json", "Receipt filename differs from its logical offset")
        require(
            type(offset) is int and offset in groups and offset not in offsets,
            "Duplicate or unassigned receipt cohort",
        )
        group = groups[offset]
        metadata = {key: value for key, value in group.items() if key != "choices"}
        site = group["site"]
        ids = [a["trial_id"] for a in group["choices"]]
        require(
            all(envelope.get(key) == value for key, value in metadata.items())
            and envelope["attempt"] == path.parent.name
            and envelope["receipt_grouping"] == execution.RECEIPT_GROUPING,
            "Receipt generation-group, site, or attempt changed",
        )
        require(
            read(root / "receipt_started" / f"{offset:07d}.json")
            == dict(manifest_sha256=digest(manifest), attempt=path.parent.name, **metadata),
            "Receipt start marker differs from its generation group",
        )
        rows = envelope["records"]
        require(
            envelope["manifest_sha256"] == digest(manifest)
            and envelope["source_trial_ids"] == ids
            and [row["source_trial_id"] for row in rows] == ids
            and envelope["records_sha256"] == digest(rows)
            and envelope["rendering_policy"] == "fresh_hf_conversation_render_nonreplay"
            and all(tid in sources for tid in ids),
            "Receipt envelope assignment/source binding differs",
        )
        offsets.add(offset)
        requests, active, base_rows = [], [], {}
        for row in rows:
            source = sources[row["source_trial_id"]]
            base = _choice_sidecar(source)
            base_rows[row["source_trial_id"]] = base
            for key in (
                "schema_version",
                "source_trial_id",
                "source_record_sha256",
                "source_task_record_sha256",
                "assignment_sha256",
                "inner_source_trial_id",
                "selected_service",
                "simulator_outcome_sha256",
                "receipt_required",
                "terminal",
            ):
                require(row.get(key) == base.get(key), "Receipt source or selected action changed")
            if not base["receipt_required"]:
                require(row == base, "Skipped or unavailable receipt fabricated a model pass")
                continue
            receipt, request, binding = receipts.render_receipt(source["task_record"], model)
            if row["receipt_input"] is not None or row["token_binding"] is not None:
                require(
                    row["receipt_input"] == receipt and row["token_binding"] == binding,
                    "Receipt includes a wrong payload or rendered prefix",
                )
            active.append(row)
            requests.append(request)
        active_ids = [row["source_trial_id"] for row in active]
        require(envelope["active_source_trial_ids"] == active_ids, "Active receipt row order changed")
        if envelope["error"] is None:
            require(
                all(
                    row.get("failure") != "receipt_execution_error" and row.get("error") is None
                    for row in active
                ),
                "Receipt failure is missing its envelope error",
            )
        else:
            require(
                bool(active)
                and all(
                    row["status"] == "execution_error"
                    and row.get("failure") == "receipt_execution_error"
                    and row.get("error") == envelope["error"]
                    for row in active
                ),
                "Receipt envelope error differs from retained active failure statuses",
            )
        guard = envelope["engineering"]
        observed = envelope["observed_measurements"]
        artifact = envelope["observed_activation_artifact"]
        arrays = (
            None
            if artifact is None
            else capture_arrays(
                root,
                artifact,
                active_ids,
                requests,
                cache,
                site=site,
                keys={"hidden"},
                width=len(manifest["site_plan"]["direction_bindings"][str(site)]["zero_sham"]["unit"]),
            )
        )
        passed = not active
        if not active:
            require(
                guard is None and observed is None and artifact is None,
                "Inactive receipts claim a model call",
            )
        elif guard is not None:
            width = (
                (max(len(r["token_prefix"]) for r in requests) + PREFILL_PADDING_MULTIPLE - 1)
                // PREFILL_PADDING_MULTIPLE
            ) * PREFILL_PADDING_MULTIPLE
            require(
                set(guard["checks"]) == set(receipts.ENGINEERING_CHECKS)
                and all(type(value) is bool for value in guard["checks"].values())
                and guard["passed"] is all(guard["checks"].values())
                and guard["source_trial_ids"] == active_ids
                and guard["active_batch_size"] == len(active)
                and guard["layers"] == [site]
                and guard["positions"] == ["record", "final"]
                and guard["max_new_tokens"] == 0
                and guard["prefill_width"] == width
                and guard["token_prefix_sha256"] == [r["token_prefix_sha256"] for r in requests],
                "Receipt guard metadata or saved checks changed",
            )
            before, after = (
                guard["baseline_full_initial_logits_sha256"],
                guard["captured_full_initial_logits_sha256"],
            )
            valid_hashes = len(before) == len(after) == len(active) and all(
                isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")
                for value in before + after
            )
            require(
                guard["checks"]["full_initial_logits_equal"] is (before == after),
                "Receipt identity check contradicts full-vocabulary fingerprints",
            )
            if arrays is None:
                require(
                    envelope["error"] is not None
                    and all(row["status"] == "execution_error" for row in active),
                    "Missing receipt capture artifact claims successful delivery",
                )
            else:
                require(
                    guard["captured_array_sha256"] == receipts.array_sha(arrays["hidden"])
                    and arrays["hidden"].shape[-1]
                    == len(manifest["site_plan"]["direction_bindings"][str(site)]["zero_sham"]["unit"]),
                    "Receipt capture arrays are unbound",
                )
            require(
                isinstance(observed, list) and len(observed) == len(active),
                "Missing observed receipt measurements",
            )
            measurements_valid = all(
                receipt_measurement(measurement, request, tokenizer, width)
                for measurement, request in zip(observed, requests, strict=True)
            )
            passed = (
                guard["passed"] is True
                and valid_hashes
                and measurements_valid
                and arrays is not None
                and bool(np.isfinite(arrays["hidden"]).all())
            )
        else:
            require(observed is None and artifact is None, "Receipt observations lack their guard")
        passed = passed and envelope["error"] is None
        for index, row in enumerate(active):
            base = base_rows[row["source_trial_id"]]
            if row["status"] == "execution_error":
                require(
                    row["receipt_model_pass_completed"] is False
                    and row["measurement"] is None
                    and row["activation_artifact"] is None
                    and row["cohort"] is None
                    and digest(row["executed_simulator_outcome"])
                    == digest(base["executed_simulator_outcome"])
                    and row.get("failure") == "receipt_execution_error",
                    "Failed receipt claims a completed selected-payload pass",
                )
                passed = False
                continue
            executed = deepcopy(base["executed_simulator_outcome"])
            executed["receipt_model_pass_completed"] = True
            executed["observation_ledger"][-1]["model_pass_completed"] = True
            require(
                passed
                and row["status"] == "completed"
                and row["receipt_model_pass_completed"] is True
                and row["measurement"] == observed[index]
                and row["activation_artifact"] == artifact
                and row["cohort"]
                == dict(
                    **metadata,
                    active_row=index,
                    active_size=len(active),
                    prefill_width=guard["prefill_width"],
                )
                and row["receipt_input"] is not None
                and row["token_binding"] is not None
                and digest(row["executed_simulator_outcome"]) == digest(executed)
                and "failure" not in row,
                "Completed receipt lacks the actual bound selected-payload forward",
            )
        gates.append(
            dict(
                offset=offset,
                site=site,
                passed=bool(passed and all(row["status"] != "execution_error" for row in rows)),
            )
        )
        all_rows.extend(rows)
        bindings.append(
            dict(path=str(path.relative_to(root)), byte_sha256=sha(path), envelope_sha256=digest(envelope))
        )
    return dict(
        complete=offsets == set(groups),
        passed=len(gates) == len(groups) and all(row["passed"] for row in gates),
        checks=gates,
        slots=len(all_rows),
        status_counts=dict(Counter(row["status"] for row in all_rows)),
        cohorts=bindings,
        record_bindings=[
            dict(source_trial_id=row["source_trial_id"], sidecar_sha256=digest(row)) for row in all_rows
        ],
        capture_artifacts=[binding for binding, _ in cache.values()],
    )


def readiness(plan, outcomes, site):
    """Use every assigned slot; readiness and effect-consistency strata differ."""
    assigned = [a for a in plan["assignments"] if a["site"] == site]
    rows = [row for row in outcomes if row["site"] == site]
    require(
        len(rows) == len(assigned)
        and len({r["trial_id"] for r in rows}) == len(rows)
        and {r["trial_id"] for r in rows} == {a["trial_id"] for a in assigned},
        "Readiness cannot drop assigned site rows",
    )
    calibration = plan["phase"] == "calibration"
    strata = sorted({r["panel_id"] for r in rows}) if calibration else ["all_reserved"]
    diagnostics, choices = [], []
    for arm in plan["arms"]:
        for panel in strata:
            subset = [r for r in rows if r["arm"] == arm and (not calibration or r["panel_id"] == panel)]
            for kind in voluntary.DIAGNOSTICS:
                counts = counts_for([r for r in subset if r["kind"] == kind])
                n, correct = (12, 11) if calibration else (16, 15)
                diagnostics.append(
                    dict(
                        site=site,
                        arm=arm,
                        panel_id=panel,
                        kind=kind,
                        **counts,
                        passed=counts["assigned"] == counts["valid"] == n
                        and counts["correct"] >= correct
                        and counts["cap_hits"] == counts["missing"] == counts["execution_failures"] == 0,
                    )
                )
            for delivery in voluntary.DELIVERIES:
                counts = counts_for(
                    [r for r in subset if r["kind"] == "choice" and r["delivery"] == delivery]
                )
                n, valid = (24, 23) if calibration else (32, 31)
                choices.append(
                    dict(
                        site=site,
                        arm=arm,
                        panel_id=panel,
                        delivery=delivery,
                        **counts,
                        passed=counts["assigned"] == n
                        and counts["valid"] >= valid
                        and counts["cap_hits"] == counts["missing"] == counts["execution_failures"] == 0,
                    )
                )
    return dict(
        assigned=len(rows),
        diagnostic_cells=diagnostics,
        choice_cells=choices,
        passed=all(r["passed"] for r in diagnostics + choices),
        choice_correctness=None,
        scope="Per-panel calibration gates; whole16-world validation gates, distinct from effect halves.",
    )


def cross_arm_identity(plan, vectors, site):
    groups = {}
    for assignment in plan["assignments"]:
        if assignment["site"] == site:
            groups.setdefault(assignment["base_trial_id"], {})[assignment["arm"]] = vectors.get(
                assignment["trial_id"]
            )
    rows = []
    for base_id, arms in sorted(groups.items()):
        available = set(arms) == set(plan["arms"]) and all(
            v is not None and np.isfinite(v).all() for v in arms.values()
        )
        identical = available and all(
            v.dtype == arms["baseline"].dtype
            and v.shape == arms["baseline"].shape
            and v.tobytes() == arms["baseline"].tobytes()
            for v in arms.values()
        )
        rows.append(
            dict(
                base_trial_id=base_id,
                available=bool(available),
                identical=bool(identical),
                arm_array_sha256={
                    arm: None if v is None else receipts.array_sha(v) for arm, v in arms.items()
                },
            )
        )
    return dict(
        site=site,
        assigned_base_rows=len(plan["base_assignments"]),
        rows=rows,
        passed=len(rows) == len(plan["base_assignments"]) and all(r["identical"] for r in rows),
    )


def frozen_documents(root, manifest):
    """Read only explicitly frozen regular documents and verify reused code bytes."""
    frozen = read(root / "freeze.json")
    require(
        frozen["manifest_sha256"] == digest(manifest)
        and frozen["source_sha256"] == manifest["source_sha256"]
        and frozen["archive_sha256"] == sha(root / "source.tar.gz"),
        "Scientific source freeze changed",
    )
    bindings = manifest["bound_receipts"]
    require(
        len({b["role"] for b in bindings}) == len(bindings)
        and len({b["path"] for b in bindings}) == len(bindings),
        "Duplicate evidence role or path",
    )
    documents, roles = {}, {}
    with tarfile.open(root / "source.tar.gz") as archive:
        files = {m.name: m for m in archive.getmembers() if m.isfile()}
        require(
            len(files) == sum(m.isfile() for m in archive.getmembers()), "Duplicate archived regular path"
        )
        source = {
            name.removeprefix("src/curiosity_study/"): archive.extractfile(member).read().decode()
            for name, member in files.items()
            if name.startswith("src/curiosity_study/") and name.endswith(".py")
        }
        require(digest(source) == manifest["source_sha256"], "Archived source inventory differs")
        for name, byte_sha in frozen["documents"].items():
            path = Path(name)
            require(
                not path.is_absolute() and ".." not in path.parts and name in files, "Unsafe frozen document"
            )
            payload = archive.extractfile(files[name]).read()
            require(hashlib.sha256(payload).hexdigest() == byte_sha, "Frozen document bytes changed")
            documents[name] = payload
        for binding in bindings:
            payload = documents[binding["path"]]
            value = json.loads(payload) if Path(binding["path"]).suffix == ".json" else None
            require(
                hashlib.sha256(payload).hexdigest() == binding["byte_sha256"]
                and binding["content_sha256"] == (None if value is None else digest(value)),
                "Evidence role bytes changed",
            )
            roles[binding["role"]] = dict(binding=binding, value=value)
        # Each imported numerical/source validator is copied in the source archive.
        from curiosity_study import (
            artifacts,
            concept_capture,
            curiosity_suppression_screen,
            identification,
            information_state,
            satisfaction_capture,
            satisfaction_screen,
            state_backend,
            sustained_engineering,
            thinking_information,
        )
        from curiosity_study import satisfaction_site_setup, reserved_object_materials

        for module in (
            study,
            capture,
            execution,
            resources,
            engineering,
            receipts,
            voluntary,
            satisfaction_capture,
            satisfaction_screen,
            state_backend,
            sustained_engineering,
            satisfaction_site_setup,
            reserved_object_materials,
            artifacts,
            concept_capture,
            curiosity_suppression_screen,
            identification,
            information_state,
            thinking_information,
        ):
            name = f"src/curiosity_study/{Path(module.__file__).name}"
            require(
                hashlib.sha256(archive.extractfile(files[name]).read()).hexdigest() == sha(module.__file__),
                "Current source validator differs from the scientific freeze",
            )
        for name in (
            Path(__file__).name,
            "audit_satisfaction_site_setup.py",
            "satisfaction_site_sources.py",
            "audit_satisfaction_screen.py",
            "prepare_satisfaction_site_setup.py",
            "prepare_satisfaction_causal_sites.py",
            "analyze_satisfaction_causal_sites.py",
            "check_satisfaction_site_decision.py",
            "prepare_curiosity_suppression_screen.py",
            "prepare_satisfaction_screen.py",
            "prepare_sustained_engineering.py",
            "prepare_voluntary_resolution.py",
        ):
            require(
                hashlib.sha256(archive.extractfile(files[f"scripts/{name}"]).read()).hexdigest()
                == sha(Path(__file__).with_name(name)),
                "Current admission/audit helper differs from freeze",
            )
    return frozen, documents, roles


def provenance(root, manifest, tokenizer):
    import audit_satisfaction_site_setup as setup_audit
    from satisfaction_site_sources import SOURCE, locked_directions
    from prepare_satisfaction_causal_sites import direction_inventory
    from prepare_satisfaction_site_setup import DESIGN_SHA

    frozen, documents, roles = frozen_documents(root, manifest)
    require(
        {
            "prospective_plan",
            "prospective_design",
            "resource_forecast",
            "direction_inventory",
            "exposure_receipt",
            "reserved_materials",
            "generation_preflight",
            "receipt_preflight",
        }
        <= set(roles),
        "Missing prospective bound evidence roles",
    )
    require(
        roles["prospective_design"]["binding"]["byte_sha256"] == DESIGN_SHA,
        "Original prospective design commitment changed",
    )
    for key in ("generation_preflight", "receipt_preflight", "reserved_materials"):
        require(roles[key]["value"] == manifest[key], "Bound input document differs from manifest")
    certificate = manifest["setup_certificate"]
    require(
        digest(certificate) == manifest["setup_certificate_sha256"]
        and certificate["passed"] is True
        and certificate["full_cap_certified"] is False,
        "Short setup certificate changed",
    )
    require(
        all(
            certificate[key] in documents
            for key in ("manifest_path", "audit_path", "freeze_path", "archive_path")
        )
        and certificate["run"] == str(Path(certificate["manifest_path"]).parent)
        and certificate["manifest_path"] == certificate["run"] + "/manifest.json"
        and certificate["freeze_path"] == certificate["run"] + "/freeze.json"
        and certificate["archive_path"] == certificate["run"] + "/source.tar.gz",
        "Setup certificate references unbundled or inconsistent paths",
    )
    with tempfile.TemporaryDirectory(prefix="satisfaction-site-audit-") as temporary:
        folder = Path(temporary)
        for name, payload in documents.items():
            path = folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        setup_root = (folder / certificate["manifest_path"]).parent
        old_manifest, old_freeze = (
            read(folder / certificate["manifest_path"]),
            read(folder / certificate["freeze_path"]),
        )
        saved = read(folder / certificate["audit_path"])
        require(
            digest(old_manifest) == certificate["manifest_sha256"]
            and old_manifest["source_sha256"] == certificate["source_sha256"]
            and digest(old_freeze) == certificate["freeze_sha256"]
            and sha(folder / certificate["archive_path"]) == certificate["archive_byte_sha256"]
            and digest(saved) == certificate["audit_sha256"]
            and sha(folder / certificate["audit_path"]) == certificate["audit_byte_sha256"],
            "Setup certificate does not bind its frozen artifacts",
        )
        recomputed = setup_audit.audit(setup_root)
        require(
            recomputed == saved
            and saved["passed"] is True
            and saved["complete"] is True
            and saved["dose_reference"]
            == certificate["dose_reference"]
            == manifest["site_plan"]["dose_reference"]
            and digest(saved["dose_reference"]) == certificate["dose_reference_sha256"],
            "Setup audit or dose formula does not independently reproduce",
        )
        setup_docs, _ = setup_audit.frozen_documents(setup_root, old_manifest)
        prior = json.loads(setup_docs[SOURCE + "/manifest.json"])
        require(
            prior == manifest["source_screen_manifest"] and digest(prior) == study.PRIOR_MANIFEST_SHA256,
            "Original S source was replaced",
        )
        source_certificate = old_manifest["source_certificate"]["provenance"]["source_certificate"]
        require(
            manifest["source_screen_certificate"] == source_certificate,
            "Original S source certificate changed",
        )
        directions, candidate = locked_directions(folder, doses=saved["dose_reference"]["doses"])
        expected = {
            str(site): {
                arm: directions[str(site)][arm] for arm in manifest["site_plan"]["arms"] if arm != "baseline"
            }
            for site in manifest["site_plan"]["sites"]
        }
        require(
            manifest["site_plan"]["direction_bindings"] == expected
            and manifest["candidate_source"] == candidate,
            "Original signed A/random directions or candidate provenance changed",
        )
        inventory = direction_inventory(expected)
        require(
            roles["direction_inventory"]["value"] == inventory
            and digest(inventory) == manifest["direction_inventory_sha256"],
            "Direction geometry inventory changed",
        )
        require(
            digest(manifest["reserved_materials"]) == saved["reserved_materials_sha256"],
            "Reserved inventory changed after the premeasurement setup",
        )
        setup_policy_rows = []
        for item in saved["stages"].values():
            wrapper = read(setup_root / item["report_artifact"]["path"])
            setup_policy_rows.append(
                runtime_policy(read(artifact_path(setup_root, wrapper["environment_artifact"])))
            )
        require(
            len(setup_policy_rows) == 3 and all(p == setup_policy_rows[0] for p in setup_policy_rows),
            "Setup stages used different runtime policies",
        )
        selection = None
        if manifest["site_phase"] == "validation":
            require(
                all(
                    manifest["selection_receipt"][key] in documents
                    for key in (
                        "manifest_path",
                        "audit_path",
                        "analysis_path",
                        "independent_check_path",
                        "freeze_path",
                        "archive_path",
                    )
                ),
                "Selection references an unbundled document",
            )
            selection = verify_selection(folder, manifest)
        exposure = verify_exposure(manifest, roles["exposure_receipt"]["value"], folder)
    forecast = roles["resource_forecast"]["value"]
    liabilities = {
        phase: policy["workers"] * (policy["shared_deadline_seconds"] + 120) * 0.00139376
        for phase, policy in resources.POLICIES.items()
    }
    require(
        digest(forecast) == manifest["resource_forecast_sha256"]
        and forecast["policies"] == resources.POLICIES
        and forecast["campaign_ceiling_usd"] == 65
        and forecast["paid_retries"] == 0
        and forecast["rate_usd_per_second"] == 0.00139376
        and forecast["maximum_reserved_compute_usd"] == liabilities
        and all(
            value <= resources.POLICIES[phase]["stage_ceiling_usd"] for phase, value in liabilities.items()
        ),
        "Resource forecast or finite stage liabilities changed",
    )
    model = token_model(tokenizer)
    require(
        execution.bind_preflight(manifest["site_plan"], model) == manifest["generation_preflight"]
        and execution.bind_receipt_preflight(manifest["site_plan"], model) == manifest["receipt_preflight"],
        "Actual pinned tokenizer does not reproduce complete preflights",
    )
    return dict(
        freeze_sha256=digest(frozen),
        archive_sha256=frozen["archive_sha256"],
        setup_audit_sha256=digest(saved),
        dose_reference_sha256=digest(saved["dose_reference"]),
        source_manifest_sha256=digest(prior),
        candidate_source=candidate,
        reserved_materials_sha256=digest(manifest["reserved_materials"]),
        setup_runtime_policy=setup_policy_rows[0],
        selection=selection,
        exposure_receipt_sha256=digest(exposure),
        bound_receipts_sha256=digest(manifest["bound_receipts"]),
        historical_scope="Exact bundled setup evidence rechecked; original history census remains its separately bound premeasurement receipt.",
    )


def verify_exposure(manifest, receipt, folder):
    """Reconcile the frozen incremental ledger, without claiming a live census replay."""
    history = manifest["reserved_materials"]["history"]
    registry = receipt["registry"]
    require(
        receipt["passed"] is True
        and receipt["scientific_admission"] is False
        and receipt["closing_inventory_unchanged"] is True
        and digest(receipt) == manifest["exposure_receipt_sha256"]
        and receipt["initial_history_sha256"] == digest(history)
        and receipt["reserved_materials_sha256"] == digest(manifest["reserved_materials"])
        and receipt["registry_sha256"] == digest(registry)
        and len({r["run"] for r in registry}) == len(registry)
        and receipt["newly_executed_worlds_outside_initial_history"]
        == receipt["reserved_exact_overlap_count"]
        == 0
        and receipt["source_subset"] == study.CALIBRATION_BASE_SHA256,
        "Incremental exposure receipt changed",
    )
    by_run = {row["run"]: row for row in registry}
    ignored = {"input_exposure_included", "explicitly_active_assigned_parent"}
    old = {
        row["run"]: {k: v for k, v in row.items() if k not in ignored}
        for row in history["historical_registry"]
    }
    require(
        all(by_run.get(name) == value for name, value in old.items()), "Frozen historical inventory changed"
    )
    setup = manifest["setup_certificate"]
    allowed = {setup["run"]: setup["manifest_sha256"]}
    if manifest["site_phase"] == "validation":
        selected = manifest["selection_receipt"]
        calibration = read(folder / selected["manifest_path"])
        allowed[selected["calibration_run"]] = digest(calibration)
        for group in study.GROUPS:
            child = deepcopy(calibration)
            child.update(
                parent_manifest_sha256=digest(calibration),
                execution_group=group,
                diagnostics=study.group_assignments(calibration["site_plan"], group),
            )
            child["assignment_sha256"] = digest(child["diagnostics"])
            allowed[selected["calibration_run"] + "-" + group.replace("_", "-")] = digest(child)
    require(
        receipt["allowed_known_world_runs"] == allowed
        and all(
            name in by_run and by_run[name]["manifest_sha256"] == expected
            for name, expected in allowed.items()
        ),
        "Unexpected incremental executed collection",
    )
    require(
        not any(
            row["run"] not in old and row["run"] not in allowed and row["execution_evidence"] != "none"
            for row in registry
        ),
        "Unreviewed execution appeared after reserved materials were fixed",
    )
    return receipt


def verify_selection(folder, manifest):
    """A complete checked calibration selects one fixed site; no raw-data refit."""
    from check_satisfaction_site_decision import check as independent_check

    receipt = manifest["selection_receipt"]
    require(digest(receipt) == manifest["selection_receipt_sha256"], "Selection receipt changed")
    calibration = read(folder / receipt["manifest_path"])
    audited = read(folder / receipt["audit_path"])
    analysis = read(folder / receipt["analysis_path"])
    checked = read(folder / receipt["independent_check_path"])
    frozen = read(folder / receipt["freeze_path"])
    require(
        digest(calibration) == receipt["manifest_sha256"]
        and calibration["assay"] == study.ASSAY
        and calibration["site_phase"] == "calibration"
        and calibration["site_plan"]["phase"] == "calibration"
        and calibration["site_plan"]["sites"] == list(study.SITES)
        and receipt["manifest_path"] == receipt["calibration_run"] + "/manifest.json"
        and receipt["freeze_path"] == receipt["calibration_run"] + "/freeze.json"
        and receipt["archive_path"] == receipt["calibration_run"] + "/source.tar.gz"
        and digest(frozen) == receipt["freeze_sha256"]
        and frozen["manifest_sha256"] == digest(calibration)
        and frozen["source_sha256"] == receipt["source_sha256"] == calibration["source_sha256"]
        and frozen["archive_sha256"]
        == receipt["archive_byte_sha256"]
        == sha(folder / receipt["archive_path"])
        and digest(audited) == receipt["audit_sha256"]
        and sha(folder / receipt["audit_path"]) == receipt["audit_byte_sha256"]
        and audited["parent_manifest_sha256"] == digest(calibration)
        and audited["complete"] is True
        and audited["integrity_passed"] is True
        and digest(analysis) == receipt["analysis_sha256"]
        and sha(folder / receipt["analysis_path"]) == receipt["analysis_byte_sha256"]
        and analysis["parent_manifest_sha256"] == digest(calibration)
        and analysis["audit_sha256"] == digest(audited)
        and digest(checked) == receipt["independent_check_sha256"]
        and sha(folder / receipt["independent_check_path"]) == receipt["independent_check_byte_sha256"]
        and checked["passed"] is True
        and checked["parent_manifest_sha256"] == digest(calibration)
        and checked["audit_sha256"] == digest(audited)
        and checked["analysis_sha256"] == digest(analysis)
        and checked["analysis_byte_sha256"] == sha(folder / receipt["analysis_path"])
        and checked["audit_byte_sha256"] == sha(folder / receipt["audit_path"])
        and receipt["selected_site"] == manifest["selected_site"] == analysis["decision"]["selected_site"]
        and receipt["selected_site"] in study.SITES
        and analysis["decision"]["positive_screen"] is True
        and receipt["decision"] == analysis["decision"] == checked["decision"],
        "Validation selection lacks complete independently checked calibration",
    )
    require(
        calibration["reserved_materials"] == manifest["reserved_materials"]
        and calibration["setup_certificate"] == manifest["setup_certificate"]
        and calibration["site_plan"]["dose_reference"] == manifest["site_plan"]["dose_reference"],
        "Validation changed reserved materials or the premeasurement dose",
    )
    reproduced = independent_check(
        folder / receipt["calibration_run"], folder / receipt["analysis_path"], folder / receipt["audit_path"]
    )
    require(
        all(
            checked[key] == reproduced[key]
            for key in (
                "checker_byte_sha256",
                "assigned",
                "compared_values",
                "decision",
                "recomputed_decision",
            )
        ),
        "Selected site and task readiness do not independently reproduce",
    )
    return deepcopy(receipt)


def audit(root, tokenizer=None, eos_token_ids=None, vocabulary_size=None):
    root = Path(root).resolve()
    manifest = read(root / "manifest.json")
    children = parent_children(root, manifest)
    if tokenizer is None or eos_token_ids is None or vocabulary_size is None:
        from transformers import AutoTokenizer, AutoConfig, GenerationConfig

        options = dict(revision=MODEL["revision"], local_files_only=True)
        tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            MODEL["model_id"], trust_remote_code=False, **options
        )
        if eos_token_ids is None:
            eos = GenerationConfig.from_pretrained(MODEL["model_id"], **options).eos_token_id
            eos_token_ids = set(eos if isinstance(eos, list) else [] if eos is None else [eos])
        if vocabulary_size is None:
            vocabulary_size = AutoConfig.from_pretrained(MODEL["model_id"], **options).vocab_size
    bound = provenance(root, manifest, tokenizer)
    records, vectors, child_checks, policies = {}, {}, [], []
    for path, child in children:
        generated = audit_generations(path, child, tokenizer, set(eos_token_ids), vocabulary_size)
        received = audit_receipts(path, child, generated["records"], tokenizer)
        env = environments(path, child, generated)
        policies.extend(item["policy"] for item in env)
        require(not set(records) & set(generated["records"]), "Duplicate rows across workers")
        records.update(generated.pop("records"))
        vectors.update(generated.pop("prepatch"))
        child_checks.append(
            dict(
                execution_group=child["execution_group"],
                manifest_sha256=digest(child),
                generation=generated,
                receipts=received,
                environments=env,
            )
        )
    plan = manifest["site_plan"]
    outcomes = [compact(a, records.get(a["trial_id"])) for a in manifest["diagnostics"]]
    complete = all(c["generation"]["complete"] and c["receipts"]["complete"] for c in child_checks)
    policies_equal = len(policies) == 3 and all(
        policy == bound["setup_runtime_policy"] for policy in policies
    )
    checks = [row for child in child_checks for row in child["generation"]["checks"]]
    receipt_checks = [row for child in child_checks for row in child["receipts"]["checks"]]
    site_checks, identities = {}, {}
    for site in plan["sites"]:
        selected = [r for r in checks if r["site"] == site]
        receipts_for_site = [r for r in receipt_checks if r["site"] == site]
        ncohorts = len(plan["base_assignments"]) // 32
        nconditions = ncohorts * len(plan["arms"])
        sham = [r for r in selected if r["arm"] == "zero_sham"]
        identity = cross_arm_identity(plan, vectors, site)
        identities[str(site)] = identity
        site_checks[str(site)] = dict(
            sham_identity_passed=len(sham) == ncohorts and all(r["sham_identity"] for r in sham),
            engagement_passed=len(selected) == nconditions
            and all(r["engagement"] and r["prefill_geometry"] for r in selected),
            receipts_passed=len(receipts_for_site) == nconditions
            and all(r["passed"] for r in receipts_for_site),
            cross_arm_prepatch_identity_passed=identity["passed"],
            worker_policies_equal=bool(policies_equal),
            readiness=readiness(plan, outcomes, site),
        )
    engineering_passed = all(
        all(
            s[key]
            for key in (
                "sham_identity_passed",
                "engagement_passed",
                "receipts_passed",
                "cross_arm_prepatch_identity_passed",
                "worker_policies_equal",
            )
        )
        for s in site_checks.values()
    )
    integrity = bool(
        complete
        and policies_equal
        and len(checks) == plan["planned_denominators"]["generation_cohorts"]
        and all(
            all(
                r[key]
                for key in ("timeline", "captures", "prefill_geometry", "bfloat16_prefill", "hook_cleanup")
            )
            and r.get("common_prefill_identity", True)
            for r in checks
        )
        and engineering_passed
    )
    return dict(
        schema_version=1,
        assay="satisfaction_causal_sites_audit",
        site_phase=manifest["site_phase"],
        parent_manifest_sha256=digest(manifest),
        source_sha256=manifest["source_sha256"],
        auditor_byte_sha256=sha(__file__),
        assigned=len(outcomes),
        assigned_choices=sum(a["kind"] == "choice" for a in manifest["diagnostics"]),
        terminal_generations=len(records),
        terminal_receipt_slots=sum(c["receipts"]["slots"] for c in child_checks),
        complete=bool(complete),
        integrity_passed=integrity,
        worker_policies_equal=bool(policies_equal),
        site_checks=site_checks,
        passed=bool(integrity and all(s["readiness"]["passed"] for s in site_checks.values())),
        provenance=bound,
        children=child_checks,
        cross_arm_prepatch_identity=identities,
        outcomes=outcomes,
        outcomes_sha256=digest(outcomes),
        validation_scope="Complete explicit-site grid, frozen task scoring, all-row prefill BF16 vectors, full active sham fingerprints, scalar decode ledger and actual selected-payload read-only receipts. No decode-vector reconstruction or subjective-state claim.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "Scientific audit receipt is immutable")
    result = audit(args.run)
    write_once(args.out, result)
    print({key: result[key] for key in ("complete", "integrity_passed", "worker_policies_equal", "passed")})
