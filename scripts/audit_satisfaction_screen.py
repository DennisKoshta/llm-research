"""Audit the complete three-child satisfaction screen without granting a causal claim.

Task parsing, receipt rendering and token-timeline validators are shared with the
frozen producers; artifacts, complete assignment coverage, paired identity and
geometry are reconstructed here. Decode residual vectors are not saved: their
scalar ledgers are checked against the frozen arithmetic and engineering proof.
"""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tarfile

import numpy as np

from curiosity_study import satisfaction_capture as capture
from curiosity_study import satisfaction_screen as screen
from curiosity_study import resource_limits
from curiosity_study import sustained_engineering as engineering
from curiosity_study import voluntary_capture as receipts
from curiosity_study import voluntary_resolution as voluntary
from curiosity_study.artifacts import digest, read, write_once
from curiosity_study.concept_capture import MODEL
from curiosity_study.state_backend import PREFILL_PADDING_MULTIPLE

ARMS = [name for name, _ in screen.ARMS]
PANELS = [name for name, _ in screen.PANELS]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_path(root, value):
    path = Path(value)
    require(not path.is_absolute() and ".." not in path.parts, "Unsafe screen artifact path")
    target = (Path(root) / path).resolve()
    require(target.is_relative_to(Path(root).resolve()), "Screen artifact escaped its collection")
    return target


def artifact_path(root, binding):
    path = safe_path(root, binding["path"])
    require(
        sha(path) == binding["sha256"] and path.stat().st_size == binding["byte_count"],
        "Screen artifact bytes changed",
    )
    return path


def token_model(tokenizer):
    def encode(messages, enable_thinking=False):
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        return rendered, SimpleNamespace(input_ids=np.asarray([ids], dtype=np.int64))

    return SimpleNamespace(tokenizer=tokenizer, encode=encode)


def compact(assignment, record):
    row = {
        key: deepcopy(assignment[key])
        for key in (
            "trial_id",
            "base_trial_id",
            "panel_id",
            "panel_seed",
            "semantic_world_id",
            "world_slot",
            "world_id",
            "base_row",
            "arm",
            "family",
            "kind",
            "delivery",
            "menu_layout",
            "sampling_seed",
            "source_assignment_sha256",
        )
    }
    row["assignment_sha256"] = digest(assignment)
    row["record_sha256"] = None if record is None else digest(record)
    inner = None if record is None else record.get("task_record")
    row.update(
        status="missing" if record is None else record["status"],
        source_task_record_sha256=None if inner is None else digest(inner),
        valid=None if inner is None else inner["valid"],
        correct=None if inner is None else inner["correct"],
        thinking_closed=None if inner is None else inner["thinking_closed"],
        closed=None if inner is None else inner["thinking_closed"],
        cap_hit=None if inner is None else inner["cap_hit"],
        selected_service=None if inner is None else inner.get("selected_service"),
        failure=None if record is None else record.get("failure"),
        generated_tokens=None if inner is None else len(inner["inference"]["generated_token_ids"]),
    )
    return row


def readiness(outcomes):
    """Preserve all 1152 assigned denominators, independent of behavior direction."""
    require(
        len(outcomes) == len({row["trial_id"] for row in outcomes}) == 1152,
        "Readiness requires every unique assigned parent row",
    )
    diagnostics, choices = [], []
    for arm in ARMS:
        for panel in PANELS:
            for kind in voluntary.DIAGNOSTICS:
                rows = [
                    r for r in outcomes if r["arm"] == arm and r["panel_id"] == panel and r["kind"] == kind
                ]
                counts = counts_for(rows)
                diagnostics.append(
                    dict(
                        arm=arm,
                        panel_id=panel,
                        kind=kind,
                        **counts,
                        passed=counts["assigned"] == counts["valid"] == 12
                        and counts["correct"] >= 11
                        and counts["cap_hits"] == counts["missing"] == counts["execution_failures"] == 0,
                    )
                )
            for delivery in voluntary.DELIVERIES:
                rows = [
                    r
                    for r in outcomes
                    if r["arm"] == arm
                    and r["panel_id"] == panel
                    and r["kind"] == "choice"
                    and r["delivery"] == delivery
                ]
                counts = counts_for(rows)
                choices.append(
                    dict(
                        arm=arm,
                        panel_id=panel,
                        delivery=delivery,
                        **counts,
                        passed=counts["assigned"] == 24
                        and counts["valid"] >= 23
                        and counts["cap_hits"] == counts["missing"] == counts["execution_failures"] == 0,
                    )
                )
    return dict(
        assigned=1152,
        choice_correctness=None,
        diagnostic_cells=diagnostics,
        choice_cells=choices,
        passed=all(row["passed"] for row in diagnostics + choices),
        interpretation="Prespecified descriptive competence gates; not noninferiority or unchanged internal confidence.",
    )


def counts_for(rows):
    return dict(
        assigned=len(rows),
        valid=sum(r["valid"] is True for r in rows),
        correct=sum(r["correct"] is True for r in rows),
        cap_hits=sum(r["cap_hit"] is True for r in rows),
        missing=sum(r["status"] == "missing" for r in rows),
        execution_failures=sum(r["status"] == "execution_error" for r in rows),
    )


def hidden_arrays(root, artifact, ids, requests, cache):
    key = artifact["path"]
    if key not in cache:
        path = artifact_path(root, artifact)
        with np.load(path, allow_pickle=False) as data:
            cache[key] = (deepcopy(artifact), {name: data[name] for name in data.files})
    binding, arrays = cache[key]
    require(binding == artifact, "One capture file has conflicting bindings")
    hidden = arrays["hidden"]
    require(
        hidden.dtype == np.float32
        and hidden.ndim == 4
        and hidden.shape[:3] == (len(ids), 1, 2)
        and hidden.shape[-1] > 0
        and arrays["available"].dtype == np.bool_
        and arrays["available"].shape == (len(ids), 1, 2)
        and np.array_equal(arrays["available"], np.isfinite(hidden).all(-1))
        and arrays["layers"].tolist() == [15]
        and arrays["positions"].tolist() == ["record", "final"]
        and arrays["trial_ids"].tolist() == ids
        and arrays["token_prefix_sha256"].tolist() == [r["token_prefix_sha256"] for r in requests]
        and arrays["token_positions"].tolist()
        == [[r["prefix_positions"][p] for p in ("record", "final")] for r in requests],
        "Capture shape, row or token binding differs",
    )
    return arrays


def geometry(trace, before, direction, dose):
    """Prefill BF16 arithmetic is rederived; decode evidence is scalar consistency."""
    import torch

    events = trace["events"]
    if not events:
        return dose is None and trace.get("cumulative_achieved_delta_norm") == 0
    if dose is None:
        return False
    vector = torch.as_tensor(np.asarray(direction, dtype=np.float32))
    unit_norm = float(vector.norm())
    unit = vector / unit_norm
    for event in events:
        fields = (
            "before_norm",
            "after_norm",
            "delta_norm",
            "requested_delta_norm",
            "rounding_error_norm",
            "relative_delta_norm",
            "direction_norm",
            "projection_before",
            "projection_after",
        )
        if not all(type(event.get(k)) in (int, float) and math.isfinite(event[k]) for k in fields):
            return False
        if any(event[k] < 0 for k in fields[:7]):
            return False
        if event["activation_dtype"] != "torch.bfloat16" or event["requested_scalar_delta"] != dose:
            return False
        if event.get("phase") == "decode" and event.get("arithmetic_scalar_float32") != float(
            np.float32(dose)
        ):
            return False
        tolerance = 2e-5 * max(event["before_norm"], event["after_norm"], 1) + 1e-6
        if not math.isclose(event["direction_norm"], unit_norm, rel_tol=2e-5, abs_tol=1e-6):
            return False
        if not math.isclose(
            event["relative_delta_norm"],
            event["delta_norm"] / max(event["before_norm"], 1e-30),
            rel_tol=2e-5,
            abs_tol=1e-6,
        ):
            return False
        if abs(event["after_norm"] - event["before_norm"]) > event["delta_norm"] + tolerance:
            return False
        if (
            abs(event["projection_before"]) > event["before_norm"] + tolerance
            or abs(event["projection_after"]) > event["after_norm"] + tolerance
        ):
            return False
        if (
            abs(event["delta_norm"] - event["requested_delta_norm"])
            > event["rounding_error_norm"] + tolerance
        ):
            return False
        if (
            abs(
                (event["projection_after"] - event["projection_before"]) - float(np.float32(dose)) * unit_norm
            )
            > event["rounding_error_norm"] + tolerance
        ):
            return False
        if dose == 0 and any(
            event[k] != 0 for k in ("delta_norm", "requested_delta_norm", "rounding_error_norm")
        ):
            return False
    a = torch.as_tensor(np.asarray(before, dtype=np.float32))
    requested = a + float(dose) * vector
    actual = requested.to(torch.bfloat16).float()
    expected = dict(
        before_norm=float(a.norm()),
        after_norm=float(actual.norm()),
        delta_norm=float((actual - a).norm()),
        requested_delta_norm=float((requested - a).norm()),
        rounding_error_norm=float((actual - requested).norm()),
        relative_delta_norm=float((actual - a).norm()) / max(float(a.norm()), 1e-30),
        direction_norm=unit_norm,
        projection_before=float(torch.dot(a, unit)),
        projection_after=float(torch.dot(actual, unit)),
    )
    return all(
        math.isclose(events[0][key], value, rel_tol=2e-5, abs_tol=1e-6) for key, value in expected.items()
    ) and math.isclose(
        trace["cumulative_achieved_delta_norm"],
        math.fsum(event["delta_norm"] for event in events),
        rel_tol=2e-5,
        abs_tol=1e-6,
    )


def engaged(trace, dose):
    if dose in (None, 0):
        return True
    events = trace["events"]

    def positive(event):
        return event["delta_norm"] > 0 and event["projection_after"] > event["projection_before"]

    return bool(events and all(positive(event) for event in events))


def validate_generation(
    evidence, assignments, requests, tokenizer, eos_ids, vocabulary_size, arrays, direction
):
    """Return measured engineering gates; score mismatches are binding errors."""
    condition = evidence["condition"]
    require(
        evidence["eos_token_ids"] == sorted(eos_ids) and evidence["pad_token_id"] == tokenizer.pad_token_id,
        "EOS or padding identity changed",
    )
    patches = capture.patches_for(assignments, np.asarray(direction, dtype=np.float32))
    timeline = engineering.validate_condition(
        condition,
        requests,
        patches,
        cap=4096,
        pad_token_id=tokenizer.pad_token_id,
        eos_ids=eos_ids,
    )
    captures = arrays is not None and bool(np.isfinite(arrays["hidden"]).all())
    geometry_passed, engagement = captures, captures
    if condition["rows"] is None:
        return dict(
            timeline=False,
            captures=captures,
            hook_cleanup=condition["hooks_clean"] is True,
            geometry=False,
            engagement=False,
        )
    require(len(condition["rows"]) == 32, "Wrong generation row count")
    require(
        all(frame["vocabulary_size"] == vocabulary_size for frame in condition["forwards"]),
        "Full-vocabulary fingerprint width changed",
    )
    for index, (row, assignment) in enumerate(zip(condition["rows"], assignments, strict=True)):
        completion = row["completion"]
        tokens = completion["generated_token_ids"]
        require(
            all(type(token) is int and 0 <= token < vocabulary_size for token in tokens)
            and completion["raw"] == tokenizer.decode(tokens, skip_special_tokens=True),
            "Generated raw answer differs from actual token IDs",
        )
        dose = None if assignment["intervention"] is None else assignment["intervention"]["dose"]
        if captures:
            geometry_passed &= geometry(
                row["intervention_trace"], arrays["hidden"][index, 0, 0], direction, dose
            )
            engagement &= engaged(row["intervention_trace"], dose)
    return dict(
        timeline=timeline,
        captures=captures,
        hook_cleanup=condition["hooks_clean"] is True,
        geometry=bool(geometry_passed),
        engagement=bool(engagement),
    )


def generation_records(root, manifest):
    assignments = capture.validate_manifest(manifest)
    require(not any((root / "attempts").rglob("*.json")), "Screen generations require atomic cohorts")
    records, offsets, bindings, attempts = {}, set(), [], set()
    for path in sorted((root / "cohorts").rglob("*.json")):
        require(
            len(path.relative_to(root / "cohorts").parts) == 2 and path.stem.isdigit(),
            "Unexpected generation cohort path",
        )
        offset = int(path.stem)
        require(
            offset in range(0, len(assignments), 32) and offset not in offsets,
            "Duplicate or unassigned generation cohort",
        )
        envelope = read(path)
        rows = envelope["records"]
        require(
            envelope["manifest_hash"] == digest(manifest)
            and envelope["sha256"] == digest(rows)
            and len(rows) == 32
            and [r["assignment"] for r in rows] == assignments[offset : offset + 32]
            and all(r["terminal"] is True and r["trial_id"] == r["assignment"]["trial_id"] for r in rows),
            "Generation cohort membership or bytes changed",
        )
        offsets.add(offset)
        attempts.add(path.parent.name)
        records.update({row["trial_id"]: row for row in rows})
        bindings.append(
            dict(path=str(path.relative_to(root)), byte_sha256=sha(path), envelope_sha256=digest(envelope))
        )
    require(len(attempts) <= 1, "Screen child cannot combine multiple execution attempts")
    return records, offsets, bindings


def audit_generations(root, manifest, tokenizer, eos_ids, vocabulary_size):
    records, offsets, cohort_bindings = generation_records(root, manifest)
    assignments = manifest["diagnostics"]
    model = token_model(tokenizer)
    require(
        capture.bind_preflight(manifest["screen_plan"], model) == manifest["generation_preflight"],
        "Actual tokenizer differs from the bound preflight",
    )
    direction = manifest["screen_plan"]["direction_binding"]["unit"]
    caches, baselines, evidence_bindings, record_bindings, checks, prepatch = {}, {}, [], [], [], {}
    for offset in sorted(offsets):
        cohort = assignments[offset : offset + 32]
        ids = [a["trial_id"] for a in cohort]
        rows = [records[tid] for tid in ids]
        artifact = rows[0]["generation_evidence"]
        require(
            all(row["generation_evidence"] == artifact for row in rows),
            "One generation cohort binds different evidence",
        )
        evidence = read(artifact_path(root, artifact))
        requests = [manifest["generation_preflight"]["requests"][a["base_row"]]["request"] for a in cohort]
        require(
            evidence["manifest_sha256"] == digest(manifest)
            and evidence["generation_offset"] == offset
            and evidence["trial_ids"] == ids
            and evidence["requests"] == requests
            and evidence["capture_plan"] == capture.CAPTURE_PLAN,
            "Generation evidence is rebound to different rows or inputs",
        )
        capture_artifact = evidence["activation_artifact"]
        arrays = (
            None if capture_artifact is None else hidden_arrays(root, capture_artifact, ids, requests, caches)
        )
        require(
            arrays is None or arrays["hidden"].shape[-1] == len(direction),
            "Residual width differs from the fixed direction",
        )
        measured = validate_generation(
            evidence, cohort, requests, tokenizer, eos_ids, vocabulary_size, arrays, direction
        )
        saved_checks = {key: measured[key] for key in ("timeline", "captures", "hook_cleanup")}
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
                and arrays is None,
                "Unavailable generation fabricates observed evidence",
            )
        else:
            require(
                condition["capture_phase"] == capture.CAPTURE_PLAN["source_phase"]
                and condition["unpadded_token_positions"] == [r["prefix_positions"] for r in requests]
                and condition["padded_token_positions"]
                == [
                    [width - len(r["token_prefix"]) + r["prefix_positions"][p] for p in ("record", "final")]
                    for r in requests
                ]
                and condition["prefill_width"] == width,
                "Source capture timing or padded sites changed",
            )
        if cohort[0]["arm"] == "baseline":
            baselines[cohort[0]["base_row"]] = (evidence, None if arrays is None else arrays["hidden"])
            require(evidence["sham_comparison"] is None, "Baseline claims a sham comparison")
        elif cohort[0]["arm"] == "zero_sham":
            reference = baselines.pop(cohort[0]["base_row"], None)
            if reference is None:
                measured["sham_identity"] = False
                require(evidence["sham_comparison"] is None, "Sham binds a missing baseline")
                saved_checks["sham_identity"] = False
            else:
                baseline, baseline_hidden = reference
                identical = (
                    condition["rows"] is not None
                    and baseline["condition"]["rows"] is not None
                    and engineering.identical_rows(baseline["condition"], condition, range(32))
                )
                expected = dict(
                    baseline_evidence_sha256=digest(baseline),
                    same_requests=baseline["requests"] == requests,
                    full_active_logits_and_tokens_equal=bool(identical),
                )
                require(
                    evidence["sham_comparison"] == expected,
                    "Saved sham comparison differs from observed predictions",
                )
                saved_checks["sham_identity"] = expected["same_requests"] and bool(identical)
                measured["sham_identity"] = (
                    saved_checks["sham_identity"]
                    and arrays is not None
                    and baseline_hidden is not None
                    and np.array_equal(baseline_hidden, arrays["hidden"])
                )
        else:
            require(evidence["sham_comparison"] is None, "Positive arm claims a baseline identity comparison")
        require(
            all(type(value) is bool for value in evidence["checks"].values())
            and evidence["checks"] == saved_checks
            and evidence["engineering_passed"] is all(saved_checks.values()),
            "Saved generation engineering checks differ from their observations",
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
            require(
                digest(row["task_record"]) == digest(expected_inner)
                and row["cohort_row"] == index
                and row["activation_artifact"] == capture_artifact
                and row["engineering_passed"] is evidence["engineering_passed"]
                and row["status"] == ("execution_error" if expected_inner is None else "completed"),
                "Saved task parsing/scoring, evidence row or disposition changed",
            )
            expected_failure = (
                "generation_execution_error"
                if observed is None
                else "generation_engineering_guard_failed"
                if not evidence["engineering_passed"]
                else None
            )
            require(row.get("failure") == expected_failure, "Generation failure was hidden or reclassified")
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
            prepatch[assignment["trial_id"]] = None if arrays is None else arrays["hidden"][index].copy()
        checks.append(dict(offset=offset, arm=cohort[0]["arm"], **measured))
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


def receipt_measurement(measurement, request, tokenizer, width):
    ids, positions = request["token_prefix"], request["prefix_positions"]
    pad = width - len(ids)
    return bool(
        measurement["unpadded_input_ids"] == ids
        and measurement["token_prefix_sha256"] == digest(ids)
        and measurement["input_ids"] == [tokenizer.pad_token_id] * pad + ids
        and measurement["attention_mask"] == [0] * pad + [1] * len(ids)
        and measurement["position_ids"] == [0] * pad + list(range(len(ids)))
        and measurement["prefill_width"] == width
        and measurement["prefill_padding_multiple"] == PREFILL_PADDING_MULTIPLE
        and measurement["unpadded_token_positions"] == positions
        and measurement["record_token_index"] == pad + positions["record"]
        and measurement["exact_token_replay"] is True
        and measurement["rendered_prompt"] == tokenizer.decode(ids, skip_special_tokens=False)
        and measurement["prompt_sha256"] == digest(measurement["rendered_prompt"])
        and measurement["hooks_remaining"] == 0
        and measurement["intervention"] is None
        and "completion" not in measurement
    )


def audit_receipts(root, manifest, sources, tokenizer):
    choices = [a for a in manifest["diagnostics"] if a["kind"] == "choice"]
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
        offset = envelope["choice_offset"]
        require(path.name == f"{offset:07d}.json", "Receipt filename differs from its logical offset")
        require(
            type(offset) is int and offset in range(0, len(choices), 32) and offset not in offsets,
            "Duplicate or unassigned receipt cohort",
        )
        ids = [a["trial_id"] for a in choices[offset : offset + 32]]
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
            base = capture._choice_sidecar(source)
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
        arrays = None if artifact is None else hidden_arrays(root, artifact, active_ids, requests, cache)
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
                and guard["layers"] == [15]
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
                    == len(manifest["screen_plan"]["direction_binding"]["unit"]),
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
                    choice_offset=offset,
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
        gates.append(bool(passed and all(row["status"] != "execution_error" for row in rows)))
        all_rows.extend(rows)
        bindings.append(
            dict(path=str(path.relative_to(root)), byte_sha256=sha(path), envelope_sha256=digest(envelope))
        )
    return dict(
        complete=offsets == set(range(0, len(choices), 32)),
        passed=len(gates) == len(range(0, len(choices), 32)) and all(gates),
        slots=len(all_rows),
        status_counts=dict(Counter(row["status"] for row in all_rows)),
        cohorts=bindings,
        record_bindings=[
            dict(source_trial_id=row["source_trial_id"], sidecar_sha256=digest(row)) for row in all_rows
        ],
        capture_artifacts=[binding for binding, _ in cache.values()],
    )


def cross_arm_identity(assignments, vectors):
    """Compare both unpatched sites by exact bytes, retaining missing base rows."""
    groups = {}
    for assignment in assignments:
        groups.setdefault(assignment["base_trial_id"], {})[assignment["arm"]] = vectors.get(
            assignment["trial_id"]
        )
    rows = []
    for base_id, arms in sorted(groups.items()):
        available = set(arms) == set(ARMS) and all(
            value is not None and np.isfinite(value).all() for value in arms.values()
        )
        identical = available and all(
            value.dtype == arms["baseline"].dtype
            and value.shape == arms["baseline"].shape
            and value.tobytes() == arms["baseline"].tobytes()
            for value in arms.values()
        )
        rows.append(
            dict(
                base_trial_id=base_id,
                available=bool(available),
                identical=bool(identical),
                arm_array_sha256={
                    arm: None if value is None else receipts.array_sha(value) for arm, value in arms.items()
                },
            )
        )
    return dict(
        passed=len(rows) == 288 and all(row["identical"] for row in rows),
        assigned_base_rows=288,
        observed_base_rows=len(rows),
        sites=["record", "final"],
        rows=rows,
    )


def parent_children(root, manifest):
    from analyze_satisfaction_screen import ANALYSIS_METADATA

    require(
        "execution_group" not in manifest and "parent_manifest_sha256" not in manifest,
        "Child-only results cannot be scientifically audited",
    )
    require(
        digest(manifest.get("budget")) == digest(resource_limits.SCREEN_BUDGET)
        and resource_limits.job_wall_limit(manifest)
        == resource_limits.SCREEN_BUDGET["shared_deadline_seconds"],
        "Parent budget differs from the fixed screen resource policy",
    )
    plan = manifest["screen_plan"]
    require(
        manifest["assay"] == "satisfaction_screen"
        and manifest["task_revision"] == screen.REVISION
        and manifest["phase"] == "development_pilot"
        and manifest["inference_scope"] == "development"
        and manifest["model"] == MODEL
        and manifest["execution_gpu"] == "H200"
        and manifest["execution"] == screen.EXECUTION
        and manifest["max_new_tokens"] == 4096
        and manifest["capture_plan"] == capture.CAPTURE_PLAN
        and manifest["schedule"] == screen.SCHEDULE
        and manifest["arithmetic"] == engineering.ARITHMETIC
        and manifest["engine_sha256"] == engineering.engine_hashes()
        and manifest["analysis_metadata"] == ANALYSIS_METADATA
        and manifest["assignments"] == []
        and manifest["diagnostics"] == plan["assignments"]
        and len(manifest["diagnostics"]) == 1152
        and manifest["assignment_sha256"] == digest(manifest["diagnostics"])
        and manifest["screen_plan_sha256"] == digest(plan),
        "Parent design or engine binding differs",
    )
    descriptors = read(root / "children.json")
    require(
        len(descriptors) == len(capture.GROUPS)
        and [row["execution_group"] for row in descriptors] == list(capture.GROUPS),
        "Require all execution groups in assigned order",
    )
    children = []
    for descriptor, (group, arms) in zip(descriptors, capture.GROUPS.items(), strict=True):
        path = Path(descriptor["path"]).resolve()
        require(
            path.parent == root.resolve().parent and path != root.resolve(),
            "Child collection is outside the parent run directory",
        )
        child = read(path / "manifest.json")
        expected = deepcopy(manifest)
        expected.update(
            execution_group=group,
            parent_manifest_sha256=digest(manifest),
            diagnostics=[a for a in manifest["diagnostics"] if a["arm"] in arms],
        )
        expected["assignment_sha256"] = digest(expected["diagnostics"])
        require(
            child == expected and digest(child) == descriptor["manifest_sha256"],
            "Child manifest differs from the exact parent decomposition",
        )
        capture.validate_manifest(child)
        children.append((path, child))
    return children


def frozen_documents(root, manifest):
    frozen = read(root / "freeze.json")
    require(
        frozen["manifest_sha256"] == digest(manifest)
        and frozen["source_sha256"] == manifest["source_sha256"]
        and frozen["archive_sha256"] == sha(root / "source.tar.gz"),
        "Parent source freeze changed",
    )
    required = {
        "prospective_plan",
        "history",
        "exposure",
        "generation_preflight",
        "receipt_preflight",
        "engineering_manifest",
        "engineering_audit",
        "engineering_freeze",
        "engineering_archive",
        "candidate_manifest",
        "candidate_audit",
        "candidate_analysis",
        "candidate_directions",
        "candidate_independent_check",
    }
    bindings = manifest["bound_receipts"]
    require(
        len(bindings) == len(required)
        and {row["role"] for row in bindings} == required
        and len({row["path"] for row in bindings}) == len(bindings),
        "Missing or duplicate prospective evidence roles",
    )
    values, payloads = {}, {}
    with tarfile.open(root / "source.tar.gz", "r:gz") as archive:
        names = [m.name for m in archive.getmembers() if m.isfile()]
        require(len(names) == len(set(names)), "Duplicate archived source paths")
        for binding in bindings:
            relative = Path(binding["path"])
            require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe frozen receipt path")
            payload = archive.extractfile(str(relative)).read()
            require(
                hashlib.sha256(payload).hexdigest()
                == binding["byte_sha256"]
                == frozen["documents"].get(str(relative)),
                "Frozen receipt bytes differ",
            )
            value = json.loads(payload) if relative.suffix == ".json" else None
            require(
                binding["content_sha256"] == (digest(value) if value is not None else None),
                "Frozen JSON content digest differs",
            )
            values[binding["role"]], payloads[binding["role"]] = value, payload
        source = {
            name.removeprefix("src/curiosity_study/"): archive.extractfile(name).read().decode()
            for name in names
            if name.startswith("src/curiosity_study/") and name.endswith(".py")
        }
        require(digest(source) == manifest["source_sha256"], "Frozen source digest differs")
        for module in (capture, screen, engineering, receipts, voluntary, resource_limits):
            relative = f"src/curiosity_study/{Path(module.__file__).name}"
            require(
                hashlib.sha256(archive.extractfile(relative).read()).hexdigest() == sha(module.__file__),
                "Shared audit validator differs from its frozen implementation",
            )
        for name, expected in manifest["engine_sha256"].items():
            require(
                hashlib.sha256(archive.extractfile(f"src/curiosity_study/{name}").read()).hexdigest()
                == expected,
                "Frozen numerical engine differs",
            )
        history = values["history"]
        require(
            hashlib.sha256(archive.extractfile("scripts/audit_voluntary_freshness.py").read()).hexdigest()
            == history["audit_script_byte_sha256"],
            "History adapter differs from its frozen implementation",
        )
        for name, expected in history["source_file_sha256"].items():
            require(
                not Path(name).is_absolute()
                and ".." not in Path(name).parts
                and hashlib.sha256(archive.extractfile(name).read()).hexdigest() == expected,
                "Historical world canonicalization source changed",
            )
    return frozen, values, payloads


def provenance(root, manifest, tokenizer):
    from audit_sustained_engineering import audit as audit_engineering
    from audit_voluntary_freshness import history_receipt
    from audit_voluntary_resolution import recheck_history
    from prepare_voluntary_resolution import candidate_binding

    # Prospective tokenization and admission helpers are shared, never fitting data.
    from prepare_satisfaction_screen import (
        LOCKED_CANDIDATE_MANIFEST,
        LOCKED_DIRECTION_BYTES,
        LOCKED_NORM,
        exposure_target_plan,
        receipt_preflight,
    )

    frozen, documents, payloads = frozen_documents(root, manifest)
    roles = {b["role"]: b for b in manifest["bound_receipts"]}
    history, exposure, plan = documents["history"], documents["exposure"], manifest["screen_plan"]
    require(
        history_receipt(exposure) == history
        and history["selection_independent"] is True
        and history["historical_exact_world_ids_sha256"] == digest(history["historical_exact_world_ids"])
        and plan
        == screen.assignment_plan(
            historical_exact_world_ids=history["historical_exact_world_ids"],
            direction_binding=plan["direction_binding"],
        ),
        "Selection does not derive exactly from bound historical exposure",
    )
    target = exposure_target_plan(plan)
    require(
        exposure["task_revision"] == screen.REVISION
        and exposure["target_plan_sha256"] == digest(target)
        and exposure["selection_sha256"] == target["selection_sha256"]
        and exposure["assignment_sha256"] == target["assignment_sha256"]
        and exposure["exact_complete_world_criterion_passed"] is True
        and exposure["exact_complete_world_overlap_count"] == 0
        and len(exposure["selected_worlds"]) == 24,
        "Exposure receipt binds different worlds or admits historical duplicates",
    )
    repository = Path(frozen["budget_root"])
    history_check = recheck_history(repository, history, target)
    candidate = candidate_binding(
        documents["candidate_manifest"],
        documents["candidate_audit"],
        documents["candidate_analysis"],
        documents["candidate_independent_check"],
        roles["candidate_analysis"]["byte_sha256"],
        roles["candidate_directions"]["byte_sha256"],
    )
    direction = plan["direction_binding"]
    require(
        candidate == manifest["candidate_source"] == direction["candidate_source"]
        and digest(documents["candidate_manifest"]) == LOCKED_CANDIDATE_MANIFEST
        and roles["candidate_directions"]["byte_sha256"] == LOCKED_DIRECTION_BYTES
        and direction["directions_byte_sha256"] == LOCKED_DIRECTION_BYTES
        and direction["norm"] == LOCKED_NORM,
        "Locked candidate provenance changed",
    )
    with np.load(BytesIO(payloads["candidate_directions"]), allow_pickle=False) as arrays:
        require(
            arrays["layers"].tolist() == [15, 31, 47, 57]
            and arrays["positions"].tolist() == ["record", "final"],
            "Candidate sites changed",
        )
        index = arrays["sources"].tolist().index("satisfaction_vs_neutral")
        require(
            arrays["unit"][index, 0, 0].tolist() == direction["unit"]
            and float(arrays["contrast_norm"][index, 0, 0]) == direction["norm"]
            and len(direction["unit"]) == 5120,
            "Screen direction differs from frozen A15/record axis",
        )
    em, ef, ea = (
        documents[name] for name in ("engineering_manifest", "engineering_freeze", "engineering_audit")
    )
    engineering_run = repository / Path(roles["engineering_manifest"]["path"]).parent
    require(
        read(engineering_run / "manifest.json") == em
        and sha(engineering_run / "source.tar.gz") == roles["engineering_archive"]["byte_sha256"]
        and read(engineering_run / "freeze.json") == ef
        and audit_engineering(engineering_run) == ea,
        "Full-cap engineering evidence no longer reproduces",
    )
    require(
        ea["complete"] is True
        and ea["passed"] is True
        and ea["integrity_passed"] is True
        and ea["cap"] == em["cap"] == 4096
        and em["model"] == MODEL
        and em["execution"] == screen.EXECUTION
        and em["schedule"] == screen.SCHEDULE
        and em["arithmetic"] == engineering.ARITHMETIC
        and em["engine_sha256"] == manifest["engine_sha256"]
        and ef["manifest_sha256"] == digest(em)
        and ef["archive_sha256"] == roles["engineering_archive"]["byte_sha256"],
        "Full-cap engineering proof does not certify the pinned execution policy",
    )
    certificate = manifest["engineering_certificate"]
    require(
        all(
            certificate[key] == expected
            for key, expected in dict(
                manifest_sha256=digest(em),
                audit_sha256=digest(ea),
                freeze_sha256=digest(ef),
                archive_sha256=ef["archive_sha256"],
            ).items()
        ),
        "Engineering certificate hash differs",
    )
    model = token_model(tokenizer)
    for role, actual in (
        ("generation_preflight", capture.bind_preflight(plan, model)),
        ("receipt_preflight", receipt_preflight(plan, model)),
    ):
        require(
            documents[role] == manifest[role] == actual and manifest[role + "_sha256"] == digest(actual),
            "Prospective tokenizer boundary check no longer reproduces",
        )
    return dict(
        freeze=frozen,
        history=history_check,
        candidate_source=candidate,
        engineering_certificate=certificate,
        engineering_environment=engineering_environment(engineering_run, ea),
        bound_receipts=manifest["bound_receipts"],
    )


def runtime_policy(value):
    require(
        all(value.get(key) == val for key, val in MODEL.items())
        and value["dtype"] == "bfloat16"
        and value["attention"] == "eager"
        and value["device"] == "cuda"
        and "H200" in value["gpu"]
        and value["cuda_numerics"]["allow_bf16_reduced_precision_reduction"] is False,
        "Actual worker model, arithmetic or GPU differs",
    )
    keys = (
        "backend",
        "model_id",
        "revision",
        "dtype",
        "attention",
        "cuda_numerics",
        "loading_policy",
        "python",
        "device",
        "gpu",
        "cuda",
        "versions",
        "template_sha256",
        "layers",
    )
    return {key: value[key] for key in keys}


def engineering_environment(root, receipt):
    report = read(artifact_path(root, receipt["report_artifact"]))
    attempt = Path(receipt["report_artifact"]["path"]).parent.name
    path = root / "environments" / f"{attempt}.json"
    value = read(path)
    require(
        digest(value) == receipt["environment_sha256"] == report["environment_sha256"],
        "Full-cap environment differs from the engineering audit/report binding",
    )
    return dict(
        path=str(path), byte_sha256=sha(path), content_sha256=digest(value), policy=runtime_policy(value)
    )


def environments(root, manifest, generation):
    attempts = {Path(b["path"]).parts[1] for b in generation["cohorts"]}
    rows = []
    for attempt in sorted(attempts):
        path = root / "environments" / f"{attempt}.json"
        value = read(path)
        rows.append(
            dict(
                path=str(path.relative_to(root)),
                byte_sha256=sha(path),
                content_sha256=digest(value),
                policy=runtime_policy(value),
            )
        )
    return rows


def runtime_compatible(study_policies, engineering_policy):
    return len(study_policies) == len(capture.GROUPS) and all(
        value == engineering_policy for value in study_policies
    )


def audit(root, tokenizer=None, eos_token_ids=None, vocabulary_size=None):
    root = Path(root).resolve()
    manifest = read(root / "manifest.json")
    children = parent_children(root, manifest)
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL["model_id"], revision=MODEL["revision"], trust_remote_code=False, local_files_only=True
        )
    if eos_token_ids is None or vocabulary_size is None:
        from transformers import AutoConfig, GenerationConfig

        if eos_token_ids is None:
            eos = GenerationConfig.from_pretrained(
                MODEL["model_id"], revision=MODEL["revision"], local_files_only=True
            ).eos_token_id
            eos_token_ids = set(eos if isinstance(eos, list) else [] if eos is None else [eos])
        if vocabulary_size is None:
            vocabulary_size = AutoConfig.from_pretrained(
                MODEL["model_id"], revision=MODEL["revision"], local_files_only=True
            ).vocab_size
    bound = provenance(root, manifest, tokenizer)
    records, vectors, child_checks, environment_policies = {}, {}, [], []
    for path, child in children:
        generated = audit_generations(path, child, tokenizer, set(eos_token_ids), vocabulary_size)
        received = audit_receipts(path, child, generated["records"], tokenizer)
        env = environments(path, child, generated)
        environment_policies.extend(row["policy"] for row in env)
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
    outcomes = [compact(a, records.get(a["trial_id"])) for a in manifest["diagnostics"]]
    identity = cross_arm_identity(manifest["diagnostics"], vectors)
    complete = all(c["generation"]["complete"] and c["receipts"]["complete"] for c in child_checks)
    checks = [row for child in child_checks for row in child["generation"]["checks"]]
    sham = [row for row in checks if row["arm"] == "zero_sham"]
    sham_passed = len(sham) == 9 and all(row["sham_identity"] for row in sham)
    engagement = len(checks) == 36 and all(row["engagement"] and row["geometry"] for row in checks)
    receipt_passed = all(child["receipts"]["passed"] for child in child_checks)
    policies_equal = runtime_compatible(environment_policies, bound["engineering_environment"]["policy"])
    integrity = bool(
        complete
        and identity["passed"]
        and policies_equal
        and len(checks) == 36
        and all(
            row["timeline"] and row["captures"] and row["hook_cleanup"] and row["geometry"] for row in checks
        )
    )
    ready = readiness(outcomes)
    return dict(
        schema_version=1,
        assay="satisfaction_screen_audit",
        parent_manifest_sha256=digest(manifest),
        source_sha256=manifest["source_sha256"],
        auditor_byte_sha256=sha(__file__),
        assigned=1152,
        assigned_choices=576,
        terminal_generations=len(records),
        terminal_receipt_slots=sum(child["receipts"]["slots"] for child in child_checks),
        complete=bool(complete),
        integrity_passed=integrity,
        sham_identity_passed=bool(sham_passed),
        engagement_passed=bool(engagement),
        receipts_passed=bool(receipt_passed),
        cross_arm_prepatch_identity_passed=identity["passed"],
        cross_arm_prepatch_identity=identity,
        worker_policies_equal=bool(policies_equal),
        readiness=ready,
        passed=bool(integrity and sham_passed and engagement and receipt_passed and ready["passed"]),
        provenance=bound,
        children=child_checks,
        outcomes=outcomes,
        outcomes_sha256=digest(outcomes),
        scientific_claim="Development optional-policy screen only; no subjective-state or confidence-preservation conclusion.",
        validation_scope=(
            "Exact archived inputs, source-task parsing, token timelines, full-vocabulary sham fingerprints, "
            "prefill BF16 geometry, and selected-payload receipts. Task/render/timeline and admission helpers "
            "are shared with frozen implementations. Decode geometry has scalar consistency checks, not "
            "independent residual-vector reconstruction. Full-cap engineering tested one C+ row; this S+ "
            "all-row screen additionally requires observed engagement and same-source prepatch identity. "
            "Arm groups use separate workers; matched software/GPU metadata and prepatch equality do not "
            "eliminate every possible hardware interaction. Receipt comparisons are selection-conditioned."
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "Audit receipts are immutable")
    result = audit(args.run)
    write_once(args.out, result)
    print(
        {
            key: result[key]
            for key in (
                "complete",
                "integrity_passed",
                "sham_identity_passed",
                "engagement_passed",
                "receipts_passed",
                "passed",
            )
        }
    )
