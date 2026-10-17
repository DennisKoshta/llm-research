"""Audit optional choices, immutable generation captures, and actual receipts.

Readiness retains all assigned denominators. Choice correctness is undefined;
no preferred action, candidate projection, or effect sign enters a gate.
"""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np

from curiosity_study import thinking_information as thinking
from curiosity_study import voluntary_capture as capture
from curiosity_study import voluntary_resolution as voluntary
from curiosity_study.artifacts import RunStore, digest, read, write_once
from curiosity_study.state_backend import PREFILL_PADDING_MULTIPLE
from collection_coverage import check_collection_coverage

REQUIRED_CRITERIA = dict(
    diagnostics=dict(
        group_by=["family", "kind"],
        assigned_each=12,
        minimum_correct=11,
        minimum_valid=12,
        maximum_cap_hits=0,
    ),
    choices=dict(group_by=["family", "delivery"], assigned_each=24, minimum_valid=23, maximum_cap_hits=0),
    complete_terminal_coverage=True,
    no_execution_failures=True,
)
REQUIRED_ANALYSIS_METADATA = dict(
    schema_version=1,
    bootstrap_samples=10000,
    bootstrap_seed=2026101434,
    primary_delivery_contrast=["decisive_evidence", "redundant_evidence"],
    secondary_delivery_contrast=["decisive_evidence", "independent_novelty"],
    primary_position="record",
    candidate_selection="locked_corpus_A_raw",
    candidate_layers=dict(curiosity_vs_indifference=15, satisfaction_vs_neutral=15),
)
V2_DOCUMENTS = dict(
    plan="reports/voluntary-resolution-v2-plan.md",
    history="reports/voluntary-resolution-v2-history.json",
    exposure="reports/voluntary-resolution-v2-freshness.json",
    preflight="reports/voluntary-resolution-v2-tokenization-preflight.json",
    materials="reports/voluntary-resolution-v2-material-comparison.json",
)
V1_RUN = "runs/qwen32-voluntary-resolution-v1"
V1_TASK_SHA = "eda9deec8a446c7af317ae3887c95c3d5af74ddd30b304d72646d4a4f0f8dc0e"
HISTORY_REPLAY_KEYS = (
    "historical_registry",
    "historical_registry_sha256",
    "source_archive_bindings",
    "historical_exact_world_ids",
    "historical_exact_world_ids_sha256",
    "exact_world_exclusions",
    "exact_world_exclusions_sha256",
    "historical_world_counts",
    "inclusion_rule",
    "exact_rule",
    "projection_rule",
    "independent_holdout_claim_supported",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def byte_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def history_plan(history):
    """Require a complete history-derived exclusion list before selecting v2 worlds."""
    ids = history.get("historical_exact_world_ids")
    require(
        history.get("schema_version") == 1
        and history.get("assay") == "voluntary-resolution-historical-exclusions"
        and history.get("selection_independent") is True
        and history.get("independent_holdout_claim_supported") is False
        and isinstance(ids, list)
        and ids == sorted(set(ids))
        and history.get("historical_exact_world_ids_sha256") == digest(ids),
        "Require the complete selection-independent historical exclusions",
    )
    registry, exclusions = history["historical_registry"], history["exact_world_exclusions"]
    require(
        history["historical_registry_sha256"] == digest(registry)
        and history["exact_world_exclusions_sha256"] == digest(exclusions)
        and [row["semantic_world_id"] for row in exclusions] == ids
        and len({row["run"] for row in registry}) == len(registry),
        "Historical registry or exclusion digest differs",
    )
    executed = {row["run"] for row in registry if row["execution_evidence"] != "none"}
    require(V1_RUN in executed, "The failed v1 collection must remain in historical exposure")
    require(
        all(
            digest(row["canonical_world"]) == row["semantic_world_id"]
            and row["source_runs"] == sorted(set(row["source_runs"]))
            and bool(row["source_runs"])
            and set(row["source_runs"]) <= executed
            for row in exclusions
        )
        and any(V1_RUN in row["source_runs"] for row in exclusions),
        "Exclusions require canonical worlds and executed provenance, including v1",
    )
    plan = voluntary.assignment_plan(voluntary.SEED, excluded_world_ids=ids)
    require(
        not {a["semantic_world_id"] for a in plan["assignments"]}.intersection(ids),
        "Selected worlds overlap historical exact exposure",
    )
    return plan


def exposure_binding(history, exposure, plan):
    from audit_voluntary_freshness import history_receipt

    require(history_receipt(exposure) == history, "Exposure does not use the complete bound history")
    require(
        exposure.get("task_revision") == voluntary.REVISION
        and exposure.get("target_seed") == voluntary.SEED
        and exposure.get("target_plan_sha256") == digest(plan)
        and exposure.get("selection_sha256") == plan["selection_sha256"]
        and exposure.get("assignment_sha256") == plan["assignment_sha256"]
        and exposure.get("exact_complete_world_criterion_passed") is True
        and exposure.get("exact_complete_world_overlap_count") == 0,
        "Prior exposure does not bind the exact fresh v2 assignment plan",
    )


def prospective_checks(plan, preflight, materials, task_sha):
    """Bind tokenizer-only coverage and the sole permitted prompt amendment."""
    for value in (preflight, materials):
        require(
            value.get("passed") is True
            and value.get("task_revision") == voluntary.REVISION
            and value.get("seed") == voluntary.SEED
            and value.get("assignment_sha256") == plan["assignment_sha256"]
            and value.get("selection_sha256") == plan["selection_sha256"]
            and value.get("task_module_sha256") == task_sha,
            "Prospective check identity differs from the selected task and assignments",
        )
    ids = [a["trial_id"] for a in plan["assignments"]]
    choice_ids = {a["trial_id"] for a in plan["assignments"] if a["kind"] == "choice"}
    sources, receipts = preflight["source_bindings"], preflight["receipt_bindings"]
    require(
        preflight.get("model") == capture.MODEL
        and preflight.get("rows") == 432
        and preflight.get("source_rows") == 288
        and preflight.get("stop_checks") == 144
        and preflight.get("synthetic_answer_only") is True
        and preflight.get("generated_model_outputs") is False
        and [row["trial_id"] for row in sources] == ids
        and len(receipts) == 432
        and {(row["trial_id"], row["selected_service"]) for row in receipts}
        == {(trial, service) for trial in choice_ids for service in voluntary.SERVICES if service != "stop"}
        and preflight["bindings_sha256"] == digest(dict(source=sources, receipts=receipts)),
        "Tokenizer preflight must cover every source and all possible non-stop receipts",
    )
    bindings = materials["bindings"]
    require(
        materials.get("frozen_v1_task_sha256") == V1_TASK_SHA
        and materials.get("counts")
        == dict(
            contexts_unchanged=288,
            optional_prompts_unchanged=144,
            other_diagnostic_prompts_unchanged=96,
            clarified_diagnostic_prompts=48,
        )
        and materials["bindings_sha256"] == digest(bindings)
        and [row["trial_id"] for row in bindings] == ids,
        "Material comparison must retain all 288 assignments and exactly 48 clarifications",
    )
    for a, row in zip(plan["assignments"], bindings, strict=True):
        prompt, _, expected = voluntary.prompt(a)
        require(
            row["assignment_sha256"] == digest(a)
            and row["kind"] == a["kind"]
            and row["context_sha256"] == digest(voluntary.context(a))
            and row["new_prompt_sha256"] == digest(prompt)
            and row["expected_answer"] == expected
            and (row["old_prompt_sha256"] != row["new_prompt_sha256"])
            == (a["kind"] in ("new_fact", "repeated_fact")),
            "Material comparison differs from an assigned prompt or permitted clarification",
        )


def recheck_history(repository, history, plan, *, prospective=False):
    """Reconstruct immutable inputs; never inspect behavioral scores or activations."""
    from audit_voluntary_freshness import audit as freshness_audit, history_receipt

    result = freshness_audit(
        repository,
        plan=plan,
        pool_candidates=0,
        **({} if prospective else {"registry_snapshot": history["historical_registry"]}),
    )
    if prospective:
        require(
            history_receipt(result) == history,
            "Historical exclusions differ from the complete current inventory",
        )
    else:
        require(
            all(result[key] == history[key] for key in HISTORY_REPLAY_KEYS),
            "Frozen historical exclusions cannot be independently reconstructed",
        )
    require(
        result["exact_complete_world_criterion_passed"] is True
        and result["exact_complete_world_overlap_count"] == 0,
        "Selected v2 worlds overlap bound historical exposure",
    )
    for binding in history["source_archive_bindings"].values():
        require(
            byte_sha(Path(repository) / binding["archive"]) == binding["archive_byte_sha256"],
            "A bound historical source archive changed",
        )
    return dict(
        history_sha256=digest(history),
        exclusions_sha256=history["historical_exact_world_ids_sha256"],
        excluded_worlds=len(history["historical_exact_world_ids"]),
        exact_complete_world_overlap_count=0,
        independently_reconstructed=True,
        replay_uses_frozen_registry=not prospective,
    )


def readiness(assignments, records, sidecars):
    """Count assigned worlds/cells without conditioning on successful output."""
    diagnostic, choices = [], []
    for family in voluntary.FAMILIES:
        for kind in voluntary.DIAGNOSTICS:
            rows = [a for a in assignments if a["family"] == family and a["kind"] == kind]
            counts = cell_counts(rows, records)
            passed = (
                counts["assigned"] == 12
                and counts["correct"] >= 11
                and counts["valid"] == 12
                and counts["cap_hits"] == 0
                and counts["missing"] == counts["execution_failures"] == 0
            )
            diagnostic.append(dict(family=family, kind=kind, **counts, passed=passed))
        for delivery in voluntary.DELIVERIES:
            rows = [
                a
                for a in assignments
                if a["family"] == family and a["kind"] == "choice" and a["delivery"] == delivery
            ]
            counts = cell_counts(rows, records)
            counts.pop("correct")
            passed = (
                counts["assigned"] == 24
                and counts["valid"] >= 23
                and counts["cap_hits"] == 0
                and counts["missing"] == counts["execution_failures"] == 0
            )
            choices.append(dict(family=family, delivery=delivery, **counts, passed=passed))
    expected = {a["trial_id"] for a in assignments}
    expected_choices = {a["trial_id"] for a in assignments if a["kind"] == "choice"}
    source_complete = len(assignments) == len(expected) == 288 and set(records) == expected
    receipt_complete = len(expected_choices) == 144 and set(sidecars) == expected_choices
    required = sum(row.get("receipt_required") is True for row in sidecars.values())
    completed = sum(row.get("receipt_model_pass_completed") is True for row in sidecars.values())
    failures = sum("failure" in row for row in records.values()) + sum(
        row.get("status") == "execution_error" for row in sidecars.values()
    )
    gates = dict(
        source_coverage=source_complete,
        receipt_slot_coverage=receipt_complete,
        required_receipts_complete=required == completed,
        no_execution_failures=failures == 0,
        diagnostics=all(row["passed"] for row in diagnostic),
        choices=all(row["passed"] for row in choices),
    )
    return dict(
        criteria=REQUIRED_CRITERIA,
        passed=all(gates.values()),
        checks=gates,
        diagnostic_cells=diagnostic,
        choice_cells=choices,
        assigned=288,
        assigned_choices=144,
        assigned_diagnostics=144,
        observed_generation_records=len(records),
        observed_receipt_slots=len(sidecars),
        required_receipts=required,
        completed_receipts=completed,
        execution_failures=failures,
        choice_correctness=None,
        causal_admission=False,
    )


def cell_counts(assignments, records):
    rows = [records[a["trial_id"]] for a in assignments if a["trial_id"] in records]
    return dict(
        assigned=len(assignments),
        observed=len(rows),
        missing=len(assignments) - len(rows),
        valid=sum(row.get("valid") is True for row in rows),
        correct=sum(row.get("correct") is True for row in rows),
        cap_hits=sum(row.get("cap_hit") is True for row in rows),
        execution_failures=sum("failure" in row for row in rows),
        unclosed=sum(row.get("thinking_closed") is False for row in rows),
    )


def archive_bindings(root, manifest):
    frozen = read(root / "freeze.json")
    require(
        frozen["manifest_sha256"] == digest(manifest)
        and frozen["source_sha256"] == manifest["source_sha256"]
        and frozen["archive_sha256"] == byte_sha(root / "source.tar.gz"),
        "Source freeze mismatch",
    )
    with tarfile.open(root / "source.tar.gz", "r:gz") as archive:
        documents = {}
        for binding in manifest["document_bindings"]:
            payload = archive.extractfile(binding["path"]).read()
            sha = hashlib.sha256(payload).hexdigest()
            require(
                sha == binding["byte_sha256"] == frozen["documents"].get(binding["path"]),
                "Frozen document bytes differ",
            )
            documents[binding["path"]] = sha
        require(V2_DOCUMENTS["plan"] in documents, "Missing prospective v2 behavior plan")
        exposure = manifest["prior_exposure"]
        require(exposure["path"] in documents, "Missing frozen prior-exposure assessment")
        exposure_bytes = archive.extractfile(exposure["path"]).read()
        require(
            hashlib.sha256(exposure_bytes).hexdigest()
            == exposure["assessment_byte_sha256"]
            == documents[exposure["path"]]
            and json.loads(exposure_bytes) == exposure["assessment"],
            "Prior-exposure assessment differs from frozen bytes",
        )
        require(exposure["path"] == V2_DOCUMENTS["exposure"], "Wrong v2 exposure document")
        for key, name in (
            ("historical_exclusions", "history"),
            ("tokenization_preflight", "preflight"),
            ("material_comparison", "materials"),
        ):
            bound = manifest[key]
            require(
                bound["path"] == V2_DOCUMENTS[name] and bound["path"] in documents,
                "Missing v2 prospective receipt",
            )
            payload = archive.extractfile(bound["path"]).read()
            require(
                hashlib.sha256(payload).hexdigest() == bound["receipt_byte_sha256"]
                and json.loads(payload) == bound["receipt"]
                and bound["receipt_sha256"] == digest(bound["receipt"]),
                "V2 prospective receipt differs from frozen bytes",
            )
        history = manifest["historical_exclusions"]["receipt"]
        plan = history_plan(history)
        exposure_binding(history, exposure["assessment"], plan)
        prospective_checks(
            plan,
            manifest["tokenization_preflight"]["receipt"],
            manifest["material_comparison"]["receipt"],
            manifest["task_module_sha256"],
        )
        for receipt, field, script in (
            (history, "audit_script_byte_sha256", "audit_voluntary_freshness.py"),
            (
                manifest["tokenization_preflight"]["receipt"],
                "preflight_script_byte_sha256",
                "preflight_voluntary_tokenization.py",
            ),
            (
                manifest["material_comparison"]["receipt"],
                "comparison_script_byte_sha256",
                "compare_voluntary_materials.py",
            ),
        ):
            require(
                hashlib.sha256(archive.extractfile(f"scripts/{script}").read()).hexdigest() == receipt[field],
                "Prospective check script differs from its frozen implementation",
            )
        for name, sha in history["source_file_sha256"].items():
            require(
                not Path(name).is_absolute()
                and ".." not in Path(name).parts
                and hashlib.sha256(archive.extractfile(name).read()).hexdigest() == sha,
                "Historical canonicalization source differs from its frozen bytes",
            )
        source_files = {}
        for member in archive.getmembers():
            prefix = "src/curiosity_study/"
            if member.isfile() and member.name.startswith(prefix) and member.name.endswith(".py"):
                source_files[member.name[len(prefix) :]] = archive.extractfile(member).read().decode()
        require(digest(source_files) == manifest["source_sha256"], "Archived source digest differs")
        expected = {"voluntary_resolution.py": manifest["task_module_sha256"], **manifest["engine_sha256"]}
        require(
            {"voluntary_capture.py", "thinking_information.py", "state_backend.py"}
            <= manifest["engine_sha256"].keys(),
            "Incomplete voluntary execution binding",
        )
        for name, sha in expected.items():
            require(
                hashlib.sha256(source_files[name].encode()).hexdigest() == sha,
                "Frozen engine or task file differs",
            )
        for module in (voluntary, capture, thinking):
            name = Path(module.__file__).name
            require(
                byte_sha(module.__file__) == expected[name],
                "Audit task/receipt implementation differs from freeze",
            )
    return dict(freeze_sha256=digest(frozen), archive_sha256=frozen["archive_sha256"], documents=documents)


def artifact_arrays(root, artifact, cache, prefix, offset):
    relative = Path(artifact["path"])
    require(
        not relative.is_absolute()
        and ".." not in relative.parts
        and len(relative.parts) == 3
        and relative.parts[0] == prefix
        and relative.name == f"{offset:07d}.npz",
        "Capture path does not match its assigned cohort",
    )
    key = str(relative)
    if key not in cache:
        require(byte_sha(root / relative) == artifact["sha256"], "Capture byte hash mismatch")
        with np.load(root / relative, allow_pickle=False) as data:
            cache[key] = (artifact["sha256"], {name: data[name] for name in data.files})
    sha, arrays = cache[key]
    require(artifact["sha256"] == sha, "One capture file has inconsistent bindings")
    return relative.parts[1], arrays


def rendered_input(tokenizer, messages, thinking_enabled):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking_enabled
    )
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return text, encoded["input_ids"], [list(value) for value in encoded["offset_mapping"]]


def generation_record(record, assignment, tokenizer, eos_ids, vocabulary_size, width, arrays, row, manifest):
    """Recompute actual scoring, model input, every saved position, and missingness."""
    inference = record["inference"]
    request = thinking.inference_request(assignment)
    text, ids, offsets = rendered_input(tokenizer, request["messages"], True)
    require(
        inference["messages"] == request["messages"]
        and inference["rendered_prompt"] == text
        and inference["prompt_sha256"] == digest(text),
        "Source prompt or rendering differs",
    )
    pad = width - len(ids)
    require(
        pad >= 0
        and inference["unpadded_input_ids"] == ids
        and inference["input_ids"] == [tokenizer.pad_token_id] * pad + ids
        and inference["attention_mask"] == [0] * pad + [1] * len(ids)
        and inference["position_ids"] == [0] * pad + list(range(len(ids))),
        "Source tokenization, padding, mask, or positions differ",
    )
    require(
        inference["enable_thinking"] is True
        and inference["intervention"] is None
        and inference["hooks_remaining"] == 0
        and inference["capture"] == {}
        and inference["sampling"]
        == dict(seed=assignment["sampling_seed"], temperature=0.7, top_p=1.0, max_new_tokens=4096),
        "Source generation regime or read-only capture differs",
    )
    generated = inference["generated_token_ids"]
    require(
        1 <= len(generated) <= 4096
        and all(type(token) is int and 0 <= token < vocabulary_size for token in generated),
        "Malformed or out-of-allowance generated tokens",
    )
    ended = generated[-1] in eos_ids
    require(
        inference["ended_with_eos"] is ended
        and not any(token in eos_ids for token in generated[:-1])
        and (ended or len(generated) == 4096),
        "Generation did not terminate at EOS or its frozen cap",
    )
    require(
        tokenizer.decode(generated, skip_special_tokens=True) == inference["raw"],
        "Raw answer does not decode from actual tokens",
    )
    scored = voluntary.record_from_thinking_inference(assignment, inference)
    for key, value in scored.items():
        require(digest(record.get(key)) == digest(value), f"Source scoring/ledger mismatch: {key}")
    task = record["task"]
    observed = {
        fact["attribute"]: fact["value"] for event in task["observation_ledger"] for fact in event["facts"]
    }
    compatible = [
        i
        for i, candidate in enumerate(task["candidates"])
        if all(candidate[key] == value for key, value in observed.items())
    ]
    require(
        compatible == task["compatible"]
        and len(compatible) == (1 if assignment["delivery"] == "decisive_evidence" else 2),
        "Independent finite-world delivery oracle failed",
    )
    end_ids = tokenizer.encode("</think>", add_special_tokens=False)
    require(len(end_ids) == 1 and text.count("End of record.") == 1, "Ambiguous reasoning/record boundary")
    closures = [i for i, token in enumerate(generated) if token == end_ids[0]]
    boundary = text.index("End of record.") + len("End of record.") - 1
    matches = [i for i, (start, end) in enumerate(offsets) if start <= boundary < end]
    require(
        len(matches) == 1 and not text[boundary + 1 : offsets[matches[0]][1]].strip(),
        "Record token contains later instructions",
    )
    positions = dict(record=matches[0], prefill=len(ids) - 1)
    for step in manifest["capture_plan"]["steps"]:
        positions[f"token_{step}"] = len(ids) + step - 1 if len(generated) > step else -1
    positions["think_end"] = len(ids) + closures[-1] if closures and closures[-1] < len(generated) - 1 else -1
    positions["decision"] = -1
    if closures:
        for index in range(closures[-1] + 1, len(generated)):
            if (
                tokenizer.decode([generated[index]], skip_special_tokens=False).strip()
                or generated[index] in eos_ids
            ):
                positions["decision"] = len(ids) + index - 1
                break
    names = list(positions)
    artifact = record["activation_artifact"]
    require(
        arrays["layers"].tolist() == artifact["layers"] == manifest["capture_plan"]["layers"]
        and arrays["positions"].tolist() == artifact["positions"] == names
        and artifact["row"] == row
        and artifact["max_new_tokens"] == 4096
        and arrays["trial_ids"][row] == assignment["trial_id"],
        "Source capture row or site mismatch",
    )
    expected_positions = list(positions.values())
    require(
        arrays["token_positions"][row].tolist() == artifact["token_positions"] == expected_positions,
        "Source capture is at the wrong token",
    )
    available, hidden = arrays["available"][row], arrays["hidden"][row]
    expected_available = np.tile(np.array(expected_positions) >= 0, (4, 1))
    require(
        np.array_equal(available, expected_available)
        and artifact["available"] == available.tolist()
        and np.array_equal(np.isfinite(hidden).all(-1), available)
        and np.isnan(hidden[~available]).all(),
        "Source missingness or finite capture mask differs",
    )
    require(
        int(arrays["closure_counts"][row]) == record["closing_delimiter_count"] == len(closures),
        "Closing delimiter count mismatch",
    )
    require(
        arrays["decision_labels"][row].tolist() == list("ABCD")
        and arrays["decision_label_ids"][row].tolist()
        == [tokenizer.encode(label, add_special_tokens=False)[0] for label in "ABCD"],
        "Decision label token binding differs",
    )
    for name in ("decision_label_logits", "decision_label_probabilities_temperature_1"):
        values = arrays[name][row]
        if positions["decision"] >= 0:
            require(
                np.isfinite(values).all() and record[name] == dict(zip("ABCD", map(float, values))),
                "Decision score mismatch",
            )
        else:
            require(np.isnan(values).all() and record[name] is None, "Unavailable decision was imputed")
    probabilities = arrays["decision_label_probabilities_temperature_1"][row]
    if positions["decision"] >= 0:
        require(
            (probabilities >= 0).all() and probabilities.sum() <= 1.00001, "Invalid decision probabilities"
        )
    return dict(
        trial_id=assignment["trial_id"],
        record_sha256=digest(record),
        artifact_sha256=artifact["sha256"],
        row=row,
        token_prefix_sha256=digest(ids),
    )


def receipt_tokens(source, tokenizer):
    """Independently locate the selected payload in its fresh actual render."""
    receipt = voluntary.receipt_input(source)
    require(receipt is not None, "Stop/invalid choices cannot be rendered as receipts")
    text, ids, offsets = rendered_input(tokenizer, receipt["messages"], False)
    last = receipt["messages"][-1]["content"]
    start = text.rfind(last)
    payload = source["simulator_outcome"]["payload"]
    end = start + len(payload)
    require(
        start >= 0 and last == payload + "\nEpisode ended. No response is requested.",
        "Receipt must contain exactly the selected payload and neutral suffix",
    )
    matches = [i for i, (left, right) in enumerate(offsets) if left <= end - 1 < right]
    require(
        len(matches) == 1
        and matches[0] < len(ids) - 1
        and not text[end : offsets[matches[0]][1]].strip()
        and offsets[-1][1] == len(text),
        "Receipt payload/final token boundaries are ambiguous",
    )
    positions = dict(record=matches[0], final=len(ids) - 1)
    binding = dict(
        rendered_prompt=text,
        rendered_prompt_sha256=digest(text),
        unpadded_input_ids=ids,
        token_prefix_sha256=digest(ids),
        prefix_positions=positions,
        offset_mapping=offsets,
        selected_payload_span=[start, end],
        final_user_message_span=[start, start + len(last)],
        enable_thinking=False,
        add_generation_prompt=True,
        add_special_tokens=False,
        exact_source_token_replay=False,
        template_may_strip_prior_thinking=True,
        rendered_contains_raw_choice=source["inference"]["raw"] in text,
        source_raw_choice_sha256=digest(source["inference"]["raw"]),
        rendering_policy="Fresh HF chat-template rendering of actual choice messages and selected payload; prior assistant reasoning may be stripped. Only these recorded token IDs are claimed received.",
    )
    return receipt, binding


def receipt_record(sidecar, source, tokenizer, width=None, arrays=None, active_row=None):
    """A simulator event alone never certifies that the model received anything."""
    expected = capture.choice_sidecar(source)
    stable = (
        "schema_version",
        "source_trial_id",
        "assignment_sha256",
        "source_record_sha256",
        "selected_service",
        "simulator_outcome_sha256",
        "terminal",
        "receipt_required",
    )
    require(
        all(sidecar.get(key) == expected[key] for key in stable),
        "Receipt source/assignment/selected-action binding differs",
    )
    status = sidecar["status"]
    if not expected["receipt_required"]:
        require(sidecar == expected, "Skipped receipt must preserve the exact invalid/stop disposition")
        return None
    if status == "execution_error":
        require(
            sidecar.get("receipt_model_pass_completed") is False
            and sidecar.get("measurement") is None
            and sidecar.get("activation_artifact") is None
            and sidecar.get("cohort") is None
            and sidecar.get("failure") == "receipt_execution_error"
            and sidecar.get("executed_simulator_outcome") == source["simulator_outcome"],
            "Failed receipt cannot claim captured evidence or a completed model pass",
        )
        return None
    require(
        status == "completed"
        and sidecar["receipt_model_pass_completed"] is True
        and arrays is not None
        and type(active_row) is int
        and "failure" not in sidecar,
        "Required receipt has no actual completed capture",
    )
    receipt, binding = receipt_tokens(source, tokenizer)
    require(
        sidecar["receipt_input"] == receipt and sidecar["token_binding"] == binding,
        "Receipt payload, rendered tokens, or truthful replay disclosure changed",
    )
    outcome = deepcopy(source["simulator_outcome"])
    outcome["receipt_model_pass_completed"] = True
    outcome["observation_ledger"][-1]["model_pass_completed"] = True
    require(
        sidecar["executed_simulator_outcome"] == outcome,
        "Executed receipt ledger changed more than completion flags",
    )
    measurement, artifact = sidecar["measurement"], sidecar["activation_artifact"]
    ids, positions = binding["unpadded_input_ids"], binding["prefix_positions"]
    pad = width - len(ids)
    require(
        measurement["unpadded_input_ids"] == ids
        and measurement["token_prefix_sha256"] == digest(ids)
        and measurement["input_ids"] == [tokenizer.pad_token_id] * pad + ids
        and measurement["attention_mask"] == [0] * pad + [1] * len(ids)
        and measurement["position_ids"] == [0] * pad + list(range(len(ids)))
        and measurement["prefill_width"] == width
        and measurement["prefill_padding_multiple"] == PREFILL_PADDING_MULTIPLE,
        "Receipt actual input tokens, padding, mask, or positions differ",
    )
    require(
        measurement["unpadded_token_positions"] == positions
        and measurement["record_token_index"] == pad + positions["record"]
        and measurement["exact_token_replay"] is True
        and measurement["rendered_prompt"] == tokenizer.decode(ids, skip_special_tokens=False)
        and measurement["prompt_sha256"] == digest(measurement["rendered_prompt"])
        and measurement["hooks_remaining"] == 0
        and measurement["intervention"] is None
        and "completion" not in measurement,
        "Receipt read-only forward or capture location differs",
    )
    # StateBackend receives these newly rendered exact IDs, not the source-generation prefix.
    require(
        set(measurement["label_logits"])
        == set(measurement["label_probabilities_temperature_1"])
        == {"A", "B"}
        and all(np.isfinite(value) for value in measurement["label_logits"].values()),
        "Invalid receipt engineering label scores",
    )
    require(
        artifact["row"] == active_row
        and artifact["layers"] == capture.LAYERS
        and artifact["positions"] == capture.POSITIONS
        and artifact["token_positions"] == list(positions.values())
        and artifact["available"] == [[True, True]] * 4,
        "Receipt artifact row or position metadata differs",
    )
    require(
        arrays["trial_ids"][active_row] == source["trial_id"]
        and arrays["token_prefix_sha256"][active_row] == digest(ids)
        and arrays["token_positions"][active_row].tolist() == list(positions.values()),
        "Receipt NPZ token/source row binding differs",
    )
    return dict(
        source_trial_id=source["trial_id"],
        sidecar_sha256=digest(sidecar),
        source_record_sha256=digest(source),
        artifact_sha256=artifact["sha256"],
        row=active_row,
        token_prefix_sha256=digest(ids),
    )


def receipt_engineering(engineering, active_ids, prefix_hashes, arrays, width):
    checks = {
        "row_coverage",
        "full_initial_logits_equal",
        "every_row_fingerprint_equal",
        "all_captures_finite",
        "capture_shape",
        "no_intervention_or_generation",
        "hook_cleanup",
    }
    require(
        engineering["passed"] is True
        and set(engineering["checks"]) == checks
        and all(value is True for value in engineering["checks"].values()),
        "Receipt engineering failed",
    )
    before, after = (engineering[f"{phase}_full_initial_logits_sha256"] for phase in ("baseline", "captured"))
    require(
        len(before) == len(active_ids)
        and before == after
        and all(
            isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
            for value in before
        ),
        "Read-only receipt capture changed full initial vocabulary logits",
    )
    hidden = arrays["hidden"]
    require(
        engineering["source_trial_ids"] == active_ids
        and engineering["token_prefix_sha256"] == prefix_hashes
        and engineering["layers"] == capture.LAYERS
        and engineering["positions"] == capture.POSITIONS
        and engineering["active_batch_size"] == len(active_ids)
        and engineering["prefill_width"] == width
        and engineering["max_new_tokens"] == 0
        and engineering["captured_array_sha256"] == capture.array_sha(hidden),
        "Receipt engineering does not bind its actual active cohort and capture array",
    )
    require(
        hidden.ndim == 4
        and hidden.shape[:3] == (len(active_ids), 4, 2)
        and hidden.shape[-1] > 0
        and np.isfinite(hidden).all()
        and arrays["available"].shape == hidden.shape[:-1]
        and arrays["available"].all()
        and arrays["layers"].tolist() == capture.LAYERS
        and arrays["positions"].tolist() == capture.POSITIONS
        and arrays["trial_ids"].tolist() == active_ids,
        "Receipt cohort captures are missing, nonfinite, or misordered",
    )


def audit_receipts(root, manifest, records, tokenizer):
    assignments = [a for a in manifest["diagnostics"] if a["kind"] == "choice"]
    sidecars, seen, arrays, bindings, cohort_bindings = {}, set(), {}, [], []
    for path in sorted((root / "receipt_cohorts").glob("*/*.json")):
        envelope = read(path)
        offset = envelope["choice_offset"]
        require(
            type(offset) is int
            and offset in range(0, 144, 32)
            and offset not in seen
            and path.name == f"{offset:07d}.json",
            "Duplicate or out-of-range receipt cohort",
        )
        ids = [a["trial_id"] for a in assignments[offset : offset + 32]]
        rows = envelope["records"]
        require(
            envelope["schema_version"] == 1
            and envelope["manifest_sha256"] == digest(manifest)
            and envelope["source_trial_ids"] == ids
            and [row["source_trial_id"] for row in rows] == ids
            and envelope["records_sha256"] == digest(rows)
            and envelope["rendering_policy"] == capture.RECEIPT_PLAN["prefix_policy"],
            "Receipt logical slots, envelope hashes, or rendering policy differ",
        )
        active_ids = [tid for tid in ids if capture.choice_sidecar(records[tid])["receipt_required"]]
        require(
            envelope["active_source_trial_ids"] == active_ids,
            "Receipt active rows exclude or add selected payloads",
        )
        active = [row for row in rows if row["source_trial_id"] in active_ids]
        completed = [row for row in active if row["status"] == "completed"]
        require(
            not completed or len(completed) == len(active), "Partial receipt forward cannot be claimed atomic"
        )
        data, width = None, None
        if completed:
            first = completed[0]["activation_artifact"]
            attempt, data = artifact_arrays(root, first, arrays, "receipt_activations", offset)
            require(attempt == path.parent.name, "Receipt attempt directory and artifact differ")
            prefixes = [receipt_tokens(records[tid], tokenizer)[1] for tid in active_ids]
            longest = max(len(prefix["unpadded_input_ids"]) for prefix in prefixes)
            width = (
                (longest + PREFILL_PADDING_MULTIPLE - 1) // PREFILL_PADDING_MULTIPLE
            ) * PREFILL_PADDING_MULTIPLE
            receipt_engineering(
                envelope["engineering"],
                active_ids,
                [prefix["token_prefix_sha256"] for prefix in prefixes],
                data,
                width,
            )
            environment = read(root / "receipt_environments" / f"{attempt}.json")
            require(
                all(environment.get(key) == value for key, value in manifest["model"].items())
                and environment["receipt_hidden_size"] == data["hidden"].shape[-1],
                "Receipt model identity or hidden width differs",
            )
        elif not active:
            require(envelope["engineering"] is None, "All-stop/invalid cohort cannot claim a receipt forward")
        for sidecar in rows:
            tid = sidecar["source_trial_id"]
            index = active_ids.index(tid) if tid in active_ids else None
            if sidecar["status"] == "completed":
                artifact = sidecar["activation_artifact"]
                require(
                    artifact["path"] == first["path"] and artifact["sha256"] == first["sha256"],
                    "One active cohort must share a single capture file",
                )
                require(
                    sidecar["cohort"]
                    == dict(
                        id=f"{path.parent.name}-{offset:07d}",
                        choice_offset=offset,
                        source_trial_ids=ids,
                        active_source_trial_ids=active_ids,
                        active_row=index,
                        active_size=len(active_ids),
                        prefill_width=width,
                    ),
                    "Receipt cohort layout differs",
                )
            binding = receipt_record(sidecar, records[tid], tokenizer, width, data, index)
            if binding is not None:
                bindings.append(binding)
            sidecars[tid] = sidecar
        seen.add(offset)
        cohort_bindings.append(dict(path=str(path.relative_to(root)), envelope_sha256=digest(envelope)))
    return sidecars, dict(
        cohorts=cohort_bindings,
        record_bindings=bindings,
        record_bindings_sha256=digest(bindings),
        capture_files=len(arrays),
    )


def startup_integrity(root, attempt, manifest):
    trace = read(root / "trace_integrity" / f"{attempt}.json")
    expected = {"tokens_equal", "probabilities_equal", "no_hooks", "prefill_captures_complete"}
    require(
        trace["passed"] is True
        and set(trace["checks"]) == expected
        and all(value is True for value in trace["checks"].values()),
        "Source capture startup failed",
    )
    baseline, measured = trace["baseline"], trace["measured"]
    require(
        len(baseline) == len(measured) == 2
        and all(
            a["generated_token_ids"] == b["generated_token_ids"]
            and a["label_probabilities_temperature_1"] == b["label_probabilities_temperature_1"]
            and a["hooks_remaining"] == b["hooks_remaining"] == 0
            and a["intervention"] is b["intervention"] is None
            for a, b in zip(baseline, measured)
        ),
        "Source startup flags disagree with recorded evidence",
    )
    environment = read(root / "environments" / f"{attempt}.json")
    require(
        all(environment.get(key) == value for key, value in manifest["model"].items()),
        "Source environment model identity differs",
    )
    return dict(
        trace_sha256=digest(trace),
        environment_sha256=digest(environment),
        scope="Existing two-row read-only generation check; not causal intervention validation",
    )


def candidate_binding(root, manifest, frozen):
    """The candidate analysis is provenance only; never rescored or fitted here."""
    binding = manifest["candidate_source"]
    names = dict(
        capture_manifest="runs/qwen32-curiosity-satisfaction-candidates-v1/manifest.json",
        capture_audit="reports/qwen32-curiosity-satisfaction-candidates-v1-audit.json",
        analysis="analyses/curiosity-satisfaction-candidates-v1/analysis.json",
        directions="analyses/curiosity-satisfaction-candidates-v1/directions.npz",
        independent_check="reports/curiosity-satisfaction-candidates-v1-independent-check.json",
    )
    with tarfile.open(root / "source.tar.gz", "r:gz") as archive:
        values = {}
        for key, name in names.items():
            require(name in frozen["documents"], "Missing frozen candidate provenance document")
            payload = archive.extractfile(name).read()
            require(
                hashlib.sha256(payload).hexdigest() == frozen["documents"][name],
                "Candidate document byte hash differs",
            )
            if key != "directions":
                values[key] = json.loads(payload)
        require(
            binding["capture_manifest_sha256"] == digest(values["capture_manifest"])
            and binding["capture_audit_sha256"] == digest(values["capture_audit"])
            and binding["analysis_sha256"] == digest(values["analysis"])
            and binding["analysis_byte_sha256"] == frozen["documents"][names["analysis"]]
            and binding["directions_byte_sha256"] == frozen["documents"][names["directions"]]
            and binding["independent_check_sha256"] == digest(values["independent_check"]),
            "Locked candidate provenance differs",
        )
        analysis, independent = values["analysis"], values["independent_check"]
        expected_sites = {
            name: dict(layer=15, position="record") for name in REQUIRED_ANALYSIS_METADATA["candidate_layers"]
        }
        require(
            binding["selected_sites"] == expected_sites
            and values["capture_audit"]["passed"] is True
            and independent["passed"] is True
            and analysis["manifest_sha256"] == binding["capture_manifest_sha256"]
            and analysis["audit_sha256"] == binding["capture_audit_sha256"]
            and independent["analysis_sha256"] == binding["analysis_sha256"]
            and analysis["directions_sha256"] == binding["directions_byte_sha256"],
            "Candidate source is not the independently checked fixed-A analysis",
        )
        for candidate in analysis["candidates"]:
            require(
                candidate["contrast"] in expected_sites
                and candidate["passed"] is True
                and candidate["selected_layer"] == 15
                and candidate["position"] == "record",
                "A candidate cannot be replaced after behavioral collection",
            )
        require(
            {row["contrast"] for row in analysis["candidates"]} == set(expected_sites),
            "Missing selected candidate",
        )
    return binding


def compact_record(assignment, record):
    result = {
        key: assignment[key]
        for key in (
            "trial_id",
            "family",
            "kind",
            "delivery",
            "menu_layout",
            "seed",
            "index",
            "world_slot",
            "semantic_world_id",
        )
    }
    if record is None:
        return dict(**result, status="missing", valid=False, correct=None, selected_service=None)
    if "failure" in record:
        return dict(**result, status="execution_failure", valid=False, correct=None, selected_service=None)
    return dict(
        **result,
        status="completed",
        record_sha256=digest(record),
        valid=record["valid"],
        correct=record["correct"],
        thinking_closed=record["thinking_closed"],
        parsed_answer=record["parsed_answer"],
        selected_service=record["selected_service"],
        cap_hit=record["cap_hit"],
        generated_tokens=len(record["inference"]["generated_token_ids"]),
        activation_artifact=record["activation_artifact"],
    )


def audit(root, tokenizer=None, eos_token_ids=None):
    root = Path(root)
    manifest = read(root / "manifest.json")
    assignments = capture.validate_manifest(manifest)
    require(
        digest(manifest["readiness_criteria"]) == digest(REQUIRED_CRITERIA)
        and digest(manifest["analysis_metadata"]) == digest(REQUIRED_ANALYSIS_METADATA),
        "Readiness or descriptive analysis differs from the prospective contract",
    )
    history = manifest["historical_exclusions"]["receipt"]
    plan = history_plan(history)
    require(
        assignments == plan["assignments"]
        and manifest["world_selection"] == plan["selection"]
        and manifest["world_selection_sha256"] == plan["selection_sha256"]
        and manifest["counterbalancing"] == plan["counterbalancing"],
        "Assigned order, semantic worlds, or counterbalancing differs",
    )
    require(
        manifest.get("inference_scope") == "development"
        and manifest["capture_plan"] == thinking.CAPTURE_PLAN,
        "This pilot requires the frozen development capture plan",
    )
    frozen = archive_bindings(root, manifest)
    history_check = recheck_history(read(root / "freeze.json")["budget_root"], history, plan)
    candidates = candidate_binding(root, manifest, frozen)
    records = RunStore(root, manifest).completed()
    compact = [compact_record(a, records.get(a["trial_id"])) for a in assignments]
    base = dict(
        schema_version=1,
        assay="voluntary_resolution",
        manifest_sha256=digest(manifest),
        source_sha256=manifest["source_sha256"],
        model=manifest["model"],
        assigned=288,
        assigned_choices=144,
        assigned_diagnostics=144,
        max_new_tokens=4096,
        inference_scope="development",
        source_freeze=frozen,
        historical_exclusions=history_check,
        candidate_source=candidates,
        auditor_byte_sha256=byte_sha(__file__),
        analysis_metadata=REQUIRED_ANALYSIS_METADATA,
        outcomes=compact,
        choice_correctness=None,
        causal_admission=False,
        interpretation="Development readiness and receipt integrity only. No preferred-action gate, binary feeling classification, causal admission, or subjective-experience claim.",
    )
    if set(records) != {a["trial_id"] for a in assignments} or any(
        "failure" in row for row in records.values()
    ):
        return dict(
            **base,
            complete=False,
            integrity_passed=False,
            passed=False,
            readiness=readiness(assignments, records, {}),
            integrity_error="Missing or failed source generations; receipt integrity cannot be established.",
        )
    coverage = check_collection_coverage(root, expected_manifest_sha256=digest(manifest))
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            manifest["model"]["model_id"], revision=manifest["model"]["revision"], trust_remote_code=False
        )
    if eos_token_ids is None:
        from transformers import GenerationConfig

        config = GenerationConfig.from_pretrained(
            manifest["model"]["model_id"], revision=manifest["model"]["revision"]
        )
        eos_token_ids = config.eos_token_id
    eos_ids = {eos_token_ids} if type(eos_token_ids) is int else set(eos_token_ids)
    require(bool(eos_ids), "Pinned generation EOS IDs are required")
    vocabulary_size = len(tokenizer)  # Cache once; HF tokenizer length may enumerate its vocabulary.
    arrays, startups, bindings = {}, {}, []
    for offset in range(0, 288, 32):
        cohort = assignments[offset : offset + 32]
        ids = [a["trial_id"] for a in cohort]
        texts = [
            rendered_input(tokenizer, thinking.inference_request(a)["messages"], True)[0] for a in cohort
        ]
        width = max(len(tokenizer.encode(text, add_special_tokens=False)) for text in texts)
        batch_id = digest(dict(cohort=ids, prompts=texts, seeds=[a["sampling_seed"] for a in cohort]))
        paths = set()
        for index, assignment in enumerate(cohort):
            record = records[assignment["trial_id"]]
            artifact = record["activation_artifact"]
            attempt, data = artifact_arrays(root, artifact, arrays, "activations", offset)
            if attempt not in startups:
                startups[attempt] = startup_integrity(root, attempt, manifest)
            require(
                record["inference"]["batch"]["id"] == batch_id
                and data["trial_ids"].tolist() == ids
                and data["hidden"].shape[:3] == (32, 4, 7)
                and data["available"].shape == (32, 4, 7),
                "Generation capture/cohort shape differs",
            )
            bindings.append(
                generation_record(
                    record, assignment, tokenizer, eos_ids, vocabulary_size, width, data, index, manifest
                )
            )
            paths.add(artifact["path"])
        require(len(paths) == 1, "Each source cohort must share one activation artifact")
    sidecars, receipt_bindings = audit_receipts(root, manifest, records, tokenizer)
    ready = readiness(assignments, records, sidecars)
    complete = ready["checks"]["source_coverage"] and ready["checks"]["receipt_slot_coverage"]
    integrity = (
        complete
        and ready["checks"]["required_receipts_complete"]
        and ready["checks"]["no_execution_failures"]
    )
    service_counts = []
    for family in voluntary.FAMILIES:
        for delivery in voluntary.DELIVERIES:
            for menu in (0, 1):
                rows = [
                    row
                    for row in compact
                    if row["kind"] == "choice"
                    and row["family"] == family
                    and row["delivery"] == delivery
                    and row["menu_layout"] == menu
                ]
                counts = Counter(row["selected_service"] if row["valid"] else "invalid" for row in rows)
                service_counts.append(
                    dict(
                        family=family,
                        delivery=delivery,
                        menu_layout=menu,
                        assigned=len(rows),
                        counts={name: counts[name] for name in (*voluntary.SERVICES, "invalid")},
                    )
                )
    return dict(
        **base,
        complete=complete,
        integrity_passed=integrity,
        passed=integrity and ready["passed"],
        readiness=ready,
        coverage=coverage,
        startup_bindings=startups,
        generation_record_bindings_sha256=digest(bindings),
        capture_files=len(arrays),
        receipt_bindings=receipt_bindings,
        service_counts=service_counts,
        generated_tokens=sum(row["generated_tokens"] for row in compact),
        cap_hits=sum(row["cap_hit"] for row in compact),
        receipt_status_counts=dict(Counter(row["status"] for row in sidecars.values())),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "Voluntary audit receipts are immutable")
    result = audit(args.run)
    write_once(args.out, result)
    print({key: result[key] for key in ("complete", "integrity_passed", "passed", "assigned")})
