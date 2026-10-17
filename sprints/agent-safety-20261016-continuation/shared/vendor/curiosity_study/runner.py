"""Unsteered baseline with separately branched diagnostics and immutable assignments."""

from pathlib import Path
import time
import uuid
from .artifacts import RunStore, digest, write_once
from .assignments import instantiate
from .diagnostics import DIAGNOSTIC_KINDS, prompt_for

FORMAT_INSTRUCTION = {
    "role": "system",
    "content": "Follow the requested response format exactly. Reply with a single allowed ID or digit "
    "and nothing else. Do not explain your choice.",
}


def source_hash():
    root = Path(__file__).parent
    return digest({str(path.relative_to(root)): path.read_text() for path in sorted(root.rglob("*.py"))})


def diagnostic_assignments(assignments):
    seen = set()
    records = []
    for assignment in assignments:
        key = (assignment["family"], assignment["index"])
        if key in seen:
            continue
        seen.add(key)
        for kind in DIAGNOSTIC_KINDS:
            copy = dict(assignment)
            copy["kind"] = kind
            copy["evidence"] = "resolved" if kind == "resolved_accuracy" else "unresolved"
            copy["price"] = 2
            copy["trial_id"] = f"diagnostic-{assignment['split']}-{key[0]}-{key[1]:04d}-{kind}"
            records.append(copy)
    return records


def run_diagnostic(assignment, model):
    trial = instantiate(assignment)
    task = trial.task
    kind = assignment["kind"]
    content, labels, expected, details = prompt_for(trial, assignment)
    inference = model.infer(
        [FORMAT_INSTRUCTION, {"role": "user", "content": content}],
        assignment["sampling_seed"],
        labels=labels,
        greedy=True,
    )
    raw = inference["raw"].strip()
    return {
        "trial_id": assignment["trial_id"],
        "assignment": assignment,
        "task": task.as_dict(),
        "kind": kind,
        "terminal": True,
        "inference": inference,
        "expected": expected,
        "correct": None if expected is None else raw == expected,
        "valid": raw in set(labels),
        "diagnostic_details": details,
    }


def run_trial(assignment, model):
    trial = instantiate(assignment)
    messages = [FORMAT_INSTRUCTION, {"role": "user", "content": trial.prompt()}]
    calls = []
    while not trial.done:
        inference = model.infer(messages.copy(), assignment["sampling_seed"] + len(calls))
        calls.append(inference)
        event = trial.apply(inference["raw"])
        messages.append({"role": "assistant", "content": inference["raw"]})
        messages.append(
            {
                "role": "user",
                "content": event["response"]
                + f"\nBalance: {trial.balance} points."
                + (" Trial ended." if trial.done else " Choose one action ID only: A, B, C, or D."),
            }
        )
    return {
        "trial_id": assignment["trial_id"],
        "assignment": assignment,
        "task": trial.task.as_dict(),
        "terminal": True,
        "kind": "behavior",
        "calls": calls,
        "ledger": trial.ledger,
        "final_messages": messages,
        "final_balance": trial.balance,
        "first_service": trial.ledger[0]["service"] or "invalid",
        "failure": next((e["failure"] for e in trial.ledger if e["failure"]), None),
    }


def build_manifest(assignments, model_identity, phase="full"):
    if phase not in ("full", "validation", "behavior"):
        raise ValueError("Unknown phase")
    return {
        "schema_version": 2,
        "task_revision": 2,
        "phase": phase,
        "purpose": "unsteered_development_only",
        "source_sha256": source_hash(),
        "model": model_identity,
        "assignments": [] if phase == "validation" else assignments,
        "diagnostics": [] if phase == "behavior" else diagnostic_assignments(assignments),
        "planned_behavior_sha256": digest(assignments),
        "validation_criteria": {
            "minimum_items_per_family_kind": 100,
            "minimum_accuracy": 0.9,
            "minimum_valid_format": 0.95,
        },
        "sampling": {"temperature": 0.7, "top_p": 1.0, "max_new_tokens": 8},
        "max_purchases": 3,
        "invalid_output_policy": "terminal_no_retry",
    }


def execute(root, manifest, model, checkpoint=None, deadline=None):
    if manifest.get("assay") == "curiosity_suppression_screen":
        from .curiosity_suppression_capture import execute as execute_curiosity

        return execute_curiosity(root, manifest, model, checkpoint, deadline)
    if manifest.get("assay") == "satisfaction_screen":
        from .satisfaction_capture import execute as execute_satisfaction

        return execute_satisfaction(root, manifest, model, checkpoint, deadline)
    if manifest.get("assay") == "sustained_engineering":
        from .sustained_engineering import execute as execute_sustained

        return execute_sustained(root, manifest, model, checkpoint, deadline)
    if manifest.get("assay") == "voluntary_resolution":
        from .voluntary_capture import execute as execute_voluntary

        return execute_voluntary(root, manifest, model, checkpoint, deadline)
    if manifest.get("assay") == "concept_capture":
        from .concept_capture import execute as execute_concepts

        return execute_concepts(root, manifest, model, checkpoint, deadline)
    if manifest.get("assay") == "natural_information":
        from .natural_information import execute as execute_natural

        return execute_natural(root, manifest, model, checkpoint, deadline)
    if manifest.get("assay") == "replay_information":
        from .replay_information import execute as execute_replay

        return execute_replay(root, manifest, model, checkpoint, deadline)
    if manifest.get("assay") == "thinking_information":
        from .thinking_information import execute as execute_thinking

        return execute_thinking(root, manifest, model, checkpoint, deadline)
    if manifest.get("assay") == "state_mechanism":
        from .state_experiment import execute as execute_state

        return execute_state(root, manifest, model, checkpoint, deadline)
    if manifest.get("execution", {}).get("backend") == "torch_batched_v1":
        from .batched import execute_batches

        return execute_batches(root, manifest, model, checkpoint, deadline)
    store = RunStore(root, manifest)
    completed = store.completed()
    attempt = uuid.uuid4().hex
    write_once(Path(root) / "environments" / f"{attempt}.json", model.metadata())
    assignments = manifest["assignments"] + manifest["diagnostics"]
    start = time.monotonic()
    for assignment in assignments:
        if assignment["trial_id"] in completed:
            continue
        if deadline is not None and time.time() >= deadline:
            break
        try:
            if assignment.get("assay") == "identification_factorial":
                from .identification import run

                record = run(assignment, model)
            elif assignment.get("assay") == "single_curiosity":
                from .single_curiosity import run

                record = run(assignment, model)
            elif assignment.get("assay") == "one_credit":
                from .credit import run

                record = run(assignment, model)
            else:
                record = (
                    run_diagnostic(assignment, model)
                    if "kind" in assignment
                    else run_trial(assignment, model)
                )
        except Exception as error:
            # An observed runtime/model failure stays terminal. A killed process has no terminal record.
            record = {
                "trial_id": assignment["trial_id"],
                "assignment": assignment,
                "kind": assignment.get("kind", "behavior"),
                "terminal": True,
                "failure": "execution_error",
                "error_type": type(error).__name__,
                "error": str(error),
                "first_service": "technical_failure",
            }
            store.save(record, attempt)
            if checkpoint:
                checkpoint()
            raise
        store.save(record, attempt)
        completed[assignment["trial_id"]] = record
        if checkpoint:
            checkpoint()
        print(f"{len(completed)}/{len(assignments)} {assignment['trial_id']}", flush=True)
    profile = {
        "attempt": attempt,
        "elapsed_seconds": time.monotonic() - start,
        "completed": len(completed),
        "assigned": len(assignments),
    }
    write_once(Path(root) / "profiles" / f"{attempt}.json", profile)
    if checkpoint:
        checkpoint()
    return store.completed()
