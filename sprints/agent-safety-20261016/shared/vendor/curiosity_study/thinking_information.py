"""Reasoning-enabled information judgments, with read-only trajectory measurements."""

from pathlib import Path
import hashlib
import time
import uuid
import numpy as np
from .artifacts import RunStore, write_once, digest
from . import information_state as state
from .identification import final_answer
from .runner import source_hash

EXECUTION = {"backend": "torch_batched_v1", "batch_size": 32}
CAPTURE_PLAN = {"layers": [15, 31, 47, 57], "steps": [32, 128, 512]}


def generation_limit(binding):
    """Legacy assignments remain 2048; a new regime must be explicit."""
    limit = binding.get("max_new_tokens", 2048)
    if type(limit) is not int or limit not in (2048, 4096):
        raise ValueError("Thinking generation requires an explicit 2048 or 4096 token allowance")
    return limit


def manifest_generation_limit(manifest):
    limit = generation_limit(manifest)
    rows = manifest.get("assignments", []) + manifest["diagnostics"]
    if not rows or any(generation_limit(a) != limit for a in rows):
        raise ValueError("Every assigned kind must share the manifest generation allowance")
    return limit


def task_module(a):
    if str(a.get("task_revision", "")).startswith("voluntary-resolution-"):
        from . import voluntary_resolution

        if a["task_revision"] != voluntary_resolution.REVISION:
            raise ValueError("Use the archived execution source for this voluntary task revision")
        return voluntary_resolution
    if a.get("task_revision") == "target-relevant-information-v1":
        from . import target_relevant_information

        return target_relevant_information
    if a.get("task_revision") == "optional-information-v1":
        from . import optional_information

        return optional_information
    if a.get("task_revision") == "count-competence-v1":
        from . import count_competence

        return count_competence
    if a.get("task_revision") == "relational-information-v1":
        from . import relational_information

        return relational_information
    return state


def inference_request(a):
    request = task_module(a).inference_request(a)
    request.update(enable_thinking=True, max_new_tokens=generation_limit(a))
    return request


def record_from_inference(a, inference):
    answer, closed = final_answer(inference["raw"], True)
    record = task_module(a).record_from_inference(a, {**inference, "raw": answer})
    record["inference"] = inference
    record["thinking_closed"] = closed
    record["valid"] = closed and record["valid"]
    record["correct"] = (
        None
        if record["expected"] is None
        else record["valid"] and record["parsed_answer"] == record["expected"]
    )
    if "max_new_tokens" in a:
        record["generation_max_new_tokens"] = generation_limit(a)
        record["cap_hit"] = not inference.get("ended_with_eos", False) and len(
            inference.get("generated_token_ids", [])
        ) == generation_limit(a)
    if str(a.get("task_revision", "")).startswith("voluntary-resolution-") and a["kind"] == "choice":
        record = task_module(a).finalize_choice(record)
    return record


def build_manifest(model, items=1, seed=2026101407, phase="engineering_smoke"):
    if model["model_id"] != "Qwen/Qwen3-32B":
        raise ValueError("This reasoning-stage plan is pinned to Qwen3-32B")
    manifest = state.build_manifest(model, items, seed)
    rows = [
        {**a, "assay": "thinking_information", "execution": EXECUTION.copy()} for a in manifest["diagnostics"]
    ]
    manifest.update(
        assay="thinking_information",
        phase=phase,
        source_sha256=source_hash(),
        execution=EXECUTION.copy(),
        execution_gpu="H200",
        diagnostics=rows,
        assignment_sha256=digest(rows),
        inference_revision="thinking-2048-v1",
        capture_plan=CAPTURE_PLAN,
        task_module_sha256=hashlib.sha256(Path(state.__file__).read_bytes()).hexdigest(),
        capture_position_rule="Record boundary and prefill; processed reasoning tokens 32,128,512; last processed closing-think token used by the final-answer parser; activation before first non-whitespace final token after that delimiter (including EOS). Missing positions remain missing.",
    )
    return manifest


def verify_capture(model, plan):
    rows = state.assignments(1, seed=424242, kinds=("new",))[:2]
    requests = [{**inference_request(a), "greedy": True, "max_new_tokens": 16} for a in rows]
    ids = ["trace-engineering-a", "trace-engineering-b"]
    baseline = model.infer_batch(requests, ids)
    captures = {}
    measured = model.infer_batch(requests, ids, capture_plan=plan, capture_output=captures)
    checks = dict(
        tokens_equal=all(
            a["generated_token_ids"] == b["generated_token_ids"] for a, b in zip(baseline, measured)
        ),
        probabilities_equal=all(
            a["label_probabilities_temperature_1"] == b["label_probabilities_temperature_1"]
            for a, b in zip(baseline, measured)
        ),
        no_hooks=model.active_hooks == 0,
        prefill_captures_complete=bool(captures["available"][:, :, :2].all()),
    )
    return dict(passed=all(checks.values()), checks=checks, baseline=baseline, measured=measured)


def execute(root, manifest, model, checkpoint=None, deadline=None):
    manifest_generation_limit(manifest)
    root = Path(root)
    store = RunStore(root, manifest)
    completed = store.completed()
    attempt = uuid.uuid4().hex
    write_once(root / "environments" / f"{attempt}.json", model.metadata())
    audit = verify_capture(model, manifest["capture_plan"])
    write_once(root / "trace_integrity" / f"{attempt}.json", audit)
    if not audit["passed"]:
        raise RuntimeError("Read-only capture changed inference")
    if checkpoint:
        checkpoint()
    assignments = manifest["diagnostics"]
    size = manifest["execution"]["batch_size"]
    started = time.monotonic()
    for offset in range(0, len(assignments), size):
        cohort = [a for a in assignments[offset : offset + size] if a["trial_id"] not in completed]
        if not cohort:
            continue
        if deadline is not None and time.time() >= deadline:
            break
        captures = {}
        try:
            inferences = model.infer_batch(
                [inference_request(a) for a in cohort],
                [a["trial_id"] for a in cohort],
                capture_plan=manifest["capture_plan"],
                capture_output=captures,
            )
            relative = Path("activations") / attempt / f"{offset:07d}.npz"
            file = root / relative
            file.parent.mkdir(parents=True, exist_ok=True)
            with file.open("xb") as f:
                np.savez_compressed(f, **captures, trial_ids=np.array([a["trial_id"] for a in cohort]))
            sha = hashlib.sha256(file.read_bytes()).hexdigest()
            records = []
            for row, (a, inf) in enumerate(zip(cohort, inferences)):
                record = record_from_inference(a, inf)
                record["activation_artifact"] = dict(
                    path=str(relative),
                    sha256=sha,
                    row=row,
                    layers=captures["layers"].tolist(),
                    positions=captures["positions"].tolist(),
                    token_positions=captures["token_positions"][row].tolist(),
                    available=captures["available"][row].tolist(),
                )
                if "max_new_tokens" in a:
                    record["activation_artifact"]["max_new_tokens"] = generation_limit(a)
                record["closing_delimiter_count"] = int(captures["closure_counts"][row])
                has_decision = captures["available"][row, :, -1].all()
                for field in ("decision_label_logits", "decision_label_probabilities_temperature_1"):
                    record[field] = (
                        {
                            str(label): float(captures[field][row, col])
                            for col, label in enumerate(captures["decision_labels"][row])
                            if label
                        }
                        if has_decision
                        else None
                    )
                records.append(record)
        except Exception as error:
            store.save_many(
                [
                    dict(
                        trial_id=a["trial_id"],
                        assignment=a,
                        kind=a["kind"],
                        terminal=True,
                        failure="execution_error",
                        error_type=type(error).__name__,
                        error=str(error),
                    )
                    for a in cohort
                ],
                attempt,
                f"{offset:07d}",
            )
            if checkpoint:
                checkpoint()
            raise
        store.save_many(records, attempt, f"{offset:07d}")
        completed.update({r["trial_id"]: r for r in records})
        if checkpoint:
            checkpoint()
        print(
            f"{len(completed)}/{len(assignments)} reasoning traces, batch={len(cohort)}, seconds={inferences[0]['batch']['wall_seconds']:.2f}",
            flush=True,
        )
    write_once(
        root / "profiles" / f"{attempt}.json",
        dict(completed=len(completed), assigned=len(assignments), seconds=time.monotonic() - started),
    )
    if checkpoint:
        checkpoint()
    return store.completed()


def render(manifest, records):
    if manifest["diagnostics"][0].get("task_revision") == "optional-information-v1":
        return task_module(manifest["diagnostics"][0]).render(manifest, records)
    correct = sum(r.get("correct") is True for r in records.values())
    valid = sum(r.get("valid") is True for r in records.values())
    text = f"# Reasoning-enabled information-state collection\n\nPhase: {manifest['phase']}. Complete {len(records)}/{len(manifest['diagnostics'])}; correct {correct}; valid {valid}.\n\n"
    if manifest["phase"] in ("validation", "development_validation"):
        text += task_module(manifest["diagnostics"][0]).render(manifest, records)
    else:
        text += "This engineering/development collection does not meet the full-validation sample requirement. Missing trace positions remain missing; reasoning text is not treated as a faithful account of internal computation.\n"
    return text
