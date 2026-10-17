"""Immutable activation collections for the information-state study."""

from pathlib import Path
import hashlib
import time
import uuid
import numpy as np
from .artifacts import RunStore, write_once
from .information_state import inference_request, task, prompt
from .state_backend import StateBackend, verify_state_backend


def layer_indices(model):
    return [max(0, round(f * len(model.layers)) - 1) for f in (0.25, 0.5, 0.75, 0.9)]


def execute(root, manifest, model, checkpoint=None, deadline=None):
    if manifest["operation"] == "intervene":
        from .state_interventions import execute as execute_interventions

        return execute_interventions(root, manifest, model, checkpoint, deadline)
    if manifest["operation"] != "capture":
        raise ValueError("Unsupported state operation")
    root = Path(root)
    store = RunStore(root, manifest)
    completed = store.completed()
    attempt = uuid.uuid4().hex
    write_once(root / "environments" / f"{attempt}.json", model.metadata())
    assignments = manifest["assignments"] + manifest["diagnostics"]
    audit = verify_state_backend(model, [inference_request(a) for a in assignments[:2]])
    write_once(root / "state_integrity" / f"{attempt}.json", audit)
    if checkpoint:
        checkpoint()
    if not audit["passed"]:
        raise RuntimeError("State backend integrity failed")
    layers = layer_indices(model)
    backend = StateBackend(model)
    batch_size = manifest["execution"]["batch_size"]
    started = time.monotonic()
    for offset in range(0, len(assignments), batch_size):
        cohort = [a for a in assignments[offset : offset + batch_size] if a["trial_id"] not in completed]
        if not cohort:
            continue
        if deadline is not None and time.time() >= deadline:
            break
        try:
            results, hidden = backend.forward([inference_request(a) for a in cohort], layers)
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
                f"{offset:06d}",
            )
            if checkpoint:
                checkpoint()
            raise
        relative = Path("activations") / attempt / f"{offset:06d}.npz"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as f:
            np.savez_compressed(
                f,
                hidden=hidden,
                layers=np.array(layers),
                positions=np.array(["record", "final"]),
                trial_ids=np.array([a["trial_id"] for a in cohort]),
            )
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        cohort_records = []
        for row, (a, result) in enumerate(zip(cohort, results)):
            record = dict(
                trial_id=a["trial_id"],
                assignment=a,
                kind=a["kind"],
                terminal=True,
                task=task(a),
                expected=prompt(a)[2],
                measurement=result,
                activation_artifact=dict(
                    path=str(relative), sha256=sha, row=row, layers=layers, positions=["record", "final"]
                ),
            )
            cohort_records.append(record)
            completed[a["trial_id"]] = record
        store.save_many(cohort_records, attempt, f"{offset:06d}")
        if checkpoint:
            checkpoint()
        print(f"{len(completed)}/{len(assignments)} state capture, batch={len(cohort)}", flush=True)
    write_once(
        root / "profiles" / f"{attempt}.json",
        dict(
            completed=len(completed),
            assigned=len(assignments),
            seconds=time.monotonic() - started,
            layers=layers,
        ),
    )
    if checkpoint:
        checkpoint()
    return store.completed()


def render(manifest, records):
    return (
        "\n".join(
            [
                "# Information-state activation collection",
                "",
                f"Model: {manifest['model']['model_id']}",
                f"Complete: {len(records)}/{len(manifest['assignments']) + len(manifest['diagnostics'])}",
                "",
                "First-token measurements are not completed responses or validated confidence measures. Binary activation artifacts are independently hashed.",
            ]
        )
        + "\n"
    )
