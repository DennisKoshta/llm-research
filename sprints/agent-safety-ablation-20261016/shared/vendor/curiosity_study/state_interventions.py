"""Frozen prefill intervention grids with complete short responses and action delivery."""

from pathlib import Path
import hashlib
import itertools
import random
import time
import uuid
import numpy as np
from .artifacts import RunStore, write_once
from .information_state import inference_request, record_from_inference, task
from .state_backend import StateBackend, verify_state_backend


def make_patch(condition, directions, random_seed):
    if condition["name"] == "baseline":
        return None
    li = list(directions["layers"]).index(condition["layer"])
    pi = list(directions["positions"]).index(condition["position"])
    unit = directions["unit"][0, li, pi]
    scale = directions["contrast_norm"][0, li, pi]
    kind = condition["direction"]
    if kind == "gain":
        vector = unit
    elif kind in ("known", "query"):
        vector = directions["controls"][0, ("known", "query").index(kind), li, pi]
    elif kind.startswith("random_"):
        index = int(kind.split("_")[1])
        rng = np.random.default_rng(random_seed + index * 10000 + condition["layer"] * 10 + pi)
        vector = rng.normal(size=len(unit))
        vector -= np.dot(vector, unit) * unit
        vector /= np.linalg.norm(vector)
    else:
        raise ValueError("Unknown direction")
    return dict(
        layer=condition["layer"],
        position=condition["position"],
        mode="add",
        direction=vector.tolist(),
        dose=float(condition["dose"] * scale),
    )


def delivery(a, record):
    t = task(a)
    services = list(itertools.permutations(("report", "pattern", "stop")))[a["index"] % 6]
    service = dict(zip("ABC", services)).get(record["parsed_answer"]) if record["valid"] else None
    if service == "report":
        text = f"The {t['attributes'][a['query']]} is {t['candidates'][t['truth']][a['query']]}."
        after = t["uncertainty_bits"] - t["information_bits"]
    elif service == "pattern":
        rng = random.Random(f"state-pattern:{a['seed']}:{a['family']}:{a['index']}")
        text = "\n".join("".join(rng.choice(".*") for _ in range(4)) for _ in range(4))
        after = t["uncertainty_bits"]
    else:
        text = "No observation delivered."
        after = t["uncertainty_bits"]
    return dict(
        service=service or "invalid",
        response=text + "\nTrial ended.",
        done=True,
        uncertainty_before=t["uncertainty_bits"],
        uncertainty_after=after,
        offered_information_bits=t["information_bits"],
        failure=None if service else "invalid_output",
    )


def execute(root, manifest, model, checkpoint=None, deadline=None):
    root = Path(root)
    store = RunStore(root, manifest)
    completed = store.completed()
    attempt = uuid.uuid4().hex
    write_once(root / "environments" / f"{attempt}.json", model.metadata())
    direction_path = Path("/artifacts") / manifest["direction_input"]["path"]
    if manifest["direction_input"]["source_model"] != manifest["model"]:
        raise ValueError("Direction/model identity mismatch")
    if hashlib.sha256(direction_path.read_bytes()).hexdigest() != manifest["direction_input"]["sha256"]:
        raise ValueError("Direction input hash mismatch")
    with np.load(direction_path, allow_pickle=False) as z:
        directions = {k: z[k] for k in z.files}
    assignments = manifest["assignments"]
    audit = verify_state_backend(model, [inference_request(a) for a in assignments[:2]])
    write_once(root / "state_integrity" / f"{attempt}.json", audit)
    if not audit["passed"]:
        raise RuntimeError("State backend integrity failed")
    conditions = {c["name"]: c for c in manifest["conditions"]}
    patches = {
        name: make_patch(c, directions, manifest["random_direction_seed"]) for name, c in conditions.items()
    }
    backend = StateBackend(model)
    size = manifest["execution"]["batch_size"]
    started = time.monotonic()
    for offset in range(0, len(assignments), size):
        cohort = [a for a in assignments[offset : offset + size] if a["trial_id"] not in completed]
        if not cohort:
            continue
        if deadline is not None and time.time() >= deadline:
            break
        try:
            measurements, _ = backend.forward(
                [inference_request(a) for a in cohort],
                interventions=[patches[a["treatment"]] for a in cohort],
                max_new_tokens=8,
            )
            records = []
            for a, measurement in zip(cohort, measurements):
                record = record_from_inference(a, measurement["completion"])
                record["measurement"] = measurement
                record["treatment"] = conditions[a["treatment"]]
                record["cohort_ids"] = [b["trial_id"] for b in cohort]
                if a["kind"] == "choice":
                    record["ledger"] = [delivery(a, record)]
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
        print(f"{len(completed)}/{len(assignments)} intervention readouts, batch={len(cohort)}", flush=True)
    write_once(
        root / "profiles" / f"{attempt}.json",
        dict(completed=len(completed), assigned=len(assignments), seconds=time.monotonic() - started),
    )
    if checkpoint:
        checkpoint()
    return store.completed()
