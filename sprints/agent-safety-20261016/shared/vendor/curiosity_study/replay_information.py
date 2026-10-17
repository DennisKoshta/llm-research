"""Fixed-transcript causal measurements; exact token-prefix provenance is mandatory."""

from pathlib import Path
import hashlib
import time
import uuid
import numpy as np
from .artifacts import RunStore, digest, read, write_once
from .state_backend import StateBackend, verify_state_backend
from .thinking_information import task_module


def engine_hashes():
    root = Path(__file__).parent
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("state_backend.py", "model.py", "batched.py", "replay_information.py")
    }


def make_prefix(record, source_run, source_manifest_sha256):
    inf, meta = record["inference"], record["activation_artifact"]
    positions = dict(zip(meta["positions"], meta["token_positions"]))
    decision = positions["decision"]
    if decision < 0 or positions["think_end"] < 0 or not record["thinking_closed"]:
        return None
    all_ids = inf["unpadded_input_ids"] + inf["generated_token_ids"]
    tokens = all_ids[: decision + 1]
    if not len(inf["unpadded_input_ids"]) <= positions["think_end"] <= decision < len(all_ids) - 1:
        raise ValueError("Decision prefix does not exclude a generated answer token")
    labels = list(record["decision_label_logits"])
    return dict(
        source_run=source_run,
        source_manifest_sha256=source_manifest_sha256,
        source_trial_id=record["trial_id"],
        source_record_sha256=digest(record),
        source_activation=meta,
        assignment=record["assignment"],
        token_prefix=tokens,
        token_prefix_sha256=digest(tokens),
        prefix_positions=dict(record=positions["record"], think_end=positions["think_end"], final=decision),
        labels=labels,
        seed=record["assignment"]["sampling_seed"] + 10000000,
        reference_label_logits=record["decision_label_logits"],
        reference_label_probabilities_temperature_1=record["decision_label_probabilities_temperature_1"],
        original_final_answer=record["parsed_answer"],
        original_correct=record["correct"],
        original_valid=record["valid"],
        closing_delimiter_count=record["closing_delimiter_count"],
    )


def source_check(prefix, cache):
    root = Path("/artifacts") / prefix["source_run"]
    if root not in cache:
        m = read(root / "manifest.json")
        cache[root] = (digest(m), RunStore(root, m).completed())
    sha, records = cache[root]
    if (
        sha != prefix["source_manifest_sha256"]
        or digest(records[prefix["source_trial_id"]]) != prefix["source_record_sha256"]
    ):
        raise ValueError("Replay source record or manifest mismatch")
    reference = make_prefix(records[prefix["source_trial_id"]], prefix["source_run"], sha)
    if reference != prefix:
        raise ValueError("Replay prefix is not the exact original pre-answer prefix")


def execute(root, manifest, model, checkpoint=None, deadline=None):
    if manifest.get("engine_sha256", engine_hashes()) != engine_hashes():
        raise ValueError("Replay engine differs from the audited implementation")
    root = Path(root)
    store = RunStore(root, manifest)
    completed = store.completed()
    attempt = uuid.uuid4().hex
    write_once(root / "environments" / f"{attempt}.json", model.metadata())
    prefixes = manifest["prefixes"]
    cache = {}
    for prefix in prefixes.values():
        source_check(prefix, cache)
    del cache
    selected = list(prefixes.values())[:2]
    audit = verify_state_backend(model, selected)
    write_once(root / "replay_integrity" / f"{attempt}.json", audit)
    if not audit["passed"]:
        raise RuntimeError("Replay backend engineering audit failed")
    backend = StateBackend(model)
    assignments = manifest["assignments"] + manifest["diagnostics"]
    directions = None
    if manifest.get("direction_input"):
        meta = manifest["direction_input"]
        path = Path("/artifacts") / meta["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
            raise ValueError("Direction file hash mismatch")
        with np.load(path, allow_pickle=False) as z:
            directions = {k: z[k] for k in z.files}
    size = manifest["execution"]["batch_size"]
    started = time.monotonic()
    for offset in range(0, len(assignments), size):
        cohort = [a for a in assignments[offset : offset + size] if a["trial_id"] not in completed]
        if not cohort:
            continue
        if deadline is not None and time.time() >= deadline:
            break
        try:
            requests = [prefixes[a["prefix_id"]] for a in cohort]
            patches = [intervention(a, directions, model.model.config.hidden_size) for a in cohort]
            measurements, hidden = backend.forward(
                requests,
                layers=manifest.get("capture_layers", []),
                positions=("record", "think_end", "final"),
                interventions=patches,
                max_new_tokens=manifest["max_new_tokens"],
                greedy=manifest.get("greedy", False),
            )
            artifact = None
            if hidden is not None:
                relative = Path("activations") / attempt / f"{offset:07d}.npz"
                file = root / relative
                file.parent.mkdir(parents=True, exist_ok=True)
                with file.open("xb") as f:
                    np.savez_compressed(
                        f,
                        hidden=hidden,
                        trial_ids=np.array([a["trial_id"] for a in cohort]),
                        layers=np.array(manifest["capture_layers"]),
                        positions=np.array(["record", "think_end", "final"]),
                    )
                artifact = dict(
                    path=str(relative),
                    sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                    layers=manifest["capture_layers"],
                    positions=["record", "think_end", "final"],
                )
            rows = []
            for row, (a, prefix, measured) in enumerate(zip(cohort, requests, measurements)):
                r = dict(
                    trial_id=a["trial_id"],
                    terminal=True,
                    assignment=a,
                    kind=a["kind"],
                    source_assignment=prefix["assignment"],
                    source_trial_id=prefix["source_trial_id"],
                    token_prefix_sha256=prefix["token_prefix_sha256"],
                    measurement=measured,
                    reference_label_logits=prefix["reference_label_logits"],
                    reference_label_probabilities_temperature_1=prefix[
                        "reference_label_probabilities_temperature_1"
                    ],
                )
                if artifact:
                    r["activation_artifact"] = {**artifact, "row": row}
                if manifest["max_new_tokens"]:
                    scored = task_module(prefix["assignment"]).record_from_inference(
                        prefix["assignment"], measured["completion"]
                    )
                    r.update(
                        {
                            key: scored[key]
                            for key in ("answer", "parsed_answer", "expected", "valid", "correct", "task")
                        }
                    )
                rows.append(r)
        except Exception as error:
            store.save_many(
                [
                    dict(
                        trial_id=a["trial_id"],
                        terminal=True,
                        assignment=a,
                        kind=a["kind"],
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
        store.save_many(rows, attempt, f"{offset:07d}")
        completed.update({r["trial_id"]: r for r in rows})
        if checkpoint:
            checkpoint()
        print(
            f"{len(completed)}/{len(assignments)} fixed-prefix measurements; batch seconds {measurements[0]['batch_seconds']:.3f}",
            flush=True,
        )
    write_once(
        root / "profiles" / f"{attempt}.json",
        dict(completed=len(completed), assigned=len(assignments), seconds=time.monotonic() - started),
    )
    return completed


def intervention(assignment, directions, dimension):
    arm = assignment["arm"]
    if arm == "baseline":
        return None
    patch = dict(
        layer=assignment["layer"], position=assignment["position"], mode="add", dose=float(assignment["dose"])
    )
    if arm == "sham":
        unit = np.zeros(dimension, dtype=np.float32)
        unit[0] = 1
        patch["dose"] = 0.0
    elif arm == "random":
        rng = np.random.default_rng(assignment["random_seed"])
        unit = rng.normal(size=dimension)
        unit /= np.linalg.norm(unit)
    elif arm == "direction":
        unit = directions[assignment.get("array", "unit")][
            assignment["source_index"], assignment["layer_index"], assignment["position_index"]
        ]
        unit = unit / np.linalg.norm(unit)
    else:
        raise ValueError("Unknown intervention arm")
    patch["direction"] = unit
    if assignment.get("patch_mode", "add") in ("project", "matched_add"):
        patch["mode"] = assignment["patch_mode"]
        patch["target"] = float(assignment["projection_target"])
        if patch["mode"] == "matched_add":
            reference = directions[assignment.get("reference_array", "unit")][
                assignment["reference_source_index"], assignment["layer_index"], assignment["position_index"]
            ]
            patch["reference_direction"] = reference / np.linalg.norm(reference)
    elif assignment.get("patch_mode", "add") != "add":
        raise ValueError("Unsupported fixed-prefix intervention mode")
    return patch


def render(manifest, records):
    return f"# Fixed transcript information experiment\n\nPhase: {manifest['phase']}. Completed {len(records)}/{len(manifest['assignments']) + len(manifest['diagnostics'])}. All visible prefix tokens are held fixed across arms. These interventions are conditional on the original generated transcript.\n"
