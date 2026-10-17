"""Original-prompt interventions followed by newly generated reasoning.

Only the original prompt token IDs are replayed. Every generated reasoning and
answer token is new, and all hooks are removed before those tokens are sampled.
Initial label scores describe the next reasoning token, not the final answer.
"""

import hashlib
from pathlib import Path
import time
import uuid

import numpy as np

from .artifacts import RunStore, digest, read, write_once
from .causal_contrasts import source_descriptor
from .target_relevant_information import REVISION as TARGET_REVISION
from .replay_information import intervention
from .state_backend import StateBackend, verify_state_backend
from .thinking_information import generation_limit, inference_request, record_from_inference

CONDITIONING = "original_prompt_only; zero original generated tokens"


def generation_regime(config, prefixes):
    """Bind source and new generation caps, with only declared development changes."""
    limit = generation_limit(config)
    source_limits = sorted({generation_limit(p["assignment"]) for p in prefixes.values()})
    if not source_limits:
        raise ValueError("A natural generation regime requires source prompts")
    matches = source_limits == [limit]
    declaration = config.get("source_generation_regime_change")
    if not matches:
        if config.get("inference_scope") != "development" or config.get("phase") == "confirmation":
            raise ValueError(
                "Natural confirmation requires matching source and natural generation allowances"
            )
        if (
            not isinstance(declaration, dict)
            or set(declaration) != {"source_max_new_tokens", "natural_max_new_tokens", "reason"}
            or declaration["source_max_new_tokens"] != source_limits
            or declaration["natural_max_new_tokens"] != limit
            or not isinstance(declaration["reason"], str)
            or not declaration["reason"].strip()
        ):
            raise ValueError(
                "A changed source generation regime requires an explicit development declaration"
            )
    elif declaration is not None:
        raise ValueError("A source generation change declaration was supplied but the regimes match")
    return dict(
        source_max_new_tokens=source_limits,
        natural_max_new_tokens=limit,
        source_matches_natural=matches,
        development_change=declaration,
    )


def engine_hashes():
    root = Path(__file__).parent
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in (
            "state_backend.py",
            "model.py",
            "batched.py",
            "replay_information.py",
            "thinking_information.py",
            "identification.py",
            "natural_information.py",
        )
    }


def make_prefix(record, source_run, source_manifest_sha256):
    """Bind a prompt without selecting on the original completion's outcome."""
    inference = record["inference"]
    assignment = record["assignment"]
    if assignment["trial_id"] != record["trial_id"]:
        raise ValueError("Source record and assignment trial identities disagree")
    request = inference_request(assignment)
    if inference.get("enable_thinking") is not True or inference["messages"] != request["messages"]:
        raise ValueError("Natural reasoning requires the exact original thinking-enabled task prompt")
    if inference.get("sampling", {}).get("max_new_tokens") != generation_limit(assignment):
        raise ValueError("Actual source generation allowance differs from its assignment")
    tokens = list(inference["unpadded_input_ids"])
    if not tokens or any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("Invalid original prompt token IDs")
    meta = record["activation_artifact"]
    if len(meta["positions"]) != len(meta["token_positions"]) or len(set(meta["positions"])) != len(
        meta["positions"]
    ):
        raise ValueError("Invalid original capture positions")
    positions = dict(zip(meta["positions"], meta["token_positions"]))
    if positions.get("prefill") != len(tokens) - 1 or not 0 <= positions.get("record", -1) < len(tokens) - 1:
        raise ValueError("Original record/prefill positions must lie inside the prompt")
    prefix = dict(
        source_run=source_run,
        source_manifest_sha256=source_manifest_sha256,
        source_trial_id=record["trial_id"],
        source_record_sha256=digest(record),
        source_prompt_sha256=digest(inference["rendered_prompt"]),
        assignment=assignment,
        token_prefix=tokens,
        token_prefix_sha256=digest(tokens),
        prefix_positions=dict(record=positions["record"], final=len(tokens) - 1),
        labels=list(request["labels"]),
        seed=assignment["sampling_seed"] + 20000000,
        conditioning=CONDITIONING,
        original_generated_tokens_in_prefix=0,
    )
    if assignment.get("task_revision") == TARGET_REVISION:
        descriptor = source_descriptor(assignment)
        # Bind the designated objective separately from full-world information.
        prefix["response_objective"] = {
            key: descriptor[key]
            for key in (
                "response_objective_basis",
                "target_attribute",
                "target_property",
                "information_bits",
                "information_bits_semantics",
                "expected_world_information_gain_bits",
                "expected_target_information_gain_bits",
                "action_label_mapping",
            )
        }
    return prefix


def source_check(prefix, cache, artifact_root=Path("/artifacts"), expected_model=None):
    if Path(prefix["source_run"]).name != prefix["source_run"]:
        raise ValueError("The source run must be a single artifact directory name")
    root = Path(artifact_root) / prefix["source_run"]
    if root not in cache:
        manifest = read(root / "manifest.json")
        cache[root] = (manifest, RunStore(root, manifest).completed())
    manifest, records = cache[root]
    sha = digest(manifest)
    if expected_model is not None and manifest["model"] != expected_model:
        raise ValueError("Natural source and intervention models differ")
    if sha != prefix["source_manifest_sha256"]:
        raise ValueError("Natural source manifest mismatch")
    record = records[prefix["source_trial_id"]]
    source_assignments = {a["trial_id"]: a for a in manifest["assignments"] + manifest["diagnostics"]}
    if record["assignment"] != source_assignments[prefix["source_trial_id"]]:
        raise ValueError("Natural source record differs from the manifest assignment")
    if digest(record) != prefix["source_record_sha256"]:
        raise ValueError("Natural source record mismatch")
    if make_prefix(record, prefix["source_run"], sha) != prefix:
        raise ValueError("Natural prefix must contain exactly the original prompt and no generated tokens")


def representative_cohort_binding(manifest):
    """Bind the actual first worker cohort for the prospective condition-major guard."""
    policy = manifest.get("packing_policy", {})
    config = manifest.get("intervention_schedule", {})
    if policy.get("mode") != "condition_major_v1" and config.get("packing_mode") != "condition_major_v1":
        return None
    if (
        policy.get("mode") != "condition_major_v1"
        or config.get("packing_mode") != "condition_major_v1"
        or manifest.get("packing_policy_sha256") != digest(policy)
    ):
        raise ValueError("Condition-major startup requires its bound packing policy")
    size = manifest["execution"]["batch_size"]
    assignments = manifest["assignments"] + manifest["diagnostics"]
    cohort = assignments[:size]
    ids = [a["prefix_id"] for a in cohort]
    if len(cohort) != size or len(set(ids)) != size or size < 2:
        raise ValueError("Condition-major startup requires a full cohort of distinct source prefixes")
    if [a.get("source_row") for a in cohort] != list(range(size)):
        raise ValueError("Condition-major startup requires the actual fixed source-row order")
    requests = [manifest["prefixes"][key] for key in ids]
    if any(
        p["source_trial_id"] != key or p["assignment"]["trial_id"] != key for key, p in zip(ids, requests)
    ):
        raise ValueError("Condition-major startup sources must be actual source trials")
    patch_row = next((a for a in assignments if a["arm"] in ("direction", "random") and a["dose"] != 0), None)
    if patch_row is None:
        raise ValueError("Condition-major startup requires a scheduled nonzero perturbation")
    return dict(
        schema_version=1,
        manifest_sha256=digest(manifest),
        packing_policy=policy,
        packing_policy_sha256=digest(policy),
        batch_size=size,
        cohort_offset=0,
        cohort_assignment_sha256=digest(cohort),
        cohort_trial_ids=[a["trial_id"] for a in cohort],
        source_ids=ids,
        source_ids_sha256=digest(ids),
        source_requests_sha256=digest(requests),
        source_token_prefix_sha256=[p["token_prefix_sha256"] for p in requests],
        source_positions_sha256=[digest(p["prefix_positions"]) for p in requests],
        source_seeds=[p["seed"] for p in requests],
        perturbation_assignment=patch_row,
        perturbed_row=0,
        unperturbed_rows=list(range(1, size)),
        capture_layers=sorted(set(manifest.get("capture_layers", [])) | {patch_row["layer"]}),
        capture_positions=["record", "final"],
        max_new_tokens=16,
    )


def startup_patch_binding(patch):
    return {**patch, "direction": np.asarray(patch["direction"]).tolist()}


def _array_fingerprint(array):
    return dict(
        shape=list(array.shape),
        dtype=str(array.dtype),
        sha256=hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
        finite=bool(np.isfinite(array).all()),
    )


def _representative_forward(backend, requests, layers, patches):
    """Capture full initial vocabulary logits without changing StateBackend."""
    first_logits = []

    def capture(module, args, output):
        if not first_logits:
            first_logits.append(output.logits[:, -1].detach().float().cpu().numpy().copy())

    handle = backend.base.model.register_forward_hook(capture)
    try:
        rows, hidden = backend.forward(
            requests, layers=layers, positions=("record", "final"), interventions=patches, max_new_tokens=16
        )
    finally:
        handle.remove()
    if len(first_logits) != 1 or first_logits[0].shape[0] != len(requests):
        raise ValueError("Representative startup did not capture every prefill logit row")
    evidence = []
    for i, (request, row) in enumerate(zip(requests, rows)):
        evidence.append(
            dict(
                source_id=request["source_trial_id"],
                token_prefix_sha256=digest(row["unpadded_input_ids"]),
                positions_sha256=digest(row["unpadded_token_positions"]),
                input_layout_sha256=digest(
                    {key: row[key] for key in ("input_ids", "attention_mask", "position_ids")}
                ),
                prefill_width=row["prefill_width"],
                label_logits=row["label_logits"],
                prefill_logits=_array_fingerprint(first_logits[0][i]),
                captures=_array_fingerprint(hidden[i]),
                generated_token_ids=row["completion"]["generated_token_ids"],
                sampling=row["completion"]["sampling"],
                intervention_schedule=row["completion"]["intervention_schedule"],
                intervention=row["intervention"],
                hooks_remaining=row["hooks_remaining"],
            )
        )
    return rows, evidence


def representative_cohort_checks(binding, evidence):
    """Recompute exact row comparisons from compact startup evidence, also used by the auditor."""
    size = binding["batch_size"]
    baseline, sham, changed = (evidence[name] for name in ("baseline", "sham", "single_row_perturbation"))
    if any(len(rows) != size for rows in (baseline, sham, changed)):
        raise ValueError("Representative startup evidence must cover every cohort row in every call")
    all_rows = baseline + sham + changed
    expected_shape = [len(binding["capture_layers"]), 2]
    patch = binding["perturbation_assignment"]
    return dict(
        representative_sources_and_seeds=all(
            row["source_id"] == binding["source_ids"][i]
            and row["token_prefix_sha256"] == binding["source_token_prefix_sha256"][i]
            and row["positions_sha256"] == binding["source_positions_sha256"][i]
            and row["sampling"]
            == dict(seed=binding["source_seeds"][i], temperature=0.7, top_p=1.0, max_new_tokens=16)
            for rows in (baseline, sham, changed)
            for i, row in enumerate(rows)
        ),
        representative_layout_constant=len({row["prefill_width"] for row in all_rows}) == 1
        and all(
            a["input_layout_sha256"] == b["input_layout_sha256"] == c["input_layout_sha256"]
            for a, b, c in zip(baseline, sham, changed)
        ),
        representative_finite_logits_and_captures=all(
            row["prefill_logits"]["finite"]
            and row["captures"]["finite"]
            and len(row["prefill_logits"]["shape"]) == 1
            and row["captures"]["shape"][:2] == expected_shape
            for row in all_rows
        ),
        representative_sham_logits_equal=all(
            a["prefill_logits"] == b["prefill_logits"] and a["label_logits"] == b["label_logits"]
            for a, b in zip(baseline, sham)
        ),
        representative_sham_captures_equal=all(
            a["captures"] == b["captures"] for a, b in zip(baseline, sham)
        ),
        representative_sham_tokens_equal=all(
            a["generated_token_ids"] == b["generated_token_ids"] for a, b in zip(baseline, sham)
        ),
        representative_unperturbed_logits_equal=all(
            baseline[i]["prefill_logits"] == changed[i]["prefill_logits"]
            and baseline[i]["label_logits"] == changed[i]["label_logits"]
            for i in binding["unperturbed_rows"]
        ),
        representative_unperturbed_captures_equal=all(
            baseline[i]["captures"] == changed[i]["captures"] for i in binding["unperturbed_rows"]
        ),
        representative_unperturbed_tokens_equal=all(
            baseline[i]["generated_token_ids"] == changed[i]["generated_token_ids"]
            for i in binding["unperturbed_rows"]
        ),
        representative_baseline_unpatched=all(row["intervention"] is None for row in baseline),
        representative_sham_zero=all(
            row["intervention"]["delta_norm"] == 0
            and row["intervention"]["layer"] == patch["layer"]
            and row["intervention"]["position"] == patch["position"]
            for row in sham
        ),
        representative_one_row_perturbed=changed[0]["intervention"]["delta_norm"] > 0
        and changed[0]["intervention"]["requested_scalar_delta"] == patch["dose"]
        and changed[0]["intervention"]["layer"] == patch["layer"]
        and changed[0]["intervention"]["position"] == patch["position"]
        and all(changed[i]["intervention"] is None for i in binding["unperturbed_rows"]),
        representative_tokens_present=all(1 <= len(row["generated_token_ids"]) <= 16 for row in all_rows),
        representative_no_decoding_hooks=all(
            row["hooks_remaining"] == 0
            and row["intervention_schedule"] == "prefill_only; no active hooks during decoding"
            for row in all_rows
        ),
    )


def verify_natural_backend(model, requests, cohort_binding=None, perturbation=None):
    """Exercise real cached sampling with independent rows and prefill-only hooks."""
    selected = list(requests[:2])
    if len(selected) == 1:
        selected.append(selected[0])
    if not selected:
        raise ValueError("Natural backend audit requires at least one source prompt")
    base_audit = verify_state_backend(model, selected)
    backend = StateBackend(model)
    if cohort_binding is not None:
        if (
            len(requests) != cohort_binding["batch_size"]
            or digest(requests) != cohort_binding["source_requests_sha256"]
        ):
            raise ValueError("Representative startup requests differ from the actual frozen cohort")
        patch_row = cohort_binding["perturbation_assignment"]
        if perturbation is None or any(
            perturbation[key] != patch_row[key] for key in ("layer", "position", "dose")
        ):
            raise ValueError("Representative startup requires its scheduled nonzero perturbation")
        zero = {**perturbation, "dose": 0.0}
        baseline, base_evidence = _representative_forward(
            backend, requests, cohort_binding["capture_layers"], None
        )
        _, sham_evidence = _representative_forward(
            backend, requests, cohort_binding["capture_layers"], [zero] * len(requests)
        )
        _, changed_evidence = _representative_forward(
            backend, requests, cohort_binding["capture_layers"], [perturbation] + [None] * (len(requests) - 1)
        )
        evidence = dict(baseline=base_evidence, sham=sham_evidence, single_row_perturbation=changed_evidence)
        checks = {**base_audit["checks"], **representative_cohort_checks(cohort_binding, evidence)}
        return dict(
            passed=all(checks.values()),
            checks=checks,
            baseline=baseline,
            representative_cohort=cohort_binding,
            representative_perturbation=startup_patch_binding(perturbation),
            representative_cohort_evidence=evidence,
        )
    layer = len(model.layers) // 2
    direction = np.zeros(model.model.config.hidden_size)
    direction[0] = 1
    zero = dict(layer=layer, position="final", mode="add", dose=0.0, direction=direction)
    baseline, _ = backend.forward(selected, max_new_tokens=16)
    sham, _ = backend.forward(selected, interventions=[zero, zero], max_new_tokens=16)
    changed, _ = backend.forward(selected, interventions=[{**zero, "dose": 1.0}, None], max_new_tokens=16)
    checks = {
        **base_audit["checks"],
        "sampled_sham_tokens_equal": all(
            a["completion"]["generated_token_ids"] == b["completion"]["generated_token_ids"]
            for a, b in zip(baseline, sham)
        ),
        "other_row_sampled_tokens_unchanged": baseline[1]["completion"]["generated_token_ids"]
        == changed[1]["completion"]["generated_token_ids"],
        "paired_seeds_retained": all(
            a["completion"]["sampling"]["seed"] == request["seed"] for a, request in zip(sham, selected)
        ),
        "no_decoding_hooks": all(
            row["completion"]["intervention_schedule"] == "prefill_only; no active hooks during decoding"
            and row["hooks_remaining"] == 0
            for row in baseline + sham + changed
        ),
    }
    return dict(passed=all(checks.values()), checks=checks, baseline=baseline)


def _validate_manifest(manifest):
    if manifest["engine_sha256"] != engine_hashes():
        raise ValueError("Natural engine differs from its frozen implementation")
    if manifest["prefixes_sha256"] != digest(manifest["prefixes"]):
        raise ValueError("Natural prefix binding mismatch")
    assignments = manifest["assignments"] + manifest["diagnostics"]
    if not assignments or len({a["trial_id"] for a in assignments}) != len(assignments):
        raise ValueError("Require nonempty distinct intervention assignments")
    if manifest["assignment_sha256"] != digest(assignments):
        raise ValueError("Natural assignment binding mismatch")
    cap = manifest.get("max_new_tokens", 2048)
    if (
        type(cap) is not int
        or (manifest["phase"] != "engineering" and cap not in (2048, 4096))
        or not 1 <= cap <= 4096
    ):
        raise ValueError("Scientific natural reasoning requires the full 2048 or 4096 token allowance")
    if cap in (2048, 4096):
        config = manifest.get(
            "intervention_schedule",
            dict(
                max_new_tokens=cap, phase=manifest["phase"], inference_scope=manifest.get("inference_scope")
            ),
        )
        if generation_limit(config) != cap:
            raise ValueError("Natural generation allowance differs from the frozen schedule")
        regime = generation_regime(config, manifest["prefixes"])
        if "generation_regime" in manifest and manifest["generation_regime"] != regime:
            raise ValueError("Natural source generation regime binding mismatch")
    if manifest.get("greedy", False) and manifest["phase"] != "engineering":
        raise ValueError("Scientific natural reasoning uses the frozen sampled-generation policy")
    if type(manifest["execution"]["batch_size"]) is not int or manifest["execution"]["batch_size"] < 1:
        raise ValueError("Require a positive fixed cohort size")
    for a in assignments:
        prefix = manifest["prefixes"][a["prefix_id"]]
        if prefix["conditioning"] != CONDITIONING or prefix["original_generated_tokens_in_prefix"] != 0:
            raise ValueError("Natural experiments cannot condition on generated source reasoning")
        if set(prefix["prefix_positions"]) != {"record", "final"}:
            raise ValueError("Natural experiments permit only original record and prefill positions")
        if a.get("patch_mode", "add") != "add":
            raise ValueError("This prospective natural stage supports additive interventions only")
        if a["arm"] != "baseline" and a["position"] not in ("record", "final"):
            raise ValueError("Natural interventions must target record or final prompt token")
    return assignments


def execute(root, manifest, model, checkpoint=None, deadline=None, artifact_root=Path("/artifacts")):
    assignments = _validate_manifest(manifest)
    cap = manifest.get("max_new_tokens", 2048)
    root = Path(root)
    store = RunStore(root, manifest)
    completed = store.completed()
    attempt = uuid.uuid4().hex
    write_once(root / "environments" / f"{attempt}.json", model.metadata())
    prefixes = manifest["prefixes"]
    cache = {}
    for prefix in prefixes.values():
        source_check(prefix, cache, artifact_root, manifest["model"])
    del cache
    directions = None
    if manifest.get("direction_input"):
        meta = manifest["direction_input"]
        path = Path(artifact_root) / meta["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
            raise ValueError("Natural direction file hash mismatch")
        with np.load(path, allow_pickle=False) as z:
            directions = {key: z[key] for key in z.files}
    binding = representative_cohort_binding(manifest)
    if binding is None:
        audit = verify_natural_backend(model, list(prefixes.values())[:2])
    else:
        audit = verify_natural_backend(
            model,
            [prefixes[key] for key in binding["source_ids"]],
            cohort_binding=binding,
            perturbation=intervention(
                binding["perturbation_assignment"], directions, model.model.config.hidden_size
            ),
        )
    write_once(root / "natural_integrity" / f"{attempt}.json", audit)
    if not audit["passed"]:
        raise RuntimeError("Natural backend engineering audit failed")
    if checkpoint:
        checkpoint()
    backend = StateBackend(model)
    size = manifest["execution"]["batch_size"]
    started = time.monotonic()
    for offset in range(0, len(assignments), size):
        cohort = assignments[offset : offset + size]
        existing = [a["trial_id"] in completed for a in cohort]
        if all(existing):
            continue
        if any(existing):
            raise ValueError("Partial cohort would change the frozen batch; reconcile before resuming")
        if deadline is not None and time.time() >= deadline:
            break
        try:
            requests = [prefixes[a["prefix_id"]] for a in cohort]
            measurements, hidden = backend.forward(
                requests,
                layers=manifest.get("capture_layers", []),
                positions=("record", "final"),
                interventions=[intervention(a, directions, model.model.config.hidden_size) for a in cohort],
                max_new_tokens=cap,
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
                        positions=np.array(["record", "final"]),
                    )
                artifact = dict(
                    path=str(relative),
                    sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                    layers=manifest["capture_layers"],
                    positions=["record", "final"],
                    capture_semantics="Layer output captured before any patch at that layer; later layers may reflect earlier patches.",
                    max_new_tokens=cap,
                )
            rows = []
            for row, (a, prefix, measured) in enumerate(zip(cohort, requests, measurements)):
                measured["measurement"] = (
                    "original_prompt_next_token; not final-answer evidence or confidence"
                )
                scored = record_from_inference(prefix["assignment"], measured["completion"])
                result = dict(
                    trial_id=a["trial_id"],
                    terminal=True,
                    assignment=a,
                    kind=a["kind"],
                    source_assignment=prefix["assignment"],
                    source_trial_id=prefix["source_trial_id"],
                    token_prefix_sha256=prefix["token_prefix_sha256"],
                    conditioning=CONDITIONING,
                    measurement=measured,
                    closing_delimiter_count=measured["completion"]["raw"].count("</think>"),
                    cap_hit=not measured["completion"]["ended_with_eos"]
                    and len(measured["completion"]["generated_token_ids"]) == cap,
                    generation_max_new_tokens=cap,
                )
                result.update(
                    {
                        key: scored[key]
                        for key in (
                            "answer",
                            "parsed_answer",
                            "expected",
                            "valid",
                            "correct",
                            "task",
                            "thinking_closed",
                        )
                    }
                )
                if artifact:
                    result["activation_artifact"] = {**artifact, "row": row}
                rows.append(result)
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
        completed.update({row["trial_id"]: row for row in rows})
        if checkpoint:
            checkpoint()
        print(
            f"{len(completed)}/{len(assignments)} fresh-reasoning measurements; batch seconds {measurements[0]['batch_seconds']:.3f}",
            flush=True,
        )
    write_once(
        root / "profiles" / f"{attempt}.json",
        dict(
            completed=len(completed),
            assigned=len(assignments),
            seconds=time.monotonic() - started,
            conditioning=CONDITIONING,
        ),
    )
    if checkpoint:
        checkpoint()
    return completed


def render(manifest, records):
    return (
        f"# Original-prompt information intervention\n\nPhase: {manifest['phase']}. "
        f"Completed {len(records)}/{len(manifest['assignments']) + len(manifest['diagnostics'])}. "
        "All reasoning and answer tokens are generated anew. Interventions occur only during the original "
        "prompt prefill; record targets precede the downstream question, while final-prefill targets include it. "
        "Initial label scores are not final-answer evidence. Invalid and capped completions remain in the dataset.\n"
    )
