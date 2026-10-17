"""Raw evidence audit for the fixed published-profile setup and benchmark.

No inference or outcome-based selection. Token scoring and sampled BF16
reconstruction are independent of the producer's scorer and numerical checks.
"""

import hashlib
import math
from pathlib import Path
import re

import numpy as np

from curiosity_study import pain_profile_design as design
from curiosity_study.artifacts import digest, read

END_TAGS = ("<|im_end|>", "<|eot_id|>", "<end_of_turn>", "</s>", "<eos>", "<|end|>", "<|endoftext|>")
CAP = 8
RUNTIME_COMPARISON_FIELDS = (
    "base", "adapter_revision", "adapter_archive_sha256", "adapter_files", "vector_receipt",
    "environment", "cuda_build", "base_dtype", "attention", "lora_layers", "adapter_merged",
    "adapter_dtypes", "python_version", "machine", "gpu_compute_capability", "numerical_flags",
    "modal_image_id", "source_sha256", "source_archive_sha256", "execution_files_sha256",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def array_binding(value):
    value = np.ascontiguousarray(value)
    return dict(dtype=str(value.dtype), shape=list(value.shape),
                byte_sha256=hashlib.sha256(value.tobytes()).hexdigest())


def tensor_binding(value):
    import torch

    value = value.detach().contiguous().cpu()
    return dict(dtype=str(value.dtype), shape=list(value.shape),
                byte_sha256=hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest())


def score_text(text, button_names, target_name):
    """Literal source grammar, including its deliberately permissive prefix match."""
    end = r"(?:(?:" + "|".join(re.escape(tag) for tag in END_TAGS) + r")\s*)+$"
    cleaned = re.sub(end, "", text).strip()
    normalized = cleaned.strip().strip("\"'").lower().rstrip(".!?,;:")
    picked = next((name for name in sorted(button_names, key=len, reverse=True)
                   if normalized.startswith(name.lower())), None)
    return dict(category="malformed" if picked is None else "target" if picked == target_name else "other",
                valid=picked is not None, picked=picked, cleaned_answer=cleaned)


def _close(left, right):
    return math.isfinite(float(left)) and math.isfinite(float(right)) and math.isclose(
        float(left), float(right), rel_tol=1e-5, abs_tol=1e-6)


def _geometry(frame, arrays, items, direction, coefficients, policy, mode, used_arrays):
    import torch

    width = len(frame["input_ids"][0])
    prefill = frame["phase"] == "prefill"
    require(frame["activation_dtypes"] == dict(before="torch.bfloat16", after="torch.bfloat16"),
            "Observed residual stream is not BF16")
    require(len(frame["geometry"]) == len(items), "Missing physical-row geometry")
    for row_index, (row, item, coefficient) in enumerate(zip(frame["geometry"], items, coefficients, strict=True)):
        expected_active = ((len(item["prompt_ids"]) if policy == "full_prefix" else 1) if prefill else 1) if coefficient else 0
        padding = width - len(item["prompt_ids"]) if prefill else 0
        require(row["trial_id"] == item["trial_id"] and row["coefficient"] == coefficient
                and row["active_positions"] == expected_active
                and row["unmodified_positions"] == width - expected_active
                and row["padding_positions"] == padding,
                "Geometry row, mask position count or coefficient changed")
        require(all(type(row.get(key)) is bool for key in (
            "finite", "exact_bf16_arithmetic", "unmodified_positions_identical",
            "padding_identical", "all_requested_positions_engaged")), "Malformed geometry flags")
        require(valid_sha(row["before_sha256"]) and valid_sha(row["after_sha256"]), "Missing residual byte fingerprints")
        desired = (torch.tensor(coefficient, dtype=torch.bfloat16) * direction).float()
        require(_close(row["requested_delta_norm_bf16"], float(desired.norm())), "Requested BF16 vector norm differs")
        if row["finite"]:
            require(all(isinstance(row.get(k), (float, int)) and not isinstance(row[k], bool)
                        and math.isfinite(row[k]) for k in (
                            "max_abs_error", "achieved_delta_norm_min", "achieved_delta_norm_max", "achieved_direction_dot_min")),
                    "Finite geometry has unavailable/nonfinite scalar evidence")
            require(row["exact_bf16_arithmetic"] == (row["max_abs_error"] == 0), "Arithmetic flag contradicts observed error")
            require(0 <= row["achieved_delta_norm_min"] <= row["achieved_delta_norm_max"], "Invalid achieved norm range")
            expected_engaged = float(desired.norm()) == 0 or row["achieved_delta_norm_min"] > 0
            require(row["all_requested_positions_engaged"] == expected_engaged, "Engagement flag contradicts observed norms")
            if expected_active == 0:
                require(row["achieved_delta_norm_min"] == row["achieved_delta_norm_max"] == row["achieved_direction_dot_min"] == 0,
                        "Inactive row claims a scheduled perturbation")
                require(row["unmodified_positions_identical"] == (row["before_sha256"] == row["after_sha256"]),
                        "Untouched-row identity flag differs from raw byte fingerprints")
        saved = row_index in ({0, len(items) - 1} if mode == "compatibility" else {0})
        positions = list(range(width)) if mode == "compatibility" else sorted({
            width - 1, max(0, width - len(item["prompt_ids"])) if prefill else 0,
        })
        for label in ("before", "after"):
            require(("saved_" + label in row) == saved, "Geometry sampling inventory changed")
        if not saved:
            continue
        samples = {}
        for label in ("before", "after"):
            binding = row["saved_" + label]
            key = f"f{frame['forward_index']:02d}_r{row_index:02d}_{label}"
            require(binding["key"] == key and binding["positions"] == positions and key in arrays,
                    "Saved residual positions or array key differ")
            array = arrays[key]
            require(array.dtype == np.float32 and array.shape == (len(positions), direction.numel())
                    and binding == dict(key=key, positions=positions, **array_binding(array)), "Saved residual array identity differs")
            used_arrays.add(key)
            value = torch.from_numpy(array.copy()).to(torch.bfloat16)
            require(np.array_equal(value.float().numpy(), array), "Saved values are not exact BF16 representations")
            samples[label] = value
            if len(positions) == width:
                require(row[label + "_sha256"] == digest(tensor_binding(value)), "Full saved residual fingerprint differs")
        before, after = samples["before"], samples["after"]
        mask = torch.tensor([
            coefficient != 0 and (not prefill or (position >= padding if policy == "full_prefix" else position == width - 1))
            for position in positions
        ], dtype=torch.bfloat16)
        delta = mask[:, None] * torch.tensor(coefficient, dtype=torch.bfloat16) * direction[None, :]
        expected = before if not any(coefficients) else before + delta
        exact = torch.equal(after.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))
        finite = bool(torch.isfinite(before).all() and torch.isfinite(after).all())
        require(not row["finite"] or finite, "Saved sample contradicts finite arithmetic claim")
        require(not row["exact_bf16_arithmetic"] or exact, "Saved sample contradicts exact BF16 arithmetic claim")
        inactive = mask == 0
        untouched = torch.equal(before[inactive].contiguous().view(torch.uint8), after[inactive].contiguous().view(torch.uint8))
        require(not row["unmodified_positions_identical"] or untouched, "Untreated saved positions changed")
        if finite:
            achieved = (after.float() - before.float())[mask.bool()]
            norms = achieved.norm(dim=-1)
            if norms.numel():
                low, high = float(norms.min()), float(norms.max())
                require(low + 1e-5 >= row["achieved_delta_norm_min"] and high - 1e-5 <= row["achieved_delta_norm_max"],
                        "Saved sample is outside observed norm range")
                if len(positions) == width:
                    require(_close(low, row["achieved_delta_norm_min"]) and _close(high, row["achieved_delta_norm_max"])
                            and _close(float((achieved @ direction.float()).min()), row["achieved_direction_dot_min"]),
                            "Scalar geometry differs from complete saved row")


def recompute_call(result, arrays, items, *, direction, pain_unit, tokenizer, vocabulary_size,
                   coefficient, prefill_policy, eos_ids, pad_id, mode,
                   steering_enabled=True, one_row_only=None, expected_fault=None,
                   expected_device="cuda"):
    """Audit one actual call; CPU identity is permitted only by explicit test input."""
    import torch

    report, records = result["evidence"], result["records"]
    direction = torch.as_tensor(direction, dtype=torch.bfloat16).cpu()
    pain_unit = torch.as_tensor(pain_unit, dtype=torch.float32).cpu()
    n, width = len(items), max(len(i["prompt_ids"]) for i in items)
    coefficients = [coefficient if one_row_only is None or index == one_row_only else 0 for index in range(n)]
    require(report["schema_version"] == 1 and report["source_entrypoint_sha256"] == design.SOURCE_FILE_SHA256["src/run_profile.py"]
            and report["device_type"] == expected_device and report["cpu_synthetic_execution"] is (expected_device == "cpu")
            and report["steer_layer"] == 38 and report["monitor_layer"] == 61
            and report["coefficient"] == coefficient and report["prefill_policy"] == prefill_policy
            and report["evidence_mode"] == mode and report["steering_enabled"] is steering_enabled
            and report["one_row_only"] == one_row_only and report["inject_failure_at"] == expected_fault
            and report["applied_coefficients"] == coefficients
            and report["direction"] == tensor_binding(direction) and report["pain_unit"] == tensor_binding(pain_unit)
            and report["temperature"] == 0.7 and report["top_p"] == 0.95 and report["max_new_tokens"] == CAP
            and report["eos_token_ids"] == sorted(eos_ids) and report["pad_token_id"] == pad_id
            and report["trial_ids"] == [i["trial_id"] for i in items] and report["items_sha256"] == digest(items)
            and report["scientific_admission"] is False, "Call numerical policy or original input binding changed")
    require(report["arrays"] == {key: array_binding(value) for key, value in arrays.items()}, "Residual array inventory differs")
    matching_fault = expected_fault is not None and report["error"] == dict(
        type="RuntimeError", message=f"controlled compatibility fault: {expected_fault}")
    if report["error"] is not None and not matching_fault:
        # A model/observer exception can interrupt a frame before its projections,
        # geometry or sampling exist. Preserve its artifacts, never infer choices
        # or certify that incomplete numeric evidence as a successful call.
        require(records == [] and report["complete"] is False and report["integrity_passed"] is False
                and isinstance(report["error"], dict)
                and isinstance(report["error"].get("type"), str)
                and isinstance(report["error"].get("message"), str)
                and isinstance(report.get("hooks_before"), dict)
                and isinstance(report.get("hooks_after"), dict), "Malformed terminal execution failure")
        return dict(complete=False, passed=False, expected_fault_observed=False,
                    checks=dict(terminal_execution_error=True,
                                hooks_clean=report["hooks_before"] == report["hooks_after"]),
                    outcomes=[], active_decode_observed=False, forwards=len(report["forwards"]),
                    partial_numerical_evidence_unavailable=True)
    initial_ids = [[pad_id] * (width - len(i["prompt_ids"])) + i["prompt_ids"] for i in items]
    initial_mask = [[0] * (width - len(i["prompt_ids"])) + [1] * len(i["prompt_ids"]) for i in items]
    initial_pos = [[0] * (width - len(i["prompt_ids"])) + list(range(len(i["prompt_ids"]))) for i in items]
    steering_mask = [[float(coefficients[r] != 0 and (m == 1 if prefill_policy == "full_prefix" else c == width - 1))
                      for c, m in enumerate(initial_mask[r])] for r in range(n)]
    alive, generated, rng_after, used_arrays = [True] * n, [[] for _ in items], None, set()
    next_ids = None
    frames = report["forwards"]
    require(len(frames) <= CAP, "Too many source forwards")
    for index, frame in enumerate(frames):
        phase = "prefill" if index == 0 else "decode"
        positions = initial_pos if index == 0 else [[len(i["prompt_ids"]) + index - 1] for i in items]
        attention = [row + [1] * index for row in initial_mask]
        require(frame["forward_index"] == index and frame["phase"] == phase
                and frame["input_ids"] == (initial_ids if index == 0 else [[token] for token in next_ids])
                and frame["attention_mask"] == attention and frame["position_ids"] == positions
                and frame["steering_mask"] == (steering_mask if index == 0 else None)
                and frame["cache_length_before"] == (0 if index == 0 else width + index - 1)
                and frame["cache_length_after"] == width + index,
                "Actual input/mask/cache/position timeline differs from the source schedule")
        require(frame["vocabulary_size"] == vocabulary_size
                and len(frame["full_fp32_logits_sha256"]) == n
                and all(valid_sha(s) for s in frame["full_fp32_logits_sha256"])
                and type(frame["finite_logits"]) is bool,
                "Missing full-vocabulary fingerprint or actual vocabulary differs")
        require(frame["steering_hook_calls"] == int(steering_enabled)
                and frame["read_only_projection_calls"] == int(not steering_enabled)
                and all(frame[k] == 1 for k in ("before_observer_calls", "after_observer_calls", "monitor_hook_calls")),
                "Missing or duplicate observed hooks")
        _geometry(frame, arrays, items, direction, coefficients, prefill_policy, mode, used_arrays)
        for key in ("prepatch_pain_projection", "post_layer61_pain_projection"):
            require(len(frame[key]) == n and all(isinstance(x, (float, int)) and math.isfinite(x) for x in frame[key]),
                    "Missing or nonfinite source pain-unit projection")
        sample = frame.get("sampling")
        if sample is None:
            require(report["error"] is not None and index == len(frames) - 1, "Completed forward lacks its sampling evidence")
            continue
        require(sample["step"] == index and sample["alive_before"] == alive
                and sample["uniform_draws_per_row"] == [1] * n,
                "Source active mask or finished-row RNG advancement changed")
        for key in ("rng_state_before_sha256", "rng_state_after_sha256"):
            require(len(sample[key]) == n and all(valid_sha(s) for s in sample[key]), "Missing row RNG fingerprints")
        if rng_after is not None:
            require(sample["rng_state_before_sha256"] == rng_after, "Per-row RNG chain breaks")
        rng_after = sample["rng_state_after_sha256"]
        require(all(a != b for a, b in zip(sample["rng_state_before_sha256"], rng_after, strict=True)), "RNG did not advance on every row")
        require(len(sample["uniforms"]) == n and all(type(u) is float and math.isfinite(u) and 0 <= u < 1 for u in sample["uniforms"]),
                "Invalid actual source uniforms")
        sampled = sample["sampled_before_alive_mask"]
        require(len(sampled) == n and all(type(t) is int and 0 <= t < vocabulary_size for t in sampled), "Invalid sampled token IDs")
        next_ids = [sampled[r] if alive[r] else pad_id for r in range(n)]
        require(sample["selected_token_ids"] == next_ids, "Inactive sampled tokens were not replaced with source padding")
        for row, token in enumerate(next_ids):
            if alive[row]:
                generated[row].append(token)
        alive = [alive[r] and next_ids[r] not in eos_ids and len(generated[r]) < CAP for r in range(n)]
        require(sample["alive_after"] == alive, "Source EOS/cap termination differs")
        require(index == len(frames) - 1 or any(alive), "A terminal cohort was forwarded again")
    require(set(arrays) == used_arrays, "Unconsumed or missing residual arrays")
    error = report["error"]
    recognized_fault = expected_fault is not None and error == dict(type="RuntimeError", message=f"controlled compatibility fault: {expected_fault}")
    if expected_fault is not None:
        count = 1 if expected_fault == "after_prefill" else 2
        require(recognized_fault and len(frames) == count and "sampling" not in frames[-1]
                and report["incomplete_forward"] is None and report["attempted_forwards"] == count,
                "Expected finite compatibility fault was not actually exercised")
    elif error is None:
        require(bool(frames) and not any(alive) and report["incomplete_forward"] is None
                and report["attempted_forwards"] == len(frames), "Completed call has unfinished sampling/forward work")
    else:
        require(isinstance(error, dict) and isinstance(error.get("type"), str) and isinstance(error.get("message"), str),
                "Malformed terminal error evidence")
        require(report["attempted_forwards"] in (len(frames), len(frames) + 1), "Failed forward count differs")
    complete = error is None and len(records) == n
    require(report["complete"] is complete and (complete or records == []), "Partial success records or complete flag differ")
    scored = []
    if complete:
        for row, item, tokens in zip(records, items, generated, strict=True):
            raw = tokenizer.decode(tokens, skip_special_tokens=False)
            require(row["trial_id"] == item["trial_id"] and row["answer_token_ids"] == tokens
                    and row["prompt_token_ids"] == item["prompt_ids"] and row["text"] == raw
                    and row["prompt_tokens"] == len(item["prompt_ids"]) and row["padded_prompt_tokens"] == width
                    and row["prefill_attended"] is True and row["prefill_position_id"] == len(item["prompt_ids"]) - 1
                    and row["prefill_generated_tokens"] == 0
                    and row["button_names"] == item["button_names"] and row["target_name"] == item["target_name"],
                    "Saved response text/tokens or source prefix differs")
            eos = bool(tokens and tokens[-1] in eos_ids)
            cap = bool(len(tokens) == CAP and not eos)
            require(row["ended_with_eos"] is eos and row["cap_hit"] is cap, "Saved cap/EOS flags differ")
            require(row["n_fwd"] == len(tokens), "Source active projection denominator differs")
            for key, frame_key in (("mean_proj", "prepatch_pain_projection"), ("mean_proj_monitor", "post_layer61_pain_projection")):
                values = [f[frame_key][len(scored)] for f in frames[:len(tokens)]]
                require(_close(row[key], float(np.asarray(values, dtype=np.float32).mean())), "Source projection average differs")
            require(row["prefill_proj_monitor"] == frames[0]["post_layer61_pain_projection"][len(scored)]
                    and all(isinstance(row[k], (float, int)) and math.isfinite(row[k]) and 0 <= row[k] <= 1 for k in ("p_x", "p_y")),
                    "Source initial projection or name-probability metadata differs")
            scored.append(dict(trial_id=item["trial_id"], status="completed", cap_hit=cap,
                               **score_text(raw, item["button_names"], item["target_name"])))
    all_geometry = [row for f in frames for row in f["geometry"]]
    hook_clean = report.get("hooks_before") == report.get("hooks_after")
    require(isinstance(report.get("hooks_before"), dict) and isinstance(report.get("hooks_after"), dict), "Missing actual hook inventories")
    checks = dict(
        hooks_clean=hook_clean,
        all_observed_geometry_finite=bool(all_geometry) and all(r["finite"] for r in all_geometry),
        all_observed_arithmetic_exact=bool(all_geometry) and all(r["exact_bf16_arithmetic"] for r in all_geometry),
        all_unmodified_positions_identical=bool(all_geometry) and all(r["unmodified_positions_identical"] for r in all_geometry),
        all_padding_identical=bool(all_geometry) and all(r["padding_identical"] for r in all_geometry),
        all_observed_logits_finite=bool(frames) and all(f["finite_logits"] for f in frames),
        all_requested_positions_engaged=bool(all_geometry) and all(r["all_requested_positions_engaged"] for r in all_geometry),
    )
    passed = complete and all(checks.values())
    require(report["checks"] == checks and report["integrity_passed"] is passed, "Saved numerical checks differ from raw evidence")
    return dict(complete=complete, passed=passed, expected_fault_observed=recognized_fault,
                checks=checks, outcomes=scored, active_decode_observed=any(
                    index and f.get("sampling", {}).get("alive_before") and any(f["sampling"]["alive_before"])
                    for index, f in enumerate(frames)), forwards=len(frames))


def compare_rows(left, right, row_indices):
    """Exact full-vocabulary/token/RNG identity while each compared row is active."""
    a, b = left["evidence"], right["evidence"]
    require(a["complete"] and b["complete"], "Identity comparison requires complete calls")
    for row in row_indices:
        x, y = left["records"][row], right["records"][row]
        for key in ("answer_token_ids", "text", "prompt_token_ids", "prompt_tokens", "padded_prompt_tokens",
                    "prefill_position_id", "prefill_proj_monitor", "p_x", "p_y", "ended_with_eos", "cap_hit"):
            require(x[key] == y[key], "Paired sampled tokens or terminal status differ")
        for index in range(len(x["answer_token_ids"])):
            f, g = a["forwards"][index], b["forwards"][index]
            for key in ("input_ids", "attention_mask", "position_ids", "full_fp32_logits_sha256"):
                require(f[key][row] == g[key][row], "Paired active full-vocabulary/capture/input identity differs")
            for key in ("rng_state_before_sha256", "rng_state_after_sha256", "uniforms", "alive_before", "alive_after", "selected_token_ids"):
                require(f["sampling"][key][row] == g["sampling"][key][row], "Paired active RNG/sampling identity differs")
    return True


def source_provenance(run, manifest):
    """Replay frozen metadata/tokenization/vectors; never load model weights."""
    import tarfile
    import tempfile
    import prepare_pain_profile_control as preparation
    from analyze_pain_profile_control import ANALYSIS_METADATA
    from curiosity_study import pain_profile_assets as assets
    from curiosity_study import pain_profile_execution as execution
    from curiosity_study import pain_profile_resources as resources

    run = Path(run).resolve()
    frozen = read(run / "freeze.json")
    require(digest(manifest) == frozen["manifest_sha256"]
            and manifest["source_sha256"] == frozen["source_sha256"]
            and sha(run / "source.tar.gz") == frozen["archive_sha256"]
            and frozen["documents"] == manifest["documents"], "Frozen campaign binding differs")
    inventory, sources, payloads = {}, {}, {}
    with tarfile.open(run / "source.tar.gz") as archive:
        for member in archive.getmembers():
            path = Path(member.name)
            require(member.isfile() and not path.is_absolute() and ".." not in path.parts
                    and str(path) == member.name and member.name not in inventory, "Unsafe or duplicate archive entry")
            data = archive.extractfile(member).read()
            inventory[member.name] = hashlib.sha256(data).hexdigest()
            if member.name.startswith("src/curiosity_study/") and member.name.endswith(".py"):
                sources[member.name.removeprefix("src/curiosity_study/")] = data.decode()
            if member.name in manifest["documents"]:
                payloads[member.name] = data
    require(digest(sources) == manifest["source_sha256"] and set(payloads) == set(manifest["documents"])
            and all(inventory[name] == expected for name, expected in manifest["documents"].items()),
            "Archive source/document inventory differs")
    project = Path(__file__).resolve().parents[1]
    numerical = [str(p.relative_to(project)) for p in (project / "src/curiosity_study").glob("pain_profile_*.py")]
    numerical += ["scripts/" + name for name in (
        "audit_pain_profile_control.py", "analyze_pain_profile_control.py", "prepare_pain_profile_control.py")]
    require(all(name in inventory and sha(project / name) == inventory[name] for name in numerical),
            "Current audited pain helpers differ from frozen bytes")
    with tempfile.TemporaryDirectory(prefix="pain-audit-") as directory:
        temporary = Path(directory)
        for name, data in payloads.items():
            path = temporary / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        require(preparation.collect_documents(temporary) == manifest["documents"], "Original document reproduction differs")
        environment = assets.check_environment(temporary / preparation.LOCK)
        require(manifest["environment_lock"] == dict(path=preparation.LOCK, **environment), "Locked audit runtime differs")
        vectors, vector_receipt = assets.load_vectors(temporary, "cpu")
        require(manifest["vector_receipt"] == vector_receipt
                and manifest["vector_receipt_sha256"] == digest(vector_receipt), "CPU vector receipt differs")
        plan = design.assignment_plan(design.load_sources(temporary / preparation.REFERENCE))
        tokenizer = preparation.load_tokenizer(temporary)
        preflight = design.tokenize_plan(plan, tokenizer, generation_eos_ids=design.BASE_GENERATION_EOS_IDS)
        require(manifest["plan"] == plan and manifest["preflight"] == preflight, "Original prompts/preflight changed")
        execution.validate_manifest(manifest, tokenizer)
        closure = manifest["prior_study_closure"]
        require(manifest["prior_study_closure_sha256"] == digest(closure) and closure["passed"] is True
                and set(closure["bindings"]) == set(preparation.C_FILES), "Prior closure receipt differs")
        prior = {}
        for key, name in preparation.C_FILES.items():
            value = read(temporary / name)
            require(closure["bindings"][key] == dict(path=name, byte_sha256=sha(temporary / name), content_sha256=digest(value)),
                    "Prior closure document differs")
            prior[key] = value
        c, cf, ca, an, ar, sc, stop = (prior[key] for key in preparation.C_FILES)
        require(digest(c) == cf["manifest_sha256"] == preparation.C_MANIFEST_SHA
                and c["source_sha256"] == cf["source_sha256"] == preparation.C_SOURCE_SHA
                and cf["archive_sha256"] == preparation.C_ARCHIVE_SHA
                and closure["source_archive"] == dict(path=preparation.C_RUN + "/source.tar.gz",
                    byte_sha256=preparation.C_ARCHIVE_SHA, verified_locally=True, copied_into_campaign=False),
                "Prior closed source identity differs")
        require(ca["complete"] is ca["integrity_passed"] is True
                and ca["parent_manifest_sha256"] == an["parent_manifest_sha256"] == preparation.C_MANIFEST_SHA
                and ca["assigned"] == ca["terminal_generations"] == 4320
                and an["complete"] is an["integrity_passed"] is True
                and an["audit_sha256"] == ar["audit_sha256"] == digest(ca)
                and ar["analysis_sha256"] == digest(an) and ar["passed"] is True
                and sc["passed"] is sc["complete"] is sc["response_scoring_complete"] is sc["computation_agrees"] is True
                and sc["assigned"] == sc["checked"] == 4320 and sc["mismatches"] == []
                and closure["prior_readiness"] == ca["readiness"] and closure["prior_decision"] == an["decision"]
                and stop["all_stopped_zero_tasks"] is True and stop["apps"]
                and all(row["State"] == "stopped" and str(row["Tasks"]) == "0" for row in stop["apps"]),
                "Prior complete collection/analysis/shutdown receipt differs")
        pins = read(temporary / assets.ASSETS / "base-weight-pins.json")
        adapter = read(temporary / assets.ASSETS / "asset-receipt.json")
        base = dict(repo=assets.BASE_REPO, revision=assets.BASE_REVISION, files=pins["files"],
                    generation_config_sha256=design.BASE_GENERATION_CONFIG_SHA256)
        config = read(temporary / assets.REFERENCE / "model-metadata/Qwen/Qwen2.5-32B-Instruct/config.json")
        require(config["num_hidden_layers"] == 64 and config["hidden_size"] == 5120
                and type(config["vocab_size"]) is int and config["vocab_size"] > 0,
                "Pinned base architecture differs")
        adapter_files = {row["name"]: row["sha256"] for row in adapter["members"] if row["type"] == "file"}
        require(manifest["analysis_metadata"] == ANALYSIS_METADATA
                and manifest["analysis_metadata_sha256"] == digest(ANALYSIS_METADATA)
                and manifest["resource_liabilities"] == resources.campaign_liabilities()
                and manifest["campaign_ceiling_usd"] == 12 and manifest["setup_ceiling_usd"] == 4
                and manifest["benchmark_ceiling_usd"] == 8, "Analysis/resource contract differs")
    return dict(freeze=frozen, inventory=inventory, tokenizer=tokenizer, vectors=vectors,
                environment=environment, base=base, adapter_files=adapter_files,
                vocabulary_size=config["vocab_size"], prior_closure_sha256=digest(closure))


def runtime_identity(identity, manifest, provenance):
    from curiosity_study import pain_profile_assets as assets

    require(identity["base"] == provenance["base"] and identity["adapter_revision"] == assets.ADAPTER_REVISION
            and identity["adapter_archive_sha256"] == assets.ADAPTER_SHA
            and identity["adapter_files"] == provenance["adapter_files"]
            and identity["vector_receipt"] == manifest["vector_receipt"]
            and identity["environment"] == provenance["environment"], "Loaded weights/adapter/vectors/packages differ")
    require("H100" in identity["gpu"] and identity["cuda_build"] == "13.0"
            and identity["base_dtype"] == "torch.bfloat16" and identity["attention"] == "sdpa"
            and identity["lora_layers"] == 448 and identity["adapter_merged"] is False
            and identity["adapter_dtypes"] == ["torch.float32"] and identity["model_inference_performed"] is False,
            "Actual model arithmetic or GPU identity differs")
    require(isinstance(identity["python_version"], str) and identity["python_version"].startswith("3.12.")
            and identity["machine"] == "x86_64" and bool(identity["driver_version"])
            and type(identity["gpu_total_memory_bytes"]) is int and identity["gpu_total_memory_bytes"] > 0
            and identity["gpu_compute_capability"] == [9, 0]
            and identity["numerical_flags"] == dict(cuda_matmul_allow_tf32=True, cudnn_allow_tf32=True,
                allow_bf16_reduced_precision_reduction=True, float32_matmul_precision="high"),
            "GPU/runtime numerical flags unavailable or changed")
    require(type(identity["load_seconds"]) in (float, int) and math.isfinite(identity["load_seconds"])
            and identity["load_seconds"] >= 0 and all(type(identity[k]) is int and identity[k] >= 0
                for k in ("allocated_bytes", "reserved_bytes")), "Invalid measured loading/memory metadata")
    require(identity["source_sha256"] == manifest["source_sha256"]
            and identity["source_archive_sha256"] == provenance["freeze"]["archive_sha256"]
            and identity["execution_files_sha256"] == digest(provenance["inventory"])
            and isinstance(identity["modal_image_id"], str) and identity["modal_image_id"].startswith("im-"),
            "Cloud executed source or image identity differs")
    return {key: identity[key] for key in RUNTIME_COMPARISON_FIELDS}


def admission_binding(run, phase, manifest, provenance):
    from curiosity_study import pain_profile_resources as resources

    run = Path(run).resolve()
    frozen = provenance["freeze"]
    budget_root = Path(frozen["budget_root"]).resolve()
    require(run == budget_root / "runs/qwen25-pain-profile-control-v1", "Campaign alias is forbidden")
    phases = ["prefetch", "setup"] + (["benchmark"] if phase == "benchmark" else [])
    previous = {}
    for name in phases:
        receipt = read(run / "admissions" / f"{name}.json")
        central = budget_root / "campaign_admissions" / resources.REVISION / f"{name}.json"
        require(receipt == read(central) and receipt["schema_version"] == 1
                and receipt["phase"] == name and receipt["campaign"] == str(run)
                and receipt["manifest_sha256"] == digest(manifest) and receipt["source_sha256"] == manifest["source_sha256"]
                and receipt["source_archive_sha256"] == frozen["archive_sha256"]
                and receipt["policy"] == resources.policy(name) and receipt["execution_files"] == provenance["inventory"]
                and receipt["prior_admissions"] == {p: digest(v) for p, v in previous.items()}
                and receipt["phase_consumed"] is True and receipt["retry_allowed"] is False,
                "Immutable one-admission phase chain differs")
        require(type(receipt["created"]) in (float, int) and math.isfinite(receipt["created"])
                and receipt["deadline"] == receipt["created"] + resources.policy(name)["wall_seconds"]
                and type(receipt["available_compute_estimate_usd"]) in (float, int)
                and receipt["available_compute_estimate_usd"] >= 12, "Bound deadline or allowance differs")
        evidence = {}
        if name != "prefetch":
            prefetch_path = run / "phases/prefetch/summary.json"
            value = read(prefetch_path)
            require(value["complete"] is value["passed"] is True and value["manifest_sha256"] == digest(manifest),
                    "CPU preparation is not complete and bound")
            evidence["prefetch_summary_sha256"] = sha(prefetch_path)
        if name == "benchmark":
            setup_path = run / "phases/setup/audit.json"
            saved = read(setup_path)
            reproduced = audit(run, "setup")
            require(saved == reproduced and saved["complete"] is saved["passed"] is True,
                    "Independent setup audit cannot be reproduced")
            summary_path = run / "phases/setup/summary.json"
            evidence.update(setup_summary_sha256=sha(summary_path), setup_audit_sha256=sha(setup_path),
                            benchmark_forecast_seconds=saved["forecast"]["benchmark_seconds"])
        require(receipt["prior_evidence"] == evidence, "Phase prerequisite evidence differs")
        previous[name] = receipt
    return previous[phase]


def _predicate(function, *args):
    try:
        function(*args)
        return True
    except (ValueError, KeyError, IndexError, TypeError):
        return False


def _repeat(left, right):
    require([{k: v for k, v in row.items() if k != "trial_id"} for row in left["records"]]
            == [{k: v for k, v in row.items() if k != "trial_id"} for row in right["records"]],
            "Repeat records differ")
    compare_rows(left, right, range(len(left["records"])))
    a, b = left["evidence"]["forwards"], right["evidence"]["forwards"]
    require(len(a) == len(b), "Repeat forward counts differ")
    keys = ("input_ids", "attention_mask", "position_ids", "cache_length_before", "cache_length_after",
            "full_fp32_logits_sha256", "sampling", "prepatch_pain_projection", "post_layer61_pain_projection")
    require(all(all(x[key] == y[key] for key in keys) for x, y in zip(a, b, strict=True)),
            "Repeat full-row trajectory differs")


def _prefill(left, right):
    a, b = left["evidence"]["forwards"][0], right["evidence"]["forwards"][0]
    require(all(a[key] == b[key] for key in ("input_ids", "attention_mask", "position_ids"))
            and [r["before_sha256"] for r in a["geometry"]] == [r["before_sha256"] for r in b["geometry"]],
            "Cross-arm initial residual states differ")


def _active_engagement(result, row=None):
    for frame in result["evidence"]["forwards"]:
        if frame["phase"] != "decode" or "sampling" not in frame:
            continue
        active = frame["sampling"]["alive_before"]
        for index in range(len(active)) if row is None else (row,):
            geometry = frame["geometry"][index]
            if active[index] and geometry["coefficient"] == 1 and geometry["achieved_delta_norm_min"] is not None and geometry["achieved_delta_norm_min"] > 0:
                return True
    return False


def independent_outcomes(manifest, call, result=None, *, missing=False):
    base = {r["base_request_id"]: r for r in manifest["plan"]["base_requests"]}
    records = {} if result is None else {r["trial_id"]: r for r in result["records"]}
    require(result is None or len(records) == len(result["records"]), "Duplicate response trial IDs")
    require(set(records) <= set(call["trial_ids"]), "Unassigned response trial ID")
    rows = []
    for trial, base_id in zip(call["trial_ids"], call["base_request_ids"], strict=True):
        row = dict(base[base_id], trial_id=trial, arm=call["arm"], call_id=call["call_id"],
                   status="missing" if missing else "execution_error", category=None, valid=None,
                   cap_hit=None, raw=None, answer_token_ids=None, record_sha256=None)
        record = records.get(trial)
        if record is not None:
            row.update(status="completed", **score_text(record["text"], row["button_names"], row["target_name"]),
                       cap_hit=record["cap_hit"], raw=record["text"], answer_token_ids=record["answer_token_ids"],
                       record_sha256=digest(record))
        elif call["expected_error"] and not missing and result is not None and result["evidence"].get("error") == dict(
                type="RuntimeError", message=f"controlled compatibility fault: {call['inject_failure_at']}"):
            row["status"] = "expected_fault"
        rows.append(row)
    return rows


def measured_forecast(identity, calls, timings):
    names = [call["call_id"] for call in calls if call["role"] == "primary"]
    if len(names) != 10 or any(name not in timings for name in names):
        return dict(benchmark_seconds=None, limit_seconds=2880, passed=False,
                    load_seconds=identity["load_seconds"], slowest_primary_cohort_seconds=None)
    slowest = max(timings[name] for name in names)
    seconds = identity["load_seconds"] + 2 * slowest * 65 + 120
    return dict(benchmark_seconds=seconds, limit_seconds=2880, passed=seconds <= 2880,
                load_seconds=identity["load_seconds"], slowest_primary_cohort_seconds=slowest,
                safety_multiplier=2, benchmark_cohorts=65, export_seconds=120)


def audit_phase(run, phase, manifest, provenance, admission):
    """Durable reconciliation separated for small synthetic filesystem tests."""
    from curiosity_study import pain_profile_execution as execution

    require(phase in ("setup", "benchmark"), "Unknown numerical phase")
    run = Path(run)
    directory = run / "phases" / phase
    calls = execution.setup_calls(manifest) if phase == "setup" else execution.benchmark_calls(manifest)
    assigned = sum(len(c["trial_ids"]) for c in calls)
    require((len(calls), assigned) == ((21, 612) if phase == "setup" else (65, 2020)), "Fixed phase denominator changed")
    paths = sorted((directory / "calls").glob("*.json"))
    require([p.name for p in paths] == [f"{index:04d}.json" for index in range(len(paths))]
            and len(paths) <= len(calls), "Unknown, duplicate or out-of-order call artifact")
    require(not any(p.is_symlink() for p in directory.rglob("*")), "Linked phase evidence is forbidden")
    started_path, identity_path = directory / "started.json", directory / "identity.json"
    started, identity = None, None
    if started_path.exists():
        started = read(started_path)
        require(started["phase"] == phase and started["manifest_sha256"] == digest(manifest)
                and started["calls"] == calls and started["calls_sha256"] == digest(calls)
                and re.fullmatch("[a-f0-9]{32}", started["attempt"])
                and type(started["started_unix"]) in (int, float) and math.isfinite(started["started_unix"])
                and admission["created"] <= started["started_unix"] < admission["deadline"]
                and started["deadline_unix"] == admission["deadline"] - 60,
                "Actual attempt/input/deadline binding differs from the 60-second export reserve")
    if identity_path.exists():
        identity = read(identity_path)
        require(started is not None and started["identity_sha256"] == digest(identity), "Runtime identity lacks its original attempt")
        require(identity["admission_sha256"] == digest(admission), "Runtime bound another phase admission")
        actual_policy = runtime_identity(identity, manifest, provenance)
        if phase == "benchmark":
            setup_identity = read(run / "phases/setup/identity.json")
            require(actual_policy == runtime_identity(setup_identity, manifest, provenance),
                    "Benchmark runtime differs from certified setup")
    require(not paths or (started is not None and identity is not None), "Observed calls lack bound attempt/runtime")
    prior, source_prior, all_rows, bindings, timings, call_audits = {}, {}, [], [], {}, []
    all_passed, stopped = True, False
    for call in calls:
        path = directory / "calls" / f"{call['call_index']:04d}.json"
        if not path.exists():
            all_rows.extend(independent_outcomes(manifest, call, missing=True))
            continue
        require(not stopped, "A failed call was retried or followed by another paid call")
        envelope = read(path)
        items = execution.engine_items(manifest, call)
        require(envelope["schema_version"] == 1 and envelope["phase"] == phase
                and envelope["attempt"] == started["attempt"] and envelope["manifest_sha256"] == digest(manifest)
                and envelope["identity_sha256"] == digest(identity) and envelope["call_spec"] == call
                and envelope["call_spec_sha256"] == digest(call) and envelope["items_sha256"] == digest(items),
                "Call source/attempt/row binding differs")
        require(envelope["prefill_reference_call_id"] == (source_prior[call["source_cohort"]][0]
                if call["source_cohort"] in source_prior else None), "Cross-arm reference call changed")
        artifact = envelope["arrays_artifact"]
        array_path = path.with_suffix(".npz")
        require(artifact["path"] == array_path.name and sha(array_path) == artifact["byte_sha256"], "Residual artifact byte binding differs")
        with np.load(array_path, allow_pickle=False) as stored:
            require(len(stored.files) == len(set(stored.files)), "Duplicate NPZ member")
            arrays = {key: stored[key] for key in stored.files}
        require(artifact["arrays"] == {key: array_binding(value) for key, value in arrays.items()}, "Array layout/content binding differs")
        result = envelope["engine_result"]
        if result is None:
            error = envelope["executor_error"]
            require(isinstance(error, dict) and isinstance(error.get("type"), str)
                    and isinstance(error.get("message"), str) and arrays == {}, "Malformed outer execution failure")
            checks = dict(executor_completed=False)
            numerical = dict(complete=False, passed=False, expected_fault_observed=False, forwards=0)
        else:
            require(envelope["executor_error"] is None, "Numerical call and outer failure conflict")
            setting = design.ARM_SETTINGS[call["arm"]]
            numerical = recompute_call(result, arrays, items,
                direction=provenance["vectors"]["pain" if setting["direction"] == "s2" else "fear"],
                pain_unit=provenance["vectors"]["pain_unit_float32"], tokenizer=provenance["tokenizer"],
                vocabulary_size=provenance["vocabulary_size"], coefficient=setting["coefficient"],
                prefill_policy=setting["prefill_policy"], eos_ids=manifest["preflight"]["eos_token_ids"],
                pad_id=manifest["preflight"]["pad_token_id"], mode="compatibility" if phase == "setup" else "benchmark",
                steering_enabled=call["steering_enabled"], one_row_only=call["one_row_only"],
                expected_fault=call["inject_failure_at"])
            evidence = result["evidence"]
            reference = source_prior.get(call["source_cohort"])
            checks = dict(hook_inventory_empty=evidence["hooks_before"] == evidence["hooks_after"] == {},
                same_cohort_prefill_identity=(bool(evidence["forwards"]) and len(evidence["forwards"][0].get("geometry", [])) == len(items))
                if reference is None else _predicate(_prefill, reference[1], result))
            if call["expected_error"]:
                checks["expected_controlled_failure"] = numerical["expected_fault_observed"] and all(numerical["checks"].values())
            else:
                checks["complete_integrity"] = numerical["passed"]
            if call["role"] in ("repeat", "unhooked", "clean_repeat"):
                checks["same_source_repeat_identity"] = _predicate(_repeat, prior[call["reference_call_id"]], result)
            if call["role"] == "one_row":
                checks["unaffected31_active_prediction_identity"] = _predicate(compare_rows, prior[call["reference_call_id"]], result, range(1, len(items)))
                checks["row0_active_decode_engaged"] = _active_engagement(result, row=0)
            if call["role"] == "primary" and call["arm"] != "baseline":
                checks["active_decode_engaged"] = _active_engagement(result)
            prior[call["call_id"]] = result
            source_prior.setdefault(call["source_cohort"], (call["call_id"], result))
        passed = all(checks.values())
        require(envelope["checks"] == checks and envelope["passed"] is passed, "Saved call checks disagree with raw reconstruction")
        outcomes = independent_outcomes(manifest, call, result)
        require(envelope["outcomes"] == outcomes and envelope["outcomes_sha256"] == digest(outcomes), "Independent raw answer scoring differs")
        all_rows.extend(outcomes)
        elapsed, engine_seconds = envelope["elapsed_seconds"], envelope["engine_seconds"]
        require(type(elapsed) in (float, int) and type(engine_seconds) in (float, int)
                and math.isfinite(elapsed) and math.isfinite(engine_seconds) and 0 <= engine_seconds <= elapsed,
                "Invalid measured call timing")
        timings[call["call_id"]] = elapsed
        bindings.append(dict(call_id=call["call_id"], path=str(path.relative_to(directory)),
                             content_sha256=digest(envelope), total_seconds=elapsed, passed=passed))
        call_audits.append(dict(call_id=call["call_id"], passed=passed, checks=checks, numerical=numerical,
                               envelope_byte_sha256=sha(path), arrays_byte_sha256=sha(array_path)))
        all_passed &= passed
        stopped = not passed
    require(len(all_rows) == assigned and len({row["trial_id"] for row in all_rows}) == assigned,
            "Assigned denominator is incomplete or duplicated")
    expected_arrays = {f"{index:04d}.npz" for index in range(len(paths))}
    actual_arrays = {p.name for p in (directory / "calls").glob("*.npz")}
    partial_arrays = {p.name for p in (directory / "calls").glob("*.partial")}
    allowed_orphan = {f"{len(paths):04d}.npz"} if len(paths) < len(calls) else set()
    require(expected_arrays <= actual_arrays and actual_arrays - expected_arrays <= allowed_orphan
            and partial_arrays <= {f"{len(paths):04d}.npz.partial"}, "Unexpected residual artifacts or attempted retries")
    complete_calls = len(paths) == len(calls)
    summary_path = directory / "summary.json"
    summary = read(summary_path) if summary_path.exists() else None
    worker_failure = None
    if summary is not None and "terminal_error" not in summary:
        worker_path = run / "transport" / phase / "worker-error.json"
        require(set(summary) == {"schema_version", "phase", "manifest_sha256", "complete", "passed",
                    "model_inference_performed", "error", "finished_unix"}
                and summary["schema_version"] == 1 and summary["phase"] == phase
                and summary["manifest_sha256"] == digest(manifest)
                and summary["complete"] is summary["passed"] is False
                and summary["model_inference_performed"] is None
                and isinstance(summary["error"], dict) and set(summary["error"]) == {"type", "message"}
                and all(isinstance(summary["error"][k], str) for k in ("type", "message"))
                and type(summary["finished_unix"]) in (int, float) and math.isfinite(summary["finished_unix"])
                and worker_path.is_file() and read(worker_path) == summary and sha(worker_path) == sha(summary_path),
                "Unbound or malformed terminal worker failure")
        worker_failure = dict(error=summary["error"], artifact_byte_sha256=sha(worker_path))
        summary = None
    prediction = measured_forecast(identity, calls, timings) if phase == "setup" and identity is not None else None
    integrity = complete_calls and all_passed and summary is not None
    if summary is not None:
        terminal = summary["terminal_error"]
        require(terminal is None or (isinstance(terminal, dict) and isinstance(terminal.get("type"), str)
                and isinstance(terminal.get("message"), str)), "Malformed phase terminal failure")
        integrity &= terminal is None
        require(terminal is not None or (complete_calls and all_passed), "Phase omitted terminal failure")
        expected = dict(schema_version=1, phase=phase, manifest_sha256=digest(manifest), source_sha256=manifest["source_sha256"],
            attempt=started["attempt"], identity_sha256=digest(identity), calls_sha256=digest(calls),
            assigned_calls=len(calls), observed_calls=len(paths), assigned_rows=assigned,
            completed_rows=sum(r["status"] == "completed" for r in all_rows),
            expected_fault_rows=sum(r["status"] == "expected_fault" for r in all_rows),
            missing_rows=sum(r["status"] == "missing" for r in all_rows),
            execution_error_rows=sum(r["status"] == "execution_error" for r in all_rows),
            complete=complete_calls, integrity_passed=bool(integrity),
            passed=bool(integrity and (prediction is None or prediction["passed"])), terminal_error=terminal,
            call_bindings=bindings, outcomes_sha256=digest(all_rows), forecast=prediction, behavior_gates_used=False)
        require(summary == expected and read(directory / "outcomes.json") == all_rows, "Phase summary or full assigned outcomes differs")
    elif (directory / "outcomes.json").exists():
        require(read(directory / "outcomes.json") == all_rows, "Partial phase outcomes differ")
    complete = complete_calls and summary is not None
    checks = dict(frozen_source_and_inputs=True, phase_admission=True,
                  actual_runtime_verified=identity is not None, complete_call_inventory=complete_calls,
                  call_checks_passed=all_passed and bool(paths), durable_summary_verified=summary is not None,
                  no_terminal_phase_error=summary is not None and summary["terminal_error"] is None,
                  setup_forecast_passed=phase != "setup" or (prediction is not None and prediction["passed"]))
    return dict(schema_version=1, assay="pain_profile_control", phase=phase, complete=complete,
        passed=complete and all(checks.values()), integrity_passed=bool(integrity), checks=checks,
        manifest_sha256=digest(manifest), source_sha256=manifest["source_sha256"],
        archive_byte_sha256=provenance["freeze"]["archive_sha256"], freeze_byte_sha256=sha(run / "freeze.json"),
        auditor_byte_sha256=sha(__file__), summary_byte_sha256=sha(summary_path) if summary_path.exists() else None,
        identity_sha256=digest(identity) if identity is not None else None, admission_sha256=digest(admission),
        assigned_calls=len(calls), observed_calls=len(paths), assigned_rows=assigned,
        observed_rows=sum(row["status"] == "completed" for row in all_rows),
        expected_fault_rows=sum(row["status"] == "expected_fault" for row in all_rows),
        missing_rows=sum(row["status"] == "missing" for row in all_rows),
        execution_error_rows=sum(row["status"] == "execution_error" for row in all_rows),
        outcomes=all_rows, outcomes_sha256=digest(all_rows), calls=call_audits, forecast=prediction,
        orphan_array_files=sorted(actual_arrays - expected_arrays | partial_arrays),
        worker_failure=worker_failure,
        behavior_gates_used=False, scientific_qualification=False,
        runtime_comparison_fields=list(RUNTIME_COMPARISON_FIELDS),
        limitations=[
            "Source answer grammar intentionally permits button-name prefixes; no grammar repair or button consequence is inferred.",
            "Every physical residual position has frozen read-only arithmetic guards and fingerprints; independent raw BF16 vector reconstruction covers only the prospectively saved sample positions.",
            "Full-vocabulary identity is compared via saved complete-FP32 row hashes; logits are not stored for independent probability recomputation.",
            "Loaded large-weight and adapter identities rely on the frozen loader's complete byte-verification receipts; this CPU audit rechecks metadata and saved vectors without loading weights.",
            "The prior C source archive was verified at preparation; its bound closure documents are replayed here, not the excluded historical archive.",
            "All missing or failed assigned slots are retained and cannot qualify this phase; an expected setup fault is an engineering disposition, not a behavioral response."])


def audit(run, phase):
    run = Path(run).resolve()
    manifest = read(run / "manifest.json")
    provenance = source_provenance(run, manifest)
    admission = admission_binding(run, phase, manifest, provenance)
    return audit_phase(run, phase, manifest, provenance, admission)


if __name__ == "__main__":
    import argparse
    import json
    from curiosity_study.artifacts import write_once

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("setup", "benchmark"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = audit(args.run, args.phase)
    out = args.out or args.run / "phases" / args.phase / "audit.json"
    write_once(out, result)
    print(json.dumps({key: result[key] for key in ("phase", "complete", "passed", "assigned_rows", "observed_rows", "missing_rows")}))
