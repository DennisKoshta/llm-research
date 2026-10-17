"""Independent read-only check of the completed five-call numerical diagnostic.

No model inference, behavioral parsing, condition-rate estimates or admission.
The check preserves the failed original campaign and its unadmitted benchmark.
"""

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import tarfile

import numpy as np
import torch

from curiosity_study.artifacts import digest, read, write_once
from curiosity_study import pain_profile_assets as assets

ROOT = Path(__file__).resolve().parents[1]
NAME = "qwen25-pain-deterministic-diagnostic-v1"
MANIFEST = "fa75488f5047854d22f92a0e822c85bac600a259cfc2d65b14286c6bfd0bed8d"
PARENT = "e87c70267be0b7813750dbef9d752f0855cf5b2db90e81f2381127d544b59a63"
ARCHIVE = "9c59998a8a162075a4adc201c4a374f821f76978b0c4ec518dce4630cd3f1fa7"
PRESET = dict(cublas_workspace_config=":4096:8", deterministic_algorithms=True, deterministic_warn_only=False,
    sdpa_backends=dict(math=True, flash=False, mem_efficient=False, cudnn=False), cudnn_deterministic=True,
    cudnn_benchmark=False, cuda_matmul_allow_tf32=True, cudnn_allow_tf32=True,
    allow_bf16_reduced_precision_reduction=True, float32_matmul_precision="high")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_binding(value):
    value = np.ascontiguousarray(value)
    return dict(dtype=str(value.dtype), shape=list(value.shape), byte_sha256=hashlib.sha256(value.tobytes()).hexdigest())


def tensor_binding(value):
    value = value.detach().contiguous().cpu()
    return dict(dtype=str(value.dtype), shape=list(value.shape), byte_sha256=hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest())


def check_call(call, items, envelope, arrays, direction, pain_unit, preflight):
    result, checks = envelope["engine_result"], envelope["checks"]
    evidence, records = result["evidence"], result["records"]
    require(envelope["call_spec"] == call and envelope["items_sha256"] == digest(items)
            and envelope["error"] is None and envelope["passed"] is True and all(v is True for v in checks.values()),
            "Call identity or saved engineering failure")
    require(evidence["complete"] is evidence["integrity_passed"] is True and evidence["error"] is None
            and evidence["hooks_before"] == evidence["hooks_after"] == {}
            and evidence["direction"] == tensor_binding(direction) and evidence["pain_unit"] == tensor_binding(pain_unit)
            and evidence["coefficient"] == call["coefficient"] and evidence["steering_enabled"] is call["steering_enabled"]
            and evidence["prefill_policy"] == "full_prefix" and evidence["evidence_mode"] == "compatibility"
            and evidence["steer_layer"] == 38 and evidence["monitor_layer"] == 61
            and evidence["max_new_tokens"] == 8 and evidence["temperature"] == .7 and evidence["top_p"] == .95
            and evidence["device_type"] == "cuda" and evidence["cpu_synthetic_execution"] is False
            and evidence["items_sha256"] == digest(items) and evidence["trial_ids"] == call["trial_ids"]
            and evidence["one_row_only"] is None and evidence["inject_failure_at"] is None
            and all(v is True for v in evidence["checks"].values()), "Observed numerical policy/geometry failed")
    require(evidence["arrays"] == {key: array_binding(a) for key, a in arrays.items()}, "Array evidence inventory differs")
    n, width = len(items), max(len(item["prompt_ids"]) for item in items)
    pad, eos = preflight["pad_token_id"], set(preflight["eos_token_ids"])
    require(evidence["pad_token_id"] == pad and evidence["eos_token_ids"] == sorted(eos)
            and evidence["applied_coefficients"] == [call["coefficient"]] * n, "Sampler/vector mask changed")
    input_ids = [[pad] * (width - len(i["prompt_ids"])) + i["prompt_ids"] for i in items]
    mask = [[0] * (width - len(i["prompt_ids"])) + [1] * len(i["prompt_ids"]) for i in items]
    alive, tokens, previous_rng = [True] * n, [[] for _ in items], None
    used, geometry, active_decode = set(), [], False
    frames = evidence["forwards"]
    require(1 <= len(frames) <= 8 and evidence["attempted_forwards"] == len(frames)
            and evidence["incomplete_forward"] is None, "Incomplete/unplanned forward count")
    for index, frame in enumerate(frames):
        prefill = index == 0
        positions = [[0] * (width - len(i["prompt_ids"])) + list(range(len(i["prompt_ids"]))) for i in items] if prefill else [[len(i["prompt_ids"]) + index - 1] for i in items]
        require(frame["forward_index"] == index and frame["phase"] == ("prefill" if prefill else "decode")
                and frame["input_ids"] == input_ids and frame["attention_mask"] == mask and frame["position_ids"] == positions
                and frame["cache_length_before"] == (0 if prefill else width + index - 1)
                and frame["cache_length_after"] == width + index, "Fresh-cache/input/position timeline differs")
        require(frame["activation_dtypes"] == dict(before="torch.bfloat16", after="torch.bfloat16")
                and frame["vocabulary_size"] == 152064 and frame["finite_logits"] is True
                and len(frame["full_fp32_logits_sha256"]) == n
                and all(isinstance(v, str) and len(v) == 64 for v in frame["full_fp32_logits_sha256"])
                and frame["steering_hook_calls"] == int(call["steering_enabled"])
                and frame["read_only_projection_calls"] == int(not call["steering_enabled"])
                and all(frame[k] == 1 for k in ("before_observer_calls", "after_observer_calls", "monitor_hook_calls")),
                "Missing full-vocabulary/capture/hook evidence")
        physical_width = width if prefill else 1
        expected_steering = [[float(value and bool(call["coefficient"])) for value in row] for row in mask] if prefill else None
        require(frame["steering_mask"] == expected_steering and len(frame["geometry"]) == n, "Steering mask inventory differs")
        for row, g in enumerate(frame["geometry"]):
            coefficient = call["coefficient"]
            padding = width - len(items[row]["prompt_ids"]) if prefill else 0
            active = torch.arange(physical_width) >= padding
            if not coefficient:
                active[:] = False
            require(g["trial_id"] == items[row]["trial_id"] and g["coefficient"] == coefficient
                    and g["active_positions"] == int(active.sum()) and g["padding_positions"] == padding
                    and g["unmodified_positions"] == physical_width - int(active.sum())
                    and all(g[k] is True for k in ("finite", "exact_bf16_arithmetic", "unmodified_positions_identical", "padding_identical", "all_requested_positions_engaged"))
                    and g["max_abs_error"] == 0, "Observed all-row BF16 guard or mask differs")
            saved = row in (0, n - 1)
            require(all(("saved_" + label in g) == saved for label in ("before", "after")), "Saved sample inventory differs")
            if not saved:
                continue
            values = {}
            for label in ("before", "after"):
                binding = g["saved_" + label]
                key = f"f{index:02d}_r{row:02d}_{label}"
                array = arrays[key]
                require(binding == dict(key=key, positions=list(range(physical_width)), **array_binding(array))
                        and array.dtype == np.float32 and array.shape == (physical_width, 5120) and np.isfinite(array).all(),
                        "Saved vector layout/hash differs")
                tensor = torch.from_numpy(array.copy()).to(torch.bfloat16)
                require(np.array_equal(tensor.float().numpy(), array)
                        and g[label + "_sha256"] == digest(tensor_binding(tensor)), "Saved BF16 fingerprint differs")
                values[label] = tensor
                used.add(key)
            before, after = values["before"], values["after"]
            delta = active.to(torch.bfloat16)[:, None] * torch.tensor(coefficient, dtype=torch.bfloat16) * direction[None, :]
            expected = before if not coefficient else before + delta
            require(torch.equal(after.view(torch.uint8), expected.view(torch.uint8)), "Independent saved BF16 arithmetic differs")
            norms = (after.double() - before.double())[active].norm(dim=-1)
            geometry.append(dict(forward_index=index, physical_row=row, saved_positions=physical_width,
                active_positions=int(active.sum()), scalar_components=before.numel(), exact_bf16=True,
                cpu_float64_min=float(norms.min()) if norms.numel() else 0., cpu_float64_max=float(norms.max()) if norms.numel() else 0.,
                norms_used_as_gate=False))
        sample = frame["sampling"]
        require(sample["step"] == index and sample["alive_before"] == alive and sample["uniform_draws_per_row"] == [1] * n
                and (previous_rng is None or sample["rng_state_before_sha256"] == previous_rng), "Original sampler/RNG chain differs")
        previous_rng = sample["rng_state_after_sha256"]
        require(len(previous_rng) == n and len(sample["uniforms"]) == n
                and all(type(x) is float and math.isfinite(x) and 0 <= x < 1 for x in sample["uniforms"]), "Unavailable row sampling evidence")
        sampled = sample["sampled_before_alive_mask"]
        require(len(sampled) == n and all(type(x) is int and 0 <= x < 152064 for x in sampled), "Sampled token IDs differ")
        selected = [sampled[row] if alive[row] else pad for row in range(n)]
        require(sample["selected_token_ids"] == selected, "Original inactive-row forwarding differs")
        active_decode |= bool(index and any(alive))
        for row in range(n):
            if alive[row]:
                tokens[row].append(selected[row])
        alive = [alive[row] and selected[row] not in eos and len(tokens[row]) < 8 for row in range(n)]
        require(sample["alive_after"] == alive and (index == len(frames) - 1 or any(alive)), "EOS/cap timeline differs")
        input_ids, mask = [[x] for x in selected], [row + [1] for row in mask]
    require(not any(alive) and active_decode and used == set(arrays), "Incomplete or decode-unexercised call")
    require(len(records) == n and [row["trial_id"] for row in records] == call["trial_ids"], "Record/trial coverage differs")
    for row, item, generated in zip(records, items, tokens, strict=True):
        require(row["answer_token_ids"] == generated and row["prompt_token_ids"] == item["prompt_ids"]
                and row["ended_with_eos"] is (generated[-1] in eos)
                and row["cap_hit"] is (len(generated) == 8 and generated[-1] not in eos), "Raw records/termination differ")
    engaged = any(f["phase"] == "decode" and any(a and g["coefficient"] == 1 and g["achieved_delta_norm_min"] > 0
        for a, g in zip(f["sampling"]["alive_before"], f["geometry"], strict=True)) for f in frames)
    require(not call["coefficient"] or engaged, "Nonzero stressor decode was not exercised")
    return dict(call_id=call["call_id"], rows=n, forwards=len(frames), active_decode=True,
                nonzero_active_decode=engaged, saved_geometry=geometry)


def check(run):
    run = Path(run).resolve()
    m, frozen, admission = (read(run / name) for name in ("manifest.json", "freeze.json", "admission.json"))
    require(run == ROOT / "runs" / NAME and digest(m) == frozen["manifest_sha256"] == MANIFEST,
            "Wrong fixed diagnostic manifest")
    execution = Path(frozen["execution_root"])
    original = m["original_manifest"]
    require(digest(original) == PARENT and frozen["original_source_archive_sha256"] == m["original_source_archive_sha256"] == ARCHIVE,
            "Original source identity differs")
    parent = ROOT / "runs/qwen25-pain-profile-control-v1"
    require(sha(parent / "source.tar.gz") == ARCHIVE and not (parent / "admissions/benchmark.json").exists()
            and read(parent / "phases/setup/summary.json")["passed"] is False, "Original failed campaign status changed")
    with tarfile.open(parent / "source.tar.gz") as archive:
        inventory = {member.name: hashlib.sha256(archive.extractfile(member).read()).hexdigest() for member in archive.getmembers()}
    require(inventory == m["original_execution_files"], "Original archive inventory changed")
    for directory, expected in ((execution / "original", inventory), (execution / "diagnostic", m["diagnostic_files"])):
        actual = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
        require(actual == set(expected) and not any(p.is_symlink() for p in directory.rglob("*")), "Frozen file inventory changed")
        require(all(sha(directory / name) == value for name, value in expected.items()), "Frozen file bytes changed")
    require(sha(assets.__file__) == inventory["src/curiosity_study/pain_profile_assets.py"], "CPU vector helper differs from frozen source")
    vectors, vector_receipt = assets.load_vectors(execution / "original", "cpu")
    env = assets.check_environment(execution / "original/configs/pain_profile_requirements.lock")
    identity, summary, outcomes, started = (read(run / name) for name in ("identity.json", "summary.json", "outcomes.json", "started.json"))
    cpu = read(parent / "phases/prefetch/summary.json")
    require(identity["base"] == cpu["assets"]["base"] and identity["adapter_files"] == cpu["assets"]["adapter_files"]
            and identity["adapter_revision"] == assets.ADAPTER_REVISION and identity["adapter_archive_sha256"] == assets.ADAPTER_SHA
            and identity["vector_receipt"] == vector_receipt and identity["environment"] == env
            and identity["deterministic_preset"] == summary["effective_preset"] == PRESET
            and identity["base_dtype"] == "torch.bfloat16" and identity["adapter_dtypes"] == ["torch.float32"]
            and identity["adapter_merged"] is False and identity["lora_layers"] == 448 and identity["attention"] == "sdpa"
            and "H100" in identity["gpu"] and identity["gpu_compute_capability"] == [9, 0]
            and identity["cuda_build"] == "13.0" and identity["python_version"].startswith("3.12."), "Runtime/assets/preset differ")
    require(admission == read(ROOT / "campaign_admissions" / NAME / "admission.json")
            and admission["manifest_sha256"] == MANIFEST and admission["phase_consumed"] is True
            and admission["retry_allowed"] is False and admission["deadline"] == admission["created"] + 900
            and admission["maximum_liability_usd"] == 1.3630464
            and started["manifest_sha256"] == identity["manifest_sha256"] == summary["manifest_sha256"] == MANIFEST
            and started["admission_sha256"] == identity["admission_sha256"] == summary["admission_sha256"] == digest(admission)
            and admission["created"] <= started["started_unix"] < summary["finished_unix"] <= admission["deadline"] - 60
            and started["execution_deadline"] == admission["deadline"] - 60, "Admission/deadline/attempt differs")
    requests = {row["base_request_id"]: row for row in original["preflight"]["requests"]}
    expected_calls, prior = [], []
    for index, label in enumerate(("unhooked", "zero_hook", "pain_full_stressor", "unhooked_repeat", "zero_hook_repeat")):
        cohort = 12 if index == 2 else 0
        ids = [row["base_request_id"] for row in original["plan"]["base_requests"] if row["source_cohort"] == cohort]
        call_id = f"deterministic-{index:04d}"
        expected_calls.append(dict(call_index=index, call_id=call_id, label=label, source_cohort=cohort,
            base_request_ids=ids, trial_ids=[f"{call_id}/{key}" for key in ids], coefficient=1. if index == 2 else 0.,
            steering_enabled=index not in (0, 3), prefill_policy="full_prefix", evidence_mode="compatibility",
            baseline_reference_call_ids=list(prior) if index != 2 else []))
        if index != 2:
            prior.append(call_id)
    require(m["call_plan"] == expected_calls and started["call_plan_sha256"] == digest(expected_calls), "Fixed five-call order changed")
    require(sorted(p.name for p in (run / "calls").iterdir()) == [f"{i:04d}{suffix}" for i in range(5) for suffix in (".json", ".npz")], "Unexpected or incomplete call artifact coverage")
    results, call_receipts, expected_outcomes, call_bindings = {}, [], [], []
    for call in expected_calls:
        path = run / "calls" / f"{call['call_index']:04d}.json"
        envelope = read(path)
        artifact = envelope["arrays_artifact"]
        apath = path.with_suffix(".npz")
        require(envelope["manifest_sha256"] == MANIFEST and envelope["admission_sha256"] == digest(admission)
                and artifact["path"] == apath.name and artifact["byte_sha256"] == sha(apath), "Call admission/array hash differs")
        with np.load(apath, allow_pickle=False) as stored:
            arrays = {key: stored[key] for key in stored.files}
        require(artifact["arrays"] == {key: array_binding(value) for key, value in arrays.items()}, "NPZ content/layout differs")
        items = [dict(requests[key], trial_id=trial) for key, trial in zip(call["base_request_ids"], call["trial_ids"], strict=True)]
        checked = check_call(call, items, envelope, arrays, vectors["pain"], vectors["pain_unit_float32"], original["preflight"])
        checked.update(json_byte_sha256=sha(path), npz_byte_sha256=sha(apath))
        call_receipts.append(checked)
        results[call["call_index"]] = envelope["engine_result"]
        expected_outcomes.extend(dict(trial_id=row["trial_id"], call_id=call["call_id"], status="completed", raw_record_sha256=digest(row)) for row in envelope["engine_result"]["records"])
        call_bindings.append(dict(call_id=call["call_id"], content_sha256=digest(envelope), passed=True))
    pairs = []
    for a, b in itertools.combinations((0, 1, 3, 4), 2):
        left, right = results[a], results[b]
        require([{k: v for k, v in row.items() if k != "trial_id"} for row in left["records"]]
                == [{k: v for k, v in row.items() if k != "trial_id"} for row in right["records"]], "Baseline raw records differ")
        require(len(left["evidence"]["forwards"]) == len(right["evidence"]["forwards"]), "Baseline trajectory lengths differ")
        for first, second in zip(left["evidence"]["forwards"], right["evidence"]["forwards"], strict=True):
            keys = ("input_ids", "attention_mask", "position_ids", "cache_length_before", "cache_length_after", "full_fp32_logits_sha256", "sampling", "prepatch_pain_projection", "post_layer61_pain_projection")
            require(all(first[key] == second[key] for key in keys)
                    and [g["before_sha256"] for g in first["geometry"]] == [g["before_sha256"] for g in second["geometry"]], "Baseline prehook/logit/input/RNG evidence differs")
        pairs.append(dict(left=a, right=b, rows=32, full_record_identity=True, all_step_prehook_logit_input_rng_identity=True))
    require(outcomes == expected_outcomes and len(outcomes) == len({r["trial_id"] for r in outcomes}) == 148
            and summary["outcomes_sha256"] == digest(outcomes) and summary["calls"] == call_bindings
            and summary["complete"] is summary["passed"] is True and summary["error"] is None
            and summary["assigned_calls"] == summary["observed_calls"] == 5
            and summary["assigned_rows"] == summary["completed_rows"] == 148
            and summary["execution_error_rows"] == summary["missing_rows"] == 0
            and summary["identity_sha256"] == digest(identity)
            and summary["behavior_scoring_performed"] is summary["scientific_admission"] is summary["retry_authorized"] is summary["norm_threshold_used"] is False,
            "Complete summary/denominator differs")
    coverage, launch = read(run / "collection-coverage.json"), read(run / "launch-outcome.json")
    shutdown = read(ROOT / "reports" / (NAME + "-cost-and-shutdown.json"))
    require(coverage["rows"] == outcomes and coverage["issues"] == [] and coverage["completed_rows"] == 148
            and launch["collection_coverage_sha256"] == digest(coverage) and launch["all_stopped_zero_tasks"] is True
            and launch["failure"] is launch["export_error"] is None and shutdown["all_stopped_zero_tasks"] is True
            and read(run / "cloud-finished.json")["error"] is None, "Coverage/export/shutdown evidence differs")
    all_geometry = [row for call in call_receipts for row in call["saved_geometry"]]
    return dict(schema_version=1, scope="Observed engineering repeatability under the fixed deterministic preset only", complete=True, passed=True,
        manifest_sha256=MANIFEST, original_manifest_sha256=PARENT, original_source_archive_sha256=ARCHIVE,
        checker_byte_sha256=sha(__file__), command=f"PYTHONPATH=src .pain-runtime-venv/bin/python scripts/{Path(__file__).name}",
        freeze_byte_sha256=sha(run / "freeze.json"), admission_sha256=digest(admission), identity_sha256=digest(identity), summary_byte_sha256=sha(run / "summary.json"),
        outcomes_sha256=digest(outcomes), original_file_inventory_sha256=digest(inventory), diagnostic_files=m["diagnostic_files"],
        assigned_calls=5, completed_calls=5, assigned_rows=148, completed_rows=148, missing_rows=0, execution_error_rows=0,
        effective_preset=PRESET, baseline_pair_comparisons=pairs, calls=call_receipts,
        saved_row_events=len(all_geometry), saved_token_positions=sum(x["saved_positions"] for x in all_geometry),
        saved_scalar_components=sum(x["scalar_components"] for x in all_geometry),
        all_saved_bf16_elements_exact=True, all_assigned_timeline_checks_passed=True, genuine_decode_all_calls=True,
        behavior_scoring_performed=False, scientific_qualification=False, original_campaign_remains_failed=True,
        original_benchmark_admitted=False, paid_retry_authorized=False,
        limitations=["Evidence establishes observed repeatability for these five fixed calls and this preset, not which setting caused the prior failure or universal repeatability.",
            "Full-vocabulary and all-row residual identities rely on saved cryptographic fingerprints; raw BF16 reconstruction covers the prospectively saved first/last physical rows.",
            "Large model/adapter identities are bound through the frozen verifying loader and CPU prefetch receipts; no weights were loaded by this check.",
            "No behavior categories, correctness, target rates, effect estimates or experience claims are evaluated. The original campaign remains failed and benchmark unadmitted."])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs" / NAME)
    parser.add_argument("--out", type=Path, default=ROOT / "reports/pain-deterministic-diagnostic-v1-independent-check.json")
    args = parser.parse_args()
    receipt = check(args.run)
    write_once(args.out, receipt)
    print(json.dumps({k: receipt[k] for k in ("passed", "completed_calls", "completed_rows", "saved_row_events", "saved_token_positions", "saved_scalar_components")}))
