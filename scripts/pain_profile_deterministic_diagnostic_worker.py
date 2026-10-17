"""One finite deterministic-backend diagnostic; no scientific scoring or retry.

Imports are stdlib-only until the required pre-process CUBLAS setting is checked.
All model/engine code is imported from the separately verified original archive.
"""

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time

NAME = "qwen25-pain-deterministic-diagnostic-v1"
ORIGINAL_MANIFEST_SHA = "e87c70267be0b7813750dbef9d752f0855cf5b2db90e81f2381127d544b59a63"
ORIGINAL_SOURCE_SHA = "63bc3e330b25f5805724bfa5b8bf2f75d6fe37d048963524e1d28138e4bd4ff9"
ORIGINAL_ARCHIVE_SHA = "9c59998a8a162075a4adc201c4a374f821f76978b0c4ec518dce4630cd3f1fa7"
PRESET = dict(
    schema_version=1, cublas_workspace_config=":4096:8", deterministic_algorithms=True,
    deterministic_warn_only=False, sdpa_backends=["math"], cudnn_deterministic=True,
    cudnn_benchmark=False, cuda_matmul_allow_tf32=True, cudnn_allow_tf32=True,
    allow_bf16_reduced_precision_reduction=True, float32_matmul_precision="high",
    max_new_tokens=8, wall_seconds=900, execution_reserve_seconds=60,
    norm_comparison="CPU float64 versus observed GPU float32 descriptive only; no norm containment gate",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def call_plan(original_manifest):
    preflight, plan = original_manifest["preflight"], original_manifest["plan"]
    require(preflight["longest_source_cohort"] == 12 and len(preflight["requests"]) == 404,
            "Original 404 requests and longest cohort12 required")
    requests = {row["base_request_id"]: row for row in preflight["requests"]}
    require(len(requests) == 404, "Duplicate original request")
    result, baselines = [], []
    labels = ("unhooked", "zero_hook", "pain_full_stressor", "unhooked_repeat", "zero_hook_repeat")
    for index, label in enumerate(labels):
        cohort, coefficient = (12, 1.0) if index == 2 else (0, 0.0)
        ids = [row["base_request_id"] for row in plan["base_requests"] if row["source_cohort"] == cohort]
        require(len(ids) == (20 if cohort == 12 else 32) and all(key in requests for key in ids),
                "Original companions changed")
        call_id = f"deterministic-{index:04d}"
        call = dict(call_index=index, call_id=call_id, label=label, source_cohort=cohort,
                    base_request_ids=ids, trial_ids=[f"{call_id}/{key}" for key in ids],
                    coefficient=coefficient, steering_enabled=index not in (0, 3),
                    prefill_policy="full_prefix", evidence_mode="compatibility",
                    baseline_reference_call_ids=list(baselines) if coefficient == 0 else [])
        if coefficient == 0:
            baselines.append(call_id)
        result.append(call)
    require(sum(len(call["trial_ids"]) for call in result) == 148, "Fixed 148-slot inventory changed")
    return result


def verify_files(root, files):
    require(isinstance(files, dict) and bool(files), "Missing immutable file inventory")
    for name, checksum in files.items():
        relative = Path(name)
        path = Path(root) / relative
        require(not relative.is_absolute() and ".." not in relative.parts and str(relative) == name
                and path.is_file() and not path.is_symlink() and sha(path) == checksum,
                "Immutable diagnostic/original file changed")


def validate_payload(original_root, run):
    original_root, run = Path(original_root), Path(run)
    manifest, admission = read(run / "manifest.json"), read(run / "admission.json")
    original, frozen = manifest["original_manifest"], manifest["original_freeze"]
    require(run.name == NAME and digest(original) == frozen["manifest_sha256"] == ORIGINAL_MANIFEST_SHA
            and original["source_sha256"] == frozen["source_sha256"] == ORIGINAL_SOURCE_SHA
            and manifest["original_source_archive_sha256"] == frozen["archive_sha256"] == ORIGINAL_ARCHIVE_SHA,
            "Wrong original campaign/source identity")
    require(manifest["preset"] == PRESET and manifest["call_plan"] == call_plan(original),
            "Fixed preset/call plan changed")
    require(admission["manifest_sha256"] == digest(manifest)
            and admission["phase_consumed"] is True and admission["retry_allowed"] is False
            and type(admission["created"]) in (int, float) and math.isfinite(admission["created"])
            and type(admission["deadline"]) in (int, float) and math.isfinite(admission["deadline"])
            and admission["deadline"] == admission["created"] + 900
            and admission["created"] <= time.time() < admission["deadline"]
            and admission["maximum_liability_usd"] == 1.3630464,
            "Invalid or expired diagnostic admission")
    verify_files(original_root, manifest["original_execution_files"])
    source_root = original_root / "src/curiosity_study"
    source = {str(path.relative_to(source_root)): path.read_text() for path in sorted(source_root.rglob("*.py"))}
    require(digest(source) == ORIGINAL_SOURCE_SHA, "Original source inventory changed")
    verify_files(Path(__file__).resolve().parent, manifest["diagnostic_files"])
    require(manifest["diagnostic_files"].get(Path(__file__).name) == sha(__file__),
            "Worker byte binding missing")
    return manifest, admission


def load_original(original_root, manifest):
    require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == PRESET["cublas_workspace_config"],
            "CUBLAS_WORKSPACE_CONFIG must be fixed by the controller before process start")
    sys.path.insert(0, str(Path(original_root) / "src"))
    modules = {}
    for name in ("artifacts", "pain_profile_assets", "pain_profile_engine", "pain_profile_execution"):
        module = importlib.import_module("curiosity_study." + name)
        relative = "src/curiosity_study/" + name + ".py"
        require(Path(module.__file__).resolve() == (Path(original_root) / relative).resolve()
                and sha(module.__file__) == manifest["original_execution_files"][relative],
                "Execution must import the immutable original helper")
        modules[name] = module
    torch = importlib.import_module("torch")
    require(not torch.cuda.is_initialized(), "CUDA initialized before deterministic preset")
    return torch, modules


def effective_preset(torch):
    cuda = torch.backends.cuda
    return dict(cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        sdpa_backends={name: getattr(cuda, name + "_sdp_enabled")() for name in ("math", "flash", "mem_efficient", "cudnn")},
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cuda_matmul_allow_tf32=cuda.matmul.allow_tf32, cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        allow_bf16_reduced_precision_reduction=cuda.matmul.allow_bf16_reduced_precision_reduction,
        float32_matmul_precision=torch.get_float32_matmul_precision())


@contextmanager
def deterministic_runtime(torch):
    require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8" and not torch.cuda.is_initialized(),
            "Deterministic settings must precede CUDA initialization")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.set_float32_matmul_precision("high")
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        observed = effective_preset(torch)
        require(observed == dict(
            cublas_workspace_config=":4096:8", deterministic_algorithms=True, deterministic_warn_only=False,
            sdpa_backends=dict(math=True, flash=False, mem_efficient=False, cudnn=False),
            cudnn_deterministic=True, cudnn_benchmark=False, cuda_matmul_allow_tf32=True,
            cudnn_allow_tf32=True, allow_bf16_reduced_precision_reduction=True, float32_matmul_precision="high"),
            "Required fixed deterministic backend is unsupported")
        yield observed


def timeline_checks(result, items, preflight):
    report = result["evidence"]
    frames, width = report["forwards"], max(len(item["prompt_ids"]) for item in items)
    pad, eos = preflight["pad_token_id"], set(preflight["eos_token_ids"])
    ids = [[pad] * (width - len(item["prompt_ids"])) + item["prompt_ids"] for item in items]
    masks = [[0] * (width - len(item["prompt_ids"])) + [1] * len(item["prompt_ids"]) for item in items]
    positions = [[0] * (width - len(item["prompt_ids"])) + list(range(len(item["prompt_ids"]))) for item in items]
    alive, tokens, rng = [True] * len(items), [[] for _ in items], None
    for index, frame in enumerate(frames):
        require(frame["input_ids"] == ids and frame["attention_mask"] == masks and frame["position_ids"] == positions
                and frame["cache_length_before"] == (0 if index == 0 else width + index - 1)
                and frame["cache_length_after"] == width + index, "Input/cache/position timeline differs")
        sample = frame["sampling"]
        require(sample["alive_before"] == alive and sample["uniform_draws_per_row"] == [1] * len(items)
                and (rng is None or sample["rng_state_before_sha256"] == rng), "Sampler activity/RNG chain differs")
        rng = sample["rng_state_after_sha256"]
        sampled = sample["sampled_before_alive_mask"]
        selected = [sampled[row] if active else pad for row, active in enumerate(alive)]
        require(selected == sample["selected_token_ids"], "Inactive row forwarding differs")
        for row, active in enumerate(alive):
            if active:
                tokens[row].append(selected[row])
        alive = [active and selected[row] not in eos and len(tokens[row]) < 8 for row, active in enumerate(alive)]
        require(alive == sample["alive_after"], "EOS/cap activity differs")
        ids = [[token] for token in selected]
        masks = [mask + [1] for mask in masks]
        positions = [[len(item["prompt_ids"]) + index] for item in items]
    require(bool(frames) and not any(alive) and [row["answer_token_ids"] for row in result["records"]] == tokens,
            "Incomplete raw-token timeline")
    return True


def vector_checks(torch, result, direction, items):
    """Exact BF16 samples; cross-device norm reductions are descriptive only."""
    import numpy as np

    descriptions, used = [], set()
    direction = direction.detach().to(device="cpu", dtype=torch.bfloat16)
    for frame in result["evidence"]["forwards"]:
        width = len(frame["input_ids"][0])
        require(frame["activation_dtypes"] == {"before": "torch.bfloat16", "after": "torch.bfloat16"},
                "Actual activation dtype changed")
        require(len(frame["geometry"]) == len(items), "Missing all-row geometry")
        for row_index, geometry in enumerate(frame["geometry"]):
            coefficient_value = result["evidence"]["coefficient"]
            active_count = (len(items[row_index]["prompt_ids"]) if frame["phase"] == "prefill" else 1) if coefficient_value else 0
            padding = width - len(items[row_index]["prompt_ids"]) if frame["phase"] == "prefill" else 0
            require(geometry["trial_id"] == items[row_index]["trial_id"]
                    and geometry["coefficient"] == coefficient_value
                    and geometry["active_positions"] == active_count
                    and geometry["unmodified_positions"] == width - active_count
                    and geometry["padding_positions"] == padding, "Observed geometry row/mask binding differs")
            saved = row_index in {0, len(items) - 1}
            require(all(("saved_" + label in geometry) == saved for label in ("before", "after")),
                    "Saved residual sample inventory differs")
            if not saved:
                continue
            values = {}
            for label in ("before", "after"):
                binding = geometry["saved_" + label]
                key = binding["key"]
                require(key == f"f{frame['forward_index']:02d}_r{row_index:02d}_{label}", "Saved sample key differs")
                value = result["arrays"][key]
                require(value.dtype == np.float32 and value.shape == (width, direction.numel())
                        and binding["positions"] == list(range(width)) and np.isfinite(value).all(),
                        "Invalid complete saved residual sample")
                actual_binding = dict(dtype=str(value.dtype), shape=list(value.shape),
                    byte_sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())
                require(binding == dict(key=key, positions=list(range(width)), **actual_binding)
                        and result["evidence"]["arrays"][key] == actual_binding,
                        "Saved residual array content binding differs")
                tensor = torch.from_numpy(value.copy()).to(torch.bfloat16)
                require(np.array_equal(tensor.float().numpy(), value), "Saved sample is not exact BF16")
                tensor_binding = dict(dtype=str(tensor.dtype), shape=list(tensor.shape),
                    byte_sha256=hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
                require(geometry[label + "_sha256"] == digest(tensor_binding), "Full residual fingerprint differs")
                values[label], used = tensor, used | {key}
            before, after = values["before"], values["after"]
            active = torch.ones(width, dtype=torch.bfloat16)
            if frame["phase"] == "prefill":
                active[:width - len(items[row_index]["prompt_ids"])]=0
            coefficient = torch.tensor(geometry["coefficient"], dtype=torch.bfloat16)
            if coefficient == 0:
                active[:] = 0
            delta = active[:, None] * coefficient * direction[None, :]
            expected = before if coefficient == 0 else before + delta
            require(torch.equal(expected.view(torch.uint8), after.view(torch.uint8)),
                    "Saved before/after BF16 elementwise arithmetic differs")
            actual = (after.double() - before.double())[active.bool()]
            norms = actual.norm(dim=-1)
            descriptions.append(dict(forward_index=frame["forward_index"], row=row_index,
                gpu_fp32_norm_min=geometry["achieved_delta_norm_min"],
                gpu_fp32_norm_max=geometry["achieved_delta_norm_max"],
                cpu_float64_norm_min=float(norms.min()) if norms.numel() else 0.0,
                cpu_float64_norm_max=float(norms.max()) if norms.numel() else 0.0,
                cross_device_norms_used_as_gate=False))
    require(used == set(result["arrays"]), "Unexpected or unverified saved residual arrays")
    return descriptions


def before_hashes(result):
    return [[row["before_sha256"] for row in frame["geometry"]] for frame in result["evidence"]["forwards"]]


def _save_arrays(path, arrays, engine):
    import numpy as np

    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    return dict(path=Path(path).name, byte_sha256=sha(path),
                arrays={key: engine.array_binding(value) for key, value in arrays.items()})


def execute(original_root, run, cache_root, checkpoint=None):
    original_root, run = Path(original_root), Path(run)
    manifest, admission = validate_payload(original_root, run)
    require(not (run / "started.json").exists(), "Diagnostic already consumed; no retry/resume")
    calls, original = manifest["call_plan"], manifest["original_manifest"]
    deadline = admission["deadline"] - 60
    write_once(run / "started.json", dict(manifest_sha256=digest(manifest), admission_sha256=digest(admission),
        started_unix=time.time(), execution_deadline=deadline, call_plan_sha256=digest(calls)))
    (run / "calls").mkdir(exist_ok=False)
    completed, observed, prior, error, identity, initial_settings = {}, [], {}, None, None, None
    try:
        torch, modules = load_original(original_root, manifest)
        assets, engine, execution = [modules[name] for name in ("pain_profile_assets", "pain_profile_engine", "pain_profile_execution")]
        with deterministic_runtime(torch) as initial_settings:
            require(time.time() < deadline, "Diagnostic execution deadline reached before model loading")
            tok, base, model, layers, vectors, identity = assets.load_model(original_root, cache_root, deadline)
            require(identity["environment"] == {k: v for k, v in original["environment_lock"].items() if k != "path"},
                    "Pinned model package environment changed")
            execution.validate_manifest(original, tok)
            execution.validate_runtime(model, vectors, identity)
            require(effective_preset(torch) == initial_settings, "Loading changed deterministic settings")
            identity = dict(identity, manifest_sha256=digest(manifest), admission_sha256=digest(admission),
                            deterministic_preset=initial_settings)
            write_once(run / "identity.json", identity)
            requests = {row["base_request_id"]: row for row in original["preflight"]["requests"]}
            for call in calls:
                require(time.time() < deadline, "Diagnostic execution deadline reached before next call")
                items = [dict(deepcopy(requests[key]), trial_id=trial) for key, trial in zip(
                    call["base_request_ids"], call["trial_ids"], strict=True)]
                began, result, call_error, checks, descriptive = time.monotonic(), None, None, {}, []
                try:
                    result = engine.generate_segment(model, tok, layers, items=items,
                        direction_bf16=vectors["pain"], pain_unit_float32=vectors["pain_unit_float32"],
                        coefficient=call["coefficient"], prefill_policy="full_prefix",
                        eos_token_ids=original["preflight"]["eos_token_ids"],
                        pad_token_id=original["preflight"]["pad_token_id"], evidence_mode="compatibility",
                        steering_enabled=call["steering_enabled"])
                    report = result["evidence"]
                    checks = dict(completed=report["complete"] is True and len(result["records"]) == len(items),
                        bound_call_policy=report["coefficient"] == call["coefficient"]
                            and report["steering_enabled"] is call["steering_enabled"]
                            and report["prefill_policy"] == "full_prefix" and report["evidence_mode"] == "compatibility"
                            and report["items_sha256"] == digest(items),
                        engine_elementwise_geometry=report["integrity_passed"] is True and all(report["checks"].values()),
                        hooks_empty=report["hooks_before"] == report["hooks_after"] == {},
                        genuine_active_decode=any(frame["phase"] == "decode" and any(frame.get("sampling", {}).get("alive_before", []))
                                                  for frame in report["forwards"]),
                        fixed_runtime=effective_preset(torch) == initial_settings)
                    if checks["completed"] and checks["engine_elementwise_geometry"]:
                        checks["input_cache_rng_timeline"] = timeline_checks(result, items, original["preflight"])
                        descriptive = vector_checks(torch, result, vectors["pain"], items)
                        checks["independent_bf16_samples"] = True
                    else:
                        checks.update(input_cache_rng_timeline=False, independent_bf16_samples=False)
                    if call["coefficient"]:
                        checks["active_nonzero_decode"] = execution.active_decode(result)
                    for ref in call["baseline_reference_call_ids"]:
                        checks["source_repeat:" + ref] = execution.repeat_equal(prior[ref], result)
                        checks["before_hook_identity:" + ref] = before_hashes(prior[ref]) == before_hashes(result)
                    prior[call["call_id"]] = result
                except Exception as caught:
                    call_error = dict(type=type(caught).__name__, message=str(caught))
                    checks["call_completed_without_exception"] = False
                arrays = {} if result is None else result.pop("arrays")
                path = run / "calls" / f"{call['call_index']:04d}"
                array_artifact = _save_arrays(path.with_suffix(".npz"), arrays, engine)
                records = [] if result is None else result["records"]
                require(not records or [row["trial_id"] for row in records] == call["trial_ids"], "Observed row IDs differ")
                completed.update({row["trial_id"]: dict(status="completed", raw_record_sha256=digest(row)) for row in records})
                for trial in call["trial_ids"]:
                    completed.setdefault(trial, dict(status="execution_error", raw_record_sha256=None))
                passed = bool(checks) and all(checks.values()) and call_error is None
                envelope = dict(schema_version=1, manifest_sha256=digest(manifest), admission_sha256=digest(admission),
                    call_spec=call, items_sha256=digest(items), engine_result=result, arrays_artifact=array_artifact,
                    checks=checks, passed=passed, error=call_error, descriptive_norms=descriptive,
                    elapsed_seconds=time.monotonic() - began)
                write_once(path.with_suffix(".json"), envelope)
                observed.append(dict(call_id=call["call_id"], content_sha256=digest(envelope), passed=passed))
                if checkpoint:
                    checkpoint()
                require(passed, "Diagnostic call failed: " + call["call_id"])
            require(time.time() <= deadline, "Diagnostic execution exceeded reserved deadline")
    except Exception as caught:
        error = dict(type=type(caught).__name__, message=str(caught))
    outcomes = [dict(trial_id=trial, call_id=call["call_id"],
                     **completed.get(trial, dict(status="missing", raw_record_sha256=None)))
                for call in calls for trial in call["trial_ids"]]
    require(len(outcomes) == 148, "Assigned denominator changed")
    write_once(run / "outcomes.json", outcomes)
    summary = dict(schema_version=1, assay="pain_profile_deterministic_diagnostic", manifest_sha256=digest(manifest),
        admission_sha256=digest(admission), original_manifest_sha256=ORIGINAL_MANIFEST_SHA,
        original_source_sha256=ORIGINAL_SOURCE_SHA, original_failed_gate_preserved=True,
        assigned_calls=5, observed_calls=len(observed), assigned_rows=148,
        completed_rows=sum(row["status"] == "completed" for row in outcomes),
        execution_error_rows=sum(row["status"] == "execution_error" for row in outcomes),
        missing_rows=sum(row["status"] == "missing" for row in outcomes),
        complete=len(observed) == 5, passed=error is None and len(observed) == 5 and all(row["passed"] for row in observed),
        error=error, calls=observed, outcomes_sha256=digest(outcomes), effective_preset=initial_settings,
        identity_sha256=digest(identity) if identity else None, finished_unix=time.time(),
        behavior_scoring_performed=False, scientific_admission=False, retry_authorized=False,
        norm_threshold_used=False, evidence_scope="All-row frozen-engine elementwise guards; independently reconstructed complete saved rows0 andlast; CPU float64 norms descriptive only")
    write_once(run / "summary.json", summary)
    if checkpoint:
        checkpoint()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()

    def checkpoint():
        (args.run / "checkpoint.request").write_text(str(time.monotonic_ns()))

    result = execute(args.original_root, args.run, args.cache, checkpoint)
    print(json.dumps(result), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
