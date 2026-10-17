"""One admitted control phase using the unchanged original executor and preset.

This module imports no numerical library before verifying its immutable inputs.
The original executor writes its own manifest under ``original_execution``;
the outer receipt binds that evidence without modifying original scientific rows.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time

NAME = "qwen25-pain-deterministic-control-v1"
ORIGINAL_MANIFEST_SHA = "e87c70267be0b7813750dbef9d752f0855cf5b2db90e81f2381127d544b59a63"
ORIGINAL_SOURCE_SHA = "63bc3e330b25f5805724bfa5b8bf2f75d6fe37d048963524e1d28138e4bd4ff9"
ORIGINAL_ARCHIVE_SHA = "9c59998a8a162075a4adc201c4a374f821f76978b0c4ec518dce4630cd3f1fa7"
REFERENCE_MANIFEST_SHA = "fa75488f5047854d22f92a0e822c85bac600a259cfc2d65b14286c6bfd0bed8d"
REFERENCE_SUMMARY_SHA = "595839f83fc2032e893bf4c34187d4cc5817bec8dd729417ddb54630ec3a7941"
REFERENCE_CHECK_SHA = "895b4f828358ecc37c470ffb0d7641ca26515781b19f933e69107c6318cc4c25"
REFERENCE_WORKER_SHA = "fada4f67d5e6dce8728f85083e8a7bed0c1ae693693d4c8d9c546aea944342f1"
PRESET = dict(
    schema_version=1, cublas_workspace_config=":4096:8", deterministic_algorithms=True,
    deterministic_warn_only=False, sdpa_backends=["math"], cudnn_deterministic=True,
    cudnn_benchmark=False, cuda_matmul_allow_tf32=True, cudnn_allow_tf32=True,
    allow_bf16_reduced_precision_reduction=True, float32_matmul_precision="high",
    max_new_tokens=8,
    norm_comparison="CPU float64 versus observed GPU float32 descriptive only; no norm containment gate",
)
POLICIES = {phase: dict(phase=phase, app_name="curiosity-study-pain-deterministic-control-" + phase + "-h10096",
    gpu="H100", cpu=2, memory_gib=96, wall_seconds=wall, teardown_seconds=120,
    rate_usd_per_second=0.00133632, maximum_liability_usd=(wall + 120) * 0.00133632,
    max_containers=1, retries=0) for phase, wall in (("setup", 900), ("benchmark", 1800))}
ASSIGNED_CALLS = dict(setup=21, benchmark=65)
ASSIGNED_ROWS = dict(setup=612, benchmark=2020)
PRIOR_PATHS = (
    "admissions/setup.json", "original_execution/phases/setup/summary.json",
    "phases/setup/phase-execution.json", "phases/setup/audit.json", "launches/setup-outcome.json",
)


def require(value, message):
    if not value:
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
    with path.open("xb") as stream:
        stream.write((json.dumps(value, sort_keys=True, ensure_ascii=False,
                                separators=(",", ":")) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


def verify_files(root, inventory):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink() and isinstance(inventory, dict) and bool(inventory),
            "Missing immutable file tree")
    require(not any(p.is_symlink() for p in root.rglob("*")), "Immutable tree contains symlinks")
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    require(actual == set(inventory), "Immutable file inventory changed")
    for name, expected in inventory.items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts and str(relative) == name
                and sha(root / relative) == expected, "Immutable file bytes changed")


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def prior_setup(run, manifest, admission):
    """Require the exact controller-bound independently audited setup chain."""
    pins = admission["prior_evidence"]
    require(set(pins) == set(PRIOR_PATHS), "Incomplete benchmark admission evidence")
    for name, expected in pins.items():
        require((run / name).is_file() and not (run / name).is_symlink()
                and sha(run / name) == expected, "Prior setup evidence changed")
    previous, summary, receipt, audit, launch = [read(run / name) for name in PRIOR_PATHS]
    require(previous["phase"] == "setup" and previous["manifest_sha256"] == digest(manifest)
            and previous["phase_consumed"] is True and previous["retry_allowed"] is False
            and previous["policy"] == POLICIES["setup"] and previous["created"] < admission["created"],
            "Wrong prior setup admission")
    require(summary["manifest_sha256"] == ORIGINAL_MANIFEST_SHA and summary["phase"] == "setup"
            and summary["complete"] is summary["passed"] is summary["integrity_passed"] is True
            and summary["assigned_calls"] == summary["observed_calls"] == 21
            and summary["assigned_rows"] == 612, "Original-engine setup did not pass")
    require(receipt["outer_manifest_sha256"] == digest(manifest) and receipt["phase"] == "setup"
            and receipt["admission_sha256"] == digest(previous)
            and receipt["complete"] is receipt["passed"] is receipt["integrity_passed"] is True
            and receipt["original_summary"] == binding(run, run / PRIOR_PATHS[1]),
            "Outer setup execution was not admitted")
    require(audit["manifest_sha256"] == digest(manifest) and audit["phase"] == "setup"
            and audit["complete"] is audit["passed"] is True,
            "Independent new setup audit did not pass")
    prediction = audit["benchmark_forecast_seconds"]
    require(finite(prediction) and 0 < prediction <= 1440
            and receipt["forecast"]["passed"] is True
            and receipt["forecast"]["limit_seconds"] == 1440
            and prediction == receipt["forecast"]["benchmark_seconds"]
            == summary["forecast"]["benchmark_seconds"], "New setup forecast exceeds 1440 seconds or differs")
    require(launch["all_stopped_zero_tasks"] is True and launch["export_error"] is None
            and launch["failure"] is None, "Setup was not exported and stopped")


def validate_payload(original_root, run, phase):
    original_root, run = Path(original_root), Path(run)
    require(phase in POLICIES and run.name == NAME, "Wrong control/phase")
    manifest, admission = read(run / "manifest.json"), read(run / "admissions" / (phase + ".json"))
    original, frozen = manifest["original_manifest"], manifest["original_freeze"]
    require(manifest["name"] == NAME and digest(original) == frozen["manifest_sha256"] == ORIGINAL_MANIFEST_SHA
            and original["source_sha256"] == frozen["source_sha256"] == ORIGINAL_SOURCE_SHA
            and manifest["original_source_archive_sha256"] == frozen["archive_sha256"] == ORIGINAL_ARCHIVE_SHA,
            "Original reference identity changed")
    require(digest(manifest["preset"]) == digest(PRESET) and digest(manifest["resources"]) == digest(POLICIES)
            and digest(manifest["assigned_calls"]) == digest(ASSIGNED_CALLS)
            and digest(manifest["assigned_rows"]) == digest(ASSIGNED_ROWS)
            and manifest["campaign_ceiling_usd"] == 5 and manifest["benchmark_forecast_limit_seconds"] == 1440
            and manifest["original_campaign_remains_failed"] is True
            and manifest["scientific_rules_changed"] is False and manifest["automatic_retry_allowed"] is False,
            "Fixed scientific/numerical/resource contract changed")
    require(admission["phase"] == phase and admission["manifest_sha256"] == digest(manifest)
            and digest(admission["policy"]) == digest(POLICIES[phase])
            and admission["phase_consumed"] is True and admission["retry_allowed"] is False
            and finite(admission["created"]) and finite(admission["deadline"])
            and admission["deadline"] == admission["created"] + POLICIES[phase]["wall_seconds"]
            and admission["created"] <= time.time() < admission["deadline"] - 60
            and admission["maximum_liability_usd"] == POLICIES[phase]["maximum_liability_usd"],
            "Invalid or expired phase admission")
    verify_files(original_root, manifest["original_execution_files"])
    source_root = original_root / "src/curiosity_study"
    require(digest({str(p.relative_to(source_root)): p.read_text() for p in sorted(source_root.rglob("*.py"))})
            == ORIGINAL_SOURCE_SHA, "Original source inventory differs")
    diagnostic = Path(__file__).resolve().parent
    require(all(Path(name).name == name for name in manifest["diagnostic_files"]), "Diagnostic files must be flat")
    verify_files(diagnostic, manifest["diagnostic_files"])
    require(manifest["diagnostic_files"].get(Path(__file__).name) == sha(__file__)
            and manifest["diagnostic_files"].get("diagnostic_reference_worker.py") == REFERENCE_WORKER_SHA,
            "Missing worker or successful-preset helper binding")
    reference = manifest["deterministic_reference"]
    require(reference == dict(manifest_sha256=REFERENCE_MANIFEST_SHA,
            summary_byte_sha256=REFERENCE_SUMMARY_SHA, independent_check_byte_sha256=REFERENCE_CHECK_SHA),
            "Successful diagnostic reference changed")
    prior_manifest = read(diagnostic / "reference-manifest.json")
    prior_summary = read(diagnostic / "reference-summary.json")
    prior_check = read(diagnostic / "reference-independent-check.json")
    require(digest(prior_manifest) == REFERENCE_MANIFEST_SHA
            and sha(diagnostic / "reference-summary.json") == REFERENCE_SUMMARY_SHA
            and sha(diagnostic / "reference-independent-check.json") == REFERENCE_CHECK_SHA
            and prior_summary["manifest_sha256"] == prior_check["manifest_sha256"] == REFERENCE_MANIFEST_SHA
            and prior_check["summary_byte_sha256"] == REFERENCE_SUMMARY_SHA
            and prior_summary["passed"] is prior_summary["complete"] is True
            and prior_check["passed"] is prior_check["complete"] is True
            and prior_summary["completed_rows"] == prior_check["completed_rows"] == 148,
            "Successful diagnostic evidence does not reconcile")
    if phase == "benchmark":
        prior_setup(run, manifest, admission)
    else:
        require(admission["prior_evidence"] == {}, "Setup must not substitute another setup")
    return manifest, admission


def load_reference():
    path = Path(__file__).resolve().parent / "diagnostic_reference_worker.py"
    require(sha(path) == REFERENCE_WORKER_SHA, "Successful diagnostic helper changed")
    spec = importlib.util.spec_from_file_location("_immutable_deterministic_reference_worker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require({k: v for k, v in module.PRESET.items() if k not in {"wall_seconds", "execution_reserve_seconds"}}
            == PRESET, "Successful preset changed")
    return module


def binding(run, path):
    return dict(path=str(path.relative_to(run)), byte_sha256=sha(path), content_sha256=digest(read(path)))


def forecast(inner, original, execution):
    identity = read(inner / "phases/setup/identity.json")
    calls = execution.setup_calls(original)
    primary = [call for call in calls if call["role"] == "primary"]
    require(len(primary) == 10, "Original primary timing inventory changed")
    values = []
    for call in primary:
        path = inner / "phases/setup/calls" / f"{call['call_index']:04d}.json"
        if not path.is_file():
            return dict(benchmark_seconds=None, limit_seconds=1440, passed=False)
        value = read(path)
        require(value["call_spec"] == call and finite(value["elapsed_seconds"])
                and value["elapsed_seconds"] >= 0, "Invalid original measured call timing")
        values.append(value["elapsed_seconds"])
    require(finite(identity["load_seconds"]) and identity["load_seconds"] >= 0, "Invalid model loading time")
    total = identity["load_seconds"] + 2 * max(values) * 65 + 120
    return dict(benchmark_seconds=total, limit_seconds=1440, passed=total <= 1440,
                load_seconds=identity["load_seconds"], slowest_primary_cohort_seconds=max(values),
                safety_multiplier=2, benchmark_cohorts=65, export_seconds=120)


def checkpoint_marker(run, phase):
    path = Path(run) / "transport" / phase / "checkpoint.request"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(time.monotonic_ns()))


def execute(original_root, run, cache, phase, checkpoint=None):
    original_root, run = Path(original_root), Path(run)
    manifest, admission = validate_payload(original_root, run, phase)
    outer = run / "phases" / phase
    require(not outer.exists(), "Control phase already consumed; no retry or resume")
    outer.mkdir(parents=True)
    write_once(outer / "started.json", dict(phase=phase, outer_manifest_sha256=digest(manifest),
        admission_sha256=digest(admission), created_unix=time.time(), retry_allowed=False))
    original, inner = manifest["original_manifest"], run / "original_execution"
    for name, value in (("manifest.json", original), ("freeze.json", manifest["original_freeze"])):
        path = inner / name
        if path.exists():
            require(not path.is_symlink() and digest(read(path)) == digest(value), "Original inner binding changed")
        else:
            write_once(path, value)
    directory = inner / "phases" / phase
    deadline = admission["deadline"] - 60
    summary, error, flags, after, prediction = None, None, None, None, None
    execution = None
    try:
        reference = load_reference()
        torch, modules = reference.load_original(original_root, manifest)
        assets, execution = modules["pain_profile_assets"], modules["pain_profile_execution"]
        with reference.deterministic_runtime(torch) as flags:
            environment = assets.check_environment(original_root / "configs/pain_profile_requirements.lock")
            require(dict(path="configs/pain_profile_requirements.lock", **environment) == original["environment_lock"],
                    "Original pinned environment differs")
            require(time.time() < deadline, "No execution time remains")
            tokenizer, base, model, layers, vectors, identity = assets.load_model(original_root, cache, deadline)
            require(reference.effective_preset(torch) == flags, "Model loading changed deterministic settings")
            identity.update(admission_sha256=digest(admission), outer_manifest_sha256=digest(manifest),
                original_manifest_sha256=digest(original), source_sha256=original["source_sha256"],
                source_archive_sha256=manifest["original_source_archive_sha256"],
                execution_files_sha256=digest(manifest["original_execution_files"]),
                deterministic_preset=flags, deterministic_reference=manifest["deterministic_reference"],
                modal_image_id=os.environ.get("PAIN_PROFILE_IMAGE_ID"))

            def commit():
                require(reference.effective_preset(torch) == flags, "Deterministic policy changed during execution")
                if checkpoint is not None:
                    checkpoint()

            summary = execution.execute(phase, inner, original, model, tokenizer, layers, vectors,
                                        identity, deadline, checkpoint=commit)
            after = reference.effective_preset(torch)
            require(after == flags, "Deterministic policy changed after execution")
            if phase == "setup":
                prediction = forecast(inner, original, execution)
                require(prediction["benchmark_seconds"] == summary["forecast"]["benchmark_seconds"],
                        "Measured forecast disagrees with original executor")
    except Exception as caught:
        error = dict(type=type(caught).__name__, message=str(caught))
        if (directory / "summary.json").is_file():
            summary = read(directory / "summary.json")
    artifacts = {key: binding(run, directory / name) if (directory / name).is_file() else None
                 for key, name in (("original_summary", "summary.json"), ("original_started", "started.json"),
                                   ("original_identity", "identity.json"), ("original_outcomes", "outcomes.json"))}
    complete = summary is not None and summary.get("complete") is True
    integrity = complete and error is None and summary.get("integrity_passed") is True and after == flags
    passed = integrity and summary.get("passed") is True and (phase == "benchmark" or bool(prediction and prediction["passed"]))
    receipt = dict(schema_version=1, phase=phase, outer_manifest_sha256=digest(manifest),
        manifest_sha256=digest(manifest), admission_sha256=digest(admission), original_manifest_sha256=digest(original),
        original_source_sha256=original["source_sha256"], original_execution_root="original_execution",
        worker_byte_sha256=sha(__file__), reference_worker_byte_sha256=REFERENCE_WORKER_SHA,
        effective_preset=flags, final_effective_preset=after, execution_deadline_unix=deadline,
        assigned_calls=ASSIGNED_CALLS[phase], assigned_rows=ASSIGNED_ROWS[phase],
        observed_calls=0 if summary is None else summary["observed_calls"],
        completed_rows=0 if summary is None else summary["completed_rows"],
        expected_fault_rows=0 if summary is None else summary["expected_fault_rows"],
        missing_rows=ASSIGNED_ROWS[phase] if summary is None else summary["missing_rows"],
        execution_error_rows=0 if summary is None else summary["execution_error_rows"],
        complete=complete, integrity_passed=integrity, passed=passed, error=error, forecast=prediction,
        original_campaign_remains_failed=True, original_scientific_rules_unchanged=True,
        geometry_policy="Exact BF16 elements; CPU float64 norms descriptive only in independent audit",
        independent_audit_required=True, behavior_gates_used=False, retry_allowed=False,
        finished_unix=time.time(), **artifacts)
    write_once(outer / "phase-execution.json", receipt)
    if checkpoint is not None:
        checkpoint()
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--phase", choices=list(POLICIES), required=True)
    args = parser.parse_args()
    result = execute(args.original_root, args.run, args.cache, args.phase,
                     checkpoint=lambda: checkpoint_marker(args.run, args.phase))
    print(json.dumps({key: result[key] for key in ("phase", "complete", "passed", "assigned_rows", "missing_rows")}), flush=True)
    if result["passed"] is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
