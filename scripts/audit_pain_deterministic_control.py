"""Independent wrapper for the separately declared deterministic-backend control.

Original parsing, all-assigned denominators, timeline/repeat checks and scientific
analysis are loaded from immutable original bytes. Only explicit outer identity
nesting, canonical adapter names, deterministic preset binding, and the new exact
BF16/descriptive-norm geometry method are adapted. The old campaign stays failed.
"""

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile
import tempfile

import numpy as np

from curiosity_study.artifacts import digest, read, write_once

ROOT = Path(__file__).resolve().parents[1]
NAME = "qwen25-pain-deterministic-control-v1"
ORIGINAL_NAME = "qwen25-pain-profile-control-v1"
ORIGINAL_MANIFEST_SHA = "e87c70267be0b7813750dbef9d752f0855cf5b2db90e81f2381127d544b59a63"
SOURCE_SHA = "63bc3e330b25f5805724bfa5b8bf2f75d6fe37d048963524e1d28138e4bd4ff9"
ARCHIVE_SHA = "9c59998a8a162075a4adc201c4a374f821f76978b0c4ec518dce4630cd3f1fa7"
AUDITOR_SHA = "ebacd48847f51754f9becd0afe20ca21103c2963e8ee97765818723a41432bc3"
ANALYZER_SHA = "c0158b2846e65108a2e245c805ac4634a83cb515d6e69fe0b4c1ca8df87882da"
REFERENCE_SHA = "fa75488f5047854d22f92a0e822c85bac600a259cfc2d65b14286c6bfd0bed8d"
REFERENCE_CHECK_SHA = "895b4f828358ecc37c470ffb0d7641ca26515781b19f933e69107c6318cc4c25"
REFERENCE_WORKER_SHA = "fada4f67d5e6dce8728f85083e8a7bed0c1ae693693d4c8d9c546aea944342f1"
AUDITOR = "scripts/audit_pain_profile_control.py"
ANALYZER = "scripts/analyze_pain_profile_control.py"
METHOD = "strict_deterministic_exact_bf16_descriptive_float64_norms_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def array_binding(value):
    value = np.ascontiguousarray(value)
    return dict(dtype=str(value.dtype), shape=list(value.shape), byte_sha256=hashlib.sha256(value.tobytes()).hexdigest())


def tensor_binding(value):
    import torch
    value = value.detach().contiguous().cpu()
    return dict(dtype=str(value.dtype), shape=list(value.shape),
                byte_sha256=hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest())


def canonical_adapter_files(raw):
    result = {}
    for name, checksum in raw.items():
        path = PurePosixPath(name)
        clean = str(path)
        require(not path.is_absolute() and ".." not in path.parts and clean not in ("", ".")
                and name in (clean, "./" + clean) and clean not in result and valid_sha(checksum),
                "Adapter canonical path collision, changed hash or invalid name")
        result[clean] = checksum
    require(bool(result), "Missing adapter inventory")
    return result


def effective_preset(preset):
    keys = ("cublas_workspace_config", "deterministic_algorithms", "deterministic_warn_only",
            "cudnn_deterministic", "cudnn_benchmark", "cuda_matmul_allow_tf32", "cudnn_allow_tf32",
            "allow_bf16_reduced_precision_reduction", "float32_matmul_precision")
    return dict(**{key: preset[key] for key in keys},
                sdpa_backends=dict(math=True, flash=False, mem_efficient=False, cudnn=False))


def geometry(frame, arrays, items, direction, coefficients, policy, mode, used_arrays, samples):
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
        require(type(row["requested_delta_norm_bf16"]) in (int, float)
                and math.isfinite(row["requested_delta_norm_bf16"]) and row["requested_delta_norm_bf16"] >= 0,
                "Requested BF16 norm descriptor is unavailable")
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
        values = {}
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
            values[label] = value
            if len(positions) == width:
                require(row[label + "_sha256"] == digest(tensor_binding(value)), "Full saved residual fingerprint differs")
        before, after = values["before"], values["after"]
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
        achieved = (after.double() - before.double())[mask.bool()]
        norms = achieved.norm(dim=-1)
        samples.append(dict(forward_index=frame["forward_index"], phase=frame["phase"], physical_row=row_index,
            trial_id=item["trial_id"], sampled_positions=positions, scalar_components=before.numel(),
            exact_bf16=exact, finite=finite, untreated_exact=untouched,
            active_positions=int(mask.bool().sum()),
            cpu_float64_norm_min=float(norms.min()) if norms.numel() and finite else None,
            cpu_float64_norm_max=float(norms.max()) if norms.numel() and finite else None,
            gpu_float32_norm_min=row["achieved_delta_norm_min"],
            gpu_float32_norm_max=row["achieved_delta_norm_max"], norms_used_as_gate=False))

def check_files(root, inventory):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink() and not any(p.is_symlink() for p in root.rglob("*")),
            "Frozen tree contains a symlink")
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    require(actual == set(inventory), "Frozen file inventory differs")
    for name, checksum in inventory.items():
        path = PurePosixPath(name)
        require(not path.is_absolute() and ".." not in path.parts and str(path) == name
                and sha(root / name) == checksum, "Frozen file byte binding differs")


def frozen_inputs(run):
    """Verify this new package plus the unchanged scientific source archive."""
    run = Path(run).resolve()
    project = run.parent.parent
    require(run == project / "runs" / NAME, "Campaign alias is forbidden")
    manifest, frozen = read(run / "manifest.json"), read(run / "freeze.json")
    original_run = project / "runs" / ORIGINAL_NAME
    original = manifest["original_manifest"]
    require(manifest["name"] == NAME and manifest["schema_version"] == 1
            and manifest["purpose"] == "backend_adapted_behavioral_method_control"
            and manifest["original_campaign_remains_failed"] is True
            and manifest["scientific_rules_changed"] is manifest["automatic_retry_allowed"] is False,
            "New campaign scope changed")
    require(digest(original) == ORIGINAL_MANIFEST_SHA and read(original_run / "manifest.json") == original
            and manifest["original_freeze"] == read(original_run / "freeze.json")
            and original["source_sha256"] == SOURCE_SHA
            and manifest["original_source_archive_sha256"] == sha(original_run / "source.tar.gz") == ARCHIVE_SHA,
            "Original campaign scientific identity differs")
    inventory = {}
    with tarfile.open(original_run / "source.tar.gz") as archive:
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            require(member.isfile() and not path.is_absolute() and ".." not in path.parts
                    and str(path) == member.name and member.name not in inventory, "Unsafe original archive")
            inventory[member.name] = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    require(manifest["original_execution_files"] == inventory and inventory[AUDITOR] == AUDITOR_SHA
            and inventory[ANALYZER] == ANALYZER_SHA, "Original engine/audit/analysis inventory differs")
    execution = project / "executions" / NAME
    require(frozen == dict(manifest_sha256=digest(manifest), execution_root=str(execution)), "New freeze binding differs")
    check_files(execution / "original", inventory)
    check_files(execution / "diagnostic", manifest["diagnostic_files"])
    require(manifest["diagnostic_files"][Path(__file__).name] == sha(__file__)
            and manifest["diagnostic_files"]["diagnostic_reference_worker.py"] == REFERENCE_WORKER_SHA,
            "New auditor/reference preset bytes differ")
    diagnostic = execution / "diagnostic"
    reference = read(diagnostic / "reference-manifest.json")
    summary = read(diagnostic / "reference-summary.json")
    independent = read(diagnostic / "reference-independent-check.json")
    require(digest(reference) == REFERENCE_SHA
            and summary["manifest_sha256"] == independent["manifest_sha256"] == REFERENCE_SHA
            and summary["complete"] is summary["passed"] is independent["complete"] is independent["passed"] is True
            and summary["completed_rows"] == independent["completed_rows"] == 148
            and independent["missing_rows"] == independent["execution_error_rows"] == 0
            and sha(diagnostic / "reference-independent-check.json") == REFERENCE_CHECK_SHA
            and independent["summary_byte_sha256"] == sha(diagnostic / "reference-summary.json")
            and manifest["deterministic_reference"] == dict(manifest_sha256=REFERENCE_SHA,
                summary_byte_sha256=sha(diagnostic / "reference-summary.json"), independent_check_byte_sha256=REFERENCE_CHECK_SHA),
            "Successful diagnostic reference differs")
    preset = {k: v for k, v in reference["preset"].items() if k not in {"wall_seconds", "execution_reserve_seconds"}}
    require(manifest["preset"] == preset and independent["effective_preset"] == effective_preset(preset)
            and manifest["assigned_calls"] == dict(setup=21, benchmark=65)
            and manifest["assigned_rows"] == dict(setup=612, benchmark=2020)
            and manifest["campaign_ceiling_usd"] == 5 and manifest["benchmark_forecast_limit_seconds"] == 1440,
            "Inherited preset/denominators/ceiling differ")
    # Match exact resource types too: bool must not substitute for integer zero.
    policies = {phase: dict(phase=phase, app_name="curiosity-study-pain-deterministic-control-" + phase + "-h10096",
        gpu="H100", cpu=2, memory_gib=96, wall_seconds=seconds, teardown_seconds=120,
        rate_usd_per_second=0.00133632, maximum_liability_usd=(seconds + 120) * 0.00133632,
        max_containers=1, retries=0) for phase, seconds in (("setup", 900), ("benchmark", 1800))}
    require(digest(manifest["resources"]) == digest(policies), "New exact resource policy differs")
    return dict(run=run, project=project, manifest=manifest, freeze=frozen, original_run=original_run,
                execution=execution, inventory=inventory, policies=policies)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def original_runtime(inputs):
    """Private original modules; do not mutate an installed/frozen module file."""
    names = ("freeze_execution", "prepare_pain_profile_control", "analyze_pain_profile_control", "_deterministic_original_auditor")
    previous = {name: sys.modules.get(name) for name in names}
    old_bytecode = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        execution = inputs["execution"] / "original"
        _load(names[0], execution / "infra/freeze_execution.py")
        _load(names[1], execution / "scripts/prepare_pain_profile_control.py")
        analysis = _load(names[2], execution / ANALYZER)
        original = _load(names[3], execution / AUDITOR)
        provenance = original.source_provenance(inputs["original_run"], inputs["manifest"]["original_manifest"])
        for name, module in tuple(sys.modules.items()):
            if name.startswith("curiosity_study.pain_profile_") or name == "curiosity_study.artifacts":
                relative = "src/" + name.replace(".", "/") + ".py"
                require(relative in inputs["inventory"] and sha(module.__file__) == inputs["inventory"][relative],
                        "Loaded numerical helper differs from original source")
        provenance = dict(provenance, adapter_files=canonical_adapter_files(provenance["adapter_files"]))
        original_identity = original.runtime_identity
        samples = []

        def checked_identity(identity, manifest, proof):
            policy = original_identity(identity, manifest, proof)
            require(identity["outer_manifest_sha256"] == digest(inputs["manifest"])
                    and identity["original_manifest_sha256"] == ORIGINAL_MANIFEST_SHA
                    and identity["deterministic_reference"] == inputs["manifest"]["deterministic_reference"]
                    and identity["deterministic_preset"] == effective_preset(inputs["manifest"]["preset"]),
                    "Actual deterministic/outer identity differs")
            return dict(policy, deterministic_preset=identity["deterministic_preset"],
                        outer_manifest_sha256=identity["outer_manifest_sha256"])

        original.runtime_identity = checked_identity
        original._geometry = lambda *args: geometry(*args, samples=samples)
        yield original, analysis, provenance, samples
    finally:
        sys.dont_write_bytecode = old_bytecode
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def admission_binding(inputs, phase, *, reproduce_setup):
    require(phase in ("setup", "benchmark"), "Unknown phase")
    run, manifest = inputs["run"], inputs["manifest"]
    path = run / "admissions" / (phase + ".json")
    admission = read(path)
    central = inputs["project"] / "campaign_admissions" / NAME / path.name
    policy = inputs["policies"][phase]
    require(admission == read(central) and sha(path) == sha(central)
            and admission["phase"] == phase and admission["manifest_sha256"] == digest(manifest)
            and digest(admission["policy"]) == digest(policy)
            and admission["phase_consumed"] is True and admission["retry_allowed"] is False
            and type(admission["created"]) in (int, float) and math.isfinite(admission["created"])
            and admission["deadline"] == admission["created"] + policy["wall_seconds"]
            and admission["maximum_liability_usd"] == policy["maximum_liability_usd"],
            "Actual one-admission phase/deadline/resource binding differs")
    expected = {}
    if phase == "benchmark":
        setup_path = run / "phases/setup/audit.json"
        setup = reproduce_setup(run, "setup")
        require(read(setup_path) == setup and setup["complete"] is setup["passed"] is True
                and 0 < setup["benchmark_forecast_seconds"] <= 1440, "New setup audit does not reproduce/pass")
        names = ("admissions/setup.json", "original_execution/phases/setup/summary.json",
                 "phases/setup/phase-execution.json", "phases/setup/audit.json", "launches/setup-outcome.json")
        expected = {name: sha(run / name) for name in names}
        stopped = read(run / names[-1])
        require(stopped["all_stopped_zero_tasks"] is True and stopped["export_error"] is None
                and stopped["failure"] is None, "Setup transport/shutdown incomplete")
    require(admission["prior_evidence"] == expected, "Phase prerequisite byte bindings differ")
    return admission
def artifact_binding(run, path):
    return dict(path=str(path.relative_to(run)), byte_sha256=sha(path), content_sha256=digest(read(path)))


def execution_receipt(inputs, phase, admission, raw):
    """Reconcile outer identity without treating a terminal failure as success."""
    run, manifest = inputs["run"], inputs["manifest"]
    inner = run / "original_execution/phases" / phase
    outer = run / "phases" / phase
    path = outer / "phase-execution.json"
    if not path.exists():
        return dict(available=False, complete=False, passed=False, integrity_passed=False,
                    binding=None, forecast=None, checks=dict(outer_receipt_available=False))
    value, started = read(path), read(outer / "started.json")
    require(started["phase"] == phase and started["outer_manifest_sha256"] == digest(manifest)
            and started["admission_sha256"] == digest(admission) and started["retry_allowed"] is False
            and type(started["created_unix"]) in (int, float)
            and admission["created"] <= started["created_unix"] < admission["deadline"], "Outer start binding differs")
    require(value["schema_version"] == 1 and value["phase"] == phase
            and value["outer_manifest_sha256"] == value["manifest_sha256"] == digest(manifest)
            and value["admission_sha256"] == digest(admission)
            and value["original_manifest_sha256"] == ORIGINAL_MANIFEST_SHA
            and value["original_source_sha256"] == SOURCE_SHA and value["original_execution_root"] == "original_execution"
            and value["worker_byte_sha256"] == manifest["diagnostic_files"]["pain_deterministic_control_worker.py"]
            and value["reference_worker_byte_sha256"] == REFERENCE_WORKER_SHA
            and value["execution_deadline_unix"] == admission["deadline"] - 60
            and value["assigned_calls"] == manifest["assigned_calls"][phase]
            and value["assigned_rows"] == manifest["assigned_rows"][phase]
            and value["original_campaign_remains_failed"] is value["original_scientific_rules_unchanged"] is True
            and value["independent_audit_required"] is True
            and value["behavior_gates_used"] is value["retry_allowed"] is False
            and value["geometry_policy"] == "Exact BF16 elements; CPU float64 norms descriptive only in independent audit"
            and type(value["finished_unix"]) in (int, float) and math.isfinite(value["finished_unix"])
            and started["created_unix"] <= value["finished_unix"], "Outer phase receipt identity differs")
    for key, filename in (("original_summary", "summary.json"), ("original_started", "started.json"),
                          ("original_identity", "identity.json"), ("original_outcomes", "outcomes.json")):
        target = inner / filename
        expected = artifact_binding(run, target) if target.exists() else None
        require(value[key] == expected, "Inner artifact byte/content binding differs")
    summary = read(inner / "summary.json") if (inner / "summary.json").exists() else None
    for field, fallback in (("observed_calls", 0), ("completed_rows", 0), ("expected_fault_rows", 0),
                            ("missing_rows", value["assigned_rows"]), ("execution_error_rows", 0)):
        require(value[field] == (summary[field] if summary is not None else fallback), "Outer reported coverage differs")
    error = value["error"]
    require(error is None or (isinstance(error, dict) and set(error) == {"type", "message"}
            and all(isinstance(error[k], str) for k in error)), "Malformed outer terminal error")
    expected_preset = effective_preset(manifest["preset"])
    require(value["effective_preset"] in (None, expected_preset), "Initial deterministic preset differs")
    final = value["final_effective_preset"]
    require(final is None or (isinstance(final, dict) and set(final) == set(expected_preset)),
            "Malformed final deterministic preset")
    require(final in (None, expected_preset) or error is not None,
            "Changed final runtime policy omitted its terminal failure")
    if raw["identity_sha256"] is not None:
        require(value["effective_preset"] == expected_preset, "Inference lacks actual initial deterministic preset")
    prediction = value["forecast"]
    expected_prediction = None
    if phase == "setup" and error is None and summary is not None:
        prior = raw["forecast"]
        expected_prediction = (dict(benchmark_seconds=None, limit_seconds=1440, passed=False)
            if prior["benchmark_seconds"] is None else dict(prior, limit_seconds=1440,
                passed=prior["benchmark_seconds"] <= 1440))
    require(prediction == expected_prediction, "New measured forecast differs from original formula/calls")
    complete = summary is not None and summary["complete"] is True
    integrity = (complete and error is None and summary["integrity_passed"] is True
                 and value["final_effective_preset"] == value["effective_preset"])
    passed = (integrity and summary["passed"] is True
              and (phase == "benchmark" or bool(prediction and prediction["passed"])))
    require(value["complete"] is complete and value["integrity_passed"] is integrity and value["passed"] is passed,
            "Outer completion/pass flags differ")
    checks = dict(outer_receipt_available=True, outer_receipt_complete=complete,
                  outer_no_terminal_error=error is None,
                  deterministic_preset_before_and_after=(value["effective_preset"] == value["final_effective_preset"] == expected_preset),
                  new_setup_forecast_passed=phase != "setup" or bool(prediction and prediction["passed"]))
    return dict(available=True, complete=complete, passed=passed, integrity_passed=integrity,
                binding=artifact_binding(run, path), forecast=prediction, checks=checks)


def audit(run, phase):
    inputs = frozen_inputs(run)
    manifest, run = inputs["manifest"], inputs["run"]
    admission = admission_binding(inputs, phase, reproduce_setup=audit)
    inner = run / "original_execution"
    historical_present = (inner / "manifest.json").exists() or (inner / "freeze.json").exists()
    if historical_present:
        require(read(inner / "manifest.json") == manifest["original_manifest"]
                and read(inner / "freeze.json") == manifest["original_freeze"], "Historical inner manifest/freeze changed")
    else:
        require(not inner.exists() or not any(inner.rglob("*")), "Inner measurements lack historical input bindings")
    require(not any(p.is_symlink() for p in (run / "phases").rglob("*")), "Linked outer phase evidence forbidden")
    with original_runtime(inputs) as (original, _, provenance, samples):
        if historical_present:
            raw = original.audit_phase(inner, phase, manifest["original_manifest"], provenance, admission)
        else:
            # Before-worker failure: ask the unchanged auditor to enumerate the
            # frozen assignments against an EMPTY temporary evidence directory.
            # Only already-bound historical inputs are supplied; no execution
            # artifacts are fabricated or written to the actual campaign.
            with tempfile.TemporaryDirectory(prefix="pain-unstarted-audit-") as name:
                empty = Path(name)
                write_once(empty / "freeze.json", manifest["original_freeze"])
                raw = original.audit_phase(empty, phase, manifest["original_manifest"], provenance, admission)
    outer = execution_receipt(inputs, phase, admission, raw)
    checks = dict(raw["checks"], **outer["checks"])
    complete = raw["complete"] and outer["complete"]
    passed = complete and raw["passed"] and outer["passed"] and all(checks.values())
    result = dict(raw, assay="pain_deterministic_control", complete=complete, passed=passed,
        integrity_passed=raw["integrity_passed"] and outer["integrity_passed"], checks=checks,
        manifest_sha256=digest(manifest), original_manifest_sha256=ORIGINAL_MANIFEST_SHA,
        original_audit_sha256=digest(raw), original_auditor_byte_sha256=AUDITOR_SHA,
        original_analyzer_byte_sha256=ANALYZER_SHA, auditor_byte_sha256=sha(__file__),
        freeze_byte_sha256=sha(run / "freeze.json"), original_freeze_byte_sha256=sha(inputs["original_run"] / "freeze.json"),
        inner_historical_metadata_present=historical_present,
        outer_execution_receipt=outer["binding"], diagnostic_files_sha256=digest(manifest["diagnostic_files"]),
        method=METHOD, effective_preset=effective_preset(manifest["preset"]),
        deterministic_reference=manifest["deterministic_reference"],
        benchmark_forecast_seconds=None if outer["forecast"] is None else outer["forecast"]["benchmark_seconds"],
        forecast=outer["forecast"], original_forecast=raw["forecast"],
        sampled_geometry=samples, sampled_row_events=len(samples),
        sampled_token_positions=sum(len(row["sampled_positions"]) for row in samples),
        sampled_scalar_components=sum(row["scalar_components"] for row in samples),
        cpu_norms_used_as_gate=False, original_campaign_remains_failed=True,
        original_benchmark_admitted=False, scientific_rules_changed=False)
    result["limitations"] = list(raw["limitations"]) + [
        "This separately declared deterministic backend adapts numerical settings; it does not retroactively pass the original campaign or identify which setting caused its failure.",
        "Saved BF16 elements are reconstructed exactly; CPU float64 and GPU float32 norm summaries are descriptive, with no CPU/GPU containment gate.",
        "Outer admissions and runtime identity are separately bound; the nested original manifest identifies inherited prompts, parser and analysis rules, not the new paid campaign."]
    result["runtime_comparison_fields"] = list(raw["runtime_comparison_fields"]) + ["deterministic_preset", "outer_manifest_sha256"]
    return result


def verify_existing(run, phase):
    path = Path(run) / "phases" / phase / "audit.json"
    result = audit(run, phase)
    require(read(path) == result, "Existing audit does not reproduce exactly")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--phase", choices=("setup", "benchmark"), required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    require(args.verify_existing != bool(args.out), "Choose either --out or --verify-existing")
    result = verify_existing(args.run, args.phase) if args.verify_existing else audit(args.run, args.phase)
    if args.out:
        write_once(args.out, result)
    print(json.dumps({key: result[key] for key in ("phase", "complete", "passed", "assigned_rows", "missing_rows")}))


if __name__ == "__main__":
    main()
