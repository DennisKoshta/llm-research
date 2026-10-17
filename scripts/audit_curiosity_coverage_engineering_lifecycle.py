"""Compose the frozen numerical/context audit with local lifecycle evidence.

This is a read-only postrun check with write-once audit outputs. It schedules no
model work and grants no scientific admission. Raw input bytes are bound before
and after the frozen context audit, including receipt JSON and NPZ artifacts.
"""

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path


NAME = "qwen32-curiosity-coverage-engineering-v1"
MANIFEST_SHA = "6de4589af3e007ddba55440988f8f07ebabcf80f1e16f1d01584d017f8d9adf5"
OUTPUTS = {"audit.json", "lifecycle-audit.json"}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inventory(root):
    require(not any(p.is_symlink() for p in root.rglob("*")), "Symlink in observed artifacts")
    return {str(p.relative_to(root)): sha(p) for p in root.rglob("*")
            if p.is_file() and str(p.relative_to(root)) not in OUTPUTS}


def write_once(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def audit(project):
    project = Path(project).resolve()
    run, execution = project / "runs" / NAME, project / "executions" / NAME
    wrapper = execution / "engineering"
    require(not any((run / name).exists() for name in OUTPUTS), "Audit outputs already exist")
    manifest, freeze = read(run / "manifest.json"), read(run / "freeze.json")
    require(digest(manifest) == freeze["manifest_sha256"] == MANIFEST_SHA,
            "Unexpected engineering freeze")
    require(Path(freeze["execution_root"]) == execution, "Unexpected frozen execution path")
    for filename in ("curiosity_coverage_engineering_worker.py",
                     "launch_curiosity_coverage_engineering.py"):
        require(sha(wrapper / filename) == manifest["wrapper_files"][filename],
                "Frozen audit/controller bytes differ")
    controller = load(wrapper / "launch_curiosity_coverage_engineering.py", "_coverage_lifecycle_controller")
    worker = load(wrapper / "curiosity_coverage_engineering_worker.py", "_coverage_lifecycle_worker")
    before = inventory(run)
    checks, errors = {}, []
    context = None
    try:
        context = worker.audit_context(run, execution / "original", execution / "runtime", wrapper)
        checks["frozen_context_audit_passed"] = context["passed"] is True
    except Exception as caught:
        errors.append(dict(stage="context", type=type(caught).__name__, message=str(caught)))
        context = dict(manifest_sha256=MANIFEST_SHA, complete=False, passed=False,
                       error=errors[-1], scientific_admission=False)
        checks["frozen_context_audit_passed"] = False
    checks["raw_inputs_unchanged_during_context_audit"] = before == inventory(run)
    state = None
    try:
        admission = read(run / "admission.json")
        global_admission = project / "campaign_admissions" / NAME / "admission.json"
        cost_path = project / "reports" / (NAME + "-cost-and-shutdown.json")
        external = {str(path.relative_to(project)): sha(path) for path in
                    (global_admission, cost_path, Path(__file__).resolve())}
        checks["global_admission_exact_bytes"] = global_admission.read_bytes() == (run / "admission.json").read_bytes()
        worker.validate_admission(manifest, admission, dt.datetime.now(dt.timezone.utc).timestamp(), historical=True)
        checks["fixed_single_admission"] = True
        launch, outcome = read(run / "launch.json"), read(run / "launch-outcome.json")
        cloud, summary = read(run / "cloud-finished.json"), read(run / "summary.json")
        environment = read(run / "environment.json")
        checks["exact_checkpoint_geometry_and_termination"] = (
            environment["hidden_size"] == 5120 and environment["vocabulary_size"] == 151936
            and environment["pad_token_id"] == 151643
            and environment["generation_eos_token_ids"] == [151643, 151645]
            and environment["model_id"] == manifest["plan"]["model"]["model_id"]
            and environment["revision"] == manifest["plan"]["model"]["revision"])
        checks["launch_and_summary_binding"] = (
            launch["app_id"] == outcome["app_id"]
            and launch["admission_sha256"] == summary["admission_sha256"] == digest(admission)
            and summary["manifest_sha256"] == MANIFEST_SHA
            and outcome["collection_coverage_sha256"] == digest(read(run / "collection-coverage.json")))
        checks["completed_execution_and_export"] = (
            outcome["failure"] is None and outcome["export_error"] is None
            and not outcome["stop_errors"] and outcome["all_stopped_zero_tasks"] is True
            and outcome["result"]["complete"] is True
            and outcome["result"]["engineering_passed"] is True
            and summary["complete"] is True and summary["engineering_passed"] is True
            and summary["error"] is None and cloud["error"] is None)
        checks["execution_finished_within_fixed_deadline"] = (
            admission["created"] <= launch["started"] <= summary["finished"]
            <= cloud["finished"] <= admission["deadline"])
        costs = read(cost_path)
        state = controller.snapshot(project, wrapper)
        matching_apps = [row["App ID"] for row in state["apps"]
                         if row.get("Description") == manifest["resources"]["app_name"]
                         and dt.datetime.fromisoformat(row["Created at"]).timestamp()
                         >= admission["created"] - 2]
        checks["single_engineering_app"] = matching_apps == [launch["app_id"]]
        charges = [row for row in state["app_charges"] if row["app_id"] == launch["app_id"]]
        require(len(charges) == 1, "Expected exactly one current app charge")
        charge = charges[0]
        checks["stopped_and_inside_resource_allowance"] = (
            costs["all_stopped_zero_tasks"] is state["all_stopped_zero_tasks"] is True
            and charge["rate_usd_per_second"] == manifest["resources"]["rate_usd_per_second"]
            and charge["compute_estimate_usd"] <= manifest["resources"]["maximum_liability_usd"]
            and state["phase_compute_estimate_usd"] <= 300
            and state["available_compute_estimate_usd"] >= state["storage_transfer_reserve_usd"])
        checks["external_inputs_unchanged"] = all(sha(project / path) == checksum
                                                 for path, checksum in external.items())
    except Exception as caught:
        errors.append(dict(stage="lifecycle", type=type(caught).__name__, message=str(caught)))
        external, charge = {}, None
    checks["raw_inputs_unchanged_through_lifecycle_check"] = before == inventory(run)
    write_once(run / "audit.json", context)
    result = dict(
        schema_version=1, assay="curiosity_coverage_engineering_lifecycle_audit",
        timestamp_utc=dt.datetime.now(dt.timezone.utc).isoformat(), manifest_sha256=MANIFEST_SHA,
        passed=not errors and all(checks.values()), checks=checks, errors=errors,
        raw_input_files=before, raw_input_files_sha256=digest(before),
        context_audit_byte_sha256=sha(run / "audit.json"), external_files=external,
        current_budget=state, app_charge=charge, scientific_admission=False,
        timing_forecast_verified=False,
        scope="Composes frozen context/numerical/receipt checks with exact raw-byte, global-admission, shutdown and cost bindings. No scientific outcomes or forecast are evaluated.")
    write_once(run / "lifecycle-audit.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    result = audit(parser.parse_args().project_root)
    print(json.dumps({key: result[key] for key in ("passed", "checks", "errors", "app_charge")}, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
