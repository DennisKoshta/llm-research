"""Compose the scientific context audit with the bounded three-app lifecycle.

Read-only execution audit; writes each result once and schedules no inference.
The final audit cannot pass on context checks without source, admission, export,
shutdown and spending checks. All consumed raw bytes are bound before and after.
"""

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
from pathlib import Path


NAME = "qwen32-curiosity-coverage-bridge-v1-admission2"
GROUPS = ("worker_0", "worker_1", "worker_2")
OUTPUTS = {"context-audit.json", "lifecycle-audit.json", "audit.json",
           "scored-input.json", "analysis.json"}
RATE, WALL, TEARDOWN = 0.00139376, 28000, 120


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def raw_inventory(root):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink()
            and not any(p.is_symlink() for p in root.rglob("*")), "Raw tree contains a symlink")
    return {str(p.relative_to(root)): sha(p) for p in root.rglob("*")
            if p.is_file() and str(p.relative_to(root)) not in OUTPUTS}


def frozen_inventory(root, expected):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink()
            and not any(p.is_symlink() for p in root.rglob("*")), "Frozen tree contains a symlink")
    actual = {str(p.relative_to(root)): sha(p) for p in root.rglob("*")
              if p.is_file() and "__pycache__" not in p.parts}
    require(actual == expected, "Frozen wrapper inventory changed")


def write_once(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def app_lifecycle(manifest, admission, launches, outcomes, state):
    """Pure three-app identity, common deadline and charge reconstruction."""
    expected = {w["execution_group"]: w["app_name"] for w in manifest["workers"]}
    require(set(expected) == set(launches) == set(outcomes) == set(GROUPS),
            "All three worker launch/outcome records are required")
    ids = [launches[g]["app_id"] for g in GROUPS]
    require(len(set(ids)) == 3 and all(isinstance(i, str) and i for i in ids),
            "Distinct accounted app IDs are required")
    resources = manifest["resources"]
    require(resources["wall_seconds"] == WALL and resources["teardown_seconds"] == TEARDOWN
            and resources["rate_usd_per_second"] == RATE
            and math.isclose(resources["maximum_liability_usd"], 3 * (WALL + TEARDOWN) * RATE,
                             rel_tol=0, abs_tol=1e-9)
            and admission["deadline"] == admission["created"] + WALL,
            "Amended resource contract differs")
    charges, checks = [], {}
    for group in GROUPS:
        launch, outcome = launches[group], outcomes[group]
        matching = [a for a in state["apps"] if a.get("Description") == expected[group]
                    and dt.datetime.fromisoformat(a["Created at"]).timestamp() >= admission["created"] - 2]
        require(len(matching) == 1 and matching[0]["App ID"] == launch["app_id"],
                "Worker app missing, replaced or duplicated: " + group)
        app = matching[0]
        rows = [c for c in state["app_charges"] if c["app_id"] == launch["app_id"]]
        require(len(rows) == 1, "Expected exactly one charge for " + group)
        charge = rows[0]
        require(app.get("Stopped at"), "Worker has no recorded shutdown: " + group)
        created = dt.datetime.fromisoformat(app["Created at"]).timestamp()
        stopped = dt.datetime.fromisoformat(app["Stopped at"]).timestamp()
        seconds = stopped - created
        checks[group + "_identity_and_export"] = (
            launch["execution_group"] == outcome["execution_group"] == group
            and outcome["app_id"] == launch["app_id"]
            and admission["created"] <= launch["started"] < admission["deadline"]
            and outcome["failure"] is None and outcome["export_error"] is None
            and not outcome["stop_errors"] and outcome["app_stopped_zero_tasks"] is True
            and type(outcome["exported_files"]) is int and outcome["exported_files"] > 0)
        checks[group + "_stopped_within_liability"] = (
            app["State"] == "stopped" and str(app["Tasks"]) == "0"
            and 0 <= seconds <= WALL + TEARDOWN
            and admission["created"] - 2 <= created <= stopped <= admission["deadline"] + TEARDOWN
            and charge["gpu"] == "H200" and charge["rate_usd_per_second"] == RATE
            and abs(charge["app_wall_seconds"] - seconds) < 1e-6
            and abs(charge["compute_estimate_usd"] - seconds * RATE) < 1e-8)
        charges.append(charge)
    total = sum(c["compute_estimate_usd"] for c in charges)
    checks["three_worker_costs_within_reservation"] = (
        total <= resources["maximum_liability_usd"] <= resources["ceiling_usd"] == 120
        and all(c["compute_estimate_usd"] <= (WALL + TEARDOWN) * RATE for c in charges))
    checks["fresh_allowance_and_retained_reserve"] = (
        state["all_stopped_zero_tasks"] is True
        and state["fresh_compute_allowance_usd"] == 300
        and state["phase_compute_estimate_usd"] <= 300
        and state["available_compute_estimate_usd"] >= state["storage_transfer_reserve_usd"] >= 10
        and abs(state["phase_compute_estimate_usd"]
                - admission["prior_phase_compute_estimate_usd"] - total) < 1e-6)
    return dict(passed=all(checks.values()), checks=checks, app_charges=charges,
                total_compute_estimate_usd=total)


def audit(project):
    project = Path(project).resolve()
    run, execution = project / "runs" / NAME, project / "executions" / NAME
    wrapper = execution / "bridge"
    require(not any((run / n).exists() for n in OUTPUTS), "Audit/analysis outputs already exist")
    manifest, freeze = read(run / "manifest.json"), read(run / "freeze.json")
    require(manifest["name"] == NAME and digest(manifest) == freeze["manifest_sha256"]
            and Path(freeze["execution_root"]) == execution, "Frozen parent identity differs")
    frozen_inventory(wrapper, manifest["wrapper_files"])
    require(sha(__file__) == manifest["wrapper_files"][Path(__file__).name], "Use frozen lifecycle auditor")
    controller = load(wrapper / "launch_curiosity_coverage_bridge.py", "_bridge_lifecycle_controller")
    worker = load(wrapper / "curiosity_coverage_bridge_worker.py", "_bridge_lifecycle_worker")
    context_module = load(wrapper / "audit_curiosity_coverage_bridge.py", "_bridge_context_auditor")
    before, checks, errors, external = raw_inventory(run), {}, [], {}
    # Context API reconciles incomplete workers and retains every assigned slot.
    context = context_module.audit(run, context_root=execution)
    checks["context_integrity_passed"] = context["integrity_passed"] is True
    checks["context_complete"] = context["complete"] is True
    checks["raw_unchanged_during_context_audit"] = before == raw_inventory(run)
    state, charged = None, None
    try:
        admission = read(run / "admission.json")
        global_root = project / "campaign_admissions" / NAME
        paths = [global_root / "admission.json",
                 project / "reports" / (NAME + "-cost-and-shutdown.json")]
        paths += [global_root / "workers" / (g + ".json") for g in GROUPS]
        external = {str(p.relative_to(project)): sha(p) for p in paths}
        checks["exclusive_parent_admission_bytes"] = (
            (global_root / "admission.json").read_bytes() == (run / "admission.json").read_bytes())
        worker.validate_manifest(manifest)
        worker.validate_admission(manifest, admission, dt.datetime.now(dt.UTC).timestamp(), historical=True)
        checks["parent_policy_and_admission"] = True
        launches, outcomes = {}, {}
        for group in GROUPS:
            root = run / "workers" / group
            claim = read(root / "worker-claim.json")
            worker.validate_worker_claim(manifest, admission, claim, group,
                                         dt.datetime.now(dt.UTC).timestamp(), historical=True)
            checks[group + "_exclusive_claim_and_parent_bytes"] = (
                (root / "worker-claim.json").read_bytes() == (global_root / "workers" / (group + ".json")).read_bytes()
                and (root / "manifest.json").read_bytes() == (run / "manifest.json").read_bytes()
                and (root / "admission.json").read_bytes() == (run / "admission.json").read_bytes()
                and claim["manifest_sha256"] == claim["parent_manifest_sha256"] == digest(manifest)
                and claim["parent_created"] == admission["created"])
            launch, outcome, summary, cloud = (read(root / name) for name in
                ("launch.json", "launch-outcome.json", "summary.json", "cloud-finished.json"))
            launches[group], outcomes[group] = launch, outcome
            environment = read(root / "environment.json")
            checks[group + "_checkpoint_geometry_and_termination"] = (
                environment["hidden_size"] == 5120 and environment["vocabulary_size"] == 151936
                and environment["pad_token_id"] == 151643
                and environment["generation_eos_token_ids"] == [151643, 151645]
                and environment["model_id"] == manifest["plan"]["model"]["model_id"]
                and environment["revision"] == manifest["plan"]["model"]["revision"])
            checks[group + "_completed_bound_execution"] = (
                launch["admission_sha256"] == summary["admission_sha256"] == digest(admission)
                and launch["worker_claim_sha256"] == summary["worker_claim_sha256"] == digest(claim)
                and summary["manifest_sha256"] == digest(manifest) and summary["execution_group"] == group
                and summary["complete"] is True and summary["engineering_passed"] is True
                and summary["error"] is None and cloud["error"] is None
                and outcome["result"]["complete"] is True and outcome["result"]["engineering_passed"] is True
                and admission["created"] <= launch["started"] <= summary["finished"]
                <= cloud["finished"] <= admission["deadline"])
        parent = read(run / "launch-outcome.json")
        coverage = read(run / "collection-coverage.json")
        checks["parent_export_and_complete_coverage"] = (
            parent["complete"] is True and parent["all_stopped_zero_tasks"] is True
            and parent["export_error"] is None
            and parent["worker_outcomes"] == outcomes
            and parent["worker_exit_codes"] == {g: 0 for g in GROUPS}
            and parent["collection_coverage_sha256"] == digest(coverage))
        costs = read(project / "reports" / (NAME + "-cost-and-shutdown.json"))
        state = controller.snapshot(project, wrapper)
        charged = app_lifecycle(manifest, admission, launches, outcomes, state)
        checks.update(charged["checks"])
        checks["saved_cost_receipt_matches_current_charges"] = (
            costs["all_stopped_zero_tasks"] is True
            and all(c in costs["app_charges"] for c in charged["app_charges"])
            and abs(costs["phase_compute_estimate_usd"] - state["phase_compute_estimate_usd"]) < 1e-6)
        checks["external_inputs_unchanged"] = all(sha(project / name) == value for name, value in external.items())
    except (ValueError, KeyError, TypeError, OSError, IndexError) as error:
        errors.append(dict(type=type(error).__name__, message=str(error)))
    checks["raw_unchanged_through_lifecycle_audit"] = before == raw_inventory(run)
    write_once(run / "context-audit.json", context)
    lifecycle = dict(schema_version=1, assay="curiosity_coverage_bridge_lifecycle_audit",
        timestamp_utc=dt.datetime.now(dt.UTC).isoformat(), manifest_sha256=digest(manifest),
        passed=not errors and all(checks.values()), checks=checks, errors=errors,
        raw_input_files=before, raw_input_files_sha256=digest(before), external_files=external,
        context_audit_byte_sha256=sha(run / "context-audit.json"), current_budget=state,
        charges=charged, scientific_admission=False, scope="Exact three-app admission, raw bytes, export, shutdown and cost composition.")
    write_once(run / "lifecycle-audit.json", lifecycle)
    combined = dict(context)
    combined.update(context_integrity_passed=context["integrity_passed"],
        integrity_passed=context["integrity_passed"] and lifecycle["passed"],
        passed=bool(context.get("passed", context["integrity_passed"]) and lifecycle["passed"]),
        lifecycle_shutdown_verified=lifecycle["passed"],
        context_audit_byte_sha256=sha(run / "context-audit.json"),
        lifecycle_audit_byte_sha256=sha(run / "lifecycle-audit.json"),
        scientific_admission=False)
    write_once(run / "audit.json", combined)
    return combined, lifecycle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    combined, result = audit(parser.parse_args().project_root)
    print(json.dumps({k: result[k] for k in ("passed", "checks", "errors", "charges")}, indent=2))
    raise SystemExit(0 if combined["passed"] else 1)
