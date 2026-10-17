"""Run the frozen local audit and estimator only after collection has closed.

This is an operational wrapper, not a scientific revision. It never imports a
collector or starts inference. Failure classifications still receive the held
analysis; incomplete provenance stops this wrapper without changing raw data.
"""

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


NAME = "qwen32-curiosity-coverage-bridge-v1-admission2"
MANIFEST_SHA256 = "0f093a1557f5ea264bd58f030f3af12462fc7fcb922d873e1d30de4d2989319c"
ANALYSIS_OUTPUTS = ("context-audit.json", "lifecycle-audit.json", "audit.json",
                    "scored-input.json", "analysis.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_once(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def closed(project, run):
    """Read only lifecycle metadata while scientific collection is active."""
    outcome = run / "launch-outcome.json"
    cost = project / "reports" / (NAME + "-cost-and-shutdown.json")
    if not outcome.exists() or not cost.exists():
        return False
    try:
        terminal, spending = read(outcome), read(cost)
    except json.JSONDecodeError:
        # The immutable producer may still be completing its one local write.
        return False
    require(terminal.get("all_stopped_zero_tasks") is True
            and spending.get("all_stopped_zero_tasks") is True,
            "Collection terminated without verified shutdown; manual reconciliation required")
    require((run / "collection-coverage.json").is_file(), "Missing terminal export inventory")
    return True


def context(project):
    run = project / "runs" / NAME
    wrapper = project / "executions" / NAME / "bridge"
    manifest, freeze = read(run / "manifest.json"), read(run / "freeze.json")
    require(digest(manifest) == freeze["manifest_sha256"] == MANIFEST_SHA256,
            "Unexpected scientific freeze")
    expected = manifest["wrapper_files"]
    require(not wrapper.is_symlink() and not any(p.is_symlink() for p in wrapper.rglob("*")),
            "Frozen wrapper contains a symlink")
    actual = {str(p.relative_to(wrapper)): sha(p) for p in wrapper.rglob("*")
              if p.is_file() and "__pycache__" not in p.parts}
    require(actual == expected, "Frozen wrapper inventory differs")
    return run, wrapper, manifest


def close(project, wait=False):
    project = Path(project).resolve()
    run, wrapper, manifest = context(project)
    require(not any((run / n).exists() for n in ANALYSIS_OUTPUTS),
            "Audit or analysis already exists; do not rerun or overwrite")
    reports = project / "reports"
    prefix = reports / (NAME + "-local-closeout")
    lock = prefix.with_suffix(".claim.json")
    write_once(lock, dict(created_utc=dt.datetime.now(dt.UTC).isoformat(), pid=os.getpid(),
                          manifest_sha256=MANIFEST_SHA256, driver_byte_sha256=sha(__file__),
                          scope="Local frozen audit and analysis only; no inference"))
    result = dict(manifest_sha256=MANIFEST_SHA256, driver_byte_sha256=sha(__file__),
                  scientific_revision=False, new_inference=False, error=None)
    try:
        deadline = read(run / "admission.json")["deadline"] + 120 + 3600 + 300
        while not closed(project, run):
            require(wait, "Collection is still active")
            require(time.time() < deadline, "Collection closeout window expired")
            time.sleep(min(30, max(0.1, deadline - time.time())))
        # Recheck the held executable bytes after the wait and before imports.
        run, wrapper, manifest = context(project)
        result["collection_closed_utc"] = dt.datetime.now(dt.UTC).isoformat()
        audit_log = prefix.with_suffix(".audit.log")
        with audit_log.open("xb") as output:
            audit_process = subprocess.run(
                [str(project / ".venv/bin/python"),
                 str(wrapper / "audit_curiosity_coverage_bridge_lifecycle.py"),
                 "--project-root", str(project)], cwd=project, stdout=output,
                stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                check=False)
        result["audit_exit_code"] = audit_process.returncode
        require(audit_process.returncode in (0, 1), "Frozen auditor terminated abnormally")
        require(all((run / name).is_file() for name in ANALYSIS_OUTPUTS[:3]),
                "Frozen auditor did not write all composition artifacts")
        final_audit = read(run / "audit.json")
        require(final_audit["manifest_sha256"] == MANIFEST_SHA256
                and final_audit["context_audit_byte_sha256"] == sha(run / "context-audit.json")
                and final_audit["lifecycle_audit_byte_sha256"] == sha(run / "lifecycle-audit.json"),
                "Composed audit byte bindings differ")
        require((audit_process.returncode == 0) == (final_audit["passed"] is True),
                "Audit exit status differs from saved classification")
        scorer = module(wrapper / "score_curiosity_coverage_bridge.py", "_closed_bridge_scorer")
        estimator = module(wrapper / "analyze_curiosity_coverage_bridge.py", "_closed_bridge_estimator")
        payload = scorer.assemble_analysis_input(manifest["plan"], final_audit,
                                                estimator.ANALYSIS_METADATA)
        write_once(run / "scored-input.json", payload)
        # An audit/readiness failure is retained in the estimator input. Never
        # condition arithmetic on a passing scientific outcome.
        analysis = estimator.analyze_payload(payload)
        analysis.update(input_byte_sha256=sha(run / "scored-input.json"),
                        analyzer_byte_sha256=sha(wrapper / "analyze_curiosity_coverage_bridge.py"))
        write_once(run / "analysis.json", analysis)
        result.update(completed=True, output_files={n: sha(run / n) for n in ANALYSIS_OUTPUTS},
                      audit_passed=final_audit["passed"],
                      lifecycle_shutdown_verified=final_audit["lifecycle_shutdown_verified"])
    except Exception as error:
        result.update(completed=False, error=dict(type=type(error).__name__, message=str(error)))
    result["finished_utc"] = dt.datetime.now(dt.UTC).isoformat()
    write_once(prefix.with_suffix(".json"), result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--wait-for-closure", action="store_true")
    args = parser.parse_args()
    value = close(args.project_root, args.wait_for_closure)
    print(json.dumps(value, indent=2))
    sys.exit(0 if value["completed"] else 1)
