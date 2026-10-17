"""One immutable scientific worker; independent numerical/scoring audit follows.

The reviewed engineering helpers are reused only for identical source-token and
runtime checks. All scientific calls come from the held coverage_bridge module.
No additional synthetic model forwards, retries, or outcome-based admission.
"""

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

NAME = "qwen32-curiosity-coverage-bridge-v1-admission2"
ENGINEERING_NAME = "qwen32-curiosity-coverage-engineering-v1"
ENGINEERING_SHA = "6de4589af3e007ddba55440988f8f07ebabcf80f1e16f1d01584d017f8d9adf5"
SOURCE_SHA = "1fd4079ed60fbd0169f702ab9c8f99a75d170450dfbd364bc4cb877e27c18f9a"
ARCHIVE_SHA = "4b82bdc563a3b257fd1b5e0302f5889cb6e14e30da9fcb950cc7f3039eb62ca4"
GROUPS = tuple(f"worker_{i}" for i in range(3))
APPS = tuple(f"curiosity-study-coverage-bridge-worker-{i}-h200" for i in range(3))
RATE, WALL, TEARDOWN, CEILING = 0.00139376, 28000, 120, 120.0
OVERLAYS = ("state_backend.py", "coverage_engineering.py", "coverage_bridge.py")
HELPER = "curiosity_coverage_engineering_worker.py"
MATERIALS = "prepare_curiosity_coverage_bridge.py"
AMENDMENT = "reports/curiosity-coverage-bridge-amended-resource-receipt-v1.json"
AMENDMENT_SHA = "e0014a09803e9b44874aa01ecd8c7158220a5a0fcb66fb94b7a02e3c3e2dce5c"
FORECAST = "reports/curiosity-coverage-bridge-v1-feasibility.json"
NOTE = "reports/curiosity-coverage-bridge-resource-amendment-v1.md"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write((canonical(value) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


def resources():
    return dict(
        app_names=list(APPS),
        workers=3,
        gpu="H200",
        cpu=2,
        memory_gib=48,
        wall_seconds=WALL,
        teardown_seconds=TEARDOWN,
        rate_usd_per_second=RATE,
        ceiling_usd=CEILING,
        maximum_liability_usd=3 * (WALL + TEARDOWN) * RATE,
        max_containers_per_app=1,
        retries=0,
        resume_allowed=False,
        local_export_only_seconds=3600,
    )


def check_files(root, inventory):
    root = Path(root)
    require(
        root.is_dir()
        and not root.is_symlink()
        and inventory
        and not any(p.is_symlink() for p in root.rglob("*")),
        "Invalid frozen tree",
    )
    actual = {
        str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts
    }
    require(actual == set(inventory), "Frozen inventory differs")
    for name, checksum in inventory.items():
        relative = Path(name)
        require(
            not relative.is_absolute()
            and ".." not in relative.parts
            and str(relative) == name
            and sha(root / name) == checksum,
            "Frozen bytes differ: " + name,
        )


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def engineering_helper(wrapper, manifest=None):
    if manifest is not None:
        check_files(wrapper, manifest["wrapper_files"])
        require(
            sha(Path(wrapper) / HELPER) == manifest["engineering_manifest"]["wrapper_files"][HELPER],
            "Reviewed engineering helper differs",
        )
    return load_module(Path(wrapper) / HELPER, "_bridge_engineering_helpers")


def worker_descriptors(plan):
    result = []
    for index, group in enumerate(GROUPS):
        rows = [a for a in plan["assignments"] if a["execution_group"] == group]
        cohorts = [i for i, c in enumerate(plan["execution_cohorts"]) if c["execution_group"] == group]
        choices = sum(a["kind"] == "choice" for a in rows)
        result.append(
            dict(
                execution_group=group,
                app_name=APPS[index],
                cohort_indices=cohorts,
                assignment_sha256=digest(rows),
                assigned_generations=len(rows),
                assigned_choices=choices,
                receipt_groups=(choices + 31) // 32,
            )
        )
    return result


def validate_manifest(manifest):
    eng, plan = manifest["engineering_manifest"], manifest["plan"]
    require(
        manifest["schema_version"] == 1
        and manifest["name"] == NAME
        and manifest["assay"] == "curiosity_coverage_bridge"
        and digest(eng) == manifest["engineering_manifest_sha256"] == ENGINEERING_SHA,
        "Unexpected scientific/engineering parent",
    )
    require(
        manifest["resources"] == resources()
        and manifest["scientific_admission"] is False
        and manifest["causal_admission"] is False,
        "Resource or admission policy changed",
    )
    require(
        plan == eng["plan"]
        and digest(plan) == manifest["plan_sha256"] == eng["plan_sha256"]
        and manifest["preflight"] == eng["preflight"]
        and manifest["preflight_sha256"] == eng["preflight_sha256"] == digest(manifest["preflight"]),
        "Scientific materials/tokenization changed",
    )
    for key in ("original_manifest", "original_freeze", "original_execution_files"):
        require(manifest[key] == eng[key], "Original source binding changed: " + key)
    require(
        manifest["source_sha256"] == eng["source_sha256"] == SOURCE_SHA
        and manifest["original_source_archive_sha256"]
        == eng["original_source_archive_sha256"]
        == ARCHIVE_SHA,
        "Original archive/source changed",
    )
    require(
        set(manifest["runtime_delta"]) == set(OVERLAYS)
        and all(manifest["runtime_delta"][name] == eng["runtime_delta"][name] for name in OVERLAYS[:2]),
        "The two admitted numerical overlays must remain unchanged",
    )
    require(
        manifest["runtime_delta"]["coverage_bridge.py"]["replacement_byte_sha256"]
        == manifest["documents"]["src/curiosity_study/coverage_bridge.py"]
        and manifest["wrapper_files"]["analyze_curiosity_coverage_bridge.py"]
        == manifest["documents"]["scripts/analyze_curiosity_coverage_bridge.py"],
        "Scientific collector/estimator differ from the resource amendment",
    )
    require(
        len(plan["assignments"]) == 8352
        and len(plan["execution_cohorts"]) == 261
        and manifest["workers"] == worker_descriptors(plan)
        and [w["assigned_generations"] for w in manifest["workers"]] == [2784] * 3
        and [w["assigned_choices"] for w in manifest["workers"]] == [1537, 1363, 1276],
        "Complete fixed three-worker decomposition changed",
    )
    return manifest


def validate_evidence(manifest, wrapper):
    """Bound prior receipts are prerequisites, not new behavioral evidence."""
    wrapper = Path(wrapper)
    for relative, checksum in manifest["documents"].items():
        require(sha(wrapper / "evidence" / relative) == checksum, "Prior evidence changed: " + relative)
    base = wrapper / "evidence" / "runs" / ENGINEERING_NAME
    context, lifecycle = read(base / "audit.json"), read(base / "lifecycle-audit.json")
    forecast, amended = (read(wrapper / "evidence" / p) for p in (FORECAST, AMENDMENT))
    require(
        context["passed"] is True
        and context["complete"] is True
        and lifecycle["passed"] is True
        and context["manifest_sha256"] == lifecycle["manifest_sha256"] == ENGINEERING_SHA
        and lifecycle["context_audit_byte_sha256"]
        == manifest["documents"][f"runs/{ENGINEERING_NAME}/audit.json"],
        "Complete passing engineering/context/lifecycle required",
    )
    require(
        forecast["available"] is True
        and forecast["passed"] is False
        and len(forecast["workers"]) == 3
        and all(
            math.isfinite(w["forecast_seconds"]) and 0 < w["forecast_seconds"] <= WALL
            for w in forecast["workers"]
        ),
        "Original forecast must remain failed and fit amended wall",
    )
    require(amended == manifest["resource_amendment"], "Amendment receipt changed")
    require(
        sha(wrapper / "evidence" / AMENDMENT) == AMENDMENT_SHA
        and amended["amended_resource_check_passed"] is True
        and amended["scientific_admission"] is False
        and amended["protocol_deviation"] is True
        and amended["original_forecast_passed"] is False
        and amended["assignment_sha256"] == manifest["plan"]["assignment_sha256"]
        and amended["engineering_manifest_sha256"] == ENGINEERING_SHA
        and amended["engineering_raw_input_files_sha256"] == lifecycle["raw_input_files_sha256"]
        and amended["resources"]["common_startup_inclusive_wall_seconds"] == WALL
        and float(amended["resources"]["maximum_compute_liability_usd"])
        == resources()["maximum_liability_usd"]
        and float(amended["resources"]["stage_ceiling_usd"]) == CEILING,
        "Passing exact explicit resource amendment required",
    )
    for item in amended["inputs"].values():
        require(manifest["documents"].get(item["path"]) == item["byte_sha256"], "Amendment input differs")
    require(
        [w["unchanged_forecast_seconds"] for w in amended["workers"]]
        == [w["forecast_seconds"] for w in forecast["workers"]],
        "Forecast values were changed",
    )
    return dict(context=context, lifecycle=lifecycle, forecast=forecast, amendment=amended)


def load_runtime(runtime):
    runtime = Path(runtime).resolve()
    sys.path.insert(0, str(runtime))
    names = (
        "artifacts",
        "batched",
        "model",
        "state_backend",
        "coverage_engineering",
        "coverage_bridge",
        "voluntary_resolution",
        "voluntary_capture",
    )
    modules = {name: importlib.import_module("curiosity_study." + name) for name in names}
    for name, module in modules.items():
        require(
            Path(module.__file__).resolve() == runtime / "curiosity_study" / (name + ".py"),
            "Unbound runtime import; use a fresh process",
        )
    return modules


def verify_sources(manifest, original_root, runtime_root, bridge_root):
    global _HELPER
    validate_manifest(manifest)
    original_root, runtime_root, bridge_root = map(Path, (original_root, runtime_root, bridge_root))
    for root, key in (
        (original_root, "original_execution_files"),
        (runtime_root, "runtime_files"),
        (bridge_root, "wrapper_files"),
    ):
        check_files(root, manifest[key])
    helper = engineering_helper(bridge_root, manifest)
    _HELPER = helper
    for name in (HELPER, MATERIALS, "plan.md", "audit_coverage_engineering.py"):
        require(
            manifest["wrapper_files"][name] == manifest["engineering_manifest"]["wrapper_files"][name],
            "Held engineering wrapper differs: " + name,
        )
    helper.validate_manifest(manifest["engineering_manifest"])
    validate_evidence(manifest, bridge_root)
    expected = {
        name.removeprefix("src/"): checksum
        for name, checksum in manifest["original_execution_files"].items()
        if name.startswith("src/curiosity_study/")
    }
    for name in OVERLAYS:
        key, delta = "curiosity_study/" + name, manifest["runtime_delta"][name]
        require(delta["original_byte_sha256"] == expected.get(key), "Overlay original binding differs")
        expected[key] = delta["replacement_byte_sha256"]
    require(expected == manifest["runtime_files"], "Undeclared runtime modification")
    for path, checksum in (
        (original_root / "src/curiosity_study", SOURCE_SHA),
        (runtime_root / "curiosity_study", manifest["runtime_source_sha256"]),
    ):
        require(
            digest({str(p.relative_to(path)): p.read_text() for p in sorted(path.rglob("*.py"))}) == checksum,
            "Source package digest differs",
        )
    modules = load_runtime(runtime_root)
    builder = load_module(bridge_root / MATERIALS, "_bridge_materials")
    original, plan = manifest["original_manifest"], manifest["plan"]
    directions, inventory = builder.validate_directions(
        original,
        {
            p: (original_root / p).read_bytes()
            for p in (builder.DIRECTIONS_PATH, builder.DIRECTION_INVENTORY_PATH)
        },
    )
    require(
        builder.build_materials(
            original["source_screen_manifest"],
            original,
            directions,
            inventory,
            plan["source_provenance"],
            modules["coverage_engineering"].call_plan,
        )
        == plan,
        "Complete original scientific inputs differ",
    )
    modules["coverage_bridge"].validate_materials(plan)
    analyzer = load_module(bridge_root / "analyze_curiosity_coverage_bridge.py", "_bridge_analyzer_metadata")
    require(analyzer.ANALYSIS_METADATA == manifest["analysis_metadata"], "Frozen analysis differs")
    return modules


def validate_admission(manifest, admission, now, historical=False):
    require(
        all(
            type(v) in (int, float) and math.isfinite(v)
            for v in (admission["created"], admission["deadline"], now)
        )
        and admission["created"] >= 0
        and admission["manifest_sha256"] == digest(manifest)
        and admission["workers_sha256"] == digest(manifest["workers"])
        and admission["phase_consumed"] is True
        and admission["retry_allowed"] is False
        and admission["deadline"] == admission["created"] + WALL
        and admission["maximum_liability_usd"] == resources()["maximum_liability_usd"]
        and (historical or admission["created"] <= now < admission["deadline"]),
        "Invalid one-time common parent admission",
    )


def validate_worker_claim(manifest, admission, claim, group, now, historical=False):
    validate_admission(manifest, admission, now, historical)
    require(
        group in GROUPS
        and claim["execution_group"] == group
        and claim["worker"] == manifest["workers"][GROUPS.index(group)]
        and claim["parent_manifest_sha256"] == claim["manifest_sha256"] == digest(manifest)
        and claim["admission_sha256"] == digest(admission)
        and claim["parent_created"] == admission["created"]
        and claim["app_name"] == APPS[GROUPS.index(group)]
        and claim["deadline"] == admission["deadline"]
        and claim["consumed"] is True
        and admission["created"] <= claim["created"] < admission["deadline"],
        "Invalid fixed worker claim",
    )


def tokenizer_preflight(plan, tokenizer, task):
    # Function is imported only after its exact helper bytes were verified.
    return _HELPER.tokenizer_preflight(plan, tokenizer, task)


def validate_runtime_metadata(metadata, original_root):
    return _HELPER.validate_runtime_metadata(metadata, original_root)


_HELPER = None


def operational_coverage(root, manifest, group):
    """Exact terminal inventory; not a numerical, scoring, or receipt audit."""
    require(group in GROUPS, "Unknown coverage worker")
    root, plan = Path(root), manifest["plan"]
    assigned = [a for a in plan["assignments"] if a["execution_group"] == group]
    cohorts = [c for c in plan["execution_cohorts"] if c["execution_group"] == group]
    choices = [a for a in assigned if a["kind"] == "choice"]
    issues, generation, receipt, artifacts = [], [], [], []
    expected = {f"{c['offset'] // 32:03d}" for c in cohorts}
    for folder in ("generation", "generation_cohorts", "generation_started", "generation_failures"):
        if (root / folder).exists():
            require(not any(p.is_symlink() for p in (root / folder).rglob("*")), "Symlink in worker evidence")
            if any(p.stem not in expected for p in (root / folder).iterdir() if p.is_file()):
                issues.append("Unexpected generation artifact in " + folder)
    for cohort in cohorts:
        index, rows = cohort["offset"] // 32, plan["assignments"][cohort["offset"] : cohort["offset"] + 32]
        marker, path = (
            root / f"generation_started/{index:03d}.json",
            root / f"generation_cohorts/{index:03d}.json",
        )
        records, error = None, None
        try:
            if marker.exists():
                attempt = read(marker)
                require(
                    attempt["materials_sha256"] == manifest["plan_sha256"]
                    and attempt["assignments_sha256"] == digest(rows),
                    "Generation attempt differs",
                )
            if path.exists():
                env = read(path)
                require(
                    marker.exists()
                    and env["materials_sha256"] == manifest["plan_sha256"]
                    and env["execution_group"] == group
                    and env["cohort"] == cohort
                    and env["records_sha256"] == digest(env["records"])
                    and len(env["records"]) == 32,
                    "Generation envelope differs",
                )
                report_path = root / f"generation/{index:03d}.json"
                require(
                    env["artifact"]["path"] == f"generation/{index:03d}.json"
                    and sha(report_path) == env["artifact"]["byte_sha256"],
                    "Generation evidence differs",
                )
                report = read(report_path)
                require(report["call"] == env["call"] == attempt["call"], "Call binding differs")
                array = report["array_artifact"]
                require(
                    array["path"] == f"generation/{index:03d}.npz"
                    and sha(root / array["path"]) == array["byte_sha256"],
                    "Generation array differs",
                )
                for a, r in zip(rows, env["records"], strict=True):
                    require(
                        r["trial_id"] == a["trial_id"]
                        and r["assignment"] == a
                        and r["assignment_sha256"] == digest(a)
                        and r["terminal"] is True
                        and r["status"] in ("completed", "execution_error"),
                        "Terminal assignment differs",
                    )
                records = env["records"]
                artifacts.append(
                    dict(
                        path=str(path.relative_to(root)),
                        byte_sha256=sha(path),
                        report=env["artifact"],
                        array=array,
                    )
                )
        except (KeyError, TypeError, ValueError, OSError) as caught:
            error = str(caught)
            issues.append(dict(call_index=index, error=error))
        for row_index, a in enumerate(rows):
            record = None if records is None else records[row_index]
            generation.append(
                dict(
                    trial_id=a["trial_id"],
                    status=record["status"]
                    if record
                    else "execution_error"
                    if marker.exists() or path.exists()
                    else "missing",
                    record_sha256=digest(record) if record else None,
                )
            )
    groups = [choices[i : i + 32] for i in range(0, len(choices), 32)]
    for folder in ("receipts", "receipt_started", "receipt_failures"):
        if (root / folder).exists() and any(
            p.stem not in {f"{i:03d}" for i in range(len(groups))}
            for p in (root / folder).iterdir()
            if p.is_file()
        ):
            issues.append("Unexpected receipt artifact in " + folder)
    for index, rows in enumerate(groups):
        marker, path = root / f"receipt_started/{index:03d}.json", root / f"receipts/{index:03d}.json"
        records = None
        try:
            ids = [a["trial_id"] for a in rows]
            if marker.exists():
                attempt = read(marker)
                require(
                    attempt["materials_sha256"] == manifest["plan_sha256"]
                    and attempt["logical_choice_offset"] == index * 32
                    and attempt["source_trial_ids"] == ids,
                    "Receipt attempt differs",
                )
            if path.exists():
                report = read(path)
                require(
                    marker.exists()
                    and report["materials_sha256"] == manifest["plan_sha256"]
                    and report["execution_group"] == group
                    and report["logical_choice_offset"] == index * 32
                    and report["source_trial_ids"] == ids
                    and [r["source_trial_id"] for r in report["records"]] == ids
                    and report["records_sha256"] == digest(report["records"]),
                    "Receipt group differs",
                )
                array = report["array_artifact"]
                require(
                    array["path"] == f"receipts/{index:03d}.npz"
                    and sha(root / array["path"]) == array["byte_sha256"],
                    "Receipt array differs",
                )
                records = report["records"]
                artifacts.append(dict(path=str(path.relative_to(root)), byte_sha256=sha(path), array=array))
        except (KeyError, TypeError, ValueError, OSError) as caught:
            issues.append(dict(receipt_group=index, error=str(caught)))
        for row_index, a in enumerate(rows):
            record = None if records is None else records[row_index]
            status = (
                record["status"]
                if record
                else "execution_error"
                if marker.exists() or path.exists()
                else "missing"
            )
            receipt.append(
                dict(trial_id=a["trial_id"], status=status, sidecar_sha256=digest(record) if record else None)
            )
    complete = (
        not issues
        and all(r["status"] == "completed" for r in generation)
        and all(r["status"] in ("completed", "stop", "invalid_choice", "capped_choice") for r in receipt)
    )
    return dict(
        execution_group=group,
        assigned_generations=len(assigned),
        assigned_choices=len(choices),
        generation_slots=generation,
        choice_slots=receipt,
        complete=complete,
        issues=issues,
        artifacts=artifacts,
        numeric_engineering_verified=False,
        selected_receipts_verified=False,
        task_readiness_verified=False,
    )


def run(original_root, runtime, wrapper, root, group):
    global _HELPER
    root = Path(root)
    manifest, admission, claim = (
        read(root / name) for name in ("manifest.json", "admission.json", "worker-claim.json")
    )
    validate_worker_claim(manifest, admission, claim, group, time.time())
    modules = verify_sources(manifest, original_root, runtime, wrapper)
    _HELPER = engineering_helper(wrapper, manifest)
    require(sha(__file__) == manifest["wrapper_files"][Path(__file__).name], "Unbound worker")
    require(
        all(os.environ.get(k) == "1" for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")),
        "Cache-only model loading required",
    )
    write_once(
        root / "worker-started.json",
        dict(
            manifest_sha256=digest(manifest),
            execution_group=group,
            admission_sha256=digest(admission),
            started=time.time(),
            pid=os.getpid(),
        ),
    )
    error, result, count = None, None, 0

    def checkpoint():
        nonlocal count
        count += 1
        (root / "checkpoint.request").write_text(str(count))

    try:
        model = modules["batched"].BatchedHFModel(**manifest["plan"]["model"])
        metadata = _HELPER.runtime_metadata(model, original_root)
        metadata["startup_certificate"] = (
            "bound_completed_coverage_engineering; no additional startup forwards"
        )
        write_once(root / "environment.json", metadata)
        require(
            tokenizer_preflight(manifest["plan"], model.tokenizer, modules["voluntary_resolution"])
            == manifest["preflight"],
            "Actual original tokenization differs",
        )
        checkpoint()
        result = modules["coverage_bridge"].execute(
            root,
            model,
            manifest["plan"],
            execution_group=group,
            checkpoint=checkpoint,
            deadline=admission["deadline"] - 60,
        )
    except Exception as caught:
        error = dict(type=type(caught).__name__, message=str(caught))
    coverage = operational_coverage(root, manifest, group)
    summary = dict(
        manifest_sha256=digest(manifest),
        admission_sha256=digest(admission),
        worker_claim_sha256=digest(claim),
        execution_group=group,
        complete=coverage["complete"] and bool(result and result["complete"]),
        engineering_passed=bool(result and result["engineering_passed"] and error is None),
        operational_coverage=coverage,
        collector_summary_sha256=digest(result) if result else None,
        error=error,
        finished=time.time(),
        independent_audit_required=True,
        checkpoint_count=count,
        scientific_success=False,
        causal_claim=False,
    )
    write_once(root / "summary.json", summary)
    checkpoint()
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("original-root", "runtime", "wrapper", "run"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--group", choices=GROUPS, required=True)
    args = parser.parse_args()
    result = run(args.original_root, args.runtime, args.wrapper, args.run, args.group)
    print(json.dumps({k: result[k] for k in ("complete", "engineering_passed", "execution_group", "error")}))
    raise SystemExit(0 if result["complete"] and result["engineering_passed"] else 1)
