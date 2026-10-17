"""Metadata-only, fail-closed resource forecast. Never grants cloud admission.

Behavioral responses and answer categories are neither extracted nor used. The
pure arithmetic is separate from the byte-bound input reader. See the prospective
method note for deliberately overlapping overhead reserves and extrapolations.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = "qwen32-curiosity-suppression-screen-v1-admission2"
PARENT_SHA = "535f3c1237a8b876fa1059d3f9949f178a55b5627e2be412008ef08cbc90b02f"
ARCHIVE_SHA = "4b82bdc563a3b257fd1b5e0302f5889cb6e14e30da9fcb950cc7f3039eb62ca4"
METHOD = "reports/curiosity-coverage-forecast-method.md"
DIRECTIONS = (
    "zero_sham", "curiosity_negative", "confidence_positive", "completion_positive",
    "valence_positive", "valence_negative", "motivation_negative", "novelty_negative",
    "task_demand_negative", *(f"random_{i:02d}" for i in range(1, 6)),
)
OLD_ARMS = ("baseline", *DIRECTIONS)
ARMS = ("baseline", *(f"{s}__{d}" for d in DIRECTIONS for s in ("record", "full")))
RATE, WALL, TEARDOWN, CEILING, STORAGE, MARGIN = .00139376, 19000, 120, 80., 10., 1.25


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def number(value, name, *, positive=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0), "Unavailable/invalid " + name)
    return float(value)


def width(value):
    require(type(value) is int and value > 0, "Invalid token width")
    return 128 * math.ceil(value / 128)


def receipt_batches(plan):
    """Original cross-cohort logical grouping, including every possible receipt."""
    assignments = plan["assignments"]
    dispositions = plan["receipt_materials"]["assigned_dispositions"]
    require(digest(dispositions) == plan["receipt_materials"]["assigned_dispositions_sha256"],
            "Receipt inventory digest differs")
    by_id = {r["trial_id"]: r for r in dispositions}
    choices = [a for a in assignments if a["kind"] == "choice"]
    require(len(by_id) == len(dispositions) == len(choices) == 4176
            and set(by_id) == {a["trial_id"] for a in choices}, "All4176 receipt slots required")
    batches = []
    for worker in range(3):
        rows = [a for a in choices if a["execution_group"] == f"worker_{worker}"]
        for offset in range(0, len(rows), 32):
            block, lengths = rows[offset:offset + 32], []
            for a in block:
                item = by_id[a["trial_id"]]
                require(item["base_trial_id"] == a["base_trial_id"], "Receipt base row differs")
                possible = item["non_stop_bindings"]
                require(set(possible) == {"report", "repeat", "pattern"}, "Missing possible payload")
                lengths.append(max(p["prefix_positions"]["final"] + 1 for p in possible.values()))
            batches.append(dict(worker=worker, choice_offset=offset, rows=len(block),
                trial_ids=[a["trial_id"] for a in block], padded_width=width(max(lengths))))
    require(Counter(b["worker"] for b in batches) == {0: 49, 1: 43, 2: 40},
            "Original receipt regrouping/worker denominator differs")
    return batches


def forecast_tables(plan, historical, engineering):
    """All arguments are explicit metadata. Raises on unavailable decomposition."""
    assignments, cohorts = plan["assignments"], plan["execution_cohorts"]
    require(len(assignments) == len({a["trial_id"] for a in assignments}) == 8352
            and digest(assignments) == plan["assignment_sha256"]
            and len(cohorts) == 261 and digest(cohorts) == plan["execution_cohorts_sha256"],
            "All8352 rows/261 original cohorts required")
    expected = {(i, a) for i in range(9) for a in ARMS}
    require({(c["source_cohort"], c["arm"]) for c in cohorts} == expected,
            "The complete29-arm crossing is required")
    for index, c in enumerate(cohorts):
        block = assignments[index * 32:(index + 1) * 32]
        require(c["offset"] == index * 32 and c["trial_ids"] == [a["trial_id"] for a in block]
                and c["execution_group"] == f"worker_{c['source_cohort'] % 3}"
                and all(a["arm"] == c["arm"] and a["base_cohort"] == c["source_cohort"]
                        and a["execution_group"] == c["execution_group"] for a in block),
                "Original complete cohort order/worker differs")
    old_rows = historical["calls"]
    old = {(r["source_cohort"], r["arm"]): number(r["generation_seconds"], "old generation", positive=True)
           for r in old_rows}
    require(len(old_rows) == len(old) == 135
            and set(old) == {(i, a) for i in range(9) for a in OLD_ARMS}, "All135 old timings required")
    profiles = {p["worker"]: p for p in historical["profiles"]}
    require(len(historical["profiles"]) == len(profiles) == 3 and set(profiles) == {0, 1, 2},
            "All three historical execution profiles required")
    old_residual = {}
    for w, p in profiles.items():
        total = sum(t for (c, _), t in old.items() if c % 3 == w)
        execution = number(p["execution_seconds"], "old execution", positive=True)
        require(execution >= total, "Historical execution cannot be less than its generation timers")
        old_residual[w] = execution - total

    require(engineering["passed"] is True, "Complete passing engineering evidence required")
    samples = engineering["timing_calls"]
    new = {(r["source_cohort"], r["condition"]): r for r in samples}
    require(len(samples) == len(new) == 6 and set(new) == {(c, k) for c in (0, 3)
            for k in ("baseline", "record_nonzero", "full_nonzero")}, "All six fixed timing calls required")
    for s in samples:
        for key in ("generation_seconds", "generation_artifact_bytes", "generation_serialization_seconds",
                    "receipt_artifact_bytes", "receipt_serialization_seconds"):
            number(s[key], key, positive=True)
        for key in ("receipt_forward_seconds", "receipt_parsing_seconds"):
            number(s[key], key)
        require(type(s["receipt_rows"]) is int and 0 <= s["receipt_rows"] <= 32,
                "Invalid observed receipt count")
        if s["receipt_rows"]:
            require(s["receipt_forward_calls"] == 2 and s["receipt_padded_width"] == width(s["receipt_padded_width"]),
                    "Complete paired receipt timer/width required")
            number(s["receipt_forward_seconds"], "paired receipt seconds", positive=True)
        else:
            require(s["receipt_forward_calls"] == 0 and s["receipt_padded_width"] is None,
                    "Skipped receipts cannot claim measured costs")
    matches = [dict(source_cohort=c, condition=k, old_arm=a,
        new_seconds=new[c, k]["generation_seconds"], old_seconds=old[c, a],
        ratio=new[c, k]["generation_seconds"] / old[c, a])
        for c in (0, 3) for k, a in (("baseline", "baseline"), ("record_nonzero", "curiosity_negative"))]
    common = max(1., *(r["ratio"] for r in matches))
    full_matches = [dict(source_cohort=c, ratio=new[c, "full_nonzero"]["generation_seconds"]
                        / new[c, "record_nonzero"]["generation_seconds"]) for c in (0, 3)]
    full = max(1., *(r["ratio"] for r in full_matches))
    active = [s for s in samples if s["receipt_rows"]]
    require(active, "No complete actual-selected paired receipt measurement; forecast unavailable")
    observed_bytes = number(engineering["observed_artifact_bytes"], "observed artifact bytes", positive=True)
    commits = engineering["checkpoint_commit_seconds"]
    require(isinstance(commits, list) and commits, "Unavailable checkpoint timing")
    commits = [number(v, "checkpoint seconds") for v in commits]
    export_seconds = number(engineering["export_seconds"], "download plus coverage/hash seconds")
    elapsed = number(engineering["collector_elapsed_seconds"], "collector end-to-end seconds", positive=True)
    all_generation = number(engineering["all16_generation_seconds"], "all16 generation seconds", positive=True)
    require(elapsed >= all_generation, "Collector end-to-end time is below its generation timers")
    mixed = elapsed - all_generation
    loading = max(number(engineering["load_seconds"], "new load", positive=True),
                  *(number(p["load_seconds"], "old load", positive=True) for p in profiles.values()))
    startup = number(engineering["startup_nonloading_seconds"], "startup nonloading seconds")
    lifecycle = number(engineering["lifecycle_overhead_seconds"], "noncollector lifecycle seconds")
    persistence_rate = max(s[k + "_serialization_seconds"] / s[k + "_artifact_bytes"]
                           for s in samples for k in ("generation", "receipt"))
    generation, receipts = [], []
    for c in cohorts:
        schedule, arm = ("baseline", "baseline") if c["arm"] == "baseline" else c["arm"].split("__", 1)
        predicted = old[c["source_cohort"], arm] * common * (full if schedule == "full" else 1)
        condition = {"baseline": "baseline", "record": "record_nonzero", "full": "full_nonzero"}[schedule]
        refs = [s for s in samples if s["condition"] == condition]
        size = max(s["generation_artifact_bytes"] * max(1., predicted / s["generation_seconds"]) for s in refs)
        serialization = max(size * persistence_rate, *(s["generation_serialization_seconds"] for s in refs))
        generation.append(dict(source_cohort=c["source_cohort"], arm=c["arm"], worker=c["source_cohort"] % 3,
            historical_arm=arm, historical_seconds=old[c["source_cohort"], arm],
            generation_seconds=predicted, artifact_bytes=size, serialization_seconds=serialization))
    batches = receipt_batches(plan)
    require(all(s["receipt_padded_width"] <= max(b["padded_width"] for b in batches) for s in active),
            "Observed receipt exceeds the input-only possible-payload width inventory")
    for batch in batches:
        def scale(s, power):
            return batch["rows"] / s["receipt_rows"] * max(1., batch["padded_width"] / s["receipt_padded_width"]) ** power
        seconds = max((s["receipt_forward_seconds"] + s["receipt_parsing_seconds"]) * scale(s, 2) for s in active)
        size = max(s["receipt_artifact_bytes"] * scale(s, 1) for s in active)
        serialization = max(size * persistence_rate,
            *(s["receipt_serialization_seconds"] * scale(s, 1) for s in active))
        receipts.append(dict(**batch, paired_forward_and_parsing_seconds=seconds,
                             artifact_bytes=size, serialization_seconds=serialization))
    workers = []
    for w in range(3):
        gs, rs = [r for r in generation if r["worker"] == w], [r for r in receipts if r["worker"] == w]
        artifact_bytes = sum(r["artifact_bytes"] for r in gs + rs)
        checkpoints = len(gs) + len(rs) + 2  # model-loaded and summary; final commit added below
        checkpoint = max(checkpoints * max(commits), artifact_bytes / observed_bytes * sum(commits)) + max(commits)
        terms = dict(generation_seconds=sum(r["generation_seconds"] for r in gs),
            mixed_overhead_reserve_seconds=max(old_residual[w] * 87 / 45, mixed * 87 / 16),
            paired_receipts_and_parsing_seconds=sum(r["paired_forward_and_parsing_seconds"] for r in rs),
            serialization_seconds=sum(r["serialization_seconds"] for r in gs + rs),
            checkpoint_seconds=checkpoint,
            export_and_coverage_hash_seconds=export_seconds * artifact_bytes / observed_bytes,
            noncollector_lifecycle_seconds=lifecycle)
        forecast = startup + loading + MARGIN * sum(terms.values())
        workers.append(dict(worker=w, generation_calls=len(gs), receipt_batches=len(rs),
            maximum_receipt_rows=sum(r["rows"] for r in rs), predicted_artifact_bytes=artifact_bytes,
            components=terms, startup_nonloading_seconds=startup, loading_seconds=loading,
            execution_margin_multiplier=MARGIN, forecast_seconds=forecast,
            deadline_seconds=WALL, fits_deadline=forecast <= WALL,
            historical_mixed_overhead_seconds=old_residual[w], new_mixed_overhead_seconds=mixed))
    liability = 3 * (WALL + TEARDOWN) * RATE
    return dict(schema_version=1, available=True, passed=all(w["fits_deadline"] for w in workers) and liability <= CEILING,
        method="prospective_matched_ratios_and_explicit_overhead_v1", matches=matches, full_matches=full_matches,
        common_engine_slowdown=common, full_schedule_slowdown=full, generation_calls=generation,
        receipt_batches=receipts, workers=workers, maximum_compute_liability_usd=liability,
        compute_ceiling_usd=CEILING, retained_storage_reserve_usd=STORAGE,
        scientific_admission=False, causal_admission=False, global_budget_admission=False,
        limitations=["Planning extrapolation, not an all-cap or unobserved-control runtime guarantee.",
            "Receipt per-row/squared-width and byte/export scaling are assumptions, not measured upper bounds.",
            "Mixed-overhead reserve deliberately overlaps explicit receipt/persistence allowances.",
            "Persistence serialization timings are producer measurements; export includes local coverage hashing.",
            "The final commit is unmeasured and receives an explicit largest-observed-commit allowance.",
            "A separate global budget/closed-app check and frozen admission remain required."])


class Reader:
    def __init__(self):
        self.bindings = {}

    def bind(self, path):
        path = Path(path)
        require(path.is_file() and not path.is_symlink(), "Missing/unsafe measured file: " + str(path))
        path = path.resolve()
        with path.open("rb") as stream:
            value = dict(path=str(path), byte_sha256=hashlib.file_digest(stream, "sha256").hexdigest(), byte_count=path.stat().st_size)
        require(str(path) not in self.bindings or self.bindings[str(path)] == value,
                "Previously bound input changed: " + str(path))
        self.bindings[str(path)] = value
        return value

    def json(self, path):
        bound = self.bind(path)
        payload = Path(path).read_bytes()
        require(hashlib.sha256(payload).hexdigest() == bound["byte_sha256"]
                and len(payload) == bound["byte_count"], "JSON changed between binding and parsing")
        return json.loads(payload)

    def verify_unchanged(self):
        for path in list(self.bindings):
            self.bind(path)


def historical_inputs(project, reader):
    parent = Path(project) / "runs" / PARENT
    manifest = reader.json(parent / "manifest.json")
    freeze = reader.json(parent / "freeze.json")
    require(digest(manifest) == freeze["manifest_sha256"] == PARENT_SHA
            and freeze["archive_sha256"] == ARCHIVE_SHA
            and reader.bind(parent / "source.tar.gz")["byte_sha256"] == ARCHIVE_SHA,
            "Wrong closed C source package")
    calls, profiles = [], []
    for w in range(3):
        child = Path(str(parent) + f"-worker-{w}")
        m = reader.json(child / "manifest.json")
        rows = [a for a in manifest["diagnostics"] if a["execution_group"] == f"worker_{w}"]
        require(m["parent_manifest_sha256"] == PARENT_SHA and m["diagnostics"] == rows
                and len(rows) == 1440 and m["assignment_sha256"] == digest(rows), "Child original assignments differ")
        paths = sorted((child / "generation_evidence").glob("*/*.json"))
        require(len(paths) == 45 and len({p.parent.name for p in paths}) == 1, "All45 one-attempt old timings required")
        offsets = set()
        for path in paths:
            value = reader.json(path)
            offset = value["generation_offset"]
            require(type(offset) is int and offset in range(0, 1440, 32) and offset not in offsets,
                    "Duplicate/missing historical cohort offset")
            offsets.add(offset)
            block = rows[offset:offset + 32]
            require(value["manifest_sha256"] == digest(m) and value["trial_ids"] == [a["trial_id"] for a in block]
                    and value["engineering_passed"] is True and value["condition"]["error"] is None
                    and len({(a["base_cohort"], a["arm"]) for a in block}) == 1,
                    "Historical timing/source binding differs")
            calls.append(dict(source_cohort=block[0]["base_cohort"], arm=block[0]["arm"],
                generation_seconds=value["condition"]["elapsed_seconds"], evidence_path=str(path)))
        paths = sorted((child / "cloud_profiles").glob("*.json"))
        require(len(paths) == 1, "One complete cloud profile required")
        p = reader.json(paths[0])
        require(p["assigned"] == p["completed"] == 1440, "Incomplete historical profile")
        profiles.append(dict(worker=w, execution_seconds=p["execution_seconds"], load_seconds=p["load_seconds"]))
    return dict(calls=calls, profiles=profiles)


def lifecycle_binding(run, project, audit_path, reader):
    """Bind the exact inputs seen by the frozen context audit, including receipts."""
    run, project = Path(run), Path(project)
    lifecycle = reader.json(run / "lifecycle-audit.json")
    required = {"manifest.json", "freeze.json", "admission.json", "environment.json",
                "engineering-started.json", "worker-started.json", "engineering-summary.json",
                "launch-outcome.json", "cloud-finished.json"}
    required |= {f"calls/{i:02d}.{suffix}" for i in range(16) for suffix in ("json", "npz")}
    required |= {f"receipts/{i:02d}.{suffix}" for i in range(10, 16) for suffix in ("json", "npz")}
    require(lifecycle["passed"] is True and lifecycle["scientific_admission"] is False
            and required <= set(lifecycle["raw_input_files"])
            and digest(lifecycle["raw_input_files"]) == lifecycle["raw_input_files_sha256"]
            and reader.bind(audit_path)["byte_sha256"] == lifecycle["context_audit_byte_sha256"],
            "Passing contemporaneous lifecycle/input binding required")
    for root, inventory in ((run, lifecycle["raw_input_files"]), (project, lifecycle["external_files"])):
        require(isinstance(inventory, dict) and inventory, "Missing lifecycle file inventory")
        for name, checksum in inventory.items():
            relative = Path(name)
            require(not relative.is_absolute() and ".." not in relative.parts and str(relative) == name,
                    "Unsafe lifecycle input path")
            require(reader.bind(root / relative)["byte_sha256"] == checksum, "Audited input bytes changed: " + name)
    return lifecycle


def engineering_inputs(run, audit_path, reader, project=None):
    run = Path(run)
    lifecycle = lifecycle_binding(run, project or run.parent.parent, audit_path, reader)
    m = reader.json(run / "manifest.json")
    freeze = reader.json(run / "freeze.json")
    audit = reader.json(audit_path)
    require(digest(m) == freeze["manifest_sha256"] == audit["manifest_sha256"] == lifecycle["manifest_sha256"]
            and m["original_manifest"] and digest(m["original_manifest"]) == PARENT_SHA
            and audit["complete"] is True and audit["passed"] is True
            and audit["source_provenance_verified"] is True and audit["runtime_verified"] is True
            and audit["selected_receipts_verified"] is True
            and audit["numeric"]["numeric_engineering_passed"] is True,
            "Complete bound independent engineering audit required")
    admission = reader.json(run / "admission.json")
    require(digest(admission) == audit["admission_sha256"] and admission["manifest_sha256"] == digest(m),
            "Engineering admission binding differs")
    environment = reader.json(run / "environment.json")
    require(reader.bind(run / "environment.json")["byte_sha256"] == audit["environment_byte_sha256"],
            "Audited runtime bytes differ")
    started = reader.json(run / "engineering-started.json")
    worker_started = reader.json(run / "worker-started.json")
    summary = reader.json(run / "engineering-summary.json")
    outcome = reader.json(run / "launch-outcome.json")
    cloud = reader.json(run / "cloud-finished.json")
    require(outcome["all_stopped_zero_tasks"] is True and outcome["export_error"] is None
            and outcome["failure"] is None and cloud["error"] is None,
            "Closed and fully exported engineering required")
    calls = m["plan"]["engineering"]["call_plan"]
    require(len(calls) == 16 and started["call_plan_sha256"] == summary["call_plan_sha256"] == digest(calls)
            and summary["complete"] is True and summary["engineering_passed"] is True,
            "Fixed complete engineering summary required")
    audited = {r["call_index"]: r for r in audit["operational_coverage"]["artifacts"]}
    dispositions = {r["call"]["call_index"]: r for r in summary["call_dispositions"]}
    timing_by_index = {r["call_index"]: r for r in summary["timing_calls"]}
    require(set(audited) == set(dispositions) == set(range(16)) and set(timing_by_index) == set(range(10, 16)),
            "Complete timing/artifact inventory required")
    all_seconds, samples, total_bytes = [], [], 0
    for index, call in enumerate(calls):
        path = run / "calls" / f"{index:02d}.json"
        r = reader.json(path)
        bound, artifact = reader.bind(path), r["array_artifact"]
        array = reader.bind(run / "calls" / f"{index:02d}.npz")
        require(r["call"] == call and r["passed"] is True
                and bound["byte_sha256"] == audited[index]["report_byte_sha256"]
                and array["byte_sha256"] == audited[index]["array_byte_sha256"] == artifact["byte_sha256"]
                and artifact["path"] == f"calls/{index:02d}.npz", "Audited generation bytes differ")
        all_seconds.append(number(r["generation_seconds"], "all16 generation timer", positive=True))
        gen_bytes = bound["byte_count"] + array["byte_count"]
        total_bytes += gen_bytes
        persisted = dispositions[index]["artifact"]
        require(persisted["byte_sha256"] == bound["byte_sha256"] and persisted["artifact_bytes"] == gen_bytes,
                "Persistence generation binding differs")
        if index < 10:
            continue
        path = run / "receipts" / f"{index:02d}.json"
        receipt = reader.json(path)
        receipt_bound = reader.bind(path)
        receipt_array = reader.bind(run / "receipts" / f"{index:02d}.npz")
        checked = audit["receipt_checks"][str(index)]
        require(checked["passed"] is True
                and checked["assigned_choices"] == receipt["assigned_choices"]
                and checked["actual_selected_forwards"] == receipt["actual_receipt_rows"]
                and receipt_array["byte_sha256"] == receipt["array_artifact"]["byte_sha256"]
                and receipt["array_artifact"]["path"] == f"receipts/{index:02d}.npz"
                and receipt["call_index"] == index and receipt["passed"] is True and receipt["error"] is None,
                "Audited receipt bytes differ")
        receipt_bytes = receipt_bound["byte_count"] + receipt_array["byte_count"]
        total_bytes += receipt_bytes
        require(persisted["receipt"]["byte_sha256"] == receipt_bound["byte_sha256"]
                and persisted["receipt"]["artifact_bytes"] == receipt_bytes, "Receipt persistence binding differs")
        t = timing_by_index[index]
        require(t["generation_seconds"] == r["generation_seconds"]
                and t["receipt_parsing_seconds"] == receipt["parsing_seconds"]
                and t["receipt_forward_seconds"] == receipt["receipt_forward_seconds"]
                and t["actual_receipt_rows"] == receipt["actual_receipt_rows"]
                and t["actual_receipt_width"] == receipt["actual_padded_width"]
                and t["artifact_bytes"] == gen_bytes + receipt_bytes
                and t["serialization_seconds"] == persisted["serialization_seconds"]
                and t["receipt_serialization_seconds"] == persisted["receipt"]["serialization_seconds"],
                "Summary timing differs from saved raw observations")
        samples.append(dict(source_cohort=call["source_cohort"], condition=call["condition"],
            generation_seconds=r["generation_seconds"], generation_artifact_bytes=gen_bytes,
            generation_serialization_seconds=persisted["serialization_seconds"],
            receipt_rows=receipt["actual_receipt_rows"], receipt_padded_width=receipt["actual_padded_width"],
            receipt_forward_calls=receipt["actual_forward_calls"], receipt_forward_seconds=receipt["receipt_forward_seconds"],
            receipt_parsing_seconds=receipt["parsing_seconds"], receipt_artifact_bytes=receipt_bytes,
            receipt_serialization_seconds=persisted["receipt"]["serialization_seconds"]))
    require(cloud["checkpoint_commit_total_seconds"] == sum(cloud["checkpoint_commit_seconds"])
            and cloud["final_commit_timing_excluded"] is True, "Checkpoint timing scope differs")
    lifecycle = ((started["started"] - worker_started["started"] - environment["load_seconds"])
                 + (cloud["finished"] - summary["finished"]))
    require(lifecycle >= 0, "Unavailable noncollector lifecycle decomposition")
    return m["plan"], dict(passed=True, timing_calls=samples,
        all16_generation_seconds=sum(all_seconds), collector_elapsed_seconds=summary["finished"] - started["started"],
        load_seconds=environment["load_seconds"], startup_nonloading_seconds=worker_started["started"] - admission["created"],
        lifecycle_overhead_seconds=lifecycle, checkpoint_commit_seconds=cloud["checkpoint_commit_seconds"],
        export_seconds=outcome["export_seconds"], observed_artifact_bytes=total_bytes)


def forecast_run(project, engineering_run, audit_path):
    reader = Reader()
    try:
        reader.bind(Path(project) / METHOD)
        reader.bind(__file__)
        historical = historical_inputs(project, reader)
        plan, engineering = engineering_inputs(engineering_run, audit_path, reader, project)
        result = forecast_tables(plan, historical, engineering)
        result["historical_generation_seconds_by_worker"] = {
            str(w): sum(r["generation_seconds"] for r in historical["calls"] if r["source_cohort"] % 3 == w)
            for w in range(3)}
        result["normalized_inputs_sha256"] = digest(dict(plan=plan, historical=historical, engineering=engineering))
        result["receipt_audit_byte_linkage_scope"] = (
            "Every raw engineering input and the context audit are byte-bound through the passing "
            "outer lifecycle receipt, including all selected-receipt JSON/NPZ files. This forecast grants no admission.")
        reader.verify_unchanged()
    except (ValueError, KeyError, TypeError, OSError, IndexError) as error:
        result = dict(schema_version=1, available=False, passed=False, scientific_admission=False,
            causal_admission=False, global_budget_admission=False, retained_storage_reserve_usd=STORAGE,
            error=dict(type=type(error).__name__, message=str(error)))
    result["input_files"] = list(reader.bindings.values())
    result["input_files_sha256"] = digest(result["input_files"])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--engineering-run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = forecast_run(args.project_root.resolve(), args.engineering_run.resolve(), args.audit.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("xb") as stream:
        stream.write((canonical(result) + "\n").encode())
    print(json.dumps({k: result[k] for k in ("available", "passed", "scientific_admission")}, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
