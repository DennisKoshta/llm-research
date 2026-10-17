"""Independent collection audit for the fixed coverage bridge; never model inference.

Numerical reconstruction reuses the separately implemented coverage-engineering
auditor. Source/runtime checks and receipt rendering reuse frozen engineering
helpers. Independent scoring is a separate adapter, never the producer scorer.
"""

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

import audit_coverage_engineering as numeric

GROUPS = ("worker_0", "worker_1", "worker_2")
GATES = ("sham_identity_passed", "engagement_passed", "receipts_passed",
         "cross_arm_prepatch_identity_passed", "worker_policies_equal")
ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class Reader:
    """Keep the first byte binding and reject any later input replacement."""
    def __init__(self):
        self.files = {}

    def bind(self, path):
        path = Path(path)
        require(path.is_file() and not path.is_symlink(), "Missing/unsafe file: " + str(path))
        path = path.resolve()
        with path.open("rb") as stream:
            value = dict(byte_sha256=hashlib.file_digest(stream, "sha256").hexdigest(),
                         byte_count=path.stat().st_size)
        require(str(path) not in self.files or self.files[str(path)] == value,
                "Previously bound file changed: " + str(path))
        self.files[str(path)] = value
        return value

    def json(self, path):
        bound = self.bind(path)
        payload = Path(path).read_bytes()
        require(hashlib.sha256(payload).hexdigest() == bound["byte_sha256"]
                and len(payload) == bound["byte_count"], "JSON changed while reading")
        return json.loads(payload)

    def unchanged(self):
        for path in list(self.files):
            self.bind(path)


def artifact(root, stem, reader):
    root = Path(root)
    report = reader.json(root / (stem + ".json"))
    descriptor = report["array_artifact"]
    require(descriptor["path"] == stem + ".npz", "Artifact path changed")
    path = root / descriptor["path"]
    require(reader.bind(path)["byte_sha256"] == descriptor["byte_sha256"], "Artifact bytes changed")
    with np.load(path, allow_pickle=False) as archive:
        require(len(archive.files) == len(set(archive.files)), "Duplicate NPZ member")
        arrays = {key: archive[key] for key in archive.files}
    require({key: numeric.array_binding(value) for key, value in arrays.items()}
            == descriptor["arrays"], "Artifact array inventory changed")
    reader.bind(path)
    return report, arrays


def expected_call(materials, cohort):
    """Reconstruct the scientific schedule without the collector's constructor."""
    arm = cohort["arm"]
    schedule, direction = ("record", None) if arm == "baseline" else arm.split("__")
    require(schedule in ("record", "full"), "Unknown schedule")
    index = cohort["offset"] // 32
    requests = materials["base_requests"][cohort["source_cohort"] * 32:(cohort["source_cohort"] + 1) * 32]
    return dict(call_index=index, call_id=f"bridge-{index:03d}-{arm}",
        source_cohort=cohort["source_cohort"], cap=4096, stage="scientific",
        condition="baseline" if direction is None else f"{schedule}_{'zero' if direction == 'zero_sham' else 'nonzero'}",
        schedule=numeric.RECORD if schedule == "record" else numeric.FULL,
        position=None if direction is None else "record" if schedule == "record" else "all_attended",
        direction_key=direction, dose_multiplier=0 if direction == "zero_sham" else 1,
        one_row_only=None, inject_failure_at=None, expected_error=None,
        base_trial_ids=cohort["base_trial_ids"], trial_ids=cohort["trial_ids"],
        requests_sha256=digest(requests))


def logical_receipt_groups(materials, group):
    choices = [row for row in materials["assignments"]
               if row["execution_group"] == group and row["kind"] == "choice"]
    return [choices[start:start + 32] for start in range(0, len(choices), 32)]


def relationships(report, baseline, call):
    if call["position"] is None:
        return dict(common_prefill_identity=True, sham_identity=True)
    common = bool(baseline and baseline["forwards"] and report["forwards"]
        and baseline["requests_sha256"] == report["requests_sha256"]
        and [g["before"] for g in baseline["forwards"][0]["geometry"]]
        == [g["before"] for g in report["forwards"][0]["geometry"]])
    sham = bool(baseline and numeric.identical_active_rows(baseline, report, range(32))) \
        if call["dose_multiplier"] == 0 else True
    return dict(common_prefill_identity=common, sham_identity=sham)


def verify_descriptor(root, stem, descriptor, reader):
    require(descriptor["path"] == stem + ".json"
        and descriptor["byte_sha256"] == reader.bind(Path(root) / descriptor["path"])["byte_sha256"]
        and type(descriptor["serialization_seconds"]) in (int, float)
        and math.isfinite(descriptor["serialization_seconds"]) and descriptor["serialization_seconds"] >= 0
        and descriptor["artifact_bytes"] == sum(reader.bind(Path(root) / (stem + suffix))["byte_count"]
            for suffix in (".json", ".npz")), "Terminal evidence descriptor changed")


def receipt_expected(source, receipts):
    """Only frozen sidecar/rendering utilities are shared; scoring is independent."""
    inner = source["task_record"]
    require(inner is not None, "Receipts require an available source generation")
    row = receipts.choice_sidecar(inner)
    row.update(source_trial_id=source["trial_id"], inner_source_trial_id=inner["trial_id"],
        source_record_sha256=digest(source), source_task_record_sha256=digest(inner),
        assignment_sha256=digest(source["assignment"]), source_assignment_sha256=digest(inner["assignment"]))
    if inner["cap_hit"]:
        row.update(status="capped_choice", receipt_required=False)
    return row


def receipt_guard(report, arrays, requests, ids, tokenizer, receipts, hidden_size):
    """Audit actual saved selected-payload model inputs and full-vocabulary parity."""
    guard, measurements = report["engineering"], report["observed_measurements"]
    if guard is None:
        require(measurements is None and not arrays, "Unbound receipt measurements")
        return not requests and report["actual_forward_calls"] == 0
    require(requests and report["actual_forward_calls"] == 2
        and set(guard["checks"]) == set(receipts.ENGINEERING_CHECKS)
        and all(type(value) is bool for value in guard["checks"].values())
        and type(guard["passed"]) is bool and guard["passed"] is all(guard["checks"].values())
        and guard["source_trial_ids"] == ids
        and guard["token_prefix_sha256"] == [r["token_prefix_sha256"] for r in requests]
        and guard["layers"] == [15] and guard["positions"] == ["record", "final"]
        and guard["max_new_tokens"] == 0 and guard["active_batch_size"] == len(requests)
        and len(measurements) == len(requests), "Receipt guard identity/inventory changed")
    require(set(arrays) == {"hidden"} and arrays["hidden"].dtype == np.float32
        and list(arrays["hidden"].shape) == [len(requests), 1, 2, hidden_size]
        and receipts.array_sha(arrays["hidden"]) == guard["captured_array_sha256"],
        "Receipt capture byte/layout binding changed")
    finite_bf16 = bool(np.isfinite(arrays["hidden"]).all()
        and np.all((arrays["hidden"].view(np.uint32) & np.uint32(0xFFFF)) == 0))
    width = 128 * math.ceil(max(len(r["token_prefix"]) for r in requests) / 128)
    require(guard["prefill_width"] == report["actual_padded_width"] == width, "Receipt width changed")
    for request, measurement in zip(requests, measurements, strict=True):
        tokens = request["token_prefix"]
        padding = width - len(tokens)
        require(measurement["unpadded_input_ids"] == tokens
            and measurement["input_ids"] == [tokenizer.pad_token_id] * padding + tokens
            and measurement["attention_mask"] == [0] * padding + [1] * len(tokens)
            and measurement["position_ids"] == [0] * padding + list(range(len(tokens)))
            and measurement["unpadded_token_positions"] == request["prefix_positions"]
            and measurement["record_token_index"] == padding + request["prefix_positions"]["record"]
            and measurement["token_prefix_sha256"] == request["token_prefix_sha256"]
            and measurement["exact_token_replay"] is True and measurement["prefill_width"] == width
            and measurement["prefill_padding_multiple"] == 128 and measurement["intervention"] is None
            and "completion" not in measurement and measurement["hooks_remaining"] == 0,
            "Receipt measured input/mask/position or cleanup changed")
    left, right = guard["baseline_full_initial_logits_sha256"], guard["captured_full_initial_logits_sha256"]
    require(len(left) == len(right) == len(requests) and all(numeric.valid_hash(v) for v in left + right),
            "Missing full-vocabulary receipt fingerprints")
    return guard["passed"] and left == right and finite_bf16


def audit_receipt_group(report, arrays, choices, sources, *, materials, group, index,
                        tokenizer, receipts, hidden_size):
    require(report["materials_sha256"] == digest(materials) and report["execution_group"] == group
        and report["logical_choice_offset"] == index * 32
        and report["source_trial_ids"] == [a["trial_id"] for a in choices]
        and report["assigned_choices"] == len(choices)
        and report["records_sha256"] == digest(report["records"])
        and len(report["records"]) == len(choices), "Fixed logical receipt group changed")

    def encode(messages, enable_thinking=False):
        rendered = tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=enable_thinking)
        return rendered, tokenizer(rendered, add_special_tokens=False, return_tensors="pt")

    model = SimpleNamespace(tokenizer=tokenizer, encode=encode)
    expected = [receipt_expected(sources[a["trial_id"]], receipts) for a in choices]
    active = [row for row in expected if row["receipt_required"]]
    active_ids = [row["source_trial_id"] for row in active]
    requests = []
    for row in active:
        payload, request, binding = receipts.render_receipt(sources[row["source_trial_id"]]["task_record"], model)
        row.update(receipt_input=payload, token_binding=binding)
        requests.append(request)
    error = report["error"]
    require(report["active_source_trial_ids"] == active_ids
        and report["requests_sha256"] == digest(report["requests"])
        and report["actual_receipt_rows"] == len(report["requests"])
        and type(report["passed"]) is bool and report["passed"] is (error is None),
        "Receipt eligibility/declared status changed")
    require(report["requests"] == requests[:len(report["requests"])]
        and len(report["requests"]) <= len(requests), "Receipt is not an exact selected-only prefix")
    require(type(report["receipt_forward_attempted"]) is bool
        and (not report["receipt_forward_attempted"] or report["requests"] == requests and bool(requests))
        and report["actual_forward_calls"] == (2 if report["engineering"] is not None
            else None if report["receipt_forward_attempted"] else 0), "Receipt forward disposition changed")
    if error is None:
        require(report["requests"] == requests and report["receipt_forward_attempted"] is bool(requests),
                "Successful receipt omitted a required selected payload")
    else:
        require(isinstance(error, dict) and isinstance(error.get("type"), str)
            and isinstance(error.get("message"), str), "Malformed retained receipt error")
    for row, actual in zip(expected, report["records"], strict=True):
        if row["receipt_required"]:
            offset = active_ids.index(row["source_trial_id"])
            if error is None:
                executed = deepcopy(row["executed_simulator_outcome"])
                executed["receipt_model_pass_completed"] = True
                executed["observation_ledger"][-1]["model_pass_completed"] = True
                row.update(status="completed", receipt_model_pass_completed=True,
                    measurement=report["observed_measurements"][offset], activation_row=offset,
                    executed_simulator_outcome=executed)
            else:
                if offset >= len(report["requests"]):
                    row.update(receipt_input=None, token_binding=None)
                row.update(status="execution_error", receipt_model_pass_completed=False,
                    measurement=None, activation_row=None, error=error,
                    executed_simulator_outcome=deepcopy(sources[row["source_trial_id"]]["task_record"]["simulator_outcome"]))
        require(digest(row) == digest(actual), "Receipt disposition/source/payload changed")
    guard_ok = receipt_guard(report, arrays, report["requests"], active_ids, tokenizer, receipts, hidden_size)
    return dict(passed=error is None and guard_ok, assigned_choices=len(choices),
                actual_selected_rows=len(report["requests"]), error=error)


def inventory(root, folder, allowed):
    directory = Path(root) / folder
    if not directory.exists():
        return
    require(directory.is_dir() and not directory.is_symlink(), "Unsafe collection directory")
    require(all(path.is_file() and not path.is_symlink() and path.name in allowed
                for path in directory.iterdir()), "Unexpected artifact in " + folder)


def audit_worker(materials, root, group, *, tokenizer, receipts, metadata, reader=None):
    """Read one worker; retain all slots even when a call or its persistence failed."""
    reader = reader or Reader()
    root = Path(root)
    materials_sha = digest(materials)
    require(group in GROUPS, "Unknown worker")
    assigned = [a for a in materials["assignments"] if a["execution_group"] == group]
    cohorts = [c for c in materials["execution_cohorts"] if c["execution_group"] == group]
    groups = logical_receipt_groups(materials, group)
    indices = {c["offset"] // 32 for c in cohorts}
    for folder in ("generation", "generation_cohorts", "generation_started", "generation_failures"):
        inventory(root, folder, {f"{i:03d}.{suffix}" for i in indices
            for suffix in (("json", "npz") if folder == "generation" else ("json",))})
    for folder in ("receipts", "receipt_started", "receipt_failures"):
        inventory(root, folder, {f"{i:03d}.{suffix}" for i in range(len(groups))
            for suffix in (("json", "npz") if folder == "receipts" else ("json",))})
    observed, sources, receipt_rows, generation_slots, choice_slots = {}, {}, {}, [], []
    calls, receipt_results, issues = [], [], []
    baseline, baseline_descriptor, baseline_cohort = None, None, None
    sham_passed, prepatch_passed, engagement_passed = True, True, True
    for cohort in cohorts:
        call = expected_call(materials, cohort)
        index, start = call["call_index"], cohort["source_cohort"] * 32
        rows = materials["assignments"][cohort["offset"]:cohort["offset"] + 32]
        requests = materials["base_requests"][start:start + 32]
        attempted = (root / "generation_started" / f"{index:03d}.json").exists()
        envelope, report, evidence_descriptor = None, None, None
        good = False
        try:
            if attempted:
                entry = reader.json(root / "generation_started" / f"{index:03d}.json")
                require(entry["call"] == call and entry["materials_sha256"] == materials_sha
                    and entry["assignments_sha256"] == digest(rows), "Generation attempt changed")
            if (root / "generation" / f"{index:03d}.json").exists():
                require(attempted, "Generation evidence without a recorded attempt")
                report, arrays = artifact(root, f"generation/{index:03d}", reader)
                binding = materials["direction_bindings"][call["direction_key"] or "zero_sham"]
                require(report["materials_sha256"] == materials_sha
                    and report["execution_group"] == group and report["scientific_arm"] == cohort["arm"]
                    and report["direction_binding_key"] == (call["direction_key"] or "zero_sham")
                    and report["direction_binding_sha256"] == digest(binding), "Generation arm/direction changed")
                require(report["rows"] is None or "hidden" in arrays,
                        "Completed scientific call lacks mandatory all32 prepatch captures")
                checked = numeric.audit_call(report, arrays, requests, call, binding["unit"],
                    hidden_size=metadata["hidden_size"], vocabulary_size=metadata["vocabulary_size"],
                    pad_token_id=metadata["pad_token_id"], eos_ids=metadata["generation_eos_token_ids"])
                if report["rows"] is not None:
                    for measured in report["rows"]:
                        completion = measured["completion"]
                        require(completion["raw"] == tokenizer.decode(completion["generated_token_ids"],
                            skip_special_tokens=True), "Raw text differs from actual tokens")
                require(call["position"] is None or baseline_cohort == cohort["source_cohort"],
                        "Missing same-cohort baseline")
                common = relationships(report, baseline, call)
                require(report["bridge_relationship_checks"] == common
                    and report["baseline_artifact"] == baseline_descriptor,
                    "Saved baseline relationship does not reproduce")
                producer_good = report["passed"] is True and all(common.values())
                require(report["bridge_engineering_passed"] is producer_good,
                        "Saved combined generation gate does not reproduce")
                good = checked["passed"] and all(common.values())
                engagement_passed &= checked["passed"]
                prepatch_passed &= common["common_prefill_identity"]
                if call["dose_multiplier"] == 0:
                    sham_passed &= common["sham_identity"]
                calls.append(dict(call_index=index, passed=good, checks=checked["checks"],
                    relationships=common, numerical_receipt_sha256=digest(checked)))
            path = root / "generation_cohorts" / f"{index:03d}.json"
            if path.exists():
                envelope = reader.json(path)
                require(report is not None and envelope["materials_sha256"] == materials_sha
                    and envelope["execution_group"] == group and envelope["cohort"] == cohort
                    and envelope["call"] == call and len(envelope["records"]) == 32
                    and envelope["records_sha256"] == digest(envelope["records"])
                    and envelope["engineering_passed"] is report["bridge_engineering_passed"],
                    "Generation terminal envelope changed")
                evidence_descriptor = envelope["artifact"]
                verify_descriptor(root, f"generation/{index:03d}", evidence_descriptor, reader)
                for offset, (assignment, record) in enumerate(zip(rows, envelope["records"], strict=True)):
                    require(record["trial_id"] == assignment["trial_id"] and record["assignment"] == assignment
                        and record["assignment_sha256"] == digest(assignment) and record["kind"] == assignment["kind"]
                        and record["terminal"] is True and record["cohort_row"] == offset
                        and record["generation_evidence"] == evidence_descriptor
                        and record["engineering_passed"] is envelope["engineering_passed"],
                        "Generation record assignment/evidence changed")
                    completion = None if report["rows"] is None else report["rows"][offset]["completion"]
                    require(record["status"] == ("execution_error" if completion is None else "completed")
                        and (record["task_record"] is None) == (completion is None), "Generation disposition changed")
                    sources[assignment["trial_id"]] = record
                    observed[assignment["trial_id"]] = dict(assignment=assignment, task_record=record["task_record"],
                        completion=completion, record_sha256=digest(record), status=record["status"])
                if call["position"] is None:
                    baseline, baseline_descriptor, baseline_cohort = report, evidence_descriptor, cohort["source_cohort"]
        except (ValueError, KeyError, TypeError, OSError, IndexError) as error:
            issues.append(dict(call_index=index, type=type(error).__name__, message=str(error)))
            good = False
        for assignment in rows:
            key = assignment["trial_id"]
            if key not in observed:
                observed[key] = dict(assignment=assignment, task_record=None, completion=None, record_sha256=None,
                    status="execution_error" if attempted else "missing")
            row = observed[key]
            generation_slots.append(dict(trial_id=key, assignment_sha256=digest(assignment), status=row["status"],
                record_sha256=row["record_sha256"], engineering_passed=good))
        if report is None or envelope is None:
            engagement_passed = False
            prepatch_passed = False
            if call["dose_multiplier"] == 0:
                sham_passed = False
    for index, choices in enumerate(groups):
        attempted = (root / "receipt_started" / f"{index:03d}.json").exists()
        try:
            if attempted:
                entry = reader.json(root / "receipt_started" / f"{index:03d}.json")
                require(entry["materials_sha256"] == materials_sha and entry["logical_choice_offset"] == index * 32
                    and entry["source_trial_ids"] == [a["trial_id"] for a in choices], "Receipt attempt/group changed")
            if (root / "receipts" / f"{index:03d}.json").exists():
                require(attempted and len(sources) == len(assigned), "Receipt before complete generation inventory")
                report, arrays = artifact(root, f"receipts/{index:03d}", reader)
                checked = audit_receipt_group(report, arrays, choices, sources, materials=materials, group=group,
                    index=index, tokenizer=tokenizer, receipts=receipts, hidden_size=metadata["hidden_size"])
                receipt_results.append(dict(index=index, **checked))
                receipt_rows.update({row["source_trial_id"]: row for row in report["records"]})
        except (ValueError, KeyError, TypeError, OSError, IndexError) as error:
            issues.append(dict(receipt_group=index, type=type(error).__name__, message=str(error)))
        for assignment in choices:
            row = receipt_rows.get(assignment["trial_id"])
            choice_slots.append(dict(trial_id=assignment["trial_id"], assignment_sha256=digest(assignment),
                logical_choice_offset=index * 32,
                status=row["status"] if row else "execution_error" if attempted else "missing",
                receipt_required=None if row is None else row["receipt_required"],
                receipt_model_pass_completed=False if row is None else row["receipt_model_pass_completed"],
                sidecar_sha256=None if row is None else digest(row)))
    if (root / "bridge-summary.json").exists():
        summary = reader.json(root / "bridge-summary.json")
        # Failure flags stay failures; summary cannot manufacture completed slots.
        if not (summary["materials_sha256"] == materials_sha and summary["execution_group"] == group
                and summary["generation_slots"] == generation_slots and summary["choice_slots"] == choice_slots):
            issues.append(dict(summary_error="Saved slot inventory differs from durable records"))
    complete = len(sources) == len(assigned) and len(receipt_rows) == sum(map(len, groups))
    return dict(execution_group=group, complete=complete, integrity_passed=not issues,
        assigned_generations=len(assigned), assigned_choices=sum(map(len, groups)),
        observed=observed, receipt_rows=receipt_rows, generation_slots=generation_slots, choice_slots=choice_slots,
        call_checks=calls, receipt_checks=receipt_results, issues=issues,
        sham_identity_passed=sham_passed and len(calls) == len(cohorts),
        engagement_passed=engagement_passed and len(calls) == len(cohorts),
        cross_arm_prepatch_identity_passed=prepatch_passed and len(calls) == len(cohorts),
        receipts_passed=len(receipt_results) == len(groups) and all(r["passed"] for r in receipt_results))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runtime_policy(metadata):
    """Loading duration is the sole non-policy field in the recorded metadata."""
    return {key: value for key, value in metadata.items() if key != "load_seconds"}


def require_worker_evidence(root, *, complete=False):
    root = Path(root)
    started = complete or any((root / folder).exists() and any((root / folder).iterdir())
        for folder in ("generation_started", "generation", "generation_cohorts"))
    if started:
        names = ["worker-started.json", "bridge-started.json", "bridge-environment.json"]
        if complete:
            names += ["summary.json", "bridge-summary.json"]
        require(all((root / name).is_file() and not (root / name).is_symlink() for name in names),
                "Collected worker lacks required startup/completion evidence")


def source_context(run, context_root, reader, tokenizer=None):
    """Verify inert inventories before importing their source/runtime helpers."""
    run = Path(run)
    manifest, freeze = reader.json(run / "manifest.json"), reader.json(run / "freeze.json")
    require(digest(manifest) == freeze["manifest_sha256"], "Parent manifest/freeze changed")
    context_root = Path(context_root or freeze["execution_root"]).resolve()
    require(Path(freeze["execution_root"]).resolve() == context_root
        and freeze["source_sha256"] == manifest["source_sha256"]
        and freeze["original_source_archive_sha256"] == manifest["original_source_archive_sha256"]
        and freeze["runtime_files_sha256"] == digest(manifest["runtime_files"])
        and freeze["wrapper_files_sha256"] == digest(manifest["wrapper_files"]),
        "Frozen context identity changed")
    original, runtime, wrapper = (context_root / name for name in ("original", "runtime", "bridge"))
    for base, key in ((original, "original_execution_files"), (runtime, "runtime_files"), (wrapper, "wrapper_files")):
        require(not base.is_symlink(), "Unsafe source root")
        for relative, checksum in manifest[key].items():
            path = Path(relative)
            require(not path.is_absolute() and ".." not in path.parts and str(path) == relative,
                    "Unsafe frozen source path")
            require(reader.bind(base / relative)["byte_sha256"] == checksum, "Frozen source bytes changed")
    require(reader.bind(__file__)["byte_sha256"] == manifest["wrapper_files"][Path(__file__).name]
        and reader.bind(numeric.__file__)["byte_sha256"] == manifest["wrapper_files"]["audit_coverage_engineering.py"],
        "Run the reviewed frozen auditor bytes")
    worker = load(wrapper / "curiosity_coverage_bridge_worker.py", "_coverage_scientific_worker_audit")
    modules = worker.verify_sources(manifest, original, runtime, wrapper)
    admission = reader.json(run / "admission.json")
    worker.validate_admission(manifest, admission, time.time(), historical=True)
    scorer = load(wrapper / "score_curiosity_coverage_bridge.py", "_coverage_independent_scorer")
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(manifest["plan"]["model"]["model_id"],
            revision=manifest["plan"]["model"]["revision"], local_files_only=True, trust_remote_code=False)
    require(worker.tokenizer_preflight(manifest["plan"], tokenizer, modules["voluntary_resolution"])
            == manifest["preflight"], "Frozen tokenizer/prefix preflight does not reproduce")
    return manifest, admission, original, modules, worker, scorer, tokenizer


def audit(run, *, context_root=None):
    """Return a core receipt without writing. Lifecycle composition is mandatory."""
    run, reader = Path(run).resolve(), Reader()
    manifest, admission, original, modules, worker, scorer, tokenizer = source_context(run, context_root, reader)
    materials = manifest["plan"]
    observed, receipt_rows, workers, policies, issues = {}, {}, [], [], []
    for group in GROUPS:
        root = run / "workers" / group
        metadata = dict(hidden_size=5120, vocabulary_size=151936, pad_token_id=151643,
                        generation_eos_token_ids=[151643, 151645])
        runtime_ok = False
        try:
            require(reader.json(root / "manifest.json") == manifest and reader.json(root / "admission.json") == admission,
                    "Worker parent/admission differs")
            claim = reader.json(root / "worker-claim.json")
            worker.validate_worker_claim(manifest, admission, claim, group, time.time(), historical=True)
            require_worker_evidence(root)
            if (root / "environment.json").exists():
                metadata = reader.json(root / "environment.json")
                worker.validate_runtime_metadata(metadata, original)
                require(all(metadata[key] == value for key, value in materials["model"].items())
                    and metadata["template_sha256"] == digest(tokenizer.chat_template)
                    and metadata["pad_token_id"] == tokenizer.pad_token_id,
                    "Actual worker model/tokenizer differs")
                policies.append(runtime_policy(metadata))
                runtime_ok = True
            if (root / "worker-started.json").exists():
                started = reader.json(root / "worker-started.json")
                require(started["manifest_sha256"] == digest(manifest) and started["execution_group"] == group
                    and started["admission_sha256"] == digest(admission)
                    and admission["created"] <= started["started"] < admission["deadline"], "Worker start binding changed")
            if (root / "bridge-started.json").exists():
                started = reader.json(root / "bridge-started.json")
                require(started["materials_sha256"] == digest(materials) and started["execution_group"] == group
                    and started["deadline"] == admission["deadline"] - 60 and started["resume_allowed"] is False
                    and started["parent_generations"] == 8352 and started["parent_choices"] == 4176,
                    "Collector deadline, source or parent denominator changed")
                base_metadata = reader.json(root / "bridge-environment.json")
                require(runtime_ok and all(metadata.get(k) == v for k, v in base_metadata.items()),
                        "Generation runtime differs from verified worker")
        except (ValueError, KeyError, TypeError, OSError, IndexError) as error:
            issues.append(dict(execution_group=group, phase="runtime", type=type(error).__name__, message=str(error)))
        try:
            result = audit_worker(materials, root, group, tokenizer=tokenizer,
                receipts=modules["voluntary_capture"], metadata=metadata, reader=reader)
        except (ValueError, KeyError, TypeError, OSError, IndexError) as error:
            # An unsafe/extra artifact cannot remove the worker's assigned rows.
            result = dict(execution_group=group, complete=False, integrity_passed=False,
                observed={a["trial_id"]: dict(assignment=a, task_record=None, completion=None, record_sha256=None,
                    status="execution_error" if (root / "bridge-started.json").exists() else "missing")
                    for a in materials["assignments"] if a["execution_group"] == group},
                receipt_rows={}, generation_slots=[], choice_slots=[], call_checks=[], receipt_checks=[],
                issues=[dict(type=type(error).__name__, message=str(error))],
                **{key: False for key in GATES if key != "worker_policies_equal"})
        if result["complete"]:
            try:
                require_worker_evidence(root, complete=True)
            except ValueError as error:
                result["issues"].append(dict(evidence_presence_error=str(error)))
                result["integrity_passed"] = False
        if (root / "summary.json").exists():
            try:
                summary = reader.json(root / "summary.json")
                collector = reader.json(root / "bridge-summary.json") if (root / "bridge-summary.json").exists() else None
                require(summary["manifest_sha256"] == digest(manifest)
                    and summary["admission_sha256"] == digest(admission)
                    and summary["worker_claim_sha256"] == digest(reader.json(root / "worker-claim.json"))
                    and summary["execution_group"] == group
                    and summary["collector_summary_sha256"] == (digest(collector) if collector else None),
                    "Worker summary source/collector binding changed")
            except (ValueError, KeyError, TypeError, OSError, IndexError) as error:
                result["issues"].append(dict(summary_error=str(error)))
                result["integrity_passed"] = False
        result["runtime_verified"] = runtime_ok
        observed.update(result.pop("observed"))
        receipt_rows.update(result.pop("receipt_rows"))
        workers.append(result)
    scoring = scorer.score_collection(materials, observed)
    semantics = scorer.check_receipts(materials, observed, receipt_rows)
    require(all(type(v) is bool for v in scoring["checks"].values())
        and all(type(v) is bool for v in semantics["checks"].values())
        and len(scoring["outcomes"]) == 8352, "Scoring adapter lost assigned slots or strict flags")
    checks = {key: all(w[key] for w in workers) for key in GATES if key != "worker_policies_equal"}
    checks["worker_policies_equal"] = len(policies) == 3 and all(p == policies[0] for p in policies)
    checks["receipts_passed"] &= all(semantics["checks"].values())
    complete = all(w["complete"] for w in workers) and scoring["checks"]["complete_response_coverage"]
    integrity = not issues and all(w["integrity_passed"] and w["runtime_verified"] for w in workers) \
        and scoring["checks"]["computation_agrees"] and semantics["checks"]["semantic_agreement"]
    reader.unchanged()
    return dict(schema_version=1, assay="curiosity_coverage_bridge_collection_audit",
        manifest_sha256=digest(manifest), parent_manifest_sha256=digest(manifest),
        materials_sha256=digest(materials), assignment_sha256=materials["assignment_sha256"],
        admission_sha256=digest(admission), complete=complete, integrity_passed=integrity,
        passed=bool(complete and integrity and all(checks.values()) and scoring["readiness"]["passed"]),
        checks=checks, **checks, readiness=scoring["readiness"], outcomes=scoring["outcomes"],
        outcome_sha256=digest(scoring["outcomes"]), assigned_generations=8352, assigned_choices=4176,
        generation_record_bindings_sha256=digest([{k: r[k] for k in ("assignment", "record_sha256", "status")}
            for r in observed.values()]), receipt_bindings_sha256=digest(receipt_rows),
        independent_scoring={key: value for key, value in scoring.items() if key not in ("outcomes", "readiness")},
        independent_receipt_semantics=semantics, workers=workers, issues=issues,
        raw_input_files=reader.files, raw_input_files_sha256=digest(reader.files),
        source_provenance_verified=True, runtime_verified=all(w["runtime_verified"] for w in workers),
        lifecycle_shutdown_verified=False, scientific_admission=False, causal_admission=False,
        limitations=["Lifecycle, global admission mirrors, app shutdown and spending require the separate composed audit.",
            "Raw BF16 arrays independently reconstruct saved row0 samples; unsaved rows retain online all-position guards and hashes.",
            "Frozen source/runtime validation and receipt sidecar/render helpers are shared. Raw scoring and numerical reconstruction are independent.",
            "Readiness is the declared instrument criterion, not belief/confidence preservation or experience evidence."])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--context-root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run, context_root=args.context_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("complete", "integrity_passed", "checks", "passed")}))
