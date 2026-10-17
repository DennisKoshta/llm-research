"""One fixed 16-call engineering sequence; no scientific or causal admission.

The immutable C runtime receives exactly two declared source overlays. Source,
tokenizer and actual-selected-receipt checks complement the numerical auditor.
There are no additional synthetic startup forwards or automatic retries.
"""

import argparse
from copy import deepcopy
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import tomllib
from types import SimpleNamespace

NAME = "qwen32-curiosity-coverage-engineering-v1"
MANIFEST_SHA = "535f3c1237a8b876fa1059d3f9949f178a55b5627e2be412008ef08cbc90b02f"
SOURCE_SHA = "1fd4079ed60fbd0169f702ab9c8f99a75d170450dfbd364bc4cb877e27c18f9a"
ARCHIVE_SHA = "4b82bdc563a3b257fd1b5e0302f5889cb6e14e30da9fcb950cc7f3039eb62ca4"
MATERIALS = "prepare_curiosity_coverage_bridge.py"
NUMERICAL_AUDITOR = "audit_coverage_engineering.py"
OVERLAYS = ("state_backend.py", "coverage_engineering.py")
RATE, WALL, TEARDOWN, CEILING = 0.00139376, 3400, 120, 5.0
APP = "curiosity-study-coverage-engineering-h200"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


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
        app_name=APP,
        gpu="H200",
        cpu=2,
        memory_gib=48,
        wall_seconds=WALL,
        teardown_seconds=TEARDOWN,
        rate_usd_per_second=RATE,
        ceiling_usd=CEILING,
        maximum_liability_usd=(WALL + TEARDOWN) * RATE,
        max_containers=1,
        retries=0,
    )


def check_files(root, inventory):
    root = Path(root)
    require(
        isinstance(inventory, dict)
        and inventory
        and root.is_dir()
        and not root.is_symlink()
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


def load_runtime(runtime):
    runtime = Path(runtime).resolve()
    sys.path.insert(0, str(runtime))
    names = (
        "artifacts",
        "batched",
        "model",
        "state_backend",
        "coverage_engineering",
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


def validate_manifest(manifest):
    original, freeze, plan = manifest["original_manifest"], manifest["original_freeze"], manifest["plan"]
    require(
        manifest["schema_version"] == 1
        and manifest["name"] == NAME
        and manifest["assay"] == "curiosity_coverage_engineering"
        and digest(original) == freeze["manifest_sha256"] == MANIFEST_SHA
        and original["source_sha256"] == freeze["source_sha256"] == manifest["source_sha256"] == SOURCE_SHA
        and freeze["archive_sha256"] == manifest["original_source_archive_sha256"] == ARCHIVE_SHA,
        "Original package identity differs",
    )
    require(
        manifest["resources"] == resources()
        and manifest["engineering_only"] is True
        and manifest["scientific_admission"] is False
        and manifest["causal_admission"] is False,
        "Fixed engineering resource/scope differs",
    )
    require(
        digest(plan) == manifest["plan_sha256"]
        and plan["model"] == original["model"]
        and plan["assignment_sha256"] == digest(plan["assignments"])
        and len(plan["assignments"]) == 8352
        and plan["scientific_admission"] is False
        and plan["engineering_admission"] is False
        and plan["causal_admission"] is False,
        "Draft material identity/scope differs",
    )
    engineering = plan["engineering"]
    calls = engineering["call_plan"]
    require(
        len(calls) == 16
        and engineering["call_plan_sha256"] == digest(calls)
        and [c["call_index"] for c in calls] == list(range(16))
        and len({trial for c in calls for trial in c["trial_ids"]}) == 512
        and all(len(c["trial_ids"]) == 32 for c in calls)
        and engineering["assigned_logical_slots"] == 512,
        "Fixed 512 engineering slots differ",
    )
    preflight = manifest["preflight"]
    require(
        digest(preflight) == manifest["preflight_sha256"]
        and preflight["passed"] is True
        and preflight["checked_requests"] == 288
        and preflight["all_original_prefixes_exact"] is True
        and preflight["source_requests_sha256"] == plan["source_generation_preflight"]["requests_sha256"],
        "Exact original tokenization is required",
    )
    require(set(manifest["runtime_delta"]) == set(OVERLAYS), "Exactly two reviewed overlays required")
    return manifest


def verify_sources(manifest, original_root, runtime, wrapper):
    validate_manifest(manifest)
    original_root, runtime, wrapper = map(Path, (original_root, runtime, wrapper))
    for root, key in (
        (original_root, "original_execution_files"),
        (runtime, "runtime_files"),
        (wrapper, "wrapper_files"),
    ):
        check_files(root, manifest[key])
    source = original_root / "src/curiosity_study"
    require(
        digest({str(p.relative_to(source)): p.read_text() for p in sorted(source.rglob("*.py"))})
        == SOURCE_SHA,
        "Original source inventory differs",
    )
    expected = {
        name.removeprefix("src/"): checksum
        for name, checksum in manifest["original_execution_files"].items()
        if name.startswith("src/curiosity_study/")
    }
    for name in OVERLAYS:
        key = "curiosity_study/" + name
        delta = manifest["runtime_delta"][name]
        require(delta["original_byte_sha256"] == expected.get(key), "Overlay original binding differs")
        expected[key] = delta["replacement_byte_sha256"]
    require(manifest["runtime_files"] == expected, "Undeclared runtime modification")
    source = runtime / "curiosity_study"
    require(
        digest({str(p.relative_to(source)): p.read_text() for p in sorted(source.rglob("*.py"))})
        == manifest["runtime_source_sha256"],
        "Derivative source inventory differs",
    )
    builder = load_module(wrapper / MATERIALS, "_coverage_frozen_materials")
    original, plan = manifest["original_manifest"], manifest["plan"]
    directions, inventory = builder.validate_directions(
        original,
        {
            name: (original_root / name).read_bytes()
            for name in (builder.DIRECTIONS_PATH, builder.DIRECTION_INVENTORY_PATH)
        },
    )
    modules = load_runtime(runtime)
    rebuilt = builder.build_materials(
        original["source_screen_manifest"],
        original,
        directions,
        inventory,
        plan["source_provenance"],
        modules["coverage_engineering"].call_plan,
    )
    require(rebuilt == plan, "Materials differ from complete inherited inputs")
    provenance = plan["source_provenance"]
    require(
        provenance["material_builder_byte_sha256"] == sha(wrapper / MATERIALS)
        and provenance["plan_byte_sha256"] == sha(wrapper / "plan.md")
        and provenance["engineering_helper_byte_sha256"] == sha(source / "coverage_engineering.py"),
        "Builder/plan/collector provenance differs",
    )
    return modules


def validate_admission(manifest, admission, now, historical=False):
    require(
        all(
            type(v) in (int, float) and math.isfinite(v)
            for v in (admission["created"], admission["deadline"], now)
        )
        and admission["created"] >= 0
        and admission["manifest_sha256"] == digest(manifest)
        and admission["phase_consumed"] is True
        and admission["retry_allowed"] is False
        and admission["deadline"] == admission["created"] + WALL
        and (historical or admission["created"] <= now < admission["deadline"])
        and admission["maximum_liability_usd"] == resources()["maximum_liability_usd"],
        "Invalid or expired one-time admission",
    )


def tokenizer_preflight(plan, tokenizer, task):
    checks = []
    for base, item in zip(
        plan["base_assignments"], plan["source_generation_preflight"]["requests"], strict=True
    ):
        request = task.inference_request(base["source_assignment"])
        rendered = tokenizer.apply_chat_template(
            request["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        tokenized = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        offsets = [list(pair) for pair in tokenized["offset_mapping"]]
        ids = tokenized["input_ids"]
        require(
            digest(request) == base["inference_request_sha256"]
            and rendered == item["rendered_prompt"]
            and offsets == item["offset_mapping"]
            and ids == item["request"]["token_prefix"]
            and request["seed"] == item["request"]["seed"]
            and request["labels"] == "ABCD",
            "Actual original prompt/token/sampler differs",
        )
        checks.append(
            dict(
                base_trial_id=base["base_trial_id"],
                input_ids_sha256=digest(ids),
                offsets_sha256=digest(offsets),
                rendered_prompt_sha256=digest(rendered),
            )
        )
    require(len(checks) == 288, "All original 288 prefixes required")
    return dict(
        passed=True,
        checked_requests=288,
        all_original_prefixes_exact=True,
        source_requests_sha256=plan["source_generation_preflight"]["requests_sha256"],
        tokenizer_template_sha256=digest(tokenizer.chat_template),
        pad_token_id=tokenizer.pad_token_id,
        requests=checks,
        requests_sha256=digest(checks),
        model_outputs_generated=False,
    )


def runtime_metadata(model, original_root):
    metadata = model.metadata()
    eos = model.model.generation_config.eos_token_id
    eos = [eos] if type(eos) is int else eos
    require(
        isinstance(eos, (list, tuple)) and eos and all(type(v) is int and v >= 0 for v in eos),
        "Invalid EOS configuration",
    )
    metadata.update(
        generation_eos_token_ids=sorted(set(eos)),
        pad_token_id=model.tokenizer.pad_token_id,
        hidden_size=int(model.model.config.hidden_size),
        vocabulary_size=int(model.model.config.vocab_size),
        inherited_synthetic_startup_forwards=0,
        startup_certificate="fixed16_coverage_calls",
    )
    validate_runtime_metadata(metadata, original_root)
    return metadata


def validate_runtime_metadata(metadata, original_root):
    lock = tomllib.loads((Path(original_root) / "uv.lock").read_text())
    versions = {p["name"]: p["version"] for p in lock["package"] if "version" in p}
    require(
        "H200" in metadata["gpu"]
        and metadata["device"] == "cuda"
        and metadata["dtype"] == "bfloat16"
        and metadata["attention"] == "eager"
        and metadata["python"].startswith("3.12.")
        and metadata["backend"] == "transformers"
        and metadata["execution_backend"] == "torch_batched_v1"
        and metadata["logits_to_keep"] == 1
        and metadata["layers"] == 64
        and metadata["cuda"] == "12.4"
        and metadata["cuda_numerics"]["policy"] == "bf16_reduced_precision_reduction_disabled_v1"
        and metadata["cuda_numerics"]["allow_bf16_reduced_precision_reduction"] is False
        and metadata["cuda_numerics"]["allow_fp16_reduced_precision_reduction"] is True
        and metadata["cuda_numerics"]["allow_tf32"] is False
        and metadata["loading_policy"] == "direct_device_sharded"
        and metadata["hidden_size"] == 5120
        and metadata["vocabulary_size"] == 151936
        and metadata["pad_token_id"] == 151643
        and metadata["generation_eos_token_ids"] == [151643, 151645]
        and metadata["inherited_synthetic_startup_forwards"] == 0,
        "Runtime/arithmetic policy differs",
    )
    require(
        metadata["versions"]
        == {name: versions[name] for name in ("torch", "transformers", "accelerate", "tokenizers")},
        "Runtime package versions differ from original lock",
    )


def operational_coverage(root, manifest):
    """Hard-kill accounting: slots never disappear; this is not a numerical audit."""
    root, slots, issues, artifacts = Path(root), [], [], []
    calls = manifest["plan"]["engineering"]["call_plan"]
    expected_names = {f"{c['call_index']:02d}.json" for c in calls}
    for folder in ("calls", "attempts", "failures"):
        if (root / folder).exists():
            extras = {p.name for p in (root / folder).glob("*.json")} - expected_names
            if extras:
                issues.append(dict(folder=folder, extra_names=sorted(extras)))
    for call in calls:
        name = f"{call['call_index']:02d}.json"
        attempted, path = (root / "attempts" / name).exists(), root / "calls" / name
        status = "execution_error" if attempted or path.exists() else "missing"
        if path.exists():
            try:
                attempt, report = read(root / "attempts" / name), read(path)
                require(
                    attempt["call"] == call
                    and attempt["materials_sha256"] == manifest["plan_sha256"]
                    and report["call"] == call,
                    "Attempt/call input binding differs",
                )
                descriptor = report["array_artifact"]
                require(
                    descriptor["path"] == f"calls/{call['call_index']:02d}.npz"
                    and sha(root / descriptor["path"]) == descriptor["byte_sha256"],
                    "Call array artifact differs",
                )
                if call["expected_error"]:
                    require(
                        report["passed"] is True
                        and report["fault_injected"] is True
                        and report["error"]
                        == dict(type="ControlledCoverageFault", message=call["inject_failure_at"])
                        and report["completed_forwards"]
                        == (1 if call["inject_failure_at"] == "after_prefill" else 2),
                        "Expected fault was not positively verified",
                    )
                    status = "expected_fault"
                elif report["rows"] is not None:
                    require(len(report["rows"]) == 32, "Incomplete atomic call rows")
                    status = "completed"
                artifacts.append(
                    dict(
                        call_index=call["call_index"],
                        report_byte_sha256=sha(path),
                        array_byte_sha256=descriptor["byte_sha256"],
                        report_bytes=path.stat().st_size,
                        array_bytes=(root / descriptor["path"]).stat().st_size,
                    )
                )
            except Exception as caught:
                issues.append(dict(call_index=call["call_index"], error=str(caught)))
        slots.extend(
            dict(trial_id=trial, base_trial_id=base, call_index=call["call_index"], status=status)
            for trial, base in zip(call["trial_ids"], call["base_trial_ids"], strict=True)
        )
    require(len(slots) == len({r["trial_id"] for r in slots}) == 512, "Fixed 512-slot denominator differs")
    return dict(
        manifest_sha256=digest(manifest),
        assigned_slots=512,
        slots=slots,
        issues=issues,
        artifacts=artifacts,
        complete=not issues and all(r["status"] in ("completed", "expected_fault") for r in slots),
        numeric_engineering_verified=False,
        selected_receipts_verified=False,
        scientific_admission=False,
    )


def load_artifacts(root, folder, index, np):
    path = Path(root) / folder / f"{index:02d}.json"
    report = read(path)
    descriptor = report["array_artifact"]
    require(
        descriptor["path"] == f"{folder}/{index:02d}.npz"
        and sha(Path(root) / descriptor["path"]) == descriptor["byte_sha256"],
        "Artifact path/bytes differ",
    )
    with np.load(Path(root) / descriptor["path"], allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    bindings = {
        key: dict(
            dtype=str(value.dtype),
            shape=list(value.shape),
            byte_sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
        )
        for key, value in arrays.items()
    }
    require(bindings == descriptor["arrays"], "Artifact array inventory differs")
    return report, arrays


def audit_receipts(report, arrays, call, generation, materials, modules, tokenizer, np):
    """Reconstruct selected contexts; reuse the immutable parser/renderer, not generation."""
    task, receipts = modules["voluntary_resolution"], modules["voluntary_capture"]
    require(
        report["call_index"] == call["call_index"] and report["call_id"] == call["call_id"],
        "Receipt call identity differs",
    )
    source_assignments = materials["engineering"]["source_assignments_by_cohort"][str(call["source_cohort"])]
    source = [
        task.record_from_thinking_inference(a, row["completion"])
        for a, row in zip(source_assignments, generation["rows"], strict=True)
    ]
    require(
        digest(source) == digest(report["source_records"]), "Receipt source differs from actual generation"
    )
    requests, ids, expected_rows, active = [], [], [], []

    def encode(messages, enable_thinking=False):
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )
        return rendered, tokenizer(rendered, add_special_tokens=False, return_tensors="pt")

    model = SimpleNamespace(tokenizer=tokenizer, encode=encode)
    for index, row in enumerate(source):
        if source_assignments[index]["kind"] != "choice":
            continue
        expected = receipts.choice_sidecar(row)
        expected.update(
            source_trial_id=call["trial_ids"][index],
            base_trial_id=call["base_trial_ids"][index],
            generation_row=index,
            source_assignment_sha256=digest(source_assignments[index]),
        )
        if row["cap_hit"]:
            expected.update(status="capped_choice", receipt_required=False)
        if expected["receipt_required"]:
            payload, request, binding = receipts.render_receipt(row, model)
            active.append(len(expected_rows))
            requests.append(request)
            ids.append(call["trial_ids"][index])
            expected.update(receipt_input=payload, token_binding=binding)
        expected_rows.append(expected)
    require(
        report["requests"] == requests
        and report["requests_sha256"] == digest(requests)
        and report["active_source_trial_ids"] == ids
        and report["assigned_choices"] == len(expected_rows)
        and report["actual_receipt_rows"] == len(requests)
        and report["receipt_forward_attempted"] is bool(requests)
        and len(report["records"]) == len(expected_rows),
        "Selected-only receipt request inventory differs",
    )
    for index, (expected, actual) in enumerate(zip(expected_rows, report["records"], strict=True)):
        if index in active:
            active_row = active.index(index)
            executed = deepcopy(expected["executed_simulator_outcome"])
            executed["receipt_model_pass_completed"] = True
            executed["observation_ledger"][-1]["model_pass_completed"] = True
            expected.update(
                status="completed",
                receipt_model_pass_completed=True,
                measurement=report["observed_measurements"][active_row],
                activation_row=active_row,
                executed_simulator_outcome=executed,
            )
        require(digest(expected) == digest(actual), "Receipt disposition/payload execution differs")
    require(
        report["records_sha256"] == digest(report["records"])
        and report["passed"] is True
        and report["error"] is None
        and report["actual_forward_calls"] == (2 if requests else 0),
        "Receipt completion gate differs",
    )
    if requests:
        guard = report["engineering"]
        require(
            guard["passed"] is True
            and set(guard["checks"]) == set(receipts.ENGINEERING_CHECKS)
            and all(v is True for v in guard["checks"].values())
            and guard["source_trial_ids"] == ids
            and guard["token_prefix_sha256"] == [r["token_prefix_sha256"] for r in requests]
            and guard["layers"] == [15]
            and guard["positions"] == ["record", "final"]
            and guard["max_new_tokens"] == 0
            and guard["active_batch_size"] == len(requests)
            and len(report["observed_measurements"]) == len(requests)
            and len(guard["baseline_full_initial_logits_sha256"]) == len(requests)
            and all(
                isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")
                for value in guard["baseline_full_initial_logits_sha256"]
            )
            and guard["baseline_full_initial_logits_sha256"] == guard["captured_full_initial_logits_sha256"]
            and set(arrays) == {"hidden"}
            and arrays["hidden"].dtype == np.float32
            and np.isfinite(arrays["hidden"]).all()
            and np.all((arrays["hidden"].view(np.uint32) & np.uint32(0xFFFF)) == 0)
            and list(arrays["hidden"].shape) == [len(requests), 1, 2, 5120]
            and receipts.array_sha(arrays["hidden"]) == guard["captured_array_sha256"],
            "Receipt read-only guard differs",
        )
        width = 128 * math.ceil(max(len(r["token_prefix"]) for r in requests) / 128)
        require(guard["prefill_width"] == report["actual_padded_width"] == width, "Receipt width differs")
        for request, measurement in zip(requests, report["observed_measurements"], strict=True):
            tokens = request["token_prefix"]
            padding = width - len(tokens)
            require(
                measurement["unpadded_input_ids"] == tokens
                and measurement["input_ids"] == [tokenizer.pad_token_id] * padding + tokens
                and measurement["attention_mask"] == [0] * padding + [1] * len(tokens)
                and measurement["position_ids"] == [0] * padding + list(range(len(tokens)))
                and measurement["unpadded_token_positions"] == request["prefix_positions"]
                and measurement["record_token_index"] == padding + request["prefix_positions"]["record"]
                and measurement["token_prefix_sha256"] == request["token_prefix_sha256"]
                and measurement["exact_token_replay"] is True
                and measurement["prefill_width"] == width
                and measurement["prefill_padding_multiple"] == 128
                and measurement["intervention"] is None
                and "completion" not in measurement
                and measurement["hooks_remaining"] == 0,
                "Receipt actual model input differs",
            )
    else:
        require(
            not arrays and report["engineering"] is None and report["observed_measurements"] is None,
            "Unexpected receipt forward without selected payload",
        )
    return dict(
        passed=True,
        assigned_choices=len(expected_rows),
        actual_selected_forwards=len(requests),
        scope="Immutable original parsing/rendering reused; saved input and equal full-logit fingerprints checked. No new model pass.",
    )


def audit_context(root, original_root, runtime, wrapper, tokenizer=None):
    """Postrun composition; numeric observer audit and receipt/source scopes explicit."""
    import numpy as np

    root, wrapper = Path(root), Path(wrapper)
    manifest, admission = read(root / "manifest.json"), read(root / "admission.json")
    modules = verify_sources(manifest, original_root, runtime, wrapper)
    validate_admission(manifest, admission, time.time(), historical=True)
    metadata = read(root / "environment.json")
    validate_runtime_metadata(metadata, original_root)
    require(
        all(metadata[key] == manifest["plan"]["model"][key] for key in ("model_id", "revision")),
        "Model checkpoint differs",
    )
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            **dict(
                pretrained_model_name_or_path=metadata["model_id"],
                revision=metadata["revision"],
                local_files_only=True,
                trust_remote_code=False,
            )
        )
    require(
        tokenizer_preflight(manifest["plan"], tokenizer, modules["voluntary_resolution"])
        == manifest["preflight"],
        "Postrun tokenizer differs",
    )
    require(
        metadata["pad_token_id"] == tokenizer.pad_token_id
        and metadata["template_sha256"] == digest(tokenizer.chat_template),
        "Runtime tokenizer metadata differs",
    )
    calls = manifest["plan"]["engineering"]["call_plan"]
    coverage = operational_coverage(root, manifest)
    reports, arrays, receipts, issues = {}, {}, {}, list(coverage["issues"])
    for call in calls:
        index = call["call_index"]
        if not (root / "calls" / f"{index:02d}.json").exists():
            continue
        report, array = load_artifacts(root, "calls", index, np)
        reports[index], arrays[index] = report, array
        require(
            report["eos_token_ids"] == metadata["generation_eos_token_ids"]
            and report["pad_token_id"] == metadata["pad_token_id"],
            "Actual EOS/pad differs",
        )
        if report["rows"] is not None:
            for row in report["rows"]:
                completion = row["completion"]
                generated = completion["generated_token_ids"]
                require(
                    completion["raw"] == tokenizer.decode(generated, skip_special_tokens=True)
                    and completion["ended_with_eos"]
                    is (generated[-1] in metadata["generation_eos_token_ids"]),
                    "Raw generated text/token/EOS differs",
                )
        if call["stage"] == "timing":
            try:
                receipt, receipt_arrays = load_artifacts(root, "receipts", index, np)
                receipts[index] = audit_receipts(
                    receipt, receipt_arrays, call, report, manifest["plan"], modules, tokenizer, np
                )
            except Exception as caught:
                issues.append(dict(call_index=index, receipt_error=str(caught)))
    auditor = load_module(wrapper / NUMERICAL_AUDITOR, "_coverage_numeric_auditor")
    numeric = auditor.audit_collection(
        calls,
        reports,
        arrays,
        {int(k): v for k, v in manifest["plan"]["engineering"]["requests_by_cohort"].items()},
        manifest["plan"]["direction_bindings"]["curiosity_negative"]["unit"],
        hidden_size=metadata["hidden_size"],
        vocabulary_size=metadata["vocabulary_size"],
        pad_token_id=metadata["pad_token_id"],
        eos_ids=metadata["generation_eos_token_ids"],
    )
    summary = (
        read(root / "engineering-summary.json") if (root / "engineering-summary.json").exists() else None
    )
    require(
        summary is None
        or (
            summary["materials_sha256"] == manifest["plan_sha256"]
            and summary["call_plan_sha256"] == manifest["plan"]["engineering"]["call_plan_sha256"]
            and summary["slots"] == coverage["slots"]
        ),
        "Collector summary identity/denominator differs",
    )
    complete = coverage["complete"] and len(receipts) == 6
    passed = bool(
        complete
        and not issues
        and numeric["numeric_engineering_passed"]
        and summary
        and summary["engineering_passed"] is True
    )
    return dict(
        schema_version=1,
        assay="curiosity_coverage_engineering_context_audit",
        manifest_sha256=digest(manifest),
        admission_sha256=digest(admission),
        environment_byte_sha256=sha(root / "environment.json"),
        complete=complete,
        passed=passed,
        source_provenance_verified=True,
        runtime_verified=True,
        selected_receipts_verified=len(receipts) == 6,
        numeric=numeric,
        receipt_checks=receipts,
        operational_coverage=coverage,
        issues=issues,
        scientific_admission=False,
        lifecycle_shutdown_verified=False,
        timing_forecast_verified=False,
        limitations=[
            "Source and receipt checks reuse frozen parser/render helpers; numerical reconstruction is separately implemented.",
            "Online unsaved-row evidence is not a saved full residual replay.",
            "Global admission ledger, cost and stopped-app evidence require the outer local lifecycle check.",
            "Timing/serialization/receipt-slot summary arithmetic and forecast inputs require the separate prospective forecast check.",
        ],
    )


def run(original_root, runtime, wrapper, root):
    root = Path(root)
    manifest, admission = read(root / "manifest.json"), read(root / "admission.json")
    validate_admission(manifest, admission, time.time())
    modules = verify_sources(manifest, original_root, runtime, wrapper)
    require(sha(__file__) == manifest["wrapper_files"][Path(__file__).name], "Unbound worker")
    require(
        all(os.environ.get(k) == "1" for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")),
        "Cache-only model loading required",
    )
    write_once(
        root / "worker-started.json",
        dict(manifest_sha256=digest(manifest), started=time.time(), pid=os.getpid()),
    )
    error, result, checkpoint_count = None, None, 0

    def checkpoint():
        nonlocal checkpoint_count
        checkpoint_count += 1
        (root / "checkpoint.request").write_text(str(checkpoint_count))

    try:
        model = modules["batched"].BatchedHFModel(**manifest["plan"]["model"])
        metadata = runtime_metadata(model, original_root)
        write_once(root / "environment.json", metadata)
        require(
            tokenizer_preflight(manifest["plan"], model.tokenizer, modules["voluntary_resolution"])
            == manifest["preflight"],
            "Actual tokenizer differs from frozen original prefixes",
        )
        require(
            model.active_hooks == 0 and not modules["coverage_engineering"].hook_inventory(model),
            "Unexpected startup hooks",
        )
        checkpoint()
        result = modules["coverage_engineering"].execute(
            root, model, manifest["plan"], checkpoint=checkpoint, deadline=admission["deadline"] - 60
        )
    except Exception as caught:
        error = dict(type=type(caught).__name__, message=str(caught))
    coverage = operational_coverage(root, manifest)
    summary = dict(
        manifest_sha256=digest(manifest),
        admission_sha256=digest(admission),
        assigned_slots=512,
        complete=coverage["complete"] and bool(result and result["complete"]),
        engineering_passed=bool(result and result["engineering_passed"] and error is None),
        operational_coverage=coverage,
        collector_summary_sha256=None if result is None else digest(result),
        error=error,
        finished=time.time(),
        scientific_admission=False,
        causal_admission=False,
        independent_audit_required=True,
        checkpoint_count=checkpoint_count,
    )
    write_once(root / "summary.json", summary)
    checkpoint()
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.audit:
        result = audit_context(args.run, args.original_root, args.runtime, args.wrapper)
    else:
        result = run(args.original_root, args.runtime, args.wrapper, args.run)
    print(
        json.dumps(
            result
            if args.audit
            else {
                k: result[k]
                for k in ("complete", "engineering_passed", "assigned_slots", "scientific_admission")
            },
            indent=2,
        )
    )
    raise SystemExit(0 if result.get("passed", result.get("complete", False)) else 1)
