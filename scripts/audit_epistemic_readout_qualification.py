"""Independent displayed-table oracle and strict-output audit for the fixed 48 probes.

Shared frozen helpers check source/token/layout engineering only. Neither the worker's
score_output/qualify nor the task's oracle is used to calculate a response score.
No model weights, inference, cloud requests, normalization or scoring rescue.
"""

import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import tarfile
import tomllib

LABELS = tuple("WXYZ")
READOUTS = ("remaining_set", "candidate_distribution", "report_information")
DELIVERIES = ("decisive_evidence", "redundant_evidence")
ZERO_HOOKS = dict(active_hooks=0, forward_hooks=0, forward_pre_hooks=0)


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def displayed_oracle(a):
    """Read actual displayed candidates, both observed facts and the offered property."""
    text = a["request"]["messages"][-1]["content"]
    require(text.count("End of record.") == 1, "Ambiguous evidence boundary")
    context, question = text.split("End of record.")
    require(context + "End of record." == a["common_context"]
            and "Every row was equally possible initially." in context
            and "All property observations and reports are accurate, with no measurement noise." in context,
            "Equal-prior/noiseless complete record differs")
    table_lines = [line.strip() for line in context.splitlines() if "|" in line]
    require(len(table_lines) == 5, "Expected one displayed four-row table")
    header = [s.strip() for s in table_lines[0].split("|")]
    require(len(header) == 3 and header[0] == "Candidate" and len(set(header[1:])) == 2,
            "Candidate table header differs")
    table = {}
    for line in table_lines[1:]:
        values = [s.strip() for s in line.split("|")]
        require(len(values) == 3 and values[0] in LABELS and values[0] not in table, "Candidate row differs")
        table[values[0]] = dict(zip(header[1:], values[1:]))
    require(set(table) == set(LABELS) and len({tuple(row.values()) for row in table.values()}) == 4,
            "Displayed candidates are not four distinct worlds")
    facts = []
    for label in ("Initial observation", "Additional delivery"):
        match = re.search(re.escape(label) + r":\nThe ([^\n]+) is ([^\n]+)\.\n", context)
        require(match is not None and match[1] in header[1:], "Missing/ambiguous factual observation")
        facts.append((match[1], match[2]))
    support = [label for label in LABELS if all(table[label][key] == value for key, value in facts)]
    require(len(support) == (1 if a["delivery"] == "decisive_evidence" else 2), "Displayed evidence state differs")
    posterior = {label: 1 / len(support) if label in support else 0.0 for label in LABELS}
    # The offered report is the other property relative to the original observation.
    report_property = next(key for key in header[1:] if key != facts[0][0])
    partition = {}
    for label in support:
        partition.setdefault(table[label][report_property], []).append(label)
    gain = math.log2(len(support)) - sum(len(group) / len(support) * math.log2(len(group))
                                        for group in partition.values())
    require(gain in (0.0, 1.0), "Report partition does not match the declared finite task")
    information = "new" if gain == 1 else "repeated"
    correct = None
    if a["readout"] == "report_information":
        offered = re.search(r"Now suppose you request the hidden object's (.+?) report and receive it\.", question)
        require(offered is not None and offered[1] == report_property, "Question offers a different report")
        options = dict(re.findall(r"^([ABCD]): (.+)$", question, re.MULTILINE))
        semantics = {
            "new": "A property value not already established by the complete record; it would distinguish candidates still possible now.",
            "repeated": "A property value already established by the complete record; it would not distinguish any candidates still possible now.",
            "pattern": "Only an independent decorative pattern, with no property report.",
            "none": "Neither a property report nor a decorative pattern.",
        }
        require(set(options) == set("ABCD") and set(options.values()) == set(semantics.values()), "Displayed report options differ")
        correct = next(label for label, value in options.items() if value == semantics[information])
    return dict(remaining_candidates=support, probabilities=posterior, report_information=information,
                expected_information_gain_bits=gain, report_value_partition=partition, correct_label=correct)


def object_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON object key")
        result[key] = value
    return result


def strict_json(text):
    def nonfinite(_):
        raise ValueError("Nonfinite JSON number")
    return json.loads(text, object_pairs_hook=object_pairs, parse_constant=nonfinite)


def score(a, inference, oracle):
    result = {key: a[key] for key in ("trial_id", "world_slot", "semantic_world_id", "delivery", "readout", "answer_ordering")}
    result.update(assigned=True, observed=inference is not None, closed=False, cap_hit=False, valid=False,
                  correct=None, parsed=None, total_variation=None, incompatible_mass=None,
                  max_probability=None, distribution_passed=False, error=None)
    if inference is None:
        result["error"] = "missing_output"
        return result
    raw, tokens, eos = inference.get("raw"), inference.get("generated_token_ids"), inference.get("ended_with_eos")
    if not (isinstance(raw, str) and isinstance(tokens, list) and 0 < len(tokens) <= 4096
            and all(type(t) is int and t >= 0 for t in tokens) and type(eos) is bool and (eos or len(tokens) == 4096)):
        result["error"] = "unavailable_generation_evidence"
        return result
    result.update(closed="</think>" in raw, cap_hit=not eos and len(tokens) == 4096)
    if not result["closed"]:
        result["error"] = "unclosed_thinking"
        return result
    answer = raw.rsplit("</think>", 1)[1].strip()
    try:
        if a["readout"] == "report_information":
            require(re.fullmatch("[ABCD]", answer) is not None, "Expected exactly one letter")
            result.update(valid=True, parsed=answer, correct=answer == oracle["correct_label"])
        elif a["readout"] == "remaining_set":
            parsed = strict_json(answer)
            require(type(parsed) is list and all(type(v) is str and v in LABELS for v in parsed)
                    and len(set(parsed)) == len(parsed), "Expected a unique candidate-label array")
            result.update(valid=True, parsed=parsed, correct=set(parsed) == set(oracle["remaining_candidates"]))
        else:
            parsed = strict_json(answer)
            require(type(parsed) is dict and set(parsed) == set(LABELS)
                    and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in parsed.values()),
                    "Invalid complete probability allocation")
            probabilities = {key: Decimal(str(value)) for key, value in parsed.items()}
            require(abs(sum(probabilities.values()) - 1) <= Decimal("0.000001"), "Probability allocation does not sum to one")
            tv = sum(abs(probabilities[k] - Decimal(str(oracle["probabilities"][k]))) for k in LABELS) / 2
            incompatible = sum(probabilities[k] for k in LABELS if k not in oracle["remaining_candidates"])
            maximum = max(probabilities.values())
            state = maximum >= Decimal(".95") if len(oracle["remaining_candidates"]) == 1 else Decimal(".45") <= maximum <= Decimal(".55")
            result.update(valid=True, parsed=parsed, total_variation=float(tv), incompatible_mass=float(incompatible),
                          max_probability=float(maximum), distribution_passed=tv <= Decimal(".05") and incompatible <= Decimal(".05") and state)
    except (ValueError, TypeError, OverflowError) as error:
        result["error"] = str(error)
    return result


def qualify(plan, inferences):
    assignments = plan["assignments"]
    require(len(assignments) == 48 and len({a["trial_id"] for a in assignments}) == 48
            and set(inferences) <= {a["trial_id"] for a in assignments}, "Assigned/observed inventory differs")
    required = {(slot, delivery, readout, ordering) for slot in range(4) for delivery in DELIVERIES
                for readout in READOUTS for ordering in (0, 1)}
    require({(a["world_slot"], a["delivery"], a["readout"], a["answer_ordering"]) for a in assignments} == required,
            "Incomplete qualification factorial")
    scores, pairs = [], []
    for a in assignments:
        oracle = displayed_oracle(a)
        require(oracle == a["expected"], "Frozen oracle disagrees with displayed evidence")
        scores.append(score(a, inferences.get(a["trial_id"]), oracle))
    for slot in range(4):
        for delivery in sorted(DELIVERIES):
            pair = sorted((s for s in scores if s["world_slot"] == slot and s["delivery"] == delivery
                           and s["readout"] == "candidate_distribution"), key=lambda s: s["answer_ordering"])
            available = all(s["valid"] for s in pair)
            tv = sum(abs(Decimal(str(pair[0]["parsed"][k])) - Decimal(str(pair[1]["parsed"][k]))) for k in LABELS) / 2 if available else None
            pairs.append(dict(world_slot=slot, delivery=delivery, total_variation=None if tv is None else float(tv),
                              passed=available and tv <= Decimal(".05")))
    checks = dict(all48_closed_valid_uncapped=all(s["closed"] and s["valid"] and not s["cap_hit"] for s in scores),
        all16_sets_exact=sum(s["readout"] == "remaining_set" and s["correct"] is True for s in scores) == 16,
        all16_report_judgments_exact=sum(s["readout"] == "report_information" and s["correct"] is True for s in scores) == 16,
        all16_distributions_within_bounds=sum(s["readout"] == "candidate_distribution" and s["distribution_passed"] for s in scores) == 16,
        all8_ordering_pairs_within_bounds=all(p["passed"] for p in pairs))
    return dict(assignment_sha256=plan["assignment_sha256"], assigned_rows=48, distinct_worlds=4, distinct_contexts=8,
        observed_rows=len(inferences), counts=dict(Counter(s["readout"] for s in scores)), passed=all(checks.values()),
        checks=checks, scores=scores, probability_ordering_pairs=pairs, internal_confidence_validated=False,
        causal_admission=False, missing_outputs_retained=True)


def source_provenance(run):
    run = Path(run).resolve()
    project = run.parent.parent
    manifest, freeze = read(run / "manifest.json"), read(run / "freeze.json")
    execution = project / "executions" / manifest["name"]
    original, wrapper = execution / "original", execution / "qualification"
    require(freeze == dict(manifest_sha256=digest(manifest), execution_root=str(execution),
            original_source_archive_sha256=manifest["original_source_archive_sha256"],
            wrapper_files_sha256=digest(manifest["wrapper_files"])), "Freeze identity differs")
    require(sha(__file__) == manifest["wrapper_files"][Path(__file__).name], "Use the frozen auditor bytes")
    worker_path = wrapper / "epistemic_readout_qualification_worker.py"
    require(sha(worker_path) == manifest["wrapper_files"][worker_path.name], "Worker bytes differ")
    worker = load(worker_path, "_qualification_audit_worker")
    worker.check_files(original, manifest["original_execution_files"])
    worker.check_files(wrapper, manifest["wrapper_files"])
    worker.validate_manifest(manifest)
    source = original / "src/curiosity_study"
    require(digest({str(p.relative_to(source)): p.read_text() for p in sorted(source.rglob("*.py"))}) == manifest["source_sha256"],
            "Original source digest differs")
    archive_path = project / "runs" / worker.SOURCE_RUN / "source.tar.gz"
    require(sha(archive_path) == manifest["original_source_archive_sha256"], "Original source archive differs")
    inventory = {}
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            relative = Path(member.name)
            require(member.isfile() and not relative.is_absolute() and ".." not in relative.parts
                    and str(relative) == member.name and member.name not in inventory, "Unsafe archive member")
            inventory[member.name] = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    require(inventory == manifest["original_execution_files"], "Extracted archive inventory differs")
    materials = worker.load_materials(wrapper, manifest)
    require(digest(materials.build_plan(project)) == manifest["plan_sha256"], "Frozen material reconstruction differs")
    from transformers import AutoTokenizer, GenerationConfig
    tokenizer = AutoTokenizer.from_pretrained(
        manifest["plan"]["model"]["model_id"], revision=manifest["plan"]["model"]["revision"], local_files_only=True, trust_remote_code=False)
    require(materials.tokenize_plan(manifest["plan"], tokenizer) == manifest["preflight"], "Pinned tokenizer preflight differs")
    config = GenerationConfig.from_pretrained(manifest["plan"]["model"]["model_id"],
        revision=manifest["plan"]["model"]["revision"], local_files_only=True)
    eos = config.eos_token_id
    eos = sorted(set([eos] if type(eos) is int else eos))
    admission = read(run / "admission.json")
    global_admission = project / "campaign_admissions" / manifest["name"] / "admission.json"
    require(global_admission.read_bytes() == (run / "admission.json").read_bytes(), "Global/local admission mirror differs")
    worker.validate_admission(manifest, admission, admission["created"])
    require(manifest["assay"] == "epistemic_readout_qualification" and manifest["plan"]["wording_retry_allowed"] is False,
            "Readout-only scope differs")
    return dict(manifest=manifest, freeze=freeze, admission=admission, worker=worker, original=original,
                tokenizer=tokenizer, eos_ids=eos, execution=execution)


def runtime(metadata, manifest, original, eos_ids):
    packages = tomllib.loads((Path(original) / "uv.lock").read_text())["package"]
    versions = {p["name"]: p["version"] for p in packages if p["name"] in ("torch", "transformers", "accelerate", "tokenizers")}
    expected = dict(backend="transformers", dtype="bfloat16", attention="eager", device="cuda", layers=64,
        execution_backend="torch_batched_v1", logits_to_keep=1, loading_policy="direct_device_sharded",
        default_enable_thinking=False, cuda="12.4", template_sha256=manifest["preflight"]["tokenizer_template_sha256"],
        versions=versions, **manifest["plan"]["model"], cuda_numerics=dict(policy="bf16_reduced_precision_reduction_disabled_v1",
        allow_bf16_reduced_precision_reduction=False, allow_fp16_reduced_precision_reduction=True, allow_tf32=False))
    require(all(metadata.get(k) == v for k, v in expected.items())
            and str(metadata.get("python", "")).split(".")[:2] == ["3", "12"]
            and "H200" in str(metadata.get("gpu", ""))
            and metadata.get("generation_eos_token_ids") == eos_ids, "Runtime/model/EOS policy differs")
    return expected


def startup_checks(startup):
    """Recompute frozen startup predicates from its saved records, not passed flags."""
    b, s, n, f, e = (startup["single"]["records"][key] for key in ("baseline", "sham", "nonzero", "fresh", "after_error"))
    prob = "label_probabilities_temperature_1"
    single = dict(sham_tokens_equal=b["generated_token_ids"] == s["generated_token_ids"],
        sham_probabilities_equal=b[prob] == s[prob], zero_dose_capture_equal=s["capture"]["before"] == s["capture"]["after"],
        hook_called_once=s["capture"]["calls"] == n["capture"]["calls"] == 1,
        nonzero_changes_activation=n["capture"]["before"] != n["capture"]["after"],
        subsequent_branch_equal=b[prob] == f[prob], exception_cleanup_equal=b[prob] == e[prob],
        no_hooks_remaining=all(r["hooks_remaining"] == 0 for r in (b, s, n, f, e)) and startup["hook_inventory_after"] == ZERO_HOOKS,
        identical_prompt_tokens=b["input_ids"] == s["input_ids"] == n["input_ids"])
    b, s, n, f = (startup["batch"]["records"][key] for key in ("baseline", "sham", "nonzero", "fresh"))
    require(all(len(rows) == 2 for rows in (b, s, n, f)), "Startup two-row inventory differs")
    batch = dict(sham_tokens_equal=all(x["generated_token_ids"] == y["generated_token_ids"] for x, y in zip(b, s)),
        sham_probabilities_equal=all(x[prob] == y[prob] for x, y in zip(b, s)),
        zero_dose_capture_equal=all(x["capture"]["before"] == x["capture"]["after"] for x in s),
        one_hook_call_per_row=all(x["capture"]["calls"] == 1 for x in s + n),
        target_activation_changed=n[0]["capture"]["before"] != n[0]["capture"]["after"],
        other_row_unchanged=b[1]["generated_token_ids"] == n[1]["generated_token_ids"] and b[1][prob] == n[1][prob],
        fresh_after_exception_equal=all(x["generated_token_ids"] == y["generated_token_ids"] and x[prob] == y[prob] for x, y in zip(b, f)),
        no_hooks_remaining=all(x["hooks_remaining"] == 0 for rows in (b, s, n, f) for x in rows) and startup["hook_inventory_after"] == ZERO_HOOKS,
        unequal_lengths_exercised=len(b[0]["unpadded_input_ids"]) != len(b[1]["unpadded_input_ids"]),
        left_padding_correct=all([t for t, mask in zip(x["input_ids"], x["attention_mask"]) if mask] == x["unpadded_input_ids"]
            and [p for p, mask in zip(x["position_ids"], x["attention_mask"]) if mask] == list(range(len(x["unpadded_input_ids"]))) for x in b))
    for name, checks in (("single", single), ("batch", batch)):
        require(startup[name]["checks"] == checks and startup[name]["passed"] is all(checks.values()), "Saved startup checks disagree: " + name)
    passed = all(single.values()) and all(batch.values())
    require(startup["passed"] is passed, "Saved startup passed flag disagrees")
    return dict(single=single, batch=batch, passed=passed)


def collect_rows(run, manifest, admission, worker, tokenizer, eos_ids):
    rows, inferences, bindings = [], {}, []
    expected_files = {f"{c['offset']:04d}.json" for c in manifest["plan"]["execution_cohorts"]}
    for directory in ("cohorts", "attempts"):
        path = run / directory
        require(not path.exists() or (not path.is_symlink() and all(p.is_file() and not p.is_symlink()
                and p.name in expected_files for p in path.iterdir())), "Unexpected/retried cohort artifact")
    stopped, previous_start = False, admission["created"]
    for cohort in manifest["plan"]["execution_cohorts"]:
        offset, size = cohort["offset"], len(cohort["trial_ids"])
        selected = manifest["plan"]["assignments"][offset:offset + size]
        tokens = manifest["preflight"]["requests"][offset:offset + size]
        path, attempt_path = run / "cohorts" / f"{offset:04d}.json", run / "attempts" / f"{offset:04d}.json"
        attempted = attempt_path.exists()
        if attempted:
            require(not stopped, "A failed/missing cohort was retried or followed by another attempt")
            attempt = read(attempt_path)
            require(attempt["manifest_sha256"] == digest(manifest) and attempt["cohort"] == cohort
                    and type(attempt["started"]) in (int, float) and math.isfinite(attempt["started"])
                    and previous_start <= attempt["started"] < admission["deadline"] - 60, "Attempt/time binding differs")
            previous_start = attempt["started"]
        status, error, records = "missing", None, []
        if path.exists():
            require(attempted, "Cohort has no preceding immutable attempt")
            value = read(path)
            require(value["manifest_sha256"] == digest(manifest) and value["cohort"] == cohort, "Cohort binding differs")
            status = value["status"]
            if status == "completed":
                require(value["hook_inventory_before"] == value["hook_inventory_after"] == ZERO_HOOKS,
                        "Readout cohort had hooks")
                records = value["records"]
                require([r["trial_id"] for r in records] == cohort["trial_ids"], "Atomic row coverage/order differs")
                for row, (a, token, record) in enumerate(zip(selected, tokens, records)):
                    require(record["assignment_sha256"] == digest(a), "Stored row assignment differs")
                    inference = record["inference"]
                    worker.validate_inference(a, token, inference, cohort, row, tokenizer, eos_ids,
                                              max(len(t["input_ids"]) for t in tokens))
                    require(not any(t in eos_ids for t in inference["generated_token_ids"][:-1]), "Generated tokens continue after EOS")
                    inferences[a["trial_id"]] = inference
            else:
                require(status == "execution_error" and isinstance(value["error"], dict)
                        and set(value["error"]) == {"type", "message"}, "Unknown terminal disposition")
                error, stopped = value["error"], True
            bindings.append(dict(path=str(path.relative_to(run)), byte_sha256=sha(path), content_sha256=digest(value)))
        else:
            stopped = True
            if attempted:
                status = "execution_error"
        rows.extend(dict(trial_id=a["trial_id"], assignment_sha256=digest(a), status=status, error=error) for a in selected)
    return rows, inferences, bindings


def verify_summary(summary, manifest, rows, independent):
    require(summary["manifest_sha256"] == digest(manifest) and summary["assigned_rows"] == 48
            and summary["causal_admission"] is False, "Summary binding/scope differs")
    expected = [{k: row[k] for k in ("trial_id", "assignment_sha256", "status")} for row in rows]
    require([{k: row[k] for k in ("trial_id", "assignment_sha256", "status")} for row in summary["dispositions"]] == expected,
            "Summary disposition inventory differs")
    for actual, saved in zip(rows, summary["dispositions"]):
        require(actual["error"] == saved.get("error"), "Summary failure disposition differs")
    complete = all(row["status"] == "completed" for row in rows)
    require(summary["complete"] is complete and summary["integrity_passed"] is complete,
            "Summary completion differs from raw cohorts")
    # Error prose is diagnostic only; parsed values, all scores, predicates and denominators must match.
    observed = summary["qualification"]
    for key in independent:
        if key != "scores":
            require(observed[key] == independent[key], "Independent qualification differs: " + key)
    require(len(observed["scores"]) == 48, "Summary scores omit assignments")
    for actual, saved in zip(independent["scores"], observed["scores"]):
        require({k: v for k, v in actual.items() if k != "error"} == {k: v for k, v in saved.items() if k != "error"}
                and (actual["error"] is None) == (saved["error"] is None), "Independent raw response score differs")
    require(summary["qualification_passed"] is independent["passed"], "Summary qualification gate differs")


def audit(run):
    run = Path(run).resolve()
    p = source_provenance(run)
    manifest, admission = p["manifest"], p["admission"]
    rows, inferences, cohorts = collect_rows(run, manifest, admission, p["worker"], p["tokenizer"], p["eos_ids"])
    independent = qualify(manifest["plan"], inferences)
    environment_path, startup_path = run / "environment.json", run / "startup.json"
    runtime_ok, startup_ok = False, False
    policy, startup = None, None
    if environment_path.exists():
        environment = read(environment_path)
        policy = runtime(environment, manifest, p["original"], p["eos_ids"])
        runtime_ok = True
    if startup_path.exists():
        require(runtime_ok, "Startup has no runtime metadata")
        raw_startup = read(startup_path)
        for kind in ("single", "batch"):
            meta = dict(raw_startup[kind]["model"], generation_eos_token_ids=p["eos_ids"])
            runtime(meta, manifest, p["original"], p["eos_ids"])
        startup = startup_checks(raw_startup)
        startup_ok = startup["passed"]
    if inferences:
        require(runtime_ok and startup_ok, "Completed scientific rows lack passed runtime/startup evidence")
    started_path = run / "worker-started.json"
    if started_path.exists():
        started = read(started_path)
        require(started["manifest_sha256"] == digest(manifest), "Worker-start binding differs")
        p["worker"].validate_admission(manifest, admission, started["time"])
    require(not inferences or started_path.exists(), "Completed rows lack worker-start evidence")
    summary_path = run / "summary.json"
    computation_agrees = False
    if summary_path.exists():
        summary = read(summary_path)
        verify_summary(summary, manifest, rows, independent)
        computation_agrees = True
    launch_path, outcome_path = run / "launch.json", run / "launch-outcome.json"
    closed, transport_ok = False, False
    cost_binding = None
    if outcome_path.exists():
        launch, outcome = read(launch_path), read(outcome_path)
        require(launch["admission_sha256"] == digest(admission) and launch["app_id"] == outcome["app_id"]
                and outcome["causal_admission"] is False, "Launcher identity/scope differs")
        cloud = read(run / "cloud-started.json") if (run / "cloud-started.json").exists() else None
        if cloud:
            require(cloud["image_id"] == launch["image_id"] and admission["created"] <= cloud["started"] < admission["deadline"],
                    "Actual image/start binding differs")
        cost_path = run.parent.parent / "reports" / (manifest["name"] + "-cost-and-shutdown.json")
        cost = read(cost_path)
        apps = [a for a in cost.get("apps", []) if a["App ID"] == launch["app_id"]]
        closed = (outcome["all_stopped_zero_tasks"] is True and cost["all_stopped_zero_tasks"] is True
                  and len(apps) == 1 and apps[0]["Description"] == p["worker"].APP
                  and apps[0]["State"] == "stopped" and str(apps[0]["Tasks"]) == "0")
        cost_binding = dict(path=str(cost_path), byte_sha256=sha(cost_path))
        coverage = read(run / "collection-coverage.json")
        expected_coverage = []
        for cohort in manifest["plan"]["execution_cohorts"]:
            cohort_path = run / "cohorts" / f"{cohort['offset']:04d}.json"
            envelope = read(cohort_path) if cohort_path.exists() else {}
            record_hashes = {r["trial_id"]: digest(r) for r in envelope.get("records", [])} if envelope.get("status") == "completed" else {}
            expected_coverage.extend(dict(trial_id=row["trial_id"], status=row["status"],
                record_sha256=record_hashes.get(row["trial_id"])) for row in rows[cohort["offset"]:cohort["offset"] + len(cohort["trial_ids"])])
        require(outcome["collection_coverage_sha256"] == digest(coverage)
                and coverage["manifest_sha256"] == digest(manifest) and coverage["assigned_rows"] == 48
                and coverage["rows"] == expected_coverage and coverage["issues"] == []
                and coverage["complete"] is (len(inferences) == 48)
                and coverage["scientific_qualification"] is False and coverage["causal_admission"] is False,
                "Operational coverage binding differs")
        cloud_finish = read(run / "cloud-finished.json") if (run / "cloud-finished.json").exists() else None
        result = outcome.get("result")
        transport_ok = (cloud is not None and cloud_finish is not None and cloud_finish["error"] is None
                        and outcome["failure"] is None and outcome["export_error"] is None
                        and isinstance(result, dict) and result["complete"] is True and result["error"] is None
                        and result["qualification_passed"] is independent["passed"] and result["causal_admission"] is False)
    complete = len(inferences) == 48 and summary_path.exists()
    integrity = complete and computation_agrees and runtime_ok and startup_ok and closed and transport_ok
    bindings = {str(path.relative_to(run)): sha(path) for path in run.rglob("*.json") if path.name != "audit.json"}
    return dict(schema_version=1, manifest_sha256=digest(manifest), source_sha256=manifest["source_sha256"],
        auditor_byte_sha256=sha(__file__), frozen_wrapper_files_sha256=digest(manifest["wrapper_files"]),
        complete=complete, integrity_passed=integrity, computation_agrees=computation_agrees,
        qualification_passed=integrity and independent["passed"], passed=integrity and independent["passed"],
        assigned_rows=48, observed_rows=len(inferences), dispositions=rows, qualification=independent,
        checks=dict(runtime=runtime_ok, startup=startup_ok, all_apps_stopped=closed, complete_coverage=complete,
                    worker_scoring_reproduced=computation_agrees, transport_complete=transport_ok), cost_binding=cost_binding, runtime_policy=policy, startup_checks=startup,
        cohort_bindings=cohorts, artifact_byte_sha256=bindings, internal_confidence_validated=False,
        causal_admission=False, limitations=["Four previously exposed object worlds; no general equivalence claim.",
            "Long independent reasoning reports are normative task readouts, not contemporaneous internal beliefs.",
            "Tokenizer and frozen engineering validators are shared dependencies; displayed-table oracle and response scoring are independent.",
            "Missing or failed atomic cohorts remain assigned and unscored; agreement on observed outputs cannot certify their scoring."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, sort_keys=True, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("complete", "integrity_passed", "qualification_passed", "observed_rows")}))


if __name__ == "__main__":
    main()
