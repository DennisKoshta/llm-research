"""Independent raw scoring and bounded filesystem evidence; no inference/cloud."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import audit_epistemic_readout_qualification as audit  # noqa: E402
import epistemic_readout_qualification_worker as worker  # noqa: E402
import prepare_epistemic_readout_qualification as materials  # noqa: E402


@pytest.fixture(scope="module")
def plan():
    return materials.build_plan(ROOT)


def inference(answer, *, tokens=None, eos=True):
    return dict(raw="<think>synthetic reasoning</think>" + answer,
                generated_token_ids=[1, 99] if tokens is None else tokens, ended_with_eos=eos)


def perfect(a):
    oracle = audit.displayed_oracle(a)
    answer = oracle["correct_label"] if a["readout"] == "report_information" else json.dumps(
        oracle["remaining_candidates"] if a["readout"] == "remaining_set" else oracle["probabilities"])
    return inference(answer)


def test_independent_oracle_reconstructs_all_actual_fixed_materials(plan):
    for a in plan["assignments"]:
        assert audit.displayed_oracle(a) == a["expected"]
    result = audit.qualify(plan, {a["trial_id"]: perfect(a) for a in plan["assignments"]})
    assert result["passed"] and result["observed_rows"] == 48
    assert result["counts"] == {key: 16 for key in audit.READOUTS}
    assert len(result["probability_ordering_pairs"]) == 8
    assert not result["internal_confidence_validated"] and not result["causal_admission"]


def test_rehashed_oracle_cannot_override_displayed_table(plan):
    changed = deepcopy(plan)
    changed["assignments"][0]["expected"]["probabilities"] = dict(W=1., X=0., Y=0., Z=0.)
    if changed["assignments"][0]["expected"] == plan["assignments"][0]["expected"]:
        changed["assignments"][0]["expected"]["probabilities"] = dict(W=0., X=1., Y=0., Z=0.)
    with pytest.raises(ValueError, match="oracle disagrees"):
        audit.qualify(changed, {})


@pytest.mark.parametrize("answer", [
    '{"W":0.5,"W":0.5,"X":0,"Y":0,"Z":0}',
    '{"W":true,"X":0,"Y":0,"Z":0}',
    '{"W":NaN,"X":0,"Y":0,"Z":0}',
    '{"W":1e999,"X":0,"Y":0,"Z":0}',
    '{"W":1,"X":0,"Y":0}',
    '```json\n{"W":1,"X":0,"Y":0,"Z":0}\n```',
    '{"W":0.4,"X":0.4,"Y":0,"Z":0}',
])
def test_malformed_probability_outputs_never_normalized(plan, answer):
    a = next(a for a in plan["assignments"] if a["readout"] == "candidate_distribution")
    assert not audit.score(a, inference(answer), audit.displayed_oracle(a))["valid"]


@pytest.mark.parametrize("readout,answer", [("remaining_set", '["W","W"]'),
    ("remaining_set", '["W",true]'), ("report_information", 'A.'),
    ("report_information", '\\boxed{A}'), ("report_information", 'The answer is A')])
def test_no_grammar_relaxation(plan, readout, answer):
    a = next(a for a in plan["assignments"] if a["readout"] == readout)
    assert not audit.score(a, inference(answer), audit.displayed_oracle(a))["valid"]


def test_decimal_boundary_and_order_pair_gate(plan):
    observed = {a["trial_id"]: perfect(a) for a in plan["assignments"]}
    selected = [a for a in plan["assignments"] if a["readout"] == "candidate_distribution"
                and a["delivery"] == "redundant_evidence" and a["world_slot"] == 0]
    for a in selected:
        labels = audit.displayed_oracle(a)["remaining_candidates"]
        probs = {key: 0 for key in audit.LABELS}
        probs[labels[0]], probs[labels[1]] = (.55, .45) if a["answer_ordering"] == 0 else (.45, .55)
        observed[a["trial_id"]] = inference(json.dumps(probs))
    result = audit.qualify(plan, observed)
    assert result["checks"]["all16_distributions_within_bounds"]
    assert not result["checks"]["all8_ordering_pairs_within_bounds"]
    resolved = next(a for a in plan["assignments"] if a["readout"] == "candidate_distribution" and a["delivery"] == "decisive_evidence")
    oracle = audit.displayed_oracle(resolved)
    correct = oracle["remaining_candidates"][0]
    wrong = next(key for key in audit.LABELS if key != correct)
    probs = dict.fromkeys(audit.LABELS, 0)
    probs[correct], probs[wrong] = .95, .05
    score = audit.score(resolved, inference(json.dumps(probs)), oracle)
    assert score["distribution_passed"] and score["total_variation"] == .05
    probs[correct], probs[wrong] = .949999999, .050000001
    assert not audit.score(resolved, inference(json.dumps(probs)), oracle)["distribution_passed"]


def test_missing_premature_unclosed_and_cap_remain_assigned(plan):
    a = plan["assignments"][0]
    oracle = audit.displayed_oracle(a)
    assert audit.score(a, inference("A", tokens=[1], eos=False), oracle)["error"] == "unavailable_generation_evidence"
    assert audit.score(a, dict(raw="not closed", generated_token_ids=[99], ended_with_eos=True), oracle)["error"] == "unclosed_thinking"
    complete = perfect(a)
    complete.update(generated_token_ids=[1] * 4096, ended_with_eos=False)
    assert audit.score(a, complete, oracle)["cap_hit"]
    complete["ended_with_eos"] = True
    assert not audit.score(a, complete, oracle)["cap_hit"]
    result = audit.qualify(plan, {})
    assert len(result["scores"]) == 48 and result["observed_rows"] == 0 and not result["passed"]


def startup_fixture():
    def row(ids):
        return dict(generated_token_ids=[7], label_probabilities_temperature_1={"A": .5},
            capture={}, hooks_remaining=0, input_ids=ids, unpadded_input_ids=ids,
            attention_mask=[1] * len(ids), position_ids=list(range(len(ids))))
    baseline = row([1, 2])
    sham, changed = deepcopy(baseline), deepcopy(baseline)
    sham["capture"] = dict(before=[1.], after=[1.], calls=1)
    changed["capture"] = dict(before=[1.], after=[2.], calls=1)
    single = dict(baseline=baseline, sham=sham, nonzero=changed, fresh=deepcopy(baseline), after_error=deepcopy(baseline))
    batch = dict(baseline=[deepcopy(baseline), row([1])], sham=[deepcopy(sham), row([1])],
                 nonzero=[deepcopy(changed), row([1])], fresh=[deepcopy(baseline), row([1])])
    for key in ("sham", "nonzero"):
        batch[key][1]["capture"] = dict(before=[1.], after=[1.], calls=1)
    single_keys = ("sham_tokens_equal", "sham_probabilities_equal", "zero_dose_capture_equal", "hook_called_once",
                   "nonzero_changes_activation", "subsequent_branch_equal", "exception_cleanup_equal", "no_hooks_remaining", "identical_prompt_tokens")
    batch_keys = ("sham_tokens_equal", "sham_probabilities_equal", "zero_dose_capture_equal", "one_hook_call_per_row",
        "target_activation_changed", "other_row_unchanged", "fresh_after_exception_equal", "no_hooks_remaining",
        "unequal_lengths_exercised", "left_padding_correct")
    return dict(single=dict(records=single, passed=True, checks=dict.fromkeys(single_keys, True)),
                batch=dict(records=batch, passed=True, checks=dict.fromkeys(batch_keys, True)),
                passed=True, hook_inventory_after=deepcopy(audit.ZERO_HOOKS))


def test_startup_recomputed_from_records_not_flags():
    value = startup_fixture()
    assert audit.startup_checks(value)["passed"]
    value["batch"]["records"]["nonzero"][1]["generated_token_ids"] = [8]
    with pytest.raises(ValueError, match="startup checks disagree"):
        audit.startup_checks(value)
    value["batch"]["checks"]["other_row_unchanged"] = False
    value["batch"]["passed"] = value["passed"] = False
    assert not audit.startup_checks(value)["passed"]


class Tokenizer:
    pad_token_id = 0

    def __init__(self, outputs):
        self.outputs = outputs

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(m["content"] for m in messages)

    def decode(self, tokens, **kwargs):
        return self.outputs[tokens[0]]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def collection(tmp_path, plan):
    manifest = dict(name=worker.NAME, plan=deepcopy(plan), preflight=dict(requests=[]))
    outputs = {i + 100: perfect(a)["raw"] for i, a in enumerate(plan["assignments"])}
    tokenizer = Tokenizer(outputs)
    for i, a in enumerate(plan["assignments"]):
        manifest["preflight"]["requests"].append(dict(trial_id=a["trial_id"], input_ids=[1] * (3 + i % 2)))
    admission = dict(created=0, deadline=1800)
    for cohort in plan["execution_cohorts"]:
        offset, size = cohort["offset"], len(cohort["trial_ids"])
        selected = plan["assignments"][offset:offset + size]
        records = []
        for row, a in enumerate(selected):
            tokens = manifest["preflight"]["requests"][offset + row]["input_ids"]
            padding = 4 - len(tokens)
            rendered = tokenizer.apply_chat_template(a["request"]["messages"])
            raw = dict(raw=outputs[offset + row + 100], generated_token_ids=[offset + row + 100, 99], ended_with_eos=True,
                messages=a["request"]["messages"], unpadded_input_ids=tokens, rendered_prompt=rendered,
                prompt_sha256=audit.digest(rendered), enable_thinking=True, hooks_remaining=0, intervention=None, capture={},
                sampling=dict(seed=a["request"]["seed"], temperature=.7, top_p=1., max_new_tokens=4096),
                batch=dict(backend="torch_batched_v1", cohort_ids=cohort["trial_ids"], row=row, size=size),
                input_ids=[0] * padding + tokens, attention_mask=[0] * padding + [1] * len(tokens),
                position_ids=[0] * padding + list(range(len(tokens))))
            records.append(dict(trial_id=a["trial_id"], assignment_sha256=audit.digest(a), inference=raw))
        write(tmp_path / "attempts" / f"{offset:04d}.json", dict(manifest_sha256=audit.digest(manifest), cohort=cohort, started=10 + offset))
        write(tmp_path / "cohorts" / f"{offset:04d}.json", dict(manifest_sha256=audit.digest(manifest), cohort=cohort,
            status="completed", hook_inventory_before=audit.ZERO_HOOKS, hook_inventory_after=audit.ZERO_HOOKS, records=records))
    return tmp_path, manifest, admission, tokenizer


def collected(collection):
    root, manifest, admission, tokenizer = collection
    return audit.collect_rows(root, manifest, admission, worker, tokenizer, [99])


def test_full_raw_collection_matches_independent_summary(collection):
    rows, observed, bindings = collected(collection)
    _, manifest, _, _ = collection
    independent = audit.qualify(manifest["plan"], observed)
    production = materials.qualify(manifest["plan"], observed)
    assert production == independent and len(bindings) == 2 and len(rows) == 48
    summary = dict(manifest_sha256=audit.digest(manifest), assigned_rows=48, causal_admission=False,
        dispositions=rows, complete=True, integrity_passed=True, qualification=production, qualification_passed=True)
    audit.verify_summary(summary, manifest, rows, independent)
    summary["qualification"]["scores"][0]["valid"] = False
    with pytest.raises(ValueError, match="raw response score differs"):
        audit.verify_summary(summary, manifest, rows, independent)


def test_atomic_failed_second_cohort_retains16_and_all_denominators(collection):
    root, manifest, _, _ = collection
    path = root / "cohorts/0032.json"
    value = audit.read(path)
    value.update(status="execution_error", error=dict(type="RuntimeError", message="synthetic"), returned_inferences=value.pop("records"))
    write(path, value)
    rows, observed, _ = collected(collection)
    assert len(rows) == 48 and len(observed) == 32
    assert sum(r["status"] == "execution_error" for r in rows) == 16
    assert not audit.qualify(manifest["plan"], observed)["passed"]


def test_missing_first_cohort_cannot_be_followed_by_second(collection):
    root, _, _, _ = collection
    (root / "cohorts/0000.json").unlink()
    with pytest.raises(ValueError, match="followed by another attempt"):
        collected(collection)


@pytest.mark.parametrize("change", ["tokens", "seed", "cohort", "hooks", "eos_interior", "late_attempt", "retry"])
def test_raw_bindings_cannot_be_overridden_by_worker_passed_flags(collection, change):
    root, _, _, _ = collection
    path = root / "cohorts/0000.json"
    value = audit.read(path)
    row = value["records"][0]["inference"]
    if change == "tokens":
        row["unpadded_input_ids"] = [999]
    elif change == "seed":
        row["sampling"]["seed"] += 1
    elif change == "cohort":
        row["batch"]["row"] = 1
    elif change == "hooks":
        value["hook_inventory_after"]["active_hooks"] = 1
    elif change == "eos_interior":
        row["generated_token_ids"] = [100, 99, 99]
    elif change == "late_attempt":
        attempt_path = root / "attempts/0000.json"
        attempt = audit.read(attempt_path)
        attempt["started"] = 1740
        write(attempt_path, attempt)
    else:
        write(root / "attempts/0001.json", {})
    write(path, value)
    with pytest.raises(ValueError):
        collected(collection)


def runtime_metadata(manifest):
    return dict(backend="transformers", dtype="bfloat16", attention="eager", device="cuda", layers=64,
        execution_backend="torch_batched_v1", logits_to_keep=1, loading_policy="direct_device_sharded",
        default_enable_thinking=False, cuda="12.4", template_sha256=manifest["preflight"]["tokenizer_template_sha256"],
        versions=dict(torch="2.6.0+cu124", transformers="4.51.3", accelerate="1.6.0", tokenizers="0.21.4"),
        **manifest["plan"]["model"], cuda_numerics=dict(policy="bf16_reduced_precision_reduction_disabled_v1",
            allow_bf16_reduced_precision_reduction=False, allow_fp16_reduced_precision_reduction=True, allow_tf32=False),
        python="3.12.10", gpu="NVIDIA H200", generation_eos_token_ids=[99])


@pytest.mark.parametrize("field,value", [("gpu", "NVIDIA H100"), ("python", "3.11.9"), ("cuda", "12.8"),
    ("generation_eos_token_ids", [98]), ("dtype", "float32"), ("attention", "sdpa")])
def test_runtime_policy_is_not_self_reported_pass(plan, field, value):
    manifest = dict(plan=plan, preflight=dict(tokenizer_template_sha256="synthetic"))
    metadata = runtime_metadata(manifest)
    original = ROOT / "executions/qwen32-voluntary-resolution-v2"
    audit.runtime(metadata, manifest, original, [99])
    metadata[field] = value
    with pytest.raises(ValueError, match="Runtime/model/EOS"):
        audit.runtime(metadata, manifest, original, [99])


def test_complete_filesystem_audit_and_honest_qualification_failure(collection, monkeypatch):
    import shutil
    temporary, manifest, admission, tokenizer = collection
    project = temporary / "project"
    run = project / "runs" / worker.NAME
    run.mkdir(parents=True)
    for directory in ("cohorts", "attempts"):
        shutil.move(temporary / directory, run / directory)
    manifest.update(source_sha256="synthetic-source", wrapper_files={"auditor": "synthetic"})
    manifest["preflight"]["tokenizer_template_sha256"] = "synthetic-template"
    admission.update(manifest_sha256=audit.digest(manifest), phase_consumed=True, retry_allowed=False,
                     maximum_liability_usd=worker.resources()["maximum_liability_usd"])
    for directory in ("cohorts", "attempts"):
        for path in (run / directory).glob("*.json"):
            value = audit.read(path)
            value["manifest_sha256"] = audit.digest(manifest)
            write(path, value)
    # One malformed probability response: operationally complete, scientifically failed.
    cohort_path = run / "cohorts/0000.json"
    envelope = audit.read(cohort_path)
    index = next(i for i, a in enumerate(manifest["plan"]["assignments"][:32]) if a["readout"] == "candidate_distribution")
    bad = envelope["records"][index]["inference"]
    bad["raw"] = '<think>done</think>{"W":true,"X":0,"Y":0,"Z":0}'
    tokenizer.outputs[bad["generated_token_ids"][0]] = bad["raw"]
    write(cohort_path, envelope)
    p = dict(manifest=manifest, admission=admission, worker=worker, tokenizer=tokenizer, eos_ids=[99],
             original=ROOT / "executions/qwen32-voluntary-resolution-v2")
    monkeypatch.setattr(audit, "source_provenance", lambda _: p)
    rows, observed, _ = audit.collect_rows(run, manifest, admission, worker, tokenizer, [99])
    production = materials.qualify(manifest["plan"], observed)
    write(run / "summary.json", dict(manifest_sha256=audit.digest(manifest), assigned_rows=48, causal_admission=False,
        dispositions=rows, complete=True, integrity_passed=True, qualification=production, qualification_passed=False))
    meta = runtime_metadata(manifest)
    write(run / "environment.json", meta)
    startup = startup_fixture()
    for kind in ("single", "batch"):
        startup[kind]["model"] = meta
    write(run / "startup.json", startup)
    write(run / "worker-started.json", dict(manifest_sha256=audit.digest(manifest), time=2, pid=1))
    write(run / "launch.json", dict(app_id="synthetic", admission_sha256=audit.digest(admission), image_id="image", started=1))
    write(run / "cloud-started.json", dict(started=1, image_id="image"))
    write(run / "cloud-finished.json", dict(finished=100, error=None))
    coverage_rows = []
    for cohort in manifest["plan"]["execution_cohorts"]:
        for record in audit.read(run / "cohorts" / f"{cohort['offset']:04d}.json")["records"]:
            coverage_rows.append(dict(trial_id=record["trial_id"], status="completed", record_sha256=audit.digest(record)))
    coverage = dict(manifest_sha256=audit.digest(manifest), assigned_rows=48, rows=coverage_rows, issues=[],
                    complete=True, scientific_qualification=False, causal_admission=False)
    write(run / "collection-coverage.json", coverage)
    write(run / "launch-outcome.json", dict(app_id="synthetic", causal_admission=False, all_stopped_zero_tasks=True,
        collection_coverage_sha256=audit.digest(coverage), failure=None, export_error=None,
        result=dict(complete=True, error=None, qualification_passed=False, causal_admission=False)))
    write(project / "reports" / (worker.NAME + "-cost-and-shutdown.json"), dict(all_stopped_zero_tasks=True,
        apps=[{"App ID": "synthetic", "Description": worker.APP, "State": "stopped", "Tasks": "0"}]))
    result = audit.audit(run)
    assert result["complete"] and result["integrity_passed"] and result["computation_agrees"]
    assert not result["qualification_passed"] and not result["passed"]
    assert result["assigned_rows"] == result["observed_rows"] == 48
    assert sum(not s["valid"] for s in result["qualification"]["scores"]) == 1
    coverage["rows"][0]["record_sha256"] = "wrong"
    write(run / "collection-coverage.json", coverage)
    outcome = audit.read(run / "launch-outcome.json")
    outcome["collection_coverage_sha256"] = audit.digest(coverage)
    write(run / "launch-outcome.json", outcome)
    with pytest.raises(ValueError, match="Operational coverage binding"):
        audit.audit(run)
