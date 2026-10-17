"""Independent raw collection checks; synthetic files and actual tiny BF16 only."""
# ruff: noqa: F811 -- pytest fixtures imported from the existing synthetic suites.

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import audit_curiosity_coverage_bridge as audit  # noqa: E402
from curiosity_study import coverage_bridge as producer  # noqa: E402
from curiosity_study import coverage_engineering as engine  # noqa: E402
from test_coverage_bridge import template  # noqa: E402,F401
from test_coverage_engineering_audit import observed_calls  # noqa: E402,F401


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def test_all261_calls_and_original_receipt_groups_reconstructed_independently(template):
    for cohort in template["execution_cohorts"]:
        assert audit.expected_call(template, cohort) == producer.scientific_call(template, cohort)
    counts = []
    for group in audit.GROUPS:
        expected = producer.receipt_groups(template, group)
        assert audit.logical_receipt_groups(template, group) == expected
        counts.append(len(expected))
    # Synthetic source rows have a different kind ordering; the fixed production
    # source inventory is separately pinned by provenance, not this fixture.
    assert sum(counts) == 132
    assert sum(len(g) for w in audit.GROUPS for g in audit.logical_receipt_groups(template, w)) == 4176


def test_all_missing_retains_parent8352_and4176_denominators(template, tmp_path):
    results = [audit.audit_worker(template, tmp_path / group, group, tokenizer=None,
        receipts=None, metadata={}, reader=audit.Reader()) for group in audit.GROUPS]
    assert sum(len(r["observed"]) for r in results) == 8352
    assert sum(len(r["choice_slots"]) for r in results) == 4176
    assert all(not r["complete"] and r["integrity_passed"] for r in results)
    assert all(row["status"] == "missing" for r in results for row in r["observed"].values())
    assert all(not r[key] for r in results for key in audit.GATES if key != "worker_policies_equal")


def test_attempt_without_result_is_error_not_missing(template, tmp_path):
    cohort = template["execution_cohorts"][0]
    rows = template["assignments"][:32]
    write(tmp_path / "generation_started/000.json", dict(call=audit.expected_call(template, cohort),
        materials_sha256=audit.digest(template), assignments_sha256=audit.digest(rows)))
    result = audit.audit_worker(template, tmp_path, "worker_0", tokenizer=None, receipts=None, metadata={})
    assert sum(r["status"] == "execution_error" for r in result["observed"].values()) == 32
    assert not result["complete"] and result["integrity_passed"]


def test_unassigned_or_duplicate_group_artifacts_rejected(template, tmp_path):
    write(tmp_path / "receipts/999.json", {})
    with pytest.raises(ValueError, match="Unexpected artifact"):
        audit.audit_worker(template, tmp_path, "worker_0", tokenizer=None, receipts=None, metadata={})


def test_reader_rejects_replacement_and_mismatched_array_bytes(tmp_path):
    path = tmp_path / "raw.json"
    write(path, {"x": 1})
    reader = audit.Reader()
    reader.json(path)
    write(path, {"x": 2})
    with pytest.raises(ValueError, match="changed"):
        reader.unchanged()
    engine._persist(tmp_path, "vectors/000", {}, {"hidden": np.zeros((2, 3), np.float32)})
    audit.artifact(tmp_path, "vectors/000", audit.Reader())
    with (tmp_path / "vectors/000.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Artifact bytes"):
        audit.artifact(tmp_path, "vectors/000", audit.Reader())


@pytest.fixture
def raw_worker(tmp_path, observed_calls):
    """Five scientific calls reuse genuine tiny BF16 engine observations."""
    _, requests, direction, reports, arrays = observed_calls
    arms = ("baseline", "record__zero_sham", "full__zero_sham",
            "record__curiosity_negative", "full__curiosity_negative")
    materials = dict(assignments=[], execution_cohorts=[], base_requests=requests[0],
        direction_bindings={key: {"unit": direction.tolist()} for key in ("zero_sham", "curiosity_negative")})
    for index, arm in enumerate(arms):
        rows = [dict(trial_id=f"{arm}-{i}", kind="diagnostic", execution_group="worker_0",
            base_trial_id=f"base-{i}", base_row=i, parent_row=index * 32 + i, arm=arm,
            source_assignment={"trial_id": f"base-{i}"}) for i in range(32)]
        materials["assignments"].extend(rows)
        materials["execution_cohorts"].append(dict(offset=index * 32, arm=arm, source_cohort=0,
            execution_group="worker_0", trial_ids=[r["trial_id"] for r in rows],
            base_trial_ids=[r["base_trial_id"] for r in rows]))
    baseline, descriptor = None, None
    decoded = {}
    for index, original_index in enumerate((0, 1, 2, 4, 5)):
        cohort = materials["execution_cohorts"][index]
        call = audit.expected_call(materials, cohort)
        report, vectors = deepcopy(reports[original_index]), deepcopy(arrays[original_index])
        report["call"] = call
        for row in report["rows"]:
            row["completion"]["sampling"]["max_new_tokens"] = 4096
            decoded[tuple(row["completion"]["generated_token_ids"])] = row["completion"]["raw"]
        common = audit.relationships(report, baseline, call)
        key = call["direction_key"] or "zero_sham"
        report.update(materials_sha256=audit.digest(materials), execution_group="worker_0",
            scientific_arm=cohort["arm"], direction_binding_key=key,
            direction_binding_sha256=audit.digest(materials["direction_bindings"][key]),
            baseline_artifact=descriptor, bridge_relationship_checks=common, bridge_engineering_passed=True)
        rows = materials["assignments"][index * 32:(index + 1) * 32]
        write(tmp_path / f"generation_started/{index:03d}.json", dict(call=call,
            materials_sha256=audit.digest(materials), assignments_sha256=audit.digest(rows)))
        artifact = engine._persist(tmp_path, f"generation/{index:03d}", report, vectors)
        records = [dict(trial_id=a["trial_id"], assignment=a, assignment_sha256=audit.digest(a),
            kind=a["kind"], terminal=True, status="completed", task_record={"fixture": True},
            generation_evidence=artifact, cohort_row=i, engineering_passed=True) for i, a in enumerate(rows)]
        write(tmp_path / f"generation_cohorts/{index:03d}.json", dict(materials_sha256=audit.digest(materials),
            execution_group="worker_0", cohort=cohort, call=call, artifact=artifact,
            records=records, records_sha256=audit.digest(records), engineering_passed=True))
        if index == 0:
            baseline, descriptor = report, artifact
    tokenizer = SimpleNamespace(decode=lambda ids, **kw: decoded[tuple(ids)])
    metadata = dict(hidden_size=16, vocabulary_size=128, pad_token_id=0, generation_eos_token_ids=[127])
    return tmp_path, materials, tokenizer, metadata


def raw_audit(fixture):
    root, materials, tokenizer, metadata = fixture
    return audit.audit_worker(materials, root, "worker_0", tokenizer=tokenizer, receipts=None, metadata=metadata)


def test_actual_bf16_scientific_record_full_and_both_shams(raw_worker):
    result = raw_audit(raw_worker)
    assert result["complete"] and result["integrity_passed"], result["issues"]
    assert len(result["observed"]) == 160
    assert all(result[key] for key in audit.GATES if key != "worker_policies_equal")


@pytest.mark.parametrize("change", ["prepatch", "sham", "arm", "row", "raw", "array"])
def test_rehashed_evidence_cannot_hide_scientific_binding_corruption(raw_worker, change):
    root, _, _, _ = raw_worker
    index = 2
    path = root / f"generation/{index:03d}.json"
    report = json.loads(path.read_text())
    if change == "prepatch":
        report["forwards"][0]["geometry"][0]["before"]["byte_sha256"] = "f" * 64
    elif change == "sham":
        report["forwards"][1]["full_fp32_logits_sha256"][0] = "f" * 64
    elif change == "arm":
        report["direction_binding_key"] = "curiosity_negative"
    elif change == "raw":
        report["rows"][0]["completion"]["raw"] += "unobserved"
    elif change == "array":
        report["array_artifact"]["arrays"]["hidden"]["byte_sha256"] = "f" * 64
    else:
        envelope_path = root / f"generation_cohorts/{index:03d}.json"
        envelope = json.loads(envelope_path.read_text())
        envelope["records"][0]["cohort_row"] = 1
        envelope["records_sha256"] = audit.digest(envelope["records"])
        write(envelope_path, envelope)
    if change != "row":
        write(path, report)
    result = raw_audit(raw_worker)
    assert not result["integrity_passed"] and result["issues"]
    assert len(result["observed"]) == 160


def test_saved_raw_without_terminal_envelope_remains_execution_error(raw_worker):
    root, _, _, _ = raw_worker
    (root / "generation_cohorts/004.json").unlink()
    result = raw_audit(raw_worker)
    assert not result["complete"] and result["integrity_passed"]
    assert sum(r["status"] == "execution_error" for r in result["observed"].values()) == 32


def test_rehashed_deletion_of_all32_prepatch_captures_cannot_pass(raw_worker):
    root, materials, _, metadata = raw_worker
    report, arrays = audit.artifact(root, "generation/004", audit.Reader())
    arrays.pop("hidden")
    report["arrays"].pop("hidden")
    # The shared engineering helper intentionally allows sample-only evidence.
    # Scientific collection separately requires every record/final capture.
    call = report["call"]
    assert audit.numeric.audit_call(report, arrays, materials["base_requests"], call,
        materials["direction_bindings"]["curiosity_negative"]["unit"],
        hidden_size=metadata["hidden_size"], vocabulary_size=metadata["vocabulary_size"],
        pad_token_id=metadata["pad_token_id"], eos_ids=metadata["generation_eos_token_ids"])["passed"]
    for suffix in ("json", "npz"):
        (root / f"generation/004.{suffix}").unlink()
    descriptor = engine._persist(root, "generation/004", report, arrays)
    path = root / "generation_cohorts/004.json"
    envelope = json.loads(path.read_text())
    envelope["artifact"] = descriptor
    for record in envelope["records"]:
        record["generation_evidence"] = descriptor
    envelope["records_sha256"] = audit.digest(envelope["records"])
    write(path, envelope)
    result = raw_audit(raw_worker)
    assert not result["integrity_passed"] and len(result["observed"]) == 160
    assert any("mandatory all32 prepatch" in row.get("message", "") for row in result["issues"])


@pytest.mark.parametrize("missing", ["worker-started.json", "bridge-started.json", "bridge-environment.json",
                                    "summary.json", "bridge-summary.json"])
def test_collected_worker_cannot_omit_start_or_completion_receipts(tmp_path, missing):
    audit.require_worker_evidence(tmp_path)  # Honest never-started workers remain auditable.
    write(tmp_path / "generation_started/000.json", {})
    for name in ("worker-started.json", "bridge-started.json", "bridge-environment.json", "summary.json", "bridge-summary.json"):
        write(tmp_path / name, {})
    audit.require_worker_evidence(tmp_path, complete=True)
    (tmp_path / missing).unlink()
    with pytest.raises(ValueError, match="startup/completion evidence"):
        audit.require_worker_evidence(tmp_path, complete=missing in ("summary.json", "bridge-summary.json"))


def test_sham_comparison_ignores_schedule_label_only(observed_calls):
    left, right = deepcopy(observed_calls[3][0]), deepcopy(observed_calls[3][2])
    assert audit.relationships(right, left, right["call"])["sham_identity"]
    right["rows"][0]["completion"]["raw"] += "different"
    assert not audit.relationships(right, left, right["call"])["sham_identity"]


@pytest.fixture
def receipt_example(monkeypatch):
    """Real collector disposition logic, synthetic sidecar semantics/forward."""
    materials = {"synthetic": True}
    choices = [dict(trial_id=f"outer-{i}", kind="choice") for i in range(5)]
    sources = {}
    for i, row in enumerate(choices):
        sources[row["trial_id"]] = dict(trial_id=row["trial_id"], assignment=row,
            task_record=dict(trial_id=f"inner-{i}", assignment={"i": i}, index=i, cap_hit=False,
                simulator_outcome=dict(receipt_model_pass_completed=False,
                    observation_ledger=[dict(model_pass_completed=False)])))

    def sidecar(inner):
        required = inner["index"] in (0, 3)
        return dict(schema_version=1, source_trial_id=inner["trial_id"], selected_service="report" if required else "stop",
            simulator_outcome_sha256=audit.digest(inner["simulator_outcome"]), terminal=True,
            status="pending" if required else "stop", receipt_required=required, receipt_model_pass_completed=False,
            receipt_input=None, token_binding=None, measurement=None, activation_artifact=None, cohort=None,
            executed_simulator_outcome=deepcopy(inner["simulator_outcome"]))

    def render(inner, model):
        ids = [1, 2, inner["index"] + 3]
        request = dict(token_prefix=ids, token_prefix_sha256=audit.digest(ids), prefix_positions=dict(record=1, final=2))
        return dict(selected_payload=inner["index"]), request, dict(binding=inner["index"])

    def forward(model, requests, ids, layers):
        hidden = np.zeros((len(requests), 1, 2, 4), np.float32)
        measurements = [dict(unpadded_input_ids=r["token_prefix"], input_ids=[0] * 125 + r["token_prefix"],
            attention_mask=[0] * 125 + [1] * 3, position_ids=[0] * 125 + [0, 1, 2],
            unpadded_token_positions=r["prefix_positions"], record_token_index=126,
            token_prefix_sha256=r["token_prefix_sha256"], exact_token_replay=True, prefill_width=128,
            prefill_padding_multiple=128, intervention=None, hooks_remaining=0) for r in requests]
        guard = dict(passed=True, checks={key: True for key in producer.receipts.ENGINEERING_CHECKS},
            source_trial_ids=ids, token_prefix_sha256=[r["token_prefix_sha256"] for r in requests],
            layers=layers, positions=["record", "final"], max_new_tokens=0, active_batch_size=len(requests),
            prefill_width=128, captured_array_sha256=producer.receipts.array_sha(hidden),
            baseline_full_initial_logits_sha256=[audit.digest(i) for i in ids],
            captured_full_initial_logits_sha256=[audit.digest(i) for i in ids])
        return measurements, hidden, guard

    monkeypatch.setattr(producer.receipts, "choice_sidecar", sidecar)
    monkeypatch.setattr(producer.receipts, "render_receipt", render)
    monkeypatch.setattr(producer.receipts, "forward_receipts", forward)
    monkeypatch.setattr(producer.engine, "hook_inventory", lambda model: {})
    model = SimpleNamespace(active_hooks=0)
    report, arrays = producer.collect_receipt_group(model, choices, sources)
    report.update(materials_sha256=audit.digest(materials), execution_group="worker_0", logical_choice_offset=0)
    return report, arrays, choices, sources, dict(materials=materials, group="worker_0", index=0,
        tokenizer=SimpleNamespace(pad_token_id=0), receipts=producer.receipts, hidden_size=4), model


def test_receipt_original_slots_and_active_subset_match_actual_collector(receipt_example):
    report, arrays, choices, sources, kwargs, _ = receipt_example
    checked = audit.audit_receipt_group(report, arrays, choices, sources, **kwargs)
    assert checked["passed"] and checked["assigned_choices"] == 5 and checked["actual_selected_rows"] == 2
    assert [row["status"] for row in report["records"]] == ["completed", "stop", "stop", "completed", "stop"]


@pytest.mark.parametrize("change", ["group", "selected_payload", "padding", "source_hash", "array", "error"])
def test_rehashed_receipt_semantic_and_actual_model_bindings_cannot_change(receipt_example, change):
    report, arrays, choices, sources, kwargs, _ = receipt_example
    report, arrays = deepcopy(report), deepcopy(arrays)
    if change == "group":
        report["source_trial_ids"] = report["active_source_trial_ids"]
    elif change == "selected_payload":
        report["records"][0]["receipt_input"]["selected_payload"] = 3
    elif change == "padding":
        report["observed_measurements"][0]["attention_mask"][0] = 1
    elif change == "source_hash":
        report["records"][0]["source_record_sha256"] = "f" * 64
    elif change == "array":
        arrays["hidden"][0, 0, 0, 0] = 1
    else:
        report["error"] = dict(type="RuntimeError", message="not actually successful")
    report["records_sha256"] = audit.digest(report["records"])
    with pytest.raises(ValueError):
        audit.audit_receipt_group(report, arrays, choices, sources, **kwargs)


def test_actual_receipt_forward_failure_preserves_terminal_error_slots(receipt_example, monkeypatch):
    _, _, choices, sources, kwargs, model = receipt_example

    def failed(*args, **kw):
        raise RuntimeError("synthetic receipt failure")

    monkeypatch.setattr(producer.receipts, "forward_receipts", failed)
    report, arrays = producer.collect_receipt_group(model, choices, sources)
    report.update(materials_sha256=audit.digest(kwargs["materials"]), execution_group="worker_0", logical_choice_offset=0)
    checked = audit.audit_receipt_group(report, arrays, choices, sources, **kwargs)
    assert not checked["passed"] and checked["assigned_choices"] == 5
    assert [row["status"] for row in report["records"]] == ["execution_error", "stop", "stop", "execution_error", "stop"]
    assert not any(row["receipt_model_pass_completed"] for row in report["records"])


@pytest.mark.parametrize("fault", ["logits", "nonfinite_capture"])
def test_actual_failed_receipt_guard_keeps_measurements_and_failure(receipt_example, monkeypatch, fault):
    _, _, choices, sources, kwargs, model = receipt_example
    original = producer.receipts.forward_receipts

    def failed(*args, **kw):
        measurements, hidden, guard = original(*args, **kw)
        if fault == "logits":
            guard["captured_full_initial_logits_sha256"][0] = "f" * 64
            guard["checks"]["full_initial_logits_equal"] = False
        else:
            hidden[0, 0, 0, 0] = np.nan
            guard["captured_array_sha256"] = producer.receipts.array_sha(hidden)
            guard["checks"]["all_captures_finite"] = False
        guard["passed"] = False
        return measurements, hidden, guard

    monkeypatch.setattr(producer.receipts, "forward_receipts", failed)
    report, arrays = producer.collect_receipt_group(model, choices, sources)
    report.update(materials_sha256=audit.digest(kwargs["materials"]), execution_group="worker_0", logical_choice_offset=0)
    checked = audit.audit_receipt_group(report, arrays, choices, sources, **kwargs)
    assert not checked["passed"] and report["observed_measurements"] is not None and "hidden" in arrays
    assert not any(row["receipt_model_pass_completed"] for row in report["records"])


@pytest.fixture
def root_context(template, tmp_path, monkeypatch):
    """Exercise root composition without importing or reading any real run."""
    manifest = dict(plan=template, plan_sha256=audit.digest(template))
    admission = dict(created=1., deadline=28001.)
    metadata = dict(template["model"], template_sha256=audit.digest("synthetic"), pad_token_id=0,
        hidden_size=4, vocabulary_size=128, generation_eos_token_ids=[127], load_seconds=1.,
        versions={"torch": "synthetic"})
    for group in audit.GROUPS:
        root = tmp_path / "workers" / group
        for name, value in (("manifest", manifest), ("admission", admission),
                            ("worker-claim", {}), ("environment", metadata)):
            write(root / (name + ".json"), value)

    def score(materials, observed):
        return dict(outcomes=[dict(trial_id=a["trial_id"], status=observed[a["trial_id"]]["status"])
            for a in materials["assignments"]], readiness=dict(passed=False),
            checks=dict(computation_agrees=True, complete_response_coverage=False, all_assigned_slots_retained=True))

    def semantics(*args):
        return dict(checks=dict(semantic_agreement=True, complete_receipt_coverage=False,
            required_semantic_dispositions_passed=False))

    scorer = SimpleNamespace(score_collection=score, check_receipts=semantics)
    worker = SimpleNamespace(validate_worker_claim=lambda *a, **kw: None,
        validate_runtime_metadata=lambda *a: None)

    def context(*args):
        return manifest, admission, tmp_path, {"voluntary_capture": None}, worker, scorer, \
            SimpleNamespace(chat_template="synthetic", pad_token_id=0)

    monkeypatch.setattr(audit, "source_context", context)
    return tmp_path, scorer


def test_root_combination_retains_all_slots_and_never_admits_missing_collection(root_context):
    root, _ = root_context
    result = audit.audit(root)
    assert not result["complete"] and result["integrity_passed"]
    assert len(result["outcomes"]) == 8352 and result["assigned_choices"] == 4176
    assert all(r["status"] == "missing" for r in result["outcomes"])
    assert set(result["checks"]) == set(audit.GATES)
    assert result["worker_policies_equal"]
    assert not result["passed"] and not result["lifecycle_shutdown_verified"] and not result["scientific_admission"]


def test_three_worker_policies_exclude_loading_time_but_not_numerical_versions(root_context):
    root, _ = root_context
    path = root / "workers/worker_2/environment.json"
    metadata = json.loads(path.read_text())
    metadata["load_seconds"] = 100
    write(path, metadata)
    assert audit.audit(root)["worker_policies_equal"]
    metadata["versions"]["torch"] = "different"
    write(path, metadata)
    result = audit.audit(root)
    assert not result["worker_policies_equal"] and not result["passed"]


def test_independent_scoring_disagreement_vetoes_integrity_without_dropping_rows(root_context):
    root, scorer = root_context
    original = scorer.score_collection

    def changed(*args):
        value = original(*args)
        value["checks"]["computation_agrees"] = False
        return value

    scorer.score_collection = changed
    result = audit.audit(root)
    assert not result["integrity_passed"] and len(result["outcomes"]) == 8352


def test_missing_runtime_is_unavailable_and_does_not_drop_worker(root_context):
    root, _ = root_context
    (root / "workers/worker_1/environment.json").unlink()
    result = audit.audit(root)
    assert not result["integrity_passed"] and not result["runtime_verified"]
    assert not result["worker_policies_equal"] and len(result["outcomes"]) == 8352
