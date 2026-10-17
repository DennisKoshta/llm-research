"""Unstarted/partial scientific coverage and hook-free execution wiring."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

import curiosity_coverage_bridge_worker as worker
import launch_curiosity_coverage_bridge as controller
from test_coverage_bridge import template as template_fixture


@pytest.fixture(scope="module")
def plan():
    return template_fixture.__wrapped__()


@pytest.fixture
def manifest(plan):
    return dict(
        name=worker.NAME,
        plan=plan,
        plan_sha256=worker.digest(plan),
        resources=worker.resources(),
        workers=worker.worker_descriptors(plan),
        preflight=dict(synthetic=True),
        wrapper_files={Path(worker.__file__).name: worker.sha(worker.__file__)},
        scientific_admission=False,
        causal_admission=False,
        resource_amendment=dict(amended_resource_check_passed=True, original_forecast_passed=False),
    )


def source_run(tmp_path, manifest):
    state = dict(
        all_stopped_zero_tasks=True,
        available_compute_estimate_usd=207,
        phase_compute_estimate_usd=93,
        timestamp_utc="synthetic",
    )
    admission = controller.claim(tmp_path, manifest, state, now=1000)
    claim = controller.claim_worker(tmp_path, manifest, admission, "worker_0", worker, now=1001)
    return tmp_path / "runs" / worker.NAME / "workers/worker_0", admission, claim


def test_all8352_and4176_slots_survive_without_any_worker_output(tmp_path, manifest):
    reports = [worker.operational_coverage(tmp_path / g, manifest, g) for g in worker.GROUPS]
    assert sum(len(r["generation_slots"]) for r in reports) == 8352
    assert [len(r["choice_slots"]) for r in reports] == [
        1566,
        1392,
        1218,
    ]  # Synthetic fixture's distinct ordering.
    assert sum(len(r["choice_slots"]) for r in reports) == 4176
    assert all(
        s["status"] == "missing" for r in reports for k in ("generation_slots", "choice_slots") for s in r[k]
    )
    assert not any(r["complete"] for r in reports)


def test_started_generation_and_receipt_slots_are_errors_not_dropped(tmp_path, manifest):
    cohort = manifest["plan"]["execution_cohorts"][0]
    rows = manifest["plan"]["assignments"][:32]
    worker.write_once(
        tmp_path / "generation_started/000.json",
        dict(
            materials_sha256=manifest["plan_sha256"],
            assignments_sha256=worker.digest(rows),
            call=dict(synthetic=True),
        ),
    )
    choices = [
        a
        for a in manifest["plan"]["assignments"]
        if a["execution_group"] == "worker_0" and a["kind"] == "choice"
    ][:32]
    worker.write_once(
        tmp_path / "receipt_started/000.json",
        dict(
            materials_sha256=manifest["plan_sha256"],
            logical_choice_offset=0,
            source_trial_ids=[a["trial_id"] for a in choices],
        ),
    )
    result = worker.operational_coverage(tmp_path, manifest, "worker_0")
    assert cohort["offset"] == 0
    assert sum(r["status"] == "execution_error" for r in result["generation_slots"]) == 32
    assert sum(r["status"] == "execution_error" for r in result["choice_slots"]) == 32
    assert len(result["generation_slots"]) == 2784 and len(result["choice_slots"]) == 1566
    assert not result["complete"]


def durable_cohort(root, manifest):
    plan, cohort = manifest["plan"], manifest["plan"]["execution_cohorts"][0]
    rows, call = plan["assignments"][:32], dict(call_index=0, synthetic=True)
    worker.write_once(
        root / "generation_started/000.json",
        dict(materials_sha256=manifest["plan_sha256"], assignments_sha256=worker.digest(rows), call=call),
    )
    array_path = root / "generation/000.npz"
    array_path.parent.mkdir()
    np.savez_compressed(array_path, hidden=np.array([1], np.float32))
    worker.write_once(
        root / "generation/000.json",
        dict(call=call, array_artifact=dict(path="generation/000.npz", byte_sha256=worker.sha(array_path))),
    )
    artifact = dict(path="generation/000.json", byte_sha256=worker.sha(root / "generation/000.json"))
    records = [
        dict(
            trial_id=a["trial_id"],
            assignment=a,
            assignment_sha256=worker.digest(a),
            terminal=True,
            status="completed",
            task_record=dict(synthetic=True),
        )
        for a in rows
    ]
    envelope = dict(
        materials_sha256=manifest["plan_sha256"],
        execution_group="worker_0",
        cohort=cohort,
        call=call,
        records=records,
        records_sha256=worker.digest(records),
        artifact=artifact,
    )
    worker.write_once(root / "generation_cohorts/000.json", envelope)
    return envelope


@pytest.mark.parametrize("change", [None, "array", "assignment", "extra", "missing_attempt"])
def test_partial_terminal_evidence_requires_exact_bindings(tmp_path, manifest, change):
    envelope = durable_cohort(tmp_path, manifest)
    if change == "array":
        (tmp_path / "generation/000.npz").write_bytes(b"changed")
    elif change == "assignment":
        envelope["records"][0]["assignment"] = {}
        envelope["records_sha256"] = worker.digest(envelope["records"])
        (tmp_path / "generation_cohorts/000.json").write_text(worker.canonical(envelope))
    elif change == "extra":
        worker.write_once(tmp_path / "generation/999.json", {})
    elif change == "missing_attempt":
        (tmp_path / "generation_started/000.json").unlink()
    result = worker.operational_coverage(tmp_path, manifest, "worker_0")
    assert not result["complete"] and len(result["generation_slots"]) == 2784
    if change is None:
        assert not result["issues"]
        assert sum(r["status"] == "completed" for r in result["generation_slots"]) == 32
    else:
        assert result["issues"]


def test_constructor_failure_retains_inventory_and_cannot_retry(tmp_path, manifest, monkeypatch):
    root, _, _ = source_run(tmp_path, manifest)
    calls = []

    def construct(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("synthetic model load failure")

    monkeypatch.setattr(
        worker, "verify_sources", lambda *a: dict(batched=SimpleNamespace(BatchedHFModel=construct))
    )
    monkeypatch.setattr(worker, "engineering_helper", lambda *a: object())
    monkeypatch.setattr(worker.time, "time", lambda: 1002)
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(key, "1")
    result = worker.run(tmp_path, tmp_path, tmp_path, root, "worker_0")
    assert len(calls) == 1 and result["error"]["message"] == "synthetic model load failure"
    assert len(result["operational_coverage"]["generation_slots"]) == 2784
    assert not result["complete"] and not result["engineering_passed"]
    with pytest.raises(FileExistsError):
        worker.run(tmp_path, tmp_path, tmp_path, root, "worker_0")
    assert len(calls) == 1


def test_worker_calls_only_held_collector_once_with_original_common_deadline(tmp_path, manifest, monkeypatch):
    root, admission, _ = source_run(tmp_path, manifest)
    seen = []
    model = SimpleNamespace(tokenizer=object())

    def execute(out, actual_model, plan, **kwargs):
        seen.append((out, actual_model, plan, kwargs))
        kwargs["checkpoint"]()
        return dict(complete=False, engineering_passed=False)

    modules = dict(
        batched=SimpleNamespace(BatchedHFModel=lambda **kw: model),
        voluntary_resolution=object(),
        coverage_bridge=SimpleNamespace(execute=execute),
    )
    monkeypatch.setattr(worker, "verify_sources", lambda *a: modules)
    helper = SimpleNamespace(runtime_metadata=lambda *a: dict(inherited_synthetic_startup_forwards=0))
    monkeypatch.setattr(worker, "engineering_helper", lambda *a: helper)
    monkeypatch.setattr(worker, "tokenizer_preflight", lambda *a: manifest["preflight"])
    monkeypatch.setattr(worker.time, "time", lambda: 1002)
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(key, "1")
    result = worker.run(tmp_path, tmp_path, tmp_path, root, "worker_0")
    assert len(seen) == 1 and seen[0][1] is model and seen[0][2] == manifest["plan"]
    assert seen[0][3]["execution_group"] == "worker_0"
    assert seen[0][3]["deadline"] == admission["deadline"] - 60
    assert not result["complete"]
    assert worker.read(root / "environment.json")["inherited_synthetic_startup_forwards"] == 0


def test_actual_parent_admission_required_for_worker_claim(tmp_path, manifest):
    root, admission, _ = source_run(tmp_path, manifest)
    global_path = tmp_path / "campaign_admissions" / worker.NAME / "admission.json"
    altered = deepcopy(admission)
    altered["deadline"] += 1
    global_path.write_text(worker.canonical(altered))
    with pytest.raises(ValueError, match="exclusive parent"):
        controller.claim_worker(tmp_path, manifest, admission, "worker_1", worker, now=1002)
    assert not (root.parent / "worker_1").exists()


@pytest.mark.parametrize(
    "change", [None, "engineering_failed", "reclassified_v1", "retuned_estimate", "changed_document"]
)
def test_engineering_and_amended_forecast_gate_preserves_original_failure(tmp_path, monkeypatch, change):
    prefix = f"runs/{worker.ENGINEERING_NAME}/"
    context = dict(passed=True, complete=True, manifest_sha256=worker.ENGINEERING_SHA)
    lifecycle = dict(passed=True, manifest_sha256=worker.ENGINEERING_SHA, raw_input_files_sha256="b" * 64)
    forecast = dict(
        available=True, passed=False, workers=[dict(forecast_seconds=x) for x in (27000.0, 26000.0, 25000.0)]
    )
    paths = {
        prefix + "audit.json": context,
        prefix + "lifecycle-audit.json": lifecycle,
        worker.FORECAST: forecast,
    }
    for relative, value in paths.items():
        if relative.endswith("lifecycle-audit.json"):
            value["context_audit_byte_sha256"] = worker.sha(tmp_path / "evidence" / prefix / "audit.json")
        worker.write_once(tmp_path / "evidence" / relative, value)
    documents = {name: worker.sha(tmp_path / "evidence" / name) for name in paths}
    amended = dict(
        amended_resource_check_passed=True,
        scientific_admission=False,
        protocol_deviation=True,
        original_forecast_passed=False,
        assignment_sha256="a" * 64,
        engineering_manifest_sha256=worker.ENGINEERING_SHA,
        engineering_raw_input_files_sha256=lifecycle["raw_input_files_sha256"],
        resources=dict(
            common_startup_inclusive_wall_seconds=28000,
            maximum_compute_liability_usd="117.57759360",
            stage_ceiling_usd="120.00",
        ),
        inputs={name: dict(path=name, byte_sha256=value) for name, value in documents.items()},
        workers=[dict(unchanged_forecast_seconds=w["forecast_seconds"]) for w in forecast["workers"]],
    )
    worker.write_once(tmp_path / "evidence" / worker.AMENDMENT, amended)
    documents[worker.AMENDMENT] = worker.sha(tmp_path / "evidence" / worker.AMENDMENT)
    monkeypatch.setattr(worker, "AMENDMENT_SHA", documents[worker.AMENDMENT])
    manifest = dict(documents=documents, resource_amendment=amended, plan=dict(assignment_sha256="a" * 64))
    if change == "engineering_failed":
        context["passed"] = False
        path = tmp_path / "evidence" / prefix / "audit.json"
        path.write_text(worker.canonical(context))
        documents[prefix + "audit.json"] = worker.sha(path)
    elif change in ("reclassified_v1", "retuned_estimate"):
        if change == "reclassified_v1":
            forecast["passed"] = True
        else:
            forecast["workers"][0]["forecast_seconds"] = 1000.0
        path = tmp_path / "evidence" / worker.FORECAST
        path.write_text(worker.canonical(forecast))
        documents[worker.FORECAST] = worker.sha(path)
    elif change == "changed_document":
        (tmp_path / "evidence" / worker.AMENDMENT).write_text("{}")
    if change is None:
        assert worker.validate_evidence(manifest, tmp_path)["forecast"]["passed"] is False
    else:
        with pytest.raises(ValueError):
            worker.validate_evidence(manifest, tmp_path)
