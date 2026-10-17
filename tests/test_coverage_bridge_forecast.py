"""Prospective arithmetic and metadata bindings; no real engineering outcomes."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import forecast_curiosity_coverage_bridge as forecast  # noqa: E402

CHOICES = [20, 17, 16, 15, 13, 18, 18, 17, 10]


@pytest.fixture
def inputs():
    rows, cohorts, dispositions = [], [], []
    for cohort in range(9):
        for arm in forecast.ARMS:
            block = []
            for index in range(32):
                row = dict(trial_id=f"{cohort}:{arm}:{index}", base_trial_id=f"base{cohort}:{index}",
                    kind="choice" if index < CHOICES[cohort] else "diagnostic", arm=arm,
                    base_cohort=cohort, execution_group=f"worker_{cohort % 3}")
                block.append(row)
                if row["kind"] == "choice":
                    dispositions.append(dict(trial_id=row["trial_id"], base_trial_id=row["base_trial_id"],
                        non_stop_bindings={s: dict(prefix_positions=dict(record=98, final=99))
                                           for s in ("report", "repeat", "pattern")}))
            cohorts.append(dict(offset=len(rows), source_cohort=cohort, arm=arm,
                execution_group=f"worker_{cohort % 3}", trial_ids=[a["trial_id"] for a in block]))
            rows.extend(block)
    plan = dict(assignments=rows, assignment_sha256=forecast.digest(rows), execution_cohorts=cohorts,
        execution_cohorts_sha256=forecast.digest(cohorts), receipt_materials=dict(
            assigned_dispositions=dispositions, assigned_dispositions_sha256=forecast.digest(dispositions)))
    historical = dict(calls=[dict(source_cohort=c, arm=a, generation_seconds=10.)
                            for c in range(9) for a in forecast.OLD_ARMS],
        profiles=[dict(worker=w, execution_seconds=600., load_seconds=50.) for w in range(3)])
    timing = []
    for cohort in (0, 3):
        for condition, seconds in (("baseline", 20.), ("record_nonzero", 30.), ("full_nonzero", 60.)):
            timing.append(dict(source_cohort=cohort, condition=condition, generation_seconds=seconds,
                generation_artifact_bytes=10000, generation_serialization_seconds=.1,
                receipt_rows=20, receipt_padded_width=128, receipt_forward_calls=2,
                receipt_forward_seconds=2., receipt_parsing_seconds=.1,
                receipt_artifact_bytes=1000, receipt_serialization_seconds=.01))
    engineering = dict(passed=True, timing_calls=timing, all16_generation_seconds=230.,
        collector_elapsed_seconds=300., observed_artifact_bytes=180000,
        checkpoint_commit_seconds=[.1, .2, .1], export_seconds=2., load_seconds=40.,
        startup_nonloading_seconds=15., lifecycle_overhead_seconds=5.)
    return plan, historical, engineering


def test_fixed_ratios_all261_calls_and_original_receipt_groups(inputs):
    result = forecast.forecast_tables(*inputs)
    assert result["available"] and result["passed"]
    assert result["common_engine_slowdown"] == 3 and result["full_schedule_slowdown"] == 2
    assert len(result["generation_calls"]) == 261 and len(result["receipt_batches"]) == 132
    assert [w["receipt_batches"] for w in result["workers"]] == [49, 43, 40]
    assert [w["maximum_receipt_rows"] for w in result["workers"]] == [1537, 1363, 1276]
    assert [w["components"]["generation_seconds"] for w in result["workers"]] == [3870.] * 3
    assert result["maximum_compute_liability_usd"] == pytest.approx(79.9460736)
    assert result["retained_storage_reserve_usd"] == 10
    assert not result["scientific_admission"] and not result["global_budget_admission"]
    # Group0 contains20 choices from baseline and12 from the next arm; no
    # generation-cohort or arm-boundary reblocking is silently introduced.
    batch = result["receipt_batches"][0]
    assert batch["trial_ids"][:20] == [f"0:baseline:{i}" for i in range(20)]
    assert batch["trial_ids"][20:] == [f"0:{forecast.ARMS[1]}:{i}" for i in range(12)]
    assert result["receipt_batches"][48]["rows"] == 1


def test_each_control_uses_its_own_historical_runtime(inputs):
    plan, history, engineering = inputs
    chosen = next(c for c in history["calls"] if (c["source_cohort"], c["arm"]) == (4, "random_05"))
    chosen["generation_seconds"] = 100.
    history["profiles"][1]["execution_seconds"] += 90
    result = forecast.forecast_tables(plan, history, engineering)
    predictions = {r["arm"]: r for r in result["generation_calls"] if r["source_cohort"] == 4}
    assert predictions["record__random_05"]["generation_seconds"] == 300
    assert predictions["full__random_05"]["generation_seconds"] == 600
    assert predictions["record__curiosity_negative"]["generation_seconds"] == 30


def test_both_ratios_are_floored_at_one(inputs):
    for row in inputs[2]["timing_calls"]:
        row["generation_seconds"] = 1 if row["condition"] == "full_nonzero" else 2
    result = forecast.forecast_tables(*inputs)
    assert result["common_engine_slowdown"] == result["full_schedule_slowdown"] == 1


def test_squared_receipt_width_assumption_is_explicit_and_not_average(inputs):
    first = forecast.forecast_tables(*inputs)
    for item in inputs[0]["receipt_materials"]["assigned_dispositions"]:
        for binding in item["non_stop_bindings"].values():
            binding["prefix_positions"]["final"] = 255
    inputs[0]["receipt_materials"]["assigned_dispositions_sha256"] = forecast.digest(
        inputs[0]["receipt_materials"]["assigned_dispositions"])
    second = forecast.forecast_tables(*inputs)
    for a, b in zip(first["receipt_batches"], second["receipt_batches"], strict=True):
        assert b["paired_forward_and_parsing_seconds"] == 4 * a["paired_forward_and_parsing_seconds"]
        assert b["artifact_bytes"] == 2 * a["artifact_bytes"]


def test_new_end_to_end_residual_covers_unallocated_prepare_and_checks(inputs):
    result = forecast.forecast_tables(*inputs)
    for worker in result["workers"]:
        assert worker["new_mixed_overhead_seconds"] == 70
        assert worker["historical_mixed_overhead_seconds"] == 150
        assert worker["components"]["mixed_overhead_reserve_seconds"] == 70 * 87 / 16
        assert worker["forecast_seconds"] == (worker["startup_nonloading_seconds"] + worker["loading_seconds"]
                                              + 1.25 * sum(worker["components"].values()))


def test_deadline_failure_keeps_every_estimate_and_never_shrinks_panel(inputs):
    result = forecast.forecast_tables(*inputs)
    maximum = max(w["forecast_seconds"] for w in result["workers"])
    inputs[2]["startup_nonloading_seconds"] += 19000 - maximum + .001
    result = forecast.forecast_tables(*inputs)
    assert result["available"] and not result["passed"]
    assert len(result["generation_calls"]) == 261 and len(result["receipt_batches"]) == 132
    assert any(not w["fits_deadline"] for w in result["workers"])
    assert result["retained_storage_reserve_usd"] == 10


@pytest.mark.parametrize("mutation", ["old_missing", "old_duplicate", "new_missing", "no_receipt",
    "single_pass_receipt", "negative_old_residual", "negative_new_residual", "nan", "missing_commits",
    "failed_audit", "wider_actual_receipt", "missing_payload", "wrong_worker"])
def test_missing_or_invalid_decomposition_never_certifies_feasibility(inputs, mutation):
    plan, history, engineering = inputs
    if mutation == "old_missing":
        history["calls"].pop()
    elif mutation == "old_duplicate":
        history["calls"][-1] = deepcopy(history["calls"][0])
    elif mutation == "new_missing":
        engineering["timing_calls"].pop()
    elif mutation == "no_receipt":
        for row in engineering["timing_calls"]:
            row.update(receipt_rows=0, receipt_padded_width=None, receipt_forward_calls=0, receipt_forward_seconds=0)
    elif mutation == "single_pass_receipt":
        engineering["timing_calls"][0]["receipt_forward_calls"] = 1
    elif mutation == "negative_old_residual":
        history["profiles"][0]["execution_seconds"] = 1
    elif mutation == "negative_new_residual":
        engineering["collector_elapsed_seconds"] = 1
    elif mutation == "nan":
        engineering["export_seconds"] = float("nan")
    elif mutation == "missing_commits":
        engineering["checkpoint_commit_seconds"] = []
    elif mutation == "failed_audit":
        engineering["passed"] = False
    elif mutation == "wider_actual_receipt":
        engineering["timing_calls"][0]["receipt_padded_width"] = 256
    elif mutation == "missing_payload":
        plan["receipt_materials"]["assigned_dispositions"][0]["non_stop_bindings"].pop("repeat")
        plan["receipt_materials"]["assigned_dispositions_sha256"] = forecast.digest(
            plan["receipt_materials"]["assigned_dispositions"])
    else:
        plan["assignments"][0]["execution_group"] = "worker_2"
        plan["assignment_sha256"] = forecast.digest(plan["assignments"])
    with pytest.raises(ValueError):
        forecast.forecast_tables(plan, history, engineering)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(forecast.canonical(value) + "\n")


def test_reader_binds_bytes_and_rejects_symlink(tmp_path):
    path = tmp_path / "profile.json"
    write(path, {"seconds": 1})
    reader = forecast.Reader()
    assert reader.json(path) == {"seconds": 1}
    assert reader.bindings[str(path)]["byte_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    link = tmp_path / "alias.json"
    link.symlink_to(path)
    with pytest.raises(ValueError):
        reader.json(link)


def test_rebinding_or_final_recheck_cannot_replace_original_audited_bytes(tmp_path):
    path = tmp_path / "raw.json"
    write(path, {"seconds": 1})
    reader = forecast.Reader()
    initial = reader.bind(path)
    write(path, {"seconds": 2})
    for check in (lambda: reader.bind(path), reader.verify_unchanged):
        with pytest.raises(ValueError, match="Previously bound input changed"):
            check()
    assert reader.bindings[str(path)] == initial


def test_json_parsing_cannot_use_bytes_changed_after_binding(tmp_path, monkeypatch):
    path = tmp_path / "raw.json"
    write(path, {"seconds": 1})
    reader = forecast.Reader()
    original = reader.bind

    def changing_bind(value):
        bound = original(value)
        write(path, {"seconds": 2})
        return bound

    monkeypatch.setattr(reader, "bind", changing_bind)
    with pytest.raises(ValueError, match="between binding and parsing"):
        reader.json(path)


def test_successful_arithmetic_cannot_certify_inputs_changed_during_forecast(tmp_path, inputs, monkeypatch):
    path = tmp_path / "measured.json"
    write(path, {"seconds": 1})
    (tmp_path / "reports").mkdir()
    (tmp_path / forecast.METHOD).write_text("synthetic method")

    def history(project, reader):
        reader.json(path)
        return inputs[1]

    original = forecast.forecast_tables

    def changed(*args):
        result = original(*args)
        assert result["passed"]
        write(path, {"seconds": 2})
        return result

    monkeypatch.setattr(forecast, "historical_inputs", history)
    monkeypatch.setattr(forecast, "engineering_inputs", lambda *args: (inputs[0], inputs[2]))
    monkeypatch.setattr(forecast, "forecast_tables", changed)
    result = forecast.forecast_run(tmp_path, tmp_path / "run", tmp_path / "audit.json")
    assert not result["available"] and not result["passed"]
    assert "Previously bound input changed" in result["error"]["message"]


def test_unavailable_run_returns_failed_receipt_and_preserves_input_bindings(tmp_path):
    (tmp_path / "reports").mkdir()
    (tmp_path / forecast.METHOD).write_text("synthetic method")
    result = forecast.forecast_run(tmp_path, tmp_path / "missing", tmp_path / "audit.json")
    assert not result["available"] and not result["passed"] and not result["scientific_admission"]
    assert result["retained_storage_reserve_usd"] == 10 and len(result["input_files"]) >= 2


@pytest.fixture
def historical_files(tmp_path, monkeypatch):
    parent = tmp_path / "runs" / forecast.PARENT
    assignments = []
    for c in range(9):
        for arm in forecast.OLD_ARMS:
            assignments.extend(dict(trial_id=f"{c}:{arm}:{i}", base_cohort=c, arm=arm,
                execution_group=f"worker_{c % 3}") for i in range(32))
    manifest = dict(diagnostics=assignments)
    checksum = forecast.digest(manifest)
    monkeypatch.setattr(forecast, "PARENT_SHA", checksum)
    write(parent / "manifest.json", manifest)
    archive = parent / "source.tar.gz"
    archive.write_bytes(b"synthetic immutable archive")
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    monkeypatch.setattr(forecast, "ARCHIVE_SHA", archive_sha)
    write(parent / "freeze.json", dict(manifest_sha256=checksum, archive_sha256=archive_sha))
    for w in range(3):
        child = Path(str(parent) + f"-worker-{w}")
        rows = [a for a in assignments if a["execution_group"] == f"worker_{w}"]
        m = dict(diagnostics=rows, assignment_sha256=forecast.digest(rows), parent_manifest_sha256=checksum)
        write(child / "manifest.json", m)
        for offset in range(0, 1440, 32):
            block = rows[offset:offset + 32]
            write(child / "generation_evidence/attempt" / f"{offset:07d}.json",
                dict(generation_offset=offset, manifest_sha256=forecast.digest(m),
                    trial_ids=[a["trial_id"] for a in block], engineering_passed=True,
                    condition=dict(elapsed_seconds=10., error=None,
                        rows="Deliberately not a response list: metadata reader must never parse behavior.")))
        write(child / "cloud_profiles/attempt.json", dict(assigned=1440, completed=1440,
                                                          execution_seconds=500., load_seconds=50.))
    return tmp_path, parent


def test_historical_metadata_reader_uses_all135_calls_and_never_behavior(historical_files):
    project, _ = historical_files
    reader = forecast.Reader()
    result = forecast.historical_inputs(project, reader)
    assert len(result["calls"]) == 135 and len(result["profiles"]) == 3
    assert sum(r["generation_seconds"] for r in result["calls"]) == 1350
    assert all(set(r) == {"source_cohort", "arm", "generation_seconds", "evidence_path"} for r in result["calls"])
    assert len(reader.bindings) == 144


@pytest.mark.parametrize("mutation", ["trial_order", "offset", "failed", "profile", "missing_call"])
def test_historical_reader_rejects_rehashed_unmatched_or_missing_metadata(historical_files, mutation):
    project, parent = historical_files
    child = Path(str(parent) + "-worker-0")
    path = child / "generation_evidence/attempt/0000000.json"
    if mutation == "missing_call":
        path.unlink()
    elif mutation == "profile":
        write(child / "cloud_profiles/attempt.json", dict(assigned=1440, completed=1439,
                                                          execution_seconds=500., load_seconds=50.))
    else:
        value = json.loads(path.read_text())
        if mutation == "trial_order":
            value["trial_ids"].reverse()
        elif mutation == "offset":
            value["generation_offset"] = 32
        else:
            value["engineering_passed"] = False
        write(path, value)
    with pytest.raises(ValueError):
        forecast.historical_inputs(project, forecast.Reader())


@pytest.fixture
def engineering_files(tmp_path, inputs, monkeypatch):
    project, run = tmp_path, tmp_path / "runs/engineering"
    plan = inputs[0]
    original = {"synthetic_original": True}
    monkeypatch.setattr(forecast, "PARENT_SHA", forecast.digest(original))
    calls = [dict(call_index=i, source_cohort=0 if i < 13 else 3,
                  condition=("baseline", "record_nonzero", "full_nonzero")[(i - 10) % 3]
                  if i >= 10 else "synthetic_check") for i in range(16)]
    plan["engineering"] = {"call_plan": calls}
    manifest = dict(plan=plan, original_manifest=original)
    manifest_sha = forecast.digest(manifest)
    write(run / "manifest.json", manifest)
    write(run / "freeze.json", dict(manifest_sha256=manifest_sha))
    admission = dict(created=0., manifest_sha256=manifest_sha)
    write(run / "admission.json", admission)
    environment = dict(load_seconds=50.)
    write(run / "environment.json", environment)
    write(run / "worker-started.json", dict(started=20.))
    write(run / "engineering-started.json", dict(started=80., call_plan_sha256=forecast.digest(calls)))
    write(run / "launch-outcome.json", dict(all_stopped_zero_tasks=True,
        export_error=None, failure=None, export_seconds=2.))
    write(run / "cloud-finished.json", dict(finished=400., error=None,
        checkpoint_commit_seconds=[.1, .2], checkpoint_commit_total_seconds=.1 + .2,
        final_commit_timing_excluded=True))
    dispositions, timings, artifacts, receipt_checks = [], [], [], {}

    def artifact(folder, index, report):
        relative = f"{folder}/{index:02d}.npz"
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic immutable numerical arrays: checked separately")
        array_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        report["array_artifact"] = dict(path=relative, byte_sha256=array_sha)
        json_path = run / folder / f"{index:02d}.json"
        write(json_path, report)
        size = path.stat().st_size + json_path.stat().st_size
        return dict(byte_sha256=hashlib.sha256(json_path.read_bytes()).hexdigest(),
                    artifact_bytes=size, serialization_seconds=.01), array_sha

    for i, call in enumerate(calls):
        seconds = (20., 30., 60.)[(i - 10) % 3] if i >= 10 else 1.
        persisted, array_sha = artifact("calls", i,
            dict(call=call, passed=True, generation_seconds=seconds,
                 rows="Not parsed: behavior has no place in this forecast"))
        artifacts.append(dict(call_index=i, report_byte_sha256=persisted["byte_sha256"], array_byte_sha256=array_sha))
        if i >= 10:
            receipt = dict(call_index=i, passed=True, error=None, assigned_choices=20,
                actual_receipt_rows=10, actual_padded_width=128, actual_forward_calls=2,
                receipt_forward_seconds=1., parsing_seconds=.1,
                records="Not parsed: selected answer services are irrelevant to resource arithmetic")
            persisted["receipt"], _ = artifact("receipts", i, receipt)
            receipt_checks[str(i)] = dict(passed=True, assigned_choices=20, actual_selected_forwards=10)
            timings.append(dict(call_index=i, generation_seconds=seconds, receipt_parsing_seconds=.1,
                receipt_forward_seconds=1., actual_receipt_rows=10, actual_receipt_width=128,
                artifact_bytes=persisted["artifact_bytes"] + persisted["receipt"]["artifact_bytes"],
                serialization_seconds=.01, receipt_serialization_seconds=.01))
        dispositions.append(dict(call=call, artifact=persisted))
    summary = dict(complete=True, engineering_passed=True, finished=380.,
        call_plan_sha256=forecast.digest(calls), call_dispositions=dispositions, timing_calls=timings)
    write(run / "engineering-summary.json", summary)
    audit = dict(manifest_sha256=manifest_sha, admission_sha256=forecast.digest(admission),
        environment_byte_sha256=hashlib.sha256((run / "environment.json").read_bytes()).hexdigest(),
        complete=True, passed=True, source_provenance_verified=True, runtime_verified=True,
        selected_receipts_verified=True, numeric=dict(numeric_engineering_passed=True),
        operational_coverage=dict(artifacts=artifacts), receipt_checks=receipt_checks)
    audit_path = run / "audit.json"
    write(audit_path, audit)
    external = project / "global-admission.json"
    write(external, admission)

    def rebind_lifecycle():
        inventory = {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in run.rglob("*") if p.is_file() and p.name not in ("audit.json", "lifecycle-audit.json")}
        write(run / "lifecycle-audit.json", dict(passed=True, manifest_sha256=manifest_sha,
            scientific_admission=False, raw_input_files=inventory,
            raw_input_files_sha256=forecast.digest(inventory),
            context_audit_byte_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
            external_files={"global-admission.json": hashlib.sha256(external.read_bytes()).hexdigest()}))
    rebind_lifecycle()
    return project, run, audit_path, rebind_lifecycle


def test_engineering_reader_reconciles_raw_times_sizes_and_full_lifecycle_inventory(engineering_files):
    project, run, audit_path, _ = engineering_files
    reader = forecast.Reader()
    plan, engineering = forecast.engineering_inputs(run, audit_path, reader, project)
    assert len(plan["assignments"]) == 8352
    assert engineering["all16_generation_seconds"] == 230
    assert engineering["collector_elapsed_seconds"] == 300
    assert engineering["startup_nonloading_seconds"] == 20
    assert engineering["lifecycle_overhead_seconds"] == 30
    assert len(engineering["timing_calls"]) == 6
    assert engineering["observed_artifact_bytes"] == sum(
        p.stat().st_size for folder in ("calls", "receipts") for p in (run / folder).glob("*"))
    assert str(run / "receipts/15.json") in reader.bindings
    assert str(run / "receipts/15.npz") in reader.bindings
    assert str(project / "global-admission.json") in reader.bindings


@pytest.mark.parametrize("mutation", ["receipt_after_audit", "audit_after_lifecycle", "omit_raw_receipt",
    "rehashed_summary_time", "rehashed_summary_bytes", "captured_pass_only", "failed_closure"])
def test_engineering_reader_rejects_changed_or_incomplete_decomposition(engineering_files, mutation):
    project, run, audit_path, rebind = engineering_files
    if mutation == "receipt_after_audit":
        (run / "receipts/10.npz").write_bytes(b"changed after audit")
    elif mutation == "audit_after_lifecycle":
        audit_path.write_text(audit_path.read_text() + " ")
    elif mutation == "omit_raw_receipt":
        path = run / "lifecycle-audit.json"
        value = json.loads(path.read_text())
        value["raw_input_files"].pop("receipts/10.json")
        value["raw_input_files_sha256"] = forecast.digest(value["raw_input_files"])
        write(path, value)
    elif mutation in ("rehashed_summary_time", "rehashed_summary_bytes"):
        path = run / "engineering-summary.json"
        value = json.loads(path.read_text())
        value["timing_calls"][0]["receipt_forward_seconds" if mutation.endswith("time") else "artifact_bytes"] = .01
        write(path, value)
        rebind()  # Even a consistent input inventory does not replace arithmetic reconciliation.
    elif mutation == "captured_pass_only":
        path = run / "receipts/10.json"
        value = json.loads(path.read_text())
        value["actual_forward_calls"] = 1
        write(path, value)
        rebind()
    else:
        path = run / "launch-outcome.json"
        value = json.loads(path.read_text())
        value["all_stopped_zero_tasks"] = False
        write(path, value)
        rebind()
    with pytest.raises(ValueError):
        forecast.engineering_inputs(run, audit_path, forecast.Reader(), project)
