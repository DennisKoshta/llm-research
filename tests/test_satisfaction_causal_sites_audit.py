"""Explicit-site audit regressions: synthetic materials and tiny CPU BF16 only."""

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_satisfaction_causal_sites as audit  # noqa: E402
from curiosity_study.artifacts import canonical, digest, read, write_once  # noqa: E402
from test_causal_site_engineering import inputs, model_and_direction  # noqa: E402
from test_satisfaction_site_execution import frozen as _source_frozen, manifest  # noqa: E402
from test_voluntary_capture import fake_forward  # noqa: E402


@pytest.fixture(scope="module", params=["calibration", "validation"])
def frozen(request):
    yield from _source_frozen.__wrapped__(request)


@pytest.fixture(scope="module", params=[31, 47])
def numeric(request):
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model, unit = model_and_direction()
        model.model.generation_config.eos_token_id = 127
        original = model.model.forward

        def bounded(*args, **kwargs):
            output = original(*args, **kwargs)
            logits = torch.full_like(output.logits, -1000)
            logits[:, -1, 127 if kwargs["input_ids"].shape[1] == 1 else 7] = 1000
            output.logits = logits
            return output

        model.model.forward = bounded
        requests, site, conditions = inputs(), request.param, {}
        for arm, dose in (("baseline", None), ("zero_sham", 0.0), ("satisfaction_positive", 0.5)):
            cohort = [
                dict(site=site, intervention=None if dose is None else dict(dose=dose)) for _ in requests
            ]
            patches = audit.execution.patches_for(cohort, unit)
            condition, arrays = audit.capture.collect_generation(
                model, requests, patches, layer=site, cap=4096
            )
            evidence = dict(condition=condition, eos_token_ids=[127], pad_token_id=0)
            conditions[arm] = dict(evidence=evidence, arrays=arrays, cohort=cohort)
        return dict(site=site, model=model, requests=requests, unit=unit, conditions=conditions)
    finally:
        torch.set_num_threads(old)


def validate(numeric, value):
    return audit.validate_generation(
        value["evidence"],
        value["cohort"],
        numeric["requests"],
        numeric["model"].tokenizer,
        {127},
        128,
        value["arrays"],
        numeric["unit"],
    )


def test_real_bf16_all32_prefill_vectors_sham_and_full_token_timeline(numeric):
    for value in numeric["conditions"].values():
        assert all(validate(numeric, value).values())
        assert value["arrays"]["hidden"].shape == (32, 1, 2, 16)
        assert len(value["evidence"]["condition"]["forwards"]) == 2
    baseline, sham = (
        numeric["conditions"][name]["evidence"]["condition"] for name in ("baseline", "zero_sham")
    )
    assert audit.engineering.identical_rows(baseline, sham, range(32))
    assert np.array_equal(
        numeric["conditions"]["baseline"]["arrays"]["hidden"],
        numeric["conditions"]["satisfaction_positive"]["arrays"]["hidden"],
    )


@pytest.mark.parametrize("mutation", ["after_record", "after_final", "event_layer", "mask", "dtype"])
def test_actual_geometry_and_timing_corruption_cannot_pass(numeric, mutation):
    value = deepcopy(numeric["conditions"]["satisfaction_positive"])
    condition = value["evidence"]["condition"]
    if mutation.startswith("after"):
        value["arrays"]["prefill_after"][31, 0, int(mutation == "after_final"), 0] += 1
    elif mutation == "event_layer":
        condition["rows"][31]["intervention_trace"]["events"][0]["layer"] = 15
    elif mutation == "mask":
        condition["forwards"][1]["attention_mask_row_sha256"][31] = "0" * 64
    else:
        condition["activation_dtypes"] = dict(before="torch.float32", after="torch.float32")
    assert not all(validate(numeric, value).values())


@pytest.mark.parametrize("mutation", ["vocabulary", "raw", "eos"])
def test_token_identity_config_mismatch_is_a_binding_error(numeric, mutation):
    value = deepcopy(numeric["conditions"]["baseline"])
    if mutation == "vocabulary":
        value["evidence"]["condition"]["forwards"][0]["vocabulary_size"] += 1
    elif mutation == "raw":
        value["evidence"]["condition"]["rows"][0]["completion"]["raw"] = "invented"
    else:
        value["evidence"]["eos_token_ids"] = []
    with pytest.raises(ValueError):
        validate(numeric, value)


def test_same_wrong_sham_tokens_do_not_hide_full_vocabulary_mismatch(numeric):
    baseline = deepcopy(numeric["conditions"]["baseline"]["evidence"]["condition"])
    sham = deepcopy(numeric["conditions"]["zero_sham"]["evidence"]["condition"])
    sham["forwards"][1]["full_fp32_logits_sha256"][31] = "0" * 64
    assert not audit.engineering.identical_rows(baseline, sham, range(32))


def parent_files(tmp_path, frozen):
    from analyze_satisfaction_causal_sites import ANALYSIS_METADATA

    parent = deepcopy(manifest(frozen))
    parent.pop("execution_group")
    parent.pop("parent_manifest_sha256")
    parent.update(
        phase="development_pilot",
        inference_scope="development",
        execution_gpu="H200",
        source_sha256="a" * 64,
        diagnostics=frozen["plan"]["assignments"],
        assignment_sha256=digest(frozen["plan"]["assignments"]),
        analysis_metadata=ANALYSIS_METADATA,
    )
    root = tmp_path / "parent"
    write_once(root / "manifest.json", parent)
    children = []
    for group in audit.study.GROUPS:
        child = deepcopy(parent)
        child.update(
            execution_group=group,
            parent_manifest_sha256=digest(parent),
            diagnostics=audit.study.group_assignments(frozen["plan"], group),
        )
        child["assignment_sha256"] = digest(child["diagnostics"])
        path = tmp_path / group
        write_once(path / "manifest.json", child)
        children.append(dict(path=str(path), execution_group=group, manifest_sha256=digest(child)))
    write_once(root / "children.json", children)
    return root, parent


def test_parent_reconciles_exact_three_worker_full_grid(tmp_path, frozen):
    root, parent = parent_files(tmp_path, frozen)
    children = audit.parent_children(root, parent)
    assert len(children) == 3
    assert sum(len(m["diagnostics"]) for _, m in children) == len(parent["diagnostics"])
    descriptors = read(root / "children.json")
    (root / "children.json").write_text(__import__("json").dumps(descriptors[:-1]))
    with pytest.raises(ValueError):
        audit.parent_children(root, parent)


def test_completely_unobserved_parent_keeps_every_assigned_denominator(tmp_path, frozen, monkeypatch):
    root, parent = parent_files(tmp_path, frozen)
    monkeypatch.setattr(audit, "provenance", lambda *a: dict(setup_runtime_policy={}))
    monkeypatch.setattr(
        audit,
        "audit_generations",
        lambda *a: dict(
            records={},
            prepatch={},
            complete=False,
            checks=[],
            cohorts=[],
            evidence=[],
            record_bindings=[],
            capture_artifacts=[],
        ),
    )
    result = audit.audit(
        root, tokenizer=frozen["model"].tokenizer, eos_token_ids={9999}, vocabulary_size=10000
    )
    assert result["assigned"] == len(parent["diagnostics"]) == len(result["outcomes"])
    assert all(r["status"] == "missing" and r["valid"] is None for r in result["outcomes"])
    assert not result["complete"] and not result["integrity_passed"] and not result["passed"]
    assert all(not v["readiness"]["passed"] for v in result["site_checks"].values())


def ideal_outcomes(plan):
    result = []
    for assignment in plan["assignments"]:
        row = audit.compact(assignment, None)
        row.update(
            status="completed",
            valid=True,
            correct=assignment["kind"] != "choice",
            thinking_closed=True,
            closed=True,
            cap_hit=False,
        )
        result.append(row)
    return result


def test_readiness_uses_calibration_panels_but_whole_reserved16(frozen):
    plan = frozen["plan"]
    rows = ideal_outcomes(plan)
    site = plan["sites"][0]
    good = audit.readiness(plan, rows, site)
    assert good["passed"]
    expected = 12 if plan["phase"] == "calibration" else 16
    assert {r["assigned"] for r in good["diagnostic_cells"]} == {expected}
    row = next(r for r in rows if r["site"] == site and r["kind"] == "choice")
    row.update(status="missing", valid=None, correct=None, cap_hit=None)
    assert not audit.readiness(plan, rows, site)["passed"]
    row.update(status="completed", valid=True, cap_hit=True)
    assert not audit.readiness(plan, rows, site)["passed"]
    rows.pop()
    with pytest.raises(ValueError):
        audit.readiness(plan, rows, plan["assignments"][-1]["site"])


def test_cross_arm_identity_is_site_specific_and_never_fills_missing(frozen):
    plan = frozen["plan"]
    vectors = {a["trial_id"]: np.full((1, 2, 4), a["site"], dtype=np.float32) for a in plan["assignments"]}
    for site in plan["sites"]:
        assert audit.cross_arm_identity(plan, vectors, site)["passed"]
    assignment = plan["assignments"][0]
    vectors.pop(assignment["trial_id"])
    assert not audit.cross_arm_identity(plan, vectors, assignment["site"])["passed"]


@pytest.fixture
def receipts_fixture(tmp_path, frozen, monkeypatch):
    monkeypatch.setattr(
        frozen["model"].tokenizer,
        "apply_chat_template",
        lambda messages, **kwargs: frozen["model"].encode(
            messages, enable_thinking=kwargs.get("enable_thinking", False)
        )[0],
        raising=False,
    )
    value = deepcopy(manifest(frozen))
    value["diagnostics"] = value["diagnostics"][:32]
    value["assignment_sha256"] = digest(value["diagnostics"])
    for bindings in value["site_plan"]["direction_bindings"].values():
        for binding in bindings.values():
            binding["unit"] = [1.0, 0.0, 0.0, 0.0]
    root, attempt = tmp_path / "worker", "one-attempt"
    sources = {}
    for index, assignment in enumerate(value["diagnostics"]):
        source = assignment["source_assignment"]
        if source["kind"] == "choice":
            service = "stop" if index % 3 == 0 else "report"
            label = next(k for k, v in audit.voluntary.task(source)["action_mapping"].items() if v == service)
        else:
            label = audit.voluntary.prompt(source)[2]
        raw = f"Reasoning </think> {label}."
        inner = audit.voluntary.record_from_thinking_inference(
            source,
            dict(raw=raw, generated_token_ids=frozen["model"].tokenizer.encode(raw), ended_with_eos=True),
        )
        sources[assignment["trial_id"]] = dict(
            trial_id=assignment["trial_id"],
            assignment=assignment,
            kind=assignment["kind"],
            task_record=inner,
            status="completed",
            terminal=True,
        )
    # Receipt-only tests use the real source task and render/selected-payload helper;
    # generation/timeline and prefill arithmetic have independent tests above.
    sources = json.loads(canonical(sources))
    write_once(root / "cohorts" / attempt / "0000000.json", {"receipt_test_generation_attempt": True})
    monkeypatch.setattr(audit.receipts, "_logit_hashes", fake_forward([]))
    assert audit.execution.execute_receipts(root, value, frozen["model"], sources, attempt)
    return root, value, sources, frozen["model"].tokenizer


def test_receipts_use_actual_site_and_generation_cohort_choice_subset(receipts_fixture):
    root, value, sources, tokenizer = receipts_fixture
    result = audit.audit_receipts(root, value, sources, tokenizer)
    assert result["complete"] and result["passed"]
    assert result["slots"] == sum(a["kind"] == "choice" for a in value["diagnostics"])
    assert result["slots"] != 32
    envelope = read(next((root / "receipt_cohorts").glob("*/*.json")))
    assert envelope["engineering"]["layers"] == [value["diagnostics"][0]["site"]]


@pytest.mark.parametrize("mutation", ["site", "attempt", "payload", "logits", "error", "slot"])
def test_receipt_tampering_fails_without_reclassifying_delivery(receipts_fixture, mutation):
    root, value, sources, tokenizer = receipts_fixture
    path = next((root / "receipt_cohorts").glob("*/*.json"))
    envelope = read(path)
    if mutation == "attempt":
        dest = path.parent.parent / "resumed" / path.name
        dest.parent.mkdir()
        shutil.move(path, dest)
    else:
        if mutation == "site":
            envelope["engineering"]["layers"] = [15]
        elif mutation == "payload":
            row = next(r for r in envelope["records"] if r["receipt_required"])
            row["receipt_input"]["messages"][-1]["content"] += " Extra unselected payload"
        elif mutation == "logits":
            envelope["engineering"]["captured_full_initial_logits_sha256"][-1] = "f" * 64
        elif mutation == "error":
            envelope["error"] = dict(type="RuntimeError", message="failed but claimed success")
        else:
            envelope["records"].pop()
        envelope["records_sha256"] = digest(envelope["records"])
        path.write_text(__import__("json").dumps(envelope))
    with pytest.raises((ValueError, KeyError)):
        audit.audit_receipts(root, value, sources, tokenizer)


def test_real_producer_terminal_generation_exception_retains_failed_and_missing_slots(
    tmp_path, frozen, monkeypatch
):
    value = manifest(frozen)

    def unavailable(*args, **kwargs):
        raise RuntimeError("Synthetic inference exception before observations")

    monkeypatch.setattr(audit.capture, "collect_generation", unavailable)
    monkeypatch.setattr(
        frozen["model"].tokenizer,
        "apply_chat_template",
        lambda messages, **kwargs: frozen["model"].encode(
            messages, enable_thinking=kwargs.get("enable_thinking", False)
        )[0],
        raising=False,
    )
    produced = audit.execution.execute(tmp_path, value, frozen["model"])
    assert len(produced) == 32
    checked = audit.audit_generations(tmp_path, value, frozen["model"].tokenizer, {9999}, 10000)
    assert not checked["complete"]
    assert len(checked["records"]) == 32
    assert not checked["checks"][0]["timeline"]
    assert not checked["checks"][0]["captures"]
    outcomes = [audit.compact(a, checked["records"].get(a["trial_id"])) for a in value["diagnostics"]]
    assert len(outcomes) == len(value["diagnostics"])
    assert sum(r["status"] == "execution_error" for r in outcomes) == 32
    assert sum(r["status"] == "missing" for r in outcomes) == len(outcomes) - 32
    assert all(r["valid"] is None for r in outcomes)
    receipts = audit.audit_receipts(tmp_path, value, checked["records"], frozen["model"].tokenizer)
    assert not receipts["complete"] and not receipts["passed"]
    assert receipts["slots"] == 0


def test_receipt_exception_is_terminal_but_never_successful(receipts_fixture, frozen, tmp_path, monkeypatch):
    _, value, sources, tokenizer = receipts_fixture
    root, attempt = tmp_path / "failed-worker", "one-attempt"
    write_once(root / "cohorts" / attempt / "0000000.json", {"receipt_test_generation_attempt": True})

    def unavailable(*args, **kwargs):
        raise RuntimeError("Synthetic selected-payload forward exception")

    monkeypatch.setattr(audit.capture, "forward_receipts", unavailable)
    assert not audit.execution.execute_receipts(root, value, frozen["model"], sources, attempt)
    result = audit.audit_receipts(root, value, sources, tokenizer)
    assert result["complete"] and not result["passed"]
    assert result["slots"] == sum(a["kind"] == "choice" for a in value["diagnostics"])
    assert result["status_counts"]["execution_error"] > 0
    envelope = read(next((root / "receipt_cohorts").glob("*/*.json")))
    assert envelope["observed_activation_artifact"] is None
    assert all(
        r["receipt_model_pass_completed"] is False
        and r["executed_simulator_outcome"]["observation_ledger"][-1]["model_pass_completed"] is False
        for r in envelope["records"]
        if r["receipt_required"]
    )


def exposure_materials():
    history = dict(
        historical_registry=[
            dict(run="runs/old", manifest_sha256="a" * 64, execution_evidence="persisted_records")
        ]
    )
    bundle = dict(history=history)
    setup = dict(run="runs/setup", manifest_sha256="b" * 64)
    registry = [
        *history["historical_registry"],
        dict(
            run=setup["run"], manifest_sha256=setup["manifest_sha256"], execution_evidence="persisted_records"
        ),
    ]
    receipt = dict(
        passed=True,
        scientific_admission=False,
        closing_inventory_unchanged=True,
        initial_history_sha256=digest(history),
        reserved_materials_sha256=digest(bundle),
        registry=registry,
        registry_sha256=digest(registry),
        allowed_known_world_runs={setup["run"]: setup["manifest_sha256"]},
        newly_executed_worlds_outside_initial_history=0,
        reserved_exact_overlap_count=0,
        source_subset=audit.study.CALIBRATION_BASE_SHA256,
    )
    value = dict(
        site_phase="calibration",
        reserved_materials=bundle,
        setup_certificate=setup,
        exposure_receipt_sha256=digest(receipt),
    )
    return value, receipt


def test_frozen_incremental_exposure_allows_only_bound_known_world_work(tmp_path):
    value, receipt = exposure_materials()
    assert audit.verify_exposure(value, receipt, tmp_path) == receipt


@pytest.mark.parametrize("mutation", ["old_binding", "allowed_hash", "new_execution", "duplicate", "closing"])
def test_incremental_exposure_cannot_relabel_new_execution_or_change_old_inventory(tmp_path, mutation):
    value, receipt = exposure_materials()
    if mutation == "old_binding":
        receipt["registry"][0]["manifest_sha256"] = "c" * 64
    elif mutation == "allowed_hash":
        receipt["registry"][1]["manifest_sha256"] = "c" * 64
    elif mutation == "new_execution":
        receipt["registry"].append(
            dict(run="runs/other", manifest_sha256="c" * 64, execution_evidence="persisted_records")
        )
    elif mutation == "duplicate":
        receipt["registry"].append(deepcopy(receipt["registry"][1]))
    else:
        receipt["closing_inventory_unchanged"] = False
    receipt["registry_sha256"] = digest(receipt["registry"])
    value["exposure_receipt_sha256"] = digest(receipt)
    with pytest.raises(ValueError):
        audit.verify_exposure(value, receipt, tmp_path)


def selection_materials(tmp_path):
    import check_satisfaction_site_decision as checker
    import test_satisfaction_causal_sites_analysis as tables

    plan, outcomes = tables.boundary()
    plan["dose_reference"] = {"31": 1.0}
    calibration = dict(
        assay=audit.study.ASSAY,
        site_phase="calibration",
        source_sha256="a" * 64,
        reserved_materials={"fixed_worlds": [1, 2]},
        setup_certificate={"fixed_short_audit": "a" * 64},
        site_plan=plan,
    )
    top, per_site = tables.checks(plan)
    audited = dict(parent_manifest_sha256=digest(calibration), **top, site_checks=per_site, outcomes=outcomes)
    recomputed = checker.recompute(plan, outcomes, checks=top, site_checks=per_site)
    analysis = dict(
        parent_manifest_sha256=digest(calibration),
        audit_sha256=digest(audited),
        checks=top,
        site_checks=per_site,
        **recomputed,
    )
    analysis["decision"]["interpretation"] = "Descriptive prose is retained beyond the scientific projection."
    run = tmp_path / "runs" / "calibration"
    write_once(run / "manifest.json", calibration)
    (run / "source.tar.gz").write_bytes(b"synthetic archive binding, not an executed model")
    frozen = dict(
        manifest_sha256=digest(calibration),
        source_sha256=calibration["source_sha256"],
        archive_sha256=audit.sha(run / "source.tar.gz"),
    )
    write_once(run / "freeze.json", frozen)
    write_once(tmp_path / "audit.json", audited)
    write_once(tmp_path / "analysis.json", analysis)
    checked = checker.check(run, tmp_path / "analysis.json", tmp_path / "audit.json")
    write_once(tmp_path / "check.json", checked)
    receipt = dict(
        calibration_run="runs/calibration",
        manifest_path="runs/calibration/manifest.json",
        manifest_sha256=digest(calibration),
        source_sha256=calibration["source_sha256"],
        freeze_path="runs/calibration/freeze.json",
        freeze_sha256=digest(frozen),
        archive_path="runs/calibration/source.tar.gz",
        archive_byte_sha256=frozen["archive_sha256"],
        audit_path="audit.json",
        audit_sha256=digest(audited),
        audit_byte_sha256=audit.sha(tmp_path / "audit.json"),
        analysis_path="analysis.json",
        analysis_sha256=digest(analysis),
        analysis_byte_sha256=audit.sha(tmp_path / "analysis.json"),
        independent_check_path="check.json",
        independent_check_sha256=digest(checked),
        independent_check_byte_sha256=audit.sha(tmp_path / "check.json"),
        selected_site=31,
        decision=analysis["decision"],
    )
    return dict(
        calibration, selected_site=31, selection_receipt=receipt, selection_receipt_sha256=digest(receipt)
    )


def test_validation_requires_complete_same_site_independently_checked_selection(tmp_path):
    value = selection_materials(tmp_path)
    assert audit.verify_selection(tmp_path, value) == value["selection_receipt"]


@pytest.mark.parametrize(
    "mutation",
    [
        "incomplete",
        "unchecked",
        "selected_site",
        "reserved",
        "dose",
        "archive_bytes",
        "fully_rehashed_decision",
    ],
)
def test_validation_cannot_rescue_incomplete_or_changed_calibration(tmp_path, mutation):
    value = selection_materials(tmp_path)
    if mutation == "incomplete":
        old = read(tmp_path / "audit.json")
        old["complete"] = False
        (tmp_path / "audit.json").write_text(canonical(old))
        value["selection_receipt"]["audit_sha256"] = digest(old)
    elif mutation == "unchecked":
        old = read(tmp_path / "check.json")
        old["passed"] = False
        (tmp_path / "check.json").write_text(canonical(old))
        value["selection_receipt"]["independent_check_sha256"] = digest(old)
    elif mutation == "selected_site":
        value["selected_site"] = value["selection_receipt"]["selected_site"] = 47
    elif mutation == "reserved":
        value["reserved_materials"]["fixed_worlds"] = [1, 3]
    elif mutation == "dose":
        value["site_plan"]["dose_reference"]["31"] = 2.0
    elif mutation == "archive_bytes":
        (tmp_path / "runs/calibration/source.tar.gz").write_bytes(b"a different source package")
    else:
        analysis, checked = read(tmp_path / "analysis.json"), read(tmp_path / "check.json")
        analysis["decision"]["selected_site"] = 47
        (tmp_path / "analysis.json").write_text(canonical(analysis))
        checked["analysis_sha256"], checked["analysis_byte_sha256"] = (
            digest(analysis),
            audit.sha(tmp_path / "analysis.json"),
        )
        checked["decision"] = analysis["decision"]
        checked["recomputed_decision"]["selected_site"] = 47
        (tmp_path / "check.json").write_text(canonical(checked))
        value["selected_site"] = 47
        value["selection_receipt"].update(
            analysis_sha256=digest(analysis),
            analysis_byte_sha256=audit.sha(tmp_path / "analysis.json"),
            independent_check_sha256=digest(checked),
            independent_check_byte_sha256=audit.sha(tmp_path / "check.json"),
            selected_site=47,
            decision=analysis["decision"],
        )
    value["selection_receipt_sha256"] = digest(value["selection_receipt"])
    with pytest.raises(
        ValueError, match="Decision mismatch" if mutation == "fully_rehashed_decision" else None
    ):
        audit.verify_selection(tmp_path, value)


def frozen_code_archive(tmp_path, *, alter_oracle=False):
    """Synthetic archive only; no data preparation or execution freeze is invoked."""
    project = Path(__file__).resolve().parents[1]
    paths = list((project / "src" / "curiosity_study").rglob("*.py")) + list(
        (project / "scripts").glob("*.py")
    )
    payloads = {str(path.relative_to(project)): path.read_bytes() for path in paths}
    if alter_oracle:
        payloads["src/curiosity_study/information_state.py"] += b"\n# altered oracle implementation\n"
    name, data = "reports/synthetic-evidence.json", (canonical({"fixed": True}) + "\n").encode()
    payloads[name] = data
    manifest_value = dict(
        source_sha256=digest(
            {
                name.removeprefix("src/curiosity_study/"): value.decode()
                for name, value in payloads.items()
                if name.startswith("src/curiosity_study/")
            }
        ),
        bound_receipts=[
            dict(
                role="synthetic",
                path=name,
                byte_sha256=hashlib.sha256(data).hexdigest(),
                content_sha256=digest({"fixed": True}),
            )
        ],
    )
    with tarfile.open(tmp_path / "source.tar.gz", "w:gz") as archive:
        for name, payload in payloads.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    write_once(
        tmp_path / "freeze.json",
        dict(
            manifest_sha256=digest(manifest_value),
            source_sha256=manifest_value["source_sha256"],
            archive_sha256=audit.sha(tmp_path / "source.tar.gz"),
            documents={
                name: hashlib.sha256(data).hexdigest() for name in ["reports/synthetic-evidence.json"]
            },
        ),
    )
    return manifest_value


def test_frozen_code_and_explicit_document_bytes_reconcile(tmp_path):
    value = frozen_code_archive(tmp_path)
    _, _, roles = audit.frozen_documents(tmp_path, value)
    assert roles["synthetic"]["value"] == {"fixed": True}


def test_self_consistent_archive_cannot_change_the_imported_task_oracle(tmp_path):
    value = frozen_code_archive(tmp_path, alter_oracle=True)
    with pytest.raises(ValueError, match="validator differs"):
        audit.frozen_documents(tmp_path, value)
