"""Synthetic complete site workers and selected-only receipts; no real outcomes."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from curiosity_study import satisfaction_site_execution as execution
from curiosity_study import satisfaction_causal_sites as study
from curiosity_study import satisfaction_capture as old_capture
from curiosity_study import satisfaction_screen as old_screen
from curiosity_study import voluntary_resolution as voluntary
from curiosity_study.artifacts import RunStore, digest, read
from test_satisfaction_capture import fake_model
from test_satisfaction_screen import direction
from test_reserved_object_inventory import prepare, pure_history


def directions(site, arms, dose):
    result = {}
    names = {"zero_sham": ("satisfaction_vs_neutral", 1), study.PRIMARY: ("satisfaction_vs_neutral", 1),
             **{a: (s, sign) for a, s, sign in study.controls.NEIGHBORS},
             **{a: ("seeded_gaussian", 1) for a in study.controls.RANDOM_ARMS}}
    for index, arm in enumerate(arms[1:]):
        name, sign = names[arm]
        unit = np.zeros(5120)
        unit[0 if arm in ("zero_sham", study.PRIMARY) else 4 if arm.startswith("valence_") else index] = sign
        metadata = dict(corpus="A", source_raw_norm=2.0, directions_byte_sha256=study.LOCKED_DIRECTION_BYTES)
        if arm in study.controls.RANDOM_ARMS:
            seed = study.controls.RANDOM_SEEDS[study.controls.RANDOM_ARMS.index(arm)]
            raw = np.random.default_rng(seed).standard_normal(5120)
            unit = raw / np.linalg.norm(raw)
            metadata = dict(corpus=None, seed=seed, construction="numpy_default_rng_standard_normal_float64_unit_no_resampling")
        result[arm] = dict(**metadata, source=name, orientation=sign, layer=site, position="record",
                           dose=0.0 if arm == "zero_sham" else dose, unit=unit.tolist(), unit_sha256=digest(unit.tolist()))
    return result


@pytest.fixture(scope="module", params=["calibration", "validation"])
def frozen(request):
    """Use fictional exposure history and patch only NEW pure-module source pins."""
    patch = pytest.MonkeyPatch()
    source_plan = old_screen.assignment_plan(historical_exact_world_ids=[], direction_binding=direction())
    model = fake_model()
    source = dict(assay="satisfaction_screen", screen_plan=source_plan,
                  generation_preflight=old_capture.bind_preflight(source_plan, model))
    patch.setattr(study, "PRIOR_MANIFEST_SHA256", digest(source))
    patch.setattr(study, "CALIBRATION_BASE_SHA256", digest(source_plan["base_assignments"]))
    patch.setattr(study, "CALIBRATION_SELECTION_SHA256", digest(source_plan["selection"]))
    materials = prepare.build_reserved_plan(pure_history())
    medians = {"15": 100.0, "31": 150.0, "47": 200.0}
    reference = dict(formula=study.DOSE_FORMULA, original_dose=study.ORIGINAL_S_DOSE, base_branches=288,
                     source_screen_manifest_sha256=digest(source), capture_sha256="a" * 64,
                     median_record_norm=medians,
                     doses={str(s): study.ORIGINAL_S_DOSE * medians[str(s)] / medians["15"] for s in study.SITES})
    phase = request.param
    sites, arms = (study.SITES, study.CALIBRATION_ARMS) if phase == "calibration" else ((47,), study.VALIDATION_ARMS)
    inputs = source_plan if phase == "calibration" else materials
    plan = study.assignment_plan(phase=phase, base_assignments=inputs["base_assignments"], selection=inputs["selection"],
                                 direction_bindings={str(s): directions(s, arms, reference["doses"][str(s)]) for s in sites},
                                 dose_reference=reference, selected_site=None if phase == "calibration" else 47)
    preflight = execution.bind_preflight(plan, model)
    receipt = execution.bind_receipt_preflight(plan, model)
    yield dict(plan=plan, preflight=preflight, receipt=receipt, source=source, materials=materials, model=model)
    patch.undo()


def manifest(frozen, group="worker_0"):
    plan = frozen["plan"]
    rows = study.group_assignments(plan, group)
    certificate = dict(dose_reference=plan["dose_reference"], synthetic_setup_audit="a" * 64)
    result = dict(assay=study.ASSAY, task_revision=study.REVISION, site_phase=plan["phase"], model=execution.MODEL,
                  site_plan=plan, site_plan_sha256=digest(plan), source_screen_manifest=frozen["source"],
                  source_screen_manifest_sha256=digest(frozen["source"]), reserved_materials=frozen["materials"],
                  reserved_materials_sha256=digest(frozen["materials"]), setup_certificate=certificate,
                  setup_certificate_sha256=digest(certificate), execution_group=group,
                  parent_manifest_sha256="b" * 64, assignments=[], diagnostics=rows, assignment_sha256=digest(rows),
                  execution=study.EXECUTION, max_new_tokens=4096,
                  capture_plans={str(s): execution.capture.capture_plan(s) for s in plan["sites"]},
                  arithmetic=execution.engineering.ARITHMETIC, schedule=study.SCHEDULE,
                  receipt_grouping=execution.RECEIPT_GROUPING,
                  generation_preflight=frozen["preflight"], generation_preflight_sha256=digest(frozen["preflight"]),
                  receipt_preflight=frozen["receipt"], receipt_preflight_sha256=digest(frozen["receipt"]),
                  budget=execution.resources.POLICIES[plan["phase"]])
    if plan["phase"] == "validation":
        receipt = dict(selected_site=47, synthetic_calibration_analysis="c" * 64)
        result.update(selected_site=47, selection_receipt=receipt, selection_receipt_sha256=digest(receipt))
    return result


def install_generation(monkeypatch, frozen, calls, *, corrupt=None, unavailable=False):
    base = {row["base_trial_id"]: row for row in frozen["plan"]["base_assignments"]}

    def collect(model, requests, patches, *, layer, cap, deadline):
        calls.append(dict(layer=layer, ids=[r["source_trial_id"] for r in requests], patches=patches))
        outputs = []
        for request, patch in zip(requests, patches, strict=True):
            source = base[request["source_trial_id"]]["source_assignment"]
            if source["kind"] == "choice":
                service = "stop" if source["world_slot"] % 3 == 0 else "report"
                answer = next(k for k, value in voluntary.task(source)["action_mapping"].items() if value == service)
                raw = "Still thinking" if source["world_slot"] % 3 == 2 else f"Reasoning </think> {answer}."
            else:
                raw = f"Reasoning </think> {voluntary.prompt(source)[2]}."
            events = [] if patch is None else [dict(requested_scalar_delta=patch["dose"],
                                                    delta_norm=patch["dose"], projection_before=1.0,
                                                    projection_after=1.0 + patch["dose"])]
            outputs.append(dict(token_prefix_sha256=request["token_prefix_sha256"], hooks_remaining=0,
                                intervention=None if patch is None else dict(dose=patch["dose"]),
                                intervention_trace=dict(schedule=study.SCHEDULE, events=events),
                                completion=dict(raw=raw, generated_token_ids=[2, 9999], ended_with_eos=True,
                                                sampling=dict(seed=request["seed"], temperature=0.7, top_p=1.0,
                                                              max_new_tokens=4096))))
        hashes = [digest(r["token_prefix"]) for r in requests]
        if corrupt == "sham" and patches[0] is not None and patches[0]["dose"] == 0:
            hashes[-1] = "f" * 64
        frames = [dict(forward_index=0, prediction_ordinal=1, full_fp32_logits_sha256=hashes,
                       input_ids=[r["token_prefix"] for r in requests],
                       position_ids=[list(range(len(r["token_prefix"]))) for r in requests],
                       attention_mask_row_sha256=[digest([1] * len(r["token_prefix"])) for r in requests],
                       active_prediction_rows=[True] * 32)]
        hidden = np.arange(32 * 2 * 4, dtype=np.float32).reshape(32, 1, 2, 4) + layer
        after = hidden.copy()
        if patches[0] is not None:
            after[:, 0, 0, 0] += patches[0]["dose"]
        if corrupt == "prepatch" and patches[0] is not None:
            hidden[0, 0, 0, 0] += 1
        arrays = dict(hidden=hidden, prefill_after=after)
        return dict(rows=None if unavailable else outputs, forwards=frames, attempted_forwards=1,
                    hooks_clean=True, error=dict(type="SyntheticFailure", message="unavailable") if unavailable else None,
                    layer=layer, capture_plan=execution.capture.capture_plan(layer),
                    activation_dtypes=dict.fromkeys(("before", "after"),
                                                   "torch.float32" if corrupt == "dtype" else "torch.bfloat16")), arrays

    monkeypatch.setattr(execution.capture, "collect_generation", collect)
    monkeypatch.setattr(execution.engineering, "validate_condition", lambda condition, *a, **k: condition["rows"] is not None)
    monkeypatch.setattr(execution.capture, "validate_prefill_geometry",
                        lambda condition, arrays, patches, **k: condition["rows"] is not None and corrupt != "geometry")


def install_receipts(monkeypatch, calls, *, fail=False, corrupt=False):
    def forward(model, requests, ids, *, layer):
        calls.append(dict(layer=layer, ids=ids, requests=requests))
        if fail:
            raise RuntimeError("synthetic receipt failure")
        hidden = np.ones((len(requests), 1, 2, 4), dtype=np.float32) * layer
        measurements = [dict(token_prefix_sha256=r["token_prefix_sha256"], unpadded_input_ids=r["token_prefix"],
                             intervention=None, hooks_remaining=0) for r in requests]
        guard = dict(passed=not corrupt, layers=[layer], source_trial_ids=ids, prefill_width=128,
                     captured_array_sha256=execution.receipts.array_sha(hidden))
        return measurements, hidden, guard

    monkeypatch.setattr(execution.capture, "forward_receipts", forward)


def test_preflights_and_all_worker_material_bindings(frozen):
    plan = frozen["plan"]
    expected_base = 288 if plan["phase"] == "calibration" else 192
    assert len(frozen["preflight"]["requests"]) == expected_base
    assert len(frozen["preflight"]["outer_request_bindings"]) == (4032 if expected_base == 288 else 2880)
    assert len(frozen["receipt"]["non_stop_bindings"]) == expected_base * 3 // 2
    assert len(frozen["receipt"]["stop_checks"]) == expected_base // 2
    for group in study.GROUPS:
        rows = execution.validate_manifest(manifest(frozen, group))
        assert len(rows) == (1344 if expected_base == 288 else 960)
        assert {r["base_cohort"] % 3 for r in rows} == {study.GROUPS.index(group)}


@pytest.mark.parametrize("mutation", ["plan", "site", "receipt_grouping", "preflight_order", "request_seed",
                                      "reserved", "setup_dose", "receipt_case", "budget"])
def test_manifest_tampering_rejected(frozen, mutation):
    value = deepcopy(manifest(frozen))
    if mutation == "plan":
        value["site_plan"]["assignments"][0]["site"] = 15
        value["site_plan_sha256"] = digest(value["site_plan"])
    elif mutation == "site":
        value["capture_plans"][str(value["site_plan"]["sites"][0])]["layers"] = [15]
    elif mutation == "receipt_grouping":
        value["receipt_grouping"] = "mix_all_sites"
    elif mutation in ("preflight_order", "request_seed"):
        requests = value["generation_preflight"]["requests"]
        if mutation == "preflight_order":
            requests[0], requests[1] = requests[1], requests[0]
        else:
            requests[0]["request"]["seed"] += 1
        value["generation_preflight"]["requests_sha256"] = digest(requests)
        value["generation_preflight_sha256"] = digest(value["generation_preflight"])
    elif mutation == "reserved":
        value["reserved_materials"]["base_assignments"][0]["world_slot"] = (
            value["reserved_materials"]["base_assignments"][0]["world_slot"] + 1) % 12
        value["reserved_materials"]["base_assignment_sha256"] = digest(value["reserved_materials"]["base_assignments"])
        value["reserved_materials_sha256"] = digest(value["reserved_materials"])
    elif mutation == "setup_dose":
        value["setup_certificate"] = deepcopy(value["setup_certificate"])
        value["setup_certificate"]["dose_reference"]["doses"]["31"] += 1
        value["setup_certificate_sha256"] = digest(value["setup_certificate"])
    elif mutation == "receipt_case":
        value["receipt_preflight"]["non_stop_bindings"].pop()
        value["receipt_preflight_sha256"] = digest(value["receipt_preflight"])
    else:
        value["budget"]["stage_ceiling_usd"] += 1
    with pytest.raises((ValueError, KeyError)):
        execution.validate_manifest(value)


@pytest.mark.parametrize("group", study.GROUPS)
def test_complete_worker_sites_after_vectors_and_source_cohort_receipts(frozen, group, monkeypatch, tmp_path):
    value = manifest(frozen, group)
    calls, forwards = [], []
    install_generation(monkeypatch, frozen, calls)
    install_receipts(monkeypatch, forwards)
    records = execution.execute(tmp_path, value, frozen["model"])
    expected = len(value["diagnostics"])
    assert len(records) == expected and len(calls) == expected // 32
    assert all(r["engineering_passed"] for r in records.values())
    evidence = [read(path) for path in (tmp_path / "generation_evidence").glob("*/*.json")]
    assert sum(e["sham_comparison"] is not None for e in evidence) == (6 if expected == 1344 else 2)
    for e in evidence:
        with np.load(tmp_path / e["activation_artifact"]["path"], allow_pickle=False) as arrays:
            assert arrays["layers"].tolist() == [e["site"]]
            assert arrays["hidden"].shape == arrays["prefill_after"].shape == (32, 1, 2, 4)
    envelopes = sorted((read(path) for path in (tmp_path / "receipt_cohorts").glob("*/*.json")),
                       key=lambda row: row["generation_offset"])
    groups = list(execution.receipt_groups(value["diagnostics"]))
    assert len(envelopes) == len(groups) == expected // 32
    for envelope, assigned in zip(envelopes, groups, strict=True):
        assert envelope["source_trial_ids"] == [row["trial_id"] for row in assigned["choices"]]
        assert envelope["site"] == assigned["site"] and envelope["arm"] == assigned["arm"]
        assert envelope["choice_offset"] == assigned["choice_offset"]
        if envelope["engineering"] is not None:
            assert envelope["engineering"]["layers"] == [assigned["site"]]
        for row in envelope["records"]:
            assert row["source_record_sha256"] == digest(records[row["source_trial_id"]])
    assert sum(len(e["records"]) for e in envelopes) == sum(r["kind"] == "choice" for r in value["diagnostics"])
    assert {s["status"] for e in envelopes for s in e["records"]} == {"completed", "stop", "invalid_choice"}
    assert all(not (r["task_record"].get("simulator_outcome") or {}).get("receipt_model_pass_completed", False)
               for r in records.values())
    profile = read(next((tmp_path / "execution_profiles").glob("*.json")))
    assert profile["receipts_complete"] and profile["parent_generations_required"] in (4032, 2880)
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from collection_coverage import check_collection_coverage
    assert check_collection_coverage(tmp_path, expected_manifest_sha256=digest(value))["passed"]
    with pytest.raises(ValueError, match="cannot resume or retry"):
        execution.execute(tmp_path, value, frozen["model"])


@pytest.mark.parametrize("failure", ["sham", "prepatch", "geometry", "dtype"])
def test_measured_guard_failures_retained_without_dropping_rows(frozen, monkeypatch, tmp_path, failure):
    value = manifest(frozen)
    install_generation(monkeypatch, frozen, [], corrupt=failure)
    monkeypatch.setattr(execution, "execute_receipts", lambda *a, **k: False)
    records = execution.execute(tmp_path, value, frozen["model"])
    assert len(records) == len(value["diagnostics"])
    failed = [r for r in records.values() if not r["engineering_passed"]]
    assert failed and all(r["task_record"] is not None and r["failure"] == "generation_engineering_guard_failed" for r in failed)
    if failure == "dtype":
        evidence = [read(path) for path in (tmp_path / "generation_evidence").glob("*/*.json")]
        assert all(e["checks"]["prefill_geometry"] and not e["checks"]["bfloat16_prefill"] for e in evidence)


def test_generation_failure_retains32_rows_and_never_retries(frozen, monkeypatch, tmp_path):
    value, calls = manifest(frozen), []
    install_generation(monkeypatch, frozen, calls, unavailable=True)
    records = execution.execute(tmp_path, value, frozen["model"])
    assert len(records) == 32 and all(r["status"] == "execution_error" for r in records.values())
    with pytest.raises(ValueError, match="cannot resume or retry"):
        execution.execute(tmp_path, value, frozen["model"])
    assert len(calls) == 1 and not list((tmp_path / "receipt_cohorts").glob("*/*.json"))


@pytest.mark.parametrize("kind", ["exception", "identity"])
def test_receipt_failure_cannot_claim_delivery_and_observed_arrays_remain_bound(frozen, monkeypatch, tmp_path, kind):
    value = manifest(frozen)
    install_generation(monkeypatch, frozen, [])
    install_receipts(monkeypatch, [], fail=kind == "exception", corrupt=kind == "identity")
    records = execution.execute(tmp_path, value, frozen["model"])
    envelopes = [read(path) for path in (tmp_path / "receipt_cohorts").glob("*/*.json")]
    failed = next(e for e in envelopes if e["error"] is not None)
    assert (failed["observed_activation_artifact"] is not None) is (kind == "identity")
    for row in failed["records"]:
        if row["receipt_required"]:
            assert not row["receipt_model_pass_completed"] and row["measurement"] is None
            assert row["activation_artifact"] is None and row["status"] == "execution_error"
            assert digest(row["executed_simulator_outcome"]) == digest(records[row["source_trial_id"]]["task_record"]["simulator_outcome"])
    assert not read(next((tmp_path / "execution_profiles").glob("*.json")))["receipts_complete"]


def test_expired_deadline_invents_no_rows(frozen, monkeypatch, tmp_path):
    calls = []
    install_generation(monkeypatch, frozen, calls)
    assert execution.execute(tmp_path, manifest(frozen), frozen["model"], deadline=0) == {}
    assert not calls and not RunStore(tmp_path, manifest(frozen)).completed()
