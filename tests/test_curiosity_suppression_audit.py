"""Synthetic15-arm audit regressions; no scientific output reads or model calls."""

from copy import deepcopy
import hashlib
from io import BytesIO
from pathlib import Path
import shutil
import sys
import tarfile

import numpy as np
import pytest
import torch

from curiosity_study import curiosity_suppression_capture as capture
from curiosity_study import curiosity_suppression_screen as screen
from curiosity_study.artifacts import canonical, digest, read, write_once
from test_curiosity_suppression_capture import frozen as _source_frozen, install_fake_generation
from test_sustained_engineering import inputs, model_and_direction
from test_voluntary_capture import fake_forward

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import audit_curiosity_suppression_screen as audit  # noqa: E402
import prepare_curiosity_suppression_screen as prepare  # noqa: E402


@pytest.fixture(scope="module")
def source_frozen():
    # Test-only compatibility repair after the first unused admission freeze:
    # the preparer now explicitly compares the original288 tokenized requests.
    for original, _, model, source in _source_frozen.__wrapped__():
        source = deepcopy(source)
        source["generation_preflight"] = capture.legacy.bind_preflight(source["screen_plan"], model)
        source["generation_preflight_sha256"] = digest(source["generation_preflight"])
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(screen, "PRIOR_MANIFEST_SHA256", digest(source))
            plan = screen.assignment_plan(
                source_manifest=source, direction_bindings=original["direction_bindings"]
            )
            yield plan, capture.bind_preflight(plan, model), model, source


@pytest.fixture(scope="module")
def collected(tmp_path_factory, source_frozen):
    root = tmp_path_factory.mktemp("synthetic-curiosity-audit")
    original, _, model, source = source_frozen
    patch = pytest.MonkeyPatch()
    # Only structural source-direction admission is replaced in this synthetic
    # assembly fixture. Independent signed geometry and source arrays are below.
    patch.setattr(screen, "validate_directions", lambda bindings: deepcopy(bindings))
    bindings = deepcopy(original["direction_bindings"])
    for binding in bindings.values():
        binding["unit"] = [0.6 * binding["orientation"], 0.8 * binding["orientation"], 0.0, 0.0]
        binding["unit_sha256"] = digest(binding["unit"])
    plan = screen.assignment_plan(source_manifest=source, direction_bindings=bindings)
    preflight = capture.bind_preflight(plan, model)
    parent, children = prepare.build_manifests(plan, preflight, {}, source, {}, {}, {}, {}, {}, [])
    run = root / "parent"
    write_once(run / "manifest.json", parent)
    descriptors = []
    with pytest.MonkeyPatch.context() as execution:
        install_fake_generation(execution, (plan, preflight, model, source), [])
        original_collect = capture.collect_generation

        def collect(model, requests, patches, **kwargs):
            condition, hidden = original_collect(model, requests, patches, **kwargs)
            width = ((max(len(r["token_prefix"]) for r in requests) + 127) // 128) * 128
            condition.update(
                capture_phase=capture.CAPTURE_PLAN["source_phase"],
                unpadded_token_positions=[r["prefix_positions"] for r in requests],
                padded_token_positions=[
                    [width - len(r["token_prefix"]) + r["prefix_positions"][p] for p in ("record", "final")]
                    for r in requests
                ],
                prefill_width=width,
            )
            return condition, hidden

        execution.setattr(capture, "collect_generation", collect)
        execution.setattr(capture.receipts, "_logit_hashes", fake_forward([]))
        for child in children:
            path = root / child["execution_group"]
            capture.execute(path, child, model)
            descriptors.append(
                dict(path=str(path), manifest_sha256=digest(child), execution_group=child["execution_group"])
            )
    write_once(run / "children.json", descriptors)
    yield run, model
    patch.undo()


def bypass_separate_layers(monkeypatch, model):
    monkeypatch.setattr(
        audit, "provenance", lambda *a: {"engineering_environment": {"policy": {"test": True}}}
    )
    monkeypatch.setattr(audit, "token_model", lambda _: model)
    monkeypatch.setattr(audit, "environments", lambda *a: [dict(policy={"test": True})])
    monkeypatch.setattr(
        audit,
        "validate_generation",
        lambda evidence, *a: dict(
            timeline=evidence["condition"]["rows"] is not None,
            captures=evidence["activation_artifact"] is not None,
            hook_cleanup=evidence["condition"]["hooks_clean"],
            geometry=True,
            engagement=True,
        ),
    )


def run_audit(run, model):
    return audit.audit(run, tokenizer=model.tokenizer, eos_token_ids={9999}, vocabulary_size=10000)


def clone(collected, tmp_path):
    run, model = collected
    shutil.copytree(run.parent, tmp_path / "copy")
    target = tmp_path / "copy" / run.name
    descriptors = read(target / "children.json")
    for descriptor in descriptors:
        descriptor["path"] = str(target.parent / Path(descriptor["path"]).name)
    (target / "children.json").write_text(canonical(descriptors))
    return target, model


def test_full4320_and2160_dispositions_preserve_failed_readiness(collected, monkeypatch):
    run, model = collected
    bypass_separate_layers(monkeypatch, model)
    result = run_audit(run, model)
    assert result["complete"] and result["integrity_passed"]
    assert all(
        result[k]
        for k in (
            "sham_identity_passed",
            "engagement_passed",
            "receipts_passed",
            "cross_arm_prepatch_identity_passed",
            "worker_policies_equal",
        )
    )
    assert result["assigned"] == result["terminal_generations"] == len(result["outcomes"]) == 4320
    assert result["assigned_choices"] == result["terminal_receipt_slots"] == 2160
    assert [c["receipts"]["slots"] for c in result["children"]] == [795, 705, 660]
    assert len(result["readiness"]["diagnostic_cells"]) == 180
    assert len(result["readiness"]["choice_cells"]) == 90
    assert not result["readiness"]["passed"] and not result["passed"]
    checks = [r for c in result["children"] for r in c["generation"]["checks"]]
    assert len(checks) == 135 and sum(r["arm"] == "zero_sham" for r in checks) == 9
    assert sum("common_prefill_identity" in r for r in checks) == 126


@pytest.mark.parametrize("change", ["child_only", "missing_child", "budget", "balanced_rows"])
def test_parent_admission_rejects_partial_or_changed_design(collected, tmp_path, change):
    run, _ = clone(collected, tmp_path)
    parent = read(run / "manifest.json")
    descriptors = read(run / "children.json")
    if change == "child_only":
        run = Path(descriptors[0]["path"])
        parent = read(run / "manifest.json")
    elif change == "missing_child":
        (run / "children.json").write_text(canonical(descriptors[:-1]))
    elif change == "budget":
        parent["budget"]["paid_retries"] = True
    else:
        child_path = Path(descriptors[0]["path"]) / "manifest.json"
        child = read(child_path)
        child["diagnostics"][0], child["diagnostics"][32] = child["diagnostics"][32], child["diagnostics"][0]
        child["assignment_sha256"] = digest(child["diagnostics"])
        child_path.write_text(canonical(child))
        descriptors[0]["manifest_sha256"] = digest(child)
        (run / "children.json").write_text(canonical(descriptors))
    with pytest.raises(ValueError):
        audit.parent_children(run, parent)


def test_missing_tail_retains_all4320_and_exact_choice_denominator(collected, tmp_path, monkeypatch):
    run, model = clone(collected, tmp_path)
    child = Path(read(run / "children.json")[-1]["path"])
    # Worker2's final20 logical choices remain assigned after missing receipt.
    next((child / "receipt_cohorts").glob("*/0000640.json")).unlink()
    bypass_separate_layers(monkeypatch, model)
    result = run_audit(run, model)
    assert not result["complete"] and not result["integrity_passed"] and not result["receipts_passed"]
    assert len(result["outcomes"]) == 4320 and result["terminal_receipt_slots"] == 2140


def test_unavailable_terminal_failure_keeps_every_assigned_row(collected, tmp_path, monkeypatch):
    original, model = collected
    run = tmp_path / "parent"
    parent = read(original / "manifest.json")
    write_once(run / "manifest.json", parent)
    descriptors = read(original / "children.json")
    for descriptor in descriptors:
        child = read(Path(descriptor["path"]) / "manifest.json")
        path = tmp_path / descriptor["execution_group"]
        write_once(path / "manifest.json", child)
        descriptor["path"] = str(path)
    write_once(run / "children.json", descriptors)

    def unavailable(*args, **kwargs):
        raise RuntimeError("Synthetic unavailable observation")

    monkeypatch.setattr(capture, "collect_generation", unavailable)
    first = Path(descriptors[0]["path"])
    capture.execute(first, read(first / "manifest.json"), model)
    monkeypatch.setattr(
        audit, "provenance", lambda *a: {"engineering_environment": {"policy": {"test": True}}}
    )
    monkeypatch.setattr(audit, "token_model", lambda _: model)
    monkeypatch.setattr(audit, "environments", lambda *a: [dict(policy={"test": True})])
    result = run_audit(run, model)
    assert not result["complete"] and not result["integrity_passed"] and not result["passed"]
    assert len(result["outcomes"]) == 4320 and result["terminal_generations"] == 32
    assert sum(r["status"] == "execution_error" for r in result["outcomes"]) == 32
    assert sum(r["status"] == "missing" for r in result["outcomes"]) == 4288
    assert all(r["valid"] is None for r in result["outcomes"])


@pytest.mark.parametrize("change", ["attempt", "error", "source"])
def test_receipt_runtime_or_source_cannot_be_rebound(collected, tmp_path, monkeypatch, change):
    run, model = clone(collected, tmp_path)
    child = Path(read(run / "children.json")[0]["path"])
    path = next((child / "receipt_cohorts").glob("*/0000000.json"))
    if change == "attempt":
        target = path.parent.with_name("different-attempt")
        target.mkdir()
        path.rename(target / path.name)
    else:
        envelope = read(path)
        if change == "error":
            envelope["error"] = {"type": "SyntheticFailure"}
        else:
            envelope["records"][0]["source_record_sha256"] = "0" * 64
            envelope["records_sha256"] = digest(envelope["records"])
        path.write_text(canonical(envelope))
    bypass_separate_layers(monkeypatch, model)
    with pytest.raises(ValueError):
        run_audit(run, model)


def test_common_prefill_claim_checked_against_observed_arrays(collected, tmp_path, monkeypatch):
    run, model = clone(collected, tmp_path)
    child = Path(read(run / "children.json")[0]["path"])
    path = next((child / "generation_evidence").glob("*/0000032.json"))
    evidence = read(path)
    evidence["common_prefill_comparison"]["layer15_prepatch_captures_equal"] = False
    path.write_text(canonical(evidence))
    cohort_path = next((child / "cohorts").glob("*/0000032.json"))
    cohort = read(cohort_path)
    for row in cohort["records"]:
        row["generation_evidence"].update(sha256=audit.sha(path), byte_count=path.stat().st_size)
    cohort["sha256"] = digest(cohort["records"])
    cohort_path.write_text(canonical(cohort))
    bypass_separate_layers(monkeypatch, model)
    with pytest.raises(ValueError, match="common prefill"):
        run_audit(run, model)


def test_signed_bf16_geometry_checks_actual_applied_axis_and_decode_dose():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model, original = model_and_direction()
        direction = -original
        requests = inputs()
        patches = [
            dict(layer=15, position="record", mode="add", direction=direction, dose=screen.LOCKED_C_NORM)
            for _ in requests
        ]
        condition, hidden = capture.legacy.collect_generation(model, requests, patches, cap=3)
        assert condition["error"] is None
        for i, row in enumerate(condition["rows"]):
            assert audit.geometry(row["intervention_trace"], hidden[i, 0, 0], direction, screen.LOCKED_C_NORM)
            assert audit.engaged(row["intervention_trace"], screen.LOCKED_C_NORM)
        trace = deepcopy(condition["rows"][0]["intervention_trace"])
        assert not audit.geometry(trace, hidden[0, 0, 0], original, screen.LOCKED_C_NORM)
        trace["events"][-1]["requested_scalar_delta"] *= -1
        assert not audit.geometry(trace, hidden[0, 0, 0], direction, screen.LOCKED_C_NORM)
    finally:
        torch.set_num_threads(previous)


def test15arm_prepatch_identity_does_not_average_away_one_neighbor_mismatch(source_frozen):
    plan = source_frozen[0]
    vectors = {r["trial_id"]: np.zeros((1, 2, 4), np.float32) for r in plan["assignments"]}
    assert audit.cross_arm_identity(plan["assignments"], vectors)["passed"]
    one = next(r for r in plan["assignments"] if r["arm"] == "valence_negative")
    vectors[one["trial_id"]][0, 1, 0] = 1
    result = audit.cross_arm_identity(plan["assignments"], vectors)
    assert not result["passed"] and sum(not r["identical"] for r in result["rows"]) == 1


def synthetic_directions(monkeypatch):
    sources = ["curiosity_vs_indifference", *dict.fromkeys(name for _, name, _ in screen.NEIGHBORS)]
    units = np.zeros((len(sources), 4, 2, 5120), dtype=np.float64)
    norms = np.full((len(sources), 4, 2), 2.0)
    norms[0, 0, 0] = screen.LOCKED_C_NORM
    for i in range(len(sources)):
        units[i, :, :, i] = 1
    stream = BytesIO()
    np.savez(
        stream,
        sources=np.asarray(sources),
        unit=units,
        contrast_norm=norms,
        layers=np.asarray([15, 31, 47, 57]),
        positions=np.asarray(["record", "final"]),
    )
    payload = stream.getvalue()
    byte_sha = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(screen, "LOCKED_DIRECTION_BYTES", byte_sha)
    bindings = {}
    for arm, source, orientation in (
        ("zero_sham", "curiosity_vs_indifference", 1),
        (screen.PRIMARY, "curiosity_vs_indifference", -1),
        *screen.NEIGHBORS,
    ):
        i = sources.index(source)
        original = units[i, 0, 0]
        unit = (orientation * original).tolist()
        bindings[arm] = dict(
            source=source,
            corpus="A",
            layer=15,
            position="record",
            orientation=orientation,
            source_raw_norm=float(norms[i, 0, 0]),
            source_unit_sha256=digest(original.tolist()),
            unit=unit,
            unit_sha256=digest(unit),
            reference_raw_norm=screen.LOCKED_C_NORM,
            dose=0.0 if arm == "zero_sham" else screen.LOCKED_C_NORM,
            directions_byte_sha256=byte_sha,
        )
    for arm, seed in zip(screen.RANDOM_ARMS, screen.RANDOM_SEEDS, strict=True):
        raw = np.random.default_rng(seed).standard_normal(5120)
        unit = (raw / np.linalg.norm(raw)).tolist()
        bindings[arm] = dict(
            source="seeded_gaussian",
            corpus=None,
            layer=15,
            position="record",
            orientation=1,
            seed=seed,
            construction="numpy_default_rng_standard_normal_float64_unit_no_resampling",
            generator="PCG64",
            numpy_version=np.__version__,
            original_draw_byte_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
            original_draw_norm=float(np.linalg.norm(raw)),
            unit=unit,
            unit_sha256=digest(unit),
            reference_raw_norm=screen.LOCKED_C_NORM,
            dose=screen.LOCKED_C_NORM,
        )
    return bindings, payload


def test_all14_signed_axes_and_fixed_random_draws_reconstructed_independently(monkeypatch):
    bindings, payload = synthetic_directions(monkeypatch)
    assert audit.verify_directions({"direction_bindings": bindings}, payload) == bindings
    inventory = prepare.direction_inventory(bindings)
    audit.verify_inventory(bindings, inventory)
    assert inventory["signed_unit_cosines"][0][1] == -1


@pytest.mark.parametrize(
    "change", ["target_sign", "confidence_sign", "random_reoriented", "random_redraw", "fp32", "cosine"]
)
def test_rehashed_changed_signed_vectors_and_inventory_fail(monkeypatch, change):
    bindings, payload = synthetic_directions(monkeypatch)
    inventory = prepare.direction_inventory(bindings)
    if change in ("fp32", "cosine"):
        if change == "fp32":
            inventory["directions"][1]["effective_float32_byte_sha256"] = "0" * 64
        else:
            inventory["signed_unit_cosines"][0][1] = 1
        with pytest.raises(ValueError):
            audit.verify_inventory(bindings, inventory)
        return
    arm = {
        "target_sign": screen.PRIMARY,
        "confidence_sign": "confidence_positive",
        "random_reoriented": screen.RANDOM_ARMS[0],
        "random_redraw": screen.RANDOM_ARMS[0],
    }[change]
    if change == "random_redraw":
        raw = np.random.default_rng(999).standard_normal(5120)
        bindings[arm]["unit"] = (raw / np.linalg.norm(raw)).tolist()
    else:
        bindings[arm]["unit"] = [-x for x in bindings[arm]["unit"]]
    bindings[arm]["unit_sha256"] = digest(bindings[arm]["unit"])
    with pytest.raises(ValueError):
        audit.verify_directions({"direction_bindings": bindings}, payload)


@pytest.mark.parametrize("change", [None, "cost", "profile_bytes", "headroom"])
def test_forecast_recomputed_from_bound_completed_runtime_profiles(tmp_path, change):
    for suffix, cohorts in (("baseline-zero", 18), ("positive-half", 9), ("positive-full", 9)):
        write_once(
            tmp_path / f"runs/qwen32-satisfaction-screen-v1-{suffix}/cloud_profiles/attempt.json",
            dict(
                gpu="NVIDIA H200",
                assigned=32 * cohorts,
                completed=32 * cohorts,
                execution_seconds=1000.0,
                load_seconds=10.0,
            ),
        )
    forecast = prepare.resource_forecast(tmp_path)
    if change == "cost":
        forecast["estimated_combined_compute_usd_range"][1] /= 2
    elif change == "profile_bytes":
        path = tmp_path / forecast["source_profiles"][0]["path"]
        path.write_text(path.read_text() + " ")
    elif change == "headroom":
        for entry in forecast["source_profiles"]:
            entry["profile"]["execution_seconds"] = 20000.0
            path = tmp_path / entry["path"]
            path.write_text(canonical(entry["profile"]))
        forecast = prepare.resource_forecast(tmp_path)
    if change is None:
        audit.verify_forecast(tmp_path, forecast, audit.resources.CURIOSITY_SCREEN_BUDGET)
    else:
        with pytest.raises(ValueError):
            audit.verify_forecast(tmp_path, forecast, audit.resources.CURIOSITY_SCREEN_BUDGET)


@pytest.mark.parametrize("change", [None, "document", "validator"])
def test_frozen_evidence_and_shared_validator_bytes_are_bound(tmp_path, change):
    repository = Path(__file__).parents[1]
    payloads = {
        str(path.relative_to(repository)): path.read_bytes()
        for path in (repository / "src/curiosity_study").glob("*.py")
    }
    for filename in (
        "audit_satisfaction_screen.py",
        "audit_curiosity_suppression_screen.py",
        "analyze_curiosity_suppression_screen.py",
        "prepare_curiosity_suppression_screen.py",
        "prepare_satisfaction_screen.py",
        "prepare_voluntary_resolution.py",
        "audit_sustained_engineering.py",
    ):
        payloads[f"scripts/{filename}"] = (repository / "scripts" / filename).read_bytes()
    source_sha = digest(
        {
            name.removeprefix("src/curiosity_study/"): value.decode()
            for name, value in payloads.items()
            if name.startswith("src/")
        }
    )
    bindings = []
    for role in sorted(audit.ROLES):
        path = f"documents/{role}.json"
        value = {"synthetic_role": role}
        payload = canonical(value).encode()
        payloads[path] = payload
        bindings.append(
            dict(
                role=role,
                path=path,
                byte_sha256=hashlib.sha256(payload).hexdigest(),
                content_sha256=digest(value),
            )
        )
    manifest = dict(
        source_sha256=source_sha, bound_receipts=bindings, engine_sha256=audit.engineering.engine_hashes()
    )
    if change == "document":
        payloads[bindings[0]["path"]] += b" "
    elif change == "validator":
        payloads["scripts/audit_satisfaction_screen.py"] += b"\n# changed\n"
    archive_path = tmp_path / "source.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for name, payload in payloads.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, BytesIO(payload))
    frozen = dict(
        manifest_sha256=digest(manifest),
        source_sha256=source_sha,
        archive_sha256=audit.sha(archive_path),
        documents={binding["path"]: binding["byte_sha256"] for binding in bindings},
    )
    write_once(tmp_path / "freeze.json", frozen)
    if change is None:
        actual, documents, _ = audit.frozen_documents(tmp_path, manifest)
        assert actual == frozen and set(documents) == audit.ROLES
    else:
        with pytest.raises(ValueError):
            audit.frozen_documents(tmp_path, manifest)


def test_compact_prior_decision_matches_fixed_results_despite_analysis_prose():
    compact = dict(
        classification="readiness_failed",
        effect_threshold_met=False,
        positive_screen=False,
        primary_panel_differences=dict(panel_1=0.0, panel_2=0.0, pooled=0.0),
        primary_pooled_menu_differences={"0": 0.0, "1": 0.0},
        sensitivity_can_rescue=False,
    )
    expanded = dict(
        compact, rule="Frozen point thresholds", interpretation="No positive screen", stopping="No rescue"
    )
    assert compact != expanded and audit.prior_decision_agrees(compact, expanded)
    for field in ("positive_screen", "effect_threshold_met", "sensitivity_can_rescue"):
        changed = {**compact, field: True}
        assert not audit.prior_decision_agrees(changed, expanded)
    changed = deepcopy(compact)
    changed["primary_panel_differences"]["panel_1"] = 0.25
    assert not audit.prior_decision_agrees(changed, expanded)
    # Do not accept arbitrary subsets, added claims, or bool-to-number coercion.
    with pytest.raises(ValueError, match="six fixed"):
        audit.prior_decision_agrees({"classification": "readiness_failed"}, expanded)
    with pytest.raises(ValueError, match="six fixed"):
        audit.prior_decision_agrees({**compact, "extra_claim": True}, expanded)
    assert not audit.prior_decision_agrees({**compact, "positive_screen": 0}, expanded)
