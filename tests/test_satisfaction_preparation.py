"""Synthetic admission/preparation checks; no study selection, real freeze or model calls."""

from copy import deepcopy
from io import BytesIO
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from curiosity_study import satisfaction_capture as capture
from curiosity_study import satisfaction_screen as screen
from curiosity_study import sustained_engineering as engineering
from curiosity_study.artifacts import canonical, digest, read, write_once
from test_satisfaction_capture import fake_model
from test_satisfaction_screen import direction
from test_sustained_preparation import candidate_payloads

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import prepare_satisfaction_screen as prepare  # noqa: E402


@pytest.fixture(scope="module")
def synthetic():
    # These are reusable unit-test inputs with synthetic provenance, not a
    # selection from the project's actual executed-history inventory.
    plan = screen.assignment_plan(historical_exact_world_ids=[], direction_binding=direction())
    model = fake_model()
    return plan, capture.bind_preflight(plan, model), prepare.receipt_preflight(plan, model), model


def engineering_tree(root, monkeypatch):
    run = root / prepare.ENGINEERING
    manifest = dict(
        model=capture.MODEL,
        execution=screen.EXECUTION,
        schedule=screen.SCHEDULE,
        arithmetic=engineering.ARITHMETIC,
        engine_sha256=engineering.engine_hashes(),
        cap=4096,
    )
    write_once(run / "manifest.json", manifest)
    (run / "source.tar.gz").write_bytes(b"synthetic frozen archive")
    write_once(
        run / "freeze.json",
        dict(manifest_sha256=digest(manifest), archive_sha256=prepare.sha(run / "source.tar.gz")),
    )
    receipt = dict(complete=True, passed=True, integrity_passed=True, cap=4096)
    write_once(root / prepare.ENGINEERING_AUDIT, receipt)
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "audit_sustained_engineering",
        SimpleNamespace(audit=lambda p: calls.append(p) or deepcopy(receipt)),
    )
    return manifest, receipt, calls


def candidate_tree(root, monkeypatch):
    payloads, _ = candidate_payloads()
    import json

    values = {k: json.loads(v) for k, v in payloads.items() if k != "directions"}
    stream = BytesIO()
    unit = np.zeros((2, 4, 2, 5120), dtype=np.float64)
    unit[..., 0] = 1
    np.savez_compressed(
        stream,
        unit=unit,
        contrast_norm=np.full((2, 4, 2), prepare.LOCKED_NORM),
        sources=np.array(["curiosity_vs_indifference", "satisfaction_vs_neutral"]),
        layers=np.array([15, 31, 47, 57]),
        positions=np.array(["record", "final"]),
    )
    path = root / prepare.CANDIDATE_DOCUMENTS["directions"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(stream.getvalue())
    direction_sha = prepare.sha(path)
    analysis = values["analysis"]
    analysis["directions_sha256"] = direction_sha
    for item in analysis["candidates"]:
        item["a_raw_contrast_norm"] = prepare.LOCKED_NORM
    analysis_path = root / prepare.CANDIDATE_DOCUMENTS["analysis"]
    write_once(analysis_path, analysis)
    values["check"].update(
        analysis_sha256=digest(analysis),
        analysis_byte_sha256=prepare.sha(analysis_path),
        directions_byte_sha256=direction_sha,
    )
    for key, value in values.items():
        if key != "analysis":
            write_once(root / prepare.CANDIDATE_DOCUMENTS[key], value)
    monkeypatch.setattr(prepare, "LOCKED_CANDIDATE_MANIFEST", digest(values["manifest"]))
    monkeypatch.setattr(prepare, "LOCKED_DIRECTION_BYTES", direction_sha)
    return values, path


def test_original_direction_is_read_at_a15_record_with_exact_bytes_and_full_norm(tmp_path, monkeypatch):
    values, path = candidate_tree(tmp_path, monkeypatch)
    binding = prepare.locked_direction(tmp_path)
    assert binding["norm"] == prepare.LOCKED_NORM and binding["unit"] == [1.0] + [0.0] * 5119
    assert binding["candidate_source"]["analysis_sha256"] == digest(values["analysis"])
    assert binding["directions_byte_sha256"] == prepare.sha(path)
    path.write_bytes(path.read_bytes() + b"edited archive")
    with pytest.raises(ValueError, match="original locked"):
        prepare.locked_direction(tmp_path)


@pytest.mark.parametrize(
    "change", ["short_only", "failed", "unreproduced", "model", "schedule", "arithmetic", "engine", "archive"]
)
def test_full_engineering_admission_cannot_be_replaced_by_short_or_changed_evidence(
    tmp_path, monkeypatch, change
):
    manifest, receipt, calls = engineering_tree(tmp_path, monkeypatch)
    run = tmp_path / prepare.ENGINEERING
    if change == "short_only":
        receipt["cap"] = 128
    elif change == "failed":
        receipt["passed"] = False
    elif change == "unreproduced":
        monkeypatch.setitem(
            sys.modules,
            "audit_sustained_engineering",
            SimpleNamespace(audit=lambda _: {**receipt, "extra": True}),
        )
    elif change == "archive":
        (run / "source.tar.gz").write_bytes(b"modified")
    else:
        if change == "model":
            manifest["model"] = {**capture.MODEL, "revision": "changed"}
        elif change == "schedule":
            manifest["schedule"] = "prefill_only"
        elif change == "arithmetic":
            manifest["arithmetic"] = {**engineering.ARITHMETIC, "parameter_dtype": "float32"}
        else:
            manifest["engine_sha256"] = {**engineering.engine_hashes(), "state_backend.py": "changed"}
        (run / "manifest.json").write_text(canonical(manifest))
        freeze = read(run / "freeze.json")
        freeze["manifest_sha256"] = digest(manifest)
        (run / "freeze.json").write_text(canonical(freeze))
    if change in ("short_only", "failed"):
        (tmp_path / prepare.ENGINEERING_AUDIT).write_text(canonical(receipt))
    with pytest.raises(ValueError, match="Full-cap engineering"):
        prepare.engineering_certificate(tmp_path)
    assert not (tmp_path / "executions").exists()


def test_passing_engineering_is_recomputed_before_admission(tmp_path, monkeypatch):
    manifest, receipt, calls = engineering_tree(tmp_path, monkeypatch)
    binding = prepare.engineering_certificate(tmp_path)
    assert calls == [tmp_path / prepare.ENGINEERING]
    assert binding["manifest_sha256"] == digest(manifest) and binding["audit_sha256"] == digest(receipt)


def test_synthetic_tokenizer_preflight_covers_every_possible_receipt_without_inference(synthetic):
    plan, preflight, receipt, model = synthetic
    assert not hasattr(model.model, "forward")
    assert len(preflight["requests"]) == 288 and len(preflight["outer_request_bindings"]) == 1152
    assert receipt["synthetic_answer_only"] and not receipt["generated_model_outputs"]
    assert len(receipt["non_stop_bindings"]) == 432 and len(receipt["stop_checks"]) == 144
    assert {(r["base_trial_id"], r["selected_service"]) for r in receipt["non_stop_bindings"]} == {
        (a["base_trial_id"], service)
        for a in plan["base_assignments"]
        if a["kind"] == "choice"
        for service in ("report", "repeat", "pattern")
    }


def test_parent_and_three_children_preserve_all_rows_prompts_seed_and_frozen_analysis(synthetic, monkeypatch):
    plan, preflight, receipt, _ = synthetic
    monkeypatch.setattr(prepare, "source_hash", lambda: "synthetic-source")
    parent, children = prepare.build_manifests(plan, preflight, receipt, direction(), {"synthetic": True}, [])
    assert parent["analysis_metadata"] == prepare.ANALYSIS_METADATA
    assert len(parent["diagnostics"]) == 1152 and len(children) == 3
    assert [(child["execution_group"], len(child["diagnostics"])) for child in children] == [
        ("baseline_zero", 576),
        ("positive_half", 288),
        ("positive_full", 288),
    ]
    assert parent["budget"] == dict(
        workers=3, shared_deadline_seconds=12000, stage_ceiling_usd=55, paid_retries=0
    )
    assert [r for child in children for r in child["diagnostics"]] == parent["diagnostics"]
    for child in children:
        assert child["parent_manifest_sha256"] == digest(parent)
        assert len(capture.validate_manifest(child)) == len(child["diagnostics"])
        assert {row["arm"] for row in child["diagnostics"]} == set(capture.GROUPS[child["execution_group"]])
        assert child["generation_preflight"] == parent["generation_preflight"]
        assert child["screen_plan"] == parent["screen_plan"]
    children[0]["screen_plan"]["assignments"][0]["arm"] = "changed"
    assert parent["screen_plan"] == plan and children[1]["screen_plan"] == plan


def test_exposure_adapter_retains_actual24_worlds_and_both_panel_seeds(synthetic):
    plan = synthetic[0]
    target = prepare.exposure_target_plan(plan)
    assert target["selection"]["worlds"] == plan["selection"]["worlds"]
    assert {r["seed"] for r in target["selection"]["worlds"]} == {2026101436, 2026101437}
    assert target["assignments"] == [r["source_assignment"] for r in plan["base_assignments"]]
    assert target["selection_sha256"] == digest(target["selection"])
    target["selection"]["worlds"].clear()
    assert len(plan["selection"]["worlds"]) == 24


def preparation_pipeline(root, monkeypatch, synthetic, *, failure=None):
    plan, preflight, receipt_check, model = synthetic
    # Fixed prerequisite files are synthetic, and both the selector and freezer
    # are intercepted: this test never selects from real study history or freezes source.
    docs = [
        prepare.PLAN,
        prepare.ENGINEERING_AUDIT,
        *prepare.CANDIDATE_DOCUMENTS.values(),
        f"{prepare.ENGINEERING}/manifest.json",
        f"{prepare.ENGINEERING}/freeze.json",
        f"{prepare.ENGINEERING}/source.tar.gz",
    ]
    for name in docs:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"synthetic":true}' if path.suffix == ".json" else "Synthetic prerequisite")
    events = []
    history = dict(historical_exact_world_ids=[], inventory="synthetic-complete")

    def check_engineering(_):
        events.append("engineering")
        if failure == "engineering":
            raise ValueError("Synthetic failed admission")
        return dict(synthetic_engineering=True)

    def choose(**kwargs):
        events.append("selection")
        assert kwargs == dict(historical_exact_world_ids=[], direction_binding=direction())
        return deepcopy(plan)

    def exposure(_, plan=None):
        events.append("history" if plan is None else "exposure")
        bound_history = {**history, "changed": True} if plan is not None and failure == "history" else history
        return dict(
            _history=bound_history,
            exact_complete_world_criterion_passed=failure != "overlap",
            selected_worlds=[None] * 24,
            summary=[{"synthetic": True}],
        )

    def tokenizer():
        events.append("tokenization")
        if failure == "tokenization":
            raise ValueError("Synthetic token boundary failure")
        return model

    def freezer(project, run, documents):
        events.append("freeze")
        parent = read(run / "manifest.json")
        children = read(run / "children.json")
        assert len(children) == 3 and len(parent["diagnostics"]) == 1152
        assert [c["execution_group"] for c in children] == ["baseline_zero", "positive_half", "positive_full"]
        assert all(digest(read(Path(c["path"]) / "manifest.json")) == c["manifest_sha256"] for c in children)
        assert set(documents) == set(
            docs + [prepare.HISTORY, prepare.EXPOSURE, prepare.GENERATION, prepare.RECEIPTS]
        )
        return dict(source_sha256="synthetic-source")

    monkeypatch.setattr(prepare, "engineering_certificate", check_engineering)
    monkeypatch.setattr(prepare, "locked_direction", lambda _: direction())
    monkeypatch.setattr(prepare.screen, "assignment_plan", choose)
    monkeypatch.setattr(prepare, "tokenizer_model", tokenizer)
    monkeypatch.setattr(prepare, "source_hash", lambda: "synthetic-source")
    monkeypatch.setattr(prepare, "freeze", freezer)
    monkeypatch.setitem(
        sys.modules,
        "audit_voluntary_freshness",
        SimpleNamespace(audit=exposure, history_receipt=lambda x: x["_history"]),
    )
    return events


@pytest.mark.parametrize("failure", ["engineering", "history", "overlap", "tokenization"])
def test_failed_admission_history_exposure_or_tokenization_writes_no_screen_artifacts(
    tmp_path, monkeypatch, synthetic, failure
):
    events = preparation_pipeline(tmp_path, monkeypatch, synthetic, failure=failure)
    out = tmp_path / "runs/synthetic-screen"
    with pytest.raises(ValueError):
        prepare.prepare(out, root=tmp_path)
    assert "freeze" not in events and not out.exists()
    assert all(
        not (tmp_path / name).exists()
        for name in (prepare.HISTORY, prepare.EXPOSURE, prepare.GENERATION, prepare.RECEIPTS)
    )


def test_pipeline_checks_before_writes_binds_three_children_then_refuses_overwrite(
    tmp_path, monkeypatch, synthetic
):
    events = preparation_pipeline(tmp_path, monkeypatch, synthetic)
    out = tmp_path / "runs/synthetic-screen"
    result = prepare.prepare(out, root=tmp_path)
    assert result["assigned"] == 1152 and result["workers"] == 3 and not result["gpu_launched"]
    assert events == [
        "engineering",
        "history",
        "selection",
        "exposure",
        "tokenization",
        "selection",
        "selection",
        "selection",
        "freeze",
    ]
    before = list(events)
    with pytest.raises(ValueError, match="immutable"):
        prepare.prepare(out, root=tmp_path)
    assert events == before
    assert not (tmp_path / "executions").exists()
