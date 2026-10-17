"""Synthetic provenance and fail-before-publication tests; no real preparation or model calls."""

from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from curiosity_study import curiosity_suppression_capture as capture
from curiosity_study import curiosity_suppression_screen as screen
from curiosity_study.artifacts import canonical, digest, read, write_once
from test_satisfaction_capture import fake_model
from test_sustained_preparation import candidate_payloads

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import prepare_curiosity_suppression_screen as p  # noqa: E402


@pytest.fixture
def source_inputs(monkeypatch):
    from test_curiosity_suppression_screen import inputs

    return inputs.__wrapped__(monkeypatch)


def candidate_tree(root, monkeypatch):
    payloads, _ = candidate_payloads()
    values = {k: json.loads(v) for k, v in payloads.items() if k != "directions"}
    sources = list(
        dict.fromkeys(
            ["curiosity_vs_indifference", "satisfaction_vs_neutral", *[s for _, s, _ in screen.NEIGHBORS]]
        )
    )
    unit = np.zeros((len(sources), 4, 2, 5120), dtype=np.float64)
    norms = np.empty((len(sources), 4, 2), dtype=np.float64)
    for s in range(len(sources)):
        for layer in range(4):
            for position in range(2):
                unit[s, layer, position, 8 * s + 2 * layer + position] = 1
                norms[s, layer, position] = s + layer + position + 1
    norms[0, 0, 0] = screen.LOCKED_C_NORM
    stream = BytesIO()
    np.savez_compressed(
        stream,
        sources=np.array(sources),
        layers=np.array([15, 31, 47, 57]),
        positions=np.array(["record", "final"]),
        unit=unit,
        contrast_norm=norms,
    )
    path = root / p.CANDIDATE_DOCUMENTS["directions"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(stream.getvalue())
    values["analysis"]["directions_sha256"] = p.sha(path)
    analysis_path = root / p.CANDIDATE_DOCUMENTS["analysis"]
    write_once(analysis_path, values["analysis"])
    values["check"].update(
        analysis_sha256=digest(values["analysis"]),
        analysis_byte_sha256=p.sha(analysis_path),
        directions_byte_sha256=p.sha(path),
    )
    for key, value in values.items():
        if key != "analysis":
            write_once(root / p.CANDIDATE_DOCUMENTS[key], value)
    monkeypatch.setattr(p, "LOCKED_CANDIDATE_MANIFEST", digest(values["manifest"]))
    monkeypatch.setattr(screen, "LOCKED_DIRECTION_BYTES", p.sha(path))
    return path, sources, unit


def test_locked_axes_keep_source_sites_signed_units_and_exact_unselected_random_draws(tmp_path, monkeypatch):
    path, sources, units = candidate_tree(tmp_path, monkeypatch)
    bindings, candidate = p.locked_directions(tmp_path)
    assert set(bindings) == set(screen.ARMS) - {"baseline"}
    specs = (
        ("zero_sham", "curiosity_vs_indifference", 1),
        (screen.PRIMARY, "curiosity_vs_indifference", -1),
        *screen.NEIGHBORS,
    )
    for arm, source, sign in specs:
        original = units[sources.index(source), 0, 0]
        np.testing.assert_array_equal(bindings[arm]["unit"], sign * original)
        assert bindings[arm]["source_unit_sha256"] == digest(original.tolist())
        assert bindings[arm]["dose"] == (0 if arm == "zero_sham" else screen.LOCKED_C_NORM)
    for arm, seed in zip(screen.RANDOM_ARMS, range(2026101501, 2026101506), strict=True):
        draw = np.random.Generator(np.random.PCG64(seed)).standard_normal(5120, dtype=np.float64)
        np.testing.assert_array_equal(bindings[arm]["unit"], draw / np.linalg.norm(draw))
        assert bindings[arm]["original_draw_byte_sha256"] == p.array_sha(draw)
    inv = p.direction_inventory(bindings)
    assert not inv["resampled"] and not inv["orientation_selected_by_outcome"]
    assert candidate["selected_sites"]["curiosity_vs_indifference"] == dict(layer=15, position="record")
    for row in inv["directions"]:
        assert row["effective_float32_byte_sha256"] == p.array_sha(
            np.asarray(bindings[row["arm"]]["unit"], dtype=np.float32)
        )
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="original independently selected"):
        p.locked_directions(tmp_path)


def source_tree(root, monkeypatch, source):
    run = root / p.SOURCE_SCREEN
    write_once(run / "manifest.json", source)
    (run / "source.tar.gz").write_bytes(b"synthetic source archive")
    write_once(
        run / "freeze.json", dict(manifest_sha256=digest(source), archive_sha256=p.sha(run / "source.tar.gz"))
    )
    audit = dict(
        parent_manifest_sha256=digest(source),
        complete=True,
        integrity_passed=True,
        sham_identity_passed=True,
        engagement_passed=True,
        receipts_passed=True,
        readiness=dict(passed=False),
        passed=False,
    )
    write_once(root / p.SOURCE_AUDIT, audit)
    analysis = dict(
        parent_manifest_sha256=digest(source),
        audit_sha256=digest(audit),
        decision=dict(
            classification="readiness_failed",
            positive_screen=False,
            effect_threshold_met=False,
            primary_panel_differences=dict(panel_1=0.0, panel_2=0.0, pooled=0.0),
            primary_pooled_menu_differences={"0": 0.0, "1": 0.0},
        ),
    )
    write_once(root / p.SOURCE_ANALYSIS, analysis)
    check = dict(
        passed=True,
        parent_manifest_sha256=digest(source),
        analysis_sha256=digest(analysis),
        analysis_byte_sha256=p.sha(root / p.SOURCE_ANALYSIS),
        audit_sha256=digest(audit),
        audit_byte_sha256=p.sha(root / p.SOURCE_AUDIT),
    )
    write_once(root / p.SOURCE_CHECK, check)
    write_once(
        root / p.SOURCE_SCORING,
        dict(
            passed=True,
            checked=1152,
            assigned=1152,
            parent_manifest_sha256=digest(source),
            complete=True,
            all_scoring_agrees=True,
            mismatches=[],
        ),
    )
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "audit_satisfaction_screen",
        SimpleNamespace(audit=lambda run: calls.append(run) or deepcopy(audit)),
    )
    return audit, analysis, calls


def test_failed_original_scientific_decision_is_preserved_while_integrity_is_reproduced(
    tmp_path, monkeypatch, source_inputs
):
    source = source_inputs[0]
    _, _, calls = source_tree(tmp_path, monkeypatch, source)
    result, certificate = p.source_certificate(tmp_path)
    assert result == source and calls == [tmp_path / p.SOURCE_SCREEN]
    assert certificate["prior_classification"] == "readiness_failed"
    assert certificate["prior_positive_screen"] is False
    assert certificate["prior_exact_choice_effect"] == 0


@pytest.mark.parametrize("change", ["unreproduced", "archive", "scientific_relabel", "scoring_subset"])
def test_original_source_admission_rejects_changed_or_unreproduced_evidence(
    tmp_path, monkeypatch, source_inputs, change
):
    _, analysis, _ = source_tree(tmp_path, monkeypatch, source_inputs[0])
    if change == "unreproduced":
        monkeypatch.setitem(sys.modules, "audit_satisfaction_screen", SimpleNamespace(audit=lambda _: {}))
    elif change == "archive":
        (tmp_path / p.SOURCE_SCREEN / "source.tar.gz").write_bytes(b"changed")
    elif change == "scientific_relabel":
        analysis["decision"]["classification"] = "passed"
        (tmp_path / p.SOURCE_ANALYSIS).write_text(canonical(analysis))
        check = read(tmp_path / p.SOURCE_CHECK)
        check.update(
            analysis_sha256=digest(analysis), analysis_byte_sha256=p.sha(tmp_path / p.SOURCE_ANALYSIS)
        )
        (tmp_path / p.SOURCE_CHECK).write_text(canonical(check))
    else:
        scoring = read(tmp_path / p.SOURCE_SCORING)
        scoring["checked"] = 1151
        (tmp_path / p.SOURCE_SCORING).write_text(canonical(scoring))
    with pytest.raises(ValueError):
        p.source_certificate(tmp_path)
    assert not (tmp_path / "executions").exists()


def profiles(root):
    paths = []
    for suffix, cohorts, execution, load in [
        ("baseline-zero", 18, 1800.0, 40.0),
        ("positive-half", 9, 1200.0, 60.0),
        ("positive-full", 9, 900.0, 50.0),
    ]:
        path = root / f"{p.SOURCE_SCREEN}-{suffix}" / "cloud_profiles/synthetic.json"
        write_once(
            path,
            dict(
                gpu="NVIDIA H200",
                assigned=cohorts * 32,
                completed=cohorts * 32,
                execution_seconds=execution,
                load_seconds=load,
            ),
        )
        paths.append(path)
    return paths


def test_forecast_scales_all45_cohorts_binds_profiles_and_retains_allcap_risk(tmp_path):
    paths = profiles(tmp_path)
    report = p.resource_forecast(tmp_path)
    assert report["estimated_execution_seconds_per_worker_range"] == [4500.0, 6000.0]
    assert report["maximum_observed_load_seconds"] == 60
    assert report["cohorts_per_worker"] == 45 and report["branches_per_worker"] == 1440
    assert report["estimated_combined_compute_usd_range"] == [
        3 * (4500 + 60 + 120) * report["rate_usd_per_second"],
        3 * (6000 + 60 + 120) * report["rate_usd_per_second"],
    ]
    assert report["hypothetical_all_cap_seconds_per_worker"] == 45 * 4096 * 0.218 > 12000
    assert not report["completion_guaranteed"]
    assert [r["byte_sha256"] for r in report["source_profiles"]] == [p.sha(path) for path in paths]


def test_incomplete_or_ambiguous_runtime_profile_cannot_support_forecast(tmp_path):
    paths = profiles(tmp_path)
    value = read(paths[0])
    value["completed"] -= 1
    paths[0].write_text(canonical(value))
    with pytest.raises(ValueError, match="incomplete"):
        p.resource_forecast(tmp_path)
    value["completed"] += 1
    paths[0].write_text(canonical(value))
    write_once(paths[0].parent / "duplicate.json", value)
    with pytest.raises(ValueError, match="unique"):
        p.resource_forecast(tmp_path)


def original_preflight_inputs(source_inputs, monkeypatch):
    source, directions = deepcopy(source_inputs)
    initial_plan = screen.assignment_plan(source_manifest=source, direction_bindings=directions)
    model = fake_model()
    preflight = capture.bind_preflight(initial_plan, model)
    source["generation_preflight"] = deepcopy(preflight)
    source["generation_preflight_sha256"] = digest(preflight)
    monkeypatch.setattr(screen, "PRIOR_MANIFEST_SHA256", digest(source))
    plan = screen.assignment_plan(source_manifest=source, direction_bindings=directions)
    assert capture.bind_preflight(plan, model) == preflight
    return source, directions, plan, preflight, p.receipt_preflight(plan, model)


def test_real_builder_validates_three_complete_children_before_returning(source_inputs, monkeypatch):
    source, directions, plan, preflight, receipt = original_preflight_inputs(source_inputs, monkeypatch)

    monkeypatch.setattr(p, "source_hash", lambda: "synthetic-source")
    parent, children = p.build_manifests(
        plan,
        preflight,
        receipt,
        source,
        {"synthetic": True},
        {"synthetic": True},
        {"synthetic": True},
        p.direction_inventory(directions),
        {"synthetic": True},
        [],
    )
    assert len(children) == 3 and len(parent["diagnostics"]) == 4320
    assert parent["analysis_metadata"]["minimum_each_control_panel_difference"] == 0.125
    assert all(len(capture.validate_manifest(c)) == 1440 for c in children)
    assert (
        sorted((r for c in children for r in c["diagnostics"]), key=lambda r: r["parent_row"])
        == parent["diagnostics"]
    )
    assert all(c["parent_manifest_sha256"] == digest(parent) for c in children)
    assert sum(sum(r["kind"] == "choice" for r in c["diagnostics"]) for c in children) == 2160
    assert receipt["synthetic_answer_only"] and not receipt["generated_model_outputs"]


def pipeline(root, monkeypatch, *, failure=None):
    # Control-flow fixture; all prerequisites and heavyweight operations are synthetic.
    doc_names = [
        p.PLAN,
        *p.CANDIDATE_DOCUMENTS.values(),
        f"{p.SOURCE_SCREEN}/manifest.json",
        f"{p.SOURCE_SCREEN}/freeze.json",
        f"{p.SOURCE_SCREEN}/source.tar.gz",
        p.SOURCE_AUDIT,
        p.SOURCE_ANALYSIS,
        p.SOURCE_CHECK,
        p.SOURCE_SCORING,
        "reports/satisfaction-screen-v1-results.md",
        f"{p.ENGINEERING}/manifest.json",
        f"{p.ENGINEERING}/freeze.json",
        f"{p.ENGINEERING}/source.tar.gz",
        p.ENGINEERING_AUDIT,
    ]
    for name in doc_names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"synthetic":true}' if path.suffix == ".json" else "synthetic prerequisite")
    events = []

    def stage(name, value):
        def call(*args, **kwargs):
            events.append(name)
            if failure == name:
                raise ValueError(f"Synthetic {name} failure")
            return deepcopy(value)

        return call

    monkeypatch.setattr(p, "source_certificate", stage("source", ({"source": True}, {"certificate": True})))
    monkeypatch.setattr(p, "previous_engineering_certificate", stage("engineering", {"engineering": True}))
    monkeypatch.setattr(
        p, "locked_directions", stage("directions", ({"directions": True}, {"candidate": True}))
    )
    monkeypatch.setattr(screen, "assignment_plan", stage("plan", {"plan": True}))
    monkeypatch.setattr(p, "tokenizer_model", stage("tokenizer", None))
    monkeypatch.setattr(capture, "bind_preflight", stage("preflight", {"preflight": True}))
    monkeypatch.setattr(p, "receipt_preflight", stage("receipt", {"receipt": True}))
    monkeypatch.setattr(p, "direction_inventory", stage("inventory", {"inventory": True}))
    forecast = dict(
        maximum_reserved_compute_usd=50.0,
        estimated_execution_seconds_per_worker_range=[5000.0, 6000.0],
        maximum_observed_load_seconds=60.0,
    )
    if failure == "headroom":
        forecast["estimated_execution_seconds_per_worker_range"][1] = 12000
    monkeypatch.setattr(p, "resource_forecast", stage("forecast", forecast))

    def build(*args):
        events.append("build")
        assert not (root / "runs/new-screen").exists()
        assert all(
            not (root / name).exists() for name in (p.GENERATION, p.RECEIPTS, p.DIRECTIONS, p.FORECAST)
        )
        if failure == "build":
            raise ValueError("Synthetic last-manifest validation failure")
        parent = dict(bound_receipts=args[-1])
        return parent, [
            dict(execution_group=group, parent_manifest_sha256=digest(parent)) for group in screen.GROUPS
        ]

    monkeypatch.setattr(p, "build_manifests", build)

    def freeze(project, out, documents):
        events.append("freeze")
        assert len(read(out / "children.json")) == 3
        parent = read(out / "manifest.json")
        assert all(p.sha(root / item["path"]) == item["byte_sha256"] for item in parent["bound_receipts"])
        return dict(source_sha256="synthetic-source")

    monkeypatch.setattr(p, "freeze", freeze)
    return events


@pytest.mark.parametrize(
    "failure",
    ["source", "engineering", "directions", "tokenizer", "preflight", "receipt", "headroom", "build"],
)
def test_any_failed_prerequisite_or_late_manifest_validation_publishes_nothing(
    tmp_path, monkeypatch, failure
):
    events = pipeline(tmp_path, monkeypatch, failure=failure)
    with pytest.raises(ValueError):
        p.prepare(tmp_path / "runs/new-screen", root=tmp_path)
    assert not (tmp_path / "runs/new-screen").exists() and "freeze" not in events
    assert all(
        not (tmp_path / name).exists() for name in (p.GENERATION, p.RECEIPTS, p.DIRECTIONS, p.FORECAST)
    )


def test_success_publishes_exact_bound_bytes_only_after_build_and_refuses_overwrite(tmp_path, monkeypatch):
    events = pipeline(tmp_path, monkeypatch)
    result = p.prepare(tmp_path / "runs/new-screen", root=tmp_path)
    assert result["assigned"] == 4320 and result["workers"] == 3 and not result["gpu_launched"]
    assert events[-2:] == ["build", "freeze"]
    before = list(events)
    with pytest.raises(ValueError, match="immutable"):
        p.prepare(tmp_path / "runs/new-screen", root=tmp_path)
    assert events == before


def test_existing_execution_copy_is_rejected_before_any_new_publication(tmp_path, monkeypatch):
    events = pipeline(tmp_path, monkeypatch)
    (tmp_path / "executions/new-screen").mkdir(parents=True)
    with pytest.raises(ValueError, match="immutable"):
        p.prepare(tmp_path / "runs/new-screen", root=tmp_path)
    assert events == [] and not (tmp_path / "runs/new-screen").exists()
    assert all(
        not (tmp_path / name).exists() for name in (p.GENERATION, p.RECEIPTS, p.DIRECTIONS, p.FORECAST)
    )


@pytest.mark.parametrize(
    "field,value",
    [("execution_seconds", v) for v in (0, -1, float("nan"), float("inf"), -float("inf"), True, None, "5")]
    + [("load_seconds", v) for v in (-1, float("nan"), float("inf"), -float("inf"), True, None, "5")],
)
def test_nonfinite_negative_or_nonnumeric_runtime_profile_rejected(tmp_path, field, value):
    path = profiles(tmp_path)[0]
    row = read(path)
    row[field] = value
    path.write_text(canonical(row))
    with pytest.raises(ValueError, match="finite positive execution"):
        p.resource_forecast(tmp_path)


def test_zero_load_with_positive_integer_execution_is_valid(tmp_path):
    paths = profiles(tmp_path)
    for path in paths:
        row = read(path)
        row.update(load_seconds=0, execution_seconds=1800)
        path.write_text(canonical(row))
    forecast = p.resource_forecast(tmp_path)
    assert forecast["maximum_observed_load_seconds"] == 0
    assert forecast["estimated_execution_seconds_per_worker_range"] == [4500, 9000]


@pytest.mark.parametrize(
    "change",
    ["token_ids", "record_position", "final_position", "seed", "order", "request_digest", "base_binding"],
)
def test_rehashed_current_preflight_must_still_equal_original_source_requests(
    source_inputs, monkeypatch, change
):
    source, _, plan, preflight, receipt = original_preflight_inputs(source_inputs, monkeypatch)
    changed = deepcopy(preflight)
    request = changed["requests"][0]["request"]
    if change == "token_ids":
        request["token_prefix"][0] += 1
        request["token_prefix_sha256"] = digest(request["token_prefix"])
    elif change == "record_position":
        request["prefix_positions"]["record"] -= 1
    elif change == "final_position":
        request["prefix_positions"]["final"] -= 1
    elif change == "seed":
        request["seed"] += 1
    elif change == "order":
        changed["requests"][0], changed["requests"][1] = changed["requests"][1], changed["requests"][0]
    elif change == "base_binding":
        changed["base_assignment_sha256"] = digest("different base assignments")
    changed["requests_sha256"] = digest(changed["requests"])
    if change == "request_digest":
        changed["requests_sha256"] = digest("false request digest")
    reached_validation = []
    monkeypatch.setattr(capture, "validate_manifest", lambda _: reached_validation.append(True))
    with pytest.raises(ValueError, match="exact original token requests"):
        p.build_manifests(plan, changed, receipt, source, {}, {}, {}, {}, {}, [])
    assert reached_validation == []
    assert source["generation_preflight"] == preflight
