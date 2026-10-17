"""Synthetic prospective eligibility receipts; no study outcomes or fitted directions."""

import hashlib
import importlib.util
import itertools
from pathlib import Path
import sys

import numpy as np
import pytest

from curiosity_study import information_state as state, target_relevant_information as target
from curiosity_study.artifacts import digest, read, write_once
from curiosity_study.transfer_representation import transfer_estimand, transfer_groups


def evaluator():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "target_candidate_evaluator", scripts / "evaluate_target_transfer_candidate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_manifest(rows):
    return dict(
        assignments=[],
        diagnostics=rows,
        assignment_sha256=digest(rows),
        task_revision=target.REVISION,
        phase="development_validation",
        inference_scope="development",
        model=dict(model_id="Qwen/Qwen3-32B", revision="synthetic-only"),
        task_module_sha256=hashlib.sha256(Path(target.__file__).read_bytes()).hexdigest(),
        validation_criteria=dict(
            minimum_distinct_items_per_cell=24, minimum_accuracy=0.9, minimum_valid_format=0.95
        ),
    )


def inputs(tmp_path, module, monkeypatch, score_change=None):
    rows = [{**a, "max_new_tokens": 4096} for a in target.assignments(items=24, seed=module.SEED)]
    manifest = source_manifest(rows)
    source = tmp_path / "source"
    write_once(source / "manifest.json", manifest)
    audit = dict(
        manifest_sha256=digest(manifest),
        complete=True,
        assigned=960,
        max_new_tokens=4096,
        validation_criteria=manifest["validation_criteria"],
        target_relevance_oracle_audit_version=1,
        gates=dict(
            passed=True,
            cells=[
                dict(
                    family=family,
                    kind=kind,
                    independent_report=u,
                    query=q,
                    n=24,
                    items=24,
                    completed=24,
                    correct=24,
                    valid=24,
                    passed=True,
                )
                for family, kind, u, q in itertools.product(module.FAMILIES, target.KINDS, (0, 1), (0, 1))
            ],
        ),
    )
    audit_path = tmp_path / "audit.json"
    write_once(audit_path, audit)
    extraction = tmp_path / "extraction"
    extraction_rows = state.assignments(items=2, kinds=("new", "repeat"))
    for a in extraction_rows:
        a["representation_split"] = "extraction" if a["index"] == 0 else "development"
    extraction_manifest = dict(
        diagnostics=extraction_rows,
        assignment_sha256=digest(extraction_rows),
        task_module_sha256=hashlib.sha256(Path(state.__file__).read_bytes()).hexdigest(),
    )
    write_once(extraction / "manifest.json", extraction_manifest)
    directions = tmp_path / "directions/directions.npz"
    directions.parent.mkdir()
    sources, layers, positions = ["pooled:balanced", "objects:balanced"], [47, 31], ["record", "prefill"]
    np.savez_compressed(
        directions,
        sources=sources,
        layers=layers,
        positions=positions,
        unit=np.ones((2, 2, 2, 3)) / np.sqrt(3),
        threshold=np.zeros((2, 2, 2)),
    )
    direction_sha = hashlib.sha256(directions.read_bytes()).hexdigest()
    monkeypatch.setattr(module, "FROZEN_DIRECTIONS_SHA256", direction_sha)
    original = dict(
        model=manifest["model"],
        manifest_sha256=digest(extraction_manifest),
        directions_sha256=direction_sha,
        development_only=True,
    )
    write_once(directions.parent / "analysis.json", original)
    scores = np.array([2 * int(a["independent_report"] != a["query"]) - 1 for a in rows], dtype=float)
    scores = np.broadcast_to(scores[:, None, None, None], (960, 2, 2, 2)).copy()
    if score_change is not None:
        score_change(rows, scores)
    transfer = tmp_path / "transfer"
    transfer.mkdir()
    np.savez_compressed(
        transfer / "scores.npz",
        scores=scores,
        trial_ids=[a["trial_id"] for a in rows],
        sources=sources,
        layers=layers,
        positions=positions,
        information_quantity="expected_target_information_gain_bits",
        contrast_gain_labels_bits=[0, 1, 1, 0],
    )
    groups = transfer_groups(rows)
    metrics = []
    for si, name in enumerate(sources):
        for li, layer in enumerate(layers):
            for pi, position in enumerate(positions):
                for family, kind in itertools.product(module.FAMILIES, target.KINDS):
                    values = [
                        scores[indices, si, li, pi]
                        for key, indices in groups.items()
                        if key[0] == family and key[3] == kind
                    ]
                    metrics.append(
                        dict(
                            source=name,
                            layer=layer,
                            position=position,
                            target_family=family,
                            target_kind=kind,
                            template="table",
                            source_axis_available=True,
                            **module.point_metrics(values, 0),
                        )
                    )
    analysis = dict(
        manifest_sha256=digest(manifest),
        source_audit_sha256=digest(audit),
        source_analysis_sha256=digest(original),
        directions_sha256=direction_sha,
        direction_extraction_manifest_sha256=digest(extraction_manifest),
        model=manifest["model"],
        assigned=960,
        task_revision=target.REVISION,
        refitted_on_transfer=False,
        descriptive_only=True,
        transfer_estimand=transfer_estimand(rows),
        metrics=metrics,
        split_provenance=module.split_provenance(
            dict(inference_scope="development", direction_extraction_collection=str(extraction)),
            manifest,
            original,
            rows,
        ),
    )
    write_once(transfer / "analysis.json", analysis)
    plan = tmp_path / "prospective-plan.md"
    plan.write_text("Synthetic plan fixture: fixed original pooled:balanced/layer47/record candidate.\n")
    return transfer, source, audit_path, directions, extraction, plan


def test_complete_fixed_candidate_receipt_binds_inputs_and_is_immutable(tmp_path, monkeypatch):
    module = evaluator()
    args = inputs(tmp_path, module, monkeypatch)
    out = tmp_path / "receipt.json"
    result = module.evaluate(*args, out)
    assert result["eligible_original_axis"] and all(result["checks"].values())
    assert result["candidate"] == dict(source="pooled:balanced", layer=47, position="record")
    assert result["assigned_record_captures"] == result["available_record_captures"] == 960
    assert result["descriptive_metric_rows"] == 80 and not result["other_axes_selected"]
    assert result["no_fitting_performed"] and not result["threshold_accuracy_is_admission_criterion"]
    assert len(result["world_ledger"]) == 48 and len(result["summaries"]) == 4
    for summary in result["summaries"]:
        assert summary["raw_index_metrics"]["assigned_worlds"] == 24
        assert summary["equal_semantic_world_metrics"]["auc"] == 1
        assert summary["equal_semantic_world_metrics"]["mean_factorial_gap"] == 2
    for split in result["calibration_check_semantics"]:
        assert split["calibration_assigned_indices"] == split["check_assigned_indices"] == 12
        assert split["unique_semantic_worlds"] >= 20
    for key, path in (
        ("prospective_plan", args[-1]),
        ("transfer_scores", args[0] / "scores.npz"),
        ("original_directions", args[3]),
    ):
        assert result["bindings"][key]["byte_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert read(out) == result
    with pytest.raises(ValueError, match="immutable"):
        module.evaluate(*args, out)


def test_failed_candidate_is_not_rescued_by_another_site_or_axis(tmp_path, monkeypatch):
    module = evaluator()

    def change(rows, scores):
        for i, a in enumerate(rows):
            if (a["family"], a["kind"]) == ("objects", "new"):
                scores[i, 0, 0, 0] *= -1

    args = inputs(tmp_path, module, monkeypatch, change)
    result = module.evaluate(*args, tmp_path / "receipt.json")
    assert not result["eligible_original_axis"]
    failed = next(s for s in result["summaries"] if (s["family"], s["kind"]) == ("objects", "new"))
    assert failed["equal_semantic_world_metrics"]["auc"] == 0
    assert failed["equal_semantic_world_metrics"]["mean_factorial_gap"] == -2
    assert not result["other_axes_selected"]


def test_auc_boundary_is_inclusive_without_using_original_threshold_accuracy(tmp_path, monkeypatch):
    module = evaluator()

    def change(rows, scores):
        for i, a in enumerate(rows):
            task = target.task(a)
            names = sorted(task["attributes"][:2])
            cell = (
                names.index(task["attributes"][a["independent_report"]]),
                names.index(task["attributes"][a["query"]]),
            )
            scores[i, 0, 0, 0] = 100 + int(cell == (0, 1))

    args = inputs(tmp_path, module, monkeypatch, change)
    result = module.evaluate(*args, tmp_path / "receipt.json")
    assert result["eligible_original_axis"]
    for item in result["summaries"]:
        metrics = item["equal_semantic_world_metrics"]
        assert metrics["auc"] == 0.75 and metrics["mean_factorial_gap"] == 0.5
        assert metrics["accuracy"] == 0.5  # The frozen threshold is deliberately poor.


def test_high_auc_cannot_rescue_zero_mean_gain_gap(tmp_path, monkeypatch):
    module = evaluator()

    def change(rows, scores):
        semantic_ids = {}
        for a in rows:
            if a["family"] == "objects" and a["index"] not in semantic_ids:
                semantic_ids[a["index"]] = digest(
                    module.canonical_target_world(
                        a["family"], [target.task({**a, "independent_report": u}) for u in (0, 1)]
                    )
                )
        unique = sorted(set(semantic_ids.values()))
        for i, a in enumerate(rows):
            if (a["family"], a["kind"]) == ("objects", "new"):
                amplitude = -(len(unique) - 1) if semantic_ids[a["index"]] == unique[0] else 1
                scores[i, 0, 0, 0] = amplitude if a["query"] != a["independent_report"] else 0

    args = inputs(tmp_path, module, monkeypatch, change)
    result = module.evaluate(*args, tmp_path / "receipt.json")
    item = next(s for s in result["summaries"] if (s["family"], s["kind"]) == ("objects", "new"))
    assert item["equal_semantic_world_metrics"]["auc"] > 0.75
    assert item["equal_semantic_world_metrics"]["mean_factorial_gap"] == 0
    assert not item["signed_gap_positive"] and not result["eligible_original_axis"]


def test_a_missing_count_capture_fails_even_if_all_judgment_metrics_pass(tmp_path, monkeypatch):
    module = evaluator()

    def change(rows, scores):
        index = next(i for i, a in enumerate(rows) if a["kind"] == "count")
        scores[index, 0, 0, 0] = np.nan

    args = inputs(tmp_path, module, monkeypatch, change)
    result = module.evaluate(*args, tmp_path / "receipt.json")
    assert result["checks"]["signed_auc_and_positive_gap_in_each_family_and_polarity"]
    assert not result["checks"]["all_assigned_record_captures_present"]
    assert not result["eligible_original_axis"] and result["available_record_captures"] == 959


def test_duplicate_worlds_cannot_win_through_raw_index_frequency(monkeypatch):
    module = evaluator()
    original = target.task
    monkeypatch.setattr(
        target,
        "task",
        lambda a: original({**a, "index": 0 if a["family"] == "objects" and a["index"] != 1 else a["index"]}),
    )
    rows = target.assignments(items=24, seed=module.SEED)
    scores = np.array(
        [
            (2 * int(a["query"] != a["independent_report"]) - 1)
            * (-2 if a["family"] == "objects" and a["index"] == 1 else 1)
            for a in rows
        ],
        dtype=float,
    )
    result = module.semantic_summary(rows, scores, source_manifest(rows), 0)
    for item in (s for s in result["summaries"] if s["family"] == "objects"):
        assert item["raw_index_metrics"]["auc"] == pytest.approx(529 / 576)
        assert item["raw_index_metrics"]["mean_factorial_gap"] > 0
        assert item["equal_semantic_world_metrics"]["assigned_worlds"] == 2
        assert item["equal_semantic_world_metrics"]["auc"] == 0.25
        assert item["equal_semantic_world_metrics"]["mean_factorial_gap"] == -1


def test_split_ledger_detects_check_semantics_reused_from_calibration(tmp_path, monkeypatch):
    module = evaluator()
    original = target.task
    monkeypatch.setattr(target, "task", lambda a: original({**a, "index": a["index"] % 12}))
    args = inputs(tmp_path, module, monkeypatch)
    result = module.evaluate(*args, tmp_path / "receipt.json")
    assert not result["eligible_original_axis"] and not result["development_check_semantic_readiness"]
    for split in result["calibration_check_semantics"]:
        assert split["check_semantic_worlds_absent_from_calibration"] == 0
        assert set(split["calibration_semantic_world_ids"]) == set(split["overlap_semantic_world_ids"])
        assert split["unique_semantic_worlds"] <= 12


def test_transfer_eligibility_and_later_check_semantic_readiness_are_separate(tmp_path, monkeypatch):
    module = evaluator()
    original = target.task
    monkeypatch.setattr(
        target,
        "task",
        lambda a: original({**a, "index": a["index"] - 12 if 12 <= a["index"] < 15 else a["index"]}),
    )
    args = inputs(tmp_path, module, monkeypatch)
    result = module.evaluate(*args, tmp_path / "receipt.json")
    assert result["eligible_original_axis"]
    assert not result["development_check_semantic_readiness"]
    assert all(
        s["unique_semantic_worlds"] >= 20 and s["check_semantic_worlds_absent_from_calibration"] <= 9
        for s in result["calibration_check_semantics"]
    )


def test_duplicate_cells_align_by_named_report_not_display_order_and_missingness_propagates(monkeypatch):
    module = evaluator()
    original = target.task

    def duplicate_with_swapped_columns(a):
        if a["index"] == 0:
            return original(a)
        t = original(
            {**a, "index": 0, "independent_report": 1 - a["independent_report"], "query": 1 - a["query"]}
        )
        order = [1, 0, 2]
        t["attributes"] = [t["attributes"][i] for i in order]
        t["values"] = [t["values"][i] for i in order]
        t["candidates"] = [[row[i] for i in order] for row in t["candidates"]]
        for field in ("independent_report_attribute", "target_linked_report_attribute", "queried_attribute"):
            t[field] = 1 - t[field]
        return t

    monkeypatch.setattr(target, "task", duplicate_with_swapped_columns)
    rows = target.assignments(items=2, seed=module.SEED)
    scores = []
    for a in rows:
        task = target.task(a)
        names = sorted(task["attributes"][:2])
        cell = (
            names.index(task["attributes"][a["independent_report"]]),
            names.index(task["attributes"][a["query"]]),
        )
        scores.append({(0, 0): 0, (0, 1): 2, (1, 0): 4, (1, 1): 10}[cell])
    result = module.semantic_summary(rows, np.array(scores, dtype=float), source_manifest(rows), 5)
    for item in result["summaries"]:
        metric = item["equal_semantic_world_metrics"]
        assert metric["assigned_worlds"] == 1 and metric["auc"] == 0.5
        assert metric["accuracy"] == 0.25 and metric["mean_factorial_gap"] == -2
    missing = next(i for i, a in enumerate(rows) if a["family"] == "objects" and a["kind"] == "new")
    scores[missing] = np.nan
    result = module.semantic_summary(rows, np.array(scores), source_manifest(rows), 5)
    item = next(s for s in result["summaries"] if (s["family"], s["kind"]) == ("objects", "new"))
    assert item["raw_index_metrics"]["worlds"] == 1
    assert item["equal_semantic_world_metrics"]["worlds"] == 0


@pytest.mark.parametrize(
    "problem",
    [
        "source_gate",
        "source_subset",
        "direction",
        "score_order",
        "score_values",
        "target_quantity",
        "omitted_other_axis",
    ],
)
def test_receipt_rejects_unbound_or_incomplete_inputs(tmp_path, monkeypatch, problem):
    module = evaluator()
    args = inputs(tmp_path, module, monkeypatch)
    transfer, source, audit_path, direction, _, _ = args
    if problem in ("score_order", "score_values", "target_quantity"):
        path = transfer / "scores.npz"
        with np.load(path, allow_pickle=False) as z:
            contents = {key: z[key] for key in z.files}
        if problem == "score_order":
            contents["trial_ids"] = contents["trial_ids"][::-1]
        elif problem == "score_values":
            contents["scores"][:, 0, 0, 0] *= 5
        else:
            contents["information_quantity"] = np.array("information_bits")
        np.savez_compressed(path, **contents)
    elif problem == "direction":
        direction.write_bytes(direction.read_bytes() + b"changed")
    else:
        path = (
            audit_path
            if problem == "source_gate"
            else source / "manifest.json"
            if problem == "source_subset"
            else transfer / "analysis.json"
        )
        data = read(path)
        if problem == "source_gate":
            data["gates"]["cells"][0]["correct"] = 21
        elif problem == "source_subset":
            data["diagnostics"] = data["diagnostics"][:-1]
            data["assignment_sha256"] = digest(data["diagnostics"])
        else:
            data["metrics"].pop()
        path.unlink()  # Corrupt only this synthetic fixture to test admission.
        write_once(path, data)
    with pytest.raises(ValueError):
        module.evaluate(*args, tmp_path / "receipt.json")
    assert not (tmp_path / "receipt.json").exists()
