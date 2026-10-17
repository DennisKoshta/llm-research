"""Synthetic complete artifacts for the one prospective target/control fit."""

import hashlib
import importlib.util
import io
import itertools
from pathlib import Path
import sys
import tarfile

import numpy as np
import pytest

from curiosity_study import information_state as state, target_relevant_information as target
from curiosity_study.artifacts import RunStore, digest, read, write_once
from curiosity_study.transfer_representation import transfer_estimand, transfer_groups


def fitter():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "target_candidate_fitter", scripts / "fit_target_transfer_candidate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path, monkeypatch, *, original_passes=True, change=None):
    module = fitter()
    evaluator = module.eligibility
    rows = [{**a, "max_new_tokens": 4096} for a in target.assignments(items=24, seed=evaluator.SEED)]
    manifest = dict(
        assignments=[],
        diagnostics=rows,
        assignment_sha256=digest(rows),
        source_sha256="synthetic-source",
        task_revision=target.REVISION,
        phase="development_validation",
        inference_scope="development",
        model=dict(model_id="Qwen/Qwen3-32B", revision="synthetic-only"),
        task_module_sha256=sha(Path(target.__file__)),
        validation_criteria=dict(
            minimum_distinct_items_per_cell=24, minimum_accuracy=0.9, minimum_valid_format=0.95
        ),
        development_split=dict(calibration_indices=[0, 12], check_indices=[12, 24]),
    )
    source = tmp_path / "source"
    store = RunStore(source, manifest)
    hidden = np.zeros((960, 1, 1, 4), dtype=np.float32)
    for i, a in enumerate(rows):
        task = target.task(a)
        names = sorted(task["attributes"][:2])
        u, q = [names.index(task["attributes"][a[key]]) for key in ("independent_report", "query")]
        gain_sign = 2 * int(u != q) - 1
        hidden[i, 0, 0] = [
            gain_sign if original_passes else 0,
            2 * gain_sign,
            0.5 * (2 * u - 1),
            0.75 * (2 * q - 1),
        ]
    if change is not None:
        change(rows, hidden)
    capture = source / "activations.npz"
    np.savez_compressed(
        capture,
        hidden=hidden,
        available=np.isfinite(hidden).all(-1),
        layers=[47],
        positions=["record"],
        trial_ids=[a["trial_id"] for a in rows],
    )
    records = [
        dict(
            trial_id=a["trial_id"],
            assignment=a,
            task=target.task(a),
            terminal=True,
            # Deliberate failures prove the fitter never filters by either field.
            valid=False,
            correct=False,
            activation_artifact=dict(
                path=capture.name, sha256=sha(capture), row=i, layers=[47], positions=["record"]
            ),
        )
        for i, a in enumerate(rows)
    ]
    store.save_many(records, "synthetic", "all")
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
                for family, kind, u, q in itertools.product(evaluator.FAMILIES, target.KINDS, (0, 1), (0, 1))
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
        task_module_sha256=sha(Path(state.__file__)),
    )
    write_once(extraction / "manifest.json", extraction_manifest)
    directions = tmp_path / "directions" / "directions.npz"
    directions.parent.mkdir()
    original_arrays = dict(
        sources=np.array(["pooled:balanced"]),
        layers=np.array([47]),
        positions=np.array(["record"]),
        unit=np.array([[[[1, 0, 0, 0]]]], dtype=np.float32),
        contrast=np.array([[[[3, 0, 0, 0]]]], dtype=np.float32),
        contrast_norm=np.array([[[3]]], dtype=np.float32),
        center=np.array([[[[0.25, 0, 0, 0]]]], dtype=np.float32),
        threshold=np.array([[[0.25]]], dtype=np.float32),
    )
    np.savez_compressed(directions, **original_arrays)
    direction_sha = sha(directions)
    monkeypatch.setattr(evaluator, "FROZEN_DIRECTIONS_SHA256", direction_sha)
    original = dict(
        model=manifest["model"],
        manifest_sha256=digest(extraction_manifest),
        directions_sha256=direction_sha,
        development_only=True,
    )
    write_once(directions.parent / "analysis.json", original)
    scores = hidden[:, :, :, 0].astype(np.float64)[:, None]
    transfer = tmp_path / "transfer"
    transfer.mkdir()
    np.savez_compressed(
        transfer / "scores.npz",
        scores=scores,
        trial_ids=[a["trial_id"] for a in rows],
        sources=original_arrays["sources"],
        layers=[47],
        positions=["record"],
        information_quantity="expected_target_information_gain_bits",
        contrast_gain_labels_bits=[0, 1, 1, 0],
    )
    groups = transfer_groups(rows)
    metrics = [
        dict(
            source="pooled:balanced",
            layer=47,
            position="record",
            target_family=family,
            target_kind=kind,
            template="table",
            source_axis_available=True,
            **evaluator.point_metrics(
                [
                    scores[indices, 0, 0, 0]
                    for key, indices in groups.items()
                    if (key[0], key[3]) == (family, kind)
                ],
                0.25,
            ),
        )
        for family, kind in itertools.product(evaluator.FAMILIES, target.KINDS)
    ]
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
        split_provenance=evaluator.split_provenance(
            dict(inference_scope="development", direction_extraction_collection=str(extraction)),
            manifest,
            original,
            rows,
        ),
    )
    write_once(transfer / "analysis.json", analysis)
    plan = tmp_path / "prospective-plan.md"
    plan.write_text("Synthetic current plan with pre-outcome named-control clarification.\n")
    archived_plan = b"Synthetic original frozen target plan before the clarification.\n"
    archive = source / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo(module.PLAN_MEMBER)
        member.size = len(archived_plan)
        tar.addfile(member, io.BytesIO(archived_plan))
    write_once(
        source / "freeze.json",
        dict(
            manifest_sha256=digest(manifest),
            source_sha256=manifest["source_sha256"],
            archive_sha256=sha(archive),
            documents={module.PLAN_MEMBER: hashlib.sha256(archived_plan).hexdigest()},
        ),
    )
    receipt = tmp_path / "receipt.json"
    evaluator.evaluate(transfer, source, audit_path, directions, extraction, plan, receipt)
    args = transfer, source, audit_path, directions, extraction, receipt, plan
    return module, args, original_arrays


def run_fit(tmp_path, monkeypatch, **kwargs):
    module, args, original = fixture(tmp_path, monkeypatch, **kwargs)
    out = tmp_path / "fit"
    result = module.fit(*args, out)
    with np.load(out / "directions.npz", allow_pickle=False) as artifact:
        arrays = {key: artifact[key] for key in artifact.files}
    return module, args, original, out, result, arrays


def test_eligible_original_retains_exact_bytes_norm_threshold_and_component_origins(tmp_path, monkeypatch):
    module, args, original, out, result, arrays = run_fit(tmp_path, monkeypatch)
    assert result["passed"] and result["original_transfer_eligible"]
    for name in ("unit", "contrast", "contrast_norm", "threshold", "center"):
        assert arrays[name].dtype == original[name].dtype
        assert arrays[name].tobytes() == original[name].tobytes()
    assert arrays["sources"].tolist() == ["target:balanced"]
    assert "known_contrast" not in arrays
    np.testing.assert_array_equal(arrays["target_u_contrast"][0, 0, 0], [0, 0, 1, 0])
    np.testing.assert_array_equal(arrays["target_q_contrast"][0, 0, 0], [0, 0, 0, 1.5])
    origin = result["component_origins"]["information"]
    assert (
        origin["manifest_sha256"]
        == origin["original_extraction_manifest_sha256"]
        == digest(read(args[4] / "manifest.json"))
    )
    assert origin["target_calibration"] is None and not origin["fit_on_target_calibration"]
    target_sha = digest(read(args[1] / "manifest.json"))
    assert result["component_origins"]["target_u"]["manifest_sha256"] == target_sha
    assert result["assignment_split"] == module.assignment_split(read(args[1] / "manifest.json"))
    assert len(result["assignment_split"]["assignments"]) == 960
    assert result["all_assigned_record_captures_retained"] == 960
    assert result["source_plan_versions"]["current_plan_matches_source_archive"] is False
    assert result["no_outcome_correctness_filter"]
    assert read(out / "analysis.json") == result
    manifest = read(args[1] / "manifest.json")
    provenance = module.eligibility.split_provenance(
        dict(inference_scope="development", direction_extraction_collection=str(args[1])),
        manifest,
        result,
        manifest["diagnostics"],
    )
    assert provenance["extraction_binding_verified"]
    assert provenance["direction_assignment_split_sha256"] == digest(result["assignment_split"])
    assert not provenance["confirmation_eligible"]
    with pytest.raises(ValueError, match="immutable"):
        module.fit(*args, out)


def test_failed_original_gets_exactly_one_calibration_contrast_and_keeps_failure_label(tmp_path, monkeypatch):
    _, _, _, _, result, arrays = run_fit(tmp_path, monkeypatch, original_passes=False)
    assert result["passed"] and not result["original_transfer_eligible"]
    assert result["selected_information_origin"] == "target_calibration"
    np.testing.assert_array_equal(arrays["contrast"][0, 0, 0], [0, 4, 0, 0])
    np.testing.assert_array_equal(arrays["unit"][0, 0, 0], [0, 1, 0, 0])
    assert arrays["threshold"].item() == 0
    assert all(m["auc"] == 1 and m["passed"] for m in result["check_block_evaluation"]["summaries"])
    origin = result["component_origins"]["information"]
    assert origin["manifest_sha256"] == result["manifest_sha256"]
    assert origin["original_extraction_manifest_sha256"] != result["manifest_sha256"]


@pytest.mark.parametrize("offset,invert", [(100, False), (0, True)])
def test_check_observations_cannot_fit_threshold_or_flip_fallback(tmp_path, monkeypatch, offset, invert):
    def change(rows, hidden):
        for i, a in enumerate(rows):
            if a["index"] >= 12:
                hidden[i, 0, 0, 1] = hidden[i, 0, 0, 1] * (-1 if invert else 1) + offset

    _, _, _, _, result, arrays = run_fit(tmp_path, monkeypatch, original_passes=False, change=change)
    np.testing.assert_array_equal(arrays["unit"][0, 0, 0], [0, 1, 0, 0])
    assert arrays["threshold"].item() == 0
    assert result["candidate_eligible"] is (not invert)
    for item in result["check_block_evaluation"]["summaries"]:
        assert item["auc"] == (0 if invert else 1)
        assert item["accuracy"] == (0 if invert else 0.5)


def test_tiny_raw_control_stops_panel_without_normalizing_it_into_availability(tmp_path, monkeypatch):
    def change(rows, hidden):
        hidden[:, :, :, 2] = 0

    _, _, _, _, result, arrays = run_fit(tmp_path, monkeypatch, change=change)
    assert result["candidate_eligible"] and not result["controls_available"] and not result["passed"]
    assert arrays["target_u_norm"].item() == 0
    assert not arrays["target_u_contrast"].any()


def test_calibration_equal_weights_worlds_families_and_redundant_polarities():
    module = fitter()
    sign = np.array([-1, 1, 1, -1])[:, None]
    captures = {}
    # Objects have two unique worlds, machines one. New/repeat are averaged first.
    for family, semantic, amplitude in (("objects", "a", 1), ("objects", "b", 3), ("machines", "c", 5)):
        for kind, shift in (("new", 2), ("repeat", -2)):
            captures[("calibration", family, semantic, kind)] = sign * (amplitude + shift)
    components, counts = module.calibration_components(captures)
    assert counts == dict(objects=2, machines=1)
    # World gain contrasts: objects 2 and 6 -> 4; machines 10 -> pooled 7.
    np.testing.assert_array_equal(components["information"], [7])
    np.testing.assert_array_equal(components["target_u"], [0])


def test_duplicate_record_captures_align_named_properties_before_cell_average(monkeypatch):
    module = fitter()
    original = target.task

    def duplicate_with_swapped_columns(a):
        if a["index"] == 0:
            return original(a)
        task = original(
            {**a, "index": 0, "independent_report": 1 - a["independent_report"], "query": 1 - a["query"]}
        )
        order = [1, 0, 2]
        task["attributes"] = [task["attributes"][i] for i in order]
        task["values"] = [task["values"][i] for i in order]
        task["candidates"] = [[row[i] for i in order] for row in task["candidates"]]
        for key in ("independent_report_attribute", "target_linked_report_attribute", "queried_attribute"):
            task[key] = 1 - task[key]
        return task

    monkeypatch.setattr(target, "task", duplicate_with_swapped_columns)
    rows = target.assignments(items=2, seed=module.eligibility.SEED)
    vectors = []
    for a in rows:
        task = target.task(a)
        names = sorted(task["attributes"][:2])
        u, q = [names.index(task["attributes"][a[key]]) for key in ("independent_report", "query")]
        # Duplicate indices differ only in nuisance offset, which must average to zero.
        offset = 9 if a["index"] == 0 else -9
        vectors.append([2 * int(u != q) - 1, 2 * u - 1, 2 * q - 1, offset])
    captures = module.semantic_captures(rows, np.array(vectors, dtype=float))
    components, counts = module.calibration_components(captures)
    assert counts == dict(objects=1, machines=1)
    np.testing.assert_array_equal(components["information"], [2, 0, 0, 0])
    np.testing.assert_array_equal(components["target_u"], [0, 2, 0, 0])
    np.testing.assert_array_equal(components["target_q"], [0, 0, 2, 0])
    np.testing.assert_array_equal(components["center"], [0, 0, 0, 0])


def test_check_excludes_seen_semantics_and_stops_below_ten_novel_per_family():
    module = fitter()
    captures = {}
    correct = np.array([-1, 1, 1, -1], dtype=float)[:, None]
    for family, novel_count in (("objects", 9), ("machines", 10)):
        for kind in module.eligibility.POLARITIES:
            captures[("calibration", family, "shared", kind)] = correct
            captures[("check", family, "shared", kind)] = -100 * correct
            for i in range(novel_count):
                captures[("check", family, str(i), kind)] = correct
    result = module.check_candidate(captures, np.array([1]), 0)
    assert not result["passed"]
    assert result["novel_semantic_world_counts"] == dict(objects=9, machines=10)
    assert result["excluded_calibration_overlap"] == dict(objects=["shared"], machines=["shared"])
    assert all(row["auc"] == 1 and row["mean_factorial_gap"] == 2 for row in result["summaries"])
    assert all(row["passed"] is (row["family"] == "machines") for row in result["summaries"])


def test_rejects_a_forged_original_receipt_before_output(tmp_path, monkeypatch):
    module, args, _ = fixture(tmp_path, monkeypatch, original_passes=False)
    receipt = read(args[5])
    receipt["eligible_original_axis"] = True
    args[5].unlink()
    write_once(args[5], receipt)
    with pytest.raises(ValueError, match="independently recomputed"):
        module.fit(*args, tmp_path / "fit")
    assert not (tmp_path / "fit").exists()


def test_rejects_source_captures_not_equal_to_bound_transfer_scores(tmp_path, monkeypatch):
    module, args, _ = fixture(tmp_path, monkeypatch)
    original_loader = module.load_collection

    def corrupted_loader(path):
        manifest, assignments, records, locations, vectors, layers, positions = original_loader(path)
        vectors[0, 0, 0, 0] += 1
        return manifest, assignments, records, locations, vectors, layers, positions

    monkeypatch.setattr(module, "load_collection", corrupted_loader)
    with pytest.raises(ValueError, match="actual source record captures"):
        module.fit(*args, tmp_path / "fit")
    assert not (tmp_path / "fit").exists()


def test_missing_capture_in_control_readout_cannot_be_filtered(tmp_path, monkeypatch):
    module, args, _ = fixture(tmp_path, monkeypatch)
    original_loader = module.load_collection

    def corrupted_loader(path):
        manifest, assignments, records, locations, vectors, layers, positions = original_loader(path)
        index = next(i for i, a in enumerate(assignments) if a["kind"] == "count")
        vectors[index, 0, 0, 2] = np.nan
        return manifest, assignments, records, locations, vectors, layers, positions

    monkeypatch.setattr(module, "load_collection", corrupted_loader)
    with pytest.raises(ValueError):
        module.fit(*args, tmp_path / "fit")
    assert not (tmp_path / "fit").exists()


def test_archive_hash_and_development_split_fail_closed(tmp_path, monkeypatch):
    module, args, _ = fixture(tmp_path, monkeypatch)
    manifest = read(args[1] / "manifest.json")
    changed = {**manifest, "development_split": dict(calibration_indices=[0, 11], check_indices=[11, 24])}
    with pytest.raises(ValueError, match="twelve-plus-twelve"):
        module.assignment_split(changed)
    with (args[1] / "source.tar.gz").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="archive hash mismatch"):
        module.fit(*args, tmp_path / "fit")
    assert not (tmp_path / "fit").exists()
