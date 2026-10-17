"""Synthetic artifacts exercise transfer admission, binding, and complete worlds."""

import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from curiosity_study import information_state as state, target_relevant_information as target
from curiosity_study.artifacts import RunStore, digest, write_once


def analyzer():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "target_transfer_analysis", scripts / "analyze_transfer_representations.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_inputs(tmp_path, *, gate_passed=True):
    model = dict(name="synthetic-test-model", revision="synthetic-revision")
    rows = target.assignments(items=2, kinds=("new", "repeat"))
    manifest = dict(
        task_revision=target.REVISION,
        model=model,
        phase="development_validation",
        inference_scope="development",
        diagnostics=rows,
        assignment_sha256=digest(rows),
        task_module_sha256=hashlib.sha256(Path(target.__file__).read_bytes()).hexdigest(),
    )
    root = tmp_path / "target"
    store = RunStore(root, manifest)
    layers, positions = [47], ["record", "think_end"]
    hidden = np.zeros((len(rows), 1, 2, 2), dtype=np.float32)
    for i, a in enumerate(rows):
        gain = target.task(a)["expected_target_information_gain_bits"]
        hidden[i, :, :, 0] = 4 * gain - 2
        hidden[i, :, :, 1] = 100  # Orthogonal nuisance must not change the frozen projection.
        if (a["family"], a["index"], a["kind"], a["independent_report"], a["query"]) == (
            "objects",
            0,
            "new",
            0,
            0,
        ):
            hidden[i, 0, 0] = np.nan
    activation = root / "activation.npz"
    np.savez_compressed(
        activation,
        hidden=hidden,
        trial_ids=np.array([a["trial_id"] for a in rows]),
        available=np.isfinite(hidden).all(-1),
        layers=np.array(layers),
        positions=np.array(positions),
    )
    sha = hashlib.sha256(activation.read_bytes()).hexdigest()
    for i, a in enumerate(rows):
        store.save(
            dict(
                trial_id=a["trial_id"],
                terminal=True,
                assignment=a,
                task=target.task(a),
                valid=False,
                correct=False,  # Transfer must never select only successful answers.
                activation_artifact=dict(
                    path="activation.npz", sha256=sha, row=i, layers=layers, positions=positions
                ),
            ),
            "synthetic",
        )
    audit = tmp_path / "audit.json"
    write_once(audit, dict(manifest_sha256=digest(manifest), complete=True, gates=dict(passed=gate_passed)))
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
    directions = tmp_path / "directions" / "directions.npz"
    directions.parent.mkdir()
    unit = np.array([[[[1, 0], [1, 0]]], [[[-1, 0], [np.nan, np.nan]]]], dtype=float)
    np.savez_compressed(
        directions,
        unit=unit,
        threshold=np.array([[[0, 0]], [[0, np.nan]]]),
        sources=np.array(["synthetic:positive", "synthetic:negative"]),
        layers=np.array(layers),
        positions=np.array(positions),
    )
    write_once(
        directions.parent / "analysis.json",
        dict(
            manifest_sha256=digest(extraction_manifest),
            development_only=True,
            directions_sha256=hashlib.sha256(directions.read_bytes()).hexdigest(),
            model=model,
        ),
    )
    return root, audit, directions, extraction


def test_target_transfer_uses_target_gain_retains_sign_and_complete_world_missingness(tmp_path):
    module = analyzer()
    arguments = synthetic_inputs(tmp_path)
    result = module.analyze(*arguments, tmp_path / "result")
    assert result["assigned"] == 32 and not result["refitted_on_transfer"]
    assert result["descriptive_only"]
    assert result["transfer_estimand"]["information_quantity"] == "expected_target_information_gain_bits"
    assert result["transfer_estimand"]["full_world_information_gain_bits"] == 1
    assert result["split_provenance"]["confirmation_eligible"]
    assert len(result["split_provenance"]["evaluation_semantic_world_ids"]) == 4
    assert len(result["metrics"]) == 16  # Every source, family, polarity, position retained.
    for metric in result["metrics"]:
        assert metric["assigned_worlds"] == 2
        assert metric["information_quantity"] == "expected_target_information_gain_bits"
        negative = metric["source"] == "synthetic:negative"
        if negative and metric["position"] == "think_end":
            assert not metric["source_axis_available"]
            assert metric["worlds"] == 0 and metric["auc"] is None
        else:
            missing_world = (metric["target_family"], metric["target_kind"], metric["position"]) == (
                "objects",
                "new",
                "record",
            )
            assert metric["worlds"] == 2 - missing_world
            assert metric["mean_factorial_gap"] == (-4 if negative else 4)
            assert metric["auc"] == (0 if negative else 1)
            assert metric["accuracy"] == (0 if negative else 1)
    with np.load(tmp_path / "result" / "scores.npz", allow_pickle=False) as z:
        assert z["information_quantity"].item() == "expected_target_information_gain_bits"
        assert z["contrast_gain_labels_bits"].tolist() == [0, 1, 1, 0]
        assert z["scores"].shape == (32, 2, 1, 2)
        assert np.isnan(z["scores"][:, 1, 0, 1]).all()


def test_target_smoke_without_passing_global_gate_cannot_enter_transfer(tmp_path):
    module = analyzer()
    arguments = synthetic_inputs(tmp_path, gate_passed=False)
    with pytest.raises(AssertionError, match="Validate the transfer instrument"):
        module.analyze(*arguments, tmp_path / "result")
    assert not (tmp_path / "result").exists()
