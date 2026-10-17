"""Synthetic-only fixed-decoder tests; no closed-source activations are fitted."""

from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "choice_support_producer_test", Path(__file__).resolve().parents[1] / "scripts/choice_support_readout_offline.py"
)
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)


def rows():
    result = []
    for split in ("fit", "evaluation"):
        for world in range(12):
            for delivery in p.DELIVERIES:
                for menu in (0, 1):
                    bits = [int(j == world % 4 or (delivery != p.DELIVERIES[0] and j == (world + 1) % 4)) for j in range(4)]
                    overlap = split == "fit" or (world == 3 and delivery != p.DELIVERIES[0])
                    result.append(dict(
                        split=split, trial_id=f"{split}-{world}-{delivery}-{menu}", world_slot=world,
                        semantic_world_id=f"{split}-world-{world}", delivery=delivery, menu_layout=menu,
                        known_attribute=world % 2, oracle_bits=bits, visible_problem_sha256=f"visible-{world}-{delivery}",
                        seen_in_fit=overlap, available=True,
                        source_flags=dict(valid=True, cap_hit=False, thinking_closed=True, closure_count=1),
                    ))
    return result


def perfect_predictions():
    output = rows()
    reference = p.shortcut(output[:72])
    for row in output:
        row.update(scores=[float(x) for x in row["oracle_bits"]], predicted_bits=row["oracle_bits"].copy(),
                   shortcut_bits=reference[str(sum(row["oracle_bits"]))]["bits"], prediction_status="predicted")
    return output


def features(assigned, width=8):
    rng = np.random.default_rng(441)
    result = rng.normal(0, 0.01, (144, width))
    result[:, :4] += np.asarray([r["oracle_bits"] for r in assigned])
    return result


def test_ridge_matches_fixed_objective_and_degrees_of_freedom():
    rng = np.random.default_rng(3)
    X, Y = rng.normal(size=(72, 128)), rng.integers(0, 2, (72, 4))
    model, diagnostics = p.fit_ridge(X, Y)
    Z = (X - model["mu"]) / model["scale"]
    np.testing.assert_allclose(np.trace(Z @ Z.T), 72, rtol=1e-13)
    np.testing.assert_allclose(Z.T @ (Z @ model["W"] - (Y - model["ybar"])) / 72 + p.LAMBDA * model["W"], 0, atol=2e-16)
    assert diagnostics["rank"] <= 71
    assert 0 < diagnostics["effective_df"] <= 12
    assert model["W"].dtype == model["B"].dtype == np.float64


def test_transform_and_fit_never_use_evaluation_features(monkeypatch):
    monkeypatch.setitem(p.METHOD, "features", 8)
    assigned = rows()
    X = features(assigned)
    first, _, _, predictions = p.build_predictions(assigned, X)
    changed = X.copy()
    changed[72:] = 1000 * X[72:] + 77
    second, _, _, later = p.build_predictions(assigned, changed)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
    assert predictions[0] == later[0]
    assert predictions[72]["scores"] != later[72]["scores"]


def test_scores_are_not_clipped_normalized_or_top_k_repaired():
    model = dict(mu=np.zeros(4), scale=np.asarray(1.0), ybar=np.zeros(4), W=np.eye(4))
    np.testing.assert_array_equal(p.predict(model, [[-2, 0.5, 3, 4]]), [[-2, 0.5, 3, 4]])


def test_shortcut_uses_only_training_labels_and_lexical_ties():
    training = [dict(oracle_bits=list(map(int, mask))) for mask in ("1000", "0100", "1100", "0110")]
    result = p.shortcut(training)
    assert result["1"]["bits"] == [0, 1, 0, 0]
    assert result["2"]["bits"] == [0, 1, 1, 0]


def test_all_assigned_perfect_screen_passes_and_has_exact_denominators():
    result = p.assess_predictions(perfect_predictions(), model_available=True)
    assert result["instrument_feasibility_passed"]
    assert [c["assigned"] for c in result["metrics"]["primary_cells"]] == [12] * 6
    assert [c["assigned"] for c in result["metrics"]["nonoverlap_cells"]] == [12, 12, 11, 11, 11, 11]
    assert [c["required"] for c in result["metrics"]["nonoverlap_cells"]] == [11, 11, 10, 10, 10, 10]
    assert result["metrics"]["summaries"]["evaluation"]["all_six_correct_worlds"] == 12


@pytest.mark.parametrize("errors,expected", [(1, True), (2, False)])
def test_exact_cell_integer_boundary(errors, expected):
    assigned = perfect_predictions()
    affected = [r for r in assigned if r["split"] == "evaluation" and r["delivery"] == p.DELIVERIES[0] and r["menu_layout"] == 0]
    for row in affected[:errors]:
        row["predicted_bits"] = [0, 0, 0, 0]
        row["scores"] = [0.0, 0.0, 0.0, 0.0]
    result = p.assess_predictions(assigned, model_available=True)
    assert result["checks"]["primary_exact_sets"] is expected
    assert result["metrics"]["primary_cells"][0]["correct"] == 12 - errors


@pytest.mark.parametrize("k,n,reference,passed", [(1, 24, 18, True), (1, 24, 19, False), (2, 48, 36, True), (2, 48, 37, False)])
def test_shortcut_gain_is_exact_all_assigned_count_difference(k, n, reference, passed):
    assigned = perfect_predictions()
    selected = [r for r in assigned if r["split"] == "evaluation" and sum(r["oracle_bits"]) == k]
    for i, row in enumerate(selected):
        row["shortcut_bits"] = row["oracle_bits"].copy() if i < reference else [0, 0, 0, 0]
    result = p.assess_predictions(assigned, model_available=True)
    cell = next(c for c in result["metrics"]["cardinality_strata"] if c["cardinality"] == k)
    assert cell["assigned"] == n and cell["passed"] is passed


def test_pair_checks_detect_errors_on_different_worlds_despite_passing_each_cell():
    assigned = perfect_predictions()
    for row in assigned:
        if row["split"] == "evaluation" and row["menu_layout"] == 0 and (
            (row["world_slot"] == 0 and row["delivery"] == "decisive_evidence")
            or (row["world_slot"] == 1 and row["delivery"] == "redundant_evidence")
        ):
            row["predicted_bits"] = [0, 0, 0, 0]
            row["scores"] = [0.0, 0.0, 0.0, 0.0]
    result = p.assess_predictions(assigned, model_available=True)
    assert result["checks"]["primary_exact_sets"]
    assert not result["checks"]["evidence_sensitivity"]


def test_missing_evaluation_capture_is_not_dropped_or_filled():
    assigned = perfect_predictions()
    row = assigned[72]
    row.update(available=False, scores=None, predicted_bits=None, prediction_status="capture_unavailable")
    result = p.assess_predictions(assigned, model_available=True)
    assert not result["instrument_feasibility_passed"]
    assert result["metrics"]["summaries"]["evaluation"]["assigned"] == 72
    assert result["metrics"]["primary_cells"][0]["assigned"] == 12
    assert result["metrics"]["summaries"]["evaluation"]["exact_correct"] == 71
    assert all(c["missing"] == 1 for c in result["metrics"]["per_bit"])


def test_missing_training_state_disables_fit_and_retains_all_rows(monkeypatch):
    monkeypatch.setitem(p.METHOD, "features", 8)
    assigned = rows()
    assigned[2]["available"] = False
    X = features(assigned)
    X[2] = np.nan
    monkeypatch.setattr(p, "fit_ridge", lambda *_: pytest.fail("Reduced fit was attempted"))
    model, diagnostics, _, predictions = p.build_predictions(assigned, X)
    assert not model["available"] and diagnostics["reason"] == "missing_training_capture"
    assert len(predictions) == 144 and all(r["predicted_bits"] is None for r in predictions)


def test_degenerate_scale_is_unavailable_not_an_alternate_fit(monkeypatch):
    monkeypatch.setitem(p.METHOD, "features", 8)
    model, diagnostics, _, predictions = p.build_predictions(rows(), np.ones((144, 8)))
    assert not model["available"] and diagnostics["reason"] == "fixed_fit_unavailable"
    assert all(r["prediction_status"] == "model_unavailable" for r in predictions)


def test_original_invalid_response_is_retained_when_state_exists():
    assigned = perfect_predictions()
    assigned[73]["source_flags"].update(valid=False, cap_hit=True)
    result = p.assess_predictions(assigned, model_available=True)
    assert result["metrics"]["summaries"]["evaluation"]["source_invalid"] == 1
    assert result["metrics"]["summaries"]["evaluation"]["source_caps"] == 1
    assert result["metrics"]["summaries"]["evaluation"]["exact_correct"] == 72


@pytest.mark.parametrize("mutation", ["drop", "duplicate", "overlap"])
def test_fixed_denominators_cannot_be_changed(mutation):
    assigned = perfect_predictions()
    if mutation == "drop":
        assigned.pop()
    elif mutation == "duplicate":
        assigned[-1] = deepcopy(assigned[-2])
    else:
        assigned[72]["seen_in_fit"] = True
    with pytest.raises(ValueError):
        p.assess_predictions(assigned, model_available=True)


def test_displayed_oracle_ignores_truth_actions_and_pattern_but_binds_row_labels():
    text = """Candidate | shape | color
W | diamond | teal
X | star | navy
Y | diamond | navy
Z | star | teal
Initial observation:
The shape is diamond.
Additional delivery:
Independent decorative pattern:
* . * .
The hidden candidate does not change.
End of record.
A. stop B. report"""
    first = p.displayed_oracle(text)
    assert first["bits"] == [1, 0, 1, 0]
    changed = text.replace("A. stop B. report", "A. report B. stop").replace("* . * .", ". . . .")
    assert p.displayed_oracle(changed) == first
    swapped = text.replace("W | diamond | teal", "W | star | navy").replace("X | star | navy", "X | diamond | teal")
    second = p.displayed_oracle(swapped)
    assert second["bits"] == [0, 1, 1, 0]
    assert second["visible_problem_sha256"] == first["visible_problem_sha256"]


def test_capture_preparation_reads_header_and_metadata_never_hidden_values(tmp_path, monkeypatch):
    path = tmp_path / "capture.npz"
    np.savez_compressed(path, hidden=np.zeros((32, 4, 7, 5120), dtype=np.float32),
        trial_ids=np.array([str(i) for i in range(32)]), layers=np.array([15, 31, 47, 57]),
        positions=np.array(["record", "prefill", "token_32", "token_128", "token_512", "think_end", "decision"]),
        available=np.ones((32, 4, 7), dtype=bool), token_positions=np.zeros((32, 7), dtype=np.int64),
        closure_counts=np.ones(32, dtype=np.int64))
    original = np.lib.npyio.NpzFile.__getitem__
    def guarded(self, key):
        assert key != "hidden", "Preparation touched hidden feature values"
        return original(self, key)
    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", guarded)
    assert p.capture_metadata(path)["available"].shape == (32, 4, 7)


def setup_execute(monkeypatch, *, missing=False):
    monkeypatch.setitem(p.METHOD, "features", 8)
    assigned = rows()
    if missing:
        assigned[0]["available"] = False
    ledger = dict(rows=assigned, collections=[], code_files={})
    manifest = dict(schema_version=1, assay=p.ASSAY, admitted=True,
                    input_ledger=ledger, input_ledger_sha256=p.digest(ledger), output_relative_path="result")
    monkeypatch.setattr(p, "prepare", lambda _: deepcopy(ledger))
    monkeypatch.setattr(p, "load_features", lambda *_: features(assigned))
    monkeypatch.setattr(p, "verify_bound_files", lambda *_: None)
    return manifest


def test_execute_writes_once_and_binds_all_artifacts(tmp_path, monkeypatch):
    manifest = setup_execute(monkeypatch)
    result = p.execute(tmp_path, manifest, tmp_path / "result")
    assert result["complete"] and result["model_available"]
    assert len(p.read(tmp_path / "result/predictions.json")["rows"]) == 144
    for value in result["artifacts"].values():
        assert p.sha(tmp_path / "result" / value["path"]) == value["byte_sha256"]
    with pytest.raises(FileExistsError):
        p.execute(tmp_path, manifest, tmp_path / "result")


def test_unavailable_fit_writes_all_rows_without_model_artifact(tmp_path, monkeypatch):
    manifest = setup_execute(monkeypatch, missing=True)
    result = p.execute(tmp_path, manifest, tmp_path / "result")
    assert result["complete"] and not result["model_available"]
    assert not result["instrument_feasibility_passed"] and not result["causal_admission"]
    assert not (tmp_path / "result/model.npz").exists()
    assert len(p.read(tmp_path / "result/predictions.json")["rows"]) == 144


@pytest.mark.parametrize("mutation", ["admission", "ledger", "reprepare"])
def test_unadmitted_or_changed_inputs_never_read_features(tmp_path, monkeypatch, mutation):
    manifest = setup_execute(monkeypatch)
    monkeypatch.setattr(p, "load_features", lambda *_: pytest.fail("Premature hidden feature read"))
    if mutation == "admission":
        manifest["admitted"] = False
    elif mutation == "ledger":
        manifest["input_ledger_sha256"] = "wrong"
    else:
        monkeypatch.setattr(p, "prepare", lambda _: {})
    with pytest.raises(ValueError):
        p.execute(tmp_path, manifest, tmp_path / "result")
    assert not (tmp_path / "result").exists()


def test_mid_execution_input_change_preserves_failed_attempt_no_retry(tmp_path, monkeypatch):
    manifest = setup_execute(monkeypatch)
    def fail(*_):
        raise ValueError("Input file changed")
    monkeypatch.setattr(p, "verify_bound_files", fail)
    with pytest.raises(ValueError, match="Input file changed"):
        p.execute(tmp_path, manifest, tmp_path / "result")
    failure = p.read(tmp_path / "result/failure.json")
    assert not failure["complete"] and not failure["instrument_feasibility_passed"]
    assert (tmp_path / "result/started.json").exists()
    assert not (tmp_path / "result/result.json").exists()
    with pytest.raises(FileExistsError):
        p.execute(tmp_path, manifest, tmp_path / "result")


@pytest.mark.parametrize("flag", ["false", 0, 1, None])
def test_nonboolean_admission_flags_fail_closed(flag):
    with pytest.raises(ValueError, match="booleans"):
        p.assess_predictions(perfect_predictions(), model_available=flag)
    with pytest.raises(ValueError, match="booleans"):
        p.assess_predictions(perfect_predictions(), model_available=True, integrity_passed=flag)


def test_manifest_cannot_repeat_fit_under_different_output_path(tmp_path, monkeypatch):
    manifest = setup_execute(monkeypatch)
    monkeypatch.setattr(p, "load_features", lambda *_: pytest.fail("Unbound fit destination"))
    with pytest.raises(ValueError, match="sole admitted destination"):
        p.execute(tmp_path, manifest, tmp_path / "different")


def test_prepare_code_bindings_survive_frozen_relocation_without_absolute_paths(tmp_path, monkeypatch):
    assigned = rows()
    for row in assigned:
        row["visible_problem_sha256"] = f"{row['split']}-{row['world_slot']}-{row['delivery']}"
        if row["delivery"] != p.DELIVERIES[0] and (
            (row["split"] == "fit" and row["world_slot"] == 9)
            or (row["split"] == "evaluation" and row["world_slot"] == 3)
        ):
            row["visible_problem_sha256"] = p.SHARED_VISIBLE
    def collection(_project, version):
        split = "fit" if version == "v1" else "evaluation"
        return dict(version=version, files={}, rows=deepcopy([r for r in assigned if r["split"] == split]))
    monkeypatch.setattr(p, "prepare_collection", collection)
    monkeypatch.setattr(p, "verify_bound_files", lambda *_: None)
    results = []
    for folder in ("live", "frozen"):
        root = tmp_path / folder
        for path in p.CODE_FILES:
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("same frozen bytes:" + path)
        monkeypatch.setattr(p, "__file__", str(root / "scripts/choice_support_readout_offline.py"))
        results.append(p.prepare(tmp_path))
    assert results[0] == results[1]
    assert all(not Path(path).is_absolute() for path in results[0]["code_files"])
    assert results[0]["hidden_feature_values_read"] is False


@pytest.mark.parametrize("mutation", ["missing_bits", "missing_scores", "threshold", "nonfinite", "wrong_status"])
def test_available_prediction_cannot_be_missing_or_disagree_with_scores(mutation):
    assigned = perfect_predictions()
    row = assigned[72]
    if mutation == "missing_bits":
        row["predicted_bits"] = None
    elif mutation == "missing_scores":
        row["scores"] = None
    elif mutation == "threshold":
        row["scores"] = [0.0] * 4
    elif mutation == "nonfinite":
        row["scores"][0] = float("nan")
    else:
        row["prediction_status"] = "model_unavailable"
    with pytest.raises(ValueError):
        p.assess_predictions(assigned, model_available=True)


def test_unavailable_model_cannot_claim_available_predictions():
    with pytest.raises(ValueError, match="Unavailable prediction"):
        p.assess_predictions(perfect_predictions(), model_available=False)
