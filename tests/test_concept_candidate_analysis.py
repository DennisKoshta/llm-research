"""Synthetic counterexamples for locked corpus-A selection and corpus-B screening.

No authored corpus, real model, or actual capture collection is read here.
"""

import copy
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from curiosity_study.artifacts import RunStore, digest, read, write_once


@pytest.fixture
def analyzer():
    path = Path(__file__).resolve().parents[1] / "scripts/analyze_concept_candidates.py"
    spec = importlib.util.spec_from_file_location("synthetic_concept_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def paired(delta, n=12, center=None):
    delta = np.asarray(delta, dtype=float)
    if delta.ndim == 1:
        delta = np.repeat(delta[None], n, axis=0)
    center = np.zeros_like(delta) if center is None else np.broadcast_to(center, delta.shape)
    return np.stack([center + delta / 2, center - delta / 2], axis=1)


def synthetic(analyzer, dimensions=10):
    assignments, hidden = [], []
    for corpus in ("A", "B"):
        for scenario in range(12):
            for ci, contrast in enumerate(analyzer.CONTRASTS):
                for member, sign in (("positive", 1), ("negative", -1)):
                    key = f"{corpus}-{scenario:02d}-{contrast}-{member}"
                    assignments.append(
                        dict(
                            trial_id=key,
                            corpus_id=corpus,
                            scenario_id=f"{corpus}-scene-{scenario:02d}",
                            contrast=contrast,
                            polarity=member,
                        )
                    )
                    value = np.zeros((4, 2, dimensions), dtype=float)
                    value[:, :, ci] = sign  # Full contrast length is two.
                    hidden.append(value)
    return assignments, np.asarray(hidden)


def target_row(result, target="curiosity_vs_indifference"):
    return next(row for row in result["candidates"] if row["contrast"] == target)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_paired_fit_exports_full_difference_norm_center_and_threshold(analyzer):
    values = np.repeat(np.array([[[7.0, 4.0], [1.0, -4.0]]]), 12, axis=0)
    fit = analyzer.paired_fit(values)
    np.testing.assert_allclose(fit["contrast"], [6, 8])
    np.testing.assert_allclose(fit["center"], [4, 0])
    np.testing.assert_allclose(fit["unit"], [0.6, 0.8])
    assert fit["norm"] == pytest.approx(10)  # Half-contrast scaling would incorrectly give five.
    assert fit["threshold"] == pytest.approx(2.4)


def test_private_scenario_features_expose_in_sample_overfit_with_grouped_loso(analyzer):
    values = paired(np.eye(12))
    full = analyzer.paired_fit(values)
    assert analyzer.project_metrics(values, full["unit"], full["threshold"])["concordance"] == 1
    heldout = analyzer.leave_scenario_out(values)
    assert heldout["available"]
    assert heldout["concordance"] == 0.5
    assert heldout["mean_gap"] == 0 and heldout["gaps"] == [0.0] * 12


def test_layer_choice_is_grouped_a_record_only_and_ties_choose_earlier(analyzer):
    values = np.zeros((12, 2, 4, 2, 2))
    values[:, :, 0, 0] = paired(np.array([[2.0, 0.0]] * 8 + [[-1.0, 0.0]] * 4))
    values[:, :, 1, 0] = paired([1.0, 0.0])
    values[:, :, 2, 0] = paired([50.0, 0.0])
    values[:, :, 3, 0] = paired([100.0, 0.0])
    values[:, :, :, 1] = np.broadcast_to(paired([1000.0, 0.0])[:, :, None], (12, 2, 4, 2))
    selected, checks = analyzer.select_layer(values)
    assert analyzer.LAYERS[selected] == 31
    assert checks[0]["concordance"] == pytest.approx(8 / 12)
    assert [row["concordance"] for row in checks[1:]] == [1, 1, 1]


def test_zero_axis_in_any_training_fold_is_unavailable(analyzer):
    values = paired(np.array([[2.0, 0.0]] + [[0.0, 0.0]] * 11))
    assert analyzer.paired_fit(values)["available"]
    cv = analyzer.leave_scenario_out(values)
    assert not cv["available"] and cv["concordance"] is None
    assert not analyzer.paired_fit(paired([0.0, 0.0]))["available"]


def test_b_sign_failure_never_changes_layer_position_orientation_or_a_fit(analyzer):
    rows, hidden = synthetic(analyzer)
    _, reference_axes = analyzer.analyze_arrays(rows, hidden)
    for row, value in zip(rows, hidden):
        if row["corpus_id"] == "B" and row["contrast"] == analyzer.TARGETS[0]:
            value[0, 0] *= -1
            value[1:, 0] *= 100  # Tempting later B sites must not repair the selected site.
    result, axes = analyzer.analyze_arrays(rows, hidden)
    candidate = target_row(result)
    assert candidate["selected_layer"] == 15 and candidate["position"] == "record"
    assert candidate["a_selection_passed"] and not candidate["b_transfer_passed"]
    assert not candidate["passed"] and candidate["b_target"]["concordance"] == 0
    assert candidate["independent_b_direction"]["a_b_cosine"] == pytest.approx(-1)
    for name in axes:
        np.testing.assert_array_equal(axes[name], reference_axes[name])
    assert any(
        row["source"] == analyzer.TARGETS[0]
        and row["evaluated_contrast"] == analyzer.TARGETS[0]
        and row["layer"] == 15
        and row["position"] == "final"
        and row["concordance"] == 1
        for row in result["all_site_metrics"]
    )


@pytest.mark.parametrize("control_gap", [2.0, 3.0, -3.0])
def test_large_or_equal_nuisance_effect_fails_despite_perfect_target_ranking(analyzer, control_gap):
    rows, hidden = synthetic(analyzer)
    for row, value in zip(rows, hidden):
        if row["corpus_id"] == "B" and row["contrast"] == analyzer.CONTROLS[0]:
            value[:, :, 0] = control_gap / 2 * (1 if row["polarity"] == "positive" else -1)
    result, _ = analyzer.analyze_arrays(rows, hidden)
    candidate = target_row(result)
    assert candidate["b_target"]["auc"] == candidate["b_target"]["concordance"] == 1
    assert candidate["b_transfer_passed"] and not candidate["b_specificity_point_screen_passed"]
    assert not candidate["passed"]


def test_fixed_a_threshold_accuracy_is_reported_but_not_an_admission_gate(analyzer):
    rows, hidden = synthetic(analyzer)
    for row, value in zip(rows, hidden):
        if row["corpus_id"] == "B":
            value[:, :, :2] += 100
    result, axes = analyzer.analyze_arrays(rows, hidden)
    for candidate in result["candidates"]:
        assert candidate["passed"]
        assert candidate["a_threshold"] == 0
        assert candidate["b_target"]["accuracy"] == 0.5
        assert candidate["b_target"]["auc"] == candidate["b_target"]["concordance"] == 1
    np.testing.assert_array_equal(axes["threshold"], 0)


def test_specificity_bootstrap_preserves_scenarios_and_uses_absolute_mean_control(analyzer):
    target = np.arange(1.0, 13.0)
    controls = np.zeros((6, 12))
    controls[0] = target - 1
    controls[1] = [-100.0, 100.0] * 6
    result = analyzer.specificity_intervals(target, controls)
    assert analyzer.BOOTSTRAPS == 10000 and analyzer.SEED == 2026101431
    # A shared scenario resample keeps this contrast exactly one. Independent
    # target/control resamples would produce a spuriously wide interval.
    assert result[0]["target_minus_absolute_control"] == 1
    np.testing.assert_allclose(result[0]["ci95"], [1, 1], rtol=0, atol=1e-14)
    assert result[1]["target_minus_absolute_control"] == target.mean()
    assert result[1]["target_minus_absolute_control"] != target.mean() - np.abs(controls[1]).mean()
    ix = np.random.default_rng(2026101431).integers(0, 12, (10000, 12))
    expected = target[ix].mean(1) - abs(controls[1, ix].mean(1))
    np.testing.assert_allclose(result[1]["ci95"], np.quantile(expected, [0.025, 0.975]))
    assert result == analyzer.specificity_intervals(target, controls)


def test_pca_uses_fixed_50_percent_rule_full_contrast_and_zero_residual_unavailable(analyzer):
    controls = np.array(
        [[3.0, 0, 0], [-3.0, 0, 0], [0, 2.0, 0], [0, -2.0, 0], [0, 0, 1.0], [0, 0, -1.0]]
    ).reshape(3, 2, 3)
    fit = analyzer.denoised_fit(paired([1.0, 2.0, 0.0]), controls)
    assert fit["available"] and fit["components"] == 1
    assert fit["variance_fraction"] == pytest.approx(9 / 14)
    assert fit["retained_norm_ratio"] == pytest.approx(2 / np.sqrt(5))
    np.testing.assert_allclose(fit["unit"], [0, 1, 0], atol=1e-14)
    zero = analyzer.denoised_fit(paired([1.0, 0.0, 0.0]), controls)
    assert not zero["available"] and zero["retained_norm_ratio"] < 1e-12
    np.testing.assert_array_equal(zero["unit"], [0, 0, 0])


def test_pca_success_cannot_rescue_a_failed_raw_candidate(analyzer):
    rows, hidden = synthetic(analyzer)
    for row, value in zip(rows, hidden):
        sign = 1 if row["polarity"] == "positive" else -1
        if row["contrast"] == analyzer.TARGETS[0]:
            value[:] = 0
            value[:, :, :2] = np.array([1.0, 1.0] if row["corpus_id"] == "A" else [-3.0, 2.0]) * sign / 2
        if row["corpus_id"] == "A" and row["contrast"] == analyzer.CONTROLS[0]:
            value[:] = 0
            value[:, :, 0] = 100 * sign
    result, _ = analyzer.analyze_arrays(rows, hidden)
    candidate = target_row(result)
    assert candidate["b_target"]["concordance"] == 0
    assert candidate["pca50_sensitivity"]["available"]
    assert candidate["pca50_sensitivity"]["b_target"]["concordance"] == 1
    assert not candidate["b_transfer_passed"] and not candidate["passed"]


@pytest.mark.parametrize(
    "positive_pairs,negative_gap,expected", [(10, -1.0, True), (9, -1.0, False), (10, -5.0, False)]
)
def test_b_ordering_and_strict_positive_mean_are_separate_gates(
    analyzer, positive_pairs, negative_gap, expected
):
    rows, hidden = synthetic(analyzer)
    for row, value in zip(rows, hidden):
        if row["corpus_id"] == "B" and row["contrast"] == analyzer.TARGETS[0]:
            scenario = int(row["scenario_id"].rsplit("-", 1)[1])
            gap = 1 if scenario < positive_pairs else negative_gap
            value[:, :, 0] = gap / 2 * (1 if row["polarity"] == "positive" else -1)
    result, _ = analyzer.analyze_arrays(rows, hidden)
    candidate = target_row(result)
    assert candidate["b_target"]["concordance"] == pytest.approx(positive_pairs / 12)
    assert candidate["b_transfer_passed"] is expected
    assert candidate["passed"] is expected


def test_zero_axes_remain_in_all_site_export_with_assigned_denominators(analyzer):
    rows, hidden = synthetic(analyzer)
    hidden[:] = 0
    result, axes = analyzer.analyze_arrays(rows, hidden)
    assert len(result["all_site_metrics"]) == 8 * 8 * 4 * 2
    for row in result["all_site_metrics"]:
        assert not row["available"] and row["assigned_worlds"] == 12
        assert row["concordance"] is None and row["mean_gap"] is None
    assert all(c["selected_layer"] is None and not c["passed"] for c in result["candidates"])
    np.testing.assert_array_equal(axes["contrast_norm"], 0)


def test_paired_uncertainty_matches_independent_whole_scenario_calculation(analyzer):
    gaps = np.array([-2.0, 0.0, 1.0, 3.0, 6.0, 7.0, 8.0, 8.0, 10.0, 12.0, 15.0, 30.0])
    result = analyzer.paired_intervals(gaps)
    ix = np.random.default_rng(analyzer.SEED).integers(0, 12, (analyzer.BOOTSTRAPS, 12))
    samples = gaps[ix]
    np.testing.assert_allclose(result["mean_gap_ci95"], np.quantile(samples.mean(1), [0.025, 0.975]))
    np.testing.assert_allclose(
        result["concordance_ci95"],
        np.quantile(((samples > 0) + 0.5 * (samples == 0)).mean(1), [0.025, 0.975]),
    )
    assert "conditional" in result["role"].lower() and "no fitting/selection uncertainty" in result["role"]


def test_pca_fits_a_only_and_removes_the_entire_heldout_control_scenario(analyzer, monkeypatch):
    rows, hidden = synthetic(analyzer)
    for row, value in zip(rows, hidden):
        if row["corpus_id"] == "A" and row["contrast"] in analyzer.CONTROLS:
            value[:, :, 8] = 10 * int(row["scenario_id"].rsplit("-", 1)[1])
        if row["corpus_id"] == "B":
            value[:, :, 9] = 1_000_000  # Forbidden B data are unmistakable in a PCA fit.
    pairs, _ = analyzer.build_pairs(rows, hidden)
    expected = np.stack([pairs[("A", c)][:, :, 0, 0] for c in analyzer.CONTROLS], axis=1)
    fitted_controls = []
    actual = analyzer.denoised_fit

    def observe(target, controls):
        fitted_controls.append(controls.copy())
        return actual(target, controls)

    monkeypatch.setattr(analyzer, "denoised_fit", observe)
    analyzer.analyze_arrays(rows, hidden)
    assert len(fitted_controls) == 2 * 13
    for offset in (0, 13):
        np.testing.assert_array_equal(fitted_controls[offset], expected)
        for heldout in range(12):
            np.testing.assert_array_equal(
                fitted_controls[offset + heldout + 1], np.delete(expected, heldout, axis=0)
            )


@pytest.mark.parametrize("change", ["missing_text", "duplicate_member", "missing_contrast", "unknown_corpus"])
def test_missing_or_duplicated_members_are_rejected_before_fitting(analyzer, change):
    rows, hidden = synthetic(analyzer)
    if change == "missing_text":
        rows, hidden = rows[:-1], hidden[:-1]
    elif change == "duplicate_member":
        rows[-1] = copy.deepcopy(rows[-2])
    elif change == "missing_contrast":
        rows[-1]["contrast"] = "unplanned_contrast"
    else:
        rows[-1]["corpus_id"] = "C"
    with pytest.raises(ValueError):
        analyzer.analyze_arrays(rows, hidden)


def artifact_fixture(tmp_path, analyzer):
    rows, hidden = synthetic(analyzer)
    plan = tmp_path / "curiosity-satisfaction-candidate-pilot-plan.md"
    plan.write_text("Synthetic prospective methods binding, not real corpus material.\n")
    manifest = dict(
        diagnostics=rows,
        assignments=[],
        assignment_sha256=digest(rows),
        capture_plan=dict(layers=analyzer.LAYERS, positions=analyzer.POSITIONS),
        model=dict(model_id="synthetic", revision="fixture"),
        source_sha256="synthetic-source",
        document_bindings=[
            dict(path="reports/curiosity-satisfaction-candidate-pilot-plan.md", byte_sha256=sha(plan))
        ],
    )
    run = tmp_path / "capture"
    store = RunStore(run, manifest)
    capture = run / "activations.npz"
    prefixes = [digest([i]) for i in range(len(rows))]
    np.savez_compressed(
        capture,
        hidden=hidden,
        available=np.ones((384, 4, 2), dtype=bool),
        trial_ids=[r["trial_id"] for r in rows],
        layers=analyzer.LAYERS,
        positions=analyzer.POSITIONS,
        token_prefix_sha256=prefixes,
    )
    records = [
        dict(
            trial_id=a["trial_id"],
            assignment=a,
            terminal=True,
            tokenization=dict(token_prefix_sha256=prefixes[i]),
            measurement=dict(token_prefix_sha256=prefixes[i]),
            activation_artifact=dict(path=capture.name, sha256=sha(capture), row=i),
        )
        for i, a in enumerate(rows)
    ]
    store.save_many(records, "synthetic", "all")
    bindings = [
        dict(
            trial_id=r["trial_id"],
            record_sha256=digest(r),
            artifact_sha256=r["activation_artifact"]["sha256"],
            row=r["activation_artifact"]["row"],
            token_prefix_sha256=r["tokenization"]["token_prefix_sha256"],
        )
        for r in records
    ]
    audit = dict(
        passed=True,
        complete=True,
        assigned=384,
        manifest_sha256=digest(manifest),
        model=manifest["model"],
        source_sha256=manifest["source_sha256"],
        capture_plan=manifest["capture_plan"],
        record_bindings_sha256=digest(bindings),
    )
    audit_path = tmp_path / "audit.json"
    write_once(audit_path, audit)
    return dict(
        run=run,
        manifest=manifest,
        capture=capture,
        hidden=hidden,
        audit=audit,
        audit_path=audit_path,
        plan=plan,
        records=records,
    )


def test_export_binds_inputs_direction_bytes_and_refuses_existing_output(analyzer, tmp_path):
    f = artifact_fixture(tmp_path, analyzer)
    out = tmp_path / "analysis"
    result = analyzer.main(f["run"], f["audit_path"], f["plan"], out)
    assert result["manifest_sha256"] == digest(f["manifest"])
    assert result["audit_sha256"] == digest(f["audit"])
    assert result["plan_byte_sha256"] == sha(f["plan"])
    assert result["analyzer_byte_sha256"] == sha(analyzer.__file__)
    assert result["directions_sha256"] == sha(out / "directions.npz")
    with np.load(out / "directions.npz", allow_pickle=False) as z:
        np.testing.assert_array_equal(z["contrast_norm"], 2)
        assert z["unit"].shape == (8, 4, 2, 10)
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    with pytest.raises(ValueError, match="immutable"):
        analyzer.main(f["run"], f["audit_path"], f["plan"], out)
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before


def test_analyzer_rejects_stale_audit_after_record_and_capture_are_rehashed(analyzer, tmp_path):
    f = artifact_fixture(tmp_path, analyzer)
    with np.load(f["capture"], allow_pickle=False) as z:
        contents = {key: z[key] for key in z.files}
    contents["hidden"][0, 0, 0, 0] += 100
    np.savez_compressed(f["capture"], **contents)
    envelope_path = f["run"] / "cohorts/synthetic/all.json"
    envelope = read(envelope_path)
    for record in envelope["records"]:
        record["activation_artifact"]["sha256"] = sha(f["capture"])
    envelope["sha256"] = digest(envelope["records"])
    # Deliberate tampering, confined to synthetic temporary artifacts.
    import json

    envelope_path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="(?i)audit|binding"):
        analyzer.load_captures(f["run"], f["audit"])


def test_analyzer_rejects_plan_bytes_not_bound_to_capture_manifest(analyzer, tmp_path):
    f = artifact_fixture(tmp_path, analyzer)
    f["plan"].write_text("A replacement plan supplied after capture.\n")
    with pytest.raises(ValueError, match="(?i)plan|document|binding"):
        analyzer.main(f["run"], f["audit_path"], f["plan"], tmp_path / "analysis")
    assert not (tmp_path / "analysis").exists()
