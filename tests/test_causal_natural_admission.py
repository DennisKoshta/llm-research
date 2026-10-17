import copy
import importlib.util
from pathlib import Path

import pytest

from curiosity_study import information_state
from curiosity_study.artifacts import digest, read, write_once
from curiosity_study.causal_contrasts import (
    RESPONSE_OUTCOME_SCHEMA,
    natural_response_indicators,
    source_descriptor,
)


def script_module():
    path = Path(__file__).resolve().parents[1] / "scripts/analyze_causal_contrasts.py"
    spec = importlib.util.spec_from_file_location("natural_contrast_admission_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_data(tmp_path, assay="natural_information"):
    source = tmp_path / "source"
    aa = information_state.assignments(items=2, seed=218, kinds=("new",))
    source_manifest = dict(diagnostics=aa, assignment_sha256=digest(aa))
    write_once(source / "manifest.json", source_manifest)
    rows, diagnostics = [], []
    natural = assay == "natural_information"
    position = "prefill" if natural else "decision"
    for sa in aa:
        for arm in ("baseline", "target", "control"):
            a = dict(
                trial_id=f"{sa['trial_id']}-{arm}",
                prefix_id=sa["trial_id"],
                arm="baseline" if arm == "baseline" else "direction",
                direction_name=arm,
                layer=None if arm == "baseline" else 31,
                capture_position=None if arm == "baseline" else position,
                dose_multiplier=0 if arm == "baseline" else 1,
            )
            diagnostics.append(a)
            score = 1 if arm == "target" else -1
            descriptor = source_descriptor(sa)
            rows.append(
                dict(
                    **descriptor,
                    trial_id=a["trial_id"],
                    prefix_id=a["prefix_id"],
                    arm=a["arm"],
                    direction=arm,
                    layer=a["layer"],
                    position=a["capture_position"],
                    dose_multiplier=a["dose_multiplier"],
                    margin=score,
                    margin_type="categorical_information_response_score_not_logit"
                    if natural
                    else "information_gain_yes_minus_no_with_polarity",
                    margin_units="categorical response score; not logits" if natural else "logit difference",
                    valid=True,
                    correct=(score == (1 if descriptor["information_bits"] else -1)),
                )
            )
            if natural:
                row = rows[-1]
                row.update(
                    answer="Yes" if score == 1 else "No",
                    response_outcome_schema=RESPONSE_OUTCOME_SCHEMA,
                )
                row.update(
                    natural_response_indicators(
                        row["kind"],
                        row["index"],
                        row["information_bits"],
                        row["answer"],
                        row["valid"],
                        row["correct"],
                    )
                )
    plan = dict(
        alpha=0.05,
        bootstrap=2000,
        seed=633,
        contrasts=[
            dict(
                name="target_vs_control",
                left=dict(direction="target", layer=31, position=position, dose_multiplier=1),
                right=dict(direction="control", layer=31, position=position, dose_multiplier=1),
                orientation="information_increase",
                pooled_families=True,
            )
        ],
    )
    schedule = dict(source_collection=str(source), planned_contrasts=plan)
    provenance = dict(evaluation_assignments_sha256=digest(aa))
    manifest = dict(
        assay=assay,
        phase="development",
        inference_scope="development",
        model=dict(model_id="Qwen/Qwen3-32B", revision="synthetic"),
        engine_sha256={"fixture": "engine"},
        diagnostics=diagnostics,
        assignment_sha256=digest(diagnostics),
        prefixes={a["trial_id"]: {"assignment": a} for a in aa},
        assigned_sources=len(aa),
        excluded_sources=[],
        source_parent_manifest_sha256=digest(source_manifest),
        split_provenance=provenance,
        split_provenance_sha256=digest(provenance),
        direction_analysis_sha256="direction-analysis",
        direction_input=dict(sha256="direction-file"),
        intervention_schedule=schedule,
        intervention_schedule_sha256=digest(schedule),
    )
    audit = dict(
        assay=assay,
        phase=manifest["phase"],
        model=manifest["model"],
        engine_sha256=manifest["engine_sha256"],
        complete=True,
        integrity_passed=True,
        manifest_sha256=digest(manifest),
        compact_outcomes_sha256=digest(rows),
        assigned_measurements=len(rows),
        eligible_for_interpretation=True,
        interpretation_checks={key: True for key in script_module().NATURAL_READINESS_CHECKS},
        previous_engineering=dict(passed=True),
        baseline_readiness=dict(passed=True),
        response_outcome_schema=RESPONSE_OUTCOME_SCHEMA if natural else None,
    )
    return manifest, audit, rows


def save_fixture(tmp_path, manifest, audit, rows):
    run, directory = tmp_path / "run", tmp_path / "audit"
    audit = {**audit, "manifest_sha256": digest(manifest), "compact_outcomes_sha256": digest(rows)}
    write_once(run / "manifest.json", manifest)
    write_once(directory / "analysis.json", audit)
    write_once(directory / "compact_outcomes.json", rows)
    return run, directory


def test_ready_natural_stats_propagate_categorical_units_to_every_endpoint(tmp_path):
    script = script_module()
    manifest, audit, rows = fixture_data(tmp_path)
    run, audit_dir = save_fixture(tmp_path, manifest, audit, rows)
    result = script.analyze(run, audit_dir, tmp_path / "out")
    assert result["assay"] == "natural_information"
    assert result["outcome_measurement"] == "newly_generated_completed_final_answer"
    assert result["outcome_units"] == "categorical response score; not logits"
    assert result["cross_assay_pooling_permitted"] is False
    assert "natural_audit_sha256" in result and "replay_audit_sha256" not in result
    assert result["planned_endpoints"] == 3
    for endpoint in result["results"]:
        assert endpoint["estimate"]["mean"] == 2
        assert endpoint["estimate"]["units"] == endpoint["outcome_units"] == result["outcome_units"]
        assert endpoint["margin_type"] == "categorical_information_response_score_not_logit"
    assert read(tmp_path / "out/analysis.json")["outcome_units"] == result["outcome_units"]


@pytest.mark.parametrize(
    "problem",
    [
        "not_ready",
        "missing_readiness",
        "false_check",
        "missing_check",
        "missing_prior",
        "null_prior",
        "bad_baseline",
        "engineering",
        "wrong_assay",
        "wrong_model",
        "wrong_engine",
        "missing_model_and_engine",
    ],
)
def test_natural_statistics_require_full_audited_scientific_readiness(tmp_path, problem):
    script = script_module()
    manifest, audit, rows = fixture_data(tmp_path)
    if problem == "not_ready":
        audit["eligible_for_interpretation"] = False
    elif problem == "missing_readiness":
        del audit["eligible_for_interpretation"]
    elif problem == "false_check":
        audit["interpretation_checks"]["prior_32b_engineering_bound"] = False
    elif problem == "missing_check":
        del audit["interpretation_checks"]["complete_source_worlds"]
    elif problem == "missing_prior":
        del audit["previous_engineering"]
    elif problem == "null_prior":
        audit["previous_engineering"] = None
    elif problem == "bad_baseline":
        audit["baseline_readiness"]["passed"] = False
    elif problem == "engineering":
        manifest["phase"] = audit["phase"] = "engineering"
    elif problem == "wrong_assay":
        audit["assay"] = "replay_information"
    elif problem == "wrong_model":
        audit["model"] = {**manifest["model"], "model_id": "Qwen/Qwen3-1.7B"}
    elif problem == "wrong_engine":
        audit["engine_sha256"] = {"fixture": "other-engine"}
    elif problem == "missing_model_and_engine":
        for value in (manifest, audit):
            del value["model"]
            del value["engine_sha256"]
    run, audit_dir = save_fixture(tmp_path, manifest, audit, rows)
    with pytest.raises(ValueError, match="Natural"):
        script.analyze(run, audit_dir, tmp_path / "blocked")
    assert not (tmp_path / "blocked").exists()


def test_explicit_posthoc_option_cannot_bypass_natural_readiness(tmp_path):
    script = script_module()
    manifest, audit, rows = fixture_data(tmp_path)
    audit["eligible_for_interpretation"] = False
    run, audit_dir = save_fixture(tmp_path, manifest, audit, rows)
    write_once(tmp_path / "override.json", manifest["intervention_schedule"]["planned_contrasts"])
    with pytest.raises(ValueError, match="interpretation-readiness"):
        script.analyze(run, audit_dir, tmp_path / "blocked", tmp_path / "override.json", posthoc=True)
    assert not (tmp_path / "blocked").exists()


@pytest.mark.parametrize("assay", ["natural_information", "replay_information"])
def test_internally_consistent_family_pairs_cannot_mix_categorical_and_logit_units(tmp_path, assay):
    script = script_module()
    manifest, audit, rows = fixture_data(tmp_path, assay)
    # Every within-prefix pair still agrees on its type. The dangerous mismatch
    # is between families that the plan would otherwise pool with equal weight.
    for row in rows:
        if row["family"] == "machines":
            row["margin_type"] = (
                "information_gain_yes_minus_no_with_polarity"
                if assay == "natural_information"
                else "categorical_information_response_score_not_logit"
            )
    run, audit_dir = save_fixture(tmp_path, manifest, audit, rows)
    with pytest.raises(ValueError, match="cannot be pooled"):
        script.analyze(run, audit_dir, tmp_path / "blocked")
    assert not (tmp_path / "blocked").exists()


@pytest.mark.parametrize(
    "problem", ["missing_units", "wrong_units", "continuous_value", "invalid_nonzero", "valid_neutral"]
)
def test_natural_categorical_policy_cannot_silently_be_replaced(tmp_path, problem):
    script = script_module()
    manifest, audit, rows = fixture_data(tmp_path)
    if problem == "missing_units":
        del rows[0]["margin_units"]
    elif problem == "wrong_units":
        rows[0]["margin_units"] = "logit difference"
    elif problem == "continuous_value":
        rows[0]["margin"] = 0.25
    elif problem == "invalid_nonzero":
        rows[0]["valid"] = False
    elif problem == "valid_neutral":
        rows[0]["margin"] = 0
    run, audit_dir = save_fixture(tmp_path, manifest, audit, rows)
    with pytest.raises(ValueError, match="categorical|signed"):
        script.analyze(run, audit_dir, tmp_path / "blocked")


def test_replay_keeps_logit_units_and_legacy_receipt_without_assay(tmp_path):
    script = script_module()
    manifest, audit, rows = fixture_data(tmp_path, "replay_information")
    del audit["assay"]
    for row in rows:
        del row["margin_units"]
    run, audit_dir = save_fixture(tmp_path, manifest, audit, rows)
    result = script.analyze(run, audit_dir, tmp_path / "out")
    assert result["outcome_units"] == "logit difference"
    assert "replay_audit_sha256" in result and "natural_audit_sha256" not in result
    assert all(r["estimate"]["units"] == "logit difference" for r in result["results"])
    # A declared categorical unit is rejected even with a replay type name.
    bad = copy.deepcopy(rows)
    bad[0]["margin_units"] = "categorical response score; not logits"
    with pytest.raises(ValueError, match="categorical outcome units"):
        script.admit_outcomes(manifest, audit, bad)
