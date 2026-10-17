import copy
import importlib.util
from pathlib import Path
import pytest
from curiosity_study import information_state as state
from curiosity_study import count_competence as competence
from curiosity_study.artifacts import digest, read, write_once
from curiosity_study.causal_contrasts import analyze_contrasts, source_descriptor


def selector(direction):
    return dict(direction=direction, layer=31, position="decision", dose_multiplier=1)


def contrast(name="target_minus_control", **kwargs):
    return dict(
        name=name,
        left=selector("target"),
        right=selector("control"),
        orientation="information_increase",
        **kwargs,
    )


def plan(*specs):
    return dict(alpha=0.05, bootstrap=2000, seed=42, contrasts=list(specs or [contrast()]))


def assignments(items=3, seed=71, template="table", families=("objects", "machines")):
    return [
        a
        for a in state.assignments(items, seed=seed, template=template, kinds=("new",))
        if a["family"] in families
    ]


def compact(assignments, difference=lambda a: 2.0, control=lambda a: 10.0 * a["index"]):
    rows = []
    for a in assignments:
        for arm in ("baseline", "target", "control"):
            value = 0.0 if arm == "baseline" else control(a) + (difference(a) if arm == "target" else 0.0)
            rows.append(
                dict(
                    **source_descriptor(a),
                    trial_id=f"{a['trial_id']}-{arm}",
                    prefix_id=a["trial_id"],
                    direction=arm,
                    arm="baseline" if arm == "baseline" else "direction",
                    layer=None if arm == "baseline" else 31,
                    position=None if arm == "baseline" else "decision",
                    dose_multiplier=0 if arm == "baseline" else 1,
                    margin=float(value),
                    margin_type="semantic_margin",
                    valid=False,
                    correct=False,
                )
            )
    return rows


def test_direct_paired_contrast_cancels_shared_world_variation_and_retains_invalids():
    aa = assignments()
    rows = compact(aa)
    result = analyze_contrasts(rows, aa, plan())
    assert result == analyze_contrasts(rows, aa, plan())
    assert result["planned_endpoints"] == 2
    for endpoint in result["results"]:
        estimate = endpoint["estimate"]
        assert estimate["mean"] == 2
        assert estimate["interval"] == estimate["bonferroni_interval"] == [2, 2]
        assert estimate["bonferroni_alpha"] == 0.025
        assert estimate["bootstrap_degenerate"]
        assert endpoint["totals"]["analyzed_worlds"] == 3
        assert endpoint["totals"]["analyzed_prefixes"] == 12


def test_opposite_information_orientation_and_explicit_gain_strata():
    aa = assignments(families=("objects",))
    rows = compact(aa, difference=lambda a: (1 - 2 * source_descriptor(a)["information_bits"]) * 3)
    specs = [
        contrast(name, stratum=stratum)
        for name, stratum in (
            ("all", None),
            ("gain0", {"information_bits": 0}),
            ("gain1", {"information_bits": 1}),
        )
    ]
    for spec in specs:
        spec["orientation"] = "opposite_information"
    result = analyze_contrasts(rows, aa, plan(*specs))
    assert result["planned_endpoints"] == 3
    for endpoint in result["results"]:
        assert endpoint["estimate"]["mean"] == 3
        assert endpoint["totals"]["complete_paired_world_templates"] == 3
        assert endpoint["totals"]["analyzed_prefixes"] == (12 if endpoint["stratum"] is None else 6)


def test_full_world_completeness_precedes_gain_stratification():
    aa = assignments(families=("objects",))
    missing = next(a for a in aa if a["index"] == 0 and source_descriptor(a)["information_bits"] == 1)
    rows = [r for r in compact(aa) if r["trial_id"] != f"{missing['trial_id']}-control"]
    endpoint = analyze_contrasts(rows, aa, plan(contrast(stratum={"information_bits": 0})))["results"][0]
    assert endpoint["totals"]["paired_stratum_prefixes"] == 6
    assert endpoint["totals"]["analyzed_prefixes"] == 4
    assert endpoint["totals"]["analyzed_worlds"] == 2
    assert endpoint["totals"]["missing_right"] == 1


def test_count_strata_share_the_same_complete_six_cell_worlds():
    aa = [a for a in competence.assignments(items=2) if a["family"] == "objects"]
    rows = compact(aa, difference=lambda a: -a["correct_count"])
    bad = next(a for a in aa if a["index"] == 0 and a["correct_count"] == 4 and a["query"] == 1)
    rows = [r for r in rows if r["trial_id"] != f"{bad['trial_id']}-control"]
    result = analyze_contrasts(
        rows, aa, plan(*[contrast(f"count{count}", stratum={"correct_count": count}) for count in (1, 2, 4)])
    )
    for endpoint in result["results"]:
        assert endpoint["totals"]["analyzed_worlds"] == 1
        assert endpoint["totals"]["analyzed_prefixes"] == 2
        assert endpoint["counts_by_family"]["objects"]["required_cells_per_world_template"] == 6
        assert endpoint["estimate"]["mean"] == -endpoint["stratum"]["correct_count"]


def test_templates_average_within_world_and_distinct_seeds_do_not_merge():
    aa = sum(
        (
            assignments(items=1, seed=seed, template=template, families=("objects",))
            for seed, template in ((7, "table"), (7, "prose"), (8, "table"))
        ),
        [],
    )
    rows = compact(
        aa, difference=lambda a: 3 if (a["seed"], a["template"]) == (7, "table") else 0, control=lambda a: 0
    )
    spec = contrast()
    spec["right"] = {"arm": "baseline"}
    endpoint = analyze_contrasts(rows, aa, plan(spec))["results"][0]
    assert endpoint["totals"]["analyzed_worlds"] == 2
    assert endpoint["totals"]["complete_paired_world_templates"] == 3
    assert endpoint["estimate"]["mean"] == 0.75
    assert endpoint["estimate"]["resampling_worlds_by_family"] == {"objects": 2}
    assert sorted(len(row["templates"]) for row in endpoint["paired_world_values"]) == [1, 2]


def test_pooled_estimate_weights_families_equally_and_includes_missing_endpoints_in_multiplicity():
    aa = assignments(items=1, families=("objects",)) + assignments(items=3, families=("machines",))
    rows = compact(aa, difference=lambda a: 10 if a["family"] == "objects" else 0)
    cfg = plan(contrast(pooled_families=True))
    result = analyze_contrasts(rows, aa, cfg)
    pooled = next(row for row in result["results"] if row["pooled_families"])
    assert result["planned_endpoints"] == 3
    assert pooled["estimate"]["mean"] == 5
    assert pooled["estimate"]["bonferroni_alpha"] == pytest.approx(0.05 / 3)
    rows = [r for r in rows if not (r["family"] == "machines" and r["direction"] == "control")]
    missing = analyze_contrasts(rows, aa, cfg)
    assert missing["planned_endpoints"] == 3
    assert next(r for r in missing["results"] if r["pooled_families"])["estimate"]["mean"] is None
    objects = next(r for r in missing["results"] if r["families"] == ["objects"])
    assert objects["estimate"]["bonferroni_alpha"] == pytest.approx(0.05 / 3)


@pytest.mark.parametrize(
    "problem", ["ambiguous", "nonfinite", "wrong_source", "partial_selector", "small_bootstrap"]
)
def test_invalid_or_ambiguous_inputs_fail_closed(problem):
    aa, cfg = assignments(items=1), plan()
    rows = compact(aa)
    if problem == "ambiguous":
        rows.append({**next(r for r in rows if r["direction"] == "target"), "trial_id": "duplicate_match"})
    elif problem == "nonfinite":
        rows[0]["margin"] = float("nan")
    elif problem == "wrong_source":
        rows[0]["information_bits"] = 1 - rows[0]["information_bits"]
    elif problem == "partial_selector":
        del cfg["contrasts"][0]["left"]["dose_multiplier"]
    elif problem == "small_bootstrap":
        cfg["bootstrap"] = 1999
    with pytest.raises(ValueError):
        analyze_contrasts(rows, aa, cfg)


def script_module():
    path = Path(__file__).resolve().parents[1] / "scripts/analyze_causal_contrasts.py"
    spec = importlib.util.spec_from_file_location("analyze_causal_contrasts_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_confirmation_cannot_override_the_frozen_plan_and_development_must_label_posthoc():
    script = script_module()
    schedule = dict(planned_contrasts=plan())
    manifest = dict(
        inference_scope="confirmation",
        intervention_schedule=schedule,
        intervention_schedule_sha256=digest(schedule),
    )
    assert script.select_plan(manifest) == (schedule["planned_contrasts"], "frozen_parent_manifest")
    with pytest.raises(ValueError, match="only the planned contrasts"):
        script.select_plan(manifest, plan(), True)
    manifest["inference_scope"] = "development"
    with pytest.raises(ValueError, match="--posthoc"):
        script.select_plan(manifest, plan())
    assert script.select_plan(manifest, plan(), True)[1] == "explicit_posthoc_development"


def script_fixture(tmp_path, tamper=False):
    source = tmp_path / "source"
    aa = assignments(families=("objects",))
    source_manifest = dict(diagnostics=aa, assignment_sha256=digest(aa))
    write_once(source / "manifest.json", source_manifest)
    eligible = [a for a in aa if a["index"] != 0]
    excluded = [a["trial_id"] for a in aa if a["index"] == 0]
    rows = compact(eligible)
    for row in rows:
        row["margin_type"] = "information_gain_yes_minus_no_with_polarity"
    assignments_replay = [
        dict(
            trial_id=r["trial_id"],
            prefix_id=r["prefix_id"],
            arm=r["arm"],
            direction_name=r["direction"],
            layer=r["layer"],
            capture_position=r["position"],
            dose_multiplier=r["dose_multiplier"],
        )
        for r in rows
    ]
    schedule = dict(source_collection=str(source), planned_contrasts=plan())
    provenance = dict(evaluation_assignments_sha256=digest(aa))
    manifest = dict(
        assay="replay_information",
        inference_scope="confirmation",
        intervention_schedule=schedule,
        intervention_schedule_sha256=digest(schedule),
        diagnostics=assignments_replay,
        assignment_sha256=digest(assignments_replay),
        source_parent_manifest_sha256=digest(source_manifest),
        prefixes={a["trial_id"]: {"assignment": a} for a in eligible},
        excluded_sources=excluded,
        assigned_sources=len(aa),
        split_provenance=provenance,
        split_provenance_sha256=digest(provenance),
        direction_analysis_sha256="fixed-analysis",
        direction_input={"sha256": "fixed-direction"},
    )
    run, audit = tmp_path / "run", tmp_path / "audit"
    write_once(run / "manifest.json", manifest)
    receipt = dict(
        complete=True,
        integrity_passed=True,
        manifest_sha256=digest(manifest),
        compact_outcomes_sha256=digest(rows),
        assigned_measurements=len(rows),
    )
    if tamper:
        rows = copy.deepcopy(rows)
        rows[0]["margin"] += 100
    write_once(audit / "analysis.json", receipt)
    write_once(audit / "compact_outcomes.json", rows)
    return run, audit


def test_script_requires_audited_compact_hash_and_recovers_excluded_assignments(tmp_path):
    script = script_module()
    run, audit = script_fixture(tmp_path / "good")
    result = script.analyze(run, audit, tmp_path / "output")
    assert result["inference_scope"] == "confirmation" and not result["posthoc"]
    assert result["source_eligibility"]["assigned"] == 12
    assert result["source_eligibility"]["eligible"] == 8
    assert result["source_eligibility"]["excluded"] == 4
    endpoint = result["results"][0]
    assert endpoint["totals"]["assigned_worlds"] == 3 and endpoint["totals"]["analyzed_worlds"] == 2
    assert endpoint["totals"]["missing_both"] == 4
    assert read(tmp_path / "output/analysis.json")["plan_origin"] == "frozen_parent_manifest"
    run, audit = script_fixture(tmp_path / "bad", tamper=True)
    with pytest.raises(ValueError, match="not bound to the passing replay audit"):
        script.analyze(run, audit, tmp_path / "must_not_exist")
    assert not (tmp_path / "must_not_exist").exists()
