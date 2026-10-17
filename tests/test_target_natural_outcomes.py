"""Target gain is an explicit response objective, never an alias for world gain."""

import copy

import pytest

from curiosity_study import target_relevant_information as target
from curiosity_study.artifacts import digest, read
from curiosity_study.causal_contrasts import (
    CATEGORICAL_TYPES,
    RESPONSE_OUTCOME_SCHEMA,
    RESPONSE_OUTCOMES,
    TARGET_OBJECTIVE,
    analyze_contrasts,
    descriptor_response_indicators,
    source_descriptor,
    validate_natural_response_row,
    validate_plan,
)
from curiosity_study.thinking_information import inference_request, record_from_inference, task_module
from curiosity_study.natural_information import source_check
from test_causal_natural_admission import fixture_data, script_module
from test_natural_audit import fixture_collection, modules as natural_modules_fixture, run_audit

modules = natural_modules_fixture


def response_row(a, answer, arm="baseline", valid=True):
    scored = record_from_inference(a, dict(raw=f"Reasoning. </think> {answer}"))
    if not valid:
        scored.update(valid=False, correct=False, parsed_answer=None)
    d = source_descriptor(a)
    values = descriptor_response_indicators(d, scored["parsed_answer"], scored["valid"], scored["correct"])
    return dict(
        **d,
        **values,
        trial_id=f"{a['trial_id']}-{arm}",
        prefix_id=a["trial_id"],
        arm="baseline" if arm == "baseline" else "direction",
        direction=arm,
        layer=None if arm == "baseline" else 31,
        position=None if arm == "baseline" else "prefill",
        dose_multiplier=0 if arm == "baseline" else 1,
        valid=scored["valid"],
        correct=scored["correct"],
        answer=scored["parsed_answer"],
        response_outcome_schema=RESPONSE_OUTCOME_SCHEMA,
        margin=int(scored["correct"])
        if a["kind"] in ("memory", "count")
        else values["valid_informative_response"] - values["valid_uninformative_response"],
        margin_type=CATEGORICAL_TYPES[a["kind"]],
        margin_units="categorical response score; not logits",
    )


def plan(outcome="valid_counterfactual_response", orientation="as_recorded", stratum=None):
    return dict(
        alpha=0.05,
        bootstrap=2000,
        seed=41,
        contrasts=[
            dict(
                name="target_change",
                left=dict(direction="changed", layer=31, position="prefill", dose_multiplier=1),
                right=dict(arm="baseline"),
                outcome=outcome,
                orientation=orientation,
                stratum=stratum,
                kinds=["new", "repeat", "instrument"],
                pooled_families=True,
            )
        ],
    )


def target_rows():
    assignments = target.assignments(items=2, kinds=("new", "repeat", "instrument"))
    rows = []
    for a in assignments:
        d = source_descriptor(a)
        gain = d["expected_target_information_gain_bits"]
        if a["kind"] == "instrument":
            opposite = next(
                k for k, v in d["action_label_mapping"].items() if v == ("stop" if gain else "report")
            )
        else:
            opposite = "No" if target.prompt(a)[2] == "Yes" else "Yes"
        rows += [response_row(a, target.prompt(a)[2]), response_row(a, opposite, "changed")]
    return assignments, rows


def test_target_dispatcher_and_descriptor_keep_both_oracles_explicit():
    for a in target.assignments(items=1):
        assert task_module(a) is target
        assert inference_request(a)["enable_thinking"] is True
        d = source_descriptor(a)
        assert d["cell"] == (a["independent_report"], a["query"])
        assert d["information_bits"] == d["expected_world_information_gain_bits"] == 1
        assert d["expected_target_information_gain_bits"] == int(a["query"] != a["independent_report"])
        assert d["response_objective_basis"] == TARGET_OBJECTIVE
        assert d["target_property"] == target.task(a)["attributes"][2]


def test_every_target_label_mapping_and_polarity_scores_actual_positive_events():
    # Twelve worlds cross both target complements with all six menu mappings.
    for a in target.assignments(items=12):
        d, t = source_descriptor(a), target.task(a)
        for answer in target.prompt(a)[1]:
            row = response_row(a, answer)
            validate_natural_response_row(row)
            if a["kind"] in ("memory", "count"):
                assert all(row[key] is None for key in RESPONSE_OUTCOMES)
                continue
            positive = (
                ("Yes" if a["kind"] == "new" else "No")
                if a["kind"] != "instrument"
                else next(label for label, action in t["action_label_mapping"].items() if action == "report")
            )
            negative = (
                ("No" if a["kind"] == "new" else "Yes")
                if a["kind"] != "instrument"
                else next(label for label, action in t["action_label_mapping"].items() if action == "stop")
            )
            assert row["valid_informative_response"] == int(answer == positive)
            assert row["valid_uninformative_response"] == int(answer == negative)
            assert row["valid_counterfactual_response"] == int(
                answer == (negative if d["expected_target_information_gain_bits"] else positive)
            )
            invalid = response_row(a, answer, valid=False)
            assert all(invalid[key] == 0 for key in RESPONSE_OUTCOMES)


@pytest.mark.parametrize("gain", [0, 1])
def test_target_contrasts_use_target_gain_and_retain_whole_worlds(gain):
    assignments, rows = target_rows()
    selected = {"expected_target_information_gain_bits": gain}
    result = analyze_contrasts(rows, assignments, plan(stratum=selected))
    assert len(result["results"]) == 9
    for endpoint in result["results"]:
        assert endpoint["estimate"]["mean"] == 1
        assert endpoint["response_objective_basis"] == TARGET_OBJECTIVE
        assert endpoint["response_gain_field"] == "expected_target_information_gain_bits"
        assert endpoint["totals"]["binary_response_denominator"] == 4 * len(endpoint["families"])
        assert endpoint["totals"]["analyzed_worlds"] == 2 * len(endpoint["families"])
    signed = analyze_contrasts(rows, assignments, plan("margin", "opposite_target_information", selected))
    assert all(e["estimate"]["mean"] == 2 for e in signed["results"])


@pytest.mark.parametrize("orientation", ["information_increase", "opposite_information"])
def test_target_margin_rejects_ambiguous_world_orientation(orientation):
    assignments, rows = target_rows()
    with pytest.raises(ValueError, match="explicit target-information"):
        analyze_contrasts(rows, assignments, plan("margin", orientation))


def test_target_stratum_never_relabels_world_gain_and_binary_is_not_double_oriented():
    assignments, rows = target_rows()
    with pytest.raises(ValueError, match="explicitly use"):
        analyze_contrasts(rows, assignments, plan(stratum={"information_bits": 0}))
    with pytest.raises(ValueError, match="as_recorded"):
        validate_plan(plan(orientation="opposite_target_information"))
    # Completeness precedes the target-gain stratum: dropping even the other
    # gain's cell removes that entire world, rather than a favorable response.
    drop = next(
        r for r in rows if r["arm"] == "direction" and r["expected_target_information_gain_bits"] == 1
    )
    rows.remove(drop)
    result = analyze_contrasts(rows, assignments, plan(stratum={"expected_target_information_gain_bits": 0}))
    affected = next(
        e for e in result["results"] if e["families"] == [drop["family"]] and e["kind"] == drop["kind"]
    )
    assert affected["totals"]["analyzed_worlds"] == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("information_bits", 0),
        ("expected_target_information_gain_bits", 1),
        ("response_objective_basis", "full_world_information_gain"),
        ("action_label_mapping", dict(A="stop", B="reread", C="report")),
    ],
)
def test_target_scoring_binding_cannot_be_rehashed_into_another_estimand(field, value):
    assignments, rows = target_rows()
    chosen = next(r for r in rows if r["expected_target_information_gain_bits"] == 0)
    chosen[field] = value
    with pytest.raises(ValueError, match="metadata mismatch"):
        analyze_contrasts(rows, assignments, plan())


def test_target_and_world_objectives_cannot_share_natural_analysis(tmp_path):
    manifest, audit, rows = fixture_data(tmp_path)
    a = target.assignments(items=1, kinds=("new",))[0]
    rows.append(response_row(a, target.prompt(a)[2]))
    with pytest.raises(ValueError, match="separate analyses"):
        script_module().admit_outcomes(manifest, audit, rows)


def test_full_target_natural_audit_preserves_prompt_binding_gain_and_controls(tmp_path, modules):
    sources = [a for a in target.assignments(items=1) if a["family"] == "objects"]
    root, tokenizer = fixture_collection(tmp_path, modules, sources=sources, batch_size=32)
    result = run_audit(root, tokenizer, modules, tmp_path / "analysis")
    assert result["integrity_passed"] and result["engineering_passed"]
    assert result["assigned_sources"] == 20 and result["excluded_sources"] == []
    assert result["response_objective_bases"] == [TARGET_OBJECTIVE]
    assert "fixed-answer" in result["target_count_control_note"]
    manifest = read(root / "manifest.json")
    for prefix in manifest["prefixes"].values():
        assert prefix["original_generated_tokens_in_prefix"] == 0
        assert prefix["response_objective"]["information_bits"] == 1
        assert prefix["response_objective"]["expected_target_information_gain_bits"] in (0, 1)
    # A rehashed prompt prefix cannot silently switch the scientific objective.
    altered = copy.deepcopy(next(iter(manifest["prefixes"].values())))
    altered["response_objective"]["expected_target_information_gain_bits"] ^= 1
    with pytest.raises(ValueError, match="exactly the original prompt"):
        source_check(altered, {}, artifact_root=tmp_path, expected_model=manifest["model"])
    compact = read(tmp_path / "analysis/compact_outcomes.json")
    assert all(r["information_bits"] == 1 for r in compact)
    assert {r["expected_target_information_gain_bits"] for r in compact} == {0, 1}
    for metric in result["metrics"]:
        assert metric["required_cells_per_world_template"] == 4
        assert metric["complete_worlds"] == 1
        if metric["stratum"]:
            assert set(metric["stratum"]) == {"expected_target_information_gain_bits"}
        if metric["kind"] in ("memory", "count"):
            assert metric["binary_response_outcomes"]["valid_counterfactual_response"] is None


def test_expected_source_metadata_is_exact_and_legacy_optional(modules):
    check = modules[1].expected_source_metadata
    assignments = target.assignments(items=1)
    criteria = dict(
        minimum_distinct_items_per_cell=20,
        minimum_accuracy=0.9,
        minimum_valid_format=0.95,
        cell_axes=["family", "kind", "independent_report", "query"],
    )
    manifest = dict(
        task_revision=target.REVISION,
        phase="development_validation",
        inference_scope="development",
        assignments=[],
        diagnostics=assignments,
        max_new_tokens=2048,
        validation_criteria=criteria,
    )
    audit = dict(
        assigned=40,
        phase=manifest["phase"],
        inference_scope="development",
        validation_criteria=criteria,
        complete=True,
        gates=dict(passed=True),
    )
    expected = dict(
        task_revision=target.REVISION,
        phase=manifest["phase"],
        inference_scope="development",
        assigned=40,
        max_new_tokens=2048,
        validation_criteria=criteria,
        complete=True,
        global_gates_passed=True,
    )
    check({}, {}, {})
    check(dict(expected_source_metadata=expected), manifest, audit)
    for key in expected:
        wrong = copy.deepcopy(expected)
        wrong[key] = None
        with pytest.raises(ValueError, match="expected_source_metadata"):
            check(dict(expected_source_metadata=wrong), manifest, audit)
    changed = copy.deepcopy(audit)
    changed["validation_criteria"]["minimum_distinct_items_per_cell"] = 100
    with pytest.raises(ValueError, match="expected_source_metadata"):
        check(dict(expected_source_metadata=expected), manifest, changed)
    assert digest(manifest["validation_criteria"]) == digest(criteria)
