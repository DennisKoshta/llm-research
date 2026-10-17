"""Valid counterfactual evidence cannot be manufactured by damaging completions."""

import copy
import itertools

import pytest

from curiosity_study import information_state
from curiosity_study.artifacts import digest
from curiosity_study.causal_contrasts import (
    CATEGORICAL_TYPES,
    RESPONSE_OUTCOME_SCHEMA,
    RESPONSE_OUTCOMES,
    analyze_contrasts,
    natural_response_indicators,
    source_descriptor,
    validate_plan,
)
from test_causal_natural_admission import fixture_data, save_fixture, script_module


@pytest.mark.parametrize("kind", ["new", "repeat"])
@pytest.mark.parametrize("gain", [0, 1])
def test_reversed_polarities_and_invalidation_never_increase_counterfactual(kind, gain):
    positive = "Yes" if kind == "new" else "No"
    for answer in ("Yes", "No"):
        correct = (answer == positive) == bool(gain)
        scored = natural_response_indicators(kind, 0, gain, answer, True, correct)
        assert scored["valid_informative_response"] == int(answer == positive)
        assert scored["valid_uninformative_response"] == int(answer != positive)
        assert scored["valid_counterfactual_response"] == int(not correct)
        invalid = natural_response_indicators(kind, 0, gain, answer, False, False)
        assert set(invalid.values()) == {0}
        assert invalid["valid_counterfactual_response"] <= scored["valid_counterfactual_response"]


@pytest.mark.parametrize("gain", [0, 1])
def test_instrument_all_menu_permutations_distinguish_report_stop_and_distractor(gain):
    for index, services in enumerate(itertools.permutations(("report", "pattern", "stop"))):
        for label, service in zip("ABC", services):
            correct = service == ("report" if gain else "stop")
            scored = natural_response_indicators("instrument", index, gain, label, True, correct)
            assert scored["valid_informative_response"] == int(service == "report")
            assert scored["valid_uninformative_response"] == int(service == "stop")
            assert scored["valid_counterfactual_response"] == int(service == ("stop" if gain else "report"))
            if service == "pattern":
                assert set(scored.values()) == {0}
            assert set(
                natural_response_indicators("instrument", index, gain, label, False, False).values()
            ) == {0}


@pytest.mark.parametrize("kind", ["memory", "count"])
def test_competence_controls_do_not_invent_counterfactual_semantics(kind):
    assert natural_response_indicators(kind, 0, 0, "A", True, True) == dict.fromkeys(RESPONSE_OUTCOMES)


def degradation_fixture():
    assignments = [
        a for a in information_state.assignments(items=2, kinds=("new", "repeat")) if a["family"] == "objects"
    ]
    rows = []
    for a in assignments:
        d = source_descriptor(a)
        answer = information_state.prompt(a)[2]
        for arm in ("baseline", "damaged"):
            valid = arm == "baseline"
            scores = natural_response_indicators(
                a["kind"], a["index"], d["information_bits"], answer if valid else None, valid, valid
            )
            rows.append(
                dict(
                    **d,
                    trial_id=f"{a['trial_id']}-{arm}",
                    prefix_id=a["trial_id"],
                    arm="baseline" if valid else "direction",
                    direction=arm,
                    layer=None if valid else 31,
                    position=None if valid else "prefill",
                    dose_multiplier=0 if valid else 1,
                    valid=valid,
                    correct=valid,
                    answer=answer if valid else None,
                    response_outcome_schema=RESPONSE_OUTCOME_SCHEMA,
                    **scores,
                    margin=scores["valid_informative_response"] - scores["valid_uninformative_response"],
                    margin_type=CATEGORICAL_TYPES[a["kind"]],
                    margin_units="categorical response score; not logits",
                )
            )
    return assignments, rows


def binary_plan(outcome="valid_counterfactual_response"):
    return dict(
        alpha=0.05,
        bootstrap=2000,
        seed=38,
        contrasts=[
            dict(
                name="damaged_versus_baseline",
                left=dict(direction="damaged", layer=31, position="prefill", dose_multiplier=1),
                right=dict(arm="baseline"),
                outcome=outcome,
                orientation="as_recorded",
            )
        ],
    )


def test_invalidating_every_response_cannot_create_counterfactual_effect_and_keeps_denominators():
    assignments, rows = degradation_fixture()
    plan = binary_plan()
    for outcome in ("valid_counterfactual_response", "valid", "correct"):
        result = analyze_contrasts(rows, assignments, binary_plan(outcome))
        for endpoint in result["results"]:
            assert endpoint["estimate"]["mean"] == (0 if outcome == "valid_counterfactual_response" else -1)
            assert (
                endpoint["totals"]["assigned_prefixes"]
                == endpoint["totals"]["binary_response_denominator"]
                == 8
            )
            assert endpoint["totals"]["analyzed_worlds"] == 2
    # Preserve the old score for transparent comparison: it has the failure mode
    # motivating the explicit binary primary outcome, despite identical data.
    legacy = copy.deepcopy(plan)
    legacy["contrasts"][0].update(outcome="margin", orientation="opposite_information")
    assert all(e["estimate"]["mean"] == 1 for e in analyze_contrasts(rows, assignments, legacy)["results"])


def test_actual_valid_counterfactual_answers_increase_binary_outcome_in_both_gain_strata():
    assignments, rows = degradation_fixture()
    for row in rows:
        if row["arm"] == "baseline":
            continue
        positive = "Yes" if row["kind"] == "new" else "No"
        negative = "No" if row["kind"] == "new" else "Yes"
        row.update(valid=True, correct=False, answer=positive if row["information_bits"] == 0 else negative)
        row.update(
            natural_response_indicators(
                row["kind"], row["index"], row["information_bits"], row["answer"], True, False
            )
        )
        row["margin"] = row["valid_informative_response"] - row["valid_uninformative_response"]
    plan = binary_plan()
    base = plan["contrasts"][0]
    plan["contrasts"] = [
        {**base, "name": f"gain{gain}", "stratum": {"information_bits": gain}} for gain in (0, 1)
    ]
    for endpoint in analyze_contrasts(rows, assignments, plan)["results"]:
        assert endpoint["estimate"]["mean"] == 1
        assert (
            endpoint["totals"]["assigned_stratum_prefixes"]
            == endpoint["totals"]["binary_response_denominator"]
            == 4
        )
        assert endpoint["totals"]["analyzed_left_positive_responses"] == 4
        assert endpoint["totals"]["analyzed_right_positive_responses"] == 0


def test_unsupported_counterfactual_kind_fails_before_any_outcome_is_present():
    assignments = information_state.assignments(items=1, kinds=("memory",))
    with pytest.raises(ValueError, match="unsupported for memory/count"):
        analyze_contrasts([], assignments, binary_plan())


@pytest.mark.parametrize("outcome", [*RESPONSE_OUTCOMES, "valid", "correct"])
@pytest.mark.parametrize("orientation", ["information_increase", "opposite_information"])
def test_binary_outcomes_cannot_receive_an_additional_orientation(outcome, orientation):
    plan = binary_plan(outcome)
    plan["contrasts"][0]["orientation"] = orientation
    with pytest.raises(ValueError, match="as_recorded"):
        validate_plan(plan)


def test_binary_script_propagates_probability_units_and_assigned_denominators(tmp_path):
    manifest, audit, rows = fixture_data(tmp_path)
    spec = manifest["intervention_schedule"]["planned_contrasts"]["contrasts"][0]
    spec.update(outcome="valid_informative_response", orientation="as_recorded")
    manifest["intervention_schedule_sha256"] = digest(manifest["intervention_schedule"])
    run, directory = save_fixture(tmp_path, manifest, audit, rows)
    result = script_module().analyze(run, directory, tmp_path / "out")
    assert result["response_outcome_schema"] == RESPONSE_OUTCOME_SCHEMA
    assert result["outcome_units"].startswith("probability difference")
    for endpoint in result["results"]:
        assert endpoint["outcome"] == "valid_informative_response"
        assert endpoint["estimate"]["mean"] == 1
        assert not endpoint["legacy_descriptive_margin"]
        assert endpoint["totals"]["binary_response_denominator"] == endpoint["totals"]["assigned_prefixes"]


@pytest.mark.parametrize(
    "problem", ["fake_binary", "fake_margin", "missing_schema", "wrong_label", "wrong_truth", "binary_type"]
)
def test_rehashed_family_level_scoring_mismatch_cannot_be_pooled(tmp_path, problem):
    manifest, audit, rows = fixture_data(tmp_path)
    for row in rows:
        if row["family"] != "machines":
            continue
        if problem == "fake_binary":
            row["valid_counterfactual_response"] = 1 - row["valid_counterfactual_response"]
        elif problem == "fake_margin":
            row["margin"] *= -1
        elif problem == "missing_schema":
            del row["response_outcome_schema"]
        elif problem == "wrong_label":
            row["answer"] = "Yes" if row["answer"] == "No" else "No"
        elif problem == "wrong_truth":
            row["correct"] = not row["correct"]
        else:
            row["valid_counterfactual_response"] = bool(row["valid_counterfactual_response"])
    run, directory = save_fixture(tmp_path, manifest, audit, rows)
    with pytest.raises(ValueError, match="Natural"):
        script_module().analyze(run, directory, tmp_path / "blocked")
