"""Prospective pure assignment tests with synthetic exclusions/provenance only."""

from collections import Counter, defaultdict
from copy import deepcopy
import json

import pytest

from curiosity_study import satisfaction_screen as screen
from curiosity_study import voluntary_resolution as voluntary
from curiosity_study.artifacts import digest


def direction():
    provenance = {
        key: digest(key)
        for key in (
            "capture_manifest_sha256",
            "capture_audit_sha256",
            "analysis_sha256",
            "analysis_byte_sha256",
            "directions_byte_sha256",
            "independent_check_sha256",
        )
    }
    provenance["selected_sites"] = {"satisfaction_vs_neutral": dict(layer=15, position="record")}
    return dict(
        source="satisfaction_vs_neutral",
        corpus="A",
        layer=15,
        position="record",
        unit=[0.6, 0.8],
        unit_sha256=digest([0.6, 0.8]),
        norm=2.0,
        directions_byte_sha256=provenance["directions_byte_sha256"],
        candidate_source=provenance,
    )


@pytest.fixture(scope="module")
def plan():
    return screen.assignment_plan(historical_exact_world_ids=[], direction_binding=direction())


def test_all1152_assignments_retain24_unique_worlds_and_complete_panel_denominators(plan):
    assert plan == screen.assignment_plan(historical_exact_world_ids=[], direction_binding=direction())
    assert json.loads(json.dumps(plan)) == plan
    assert not plan["scientific_admission"] and plan["eligibility"]["assignment_only"]
    rows, base = plan["assignments"], plan["base_assignments"]
    assert len(rows) == len({row["trial_id"] for row in rows}) == 1152
    assert len(base) == len({row["base_trial_id"] for row in base}) == 288
    assert plan["assignment_sha256"] == digest(rows)
    assert plan["base_assignment_sha256"] == digest(base)
    assert plan["selection_sha256"] == digest(plan["selection"])
    assert len({row["semantic_world_id"] for row in base}) == len({row["world_id"] for row in base}) == 24
    assert {row["family"] for row in rows} == {"objects"}
    assert plan["planned_denominators"]["total_cohorts"] == 36
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["arm"], row["panel_id"]].append(row)
    assert len(grouped) == 8
    for panel in grouped.values():
        assert len(panel) == 144
        assert Counter(row["kind"] for row in panel) == {
            "choice": 72,
            **dict.fromkeys(voluntary.DIAGNOSTICS, 12),
        }
        choices = [row for row in panel if row["kind"] == "choice"]
        assert Counter(row["delivery"] for row in choices) == dict.fromkeys(voluntary.DELIVERIES, 24)
        assert Counter((row["delivery"], row["menu_layout"]) for row in choices) == {
            (delivery, menu): 12 for delivery in voluntary.DELIVERIES for menu in (0, 1)
        }
        assert {row["world_slot"] for row in panel} == set(range(12))


def test_each_arm_reuses_exact_prompts_seeds_rows_and_companions_with_true_zero_sham(plan):
    rows = plan["assignments"]
    unit = direction()
    for arm_index, (arm, multiplier) in enumerate(screen.ARMS):
        current = rows[arm_index * 288 : (arm_index + 1) * 288]
        assert [row["base_row"] for row in current] == list(range(288))
        assert [row["base_trial_id"] for row in current] == [row["base_trial_id"] for row in rows[:288]]
        for index, row in enumerate(current):
            assert row["arm"] == arm and row["intervention_schedule"] == "record_prefill_plus_decode"
            assert row["max_new_tokens"] == 4096
            assert row["sampling_seed"] == rows[index]["sampling_seed"]
            assert row["source_assignment"] == rows[index]["source_assignment"]
            assert screen.inference_request(row) == screen.inference_request(rows[index])
            if multiplier is None:
                assert row["intervention"] is None
            else:
                patch = row["intervention"]
                assert patch["dose"] == multiplier * unit["norm"]
                assert patch["dose_multiplier"] == multiplier
                assert (patch["source"], patch["layer"], patch["position"]) == (
                    "satisfaction_vs_neutral",
                    15,
                    "record",
                )
        for offset in range(0, 288, 32):
            assert [row["base_trial_id"] for row in current[offset : offset + 32]] == [
                row["base_trial_id"] for row in rows[offset : offset + 32]
            ]


def test_source_prompts_oracles_and_v2_label_balance_are_unchanged(plan):
    counts = Counter()
    first_panel_reference = {
        row["trial_id"]: row
        for row in voluntary.assignments(seed=screen.PANELS[0][1])
        if row["family"] == "objects"
    }
    for row in plan["assignments"][:288]:
        source = row["source_assignment"]
        assert screen.prompt(row) == voluntary.prompt(source)
        assert screen.task(row) == voluntary.task(source)
        assert screen.inference_request(row) == voluntary.inference_request(source)
        if row["panel_id"] == "panel_1":
            assert source == first_panel_reference[source["trial_id"]]
        if row["kind"] != "choice":
            counts[row["panel_id"], row["kind"], screen.prompt(row)[2]] += 1
    assert len(counts) == 2 * 6 * 4 and set(counts.values()) == {3}
    for panel_id, _ in screen.PANELS:
        worlds = [world for world in plan["selection"]["worlds"] if world["panel_id"] == panel_id]
        assert Counter(world["known"][0] for world in worlds) == {0: 6, 1: 6}
        assert Counter(world["world_slot"] % 4 for world in worlds) == dict.fromkeys(range(4), 3)


def test_history_and_prospective_duplicates_have_distinct_complete_ledgers(monkeypatch):
    original = screen.state.task
    first_seed = screen.PANELS[0][1]

    # Force one same-panel repeat and one second-panel repeat of a prospective
    # first-panel world. Neither is historical merely because it was selected first.
    def repeated(source):
        changed = dict(source)
        if source["index"] == 1:
            changed["index"] = 0
        if source["seed"] == screen.PANELS[1][1] and source["index"] == 0:
            changed["seed"] = first_seed
        return original(changed)

    monkeypatch.setattr(screen.state, "task", repeated)
    empty = screen.select_object_panels(historical_exact_world_ids=[])
    within = next(
        row
        for row in empty["candidate_exclusion_ledger"]
        if row["panel_id"] == "panel_1" and row["candidate_index"] == 1
    )
    across = next(
        row
        for row in empty["candidate_exclusion_ledger"]
        if row["panel_id"] == "panel_2" and row["candidate_index"] == 0
    )
    assert within["reason"] == "within_panel_duplicate" and not within["accepted"]
    assert across["reason"] == "prior_panel_duplicate" and not across["accepted"]
    assert across["first_selected_location"] == dict(panel_id="panel_1", world_slot=0)
    assert empty["historical_exact_exclusions"] == []
    historical = [empty["worlds"][0]["semantic_world_id"]]
    selected = screen.select_object_panels(historical_exact_world_ids=historical)
    assert selected["historical_exact_exclusions"] == historical
    assert selected["historical_exact_exclusions_sha256"] == digest(historical)
    assert not set(historical) & set(selected["selected_semantic_world_ids"])
    for row in selected["candidate_exclusion_ledger"]:
        if row["semantic_world_id"] in historical:
            assert row["reason"] == "historical_exact_exposure" and row["first_selected_location"] is None
    for panel_id, _ in screen.PANELS:
        ledger = [row for row in selected["candidate_exclusion_ledger"] if row["panel_id"] == panel_id]
        assert [row["candidate_index"] for row in ledger] == list(range(ledger[-1]["candidate_index"] + 1))
        assert sum(row["accepted"] for row in ledger) == 12


def test_insufficient_object_capacity_stops_instead_of_relaxing_or_replacing(monkeypatch):
    original = screen.state.task
    monkeypatch.setattr(screen.state, "task", lambda source: original({**source, "index": 0}))
    with pytest.raises(ValueError, match="exclusions cannot be relaxed"):
        screen.select_object_panels(historical_exact_world_ids=[], max_candidates=4)


@pytest.mark.parametrize(
    "field,value",
    [("source", "curiosity_vs_indifference"), ("layer", 31), ("corpus", "B"), ("norm", -1), ("unit", [1, 1])],
)
def test_no_other_axis_or_refitted_site_can_enter_assignment_plan(field, value):
    binding = direction()
    binding[field] = value
    with pytest.raises(ValueError):
        screen.assignment_plan(historical_exact_world_ids=[], direction_binding=binding)


@pytest.mark.parametrize(
    "field,value",
    [("panel_id", "panel_2"), ("sampling_seed", 0), ("kind", "other"), ("semantic_world_id", "changed")],
)
def test_source_binding_cannot_silently_change_prompt_group_or_sampling(plan, field, value):
    row = deepcopy(next(row for row in plan["assignments"] if row["panel_id"] == "panel_1"))
    row[field] = value
    with pytest.raises(ValueError, match="binding changed"):
        screen.inference_request(row)


def test_plan_owns_its_provenance_and_per_arm_assignments(plan):
    supplied = direction()
    fresh = screen.assignment_plan(historical_exact_world_ids=[], direction_binding=supplied)
    supplied["unit"][0] = 99
    assert fresh == plan
    fresh["assignments"][0]["source_assignment"]["known"][0] = 99
    assert fresh["assignments"][288]["source_assignment"] == plan["assignments"][288]["source_assignment"]
    assert fresh["base_assignments"] == plan["base_assignments"]
