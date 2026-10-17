from collections import Counter, defaultdict
from copy import deepcopy

import pytest

from curiosity_study import information_state as state
from curiosity_study import voluntary_resolution as voluntary
from curiosity_study.artifacts import digest


@pytest.fixture(scope="module")
def plan():
    return voluntary.assignment_plan()


def choice_for_service(a, service):
    label = next(k for k, v in voluntary.task(a)["action_mapping"].items() if v == service)
    return voluntary.record_from_thinking_inference(
        a, {"raw": f"The options are understood.</think> {label}.", "ended_with_eos": True}
    )


def test_complete_prospective_inventory_and_semantic_world_selection(plan):
    assert plan == voluntary.assignment_plan()
    assert plan["task_revision"] == "voluntary-resolution-v2"
    rows = plan["assignments"]
    assert len(rows) == len({a["trial_id"] for a in rows}) == 288
    assert plan["assignment_sha256"] == digest(rows)
    assert plan["selection_sha256"] == digest(plan["selection"])
    assert sum(a["kind"] == "choice" for a in rows) == 144
    worlds = defaultdict(list)
    for a in rows:
        worlds[a["family"], a["world_slot"]].append(a)
        assert a["seed"] == 2026101435 and a["max_new_tokens"] == 4096
        assert a["vocabulary"] == "heldout"
    assert len(worlds) == 24
    for aa in worlds.values():
        assert len(aa) == 12
        assert len({a["index"] for a in aa}) == len({a["semantic_world_id"] for a in aa}) == 1
        assert {(a["delivery"], a["menu_layout"]) for a in aa if a["kind"] == "choice"} == {
            (delivery, menu) for delivery in voluntary.DELIVERIES for menu in (0, 1)
        }
        assert Counter(a["kind"] for a in aa) == {"choice": 6, **dict.fromkeys(voluntary.DIAGNOSTICS, 1)}
    for family in voluntary.FAMILIES:
        selected = [w for w in plan["selection"]["worlds"] if w["family"] == family]
        assert len({w["semantic_world_id"] for w in selected}) == 12
        assert Counter(w["known"][0] for w in selected) == {0: 6, 1: 6}
        for rotation in range(4):
            assert {w["known"][0] for w in selected if w["world_slot"] % 4 == rotation} == {0, 1}
        attempts = [r for r in plan["selection"]["candidate_exclusion_ledger"] if r["family"] == family]
        assert [r["candidate_index"] for r in attempts] == list(range(attempts[-1]["candidate_index"] + 1))
        assert sum(r["accepted"] for r in attempts) == 12
    assert "previously inspected" in plan["selection"]["freshness_scope"]
    assert plan["selection"]["historical_exact_exclusions"] == []
    assert plan["selection"]["historical_exact_exclusions_sha256"] == digest([])


def test_duplicate_candidates_are_excluded_without_changing_world_slot_balance(monkeypatch):
    original = state.task

    def duplicate(a):
        return original({**a, "index": 0 if a["index"] == 1 else a["index"]})

    monkeypatch.setattr(state, "task", duplicate)
    selection = voluntary.select_worlds()
    for family in voluntary.FAMILIES:
        rejected = next(
            r
            for r in selection["candidate_exclusion_ledger"]
            if r["family"] == family and r["candidate_index"] == 1
        )
        assert not rejected["accepted"] and rejected["reason"] == "duplicate_semantic_world"
        assert rejected["first_selected_world_slot"] == 0
        assert [w["world_slot"] for w in selection["worlds"] if w["family"] == family] == list(range(12))
    with pytest.raises(ValueError, match="limit exhausted"):
        voluntary.select_worlds(max_candidates=2)


def test_historical_exact_exclusions_are_bound_and_never_reused(plan):
    excluded = [
        world["semantic_world_id"]
        for world in plan["selection"]["worlds"]
        if world["world_slot"] < 3
    ]
    selected_plan = voluntary.assignment_plan(excluded_world_ids=excluded)
    selection = selected_plan["selection"]
    assert selection == voluntary.select_worlds(excluded_world_ids=excluded[::-1] + excluded[:1])
    assert selection["historical_exact_exclusions"] == sorted(excluded)
    assert selection["historical_exact_exclusions_sha256"] == digest(sorted(excluded))
    assert selected_plan["selection_sha256"] == digest(selection)
    assert selected_plan["assignments"] == voluntary.assignments(excluded_world_ids=excluded)
    assert len(selected_plan["assignments"]) == 288
    assert not set(excluded) & {world["semantic_world_id"] for world in selection["worlds"]}
    for family in voluntary.FAMILIES:
        worlds = [world for world in selection["worlds"] if world["family"] == family]
        assert [world["world_slot"] for world in worlds] == list(range(12))
        assert Counter(world["known"][0] for world in worlds) == {0: 6, 1: 6}
        ledger = [row for row in selection["candidate_exclusion_ledger"] if row["family"] == family]
        assert [row["candidate_index"] for row in ledger] == list(range(worlds[-1]["index"] + 1))
        assert sum(row["accepted"] for row in ledger) == 12
        historical = [row for row in ledger if row["semantic_world_id"] in excluded]
        assert len({row["semantic_world_id"] for row in historical}) == 3
        assert all(
            not row["accepted"]
            and row["reason"] == "historical_exact_exposure"
            and row["first_selected_world_slot"] is None
            for row in historical
        )
    # The oracle needs the frozen assignment, never the external exclusion inventory.
    for assignment in selected_plan["assignments"]:
        voluntary.task(assignment)


@pytest.mark.parametrize("excluded", [None, "a" * 64, {"objects": []}, ["a" * 63], ["G" * 64], [1]])
def test_historical_exclusions_reject_malformed_world_ids(excluded):
    with pytest.raises(ValueError, match="flat collection of SHA256"):
        voluntary.select_worlds(excluded_world_ids=excluded)


def test_exhausted_historical_pool_fails_without_relaxing_exclusions(monkeypatch, plan):
    original = state.task
    monkeypatch.setattr(state, "task", lambda a: original({**a, "index": 0}))
    excluded = [
        world["semantic_world_id"]
        for world in plan["selection"]["worlds"]
        if world["world_slot"] == 0
    ]
    with pytest.raises(ValueError, match="limit exhausted"):
        voluntary.select_worlds(max_candidates=4, excluded_world_ids=excluded)


def test_same_world_and_exact_oracle_uncertainty_across_deliveries_and_menus(plan):
    worlds = defaultdict(list)
    for a in plan["assignments"]:
        t = voluntary.task(a)
        raw = state.task(a)
        assert raw["uncertainty_bits"] == raw["information_bits"] == 1
        assert t["candidates"] == raw["candidates"] and t["truth"] == raw["truth"]
        expected = 0 if a["delivery"] == "decisive_evidence" else 1
        assert t["uncertainty_bits"] == t["information_bits"] == expected
        assert t["delivery_information_bits"] == 1 - expected
        assert len(t["compatible"]) == 2**expected
        assert t["truth"] in t["compatible"]
        assert len(t["observation_ledger"]) == 2
        if a["kind"] == "choice":
            worlds[a["family"], a["world_slot"]].append(t)
    for tasks in worlds.values():
        assert len({digest(t["candidates"]) for t in tasks}) == 1
        assert len({t["truth"] for t in tasks}) == 1
        assert len({digest(t["initial_fact"]) for t in tasks}) == 1


def test_canonical_identity_ignores_row_and_column_order_but_retains_truth(plan):
    a = plan["assignments"][0]
    t = state.task(a)
    canonical = voluntary.canonical_world(a["family"], t)
    shuffled = deepcopy(t)
    shuffled["attributes"].reverse()
    shuffled["candidates"] = [list(reversed(row)) for row in reversed(t["candidates"])]
    shuffled["truth"] = 3 - t["truth"]
    assert voluntary.canonical_world(a["family"], shuffled) == canonical
    shuffled["truth"] = (shuffled["truth"] + 1) % 4
    assert voluntary.canonical_world(a["family"], shuffled) != canonical


def test_action_position_mapping_balance_and_reverse_presentation(plan):
    counts = Counter()
    for a in plan["assignments"]:
        if a["kind"] != "choice":
            continue
        t = voluntary.task(a)
        for position, label in enumerate(t["menu_order"]):
            counts[a["family"], a["delivery"], a["menu_layout"], position, t["action_mapping"][label]] += 1
        opposite = {**a, "menu_layout": 1 - a["menu_layout"]}
        assert voluntary.task(opposite)["action_mapping"] == t["action_mapping"]
        assert voluntary.task(opposite)["menu_order"] == t["menu_order"][::-1]
    assert set(counts.values()) == {3}
    assert len(counts) == 2 * 3 * 2 * 4 * 4


def test_patterns_are_separate_from_truth_world_sampling_and_delivery(plan, monkeypatch):
    a = plan["assignments"][0]
    pattern = voluntary.pattern_for(a, "assigned_delivery")
    changed = {
        **a,
        "index": 9999,
        "known": [1],
        "query": 0,
        "truth": "unrelated",
        "semantic_world_id": "other",
    }
    monkeypatch.setattr(state, "task", lambda a: pytest.fail("Pattern consulted the world oracle"))
    assert voluntary.pattern_for(changed, "assigned_delivery") == pattern
    assert len(pattern.splitlines()) == 4 and all(len(line.split()) == 4 for line in pattern.splitlines())
    assert set(pattern.replace("\n", " ").split()) <= {".", "*"}
    with pytest.raises(ValueError, match="channel"):
        voluntary.pattern_for(a, "truth")


def test_optional_prompt_has_no_diagnostic_or_preference_instruction(plan):
    for a in plan["assignments"]:
        text, labels, expected = voluntary.prompt(a)
        assert labels == "ABCD" and text.count("End of record.") == 1
        if a["kind"] == "choice":
            assert expected is None and "no action is prescribed" in text
            assert "only the selected item is supplied" in text
            for forbidden in (
                "curiosity",
                "curious",
                "satisfaction",
                "satisfied",
                "diagnostic",
                "score",
                "price",
                "reward",
                "Your goal",
            ):
                assert forbidden not in text
        else:
            assert expected in labels
        request = voluntary.inference_request(a)
        assert request["enable_thinking"] and request["max_new_tokens"] == 4096
        assert request["messages"][1]["content"] == text


def test_only_report_diagnostic_temporal_wording_changes_against_frozen_v1(plan):
    # Generated independently from the actual frozen v1 module at matched v2
    # assignments (seed 2026101435, no historical exclusions), not from v2 code.
    # v1 source byte SHA: eda9deec8a446c7af317ae3887c95c3d5af74ddd30b304d72646d4a4f0f8dc0e.
    original_hashes = {
        "action_consequence": "17149098d4323592a1d96bfa119dc8e2325ef3ac6f4f66ba0c34cf226e3f49bf",
        "choice": "8a3c7048a56be97f5d328d01953e958c42ccd0a0dac0fd2b91fe00c57e932a92",
        "identify_resolved": "f7e7426d5f2268a3bf59f0b5ed6fdf30b682b83a13db7b39761efbef85d1fa85",
        "instrument_useful_action": "c8c63d0d2ac96e9aa4d873f7a9b033a41eddd2ab3f6601aed31e4bfc94aaa93e",
        "new_fact": "045c83f717599e6326c85b94b6bf9bd66bf56d29fffe91e3b9cff9e710e23216",
        "pattern_independence": "aadf9c999e3d5d57e8190ce05ba0b5a816329d2c16523a82a121501e6bc867bb",
        "repeated_fact": "e2f4136f5ce9e17c0d61c1ad09f745eb0f1a6ada1aa13b39708c717fc3331d8e",
    }
    by_kind = defaultdict(list)
    for assignment in sorted(plan["assignments"], key=lambda a: a["trial_id"]):
        text, labels, answer = voluntary.prompt(assignment)
        if assignment["kind"] in ("new_fact", "repeated_fact"):
            description = voluntary.service_descriptions(assignment)["report"]
            stem = (
                "Both the Initial observation and the Additional delivery above have already been received. "
                "Now suppose you choose this action:\n"
                + description
                + "\nCompared with all property facts supplied in the complete record above, "
                "what would this newly requested report provide?"
            )
            assert text.count(stem) == 1
            text = text.replace(stem, "What kind of information would this action supply? " + description)
            for phrase in ("not already supplied", "has already been supplied"):
                assert text.count(phrase + " in the complete record above.") == 1
                text = text.replace(phrase + " in the complete record above.", phrase + ".")
        by_kind[assignment["kind"]].append(
            dict(trial_id=assignment["trial_id"], prompt=(text, labels, answer))
        )
    assert {kind: digest(prompts) for kind, prompts in by_kind.items()} == original_hashes


def test_all_six_diagnostics_have_balanced_labels_and_never_execute_actions(plan):
    counts, consequences, tested_letters = Counter(), Counter(), Counter()
    for a in plan["assignments"]:
        if a["kind"] == "choice":
            continue
        expected = voluntary.prompt(a)[2]
        counts[a["family"], a["kind"], expected] += 1
        record = voluntary.record_from_thinking_inference(a, {"raw": "Reasoning.</think> " + expected})
        assert record["valid"] and record["correct"] and record["thinking_closed"]
        assert record["selected_service"] is None and record["simulator_outcome"] is None
        if a["kind"] == "action_consequence":
            t = voluntary.task(a)
            service = t["diagnostic_consequence_service"]
            consequences[a["family"], service] += 1
            tested_letters[a["family"], next(k for k, v in t["action_mapping"].items() if v == service)] += 1
        with pytest.raises(ValueError, match="Diagnostics"):
            voluntary.VoluntaryTrial(a)
    # Instrumental action letters follow the independently balanced service mapping.
    assert set(counts.values()) == {3}
    assert len(counts) == 2 * 6 * 4 and set(consequences.values()) == {3}
    assert len(tested_letters) == 8
    for family in voluntary.FAMILIES:
        assert [tested_letters[family, letter] for letter in voluntary.LABELS] == [4, 2, 4, 2]


def test_report_diagnostic_changes_from_new_to_repeated_after_resolution(plan):
    for family in voluntary.FAMILIES:
        for slot in range(voluntary.ITEMS):
            diagnostics = {
                a["kind"]: a
                for a in plan["assignments"]
                if a["family"] == family
                and a["world_slot"] == slot
                and a["kind"] in ("new_fact", "repeated_fact")
            }
            new, repeated = diagnostics["new_fact"], diagnostics["repeated_fact"]
            assert new["delivery"] == "redundant_evidence"
            assert repeated["delivery"] == "decisive_evidence"
            nt, rt = voluntary.task(new), voluntary.task(repeated)
            assert nt["report_fact"] == rt["report_fact"]
            assert nt["initial_fact"] == rt["initial_fact"]
            assert nt["report_fact"] not in nt["observation_ledger"][1]["facts"]
            assert rt["report_fact"] in rt["observation_ledger"][1]["facts"]
            description = voluntary.service_descriptions(new)["report"]
            for assignment, expected_gain in ((new, 1), (repeated, 0)):
                text, _, answer = voluntary.prompt(assignment)
                assert "Both the Initial observation and the Additional delivery above have already been received." in text
                assert f"Now suppose you choose this action:\n{description}" in text
                assert "Compared with all property facts supplied in the complete record above, what would this newly requested report provide?" in text
                expected_text = (
                    "A property fact about the hidden candidate that is not already supplied in the complete record above."
                    if expected_gain
                    else "A property fact about the hidden candidate that has already been supplied in the complete record above."
                )
                assert f"{answer}: {expected_text}" in text
                outcome = voluntary.VoluntaryTrial({**assignment, "kind": "choice"}).apply("report")
                assert outcome["uncertainty_before_bits"] - outcome["uncertainty_after_bits"] == expected_gain
            with pytest.raises(ValueError, match="Diagnostic contexts"):
                voluntary.task({**repeated, "delivery": "redundant_evidence"})


@pytest.mark.parametrize("service", voluntary.SERVICES)
def test_simulator_honors_only_selected_payload_and_ends_once(plan, service):
    for a in plan["assignments"]:
        if a["kind"] != "choice":
            continue
        original = deepcopy(a)
        t = voluntary.task(a)
        trial = voluntary.VoluntaryTrial(a)
        result = trial.apply(service)
        assert a == original and trial.done and result["terminal"]
        assert result["observation_ledger"][:2] == t["observation_ledger"]
        assert len(result["observation_ledger"]) == 3
        event = result["observation_ledger"][-1]
        assert event["service"] == service and event["payload"] == result["payload"]
        assert not event["model_pass_completed"] and not result["receipt_model_pass_completed"]
        if service == "report":
            assert event["facts"] == [t["report_fact"]]
            assert result["uncertainty_after_bits"] == 0
        elif service == "repeat":
            assert event["facts"] == [t["initial_fact"]]
            assert result["uncertainty_after_bits"] == t["uncertainty_bits"]
        else:
            assert event["facts"] == []
            assert result["uncertainty_after_bits"] == t["uncertainty_bits"]
        if service == "stop":
            assert result["payload"] is None and not result["receipt_required"]
        else:
            assert result["payload"] and result["receipt_required"]
        with pytest.raises(ValueError, match="unfinished"):
            trial.apply(service)


def test_full_thinking_parse_invalid_closure_and_finalization_do_not_deliver(plan):
    a = next(a for a in plan["assignments"] if a["kind"] == "choice")
    parsed = voluntary.record_from_inference(a, {"raw": "A"})
    assert parsed["valid"] and parsed["simulator_outcome"] is None
    assert voluntary.finalize_choice(parsed)["simulator_outcome"] is None
    for raw in (
        "A",
        "Still thinking",
        "Thought.</think> A because it helps",
        "Thought.</think> A B",
        "Thought.</think>",
    ):
        record = voluntary.record_from_thinking_inference(a, {"raw": raw})
        assert not record["valid"] and record["correct"] is None
        assert record["selected_service"] is None and record["simulator_outcome"] is None
    bad = voluntary.record_from_thinking_inference(a, {"raw": "A", "generated_token_ids": [0] * 4096})
    assert bad["cap_hit"] and not bad["valid"]
    with pytest.raises(ValueError, match="valid closed"):
        voluntary.receipt_input(bad)


def test_receipts_bind_actual_raw_choice_and_selected_payload_only(plan):
    a = next(
        a for a in plan["assignments"] if a["kind"] == "choice" and a["delivery"] == "redundant_evidence"
    )
    for service in voluntary.SERVICES:
        record = choice_for_service(a, service)
        assert record["valid"] and record["correct"] is None and record["selected_service"] == service
        receipt = voluntary.receipt_input(record)
        if service == "stop":
            assert receipt is None
            continue
        outcome = record["simulator_outcome"]
        assert receipt["messages"][:2] == voluntary.inference_request(a)["messages"]
        assert receipt["messages"][2] == dict(role="assistant", content=record["inference"]["raw"])
        assert (
            receipt["messages"][3]["content"]
            == outcome["payload"] + "\nEpisode ended. No response is requested."
        )
        assert receipt["max_new_tokens"] == 0 and not receipt["model_pass_completed"]
        assert receipt["messages_sha256"] == digest(receipt["messages"])
        assert receipt["selected_payload_sha256"] == digest(outcome["payload"])
        corrupted = deepcopy(record)
        corrupted["simulator_outcome"]["payload"] = "An unselected fact"
        with pytest.raises(ValueError, match="payload or ledger"):
            voluntary.receipt_input(corrupted)
        corrupted = deepcopy(record)
        corrupted["selected_service"] = "forged"
        with pytest.raises(ValueError, match="actual generated answer"):
            voluntary.receipt_input(corrupted)
        with pytest.raises(ValueError, match="unfinalized"):
            voluntary.finalize_choice(record)


def test_semantic_and_diagnostic_context_binding_rejects_tampering(plan):
    a = plan["assignments"][0]
    with pytest.raises(ValueError, match="semantic binding"):
        voluntary.task({**a, "semantic_world_id": "forged"})
    with pytest.raises(ValueError, match="assignment"):
        voluntary.task({**a, "known": []})
    diagnostic = next(a for a in plan["assignments"] if a["kind"] == "identify_resolved")
    with pytest.raises(ValueError, match="Diagnostic contexts"):
        voluntary.task({**diagnostic, "delivery": "independent_novelty"})
