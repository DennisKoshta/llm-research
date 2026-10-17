"""Fixed scientific wiring and failure accounting using synthetic source packages."""

from collections import Counter
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from curiosity_study import coverage_bridge as bridge
from curiosity_study.artifacts import digest
from test_prepare_curiosity_coverage_bridge import material, sources as source_fixture


@pytest.fixture(scope="module")
def template():
    with pytest.MonkeyPatch.context() as patch:
        inputs = source_fixture.__wrapped__(patch)
        return material(inputs)


@pytest.fixture
def plan(template, monkeypatch):
    monkeypatch.setattr(bridge, "PINNED", {key: template[key] for key in bridge.PINNED})
    return template


def fake_collection(plan, monkeypatch, *, failure=None):
    calls, receipt_calls = [], []
    model = SimpleNamespace(active_hooks=0, metadata=lambda: dict(plan["model"]))
    monkeypatch.setattr(bridge.engine, "hook_inventory", lambda model: {})

    def collect(model, requests, call, direction, *, dose, deadline):
        calls.append(deepcopy(call))
        assert dose == bridge.engine.DOSE
        key = call["direction_key"] or "zero_sham"
        np.testing.assert_array_equal(direction, plan["direction_bindings"][key]["unit"])
        if failure == "raise":
            raise RuntimeError("synthetic model failure")
        rows = [
            dict(
                completion=dict(
                    raw="<think></think>A",
                    generated_token_ids=[5, 6],
                    ended_with_eos=True,
                    sampling=dict(seed=request["seed"], temperature=0.7, top_p=1.0, max_new_tokens=4096),
                    intervention_schedule=call["schedule"],
                )
            )
            for request in requests
        ]
        frame = dict(
            forward_index=0,
            prediction_ordinal=1,
            active_prediction_rows=[True] * 32,
            full_fp32_logits_sha256=[digest(["logits", i]) for i in range(32)],
            input_ids=[r["token_prefix"] for r in requests],
            position_ids=[list(range(len(r["token_prefix"]))) for r in requests],
            attention_mask_row_sha256=[digest([1] * len(r["token_prefix"])) for r in requests],
            geometry=[dict(before=digest(request)) for request in requests],
        )
        if failure == "sham_logits" and call["condition"] == "full_zero":
            frame["full_fp32_logits_sha256"][0] = "changed"
        if failure == "prepatch" and call["position"] is not None:
            frame["geometry"][0]["before"] = "changed"
        passed = failure != "numerical"
        return dict(
            call=call,
            rows=rows,
            forwards=[frame],
            requests_sha256=digest(requests),
            passed=passed,
            checks=dict(synthetic_numerics=passed),
            generation_seconds=0.01,
        ), {}

    monkeypatch.setattr(bridge.engine, "collect_call", collect)
    monkeypatch.setattr(
        bridge.voluntary,
        "record_from_thinking_inference",
        lambda assignment, inference: dict(
            assignment=assignment, kind=assignment["kind"], inference=inference, cap_hit=False
        ),
    )

    def receipt(model, choices, sources, *, deadline):
        receipt_calls.append([a["trial_id"] for a in choices])
        assert all(a["trial_id"] in sources for a in choices)
        sidecars = [
            dict(
                source_trial_id=a["trial_id"],
                status="stop",
                receipt_required=False,
                receipt_model_pass_completed=False,
            )
            for a in choices
        ]
        return dict(
            records=sidecars,
            records_sha256=digest(sidecars),
            passed=True,
            error=None,
            total_seconds=0.01,
            receipt_forward_seconds=0.0,
            actual_receipt_rows=0,
        ), {}

    monkeypatch.setattr(bridge, "collect_receipt_group", receipt)
    return model, calls, receipt_calls


def test_all261_calls_retain_every_signed_direction_schedule_companion_and_seed(plan):
    bridge.validate_materials(plan)
    calls = [bridge.scientific_call(plan, c) for c in plan["execution_cohorts"]]
    assert len(calls) == len({c["call_id"] for c in calls}) == 261
    assert len({trial for c in calls for trial in c["trial_ids"]}) == 8352
    assert (
        sum(
            sum(a["kind"] == "choice" for a in group)
            for worker in bridge.GROUPS
            for group in bridge.receipt_groups(plan, worker)
        )
        == 4176
    )
    for cohort, call in zip(plan["execution_cohorts"], calls, strict=True):
        assert call["trial_ids"] == cohort["trial_ids"] and call["base_trial_ids"] == cohort["base_trial_ids"]
        assert call["cap"] == 4096 and call["stage"] == "scientific"
        assert call["expected_error"] is None and call["inject_failure_at"] is None
        assert call["one_row_only"] is None
        if cohort["arm"] == "baseline":
            assert call["position"] is None and call["direction_key"] is None
        else:
            schedule, direction = cohort["arm"].split("__")
            assert call["position"] == ("record" if schedule == "record" else "all_attended")
            assert call["direction_key"] == direction and call["dose_multiplier"] == (
                0 if direction == "zero_sham" else 1
            )
    for worker in bridge.GROUPS:
        groups = bridge.receipt_groups(plan, worker)
        assigned = [
            a["trial_id"]
            for a in plan["assignments"]
            if a["kind"] == "choice" and a["execution_group"] == worker
        ]
        assert [a["trial_id"] for group in groups for a in group] == assigned
        assert all(len(group) == 32 for group in groups[:-1]) and 0 < len(groups[-1]) <= 32
        assert any(len({a["arm"] for a in group}) > 1 for group in groups), (
            "Original logical groups cross generation-arm boundaries"
        )


def test_full_worker_runs_all_generations_before_fixed_logical_receipt_groups(tmp_path, plan, monkeypatch):
    model, calls, receipts = fake_collection(plan, monkeypatch)
    original = bridge.collect_receipt_group

    def checked(*args, **kwargs):
        assert len(calls) == 87, "Receipts cannot be regrouped within each generation cohort"
        return original(*args, **kwargs)

    monkeypatch.setattr(bridge, "collect_receipt_group", checked)
    result = bridge.execute(tmp_path, model, plan, execution_group="worker_0")
    assert len(calls) == 87 and len(result["generation_slots"]) == 2784
    assert all(row["status"] == "completed" for row in result["generation_slots"])
    expected = bridge.receipt_groups(plan, "worker_0")
    assert receipts == [[a["trial_id"] for a in group] for group in expected]
    assert len(result["choice_slots"]) == sum(map(len, expected))
    assert all(row["status"] == "stop" for row in result["choice_slots"])
    assert result["complete"] and result["engineering_passed"] and not result["scientific_admission"]
    assert result["parent_assigned_generations"] == 8352 and result["parent_assigned_choices"] == 4176
    assert Counter(c["direction_key"] for c in calls)[None] == 3
    with pytest.raises(ValueError, match="retried or resumed"):
        bridge.execute(tmp_path, model, plan, execution_group="worker_0")
    assert len(calls) == 87


@pytest.mark.parametrize(
    "failure,expected_calls", [("numerical", 1), ("raise", 1), ("prepatch", 2), ("sham_logits", 3)]
)
def test_first_numerical_identity_or_execution_failure_stops_without_pending_receipts(
    tmp_path, plan, monkeypatch, failure, expected_calls
):
    model, calls, receipts = fake_collection(plan, monkeypatch, failure=failure)
    result = bridge.execute(tmp_path, model, plan, execution_group="worker_0")
    assert len(calls) == expected_calls and not receipts
    assert len(result["generation_slots"]) == 2784
    assert all(row["status"] == "missing" for row in result["choice_slots"])
    assert not result["complete"] and not result["engineering_passed"] and result["error"]
    if failure == "raise":
        assert sum(row["status"] == "execution_error" for row in result["generation_slots"]) == 32
    else:
        assert sum(row["status"] == "completed" for row in result["generation_slots"]) == expected_calls * 32
        assert (
            sum(row["engineering_passed"] for row in result["generation_slots"]) == (expected_calls - 1) * 32
        )


def test_deadline_before_first_call_preserves_all_assigned_slots(tmp_path, plan, monkeypatch):
    model, calls, receipts = fake_collection(plan, monkeypatch)
    result = bridge.execute(tmp_path, model, plan, execution_group="worker_1", deadline=0.0)
    assert not calls and not receipts and not result["complete"]
    assert len(result["generation_slots"]) == 2784 and all(
        r["status"] == "missing" for r in result["generation_slots"]
    )
    assert len(result["choice_slots"]) == sum(map(len, bridge.receipt_groups(plan, "worker_1")))


@pytest.mark.parametrize("failed_stage", ["evidence", "terminal_envelope"])
def test_attempted_persistence_failure_never_certifies_rows_or_executes_pending_receipts(
    tmp_path, plan, monkeypatch, failed_stage
):
    model, calls, receipts = fake_collection(plan, monkeypatch)
    if failed_stage == "evidence":

        def failed_evidence(*args, **kwargs):
            raise OSError("Synthetic evidence persistence failure")

        monkeypatch.setattr(bridge, "_durable_report", failed_evidence)
    else:
        original_write = bridge.write_once

        def failed_envelope(path, data):
            if path.parent.name == "generation_cohorts":
                raise OSError("Synthetic terminal envelope persistence failure")
            return original_write(path, data)

        monkeypatch.setattr(bridge, "write_once", failed_envelope)

    result = bridge.execute(tmp_path, model, plan, execution_group="worker_0")
    assert len(calls) == 1 and not receipts
    assert result["error"]["type"] == "OSError"
    assert Counter(row["status"] for row in result["generation_slots"]) == {
        "execution_error": 32,
        "missing": 2752,
    }
    assert all(row["record_sha256"] is None for row in result["generation_slots"])
    assert all(row["status"] == "missing" for row in result["choice_slots"])
    assert not result["complete"] and not result["engineering_passed"]
    assert (tmp_path / "generation_started/000.json").exists()
    assert (tmp_path / "generation/000.json").exists() == (failed_stage == "terminal_envelope")
    assert not (tmp_path / "generation_cohorts/000.json").exists()


def test_rehashed_materials_or_altered_direction_cannot_start(tmp_path, plan, monkeypatch):
    changed = deepcopy(plan)
    changed["direction_bindings"]["curiosity_negative"]["dose"] *= 0.5
    changed["direction_bindings_sha256"] = digest(changed["direction_bindings"])
    model, calls, _ = fake_collection(plan, monkeypatch)
    with pytest.raises(ValueError, match="Frozen material differs"):
        bridge.execute(tmp_path, model, changed, execution_group="worker_0")
    assert not calls and not list(tmp_path.iterdir())


def test_receipt_failure_retains_first_group_and_marks_later_groups_missing(tmp_path, plan, monkeypatch):
    model, calls, receipts = fake_collection(plan, monkeypatch)

    def broken(model, choices, sources, *, deadline):
        receipts.append([a["trial_id"] for a in choices])
        records = [
            dict(
                source_trial_id=a["trial_id"],
                status="execution_error",
                receipt_required=True,
                receipt_model_pass_completed=False,
            )
            for a in choices
        ]
        return dict(
            records=records,
            records_sha256=digest(records),
            passed=False,
            error={"synthetic": True},
            total_seconds=0.01,
            receipt_forward_seconds=0.01,
            actual_receipt_rows=32,
        ), {}

    monkeypatch.setattr(bridge, "collect_receipt_group", broken)
    result = bridge.execute(tmp_path, model, plan, execution_group="worker_0")
    assert len(calls) == 87 and len(receipts) == 1
    assert len(result["choice_slots"]) > 32
    assert [
        sum(r["status"] == status for r in result["choice_slots"])
        for status in ("execution_error", "missing")
    ] == [32, len(result["choice_slots"]) - 32]
    assert not result["complete"] and not result["engineering_passed"]


def test_active_receipts_are_filtered_only_inside_the_fixed_logical_group(monkeypatch):
    calls = []
    model = SimpleNamespace(active_hooks=0)
    monkeypatch.setattr(bridge.engine, "hook_inventory", lambda model: {})
    choices = [dict(kind="choice", trial_id=f"trial{i}") for i in range(5)]
    sources = {
        a["trial_id"]: dict(
            trial_id=a["trial_id"],
            task_record=dict(
                index=i, simulator_outcome={"observation_ledger": [dict(model_pass_completed=False)]}
            ),
        )
        for i, a in enumerate(choices)
    }

    def sidecar(source):
        i = source["task_record"]["index"]
        return dict(
            source_trial_id=source["trial_id"],
            receipt_required=i in (0, 3),
            receipt_model_pass_completed=False,
            status="pending" if i in (0, 3) else "stop",
            executed_simulator_outcome=deepcopy(source["task_record"]["simulator_outcome"]),
        )

    monkeypatch.setattr(bridge, "choice_sidecar", sidecar)
    monkeypatch.setattr(
        bridge.receipts,
        "render_receipt",
        lambda row, model: (dict(payload=row["index"]), dict(index=row["index"]), {}),
    )

    def forward(model, requests, ids, *, layers):
        calls.append((requests, ids, layers))
        return (
            [{"row": i} for i in range(len(ids))],
            np.zeros((2, 1, 2, 5120), dtype=np.float32),
            dict(passed=True, prefill_width=128),
        )

    monkeypatch.setattr(bridge.receipts, "forward_receipts", forward)
    result, arrays = bridge.collect_receipt_group(model, choices, sources)
    assert result["source_trial_ids"] == [a["trial_id"] for a in choices]
    assert calls == [([{"index": 0}, {"index": 3}], ["trial0", "trial3"], [15])]
    assert result["assigned_choices"] == 5 and result["actual_receipt_rows"] == 2
    assert [r["status"] for r in result["records"]] == ["completed", "stop", "stop", "completed", "stop"]
    assert result["passed"] and arrays["hidden"].shape[0] == 2


def test_capped_choice_never_receives_payload_even_if_parser_has_a_service(monkeypatch):
    source = dict(task_record=dict(cap_hit=True))
    monkeypatch.setattr(
        bridge.legacy,
        "_choice_sidecar",
        lambda source: dict(
            status="pending",
            receipt_required=True,
            selected_service="report",
            receipt_model_pass_completed=False,
        ),
    )
    result = bridge.choice_sidecar(source)
    assert result["status"] == "capped_choice" and not result["receipt_required"]
    assert not result["receipt_model_pass_completed"]
