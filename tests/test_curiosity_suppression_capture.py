"""Synthetic complete15-arm capture; no selected real data or paid inference."""

from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from curiosity_study import curiosity_suppression_capture as capture
from curiosity_study import curiosity_suppression_screen as screen
from curiosity_study import satisfaction_screen as prior
from curiosity_study.artifacts import RunStore, digest, read, write_once
from test_satisfaction_screen import direction
from test_satisfaction_capture import fake_model, install_fake_generation as prior_fake_generation, install_fake_receipts


@pytest.fixture(scope="module")
def frozen():
    # This is deliberately synthetic: neither source data nor original direction
    # files are read. Only the new task's exact-source admission hash is replaced.
    patch = pytest.MonkeyPatch()
    source = dict(assay="satisfaction_screen", screen_plan=prior.assignment_plan(
        historical_exact_world_ids=[], direction_binding=direction()))
    patch.setattr(screen, "PRIOR_MANIFEST_SHA256", digest(source))
    bindings = {}
    specifications = {"zero_sham": ("curiosity_vs_indifference", 1),
                      screen.PRIMARY: ("curiosity_vs_indifference", -1),
                      **{a: (s, o) for a, s, o in screen.NEIGHBORS},
                      **{a: ("seeded_gaussian", 1) for a in screen.RANDOM_ARMS}}
    for index, (arm, (name, orientation)) in enumerate(specifications.items()):
        unit = [0.0] * 5120
        coordinate = 0 if arm in ("zero_sham", screen.PRIMARY) else 4 if arm.startswith("valence_") else index
        unit[coordinate] = float(orientation)
        bindings[arm] = dict(source=name, orientation=orientation, layer=15, position="record",
            dose=0.0 if arm == "zero_sham" else screen.LOCKED_C_NORM,
            reference_raw_norm=screen.LOCKED_C_NORM, unit=unit, unit_sha256=digest(unit),
            corpus=None if arm in screen.RANDOM_ARMS else "A",
            source_raw_norm=2.0, directions_byte_sha256=screen.LOCKED_DIRECTION_BYTES)
        if arm in screen.RANDOM_ARMS:
            bindings[arm].update(seed=screen.RANDOM_SEEDS[screen.RANDOM_ARMS.index(arm)],
                construction="numpy_default_rng_standard_normal_float64_unit_no_resampling")
    plan = screen.assignment_plan(source_manifest=source, direction_bindings=bindings)
    model = fake_model()
    preflight = capture.bind_preflight(plan, model)
    yield plan, preflight, model, source
    patch.undo()


def manifest(frozen, group="worker_0"):
    plan, preflight, _, source = frozen
    rows = screen.group_assignments(plan, group)
    return dict(assay=screen.ASSAY, task_revision=screen.REVISION, model=capture.MODEL,
        source_screen_manifest=source, source_screen_manifest_sha256=digest(source),
        screen_plan=plan, screen_plan_sha256=digest(plan), execution_group=group,
        parent_manifest_sha256=digest({"synthetic_parent": plan}), assignments=[], diagnostics=rows,
        assignment_sha256=digest(rows), execution=screen.EXECUTION, max_new_tokens=4096,
        capture_plan=capture.CAPTURE_PLAN, generation_preflight=preflight,
        generation_preflight_sha256=digest(preflight))


def install_fake_generation(monkeypatch, frozen, calls, **kwargs):
    # The old fixture supplies only deterministic fake observations. Redirect its
    # monkeypatch to the new executor; production globals are never changed.
    prior_fake_generation(monkeypatch, frozen, calls, **kwargs)
    monkeypatch.setattr(capture, "collect_generation", capture.legacy.collect_generation)


def test_preflight_covers288_distinct_sources_despite_cohort_major_first288(frozen):
    plan, preflight, model, _ = frozen
    assert len({row["base_row"] for row in plan["assignments"][:288]}) == 32
    assert len(preflight["requests"]) == 288
    assert len({row["base_trial_id"] for row in preflight["requests"]}) == 288
    assert [row["base_trial_id"] for row in preflight["requests"]] == [row["base_trial_id"] for row in plan["base_assignments"]]
    assert len(preflight["outer_request_bindings"]) == 4320
    assert capture.bind_preflight(plan, model) == preflight
    for group in screen.GROUPS:
        rows = capture.validate_manifest(manifest(frozen, group))
        assert len(rows) == 1440
        assert {row["base_cohort"] % 3 for row in rows} == {screen.GROUPS.index(group)}


@pytest.mark.parametrize("change", ["source", "rows", "seed", "preflight_order", "direction", "parent"])
def test_binding_tampering_fails_closed(frozen, change):
    m = deepcopy(manifest(frozen))
    if change == "source":
        m["source_screen_manifest"]["extra"] = True
        m["source_screen_manifest_sha256"] = digest(m["source_screen_manifest"])
    elif change == "rows":
        m["diagnostics"][0], m["diagnostics"][1] = m["diagnostics"][1], m["diagnostics"][0]
        m["assignment_sha256"] = digest(m["diagnostics"])
    elif change == "parent":
        m["parent_manifest_sha256"] = None
    elif change == "direction":
        m["screen_plan"]["direction_bindings"][screen.PRIMARY]["dose"] *= -1
        m["screen_plan_sha256"] = digest(m["screen_plan"])
    else:
        p = m["generation_preflight"]
        if change == "seed":
            p["requests"][0]["request"]["seed"] += 1
        else:
            p["requests"][0], p["requests"][1] = p["requests"][1], p["requests"][0]
        p["requests_sha256"] = digest(p["requests"])
        m["generation_preflight_sha256"] = digest(p)
    with pytest.raises(ValueError):
        capture.validate_manifest(m)


@pytest.mark.parametrize("group", screen.GROUPS)
def test_complete_child_all15arms_and_dynamic_receipt_tail(tmp_path, monkeypatch, frozen, group):
    m = manifest(frozen, group)
    generated, forwarded = [], []
    install_fake_generation(monkeypatch, frozen, generated)
    install_fake_receipts(monkeypatch, tmp_path, forwarded)
    rows = capture.execute(tmp_path, m, frozen[2])
    assert len(rows) == 1440 and len(generated) == 45
    assert all(row["engineering_passed"] for row in rows)
    evidence = [read(path) for path in (tmp_path / "generation_evidence").glob("*/*.json")]
    assert sum(e["sham_comparison"] is not None for e in evidence) == 3
    assert sum(e["common_prefill_comparison"] is not None for e in evidence) == 42
    assert all(e["common_prefill_comparison"]["layer15_prepatch_captures_equal"] for e in evidence if e["common_prefill_comparison"])
    envelopes = sorted((read(path) for path in (tmp_path / "receipt_cohorts").glob("*/*.json")), key=lambda e: e["choice_offset"])
    expected = [row["trial_id"] for row in m["diagnostics"] if row["kind"] == "choice"]
    assert [s["source_trial_id"] for e in envelopes for s in e["records"]] == expected
    assert len(expected) in (795, 705, 660)
    assert [len(e["records"]) for e in envelopes] == [32] * (len(expected) // 32) + [len(expected) % 32]
    assert {s["status"] for e in envelopes for s in e["records"]} == {"completed", "stop", "invalid_choice"}
    assert all(not (row["task_record"].get("simulator_outcome") or {}).get("receipt_model_pass_completed", False) for row in rows)
    profile = read(next((tmp_path / "execution_profiles").glob("*.json")))
    assert profile["parent_generations_required"] == 4320 and profile["parent_choice_receipts_required"] == 2160
    assert profile["choice_receipts_assigned"] == len(expected)
    spec = importlib.util.spec_from_file_location("new_screen_coverage", Path(__file__).parents[1] / "scripts/collection_coverage.py")
    coverage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(coverage)
    receipt = coverage.check_collection_coverage(tmp_path, expected_manifest_sha256=digest(m))
    assert receipt["assigned_trials"] == 1440 and receipt["verified_cohorts"] == 45


def test_signed_direction_and_positive_absolute_dose_reach_each_arm(frozen):
    plan = frozen[0]
    for arm in screen.ARMS:
        row = next(row for row in plan["assignments"] if row["arm"] == arm)
        direction = np.asarray(plan["direction_bindings"]["zero_sham" if arm == "baseline" else arm]["unit"], dtype=np.float32)
        patch = capture.patches_for([row], direction)[0]
        if arm == "baseline":
            assert patch is None
        else:
            assert patch["dose"] == (0 if arm == "zero_sham" else screen.LOCKED_C_NORM)
            assert np.array_equal(patch["direction"], direction)
    c = plan["direction_bindings"][screen.PRIMARY]
    assert np.dot(c["unit"], plan["direction_bindings"]["zero_sham"]["unit"]) == -1


@pytest.mark.parametrize("failure", ["sham", "prepatch"])
def test_full_observation_mismatch_is_retained_as_failed_not_dropped(tmp_path, monkeypatch, frozen, failure):
    m = manifest(frozen)
    calls = []
    install_fake_generation(monkeypatch, frozen, calls, corrupt_sham=failure == "sham")
    original = capture.collect_generation
    def changed(*args, **kwargs):
        condition, hidden = original(*args, **kwargs)
        if failure == "prepatch" and args[2][0] is not None:
            hidden[0, 0, 0, 0] += 1
        return condition, hidden
    monkeypatch.setattr(capture, "collect_generation", changed)
    monkeypatch.setattr(capture, "execute_receipts", lambda *a, **k: False)
    rows = capture.execute(tmp_path, m, frozen[2])
    assert len(rows) == 1440
    failed = [r for r in rows if not r["engineering_passed"]]
    assert len(failed) == (96 if failure == "sham" else 1344)
    assert all(r["failure"] == "generation_engineering_guard_failed" and r["task_record"] is not None for r in failed)


def test_failed_generation_is_terminal_and_cannot_be_retried(tmp_path, monkeypatch, frozen):
    m = manifest(frozen)
    calls = []
    install_fake_generation(monkeypatch, frozen, calls, unavailable=True)
    rows = capture.execute(tmp_path, m, frozen[2])
    assert len(rows) == 32 and all(r["status"] == "execution_error" for r in rows)
    assert len(RunStore(tmp_path, m).completed()) == 32
    with pytest.raises(RuntimeError, match="cannot be retried"):
        capture.execute(tmp_path, m, frozen[2])
    assert len(calls) == 1 and not list((tmp_path / "receipt_cohorts").glob("*/*.json"))


def test_interrupted_start_refuses_reexecution(tmp_path, monkeypatch, frozen):
    m = manifest(frozen)
    write_once(tmp_path / "generation_started/0000000.json", {"interrupted": True})
    with pytest.raises(ValueError, match="cannot be retried"):
        capture.execute(tmp_path, m, frozen[2])


def test_expired_deadline_starts_no_generation_and_invents_no_terminal_rows(tmp_path, monkeypatch, frozen):
    calls = []
    install_fake_generation(monkeypatch, frozen, calls)
    rows = capture.execute(tmp_path, manifest(frozen), frozen[2], deadline=0)
    assert rows == [] and calls == []
    assert not list((tmp_path / "generation_started").glob("*.json"))
    assert not list((tmp_path / "cohorts").glob("*/*.json"))
