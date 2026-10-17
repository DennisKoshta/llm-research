import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from curiosity_study import natural_information as natural
from curiosity_study.artifacts import RunStore, digest
from curiosity_study.information_state import assignments
from curiosity_study.thinking_information import inference_request, record_from_inference


def source_record(assignment, ids=None):
    ids = ids or [11, 12, 13, 14, 15]
    return dict(
        trial_id=assignment["trial_id"],
        terminal=True,
        assignment=assignment,
        inference=dict(
            enable_thinking=True,
            messages=inference_request(assignment)["messages"],
            rendered_prompt="fixture original thinking prompt",
            unpadded_input_ids=ids,
            generated_token_ids=[22, 23, 24],
            sampling=dict(max_new_tokens=inference_request(assignment)["max_new_tokens"]),
        ),
        activation_artifact=dict(
            positions=["record", "prefill", "think_end", "decision"],
            token_positions=[1, len(ids) - 1, -1, -1],
        ),
        thinking_closed=False,
        correct=False,
        valid=False,
    )


def fixture_collection(tmp_path, max_new_tokens=8, source_max_new_tokens=2048):
    aa = assignments(1, seed=3123, kinds=("new",))[:2]
    if source_max_new_tokens != 2048:
        aa = [{**a, "max_new_tokens": source_max_new_tokens} for a in aa]
    model = dict(model_id="tiny-engineering-transformer", revision="fixture")
    source = dict(model=model, assignments=[], diagnostics=aa)
    store = RunStore(tmp_path / "original", source)
    records = [source_record(a, [11, 12, 13, 14, 15] + [16] * i) for i, a in enumerate(aa)]
    store.save_many(records, "fixture", "0000000")
    prefixes = {str(i): natural.make_prefix(r, "original", digest(source)) for i, r in enumerate(records)}
    diagnostics = [
        dict(
            trial_id=f"{key}-{arm}",
            prefix_id=key,
            arm=arm,
            kind="new",
            layer=1,
            position="final",
            dose=0.0,
            patch_mode="add",
        )
        for key in prefixes
        for arm in ("baseline", "sham")
    ]
    manifest = dict(
        assay="natural_information",
        phase="engineering",
        model=model,
        prefixes=prefixes,
        prefixes_sha256=digest(prefixes),
        assignments=[],
        diagnostics=diagnostics,
        assignment_sha256=digest(diagnostics),
        execution=dict(batch_size=4),
        capture_layers=[0, 1, 2],
        max_new_tokens=max_new_tokens,
        greedy=False,
        engine_sha256=natural.engine_hashes(),
    )
    return manifest


def test_natural_prefix_excludes_all_source_generation_even_when_source_invalid():
    a = assignments(1, kinds=("new",))[0]
    record = source_record(a)
    prefix = natural.make_prefix(record, "original", "manifest")
    assert prefix["token_prefix"] == record["inference"]["unpadded_input_ids"]
    assert prefix["prefix_positions"] == {"record": 1, "final": 4}
    assert prefix["original_generated_tokens_in_prefix"] == 0
    assert prefix["labels"] == list(inference_request(a)["labels"])
    assert prefix["seed"] == a["sampling_seed"] + 20000000
    assert "original_correct" not in prefix and "original_final_answer" not in prefix
    # Neither a valid source answer nor its reasoning changes the new model input.
    changed = copy.deepcopy(record)
    changed.update(correct=True, valid=True, thinking_closed=True)
    changed["inference"]["generated_token_ids"] = [22, 100, 127]
    other = natural.make_prefix(changed, "original", "manifest")
    assert other["token_prefix"] == prefix["token_prefix"]
    assert other["source_record_sha256"] != prefix["source_record_sha256"]


def test_source_binding_rejects_injected_reasoning_tokens_and_changed_record(tmp_path):
    manifest = fixture_collection(tmp_path)
    prefix = manifest["prefixes"]["0"]
    natural.source_check(prefix, {}, tmp_path, manifest["model"])
    bad = copy.deepcopy(prefix)
    bad["token_prefix"].append(22)
    bad["token_prefix_sha256"] = digest(bad["token_prefix"])
    bad["prefix_positions"]["final"] += 1
    with pytest.raises(ValueError, match="exactly the original prompt"):
        natural.source_check(bad, {}, tmp_path, manifest["model"])
    with pytest.raises(ValueError, match="record mismatch"):
        natural.source_check({**prefix, "source_record_sha256": "changed"}, {}, tmp_path)
    with pytest.raises(ValueError, match="models differ"):
        natural.source_check(prefix, {}, tmp_path, dict(model_id="other"))


def test_actual_source_cap_must_match_its_frozen_assignment():
    assignment = {**assignments(1, kinds=("new",))[0], "max_new_tokens": 4096}
    record = source_record(assignment)
    prefix = natural.make_prefix(record, "original", "manifest")
    assert prefix["assignment"]["max_new_tokens"] == 4096
    record["inference"]["sampling"]["max_new_tokens"] = 2048
    with pytest.raises(ValueError, match="Actual source generation allowance"):
        natural.make_prefix(record, "original", "manifest")


def test_real_transformer_new_reasoning_has_paired_shams_and_retains_invalids(tmp_path):
    from test_state_backend import tiny_backend
    import torch

    # Exact duplicate-row identity is required in the production BF16 regime.
    # Float32 CPU kernels can differ by a final rounding bit across row layouts.
    model = tiny_backend(dtype=torch.bfloat16)
    model.metadata = lambda: dict(model="tiny-engineering-transformer")
    manifest = fixture_collection(tmp_path)
    rows = natural.execute(tmp_path / "natural", manifest, model, artifact_root=tmp_path)
    assert len(rows) == 4 and digest(rows) == digest(RunStore(tmp_path / "natural", manifest).completed())
    for key, prefix in manifest["prefixes"].items():
        baseline, sham = rows[f"{key}-baseline"], rows[f"{key}-sham"]
        assert baseline["measurement"]["completion"] == sham["measurement"]["completion"]
        assert baseline["measurement"]["unpadded_input_ids"] == prefix["token_prefix"]
        assert baseline["measurement"]["label_logits"] == sham["measurement"]["label_logits"]
        assert sham["measurement"]["intervention"]["delta_norm"] == 0
        assert not baseline["thinking_closed"] and not baseline["valid"] and not baseline["correct"]
        assert baseline["measurement"]["completion"]["sampling"]["seed"] == prefix["seed"]
        assert baseline["cap_hit"]
        with np.load(tmp_path / "natural" / baseline["activation_artifact"]["path"], allow_pickle=False) as z:
            assert list(z["hidden"].shape) == [4, 3, 2, 16]
    assert model.active_hooks == 0


def test_4096_execution_persists_actual_sampling_scoring_and_capture_allowance(tmp_path):
    from test_state_backend import tiny_backend

    model = tiny_backend()
    model.metadata = lambda: dict(model="tiny-engineering-transformer")
    # Exercise the complete executor at its real allowance without thousands of
    # CPU steps: every vocabulary token terminates this engineered test model.
    model.model.generation_config.eos_token_id = list(range(128))
    manifest = fixture_collection(tmp_path, max_new_tokens=4096, source_max_new_tokens=4096)
    rows = natural.execute(tmp_path / "natural", manifest, model, artifact_root=tmp_path)
    assert len(rows) == 4
    for row in rows.values():
        completion = row["measurement"]["completion"]
        assert completion["sampling"]["max_new_tokens"] == 4096
        assert row["generation_max_new_tokens"] == row["activation_artifact"]["max_new_tokens"] == 4096
        assert len(completion["generated_token_ids"]) == 1 and not row["cap_hit"]
    assert model.active_hooks == 0


def test_real_transformer_sampling_uses_independent_row_generators():
    from test_state_backend import tiny_backend
    from curiosity_study.state_backend import StateBackend

    model = tiny_backend()
    request = dict(
        token_prefix=[12, 13, 14, 15],
        token_prefix_sha256=digest([12, 13, 14, 15]),
        prefix_positions={"record": 1, "final": 3},
        labels=["Yes", "No"],
        seed=721,
    )
    second = {**request, "seed": 933}
    backend = StateBackend(model)
    rows, _ = backend.forward([request, second], max_new_tokens=16)
    reversed_rows, _ = backend.forward([second, request], max_new_tokens=16)
    changed_first, _ = backend.forward([{**request, "seed": 788}, second], max_new_tokens=16)
    assert (
        rows[0]["completion"]["generated_token_ids"] == reversed_rows[1]["completion"]["generated_token_ids"]
    )
    assert (
        rows[1]["completion"]["generated_token_ids"] == reversed_rows[0]["completion"]["generated_token_ids"]
    )
    assert (
        rows[1]["completion"]["generated_token_ids"] == changed_first[1]["completion"]["generated_token_ids"]
    )
    assert rows[0]["completion"]["generated_token_ids"] != rows[1]["completion"]["generated_token_ids"]
    audit = natural.verify_natural_backend(model, [request, second])
    assert audit["passed"], audit["checks"]


def test_thinking_closure_and_optional_period_grammar_are_preserved():
    a = assignments(1, kinds=("new",))[0]
    expected = inference_request(a)["labels"][0]
    for raw, valid, closed in [
        (f"Some reasoning. </think> {expected}.", True, True),
        (f"Some reasoning. {expected}", False, False),
        (f"Some reasoning. </think> {expected} because", False, True),
        ("Some reasoning. </think> ", False, True),
    ]:
        scored = record_from_inference(a, {"raw": raw})
        assert scored["valid"] is valid and scored["thinking_closed"] is closed


def test_scientific_token_cap_and_early_positions_cannot_silently_change(tmp_path):
    manifest = fixture_collection(tmp_path)
    with pytest.raises(ValueError, match="full 2048 or 4096"):
        natural._validate_manifest({**manifest, "phase": "confirmation"})
    bad = copy.deepcopy(manifest)
    bad["diagnostics"][1]["position"] = "think_end"
    bad["assignment_sha256"] = digest(bad["diagnostics"])
    with pytest.raises(ValueError, match="record or final"):
        natural._validate_manifest(bad)


def test_backend_accepts_long_reasoning_allowance_and_continues_beyond_short_replay_cap():
    from test_state_backend import tiny_backend
    from curiosity_study.state_backend import StateBackend

    model = tiny_backend()
    model.model.generation_config.eos_token_id = None
    ids = [12, 13, 14, 15]
    request = dict(
        token_prefix=ids,
        token_prefix_sha256=digest(ids),
        prefix_positions={"record": 1, "final": 3},
        labels=["Yes", "No"],
        seed=721,
    )
    rows, _ = StateBackend(model).forward([request], max_new_tokens=64, greedy=True)
    assert len(rows[0]["completion"]["generated_token_ids"]) == 64
    assert not rows[0]["completion"]["ended_with_eos"]
    # The full scientific allowance is accepted; a designated immediate EOS
    # keeps this boundary test small rather than adding 4,096 CPU decode steps.
    model.model.generation_config.eos_token_id = list(range(128))
    rows, _ = StateBackend(model).forward([request], max_new_tokens=4096, greedy=True)
    assert len(rows[0]["completion"]["generated_token_ids"]) == 1
    assert rows[0]["completion"]["ended_with_eos"]
    assert rows[0]["completion"]["sampling"]["max_new_tokens"] == 4096
    with pytest.raises(ValueError, match="4096"):
        StateBackend(model).forward([request], max_new_tokens=4097)


def test_execution_error_retains_the_entire_assigned_cohort(tmp_path, monkeypatch):
    from test_state_backend import tiny_backend

    manifest = fixture_collection(tmp_path)
    manifest["diagnostics"][1]["layer"] = 999
    manifest["assignment_sha256"] = digest(manifest["diagnostics"])
    model = tiny_backend()
    model.metadata = lambda: dict(model="tiny-engineering-transformer")
    monkeypatch.setattr(natural, "verify_natural_backend", lambda *args: {"passed": True})
    with pytest.raises(ValueError, match="Invalid layer"):
        natural.execute(tmp_path / "natural", manifest, model, artifact_root=tmp_path)
    stored = RunStore(tmp_path / "natural", manifest).completed()
    assert set(stored) == {a["trial_id"] for a in manifest["diagnostics"]}
    assert all(row["terminal"] and row["failure"] == "execution_error" for row in stored.values())
    assert model.active_hooks == 0


def test_partial_cohort_resume_is_rejected_without_generating_more_trials(tmp_path, monkeypatch):
    from test_state_backend import tiny_backend

    manifest = fixture_collection(tmp_path)
    store = RunStore(tmp_path / "natural", manifest)
    store.save_many([dict(trial_id=manifest["diagnostics"][0]["trial_id"], terminal=True)], "bad", "0000000")
    model = tiny_backend()
    model.metadata = lambda: dict(model="tiny-engineering-transformer")
    monkeypatch.setattr(natural, "verify_natural_backend", lambda *args: {"passed": True})
    with pytest.raises(ValueError, match="Partial cohort"):
        natural.execute(tmp_path / "natural", manifest, model, artifact_root=tmp_path)
    assert len(store.completed()) == 1


def test_natural_schedule_uses_only_early_axes_and_keeps_worlds_on_one_worker(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "prepare_natural_interventions", scripts / "prepare_natural_interventions.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prefixes = {}
    for a in assignments(2, kinds=("new", "repeat")):
        prefixes[a["trial_id"]] = natural.make_prefix(source_record(a), "original", "manifest")
    fitted = dict(
        sources=np.array(["objects:balanced"]),
        layers=np.array([1]),
        positions=np.array(["record", "prefill"]),
        contrast_norm=np.ones((1, 1, 2)),
        unit=np.ones((1, 1, 2, 16)) / 4,
    )
    config = dict(
        name="natural-fixture",
        batch_size=4,
        layers=[1],
        positions=["record", "prefill"],
        dose_reference_source="objects:balanced",
        order_seed=173,
        directions=[dict(name="information", source="objects:balanced", array="unit")],
        dose_multipliers=[-1, 1],
        random_seeds=[733],
        random_dose_multipliers=[1],
    )
    rows = module.build_schedule(config, prefixes, fitted)
    assert rows == module.build_schedule(config, prefixes, fitted)
    assert len(rows) == len(prefixes) * 12
    for key in prefixes:
        local = [a for a in rows if a["prefix_id"] == key]
        assert sum(a["arm"] == "baseline" for a in local) == 1
        assert {a["position"] for a in local if a["arm"] != "baseline"} == {"record", "final"}
        assert {a["capture_position"] for a in local if a["arm"] == "direction"} == {"record", "prefill"}
    parts = module.partition_worlds(rows, prefixes, 4, 4)
    by_world = {}
    for worker, cohorts in enumerate(parts):
        for index in cohorts:
            for a in rows[index * 4 : (index + 1) * 4]:
                world = module.world_id(prefixes[a["prefix_id"]]["assignment"])
                assert by_world.setdefault(world, worker) == worker
    with pytest.raises(ValueError, match="never generated-token"):
        module.build_schedule({**config, "positions": ["decision"]}, prefixes, fitted)
