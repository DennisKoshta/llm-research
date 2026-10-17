import numpy as np
import pytest
from curiosity_study.artifacts import digest
from curiosity_study.replay_information import make_prefix, intervention


def source_record():
    return dict(
        trial_id="source",
        inference=dict(unpadded_input_ids=[11, 12, 13], generated_token_ids=[20, 21, 22, 100, 127]),
        activation_artifact=dict(
            positions=["record", "prefill", "think_end", "decision"], token_positions=[1, 2, 4, 5]
        ),
        thinking_closed=True,
        assignment=dict(sampling_seed=123),
        decision_label_logits={"Yes": 2.0, "No": -1.0},
        decision_label_probabilities_temperature_1={"Yes": 0.7, "No": 0.01},
        parsed_answer="Yes",
        correct=True,
        valid=True,
        closing_delimiter_count=1,
    )


def test_prefix_excludes_answer_but_preserves_every_prior_token():
    r = source_record()
    p = make_prefix(r, "run", "manifest")
    assert p["token_prefix"] == [11, 12, 13, 20, 21, 22]
    assert p["token_prefix_sha256"] == digest(p["token_prefix"])
    assert p["source_record_sha256"] == digest(r)
    assert p["prefix_positions"] == dict(record=1, think_end=4, final=5)
    r["activation_artifact"]["token_positions"][-1] = -1
    assert make_prefix(r, "run", "manifest") is None


def test_prefix_rejects_position_that_cannot_precede_an_answer():
    r = source_record()
    r["activation_artifact"]["token_positions"][-1] = 7
    with pytest.raises(ValueError, match="exclude"):
        make_prefix(r, "run", "manifest")


def test_random_control_is_reproducible_unit_norm_with_explicit_dose():
    a = dict(arm="random", layer=15, position="think_end", dose=5.0, random_seed=713)
    p = intervention(a, None, 20)
    np.testing.assert_array_equal(p["direction"], intervention(a, None, 20)["direction"])
    assert abs(np.linalg.norm(p["direction"]) - 1) < 1e-10
    assert p["dose"] == 5
    assert intervention({**a, "arm": "sham"}, None, 20)["dose"] == 0


def test_matched_random_control_binds_reference_axis_and_target():
    source = np.ones((1, 1, 1, 20))
    assignment = dict(
        arm="random",
        layer=15,
        position="think_end",
        dose=0.0,
        random_seed=713,
        patch_mode="matched_add",
        projection_target=3.2,
        reference_source_index=0,
        layer_index=0,
        position_index=0,
    )
    patch = intervention(assignment, {"unit": source}, 20)
    assert patch["mode"] == "matched_add" and patch["target"] == 3.2
    np.testing.assert_allclose(patch["reference_direction"], np.ones(20) / np.sqrt(20))


def test_real_transformer_replay_collection_retains_invalids_and_sham_identity(tmp_path, monkeypatch):
    from test_state_backend import tiny_backend
    import torch
    from curiosity_study import replay_information as replay
    from curiosity_study.information_state import assignments
    from curiosity_study.artifacts import RunStore

    model = tiny_backend(dtype=torch.bfloat16)
    model.metadata = lambda: dict(model="tiny-engineering-transformer")
    aa = assignments(1, kinds=("new",))[:2]
    prefixes = {}
    for index, a in enumerate(aa):
        ids = [12, 13, 14, 15, 16] + [17] * index
        prefixes[str(index)] = dict(
            token_prefix=ids,
            token_prefix_sha256=digest(ids),
            prefix_positions=dict(record=1, think_end=len(ids) - 2, final=len(ids) - 1),
            labels=["Yes", "No"],
            seed=917,
            assignment=a,
            source_trial_id=a["trial_id"],
            reference_label_logits={"Yes": 0.0, "No": 0.0},
            reference_label_probabilities_temperature_1={"Yes": 0.1, "No": 0.1},
        )
    diagnostics = []
    for key in prefixes:
        for arm in ("baseline", "sham"):
            diagnostics.append(
                dict(
                    trial_id=f"{key}-{arm}",
                    prefix_id=key,
                    arm=arm,
                    kind="new",
                    layer=1,
                    position="think_end",
                    dose=0.0,
                )
            )
    manifest = dict(
        assay="replay_information",
        phase="engineering",
        prefixes=prefixes,
        assignments=[],
        diagnostics=diagnostics,
        execution=dict(batch_size=4),
        capture_layers=[0, 1, 2],
        max_new_tokens=3,
        greedy=True,
    )
    monkeypatch.setattr(replay, "source_check", lambda p, c: None)
    rows = replay.execute(tmp_path, manifest, model)
    assert len(rows) == 4
    assert digest(RunStore(tmp_path, manifest).completed()) == digest(rows)
    for key in prefixes:
        b, s = rows[f"{key}-baseline"], rows[f"{key}-sham"]
        assert (
            b["measurement"]["completion"]["generated_token_ids"]
            == s["measurement"]["completion"]["generated_token_ids"]
        )
        assert b["measurement"]["label_logits"] == s["measurement"]["label_logits"]
        assert b["measurement"]["unpadded_input_ids"] == prefixes[key]["token_prefix"]
        assert s["measurement"]["intervention"]["delta_norm"] == 0
        assert b["valid"] is False and b["correct"] is False
        meta = b["activation_artifact"]
        with np.load(tmp_path / meta["path"], allow_pickle=False) as z:
            assert list(z["hidden"].shape) == [4, 3, 3, 16]
            assert z["trial_ids"][meta["row"]] == b["trial_id"]
    assert model.active_hooks == 0
