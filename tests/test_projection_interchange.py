import copy
import hashlib
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from curiosity_study import information_state as state, relational_information as relational
from curiosity_study.artifacts import RunStore, digest
from curiosity_study.replay_information import intervention
from curiosity_study.projection_interchange import (
    complete_groups,
    donor_ids,
    build_recipient_rows,
    vector_sha256,
)


@pytest.mark.parametrize("module", [state, relational])
def test_interchange_pairs_real_factorial_donors_without_crossing_worlds(module):
    assignments = module.assignments(2, kinds=("new", "repeat"))
    groups = complete_groups(assignments)
    by_id = {a["trial_id"]: a for a in assignments}
    assert len(groups) == 8
    for cells in groups.values():
        for cell, recipient in cells.items():
            donors = donor_ids(cells, cell)
            original = module.task(by_id[recipient])
            assert donors["self"] == recipient
            assert (
                module.task(by_id[donors["opposite_gain"]])["information_bits"]
                == 1 - original["information_bits"]
            )
            assert module.task(by_id[donors["same_gain"]])["information_bits"] == original["information_bits"]
            assert all(by_id[key]["kind"] == by_id[recipient]["kind"] for key in donors.values())
    with pytest.raises(ValueError, match="complete assigned"):
        complete_groups(assignments[:-1])
    with pytest.raises(ValueError, match="Duplicate"):
        complete_groups(assignments + assignments[:1])


def synthetic_schedule():
    assignments = [a for a in state.assignments(1, kinds=("new",)) if a["family"] == "objects"]
    cells = next(iter(complete_groups(assignments).values()))
    by_id = {a["trial_id"]: a for a in assignments}
    # Deliberately reversed information ordering: it must not be repaired or excluded.
    first_components = {(0, 0): 10.0, (0, 1): 8.0, (1, 0): 7.0, (1, 1): 11.0}
    captures = {}
    for cell, trial in cells.items():
        vector = np.array([first_components[cell], cell[0] + 1, cell[1] + 2], dtype=np.float32)
        captures[trial] = dict(vector=vector, metadata=dict(vector_sha256=vector_sha256(vector)))
    config = dict(
        name="synthetic-interchange",
        layer=31,
        position="decision",
        random_seeds=[3, 5, 7, 11, 13],
        batch_size=8,
    )
    indices = dict(information_source=0, answer_source=1, layer=0, position=0)
    recipient = cells[(0, 0)]
    rows = build_recipient_rows(
        config,
        recipient,
        dict(assignment=by_id[recipient]),
        donor_ids(cells, (0, 0)),
        captures,
        [1, 0, 0],
        indices,
    )
    return config, indices, captures, recipient, rows


def test_projection_targets_are_observed_donors_and_controls_match_runtime_reference():
    config, indices, captures, recipient, rows = synthetic_schedule()
    assert len(rows) == 24 and len(rows) % config["batch_size"] == 0
    assert len({a["trial_id"] for a in rows}) == len(rows)
    assert all(a["prefix_id"] == recipient for a in rows)
    opposite = next(a for a in rows if a["condition"] == "information_project_opposite_gain")
    assert opposite["projection_target"] == 8 and opposite["dose"] == -2
    assert opposite["expected_gain_shift"] == 1
    assert opposite["donor_role"] == "opposite_information"
    assert opposite["comparison_set"] == "interchange_opposite_information"
    assert not opposite["original_projection_order_matches_semantics"]
    assert opposite["recipient_vector_sha256"] == captures[recipient]["metadata"]["vector_sha256"]
    same = next(a for a in rows if a["condition"] == "information_project_same_gain")
    assert same["projection_target"] == 11 and same["dose"] == 1 and same["expected_gain_shift"] == 0
    controls = [a for a in rows if a.get("matched_interchange_condition") == opposite["condition"]]
    assert len(controls) == 6 and sum(a["arm"] == "random" for a in controls) == 5
    for a in controls:
        assert a["patch_mode"] == "matched_add"
        assert a["dose"] == opposite["dose"]
        assert a["projection_target"] == opposite["projection_target"]
        assert a["reference_source_index"] == indices["information_source"]
        assert a["reference_array"] == "unit"
        assert a["comparison_set"] == opposite["comparison_set"]
    self_donor = next(a for a in rows if a["condition"] == "information_project_self")
    assert self_donor["arm"] == "direction" and self_donor["patch_mode"] == "project"
    assert self_donor["dose"] == 0 and self_donor["projection_target"] == 10
    true_sham = next(a for a in rows if a["condition"] == "zero_sham")
    assert true_sham["arm"] == "sham" and true_sham["dose"] == 0


def test_random_controls_and_vector_integrity_fail_closed():
    config, indices, captures, recipient, _ = synthetic_schedule()
    assignment = next(a for a in state.assignments(1, kinds=("new",)) if a["trial_id"] == recipient)
    config["random_seeds"] = [1, 2, 3, 4, 4]
    with pytest.raises(ValueError, match="five distinct"):
        build_recipient_rows(config, recipient, dict(assignment=assignment), {}, captures, [1, 0, 0], indices)
    with pytest.raises(ValueError, match="finite source"):
        vector_sha256([1, np.nan])
    assert vector_sha256(np.array([1, 2], dtype=np.float64)) == vector_sha256(
        np.array([1, 2], dtype=np.float32)
    )


def test_every_schedule_arm_resolves_against_the_actual_replay_engine():
    _, _, _, _, rows = synthetic_schedule()
    fitted = {"unit": np.array([[[[1.0, 0.0, 0.0]]], [[[0.0, 1.0, 0.0]]]])}
    for a in rows:
        patch = intervention(a, fitted, 3)
        if a["arm"] == "baseline":
            assert patch is None
            continue
        if a.get("patch_mode") == "matched_add":
            assert patch["mode"] == "matched_add" and patch["target"] == a["projection_target"]
            np.testing.assert_array_equal(patch["reference_direction"], [1, 0, 0])
            assert np.isclose(np.linalg.norm(patch["direction"]), 1)
        elif a.get("patch_mode") == "project":
            assert patch["mode"] == "project" and patch["target"] == a["projection_target"]


def load_prepare():
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_projection_interchange.py"
    spec = importlib.util.spec_from_file_location("prepare_projection_interchange", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_fixture(tmp_path):
    aa = [a for a in state.assignments(1, kinds=("new",)) if a["family"] == "objects"]
    manifest = dict(diagnostics=aa, assignment_sha256=digest(aa))
    store = RunStore(tmp_path, manifest)
    path = tmp_path / "activations/synthetic.npz"
    path.parent.mkdir()
    hidden = np.arange(36, dtype=np.float32).reshape(4, 1, 3, 3)
    positions = ["record", "think_end", "decision"]
    np.savez(
        path,
        hidden=hidden,
        trial_ids=np.array([a["trial_id"] for a in aa]),
        layers=np.array([31]),
        positions=np.array(positions),
        available=np.ones((4, 1, 3), dtype=bool),
    )
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    records = []
    for row, a in enumerate(aa):
        record = dict(
            trial_id=a["trial_id"],
            assignment=a,
            terminal=True,
            task=state.task(a),
            inference=dict(unpadded_input_ids=[10, 20], generated_token_ids=[30, 40, 50]),
            activation_artifact=dict(
                path="activations/synthetic.npz",
                sha256=sha,
                row=row,
                layers=[31],
                positions=positions,
                token_positions=[0, 2, 3],
                available=[[True, True, True]],
            ),
            thinking_closed=True,
            decision_label_logits={"Yes": 2, "No": 1},
            decision_label_probabilities_temperature_1={"Yes": 0.6, "No": 0.3},
            parsed_answer="Yes",
            correct=False,
            valid=True,
            closing_delimiter_count=1,
        )
        records.append(record)
    store.save_many(records, "synthetic", "cohort")
    return manifest, aa, hidden, path


def test_source_loader_binds_actual_capture_values_and_keeps_incorrect_sources(tmp_path):
    module = load_prepare()
    manifest, assignments, hidden, path = source_fixture(tmp_path)
    prefixes, captures, missing = module.load_source_vectors(tmp_path, manifest, assignments, 31, "decision")
    assert len(prefixes) == len(captures) == 4 and not missing
    for row, a in enumerate(assignments):
        capture = captures[a["trial_id"]]
        np.testing.assert_array_equal(capture["vector"], hidden[row, 0, 2])
        assert capture["metadata"]["vector_sha256"] == vector_sha256(hidden[row, 0, 2])
        assert not prefixes[a["trial_id"]]["original_correct"]  # No correctness selection.
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        module.load_source_vectors(tmp_path, manifest, assignments, 31, "decision")


def test_template_branches_remain_separate_complete_interchange_groups():
    original = state.assignments(1, kinds=("new",))
    prose = copy.deepcopy(original)
    for a in prose:
        a["template"] = "prose"
        a["trial_id"] += "-prose"
    groups = complete_groups(original + prose)
    assert len(groups) == 4
    for key, cells in groups.items():
        assert all(("-prose" in trial) == (key[-2] == "prose") for trial in cells.values())
