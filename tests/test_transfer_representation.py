import numpy as np
import pytest
from curiosity_study.transfer_representation import transfer_groups, project_frozen, transfer_estimand
from curiosity_study import information_state, relational_information, target_relevant_information as target


def test_transfer_groups_respect_relations_and_whole_worlds():
    for module in (information_state, relational_information):
        rows = module.assignments(items=2, seed=746, kinds=("new", "repeat"))
        groups = transfer_groups(rows)
        assert len(groups) == 8
        for indices in groups.values():
            assert [module.task(rows[i])["information_bits"] for i in indices] == [0, 1, 1, 0]
        with pytest.raises(ValueError, match="Incomplete"):
            transfer_groups(rows[:-1])
        with pytest.raises(ValueError, match="Duplicate"):
            transfer_groups(rows + [rows[0]])
    rows = information_state.assignments(items=1, kinds=("new",))
    with pytest.raises(ValueError, match="specified"):
        transfer_groups([{**a, "task_revision": "unrecognized"} for a in rows])


def test_frozen_projection_preserves_source_sign_and_missingness():
    axis = np.array([1.0, 0.0, 0.0]).reshape(1, 1, 1, 3)
    vectors = np.array([[2, 100, 1], [-2, 100, 1], [np.nan, np.nan, np.nan]]).reshape(3, 1, 1, 3)
    scored = project_frozen(vectors, axis).ravel()
    assert scored[:2].tolist() == [2, -2]
    assert np.isnan(scored[2])
    assert np.array_equal(project_frozen(vectors[:2], -axis).ravel(), [-2, 2])
    with pytest.raises(ValueError, match="missing"):
        project_frozen(vectors, axis * np.nan)


def test_target_transfer_names_the_varying_quantity_and_rejects_mixed_estimands():
    rows = target.assignments(items=2, kinds=("new", "repeat"))
    groups = transfer_groups(rows)
    assert len(groups) == 8
    estimand = transfer_estimand(rows)
    assert estimand["information_quantity"] == "expected_target_information_gain_bits"
    assert estimand["full_world_information_quantity"] == "information_bits"
    assert estimand["full_world_information_gain_bits"] == 1
    assert estimand["offered_report_entropy_bits"] == 1
    for indices in groups.values():
        tasks = [target.task(rows[i]) for i in indices]
        assert [t["expected_target_information_gain_bits"] for t in tasks] == [0, 1, 1, 0]
        assert [t["information_bits"] for t in tasks] == [1, 1, 1, 1]
    with pytest.raises(ValueError, match="Cannot pool"):
        transfer_groups(rows + information_state.assignments(items=1, kinds=("new",)))
    assert transfer_estimand(information_state.assignments())["information_quantity"] == "information_bits"
    with pytest.raises(ValueError, match="Incomplete"):
        transfer_groups(rows[:-1])
    with pytest.raises(ValueError, match="Duplicate"):
        transfer_groups(rows + [rows[0]])
