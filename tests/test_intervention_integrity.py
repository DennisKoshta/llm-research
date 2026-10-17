from curiosity_study.intervention_integrity import audit_intervention


def test_projection_uses_runtime_recipient_and_records_small_rounding_limited_shift():
    assignment = dict(trial_id="self", patch_mode="project", projection_target=2.0, dose=0.0)
    change = dict(
        mode="project",
        requested_scalar_delta=0.001,
        reference_projection_before=1.999,
        projection_before=1.999,
        projection_after=1.999,
        direction_norm=1.0,
        requested_delta_norm=0.001,
        delta_norm=0.0,
        relative_delta_norm=0.0,
        rounding_error_norm=0.001,
    )
    audit = audit_intervention(assignment, change)
    assert audit["implementation_verified"] and not audit["resolved_beyond_rounding"]
    assert not audit["signed_engagement"]
    bad = audit_intervention(assignment, {**change, "projection_after": 2.5})
    assert not bad["implementation_verified"]


def test_matched_control_needs_reference_scalar_and_real_target_axis_engagement():
    assignment = dict(trial_id="random", patch_mode="matched_add", projection_target=4.0, dose=1.5)
    change = dict(
        mode="matched_add",
        requested_scalar_delta=2.0,
        reference_projection_before=2.0,
        projection_before=7.0,
        projection_after=9.0,
        direction_norm=1.0,
        requested_delta_norm=2.0,
        delta_norm=2.0,
        relative_delta_norm=0.1,
        rounding_error_norm=0.0,
    )
    assert audit_intervention(assignment, change)["implementation_verified"]
    assert not audit_intervention(assignment, {**change, "reference_projection_before": 2.5})[
        "implementation_verified"
    ]
    assert not audit_intervention(assignment, {**change, "projection_after": 5.0})["implementation_verified"]
