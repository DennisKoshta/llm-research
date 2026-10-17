"""Verify requested intervention geometry, including finite-precision projections."""

import math


def audit_intervention(assignment, change):
    mode = assignment.get("patch_mode", "add")
    if change.get("mode", "add") != mode:
        raise ValueError("Executed intervention mode differs from its assignment")
    scalar = change.get("requested_scalar_delta", assignment["dose"])
    projection_delta = change["projection_after"] - change["projection_before"]
    rounding = change.get("rounding_error_norm", 0.0)
    # Vector quantization gives an L2 bound on its projection error. The small
    # arithmetic allowance covers float32 norm/dot-product accumulation only.
    allowance = rounding + 1e-4 * (1 + abs(scalar) + abs(change["projection_before"]))
    expected_scalar = assignment["dose"]
    target_error = None
    if mode in ("project", "matched_add"):
        expected_scalar = assignment["projection_target"] - change["reference_projection_before"]
        if mode == "project":
            target_error = abs(change["projection_after"] - assignment["projection_target"])
    norm = change.get("direction_norm", 1.0)
    scalar_agrees = math.isclose(scalar, expected_scalar, rel_tol=1e-5, abs_tol=1e-5)
    geometry_agrees = abs(projection_delta - scalar * norm) <= allowance
    if "requested_delta_norm" in change:
        geometry_agrees &= abs(
            change["delta_norm"] - change["requested_delta_norm"]
        ) <= allowance and math.isclose(
            change["requested_delta_norm"], abs(scalar) * norm, rel_tol=1e-3, abs_tol=1e-4
        )
    if target_error is not None:
        geometry_agrees &= target_error <= allowance
    finite = all(
        math.isfinite(value)
        for value in (scalar, projection_delta, change["delta_norm"], change["relative_delta_norm"], rounding)
    )
    # Preserve the original strict additive-dose engagement rule. Projection
    # self-controls and tiny donor differences can legitimately round to zero.
    signed = scalar * projection_delta > 0 if scalar else projection_delta == 0
    resolved = abs(scalar * norm) > allowance
    verified = finite and scalar_agrees and geometry_agrees and (signed or not resolved)
    if mode == "add" and assignment["dose"]:
        verified &= signed and change["delta_norm"] > 0
    return dict(
        trial_id=assignment["trial_id"],
        mode=mode,
        requested_dose=float(scalar),
        scheduled_reference_dose=float(assignment["dose"]),
        projection_delta=projection_delta,
        delta_norm=change["delta_norm"],
        relative_delta_norm=change["relative_delta_norm"],
        rounding_error_norm=rounding,
        numerical_allowance=allowance,
        resolved_beyond_rounding=resolved,
        signed_engagement=bool(signed),
        target_projection_error=target_error,
        implementation_verified=bool(verified),
    )
