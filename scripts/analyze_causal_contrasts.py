"""Analyze audited causal outcomes under a frozen or explicitly post-hoc plan.

Confirmation reads intervention_schedule.planned_contrasts from the immutable
parent manifest only. Development may use --config with --posthoc. No inference
is performed. Example: python scripts/analyze_causal_contrasts.py --run RUN
--audit-dir AUDIT --out CONTRASTS.
"""

import argparse
from pathlib import Path
from curiosity_study.artifacts import digest, read, write_once
from curiosity_study.causal_contrasts import (
    CATEGORICAL_TYPES,
    RESPONSE_OUTCOMES,
    RESPONSE_OUTCOME_SCHEMA,
    analyze_contrasts,
    response_objective,
    validate_natural_response_row,
    validate_plan,
)


LOGIT_TYPES = {
    "new": "information_gain_yes_minus_no_with_polarity",
    "repeat": "information_gain_yes_minus_no_with_polarity",
    "instrument": "report_minus_stop",
    "memory": "correct_minus_best_alternative",
    "count": "correct_minus_best_alternative",
}
NATURAL_READINESS_CHECKS = {
    "execution_integrity",
    "scientific_phase",
    "prior_32b_engineering_bound",
    "complete_source_worlds",
    "frozen_baseline_competence_and_validity_gates",
}


def admit_outcomes(manifest, audit, compact):
    """Require one audited measurement stage and one unit system throughout.

    Pairwise margin-type agreement is insufficient: two families could each
    contain internally consistent pairs measured in different units and then
    be pooled. Validate every row against its manifest's assay before analysis.
    """
    assay = manifest.get("assay")
    if assay not in ("replay_information", "natural_information"):
        raise ValueError("An explicit supported assay is required to establish causal outcome units")
    if assay == "natural_information":
        checks = audit.get("interpretation_checks")
        previous = audit.get("previous_engineering")
        baseline = audit.get("baseline_readiness")
        if (
            audit.get("assay") != assay
            or manifest.get("phase") == "engineering"
            or not isinstance(manifest.get("phase"), str)
            or audit.get("phase") != manifest["phase"]
            or audit.get("eligible_for_interpretation") is not True
            or not isinstance(checks, dict)
            or not NATURAL_READINESS_CHECKS <= set(checks)
            or any(value is not True for value in checks.values())
            or not isinstance(previous, dict)
            or previous.get("passed") is not True
            or not isinstance(baseline, dict)
            or baseline.get("passed") is not True
        ):
            raise ValueError(
                "Natural scientific statistics require a passing interpretation-readiness audit; engineering integrity alone is insufficient"
            )
        model, engine = manifest.get("model"), manifest.get("engine_sha256")
        if (
            not isinstance(model, dict)
            or not all(isinstance(model.get(key), str) and model[key] for key in ("model_id", "revision"))
            or not isinstance(engine, dict)
            or not engine
            or audit.get("model") != model
            or audit.get("engine_sha256") != engine
        ):
            raise ValueError("Natural readiness audit model/engine binding mismatch")
        if audit.get("response_outcome_schema") != RESPONSE_OUTCOME_SCHEMA:
            raise ValueError("Natural readiness audit must bind the explicit response-outcome schema")
        expected_types = CATEGORICAL_TYPES
        units = "categorical response score; not logits"
        measurement = "newly_generated_completed_final_answer"
    else:
        # The existing replay analyzer predates its explicit assay output field.
        if audit.get("assay", assay) != assay:
            raise ValueError("Replay outcome audit has a different assay")
        expected_types = LOGIT_TYPES
        units = "logit difference"
        measurement = "fixed_prefix_next_token_logit_margin"
    if not compact:
        raise ValueError("Nonempty audited causal outcomes are required")
    objectives = {response_objective(row) for row in compact}
    if assay == "natural_information" and len(objectives) != 1:
        raise ValueError("Natural target and full-world response objectives require separate analyses")
    for row in compact:
        kind = row.get("kind")
        if kind not in expected_types or row.get("margin_type") != expected_types[kind]:
            raise ValueError(
                "Outcome type differs from its assay; categorical responses and replay logits cannot be pooled"
            )
        declared_units = row.get("margin_units")
        if assay == "natural_information":
            allowed = (0, 1) if kind in ("memory", "count") else (-1, 0, 1)
            if (
                declared_units != units
                or type(row.get("margin")) not in (int, float)
                or row["margin"] not in allowed
            ):
                raise ValueError(
                    "Natural outcomes must retain their categorical response-score units and values"
                )
            if type(row.get("valid")) is not bool or (not row["valid"] and row["margin"] != 0):
                raise ValueError("Invalid natural completions must remain neutral categorical outcomes")
            if kind in ("new", "repeat") and row["valid"] and row["margin"] not in (-1, 1):
                raise ValueError("Valid natural information judgments must use a signed categorical response")
            if kind in ("memory", "count") and (
                type(row.get("correct")) is not bool or row["margin"] != int(row["valid"] and row["correct"])
            ):
                raise ValueError("Natural competence outcomes must be correct-answer indicators")
            validate_natural_response_row(row)
        elif declared_units is not None and declared_units != units:
            raise ValueError("Replay logit margins cannot carry categorical outcome units")
        elif "response_outcome_schema" in row or any(key in row for key in RESPONSE_OUTCOMES):
            raise ValueError("Natural binary indicators cannot be relabeled as replay logit outcomes")
    return dict(
        assay=assay,
        outcome_measurement=measurement,
        outcome_units=units,
        outcome_types_by_kind={kind: expected_types[kind] for kind in sorted({r["kind"] for r in compact})},
        cross_assay_pooling_permitted=False,
        response_outcome_schema=RESPONSE_OUTCOME_SCHEMA if assay == "natural_information" else None,
        response_objective_basis=next(iter(objectives))
        if len(objectives) == 1
        else "separate revision endpoints",
    )


def select_plan(manifest, override=None, posthoc=False):
    scope = manifest.get("inference_scope")
    if scope not in ("development", "confirmation"):
        raise ValueError("A bound development or confirmation scope is required")
    schedule = manifest["intervention_schedule"]
    if digest(schedule) != manifest.get("intervention_schedule_sha256"):
        raise ValueError("Intervention schedule hash mismatch")
    if scope == "confirmation" and (override is not None or posthoc):
        raise ValueError("Confirmation permits only the planned contrasts frozen in the parent manifest")
    if override is not None:
        if not posthoc:
            raise ValueError("A development override requires the explicit --posthoc flag")
        plan, origin = override, "explicit_posthoc_development"
    else:
        if posthoc:
            raise ValueError("The --posthoc flag requires an explicit development config")
        if "planned_contrasts" not in schedule:
            raise ValueError("No planned_contrasts were frozen in the parent manifest")
        plan, origin = schedule["planned_contrasts"], "frozen_parent_manifest"
    validate_plan(plan)
    return plan, origin


def assigned_sources(manifest):
    """Recover all assigned source worlds, including prefixes excluded before replay."""
    source = Path(manifest["intervention_schedule"]["source_collection"])
    parent = read(source / "manifest.json")
    if digest(parent) != manifest["source_parent_manifest_sha256"]:
        raise ValueError("Original source parent manifest hash mismatch")
    if digest(parent["diagnostics"]) != parent.get("assignment_sha256"):
        raise ValueError("Original source assignment hash mismatch")
    eligible = set(manifest["prefixes"])
    excluded = manifest["excluded_sources"]
    if len(set(excluded)) != len(excluded) or eligible & set(excluded):
        raise ValueError("Source eligibility and exclusion sets must be unique and disjoint")
    wanted = eligible | set(excluded)
    if len(wanted) != manifest["assigned_sources"]:
        raise ValueError("Assigned source coverage does not reconcile")
    selected = [a for a in parent["diagnostics"] if a["trial_id"] in wanted]
    if len(selected) != len(wanted) or {a["trial_id"] for a in selected} != wanted:
        raise ValueError("Missing or duplicate original source assignments")
    provenance = manifest["split_provenance"]
    if digest(provenance) != manifest["split_provenance_sha256"]:
        raise ValueError("Split provenance hash mismatch")
    if digest(selected) != provenance.get("evaluation_assignments_sha256"):
        raise ValueError("Selected source assignments differ from the prospectively bound selection")
    for a in selected:
        if a["trial_id"] in eligible and manifest["prefixes"][a["trial_id"]]["assignment"] != a:
            raise ValueError("Replay prefix source assignment differs from the original source")
    return selected


def analyze(run, audit_dir, out, config=None, posthoc=False):
    manifest = read(run / "manifest.json")
    plan, origin = select_plan(manifest, read(config) if config is not None else None, posthoc)
    audit = read(audit_dir / "analysis.json")
    if not (audit.get("complete") is True and audit.get("integrity_passed") is True):
        raise ValueError("Replay collection completeness and integrity must both pass")
    if audit["manifest_sha256"] != digest(manifest):
        raise ValueError("Replay audit belongs to a different parent manifest")
    compact = read(audit_dir / "compact_outcomes.json")
    if digest(compact) != audit.get("compact_outcomes_sha256"):
        raise ValueError("Compact outcomes are not bound to the passing replay audit")
    outcome_metadata = admit_outcomes(manifest, audit, compact)
    assignments = manifest["diagnostics"]
    if digest(assignments) != manifest.get("assignment_sha256"):
        raise ValueError("Replay assignment hash mismatch")
    expected = {a["trial_id"]: a for a in assignments}
    if (
        len(expected) != len(assignments)
        or len(compact) != len(assignments)
        or len(compact) != audit["assigned_measurements"]
        or {r["trial_id"] for r in compact} != set(expected)
    ):
        raise ValueError("Compact measurement coverage differs from the frozen assignment set")
    for row in compact:
        a = expected[row["trial_id"]]
        fields = dict(
            prefix_id=a["prefix_id"],
            arm=a["arm"],
            direction=a["direction_name"],
            layer=a["layer"],
            position=a.get("capture_position"),
            dose_multiplier=a["dose_multiplier"],
        )
        if any(row[key] != value for key, value in fields.items()):
            raise ValueError("Compact selector metadata differs from the frozen assignment")
    source_assignments = assigned_sources(manifest)
    result = analyze_contrasts(compact, source_assignments, plan)
    for endpoint in result["results"]:
        outcome = endpoint["outcome"]
        binary = outcome != "margin"
        units = (
            "probability difference; denominator includes invalid responses"
            if binary
            else outcome_metadata["outcome_units"]
        )
        measurement = (
            "fixed_prefix_completed_answer_indicator"
            if binary and outcome_metadata["assay"] == "replay_information"
            else outcome_metadata["outcome_measurement"]
        )
        endpoint.update(
            assay=outcome_metadata["assay"],
            outcome_measurement=measurement,
            outcome_units=units,
            outcome_type=f"{outcome}_indicator"
            if binary
            else outcome_metadata["outcome_types_by_kind"][endpoint["kind"]],
            margin_type=outcome_metadata["outcome_types_by_kind"][endpoint["kind"]],
            legacy_descriptive_margin=outcome_metadata["assay"] == "natural_information" and not binary,
        )
        endpoint["estimate"]["units"] = units
    result.update(
        **outcome_metadata,
        manifest_sha256=digest(manifest),
        outcome_audit_sha256=digest(audit),
        compact_outcomes_sha256=digest(compact),
        inference_scope=manifest["inference_scope"],
        plan_origin=origin,
        posthoc=posthoc,
        direction_analysis_sha256=manifest["direction_analysis_sha256"],
        direction_input_sha256=manifest["direction_input"]["sha256"],
        source_eligibility=dict(
            assigned=manifest["assigned_sources"],
            eligible=len(manifest["prefixes"]),
            excluded=len(manifest["excluded_sources"]),
            excluded_source_ids=manifest["excluded_sources"],
        ),
    )
    result["outcome_units_by_selector"] = {row["outcome"]: row["outcome_units"] for row in result["results"]}
    result["outcome_measurements_by_selector"] = {
        row["outcome"]: row["outcome_measurement"] for row in result["results"]
    }
    result["outcome_types_by_selector_and_kind"] = {
        outcome: {row["kind"]: row["outcome_type"] for row in result["results"] if row["outcome"] == outcome}
        for outcome in result["outcome_units_by_selector"]
    }
    units = set(result["outcome_units_by_selector"].values())
    result["outcome_units"] = (
        next(iter(units)) if len(units) == 1 else "mixed outcome units; separate endpoints"
    )
    measurements = set(result["outcome_measurements_by_selector"].values())
    result["outcome_measurement"] = (
        next(iter(measurements)) if len(measurements) == 1 else "mixed measurements; separate endpoints"
    )
    result[
        "natural_audit_sha256"
        if outcome_metadata["assay"] == "natural_information"
        else "replay_audit_sha256"
    ] = digest(audit)
    result["interpretation"] += (
        " Natural endpoints use completed-answer scores or explicit valid-response indicators with assigned denominators; they must not be pooled with replay measurements."
        if outcome_metadata["assay"] == "natural_information"
        else " Replay endpoints distinguish fixed-prefix next-token logit margins from completed-answer validity/correctness indicators; their units and measurement stages remain separate."
    )
    out.mkdir(parents=True, exist_ok=True)
    write_once(out / "analysis.json", result)
    write_once(out / "contrast_plan.json", plan)
    print(
        {
            key: result[key]
            for key in (
                "manifest_sha256",
                "inference_scope",
                "plan_origin",
                "posthoc",
                "planned_endpoints",
                "source_eligibility",
            )
        }
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--posthoc", action="store_true")
    args = parser.parse_args()
    analyze(args.run, args.audit_dir, args.out, args.config, args.posthoc)
