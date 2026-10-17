"""Describe bound sustained-screen traces after complete, integrity-passed audit.

Geometry was validated by the main audit. This exporter performs no model replay,
independent residual reconstruction, scientific admission or outcome filtering.
"""

import argparse
from array import array
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from curiosity_study.artifacts import digest, read, write_once

ARMS = ("baseline", "zero_sham", "satisfaction_plus_0_5", "satisfaction_plus_1_0")
GROUPS = {
    "baseline_zero": ARMS[:2],
    "positive_half": (ARMS[2],),
    "positive_full": (ARMS[3],),
}
METRICS = (
    "requested_delta_norm",
    "achieved_delta_norm",
    "relative_delta_norm",
    "projection_increment",
    "before_residual_norm",
    "after_residual_norm",
    "rounding_error_norm",
)
IDENTITY = (
    "trial_id",
    "base_trial_id",
    "arm",
    "panel_id",
    "semantic_world_id",
    "world_slot",
    "kind",
    "delivery",
    "menu_layout",
    "sampling_seed",
    "base_row",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    require(np.isfinite(values).all(), "Nonfinite descriptive measurement")
    if not len(values):
        return dict(count=0, minimum=None, p05=None, median=None, p95=None, maximum=None, mean=None)
    low, median, high = np.quantile(values, [0.05, 0.5, 0.95])
    return dict(
        count=len(values),
        minimum=float(values.min()),
        p05=float(low),
        median=float(median),
        p95=float(high),
        maximum=float(values.max()),
        mean=float(values.mean()),
    )


def evidence_file(root, binding):
    relative = Path(binding["path"])
    require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe generation-evidence path")
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), "Evidence escaped the child collection")
    payload = path.read_bytes()
    # Hash the exact bytes before parsing or inspecting any measurements.
    require(
        len(payload) == binding["byte_count"] and hashlib.sha256(payload).hexdigest() == binding["sha256"],
        "Generation-evidence bytes differ from the audit binding",
    )
    value = json.loads(payload)
    require(digest(value) == binding["content_sha256"], "Generation-evidence content binding differs")
    return path, value


def metrics(event):
    result = dict(
        requested_delta_norm=event["requested_delta_norm"],
        achieved_delta_norm=event["delta_norm"],
        relative_delta_norm=event["relative_delta_norm"],
        projection_increment=event["projection_after"] - event["projection_before"],
        before_residual_norm=event["before_norm"],
        after_residual_norm=event["after_norm"],
        rounding_error_norm=event["rounding_error_norm"],
    )
    require(
        all(type(v) in (int, float) and math.isfinite(v) for v in result.values()),
        "Missing or nonfinite geometry ledger",
    )
    return result


def summarize_row(assignment, measured, outcome, binding, record_binding, source_path, banks):
    trace, completion = measured["intervention_trace"], measured["completion"]
    events, tokens = trace["events"], completion["generated_token_ids"]
    intervention = assignment["intervention"]
    dose = None if intervention is None else intervention["dose"]
    require(
        outcome["status"] == "completed" and len(tokens) == outcome["generated_tokens"],
        "Complete audited row lacks matching observed token length",
    )
    require(
        trace["generated_token_count"] == len(tokens)
        and trace["total_patch_count"] == len(events)
        and len(events) == (0 if intervention is None else len(tokens)),
        "Trace coverage differs from the audited sustained assignment",
    )
    phases = Counter(event["phase"] for event in events)
    require(
        set(phases) <= {"prefill", "decode"}
        and phases["prefill"] == trace["prefill_patch_count"]
        and phases["decode"] == trace["decode_patch_count"],
        "Unknown or mismatched injection phase",
    )
    values = {name: [] for name in METRICS}
    for event in events:
        require(event["requested_scalar_delta"] == dose, "Requested dose differs from assignment")
        for name, value in metrics(event).items():
            values[name].append(value)
            banks[assignment["arm"]][event["phase"]][name].append(value)
    cumulative = trace["cumulative_achieved_delta_norm"]
    require(type(cumulative) in (int, float) and math.isfinite(cumulative), "Invalid cumulative trace norm")
    summed = math.fsum(values["achieved_delta_norm"])
    require(math.isclose(cumulative, summed, rel_tol=2e-5, abs_tol=1e-6), "Cumulative norm ledger mismatch")
    row = {key: assignment[key] for key in IDENTITY}
    row.update(
        status=outcome["status"],
        hooks_installed=intervention is not None,
        intervention_class="no_hook"
        if intervention is None
        else "true_zero_sham"
        if dose == 0
        else "positive_dose",
        requested_scalar_dose=dose,
        generated_tokens=len(tokens),
        ended_with_eos=completion["ended_with_eos"],
        cap_hit=outcome["cap_hit"],
        reached_generation_allowance=len(tokens) == 4096,
        injection_events=len(events),
        prefill_injection_events=phases["prefill"],
        decode_injection_events=phases["decode"],
        active_decode_forwards=trace["active_decode_forward_count"],
        cumulative_requested_delta_norm=math.fsum(values["requested_delta_norm"]),
        cumulative_achieved_delta_norm=cumulative,
        cumulative_achieved_norm_event_sum=summed,
        cumulative_norm_sum_discrepancy=abs(cumulative - summed),
        cumulative_projection_increment=math.fsum(values["projection_increment"]),
        record_sha256=outcome["record_sha256"],
        assignment_sha256=digest(assignment),
        evidence_path=str(source_path),
        evidence_byte_sha256=binding["sha256"],
        evidence_content_sha256=binding["content_sha256"],
        evidence_row=record_binding["row"],
    )
    for metric, samples in values.items():
        row.update({f"event_{metric}_{key}": value for key, value in distribution(samples).items()})
    return row


def arm_summary(arm, rows, bank):
    selected = [row for row in rows if row["arm"] == arm]
    require(len(selected) == 288, "Arm denominator differs from all288 assignments")
    event_stats = {}
    for phase in ("prefill", "decode", "all_injections"):
        event_stats[phase] = {
            name: distribution(
                np.concatenate([np.asarray(bank[p][name]) for p in ("prefill", "decode")])
                if phase == "all_injections"
                else bank[phase][name]
            )
            for name in METRICS
        }
    return dict(
        arm=arm,
        assigned_rows=len(selected),
        choice_rows=sum(r["kind"] == "choice" for r in selected),
        diagnostic_rows=sum(r["kind"] != "choice" for r in selected),
        intervention_class=selected[0]["intervention_class"],
        requested_scalar_dose=selected[0]["requested_scalar_dose"],
        total_generated_tokens=sum(r["generated_tokens"] for r in selected),
        generated_token_lengths=distribution([r["generated_tokens"] for r in selected]),
        generated_token_lengths_by_branch={
            label: distribution(
                [r["generated_tokens"] for r in selected if (r["kind"] == "choice") is is_choice]
            )
            for label, is_choice in (("choice", True), ("diagnostic", False))
        },
        ended_with_eos_rows=sum(r["ended_with_eos"] for r in selected),
        capped_rows=sum(r["cap_hit"] for r in selected),
        rows_reaching_generation_allowance=sum(r["reached_generation_allowance"] for r in selected),
        injection_event_counts={
            "total": sum(r["injection_events"] for r in selected),
            "prefill": sum(r["prefill_injection_events"] for r in selected),
            "decode": sum(r["decode_injection_events"] for r in selected),
        },
        injection_events_per_row=distribution([r["injection_events"] for r in selected]),
        event_weighted_geometry=event_stats,
        equal_row_weighted_cumulative_exposure={
            key: distribution([r[key] for r in selected])
            for key in (
                "cumulative_requested_delta_norm",
                "cumulative_achieved_delta_norm",
                "cumulative_projection_increment",
            )
        },
    )


def summarize(run, audit_path, out):
    run, audit_path, out = (Path(p).resolve() for p in (run, audit_path, out))
    require(not out.exists(), "Execution summary exports are immutable")
    manifest, audit = read(run / "manifest.json"), read(audit_path)
    require(
        manifest.get("assay") == "satisfaction_screen"
        and "execution_group" not in manifest
        and audit.get("assay") == "satisfaction_screen_audit"
        and audit.get("complete") is True
        and audit.get("integrity_passed") is True,
        "Require a complete integrity-passed parent audit; readiness may fail",
    )
    require(
        audit["parent_manifest_sha256"] == digest(manifest)
        and audit["source_sha256"] == manifest["source_sha256"],
        "Audit parent/source binding differs",
    )
    assignments = manifest["diagnostics"]
    require(
        manifest["assignments"] == []
        and len(assignments) == 1152
        and manifest["assignment_sha256"] == digest(assignments),
        "Require all1152 assigned rows",
    )
    outcomes = {r["trial_id"]: r for r in audit["outcomes"]}
    require(
        len(outcomes) == len(audit["outcomes"]) == 1152
        and set(outcomes) == {a["trial_id"] for a in assignments}
        and audit["outcomes_sha256"] == digest(audit["outcomes"]),
        "Audited outcome coverage differs",
    )
    descriptors = read(run / "children.json")
    audited_children = {c["execution_group"]: c for c in audit["children"]}
    require(
        len(descriptors) == len(audit["children"]) == 3
        and [c["execution_group"] for c in descriptors] == list(GROUPS)
        and set(audited_children) == set(GROUPS),
        "Require the exact three audited children",
    )
    banks = {
        arm: {phase: {metric: array("d") for metric in METRICS} for phase in ("prefill", "decode")}
        for arm in ARMS
    }
    rows, artifact_bindings = {}, []
    for descriptor in descriptors:
        child_root = Path(descriptor["path"]).resolve()
        require(
            child_root.parent == run.parent and child_root != run, "Child path left the parent run directory"
        )
        child = read(child_root / "manifest.json")
        group = descriptor["execution_group"]
        audited = audited_children[group]
        require(
            digest(child) == descriptor["manifest_sha256"] == audited["manifest_sha256"]
            and child["parent_manifest_sha256"] == digest(manifest)
            and child["diagnostics"] == [a for a in assignments if a["arm"] in GROUPS[group]],
            "Child assignment or audit binding differs",
        )
        generation = audited["generation"]
        record_bindings = {r["trial_id"]: r for r in generation["record_bindings"]}
        require(
            len(record_bindings) == len(generation["record_bindings"]) == len(child["diagnostics"]),
            "Missing or duplicate audited record binding",
        )
        require(
            len(generation["evidence"]) == len(child["diagnostics"]) // 32,
            "Missing or extra audited generation cohort",
        )
        for binding in generation["evidence"]:
            path, evidence = evidence_file(child_root, binding)
            offset = evidence["generation_offset"]
            require(type(offset) is int and offset >= 0 and offset % 32 == 0, "Invalid generation offset")
            cohort = child["diagnostics"][offset : offset + 32]
            require(
                len(cohort) == 32
                and evidence["manifest_sha256"] == digest(child)
                and evidence["trial_ids"] == [a["trial_id"] for a in cohort]
                and len(evidence["condition"]["rows"]) == 32,
                "Bound evidence differs from the assigned cohort",
            )
            artifact_bindings.append(dict(child=str(child_root), execution_group=group, **binding))
            for index, (a, measured) in enumerate(zip(cohort, evidence["condition"]["rows"], strict=True)):
                trial = a["trial_id"]
                observed, recorded = outcomes[trial], record_bindings[trial]
                require(
                    trial not in rows
                    and observed["assignment_sha256"] == digest(a)
                    and recorded["record_sha256"] == observed["record_sha256"]
                    and recorded["row"] == index
                    and recorded["evidence_sha256"] == binding["content_sha256"],
                    "Trial, row, record or evidence binding mismatch",
                )
                require(
                    all(observed[key] == a[key] for key in IDENTITY), "Outcome assignment identity differs"
                )
                rows[trial] = summarize_row(a, measured, observed, binding, recorded, path, banks)
    require(set(rows) == {a["trial_id"] for a in assignments}, "An assigned row was lost from the summary")
    ordered = [rows[a["trial_id"]] for a in assignments]
    arms = [arm_summary(arm, ordered, banks[arm]) for arm in ARMS]
    out.mkdir(parents=True)
    csv_path = out / "row-execution.csv"
    with csv_path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ordered[0]))
        writer.writeheader()
        writer.writerows(ordered)
    summary = dict(
        schema_version=1,
        assay="satisfaction_screen_execution_summary",
        descriptive_only=True,
        scientific_admission=False,
        assigned_rows=1152,
        arms=arms,
        audit_readiness=audit["readiness"],
        audited_checks={
            k: audit[k] for k in ("sham_identity_passed", "engagement_passed", "receipts_passed")
        },
        provenance=dict(
            run=str(run),
            parent_manifest_sha256=digest(manifest),
            source_sha256=manifest["source_sha256"],
            audit_path=str(audit_path),
            audit_byte_sha256=sha(audit_path),
            audit_content_sha256=digest(audit),
            outcome_bindings_sha256=audit["outcomes_sha256"],
            exporter_byte_sha256=sha(__file__),
            generation_evidence=artifact_bindings,
            generation_evidence_bindings_sha256=digest(artifact_bindings),
            row_csv=dict(path=csv_path.name, rows=len(ordered), byte_sha256=sha(csv_path)),
            numpy_version=np.__version__,
        ),
        definitions=dict(
            geometry="Saved scalar event ledgers already validated by the main audit; no independent decode replay.",
            baseline="No installed intervention: zero injection count/cumulative norm; event distributions unavailable.",
            zero_sham="Installed zero-dose intervention: real prefill/decode events with zero perturbation geometry.",
            dose_scale="S+0.5 and S+1.0 multiply the frozen full corpus-A satisfaction contrast norm; "
            "these are activation perturbation units, not state or score units.",
            quantiles="Empirical linear5th/50th/95th percentiles, minimum, maximum and arithmetic mean; no intervals.",
            event_weighting="Each applied event counts once; longer responses contribute more events.",
            cumulative="Sum of achieved per-injection norms, not net residual displacement or psychological intensity.",
            cumulative_projection_increment="Sum of local projection increments along successive treated "
            "activations; not the net change in the final projection or a single persistent state.",
            row_weighting="Each assigned row counts once in token-length and cumulative-exposure distributions.",
            cap="Audited cap_hit; reaching4096 tokens is separately reported even if the last token is EOS.",
        ),
        limitations=[
            "Cumulative exposure depends on response length, which may itself respond to steering. Do not equate "
            "exposure across lengths or adjust causal comparisons for it as a pretreatment covariate.",
            "All assigned choices and diagnostics are retained; no correctness, length, service or readiness filtering.",
            "Hash verification preserves audited evidence bindings; this summary does not replace the authoritative "
            "execution, geometry, source, sham or receipt audit and creates no new success gate.",
        ],
    )
    write_once(out / "arm-summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.run, args.audit, args.out)
    print(json.dumps(dict(assigned_rows=result["assigned_rows"], descriptive_only=True)))
