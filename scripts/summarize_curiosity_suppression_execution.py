"""All-assigned descriptive execution export for the fixed 15-arm C screen.

Reuses frozen scalar-summary helpers; does not independently certify geometry,
replay the model, select rows or create a scientific success gate.
"""

import argparse
from array import array
import csv
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np

from curiosity_study.artifacts import digest, read, write_once
from summarize_satisfaction_execution import (
    IDENTITY,
    METRICS,
    arm_summary,
    evidence_file,
    require,
    sha,
    summarize_row,
)
import summarize_satisfaction_execution as shared

ARMS = (
    "baseline",
    "zero_sham",
    "curiosity_negative",
    "confidence_positive",
    "completion_positive",
    "valence_positive",
    "valence_negative",
    "motivation_negative",
    "novelty_negative",
    "task_demand_negative",
    "random_01",
    "random_02",
    "random_03",
    "random_04",
    "random_05",
)
GROUPS = ("worker_0", "worker_1", "worker_2")
CHECKS = (
    "sham_identity_passed",
    "engagement_passed",
    "receipts_passed",
    "cross_arm_prepatch_identity_passed",
    "worker_policies_equal",
)


def summarize(run, audit_path, out):
    run, audit_path, out = (Path(p).resolve() for p in (run, audit_path, out))
    require(not out.exists(), "Execution summary exports are immutable")
    manifest, audit, frozen = read(run / "manifest.json"), read(audit_path), read(run / "freeze.json")
    require(
        manifest.get("assay") == "curiosity_suppression_screen"
        and "execution_group" not in manifest
        and audit.get("assay") == "curiosity_suppression_screen_audit"
        and audit.get("complete") is True
        and audit.get("integrity_passed") is True,
        "Require complete integrity-passed parent audit; readiness may fail",
    )
    require(
        audit["parent_manifest_sha256"] == digest(manifest) == frozen["manifest_sha256"]
        and audit["source_sha256"] == manifest["source_sha256"] == frozen["source_sha256"]
        and sha(run / "source.tar.gz") == frozen["archive_sha256"],
        "Audit or source freeze binding changed",
    )
    with tarfile.open(run / "source.tar.gz", "r:gz") as archive:
        for name, path in (
            ("summarize_satisfaction_execution.py", shared.__file__),
            (
                "audit_curiosity_suppression_screen.py",
                Path(__file__).with_name("audit_curiosity_suppression_screen.py"),
            ),
        ):
            require(
                hashlib.sha256(archive.extractfile("scripts/" + name).read()).hexdigest() == sha(path),
                "Frozen audit or shared descriptive helper bytes changed",
            )
    assignments = manifest["diagnostics"]
    require(
        manifest["assignments"] == []
        and len(assignments) == 4320
        and manifest["assignment_sha256"] == digest(assignments)
        and all(sum(a["arm"] == arm for a in assignments) == 288 for arm in ARMS),
        "Require all4320 assigned rows and every fixed arm",
    )
    outcomes = {r["trial_id"]: r for r in audit["outcomes"]}
    require(
        len(outcomes) == len(audit["outcomes"]) == 4320
        and set(outcomes) == {a["trial_id"] for a in assignments}
        and audit["outcomes_sha256"] == digest(audit["outcomes"]),
        "Audited outcome coverage changed",
    )
    descriptors = read(run / "children.json")
    audited_children = {c["execution_group"]: c for c in audit["children"]}
    require(
        len(descriptors) == len(audit["children"]) == 3
        and [c["execution_group"] for c in descriptors] == list(GROUPS)
        and set(audited_children) == set(GROUPS),
        "Require all three audited workers",
    )
    directions = manifest["screen_plan"]["direction_bindings"]
    banks = {
        arm: {phase: {metric: array("d") for metric in METRICS} for phase in ("prefill", "decode")}
        for arm in ARMS
    }
    rows, artifact_bindings = {}, []
    for descriptor in descriptors:
        child_root = Path(descriptor["path"]).resolve()
        require(child_root.parent == run.parent and child_root != run, "Child path escaped parent")
        child, group = read(child_root / "manifest.json"), descriptor["execution_group"]
        audited = audited_children[group]
        require(
            digest(child) == descriptor["manifest_sha256"] == audited["manifest_sha256"]
            and child["parent_manifest_sha256"] == digest(manifest)
            and child["diagnostics"] == [a for a in assignments if a["execution_group"] == group],
            "Child assignment or audit binding differs",
        )
        generation = audited["generation"]
        records = {r["trial_id"]: r for r in generation["record_bindings"]}
        require(len(records) == len(generation["record_bindings"]) == 1440, "Incomplete record bindings")
        require(len(generation["evidence"]) == 45, "Incomplete generation cohort bindings")
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
                "Bound evidence differs from assigned cohort",
            )
            artifact_bindings.append(dict(child=str(child_root), execution_group=group, **binding))
            for index, (a, measured) in enumerate(zip(cohort, evidence["condition"]["rows"], strict=True)):
                trial = a["trial_id"]
                observed, recorded = outcomes[trial], records[trial]
                require(
                    trial not in rows
                    and observed["assignment_sha256"] == digest(a)
                    and recorded["record_sha256"] == observed["record_sha256"]
                    and recorded["row"] == index
                    and recorded["evidence_sha256"] == binding["content_sha256"]
                    and all(observed[key] == a[key] for key in IDENTITY),
                    "Trial, assignment, evidence or row binding changed",
                )
                row = summarize_row(a, measured, observed, binding, recorded, path, banks)
                direction = directions.get(a["arm"])
                if a["arm"] not in ("baseline", "zero_sham"):
                    row["intervention_class"] = "nonzero_signed_direction"
                row.update(
                    execution_group=group,
                    source_direction=None if direction is None else direction["source"],
                    source_orientation=None if direction is None else direction["orientation"],
                    applied_unit_sha256=None if direction is None else direction["unit_sha256"],
                    valid=observed["valid"],
                    correct=observed["correct"],
                    selected_service=observed["selected_service"],
                )
                rows[trial] = row
    require(set(rows) == {a["trial_id"] for a in assignments}, "Assigned row lost from summary")
    ordered = [rows[a["trial_id"]] for a in assignments]
    arms = []
    for arm in ARMS:
        summary = arm_summary(arm, ordered, banks[arm])
        direction = directions.get(arm)
        summary.update(
            direction_binding=direction,
            valid_rows=sum(r["valid"] for r in ordered if r["arm"] == arm),
            workers=list(GROUPS),
        )
        arms.append(summary)
    out.mkdir(parents=True)
    csv_path = out / "row-execution.csv"
    with csv_path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ordered[0]))
        writer.writeheader()
        writer.writerows(ordered)
    summary = dict(
        schema_version=1,
        assay="curiosity_suppression_screen_execution_summary",
        descriptive_only=True,
        scientific_admission=False,
        assigned_rows=4320,
        arms=arms,
        audit_readiness=audit["readiness"],
        audited_checks={k: audit[k] for k in CHECKS},
        provenance=dict(
            run=str(run),
            parent_manifest_sha256=digest(manifest),
            source_sha256=manifest["source_sha256"],
            source_archive_sha256=frozen["archive_sha256"],
            audit_path=str(audit_path),
            audit_byte_sha256=sha(audit_path),
            audit_content_sha256=digest(audit),
            outcome_bindings_sha256=audit["outcomes_sha256"],
            exporter_byte_sha256=sha(__file__),
            shared_helper_byte_sha256=sha(shared.__file__),
            generation_evidence=artifact_bindings,
            generation_evidence_bindings_sha256=digest(artifact_bindings),
            row_csv=dict(path=csv_path.name, rows=4320, byte_sha256=sha(csv_path)),
            numpy_version=np.__version__,
        ),
        definitions=dict(
            geometry="Saved scalar ledgers validated by main audit; no independent decode residual reconstruction.",
            dose="Every nonzero arm uses the positive C raw norm times its signed applied unit, not its own contrast norm.",
            projection_increment="Positive along each signed applied unit; the primary is negative on original C. Different arms do not share a psychological scale.",
            baseline="No hook: zero injection counts and unavailable event distributions.",
            zero_sham="Real zero-dose hook events with zero perturbation geometry.",
            quantiles="Empirical linear5th/50th/95th percentiles, min/max and mean; no intervals.",
            weighting="Geometry weights events; lengths and cumulative exposure weight assigned rows equally.",
            cumulative="Sums of successive local events, not net displacement, persistent state or psychological intensity.",
            cap="Audited cap_hit; reaching4096 is also reported even when final token is EOS.",
        ),
        limitations=[
            "Response length is post-intervention; exposure cannot be treated as a pretreatment covariate.",
            "All assigned choices and diagnostics retained without service, correctness or readiness filtering.",
            "Uses shared frozen summary helpers and authoritative audit; this is not an independent execution certification.",
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
