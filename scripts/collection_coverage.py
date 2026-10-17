"""Read-only operational coverage for fixed, atomically published cohorts.

This gate inspects assignment and storage bindings, never correctness, validity,
choices, confidence, or experimental effects. It deliberately rejects legacy
individual-record storage: such files cannot establish atomic cohort coverage.
"""

import argparse
import json
from pathlib import Path

from curiosity_study.artifacts import digest, read


class CollectionCoverageError(ValueError):
    """A frozen collection is incomplete or its physical coverage is invalid."""


def require(condition, message):
    if not condition:
        raise CollectionCoverageError(message)


def load(path):
    try:
        return read(path)
    except (OSError, ValueError) as error:
        raise CollectionCoverageError(f"Cannot read operational artifact: {path}") from error


def check_collection_coverage(root, *, expected_manifest_sha256):
    """Return an operational receipt, or raise CollectionCoverageError.

    Call once per child after transport verification and before advancing to the
    next wave. Supply the child hash from its frozen parent descriptor (or the
    frozen manifest hash for an unpartitioned run). This does not check cloud
    shutdown, startup integrity, activation binaries, or scientific readiness.
    No files are created or modified, including when the run is incomplete.
    """
    root = Path(root)
    manifest = load(root / "manifest.json")
    require(isinstance(manifest, dict), "Manifest must be an object")
    manifest_sha = digest(manifest)
    require(
        isinstance(expected_manifest_sha256, str) and manifest_sha == expected_manifest_sha256,
        "Manifest differs from the frozen expected binding",
    )
    execution = manifest.get("execution")
    require(
        isinstance(execution, dict) and execution.get("backend") == "torch_batched_v1",
        "Operational atomic-cohort coverage requires a fixed batched execution plan",
    )
    size = execution.get("batch_size")
    require(type(size) is int and size > 0, "Batch size must be a positive integer")
    ordinary, diagnostics = manifest.get("assignments", []), manifest.get("diagnostics", [])
    require(isinstance(ordinary, list) and isinstance(diagnostics, list), "Assignment fields must be lists")
    assignments = ordinary + diagnostics
    require(bool(assignments), "An empty collection cannot pass operational coverage")
    require(digest(assignments) == manifest.get("assignment_sha256"), "Manifest assignment digest mismatch")
    require(
        all(
            isinstance(a, dict) and isinstance(a.get("trial_id"), str) and a["trial_id"] for a in assignments
        ),
        "Every assignment requires a nonempty trial ID",
    )
    ids = [a["trial_id"] for a in assignments]
    require(len(set(ids)) == len(ids), "Duplicate trial IDs in the frozen assignment list")
    require(
        not any((root / "attempts").rglob("*.json")),
        "Individual attempt records cannot establish atomic cohort coverage",
    )
    expected = {offset: assignments[offset : offset + size] for offset in range(0, len(assignments), size)}
    seen_offsets, seen_trials, artifacts = set(), set(), []
    for path in sorted((root / "cohorts").rglob("*.json")):
        relative = path.relative_to(root / "cohorts")
        require(len(relative.parts) == 2, "A cohort must be stored under exactly one attempt directory")
        require(path.stem.isascii() and path.stem.isdecimal(), "Cohort filename must encode its row offset")
        offset = int(path.stem)
        require(offset in expected, "Unexpected cohort offset outside the frozen assignment boundaries")
        require(offset not in seen_offsets, "Duplicate physical cohort, including identical replays")
        envelope = load(path)
        require(isinstance(envelope, dict), "Cohort envelope must be an object")
        require(envelope.get("manifest_hash") == manifest_sha, "Cohort belongs to another manifest")
        records = envelope.get("records")
        require(isinstance(records, list) and records, "Cohort must contain a nonempty record list")
        require(envelope.get("sha256") == digest(records), "Cohort record digest mismatch")
        require(all(isinstance(r, dict) for r in records), "Every cohort record must be an object")
        cohort_ids = [a["trial_id"] for a in expected[offset]]
        require(
            [r.get("trial_id") for r in records] == cohort_ids,
            "Atomic cohort membership or order differs from the frozen manifest",
        )
        batch_ids = []
        for row, (record, assignment) in enumerate(zip(records, expected[offset])):
            trial_id = assignment["trial_id"]
            require(trial_id not in seen_trials, "A trial appears in more than one physical record")
            require(record.get("terminal") is True, "A cohort contains a nonterminal trial")
            require(
                digest(record.get("assignment")) == digest(assignment),
                "Record assignment differs from the frozen assignment",
            )
            require(
                record.get("failure") != "execution_error",
                "An execution-error trial cannot pass operational coverage",
            )
            if manifest.get("assay") == "curiosity_suppression_screen":
                require(
                    record.get("status") != "execution_error"
                    and record.get("failure") != "generation_execution_error",
                    "A curiosity generation execution-error trial cannot pass operational coverage",
                )
            # Thinking records also expose actual inference row order. Natural
            # and replay cohorts use the same atomic storage without this field.
            inference = record.get("inference")
            if isinstance(inference, dict) and "batch" in inference:
                batch = inference["batch"]
                require(isinstance(batch, dict), "Inference batch metadata must be an object")
                require(
                    batch.get("cohort_ids") == cohort_ids
                    and type(batch.get("row")) is int
                    and batch["row"] == row
                    and type(batch.get("size")) is int
                    and batch.get("size") == len(cohort_ids)
                    and batch.get("backend") == execution["backend"],
                    "Inference cohort membership or row order differs from the frozen manifest",
                )
                require(isinstance(batch.get("id"), str) and batch["id"], "Inference batch ID is missing")
                batch_ids.append(batch["id"])
            seen_trials.add(trial_id)
        require(
            not batch_ids or (len(batch_ids) == len(records) and len(set(batch_ids)) == 1),
            "Inference batch binding must cover every row with one shared batch ID",
        )
        seen_offsets.add(offset)
        artifacts.append(dict(path=str(path.relative_to(root)), envelope_sha256=digest(envelope)))
    require(
        seen_offsets == set(expected) and seen_trials == set(ids),
        f"Incomplete collection: {len(seen_trials)}/{len(ids)} terminal trials and {len(seen_offsets)}/{len(expected)} atomic cohorts",
    )
    return dict(
        schema_version=1,
        passed=True,
        manifest_sha256=manifest_sha,
        assignment_sha256=manifest["assignment_sha256"],
        assigned_trials=len(ids),
        terminal_trials=len(seen_trials),
        batch_size=size,
        expected_cohorts=len(expected),
        verified_cohorts=len(seen_offsets),
        coverage_sha256=digest(dict(manifest_sha256=manifest_sha, cohorts=artifacts)),
        scientific_outcomes_checked=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(check_collection_coverage(args.run, expected_manifest_sha256=args.manifest_sha256)))
