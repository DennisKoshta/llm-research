"""Prospective revision-2 advancement rules, evaluated on every assigned diagnostic."""

from .artifacts import RunStore, read, digest
from .diagnostics import GATED_KINDS


def evaluate(manifest, records):
    if manifest.get("assay") == "single_curiosity":
        from .single_curiosity import evaluate as evaluate_single

        return evaluate_single(manifest, records)
    if manifest.get("assay") == "one_credit":
        from .credit import evaluate as evaluate_credit

        return evaluate_credit(manifest, records)
    criteria = manifest["validation_criteria"]
    rows = []
    for family in ("machines", "objects"):
        for kind in GATED_KINDS:
            assigned = [a for a in manifest["diagnostics"] if a["family"] == family and a["kind"] == kind]
            n = len(assigned)
            correct = sum(records.get(a["trial_id"], {}).get("correct") is True for a in assigned)
            valid = sum(records.get(a["trial_id"], {}).get("valid") is True for a in assigned)
            complete = all(records.get(a["trial_id"], {}).get("terminal") is True for a in assigned)
            passed = (
                n >= criteria["minimum_items_per_family_kind"]
                and complete
                and correct >= criteria["minimum_accuracy"] * n
                and valid >= criteria["minimum_valid_format"] * n
            )
            rows.append(
                {
                    "family": family,
                    "kind": kind,
                    "n": n,
                    "correct": correct,
                    "valid": valid,
                    "complete": complete,
                    "passed": passed,
                }
            )
    expected = {a["trial_id"] for a in manifest["diagnostics"]}
    return {
        "passed": all(row["passed"] for row in rows)
        and all(records.get(trial_id, {}).get("terminal") is True for trial_id in expected),
        "criteria": criteria,
        "rows": rows,
        "manifest_sha256": digest(manifest),
        "records_sha256": digest(records),
        "scope": "initial_comprehension_and_competence_only",
    }


def evaluate_run(root):
    manifest = read(root / "manifest.json")
    if manifest.get("phase") != "validation":
        raise ValueError("Expected a validation-phase manifest")
    return evaluate(manifest, RunStore(root, manifest).completed())
