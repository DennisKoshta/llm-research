"""Audit a completed pilot and produce descriptive counts without pseudoreplicated tests."""

import argparse
from collections import Counter
import json
from pathlib import Path
from curiosity_study.artifacts import RunStore, read, digest


def summarize(root):
    manifest = read(root / "manifest.json")
    records = RunStore(root, manifest).completed()
    expected = {a["trial_id"] for a in manifest["assignments"] + manifest["diagnostics"]}
    if set(records) != expected:
        raise ValueError(
            f"Incomplete or unexpected records: missing {len(expected - set(records))}, extra {len(set(records) - expected)}"
        )
    before = read(root / "resume-evidence.json")
    for trial_id, sha in before["record_hashes"].items():
        if digest(records[trial_id]) != sha:
            raise ValueError(f"Pre-restart record changed: {trial_id}")
    physical = list((root / "attempts").glob("*/*.json"))
    if len(physical) != len(expected):
        raise ValueError("Unexpected additional attempt records; inspect before interpreting")
    failures = Counter(r.get("failure") for r in records.values() if r.get("failure"))
    table = []
    timings = []
    for family in ("machines", "objects"):
        assigned = [a for a in manifest["assignments"] if a["family"] == family]
        behavioral = [records[a["trial_id"]] for a in assigned]
        diagnostic = [records[a["trial_id"]] for a in manifest["diagnostics"] if a["family"] == family]
        row = {
            "family": family,
            "behavioral_trials": len(assigned),
            "valid_first_choices": sum(
                r["first_service"] not in ("invalid", "technical_failure") for r in behavioral
            ),
            "diagnostics": {},
        }
        for kind in ("resolved_accuracy", "instrumental", "accounting"):
            subset = [r for r in diagnostic if r["kind"] == kind]
            row["diagnostics"][kind] = {
                "correct": sum(r.get("correct") is True for r in subset),
                "n": len(subset),
            }
        row["choices"] = {}
        for evidence in ("resolved", "unresolved"):
            for price in (0, 1, 2, 4):
                subset = [
                    r
                    for r in behavioral
                    if r["assignment"]["evidence"] == evidence and r["assignment"]["price"] == price
                ]
                row["choices"][f"{evidence}-p{price}"] = dict(Counter(r["first_service"] for r in subset))
        table.append(row)
        for record in behavioral:
            balance = 12
            for event in record["ledger"]:
                assert event["balance_before"] == balance
                balance -= event["deduction"]
                assert event["balance_after"] == balance and balance >= 0
                assert event["deduction"] == (
                    record["assignment"]["price"]
                    if event["service"] in ("target", "redundant", "novelty") and not event["failure"]
                    else 0
                )
            assert balance == record["final_balance"]
            for call in record["calls"]:
                assert call["hooks_remaining"] == 0
                timings.append(call["seconds"])
        for record in diagnostic:
            timings.append(record["inference"]["seconds"])
    audits = [read(path) for path in (root / "integrity").glob("*.json")]
    assert audits and all(audit["passed"] for audit in audits)
    return {
        "completed": len(records),
        "unique_assigned": len(expected),
        "pre_restart_records_unchanged": before["completed_before_resume"],
        "integrity_audits_passed": len(audits),
        "failures": dict(failures),
        "families": table,
        "model_calls": len(timings),
        "model_call_seconds": sum(timings),
        "mean_model_call_seconds": sum(timings) / len(timings),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
