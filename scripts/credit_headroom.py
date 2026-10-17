"""Descriptive limits on unresolved information-choice effects; not a new gate."""

import argparse
from collections import Counter
from pathlib import Path
from curiosity_study.artifacts import RunStore, read, write_once, digest


def summarize(manifest, records):
    assert manifest["assay"] == "one_credit" and manifest["phase"] == "behavior"
    assignments = {a["trial_id"]: a for a in manifest["assignments"]}
    assert set(records) == set(assignments)
    assert all(records[tid]["assignment"] == a for tid, a in assignments.items())
    rows = []
    for family in ("machines", "objects"):
        for layout in ("XY", "YX", "pooled"):
            selected = [
                records[tid]
                for tid, a in assignments.items()
                if a["family"] == family
                and a["condition"] == "both_unresolved"
                and (layout == "pooled" or a["layout"] == layout)
            ]
            n = len(selected)
            assert n > 0
            counts = Counter(r.get("first_service", "technical_failure") for r in selected)
            assert set(counts) <= {"X", "Y", "stop", "invalid", "technical_failure"}
            information = (counts["X"] + counts["Y"]) / n
            stop = counts["stop"] / n
            rows.append(
                {
                    "family": family,
                    "layout": layout,
                    "assigned": n,
                    "counts": {
                        key: counts[key] for key in ("X", "Y", "stop", "invalid", "technical_failure")
                    },
                    "p_information": information,
                    "p_stop": stop,
                    "maximum_upward_change_without_constraints": 1 - information,
                    "maximum_upward_change_if_invalid_and_failure_rates_stay_fixed": stop,
                    "maximum_downward_change": information,
                }
            )
    return {
        "scope": "descriptive_sensitivity_only_not_an_advancement_gate",
        "manifest_sha256": digest(manifest),
        "definition": "In both-unresolved trials either X or Y is informative; probabilities include every assigned trial.",
        "rows": rows,
        "limits": "These are algebraic limits from observed sample proportions, not confidence bounds or power estimates. A ceiling/floor can make an intervention direction uninformative without establishing absence of the proposed mechanism. This assay lacks a separate non-information optional activity control and does not establish motivational specificity.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = read(args.run / "manifest.json")
    write_once(args.out, summarize(manifest, RunStore(args.run, manifest).completed()))
