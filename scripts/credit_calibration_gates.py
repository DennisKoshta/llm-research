"""Explain the existing calibration gate, independently recomputed from complete records."""

import argparse
from collections import Counter
from pathlib import Path
from curiosity_study.artifacts import RunStore, read, write_once, digest
from curiosity_study.credit import calibrate


def explain(root):
    manifest = read(root / "manifest.json")
    assert manifest["phase"] == "behavior" and manifest["assay"] == "one_credit"
    records = RunStore(root, manifest).completed()
    assert set(records) == {a["trial_id"] for a in manifest["assignments"]}
    assert all(r["terminal"] for r in records.values())
    criteria = manifest["calibration_criteria"]
    rows, gates = {}, []
    for family in ("machines", "objects"):
        for condition in ("both_unresolved", "x_resolved", "y_resolved"):
            for layout in ("XY", "YX"):
                selected = [
                    r
                    for r in records.values()
                    if (r["assignment"]["family"], r["assignment"]["condition"], r["assignment"]["layout"])
                    == (family, condition, layout)
                ]
                counts = Counter(r.get("first_service", "technical_failure") for r in selected)
                n = len(selected)
                assert n
                valid = counts["X"] + counts["Y"] + counts["stop"]
                rows[(family, condition, layout)] = {"n": n, "counts": counts}
                gates.append(
                    {
                        "family": family,
                        "condition": condition,
                        "layout": layout,
                        "criterion": "minimum_sample",
                        "observed": n,
                        "threshold": criteria["minimum_items_per_family"],
                        "passed": n >= criteria["minimum_items_per_family"],
                    }
                )
                gates.append(
                    {
                        "family": family,
                        "condition": condition,
                        "layout": layout,
                        "criterion": "minimum_valid_format",
                        "valid": valid,
                        "assigned": n,
                        "observed": valid / n,
                        "threshold": criteria["minimum_valid_format"],
                        "passed": valid >= criteria["minimum_valid_format"] * n,
                    }
                )
        for choice in ("X", "Y"):
            rates = {}
            for layout in ("XY", "YX"):
                row = rows[(family, "both_unresolved", layout)]
                rates[layout] = row["counts"][choice] / row["n"]
                gates.append(
                    {
                        "family": family,
                        "condition": "both_unresolved",
                        "layout": layout,
                        "criterion": f"minimum_choice_probability_{choice}",
                        "observed": rates[layout],
                        "threshold": criteria["minimum_choice_probability_each_mystery"],
                        "passed": rates[layout] >= criteria["minimum_choice_probability_each_mystery"],
                    }
                )
            difference = abs(rates["XY"] - rates["YX"])
            gates.append(
                {
                    "family": family,
                    "condition": "both_unresolved",
                    "layout": "XY vs YX",
                    "criterion": f"maximum_layout_probability_difference_{choice}",
                    "observed": difference,
                    "threshold": criteria["maximum_layout_probability_difference"],
                    "passed": difference <= criteria["maximum_layout_probability_difference"] + 1e-12,
                }
            )
    overall = all(gate["passed"] for gate in gates)
    assert overall == calibrate(manifest, records)["passed"]
    return {
        "scope": "explanation_of_existing_frozen_criteria_no_new_gate",
        "manifest_sha256": digest(manifest),
        "records_sha256": digest(records),
        "passed": overall,
        "gates": gates,
        "failed_gates": [g for g in gates if not g["passed"]],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write_once(args.out, explain(args.run))
