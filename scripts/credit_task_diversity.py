"""Describe exact visible task diversity; not a semantic-generalization test or new gate."""

import argparse
from collections import Counter
from pathlib import Path
from curiosity_study.artifacts import RunStore, read, write_once, digest
from curiosity_study.credit import CreditTrial


def signatures(root):
    manifest = read(root / "manifest.json")
    records = RunStore(root, manifest).completed()
    assert set(records) == {a["trial_id"] for a in manifest["assignments"] + manifest["diagnostics"]}
    pairs = {}
    for record in records.values():
        assignment = record["assignment"]
        key = assignment["family"], assignment["index"]
        value = digest(record["tasks"])
        if key in pairs:
            assert pairs[key][0] == value
            continue
        # Canonicalize evidence condition/layout; omit action labels and hidden truth.
        trial = CreditTrial({**assignment, "condition": "both_unresolved", "layout": "XY"})
        signature = digest(
            {
                "evidence": trial.evidence(),
                "offered_queries": {side: task.target_query for side, task in trial.tasks.items()},
            }
        )
        pairs[key] = (value, signature)
    return manifest, pairs


def describe(root, prior):
    manifest, pairs = signatures(root)
    previous = [signatures(p)[1] for p in prior]
    rows = []
    for family in ("machines", "objects"):
        counts = Counter(value[1] for key, value in pairs.items() if key[0] == family)
        older = {value[1] for group in previous for key, value in group.items() if key[0] == family}
        rows.append(
            {
                "family": family,
                "assigned_pair_count": sum(counts.values()),
                "unique_visible_pair_signatures": len(counts),
                "largest_exact_signature_multiplicity": max(counts.values()),
                "signatures_also_in_compared_prior_samples": len(set(counts) & older),
            }
        )
    return {
        "scope": "exact_visible_pair_diversity_only_no_generalization_claim",
        "manifest_sha256": digest(manifest),
        "comparison_runs": [str(p) for p in prior],
        "rows": rows,
        "signature": "Both-unresolved evidence in XY order, preserving candidate labels/table order and offered queries; excludes hidden truth, item IDs, RNG seeds and action menu mappings.",
        "limits": "Distinct signatures can share the same structure or differ only in surface details. Both task families use narrow fixed generators and wording. Exact uniqueness does not establish semantic independence, held-out-template transfer, or real-world curiosity.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--prior", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write_once(args.out, describe(args.run, args.prior))
