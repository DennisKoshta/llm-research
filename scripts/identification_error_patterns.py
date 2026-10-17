"""Explicitly post-hoc error descriptions; do not alter scores or selection."""

from collections import Counter
from pathlib import Path
from curiosity_study.artifacts import read, RunStore, write_once

root = Path("runs/modal-identification-v1")
m = read(root / "manifest.json")
rows = list(RunStore(root, m).completed().values())
invalid = [r for r in rows if not r["valid"] and r["assignment"]["mode"] != "think_sampled"]
groups = []
for family in ("machines", "objects"):
    for mode in ("no_think_greedy", "no_think_sampled"):
        errors = [
            r
            for r in rows
            if r["assignment"]["context"] == "paired"
            and r["assignment"]["labels"] == "ABCD"
            and r["assignment"]["family"] == family
            and r["assignment"]["mode"] == mode
            and r["valid"]
            and not r["correct"]
        ]
        matches = eligible = 0
        position = Counter()
        for r in errors:
            a = r["assignment"]
            other = "Y" if a["side"] == "X" else "X"
            other_truth = dict(zip("JKLM", "ABCD"))[r["tasks"][other]["truth"]]
            matches += r["answer"] == other_truth
            eligible += other_truth != r["expected"]
            position["queried_first" if a["layout"][0] == a["side"] else "queried_second"] += 1
        groups.append(
            {
                "family": family,
                "mode": mode,
                "valid_wrong": len(errors),
                "other_truth_differs_from_target": eligible,
                "wrong_answer_equals_other_truth": matches,
                "error_position_counts": dict(position),
            }
        )
write_once(
    "reports/identification-v1-error-patterns.json",
    {
        "scope": "post_hoc_descriptive_only_no_rescoring_no_selection_change",
        "nonthinking_invalid_count": len(invalid),
        "nonthinking_invalid_prefix_counts": dict(
            Counter(
                r["inference"]["raw"].strip().split()[0] if r["inference"]["raw"].strip() else "<empty>"
                for r in invalid
            )
        ),
        "paired_ABCD_errors": groups,
        "interpretation_limit": "The distractor remains unresolved; its simulator truth is not available to the model. Matching that truth is not evidence that the model identified or attended to it. Position counts are error counts, not accuracy rates with balanced denominators.",
    },
)
print(groups)
