"""Descriptive pair-cluster bootstrap; never substitutes for validation or confirmation."""

import argparse
from collections import defaultdict
from pathlib import Path
import random
from curiosity_study.artifacts import RunStore, read, write_once, digest


def quantile(values, p):
    values = sorted(values)
    pos = (len(values) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] * (hi - pos) + values[hi] * (pos - lo) if hi != lo else values[lo]


def analyze(root, repetitions=5000, seed=20261008):
    manifest = read(root / "manifest.json")
    assert manifest["assay"] == "one_credit" and manifest["phase"] == "behavior"
    records = RunStore(root, manifest).completed()
    assert set(records) == {a["trial_id"] for a in manifest["assignments"]}
    results = []
    for family in ("machines", "objects"):
        pairs = defaultdict(dict)
        for a in manifest["assignments"]:
            if a["family"] != family:
                continue
            key = (a["condition"], a["layout"])
            assert key not in pairs[a["index"]]
            pairs[a["index"]][key] = records[a["trial_id"]].get("first_service", "technical_failure")
        assert len(pairs) >= 2 and all(len(p) == 6 for p in pairs.values())
        indices = sorted(pairs)
        metrics = {}
        for condition in ("both_unresolved", "x_resolved", "y_resolved"):
            for choice in ("X", "Y", "stop", "invalid", "technical_failure"):
                values = [
                    sum(pairs[i][(condition, layout)] == choice for layout in ("XY", "YX")) / 2
                    for i in indices
                ]
                metrics[f"p/{condition}/{choice}"] = values
                if condition != "both_unresolved":
                    base = [
                        sum(pairs[i][("both_unresolved", layout)] == choice for layout in ("XY", "YX")) / 2
                        for i in indices
                    ]
                    metrics[f"difference/{condition}-both_unresolved/{choice}"] = [
                        x - y for x, y in zip(values, base)
                    ]
        rng = random.Random(f"{seed}:{family}")
        distributions = {name: [] for name in metrics}
        for _ in range(repetitions):
            sample = [rng.randrange(len(indices)) for _ in indices]
            for name, values in metrics.items():
                distributions[name].append(sum(values[i] for i in sample) / len(sample))
        results.append(
            {
                "family": family,
                "pair_count": len(indices),
                "metrics": {
                    name: {
                        "estimate": sum(v) / len(v),
                        "pointwise_percentile_95_interval": [
                            quantile(distributions[name], 0.025),
                            quantile(distributions[name], 0.975),
                        ],
                    }
                    for name, v in metrics.items()
                },
            }
        )
    return {
        "scope": "descriptive_development_only",
        "manifest_sha256": digest(manifest),
        "records_sha256": digest(records),
        "analysis_script_sha256": digest(Path(__file__).read_text()),
        "method": "resample complete pair clusters within family; preserve both layouts and all evidence conditions",
        "bootstrap_repetitions": repetitions,
        "seed": seed,
        "rows": results,
        "limitations": "Pointwise intervals, no multiplicity adjustment or population/template generalization. Percentile bootstrap intervals can collapse at sample boundaries; zero width does not establish zero population uncertainty. No noninferiority or H6 claim; margin and confirmatory sample are not registered.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    write_once(args.out, analyze(args.run))
