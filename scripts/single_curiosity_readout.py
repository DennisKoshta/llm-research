"""Matched-item descriptive uncertainty and choice headroom, frozen before calibration."""

import argparse
from pathlib import Path
import numpy as np
from curiosity_study.artifacts import RunStore, read, write_once, digest


def summarize(root):
    manifest = read(root / "manifest.json")
    assert manifest["assay"] == "single_curiosity" and manifest["phase"] == "behavior"
    records = RunStore(root, manifest).completed()
    assert set(records) == {a["trial_id"] for a in manifest["assignments"]}
    categories = ("clue", "pattern", "stop", "invalid", "technical_failure")
    rng = np.random.default_rng(20261014)
    rows, contrasts, headroom = [], [], []
    for family in ("machines", "objects"):
        assignments = [a for a in manifest["assignments"] if a["family"] == family]
        indices = sorted({a["index"] for a in assignments})
        # Each row is one item; every resample preserves its two conditions and layouts.
        data = np.zeros((len(indices), 2, 2, len(categories)))
        seen = set()
        for a in assignments:
            key = (
                indices.index(a["index"]),
                ("unresolved", "resolved").index(a["condition"]),
                ("forward", "reverse").index(a["layout"]),
            )
            assert key not in seen
            seen.add(key)
            r = records[a["trial_id"]]
            assert r["terminal"]
            data[key + (categories.index(r["first_service"]),)] = 1
        assert len(seen) == len(indices) * 4
        mean = data.mean(axis=(0, 2))
        sampled = rng.integers(0, len(indices), size=(5000, len(indices)))
        bootstrap = data[sampled].mean(axis=(1, 3))
        ci = np.quantile(bootstrap, [0.025, 0.975], axis=0)
        for c, condition in enumerate(("unresolved", "resolved")):
            for k, category in enumerate(categories):
                rows.append(
                    dict(
                        family=family,
                        condition=condition,
                        outcome=category,
                        assigned=len(indices) * 2,
                        count=int(data[:, c, :, k].sum()),
                        probability=float(mean[c, k]),
                        ci95=[float(v) for v in ci[:, c, k]],
                    )
                )
            clue, pattern, stop, invalid, technical = mean[c]
            headroom.append(
                dict(
                    family=family,
                    condition=condition,
                    p_clue=float(clue),
                    downward_probability_headroom=float(clue),
                    upward_probability_headroom=float(1 - clue),
                    upward_with_invalid_and_technical_fixed=float(pattern + stop),
                    invalid_and_technical_probability=float(invalid + technical),
                )
            )
        contrast_samples = bootstrap[:, 0] - bootstrap[:, 1]
        contrast_ci = np.quantile(contrast_samples, [0.025, 0.975], axis=0)
        for k, category in enumerate(categories):
            contrasts.append(
                dict(
                    family=family,
                    outcome=category,
                    estimand="unresolved_minus_resolved",
                    difference=float(mean[0, k] - mean[1, k]),
                    ci95=[float(v) for v in contrast_ci[:, k]],
                )
            )
    return dict(
        scope="development_descriptive_not_confirmatory",
        bootstrap_unit="whole item with all four matched branches",
        replicates=5000,
        seed=20261014,
        interval="pointwise percentile 95%; not simultaneous",
        rows=rows,
        contrasts=contrasts,
        headroom=headroom,
        manifest_sha256=digest(manifest),
        records_sha256=digest(records),
        analysis_script_sha256=digest(Path(__file__).read_text()),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = summarize(args.run)
    write_once(args.out, result)
    for r in result["contrasts"]:
        if r["outcome"] == "clue":
            print(r)
