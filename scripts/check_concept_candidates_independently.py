"""Independent arithmetic check of the first described-state candidate analysis.

This checker intentionally does not import the candidate analyzer or its fitting,
scoring, selection, or bootstrap helpers. It verifies the recorded primary raw
candidate results, not psychological validity or the exploratory PCA analyses.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
from pathlib import Path

import numpy as np

from curiosity_study.artifacts import digest, read, write_once


def byte_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(run, analysis_path, audit_path):
    manifest, analysis, audit = map(read, (run / "manifest.json", analysis_path, audit_path))
    assert analysis["manifest_sha256"] == audit["manifest_sha256"] == digest(manifest)
    assert analysis["audit_sha256"] == digest(audit) and audit["passed"] is True
    directions_path = analysis_path.parent / "directions.npz"
    assert analysis["directions_sha256"] == byte_sha(directions_path)
    records = {}
    for path in sorted((run / "cohorts").glob("*/*.json")):
        envelope = read(path)
        assert envelope["manifest_hash"] == digest(manifest)
        assert envelope["sha256"] == digest(envelope["records"])
        for row in envelope["records"]:
            assert row["trial_id"] not in records
            records[row["trial_id"]] = row
    assignments = manifest["diagnostics"]
    assert len(records) == len(assignments) == 384
    arrays, pairs, bindings = {}, defaultdict(dict), []
    for assignment in assignments:
        row = records[assignment["trial_id"]]
        assert row["assignment"] == assignment
        artifact = row["activation_artifact"]
        path = artifact["path"]
        if path not in arrays:
            assert byte_sha(run / path) == artifact["sha256"]
            with np.load(run / path, allow_pickle=False) as data:
                arrays[path] = {key: data[key] for key in data.files}
        data, offset = arrays[path], artifact["row"]
        assert data["trial_ids"][offset] == assignment["trial_id"]
        assert data["layers"].tolist() == [15, 31, 47, 57]
        assert data["positions"].tolist() == ["record", "final"]
        assert data["available"][offset].all() and np.isfinite(data["hidden"][offset]).all()
        key = (assignment["corpus_id"], assignment["contrast"], assignment["scenario_id"])
        assert assignment["polarity"] not in pairs[key]
        pairs[key][assignment["polarity"]] = data["hidden"][offset].astype(float)
        bindings.append(
            dict(
                trial_id=assignment["trial_id"],
                record_sha256=digest(row),
                artifact_sha256=artifact["sha256"],
                row=offset,
                token_prefix_sha256=str(data["token_prefix_sha256"][offset]),
            )
        )
    assert digest(bindings) == audit["record_bindings_sha256"] == analysis["capture_bindings_sha256"]
    scenarios = {corpus: sorted({key[2] for key in pairs if key[0] == corpus}) for corpus in ("A", "B")}
    assert all(len(names) == 12 for names in scenarios.values())
    assert all(set(members) == {"positive", "negative"} for members in pairs.values())
    compared, max_error = 0, 0.0

    def equal(actual, expected):
        nonlocal compared, max_error
        actual, expected = np.asarray(actual), np.asarray(expected)
        assert actual.shape == expected.shape
        error = float(np.max(np.abs(actual - expected)))
        compared += actual.size
        max_error = max(max_error, error)
        assert np.allclose(actual, expected, rtol=0, atol=1e-10)

    def values(corpus, contrast):
        return np.array(
            [
                [pairs[(corpus, contrast, scenario)][polarity] for polarity in ("positive", "negative")]
                for scenario in scenarios[corpus]
            ]
        )

    samples = np.random.default_rng(2026101431).integers(0, 12, (10000, 12))
    results = []
    with np.load(directions_path, allow_pickle=False) as directions:
        for candidate in analysis["candidates"]:
            contrast = candidate["contrast"]
            a = values("A", contrast)
            folds = []
            for li, layer in enumerate((15, 31, 47, 57)):
                differences = a[:, 0, li, 0] - a[:, 1, li, 0]
                gaps = []
                for heldout in range(12):
                    training = differences[np.arange(12) != heldout].mean(0)
                    gaps.append(float(differences[heldout] @ (training / np.linalg.norm(training))))
                gaps = np.asarray(gaps)
                concordance = float(np.mean((gaps > 0) + 0.5 * (gaps == 0)))
                folds.append(dict(layer=layer, concordance=concordance, mean_gap=float(gaps.mean())))
                recorded = candidate["a_leave_scenario_out"][li]
                equal(gaps, recorded["gaps"])
                equal(concordance, recorded["concordance"])
                equal(gaps.mean(), recorded["mean_gap"])
            li = max(range(4), key=lambda index: (folds[index]["concordance"], -folds[index]["layer"]))
            assert candidate["selected_layer"] == folds[li]["layer"] and candidate["position"] == "record"
            pair = a[:, :, li, 0]
            vector = (pair[:, 0] - pair[:, 1]).mean(0)
            norm, center = np.linalg.norm(vector), pair.mean((0, 1))
            unit = vector / norm
            threshold = float(center @ unit)
            source = directions["sources"].tolist().index(contrast)
            for field, value in (
                ("unit", unit),
                ("contrast", vector),
                ("center", center),
                ("contrast_norm", norm),
                ("threshold", threshold),
            ):
                equal(value, directions[field][source, li, 0])
            equal(norm, candidate["a_raw_contrast_norm"])
            equal(threshold, candidate["a_threshold"])
            target_gaps, metrics, specificity = None, [], []
            for recorded in [dict(contrast=contrast, **candidate["b_target"]), *candidate["b_controls"]]:
                tested = recorded["contrast"]
                scores = values("B", tested)[:, :, li, 0] @ unit
                gaps = scores[:, 0] - scores[:, 1]
                auc_diffs = scores[:, 0, None] - scores[None, :, 1]
                metric = dict(
                    contrast=tested,
                    mean_gap=float(gaps.mean()),
                    concordance=float(np.mean((gaps > 0) + 0.5 * (gaps == 0))),
                    auc=float(np.mean((auc_diffs > 0) + 0.5 * (auc_diffs == 0))),
                    accuracy=float(np.mean((scores > threshold) == np.array([True, False]))),
                )
                for name in ("mean_gap", "concordance", "auc", "accuracy"):
                    equal(metric[name], recorded[name])
                equal(gaps, recorded["gaps"])
                metrics.append(metric)
                if tested == contrast:
                    target_gaps = gaps
                else:
                    estimate = float(target_gaps.mean() - abs(gaps.mean()))
                    ci = np.quantile(
                        target_gaps[samples].mean(1) - abs(gaps[samples].mean(1)), [0.025, 0.975]
                    )
                    original = next(
                        item
                        for item in candidate["specificity_descriptive_intervals"]
                        if item["contrast"] == tested
                    )
                    equal(estimate, original["target_minus_absolute_control"])
                    equal(ci, original["ci95"])
                    specificity.append(
                        dict(contrast=tested, target_minus_absolute_control=estimate, ci95=ci.tolist())
                    )
            passed = (
                folds[li]["concordance"] >= 10 / 12
                and metrics[0]["concordance"] >= 10 / 12
                and metrics[0]["mean_gap"] > 0
                and all(row["target_minus_absolute_control"] > 0 for row in specificity)
            )
            assert passed is candidate["passed"]
            results.append(
                dict(
                    contrast=contrast,
                    selected_layer=folds[li]["layer"],
                    selected_position="record",
                    a_loso=folds,
                    a_raw_contrast_norm=float(norm),
                    a_threshold=threshold,
                    b_metrics=metrics,
                    specificity=specificity,
                    point_screen_passed=passed,
                )
            )
    return dict(
        schema_version=1,
        checked_at_utc=datetime.now(timezone.utc).isoformat(),
        passed=True,
        method="Independent NumPy arithmetic; no candidate-analyzer imports; selected raw estimators only",
        checker_byte_sha256=byte_sha(__file__),
        capture_manifest_sha256=digest(manifest),
        audit_sha256=digest(audit),
        analysis_sha256=digest(analysis),
        analysis_byte_sha256=byte_sha(analysis_path),
        directions_byte_sha256=byte_sha(directions_path),
        plan_byte_sha256=analysis["plan_byte_sha256"],
        record_bindings_sha256=digest(bindings),
        records=384,
        capture_files=len(arrays),
        numerical_values_compared=int(compared),
        maximum_absolute_error=max_error,
        candidates=results,
        caveats=[
            "All four A layers tie at 12/12 LOSO ordering; layer15 is the earliest tie, not uniquely superior.",
            "Curiosity overlaps generic motivation and novelty; novelty specificity interval includes zero.",
            "Satisfaction fixed-A threshold accuracy is 58.3% on B; no binary feeling classification.",
            "Narrated-state semantics and conditional descriptive intervals do not establish owned experience, intrinsic motivation, causal regulation, or confirmatory generalization.",
            "PCA and B-fitted diagnostic axes are outside this independent primary-raw check.",
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), "Independent verification receipts are immutable"
    result = check(args.run, args.analysis, args.audit)
    write_once(args.out, result)
    print(
        {
            key: result[key]
            for key in ("passed", "records", "numerical_values_compared", "maximum_absolute_error")
        }
    )
