"""A-only candidate selection and locked B evaluation of described-state contrasts."""

import argparse
from collections import defaultdict
import hashlib
from pathlib import Path

import numpy as np

from curiosity_study.artifacts import RunStore, digest, read, write_once
from curiosity_study.representation import auc

LAYERS, POSITIONS = [15, 31, 47, 57], ["record", "final"]
TARGETS = ["curiosity_vs_indifference", "satisfaction_vs_neutral"]
CONTROLS = [
    "confidence_high_vs_low",
    "positive_valence_vs_neutral",
    "completion_vs_ongoing",
    "generic_motivation_vs_indifference",
    "novelty_interest_vs_indifference",
    "task_demand_vs_no_demand",
]
CONTRASTS = TARGETS + CONTROLS
BOOTSTRAPS, SEED = 10000, 2026101431


def require(condition, message):
    if not condition:
        raise ValueError(message)


def paired_fit(pairs):
    """Pairs are scenario × member(positive,negative) × hidden dimension."""
    pairs = np.asarray(pairs, dtype=np.float64)
    require(pairs.ndim == 3 and pairs.shape[1] == 2 and np.isfinite(pairs).all(), "Invalid paired captures")
    vector = np.mean(pairs[:, 0] - pairs[:, 1], axis=0)
    center = np.mean(pairs, axis=(0, 1))
    norm = float(np.linalg.norm(vector))
    available = bool(np.isfinite(norm) and norm > 0)
    unit = vector / norm if available else np.zeros_like(vector)
    return dict(
        contrast=vector,
        center=center,
        unit=unit,
        norm=norm,
        threshold=float(center @ unit),
        available=available,
    )


def concordance(gaps):
    gaps = np.asarray(gaps, dtype=float)
    return float(np.mean((gaps > 0) + 0.5 * (gaps == 0)))


def leave_scenario_out(pairs):
    gaps = []
    for heldout in range(len(pairs)):
        fit = paired_fit(np.delete(pairs, heldout, axis=0))
        if not fit["available"]:
            return dict(available=False, concordance=None, mean_gap=None, gaps=[])
        gaps.append(float((pairs[heldout, 0] - pairs[heldout, 1]) @ fit["unit"]))
    return dict(available=True, concordance=concordance(gaps), mean_gap=float(np.mean(gaps)), gaps=gaps)


def select_layer(a_pairs):
    """Input scenario × member × layer × position × hidden. B is not an argument."""
    checks = [leave_scenario_out(a_pairs[:, :, li, 0]) for li in range(len(LAYERS))]
    available = [li for li, item in enumerate(checks) if item["available"]]
    if not available:
        return None, checks
    best = max(available, key=lambda li: (checks[li]["concordance"], -LAYERS[li]))
    return best, checks


def project_metrics(pairs, unit, threshold):
    values = pairs @ unit
    gaps = values[:, 0] - values[:, 1]
    labels = np.tile([1, 0], len(values))
    return dict(
        available=True,
        assigned_worlds=len(values),
        worlds=len(values),
        concordance=concordance(gaps),
        mean_gap=float(np.mean(gaps)),
        auc=auc(values.ravel(), labels),
        accuracy=float(np.mean((values.ravel() > threshold) == labels)),
        gaps=gaps.tolist(),
        descriptive_intervals=paired_intervals(gaps),
    )


def paired_intervals(gaps):
    """Conditional descriptive intervals; no refitting or candidate selection."""
    gaps = np.asarray(gaps, dtype=float)
    indices = np.random.default_rng(SEED).integers(0, len(gaps), (BOOTSTRAPS, len(gaps)))
    draws = gaps[indices]
    return dict(
        mean_gap_ci95=np.quantile(draws.mean(1), [0.025, 0.975]).tolist(),
        concordance_ci95=np.quantile(((draws > 0) + 0.5 * (draws == 0)).mean(1), [0.025, 0.975]).tolist(),
        role="Descriptive scenario resampling conditional on the fitted axis; no fitting/selection uncertainty or confirmatory coverage claim",
    )


def denoised_fit(target_pairs, control_pairs):
    """PCA on A nuisance members only, retaining the fixed 50% variance rule."""
    fit = paired_fit(target_pairs)
    values = np.asarray(control_pairs, dtype=np.float64).reshape(-1, target_pairs.shape[-1])
    centered = values - values.mean(axis=0)
    _, singular, basis = np.linalg.svd(centered, full_matrices=False)
    variance = singular**2
    total = float(variance.sum())
    if total <= 0 or not fit["available"]:
        return dict(available=False, unit=np.zeros_like(fit["unit"]), components=0, retained_norm_ratio=None)
    k = int(np.searchsorted(np.cumsum(variance), 0.5 * total, side="left") + 1)
    vector = fit["contrast"] - (fit["contrast"] @ basis[:k].T) @ basis[:k]
    norm = float(np.linalg.norm(vector))
    # Numerical zero is reported as unavailable, never amplified to a unit axis.
    available = bool(np.isfinite(norm) and norm > 1e-12 * fit["norm"])
    unit = vector / norm if available else np.zeros_like(vector)
    return dict(
        available=available,
        unit=unit,
        components=k,
        variance_fraction=float(variance[:k].sum() / total),
        retained_norm_ratio=norm / fit["norm"],
        threshold=float(fit["center"] @ unit),
    )


def specificity_intervals(target_gaps, controls):
    """Resample B scenarios once, preserving all seven paired contrasts together."""
    target_gaps, controls = np.asarray(target_gaps), np.asarray(controls)
    require(controls.shape == (len(CONTROLS), len(target_gaps)), "Misaligned control scenarios")
    indices = np.random.default_rng(SEED).integers(0, len(target_gaps), (BOOTSTRAPS, len(target_gaps)))
    target_means = target_gaps[indices].mean(1)
    result = []
    for name, gaps in zip(CONTROLS, controls):
        point = float(target_gaps.mean() - abs(gaps.mean()))
        draws = target_means - np.abs(gaps[indices].mean(1))
        result.append(
            dict(
                contrast=name,
                target_minus_absolute_control=point,
                ci95=np.quantile(draws, [0.025, 0.975]).tolist(),
                point_screen_passed=point > 0,
            )
        )
    return result


def build_pairs(assignments, hidden):
    require(len(assignments) == len(hidden) == 384, "Require all 384 texts before analysis")
    grouped = defaultdict(dict)
    for assignment, vector in zip(assignments, hidden):
        key = (assignment["corpus_id"], assignment["contrast"], assignment["scenario_id"])
        member = assignment["polarity"]
        require(
            member in ("positive", "negative") and member not in grouped[key],
            "Duplicate or unknown pair member",
        )
        grouped[key][member] = vector
    result, scenarios = {}, {}
    for corpus in ("A", "B"):
        names = sorted({key[2] for key in grouped if key[0] == corpus})
        require(len(names) == 12, "Each corpus needs twelve complete scenarios")
        scenarios[corpus] = names
        for contrast in CONTRASTS:
            values = []
            for scenario in names:
                members = grouped[(corpus, contrast, scenario)]
                require(set(members) == {"positive", "negative"}, "A scenario is missing a matched contrast")
                values.append([members["positive"], members["negative"]])
            result[(corpus, contrast)] = np.asarray(values, dtype=np.float64)
    require(len(grouped) == 192, "Unexpected corpus/contrast/scenario assignments")
    return result, scenarios


def analyze_arrays(assignments, hidden):
    pairs, scenarios = build_pairs(assignments, hidden)
    arrays = {name: [] for name in ("unit", "contrast", "center", "threshold", "contrast_norm")}
    all_metrics = []
    for contrast in CONTRASTS:
        local = {name: [] for name in arrays}
        for li, layer in enumerate(LAYERS):
            layer_arrays = {name: [] for name in arrays}
            for pi, position in enumerate(POSITIONS):
                fit = paired_fit(pairs[("A", contrast)][:, :, li, pi])
                for name in arrays:
                    layer_arrays[name].append(fit["norm" if name == "contrast_norm" else name])
                if fit["available"]:
                    for tested in CONTRASTS:
                        metric = project_metrics(
                            pairs[("B", tested)][:, :, li, pi], fit["unit"], fit["threshold"]
                        )
                        all_metrics.append(
                            dict(
                                source=contrast,
                                evaluated_contrast=tested,
                                layer=layer,
                                position=position,
                                **metric,
                            )
                        )
                else:
                    for tested in CONTRASTS:
                        all_metrics.append(
                            dict(
                                source=contrast,
                                evaluated_contrast=tested,
                                layer=layer,
                                position=position,
                                available=False,
                                assigned_worlds=12,
                                worlds=0,
                                concordance=None,
                                mean_gap=None,
                                auc=None,
                                accuracy=None,
                                gaps=[],
                                reason="The full-A paired mean contrast has zero norm",
                            )
                        )
            for name in arrays:
                local[name].append(layer_arrays[name])
        for name in arrays:
            arrays[name].append(local[name])
    arrays = {name: np.asarray(value) for name, value in arrays.items()}
    candidates = []
    for target in TARGETS:
        a = pairs[("A", target)]
        li, cv = select_layer(a)
        result = dict(
            contrast=target,
            position="record",
            a_leave_scenario_out=[dict(layer=layer, **value) for layer, value in zip(LAYERS, cv)],
            selected_layer=LAYERS[li] if li is not None else None,
        )
        for fold_result in result["a_leave_scenario_out"]:
            if fold_result["available"]:
                fold_result["descriptive_intervals"] = paired_intervals(fold_result["gaps"])
                fold_result["descriptive_intervals"]["role"] = (
                    "Fixed leave-scenario-out scores resampled descriptively; overlapping training folds and fitting uncertainty are not modeled"
                )
        if li is None:
            candidates.append(dict(**result, passed=False, reason="No available raw A axis"))
            continue
        fit = paired_fit(a[:, :, li, 0])
        b = pairs[("B", target)][:, :, li, 0]
        metrics = project_metrics(b, fit["unit"], fit["threshold"])
        control_metrics = [
            project_metrics(pairs[("B", c)][:, :, li, 0], fit["unit"], fit["threshold"]) for c in CONTROLS
        ]
        specificity = specificity_intervals(metrics["gaps"], [m["gaps"] for m in control_metrics])
        a_passed = cv[li]["concordance"] >= 10 / 12 and cv[li]["mean_gap"] > 0
        b_passed = metrics["concordance"] >= 10 / 12 and metrics["mean_gap"] > 0
        control_passed = all(row["point_screen_passed"] for row in specificity)
        # Eligibility is fully determined before fitting the independent B axis.
        passed = bool(a_passed and b_passed and control_passed)
        b_fit = paired_fit(b)
        a_controls = np.stack([pairs[("A", c)][:, :, li, 0] for c in CONTROLS], axis=1)
        sensitivity = denoised_fit(a[:, :, li, 0], a_controls)
        sensitivity_unit = sensitivity.pop("unit")
        if sensitivity["available"]:
            sensitivity["b_target"] = project_metrics(b, sensitivity_unit, sensitivity["threshold"])
            sensitivity["b_controls"] = [
                dict(
                    contrast=c,
                    **project_metrics(
                        pairs[("B", c)][:, :, li, 0], sensitivity_unit, sensitivity["threshold"]
                    ),
                )
                for c in CONTROLS
            ]
            fold_gaps = []
            for heldout in range(12):
                fold = denoised_fit(
                    np.delete(a[:, :, li, 0], heldout, axis=0), np.delete(a_controls, heldout, axis=0)
                )
                if not fold["available"]:
                    fold_gaps = []
                    break
                fold_gaps.append(float((a[heldout, 0, li, 0] - a[heldout, 1, li, 0]) @ fold["unit"]))
            sensitivity["a_leave_scenario_out"] = dict(
                available=bool(fold_gaps),
                concordance=concordance(fold_gaps) if fold_gaps else None,
                gaps=fold_gaps,
            )
        candidates.append(
            dict(
                **result,
                a_selection_passed=bool(a_passed),
                b_transfer_passed=bool(b_passed),
                b_specificity_point_screen_passed=bool(control_passed),
                passed=passed,
                b_target=metrics,
                b_controls=[dict(contrast=c, **m) for c, m in zip(CONTROLS, control_metrics)],
                specificity_descriptive_intervals=specificity,
                a_raw_contrast_norm=fit["norm"],
                a_threshold=fit["threshold"],
                pca50_sensitivity=sensitivity,
                independent_b_direction=dict(
                    available=b_fit["available"],
                    a_b_cosine=float(fit["unit"] @ b_fit["unit"]) if b_fit["available"] else None,
                    b_to_a=project_metrics(a[:, :, li, 0], b_fit["unit"], b_fit["threshold"])
                    if b_fit["available"]
                    else None,
                    role="Inspected development only; never replaces A or repairs eligibility",
                ),
            )
        )
    return dict(candidates=candidates, all_site_metrics=all_metrics, scenario_ids=scenarios), arrays


def load_captures(run, audit):
    manifest = read(run / "manifest.json")
    require(
        audit.get("passed") is True
        and audit.get("complete") is True
        and audit.get("manifest_sha256") == digest(manifest),
        "Require a complete passing capture audit",
    )
    require(
        audit.get("assigned") == 384
        and audit.get("model") == manifest.get("model")
        and audit.get("source_sha256") == manifest.get("source_sha256")
        and audit.get("capture_plan") == manifest.get("capture_plan"),
        "Capture audit metadata does not bind this collection",
    )
    rows = manifest["diagnostics"]
    require(
        len(rows) == 384 and digest(rows) == manifest["assignment_sha256"],
        "Capture assignment binding mismatch",
    )
    require(
        manifest.get("capture_plan") == dict(layers=LAYERS, positions=POSITIONS),
        "Capture sites differ from plan",
    )
    records = RunStore(run, manifest).completed()
    require(set(records) == {a["trial_id"] for a in rows}, "Complete record coverage is required")
    cache, hidden, bindings = {}, [], []
    for assignment in rows:
        record = records[assignment["trial_id"]]
        require(
            record["assignment"] == assignment and not record.get("failure"),
            "Capture record differs or failed",
        )
        meta = record["activation_artifact"]
        path = (run / meta["path"]).resolve()
        require(path.is_relative_to(run.resolve()), "Capture artifact escapes its collection")
        if path not in cache:
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path, allow_pickle=False) as data:
                cache[path] = sha, {key: data[key] for key in data.files}
        sha, data = cache[path]
        require(
            sha == meta["sha256"] and data["trial_ids"][meta["row"]] == assignment["trial_id"],
            "Capture artifact row/hash mismatch",
        )
        require(
            list(data["layers"]) == LAYERS
            and list(data["positions"]) == POSITIONS
            and data["available"][meta["row"]].all(),
            "Incomplete or changed capture sites",
        )
        vector = data["hidden"][meta["row"]]
        require(vector.shape[:2] == (4, 2) and np.isfinite(vector).all(), "Nonfinite or incomplete captures")
        hidden.append(vector)
        bindings.append(
            dict(
                trial_id=assignment["trial_id"],
                record_sha256=digest(record),
                artifact_sha256=sha,
                row=meta["row"],
                token_prefix_sha256=str(data["token_prefix_sha256"][meta["row"]]),
            )
        )
    require(
        digest(bindings) == audit.get("record_bindings_sha256"),
        "Capture records changed since their passing audit",
    )
    return manifest, np.stack(hidden), bindings


def main(run, audit_path, plan_path, out):
    require(not out.exists(), "Candidate analyses are immutable")
    audit = read(audit_path)
    manifest, hidden, bindings = load_captures(run, audit)
    plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    require(
        any(
            binding.get("path") == "reports/curiosity-satisfaction-candidate-pilot-plan.md"
            and binding.get("byte_sha256") == plan_sha
            for binding in manifest.get("document_bindings", [])
        ),
        "Analysis plan must match the method document frozen before capture",
    )
    result, arrays = analyze_arrays(manifest["diagnostics"], hidden)
    result.update(
        schema_version=1,
        assay="described_state_candidate_analysis",
        development_only=True,
        manifest_sha256=digest(manifest),
        model=manifest["model"],
        assigned=384,
        source_sha256=manifest["source_sha256"],
        capture_bindings_sha256=digest(bindings),
        audit_sha256=digest(audit),
        plan_byte_sha256=plan_sha,
        analyzer_byte_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        estimator="raw paired mean; fixed record; A-only grouped selection; locked whole-B gate",
        bootstrap=dict(
            draws=BOOTSTRAPS,
            seed=SEED,
            unit="B scenario, all contrasts retained together",
            role="descriptive only",
        ),
        interpretation="Candidate semantics only. No owned internal state, voluntary motivation, causal behavior, or subjective experience is established. B fits and PCA sensitivity cannot repair primary failures.",
        any_candidate_passed=any(row["passed"] for row in result["candidates"]),
    )
    out.mkdir(parents=True, exist_ok=False)
    with (out / "directions.npz").open("xb") as handle:
        np.savez_compressed(handle, sources=CONTRASTS, layers=LAYERS, positions=POSITIONS, **arrays)
    result["directions_sha256"] = hashlib.sha256((out / "directions.npz").read_bytes()).hexdigest()
    write_once(out / "analysis.json", result)
    print([{k: c[k] for k in ("contrast", "selected_layer", "passed")} for c in result["candidates"]])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.run, args.audit, args.plan, args.out)
