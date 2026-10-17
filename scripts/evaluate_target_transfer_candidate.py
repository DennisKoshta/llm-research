"""Evaluate the one prospective target-transfer candidate; never fit or choose a site."""

import argparse
from collections import defaultdict
import hashlib
import itertools
from pathlib import Path

import numpy as np

from curiosity_study import target_relevant_information as target
from curiosity_study.artifacts import digest, read, write_once
from curiosity_study.representation import CELL_ORDER, GAIN_WEIGHTS, auc
from curiosity_study.semantic_worlds import canonical_target_world, TARGET_SEMANTIC_RULE
from curiosity_study.transfer_representation import transfer_groups
from prepare_replay_interventions import split_provenance, world_id, world_sets

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "reports/target-relevance-causal-prospective-plan.md"
FROZEN_DIRECTIONS_SHA256 = "4f6541e8967e4a517be0b42b8b2d028e95eb7ba605e7388e3a2b241eca989b60"
CANDIDATE = dict(source="pooled:balanced", layer=47, position="record")
SEED, INDICES = 2026101430, 24
FAMILIES, POLARITIES = ("objects", "machines"), ("new", "repeat")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_binding(path, *, json_value=None):
    path = Path(path)
    result = dict(path=str(path.resolve()), byte_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if json_value is not None:
        result["canonical_sha256"] = digest(json_value)
    return result


def point_metrics(values, threshold):
    values = np.asarray(values, dtype=float).reshape(-1, 4)
    assigned = len(values)
    complete = values[np.isfinite(values).all(1)]
    return dict(
        assigned_worlds=assigned,
        worlds=len(complete),
        auc=auc(complete.ravel(), np.tile([0, 1, 1, 0], len(complete))) if len(complete) else None,
        accuracy=float(np.mean((complete.ravel() > threshold) == np.tile([0, 1, 1, 0], len(complete))))
        if len(complete)
        else None,
        mean_factorial_gap=float(np.mean(complete @ GAIN_WEIGHTS)) if len(complete) else None,
    )


def semantic_summary(assignments, scores, manifest, threshold):
    """Average duplicate observations in named semantic cells before any world statistic."""
    groups = transfer_groups(assignments)
    raw, pooled, identities = defaultdict(list), defaultdict(lambda: defaultdict(list)), {}
    canonical_cells = {}
    for key, indices in groups.items():
        family, seed, index, kind, template = key
        raw[(family, kind)].append(scores[indices])
        representative = assignments[int(indices[0])]
        identity = world_id(representative)
        if identity not in identities:
            tables = [target.task({**representative, "independent_report": first}) for first in (0, 1)]
            semantic = canonical_target_world(family, tables)
            identities[identity] = dict(
                world_id=identity,
                family=family,
                seed=seed,
                index=index,
                semantic_world_id=digest(semantic),
                block="calibration" if index < 12 else "check",
            )
            reports = sorted(name for name in semantic["attributes"] if name != semantic["designated_target"])
            # A repeated semantic world may display the two report columns in the
            # opposite order. Align by property name before averaging its cells.
            for first, query in CELL_ORDER:
                task = target.task({**representative, "independent_report": first, "query": query})
                canonical_cells[(identity, first, query)] = (
                    reports.index(task["attributes"][task["independent_report_attribute"]]),
                    reports.index(task["attributes"][task["queried_attribute"]]),
                )
        semantic_id = identities[identity]["semantic_world_id"]
        for row in indices:
            assignment = assignments[int(row)]
            cell = canonical_cells[(identity, assignment["independent_report"], assignment["query"])]
            pooled[(family, kind, semantic_id)][cell].append(float(scores[row]))
    declared_worlds, declared_semantics = world_sets(assignments, manifest)
    require(declared_worlds == set(identities), "Semantic ledger lost an assigned world")
    require(
        declared_semantics == {v["semantic_world_id"] for v in identities.values()},
        "Semantic ledger disagrees with the canonical world registry",
    )
    collapsed = defaultdict(list)
    for (family, kind, semantic), cells in sorted(pooled.items()):
        require(set(cells) == set(CELL_ORDER), "A semantic world is missing factorial cells")
        # Ordinary mean deliberately propagates missing observations. A duplicate
        # cannot fill another assigned row's missing capture.
        collapsed[(family, kind)].append([float(np.mean(cells[cell])) for cell in CELL_ORDER])
    ledger = sorted(identities.values(), key=lambda row: (row["family"], row["index"]))
    splits = []
    for family in FAMILIES:
        local = [row for row in ledger if row["family"] == family]
        calibration = {row["semantic_world_id"] for row in local if row["block"] == "calibration"}
        check = {row["semantic_world_id"] for row in local if row["block"] == "check"}
        splits.append(
            dict(
                family=family,
                assigned_indices=len(local),
                unique_semantic_worlds=len(calibration | check),
                calibration_assigned_indices=sum(row["block"] == "calibration" for row in local),
                check_assigned_indices=sum(row["block"] == "check" for row in local),
                calibration_semantic_worlds=len(calibration),
                check_semantic_worlds=len(check),
                calibration_semantic_world_ids=sorted(calibration),
                check_semantic_world_ids=sorted(check),
                overlap_semantic_world_ids=sorted(calibration & check),
                check_semantic_worlds_absent_from_calibration=len(check - calibration),
                novel_check_semantic_world_ids=sorted(check - calibration),
            )
        )
    summaries = []
    for family, kind in itertools.product(FAMILIES, POLARITIES):
        semantic_metrics = point_metrics(collapsed[(family, kind)], threshold)
        summaries.append(
            dict(
                family=family,
                kind=kind,
                raw_index_metrics=point_metrics(raw[(family, kind)], threshold),
                equal_semantic_world_metrics=semantic_metrics,
                auc_at_least_075=semantic_metrics["auc"] is not None and semantic_metrics["auc"] >= 0.75,
                signed_gap_positive=semantic_metrics["mean_factorial_gap"] is not None
                and semantic_metrics["mean_factorial_gap"] > 0,
            )
        )
    return dict(summaries=summaries, world_ledger=ledger, calibration_check_semantics=splits)


def validate_source(manifest, audit):
    rows = manifest["diagnostics"]
    expected = target.assignments(items=INDICES, seed=SEED, template="table")
    expected_by_id = {a["trial_id"]: a for a in expected}
    require(
        manifest.get("assignments") == [] and len(rows) == 960 and len({a["trial_id"] for a in rows}) == 960,
        "Require all 960 assigned target development branches",
    )
    require(manifest.get("assignment_sha256") == digest(rows), "Target source assignment binding mismatch")
    require(
        manifest.get("task_revision") == target.REVISION
        and manifest.get("phase") == "development_validation"
        and manifest.get("inference_scope") == "development",
        "Require the specified target development instrument",
    )
    require(
        all(
            a["trial_id"] in expected_by_id
            and all(a.get(key) == value for key, value in expected_by_id[a["trial_id"]].items())
            and a.get("max_new_tokens") == 4096
            for a in rows
        ),
        "Target source differs from the fixed seed, indices, kinds, cells, or generation regime",
    )
    criteria = manifest["validation_criteria"]
    require(
        criteria["minimum_distinct_items_per_cell"] == 24
        and criteria["minimum_accuracy"] == 0.9
        and criteria["minimum_valid_format"] == 0.95,
        "Target instrument validation criteria differ from the prospective plan",
    )
    require(
        audit.get("manifest_sha256") == digest(manifest)
        and audit.get("complete") is True
        and audit.get("assigned") == 960
        and audit.get("max_new_tokens") == 4096
        and audit.get("validation_criteria") == criteria,
        "Source audit is not complete and bound to this full instrument",
    )
    gates = audit.get("gates", {})
    required = set(itertools.product(FAMILIES, target.KINDS, (0, 1), (0, 1)))
    cells = gates.get("cells", [])
    keys = [(c["family"], c["kind"], c["independent_report"], c["query"]) for c in cells]
    require(
        len(keys) == 40 and set(keys) == required and gates.get("passed") is True,
        "Every original family/kind/cell validation gate must pass",
    )
    require(
        all(
            c.get("passed") is True
            and c["n"] == c["items"] == c["completed"] == 24
            and 0 <= c["correct"] <= c["valid"] <= 24
            and c["correct"] / 24 >= 0.9
            and c["valid"] / 24 >= 0.95
            for c in cells
        ),
        "A failed original instrument cell cannot be rescued by transfer",
    )
    oracle_audits = (
        [audit]
        if audit.get("target_relevance_oracle_audit_version") is not None
        else audit.get("partitions", [])
    )
    require(
        bool(oracle_audits)
        and all(a.get("target_relevance_oracle_audit_version") == 1 for a in oracle_audits),
        "Require the independent target-relevance oracle audit",
    )
    return rows


def evaluate(transfer_root, source_root, source_audit_path, directions_path, extraction_root, plan_path, out):
    transfer_root, source_root, directions_path, extraction_root, out = map(
        Path, (transfer_root, source_root, directions_path, extraction_root, out)
    )
    require(not out.exists(), "Candidate receipts are immutable; choose a new output path")
    manifest, audit = read(source_root / "manifest.json"), read(source_audit_path)
    rows = validate_source(manifest, audit)
    analysis_path, scores_path = transfer_root / "analysis.json", transfer_root / "scores.npz"
    analysis, source_analysis = read(analysis_path), read(directions_path.parent / "analysis.json")
    extraction_manifest = read(extraction_root / "manifest.json")
    direction_sha = hashlib.sha256(directions_path.read_bytes()).hexdigest()
    require(
        direction_sha
        == FROZEN_DIRECTIONS_SHA256
        == analysis.get("directions_sha256")
        == source_analysis.get("directions_sha256"),
        "The prospective original direction artifact cannot be replaced",
    )
    require(
        analysis.get("manifest_sha256") == digest(manifest)
        and analysis.get("source_audit_sha256") == digest(audit)
        and analysis.get("source_analysis_sha256") == digest(source_analysis),
        "Transfer analysis/source audit binding mismatch",
    )
    require(
        source_analysis["model"] == analysis["model"] == manifest["model"]
        and analysis["assigned"] == 960
        and analysis["task_revision"] == target.REVISION,
        "Transfer analysis does not cover the complete target source/model",
    )
    require(
        analysis.get("refitted_on_transfer") is False
        and analysis.get("descriptive_only") is True
        and analysis.get("transfer_estimand", {}).get("information_quantity")
        == "expected_target_information_gain_bits",
        "Require original-sign target-relevance transfer without refitting",
    )
    require(
        source_analysis["manifest_sha256"]
        == digest(extraction_manifest)
        == analysis["direction_extraction_manifest_sha256"],
        "Original extraction manifest binding mismatch",
    )
    provenance = split_provenance(
        dict(inference_scope="development", direction_extraction_collection=str(extraction_root)),
        manifest,
        source_analysis,
        rows,
    )
    require(
        analysis["split_provenance"] == provenance,
        "Transfer split provenance differs from the bound assignments",
    )
    with np.load(directions_path, allow_pickle=False) as z:
        fitted = {key: z[key] for key in z.files}
    with np.load(scores_path, allow_pickle=False) as z:
        scored = {key: z[key] for key in z.files}
    require(
        scored["trial_ids"].tolist() == [a["trial_id"] for a in rows],
        "Transfer score rows must match every source assignment in order",
    )
    require(
        scored["information_quantity"].item() == "expected_target_information_gain_bits"
        and scored["contrast_gain_labels_bits"].tolist() == [0, 1, 1, 0],
        "Transfer score labels must refer to target gain, not full-world gain",
    )
    for key in ("sources", "layers", "positions"):
        require(
            np.array_equal(scored[key], fitted[key]) and len(set(scored[key].tolist())) == len(scored[key]),
            "Transfer axes differ from the original frozen axes",
        )
    require(
        scored["scores"].shape
        == (960, len(scored["sources"]), len(scored["layers"]), len(scored["positions"])),
        "Transfer score shape mismatch",
    )
    si, li, pi = (
        scored[key].tolist().index(CANDIDATE[name])
        for key, name in (("sources", "source"), ("layers", "layer"), ("positions", "position"))
    )
    values = scored["scores"][:, si, li, pi]
    threshold = float(fitted["threshold"][si, li, pi])
    source_available = bool(
        np.isfinite(fitted["unit"][si, li, pi]).all()
        and np.linalg.norm(fitted["unit"][si, li, pi]) > 0
        and np.isfinite(threshold)
    )
    summary = semantic_summary(rows, values, manifest, threshold)
    metric_keys = [
        tuple(m[key] for key in ("source", "layer", "position", "target_family", "target_kind", "template"))
        for m in analysis["metrics"]
    ]
    expected_keys = set(
        itertools.product(
            scored["sources"].tolist(),
            scored["layers"].tolist(),
            scored["positions"].tolist(),
            FAMILIES,
            target.KINDS,
            ("table",),
        )
    )
    require(
        len(metric_keys) == len(expected_keys) and set(metric_keys) == expected_keys,
        "The descriptive transfer report must retain every original axis and readout",
    )
    by_metric = dict(zip(metric_keys, analysis["metrics"]))
    for item in summary["summaries"]:
        key = (
            CANDIDATE["source"],
            CANDIDATE["layer"],
            CANDIDATE["position"],
            item["family"],
            item["kind"],
            "table",
        )
        metric = by_metric[key]
        require(
            metric["source_axis_available"] == source_available, "Selected source-axis availability mismatch"
        )
        for name, value in item["raw_index_metrics"].items():
            saved = metric[name]
            require(
                saved is None
                if value is None
                else saved is not None and np.isclose(value, saved, rtol=1e-12, atol=1e-12),
                "Transfer score file disagrees with its original raw-index metrics",
            )
    checks = dict(
        original_frozen_axis_available=source_available,
        all_assigned_record_captures_present=bool(np.isfinite(values).all()),
        at_least_20_unique_semantic_worlds_per_family=all(
            s["unique_semantic_worlds"] >= 20 for s in summary["calibration_check_semantics"]
        ),
        signed_auc_and_positive_gap_in_each_family_and_polarity=all(
            s["auc_at_least_075"] and s["signed_gap_positive"] for s in summary["summaries"]
        ),
    )
    result = dict(
        schema_version=1,
        assay="target_transfer_candidate_eligibility",
        inference_scope="development",
        candidate=CANDIDATE,
        original_direction_sha256=direction_sha,
        information_quantity="expected_target_information_gain_bits",
        full_world_information_gain_bits=1,
        eligible_original_axis=all(checks.values()),
        checks=checks,
        assigned_record_captures=960,
        available_record_captures=int(np.isfinite(values).sum()),
        original_threshold=threshold if np.isfinite(threshold) else None,
        threshold_accuracy_is_admission_criterion=False,
        no_fitting_performed=True,
        other_axes_selected=False,
        other_axes_reported_in_bound_transfer_analysis=True,
        descriptive_metric_rows=len(metric_keys),
        descriptive_metrics_sha256=digest(analysis["metrics"]),
        semantic_rule=TARGET_SEMANTIC_RULE,
        semantic_weighting="Average all duplicate observations in each canonical named-report cell, then give each unique semantic world equal weight. Missing observations propagate; no correctness or validity selection.",
        score_rule="Signed AUC gives half credit to ties; mean factorial gap is gain-one minus gain-zero projection. Original axis sign and threshold are retained. New/repeat share the record prefix and are polarity checks, not independent representation replications.",
        development_check_semantic_readiness=all(
            s["check_semantic_worlds_absent_from_calibration"] >= 10
            for s in summary["calibration_check_semantics"]
        ),
        calibration_check_note="Both halves are development. No fallback direction or control direction is fitted here; any check must exclude calibration-overlapping semantic worlds.",
        cross_assay_semantic_limitation="Task-specific semantic hash nonoverlap does not establish that no individual candidate table or vocabulary appeared in a different assay.",
        bindings=dict(
            prospective_plan=file_binding(plan_path),
            source_manifest=file_binding(source_root / "manifest.json", json_value=manifest),
            source_audit=file_binding(source_audit_path, json_value=audit),
            transfer_analysis=file_binding(analysis_path, json_value=analysis),
            transfer_scores=file_binding(scores_path),
            original_directions=file_binding(directions_path),
            original_direction_analysis=file_binding(
                directions_path.parent / "analysis.json", json_value=source_analysis
            ),
            original_extraction_manifest=file_binding(
                extraction_root / "manifest.json", json_value=extraction_manifest
            ),
            evaluator=file_binding(__file__),
            evaluator_dependencies={
                "semantic_canonicalization": file_binding(canonical_target_world.__code__.co_filename),
                "semantic_registry": file_binding(world_sets.__code__.co_filename),
                "auc": file_binding(auc.__code__.co_filename),
            },
            task_module_sha256=manifest["task_module_sha256"],
        ),
        **summary,
    )
    write_once(out, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transfer", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--directions", type=Path, required=True)
    parser.add_argument("--extraction", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(
        args.transfer, args.source, args.source_audit, args.directions, args.extraction, args.plan, args.out
    )
    print(
        {
            key: result[key]
            for key in ("eligible_original_axis", "checks", "development_check_semantic_readiness")
        }
    )
