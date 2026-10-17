"""Freeze the one allowed target candidate and calibration-only factor controls."""

import argparse
from collections import defaultdict
import hashlib
from pathlib import Path
import tarfile
import tempfile

import numpy as np

from curiosity_study import target_relevant_information as target
from curiosity_study.artifacts import digest, read, write_once
from curiosity_study.representation import CELL_ORDER, GAIN_WEIGHTS, KNOWN_WEIGHTS, QUERY_WEIGHTS
from curiosity_study.semantic_worlds import canonical_target_world
from analyze_thinking_representations import load_collection
import evaluate_target_transfer_candidate as eligibility

PLAN_MEMBER = "reports/target-relevance-causal-prospective-plan.md"
OUTPUT_SOURCE = "target:balanced"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def semantic_captures(assignments, vectors):
    """Keep the two halves and readouts separate; align duplicate cells by report name."""
    require(
        vectors.ndim == 2 and len(vectors) == len(assignments) and np.isfinite(vectors).all(),
        "Every assigned record capture must be present before any candidate fit",
    )
    worlds, cells = {}, defaultdict(lambda: defaultdict(list))
    for row, assignment in enumerate(assignments):
        if assignment["kind"] not in eligibility.POLARITIES:
            continue
        identity = (assignment["family"], assignment["seed"], assignment["index"])
        if identity not in worlds:
            tables = [target.task({**assignment, "independent_report": u}) for u in (0, 1)]
            canonical = canonical_target_world(assignment["family"], tables)
            names = sorted(name for name in canonical["attributes"] if name != canonical["designated_target"])
            worlds[identity] = (digest(canonical), names)
        semantic, names = worlds[identity]
        task = target.task(assignment)
        cell = (
            names.index(task["attributes"][task["independent_report_attribute"]]),
            names.index(task["attributes"][task["queried_attribute"]]),
        )
        block = "calibration" if assignment["index"] < 12 else "check"
        cells[(block, assignment["family"], semantic, assignment["kind"])][cell].append(vectors[row])
    result = {}
    for key, observations in cells.items():
        require(set(observations) == set(CELL_ORDER), "Candidate fitting requires complete semantic worlds")
        result[key] = np.stack([np.mean(observations[cell], axis=0) for cell in CELL_ORDER])
    return result


def calibration_components(captures):
    """One world contrast after averaging its redundant new/repeat record contexts."""
    weights = dict(
        information=GAIN_WEIGHTS, target_u=KNOWN_WEIGHTS, target_q=QUERY_WEIGHTS, center=np.full(4, 0.25)
    )
    family_components, counts = [], {}
    for family in eligibility.FAMILIES:
        semantics = sorted({key[2] for key in captures if key[:2] == ("calibration", family)})
        require(bool(semantics), "Both families require calibration semantic worlds")
        balanced = []
        for semantic in semantics:
            selected = [("calibration", family, semantic, kind) for kind in eligibility.POLARITIES]
            require(all(key in captures for key in selected), "Both calibration polarities must be retained")
            balanced.append(np.mean([captures[key] for key in selected], axis=0))
        cells = np.stack(balanced)
        family_components.append(
            {name: np.mean(np.einsum("c,wcd->wd", weight, cells), axis=0) for name, weight in weights.items()}
        )
        counts[family] = len(semantics)
    return {name: np.mean([row[name] for row in family_components], axis=0) for name in weights}, counts


def check_candidate(captures, unit, threshold):
    summaries, counts, overlaps = [], {}, {}
    for family in eligibility.FAMILIES:
        calibration = {key[2] for key in captures if key[:2] == ("calibration", family)}
        check = {key[2] for key in captures if key[:2] == ("check", family)}
        novel = sorted(check - calibration)
        counts[family], overlaps[family] = len(novel), sorted(check & calibration)
        for kind in eligibility.POLARITIES:
            values = (
                np.stack([captures[("check", family, semantic, kind)] @ unit for semantic in novel])
                if novel
                else np.empty((0, 4))
            )
            metric = eligibility.point_metrics(values, threshold)
            summaries.append(
                dict(
                    family=family,
                    kind=kind,
                    novel_semantic_world_ids=novel,
                    **metric,
                    passed=len(novel) >= 10
                    and metric["auc"] is not None
                    and metric["auc"] >= 0.75
                    and metric["mean_factorial_gap"] is not None
                    and metric["mean_factorial_gap"] > 0,
                )
            )
    return dict(
        passed=all(row["passed"] for row in summaries),
        summaries=summaries,
        novel_semantic_world_counts=counts,
        excluded_calibration_overlap=overlaps,
    )


def assignment_split(manifest):
    ranges = dict(calibration_indices=[0, 12], check_indices=[12, 24])
    require(
        manifest.get("development_split") == ranges, "Require the frozen twelve-plus-twelve development split"
    )
    rows = [
        dict(trial_id=a["trial_id"], representation_split="extraction" if a["index"] < 12 else "development")
        for a in manifest["diagnostics"]
    ]
    return dict(
        schema_version=1,
        rule="manifest_development_split_v1",
        manifest_sha256=digest(manifest),
        development_split_sha256=digest(ranges),
        assignments=rows,
        assignment_sha256=digest(rows),
    )


def archived_plan_binding(source_root, manifest, plan_path):
    freeze_path, archive_path = source_root / "freeze.json", source_root / "source.tar.gz"
    frozen = read(freeze_path)
    require(
        frozen["manifest_sha256"] == digest(manifest)
        and frozen["source_sha256"] == manifest["source_sha256"],
        "Target source freeze does not bind this manifest and implementation",
    )
    require(
        hashlib.sha256(archive_path.read_bytes()).hexdigest() == frozen["archive_sha256"],
        "Target source archive hash mismatch",
    )
    with tarfile.open(archive_path, "r:gz") as archive:
        member = archive.getmember(PLAN_MEMBER)
        require(member.isfile(), "Archived target plan must be a regular file")
        archived = archive.extractfile(member).read()
    archived_sha = hashlib.sha256(archived).hexdigest()
    require(
        frozen["documents"].get(PLAN_MEMBER) == archived_sha,
        "Archived target plan is not bound by the original source freeze",
    )
    return dict(
        current_plan=eligibility.file_binding(plan_path),
        source_freeze=eligibility.file_binding(freeze_path, json_value=frozen),
        source_archive=eligibility.file_binding(archive_path),
        archived_plan_member=PLAN_MEMBER,
        archived_plan_sha256=archived_sha,
        current_plan_matches_source_archive=hashlib.sha256(Path(plan_path).read_bytes()).hexdigest()
        == archived_sha,
        distinction="The target source binds its archived historical plan. This fit separately binds the current prospective plan and its pre-outcome amendments; those documents are not relabeled as identical.",
    )


def fit(
    transfer_root,
    source_root,
    source_audit_path,
    directions_path,
    extraction_root,
    eligibility_path,
    plan_path,
    out,
):
    transfer_root, source_root, directions_path, extraction_root, out = map(
        Path, (transfer_root, source_root, directions_path, extraction_root, out)
    )
    require(not out.exists(), "Candidate fits are immutable; choose a new output directory")
    receipt = read(eligibility_path)
    with tempfile.TemporaryDirectory(prefix="target-fit-eligibility-") as temporary:
        recomputed = eligibility.evaluate(
            transfer_root,
            source_root,
            source_audit_path,
            directions_path,
            extraction_root,
            plan_path,
            Path(temporary) / "receipt.json",
        )
    require(
        digest(receipt) == digest(recomputed),
        "Original eligibility receipt differs from independently recomputed bound inputs",
    )
    manifest, assignments, records, locations, vectors, layers, positions = load_collection(source_root)
    split = assignment_split(manifest)
    plans = archived_plan_binding(source_root, manifest, plan_path)
    require(manifest == read(source_root / "manifest.json"), "Target source changed during capture loading")
    li, pi = layers.index(47), positions.index("record")
    captures = vectors[:, li, pi].astype(np.float64)
    del vectors
    capture_bindings = [
        dict(
            trial_id=a["trial_id"],
            path=str(
                (locations[a["trial_id"]] / records[a["trial_id"]]["activation_artifact"]["path"]).resolve()
            ),
            sha256=records[a["trial_id"]]["activation_artifact"]["sha256"],
            row=records[a["trial_id"]]["activation_artifact"]["row"],
        )
        for a in assignments
    ]
    del records, locations
    with np.load(directions_path, allow_pickle=False) as z:
        original = {key: z[key] for key in z.files}
    with np.load(transfer_root / "scores.npz", allow_pickle=False) as z:
        scores = z["scores"]
    si = original["sources"].tolist().index("pooled:balanced")
    oli, opi = original["layers"].tolist().index(47), original["positions"].tolist().index("record")
    actual = captures @ original["unit"][:, oli, opi].T
    require(
        np.array_equal(actual, scores[:, :, oli, opi], equal_nan=True),
        "Bound transfer projections do not match the actual source record captures",
    )
    semantic = semantic_captures(assignments, captures)
    components, counts = calibration_components(semantic)
    original_passed = receipt["eligible_original_axis"]
    selected = {}
    if original_passed:
        for key in ("unit", "contrast", "contrast_norm", "threshold", "center"):
            selected[key] = original[key][si : si + 1, oli : oli + 1, opi : opi + 1].copy()
        origin = "original_extraction"
    else:
        gain, center = components["information"], components["center"]
        norm = float(np.linalg.norm(gain))
        unit = gain / norm if np.isfinite(norm) and norm > 0 else np.full_like(gain, np.nan)
        selected = dict(
            unit=unit[None, None, None],
            contrast=gain[None, None, None],
            contrast_norm=np.array([[[norm]]]),
            threshold=np.array([[[float(center @ unit)]]]),
            center=center[None, None, None],
        )
        origin = "target_calibration"
    info_norm = float(selected["contrast_norm"][0, 0, 0])
    unit, threshold = selected["unit"][0, 0, 0], float(selected["threshold"][0, 0, 0])
    available = bool(
        np.isfinite(unit).all() and np.isfinite(info_norm) and info_norm > 0 and np.isfinite(threshold)
    )
    check = check_candidate(semantic, unit, threshold)
    candidate_eligible = bool(available and (original_passed or check["passed"]))
    control_norms = {name: float(np.linalg.norm(components[name])) for name in ("target_u", "target_q")}
    controls_available = bool(
        available
        and all(np.isfinite(value) and value >= 0.01 * info_norm for value in control_norms.values())
    )
    output = dict(
        sources=np.array([OUTPUT_SOURCE]), layers=np.array([47]), positions=np.array(["record"]), **selected
    )
    for name in control_norms:
        output[f"{name}_contrast"] = components[name][None, None, None]
        output[f"{name}_norm"] = np.array([[[control_norms[name]]]])
    fit_assignments = [a for a in assignments if a["index"] < 12 and a["kind"] in eligibility.POLARITIES]
    calibration_origin = dict(
        manifest_sha256=digest(manifest),
        assignment_sha256=digest(fit_assignments),
        raw_context_rows=len(fit_assignments),
        indices=[0, 12],
        semantic_worlds_by_family=counts,
        weighting="Average duplicate named-cell observations; average new/repeat within each semantic cell; equal cells, unique semantic worlds, and families.",
    )
    original_analysis = read(directions_path.parent / "analysis.json")
    information_origin = dict(
        origin=origin,
        manifest_sha256=original_analysis["manifest_sha256"] if original_passed else digest(manifest),
        source="pooled:balanced" if original_passed else OUTPUT_SOURCE,
        layer=47,
        position="record",
        original_axis_retained_exactly=bool(original_passed),
        original_direction_sha256=receipt["original_direction_sha256"],
        original_extraction_manifest_sha256=original_analysis["manifest_sha256"],
        fit_on_target_calibration=not original_passed,
        target_calibration=None if original_passed else calibration_origin,
        threshold_origin="Exact original extraction threshold"
        if original_passed
        else "Calibration-only equal-family semantic-cell grand center projected onto the once-oriented gain contrast; no check threshold fit",
    )
    origins = dict(information=information_origin)
    for name in control_norms:
        origins[name] = dict(
            origin="target_calibration",
            **calibration_origin,
            orientation="Canonical named report order: second minus first independent report"
            if name == "target_u"
            else "Canonical named report order: second minus first offered report",
            raw_display_column_orientation=False,
            raw_contrast_norm=control_norms[name],
            normalized_before_storage=False,
        )
    result = dict(
        schema_version=1,
        direction_schema="target_candidate_controls_v1",
        development_only=True,
        model=manifest["model"],
        manifest_sha256=digest(manifest),
        assignment_split=split,
        selected_information_origin=origin,
        original_transfer_eligible=bool(original_passed),
        candidate_eligible=candidate_eligible,
        controls_available=controls_available,
        development_check_semantic_readiness=receipt["development_check_semantic_readiness"],
        passed=bool(
            candidate_eligible and controls_available and receipt["development_check_semantic_readiness"]
        ),
        fixed_site=dict(layer=47, position="record"),
        output_source=OUTPUT_SOURCE,
        information_quantity="expected_target_information_gain_bits",
        full_world_information_gain_bits=1,
        selected_information_norm=info_norm if np.isfinite(info_norm) else None,
        target_control_raw_norms=control_norms,
        minimum_control_norm_fraction=0.01,
        component_origins=origins,
        check_block_evaluation=check,
        check_block_role="Descriptive for retained original axis; the sole fallback acceptance check for a target-calibration axis. Overlapping calibration semantics are excluded before scoring.",
        fit_trial_ids=[a["trial_id"] for a in fit_assignments],
        all_assigned_record_captures_retained=len(captures),
        no_check_fitting=True,
        no_alternative_site_search=True,
        no_post_check_sign_flip=True,
        no_outcome_correctness_filter=True,
        source_context_rule="New/repeat share the factual record prefix and are averaged within a semantic cell for fitting, not counted as independent representation replications. Instrument, memory, and count captures do not enter the fit.",
        source_plan_versions=plans,
        bindings=dict(
            eligibility_receipt=eligibility.file_binding(eligibility_path, json_value=receipt),
            eligibility_inputs=receipt["bindings"],
            source_capture_selection_sha256=digest(capture_bindings),
            source_capture_files=[
                dict(path=path, sha256=sha)
                for path, sha in sorted({(row["path"], row["sha256"]) for row in capture_bindings})
            ],
            fitter=eligibility.file_binding(__file__),
            capture_loader=eligibility.file_binding(load_collection.__code__.co_filename),
        ),
        interpretation="Development candidate and control construction only. A failed original transfer remains failed even if the separate fallback passes. A failed candidate, unavailable control, or inadequate novel check panel stops this candidate program; an NPZ alone is not admission for scientific natural experiments.",
    )
    out.mkdir(parents=True, exist_ok=False)
    with (out / "directions.npz").open("xb") as handle:
        np.savez_compressed(handle, **output)
    result["directions_sha256"] = hashlib.sha256((out / "directions.npz").read_bytes()).hexdigest()
    write_once(out / "analysis.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transfer", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--directions", type=Path, required=True)
    parser.add_argument("--extraction", type=Path, required=True)
    parser.add_argument("--eligibility", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=eligibility.PLAN)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = fit(
        args.transfer,
        args.source,
        args.source_audit,
        args.directions,
        args.extraction,
        args.eligibility,
        args.plan,
        args.out,
    )
    print(
        {
            key: result[key]
            for key in ("selected_information_origin", "candidate_eligible", "controls_available", "passed")
        }
    )
