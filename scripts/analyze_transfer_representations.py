"""Descriptive transfer of every frozen axis; no fit or favorable-cell selection."""

import argparse
from collections import defaultdict
import hashlib
from pathlib import Path
import numpy as np
from curiosity_study.artifacts import read, write_once, digest
from curiosity_study.representation import projection_metrics
from curiosity_study.transfer_representation import transfer_groups, project_frozen, transfer_estimand
from analyze_thinking_representations import load_collection
from prepare_replay_interventions import split_provenance


def analyze(root, audit_path, direction_path, extraction_path, out):
    audit = read(audit_path)
    manifest = read(root / "manifest.json")
    assert audit["manifest_sha256"] == digest(manifest) and audit["complete"]
    assert audit.get("gates", {}).get("passed"), "Validate the transfer instrument before interpretation"
    source = read(direction_path.parent / "analysis.json")
    sha = hashlib.sha256(direction_path.read_bytes()).hexdigest()
    assert source["directions_sha256"] == sha and source["model"] == manifest["model"]
    assert source["manifest_sha256"] != digest(manifest), "This is a separate transfer collection"
    with np.load(direction_path, allow_pickle=False) as z:
        fitted = {key: z[key] for key in z.files}
    manifest, assignments, records, locations, vectors, layers, positions = load_collection(root)
    if any(a.get("task_revision") != manifest["task_revision"] for a in assignments):
        raise ValueError("Transfer assignments must match the manifest task revision")
    estimand = transfer_estimand(assignments)
    provenance = split_provenance(
        dict(inference_scope="development", direction_extraction_collection=str(extraction_path)),
        manifest,
        source,
        assignments,
    )
    assert layers == list(fitted["layers"]) and positions == list(fitted["positions"])
    groups = transfer_groups(assignments)
    # An extraction position absent in every source world remains absent, rather
    # than being estimated from the transfer data. Project only finite axes.
    units = fitted["unit"]
    finite = np.isfinite(units).all(-1)
    safe_units = np.where(finite[..., None], units, 0)
    scores = project_frozen(vectors, safe_units)
    scores[:, ~finite] = np.nan
    del vectors, records, locations
    targets = defaultdict(list)
    for key, rows in groups.items():
        family, seed, index, kind, template = key
        targets[(family, kind, template)].append(rows)
    metrics = []
    for si, name in enumerate(fitted["sources"]):
        for (family, kind, template), indices in sorted(targets.items()):
            for li, layer in enumerate(layers):
                for pi, position in enumerate(positions):
                    values = scores[np.array(indices), si, li, pi]
                    metrics.append(
                        dict(
                            source=str(name),
                            target_family=family,
                            target_kind=kind,
                            template=template,
                            layer=int(layer),
                            position=position,
                            assigned_worlds=len(indices),
                            source_axis_available=bool(finite[si, li, pi]),
                            information_quantity=estimand["information_quantity"],
                            **projection_metrics(values, fitted["threshold"][si, li, pi]),
                        )
                    )
    out.mkdir(parents=True, exist_ok=True)
    with (out / "scores.npz").open("xb") as f:
        np.savez_compressed(
            f,
            scores=scores,
            trial_ids=np.array([a["trial_id"] for a in assignments]),
            sources=fitted["sources"],
            layers=fitted["layers"],
            positions=fitted["positions"],
            information_quantity=np.array(estimand["information_quantity"]),
            contrast_gain_labels_bits=np.array(estimand["gain_labels_bits"]),
        )
    result = dict(
        manifest_sha256=digest(manifest),
        phase=manifest["phase"],
        inference_scope=manifest.get("inference_scope", "development"),
        validation_criteria=manifest.get("validation_criteria"),
        source_analysis_sha256=digest(source),
        direction_extraction_manifest_sha256=source["manifest_sha256"],
        directions_sha256=sha,
        source_audit_sha256=digest(audit),
        model=manifest["model"],
        assigned=len(assignments),
        task_revision=manifest["task_revision"],
        descriptive_only=True,
        split_provenance=provenance,
        refitted_on_transfer=False,
        transfer_estimand=estimand,
        metrics=metrics,
        interpretation="All frozen sources, layers and positions are reported with their original sign and extraction threshold. Whole four-cell worlds are the resampling unit. This all-cell transfer readout is descriptive, not a multiplicity-corrected confirmatory test. Missing source axes and target positions remain missing.",
    )
    write_once(out / "analysis.json", result)
    print({key: value for key, value in result.items() if key not in ("metrics", "interpretation")})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--directions", type=Path, required=True)
    parser.add_argument("--extraction", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.run, args.audit, args.directions, args.extraction, args.out)
