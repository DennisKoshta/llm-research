"""Freeze donor projection interchange after source validation and replay parity.

Required config fields match prepare_replay_interventions for source/direction/audit
paths and split provenance. Select one `layer`, `position`, `information_source`,
and `answer_source`, plus `world_index_range`, `kinds`, >=5 `random_seeds`,
`order_seed`, `batch_size`, and `workers`. Optional `control_relations` defaults to
both opposite_gain and same_gain. This script prepares a frozen run; it never launches it.
"""

import argparse
from collections import defaultdict
import copy
import hashlib
from pathlib import Path
import random
import re
import sys
import numpy as np
from curiosity_study.artifacts import read, write_once, digest, RunStore
from curiosity_study.projection_interchange import (
    build_recipient_rows,
    complete_groups,
    donor_ids,
    factorial_cell,
    unit_direction,
    vector_sha256,
)
from curiosity_study.replay_information import make_prefix, engine_hashes
from curiosity_study.runner import source_hash
from curiosity_study.thinking_information import task_module

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "infra"))
from prepare_replay_interventions import split_provenance  # noqa: E402
from freeze_execution import freeze  # noqa: E402


def load_source_vectors(source, manifest, selected, layer, position):
    """Bind exact replay prefixes and the selected original capture without fitting."""
    expected = {a["trial_id"]: a for a in selected}
    if len(expected) != len(selected):
        raise ValueError("Duplicate selected source assignment")
    children = (
        read(source / "children.json")
        if (source / "children.json").exists()
        else [dict(path=str(source), manifest_sha256=digest(manifest))]
    )
    prefixes, captures, reasons, seen = {}, {}, {}, set()
    by_file = defaultdict(list)
    for child in children:
        path = Path(child["path"])
        cm = read(path / "manifest.json")
        if digest(cm) != child["manifest_sha256"]:
            raise ValueError("Source child manifest hash mismatch")
        if path.resolve() != source.resolve() and cm.get("parent_manifest_sha256") != digest(manifest):
            raise ValueError("Source child is not bound to the requested parent")
        records = RunStore(path, cm).completed()
        for a in cm["diagnostics"]:
            key = a["trial_id"]
            if key not in expected:
                continue
            if a != expected[key] or key in seen:
                raise ValueError("Source partition disagrees with assigned parent selection")
            seen.add(key)
            record = records[key]
            if record["assignment"] != a or digest(record["task"]) != digest(task_module(a).task(a)):
                raise ValueError("Source record assignment or independent task oracle mismatch")
            cell = factorial_cell(a)
            if record["task"]["uncertainty_bits"] != 1 or record["task"]["information_bits"] != int(
                cell[0] != cell[1]
            ):
                raise ValueError("Source world does not implement the intended fixed-uncertainty factorial")
            prefix = make_prefix(record, path.name, digest(cm))
            if prefix is None:
                reasons[key] = "missing eligible pre-answer replay prefix"
                continue
            meta = record["activation_artifact"]
            if layer not in meta["layers"] or position not in meta["positions"]:
                raise ValueError("Requested interchange site was not captured")
            li, pi = meta["layers"].index(layer), meta["positions"].index(position)
            if meta["token_positions"][pi] < 0 or not meta["available"][li][pi]:
                reasons[key] = "selected original activation is unavailable"
                continue
            prefixes[key] = prefix
            by_file[path / meta["path"]].append((key, prefix, meta, li, pi))
    if seen != set(expected):
        raise ValueError("Incomplete source partition coverage")
    for file, rows in by_file.items():
        sha = hashlib.sha256(file.read_bytes()).hexdigest()
        with np.load(file, allow_pickle=False) as z:
            for key, prefix, meta, li, pi in rows:
                if sha != meta["sha256"]:
                    raise ValueError("Source activation artifact hash mismatch")
                if list(z["layers"]) != meta["layers"] or list(z["positions"]) != meta["positions"]:
                    raise ValueError("Source activation axes disagree with record metadata")
                row = meta["row"]
                if z["trial_ids"][row] != key or not bool(z["available"][row, li, pi]):
                    raise ValueError("Source activation row binding or availability mismatch")
                vector = np.asarray(z["hidden"][row, li, pi], dtype=np.float32).copy()
                if vector.ndim != 1 or not np.isfinite(vector).all():
                    raise ValueError("Available source activation must be one complete finite vector")
                captures[key] = dict(
                    vector=vector,
                    metadata=dict(
                        source_trial_id=key,
                        source_run=prefix["source_run"],
                        source_manifest_sha256=prefix["source_manifest_sha256"],
                        source_record_sha256=prefix["source_record_sha256"],
                        path=meta["path"],
                        sha256=sha,
                        row=row,
                        layer=layer,
                        position=position,
                        token_position=meta["token_positions"][pi],
                        vector_sha256=vector_sha256(vector),
                        vector_shape=list(vector.shape),
                        vector_encoding="float32 little-endian contiguous values",
                    ),
                )
    return prefixes, captures, reasons


def prepare(config_path):
    config = read(config_path)
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{1,100}", config["name"]):
        raise ValueError("Use a simple lowercase run name")
    if type(config["workers"]) is not int or not 1 <= config["workers"] <= 4:
        raise ValueError("Use one to four bounded workers")
    if config["batch_size"] not in (4, 8, 16):
        raise ValueError("Use batch size 4, 8, or 16")
    low, high = config["world_index_range"]
    if type(low) is not int or type(high) is not int or not 0 <= low < high:
        raise ValueError("World range must be nonnegative, increasing, and half-open")
    if not config["kinds"] or not set(config["kinds"]) <= {"new", "repeat", "memory", "count", "instrument"}:
        raise ValueError("Unsupported interchange diagnostic kind")
    source = Path(config["source_collection"])
    manifest = read(source / "manifest.json")
    if digest(manifest["diagnostics"]) != manifest["assignment_sha256"]:
        raise ValueError("Invalid source assignment hash")
    source_audit = read(config["source_audit"])
    if (
        source_audit["manifest_sha256"] != digest(manifest)
        or not source_audit["complete"]
        or not source_audit["gates"]["passed"]
    ):
        raise ValueError("The source instrument must pass its bound predefined validation gates")
    parity = read(config["parity_audit"])
    if (
        not parity["passed"]
        or parity["model"] != manifest["model"]
        or parity["engine_sha256"] != engine_hashes()
    ):
        raise ValueError("Replay parity must pass for this model and current audited engine")
    direction_file = Path(config["directions_local"])
    direction_sha = hashlib.sha256(direction_file.read_bytes()).hexdigest()
    if direction_sha != config["directions_sha256"]:
        raise ValueError("Frozen direction file hash mismatch")
    analysis = read(direction_file.parent / "analysis.json")
    if analysis["directions_sha256"] != direction_sha or analysis["model"] != manifest["model"]:
        raise ValueError("Direction analysis is not bound to this file/model")
    with np.load(direction_file, allow_pickle=False) as z:
        fitted = {key: z[key] for key in z.files}
    indices = dict(
        information_source=list(fitted["sources"]).index(config["information_source"]),
        answer_source=list(fitted["sources"]).index(config["answer_source"]),
        layer=list(fitted["layers"]).index(config["layer"]),
        position=list(fitted["positions"]).index(config["position"]),
    )
    li, pi = indices["layer"], indices["position"]
    information = unit_direction(fitted["unit"][indices["information_source"], li, pi])
    answer = unit_direction(fitted["unit"][indices["answer_source"], li, pi])
    if information.shape != answer.shape:
        raise ValueError("Information and answer control directions have different dimensions")
    selected = [
        a for a in manifest["diagnostics"] if low <= a["index"] < high and a["kind"] in config["kinds"]
    ]
    groups = complete_groups(selected)
    # Establish split provenance on every assigned case before availability exclusions.
    provenance = split_provenance(config, manifest, analysis, selected)
    prefixes, captures, unavailable = load_source_vectors(
        source, manifest, selected, config["layer"], config["position"]
    )
    eligible, exclusions = {}, []
    for key, cells in groups.items():
        missing = [trial for trial in cells.values() if trial not in prefixes or trial not in captures]
        if missing:
            exclusions.append(
                dict(
                    world=list(key),
                    excluded_sources=sorted(cells.values()),
                    missing_sources=sorted(missing),
                    reasons={trial: unavailable.get(trial, "missing source capture") for trial in missing},
                    policy="exclude entire assigned four-cell world; never filter on answer correctness",
                )
            )
        else:
            eligible[key] = cells
    if not eligible:
        raise ValueError("No complete eligible interchange worlds")
    eligible_ids = {trial for cells in eligible.values() for trial in cells.values()}
    prefixes = {key: value for key, value in prefixes.items() if key in eligible_ids}
    if any(captures[key]["vector"].shape != information.shape for key in eligible_ids):
        raise ValueError("Original activation and frozen direction dimensions disagree")
    donors_by_prefix = {
        trial: donor_ids(cells, cell) for cells in eligible.values() for cell, trial in cells.items()
    }
    order = sorted(prefixes)
    random.Random(config["order_seed"]).shuffle(order)
    assignments = []
    for trial in order:
        assignments.extend(
            build_recipient_rows(
                config, trial, prefixes[trial], donors_by_prefix[trial], captures, information, indices
            )
        )
    size = config["batch_size"]
    cohorts = [assignments[i : i + size] for i in range(0, len(assignments), size)]
    if config["workers"] > len(cohorts):
        raise ValueError("Workers cannot exceed assigned cohorts")
    excluded_ids = sorted({trial for row in exclusions for trial in row["excluded_sources"]})
    parent = ROOT / "runs" / config["name"]
    result = dict(
        schema_version=3,
        assay="replay_information",
        design="projection_interchange_v1",
        phase=config["phase"],
        inference_scope=config["inference_scope"],
        split_provenance=provenance,
        split_provenance_sha256=digest(provenance),
        model=manifest["model"],
        execution_gpu="H200",
        execution=dict(backend="torch_batched_v1", batch_size=size),
        source_sha256=source_hash(),
        engine_sha256=engine_hashes(),
        assignments=[],
        diagnostics=assignments,
        assignment_sha256=digest(assignments),
        prefixes=prefixes,
        prefixes_sha256=digest(prefixes),
        assigned_sources=len(selected),
        excluded_sources=excluded_ids,
        excluded_worlds=exclusions,
        assigned_worlds=len(groups),
        eligible_worlds=len(eligible),
        source_parent_manifest_sha256=digest(manifest),
        source_audit_sha256=digest(source_audit),
        capture_layers=[],
        max_new_tokens=8,
        greedy=False,
        direction_input=dict(path=config["directions_remote"], sha256=direction_sha),
        direction_analysis_sha256=digest(analysis),
        direction_extraction_manifest_sha256=analysis["manifest_sha256"],
        intervention_schedule=config,
        intervention_schedule_sha256=digest(config),
        parity_audit_sha256=digest(parity),
        invalid_output_policy="terminal_no_retry",
        padded_prefix_groups=True,
        interpretation="Raw average information-margin changes measure directional bias and can cancel for a successful bidirectional interchange. Primary counterfactual effects require the separately frozen paired contrast plan, oriented by 1 minus twice the recipient information gain before whole-world aggregation; compare against matched controls and retain both question polarities as distinct endpoints.",
        donor_policy=dict(
            opposite_gain="same first factor, opposite query; fixed world, question kind, and template",
            same_gain="both first factor and query reversed; fixed world, question kind, and template",
            self="original recipient projection; semantic self-donor control, not guaranteed numerical zero",
            zero_sham="separate exact zero-dose additive intervention",
            original_projection_sign="retain all observed targets, including unexpected semantic ordering",
            matched_controls="answer and seeded-random units; runtime scalar target minus replay information projection",
        ),
    )
    write_once(parent / "manifest.json", result)
    write_once(parent / "schedule.json", config)
    children = []
    for index in range(config["workers"]):
        child = copy.deepcopy(result)
        selected_cohorts = list(range(index, len(cohorts), config["workers"]))
        rows = [a for ci in selected_cohorts for a in cohorts[ci]]
        child.update(
            diagnostics=rows,
            assignment_sha256=digest(rows),
            parent_manifest_sha256=digest(result),
            partition_index=index,
            parent_cohort_indices=selected_cohorts,
        )
        path = ROOT / "runs" / f"{config['name']}-part-{index:02d}"
        write_once(path / "manifest.json", child)
        children.append(
            dict(
                path=str(path),
                manifest_sha256=digest(child),
                branches=len(rows),
                parent_cohorts=selected_cohorts,
            )
        )
    write_once(parent / "children.json", children)
    freeze(
        ROOT,
        parent,
        [
            "reports/reasoning-information-v1-analysis-plan.md",
            "reports/reasoning-information-review-amendment.md",
        ],
    )
    print(
        dict(
            prefixes=len(prefixes),
            assigned_worlds=len(groups),
            eligible_worlds=len(eligible),
            excluded=len(excluded_ids),
            measurements=len(assignments),
            workers=config["workers"],
        )
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    prepare(parser.parse_args().config)
