"""Freeze a separately chosen original-prompt intervention schedule.

This script never launches cloud work. Natural reasoning is a separate assay
from fixed-transcript replay and requires its own scientific schedule.
"""

import argparse
from collections import defaultdict
import copy
import hashlib
import math
from pathlib import Path
import random
import sys

import numpy as np

from curiosity_study.artifacts import RunStore, digest, read, write_once
from curiosity_study.causal_contrasts import source_descriptor
from curiosity_study.natural_information import CONDITIONING, engine_hashes, generation_regime, make_prefix
from curiosity_study.thinking_information import generation_limit, manifest_generation_limit
from curiosity_study.runner import source_hash
from prepare_replay_interventions import split_provenance, world_id

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "infra"))
from freeze_execution import freeze  # noqa: E402


def expected_source_metadata(config, manifest, audit):
    """An optional prospective source declaration is an exact admission constraint."""
    if "expected_source_metadata" not in config:
        return
    expected = config["expected_source_metadata"]
    fields = {
        "task_revision",
        "phase",
        "inference_scope",
        "assigned",
        "max_new_tokens",
        "complete",
        "global_gates_passed",
        "validation_criteria",
    }
    if not isinstance(expected, dict) or set(expected) != fields:
        raise ValueError("expected_source_metadata requires the complete explicit source declaration")
    actual = dict(
        task_revision=manifest.get("task_revision"),
        phase=manifest.get("phase"),
        inference_scope=manifest.get("inference_scope"),
        assigned=len(manifest["assignments"] + manifest["diagnostics"]),
        max_new_tokens=manifest_generation_limit(manifest),
        complete=audit.get("complete"),
        global_gates_passed=audit.get("gates", {}).get("passed"),
        validation_criteria=manifest.get("validation_criteria"),
    )
    # Digests preserve type distinctions such as true versus 1.
    if digest(actual) != digest(expected):
        raise ValueError("Source metadata differs from expected_source_metadata")
    if (
        audit.get("assigned") != actual["assigned"]
        or any(audit.get(key) != actual[key] for key in ("phase", "inference_scope", "validation_criteria"))
        or any(a.get("task_revision") != actual["task_revision"] for a in manifest["diagnostics"])
    ):
        raise ValueError("Source audit/assignments disagree with expected_source_metadata")


def packing_policy(config):
    """Describe the fixed execution layout without changing legacy schedules."""
    size = config["batch_size"]
    if type(size) is not int or size not in (4, 8, 16, 32):
        raise ValueError("Choose a prospective natural-reasoning batch size of 4, 8, 16, or 32")
    if config.get("packing_mode") == "condition_major_v1":
        if config.get("prefix_group_size") is not None:
            raise ValueError("Condition-major packing has no repeated-prefix group size")
        return dict(
            mode="condition_major_v1",
            prefixes_per_cohort=size,
            ordering="seeded_world_blocks; fixed_prefix_rows; seeded_condition_order_per_block_and_kind",
            padding="none; require complete source blocks and complete condition cohorts",
            pairing="same source row, companion prompts, seed, and prompt width across every condition",
        )
    if config.get("packing_mode") is not None:
        raise ValueError("Unknown prospective natural packing mode")
    group_size = config.get("prefix_group_size", size)
    if type(group_size) is not int or group_size != (16 if size == 32 else size):
        raise ValueError(
            "Batch 32 requires explicit prefix_group_size=16; legacy batches use their batch size"
        )
    return dict(
        mode="paired_prefixes_v1" if size == 32 else "single_prefix_cohorts_v1",
        prefix_group_size=group_size,
        prefixes_per_cohort=2 if size == 32 else 1,
        ordering="seeded_world_shuffle_then_kind_template_cell_source_id"
        if size == 32
        else "seeded_prefix_shuffle",
        padding="zero_dose_shams_of_the_same_source_prefix; not additional source units",
    )


def ordered_prefix_keys(config, prefixes, policy):
    keys = sorted(prefixes)
    if policy["mode"] in ("single_prefix_cohorts_v1", "condition_major_v1"):
        random.Random(config["order_seed"]).shuffle(keys)
        return keys
    worlds = defaultdict(lambda: defaultdict(list))
    for key in keys:
        prefix = prefixes[key]
        assignment = prefix["assignment"]
        if key != prefix.get("source_trial_id") or key != assignment["trial_id"]:
            raise ValueError("Packed groups must identify distinct actual source trials, not prefix aliases")
        worlds[world_id(assignment)][(assignment["kind"], assignment["template"])].append(key)
    world_keys = sorted(worlds)
    random.Random(config["order_seed"]).shuffle(world_keys)
    keys = []
    for world in world_keys:
        for group in sorted(worlds[world]):
            selected = worlds[world][group]
            if len(selected) % 2:
                raise ValueError(
                    "Packed cohorts require even prefix counts in every world/kind/template group"
                )
            keys.extend(
                sorted(
                    selected, key=lambda key: (source_descriptor(prefixes[key]["assignment"])["cell"], key)
                )
            )
    return keys


def condition_major_rows(config, prefixes, per_prefix):
    """Reuse identical prompt rows across conditions; never pad with duplicate sources."""
    size = config["batch_size"]
    if not prefixes:
        raise ValueError("Condition-major packing requires nonempty complete source blocks")
    worlds = defaultdict(lambda: defaultdict(list))
    for key, prefix in prefixes.items():
        assignment = prefix["assignment"]
        if key != prefix.get("source_trial_id") or key != assignment["trial_id"]:
            raise ValueError("Condition-major rows require actual distinct source trials")
        worlds[world_id(assignment)][(assignment["kind"], assignment["template"])].append(key)
    keys = sorted(worlds)
    random.Random(config["order_seed"]).shuffle(keys)
    groups = sorted(worlds[keys[0]])
    counts = {len(items) for world in worlds.values() for items in world.values()}
    if len(counts) != 1 or any(sorted(world) != groups for world in worlds.values()):
        raise ValueError("Every world must supply the same complete readout/template groups")
    cells = counts.pop()
    worlds_per_block = math.lcm(size, cells) // cells
    if cells not in (4, 6) or len(keys) % worlds_per_block:
        raise ValueError("Require complete whole-world source blocks that exactly fill each cohort")
    expected_cells = None
    for world in worlds.values():
        for group in groups:
            ordered = sorted(
                world[group], key=lambda key: (source_descriptor(prefixes[key]["assignment"])["cell"], key)
            )
            selected_cells = [source_descriptor(prefixes[key]["assignment"])["cell"] for key in ordered]
            if len(set(selected_cells)) != cells or (
                expected_cells is not None and selected_cells != expected_cells
            ):
                raise ValueError("Every source world must contain the same distinct complete factorial cells")
            expected_cells = selected_cells
            world[group] = ordered
    rows = []
    for block, offset in enumerate(range(0, len(keys), worlds_per_block)):
        selected_worlds = keys[offset : offset + worlds_per_block]
        for group in groups:
            cohort_keys = [key for world in selected_worlds for key in worlds[world][group]]
            labels = [row["condition"] for row in per_prefix[cohort_keys[0]]]
            if any([row["condition"] for row in per_prefix[key]] != labels for key in cohort_keys):
                raise ValueError("Every source row must receive exactly the same selected condition set")
            conditions = list(range(len(labels)))
            random.Random(f"{config['order_seed']}:{block}:{group}").shuffle(conditions)
            for condition in conditions:
                for position, key in enumerate(cohort_keys):
                    row = dict(per_prefix[key][condition], source_block=block, source_row=position % size)
                    if len(cohort_keys) > size:
                        row["source_subcohort"] = position // size
                    rows.append(row)
    return rows


def build_schedule(config, prefixes, fitted):
    """Construct paired additive arms without viewing any completion outcomes."""
    policy = packing_policy(config)
    group_size = policy.get("prefix_group_size")
    if not config["positions"] or not set(config["positions"]) <= {"record", "prefill"}:
        raise ValueError("Natural interventions use record or prefill axes, never generated-token axes")
    if not config["layers"] or len(set(config["layers"])) != len(config["layers"]):
        raise ValueError("Require distinct selected layers")
    if len(set(config["positions"])) != len(config["positions"]):
        raise ValueError("Require distinct selected positions")
    names = list(fitted["sources"])
    layers, positions = list(fitted["layers"]), list(fitted["positions"])
    reference = names.index(config["dose_reference_source"])
    keys = ordered_prefix_keys(config, prefixes, policy)
    rows, per_prefix = [], {}
    for key in keys:
        prefix = prefixes[key]

        def row(label, **kwargs):
            return dict(
                trial_id=f"{config['name']}-{key}-{label}",
                prefix_id=key,
                kind=prefix["assignment"]["kind"],
                condition=label,
                **kwargs,
            )

        local = [
            row(
                "baseline",
                arm="baseline",
                layer=None,
                position=None,
                dose=0.0,
                dose_multiplier=0.0,
                direction_name="baseline",
            )
        ]
        for layer in config["layers"]:
            li = layers.index(layer)
            for position in config["positions"]:
                pi = positions.index(position)
                target = "final" if position == "prefill" else position
                norm = float(fitted["contrast_norm"][reference, li, pi])
                if not np.isfinite(norm) or norm <= 0:
                    raise ValueError(
                        "Dose reference axis must be finite and nonzero at every selected location"
                    )
                tag = f"l{layer}-{position}"
                local.append(
                    row(
                        tag + "-sham",
                        arm="sham",
                        layer=layer,
                        position=target,
                        capture_position=position,
                        dose=0.0,
                        dose_multiplier=0.0,
                        direction_name="sham",
                    )
                )
                for direction in config["directions"]:
                    si = names.index(direction["source"])
                    vector = fitted[direction["array"]][si, li, pi]
                    if not np.isfinite(vector).all() or np.linalg.norm(vector) <= 0:
                        raise ValueError("Every selected intervention direction must be available")
                    for multiplier in config["dose_multipliers"]:
                        if not np.isfinite(multiplier) or multiplier == 0:
                            raise ValueError(
                                "Scientific additive doses must be finite and nonzero; use sham for zero"
                            )
                        local.append(
                            row(
                                f"{tag}-{direction['name']}-{multiplier}",
                                arm="direction",
                                layer=layer,
                                position=target,
                                capture_position=position,
                                dose=float(multiplier * norm),
                                dose_multiplier=multiplier,
                                direction_name=direction["name"],
                                source_index=si,
                                layer_index=li,
                                position_index=pi,
                                array=direction["array"],
                                patch_mode="add",
                            )
                        )
                for seed in config["random_seeds"]:
                    for multiplier in config["random_dose_multipliers"]:
                        if not np.isfinite(multiplier) or multiplier == 0:
                            raise ValueError("Random-control doses must be finite and nonzero")
                        local.append(
                            row(
                                f"{tag}-random{seed}-{multiplier}",
                                arm="random",
                                layer=layer,
                                position=target,
                                capture_position=position,
                                dose=float(multiplier * norm),
                                dose_multiplier=multiplier,
                                direction_name=f"random{seed}",
                                random_seed=seed,
                                patch_mode="add",
                            )
                        )
        if policy["mode"] == "paired_prefixes_v1" and len(local) > group_size:
            raise ValueError("Every packed prefix's complete selected arm set must fit within 16 rows")
        while group_size is not None and len(local) % group_size:
            local.append(
                row(
                    f"padding{len(local)}",
                    arm="sham",
                    layer=config["layers"][0],
                    position="final",
                    capture_position="prefill",
                    dose=0.0,
                    dose_multiplier=0.0,
                    direction_name="padding_control",
                )
            )
        per_prefix[key] = local
        rows.extend(local)
    if policy["mode"] == "condition_major_v1":
        rows = condition_major_rows(config, prefixes, per_prefix)
    if not rows or len({a["trial_id"] for a in rows}) != len(rows):
        raise ValueError("Natural schedule must have nonempty, distinct intervention trial IDs")
    return rows


def partition_worlds(rows, prefixes, size, workers):
    """Keep all selected branches and arms of a world on one worker."""
    if workers not in (1, 2, 3, 4):
        raise ValueError("Use between one and four workers")
    if not rows or len(rows) % size:
        raise ValueError("Natural partitions require complete fixed-size cohorts")
    allocation, selected = {}, [[] for _ in range(workers)]
    world_blocks = {}
    for offset in range(0, len(rows), size):
        cohort = rows[offset : offset + size]
        worlds = {world_id(prefixes[a["prefix_id"]]["assignment"]) for a in cohort}
        if any("source_block" in row for row in cohort):
            blocks = {row.get("source_block") for row in cohort}
            if (
                len(blocks) != 1
                or None in blocks
                or [row.get("source_row") for row in cohort] != list(range(size))
            ):
                raise ValueError(
                    "Condition-major cohorts must retain one complete source block in fixed row order"
                )
            key = ("source_block", next(iter(blocks)))
            for world in worlds:
                if world_blocks.setdefault(world, key) != key:
                    raise ValueError("A world cannot be split across condition-major source blocks")
        else:
            if len(worlds) != 1:
                raise ValueError("A natural cohort must belong to exactly one underlying world")
            key = next(iter(worlds))
        worker = allocation.setdefault(key, len(allocation) % workers)
        selected[worker].append(offset // size)
    if any(not indices for indices in selected):
        raise ValueError("There are fewer selected worlds than workers")
    return selected


def prepare(config_path):
    config = read(config_path)
    if config.get("source_eligibility_certificate") is not None:
        raise ValueError("Natural reasoning requires global source validation, not scoped replay admission")
    if (
        config.get("inference_scope") in ("confirmation", "fresh_confirmation")
        or config["phase"] == "confirmation"
    ) and generation_limit(config) != 4096:
        raise ValueError("New natural confirmation schedules require 4096 tokens for every assigned kind")
    if Path(config["name"]).name != config["name"]:
        raise ValueError("Use a single directory name for the natural run")
    source = Path(config["source_collection"])
    source_manifest = read(source / "manifest.json")
    source_limit = manifest_generation_limit(source_manifest)
    audit = read(config["source_audit"])
    expected_source_metadata(config, source_manifest, audit)
    if (
        audit["manifest_sha256"] != digest(source_manifest)
        or not audit["complete"]
        or not audit["gates"]["passed"]
        or audit.get("max_new_tokens", 2048) != source_limit
    ):
        raise ValueError("The source collection must pass its predefined complete instrument validation")
    directions = Path(config["directions_local"])
    direction_sha = hashlib.sha256(directions.read_bytes()).hexdigest()
    source_analysis = read(directions.parent / "analysis.json")
    if direction_sha != config["directions_sha256"] or source_analysis["directions_sha256"] != direction_sha:
        raise ValueError("Direction artifact binding mismatch")
    if source_analysis["model"] != source_manifest["model"]:
        raise ValueError("Direction extraction and natural inference must use the same model")
    with np.load(directions, allow_pickle=False) as z:
        fitted = {key: z[key] for key in z.files}
    low, high = config["world_index_range"]
    selected = [
        a for a in source_manifest["diagnostics"] if low <= a["index"] < high and a["kind"] in config["kinds"]
    ]
    provenance = split_provenance(config, source_manifest, source_analysis, selected)
    by_id = {a["trial_id"]: a for a in selected}
    if len(by_id) != len(selected):
        raise ValueError("Duplicate selected source trial IDs")
    children = (
        read(source / "children.json")
        if (source / "children.json").exists()
        else [dict(path=str(source), manifest_sha256=digest(source_manifest))]
    )
    prefixes = {}
    for child in children:
        path = Path(child["path"])
        manifest = read(path / "manifest.json")
        if digest(manifest) != child["manifest_sha256"]:
            raise ValueError("Source child manifest binding mismatch")
        if path.resolve() != source.resolve() and manifest["parent_manifest_sha256"] != digest(
            source_manifest
        ):
            raise ValueError("Source child has a different parent")
        records = RunStore(path, manifest).completed()
        for a in manifest["diagnostics"]:
            key = a["trial_id"]
            if key not in by_id:
                continue
            if a != by_id[key] or key in prefixes or records[key]["assignment"] != a:
                raise ValueError("Source partitions disagree with selected assignments")
            # No filter on accuracy, closure, valid format, or source reasoning length.
            prefixes[key] = make_prefix(records[key], path.name, digest(manifest))
    if set(prefixes) != set(by_id):
        raise ValueError(
            "Natural stage requires every assigned original prompt; no completion-based exclusions"
        )
    rows = build_schedule(config, prefixes, fitted)
    regime = generation_regime(config, prefixes)
    policy = packing_policy(config)
    size = config["batch_size"]
    partitions = partition_worlds(rows, prefixes, size, config["workers"])
    manifest = dict(
        schema_version=3,
        assay="natural_information",
        phase=config["phase"],
        inference_scope=config["inference_scope"],
        split_provenance=provenance,
        split_provenance_sha256=digest(provenance),
        model=source_manifest["model"],
        execution_gpu="H200",
        execution=dict(backend="torch_batched_v1", batch_size=size),
        packing_policy=policy,
        packing_policy_sha256=digest(policy),
        source_sha256=source_hash(),
        engine_sha256=engine_hashes(),
        assignments=[],
        diagnostics=rows,
        assignment_sha256=digest(rows),
        prefixes=prefixes,
        prefixes_sha256=digest(prefixes),
        assigned_sources=len(prefixes),
        excluded_sources=[],
        source_parent_manifest_sha256=digest(source_manifest),
        source_audit_sha256=digest(audit),
        capture_layers=config["layers"],
        max_new_tokens=generation_limit(config),
        generation_regime=regime,
        greedy=False,
        direction_input=dict(path=config["directions_remote"], sha256=direction_sha),
        direction_analysis_sha256=digest(source_analysis),
        direction_extraction_manifest_sha256=source_analysis["manifest_sha256"],
        intervention_schedule=config,
        intervention_schedule_sha256=digest(config),
        invalid_output_policy="terminal_no_retry",
        conditioning=CONDITIONING,
        padded_prefix_groups=policy["mode"] != "condition_major_v1",
        initial_label_readout="Next-token scores before newly generated reasoning; not final-answer confidence or a final-answer causal margin.",
        engineering_guard="Real prefill hook, sham, row-isolation, independent-seed and short sampled-continuation checks at worker startup; these are not a separate long-context scientific validation.",
    )
    parent = ROOT / "runs" / config["name"]
    paths = [ROOT / "runs" / f"{config['name']}-part-{i:02d}" for i in range(config["workers"])]
    if parent.exists() or any(path.exists() for path in paths):
        raise ValueError("Natural parent or child run already exists")
    write_once(parent / "manifest.json", manifest)
    write_once(parent / "schedule.json", config)
    descriptors = []
    for index, (path, selected_cohorts) in enumerate(zip(paths, partitions)):
        child = copy.deepcopy(manifest)
        child_rows = [a for ci in selected_cohorts for a in rows[ci * size : (ci + 1) * size]]
        child.update(
            diagnostics=child_rows,
            assignment_sha256=digest(child_rows),
            parent_manifest_sha256=digest(manifest),
            partition_index=index,
            parent_cohort_indices=selected_cohorts,
        )
        write_once(path / "manifest.json", child)
        descriptors.append(
            dict(
                path=str(path),
                manifest_sha256=digest(child),
                branches=len(child_rows),
                parent_cohorts=selected_cohorts,
            )
        )
    write_once(parent / "children.json", descriptors)
    freeze(
        ROOT,
        parent,
        [
            "reports/reasoning-information-v1-analysis-plan.md",
            "reports/reasoning-information-review-amendment.md",
            "reports/reasoning-information-development-scope-amendment.md",
            "reports/natural-condition-major-engineering-amendment.md",
            "reports/target-relevance-causal-prospective-plan.md",
            "reports/replay-reduction-engineering-amendment.md",
        ],
    )
    print(dict(prefixes=len(prefixes), excluded=0, measurements=len(rows), workers=config["workers"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.config)
