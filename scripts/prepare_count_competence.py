"""Freeze balanced competence validation after its scientific schedule is chosen.

This prepares files only; it never starts cloud work. The thinking-information
task dispatcher must recognize count-competence-v1 before preparation.
"""

import argparse
from collections import defaultdict
import copy
import math
from pathlib import Path
import sys

from curiosity_study import count_competence as competence
from curiosity_study import information_state as base
from curiosity_study.artifacts import digest, read, write_once
from curiosity_study.thinking_information import task_module

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "infra"))
from freeze_execution import freeze  # noqa: E402


def semantic_readiness(assignments):
    """Assignment-only inventory; display permutations never create new worlds."""
    if not assignments or any(a.get("vocabulary") != "heldout" for a in assignments):
        raise ValueError("Semantic readiness currently requires nonempty held-out count assignments")
    worlds = defaultdict(list)
    for a in assignments:
        worlds[(a["seed"], a["family"], a["index"])].append(a)
    inventory = []
    for (seed, family, index), rows in sorted(worlds.items()):
        if len(rows) != 6 or {(a["correct_count"], a["query"]) for a in rows} != {
            (count, query) for count in (1, 2, 4) for query in (0, 1)
        }:
            raise ValueError("Semantic readiness requires every assigned six-cell count world")
        task = competence.task(rows[0])
        order = sorted(range(2), key=task["attributes"].__getitem__)
        candidates = [[row[i] for i in order] for row in task["candidates"]]
        canonical = dict(
            family=family,
            attributes=[task["attributes"][i] for i in order],
            candidates=sorted(candidates),
            true_candidate=candidates[task["truth"]],
        )
        partial = next(a for a in rows if a["correct_count"] == 2)
        variant = {**canonical, "partially_observed_attribute": task["attributes"][partial["known"][0]]}
        inventory.append(
            dict(
                seed=seed,
                family=family,
                index=index,
                semantic_world_sha256=digest(canonical),
                six_cell_assay_variant_sha256=digest(variant),
            )
        )
    # The held-out object generator chooses one of four disjoint pairs for each
    # named attribute and one of four realized candidates. Its row order is irrelevant.
    capacities = dict(
        objects=len(base.HELDOUT_COLOR_PAIRS) * len(base.HELDOUT_SHAPE_PAIRS) * 4,
        machines=math.comb(9, 2) * math.comb(20, 2) ** 2 * 4,
    )
    counts = {}
    for family in ("objects", "machines"):
        selected = [row for row in inventory if row["family"] == family]
        unique = len({row["semantic_world_sha256"] for row in selected})
        counts[family] = dict(
            assigned_world_indices=len(selected),
            unique_underlying_worlds=unique,
            duplicate_underlying_world_indices=len(selected) - unique,
            unique_six_cell_assay_variants=len({row["six_cell_assay_variant_sha256"] for row in selected}),
            underlying_vocabulary_capacity=capacities[family],
            six_cell_variant_capacity=2 * capacities[family],
            maximum_underlying_worlds_remaining_after_this_selection=capacities[family] - unique,
            capacity_for_120_fresh_underlying_worlds=capacities[family] - unique >= 120,
        )
    return dict(
        schema_version=1,
        outcomes_used=False,
        semantic_rule="family, named attributes, unordered candidate rows, and true candidate; ignores index, seed, rendering, row order, query, and partial-observation variant",
        variant_rule="Underlying world plus named property observed in the two-candidate cells; variants sharing an underlying world are not independent worlds",
        families=counts,
        assigned_world_inventory=inventory,
        prospective_confirmation_ready=False,
        readiness_reason="Current object vocabulary has only 64 underlying worlds; 120 fresh unique object worlds are impossible. No expansion or replacement sampling has been authorized by this report.",
        inspected_collection_overlap_audited=False,
    )


def build_manifest(model, items=100, seed=2026101411, template="table", start_index=0, *, mode="validation"):
    """Future preparation uses 4096 for every cell; existing task defaults stay intact."""
    if mode not in ("validation", "development"):
        raise ValueError("Choose explicit validation or development mode")
    if mode == "development" and (type(items) is not int or items != 16):
        raise ValueError("The fixed development count check requires exactly 16 world indices per family")
    manifest = competence.build_manifest(model, items, seed, template, start_index)
    for assignment in manifest["diagnostics"]:
        assignment["max_new_tokens"] = 4096
    manifest.update(
        max_new_tokens=4096,
        inference_revision="thinking-4096-v1",
        assignment_sha256=digest(manifest["diagnostics"]),
    )
    if mode == "development":
        manifest.update(
            phase="development_validation",
            inference_scope="development",
            validation_criteria=dict(
                minimum_distinct_items_per_cell=16,
                minimum_accuracy=0.9,
                minimum_valid_format=0.95,
                cell_axes=["family", "correct_count", "query"],
                item_identity="task_revision, seed, family, index; all six branches retained together",
                distinctness="Distinct sampled world indices; semantic duplication is separately inventoried, never called independent confirmation.",
            ),
            semantic_readiness=semantic_readiness(manifest["diagnostics"]),
            development_design=dict(
                worlds_per_family=16,
                underlying_index_worlds=32,
                baseline_source_branches=192,
                gate_cells=12,
                baseline_batch_size=32,
                natural_source_block_size=96,
                natural_conditions=4,
                later_natural_continuations=768,
            ),
            interpretation="Development-only balanced competence qualification; all 12 family/count/query cells must pass 90% accuracy and 95% validity. At 16 assigned indices per cell this requires 15 correct and 16 valid. Current object vocabulary cannot support 120 fresh semantic confirmation worlds.",
        )
    return manifest


def partition(manifest, workers):
    """Assign whole worlds and original full cohorts to the same worker.

    Six branches per world and 32 branches per cohort coincide every 96 rows.
    Allocating these superblocks preserves both world grouping and the existing
    audit's parent-cohort provenance. The last superblock may be shorter.
    """
    if type(workers) is not int or workers not in (1, 2, 3, 4):
        raise ValueError("Require between one and four workers")
    size = manifest["execution"]["batch_size"]
    block = math.lcm(size, manifest["balance"]["branches_per_world"])
    cohorts = [[] for _ in range(workers)]
    n = len(manifest["diagnostics"])
    if not n or workers > math.ceil(n / block):
        raise ValueError("Require at least one complete-world source block per worker")
    for offset in range(0, n, block):
        worker = (offset // block) % workers
        cohorts[worker].extend(range(offset // size, math.ceil(min(offset + block, n) / size)))
    return cohorts


def prepare(
    out, items=100, seed=2026101411, template="table", start_index=0, workers=None, *, mode="validation"
):
    if mode not in ("validation", "development"):
        raise ValueError("Choose explicit validation or development mode")
    if type(items) is not int or (mode == "validation" and (items < 100 or items % 2)):
        raise ValueError(
            "Validation requires at least 100 world indices and an even count for property balance"
        )
    if mode == "development" and items != 16:
        raise ValueError("The fixed development count check requires exactly 16 world indices per family")
    workers = (2 if mode == "development" else 4) if workers is None else workers
    out = Path(out).resolve()
    if out.parent != ROOT / "runs":
        raise ValueError("Place the new run directly under the project's runs directory")
    if out.exists():
        raise ValueError("The requested run already exists")
    revision = read(ROOT / "configs/model_revisions.json")["Qwen/Qwen3-32B"]
    manifest = build_manifest(
        dict(model_id="Qwen/Qwen3-32B", revision=revision), items, seed, template, start_index, mode=mode
    )
    if task_module(manifest["diagnostics"][0]) is not competence:
        raise ValueError("Install the count-competence-v1 thinking-information dispatcher before freezing")
    groups = partition(manifest, workers)
    children = []
    for number, parent_cohorts in enumerate(groups):
        path = ROOT / "runs" / f"{out.name}-part-{number:02d}"
        if path.exists():
            raise ValueError(f"Child run already exists: {path.name}")
        child = copy.deepcopy(manifest)
        size = manifest["execution"]["batch_size"]
        child["diagnostics"] = [
            a for index in parent_cohorts for a in manifest["diagnostics"][index * size : (index + 1) * size]
        ]
        child["assignment_sha256"] = digest(child["diagnostics"])
        child["parent_manifest_sha256"] = digest(manifest)
        children.append((path, child, parent_cohorts))
    # All path and dispatch checks precede any writes.
    write_once(out / "manifest.json", manifest)
    descriptors = []
    for path, child, parent_cohorts in children:
        write_once(path / "manifest.json", child)
        descriptors.append(dict(path=str(path), manifest_sha256=digest(child), parent_cohorts=parent_cohorts))
    write_once(out / "children.json", descriptors)
    freeze(
        ROOT,
        out,
        [
            "reports/reasoning-information-v1-analysis-plan.md",
            "reports/reasoning-information-review-amendment.md",
            "reports/reasoning-information-development-scope-amendment.md",
            "reports/target-relevance-causal-prospective-plan.md",
            "reports/count-competence-semantic-capacity.md",
        ],
    )
    print(
        dict(
            run=str(out),
            branches=len(manifest["diagnostics"]),
            workers=workers,
            source=manifest["source_sha256"],
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("validation", "development"), default="validation")
    parser.add_argument("--items", type=int)
    parser.add_argument("--seed", type=int, default=2026101411)
    parser.add_argument("--template", choices=("table", "prose"), default="table")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4))
    args = parser.parse_args()
    prepare(
        args.out,
        args.items if args.items is not None else (16 if args.mode == "development" else 100),
        args.seed,
        args.template,
        args.start_index,
        args.workers,
        mode=args.mode,
    )
