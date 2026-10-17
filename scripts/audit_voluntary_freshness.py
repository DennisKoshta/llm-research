"""Assignment/input-only semantic exposure audit; never reads scientific outcomes.

Reconstructs immutable task inputs from source-hash-matched archives. The initial
local smoke lacks an archive, so only its recorded assignment/task input fields
are used. No answer, correctness, activation, probability, or preference is read.
"""

import argparse
import ast
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import tarfile
from types import SimpleNamespace

from curiosity_study import voluntary_resolution as voluntary
from curiosity_study import information_state
from curiosity_study import tasks as legacy_tasks
from curiosity_study.artifacts import digest


def byte_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def projected_worlds(family, world, *, truth_irrelevant=False):
    """Two-column exposure, not equivalence of the original complete task."""
    result = []
    for columns in itertools.combinations(range(len(world["attributes"])), 2):
        rows = [[row[i] for i in columns] for row in world["candidates"]]
        if len(rows) != 4 or len(set(map(tuple, rows))) != 4:
            continue
        projected = dict(
            attributes=[world["attributes"][i] for i in columns],
            candidates=rows,
            truth=world["truth"],
        )
        canonical = voluntary.canonical_world(family, projected)
        if truth_irrelevant:
            canonical.pop("truth")
        result.append(canonical)
    return result


def legacy_world(task):
    """Actual finite input table; machine input names retain the 0..6 domain."""
    candidates = task["candidates"]
    truth = next(i for i, c in enumerate(candidates) if c["label"] == task["truth"])
    if task["family"] == "objects":
        attributes = sorted(dict(candidates[0]["parameters"]))
        rows = [[dict(c["parameters"])[key] for key in attributes] for c in candidates]
    else:
        attributes = [f"output at input {i}" for i in range(7)]
        rows = [
            [str((c["parameters"][0] * i + c["parameters"][1]) % c["parameters"][2]) for i in range(7)]
            for c in candidates
        ]
    return dict(attributes=attributes, candidates=rows, truth=truth)


FINITE_REVISIONS = {
    "equal-uncertainty-v1": "information_state.py",
    "relational-information-v1": "relational_information.py",
    "target-relevant-information-v1": "target_relevant_information.py",
    "count-competence-v1": "count_competence.py",
    "optional-information-v1": "optional_information.py",
    "voluntary-resolution-v1": "voluntary_resolution.py",
    "voluntary-resolution-v2": "voluntary_resolution.py",
}
FINITE_ASSAYS = {
    "information_state", "thinking_information", "replay_information",
    "natural_information", "optional_information", "count_competence", "voluntary_resolution",
}
LEGACY_REVISIONS = {None, "one-credit-v2", "single-curiosity-v1", "single-curiosity-v2"}
LEGACY_ASSAYS = {None, "one_credit", "identification_factorial", "single_curiosity", "information_diagnostic"}
PURE_FUNCTIONS = {
    "task", "oracle", "_entropy", "_validate", "_check_choice", "_known_attribute",
    "canonical_world", "_fact", "_fact_text", "_event", "_remaining", "pattern_for",
}


def task_namespace(source, *, base=None):
    """Execute only literal constants and pure finite-world oracle definitions."""
    tree = ast.parse(source)
    namespace = dict(
        random=random, itertools=itertools, math=math, defaultdict=defaultdict, Counter=Counter,
        base=base, state=base, digest=digest,
    )
    for node in tree.body:
        if isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    namespace[target.id] = value
        elif isinstance(node, ast.FunctionDef) and node.name in PURE_FUNCTIONS:
            exec(
                compile(ast.Module(body=[node], type_ignores=[]), "<frozen-world-oracle>", "exec"), namespace
            )
    return SimpleNamespace(**namespace)


def revision_adapter(revision, assay):
    if revision in FINITE_REVISIONS and assay in FINITE_ASSAYS:
        return FINITE_REVISIONS[revision]
    if revision in LEGACY_REVISIONS and assay in LEGACY_ASSAYS:
        return "legacy"
    raise ValueError(f"Unknown finite-task revision/assay: {revision!r}/{assay!r}")


def frozen_finite_world(assignment, texts, modules=None):
    """Call the explicitly named frozen task, including its assignment validation."""
    filename = FINITE_REVISIONS[assignment["task_revision"]]
    modules = {} if modules is None else modules
    if "information_state.py" not in modules:
        modules["information_state.py"] = task_namespace(texts["information_state.py"])
    if filename not in modules:
        modules[filename] = task_namespace(texts[filename], base=modules["information_state.py"])
    return modules[filename].task(assignment)


def manifest_rows(manifest):
    if manifest.get("prefixes"):
        return [prefix["assignment"] for prefix in manifest["prefixes"].values()]
    return (manifest.get("assignments") or []) + (manifest.get("diagnostics") or [])


def plan_from_manifest(manifest):
    """Use a frozen target selection instead of importing mutable task defaults."""
    plan = dict(
        task_revision=manifest["task_revision"],
        selection=deepcopy(manifest["world_selection"]),
        selection_sha256=manifest["world_selection_sha256"],
        assignments=deepcopy(manifest_rows(manifest)),
        assignment_sha256=manifest["assignment_sha256"],
    )
    validate_plan(plan)
    return plan


def validate_plan(plan):
    if (
        digest(plan["selection"]) != plan["selection_sha256"]
        or digest(plan["assignments"]) != plan["assignment_sha256"]
        or type(plan["selection"]["seed"]) is not int
    ):
        raise ValueError("Target plan selection/assignment binding mismatch")


def source_archives(root, wanted, live_manifest_hashes):
    found, archived_manifests, unbound_archived = {}, [], []
    paths = sorted((root / "runs").glob("*/source.tar.gz")) + sorted((root / "archives").glob("*.tar.gz"))
    for path in paths:
        with tarfile.open(path) as archive:
            groups = defaultdict(dict)
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                if member.name.endswith(".py") and "src/curiosity_study/" in member.name:
                    prefix, suffix = member.name.split("src/curiosity_study/", 1)
                    groups[prefix][suffix] = archive.extractfile(member).read().decode()
                if member.name.endswith("/manifest.json"):
                    manifest = json.loads(archive.extractfile(member).read())
                    binding = dict(
                        archive=str(path.relative_to(root)),
                        member=member.name,
                        manifest_sha256=digest(manifest),
                    )
                    archived_manifests.append(binding)
                    if binding["manifest_sha256"] not in live_manifest_hashes:
                        unbound_archived.append(binding)
            for prefix, sources in groups.items():
                source_hash = digest(sources)
                if source_hash in wanted and source_hash not in found:
                    found[source_hash] = dict(
                        archive=str(path.relative_to(root)),
                        archive_byte_sha256=byte_sha(path),
                        member_prefix=prefix,
                        sources=sources,
                    )
    return found, archived_manifests, unbound_archived


def inventory(root):
    rows, manifests = [], {}
    for path in sorted((root / "runs").glob("*/manifest.json")):
        manifest = json.loads(path.read_text())
        records = sorted(path.parent.glob("attempts/*/*.json"))
        cohorts = sorted(path.parent.glob("cohorts/*/*.json"))
        special = (
            path.parent / "result.json"
        ).exists() and path.parent.name == "qwen32-single-reference-check-v2"
        paths = [str(p.relative_to(root)) for p in records + cohorts]
        assignments = (manifest.get("assignments") or []) + (manifest.get("diagnostics") or [])
        row = dict(
            run=str(path.parent.relative_to(root)),
            manifest_sha256=digest(manifest),
            manifest_byte_sha256=byte_sha(path),
            assignment_sha256=digest(assignments),
            source_assignment_sha256=digest(manifest_rows(manifest)),
            assigned_rows=len(assignments),
            source_sha256=manifest.get("source_sha256"),
            individual_record_files=len(records),
            cohort_record_files=len(cohorts),
            record_path_list_sha256=digest(paths),
            special_result_exists=special,
            execution_evidence="persisted_records"
            if paths
            else "reference_result_exists"
            if special
            else "none",
        )
        rows.append(row)
        manifests[row["run"]] = manifest
    by_run = {row["run"]: row for row in rows}
    for row in rows:
        children_path = root / row["run"] / "children.json"
        if row["execution_evidence"] == "none" and children_path.exists():
            children = json.loads(children_path.read_text())
            child_runs = [str(Path(child["path"]).relative_to(root)) for child in children]
            row["child_runs"] = child_runs
            row["children_byte_sha256"] = byte_sha(children_path)
            if any(by_run[child]["execution_evidence"] != "none" for child in child_runs):
                row["execution_evidence"] = "executed_child_records"
    return rows, manifests


def audit(root, *, pool_candidates=10000, plan=None, seed=None, exclude_runs=(), registry_snapshot=None):
    root = Path(root).resolve()
    if plan is None:
        plan = voluntary.assignment_plan(**({} if seed is None else {"seed": seed}))
    else:
        plan = deepcopy(plan)
    validate_plan(plan)
    if seed is not None and seed != plan["selection"]["seed"]:
        raise ValueError("Requested seed disagrees with supplied target plan")
    seed = plan["selection"]["seed"]
    current_registry, manifests = inventory(root)
    current_by_run = {row["run"]: row for row in current_registry}
    registry = current_registry
    if registry_snapshot is not None:
        registry = deepcopy(registry_snapshot)
        if len({row["run"] for row in registry}) != len(registry):
            raise ValueError("Duplicate historical snapshot entries")
        for row in registry:
            current = current_by_run.get(row["run"])
            if current is None or any(
                row[key] != current[key]
                for key in ("manifest_sha256", "manifest_byte_sha256", "assignment_sha256", "source_sha256")
            ):
                raise ValueError("Historical snapshot manifest/assignment binding changed")
            row["source_assignment_sha256"] = current["source_assignment_sha256"]
    excluded = set()
    for name in exclude_runs:
        path = Path(name)
        run = str((path if path.is_absolute() else root / path).resolve().relative_to(root))
        if run not in current_by_run or current_by_run[run]["assignment_sha256"] != plan["assignment_sha256"]:
            raise ValueError("Only the exact target collection may be excluded from its retrospective audit")
        excluded.add(run)
    archives, archived_manifests, unbound_archived = source_archives(
        root, {r["source_sha256"] for r in registry}, {r["manifest_sha256"] for r in current_registry}
    )
    if unbound_archived:
        raise ValueError("Archived manifests absent from live inventory require explicit inspection")
    exposures = {
        key: defaultdict(set)
        for key in ("exact", "relational_projection", "target_table", "legacy_projection")
    }
    world_counts, limitations, source_bindings, exact_canonicals = defaultdict(set), [], {}, {}
    validated_legacy_sources, frozen_modules = set(), {}
    legacy_ast = ast.dump(
        next(
            n
            for n in ast.parse(Path(legacy_tasks.__file__).read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "generate"
        ),
        include_attributes=False,
    )
    for entry in registry:
        run, manifest = entry["run"], manifests[entry["run"]]
        if run in excluded:
            entry["semantic_screen"] = "exact_target_collection_explicitly_excluded"
            continue
        if entry["execution_evidence"] == "none":
            entry["semantic_screen"] = "not_verified_executed"
            continue
        if manifest.get("assay") == "concept_capture":
            if {a.get("task_revision") for a in manifest_rows(manifest)} != {"short-text-concept-capture-v1"}:
                raise ValueError("Unknown finite-task revision/assay in concept-capture collection")
            entry["semantic_screen"] = "incomparable_plaintext_concept_corpus"
            continue
        oracle_source_sha = entry["source_sha256"]
        source_assay = manifest.get("assay")
        rows = manifest_rows(manifest)
        if source_assay == "sustained_engineering":
            from sustained_exposure import source_inputs

            adapted = source_inputs(root, run, manifest)
            sources, rows = adapted["sources"], adapted["assignments"]
            oracle_source_sha, source_assay = adapted["source_sha256"], adapted["assay"]
            entry["source_input_adapter"] = adapted["provenance"]
        else:
            sources = archives.get(oracle_source_sha)
        if sources is None:
            if run != "runs/local-smoke-v1":
                raise ValueError(f"Missing frozen task source for {run}")
            assigned = {
                a["trial_id"]: a
                for a in (manifest.get("assignments") or []) + (manifest.get("diagnostics") or [])
            }
            input_ids, task_bindings = set(), []
            for path in sorted((root / run).glob("attempts/*/*.json")):
                envelope = json.loads(path.read_text())
                # Deliberately access only immutable task/assignment input fields.
                record = envelope["record"]
                assignment, task = record["assignment"], record["task"]
                trial_id = record["trial_id"]
                if revision_adapter(assignment.get("task_revision"), manifest.get("assay")) != "legacy":
                    raise ValueError("Recorded-input fallback permits only explicitly known legacy tasks")
                if (
                    envelope["manifest_hash"] != entry["manifest_sha256"]
                    or envelope["sha256"] != digest(record)
                    or assignment != assigned[trial_id]
                ):
                    raise ValueError("Historical recorded input binding mismatch")
                input_ids.add(trial_id)
                task_bindings.append(
                    dict(
                        trial_id=trial_id,
                        task_sha256=digest(task),
                        assignment_sha256=digest(assignment),
                        record_file=str(path.relative_to(root)),
                        record_byte_sha256=byte_sha(path),
                    )
                )
                world = legacy_world(task)
                for canonical in projected_worlds(task["family"], world):
                    exposures["legacy_projection"][digest(canonical)].add(run)
                world_counts[("legacy", task["family"])].add(digest(world))
            if input_ids != set(assigned):
                raise ValueError("Historical input records do not cover all assigned trials")
            entry["semantic_screen"] = "all_recorded_task_inputs_no_source_archive"
            entry["recorded_task_input_bindings"] = task_bindings
            limitations.append(
                "Initial local-smoke-v1 source archive unavailable; all48 recorded task/assignment inputs cover all48 assigned trials and were screened directly, without reading answers or preferences."
            )
            continue
        if oracle_source_sha not in source_bindings:
            source_bindings[oracle_source_sha] = {k: v for k, v in sources.items() if k != "sources"}
        texts = sources["sources"]
        processed = set()
        for assignment in rows:
            family = assignment.get("family")
            if family not in voluntary.FAMILIES:
                raise ValueError(f"Unknown finite task assignment in {run}")
            revision = assignment.get("task_revision")
            adapter = revision_adapter(revision, source_assay)
            if adapter != "legacy":
                # Do not assume an adapter's world ignores any assignment factor.
                # Reconstruct each distinct complete assignment, then deduplicate
                # only the resulting semantic worlds in the exposure sets.
                key = digest(assignment)
                if key in processed:
                    continue
                processed.add(key)
                world = frozen_finite_world(
                    assignment, texts, frozen_modules.setdefault(oracle_source_sha, {})
                )
                if revision not in ("relational-information-v1", "target-relevant-information-v1"):
                    if (
                        len(world["attributes"]) != 2 or len(world["candidates"]) != 4
                        or len(set(map(tuple, world["candidates"]))) != 4
                    ):
                        raise ValueError("Exact finite-world adapter did not produce a two-property four-row world")
                    canonical = voluntary.canonical_world(family, world)
                    exposures["exact"][digest(canonical)].add(run)
                    exact_canonicals[digest(canonical)] = canonical
                    world_counts[(revision, family)].add(digest(canonical))
                else:
                    target = revision == "target-relevant-information-v1"
                    category = "target_table" if target else "relational_projection"
                    for canonical in projected_worlds(family, world, truth_irrelevant=target):
                        exposures[category][digest(canonical)].add(run)
                    semantic = dict(attributes=world["attributes"], candidates=world["candidates"])
                    if not target:
                        semantic["truth"] = world["truth"]
                    world_counts[(revision, family)].add(digest(semantic))
            else:
                if entry["source_sha256"] not in validated_legacy_sources:
                    generate = next(
                        n
                        for n in ast.parse(texts["tasks.py"]).body
                        if isinstance(n, ast.FunctionDef) and n.name == "generate"
                    )
                    if ast.dump(generate, include_attributes=False) != legacy_ast:
                        raise ValueError("Historical legacy generator differs; explicit adapter needed")
                    validated_legacy_sources.add(entry["source_sha256"])
                indices = assignment.get("item_indices", [assignment["index"]])
                legacy_seed = assignment["seed"] + (100000 if "item_indices" in assignment else 0)
                for index in indices:
                    key = (family, legacy_seed, index, assignment.get("split", "development"))
                    if key in processed:
                        continue
                    processed.add(key)
                    task = legacy_tasks.generate(
                        family, index, legacy_seed, assignment.get("split", "development")
                    ).as_dict()
                    world = legacy_world(task)
                    for canonical in projected_worlds(family, world):
                        exposures["legacy_projection"][digest(canonical)].add(run)
                    world_counts[("legacy", family)].add(digest(world))
        entry["semantic_screen"] = "frozen_source_reconstruction"
        entry["revision_adapters"] = {
            str(revision): revision_adapter(revision, source_assay)
            for revision in sorted({a.get("task_revision") for a in rows}, key=str)
        }
    selected = []
    for assignment in plan["selection"]["worlds"]:
        canonical = voluntary.canonical_world(assignment["family"], information_state.task(assignment))
        key = digest(canonical)
        if assignment.get("semantic_world_id") != key:
            raise ValueError("Target selected world disagrees with its named-table/truth identity")
        table = {k: v for k, v in canonical.items() if k != "truth"}
        selected.append(
            dict(
                family=assignment["family"],
                world_slot=assignment["world_slot"],
                candidate_index=assignment["index"],
                semantic_world_id=key,
                canonical_world=canonical,
                exact_complete_world_runs=sorted(exposures["exact"].get(key, set())),
                relational_projection_with_truth_runs=sorted(
                    exposures["relational_projection"].get(key, set())
                ),
                target_projected_table_runs=sorted(exposures["target_table"].get(digest(table), set())),
                legacy_projection_runs=sorted(exposures["legacy_projection"].get(key, set())),
            )
        )
    pool = []
    for family in voluntary.FAMILIES:
        unique, exact_unused, conservatively_unused = set(), set(), set()
        for index in range(pool_candidates):
            assignment = dict(
                family=family, seed=seed, index=index, vocabulary="heldout", known=[0], query=1
            )
            canonical = voluntary.canonical_world(family, information_state.task(assignment))
            key = digest(canonical)
            table = {k: v for k, v in canonical.items() if k != "truth"}
            unique.add(key)
            if key not in exposures["exact"] and key not in exposures["legacy_projection"]:
                exact_unused.add(key)
                if (
                    key not in exposures["relational_projection"]
                    and digest(table) not in exposures["target_table"]
                ):
                    conservatively_unused.add(key)
        pool.append(
            dict(
                family=family,
                candidate_indices=[0, pool_candidates],
                unique_complete_worlds=len(unique),
                exact_unused=len(exact_unused),
                unused_under_projected_exposure_rule=len(conservatively_unused),
                selected_worlds_retained=True,
            )
        )
    summaries = []
    for family in voluntary.FAMILIES:
        rows = [row for row in selected if row["family"] == family]
        summaries.append(
            dict(
                family=family,
                selected=len(rows),
                exact_complete_world_overlaps=sum(bool(row["exact_complete_world_runs"]) for row in rows),
                relational_projection_overlaps=sum(
                    bool(row["relational_projection_with_truth_runs"]) for row in rows
                ),
                target_projected_table_overlaps=sum(bool(row["target_projected_table_runs"]) for row in rows),
                any_projected_exposure=sum(
                    bool(
                        row["relational_projection_with_truth_runs"]
                        or row["target_projected_table_runs"]
                        or row["legacy_projection_runs"]
                    )
                    for row in rows
                ),
            )
        )
    exact_rows = [
        dict(semantic_world_id=key, canonical_world=exact_canonicals[key], source_runs=sorted(runs))
        for key, runs in sorted(exposures["exact"].items())
    ]
    return dict(
        schema_version=2,
        assay="voluntary-resolution-freshness",
        task_revision=plan["task_revision"],
        target_plan_sha256=digest(plan),
        target_seed=seed,
        excluded_target_collections=sorted(excluded),
        historical_snapshot_sha256=None if registry_snapshot is None else digest(registry_snapshot),
        selection_sha256=plan["selection_sha256"],
        assignment_sha256=plan["assignment_sha256"],
        source_file_sha256={
            str(Path(module.__file__).relative_to(root))
            if Path(module.__file__).is_relative_to(root)
            else str(Path(module.__file__)): byte_sha(module.__file__)
            for module in (voluntary, information_state)
        },
        audit_script_byte_sha256=byte_sha(__file__),
        historical_registry=registry,
        historical_registry_sha256=digest(registry),
        archive_manifest_aliases=archived_manifests,
        source_archive_bindings=source_bindings,
        exact_world_exclusions=exact_rows,
        exact_world_exclusions_sha256=digest(exact_rows),
        historical_exact_world_ids=sorted(exposures["exact"]),
        historical_exact_world_ids_sha256=digest(sorted(exposures["exact"])),
        historical_world_counts=[
            dict(task_revision=revision, family=family, unique_worlds=len(values))
            for (revision, family), values in sorted(world_counts.items())
        ],
        inclusion_rule="All assigned worlds in every collection with any persisted terminal record or a child with records; failed/partial collections retained without reading outcomes. Reference-check result existence included. Parent/child duplicates unioned semantically.",
        exact_rule="Same named two-property four-row candidate table and realized truth; ignores row/column order, delivery, wording, menu, question and sampling seed.",
        projection_rule="Weaker partial exposure: relational tasks projected to distinct four-row column pairs retain realized truth; target task projections ignore unused truth and compare tables. These three-property tasks are not exact complete-task repeats.",
        selected_worlds=selected,
        summary=summaries,
        candidate_pool=pool,
        exact_complete_world_overlap_count=sum(row["exact_complete_world_overlaps"] for row in summaries),
        exact_complete_world_criterion_passed=not any(
            row["exact_complete_world_overlaps"] for row in summaries
        ),
        independent_holdout_claim_supported=False,
        limitations=limitations
        + [
            "Local persisted history only; unexported remote, external/pretraining, or human semantic exposure is not covered.",
            "Isomorphic abstract four-row structures recur by design; matching preserves literal named properties/values and does not identify all renamings.",
            "Partial projected-table exposure is reported separately and is not removed by exact-world exclusions. This remains development transfer, not independent confirmation or novel-table generalization.",
            "Fresh calls have separate context and do not train model weights; prior study reuse still limits cross-study independence and investigator-heldout claims.",
            "Candidate-pool counts are descriptive only; no outcome-based exclusions or selection changes were made.",
        ],
    )


def history_receipt(result):
    """Selection-independent exact exclusions from the full current inventory."""
    if result["excluded_target_collections"] or result["historical_snapshot_sha256"] is not None:
        raise ValueError("New selection history must include all current exposure, without target exclusions or cutoff")
    keys = (
        "audit_script_byte_sha256", "source_file_sha256", "historical_registry",
        "historical_registry_sha256", "archive_manifest_aliases", "source_archive_bindings",
        "historical_exact_world_ids", "historical_exact_world_ids_sha256",
        "exact_world_exclusions", "exact_world_exclusions_sha256", "historical_world_counts",
        "inclusion_rule", "exact_rule", "projection_rule", "independent_holdout_claim_supported",
    )
    return dict(
        schema_version=1,
        assay="voluntary-resolution-historical-exclusions",
        selection_independent=True,
        **{key: deepcopy(result[key]) for key in keys},
        limitations=[
            "All assigned worlds in any evidenced collection are retained, including partial and failed runs; no scientific outcome fields determine exclusions.",
            "Exact exclusions concern named complete two-property tables plus realized truth; relational/target projected-table exposure remains a separate limitation.",
            "Local persisted history only; this does not establish independence from external or pretraining exposure.",
        ],
    )


def write_json_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pool-candidates", type=int, default=10000)
    targets = parser.add_mutually_exclusive_group()
    targets.add_argument("--target-plan", type=Path)
    targets.add_argument("--target-manifest", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--exclude-run", action="append", default=[])
    parser.add_argument("--registry-snapshot", type=Path)
    parser.add_argument("--history-output", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.target_plan.read_text()) if args.target_plan else None
    if args.target_manifest:
        plan = plan_from_manifest(json.loads(args.target_manifest.read_text()))
    snapshot = json.loads(args.registry_snapshot.read_text()) if args.registry_snapshot else None
    if isinstance(snapshot, dict):
        snapshot = snapshot["historical_registry"]
    result = audit(
        args.root, pool_candidates=args.pool_candidates, plan=plan, seed=args.seed,
        exclude_runs=args.exclude_run, registry_snapshot=snapshot,
    )
    history = history_receipt(result) if args.history_output else None
    write_json_once(args.output, result)
    if history is not None:
        write_json_once(args.history_output, history)
    print(
        json.dumps(
            dict(
                output=str(args.output),
                exact_criterion_passed=result["exact_complete_world_criterion_passed"],
                summary=result["summary"],
            )
        )
    )


if __name__ == "__main__":
    main()
