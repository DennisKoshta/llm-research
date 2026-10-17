"""Freeze a reviewable intervention schedule and exact fixed-prefix assignments."""

import argparse
import copy
import hashlib
from pathlib import Path
import random
import sys
import numpy as np
from curiosity_study.artifacts import read, write_once, digest, RunStore
from curiosity_study.replay_information import make_prefix, engine_hashes
from curiosity_study.runner import source_hash
from curiosity_study.thinking_information import task_module
from curiosity_study.semantic_worlds import canonical_target_world, TARGET_SEMANTIC_RULE

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "infra"))
from freeze_execution import freeze  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from source_admission import admit_source  # noqa: E402


def world_id(assignment):
    """A world remains one unit across questions, cells, and prompt templates."""
    fields = ("task_revision", "seed", "family", "index")
    if any(field not in assignment for field in fields):
        raise ValueError("World identity requires task revision, seed, family, and index")
    return digest({field: assignment[field] for field in fields})


def world_sets(assignments, manifest):
    identities, semantic = set(), set()
    for a in assignments:
        if a.get("task_revision") not in (
            "equal-uncertainty-v1",
            "relational-information-v1",
            "count-competence-v1",
            "optional-information-v1",
            "target-relevant-information-v1",
        ):
            raise ValueError("Unknown task revision cannot establish semantic nonoverlap")
        module = task_module(a)
        module_hash = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        if module_hash != manifest.get("task_module_sha256"):
            raise ValueError("Task module hash must match before deriving semantic world identities")
        identities.add(world_id(a))
        if a["task_revision"] == "target-relevant-information-v1":
            tables = [module.task({**a, "independent_report": independent}) for independent in (0, 1)]
            semantic.add(digest(canonical_target_world(a["family"], tables)))
            continue
        task = module.task(a)
        order = sorted(range(len(task["attributes"])), key=lambda i: task["attributes"][i])
        rows = [[row[i] for i in order] for row in task["candidates"]]
        semantic.add(
            digest(
                dict(
                    family=a["family"],
                    attributes=[task["attributes"][i] for i in order],
                    candidates=sorted(rows),
                    truth=rows[task["truth"]],
                )
            )
        )
    return identities, semantic


def inspected_development_registry(config, source_manifest, selected):
    """Bind every declared inspected collection, regardless of its success or completion.

    Registry membership is a prospective researcher declaration; no outcome or
    passing-gate filter can remove failed pilots from the exclusion set.
    """
    key = "inspected_development_collections"
    if key not in config:
        if config.get("inference_scope") == "confirmation":
            raise ValueError("Confirmation requires explicit inspected_development_collections (possibly [])")
        return None
    declaration = config[key]
    if not isinstance(declaration, list):
        raise ValueError("inspected_development_collections must be an explicit list")
    source_rows = source_manifest.get("diagnostics")
    if not source_rows or digest(source_rows) != source_manifest.get("assignment_sha256"):
        raise ValueError("Invalid evaluation assignment binding for inspected-development registry")
    by_id = {a["trial_id"]: a for a in source_rows}
    if len(by_id) != len(source_rows) or len({a["trial_id"] for a in selected}) != len(selected):
        raise ValueError("Duplicate evaluation assignments in inspected-development registry")
    if not selected or any(by_id.get(a["trial_id"]) != a for a in selected):
        raise ValueError("Registry evaluation selection must bind unchanged parent assignments")
    evaluation, evaluation_semantic = world_sets(selected, source_manifest)
    paths, hashes, entries = set(), set(), []
    for entry in declaration:
        if not isinstance(entry, dict) or set(entry) != {"path", "manifest_sha256", "role"}:
            raise ValueError("Each inspected collection requires exactly path, manifest_sha256, and role")
        if any(not isinstance(entry[k], str) or not entry[k].strip() for k in entry):
            raise ValueError("Inspected collection registry fields must be nonempty strings")
        path = Path(entry["path"])
        if not path.is_absolute():
            raise ValueError("Inspected collection paths must be absolute")
        path = path.resolve()
        if str(path) in paths or entry["manifest_sha256"] in hashes:
            raise ValueError("Duplicate inspected collection registry entry")
        paths.add(str(path))
        hashes.add(entry["manifest_sha256"])
        manifest = read(path / "manifest.json")
        if digest(manifest) != entry["manifest_sha256"]:
            raise ValueError("Inspected collection manifest hash mismatch")
        rows = manifest.get("diagnostics")
        if not rows or digest(rows) != manifest.get("assignment_sha256"):
            raise ValueError("Invalid inspected collection assignment binding")
        if len({a["trial_id"] for a in rows}) != len(rows):
            raise ValueError("Duplicate inspected collection assignments")
        worlds, semantics = world_sets(rows, manifest)
        entries.append(
            dict(
                path=str(path),
                role=entry["role"],
                manifest_sha256=entry["manifest_sha256"],
                assignment_sha256=manifest["assignment_sha256"],
                task_module_sha256=manifest["task_module_sha256"],
                assigned_sources=len(rows),
                world_ids=sorted(worlds),
                semantic_world_ids=sorted(semantics),
                overlap_world_ids=sorted(evaluation & worlds),
                overlap_semantic_world_ids=sorted(evaluation_semantic & semantics),
            )
        )
    return dict(
        schema_version=1,
        declaration_sha256=digest(declaration),
        entries=entries,
        entries_sha256=digest(entries),
        evaluation_manifest_sha256=digest(source_manifest),
        evaluation_assignments_sha256=digest(selected),
        evaluation_world_ids=sorted(evaluation),
        evaluation_semantic_world_ids=sorted(evaluation_semantic),
        all_assigned_worlds_before_prefix_exclusions=True,
        disjoint=all(not e["overlap_world_ids"] and not e["overlap_semantic_world_ids"] for e in entries),
    )


def audit_inspected_development_registry(manifest):
    """Recompute the complete frozen registry using assignments only; fail closed.

    The schedule's declaration, rather than the provenance's own entry list,
    determines coverage, so an entry cannot vanish by merely rehashing provenance.
    Legacy development collections without a registry remain readable.
    """
    scope = manifest.get("inference_scope")
    provenance = manifest.get("split_provenance")
    if not isinstance(provenance, dict):
        return False
    config = manifest.get("intervention_schedule")
    key = "inspected_development_collections"
    declared = isinstance(config, dict) and key in config
    recorded = key in provenance
    if not declared and not recorded:
        return scope == "development"
    if not declared or not recorded:
        return False
    try:
        if (
            digest(config) != manifest.get("intervention_schedule_sha256")
            or config.get("inference_scope") != scope
        ):
            return False
        source = read(Path(config["source_collection"]) / "manifest.json")
        if digest(source) != manifest.get("source_parent_manifest_sha256"):
            return False
        prefixes, excluded = manifest["prefixes"], manifest["excluded_sources"]
        if len(set(excluded)) != len(excluded) or set(prefixes) & set(excluded):
            return False
        wanted = set(prefixes) | set(excluded)
        selected = [a for a in source["diagnostics"] if a["trial_id"] in wanted]
        if len(wanted) != manifest["assigned_sources"] or len(selected) != len(wanted):
            return False
        if digest(selected) != provenance.get("evaluation_assignments_sha256"):
            return False
        if any(prefixes[a["trial_id"]]["assignment"] != a for a in selected if a["trial_id"] in prefixes):
            return False
        expected = inspected_development_registry(config, source, selected)
        if provenance[key] != expected:
            return False
        return scope == "development" or expected["disjoint"] is True
    except (OSError, KeyError, TypeError, ValueError):
        return False


def direction_assignment_splits(extraction_manifest, direction_analysis):
    """Recover an explicit historical split or verify a frozen range-derived ledger."""
    rows = extraction_manifest["diagnostics"]
    if len({a["trial_id"] for a in rows}) != len(rows):
        raise ValueError("Extraction assignments must have distinct trial IDs")
    binding = direction_analysis.get("assignment_split")
    if binding is None:
        return [a.get("representation_split") for a in rows]
    fields = {
        "schema_version",
        "rule",
        "manifest_sha256",
        "development_split_sha256",
        "assignments",
        "assignment_sha256",
    }
    if not isinstance(binding, dict) or set(binding) != fields:
        raise ValueError("Malformed direction assignment-split binding")
    declared = extraction_manifest.get("development_split")
    if (
        binding["schema_version"] != 1
        or binding["rule"] != "manifest_development_split_v1"
        or binding["manifest_sha256"] != digest(extraction_manifest)
        or not isinstance(declared, dict)
        or set(declared) != {"calibration_indices", "check_indices"}
        or binding["development_split_sha256"] != digest(declared)
    ):
        raise ValueError("Direction split must bind the source's frozen development ranges")
    ranges = [declared["calibration_indices"], declared["check_indices"]]
    if any(
        not isinstance(bounds, list)
        or len(bounds) != 2
        or any(type(value) is not int for value in bounds)
        or not 0 <= bounds[0] < bounds[1]
        for bounds in ranges
    ) or max(bounds[0] for bounds in ranges) < min(bounds[1] for bounds in ranges):
        raise ValueError("Development ranges must be valid disjoint half-open intervals")
    ledger = []
    for a in rows:
        if type(a.get("index")) is not int:
            raise ValueError("Frozen split assignment requires an integer world index")
        matches = [
            split
            for split, (low, high) in zip(("extraction", "development"), ranges)
            if low <= a["index"] < high
        ]
        if len(matches) != 1:
            raise ValueError("Every extraction assignment must belong to exactly one frozen split")
        if "representation_split" in a and a["representation_split"] != matches[0]:
            raise ValueError("Assignment split conflicts with its frozen range")
        ledger.append(dict(trial_id=a["trial_id"], representation_split=matches[0]))
    if binding["assignments"] != ledger or binding["assignment_sha256"] != digest(ledger):
        raise ValueError("Direction split ledger must cover every unchanged assignment in manifest order")
    return [a["representation_split"] for a in ledger]


def split_provenance(config, source_manifest, direction_analysis, selected):
    """Bind the split to the analyzed manifest, using assignments only, never outcomes."""
    if direction_analysis.get("direction_schema") == "target_candidate_controls_v1":
        if not all(
            direction_analysis.get(key) is True
            for key in (
                "passed",
                "candidate_eligible",
                "controls_available",
            )
        ):
            raise ValueError("Target candidate and control availability must pass their prospective gates")
        if not isinstance(direction_analysis.get("assignment_split"), dict):
            raise ValueError("Target direction artifact requires its full frozen assignment-split ledger")
    scope = config.get("inference_scope")
    if scope not in ("development", "confirmation"):
        raise ValueError("Set inference_scope explicitly to development or confirmation")
    if not selected:
        raise ValueError("The configured source selection is empty")
    registry = inspected_development_registry(config, source_manifest, selected)
    extraction_sha = direction_analysis["manifest_sha256"]
    provenance = dict(
        schema_version=1,
        inference_scope=scope,
        direction_analysis_sha256=digest(direction_analysis),
        direction_extraction_manifest_sha256=extraction_sha,
        evaluation_manifest_sha256=digest(source_manifest),
        evaluation_assignments_sha256=digest(selected),
        assigned_sources=len(selected),
        identity_rule="task_revision, seed, family, index; all cells/questions/templates kept together",
        semantic_rule="family, named attributes, unordered candidate rows, and true candidate; ignores display order",
        extraction_binding_verified=False,
        overlaps=None,
        confirmation_eligible=False,
    )
    if any(a.get("task_revision") == "target-relevant-information-v1" for a in selected):
        provenance["semantic_rule"] = TARGET_SEMANTIC_RULE
    if registry is not None:
        provenance["inspected_development_collections"] = registry
        if scope == "confirmation" and not registry["disjoint"]:
            raise ValueError("Confirmation overlaps an inspected development collection")
    location = config.get("direction_extraction_collection")
    if location:
        extraction_manifest = read(Path(location) / "manifest.json")
    elif digest(source_manifest) == extraction_sha:
        extraction_manifest = source_manifest
    elif (
        scope == "development"
        and direction_analysis.get("direction_schema") != "target_candidate_controls_v1"
    ):
        provenance["status"] = "development_only; extraction collection not supplied; nonoverlap unverified"
        return provenance
    else:
        raise ValueError("Confirmation requires the bound direction_extraction_collection")
    if digest(extraction_manifest) != extraction_sha:
        raise ValueError("Extraction collection does not match the direction analysis manifest")
    if direction_analysis.get("development_only") is not True:
        raise ValueError("Unsupported direction analysis split metadata")
    extraction_rows = extraction_manifest["diagnostics"]
    if not extraction_rows or digest(extraction_rows) != extraction_manifest["assignment_sha256"]:
        raise ValueError("Invalid extraction assignment binding")
    by_split = {"extraction": [], "development": []}
    world_splits = {}
    for a, split in zip(
        extraction_rows, direction_assignment_splits(extraction_manifest, direction_analysis)
    ):
        if split not in by_split:
            raise ValueError("Extraction manifest requires explicit representation_split on every assignment")
        key = world_id(a)
        if key in world_splits and world_splits[key] != split:
            raise ValueError("A world cannot cross extraction and development splits")
        world_splits[key] = split
        by_split[split].append(a)
    if not all(by_split.values()):
        raise ValueError("Both extraction and representation-development assignments must be identified")
    evaluation, evaluation_semantic = world_sets(selected, source_manifest)
    overlaps = {}
    for split, rows in by_split.items():
        original, original_semantic = world_sets(rows, extraction_manifest)
        overlaps[split] = dict(
            reference_worlds=len(original),
            reference_semantic_worlds=len(original_semantic),
            world_ids=sorted(evaluation & original),
            semantic_world_ids=sorted(evaluation_semantic & original_semantic),
        )
    eligible = all(not value["world_ids"] and not value["semantic_world_ids"] for value in overlaps.values())
    eligible = eligible and (registry is None or registry["disjoint"])
    provenance.update(
        extraction_binding_verified=True,
        evaluation_world_ids=sorted(evaluation),
        evaluation_semantic_world_ids=sorted(evaluation_semantic),
        overlaps=overlaps,
        confirmation_eligible=eligible,
        status="bound assignments; development reuse allowed"
        if scope == "development"
        else "fresh confirmation",
    )
    if direction_analysis.get("assignment_split") is not None:
        provenance["direction_assignment_split_sha256"] = digest(direction_analysis["assignment_split"])
    if scope == "confirmation" and not eligible:
        raise ValueError("Confirmation overlaps extraction or representation-development worlds")
    return provenance


def prepare(config_path):
    config = read(config_path)
    source = Path(config["source_collection"])
    m = read(source / "manifest.json")
    source_audit = read(config["source_audit"])
    parity = read(config["parity_audit"])
    assert parity["passed"]
    assert parity["model"] == m["model"] and parity["engine_sha256"] == engine_hashes()
    directions = Path(config["directions_local"])
    direction_sha = hashlib.sha256(directions.read_bytes()).hexdigest()
    assert direction_sha == config["directions_sha256"]
    direction_analysis = read(directions.parent / "analysis.json")
    assert (
        direction_analysis["directions_sha256"] == direction_sha and direction_analysis["model"] == m["model"]
    )
    with np.load(directions, allow_pickle=False) as z:
        fitted = {k: z[k] for k in z.files}
    source_names = list(fitted["sources"])
    layers, positions = list(fitted["layers"]), list(fitted["positions"])
    gain_source = source_names.index(config["dose_reference_source"])
    low, high = config["world_index_range"]
    selected_sources = [
        a for a in m["diagnostics"] if low <= a["index"] < high and a["kind"] in config["kinds"]
    ]
    admission = admit_source(config, m, source_audit, selected_sources)
    provenance = split_provenance(config, m, direction_analysis, selected_sources)
    selected_ids = {a["trial_id"] for a in selected_sources}
    source_assignments = {a["trial_id"]: a for a in selected_sources}
    if len(selected_ids) != len(selected_sources):
        raise ValueError("Duplicate selected source assignment")
    prefixes, excluded = {}, []
    children = (
        read(source / "children.json")
        if (source / "children.json").exists()
        else [dict(path=str(source), manifest_sha256=digest(m))]
    )
    assigned_sources = 0
    seen = set()
    for child in children:
        path = Path(child["path"])
        cm = read(path / "manifest.json")
        assert digest(cm) == child["manifest_sha256"]
        rows = RunStore(path, cm).completed()
        for a in cm["diagnostics"]:
            if a["trial_id"] not in selected_ids:
                continue
            if a != source_assignments[a["trial_id"]] or a["trial_id"] in seen:
                raise ValueError("Child source assignments disagree with the bound parent selection")
            seen.add(a["trial_id"])
            assigned_sources += 1
            prefix = make_prefix(rows[a["trial_id"]], path.name, digest(cm))
            if prefix is None:
                excluded.append(a["trial_id"])
            else:
                prefixes[a["trial_id"]] = prefix
    if seen != selected_ids:
        raise ValueError("Incomplete source partition coverage")
    size = config["batch_size"]
    assert size in (4, 8, 16)
    assignments = []
    prefix_keys = sorted(prefixes)
    random.Random(config["order_seed"]).shuffle(prefix_keys)
    for key in prefix_keys:
        prefix = prefixes[key]
        original_kind = prefix["assignment"]["kind"]

        def row(label, **kwargs):
            return dict(
                trial_id=f"{config['name']}-{key}-{label}",
                prefix_id=key,
                kind=original_kind,
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
                target = "final" if position == "decision" else position
                norm = float(fitted["contrast_norm"][gain_source, li, pi])
                assert np.isfinite(norm) and norm > 0
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
                    si = source_names.index(direction["source"])
                    vector = fitted[direction["array"]][si, li, pi]
                    assert np.isfinite(vector).all() and np.linalg.norm(vector) > 0
                    for multiplier in config["dose_multipliers"]:
                        local.append(
                            row(
                                f"{tag}-{direction['name']}-{multiplier}",
                                arm="direction",
                                layer=layer,
                                position=target,
                                capture_position=position,
                                dose=multiplier * norm,
                                dose_multiplier=multiplier,
                                direction_name=direction["name"],
                                source_index=si,
                                layer_index=li,
                                position_index=pi,
                                array=direction["array"],
                            )
                        )
                for seed in config["random_seeds"]:
                    for multiplier in config["random_dose_multipliers"]:
                        local.append(
                            row(
                                f"{tag}-random{seed}-{multiplier}",
                                arm="random",
                                layer=layer,
                                position=target,
                                capture_position=position,
                                dose=multiplier * norm,
                                dose_multiplier=multiplier,
                                direction_name=f"random{seed}",
                                random_seed=seed,
                            )
                        )
        # Keep every batch at a constant shape within a prefix. Extra zero-dose
        # replicas are engineering padding controls, not additional scientific units.
        while len(local) % size:
            local.append(
                row(
                    f"padding{len(local)}",
                    arm="sham",
                    layer=config["layers"][0],
                    position="final",
                    capture_position="decision",
                    dose=0.0,
                    dose_multiplier=0.0,
                    direction_name="padding_control",
                )
            )
        assignments.extend(local)
    parent = ROOT / "runs" / config["name"]
    manifest = dict(
        schema_version=3,
        assay="replay_information",
        phase=config["phase"],
        inference_scope=config["inference_scope"],
        split_provenance=provenance,
        split_provenance_sha256=digest(provenance),
        model=m["model"],
        execution_gpu="H200",
        execution=dict(backend="torch_batched_v1", batch_size=size),
        source_sha256=source_hash(),
        engine_sha256=engine_hashes(),
        assignments=[],
        diagnostics=assignments,
        assignment_sha256=digest(assignments),
        prefixes=prefixes,
        prefixes_sha256=digest(prefixes),
        assigned_sources=assigned_sources,
        excluded_sources=excluded,
        source_parent_manifest_sha256=digest(m),
        source_admission=admission,
        source_admission_sha256=digest(admission),
        capture_layers=[],
        max_new_tokens=8,
        greedy=False,
        direction_input=dict(path=config["directions_remote"], sha256=direction_sha),
        direction_analysis_sha256=digest(direction_analysis),
        direction_extraction_manifest_sha256=direction_analysis["manifest_sha256"],
        intervention_schedule=config,
        intervention_schedule_sha256=digest(config),
        parity_audit_sha256=digest(parity),
        invalid_output_policy="terminal_no_retry",
        padded_prefix_groups=True,
    )
    write_once(parent / "manifest.json", manifest)
    write_once(parent / "schedule.json", config)
    children = []
    cohorts = [assignments[i : i + size] for i in range(0, len(assignments), size)]
    for index in range(config["workers"]):
        child = copy.deepcopy(manifest)
        selected = list(range(index, len(cohorts), config["workers"]))
        rows = [a for j in selected for a in cohorts[j]]
        child.update(
            diagnostics=rows,
            assignment_sha256=digest(rows),
            parent_manifest_sha256=digest(manifest),
            partition_index=index,
            parent_cohort_indices=selected,
        )
        path = ROOT / "runs" / f"{config['name']}-part-{index:02d}"
        write_once(path / "manifest.json", child)
        children.append(
            dict(path=str(path), manifest_sha256=digest(child), branches=len(rows), parent_cohorts=selected)
        )
    write_once(parent / "children.json", children)
    freeze(
        ROOT,
        parent,
        [
            "reports/reasoning-information-v1-analysis-plan.md",
            "reports/reasoning-information-review-amendment.md",
            "reports/reasoning-information-development-scope-amendment.md",
        ],
    )
    print(
        dict(
            prefixes=len(prefixes),
            excluded=len(excluded),
            measurements=len(assignments),
            workers=config["workers"],
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args()
    prepare(args.config)
