"""Input-only inventory of the complete remaining heldout-object task space.

No response, activation, correctness, preference, or scientific audit outcome
selects a world. The unchanged historical auditor is replayed at its snapshot;
new S/C outer assignments are separately bound to their frozen v2 task inputs.
This writes metadata only, never an execution freeze or scientific admission.
"""

import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import tarfile

from curiosity_study import information_state as state
from curiosity_study import reserved_object_materials as materials
from curiosity_study import voluntary_resolution as voluntary
from curiosity_study.artifacts import digest, read, write_once

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_voluntary_freshness as freshness  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HISTORY = "reports/satisfaction-screen-v1-history.json"
HISTORY_BYTES = "0e96b243c2dfa9300b5ca8d168b5f7f32f93a53796c72e41e047f57c4ab4bfae"
S_PARENT = "runs/qwen32-satisfaction-screen-v1"
C_PARENT = "runs/qwen32-curiosity-suppression-screen-v1-admission2"
PARENT_PINS = {
    S_PARENT: "f74f1eb522487e4dea61913d41285f47d95a352f8fb88fe629c1ba277e92d528",
    C_PARENT: "535f3c1237a8b876fa1059d3f9949f178a55b5627e2be412008ef08cbc90b02f",
}
OUTER_ASSAYS = {"satisfaction_screen", "curiosity_suppression_screen"}
HALF_SEEDS, SOURCE_SLOTS = materials.HALF_SEEDS, materials.SOURCE_SLOTS
ORDER_SEED, REVISION = materials.ORDER_SEED, materials.REVISION
canonical_pool, first_indices = materials.canonical_pool, materials.first_indices


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local_path(root, name):
    path = Path(name)
    path = (path if path.is_absolute() else root / path).resolve()
    require(path.is_relative_to(root), "Input path escapes the project")
    return path


def pool_plan(*, max_candidates=10000):
    first, _ = first_indices(HALF_SEEDS[0], max_candidates=max_candidates)
    worlds = [
        dict(family="objects", seed=HALF_SEEDS[0], index=first[key], vocabulary="heldout",
             known=[0], query=1, world_slot=0, semantic_world_id=key)
        for key in canonical_pool()
    ]
    selection = dict(seed=HALF_SEEDS[0], worlds=worlds, purpose="complete_pool_exposure_query_no_selection")
    return dict(task_revision=REVISION, assignments=[], assignment_sha256=digest([]),
                selection=selection, selection_sha256=digest(selection))


def frozen_sources(path, *, prefix=""):
    start = prefix + "src/curiosity_study/"
    with tarfile.open(path) as archive:
        members = [m for m in archive.getmembers() if m.isfile() and m.name.startswith(start)
                   and m.name.endswith(".py")]
        require(len({m.name for m in members}) == len(members), "Duplicate frozen source member")
        return {m.name[len(start):]: archive.extractfile(m).read().decode() for m in members}


def verify_old_snapshot(root, history_path, query):
    path = local_path(root, history_path)
    require(sha(path) == HISTORY_BYTES, "Historical receipt differs from the fixed S-screen binding")
    history = read(path)
    require(
        history["selection_independent"] is True
        and digest(history["historical_registry"]) == history["historical_registry_sha256"]
        and digest(history["exact_world_exclusions"]) == history["exact_world_exclusions_sha256"]
        and digest(history["historical_exact_world_ids"]) == history["historical_exact_world_ids_sha256"]
        and history["historical_exact_world_ids"] == sorted(
            row["semantic_world_id"] for row in history["exact_world_exclusions"]
        ), "Malformed historical snapshot/exclusion binding",
    )
    require(sha(freshness.__file__) == history["audit_script_byte_sha256"], "Historical helper changed")
    for name, expected in history["source_file_sha256"].items():
        # Synthetic roots retain the real immutable oracle module locations.
        path = Path(name) if Path(name).is_absolute() else root / name
        require(sha(path) == expected, "Historical canonical task source changed")
    for source_sha, binding in history["source_archive_bindings"].items():
        path = local_path(root, binding["archive"])
        require(sha(path) == binding["archive_byte_sha256"]
                and digest(frozen_sources(path, prefix=binding["member_prefix"])) == source_sha,
                "Historical source archive binding changed")
    for binding in history["archive_manifest_aliases"]:
        with tarfile.open(local_path(root, binding["archive"])) as archive:
            import json

            require(digest(json.loads(archive.extractfile(binding["member"]).read()))
                    == binding["manifest_sha256"], "Historical archived manifest alias changed")
    replay = freshness.audit(root, plan=query, pool_candidates=0,
                             registry_snapshot=history["historical_registry"])
    for key in ("historical_exact_world_ids", "exact_world_exclusions", "historical_world_counts"):
        require(replay[key] == history[key], "Historical exposure snapshot did not reproduce")
    # A newer archive may contain an identical old source copy. Original archive
    # bytes were checked above; equivalent alias choice is not a changed oracle.
    require(set(replay["source_archive_bindings"]) == set(history["source_archive_bindings"]),
            "Historical oracle-source coverage changed")
    return history, dict(path=str(local_path(root, history_path).relative_to(root)),
                         byte_sha256=HISTORY_BYTES, content_sha256=digest(history),
                         registry_sha256=history["historical_registry_sha256"], reproduced=True)


def screen_inputs(root, run, manifest, manifests):
    """Reconstruct all assigned inputs, never open generation/receipt output files."""
    require(run in PARENT_PINS and digest(manifest) == PARENT_PINS[run], "Unrecognized frozen S/C parent")
    directory = root / run
    frozen = read(directory / "freeze.json")
    archive_path = directory / "source.tar.gz"
    sources = frozen_sources(archive_path)
    require(frozen["manifest_sha256"] == digest(manifest)
            and frozen["source_sha256"] == manifest["source_sha256"] == digest(sources)
            and frozen["archive_sha256"] == sha(archive_path), "S/C frozen package differs")
    for module in (voluntary, state):
        require(sources[Path(module.__file__).name] == Path(module.__file__).read_text(),
                "Current v2 oracle/request renderer differs from frozen S/C source")
    plan, rows = manifest["screen_plan"], freshness.manifest_rows(manifest)
    base, preflight = plan["base_assignments"], manifest["generation_preflight"]
    require(
        digest(plan) == manifest["screen_plan_sha256"]
        and digest(rows) == manifest["assignment_sha256"] == plan["assignment_sha256"]
        and rows == plan["assignments"] and digest(base) == plan["base_assignment_sha256"]
        and len(base) == len({b["base_trial_id"] for b in base}) == 288
        and len({b["semantic_world_id"] for b in base}) == 24
        and digest(preflight) == manifest["generation_preflight_sha256"]
        and preflight["base_assignment_sha256"] == digest(base)
        and preflight["assignment_sha256"] == digest(rows)
        and digest(preflight["requests"]) == preflight["requests_sha256"]
        and digest(preflight["outer_request_bindings"]) == preflight["outer_request_bindings_sha256"],
        "S/C complete source, assignment, or token-request inventory differs",
    )
    require(len(preflight["requests"]) == 288 and len(preflight["outer_request_bindings"]) == len(rows),
            "S/C request inventory omitted an assigned row")
    by_base, canonicals = {}, {}
    for branch, bound in zip(base, preflight["requests"], strict=True):
        source = branch["source_assignment"]
        key = branch["base_trial_id"]
        require(source["trial_id"] == key == bound["base_trial_id"]
                and digest(source) == branch["source_assignment_sha256"] == bound["source_assignment_sha256"]
                and digest(voluntary.inference_request(source)) == branch["inference_request_sha256"]
                == bound["inference_request_sha256"], "Nested source assignment/request differs")
        request = bound["request"]
        require(request["source_trial_id"] == key and request["seed"] == source["sampling_seed"]
                and digest(request["token_prefix"]) == request["token_prefix_sha256"]
                and all(branch[k] == source[k] for k in
                        ("kind", "delivery", "menu_layout", "family", "semantic_world_id", "sampling_seed")),
                "Outer source metadata or original token request differs")
        canonical = voluntary.canonical_world(source["family"], voluntary.task(source))
        require(digest(canonical) == source["semantic_world_id"], "Assigned source world binding differs")
        by_base[key] = branch
        canonicals[digest(canonical)] = canonical
    by_id = {row["trial_id"]: row for row in rows}
    require(len(by_id) == len(rows), "Duplicate outer assigned trial")
    request_by_base = {r["base_trial_id"]: r["request"] for r in preflight["requests"]}
    for row, binding in zip(rows, preflight["outer_request_bindings"], strict=True):
        source = by_base[row["base_trial_id"]]
        require(all(row[key] == value for key, value in source.items())
                and binding == dict(trial_id=row["trial_id"], base_trial_id=row["base_trial_id"],
                                    source_assignment_sha256=row["source_assignment_sha256"],
                                    sampling_seed=row["sampling_seed"],
                                    request_sha256=digest(request_by_base[row["base_trial_id"]])),
                "Outer S/C assignment is not the unchanged bound source input")
    children = read(directory / "children.json")
    require(len(children) == 3 and len({c["path"] for c in children}) == 3, "Require all three distinct S/C children")
    seen, child_bindings = set(), []
    for descriptor in children:
        child_path = local_path(root, descriptor["path"])
        name = str(child_path.relative_to(root))
        require(name in manifests, "S/C child is missing from complete current inventory")
        child = manifests[name]
        assigned = freshness.manifest_rows(child)
        expected = deepcopy(manifest)
        expected.update(execution_group=descriptor["execution_group"], parent_manifest_sha256=digest(manifest),
                        diagnostics=assigned, assignment_sha256=digest(assigned))
        require(child == expected and digest(child) == descriptor["manifest_sha256"], "S/C child binding differs")
        ids = [a["trial_id"] for a in assigned]
        id_set = set(ids)
        require(len(ids) == len(id_set) and not seen.intersection(ids)
                and all(a == by_id.get(a["trial_id"]) for a in assigned)
                and assigned == [a for a in rows if a["trial_id"] in id_set],
                "Duplicate, changed, or reordered S/C child assignment")
        parent_cohorts = {tuple(a["trial_id"] for a in rows[i:i + 32]) for i in range(0, len(rows), 32)}
        require(len(assigned) % 32 == 0 and all(
            tuple(a["trial_id"] for a in assigned[i:i + 32]) in parent_cohorts
            for i in range(0, len(assigned), 32)), "S/C child split an original complete cohort")
        seen.update(ids)
        child_bindings.append(dict(run=name, manifest_sha256=digest(child), assignment_sha256=digest(assigned)))
    require(seen == set(by_id), "S/C children do not cover every parent assignment")
    return dict(canonicals=canonicals, base=base, requests=preflight["requests"], children=child_bindings,
                binding=dict(run=run, manifest_sha256=digest(manifest), manifest_byte_sha256=sha(directory / "manifest.json"),
                             freeze_sha256=digest(frozen), archive_byte_sha256=sha(archive_path),
                             source_sha256=digest(sources), base_assignment_sha256=digest(base),
                             requests_sha256=digest(preflight["requests"]),
                             children_byte_sha256=sha(directory / "children.json"), children=child_bindings))


def audit_history(root, *, history_path=HISTORY, active_parent_paths=(C_PARENT,), max_candidates=10000):
    root = Path(root).resolve()
    query = pool_plan(max_candidates=max_candidates)
    old, old_binding = verify_old_snapshot(root, history_path, query)
    registry, manifests = freshness.inventory(root)
    active = {str(local_path(root, p).relative_to(root)) for p in active_parent_paths}
    require(active <= set(PARENT_PINS) and active <= set(manifests), "Unknown active assigned parent")
    ordinary = [r for r in registry if manifests[r["run"]].get("assay") not in OUTER_ASSAYS]
    current = freshness.audit(root, plan=query, pool_candidates=0, registry_snapshot=ordinary)
    exact = {r["semantic_world_id"]: deepcopy(r) for r in current["exact_world_exclusions"]}
    # Prepared-only runs are inventoried but their inputs are not exposed by
    # freshness.audit. Do not let later execution inherit an unearned receipt.
    screened, outer = {r["run"] for r in ordinary if r["execution_evidence"] != "none"}, {}
    for entry in registry:
        run, manifest = entry["run"], manifests[entry["run"]]
        if manifest.get("assay") not in OUTER_ASSAYS or manifest.get("parent_manifest_sha256"):
            continue
        if entry["execution_evidence"] == "none" and run not in active:
            continue
        adapted = screen_inputs(root, run, manifest, manifests)
        outer[run] = adapted
        names = [run, *(c["run"] for c in adapted["children"])]
        screened.update(names)
        for key, canonical in adapted["canonicals"].items():
            target = exact.setdefault(key, dict(semantic_world_id=key, canonical_world=canonical, source_runs=[]))
            require(target["canonical_world"] == canonical, "Canonical digest collision")
            target["source_runs"] = sorted(set(target["source_runs"]) | set(names))
    for entry in registry:
        require(entry["execution_evidence"] == "none" or entry["run"] in screened,
                "Executed collection has no explicit input-only exposure adapter")
    require(S_PARENT in outer, "The completed S exposure must be present")
    history_binding = next((b for b in manifests[S_PARENT]["bound_receipts"] if b["role"] == "history"), None)
    require(history_binding is not None and history_binding["byte_sha256"] == HISTORY_BYTES
            and local_path(root, history_binding["path"]) == local_path(root, history_path),
            "Old historical snapshot is not bound by the fixed S parent")
    if C_PARENT in outer:
        require(outer[C_PARENT]["base"] == outer[S_PARENT]["base"]
                and outer[C_PARENT]["requests"] == outer[S_PARENT]["requests"]
                and outer[C_PARENT]["canonicals"] == outer[S_PARENT]["canonicals"],
                "C inputs are not exactly the previously exposed S24")
    require(set(old["historical_exact_world_ids"]) <= set(exact), "Historical exposure was silently dropped")
    selected = deepcopy(current["selected_worlds"])
    for row in selected:
        row["exact_complete_world_runs"] = exact.get(row["semantic_world_id"], {}).get("source_runs", [])
    end_registry, end_manifests = freshness.inventory(root)
    require(set(end_manifests) == set(manifests)
            and all(digest(end_manifests[k]) == digest(v) for k, v in manifests.items()),
            "Manifest inventory changed during exposure audit; rerun the input-only audit")
    require(all(r["execution_evidence"] == "none" or r["run"] in screened for r in end_registry),
            "A previously uncounted collection executed during exposure audit")
    exclusions = [exact[key] for key in sorted(exact)]
    coverage = [dict(**r, input_exposure_included=r["run"] in screened,
                     explicitly_active_assigned_parent=r["run"] in active) for r in end_registry]
    return dict(
        schema_version=1, assay="reserved-object-exposure-history", selection_independent=True,
        scientific_admission=False, old_history=old_binding, historical_registry=coverage,
        historical_registry_sha256=digest(coverage), archive_manifest_aliases=current["archive_manifest_aliases"],
        source_archive_bindings=current["source_archive_bindings"],
        outer_source_bindings=[outer[k]["binding"] for k in sorted(outer)],
        active_assigned_parent_runs=sorted(active), historical_exact_world_ids=sorted(exact),
        historical_exact_world_ids_sha256=digest(sorted(exact)), exact_world_exclusions=exclusions,
        exact_world_exclusions_sha256=digest(exclusions), complete_pool_exposure=selected,
        complete_pool_exposure_sha256=digest(selected), pool=canonical_pool(), pool_sha256=digest(canonical_pool()),
        source_file_sha256={str(Path(m.__file__).resolve()): sha(m.__file__) for m in (voluntary, state)},
        script_byte_sha256=sha(__file__), historical_helper_byte_sha256=sha(freshness.__file__),
        inclusion_rule="All assigned worlds of every locally evidenced collection, including partial/failed runs, plus explicitly bound active S/C parents; no response fields used for inclusion.",
        limitations=[
            "Current local/archived inventory plus explicit active-parent assignments; this does not independently certify remote job shutdown or unknown external exposure.",
            "Exact named table plus realized truth differs from projected-table, vocabulary, abstract-structure and model-training exposure.",
            "Parent/child repetitions are unioned, not counted as independent worlds; prepared-only collections are listed but not presumed executed.",
            "Historical source reconstruction reuses the protected freshness helper; outer S/C input traversal is new and never reads generation/receipt responses.",
        ],
    )


def build_reserved_plan(history, *, half_seeds=HALF_SEEDS, max_candidates=10000):
    require(tuple(half_seeds) == HALF_SEEDS, "Do not change the two prospective half seeds")
    pool = canonical_pool()
    require(history["selection_independent"] is True and history["scientific_admission"] is False
            and history["pool"] == pool and history["pool_sha256"] == digest(pool)
            and history["historical_exact_world_ids"] == sorted(set(history["historical_exact_world_ids"]))
            and history["historical_exact_world_ids_sha256"] == digest(history["historical_exact_world_ids"])
            and history["complete_pool_exposure_sha256"] == digest(history["complete_pool_exposure"]),
            "Malformed complete-pool history binding")
    exposed = set(history["historical_exact_world_ids"])
    remaining = sorted(set(pool) - exposed)
    require(len(remaining) == 16, "Require all and exactly16 remaining exact-unused object worlds; no substitutes")
    exposures = {r["semantic_world_id"]: r for r in history["complete_pool_exposure"]}
    require(len(exposures) == len(history["complete_pool_exposure"]) == 64 and set(exposures) == set(pool)
            and all(bool(exposures[k]["exact_complete_world_runs"]) == (k in exposed) for k in pool),
            "Per-world exposure and exact history disagree")
    built = materials.build_materials(remaining, max_candidates=max_candidates)
    worlds, searches, base = built["worlds"], built["searches"], built["base_assignments"]
    selected_exposure = [deepcopy(exposures[k]) for k in remaining]
    selection = dict(
        seed=HALF_SEEDS[0], half_seeds=list(HALF_SEEDS), worlds=worlds, selected_semantic_world_ids=remaining,
        complete_pool_size=64, historical_exact_used_pool_size=48, reserved_worlds=16,
        history_sha256=digest(history), historical_exact_exclusions=history["historical_exact_world_ids"],
        historical_exact_exclusions_sha256=history["historical_exact_world_ids_sha256"],
        halves=[dict(panel_id=f"half_{i + 1}", seed=seed, assigned_worlds=8,
                     semantic_world_ids=remaining[8*i:8*i+8]) for i, seed in enumerate(HALF_SEEDS)],
        source_slots=list(SOURCE_SLOTS), searches=searches, assigned_exposure=selected_exposure,
        exact_overlap_count=0,
        any_projected_exposure_count=sum(bool(r["relational_projection_with_truth_runs"]
                                             or r["target_projected_table_runs"]
                                             or r["legacy_projection_runs"]) for r in selected_exposure),
        counterbalancing="Each fixed lexical half: two/menu rotation, two/consequence service, four/known property, both known properties within every rotation; two/correct diagnostic letter.",
        implementation_refinements=[
            "Keep lexical reserved_ordinal0..15 separate from source world_slot; v2 accepts only slots0..11. Reuse slots0,1,4,5,6,7,10,11 in each fixed half without changing its prompt/oracle.",
            "Use predeclared half seeds2026101541/2026101543 so repeated source slots do not reuse pattern RNG streams or trial IDs. Take each already-fixed canonical world's first matching generator index; neither IDs nor halves are selected by this index search.",
        ],
        ordering_seed=ORDER_SEED, sampling_seed_rule="first32bits SHA256(canonical JSON of 'voluntary-sample:'+trial_id)",
        scope="Reserved-world validation in the same finite task space; no independent-holdout or novel-table claim, no branch-priority decision, no launch admission.",
    )
    materials.validate_reserved_inputs(selection, base)
    return dict(schema_version=1, task_revision=REVISION, source_task_revision=voluntary.REVISION,
                scientific_admission=False, selection=selection, selection_sha256=digest(selection),
                base_assignments=base, base_assignment_sha256=digest(base),
                planned_denominators=dict(worlds=16, base_branches=192, choices=96, diagnostics=96, source_cohorts=6))


def prepare(root, *, out, history_path=HISTORY, active_parent_paths=(), max_candidates=10000):
    root = Path(root).resolve()
    path = local_path(root, out)
    require(not path.exists(), "Reserved inventory metadata is immutable")
    history = audit_history(root, history_path=history_path, active_parent_paths=active_parent_paths,
                            max_candidates=max_candidates)
    plan = build_reserved_plan(history, max_candidates=max_candidates)
    bundle = dict(schema_version=1, assay=REVISION, scientific_admission=False,
                  execution_frozen=False, gpu_launched=False, history=history, history_sha256=digest(history),
                  history_verification_scope="Recomputed against the live project's bound input history. A setup archive containing this receipt commits the inventory/materials but does not by itself contain or independently rederive every historical source/record. Scientific preparation must rerun live history before calibration outcomes.",
                  **{k: v for k, v in plan.items() if k not in ("schema_version", "scientific_admission")})
    write_once(path, bundle)
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--history", default=HISTORY)
    parser.add_argument("--active-parent", action="append", default=[])
    parser.add_argument("--max-candidates", type=int, default=10000)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--read-only", action="store_true")
    mode.add_argument("--out", type=Path)
    args = parser.parse_args()
    options = dict(history_path=args.history, active_parent_paths=args.active_parent,
                   max_candidates=args.max_candidates)
    if args.read_only:
        history = audit_history(args.root, **options)
        plan = build_reserved_plan(history, max_candidates=args.max_candidates)
    else:
        plan = prepare(args.root, out=args.out, **options)
    import json

    print(json.dumps(dict(written=not args.read_only, scientific_admission=False,
                          selection_sha256=plan["selection_sha256"],
                          base_assignment_sha256=plan["base_assignment_sha256"],
                          denominators=plan["planned_denominators"],
                          projected_exposure=plan["selection"]["any_projected_exposure_count"])))


if __name__ == "__main__":
    main()
