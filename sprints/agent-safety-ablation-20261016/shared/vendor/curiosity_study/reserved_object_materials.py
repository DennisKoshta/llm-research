"""Pure reserved-world material reconstruction; never history or launch admission."""

from collections import Counter
from copy import deepcopy
import itertools
import random

from . import information_state as state
from . import voluntary_resolution as voluntary
from .artifacts import digest

HALF_SEEDS = (2026101541, 2026101543)
SOURCE_SLOTS = (0, 1, 4, 5, 6, 7, 10, 11)
ORDER_SEED = "reserved-object-order:2026101541:2026101543"
REVISION = "reserved-object-inventory-v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_pool():
    """Exhaustive4 color-pairs ×4 shape-pairs ×4 realized truths, not a sample."""
    require(voluntary.REVISION == "voluntary-resolution-v2", "Require unchanged v2 task semantics")
    result = {}
    for colors, shapes in itertools.product(state.HELDOUT_COLOR_PAIRS, state.HELDOUT_SHAPE_PAIRS):
        rows = list(itertools.product(colors, shapes))
        for truth in range(4):
            canonical = voluntary.canonical_world(
                "objects", dict(attributes=["color", "shape"], candidates=rows, truth=truth)
            )
            result[digest(canonical)] = canonical
    require(len(result) == 64, "The complete declared heldout-object pool must contain64 worlds")
    return dict(sorted(result.items()))



def first_indices(seed, *, max_candidates=10000):
    """Verify sampled coverage of all64 and retain every first-index search attempt."""
    require(type(seed) is int and seed in HALF_SEEDS, "Require the two fixed half seeds")
    require(type(max_candidates) is int and max_candidates > 0, "Invalid candidate limit")
    pool, first, ledger = canonical_pool(), {}, []
    for index in range(max_candidates):
        assignment = dict(family="objects", seed=seed, index=index, vocabulary="heldout", known=[0], query=1)
        canonical = voluntary.canonical_world("objects", state.task(assignment))
        key = digest(canonical)
        require(key in pool and pool[key] == canonical, "Generator produced a world outside the exact64 pool")
        ledger.append(dict(candidate_index=index, semantic_world_id=key, first_occurrence=key not in first))
        first.setdefault(key, index)
        if len(first) == 64:
            return first, ledger
    raise ValueError("Fixed seed did not cover all64 worlds within the fixed candidate limit")



def build_materials(world_ids, *, max_candidates=10000):
    """Reconstruct exact v2 inputs from an already fixed complete16-ID inventory."""
    pool = canonical_pool()
    require(isinstance(world_ids, list) and len(world_ids) == 16
            and world_ids == sorted(set(world_ids)) and set(world_ids) <= set(pool),
            "Require exactly16 lexically sorted distinct canonical world IDs")
    remaining, half_seeds = world_ids, HALF_SEEDS
    worlds, searches, base = [], [], []
    for half, seed in enumerate(half_seeds):
        indices, ledger = first_indices(seed, max_candidates=max_candidates)
        searches.append(dict(seed=seed, candidate_search_ledger=ledger, ledger_sha256=digest(ledger),
                             first_indices=indices, first_indices_sha256=digest(indices), complete_pool_covered=True))
        for local, slot in enumerate(SOURCE_SLOTS):
            ordinal = half * 8 + local
            key = remaining[ordinal]
            known = voluntary._known_attribute(slot)
            world = dict(family="objects", seed=seed, index=indices[key], vocabulary="heldout",
                         known=[known], query=1 - known, world_slot=slot, semantic_world_id=key,
                         reserved_ordinal=ordinal, half_id=f"half_{half + 1}", panel_id=f"half_{half + 1}",
                         panel_seed=seed, world_id=f"half_{half + 1}:{key}")
            require(voluntary.canonical_world("objects", state.task(world)) == pool[key], "Reserved oracle identity changed")
            worlds.append(world)
            variants = [("choice", d, m) for d, m in itertools.product(voluntary.DELIVERIES, (0, 1))]
            variants += [(k, "decisive_evidence" if k in ("identify_resolved", "repeated_fact")
                          else "redundant_evidence", 0) for k in voluntary.DIAGNOSTICS]
            for kind, delivery, menu in variants:
                trial = f"reserved-{seed}-objects-r{ordinal:02d}-{kind}-{delivery}-menu{menu}"
                source = {k: v for k, v in world.items() if k not in ("half_id", "panel_id", "panel_seed", "world_id")}
                source.update(assay="thinking_information", task_revision=voluntary.REVISION,
                              kind=kind, delivery=delivery, menu_layout=menu, template="table", trial_id=trial,
                              sampling_seed=int(digest(f"voluntary-sample:{trial}")[:8], 16),
                              max_new_tokens=4096, execution=deepcopy(voluntary.EXECUTION))
                request = voluntary.inference_request(source)
                base.append(dict(base_trial_id=trial, panel_id=world["panel_id"], half_id=world["half_id"],
                                 panel_seed=seed, world_id=world["world_id"], world_slot=slot,
                                 reserved_ordinal=ordinal, semantic_world_id=key, family="objects", kind=kind,
                                 delivery=delivery, menu_layout=menu, sampling_seed=source["sampling_seed"],
                                 source_assignment=source, source_assignment_sha256=digest(source),
                                 inference_request_sha256=digest(request)))
    for half in ("half_1", "half_2"):
        members = [w for w in worlds if w["half_id"] == half]
        require(Counter(w["known"][0] for w in members) == {0: 4, 1: 4}
                and Counter(w["world_slot"] % 4 for w in members) == dict.fromkeys(range(4), 2)
                and Counter(voluntary.SERVICES[w["world_slot"] // 3] for w in members)
                == dict.fromkeys(voluntary.SERVICES, 2)
                and all({w["known"][0] for w in members if w["world_slot"] % 4 == r} == {0, 1}
                        for r in range(4)), "Reserved-half counterbalancing failed")
        for kind in voluntary.DIAGNOSTICS:
            answers = [voluntary.prompt(b["source_assignment"])[2] for b in base
                       if b["half_id"] == half and b["kind"] == kind]
            require(Counter(answers) == dict.fromkeys("ABCD", 2), "Diagnostic answer-label balance failed")
    random.Random(ORDER_SEED).shuffle(base)
    require(len(base) == len({r["base_trial_id"] for r in base}) == len({r["sampling_seed"] for r in base}) == 192,
            "Reserved branch IDs/seeds/denominators differ")
    return dict(worlds=worlds, searches=searches, base_assignments=base)


def validate_reserved_inputs(selection, base):
    """Rebuild every material/identity; historical provenance is checked elsewhere.

    The complete exclusion set can be checked for internal consistency here, but
    only an independently bound input-history audit establishes its provenance.
    """
    pool = canonical_pool()
    ids = selection["selected_semantic_world_ids"]
    rebuilt = build_materials(ids)
    exclusions = selection["historical_exact_exclusions"]
    expected_halves = [
        dict(panel_id=f"half_{i + 1}", seed=seed, assigned_worlds=8, semantic_world_ids=ids[8*i:8*i+8])
        for i, seed in enumerate(HALF_SEEDS)
    ]
    require(
        selection["seed"] == HALF_SEEDS[0]
        and selection["half_seeds"] == list(HALF_SEEDS)
        and selection["source_slots"] == list(SOURCE_SLOTS)
        and selection["ordering_seed"] == ORDER_SEED
        and selection["halves"] == expected_halves
        and selection["worlds"] == rebuilt["worlds"]
        and selection["searches"] == rebuilt["searches"]
        and base == rebuilt["base_assignments"],
        "Reserved identities, halves, source slots, first indices, requests, seeds, or ordered192 branches changed",
    )
    require(
        isinstance(exclusions, list) and exclusions == sorted(set(exclusions))
        and all(isinstance(v, str) and len(v) == 64 and set(v) <= set("0123456789abcdef") for v in exclusions)
        and digest(exclusions) == selection["historical_exact_exclusions_sha256"]
        and ids == sorted(set(pool) - set(exclusions))
        and selection["complete_pool_size"] == 64
        and selection["historical_exact_used_pool_size"] == 48
        and selection["reserved_worlds"] == 16
        and selection["exact_overlap_count"] == 0
        and isinstance(selection["history_sha256"], str) and len(selection["history_sha256"]) == 64
        and set(selection["history_sha256"]) <= set("0123456789abcdef"),
        "Reserved inventory/exclusion history is internally inconsistent",
    )
    exposure = selection["assigned_exposure"]
    require(
        [r["semantic_world_id"] for r in exposure] == ids
        and all(r["canonical_world"] == pool[r["semantic_world_id"]]
                and r["family"] == "objects" and r["exact_complete_world_runs"] == [] for r in exposure)
        and selection["any_projected_exposure_count"] == sum(
            bool(r["relational_projection_with_truth_runs"] or r["target_projected_table_runs"]
                 or r["legacy_projection_runs"]) for r in exposure
        ), "Reserved exact/projected exposure descriptions disagree",
    )
    return True
