"""Input-only exposure and exact-material tests; no model/scientific outcomes."""
from collections import Counter
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import sys
import tarfile

import pytest

from curiosity_study import information_state as state
from curiosity_study import voluntary_resolution as voluntary
from curiosity_study import reserved_object_materials as materials
from curiosity_study import satisfaction_causal_sites as sites
from curiosity_study.artifacts import digest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import prepare_reserved_object_inventory as prepare  # noqa: E402
import audit_voluntary_freshness as old  # noqa: E402
from test_voluntary_freshness import make_run  # noqa: E402


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def pure_history(used=48):
    pool = materials.canonical_pool()
    ids = list(pool)[:used]
    exposure = [dict(family="objects", semantic_world_id=key, canonical_world=value,
                    exact_complete_world_runs=["old-inputs"] if key in ids else [],
                    relational_projection_with_truth_runs=[], target_projected_table_runs=["projected"],
                    legacy_projection_runs=[]) for key, value in pool.items()]
    return dict(selection_independent=True, scientific_admission=False, pool=pool, pool_sha256=digest(pool),
                historical_exact_world_ids=ids, historical_exact_world_ids_sha256=digest(ids),
                complete_pool_exposure=exposure, complete_pool_exposure_sha256=digest(exposure))


@pytest.fixture(scope="module")
def reserved():
    return prepare.build_reserved_plan(pure_history())


def test_complete64_pool_and_first_generator_indices():
    pool = materials.canonical_pool()
    assert len(pool) == 64
    assert len({digest({k: v for k, v in row.items() if k != "truth"}) for row in pool.values()}) == 16
    for seed in materials.HALF_SEEDS:
        indices, ledger = materials.first_indices(seed)
        assert set(indices) == set(pool)
        assert [r["candidate_index"] for r in ledger] == list(range(len(ledger)))
        for key, index in indices.items():
            assert index == next(r["candidate_index"] for r in ledger if r["semantic_world_id"] == key)
            a = dict(family="objects", seed=seed, index=index, vocabulary="heldout", known=[1], query=0)
            assert voluntary.canonical_world("objects", state.task(a)) == pool[key]
        assert sum(r["first_occurrence"] for r in ledger) == 64
    with pytest.raises(ValueError, match="cover all64"):
        materials.first_indices(materials.HALF_SEEDS[0], max_candidates=1)


def test_all192_variants_balances_and_unchanged_v2_oracle(reserved):
    selection, base = reserved["selection"], reserved["base_assignments"]
    assert selection["selected_semantic_world_ids"] == list(materials.canonical_pool())[48:]
    assert len(base) == len({r["base_trial_id"] for r in base}) == len({r["sampling_seed"] for r in base}) == 192
    assert materials.validate_reserved_inputs(selection, base)
    assert prepare.build_reserved_plan(pure_history()) == reserved
    for half, seed in zip(("half_1", "half_2"), materials.HALF_SEEDS, strict=True):
        rows = [r for r in base if r["panel_id"] == half]
        worlds = [w for w in selection["worlds"] if w["panel_id"] == half]
        assert len(rows) == 96 and len(worlds) == 8
        assert {r["panel_seed"] for r in rows} == {seed}
        assert Counter(r["kind"] for r in rows) == {"choice": 48, **dict.fromkeys(voluntary.DIAGNOSTICS, 8)}
        assert Counter(w["known"][0] for w in worlds) == {0: 4, 1: 4}
        assert Counter(w["world_slot"] % 4 for w in worlds) == dict.fromkeys(range(4), 2)
        for kind in voluntary.DIAGNOSTICS:
            assert Counter(voluntary.prompt(r["source_assignment"])[2] for r in rows if r["kind"] == kind) == dict.fromkeys("ABCD", 2)
        consequences = [voluntary.task(r["source_assignment"])["diagnostic_consequence_service"]
                        for r in rows if r["kind"] == "action_consequence"]
        assert Counter(consequences) == dict.fromkeys(voluntary.SERVICES, 2)
    assert Counter(r["semantic_world_id"] for r in base) == dict.fromkeys(selection["selected_semantic_world_ids"], 12)
    for row in base:
        source = row["source_assignment"]
        assert source["seed"] == row["panel_seed"]
        assert source["world_slot"] == row["world_slot"] < 12
        assert source["reserved_ordinal"] == row["reserved_ordinal"]
        assert row["inference_request_sha256"] == digest(voluntary.inference_request(source))
        assert voluntary.task(source)["uncertainty_bits"] == (0 if row["delivery"] == "decisive_evidence" else 1)


@pytest.mark.parametrize("used", [47, 49, 64])
def test_wrong_capacity_never_substitutes(used):
    with pytest.raises(ValueError, match="exactly16"):
        prepare.build_reserved_plan(pure_history(used))


@pytest.mark.parametrize("change", ["panel_id", "reserved_ordinal", "world_slot", "world_id", "panel_seed",
                                   "sampling_seed", "request", "reorder", "worlds", "halves", "exclusions",
                                   "projected", "first_index", "source_known"])
def test_material_tampering_is_rejected(reserved, change):
    selection, base = deepcopy(reserved["selection"]), deepcopy(reserved["base_assignments"])
    if change in ("panel_id", "world_id"):
        base[0][change] = "changed"
    elif change in ("reserved_ordinal", "world_slot", "panel_seed", "sampling_seed"):
        base[0][change] += 1
    elif change == "request":
        base[0]["inference_request_sha256"] = "0" * 64
    elif change == "reorder":
        base[0], base[1] = base[1], base[0]
    elif change in ("worlds", "halves"):
        selection[change].reverse()
    elif change == "exclusions":
        selection["historical_exact_exclusions"].pop()
        selection["historical_exact_exclusions_sha256"] = digest(selection["historical_exact_exclusions"])
    elif change == "projected":
        selection["assigned_exposure"][0]["target_projected_table_runs"] = []
    elif change == "first_index":
        selection["searches"][0]["first_indices"][selection["selected_semantic_world_ids"][0]] += 1
    else:
        base[0]["source_assignment"]["known"].reverse()
        base[0]["source_assignment"]["known"][0] = 1 - base[0]["source_assignment"]["known"][0]
        base[0]["source_assignment_sha256"] = digest(base[0]["source_assignment"])
    with pytest.raises(ValueError, match="Reserved"):
        materials.validate_reserved_inputs(selection, base)


def test_real2880_condition_constructor_reconciles_every_material(reserved, monkeypatch):
    monkeypatch.setattr(sites, "validate_reference", lambda value: value)
    monkeypatch.setattr(sites, "validate_directions", lambda value, **kwargs: value)
    directions = {arm: dict(dose=0 if arm == "zero_sham" else 1) for arm in sites.VALIDATION_ARMS if arm != "baseline"}
    args = dict(phase="validation", base_assignments=reserved["base_assignments"], selection=reserved["selection"],
                direction_bindings={"31": directions}, dose_reference={"doses": {"31": 1}}, selected_site=31)
    plan = sites.assignment_plan(**args)
    assert len(plan["assignments"]) == len({a["trial_id"] for a in plan["assignments"]}) == 2880
    assert Counter(a["arm"] for a in plan["assignments"]) == dict.fromkeys(sites.VALIDATION_ARMS, 192)
    for row in plan["assignments"]:
        assert all(row[key] == value for key, value in reserved["base_assignments"][row["base_row"]].items())
    args["selection"] = deepcopy(args["selection"])
    args["selection"]["worlds"][0]["reserved_ordinal"] = 9
    with pytest.raises(ValueError, match="Reserved"):
        sites.assignment_plan(**args)


def base_inputs(ids):
    indices, _ = materials.first_indices(materials.HALF_SEEDS[0])
    rows = []
    for ordinal, key in enumerate(ids):
        slot = ordinal % 12
        known = voluntary._known_attribute(slot)
        variants = [("choice", d, m) for d in voluntary.DELIVERIES for m in (0, 1)]
        variants += [(k, "decisive_evidence" if k in ("identify_resolved", "repeated_fact") else "redundant_evidence", 0)
                     for k in voluntary.DIAGNOSTICS]
        for kind, delivery, menu in variants:
            trial = f"synthetic-{ordinal}-{kind}-{delivery}-{menu}"
            source = dict(family="objects", seed=materials.HALF_SEEDS[0], index=indices[key], vocabulary="heldout",
                          known=[known], query=1-known, world_slot=slot, semantic_world_id=key, assay="thinking_information",
                          task_revision=voluntary.REVISION, kind=kind, delivery=delivery, menu_layout=menu, template="table",
                          trial_id=trial, sampling_seed=int(digest(trial)[:8], 16), max_new_tokens=4096,
                          execution=deepcopy(voluntary.EXECUTION))
            rows.append(dict(base_trial_id=trial, source_assignment=source, source_assignment_sha256=digest(source),
                             inference_request_sha256=digest(voluntary.inference_request(source)),
                             **{k: source[k] for k in ("family", "kind", "delivery", "menu_layout", "sampling_seed", "semantic_world_id", "world_slot")},
                             panel_id=f"panel_{ordinal//12+1}", panel_seed=source["seed"], world_id=f"world:{key}"))
    return rows


def screen_tree(root, name, base, *, curiosity, history_sha, executed):
    run = root / name
    assay = "curiosity_suppression_screen" if curiosity else "satisfaction_screen"
    arms = [f"arm{i}" for i in range(15 if curiosity else 4)]
    rows = [{**deepcopy(branch), "trial_id": f"{assay}-{arm}-{branch['base_trial_id']}", "arm": arm} for arm in arms for branch in base]
    plan = dict(base_assignments=base, base_assignment_sha256=digest(base), assignments=rows, assignment_sha256=digest(rows))
    requests = []
    for i, branch in enumerate(base):
        tokens = [1, i + 2, 3]
        requests.append(dict(base_trial_id=branch["base_trial_id"], source_assignment_sha256=branch["source_assignment_sha256"],
                             inference_request_sha256=branch["inference_request_sha256"],
                             request=dict(source_trial_id=branch["base_trial_id"], seed=branch["sampling_seed"], token_prefix=tokens,
                                          token_prefix_sha256=digest(tokens), prefix_positions=dict(record=1, final=2), labels=list("ABCD"))))
    by_base = {r["base_trial_id"]: r["request"] for r in requests}
    outer = [dict(trial_id=r["trial_id"], base_trial_id=r["base_trial_id"], sampling_seed=r["sampling_seed"],
                  source_assignment_sha256=r["source_assignment_sha256"], request_sha256=digest(by_base[r["base_trial_id"]])) for r in rows]
    preflight = dict(base_assignment_sha256=digest(base), assignment_sha256=digest(rows), requests=requests,
                     requests_sha256=digest(requests), outer_request_bindings=outer, outer_request_bindings_sha256=digest(outer))
    sources = {Path(m.__file__).name: Path(m.__file__).read_text() for m in (state, voluntary)}
    manifest = dict(assay=assay, source_sha256=digest(sources), assignments=[], diagnostics=rows, assignment_sha256=digest(rows),
                    screen_plan=plan, screen_plan_sha256=digest(plan), generation_preflight=preflight,
                    generation_preflight_sha256=digest(preflight), bound_receipts=[dict(role="history", path=prepare.HISTORY, byte_sha256=history_sha)])
    put(run / "manifest.json", manifest)
    with tarfile.open(run / "source.tar.gz", "w:gz") as archive:
        for filename, source in sources.items():
            payload = source.encode()
            member = tarfile.TarInfo("src/curiosity_study/" + filename)
            member.size = len(payload)
            archive.addfile(member, BytesIO(payload))
    put(run / "freeze.json", dict(manifest_sha256=digest(manifest), source_sha256=digest(sources), archive_sha256=prepare.sha(run / "source.tar.gz")))
    descriptors = []
    for group in range(3):
        child_path = run.with_name(run.name + f"-child{group}")
        child_rows = [r for i, r in enumerate(rows) if (i//32) % 3 == group]
        child = {**deepcopy(manifest), "diagnostics": child_rows, "assignment_sha256": digest(child_rows),
                 "execution_group": f"group{group}", "parent_manifest_sha256": digest(manifest)}
        put(child_path / "manifest.json", child)
        descriptors.append(dict(path=str(child_path), manifest_sha256=digest(child), execution_group=f"group{group}"))
        if executed and group == 0:
            marker = child_path / "cohorts" / "partial" / "0.json"
            marker.parent.mkdir(parents=True)
            marker.write_text("UNPARSEABLE RESPONSE SENTINEL")
    put(run / "children.json", descriptors)
    return manifest


@pytest.fixture
def history_tree(tmp_path, monkeypatch):
    ids = list(materials.canonical_pool())
    indices, _ = materials.first_indices(materials.HALF_SEEDS[0])
    historical = [dict(family="objects", seed=materials.HALF_SEEDS[0], index=indices[k], vocabulary="heldout",
                       known=[0], query=1, trial_id=f"old{i}", task_revision="equal-uncertainty-v1") for i, k in enumerate(ids[:24])]
    make_run(tmp_path, "old-inputs", historical)
    history = old.history_receipt(old.audit(tmp_path, plan=prepare.pool_plan(), pool_candidates=0))
    put(tmp_path / prepare.HISTORY, history)
    history_sha = prepare.sha(tmp_path / prepare.HISTORY)
    monkeypatch.setattr(prepare, "HISTORY_BYTES", history_sha)
    base = base_inputs(ids[24:48])
    s = screen_tree(tmp_path, prepare.S_PARENT, base, curiosity=False, history_sha=history_sha, executed=True)
    c = screen_tree(tmp_path, prepare.C_PARENT, base, curiosity=True, history_sha=history_sha, executed=False)
    monkeypatch.setattr(prepare, "PARENT_PINS", {prepare.S_PARENT: digest(s), prepare.C_PARENT: digest(c)})
    return tmp_path, history, s, c


def test_history_includes_failed_partial_s_and_active_c_without_response_parsing(history_tree):
    root, _, _, _ = history_tree
    history = prepare.audit_history(root)
    assert len(set(history["historical_exact_world_ids"]) & set(materials.canonical_pool())) == 48
    assert history["active_assigned_parent_runs"] == [prepare.C_PARENT]
    assert len(history["outer_source_bindings"]) == 2 and history["old_history"]["reproduced"]
    assert all(r["input_exposure_included"] for r in history["historical_registry"] if r["execution_evidence"] != "none")
    assert prepare.build_reserved_plan(history)["planned_denominators"]["base_branches"] == 192


def test_old_snapshot_cannot_be_rewritten(history_tree):
    root, history, _, _ = history_tree
    history["historical_registry"].clear()
    history["historical_registry_sha256"] = digest([])
    put(root / prepare.HISTORY, history)
    with pytest.raises(ValueError, match="fixed S-screen"):
        prepare.audit_history(root)
    target = root / "reports/must-not-exist.json"
    with pytest.raises(ValueError, match="fixed S-screen"):
        prepare.prepare(root, out=target)
    assert not target.exists()


def test_historical_collection_cannot_disappear_from_current_inventory(history_tree):
    root, _, _, _ = history_tree
    (root / "runs/old-inputs/manifest.json").unlink()
    with pytest.raises(ValueError, match="snapshot manifest/assignment"):
        prepare.audit_history(root)


def test_unknown_executed_assay_fails_closed(history_tree):
    root, _, _, _ = history_tree
    assignment = dict(family="objects", seed=5, index=0, trial_id="unknown", task_revision="unknown-v9")
    run, _ = make_run(root, "unknown", [assignment], assay="unknown", executed=False)
    assert any(r["run"] == "runs/unknown" for r in prepare.audit_history(root)["historical_registry"])
    marker = run / "cohorts" / "partial" / "0.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("DO NOT PARSE")
    with pytest.raises(ValueError, match="Unknown finite-task revision/assay"):
        prepare.audit_history(root)


def test_prepared_ordinary_run_starting_during_audit_is_not_silently_excluded(history_tree, monkeypatch):
    root, _, _, _ = history_tree
    assignment = dict(family="objects", seed=5, index=0, trial_id="late",
                      task_revision="equal-uncertainty-v1", vocabulary="heldout", known=[0], query=1)
    run, _ = make_run(root, "late", [assignment], executed=False)
    original = prepare.screen_inputs

    def start_ordinary(*args, **kwargs):
        value = original(*args, **kwargs)
        marker = run / "cohorts" / "partial" / "0.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("NO RESPONSE PARSING")
        return value

    monkeypatch.setattr(prepare, "screen_inputs", start_ordinary)
    with pytest.raises(ValueError, match="previously uncounted collection executed"):
        prepare.audit_history(root)


def test_conservatively_counted_active_c_may_gain_execution_evidence(history_tree, monkeypatch):
    root, _, _, _ = history_tree
    original = prepare.screen_inputs

    def start_active(*args, **kwargs):
        value = original(*args, **kwargs)
        if args[1] == prepare.C_PARENT:
            marker = root / value["children"][0]["run"] / "cohorts" / "partial" / "0.json"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("NO RESPONSE PARSING")
        return value

    monkeypatch.setattr(prepare, "screen_inputs", start_active)
    result = prepare.audit_history(root)
    assert len(set(result["historical_exact_world_ids"]) & set(materials.canonical_pool())) == 48


@pytest.mark.parametrize("change", ["missing_child", "reordered_child", "active_unknown"])
def test_incomplete_or_mutated_outer_coverage_rejects(history_tree, change):
    root, _, _, _ = history_tree
    if change == "active_unknown":
        with pytest.raises(ValueError, match="Unknown active"):
            prepare.audit_history(root, active_parent_paths=["runs/old-inputs"])
        return
    path = root / prepare.S_PARENT / "children.json"
    descriptors = json.loads(path.read_text())
    if change == "missing_child":
        put(path, descriptors[:-1])
        match = "three distinct"
    else:
        child_path = Path(descriptors[0]["path"]) / "manifest.json"
        child = json.loads(child_path.read_text())
        child["diagnostics"][0], child["diagnostics"][32] = child["diagnostics"][32], child["diagnostics"][0]
        child["assignment_sha256"] = digest(child["diagnostics"])
        put(child_path, child)
        descriptors[0]["manifest_sha256"] = digest(child)
        put(path, descriptors)
        match = "reordered"
    with pytest.raises(ValueError, match=match):
        prepare.audit_history(root)


def test_prepare_is_immutable_and_never_launches(history_tree):
    root, _, _, _ = history_tree
    output = root / "reports/reserved.json"
    bundle = prepare.prepare(root, out=output, active_parent_paths=[prepare.C_PARENT])
    assert not bundle["execution_frozen"] and not bundle["gpu_launched"] and not bundle["scientific_admission"]
    original = output.read_bytes()
    with pytest.raises(ValueError, match="immutable"):
        prepare.prepare(root, out=output)
    assert output.read_bytes() == original
