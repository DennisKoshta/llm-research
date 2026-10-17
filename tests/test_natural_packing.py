"""Prospective layouts and tiny CPU execution; no study artifacts or cloud work."""

from collections import defaultdict
import copy
import importlib.util
from pathlib import Path
import random

import numpy as np
import pytest

from curiosity_study import count_competence, information_state, natural_information, relational_information
from curiosity_study.artifacts import RunStore, digest, read
from curiosity_study.causal_contrasts import source_descriptor


@pytest.fixture
def preparer(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "natural_packing_preparer", scripts / "prepare_natural_interventions.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs(task=information_state, batch_size=32):
    assignments = task.assignments(items=2, seed=722, template="table") + task.assignments(
        items=2, seed=722, template="prose"
    )
    prefixes = {
        a["trial_id"]: dict(assignment=a, source_trial_id=a["trial_id"], seed=a["sampling_seed"] + 20000000)
        for a in assignments
    }
    config = dict(
        name="synthetic-packed",
        batch_size=batch_size,
        layers=[1],
        positions=["record", "prefill"],
        dose_reference_source="objects:balanced",
        order_seed=173,
        directions=[
            dict(name=name, source="objects:balanced", array="unit") for name in ("information", "answer")
        ],
        dose_multipliers=[-1, 1],
        random_seeds=[733],
        random_dose_multipliers=[-1, 1],
    )
    if batch_size == 32:
        config["prefix_group_size"] = 16
    fitted = dict(
        sources=np.array(["objects:balanced"]),
        layers=np.array([1]),
        positions=np.array(["record", "prefill"]),
        contrast_norm=np.ones((1, 1, 2)),
        unit=np.ones((1, 1, 2, 16)) / 4,
    )
    return config, prefixes, fitted


@pytest.mark.parametrize("task", [information_state, relational_information, count_competence])
def test_packed_pairs_preserve_every_source_arm_and_whole_world(preparer, task):
    config, prefixes, fitted = inputs(task)
    before = copy.deepcopy(prefixes)
    rows = preparer.build_schedule(config, prefixes, fitted)
    reversed_prefixes = dict(reversed(list(prefixes.items())))
    assert rows == preparer.build_schedule(config, reversed_prefixes, fitted)
    assert prefixes == before
    legacy = preparer.build_schedule({**config, "batch_size": 16}, prefixes, fitted)
    assert {a["trial_id"]: a for a in rows} == {a["trial_id"]: a for a in legacy}
    assert len(rows) == 16 * len(prefixes)
    assert {a["prefix_id"] for a in rows} == set(prefixes)
    assert sum(a["arm"] == "baseline" for a in rows) == len(prefixes)
    assert sum(a["direction_name"] == "padding_control" for a in rows) == len(prefixes)
    world_order, within_world = [], defaultdict(list)
    for offset in range(0, len(rows), 32):
        cohort = rows[offset : offset + 32]
        first, second = cohort[0]["prefix_id"], cohort[16]["prefix_id"]
        assert first != second
        assert [a["prefix_id"] for a in cohort] == [first] * 16 + [second] * 16
        aa = [prefixes[key]["assignment"] for key in (first, second)]
        world = preparer.world_id(aa[0])
        assert preparer.world_id(aa[1]) == world
        assert (aa[0]["kind"], aa[0]["template"]) == (aa[1]["kind"], aa[1]["template"])
        if not world_order or world_order[-1] != world:
            world_order.append(world)
        within_world[world].extend(
            (a["kind"], a["template"], source_descriptor(a)["cell"], a["trial_id"]) for a in aa
        )
        for key in (first, second):
            group = [a for a in cohort if a["prefix_id"] == key]
            assert group[0]["arm"] == "baseline"
            assert sum(a["arm"] == "baseline" for a in group) == 1
            assert any(a["arm"] == "sham" and a["direction_name"] == "sham" for a in group)
    expected_worlds = sorted({preparer.world_id(p["assignment"]) for p in prefixes.values()})
    random.Random(config["order_seed"]).shuffle(expected_worlds)
    assert world_order == expected_worlds
    assert all(items == sorted(items) for items in within_world.values())
    assigned_workers = {}
    partitions = preparer.partition_worlds(rows, prefixes, 32, 4)
    assert sorted(i for part in partitions for i in part) == list(range(len(rows) // 32))
    for worker, cohorts in enumerate(partitions):
        for ci in cohorts:
            for row in rows[ci * 32 : (ci + 1) * 32]:
                world = preparer.world_id(prefixes[row["prefix_id"]]["assignment"])
                assert assigned_workers.setdefault(world, worker) == worker


@pytest.mark.parametrize("size", [4, 8, 16])
def test_legacy_sizes_retain_seeded_prefix_order_and_per_prefix_padding(preparer, size):
    config, prefixes, fitted = inputs(batch_size=size)
    rows = preparer.build_schedule(config, prefixes, fitted)
    expected = sorted(prefixes)
    random.Random(config["order_seed"]).shuffle(expected)
    assert [rows[i]["prefix_id"] for i in range(0, len(rows), 16)] == expected
    assert rows == preparer.build_schedule({**config, "prefix_group_size": size}, prefixes, fitted)
    assert all(len({a["prefix_id"] for a in rows[i : i + size]}) == 1 for i in range(0, len(rows), size))


@pytest.mark.parametrize("group_size", [None, 8, 32, 16.0, True])
def test_batch32_requires_explicit_integer_group_size16(preparer, group_size):
    config, prefixes, fitted = inputs()
    if group_size is None:
        config.pop("prefix_group_size")
    else:
        config["prefix_group_size"] = group_size
    with pytest.raises(ValueError, match="explicit prefix_group_size=16"):
        preparer.build_schedule(config, prefixes, fitted)


def test_packing_rejects_oversized_arm_sets_odd_groups_and_source_aliases(preparer):
    config, prefixes, fitted = inputs()
    with pytest.raises(ValueError, match="fit within 16"):
        preparer.build_schedule({**config, "dose_multipliers": [-2, -1, 1, 2]}, prefixes, fitted)
    missing = dict(prefixes)
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match="even prefix counts"):
        preparer.build_schedule(config, missing, fitted)
    aliases = dict(prefixes)
    key = next(iter(aliases))
    aliases[key + "-alias"] = aliases.pop(key)
    with pytest.raises(ValueError, match="actual source trials"):
        preparer.build_schedule(config, aliases, fitted)
    # A different kind or rendering cannot fill an otherwise odd group.
    for field, value in (("kind", "repeat"), ("template", "other-rendering")):
        one = next(p for p in prefixes.values() if p["assignment"]["kind"] == "new")
        changed = copy.deepcopy(prefixes)
        changed[one["source_trial_id"]]["assignment"][field] = value
        with pytest.raises(ValueError, match="even prefix counts"):
            preparer.build_schedule(config, changed, fitted)
    rows = preparer.build_schedule(config, prefixes, fitted)
    with pytest.raises(ValueError, match="complete fixed-size cohorts"):
        preparer.partition_worlds(rows[:-16], prefixes, 32, 4)


def test_real_cpu_mixed32_preserves_shams_seeds_and_other_rows():
    import torch
    from test_state_backend import tiny_backend
    from curiosity_study.state_backend import StateBackend

    torch.set_num_threads(1)
    model = tiny_backend()
    model.model.generation_config.eos_token_id = None
    backend = StateBackend(model)
    requests, patches = [], []
    direction = np.eye(16)[0]
    for tokens, seed in (([11, 12, 13, 14, 15], 721), (list(range(11, 24)), 933)):
        request = dict(
            token_prefix=tokens,
            token_prefix_sha256=digest(tokens),
            prefix_positions=dict(record=1, final=len(tokens) - 1),
            labels=["Yes", "No"],
            seed=seed,
        )
        requests.extend([request] * 16)
        zero = dict(layer=1, position="final", mode="add", dose=0.0, direction=direction)
        patches.extend([None, zero] + [{**zero, "dose": (i - 7) * 0.02} for i in range(14)])
    rows, captured = backend.forward(requests, layers=[1], interventions=patches, max_new_tokens=64)
    assert captured.shape == (32, 1, 2, 16)
    for start in (0, 16):
        assert rows[start]["completion"] == rows[start + 1]["completion"]
        assert rows[start]["label_logits"] == rows[start + 1]["label_logits"]
        assert len(rows[start]["completion"]["generated_token_ids"]) == 64
    assert [row["completion"]["sampling"]["seed"] for row in rows] == [721] * 16 + [933] * 16
    assert all(row["unpadded_input_ids"] == req["token_prefix"] for row, req in zip(rows, requests))
    changed = list(patches)
    changed[2] = {**changed[2], "dose": 0.7}
    again, _ = backend.forward(requests, interventions=changed, max_new_tokens=64)
    assert all(
        row["completion"] == other["completion"] for i, (row, other) in enumerate(zip(rows, again)) if i != 2
    )
    assert model.active_hooks == 0


def test_mixed32_cohort_commit_and_partial_resume_are_atomic(tmp_path):
    from test_natural_information import fixture_collection
    from test_state_backend import tiny_backend

    manifest = fixture_collection(tmp_path)
    initial = manifest["diagnostics"]
    expanded = []
    for start in (0, 2):
        baseline, sham = initial[start : start + 2]
        expanded.extend(
            [baseline, sham] + [{**sham, "trial_id": f"{sham['trial_id']}-extra-{i}"} for i in range(14)]
        )
    manifest.update(diagnostics=expanded, assignment_sha256=digest(expanded), execution=dict(batch_size=32))
    model = tiny_backend()
    model.metadata = lambda: dict(model="tiny-engineering-transformer")
    rows = natural_information.execute(tmp_path / "mixed32", manifest, model, artifact_root=tmp_path)
    files = list((tmp_path / "mixed32/cohorts").glob("*/*.json"))
    assert len(files) == 1 and len(read(files[0])["records"]) == len(rows) == 32
    store = RunStore(tmp_path / "partial", manifest)
    store.save_many([rows[a["trial_id"]] for a in expanded[:16]], "partial", "0000000")
    with pytest.raises(ValueError, match="Partial cohort"):
        natural_information.execute(tmp_path / "partial", manifest, model, artifact_root=tmp_path)
    assert len(store.completed()) == 16
    broken = copy.deepcopy(manifest)
    broken["diagnostics"][2]["layer"] = 999
    broken["assignment_sha256"] = digest(broken["diagnostics"])
    with pytest.raises(ValueError, match="Invalid layer"):
        natural_information.execute(tmp_path / "failed", broken, model, artifact_root=tmp_path)
    failed = RunStore(tmp_path / "failed", broken).completed()
    assert len(failed) == 32 and all(r["failure"] == "execution_error" for r in failed.values())
    assert model.active_hooks == 0


@pytest.mark.parametrize("task", [information_state, relational_information, count_competence])
def test_condition_major_same_source_rows_every_arm_and_worker(preparer, task):
    config, _, fitted = inputs(task)
    config.pop("prefix_group_size")
    config["packing_mode"] = "condition_major_v1"
    items = 16 if task is count_competence else 8
    assignments = task.assignments(items=items, seed=731, template="table") + task.assignments(
        items=items, seed=731, template="prose"
    )
    prefixes = {a["trial_id"]: dict(assignment=a, source_trial_id=a["trial_id"]) for a in assignments}
    rows = preparer.build_schedule(config, prefixes, fitted)
    assert rows == preparer.build_schedule(config, dict(reversed(list(prefixes.items()))), fitted)
    assert len(rows) == len(prefixes) * 15
    assert not any(row["direction_name"] == "padding_control" for row in rows)
    layouts, conditions = {}, defaultdict(list)
    for offset in range(0, len(rows), 32):
        cohort = rows[offset : offset + 32]
        source_keys = [row["prefix_id"] for row in cohort]
        assert len(set(source_keys)) == 32
        assert [row["source_row"] for row in cohort] == list(range(32))
        assert len({row["condition"] for row in cohort}) == 1
        a = prefixes[source_keys[0]]["assignment"]
        block = (cohort[0]["source_block"], cohort[0].get("source_subcohort", 0), a["kind"], a["template"])
        assert layouts.setdefault(block, source_keys) == source_keys
        conditions[block].append(cohort[0]["condition"])
    assert all(len(set(value)) == len(value) == 15 for value in conditions.values())
    assert len({tuple(value) for value in conditions.values()}) > 1
    workers = {}
    for worker, cohorts in enumerate(preparer.partition_worlds(rows, prefixes, 32, 2)):
        for ci in cohorts:
            for row in rows[ci * 32 : (ci + 1) * 32]:
                world = preparer.world_id(prefixes[row["prefix_id"]]["assignment"])
                assert workers.setdefault(world, worker) == worker


def test_condition_major_incomplete_or_reordered_source_blocks_rejected(preparer):
    config, prefixes, fitted = inputs(batch_size=16)
    config["packing_mode"] = "condition_major_v1"
    with pytest.raises(ValueError, match="nonempty complete source blocks"):
        preparer.build_schedule(config, {}, fitted)
    with pytest.raises(ValueError, match="complete"):
        preparer.build_schedule(config, {key: value for key, value in list(prefixes.items())[1:]}, fitted)
    rows = preparer.build_schedule(config, prefixes, fitted)
    broken = copy.deepcopy(rows)
    broken[0]["source_row"] = 1
    with pytest.raises(ValueError, match="fixed row order"):
        preparer.partition_worlds(broken, prefixes, 16, 1)
    broken = copy.deepcopy(rows)
    for row in broken[:16]:
        row["source_block"] = 999
    with pytest.raises(ValueError, match="cannot be split"):
        preparer.partition_worlds(broken, prefixes, 16, 1)
    with pytest.raises(ValueError, match="whole-world source blocks"):
        preparer.build_schedule({**config, "batch_size": 32}, prefixes, fitted)
    with pytest.raises(ValueError, match="no repeated-prefix"):
        preparer.build_schedule({**config, "prefix_group_size": 16}, prefixes, fitted)


@pytest.mark.parametrize("declaration", [None, {}, {"assigned": 800}])
def test_explicit_incomplete_source_declarations_never_bypass_admission(preparer, declaration):
    with pytest.raises(ValueError, match="complete explicit source declaration"):
        preparer.expected_source_metadata({"expected_source_metadata": declaration}, {}, {})
