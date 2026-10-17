"""Synthetic inherited packages: no real outcomes, world selection or model calls."""

from collections import Counter
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
import pytest

from curiosity_study.coverage_engineering import call_plan

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import prepare_curiosity_coverage_bridge as p  # noqa: E402


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def archive_bytes(members):
    stream = BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, payload in members:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, BytesIO(payload))
    return stream.getvalue()


def repin(satisfaction, curiosity, monkeypatch):
    base = satisfaction["screen_plan"]["base_assignments"]
    base_sha = p.digest(base)
    satisfaction["screen_plan"]["base_assignment_sha256"] = base_sha
    satisfaction["generation_preflight"]["base_assignment_sha256"] = base_sha
    satisfaction["receipt_preflight"]["base_assignment_sha256"] = base_sha
    satisfaction["generation_preflight"]["requests_sha256"] = p.digest(
        satisfaction["generation_preflight"]["requests"]
    )
    satisfaction["generation_preflight_sha256"] = p.digest(satisfaction["generation_preflight"])
    satisfaction["receipt_preflight_sha256"] = p.digest(satisfaction["receipt_preflight"])
    curiosity.update(
        source_screen_manifest=deepcopy(satisfaction),
        source_screen_manifest_sha256=p.digest(satisfaction),
        generation_preflight=deepcopy(satisfaction["generation_preflight"]),
        generation_preflight_sha256=satisfaction["generation_preflight_sha256"],
        receipt_preflight=deepcopy(satisfaction["receipt_preflight"]),
        receipt_preflight_sha256=satisfaction["receipt_preflight_sha256"],
    )
    curiosity["screen_plan"]["base_assignments"] = deepcopy(base)
    monkeypatch.setattr(p, "BASE_SHA", base_sha)
    monkeypatch.setattr(p, "BASE_REQUESTS_SHA", satisfaction["generation_preflight"]["requests_sha256"])
    pins = deepcopy(p.SOURCES)
    pins["satisfaction"]["manifest_sha256"] = p.digest(satisfaction)
    pins["curiosity"]["manifest_sha256"] = p.digest(curiosity)
    monkeypatch.setattr(p, "SOURCES", pins)


@pytest.fixture
def sources(monkeypatch):
    variants = [("choice", delivery, menu) for delivery in p.DELIVERIES for menu in (0, 1)]
    variants += [
        (
            kind,
            "decisive_evidence" if kind in ("identify_resolved", "repeated_fact") else "redundant_evidence",
            0,
        )
        for kind in p.DIAGNOSTICS
    ]
    base, requests = [], []
    for panel in (1, 2):
        for slot in range(12):
            semantic = p.digest([panel, slot])
            for kind, delivery, menu in variants:
                index = len(base)
                trial = f"synthetic:{index:03d}"
                source = dict(
                    trial_id=trial,
                    task_revision="voluntary-resolution-v2",
                    family="objects",
                    kind=kind,
                    delivery=delivery,
                    menu_layout=menu,
                    sampling_seed=1000 + index,
                    seed=2000 + panel,
                    world_slot=slot,
                    semantic_world_id=semantic,
                )
                a = dict(
                    base_trial_id=trial,
                    source_assignment=source,
                    source_assignment_sha256=p.digest(source),
                    inference_request_sha256=p.digest(["request", index]),
                    sampling_seed=source["sampling_seed"],
                    panel_id=f"panel_{panel}",
                    panel_seed=source["seed"],
                    world_slot=slot,
                    semantic_world_id=semantic,
                    world_id=f"panel_{panel}:{semantic}",
                    family="objects",
                    kind=kind,
                    delivery=delivery,
                    menu_layout=menu,
                )
                base.append(a)
                tokens = list(range(20 if index // 32 == 3 else 8))
                request = dict(
                    source_trial_id=trial,
                    token_prefix=tokens,
                    token_prefix_sha256=p.digest(tokens),
                    prefix_positions=dict(record=len(tokens) - 2, final=len(tokens) - 1),
                    labels="ABCD",
                    seed=source["sampling_seed"],
                )
                rendered = f"Synthetic prompt {index}"
                requests.append(
                    dict(
                        base_trial_id=trial,
                        request=request,
                        source_assignment_sha256=p.digest(source),
                        inference_request_sha256=a["inference_request_sha256"],
                        rendered_prompt=rendered,
                        rendered_prompt_sha256=p.digest(rendered),
                        offset_mapping=[],
                    )
                )
    choices = [a for a in base if a["kind"] == "choice"]
    non_stop = [
        dict(
            base_trial_id=a["base_trial_id"],
            selected_service=service,
            token_binding_sha256=p.digest([a["base_trial_id"], service]),
            prefix_positions=dict(record=8, final=129 if service == "pattern" else 20),
        )
        for a in choices
        for service in ("report", "repeat", "pattern")
    ]
    stops = [a["base_trial_id"] for a in choices]
    receipt = dict(
        passed=True,
        generated_model_outputs=False,
        synthetic_answer_only=True,
        non_stop_bindings=non_stop,
        stop_checks=stops,
        bindings_sha256=p.digest(dict(non_stop=non_stop, stop=stops)),
    )
    satisfaction = dict(
        model=deepcopy(p.MODEL),
        screen_plan=dict(
            base_assignments=base,
            selection=dict(panels=["panel_1", "panel_2"], worlds=[p.digest(i) for i in range(24)]),
        ),
        generation_preflight=dict(requests=requests),
        receipt_preflight=receipt,
    )
    names = list(dict.fromkeys(["curiosity_vs_indifference", *(source for _, source, _ in p.NEIGHBORS)]))
    units = np.zeros((len(names), 4, 2, 5120), dtype=np.float64)
    norms = np.ones((len(names), 4, 2), dtype=np.float64)
    for index in range(len(names)):
        units[index, :, :, index] = 1
    norms[0, 0, 0] = p.C_NORM
    stream = BytesIO()
    np.savez_compressed(
        stream,
        sources=np.array(names),
        layers=np.array([15, 31, 47, 57]),
        positions=np.array(["record", "final"]),
        unit=units,
        contrast_norm=norms,
    )
    array_bytes = stream.getvalue()
    monkeypatch.setattr(p, "DIRECTION_BYTES_SHA", p.byte_sha(array_bytes))
    bindings = {}
    for arm, name, sign in (("zero_sham", names[0], 1), ("curiosity_negative", names[0], -1), *p.NEIGHBORS):
        original = units[names.index(name), 0, 0]
        bindings[arm] = dict(
            source=name,
            orientation=sign,
            corpus="A",
            layer=15,
            position="record",
            source_raw_norm=float(norms[names.index(name), 0, 0]),
            source_unit_sha256=p.digest(original.tolist()),
            directions_byte_sha256=p.DIRECTION_BYTES_SHA,
            unit=(sign * original).tolist(),
            reference_raw_norm=p.C_NORM,
            dose=0.0 if arm == "zero_sham" else p.C_NORM,
        )
    for index, arm in enumerate(p.RANDOM_ARMS):
        seed = 2026101501 + index
        original = np.random.default_rng(seed).standard_normal(5120)
        original /= np.linalg.norm(original)
        bindings[arm] = dict(
            source="seeded_gaussian",
            orientation=1,
            corpus=None,
            layer=15,
            position="record",
            unit=original.tolist(),
            seed=seed,
            generator="PCG64",
            construction="numpy_default_rng_standard_normal_float64_unit_no_resampling",
            reference_raw_norm=p.C_NORM,
            dose=p.C_NORM,
        )
    for binding in bindings.values():
        binding["unit_sha256"] = p.digest(binding["unit"])
    monkeypatch.setattr(p, "DIRECTION_BINDINGS_SHA", p.digest(bindings))
    inventory = dict(
        direction_bindings_sha256=p.digest(bindings),
        arm_order=list(p.PAIRS),
        resampled=False,
        orientation_selected_by_outcome=False,
        directions=[
            dict(
                arm=arm,
                binding_sha256=p.digest(bindings[arm]),
                signed_unit_float64_byte_sha256=p.byte_sha(
                    np.array(bindings[arm]["unit"], dtype=np.float64).tobytes()
                ),
                effective_float32_byte_sha256=p.byte_sha(
                    np.array(bindings[arm]["unit"], dtype=np.float32).tobytes()
                ),
            )
            for arm in p.PAIRS
        ],
    )
    curiosity = dict(
        model=deepcopy(p.MODEL),
        screen_plan=dict(direction_bindings=bindings, direction_bindings_sha256=p.digest(bindings)),
        direction_inventory_sha256=p.digest(inventory),
        candidate_source=dict(directions_byte_sha256=p.DIRECTION_BYTES_SHA),
    )
    repin(satisfaction, curiosity, monkeypatch)
    return (
        satisfaction,
        curiosity,
        bindings,
        inventory,
        {p.DIRECTIONS_PATH: array_bytes, p.DIRECTION_INVENTORY_PATH: encoded(inventory)},
    )


def material(sources, helper=call_plan):
    s, c, directions, inventory, _ = sources
    return p.build_materials(s, c, directions, inventory, {"synthetic": True}, helper)


def test_complete_grid_is_paired_and_distributed_without_sampling_or_input_mutation(sources, monkeypatch):
    before = p.digest(sources[:4])
    monkeypatch.setattr(np.random, "default_rng", lambda *a, **k: pytest.fail("must inherit, never sample"))
    plan = material(sources)
    rows = plan["assignments"]
    assert len(rows) == len({r["trial_id"] for r in rows}) == 8352
    assert Counter(r["arm"] for r in rows) == dict.fromkeys(p.ARMS, 288)
    assert Counter(r["kind"] == "choice" for r in rows) == {True: 4176, False: 4176}
    assert Counter(r["execution_group"] for r in rows) == dict.fromkeys(
        [f"worker_{i}" for i in range(3)], 2784
    )
    for worker in range(3):
        assert Counter(r["arm"] for r in rows if r["execution_group"] == f"worker_{worker}") == dict.fromkeys(
            p.ARMS, 96
        )
    for cohort in plan["execution_cohorts"]:
        group = rows[cohort["offset"] : cohort["offset"] + 32]
        start = cohort["source_cohort"] * 32
        assert [r["base_row"] for r in group] == list(range(start, start + 32))
        assert [r["base_trial_id"] for r in group] == [
            r["base_trial_id"] for r in sources[0]["screen_plan"]["base_assignments"][start : start + 32]
        ]
        for row in group:
            original = plan["base_assignments"][row["base_row"]]
            assert row["source_assignment"] == original["source_assignment"]
            assert row["sampling_seed"] == original["sampling_seed"]
    assert plan["generation_preflight"]["requests"] == sources[0]["generation_preflight"]["requests"]
    assert p.digest(sources[:4]) == before
    assert not plan["scientific_admission"] and not plan["engineering_admission"]
    assert plan["selection"]["fresh_worlds"] == 0


def test_counterbalance_and_position_mask_are_distinct_from_axis_extraction(sources):
    plan = material(sources)
    for index in range(9):
        order = p.arm_order(index)
        assert order[0] == "baseline" and set(order) == set(p.ARMS)
        direction = p.PAIRS[5 * index % 14]
        schedules = ("record", "full") if index % 2 == 0 else ("full", "record")
        assert order[1:3] == tuple(f"{schedule}__{direction}" for schedule in schedules)
    specs = {row["arm"]: row for row in plan["arm_specs"]}
    assert specs["baseline"]["intervention"] is None
    for direction in p.PAIRS:
        a, b = specs[f"record__{direction}"], specs[f"full__{direction}"]
        assert a["position"] == "record" and b["position"] == "all_attended"
        assert a["intervention"]["direction_binding_sha256"] == b["intervention"]["direction_binding_sha256"]
        assert plan["direction_bindings"][direction]["position"] == "record"
        assert (
            a["intervention"]["dose"]
            == b["intervention"]["dose"]
            == (0 if direction == "zero_sham" else p.C_NORM)
        )
    assert not plan["order_policy"]["exactly_counterbalanced"]


def test_engineering_inventory_and_all_possible_receipts_remain_input_only(sources):
    plan = material(sources)
    engineering = plan["engineering"]
    calls = engineering["call_plan"]
    assert len(calls) == 16 and len({trial for call in calls for trial in call["trial_ids"]}) == 512
    assert [c["call_index"] for c in calls if c["expected_error"]] == [6, 8]
    assert [(c["source_cohort"], c["condition"]) for c in calls[10:]] == [
        (cohort, condition)
        for cohort in (0, 3)
        for condition in ("baseline", "record_nonzero", "full_nonzero")
    ]
    assert {c["cap"] for c in calls[:10]} == {128} and {c["cap"] for c in calls[10:]} == {4096}
    assert engineering["source_assignments_by_cohort"]["3"] == [
        a["source_assignment"] for a in plan["base_assignments"][96:128]
    ]
    receipts = plan["receipt_materials"]
    assert receipts["assigned_choices"] == receipts["max_scientific_nonstop_forwards"] == 4176
    assert receipts["possible_scientific_nonstop_payloads"] == 12528
    assert receipts["maximum_rendered_token_length"] == 130 and receipts["maximum_padded_width"] == 256
    for row in receipts["assigned_dispositions"]:
        assert set(row["non_stop_bindings"]) == {"report", "repeat", "pattern"}
        assert (
            row["selected_service"] is None
            and not row["execution_claimed"]
            and not row["stop_requires_forward"]
        )
    assert not receipts["supplemental_synthetic_timing_forwards"]


def test_resource_bounds_and_prospective_format_amendment(sources):
    plan = material(sources)
    assert plan["resources"]["engineering"]["maximum_liability_usd"] == pytest.approx(4.9060352)
    assert plan["resources"]["scientific"]["maximum_liability_usd"] == pytest.approx(79.9460736)
    for phase in ("engineering", "scientific"):
        resource = plan["resources"][phase]
        assert resource["maximum_liability_usd"] <= resource["ceiling_usd"] and resource["paid_retries"] == 0
    criteria = plan["criteria"]
    assert (
        criteria["diagnostic_cell_minimum_strict_valid"]
        == criteria["diagnostic_cell_minimum_strict_correct"]
        == 11
    )
    assert criteria["diagnostic_arm_minimum_strict_valid"] == 137
    assert criteria["original_pristine_diagnostic_cell_minimum_strict_valid"] == 12
    assert criteria["choice_cell_minimum_strict_valid"] == 23
    assert (
        criteria["maximum_caps"] == criteria["maximum_missing"] == criteria["maximum_execution_errors"] == 0
    )
    assert plan["all_prior_results_remain_closed"] and not criteria["belief_equivalence_claim"]


@pytest.mark.parametrize(
    "mutation", ["drop", "duplicate_slot", "reordered_companions", "wrong_cap", "wrong_hash"]
)
def test_engineering_helper_cannot_silently_omit_or_reassign_slots(sources, mutation):
    def corrupt(requests):
        calls = call_plan(requests)
        if mutation == "drop":
            calls.pop()
        elif mutation == "duplicate_slot":
            calls[1]["trial_ids"][0] = calls[0]["trial_ids"][0]
        elif mutation == "reordered_companions":
            calls[0]["base_trial_ids"].reverse()
        elif mutation == "wrong_cap":
            calls[10]["cap"] = 128
        else:
            calls[0]["requests_sha256"] = "0" * 64
        return calls

    with pytest.raises(ValueError, match="engineering|Engineering"):
        material(sources, corrupt)


@pytest.mark.parametrize("mutation", ["variant", "seed", "prefix", "receipt", "max_cohort"])
def test_rehashed_source_structure_cannot_change_variants_requests_or_receipts(
    sources, monkeypatch, mutation
):
    s, c, _, _, _ = sources
    if mutation == "variant":
        row = s["screen_plan"]["base_assignments"][0]
        row["delivery"] = row["source_assignment"]["delivery"] = "redundant_evidence"
        row["source_assignment_sha256"] = p.digest(row["source_assignment"])
    elif mutation == "seed":
        s["generation_preflight"]["requests"][0]["request"]["seed"] += 1
    elif mutation == "prefix":
        s["generation_preflight"]["requests"][0]["request"]["token_prefix"][0] += 1
    elif mutation == "receipt":
        s["receipt_preflight"]["non_stop_bindings"][0]["selected_service"] = "repeat"
        r = s["receipt_preflight"]
        r["bindings_sha256"] = p.digest(dict(non_stop=r["non_stop_bindings"], stop=r["stop_checks"]))
    else:
        request = s["generation_preflight"]["requests"][0]["request"]
        request["token_prefix"] = list(range(1000))
        request["token_prefix_sha256"] = p.digest(request["token_prefix"])
        request["prefix_positions"]["final"] = 999
    repin(s, c, monkeypatch)
    with pytest.raises(ValueError):
        material(sources)


def test_locked_direction_validation_uses_original_array_bytes_and_never_resamples(sources, monkeypatch):
    _, curiosity, bindings, inventory, documents = sources
    monkeypatch.setattr(np.random, "default_rng", lambda *a, **k: pytest.fail("must not resample"))
    actual, original_inventory = p.validate_directions(curiosity, documents)
    assert actual == bindings and original_inventory == inventory
    assert actual["curiosity_negative"]["unit"][0] == -1
    assert actual["valence_positive"]["unit"] == (-np.asarray(actual["valence_negative"]["unit"])).tolist()
    documents[p.DIRECTIONS_PATH] += b"changed"
    with pytest.raises(ValueError, match="array bytes"):
        p.validate_directions(curiosity, documents)


@pytest.mark.parametrize("mutation", ["sign", "norm", "site", "random_seed", "float32_hash"])
def test_repinned_direction_metadata_still_cannot_change_fixed_geometry(sources, monkeypatch, mutation):
    _, curiosity, bindings, inventory, documents = sources
    target = bindings["curiosity_negative"]
    if mutation == "sign":
        target["unit"] = (-np.asarray(target["unit"])).tolist()
    elif mutation == "norm":
        target["dose"] *= 0.5
    elif mutation == "site":
        target["layer"] = 31
    elif mutation == "random_seed":
        bindings["random_01"]["seed"] += 1
    else:
        inventory["directions"][0]["effective_float32_byte_sha256"] = "0" * 64
    for arm, binding in bindings.items():
        binding["unit_sha256"] = p.digest(binding["unit"])
        next(r for r in inventory["directions"] if r["arm"] == arm)["binding_sha256"] = p.digest(binding)
    curiosity["screen_plan"]["direction_bindings_sha256"] = p.digest(bindings)
    inventory["direction_bindings_sha256"] = p.digest(bindings)
    curiosity["direction_inventory_sha256"] = p.digest(inventory)
    documents[p.DIRECTION_INVENTORY_PATH] = encoded(inventory)
    monkeypatch.setattr(p, "DIRECTION_BINDINGS_SHA", p.digest(bindings))
    with pytest.raises(ValueError):
        p.validate_directions(curiosity, documents)


def package_tree(root, sources, monkeypatch):
    s, c, _, _, documents = sources
    source = {"__init__.py": "# inert synthetic source\n"}
    source_sha = p.digest(source)
    s["source_sha256"] = c["source_sha256"] = source_sha
    repin(s, c, monkeypatch)
    pins = deepcopy(p.SOURCES)
    members_by_role = {}
    for role, manifest in (("satisfaction", s), ("curiosity", c)):
        docs = {"reports/synthetic-bound.txt": b"frozen input only"}
        if role == "curiosity":
            docs.update(documents)
            docs[f"runs/{p.SOURCES['satisfaction']['run']}/manifest.json"] = encoded(s)
        members = [("src/curiosity_study/__init__.py", source["__init__.py"].encode()), *docs.items()]
        members_by_role[role] = members
        archive = archive_bytes(members)
        run = root / "runs" / pins[role]["run"]
        run.mkdir(parents=True)
        (run / "manifest.json").write_bytes(encoded(manifest))
        (run / "source.tar.gz").write_bytes(archive)
        pins[role].update(
            manifest_sha256=p.digest(manifest), source_sha256=source_sha, archive_sha256=p.byte_sha(archive)
        )
        freeze = dict(
            manifest_sha256=p.digest(manifest),
            source_sha256=source_sha,
            archive_sha256=p.byte_sha(archive),
            documents={name: p.byte_sha(value) for name, value in docs.items()},
        )
        (run / "freeze.json").write_bytes(encoded(freeze))
    monkeypatch.setattr(p, "SOURCES", pins)
    plan = root / p.PLAN_PATH
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_text("synthetic prospective plan only\n")
    return members_by_role


def test_full_input_only_builder_verifies_archives_and_writes_nothing(tmp_path, sources, monkeypatch):
    package_tree(tmp_path, sources, monkeypatch)
    before = {str(path.relative_to(tmp_path)): p.sha(path) for path in tmp_path.rglob("*") if path.is_file()}
    plan = p.build_plan(tmp_path)
    after = {str(path.relative_to(tmp_path)): p.sha(path) for path in tmp_path.rglob("*") if path.is_file()}
    assert before == after
    assert len(plan["assignments"]) == 8352 and len(plan["engineering"]["call_plan"]) == 16
    assert not plan["source_provenance"]["outcome_records_inspected"]
    assert not plan["source_provenance"]["lifecycle_admission_checked"]
    for role in ("satisfaction", "curiosity"):
        assert not plan["source_provenance"][role]["source_code_executed"]
        assert not plan["source_provenance"][role]["response_records_read"]


@pytest.mark.parametrize(
    "mutation", ["manifest", "archive", "document", "source", "traversal", "duplicate", "missing_document"]
)
def test_frozen_archive_identity_and_member_inventory_fail_closed(tmp_path, sources, monkeypatch, mutation):
    members = package_tree(tmp_path, sources, monkeypatch)["curiosity"]
    run = tmp_path / "runs" / p.SOURCES["curiosity"]["run"]
    if mutation == "manifest":
        manifest = p.read(run / "manifest.json")
        manifest["unbound"] = True
        (run / "manifest.json").write_bytes(encoded(manifest))
    elif mutation == "archive":
        (run / "source.tar.gz").write_bytes(b"wrong archive")
    else:
        if mutation == "document":
            members = [
                (name, b"altered" if name.endswith("synthetic-bound.txt") else value)
                for name, value in members
            ]
        elif mutation == "source":
            members[0] = (members[0][0], b"# changed source\n")
        elif mutation == "traversal":
            members.append(("../outside", b"never extract"))
        elif mutation == "duplicate":
            members.append(members[0])
        else:
            members = [row for row in members if not row[0].endswith("synthetic-bound.txt")]
        archive = archive_bytes(members)
        (run / "source.tar.gz").write_bytes(archive)
        freeze = p.read(run / "freeze.json")
        freeze["archive_sha256"] = p.byte_sha(archive)
        (run / "freeze.json").write_bytes(encoded(freeze))
        monkeypatch.setitem(p.SOURCES["curiosity"], "archive_sha256", p.byte_sha(archive))
    with pytest.raises(ValueError):
        p.verify_package(tmp_path, "curiosity")
