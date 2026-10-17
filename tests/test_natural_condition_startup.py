"""Real tiny BF16 model checks for the prospective full condition-major cohort."""

import copy
import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from curiosity_study import natural_information as natural
from curiosity_study.artifacts import RunStore, digest, read, write_once
from curiosity_study.information_state import assignments
from curiosity_study.replay_information import intervention
from curiosity_study.state_backend import StateBackend
from test_natural_information import source_record
from test_state_backend import tiny_backend


def script(name):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(f"startup_{name}", scripts / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def condition_manifest(tmp_path):
    prepare = script("prepare_natural_interventions")
    source_rows = [{**a, "max_new_tokens": 4096} for a in assignments(4, kinds=("new",))]
    model = dict(model_id="synthetic-tiny", revision="synthetic")
    source_manifest = dict(model=model, assignments=[], diagnostics=source_rows)
    store = RunStore(tmp_path / "source", source_manifest)
    prefixes = {}
    for i, a in enumerate(source_rows):
        record = source_record(a, ids=[11, 12, i + 20] + [13] * (i % 6 + 2))
        store.save(record, "synthetic")
        prefixes[a["trial_id"]] = natural.make_prefix(record, "source", digest(source_manifest))
    directions = tmp_path / "directions.npz"
    fitted = dict(
        sources=np.array(["synthetic:balanced"]),
        layers=np.array([1]),
        positions=np.array(["record"]),
        unit=np.arange(1, 17, dtype=float).reshape(1, 1, 1, 16),
        contrast_norm=np.full((1, 1, 1), 2.5),
    )
    np.savez_compressed(directions, **fitted)
    config = dict(
        name="condition-startup-fixture",
        phase="engineering",
        inference_scope="development",
        batch_size=32,
        packing_mode="condition_major_v1",
        max_new_tokens=4096,
        layers=[1],
        positions=["record"],
        dose_reference_source="synthetic:balanced",
        order_seed=733,
        directions=[dict(name="information", source="synthetic:balanced", array="unit")],
        dose_multipliers=[1],
        random_seeds=[],
        random_dose_multipliers=[1],
        directions_local=str(directions),
    )
    rows = prepare.build_schedule(config, prefixes, fitted)
    policy = prepare.packing_policy(config)
    manifest = dict(
        assay="natural_information",
        phase="engineering",
        model=model,
        assignments=[],
        diagnostics=rows,
        assignment_sha256=digest(rows),
        prefixes=prefixes,
        prefixes_sha256=digest(prefixes),
        execution=dict(backend="torch_batched_v1", batch_size=32),
        capture_layers=[0, 1, 2],
        max_new_tokens=4096,
        greedy=False,
        engine_sha256=natural.engine_hashes(),
        intervention_schedule=config,
        packing_policy=policy,
        packing_policy_sha256=digest(policy),
        padded_prefix_groups=False,
        direction_input=dict(
            path=directions.name, sha256=hashlib.sha256(directions.read_bytes()).hexdigest()
        ),
    )
    binding = natural.representative_cohort_binding(manifest)
    requests = [prefixes[key] for key in binding["source_ids"]]
    patch = intervention(binding["perturbation_assignment"], fitted, 16)
    return manifest, requests, binding, patch


@pytest.fixture(scope="module")
def full_guard(tmp_path_factory):
    root = tmp_path_factory.mktemp("natural-condition-startup")
    manifest, requests, binding, patch = condition_manifest(root)
    model = tiny_backend(dtype=torch.bfloat16)
    report = natural.verify_natural_backend(model, requests, binding, patch)
    assert model.active_hooks == 0 and not model.model._forward_hooks
    return manifest, report


def test_real_32_row_guard_checks_full_logits_captures_tokens_and_bound_actual_schedule(full_guard):
    manifest, report = full_guard
    assert report["passed"], report["checks"]
    assert script("audit_natural_interventions").startup_binding(report, manifest)
    binding = report["representative_cohort"]
    assert binding["batch_size"] == 32 and binding["unperturbed_rows"] == list(range(1, 32))
    assert binding["source_ids"] == [a["prefix_id"] for a in manifest["diagnostics"][:32]]
    assert binding["cohort_assignment_sha256"] == digest(manifest["diagnostics"][:32])
    assert binding["max_new_tokens"] == 16
    assert binding["perturbation_assignment"]["position"] == "record"
    assert binding["perturbation_assignment"]["dose"] == 2.5
    for rows in report["representative_cohort_evidence"].values():
        assert len(rows) == 32
        assert all(row["prefill_logits"]["shape"] == [128] for row in rows)
        assert all(row["captures"]["shape"] == [3, 2, 16] for row in rows)
        assert all(len(row["generated_token_ids"]) == 16 for row in rows)


@pytest.mark.parametrize("field", ["prefill_logits", "captures", "generated_token_ids"])
def test_auditor_recomputes_row_31_isolation_instead_of_trusting_pass_flag(full_guard, field):
    manifest, original = full_guard
    report = copy.deepcopy(original)
    row = report["representative_cohort_evidence"]["single_row_perturbation"][31]
    if field == "generated_token_ids":
        row[field][0] = (row[field][0] + 1) % 128
    else:
        row[field]["sha256"] = "changed-unrelated-row"
    audit = script("audit_natural_interventions")
    with pytest.raises(ValueError, match="checks disagree with row evidence"):
        audit.startup_binding(report, manifest)
    report["checks"].update(
        natural.representative_cohort_checks(
            report["representative_cohort"], report["representative_cohort_evidence"]
        )
    )
    report["passed"] = False
    assert not audit.startup_binding(report, manifest)


def test_actual_unrelated_full_vocabulary_logit_change_fails_guard(tmp_path, monkeypatch):
    _, requests, binding, patch = condition_manifest(tmp_path)
    model = tiny_backend(dtype=torch.bfloat16)
    original = StateBackend.forward
    corrupt = False
    calls = []

    def forward(self, selected, *args, **kwargs):
        nonlocal corrupt
        if len(selected) == 32:
            calls.append(len(selected))
        patches = kwargs.get("interventions")
        corrupt = len(selected) == 32 and bool(patches) and patches[0]["dose"] != 0
        try:
            return original(self, selected, *args, **kwargs)
        finally:
            corrupt = False

    def damage_unrelated_logit(module, args, output):
        if corrupt:
            # Not a response-label logit: only the complete vocabulary check detects this.
            output.logits[-1, -1, 7] += 1

    monkeypatch.setattr(StateBackend, "forward", forward)
    handle = model.model.register_forward_hook(damage_unrelated_logit)
    try:
        report = natural.verify_natural_backend(model, requests, binding, patch)
    finally:
        handle.remove()
    assert calls == [32, 32, 32]
    assert not report["passed"]
    assert not report["checks"]["representative_unperturbed_logits_equal"]
    assert model.active_hooks == 0 and not model.model._forward_hooks


def test_full_logit_capture_hook_is_removed_when_forward_raises(tmp_path, monkeypatch):
    _, requests, binding, patch = condition_manifest(tmp_path)
    model = tiny_backend(dtype=torch.bfloat16)

    def fail(*args, **kwargs):
        raise RuntimeError("deliberate-forward-failure")

    monkeypatch.setattr(StateBackend, "forward", fail)
    with pytest.raises(RuntimeError, match="deliberate-forward-failure"):
        natural._representative_forward(
            StateBackend(model), requests, binding["capture_layers"], [patch] + [None] * 31
        )
    assert not model.model._forward_hooks and model.active_hooks == 0


@pytest.mark.parametrize(
    "field", ["source_ids", "batch_size", "max_new_tokens", "source_seeds", "packing_policy"]
)
def test_auditor_rejects_another_cohort_or_shorter_guard(full_guard, field):
    manifest, original = full_guard
    report = copy.deepcopy(original)
    report["representative_cohort"][field] = "different"
    with pytest.raises(ValueError, match="cohort binding mismatch"):
        script("audit_natural_interventions").startup_binding(report, manifest)


def test_execute_persists_guard_for_the_actual_first_condition_cohort(tmp_path):
    manifest, _, binding, _ = condition_manifest(tmp_path)
    model = tiny_backend(dtype=torch.bfloat16)
    model.metadata = lambda: manifest["model"]
    # Full scientific allowance remains 4096; engineered EOS avoids a long CPU test.
    model.model.generation_config.eos_token_id = list(range(128))
    root = tmp_path / "natural"
    records = natural.execute(root, manifest, model, artifact_root=tmp_path)
    assert len(records) == 96
    files = list((root / "natural_integrity").glob("*.json"))
    assert len(files) == 1
    report = read(files[0])
    assert report["representative_cohort"] == binding
    assert script("audit_natural_interventions").startup_binding(report, manifest)
    assert all(r["measurement"]["completion"]["sampling"]["max_new_tokens"] == 4096 for r in records.values())
    assert model.active_hooks == 0 and not model.model._forward_hooks


@pytest.mark.parametrize("size", [4, 8, 16, 32])
def test_condition_major_requires_explicit_policy_and_no_padding_at_every_batch_size(size):
    audit = script("audit_natural_interventions")
    prepare = script("prepare_natural_interventions")
    config = dict(batch_size=size, packing_mode="condition_major_v1")
    policy = prepare.packing_policy(config)
    manifest = dict(intervention_schedule=config, execution=dict(backend="torch_batched_v1", batch_size=size))
    with pytest.raises(ValueError, match="explicit packing metadata"):
        audit.packing_binding(manifest)
    manifest.update(packing_policy=policy, packing_policy_sha256=digest(policy))
    with pytest.raises(ValueError, match="cannot pad"):
        audit.packing_binding(manifest)
    manifest["padded_prefix_groups"] = False
    assert audit.packing_binding(manifest) == policy


@pytest.mark.parametrize("receipt", ["matching", "absent", "smaller_batch", "different_packing"])
def test_prior_engineering_requires_representative_guard_for_same_execution(full_guard, tmp_path, receipt):
    original, startup = full_guard
    cohort = copy.deepcopy(startup["representative_cohort"])
    if receipt == "smaller_batch":
        cohort["batch_size"] = 16
    elif receipt == "different_packing":
        cohort["packing_policy"]["mode"] = "paired_prefixes_v1"
    report = dict(
        assay="natural_information",
        phase="engineering",
        engineering_passed=True,
        engineering_checks=dict(synthetic=True),
        integrity_passed=True,
        complete=True,
        model=original["model"],
        engine_sha256=original["engine_sha256"],
        execution=original["execution"],
        packing_policy=original["packing_policy"],
        packing_policy_sha256=original["packing_policy_sha256"],
        manifest_sha256=digest(original),
        max_new_tokens=4096,
        assigned_sources=32,
        maximum_baseline_generated_tokens=128,
    )
    if receipt != "absent":
        report["representative_startup_audits"] = [dict(passed=True, representative_cohort=cohort)]
    path = tmp_path / "previous.json"
    write_once(path, report)
    future = copy.deepcopy(original)
    future["phase"] = "confirmation"
    future["intervention_schedule"].update(
        natural_engineering_audit=str(path), natural_engineering_audit_sha256=digest(report)
    )
    result = script("audit_natural_interventions").previous_engineering(future)
    assert result["passed"] is (receipt == "matching")
