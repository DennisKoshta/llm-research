"""Synthetic audit evidence only; no scientific collection or GPU calls."""

from copy import deepcopy
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
import torch

from curiosity_study import satisfaction_capture as capture
from curiosity_study import satisfaction_screen as screen
from curiosity_study.artifacts import canonical, digest, read, write_once
from test_satisfaction_capture import fake_model, install_fake_generation
from test_satisfaction_screen import direction as base_direction
from test_voluntary_capture import fake_forward
from test_sustained_engineering import inputs, model_and_direction

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import audit_satisfaction_screen as audit  # noqa: E402
import prepare_satisfaction_screen as prepare  # noqa: E402


def direction():
    value = base_direction()
    value["unit"] = [0.6, 0.8, 0.0, 0.0]
    value["unit_sha256"] = digest(value["unit"])
    return value


def patch_generation_metadata(monkeypatch):
    original = capture.collect_generation

    def collect(model, requests, patches, **kwargs):
        condition, hidden = original(model, requests, patches, **kwargs)
        width = ((max(len(r["token_prefix"]) for r in requests) + 127) // 128) * 128
        condition.update(
            capture_phase=capture.CAPTURE_PLAN["source_phase"],
            unpadded_token_positions=[r["prefix_positions"] for r in requests],
            padded_token_positions=[
                [width - len(r["token_prefix"]) + r["prefix_positions"][p] for p in ("record", "final")]
                for r in requests
            ],
            prefill_width=width,
        )
        return condition, hidden

    monkeypatch.setattr(capture, "collect_generation", collect)


@pytest.fixture(scope="module")
def collected(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic-screen")
    model = fake_model()
    plan = screen.assignment_plan(historical_exact_world_ids=[], direction_binding=direction())
    preflight = capture.bind_preflight(plan, model)
    parent, children = prepare.build_manifests(plan, preflight, {}, plan["direction_binding"], {}, [])
    run = root / "parent"
    write_once(run / "manifest.json", parent)
    descriptors = []
    with pytest.MonkeyPatch.context() as monkeypatch:
        install_fake_generation(monkeypatch, (plan, preflight, model), [])
        patch_generation_metadata(monkeypatch)
        monkeypatch.setattr(capture.receipts, "_logit_hashes", fake_forward([]))
        for child in children:
            path = root / child["execution_group"]
            capture.execute(path, child, model)
            descriptors.append(
                dict(path=str(path), manifest_sha256=digest(child), execution_group=child["execution_group"])
            )
    write_once(run / "children.json", descriptors)
    return run, model


def bypass_separate_layers(monkeypatch, model):
    """This fixture exercises durable assembly/scoring; numerical validation is tested below."""
    monkeypatch.setattr(
        audit,
        "provenance",
        lambda *args: {
            "synthetic_separate_layer": True,
            "engineering_environment": {"policy": {"synthetic": True}},
        },
    )
    monkeypatch.setattr(audit, "token_model", lambda tokenizer: model)
    monkeypatch.setattr(audit, "environments", lambda *args: [dict(policy={"synthetic": True})])
    monkeypatch.setattr(
        audit,
        "validate_generation",
        lambda evidence, *a: dict(
            timeline=evidence["condition"]["rows"] is not None,
            captures=evidence["activation_artifact"] is not None,
            hook_cleanup=evidence["condition"]["hooks_clean"],
            geometry=True,
            engagement=True,
        ),
    )


def run_audit(run, model):
    return audit.audit(run, tokenizer=model.tokenizer, eos_token_ids={9999}, vocabulary_size=10000)


def clone(collected, tmp_path):
    run, model = collected
    shutil.copytree(run.parent, tmp_path / "copy")
    new = tmp_path / "copy" / run.name
    descriptors = read(new / "children.json")
    for descriptor in descriptors:
        descriptor["path"] = str(new.parent / Path(descriptor["path"]).name)
    (new / "children.json").write_text(canonical(descriptors))
    return new, model


def test_combined1152_preserves_failed_readiness_and576_dispositions(collected, monkeypatch):
    run, model = collected
    bypass_separate_layers(monkeypatch, model)
    result = run_audit(run, model)
    assert result["complete"] and result["integrity_passed"] and result["sham_identity_passed"]
    assert (
        result["engagement_passed"]
        and result["receipts_passed"]
        and result["cross_arm_prepatch_identity_passed"]
    )
    assert result["assigned"] == result["terminal_generations"] == len(result["outcomes"]) == 1152
    assert result["terminal_receipt_slots"] == 576
    assert [c["receipts"]["slots"] for c in result["children"]] == [288, 144, 144]
    assert not result["readiness"]["passed"] and not result["passed"]
    assert sum(r["valid"] is False for r in result["outcomes"]) == 192
    assert result == run_audit(run, model)


def test_child_only_or_missing_child_rejected(collected, tmp_path):
    run, _ = clone(collected, tmp_path)
    child = read(run / "children.json")[0]
    with pytest.raises(ValueError, match="Child-only"):
        audit.parent_children(Path(child["path"]), read(Path(child["path"]) / "manifest.json"))
    (run / "children.json").write_text(canonical([child]))
    with pytest.raises(ValueError, match="all execution groups"):
        audit.parent_children(run, read(run / "manifest.json"))


def test_missing_whole_cohort_retains_exact_parent_denominator(collected, tmp_path, monkeypatch):
    run, model = clone(collected, tmp_path)
    child = Path(read(run / "children.json")[-1]["path"])
    # Removing the final16 receipt slots is incomplete, never a smaller passing denominator.
    next((child / "receipt_cohorts").glob("*/0000128.json")).unlink()
    bypass_separate_layers(monkeypatch, model)
    result = run_audit(run, model)
    assert not result["complete"] and not result["integrity_passed"] and not result["receipts_passed"]
    assert len(result["outcomes"]) == 1152 and result["terminal_receipt_slots"] == 560


def test_changed_capture_bytes_rejected_even_if_answers_same(collected, tmp_path, monkeypatch):
    run, model = clone(collected, tmp_path)
    child = Path(read(run / "children.json")[1]["path"])
    path = next((child / "generation_activations").glob("*/*.npz"))
    path.write_bytes(path.read_bytes() + b"altered")
    bypass_separate_layers(monkeypatch, model)
    with pytest.raises(ValueError, match="artifact bytes"):
        run_audit(run, model)


def test_rehashed_wrong_task_score_still_rejected(collected, tmp_path, monkeypatch):
    run, model = clone(collected, tmp_path)
    child = Path(read(run / "children.json")[0]["path"])
    path = next((child / "cohorts").glob("*/*.json"))
    envelope = read(path)
    row = next(r for r in envelope["records"] if r["assignment"]["kind"] != "choice")
    row["task_record"]["correct"] = not row["task_record"]["correct"]
    envelope["sha256"] = digest(envelope["records"])
    path.write_text(canonical(envelope))
    bypass_separate_layers(monkeypatch, model)
    with pytest.raises(ValueError, match="parsing/scoring"):
        run_audit(run, model)


def test_cross_arm_identity_checks_final_site_signed_zero_and_missing(collected):
    assignments = read(collected[0] / "manifest.json")["diagnostics"]
    vectors = {a["trial_id"]: np.zeros((1, 2, 4), dtype=np.float32) for a in assignments}
    assert audit.cross_arm_identity(assignments, vectors)["passed"]
    a = next(a for a in assignments if a["arm"] == "satisfaction_plus_1_0")
    vectors[a["trial_id"]][0, 1, 0] = -0.0
    assert not audit.cross_arm_identity(assignments, vectors)["passed"]
    vectors.pop(a["trial_id"])
    result = audit.cross_arm_identity(assignments, vectors)
    assert not result["passed"] and result["assigned_base_rows"] == 288


def test_unexpected_generation_exception_is_auditable_unavailable(tmp_path, monkeypatch):
    model = fake_model()
    plan = screen.assignment_plan(historical_exact_world_ids=[], direction_binding=direction())
    preflight = capture.bind_preflight(plan, model)
    parent, children = prepare.build_manifests(plan, preflight, {}, plan["direction_binding"], {}, [])
    child = children[0]

    def explode(*args, **kwargs):
        raise RuntimeError("Synthetic wrapper failure")

    monkeypatch.setattr(capture, "collect_generation", explode)
    rows = capture.execute(tmp_path, child, model)
    monkeypatch.setattr(audit, "token_model", lambda _: model)
    result = audit.audit_generations(tmp_path, child, model.tokenizer, {9999}, 10000)
    assert len(rows) == len(result["records"]) == 32 and not result["complete"]
    assert result["checks"] == [
        dict(
            offset=0,
            arm="baseline",
            timeline=False,
            captures=False,
            hook_cleanup=True,
            geometry=False,
            engagement=False,
        )
    ]
    compact = [audit.compact(a, result["records"].get(a["trial_id"])) for a in parent["diagnostics"]]
    assert len(compact) == 1152 and sum(r["status"] == "missing" for r in compact) == 1120
    assert not audit.readiness(compact)["passed"]


def test_real_bf16_prefill_geometry_and_positive_each_event_engagement():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model, unit = model_and_direction()
        requests = inputs()
        patches = [dict(layer=15, position="record", mode="add", direction=unit, dose=0.25) for _ in requests]
        condition, hidden = capture.collect_generation(model, requests, patches, cap=3)
        assert condition["error"] is None
        for index, row in enumerate(condition["rows"]):
            trace = row["intervention_trace"]
            assert audit.geometry(trace, hidden[index, 0, 0], unit, 0.25)
            assert audit.engaged(trace, 0.25)
        trace = deepcopy(condition["rows"][0]["intervention_trace"])
        trace["events"][1]["requested_scalar_delta"] = 0.5
        assert not audit.geometry(trace, hidden[0, 0, 0], unit, 0.25)
        trace = deepcopy(condition["rows"][0]["intervention_trace"])
        trace["events"][1]["projection_after"] = trace["events"][1]["projection_before"]
        assert not audit.engaged(trace, 0.25)
        assert not audit.geometry(condition["rows"][0]["intervention_trace"], hidden[0, 0, 0] + 1, unit, 0.25)
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("failure", ["identity", "artifact_write"])
def test_receipt_measured_failure_is_preserved_not_rescored(collected, tmp_path, monkeypatch, failure):
    run, model = clone(collected, tmp_path)
    child = Path(read(run / "children.json")[1]["path"])
    sources = {
        row["trial_id"]: row for path in (child / "cohorts").glob("*/*.json") for row in read(path)["records"]
    }
    path = next((child / "receipt_cohorts").glob("*/0000000.json"))
    envelope = read(path)
    error = dict(type="SyntheticFailure", message=failure)
    envelope["error"] = error
    if failure == "identity":
        envelope["engineering"]["captured_full_initial_logits_sha256"][-1] = "f" * 64
        envelope["engineering"]["checks"]["full_initial_logits_equal"] = False
        envelope["engineering"]["passed"] = False
    else:
        envelope["observed_activation_artifact"] = None
    for row in envelope["records"]:
        if row["receipt_required"]:
            row.update(
                status="execution_error",
                receipt_model_pass_completed=False,
                measurement=None,
                activation_artifact=None,
                cohort=None,
                failure="receipt_execution_error",
                error=error,
                executed_simulator_outcome=deepcopy(
                    sources[row["source_trial_id"]]["task_record"]["simulator_outcome"]
                ),
            )
    envelope["records_sha256"] = digest(envelope["records"])
    path.write_text(canonical(envelope))
    monkeypatch.setattr(audit, "token_model", lambda _: model)
    result = audit.audit_receipts(child, read(child / "manifest.json"), sources, model.tokenizer)
    assert result["complete"] and result["slots"] == 144 and not result["passed"]
    assert result["status_counts"]["execution_error"] > 0
    # Failure cannot be promoted to successful delivery merely by rewriting the envelope digest.
    first = next(r for r in envelope["records"] if r["receipt_required"])
    first["receipt_model_pass_completed"] = True
    envelope["records_sha256"] = digest(envelope["records"])
    path.write_text(canonical(envelope))
    with pytest.raises(ValueError, match="Failed receipt claims"):
        audit.audit_receipts(child, read(child / "manifest.json"), sources, model.tokenizer)


def frozen_tree(root, *, changed_validator=False):
    """Bind synthetic documents to real validator source bytes without a study freeze."""
    from io import BytesIO
    import tarfile

    project = Path(audit.__file__).parents[1]
    source = {
        str(path.relative_to(project / "src/curiosity_study")): path.read_text()
        for path in sorted((project / "src/curiosity_study").rglob("*.py"))
    }
    if changed_validator:
        source[changed_validator] += "\n# Synthetic altered validator\n"
    roles = [
        "prospective_plan",
        "history",
        "exposure",
        "generation_preflight",
        "receipt_preflight",
        "engineering_manifest",
        "engineering_audit",
        "engineering_freeze",
        "engineering_archive",
        "candidate_manifest",
        "candidate_audit",
        "candidate_analysis",
        "candidate_directions",
        "candidate_independent_check",
    ]
    payloads, bindings = {}, []
    for role in roles:
        suffix = (
            ".md"
            if role == "prospective_plan"
            else ".npz"
            if role == "candidate_directions"
            else ".tar.gz"
            if role == "engineering_archive"
            else ".json"
        )
        name = "synthetic/" + role + suffix
        value = dict(role=role)
        if role == "history":
            value = dict(
                audit_script_byte_sha256=audit.sha(project / "scripts/audit_voluntary_freshness.py"),
                source_file_sha256={},
            )
        payload = canonical(value).encode() if suffix == ".json" else b"synthetic document"
        payloads[name] = payload
        bindings.append(
            dict(
                role=role,
                path=name,
                byte_sha256=__import__("hashlib").sha256(payload).hexdigest(),
                content_sha256=digest(value) if suffix == ".json" else None,
            )
        )
    for name, contents in source.items():
        payloads["src/curiosity_study/" + name] = contents.encode()
    payloads["scripts/audit_voluntary_freshness.py"] = (
        project / "scripts/audit_voluntary_freshness.py"
    ).read_bytes()
    root.mkdir()
    with tarfile.open(root / "source.tar.gz", "w:gz") as archive:
        for name, payload in payloads.items():
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, BytesIO(payload))
    manifest = dict(
        source_sha256=digest(source),
        bound_receipts=bindings,
        engine_sha256=capture.engineering.engine_hashes(),
    )
    frozen = dict(
        manifest_sha256=digest(manifest),
        source_sha256=manifest["source_sha256"],
        archive_sha256=audit.sha(root / "source.tar.gz"),
        documents={row["path"]: row["byte_sha256"] for row in bindings},
    )
    write_once(root / "freeze.json", frozen)
    return manifest, frozen


@pytest.mark.parametrize("changed_module", ["satisfaction_capture.py", "resource_limits.py"])
def test_frozen_document_and_shared_validator_bytes_are_checked(tmp_path, changed_module):
    manifest, frozen = frozen_tree(tmp_path / "okay")
    actual, documents, payloads = audit.frozen_documents(tmp_path / "okay", manifest)
    assert actual == frozen and len(documents) == len(payloads) == 14
    changed, _ = frozen_tree(tmp_path / "changed", changed_validator=changed_module)
    with pytest.raises(ValueError, match="Shared audit validator differs"):
        audit.frozen_documents(tmp_path / "changed", changed)
    # A self-consistent changed envelope still cannot replace an archived evidence role.
    manifest["bound_receipts"][0]["byte_sha256"] = "0" * 64
    frozen["manifest_sha256"] = digest(manifest)
    (tmp_path / "okay/freeze.json").write_text(canonical(frozen))
    with pytest.raises(ValueError, match="Frozen receipt bytes"):
        audit.frozen_documents(tmp_path / "okay", manifest)


def test_actual_bf16_generation_validator_checks_tokens_seed_and_geometry():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model, unit = model_and_direction()
        requests = inputs()
        vocabulary_size = model.model.config.vocab_size
        eos = set(range(vocabulary_size))
        model.model.generation_config.eos_token_id = sorted(eos)
        assignments = [dict(intervention=dict(dose=0.25)) for _ in requests]
        patches = capture.patches_for(assignments, unit)
        condition, hidden = capture.collect_generation(model, requests, patches, cap=4096)
        evidence = dict(
            condition=condition, eos_token_ids=sorted(eos), pad_token_id=model.tokenizer.pad_token_id
        )
        gates = audit.validate_generation(
            evidence, assignments, requests, model.tokenizer, eos, vocabulary_size, dict(hidden=hidden), unit
        )
        assert gates and all(gates.values())
        errored = deepcopy(evidence)
        errored["condition"]["error"] = dict(type="SyntheticFailure", message="Measured error")
        assert not audit.validate_generation(
            errored, assignments, requests, model.tokenizer, eos, vocabulary_size, dict(hidden=hidden), unit
        )["timeline"]
        changed = deepcopy(evidence)
        changed["condition"]["rows"][0]["completion"]["sampling"]["seed"] += 1
        assert not audit.validate_generation(
            changed, assignments, requests, model.tokenizer, eos, vocabulary_size, dict(hidden=hidden), unit
        )["timeline"]
        changed["condition"]["rows"][0]["completion"]["raw"] = "Invented final answer"
        with pytest.raises(ValueError, match="actual token IDs"):
            audit.validate_generation(
                changed,
                assignments,
                requests,
                model.tokenizer,
                eos,
                vocabulary_size,
                dict(hidden=hidden),
                unit,
            )
    finally:
        torch.set_num_threads(previous)


def test_same_study_workers_cannot_replace_engineering_runtime_policy():
    policy = dict(versions={"torch": "bound", "transformers": "bound"}, dtype="bfloat16")
    assert audit.runtime_compatible([deepcopy(policy) for _ in capture.GROUPS], policy)
    changed = deepcopy(policy)
    changed["versions"]["torch"] = "another version"
    assert not audit.runtime_compatible([deepcopy(changed) for _ in capture.GROUPS], policy)
    assert not audit.runtime_compatible([policy] * 2, policy)


def test_full_engineering_environment_requires_both_report_and_audit_hashes(tmp_path):
    environment = dict(
        **capture.MODEL,
        dtype="bfloat16",
        attention="eager",
        device="cuda",
        gpu="NVIDIA H200",
        cuda_numerics={"allow_bf16_reduced_precision_reduction": False},
        backend="transformers",
        loading_policy="synthetic",
        python="synthetic",
        cuda="synthetic",
        versions={"torch": "synthetic"},
        template_sha256="a" * 64,
        layers=64,
    )
    path = tmp_path / "engineering/attempt/report.json"
    write_once(path, dict(environment_sha256=digest(environment)))
    write_once(tmp_path / "environments/attempt.json", environment)
    receipt = dict(
        report_artifact=dict(
            path="engineering/attempt/report.json", sha256=audit.sha(path), byte_count=path.stat().st_size
        ),
        environment_sha256=digest(environment),
    )
    result = audit.engineering_environment(tmp_path, receipt)
    assert result["content_sha256"] == digest(environment)
    assert result["byte_sha256"] == audit.sha(tmp_path / "environments/attempt.json")
    environment["versions"]["torch"] = "changed"
    (tmp_path / "environments/attempt.json").write_text(canonical(environment))
    receipt["environment_sha256"] = digest(environment)
    with pytest.raises(ValueError, match="audit/report"):
        audit.engineering_environment(tmp_path, receipt)


def test_receipt_only_resume_cannot_reuse_another_generation_runtime(collected, tmp_path, monkeypatch):
    run, model = clone(collected, tmp_path)
    child = Path(read(run / "children.json")[1]["path"])
    sources = {
        row["trial_id"]: row for path in (child / "cohorts").glob("*/*.json") for row in read(path)["records"]
    }
    path = next((child / "receipt_cohorts").glob("*/0000000.json"))
    moved = child / "receipt_cohorts/resumed-attempt" / path.name
    moved.parent.mkdir()
    original_bytes = path.read_bytes()
    path.rename(moved)
    assert moved.read_bytes() == original_bytes  # Every source, payload and artifact hash still matches.
    monkeypatch.setattr(audit, "token_model", lambda _: model)
    with pytest.raises(ValueError, match="sole verified generation attempt"):
        audit.audit_receipts(child, read(child / "manifest.json"), sources, model.tokenizer)


@pytest.mark.parametrize(
    "mismatch", ["successful_with_error", "row_error_disagrees", "missing_envelope_error"]
)
def test_receipt_error_flags_cannot_coexist_with_success_or_lose_provenance(
    collected, tmp_path, monkeypatch, mismatch
):
    run, model = clone(collected, tmp_path)
    child = Path(read(run / "children.json")[1]["path"])
    sources = {
        row["trial_id"]: row for path in (child / "cohorts").glob("*/*.json") for row in read(path)["records"]
    }
    path = next((child / "receipt_cohorts").glob("*/0000000.json"))
    envelope = read(path)
    error = dict(type="SyntheticFailure", message="An actual failed receipt call")
    envelope["error"] = error
    if mismatch != "successful_with_error":
        active = [row for row in envelope["records"] if row["receipt_required"]]
        for row in active:
            row.update(
                status="execution_error",
                failure="receipt_execution_error",
                error=deepcopy(error),
                receipt_model_pass_completed=False,
                measurement=None,
                activation_artifact=None,
                cohort=None,
                executed_simulator_outcome=deepcopy(
                    sources[row["source_trial_id"]]["task_record"]["simulator_outcome"]
                ),
            )
        if mismatch == "row_error_disagrees":
            active[0]["error"]["message"] = "A different event"
        else:
            envelope["error"] = None
    envelope["records_sha256"] = digest(envelope["records"])
    path.write_text(canonical(envelope))
    monkeypatch.setattr(audit, "token_model", lambda _: model)
    with pytest.raises(ValueError, match="Receipt (envelope error|failure is missing)"):
        audit.audit_receipts(child, read(child / "manifest.json"), sources, model.tokenizer)


@pytest.mark.parametrize(
    "field,value",
    [("workers", 2), ("shared_deadline_seconds", 7200), ("stage_ceiling_usd", 55.0), ("paid_retries", 1)],
)
def test_parent_budget_must_match_exact_prospective_policy(collected, field, value):
    run, _ = collected
    manifest = read(run / "manifest.json")
    manifest["budget"][field] = value
    with pytest.raises(ValueError, match="fixed screen resource policy"):
        audit.parent_children(run, manifest)
