"""Preparation provenance and no-write admission failures; no inference or cloud."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

import pytest

from curiosity_study.artifacts import digest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_pain_profile_control as prep  # noqa: E402
from test_pain_profile_design import Tokenizer  # noqa: E402


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def closure(root, monkeypatch, *, positive=False):
    """Synthetic closed study, including an intentionally failed readiness gate."""
    archive = root / prep.C_RUN / "source.tar.gz"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"synthetic closed source archive")
    source = "a" * 64
    manifest = dict(assay="synthetic_preceding_study", source_sha256=source)
    manifest_sha, archive_sha = digest(manifest), prep.sha(archive)
    monkeypatch.setattr(prep, "C_MANIFEST_SHA", manifest_sha)
    monkeypatch.setattr(prep, "C_SOURCE_SHA", source)
    monkeypatch.setattr(prep, "C_ARCHIVE_SHA", archive_sha)
    frozen = dict(manifest_sha256=manifest_sha, source_sha256=source, archive_sha256=archive_sha)
    write(root / prep.C_FILES["manifest"], manifest)
    write(root / prep.C_FILES["freeze"], frozen)
    audit = dict(
        parent_manifest_sha256=manifest_sha,
        source_sha256=source,
        complete=True,
        integrity_passed=True,
        assigned=4320,
        terminal_generations=4320,
        readiness=dict(passed=False, reason="synthetic formatting failure"),
    )
    write(root / prep.C_FILES["audit"], audit)
    decision = dict(classification="positive" if positive else "readiness_failed", positive_screen=positive)
    analysis = dict(
        parent_manifest_sha256=manifest_sha,
        source_sha256=source,
        complete=True,
        integrity_passed=True,
        audit_sha256=digest(audit),
        audit_byte_sha256=prep.sha(root / prep.C_FILES["audit"]),
        decision=decision,
    )
    write(root / prep.C_FILES["analysis"], analysis)
    arithmetic = dict(
        parent_manifest_sha256=manifest_sha,
        source_sha256=source,
        archive_byte_sha256=archive_sha,
        passed=True,
        assigned=4320,
        audit_sha256=digest(audit),
        analysis_sha256=digest(analysis),
        audit_byte_sha256=prep.sha(root / prep.C_FILES["audit"]),
        analysis_byte_sha256=prep.sha(root / prep.C_FILES["analysis"]),
        decision=decision,
    )
    write(root / prep.C_FILES["arithmetic"], arithmetic)
    scoring = dict(
        parent_manifest_sha256=manifest_sha,
        source_sha256=source,
        source_archive_sha256=archive_sha,
        freeze_byte_sha256=prep.sha(root / prep.C_FILES["freeze"]),
        passed=True,
        complete=True,
        response_scoring_complete=True,
        computation_agrees=True,
        assigned=4320,
        checked=4320,
        mismatches=[],
    )
    write(root / prep.C_FILES["scoring"], scoring)
    write(
        root / prep.C_FILES["shutdown"],
        dict(all_stopped_zero_tasks=True, apps=[{"State": "stopped", "Tasks": "0"}]),
    )


@pytest.fixture
def closed_root(tmp_path, monkeypatch):
    closure(tmp_path, monkeypatch)
    return tmp_path


@pytest.fixture
def source_tree(closed_root):
    root = closed_root
    shutil.copytree(ROOT / prep.REFERENCE, root / prep.REFERENCE)
    shutil.copytree(ROOT / prep.ASSETS / "metadata", root / prep.ASSETS / "metadata")
    files = [
        prep.PLAN,
        prep.ADAPTATION,
        prep.LOCK,
        "configs/pain_profile_requirements.in",
        *[
            prep.ASSETS + "/" + name
            for name in (
                "asset-receipt.json",
                "base-weight-pins.json",
                "base-generation-config.json",
                "base-generation-config-receipt.json",
            )
        ],
    ]
    files += [row["path"] for row in json.loads((ROOT / prep.ADAPTATION).read_text())["files"]]
    for name in files:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
    # This temporary fixture is synthetic: live engine work may legitimately
    # invalidate the old real review receipt until its owner re-reviews/releases.
    adaptation = json.loads((root / prep.ADAPTATION).read_text())
    adaptation["purpose"] = "synthetic test receipt; never model admission"
    for row in adaptation["files"]:
        row["byte_sha256"] = prep.sha(root / row["path"])
    write(root / prep.ADAPTATION, adaptation)
    return root


def test_closure_retains_failed_readiness_and_does_not_gate_effect(closed_root, monkeypatch):
    receipt = prep.closed_c_evidence(closed_root)
    assert receipt["passed"] and receipt["prior_readiness"]["passed"] is False
    assert receipt["source_archive"]["copied_into_campaign"] is False
    # The method benchmark is not a rescue selected by a C threshold.
    other = closed_root / "other"
    closure(other, monkeypatch, positive=True)
    assert prep.closed_c_evidence(other)["prior_decision"]["positive_screen"] is True


@pytest.mark.parametrize(
    "key,field,value",
    [
        ("audit", "complete", False),
        ("audit", "integrity_passed", False),
        ("analysis", "audit_sha256", "0" * 64),
        ("arithmetic", "passed", False),
        ("arithmetic", "analysis_byte_sha256", "0" * 64),
        ("scoring", "checked", 4319),
        ("scoring", "response_scoring_complete", False),
        ("scoring", "mismatches", ["bad"]),
        ("scoring", "freeze_byte_sha256", "0" * 64),
        ("shutdown", "all_stopped_zero_tasks", False),
        ("shutdown", "apps", [{"State": "running", "Tasks": "1"}]),
    ],
)
def test_closed_evidence_cannot_be_admitted_by_unbound_pass_flag(closed_root, key, field, value):
    path = closed_root / prep.C_FILES[key]
    doc = json.loads(path.read_text())
    doc[field] = value
    write(path, doc)
    with pytest.raises(ValueError):
        prep.closed_c_evidence(closed_root)


def test_closed_archive_bytes_rechecked(closed_root):
    (closed_root / prep.C_RUN / "source.tar.gz").write_bytes(b"edited")
    with pytest.raises(ValueError, match="archive changed"):
        prep.closed_c_evidence(closed_root)


def test_exact_small_document_inventory_excludes_all_weight_payloads(source_tree):
    # A large local adapter file is deliberately irrelevant to source copying.
    archive = source_tree / prep.ASSETS / prep.assets.ADAPTER_NAME
    archive.write_bytes(b"not loaded or copied")
    documents = prep.collect_documents(source_tree)
    assert prep.ASSETS + "/asset-receipt.json" in documents
    assert prep.TOKENIZER + "/tokenizer.json" in documents
    assert prep.REFERENCE + "/upstream/LICENSE" in documents
    assert prep.C_RUN + "/source.tar.gz" not in documents
    assert not any(name.endswith(prep.assets.ADAPTER_NAME) for name in documents)
    assert len([name for name in documents if name.startswith(prep.ASSETS + "/metadata/")]) == 6
    assert all(prep.sha(source_tree / name) == value for name, value in documents.items())


@pytest.mark.parametrize(
    "name",
    [
        prep.TOKENIZER + "/tokenizer.json",
        prep.ASSETS + "/base-weight-pins.json",
        prep.REFERENCE + "/upstream/LICENSE",
        "src/curiosity_study/pain_profile_engine.py",
    ],
)
def test_modified_pinned_inputs_or_reviewed_engine_rejected(source_tree, name):
    with (source_tree / name).open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(ValueError):
        prep.collect_documents(source_tree)


def test_unlisted_metadata_is_not_silently_frozen(source_tree):
    (source_tree / prep.TOKENIZER / "extra.json").write_text("{}")
    with pytest.raises(ValueError, match="Unexpected or omitted retained"):
        prep.collect_documents(source_tree)


def test_linked_document_rejected_even_if_bytes_match(source_tree, tmp_path):
    path = source_tree / prep.TOKENIZER / "adapter_config.json"
    outside = tmp_path / "external.json"
    path.rename(outside)
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="linked"):
        prep.collect_documents(source_tree)


@pytest.mark.parametrize("name", ["../escape", "/etc/hosts", "a/../b", "./file"])
def test_document_paths_cannot_escape_root(tmp_path, name):
    with pytest.raises(ValueError, match="Unsafe"):
        prep.safe_file(tmp_path, name)


def test_readonly_build_binds_full_grid_resources_and_cpu_verification(source_tree, monkeypatch):
    events = []
    monkeypatch.setattr(
        prep.assets,
        "check_environment",
        lambda p: dict(lock_byte_sha256=prep.sha(p), distributions={"synthetic": "test"}),
    )

    def vectors(root, device):
        assert root == source_tree and device == "cpu"
        events.append("vectors_cpu")
        return {}, {"pain": {"fp32_tensor_sha256": "a" * 64, "norm_fp32": 143.8}}

    monkeypatch.setattr(prep.assets, "load_vectors", vectors)
    tokenizer = Tokenizer()
    tokenizer.chat_template = (source_tree / prep.TOKENIZER / "chat_template.jinja").read_text()
    before = {str(p.relative_to(source_tree)) for p in source_tree.rglob("*")}
    manifest = prep.build_manifest(source_tree, tokenizer=tokenizer)
    assert before == {str(p.relative_to(source_tree)) for p in source_tree.rglob("*")}
    assert events == ["vectors_cpu"]
    assert len(manifest["plan"]["assignments"]) == 2020
    assert len(manifest["preflight"]["requests"]) == 404
    assert manifest["setup_call_plan_sha256"] == digest(manifest["setup_call_plan"])
    assert len(manifest["setup_call_plan"]) == 21
    assert sum(len(row["trial_ids"]) for row in manifest["setup_call_plan"]) == 612
    assert [row["inject_failure_at"] for row in manifest["setup_call_plan"] if row["expected_error"]] == [
        "after_prefill",
        "after_first_decode",
    ]
    assert manifest["plan_sha256"] == digest(manifest["plan"])
    assert manifest["preflight_sha256"] == digest(manifest["preflight"])
    assert manifest["campaign_ceiling_usd"] == 12
    assert {key: value["wall_seconds"] for key, value in manifest["resources"].items()} == {
        "prefetch": 1800,
        "setup": 1800,
        "benchmark": 3600,
    }
    assert manifest["scientific_admission"] is False and manifest["admitted_phases"] == []
    assert manifest["prior_study_closure"]["prior_readiness"]["passed"] is False


@pytest.mark.parametrize("existing", ["run", "execution", "alias"])
def test_existing_campaign_rejected_before_any_build(tmp_path, monkeypatch, existing):
    if existing == "run":
        (tmp_path / prep.RUN).mkdir(parents=True)
    elif existing == "execution":
        (tmp_path / "executions" / Path(prep.RUN).name).mkdir(parents=True)
    else:
        write(tmp_path / "runs/alias/manifest.json", dict(assay=prep.resources.ASSAY))
    monkeypatch.setattr(prep, "build_manifest", lambda *a, **k: pytest.fail("build should not run"))
    with pytest.raises(ValueError):
        prep.prepare(root=tmp_path)


def test_invalid_inputs_leave_no_campaign_or_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(
        prep, "build_manifest", lambda *a, **k: (_ for _ in ()).throw(ValueError("bad input"))
    )
    with pytest.raises(ValueError, match="bad input"):
        prep.prepare(root=tmp_path)
    assert not (tmp_path / prep.RUN).exists() and not (tmp_path / "executions").exists()


def test_late_source_change_rejected_before_first_write(tmp_path, monkeypatch):
    manifest = dict(source_sha256="original", documents={"doc": "hash"})
    monkeypatch.setattr(prep, "build_manifest", lambda *a, **k: deepcopy(manifest))
    monkeypatch.setattr(prep, "source_hash", lambda root: "changed")
    with pytest.raises(ValueError, match="Inputs changed"):
        prep.prepare(root=tmp_path)
    assert not (tmp_path / prep.RUN).exists()


def test_success_calls_generic_freezer_only_after_immutable_manifest(tmp_path, monkeypatch):
    manifest = dict(source_sha256="source", documents={"doc": "hash"})
    monkeypatch.setattr(prep, "build_manifest", lambda *a, **k: deepcopy(manifest))
    monkeypatch.setattr(prep, "source_hash", lambda root: "source")
    monkeypatch.setattr(prep, "collect_documents", lambda root: {"doc": "hash"})

    def fake_freeze(root, out, documents):
        assert json.loads((out / "manifest.json").read_text()) == manifest
        assert documents == ("doc",)
        return dict(synthetic=True)

    monkeypatch.setattr(prep, "freeze", fake_freeze)
    result = prep.prepare(root=tmp_path)
    assert result["cloud_launched"] is False and result["paid_phases_admitted"] == []


def test_freezer_failure_is_retained_and_cannot_silently_resume(tmp_path, monkeypatch):
    manifest = dict(source_sha256="source", documents={})
    monkeypatch.setattr(prep, "build_manifest", lambda *a, **k: deepcopy(manifest))
    monkeypatch.setattr(prep, "source_hash", lambda root: "source")
    monkeypatch.setattr(prep, "collect_documents", lambda root: {})
    monkeypatch.setattr(prep, "freeze", lambda *a, **k: (_ for _ in ()).throw(OSError("disk failed")))
    with pytest.raises(OSError, match="disk failed"):
        prep.prepare(root=tmp_path)
    assert (tmp_path / prep.RUN / "manifest.json").is_file()
    with pytest.raises(ValueError, match="cannot be replaced"):
        prep.prepare(root=tmp_path)
