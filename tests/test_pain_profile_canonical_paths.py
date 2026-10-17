"""Reporting-only correction; synthetic outcomes and no paid model execution."""

from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from curiosity_study.artifacts import digest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("canonical_path_test", ROOT / "scripts/audit_pain_profile_control_canonical_paths.py")
wrapper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wrapper)

RAW = {"./adapter/config.json": "a" * 64, "./adapter/weights.safetensors": "b" * 64}
CANONICAL = {"adapter/config.json": "a" * 64, "adapter/weights.safetensors": "b" * 64}


def test_only_leading_dot_changes_and_hashes_remain_identical():
    assert wrapper.canonical_adapter_files(RAW) == CANONICAL
    assert wrapper.canonical_adapter_files(CANONICAL) == CANONICAL
    assert RAW == {"./adapter/config.json": "a" * 64, "./adapter/weights.safetensors": "b" * 64}


@pytest.mark.parametrize("names", [
    {"./a": "a" * 64, "a": "a" * 64},
    {"../a": "a" * 64}, {"/a": "a" * 64}, {"./../a": "a" * 64},
    {"a/../b": "a" * 64}, {"a//b": "a" * 64}, {"././a": "a" * 64},
    {"a/./b": "a" * 64}, {"a/": "a" * 64}, {".": "a" * 64},
    {"./a": "changed-content"}, {},
])
def test_collision_traversal_or_other_normalization_rejected(names):
    with pytest.raises(ValueError):
        wrapper.canonical_adapter_files(names)


def fake_original(*, fail=False):
    original = SimpleNamespace()
    payload = object()
    proof = dict(adapter_files=dict(RAW), tokenizer=payload, vectors=payload, numerical_checks={"exact": True})
    original.source_provenance = lambda run, manifest: proof
    calls = []

    def audit(run, phase):
        calls.append(phase)
        observed = original.source_provenance(run, {})
        assert observed["adapter_files"] == CANONICAL
        assert observed["tokenizer"] is observed["vectors"] is payload
        assert observed["numerical_checks"] is proof["numerical_checks"]
        if phase == "benchmark":
            # This is the frozen admission_binding recursion contract: compare
            # the ENTIRE saved setup receipt, including reporting metadata.
            assert original.audit(run, "setup") == json.loads((run / "setup.json").read_text())
        return dict(phase=phase, complete=not fail, passed=not fail, exact_repeat=not fail,
                    missing_rows=320 if fail else 0, original_auditor_byte_sha256=wrapper.AUDITOR_SHA)
    original.audit = audit
    return original, proof, calls


def test_recursive_benchmark_setup_reproduction_has_identical_wrapper_schema(tmp_path):
    original, proof, calls = fake_original()
    original_source = original.source_provenance
    metadata = dict(correction="synthetic", amendment_byte_sha256="c" * 64)
    corrected = wrapper.attach_correction(original, RAW, metadata)
    setup = corrected(tmp_path, "setup")
    (tmp_path / "setup.json").write_text(json.dumps(setup))
    benchmark = corrected(tmp_path, "benchmark")
    assert benchmark["passed"] and benchmark["reporting_correction"] == setup["reporting_correction"] == metadata
    assert calls == ["setup", "benchmark", "setup"]
    assert proof["adapter_files"] == RAW and original_source(None, None)["adapter_files"] == RAW


def test_failed_repeat_and_all_missing_rows_are_preserved_exactly(tmp_path):
    original, _, _ = fake_original(fail=True)
    corrected = wrapper.attach_correction(original, RAW, dict(correction="synthetic"))
    result = corrected(tmp_path, "setup")
    assert not result["passed"] and not result["complete"] and not result["exact_repeat"]
    assert result["missing_rows"] == 320


@pytest.mark.parametrize("change", ["hash", "name", "extra"])
def test_wrapper_does_not_tolerate_nonpath_inventory_changes(tmp_path, change):
    original, proof, _ = fake_original()
    corrected = wrapper.attach_correction(original, RAW, {})
    if change == "hash":
        proof["adapter_files"]["./adapter/config.json"] = "f" * 64
    elif change == "name":
        proof["adapter_files"]["./other/config.json"] = proof["adapter_files"].pop("./adapter/config.json")
    else:
        proof["adapter_files"]["./adapter/extra"] = "e" * 64
    with pytest.raises(ValueError, match="Unanticipated"):
        corrected(tmp_path, "setup")


def test_real_frozen_seven_file_receipt_agrees_without_reading_setup(tmp_path):
    # Read-only original source + CPU prefetch metadata. No setup/benchmark raw
    # artifacts are loaded by amendment construction or this test.
    expected = wrapper.amendment_payload()
    assert expected["member_count"] == len(expected["members"]) == 7
    assert expected["manifest_sha256"] == wrapper.MANIFEST_SHA
    assert expected["source_archive_sha256"] == wrapper.ARCHIVE_SHA
    assert all(row["raw_path"] == "./" + row["canonical_path"]
               and row["original_byte_sha256"] == row["cpu_byte_sha256"]
               for row in expected["members"])
    assert not expected["execution_or_scoring_changed"] and not expected["numerical_checks_changed"]
    assert "no claim of preceding measurements" in expected["timing"]
    path = tmp_path / "amendment.json"
    path.write_text(json.dumps(expected))
    original_sys_path, original_bytecode = list(sys.path), sys.dont_write_bytecode
    previous = sys.modules.get("analyze_pain_profile_control")
    with wrapper.corrected_runtime(amendment=path) as (corrected, analysis, metadata):
        assert callable(corrected)
        assert Path(analysis.__file__) == ROOT / "executions/qwen25-pain-profile-control-v1" / wrapper.ANALYZER
        assert metadata["member_count"] == 7
        assert sys.dont_write_bytecode
    assert sys.path == original_sys_path and sys.dont_write_bytecode == original_bytecode
    assert sys.modules.get("analyze_pain_profile_control") is previous
    changed = deepcopy(expected)
    changed["members"][0]["cpu_byte_sha256"] = "f" * 64
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="amendment differs"):
        wrapper.validate_amendment(wrapper.RUN, path)


@pytest.mark.parametrize("failure", ["incomplete", "integrity"])
def test_corrected_analysis_keeps_authentic_failures_unavailable(tmp_path, monkeypatch, failure):
    spec = importlib.util.spec_from_file_location("original_analysis_test", ROOT / "executions/qwen25-pain-profile-control-v1" / wrapper.ANALYZER)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    from curiosity_study import pain_profile_design as design
    plan = design.assignment_plan(design.load_sources(ROOT / "references/pain-axis-positive-control"))
    rows = [dict(**{k: a[k] for k in original.METADATA}, status="completed", category="other", valid=True, cap_hit=False)
            for a in plan["assignments"]]
    if failure == "incomplete":
        rows[0].update(status="missing", category=None, valid=None, cap_hit=None)
    manifest = dict(plan=plan, source_sha256=wrapper.SOURCE_SHA)
    receipt = dict(phase="benchmark", manifest_sha256=digest(manifest), assigned_rows=2020,
                   outcomes=rows, complete=failure != "incomplete", passed=False,
                   archive_byte_sha256=wrapper.ARCHIVE_SHA, freeze_byte_sha256="f" * 64)
    metadata = dict(correction="synthetic")
    receipt["reporting_correction"] = metadata
    path = tmp_path / "phases/benchmark/audit.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(receipt))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    @contextmanager
    def fake_runtime(*args):
        yield lambda run, phase: deepcopy(receipt), original, metadata
    monkeypatch.setattr(wrapper, "corrected_runtime", fake_runtime)
    result = wrapper.analyze_run(tmp_path)
    assert result["assigned"] == len(result["outcomes"]) == 2020
    assert not result["integrity_passed"]
    assert result["decision"]["full_mask_positive_control"]["classification"] == "incomplete_or_integrity_failed"
    assert not result["decision"]["schedule_comparison"]["appreciable_difference"]
    assert result["reporting_correction"] == metadata
    changed = deepcopy(receipt)
    changed["passed"] = True
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="does not reproduce"):
        wrapper.analyze_run(tmp_path)
