import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from curiosity_study import information_state as state
from curiosity_study import thinking_information as thinking
from curiosity_study.artifacts import digest, read


def script_module(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_legacy_request_and_record_keep_the_original_default_and_schema():
    assignment = state.assignments(1, kinds=("new",))[0]
    old_request = {**state.inference_request(assignment), "enable_thinking": True, "max_new_tokens": 2048}
    assert thinking.inference_request(assignment) == old_request
    inference = dict(raw="reason </think> Yes", generated_token_ids=[2, 3], ended_with_eos=True)
    legacy = thinking.record_from_inference(assignment, inference)
    assert "generation_max_new_tokens" not in legacy and "cap_hit" not in legacy
    explicit = thinking.record_from_inference({**assignment, "max_new_tokens": 4096}, inference)
    assert explicit["generation_max_new_tokens"] == 4096 and not explicit["cap_hit"]
    assert {k: explicit[k] for k in ("valid", "correct", "parsed_answer", "thinking_closed")} == {
        k: legacy[k] for k in ("valid", "correct", "parsed_answer", "thinking_closed")
    }


@pytest.mark.parametrize("limit", [None, True, 4096.0, "4096", 1024, 4097])
def test_generation_limit_rejects_unfrozen_or_ambiguous_caps(limit):
    with pytest.raises(ValueError, match="2048 or 4096"):
        thinking.inference_request({**state.assignments(1, kinds=("new",))[0], "max_new_tokens": limit})


def test_manifest_requires_the_same_explicit_allowance_across_all_kinds():
    rows = [{**a, "max_new_tokens": 4096} for a in state.assignments(1)]
    manifest = dict(assignments=[], diagnostics=rows, max_new_tokens=4096)
    assert thinking.manifest_generation_limit(manifest) == 4096
    for kind in {a["kind"] for a in rows}:
        bad = copy.deepcopy(manifest)
        next(a for a in bad["diagnostics"] if a["kind"] == kind).pop("max_new_tokens")
        with pytest.raises(ValueError, match="Every assigned kind"):
            thinking.manifest_generation_limit(bad)


@pytest.mark.parametrize("preparer", ["prepare_relational_smoke", "prepare_count_competence"])
def test_future_preparers_use_4096_for_every_assignment_without_writing(preparer, tmp_path, monkeypatch):
    module = script_module(preparer)
    monkeypatch.setattr(module, "freeze", lambda *args: pytest.fail("Pure builder must not freeze"))
    monkeypatch.setattr(module, "write_once", lambda *args: pytest.fail("Pure builder must not write"))
    kwargs = {"items": 2} if preparer == "prepare_count_competence" else {}
    manifest = module.build_manifest(dict(model_id="Qwen/Qwen3-32B", revision="synthetic"), **kwargs)
    assert thinking.manifest_generation_limit(manifest) == 4096
    assert manifest["inference_revision"] == "thinking-4096-v1"
    assert manifest["assignment_sha256"] == digest(manifest["diagnostics"])
    assert all(a["max_new_tokens"] == 4096 for a in manifest["diagnostics"])
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "scope,phase",
    [("confirmation", "scientific"), ("fresh_confirmation", "scientific"), ("development", "confirmation")],
)
def test_new_natural_confirmation_rejects_legacy_allowance_before_preparation(scope, phase, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = script_module("prepare_natural_interventions")
    config = dict(inference_scope=scope, phase=phase)
    monkeypatch.setattr(module, "read", lambda _: config)
    monkeypatch.setattr(module, "freeze", lambda *args: pytest.fail("Must not freeze a rejected schedule"))
    with pytest.raises(ValueError, match="require 4096"):
        module.prepare("synthetic-config")


@pytest.mark.parametrize("limit", [2048, 4096])
def test_full_thinking_audit_tracks_late_closure_and_dynamic_cap(tmp_path, monkeypatch, limit):
    from test_reasoning_trace import backend

    model = backend()
    model.metadata = lambda: dict(model="scripted-cpu")

    def encode(messages, enable_thinking=False):
        text = "\n".join(m["content"] for m in messages)
        return text, SimpleNamespace(input_ids=torch.tensor([model.tokenizer(text)["input_ids"]]))

    model.encode = encode
    late = 2050 if limit == 4096 else 20
    model.model.sequences = [[5] * late + [102, 10, 100, 127], [5] * limit]
    assignments = state.assignments(1, kinds=("new",))[:2]
    if limit == 4096:
        assignments = [{**a, "max_new_tokens": limit} for a in assignments]
    manifest = dict(
        model=dict(model_id="scripted-cpu", revision="fixture"),
        phase="engineering_smoke",
        assignments=[],
        diagnostics=assignments,
        execution=dict(batch_size=2),
        capture_plan=dict(layers=[0, 1], steps=[32, 128, 512, 2048]),
    )
    if limit == 4096:
        manifest["max_new_tokens"] = limit
    # Capture invariance has its own real-model tests; this scripted sequence is
    # consumed once so that the auditor sees a completed and a capped row.
    monkeypatch.setattr(thinking, "verify_capture", lambda *args: dict(passed=True))
    records = thinking.execute(tmp_path, manifest, model)
    audit = script_module("audit_thinking_information")
    monkeypatch.setattr(audit.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: model.tokenizer)
    result = audit.audit(tmp_path)
    assert result["max_new_tokens"] == limit and result["cap_hits"] == 1
    assert result["valid"] == 1 and result["assigned"] == 2
    closed, capped = (records[a["trial_id"]] for a in assignments)
    assert closed["thinking_closed"] and not capped["thinking_closed"]
    assert len(closed["inference"]["generated_token_ids"]) == late + 4
    assert len(capped["inference"]["generated_token_ids"]) == limit
    if limit == 4096:
        assert closed["activation_artifact"]["max_new_tokens"] == 4096
        assert capped["generation_max_new_tokens"] == 4096 and capped["cap_hit"]
        assert closed["activation_artifact"]["token_positions"][-1] > 2048

    # A rehashed envelope does not let a different actual sampling cap through.
    envelope_path = next((tmp_path / "cohorts").glob("*/*.json"))
    envelope = read(envelope_path)
    envelope["records"][0]["inference"]["sampling"]["max_new_tokens"] = 1024
    envelope["sha256"] = digest(envelope["records"])
    import json

    envelope_path.write_text(json.dumps(envelope))
    with pytest.raises(AssertionError):
        audit.audit(tmp_path)
