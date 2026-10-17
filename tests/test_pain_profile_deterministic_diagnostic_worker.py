"""CPU-only diagnostic boundaries; no weights, admission or cloud operations."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from curiosity_study import pain_profile_engine as engine
from curiosity_study import pain_profile_execution as execution
from test_pain_profile_engine import collect, items

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/pain_profile_deterministic_diagnostic_worker.py"
spec = importlib.util.spec_from_file_location("deterministic_worker_tested", PATH)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


@pytest.fixture(autouse=True)
def settings(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    previous = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
                torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
                torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                torch.get_float32_matmul_precision(), torch.get_num_threads())
    torch.set_num_threads(1)
    yield
    torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = previous[2:4]
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous[4:6]
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous[6]
    torch.set_float32_matmul_precision(previous[7])
    torch.set_num_threads(previous[8])


@pytest.fixture(scope="module")
def original():
    # Immutable input metadata only; never reads earlier responses or evidence.
    return json.loads((ROOT / "runs/qwen25-pain-profile-control-v1/manifest.json").read_text())


def test_module_import_is_stdlib_only_before_cublas_environment():
    code = ("import importlib.util,sys; p=" + repr(str(PATH)) + "; "
            "s=importlib.util.spec_from_file_location('pure_worker',p);m=importlib.util.module_from_spec(s);"
            "s.loader.exec_module(m);assert 'torch' not in sys.modules;assert 'transformers' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True, timeout=10)


def test_fixed_original_inventory_has148_slots_without_rebatching(original):
    calls = worker.call_plan(original)
    assert [len(c["trial_ids"]) for c in calls] == [32, 32, 20, 32, 32]
    assert [c["source_cohort"] for c in calls] == [0, 0, 12, 0, 0]
    assert [c["steering_enabled"] for c in calls] == [False, True, True, False, True]
    assert [c["coefficient"] for c in calls] == [0, 0, 1, 0, 0]
    assert len({t for c in calls for t in c["trial_ids"]}) == 148
    assert sum(len(c["baseline_reference_call_ids"]) for c in calls) == 6
    assert all(c["base_request_ids"] == calls[0]["base_request_ids"] for c in (calls[1], calls[3], calls[4]))
    changed = deepcopy(original)
    changed["preflight"]["longest_source_cohort"] = 11
    with pytest.raises(ValueError, match="longest"):
        worker.call_plan(changed)


def test_strict_fixed_flags_and_only_math_backend_before_loading(monkeypatch):
    assert not torch.cuda.is_initialized()
    with worker.deterministic_runtime(torch) as flags:
        assert flags["deterministic_algorithms"] is True and flags["deterministic_warn_only"] is False
        assert flags["sdpa_backends"] == {"math": True, "flash": False, "mem_efficient": False, "cudnn": False}
        assert flags["cudnn_deterministic"] is True and flags["cudnn_benchmark"] is False
        assert flags["allow_bf16_reduced_precision_reduction"] is True
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "bad")
    with pytest.raises(ValueError, match="precede CUDA"):
        with worker.deterministic_runtime(torch):
            pytest.fail("bad environment admitted")


def test_already_initialized_cuda_rejected_before_flag_changes(monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda, "is_initialized", lambda: True)
        patch.setattr(torch, "use_deterministic_algorithms", lambda *a, **k: pytest.fail("too late to configure"))
        with pytest.raises(ValueError, match="precede CUDA"):
            with worker.deterministic_runtime(torch):
                pytest.fail("initialized CUDA admitted")


def test_cublas_environment_is_checked_before_loading_original_helpers(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
    monkeypatch.setattr(worker.importlib, "import_module", lambda *a: pytest.fail("numerical import before env guard"))
    with pytest.raises(ValueError, match="before process start"):
        worker.load_original(Path("/unused"), {})


def test_file_binding_rejects_changed_bytes_and_equal_byte_symlink(tmp_path):
    path = tmp_path / "held.py"
    path.write_text("fixed")
    pins = {path.name: worker.sha(path)}
    worker.verify_files(tmp_path, pins)
    path.write_text("changed")
    with pytest.raises(ValueError, match="file changed"):
        worker.verify_files(tmp_path, pins)
    target = tmp_path / "other.py"
    target.write_text("fixed")
    path.unlink()
    path.symlink_to(target)
    with pytest.raises(ValueError, match="file changed"):
        worker.verify_files(tmp_path, pins)


@pytest.fixture(scope="module")
def real_vectors():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return collect(), torch.arange(-4, 4, dtype=torch.bfloat16) / 8
    finally:
        torch.set_num_threads(old)


def test_real_bf16_samples_and_timeline_with_descriptive_norm_disagreement(real_vectors):
    source, direction = real_vectors
    result = deepcopy(source)
    for frame in result["evidence"]["forwards"]:
        for row in frame["geometry"]:
            row["achieved_delta_norm_min"] += 0.01
            row["achieved_delta_norm_max"] += 0.01
    values = worker.vector_checks(torch, result, direction, items())
    assert len(values) == 16 and all(v["cross_device_norms_used_as_gate"] is False for v in values)
    assert any(v["gpu_fp32_norm_min"] != v["cpu_float64_norm_min"] for v in values)
    assert worker.timeline_checks(result, items(), {"pad_token_id": 0, "eos_token_ids": []})


@pytest.mark.parametrize("change", ["drop_samples", "content", "hash", "mask", "token_timeline"])
def test_real_saved_evidence_cannot_be_omitted_or_falsified(real_vectors, change):
    source, direction = real_vectors
    result = deepcopy(source)
    if change == "drop_samples":
        result["arrays"] = {}
        result["evidence"]["arrays"] = {}
        for frame in result["evidence"]["forwards"]:
            for row in frame["geometry"]:
                row.pop("saved_before", None)
                row.pop("saved_after", None)
    elif change == "content":
        result["arrays"]["f00_r00_after"][0, 0] += 1
    elif change == "hash":
        result["evidence"]["forwards"][0]["geometry"][0]["before_sha256"] = "0" * 64
    elif change == "mask":
        result["evidence"]["forwards"][0]["geometry"][0]["active_positions"] = 0
    else:
        result["evidence"]["forwards"][1]["input_ids"][0][0] += 1
        with pytest.raises(ValueError, match="timeline"):
            worker.timeline_checks(result, items(), {"pad_token_id": 0, "eos_token_ids": []})
        return
    with pytest.raises(ValueError):
        worker.vector_checks(torch, result, direction, items())


def fake_result(rows, coefficient, enabled, preflight, direction, *, no_decode=False):
    width = max(len(row["prompt_ids"]) for row in rows)
    pad, eos = preflight["pad_token_id"], preflight["eos_token_ids"][0]
    masks = [[0] * (width - len(r["prompt_ids"])) + [1] * len(r["prompt_ids"]) for r in rows]
    ids = [[pad] * (width - len(r["prompt_ids"])) + r["prompt_ids"] for r in rows]
    positions = [[0] * (width - len(r["prompt_ids"])) + list(range(len(r["prompt_ids"]))) for r in rows]
    frames, arrays = [], {}
    for step in range(1 if no_decode else 2):
        columns = width if step == 0 else 1
        frame = dict(forward_index=step, phase="prefill" if step == 0 else "decode", input_ids=ids,
            attention_mask=masks, position_ids=positions, cache_length_before=0 if step == 0 else width,
            cache_length_after=width + step, full_fp32_logits_sha256=["1" * 64] * len(rows),
            prepatch_pain_projection=[1.] * len(rows), post_layer61_pain_projection=[2.] * len(rows),
            activation_dtypes={"before": "torch.bfloat16", "after": "torch.bfloat16"}, geometry=[])
        for i, row in enumerate(rows):
            before = torch.ones(columns, direction.numel(), dtype=torch.bfloat16) * (i + 1)
            active = torch.ones(columns, dtype=torch.bfloat16)
            padding = width - len(row["prompt_ids"]) if step == 0 else 0
            active[:padding] = 0
            after = before if coefficient == 0 else before + active[:, None] * coefficient * direction
            geometry = dict(trial_id=row["trial_id"], coefficient=coefficient,
                active_positions=int(active.sum()) if coefficient else 0,
                unmodified_positions=columns - (int(active.sum()) if coefficient else 0), padding_positions=padding,
                before_sha256=worker.digest(engine.tensor_binding(before)), after_sha256=worker.digest(engine.tensor_binding(after)),
                achieved_delta_norm_min=float(direction.float().norm()) if coefficient else 0.0,
                achieved_delta_norm_max=float(direction.float().norm()) if coefficient else 0.0)
            if i in (0, len(rows) - 1):
                for label, value in (("before", before), ("after", after)):
                    key = f"f{step:02d}_r{i:02d}_{label}"
                    arrays[key] = value.float().numpy().copy()
                    geometry["saved_" + label] = dict(key=key, positions=list(range(columns)), **engine.array_binding(arrays[key]))
            frame["geometry"].append(geometry)
        selected = [eos if no_decode or step else 7] * len(rows)
        frame["sampling"] = dict(alive_before=[True] * len(rows), alive_after=[not(step or no_decode)] * len(rows),
            uniform_draws_per_row=[1] * len(rows), rng_state_before_sha256=[str(step) * 64] * len(rows),
            rng_state_after_sha256=[str(step + 1) * 64] * len(rows), sampled_before_alive_mask=selected,
            selected_token_ids=selected, uniforms=[0.5] * len(rows))
        frames.append(frame)
        ids, masks = [[token] for token in selected], [mask + [1] for mask in masks]
        positions = [[len(row["prompt_ids"])] for row in rows]
    records = [dict(trial_id=row["trial_id"], answer_token_ids=[eos] if no_decode else [7, eos],
                    text="synthetic raw response", mean_proj=1.0) for row in rows]
    return dict(records=records, arrays=arrays, evidence=dict(complete=True, integrity_passed=True,
        coefficient=coefficient, steering_enabled=enabled, prefill_policy="full_prefix", evidence_mode="compatibility",
        items_sha256=worker.digest(rows), hooks_before={}, hooks_after={}, checks={"exact_geometry": True}, forwards=frames,
        arrays={key: engine.array_binding(value) for key, value in arrays.items()}))


@pytest.fixture
def run_context(tmp_path, monkeypatch, original):
    original = deepcopy(original)
    run = tmp_path / worker.NAME
    run.mkdir()
    manifest = {"original_manifest": original, "call_plan": worker.call_plan(original), "preset": deepcopy(worker.PRESET)}
    admission = {"created": 0, "deadline": 900, "manifest_sha256": worker.digest(manifest)}
    worker.write_once(run / "manifest.json", manifest)
    scenario = dict(fail_call=None, changed_hash_call=None, no_decode=False, load_failure=False, clock=100)
    events = []
    direction = torch.tensor([0.0, 0.25, 0.5, 0.75], dtype=torch.bfloat16)
    vectors = {"pain": direction, "pain_unit_float32": direction.float()}
    monkeypatch.setattr(worker, "validate_payload", lambda *a: (manifest, admission))
    monkeypatch.setattr(worker, "time", SimpleNamespace(time=lambda: scenario["clock"], monotonic=lambda: 0.0))

    def load(root, cache, deadline):
        flags = worker.effective_preset(torch)
        assert flags["deterministic_algorithms"] and flags["sdpa_backends"]["math"]
        assert not flags["sdpa_backends"]["flash"] and deadline == 840
        events.append("load")
        if scenario["load_failure"]:
            raise RuntimeError("synthetic load OOM")
        return None, None, None, None, vectors, {"environment": {k: v for k, v in original["environment_lock"].items() if k != "path"}}

    def generate(*args, **kw):
        call_index = sum(isinstance(event, tuple) for event in events)
        events.append((call_index, len(kw["items"]), kw["coefficient"], kw["steering_enabled"]))
        if call_index == scenario["fail_call"]:
            raise RuntimeError("synthetic unsupported deterministic operation")
        result = fake_result(kw["items"], kw["coefficient"], kw["steering_enabled"], original["preflight"], direction,
                             no_decode=scenario["no_decode"])
        if call_index == scenario["changed_hash_call"]:
            # Preserve raw outputs and full-logit evidence while corrupting a nonsampled row's prehook state.
            result["evidence"]["forwards"][1]["geometry"][1]["before_sha256"] = "9" * 64
        return result

    modules = dict(pain_profile_assets=SimpleNamespace(load_model=load),
        pain_profile_engine=SimpleNamespace(generate_segment=generate, array_binding=engine.array_binding),
        pain_profile_execution=SimpleNamespace(validate_manifest=lambda *a: None, validate_runtime=lambda *a: None,
                                               repeat_equal=execution.repeat_equal, active_decode=execution.active_decode))
    monkeypatch.setattr(worker, "load_original", lambda *a: (torch, modules))
    return run, manifest, admission, scenario, events


def test_complete_fake_engine_five_calls_bound_arrays_and_no_scoring(run_context):
    run, manifest, _, _, events = run_context
    summary = worker.execute(run / "original", run, run / "cache")
    assert summary["passed"] is summary["complete"] is True
    assert summary["assigned_rows"] == summary["completed_rows"] == 148
    assert summary["behavior_scoring_performed"] is summary["scientific_admission"] is False
    assert events == ["load", (0, 32, 0, False), (1, 32, 0, True), (2, 20, 1, True), (3, 32, 0, False), (4, 32, 0, True)]
    last = worker.read(run / "calls/0004.json")
    assert len([k for k in last["checks"] if k.startswith("source_repeat:")]) == 3
    assert len([k for k in last["checks"] if k.startswith("before_hook_identity:")]) == 3
    assert last["arrays_artifact"]["byte_sha256"] == worker.sha(run / "calls/0004.npz")
    assert len(worker.read(run / "outcomes.json")) == 148
    assert worker.read(run / "started.json")["execution_deadline"] == 840
    with pytest.raises(ValueError, match="no retry"):
        worker.execute(run / "original", run, run / "cache")
    assert worker.read(run / "manifest.json") == manifest


@pytest.mark.parametrize("failure,observed,complete,missing", [
    ("load_failure", 0, 0, 148), ("fail_call", 3, 64, 64),
    ("changed_hash_call", 4, 116, 32), ("no_decode", 1, 32, 116), ("deadline", 0, 0, 148),
])
def test_terminal_failures_preserve_every_assignment_without_retry(run_context, failure, observed, complete, missing):
    run, _, _, scenario, events = run_context
    if failure == "deadline":
        scenario["clock"] = 841
    else:
        scenario[failure] = 2 if failure == "fail_call" else 3 if failure == "changed_hash_call" else True
    summary = worker.execute(run / "original", run, run / "cache")
    assert summary["passed"] is False and summary["observed_calls"] == observed
    assert summary["completed_rows"] == complete and summary["missing_rows"] == missing
    assert summary["assigned_rows"] == 148 and len(worker.read(run / "outcomes.json")) == 148
    assert len([event for event in events if isinstance(event, tuple)]) == observed
    assert len(list((run / "calls").glob("*.json"))) == observed
    if failure == "changed_hash_call":
        call = worker.read(run / "calls/0003.json")
        assert call["checks"]["source_repeat:deterministic-0000"] is True
        assert call["checks"]["before_hook_identity:deterministic-0000"] is False
    if failure == "fail_call":
        assert summary["execution_error_rows"] == 20
        assert worker.read(run / "calls/0002.json")["error"]["type"] == "RuntimeError"


def test_commit_failure_retains_call_and_terminal_summary(run_context):
    run, _, _, _, _ = run_context
    count = 0

    def checkpoint():
        nonlocal count
        count += 1
        if count == 1:
            raise OSError("synthetic checkpoint failure")

    summary = worker.execute(run / "original", run, run / "cache", checkpoint)
    assert summary["passed"] is False and summary["completed_rows"] == 32 and summary["missing_rows"] == 116
    assert summary["error"]["type"] == "OSError" and count == 2
