"""Independent raw-audit checks on real tiny BF16 forwards and tampered copies."""

from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from curiosity_study.artifacts import digest
from test_pain_profile_engine import Tokenizer, collect, items, tiny

SPEC = importlib.util.spec_from_file_location("pain_audit_test", Path(__file__).parents[1] / "scripts/audit_pain_profile_control.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


@pytest.fixture(scope="module")
def calls():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = tiny()
        return {name: collect(model, **options) for name, options in {
            "baseline": dict(coefficient=0),
            "unhooked": dict(coefficient=0, steering_enabled=False),
            "full": {}, "final": dict(prefill_policy="final_prefix"),
            "one_row": dict(one_row_only=0),
            "prefill_fault": dict(inject_failure_at="after_prefill"),
            "decode_fault": dict(inject_failure_at="after_first_decode"),
            "clean_repeat": dict(coefficient=0),
            "benchmark": dict(evidence_mode="benchmark"),
        }.items()}
    finally:
        torch.set_num_threads(previous)


def check(result, **overrides):
    parameters = dict(
        direction=torch.arange(-4, 4, dtype=torch.bfloat16) / 8,
        pain_unit=torch.arange(1, 9, dtype=torch.float32) / torch.arange(1, 9, dtype=torch.float32).norm(),
        tokenizer=Tokenizer(), vocabulary_size=32, coefficient=1,
        prefill_policy="full_prefix", eos_ids=[], pad_id=0, mode="compatibility", expected_device="cpu",
    )
    parameters.update(overrides)
    return audit.recompute_call(result, result["arrays"], items(), **parameters)


@pytest.mark.parametrize("name,options", [
    ("baseline", dict(coefficient=0)), ("unhooked", dict(coefficient=0, steering_enabled=False)),
    ("full", {}), ("final", dict(prefill_policy="final_prefix")),
    ("one_row", dict(one_row_only=0)), ("benchmark", dict(mode="benchmark")),
])
def test_actual_tiny_bf16_source_calls_recompute(calls, name, options):
    result = check(calls[name], **options)
    assert result["complete"] and result["passed"] and result["forwards"] == 8
    assert result["active_decode_observed"]
    assert len(result["outcomes"]) == 3
    assert all(row["category"] == "malformed" and row["cap_hit"] for row in result["outcomes"])


@pytest.mark.parametrize("name,fault", [("prefill_fault", "after_prefill"), ("decode_fault", "after_first_decode")])
def test_controlled_faults_are_recognized_without_fabricating_responses(calls, name, fault):
    result = check(calls[name], expected_fault=fault)
    assert result["expected_fault_observed"] and all(result["checks"].values())
    assert not result["complete"] and not result["passed"] and not result["outcomes"]


def test_unexpected_failure_stays_unavailable_with_its_partial_artifacts(calls):
    result = deepcopy(calls["decode_fault"])
    result["evidence"]["inject_failure_at"] = None
    result["evidence"]["error"] = dict(type="RuntimeError", message="unexpected model failure")
    result["evidence"]["forwards"][-1].pop("prepatch_pain_projection")
    checked = check(result)
    assert not checked["passed"] and checked["partial_numerical_evidence_unavailable"]
    assert checked["outcomes"] == []


def test_unrelated_failure_inside_scheduled_fault_is_not_controlled_success(calls):
    result = deepcopy(calls["decode_fault"])
    result["evidence"]["error"] = dict(type="RuntimeError", message="unrelated model failure")
    result["evidence"]["forwards"][-1].pop("prepatch_pain_projection")
    checked = check(result, expected_fault="after_first_decode")
    assert not checked["passed"] and not checked["expected_fault_observed"]
    assert checked["partial_numerical_evidence_unavailable"] and checked["outcomes"] == []


@pytest.mark.parametrize("left,right,rows", [
    ("baseline", "unhooked", [0, 1, 2]),
    ("baseline", "clean_repeat", [0, 1, 2]),
    ("baseline", "one_row", [1, 2]),
])
def test_actual_repeat_sham_and_untouched_row_full_vocab_identity(calls, left, right, rows):
    assert audit.compare_rows(calls[left], calls[right], rows)


def test_same_generated_tokens_do_not_hide_changed_full_vocabulary(calls):
    changed = deepcopy(calls["unhooked"])
    changed["evidence"]["forwards"][2]["full_fp32_logits_sha256"][1] = "0" * 64
    with pytest.raises(ValueError, match="full-vocabulary"):
        audit.compare_rows(calls["baseline"], changed, [1])


@pytest.mark.parametrize("mutation,message", [
    ("mask", "timeline"), ("cache", "timeline"), ("position", "timeline"),
    ("rng", "RNG chain"), ("terminal", "termination"),
    ("count", "position count"), ("vocab", "vocabulary"),
    ("dtype", "BF16"), ("text", "text/tokens"),
    ("sampled_record", "text/tokens"), ("flag", "checks differ"),
    ("hook_count", "observed hooks"), ("extra_array", "array inventory"),
])
def test_raw_evidence_tampering_fails_independently(calls, mutation, message):
    result = deepcopy(calls["full"])
    report = result["evidence"]
    frame = report["forwards"][1]
    if mutation == "mask":
        frame["attention_mask"][0][-1] = 0
    elif mutation == "cache":
        frame["cache_length_before"] += 1
    elif mutation == "position":
        frame["position_ids"][0][0] += 1
    elif mutation == "rng":
        frame["sampling"]["rng_state_before_sha256"][0] = "0" * 64
    elif mutation == "terminal":
        frame["sampling"]["alive_after"][0] = False
    elif mutation == "count":
        frame["geometry"][0]["active_positions"] = 0
    elif mutation == "vocab":
        frame["vocabulary_size"] += 1
    elif mutation == "dtype":
        frame["activation_dtypes"]["after"] = "torch.float32"
    elif mutation == "text":
        result["records"][0]["text"] = "violet"
    elif mutation == "sampled_record":
        result["records"][0]["answer_token_ids"][0] = 31
    elif mutation == "flag":
        report["checks"]["hooks_clean"] = False
    elif mutation == "hook_count":
        frame["before_observer_calls"] = 0
    else:
        result["arrays"]["unassigned"] = np.ones(8, dtype=np.float32)
    with pytest.raises(ValueError, match=message):
        check(result)


def test_byte_consistent_changed_after_sample_still_fails_bf16_reconstruction(calls):
    result = deepcopy(calls["full"])
    key = "f00_r00_after"
    result["arrays"][key][0, 0] = float(torch.tensor(99.0, dtype=torch.bfloat16))
    binding = audit.array_binding(result["arrays"][key])
    result["evidence"]["arrays"][key] = binding
    geometry = result["evidence"]["forwards"][0]["geometry"][0]
    geometry["saved_after"].update(binding)
    geometry["after_sha256"] = digest(audit.tensor_binding(torch.from_numpy(result["arrays"][key]).to(torch.bfloat16)))
    with pytest.raises(ValueError, match="exact BF16"):
        check(result)


@pytest.mark.parametrize("text,names,target,category,picked", [
    (" 'VIOLET!' <|im_end|>\n", ["violet", "yellow"], "violet", "target", "violet"),
    ("violet because I chose it", ["violet", "yellow"], "violet", "target", "violet"),
    ("violetness", ["violet", "yellow"], "yellow", "other", "violet"),
    ("lever95<|im_end|>", ["lever64", "lever95"], "lever95", "target", "lever95"),
    ("I choose violet", ["violet", "yellow"], "violet", "malformed", None),
    ("<|im_end|>", ["violet", "yellow"], "violet", "malformed", None),
])
def test_independent_literal_source_parser_not_stricter_button_grammar(text, names, target, category, picked):
    result = audit.score_text(text, names, target)
    assert result["category"] == category and result["picked"] == picked
    assert result["valid"] is (picked is not None)

# Filesystem tests mock numerical inference only; the real tiny-BF16 tests above
# separately check the raw arithmetic, timeline, scoring and identity core.
from test_pain_profile_execution import install_fake_engine, run  # noqa: E402
import test_pain_profile_execution as execution_tests  # noqa: E402
material = execution_tests.material
from curiosity_study import pain_profile_engine as engine  # noqa: E402
from curiosity_study.artifacts import read  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def synthetic_phase(tmp_path, monkeypatch, material, phase, **options):
    install_fake_engine(monkeypatch, **options)
    original = engine.generate_segment

    def populated(*args, **kwargs):
        result = original(*args, **kwargs)
        for row, item in zip(result["records"], kwargs["items"]):
            if options.get("malformed"):
                row.update(text="?", answer_token_ids=material[1].encode("?", add_special_tokens=False))
            row.update(prompt_token_ids=item["prompt_ids"], prompt_tokens=len(item["prompt_ids"]),
                       padded_prompt_tokens=max(len(i["prompt_ids"]) for i in kwargs["items"]),
                       prefill_position_id=len(item["prompt_ids"]) - 1, prefill_proj_monitor=0,
                       p_x=0.2, p_y=0.3, ended_with_eos=True)
        for frame in result["evidence"]["forwards"]:
            if "sampling" in frame:
                frame["sampling"]["alive_after"] = [True] * len(kwargs["items"])
        return result

    monkeypatch.setattr(engine, "generate_segment", populated)
    def numerical(result, arrays, items, **kwargs):
        report = result["evidence"]
        return dict(complete=report["complete"], passed=report["integrity_passed"],
                    expected_fault_observed=kwargs["expected_fault"] is not None,
                    checks=report["checks"], forwards=len(report["forwards"]))
    monkeypatch.setattr(audit, "recompute_call", numerical)
    monkeypatch.setattr(audit, "runtime_identity", lambda *args: {"synthetic_policy": True})
    admission = dict(created=time.time() - 1, deadline=time.time() + 1000)
    identity = dict(load_seconds=5.0, admission_sha256=digest(admission))
    if phase == "benchmark":
        put(tmp_path / "phases/setup/identity.json", identity)
    summary = run(phase, tmp_path, material, identity=identity, deadline=admission["deadline"] - 60)
    put(tmp_path / "freeze.json", {"synthetic": True})
    provenance = dict(freeze=dict(archive_sha256="a" * 64), vectors=dict(pain=None, fear=None, pain_unit_float32=None),
                      tokenizer=material[1], vocabulary_size=151936)
    return summary, provenance, admission


def phase_check(path, phase, material, context):
    _, provenance, admission = context
    return audit.audit_phase(path, phase, material[0], provenance, admission)


@pytest.mark.parametrize("phase", ["setup", "benchmark"])
def test_complete_raw_phase_reconciles_every_assigned_slot(tmp_path, monkeypatch, material, phase):
    context = synthetic_phase(tmp_path, monkeypatch, material, phase, malformed=True)
    checked = phase_check(tmp_path, phase, material, context)
    assert checked["complete"] and checked["passed"] and checked["integrity_passed"]
    assert checked["assigned_rows"] == len(checked["outcomes"]) == (612 if phase == "setup" else 2020)
    assert checked["expected_fault_rows"] == (64 if phase == "setup" else 0)
    assert all(row["category"] == "malformed" for row in checked["outcomes"] if row["status"] == "completed")
    assert not checked["behavior_gates_used"] and not checked["scientific_qualification"]
    if phase == "setup":
        assert checked["forecast"] == context[0]["forecast"]


@pytest.mark.parametrize("options", [dict(fail_call=0), dict(break_prefill=True), dict(absent_decode=True)])
def test_terminal_engineering_failure_preserves_full_denominator(tmp_path, monkeypatch, material, options):
    context = synthetic_phase(tmp_path, monkeypatch, material, "setup", **options)
    checked = phase_check(tmp_path, "setup", material, context)
    assert not checked["complete"] and not checked["passed"]
    assert len(checked["outcomes"]) == 612 and checked["missing_rows"] > 0
    if options.get("fail_call") == 0:
        assert checked["execution_error_rows"] == 32 and checked["observed_rows"] == 0


def test_hardkill_without_summary_does_not_certify_completed_calls(tmp_path, monkeypatch, material):
    context = synthetic_phase(tmp_path, monkeypatch, material, "benchmark")
    directory = tmp_path / "phases/benchmark"
    (directory / "summary.json").unlink()
    checked = phase_check(tmp_path, "benchmark", material, context)
    assert not checked["complete"] and not checked["passed"] and checked["observed_rows"] == 2020


def test_phase_never_started_preserves_all_missing(tmp_path, monkeypatch, material):
    put(tmp_path / "freeze.json", {"synthetic": True})
    checked = audit.audit_phase(tmp_path, "benchmark", material[0], dict(freeze=dict(archive_sha256="a" * 64)), {})
    assert not checked["complete"] and not checked["passed"]
    assert checked["missing_rows"] == len(checked["outcomes"]) == 2020


@pytest.mark.parametrize("change,message", [
    ("scoring", "scoring differs"), ("array", "byte binding"), ("attempt", "attempt/row"),
    ("reference", "reference call"), ("timing", "call timing"), ("summary", "summary"),
    ("unknown", "out-of-order"), ("identity", "admission"),
])
def test_durable_phase_tampering_fails_closed(tmp_path, monkeypatch, material, change, message):
    context = synthetic_phase(tmp_path, monkeypatch, material, "benchmark")
    directory = tmp_path / "phases/benchmark"
    path = directory / "calls/0000.json"
    value = read(path)
    if change == "scoring":
        value["outcomes"][0]["category"] = "other"
        value["outcomes_sha256"] = digest(value["outcomes"])
    elif change == "array":
        with path.with_suffix(".npz").open("ab") as handle:
            handle.write(b"changed")
    elif change == "attempt":
        value["attempt"] = "f" * 32
    elif change == "reference":
        value["prefill_reference_call_id"] = "benchmark-0001"
    elif change == "timing":
        value["elapsed_seconds"] = -1
    elif change == "summary":
        summary = read(directory / "summary.json")
        summary["completed_rows"] -= 1
        put(directory / "summary.json", summary)
    elif change == "unknown":
        put(directory / "calls/9999.json", value)
    elif change == "identity":
        identity = read(directory / "identity.json")
        identity["admission_sha256"] = "f" * 64
        put(directory / "identity.json", identity)
        started = read(directory / "started.json")
        started["identity_sha256"] = digest(identity)
        put(directory / "started.json", started)
    put(path, value)
    with pytest.raises(ValueError, match=message):
        phase_check(tmp_path, "benchmark", material, context)


def test_forecast_uses_all10_original_primary_elapsed_not_generation_only():
    calls = [dict(call_id=str(i), role="primary") for i in range(10)]
    result = audit.measured_forecast(dict(load_seconds=5), calls, {str(i): i + 1 for i in range(10)})
    assert result["benchmark_seconds"] == 5 + 2 * 10 * 65 + 120
    assert result["passed"]
    assert not audit.measured_forecast(dict(load_seconds=5), calls, {"0": 1})["passed"]
    assert not audit.measured_forecast(dict(load_seconds=5), calls, {str(i): 30 for i in range(10)})["passed"]

import test_prepare_pain_profile_control as preparation_tests  # noqa: E402
closed_root = preparation_tests.closed_root
source_tree = preparation_tests.source_tree
import prepare_pain_profile_control as preparation  # noqa: E402
import shutil  # noqa: E402
import tarfile  # noqa: E402


@pytest.fixture
def frozen_metadata(source_tree):
    project = Path(__file__).resolve().parents[1]
    names = [str(p.relative_to(project)) for p in (project / "src/curiosity_study").glob("pain_profile_*.py")]
    names += ["scripts/" + name for name in ("audit_pain_profile_control.py", "analyze_pain_profile_control.py", "prepare_pain_profile_control.py")]
    for name in names:
        target = source_tree / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project / name, target)
    manifest = preparation.build_manifest(source_tree)
    run_path = source_tree / "synthetic-campaign"
    put(run_path / "manifest.json", manifest)
    with tarfile.open(run_path / "source.tar.gz", "w:gz") as archive:
        for name in sorted(set(names) | set(manifest["documents"])):
            archive.add(source_tree / name, arcname=name)
    frozen = dict(manifest_sha256=digest(manifest), source_sha256=manifest["source_sha256"],
                  archive_sha256=audit.sha(run_path / "source.tar.gz"), documents=manifest["documents"])
    put(run_path / "freeze.json", frozen)
    return run_path, manifest


def test_frozen_metadata_reproduces_real_vocabulary_tokens_and_vectors(frozen_metadata):
    run_path, manifest = frozen_metadata
    proof = audit.source_provenance(run_path, manifest)
    assert proof["vocabulary_size"] == 152064
    assert proof["vectors"]["pain"].shape == (5120,)
    assert proof["vectors"]["pain"].dtype == torch.bfloat16
    assert len(manifest["preflight"]["requests"]) == 404
    assert len(proof["environment"]["distributions"]) == 57
    assert proof["prior_closure_sha256"] == manifest["prior_study_closure_sha256"]


def test_frozen_archive_cannot_be_replaced_by_consistent_new_manifest(frozen_metadata):
    run_path, manifest = frozen_metadata
    manifest["preflight"]["requests"][0]["prompt_ids"][0] += 1
    put(run_path / "manifest.json", manifest)
    frozen = read(run_path / "freeze.json")
    frozen["manifest_sha256"] = digest(manifest)
    put(run_path / "freeze.json", frozen)
    with pytest.raises(ValueError, match="prompts/preflight"):
        audit.source_provenance(run_path, manifest)

from curiosity_study import pain_profile_assets as assets  # noqa: E402
from curiosity_study import pain_profile_admission as admission_module  # noqa: E402
from test_pain_profile_admission import STATE, evidence, put as admission_put  # noqa: E402
import test_pain_profile_admission as admission_tests  # noqa: E402
campaign = admission_tests.campaign


def runtime_fixture():
    provenance = dict(base={"synthetic": "base"}, adapter_files={"weights": "a" * 64},
                      environment={"lock": "synthetic"}, freeze=dict(archive_sha256="b" * 64), inventory={"f": "c" * 64})
    manifest = dict(vector_receipt={"synthetic": "vectors"}, source_sha256="d" * 64)
    identity = dict(base=provenance["base"], adapter_revision=assets.ADAPTER_REVISION,
        adapter_archive_sha256=assets.ADAPTER_SHA, adapter_files=provenance["adapter_files"],
        vector_receipt=manifest["vector_receipt"], environment=provenance["environment"],
        gpu="NVIDIA H100 80GB HBM3", cuda_build="13.0", base_dtype="torch.bfloat16", attention="sdpa",
        lora_layers=448, adapter_merged=False, adapter_dtypes=["torch.float32"], model_inference_performed=False,
        python_version="3.12.14", machine="x86_64", driver_version="580.0", gpu_total_memory_bytes=80_000_000_000,
        gpu_compute_capability=[9, 0], numerical_flags=dict(cuda_matmul_allow_tf32=True, cudnn_allow_tf32=True,
            allow_bf16_reduced_precision_reduction=True, float32_matmul_precision="high"),
        load_seconds=5.0, allocated_bytes=70_000, reserved_bytes=80_000, source_sha256=manifest["source_sha256"],
        source_archive_sha256=provenance["freeze"]["archive_sha256"], execution_files_sha256=digest(provenance["inventory"]),
        modal_image_id="im-synthetic", admission_sha256="e" * 64)
    return identity, manifest, provenance


def test_actual_runtime_comparison_ignores_host_driver_memory_but_not_arithmetic():
    identity, manifest, provenance = runtime_fixture()
    first = audit.runtime_identity(identity, manifest, provenance)
    identity.update(driver_version="581.1", gpu_total_memory_bytes=79_900_000_000, gpu="NVIDIA H100 SXM", load_seconds=10,
                    admission_sha256="f" * 64)
    assert first == audit.runtime_identity(identity, manifest, provenance)
    identity["modal_image_id"] = "im-other"
    assert first != audit.runtime_identity(identity, manifest, provenance)


@pytest.mark.parametrize("field,value", [
    ("python_version", "3.11.0"), ("base_dtype", "torch.float32"), ("adapter_merged", True),
    ("gpu_compute_capability", [8, 0]), ("adapter_dtypes", ["torch.bfloat16"]),
    ("execution_files_sha256", "f" * 64), ("modal_image_id", None), ("cuda_build", "12.8"),
])
def test_runtime_policy_drift_rejected(field, value):
    identity, manifest, provenance = runtime_fixture()
    identity[field] = value
    with pytest.raises(ValueError):
        audit.runtime_identity(identity, manifest, provenance)


def test_original_phase_admission_and_matching_central_ledger_rechecked(campaign):
    path, manifest, frozen = campaign
    admission_module.claim(path, "prefetch", STATE, now=1000)
    evidence(path)
    admitted = admission_module.claim(path, "setup", STATE, now=3000)
    provenance = dict(freeze=frozen, inventory=admitted["execution_files"])
    assert audit.admission_binding(path, "setup", manifest, provenance) == admitted
    admitted["retry_allowed"] = True
    admission_put(path / "admissions/setup.json", admitted)
    admission_put(admission_tests.ledger(path) / "setup.json", admitted)
    with pytest.raises(ValueError, match="one-admission"):
        audit.admission_binding(path, "setup", manifest, provenance)


def test_benchmark_admission_requires_reproducible_immutable_setup_audit(campaign, monkeypatch):
    path, manifest, frozen = campaign
    admission_tests.setup_done(path)
    admitted = admission_module.claim(path, "benchmark", STATE, now=5000)
    setup = read(path / "phases/setup/audit.json")
    provenance = dict(freeze=frozen, inventory=admitted["execution_files"])
    monkeypatch.setattr(audit, "audit", lambda run, phase: dict(setup, passed=False))
    with pytest.raises(ValueError, match="cannot be reproduced"):
        audit.admission_binding(path, "benchmark", manifest, provenance)


def test_sampled_tokens_same_but_changed_initial_readout_fails_isolation(calls):
    changed = deepcopy(calls["one_row"])
    changed["records"][1]["p_x"] += .001
    with pytest.raises(ValueError, match="terminal status"):
        audit.compare_rows(calls["baseline"], changed, [1, 2])


def test_worker_load_failure_is_unavailable_with_all_denominators(tmp_path, material):
    put(tmp_path / "freeze.json", {"synthetic": True})
    failure = dict(schema_version=1, phase="benchmark", manifest_sha256=digest(material[0]), complete=False,
                   passed=False, model_inference_performed=None,
                   error=dict(type="OutOfMemoryError", message="synthetic model load failure"), finished_unix=1234.)
    put(tmp_path / "phases/benchmark/summary.json", failure)
    put(tmp_path / "transport/benchmark/worker-error.json", failure)
    checked = audit.audit_phase(tmp_path, "benchmark", material[0], dict(freeze=dict(archive_sha256="a" * 64)), {})
    assert not checked["complete"] and not checked["passed"]
    assert checked["missing_rows"] == 2020 and checked["execution_error_rows"] == 0
    assert checked["worker_failure"]["error"] == failure["error"]
    assert checked["summary_byte_sha256"] == audit.sha(tmp_path / "phases/benchmark/summary.json")
    (tmp_path / "transport/benchmark/worker-error.json").unlink()
    with pytest.raises(ValueError, match="Unbound"):
        audit.audit_phase(tmp_path, "benchmark", material[0], dict(freeze=dict(archive_sha256="a" * 64)), {})


def test_executor_deadline_must_reserve_exact_export_minute(tmp_path, monkeypatch, material):
    context = synthetic_phase(tmp_path, monkeypatch, material, "setup", fail_call=0)
    path = tmp_path / "phases/setup/started.json"
    started = read(path)
    assert started["deadline_unix"] == context[2]["deadline"] - 60
    started["deadline_unix"] = context[2]["deadline"]
    put(path, started)
    with pytest.raises(ValueError, match="60-second export reserve"):
        phase_check(tmp_path, "setup", material, context)
