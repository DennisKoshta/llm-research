"""Pure source/terminal/receipt checks and injected failure; no model or cloud calls."""

from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from curiosity_study.coverage_engineering import call_plan
from curiosity_study import voluntary_capture

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import curiosity_coverage_engineering_worker as worker  # noqa: E402


@pytest.fixture
def manifest():
    requests = {
        cohort: [
            dict(
                source_trial_id=f"base{cohort}-{i}",
                token_prefix=[1, 2, 3],
                token_prefix_sha256=worker.digest([1, 2, 3]),
                seed=100 + i,
                prefix_positions=dict(record=1, final=2),
                labels="ABCD",
            )
            for i in range(32)
        ]
        for cohort in (0, 3)
    }
    calls = call_plan(requests)
    return dict(
        name=worker.NAME,
        resources=worker.resources(),
        plan_sha256=worker.digest("synthetic materials"),
        plan=dict(model=dict(model_id="synthetic", revision="a" * 40), engineering=dict(call_plan=calls)),
        preflight={"synthetic": True},
        wrapper_files={Path(worker.__file__).name: worker.sha(worker.__file__)},
    )


def durable_call(root, manifest, index, *, expected=True, rows=True):
    call = manifest["plan"]["engineering"]["call_plan"][index]
    worker.write_once(
        root / "attempts" / f"{index:02d}.json", dict(call=call, materials_sha256=manifest["plan_sha256"])
    )
    path = root / "calls" / f"{index:02d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, synthetic=np.array([1], dtype=np.float32))
    fault = call["inject_failure_at"]
    report = dict(
        call=call,
        passed=expected,
        fault_injected=bool(fault) and expected,
        error=dict(type="ControlledCoverageFault", message=fault) if fault else None,
        completed_forwards=(1 if fault == "after_prefill" else 2),
        rows=[{} for _ in range(32)] if rows and not fault else None,
        array_artifact=dict(path=f"calls/{index:02d}.npz", byte_sha256=worker.sha(path)),
    )
    worker.write_once(root / "calls" / f"{index:02d}.json", report)
    return report


def test_all512_slots_retained_with_missing_and_attempted_only_failure(tmp_path, manifest):
    calls = manifest["plan"]["engineering"]["call_plan"]
    durable_call(tmp_path, manifest, 0)
    worker.write_once(
        tmp_path / "attempts/01.json", dict(call=calls[1], materials_sha256=manifest["plan_sha256"])
    )
    result = worker.operational_coverage(tmp_path, manifest)
    assert len(result["slots"]) == len({r["trial_id"] for r in result["slots"]}) == 512
    assert [
        sum(r["status"] == status for r in result["slots"])
        for status in ("completed", "execution_error", "missing")
    ] == [32, 32, 448]
    assert not result["complete"] and not result["numeric_engineering_verified"]


def test_expected_fault_requires_positive_exact_stage_evidence(tmp_path, manifest):
    durable_call(tmp_path, manifest, 6, expected=False)
    durable_call(tmp_path, manifest, 8, expected=True)
    result = worker.operational_coverage(tmp_path, manifest)
    assert {r["status"] for r in result["slots"] if r["call_index"] == 6} == {"execution_error"}
    assert {r["status"] for r in result["slots"] if r["call_index"] == 8} == {"expected_fault"}
    assert not result["complete"]


def test_measured_failure_is_complete_but_never_numeric_certified(tmp_path, manifest):
    for index in range(16):
        durable_call(tmp_path, manifest, index, expected=index != 3)
    result = worker.operational_coverage(tmp_path, manifest)
    assert result["complete"] and len(result["artifacts"]) == 16
    assert not result["numeric_engineering_verified"] and not result["selected_receipts_verified"]


@pytest.mark.parametrize("mutation", ["request", "npz", "extra", "missing_attempt"])
def test_rehashed_or_missing_call_binding_cannot_certify_coverage(tmp_path, manifest, mutation):
    report = durable_call(tmp_path, manifest, 0)
    if mutation == "request":
        report["call"] = dict(report["call"], requests_sha256="0" * 64)
        (tmp_path / "calls/00.json").write_text(worker.canonical(report))
    elif mutation == "npz":
        (tmp_path / "calls/00.npz").write_bytes(b"changed")
    elif mutation == "extra":
        worker.write_once(tmp_path / "calls/16.json", {})
    else:
        (tmp_path / "attempts/00.json").unlink()
    result = worker.operational_coverage(tmp_path, manifest)
    assert result["issues"] and not result["complete"]


def test_admission_has_exact_deadline_and_no_retry(manifest):
    value = dict(
        manifest_sha256=worker.digest(manifest),
        created=100.0,
        deadline=3500.0,
        phase_consumed=True,
        retry_allowed=False,
        maximum_liability_usd=4.9060352,
    )
    worker.validate_admission(manifest, value, 101.0)
    for changed in (
        dict(value, deadline=3501.0),
        dict(value, retry_allowed=True),
        dict(value, phase_consumed=1),
    ):
        with pytest.raises(ValueError):
            worker.validate_admission(manifest, changed, 101.0)
    with pytest.raises(ValueError):
        worker.validate_admission(manifest, value, 3500.0)
    worker.validate_admission(manifest, value, 4000.0, historical=True)


def test_run_has_no_added_startup_and_preserves_constructor_failure(tmp_path, manifest, monkeypatch):
    worker.write_once(tmp_path / "manifest.json", manifest)
    admission = dict(
        manifest_sha256=worker.digest(manifest),
        created=100.0,
        deadline=3500.0,
        phase_consumed=True,
        retry_allowed=False,
        maximum_liability_usd=4.9060352,
    )
    worker.write_once(tmp_path / "admission.json", admission)
    calls = []

    def construct(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("synthetic loading failure")

    monkeypatch.setattr(
        worker, "verify_sources", lambda *args: {"batched": SimpleNamespace(BatchedHFModel=construct)}
    )
    monkeypatch.setattr(worker.time, "time", lambda: 101.0)
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(key, "1")
    result = worker.run(tmp_path, tmp_path, tmp_path, tmp_path)
    assert len(calls) == 1 and result["error"]["message"] == "synthetic loading failure"
    assert len(result["operational_coverage"]["slots"]) == 512
    assert all(r["status"] == "missing" for r in result["operational_coverage"]["slots"])
    assert result["complete"] is False and result["engineering_passed"] is False
    with pytest.raises(FileExistsError):
        worker.run(tmp_path, tmp_path, tmp_path, tmp_path)
    assert len(calls) == 1


def runtime_fixture(root):
    versions = dict(torch="2.6.0+cu124", transformers="4.51.3", accelerate="1.6.0", tokenizers="0.21.1")
    (root / "uv.lock").write_text(
        "\n".join(
            f'[[package]]\nname = "{name}"\nversion = "{version}"' for name, version in versions.items()
        )
    )
    return dict(
        gpu="NVIDIA H200",
        device="cuda",
        dtype="bfloat16",
        attention="eager",
        python="3.12.8",
        backend="transformers",
        execution_backend="torch_batched_v1",
        logits_to_keep=1,
        layers=64,
        cuda="12.4",
        cuda_numerics=dict(
            policy="bf16_reduced_precision_reduction_disabled_v1",
            allow_bf16_reduced_precision_reduction=False,
            allow_fp16_reduced_precision_reduction=True,
            allow_tf32=False,
        ),
        loading_policy="direct_device_sharded",
        hidden_size=5120,
        vocabulary_size=151936,
        pad_token_id=151643,
        generation_eos_token_ids=[151643, 151645],
        inherited_synthetic_startup_forwards=0,
        versions=versions,
    )


def test_run_calls_only_fixed_collector_once_without_startup_probes(tmp_path, manifest, monkeypatch):
    worker.write_once(tmp_path / "manifest.json", manifest)
    worker.write_once(
        tmp_path / "admission.json",
        dict(
            manifest_sha256=worker.digest(manifest),
            created=100.0,
            deadline=3500.0,
            phase_consumed=True,
            retry_allowed=False,
            maximum_liability_usd=4.9060352,
        ),
    )
    observed = []
    model = SimpleNamespace(active_hooks=0, tokenizer=object())

    def execute(root, actual_model, plan, *, checkpoint, deadline):
        observed.append((root, actual_model, plan, deadline))
        checkpoint()
        return dict(complete=False, engineering_passed=False)

    modules = dict(
        batched=SimpleNamespace(BatchedHFModel=lambda **kwargs: model),
        coverage_engineering=SimpleNamespace(hook_inventory=lambda value: {}, execute=execute),
        voluntary_resolution=None,
    )
    monkeypatch.setattr(worker, "verify_sources", lambda *args: modules)
    monkeypatch.setattr(worker, "runtime_metadata", lambda *args: {"synthetic": True})
    monkeypatch.setattr(worker, "tokenizer_preflight", lambda *args: manifest["preflight"])
    monkeypatch.setattr(worker.time, "time", lambda: 101.0)
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(key, "1")
    result = worker.run(tmp_path, tmp_path, tmp_path, tmp_path)
    assert observed == [(tmp_path, model, manifest["plan"], 3440.0)]
    assert result["error"] is None and result["checkpoint_count"] == 2
    assert not result["complete"] and not result["engineering_passed"]


@pytest.mark.parametrize(
    "mutation", ["empty_versions", "cuda_suffix", "precision", "attention", "hidden", "extra_forward"]
)
def test_runtime_versions_and_arithmetic_are_exact_not_empty_all_checks(tmp_path, mutation):
    metadata = runtime_fixture(tmp_path)
    worker.validate_runtime_metadata(metadata, tmp_path)
    if mutation == "empty_versions":
        metadata["versions"] = {}
    elif mutation == "cuda_suffix":
        metadata["versions"]["torch"] = "2.6.0"
    elif mutation == "precision":
        metadata["cuda_numerics"]["allow_tf32"] = True
    elif mutation == "attention":
        metadata["attention"] = "sdpa"
    elif mutation == "hidden":
        metadata["hidden_size"] = 4096
    else:
        metadata["inherited_synthetic_startup_forwards"] = 1
    with pytest.raises(ValueError):
        worker.validate_runtime_metadata(metadata, tmp_path)


@pytest.fixture
def receipt_fixture():
    source = [
        dict(kind="choice" if index == 0 else "diagnostic", cap_hit=False, assignment={"index": index})
        for index in range(32)
    ]
    source[0]["simulator_outcome"] = dict(observation_ledger=[dict(model_pass_completed=False)])
    assignments = [dict(kind=row["kind"], index=i) for i, row in enumerate(source)]
    request = dict(
        token_prefix=[1, 2, 3],
        token_prefix_sha256=worker.digest([1, 2, 3]),
        prefix_positions=dict(record=1, final=2),
    )

    def sidecar(row):
        return dict(
            status="pending",
            receipt_required=True,
            receipt_model_pass_completed=False,
            executed_simulator_outcome=deepcopy(row["simulator_outcome"]),
        )

    binding, payload = (
        dict(token_prefix_sha256=request["token_prefix_sha256"]),
        dict(payload="actual selected report"),
    )
    receipts = SimpleNamespace(
        choice_sidecar=sidecar,
        render_receipt=lambda *args: (payload, request, binding),
        ENGINEERING_CHECKS=voluntary_capture.ENGINEERING_CHECKS,
        array_sha=voluntary_capture.array_sha,
    )
    modules = dict(
        voluntary_resolution=SimpleNamespace(
            record_from_thinking_inference=lambda a, inference: source[a["index"]]
        ),
        voluntary_capture=receipts,
    )
    call = dict(
        call_index=10,
        call_id="synthetic timing",
        source_cohort=0,
        trial_ids=[f"trial{i}" for i in range(32)],
        base_trial_ids=[f"base{i}" for i in range(32)],
    )
    generation = dict(rows=[dict(completion={}) for _ in range(32)])
    materials = dict(engineering=dict(source_assignments_by_cohort={"0": assignments}))
    hidden = np.zeros((1, 1, 2, 5120), dtype=np.float32)
    measurement = dict(
        unpadded_input_ids=[1, 2, 3],
        input_ids=[0] * 125 + [1, 2, 3],
        attention_mask=[0] * 125 + [1] * 3,
        position_ids=[0] * 125 + [0, 1, 2],
        unpadded_token_positions=request["prefix_positions"],
        record_token_index=126,
        token_prefix_sha256=request["token_prefix_sha256"],
        exact_token_replay=True,
        prefill_width=128,
        prefill_padding_multiple=128,
        intervention=None,
        hooks_remaining=0,
    )
    expected = sidecar(source[0])
    executed = deepcopy(expected["executed_simulator_outcome"])
    executed["receipt_model_pass_completed"] = executed["observation_ledger"][-1]["model_pass_completed"] = (
        True
    )
    expected.update(
        source_trial_id="trial0",
        base_trial_id="base0",
        generation_row=0,
        source_assignment_sha256=worker.digest(assignments[0]),
        receipt_input=payload,
        token_binding=binding,
        status="completed",
        receipt_model_pass_completed=True,
        measurement=measurement,
        activation_row=0,
        executed_simulator_outcome=executed,
    )
    guard = dict(
        passed=True,
        checks=dict.fromkeys(receipts.ENGINEERING_CHECKS, True),
        source_trial_ids=["trial0"],
        token_prefix_sha256=[request["token_prefix_sha256"]],
        layers=[15],
        positions=["record", "final"],
        max_new_tokens=0,
        active_batch_size=1,
        baseline_full_initial_logits_sha256=["a" * 64],
        captured_full_initial_logits_sha256=["a" * 64],
        captured_array_sha256=receipts.array_sha(hidden),
        prefill_width=128,
    )
    report = dict(
        call_index=10,
        call_id=call["call_id"],
        source_records=source,
        requests=[request],
        requests_sha256=worker.digest([request]),
        active_source_trial_ids=["trial0"],
        assigned_choices=1,
        actual_receipt_rows=1,
        receipt_forward_attempted=True,
        records=[expected],
        records_sha256=worker.digest([expected]),
        passed=True,
        error=None,
        actual_forward_calls=2,
        engineering=guard,
        observed_measurements=[measurement],
        actual_padded_width=128,
    )
    return report, {"hidden": hidden}, call, generation, materials, modules, SimpleNamespace(pad_token_id=0)


def test_receipt_context_reconstruction_binds_actual_selected_payload_and_capture(receipt_fixture):
    result = worker.audit_receipts(*receipt_fixture, np)
    assert result["passed"] and result["assigned_choices"] == result["actual_selected_forwards"] == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "empty_guard",
        "wrong_position",
        "missing_measurement",
        "extra_request",
        "payload",
        "logits",
        "null_logits",
        "capture_precision",
        "forward_flag",
        "wrong_call",
    ],
)
def test_rehashed_receipt_cannot_omit_checks_change_capture_or_deliver_unselected_info(
    receipt_fixture, mutation
):
    report = receipt_fixture[0]
    if mutation == "empty_guard":
        report["engineering"]["checks"] = {}
    elif mutation == "wrong_position":
        report["observed_measurements"][0]["record_token_index"] -= 1
        report["records_sha256"] = worker.digest(report["records"])
    elif mutation == "missing_measurement":
        report["observed_measurements"].append({})
    elif mutation == "extra_request":
        report["requests"].append(report["requests"][0])
        report["requests_sha256"] = worker.digest(report["requests"])
    elif mutation == "payload":
        report["records"][0]["receipt_input"] = {"payload": "unselected extra information"}
        report["records_sha256"] = worker.digest(report["records"])
    elif mutation == "logits":
        report["engineering"]["captured_full_initial_logits_sha256"][0] = "b" * 64
    elif mutation == "null_logits":
        report["engineering"]["captured_full_initial_logits_sha256"] = [None]
        report["engineering"]["baseline_full_initial_logits_sha256"] = [None]
    elif mutation == "capture_precision":
        receipt_fixture[1]["hidden"][0, 0, 0, 0] = np.float32(1.00001)
        report["engineering"]["captured_array_sha256"] = voluntary_capture.array_sha(
            receipt_fixture[1]["hidden"]
        )
    elif mutation == "forward_flag":
        report["receipt_forward_attempted"] = False
    else:
        report["call_index"] = 11
    with pytest.raises(ValueError):
        worker.audit_receipts(*receipt_fixture, np)


@pytest.mark.parametrize("mutation", ["third_file", "wrong_original_overlay", "undeclared_overlay"])
def test_rehashed_runtime_inventory_cannot_hide_a_third_source_change(tmp_path, monkeypatch, mutation):
    original, runtime, wrapper = (tmp_path / name for name in ("original", "runtime", "wrapper"))
    old = original / "src/curiosity_study"
    new = runtime / "curiosity_study"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    wrapper.mkdir()
    texts = {
        "__init__.py": "# synthetic package",
        "model.py": "# immutable model",
        "state_backend.py": "# old backend",
    }
    for name, value in texts.items():
        (old / name).write_text(value)
        (new / name).write_text(value)
    delta = {}
    for name in worker.OVERLAYS:
        (new / name).write_text("# reviewed replacement " + name)
        delta[name] = dict(
            original_byte_sha256=worker.sha(old / name) if (old / name).exists() else None,
            replacement_byte_sha256=worker.sha(new / name),
        )
    (wrapper / "placeholder").write_text("never import")
    if mutation == "third_file":
        (new / "model.py").write_text("# changed model outside permitted overlay")
    elif mutation == "wrong_original_overlay":
        delta["state_backend.py"]["original_byte_sha256"] = "0" * 64
    else:
        (new / "extra.py").write_text("# hidden third overlay")

    def inventory(root):
        return {str(p.relative_to(root)): worker.sha(p) for p in root.rglob("*") if p.is_file()}

    bound = dict(
        original_execution_files=inventory(original),
        runtime_files=inventory(runtime),
        wrapper_files=inventory(wrapper),
        runtime_delta=delta,
    )
    monkeypatch.setattr(worker, "validate_manifest", lambda value: value)
    monkeypatch.setattr(worker, "SOURCE_SHA", worker.digest(texts))
    monkeypatch.setattr(
        worker, "load_module", lambda *args: pytest.fail("reject before executing any module")
    )
    with pytest.raises(ValueError, match="Overlay|Undeclared"):
        worker.verify_sources(bound, original, runtime, wrapper)


def test_tokenizer_checks_all288_prefixes_with_no_generation():
    base, items, requests = [], [], {}
    for index in range(288):
        trial = f"synthetic-{index}"
        request = dict(messages=[dict(role="user", content=trial)], labels="ABCD", seed=index)
        requests[trial] = request
        base.append(
            dict(
                base_trial_id=trial,
                source_assignment={"trial_id": trial},
                inference_request_sha256=worker.digest(request),
            )
        )
        items.append(
            dict(
                rendered_prompt=trial,
                offset_mapping=[[0, len(trial)]],
                request=dict(token_prefix=[index + 1], labels="ABCD", seed=index),
            )
        )
    plan = dict(
        base_assignments=base,
        source_generation_preflight=dict(requests=items, requests_sha256=worker.digest(items)),
    )

    class Tokenizer:
        chat_template, pad_token_id = "fixed", 0

        def __init__(self):
            self.changed = False

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=True)
            return messages[0]["content"]

        def __call__(self, text, **kwargs):
            assert kwargs == dict(add_special_tokens=False, return_offsets_mapping=True)
            index = int(text.rsplit("-", 1)[1])
            return dict(
                input_ids=[index + 1 + (self.changed and index == 287)], offset_mapping=[[0, len(text)]]
            )

    tokenizer = Tokenizer()
    task = SimpleNamespace(inference_request=lambda assignment: requests[assignment["trial_id"]])
    result = worker.tokenizer_preflight(plan, tokenizer, task)
    assert result["checked_requests"] == 288 and not result["model_outputs_generated"]
    tokenizer.changed = True
    with pytest.raises(ValueError, match="Actual original"):
        worker.tokenizer_preflight(plan, tokenizer, task)
