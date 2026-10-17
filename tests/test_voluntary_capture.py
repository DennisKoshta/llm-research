"""Receipt execution uses synthetic choices; real CPU tests exercise the read-only forward."""

from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from curiosity_study import thinking_information as thinking
from curiosity_study import voluntary_capture as capture
from curiosity_study import voluntary_resolution as voluntary
from curiosity_study.artifacts import RunStore, digest, read
from curiosity_study.state_backend import StateBackend
from test_concept_capture import WordTokenizer, tiny_model


def manifest():
    rows = voluntary.assignments()
    return dict(
        schema_version=3,
        assay="voluntary_resolution",
        task_revision=voluntary.REVISION,
        model=deepcopy(capture.MODEL),
        execution=deepcopy(voluntary.EXECUTION),
        diagnostics=rows,
        assignments=[],
        assignment_sha256=digest(rows),
        world_selection=voluntary.assignment_plan()["selection"],
        max_new_tokens=4096,
        capture_plan=dict(layers=capture.LAYERS, steps=[]),
        receipt_capture_plan=deepcopy(capture.RECEIPT_PLAN),
        prechoice_capture_aliases=deepcopy(capture.PRECHOICE_SITES),
    )


def source_record(a, service="report", raw=None):
    mapping = voluntary.task(a)["action_mapping"]
    label = (
        next(key for key, value in mapping.items() if value == service)
        if a["kind"] == "choice"
        else voluntary.prompt(a)[2]
    )
    raw = raw if raw is not None else f"Private reasoning mentioning End of record. </think> {label}."
    record = thinking.record_from_inference(a, dict(raw=raw, ended_with_eos=True, generated_token_ids=[3, 4]))
    record["activation_artifact"] = dict(
        path="synthetic-source.npz",
        sha256="synthetic",
        row=0,
        layers=capture.LAYERS,
        positions=["record", "prefill", "think_end", "decision"],
        available=[[True] * 4 for _ in capture.LAYERS],
        token_positions=[0, 1, 2, 3],
    )
    return record


def with_encoder(model):
    def encode(messages, enable_thinking=False):
        assert enable_thinking is False
        chunks = []
        for message in messages:
            text = message["content"]
            if message["role"] == "assistant":
                text = text.rsplit("</think>", 1)[-1].strip()
            chunks.append(f"<{message['role']}>\n{text}\n</message>")
        rendered = "\n".join(chunks) + "\n<assistant>\n<think></think>"
        ids = model.tokenizer.encode(rendered, add_special_tokens=False)
        return rendered, SimpleNamespace(input_ids=torch.tensor([ids]))

    model.encode = encode
    return model


def fake_model():
    return with_encoder(
        SimpleNamespace(
            torch=torch,
            device="cpu",
            active_hooks=0,
            tokenizer=WordTokenizer(),
            model=SimpleNamespace(config=SimpleNamespace(vocab_size=10000, hidden_size=4)),
            metadata=lambda: {**capture.MODEL, "synthetic": True},
        )
    )


def fake_forward(calls, *, corrupt=False, fail=False):
    def forward(model, requests, layers):
        calls.append(dict(layers=list(layers), ids=[r["token_prefix_sha256"] for r in requests]))
        if fail:
            raise RuntimeError("Synthetic receipt failure")
        rendered, ids, inputs, targets = StateBackend(model).prepare(requests)
        width = inputs["input_ids"].shape[1]
        rows = []
        for i, request in enumerate(requests):
            rows.append(
                dict(
                    prompt_sha256=digest(rendered[i]),
                    rendered_prompt=rendered[i],
                    input_ids=inputs["input_ids"][i].tolist(),
                    attention_mask=inputs["attention_mask"][i].tolist(),
                    position_ids=inputs["position_ids"][i].tolist(),
                    prefill_padding_multiple=128,
                    prefill_width=width,
                    unpadded_input_ids=ids[i],
                    token_prefix_sha256=digest(ids[i]),
                    exact_token_replay=True,
                    unpadded_token_positions=request["prefix_positions"],
                    record_token_index=targets["record"][i],
                    label_logits=dict(A=0.0, B=0.0),
                    label_probabilities_temperature_1=dict(A=0.01, B=0.01),
                    label_mass=0.02,
                    greedy_first_token_id=1,
                    greedy_first_token="A",
                    intervention=None,
                    hooks_remaining=0,
                    batch_seconds=0.001,
                )
            )
        hashes = [digest(dict(synthetic_logits_of=ids)) for ids in ids]
        if corrupt and layers:
            hashes[-1] = "f" * 64
        hidden = (
            np.arange(len(requests) * len(layers) * 2 * 4, dtype=np.float32).reshape(
                len(requests), len(layers), 2, 4
            )
            if layers
            else None
        )
        return rows, hidden, hashes

    return forward


def saved_sources(root, m, *, all_stop=False):
    store = RunStore(root, m)
    source = {}
    for index, assignment in enumerate(m["diagnostics"]):
        service = "stop" if all_stop else voluntary.SERVICES[assignment["world_slot"] % 4]
        raw = (
            "Unclosed reasoning A"
            if not all_stop and assignment["kind"] == "choice" and assignment["world_slot"] == 11
            else None
        )
        source[assignment["trial_id"]] = source_record(assignment, service, raw)
    for offset in range(0, 288, 32):
        store.save_many(
            [source[a["trial_id"]] for a in m["diagnostics"][offset : offset + 32]],
            "synthetic",
            f"{offset:07d}",
        )
    return source


def test_thinking_dispatch_closes_before_finalizing_only_optional_choices():
    for assignment in voluntary.assignments():
        request = thinking.inference_request(assignment)
        assert request == voluntary.inference_request(assignment)
        raw = "Thought.</think> A."
        inf = dict(raw=raw, ended_with_eos=True, generated_token_ids=[1])
        assert thinking.record_from_inference(assignment, inf) == voluntary.record_from_thinking_inference(
            assignment, inf
        )
    a = next(a for a in voluntary.assignments() if a["kind"] == "choice")
    for raw in ("A", "Still thinking", "Thought.</think> A B"):
        record = source_record(a, raw=raw)
        assert (
            not record["valid"] and record["selected_service"] is None and record["simulator_outcome"] is None
        )
    stop = source_record(a, "stop")
    assert stop["valid"] and not stop["simulator_outcome"]["receipt_required"]


def test_fresh_receipt_renders_only_selected_payload_and_discloses_removed_reasoning():
    model = fake_model()
    a = next(
        a for a in voluntary.assignments() if a["kind"] == "choice" and a["delivery"] == "redundant_evidence"
    )
    for service in ("report", "repeat", "pattern"):
        record = source_record(a, service)
        source_hash = digest(record)
        receipt, request, binding = capture.render_receipt(record, model)
        assert receipt == voluntary.receipt_input(record) and receipt["max_new_tokens"] == 0
        assert not binding["exact_source_token_replay"] and binding["template_may_strip_prior_thinking"]
        assert not binding["rendered_contains_raw_choice"]
        assert receipt["messages"][2]["content"] == record["inference"]["raw"]
        rendered = binding["rendered_prompt"]
        start, end = binding["selected_payload_span"]
        assert rendered[start:end] == record["simulator_outcome"]["payload"]
        token = request["prefix_positions"]["record"]
        left, right = binding["offset_mapping"][token]
        assert left <= end - 1 < right and not rendered[end:right].strip()
        assert request["prefix_positions"]["final"] == len(request["token_prefix"]) - 1
        assert (
            digest(record) == source_hash and not record["simulator_outcome"]["receipt_model_pass_completed"]
        )


def test_real_bf16_receipt_capture_has_all_sites_and_read_only_initial_logit_identity():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = with_encoder(tiny_model())
        choices = [a for a in voluntary.assignments() if a["kind"] == "choice"][:4]
        rendered = [capture.render_receipt(source_record(a, "report"), model) for a in choices]
        rows, hidden, guard = capture.forward_receipts(
            model, [r[1] for r in rendered], [a["trial_id"] for a in choices], [0, 1, 2]
        )
        assert guard["passed"] and guard["checks"]["row_coverage"]
        assert hidden.shape == (4, 3, 2, 8) and guard["captured_array_sha256"] == capture.array_sha(hidden)
        assert all(
            row["intervention"] is None and "completion" not in row and row["hooks_remaining"] == 0
            for row in rows
        )
        assert model.active_hooks == 0 and not model.model._forward_hooks
    finally:
        torch.set_num_threads(previous)


def test_full_wrapper_retains288_sources_and_all144_sidecars_without_rewriting_choices(tmp_path, monkeypatch):
    m, model, calls = manifest(), fake_model(), []
    original = saved_sources(tmp_path, m)
    before = {p: p.read_bytes() for p in (tmp_path / "cohorts").glob("*/*.json")}
    monkeypatch.setattr(thinking, "execute", lambda *args: original)
    monkeypatch.setattr(capture, "_logit_hashes", fake_forward(calls))
    assert capture.execute(tmp_path, m, model) == original
    assert all(path.read_bytes() == contents for path, contents in before.items())
    receipts = capture.load_receipts(tmp_path, m, original)
    assert len(receipts) == 144
    assert {r["status"] for r in receipts.values()} == {"completed", "stop", "invalid_choice"}
    assert len(list((tmp_path / "receipt_cohorts").glob("*/*.json"))) == 5
    assert len(calls) == 10 and all(len(call["ids"]) <= 32 for call in calls)
    for receipt in receipts.values():
        source = original[receipt["source_trial_id"]]
        if receipt["receipt_required"]:
            assert receipt["receipt_model_pass_completed"] and receipt["activation_artifact"]
            assert receipt["executed_simulator_outcome"]["observation_ledger"][-1]["model_pass_completed"]
            assert not source["simulator_outcome"]["receipt_model_pass_completed"]
        else:
            assert not receipt["receipt_model_pass_completed"] and receipt["measurement"] is None
    count = len(calls)
    # Resume re-reads JSON: integer observation keys become strings. Canonical
    # values must match both fresh in-memory and durable source representations.
    capture.execute_receipts(tmp_path, m, model, RunStore(tmp_path, m).completed())
    assert len(calls) == count  # No receipt or payload is retried on resume.


def test_all_stop_has144_explicit_skips_and_no_model_forward(tmp_path, monkeypatch):
    m, model, calls = manifest(), fake_model(), []
    source = saved_sources(tmp_path, m, all_stop=True)
    monkeypatch.setattr(capture, "_logit_hashes", fake_forward(calls))
    receipts = capture.execute_receipts(tmp_path, m, model, source)
    assert len(receipts) == 144 and not calls
    assert all(
        r["status"] == "stop" and r["receipt_input"] is None and not r["receipt_model_pass_completed"]
        for r in receipts.values()
    )
    assert all(read(p)["engineering"] is None for p in (tmp_path / "receipt_cohorts").glob("*/*.json"))


@pytest.mark.parametrize("problem", ["forward_failure", "full_logits_changed"])
def test_receipt_failure_is_saved_and_never_becomes_a_success_or_retry(tmp_path, monkeypatch, problem):
    m, model, calls = manifest(), fake_model(), []
    source = saved_sources(tmp_path, m)
    monkeypatch.setattr(
        capture,
        "_logit_hashes",
        fake_forward(calls, fail=problem == "forward_failure", corrupt=problem == "full_logits_changed"),
    )
    with pytest.raises((RuntimeError, ValueError)):
        capture.execute_receipts(tmp_path, m, model, source)
    envelopes = [read(path) for path in (tmp_path / "receipt_cohorts").glob("*/*.json")]
    assert len(envelopes) == 1 and len(envelopes[0]["records"]) == 32
    failed = [r for r in envelopes[0]["records"] if r["receipt_required"]]
    assert failed and all(
        r["status"] == "execution_error"
        and not r["receipt_model_pass_completed"]
        and r["activation_artifact"] is None
        for r in failed
    )
    count = len(calls)
    with pytest.raises(RuntimeError, match="cannot be retried"):
        capture.execute_receipts(tmp_path, m, model, source)
    assert len(calls) == count


def test_deadline_preserves_missing_receipts_instead_of_publishing_success(tmp_path, monkeypatch):
    m, model, calls = manifest(), fake_model(), []
    source = saved_sources(tmp_path, m)
    monkeypatch.setattr(capture, "_logit_hashes", fake_forward(calls))
    with pytest.raises(RuntimeError, match="missing logical choices"):
        capture.execute_receipts(tmp_path, m, model, source, deadline=0)
    assert not calls and not list((tmp_path / "receipt_cohorts").glob("*/*.json"))


@pytest.mark.parametrize(
    "tamper", ["bare_pass", "different_logits", "wrong_capture_row", "false_receipt_flags"]
)
def test_resume_rejects_fabricated_completion_even_with_rehashed_envelope(tmp_path, monkeypatch, tamper):
    m, model, calls = manifest(), fake_model(), []
    source = saved_sources(tmp_path, m)
    monkeypatch.setattr(capture, "_logit_hashes", fake_forward(calls))
    capture.execute_receipts(tmp_path, m, model, source)
    path = sorted((tmp_path / "receipt_cohorts").glob("*/*.json"))[0]
    envelope = read(path)
    active = next(row for row in envelope["records"] if row["receipt_required"])
    if tamper == "bare_pass":
        envelope["engineering"]["checks"]["full_initial_logits_equal"] = False
    elif tamper == "different_logits":
        envelope["engineering"]["captured_full_initial_logits_sha256"][-1] = "changed"
    elif tamper == "wrong_capture_row":
        active["activation_artifact"]["row"] += 1
    else:
        active["executed_simulator_outcome"]["receipt_model_pass_completed"] = False
    envelope["records_sha256"] = digest(envelope["records"])
    path.write_text(json.dumps(envelope))
    count = len(calls)
    with pytest.raises(ValueError):
        capture.execute_receipts(tmp_path, m, model, source)
    assert len(calls) == count


def test_wrong_source_payload_and_missing_initial_site_fail_before_receipt(tmp_path, monkeypatch):
    m, model, calls = manifest(), fake_model(), []
    source = saved_sources(tmp_path, m)
    a = next(a for a in m["diagnostics"] if a["kind"] == "choice")
    tampered = deepcopy(source[a["trial_id"]])
    if tampered["simulator_outcome"] is None:
        tampered = source_record(a, "report")
    tampered["simulator_outcome"]["payload"] = "Unselected forged fact."
    with pytest.raises(ValueError, match="actual closed"):
        capture.choice_sidecar(tampered)
    source[a["trial_id"]]["activation_artifact"]["available"][0][0] = False
    monkeypatch.setattr(thinking, "execute", lambda *args: source)
    monkeypatch.setattr(capture, "_logit_hashes", fake_forward(calls))
    with pytest.raises(ValueError, match="eight frozen"):
        capture.execute(tmp_path, m, model)
    assert not calls


def test_receipt_plan_missing_or_replaced_is_not_an_implicit_default():
    m = manifest()
    assert len(capture.validate_manifest(m)) == 288
    del m["receipt_capture_plan"]
    with pytest.raises(ValueError, match="explicit and frozen"):
        capture.validate_manifest(m)


def test_history_exclusions_cannot_disappear_at_execution():
    m = manifest()
    del m["world_selection"]
    with pytest.raises(ValueError, match="historical world exclusion"):
        capture.validate_manifest(m)
