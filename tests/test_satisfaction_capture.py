"""Synthetic screen execution only; no real inventory, selected dataset, or GPU."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from curiosity_study import satisfaction_capture as capture
from curiosity_study import satisfaction_screen as screen
from curiosity_study import sustained_engineering as engineering
from curiosity_study import voluntary_resolution as voluntary
from curiosity_study.artifacts import RunStore, digest, read, write_once
from test_concept_capture import WordTokenizer
from test_satisfaction_screen import direction
from test_sustained_engineering import inputs, model_and_direction
from test_voluntary_capture import fake_forward


def fake_model():
    model = SimpleNamespace(
        torch=torch,
        device="cpu",
        active_hooks=0,
        tokenizer=WordTokenizer(),
        model=SimpleNamespace(
            config=SimpleNamespace(vocab_size=10000, hidden_size=4),
            generation_config=SimpleNamespace(eos_token_id=9999),
            named_modules=lambda: [],
        ),
        metadata=lambda: {**capture.MODEL, "synthetic": True},
    )

    def encode(messages, enable_thinking=False):
        chunks = []
        for message in messages:
            text = message["content"]
            if message["role"] == "assistant":
                text = text.rsplit("</think>", 1)[-1].strip()
            chunks.append(f"<{message['role']}>\n{text}\n</message>")
        suffix = "<think>" if enable_thinking else "<think></think>"
        rendered = "\n".join(chunks) + "\n<assistant>\n" + suffix
        return rendered, SimpleNamespace(input_ids=torch.tensor([model.tokenizer.encode(rendered)]))

    model.encode = encode
    return model


@pytest.fixture(scope="module")
def frozen():
    plan = screen.assignment_plan(historical_exact_world_ids=[], direction_binding=direction())
    model = fake_model()
    preflight = capture.bind_preflight(plan, model)
    return plan, preflight, model


def manifest(frozen, group="baseline_zero"):
    plan, preflight, _ = frozen
    rows = [a for a in plan["assignments"] if a["arm"] in capture.GROUPS[group]]
    return dict(
        assay="satisfaction_screen",
        task_revision=screen.REVISION,
        model=capture.MODEL,
        screen_plan=plan,
        screen_plan_sha256=digest(plan),
        execution_group=group,
        parent_manifest_sha256=digest({"synthetic_parent_plan": plan}),
        assignments=[],
        diagnostics=rows,
        assignment_sha256=digest(rows),
        execution=screen.EXECUTION,
        max_new_tokens=4096,
        capture_plan=capture.CAPTURE_PLAN,
        generation_preflight=preflight,
        generation_preflight_sha256=digest(preflight),
    )


def install_fake_generation(monkeypatch, frozen, calls, *, corrupt_sham=False, unavailable=False):
    base = {a["base_trial_id"]: a for a in frozen[0]["base_assignments"]}

    def collect(model, requests, patches, **kwargs):
        calls.append([r["source_trial_id"] for r in requests])
        outputs = []
        for i, request in enumerate(requests):
            source = base[request["source_trial_id"]]["source_assignment"]
            if source["kind"] == "choice":
                # Every source arm includes required, stopped and invalid dispositions.
                service = "stop" if source["world_slot"] % 3 == 0 else "report"
                answer = next(k for k, v in voluntary.task(source)["action_mapping"].items() if v == service)
                raw = "Still thinking" if source["world_slot"] % 3 == 2 else f"Reasoning </think> {answer}."
            else:
                raw = f"Reasoning </think> {voluntary.prompt(source)[2]}."
            outputs.append(
                dict(
                    token_prefix_sha256=request["token_prefix_sha256"],
                    hooks_remaining=0,
                    intervention=None if patches[i] is None else {"dose": patches[i]["dose"]},
                    intervention_trace=dict(schedule=screen.SCHEDULE, events=[]),
                    completion=dict(
                        raw=raw,
                        generated_token_ids=[2, 9999],
                        ended_with_eos=True,
                        sampling=dict(seed=request["seed"], temperature=0.7, top_p=1.0, max_new_tokens=4096),
                    ),
                )
            )
        hashes = [digest(r["token_prefix"]) for r in requests]
        if corrupt_sham and patches[0] is not None and patches[0]["dose"] == 0:
            hashes[-1] = "f" * 64
        frames = [
            dict(
                forward_index=0,
                prediction_ordinal=1,
                full_fp32_logits_sha256=hashes,
                input_ids=[r["token_prefix"] for r in requests],
                position_ids=[list(range(len(r["token_prefix"]))) for r in requests],
                attention_mask_row_sha256=[digest([1] * len(r["token_prefix"])) for r in requests],
                active_prediction_rows=[True] * 32,
            )
        ]
        return dict(
            rows=None if unavailable else outputs,
            forwards=frames,
            attempted_forwards=1,
            hooks_clean=True,
            error={"type": "SyntheticFailure"} if unavailable else None,
        ), np.arange(32 * 2 * 4, dtype=np.float32).reshape(32, 1, 2, 4)

    monkeypatch.setattr(capture, "collect_generation", collect)
    monkeypatch.setattr(
        engineering, "validate_condition", lambda condition, *a, **k: condition["rows"] is not None
    )


def install_fake_receipts(monkeypatch, root, calls, *, corrupt=False, fail=False):
    original = fake_forward(calls, corrupt=corrupt, fail=fail)

    def observed(model, requests, layers):
        m = read(root / "manifest.json")
        assert len(RunStore(root, m).completed()) == len(m["diagnostics"])
        assert not model.active_hooks
        assert list(layers) in ([], [15])
        return original(model, requests, layers)

    monkeypatch.setattr(capture.receipts, "_logit_hashes", observed)


def test_preflight_has_exact288_inputs_and1152_outer_bindings_with_reused_seeds(frozen):
    plan, preflight, model = frozen
    assert capture.bind_preflight(plan, model) == preflight
    assert len(preflight["requests"]) == 288
    assert len(preflight["outer_request_bindings"]) == 1152
    for row in preflight["requests"]:
        request = row["request"]
        position = request["prefix_positions"]["record"]
        start, end = row["offset_mapping"][position]
        assert row["rendered_prompt"][start:end] == "record."
        assert request["prefix_positions"]["final"] == len(request["token_prefix"]) - 1
    for group in capture.GROUPS:
        assert len(capture.validate_manifest(manifest(frozen, group))) == 288 * len(capture.GROUPS[group])


@pytest.mark.parametrize("tamper", ["rows", "seed", "capture", "request", "outer"])
def test_manifest_rejects_reordered_companions_changed_seeds_sites_or_request_bindings(frozen, tamper):
    m = deepcopy(manifest(frozen))
    if tamper == "rows":
        m["diagnostics"][0], m["diagnostics"][1] = m["diagnostics"][1], m["diagnostics"][0]
        m["assignment_sha256"] = digest(m["diagnostics"])
    elif tamper == "capture":
        m["capture_plan"]["layers"] = [15, 31]
    else:
        p = m["generation_preflight"]
        if tamper == "seed":
            p["requests"][0]["request"]["seed"] += 1
        elif tamper == "request":
            p["requests"][0]["request"]["token_prefix"][0] += 1
        else:
            p["outer_request_bindings"][0]["trial_id"] = "unassigned"
        p["requests_sha256"] = digest(p["requests"])
        p["outer_request_bindings_sha256"] = digest(p["outer_request_bindings"])
        m["generation_preflight_sha256"] = digest(p)
    with pytest.raises(ValueError):
        capture.validate_manifest(m)


@pytest.mark.parametrize("group", capture.GROUPS)
def test_complete_child_retains_every_generation_and_choice_slot_with_actual_receipt_evidence(
    tmp_path, monkeypatch, frozen, group
):
    m = manifest(frozen, group)
    generation_calls, receipt_calls = [], []
    install_fake_generation(monkeypatch, frozen, generation_calls)
    install_fake_receipts(monkeypatch, tmp_path, receipt_calls)
    rows = capture.execute(tmp_path, m, frozen[2])
    arm_count = len(capture.GROUPS[group])
    assert len(rows) == 288 * arm_count and len(generation_calls) == 9 * arm_count
    assert all(r["engineering_passed"] and r["task_record"] is not None for r in rows)
    sidecars = [
        row
        for path in sorted((tmp_path / "receipt_cohorts").glob("*/*.json"))
        for row in read(path)["records"]
    ]
    assert len(sidecars) == len({r["source_trial_id"] for r in sidecars}) == 144 * arm_count
    assert len({r["inner_source_trial_id"] for r in sidecars}) == 144
    assert {r["status"] for r in sidecars} == {"completed", "stop", "invalid_choice"}
    assert receipt_calls and len(receipt_calls) % 2 == 0
    for row in sidecars:
        source = next(r for r in rows if r["trial_id"] == row["source_trial_id"])
        assert row["source_record_sha256"] == digest(source)
        assert row["source_task_record_sha256"] == digest(source["task_record"])
        if row["status"] == "completed":
            assert row["receipt_model_pass_completed"]
            assert row["executed_simulator_outcome"]["receipt_model_pass_completed"]
            assert not source["task_record"]["simulator_outcome"]["receipt_model_pass_completed"]
            assert row["receipt_input"]["source_trial_id"] == row["inner_source_trial_id"]
            assert not row["token_binding"]["exact_source_token_replay"]
        else:
            assert not row["receipt_model_pass_completed"] and row["measurement"] is None
    generation_count, receipt_count = len(generation_calls), len(receipt_calls)
    assert digest(capture.execute(tmp_path, m, frozen[2])) == digest(rows)
    assert len(generation_calls) == generation_count and len(receipt_calls) == receipt_count
    assert len(list((tmp_path / "generation_evidence").glob("*/*.json"))) == 9 * arm_count
    envelopes = sorted(
        [read(p) for p in (tmp_path / "receipt_cohorts").glob("*/*.json")], key=lambda e: e["choice_offset"]
    )
    assert [len(e["records"]) for e in envelopes] == ([32] * 9 if arm_count == 2 else [32] * 4 + [16])
    if arm_count == 1:
        assert envelopes[-1]["choice_offset"] == 128
        expected = [a["trial_id"] for a in m["diagnostics"] if a["kind"] == "choice"][128:]
        assert envelopes[-1]["source_trial_ids"] == expected
        assert all(len(e["active_source_trial_ids"]) <= len(e["source_trial_ids"]) for e in envelopes)
    for p in (tmp_path / "execution_profiles").glob("*.json"):
        profile = read(p)
        assert profile["generation_assigned"] == 288 * arm_count
        assert profile["choice_receipts_assigned"] == 144 * arm_count
        assert profile["parent_generations_required"] == 1152
        assert profile["parent_choice_receipts_required"] == 576
    if group == "baseline_zero":
        shams = [
            read(p) for p in (tmp_path / "generation_evidence").glob("*/*.json") if read(p)["sham_comparison"]
        ]
        assert len(shams) == 9 and all(s["checks"]["sham_identity"] for s in shams)


def test_sham_same_answers_but_changed_full_logits_explicitly_fails_without_dropping_rows(
    tmp_path, monkeypatch, frozen
):
    calls = []
    install_fake_generation(monkeypatch, frozen, calls, corrupt_sham=True)
    install_fake_receipts(monkeypatch, tmp_path, [])
    rows = capture.execute(tmp_path, manifest(frozen), frozen[2])
    assert len(rows) == 576
    assert all(r["engineering_passed"] for r in rows[:288])
    assert all(
        not r["engineering_passed"] and r["failure"] == "generation_engineering_guard_failed"
        for r in rows[288:]
    )
    assert [r["task_record"]["inference"] for r in rows[:288]] == [
        r["task_record"]["inference"] for r in rows[288:]
    ]
    assert len(list((tmp_path / "receipt_cohorts").glob("*/*.json"))) == 9


@pytest.mark.parametrize("failure", ["exception", "identity"])
def test_receipt_failure_is_terminal_bound_and_never_retried(tmp_path, monkeypatch, frozen, failure):
    install_fake_generation(monkeypatch, frozen, [])
    calls = []
    install_fake_receipts(
        monkeypatch, tmp_path, calls, fail=failure == "exception", corrupt=failure == "identity"
    )
    capture.execute(tmp_path, manifest(frozen), frozen[2])
    envelopes = [read(p) for p in (tmp_path / "receipt_cohorts").glob("*/*.json")]
    assert len(envelopes) == 1
    active = [r for r in envelopes[0]["records"] if r["receipt_required"]]
    assert active and all(
        r["status"] == "execution_error" and not r["receipt_model_pass_completed"] for r in active
    )
    assert all(r["measurement"] is None and r["activation_artifact"] is None for r in active)
    assert all(not r["executed_simulator_outcome"]["receipt_model_pass_completed"] for r in active)
    count = len(calls)
    with pytest.raises(RuntimeError, match="cannot be retried"):
        capture.execute(tmp_path, manifest(frozen), frozen[2])
    assert len(calls) == count


def test_unavailable_generation_persists32_errors_and_partial_evidence_without_receipt_or_retry(
    tmp_path, monkeypatch, frozen
):
    calls = []
    install_fake_generation(monkeypatch, frozen, calls, unavailable=True)
    rows = capture.execute(tmp_path, manifest(frozen), frozen[2])
    assert len(rows) == 32 and all(
        r["status"] == "execution_error" and r["task_record"] is None for r in rows
    )
    assert len(list((tmp_path / "generation_evidence").glob("*/*.json"))) == 1
    assert not list((tmp_path / "receipt_cohorts").glob("*/*.json"))
    with pytest.raises(RuntimeError, match="cannot be retried"):
        capture.execute(tmp_path, manifest(frozen), frozen[2])
    assert len(calls) == 1


def test_interrupted_unpublished_cohort_cannot_be_paid_retried(tmp_path, frozen):
    m = manifest(frozen)
    RunStore(tmp_path, m)
    write_once(tmp_path / "generation_started" / "0000000.json", {"interrupted": True})
    with pytest.raises(ValueError, match="interrupted generation"):
        capture.execute(tmp_path, m, frozen[2])


def test_deadline_has_no_forward_and_keeps_missing_coverage(tmp_path, monkeypatch, frozen):
    calls = []
    install_fake_generation(monkeypatch, frozen, calls)
    assert capture.execute(tmp_path, manifest(frozen), frozen[2], deadline=0) == []
    assert calls == []
    assert not list((tmp_path / "generation_started").glob("*.json"))


def test_real_bf16_capture_is_before_patch_and_baseline_sham_match_full_logits():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model, direction = model_and_direction()
        requests = inputs()
        baseline, hidden = capture.collect_generation(model, requests, [None] * 32, cap=3)
        zero = [dict(layer=15, position="record", mode="add", direction=direction, dose=0.0)] * 32
        sham, zero_hidden = capture.collect_generation(model, requests, zero, cap=3)
        positive = [dict(layer=15, position="record", mode="add", direction=direction, dose=0.5)] * 32
        steered, steered_hidden = capture.collect_generation(model, requests, positive, cap=3)
        assert engineering.identical_rows(baseline, sham, range(32))
        assert np.array_equal(hidden, zero_hidden) and np.array_equal(hidden, steered_hidden)
        assert hidden.shape == (32, 1, 2, 16)
        assert steered["capture_phase"] == "prefill_before_layer15_patch"
        assert all(r["intervention_trace"]["total_patch_count"] == 3 for r in steered["rows"])
        assert all(r["intervention_trace"]["events"][0]["delta_norm"] > 0 for r in steered["rows"])
        for index, r in enumerate(requests):
            pad = steered["prefill_width"] - len(r["token_prefix"])
            assert steered["padded_token_positions"][index] == [
                pad + r["prefix_positions"][p] for p in capture.POSITIONS
            ]
        assert not engineering.hook_inventory(model) and model.active_hooks == 0
        bad = [{**positive[0], "direction": [1.0]}] + positive[1:]
        failed, _ = capture.collect_generation(model, requests, bad, cap=3)
        assert failed["rows"] is None and failed["error"]["type"] == "ValueError"
        assert not engineering.hook_inventory(model) and model.active_hooks == 0
        clean, _ = capture.collect_generation(model, requests, [None] * 32, cap=3)
        assert engineering.identical_rows(baseline, clean, range(32))
    finally:
        torch.set_num_threads(old)


def test_failure_after_first_receipt_update_resets_nested_completion_flags(tmp_path, monkeypatch, frozen):
    m = manifest(frozen)
    sources = {}
    for row in m["diagnostics"]:
        if row["kind"] != "choice":
            continue
        a = row["source_assignment"]
        letter = next(k for k, v in voluntary.task(a)["action_mapping"].items() if v == "report")
        inner = voluntary.record_from_thinking_inference(
            a, dict(raw=f"Reasoning </think> {letter}.", ended_with_eos=True, generated_token_ids=[2, 9999])
        )
        sources[row["trial_id"]] = dict(trial_id=row["trial_id"], assignment=row, task_record=inner)
    monkeypatch.setattr(capture.receipts, "_logit_hashes", fake_forward([]))
    actual_copy, count = capture.deepcopy, 0

    def fail_second_outcome(value):
        nonlocal count
        if isinstance(value, dict) and "observation_ledger" in value:
            count += 1
            if count == 2:
                raise RuntimeError("Synthetic interruption after one updated row")
        return actual_copy(value)

    monkeypatch.setattr(capture, "deepcopy", fail_second_outcome)
    assert not capture.execute_receipts(tmp_path, m, frozen[2], sources, "synthetic")
    envelope = read(next((tmp_path / "receipt_cohorts").glob("*/*.json")))
    assert envelope["observed_activation_artifact"] is not None
    assert len(envelope["observed_measurements"]) == 32
    assert all(
        row["status"] == "execution_error" and not row["receipt_model_pass_completed"]
        for row in envelope["records"]
    )
    assert all(
        not row["executed_simulator_outcome"]["receipt_model_pass_completed"] for row in envelope["records"]
    )
    assert all(
        not row["executed_simulator_outcome"]["observation_ledger"][-1]["model_pass_completed"]
        for row in envelope["records"]
    )


def test_unexpected_generation_wrapper_exception_becomes_durable_unavailable_evidence(
    tmp_path, monkeypatch, frozen
):
    def fail(*args, **kwargs):
        raise ValueError("Synthetic malformed bound request")

    monkeypatch.setattr(capture, "collect_generation", fail)
    rows = capture.execute(tmp_path, manifest(frozen), frozen[2])
    assert len(rows) == 32 and all(r["task_record"] is None for r in rows)
    evidence = read(tmp_path / rows[0]["generation_evidence"]["path"])
    assert evidence["condition"]["observations_unavailable"]
    assert evidence["condition"]["attempted_forwards"] is None
    assert not evidence["engineering_passed"]
    assert evidence["eos_token_ids"] == [9999] and evidence["pad_token_id"] == 0
    choice = next(r for r in rows if r["kind"] == "choice")
    sidecar = capture._choice_sidecar(choice)
    assert sidecar["status"] == "execution_error" and sidecar["receipt_required"] is None
    assert sidecar["selected_service"] is None and sidecar["simulator_outcome_sha256"] == digest(None)


def test_worker_retokenization_rejects_changed_exact_prefix_before_forward(tmp_path, monkeypatch, frozen):
    model = fake_model()
    model.tokenizer.words["A"] = 7777
    with pytest.raises(ValueError, match="tokenizer differs"):
        capture.execute(tmp_path, manifest(frozen), model)
    assert not (tmp_path / "generation_started").exists()


def test_real_layer15_selected_receipt_has_actual_max0_identity_and_exact_payload_span(frozen):
    from transformers import Qwen3ForCausalLM
    from test_voluntary_capture import with_encoder

    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model, _ = model_and_direction()
        config = deepcopy(model.model.config)
        config.vocab_size = 1024
        torch.manual_seed(321)
        model.model = Qwen3ForCausalLM(config).to(torch.bfloat16).eval()
        model.layers = model.model.model.layers
        model.tokenizer = WordTokenizer()
        with_encoder(model)
        requests, outer_ids = [], []
        for row in [a for a in frozen[0]["assignments"][:288] if a["kind"] == "choice"][:4]:
            source = row["source_assignment"]
            label = next(k for k, v in voluntary.task(source)["action_mapping"].items() if v == "report")
            inner = voluntary.record_from_thinking_inference(
                source, dict(raw=f"Reasoning </think> {label}.", ended_with_eos=True, generated_token_ids=[3])
            )
            receipt, request, binding = capture.receipts.render_receipt(inner, model)
            start, end = binding["selected_payload_span"]
            assert binding["rendered_prompt"][start:end] == inner["simulator_outcome"]["payload"]
            assert receipt["max_new_tokens"] == 0 and not receipt["model_pass_completed"]
            requests.append(request)
            outer_ids.append(row["trial_id"])
        measured, hidden, guard = capture.receipts.forward_receipts(
            model, requests, outer_ids, layers=capture.LAYERS
        )
        assert guard["passed"] and guard["source_trial_ids"] == outer_ids
        assert hidden.shape == (4, 1, 2, 16)
        assert all(r["intervention"] is None and "completion" not in r for r in measured)
        assert not engineering.hook_inventory(model) and model.active_hooks == 0
    finally:
        torch.set_num_threads(old)
