"""Synthetic structural/scoring evidence; no paid model or behavioral outcomes."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from curiosity_study import thinking_information as thinking
from curiosity_study import voluntary_capture as capture
from curiosity_study import voluntary_resolution as voluntary
from curiosity_study.artifacts import digest, read, write_once

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "audit_voluntary_resolution", SCRIPTS / "audit_voluntary_resolution.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class Tokenizer:
    pad_token_id = 0

    def __len__(self):
        return 1000000

    def __call__(self, text, **kwargs):
        ids, offsets, index = [], [], 0
        while index < len(text):
            end = index + (8 if text.startswith("</think>", index) else 1)
            ids.append(1 if end - index == 8 else ord(text[index]) + 10)
            offsets.append((index, end))
            index = end
        return dict(input_ids=ids, offset_mapping=offsets)

    def encode(self, text, **kwargs):
        return self(text)["input_ids"]

    def decode(self, tokens, skip_special_tokens=False):
        return "".join(
            "</think>"
            if t == 1
            else ""
            if t == 2 and skip_special_tokens
            else "<eos>"
            if t == 2
            else chr(t - 10)
            for t in tokens
            if t
        )

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    ):
        assert not tokenize and add_generation_prompt
        return "\n".join(f"<{row['role']}>{row['content']}" for row in messages) + (
            "\n<assistant><think>" if enable_thinking else "\n<assistant>"
        )


def source(kind="choice", service="report", valid=True, family="objects", delivery=None, assignment=None):
    a = assignment or next(
        a
        for a in voluntary.assignments()
        if a["kind"] == kind and a["family"] == family and (delivery is None or a["delivery"] == delivery)
    )
    if kind == "choice":
        label = next(label for label, name in voluntary.task(a)["action_mapping"].items() if name == service)
    else:
        label = voluntary.prompt(a)[2]
    tokenizer = Tokenizer()
    text = tokenizer.apply_chat_template(thinking.inference_request(a)["messages"], enable_thinking=True)
    ids = tokenizer.encode(text)
    raw = "</think>\n" + (label if valid else "not a label")
    inference = dict(
        raw=raw,
        generated_token_ids=tokenizer.encode(raw) + [2],
        ended_with_eos=True,
        messages=thinking.inference_request(a)["messages"],
        rendered_prompt=text,
        prompt_sha256=digest(text),
        unpadded_input_ids=ids,
        input_ids=ids,
        attention_mask=[1] * len(ids),
        position_ids=list(range(len(ids))),
        enable_thinking=True,
        intervention=None,
        hooks_remaining=0,
        capture={},
        sampling=dict(seed=a["sampling_seed"], temperature=0.7, top_p=1.0, max_new_tokens=4096),
    )
    record = thinking.record_from_inference(a, inference)
    record_position = text.index("End of record.") + len("End of record.") - 1
    positions = [record_position, len(ids) - 1, -1, -1, -1, len(ids), len(ids) + 1]
    available = np.tile(np.array(positions) >= 0, (4, 1))
    hidden = np.ones((1, 4, 7, 3), dtype=np.float32)
    hidden[0, ~available] = np.nan
    arrays = dict(
        hidden=hidden,
        available=available[None],
        layers=np.array(capture.LAYERS),
        positions=np.array(
            ["record", "prefill", "token_32", "token_128", "token_512", "think_end", "decision"]
        ),
        trial_ids=np.array([a["trial_id"]]),
        token_positions=np.array([positions]),
        closure_counts=np.array([1]),
        decision_labels=np.array([list("ABCD")]),
        decision_label_ids=np.array([[tokenizer.encode(label)[0] for label in "ABCD"]]),
        decision_label_logits=np.zeros((1, 4), np.float32),
        decision_label_probabilities_temperature_1=np.full((1, 4), 0.125, np.float32),
    )
    record.update(
        activation_artifact=dict(
            path="activations/test/0000000.npz",
            sha256="a" * 64,
            row=0,
            layers=capture.LAYERS,
            positions=arrays["positions"].tolist(),
            token_positions=positions,
            available=available.tolist(),
            max_new_tokens=4096,
        ),
        closing_delimiter_count=1,
        decision_label_logits=dict.fromkeys("ABCD", 0.0),
        decision_label_probabilities_temperature_1=dict.fromkeys("ABCD", 0.125),
    )
    return a, record, arrays, tokenizer


def successful_receipt(record, tokenizer):
    sidecar = capture.choice_sidecar(record)
    receipt, binding = audit.receipt_tokens(record, tokenizer)
    ids = binding["unpadded_input_ids"]
    width = ((len(ids) + 127) // 128) * 128
    pad = width - len(ids)
    positions = binding["prefix_positions"]
    hidden = np.ones((1, 4, 2, 3), np.float32)
    arrays = dict(
        hidden=hidden,
        available=np.ones((1, 4, 2), bool),
        trial_ids=np.array([record["trial_id"]]),
        layers=np.array(capture.LAYERS),
        positions=np.array(capture.POSITIONS),
        token_positions=np.array([list(positions.values())]),
        token_prefix_sha256=np.array([digest(ids)]),
    )
    outcome = deepcopy(record["simulator_outcome"])
    outcome["receipt_model_pass_completed"] = True
    outcome["observation_ledger"][-1]["model_pass_completed"] = True
    sidecar.update(
        status="completed",
        receipt_model_pass_completed=True,
        receipt_input=receipt,
        token_binding=binding,
        executed_simulator_outcome=outcome,
        measurement=dict(
            unpadded_input_ids=ids,
            token_prefix_sha256=digest(ids),
            input_ids=[0] * pad + ids,
            attention_mask=[0] * pad + [1] * len(ids),
            position_ids=[0] * pad + list(range(len(ids))),
            prefill_width=width,
            prefill_padding_multiple=128,
            unpadded_token_positions=positions,
            record_token_index=pad + positions["record"],
            exact_token_replay=True,
            rendered_prompt=tokenizer.decode(ids),
            prompt_sha256=digest(tokenizer.decode(ids)),
            hooks_remaining=0,
            intervention=None,
            label_logits=dict(A=0.0, B=0.0),
            label_probabilities_temperature_1=dict(A=0.1, B=0.1),
        ),
        activation_artifact=dict(
            path="receipt_activations/test/0000000.npz",
            sha256="b" * 64,
            row=0,
            layers=capture.LAYERS,
            positions=capture.POSITIONS,
            token_positions=list(positions.values()),
            available=[[True, True]] * 4,
        ),
    )
    engineering = dict(
        passed=True,
        checks=dict.fromkeys(
            (
                "row_coverage",
                "full_initial_logits_equal",
                "every_row_fingerprint_equal",
                "all_captures_finite",
                "capture_shape",
                "no_intervention_or_generation",
                "hook_cleanup",
            ),
            True,
        ),
        baseline_full_initial_logits_sha256=["a" * 64],
        captured_full_initial_logits_sha256=["a" * 64],
        source_trial_ids=[record["trial_id"]],
        token_prefix_sha256=[digest(ids)],
        layers=capture.LAYERS,
        positions=capture.POSITIONS,
        active_batch_size=1,
        prefill_width=width,
        max_new_tokens=0,
        captured_array_sha256=capture.array_sha(hidden),
    )
    return sidecar, arrays, width, engineering


def complete_readiness():
    assignments = voluntary.assignments()
    records = {
        a["trial_id"]: dict(
            valid=True, correct=None if a["kind"] == "choice" else True, thinking_closed=True, cap_hit=False
        )
        for a in assignments
    }
    sidecars = {
        a["trial_id"]: dict(receipt_required=False, receipt_model_pass_completed=False, status="stop")
        for a in assignments
        if a["kind"] == "choice"
    }
    return assignments, records, sidecars


def test_exact_readiness_cells_include_every_assignment_and_no_choice_correctness():
    assignments, records, sidecars = complete_readiness()
    result = audit.readiness(assignments, records, sidecars)
    assert result["passed"] and len(result["diagnostic_cells"]) == 12 and len(result["choice_cells"]) == 6
    assert (
        result["assigned"] == 288 and result["choice_correctness"] is None and not result["causal_admission"]
    )
    assert all("correct" not in cell for cell in result["choice_cells"])


def test_one_choice_invalid_is_retained_and_allowed_two_fail_same_assigned_cell():
    assignments, records, sidecars = complete_readiness()
    selected = [
        a
        for a in assignments
        if a["kind"] == "choice" and a["family"] == "objects" and a["delivery"] == "decisive_evidence"
    ]
    records[selected[0]["trial_id"]]["valid"] = False
    assert audit.readiness(assignments, records, sidecars)["passed"]
    records[selected[1]["trial_id"]]["valid"] = False
    result = audit.readiness(assignments, records, sidecars)
    cell = next(
        c for c in result["choice_cells"] if c["family"] == "objects" and c["delivery"] == "decisive_evidence"
    )
    assert not result["passed"] and cell["assigned"] == 24 and cell["valid"] == 22


@pytest.mark.parametrize("change", ["missing", "failure", "cap", "receipt_missing", "receipt_failure"])
def test_operational_loss_never_excludes_an_assignment(change):
    assignments, records, sidecars = complete_readiness()
    tid = next(iter(sidecars))
    if change == "missing":
        records.pop(tid)
    elif change == "failure":
        records[tid]["failure"] = "execution_error"
    elif change == "cap":
        records[tid]["cap_hit"] = True
    elif change == "receipt_missing":
        sidecars.pop(tid)
    else:
        sidecars[tid].update(status="execution_error", receipt_required=True)
    result = audit.readiness(assignments, records, sidecars)
    assert not result["passed"] and result["assigned"] == 288
    assert sum(c["assigned"] for c in result["choice_cells"]) == 144


def test_diagnostic_one_error_allowed_but_one_invalid_or_two_errors_fails():
    assignments, records, sidecars = complete_readiness()
    rows = [a for a in assignments if a["kind"] == "repeated_fact" and a["family"] == "machines"]
    records[rows[0]["trial_id"]]["correct"] = False
    assert audit.readiness(assignments, records, sidecars)["passed"]
    records[rows[0]["trial_id"]]["valid"] = False
    assert not audit.readiness(assignments, records, sidecars)["passed"]
    records[rows[0]["trial_id"]]["valid"] = True
    records[rows[1]["trial_id"]]["correct"] = False
    assert not audit.readiness(assignments, records, sidecars)["passed"]


@pytest.mark.parametrize("kind", ["choice", *voluntary.DIAGNOSTICS])
def test_generation_recomputed_score_tokens_finite_world_and_every_position(kind):
    a, record, arrays, tokenizer = source(kind)
    binding = audit.generation_record(
        record,
        a,
        tokenizer,
        {2},
        len(tokenizer),
        len(record["inference"]["input_ids"]),
        arrays,
        0,
        dict(capture_plan=thinking.CAPTURE_PLAN),
    )
    assert binding["record_sha256"] == digest(record)


@pytest.mark.parametrize("tamper", ["answer", "payload", "decision", "missing", "closure", "eos", "padding"])
def test_generation_rejects_rehashed_semantic_token_and_capture_tampering(tamper):
    a, record, arrays, tokenizer = source()
    if tamper == "answer":
        record["parsed_answer"] = "!"
    elif tamper == "payload":
        record["simulator_outcome"]["payload"] += " extra fact"
    elif tamper == "decision":
        arrays["token_positions"][0, -1] -= 1
        record["activation_artifact"]["token_positions"][-1] -= 1
    elif tamper == "missing":
        arrays["hidden"][0, 0, 2] = 0
    elif tamper == "closure":
        record["closing_delimiter_count"] = 2
    elif tamper == "eos":
        record["inference"]["generated_token_ids"].insert(0, 2)
    else:
        record["inference"]["position_ids"][0] = 3
    with pytest.raises(ValueError):
        audit.generation_record(
            record,
            a,
            tokenizer,
            {2},
            len(tokenizer),
            len(record["inference"]["input_ids"]),
            arrays,
            0,
            dict(capture_plan=thinking.CAPTURE_PLAN),
        )


@pytest.mark.parametrize("service", ["report", "repeat", "pattern"])
def test_only_actual_selected_payload_can_receive_completed_receipt(service):
    _, record, _, tokenizer = source(service=service)
    before = deepcopy(record)
    sidecar, arrays, width, engineering = successful_receipt(record, tokenizer)
    binding = audit.receipt_record(sidecar, record, tokenizer, width, arrays, 0)
    audit.receipt_engineering(
        engineering, [record["trial_id"]], [sidecar["token_binding"]["token_prefix_sha256"]], arrays, width
    )
    assert binding["source_record_sha256"] == digest(record) and record == before
    assert record["simulator_outcome"]["receipt_model_pass_completed"] is False


@pytest.mark.parametrize("valid,service", [(True, "stop"), (False, "report")])
def test_stop_and_invalid_choices_have_no_forward_evidence(valid, service):
    _, record, _, tokenizer = source(service=service, valid=valid)
    sidecar = capture.choice_sidecar(record)
    assert audit.receipt_record(sidecar, record, tokenizer) is None
    sidecar["receipt_model_pass_completed"] = True
    with pytest.raises(ValueError, match="Skipped receipt"):
        audit.receipt_record(sidecar, record, tokenizer)


@pytest.mark.parametrize(
    "tamper",
    [
        "payload",
        "wrong_service",
        "replay_claim",
        "token",
        "position",
        "extra_event",
        "generated",
        "row",
        "sourcehash",
        "no_capture",
    ],
)
def test_completed_receipt_rejects_changed_payload_false_receipt_or_capture(tamper):
    _, record, _, tokenizer = source()
    sidecar, arrays, width, _ = successful_receipt(record, tokenizer)
    if tamper == "payload":
        sidecar["receipt_input"]["messages"][-1]["content"] += " another unselected report"
    elif tamper == "wrong_service":
        sidecar["selected_service"] = "pattern"
    elif tamper == "replay_claim":
        sidecar["token_binding"]["exact_source_token_replay"] = True
    elif tamper == "token":
        sidecar["measurement"]["input_ids"][-1] += 1
    elif tamper == "position":
        sidecar["token_binding"]["prefix_positions"]["record"] -= 1
    elif tamper == "extra_event":
        sidecar["executed_simulator_outcome"]["observation_ledger"].append({})
    elif tamper == "generated":
        sidecar["measurement"]["completion"] = {"raw": "A"}
    elif tamper == "row":
        arrays["trial_ids"][0] = "another source"
    elif tamper == "sourcehash":
        sidecar["source_record_sha256"] = "c" * 64
    else:
        sidecar["receipt_model_pass_completed"] = False
    with pytest.raises(ValueError):
        audit.receipt_record(sidecar, record, tokenizer, width, arrays, 0)


@pytest.mark.parametrize("tamper", ["vocabulary", "array", "cohort", "missing", "nan", "flag"])
def test_full_vocabulary_and_array_evidence_recomputed_not_just_pass_flag(tamper):
    _, record, _, tokenizer = source()
    sidecar, arrays, width, engineering = successful_receipt(record, tokenizer)
    if tamper == "vocabulary":
        engineering["captured_full_initial_logits_sha256"][0] = "b" * 64
    elif tamper == "array":
        arrays["hidden"][0, 0, 0, 0] = 2
    elif tamper == "cohort":
        engineering["active_batch_size"] += 1
    elif tamper == "missing":
        arrays["available"][0, 0, 0] = False
    elif tamper == "nan":
        arrays["hidden"][0, 0, 0, 0] = np.nan
        engineering["captured_array_sha256"] = capture.array_sha(arrays["hidden"])
    else:
        engineering["checks"]["hook_cleanup"] = False
    with pytest.raises(ValueError):
        audit.receipt_engineering(
            engineering,
            [record["trial_id"]],
            [sidecar["token_binding"]["token_prefix_sha256"]],
            arrays,
            width,
        )


def receipt_collection(tmp_path, one_active=False):
    assignments = voluntary.assignments()
    choices = [a for a in assignments if a["kind"] == "choice"]
    manifest = dict(diagnostics=assignments, model=capture.MODEL)
    records = {}
    for index, a in enumerate(choices):
        _, record, _, tokenizer = source(
            assignment=a, service="report" if one_active and index == 0 else "stop"
        )
        records[a["trial_id"]] = json.loads(json.dumps(record))
    for offset in range(0, 144, 32):
        ids = [a["trial_id"] for a in choices[offset : offset + 32]]
        rows = [capture.choice_sidecar(records[tid]) for tid in ids]
        active_ids, engineering = [], None
        if one_active and offset == 0:
            completed, arrays, width, engineering = successful_receipt(records[ids[0]], tokenizer)
            path = tmp_path / completed["activation_artifact"]["path"]
            path.parent.mkdir(parents=True)
            np.savez_compressed(path, **arrays)
            completed["activation_artifact"]["sha256"] = audit.byte_sha(path)
            active_ids = [ids[0]]
            completed["cohort"] = dict(
                id="test-0000000",
                choice_offset=0,
                source_trial_ids=ids,
                active_source_trial_ids=active_ids,
                active_row=0,
                active_size=1,
                prefill_width=width,
            )
            rows[0] = completed
            write_once(
                tmp_path / "receipt_environments/test.json", dict(**capture.MODEL, receipt_hidden_size=3)
            )
        envelope = dict(
            schema_version=1,
            manifest_sha256=digest(manifest),
            choice_offset=offset,
            source_trial_ids=ids,
            active_source_trial_ids=active_ids,
            records=rows,
            records_sha256=digest(rows),
            engineering=engineering,
            rendering_policy=capture.RECEIPT_PLAN["prefix_policy"],
        )
        write_once(tmp_path / f"receipt_cohorts/test/{offset:07d}.json", envelope)
    return manifest, records, tokenizer


@pytest.mark.parametrize("one_active", [False, True])
def test_all_144_logical_receipt_slots_and_actual_outcome_dependent_active_rows(tmp_path, one_active):
    manifest, records, tokenizer = receipt_collection(tmp_path, one_active)
    rows, bindings = audit.audit_receipts(tmp_path, manifest, records, tokenizer)
    assert len(rows) == 144 and len(bindings["cohorts"]) == 5
    assert bindings["capture_files"] == int(one_active)
    assert len(bindings["record_bindings"]) == int(one_active)
    assert sum(row["receipt_model_pass_completed"] for row in rows.values()) == int(one_active)


@pytest.mark.parametrize("tamper", ["duplicate", "reorder", "active", "source", "capture_bytes"])
def test_physical_receipt_coverage_and_rehashed_envelopes_reject_tampering(tmp_path, tamper):
    manifest, records, tokenizer = receipt_collection(tmp_path, True)
    path = tmp_path / "receipt_cohorts/test/0000000.json"
    envelope = read(path)
    if tamper == "duplicate":
        write_once(tmp_path / "receipt_cohorts/retry/0000000.json", envelope)
    elif tamper == "reorder":
        envelope["records"][1:3] = list(reversed(envelope["records"][1:3]))
    elif tamper == "active":
        envelope["active_source_trial_ids"].append(envelope["source_trial_ids"][1])
    elif tamper == "source":
        records[envelope["source_trial_ids"][0]]["inference"]["raw"] += " changed"
    else:
        (tmp_path / envelope["records"][0]["activation_artifact"]["path"]).write_bytes(
            b"not the captured array"
        )
    if tamper in ("reorder", "active"):
        envelope["records_sha256"] = digest(envelope["records"])
        path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError):
        audit.audit_receipts(tmp_path, manifest, records, tokenizer)


def test_missing_last16_receipt_slots_remain_missing_in_readiness(tmp_path):
    manifest, records, tokenizer = receipt_collection(tmp_path)
    (tmp_path / "receipt_cohorts/test/0000128.json").unlink()
    rows, _ = audit.audit_receipts(tmp_path, manifest, records, tokenizer)
    result = audit.readiness(manifest["diagnostics"], records, rows)
    assert len(rows) == 128 and not result["passed"]
    assert result["assigned_choices"] == 144 and not result["checks"]["receipt_slot_coverage"]
