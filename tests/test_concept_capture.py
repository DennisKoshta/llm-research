"""Synthetic materials only; no candidate corpus, model download, or GPU job."""

import copy
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from curiosity_study import concept_capture as concept
from curiosity_study.artifacts import RunStore, digest, read, write_once
from curiosity_study.state_backend import StateBackend


def script(name):
    directory = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location(name, directory / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def materials():
    corpora = {}
    for name, author in (("A", "root"), ("B", "replay_review")):
        pairs = []
        for scenario in range(12):
            for ci, contrast in enumerate(concept.CONTRASTS):
                facts = (
                    f"At station {name}{scenario} on route {ci}, the observer has a notebook, a small table, "
                    "and a map showing the surrounding buildings."
                )
                positive = (
                    facts
                    + " The observer pauses by the window and considers the available details carefully, "
                    "while keeping the map beside the notebook and leaving all equipment in its previous location."
                )
                negative = (
                    facts + " The observer waits by the doorway and considers the available details briefly, "
                    "while keeping the map beside the notebook and leaving all equipment in its previous location."
                )
                pairs.append(
                    dict(
                        id=f"{name}-{scenario}-{ci}",
                        scenario_id=f"scenario-{scenario:02d}",
                        domain=f"domain-{scenario % 6}",
                        contrast=contrast,
                        shared_facts=facts,
                        positive=positive,
                        negative=negative,
                        intended_difference="Synthetic fixture only.",
                        residual_confounds="These fixtures do not instantiate valid psychological contrasts.",
                    )
                )
        corpora[name] = dict(
            schema_version=1,
            corpus_id=name,
            author_id=author,
            authorship="Synthetic independent-author metadata for unit tests only.",
            pairs=pairs,
        )
    return corpora


class WordTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.words = {"A": 1, "B": 2}

    def __call__(self, text, **kwargs):
        matches = list(re.finditer(r"\S+", text))
        ids = []
        for match in matches:
            word = match.group()
            if word not in self.words:
                self.words[word] = len(self.words) + 1
            ids.append(self.words[word])
        return dict(input_ids=ids, offset_mapping=[(m.start(), m.end()) for m in matches])

    def encode(self, text, **kwargs):
        return self(text)["input_ids"]

    def decode(self, ids, **kwargs):
        reverse = {value: key for key, value in self.words.items()}
        return " ".join(reverse.get(i, f"token-{i}") for i in ids if i)


def request(text, tokenizer):
    return concept.inference_request(dict(text=text, text_sha256=concept.text_sha(text)), tokenizer)


def tiny_model():
    torch.manual_seed(928)
    network = (
        Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=512,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=3,
                num_attention_heads=1,
                num_key_value_heads=1,
                head_dim=8,
                attn_implementation="eager",
            )
        )
        .to(torch.bfloat16)
        .eval()
    )
    return SimpleNamespace(
        torch=torch,
        device="cpu",
        active_hooks=0,
        model=network,
        layers=network.model.layers,
        tokenizer=WordTokenizer(),
    )


def test_real_tiny_bf16_transformer_read_only_capture_full_logits_and_real_padded_positions():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = tiny_model()
        texts = ["A person studies a small map" + " beside the table" * (i % 3) + "." for i in range(32)]
        prepared = [request(text, model.tokenizer)[0] for text in texts]
        audit = concept.verify_capture(model, prepared, [0, 1, 2])
        assert audit["passed"] and audit["batch_size"] == 32
        assert audit["baseline_full_initial_logits_sha256"] == audit["captured_full_initial_logits_sha256"]
        rows, values = StateBackend(model).forward(prepared, layers=[0, 1, 2], max_new_tokens=0)
        assert values.shape == (32, 3, 2, 8)
        for row, original in zip(rows, prepared):
            assert "completion" not in row
            pad = row["prefill_width"] - len(original["token_prefix"])
            assert row["record_token_index"] == pad + original["prefix_positions"]["record"]
            assert row["unpadded_token_positions"]["final"] == len(original["token_prefix"]) - 1
            assert row["record_token_index"] < row["prefill_width"] - 1
        assert model.active_hooks == 0 and not model.model._forward_hooks
        assert all(not layer._forward_hooks for layer in model.layers)
    finally:
        torch.set_num_threads(old_threads)


def test_full_vocabulary_guard_detects_change_outside_engineering_labels_on_last_row(monkeypatch):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = tiny_model()
        prepared = [request(f"The observer describes location {i}.", model.tokenizer)[0] for i in range(32)]
        forward = model.model.forward

        def changed(*args, **kwargs):
            result = forward(*args, **kwargs)
            if model.layers[0]._forward_hooks:
                result.logits[31, -1, 500] -= 100
            return result

        monkeypatch.setattr(model.model, "forward", changed)
        audit = concept.verify_capture(model, prepared, [0, 1, 2])
        assert not audit["passed"] and not audit["checks"]["full_initial_logits_equal"]
        assert audit["checks"]["every_row_fingerprint_equal"]
        assert model.active_hooks == 0 and not model.model._forward_hooks
    finally:
        torch.set_num_threads(old_threads)


def test_capture_and_temporary_logit_hooks_are_removed_when_model_raises(monkeypatch):
    model = tiny_model()

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic forward failure")

    monkeypatch.setattr(model.model, "forward", fail)
    with pytest.raises(RuntimeError, match="synthetic"):
        concept._logit_hashes(model, [request("A short narrative.", model.tokenizer)[0]], [0, 1, 2])
    assert model.active_hooks == 0 and not model.model._forward_hooks
    assert all(not layer._forward_hooks for layer in model.layers)


def test_materials_cover_every_pair_and_plain_prefix_excludes_all_metadata():
    corpora = materials()
    audit = concept.validate_corpora(corpora)
    rows = concept.assignments(corpora)
    assert audit["texts"] == len(rows) == 384
    assert all(len(rows[i : i + 32]) == 32 for i in range(0, 384, 32))
    first, binding = concept.inference_request(rows[0], WordTokenizer())
    assert binding["prompt"] == rows[0]["text"] + concept.SUFFIX
    assert not binding["apply_chat_template"] and not binding["add_special_tokens"]
    assert set(first) == {"token_prefix", "token_prefix_sha256", "prefix_positions", "labels", "seed"}
    assert first["labels"] == ["A", "B"]


def test_assignment_order_counterbalances_polarity_and_preserves_whole_scenarios():
    corpora = materials()
    rows = concept.assignments(corpora)
    reversed_materials = copy.deepcopy(corpora)
    for corpus in reversed_materials.values():
        corpus["pairs"].reverse()
    assert concept.assignments(corpora) == rows == concept.assignments(reversed_materials)
    assert concept.ASSIGNMENT_ORDER["seed"] == 2026101433
    assert len({row["trial_id"] for row in rows}) == 384
    first_members = defaultdict(Counter)
    positive_first_scenarios = defaultdict(set)
    for index in range(0, len(rows), 2):
        pair = rows[index : index + 2]
        assert len({(a["corpus_id"], a["pair_id"], a["scenario_id"], a["contrast"]) for a in pair}) == 1
        assert {a["polarity"] for a in pair} == {"positive", "negative"}
        first = pair[0]
        key = first["corpus_id"], first["contrast"]
        first_members[key][first["polarity"]] += 1
        if first["polarity"] == "positive":
            positive_first_scenarios[key].add(first["scenario_id"])
    assert len(first_members) == 16
    assert all(counts == Counter(positive=6, negative=6) for counts in first_members.values())
    assert any(
        positive_first_scenarios[("A", contrast)] != positive_first_scenarios[("B", contrast)]
        for contrast in concept.CONTRASTS
    )
    pair_orders = []
    for offset in range(0, len(rows), 16):
        scenario = rows[offset : offset + 16]
        assert len({(a["corpus_id"], a["scenario_id"]) for a in scenario}) == 1
        assert Counter(a["contrast"] for a in scenario) == Counter({c: 2 for c in concept.CONTRASTS})
        pair_orders.append(tuple(a["contrast"] for a in scenario[::2]))
    assert len(set(pair_orders)) > 1 and any(order != concept.CONTRASTS for order in pair_orders)
    for offset in range(0, len(rows), 32):
        counts = Counter((a["corpus_id"], a["scenario_id"]) for a in rows[offset : offset + 32])
        assert len(counts) == 2 and set(counts.values()) == {16}
    expected = {
        (name, pair["id"], polarity, pair[polarity])
        for name, corpus in corpora.items()
        for pair in corpus["pairs"]
        for polarity in ("positive", "negative")
    }
    assert {(a["corpus_id"], a["pair_id"], a["polarity"], a["text"]) for a in rows} == expected


@pytest.mark.parametrize(
    "problem", ["banned", "too_short", "prefix", "pair_ratio", "missing_pair", "author", "duplicate"]
)
def test_invalid_materials_are_rejected(problem):
    corpora = materials()
    pair = corpora["A"]["pairs"][0]
    if problem == "banned":
        pair["positive"] += " Curious."
    elif problem == "too_short":
        pair["positive"] = pair["shared_facts"]
    elif problem == "prefix":
        pair["negative"] = "Different " + pair["negative"]
    elif problem == "pair_ratio":
        pair["positive"] += " More" * 18
    elif problem == "missing_pair":
        corpora["A"]["pairs"].pop()
    elif problem == "author":
        corpora["B"]["author_id"] = "root"
    else:
        pair["negative"] = pair["positive"]
    with pytest.raises(ValueError):
        concept.validate_corpora(corpora)


def test_merged_token_cannot_carry_suffix_information_into_record_capture():
    class MergedTokenizer(WordTokenizer):
        def __call__(self, text, **kwargs):
            result = super().__call__(text, **kwargs)
            boundary = text.index(concept.SUFFIX)
            index = next(i for i, (_, end) in enumerate(result["offset_mapping"]) if end == boundary)
            result["offset_mapping"][index] = (result["offset_mapping"][index][0], boundary + 4)
            result["offset_mapping"][index + 1] = (boundary + 4, boundary + 4)
            return result

    with pytest.raises(ValueError, match="nonwhitespace suffix"):
        request("A short narrative.", MergedTokenizer())


def synthetic_manifest(tmp_path):
    preparer = script("prepare_concept_capture")
    corpora = materials()
    paths = {}
    for name, corpus in corpora.items():
        paths[name] = tmp_path / f"corpus_{name}.json"
        write_once(paths[name], corpus)
    hashes = {name: preparer.byte_sha(path) for name, path in paths.items()}
    review = dict(
        schema_version=1,
        passed=True,
        corpus_byte_sha256=hashes,
        reviewer="design_review",
        scope="structural_and_semantic_prefreeze_review",
        limitations=["Synthetic test fixture, not research material."],
    )
    review_path = tmp_path / "review.json"
    write_once(review_path, review)
    plan = tmp_path / "plan.md"
    plan.write_text("Synthetic prospective methods fixture. No fitting or cloud launches.\n")
    bindings = [
        dict(corpus_id=name, path=path.name, byte_sha256=hashes[name], canonical_sha256=digest(corpora[name]))
        for name, path in paths.items()
    ]
    review_binding = dict(
        path=review_path.name, byte_sha256=preparer.byte_sha(review_path), canonical_sha256=digest(review)
    )
    documents = [
        dict(path=path.name, byte_sha256=preparer.byte_sha(path))
        for path in [*paths.values(), review_path, plan]
    ]
    manifest = preparer.build_manifest(corpora, concept.MODEL, bindings, review, review_binding, documents)
    return manifest, preparer


def freeze_fixture(root, manifest):
    # A synthetic archive fixture, not a call to production freeze or job preparation.
    with tarfile.open(root / "source.tar.gz", "w:gz") as archive:
        for binding in manifest["document_bindings"]:
            archive.add(root / binding["path"], arcname=binding["path"])
        archive.add(concept.__file__, arcname="src/curiosity_study/concept_capture.py")
    write_once(
        root / "freeze.json",
        dict(
            manifest_sha256=digest(manifest),
            source_sha256=manifest["source_sha256"],
            archive_sha256=hashlib.sha256((root / "source.tar.gz").read_bytes()).hexdigest(),
            documents={b["path"]: b["byte_sha256"] for b in manifest["document_bindings"]},
        ),
    )


class FakeCaptureBackend:
    def __init__(self, model):
        self.model = model

    def forward(self, requests, layers=(), positions=(), max_new_tokens=None):
        assert max_new_tokens == 0
        rendered, ids, inputs, targets = StateBackend(self.model).prepare(requests)
        width = inputs["input_ids"].shape[1]
        result = []
        hidden = np.zeros((len(requests), len(layers), len(positions), 4), dtype=np.float32)
        for i, row in enumerate(requests):
            hidden[i] = i + 1
            result.append(
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
                    unpadded_token_positions=row["prefix_positions"],
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
        return result, hidden


def synthetic_execution(tmp_path, monkeypatch):
    manifest, _ = synthetic_manifest(tmp_path)
    model = SimpleNamespace(
        torch=torch,
        device="cpu",
        active_hooks=0,
        tokenizer=WordTokenizer(),
        model=SimpleNamespace(config=SimpleNamespace(vocab_size=2048, hidden_size=4)),
        metadata=lambda: {**concept.MODEL, "synthetic": True},
    )

    def startup(model, requests, layers):
        _, _, inputs, _ = StateBackend(model).prepare(requests)
        checks = dict(
            full_initial_logits_equal=True,
            every_row_fingerprint_equal=True,
            all_captures_finite=True,
            capture_shape=True,
            no_intervention_or_generation=True,
            hook_cleanup=True,
        )
        hashes = [digest({"synthetic_row": i}) for i in range(32)]
        return dict(
            passed=True,
            checks=checks,
            layers=list(layers),
            positions=concept.POSITIONS,
            source_prefix_sha256=[r["token_prefix_sha256"] for r in requests],
            batch_size=32,
            prefill_width=inputs["input_ids"].shape[1],
            baseline_full_initial_logits_sha256=hashes,
            captured_full_initial_logits_sha256=hashes.copy(),
        )

    monkeypatch.setattr(concept, "StateBackend", FakeCaptureBackend)
    monkeypatch.setattr(concept, "verify_capture", startup)
    result = concept.execute(tmp_path, manifest, model)
    freeze_fixture(tmp_path, manifest)
    return manifest, model, result


def test_complete_capture_audit_covers384_texts_and_no_semantic_scores(tmp_path, monkeypatch):
    manifest, model, records = synthetic_execution(tmp_path, monkeypatch)
    auditor = script("audit_concept_capture")
    audit = auditor.audit(tmp_path, model.tokenizer)
    assert audit["passed"] and audit["complete"] and audit["assigned"] == 384
    assert audit["atomic_cohorts"] == audit["capture_files"] == 12
    assert audit["capture_plan"] == dict(layers=[15, 31, 47, 57], positions=["record", "final"])
    assert audit["manifest_sha256"] == digest(manifest)
    assert not audit["semantic_metrics_computed"] and not audit["directions_fitted"]
    assert all("correct" not in r and "valid" not in r for r in records.values())
    # Resume skips complete atomic cohorts and does not write duplicate terminal rows.
    concept.execute(tmp_path, manifest, model)
    assert auditor.audit(tmp_path, model.tokenizer)["passed"]


def alter_cohort(root, mutate):
    path = sorted((root / "cohorts").glob("*/*.json"))[0]
    envelope = read(path)
    mutate(envelope["records"])
    envelope["sha256"] = digest(envelope["records"])
    path.write_text(json.dumps(envelope))


@pytest.mark.parametrize(
    "problem", ["wrong_token", "wrong_position", "missing_row", "wrong_capture_hash", "startup_last_row"]
)
def test_audit_rejects_cohort_token_position_capture_or_startup_corruption(tmp_path, monkeypatch, problem):
    _, model, _ = synthetic_execution(tmp_path, monkeypatch)
    if problem == "startup_last_row":
        path = next((tmp_path / "concept_integrity").glob("*.json"))
        value = read(path)
        value["captured_full_initial_logits_sha256"][-1] = "f" * 64
        path.write_text(json.dumps(value))
    else:

        def change(rows):
            if problem == "wrong_token":
                rows[0]["measurement"]["unpadded_input_ids"][-1] = 1
            elif problem == "wrong_position":
                rows[0]["measurement"]["record_token_index"] += 1
            elif problem == "missing_row":
                rows.pop()
            else:
                rows[1]["activation_artifact"]["sha256"] = "f" * 64

        alter_cohort(tmp_path, change)
    with pytest.raises(ValueError):
        script("audit_concept_capture").audit(tmp_path, model.tokenizer)


def test_preparation_requires_exact_review_before_any_output_and_archives_raw_materials(
    tmp_path, monkeypatch
):
    _, preparer = synthetic_manifest(tmp_path)
    config = tmp_path / "model.json"
    write_once(config, concept.MODEL)
    calls = []

    def fake_freeze(root, out, documents):
        calls.append(documents)
        return dict(source_sha256=read(out / "manifest.json")["source_sha256"])

    monkeypatch.setattr(preparer, "freeze", fake_freeze)
    args = (
        tmp_path / "corpus_A.json",
        tmp_path / "corpus_B.json",
        config,
        tmp_path / "review.json",
        [tmp_path / "plan.md"],
    )
    result = preparer.prepare(*args, tmp_path / "prepared", root=tmp_path)
    assert result["assigned"] == 384 and not result["gpu_launched"]
    assert set(calls[0]) == {"corpus_A.json", "corpus_B.json", "model.json", "review.json", "plan.md"}
    with pytest.raises(ValueError, match="immutable"):
        preparer.prepare(*args, tmp_path / "prepared", root=tmp_path)
    review = read(tmp_path / "review.json")
    review["corpus_byte_sha256"]["B"] = "bad"
    (tmp_path / "review.json").write_text(json.dumps(review))
    with pytest.raises(ValueError, match="material-review"):
        preparer.prepare(*args, tmp_path / "rejected", root=tmp_path)
    assert not (tmp_path / "rejected").exists() and len(calls) == 1


def test_partial_cohort_resume_and_execution_failure_remain_operational_failures(tmp_path, monkeypatch):
    manifest, model, _ = synthetic_execution(tmp_path, monkeypatch)
    alter_cohort(tmp_path, lambda rows: rows.pop())
    with pytest.raises(ValueError, match="partially published"):
        concept.execute(tmp_path, manifest, model)
    incomplete = copy.deepcopy(manifest)
    incomplete["diagnostics"] = incomplete["diagnostics"][:-1]
    with pytest.raises(ValueError, match="assignment coverage"):
        concept.validate_manifest(incomplete)

    def failed_forward(*args, **kwargs):
        raise RuntimeError("synthetic capture failure")

    monkeypatch.setattr(FakeCaptureBackend, "forward", failed_forward)
    failed_root = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="synthetic capture failure"):
        concept.execute(failed_root, manifest, model)
    failures = RunStore(failed_root, manifest).completed()
    assert len(failures) == 32 and all(r["failure"] == "execution_error" for r in failures.values())
    assert len(list((failed_root / "cohorts").glob("*/*.json"))) == 1
