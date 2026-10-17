from collections import Counter, defaultdict
import importlib.util
import itertools
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from curiosity_study import target_relevant_information as target
from curiosity_study.artifacts import digest
from curiosity_study.thinking_information import manifest_generation_limit


def independent_entropy(counts):
    total = sum(counts)
    return sum(-n / total * math.log2(n / total) for n in counts if n)


def test_oracle_independently_enumerates_world_target_and_report_information():
    for assignment in target.assignments(items=16, kinds=("new",)):
        t = target.task(assignment)
        candidates = t["candidates"]
        query, target_index = assignment["query"], t["target_attribute"]
        assert len(candidates) == len({tuple(c) for c in candidates}) == 4
        assert len(set(t["attributes"])) == 3 and query != target_index
        assert t["observations"] == {} and t["compatible"] == list(range(4))
        joint = Counter((c[target_index], c[query]) for c in candidates)
        report = Counter(c[query] for c in candidates)
        hidden_target = Counter(c[target_index] for c in candidates)
        h_target = independent_entropy(hidden_target.values())
        h_report = independent_entropy(report.values())
        h_joint = independent_entropy(joint.values())
        target_gain = h_target + h_report - h_joint
        after_target = sum(
            n
            / 4
            * independent_entropy(Counter(c[target_index] for c in candidates if c[query] == value).values())
            for value, n in report.items()
        )
        after_world = sum(n / 4 * math.log2(n) for n in report.values())
        assert t["world_entropy_before_bits"] == t["uncertainty_bits"] == 2
        assert t["remaining_candidates_before"] == 4
        assert t["expected_remaining_candidates_after"] == sum(n * n / 4 for n in report.values()) == 2
        assert t["expected_world_entropy_after_bits"] == after_world == 1
        assert t["information_bits"] == t["expected_world_information_gain_bits"] == 2 - after_world == 1
        assert t["offered_report_entropy_bits"] == h_report == 1
        assert t["target_entropy_before_bits"] == h_target == 1
        assert t["report_accuracy"] == 1
        assert (
            t["expected_target_information_gain_bits"]
            == target_gain
            == int(query != assignment["independent_report"])
        )
        assert t["expected_target_entropy_after_bits"] == after_target == 1 - target_gain
        for row in candidates:
            encoded = [t["values"][i].index(value) for i, value in enumerate(row)]
            assert encoded[2] == encoded[1 - assignment["independent_report"]] ^ t["target_complement_bit"]


def test_four_cells_share_latent_world_and_keep_the_existing_gain_orientation():
    grouped = defaultdict(list)
    for assignment in target.assignments(items=8, kinds=("new",)):
        grouped[(assignment["family"], assignment["seed"], assignment["index"])].append(assignment)
    for rows in grouped.values():
        rows.sort(key=lambda a: (a["independent_report"], a["query"]))
        worlds = [target.task(a) for a in rows]
        assert [2 * t["expected_target_information_gain_bits"] - 1 for t in worlds] == [-1, 1, 1, -1]
        assert len({tuple(tuple(row[:2]) for row in t["candidates"]) for t in worlds}) == 1
        assert len({tuple(t["attributes"]) for t in worlds}) == 1
        assert len({t["truth"] for t in worlds}) == 1
        assert worlds[0]["candidates"] == worlds[1]["candidates"]
        assert worlds[2]["candidates"] == worlds[3]["candidates"]
        assert worlds[0]["candidates"] != worlds[2]["candidates"]


def test_factual_prefix_is_identical_across_five_kinds_with_query_before_marker():
    for assignment in target.assignments(items=4, kinds=("new",)):
        contexts = []
        for kind in target.KINDS:
            current = {**assignment, "kind": kind}
            text, _, _ = target.prompt(current)
            assert text.count("End of record.") == 1
            prefix, suffix = text.split("End of record.")
            contexts.append(prefix)
            assert "Reply with" in suffix
            assert prefix.index("designated target property") < prefix.index("An available report")
            assert target.task(current) == target.task(assignment)
        assert len(set(contexts)) == 1
        other = {**assignment, "query": 1 - assignment["query"]}
        assert target.context(assignment) != target.context(other)


def test_polarities_instrument_and_all_action_mappings_follow_target_gain():
    mappings = set()
    for assignment in target.assignments(items=6, kinds=("new",)):
        t = target.task(assignment)
        gain = t["expected_target_information_gain_bits"]
        expected = target.prompt(assignment)[2]
        assert expected == ("Yes" if gain else "No")
        assert target.prompt({**assignment, "kind": "repeat"})[2] != expected
        action = target.prompt({**assignment, "kind": "instrument"})[2]
        assert t["action_label_mapping"][action] == ("report" if gain else "stop")
        mappings.add(tuple(t["action_label_mapping"].values()))
    assert mappings == set(itertools.permutations(("report", "reread", "stop")))


def test_each_twelve_world_block_crosses_target_complement_with_every_action_mapping():
    required = set(itertools.product((0, 1), itertools.permutations(("report", "reread", "stop"))))
    for start in (0, 12, 24):
        crossed = defaultdict(Counter)
        for assignment in target.assignments(items=12, start_index=start, kinds=("new",)):
            t = target.task(assignment)
            crossed[assignment["family"]][
                (t["target_complement_bit"], tuple(t["action_label_mapping"].values()))
            ] += 1
        assert set(crossed) == {"objects", "machines"}
        assert all(set(counts) == required and set(counts.values()) == {4} for counts in crossed.values())
    for assignment in target.assignments(items=1):
        t = target.task(assignment)
        assert t["target_complement_bit"] == 0  # The existing prospective smoke remains identical.
        assert t["action_label_mapping"] == {"A": "report", "B": "reread", "C": "stop"}


def test_memory_is_table_lookup_and_count_is_explicitly_constant():
    answers = Counter()
    for assignment in target.assignments(items=12, kinds=("memory",)):
        t = target.task(assignment)
        text, labels, expected = target.prompt(assignment)
        value = t["candidates"][t["memory_candidate"]][t["memory_attribute"]]
        assert f"{expected}: {value}." in text and labels == "AB"
        assert f"candidate {'WXYZ'[t['memory_candidate']]}" in text
        assert t["observations"] == {}
        answers[(assignment["family"], expected)] += 1
        assert target.prompt({**assignment, "kind": "count"})[2] == "4"
        other = {
            **assignment,
            "independent_report": 1 - assignment["independent_report"],
            "query": 1 - assignment["query"],
        }
        assert target.prompt(other)[2] == expected
    assert len(set(answers.values())) == 1


def test_rendering_and_exact_response_grammar_preserve_oracle_and_scoring():
    for assignment in target.assignments(items=2):
        other = {**assignment, "template": "prose"}
        assert target.task(assignment) == target.task(other)
        assert target.prompt(assignment)[1:] == target.prompt(other)[1:]
        assert target.context(assignment) != target.context(other)
        expected = target.prompt(assignment)[2]
        for raw in (expected, f" {expected}.\n"):
            assert target.record_from_inference(assignment, dict(raw=raw))["correct"]
        for raw in ("", expected + " because", expected + "..", "</think> " + expected):
            assert not target.record_from_inference(assignment, dict(raw=raw))["valid"]


def test_whole_world_units_ids_vocabulary_and_complements_are_deterministic():
    rows = target.assignments(items=12)
    assert rows == target.assignments(items=12)
    assert len(rows) == len({a["trial_id"] for a in rows}) == 480
    seen = []
    for offset in range(0, len(rows), 20):
        group = rows[offset : offset + 20]
        world = {(a["seed"], a["family"], a["index"]) for a in group}
        assert len(world) == 1
        seen.extend(world)
        assert {(a["independent_report"], a["query"], a["kind"]) for a in group} == set(
            itertools.product((0, 1), (0, 1), target.KINDS)
        )
        assert all(a["known"] == [] and a["vocabulary"] == "heldout" for a in group)
    assert len(seen) == len(set(seen)) == 24
    assert Counter(target.task(a)["target_complement_bit"] for a in rows) == {0: 240, 1: 240}
    shifted = target.assignments(items=1, start_index=100)
    assert {a["index"] for a in shifted} == {100}
    assert not {a["trial_id"] for a in shifted} & {a["trial_id"] for a in rows}


def preparation_module():
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_target_relevant_smoke.py"
    spec = importlib.util.spec_from_file_location("prepare_target_relevant_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prospective_smoke_is_only_40_branches_and_cannot_pass_validation(monkeypatch):
    prepare = preparation_module()
    monkeypatch.setattr(prepare, "write_once", lambda *args: pytest.fail("Pure builder must not write"))
    monkeypatch.setattr(prepare, "freeze", lambda *args: pytest.fail("Pure builder must not freeze"))
    manifest = prepare.build_manifest(dict(model_id="Qwen/Qwen3-32B", revision="synthetic"))
    assert len(manifest["diagnostics"]) == 40
    assert manifest_generation_limit(manifest) == 4096
    assert manifest["assignment_sha256"] == digest(manifest["diagnostics"])
    assert manifest["phase"] == "development_smoke" and manifest["inference_scope"] == "development"
    assert manifest["response_oracle_field"] == "expected_target_information_gain_bits"
    perfect = {a["trial_id"]: dict(valid=True, correct=True) for a in manifest["diagnostics"]}
    result = target.evaluate(manifest, perfect)
    assert not result["passed"] and len(result["cells"]) == 40
    assert all(
        c["n"] == c["items"] == c["completed"] == c["correct"] == c["valid"] == 1 for c in result["cells"]
    )
    assert "reports/target-relevant-information-design.md" in prepare.DOCUMENTS
    assert "reports/reasoning-information-development-scope-amendment.md" in prepare.DOCUMENTS


def test_preparer_rejects_missing_dispatch_before_writing(monkeypatch):
    prepare = preparation_module()
    monkeypatch.setattr(prepare, "read", lambda _: {"Qwen/Qwen3-32B": "synthetic"})
    monkeypatch.setattr(prepare, "task_module", lambda _: None)
    monkeypatch.setattr(prepare, "write_once", lambda *args: pytest.fail("Must not write without dispatcher"))
    monkeypatch.setattr(prepare, "freeze", lambda *args: pytest.fail("Must not freeze without dispatcher"))
    with pytest.raises(ValueError, match="dispatcher and audit"):
        prepare.main()


def test_preparer_rejects_missing_audit_support_before_writing(monkeypatch):
    prepare = preparation_module()
    monkeypatch.setattr(prepare, "read", lambda _: {"Qwen/Qwen3-32B": "synthetic"})
    monkeypatch.setattr(prepare, "task_module", lambda _: target)
    monkeypatch.setitem(
        sys.modules, "audit_thinking_information", SimpleNamespace(TARGET_RELEVANCE_AUDIT_VERSION=0)
    )
    monkeypatch.setattr(prepare, "write_once", lambda *args: pytest.fail("Must not write without audit"))
    monkeypatch.setattr(prepare, "freeze", lambda *args: pytest.fail("Must not freeze without audit"))
    with pytest.raises(ValueError, match="independently checked oracle audit"):
        prepare.main()


def test_full_smoke_execution_audit_reports_every_cell_and_both_information_quantities(tmp_path, monkeypatch):
    import copy
    import torch
    from test_reasoning_trace import Tokenizer, backend
    from test_thinking_generation_regime import script_module
    from curiosity_study import thinking_information as thinking

    class TaskTokenizer(Tokenizer):
        extra = {"A": 110, "B": 111, "C": 112, "1": 113, "2": 114, "3": 115, "4": 116}

        def encode(self, text, **kwargs):
            return [self.extra[text]] if text in self.extra else super().encode(text, **kwargs)

        def decode(self, ids, **kwargs):
            reverse = {value: key for key, value in self.extra.items()}
            return "".join(
                reverse[token] if token in reverse else super(TaskTokenizer, self).decode([token], **kwargs)
                for token in ids
            )

    model = backend()
    model.tokenizer = TaskTokenizer()
    model.metadata = lambda: dict(model="scripted-target-smoke-cpu")

    def encode(messages, enable_thinking=False):
        text = "\n".join(m["content"] for m in messages)
        return text, SimpleNamespace(input_ids=torch.tensor([model.tokenizer(text)["input_ids"]]))

    model.encode = encode
    manifest = preparation_module().build_manifest(dict(model_id="Qwen/Qwen3-32B", revision="synthetic"))
    manifest["execution"]["batch_size"] = 2
    manifest["capture_plan"] = dict(layers=[0, 1], steps=[32, 128, 512])
    assignments = manifest["diagnostics"]
    by_id = {a["trial_id"]: a for a in assignments}
    capped_id = assignments[0]["trial_id"]
    original_infer = model.infer_batch

    def infer(requests, ids, **kwargs):
        model.model.step = 0
        model.model.sequences = [
            [5] * 4096
            if trial_id == capped_id
            else [5, 102, 10, model.tokenizer.encode(target.prompt(by_id[trial_id])[2])[0], 127]
            for trial_id in ids
        ]
        return original_infer(requests, ids, **kwargs)

    model.infer_batch = infer
    monkeypatch.setattr(thinking, "verify_capture", lambda *args: dict(passed=True))
    thinking.execute(tmp_path, manifest, model)
    audit = script_module("audit_thinking_information")
    monkeypatch.setattr(audit.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: model.tokenizer)
    report = audit.audit(tmp_path)
    assert report["target_relevance_oracle_audit_version"] == 1
    assert report["assigned"] == 40 and report["valid"] == report["correct"] == 39
    assert report["cap_hits"] == 1 and report["feasibility_passed"]
    assert report["target_relevance_smoke_cell_coverage"]
    assert "gates" not in report
    cells = report["target_relevance_cell_scores"]
    assert len(cells) == 40 and all(c["assigned"] == 1 for c in cells)
    assert sum(c["valid"] for c in cells) == sum(c["correct"] for c in cells) == 39
    assert sum(c["cap_hits"] for c in cells) == 1
    coverage = {(c["family"], c["kind"], c["independent_report"], c["query"]): c for c in cells}
    assert audit.target_smoke_coverage(coverage)
    missing = copy.deepcopy(coverage)
    missing.pop(next(iter(missing)))
    next(iter(missing.values()))["assigned"] += 1
    assert not audit.target_smoke_coverage(missing)  # Forty rows cannot substitute for all forty cells.
    availability = report["capture_availability"]
    assert {c["information_bits"] for c in availability} == {1}
    assert {c["expected_target_information_gain_bits"] for c in availability} == {0, 1}
    assert sum(c["assigned"] for c in availability if c["position"] == "record") == 40
    assert sum(c["available"] for c in availability if c["position"] == "decision") == 39
    assert thinking.task_module(assignments[0]) is target
    assert thinking.inference_request(assignments[0])["max_new_tokens"] == 4096

    # The audit checks target gain independently of the task generator's formula.
    for field in (
        "expected_target_information_gain_bits",
        "expected_world_information_gain_bits",
        "offered_report_entropy_bits",
    ):
        corrupted = copy.deepcopy(target.task(assignments[0]))
        corrupted[field] += 1
        with pytest.raises(AssertionError):
            audit.audit_target_oracle(assignments[0], corrupted, list(range(4)))
