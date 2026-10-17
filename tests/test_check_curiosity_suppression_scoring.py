"""Independent-checker tests using synthetic assignments/responses, never study outputs."""

from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import shutil

import pytest

from curiosity_study import information_state
from curiosity_study import voluntary_resolution as prompt_materials
from test_curiosity_suppression_capture import frozen as _frozen

SPEC = importlib.util.spec_from_file_location("independent_c_scoring", Path(__file__).parents[1] / "scripts/check_curiosity_suppression_scoring.py")
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(check.canonical(value))


class Tokenizer:
    pad_token_id = 0

    def __init__(self, base):
        self.base = base

    def __call__(self, text, **kwargs):
        return self.base(text, **kwargs)

    def decode(self, ids, skip_special_tokens=False):
        return self.base.decode([i for i in ids if i != 9999] if skip_special_tokens else ids)

    def encode(self, text):
        return self.base.encode(text)


@pytest.fixture(scope="module")
def materials():
    # Reuses only synthetic prompt construction and the deterministic world
    # generator. No production scoring or main-audit functions are called.
    source = _frozen.__wrapped__()
    plan, preflight, model, _ = next(source)
    tokenizer = Tokenizer(model.tokenizer)
    baseline = sorted((a for a in plan["assignments"] if a["arm"] == "baseline"), key=lambda a: a["base_row"])
    oracles, inners = {}, {}
    for a, frozen in zip(baseline, preflight["requests"], strict=True):
        source_assignment = a["source_assignment"]
        o = check.oracle(a, frozen, tokenizer, information_state.task)
        # Expected labels come from the existing synthetic prompt construction,
        # independently of the new parser; the new checker derives them from text.
        expected = prompt_materials.prompt(source_assignment)[2]
        assert o["expected"] == expected
        mapping = dict(zip("ABCD", check.SERVICES[a["world_slot"] % 4:] + check.SERVICES[:a["world_slot"] % 4], strict=True))
        service = "stop" if a["delivery"] == "decisive_evidence" else "report"
        answer = next(k for k, v in mapping.items() if v == service) if a["kind"] == "choice" else expected
        tokens = tokenizer.encode(f"Reasoning </think> {answer}.") + [9999]
        raw = tokenizer.decode(tokens, skip_special_tokens=True)
        inf = dict(raw=raw, generated_token_ids=tokens, ended_with_eos=True,
                   sampling=dict(seed=a["sampling_seed"], temperature=0.7, top_p=1.0, max_new_tokens=4096))
        world = deepcopy(information_state.task(source_assignment))
        observed = [source_assignment["known"][0]]
        if a["delivery"] == "decisive_evidence":
            observed.append(source_assignment["query"])
        truth = world["candidates"][world["truth"]]
        observations = {str(i): truth[i] for i in observed}
        compatible = [i for i, values in enumerate(world["candidates"]) if all(values[int(j)] == v for j, v in observations.items())]
        task = {**world, "observations": observations, "compatible": compatible}
        inner = dict(assignment=source_assignment, inference=inf, answer=f"{answer}.", parsed_answer=answer,
                     thinking_closed=True, valid=True, correct=None if expected is None else True,
                     cap_hit=False, expected=expected, selected_service=service if a["kind"] == "choice" else None,
                     generation_max_new_tokens=4096, action_mapping=mapping, task=task,
                     simulator_outcome=None if a["kind"] != "choice" else dict(selected_service=service,
                        receipt_required=service != "stop", receipt_model_pass_completed=False))
        oracles[a["base_trial_id"]] = o
        inners[a["base_trial_id"]] = inner
    yield plan, preflight, tokenizer, oracles, inners
    try:
        next(source)
    except StopIteration:
        pass


@pytest.fixture(scope="module")
def collected(tmp_path_factory, materials):
    root = tmp_path_factory.mktemp("independent-c-synthetic")
    plan, preflight, tokenizer, _, inners = materials
    parent = root / "parent"
    m = dict(assay="curiosity_suppression_screen", assignments=[], diagnostics=plan["assignments"],
             assignment_sha256=check.digest(plan["assignments"]), model={"synthetic": True},
             generation_preflight=preflight, generation_preflight_sha256=check.digest(preflight))
    write(parent / "manifest.json", m)
    descriptors = []
    for group in check.GROUPS:
        path = root / group
        rows = [a for a in plan["assignments"] if a["execution_group"] == group]
        child = dict(m, diagnostics=rows, assignment_sha256=check.digest(rows), parent_manifest_sha256=check.digest(m))
        child_sha = check.digest(child)
        write(path / "manifest.json", child)
        descriptors.append(dict(path=str(path), manifest_sha256=child_sha, execution_group=group))
        for offset in range(0, 1440, 32):
            cohort = rows[offset:offset + 32]
            evidence = dict(manifest_sha256=child_sha, generation_offset=offset,
                trial_ids=[a["trial_id"] for a in cohort], eos_token_ids=[9999], pad_token_id=0,
                requests=[preflight["requests"][a["base_row"]]["request"] for a in cohort],
                condition=dict(rows=[dict(completion=inners[a["base_trial_id"]]["inference"]) for a in cohort]))
            relative = Path("generation_evidence/fixture") / f"{offset:07d}.json"
            write(path / relative, evidence)
            binding = dict(path=str(relative), sha256=check.sha(path / relative), byte_count=(path / relative).stat().st_size)
            records = [dict(trial_id=a["trial_id"], assignment=a, terminal=True, cohort_row=i,
                status="completed", generation_evidence=binding, task_record=inners[a["base_trial_id"]])
                for i, a in enumerate(cohort)]
            write(path / "cohorts/fixture" / f"{offset:07d}.json",
                  dict(manifest_hash=child_sha, records=records, sha256=check.digest(records)))
    write(parent / "children.json", descriptors)
    return parent, tokenizer


def run(parent, tokenizer):
    return check.check_collection(parent, tokenizer=tokenizer, world_generator=information_state.task,
                                  eos_ids={9999}, vocabulary_size=10000)


def clone(collected, tmp_path):
    parent, tokenizer = collected
    shutil.copytree(parent.parent, tmp_path / "copy")
    copied = tmp_path / "copy/parent"
    descriptors = check.read(copied / "children.json")
    for d in descriptors:
        d["path"] = str(copied.parent / d["execution_group"])
    write(copied / "children.json", descriptors)
    return copied, tokenizer


def test_complete15arm_scoring_is_parent_ordered_and_has_all270_cells(collected):
    parent, tokenizer = collected
    result = run(parent, tokenizer)
    assert result["passed"] and result["complete"] and result["response_scoring_complete"]
    assert result["computation_agrees"] is True and result["mismatches"] == []
    assert result["checked"] == result["observed_terminal_records"] == result["assigned"] == 4320
    assert len(result["diagnostic_cells"]) == 180 and len(result["choice_cells"]) == 90
    assert result["diagnostic_readiness_passed"] and result["choice_readiness_passed"]
    assert [r["trial_id"] for r in result["rows"]] == [a["trial_id"] for a in check.read(parent / "manifest.json")["diagnostics"]]
    assert result["record_bindings_sha256"] == check.digest([dict(trial_id=r["trial_id"], record_sha256=r["record_sha256"]) for r in result["rows"]])
    assert result["classifications"] == {"valid": 4320}


@pytest.mark.parametrize("failure", ["missing", "unavailable", "empty"])
def test_partial_or_unavailable_rows_cannot_be_certified_by_observed_agreement(collected, tmp_path, failure):
    parent, tokenizer = clone(collected, tmp_path)
    child = parent.parent / "worker_2"
    path = child / "cohorts/fixture/0001408.json"
    if failure == "missing":
        path.unlink()
    elif failure == "empty":
        for group in check.GROUPS:
            shutil.rmtree(parent.parent / group / "cohorts")
    else:
        envelope = check.read(path)
        evidence_path = child / envelope["records"][0]["generation_evidence"]["path"]
        evidence = check.read(evidence_path)
        evidence["condition"]["rows"] = None
        write(evidence_path, evidence)
        for record in envelope["records"]:
            record.update(status="execution_error", task_record=None, failure="generation_execution_error")
            record["generation_evidence"].update(sha256=check.sha(evidence_path), byte_count=evidence_path.stat().st_size)
        envelope["sha256"] = check.digest(envelope["records"])
        write(path, envelope)
    result = run(parent, tokenizer)
    assert len(result["rows"]) == result["assigned"] == 4320 and not result["passed"]
    assert not result["response_scoring_complete"]
    assert result["computation_agrees"] is (None if failure == "empty" else True)
    assert result["checked"] == (0 if failure == "empty" else 4288)
    assert result["complete"] is (failure == "unavailable")
    unknown = [r for r in result["rows"] if not r["scoring_checked"]]
    assert all(r["valid"] is None and r["correct"] is None and r["cap_hit"] is None for r in unknown)
    assert not result["diagnostic_readiness_passed"] and not result["choice_readiness_passed"]


@pytest.mark.parametrize("text,valid,answer", [
    ("Reasoning </think> A", True, "A"), ("Reasoning </think> A.", True, "A."),
    ("Reasoning </think> B </think> A", True, "A"),
    ("Reasoning </think> The answer is \\boxed{A}", False, "The answer is \\boxed{A}"),
    ("Reasoning </think> A..", False, "A.."), ("A", False, ""),
])
def test_exact_closure_and_grammar_never_extracts_letter_from_explanation(materials, text, valid, answer):
    tokenizer = materials[2]
    value = check.parse_completion(dict(generated_token_ids=tokenizer.encode(text) + [9999]), tokenizer, {9999}, 10000)
    assert value["valid"] is valid and value["answer"] == answer


def test_valid_final_at_cap_remains_capped(materials):
    tokenizer = materials[2]
    tokens = tokenizer.encode("Reasoning") * (4096 - 2) + tokenizer.encode("</think> A")
    value = check.parse_completion(dict(generated_token_ids=tokens), tokenizer, {9999}, 10000)
    assert value["valid"] and value["cap_hit"] and not value["ended_with_eos"]


@pytest.mark.parametrize("tokens", [[], [1], [9999, 1], [True, 9999], [10000, 9999]])
def test_malformed_or_incomplete_token_sequences_rejected(materials, tokens):
    with pytest.raises(ValueError):
        check.parse_completion(dict(generated_token_ids=tokens), materials[2], {9999}, 10000)


def test_token_decode_overrides_stored_raw_and_expected_fields(materials):
    plan, _, tokenizer, oracles, inners = materials
    a = next(a for a in plan["assignments"] if a["kind"] == "new_fact")
    inner = deepcopy(inners[a["base_trial_id"]])
    correct = inner["expected"]
    wrong = next(label for label in "ABCD" if label != correct)
    inner["inference"]["generated_token_ids"] = tokenizer.encode(f"Reasoning </think> {wrong}.") + [9999]
    record = dict(status="completed", task_record=inner, generation_evidence={"sha256": "a" * 64})
    evidence = dict(condition=dict(rows=[dict(completion=inner["inference"])] * 32))
    row, mismatches = check.score_record(a, record, evidence, 0, oracles[a["base_trial_id"]], tokenizer, {9999}, 10000)
    assert row["valid"] and row["correct"] is False and row["classification"] == "semantic_error"
    assert {r["field"] for r in mismatches} >= {"inference.raw_sha256", "answer", "parsed_answer", "correct"}
    assert all("Reasoning" not in str(r) for r in mismatches)


def test_choice_missing_or_execution_error_fails_even_at23_valid(materials):
    rows = []
    for a in materials[0]["assignments"]:
        row = check.base_row(a, None if a["kind"] == "choice" else "A")
        row.update(status="completed", valid=True, correct=None if a["kind"] == "choice" else True,
                   cap_hit=False, thinking_closed=True, scoring_checked=True, selected_service="report")
        rows.append(row)
    for status in ("missing", "execution_error"):
        altered = deepcopy(rows)
        row = next(r for r in altered if r["kind"] == "choice")
        row.update(status=status, valid=None, scoring_checked=False, execution_failure=status == "execution_error")
        _, choices = check.summaries(altered)
        failed = [c for c in choices if not c["readiness_passed"]]
        assert len(failed) == 1 and failed[0]["valid"] == 23
        assert failed[0]["missing"] + failed[0]["execution_failures"] == 1


def test_receipt_binds_standalone_source_and_is_immutable(tmp_path):
    target = tmp_path / "receipt.json"
    result = {"passed": False, "computation_agrees": None}
    command = ["python", "checker.py", "--out", str(target)]
    check.save_receipt(target, result, command)
    saved = check.read(target)
    assert saved["verification_program_sha256"] == hashlib.sha256(Path(check.__file__).read_bytes()).hexdigest()
    assert saved["verification_command"] == command
    with pytest.raises(ValueError, match="immutable"):
        check.save_receipt(target, result, command)
    assert not list(tmp_path.glob("*.tmp"))


def test_import_and_helpers_do_not_open_real_parent_or_run_analysis():
    # Importing the standalone module only defines functions and constants.
    import ast
    tree = ast.parse(Path(check.__file__).read_text())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.extend([node.module or "", *[alias.name for alias in node.names]])
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    assert not any("audit_" in name or name in ("voluntary_resolution", "thinking_information", "identification") for name in imported)
    assert "information_state" in imported


def test_complete_consistently_scored_malformed_final_agrees_but_fails_readiness(collected, tmp_path):
    parent, tokenizer = clone(collected, tmp_path)
    child = parent.parent / "worker_0"
    selected = None
    for path in sorted((child / "cohorts/fixture").glob("*.json")):
        envelope = check.read(path)
        for index, record in enumerate(envelope["records"]):
            if record["assignment"]["kind"] == "new_fact":
                selected = path, envelope, index
                break
        if selected:
            break
    path, envelope, index = selected
    inner = envelope["records"][index]["task_record"]
    answer = f"The answer is \\boxed{{{inner['expected']}}}"
    tokens = tokenizer.encode(f"Reasoning </think> {answer}") + [9999]
    inner["inference"].update(raw=tokenizer.decode(tokens, skip_special_tokens=True), generated_token_ids=tokens)
    inner.update(answer=answer, parsed_answer=answer, valid=False, correct=False)
    evidence_path = child / envelope["records"][index]["generation_evidence"]["path"]
    evidence = check.read(evidence_path)
    evidence["condition"]["rows"][index]["completion"] = inner["inference"]
    write(evidence_path, evidence)
    for record in envelope["records"]:
        record["generation_evidence"].update(sha256=check.sha(evidence_path), byte_count=evidence_path.stat().st_size)
    envelope["sha256"] = check.digest(envelope["records"])
    write(path, envelope)
    result = run(parent, tokenizer)
    assert result["complete"] and result["response_scoring_complete"] and result["computation_agrees"]
    assert result["passed"] and not result["diagnostic_readiness_passed"]
    assert result["classifications"] == {"valid": 4319, "malformed_final": 1}
    assert result["exception_rows"][0]["parsed_answer"] is None


@pytest.mark.parametrize("changed", [None, "source", "archive", "parent"])
def test_pinned_source_and_archive_verification(tmp_path, monkeypatch, changed):
    parent = tmp_path / "parent"
    execution = tmp_path / "execution"
    source_path = execution / "src/curiosity_study/information_state.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("# Synthetic frozen source only\n")
    source_sha = check.digest({"information_state.py": source_path.read_text()})
    manifest = dict(source_sha256=source_sha)
    write(parent / "manifest.json", manifest)
    archive = parent / "source.tar.gz"
    archive.write_bytes(b"Synthetic archive binding fixture")
    freeze = dict(execution_root=str(execution), manifest_sha256=check.digest(manifest),
                  source_sha256=source_sha, archive_sha256=check.sha(archive))
    write(parent / "freeze.json", freeze)
    monkeypatch.setattr(check, "PARENT_SHA256", check.digest(manifest))
    monkeypatch.setattr(check, "SOURCE_SHA256", source_sha)
    if changed == "source":
        source_path.write_text(source_path.read_text() + "# changed\n")
    elif changed == "archive":
        archive.write_bytes(archive.read_bytes() + b"changed")
    elif changed == "parent":
        write(parent / "manifest.json", dict(manifest, changed=True))
    if changed is None:
        actual, binding = check.verify_source(parent)
        assert actual == execution and binding["source_sha256"] == source_sha
    else:
        with pytest.raises(ValueError):
            check.verify_source(parent)
