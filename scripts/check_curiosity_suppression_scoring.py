"""Independent raw-token and finite-table scoring; no inference or gate relaxation.

Only the pinned tokenizer and frozen deterministic world generator are shared.
This module does not import the producer's task scorer, parser, or main auditor.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
PARENT_NAME = "qwen32-curiosity-suppression-screen-v1-admission2"
PARENT_SHA256 = "535f3c1237a8b876fa1059d3f9949f178a55b5627e2be412008ef08cbc90b02f"
SOURCE_SHA256 = "1fd4079ed60fbd0169f702ab9c8f99a75d170450dfbd364bc4cb877e27c18f9a"
ARMS = (
    "baseline", "zero_sham", "curiosity_negative", "confidence_positive", "completion_positive",
    "valence_positive", "valence_negative", "motivation_negative", "novelty_negative",
    "task_demand_negative", "random_01", "random_02", "random_03", "random_04", "random_05",
)
GROUPS = ("worker_0", "worker_1", "worker_2")
KINDS = (
    "identify_resolved", "new_fact", "repeated_fact", "pattern_independence",
    "instrument_useful_action", "action_consequence",
)
DELIVERIES = ("decisive_evidence", "redundant_evidence", "independent_novelty")
SERVICES = ("report", "repeat", "pattern", "stop")
CAP = 4096
ASSIGNED = 4320


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound_artifact(root, binding):
    path = (root / binding["path"]).resolve()
    require(path.is_relative_to(root.resolve()), "Evidence path escaped its collection")
    require(sha(path) == binding["sha256"] and path.stat().st_size == binding["byte_count"],
            "Evidence bytes changed")
    return path


def oracle(outer, frozen, tokenizer, world_generator):
    """Derive answer options from the displayed table/facts, without stored labels."""
    a = outer["source_assignment"]
    require(digest(a) == outer["source_assignment_sha256"] == frozen["source_assignment_sha256"],
            "Source assignment binding changed")
    require(a["trial_id"] == outer["base_trial_id"] == frozen["base_trial_id"], "Source ID changed")
    text = frozen["rendered_prompt"]
    request = frozen["request"]
    require(digest(text) == frozen["rendered_prompt_sha256"]
            and tokenizer(text, add_special_tokens=False)["input_ids"] == request["token_prefix"]
            and digest(request["token_prefix"]) == request["token_prefix_sha256"]
            and request["source_trial_id"] == a["trial_id"]
            and request["seed"] == outer["sampling_seed"], "Exact prompt or seed changed")
    require(text.count("End of record.") == 1, "Ambiguous record boundary")
    factual = text.split("End of record.", 1)[0]
    heading = re.search(r"^Candidate \| (.+) \| (.+)$", factual, re.M)
    require(heading is not None, "Missing finite candidate table")
    attributes = [part.strip() for part in heading.groups()]
    displayed = re.findall(r"^([WXYZ]) \| ([^|\n]+) \| ([^|\n]+)$", factual, re.M)
    require([row[0] for row in displayed] == list("WXYZ"), "Candidate rows missing or repeated")
    candidates = {label: [v0.strip(), v1.strip()] for label, v0, v1 in displayed}
    world = json.loads(json.dumps(world_generator(a)))
    require(attributes == world["attributes"] and list(candidates.values()) == world["candidates"],
            "Displayed table differs from the frozen finite-world generator")
    observations = {}
    for name, value in re.findall(r"^The (.+?) is (.+?)\.$", factual, re.M):
        require(name in attributes and (name not in observations or observations[name] == value),
                "Unknown or conflicting observation")
        observations[name] = value
    expected_names = {attributes[a["known"][0]]}
    if a["delivery"] == "decisive_evidence":
        expected_names.add(attributes[a["query"]])
    require(set(observations) == expected_names, "Delivered facts differ from assignment")
    truth = world["candidates"][world["truth"]]
    require(all(value == truth[attributes.index(name)] for name, value in observations.items()),
            "Observation contradicts realized truth")
    compatible = [label for label, values in candidates.items()
                  if all(values[attributes.index(name)] == value for name, value in observations.items())]
    require(len(compatible) == (1 if a["delivery"] == "decisive_evidence" else 2),
            "Wrong number of compatible candidates")
    order = sorted(range(2), key=attributes.__getitem__)
    semantic = dict(schema="voluntary_finite_world_v1", family=a["family"],
                    attributes=[attributes[i] for i in order],
                    candidates=sorted([row[i] for i in order] for row in candidates.values()),
                    truth=[truth[i] for i in order])
    require(digest(semantic) == a["semantic_world_id"] == outer["semantic_world_id"],
            "Canonical named world changed")
    option_rows = re.findall(r"^([ABCD]): ([^\n]+)$", text, re.M)
    options = dict(option_rows[-4:])
    require(len(options) == 4, "Missing response options")
    rotation = a["world_slot"] % 4
    mapping = dict(zip("ABCD", SERVICES[rotation:] + SERVICES[:rotation], strict=True))
    report = f"Receive the hidden candidate's {attributes[a['query']]} report, then end."
    descriptions = dict(report=report,
        repeat=f"Receive the initial {attributes[a['known'][0]]} observation again, then end.",
        pattern="View a new independent decorative pattern, then end.",
        stop="End without receiving another observation or pattern.")
    kind = a["kind"]
    if kind in ("choice", "instrument_useful_action"):
        require(options == {label: descriptions[service] for label, service in mapping.items()},
                "Action menu differs from its assigned mapping")
        target = report
    elif kind == "identify_resolved":
        require(len(compatible) == 1, "Identification diagnostic has unresolved facts")
        target = f"Candidate {compatible[0]}."
    elif kind in ("new_fact", "repeated_fact"):
        target = (
            "A property fact about the hidden candidate that has already been supplied in the complete record above."
            if attributes[a["query"]] in observations else
            "A property fact about the hidden candidate that is not already supplied in the complete record above."
        )
    elif kind == "pattern_independence":
        target = "None; its generation is independent of the hidden candidate."
    elif kind == "action_consequence":
        require(dict(option_rows[:4]) == {label: descriptions[service] for label, service in mapping.items()},
                "Consequence diagnostic action menu changed")
        question = re.search(r"What happens if action ([ABCD]) is selected\?", text)
        require(question is not None, "Missing consequence action")
        service = mapping[question.group(1)]
        require(service == SERVICES[a["world_slot"] // 3], "Consequence action mapping changed")
        target = dict(report=f"Only the {attributes[a['query']]} report is supplied, and the episode ends.",
                      repeat="Only the initial observation is supplied again, and the episode ends.",
                      pattern="Only a new independent decorative pattern is supplied, and the episode ends.",
                      stop="Nothing further is supplied, and the episode ends.")[service]
    else:
        raise ValueError(f"Unknown task kind: {kind}")
    matches = [label for label, value in options.items() if value == target]
    require(len(matches) == 1, "Correct semantic option is missing or duplicated")
    return dict(expected=None if kind == "choice" else matches[0], mapping=mapping, world=world,
                observations={str(attributes.index(name)): value for name, value in observations.items()},
                compatible=["WXYZ".index(label) for label in compatible], request=request)


def parse_completion(completion, tokenizer, eos_ids, vocabulary_size):
    """Exact frozen final grammar; no answer extraction from malformed explanations."""
    tokens = completion["generated_token_ids"]
    require(isinstance(tokens, list) and 1 <= len(tokens) <= CAP
            and all(type(token) is int and 0 <= token < vocabulary_size for token in tokens),
            "Invalid generated token sequence")
    require(not any(token in eos_ids for token in tokens[:-1]), "Tokens continue after EOS")
    eos = tokens[-1] in eos_ids
    require(eos or len(tokens) == CAP, "Incomplete response is not EOS-terminated or capped")
    raw = tokenizer.decode(tokens, skip_special_tokens=True)
    closed = "</think>" in raw
    answer = raw.rpartition("</think>")[2].strip() if closed else ""
    parsed = answer[:-1] if answer.endswith(".") else answer
    valid = closed and re.fullmatch(r"[ABCD]\.?", answer) is not None
    return dict(raw=raw, answer=answer, parsed_answer=parsed, thinking_closed=closed,
                valid=bool(valid), ended_with_eos=eos, cap_hit=not eos and len(tokens) == CAP)


def base_row(assignment, expected):
    row = {key: assignment[key] for key in (
        "trial_id", "base_trial_id", "arm", "panel_id", "semantic_world_id", "world_slot",
        "family", "kind", "delivery", "menu_layout")}
    row.update(expected=expected, status="missing", classification="missing", record_path=None,
               cohort_row=None, record_sha256=None, generation_evidence_sha256=None,
               valid=None, correct=None, cap_hit=None, thinking_closed=None, parsed_answer=None,
               selected_service=None, generated_token_count=None, execution_failure=False,
               scoring_checked=False)
    return row


def score_record(assignment, record, evidence, index, oracle_value, tokenizer, eos_ids, vocabulary_size):
    """Return independently derived fields and mismatches for one observed row."""
    a, o, tid = assignment, oracle_value, assignment["trial_id"]
    row = base_row(a, o["expected"])
    row.update(status=record["status"], record_sha256=digest(record), cohort_row=index,
               generation_evidence_sha256=record["generation_evidence"]["sha256"])
    inner = record["task_record"]
    measured = evidence["condition"]["rows"]
    if inner is None:
        require(measured is None and record["status"] == "execution_error"
                and record.get("failure") == "generation_execution_error",
                "Unavailable response was hidden or reclassified")
        row.update(classification="execution_failure", execution_failure=True)
        return row, []
    require(record["status"] == "completed" and measured is not None and len(measured) == 32,
            "Available response disposition differs")
    inf = inner["inference"]
    require(inf == measured[index]["completion"] and inner["assignment"] == a["source_assignment"],
            "Stored inference differs from observed completion/source")
    parsed = parse_completion(inf, tokenizer, eos_ids, vocabulary_size)
    correct = None if o["expected"] is None else bool(parsed["valid"] and parsed["parsed_answer"] == o["expected"])
    selected = o["mapping"].get(parsed["parsed_answer"]) if parsed["valid"] and a["kind"] == "choice" else None
    mismatches = []

    def compare(field, stored, derived):
        if stored != derived:
            mismatches.append(dict(trial_id=tid, field=field, stored=stored, independently_derived=derived))

    compare("inference.raw_sha256", hashlib.sha256(inf["raw"].encode()).hexdigest(),
            hashlib.sha256(parsed["raw"].encode()).hexdigest())
    for field in ("answer", "parsed_answer", "thinking_closed", "valid", "cap_hit"):
        compare(field, inner.get(field), parsed[field])
    for field, derived in dict(correct=correct, expected=o["expected"], selected_service=selected,
                               generation_max_new_tokens=CAP, action_mapping=o["mapping"]).items():
        compare(field, inner.get(field), derived)
    compare("ended_with_eos", inf["ended_with_eos"], parsed["ended_with_eos"])
    compare("sampling", inf["sampling"], dict(seed=o["request"]["seed"], temperature=0.7, top_p=1.0, max_new_tokens=CAP))
    for field in ("attributes", "candidates", "truth"):
        compare("task." + field, inner["task"][field], o["world"][field])
    compare("task.observations", inner["task"]["observations"], o["observations"])
    compare("task.compatible", inner["task"]["compatible"], o["compatible"])
    if a["kind"] == "choice":
        simulator = inner["simulator_outcome"]
        if parsed["valid"]:
            require(isinstance(simulator, dict), "Valid choice lacks its selected-service outcome")
            compare("simulator.selected_service", simulator["selected_service"], selected)
            compare("simulator.receipt_required", simulator["receipt_required"], selected != "stop")
            compare("simulator.receipt_model_pass_completed", simulator["receipt_model_pass_completed"], False)
        else:
            compare("simulator_outcome", simulator, None)
    row.update(**{key: parsed[key] for key in ("valid", "thinking_closed", "cap_hit", "ended_with_eos")},
               correct=correct, parsed_answer=parsed["parsed_answer"] if parsed["valid"] else None,
               selected_service=selected, generated_token_count=len(inf["generated_token_ids"]),
               final_answer_text=parsed["answer"], raw_response_sha256=hashlib.sha256(parsed["raw"].encode()).hexdigest(),
               generated_token_ids_sha256=digest(inf["generated_token_ids"]), scoring_checked=True,
               classification="cap_hit" if parsed["cap_hit"] else "unclosed_thinking" if not parsed["thinking_closed"]
               else "malformed_final" if not parsed["valid"] else "semantic_error" if correct is False else "valid")
    return row, mismatches


def summaries(rows):
    diagnostics, choices = [], []
    for arm in ARMS:
        for panel in ("panel_1", "panel_2"):
            for kind, delivery in [(kind, None) for kind in KINDS] + [("choice", d) for d in DELIVERIES]:
                cell = [r for r in rows if r["arm"] == arm and r["panel_id"] == panel
                        and r["kind"] == kind and (delivery is None or r["delivery"] == delivery)]
                expected = 12 if delivery is None else 24
                require(len(cell) == expected, "Assigned readiness cell denominator changed")
                value = dict(arm=arm, panel_id=panel, family="objects", assigned=len(cell),
                    valid=sum(r["valid"] is True for r in cell), correct=sum(r["correct"] is True for r in cell),
                    capped=sum(r["cap_hit"] is True for r in cell),
                    unclosed=sum(r["thinking_closed"] is False for r in cell),
                    missing=sum(r["status"] == "missing" for r in cell),
                    execution_failures=sum(r["execution_failure"] for r in cell),
                    scoring_checked=sum(r["scoring_checked"] for r in cell),
                    invalid_or_unknown_trial_ids=[r["trial_id"] for r in cell if r["valid"] is not True])
                value["readiness_passed"] = (value["valid"] == 12 and value["correct"] >= 11
                    if delivery is None else value["valid"] >= 23) and all(
                        value[key] == 0 for key in ("capped", "missing", "execution_failures"))
                if delivery is None:
                    value.update(kind=kind, incorrect_or_unknown_trial_ids=[r["trial_id"] for r in cell if r["correct"] is not True])
                    diagnostics.append(value)
                else:
                    value.update(delivery=delivery,
                        valid_service_counts=dict(Counter(r["selected_service"] for r in cell if r["valid"] is True)),
                        valid_closed_uncapped_service_counts=dict(Counter(r["selected_service"] for r in cell if r["valid"] is True and r["cap_hit"] is False)))
                    choices.append(value)
    return diagnostics, choices


def check_collection(parent, *, tokenizer, world_generator, eos_ids, vocabulary_size):
    """Read assigned artifacts; main() separately verifies the pinned source freeze."""
    parent = Path(parent).resolve()
    manifest = read(parent / "manifest.json")
    parent_sha = digest(manifest)
    assigned = manifest["diagnostics"]
    require(manifest["assay"] == "curiosity_suppression_screen" and manifest["assignments"] == []
            and len(assigned) == len({a["trial_id"] for a in assigned}) == ASSIGNED
            and manifest["assignment_sha256"] == digest(assigned)
            and {a["arm"] for a in assigned} == set(ARMS), "Wrong parent assignment inventory")
    preflight = manifest["generation_preflight"]
    require(digest(preflight) == manifest["generation_preflight_sha256"]
            and digest(preflight["requests"]) == preflight["requests_sha256"], "Preflight binding changed")
    requests = {r["base_trial_id"]: r for r in preflight["requests"]}
    require(len(requests) == len(preflight["requests"]) == 288, "Missing or duplicate original source prompts")
    oracles = {}
    for a in assigned:
        require(a["family"] == "objects", "Unexpected source family")
        if a["base_trial_id"] not in oracles:
            oracles[a["base_trial_id"]] = oracle(a, requests[a["base_trial_id"]], tokenizer, world_generator)
        require(digest(a["source_assignment"]) == a["source_assignment_sha256"]
                == requests[a["base_trial_id"]]["source_assignment_sha256"]
                and a["sampling_seed"] == oracles[a["base_trial_id"]]["request"]["seed"],
                "A treatment changed its source assignment or paired seed")
    require(set(oracles) == set(requests), "Source prompts differ from assigned source rows")
    descriptors = read(parent / "children.json")
    require([d["execution_group"] for d in descriptors] == list(GROUPS), "Require all three fixed workers")
    seen, checked, mismatches = set(), {}, []
    cohorts, evidences, child_bindings = [], [], []
    for descriptor in descriptors:
        root = Path(descriptor["path"]).resolve()
        require(root.parent == parent.parent and root != parent, "Child path is outside its bound parent")
        child = read(root / "manifest.json")
        child_sha = digest(child)
        expected = [a for a in assigned if a["execution_group"] == descriptor["execution_group"]]
        require(child_sha == descriptor["manifest_sha256"] and child["parent_manifest_sha256"] == parent_sha
                and child["diagnostics"] == expected and len(expected) == 1440
                and child["assignments"] == [] and child["assignment_sha256"] == digest(expected),
                "Child differs from the complete parent partition")
        child_bindings.append(dict(path=str(root), manifest_sha256=child_sha, assigned=len(expected)))
        require(not any((root / "attempts").rglob("*.json")), "Non-atomic generation storage")
        offsets, attempts = set(), set()
        for path in sorted((root / "cohorts").rglob("*.json")):
            require(len(path.relative_to(root / "cohorts").parts) == 2 and path.stem.isdecimal(), "Unexpected cohort path")
            offset = int(path.stem)
            require(offset in range(0, 1440, 32) and offset not in offsets, "Repeated or unassigned cohort")
            offsets.add(offset)
            attempts.add(path.parent.name)
            envelope = read(path)
            records = envelope["records"]
            require(envelope["manifest_hash"] == child_sha and envelope["sha256"] == digest(records)
                    and len(records) == 32 and [r["assignment"] for r in records] == expected[offset:offset + 32],
                    "Atomic cohort assignment or digest changed")
            binding = records[0]["generation_evidence"]
            ep = bound_artifact(root, binding)
            evidence = read(ep)
            require(evidence["manifest_sha256"] == child_sha and evidence["generation_offset"] == offset
                    and evidence["trial_ids"] == [r["trial_id"] for r in records]
                    and set(evidence["eos_token_ids"]) == set(eos_ids)
                    and evidence["pad_token_id"] == tokenizer.pad_token_id,
                    "Completion evidence belongs to different inputs or token policy")
            cohorts.append(dict(path=str(path), byte_sha256=sha(path), content_sha256=digest(envelope)))
            evidences.append(dict(path=str(ep), byte_sha256=binding["sha256"], byte_count=binding["byte_count"]))
            for index, (record, a) in enumerate(zip(records, expected[offset:offset + 32], strict=True)):
                tid = a["trial_id"]
                require(record["trial_id"] == tid and tid not in seen and record["terminal"] is True
                        and record["cohort_row"] == index and record["generation_evidence"] == binding
                        and evidence["requests"][index] == oracles[a["base_trial_id"]]["request"],
                        "Response is duplicated or rebound to another assignment/source")
                row, errors = score_record(a, record, evidence, index, oracles[a["base_trial_id"]], tokenizer, set(eos_ids), vocabulary_size)
                row["record_path"] = str(path)
                seen.add(tid)
                checked[tid] = row
                mismatches.extend(errors)
            del evidence
        require(len(attempts) <= 1, "Worker combines multiple generation attempts")
    rows = [checked.get(a["trial_id"], base_row(a, oracles[a["base_trial_id"]]["expected"])) for a in assigned]
    diagnostics, choices = summaries(rows)
    response_count = sum(r["scoring_checked"] for r in rows)
    agreement = not mismatches if response_count else None
    coverage_complete = len(seen) == ASSIGNED
    response_complete = response_count == ASSIGNED
    return dict(schema_version=1, scope="Independent raw-response scoring, not an engineering, receipt or causal certificate.",
        parent_run=str(parent), parent_manifest_sha256=parent_sha, model=manifest["model"],
        assigned=ASSIGNED, observed_terminal_records=len(seen), checked=response_count,
        choices_assigned=2160, diagnostics_assigned=2160,
        complete=coverage_complete, response_scoring_complete=response_complete,
        computation_agrees=agreement, mismatches=mismatches,
        passed=coverage_complete and response_complete and agreement is True,
        child_bindings=child_bindings, cohort_bindings=cohorts, generation_evidence_bindings=evidences,
        record_bindings_sha256=digest([dict(trial_id=r["trial_id"], record_sha256=r["record_sha256"]) for r in rows]),
        rows=rows, classifications=dict(Counter(r["classification"] for r in rows)),
        diagnostic_cells=diagnostics, choice_cells=choices,
        diagnostic_readiness_passed=all(c["readiness_passed"] for c in diagnostics),
        choice_readiness_passed=all(c["readiness_passed"] for c in choices),
        exception_rows=[r for r in rows if r["classification"] != "valid"])


def verify_source(parent):
    freeze = read(parent / "freeze.json")
    manifest = read(parent / "manifest.json")
    require(digest(manifest) == freeze["manifest_sha256"] == PARENT_SHA256,
            "Require the one frozen C-screen parent, without outcome-based substitution")
    require(sha(parent / "source.tar.gz") == freeze["archive_sha256"], "Source archive bytes changed")
    frozen = Path(freeze["execution_root"]).resolve()
    source = {str(path.relative_to(frozen / "src/curiosity_study")): path.read_text()
              for path in sorted((frozen / "src/curiosity_study").rglob("*.py"))}
    require(digest(source) == manifest["source_sha256"] == freeze["source_sha256"] == SOURCE_SHA256,
            "Frozen execution source changed")
    return frozen, dict(source_sha256=SOURCE_SHA256, source_archive_sha256=freeze["archive_sha256"],
                       freeze_byte_sha256=sha(parent / "freeze.json"), frozen_execution=str(frozen))


def save_receipt(out, result, command):
    """Publish once; the receipt includes this standalone verification program."""
    out = Path(out).resolve()
    require(not out.exists(), "Independent scoring receipts are immutable")
    program = Path(__file__).read_text()
    result.update(created_utc=datetime.now(timezone.utc).isoformat(), verification_program=program,
                  verification_program_sha256=hashlib.sha256(program.encode()).hexdigest(),
                  verification_program_path=str(Path(__file__).resolve()), verification_command=command)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(f".{out.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x") as handle:
            handle.write(canonical(result) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, out)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs" / PARENT_NAME)
    parser.add_argument("--out", type=Path, default=ROOT / "reports/curiosity-suppression-screen-v1-independent-scoring.json")
    args = parser.parse_args()
    parent = args.run.resolve()
    frozen, binding = verify_source(parent)
    sys.path.insert(0, str(frozen / "src"))
    from curiosity_study import information_state
    from transformers import AutoConfig, AutoTokenizer, GenerationConfig

    require(Path(information_state.__file__).resolve().is_relative_to(frozen / "src"),
            "World generator was imported from unfrozen source")
    manifest = read(parent / "manifest.json")
    model = manifest["model"]
    kwargs = dict(revision=model["revision"], local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model["model_id"], trust_remote_code=False, **kwargs)
    eos = GenerationConfig.from_pretrained(model["model_id"], **kwargs).eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [] if eos is None else [eos])
    vocabulary_size = AutoConfig.from_pretrained(model["model_id"], **kwargs).vocab_size
    result = check_collection(parent, tokenizer=tokenizer, world_generator=information_state.task,
                              eos_ids=eos_ids, vocabulary_size=vocabulary_size)
    result.update(**binding,
        method=[
            "Verified the fixed frozen parent/source and exact child/cohort/evidence bindings; retained all 4,320 assignments in parent order.",
            "Reconstructed all 288 finite tables with the frozen world generator, parsed rendered facts and answer options independently, and re-tokenized each exact prompt.",
            "Decoded saved generated IDs; independently applied the last closing-think delimiter, exact A/B/C/D with optional one period, EOS and 4096-token cap rules.",
            "Compared stored fields only after deriving independent labels, grammar and choice mappings; never inferred answers from malformed or unavailable responses.",
            "Scoring agreement concerns checked observed responses only. Complete coverage, complete response scoring and readiness are reported separately.",
        ],
        limitations=[
            "The deterministic finite-world generator and pinned tokenizer are shared dependencies; the producer's scorer/parser and main auditor are not imported.",
            "No neural inference is rerun. Full logits, intervention geometry, receipt delivery, runtime equivalence, budget reconciliation and causal statistics are outside this check.",
            "Missing and execution-failed rows remain assigned with unknown response fields and cannot be certified by agreement on observed rows.",
            "passed certifies complete stored-response scoring agreement, not readiness or a positive scientific effect; no scoring relaxation or failed-gate rescue is permitted.",
        ])
    command = [sys.executable, str(Path(__file__).resolve()), "--run", str(parent), "--out", str(args.out.resolve())]
    save_receipt(args.out, result, command)
    print(json.dumps({key: result[key] for key in ("assigned", "observed_terminal_records", "checked", "complete",
        "response_scoring_complete", "computation_agrees", "passed", "diagnostic_readiness_passed", "choice_readiness_passed")}, indent=2))


if __name__ == "__main__":
    main()
