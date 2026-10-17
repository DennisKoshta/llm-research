"""Independent response scoring of the completed frozen screen; no inference."""

import hashlib
import json
import argparse
from pathlib import Path
import re
import sys
from collections import Counter
from datetime import datetime, timezone

ROOT = Path("/home/dennis/projects/llm-curiosity-and-satisfaction")
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--out", type=Path, default=ROOT / "reports/satisfaction-screen-v1-independent-scoring.json"
)
args = parser.parse_args()
PARENT = ROOT / "runs/qwen32-satisfaction-screen-v1"
freeze = json.loads((PARENT / "freeze.json").read_text())
FROZEN = Path(freeze["execution_root"])
sys.path.insert(0, str(FROZEN / "src"))
from curiosity_study.artifacts import digest, read, write_once  # noqa: E402
from curiosity_study import information_state  # noqa: E402
from transformers import AutoTokenizer, GenerationConfig  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


manifest = read(PARENT / "manifest.json")
assert digest(manifest) == freeze["manifest_sha256"]
assert sha(PARENT / "source.tar.gz") == freeze["archive_sha256"]
source_files = {
    str(p.relative_to(FROZEN / "src/curiosity_study")): p.read_text()
    for p in sorted((FROZEN / "src/curiosity_study").rglob("*.py"))
}
assert digest(source_files) == manifest["source_sha256"] == freeze["source_sha256"]
tokenizer = AutoTokenizer.from_pretrained(
    manifest["model"]["model_id"],
    revision=manifest["model"]["revision"],
    trust_remote_code=False,
    local_files_only=True,
)
vocabulary_size = len(tokenizer)
config = GenerationConfig.from_pretrained(
    manifest["model"]["model_id"], revision=manifest["model"]["revision"], local_files_only=True
)
eos_ids = {config.eos_token_id} if isinstance(config.eos_token_id, int) else set(config.eos_token_id)
preflight = manifest["generation_preflight"]
assert digest(preflight) == manifest["generation_preflight_sha256"]
assert digest(preflight["requests"]) == preflight["requests_sha256"]
requests = {row["base_trial_id"]: row for row in preflight["requests"]}
assert len(requests) == 288
SERVICES = ("report", "repeat", "pattern", "stop")
ARMS = ("baseline", "zero_sham", "satisfaction_plus_0_5", "satisfaction_plus_1_0")
KINDS = (
    "identify_resolved",
    "new_fact",
    "repeated_fact",
    "pattern_independence",
    "instrument_useful_action",
    "action_consequence",
)
DELIVERIES = ("decisive_evidence", "redundant_evidence", "independent_novelty")


def oracle(outer):
    a = outer["source_assignment"]
    frozen = requests[a["trial_id"]]
    assert digest(a) == outer["source_assignment_sha256"] == frozen["source_assignment_sha256"]
    text = frozen["rendered_prompt"]
    assert digest(text) == frozen["rendered_prompt_sha256"]
    assert tokenizer(text, add_special_tokens=False)["input_ids"] == frozen["request"]["token_prefix"]
    factual = text.split("End of record.", 1)[0]
    attributes = [x.strip() for x in re.search(r"^Candidate \| (.+) \| (.+)$", factual, re.M).groups()]
    candidates = {
        label: [v0.strip(), v1.strip()]
        for label, v0, v1 in re.findall(r"^([WXYZ]) \| ([^|\n]+) \| ([^|\n]+)$", factual, re.M)
    }
    assert list(candidates) == list("WXYZ")
    world = json.loads(json.dumps(information_state.task(a)))
    assert attributes == world["attributes"] and list(candidates.values()) == world["candidates"]
    observations = {}
    for name, value in re.findall(r"^The (.+?) is (.+?)\.$", factual, re.M):
        assert name in attributes
        observations[name] = value
    expected_names = {attributes[a["known"][0]]}
    if a["delivery"] == "decisive_evidence":
        expected_names.add(attributes[a["query"]])
    assert set(observations) == expected_names
    truth = world["candidates"][world["truth"]]
    assert all(value == truth[attributes.index(name)] for name, value in observations.items())
    compatible = [
        label
        for label, values in candidates.items()
        if all(values[attributes.index(name)] == value for name, value in observations.items())
    ]
    assert len(compatible) == (1 if a["delivery"] == "decisive_evidence" else 2)
    canonical_order = sorted(range(2), key=attributes.__getitem__)
    semantic = dict(
        schema="voluntary_finite_world_v1",
        family=a["family"],
        attributes=[attributes[i] for i in canonical_order],
        candidates=sorted([row[i] for i in canonical_order] for row in candidates.values()),
        truth=[truth[i] for i in canonical_order],
    )
    assert digest(semantic) == a["semantic_world_id"] == outer["semantic_world_id"]
    option_rows = re.findall(r"^([ABCD]): ([^\n]+)$", text, re.M)
    options = dict(option_rows[-4:])
    assert len(options) == 4
    rotation = a["world_slot"] % 4
    mapping = dict(zip("ABCD", SERVICES[rotation:] + SERVICES[:rotation]))
    report = f"Receive the hidden candidate's {attributes[a['query']]} report, then end."
    repeat = f"Receive the initial {attributes[a['known'][0]]} observation again, then end."
    descriptions = {
        "report": report,
        "repeat": repeat,
        "pattern": "View a new independent decorative pattern, then end.",
        "stop": "End without receiving another observation or pattern.",
    }
    kind = a["kind"]
    if kind in ("choice", "instrument_useful_action"):
        assert options == {label: descriptions[service] for label, service in mapping.items()}
        target_text = report
    elif kind == "identify_resolved":
        assert len(compatible) == 1
        target_text = f"Candidate {compatible[0]}."
    elif kind in ("new_fact", "repeated_fact"):
        already_known = attributes[a["query"]] in observations
        target_text = (
            "A property fact about the hidden candidate that has already been supplied in the complete record above."
            if already_known
            else "A property fact about the hidden candidate that is not already supplied in the complete record above."
        )
    elif kind == "pattern_independence":
        target_text = "None; its generation is independent of the hidden candidate."
    elif kind == "action_consequence":
        menu = dict(option_rows[:4])
        assert menu == {label: descriptions[service] for label, service in mapping.items()}
        selected = re.search(r"What happens if action ([ABCD]) is selected\?", text).group(1)
        service = mapping[selected]
        assert service == SERVICES[a["world_slot"] // 3]
        target_text = {
            "report": f"Only the {attributes[a['query']]} report is supplied, and the episode ends.",
            "repeat": "Only the initial observation is supplied again, and the episode ends.",
            "pattern": "Only a new independent decorative pattern is supplied, and the episode ends.",
            "stop": "Nothing further is supplied, and the episode ends.",
        }[service]
    else:
        raise ValueError(kind)
    expected = (
        None if kind == "choice" else next(label for label, value in options.items() if value == target_text)
    )
    return dict(
        expected=expected,
        mapping=mapping,
        world=world,
        observations={str(attributes.index(name)): value for name, value in observations.items()},
        compatible=["WXYZ".index(label) for label in compatible],
        request=frozen["request"],
    )


oracle_by_id = {}
for a in manifest["diagnostics"]:
    if a["base_trial_id"] not in oracle_by_id:
        oracle_by_id[a["base_trial_id"]] = oracle(a)
assert len(oracle_by_id) == 288
print("Independently checked all288 prompt/table/oracle bindings.", flush=True)

rows, mismatches, cohort_bindings, evidence_bindings, child_bindings = [], [], [], [], []
seen = set()
parent_assignments = {a["trial_id"]: a for a in manifest["diagnostics"]}
assert len(parent_assignments) == 1152


def mismatch(tid, field, stored, derived):
    if stored != derived:
        mismatches.append(dict(trial_id=tid, field=field, stored=stored, independently_derived=derived))


for child in read(PARENT / "children.json"):
    root = Path(child["path"])
    cm = read(root / "manifest.json")
    assert digest(cm) == child["manifest_sha256"]
    assert cm["parent_manifest_sha256"] == digest(manifest)
    assignments = {a["trial_id"]: a for a in cm["diagnostics"]}
    assert all(parent_assignments[k] == v for k, v in assignments.items())
    child_bindings.append(dict(path=str(root), manifest_sha256=digest(cm), assigned=len(assignments)))
    observed = set()
    for path in sorted((root / "cohorts").glob("*/*.json")):
        envelope = read(path)
        assert envelope["manifest_hash"] == digest(cm)
        assert envelope["sha256"] == digest(envelope["records"])
        records = envelope["records"]
        assert len(records) == 32
        bound = records[0]["generation_evidence"]
        ep = root / bound["path"]
        assert sha(ep) == bound["sha256"] and ep.stat().st_size == bound["byte_count"]
        evidence = read(ep)
        assert evidence["manifest_sha256"] == digest(cm)
        assert set(evidence["eos_token_ids"]) == eos_ids
        assert evidence["pad_token_id"] == tokenizer.pad_token_id
        assert evidence["trial_ids"] == [r["trial_id"] for r in records]
        cohort_bindings.append(dict(path=str(path), byte_sha256=sha(path), content_sha256=digest(envelope)))
        evidence_bindings.append(
            dict(path=str(ep), byte_sha256=bound["sha256"], byte_count=bound["byte_count"])
        )
        for index, record in enumerate(records):
            tid = record["trial_id"]
            a = assignments[tid]
            source = a["source_assignment"]
            o = oracle_by_id[a["base_trial_id"]]
            assert tid not in seen and record["assignment"] == a and record["terminal"] is True
            assert record["cohort_row"] == index and record["generation_evidence"] == bound
            assert evidence["requests"][index] == o["request"]
            seen.add(tid)
            observed.add(tid)
            inner = record["task_record"]
            measured = evidence["condition"]["rows"]
            row = dict(
                trial_id=tid,
                base_trial_id=a["base_trial_id"],
                arm=a["arm"],
                panel_id=a["panel_id"],
                semantic_world_id=a["semantic_world_id"],
                world_slot=a["world_slot"],
                family=a["family"],
                kind=a["kind"],
                delivery=a["delivery"],
                menu_layout=a["menu_layout"],
                record_path=str(path),
                cohort_row=index,
                record_sha256=digest(record),
                generation_evidence_sha256=bound["sha256"],
                status=record["status"],
                expected=o["expected"],
                valid=False,
                correct=None if a["kind"] == "choice" else False,
                cap_hit=False,
                thinking_closed=False,
                parsed_answer=None,
                selected_service=None,
                generated_token_count=None,
                execution_failure=inner is None,
            )
            if inner is None:
                assert measured is None
                row["classification"] = "execution_failure"
                rows.append(row)
                continue
            assert measured is not None
            inf = inner["inference"]
            assert inf == measured[index]["completion"]
            assert inner["assignment"] == source
            tokens = inf["generated_token_ids"]
            raw = tokenizer.decode(tokens, skip_special_tokens=True)
            mismatch(tid, "inference.raw", inf["raw"], raw)
            assert len(tokens) <= 4096 and all(type(t) is int and 0 <= t < vocabulary_size for t in tokens)
            eos = bool(tokens and tokens[-1] in eos_ids)
            closed = "</think>" in raw
            answer = raw.rpartition("</think>")[2].strip() if closed else ""
            parsed = answer[:-1] if answer.endswith(".") else answer
            valid = closed and re.fullmatch(r"[ABCD]\.?", answer) is not None
            correct = None if o["expected"] is None else bool(valid and parsed == o["expected"])
            cap = not eos and len(tokens) == 4096
            selected = o["mapping"].get(parsed) if valid and a["kind"] == "choice" else None
            for field, derived in dict(
                answer=answer,
                parsed_answer=parsed,
                thinking_closed=closed,
                valid=valid,
                correct=correct,
                cap_hit=cap,
                expected=o["expected"],
                selected_service=selected,
                generation_max_new_tokens=4096,
                action_mapping=o["mapping"],
            ).items():
                mismatch(tid, field, inner.get(field), derived)
            mismatch(tid, "ended_with_eos", inf["ended_with_eos"], eos)
            for field in ("attributes", "candidates", "truth"):
                mismatch(tid, "task." + field, inner["task"][field], o["world"][field])
            mismatch(tid, "task.observations", inner["task"]["observations"], o["observations"])
            mismatch(tid, "task.compatible", inner["task"]["compatible"], o["compatible"])
            if a["kind"] == "choice":
                simulator = inner["simulator_outcome"]
                if valid:
                    mismatch(tid, "simulator.selected_service", simulator["selected_service"], selected)
                    mismatch(
                        tid, "simulator.receipt_required", simulator["receipt_required"], selected != "stop"
                    )
                    mismatch(
                        tid,
                        "simulator.receipt_model_pass_completed",
                        simulator["receipt_model_pass_completed"],
                        False,
                    )
                else:
                    mismatch(tid, "simulator_outcome", simulator, None)
            row.update(
                valid=bool(valid),
                correct=correct,
                cap_hit=cap,
                thinking_closed=closed,
                parsed_answer=parsed if valid else None,
                selected_service=selected,
                generated_token_count=len(tokens),
                ended_with_eos=eos,
                final_answer_text=answer,
                raw_response_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                generated_token_ids_sha256=digest(tokens),
                classification=(
                    "cap_hit"
                    if cap
                    else "unclosed_thinking"
                    if not closed
                    else "malformed_final"
                    if not valid
                    else "semantic_error"
                    if correct is False
                    else "valid"
                ),
            )
            rows.append(row)
        del evidence
    assert observed == set(assignments)
    print(f"Verified{len(observed)} rows for{child['execution_group']}.", flush=True)

assert seen == set(parent_assignments) and len(rows) == 1152
diagnostic_cells = []
choice_cells = []
for arm in ARMS:
    for panel in ("panel_1", "panel_2"):
        for kind in KINDS:
            cell = [r for r in rows if r["arm"] == arm and r["panel_id"] == panel and r["kind"] == kind]
            assert len(cell) == 12
            summary = dict(
                arm=arm,
                panel_id=panel,
                family="objects",
                kind=kind,
                assigned=len(cell),
                valid=sum(r["valid"] for r in cell),
                correct=sum(r["correct"] is True for r in cell),
                capped=sum(r["cap_hit"] for r in cell),
                unclosed=sum(not r["thinking_closed"] for r in cell),
                execution_failures=sum(r["execution_failure"] for r in cell),
                invalid_trial_ids=[r["trial_id"] for r in cell if not r["valid"]],
                incorrect_trial_ids=[r["trial_id"] for r in cell if r["correct"] is not True],
            )
            summary["readiness_passed"] = (
                summary["valid"] == 12 and summary["correct"] >= 11 and summary["capped"] == 0
            )
            diagnostic_cells.append(summary)
        for delivery in DELIVERIES:
            cell = [
                r
                for r in rows
                if r["arm"] == arm
                and r["panel_id"] == panel
                and r["kind"] == "choice"
                and r["delivery"] == delivery
            ]
            assert len(cell) == 24
            summary = dict(
                arm=arm,
                panel_id=panel,
                family="objects",
                delivery=delivery,
                assigned=len(cell),
                valid=sum(r["valid"] for r in cell),
                capped=sum(r["cap_hit"] for r in cell),
                execution_failures=sum(r["execution_failure"] for r in cell),
                valid_service_counts=dict(Counter(r["selected_service"] for r in cell if r["valid"])),
                valid_closed_uncapped_service_counts=dict(
                    Counter(r["selected_service"] for r in cell if r["valid"] and not r["cap_hit"])
                ),
                invalid_trial_ids=[r["trial_id"] for r in cell if not r["valid"]],
            )
            summary["readiness_passed"] = summary["valid"] >= 23 and summary["capped"] == 0
            choice_cells.append(summary)

exception_rows = [r for r in rows if r["classification"] != "valid"]
result = dict(
    schema_version=1,
    scope="Independent complete-response scoring only; not an engineering/receipt/causal certificate.",
    created_utc=datetime.now(timezone.utc).isoformat(),
    parent_run=str(PARENT),
    parent_manifest_sha256=digest(manifest),
    source_sha256=manifest["source_sha256"],
    source_archive_sha256=freeze["archive_sha256"],
    child_bindings=child_bindings,
    model=manifest["model"],
    assigned=1152,
    checked=1152,
    choices_assigned=576,
    diagnostics_assigned=576,
    record_bindings_sha256=digest(
        [dict(trial_id=r["trial_id"], record_sha256=r["record_sha256"]) for r in rows]
    ),
    cohort_bindings=cohort_bindings,
    generation_evidence_bindings=evidence_bindings,
    all_scoring_agrees=not mismatches,
    mismatches=mismatches,
    complete=True,
    passed=not mismatches,
    classifications=dict(Counter(r["classification"] for r in rows)),
    diagnostic_cells=diagnostic_cells,
    choice_cells=choice_cells,
    diagnostic_readiness_passed=all(c["readiness_passed"] for c in diagnostic_cells),
    choice_readiness_passed=all(c["readiness_passed"] for c in choice_cells),
    exception_rows=exception_rows,
    rows=rows,
    method=[
        "Verified parent/child/frozen-source/cohort/evidence byte and content bindings and exact1152 assignment coverage.",
        "Read independently reconstructed oracle tables from pinned frozen information_state.task; checked all288 rendered prompts against those tables and exact tokenizer IDs.",
        "Derived expected labels from parsed rendered candidate/observation/menu/answer-option text, not stored expected/correct fields or the main audit compact rows.",
        "Decoded every saved generated-token sequence with the pinned local tokenizer; independently applied last closing-think delimiter, exact single A/B/C/D with optional one period, EOS and4096 cap rules.",
        "Retained all assigned rows; capped responses remain separately counted even if syntactically valid; no inferred answer for malformed/unclosed/failed rows.",
    ],
    limitations=[
        "The deterministic finite-world generator and pinned tokenizer are shared source dependencies, although parsing/scoring and final-label oracle derivation are independent of the producer/main auditor.",
        "No neural inference was rerun, and full-vocabulary fingerprints, intervention geometry, receipt execution, runtime equivalence and scientific advancement statistics are outside this scoring check.",
        "This receipt uses the frozen v2 wording and scoring rules as written; it does not adjudicate hidden experience, revise ambiguous answers, or rescue failed readiness cells.",
        "passed means stored scoring agrees with independently derived scoring and complete assigned coverage; it does not mean the scientific readiness gates passed.",
    ],
)
program = Path(__file__).read_text()
result["verification_program"] = program
result["verification_program_sha256"] = hashlib.sha256(program.encode()).hexdigest()
result["verification_program_path"] = str(Path(__file__).resolve())
result["verification_command"] = [
    str(ROOT / ".venv/bin/python"),
    str(Path(__file__).resolve()),
    "--out",
    str(args.out.resolve()),
]
out = args.out.resolve()
write_once(out, result)
print(
    json.dumps(
        {
            "receipt": str(out),
            "receipt_byte_sha256": sha(out),
            "checked": len(rows),
            "scoring_mismatches": len(mismatches),
            "classifications": result["classifications"],
            "diagnostic_readiness_passed": result["diagnostic_readiness_passed"],
            "choice_readiness_passed": result["choice_readiness_passed"],
            "failed_diagnostic_cells": [c for c in diagnostic_cells if not c["readiness_passed"]],
            "exception_rows": exception_rows,
        },
        indent=2,
    ),
    flush=True,
)
