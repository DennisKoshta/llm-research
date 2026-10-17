"""Independent finite-table scoring for all assigned coverage-bridge responses.

No producer parser, task scorer, collector, or outcome selection is imported.
Raw token decoding/EOS-ID identity, source archive identity and actual receipt
forwards are external audit obligations. This adapter checks raw-text grammar,
terminal metadata, displayed-world oracles and selected-payload semantics.
"""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import re

PANELS = ("panel_1", "panel_2")
DIAGNOSTICS = ("identify_resolved", "new_fact", "repeated_fact", "pattern_independence",
               "instrument_useful_action", "action_consequence")
DELIVERIES = ("decisive_evidence", "redundant_evidence", "independent_novelty")
SERVICES = ("report", "repeat", "pattern", "stop")
DIRECTIONS = ("zero_sham", "curiosity_negative", "confidence_positive", "completion_positive",
              "valence_positive", "valence_negative", "motivation_negative", "novelty_negative",
              "task_demand_negative", "random_01", "random_02", "random_03", "random_04", "random_05")
ARMS = ("baseline", *(f"{s}__{d}" for d in DIRECTIONS for s in ("record", "full")))
METADATA = ("trial_id", "base_trial_id", "panel_id", "semantic_world_id", "world_slot", "arm",
            "schedule", "direction_key", "family", "kind", "delivery", "menu_layout")
ASSIGNED, CAP = 8352, 4096


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def same(a, b):
    return digest(a) == digest(b)


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def pattern(a, channel):
    rng = random.Random(f"voluntary-pattern:{a['seed']}:{a['family']}:{a['world_slot']}:{channel}")
    return "Independent decorative pattern:\n" + "\n".join(
        " ".join(rng.choice((".", "*")) for _ in range(4)) for _ in range(4))


def event(source, payload, facts):
    return dict(source=source, payload=payload, payload_sha256=digest(payload), facts=facts)


def fact_text(fact):
    return f"The {fact['name']} is {fact['value']}."


def displayed(frozen, a):
    text = frozen["rendered_prompt"]
    require(digest(text) == frozen["rendered_prompt_sha256"] and text.count("End of record.") == 1,
            "Unbound or ambiguous source prompt")
    factual = text.split("End of record.", 1)[0]
    heading = re.findall(r"^Candidate \| (.+) \| (.+)$", factual, re.M)
    rows = re.findall(r"^([WXYZ]) \| ([^|\n]+) \| ([^|\n]+)$", factual, re.M)
    require(len(heading) == 1 and [r[0] for r in rows] == list("WXYZ"), "Incomplete candidate table")
    attributes = list(heading[0])
    candidates = [list(r[1:]) for r in rows]
    require(len(set(attributes)) == 2 and len({tuple(r) for r in candidates}) == 4,
            "Ambiguous attributes or candidates")
    observations = {}
    for name, value in re.findall(r"^The (.+?) is (.+?)\.$", factual, re.M):
        require(name in attributes and (name not in observations or observations[name] == value),
                "Conflicting displayed observation")
        observations[name] = value
    expected_names = {attributes[a["known"][0]]}
    if a["delivery"] == "decisive_evidence":
        expected_names.add(attributes[a["query"]])
    require(set(observations) == expected_names, "Wrong delivered property inventory")
    compatible = [i for i, r in enumerate(candidates)
                  if all(r[attributes.index(k)] == v for k, v in observations.items())]
    require(len(compatible) == (1 if a["delivery"] == "decisive_evidence" else 2),
            "Wrong finite-world support")
    return dict(text=text, attributes=attributes, candidates=candidates, compatible=compatible,
                observations={str(attributes.index(k)): v for k, v in observations.items()})


def oracle(a, shown, truth):
    attributes, candidates = shown["attributes"], shown["candidates"]
    order = sorted(range(2), key=attributes.__getitem__)
    canonical = dict(schema="voluntary_finite_world_v1", family=a["family"],
                     attributes=[attributes[i] for i in order],
                     candidates=sorted([r[i] for i in order] for r in candidates),
                     truth=[candidates[truth][i] for i in order])
    require(digest(canonical) == a["semantic_world_id"], "Displayed canonical world differs")
    facts = [dict(attribute=i, name=name, value=candidates[truth][i]) for i, name in enumerate(attributes)]
    initial, report = facts[a["known"][0]], facts[a["query"]]
    delivered = report if a["delivery"] == "decisive_evidence" else initial
    ledger = [event("initial_observation", fact_text(initial), [initial]),
              event("assigned_delivery", pattern(a, "assigned_delivery"), [])
              if a["delivery"] == "independent_novelty" else
              event("assigned_delivery", fact_text(delivered), [delivered])]
    require(all(candidates[truth][int(k)] == v for k, v in shown["observations"].items()),
            "Displayed fact disagrees with resolved-world truth")
    text = shown["text"]
    require("Initial observation:\n" + ledger[0]["payload"] + "\nAdditional delivery:\n" +
            ledger[1]["payload"] + "\n" in text, "Actual displayed payload differs")
    rotation = a["world_slot"] % 4
    mapping = dict(zip("ABCD", SERVICES[rotation:] + SERVICES[:rotation], strict=True))
    descriptions = dict(report=f"Receive the hidden candidate's {report['name']} report, then end.",
                        repeat=f"Receive the initial {initial['name']} observation again, then end.",
                        pattern="View a new independent decorative pattern, then end.",
                        stop="End without receiving another observation or pattern.")
    option_rows = re.findall(r"^([ABCD]): ([^\n]+)$", text, re.M)
    require(len(option_rows) == (8 if a["kind"] == "action_consequence" else 4), "Wrong option inventory")
    options = dict(option_rows[-4:])
    require(set(options) == set("ABCD"), "Duplicate/missing answer label")
    kind = a["kind"]
    if kind in ("choice", "instrument_useful_action"):
        require(options == {label: descriptions[service] for label, service in mapping.items()},
                "Action descriptions or mapping changed")
        require([r[0] for r in option_rows] == list("ABCD" if a["menu_layout"] == 0 else "DCBA"),
                "Assigned menu order changed")
        target = descriptions["report"]
    elif kind == "identify_resolved":
        target = f"Candidate {'WXYZ'[shown['compatible'][0]]}."
    elif kind in ("new_fact", "repeated_fact"):
        adjective = "already been supplied" if str(a["query"]) in shown["observations"] else "not already supplied"
        target = f"A property fact about the hidden candidate that {'has ' if kind == 'repeated_fact' else 'is '}{adjective} in the complete record above."
    elif kind == "pattern_independence":
        target = "None; its generation is independent of the hidden candidate."
    elif kind == "action_consequence":
        require(dict(option_rows[:4]) == {k: descriptions[v] for k, v in mapping.items()}, "Consequence menu changed")
        match = re.search(r"What happens if action ([ABCD]) is selected\?", text)
        require(match is not None and mapping[match[1]] == SERVICES[a["world_slot"] // 3], "Wrong consequence action")
        target = dict(report=f"Only the {report['name']} report is supplied, and the episode ends.",
                      repeat="Only the initial observation is supplied again, and the episode ends.",
                      pattern="Only a new independent decorative pattern is supplied, and the episode ends.",
                      stop="Nothing further is supplied, and the episode ends.")[mapping[match[1]]]
    else:
        raise ValueError("Unknown diagnostic kind")
    matches = [label for label, value in options.items() if value == target]
    require(len(matches) == 1, "Semantic answer missing/duplicated")
    uncertainty = math.log2(len(shown["compatible"]))
    return dict(**shown, truth=truth, expected=None if kind == "choice" else matches[0], mapping=mapping,
                initial_fact=initial, report_fact=report, observation_ledger=ledger,
                uncertainty_bits=uncertainty)


def build_oracles(materials):
    """All truth values come from resolved displayed records, never generated answers."""
    assignments, base = materials["assignments"], materials["base_assignments"]
    require(len(assignments) == len({a["trial_id"] for a in assignments}) == ASSIGNED
            and digest(assignments) == materials["assignment_sha256"], "Incomplete/unbound assigned inventory")
    require(len(base) == len({a["base_trial_id"] for a in base}) == 288
            and digest(base) == materials["base_assignment_sha256"], "Incomplete/unbound base inventory")
    frozen = materials["source_generation_preflight"]["requests"]
    require(len(frozen) == 288 and digest(frozen) == materials["source_generation_preflight"]["requests_sha256"],
            "Unbound source preflight")
    displays, truths, tables = {}, {}, {}
    bases = {a["base_trial_id"]: a for a in base}
    for b, f in zip(base, frozen, strict=True):
        a, request = b["source_assignment"], f["request"]
        require(a["trial_id"] == b["base_trial_id"] == f["base_trial_id"] == request["source_trial_id"]
                and digest(a) == b["source_assignment_sha256"] == f["source_assignment_sha256"]
                and digest(request["token_prefix"]) == request["token_prefix_sha256"]
                and request["seed"] == a["sampling_seed"] == b["sampling_seed"], "Source request/assignment changed")
        require(a["task_revision"] == "voluntary-resolution-v2" and a["family"] == "objects"
                and a["known"] == [(a["world_slot"] + a["world_slot"] // 4) % 2]
                and a["query"] == 1 - a["known"][0], "Wrong frozen task assignment")
        shown = displayed(f, a)
        key = b["semantic_world_id"]
        require(key == a["semantic_world_id"] and all(b[k] == a[k] for k in
                ("family", "kind", "delivery", "menu_layout", "world_slot")), "Base/source metadata changed")
        table = (shown["attributes"], shown["candidates"])
        require(tables.setdefault(key, table) == table, "Paired world table changed")
        if a["delivery"] == "decisive_evidence":
            require(truths.setdefault(key, shown["compatible"][0]) == shown["compatible"][0], "Resolved truths disagree")
        displays[b["base_trial_id"]] = shown
    require(len(tables) == len(truths) == 24, "Require24 independently resolved distinct worlds")
    result = {key: oracle(b["source_assignment"], displays[key], truths[b["semantic_world_id"]]) for key, b in bases.items()}
    cells = set()
    slots = defaultdict(dict)
    for a in assignments:
        b = bases[a["base_trial_id"]]
        require(all(same(a[k], v) for k, v in b.items()), "Base prompt/assignment changed across arms")
        require(a["arm"] in ARMS and a["panel_id"] in PANELS and a["world_slot"] in range(12), "Unknown arm/panel/world")
        schedule, direction = ("baseline", None) if a["arm"] == "baseline" else a["arm"].split("__")
        require(a["schedule"] == schedule and a["direction_key"] == direction, "Arm schedule changed")
        key = (a["panel_id"], a["world_slot"], a["arm"], a["kind"], a["delivery"], a["menu_layout"])
        require(key not in cells, "Duplicate scientific cell")
        cells.add(key)
        require(slots[a["panel_id"]].setdefault(a["world_slot"], a["semantic_world_id"]) == a["semantic_world_id"], "World slot changed")
    branches = [("choice", d, m) for d in DELIVERIES for m in (0, 1)] + [
        (k, "decisive_evidence" if k in ("identify_resolved", "repeated_fact") else "redundant_evidence", 0) for k in DIAGNOSTICS]
    require(cells == {(p, w, arm, k, d, m) for p in PANELS for w in range(12) for arm in ARMS for k, d, m in branches},
            "Incomplete fixed factorial")
    require(len(set(v for panel in slots.values() for v in panel.values())) == 24, "Repeated world across panels")
    return result


def parse_completion(completion, seed):
    raw, tokens, eos = completion["raw"], completion["generated_token_ids"], completion["ended_with_eos"]
    require(isinstance(raw, str) and isinstance(tokens, list) and 1 <= len(tokens) <= CAP
            and all(type(t) is int and t >= 0 for t in tokens) and type(eos) is bool,
            "Malformed raw completion metadata")
    require(eos or len(tokens) == CAP, "Unterminated sub-cap response is unavailable")
    require(same(completion["sampling"], dict(seed=seed, temperature=0.7, top_p=1.0, max_new_tokens=CAP)),
            "Sampling regime changed")
    closed = "</think>" in raw
    answer = raw.rpartition("</think>")[2].strip() if closed else ""
    parsed = answer[:-1] if answer.endswith(".") else answer
    valid = closed and re.fullmatch(r"[ABCD]\.?", answer) is not None
    return dict(answer=answer, parsed_answer=parsed, valid=bool(valid), thinking_closed=closed,
                cap_hit=not eos and len(tokens) == CAP, ended_with_eos=eos,
                generated_token_count=len(tokens), raw_response_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                generated_token_ids_sha256=digest(tokens))


def simulator(a, o, service):
    if service in ("report", "repeat"):
        fact = o["report_fact"] if service == "report" else o["initial_fact"]
        payload, facts = fact_text(fact), [fact]
    else:
        payload, facts = (pattern(a, "selected_service") if service == "pattern" else None), []
    selected = event("selected_service", payload, facts)
    selected.update(service=service, episode_ended=True, model_pass_completed=False)
    ledger = deepcopy(o["observation_ledger"]) + [selected]
    observations = {str(f["attribute"]): f["value"] for e in ledger for f in e["facts"]}
    compatible = [i for i, r in enumerate(o["candidates"]) if all(r[int(k)] == v for k, v in observations.items())]
    return dict(selected_service=service, payload=payload, payload_sha256=digest(payload), terminal=True,
                observation_ledger=ledger, observations=observations, compatible=compatible,
                uncertainty_before_bits=o["uncertainty_bits"], uncertainty_after_bits=math.log2(len(compatible)),
                receipt_required=payload is not None, receipt_model_pass_completed=False)


def score_row(a, observed, o):
    row = {k: a[k] for k in METADATA}
    row.update(status="missing", valid=None, correct=None, thinking_closed=None, cap_hit=None,
               selected_service=None, expected=o["expected"], record_sha256=None, scoring_checked=False)
    mismatches = []
    if observed is None:
        return row, mismatches

    def compare(field, actual, expected):
        if not same(actual, expected):
            mismatches.append(dict(trial_id=a["trial_id"], field=field,
                                   stored_sha256=digest(actual), independently_derived_sha256=digest(expected)))

    compare("assignment", observed.get("assignment"), a)
    row["record_sha256"] = observed.get("record_sha256")
    status = observed["status"]
    require(status in ("completed", "missing", "execution_error"), "Unknown generation status")
    row["status"] = status
    if status != "completed":
        compare("unavailable.task_record", observed.get("task_record"), None)
        compare("unavailable.completion", observed.get("completion"), None)
        return row, mismatches
    inner, completion = observed.get("task_record"), observed.get("completion")
    try:
        require(isinstance(inner, dict) and isinstance(completion, dict), "Completed row lacks raw evidence")
        parsed = parse_completion(completion, a["sampling_seed"])
    except (ValueError, KeyError, TypeError) as error:
        row.update(status="execution_error", scoring_error=str(error))
        mismatches.append(dict(trial_id=a["trial_id"], field="completion_availability", error=str(error)))
        return row, mismatches
    if not isinstance(row["record_sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", row["record_sha256"]) is None:
        mismatches.append(dict(trial_id=a["trial_id"], field="record_sha256", error="Missing durable record binding"))
    compare("inference", inner.get("inference"), completion)
    compare("source_assignment", inner.get("assignment"), a["source_assignment"])
    for field, expected in dict(trial_id=a["base_trial_id"], kind=a["kind"], terminal=True,
                                expected=o["expected"], generation_max_new_tokens=CAP, action_mapping=o["mapping"]).items():
        compare(field, inner.get(field), expected)
    correct = None if a["kind"] == "choice" else parsed["valid"] and parsed["parsed_answer"] == o["expected"]
    selected = o["mapping"].get(parsed["parsed_answer"]) if a["kind"] == "choice" and parsed["valid"] else None
    for key in ("answer", "parsed_answer", "valid", "thinking_closed", "cap_hit"):
        compare(key, inner.get(key), parsed[key])
    compare("correct", inner.get("correct"), correct)
    compare("selected_service", inner.get("selected_service"), selected)
    derived_simulator = simulator(a["source_assignment"], o, selected) if selected else None
    compare("simulator_outcome", inner.get("simulator_outcome"), derived_simulator)
    for field in ("attributes", "candidates", "truth", "observations", "compatible", "initial_fact", "report_fact", "observation_ledger", "uncertainty_bits"):
        compare("task." + field, inner.get("task", {}).get(field), o[field])
    for field, expected in dict(information_bits=o["uncertainty_bits"], initial_uncertainty_bits=1.0,
                                delivery_information_bits=1.0-o["uncertainty_bits"], action_mapping=o["mapping"],
                                menu_order=list("ABCD" if a["menu_layout"] == 0 else "DCBA"),
                                diagnostic_consequence_service=SERVICES[a["world_slot"] // 3]).items():
        compare("task." + field, inner.get("task", {}).get(field), expected)
    row.update({k: v for k, v in parsed.items() if k not in ("answer", "parsed_answer")})
    row.update(correct=correct, selected_service=selected, parsed_answer=parsed["parsed_answer"] if parsed["valid"] else None,
               scoring_checked=True)
    return row, mismatches


def counts(rows):
    observed = [r for r in rows if r["status"] == "completed"]
    return dict(assigned=len(rows), observed=len(observed), valid=sum(r["valid"] is True and r["thinking_closed"] is True for r in observed),
                correct=sum(r["correct"] is True for r in observed), cap_hits=sum(r["cap_hit"] is True for r in observed),
                unclosed=sum(r["thinking_closed"] is False for r in observed), missing=sum(r["status"] == "missing" for r in rows),
                execution_failures=sum(r["status"] == "execution_error" for r in rows))


def readiness(outcomes):
    groups = defaultdict(list)
    for r in outcomes:
        groups[r["arm"], r["panel_id"], r["kind"], r["delivery"]].append(r)
    diagnostics, choices, arms = [], [], []
    for arm in ARMS:
        combined = []
        for panel in PANELS:
            for kind in DIAGNOSTICS:
                rows = [r for d in DELIVERIES for r in groups[arm, panel, kind, d]]
                combined.extend(rows)
                c = counts(rows)
                diagnostics.append(dict(arm=arm, panel_id=panel, kind=kind, **c,
                    passed=c["assigned"] == c["observed"] == 12 and c["valid"] >= 11 and c["correct"] >= 11 and c["cap_hits"] == 0,
                    original_pristine_format_passed=c["valid"] == 12))
            for delivery in DELIVERIES:
                c = counts(groups[arm, panel, "choice", delivery])
                c["correct"] = None
                choices.append(dict(arm=arm, panel_id=panel, delivery=delivery, **c,
                    passed=c["assigned"] == c["observed"] == 24 and c["valid"] >= 23 and c["cap_hits"] == 0))
        c = counts(combined)
        arms.append(dict(arm=arm, **c, passed=c["assigned"] == c["observed"] == 144 and c["valid"] >= 137 and c["cap_hits"] == 0))
    return dict(passed=all(r["passed"] for r in diagnostics + choices + arms), diagnostic_cells=diagnostics,
                choice_cells=choices, arm_diagnostics=arms,
                original_all_pristine_format_passed=all(r["original_pristine_format_passed"] for r in diagnostics),
                interpretation="Observed instrument readiness only; no belief preservation or confidence equivalence.")


def score_collection(materials, observed):
    oracles = build_oracles(materials)
    assigned = materials["assignments"]
    require(set(observed) <= {a["trial_id"] for a in assigned}, "Unassigned generation observation")
    rows, mismatches = [], []
    for a in assigned:
        row, changed = score_row(a, observed.get(a["trial_id"]), oracles[a["base_trial_id"]])
        rows.append(row)
        mismatches.extend(changed)
    checked = sum(r["scoring_checked"] for r in rows)
    return dict(outcomes=rows, readiness=readiness(rows), mismatches=mismatches,
        checks=dict(computation_agrees=not mismatches, complete_response_coverage=checked == ASSIGNED,
                    all_assigned_slots_retained=len(rows) == ASSIGNED),
        counts=dict(assigned=ASSIGNED, scoring_checked=checked, **dict(Counter(r["status"] for r in rows))),
        bindings=dict(materials_sha256=digest(materials), assignment_sha256=digest(assigned),
                      observed_sha256=digest(observed), outcomes_sha256=digest(rows)),
        scope=dict(independent_displayed_world_oracle=True, independent_raw_text_grammar=True,
                   raw_token_decode_verified=False, source_and_runtime_verified=False,
                   actual_receipt_forward_verified=False, scientific_admission=False))


def receipt_expected(a, observed, o):
    row, _ = score_row(a, observed, o)
    if row["status"] != "completed":
        return row, None, "execution_error" if row["status"] == "execution_error" else "missing"
    outcome = simulator(a["source_assignment"], o, row["selected_service"]) if row["selected_service"] else None
    status = "capped_choice" if row["cap_hit"] else "invalid_choice" if not row["valid"] else "stop" if row["selected_service"] == "stop" else "completed"
    return row, outcome, status


def check_receipts(materials, observed, receipt_rows):
    oracles = build_oracles(materials)
    choices = [a for a in materials["assignments"] if a["kind"] == "choice"]
    require(set(receipt_rows) <= {a["trial_id"] for a in choices}, "Unassigned/diagnostic receipt")
    findings, dispositions = [], []
    for a in choices:
        tid = a["trial_id"]
        source, side = observed.get(tid), receipt_rows.get(tid)
        row, outcome, expected_status = receipt_expected(a, source, oracles[a["base_trial_id"]])
        if side is None:
            dispositions.append(dict(trial_id=tid, status="missing", expected_status=expected_status, passed=False))
            continue
        changed = []

        def compare(key, value, expected):
            if not same(value, expected):
                changed.append(dict(trial_id=tid, field=key, stored_sha256=digest(value), independently_derived_sha256=digest(expected)))

        compare("source_trial_id", side.get("source_trial_id"), tid)
        compare("inner_source_trial_id", side.get("inner_source_trial_id"), a["base_trial_id"])
        compare("assignment_sha256", side.get("assignment_sha256"), digest(a))
        compare("terminal", side.get("terminal"), True)
        compare("source_record_sha256", side.get("source_record_sha256"), source.get("record_sha256") if source else None)
        compare("source_task_record_sha256", side.get("source_task_record_sha256"), digest(source["task_record"]) if source and source.get("task_record") is not None else None)
        if source and source.get("task_record") is not None:
            compare("source_assignment_sha256", side.get("source_assignment_sha256"), digest(a["source_assignment"]))
        compare("selected_service", side.get("selected_service"), row["selected_service"])
        compare("simulator_outcome_sha256", side.get("simulator_outcome_sha256"), digest(outcome))
        active = expected_status == "completed"
        required = active if row["status"] == "completed" else None
        compare("receipt_required", side.get("receipt_required"), required)
        actual_status = side.get("status")
        completed = active and actual_status == "completed"
        compare("receipt_model_pass_completed", side.get("receipt_model_pass_completed"), completed)
        if expected_status != "missing":
            allowed = ("completed", "execution_error") if active else (expected_status,)
            if actual_status not in allowed:
                changed.append(dict(trial_id=tid, field="status", error="Unexpected receipt disposition"))
        else:
            changed.append(dict(trial_id=tid, field="status", error="Receipt for missing generation"))
        executed = deepcopy(outcome)
        if completed:
            executed["receipt_model_pass_completed"] = True
            executed["observation_ledger"][-1]["model_pass_completed"] = True
        compare("executed_simulator_outcome", side.get("executed_simulator_outcome"), executed)
        request = side.get("receipt_input")
        if active and request is not None:
            text = oracles[a["base_trial_id"]]["text"]
            parts = text.split("<|im_start|>user\n")
            require(len(parts) == 2 and "<|im_end|>" in parts[1], "Cannot independently recover source user message")
            user = parts[1].split("<|im_end|>", 1)[0]
            messages = [dict(role="system", content="Follow the instructions and requested response format."),
                        dict(role="user", content=user), dict(role="assistant", content=source["completion"]["raw"]),
                        dict(role="user", content=outcome["payload"] + "\nEpisode ended. No response is requested.")]
            expected = dict(messages=messages, messages_sha256=digest(messages), source_trial_id=a["base_trial_id"],
                            assignment_sha256=digest(a["source_assignment"]), selected_payload_sha256=outcome["payload_sha256"],
                            simulator_outcome_sha256=digest(outcome), max_new_tokens=0, seed=a["sampling_seed"], model_pass_completed=False)
            for key, value in expected.items():
                compare("receipt_input." + key, request.get(key), value)
        elif completed:
            changed.append(dict(trial_id=tid, field="receipt_input", error="Completed receipt lacks input"))
        elif not active:
            compare("receipt_input", request, None)
            for field in ("token_binding", "measurement", "activation_artifact", "cohort"):
                compare(field, side.get(field), None)
        if not completed:
            compare("measurement", side.get("measurement"), None)
        passed = not changed and actual_status not in ("execution_error", "missing", "pending")
        dispositions.append(dict(trial_id=tid, status=actual_status, expected_status=expected_status, passed=passed))
        findings.extend(changed)
    return dict(checks=dict(semantic_agreement=not findings, complete_receipt_coverage=len(receipt_rows) == 4176,
                           required_semantic_dispositions_passed=all(r["passed"] for r in dispositions)),
                counts=dict(assigned=4176, observed=len(receipt_rows), **dict(Counter(r["status"] for r in dispositions))),
                dispositions=dispositions, mismatches=findings,
                bindings=dict(materials_sha256=digest(materials), observed_sha256=digest(observed), receipt_rows_sha256=digest(receipt_rows)),
                actual_delivery_verified=False,
                scope="Selected payload/ledger/messages only; renderer, zero-generation calls, hashes, geometry and read-only guards require external audit.")



def assemble_analysis_input(materials, final_audit, analysis_metadata):
    """Project a composed audit without promoting failed lifecycle or readiness.

    The caller still verifies the audit file's byte/source provenance. This pure
    adapter checks its material/outcome bindings and requires both composition
    receipts; the held estimator independently validates its own metadata.
    """
    gates = ("sham_identity_passed", "engagement_passed", "receipts_passed",
             "cross_arm_prepatch_identity_passed", "worker_policies_equal")
    assignments = materials["assignments"]
    outcomes = final_audit["outcomes"]
    require(len(assignments) == len(outcomes) == ASSIGNED
            and digest(assignments) == materials["assignment_sha256"] == final_audit["assignment_sha256"]
            and final_audit["materials_sha256"] == digest(materials)
            and final_audit["outcome_sha256"] == digest(outcomes), "Composed audit material/outcome binding differs")
    for key in ("manifest_sha256", "context_audit_byte_sha256", "lifecycle_audit_byte_sha256"):
        require(isinstance(final_audit.get(key), str) and re.fullmatch(r"[0-9a-f]{64}", final_audit[key]) is not None,
                "Require final composed audit and receipt bindings: " + key)
    require(final_audit.get("parent_manifest_sha256") == final_audit["manifest_sha256"], "Audit parent identity differs")
    require(all(type(final_audit.get(k)) is bool for k in ("complete", "integrity_passed", "lifecycle_shutdown_verified"))
            and set(final_audit["checks"]) == set(gates)
            and all(type(v) is bool for v in final_audit["checks"].values())
            and type(final_audit["readiness"]["passed"]) is bool,
            "Require explicit strict audit gates")
    require(final_audit["lifecycle_shutdown_verified"] or not final_audit["integrity_passed"],
            "A failed lifecycle cannot retain integrity admission")
    compact = [{key: a[key] for key in METADATA} for a in assignments]
    require(len({r["trial_id"] for r in outcomes}) == ASSIGNED
            and all(all(same(r.get(key), a[key]) for key in METADATA)
                    for a, r in zip(compact, outcomes, strict=True)), "Audited outcome assignment/order changed")
    return dict(analysis_metadata=deepcopy(analysis_metadata), assignments=compact,
                assignment_sha256=digest(compact), outcomes=deepcopy(outcomes),
                complete=final_audit["complete"], integrity_passed=final_audit["integrity_passed"],
                checks=deepcopy(final_audit["checks"]), readiness=deepcopy(final_audit["readiness"]),
                source_assignment_sha256=materials["assignment_sha256"], final_audit_sha256=digest(final_audit),
                manifest_sha256=final_audit["manifest_sha256"], lifecycle_shutdown_verified=final_audit["lifecycle_shutdown_verified"],
                context_audit_byte_sha256=final_audit["context_audit_byte_sha256"],
                lifecycle_audit_byte_sha256=final_audit["lifecycle_audit_byte_sha256"])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSON {materials,observed,receipt_rows(optional)}")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "Scoring output is immutable")
    raw = args.input.read_bytes()
    payload = strict_json(raw)
    result = score_collection(payload["materials"], payload["observed"])
    if "receipt_rows" in payload:
        result["receipts"] = check_receipts(payload["materials"], payload["observed"], payload["receipt_rows"])
    result.update(input_byte_sha256=hashlib.sha256(raw).hexdigest(), scorer_byte_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
