"""One optional information action after randomized evidence; no execution or fitting."""

from copy import deepcopy
import itertools
import math
import random

from . import information_state as state
from .artifacts import digest
from .identification import final_answer

REVISION = "voluntary-resolution-v2"
SEED = 2026101435
ITEMS = 12
FAMILIES = ("objects", "machines")
SERVICES = ("report", "repeat", "pattern", "stop")
DELIVERIES = ("decisive_evidence", "redundant_evidence", "independent_novelty")
DIAGNOSTICS = (
    "identify_resolved",
    "new_fact",
    "repeated_fact",
    "pattern_independence",
    "instrument_useful_action",
    "action_consequence",
)
LABELS = "ABCD"
EXECUTION = {"backend": "torch_batched_v1", "batch_size": 32}


def _known_attribute(slot):
    # Six of each property, and both roles within every three-world mapping stratum.
    return (slot + slot // 4) % 2


def canonical_world(family, world):
    """Ignore display order and question; bind the named table and realized truth."""
    order = sorted(range(2), key=world["attributes"].__getitem__)
    return dict(
        schema="voluntary_finite_world_v1",
        family=family,
        attributes=[world["attributes"][i] for i in order],
        candidates=sorted([row[i] for i in order] for row in world["candidates"]),
        truth=[world["candidates"][world["truth"]][i] for i in order],
    )


def select_worlds(seed=SEED, items=ITEMS, max_candidates=10000, excluded_world_ids=()):
    """Deterministic assignment-only rejection sampling, with every attempt retained."""
    if type(items) is not int or items != ITEMS:
        raise ValueError("This bounded pilot has twelve worlds per family")
    if not isinstance(excluded_world_ids, (list, tuple, set, frozenset)) or any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in excluded_world_ids
    ):
        raise ValueError("Historical exact exclusions must be a flat collection of SHA256 world IDs")
    historical = sorted(set(excluded_world_ids))
    excluded = set(historical)
    worlds, ledger = [], []
    for family in FAMILIES:
        seen = {}
        for index in range(max_candidates):
            slot = len(seen)
            source = dict(
                family=family,
                seed=seed,
                index=index,
                vocabulary="heldout",
                known=[_known_attribute(slot)],
                query=1 - _known_attribute(slot),
            )
            semantic = digest(canonical_world(family, state.task(source)))
            accepted = semantic not in seen and semantic not in excluded
            ledger.append(
                dict(
                    family=family,
                    candidate_index=index,
                    semantic_world_id=semantic,
                    accepted=accepted,
                    reason=(
                        "selected"
                        if accepted
                        else "historical_exact_exposure"
                        if semantic in excluded
                        else "duplicate_semantic_world"
                    ),
                    first_selected_world_slot=slot if accepted else seen.get(semantic),
                )
            )
            if accepted:
                seen[semantic] = slot
                worlds.append({**source, "world_slot": slot, "semantic_world_id": semantic})
            if len(seen) == items:
                break
        else:
            raise ValueError("Candidate limit exhausted before twelve unique worlds")
    return dict(
        seed=seed,
        worlds=worlds,
        candidate_exclusion_ledger=ledger,
        historical_exact_exclusions=historical,
        historical_exact_exclusions_sha256=digest(historical),
        semantic_rule="Named candidate table and realized truth; ignores row order, delivery, menu and question.",
        freshness_scope="Unique within this collection and disjoint from the supplied historical exact exclusions; coverage of all previously inspected collections and projected overlap need a separate audit.",
    )


def assignment_plan(seed=SEED, excluded_world_ids=()):
    selection = select_worlds(seed, excluded_world_ids=excluded_world_ids)
    rows = []
    for world in selection["worlds"]:
        variants = [("choice", delivery, menu) for delivery, menu in itertools.product(DELIVERIES, (0, 1))]
        variants += [
            (
                kind,
                "decisive_evidence"
                if kind in ("identify_resolved", "repeated_fact")
                else "redundant_evidence",
                0,
            )
            for kind in DIAGNOSTICS
        ]
        for kind, delivery, menu in variants:
            key = (
                f"voluntary-{seed}-{world['family']}-w{world['world_slot']:02d}-{kind}-{delivery}-menu{menu}"
            )
            rows.append(
                dict(
                    **world,
                    assay="thinking_information",
                    task_revision=REVISION,
                    kind=kind,
                    delivery=delivery,
                    menu_layout=menu,
                    template="table",
                    trial_id=key,
                    sampling_seed=int(digest(f"voluntary-sample:{key}")[:8], 16),
                    max_new_tokens=4096,
                    execution=EXECUTION.copy(),
                )
            )
    random.Random(f"voluntary-order:{seed}").shuffle(rows)
    return dict(
        task_revision=REVISION,
        inference_scope="development",
        assignments=rows,
        assignment_sha256=digest(rows),
        selection=selection,
        selection_sha256=digest(selection),
        counterbalancing=dict(
            known_property="(world_slot + floor(world_slot/4)) mod 2; six of each property, both within each mapping stratum",
            service_mapping="Four cyclic mappings, three worlds each per family; reverse layout preserves letter meanings",
            consequence_service="floor(world_slot/3); each service tested three times per family",
            consequence_tested_letters=dict(A=4, B=2, C=4, D=2),
            limitation="The consequence panel is sparse and does not equally cross tested service with tested action letter; diagnostic answer labels are separately balanced three each.",
        ),
    )


def assignments(seed=SEED, excluded_world_ids=()):
    return assignment_plan(seed, excluded_world_ids=excluded_world_ids)["assignments"]


def pattern_for(a, channel):
    """Separate random stream: no candidate index, truth, observations or table values."""
    if channel not in ("assigned_delivery", "selected_service"):
        raise ValueError("Unknown pattern channel")
    rng = random.Random(f"voluntary-pattern:{a['seed']}:{a['family']}:{a['world_slot']}:{channel}")
    return "\n".join(" ".join(rng.choice((".", "*")) for _ in range(4)) for _ in range(4))


def _fact(world, attribute):
    return dict(
        attribute=attribute,
        name=world["attributes"][attribute],
        value=world["candidates"][world["truth"]][attribute],
    )


def _fact_text(fact):
    return f"The {fact['name']} is {fact['value']}."


def _event(source, payload, facts):
    return dict(source=source, payload=payload, payload_sha256=digest(payload), facts=facts)


def _remaining(world, observations):
    return [
        i
        for i, row in enumerate(world["candidates"])
        if all(row[int(attribute)] == value for attribute, value in observations.items())
    ]


def task(a):
    slot = a["world_slot"]
    if (
        a.get("task_revision") != REVISION
        or a["family"] not in FAMILIES
        or type(slot) is not int
        or not 0 <= slot < ITEMS
        or a["known"] != [_known_attribute(slot)]
        or a["query"] != 1 - _known_attribute(slot)
        or a["delivery"] not in DELIVERIES
        or type(a["menu_layout"]) is not int
        or a["menu_layout"] not in (0, 1)
        or a["kind"] not in ("choice", *DIAGNOSTICS)
        or a.get("vocabulary") != "heldout"
    ):
        raise ValueError("Invalid voluntary-resolution assignment")
    if a["kind"] != "choice":
        expected = (
            "decisive_evidence"
            if a["kind"] in ("identify_resolved", "repeated_fact")
            else "redundant_evidence"
        )
        if a["delivery"] != expected or a["menu_layout"] != 0:
            raise ValueError("Diagnostic contexts have a fixed independent delivery and layout")
    world = state.task(a)
    if digest(canonical_world(a["family"], world)) != a["semantic_world_id"]:
        raise ValueError("Source world does not match its semantic binding")
    initial, report = _fact(world, a["known"][0]), _fact(world, a["query"])
    ledger = [_event("initial_observation", _fact_text(initial), [initial])]
    if a["delivery"] == "independent_novelty":
        delivery = _event(
            "assigned_delivery", "Independent decorative pattern:\n" + pattern_for(a, "assigned_delivery"), []
        )
    else:
        fact = report if a["delivery"] == "decisive_evidence" else initial
        delivery = _event("assigned_delivery", _fact_text(fact), [fact])
    ledger.append(delivery)
    observations = {fact["attribute"]: fact["value"] for event in ledger for fact in event["facts"]}
    compatible = _remaining(world, observations)
    uncertainty = math.log2(len(compatible))
    rotation = slot % 4
    mapping = dict(zip(LABELS, SERVICES[rotation:] + SERVICES[:rotation]))
    result = dict(world)
    result.update(
        observations=observations,
        compatible=compatible,
        uncertainty_bits=uncertainty,
        information_bits=uncertainty,
        initial_uncertainty_bits=1.0,
        delivery_information_bits=1.0 - uncertainty,
        initial_fact=initial,
        report_fact=report,
        observation_ledger=ledger,
        action_mapping=mapping,
        menu_order=list(LABELS if a["menu_layout"] == 0 else LABELS[::-1]),
        # Independent of the menu rotation: all services three times per family;
        # the tested action letters occur 4/2/4/2 times, not always A.
        diagnostic_consequence_service=SERVICES[slot // 3],
    )
    return result


def context(a):
    t = task(a)
    noun = "object" if a["family"] == "objects" else "machine"
    text = f"One hidden {noun} is one of the four rows below. Every row was equally possible initially.\n"
    text += "Candidate | " + " | ".join(t["attributes"]) + "\n"
    text += "\n".join(label + " | " + " | ".join(row) for label, row in zip("WXYZ", t["candidates"]))
    text += "\nInitial observation:\n" + t["observation_ledger"][0]["payload"]
    text += "\nAdditional delivery:\n" + t["observation_ledger"][1]["payload"]
    text += (
        "\nThe hidden candidate does not change. All property observations and reports are accurate, with no measurement noise. "
        "Decorative patterns are generated independently of the hidden candidate and give no evidence about it.\nEnd of record."
    )
    return text


def service_descriptions(a):
    t = task(a)
    return dict(
        report=f"Receive the hidden candidate's {t['attributes'][a['query']]} report, then end.",
        repeat=f"Receive the initial {t['attributes'][a['known'][0]]} observation again, then end.",
        pattern="View a new independent decorative pattern, then end.",
        stop="End without receiving another observation or pattern.",
    )


def _menu(a):
    t, descriptions = task(a), service_descriptions(a)
    return "\n".join(f"{label}: {descriptions[t['action_mapping'][label]]}" for label in t["menu_order"])


def _diagnostic_options(a, options, target):
    """Balance correct labels within each family/kind without changing any world."""
    position = (a["world_slot"] + DIAGNOSTICS.index(a["kind"])) % 4
    others = [key for key in options if key != target]
    random.Random(f"voluntary-options:{a['seed']}:{a['family']}:{a['world_slot']}:{a['kind']}").shuffle(
        others
    )
    ordered = others[:position] + [target] + others[position:]
    return "\n".join(f"{label}: {options[key]}" for label, key in zip(LABELS, ordered)), LABELS[position]


def prompt(a):
    t, text, kind = task(a), context(a), a["kind"]
    ending = "\nReply with one letter only: A, B, C, or D."
    if kind in ("choice", "instrument_useful_action"):
        rules = "\nChoose one action. Every action ends this episode; only the selected item is supplied. "
        goal = (
            "This choice is optional; no action is prescribed."
            if kind == "choice"
            else "For this diagnostic, select the action that supplies a fact about the hidden candidate that you do not already know."
        )
        expected = (
            None if kind == "choice" else next(k for k, v in t["action_mapping"].items() if v == "report")
        )
        return text + rules + goal + "\n" + _menu(a) + ending, LABELS, expected
    if kind == "identify_resolved":
        question = "Which candidate matches all the supplied property observations?"
        options = {i: f"Candidate {label}." for i, label in enumerate("WXYZ")}
        target = t["truth"]
    elif kind in ("new_fact", "repeated_fact"):
        question = (
            "Both the Initial observation and the Additional delivery above have already been received. "
            "Now suppose you choose this action:\n"
            + service_descriptions(a)["report"]
            + "\nCompared with all property facts supplied in the complete record above, "
            "what would this newly requested report provide?"
        )
        options = dict(
            new="A property fact about the hidden candidate that is not already supplied in the complete record above.",
            repeated="A property fact about the hidden candidate that has already been supplied in the complete record above.",
            pattern="Only an independent decorative pattern.",
            none="No property observation and no decorative pattern.",
        )
        target = "new" if kind == "new_fact" else "repeated"
    elif kind == "pattern_independence":
        question = "What evidence about the hidden candidate does an independent decorative pattern provide?"
        options = dict(
            none="None; its generation is independent of the hidden candidate.",
            initial="It identifies the candidate's initially observed property.",
            report="It identifies the property that the available report would state.",
            both="It identifies both properties of the hidden candidate.",
        )
        target = "none"
    else:
        service = t["diagnostic_consequence_service"]
        label = next(k for k, v in t["action_mapping"].items() if v == service)
        question = "Consider this menu:\n" + _menu(a) + f"\nWhat happens if action {label} is selected?"
        options = dict(
            report=f"Only the {t['attributes'][a['query']]} report is supplied, and the episode ends.",
            repeat="Only the initial observation is supplied again, and the episode ends.",
            pattern="Only a new independent decorative pattern is supplied, and the episode ends.",
            stop="Nothing further is supplied, and the episode ends.",
        )
        target = service
    menu, expected = _diagnostic_options(a, options, target)
    return text + "\n" + question + "\n" + menu + ending, LABELS, expected


def inference_request(a):
    text, labels, _ = prompt(a)
    return dict(
        messages=[
            dict(role="system", content="Follow the instructions and requested response format."),
            dict(role="user", content=text),
        ],
        seed=a["sampling_seed"],
        labels=labels,
        greedy=False,
        max_new_tokens=4096,
        enable_thinking=True,
    )


class VoluntaryTrial:
    """Simulator state only: supplying a payload here does not execute a model receipt."""

    def __init__(self, assignment):
        if assignment["kind"] != "choice":
            raise ValueError("Diagnostics cannot acquire payloads")
        self.assignment = deepcopy(assignment)
        self.task = task(assignment)
        self.done = False

    def apply(self, service):
        if self.done or service not in SERVICES:
            raise ValueError("Require one valid action in an unfinished episode")
        t, a = self.task, self.assignment
        if service in ("report", "repeat"):
            fact = deepcopy(t["report_fact"] if service == "report" else t["initial_fact"])
            payload, facts = _fact_text(fact), [fact]
        elif service == "pattern":
            payload, facts = "Independent decorative pattern:\n" + pattern_for(a, "selected_service"), []
        else:
            payload, facts = None, []
        event = _event("selected_service", payload, facts)
        event.update(service=service, episode_ended=True, model_pass_completed=False)
        ledger = deepcopy(t["observation_ledger"]) + [event]
        observed = {f["attribute"]: f["value"] for row in ledger for f in row["facts"]}
        remaining = _remaining(t, observed)
        self.done = True
        return dict(
            selected_service=service,
            payload=payload,
            payload_sha256=digest(payload),
            terminal=True,
            observation_ledger=ledger,
            observations=observed,
            compatible=remaining,
            uncertainty_before_bits=t["uncertainty_bits"],
            uncertainty_after_bits=math.log2(len(remaining)),
            receipt_required=payload is not None,
            receipt_model_pass_completed=False,
        )


def record_from_inference(a, inference):
    """Parse a final answer only; the thinking wrapper must establish closure first."""
    _, labels, expected = prompt(a)
    raw = inference["raw"].strip()
    parsed = raw[:-1] if raw.endswith(".") else raw
    valid = parsed in labels and len(parsed) == 1
    t = task(a)
    return dict(
        trial_id=a["trial_id"],
        assignment=a,
        kind=a["kind"],
        terminal=True,
        task=t,
        inference=inference,
        answer=raw,
        parsed_answer=parsed,
        expected=expected,
        valid=valid,
        correct=valid and parsed == expected if expected is not None else None,
        action_mapping=t["action_mapping"],
        selected_service=t["action_mapping"].get(parsed) if valid and a["kind"] == "choice" else None,
        simulator_outcome=None,
    )


def finalize_choice(record):
    """Require closed thinking before producing any simulated selected consequence."""
    result = deepcopy(record)
    a = result["assignment"]
    if a["kind"] != "choice" or result.get("simulator_outcome") is not None:
        raise ValueError("Only an unfinalized optional choice can be finalized")
    if result.get("thinking_closed") is not True or result.get("valid") is not True:
        result["selected_service"] = None
        return result
    expected = record_from_inference(a, {"raw": result["parsed_answer"]})
    if expected["valid"] is not True or result["selected_service"] != expected["selected_service"]:
        raise ValueError("Choice binding does not match its assigned action mapping")
    result["simulator_outcome"] = VoluntaryTrial(a).apply(result["selected_service"])
    return result


def record_from_thinking_inference(a, inference):
    answer, closed = final_answer(inference["raw"], True)
    record = record_from_inference(a, {**inference, "raw": answer})
    record.update(inference=inference, thinking_closed=closed, valid=closed and record["valid"])
    record["correct"] = (
        None
        if record["expected"] is None
        else record["valid"] and record["parsed_answer"] == record["expected"]
    )
    record["generation_max_new_tokens"] = 4096
    record["cap_hit"] = (
        not inference.get("ended_with_eos", False) and len(inference.get("generated_token_ids", [])) == 4096
    )
    return finalize_choice(record) if a["kind"] == "choice" else record


def receipt_input(record):
    """Exact conversation messages for a future, separately audited read-only pass."""
    if record.get("valid") is not True or record.get("thinking_closed") is not True:
        raise ValueError("A receipt requires a valid closed optional choice")
    a = record["assignment"]
    expected = record_from_thinking_inference(a, record["inference"])
    if any(
        record.get(key) != expected.get(key)
        for key in ("parsed_answer", "selected_service", "action_mapping")
    ):
        raise ValueError("Receipt record does not match its actual generated answer")
    if digest(record.get("simulator_outcome")) != digest(expected["simulator_outcome"]):
        raise ValueError("Selected payload or ledger differs from the actual closed choice")
    outcome = expected["simulator_outcome"]
    if outcome is None:
        raise ValueError("Diagnostic answers have no receipts")
    if outcome["payload"] is None:
        return None
    messages = inference_request(a)["messages"] + [
        dict(role="assistant", content=record["inference"]["raw"]),
        dict(role="user", content=outcome["payload"] + "\nEpisode ended. No response is requested."),
    ]
    return dict(
        messages=messages,
        messages_sha256=digest(messages),
        source_trial_id=a["trial_id"],
        assignment_sha256=digest(a),
        selected_payload_sha256=outcome["payload_sha256"],
        simulator_outcome_sha256=digest(outcome),
        max_new_tokens=0,
        seed=a["sampling_seed"],
        model_pass_completed=False,
        token_binding_status="Conversation messages only; the later receipt executor must bind actual rendered tokens and captures.",
    )
