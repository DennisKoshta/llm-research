"""Draft 48 fixed unsteered probes; no model loading, execution, freeze or admission."""

import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from decimal import Decimal
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
NAME = "epistemic-readout-qualification-v1"
SOURCE_RUN = "qwen32-voluntary-resolution-v2"
SOURCE_MANIFEST_SHA = "b2ef2c264176003c9a0bcedf03a3e7bd80668cef38167d29f9ce3d9005861ab6"
SOURCE_SHA = "88a91e516e2580b23b9468bb1d38ba1c8a2b1e818e5f4e363ded74874a666623"
ARCHIVE_SHA = "042a017d07d97ee149a7648608d7c51aeac4ef3f2f6d241f83179249ffe9a174"
MODEL = dict(model_id="Qwen/Qwen3-32B", revision="9216db5781bf21249d130ec9da846c4624c16137")
READOUTS = ("remaining_set", "candidate_distribution", "report_information")
DELIVERIES = ("decisive_evidence", "redundant_evidence")
CANDIDATES = tuple("WXYZ")
ORDER_SEED = 2026101551
CRITERIA = dict(assigned_worlds=4, assigned_contexts=8, assigned_rows=48,
    all_closed_valid_uncapped=True, all_set_responses_exact=True, all_report_responses_exact=True,
    probability_sum_tolerance=1e-6, maximum_total_variation=0.05,
    maximum_incompatible_mass=0.05, minimum_resolved_max_probability=0.95,
    unresolved_max_probability_interval=[0.45, 0.55], maximum_ordering_total_variation=0.05)
RESOURCES = dict(gpu="H200", cpu=2, memory_gib=48, wall_seconds=1800,
                 teardown_seconds=120, ceiling_usd=3, paid_retries=0)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


@contextmanager
def frozen_source(root):
    """Private immutable v2 package; importing it performs no inference/download."""
    root = Path(root)
    run = root / "runs" / SOURCE_RUN
    manifest, freeze = read(run / "manifest.json"), read(run / "freeze.json")
    source = root / "executions" / SOURCE_RUN / "src/curiosity_study"
    require(digest(manifest) == freeze["manifest_sha256"] == SOURCE_MANIFEST_SHA
            and manifest["model"] == MODEL and manifest["source_sha256"] == freeze["source_sha256"] == SOURCE_SHA
            and freeze["archive_sha256"] == sha(run / "source.tar.gz") == ARCHIVE_SHA,
            "Frozen v2 source identity differs")
    files = {str(p.relative_to(source)): p.read_text() for p in sorted(source.rglob("*.py"))}
    require(digest(files) == SOURCE_SHA and sha(source / "voluntary_resolution.py") == manifest["task_module_sha256"],
            "Frozen v2 task bytes differ")
    name = "_epistemic_qualification_frozen_v2"
    require(not any(k == name or k.startswith(name + ".") for k in sys.modules), "Private source import already active")
    old = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec = importlib.util.spec_from_file_location(name, source / "__init__.py", submodule_search_locations=[str(source)])
        package = importlib.util.module_from_spec(spec)
        sys.modules[name] = package
        spec.loader.exec_module(package)
        task = importlib.import_module(name + ".voluntary_resolution")
        yield task, manifest, dict(run=SOURCE_RUN, manifest_sha256=SOURCE_MANIFEST_SHA,
            source_sha256=SOURCE_SHA, archive_byte_sha256=ARCHIVE_SHA,
            freeze_byte_sha256=sha(run / "freeze.json"), task_module_sha256=manifest["task_module_sha256"],
            source_revision=manifest["task_revision"])
    finally:
        for key in list(sys.modules):
            if key == name or key.startswith(name + "."):
                del sys.modules[key]
        sys.dont_write_bytecode = old


def suffix(readout, ordering, slot, property_name):
    labels = list(CANDIDATES if ordering == 0 else CANDIDATES[::-1])
    current = "Both the Initial observation and the Additional delivery have already been received. Use all the property observations in the complete record above.\n"
    if readout == "remaining_set":
        return (current + "Which candidates are still possible now? Include every candidate compatible with all those observations, and exclude every incompatible candidate.\n"
            + "Candidate order for this question: " + ", ".join(labels) + ".\n"
            + "Reply with only a JSON array of candidate labels. The array must contain each selected label once; label order in the array does not matter."), None
    if readout == "candidate_distribution":
        return (current + "What probability do you assign to each candidate being the hidden object now? Use the stated equal initial chances and the accurate observations.\n"
            + "Candidate order for this question: " + ", ".join(labels) + ".\n"
            + "Reply with only a JSON object containing exactly these four candidate labels as keys and numbers between 0 and 1 as values, summing to 1. Include zero values. Key order does not matter."), None
    options = [
        ("new", "A property value not already established by the complete record; it would distinguish candidates still possible now."),
        ("repeated", "A property value already established by the complete record; it would not distinguish any candidates still possible now."),
        ("pattern", "Only an independent decorative pattern, with no property report."),
        ("none", "Neither a property report nor a decorative pattern."),
    ]
    options = options[slot:] + options[:slot]
    if ordering:
        options.reverse()
    mapping = {label: key for label, (key, _) in zip("ABCD", options)}
    text = (current + f"Now suppose you request the hidden object's {property_name} report and receive it. "
        + "Compared with everything already supplied in the complete record, what additional information would this newly requested report provide?\n")
    text += "\n".join(f"{label}: {value}" for label, (_, value) in zip("ABCD", options))
    return text + "\nReply with one letter only: A, B, C, or D.", mapping


def build_plan(root=ROOT):
    rows, worlds = [], {}
    with frozen_source(root) as (task, manifest, binding):
        selected = [a for a in manifest["diagnostics"] if a["family"] == "objects" and a["world_slot"] in range(4)
                    and a["kind"] == "choice" and a["menu_layout"] == 0 and a["delivery"] in DELIVERIES]
        require(len(selected) == 8 and {(a["world_slot"], a["delivery"]) for a in selected}
                == {(slot, delivery) for slot in range(4) for delivery in DELIVERIES}, "Fixed eight source contexts differ")
        for a in sorted(selected, key=lambda x: (x["world_slot"], DELIVERIES.index(x["delivery"]))):
            t, context, original_request = task.task(a), task.context(a), task.inference_request(a)
            require("Every row was equally possible initially." in context
                    and "All property observations and reports are accurate, with no measurement noise." in context
                    and context.endswith("End of record.") and context.count("End of record.") == 1
                    and original_request["messages"][-1]["content"].startswith(context + "\n"), "Common evidence prefix changed")
            require(a["known"] == [a["world_slot"] % 2], "Four slots no longer cross both known-property roles")
            worlds[a["world_slot"]] = a["semantic_world_id"]
            possible = [CANDIDATES[i] for i in t["compatible"]]
            require(len(possible) == (1 if a["delivery"] == "decisive_evidence" else 2), "Source posterior size differs")
            posterior = {label: 1 / len(possible) if label in possible else 0.0 for label in CANDIDATES}
            partition = {}
            for index in t["compatible"]:
                partition.setdefault(t["candidates"][index][a["query"]], []).append(CANDIDATES[index])
            gain = math.log2(len(possible)) - sum(len(group) / len(possible) * math.log2(len(group))
                                                  for group in partition.values())
            require(gain == (0 if len(possible) == 1 else 1), "Offered report no longer has the stated information gain")
            expected_information = "new" if gain > 0 else "repeated"
            for readout in READOUTS:
                # Shared stream across the two answer orderings, not independent repeats.
                seed = int(digest([NAME, a["semantic_world_id"], a["delivery"], readout])[:8], 16)
                for ordering in (0, 1):
                    text, mapping = suffix(readout, ordering, a["world_slot"], t["attributes"][a["query"]])
                    request = dict(messages=[deepcopy(original_request["messages"][0]),
                        dict(role="user", content=context + "\n" + text)], seed=seed, labels="", greedy=False,
                        enable_thinking=True, max_new_tokens=4096)
                    expected = dict(remaining_candidates=possible, probabilities=posterior,
                        report_information=expected_information, expected_information_gain_bits=gain,
                        report_value_partition=partition,
                        correct_label=None if mapping is None else next(k for k, v in mapping.items() if v == expected_information))
                    rows.append(dict(trial_id=f"{NAME}-w{a['world_slot']:02d}-{a['delivery']}-{readout}-o{ordering}",
                        world_slot=a["world_slot"], semantic_world_id=a["semantic_world_id"], family="objects",
                        delivery=a["delivery"], readout=readout, answer_ordering=ordering,
                        known_property_index=a["known"][0], source_assignment=deepcopy(a), source_assignment_sha256=digest(a),
                        source_request=original_request, source_request_sha256=digest(original_request),
                        common_context=context, common_context_sha256=digest(context), request=request,
                        request_sha256=digest(request), expected=expected, expected_sha256=digest(expected), option_mapping=mapping))
    random.Random(ORDER_SEED).shuffle(rows)
    require(len(rows) == 48 and len(set(worlds.values())) == 4, "Qualification inventory differs")
    cohorts = [dict(offset=i, trial_ids=[a["trial_id"] for a in rows[i:i + 32]]) for i in (0, 32)]
    return dict(schema_version=1, name=NAME, phase="draft_unsteered_readout_qualification", inference_scope="development",
        model=MODEL, source_binding=binding, source_worlds=worlds, worlds_are_previously_exposed=True,
        assignments=rows, assignment_sha256=digest(rows), execution_cohorts=cohorts, order_seed=ORDER_SEED,
        criteria=deepcopy(CRITERIA), resources=deepcopy(RESOURCES), sampling=dict(temperature=0.7, top_p=1.0,
        enable_thinking=True, max_new_tokens=4096), implementation_sha256=sha(__file__),
        native_direction_selected=False, paid_execution_admitted=False, wording_retry_allowed=False)


def tokenize_plan(plan, tokenizer):
    """Offline preflight: preserve original token prefix, not merely context text."""
    def encode(request):
        text = tokenizer.apply_chat_template(request["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=True)
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        end = text.index("End of record.") + len("End of record.")
        positions = [i for i, (start, stop) in enumerate(encoded["offset_mapping"]) if start < end <= stop]
        require(len(positions) == 1 and not text[end:encoded["offset_mapping"][positions[0]][1]].strip(),
                "Record token includes question material")
        return encoded["input_ids"], positions[0]
    requests = []
    for a in plan["assignments"]:
        source, source_position = encode(a["source_request"])
        current, position = encode(a["request"])
        require(source_position == position and source[:source_position + 1] == current[:position + 1],
                "Question suffix changed the original prequestion token prefix")
        requests.append(dict(trial_id=a["trial_id"], input_ids=current, input_ids_sha256=digest(current),
            record_token_index=position, original_common_prefix_sha256=digest(source[:source_position + 1])))
    return dict(assignment_sha256=plan["assignment_sha256"], tokenizer_template_sha256=digest(tokenizer.chat_template),
                requests=requests, requests_sha256=digest(requests), all_original_prefixes_exact=True)


def _object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON object key")
        result[key] = value
    return result


def strict_json(text):
    return json.loads(text, object_pairs_hook=_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON number")))


def score_output(a, inference):
    result = dict(trial_id=a["trial_id"], world_slot=a["world_slot"], semantic_world_id=a["semantic_world_id"],
        delivery=a["delivery"], readout=a["readout"], answer_ordering=a["answer_ordering"],
        assigned=True, observed=inference is not None, closed=False, cap_hit=False, valid=False,
        correct=None, parsed=None, total_variation=None, incompatible_mass=None, max_probability=None,
        distribution_passed=False, error=None)
    if inference is None:
        result["error"] = "missing_output"
        return result
    raw, tokens = inference.get("raw"), inference.get("generated_token_ids")
    if not (isinstance(raw, str) and isinstance(tokens, list) and 0 < len(tokens) <= 4096
            and all(type(t) is int and t >= 0 for t in tokens) and type(inference.get("ended_with_eos")) is bool
            and (inference["ended_with_eos"] or len(tokens) == 4096)):
        result["error"] = "unavailable_generation_evidence"
        return result
    result["closed"] = "</think>" in raw
    result["cap_hit"] = not inference["ended_with_eos"] and len(tokens) == 4096
    if not result["closed"]:
        result["error"] = "unclosed_thinking"
        return result
    final = raw.rsplit("</think>", 1)[1].strip()
    try:
        if a["readout"] == "report_information":
            require(final in "ABCD" and len(final) == 1, "Expected exactly one letter")
            result.update(valid=True, parsed=final, correct=final == a["expected"]["correct_label"])
        elif a["readout"] == "remaining_set":
            value = strict_json(final)
            require(isinstance(value, list) and all(type(v) is str and v in CANDIDATES for v in value)
                    and len(value) == len(set(value)), "Expected a unique candidate-label array")
            result.update(valid=True, parsed=value, correct=set(value) == set(a["expected"]["remaining_candidates"]))
        else:
            value = strict_json(final)
            require(isinstance(value, dict) and set(value) == set(CANDIDATES)
                    and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in value.values())
                    , "Invalid complete probability allocation")
            # Decimal comparison avoids rejecting a boundary allocation such as
            # .95/.05 solely because binary float subtraction exceeds .05.
            decimal = {k: Decimal(str(v)) for k, v in value.items()}
            require(abs(sum(decimal.values()) - 1) <= Decimal("0.000001"), "Probability allocation does not sum to one")
            tv = sum(abs(decimal[k] - Decimal(str(a["expected"]["probabilities"][k]))) for k in CANDIDATES) / 2
            incompatible = sum(decimal[k] for k in CANDIDATES if k not in a["expected"]["remaining_candidates"])
            maximum = max(decimal.values())
            state_check = maximum >= Decimal("0.95") if a["delivery"] == "decisive_evidence" else Decimal("0.45") <= maximum <= Decimal("0.55")
            result.update(valid=True, parsed=value, total_variation=float(tv), incompatible_mass=float(incompatible),
                          max_probability=float(maximum), distribution_passed=tv <= Decimal("0.05") and incompatible <= Decimal("0.05") and state_check)
    except (ValueError, TypeError, OverflowError) as error:
        result["error"] = str(error)
    return result


def qualify(plan, inferences):
    require(isinstance(inferences, dict) and set(inferences) <= {a["trial_id"] for a in plan["assignments"]}, "Unexpected output ID")
    require(plan["criteria"] == CRITERIA and digest(plan["assignments"]) == plan["assignment_sha256"], "Draft criteria/assignments differ")
    scores = [score_output(a, inferences.get(a["trial_id"])) for a in plan["assignments"]]
    pairs, orderings = {}, []
    for score in scores:
        if score["readout"] == "candidate_distribution":
            pairs.setdefault((score["world_slot"], score["delivery"]), {})[score["answer_ordering"]] = score
    for (slot, delivery), pair in sorted(pairs.items()):
        available = set(pair) == {0, 1} and all(v["valid"] for v in pair.values())
        tv = sum(abs(Decimal(str(pair[0]["parsed"][k])) - Decimal(str(pair[1]["parsed"][k]))) for k in CANDIDATES) / 2 if available else None
        orderings.append(dict(world_slot=slot, delivery=delivery, total_variation=None if tv is None else float(tv),
                              passed=available and tv <= Decimal("0.05")))
    checks = dict(all48_closed_valid_uncapped=len(scores) == 48 and all(s["closed"] and s["valid"] and not s["cap_hit"] for s in scores),
        all16_sets_exact=sum(s["readout"] == "remaining_set" and s["correct"] is True for s in scores) == 16,
        all16_report_judgments_exact=sum(s["readout"] == "report_information" and s["correct"] is True for s in scores) == 16,
        all16_distributions_within_bounds=sum(s["readout"] == "candidate_distribution" and s["distribution_passed"] for s in scores) == 16,
        all8_ordering_pairs_within_bounds=len(orderings) == 8 and all(p["passed"] for p in orderings))
    return dict(assignment_sha256=plan["assignment_sha256"], assigned_rows=48, distinct_worlds=4, distinct_contexts=8,
        observed_rows=sum(s["observed"] for s in scores), counts=dict(Counter(s["readout"] for s in scores)),
        passed=all(checks.values()), checks=checks, scores=scores, probability_ordering_pairs=orderings,
        internal_confidence_validated=False, causal_admission=False, missing_outputs_retained=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, required=True, help="Write draft materials only; never a run/freeze")
    args = parser.parse_args()
    plan = build_plan(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(plan, stream, sort_keys=True, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(dict(draft_path=str(args.out), assigned=48, admitted=False, assignment_sha256=plan["assignment_sha256"])))


if __name__ == "__main__":
    main()
