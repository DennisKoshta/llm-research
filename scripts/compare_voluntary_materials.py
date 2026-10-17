"""Bind v2 materials to exactly one permitted wording change against frozen v1."""

import argparse
from collections import Counter
import hashlib
import importlib.util
from pathlib import Path

from curiosity_study import voluntary_resolution as voluntary
from curiosity_study.artifacts import digest, read, write_once

V1_SHA = "eda9deec8a446c7af317ae3887c95c3d5af74ddd30b304d72646d4a4f0f8dc0e"


def compare(plan, old_source):
    old_source = Path(old_source)
    if hashlib.sha256(old_source.read_bytes()).hexdigest() != V1_SHA:
        raise ValueError("Require the actual frozen v1 task source")
    if plan != voluntary.assignment_plan(
        plan["selection"]["seed"], excluded_world_ids=plan["selection"]["historical_exact_exclusions"]
    ):
        raise ValueError("Require exact prospective v2 assignments")
    spec = importlib.util.spec_from_file_location("curiosity_study._frozen_voluntary_materials", old_source)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    counts, bindings = Counter(), []
    for a in plan["assignments"]:
        prior = {**a, "task_revision": old.REVISION}
        old_prompt, old_labels, old_expected = old.prompt(prior)
        new_prompt, labels, expected = voluntary.prompt(a)
        if old.context(prior) != voluntary.context(a) or (old_labels, old_expected) != (labels, expected):
            raise ValueError("Context or answer semantics changed")
        counts["contexts_unchanged"] += 1
        if a["kind"] in ("new_fact", "repeated_fact"):
            service = old.service_descriptions(prior)["report"]
            old_question = f"What kind of information would this action supply? {service}"
            new_question = (
                "Both the Initial observation and the Additional delivery above have already been received. "
                f"Now suppose you choose this action:\n{service}\n"
                "Compared with all property facts supplied in the complete record above, "
                "what would this newly requested report provide?"
            )
            expected_prompt = (
                old_prompt.replace(old_question, new_question)
                .replace("not already supplied.", "not already supplied in the complete record above.")
                .replace(
                    "has already been supplied.", "has already been supplied in the complete record above."
                )
            )
            if old_prompt == new_prompt or expected_prompt != new_prompt:
                raise ValueError("Report diagnostic differs beyond the fixed clarification")
            counts["clarified_diagnostic_prompts"] += 1
        else:
            if old_prompt != new_prompt:
                raise ValueError("A non-targeted prompt changed")
            counts[
                "optional_prompts_unchanged"
                if a["kind"] == "choice"
                else "other_diagnostic_prompts_unchanged"
            ] += 1
        bindings.append(
            dict(
                trial_id=a["trial_id"],
                kind=a["kind"],
                assignment_sha256=digest(a),
                context_sha256=digest(voluntary.context(a)),
                old_prompt_sha256=digest(old_prompt),
                new_prompt_sha256=digest(new_prompt),
                expected_answer=expected,
            )
        )
    required = dict(
        contexts_unchanged=288,
        optional_prompts_unchanged=144,
        other_diagnostic_prompts_unchanged=96,
        clarified_diagnostic_prompts=48,
    )
    if dict(counts) != required:
        raise ValueError("Material comparison omitted assigned cases")
    return dict(
        passed=True,
        task_revision=voluntary.REVISION,
        seed=plan["selection"]["seed"],
        assignment_sha256=plan["assignment_sha256"],
        selection_sha256=plan["selection_sha256"],
        task_module_sha256=hashlib.sha256(Path(voluntary.__file__).read_bytes()).hexdigest(),
        frozen_v1_task_sha256=V1_SHA,
        comparison_script_byte_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        counts=dict(counts),
        bindings_sha256=digest(bindings),
        bindings=bindings,
        interpretation="Assignment-only material check; no model outputs or scientific readiness inferred.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--v1-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Material comparison receipts are immutable")
    result = compare(read(args.plan), args.v1_source)
    write_once(args.out, result)
    print({key: result[key] for key in ("passed", "counts")})
