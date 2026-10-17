"""Check every assigned prompt and possible receipt with tokenizer only; no weights."""

import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace

from transformers import AutoTokenizer

from curiosity_study import thinking_information as thinking
from curiosity_study import voluntary_capture as capture
from curiosity_study import voluntary_resolution as voluntary
from curiosity_study.artifacts import digest, read, write_once


def preflight(plan):
    if plan != voluntary.assignment_plan(
        plan["selection"]["seed"],
        excluded_world_ids=plan["selection"]["historical_exact_exclusions"],
    ):
        raise ValueError("Preflight requires the exact frozen prospective assignment plan")
    tokenizer = AutoTokenizer.from_pretrained(
        **dict(
            pretrained_model_name_or_path=capture.MODEL["model_id"],
            revision=capture.MODEL["revision"],
            local_files_only=True,
            trust_remote_code=False,
        )
    )

    def encode(messages, enable_thinking=False):
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )
        return rendered, tokenizer(rendered, add_special_tokens=False, return_tensors="pt")

    model = SimpleNamespace(tokenizer=tokenizer, encode=encode)
    source_bindings, receipt_bindings, stops = [], [], 0
    for assignment in plan["assignments"]:
        request = thinking.inference_request(assignment)
        rendered, inputs = encode(request["messages"], enable_thinking=True)
        offsets = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        marker = "End of record."
        if rendered.count(marker) != 1 or offsets["input_ids"] != inputs.input_ids[0].tolist():
            raise ValueError("Invalid source record boundary or tokenizer identity")
        end = rendered.index(marker) + len(marker)
        matches = [i for i, (start, stop) in enumerate(offsets["offset_mapping"]) if start <= end - 1 < stop]
        if len(matches) != 1 or rendered[end : offsets["offset_mapping"][matches[0]][1]].strip():
            raise ValueError("Source record token includes later question content")
        source_bindings.append(
            dict(
                trial_id=assignment["trial_id"],
                rendered_sha256=digest(rendered),
                token_ids_sha256=digest(offsets["input_ids"]),
                token_count=len(offsets["input_ids"]),
                record_position=matches[0],
            )
        )
        if assignment["kind"] != "choice":
            continue
        for service in voluntary.SERVICES:
            label = next(k for k, v in voluntary.task(assignment)["action_mapping"].items() if v == service)
            raw = f"</think>\n{label}"
            record = thinking.record_from_inference(
                assignment,
                dict(
                    raw=raw,
                    generated_token_ids=tokenizer.encode(raw, add_special_tokens=False),
                    ended_with_eos=True,
                ),
            )
            if service == "stop":
                if voluntary.receipt_input(record) is not None:
                    raise ValueError("Stopping cannot require a payload receipt")
                stops += 1
                continue
            _, _, binding = capture.render_receipt(record, model)
            receipt_bindings.append(
                dict(
                    trial_id=assignment["trial_id"],
                    selected_service=service,
                    token_binding_sha256=digest(binding),
                    prefix_positions=binding["prefix_positions"],
                )
            )
    if (len(source_bindings), len(receipt_bindings), stops) != (288, 432, 144):
        raise ValueError("Preflight must retain every assigned prompt and possible action")
    return dict(
        passed=True,
        task_revision=voluntary.REVISION,
        seed=plan["selection"]["seed"],
        assignment_sha256=plan["assignment_sha256"],
        selection_sha256=plan["selection_sha256"],
        model=capture.MODEL,
        rows=432,
        source_rows=288,
        stop_checks=144,
        synthetic_answer_only=True,
        generated_model_outputs=False,
        task_module_sha256=hashlib.sha256(Path(voluntary.__file__).read_bytes()).hexdigest(),
        preflight_script_byte_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        bindings_sha256=digest(dict(source=source_bindings, receipts=receipt_bindings)),
        source_bindings=source_bindings,
        receipt_bindings=receipt_bindings,
        checks=[
            "All source record offsets preserve the question boundary",
            "Every possible non-stop receipt has valid payload/final offsets",
            "Stop never requires a receipt",
            "Only pinned cached tokenizer loaded; no model weights or inference",
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Preflight receipts are immutable")
    result = preflight(read(args.plan))
    write_once(args.out, result)
    print({key: result[key] for key in ("passed", "rows", "source_rows", "generated_model_outputs")})
