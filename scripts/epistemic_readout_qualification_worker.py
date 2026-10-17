"""Fixed unsteered 48-row qualification using the immutable v2 batch engine.

No capture, intervention, direction fitting, dispatcher, or causal admission.
The controller enforces the process deadline; incomplete rows remain assigned.
"""

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

NAME = "qwen32-epistemic-readout-qualification-v1"
SOURCE_RUN = "qwen32-voluntary-resolution-v2"
MANIFEST_SHA = "b2ef2c264176003c9a0bcedf03a3e7bd80668cef38167d29f9ce3d9005861ab6"
SOURCE_SHA = "88a91e516e2580b23b9468bb1d38ba1c8a2b1e818e5f4e363ded74874a666623"
ARCHIVE_SHA = "042a017d07d97ee149a7648608d7c51aeac4ef3f2f6d241f83179249ffe9a174"
MATERIALS = "prepare_epistemic_readout_qualification.py"
RATE, WALL, TEARDOWN, CEILING = 0.00139376, 1800, 120, 3.0
APP = "curiosity-study-epistemic-readout-qualification-h200"


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
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resources():
    return dict(app_name=APP, gpu="H200", cpu=2, memory_gib=48, wall_seconds=WALL,
                teardown_seconds=TEARDOWN, rate_usd_per_second=RATE, ceiling_usd=CEILING,
                maximum_liability_usd=(WALL + TEARDOWN) * RATE, max_containers=1, retries=0)


def check_files(root, inventory):
    root = Path(root)
    require(isinstance(inventory, dict) and inventory and root.is_dir() and not root.is_symlink()
            and not any(p.is_symlink() for p in root.rglob("*")), "Invalid frozen tree")
    actual = {str(p.relative_to(root)) for p in root.rglob("*")
              if p.is_file() and "__pycache__" not in p.parts}
    require(actual == set(inventory), "Frozen inventory differs")
    for name, checksum in inventory.items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts and str(relative) == name
                and sha(root / name) == checksum, "Frozen bytes differ: " + name)


def validate_manifest(manifest):
    original, freeze = manifest["original_manifest"], manifest["original_freeze"]
    require(manifest["name"] == NAME and manifest["schema_version"] == 1
            and digest(original) == freeze["manifest_sha256"] == MANIFEST_SHA
            and original["source_sha256"] == freeze["source_sha256"] == manifest["source_sha256"] == SOURCE_SHA
            and manifest["original_source_archive_sha256"] == freeze["archive_sha256"] == ARCHIVE_SHA,
            "Immutable original source identity differs")
    require(manifest["resources"] == resources() and manifest["causal_admission"] is False
            and manifest["qualification_only"] is True, "Qualification resource/scope policy differs")
    plan, preflight = manifest["plan"], manifest["preflight"]
    require(digest(plan) == manifest["plan_sha256"] and digest(preflight) == manifest["preflight_sha256"]
            and plan["assignment_sha256"] == digest(plan["assignments"])
            and plan["model"] == original["model"] and plan["native_direction_selected"] is False
            and plan["paid_execution_admitted"] is False, "Plan identity differs")
    require(plan["source_binding"]["manifest_sha256"] == MANIFEST_SHA
            and plan["source_binding"]["source_sha256"] == SOURCE_SHA
            and plan["source_binding"]["archive_byte_sha256"] == ARCHIVE_SHA, "Material source differs")
    assignments = plan["assignments"]
    require(len(assignments) == 48 and len({a["trial_id"] for a in assignments}) == 48,
            "Exactly 48 unique assignments required")
    require(plan["execution_cohorts"] == [dict(offset=i, trial_ids=[a["trial_id"] for a in assignments[i:i + 32]])
                                        for i in (0, 32)], "Fixed 32+16 companions differ")
    require(preflight["assignment_sha256"] == plan["assignment_sha256"]
            and preflight["requests_sha256"] == digest(preflight["requests"])
            and preflight["all_original_prefixes_exact"] is True
            and [r["trial_id"] for r in preflight["requests"]] == [a["trial_id"] for a in assignments],
            "Tokenization inventory differs")
    for a, token in zip(assignments, preflight["requests"]):
        request = a["request"]
        require(set(request) == {"messages", "seed", "labels", "greedy", "enable_thinking", "max_new_tokens"}
                and request["labels"] == "" and request["greedy"] is False
                and request["enable_thinking"] is True and request["max_new_tokens"] == 4096
                and type(request["seed"]) is int and request["seed"] >= 0
                and digest(request) == a["request_sha256"]
                and digest(a["source_assignment"]) == a["source_assignment_sha256"]
                and digest(a["source_request"]) == a["source_request_sha256"]
                and digest(a["common_context"]) == a["common_context_sha256"]
                and digest(a["expected"]) == a["expected_sha256"], "Request/sampler binding differs")
        ids = token["input_ids"]
        require(isinstance(ids, list) and ids and all(type(v) is int and v >= 0 for v in ids)
                and digest(ids) == token["input_ids_sha256"]
                and type(token["record_token_index"]) is int
                and 0 <= token["record_token_index"] < len(ids)
                and digest(ids[:token["record_token_index"] + 1]) == token["original_common_prefix_sha256"],
                "Token prefix binding differs")
    return manifest


def validate_admission(manifest, admission, now):
    require(all(type(v) in (int, float) and math.isfinite(v)
                for v in (admission["created"], admission["deadline"], now))
            and admission["manifest_sha256"] == digest(manifest)
            and admission["phase_consumed"] is True and admission["retry_allowed"] is False
            and admission["deadline"] == admission["created"] + WALL
            and admission["created"] <= now < admission["deadline"]
            and admission["maximum_liability_usd"] == resources()["maximum_liability_usd"],
            "Invalid or expired one-time admission")


def load_materials(wrapper, manifest):
    path = Path(wrapper) / MATERIALS
    require(sha(path) == manifest["wrapper_files"][MATERIALS], "Material scorer bytes differ")
    spec = importlib.util.spec_from_file_location("_frozen_epistemic_materials", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(manifest["plan"]["criteria"] == module.CRITERIA, "Qualification criteria differ")
    return module


def hook_inventory(model):
    return dict(active_hooks=model.active_hooks,
        forward_hooks=sum(len(m._forward_hooks) for m in model.model.modules()),
        forward_pre_hooks=sum(len(m._forward_pre_hooks) for m in model.model.modules()))


def require_hookfree(model):
    observed = hook_inventory(model)
    require(all(value == 0 for value in observed.values()), "Unexpected generation hook")
    return observed


def eos_token_ids(model):
    value = model.model.generation_config.eos_token_id
    ids = [value] if type(value) is int else value
    require(isinstance(ids, (list, tuple)) and ids and all(type(v) is int and v >= 0 for v in ids),
            "Invalid actual generation EOS configuration")
    return sorted(set(ids))


def validate_inference(a, token, inference, cohort, row, tokenizer, eos_ids, expected_width):
    ids = token["input_ids"]
    require(inference["messages"] == a["request"]["messages"]
            and inference["unpadded_input_ids"] == ids
            and inference["prompt_sha256"] == digest(inference["rendered_prompt"])
            and inference["rendered_prompt"] == tokenizer.apply_chat_template(a["request"]["messages"],
                tokenize=False, add_generation_prompt=True, enable_thinking=True)
            and inference["enable_thinking"] is True and inference["hooks_remaining"] == 0
            and inference["intervention"] is None and inference["capture"] == {}
            and inference["sampling"] == dict(seed=a["request"]["seed"], temperature=0.7,
                                               top_p=1.0, max_new_tokens=4096), "Inference/request binding differs")
    batch = inference["batch"]
    require(batch["backend"] == "torch_batched_v1" and batch["cohort_ids"] == cohort["trial_ids"]
            and batch["row"] == row and batch["size"] == len(cohort["trial_ids"]), "Batch companions/order differ")
    padded, mask, positions = inference["input_ids"], inference["attention_mask"], inference["position_ids"]
    padding = len(padded) - len(ids)
    require(padding >= 0 and len(padded) == expected_width and padded == [tokenizer.pad_token_id] * padding + ids
            and mask == [0] * padding + [1] * len(ids)
            and positions == [0] * padding + list(range(len(ids))), "Padding/positions differ")
    generated = inference["generated_token_ids"]
    require(isinstance(generated, list) and 0 < len(generated) <= 4096
            and all(type(v) is int and v >= 0 for v in generated)
            and type(inference["ended_with_eos"]) is bool
            and inference["ended_with_eos"] is (generated[-1] in eos_ids)
            and inference["raw"] == tokenizer.decode(generated, skip_special_tokens=True), "Raw generation evidence differs")


def execute_batches(run, manifest, admission, model, materials, write_once, clock=time.time):
    """Only two fixed forwards; no retry. Persist the attempt before entering each batch."""
    rows, observed = [], {}
    assignments, tokens = manifest["plan"]["assignments"], manifest["preflight"]["requests"]
    stopped = False
    for cohort in manifest["plan"]["execution_cohorts"]:
        offset, size = cohort["offset"], len(cohort["trial_ids"])
        selected = assignments[offset:offset + size]
        error, inferences, attempted = None, None, False
        if not stopped and clock() < admission["deadline"] - 60:
            attempted = True
            write_once(run / "attempts" / f"{offset:04d}.json", dict(manifest_sha256=digest(manifest),
                cohort=cohort, started=clock()))
            try:
                before = require_hookfree(model)
                inferences = model.infer_batch([a["request"] for a in selected], cohort_ids=cohort["trial_ids"])
                require(isinstance(inferences, list) and len(inferences) == size, "Incomplete atomic batch")
                after = require_hookfree(model)
                for row, (a, token, inference) in enumerate(zip(selected, tokens[offset:offset + size], inferences)):
                    validate_inference(a, token, inference, cohort, row, model.tokenizer, eos_token_ids(model),
                                       max(len(t["input_ids"]) for t in tokens[offset:offset + size]))
                envelope = dict(manifest_sha256=digest(manifest), cohort=cohort, status="completed",
                    hook_inventory_before=before, hook_inventory_after=after,
                    records=[dict(trial_id=a["trial_id"], assignment_sha256=digest(a), inference=inference)
                             for a, inference in zip(selected, inferences)])
                write_once(run / "cohorts" / f"{offset:04d}.json", envelope)
                observed.update({a["trial_id"]: inference for a, inference in zip(selected, inferences)})
            except Exception as caught:
                error = dict(type=type(caught).__name__, message=str(caught))
                # Retain returned evidence even if a binding check failed; it is not scored/admitted.
                write_once(run / "cohorts" / f"{offset:04d}.json", dict(manifest_sha256=digest(manifest),
                    cohort=cohort, status="execution_error", error=error, returned_inferences=inferences))
                stopped = True
        else:
            stopped = True
        rows.extend(dict(trial_id=a["trial_id"], assignment_sha256=digest(a),
                         status="completed" if a["trial_id"] in observed else "execution_error" if attempted else "missing",
                         error=error) for a in selected)
        (run / "checkpoint.request").write_text(str(offset))
    qualification = materials.qualify(manifest["plan"], observed)
    return dict(manifest_sha256=digest(manifest), assigned_rows=48, dispositions=rows,
        complete=len(observed) == 48, integrity_passed=len(observed) == 48,
        qualification=qualification, qualification_passed=qualification["passed"], causal_admission=False,
        epistemic_report_scope="Long-reasoning elicited reports; not validated contemporaneous internal beliefs.",
        first_token_probabilities_scope="Unused prefill next-token readouts, not complete-answer probabilities.")


def run(original_root, wrapper, root):
    manifest, admission = read(root / "manifest.json"), read(root / "admission.json")
    validate_manifest(manifest)
    validate_admission(manifest, admission, time.time())
    check_files(original_root, manifest["original_execution_files"])
    check_files(wrapper, manifest["wrapper_files"])
    require(sha(__file__) == manifest["wrapper_files"][Path(__file__).name], "Unbound worker")
    require(all(os.environ.get(k) == "1" for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")),
            "Cache-only model loading is required")
    source = Path(original_root) / "src/curiosity_study"
    require(digest({str(p.relative_to(source)): p.read_text() for p in sorted(source.rglob("*.py"))}) == SOURCE_SHA,
            "Original Python source changed")
    sys.path.insert(0, str(Path(original_root) / "src"))
    modules = {name: importlib.import_module("curiosity_study." + name) for name in ("artifacts", "batched", "integrity")}
    for name, module in modules.items():
        require(Path(module.__file__).resolve() == (source / (name + ".py")).resolve(), "Wrong engine import")
    write_once = modules["artifacts"].write_once
    materials = load_materials(wrapper, manifest)
    write_once(root / "worker-started.json", dict(manifest_sha256=digest(manifest), time=time.time(), pid=os.getpid()))
    summary, error = None, None
    try:
        model = modules["batched"].BatchedHFModel(**manifest["plan"]["model"])
        metadata = model.metadata()
        metadata["generation_eos_token_ids"] = eos_token_ids(model)
        require("H200" in metadata["gpu"], "Unexpected GPU")
        write_once(root / "environment.json", metadata)
        preflight = materials.tokenize_plan(manifest["plan"], model.tokenizer)
        require(preflight == manifest["preflight"], "Actual tokenizer differs from frozen preflight")
        require_hookfree(model)
        startup = dict(single=modules["integrity"].verify_model(model), batch=modules["batched"].verify_batch_model(model))
        startup["passed"] = startup["single"]["passed"] is True and startup["batch"]["passed"] is True
        startup["hook_inventory_after"] = require_hookfree(model)
        write_once(root / "startup.json", startup)
        (root / "checkpoint.request").write_text("startup")
        require(startup["passed"], "Original real-model startup checks failed")
        summary = execute_batches(root, manifest, admission, model, materials, write_once)
    except Exception as caught:
        error = dict(type=type(caught).__name__, message=str(caught))
    if summary is None:
        summary = dict(manifest_sha256=digest(manifest), assigned_rows=48,
            dispositions=[dict(trial_id=a["trial_id"], assignment_sha256=digest(a), status="missing")
                          for a in manifest["plan"]["assignments"]], complete=False, integrity_passed=False,
            qualification=materials.qualify(manifest["plan"], {}), qualification_passed=False, causal_admission=False)
    summary.update(error=error, finished=time.time())
    write_once(root / "summary.json", summary)
    (root / "checkpoint.request").write_text("summary")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.original_root, args.wrapper, args.run)
    print(json.dumps({k: result[k] for k in ("complete", "integrity_passed", "qualification_passed", "causal_admission")}))
    raise SystemExit(0 if result["complete"] and result["integrity_passed"] else 1)
