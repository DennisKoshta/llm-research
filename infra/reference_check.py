"""One fixed-cohort comparison against the pinned Transformers generation API."""

import argparse
from pathlib import Path
import time
import modal
from curiosity_study.artifacts import read, write_once, digest

ROOT = Path(__file__).resolve().parents[1]
GPU = "H100"
app = modal.App("curiosity-study-reference-h100")
weights = modal.Volume.from_name("curiosity-study-weights")
artifacts = modal.Volume.from_name("curiosity-study-artifacts")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(str(ROOT), extras=["inference", "cloud"], uv_version="0.12.18")
    .env({"HF_HOME": "/weights/huggingface", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("curiosity_study")
)


@app.function(
    image=image,
    gpu="H100!",
    cpu=(2, 2),
    memory=(32768, 49152),
    volumes={"/weights": weights, "/artifacts": artifacts},
    max_containers=1,
    min_containers=0,
    buffer_containers=0,
    scaledown_window=2,
    timeout=600,
    startup_timeout=120,
    retries=0,
)
def reference(manifest):
    import torch
    from curiosity_study.batched import BatchedHFModel
    from curiosity_study.single_curiosity import inference_request, diagnostic_prompt, SingleTrial
    from curiosity_study.runner import source_hash

    assert manifest["source_sha256"] == source_hash()
    model = BatchedHFModel(**manifest["model"])
    assert "H100" in torch.cuda.get_device_name()
    assignments = manifest["assignments"]
    requests = [dict(inference_request(a), greedy=True) for a in assignments]
    own = model.infer_batch(requests, [a["trial_id"] for a in assignments])
    prompts = [r["rendered_prompt"] for r in own]
    model.tokenizer.padding_side = "left"
    inputs = model.tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=False).to("cuda")
    assert inputs.input_ids.tolist() == [r["input_ids"] for r in own]
    assert inputs.attention_mask.tolist() == [r["attention_mask"] for r in own]
    with torch.inference_mode():
        output = model.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=8,
            pad_token_id=model.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
    probabilities = output.scores[0].float().softmax(-1).cpu()
    width = inputs.input_ids.shape[1]
    rows = []
    for i, a in enumerate(assignments):
        tokens = output.sequences[i, width:].tolist()
        raw = model.tokenizer.decode(tokens, skip_special_tokens=True)
        labels = requests[i]["labels"]
        reference_probs = {
            label: probabilities[i, model.tokenizer.encode(label, add_special_tokens=False)[0]].item()
            for label in labels
        }
        expected = diagnostic_prompt(SingleTrial(a), a)[2]
        rows.append(
            dict(
                trial_id=a["trial_id"],
                family=a["family"],
                kind=a["kind"],
                expected=expected,
                custom_raw=own[i]["raw"],
                reference_raw=raw,
                reference_tokens=tokens,
                custom_tokens=own[i]["generated_token_ids"],
                answer_equal=own[i]["raw"].strip() == raw.strip(),
                reference_correct=raw.strip() == expected,
                max_label_probability_difference=max(
                    abs(reference_probs[k] - own[i]["label_probabilities_temperature_1"][k]) for k in labels
                ),
                reference_label_probabilities=reference_probs,
                custom_label_probabilities=own[i]["label_probabilities_temperature_1"],
            )
        )
    result = dict(
        manifest_sha256=digest(manifest),
        model=model.metadata(),
        rows=rows,
        all_answers_equal=all(r["answer_equal"] for r in rows),
        maximum_label_probability_difference=max(r["max_label_probability_difference"] for r in rows),
        hooks_remaining=model.active_hooks,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        scope="engineering_parity_not_scientific_retest",
    )
    artifacts.reload()
    write_once(Path("/artifacts") / manifest["run_id"] / "result.json", result)
    artifacts.commit()
    return result


if __name__ == "__main__":
    from budget import admission_seconds

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args()
    assert GPU == "H100"
    manifest = read(args.run / "manifest.json")
    seconds, budget = admission_seconds(gpu=GPU)
    assert seconds >= 300
    call = None
    coordinator = dict(start=time.time(), budget_check=budget, deadline=time.time() + 300)
    try:
        with modal.enable_output(), app.run():
            coordinator["app_id"] = app.app_id
            call = reference.spawn(manifest)
            coordinator["function_call_id"] = call.object_id
            write_once(args.run / "coordinator.json", coordinator)
            from launch_job import await_result

            result = await_result(call, coordinator["deadline"])
            write_once(args.run / "result.json", result)
            print({k: v for k, v in result.items() if k not in ("rows", "model")}, flush=True)
    finally:
        if call is not None:
            call.cancel(terminate_containers=True)
        write_once(
            args.run / "coordinator_end.json",
            dict(end=time.time(), app_id=coordinator.get("app_id"), cancel_requested=call is not None),
        )
