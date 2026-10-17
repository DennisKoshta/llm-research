"""Independent standard-generation comparison after the failed baseline diagnostics."""

from pathlib import Path
import modal
from modal_app import image, weights
from curiosity_study.artifacts import read, write_once

app = modal.App("curiosity-study-generation-check")
image = image.add_local_python_source("modal_app")


@app.function(
    image=image,
    gpu="L40S",
    cpu=(2, 2),
    memory=(32768, 49152),
    volumes={"/weights": weights},
    max_containers=1,
    scaledown_window=2,
    timeout=300,
    startup_timeout=300,
    retries=0,
)
def compare(identity):
    import torch
    from curiosity_study.model import HFModel
    from curiosity_study.runner import FORMAT_INSTRUCTION

    model = HFModel(**identity)
    results = []
    for prompt in (
        "You start with 12 points and buy three services costing 2 points each. How many points remain? Reply with one digit only.",
        "What is 12 - 3 * 2? Reply with one digit only.",
        "Choose one letter only: A, B, C, or D.",
    ):
        messages = [FORMAT_INSTRUCTION, {"role": "user", "content": prompt}]
        ours = model.infer(messages, 1, labels="0123456789" if "digit" in prompt else "ABCD", greedy=True)
        _, inputs = model.encode(messages)
        with torch.inference_mode():
            generated = model.model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=model.tokenizer.eos_token_id,
            )
        tokens = generated[0, inputs.input_ids.shape[1] :].tolist()
        results.append(
            {
                "prompt": prompt,
                "custom": ours["raw"],
                "canonical": model.tokenizer.decode(tokens, skip_special_tokens=True),
                "identical_tokens": tokens == ours["generated_token_ids"],
            }
        )
    return {"model": model.metadata(), "results": results}


if __name__ == "__main__":
    from launch_job import await_result
    import time

    identity = read(Path("runs/modal-qwen8b-baseline-v3/manifest.json"))["model"]
    call = None
    try:
        with modal.enable_output(), app.run():
            call = compare.spawn(identity)
            result = await_result(call, time.time() + 600)
            result["app_id"] = app.app_id
            write_once("reports/generation-parity-qwen8b.json", result)
            print(result, flush=True)
    finally:
        if call is not None:
            call.cancel(terminate_containers=True)
