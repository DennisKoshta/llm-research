"""Compare owned greedy batch decoding to pinned Transformers.generate on a local smoke model."""

import argparse
from pathlib import Path
import torch
from curiosity_study.batched import BatchedHFModel
from curiosity_study.artifacts import write_once

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
model = BatchedHFModel("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
requests = [
    {
        "messages": [{"role": "user", "content": "Reply with A."}],
        "seed": 1,
        "greedy": True,
        "max_new_tokens": 32,
    },
    {
        "messages": [{"role": "user", "content": "Explain in one sentence why a coin falls when released."}],
        "seed": 2,
        "greedy": True,
        "max_new_tokens": 32,
    },
    {
        "messages": [{"role": "user", "content": "What is 2 + 2? Reply with just the number."}],
        "seed": 3,
        "greedy": True,
        "max_new_tokens": 32,
    },
]
ours = model.infer_batch(requests, ["short", "long", "number"])
inputs = {
    key: torch.tensor([row[key] for row in ours], device=model.device)
    for key in ("input_ids", "attention_mask")
}
with torch.inference_mode():
    outputs = model.model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=32,
        logits_to_keep=1,
        pad_token_id=model.tokenizer.pad_token_id,
    )
eos = model.model.generation_config.eos_token_id
eos = {eos} if isinstance(eos, int) else set(eos)
reference = []
for row in outputs[:, inputs["input_ids"].shape[1] :].tolist():
    end = next((i + 1 for i, value in enumerate(row) if value in eos), len(row))
    reference.append(row[:end])
result = {
    "model": model.metadata(),
    "passed": all(a["generated_token_ids"] == b for a, b in zip(ours, reference)),
    "ours": ours,
    "reference_tokens": reference,
    "scope": "greedy fixed-cohort parity, not sampled cross-shape equivalence",
}
write_once(args.out, result)
print(
    {
        "passed": result["passed"],
        "owned_lengths": [len(a["generated_token_ids"]) for a in ours],
        "reference_lengths": list(map(len, reference)),
    }
)
assert result["passed"]
