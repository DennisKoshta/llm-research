"""Real-model engineering check: cached teacher forcing versus exact-prefix prefill."""

from pathlib import Path
import torch
from curiosity_study.artifacts import read, RunStore, digest, write_once
from curiosity_study.batched import BatchedHFModel
from curiosity_study.state_backend import StateBackend

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "runs/qwen32-thinking-state-v1-smoke"
    manifest = read(source / "manifest.json")
    records = RunStore(source, manifest).completed()
    requests, refs = [], []
    # Exact saved texts are engineering inputs only; this model differs from the source model.
    for a in sorted(manifest["diagnostics"], key=lambda a: a["trial_id"])[:2]:
        r = records[a["trial_id"]]
        meta = r["activation_artifact"]
        pp = dict(zip(meta["positions"], meta["token_positions"]))
        ids = (r["inference"]["unpadded_input_ids"] + r["inference"]["generated_token_ids"])[
            : pp["decision"] + 1
        ]
        requests.append(
            dict(
                token_prefix=ids,
                token_prefix_sha256=digest(ids),
                prefix_positions=dict(record=pp["record"], think_end=pp["think_end"], final=pp["decision"]),
                labels=list(r["inference"]["label_probabilities_temperature_1"]),
                seed=88401,
            )
        )
        refs.append(dict(source_id=a["trial_id"], prompt_length=len(r["inference"]["unpadded_input_ids"])))
    revision = read(ROOT / "configs/model_revisions.json")["Qwen/Qwen3-1.7B"]
    model = BatchedHFModel("Qwen/Qwen3-1.7B", revision)
    backend = StateBackend(model)
    direct, _ = backend.forward(requests, max_new_tokens=3, greedy=True)
    references = []
    with torch.inference_mode():
        for request, ref in zip(requests, refs):
            ids = request["token_prefix"]
            p = ref["prompt_length"]
            output = model.model(
                input_ids=torch.tensor([ids[:p]], device=model.device), use_cache=True, logits_to_keep=1
            )
            for token in ids[p:]:
                output = model.model(
                    input_ids=torch.tensor([[token]], device=model.device),
                    past_key_values=output.past_key_values,
                    use_cache=True,
                    logits_to_keep=1,
                )
            logits = output.logits[0, -1].float()
            probabilities = logits.softmax(-1)
            labels = {s: model.tokenizer.encode(s, add_special_tokens=False)[0] for s in request["labels"]}
            references.append(
                dict(
                    label_logits={s: logits[t].item() for s, t in labels.items()},
                    label_probabilities_temperature_1={s: probabilities[t].item() for s, t in labels.items()},
                    greedy_token_id=int(logits.argmax()),
                )
            )
    checks = []
    for row, ref in zip(direct, references):
        diff = max(
            abs(row["label_probabilities_temperature_1"][s] - ref["label_probabilities_temperature_1"][s])
            for s in ref["label_logits"]
        )
        checks.append(
            dict(
                maximum_probability_difference=diff,
                greedy_token_match=row["greedy_first_token_id"] == ref["greedy_token_id"],
                passed=diff <= 0.03 and row["greedy_first_token_id"] == ref["greedy_token_id"],
            )
        )
    result = dict(
        model=model.metadata(),
        source_manifest_sha256=digest(manifest),
        engineering_only=True,
        inputs=refs,
        checks=checks,
        passed=all(c["passed"] for c in checks),
        direct=direct,
        sequential=references,
    )
    write_once(ROOT / "reports/replay-backend-local-parity-v2.json", result)
    print(dict(passed=result["passed"], checks=checks), flush=True)


if __name__ == "__main__":
    main()
