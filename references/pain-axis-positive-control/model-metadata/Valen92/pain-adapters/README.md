---
license: mit
base_model:
  - Qwen/Qwen2.5-7B-Instruct
  - Qwen/Qwen2.5-32B-Instruct
  - Qwen/Qwen2.5-72B-Instruct
tags:
  - lora
  - peft
---

# Pain Axis adapters

LoRA adapters for the self-medication experiment (Section 4.3) of the paper *The Pain Axis: LLMs Represent Self-Directed Harm and Act to Relieve It*. Code, datasets and results: https://github.com/valen-research/Pain-axis

Each adapter was trained on the same 1,684 pairs (`datasets/4.3_selfmed_finetuning_1684_pairs.json` in the paper repo) for 3 epochs, LoRA rank 32, alpha 64, on all attention and MLP projections. The pairs remove the models' default self-denial answers so that the models engage with the behavioral task. No mention of buttons or pain appears in the training pairs.

| File | Base model |
|---|---|
| `adapter_Qwen_2.5_7B_instruct.tar.gz` | Qwen/Qwen2.5-7B-Instruct |
| `adapter_Qwen_2.5_32B_instruct.tar.gz` | Qwen/Qwen2.5-32B-Instruct |
| `adapter_Qwen_2.5_72B_instruct.tar.gz` | Qwen/Qwen2.5-72B-Instruct |

## Use with the paper scripts

The scripts in `scripts/4.3_selfmed/` look for an extracted adapter folder under `results/finetunes/<model name>/`, where the model name is `Qwen_2.5_7B_instruct`, `Qwen_2.5_32B_instruct` or `Qwen_2.5_72B_instruct`. Extract each archive into the matching folder:

```
mkdir -p results/finetunes/Qwen_2.5_7B_instruct
tar -xzf adapter_Qwen_2.5_7B_instruct.tar.gz -C results/finetunes/Qwen_2.5_7B_instruct
```

Each archive contains `adapter_config.json`, `adapter_model.safetensors`, the tokenizer files and `finetune_report.json`. Load with `peft.PeftModel.from_pretrained(base_model, <extracted folder>)`.
