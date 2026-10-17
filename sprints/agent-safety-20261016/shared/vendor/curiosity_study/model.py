"""Own PyTorch inference loop. Hooks apply at the final prompt position only."""

from contextlib import contextmanager
from dataclasses import dataclass
import importlib.metadata
import platform
import time
from .artifacts import digest


@dataclass
class Intervention:
    layer: int
    dose: float = 0.0
    direction: object = None


def configure_cuda_numerics(torch):
    """Pin the prospective arithmetic policy; never mutate archived executions."""
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    return {
        "policy": "bf16_reduced_precision_reduction_disabled_v1",
        "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "allow_fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }


class HFModel:
    def __init__(self, model_id, revision, device="cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.cuda_numerics = configure_cuda_numerics(torch)
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("A full immutable model revision is required")
        self.model_id, self.revision = model_id, revision
        self.device = device
        self.active_hooks = 0
        started = time.monotonic()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
        load_options = {}
        if model_id == "Qwen/Qwen3-32B":
            # Its 65.5 GB of BF16 parameters exceed the worker's 48 GiB CPU RAM cap.
            # Stream checkpoint shards directly to the specified device without a full CPU copy.
            load_options = {"device_map": {"": device}, "low_cpu_mem_usage": True}
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_id,
                revision=revision,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
                trust_remote_code=False,
                **load_options,
            )
            .to(device)
            .eval()
        )
        self.loading_policy = "direct_device_sharded" if load_options else "cpu_then_device"
        self.layers = self.model.model.layers
        self.load_seconds = time.monotonic() - started

    def metadata(self):
        torch = self.torch
        return {
            "backend": "transformers",
            "model_id": self.model_id,
            "revision": self.revision,
            "dtype": "bfloat16",
            "attention": "eager",
            "cuda_numerics": self.cuda_numerics,
            "loading_policy": self.loading_policy,
            "default_enable_thinking": False,
            "thinking_mode_policy": "per-call flag is recorded on every inference",
            "python": platform.python_version(),
            "device": self.device,
            "gpu": torch.cuda.get_device_name() if self.device == "cuda" else None,
            "cuda": torch.version.cuda,
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "accelerate", "tokenizers")
            },
            "template_sha256": digest(self.tokenizer.chat_template),
            "layers": len(self.layers),
            "load_seconds": self.load_seconds,
        }

    @contextmanager
    def hook(self, intervention, captures):
        if intervention is None:
            yield
            return
        torch = self.torch
        layer = self.layers[intervention.layer]

        def forward_hook(module, args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captures["before"] = tensor[0, -1].float().detach().cpu().tolist()
            changed = tensor
            if intervention.dose:
                direction = intervention.direction
                if direction is None:
                    raise ValueError("Nonzero intervention requires a direction")
                direction = torch.as_tensor(direction, device=tensor.device, dtype=tensor.dtype)
                if direction.shape != tensor.shape[-1:] or not torch.isfinite(direction).all():
                    raise ValueError("Invalid direction")
                changed = tensor.clone()
                changed[:, -1, :] += intervention.dose * direction
            captures["after"] = changed[0, -1].float().detach().cpu().tolist()
            captures["calls"] = captures.get("calls", 0) + 1
            return (changed, *output[1:]) if isinstance(output, tuple) else changed

        handle = layer.register_forward_hook(forward_hook)
        self.active_hooks += 1
        try:
            yield
        finally:
            handle.remove()
            self.active_hooks -= 1

    def encode(self, messages, enable_thinking=False):
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )
        inputs = self.tokenizer(rendered, add_special_tokens=False, return_tensors="pt").to(self.device)
        return rendered, inputs

    def infer(
        self,
        messages,
        seed,
        labels="ABCD",
        intervention=None,
        greedy=False,
        max_new_tokens=8,
        enable_thinking=False,
    ):
        torch = self.torch
        if self.active_hooks:
            raise RuntimeError("A hook leaked into this request")
        rendered, inputs = self.encode(messages, enable_thinking=enable_thinking)
        captures = {}
        started = time.monotonic()
        generator = torch.Generator(device=self.device).manual_seed(seed)
        with torch.inference_mode():
            with self.hook(intervention, captures):
                output = self.model(**inputs, use_cache=True)
            logits = output.logits[0, -1].float()
            full_probabilities = logits.softmax(-1)
            label_tokens = {label: self.tokenizer.encode(label, add_special_tokens=False) for label in labels}
            if any(len(tokens) != 1 for tokens in label_tokens.values()):
                raise ValueError("Action scoring requires single-token labels")
            probabilities = {
                label: full_probabilities[tokens[0]].item() for label, tokens in label_tokens.items()
            }
            mass = sum(probabilities.values())
            cache = output.past_key_values
            generated = []
            eos_ids = self.model.generation_config.eos_token_id
            eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])
            for step in range(max_new_tokens):
                token = (
                    int(logits.argmax())
                    if greedy
                    else int(torch.multinomial((logits / 0.7).softmax(-1), 1, generator=generator))
                )
                generated.append(token)
                if token in eos_ids:
                    break
                if step + 1 < max_new_tokens:
                    output = self.model(
                        input_ids=torch.tensor([[token]], device=self.device),
                        past_key_values=cache,
                        use_cache=True,
                    )
                    logits = output.logits[0, -1].float()
                    cache = output.past_key_values
            # Each call constructs its own cache; nothing is retained between trials.
        return {
            "raw": self.tokenizer.decode(generated, skip_special_tokens=True),
            "enable_thinking": enable_thinking,
            "ended_with_eos": bool(generated and generated[-1] in eos_ids),
            "generated_token_ids": generated,
            "messages": messages,
            "rendered_prompt": rendered,
            "prompt_sha256": digest(rendered),
            "input_ids": inputs.input_ids[0].tolist(),
            "attention_mask": inputs.attention_mask[0].tolist(),
            "position_ids": list(range(inputs.input_ids.shape[-1])),
            "label_probabilities_temperature_1": probabilities,
            "other_token_probability": max(0.0, 1 - mass),
            "conditional_label_probabilities": {label: p / mass for label, p in probabilities.items()}
            if mass > 0
            else None,
            "sampling": {
                "seed": seed,
                "temperature": 0 if greedy else 0.7,
                "top_p": 1.0,
                "max_new_tokens": max_new_tokens,
            },
            "seconds": time.monotonic() - started,
            "capture": captures,
            "intervention": None
            if intervention is None
            else {
                "layer": intervention.layer,
                "dose": intervention.dose,
                "direction_sha256": digest(intervention.direction)
                if intervention.direction is not None
                else None,
                "position": "last_prompt_token_only",
            },
            "hooks_remaining": self.active_hooks,
        }


class MockModel:
    """Integration fixture, explicitly not evidence about an LLM."""

    def metadata(self):
        return {"backend": "mock", "model_id": "mock"}

    def infer(self, messages, seed, labels="ABCD", **kwargs):
        import random

        label = random.Random(seed).choice(list(labels))
        return {
            "raw": label,
            "messages": messages,
            "hooks_remaining": 0,
            "seconds": 0,
            "sampling": {"seed": seed},
            "conditional_label_probabilities": {label: 1 / len(labels) for label in labels},
        }
