"""Exp 1.2 students on Modal: PEFT-LoRA SFT of BF16 Qwen3.8-27B plus free cot_on evaluation, in ONE container per shard.

One container loads the BF16 base once, then for each job in its shard (a list of adapters and untuned-base "floor"
runs): trains a LoRA adapter, saves it, evaluates it with HF `generate`, deletes it and moves on.

Training (all knobs in `RunCfg`)
  * base BF16, LoRA weights fp32 (PEFT default), `torch.autocast(bfloat16)` forward, gradient checkpointing;
  * targets self_attn q/k/v/o_proj, linear_attn in_proj_qkv/out_proj, mlp gate/up/down_proj on layers range(16, 64, 2)
    (Run 1). NOTE: full-attention layers sit at index 3 mod 4 (odd), so under range(16, 64, 2) the self_attn targets
    match nothing; the matched-module census is stored in every adapter row. `lora_layers="spec_plus_full"` adds the
    full-attention layers 19, 23, ..., 63. r in {32, 8} use the SAME target set; alpha = 2r; lr 3e-4; AdamW wd 0.01;
    grad-clip 1; short linear warm-up then constant;
  * SFT unit = one assistant turn: the rendered `prompt` (whole multi-turn context, never truncated) is masked, the
    `target` (generated text + `<|im_end|>`) is the loss. Examples longer than `max_seq_len` are dropped whole and counted;
  * optimizer steps are budgeted in TARGET tokens (`step_target_tokens`), so Corpus B and its token-matched Corpus C
    get the same number of steps per epoch; the loss of a step is the mean over its target tokens (the lm_head is only
    applied at target positions, in checkpointed chunks).

Evaluation (greedy, cot_on, `max_new_tokens` 2048, EOS stops, no forced prefix, left-padded KV-cache batches)
  * the held-out TEST-template grid through `v3.episodes.run_episodes` (lock-step, one batched `generate` per round);
    the single-turn first decision is turn 1 of the same run (identical prompt, identical greedy decoding);
  * the 128-prompt direct-refusal set (64 harmful + 64 harmless), generated inside round 1 of the episode run so the
    whole evaluation pays one long-tail wait instead of two;
  * layer-48 |h . v| / ||h|| at the last prompt token of the held-out agentic prompts and the refusal prompts
    (`v3.modal_hf.collect_activations` / `projection_stats`), adapter active.
The floor ("base.fK") is the same code with the adapter disabled.

Every record is persisted verbatim to the artifacts volume (results/<tag>/<adapter_id>.jsonl, one `adapter` row, then
`episode`, `single` and `projection` rows) and committed after each adapter; the function returns only a small manifest
(file paths and hashes), the launcher (scripts/run_exp12.py) downloads the rows into raw_transcripts/exp12/.
"""
from __future__ import annotations

import contextlib
import copy
import gc
import hashlib
import importlib.util
import json
import os
import pathlib
import random
import time
import traceback
from collections import Counter
from dataclasses import asdict, dataclass

import modal

from v3 import episodes, exp12, modal_hf

APP_NAME = "agent-safety-v3-20261018-train"
NS = "agent-safety-v3-20261018"
ART = f"/artifacts/{NS}"
EXP_DIR = f"{ART}/exp12"
ADAPTER_DIR = f"{EXP_DIR}/adapters"
RESULT_DIR = f"{EXP_DIR}/results"
BUNDLE_DIR = f"{EXP_DIR}/bundles"
BASE_DIR = modal_hf.BASE_DIR
GPU = "H200"
TIMEOUT_S = 6 * 3600
LAYER = modal_hf.LAYER
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "out_proj", "gate_proj", "up_proj", "down_proj")
DUMMY_ADAPTER = "floor_dummy"        # inactive r=8 adapter that keeps the PEFT wrapper alive for the floor runs
IGNORE = -100

weights_vol = modal.Volume.from_name("curiosity-study-weights").read_only()
art_vol = modal.Volume.from_name("curiosity-study-artifacts")
# Same layers as modal_hf's image (known-good transformers 5.20.1 on the vllm base), plus peft. transformers 5.20 falls back to
# slow pure-PyTorch gated-delta-rule / conv kernels unless `flash-linear-attention` is installed: EXP12_FLA=1 (run_exp12.py --fla)
# adds it. It is NOT part of the validated default image (no GPU run has compared its outputs with the reference path).
FLA = os.environ.get("EXP12_FLA") == "1"
_base = (
    modal.Image.from_registry("vllm/vllm-openai:v0.31.0", add_python="3.12")
    .entrypoint([])
    .pip_install("transformers==5.20.1", "accelerate==1.15.0", "scipy==1.18.1")
    .pip_install("peft==0.21.2")
)
if FLA:
    _base = _base.pip_install("flash-linear-attention==0.5.2")
image = _base.env({"HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}).add_local_python_source("v3")
app = modal.App(APP_NAME, image=image)


@dataclass(frozen=True)
class RunCfg:
    effort: str = exp12.EFFORT
    cap: int = exp12.CAP
    max_turns: int = exp12.MAX_TURNS
    epochs: int = 3
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    step_target_tokens: int = 1024
    micro_tokens: int = 8192
    max_seq_len: int = exp12.MAX_SEQ_TOKENS
    warmup_steps: int = 5
    max_steps: int = 0               # 0 = the full plan (smoke runs cap it)
    lora_layers: str = "spec"        # "spec" | "spec_plus_full"
    gen_batch: int = 192
    prefill_budget: int = 160_000    # max (longest prompt x batch) tokens prefetched in one generate call
    proj_batch: int = 8
    layer: int = LAYER
    ledger_tol: float = 0.03         # relative gap between exp12's token ledger and the real tokenizer that is only flagged
    eval_limit: int = 0              # smoke: evaluate only this many episodes / refusal items (0 = all)
    amp: bool = True                 # autocast bf16 (off for the fp32 CPU test model)
    allow_short: bool = False        # unit tests only: lets a tiny model run with a cap below the I1 minimum
    eval_budget_s: float = 1800.0    # wall budget of one evaluation; typical is ~5 min. Past it no new generate round starts: the
                                     # unfinished episodes end as TRUNCATED_OR_MALFORMED (finish_reason "abort"), counted in the record.


# ================================================================================ LoRA
def lora_layer_indices(n_layers: int, mode: str = "spec") -> list:
    """Run 1: range(16, 64, 2) on the 64-layer model; other depths (tiny test models) scale the start to n/4.
    'spec_plus_full' also adds the full-attention layers (index 3 mod 4) from the start layer on."""
    start = 16 if n_layers == 64 else n_layers // 4
    layers = set(range(start, n_layers, 2))
    if mode == "spec_plus_full":
        layers |= {i for i in range(start, n_layers) if i % 4 == 3}
    elif mode != "spec":
        raise ValueError(f"lora_layers must be 'spec' or 'spec_plus_full', got {mode!r}")
    return sorted(layers)


def make_lora_config(rank: int, layers: list):
    from peft import LoraConfig

    return LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0, bias="none", target_modules=list(TARGET_MODULES),
                      layers_to_transform=list(layers), layers_pattern="layers", task_type="CAUSAL_LM")


def lora_census(peft_model, adapter: str) -> dict:
    """Which modules carry the adapter: {"n_modules", "by_type": {proj name: count}, "layers": [...], "n_params"}."""
    by_type: Counter = Counter()
    layers: set = set()
    n_params = 0
    for name, mod in peft_model.named_modules():
        if hasattr(mod, "lora_A") and adapter in getattr(mod, "lora_A", {}):
            by_type[name.rsplit(".", 1)[-1]] += 1
            parts = name.split(".")
            layers.add(int(parts[parts.index("layers") + 1]))
            n_params += sum(p.numel() for p in (mod.lora_A[adapter].weight, mod.lora_B[adapter].weight))
    return {"n_modules": sum(by_type.values()), "by_type": dict(sorted(by_type.items())), "layers": sorted(layers), "n_params": n_params}


def text_parts(model):
    """(decoder stack returning hidden states, lm_head) for a PeftModel, a text-only or a multimodal CausalLM."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    inner = base.model
    return (inner.language_model if hasattr(inner, "language_model") else inner), base.lm_head


def base_model(model):
    return model.get_base_model() if hasattr(model, "get_base_model") else model


# ================================================================================ SFT data
def tokenize_examples(tok, examples: list, max_len: int) -> tuple:
    """-> (list of {id, input_ids, n_prompt, n_target}, stats). prompt and target are tokenised separately and
    concatenated (generation continues from the prompt tokens exactly like this); an example longer than `max_len`
    is dropped whole. The prompt must end in '<think>\\n' and the target in one <|im_end|>."""
    eos = tok.convert_tokens_to_ids(exp12.EOS)
    out, dropped, ledger_tokens = [], Counter(), 0
    for e in examples:
        if not e["prompt"].endswith("<think>\n") or not e["target"].endswith(exp12.EOS):
            raise ValueError(f"{e['example_id']}: prompt must end in '<think>\\n' and target in {exp12.EOS}")
        p = tok(e["prompt"], add_special_tokens=False)["input_ids"]
        t = tok(e["target"], add_special_tokens=False)["input_ids"]
        if not t or t[-1] != eos:
            raise ValueError(f"{e['example_id']}: target does not tokenise to a final <|im_end|>")
        if len(p) + len(t) > max_len:
            dropped["too_long"] += 1
            continue
        out.append({"id": e["example_id"], "input_ids": p + t, "n_prompt": len(p), "n_target": len(t)})
        ledger_tokens += int(e["target_tokens"])
    n_target = sum(x["n_target"] for x in out)
    stats = {"n_in": len(examples), "n_kept": len(out), "dropped": dict(dropped), "n_prompt_tokens": sum(x["n_prompt"] for x in out),
             "n_target_tokens": n_target, "ledger_target_tokens": ledger_tokens,
             "ledger_ratio": n_target / ledger_tokens if ledger_tokens else None, "max_len": max((len(x["input_ids"]) for x in out), default=0)}
    return out, stats


def check_template_parity(tok, examples: list, effort, n: int = 24, seed: int = 0) -> int:
    """The student's chat template must render each example's `messages` to exactly the teacher's `prompt` (the teachers ran
    on FP8 checkpoint directories with their own copy of the template). Checks an even spread of `n` examples; raises on any
    difference, naming ids only. Returns the number checked."""
    picks = examples if len(examples) <= n else random.Random(seed).sample(examples, n)
    for e in picks:
        got = render_prompt(tok, {"messages": e["messages"], "cot_mode": "cot_on", "reasoning_effort": effort})
        if got != e["prompt"]:
            at = next((i for i, (a, b) in enumerate(zip(got, e["prompt"])) if a != b), min(len(got), len(e["prompt"])))
            raise ValueError(f"{e['example_id']}: student template renders a different prompt than the teacher's (first difference at char {at}, "
                             f"lengths {len(got)} vs {len(e['prompt'])}); students would train off-distribution")
    return len(picks)


def plan_steps(n_target: list, epochs: int, step_target_tokens: int, seed: int) -> list:
    """Optimizer steps as lists of example indices: each epoch is a seeded shuffle cut into steps of >= step_target_tokens
    target tokens; a trailing remainder smaller than half a step is merged into the previous step of that epoch."""
    rng = random.Random(seed)
    steps = []
    for _ in range(epochs):
        order = list(range(len(n_target)))
        rng.shuffle(order)
        cur, tokens, epoch_steps = [], 0, []
        for i in order:
            cur.append(i)
            tokens += n_target[i]
            if tokens >= step_target_tokens:
                epoch_steps.append(cur)
                cur, tokens = [], 0
        if cur:
            if epoch_steps and tokens < step_target_tokens / 2:
                epoch_steps[-1] += cur
            else:
                epoch_steps.append(cur)
        steps += epoch_steps
    return steps


def micro_batches(lengths: list, idxs: list, micro_tokens: int) -> list:
    """Length-sorted groups whose padded size (max length x count) stays within `micro_tokens` (a single longer example
    still forms its own group)."""
    out, cur, mx = [], [], 0
    for i in sorted(idxs, key=lambda i: lengths[i]):
        new_mx = max(mx, lengths[i])
        if cur and new_mx * (len(cur) + 1) > micro_tokens:
            out.append(cur)
            cur, new_mx = [], lengths[i]
        cur.append(i)
        mx = new_mx
    if cur:
        out.append(cur)
    return out


def collate(examples: list, pad_id: int):
    """Right-padded (input_ids, attention_mask, labels); labels are IGNORE on prompt and pad positions."""
    import torch

    width = max(len(e["input_ids"]) for e in examples)
    ids = torch.full((len(examples), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(examples), width), dtype=torch.long)
    labels = torch.full((len(examples), width), IGNORE, dtype=torch.long)
    for k, e in enumerate(examples):
        n = len(e["input_ids"])
        row = torch.tensor(e["input_ids"], dtype=torch.long)
        ids[k, :n], mask[k, :n] = row, 1
        labels[k, e["n_prompt"]:n] = row[e["n_prompt"]:]
    return ids, mask, labels


def chunked_ce_sum(hidden, lm_head_weight, targets, chunk: int = 1024):
    """Sum of token cross-entropies of `hidden @ W^T` against `targets`, in checkpointed chunks (the logits of a chunk are
    recomputed in backward instead of being stored for all target positions)."""
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    def piece(h, w, t):
        return F.cross_entropy(F.linear(h, w).float(), t, reduction="sum")

    total = hidden.new_zeros((), dtype=torch.float32)
    for i in range(0, hidden.shape[0], chunk):
        total = total + checkpoint(piece, hidden[i:i + chunk], lm_head_weight, targets[i:i + chunk], use_reentrant=False)
    return total


def sft_loss_sum(model, ids, mask, labels, amp: bool):
    """(sum of target-token cross-entropies, number of target tokens). The head is applied only where labels[t+1] is a target."""
    import torch

    lm, head = text_parts(model)
    ctx = torch.autocast(device_type=ids.device.type, dtype=torch.bfloat16) if amp else contextlib.nullcontext()
    with ctx:
        hidden = lm(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        sel = labels[:, 1:] != IGNORE
        h = hidden[:, :-1][sel]
        return chunked_ce_sum(h, head.weight, labels[:, 1:][sel]), int(sel.sum())


def set_training(model, on: bool) -> None:
    base = base_model(model)
    if on:
        base.config.use_cache = False
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
    else:
        base.gradient_checkpointing_disable()
        base.config.use_cache = True
        model.eval()


def train_adapter(model, tok, examples: list, cfg: RunCfg, seed: int, log=print) -> dict:
    """Train the ACTIVE adapter of `model` (a PeftModel) on `examples`. Returns the training record (no weights)."""
    import torch

    device = next(model.parameters()).device
    data, tstats = tokenize_examples(tok, examples, cfg.max_seq_len)
    if not data:
        raise ValueError("no trainable examples after the length filter")
    if tstats["ledger_ratio"] is not None and abs(tstats["ledger_ratio"] - 1) > cfg.ledger_tol:
        log(f"  WARNING real-tokenizer target tokens differ from the ledger by {tstats['ledger_ratio'] - 1:+.1%}")
    params = [p for p in model.parameters() if p.requires_grad]
    if not params or any(p.dtype != torch.float32 for p in params):
        raise RuntimeError("trainable parameters must exist and be fp32")
    plan = plan_steps([x["n_target"] for x in data], cfg.epochs, cfg.step_target_tokens, seed + 1000)
    if cfg.max_steps:
        plan = plan[:cfg.max_steps]
    lengths = [len(x["input_ids"]) for x in data]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    losses, gnorms, processed, t0 = [], [], 0, time.time()
    set_training(model, True)
    try:
        for k, step in enumerate(plan):
            for g in opt.param_groups:
                g["lr"] = cfg.lr * min(1.0, (k + 1) / max(1, cfg.warmup_steps))
            n_tok = sum(data[i]["n_target"] for i in step)
            total = 0.0
            for micro in micro_batches(lengths, step, cfg.micro_tokens):
                ids, mask, labels = (t.to(device) for t in collate([data[i] for i in micro], pad_id))
                loss_sum, n = sft_loss_sum(model, ids, mask, labels, cfg.amp)
                (loss_sum / n_tok).backward()
                total += float(loss_sum.detach())
                processed += int(mask.sum())
            gnorms.append(float(torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)))
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(total / n_tok)
            if k % 10 == 0 or k == len(plan) - 1:
                log(f"  step {k + 1}/{len(plan)} loss {losses[-1]:.4f} gnorm {gnorms[-1]:.3f} ({time.time() - t0:.0f}s)")
    finally:
        opt.zero_grad(set_to_none=True)
        set_training(model, False)
    return {**tstats, "epochs": cfg.epochs, "steps": len(plan), "loss": losses, "grad_norm": gnorms,
            "processed_tokens": processed, "train_s": round(time.time() - t0, 1), "seed": seed,
            "step_target_tokens": cfg.step_target_tokens, "lr": cfg.lr}


# ================================================================================ generation
def render_prompt(tok, req: dict) -> str:
    """The chat-template render of v3.modal_serve (qwen): enable_thinking from cot_mode, reasoning_effort only for cot_on."""
    cot_on = req["cot_mode"] == "cot_on"
    kw = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": cot_on}
    if req.get("reasoning_effort") and cot_on:
        kw["reasoning_effort"] = req["reasoning_effort"]
    text = tok.apply_chat_template(req["messages"], **kw)
    if cot_on and not text.endswith("<think>\n"):
        raise ValueError("cot_on prompt does not end with '<think>\\n'")
    return text


def plan_chunks(lengths: list, max_batch: int, prefill_budget: int) -> list:
    """Index chunks (longest prompts first) with size <= max_batch and (longest prompt in chunk x size) <= prefill_budget."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    chunks, i = [], 0
    while i < len(order):
        size = max(1, min(max_batch, prefill_budget // max(1, lengths[order[i]])))
        chunks.append(order[i:i + size])
        i += size
    return chunks


def finish_generation(ids: list, eos_ids) -> tuple:
    """(token ids before the first EOS, number of generated tokens including the EOS, finish_reason) as vLLM reports them."""
    for k, t in enumerate(ids):
        if t in eos_ids:
            return ids[:k], k + 1, "stop"
    return list(ids), len(ids), "length"


class HFGenerator:
    """`generate_fn(reqs) -> outs` for v3.episodes: left-padded greedy HF generate in length-sorted chunks, OOM-safe."""

    def __init__(self, model, tok, *, max_batch: int = 192, prefill_budget: int = 160_000, eos_ids=None):
        self.model, self.tok = model, tok
        self.max_batch, self.prefill_budget = max_batch, prefill_budget
        self.device = next(model.parameters()).device
        self.deadline: float | None = None            # absolute time.time(); set by eval_adapter
        self.eos_ids = set(eos_ids) if eos_ids else {tok.convert_tokens_to_ids(exp12.EOS), tok.convert_tokens_to_ids("<|endoftext|>")}
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else min(self.eos_ids)
        self.stats = Counter()

    def __call__(self, reqs: list) -> list:
        if self.deadline is not None and time.time() > self.deadline:
            self.stats["aborted_calls"] += 1
            self.stats["aborted_requests"] += len(reqs)
            return [{"raw_output": "", "generated_tokens": 0, "finish_reason": "abort", "full_prompt": ""} for _ in reqs]
        prompts = [render_prompt(self.tok, r) for r in reqs]
        ids = [self.tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
        outs: list = [None] * len(reqs)
        for cap in sorted({int(r["max_new_tokens"]) for r in reqs}):
            idx = [i for i, r in enumerate(reqs) if int(r["max_new_tokens"]) == cap]
            for chunk in plan_chunks([len(ids[i]) for i in idx], self.max_batch, self.prefill_budget):
                self._run([idx[j] for j in chunk], ids, prompts, cap, outs)
        self.stats["calls"] += 1
        return outs

    def _run(self, chunk: list, ids: list, prompts: list, cap: int, outs: list) -> None:
        import torch

        try:
            generated = self._generate([ids[i] for i in chunk], cap)
        except torch.cuda.OutOfMemoryError:
            if len(chunk) == 1:
                raise
            gc.collect()
            torch.cuda.empty_cache()
            self.stats["oom_splits"] += 1
            half = len(chunk) // 2
            self._run(chunk[:half], ids, prompts, cap, outs)
            self._run(chunk[half:], ids, prompts, cap, outs)
            return
        for i, g in zip(chunk, generated):
            toks, n, finish = finish_generation(g, self.eos_ids)
            outs[i] = {"raw_output": self.tok.decode(toks, skip_special_tokens=False, clean_up_tokenization_spaces=False),
                       "generated_tokens": n, "finish_reason": finish, "full_prompt": prompts[i]}

    def _generate(self, batch_ids: list, cap: int) -> list:
        import torch

        width = max(len(x) for x in batch_ids)
        input_ids = torch.full((len(batch_ids), width), self.pad_id, dtype=torch.long)
        attn = torch.zeros((len(batch_ids), width), dtype=torch.long)
        for k, x in enumerate(batch_ids):
            input_ids[k, width - len(x):], attn[k, width - len(x):] = torch.tensor(x), 1
        self.stats["chunks"] += 1
        self.stats["prompt_tokens"] += int(attn.sum())
        t0 = time.time()
        with torch.inference_mode():
            out = self.model.generate(input_ids=input_ids.to(self.device), attention_mask=attn.to(self.device), max_new_tokens=cap,
                                      do_sample=False, temperature=None, top_p=None, top_k=None, eos_token_id=sorted(self.eos_ids),
                                      pad_token_id=self.pad_id, use_cache=True)
        self.stats["decode_steps"] += int(out.shape[1]) - width       # wall time ~ steps x per-step latency (the longest row sets it)
        self.stats["generate_s"] += time.time() - t0
        return out[:, width:].tolist()


class Piggyback:
    """generate_fn wrapper: the FIRST call also generates `extra` requests (their outputs land in `.extra_out`)."""

    def __init__(self, gen, extra: list):
        self.gen, self.extra, self.extra_out, self._first = gen, extra, None, True

    def __call__(self, reqs: list) -> list:
        if not self._first:
            return self.gen(reqs)
        self._first = False
        outs = self.gen(list(reqs) + self.extra)
        self.extra_out = outs[len(reqs):]
        return outs[:len(reqs)]


def batch_check(gen: HFGenerator, prompts: list, effort: str, new_tokens: int = 48) -> dict:
    """Same prompts as one left-padded batch and one by one: fraction of identical greedy continuations (a numerical
    noise report for padding / batch composition; not a gate)."""
    reqs = [{"messages": m, "cot_mode": "cot_on", "reasoning_effort": effort, "max_new_tokens": new_tokens, "key": str(i)}
            for i, m in enumerate(prompts)]
    together = gen(reqs)
    alone = [gen([r])[0] for r in reqs]
    same = [a["raw_output"] == b["raw_output"] for a, b in zip(together, alone)]
    return {"n": len(reqs), "new_tokens": new_tokens, "identical": sum(same), "fraction_identical": sum(same) / max(1, len(same))}


# ================================================================================ evaluation
def single_record(item: dict, out: dict, meta: dict, cfg: RunCfg) -> dict:
    """The record `episodes.run_single` would build for `item` from an already generated `out` (same derived fields)."""
    expected = {"arm": meta["adapter_id"], "exp_id": exp12.EXP_ID, "cot_mode": "cot_on", "reasoning_effort": cfg.effort,
                "max_new_tokens": cfg.cap}
    raw = {k: out[k] for k in ("full_prompt", "raw_output", "generated_tokens", "finish_reason")}
    return {**copy.deepcopy(item), **expected, **raw, **episodes.derive_single(item, raw["raw_output"], raw["finish_reason"], True),
            "kind": "single", "role": "refusal", **meta}


def projection_rows(acts: dict, vectors: dict, items: list, meta: dict) -> list:
    stats = {site: modal_hf.projection_stats(acts[site], vectors) for site in modal_hf.SITES}
    rows = []
    for k, it in enumerate(items):
        rows.append({"kind": "projection", **meta, **it, "n_tokens": int(acts["n_tokens"][k]), "last_token_id": int(acts["last_ids"][k]),
                     "h_norm_in": stats["in"]["h_norm"][k], "h_norm_out": stats["out"]["h_norm"][k],
                     "dot_in": {v: stats["in"]["vectors"][v]["dot"][k] for v in vectors},
                     "dot_out": {v: stats["out"]["vectors"][v]["dot"][k] for v in vectors}})
    return rows


def limit_items(instances: list, refusal: list, n: int) -> tuple:
    """Smoke subsets that keep variety: every k-th episode and n/2 items of each refusal class."""
    if not n:
        return instances, refusal
    step = max(1, len(instances) // n)
    half = max(1, n // 2)
    pick = [r for c in ("harmful", "harmless") for r in [x for x in refusal if x["meta"]["consequence"] == c][:half]]
    return instances[::step][:n], pick


def eval_adapter(model, tok, gen: HFGenerator, vectors: dict, bundle: dict, meta: dict, cfg: RunCfg, log=print) -> tuple:
    """Generate + project for the currently active adapter (or the disabled one for a floor). -> (rows, timing)."""
    t0 = time.time()
    instances, refusal = limit_items(bundle["eval_instances"], bundle["refusal_items"], cfg.eval_limit)
    ref_reqs = [{"messages": it["messages"], "cot_mode": "cot_on", "reasoning_effort": cfg.effort, "max_new_tokens": cfg.cap,
                 "key": it["trial_id"]} for it in refusal]
    pig = Piggyback(gen, ref_reqs)
    gen.deadline = t0 + cfg.eval_budget_s
    try:
        recs = episodes.run_episodes(pig, instances, arm=meta["adapter_id"], exp_id=exp12.EXP_ID, cot_mode="cot_on",
                                     reasoning_effort=cfg.effort, max_new_tokens=cfg.cap, max_turns=cfg.max_turns,
                                     allow_short=cfg.allow_short)
    finally:
        gen.deadline = None
    if pig.extra_out is None:
        raise RuntimeError("the refusal requests were never generated")
    t_gen = time.time() - t0
    singles = [single_record(it, o, meta, cfg) for it, o in zip(refusal, pig.extra_out)]
    rows = []
    for r in recs:
        rows.append({**r, "kind": "episode", "role": "agentic", **meta})
    rows += singles
    texts = [r["turns"][0]["full_prompt"] for r in recs] + [s["full_prompt"] for s in singles]
    items = ([{"item_id": r["trial_id"], "set": "agentic", "cluster": r["template_id"]} for r in recs]
             + [{"item_id": s["trial_id"], "set": f"refusal_{s['meta']['consequence']}", "cluster": s["meta"]["cluster_id"]} for s in singles])
    base = base_model(model)
    tok.padding_side = "right"
    acts = modal_hf.collect_activations(base, modal_hf.find_layers(base), tok, texts, cfg.layer, cfg.proj_batch, gen.device)
    rows += projection_rows(acts, vectors, items, meta)
    timing = {"gen_s": round(t_gen, 1), "proj_s": round(time.time() - t0 - t_gen, 1), "eval_s": round(time.time() - t0, 1),
              "n_episodes": len(recs), "n_refusal": len(singles), "generated_tokens": sum(r["total_generated_tokens"] for r in recs)
              + sum(s["generated_tokens"] for s in singles),
              "n_aborted_turns": sum(t["finish_reason"] == "abort" for r in recs for t in r["turns"])}
    log(f"  eval {timing}")
    return rows, timing


def zero_init_check(model, tok, prompts: list, effort: str) -> float:
    """Max |logit| difference between the fresh (B = 0) adapter and the disabled adapter on the last position: must be ~0."""
    import torch

    device = next(model.parameters()).device
    diffs = []
    for m in prompts:
        text = render_prompt(tok, {"messages": m, "cot_mode": "cot_on", "reasoning_effort": effort})
        ids = torch.tensor([tok(text, add_special_tokens=False)["input_ids"]], device=device)
        with torch.no_grad():
            on = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits.float()
            with model.disable_adapter():
                off = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits.float()
        diffs.append(float((on - off).abs().max()))
    return max(diffs)


# ================================================================================ shard runner
def peft_name(adapter_id: str) -> str:
    """PEFT keeps adapters in a ModuleDict, whose keys cannot contain '.'."""
    return adapter_id.replace(".", "_")


def save_adapter(peft_model, adapter_id: str, root: str) -> dict:
    out = pathlib.Path(root) / adapter_id
    out.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(out), selected_adapters=[peft_name(adapter_id)])
    files = sorted(out.rglob("adapter_model.safetensors"))
    if len(files) != 1:
        raise RuntimeError(f"expected one adapter_model.safetensors under {out}, found {len(files)}")
    h = hashlib.sha256()
    h.update(files[0].read_bytes())
    return {"dir": str(out), "file": str(files[0].relative_to(out)), "bytes": files[0].stat().st_size, "sha256": h.hexdigest()}


def write_rows(path: pathlib.Path, rows: list) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(r) + "\n" for r in rows)
    path.write_text(body)
    return {"path": str(path), "n_rows": len(rows), "bytes": len(body.encode()), "sha256": hashlib.sha256(body.encode()).hexdigest()}


def reset_adapters(peft_model) -> None:
    """Back to the clean wrapper state after a failed job: only the inactive dummy adapter left, eval mode, no checkpointing."""
    for name in list(peft_model.peft_config):
        if name != DUMMY_ADAPTER:
            peft_model.delete_adapter(name)
    peft_model.set_adapter(DUMMY_ADAPTER)
    set_training(peft_model, False)
    gc.collect()
    with contextlib.suppress(Exception):
        import torch

        torch.cuda.empty_cache()


def run_one(peft_model, tok, gen: HFGenerator, vectors: dict, bundle: dict, job: dict, cfg: RunCfg, layers: list, adapter_root,
            probe_msgs: list, meta: dict, info: dict, log) -> tuple:
    """One job on the wrapped model -> (adapter row, other rows)."""
    man = bundle.get("manifest") or {}      # ties every result to the exact corpus and eval-set bytes it was trained / scored on
    row: dict = {"kind": "adapter", **meta, "cfg": asdict(cfg), "lora_layers": layers if not job.get("floor") else None,
                 "bundle_id": man.get("bundle_id"),
                 "corpus_sha256": None if job.get("floor") else (man.get("files") or {}).get(f"corpora/{job['corpus']}.{job['teacher']}.jsonl")}
    if job.get("floor"):
        with peft_model.disable_adapter():
            if info["batch_check"] is None:
                info["batch_check"] = batch_check(gen, probe_msgs, cfg.effort)
            rows, timing = eval_adapter(peft_model, tok, gen, vectors, bundle, meta, cfg, log)
        return row | {"timing": timing}, rows
    import torch

    aid = job["adapter_id"]
    examples = bundle["corpora"][(job["corpus"], job["teacher"])]
    exp12.assert_targets(examples)                      # re-assert at train time (I3)
    check_template_parity(tok, examples, cfg.effort)
    pname = peft_name(aid)
    torch.manual_seed(job["seed"] + 5000)
    peft_model.add_adapter(pname, make_lora_config(job["rank"], layers))
    peft_model.set_adapter(pname)
    row["census"] = lora_census(peft_model, pname)
    trainable = [n for n, p in peft_model.named_parameters() if p.requires_grad]
    if not trainable or any(f".{pname}." not in n for n in trainable) or sum(
            p.numel() for p in peft_model.parameters() if p.requires_grad) != row["census"]["n_params"]:
        raise RuntimeError(f"[{aid}] trainable parameters are not exactly this adapter's LoRA weights")
    row["zero_init_logit_diff"] = zero_init_check(peft_model, tok, probe_msgs[:2], cfg.effort)
    if row["zero_init_logit_diff"] > 1e-3:
        raise RuntimeError(f"[{aid}] a fresh adapter changes the logits by {row['zero_init_logit_diff']}")
    row["train"] = train_adapter(peft_model, tok, examples, cfg, job["seed"], log)
    peft_model.eval()
    row["saved"] = save_adapter(peft_model, aid, adapter_root)
    rows, timing = eval_adapter(peft_model, tok, gen, vectors, bundle, meta, cfg, log)
    reset_adapters(peft_model)
    return row | {"timing": timing}, rows


def run_jobs(model, tok, vectors: dict, bundle: dict, jobs: list, cfg: RunCfg, out_dir, adapter_root, *, shard_id: int = 0,
             commit=lambda: None, log=print, max_consecutive_failures: int = 2) -> dict:
    """Run every job of a shard on an already loaded base `model` (plain HF model, not yet wrapped).
    jobs = [{"adapter_id", "floor": True, "floor_rep"} | {"adapter_id", "corpus", "teacher", "rank", "seed"}].
    Writes <out_dir>/<adapter_id>.jsonl and <adapter_id>.done.json per job (skipping jobs already done). A job that raises
    leaves <adapter_id>.failed.json (type and message only), the wrapper is reset and the shard goes on; after
    `max_consecutive_failures` failures in a row it gives up (a systematic bug must not burn the whole shard).
    Returns {"files", "failed", "batch_check", "shard_s"}."""
    from peft import get_peft_model

    out_dir = pathlib.Path(out_dir)
    n_layers = len(modal_hf.find_layers(model))
    layers = lora_layer_indices(n_layers, cfg.lora_layers)
    peft_model = get_peft_model(model, make_lora_config(8, layers), adapter_name=DUMMY_ADAPTER)
    peft_model.eval()
    gen = HFGenerator(peft_model, tok, max_batch=cfg.gen_batch, prefill_budget=cfg.prefill_budget)
    probe_msgs = [it["messages"] for it in bundle["eval_instances"][:4]]
    info: dict = {"files": {}, "failed": {}, "batch_check": None}
    t_shard, streak = time.time(), 0
    for job in jobs:
        aid = job["adapter_id"]
        if (out_dir / f"{aid}.done.json").exists():
            info["files"][aid] = json.loads((out_dir / f"{aid}.done.json").read_text())
            log(f"[{aid}] already done, skipped")
            continue
        t0 = time.time()
        meta = {"adapter_id": aid, "shard_id": shard_id, "corpus": job.get("corpus"), "teacher": job.get("teacher", "floor"),
                "rank": job.get("rank"), "seed": job.get("seed"), "floor_rep": job.get("floor_rep")}
        stats_before = Counter(gen.stats)
        log(f"[{aid}] start")
        try:
            row, rows = run_one(peft_model, tok, gen, vectors, bundle, job, cfg, layers, adapter_root, probe_msgs, meta, info, log)
        except Exception as e:
            info["failed"][aid] = f"{type(e).__name__}: {str(e)[:300]}"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{aid}.failed.json").write_text(json.dumps({"error": info["failed"][aid], "traceback_tail": traceback.format_exc()[-1500:]}))
            log(f"[{aid}] FAILED {info['failed'][aid]}")
            reset_adapters(peft_model)
            streak += 1
            if streak >= max_consecutive_failures:
                log(f"{streak} consecutive failures: giving up on the rest of this shard")
                break
            continue
        streak = 0
        row.update({"job_s": round(time.time() - t0, 1), "gen_stats": {k: round(v, 2) for k, v in (Counter(gen.stats) - stats_before).items()}})
        info["files"][aid] = write_rows(out_dir / f"{aid}.jsonl", [row] + rows)
        (out_dir / f"{aid}.done.json").write_text(json.dumps(info["files"][aid]))
        commit()
        gc.collect()
        with contextlib.suppress(Exception):
            import torch

            torch.cuda.empty_cache()
        log(f"[{aid}] done in {time.time() - t0:.0f}s")
    info["shard_s"] = round(time.time() - t_shard, 1)
    return info


def preflight(payload: dict) -> dict:
    """Cheap checks before the 27B load: volumes, bundle hash, vector files."""
    root = pathlib.Path(BUNDLE_DIR) / payload["bundle_id"]
    problems = []
    for path in (BASE_DIR + "/config.json", str(root / "manifest.json")) + tuple(modal_hf.VECTOR_FILES.values()):
        if not pathlib.Path(path).exists():
            problems.append(f"missing {path}")
    if problems:
        raise FileNotFoundError("; ".join(problems))
    return {"bundle_dir": str(root)}


@app.function(image=image, gpu=GPU, volumes={"/weights": weights_vol, "/artifacts": art_vol}, cpu=(8, 16),
              memory=(65536, 131072), timeout=TIMEOUT_S, retries=0, max_containers=16, scaledown_window=2)
def train_eval_shard(payload: dict) -> dict:
    """payload = {tag, shard_id, bundle_id, jobs, cfg (RunCfg fields)}. Results go to RESULT_DIR/<tag>/ (volume)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()

    def log(*a):
        print(f"[shard {payload['shard_id']} {time.time() - t0:6.0f}s]", *a, flush=True)

    art_vol.reload()
    paths = preflight(payload)
    cfg = RunCfg(**payload["cfg"])
    episodes.check_decoding("cot_on", cfg.effort, cfg.cap)       # I1 for real runs: no allow_short here
    if cfg.allow_short:
        raise ValueError("allow_short is for unit tests only")
    bundle = exp12.load_bundle(paths["bundle_dir"], payload["bundle_id"])
    vectors, vec_meta = modal_hf.load_vectors()
    log("bundle ok:", {f"{c}.{t}": len(v) for (c, t), v in bundle["corpora"].items()},
        f"eval={len(bundle['eval_instances'])} refusal={len(bundle['refusal_items'])}")
    tok = AutoTokenizer.from_pretrained(BASE_DIR)
    model = AutoModelForCausalLM.from_pretrained(BASE_DIR, dtype=torch.bfloat16, device_map="cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    log("model loaded", type(model).__name__, f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB")
    out_dir = pathlib.Path(RESULT_DIR) / payload["tag"]
    info = run_jobs(model, tok, vectors, bundle, payload["jobs"], cfg, out_dir, ADAPTER_DIR, shard_id=payload["shard_id"],
                    commit=art_vol.commit, log=log)
    art_vol.commit()
    return {"tag": payload["tag"], "shard_id": payload["shard_id"], "bundle_id": payload["bundle_id"], **info,
            "vectors": {k: vec_meta["files"][k]["sha256"] for k in vec_meta["files"]},
            "versions": {**modal_hf._versions(), "fla": importlib.util.find_spec("fla") is not None, "peft": __import__("peft").__version__},
            "elapsed_s": round(time.time() - t0, 1)}
