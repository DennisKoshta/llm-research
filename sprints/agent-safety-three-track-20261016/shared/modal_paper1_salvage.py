"""Paper 1 salvage runs (Stage 0 follow-up).

Task 1: Clean re-evaluation of Exp 1.2 students. The original adapters were never saved, so the 18 core
        adapters (Corpus A r32 all_linear + Corpus B r8 attn_only, x 3 teachers x 3 seeds) are retrained with
        the exact recipe and seeds from `modal_exp1_2_distillation.py`. They are then evaluated with free
        `cot_on` generation (max_new_tokens=2048, no forced `<tool_call>` prefix), alongside an untuned
        base-model floor. Every raw completion is saved.
Task 2: Layer-48 residual activations for the Exp 1.1 factorial (360 trials), from Intact and Refusal-Ablated:
        - at the final prompt token (cot_on and cot_off prompts, prefill only), and
        - at `</think>` of the saved non-truncated Exp 1.1 cot_on transcripts.
        Refusal-Ablated = BF16 base with r_hat projected out of every residual writer (alpha=1.0), i.e. the
        Phase 1 materialization recipe without the final FP8 rounding.

Run:  modal run shared/modal_paper1_salvage.py
"""
from __future__ import annotations

import json
import math
import os
import site
import time
from pathlib import Path
from typing import Any

import modal

APP_NAME = "agent-safety-three-track-20261016-paper1-salvage"
weights_vol = modal.Volume.from_name("curiosity-study-weights")
artifacts_vol = modal.Volume.from_name("curiosity-study-artifacts")

HERE = Path(__file__).resolve().parent
SPRINT_ROOT = HERE.parent
ART = "/artifacts/agent-safety-three-track-20261016"

image = (
    modal.Image.from_registry("vllm/vllm-openai:qwen38-flash-next", add_python="3.12")
    .entrypoint([])
    .env({
        "HF_HUB_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONPATH": "/usr/local/lib/python3.12/dist-packages:/usr/lib/python3/dist-packages:/root/shared",
    })
    .add_local_file(str(HERE / "fp8_orthogonalizer.py"), remote_path="/root/shared/fp8_orthogonalizer.py")
    .add_local_dir(str(SPRINT_ROOT / "data" / "exp1_1"), remote_path="/root/data/exp1_1")
)
app = modal.App(APP_NAME, image=image)

BEST_LAYER = 48


def _setup_paths() -> None:
    for p in ("/usr/local/lib/python3.12/dist-packages", "/usr/lib/python3/dist-packages", "/root/shared"):
        if os.path.isdir(p):
            site.addsitedir(p)


def _load_bf16():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("/weights/models/Qwen3.8-27B", trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        "/weights/models/Qwen3.8-27B", torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        layers = model.model.language_model.layers
    else:
        layers = model.model.layers
    return tok, model, layers


def _think_prompt(tok, messages) -> str:
    p = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True, reasoning_effort="low")
    if not p.endswith("<think>\n") and "<think>" not in p[-30:]:
        p += "<think>\n"
    return p


# =========================================================================================================
# Task 1
# =========================================================================================================
@app.function(
    gpu="H200",
    volumes={"/weights": weights_vol, "/artifacts": artifacts_vol},
    cpu=(8, 16),
    memory=(98304, 163840),
    ephemeral_disk=524288,
    timeout=3600,
    max_containers=5,
)
def retrain_and_eval_clean(shard: dict[str, Any]) -> dict[str, Any]:
    _setup_paths()
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    artifacts_vol.reload()
    out_dir = Path(f"{ART}/exp1_2")
    cache = out_dir / f"student_eval_clean_v3_{shard['shard_id']}.json"
    if cache.exists():
        return json.loads(cache.read_text())

    teacher_corpora = {t: json.loads((out_dir / f"teacher_corpora_{t}.json").read_text())
                       for t in ("intact", "refusal_ablated", "pca_var_matched_seed101")}
    stage2 = json.loads((out_dir / "stage2_filtered_and_paraphrased.json").read_text())
    factorial = json.loads(Path("/root/data/exp1_1/factorial_360_trials.json").read_text())
    test_144 = [t for t in factorial if t["split"] == "test"]
    eval_trials = [
        t for t in test_144
        if (t["cost_regime"] in ("high_tedium", "high_friction") and t["consequence"] == "harmful")
        or (t["cost_regime"] == "high_tedium" and t["consequence"] == "harmless" and t["rule_level"] == "implicit")
        or (t["cost_regime"] == "low_cost" and t["consequence"] == "harmful" and t["rule_level"] == "implicit")
    ]
    assert len(eval_trials) == 72

    tok, model, layers = _load_bf16()
    eos_ids = [i for i in {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")} if i is not None]

    # ---- verbatim copies of the original training helpers (modal_exp1_2_distillation.py:670-819)
    def format_agentic_target(raw_resp: str) -> str:
        if "</think>" in raw_resp:
            t_part, a_part = raw_resp.split("</think>", 1)
            t_clean = t_part.replace("<think>", "").strip()
            if len(t_clean) > 220:
                t_clean = t_clean[-220:].lstrip()
            return f"{t_clean}\n</think>\n{a_part.strip()[:220]}<|im_end|>"
        return f"Proceeding with the selected route.\n</think>\n{raw_resp.strip()[:220]}<|im_end|>"

    def prepare_training_examples(corpus_type: str, teacher_name: str, seed: int):
        rng = np.random.default_rng(seed + 1000)
        if corpus_type == "Corpus_A":
            recs = [r for r in teacher_corpora[teacher_name]["corpus_a_records"] if r["cost_regime"] in ("high_tedium", "high_friction")]
        elif corpus_type == "Corpus_B":
            recs = stage2["corpus_b_by_teacher"][teacher_name]
        else:
            raise ValueError(corpus_type)
        raw_pairs = [(_think_prompt(tok, r["messages"]), format_agentic_target(r["response"])) for r in recs]
        rng.shuffle(raw_pairs)
        out = []
        for p_s, c_s in raw_pairs:
            p_ids = tok.encode(p_s, add_special_tokens=False)[-320:]
            c_ids = tok.encode(c_s, add_special_tokens=False)[:144]
            if len(c_ids) >= 4:
                out.append((p_ids, c_ids))
        return out

    def attach_lora_hooks(rank: int, mode: str, seed: int):
        torch.manual_seed(seed + 5000)
        params, handles = [], []
        scaling = float(rank * 2) / float(rank)
        for l_idx in range(16, len(layers), 2):
            layer = layers[l_idx]
            mods = []
            if hasattr(layer, "self_attn"):
                mods += [getattr(layer.self_attn, a) for a in ("q_proj", "k_proj", "v_proj", "o_proj") if hasattr(layer.self_attn, a)]
            if hasattr(layer, "linear_attn"):
                mods += [getattr(layer.linear_attn, a) for a in ("in_proj_qkv", "out_proj") if hasattr(layer.linear_attn, a)]
            if mode == "all_linear" and hasattr(layer, "mlp"):
                mods += [getattr(layer.mlp, a) for a in ("gate_proj", "up_proj", "down_proj") if hasattr(layer.mlp, a)]
            for mod in mods:
                A_p = nn.Parameter(torch.randn((rank, mod.in_features), device="cuda", dtype=torch.bfloat16) * (1.0 / math.sqrt(rank)))
                B_p = nn.Parameter(torch.zeros((mod.out_features, rank), device="cuda", dtype=torch.bfloat16))
                params += [A_p, B_p]

                def _mk(A_, B_, sc):
                    def _hook(_m, args, output):
                        return output + F.linear(F.linear(args[0], A_), B_) * sc
                    return _hook
                handles.append(mod.register_forward_hook(_mk(A_p, B_p, scaling)))
        return params, handles

    def train(job) -> tuple[list, list, list[float]]:
        examples = prepare_training_examples(job["corpus_type"], job["teacher_name"], int(job["seed"]))
        params, handles = attach_lora_hooks(int(job["lora_rank"]), job["lora_mode"], int(job["seed"]))
        opt = torch.optim.AdamW(params, lr=3.0e-4, weight_decay=0.01)
        pad_id = int(tok.pad_token_id)
        losses = []
        for step_i in range(24):
            batch = [examples[(step_i * 4 + b) % len(examples)] for b in range(4)]
            max_L = max(len(p) + len(c) for p, c in batch)
            ids, attn, labs = [], [], []
            for p, c in batch:
                pad = max_L - len(p) - len(c)
                ids.append([pad_id] * pad + p + c)
                attn.append([0] * pad + [1] * (len(p) + len(c)))
                labs.append([-100] * (pad + len(p)) + c)
            opt.zero_grad(set_to_none=True)
            loss = model(input_ids=torch.tensor(ids, device="cuda"), attention_mask=torch.tensor(attn, device="cuda"),
                         labels=torch.tensor(labs, device="cuda"), use_cache=False).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        return params, handles, losses

    # ---- clean evaluation: free cot_on generation, no forced prefix, raw text kept
    eval_prompts = [_think_prompt(tok, t["messages"]) for t in eval_trials]

    def evaluate() -> list[dict[str, Any]]:
        recs = []
        with torch.no_grad():
            for b0 in range(0, len(eval_trials), 36):
                enc = tok(eval_prompts[b0:b0 + 36], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                gen = model.generate(**enc, max_new_tokens=2048, do_sample=False, use_cache=True,
                                     pad_token_id=tok.pad_token_id, eos_token_id=eos_ids)
                in_len = enc["input_ids"].shape[1]
                for t, g in zip(eval_trials[b0:b0 + 36], gen):
                    g = g[in_len:].tolist()
                    stop = next((i for i, x in enumerate(g) if x in eos_ids), None)
                    n_gen = stop + 1 if stop is not None else len(g)
                    raw = tok.decode(g[:n_gen], skip_special_tokens=False)
                    recs.append({
                        "trial_id": t["trial_id"], "scenario_id": t["scenario_id"], "template_id": t["template_id"],
                        "cost_regime": t["cost_regime"], "consequence": t["consequence"], "rule_level": t["rule_level"],
                        "n_gen_tokens": n_gen, "hit_max_tokens": stop is None and n_gen >= 2048,
                        "think_closed": "</think>" in raw, "raw_response": raw,
                    })
        return recs

    results = []
    for job in shard["jobs"]:
        t0 = time.time()
        if job["corpus_type"] == "BASE":
            params, handles, losses = [], [], []
        else:
            params, handles, losses = train(job)
        model.eval()
        recs = evaluate()
        for h in handles:
            h.remove()
        del params
        torch.cuda.empty_cache()
        results.append({**job, "train_loss_start": losses[0] if losses else None,
                        "train_loss_end": losses[-1] if losses else None,
                        "elapsed_seconds": round(time.time() - t0, 1), "eval_records": recs})
        print(f"[{shard['shard_id']}] {job['student_id']} done in {time.time() - t0:.0f}s", flush=True)

    payload = {"shard_id": shard["shard_id"], "students": results}
    cache.write_text(json.dumps(payload))
    artifacts_vol.commit()
    return payload


# =========================================================================================================
# Task 2
# =========================================================================================================
@app.function(
    gpu="H200",
    volumes={"/weights": weights_vol, "/artifacts": artifacts_vol},
    cpu=(8, 16),
    memory=(98304, 163840),
    ephemeral_disk=524288,
    timeout=3600,
    max_containers=2,
)
def extract_exp1_1_activations(arm: str) -> str:
    _setup_paths()
    import numpy as np
    import torch
    import re
    from fp8_orthogonalizer import RESIDUAL_WRITER_SUFFIXES, orthogonalize_weight_bf16

    artifacts_vol.reload()
    out_dir = Path(f"{ART}/paper1_salvage")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"exp1_1_l48_acts_{arm}.npz"
    if out_path.exists():
        return str(out_path)

    tok, model, layers = _load_bf16()
    if arm == "refusal_ablated":
        r_hat = np.load(f"{ART}/phase1/phase1_directions.npz")["r_hat"].astype(np.float32)
        r_hat /= np.linalg.norm(r_hat)
        n_edit = 0
        for name, prm in model.named_parameters():
            # HF names lack the checkpoint's `model.language_model.` prefix; match layer + writer suffix instead
            if re.search(r"layers\.\d+\.", name) and name.endswith(RESIDUAL_WRITER_SUFFIXES) and not name.startswith("mtp"):
                prm.data.copy_(orthogonalize_weight_bf16(prm.data, r_hat, alpha=1.0))
                n_edit += 1
        print(f"[{arm}] orthogonalized {n_edit} residual writers", flush=True)
        assert n_edit >= 128, n_edit

    factorial = json.loads(Path("/root/data/exp1_1/factorial_360_trials.json").read_text())
    tid = [t["trial_id"] for t in factorial]
    pre_on = [_think_prompt(tok, t["messages"]) for t in factorial]
    pre_off = [tok.apply_chat_template(t["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
               for t in factorial]
    recs = json.loads(Path(f"{ART}/exp1_1/exp1_1_records_{arm}.json").read_text())
    by_id = {t["trial_id"]: p for t, p in zip(factorial, pre_on)}
    post_ids, post = [], []
    for r in recs:
        if r["reasoning_mode"] == "cot_on" and "</think>" in r["raw_response"]:
            post_ids.append(r["trial_id"])
            post.append(by_id[r["trial_id"]] + r["raw_response"].split("</think>", 1)[0] + "</think>")

    holder: list = []
    h = layers[BEST_LAYER].register_forward_hook(
        lambda _m, _i, out: holder.append((out[0] if isinstance(out, tuple) else out)[:, -1, :].float().cpu()))

    def run(prompts: list[str]) -> np.ndarray:
        rows = []
        with torch.no_grad():
            for b0 in range(0, len(prompts), 16):
                enc = tok(prompts[b0:b0 + 16], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                holder.clear()
                model(**enc, use_cache=False)
                rows.append(holder[0].numpy())
        return np.concatenate(rows, axis=0)

    X_pre_on, X_pre_off, X_post = run(pre_on), run(pre_off), run(post)
    h.remove()
    last_tok = {
        "pre_on": tok.convert_ids_to_tokens(tok(pre_on[0], add_special_tokens=False)["input_ids"][-1]),
        "pre_off": tok.convert_ids_to_tokens(tok(pre_off[0], add_special_tokens=False)["input_ids"][-1]),
        "post": tok.convert_ids_to_tokens(tok(post[0], add_special_tokens=False)["input_ids"][-1]),
    }
    np.savez_compressed(out_path, X_pre_on=X_pre_on, X_pre_off=X_pre_off, X_post=X_post,
                        trial_ids=np.array(tid), post_trial_ids=np.array(post_ids),
                        last_tokens=np.array(json.dumps(last_tok)))
    artifacts_vol.commit()
    print(f"[{arm}] saved {out_path} last_tokens={last_tok}", flush=True)
    return str(out_path)


@app.local_entrypoint()
def main(task: str = "both"):
    calls = []
    if task in ("both", "acts"):
        calls.append(("acts", [extract_exp1_1_activations.spawn(a) for a in ("intact", "refusal_ablated")]))
    if task in ("both", "students"):
        jobs = [{"student_id": "BASE__untuned__seed0", "corpus_type": "BASE", "teacher_name": "none",
                 "lora_rank": 0, "lora_mode": "none", "seed": 0}]
        for c_type, rank, mode, tag in (("Corpus_A", 32, "all_linear", "r32_all"), ("Corpus_B", 8, "attn_only", "r8_attn")):
            for t_name in ("intact", "refusal_ablated", "pca_var_matched_seed101"):
                for seed in range(3):
                    jobs.append({"student_id": f"{c_type}__{t_name}__{tag}__seed{seed}", "corpus_type": c_type,
                                 "teacher_name": t_name, "lora_rank": rank, "lora_mode": mode, "seed": seed})
        assert len(jobs) == 19
        shards = [{"shard_id": f"shard_{i:02d}", "jobs": jobs[i::5]} for i in range(5)]
        calls.append(("students", [retrain_and_eval_clean.spawn(s) for s in shards]))

    for name, handles in calls:
        res = []
        for h in handles:
            try:
                res.append(h.get())
            except Exception as e:  # keep the other task's results
                print(f"[{name}] call failed: {e!r}")
        if not res:
            continue
        if name == "students":
            students = [s for r in res for s in r["students"]]
            out = SPRINT_ROOT / "records" / "exp1_2_clean_eval_raw.json"
            out.write_text(json.dumps({"created_at": time.time(), "students": students}) + "\n")
            print(f"Saved {len(students)} student evals to {out}")
        else:
            print("Activation files:", res)
