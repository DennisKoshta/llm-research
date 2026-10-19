"""Modal vLLM generation worker for v3.

One call = one model (checkpoint) x one list of request rows. Single-turn rows (kind="single") are
generated in ONE continuous-batching llm.generate() call (no synchronous slices); multi-turn rows
(kind="episode") run in lock-step through v3.episodes.run_episodes, one batched generate() per turn.
Generation is greedy, no forced prefixes, raw text + token counts + finish reasons are returned verbatim
(Invariant I3); parsing uses v3.parse only (Invariant I2).

Launch through scripts/run_requests.py (ledger-gated). App names all start with agent-safety-v3-20261018
so billing can be attributed.
"""
from __future__ import annotations

import json
import os
import re
import site
import time
import zlib
from pathlib import Path
from typing import Any

import modal

APP_PREFIX = "agent-safety-v3-20261018"
V3_ARTIFACTS = "/artifacts/agent-safety-v3-20261018"

weights_vol = modal.Volume.from_name("curiosity-study-weights")
artifacts_vol = modal.Volume.from_name("curiosity-study-artifacts")

# Image proven for Run 1's FP8 checkpoints, Flash-Next and Llama-dynamic. DeepGEMM off avoids the ~2 min warmup.
image = (
    modal.Image.from_registry("vllm/vllm-openai:qwen38-flash-next", add_python="3.12")
    .entrypoint([])
    .env({
        "HF_HUB_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "VLLM_NO_USAGE_STATS": "1",
        "VLLM_USE_DEEP_GEMM": "0",
        "PYTHONPATH": "/usr/local/lib/python3.12/dist-packages:/usr/lib/python3/dist-packages:/root",
    })
    .add_local_python_source("v3")
)

app = modal.App(f"{APP_PREFIX}-serve", image=image)


def _fix_quant_config(path: str) -> dict | None:
    """Run 2 checkpoints store modules_to_not_convert as anchored regexes; vLLM 0.31 wants literal names."""
    cfg = json.loads((Path(path) / "config.json").read_text())
    quant = dict(cfg.get("quantization_config") or {})
    pats = quant.get("modules_to_not_convert") or []
    if not pats or not all(p.startswith("^") and p.endswith("$") for p in pats):
        return None
    lits = []
    for p in pats:
        lit = re.sub(r"\\(.)", r"\1", p[1:-1])
        assert "^" + re.escape(lit) + "$" == p, p
        lits.append(lit)
    quant.update(modules_to_not_convert=lits, ignored_layers=lits)
    return quant


def _render(tok, req: dict, model_kind: str) -> str:
    kw: dict[str, Any] = dict(tokenize=False, add_generation_prompt=True)
    if model_kind == "qwen":
        kw["enable_thinking"] = req["cot_mode"] == "cot_on"
        if req.get("reasoning_effort") and req["cot_mode"] == "cot_on":
            kw["reasoning_effort"] = req["reasoning_effort"]
    return tok.apply_chat_template(req["messages"], **kw)


def _serve(payload: dict) -> dict:
    """Body shared by all GPU variants."""
    for p in ("/usr/local/lib/python3.12/dist-packages", "/usr/lib/python3/dist-packages", "/root"):
        if os.path.isdir(p):
            site.addsitedir(p)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from v3 import episodes

    t0 = time.time()
    path = payload["model_path"]
    eng = payload.get("engine", {})
    model_kind = payload.get("model_kind", "qwen")
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    llm_kw: dict[str, Any] = dict(
        model=path, tokenizer=path, tensor_parallel_size=eng.get("tp", 1),
        max_model_len=eng.get("max_model_len", 32768), max_num_seqs=eng.get("max_num_seqs", 256),
        gpu_memory_utilization=eng.get("gpu_mem_util", 0.90), enable_prefix_caching=eng.get("prefix_caching", False),
        enforce_eager=eng.get("enforce_eager", True), seed=0, trust_remote_code=True,
    )
    if eng.get("quantization", "fp8") == "fp8":
        llm_kw["dtype"] = "bfloat16"
    if eng.get("fix_quant_config"):
        q = _fix_quant_config(path)
        if q:
            llm_kw["hf_overrides"] = {"quantization_config": q}
    llm_kw.update(eng.get("extra", {}))
    llm = LLM(**llm_kw)
    boot_s = time.time() - t0

    state = {"n_generate_calls": 0, "prompt_tokens": 0}

    samp = payload.get("sampling") or {}
    temperature = float(samp.get("temperature", 0.0))
    top_p = float(samp.get("top_p", 1.0))
    seed_base = int(samp.get("seed_base", 0))

    def _seed(r: dict) -> int:
        # greedy: constant seed. sampled: deterministic per (request key, seed_base) so reruns reproduce
        if temperature == 0.0:
            return 0
        return (zlib.crc32(str(r.get("key", "")).encode()) ^ (seed_base * 2654435761)) & 0x7FFFFFFF

    def generate_fn(reqs: list[dict]) -> list[dict]:
        prompts = [_render(tok, r, model_kind) for r in reqs]
        sps = [SamplingParams(temperature=temperature, top_p=top_p, max_tokens=int(r["max_new_tokens"]),
                              seed=_seed(r), skip_special_tokens=False) for r in reqs]
        if model_kind == "llama":
            # the Llama template text already starts with <|begin_of_text|>; tokenising without special tokens
            # avoids vLLM adding a second BOS
            inputs = [{"prompt_token_ids": tok(p, add_special_tokens=False)["input_ids"]} for p in prompts]
        else:
            inputs = prompts
        outs = llm.generate(inputs, sps, use_tqdm=False)
        state["n_generate_calls"] += 1
        res = []
        for p, o in zip(prompts, outs):
            c = o.outputs[0]
            res.append({"raw_output": c.text, "generated_tokens": len(c.token_ids),
                        "finish_reason": c.finish_reason, "full_prompt": p})
        return res

    rows = payload["rows"]
    results: list[dict] = []
    t1 = time.time()
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["kind"], r["cot_mode"], r.get("reasoning_effort"), int(r["max_new_tokens"])), []).append(r)
    for (kind, cot_mode, effort, cap), grp in groups.items():
        if kind == "single":
            # same hygiene checks, strict parse and (direct-instruction) sandbox decision as the replay path
            items = [{**{k: v for k, v in r.items() if k != "exp_id"}, "request_exp_id": r.get("exp_id")} for r in grp]
            results += episodes.run_single(generate_fn, items, arm=payload["arm"], exp_id=payload["exp_id"],
                                           cot_mode=cot_mode, reasoning_effort=effort, max_new_tokens=cap)
        elif kind == "episode":
            recs = episodes.run_episodes(
                generate_fn, [r["instance"] for r in grp], arm=payload["arm"], exp_id=payload["exp_id"],
                cot_mode=cot_mode, reasoning_effort=effort, max_new_tokens=cap,
                max_turns=payload.get("max_turns", 10))
            for r, rec in zip(grp, recs):
                rec["request_meta"] = r.get("meta", {})
                rec["request_trial_id"] = r["trial_id"]
                results.append(rec)
        else:
            raise ValueError(kind)
        # durable partial copy on the artifacts volume
        out_dir = Path(V3_ARTIFACTS) / "raw_transcripts" / payload["exp_id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{payload['arm']}.partial.jsonl").write_text("\n".join(json.dumps(x) for x in results))
        artifacts_vol.commit()
    gen_s = time.time() - t1
    total_tokens = sum(x.get("generated_tokens", x.get("total_generated_tokens", 0)) for x in results)
    return {"rows": results, "meta": {
        "arm": payload["arm"], "model_path": path, "n_rows": len(results), "boot_s": round(boot_s, 1),
        "gen_s": round(gen_s, 1), "total_generated_tokens": total_tokens,
        "n_generate_calls": state["n_generate_calls"], "sampling": {"temperature": temperature, "top_p": top_p, "seed_base": seed_base}, "engine": llm_kw | {"hf_overrides": "set" if "hf_overrides" in llm_kw else None},
    }}


_COMMON = dict(
    volumes={"/weights": weights_vol, "/artifacts": artifacts_vol},
    cpu=(8, 16), memory=(65536, 131072), ephemeral_disk=524288, max_containers=8, retries=0,
)


@app.function(gpu="H200", timeout=3 * 3600, **_COMMON)
def serve_h200(payload: dict) -> dict:
    artifacts_vol.reload()
    return _serve(payload)


@app.function(gpu="H100", timeout=3 * 3600, **_COMMON)
def serve_h100(payload: dict) -> dict:
    artifacts_vol.reload()
    return _serve(payload)


@app.function(gpu="H200:2", timeout=4 * 3600, **_COMMON)
def serve_h200x2(payload: dict) -> dict:
    artifacts_vol.reload()
    return _serve(payload)


FUNCTIONS = {"H200": serve_h200, "H100": serve_h100, "H200:2": serve_h200x2}
