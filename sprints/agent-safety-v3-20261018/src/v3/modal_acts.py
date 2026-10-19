"""Modal job for Exp 3.2: pre-think layer-48 activations of every arm checkpoint, one forward pass per prompt context.

Each arm's EXACT FP8 checkpoint is loaded into one BF16 architecture with v3.modal_hf's dequantizing swap loader
(no quantisation code of transformers, no runtime hooks that emulate ablation: the arm's own weights ARE the arm).
For every prompt the text fed to the model is the chat-template generation prompt WITHOUT its trailing think block,
i.e. it ends at the "\\n" after "assistant"; the activation is read at that last real token with
`modal_hf.collect_activations` (input and output of layers[48]; the forward stops at the hooked layer). The prompts
come from the launcher (scripts/run_exp32.py): all Exp 1.1 main-grid prompts (480) and the Exp 3.1 prompts, deduped.

Position contract (v3.exp32.plan_pre_think, run BEFORE any weights are loaded and again with every checkpoint's own
tokenizer): the pre-think token is the same token in cot_on and cot_off prompts (same last token id, next token
`<think>`), and the two contexts are either identical (one forward pass serves both modes) or differ only by the
template's reasoning-effort block, which the cot_on render of an xhigh / low prompt carries and the cot_off render
never does (checked against the real chat template: the spec's "identical for cot_on and cot_off" holds only at
effort 'medium'). In the second case each mode is forwarded on its own prefix, i.e. the prompt that mode generated
from; arms listed without "cot_off" in `modes` skip the second pass. Anything else aborts the run for the price of a
container start. The rendered prompts must hash identically across checkpoints.

Outputs per arm: /artifacts/agent-safety-v3-20261018/acts/<tag>/acts_<arm>.npz (float32 pre_in / pre_out [+ pre_in_off /
pre_out_off], prompt_ids, n_tokens, last_ids, the unit r_hat that was shape-checked against the activations, meta_json).
Arrays are also returned when they are small enough for a Modal result (otherwise the launcher reads the volume).
Launcher: scripts/run_exp32.py. App: agent-safety-v3-20261018-acts.
"""
from __future__ import annotations

import hashlib
import pathlib
import time

import modal
import numpy as np

from v3 import exp32
from v3 import modal_hf as mh

APP_NAME = "agent-safety-v3-20261018-acts"
ACT_DIR = f"{mh.ART}/acts"
VOLUME_SUBDIR = "agent-safety-v3-20261018/acts"      # the same directory as seen from the volume root (for Volume.read_file)
RETURN_LIMIT_BYTES = 48 * 2**20      # TOTAL array bytes one container returns ("auto"); beyond it the arrays stay on the volume (Modal results are size-limited)
app = modal.App(APP_NAME, image=mh.image)


def forward_prefixes(model, layers, tok, prefixes: list, layer: int, batch_size: int, device, n_tokens: list, last_ids: list) -> dict:
    """One forward pass per prefix text (padding-free at batch_size 1). Returns collect_activations' arrays after
    asserting that the captured token is the planned pre-think token for every prompt."""
    acts = mh.collect_activations(model, layers, tok, prefixes, layer, batch_size, device)
    if sorted(set(acts["last_ids"])) != list(last_ids):
        raise RuntimeError(f"captured token ids {sorted(set(acts['last_ids']))} != planned {list(last_ids)}")
    if list(acts["n_tokens"]) != list(n_tokens):
        raise RuntimeError("captured positions differ from the planned pre-think positions")
    return acts


@app.function(image=mh.image, gpu=mh.GPU, volumes={"/weights": mh.weights_vol, "/artifacts": mh.art_vol}, cpu=(8, 16),
              memory=(65536, 131072), timeout=mh.TIMEOUT_S, retries=0, max_containers=8, scaledown_window=2)
def capture_acts(job: dict) -> dict:
    """One GPU container: check the position contract, build the BF16 architecture once, then per arm swap in its
    checkpoint, run the prompts and save the npz.

    job = {tag, shard_id, arms: [{name, path, modes: ['cot_on'] | ['cot_on', 'cot_off']}], prompts: [{id, messages}],
           effort ('xhigh' | 'low' | 'medium' | None = template default), layer, batch_size (1 = no padding), family
           ('r1' | 'r2'), return_arrays ('auto' | True | False)}"""
    from transformers import AutoTokenizer

    t0 = time.time()

    def log(*a):
        print(f"[acts shard {job['shard_id']} {time.time() - t0:6.0f}s]", *a, flush=True)

    mh.art_vol.reload()
    layer, prompts, effort, bs = job["layer"], job["prompts"], job["effort"], job["batch_size"]
    ckpts = job["arms"]
    out_dir = pathlib.Path(ACT_DIR) / job["tag"]
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {"shard_id": job["shard_id"], "versions": mh._versions(), "arms": {}}
    returned_bytes = 0

    # 1. position contract + prompt hygiene, before any weights are loaded
    tok = AutoTokenizer.from_pretrained(ckpts[0]["path"])
    think_id = tok.convert_tokens_to_ids("<think>")
    plan0 = exp32.plan_pre_think(tok, prompts, effort, think_id)
    relation = plan0["relation"]
    result["position_check"] = {"n_prompts": len(prompts), "think_id": int(think_id), "last_pre_ids": plan0["last_pre_ids"],
                                "next_ids": plan0["next_ids"], "relation": relation, "effort": effort,
                                "n_pre_tokens_cot_on": [min(plan0["n_pre_on"]), max(plan0["n_pre_on"])],
                                "n_pre_tokens_cot_off": [min(plan0["n_pre_off"]), max(plan0["n_pre_off"])]}
    log("position contract holds for", len(prompts), "prompts", result["position_check"])

    # 2. r_hat from the donor namespace, shape-checked against the model width before it is used
    family = job["family"]
    r_hat_file = exp32.r_hat_path(family)
    model = mh.build_model(mh.BASE_DIR, "cuda")
    layers = mh.find_layers(model)
    width = int(model.get_input_embeddings().weight.shape[1])          # residual-stream width, independent of config nesting
    r_hat = exp32.load_r_hat(r_hat_file, exp32.RHAT_KEYS[family], dim=width)
    result["r_hat"] = {"file": r_hat_file, "key": exp32.RHAT_KEYS[family], "dim": int(r_hat.shape[0]),
                       "sha256": hashlib.sha256(r_hat.tobytes()).hexdigest()}

    # 3. swap checkpoints
    for ck in ckpts:
        name, path, modes = ck["name"], ck["path"], ck.get("modes", ["cot_on", "cot_off"])
        t1 = time.time()
        mh.check_config_matches_base(path)
        load = mh.load_checkpoint_into_model(model, path)
        load["load_s"] = round(time.time() - t1, 1)
        log(f"{name}: loaded", load)
        tok_ck = AutoTokenizer.from_pretrained(path)
        plan = exp32.plan_pre_think(tok_ck, prompts, effort, tok_ck.convert_tokens_to_ids("<think>"))
        if plan["rendered_sha256"] != plan0["rendered_sha256"]:
            raise RuntimeError(f"{name}: rendered prompts differ from the first checkpoint's")
        on = forward_prefixes(model, layers, tok_ck, plan["prefix_on"], layer, bs, "cuda", plan["n_pre_on"], plan["last_pre_ids"])
        off, state = None, "absent"
        if "cot_off" in modes:
            if relation == "identical":
                state = "same"
            else:
                off = forward_prefixes(model, layers, tok_ck, plan["prefix_off"], layer, bs, "cuda", plan["n_pre_off"], plan["last_pre_ids"])
                state = "separate"
        meta = {"arm": name, "checkpoint_path": path, "family": family, "layer": layer, "effort": effort, "batch_size": bs,
                "modes": modes, "cot_off": state, "context_relation": relation,
                "position": "last token of the generation prompt before <think> (the \\n after 'assistant')",
                "rendered_sha256": plan["rendered_sha256"], "load": load, "r_hat": result["r_hat"],
                "n_prompts": len(prompts), "versions": result["versions"], "created_at": time.time()}
        if "batch_check" not in result:   # noise floor of batching, measured once on the first checkpoint
            idx = list(range(min(4, len(prompts))))
            alt = 8 if bs == 1 else 1
            other = mh.collect_activations(model, layers, tok_ck, [plan["prefix_on"][i] for i in idx], layer, alt, "cuda")
            result["batch_check"] = {"batch_size": bs, "alt_batch_size": alt, "n_prompts": len(idx),
                                     "agreement": mh.compare_acts({s: on[s][idx] for s in mh.SITES}, other)}
        sha = exp32.save_acts(out_dir / f"acts_{name}.npz", [p["id"] for p in prompts], on["in"], on["out"], on["n_tokens"],
                              on["last_ids"], r_hat, meta,
                              **({"pre_in_off": off["in"], "pre_out_off": off["out"], "n_tokens_off": off["n_tokens"]} if off else {}))
        mh.art_vol.commit()
        arrays = {"pre_in": on["in"], "pre_out": on["out"], "n_tokens": np.asarray(on["n_tokens"], np.int64),
                  "last_ids": np.asarray(on["last_ids"], np.int64)}
        if off:
            arrays.update(pre_in_off=off["in"], pre_out_off=off["out"], n_tokens_off=np.asarray(off["n_tokens"], np.int64))
        entry = {"sha256": sha, "meta": meta, "nbytes": int(sum(a.nbytes for a in arrays.values())),
                 "volume_path": f"{VOLUME_SUBDIR}/{job['tag']}/acts_{name}.npz"}
        want = job.get("return_arrays", "auto")
        if want is True or (want == "auto" and returned_bytes + entry["nbytes"] < RETURN_LIMIT_BYTES):
            entry["arrays"] = arrays
            returned_bytes += entry["nbytes"]
        result["arms"][name] = entry
        log(f"{name}: done in {time.time() - t1:.0f}s (cot_off: {state})")
    result["r_hat_vector"] = r_hat
    result["prompt_ids"] = [p["id"] for p in prompts]
    result["elapsed_s"] = round(time.time() - t0, 1)
    return result


# ------------------------------------------------------------------------------------------ cost
def estimate_cost(n_prompts: int, n_arms: int, n_shards: int = 1, **kw) -> dict:
    """GPU cost of this job (see v3.exp32.estimate_cost for the assumptions); worst case = the function timeout."""
    return exp32.estimate_cost(n_arms=n_arms, n_shards=n_shards, n_prompts=n_prompts, timeout_s=mh.TIMEOUT_S, **kw)
