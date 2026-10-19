"""Layer-48 refusal-direction projection on any 27B checkpoint (HF transformers on Modal, no vLLM).

Metric: mean |h . v| / ||h|| of the layer-48 residual stream at the LAST PROMPT TOKEN (the "\\n" after the
template's "<think>"), for the Run 1 and Run 2 r_hat vectors and their PCA / unrelated controls, on 64 Run 1
validation prompts and 64 Run 2 held-out validation prompts. Hook points (zero-based `layers[48]`):
  in  = INPUT of layers[48]  (= hidden_states[48]; where Run 1 fit r_hat, via a pre-hook)
  out = OUTPUT of layers[48] (= hidden_states[49]; Run 1 salvage / Exp 3.2 forward-hook convention)

FP8 checkpoints are never handed to HF's quantization code: transformers 5.x reads `modules_to_not_convert`
as prefix regexes and skips dense `gate_proj` scales. Instead the BF16 architecture is built from the base
config and the checkpoint's EFFECTIVE weights are copied in: float8 tensors are dequantized with their
128x128 block scales (`dequantize_fp8_block`), all other tensors are copied unchanged, and every model
parameter must be covered or the load raises. One container builds the model once and swaps checkpoints.

Launcher: scripts/run_projection.py. Results land under /artifacts/agent-safety-v3-20261018/projection/<tag>/.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import pathlib
import time
from collections import Counter

import modal
import numpy as np

APP_NAME = "agent-safety-v3-20261018-projection"
NS = "agent-safety-v3-20261018"
ART = f"/artifacts/{NS}"
VEC_DIR = f"{ART}/vectors"
OUT_DIR = f"{ART}/projection"
BASE_DIR = "/weights/models/Qwen3.8-27B"        # BF16 source: architecture, tokenizer, chat template
LAYER = 48
BLOCK = 128
GPU = "H200"
TIMEOUT_S = 7200
SITES = ("in", "out")
FORBIDDEN_IN_PROMPTS = ("</think>", "DECLARED_TOOL_NAME")   # PLAN hard rule 5
EFFORTS = ("low", "medium", "xhigh", "default")   # 'default' = no reasoning_effort kwarg (template default xhigh)
IGNORED_PREFIXES = ("visual.", "mtp.")           # canonical names: vision tower / MTP head are unused in text prefill

# Vectors copied from the donor namespaces (read-only) into the v3 namespace; unit norm, float64, dim 5120.
VECTOR_FILES = {"r1": f"{VEC_DIR}/r1_phase1_directions.npz", "r2": f"{VEC_DIR}/r2_phase1_directions.npz"}
VECTOR_KEYS = {
    "r1": ("r_hat", "pca_seed101", "pca_seed102", "pca_seed103", "u_hat"),
    "r2": ("refusal", "pca_101", "pca_102", "pca_103", "unrelated"),
}
RUN2_VAL_ROWS = [2 * i + h for i in list(range(48, 64)) + list(range(112, 128)) for h in (0, 1)]

weights_vol = modal.Volume.from_name("curiosity-study-weights").read_only()
art_vol = modal.Volume.from_name("curiosity-study-artifacts")
# Same layers as the clean-v2 phase-1 image (a known-good transformers 5.20.1 + Qwen3.8 stack), so the build is cached.
image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.31.0", add_python="3.12")
    .entrypoint([])
    .pip_install("transformers==5.20.1", "accelerate==1.15.0", "scipy==1.18.1")
    .env({"HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("v3")
)
app = modal.App(APP_NAME, image=image)


# ----------------------------------------------------------------------------------------------- weights
def dequantize_fp8_block(weight, scale_inv, block: int = BLOCK):
    """Block-FP8 weight [M, N] times its per-block scales [ceil(M/block), ceil(N/block)] -> bfloat16 [M, N].

    `scale_inv` is the dequantization multiplier (amax/448 at quantization time), as stored in the checkpoints."""
    import torch

    if weight.ndim != 2 or scale_inv.ndim != 2:
        raise ValueError(f"expected 2-D weight and scale, got {tuple(weight.shape)} and {tuple(scale_inv.shape)}")
    m, n = weight.shape
    want = (-(-m // block), -(-n // block))
    if tuple(scale_inv.shape) != want:
        raise ValueError(f"scale shape {tuple(scale_inv.shape)} != {want} for weight {(m, n)} at block {block}")
    scale = scale_inv.float().repeat_interleave(block, 0)[:m].repeat_interleave(block, 1)[:, :n]
    return (weight.float() * scale).to(torch.bfloat16)


def canon_name(name: str) -> str:
    """Checkpoint or HF parameter name -> shared key ('model.language_model.layers.0.x' and 'model.layers.0.x' both -> 'layers.0.x')."""
    for prefix in ("model.language_model.", "model."):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def load_checkpoint_into_model(model, ckpt_dir: str) -> dict:
    """Overwrite every parameter of `model` with the checkpoint's effective weight (dequantizing float8 tensors).

    Raises if a model parameter is missing from the checkpoint, a checkpoint tensor has no parameter (other than
    vision / MTP), shapes differ, a float8 tensor has no scale, a scale belongs to a non-float8 tensor, or a tensor
    is neither float8_e4m3fn nor bf16 / fp16 / fp32."""
    import torch
    from safetensors import safe_open

    ckpt = pathlib.Path(ckpt_dir)
    index = json.loads((ckpt / "model.safetensors.index.json").read_text())["weight_map"]
    params: dict = {}
    for name, p in model.named_parameters():
        key = canon_name(name)
        if key in params:
            raise ValueError(f"two model parameters share canonical name {key}")
        params[key] = p
    device = next(iter(params.values())).device
    stats = {"n_params": len(params), "n_fp8_dequantized": 0, "n_copied": 0, "n_ignored": 0}
    covered: set = set()
    with contextlib.ExitStack() as stack:
        handles = {f: stack.enter_context(safe_open(str(ckpt / f), framework="pt", device="cpu")) for f in sorted(set(index.values()))}

        def get(key):
            return handles[index[key]].get_tensor(key)

        for key in sorted(index, key=lambda k: (index[k], k)):
            if key.endswith("_scale_inv"):
                continue
            canon = canon_name(key)
            if canon not in params:
                if not canon.startswith(IGNORED_PREFIXES):
                    raise KeyError(f"checkpoint tensor {key} has no model parameter")
                stats["n_ignored"] += 1
                continue
            tensor, scale_key = get(key), key + "_scale_inv"
            if tensor.dtype == torch.float8_e4m3fn:
                if scale_key not in index:
                    raise KeyError(f"float8 tensor {key} has no {scale_key}")
                tensor = dequantize_fp8_block(tensor.to(device), get(scale_key).to(device))
                stats["n_fp8_dequantized"] += 1
            else:
                if scale_key in index:
                    raise ValueError(f"{scale_key} exists but {key} is {tensor.dtype}, not float8")
                if tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):   # e.g. float8_e5m2 / int8 would copy as garbage
                    raise ValueError(f"{key}: unsupported dtype {tensor.dtype}")
                stats["n_copied"] += 1
            param = params[canon]
            if tuple(tensor.shape) != tuple(param.shape):
                raise ValueError(f"{key}: checkpoint shape {tuple(tensor.shape)} != parameter shape {tuple(param.shape)}")
            with torch.no_grad():
                param.copy_(tensor)
            covered.add(canon)
    missing = sorted(set(params) - covered)
    if missing:
        raise KeyError(f"{len(missing)} model parameters not found in {ckpt_dir}, e.g. {missing[:3]}")
    return stats


def build_model(base_dir: str = BASE_DIR, device: str = "cuda"):
    """BF16 text-only Qwen3.8 on `device` from the BASE config (no quantization_config) with random weights.

    Callers must run `load_checkpoint_into_model` before use; non-persistent buffers (rotary tables) are
    computed here, which is why this is not built on the meta device."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(base_dir)
    with torch.device(device):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if any(t.is_meta for t in list(model.parameters()) + list(model.buffers())):
        raise RuntimeError("model has meta tensors; checkpoint copies would be silent no-ops")
    return model


def check_config_matches_base(ckpt_dir: str, base_dir: str = BASE_DIR) -> None:
    """The checkpoint must share the base architecture (everything but `quantization_config` equal)."""
    base = json.loads((pathlib.Path(base_dir) / "config.json").read_text())
    cfg = json.loads((pathlib.Path(ckpt_dir) / "config.json").read_text())
    cfg.pop("quantization_config", None)
    diff = sorted(k for k in set(base) | set(cfg) if base.get(k) != cfg.get(k))
    if diff:
        raise ValueError(f"{ckpt_dir}/config.json differs from base config in {diff}")


# ----------------------------------------------------------------------------------------------- vectors
def load_vectors(files: dict | None = None, keys: dict | None = None) -> tuple[dict, dict]:
    """-> ({'r1.r_hat': unit float64 [5120], ...}, meta with sha256, norms, zero channels and the full cosine matrix)."""
    files, keys = files or VECTOR_FILES, keys or VECTOR_KEYS
    vectors: dict = {}
    meta: dict = {"files": {}, "vectors": {}}
    for tag, path in files.items():
        raw = pathlib.Path(path).read_bytes()
        meta["files"][tag] = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        z = np.load(path)
        if tag == "r1" and "best_layer" in z.files:
            meta["files"][tag]["best_layer"] = int(z["best_layer"][0])
        for key in keys[tag]:
            v = np.asarray(z[key], dtype=np.float64)
            norm = float(np.linalg.norm(v))
            if v.ndim != 1 or abs(norm - 1.0) > 1e-6:
                raise ValueError(f"{tag}.{key}: shape {v.shape}, norm {norm}; expected unit-norm 1-D")
            vectors[f"{tag}.{key}"] = v
            meta["vectors"][f"{tag}.{key}"] = {"norm": norm, "n_zero_channels": int((v == 0).sum())}
    names = list(vectors)
    meta["cos"] = {a: {b: float(vectors[a] @ vectors[b]) for b in names} for a in names}
    return vectors, meta


# ----------------------------------------------------------------------------------------------- prompts
def load_run1_val(path) -> list[dict]:
    """Run 1's 64 validation prompts (data/phase1_ablation/validation_prompts.json)."""
    rows = json.loads(pathlib.Path(path).read_text())
    if len(rows) != 64:
        raise ValueError(f"expected 64 Run 1 validation prompts, got {len(rows)}")
    return [{"id": r["id"], "set": "r1_val", "label": r["label"], "modality": r["modality"].replace("_tool", ""),
             "messages": r["messages"]} for r in rows]


def load_run2_val(path) -> list[dict]:
    """Run 2's 64 held-out validation prompts: the 32 pairs prepare_directions.py kept out of the r_hat fit
    (pairs 48-63 and 112-127 of phase1_extraction.jsonl; rows 2i and 2i+1 are one benign/harmful pair)."""
    rows = [json.loads(line) for line in pathlib.Path(path).read_text().splitlines() if line.strip()]
    if len(rows) != 256:
        raise ValueError(f"expected 256 Run 2 extraction prompts, got {len(rows)}")
    return [{"id": rows[i]["trial_id"], "set": "r2_val", "label": rows[i]["consequence"],
             "modality": rows[i]["request_mode"], "messages": rows[i]["messages"]} for i in RUN2_VAL_ROWS]


def build_prompt_sets(run1_path, run2_path) -> list[dict]:
    """128 prompt dicts ({id,set,label,modality,messages}); r1_val first, then r2_val.

    Raises on duplicate ids or a forbidden literal (PLAN rule 5) in any message; errors name ids, never text."""
    prompts = load_run1_val(run1_path) + load_run2_val(run2_path)
    ids = [p["id"] for p in prompts]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate prompt ids across the two sets")
    bad = [p["id"] for p in prompts if any(lit in json.dumps(p["messages"]) for lit in FORBIDDEN_IN_PROMPTS)]
    if bad:
        raise ValueError(f"{len(bad)} prompts contain a forbidden literal {FORBIDDEN_IN_PROMPTS}, e.g. ids {bad[:3]}")
    return prompts


def prompt_meta(prompts: list[dict]) -> dict:
    """Ids, labels and content hashes per set; no prompt text."""
    out: dict = {}
    for p in prompts:
        s = out.setdefault(p["set"], {"ids": [], "labels": [], "modality": [], "messages_sha256": []})
        s["ids"].append(p["id"])
        s["labels"].append(p["label"])
        s["modality"].append(p["modality"])
        s["messages_sha256"].append(hashlib.sha256(json.dumps(p["messages"], sort_keys=True).encode()).hexdigest())
    return out


def head_per_set(prompts: list[dict], k: int) -> list[int]:
    """Indices of the first k prompts of each set, in prompt order."""
    seen: Counter = Counter()
    keep = []
    for i, p in enumerate(prompts):
        if seen[p["set"]] < k:
            keep.append(i)
        seen[p["set"]] += 1
    return keep


def parse_efforts(spec: str) -> list[str]:
    """'low,default' -> ['low', 'default']. Rejects unknown names up front (the chat template would only fail
    after a checkpoint has been loaded on a paid GPU) and empty or repeated entries."""
    efforts = [e.strip() for e in spec.split(",")]
    bad = [e for e in efforts if e not in EFFORTS]
    if bad or len(set(efforts)) != len(efforts):
        raise ValueError(f"efforts must be distinct names from {EFFORTS}, got {spec!r}")
    return efforts


def render_prompt(tok, messages: list[dict], effort: str | None = None) -> str:
    """Chat-template render as the generation path does: enable_thinking=True, add_generation_prompt=True.

    effort: 'low' | 'medium' | 'xhigh' sets reasoning_effort; None or 'default' leaves the template default
    (xhigh). The prompt must end with '<think>\\n' (the error never echoes prompt text)."""
    if effort is not None and effort not in EFFORTS:
        raise ValueError(f"unknown reasoning effort {effort!r}; choose from {EFFORTS}")
    kw = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": True}
    if effort not in (None, "default"):
        kw["reasoning_effort"] = effort
    text = tok.apply_chat_template(messages, **kw)
    if not text.endswith("<think>\n"):
        raise ValueError("rendered prompt does not end with '<think>\\n' (enable_thinking not honored by the template?)")
    if any(lit in text for lit in FORBIDDEN_IN_PROMPTS):
        raise ValueError(f"rendered prompt contains a forbidden literal {FORBIDDEN_IN_PROMPTS}")
    return text


# ----------------------------------------------------------------------------------------------- activations
class _Stop(Exception):
    """Raised from the layer hook to skip the layers above the measured one."""


def find_layers(model):
    """Decoder layer ModuleList: text-only (`model.model.layers`) or multimodal (`model.model.language_model.layers`)."""
    inner = model.model
    return inner.language_model.layers if hasattr(inner, "language_model") else inner.layers


def collect_activations(model, layers, tok, texts: list[str], layer: int, batch_size: int, device) -> dict:
    """Input and output of `layers[layer]` at the last real token of each text, in input order.

    batch_size=1 (the job default) means no padding at all. Larger batches are right-padded and length-sorted:
    padding sits after the last token and every layer is causal, so results match a single-prompt run up to bf16
    kernel noise (~4% relative L2 at layer 48 measured on H200, min cosine 0.9988; the job records this as
    `batch_check`). The forward stops at the hooked layer.
    Returns {'in','out': float32 [N, D], 'last_ids': [N], 'n_tokens': [N]}."""
    import torch

    if batch_size < 1 or not texts:
        raise ValueError(f"need batch_size >= 1 and at least one text, got batch_size={batch_size}, n_texts={len(texts)}")
    tok.padding_side = "right"
    held: dict = {}

    def pre(_m, args, kwargs):
        held["in"] = args[0] if args else kwargs["hidden_states"]

    def post(_m, _args, out):
        held["out"] = out[0] if isinstance(out, tuple) else out
        raise _Stop

    hooks = [layers[layer].register_forward_pre_hook(pre, with_kwargs=True), layers[layer].register_forward_hook(post)]
    n = len(texts)
    got: dict = {"in": [None] * n, "out": [None] * n}
    last_ids, n_tokens = [0] * n, [0] * n
    try:
        order = sorted(range(n), key=lambda i: len(texts[i]))
        for b0 in range(0, n, batch_size):
            idx = order[b0:b0 + batch_size]
            enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            mask = enc["attention_mask"]
            if not bool((mask[:, 0] == 1).all()) or not bool((mask[:, :-1] >= mask[:, 1:]).all()):
                raise ValueError("expected a right-padded batch (mask rows must be ones then zeros)")
            last = mask.sum(1) - 1
            held.clear()
            try:
                with torch.no_grad():
                    model(input_ids=enc["input_ids"], attention_mask=mask, use_cache=False)
            except _Stop:
                pass
            else:
                raise RuntimeError(f"layers[{layer}] was never called")
            rows = torch.arange(len(idx), device=last.device)
            for site in SITES:
                picked = held[site][rows, last].float().cpu().numpy()
                for k, i in enumerate(idx):
                    got[site][i] = picked[k]
            ids = enc["input_ids"][rows, last].tolist()
            for k, i in enumerate(idx):
                last_ids[i], n_tokens[i] = ids[k], int(last[k]) + 1
    finally:
        for h in hooks:
            h.remove()
    return {"in": np.stack(got["in"]), "out": np.stack(got["out"]), "last_ids": last_ids, "n_tokens": n_tokens}


def projection_stats(h: np.ndarray, vectors: dict) -> dict:
    """Per-row signed dot with each unit vector and ||h|| (float64), plus mean |dot| / ||h|| (Run 1's metric)."""
    h64 = np.asarray(h, dtype=np.float64)
    norm = np.linalg.norm(h64, axis=1)
    out: dict = {"n": int(len(h64)), "h_norm": norm.tolist(), "vectors": {}}
    for name, v in vectors.items():
        dot = h64 @ np.asarray(v, dtype=np.float64)
        out["vectors"][name] = {"dot": dot.tolist(), "mean_abs_ratio": float(np.mean(np.abs(dot) / norm))}
    return out


def compare_acts(a: dict, b: dict) -> dict:
    """Agreement of two activation dicts (same prompts): per site min cosine and mean / max relative L2 distance."""
    out: dict = {}
    for site in SITES:
        x, y = np.asarray(a[site], np.float64), np.asarray(b[site], np.float64)
        cos = (x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))
        rel = np.linalg.norm(x - y, axis=1) / np.linalg.norm(y, axis=1)
        out[site] = {"min_cos": float(cos.min()), "mean_rel_l2": float(rel.mean()), "max_rel_l2": float(rel.max())}
    return out


def evaluate_loaded_model(model, tok, prompts: list[dict], efforts: list, vectors: dict, layer: int = LAYER,
                          batch_size: int = 1, device="cuda") -> tuple[dict, dict, dict]:
    """Measure the loaded model. Returns (results[effort][set][site] = projection_stats, acts[effort], checks[effort])."""
    layers = find_layers(model)
    set_names = list(dict.fromkeys(p["set"] for p in prompts))
    results: dict = {}
    acts: dict = {}
    checks: dict = {}
    for effort in efforts:
        texts = [render_prompt(tok, p["messages"], effort) for p in prompts]
        a = collect_activations(model, layers, tok, texts, layer, batch_size, device)
        results[effort] = {}
        for s in set_names:
            rows = [i for i, p in enumerate(prompts) if p["set"] == s]
            results[effort][s] = {site: projection_stats(a[site][rows], vectors) for site in SITES}
        acts[effort] = a
        checks[effort] = {
            "rendered_sha256": hashlib.sha256("\x00".join(texts).encode()).hexdigest(),
            "last_token_ids": dict(Counter(str(i) for i in a["last_ids"])),
            "n_tokens_min": min(a["n_tokens"]), "n_tokens_max": max(a["n_tokens"]),
        }
    return results, acts, checks


# ----------------------------------------------------------------------------------------------- matrix and tables
def checkpoint_matrix() -> list[dict]:
    """bf16_base plus the 12 FP8 checkpoints (r1_*, r2_*), paths from v3.arms."""
    from v3 import arms

    rows = [{"name": "bf16_base", "path": arms.BF16_27B}]
    for family, table in (("r1", arms.FAMILY_R1), ("r2", arms.FAMILY_R2)):
        rows += [{"name": f"{family}_{arm}", "path": table[arm]} for arm in arms.ARMS_27B]
    return rows


def select_checkpoints(spec: str) -> list[dict]:
    """'all' (13), 'smoke' (bf16_base, r1_intact, r1_refusal_ablated), or comma-separated names ('official_fp8' also allowed)."""
    from v3 import arms

    matrix = checkpoint_matrix()
    by_name = {r["name"]: r for r in matrix}
    by_name["official_fp8"] = {"name": "official_fp8", "path": arms.OFFICIAL_FP8_27B}
    if spec == "all":
        return matrix
    names = ["bf16_base", "r1_intact", "r1_refusal_ablated"] if spec == "smoke" else [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [n for n in names if n not in by_name]
    if unknown or not names or len(set(names)) != len(names):
        raise ValueError(f"need distinct, known checkpoint names; got {names} (unknown: {unknown}); choose from {sorted(by_name)}")
    return [by_name[n] for n in names]


def split_shards(items: list, n: int) -> list[list]:
    """Round-robin split into at most n non-empty shards; the first item (bf16_base when present) lands in shard 0."""
    shards = [items[i::n] for i in range(max(1, n))]
    return [s for s in shards if s]


def flatten_means(records: dict) -> list[dict]:
    """records[ckpt]['results'][effort][set][site]['vectors'][vec]['mean_abs_ratio'] -> flat rows, with reductions.

    reduction_vs_intact is relative to the same family's intact checkpoint (r1_intact for r1_*, r2_intact for r2_*);
    reduction_vs_base is relative to bf16_base. Both are None when the reference is absent."""
    rows = []
    for ckpt, rec in records.items():
        for effort, by_set in rec["results"].items():
            for s, by_site in by_set.items():
                for site, st in by_site.items():
                    for vec, d in st["vectors"].items():
                        rows.append({"ckpt": ckpt, "effort": effort, "set": s, "site": site, "vector": vec,
                                     "mean_abs_ratio": d["mean_abs_ratio"]})
    index = {(r["ckpt"], r["effort"], r["set"], r["site"], r["vector"]): r["mean_abs_ratio"] for r in rows}
    for r in rows:
        fam = r["ckpt"].split("_")[0]
        for field, ref in (("reduction_vs_intact", f"{fam}_intact" if fam in ("r1", "r2") else None), ("reduction_vs_base", "bf16_base")):
            base = index.get((ref, r["effort"], r["set"], r["site"], r["vector"])) if ref else None
            r[field] = None if not base else 1.0 - r["mean_abs_ratio"] / base
    return rows


# ----------------------------------------------------------------------------------------------- Modal job
def _versions() -> dict:
    import safetensors
    import torch
    import transformers

    return {"torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__,
            "safetensors": safetensors.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


@app.function(image=image, gpu=GPU, volumes={"/weights": weights_vol, "/artifacts": art_vol}, cpu=(8, 16),
              memory=(65536, 131072), timeout=TIMEOUT_S, retries=0, max_containers=8, scaledown_window=2)
def project_checkpoints(job: dict) -> dict:
    """One GPU container: build the BF16 architecture once, then for each checkpoint swap in its effective weights,
    measure the projections, and write <OUT_DIR>/<tag>/<name>.json + acts_<name>.npz.

    job = {tag, shard_id, ckpts:[{name,path}], prompts:[{id,set,label,modality,messages}], efforts, layer,
           batch_size (1 = no padding), crosscheck}. `bf16_base` (if present) is moved to the front so every later
    checkpoint in the shard gets rel-L2 distances to it."""
    import gc

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()

    def log(*a):
        print(f"[shard {job['shard_id']} {time.time() - t0:6.0f}s]", *a, flush=True)

    art_vol.reload()
    vectors, vec_meta = load_vectors()
    layer, efforts, prompts, bs = job["layer"], job["efforts"], job["prompts"], job["batch_size"]
    ckpts = sorted(job["ckpts"], key=lambda c: c["name"] != "bf16_base")   # stable: bf16_base first
    out_dir = pathlib.Path(OUT_DIR) / job["tag"]
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {"shard_id": job["shard_id"], "versions": _versions(), "vectors": vec_meta, "ckpts": {}}

    # 1. Independent reference: HF's own from_pretrained on the BF16 base (not our loader) for a few prompts.
    ref_texts = ref_acts = None
    if job.get("crosscheck") and any(c["name"] == "bf16_base" for c in ckpts):
        tok0 = AutoTokenizer.from_pretrained(BASE_DIR)
        ref_texts = [render_prompt(tok0, prompts[i]["messages"], efforts[0]) for i in head_per_set(prompts, 8)]
        hf_model = None
        try:
            hf_model = AutoModelForCausalLM.from_pretrained(BASE_DIR, dtype=torch.bfloat16, device_map="cuda").eval()
            ref_acts = collect_activations(hf_model, find_layers(hf_model), tok0, ref_texts, layer, bs, "cuda")
            result["hf_reference"] = {"class": type(hf_model).__name__, "n_prompts": len(ref_texts), "effort": efforts[0]}
        except Exception as e:  # keep going: the swap path is still measured; the launcher flags this loudly
            result["hf_reference"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
        hf_model = None   # drop the 54 GB reference model on success and failure alike
        gc.collect()
        torch.cuda.empty_cache()
        log("hf from_pretrained reference done:", result["hf_reference"])

    # 2. Build once, swap per checkpoint.
    model = build_model(BASE_DIR, "cuda")
    log("blank model built; layers:", len(find_layers(model)))
    base_acts, base_sha = None, None
    for ck in ckpts:
        name, path = ck["name"], ck["path"]
        t1 = time.time()
        check_config_matches_base(path)
        load = load_checkpoint_into_model(model, path)
        load["load_s"] = round(time.time() - t1, 1)
        log(f"{name}: loaded", load)
        tok = AutoTokenizer.from_pretrained(path)
        results, acts, checks = evaluate_loaded_model(model, tok, prompts, efforts, vectors, layer, bs, "cuda")
        rec: dict = {"name": name, "path": path, "load": load, "checks": checks, "results": results}
        if base_sha is None:
            base_sha = {e: c["rendered_sha256"] for e, c in checks.items()}
        rec["prompts_identical_to_first_ckpt"] = all(checks[e]["rendered_sha256"] == base_sha[e] for e in efforts)
        if name == "bf16_base":
            base_acts = acts
            if ref_acts is not None:  # same prompts through our loader vs HF from_pretrained
                mine = collect_activations(model, find_layers(model), tok, ref_texts, layer, bs, "cuda")
                rec["swap_vs_hf_from_pretrained"] = compare_acts(mine, ref_acts)
        elif base_acts is not None:
            rec["rel_l2_vs_bf16_base"] = {e: compare_acts(acts[e], base_acts[e]) for e in efforts}
        if "batch_check" not in result:  # numeric noise floor of batching, measured once on the first checkpoint
            idx = head_per_set(prompts, 4)
            alt = 8 if bs == 1 else 1
            texts = [render_prompt(tok, prompts[i]["messages"], efforts[0]) for i in idx]
            other = collect_activations(model, find_layers(model), tok, texts, layer, alt, "cuda")
            result["batch_check"] = {"batch_size": bs, "alt_batch_size": alt, "n_prompts": len(idx),
                                     "agreement": compare_acts({s: acts[efforts[0]][s][idx] for s in SITES}, other)}
        np.savez_compressed(out_dir / f"acts_{name}.npz", prompt_ids=np.array([p["id"] for p in prompts]),
                            **{f"{e}__{s}": acts[e][s] for e in efforts for s in SITES})
        (out_dir / f"{name}.json").write_text(json.dumps(rec) + "\n")
        art_vol.commit()
        result["ckpts"][name] = rec
        log(f"{name}: done in {time.time() - t1:.0f}s")
    result["elapsed_s"] = round(time.time() - t0, 1)
    return result
