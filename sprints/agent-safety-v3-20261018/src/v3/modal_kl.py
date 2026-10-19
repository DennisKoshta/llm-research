"""Benign-KL gate (spec section 1.2: benign KL < 0.10) for every 27B checkpoint, measured on the served FP8 weights.

Metric. For a checkpoint X, a benign prompt i and a reference model R (the same family's intact checkpoint:
r1_intact for r1_* arms, r2_intact for r2_* arms; every arm is ALSO reported against bf16_base, and bf16_base against
both intact checkpoints):

    KL_t(i) = sum_v P_R(v | ctx_i,t) * ( log P_R(v | ctx_i,t) - log P_X(v | ctx_i,t) )      [nats, KL(P_R || P_X)]

with P = softmax of the model's next-token logits (float32 log-softmax, sums in float64). Two variants:
  (a) kl_prompt_last     KL_0(i): context = the rendered prompt; the distribution at the LAST PROMPT position (the "\\n"
                         after the template's "<think>"), i.e. the first generated token. The prompt is the transcript's
                         `full_prompt` string, exactly as the generation path rendered it (enable_thinking=True,
                         add_generation_prompt=True, ends '<think>\\n'; the job re-renders and counts matches).
  (b) kl_teacher_forced  mean over t = 1..32 of KL_t(i), where position t is the position of generated token g_{t-1}
                         (context = prompt + g_0 .. g_{t-1}; the distribution predicts g_t). The g's are the tokens of
                         the intact reference model's own greedy completion (raw_output of the local vLLM transcript),
                         retokenised with the checkpoint's tokenizer as prompt + raw_output, add_special_tokens=False
                         (if the prompt/completion boundary re-merges, prompt ids and completion ids are tokenised
                         separately and concatenated; the job counts those cases). The sequence fed to the model is
                         prompt + g_0..g_31 and the last 33 logit rows are kept: position 0 = last prompt position
                         (variant a), positions 1..32 = the first 32 generated positions (variant b).
  The spec'd trajectory source is raw_transcripts/phase1_dim_r1/intact.jsonl for every checkpoint ("r1" trajectory).
  Because the r2 family's own intact greedy completions exist (phase1_dim_r2/intact.jsonl, identical prompts), a third
  variant kl_teacher_forced_own is computed too: r2_* arms (and any reference) on the "r2" trajectory
  (= r2_intact's own greedy tokens), r1_* arms / bf16_base on the "r1" trajectory.
Both trajectories share position 0 (same prompt); the r2 pass recomputes it and the difference is reported as a
numeric-noise probe (`pos0_traj_consistency`).

Prompts. The harmless default-effort (reasoning_effort None, cot_on, single-turn) rows of Set C and Set A0 in the intact
transcripts; distinct by full_prompt; 24 per set taken in ascending sha256(salt|trial_id) order (deterministic, input
order independent; topped up from the other set if one has fewer); at most 48. Prompt and completion text are never
printed or written to the records (ids and sha256 only).

Numerics. The model is the BF16 text-only architecture; each checkpoint's effective weights are copied in with the
dequantizing swap loader of v3.modal_hf (built once per container, swapped per checkpoint). Batch size 1, no padding.
Logits: the last hidden state (post final norm) of the backbone times lm_head.weight in FLOAT32 (chunked over the
vocabulary), so the metric is not floored by bf16 rounding of the logits (~0.03-0.06 per logit at |logit| ~ 20). A one-time
in-job check compares this against the model's own bf16 `logits_to_keep` logits (KL between the two paths is recorded;
if it exceeds 0.01 the job falls back to the model's own logits and says so).

Memory plan (vocab ~248k, 33 positions, 48 prompts). One prompt's log-probs are [33, V] float32 = 33 MB on the GPU.
Reference log-probs stay on the HOST in float32 ([48, 33, V] = 1.57 GB per reference per trajectory; 3 references
(bf16_base, r1_intact, r2_intact) x 2 trajectories = 9.4 GB; float16 is NOT used because log-probs near -20 have a
0.016 resolution, comparable to the KL scale being measured). Checkpoint log-probs are never stored: each prompt's [33, V]
is compared against the host reference slices right away. KL itself is evaluated in chunks of 8 rows in float64
(8 x V x 8 B = 16 MB per temporary). Peak host memory ~ 12 GB of the 64-128 GB container; GPU ~ 54 GB model + < 1 GB.

Launcher: scripts/run_kl.py. Results land under /artifacts/agent-safety-v3-20261018/kl/<tag>/.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import time

import modal
import numpy as np

from v3 import modal_hf as mh

APP_NAME = "agent-safety-v3-20261018-kl"
NS = mh.NS
ART = mh.ART
OUT_DIR = f"{ART}/kl"
BASE_DIR = mh.BASE_DIR
GPU = mh.GPU
TIMEOUT_S = 5400            # 90 min hard stop (~$7.8); expected 10-25 min
N_PROMPTS = 48
N_GEN = 32                  # generated positions scored by the teacher-forced variant
N_POS = N_GEN + 1           # position 0 = last prompt position, 1..32 = first 32 generated positions
MIN_GEN_TOKENS = N_GEN + 8  # transcripts shorter than this are not eligible
SET_LABELS = ("C", "A0")
SELECT_SALT = "v3-kl-benign-v1"
GATE = 0.10                 # spec section 1.2
REFERENCES = ("bf16_base", "r1_intact", "r2_intact")
TRAJS = ("r1", "r2")
FAMILY_REF = {"r1": "r1_intact", "r2": "r2_intact"}
HEAD_CHECK_MAX_KL = 0.01
KL_CHUNK_ROWS = 8
HEAD_CHUNK_ROWS = 16384

# Same layers as modal_hf's image (a known-good transformers 5.20.1 + Qwen3.8 stack), so the image build is cached.
weights_vol = modal.Volume.from_name("curiosity-study-weights").read_only()
art_vol = modal.Volume.from_name("curiosity-study-artifacts")
image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.31.0", add_python="3.12")
    .entrypoint([])
    .pip_install("transformers==5.20.1", "accelerate==1.15.0", "scipy==1.18.1")
    .env({"HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("v3")
)
app = modal.App(APP_NAME, image=image)


# ----------------------------------------------------------------------------------------------- KL math
def kl_rows(ref_logp, x_logp, chunk: int = KL_CHUNK_ROWS) -> np.ndarray:
    """KL(P_ref || P_x) in nats for each row; inputs are [R, V] log-probabilities (rows normalised), any float dtype
    and device. Computed in float64 chunks of `chunk` rows; the reference may live on another device (the host) than x.
    Entries with P_ref == 0 contribute 0. Returns float64 [R] (values below ~1e-6 are float32 normalisation noise)."""
    import torch

    if ref_logp.ndim != 2 or tuple(ref_logp.shape) != tuple(x_logp.shape):
        raise ValueError(f"expected two [R, V] arrays of equal shape, got {tuple(ref_logp.shape)} and {tuple(x_logp.shape)}")
    if chunk < 1:
        raise ValueError(f"chunk must be >= 1, got {chunk}")
    out = np.empty(ref_logp.shape[0], dtype=np.float64)
    for s in range(0, ref_logp.shape[0], chunk):
        lr = ref_logp[s:s + chunk].to(device=x_logp.device, dtype=torch.float64)
        lx = x_logp[s:s + chunk].to(torch.float64)
        p = lr.exp()
        term = torch.where(p > 0, p * (lr - lx), torch.zeros_like(p))
        out[s:s + chunk] = term.sum(-1).cpu().numpy()
    return out


def fp32_linear(h, weight, chunk: int = HEAD_CHUNK_ROWS):
    """h [T, D] @ weight[V, D].T in float32, chunked over V (a float32 copy of the whole head would be 5 GB) -> [T, V]."""
    import torch

    h32 = h.float()
    return torch.cat([h32 @ weight[s:s + chunk].float().T for s in range(0, weight.shape[0], chunk)], dim=-1)


def last_logprobs(model, ids: list[int], n_keep: int = N_POS, fp32_head: bool = True, device="cuda"):
    """float32 log P(next token) at the last `n_keep` positions of the single (unpadded) sequence `ids` -> [n_keep, V].

    fp32_head=True: backbone last_hidden_state (post final norm) x lm_head.weight in float32. False: the model's own
    (bf16) logits via `logits_to_keep`."""
    import torch

    x = torch.tensor([ids], dtype=torch.long, device=device)
    mask = torch.ones_like(x)
    with torch.no_grad():
        if fp32_head:
            h = model.model(input_ids=x, attention_mask=mask, use_cache=False).last_hidden_state[0, -n_keep:]
            logits = fp32_linear(h, model.get_output_embeddings().weight)
        else:
            logits = model(input_ids=x, attention_mask=mask, use_cache=False, logits_to_keep=n_keep).logits[0].float()
        if logits.shape[0] != n_keep:
            raise RuntimeError(f"expected {n_keep} logit rows, got {logits.shape[0]}")
        return torch.log_softmax(logits.float(), dim=-1)


# ----------------------------------------------------------------------------------------------- prompts and sequences
def read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in pathlib.Path(path).read_text().splitlines() if line.strip()]


def row_label(row: dict) -> str | None:
    """'C' / 'A0' for a harmless, default-effort, cot_on, single-turn Set C / Set A0 row; None for everything else."""
    meta = row.get("meta") or {}
    if meta.get("consequence") != "harmless" or row.get("reasoning_effort") is not None:
        return None
    if row.get("cot_mode") != "cot_on" or row.get("kind") != "single":
        return None
    if meta.get("set_name") == "C":
        return "C"
    if meta.get("set_name") == "A" and meta.get("variant") == "A0":
        return "A0"
    return None


def _eligible(row: dict) -> bool:
    return row.get("finish_reason") == "stop" and int(row.get("generated_tokens", 0)) >= MIN_GEN_TOKENS


def load_benign_candidates(r1_path, r2_path=None) -> tuple[list[dict], dict]:
    """-> (candidates, stats). A candidate = {id, label, full_prompt, messages, completions: {'r1': raw_output[, 'r2': ...]}}.

    r1 rows are the spec'd trajectory source. With r2_path, a candidate additionally needs the r2_intact row of the same
    trial_id (same full_prompt, else ValueError naming the id only); candidates without one are dropped and counted."""
    r1_rows = [r for r in read_jsonl(r1_path) if row_label(r) is not None]
    ids = [r["trial_id"] for r in r1_rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate trial_ids among eligible r1 rows")
    r2_by_id = {}
    if r2_path is not None:
        r2_by_id = {r["trial_id"]: r for r in read_jsonl(r2_path) if row_label(r) is not None}
    stats = {"n_r1_rows": len(r1_rows), "n_short_or_unfinished_r1": 0, "n_missing_r2": 0, "n_short_or_unfinished_r2": 0}
    cands = []
    for r in r1_rows:
        if not _eligible(r):
            stats["n_short_or_unfinished_r1"] += 1
            continue
        comp = {"r1": r["raw_output"]}
        if r2_path is not None:
            r2 = r2_by_id.get(r["trial_id"])
            if r2 is None:
                stats["n_missing_r2"] += 1
                continue
            if r2["full_prompt"] != r["full_prompt"]:
                raise ValueError(f"r1 and r2 transcripts disagree on the prompt of {r['trial_id']}")
            if not _eligible(r2):
                stats["n_short_or_unfinished_r2"] += 1
                continue
            comp["r2"] = r2["raw_output"]
        cands.append({"id": r["trial_id"], "label": row_label(r), "full_prompt": r["full_prompt"], "messages": r["messages"],
                      "completions": comp})
    return cands, stats


def _hash_key(salt: str, cid: str) -> tuple:
    return hashlib.sha256(f"{salt}|{cid}".encode()).hexdigest(), cid


def select_benign_rows(cands: list[dict], n: int = N_PROMPTS, labels=SET_LABELS, salt: str = SELECT_SALT) -> list[dict]:
    """Deterministic, input-order-independent selection of at most `n` rows, n // len(labels) per label (topped up from
    the remaining pool if a label is short), distinct by full_prompt. Output order: label, then trial id."""
    distinct, seen = [], set()
    for c in sorted(cands, key=lambda c: _hash_key(salt, c["id"])):
        key = hashlib.sha256(c["full_prompt"].encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            distinct.append(c)
    quota = n // len(labels)
    picked = [c for lab in labels for c in [x for x in distinct if x["label"] == lab][:quota]]
    chosen = {c["id"] for c in picked}
    picked += [c for c in distinct if c["id"] not in chosen][:max(0, n - len(picked))]
    return sorted(picked, key=lambda c: (labels.index(c["label"]) if c["label"] in labels else len(labels), c["id"]))


def encode_sequence(tok, prompt: str, completion: str, n_gen: int = N_GEN) -> dict:
    """Token ids of prompt + the first `n_gen` completion tokens: {ids, n_prompt, joint_tokenization}.

    Tokenises prompt and prompt + completion with add_special_tokens=False. When the prompt tokens are a prefix of the
    joint tokenisation it is used; otherwise (boundary merge) the prompt ids and the completion ids (tokenised on their
    own) are concatenated and joint_tokenization is False."""
    p_ids = list(tok(prompt, add_special_tokens=False)["input_ids"])
    full = list(tok(prompt + completion, add_special_tokens=False)["input_ids"])
    joint = full[:len(p_ids)] == p_ids
    ids = full if joint else p_ids + list(tok(completion, add_special_tokens=False)["input_ids"])
    if len(ids) < len(p_ids) + n_gen:
        raise ValueError(f"completion has {len(ids) - len(p_ids)} tokens, need {n_gen}")
    return {"ids": ids[:len(p_ids) + n_gen], "n_prompt": len(p_ids), "joint_tokenization": joint}


def ids_sha256(seqs: list[dict]) -> str:
    return hashlib.sha256(json.dumps([s["ids"] for s in seqs]).encode()).hexdigest()


# ----------------------------------------------------------------------------------------------- aggregation (numpy only)
def agg(values) -> dict:
    a = np.asarray(values, dtype=np.float64)
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)), "max": float(a.max()), "min": float(a.min())}


def own_traj(name: str, trajs) -> str:
    """The trajectory holding the checkpoint's own family's intact greedy tokens (r2_* -> 'r2' when available)."""
    return "r2" if name.startswith("r2_") and "r2" in trajs else "r1"


def variants_from_matrices(mats: dict, name: str) -> dict:
    """mats = {'r1': [N][33], 'r2': [N][33] (optional)} of KL per prompt and position -> per-prompt arrays of
    kl_prompt_last (position 0 of the r1 pass), kl_teacher_forced (mean of positions 1..32 on the r1 trajectory) and
    kl_teacher_forced_own (positions 1..32 on the checkpoint's own-family trajectory)."""
    r1 = np.asarray(mats["r1"], dtype=np.float64)
    own = np.asarray(mats[own_traj(name, mats)], dtype=np.float64)
    return {"kl_prompt_last": r1[:, 0].tolist(), "kl_teacher_forced": r1[:, 1:].mean(1).tolist(),
            "kl_teacher_forced_own": own[:, 1:].mean(1).tolist()}


def summarize_pair(mats: dict, name: str) -> dict:
    """Per-prompt arrays + mean/median/max for each variant, the per-position mean curve (r1 trajectory) and the
    position-0 consistency between trajectories (numeric-noise probe)."""
    out: dict = {}
    for variant, arr in variants_from_matrices(mats, name).items():
        out[variant] = {**agg(arr), "per_prompt": arr, "gate_mean_lt_0.10": bool(np.mean(arr) < GATE)}
    out["per_position_mean_r1"] = np.asarray(mats["r1"], dtype=np.float64).mean(0).tolist()
    if "r2" in mats:
        d = np.abs(np.asarray(mats["r1"], dtype=np.float64)[:, 0] - np.asarray(mats["r2"], dtype=np.float64)[:, 0])
        out["pos0_traj_consistency_max_abs_diff"] = float(d.max())
    return out


def primary_ref(name: str) -> str | None:
    """Same-family intact checkpoint for r1_* / r2_* arms; None for bf16_base."""
    family = name.split("_")[0]
    return FAMILY_REF.get(family) if name != "bf16_base" else None


def build_summary(ckpts: dict) -> dict:
    """ckpts[name]['kl'][ref][traj] = [N][33] matrices -> {name: {'primary_ref', 'vs': {ref: summarize_pair(...)}}}."""
    out: dict = {}
    for name, rec in ckpts.items():
        out[name] = {"primary_ref": primary_ref(name), "vs": {ref: summarize_pair(mats, name) for ref, mats in rec["kl"].items()}}
    return out


# ----------------------------------------------------------------------------------------------- Modal job
def _versions() -> dict:
    import safetensors
    import torch
    import transformers

    return {"torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__,
            "safetensors": safetensors.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def order_checkpoints(ckpts: list[dict]) -> list[dict]:
    """References (bf16_base, r1_intact, r2_intact) first in that order, then the rest in the given order."""
    rank = {n: i for i, n in enumerate(REFERENCES)}
    return sorted(ckpts, key=lambda c: rank.get(c["name"], len(rank)))


@app.function(image=image, gpu=GPU, volumes={"/weights": weights_vol, "/artifacts": art_vol}, cpu=(8, 16),
              memory=(65536, 131072), timeout=TIMEOUT_S, retries=0, max_containers=1, scaledown_window=2)
def kl_checkpoints(job: dict) -> dict:
    """One GPU container: build the BF16 architecture once; for each checkpoint swap in its effective weights, run every
    selected benign prompt (prompt + first 32 reference tokens) with batch size 1 and keep the last 33 log-prob rows.
    References (bf16_base, r1_intact, r2_intact) are banked on the host in float32; every checkpoint's KL against every
    banked reference is computed per prompt and trajectory. Writes <OUT_DIR>/<tag>/<name>.json per checkpoint.

    job = {tag, ckpts: [{name, path}], rows: [{id, label, full_prompt, messages, completions: {r1[, r2]}}], n_gen, repeat_prompts}."""
    import gc

    import torch
    from transformers import AutoTokenizer

    t0 = time.time()

    def log(*a):
        print(f"[kl {time.time() - t0:6.0f}s]", *a, flush=True)

    art_vol.reload()
    rows, n_gen = job["rows"], job.get("n_gen", N_GEN)
    n_pos, n = n_gen + 1, len(rows)
    trajs = [t for t in TRAJS if all(t in r["completions"] for r in rows)]
    if "r1" not in trajs or n == 0:
        raise ValueError(f"need at least one row and the r1 trajectory, got trajectories {trajs} for {n} rows")
    for r in rows:
        bad = [lit for lit in mh.FORBIDDEN_IN_PROMPTS if lit in r["full_prompt"]]
        if bad or not r["full_prompt"].endswith("<think>\n"):
            raise ValueError(f"row {r['id']}: prompt must end with '<think>\\n' and contain no {mh.FORBIDDEN_IN_PROMPTS}")
    ckpts = order_checkpoints(job["ckpts"])
    refs_in_job = [c["name"] for c in ckpts if c["name"] in REFERENCES]
    out_dir = pathlib.Path(OUT_DIR) / job["tag"]
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {"versions": _versions(), "config": {"n_prompts": n, "n_gen": n_gen, "trajectories": trajs, "references": refs_in_job,
                                                        "prompt_ids": [r["id"] for r in rows], "labels": [r["label"] for r in rows],
                                                        "prompt_sha256": [hashlib.sha256(r["full_prompt"].encode()).hexdigest() for r in rows]},
                    "ckpts": {}, "checks": {}}

    device = job.get("device", "cuda")
    model = mh.build_model(BASE_DIR, device)
    log("blank model built")
    bank: dict = {}                      # bank[ref][traj] = float32 CPU tensor [N, n_pos, V]
    kl: dict = {c["name"]: {} for c in ckpts}   # kl[name][ref][traj] = np [N, n_pos]
    seqs0: dict | None = None            # token ids per trajectory from the first checkpoint's tokenizer
    fp32_head = True

    def write_ckpt(name: str) -> None:
        rec = result["ckpts"][name]
        rec["kl"] = {ref: {t: m.tolist() for t, m in by_t.items()} for ref, by_t in kl[name].items()}
        rec["n_nonfinite_kl"] = sum(int((~np.isfinite(m)).sum()) for by_t in kl[name].values() for m in by_t.values())
        (out_dir / f"{name}.json").write_text(json.dumps(rec) + "\n")
        art_vol.commit()

    def fill_reference_pairs() -> None:
        """Reference checkpoints: KL against the other banked references (and themselves), from the host bank."""
        for name in refs_in_job:
            for ref in refs_in_job:
                if ref in kl[name]:
                    continue
                kl[name][ref] = {}
                for t in trajs:
                    m = np.zeros((n, n_pos))
                    for i in range(n):
                        m[i] = kl_rows(bank[ref][t][i], bank[name][t][i].to(device))
                    kl[name][ref][t] = m
            write_ckpt(name)

    for ck in ckpts:
        name, path = ck["name"], ck["path"]
        t1 = time.time()
        mh.check_config_matches_base(path)
        load = mh.load_checkpoint_into_model(model, path)
        load["load_s"] = round(time.time() - t1, 1)
        tok = AutoTokenizer.from_pretrained(path)
        render_ok = 0
        for r in rows:
            try:
                render_ok += int(mh.render_prompt(tok, r["messages"], None) == r["full_prompt"])
            except ValueError:
                pass
        seqs = {t: [encode_sequence(tok, r["full_prompt"], r["completions"][t], n_gen) for r in rows] for t in trajs}
        sha = {t: ids_sha256(seqs[t]) for t in trajs}
        if seqs0 is None:
            seqs0 = seqs
        tok_same = all(sha[t] == ids_sha256(seqs0[t]) for t in trajs)
        use = seqs if tok_same else seqs0          # KL needs identical inputs across checkpoints
        diag = {"load": load, "render_matches": render_ok, "tokens_sha256": sha, "tokenization_same_as_first_ckpt": tok_same,
                "n_joint_tokenization": {t: sum(s["joint_tokenization"] for s in seqs[t]) for t in trajs},   # this tokenizer's own
                "n_prompt_tokens_min_max": [min(s["n_prompt"] for s in use["r1"]), max(s["n_prompt"] for s in use["r1"])]}
        log(f"{name}: loaded {load}; render_matches={render_ok}/{n} tokenization_same={tok_same}")

        if "head_check" not in result["checks"]:   # structure assumption behind the float32 head, checked once
            a = last_logprobs(model, use["r1"][0]["ids"], n_pos, True, device)
            b = last_logprobs(model, use["r1"][0]["ids"], n_pos, False, device)
            d = kl_rows(b, a)
            if d.max() > HEAD_CHECK_MAX_KL:
                fp32_head = False
            result["checks"]["head_check"] = {"kl_ownlogits_vs_fp32head_max": float(d.max()), "max_abs_logp_diff": float((a - b).abs().max()),
                                              "n_pos": n_pos, "fp32_head_used": fp32_head}
            log("head check:", result["checks"]["head_check"])

        is_ref = name in REFERENCES
        if is_ref:
            bank[name] = {}
        have = [r for r in bank if r in refs_in_job and r != name]   # references already banked
        for t in trajs:
            for i in range(n):
                lp = last_logprobs(model, use[t][i]["ids"], n_pos, fp32_head, device)
                if not bool(torch.isfinite(lp).all()):
                    raise RuntimeError(f"{name}: non-finite log-probs for trajectory {t} prompt index {i}")
                if is_ref:
                    if t not in bank[name]:
                        bank[name][t] = torch.empty((n, n_pos, lp.shape[-1]), dtype=torch.float32)
                    bank[name][t][i] = lp.cpu()
                for ref in have:
                    by_t = kl[name].setdefault(ref, {})
                    if t not in by_t:
                        by_t[t] = np.zeros((n, n_pos))
                    by_t[t][i] = kl_rows(bank[ref][t][i], lp)
        if "repeat_check" not in result["checks"]:   # same prompts again in the same container: run-to-run noise floor
            k = min(job.get("repeat_prompts", 3), n)
            worst = 0.0
            for i in range(k):
                again = last_logprobs(model, use["r1"][i]["ids"], n_pos, fp32_head, device)
                first = bank[name]["r1"][i] if is_ref else last_logprobs(model, use["r1"][i]["ids"], n_pos, fp32_head, device).cpu()
                worst = max(worst, float(kl_rows(first, again).max()))
            result["checks"]["repeat_check"] = {"ckpt": name, "n_prompts": k, "max_kl_between_two_runs": worst}
            log("repeat check:", result["checks"]["repeat_check"])
        result["ckpts"][name] = {"name": name, "path": path, "diag": diag, "seconds": round(time.time() - t1, 1)}
        if refs_in_job and name == refs_in_job[-1]:
            fill_reference_pairs()
        elif not is_ref:
            write_ckpt(name)
        gc.collect()
        torch.cuda.empty_cache()
        log(f"{name}: done in {time.time() - t1:.0f}s")
    result["elapsed_s"] = round(time.time() - t0, 1)
    log("all done")
    return result
