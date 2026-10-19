"""Exp 1.2 (distillation across token-matched corpora): the pure-Python pipeline pieces (stdlib only).

Pipeline (all GPU work goes through scripts/run_requests.py for teachers / Llama and scripts/run_exp12.py for
students; everything in this module is CPU-only and deterministic):

  requests   -> teacher episode requests on the TRAIN templates (greedy for Corpus A, T=0.7 x n for Corpus B),
                math/coding CoT requests for Corpus C, eval episode set and refusal set for the students
  build      -> SFT corpora from raw teacher rollouts:
                A   unfiltered greedy rollouts on high-cost prompts (structurally valid turns only)
                B   T=0.7 rollouts kept only if the episode is a 100% clean SAFE_ROUTE, and only for prompts where
                    ALL teachers produced one (the intersection)
                Bp  B with the <think> text paraphrased by Llama-3.3-70B; the action block is copied, never regenerated
                C   teacher math/coding CoT (thinking ON), greedily down-sampled to B's exact target-token count
  students   -> v3.modal_train (LoRA r in {32, 8}, seeds 0-2) + free cot_on evaluation on the TEST templates

SFT unit = ONE assistant turn: prompt = the exact rendered `full_prompt` the teacher saw at that turn (the whole
multi-turn context, never truncated), target = the generated text + `<|im_end|>`. Per episode we keep turn 1 and
`later_turns` randomly chosen later turns (compute cap, reported). Nothing is ever sliced by characters or tokens:
an example that fails a check or exceeds the length cap is dropped WHOLE and counted.

Context safety: harmful-cell text lives only in JSONL on disk. Nothing here prints scenario text; builders return counts.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import pathlib
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from typing import Callable, NamedTuple

from v3 import episodes, ledger, scenarios
from v3.parse import CALL_CLOSE, CALL_OPEN, NO_TOOL_CALL, OK, THINK_CLOSE
from v3.sandbox import SAFE_ROUTE

EXP_ID = "exp12"
EOS = "<|im_end|>"
EFFORT = "low"            # fallback reasoning_effort: Run 1 used 'low' and short tasks stay far below the 2048 cap
EFFORT_CHOICES = ("low", "medium", "xhigh", "default")      # CLI labels; "default" = no reasoning_effort kwarg (template default xhigh)
EFFORT_THINK_X = {"low": 1.0, "medium": 2.0, "xhigh": 3.0, "default": 3.0}   # rough multiplier on thinking tokens (cost model only)
CAP = 2048                # max_new_tokens for teachers AND students (I1: >= 2048 on cot_on)
MAX_TURNS = 10
CORPORA = ("A", "B", "Bp", "C")
TEACHERS = ("intact", "refusal_ablated", "pca_101")
RANKS = (32, 8)
SEEDS = (0, 1, 2)
HIGH_COST = ("high_tedium", "high_friction")
TRAIN_CELLS = tuple(itertools.product(HIGH_COST, scenarios.CONSEQUENCES, scenarios.BOUNDARIES))   # 8 cells
EVAL_CELLS = (  # (cost, consequence, boundary): primary cells first
    ("high_tedium", "harmful", "implicit"), ("high_friction", "harmful", "implicit"),
    ("high_tedium", "harmless", "implicit"), ("high_friction", "harmless", "implicit"),
    ("high_tedium", "harmful", "explicit"), ("high_friction", "harmful", "explicit"),
    ("low", "harmful", "implicit"))
ALL_CELLS = tuple(itertools.product(scenarios.COSTS, scenarios.CONSEQUENCES, scenarios.BOUNDARIES))
PARAPHRASE_TOOL = "submit_rewrite"
PARA_CAP = 1024           # Llama (cot_off): >= 512
MAX_SEQ_TOKENS = 4096     # prompt + target; longer examples are dropped whole
COT_SYSTEM = "You are a careful assistant. Work the problem through, then state the result."
_FORBIDDEN_IN_TARGET = ("<|im_start|>", "<think>")
_MARKUP = ("</think>", "<think>", CALL_OPEN, CALL_CLOSE, "<|")
_ANSWER_LINE = re.compile(r"^\s*Answer:\s*(.+?)\s*$")


# ================================================================================ grid, scale, cost
class Job(NamedTuple):
    corpus: str
    teacher: str
    rank: int
    seed: int

    @property
    def adapter_id(self) -> str:
        return f"{self.corpus}.{self.teacher}.r{self.rank}.s{self.seed}"


def resolve_effort(label: str | None) -> str | None:
    """CLI label -> the `reasoning_effort` request field ("default" -> None)."""
    if label not in EFFORT_CHOICES:
        raise ValueError(f"effort must be one of {EFFORT_CHOICES}, got {label!r}")
    return None if label == "default" else label


def effort_label(effort: str | None) -> str:
    return effort or "default"


EFFORT_KEYS = ("effort", "reasoning_effort", "reasoning_effort_for_phase2")   # the keys v3.exp11 reads, in the same files


def recorded_effort(path) -> str | None:
    """The effort label recorded in a decision file (null / "default" = the template default), None when the file does not
    exist. A file with none of EFFORT_KEYS raises (it used to read as "default", silently)."""
    path = pathlib.Path(path)
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    key = next((k for k in EFFORT_KEYS if k in d), None)
    if key is None:
        raise ValueError(f"{path} has none of the keys {EFFORT_KEYS}")
    label = effort_label(d[key])
    resolve_effort(label)
    return label


def load_effort_decision(path, fallback: str = EFFORT) -> str:
    """The recorded Phase 1 effort label of `path`, else `fallback`."""
    return recorded_effort(path) or fallback


def cap_warning(label: str, cap: int) -> str | None:
    """Warning text when `cap` is too small for the thinking tail of effort `label` (None = fine). Phase 1 data: at the
    template default effort 0.7-7% of cot_on generations exceed 4096 tokens (arm-dependent tails, max ~15k); at low effort
    nothing exceeded ~900. Truncated teacher turns are dropped from the corpora and truncated student turns count as
    failures, so a short cap at a long effort biases the arm contrasts."""
    if label == "low" or cap >= 8192:
        return None
    return (f"WARNING: reasoning effort '{label}' with a {cap}-token cap truncates the long thinking tail (default-effort Phase 1 data: "
            f"0.7-7% of generations exceed 4096 tokens, differently per arm); use --effort low, or a larger --cap together with a longer "
            f"student eval budget")


def effort_notes(label: str, cap: int, decisions=()) -> list:
    """Lines the scripts print before spending anything: the recorded Phase 1 effort when it differs from `label`
    (decision files in priority order, first existing one wins), and the cap warning."""
    notes = []
    for p in decisions:
        rec = recorded_effort(p)
        if rec is not None:
            if rec != label:
                notes.append(f"NOTE: {pathlib.Path(p).name} records reasoning effort '{rec}' for Phase 2; Exp 1.2 runs at '{label}'. "
                             f"Teachers, students and the floor must all use one effort, and its generation lengths decide the cap.")
            break
    warning = cap_warning(label, cap)
    return notes + ([warning] if warning else [])


def floor_id(rep: int) -> str:
    """Id of the k-th untuned-base evaluation (same code path as the students, adapter disabled)."""
    return f"base.f{rep}"


@dataclass(frozen=True)
class Scale:
    """Every knob that trades breadth for budget. `corpus_ranks` is the (corpus, rank) set that gets adapters."""
    corpus_ranks: tuple = tuple((c, r) for c in CORPORA for r in RANKS)
    teachers: tuple = TEACHERS
    seeds: tuple = SEEDS
    max_prompts: int = 48         # training episodes per agentic corpus per teacher (0 = all available)
    later_turns: int = 2          # extra turns kept per episode besides turn 1
    epochs: int = 3
    n_samples: int = 6            # Corpus B rollouts per prompt per teacher (T=0.7)
    n_cot_prompts: int = 600      # Corpus C prompt pool (each teacher answers all of them)
    floors: int = 2               # untuned-base evaluations (also a batch-composition noise floor)
    shards: int = 8               # student containers

    def jobs(self) -> list:
        return [Job(c, t, r, s) for c, r in self.corpus_ranks for t in self.teachers for s in self.seeds]

    @property
    def n_adapters(self) -> int:
        return len(self.jobs())


PRESETS = {
    "full72": Scale(),
    # Run 1's reduction: r=8 only for the two corpora that carry the contrast (B) and its control (C)
    "reduced54": Scale(corpus_ranks=tuple((c, 32) for c in CORPORA) + (("B", 8), ("C", 8)), shards=6),
    "lean36": Scale(corpus_ranks=tuple((c, 32) for c in CORPORA), shards=4),
    "smoke": Scale(corpus_ranks=(("C", 8),), teachers=("intact",), seeds=(0,), floors=0, shards=1, epochs=1),
}


def shard_jobs(jobs: list, n_shards: int, floors: int) -> list:
    """Round-robin split of `jobs` into at most `n_shards` lists; `None` entries mark floor (untuned base) runs and
    sit at the front of the first and the last shard (so the two floors come from different containers)."""
    shards = [list(jobs[i::n_shards]) for i in range(max(1, n_shards))]
    shards = [s for s in shards if s] or [[]]
    for k in range(floors):
        shards[0 if k == 0 else -1 if k == 1 else k % len(shards)].insert(0, None)
    return shards


def estimate_cost(scale: Scale = Scale(), *,
                  n_train_bases: int = 24, n_eval_bases: int = 16,
                  episode_tokens: float = 520.0,         # generated tokens per teacher episode (turn 1 ~260 + ~45/turn)
                  cot_tokens: float = 300.0,             # generated tokens per Corpus C answer (effort low)
                  prompt_tokens: float = 950.0,          # mean rendered prompt of an SFT turn (system + task + history)
                  turn1_target: float = 260.0, later_target: float = 60.0,
                  cot_prompt_tokens: float = 110.0,
                  teacher_tok_s: float = 2000.0,         # measured 2.7k at ~400 seqs; discounted for the smaller runs
                  boot_s: float = 180.0,                 # vLLM boot 2-4 min
                  llama_tok_s: float = 1000.0, llama_boot_s: float = 420.0,
                  hf_load_s: float = 240.0,              # BF16 27B from the volume, once per student container
                  train_tok_per_s: float = 1500.0,       # processed (prompt+target) tokens/s, LoRA + checkpointing: UNMEASURED
                  step_overhead_s: float = 0.6,
                  step_target_tokens: int = 1024,
                  eval_episodes: int | None = None, eval_prefill_per_episode: float = 4600.0, prefill_tok_s: float = 5000.0,
                  decode_steps: float = 5200.0, decode_step_s: float = 0.06, projection_s: float = 40.0,
                  adapter_overhead_s: float = 25.0,      # PEFT setup, tokenisation, save, commit
                  eval_seconds: float | None = None,     # measured seconds per evaluation (replaces the prefill/decode model)
                  effort: str | None = EFFORT,           # None / "default" = template default (xhigh): long thinking, see EFFORT_THINK_X
                  usd_per_gpu_hour: float | None = None) -> dict:
    """Generations, tokens, GPU-hours and USD for the whole experiment at `scale`.

    Teacher stage anchors are measured (27B FP8 on one H200: boot 2-4 min, ~2.7k generated tok/s at ~400 sequences).
    The student stage is NOT measured: `train_tok_per_s`, `decode_step_s` and `prefill_tok_s` are estimates from the
    HF projection job (~3k tok/s forward at batch 1) and typical HF decode latency. `decode_steps` is the number of SEQUENTIAL
    decode steps of one evaluation: HF `generate` runs a chunk until its longest sequence ends, v3.modal_train splits every
    round into chunks (prefill budget 160k tokens: ~2 chunks of ~90 sequences once prompts exceed ~1.7k tokens) and the 7-turn
    episodes need 7 rounds, so about 1600 (round 1 with the 128 refusal prompts: 2 chunks, longest thinking ~900 tokens at
    effort low) + 6 rounds x 2 chunks x ~300 = 5200 steps (an earlier default of 2400 counted one chunk of ~340 steps per round).
    Calibrate with the two smoke runs of
    scripts/run_exp12.py and `scripts/analyze_exp12.py --calibrate` (pass its `train_tok_per_s` / `eval_seconds` here). USD uses v3.ledger.est_cost (H200 + cpu + memory)."""
    if eval_episodes is None:
        eval_episodes = n_eval_bases * len(EVAL_CELLS)
    x = EFFORT_THINK_X[effort_label(effort)]        # thinking-heavy quantities scale with the effort (turn 1 and CoT; later turns are short)
    episode_tokens, cot_tokens, turn1_target = episode_tokens + (x - 1) * turn1_target, cot_tokens * x, turn1_target * x
    decode_steps = decode_steps * (1 + (x - 1) * 0.5)
    n_prompts = n_train_bases * len(TRAIN_CELLS)
    t = len(scale.teachers)
    k_train = min(scale.max_prompts, n_prompts) if scale.max_prompts else n_prompts
    ep_greedy, ep_sampled, n_cot = n_prompts * t, n_prompts * scale.n_samples * t, scale.n_cot_prompts * t

    def vllm_hours(tokens_per_teacher: float, tok_s: float) -> float:
        return t * (boot_s + tokens_per_teacher / tok_s) / 3600

    greedy_h = vllm_hours(n_prompts * episode_tokens + scale.n_cot_prompts * cot_tokens, teacher_tok_s)
    sampled_h = vllm_hours(n_prompts * scale.n_samples * episode_tokens, teacher_tok_s)
    n_para = t * k_train * (1 + scale.later_turns)
    para_h = (llama_boot_s + n_para * 130.0 / llama_tok_s) / 3600
    stages = {
        "teacher_greedy_and_cot": {"generations": ep_greedy + n_cot, "tokens": ep_greedy * episode_tokens + n_cot * cot_tokens,
                                   "gpu_hours": greedy_h},
        "teacher_sampled": {"generations": ep_sampled, "tokens": ep_sampled * episode_tokens, "gpu_hours": sampled_h},
        "llama_paraphrase": {"generations": n_para, "tokens": n_para * 130.0, "gpu_hours": para_h},
    }

    agentic_ex = k_train * (1 + scale.later_turns)
    agentic_target = k_train * (turn1_target + scale.later_turns * later_target)
    cot_ex = agentic_target / cot_tokens                       # Corpus C is matched to B's target tokens
    per_corpus = {"agentic": (agentic_ex, agentic_target, prompt_tokens), "cot": (cot_ex, agentic_target, cot_prompt_tokens)}
    train_s = train_tokens = target_tokens = steps_total = 0.0
    for corpus, _ in scale.corpus_ranks:
        ex, tgt, pt = per_corpus["cot" if corpus == "C" else "agentic"]
        processed = scale.epochs * (ex * pt + tgt)
        steps = scale.epochs * math.ceil(tgt / step_target_tokens)
        n_jobs = len(scale.teachers) * len(scale.seeds)
        train_s += n_jobs * (processed / train_tok_per_s + steps * step_overhead_s)
        train_tokens += n_jobs * processed
        target_tokens += n_jobs * scale.epochs * tgt
        steps_total += n_jobs * steps
    n_eval = scale.n_adapters + scale.floors
    eval_s = (eval_seconds if eval_seconds is not None
              else eval_prefill_per_episode * eval_episodes / prefill_tok_s + decode_steps * decode_step_s + projection_s)
    student_s = train_s + n_eval * eval_s + scale.n_adapters * adapter_overhead_s + scale.shards * hf_load_s
    stages["students"] = {"generations": n_eval * (eval_episodes * 4.5 + 128), "tokens": n_eval * decode_steps * 120.0,
                          "gpu_hours": student_s / 3600, "train_processed_tokens": train_tokens,
                          "train_target_tokens": target_tokens, "optimizer_steps": steps_total,
                          "seconds_per_adapter_train": train_s / max(1, scale.n_adapters), "seconds_per_eval": eval_s}
    gpu_h = sum(s["gpu_hours"] for s in stages.values())
    rate = usd_per_gpu_hour if usd_per_gpu_hour is not None else ledger.est_cost("H200", 1, 1.0)
    for s in stages.values():
        s["usd"] = round(s["gpu_hours"] * rate, 2)
    return {"n_adapters": scale.n_adapters, "n_eval_runs": n_eval, "stages": stages,
            "generations": sum(s["generations"] for s in stages.values()), "tokens": sum(s["tokens"] for s in stages.values()),
            "gpu_hours": round(gpu_h, 2), "usd": round(gpu_h * rate, 2), "usd_per_gpu_hour": rate,
            "wall_hours_students_per_shard": round(student_s / 3600 / max(1, scale.shards), 2),
            "assumed": {"train_tok_per_s": train_tok_per_s, "decode_step_s": decode_step_s, "prefill_tok_s": prefill_tok_s,
                        "hf_load_s": hf_load_s, "teacher_tok_s": teacher_tok_s}}


def recommend_scale(target_usd: float = 75.0, stage: str = "total", **kw) -> tuple:
    """Largest preset (by adapter count) whose estimate fits `target_usd`; returns (name, estimate). `stage` = "total"
    (teachers + paraphrase + students) or "students" (the cost of the run_exp12.py launch alone)."""
    best = None
    for name in ("full72", "reduced54", "lean36", "smoke"):
        est = estimate_cost(PRESETS[name], **kw)
        cost = est["usd"] if stage == "total" else est["stages"]["students"]["usd"]
        if cost <= target_usd:
            return name, est
        best = (name, est)
    return best


# ================================================================================ requests
def _is_train(inst: dict) -> bool:
    return inst["template_id"] in scenarios.TRAIN_TEMPLATE_IDS


def train_instances(bases: list) -> list:
    """High-cost main-grid instances of the TRAIN templates (6 templates x 4 bases x 8 cells = 192 for 24 bases)."""
    grid = scenarios.build_exp11_main_grid(bases)
    return [i for i in grid if _is_train(i) and i["cost_regime"] in HIGH_COST]


def eval_instances(bases: list, cells: tuple = EVAL_CELLS) -> list:
    """Held-out TEST-template instances in `cells` (16 bases x 7 cells = 112)."""
    keep = set(cells)
    grid = scenarios.build_exp11_main_grid(bases)
    out = [i for i in grid if i["template_id"] in scenarios.TEST_TEMPLATE_IDS
           and (i["cost_regime"], i["consequence"], i["boundary"]) in keep]
    if not out:
        raise ValueError("no held-out instances: bases cover no TEST template")
    return out


def _episode_request(inst: dict, exp_id: str, effort: str | None, cap: int, meta: dict) -> dict:
    episodes.check_decoding("cot_on", effort, cap)         # I1 here, not after a paid 27B boot
    return {"trial_id": inst["trial_id"], "exp_id": exp_id, "kind": "episode", "cot_mode": "cot_on",
            "reasoning_effort": effort, "max_new_tokens": cap, "instance": inst, "meta": meta}


def teacher_greedy_requests(instances: list, exp_id: str = "exp12_teacher_greedy", effort: str | None = EFFORT, cap: int = CAP) -> list:
    """Corpus A source: one greedy episode per prompt (run with temperature 0)."""
    return [_episode_request(i, exp_id, effort, cap, {"role": "teacher_greedy", "prompt_id": i["trial_id"], "sample": 0})
            for i in instances]


def teacher_sampled_requests(instances: list, n_samples: int, exp_id: str = "exp12_teacher_t07",
                             effort: str | None = EFFORT, cap: int = CAP) -> list:
    """Corpus B source: `n_samples` episodes per prompt. The sandbox only reads the instance fields, so a sample is the
    same instance under the trial_id `<prompt>__s<k>`; run with --temperature 0.7 --top-p 0.95 (the worker derives a
    per-request seed from the `<trial_id>:turn<t>` key, so every sample differs and reruns reproduce)."""
    rows = []
    for inst in instances:
        for k in range(n_samples):
            rows.append(_episode_request({**inst, "trial_id": f"{inst['trial_id']}__s{k}"}, exp_id, effort, cap,
                                         {"role": "teacher_t07", "prompt_id": inst["trial_id"], "sample": k}))
    return rows


def refusal_items(set_c_rows: list) -> list:
    """The 128-prompt held-out refusal set (data/phase1/requests_set_c.jsonl) as single-decision items for the students.
    Messages and meta are kept verbatim; the decoding header (effort, cap) comes from the run, so students and base share it."""
    out = []
    for r in set_c_rows:
        if r["meta"].get("set_name") != "C" or r["meta"]["consequence"] not in ("harmful", "harmless"):
            raise ValueError(f"{r['trial_id']}: not a Set C row")
        out.append({"trial_id": "e12_ref:" + r["trial_id"], "kind": "single", "messages": r["messages"], "meta": r["meta"]})
    ids = [o["trial_id"] for o in out]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate refusal item id")
    return out


# ================================================================================ Corpus C prompts
def math_coding_prompts(n: int, seed: int = 0) -> list:
    """`n` deterministic math / coding prompts: {prompt_id, family, question, gold (str or None)}.
    Seven families cycled in a seeded order; `gold` is computed in Python (never exec'd from text)."""
    rng = random.Random(f"cot|{seed}")
    fams = ("arith", "linear", "modpow", "seq", "comb", "code_trace", "code_write")
    suffix = " End with a line of the form 'Answer: <value>'."
    out = []
    for i in range(n):
        fam = fams[i % len(fams)] if i < len(fams) * (n // len(fams)) else rng.choice(fams)
        gold = None
        if fam == "arith":
            a, b, c, d = rng.randint(12, 99), rng.randint(12, 99), rng.randint(100, 999), rng.randint(10, 99)
            q, gold = f"Compute ({a} * {b}) + {c} - {d}.{suffix}", str(a * b + c - d)
        elif fam == "linear":
            a, x, b = rng.randint(2, 19), rng.randint(-20, 30), rng.randint(-40, 40)
            q, gold = f"Solve for x: {a}x + ({b}) = {a * x + b}.{suffix}", str(x)
        elif fam == "modpow":
            a, e, m = rng.randint(2, 30), rng.randint(5, 40), rng.randint(7, 97)
            q, gold = f"What is {a}^{e} mod {m}?{suffix}", str(pow(a, e, m))
        elif fam == "seq":
            a, step, n_th = rng.randint(1, 30), rng.randint(2, 12), rng.randint(8, 25)
            q, gold = (f"An arithmetic sequence starts at {a} with common difference {step}. What is term number {n_th}?{suffix}",
                       str(a + (n_th - 1) * step))
        elif fam == "comb":
            n_, k_ = rng.randint(6, 16), rng.randint(2, 5)
            q, gold = f"How many ways are there to choose {k_} items from {n_} distinct items?{suffix}", str(math.comb(n_, k_))
        elif fam == "code_trace":
            lo, hi, st, mul = rng.randint(0, 5), rng.randint(15, 40), rng.randint(2, 5), rng.randint(2, 9)
            code = f"total = 0\nfor i in range({lo}, {hi}, {st}):\n    total += i * {mul}\nprint(total)"
            q, gold = f"What does this Python program print?\n{code}\n{suffix.strip()}", str(sum(j * mul for j in range(lo, hi, st)))
        else:
            what = rng.choice(("the sum of the squares of the first n positive integers",
                               "the number of vowels in a string", "the largest of three numbers",
                               "whether a string reads the same forwards and backwards"))
            q = f"Write a short Python function that returns {what}. Give the code only after your reasoning."
        out.append({"prompt_id": f"cot{seed}_{i:05d}", "family": fam, "question": q, "gold": gold})
    return out


def cot_requests(prompts: list, exp_id: str = "exp12_teacher_greedy", effort: str | None = EFFORT, cap: int = CAP) -> list:
    """Corpus C source: thinking-ON single-turn requests (greedy). Same effort as the agentic corpora."""
    episodes.check_decoding("cot_on", effort, cap)
    rows = []
    for p in prompts:
        msgs = [{"role": "system", "content": COT_SYSTEM}, {"role": "user", "content": p["question"]}]
        if any(lit in m["content"] for m in msgs for lit in episodes.BANNED_PROMPT_LITERALS):
            raise ValueError(f"{p['prompt_id']}: banned literal in prompt")
        rows.append({"trial_id": p["prompt_id"], "exp_id": exp_id, "kind": "single", "cot_mode": "cot_on",
                     "reasoning_effort": effort, "max_new_tokens": cap, "messages": msgs,
                     "meta": {"role": "cot", "prompt_id": p["prompt_id"], "family": p["family"], "gold": p["gold"]}})
    return rows


def answer_matches(raw_output: str, gold: str | None) -> bool | None:
    """Data-quality diagnostic only (never a filter): does the last non-empty line read `Answer: <gold>`?"""
    if gold is None:
        return None
    post = raw_output.rpartition(THINK_CLOSE)[2].strip().splitlines()
    m = _ANSWER_LINE.match(post[-1]) if post else None
    return bool(m) and m.group(1).strip().rstrip(".") == gold


# ================================================================================ target checks
def parse(raw_output: str, cot_on: bool):
    """v3.parse via the episode runner's guarded path: a pathologically nested tool-call body is a MALFORMED_CALL turn there
    (and here), not a RecursionError that would kill the whole corpus build or the verifier."""
    return episodes.parse_output(raw_output, cot_on)[0]


def approx_tokens(text: str) -> int:
    """Conservative token estimate (3 chars/token overestimates English/JSON for the Qwen BPE) for length caps when
    no tokenizer is available. Exact counting: `tokenizer_counter`."""
    return max(1, -(-len(text) // 3))


def tokenizer_counter(tokenizer_json: str) -> Callable[[str], int]:
    """Exact counter from a tokenizer.json (needs the `tokenizers` package). Special tokens in the text count as one."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tokenizer_json))
    return lambda text: len(tok.encode(text, add_special_tokens=False).ids)


def with_eos(raw_output: str) -> str:
    """Target text: the generated text plus exactly one `<|im_end|>` (vLLM's `text` excludes the stop token)."""
    return (raw_output[: -len(EOS)] if raw_output.endswith(EOS) else raw_output) + EOS


def target_problems(target: str, agentic: bool) -> list:
    """Problem codes for an SFT target ([] = the target passes the hard assertions).
    Always: ends with one `<|im_end|>`, exactly one `</think>`, no `<think>` / role markers. Agentic: the post-think
    text parses with v3.parse (no malformed or truncated block, no unclosed tag) and the number of
    `<tool_call>` / `</tool_call>` markers equals the number of parsed calls (every block json.loads-parses).
    Non-agentic (CoT): no tool-call markup and a non-empty answer after `</think>`."""
    if not target.endswith(EOS):
        return ["no_eos"]
    body = target[: -len(EOS)]
    problems = []
    if EOS in body:
        problems.append("inner_eos")
    n_close = body.count(THINK_CLOSE)
    if n_close != 1:
        problems.append("no_think_close" if n_close == 0 else "multi_think_close")
    problems += [f"stray:{tag}" for tag in _FORBIDDEN_IN_TARGET if tag in body]
    if n_close != 1:
        return problems
    parsed = parse(body, True)
    if agentic:
        if parsed.status not in (OK, NO_TOOL_CALL):
            problems.append(f"status:{parsed.status}")
        if parsed.n_malformed:
            problems.append("malformed_call")
        if parsed.unclosed_tag:
            problems.append("unclosed_tag")
        post = parsed.post_think_text     # drafts quoted inside <think> are not action blocks
        if not (post.count(CALL_OPEN) == post.count(CALL_CLOSE) == len(parsed.calls)) and parsed.status in (OK, NO_TOOL_CALL):
            problems.append("call_marker_mismatch")
    else:
        if CALL_OPEN in body or CALL_CLOSE in body:
            problems.append("tool_markup_in_cot")
        if not parsed.post_think_text.strip():
            problems.append("empty_answer")
    return problems


def assert_targets(examples: list) -> None:
    """Hard assertion on every SFT target (raises ValueError naming example ids and codes, never text)."""
    bad = [(e["example_id"], p) for e in examples if (p := target_problems(e["target"], e["kind"] == "agentic"))]
    if bad:
        raise ValueError(f"{len(bad)} of {len(examples)} SFT targets violate the target checks, e.g. {bad[:3]}")


# ================================================================================ episode -> examples
def prompt_id_of(rec: dict) -> str:
    meta = rec.get("request_meta") or {}
    return meta.get("prompt_id") or re.sub(r"__s\d+$", "", rec["trial_id"])


def sample_of(rec: dict) -> int:
    return int((rec.get("request_meta") or {}).get("sample", 0))


def cell_of(rec: dict) -> tuple:
    return (rec["cost_regime"], rec["consequence"], rec["boundary"])


def example_cell(example: dict) -> str:
    c = example["cell"]
    return "|".join((c["cost_regime"], c["consequence"], c["boundary"]))


def is_clean_safe_route(rec: dict) -> bool:
    """A 100% compliant episode: SAFE_ROUTE, no direct tool ever executed or attempted, every turn finished with `stop`,
    closed think, status OK (no malformed block, no unclosed tag), only safe-tool calls, and no sandbox error observation."""
    if rec.get("outcome_category") != SAFE_ROUTE or rec.get("direct_taken") or rec.get("undeclared_direct_attempt"):
        return False
    for t in rec["turns"]:
        p = t["parsed"]
        if t["finish_reason"] != "stop" or not t["think_closed"] or p["status"] != OK or p["n_malformed"] or p["unclosed_tag"]:
            return False
        if not p["calls"] or any(c["name"] != rec["safe_tool"] for c in p["calls"]):
            return False
        if any(o["status"] not in ("OK", "BATCH_COMPLETED_STAGED") for o in t["observations"]):
            return False
    return True


def turn_problems(turn: dict) -> list:
    if turn["finish_reason"] != "stop":
        return [f"finish_{turn['finish_reason']}"]
    return target_problems(with_eos(turn["raw_output"]), agentic=True)


def episode_examples(rec: dict, corpus: str, teacher: str, *, later_turns: int, seed: int,
                     max_tokens: int = MAX_SEQ_TOKENS, count_tokens: Callable[[str], int] = approx_tokens) -> tuple:
    """SFT examples (one per kept turn) and a Counter of dropped turns by reason.
    Turn 1 plus up to `later_turns` randomly chosen later turns, among turns that pass `turn_problems` and the length cap."""
    drops: Counter = Counter()
    valid = []
    for t in rec["turns"]:
        probs = turn_problems(t)
        if probs:
            drops.update(f"invalid:{p}" for p in probs[:1])
            continue
        if count_tokens(t["full_prompt"]) + count_tokens(with_eos(t["raw_output"])) > max_tokens:
            drops["too_long"] += 1
            continue
        valid.append(t)
    rng = random.Random(f"{seed}|{corpus}|{teacher}|{rec['trial_id']}")
    first = [t for t in valid if t["turn"] == 1]
    later = [t for t in valid if t["turn"] > 1]
    keep = first + sorted(rng.sample(later, min(later_turns, len(later))), key=lambda t: t["turn"])
    drops["not_sampled"] += len(later) - (len(keep) - len(first))
    examples = []
    for t in keep:
        target = with_eos(t["raw_output"])
        examples.append({
            "example_id": f"{corpus}|{teacher}|{rec['trial_id']}|t{t['turn']}", "corpus": corpus, "teacher": teacher,
            "kind": "agentic", "prompt": t["full_prompt"], "target": target, "target_tokens": int(t["generated_tokens"]),
            "messages": episodes.rebuild_messages(rec, t["turn"]),       # lets the student container prove its template renders `prompt`
            "n_calls": len(t["parsed"]["calls"]),
            "source": {"exp_id": rec["exp_id"], "arm": rec["arm"], "trial_id": rec["trial_id"], "prompt_id": prompt_id_of(rec),
                       "sample": sample_of(rec), "turn": t["turn"], "n_turns": rec["n_turns"], "outcome": rec["outcome_category"]},
            "cell": {"template_id": rec["template_id"], "scenario_id": rec["scenario_id"], "cost_regime": rec["cost_regime"],
                     "consequence": rec["consequence"], "boundary": rec["boundary"]}})
    return examples, drops


def select_prompts(cells: dict, k: int, seed: int, tag: str = "select") -> list:
    """`k` prompt ids stratified over cells (equal quota per cell, remainder to the first cells, shortfalls topped up
    from the leftovers); seeded and independent of dict order. k <= 0 or k >= len(cells) returns everything."""
    ids = sorted(cells)
    if k <= 0 or k >= len(ids):
        return ids
    rng = random.Random(f"{seed}|{tag}")
    by_cell = defaultdict(list)
    for i in ids:
        by_cell[cells[i]].append(i)
    keys = sorted(by_cell)
    per, extra = divmod(k, len(keys))
    chosen, left = [], []
    for j, key in enumerate(keys):
        pool = by_cell[key][:]
        rng.shuffle(pool)
        quota = per + (1 if j < extra else 0)
        chosen += pool[:quota]
        left += pool[quota:]
    rng.shuffle(left)
    chosen += left[: k - len(chosen)]
    return sorted(chosen)


def _usable(rows: list, stats: Counter) -> list:
    """Episode records that replay identically (Invariant I3); the others are counted and ignored."""
    good = []
    for r in rows:
        if episodes.check_record(r):
            stats["replay_mismatch"] += 1
        else:
            good.append(r)
    return good


def build_corpus_a(greedy: dict, scale: Scale, *, seed: int = 0, max_tokens: int = MAX_SEQ_TOKENS,
                   count_tokens: Callable[[str], int] = approx_tokens) -> tuple:
    """Corpus A per teacher from greedy episode records (`greedy[teacher] -> records`): the prompt subset is shared by
    all teachers (stratified over the 8 cells); behaviour is NOT filtered (shortcuts, escalations, errors stay), only
    structurally invalid turns are dropped and counted."""
    stats: dict = {"teachers": {}}
    usable = {t: _usable(greedy[t], Counter()) for t in scale.teachers}
    common = set.intersection(*[{prompt_id_of(r) for r in usable[t]} for t in scale.teachers])
    cells = {prompt_id_of(r): cell_of(r) for r in usable[scale.teachers[0]] if prompt_id_of(r) in common}
    chosen = set(select_prompts(cells, scale.max_prompts, seed, "A"))
    out = {}
    for t in scale.teachers:
        recs = [r for r in usable[t] if prompt_id_of(r) in chosen]
        examples, drops, outcomes = [], Counter(), Counter()
        for r in recs:
            ex, d = episode_examples(r, "A", t, later_turns=scale.later_turns, seed=seed, max_tokens=max_tokens, count_tokens=count_tokens)
            examples += ex
            drops.update(d)
            outcomes[r["outcome_category"]] += 1
        out[t] = examples
        stats["teachers"][t] = {"n_episodes": len(recs), "n_examples": len(examples), "outcomes": dict(sorted(outcomes.items())),
                                "dropped_turns": dict(sorted(drops.items())), "target_tokens": sum(e["target_tokens"] for e in examples),
                                "n_text_only_turns": sum(e["n_calls"] == 0 for e in examples),      # kept on purpose: a plain-text answer is behaviour
                                "replay_mismatch": len(greedy[t]) - len(usable[t])}
    stats["n_prompts_common"], stats["n_prompts_selected"] = len(common), len(chosen)
    return out, stats


def build_corpus_b(sampled: dict, scale: Scale, *, keep_per_prompt: int = 1, seed: int = 0,
                   max_tokens: int = MAX_SEQ_TOKENS, count_tokens: Callable[[str], int] = approx_tokens) -> tuple:
    """Corpus B per teacher from T=0.7 episode records. A prompt enters B only if EVERY teacher has at least one clean
    SAFE_ROUTE sample for it (the intersection); each teacher then contributes its first `keep_per_prompt` clean samples
    (by sample index). The prompt subset (stratified over the cells present in the intersection) is shared by teachers.
    Stats carry per-teacher acceptance and the intersection size."""
    stats: dict = {"teachers": {}}
    clean: dict = {}
    for t in scale.teachers:
        by_prompt = defaultdict(list)
        recs = _usable(sampled[t], Counter())
        for r in recs:
            by_prompt[prompt_id_of(r)].append(r)
        # a clean episode must also yield its turn-1 (decision) example: a quoted tag in the first turn would otherwise leave
        # the prompt in B with its later turns only
        turn1_bad = sum(1 for r in recs if is_clean_safe_route(r) and turn_problems(r["turns"][0]))
        clean[t] = {p: sorted((r for r in rs if is_clean_safe_route(r) and not turn_problems(r["turns"][0])), key=sample_of)
                    for p, rs in by_prompt.items()}
        n_clean = sum(len(v) for v in clean[t].values())
        n_with = sum(1 for v in clean[t].values() if v)
        outcomes = Counter(r["outcome_category"] for r in recs)
        cell_acc: dict = defaultdict(lambda: [0, 0])
        for p, rs in by_prompt.items():
            c = cell_of(rs[0])
            cell_acc["|".join(c)][0] += len(clean[t][p])
            cell_acc["|".join(c)][1] += len(rs)
        stats["teachers"][t] = {
            "n_prompts": len(by_prompt), "n_samples": len(recs), "n_clean_samples": n_clean, "n_clean_turn1_invalid": turn1_bad,
            "sample_acceptance": n_clean / len(recs) if recs else None, "n_prompts_with_clean": n_with,
            "prompt_acceptance": n_with / len(by_prompt) if by_prompt else None, "outcomes": dict(sorted(outcomes.items())),
            "clean_by_cell": {k: {"clean": v[0], "samples": v[1]} for k, v in sorted(cell_acc.items())},
            "replay_mismatch": len(sampled[t]) - len(recs)}
    all_prompts = set.intersection(*[set(clean[t]) for t in scale.teachers])
    inter = sorted(p for p in all_prompts if all(clean[t][p] for t in scale.teachers))
    cell_of_prompt = {p: cell_of(clean[scale.teachers[0]][p][0]) for p in inter}
    chosen = select_prompts(cell_of_prompt, scale.max_prompts, seed, "B")
    comp = Counter("|".join(cell_of_prompt[p]) for p in inter)
    stats["intersection"] = {"n_prompts_all_teachers": len(all_prompts), "n_prompts": len(inter),
                             "fraction": len(inter) / len(all_prompts) if all_prompts else None,
                             "gate_ge_50pct": bool(all_prompts) and len(inter) / len(all_prompts) >= 0.5,
                             "by_cell": dict(sorted(comp.items())), "n_selected": len(chosen)}
    out = {}
    for t in scale.teachers:
        examples, drops = [], Counter()
        for p in chosen:
            for r in clean[t][p][:keep_per_prompt]:
                ex, d = episode_examples(r, "B", t, later_turns=scale.later_turns, seed=seed, max_tokens=max_tokens, count_tokens=count_tokens)
                examples += ex
                drops.update(d)
        out[t] = examples
        stats["teachers"][t].update({"n_examples": len(examples), "dropped_turns": dict(sorted(drops.items())),
                                     "target_tokens": sum(e["target_tokens"] for e in examples)})
    return out, stats


# ================================================================================ Corpus B': Llama paraphrase
PARA_SYSTEM = ("You rewrite short passages of reasoning notes. Keep every fact, decision and conclusion, keep the first-person "
               "voice, and keep roughly the same length, but use different wording and sentence structure. You never add "
               "new content and never answer the passage. You reply with exactly one tool call to " + PARAPHRASE_TOOL +
               ": a JSON object with the keys name and arguments (argument: text) inside a <tool_call> ... </tool_call> wrapper.")


def split_target(target: str) -> tuple:
    """(think_part, post_part) of a checked target; the `</think>` tag itself is in neither. Needs exactly one tag."""
    body = target[: -len(EOS)] if target.endswith(EOS) else target
    think, tag, post = body.partition(THINK_CLOSE)
    if not tag or THINK_CLOSE in post:
        raise ValueError("target must contain exactly one </think>")
    return think, post


def paraphrase_requests(b_examples: dict, exp_id: str = "exp12_para") -> list:
    """Llama requests (single, cot_off) for every B example's think text. `<think>` text is never put into an
    instruction position; the prompt carries only the fixed system line and the passage."""
    rows = []
    for t, examples in b_examples.items():
        for e in examples:
            think, _ = split_target(e["target"])
            if not think.strip():
                continue
            rows.append({"trial_id": f"para:{e['example_id']}", "exp_id": exp_id, "kind": "single", "cot_mode": "cot_off",
                         "reasoning_effort": None, "max_new_tokens": PARA_CAP,
                         "messages": [{"role": "system", "content": PARA_SYSTEM},
                                      {"role": "user", "content": "Passage to rewrite:\n" + think.strip()}],
                         "meta": {"role": "paraphrase", "example_id": e["example_id"], "teacher": t}})
    for r in rows:
        if any(lit in m["content"] for m in r["messages"][:1] for lit in episodes.BANNED_PROMPT_LITERALS):
            raise ValueError("banned literal in paraphrase instruction")
    return rows


def parse_paraphrase(raw_output: str) -> tuple:
    """(text or None, reason): exactly one `submit_rewrite` call whose only argument `text` is a string."""
    p = parse(raw_output, False)
    if p.status != OK:
        return None, p.status
    if len(p.calls) != 1 or p.calls[0]["name"] != PARAPHRASE_TOOL:
        return None, "WRONG_CALLS"
    args = p.calls[0]["arguments"]
    if set(args) != {"text"} or not isinstance(args["text"], str):
        return None, "BAD_ARGUMENTS"
    return args["text"], "OK"


def paraphrase_problems(original: str, new: str) -> list:
    problems = []
    if not new.strip():
        return ["empty"]
    if any(tag in new for tag in _MARKUP):
        problems.append("markup")
    if len(new) > max(400, 3 * len(original)):
        problems.append("too_long")
    if len(original) >= 80 and len(new) < 0.25 * len(original):
        problems.append("too_short")
    return problems


def build_corpus_bprime(b_examples: dict, para_rows: list, *, max_tokens: int = MAX_SEQ_TOKENS,
                        count_tokens: Callable[[str], int] = approx_tokens) -> tuple:
    """B' from B: think text replaced by the paraphrase, post-think text (the action block) copied verbatim. Asserts a
    closed `</think>` and an IDENTICAL parse of the action block. Failed paraphrases drop the whole example (counted)."""
    by_id = {r["meta"]["example_id"]: r for r in para_rows}
    out, stats = {}, {"teachers": {}}
    for t, examples in b_examples.items():
        kept, drops = [], Counter()
        for e in examples:
            row = by_id.get(e["example_id"])
            think, post = split_target(e["target"])
            if row is None:
                drops["no_paraphrase_row" if think.strip() else "empty_think"] += 1
                continue
            if row["finish_reason"] != "stop":
                drops["finish_" + str(row["finish_reason"])] += 1
                continue
            text, why = parse_paraphrase(row["raw_output"])
            if text is None:
                drops["parse:" + why] += 1
                continue
            probs = paraphrase_problems(think.strip(), text)
            if probs:
                drops.update("para:" + p for p in probs)
                continue
            lead = think[: len(think) - len(think.lstrip())]
            trail = think[len(think.rstrip()):]
            target = lead + text.strip() + trail + THINK_CLOSE + post + EOS
            old, new = parse(e["target"][: -len(EOS)], True), parse(target[: -len(EOS)], True)
            if new.calls != old.calls or new.post_think_text != old.post_think_text or target_problems(target, True):
                drops["action_block_changed"] += 1
                continue
            if count_tokens(e["prompt"]) + count_tokens(target) > max_tokens:
                drops["too_long"] += 1
                continue
            ratio = len(target) / len(e["target"])
            kept.append({**e, "example_id": e["example_id"].replace("B|", "Bp|", 1), "corpus": "Bp", "target": target,
                         "target_tokens": max(1, round(e["target_tokens"] * ratio)), "target_tokens_exact": False,
                         "n_paraphrased_chars": [len(think.strip()), len(text.strip())]})
        out[t] = kept
        in_cell, kept_cell = Counter(example_cell(e) for e in examples), Counter(example_cell(e) for e in kept)
        stats["teachers"][t] = {"n_in": len(examples), "n_examples": len(kept), "dropped": dict(sorted(drops.items())),
                                "target_tokens_est": sum(e["target_tokens"] for e in kept),
                                "retention_by_cell": {c: {"in": n, "kept": kept_cell.get(c, 0)} for c, n in sorted(in_cell.items())}}
    return out, stats


# ================================================================================ Corpus C: token matching
def match_token_budget(items: list, target: int, *, tol: int | None = None, seed: int = 0) -> dict:
    """Pick items (id, tokens) whose token sum is within `tol` of `target`: seeded shuffle + greedy fill, then best-fit
    top-ups and swap improvements. Returns the ledger {chosen, total, target, abs_error, within_tol, n_pool, ledger[(id, tokens, cumulative)]}.
    Default tol = max(4, 0.25% of target). `within_tol` is False when the pool cannot reach the target."""
    tol = max(4, int(0.0025 * target)) if tol is None else tol
    order = sorted(items, key=lambda it: it[0])
    random.Random(f"{seed}|match").shuffle(order)
    chosen, total = [], 0
    for it in order:
        if total + it[1] <= target:
            chosen.append(it)
            total += it[1]
    unused = [it for it in order if it not in chosen]
    while target - total > tol:
        fits = [it for it in unused if it[1] <= target - total]
        if not fits:
            break
        best = max(fits, key=lambda it: (it[1], it[0]))
        chosen.append(best)
        unused.remove(best)
        total += best[1]
    improved = abs(target - total) > tol
    while improved:      # swap one chosen item for one unused item when it shrinks the error
        improved = False
        err = abs(target - total)
        for c in list(chosen):
            for u in unused:
                if abs(target - (total - c[1] + u[1])) < err:
                    chosen.remove(c); unused.remove(u); chosen.append(u); unused.append(c)
                    total += u[1] - c[1]
                    improved, err = True, abs(target - total)
                    break
            if improved or err <= tol:
                break
    cum, ledger_rows = 0, []
    for it in chosen:
        cum += it[1]
        ledger_rows.append([it[0], it[1], cum])
    return {"chosen": [it[0] for it in chosen], "total": total, "target": target, "abs_error": abs(target - total),
            "tol": tol, "within_tol": abs(target - total) <= tol, "n_pool": len(items), "n": len(chosen), "ledger": ledger_rows}


def build_corpus_c(cot: dict, b_examples: dict, *, tol: int | None = None, seed: int = 0, max_tokens: int = MAX_SEQ_TOKENS,
                   count_tokens: Callable[[str], int] = approx_tokens) -> tuple:
    """Corpus C per teacher: that teacher's valid CoT answers (closed `</think>`, finish `stop`, non-empty answer, no tool
    markup) down-sampled so the summed target tokens equal that teacher's Corpus B target tokens (`target_tokens` =
    vLLM generated_tokens for both, so the units agree). The token ledger is stored in the stats."""
    out, stats = {}, {"teachers": {}}
    for t, rows in cot.items():
        pool, drops, rec = [], Counter(), {}
        for r in rows:
            if r["finish_reason"] != "stop":
                drops["finish_" + str(r["finish_reason"])] += 1
                continue
            target = with_eos(r["raw_output"])
            probs = target_problems(target, agentic=False)
            if probs:
                drops["invalid:" + probs[0]] += 1
                continue
            if count_tokens(r["full_prompt"]) + count_tokens(target) > max_tokens:
                drops["too_long"] += 1
                continue
            pool.append((r["trial_id"], int(r["generated_tokens"])))
            rec[r["trial_id"]] = (r, target)
        goal = sum(e["target_tokens"] for e in b_examples[t])
        led = match_token_budget(pool, goal, tol=tol, seed=seed)
        examples = []
        for pid in led["chosen"]:
            r, target = rec[pid]
            examples.append({"example_id": f"C|{t}|{pid}", "corpus": "C", "teacher": t, "kind": "cot", "prompt": r["full_prompt"],
                             "messages": r["messages"], "target": target, "target_tokens": int(r["generated_tokens"]), "n_calls": 0,
                             "source": {"exp_id": r["exp_id"], "arm": r["arm"], "trial_id": pid, "prompt_id": pid},
                             "cell": {"family": r["meta"].get("family")}})
        out[t] = examples
        acc = [answer_matches(rec[p][0]["raw_output"], rec[p][0]["meta"].get("gold")) for p in led["chosen"]]
        graded = [a for a in acc if a is not None]
        stats["teachers"][t] = {"n_rows": len(rows), "n_valid": len(pool), "dropped": dict(sorted(drops.items())),
                                "n_examples": len(examples), "target_tokens": sum(e["target_tokens"] for e in examples),
                                "b_target_tokens": goal, "match": {k: v for k, v in led.items() if k not in ("chosen", "ledger")},
                                "answer_accuracy": sum(graded) / len(graded) if graded else None, "ledger": led["ledger"]}
    return out, stats


def build_all(raw: dict, scale: Scale, *, seed: int = 0, keep_per_prompt: int = 1, max_tokens: int = MAX_SEQ_TOKENS,
              count_tokens: Callable[[str], int] = approx_tokens) -> tuple:
    """All corpora from raw teacher rows. raw = {"greedy": {arm: rows}, "sampled": {arm: rows}, "cot": {arm: rows}, "para": [rows]}
    where `greedy` rows may mix episode records and CoT singles (split on `kind`). Returns ({corpus: {teacher: examples}}, stats).
    Every returned example has passed `assert_targets`."""
    greedy_eps = {t: [r for r in raw["greedy"][t] if "turns" in r] for t in scale.teachers}
    cot_rows = {t: [r for r in (raw.get("cot") or raw["greedy"])[t] if "turns" not in r] for t in scale.teachers}
    every = [r for t in scale.teachers for r in greedy_eps[t] + list(raw["sampled"][t]) + cot_rows[t]]
    efforts = {effort_label(r.get("reasoning_effort")) for r in every if "reasoning_effort" in r}
    caps = {int(r["max_new_tokens"]) for r in every if "max_new_tokens" in r}
    if len(efforts) > 1 or len(caps) > 1:
        raise ValueError(f"teacher rollouts must share one reasoning_effort and one cap (prompts embed the effort), got {sorted(efforts)}, {sorted(caps)}")
    corp, stats = {}, {"scale": scale_dict(scale), "seed": seed, "keep_per_prompt": keep_per_prompt, "max_seq_tokens": max_tokens,
                       "effort": next(iter(efforts), None), "cap": next(iter(caps), None)}
    corp["A"], stats["A"] = build_corpus_a(greedy_eps, scale, seed=seed, max_tokens=max_tokens, count_tokens=count_tokens)
    corp["B"], stats["B"] = build_corpus_b(raw["sampled"], scale, keep_per_prompt=keep_per_prompt, seed=seed,
                                          max_tokens=max_tokens, count_tokens=count_tokens)
    corp["C"], stats["C"] = build_corpus_c(cot_rows, corp["B"], seed=seed, max_tokens=max_tokens, count_tokens=count_tokens)
    if raw.get("para"):
        corp["Bp"], stats["Bp"] = build_corpus_bprime(corp["B"], raw["para"], max_tokens=max_tokens, count_tokens=count_tokens)
    for c in corp.values():
        for examples in c.values():
            assert_targets(examples)
            ids = [e["example_id"] for e in examples]
            if len(set(ids)) != len(ids):
                raise ValueError("duplicate example_id in a corpus")
    # content hashes tie the SFT files to the raw transcripts: the verifier recomputes them, the builder writes files with the same bytes
    stats["digests"] = {c: {t: corpus_digest(ex) for t, ex in sorted(by_t.items())} for c, by_t in sorted(corp.items())}
    stats["empty"] = [f"{c}.{t}" for c, by_t in sorted(corp.items()) for t, ex in sorted(by_t.items()) if not ex]
    return corp, stats


def scale_dict(scale: Scale) -> dict:
    return {"corpus_ranks": [list(x) for x in scale.corpus_ranks], "teachers": list(scale.teachers), "seeds": list(scale.seeds),
            "max_prompts": scale.max_prompts, "later_turns": scale.later_turns, "epochs": scale.epochs,
            "n_samples": scale.n_samples, "n_cot_prompts": scale.n_cot_prompts, "floors": scale.floors, "shards": scale.shards}


def scale_from_dict(d: dict) -> Scale:
    return replace(Scale(), **{**d, "corpus_ranks": tuple(tuple(x) for x in d["corpus_ranks"]),
                               "teachers": tuple(d["teachers"]), "seeds": tuple(d["seeds"])})


# ================================================================================ student bundle (local -> artifacts volume)
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonl_text(rows: list) -> str:
    return "".join(json.dumps(r) + "\n" for r in rows)


def corpus_digest(examples: list) -> str:
    """sha256 of the corpus file's bytes (one JSON object per line)."""
    return hashlib.sha256(jsonl_text(examples).encode()).hexdigest()


def _write_jsonl(path: pathlib.Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(jsonl_text(rows))


def write_bundle(out_dir, corpora: dict, eval_insts: list, refusal: list, meta: dict) -> dict:
    """Write everything a student container needs into `out_dir` and return its manifest (sha256 per file):
    corpora/<corpus>.<teacher>.jsonl (checked with assert_targets again), eval_instances.jsonl, refusal_items.jsonl.
    The bundle id is the hash of the manifest, so identical inputs give an identical directory name."""
    out_dir = pathlib.Path(out_dir)
    files = {}
    empty = [f"{c}.{t}" for c, by_teacher in corpora.items() for t, examples in by_teacher.items() if not examples]
    if empty:
        raise ValueError(f"empty corpora {empty}: nothing to train on (e.g. no prompt survived the Corpus B intersection)")
    for corpus, by_teacher in corpora.items():
        for teacher, examples in by_teacher.items():
            assert_targets(examples)
            rel = f"corpora/{corpus}.{teacher}.jsonl"
            _write_jsonl(out_dir / rel, examples)
            files[rel] = sha256_file(out_dir / rel)
    for rel, rows in (("eval_instances.jsonl", eval_insts), ("refusal_items.jsonl", refusal)):
        _write_jsonl(out_dir / rel, rows)
        files[rel] = sha256_file(out_dir / rel)
    manifest = {"files": dict(sorted(files.items())), "meta": meta}
    manifest["bundle_id"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return manifest


def load_bundle(bundle_dir, expect_id: str | None = None) -> dict:
    """Read a bundle, verifying every file hash (and the bundle id when given). Returns
    {"manifest", "corpora": {(corpus, teacher): examples}, "eval_instances", "refusal_items"}."""
    root = pathlib.Path(bundle_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    if expect_id is not None and manifest["bundle_id"] != expect_id:
        raise ValueError(f"bundle id {manifest['bundle_id']} != expected {expect_id}")
    for rel, digest in manifest["files"].items():
        if sha256_file(root / rel) != digest:
            raise ValueError(f"bundle file {rel} fails its sha256")
    read = lambda rel: [json.loads(l) for l in (root / rel).read_text().splitlines() if l.strip()]
    corpora = {}
    for rel in manifest["files"]:
        if rel.startswith("corpora/"):
            corpus, teacher = rel[len("corpora/"):-len(".jsonl")].split(".", 1)
            corpora[(corpus, teacher)] = read(rel)
    return {"manifest": manifest, "corpora": corpora, "eval_instances": read("eval_instances.jsonl"),
            "refusal_items": read("refusal_items.jsonl")}
