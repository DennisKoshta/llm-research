"""Exp 1.2 data pipeline (CPU only): teacher request files -> SFT corpora A / B / B' / C from raw teacher transcripts.

  requests   (--effort low|medium|xhigh|default, default = records/phase1_effort_decision.json else low; --cap 2048)
             data/exp12/requests_teacher_greedy.jsonl  greedy TRAIN-template episodes (Corpus A) + math/coding CoT singles (Corpus C)
             data/exp12/requests_teacher_t07.jsonl     n_samples T=0.7 episodes per prompt (Corpus B)
             data/exp12/eval_instances.jsonl, refusal_items.jsonl   what the students are evaluated on (TEST templates, Set C)
  build      raw_transcripts/exp12_teacher_{greedy,t07}/<teacher>.jsonl -> data/exp12/corpora/<corpus>.<teacher>.jsonl
             (+ data/exp12/token_ledger.json: the B-vs-C token ledger per teacher; data/exp12/requests_para.jsonl for the Llama
             B' paraphrases; records/exp12_corpora.json); run it a second time
             after raw_transcripts/exp12_para/llama70b.jsonl exists to add Corpus B'
  status     row counts of the raw inputs

Full sequence (bases = data/author/final_bases.jsonl from scripts/run_author.py; family = the Phase 1 decision, r1 or r2):

  PYTHONPATH=src .venv/bin/python scripts/build_exp12_corpora.py requests
  PYTHONPATH=src .venv/bin/python scripts/run_requests.py --exp-id exp12_teacher_greedy --family r1 \\
      --requests data/exp12/requests_teacher_greedy.jsonl --arms intact,refusal_ablated,pca_101 --gpu H200 --est-hours 0.5 --max-num-seqs 512 --eager 0
  PYTHONPATH=src .venv/bin/python scripts/run_requests.py --exp-id exp12_teacher_t07 --family r1 \\
      --requests data/exp12/requests_teacher_t07.jsonl --arms intact,refusal_ablated,pca_101 --gpu H200 --est-hours 0.8 --max-num-seqs 512 --eager 0 \\
      --temperature 0.7 --top-p 0.95 --seed-base 12
  PYTHONPATH=src .venv/bin/python scripts/build_exp12_corpora.py build            # A, B, C + requests_para.jsonl
  PYTHONPATH=src .venv/bin/python scripts/run_requests.py --exp-id exp12_para \\
      --requests data/exp12/requests_para.jsonl --arms llama70b --model-kind llama --gpu H200 --est-hours 0.4 --max-model-len 8192 --max-num-seqs 256 --eager 0
  PYTHONPATH=src .venv/bin/python scripts/build_exp12_corpora.py build            # adds B'

Contract with scripts/verify_all_receipts.py: `recompute(raw, params)` re-derives records/exp12_corpora.json["summary"] from
the raw transcripts alone; `load_raw()` reads them; `RECORD_NAME` names the record. Prints only counts, never text.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v3 import exp12, scenarios  # noqa: E402

RECORD_NAME = "exp12_corpora"
RAW = {"greedy": "exp12_teacher_greedy", "sampled": "exp12_teacher_t07", "para": "exp12_para"}
DATA = ROOT / "data" / "exp12"
SET_C = ROOT / "data" / "phase1" / "requests_set_c.jsonl"
DEFAULT_BASES = ROOT / "data" / "author" / "final_bases.jsonl"
EFFORT_DECISION = ROOT / "records" / "phase1_effort_decision.json"
FAMILY_DECISION = ROOT / "records" / "phase1_family_decision.json"     # reasoning_effort_for_phase2, the file v3.exp11 follows


def read_jsonl(path) -> list:
    return [json.loads(line) for line in pathlib.Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows) -> str:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(r) + "\n" for r in rows)
    path.write_text(body)
    return hashlib.sha256(body.encode()).hexdigest()


def exp11_greedy_episodes(rows: list) -> list:
    """Exp 1.1 greedy cot_on main-grid rollouts that Corpus A can reuse: TRAIN templates, high-cost cells."""
    return [r for r in rows if "turns" in r and r.get("family") == "main" and r.get("cot_mode") == "cot_on"
            and r["template_id"] in scenarios.TRAIN_TEMPLATE_IDS and r["cost_regime"] in exp12.HIGH_COST]


def load_raw(raw_root=ROOT / "raw_transcripts", greedy_from: str | None = None) -> dict:
    """{"greedy": {teacher: rows}, "sampled": {teacher: rows}, "para": rows} from raw_transcripts/ (missing inputs are omitted).
    With `greedy_from` (a raw dir such as raw_transcripts/exp11 holding <teacher>.jsonl), Corpus A's greedy episodes come from
    that Exp 1.1 run (filtered by exp11_greedy_episodes) and exp12_teacher_greedy contributes only the Corpus C CoT singles."""
    raw: dict = {}
    for key, exp in RAW.items():
        d = pathlib.Path(raw_root) / exp
        files = {p.stem: p for p in sorted(d.glob("*.jsonl"))} if d.exists() else {}
        if key == "para":
            if "llama70b" in files:
                raw["para"] = read_jsonl(files["llama70b"])
        elif files:
            raw[key] = {t: read_jsonl(p) for t, p in files.items() if t in exp12.TEACHERS}
    if greedy_from:
        d = pathlib.Path(greedy_from)
        for t in exp12.TEACHERS:
            if (d / f"{t}.jsonl").exists():
                own = [r for r in raw.get("greedy", {}).get(t, []) if "turns" not in r]
                raw.setdefault("greedy", {})[t] = exp11_greedy_episodes(read_jsonl(d / f"{t}.jsonl")) + own
    return raw


def params_dict(scale: exp12.Scale, seed: int, keep_per_prompt: int, tokenizer: str | None, greedy_from: str | None = None,
                max_seq_tokens: int = exp12.MAX_SEQ_TOKENS) -> dict:
    return {"scale": exp12.scale_dict(scale), "seed": seed, "keep_per_prompt": keep_per_prompt, "tokenizer_json": tokenizer,
            "greedy_from": greedy_from, "max_seq_tokens": max_seq_tokens}


def recompute(raw: dict | None, params: dict | None = None) -> dict:
    """The corpus statistics ("summary") from the raw teacher transcripts alone. `params` = the record's "params";
    `raw=None` reads raw_transcripts/ itself (including params["greedy_from"])."""
    params = params or params_dict(exp12.Scale(), 0, 1, None)
    if raw is None:
        raw = load_raw(greedy_from=params.get("greedy_from"))
    count = exp12.tokenizer_counter(params["tokenizer_json"]) if params.get("tokenizer_json") else exp12.approx_tokens
    _, stats = exp12.build_all(raw, exp12.scale_from_dict(params["scale"]), seed=params["seed"], keep_per_prompt=params["keep_per_prompt"],
                               max_tokens=params.get("max_seq_tokens", exp12.MAX_SEQ_TOKENS), count_tokens=count)
    stats["token_counter"] = "tokenizers:" + params["tokenizer_json"] if params.get("tokenizer_json") else "approx_chars/3"
    return stats


def load_bases(path, placeholder: bool) -> list:
    return scenarios.make_placeholder_bases(40) if placeholder else scenarios.load_bases(path)


def cmd_requests(a) -> int:
    out = pathlib.Path(a.out)
    if a.placeholder and out.resolve() == DATA.resolve():
        raise SystemExit("--placeholder bases must not be written to data/exp12 (pass --out)")
    scale = exp12.PRESETS[a.preset] if a.preset else exp12.Scale()
    bases = load_bases(a.bases, a.placeholder)
    train = exp12.train_instances(bases)
    label = a.effort or exp12.load_effort_decision(EFFORT_DECISION)
    effort = exp12.resolve_effort(label)
    for note in exp12.effort_notes(label, a.cap, (EFFORT_DECISION, FAMILY_DECISION)):
        print(note)
    greedy = ([] if a.greedy_from_exp11 else exp12.teacher_greedy_requests(train, effort=effort, cap=a.cap)) \
        + exp12.cot_requests(exp12.math_coding_prompts(a.n_cot or scale.n_cot_prompts, a.seed), effort=effort, cap=a.cap)
    sampled = exp12.teacher_sampled_requests(train, a.n_samples or scale.n_samples, effort=effort, cap=a.cap)
    ev = exp12.eval_instances(bases, exp12.ALL_CELLS if a.eval_cells == "all" else exp12.EVAL_CELLS)
    refusal = exp12.refusal_items(read_jsonl(a.set_c))
    hashes = {name: write_jsonl(out / name, rows) for name, rows in (
        ("requests_teacher_greedy.jsonl", greedy), ("requests_teacher_t07.jsonl", sampled),
        ("eval_instances.jsonl", ev), ("refusal_items.jsonl", refusal))}
    est = exp12.estimate_cost(scale, n_train_bases=len(train) // len(exp12.TRAIN_CELLS), eval_episodes=len(ev), effort=effort)
    (out / "requests_manifest.json").write_text(json.dumps({"sha256": hashes, "n_train_prompts": len(train), "n_greedy_rows": len(greedy),
                                                            "n_sampled_rows": len(sampled), "n_eval_instances": len(ev), "eval_cells": a.eval_cells,
                                                            "greedy_from_exp11": a.greedy_from_exp11,
                                                            "n_refusal_items": len(refusal), "placeholder": a.placeholder,
                                                            "effort": label, "cap": a.cap}, indent=1))
    n_greedy_eps = sum(r["kind"] == "episode" for r in greedy)
    print(f"effort={label} cap={a.cap}; train prompts={len(train)} (greedy episodes {n_greedy_eps} + CoT singles {len(greedy) - n_greedy_eps}; "
          f"T=0.7 episodes {len(sampled)}); eval episodes={len(ev)}; refusal items={len(refusal)}; files in {out}")
    print(f"teacher-stage estimate: ${sum(est['stages'][k]['usd'] for k in list(est['stages'])[:3]):.2f} "
          f"(students: ${est['stages']['students']['usd']:.2f} for preset {a.preset or 'full72'})")
    return 0


def cmd_build(a) -> int:
    scale = exp12.PRESETS[a.preset] if a.preset else exp12.Scale()
    if a.max_prompts is not None or a.later_turns is not None:
        scale = dataclasses.replace(scale, **{k: v for k, v in (("max_prompts", a.max_prompts), ("later_turns", a.later_turns)) if v is not None})
    raw = load_raw(pathlib.Path(a.raw_root), a.greedy_from)
    for need in ("greedy", "sampled"):
        if need not in raw or not set(scale.teachers) <= set(raw[need]):
            raise SystemExit(f"missing raw teacher transcripts for '{need}' (have {sorted(raw.get(need, {}))}); see the module docstring")
    count = exp12.tokenizer_counter(a.tokenizer) if a.tokenizer else exp12.approx_tokens
    corpora, stats = exp12.build_all(raw, scale, seed=a.seed, keep_per_prompt=a.keep_per_prompt, max_tokens=a.max_seq_tokens, count_tokens=count)
    stats["token_counter"] = "tokenizers:" + a.tokenizer if a.tokenizer else "approx_chars/3"
    out = pathlib.Path(a.out)
    files = {}
    for corpus, by_teacher in corpora.items():
        for teacher, examples in by_teacher.items():
            rel = f"corpora/{corpus}.{teacher}.jsonl"
            files[rel] = write_jsonl(out / rel, examples)
    if "para" not in raw:
        reqs = exp12.paraphrase_requests(corpora["B"])
        files["requests_para.jsonl"] = write_jsonl(out / "requests_para.jsonl", reqs)
        print(f"wrote {len(reqs)} paraphrase requests (run Llama, then build again for Corpus B')")
    (out / "token_ledger.json").write_text(json.dumps(
        {t: {k: v for k, v in st.items() if k in ("b_target_tokens", "target_tokens", "match", "ledger")} for t, st in stats["C"]["teachers"].items()}))
    params = params_dict(scale, a.seed, a.keep_per_prompt, a.tokenizer, a.greedy_from, a.max_seq_tokens)
    dirs = [pathlib.Path(a.raw_root) / e for e in RAW.values()] + ([pathlib.Path(a.greedy_from)] if a.greedy_from else [])
    provenance = {f"{d.name}/{p.name}": hashlib.sha256(p.read_bytes()).hexdigest() for d in dirs if d.exists() for p in sorted(d.glob("*.jsonl"))}
    record = {"schema": "exp12_corpora_v1", "params": params, "summary": stats, "files": files, "raw_sha256": provenance}
    (out / "corpora_manifest.json").write_text(json.dumps(record["params"] | {"files": files, "effort": stats["effort"], "cap": stats["cap"], "max_seq_tokens": a.max_seq_tokens}, indent=1))
    rec_path = pathlib.Path(a.record) if a.record else ROOT / "records" / f"{RECORD_NAME}.json"
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    rec_path.write_text(json.dumps(record, indent=1))
    print_summary(stats)
    print(f"wrote {rec_path}")
    if stats["empty"]:
        print(f"ERROR: empty corpora {stats['empty']}: no prompt survived the filters (see the intersection and acceptance numbers above); "
              f"the students cannot train on them")
        return 2
    return 0


def pct(x) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def print_summary(stats: dict) -> None:
    b = stats["B"]
    inter = b["intersection"]
    print(f"Corpus B: intersection {inter['n_prompts']}/{inter['n_prompts_all_teachers']} prompts "
          f"({pct(inter['fraction'])}, gate>=50%: {inter['gate_ge_50pct']}); selected {inter['n_selected']}")
    for t, s in b["teachers"].items():
        print(f"  {t}: clean samples {s['n_clean_samples']}/{s['n_samples']} ({pct(s['sample_acceptance'])}), prompts with a clean sample "
              f"{s['n_prompts_with_clean']}/{s['n_prompts']} ({pct(s['prompt_acceptance'])}), B examples {s['n_examples']}, target tokens {s['target_tokens']}")
    for t, s in stats["A"]["teachers"].items():
        print(f"Corpus A {t}: episodes {s['n_episodes']} examples {s['n_examples']} ({s['n_text_only_turns']} text-only) outcomes {s['outcomes']} dropped {s['dropped_turns']}")
    for t, s in stats["C"]["teachers"].items():
        print(f"Corpus C {t}: {s['n_examples']} examples, target tokens {s['target_tokens']} vs B {s['b_target_tokens']} "
              f"(error {s['match']['abs_error']}, tol {s['match']['tol']}, within: {s['match']['within_tol']}), answer accuracy {s['answer_accuracy']}")
    if "Bp" in stats:
        for t, s in stats["Bp"]["teachers"].items():
            lost = {c: v["in"] - v["kept"] for c, v in s["retention_by_cell"].items() if v["in"] != v["kept"]}
            print(f"Corpus B' {t}: {s['n_examples']}/{s['n_in']} paraphrased, dropped {s['dropped']}, lost by cell {lost or 'none'}")


def cmd_status(a) -> int:
    raw = load_raw(pathlib.Path(a.raw_root))
    for key in ("greedy", "sampled"):
        print(key, {t: len(rows) for t, rows in raw.get(key, {}).items()} or "missing")
    print("para", len(raw["para"]) if "para" in raw else "missing")
    corp = sorted((DATA / "corpora").glob("*.jsonl")) if (DATA / "corpora").exists() else []
    print("corpora files:", [p.name for p in corp] or "none")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", default=str(DATA))
    common.add_argument("--preset", "--scale", dest="preset", choices=sorted(exp12.PRESETS), default=None, help="Scale preset (default full72)")
    common.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("requests", parents=[common])
    r.add_argument("--bases", default=str(DEFAULT_BASES))
    r.add_argument("--placeholder", action="store_true", help="placeholder bases (dry run into a scratch --out only)")
    r.add_argument("--set-c", default=str(SET_C))
    r.add_argument("--n-samples", type=int, default=None)
    r.add_argument("--n-cot", type=int, default=None)
    r.add_argument("--eval-cells", choices=["core", "all"], default="core",
                   help="core = 7 cells (primary + controls, 112 episodes); all = the full 12-cell grid on the TEST templates (192)")
    r.add_argument("--greedy-from-exp11", action="store_true",
                   help="do not request Corpus A greedy episodes: reuse the Exp 1.1 rollouts (build --greedy-from raw_transcripts/exp11)")
    r.add_argument("--effort", choices=exp12.EFFORT_CHOICES, default=None,
                   help="reasoning effort of every teacher / student prompt; default = records/phase1_effort_decision.json, else low")
    r.add_argument("--cap", type=int, default=exp12.CAP, help="max_new_tokens for teacher rollouts (>= 2048; raise with --max-seq-tokens at xhigh)")
    b = sub.add_parser("build", parents=[common])
    b.add_argument("--raw-root", default=str(ROOT / "raw_transcripts"))
    b.add_argument("--record", default=None)
    b.add_argument("--max-prompts", type=int, default=None)
    b.add_argument("--later-turns", type=int, default=None)
    b.add_argument("--keep-per-prompt", type=int, default=1)
    b.add_argument("--max-seq-tokens", type=int, default=exp12.MAX_SEQ_TOKENS,
                   help="prompt+target length cap; longer examples are dropped whole (counted). Raise it with --cap at xhigh effort")
    b.add_argument("--greedy-from", default=None, help="raw dir of an Exp 1.1 run (e.g. raw_transcripts/exp11) whose greedy rollouts become Corpus A")
    b.add_argument("--tokenizer", default=None, help="tokenizer.json for exact length / token counts (needs the `tokenizers` package)")
    s = sub.add_parser("status")
    s.add_argument("--raw-root", default=str(ROOT / "raw_transcripts"))
    a = ap.parse_args(argv)
    return {"requests": cmd_requests, "build": cmd_build, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
