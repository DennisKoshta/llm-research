"""Build the Exp 2.3 staged request files (data/exp23/) from raw_transcripts/. Prints counts and problem CODES only:
authored scenario text, memos, reviewer notes and plans live unredacted in the JSONL files and are never printed.

Run the stages in order; after each Modal run (scripts/run_requests.py, see `commands`) call the next builder. Every
builder re-assembles the stimulus bundle from raw_transcripts/exp23_*/ (v3.exp23.assemble), so retries just re-run it.

  cost [--scale S ...]        generations / tokens / GPU-hours / USD for the whole pipeline (no files written)
  s0-author [--round N] [--speculative]
                              author requests; round > 1 = only scenarios without an accepted set, with problem codes.
                              --speculative also authors an extra candidate for scenarios whose set still awaits a judge
                              verdict (so a failed verdict finds the next candidate ready: fewer sequential cycles)
  s0-judge  [--round N] [--parallel]
                              numeric judge requests: per scenario the first structurally valid author attempt without a
                              parsed verdict (a set re-authored after a failed verdict is judged too; a re-run of an
                              unparseable verdict gets a fresh round, hence a fresh exp id). --parallel judges EVERY
                              structurally valid attempt without a parsed verdict (until a passing attempt exists)
  s1-plan   [--attempt N]     planner requests (harmful and harmless state)
  s2-review [--attempt N]     reviewer requests B1 / B2 / R1a / R1b (needs usable harmful-state plans)
  s3-aux    [--attempt N]     Llama paraphrases (3 per scenario) and clause-extraction requests
  s3-judge  [--round N]       fidelity-judge requests for the paraphrases without a parsed verdict
  s4-exec --executor intact|llama|flash [--orders 6|3] [--shard-rows N]
  status                      coverage of every stage (counts, problem codes)
  commands [--file SUBSTR]    the exact run_requests.py invocations for the request files in data/exp23/

Retries (attempt / round > 1) are launched with seeded sampling (exp23.RETRY_SAMPLING, --seed-base = attempt): a greedy
re-run of an identical prompt returns the same output, so only a truncation could ever be cured by it.

One exp id per request file (run_requests.py sends every row of a file whose arm matches --arms, whatever the row's exp id),
so a builder call that spans several exp ids (retry attempts, judge rounds) writes `<stem>__<exp_id>.jsonl` files. A builder
refuses to emit rows whose exp id already holds raw rows (the launcher would overwrite them): take the next --round /
--attempt, or pass the global --force for a deliberate rebuild.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v3 import exp23  # noqa: E402

D = ROOT / "data" / "exp23"
RAW = ROOT / "raw_transcripts"
ARM_RUN = {"flash_next_uncensored": ("H200:2", "flash"), "flash_next": ("H200:2", "flash"), "llama70b": ("H200", "llama")}
RUN_FLAGS = {  # engine flags per run_requests.py invocation (27B: CUDA graphs measured ~3x faster than eager)
    "qwen27": "--eager 0 --max-num-seqs 384 --max-model-len 16384",
    "flash": "--eager 1 --max-model-len 16384 --extra-json '{\"safetensors_load_strategy\":\"eager\"}'",
    "llama": "--eager 0 --max-num-seqs 256 --max-model-len 8192 --model-kind llama",
}
S4_KEY = {"intact": "intact", "llama70b": "llama", "flash_next": "flash"}


def guard(rows: list[dict], a) -> None:
    """Refuse to emit rows whose exp id already holds raw rows: run_requests.py writes raw_transcripts/<exp_id>/<arm>.jsonl
    fresh, so launching the rebuilt file would overwrite stored generations (I3). Use the next --round / --attempt;
    --force only for a deliberate rebuild."""
    if getattr(a, "force", False):
        return
    taken = sorted({r["exp_id"] for r in rows if any(not p.name.endswith(".partial.jsonl") for p in (RAW / r["exp_id"]).glob("*.jsonl"))})
    if taken:
        raise SystemExit(f"raw_transcripts already holds {taken}: use the next --round / --attempt (or --force)")


def write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    D.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    path.chmod(0o600)  # request files hold unredacted authored text


def selected(a) -> list[str]:
    return exp23.select_scenarios(exp23.n_per_domain_for(a.scale))


def load_bundle(require_judge: bool = True) -> tuple[dict, list]:
    rows = exp23.read_rows(RAW, exclude=("exp23_s4_",))  # executor outputs are not inputs to any builder
    return exp23.assemble(rows, require_judge=require_judge), rows


def emit(name: str, stem: str, rows: list[dict], a) -> None:
    """Guard, then write the request file(s). run_requests.py sends EVERY row of a file whose arm is listed, whatever its
    exp id, so rows of different exp ids (retry attempts, judge rounds) go to separate files: one exp id per file."""
    guard(rows, a)
    by_exp: dict[str, list] = collections.defaultdict(list)
    for r in rows:
        by_exp[r["exp_id"]].append(r)
    if len(by_exp) <= 1:
        write_jsonl(D / f"{stem}.jsonl", rows)
        report(name, rows, D / f"{stem}.jsonl")
        return
    for exp_id, part in by_exp.items():
        write_jsonl(D / f"{stem}__{exp_id}.jsonl", part)
        report(name, part, D / f"{stem}__{exp_id}.jsonl")


def report(name: str, rows: list[dict], path: pathlib.Path) -> None:
    arms = dict(sorted(collections.Counter(r["arm"] for r in rows).items()))
    exps = dict(sorted(collections.Counter(r["exp_id"] for r in rows).items()))
    print(f"{name}: {len(rows)} rows arms={arms} exp_ids={exps} -> {path}")


# ---- stage builders -------------------------------------------------------------------------------------------------
def cmd_cost(a) -> None:
    est = exp23.estimate_cost(a.scale, orders_primary=a.orders_primary, orders_secondary=a.orders_secondary,
                              effort=a.effort, secondary_conditions=a.secondary_conditions, pessimism=a.pessimism)
    for s in est["stages"]:
        print(f"{s['stage']:34s} {s['gpu']:8s} x{s['containers']} gens={s['generations']:>7} gen_tok={s['gen_tokens']:>9} "
              f"prompt_tok={s['prompt_tokens']:>9} wall_h={s['wall_hours']:6.3f} usd={s['usd']:7.2f}")
    t = est["totals"]
    print(f"TOTAL gens={t['generations']:.0f} gen_tok={t['gen_tokens']:.0f} prompt_tok={t['prompt_tokens']:.0f} "
          f"gpu_h={t['gpu_hours']:.2f} usd={t['usd']:.2f}  (scale={a.scale} -> {est['n_scenarios']} scenarios, "
          f"orders primary/secondary={a.orders_primary}/{a.orders_secondary}, effort={a.effort}, pessimism={a.pessimism})")


def cmd_s0_author(a) -> None:
    sids = selected(a)
    bundle, _ = load_bundle()
    rows = []
    for sid in sids:
        idx = exp23.LAYOUT_BY_ID[sid]["author_index"]
        if a.round == 1:
            rows.append(exp23.author_request(idx, 1))
            continue
        st = bundle["author_status"].get(sid)
        if sid in bundle["bases"]:
            continue
        if getattr(a, "speculative", False):  # extra candidate also for a scenario whose set still awaits its verdict
            codes = [c for c in ((st or {}).get("last_problems") or []) if not c.startswith("judge_")]  # judge_<STATUS> = unparsed verdict
            rows.append(exp23.author_request(idx, a.round, feedback=codes or None, nudge=list(exp23.SPECULATIVE_NUDGE)))
            continue
        if st and st["state"] == "awaiting_judge":
            continue
        rows.append(exp23.author_request(idx, a.round, feedback=(st or {}).get("last_problems") or None))
    emit("s0-author", f"requests_s0_a{a.round}", rows, a)


def cmd_s0_judge(a) -> None:
    """Judge rows for the first structurally valid author attempt of every scenario that has no passing verdict: a set
    re-authored after a failed verdict is judged in its own right, an unparseable verdict is re-run in a fresh round."""
    g = exp23.group_rows(exp23.read_rows(RAW, exclude=("exp23_s4_",)))
    rows = [exp23.judge_request(b, b["attempt"], exp23.next_judge_round(g.get("s0j", []), b["scenario_id"], b["attempt"], a.round))
            for b in exp23.judge_candidates(g.get("s0", []), g.get("s0j", []), parallel=getattr(a, "parallel", False))]
    emit("s0-judge", f"requests_s0j_j{a.round}", rows, a)


def cmd_s1(a) -> None:
    bundle, _ = load_bundle()
    rows = exp23.planner_requests(bundle, selected(a), a.effort, a.attempt)
    emit("s1-plan", f"requests_s1_a{a.attempt}", rows, a)


def cmd_s2(a) -> None:
    bundle, _ = load_bundle()
    rows = exp23.reviewer_requests(bundle, selected(a), a.effort, a.attempt)
    emit("s2-review", f"requests_s2_a{a.attempt}", rows, a)


def cmd_s3(a) -> None:
    bundle, _ = load_bundle()
    rows = exp23.paraphrase_requests(bundle, selected(a), a.attempt) + exp23.clause_requests(bundle, selected(a), a.attempt)
    emit("s3-aux", f"requests_s3_a{a.attempt}", rows, a)


def cmd_s3_judge(a) -> None:
    """Fidelity-judge rows for the first acceptable paraphrase attempt of every (scenario, variant) without a verdict."""
    bundle, rows_all = load_bundle(require_judge=False)
    g = exp23.group_rows(rows_all)
    keep = set(selected(a))
    s3p = [r for r in g.get("s3p", []) if r["meta"]["scenario_id"] in keep]
    rows = exp23.fidelity_requests(bundle, s3p, g.get("s3j", []), rnd=a.round)
    emit("s3-judge", f"requests_s3j_j{a.round}", rows, a)


def cmd_s4(a) -> None:
    bundle, _ = load_bundle()
    n_orders = a.orders or (a.orders_primary if a.executor == "intact" else a.orders_secondary)
    conditions = exp23.CORE_CONDITIONS if (a.executor != "intact" and a.secondary_conditions == "core") else exp23.CONDITIONS
    rows = exp23.executor_requests(bundle, a.executor, sids=selected(a), n_orders=n_orders, effort=a.effort,
                                   conditions=conditions)
    shards = max(1, math.ceil(len(rows) / a.shard_rows))
    per = math.ceil(len(rows) / shards) if rows else 0
    if shards > 1:  # one run_requests invocation per shard needs its own exp id (output path and ledger job)
        for k in range(shards):
            for r in rows[k * per:(k + 1) * per]:
                r["exp_id"] = f"{r['exp_id']}_p{k}"
    guard(rows, a)
    for old in D.glob(f"requests_s4_{a.executor}_p*.jsonl"):  # stale shards of an earlier build (after the guard: nothing is lost)
        old.unlink()
    for k in range(shards):
        part = rows[k * per:(k + 1) * per]
        path = D / f"requests_s4_{a.executor}_p{k}.jsonl"
        write_jsonl(path, part)
        report(f"s4-exec[{a.executor}] part {k}", part, path)
    c = collections.Counter((r["cot_mode"], r["meta"]["framing"]) for r in rows)
    print(f"s4-exec[{a.executor}]: {len(rows)} rows, orders per cell={n_orders}, scenarios={len({r['meta']['scenario_id'] for r in rows})}, "
          f"by (cot_mode, framing)={dict(sorted(c.items()))}")


def cmd_status(a) -> None:
    bundle, rows = load_bundle()
    by_stage = collections.Counter(r["meta"]["stage"] for r in rows)
    print("raw rows by stage:", dict(sorted(by_stage.items())))
    st = bundle["author_status"]
    print(f"s0: scenarios seen={len(st)} accepted={len(bundle['bases'])} states={dict(collections.Counter(v['state'] for v in st.values()))}")
    codes = collections.Counter(c.split(':')[0] + (':' + c.split(':')[1] if c.startswith('judge:') else '')
                                for v in st.values() if v["state"] == "rejected" for c in v["last_problems"])
    print("s0 unresolved problem codes:", dict(sorted(codes.items())))
    for name, status in (("s1 plan", bundle["plan_status"]), ("s2 review", bundle["review_status"]),
                         ("s3p paraphrase", bundle["paraphrase_status"]), ("s3c clause", bundle["clause_status"])):
        print(f"{name}: {dict(sorted(collections.Counter(status.values()).items()))}")
    print("condition support (scenarios per condition):", {c: len(v) for c, v in bundle["notes"].items()})
    print("plans:", {c: sum(1 for k in bundle['plans'] if k[1] == c) for c in ('harmful', 'harmless')})
    print("clause-grid scenarios:", len(bundle["clauses"]))


def stage_label(meta_stage: str, arm: str) -> str:
    if meta_stage == "s4":
        return f"s4 executor {S4_KEY[arm]}"
    if meta_stage == "s2":
        return "s2b review flash (R1b)" if arm == "flash_next_uncensored" else "s2a review 27B (B1,B2,R1a)"
    return {"s0": "s0 author", "s0j": "s0j judge", "s1": "s1 plan", "s3p": "s3 paraphrase+clauses",
            "s3c": "s3 paraphrase+clauses", "s3j": "s3j fidelity judge"}[meta_stage]


def cmd_commands(a) -> None:
    """One run_requests.py line per (request file, exp id): the exp ids, arms, GPU and engine flags come from the
    rows themselves, so retries (_aN) and shards (_pK) print correctly. --est-hours = 1.5x the estimator share + 0.1."""
    est = exp23.estimate_cost(a.scale, orders_primary=a.orders_primary, orders_secondary=a.orders_secondary, effort=a.effort,
                              secondary_conditions=a.secondary_conditions, pessimism=a.pessimism)
    stage = {s["stage"]: s for s in est["stages"]}
    print("# run in stage order; wait for each stage to finish and call the next builder in between")
    for f in sorted(D.glob("requests_*.jsonl")):
        if a.file and a.file not in f.name:
            continue
        rows = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
        groups: dict[tuple, list] = collections.defaultdict(list)
        for r in rows:
            gpu, model = ARM_RUN.get(r["arm"], ("H200", "qwen27"))
            groups[(r["exp_id"], gpu, model)].append(r)
        arm_exps = collections.defaultdict(set)
        for r in rows:
            arm_exps[r["arm"]].add(r["exp_id"])
        shared = sorted(arm for arm, exps in arm_exps.items() if len(exps) > 1)
        if shared:  # run_requests.py would send every row of the file for that arm under each exp id
            raise SystemExit(f"{f.name}: arm(s) {shared} appear under several exp ids; rebuild with the current builder")
        for (exp_id, gpu, model), rs in sorted(groups.items()):
            st = stage[stage_label(rs[0]["meta"]["stage"], rs[0]["arm"])]
            arm_list = sorted({r["arm"] for r in rs})
            frac = min(1.0, len(rs) / len(arm_list) / max(1, st["generations"]))  # --est-hours is PER ARM (one container each)
            compute_h = st["wall_hours"] - st["containers"] * st["boot_hours"]
            est_h = max(0.3, round(1.5 * (st["boot_hours"] + frac * compute_h) + 0.1, 1))
            arms = ",".join(arm_list)
            retry = max(max(r["meta"].get("attempt", 1), r["meta"].get("round", 1)) for r in rs)
            sampling = (f" --temperature {exp23.RETRY_SAMPLING['temperature']} --top-p {exp23.RETRY_SAMPLING['top_p']}"
                        f" --seed-base {retry}") if retry > 1 else ""  # seeded sampling, see exp23.RETRY_SAMPLING
            print(f"PYTHONPATH=src .venv/bin/python scripts/run_requests.py --exp-id {exp_id} --requests data/exp23/{f.name} "
                  f"--arms {arms} --gpu {gpu} --est-hours {est_h} {RUN_FLAGS[model]}{sampling}   # {len(rs)} rows")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", type=float, default=1.0, help="fraction of the 12 scenarios per domain (1.0 = all 48)")
    ap.add_argument("--effort", default=exp23.DEFAULT_EFFORT, choices=sorted(exp23.CAP_COT_ON))
    ap.add_argument("--orders-primary", type=int, default=6)
    ap.add_argument("--orders-secondary", type=int, default=3, choices=(3, 6))
    ap.add_argument("--secondary-conditions", default="all", choices=("all", "core"))
    ap.add_argument("--pessimism", type=float, default=1.0)
    ap.add_argument("--force", action="store_true", help="rebuild a request file although its exp id already has raw rows")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("cost")
    p = sub.add_parser("s0-author"); p.add_argument("--round", type=int, default=1)
    p.add_argument("--speculative", action="store_true",
                   help="also author an extra candidate for scenarios whose earlier set still awaits its judge verdict "
                        "(adds a reminder of the most common rejection reason, exp23.SPECULATIVE_NUDGE)")
    p = sub.add_parser("s0-judge"); p.add_argument("--round", type=int, default=1)
    p.add_argument("--parallel", action="store_true",
                   help="judge every structurally valid attempt that has no parsed verdict (not only the first one per scenario)")
    for name in ("s1-plan", "s2-review", "s3-aux"):
        p = sub.add_parser(name); p.add_argument("--attempt", type=int, default=1)
    p = sub.add_parser("s3-judge"); p.add_argument("--round", type=int, default=1)
    p = sub.add_parser("s4-exec")
    p.add_argument("--executor", required=True, choices=sorted(exp23.EXECUTORS))
    p.add_argument("--orders", type=int, default=None, choices=(3, 6))
    p.add_argument("--shard-rows", type=int, default=exp23.SHARD_ROWS, help="max rows per request file (keeps one Modal result < ~100 MiB)")
    sub.add_parser("status")
    p = sub.add_parser("commands"); p.add_argument("--file", default=None, help="only request files whose name contains this")
    a = ap.parse_args()
    {"cost": cmd_cost, "s0-author": cmd_s0_author, "s0-judge": cmd_s0_judge, "s1-plan": cmd_s1, "s2-review": cmd_s2,
     "s3-aux": cmd_s3, "s3-judge": cmd_s3_judge, "s4-exec": cmd_s4, "status": cmd_status, "commands": cmd_commands}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
