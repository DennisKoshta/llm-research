"""Launch the benign-KL job (v3.modal_kl) and save records/kl_<tag>.json (spec section 1.2: benign KL < 0.10).

Flow: select the benign prompts from the local intact transcripts -> reserve in the v3 ledger (waiting up to --wait-min
for a free GPU slot) -> one Modal H200 container measures every checkpoint -> release with an estimated cost -> write the
record (per-checkpoint mean / median / max KL and per-prompt arrays for each variant, plus the per-position KL matrices)
-> print aggregate numbers only. Prompt and completion text is shipped to Modal programmatically (harmless rows) and is
never printed or written to the record, which keeps trial ids and sha256 hashes.

  all 13 checkpoints on one H200, reserving automatically ($10):
    PYTHONPATH=src .venv/bin/python scripts/run_kl.py --tag full13 --ckpts all
  (or reserve with the ledger CLI first:  python -m v3.ledger reserve kl-full13 10 --gpus 1   and pass --already-reserved)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v3 import ledger, modal_hf, modal_kl  # noqa: E402

R1_TRANSCRIPT = ROOT / "raw_transcripts/phase1_dim_r1/intact.jsonl"
R2_TRANSCRIPT = ROOT / "raw_transcripts/phase1_dim_r2/intact.jsonl"
CONTAINER_OVERHEAD_S = 120   # container start / image pull, added to the reported runtime
DEFAULT_RESERVE_USD = 10.0
POLL_S = 60
SELF_KL_TOL = 1e-9
REPEAT_KL_TOL = 1e-4
POS0_TOL = 1e-3


def sha256_file(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _rel(path) -> str:
    p = pathlib.Path(path).resolve()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def build_rows(args) -> tuple[list[dict], dict]:
    """The selected benign rows ({id,label,full_prompt,messages,completions}) and candidate statistics."""
    r2 = None if args.no_r2_trajectory else args.r2_transcript
    cands, stats = modal_kl.load_benign_candidates(args.r1_transcript, r2)
    rows = modal_kl.select_benign_rows(cands, args.n_prompts)
    check_rows(rows)
    stats = {**stats, "n_candidates": len(cands), "n_selected": len(rows)}
    return rows, stats


def check_rows(rows: list[dict]) -> None:
    """Fail before any reservation: every prompt ends '<think>\\n' and has no forbidden literal (PLAN rule 5). Errors name ids only."""
    bad = [r["id"] for r in rows if not r["full_prompt"].endswith("<think>\n") or any(lit in r["full_prompt"] for lit in modal_hf.FORBIDDEN_IN_PROMPTS)]
    if bad:
        raise ValueError(f"{len(bad)} prompts do not end with '<think>\\n' or contain a forbidden literal, e.g. ids {bad[:3]}")


def make_job(args, rows: list[dict], ckpts: list[dict]) -> dict:
    return {"tag": args.tag, "ckpts": ckpts, "rows": rows, "n_gen": modal_kl.N_GEN, "repeat_prompts": 3}


def run_modal(job: dict) -> dict:
    import modal

    with modal.enable_output(), modal_kl.app.run():
        return modal_kl.kl_checkpoints.remote(job)


def reserve_with_wait(job_name: str, usd: float, gpus: int, wait_min: float, poll_s: float = POLL_S, sleep=time.sleep,
                      clock=time.time) -> None:
    """ledger.reserve, retried every poll_s while the refusal is the GPU-concurrency cap (budget refusals raise at once)."""
    deadline = clock() + wait_min * 60
    while True:
        try:
            ledger.reserve(job_name, usd, gpus=gpus)
            return
        except RuntimeError as e:
            if not str(e).startswith("concurrency") or clock() >= deadline:
                raise
            print(f"waiting for a GPU slot ({e})", flush=True)
            sleep(poll_s)


def sanity_flags(record: dict) -> list[str]:
    """Reasons not to trust the numbers (empty list = all built-in cross-checks passed)."""
    flags = []
    n = record["prompt_set"]["n"]
    checks = record.get("checks", {})
    head = checks.get("head_check")
    if head and not head.get("fp32_head_used", True):
        flags.append(f"float32-head logits disagreed with the model's own logits (KL {head['kl_ownlogits_vs_fp32head_max']:.3g}); own logits used")
    rep = checks.get("repeat_check")
    if rep and rep["max_kl_between_two_runs"] > REPEAT_KL_TOL:
        flags.append(f"two identical forward passes differ: max KL {rep['max_kl_between_two_runs']:.3g} on {rep['ckpt']}")
    if n < modal_kl.N_PROMPTS:
        flags.append(f"only {n} benign prompts (target {modal_kl.N_PROMPTS})")
    for name, ck in record["checkpoints"].items():
        d = ck["diag"]
        if d["render_matches"] != n:
            flags.append(f"{name}: chat-template render matched the transcript prompt on {d['render_matches']}/{n} rows")
        if not d["tokenization_same_as_first_ckpt"]:
            flags.append(f"{name}: tokenizer produced different ids than the first checkpoint's (first checkpoint's ids were used)")
        if ck.get("n_nonfinite_kl"):
            flags.append(f"{name}: {ck['n_nonfinite_kl']} non-finite KL entries")
        if name in modal_kl.REFERENCES and name in ck["vs"]:
            self_kl = ck["vs"][name]["kl_teacher_forced"]["max"]
            if abs(self_kl) > SELF_KL_TOL or abs(ck["vs"][name]["kl_prompt_last"]["max"]) > SELF_KL_TOL:
                flags.append(f"{name}: KL against itself is not 0 ({self_kl:.3g})")
        for ref, st in ck["vs"].items():
            if st.get("pos0_traj_consistency_max_abs_diff", 0.0) > POS0_TOL:
                flags.append(f"{name} vs {ref}: position-0 KL differs between trajectories by {st['pos0_traj_consistency_max_abs_diff']:.3g}")
    return flags


def build_record(args, rows, row_stats, out, wall_s, est_usd) -> dict:
    ckpts_out = out["ckpts"]
    summary = modal_kl.build_summary(ckpts_out)
    checkpoints = {}
    for name, rec in ckpts_out.items():
        checkpoints[name] = {"path": rec["path"], "seconds": rec["seconds"], "diag": rec["diag"], "n_nonfinite_kl": rec["n_nonfinite_kl"],
                             "primary_ref": summary[name]["primary_ref"], "vs": summary[name]["vs"]}
    record = {
        "schema": "kl_v1", "tag": args.tag, "created_at": time.time(), "args": vars(args), "gate": modal_kl.GATE,
        "metric": "KL(P_ref || P_ckpt) in nats, ref = the checkpoint's own-family intact (r1_intact / r2_intact; every arm also vs bf16_base). "
                  "kl_prompt_last: next-token distribution at the last prompt position (prompt ends '<think>\\n', enable_thinking=True). "
                  "kl_teacher_forced: mean over the first 32 generated positions teacher-forcing the r1_intact greedy completion tokens "
                  "(transcript phase1_dim_r1); kl_teacher_forced_own: same with each family's own intact greedy tokens (r2_* on phase1_dim_r2). "
                  "Float32 head, batch size 1, no padding. Gate (spec 1.2): mean over prompts < 0.10.",
        "prompt_set": {"n": len(rows), "ids": [r["id"] for r in rows], "labels": [r["label"] for r in rows],
                       "prompt_sha256": out["config"]["prompt_sha256"], "select_salt": modal_kl.SELECT_SALT, "candidate_stats": row_stats,
                       "by_label": {lab: sum(r["label"] == lab for r in rows) for lab in modal_kl.SET_LABELS},
                       "transcripts": {t: {"path": _rel(path), "sha256": sha256_file(path)}
                                       for t, path in (("r1", args.r1_transcript), ("r2", args.r2_transcript))
                                       if t in out["config"]["trajectories"]}},
        "config": out["config"], "versions": out["versions"], "checks": out["checks"],
        "elapsed_s": out["elapsed_s"], "wall_s": round(wall_s, 1), "est_cost_usd": est_usd,
        "checkpoints": checkpoints,
        "kl_position_matrix": {name: rec["kl"] for name, rec in ckpts_out.items()},
    }
    record["flags"] = sanity_flags(record)
    return record


def _cell(st: dict) -> str:
    return f"{st['mean']:.4f}/{st['median']:.4f}/{st['max']:.4f}"


def print_summary(record: dict) -> None:
    """Aggregate numbers only: mean/median/max KL per checkpoint and variant."""
    for flag in record.get("flags", []):
        print(f"!! SANITY FLAG: {flag}")
    ps = record["prompt_set"]
    print(f"prompts={ps['n']} {ps['by_label']} trajectories={record['config']['trajectories']} wall={record['wall_s']:.0f}s "
          f"container={record['elapsed_s']:.0f}s est_cost=${record['est_cost_usd']:.2f}  versions={record['versions']}")
    print(f"head_check={record['checks'].get('head_check')}  repeat_check={record['checks'].get('repeat_check')}")
    variants = ("kl_prompt_last", "kl_teacher_forced", "kl_teacher_forced_own")

    def table(title, pick):
        print(f"\n== {title}   (cell = mean/median/max over {ps['n']} prompts, nats; gate: mean < {record['gate']})")
        print(f"{'checkpoint':20s}{'ref':11s}" + "".join(f"{v:>30s}" for v in variants) + "  gate")
        for name, ck in record["checkpoints"].items():
            for ref in pick(name, ck):
                st = ck["vs"][ref]
                ok = all(st[v]["gate_mean_lt_0.10"] for v in variants[:2])
                print(f"{name:20s}{ref:11s}" + "".join(f"{_cell(st[v]):>30s}" for v in variants) + f"  {'PASS' if ok else 'FAIL'}")

    table("vs own-family intact (bf16_base: vs both intact checkpoints)",
          lambda n, ck: [ck["primary_ref"]] if ck["primary_ref"] in ck["vs"] else ([r for r in ("r1_intact", "r2_intact") if r in ck["vs"]] if n == "bf16_base" else []))
    table("vs bf16_base", lambda n, ck: ["bf16_base"] if "bf16_base" in ck["vs"] and n != "bf16_base" else [])
    cons = [st.get("pos0_traj_consistency_max_abs_diff") for ck in record["checkpoints"].values() for st in ck["vs"].values()]
    cons = [c for c in cons if c is not None]
    if cons:
        print(f"\nposition-0 KL, r1 vs r2 trajectory pass (numeric noise probe): max abs diff over all cells = {max(cons):.3g}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="full13")
    ap.add_argument("--ckpts", default="all", help="'all', 'smoke', or comma-separated names (see v3.modal_hf.select_checkpoints)")
    ap.add_argument("--n-prompts", type=int, default=modal_kl.N_PROMPTS)
    ap.add_argument("--no-r2-trajectory", action="store_true", help="only the spec'd r1_intact teacher-forcing tokens (no kl_teacher_forced_own for r2)")
    ap.add_argument("--r1-transcript", default=str(R1_TRANSCRIPT))
    ap.add_argument("--r2-transcript", default=str(R2_TRANSCRIPT))
    ap.add_argument("--job", help="ledger job name (default kl-<tag>)")
    ap.add_argument("--reserve-usd", type=float, default=DEFAULT_RESERVE_USD)
    ap.add_argument("--already-reserved", action="store_true", help="the ledger reservation exists (made with the ledger CLI)")
    ap.add_argument("--wait-min", type=float, default=30.0, help="wait up to this long for a free GPU slot when the ledger refuses on concurrency")
    ap.add_argument("--dry-run", action="store_true", help="select prompts and build the job, print counts, do not touch Modal or the ledger")
    args = ap.parse_args(argv)

    ckpts = modal_hf.select_checkpoints(args.ckpts)
    rows, row_stats = build_rows(args)
    if not rows:
        raise ValueError("no benign rows selected")
    job = make_job(args, rows, ckpts)
    print(f"prompts={len(rows)} by_label={ {lab: sum(r['label'] == lab for r in rows) for lab in modal_kl.SET_LABELS} } "
          f"candidates={row_stats['n_candidates']} stats={row_stats} ckpts={[c['name'] for c in ckpts]} "
          f"trajectories={'r1' if args.no_r2_trajectory else 'r1,r2'}")
    if args.dry_run:
        return 0

    job_name = args.job or f"kl-{args.tag}"
    if not args.already_reserved:
        reserve_with_wait(job_name, args.reserve_usd, 1, args.wait_min)
    t0 = time.time()
    out, error = None, None
    try:
        out = run_modal(job)
    except BaseException as e:  # release the reservation even on Ctrl-C, then re-raise
        error = e
    wall_s = time.time() - t0
    container_s = out["elapsed_s"] + CONTAINER_OVERHEAD_S if out else wall_s
    est_usd = ledger.est_cost(modal_kl.GPU, 1, container_s / 3600)
    ledger.release(job_name, actual=est_usd)
    print(f"released {job_name}: wall={wall_s:.0f}s container_s={container_s:.0f} estimated cost=${est_usd:.2f} (reserved ${args.reserve_usd:.2f})")
    if error is not None:
        raise error
    record = build_record(args, rows, row_stats, out, wall_s, est_usd)
    path = ROOT / "records" / f"kl_{args.tag}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(record) + "\n")
    print(f"saved {path}")
    print_summary(record)
    return 0


if __name__ == "__main__":
    sys.exit(main())
