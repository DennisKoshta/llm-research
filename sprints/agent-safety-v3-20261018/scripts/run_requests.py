"""Ledger-gated launcher: run request rows on one or more checkpoints and persist raw transcripts.

  PYTHONPATH=src .venv/bin/python scripts/run_requests.py --exp-id phase1_probe \
      --requests data/phase1/requests_set_c.jsonl --arms intact,refusal_ablated --family r1 \
      --gpu H200 --est-hours 0.4

Writes raw_transcripts/<exp-id>/<arm>.jsonl (unredacted, 100% of rows) and prints ONLY aggregate
structural counts (never prompt/completion text). Jobs run detached; call ids are saved so results can
be re-fetched with --resume if this process dies.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v3 import arms as arms_mod, ledger  # noqa: E402

GPU_COUNT = {"H200": 1, "H100": 1, "H200:2": 2}
GPU_RATE_KEY = {"H200": "H200", "H100": "H100", "H200:2": "H200"}


def load_rows(paths: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in open(p) if l.strip()]
    ids = [r["trial_id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate trial_id"
    return rows


def summarize(rows: list[dict]) -> dict:
    """Aggregate, text-free summary safe to print."""
    c = collections.Counter()
    toks = []
    for r in rows:
        if "parsed" in r:
            c["status:" + r["parsed"]["status"]] += 1
            c["tool:" + str(r["parsed"].get("first_tool_name") or (r["parsed"]["tool_names"] or [None])[0])] += 1
        if "outcome_category" in r:
            c["outcome:" + r["outcome_category"]] += 1
        c["finish:" + str(r.get("finish_reason", "episode"))] += 1
        toks.append(r.get("generated_tokens", r.get("total_generated_tokens", 0)))
    toks.sort()
    q = lambda p: toks[min(len(toks) - 1, int(p * len(toks)))] if toks else 0
    return {"n": len(rows), "counts": dict(sorted(c.items())), "tokens_p50": q(.5), "tokens_p90": q(.9), "tokens_max": toks[-1] if toks else 0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--requests", required=True, help="comma-separated JSONL files")
    ap.add_argument("--arms", required=True, help="comma-separated arm keys (see v3.arms)")
    ap.add_argument("--family", default=None, help="r1|r2 (27B arms); default = recorded active family")
    ap.add_argument("--gpu", default="H200", choices=list(GPU_COUNT))
    ap.add_argument("--est-hours", type=float, required=True, help="worst-case wall hours PER ARM (for the reservation)")
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--eager", type=int, default=1, help="enforce_eager (1/0)")
    ap.add_argument("--model-kind", default="qwen", choices=["qwen", "llama"])
    ap.add_argument("--quantization", default="fp8")
    ap.add_argument("--tp", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--extra-json", default="{}", help="extra vLLM LLM() kwargs as JSON, e.g. {\"safetensors_load_strategy\":\"eager\"}")
    ap.add_argument("--resume", default=None, help="path to a saved calls json to re-fetch instead of launching")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    rows = load_rows(a.requests.split(","))
    arm_list = a.arms.split(",")
    out_dir = ROOT / "raw_transcripts" / a.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    calls_file = ROOT / "ledger" / f"calls_{a.exp_id}.json"

    payloads = []
    for arm in arm_list:
        mine = [r for r in rows if r.get("arm") in (None, arm)]
        fam = a.family if arm in arms_mod.ARMS_27B else None
        payloads.append({
            "exp_id": a.exp_id, "arm": arm, "model_kind": a.model_kind,
            "model_path": arms_mod.checkpoint_path(arm, fam),
            "engine": {"tp": a.tp or GPU_COUNT[a.gpu], "max_model_len": a.max_model_len,
                       "max_num_seqs": a.max_num_seqs, "enforce_eager": bool(a.eager),
                       "quantization": a.quantization, "fix_quant_config": fam == "r2",
                       "extra": json.loads(a.extra_json)},
            "sampling": {"temperature": a.temperature, "top_p": a.top_p, "seed_base": a.seed_base},
            "rows": mine,
        })
        print(f"[plan] arm={arm} rows={len(mine)} path={payloads[-1]['model_path']}")

    n_gpus = GPU_COUNT[a.gpu] * len(arm_list)
    est = ledger.est_cost(GPU_RATE_KEY[a.gpu], n_gpus, a.est_hours)
    print(f"[plan] gpu={a.gpu} x{len(arm_list)} arms, reservation ${est}")
    if a.dry_run:
        return 0

    from v3 import modal_serve  # imports modal; app functions below

    fn = modal_serve.FUNCTIONS[a.gpu]
    job = f"serve:{a.exp_id}"
    if not a.resume:
        ledger.reserve(job, est, gpus=n_gpus)
    t0 = time.time()
    results: dict[str, dict] = {}
    complete = False
    try:
        if a.resume:
            ids = json.loads(open(a.resume).read())
        else:
            # detach=True: the remote calls keep running (and billing) even if this client dies, so the call ids are
            # persisted immediately and the wait happens OUTSIDE the app context with reconnect retries.
            with modal_serve.modal.enable_output():
                with modal_serve.app.run(detach=True):
                    calls = {p["arm"]: fn.spawn(p) for p in payloads}
                    ids = {arm: c.object_id for arm, c in calls.items()}
                    calls_file.write_text(json.dumps(ids))
                    print(f"[launched] call ids saved to {calls_file}", flush=True)
        for arm, cid in ids.items():
            res, tries = None, 0
            while res is None:
                try:
                    res = modal_serve.modal.FunctionCall.from_id(cid).get(timeout=900)
                except TimeoutError:
                    continue                              # still running: keep waiting
                except Exception as e:                    # connection blip: the remote call is unaffected
                    if "Deadline exceeded" in str(e):     # long-poll expiry, not a failure
                        continue
                    tries += 1
                    print(f"[retry {tries}] {arm}: {type(e).__name__}: {str(e)[:120]}", flush=True)
                    if tries > 40:
                        raise
                    time.sleep(30)
            results[arm] = res
            with open(out_dir / f"{arm}.jsonl", "w") as f:
                for r in res["rows"]:
                    f.write(json.dumps(r) + "\n")
            (out_dir / f"{arm}.meta.json").write_text(json.dumps(res["meta"], indent=1, default=str))
            print(f"[done] {arm}: meta=" + json.dumps({k: v for k, v in res["meta"].items() if k != "engine"}))
            print(f"[summary] {arm}: " + json.dumps(summarize(res["rows"])))
        complete = len(results) == len(ids)
    finally:
        wall_h = (time.time() - t0) / 3600
        actual = ledger.est_cost(GPU_RATE_KEY[a.gpu], n_gpus, wall_h)
        if complete:
            try:    # a resumed fetch settles the manual `inflight:<exp>` reservation made when the client lost its connection
                ledger.release(job if not a.resume else f"inflight:{a.exp_id}", actual)
            except Exception as e:
                print(f"[ledger] release skipped: {type(e).__name__}: {str(e)[:100]}")
        elif not complete:
            print(f"[ledger] NOT released: remote calls may still be running; re-fetch with --resume {calls_file} "
                  f"(then `python -m v3.ledger release {job} --actual X`)")
        print(f"[ledger] wall {wall_h*60:.1f} min, est actual ${actual}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
