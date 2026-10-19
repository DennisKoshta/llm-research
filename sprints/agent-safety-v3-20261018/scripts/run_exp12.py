"""Ledger-gated launcher for the Exp 1.2 students (v3.modal_train): plan, run, fetch.

  plan   no network, no ledger: job table, cost estimate, the largest preset that fits --target-usd (whole experiment)
  run    build the bundle (corpora + held-out eval sets) -> ledger reserve -> upload -> one H200 container per shard
         -> download every finished adapter's rows into raw_transcripts/exp12/ -> ledger release
  fetch  (re)download finished adapters of a tag from the artifacts volume (also after a crashed run)

Needs: corpora in data/exp12/corpora (scripts/build_exp12_corpora.py build), data/exp12/{eval_instances,refusal_items}.jsonl
(its `requests` step) and the projection vectors on the artifacts volume (v3.modal_hf.VECTOR_FILES, copied there by the
Phase 1 projection job). --effort must equal the effort the corpora were generated with (default: records/phase1_effort_decision.json,
else low); --cap is the student max_new_tokens (>= 2048 and >= the teacher cap).

  PYTHONPATH=src .venv/bin/python scripts/run_exp12.py plan --preset full72
  # 1. cheapest code-path check on the real 27B (floor only: load, HF generate, episodes, projection; no corpora needed)
  PYTHONPATH=src .venv/bin/python scripts/run_exp12.py run --tag exp12_floor_smoke --floor-only --target-usd 10
  # 2. one real adapter (3 optimizer steps, 8 eval episodes): measures train tok/s
  PYTHONPATH=src .venv/bin/python scripts/run_exp12.py run --tag exp12_train_smoke --smoke --target-usd 10
  PYTHONPATH=src .venv/bin/python scripts/analyze_exp12.py --calibrate      # reads raw_transcripts/exp12_smoke -> train_tok_per_s, eval_seconds, decode_step_s
  # 3. re-plan with the measured speeds, then the experiment
  PYTHONPATH=src .venv/bin/python scripts/run_exp12.py plan --preset full72 --train-tok-s X --eval-seconds Y
  PYTHONPATH=src .venv/bin/python scripts/run_exp12.py run --tag exp12_main --preset full72 --train-tok-s X --eval-seconds Y --target-usd 75
  PYTHONPATH=src .venv/bin/python scripts/analyze_exp12.py

--scale/--preset full72 | reduced54 | lean36 | smoke; --grid "A:32,8;B:32,8;Bp:32;C:32" overrides the (corpus, rank) set, --teachers /
--seeds restrict those axes, --lora-layers spec_plus_full adds the full-attention layers to the LoRA set, --fla installs
flash-linear-attention in the image (faster, unvalidated). --smoke and --floor-only rows are downloaded into raw_transcripts/exp12_smoke,
everything else into raw_transcripts/exp12 (the analysis also drops adapters whose cfg is a smoke subset). A run that dies midway keeps every finished adapter on the volume
(committed after each one) and a job that raises is isolated (re-run the same tag with --resume). The launcher refuses
students-stage estimates above --target-usd unless --over-budget. Prints aggregate numbers only (never prompt or completion text).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import os
import pathlib
import shutil
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from v3 import episodes, exp12, ledger  # noqa: E402

DATA = ROOT / "data" / "exp12"
EFFORT_DECISION = ROOT / "records" / "phase1_effort_decision.json"
FAMILY_DECISION = ROOT / "records" / "phase1_family_decision.json"     # holds reasoning_effort_for_phase2 (what v3.exp11 follows)
RAW_OUT = ROOT / "raw_transcripts" / "exp12"
SMOKE_OUT = ROOT / "raw_transcripts" / "exp12_smoke"     # --smoke and --floor-only launches: never mixed into the experiment's raw rows
REMOTE = "/agent-safety-v3-20261018/exp12"
CONTAINER_OVERHEAD_S = 150     # container start + image pull, added to each shard's reported runtime when estimating cost


def parse_grid(spec: str) -> tuple:
    """'A:32,8;B:32' -> (('A', 32), ('A', 8), ('B', 32))."""
    out = []
    for part in spec.split(";"):
        corpus, _, ranks = part.partition(":")
        if corpus not in exp12.CORPORA or not ranks:
            raise ValueError(f"bad --grid entry {part!r}; corpora are {exp12.CORPORA}")
        out += [(corpus, int(r)) for r in ranks.split(",")]
    return tuple(out)


def scale_from_args(a) -> exp12.Scale:
    scale = exp12.PRESETS["smoke" if a.smoke else a.preset]
    over = {}
    if a.grid:
        over["corpus_ranks"] = parse_grid(a.grid)
    if a.teachers:
        over["teachers"] = tuple(a.teachers.split(","))
    if a.seeds:
        over["seeds"] = tuple(int(s) for s in a.seeds.split(","))
    for name in ("epochs", "floors", "shards", "max_prompts", "later_turns"):
        if getattr(a, name) is not None:
            over[name] = getattr(a, name)
    if a.floor_only:
        over.update(corpus_ranks=(), floors=max(1, over.get("floors", 1)), shards=1)
    return dataclasses.replace(scale, **over)


def build_jobs(scale: exp12.Scale) -> tuple:
    """(job dicts per shard, number of shards). Floors are numbered in order of appearance."""
    shards = exp12.shard_jobs(scale.jobs(), scale.shards, scale.floors)
    out, rep = [], 0
    for shard in shards:
        jobs = []
        for j in shard:
            if j is None:
                jobs.append({"adapter_id": exp12.floor_id(rep), "floor": True, "floor_rep": rep})
                rep += 1
            else:
                jobs.append({"adapter_id": j.adapter_id, "corpus": j.corpus, "teacher": j.teacher, "rank": j.rank, "seed": j.seed})
        out.append(jobs)
    return out, len(out)


def run_cfg(a, scale: exp12.Scale) -> dict:
    from v3 import modal_train

    kw = {"epochs": scale.epochs, "effort": exp12.resolve_effort(a.effort), "cap": getattr(a, "cap", exp12.CAP), "lora_layers": a.lora_layers}
    built = DATA / "corpora_manifest.json"
    if built.exists():        # the student length filter must admit everything the builder admitted
        kw["max_seq_len"] = max(exp12.MAX_SEQ_TOKENS, json.loads(built.read_text()).get("max_seq_tokens", exp12.MAX_SEQ_TOKENS))
    if a.smoke:
        kw.update(epochs=1, max_steps=3, eval_limit=8)
    episodes.check_decoding("cot_on", kw["effort"], kw["cap"])      # I1 before anything is reserved or uploaded
    return dataclasses.asdict(modal_train.RunCfg(**kw))


def estimate_kwargs(a) -> dict:
    """estimate_cost keywords: eval/train set sizes from the request manifest, measured speeds from the calibration flags."""
    kw = {}
    requests = DATA / "requests_manifest.json"
    if requests.exists():
        m = json.loads(requests.read_text())
        kw = {"n_train_bases": m["n_train_prompts"] // len(exp12.TRAIN_CELLS), "eval_episodes": m["n_eval_instances"]}
    for flag, name in (("train_tok_s", "train_tok_per_s"), ("eval_seconds", "eval_seconds"), ("decode_step_s", "decode_step_s")):
        if getattr(a, flag, None) is not None:
            kw[name] = getattr(a, flag)
    if getattr(a, "effort", None):
        kw["effort"] = exp12.resolve_effort(a.effort)
    return kw


def build_manifest_estimate(a, scale: exp12.Scale) -> tuple:
    """(estimate, kwargs). The corpus size knobs (max_prompts, later_turns) come from the build step when its manifest exists."""
    built = DATA / "corpora_manifest.json"
    if built.exists() and a.max_prompts is None and a.later_turns is None:
        b = json.loads(built.read_text())["scale"]
        scale = dataclasses.replace(scale, max_prompts=b["max_prompts"], later_turns=b["later_turns"])
    kw = estimate_kwargs(a)
    return exp12.estimate_cost(scale, **kw), kw


def print_plan(scale: exp12.Scale, shards: list, est: dict, target: float, kw: dict) -> None:
    print(f"{scale.n_adapters} adapters + {scale.floors} floors on {len(shards)} shards; corpora x ranks {list(scale.corpus_ranks)}")
    print(f"teachers {list(scale.teachers)} seeds {list(scale.seeds)} epochs {scale.epochs} max_prompts {scale.max_prompts} later_turns {scale.later_turns}")
    for i, s in enumerate(shards):
        print(f"  shard {i}: {len(s)} jobs  {', '.join(j['adapter_id'] for j in s[:4])}{' ...' if len(s) > 4 else ''}")
    print(f"estimate: ${est['usd']:.2f} total ({est['gpu_hours']:.1f} GPU-h; students ${est['stages']['students']['usd']:.2f}, "
          f"~{est['wall_hours_students_per_shard']:.1f} h wall per shard); generations {est['generations']:.0f}, tokens {est['tokens']:.3g}")
    for name, st in est["stages"].items():
        print(f"    {name:24s} ${st['usd']:7.2f}  {st['gpu_hours']:6.2f} GPU-h  gens {st['generations']:.0f}")
    name, rec = exp12.recommend_scale(target, **kw)
    print(f"largest preset within ${target:.0f}: {name} (${rec['usd']:.2f}); assumptions {est['assumed']}")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def read_jsonl(path) -> list:
    return [json.loads(line) for line in pathlib.Path(path).read_text().splitlines() if line.strip()]


def make_bundle(scale: exp12.Scale, jobs: list, cfg: dict) -> tuple:
    """Assemble the bundle for the corpora the jobs need -> (local dir, manifest)."""
    needed = sorted({(j["corpus"], j["teacher"]) for shard in jobs for j in shard if not j.get("floor")})
    corpora: dict = {}
    for corpus, teacher in needed:
        path = DATA / "corpora" / f"{corpus}.{teacher}.jsonl"
        if not path.exists():
            raise SystemExit(f"missing {path}: run scripts/build_exp12_corpora.py build")
        corpora.setdefault(corpus, {})[teacher] = read_jsonl(path)
        if not corpora[corpus][teacher]:
            raise SystemExit(f"{path} is empty (no prompt survived the corpus filters): rebuild or drop {corpus}.{teacher} from --grid / --teachers")
    built = DATA / "corpora_manifest.json"
    if needed and built.exists():
        b = json.loads(built.read_text())
        if b.get("effort") != exp12.effort_label(cfg["effort"]):
            raise SystemExit(f"--effort {exp12.effort_label(cfg['effort'])} differs from the effort the corpora were generated with "
                             f"({b.get('effort')}): prompts embed it, so students would train off-distribution")
        if b.get("cap") and cfg["cap"] < b["cap"]:
            raise SystemExit(f"student cap {cfg['cap']} is below the teacher cap {b['cap']}: students could not reproduce the longest targets")
    ev, refusal = DATA / "eval_instances.jsonl", DATA / "refusal_items.jsonl"
    if not ev.exists() or not refusal.exists():
        raise SystemExit("missing eval sets: run scripts/build_exp12_corpora.py requests")
    out = DATA / "bundle_local"
    shutil.rmtree(out, ignore_errors=True)
    manifest = exp12.write_bundle(out, corpora, read_jsonl(ev), read_jsonl(refusal),
                                  {"effort": cfg["effort"], "cap": cfg["cap"], "scale": exp12.scale_dict(scale)})
    return out, manifest


def upload_bundle(local: pathlib.Path, bundle_id: str) -> str:
    """Put the bundle on the artifacts volume (file by file: no directory-semantics guesswork) and check it landed."""
    from v3 import modal_train

    remote = f"{REMOTE}/bundles/{bundle_id}"
    vol = modal_train.art_vol
    try:
        have = {e.path.rsplit("/", 1)[-1] for e in vol.listdir(remote)}
    except Exception:
        have = set()
    if "manifest.json" in have:
        print(f"bundle {bundle_id} already on the volume")
        return remote
    files = sorted(p for p in local.rglob("*") if p.is_file())
    with vol.batch_upload(force=True) as batch:
        for f in files:
            batch.put_file(str(f), f"{remote}/{f.relative_to(local).as_posix()}")
    landed = {e.path.rsplit("/", 1)[-1] for e in vol.listdir(remote)}
    if "manifest.json" not in landed:
        raise RuntimeError(f"bundle upload to {remote} did not produce manifest.json (got {sorted(landed)})")
    print(f"uploaded bundle {bundle_id}: {len(files)} files -> {remote}")
    return remote


def read_remote(vol, path: str) -> bytes:
    buf = io.BytesIO()
    vol.read_file_into_fileobj(path, buf)
    return buf.getvalue()


def fetch(tag: str, out_dir: pathlib.Path = RAW_OUT) -> list:
    """Download every finished adapter of `tag` (verified against its .done.json hash) into raw_transcripts/exp12/."""
    from v3 import modal_train

    vol = modal_train.art_vol
    base = f"{REMOTE}/results/{tag}"
    names = {e.path.rsplit("/", 1)[-1] for e in vol.listdir(base)}
    done = sorted(n[:-len(".done.json")] for n in names if n.endswith(".done.json"))
    out_dir.mkdir(parents=True, exist_ok=True)
    got = []
    for aid in done:
        meta = json.loads(read_remote(vol, f"{base}/{aid}.done.json"))
        dest = out_dir / f"{aid}.jsonl"
        if dest.exists() and sha256_bytes(dest.read_bytes()) == meta["sha256"]:
            got.append(aid)
            continue
        body = read_remote(vol, f"{base}/{aid}.jsonl")
        if sha256_bytes(body) != meta["sha256"] or body.count(b"\n") != meta["n_rows"]:
            raise RuntimeError(f"{aid}: downloaded file fails its hash / row count")
        dest.write_bytes(body)
        got.append(aid)
    print(f"fetched {len(got)} adapters of {tag} into {out_dir}")
    return got


def summarize_local(adapter_ids: list, out_dir: pathlib.Path | None = None) -> None:
    import analyze_exp12

    for aid in adapter_ids:        # one adapter at a time: the full set is several hundred MB
        for line in analyze_exp12.adapter_overview(read_jsonl((out_dir or RAW_OUT) / f"{aid}.jsonl")):
            print("  " + line)


def common_args(ap) -> None:
    ap.add_argument("--preset", "--scale", dest="preset", choices=sorted(exp12.PRESETS), default="full72",
                    help="scale preset: full72 (4 corpora x 3 teachers x 2 ranks x 3 seeds), reduced54, lean36, smoke")
    ap.add_argument("--grid", default=None)
    ap.add_argument("--teachers", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--floors", type=int, default=None)
    ap.add_argument("--shards", type=int, default=None)
    ap.add_argument("--max-prompts", dest="max_prompts", type=int, default=None, help="cost estimate only; corpora are fixed by the build step")
    ap.add_argument("--later-turns", dest="later_turns", type=int, default=None)
    ap.add_argument("--effort", choices=exp12.EFFORT_CHOICES, default=exp12.load_effort_decision(EFFORT_DECISION),
                    help="reasoning effort (default: records/phase1_effort_decision.json if present, else low); must match the corpora")
    ap.add_argument("--cap", type=int, default=exp12.CAP, help="student max_new_tokens (>= 2048, and >= the teacher cap)")
    ap.add_argument("--lora-layers", default="spec", choices=["spec", "spec_plus_full"])
    ap.add_argument("--target-usd", type=float, default=75.0)
    ap.add_argument("--train-tok-s", dest="train_tok_s", type=float, default=None, help="measured processed tokens/s (analyze_exp12 --calibrate)")
    ap.add_argument("--eval-seconds", dest="eval_seconds", type=float, default=None, help="measured seconds per evaluation")
    ap.add_argument("--decode-step-s", dest="decode_step_s", type=float, default=None)
    ap.add_argument("--smoke", action="store_true", help="one adapter, 3 optimizer steps, 8 eval episodes")
    ap.add_argument("--fla", action="store_true", help="install flash-linear-attention in the image (faster, but unvalidated: compare a --floor-only run with and without)")
    ap.add_argument("--floor-only", action="store_true", help="only the untuned-base floor run (needs no corpora)")


def print_notes(a) -> None:
    for note in exp12.effort_notes(a.effort, a.cap, (EFFORT_DECISION, FAMILY_DECISION)):
        print(note)


def cmd_plan(a) -> int:
    print_notes(a)
    scale = scale_from_args(a)
    shards, _ = build_jobs(scale)
    est, kw = build_manifest_estimate(a, scale)
    print_plan(scale, shards, est, a.target_usd, kw)
    return 0


def wait_call(call, max_blips: int = 40):
    """call.get() that rides out client-side connection blips ("Deadline exceeded", dropped streams): the remote shard keeps
    running and billing regardless, so a transient error must never be recorded as a shard failure."""
    blips = 0
    while True:
        try:
            return call.get(timeout=900)
        except TimeoutError:
            continue
        except Exception as e:
            msg = f"{type(e).__name__} {e}"
            if "Deadline exceeded" in msg:
                continue
            if any(t in msg for t in ("ConnectionError", "UNAVAILABLE", "Connection lost", "StreamTerminated", "ClientClosed")) and blips < max_blips:
                blips += 1
                print(f"[retry {blips}] {type(e).__name__}: {str(e)[:100]}", flush=True)
                time.sleep(30)
                continue
            raise


def cmd_run(a) -> int:
    print_notes(a)
    scale = scale_from_args(a)
    shards, n = build_jobs(scale)
    cfg = run_cfg(a, scale)
    est, kw = build_manifest_estimate(a, scale)
    print_plan(scale, shards, est, a.target_usd, kw)
    launch_usd = est["stages"]["students"]["usd"]           # teachers / Llama have their own run_requests.py launches
    print(f"this launch (students stage): ${launch_usd:.2f}")
    if launch_usd > a.target_usd and not a.over_budget:
        name, _ = exp12.recommend_scale(a.target_usd, stage="students", **kw)
        raise SystemExit(f"estimate ${launch_usd:.2f} exceeds --target-usd {a.target_usd}: use --preset {name}, reduce --grid/--seeds, or pass --over-budget")
    local, manifest = make_bundle(scale, shards, cfg)
    payloads = [{"tag": a.tag, "shard_id": i, "bundle_id": manifest["bundle_id"], "jobs": jobs, "cfg": cfg} for i, jobs in enumerate(shards)]
    print(f"bundle {manifest['bundle_id']}: {len(manifest['files'])} files; {n} payloads")
    if a.dry_run:
        return 0

    from v3 import modal_train

    job = f"exp12:{a.tag}"
    rate_cap = ledger.est_cost(modal_train.GPU, n, modal_train.TIMEOUT_S / 3600)
    reserve = a.reserve_usd or round(min(rate_cap, max(5.0, 1.5 * est["stages"]["students"]["usd"])), 2)
    calls_file = ROOT / "ledger" / f"calls_{a.tag}.json"
    if not a.resume:
        ledger.reserve(job, reserve, gpus=n)
    t0 = time.time()
    results, failures = {}, {}
    try:
        upload_bundle(local, manifest["bundle_id"])
        with modal_train.modal.enable_output(), modal_train.app.run(detach=True):
            if a.resume and calls_file.exists():
                calls = {int(k): modal_train.modal.FunctionCall.from_id(v) for k, v in json.loads(calls_file.read_text()).items()}
            elif a.resume:
                calls = {}
            else:
                calls = {p["shard_id"]: modal_train.train_eval_shard.spawn(p) for p in payloads}
                calls_file.write_text(json.dumps({k: c.object_id for k, c in calls.items()}))
                print(f"[launched] {len(calls)} shards, call ids in {calls_file}")
            for k, call in calls.items():
                try:
                    results[k] = wait_call(call)
                    print(f"[shard {k}] done in {results[k]['elapsed_s']:.0f}s: {len(results[k]['files'])} adapters"
                          + (f", FAILED jobs {results[k]['failed']}" if results[k].get("failed") else ""))
                    if results[k].get("failed"):
                        failures[f"{k}:jobs"] = results[k]["failed"]
                except Exception as e:   # keep collecting the other shards; finished adapters are on the volume anyway
                    failures[k] = f"{type(e).__name__}: {str(e)[:300]}"
                    print(f"[shard {k}] FAILED {failures[k]}")
    finally:
        wall_h = (time.time() - t0) / 3600
        container_s = sum(r["elapsed_s"] + CONTAINER_OVERHEAD_S for r in results.values()) + wall_h * 3600 * len(failures)
        actual = ledger.est_cost(modal_train.GPU, 1, (container_s if results or failures else wall_h * 3600 * n) / 3600)
        if not a.resume:
            ledger.release(job, actual)
        print(f"[ledger] wall {wall_h * 60:.1f} min, estimated cost ${actual:.2f} (reserved ${reserve:.2f}, estimate ${est['usd']:.2f})")
    out_dir = SMOKE_OUT if a.smoke or a.floor_only else RAW_OUT
    got = fetch(a.tag, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, r in results.items():
        (out_dir / f"_shard{k}_{a.tag}.meta.json").write_text(json.dumps({k2: v for k2, v in r.items() if k2 != "files"} | {"files": r["files"]}, indent=1))
    expected = {j["adapter_id"] for s in shards for j in s}
    summarize_local(sorted(got), out_dir)
    print(f"missing adapters: {sorted(expected - set(got)) or 'none'}; failed shards: {failures or 'none'}")
    return 1 if failures or expected - set(got) else 0


def cmd_fetch(a) -> int:
    out_dir = SMOKE_OUT if a.smoke else RAW_OUT
    got = fetch(a.tag, out_dir)
    summarize_local(got, out_dir)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    common_args(p)
    r = sub.add_parser("run")
    common_args(r)
    r.add_argument("--tag", required=True)
    r.add_argument("--reserve-usd", type=float, default=None)
    r.add_argument("--over-budget", action="store_true")
    r.add_argument("--resume", action="store_true")
    r.add_argument("--dry-run", action="store_true")
    f = sub.add_parser("fetch")
    f.add_argument("--tag", required=True)
    f.add_argument("--smoke", action="store_true", help=f"the tag was a --smoke / --floor-only launch: download into {SMOKE_OUT.name}")
    a = ap.parse_args(argv)
    if getattr(a, "fla", False):
        os.environ["EXP12_FLA"] = "1"          # read when v3.modal_train builds its image, so set it before that import
    return {"plan": cmd_plan, "run": cmd_run, "fetch": cmd_fetch}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
