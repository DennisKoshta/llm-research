"""Ledger-gated launcher for the Exp 3.2 activation capture (v3.modal_acts). Prints counts and checks only.

  # CPU only: build the prompt set (Exp 1.1 main grid + Exp 3.1 prompts, deduped), print plan and cost
  PYTHONPATH=src .venv/bin/python scripts/run_exp32.py --tag main --dry-run
  # real run (reserves in the ledger, one H200 per shard, releases with the estimated cost, verifies the app stopped)
  PYTHONPATH=src .venv/bin/python scripts/run_exp32.py --tag main --shards 3     # family / effort: recorded Phase 1 decision

Prompt text is read from data/author/final_bases.jsonl (+ judge_verdicts.json for the Exp 3.1 numeric table) and
shipped to Modal programmatically; it is never printed. Results land in data/exp32/<tag>/ (the npz files are read
back from the artifacts volume and sha256-verified; arrays that were also returned are compared with them):
acts_<arm>.npz (float32 pre_in / pre_out [+ pre_in_off / pre_out_off] + r_hat + metadata), prompts_index.json (prompt
id -> structure, no text), run.json (checks, hashes, cost). `--effort` must equal the reasoning_effort of the Exp 1.1 /
Exp 3.1 sweeps whose outcomes label these activations: at xhigh / low / default the cot_on system message carries the
template's effort block and the cot_off one does not, so cot_on and cot_off are forwarded on their own prefixes (only
at 'medium' do they coincide); arms in --cot-off-arms get the second pass.
Analysis: scripts/analyze_exp3.py exp32 --acts data/exp32/<tag>.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v3 import arms as arms_mod, exp31, exp32, ledger, scenarios  # noqa: E402

MAX_ARMS_PER_SHARD = 2    # one container builds the model once and swaps this many checkpoints
ALL_ARMS = arms_mod.ARMS_27B
COT_OFF_ARMS = "intact,refusal_ablated,pca_101"   # arms with cot_off labels: Exp 1.1 (intact, ablated) and Exp 3.1 (+ pca_101)
VOLUME = "curiosity-study-artifacts"


def build_prompts(a) -> tuple[list, dict, dict]:
    """(prompts, index, counts): Exp 1.1 main grid of ALL 40 bases + the Exp 3.1 instances, deduped by content."""
    if a.placeholder:
        bases = exp31.placeholder_bases(40)
        verdicts = exp31.placeholder_verdicts(bases)
    else:
        bases = scenarios.load_bases(a.bases)
        verdicts = json.loads(pathlib.Path(a.verdicts).read_text())
    design = exp31.scale_design(a.scale)
    grid = scenarios.build_exp11_main_grid(bases)
    sources = {"exp11": grid}
    if not a.no_exp31:
        sources["exp31"] = exp31.build_exp31_instances(
            bases, verdicts, per_domain=a.bases_per_domain or design["bases_per_domain"], cells=a.cells,
            orders=a.orders or design["orders"])
    prompts, index = exp32.build_prompt_set(sources)
    counts = {"exp11_instances": len(grid), "exp31_instances": len(sources.get("exp31", [])), "unique_prompts": len(prompts),
              "exp31_prompts_shared_with_exp11": sum(1 for v in index.values() if {m["source"] for m in v} == {"exp11", "exp31"})}
    return prompts, index, counts


def make_jobs(a, prompts: list) -> list:
    arm_list = a.arms.split(",")
    unknown = [x for x in arm_list if x not in ALL_ARMS]
    if unknown:
        raise ValueError(f"unknown arms {unknown}; choose from {ALL_ARMS}")
    off = set(a.cot_off_arms.split(",")) if a.cot_off_arms else set()
    ckpts = [{"name": x, "path": arms_mod.checkpoint_path(x, a.family), "modes": ["cot_on", "cot_off"] if x in off else ["cot_on"]}
             for x in arm_list]
    n_shards = a.shards or -(-len(ckpts) // MAX_ARMS_PER_SHARD)
    shards = [ckpts[i::n_shards] for i in range(n_shards)]
    return [{"tag": a.tag, "shard_id": i, "arms": s, "prompts": prompts, "effort": a.effort, "layer": a.layer,
             "batch_size": a.batch_size, "family": a.family, "return_arrays": False if a.no_return_arrays else "auto"}
            for i, s in enumerate(shards) if s]


def app_state(app_name: str) -> str:
    """Best-effort `modal app list` check that the app is no longer running."""
    try:
        out = subprocess.run([str(ROOT / ".venv/bin/modal"), "app", "list", "--json"], capture_output=True, text=True, timeout=60)
        rows = [r for r in json.loads(out.stdout) if r.get("Description") == app_name or r.get("description") == app_name]
        return ",".join(sorted({str(r.get("State") or r.get("state")) for r in rows})) or "not listed"
    except Exception as e:   # noqa: BLE001 - informational only
        return f"unknown ({type(e).__name__})"


def run_modal(jobs: list) -> list:
    import modal
    from v3 import modal_acts

    with modal.enable_output(), modal_acts.app.run():
        return list(modal_acts.capture_acts.map(jobs))


def flags(outs: list) -> list[str]:
    """Reasons not to trust the capture (empty = every built-in check passed)."""
    bad = []
    if len({o["r_hat"]["sha256"] for o in outs}) != 1:
        bad.append("shards used different r_hat vectors")
    if len({json.dumps(o["position_check"], sort_keys=True) for o in outs}) != 1:
        bad.append("shards report different position checks")
    for o in outs:
        for name, rec in o["arms"].items():
            if rec["meta"]["rendered_sha256"] != next(iter(outs[0]["arms"].values()))["meta"]["rendered_sha256"]:
                bad.append(f"{name}: rendered prompts differ from the first arm's")
        agree = o["batch_check"]["agreement"] if "batch_check" in o else None
        if agree and max(agree[s]["max_rel_l2"] for s in agree) > 0.2:
            bad.append(f"shard {o['shard_id']}: batching noise floor {agree}")
    return bad


def fetch_from_volume(remote_path: str, dest: pathlib.Path) -> None:
    import modal

    with open(dest, "wb") as fh:
        modal.Volume.from_name(VOLUME).read_file_into_fileobj(remote_path, fh)


def save_outputs(a, outs: list, index: dict, counts: dict, est_usd: float, wall_s: float, fetch=fetch_from_volume) -> pathlib.Path:
    """Read every arm's npz back from the volume, verify sha256 and (when the job also returned them) the arrays."""
    out_dir = ROOT / "data" / "exp32" / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prompts_index.json").write_text(json.dumps(index))
    hashes, problems = {}, []
    for o in outs:
        for name, rec in o["arms"].items():
            dest = out_dir / f"acts_{name}.npz"
            fetch(rec["volume_path"], dest)
            sha = hashlib.sha256(dest.read_bytes()).hexdigest()
            if sha != rec["sha256"]:
                problems.append(f"{name}: downloaded npz sha256 differs from the volume copy")
            if "arrays" in rec:
                z = np.load(dest)
                if not all(np.array_equal(v, z[k]) for k, v in rec["arrays"].items()):
                    problems.append(f"{name}: returned arrays differ from the npz on the volume")
            hashes[name] = {"volume_sha256": rec["sha256"], "local_sha256": sha, "arrays_returned": "arrays" in rec,
                            "cot_off": rec["meta"]["cot_off"]}
    run = {"args": vars(a), "counts": counts, "position_check": outs[0]["position_check"], "r_hat": outs[0]["r_hat"],
           "versions": outs[0]["versions"], "batch_check": next((o["batch_check"] for o in outs if "batch_check" in o), None),
           "shards": [{"shard_id": o["shard_id"], "elapsed_s": o["elapsed_s"], "arms": list(o["arms"])} for o in outs],
           "sha256": hashes, "wall_s": round(wall_s, 1), "est_cost_usd": est_usd, "flags": flags(outs) + problems}
    (out_dir / "run.json").write_text(json.dumps(run, indent=1, default=str))
    return out_dir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--family", default=None, choices=["r1", "r2"], help="checkpoint family AND r_hat file; default = recorded Phase 1 decision")
    ap.add_argument("--arms", default=",".join(ALL_ARMS))
    ap.add_argument("--effort", default=None, choices=["low", "medium", "xhigh", "default"],
                    help="default = the recorded Phase 1 decision; must equal the sweeps' ('default' = no reasoning_effort kwarg)")
    ap.add_argument("--layer", type=int, default=exp32.LAYER)
    ap.add_argument("--batch-size", type=int, default=1, help="1 = one forward per prompt, no padding")
    ap.add_argument("--shards", type=int, default=0, help="parallel containers (default: <= 2 arms each)")
    ap.add_argument("--scale", type=float, default=1.0, help="Exp 3.1 prompt-set scale (see exp31.scale_design)")
    ap.add_argument("--bases-per-domain", type=int, default=None)
    ap.add_argument("--orders", type=int, default=None)
    ap.add_argument("--cells", default="all12", choices=sorted(exp31.CELL_SETS))
    ap.add_argument("--no-exp31", action="store_true", help="Exp 1.1 grid prompts only")
    ap.add_argument("--cot-off-arms", default=COT_OFF_ARMS, help="arms that also get the cot_off pass (those with cot_off labels)")
    ap.add_argument("--no-return-arrays", action="store_true", help="never return arrays from the job (the npz is read from the volume anyway)")
    ap.add_argument("--bases", default=str(ROOT / "data/author/final_bases.jsonl"))
    ap.add_argument("--verdicts", default=str(ROOT / "data/author/judge_verdicts.json"))
    ap.add_argument("--placeholder", action="store_true", help="placeholder bases (plan / dry-run checks without authored data)")
    ap.add_argument("--job", help="ledger job name (default acts:exp32_<tag>)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    a.family = a.family or arms_mod.active_family()
    a.effort_label = a.effort or exp31.default_effort(ROOT)
    a.effort = None if a.effort_label == "default" else a.effort_label

    from v3 import modal_acts

    prompts, index, counts = build_prompts(a)
    jobs = make_jobs(a, prompts)
    n_arms = sum(len(j["arms"]) for j in jobs)
    n_off = sum("cot_off" in x["modes"] for j in jobs for x in j["arms"])
    cost = modal_acts.estimate_cost(len(prompts), n_arms, len(jobs), n_cot_off_arms=n_off, separate_cot_off=a.effort_label != "medium")
    print(f"prompts={counts} arms={[[x['name'] + ('+off' if 'cot_off' in x['modes'] else '') for x in j['arms']] for j in jobs]} "
          f"family={a.family} effort={a.effort_label} layer={a.layer}")
    print(f"estimate: forwards={cost['forwards']} gpu_hours={cost['gpu_hours']} usd={cost['usd']} "
          f"wall_h_if_parallel={cost['wall_hours_if_parallel']} reservation_usd={cost['worst_case_usd']}")
    if a.dry_run:
        return 0

    job_name = a.job or f"acts:exp32_{a.tag}"
    reserve_usd = round(min(cost["worst_case_usd"], max(2.5 * cost["usd"], 6.0)), 2)   # 2.5x the estimate, never above the timeout bound
    ledger.reserve(job_name, reserve_usd, gpus=len(jobs))
    t0 = time.time()
    outs, error = None, None
    try:
        outs = run_modal(jobs)
    except BaseException as e:   # release the reservation even on Ctrl-C, then re-raise
        error = e
    wall_s = time.time() - t0
    container_s = sum(o["elapsed_s"] + 120 for o in outs) if outs else wall_s * len(jobs)
    est_usd = ledger.est_cost("H200", 1, container_s / 3600)
    ledger.release(job_name, actual=est_usd)
    print(f"released {job_name}: wall={wall_s:.0f}s container_s={container_s:.0f} estimated cost=${est_usd:.2f} "
          f"(reserved ${reserve_usd:.2f}); app state: {app_state(modal_acts.APP_NAME)}")
    if error is not None:
        raise error
    out_dir = save_outputs(a, outs, index, counts, est_usd, wall_s)
    bad = flags(outs)
    for f in bad:
        print(f"!! SANITY FLAG: {f}")
    print(f"saved {out_dir}; position contract: {outs[0]['position_check']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
