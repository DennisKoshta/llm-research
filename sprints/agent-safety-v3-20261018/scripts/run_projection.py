"""Launch the layer-48 projection job (v3.modal_hf) and save records/projection_<tag>.json.

Flow: reserve in the v3 ledger -> run one Modal GPU container per shard -> release with an estimated cost ->
write the record -> print aggregate numbers only. Donor prompt text (harmful-category) is read and shipped
programmatically and is never printed; the record keeps ids and content hashes, not text.

  smoke (reserve first with the ledger CLI, then):
    PYTHONPATH=src .venv/bin/python -m v3.ledger reserve hf-projection-smoke 12 --gpus 1
    PYTHONPATH=src .venv/bin/python scripts/run_projection.py --tag smoke --ckpts smoke --job hf-projection-smoke --already-reserved
  all 13 checkpoints (bf16_base + 12 FP8) on 4 H200s, reserving automatically:
    PYTHONPATH=src .venv/bin/python scripts/run_projection.py --tag full13 --ckpts all --shards 4 --reserve-usd 30
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v3 import ledger, modal_hf  # noqa: E402

DONORS = ROOT.parent
RUN1_VAL = DONORS / "agent-safety-three-track-20261016/data/phase1_ablation/validation_prompts.json"
RUN2_EXTRACTION = DONORS / "agent-safety-clean-v2-20261017/stimuli/phase1_extraction.jsonl"
HEADLINE_VECTORS = ("r1.r_hat", "r2.refusal")
SETS = ("r1_val", "r2_val")
CONTAINER_OVERHEAD_S = 120   # container start / image pull per shard, added to the reported runtime


def make_jobs(args, prompts: list[dict], ckpts: list[dict]) -> list[dict]:
    """One job dict per shard; bf16_base (when selected) is first in shard 0."""
    return [{"tag": args.tag, "shard_id": i, "ckpts": shard, "prompts": prompts, "efforts": modal_hf.parse_efforts(args.efforts),
             "layer": args.layer, "batch_size": args.batch_size, "crosscheck": not args.no_crosscheck}
            for i, shard in enumerate(modal_hf.split_shards(ckpts, args.shards))]


def run_modal(jobs: list[dict]) -> list[dict]:
    import modal

    with modal.enable_output(), modal_hf.app.run():
        return list(modal_hf.project_checkpoints.map(jobs))


def sanity_flags(record: dict, outs: list[dict]) -> list[str]:
    """Reasons not to trust the numbers (empty list = all built-in cross-checks passed). `outs` = shard results."""
    flags = []
    ref = record.get("hf_reference")
    if ref and "error" in ref:
        flags.append(f"HF from_pretrained cross-check FAILED to run: {ref['error']}")
    for ck, rec in record["checkpoints"].items():
        swap = rec.get("swap_vs_hf_from_pretrained")
        if swap and max(swap[s]["max_rel_l2"] for s in modal_hf.SITES) > 1e-3:
            flags.append(f"{ck}: swap loader disagrees with HF from_pretrained: {swap}")
    shas = {(e, c["rendered_sha256"]) for rec in record["checkpoints"].values() for e, c in rec["checks"].items()}
    if len(shas) != len(record["efforts"]):
        flags.append(f"rendered prompts differ between checkpoints (distinct (effort, sha) pairs: {len(shas)})")
    for tag in modal_hf.VECTOR_FILES:
        if len({o["vectors"]["files"][tag]["sha256"] for o in outs}) != 1:
            flags.append(f"shards loaded different {tag} vector files")
    return flags


def build_record(args, prompts, outs, wall_s, est_usd) -> dict:
    checkpoints = {name: rec for out in outs for name, rec in out["ckpts"].items()}
    record = {
        "schema": "projection_v1", "tag": args.tag, "created_at": time.time(), "args": vars(args),
        "metric": "mean |h . v| / ||h|| at the last prompt token of the cot_on prompt (enable_thinking=True, ends '<think>\\n'); "
                  "in = input of layers[layer], out = output of layers[layer]; per-prompt signed dots and norms are stored",
        "layer": args.layer, "efforts": modal_hf.parse_efforts(args.efforts), "prompt_sets": modal_hf.prompt_meta(prompts),
        "vectors": outs[0]["vectors"], "versions": outs[0]["versions"],
        "hf_reference": next((o["hf_reference"] for o in outs if "hf_reference" in o), None),
        "batch_check": next((o["batch_check"] for o in outs if "batch_check" in o), None),
        "shards": [{"shard_id": o["shard_id"], "elapsed_s": o["elapsed_s"], "ckpts": list(o["ckpts"])} for o in outs],
        "wall_s": round(wall_s, 1), "est_cost_usd": est_usd,
        "checkpoints": checkpoints, "summary": modal_hf.flatten_means(checkpoints),
    }
    record["flags"] = sanity_flags(record, outs)
    return record


CONTROL_GROUPS = {"r1.pca(3)": ("r1.pca_seed101", "r1.pca_seed102", "r1.pca_seed103"), "r1.u_hat": ("r1.u_hat",),
                  "r2.pca(3)": ("r2.pca_101", "r2.pca_102", "r2.pca_103"), "r2.unrel": ("r2.unrelated",)}


def print_controls(record: dict) -> None:
    """Mean |h.v|/||h|| of the PCA / unrelated control vectors (mean over each group) at the first effort, layer output."""
    effort = record["efforts"][0]
    cell = {(r["ckpt"], r["effort"], r["set"], r["site"], r["vector"]): r["mean_abs_ratio"] for r in record["summary"]}
    print(f"\n== control vectors, effort={effort} site=layer{record['layer']}_out")
    cols = [(s, g) for s in SETS for g in CONTROL_GROUPS]
    print(f"{'checkpoint':20s}" + "".join(f"{g + '@' + s:>17s}" for s, g in cols))
    for ck in record["checkpoints"]:
        vals = [sum(cell[(ck, effort, s, "out", v)] for v in CONTROL_GROUPS[g]) / len(CONTROL_GROUPS[g]) for s, g in cols]
        print(f"{ck:20s}" + "".join(f"{x:17.5f}" for x in vals))


def print_summary(record: dict) -> None:
    """Aggregate numbers only: mean |h.v|/||h|| (reduction vs the family's intact checkpoint) per checkpoint."""
    cos = record["vectors"]["cos"]["r1.r_hat"]["r2.refusal"]
    for flag in record.get("flags", []):
        print(f"!! SANITY FLAG: {flag}")
    print(f"cos(r1.r_hat, r2.refusal) = {cos:.4f}   shards={len(record['shards'])} wall={record['wall_s']:.0f}s est_cost=${record['est_cost_usd']:.2f}")
    print(f"versions: {record['versions']}")
    if record.get("hf_reference"):
        print(f"hf from_pretrained reference: {record['hf_reference']}")
    batch_check = record.get("batch_check") or record.get("batch_vs_single")   # older records used the second name
    if batch_check:
        print(f"batching noise floor (first ckpt): {batch_check}")
    cell = {(r["ckpt"], r["effort"], r["set"], r["site"], r["vector"]): r for r in record["summary"]}
    for effort in record["efforts"]:
        for site in modal_hf.SITES:
            print(f"\n== effort={effort} site=layer{record['layer']}_{site}   cell = mean |h.v|/||h||  (reduction vs same-family intact)")
            cols = [(s, v) for s in SETS for v in HEADLINE_VECTORS]
            print(f"{'checkpoint':20s}" + "".join(f"{v + '@' + s:>30s}" for s, v in cols))
            for ck in record["checkpoints"]:
                cells = []
                for s, v in cols:
                    r = cell[(ck, effort, s, site, v)]
                    red = r["reduction_vs_intact"]
                    cells.append(f"{r['mean_abs_ratio']:.5f}" + (f" ({-100 * red:+6.1f}%)" if red else ""))
                print(f"{ck:20s}" + "".join(f"{c:>30s}" for c in cells))
    print_controls(record)
    print("\nrel-L2 of layer-48 output vs bf16_base (mean over prompts), first effort:")
    e0 = record["efforts"][0]
    for ck, rec in record["checkpoints"].items():
        if "rel_l2_vs_bf16_base" in rec:
            print(f"  {ck:20s} in={rec['rel_l2_vs_bf16_base'][e0]['in']['mean_rel_l2']:.4f} out={rec['rel_l2_vs_bf16_base'][e0]['out']['mean_rel_l2']:.4f}")
        if "swap_vs_hf_from_pretrained" in rec:
            print(f"  {ck:20s} swap-loader vs HF from_pretrained: {rec['swap_vs_hf_from_pretrained']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ckpts", default="all", help="'all', 'smoke', or comma-separated names (see v3.modal_hf.select_checkpoints)")
    ap.add_argument("--efforts", default="low,default", help="reasoning_effort renderings: low (Run 1 extraction), default (=xhigh, Run 2 extraction)")
    ap.add_argument("--shards", type=int, default=1, help="parallel GPU containers (each builds the model once)")
    ap.add_argument("--layer", type=int, default=modal_hf.LAYER)
    ap.add_argument("--batch-size", type=int, default=1, help="1 = one prompt at a time, no padding (the smoke test used 8)")
    ap.add_argument("--no-crosscheck", action="store_true", help="skip the HF from_pretrained reference on the BF16 base")
    ap.add_argument("--job", help="ledger job name (default hf-projection-<tag>)")
    ap.add_argument("--reserve-usd", type=float, help="default: worst case = shards x function timeout")
    ap.add_argument("--already-reserved", action="store_true", help="the ledger reservation exists (made with the ledger CLI)")
    ap.add_argument("--dry-run", action="store_true", help="build prompts and jobs, print counts, do not touch Modal or the ledger")
    ap.add_argument("--run1-prompts", default=str(RUN1_VAL))
    ap.add_argument("--run2-prompts", default=str(RUN2_EXTRACTION))
    args = ap.parse_args(argv)

    modal_hf.parse_efforts(args.efforts)   # fail before any reservation
    prompts = modal_hf.build_prompt_sets(args.run1_prompts, args.run2_prompts)
    ckpts = modal_hf.select_checkpoints(args.ckpts)
    jobs = make_jobs(args, prompts, ckpts)
    print(f"prompts={len(prompts)} ({', '.join(f'{k}: {len(v['ids'])}' for k, v in modal_hf.prompt_meta(prompts).items())}) "
          f"ckpts={[c['name'] for c in ckpts]} shards={[[c['name'] for c in j['ckpts']] for j in jobs]} efforts={args.efforts}")
    if args.dry_run:
        return 0

    job_name = args.job or f"hf-projection-{args.tag}"
    reserve_usd = args.reserve_usd or ledger.est_cost(modal_hf.GPU, len(jobs), modal_hf.TIMEOUT_S / 3600)
    if not args.already_reserved:
        ledger.reserve(job_name, reserve_usd, gpus=len(jobs))
    t0 = time.time()
    outs, error = None, None
    try:
        outs = run_modal(jobs)
    except BaseException as e:  # release the reservation even on Ctrl-C, then re-raise
        error = e
    wall_s = time.time() - t0
    # Containers are billed for their own runtime (+ startup), not for the image build: use the reported
    # elapsed time when the shards returned, else the whole wall time per shard (conservative).
    container_s = sum(o["elapsed_s"] + CONTAINER_OVERHEAD_S for o in outs) if outs else wall_s * len(jobs)
    est_usd = ledger.est_cost(modal_hf.GPU, 1, container_s / 3600)
    ledger.release(job_name, actual=est_usd)
    print(f"released {job_name}: wall={wall_s:.0f}s container_s={container_s:.0f} estimated cost=${est_usd:.2f} (reserved ${reserve_usd:.2f})")
    if error is not None:
        raise error
    record = build_record(args, prompts, outs, wall_s, est_usd)
    path = ROOT / "records" / f"projection_{args.tag}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(record) + "\n")
    print(f"saved {path}")
    print_summary(record)
    return 0


if __name__ == "__main__":
    sys.exit(main())
