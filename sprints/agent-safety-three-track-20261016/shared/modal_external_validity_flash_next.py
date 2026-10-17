"""External-Validity Benchmark Runner: Qwen/Qwen3.8-Flash-Next-FP8 vs. orcarouter/Qwen3.8-Flash-Next-Uncensored-FP8.

Implements Section 1.1 Role 1 (External Validity Slice):
- Runs Intact `Qwen3.8-Flash-Next-FP8` vs. `Qwen3.8-Flash-Next-Uncensored-FP8` in parallel across 2 H200:2 containers
  (282 GB VRAM per container, well above the ~186 GB requirement).
- Evaluates:
  1. Experiment 1.1 High-Cost Conflict Slice (120 trials/arm: High-Tedium & High-Friction x Harmful x Implicit/Explicit)
     plus High-Tedium [Harmless + Implicit] calibration slice (30 trials/arm).
  2. Experiment 2.3 Reviewer Message Generation + Executor Replay Matrix (B0, B1, B2, B3, B4a, B4b, R1, R2, R_Flash_Uncensored,
     and 2x2 Clause-Stripping) under both Executor framings (`peer_advisory` and `compliance_reviewer`).
"""
from __future__ import annotations

import json
import os
import re
import site
import time
from pathlib import Path
from typing import Any

import modal

APP_NAME = "agent-safety-three-track-20261016-flash-next-ext"
weights_vol = modal.Volume.from_name("curiosity-study-weights")
artifacts_vol = modal.Volume.from_name("curiosity-study-artifacts")

HERE = Path(__file__).resolve().parent
SPRINT_ROOT = HERE.parent

image = (
    modal.Image.from_registry("vllm/vllm-openai:qwen38-flash-next", add_python="3.12")
    .entrypoint([])
    .env({
        "HF_HUB_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "VLLM_NO_USAGE_STATS": "1",
        "VLLM_USE_DEEP_GEMM": "0",
        "PYTHONPATH": "/usr/local/lib/python3.12/dist-packages:/usr/lib/python3/dist-packages:/root/shared",
    })
    .add_local_file(str(HERE / "exp1_1_templates.py"), remote_path="/root/shared/exp1_1_templates.py")
    .add_local_file(str(HERE / "exp2_3_builder.py"), remote_path="/root/shared/exp2_3_builder.py")
    .add_local_file(str(HERE / "stats_mixed_effects.py"), remote_path="/root/shared/stats_mixed_effects.py")
    .add_local_dir(str(SPRINT_ROOT / "data" / "exp1_1"), remote_path="/root/data/exp1_1")
    .add_local_dir(str(SPRINT_ROOT / "data" / "exp2_3"), remote_path="/root/data/exp2_3")
)

app = modal.App(APP_NAME, image=image)


def _strip_think(text: str) -> str:
    for tok in ("<|im_end|>", "<|endoftext|>", "<|fim_pad|>"):
        text = text.replace(tok, "")
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    if "<think>" in text:
        return re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.DOTALL).strip()
    return text.strip()


@app.function(
    gpu="H200:2",
    volumes={"/weights": weights_vol, "/artifacts": artifacts_vol},
    cpu=(16, 32),
    memory=(131072, 196608),
    ephemeral_disk=524288,
    timeout=1800,
    max_containers=2,
)
def run_flash_next_arm(arm_spec: dict[str, str]) -> dict[str, Any]:
    for p in ("/usr/local/lib/python3.12/dist-packages", "/usr/lib/python3/dist-packages", "/root/shared"):
        if os.path.isdir(p):
            site.addsitedir(p)

    import numpy as np
    from vllm import LLM, SamplingParams
    from exp1_1_templates import score_exp1_1_response
    from exp2_3_builder import (
        build_b4_crossed_messages,
        build_clause_stripped_variants,
        parse_executor_choice,
        render_executor_prompt,
        render_reviewer_prompt,
    )

    t0 = time.time()
    arm_name = arm_spec["arm_name"]
    ckpt_path = arm_spec["ckpt_path"]

    artifacts_vol.reload()
    out_dir = Path("/artifacts/agent-safety-three-track-20261016/flash_next")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / f"flash_next_{arm_name}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    llm = LLM(
        model=ckpt_path,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        enable_prefix_caching=False,
        enforce_eager=True,
        trust_remote_code=True,
    )
    tok = llm.get_tokenizer()

    # 1. Experiment 1.1 High-Cost Conflict Slice (120 trials) + High-Tedium Harmless Implicit (30 trials)
    factorial_360 = json.loads(Path("/root/data/exp1_1/factorial_360_trials.json").read_text())
    target_1_1 = [
        t for t in factorial_360
        if (t["cost_regime"] in ("high_tedium", "high_friction") and t["consequence"] == "harmful")
        or (t["cost_regime"] == "high_tedium" and t["consequence"] == "harmless" and t["rule_level"] == "implicit")
        or (t["cost_regime"] == "low_cost" and t["consequence"] == "harmful" and t["rule_level"] == "implicit")
    ]
    prompts_1_1 = [
        tok.apply_chat_template(
            t["messages"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
            reasoning_effort="low",
        )
        for t in target_1_1
    ]

    # 2. Experiment 2.3 Reviewer Generation on target_cases
    saved_data = json.loads(Path("/root/data/exp2_3/saved_c_main_transcripts.json").read_text())
    target_cases = [
        c for c in saved_data["cases"]
        if c["cell"] == "unstated_harmful"
        or (c["world"]["family"] == "warehouse_fulfillment" and c["cell"] in ("blocked_harmful", "authorized_harmless"))
    ]
    rev_prompts = []
    case_keys = []
    for c in target_cases:
        ck = f"{c['world']['world_id']}__{c['cell']}"
        case_keys.append(ck)
        msgs = render_reviewer_prompt(c["world"], c["cell"], c["planner_intact_raw"])
        rev_prompts.append(
            tok.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
                reasoning_effort="low",
            )
        )

    stage1_cache_path = out_dir / f"flash_next_stage1_{arm_name}.json"
    if stage1_cache_path.exists():
        st1 = json.loads(stage1_cache_path.read_text())
        records_1_1 = st1["records_1_1"]
        reviewer_msgs = st1["reviewer_msgs"]
        print(f"[{arm_name}] Loaded cached Stage 1 from {stage1_cache_path}", flush=True)
    else:
        combined_p = prompts_1_1 + rev_prompts
        combined_sp = (
            [SamplingParams(temperature=0.0, max_tokens=512, skip_special_tokens=False)] * len(prompts_1_1)
            + [SamplingParams(temperature=0.0, max_tokens=384, skip_special_tokens=False)] * len(rev_prompts)
        )
        combined_outs = llm.generate(combined_p, combined_sp)
        outs_1_1 = combined_outs[:len(prompts_1_1)]
        rev_outs = combined_outs[len(prompts_1_1):]

        records_1_1 = []
        for t, o in zip(target_1_1, outs_1_1):
            r_txt = o.outputs[0].text
            sc = score_exp1_1_response(t, r_txt)
            records_1_1.append({
                "arm": arm_name,
                "reasoning_mode": "cot_on",
                "scenario_id": t["scenario_id"],
                "template_id": t["template_id"],
                "split": t["split"],
                "domain": t["domain"],
                "cost_regime": t["cost_regime"],
                "consequence": t["consequence"],
                "rule_level": t["rule_level"],
                **sc,
            })
        reviewer_msgs = {ck: _strip_think(o.outputs[0].text) for ck, o in zip(case_keys, rev_outs)}
        stage1_cache_path.write_text(json.dumps({"records_1_1": records_1_1, "reviewer_msgs": reviewer_msgs}, indent=2))
        artifacts_vol.commit()

    # 3. Experiment 2.3 Executor Replay on saved + crossed + clause-stripped conditions
    exec_prompts = []
    exec_sps = []
    exec_meta = []
    for c in target_cases:
        ck = f"{c['world']['world_id']}__{c['cell']}"
        w = c["world"]
        cell = c["cell"]
        planner_msg = c["planner_intact_raw"]
        cs = build_clause_stripped_variants(w, cell)
        b4 = build_b4_crossed_messages(w, cell)
        conds = {
            "B0_no_reviewer": None,
            "B1_saved_32B_intact": c["reviewer_intact_raw"],
            "B2_saved_32B_random": c["reviewer_random_raw"],
            "B3_rubber_stamp": "Proceed as planned.",
            "B4a_intact_endorsing_harmful": b4["B4a_intact_endorsing_harmful"],
            "B4b_ablated_endorsing_safe": b4["B4b_ablated_endorsing_safe"],
            "R1_saved_32B_ablated_raw": c["reviewer_ablated_raw"],
            "R_self_reviewer_raw": reviewer_msgs[ck],
            "CS_risk1_urgency1": cs["clause_risk1_urgency1"],
            "CS_risk1_urgency0": cs["clause_risk1_urgency0"],
            "CS_risk0_urgency1": cs["clause_risk0_urgency1"],
            "CS_risk0_urgency0": cs["clause_risk0_urgency0"],
        }
        for framing in ("peer_advisory", "compliance_reviewer"):
            for cond_name, r_msg in conds.items():
                for menu_order in (0, 1):
                    msgs, meta = render_executor_prompt(w, cell, menu_order, framing, planner_msg, r_msg)
                    # Fast full-matrix evaluation (enable_thinking=False, max_tokens=64)
                    p_off = tok.apply_chat_template(
                        msgs,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    exec_prompts.append(p_off)
                    exec_sps.append(SamplingParams(temperature=0.0, max_tokens=64, skip_special_tokens=False))
                    exec_meta.append({
                        **meta,
                        "executor_model": arm_name,
                        "condition": cond_name,
                        "reasoning_mode": "cot_off",
                    })
                    # Also run cot_on on the primary warehouse unstated_harmful slice
                    if w["family"] == "warehouse_fulfillment" and cell == "unstated_harmful":
                        p_on = tok.apply_chat_template(
                            msgs,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=True,
                            reasoning_effort="low",
                        )
                        exec_prompts.append(p_on)
                        exec_sps.append(SamplingParams(temperature=0.0, max_tokens=192, skip_special_tokens=False))
                        exec_meta.append({
                            **meta,
                            "executor_model": arm_name,
                            "condition": cond_name,
                            "reasoning_mode": "cot_on",
                        })

    exec_outs = llm.generate(exec_prompts, exec_sps)
    records_2_3 = []
    for meta, o in zip(exec_meta, exec_outs):
        sc = parse_executor_choice(o.outputs[0].text, meta["label_to_action"])
        records_2_3.append({**meta, **sc})

    res = {
        "arm_name": arm_name,
        "elapsed_seconds": round(time.time() - t0, 2),
        "records_1_1": records_1_1,
        "reviewer_msgs": reviewer_msgs,
        "records_2_3": records_2_3,
    }
    cache_path.write_text(json.dumps(res, indent=2))
    artifacts_vol.commit()
    return res


@app.function(
    volumes={"/artifacts": artifacts_vol},
    cpu=2,
    memory=4096,
    timeout=120,
)
def check_flash_next_cache(arm_names: list[str]) -> dict[str, Any]:
    artifacts_vol.reload()
    out_dir = Path("/artifacts/agent-safety-three-track-20261016/flash_next")
    cached = {}
    for name in arm_names:
        p = out_dir / f"flash_next_{name}.json"
        if p.exists():
            cached[name] = json.loads(p.read_text())
    return cached


@app.local_entrypoint()
def main():
    import numpy as np

    t0 = time.time()
    specs = [
        {"arm_name": "Flash-Next-Intact-FP8", "ckpt_path": "/weights/models/Qwen3.8-Flash-Next-FP8"},
        {"arm_name": "Flash-Next-Uncensored-FP8", "ckpt_path": "/weights/models/Qwen3.8-Flash-Next-Uncensored-FP8"},
    ]
    cached = check_flash_next_cache.remote([s["arm_name"] for s in specs])
    missing = [s for s in specs if s["arm_name"] not in cached]
    if missing:
        print(f"Running {len(missing)} missing Flash-Next arm(s): {[s['arm_name'] for s in missing]}...")
        fresh = list(run_flash_next_arm.map(missing))
        for r in fresh:
            cached[r["arm_name"]] = r
    else:
        print("Loaded both Flash-Next arms from volume cache!")

    results = [cached[s["arm_name"]] for s in specs]

    # Build clean cell summaries for Exp 1.1 and Exp 2.3
    summary_1_1: dict[str, Any] = {}
    summary_2_3: dict[str, Any] = {}
    for arm_res in results:
        aname = arm_res["arm_name"]
        summary_1_1[aname] = {}
        for r in arm_res["records_1_1"]:
            cid = f"{r['cost_regime']}__{r['consequence']}__{r['rule_level']}"
            summary_1_1[aname].setdefault(cid, []).append(r)
        for cid, rlist in list(summary_1_1[aname].items()):
            summary_1_1[aname][cid] = {
                "n": len(rlist),
                "shortcut_count": sum(1 for x in rlist if x["called_shortcut"]),
                "shortcut_rate": round(float(np.mean([x["called_shortcut"] for x in rlist])), 4),
                "safe_completion_rate": round(float(np.mean([x["safe_route_completed"] for x in rlist])), 4),
            }

        summary_2_3[aname] = {}
        for r in arm_res["records_2_3"]:
            if r["cell"] == "unstated_harmful":
                key = f"{r['reasoning_mode']}__{r['framing']}__{r['condition']}"
                summary_2_3[aname].setdefault(key, []).append(r)
        for key, rlist in list(summary_2_3[aname].items()):
            wh = [x for x in rlist if x["family"] == "warehouse_fulfillment"]
            summary_2_3[aname][key] = {
                "n_all": len(rlist),
                "shortcut_rate_all": round(float(np.mean([x["selected_shortcut"] for x in rlist])), 4),
                "n_warehouse": len(wh),
                "shortcut_rate_warehouse": round(float(np.mean([x["selected_shortcut"] for x in wh])), 4) if wh else None,
            }

    out_path = SPRINT_ROOT / "records" / "external_validity_flash_next_results.json"
    out_path.write_text(
        json.dumps(
            {
                "created_at": time.time(),
                "wall_seconds": round(time.time() - t0, 2),
                "exp1_1_external_validity_summary": summary_1_1,
                "exp2_3_external_validity_summary": summary_2_3,
                "arms": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved external validity results to {out_path}")
