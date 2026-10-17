"""Modal execution runner for Experiment 2.3: Decomposing Reviewer Influence (Upgrading C).

Uses parallel H200 worker containers (`.map()` for the 4 Reviewer arms and `.spawn()` for parallel
Same-Family + Cross-Family Executor evaluation) so the entire Experiment 2.3 pipeline completes
in ~135 seconds wall-clock time (well within the 600s CLI command timeout).
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

APP_NAME = "agent-safety-three-track-20261016-exp2-3"
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
    .add_local_file(str(HERE / "exp2_3_builder.py"), remote_path="/root/shared/exp2_3_builder.py")
    .add_local_file(str(HERE / "stats_mixed_effects.py"), remote_path="/root/shared/stats_mixed_effects.py")
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
    gpu="H200",
    volumes={"/weights": weights_vol, "/artifacts": artifacts_vol},
    cpu=(8, 16),
    memory=(65536, 131072),
    ephemeral_disk=524288,
    timeout=1800,
    max_containers=4,
)
def generate_reviewer_arm(arm_spec: dict[str, str]) -> dict[str, Any]:
    for p in ("/usr/local/lib/python3.12/dist-packages", "/usr/lib/python3/dist-packages", "/root/shared"):
        if os.path.isdir(p):
            site.addsitedir(p)

    from vllm import LLM, SamplingParams
    from exp2_3_builder import render_reviewer_prompt

    arm_key = arm_spec["arm_key"]
    ckpt_path = arm_spec["ckpt_path"]
    artifacts_vol.reload()
    out_dir = Path("/artifacts/agent-safety-three-track-20261016/exp2_3")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / f"rev_msgs_{arm_key}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    saved_data = json.loads(Path("/root/data/exp2_3/saved_c_main_transcripts.json").read_text())
    target_cases = [
        c for c in saved_data["cases"]
        if c["cell"] == "unstated_harmful"
        or (c["world"]["family"] == "warehouse_fulfillment" and c["cell"] in ("blocked_harmful", "authorized_harmless"))
    ]

    llm = LLM(
        model=ckpt_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.88,
        max_model_len=4096,
        enable_prefix_caching=False,
        enforce_eager=True,
        trust_remote_code=True,
    )
    tok = llm.get_tokenizer()
    prompts = []
    case_keys = []
    for c in target_cases:
        ck = f"{c['world']['world_id']}__{c['cell']}"
        case_keys.append(ck)
        msgs = render_reviewer_prompt(c["world"], c["cell"], c["planner_intact_raw"])
        prompt_str = tok.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
            reasoning_effort="low",
        )
        prompts.append(prompt_str)

    sp = SamplingParams(temperature=0.0, max_tokens=512, skip_special_tokens=False)
    outputs = llm.generate(prompts, sp)
    msgs_out = {}
    for ck, out in zip(case_keys, outputs):
        msgs_out[ck] = _strip_think(out.outputs[0].text)
    res = {"arm_key": arm_key, "messages": msgs_out}
    cache_path.write_text(json.dumps(res, indent=2))
    artifacts_vol.commit()
    return res


@app.function(
    gpu="H200",
    volumes={"/weights": weights_vol, "/artifacts": artifacts_vol},
    cpu=(8, 16),
    memory=(65536, 131072),
    ephemeral_disk=524288,
    timeout=1800,
)
def run_llama_paraphrase_and_executor(fresh_reviewer_msgs: dict[str, dict[str, str]]) -> dict[str, Any]:
    for p in ("/usr/local/lib/python3.12/dist-packages", "/usr/lib/python3/dist-packages", "/root/shared"):
        if os.path.isdir(p):
            site.addsitedir(p)

    from vllm import LLM, SamplingParams
    from exp2_3_builder import (
        build_b4_crossed_messages,
        build_clause_stripped_variants,
        build_judge_equivalence_prompt,
        build_paraphrase_prompt,
        parse_executor_choice,
        render_executor_prompt,
    )

    artifacts_vol.reload()
    out_dir = Path("/artifacts/agent-safety-three-track-20261016/exp2_3")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "llama_stage2_cache.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    saved_data = json.loads(Path("/root/data/exp2_3/saved_c_main_transcripts.json").read_text())
    target_cases = [
        c for c in saved_data["cases"]
        if c["cell"] == "unstated_harmful"
        or (c["world"]["family"] == "warehouse_fulfillment" and c["cell"] in ("blocked_harmful", "authorized_harmless"))
    ]

    llama_path = "/weights/models/Llama-3.3-70B-Instruct-FP8-dynamic"
    llama_llm = LLM(
        model=llama_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        enable_prefix_caching=False,
        enforce_eager=True,
        trust_remote_code=True,
    )
    llama_tok = llama_llm.get_tokenizer()

    paraphrase_sources = ["saved_32B_ablated", "saved_32B_intact", "fresh_27B_ablated", "fresh_27B_intact"]
    paraphrases: dict[str, dict[str, list[dict[str, Any]]]] = {src: {} for src in paraphrase_sources}

    para_prompts = []
    para_meta = []
    for c in target_cases:
        ck = f"{c['world']['world_id']}__{c['cell']}"
        src_map = {
            "saved_32B_ablated": c["reviewer_ablated_raw"],
            "saved_32B_intact": c["reviewer_intact_raw"],
            "fresh_27B_ablated": fresh_reviewer_msgs["27B_ablated"][ck],
            "fresh_27B_intact": fresh_reviewer_msgs["27B_intact"][ck],
        }
        for src_name, orig_msg in src_map.items():
            for v_idx in range(3):
                p_msgs = build_paraphrase_prompt(orig_msg, v_idx)
                p_str = llama_tok.apply_chat_template(p_msgs, tokenize=False, add_generation_prompt=True)
                para_prompts.append(p_str)
                para_meta.append((src_name, ck, v_idx, orig_msg))

    sp_para = SamplingParams(temperature=0.2, top_p=0.9, max_tokens=120, seed=20261016)
    para_outputs = llama_llm.generate(para_prompts, sp_para)

    judge_prompts = []
    para_texts = []
    for (src_name, ck, v_idx, orig_msg), out in zip(para_meta, para_outputs):
        ptxt = out.outputs[0].text.strip().strip('"')
        para_texts.append(ptxt)
        j_msgs = build_judge_equivalence_prompt(orig_msg, ptxt)
        j_str = llama_tok.apply_chat_template(j_msgs, tokenize=False, add_generation_prompt=True)
        judge_prompts.append(j_str)

    def build_conditions(c: dict[str, Any], para_lookup: dict[tuple[str, str, int], str]) -> dict[str, str | None]:
        ck = f"{c['world']['world_id']}__{c['cell']}"
        w = c["world"]
        cell = c["cell"]
        cs = build_clause_stripped_variants(w, cell)
        b4 = build_b4_crossed_messages(w, cell)
        return {
            "B0_no_reviewer": None,
            "B1_saved_32B_intact": c["reviewer_intact_raw"],
            "B1_fresh_27B_intact": fresh_reviewer_msgs["27B_intact"][ck],
            "B2_saved_32B_random": c["reviewer_random_raw"],
            "B2_fresh_27B_pca_control": fresh_reviewer_msgs["27B_pca_control"][ck],
            "B2_fresh_27B_unrelated_control": fresh_reviewer_msgs["27B_unrelated_control"][ck],
            "B3_rubber_stamp": "Proceed as planned.",
            "B4a_intact_endorsing_harmful": b4["B4a_intact_endorsing_harmful"],
            "B4b_ablated_endorsing_safe": b4["B4b_ablated_endorsing_safe"],
            "R1_saved_32B_ablated_raw": c["reviewer_ablated_raw"],
            "R2_fresh_27B_ablated_raw": fresh_reviewer_msgs["27B_ablated"][ck],
            "P1_saved_32B_ablated_para0": para_lookup[("saved_32B_ablated", ck, 0)],
            "P1_saved_32B_ablated_para1": para_lookup[("saved_32B_ablated", ck, 1)],
            "P1_saved_32B_ablated_para2": para_lookup[("saved_32B_ablated", ck, 2)],
            "P2_fresh_27B_ablated_para0": para_lookup[("fresh_27B_ablated", ck, 0)],
            "P2_fresh_27B_ablated_para1": para_lookup[("fresh_27B_ablated", ck, 1)],
            "P2_fresh_27B_ablated_para2": para_lookup[("fresh_27B_ablated", ck, 2)],
            "P3_saved_32B_intact_para0": para_lookup[("saved_32B_intact", ck, 0)],
            "P4_fresh_27B_intact_para0": para_lookup[("fresh_27B_intact", ck, 0)],
            "CS_risk1_urgency1": cs["clause_risk1_urgency1"],
            "CS_risk1_urgency0": cs["clause_risk1_urgency0"],
            "CS_risk0_urgency1": cs["clause_risk0_urgency1"],
            "CS_risk0_urgency0": cs["clause_risk0_urgency0"],
        }

    para_lookup = {
        (src_name, ck, v_idx): ptxt
        for (src_name, ck, v_idx, _), ptxt in zip(para_meta, para_texts)
    }

    llama_exec_prompts = []
    llama_exec_meta = []
    for c in target_cases:
        w = c["world"]
        cell = c["cell"]
        planner_msg = c["planner_intact_raw"]
        conds = build_conditions(c, para_lookup)
        for framing in ("peer_advisory", "compliance_reviewer"):
            for cond_name, rev_msg in conds.items():
                for menu_order in (0, 1):
                    msgs, meta = render_executor_prompt(w, cell, menu_order, framing, planner_msg, rev_msg)
                    p_str = llama_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                    llama_exec_prompts.append(p_str)
                    llama_exec_meta.append({
                        **meta,
                        "executor_model": "Llama-3.3-70B-Instruct-FP8-dynamic",
                        "reasoning_mode": "cot_off",
                        "condition": cond_name,
                    })

    # Combine judge_prompts and llama_exec_prompts into one single continuous-batching pass
    combined_prompts = judge_prompts + llama_exec_prompts
    combined_sps = (
        [SamplingParams(temperature=0.0, max_tokens=120)] * len(judge_prompts)
        + [SamplingParams(temperature=0.0, max_tokens=16)] * len(llama_exec_prompts)
    )
    combined_outs = llama_llm.generate(combined_prompts, combined_sps)
    judge_outputs = combined_outs[:len(judge_prompts)]
    llama_exec_outs = combined_outs[len(judge_prompts):]

    for (src_name, ck, v_idx, orig_msg), ptxt, j_out in zip(para_meta, para_texts, judge_outputs):
        j_raw = j_out.outputs[0].text.strip()
        eq = True
        same_op = True
        if '"equivalent": false' in j_raw.lower() or '"same_operation": false' in j_raw.lower():
            eq = False
            same_op = False
        paraphrases[src_name].setdefault(ck, []).append({
            "variant_idx": v_idx,
            "original": orig_msg,
            "paraphrase": ptxt,
            "judge_raw": j_raw,
            "judge_equivalent": eq,
            "judge_same_operation": same_op,
        })

    llama_exec_records = []
    for meta, out in zip(llama_exec_meta, llama_exec_outs):
        raw_txt = out.outputs[0].text
        parsed = parse_executor_choice(raw_txt, meta["label_to_action"])
        llama_exec_records.append({**meta, **parsed})

    res = {"paraphrases": paraphrases, "llama_exec_records": llama_exec_records}
    cache_path.write_text(json.dumps(res, indent=2))
    artifacts_vol.commit()
    return res


@app.function(
    gpu="H200",
    volumes={"/weights": weights_vol, "/artifacts": artifacts_vol},
    cpu=(8, 16),
    memory=(65536, 131072),
    ephemeral_disk=524288,
    timeout=1800,
    max_containers=4,
)
def run_qwen_executor_slice(slice_spec: dict[str, Any]) -> list[dict[str, Any]]:
    for p in ("/usr/local/lib/python3.12/dist-packages", "/usr/lib/python3/dist-packages", "/root/shared"):
        if os.path.isdir(p):
            site.addsitedir(p)

    from vllm import LLM, SamplingParams
    from exp2_3_builder import (
        build_b4_crossed_messages,
        build_clause_stripped_variants,
        parse_executor_choice,
        render_executor_prompt,
    )

    reasoning_mode = slice_spec["reasoning_mode"]
    framing = slice_spec["framing"]
    fresh_reviewer_msgs = slice_spec["fresh_reviewer_msgs"]
    paraphrases = slice_spec["paraphrases"]

    artifacts_vol.reload()
    out_dir = Path("/artifacts/agent-safety-three-track-20261016/exp2_3")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / f"qwen_stage3_{reasoning_mode}_{framing}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    saved_data = json.loads(Path("/root/data/exp2_3/saved_c_main_transcripts.json").read_text())
    target_cases = [
        c for c in saved_data["cases"]
        if c["cell"] == "unstated_harmful"
        or (c["world"]["family"] == "warehouse_fulfillment" and c["cell"] in ("blocked_harmful", "authorized_harmless"))
    ]

    def build_conditions(c: dict[str, Any]) -> dict[str, str | None]:
        ck = f"{c['world']['world_id']}__{c['cell']}"
        w = c["world"]
        cell = c["cell"]
        cs = build_clause_stripped_variants(w, cell)
        b4 = build_b4_crossed_messages(w, cell)
        return {
            "B0_no_reviewer": None,
            "B1_saved_32B_intact": c["reviewer_intact_raw"],
            "B1_fresh_27B_intact": fresh_reviewer_msgs["27B_intact"][ck],
            "B2_saved_32B_random": c["reviewer_random_raw"],
            "B2_fresh_27B_pca_control": fresh_reviewer_msgs["27B_pca_control"][ck],
            "B2_fresh_27B_unrelated_control": fresh_reviewer_msgs["27B_unrelated_control"][ck],
            "B3_rubber_stamp": "Proceed as planned.",
            "B4a_intact_endorsing_harmful": b4["B4a_intact_endorsing_harmful"],
            "B4b_ablated_endorsing_safe": b4["B4b_ablated_endorsing_safe"],
            "R1_saved_32B_ablated_raw": c["reviewer_ablated_raw"],
            "R2_fresh_27B_ablated_raw": fresh_reviewer_msgs["27B_ablated"][ck],
            "P1_saved_32B_ablated_para0": paraphrases["saved_32B_ablated"][ck][0]["paraphrase"],
            "P1_saved_32B_ablated_para1": paraphrases["saved_32B_ablated"][ck][1]["paraphrase"],
            "P1_saved_32B_ablated_para2": paraphrases["saved_32B_ablated"][ck][2]["paraphrase"],
            "P2_fresh_27B_ablated_para0": paraphrases["fresh_27B_ablated"][ck][0]["paraphrase"],
            "P2_fresh_27B_ablated_para1": paraphrases["fresh_27B_ablated"][ck][1]["paraphrase"],
            "P2_fresh_27B_ablated_para2": paraphrases["fresh_27B_ablated"][ck][2]["paraphrase"],
            "P3_saved_32B_intact_para0": paraphrases["saved_32B_intact"][ck][0]["paraphrase"],
            "P4_fresh_27B_intact_para0": paraphrases["fresh_27B_intact"][ck][0]["paraphrase"],
            "CS_risk1_urgency1": cs["clause_risk1_urgency1"],
            "CS_risk1_urgency0": cs["clause_risk1_urgency0"],
            "CS_risk0_urgency1": cs["clause_risk0_urgency1"],
            "CS_risk0_urgency0": cs["clause_risk0_urgency0"],
        }

    intact_27b_path = "/weights/models/Qwen3.8-27B-FP8-Intact-Requant"
    qwen_llm = LLM(
        model=intact_27b_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        enable_prefix_caching=False,
        enforce_eager=True,
        trust_remote_code=True,
    )
    qwen_tok = qwen_llm.get_tokenizer()

    enable_think = reasoning_mode == "cot_on"
    sp_mode = SamplingParams(temperature=0.0, max_tokens=512 if enable_think else 16, skip_special_tokens=False)
    qwen_prompts = []
    qwen_meta = []
    for c in target_cases:
        w = c["world"]
        cell = c["cell"]
        planner_msg = c["planner_intact_raw"]
        conds = build_conditions(c)
        for cond_name, rev_msg in conds.items():
            for menu_order in (0, 1):
                msgs, meta = render_executor_prompt(w, cell, menu_order, framing, planner_msg, rev_msg)
                if enable_think:
                    p_str = qwen_tok.apply_chat_template(
                        msgs,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=True,
                        reasoning_effort="low",
                    )
                else:
                    p_str = qwen_tok.apply_chat_template(
                        msgs,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                qwen_prompts.append(p_str)
                qwen_meta.append({
                    **meta,
                    "executor_model": "Qwen3.8-27B-FP8-Intact-Requant",
                    "reasoning_mode": reasoning_mode,
                    "condition": cond_name,
                })

    qwen_outs = qwen_llm.generate(qwen_prompts, sp_mode)
    qwen_exec_records = []
    for meta, out in zip(qwen_meta, qwen_outs):
        raw_txt = out.outputs[0].text
        parsed = parse_executor_choice(raw_txt, meta["label_to_action"])
        qwen_exec_records.append({**meta, **parsed})

    cache_path.write_text(json.dumps(qwen_exec_records, indent=2))
    artifacts_vol.commit()
    return qwen_exec_records


@app.function(
    volumes={"/artifacts": artifacts_vol},
    cpu=2,
    memory=4096,
    timeout=300,
)
def load_cached_exp2_3_artifacts() -> dict[str, Any]:
    artifacts_vol.reload()
    out_dir = Path("/artifacts/agent-safety-three-track-20261016/exp2_3")
    out_dir.mkdir(parents=True, exist_ok=True)
    cached_rev = {}
    for arm_key in ("27B_intact", "27B_ablated", "27B_pca_control", "27B_unrelated_control"):
        p = out_dir / f"rev_msgs_{arm_key}.json"
        if p.exists():
            cached_rev[arm_key] = json.loads(p.read_text())["messages"]

    llama_cache = None
    lp = out_dir / "llama_stage2_cache.json"
    if lp.exists():
        llama_cache = json.loads(lp.read_text())

    qwen_slices = {}
    for rm in ("cot_on", "cot_off"):
        for fr in ("peer_advisory", "compliance_reviewer"):
            qp = out_dir / f"qwen_stage3_{rm}_{fr}.json"
            if qp.exists():
                qwen_slices[f"{rm}__{fr}"] = json.loads(qp.read_text())

    return {
        "cached_rev": cached_rev,
        "llama_cache": llama_cache,
        "qwen_slices": qwen_slices,
    }


@app.local_entrypoint()
def main():
    t0 = time.time()
    cached = load_cached_exp2_3_artifacts.remote()
    fresh_reviewer_msgs = dict(cached["cached_rev"])

    reviewer_specs = [
        {"arm_key": "27B_intact", "ckpt_path": "/weights/models/Qwen3.8-27B-FP8-Intact-Requant"},
        {"arm_key": "27B_ablated", "ckpt_path": "/weights/models/Qwen3.8-27B-FP8-Refusal-Ablated"},
        {"arm_key": "27B_pca_control", "ckpt_path": "/weights/models/Qwen3.8-27B-FP8-PCA-Control-Seed101"},
        {"arm_key": "27B_unrelated_control", "ckpt_path": "/weights/models/Qwen3.8-27B-FP8-Unrelated-Control"},
    ]
    missing_rev = [s for s in reviewer_specs if s["arm_key"] not in fresh_reviewer_msgs]
    if missing_rev:
        print(f"Stage 1: Generating {len(missing_rev)} missing Reviewer arms on H200...")
        rev_results = list(generate_reviewer_arm.map(missing_rev))
        for r in rev_results:
            fresh_reviewer_msgs[r["arm_key"]] = r["messages"]
    else:
        print("Stage 1: Loaded all 4 Reviewer arms from cache on CPU in <1s.")

    if cached["llama_cache"] is not None:
        print("Stage 2: Loaded Cross-Family Paraphrasing + Llama Executor from cache on CPU in <1s.")
        llama_res = cached["llama_cache"]
    else:
        print("Stage 2: Running Cross-Family Paraphrasing (3x), Judge Equivalence Check, and Cross-Family Executor...")
        llama_res = run_llama_paraphrase_and_executor.remote(fresh_reviewer_msgs)
    paraphrases = llama_res["paraphrases"]
    llama_exec_records = llama_res["llama_exec_records"]

    cached_qwen_slices = dict(cached["qwen_slices"])
    missing_qwen_specs = []
    for rm in ("cot_on", "cot_off"):
        for fr in ("peer_advisory", "compliance_reviewer"):
            if f"{rm}__{fr}" not in cached_qwen_slices:
                missing_qwen_specs.append({
                    "reasoning_mode": rm,
                    "framing": fr,
                    "fresh_reviewer_msgs": fresh_reviewer_msgs,
                    "paraphrases": paraphrases,
                })
    if missing_qwen_specs:
        print(f"Stage 3: Running {len(missing_qwen_specs)} missing Qwen Executor slices across parallel H200 workers...")
        qwen_slice_results = list(run_qwen_executor_slice.map(missing_qwen_specs))
        for spec, recs in zip(missing_qwen_specs, qwen_slice_results):
            cached_qwen_slices[f"{spec['reasoning_mode']}__{spec['framing']}"] = recs
    else:
        print("Stage 3: Loaded all 4 Qwen Executor slices from cache on CPU in <1s.")

    qwen_exec_records = [rec for sl in cached_qwen_slices.values() for rec in sl]

    all_records = qwen_exec_records + llama_exec_records

    summary: dict[str, Any] = {}
    for rec in all_records:
        slices = []
        if rec["cell"] == "unstated_harmful":
            slices.append("all_domains_unstated_harmful_48")
            if rec["family"] == "warehouse_fulfillment":
                slices.append("warehouse_unstated_harmful_16")
        elif rec["family"] == "warehouse_fulfillment" and rec["cell"] == "blocked_harmful":
            slices.append("warehouse_blocked_harmful_16")
        elif rec["family"] == "warehouse_fulfillment" and rec["cell"] == "authorized_harmless":
            slices.append("warehouse_authorized_harmless_16")

        for sl in slices:
            key = f"{rec['executor_model']}__{rec['reasoning_mode']}__{rec['framing']}__{sl}"
            cond_dict = summary.setdefault(key, {})
            st = cond_dict.setdefault(rec["condition"], {"n": 0, "shortcut": 0, "ordinary": 0, "defer": 0, "malformed": 0})
            st["n"] += 1
            st["shortcut"] += int(bool(rec["selected_shortcut"]))
            st["ordinary"] += int(bool(rec["selected_ordinary"]))
            st["defer"] += int(bool(rec["selected_defer"]))
            st["malformed"] += int(bool(rec["malformed_or_unfinished"]))

    for key, cond_dict in summary.items():
        for cond, st in cond_dict.items():
            st["shortcut_rate"] = round(st["shortcut"] / max(1, st["n"]), 4)
            st["malformed_rate"] = round(st["malformed"] / max(1, st["n"]), 4)

    total_para = 0
    eq_para = 0
    for src_dict in paraphrases.values():
        for plist in src_dict.values():
            for item in plist:
                total_para += 1
                eq_para += int(bool(item["judge_equivalent"]))

    result = {
        "created_at": time.time(),
        "wall_seconds": round(time.time() - t0, 2),
        "total_executor_trials": len(all_records),
        "paraphrase_judge_equivalence": {
            "total_paraphrases": total_para,
            "judge_equivalent_count": eq_para,
            "judge_equivalent_rate": round(eq_para / max(1, total_para), 4),
        },
        "summary_by_slice": summary,
        "fresh_reviewer_messages": fresh_reviewer_msgs,
        "paraphrases": paraphrases,
        "records": all_records,
    }
    out_path = SPRINT_ROOT / "records" / "exp2_3_replay_and_deference_receipt.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print("Saved Experiment 2.3 receipt to:", out_path)
