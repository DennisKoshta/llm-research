"""Unit tests for Phase 1 Step 2 & Step 3: BF16 weight orthogonalization, [128,128] block-FP8 quantization,
zero-vector projection identity against Qwen/Qwen3.8-27B-FP8, and variance-matched PCA controls.
"""
from __future__ import annotations

import json
import os
import struct
import time
import urllib.request
from pathlib import Path
import numpy as np
import torch
from huggingface_hub import hf_hub_url

from fp8_orthogonalizer import (
    compute_masked_dim_direction,
    compute_variance_matched_pca_controls,
    dequantize_block_fp8_128,
    is_residual_writer,
    orthogonalize_weight_bf16,
    quantize_block_fp8_128,
)


def fetch_remote_tensor(repo: str, key: str) -> torch.Tensor:
    token = os.environ["HF_TOKEN"]
    idx_url = hf_hub_url(repo, "model.safetensors.index.json")
    req = urllib.request.Request(idx_url, headers={"Authorization": f"Bearer {token}"})
    wmap = json.loads(urllib.request.urlopen(req).read())["weight_map"]
    shard = wmap[key]
    s_url = hf_hub_url(repo, shard)
    req_h = urllib.request.Request(
        s_url, headers={"Authorization": f"Bearer {token}", "Range": "bytes=0-65535"}
    )
    buf = urllib.request.urlopen(req_h).read()
    hlen = struct.unpack("<Q", buf[:8])[0]
    if hlen > len(buf) - 8:
        req_h = urllib.request.Request(
            s_url, headers={"Authorization": f"Bearer {token}", "Range": f"bytes=0-{hlen + 16}"}
        )
        buf = urllib.request.urlopen(req_h).read()
    header = json.loads(buf[8 : 8 + hlen])
    meta = header[key]
    start, end = 8 + hlen + meta["data_offsets"][0], 8 + hlen + meta["data_offsets"][1]
    req_d = urllib.request.Request(
        s_url, headers={"Authorization": f"Bearer {token}", "Range": f"bytes={start}-{end - 1}"}
    )
    raw = urllib.request.urlopen(req_d).read()
    dt = {"BF16": torch.bfloat16, "F8_E4M3": torch.float8_e4m3fn, "F32": torch.float32}[meta["dtype"]]
    return torch.frombuffer(bytearray(raw), dtype=dt).reshape(meta["shape"]).clone()


def run_all_tests() -> dict:
    # 1. Test zero-vector projection identity and real Qwen/Qwen3.8-27B vs Qwen/Qwen3.8-27B-FP8 tensor match
    key = "model.language_model.layers.0.linear_attn.out_proj.weight"
    assert is_residual_writer(key, target_layers=[0])
    assert not is_residual_writer("model.language_model.layers.0.linear_attn.in_proj_qkv.weight")

    w_bf16 = fetch_remote_tensor("Qwen/Qwen3.8-27B", key)
    w_fp8_official = fetch_remote_tensor("Qwen/Qwen3.8-27B-FP8", key)
    s_fp8_official = fetch_remote_tensor("Qwen/Qwen3.8-27B-FP8", key + "_scale_inv")

    zero_dir = np.zeros(w_bf16.shape[0], dtype=np.float32)
    w_zero_proj = orthogonalize_weight_bf16(w_bf16, zero_dir, alpha=1.0)
    assert torch.equal(w_zero_proj, w_bf16), "Zero-vector projection must match unmodified BF16 weights bit-for-bit"

    rng = np.random.default_rng(42)
    unit_dir = rng.normal(size=w_bf16.shape[0])
    unit_dir /= np.linalg.norm(unit_dir)
    w_alpha0 = orthogonalize_weight_bf16(w_bf16, unit_dir, alpha=0.0)
    assert torch.equal(w_alpha0, w_bf16), "alpha=0 projection must match unmodified BF16 weights bit-for-bit"

    q_zero, s_zero = quantize_block_fp8_128(w_zero_proj)
    scale_exact_match = float((s_zero == s_fp8_official).float().mean().item())
    scale_max_rel_err = float(((s_zero.float() - s_fp8_official.float()).abs() / s_fp8_official.float()).max().item())
    fp8_byte_match = float((q_zero.view(torch.uint8) == w_fp8_official.view(torch.uint8)).float().mean().item())
    assert scale_exact_match == 1.0 and scale_max_rel_err == 0.0
    assert fp8_byte_match > 0.999

    # 2. Test rank-1 orthogonalization removes direction from residual writer output space
    r_t = torch.as_tensor(unit_dir, dtype=torch.float32)
    w_f32 = w_bf16.float()
    w_orth_f32 = w_f32 - r_t.unsqueeze(1) * torch.matmul(r_t, w_f32).unsqueeze(0)
    w_orth_bf16 = orthogonalize_weight_bf16(w_bf16, unit_dir, alpha=1.0)
    before_proj_norm = float(torch.matmul(r_t, w_f32).norm().item())
    after_f32_proj_norm = float(torch.matmul(r_t, w_orth_f32).norm().item())
    after_bf16_proj_norm = float(torch.matmul(r_t, w_orth_bf16.float()).norm().item())
    q_orth, s_orth = quantize_block_fp8_128(w_orth_bf16)
    w_orth_deq = dequantize_block_fp8_128(q_orth, s_orth)
    after_fp8_proj_norm = float(torch.matmul(r_t, w_orth_deq.float()).norm().item())
    assert after_f32_proj_norm / before_proj_norm < 1e-6
    assert after_bf16_proj_norm / before_proj_norm < 0.08
    assert after_fp8_proj_norm / before_proj_norm < 0.08

    # 3. Test massive-activation outlier channel masking & Variance-Matched PCA controls
    d = 5120
    pos_acts = rng.normal(loc=0.05, scale=1.0, size=(128, d))
    neg_acts = rng.normal(loc=-0.05, scale=1.0, size=(128, d))
    # Inject massive activation outliers into channels 17 and 1024
    pos_acts[:, [17, 1024]] = 500.0
    neg_acts[:, [17, 1024]] = 480.0
    dim_res = compute_masked_dim_direction(pos_acts, neg_acts, outlier_rms_multiplier=10.0)
    r_hat = dim_res["direction"]
    assert 17 in dim_res["masked_outlier_channels"] and 1024 in dim_res["masked_outlier_channels"]
    assert r_hat[17] == 0.0 and r_hat[1024] == 0.0
    assert abs(np.linalg.norm(r_hat) - 1.0) < 1e-12

    # Synthetic validation activations with realistic PCA spectrum
    scales = np.exp(-np.linspace(0, 4, 64))
    latent = rng.normal(size=(64, 64)) * scales
    proj_mat = rng.normal(size=(64, d)) / np.sqrt(d)
    val_acts = latent @ proj_mat
    pca_res = compute_variance_matched_pca_controls(val_acts, r_hat, seeds=(101, 102, 103), pca_rank=32)
    for name, ctrl in pca_res["controls"].items():
        assert abs(np.linalg.norm(ctrl["direction"]) - 1.0) < 1e-12
        assert abs(ctrl["dot_with_r_hat"]) < 1e-10
        assert ctrl["variance_rel_err"] < 1e-9

    receipt = {
        "created_at": time.time(),
        "passed": True,
        "tested_tensor": key,
        "zero_projection_bf16_exact_identity": True,
        "alpha0_projection_bf16_exact_identity": True,
        "official_fp8_scale_exact_match_fraction": scale_exact_match,
        "official_fp8_scale_max_rel_err": scale_max_rel_err,
        "official_fp8_code_exact_match_fraction": fp8_byte_match,
        "residual_projection_leakage_ratio_f32": after_f32_proj_norm / before_proj_norm,
        "residual_projection_leakage_ratio_bf16": after_bf16_proj_norm / before_proj_norm,
        "residual_projection_leakage_ratio_fp8": after_fp8_proj_norm / before_proj_norm,
        "outlier_masking_verified_channels": dim_res["masked_outlier_channels"],
        "pca_variance_matched_controls_verified": {
            k: {
                "target_variance": v["target_variance"],
                "actual_variance": v["actual_variance"],
                "variance_rel_err": v["variance_rel_err"],
                "dot_with_r_hat": v["dot_with_r_hat"],
            }
            for k, v in pca_res["controls"].items()
        },
    }
    out_path = (
        Path(__file__).resolve().parents[1] / "records" / "phase1_step2_zero_vector_and_fp8_unit_test.json"
    )
    out_path.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    run_all_tests()
