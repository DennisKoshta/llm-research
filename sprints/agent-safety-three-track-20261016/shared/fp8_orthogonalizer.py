"""Static BF16 weight orthogonalization + matched [128, 128] E4M3 block-FP8 quantization for Qwen3.8-27B.

Implements Section 1.2 of the Research Specification:
- Step 1: Agentic + text residual-stream difference-in-means extraction with massive-activation outlier masking.
- Step 2: Static rank-1 projection W' = W - alpha * r_hat (r_hat^T W) on residual-writing matrices
  (self_attn.o_proj, linear_attn.out_proj, mlp.down_proj) in float32 -> BF16, followed by exact
  [128, 128] block-FP8 (E4M3 + BF16 weight_scale_inv) quantization matching Qwen/Qwen3.8-27B-FP8.
- Step 3: Variance-Matched PCA Subspace Control (3 seeds removing the exact same empirical activation
  variance on validation prompts as r_hat) and Unrelated Concept Direction Control (formal vs. casual register).
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple
import numpy as np
import torch


RESIDUAL_WRITER_SUFFIXES = (
    "self_attn.o_proj.weight",
    "linear_attn.out_proj.weight",
    "mlp.down_proj.weight",
)


def is_residual_writer(tensor_name: str, target_layers: List[int] | None = None, include_mtp: bool = True) -> bool:
    """Return True if tensor_name writes directly into the 5120-d residual stream in a target layer."""
    if not any(tensor_name.endswith(sfx) for sfx in RESIDUAL_WRITER_SUFFIXES):
        return False
    if tensor_name.startswith("mtp.layers."):
        return include_mtp
    if tensor_name.startswith("model.language_model.layers."):
        parts = tensor_name.split(".")
        layer_idx = int(parts[3])
        return target_layers is None or layer_idx in target_layers
    return False


def orthogonalize_weight_bf16(
    weight_bf16: torch.Tensor,
    direction: torch.Tensor | np.ndarray,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Project direction out of the output (row) space of a residual-writing matrix W in float32, returning BF16.

    In PyTorch Linear(in_features, out_features), weight has shape [out_features, in_features] = [d_model, d_in].
    For output y = W x, removing component along unit vector r_hat (with scaling alpha) is:
        W' = W - alpha * r_hat (r_hat^T W)
    When alpha == 0.0 or direction == 0, W' is bit-identical to weight_bf16.
    """
    if weight_bf16.dtype != torch.bfloat16:
        raise ValueError(f"Expected BF16 weight tensor, got {weight_bf16.dtype}")
    r = torch.as_tensor(direction, device=weight_bf16.device, dtype=torch.float32)
    if r.ndim != 1 or r.shape[0] != weight_bf16.shape[0]:
        raise ValueError(f"Direction shape {tuple(r.shape)} does not match weight output dim {weight_bf16.shape[0]}")
    if alpha == 0.0 or float(r.norm().item()) == 0.0:
        return weight_bf16.clone()
    r_norm = float(r.norm().item())
    if abs(r_norm - 1.0) > 1e-4:
        raise ValueError(f"Non-zero direction must be unit norm, got {r_norm}")
    w_f32 = weight_bf16.float()
    proj = torch.matmul(r, w_f32)  # [d_in]
    w_edited = w_f32 - (alpha * r).unsqueeze(1) * proj.unsqueeze(0)
    return w_edited.to(torch.bfloat16)


def quantize_block_fp8_128(
    weight_bf16: torch.Tensor,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a [M, N] BF16 weight matrix to [128, 128] block-FP8 (float8_e4m3fn + BF16 weight_scale_inv).

    Matches the official Qwen/Qwen3.8-27B-FP8 quantization scheme byte-for-byte on block scales.
    """
    if weight_bf16.ndim != 2:
        raise ValueError(f"Expected 2D weight matrix, got {weight_bf16.ndim}D")
    M, N = weight_bf16.shape
    if M % block_size != 0 or N % block_size != 0:
        raise ValueError(f"Shape {(M, N)} not divisible by block_size={block_size}")
    bm, bn = M // block_size, N // block_size
    blocks = weight_bf16.reshape(bm, block_size, bn, block_size).permute(0, 2, 1, 3).float()
    fmax = float(torch.finfo(torch.float8_e4m3fn).max)  # 448.0
    amax = blocks.abs().amax(dim=(-2, -1))
    scale_f32 = torch.where(amax > 0, amax / fmax, torch.ones_like(amax))
    scale_bf16 = scale_f32.to(torch.bfloat16)
    q_blocks = (blocks / scale_f32.unsqueeze(-1).unsqueeze(-1)).clamp(-fmax, fmax).to(torch.float8_e4m3fn)
    q_weight = q_blocks.permute(0, 2, 1, 3).reshape(M, N).contiguous()
    return q_weight, scale_bf16.contiguous()


def dequantize_block_fp8_128(
    q_weight: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    """Dequantize a [128, 128] block-FP8 weight back to BF16 for verification."""
    M, N = q_weight.shape
    bm, bn = M // block_size, N // block_size
    blocks = q_weight.reshape(bm, block_size, bn, block_size).permute(0, 2, 1, 3).float()
    deq = (blocks * scale_inv.float().unsqueeze(-1).unsqueeze(-1)).permute(0, 2, 1, 3).reshape(M, N)
    return deq.to(torch.bfloat16)


def compute_masked_dim_direction(
    pos_acts: np.ndarray,
    neg_acts: np.ndarray,
    outlier_rms_multiplier: float = 10.0,
) -> Dict[str, object]:
    """Compute massive-activation-masked normalized Difference-in-Means unit vector r_hat.

    Args:
        pos_acts: [N_pos, d] float64/float32 array (e.g. harmful or formal activations).
        neg_acts: [N_neg, d] float64/float32 array (e.g. harmless or casual activations).
        outlier_rms_multiplier: Channels whose combined RMS exceeds this multiple of the median
            channel RMS across all extraction prompts are masked to 0 before normalization.
    """
    pos = np.asarray(pos_acts, dtype=np.float64)
    neg = np.asarray(neg_acts, dtype=np.float64)
    pooled = np.concatenate([pos, neg], axis=0)
    channel_rms = np.sqrt(np.mean(np.square(pooled), axis=0))
    median_rms = float(np.median(channel_rms))
    threshold = outlier_rms_multiplier * median_rms
    outlier_mask = channel_rms > threshold
    outlier_indices = np.where(outlier_mask)[0].tolist()

    raw_diff = pos.mean(axis=0) - neg.mean(axis=0)
    raw_norm = float(np.linalg.norm(raw_diff))
    masked_diff = raw_diff.copy()
    masked_diff[outlier_mask] = 0.0
    masked_norm = float(np.linalg.norm(masked_diff))
    if masked_norm <= 0 or not np.isfinite(masked_norm):
        raise ValueError("Invalid masked difference norm")
    unit_vec = masked_diff / masked_norm
    return {
        "direction": unit_vec.astype(np.float64),
        "raw_norm": raw_norm,
        "masked_norm": masked_norm,
        "median_channel_rms": median_rms,
        "outlier_threshold_rms": threshold,
        "masked_outlier_channels": outlier_indices,
        "num_masked_channels": len(outlier_indices),
    }


def compute_variance_matched_pca_controls(
    val_acts: np.ndarray,
    r_hat: np.ndarray,
    seeds: Tuple[int, ...] = (101, 102, 103),
    pca_rank: int = 32,
    outlier_rms_multiplier: float = 10.0,
) -> Dict[str, object]:
    """Construct Variance-Matched PCA Subspace Control unit vectors (Section 1.2 Step 3.1).

    Each control vector v_s:
    1. Masks massive-activation outlier channels (matching r_hat's outlier mask) so control
       directions do not corrupt attention-sink / massive-activation channels.
    2. Lies strictly within the top-PCA subspace of non-outlier validation residual-stream activations.
    3. Is strictly orthogonal to r_hat (v_s . r_hat == 0).
    4. Removes the EXACT same empirical activation variance on validation prompts as r_hat:
       Var(X_val @ v_s) == Var(X_val @ r_hat) to machine precision (< 1e-10 relative error).
    """
    X = np.asarray(val_acts, dtype=np.float64)
    r = np.asarray(r_hat, dtype=np.float64)
    r = r / np.linalg.norm(r)

    # Identify massive-activation outlier channels (plus any channel already zeroed in r_hat)
    channel_rms = np.sqrt(np.mean(np.square(X), axis=0))
    median_rms = float(np.median(channel_rms))
    outlier_mask = (channel_rms > outlier_rms_multiplier * median_rms) | (np.abs(r) == 0.0)

    X_centered = X - X.mean(axis=0, keepdims=True)
    N = X_centered.shape[0]

    def emp_var(v: np.ndarray) -> float:
        proj = X_centered @ v
        return float(np.mean(np.square(proj)))

    target_var = emp_var(r)
    # Zero out outlier channels on a copy for PCA basis construction so v_s[outlier_mask] == 0
    X_clean = X_centered.copy()
    X_clean[:, outlier_mask] = 0.0
    # Project r out of X_clean so all PCA basis vectors are strictly orthogonal to r
    X_orth = X_clean - np.outer(X_clean @ r, r)
    # SVD of X_orth / sqrt(N)
    _, svals, vt = np.linalg.svd(X_orth / math.sqrt(N), full_matrices=False)
    eigvals = np.square(svals)
    K = min(pca_rank, len(eigvals))
    basis = vt[:K]  # [K, d], rows are orthonormal eigenvectors with variance eigvals[:K]
    basis_vars = eigvals[:K]

    if basis_vars[0] < target_var or basis_vars[-1] > target_var:
        # Expand K to full rank if target_var is below top-K minimum
        K_full = len(eigvals)
        basis = vt[:K_full]
        basis_vars = eigvals[:K_full]

    high_idx = np.where(basis_vars >= target_var)[0]
    low_idx = np.where(basis_vars < target_var)[0]
    if len(high_idx) == 0 or len(low_idx) == 0:
        raise ValueError(
            f"Target variance {target_var:.6g} outside PCA variance range [{basis_vars[-1]:.6g}, {basis_vars[0]:.6g}]"
        )

    controls = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        # Draw random unit combinations in the high-variance PCA group and low-variance PCA group
        w_h_coeffs = rng.normal(size=len(high_idx))
        w_h_coeffs /= np.linalg.norm(w_h_coeffs)
        w_l_coeffs = rng.normal(size=min(16, len(low_idx)))
        w_l_coeffs /= np.linalg.norm(w_l_coeffs)

        u_high = w_h_coeffs @ basis[high_idx]
        u_high /= np.linalg.norm(u_high)
        u_low = w_l_coeffs @ basis[low_idx[: len(w_l_coeffs)]]
        u_low -= np.dot(u_low, u_high) * u_high
        u_low /= np.linalg.norm(u_low)

        var_h = emp_var(u_high)
        var_l = emp_var(u_low)
        # Since u_high and u_low are in disjoint eigenspaces of X_orth^T X_orth / N, cross term is 0,
        # but compute explicitly for exact numerical precision:
        cross = float(np.mean((X_centered @ u_high) * (X_centered @ u_low)))
        # Solve cos^2(t)*var_h + sin^2(t)*var_l + 2*sin(t)*cos(t)*cross = target_var
        # Divide by cos^2(t), let z = tan(t):
        # (var_l - target_var)*z^2 + 2*cross*z + (var_h - target_var) = 0
        a = var_l - target_var
        b = 2.0 * cross
        c = var_h - target_var
        disc = max(0.0, b * b - 4.0 * a * c)
        z = (-b - math.sqrt(disc)) / (2.0 * a)
        cos_t = 1.0 / math.sqrt(1.0 + z * z)
        sin_t = z * cos_t
        v = cos_t * u_high + sin_t * u_low
        # Re-orthogonalize against r and normalize
        v -= np.dot(v, r) * r
        v /= np.linalg.norm(v)
        actual_var = emp_var(v)
        rel_err = abs(actual_var - target_var) / max(target_var, 1e-12)
        if rel_err > 1e-6:
            raise RuntimeError(f"Variance match error too large: {rel_err}")
        controls[f"pca_var_matched_seed{seed}"] = {
            "seed": seed,
            "direction": v.astype(np.float64),
            "target_variance": target_var,
            "actual_variance": actual_var,
            "variance_rel_err": rel_err,
            "dot_with_r_hat": float(np.dot(v, r)),
            "pca_high_dims": len(high_idx),
            "pca_low_dims": int(min(16, len(low_idx))),
        }
    return {
        "target_variance": target_var,
        "pca_top_eigenvalues": eigvals[:10].tolist(),
        "controls": controls,
    }
