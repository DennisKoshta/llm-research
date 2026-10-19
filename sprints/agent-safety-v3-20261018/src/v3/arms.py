"""Checkpoint registry. Paths are container paths under the `curiosity-study-weights` volume (/weights).

Both families are READ-ONLY donors. The active family for Phase 2/3 is chosen by the Phase 1 positive
control and recorded in records/phase1_family_decision.json (see ACTIVE_FAMILY_FILE).
"""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
ACTIVE_FAMILY_FILE = ROOT / "records" / "phase1_family_decision.json"

ARMS_27B = ("intact", "refusal_ablated", "pca_101", "pca_102", "pca_103", "unrelated")

FAMILY_R1 = {  # Run 1 (three-track): replaces 130 residual-writer tensors; PCA controls orthogonal to r_hat
    "intact": "/weights/models/Qwen3.8-27B-FP8-Intact-Requant",
    "refusal_ablated": "/weights/models/Qwen3.8-27B-FP8-Refusal-Ablated",
    "pca_101": "/weights/models/Qwen3.8-27B-FP8-PCA-Control-Seed101",
    "pca_102": "/weights/models/Qwen3.8-27B-FP8-PCA-Control-Seed102",
    "pca_103": "/weights/models/Qwen3.8-27B-FP8-PCA-Control-Seed103",
    "unrelated": "/weights/models/Qwen3.8-27B-FP8-Unrelated-Control",
}
_R2 = "/weights/agent-safety-clean-v2-20261017/checkpoints_full_residual_v2"
FAMILY_R2 = {  # Run 2 (clean-v2): full residual incl. embed_tokens; NOTE needs hf_overrides regex->literal rewrite
    a: f"{_R2}/{a}" for a in ARMS_27B
}
FAMILIES = {"r1": FAMILY_R1, "r2": FAMILY_R2}

BF16_27B = "/weights/models/Qwen3.8-27B"
OFFICIAL_FP8_27B = "/weights/models/Qwen3.8-27B-FP8"
EXTERNAL = {
    "flash_next": "/weights/models/Qwen3.8-Flash-Next-FP8",
    "flash_next_uncensored": "/weights/models/Qwen3.8-Flash-Next-Uncensored-FP8",
    "llama70b": "/weights/models/Llama-3.3-70B-Instruct-FP8-dynamic",
}

# r_hat vectors (unit-norm float64, dim 5120). Modal artifact-volume paths; layer index 48.
R_HAT_LAYER = 48
VECTORS = {
    "r1": "/artifacts/agent-safety-three-track-20261016/phase1/phase1_directions.npz",   # keys: best_layer, r_hat, pca_seed101..103, u_hat
    "r2": "/artifacts/agent-safety-clean-v2-20261017/activations/phase1_directions.npz",  # keys: refusal, pca_101..103, unrelated
}


def active_family() -> str:
    if ACTIVE_FAMILY_FILE.exists():
        return json.loads(ACTIVE_FAMILY_FILE.read_text())["family"]
    raise RuntimeError("Phase 1 family decision not recorded yet")


def checkpoint_path(arm: str, family: str | None = None) -> str:
    if arm in EXTERNAL:
        return EXTERNAL[arm]
    return FAMILIES[family or active_family()][arm]
