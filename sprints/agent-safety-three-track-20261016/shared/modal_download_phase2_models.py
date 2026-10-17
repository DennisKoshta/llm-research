"""Download Phase 2 external-validity and cross-family models into the curiosity-study-weights Modal Volume on CPU workers:
1. RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic (Cross-family Executor & Paraphraser for Exp 2.3 / 1.2)
2. Qwen/Qwen3.8-Flash-Next-FP8 (External-validity Intact benchmark for Exp 1.1 & 2.3)
3. orcarouter/Qwen3.8-Flash-Next-Uncensored-FP8 (Unified Scenario Author + External-validity Uncensored benchmark for Exp 1.1 & 2.3)
"""
from __future__ import annotations

import json
import os
import site
import time
from pathlib import Path
import modal

APP_NAME = "agent-safety-three-track-20261016-download-phase2"
weights_vol = modal.Volume.from_name("curiosity-study-weights")

image = (
    modal.Image.from_registry("vllm/vllm-openai:qwen38-flash-next", add_python="3.12")
    .entrypoint([])
    .pip_install("huggingface_hub[hf_xet]>=0.34.0,<1.0", "hf_transfer")
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONPATH": "/usr/local/lib/python3.12/dist-packages:/usr/lib/python3/dist-packages",
    })
)

app = modal.App(APP_NAME, image=image)

MODELS_TO_FETCH = {
    "Llama-3.3-70B-Instruct-FP8-dynamic": "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic",
    "Qwen3.8-Flash-Next-FP8": "Qwen/Qwen3.8-Flash-Next-FP8",
    "Qwen3.8-Flash-Next-Uncensored-FP8": "orcarouter/Qwen3.8-Flash-Next-Uncensored-FP8",
}


@app.function(
    volumes={"/weights": weights_vol},
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
    cpu=(16, 32),
    memory=(32768, 65536),
    ephemeral_disk=524288,
    timeout=3600,
    max_containers=3,
)
def download_phase2_model(model_key: str) -> dict:
    for p in ("/usr/local/lib/python3.12/dist-packages", "/usr/lib/python3/dist-packages"):
        if os.path.isdir(p):
            site.addsitedir(p)
    from huggingface_hub import HfApi, snapshot_download

    weights_vol.reload()
    repo_id = MODELS_TO_FETCH[model_key]
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    info = api.model_info(repo_id=repo_id)
    sha = info.sha

    dest = Path(f"/weights/models/{model_key}")
    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    existing_st = list(dest.glob("*.safetensors"))
    expected_st = [s for s in info.siblings if s.rfilename.endswith(".safetensors")]
    if len(existing_st) < len(expected_st) or not (dest / "config.json").exists():
        snapshot_download(
            repo_id=repo_id,
            revision=sha,
            local_dir=str(dest),
            token=token,
            max_workers=16,
        )
        weights_vol.commit()
    files = [p for p in dest.rglob("*") if p.is_file() and ".cache" not in str(p)]
    total_bytes = sum(p.stat().st_size for p in files)
    return {
        "model_key": model_key,
        "repo_id": repo_id,
        "revision": sha,
        "dest": str(dest),
        "file_count": len(files),
        "safetensor_count": len(list(dest.glob("*.safetensors"))),
        "expected_safetensor_count": len(expected_st),
        "total_gb": round(total_bytes / 1e9, 2),
        "elapsed_seconds": round(time.time() - t0, 2),
    }


@app.local_entrypoint()
def main():
    t0 = time.time()
    results = list(download_phase2_model.map(list(MODELS_TO_FETCH.keys())))
    receipt = {
        "created_at": time.time(),
        "wall_seconds": round(time.time() - t0, 2),
        "models": results,
    }
    out_path = (
        Path(__file__).resolve().parents[1] / "records" / "phase2_model_download_receipt.json"
    )
    out_path.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
