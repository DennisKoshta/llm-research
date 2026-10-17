"""Download Qwen/Qwen3.8-27B (BF16) and Qwen/Qwen3.8-27B-FP8 into the curiosity-study-weights Modal Volume on a CPU container,
and verify Qwen3_5ForConditionalGeneration / vLLM support in vllm/vllm-openai:qwen38-flash-next.
"""
from __future__ import annotations

import json
import os
import site
import time
from pathlib import Path
import modal

APP_NAME = "agent-safety-three-track-20261016-download"
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
    "Qwen3.8-27B": ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"),
    "Qwen3.8-27B-FP8": ("Qwen/Qwen3.8-27B-FP8", "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"),
}


@app.function(
    volumes={"/weights": weights_vol},
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
    cpu=(4, 8),
    memory=(16384, 32768),
    ephemeral_disk=524288,
    timeout=1800,
    max_containers=2,
)
def download_and_verify(model_key: str) -> dict:
    for p in ("/usr/local/lib/python3.12/dist-packages", "/usr/lib/python3/dist-packages"):
        if os.path.isdir(p):
            site.addsitedir(p)
    import transformers
    import vllm
    from huggingface_hub import snapshot_download

    repo_id, revision = MODELS_TO_FETCH[model_key]
    dest = Path(f"/weights/models/{model_key}")
    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(dest),
        token=os.environ.get("HF_TOKEN"),
        max_workers=8,
    )
    weights_vol.commit()
    files = [p for p in dest.rglob("*") if p.is_file() and ".cache" not in str(p)]
    total_bytes = sum(p.stat().st_size for p in files)
    has_qwen35 = hasattr(transformers, "Qwen3_5ForConditionalGeneration")
    return {
        "model_key": model_key,
        "repo_id": repo_id,
        "revision": revision,
        "dest": str(dest),
        "file_count": len(files),
        "total_gb": round(total_bytes / 1e9, 2),
        "elapsed_seconds": round(time.time() - t0, 2),
        "transformers_version": transformers.__version__,
        "vllm_version": vllm.__version__,
        "has_Qwen3_5ForConditionalGeneration": has_qwen35,
    }


@app.local_entrypoint()
def main():
    t0 = time.time()
    results = list(download_and_verify.map(list(MODELS_TO_FETCH.keys())))
    receipt = {
        "created_at": time.time(),
        "wall_seconds": round(time.time() - t0, 2),
        "models": results,
    }
    out_path = (
        Path(__file__).resolve().parents[1] / "records" / "phase1_model_download_receipt.json"
    )
    out_path.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
