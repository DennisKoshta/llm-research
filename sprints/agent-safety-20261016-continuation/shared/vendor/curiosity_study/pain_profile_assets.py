"""Pinned public assets and original numerical environment for the method control."""

import hashlib
from importlib.metadata import version
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import time

BASE_REPO = "Qwen/Qwen2.5-32B-Instruct"
BASE_REVISION = "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"
ADAPTER_REPO = "Valen92/pain-adapters"
ADAPTER_REVISION = "b64bd64b4bc7ca6e0733a489b8372a099d55ef05"
ADAPTER_NAME = "adapter_Qwen_2.5_32B_instruct.tar.gz"
ADAPTER_SHA = "cd96d4d43a3f7d6a8c67804ed4f4ee567ee4942e168804c757e69937b857127f"
REFERENCE = Path("references/pain-axis-positive-control")
ASSETS = Path("references/pain-axis-assets")
PROFILE = REFERENCE / "upstream/v2_controls/choice_controls/profile"
VECTORS = REFERENCE / "upstream/v2_controls/choice_controls/vectors/common"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def check_environment(lock_path):
    pins = dict(re.findall(r"^([A-Za-z0-9_.-]+)==([^\s\\]+)", Path(lock_path).read_text(), re.M))
    require(len(pins) == 57, "Unexpected numerical package lock")
    actual = {name: version(name) for name in pins}
    require(actual == pins, "Numerical environment differs from the complete lock")
    return dict(lock_byte_sha256=sha(lock_path), distributions=actual)


def verify_adapter(root, extracted):
    receipt = read(Path(root) / ASSETS / "asset-receipt.json")
    require(receipt["archive_sha256"] == ADAPTER_SHA, "Wrong adapter inspection receipt")
    expected = {}
    for member in receipt["members"]:
        if member["type"] != "file":
            continue
        relative = PurePosixPath(member["name"])
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe adapter member")
        path = Path(extracted) / relative
        require(path.is_file() and not path.is_symlink(), "Missing or linked adapter member")
        require(path.stat().st_size == member["byte_count"] and sha(path) == member["sha256"],
                "Adapter member identity changed")
        expected[str(relative)] = member["sha256"]
    actual = {str(p.relative_to(extracted)) for p in Path(extracted).rglob("*") if p.is_file()}
    require(actual == set(expected), "Unexpected adapter files")
    configs = list(Path(extracted).rglob("adapter_config.json"))
    require(len(configs) == 1, "Ambiguous adapter directory")
    require(read(configs[0])["base_model_name_or_path"] == BASE_REPO, "Adapter base mismatch")
    return configs[0].parent, expected


def verify_base(root, snapshot, *, deadline=None):
    root, snapshot = Path(root), Path(snapshot)
    pins = read(root / ASSETS / "base-weight-pins.json")
    require(pins["repo"] == BASE_REPO and pins["revision"] == BASE_REVISION,
            "Wrong base weight pins")
    files = []
    for row in pins["files"]:
        if deadline is not None:
            require(time.time() < deadline, "Asset verification deadline reached")
        path = snapshot / row["path"]
        require(Path(row["path"]).name == row["path"], "Unexpected weight path")
        require(path.stat().st_size == row["size"] and sha(path) == row["sha256"],
                "Base weight identity changed")
        files.append(dict(row))
    metadata = read(root / REFERENCE / "model-files.json")["files"]
    for row in metadata:
        if row["repository"] == BASE_REPO and row["path"] in ("config.json", "model.safetensors.index.json"):
            require(sha(snapshot / row["path"]) == row["sha256"], "Base metadata changed")
    generation = root / ASSETS / "base-generation-config.json"
    require(sha(snapshot / "generation_config.json") == sha(generation), "Base generation config changed")
    return dict(repo=BASE_REPO, revision=BASE_REVISION, files=files,
                generation_config_sha256=sha(generation))


def prefetch(root, cache_root, deadline):
    """CPU-only bounded preparation. No model is instantiated or imported here."""
    from huggingface_hub import hf_hub_download, snapshot_download

    root, cache_root = Path(root), Path(cache_root)
    require(time.time() < deadline, "Expired asset prefetch")
    hf_cache = cache_root / "huggingface"
    snapshot = Path(snapshot_download(BASE_REPO, revision=BASE_REVISION, cache_dir=hf_cache,
                                     allow_patterns=["*.safetensors", "*.json"], max_workers=4, token=False))
    base_receipt = verify_base(root, snapshot, deadline=deadline)
    require(time.time() < deadline, "Asset download deadline reached")
    archive = Path(hf_hub_download(ADAPTER_REPO, ADAPTER_NAME, revision=ADAPTER_REVISION,
                                  cache_dir=hf_cache, token=False))
    require(sha(archive) == ADAPTER_SHA, "Adapter archive identity changed")
    extracted = cache_root / "adapter"
    if not extracted.exists():
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                path = PurePosixPath(member.name)
                require((member.isfile() or member.isdir()) and not path.is_absolute()
                        and ".." not in path.parts, "Unsafe adapter archive")
            extracted.mkdir(parents=True)
            tar.extractall(extracted, filter="data")
    adapter, adapter_files = verify_adapter(root, extracted)
    require(time.time() < deadline, "Asset prefetch exceeded its deadline")
    return dict(base=base_receipt, snapshot_path=str(snapshot), adapter_path=str(adapter),
                adapter_archive_sha256=ADAPTER_SHA, adapter_files=adapter_files,
                model_inference_performed=False)


def load_vectors(root, device):
    import torch
    from safetensors.torch import load_file

    root = Path(root)
    pins = read(root / PROFILE / "inputs/portable_vector_pins.json")
    require(pins["layer"] == 61 and pins["width"] == 5120, "Wrong vector site")
    for name, expected in pins["files"].items():
        require(sha(root / VECTORS / name) == expected, "Vector-file identity changed")
    pain = load_file(root / VECTORS / "Qwen_2.5_32B_instruct_pain_vectors.safetensors")["s2_pain_vector"]
    fear = torch.load(root / VECTORS / "fear_Qwen_2.5_32B_instruct.pt", map_location="cpu",
                      weights_only=True)["fear_vector_matched_fp32"]
    receipt = {}
    for name, vector, key in (("pain", pain, "s2_pain_vector"), ("fear", fear, "fear_vector_matched_fp32")):
        require(vector.dtype == torch.float32 and vector.shape == (5120,)
                and bool(torch.isfinite(vector).all()), "Invalid saved vector")
        identity = hashlib.sha256(vector.contiguous().numpy().tobytes()).hexdigest()
        require(identity == pins["fp32_tensor_sha256"][key], "Saved tensor identity changed")
        receipt[name] = dict(fp32_tensor_sha256=identity, norm_fp32=float(vector.norm()))
    require(float(pain.norm()) == pins["s2_norm_fp32"]
            and bool(torch.isclose(fear.norm(), pain.norm(), rtol=1e-6, atol=1e-5)),
            "Original vector norms changed")
    return dict(pain=pain.to(device, dtype=torch.bfloat16),
                fear=fear.to(device, dtype=torch.bfloat16),
                pain_unit_float32=(pain / pain.norm()).to(device, dtype=torch.float32)), receipt


def load_model(root, cache_root, deadline):
    """Original H100/BF16/SDPA/unmerged-PEFT loading, using verified local assets."""
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    import torch
    import platform
    import subprocess
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.monotonic()
    root, cache_root = Path(root), Path(cache_root)
    environment = check_environment(root / "configs/pain_profile_requirements.lock")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "Exactly one CUDA device is required")
    gpu = torch.cuda.get_device_name()
    require("H100" in gpu, "Original source H100 requirement failed")
    require(torch.version.cuda == "13.0", "Unexpected CUDA build")
    hf_cache = cache_root / "huggingface"
    snapshot = Path(snapshot_download(BASE_REPO, revision=BASE_REVISION, cache_dir=hf_cache,
                                     local_files_only=True, token=False))
    base_receipt = verify_base(root, snapshot, deadline=deadline)
    adapter, adapter_files = verify_adapter(root, cache_root / "adapter")
    require(time.time() < deadline, "Model loading deadline reached")
    tok = AutoTokenizer.from_pretrained(str(adapter), local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()
    base = AutoModelForCausalLM.from_pretrained(
        BASE_REPO, revision=BASE_REVISION, cache_dir=hf_cache, local_files_only=True,
        dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="sdpa", device_map="cuda",
    )
    require(base.config._commit_hash == BASE_REVISION, "Loaded base revision mismatch")
    require(all(p.dtype == torch.bfloat16 for p in base.parameters()), "Base is not unquantized BF16")
    model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False, local_files_only=True)
    model.eval()
    require(not any(getattr(m, "merged", False) for m in model.modules()), "Merged adapter forbidden")
    require(all(not p.requires_grad for p in model.parameters()), "Trainable model forbidden")
    layers = base.model.layers
    require(len(layers) == base.config.num_hidden_layers == 64 and base.config.hidden_size == 5120,
            "Unexpected base architecture")
    lora = [m for m in model.modules() if hasattr(m, "lora_A")]
    require(len(lora) == 448, "Unexpected adapter coverage")
    adapter_dtypes = sorted({str(p.dtype) for n, p in model.named_parameters() if "lora_" in n})
    require(adapter_dtypes == ["torch.float32"], "Released unmerged adapter dtype changed")
    vectors, vector_receipt = load_vectors(root, "cuda")
    require(time.time() < deadline, "Model load exceeded deadline")
    identity = dict(
        base=base_receipt, adapter_revision=ADAPTER_REVISION, adapter_archive_sha256=ADAPTER_SHA,
        adapter_files=adapter_files, vector_receipt=vector_receipt, environment=environment,
        gpu=gpu, cuda_build=torch.version.cuda, base_dtype=str(base.dtype),
        attention=base.config._attn_implementation, lora_layers=len(lora),
        adapter_merged=False, adapter_dtypes=adapter_dtypes, load_seconds=time.monotonic() - started,
        allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
        python_version=platform.python_version(), machine=platform.machine(),
        driver_version=subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True, timeout=10,
        ).strip(),
        gpu_total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
        gpu_compute_capability=list(torch.cuda.get_device_capability(0)),
        numerical_flags=dict(
            cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            float32_matmul_precision=torch.get_float32_matmul_precision(),
        ),
        model_inference_performed=False,
    )
    return tok, base, model, layers, vectors, identity
