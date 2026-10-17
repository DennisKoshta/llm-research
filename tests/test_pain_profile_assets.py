"""Local synthetic asset/cache failures; no downloads or model construction."""

import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tarfile
from types import SimpleNamespace

import pytest

from curiosity_study import pain_profile_assets as assets

ROOT = Path(__file__).resolve().parents[1]


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def adapter_tree(root, extracted):
    files = {
        "adapter/adapter_config.json": json.dumps({"base_model_name_or_path": assets.BASE_REPO}).encode(),
        "adapter/adapter_model.safetensors": b"synthetic tensor payload",
        "adapter/tokenizer.json": b"{}",
    }
    members = []
    for name, data in files.items():
        path = extracted / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        members.append(
            dict(name="./" + name, type="file", byte_count=len(data), sha256=hashlib.sha256(data).hexdigest())
        )
    put(root / assets.ASSETS / "asset-receipt.json", dict(archive_sha256=assets.ADAPTER_SHA, members=members))
    return files


def base_tree(root, snapshot):
    snapshot.mkdir(parents=True)
    pins = []
    for i in (1, 2):
        path = snapshot / f"model-{i:05d}-of-00002.safetensors"
        path.write_bytes(f"synthetic weights {i}".encode())
        pins.append(dict(path=path.name, size=path.stat().st_size, sha256=assets.sha(path)))
    put(
        root / assets.ASSETS / "base-weight-pins.json",
        dict(repo=assets.BASE_REPO, revision=assets.BASE_REVISION, files=pins),
    )
    metadata = []
    for name in ("config.json", "model.safetensors.index.json"):
        put(snapshot / name, {"synthetic": name})
        metadata.append(dict(repository=assets.BASE_REPO, path=name, sha256=assets.sha(snapshot / name)))
    put(root / assets.REFERENCE / "model-files.json", {"files": metadata})
    generation = {"eos_token_id": [151645, 151643]}
    put(snapshot / "generation_config.json", generation)
    put(root / assets.ASSETS / "base-generation-config.json", generation)


def test_complete_lock_checks_all57_installed_distributions(monkeypatch):
    lock = ROOT / "configs/pain_profile_requirements.lock"
    pins = dict(re.findall(r"^([A-Za-z0-9_.-]+)==([^\s\\]+)", lock.read_text(), re.M))
    seen = []
    monkeypatch.setattr(assets, "version", lambda name: seen.append(name) or pins[name])
    result = assets.check_environment(lock)
    assert len(seen) == len(set(seen)) == 57
    assert result["distributions"] == pins and result["lock_byte_sha256"] == assets.sha(lock)
    monkeypatch.setattr(assets, "version", lambda name: "unexpected" if name == "torch" else pins[name])
    with pytest.raises(ValueError, match="environment differs"):
        assets.check_environment(lock)


def test_incomplete_environment_lock_fails_before_package_lookup(tmp_path, monkeypatch):
    lock = tmp_path / "lock"
    lock.write_text("torch==2.11.0+cu130\n")
    monkeypatch.setattr(assets, "version", lambda name: pytest.fail("must reject lock first"))
    with pytest.raises(ValueError, match="package lock"):
        assets.check_environment(lock)


def test_adapter_members_use_exact_bytes_and_no_unlisted_files(tmp_path):
    extracted = tmp_path / "cache/adapter"
    files = adapter_tree(tmp_path, extracted)
    directory, hashes = assets.verify_adapter(tmp_path, extracted)
    assert directory == extracted / "adapter" and set(hashes) == set(files)
    (directory / "extra.json").write_text("{}")
    with pytest.raises(ValueError, match="Unexpected adapter files"):
        assets.verify_adapter(tmp_path, extracted)


@pytest.mark.parametrize(
    "change", ["same_size_edit", "missing", "symlink", "traversal", "archive_pin", "base"]
)
def test_adapter_cache_tampering_and_wrong_identity_fail(tmp_path, change):
    extracted = tmp_path / "cache/adapter"
    adapter_tree(tmp_path, extracted)
    path = extracted / "adapter/adapter_model.safetensors"
    receipt_path = tmp_path / assets.ASSETS / "asset-receipt.json"
    receipt = assets.read(receipt_path)
    if change == "same_size_edit":
        path.write_bytes(b"x" * path.stat().st_size)
    elif change == "missing":
        path.unlink()
    elif change == "symlink":
        target = tmp_path / "external"
        path.rename(target)
        path.symlink_to(target)
    elif change == "traversal":
        receipt["members"][0]["name"] = "../external"
        put(receipt_path, receipt)
    elif change == "archive_pin":
        receipt["archive_sha256"] = "0" * 64
        put(receipt_path, receipt)
    else:
        config = extracted / "adapter/adapter_config.json"
        put(config, {"base_model_name_or_path": "other/model"})
        receipt["members"][0].update(byte_count=config.stat().st_size, sha256=assets.sha(config))
        put(receipt_path, receipt)
    with pytest.raises(ValueError):
        assets.verify_adapter(tmp_path, extracted)


def test_base_verification_hashes_weights_metadata_and_generation(tmp_path):
    snapshot = tmp_path / "cache/base"
    base_tree(tmp_path, snapshot)
    receipt = assets.verify_base(tmp_path, snapshot)
    assert receipt["revision"] == assets.BASE_REVISION and len(receipt["files"]) == 2
    # Hugging Face snapshot files are legitimately symlinks to hashed blobs.
    shard = snapshot / receipt["files"][0]["path"]
    blob = tmp_path / "blob"
    shard.rename(blob)
    shard.symlink_to(blob)
    assert assets.verify_base(tmp_path, snapshot) == receipt


@pytest.mark.parametrize("change", ["weight", "metadata", "generation", "revision", "traversal", "expired"])
def test_bad_base_cache_cannot_be_used(tmp_path, monkeypatch, change):
    snapshot = tmp_path / "cache/base"
    base_tree(tmp_path, snapshot)
    if change in ("weight", "metadata", "generation"):
        name = {
            "weight": "model-00001-of-00002.safetensors",
            "metadata": "config.json",
            "generation": "generation_config.json",
        }[change]
        (snapshot / name).write_bytes(b"changed")
    elif change in ("revision", "traversal"):
        path = tmp_path / assets.ASSETS / "base-weight-pins.json"
        pins = assets.read(path)
        if change == "revision":
            pins["revision"] = "wrong"
        else:
            pins["files"][0]["path"] = "../outside"
        put(path, pins)
    monkeypatch.setattr(assets.time, "time", lambda: 10)
    with pytest.raises(ValueError):
        assets.verify_base(tmp_path, snapshot, deadline=10 if change == "expired" else 20)


def test_prefetch_uses_fixed_revisions_and_safe_extraction_with_no_model_import(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    snapshot = cache / "base"
    base_tree(tmp_path, snapshot)
    staging = tmp_path / "staging"
    files = adapter_tree(tmp_path, staging)
    archive = tmp_path / "adapter.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for name, data in files.items():
            item = tarfile.TarInfo(name)
            item.size = len(data)
            handle.addfile(item, io.BytesIO(data))
    monkeypatch.setattr(assets, "ADAPTER_SHA", assets.sha(archive))
    receipt_path = tmp_path / assets.ASSETS / "asset-receipt.json"
    receipt = assets.read(receipt_path)
    receipt["archive_sha256"] = assets.ADAPTER_SHA
    put(receipt_path, receipt)
    calls = []

    def snapshot_download(repo, **kwargs):
        calls.append((repo, kwargs))
        return str(snapshot)

    def hf_download(repo, filename, **kwargs):
        calls.append((repo, filename, kwargs))
        return str(archive)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download, hf_hub_download=hf_download),
    )
    monkeypatch.setattr(assets.time, "time", lambda: 10)
    result = assets.prefetch(tmp_path, cache, 20)
    assert result["model_inference_performed"] is False
    assert calls[0][0] == assets.BASE_REPO and calls[0][1]["revision"] == assets.BASE_REVISION
    assert calls[1][0:2] == (assets.ADAPTER_REPO, assets.ADAPTER_NAME)
    assert calls[1][2]["revision"] == assets.ADAPTER_REVISION
    assert all(call[-1]["token"] is False for call in calls)
    # A durable partial or corrupted extracted cache is rejected, never repaired silently.
    (Path(result["adapter_path"]) / "tokenizer.json").write_text("bad")
    with pytest.raises(ValueError, match="Adapter member identity"):
        assets.prefetch(tmp_path, cache, 20)


def test_expired_prefetch_does_not_start_even_a_mock_download(tmp_path, monkeypatch):
    def forbidden(*a, **k):
        pytest.fail("download after deadline")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=forbidden, hf_hub_download=forbidden),
    )
    monkeypatch.setattr(assets.time, "time", lambda: 20)
    with pytest.raises(ValueError, match="Expired"):
        assets.prefetch(tmp_path, tmp_path / "cache", 20)


@pytest.mark.parametrize("bad", ["hash", "nonfinite", "wrong_dtype", "wrong_width", "norm"])
def test_cpu_vector_validation_rejects_malformed_saved_tensors(tmp_path, bad):
    import torch
    from safetensors.torch import save_file

    vectors = tmp_path / assets.VECTORS
    vectors.mkdir(parents=True)
    pain = torch.arange(5120, dtype=torch.float32) / 5120
    fear = pain.flip(0).contiguous()
    if bad == "nonfinite":
        fear[0] = float("nan")
    elif bad == "wrong_dtype":
        fear = fear.to(torch.float64)
    elif bad == "wrong_width":
        fear = fear[:4]
    elif bad == "norm":
        fear *= 2
    pain_path = vectors / "Qwen_2.5_32B_instruct_pain_vectors.safetensors"
    fear_path = vectors / "fear_Qwen_2.5_32B_instruct.pt"
    save_file({"s2_pain_vector": pain}, pain_path)
    torch.save({"fear_vector_matched_fp32": fear}, fear_path)
    pins = dict(
        layer=61,
        width=5120,
        files={pain_path.name: assets.sha(pain_path), fear_path.name: assets.sha(fear_path)},
        fp32_tensor_sha256={
            "s2_pain_vector": hashlib.sha256(pain.numpy().tobytes()).hexdigest(),
            "fear_vector_matched_fp32": hashlib.sha256(fear.numpy().tobytes()).hexdigest(),
        },
        s2_norm_fp32=float(pain.norm()),
    )
    if bad == "hash":
        pins["fp32_tensor_sha256"]["fear_vector_matched_fp32"] = "0" * 64
    put(tmp_path / assets.PROFILE / "inputs/portable_vector_pins.json", pins)
    with pytest.raises(ValueError):
        assets.load_vectors(tmp_path, "cpu")


def test_actual_small_pinned_vectors_cpu_only():
    import torch

    vectors, receipt = assets.load_vectors(ROOT, "cpu")
    assert vectors["pain"].dtype == vectors["fear"].dtype == torch.bfloat16
    assert vectors["pain_unit_float32"].dtype == torch.float32
    assert all(value.device.type == "cpu" and value.shape == (5120,) for value in vectors.values())
    assert receipt["pain"]["norm_fp32"] == pytest.approx(143.83343505859375)


@pytest.mark.parametrize("available,count,name", [(False, 0, ""), (True, 2, "H100"), (True, 1, "H200")])
def test_model_loader_rejects_wrong_hardware_before_any_weight_loader(
    tmp_path, monkeypatch, available, count, name
):
    import torch
    from transformers import AutoModelForCausalLM

    monkeypatch.setattr(assets, "check_environment", lambda path: {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: name)
    monkeypatch.setattr(
        AutoModelForCausalLM, "from_pretrained", lambda *a, **k: pytest.fail("weight loader called")
    )
    with pytest.raises(ValueError, match="CUDA device|H100"):
        assets.load_model(tmp_path, tmp_path / "cache", 100)
