"""Fetch one pinned archive and inspect data without loading tensors or code."""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import struct
import tarfile
from datetime import datetime, timezone
import urllib.request


ROOT = Path(__file__).resolve().parent
URL = (
    "https://huggingface.co/Valen92/pain-adapters/resolve/"
    "b64bd64b4bc7ca6e0733a489b8372a099d55ef05/"
    "adapter_Qwen_2.5_32B_instruct.tar.gz"
)
EXPECTED = "cd96d4d43a3f7d6a8c67804ed4f4ee567ee4942e168804c757e69937b857127f"
LIMIT = 2_000_000_000
ARCHIVE = ROOT / "adapter_Qwen_2.5_32B_instruct.tar.gz"


def require(value, message):
    if not value:
        raise ValueError(message)


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    require(not (ROOT / "asset-receipt.json").exists(), "immutable receipt already exists")
    with urllib.request.urlopen(urllib.request.Request(URL, method="HEAD"), timeout=45) as response:
        advertised = int(response.headers["Content-Length"])
        require(0 < advertised <= LIMIT, "advertised archive exceeds download limit")
        head = {"status": response.status, "content_length": advertised,
                "etag": response.headers.get("ETag"),
                "content_type": response.headers.get("Content-Type")}
    started = datetime.now(timezone.utc).isoformat()
    if not ARCHIVE.exists():
        temporary = ARCHIVE.with_suffix(ARCHIVE.suffix + ".part")
        require(not temporary.exists(), "partial download exists; no automatic retry")
        count = 0
        digest = hashlib.sha256()
        with urllib.request.urlopen(URL, timeout=60) as response, temporary.open("xb") as handle:
            while chunk := response.read(1024 * 1024):
                count += len(chunk)
                require(count <= LIMIT, "stream exceeds download limit")
                digest.update(chunk)
                handle.write(chunk)
        require(count == advertised, "advertised/downloaded size mismatch")
        require(digest.hexdigest() == EXPECTED, "archive hash mismatch")
        os.replace(temporary, ARCHIVE)
    require(ARCHIVE.stat().st_size == advertised and file_sha(ARCHIVE) == EXPECTED,
            "existing archive does not match pinned bytes")
    members = []
    names = set()
    total = 0
    tensor_summary = None
    # Streaming tar access: never extractall, import, unpickle, or instantiate a tokenizer.
    with tarfile.open(ARCHIVE, mode="r|gz") as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            require(not name.is_absolute() and ".." not in name.parts and "\\" not in member.name,
                    "unsafe member path")
            require(str(name) not in names, "duplicate member name")
            names.add(str(name))
            require(len(names) <= 100, "too many archive members")
            require(member.isdir() or member.isfile(), "non-regular archive member")
            if member.isdir():
                members.append({"name": member.name, "type": "directory"})
                continue
            total += member.size
            require(total <= LIMIT, "uncompressed archive exceeds inspection limit")
            stream = archive.extractfile(member)
            require(stream is not None, "missing member payload")
            digest = hashlib.sha256()
            retained = bytearray()
            tensor = name.name == "adapter_model.safetensors"
            require(tensor or name.suffix in (".json", ".txt", ".md", ".jinja"),
                    "unexpected archive payload type")
            if tensor:
                prefix = stream.read(8)
                require(len(prefix) == 8, "short safetensors header")
                header_size = struct.unpack("<Q", prefix)[0]
                require(header_size <= 16 * 1024 * 1024, "oversized safetensors header")
                header_bytes = stream.read(header_size)
                require(len(header_bytes) == header_size, "short safetensors metadata")
                header = json.loads(header_bytes)
                tensors = {key: value for key, value in header.items() if key != "__metadata__"}
                require(all(isinstance(value, dict) and set(value) == {"dtype", "shape", "data_offsets"}
                            for value in tensors.values()), "unexpected tensor metadata schema")
                tensor_summary = {"tensor_count": len(tensors),
                                  "dtype_counts": {}, "metadata": header.get("__metadata__"),
                                  "header_sha256": hashlib.sha256(header_bytes).hexdigest()}
                for value in tensors.values():
                    dtype = value["dtype"]
                    tensor_summary["dtype_counts"][dtype] = tensor_summary["dtype_counts"].get(dtype, 0) + 1
                digest.update(prefix)
                digest.update(header_bytes)
            else:
                require(member.size <= 64 * 1024 * 1024, "metadata member exceeds size limit")
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                if not tensor:
                    retained.extend(chunk)
            record = {"name": member.name, "type": "file", "byte_count": member.size,
                      "sha256": digest.hexdigest(), "tensor_data_loaded": False}
            if not tensor:
                destination = ROOT / "metadata" / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                require(not destination.exists(), "metadata output already exists")
                destination.write_bytes(retained)
                record["retained_path"] = str(destination.relative_to(ROOT))
            members.append(record)
    receipt = {"schema_version": 1, "purpose": "pinned adapter asset inspection only",
               "url": URL, "repository": "Valen92/pain-adapters",
               "revision": "b64bd64b4bc7ca6e0733a489b8372a099d55ef05",
               "started_utc": started, "completed_utc": datetime.now(timezone.utc).isoformat(),
               "head": head, "download_limit_bytes": LIMIT,
               "archive_path": ARCHIVE.name, "archive_sha256": EXPECTED,
               "archive_byte_count": ARCHIVE.stat().st_size,
               "all_members_regular_files_or_directories": True,
               "uncompressed_file_bytes": total, "members": members,
               "safetensors_header_only": tensor_summary,
               "inspection_script_sha256": file_sha(Path(__file__)),
               "downloaded_code_executed": False, "pickle_loaded": False,
               "tensor_values_loaded": False, "base_weights_downloaded": False,
               "model_inference_performed": False, "gpu_used": False,
               "scientific_admission": False}
    (ROOT / "asset-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"archive_sha256": EXPECTED, "member_count": len(members),
                      "receipt": str(ROOT / "asset-receipt.json")}))


if __name__ == "__main__":
    main()
