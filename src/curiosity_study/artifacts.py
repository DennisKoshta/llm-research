"""Durable immutable JSON records, with hashes and replay reconciliation."""

import hashlib
import errno
import json
import os
from pathlib import Path
import uuid


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (canonical(value) + "\n").encode()
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)  # Atomic publish, never overwrites another writer.
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ValueError(f"Refusing to overwrite immutable artifact: {path}") from None
        except OSError as error:
            if error.errno not in (errno.EPERM, errno.EOPNOTSUPP):
                raise
            # Modal Volumes do not support hard links. Each artifact has exactly one
            # writer (unique attempt namespace; one worker for shared manifests).
            # Rename publishes a complete file atomically under that invariant.
            if path.exists():
                if path.read_bytes() != payload:
                    raise ValueError(f"Refusing to overwrite immutable artifact: {path}") from None
            else:
                os.rename(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def read(path):
    return json.loads(Path(path).read_text())


class RunStore:
    def __init__(self, root, manifest):
        self.root = Path(root)
        self.manifest = manifest
        self.manifest_hash = digest(manifest)
        write_once(self.root / "manifest.json", manifest)

    def completed(self):
        records = {}

        def consume(record):
            if not record.get("terminal", False):
                return
            trial_id = record["trial_id"]
            if trial_id in records and records[trial_id] != record:
                raise ValueError(f"Conflicting terminal attempts for {trial_id}")
            records[trial_id] = record

        for path in sorted((self.root / "attempts").glob("*/*.json")):
            if path.name.startswith("."):
                continue
            envelope = read(path)
            if (
                envelope["manifest_hash"] != self.manifest_hash
                or digest(envelope["record"]) != envelope["sha256"]
            ):
                raise ValueError(f"Artifact integrity failure: {path}")
            consume(envelope["record"])
        for path in sorted((self.root / "cohorts").glob("*/*.json")):
            if path.name.startswith("."):
                continue
            envelope = read(path)
            if (
                envelope["manifest_hash"] != self.manifest_hash
                or digest(envelope["records"]) != envelope["sha256"]
            ):
                raise ValueError(f"Artifact integrity failure: {path}")
            for record in envelope["records"]:
                consume(record)
        return records

    def save(self, record, attempt):
        write_once(
            self.root / "attempts" / attempt / f"{record['trial_id']}.json",
            {"manifest_hash": self.manifest_hash, "sha256": digest(record), "record": record},
        )

    def save_many(self, records, attempt, cohort_id):
        """Publish an entire fixed cohort atomically, retaining every assigned record."""
        if not records or len({r["trial_id"] for r in records}) != len(records):
            raise ValueError("Cohort records must have distinct trial IDs")
        write_once(
            self.root / "cohorts" / attempt / f"{cohort_id}.json",
            {"manifest_hash": self.manifest_hash, "sha256": digest(records), "records": records},
        )
