"""Create an isolated source copy for a bounded cloud collection."""

import hashlib
import datetime as dt
import os
from pathlib import Path
import shutil
import tarfile
from curiosity_study.artifacts import digest, read, write_once


def freeze(root, run, documents=()):
    root, run = Path(root).resolve(), Path(run).resolve()
    manifest = read(run / "manifest.json")
    target = root / "executions" / run.name
    if target.exists():
        raise ValueError("Execution copy already exists")
    target.mkdir(parents=True)
    for directory in ("src", "infra", "scripts", "tests"):
        for source in sorted((root / directory).rglob("*.py")):
            path = target / source.relative_to(root)
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, path)
    names = ["pyproject.toml", "uv.lock", "configs/model_revisions.json", *documents]
    if (root / ".python-version").exists():
        names.append(".python-version")
    for name in names:
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, destination)
    os.symlink(root / ".venv", target / ".venv", target_is_directory=True)
    source_hash = digest(
        {
            str(p.relative_to(target / "src/curiosity_study")): p.read_text()
            for p in sorted((target / "src/curiosity_study").rglob("*.py"))
        }
    )
    assert source_hash == manifest["source_sha256"]
    archive = run / "source.tar.gz"
    if archive.exists():
        raise ValueError("Source archive already exists")
    with tarfile.open(archive, "w:gz") as tar:
        for file in sorted(target.rglob("*")):
            if file.is_file() and not file.is_symlink():
                tar.add(file, arcname=str(file.relative_to(target)))
    result = dict(
        frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        manifest_sha256=digest(manifest),
        source_sha256=source_hash,
        execution_root=str(target),
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        budget_root=str(root),
        documents={name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in documents},
    )
    write_once(run / "freeze.json", result)
    return result
