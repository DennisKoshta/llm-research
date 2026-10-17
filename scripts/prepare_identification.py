"""Freeze and archive a prospective identification comparison before launch."""

import argparse
from pathlib import Path
import tarfile
from curiosity_study.artifacts import read, write_once
from curiosity_study.identification import manifest

p = argparse.ArgumentParser()
p.add_argument("--out", type=Path, required=True)
p.add_argument("--items", type=int, default=25)
p.add_argument("--seed", type=int, default=20261007)
p.add_argument("--model", default="Qwen/Qwen3-8B")
args = p.parse_args()
if args.out.exists():
    raise ValueError("Use a new run directory; existing runs are immutable")
m = manifest(
    {"model_id": args.model, "revision": read("configs/model_revisions.json")[args.model]},
    args.items,
    args.seed,
)
write_once(args.out / "manifest.json", m)
with tarfile.open(args.out / "source.tar.gz", "w:gz") as tar:
    for path in [
        "src",
        "tests",
        "infra",
        "scripts",
        "configs",
        "pyproject.toml",
        "uv.lock",
        "reports/identification-v1-plan.md",
    ]:
        tar.add(path, arcname=path, filter=lambda x: None if "__pycache__" in x.name else x)
print(f"Frozen {len(m['diagnostics'])} diagnostic branches; source {m['source_sha256']}")
