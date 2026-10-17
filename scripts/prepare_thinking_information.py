import argparse
import hashlib
from pathlib import Path
import tarfile
from curiosity_study.artifacts import read, write_once
from curiosity_study.thinking_information import build_manifest
from curiosity_study.information_state import prompt

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--items", type=int, default=1)
    p.add_argument("--seed", type=int, default=2026101407)
    p.add_argument("--phase", default="development_smoke")
    a = p.parse_args()
    revision = read(ROOT / "configs/model_revisions.json")["Qwen/Qwen3-32B"]
    m = build_manifest(dict(model_id="Qwen/Qwen3-32B", revision=revision), a.items, a.seed, a.phase)
    write_once(a.out / "manifest.json", m)
    write_once(
        a.out / "prompt-examples.json",
        [
            dict(assignment=row, prompt=prompt(row)[0], expected=prompt(row)[2])
            for row in m["diagnostics"][:8]
        ],
    )
    archive = a.out / "source.tar.gz"
    if archive.exists():
        raise ValueError("Source archive already exists")
    with tarfile.open(archive, "w:gz") as tar:
        for directory in ("src", "infra", "scripts", "tests"):
            for path in sorted((ROOT / directory).rglob("*.py")):
                tar.add(path, arcname=str(path.relative_to(ROOT)))
        for path in (
            "pyproject.toml",
            "uv.lock",
            "configs/model_revisions.json",
            "reports/reasoning-information-v1-plan.md",
            "reports/equal-uncertainty-v1-plan.md",
        ):
            tar.add(ROOT / path, arcname=path)
    write_once(
        a.out / "freeze.json",
        dict(
            source_sha256=m["source_sha256"],
            archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            plan_sha256=hashlib.sha256(
                (ROOT / "reports/reasoning-information-v1-plan.md").read_bytes()
            ).hexdigest(),
            branches=len(m["diagnostics"]),
        ),
    )
    print(a.out, m["source_sha256"], len(m["diagnostics"]))


if __name__ == "__main__":
    main()
