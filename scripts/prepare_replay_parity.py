"""Freeze exact-prefix parity using the prospectively selected first two world indices."""

from pathlib import Path
import argparse
import sys
from curiosity_study.artifacts import read, write_once, digest, RunStore
from curiosity_study.replay_information import make_prefix, engine_hashes
from curiosity_study.state_backend import PREFILL_PADDING_MULTIPLE
from curiosity_study.runner import source_hash

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "infra"))
from freeze_execution import freeze  # noqa: E402


def main(name="qwen32-thinking-replay-v1-parity"):
    if Path(name).name != name:
        raise ValueError("Parity run name must be one directory component")
    parent = ROOT / "runs/qwen32-thinking-state-v1-validation"
    m = read(parent / "manifest.json")
    audit = read(ROOT / "reports/qwen32-thinking-state-v1-validation-audit.json")
    assert audit["complete"] and audit["manifest_sha256"] == digest(m)
    prefixes, excluded = {}, []
    for child in read(parent / "children.json"):
        path = Path(child["path"])
        cm = read(path / "manifest.json")
        assert digest(cm) == child["manifest_sha256"]
        records = RunStore(path, cm).completed()
        for a in cm["diagnostics"]:
            if a["index"] >= 2:
                continue
            r = records[a["trial_id"]]
            p = make_prefix(r, path.name, digest(cm))
            if p is None:
                excluded.append(a["trial_id"])
            else:
                prefixes[a["trial_id"]] = p
    assert len(prefixes) + len(excluded) == 80
    assignments = []
    for key in sorted(prefixes):
        for arm in ("baseline", "sham"):
            assignments.append(
                dict(
                    trial_id=f"replay-parity-{key}-{arm}",
                    prefix_id=key,
                    arm=arm,
                    kind=prefixes[key]["assignment"]["kind"],
                    layer=31,
                    position="think_end",
                    dose=0.0,
                )
            )
    manifest = dict(
        schema_version=3,
        assay="replay_information",
        phase="engineering_parity",
        model=m["model"],
        execution_gpu="H200",
        execution=dict(backend="torch_batched_v1", batch_size=16),
        source_sha256=source_hash(),
        engine_sha256=engine_hashes(),
        prefill_padding_multiple=PREFILL_PADDING_MULTIPLE,
        assignments=[],
        diagnostics=assignments,
        assignment_sha256=digest(assignments),
        prefixes=prefixes,
        prefixes_sha256=digest(prefixes),
        excluded_sources=excluded,
        source_parent_manifest_sha256=digest(m),
        capture_layers=[15, 31, 47, 57],
        max_new_tokens=0,
        greedy=True,
        parity_criteria=dict(
            assigned_sources=80,
            complete_eligible=True,
            allowed_label_argmax_match_fraction=1.0,
            maximum_label_probability_difference=0.03,
            fraction_within_0_01=0.99,
        ),
        invalid_output_policy="terminal_no_retry",
    )
    run = ROOT / "runs" / name
    write_once(run / "manifest.json", manifest)
    freeze(
        ROOT,
        run,
        [
            "reports/reasoning-information-v1-analysis-plan.md",
            "reports/reasoning-information-review-amendment.md",
            "reports/reasoning-information-development-scope-amendment.md",
            "reports/replay-padding-engineering-amendment.md",
            "reports/replay-reduction-engineering-amendment.md",
        ],
    )
    print(
        dict(
            prefixes=len(prefixes),
            excluded=excluded,
            assignments=len(assignments),
            source=manifest["source_sha256"],
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="qwen32-thinking-replay-v1-parity")
    main(parser.parse_args().name)
