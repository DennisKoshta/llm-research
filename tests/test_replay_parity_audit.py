"""A scientific parity failure must produce a failed receipt, not only an exception."""

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from curiosity_study.artifacts import RunStore, digest, read, write_once


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_replay_parity.py"
spec = importlib.util.spec_from_file_location("replay_parity_audit", SCRIPT)
auditor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditor)


@pytest.mark.parametrize("different_sham", [False, True])
def test_hidden_sham_difference_produces_failed_bound_receipt(tmp_path, different_sham):
    source = tmp_path / "original"
    run = tmp_path / "parity"
    source.mkdir()
    run.mkdir()
    source_ids = [f"source-{i}" for i in range(80)]
    original = np.ones((80, 2, 3, 4), dtype=np.float32)
    source_path = source / "activations.npz"
    np.savez(
        source_path,
        hidden=original,
        trial_ids=source_ids,
        layers=[15, 31],
        positions=["record", "think_end", "decision"],
    )
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    hidden = np.repeat(original, 2, axis=0)
    if different_sham:
        hidden[1, 0, 0, 0] += 0.25  # Before the layer-31 zero patch; logits remain identical.
    assignments, prefixes, records = [], {}, []
    for i, key in enumerate(source_ids):
        assignment = dict(trial_id=key, kind="new")
        prefix = dict(
            assignment=assignment,
            token_prefix=[10, 20, 30],
            token_prefix_sha256=digest([10, 20, 30]),
            prefix_positions=dict(record=0, think_end=1, final=2),
            reference_label_logits=dict(Yes=1.0, No=0.0),
            reference_label_probabilities_temperature_1=dict(Yes=0.6, No=0.4),
            source_run=source.name,
            source_activation=dict(path=source_path.name, sha256=source_sha, row=i),
        )
        prefixes[key] = prefix
        for arm in ("baseline", "sham"):
            a = dict(trial_id=f"{key}-{arm}", prefix_id=key, arm=arm, layer=31)
            assignments.append(a)
            records.append(
                dict(
                    trial_id=a["trial_id"],
                    terminal=True,
                    assignment=a,
                    source_assignment=assignment,
                    reference_label_logits=prefix["reference_label_logits"],
                    reference_label_probabilities_temperature_1=prefix[
                        "reference_label_probabilities_temperature_1"
                    ],
                    measurement=dict(
                        unpadded_input_ids=prefix["token_prefix"],
                        token_prefix_sha256=prefix["token_prefix_sha256"],
                        unpadded_token_positions=prefix["prefix_positions"],
                        exact_token_replay=True,
                        hooks_remaining=0,
                        label_logits=prefix["reference_label_logits"],
                        label_probabilities_temperature_1=prefix[
                            "reference_label_probabilities_temperature_1"
                        ],
                        intervention=None if arm == "baseline" else dict(delta_norm=0.0),
                    ),
                )
            )
    artifact = run / "activations.npz"
    np.savez(
        artifact,
        hidden=hidden,
        trial_ids=[a["trial_id"] for a in assignments],
        layers=[15, 31],
        positions=["record", "think_end", "final"],
    )
    artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    for row, record in enumerate(records):
        record["activation_artifact"] = dict(
            path=artifact.name,
            sha256=artifact_sha,
            row=row,
            layers=[15, 31],
            positions=["record", "think_end", "final"],
        )
    manifest = dict(
        diagnostics=assignments,
        prefixes=prefixes,
        excluded_sources=[],
        model=dict(model_id="synthetic"),
        source_sha256="code",
        engine_sha256=dict(state_backend="synthetic"),
        parity_criteria=dict(maximum_label_probability_difference=0.03, fraction_within_0_01=0.99),
    )
    RunStore(run, manifest).save_many(records, "synthetic", "cohort")
    result = auditor.audit(run)
    write_once(tmp_path / "audit.json", result)
    assert result["passed"] is not different_sham
    assert result["checks"]["sham_capture_identity"] is not different_sham
    assert all(value for key, value in result["checks"].items() if key != "sham_capture_identity")
    assert result["sham_capture_pairs_identical"] == 80 - int(different_sham)
    assert result["maximum_sham_capture_absolute_difference"] == (0.25 if different_sham else 0)
    assert result["audit_implementation_sha256"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    assert read(tmp_path / "audit.json") == result
