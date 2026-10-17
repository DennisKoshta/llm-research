"""Independent saved-evidence rechecks using a real tiny BF16 transformer fixture."""

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

from curiosity_study.artifacts import digest, read, write_once
import test_sustained_audit as real_fixture
from test_sustained_execution import manifest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import check_sustained_arithmetic as check  # noqa: E402


@pytest.fixture(scope="module")
def evidence():
    return real_fixture.evidence.__wrapped__()


def wrapper(evidence):
    harness, arrays = deepcopy(evidence)
    return dict(status="measured", cap=128, harness=harness), arrays


def test_real_bf16_vectors_reconstruct_and_all_active_predictions_match(evidence):
    report, arrays = wrapper(evidence)
    result = check.check_evidence(manifest(), report, arrays)
    assert result["passed"]
    assert result["arithmetic"]["vector_pairs"] == 4
    assert result["arithmetic"]["decode_events"] == 3
    for label, rows in (("sham", 32), ("unaffected31", 31), ("cleanup_repeat", 32)):
        assert result["identities"][label]["identical_completions"] == rows
        assert result["identities"][label]["identical_full_fp32_vocabulary_fingerprints"] == rows * 4
    assert result["observed_maximum_tokens"] == 4
    assert all(r["ended_with_eos"] == 32 for r in result["timelines"].values())


@pytest.mark.parametrize("failure", ["logit", "token", "mask", "cache", "active", "scalar", "vector", "site"])
def test_rehashed_or_forged_pass_flags_cannot_hide_numeric_or_timeline_changes(evidence, failure):
    report, arrays = wrapper(evidence)
    conditions = report["harness"]["conditions"]
    changed = conditions["one_row_nonzero"]
    if failure == "logit":
        conditions["zero_sham"]["forwards"][2]["full_fp32_logits_sha256"][17] = "a" * 64
    elif failure == "token":
        conditions["clean_repeat"]["rows"][3]["completion"]["generated_token_ids"][0] += 1
    elif failure == "mask":
        conditions["zero_sham"]["forwards"][1]["attention_mask_row_sha256"][5] = "b" * 64
    elif failure == "cache":
        changed["forwards"][2]["cache_length_before"] += 1
    elif failure == "active":
        changed["forwards"][2]["active_prediction_rows"][4] = False
    elif failure == "scalar":
        changed["rows"][0]["intervention_trace"]["events"][1]["arithmetic_scalar_float32"] += 0.5
    elif failure == "vector":
        arrays["after"][2, 7] += 1
    else:
        changed["vector_sites"][1]["forward_index"] += 1
    assert report["harness"]["passed"] is True
    assert not check.check_evidence(manifest(), report, arrays)["passed"]


def test_no_decode_cannot_certify_sustained_geometry(evidence):
    report, arrays = wrapper(evidence)
    c = report["harness"]["conditions"]["one_row_nonzero"]
    c["vector_sites"] = c["vector_sites"][:1]
    c["vector_activation_dtypes"] = c["vector_activation_dtypes"][:1]
    c["rows"][0]["intervention_trace"]["events"] = c["rows"][0]["intervention_trace"]["events"][:1]
    arrays["before"], arrays["after"] = arrays["before"][:1], arrays["after"][:1]
    assert not check.geometry(c, arrays, arrays["direction_float32"], 0.25)["passed"]


def test_completed_early_row_has_null_later_fingerprint_and_is_not_dropped(evidence):
    report, _ = wrapper(evidence)
    c = report["harness"]["conditions"]["baseline"]
    for name in ("baseline", "zero_sham"):
        changed = report["harness"]["conditions"][name]
        row = changed["rows"][2]
        row["completion"]["generated_token_ids"] = [127]
        for step, frame in enumerate(changed["forwards"][1:], 1):
            frame["active_prediction_rows"][2] = False
            frame["full_fp32_logits_sha256"][2] = None
            frame["input_ids"][2] = [127 if step == 1 else report["harness"]["pad_token_id"]]
    result = check.timeline(c, manifest()["requests"], 128, 128, {127}, report["harness"]["pad_token_id"])
    assert result["passed"] and result["row_lengths"][2] == 1
    assert result["mixed_active_inactive_decode_observed"]
    compared = check.compare(c, report["harness"]["conditions"]["zero_sham"], list(range(32)))
    assert compared["passed"] and compared["compared_active_predictions"] == 125
    c["forwards"][2]["full_fp32_logits_sha256"][2] = "0" * 64
    assert not check.timeline(c, manifest()["requests"], 128, 128, {127}, 0)["passed"]


def saved_evidence(root, evidence):
    report, arrays = wrapper(evidence)
    m = manifest()
    write_once(root / "manifest.json", m)
    np.savez_compressed(root / "vectors.npz", **arrays)
    vector = dict(path="vectors.npz", sha256=check.sha(root / "vectors.npz"))
    report.update(manifest_sha256=digest(m), vector_artifact=vector, performance=dict(synthetic=True))
    write_once(root / "report.json", report)
    audit = dict(
        passed=True,
        complete=True,
        integrity_passed=True,
        manifest_sha256=digest(m),
        report_artifact=dict(path="report.json", sha256=check.sha(root / "report.json")),
        vector_artifact=vector,
    )
    write_once(root / "audit.json", audit)
    return root, root / "audit.json"


def test_checker_binds_actual_audit_report_and_vector_bytes(tmp_path, evidence):
    run, audit_path = saved_evidence(tmp_path / "run", evidence)
    result = check.check_run(run, audit_path)
    assert result["passed"] and result["cap"] == 128
    assert result["audit_byte_sha256"] == check.sha(audit_path)
    assert result["report_byte_sha256"] == check.sha(run / "report.json")
    (run / "vectors.npz").write_bytes((run / "vectors.npz").read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="Vector bytes"):
        check.check_run(run, audit_path)


def test_final_check_requires_both_stages_and_writes_immutable_receipt(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(check, "check_run", lambda r, a: calls.append((r, a)) or dict(cap=r, passed=True))
    out = tmp_path / "check.json"
    receipt = check.main([128, 4096], ["short", "full"], out)
    assert receipt == read(out) and receipt["passed"] and not receipt["scientific_admission"]
    assert calls == [(128, "short"), (4096, "full")]
    with pytest.raises(ValueError, match="immutable"):
        check.main([128, 4096], ["short", "full"], out)
    with pytest.raises(ValueError, match="ordered"):
        check.main([128, 128], ["short", "short"], tmp_path / "other.json")
