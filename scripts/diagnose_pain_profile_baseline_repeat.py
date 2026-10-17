"""Saved-artifact-only diagnosis of the two failed-setup baseline repeat calls.

Reads no treatment call, performs no model inference, and changes no audit gate.
"""

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/qwen25-pain-profile-control-v1"
OUT = ROOT / "reports/pain-profile-baseline-repeat-diagnosis.json"
MANIFEST_SHA = "e87c70267be0b7813750dbef9d752f0855cf5b2db90e81f2381127d544b59a63"
SOURCE_SHA = "63bc3e330b25f5805724bfa5b8bf2f75d6fe37d048963524e1d28138e4bd4ff9"
ARCHIVE_SHA = "9c59998a8a162075a4adc201c4a374f821f76978b0c4ec518dce4630cd3f1fa7"
ENGINE = "src/curiosity_study/pain_profile_engine.py"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def binding(path, value=None):
    result = dict(path=str(Path(path).relative_to(ROOT)), byte_sha256=sha(path))
    if value is not None:
        result["content_sha256"] = digest(value)
    return result


def array_binding(value):
    return dict(dtype=str(value.dtype), shape=list(value.shape),
                byte_sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())


def diagnose():
    manifest, frozen = read(RUN / "manifest.json"), read(RUN / "freeze.json")
    require(digest(manifest) == frozen["manifest_sha256"] == MANIFEST_SHA
            and manifest["source_sha256"] == frozen["source_sha256"] == SOURCE_SHA
            and sha(RUN / "source.tar.gz") == frozen["archive_sha256"] == ARCHIVE_SHA,
            "Original campaign binding changed")
    with tarfile.open(RUN / "source.tar.gz") as archive:
        code = archive.extractfile(ENGINE).read()
    engine_sha = hashlib.sha256(code).hexdigest()
    require(engine_sha == sha(Path(frozen["execution_root"]) / ENGINE), "Frozen engine changed")
    phase = RUN / "phases/setup"
    summary, identity = read(phase / "summary.json"), read(phase / "identity.json")
    require(summary["manifest_sha256"] == MANIFEST_SHA and summary["source_sha256"] == SOURCE_SHA
            and summary["identity_sha256"] == digest(identity)
            and summary["complete"] is summary["passed"] is summary["integrity_passed"] is False,
            "This receipt must preserve the original failed setup")
    envelopes, arrays, inputs = [], [], {}
    for index in (0, 10):
        path = phase / "calls" / f"{index:04d}.json"
        item = read(path)
        require(item["manifest_sha256"] == MANIFEST_SHA and item["attempt"] == summary["attempt"]
                and item["identity_sha256"] == digest(identity)
                and item["call_spec"]["arm"] == "baseline" and item["call_spec"]["source_cohort"] == 0
                and item["call_spec_sha256"] == digest(item["call_spec"]), "Not the bound baseline pair")
        row = summary["call_bindings"][index]
        require(row["call_id"] == item["call_spec"]["call_id"] and row["content_sha256"] == digest(item),
                "Summary does not bind the observed baseline call")
        array_path = path.with_suffix(".npz")
        artifact = item["arrays_artifact"]
        require(artifact["path"] == array_path.name and sha(array_path) == artifact["byte_sha256"],
                "Array bytes changed")
        with np.load(array_path, allow_pickle=False) as stored:
            require(len(stored.files) == len(set(stored.files)), "Duplicate array key")
            values = {key: stored[key] for key in stored.files}
        bindings = {key: array_binding(value) for key, value in values.items()}
        require(bindings == artifact["arrays"] == item["engine_result"]["evidence"]["arrays"],
                "Array layout/content changed")
        inputs[f"call_{index:04d}"] = binding(path, item)
        inputs[f"arrays_{index:04d}"] = binding(array_path)
        envelopes.append(item)
        arrays.append(values)
    left, right = envelopes
    require(right["call_spec"]["reference_call_id"] == left["call_spec"]["call_id"]
            and left["call_spec"]["role"] == "primary" and right["call_spec"]["role"] == "repeat"
            and right["checks"]["same_source_repeat_identity"] is False,
            "Not the actual failed repeat gate")
    results = [item["engine_result"] for item in envelopes]
    reports = [result["evidence"] for result in results]
    records = [result["records"] for result in results]
    require(all(len(rows) == 32 for rows in records)
            and all(report["coefficient"] == 0 and report["applied_coefficients"] == [0] * 32
                    and report["complete"] is True and report["integrity_passed"] is True for report in reports),
            "Baseline call arithmetic/coverage changed")
    requests = {row["base_request_id"]: row for row in manifest["preflight"]["requests"]}
    for item, report in zip(envelopes, reports, strict=True):
        source = [dict(requests[key], trial_id=trial) for key, trial in zip(
            item["call_spec"]["base_request_ids"], item["call_spec"]["trial_ids"], strict=True)]
        require(digest(source) == item["items_sha256"] == report["items_sha256"], "Input/seed binding differs")
    steps = []
    for index, (a, b) in enumerate(zip(reports[0]["forwards"], reports[1]["forwards"], strict=True)):
        active = a["sampling"]["alive_before"]
        require(active == b["sampling"]["alive_before"] and len(active) == 32, "Active-row inventory changed")
        metrics = {}
        for label, x, y in (
            ("full_vocabulary_logits", a["full_fp32_logits_sha256"], b["full_fp32_logits_sha256"]),
            ("before_hook_residual", [r["before_sha256"] for r in a["geometry"]],
             [r["before_sha256"] for r in b["geometry"]]),
            ("after_hook_residual", [r["after_sha256"] for r in a["geometry"]],
             [r["after_sha256"] for r in b["geometry"]]),
        ):
            rows = [j for j, (u, v) in enumerate(zip(x, y, strict=True)) if u != v]
            metrics[label] = dict(mismatched_all_rows=len(rows), mismatched_active_rows=sum(active[j] for j in rows),
                                  mismatched_row_indices=rows)
        samples = []
        for row in (0, 31):
            for label in ("before", "after"):
                key = f"f{index:02d}_r{row:02d}_{label}"
                x, y = [value[key] for value in arrays]
                require(x.dtype == y.dtype == np.float32 and x.shape == y.shape
                        and np.isfinite(x).all() and np.isfinite(y).all(), "Invalid saved vectors")
                delta = y.astype(np.float64) - x.astype(np.float64)
                samples.append(dict(row=row, observation=label, shape=list(x.shape), active=active[row],
                    changed_components=int(np.count_nonzero(delta)), total_components=int(delta.size),
                    max_absolute_difference=float(np.abs(delta).max()), difference_l2=float(np.linalg.norm(delta)),
                    original_l2=float(np.linalg.norm(x.astype(np.float64))),
                    repeat_l2=float(np.linalg.norm(y.astype(np.float64)))))
        keys = ("input_ids", "attention_mask", "position_ids", "cache_length_before", "cache_length_after", "sampling")
        steps.append(dict(forward_index=index, phase=a["phase"], physical_rows=32, active_rows=sum(active),
            exact_equal={key: a[key] == b[key] for key in keys}, mismatch_counts=metrics,
            zero_dose_before_after_identity=all(r["before_sha256"] == r["after_sha256"]
                                               for f in (a, b) for r in f["geometry"]),
            saved_vector_comparisons=samples))
    raw = [[{key: row[key] for key in ("answer_token_ids", "text")} for row in rows] for rows in records]
    changed = sorted({key for a, b in zip(*records, strict=True) for key in a
                      if key != "trial_id" and a[key] != b[key]})
    projections = {key: dict(changed_rows=sum(a[key] != b[key] for a, b in zip(*records, strict=True)),
                            max_absolute_difference=max(abs(a[key] - b[key]) for a, b in zip(*records, strict=True)))
                   for key in ("mean_proj", "mean_proj_monitor")}
    return dict(schema_version=1, engineering_only=True, treatment_effects_inspected=False,
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        scope="Saved primary baseline0000 versus repeat baseline0010 only; no condition effect analysis",
        checker=binding(__file__), command="OPENBLAS_NUM_THREADS=2 .pain-runtime-venv/bin/python scripts/diagnose_pain_profile_baseline_repeat.py",
        numpy_version=np.__version__,
        source_bindings=dict(manifest=binding(RUN / "manifest.json", manifest), freeze=binding(RUN / "freeze.json", frozen),
            source_sha256=SOURCE_SHA, source_archive_byte_sha256=ARCHIVE_SHA,
            archived_engine=dict(path=ENGINE, byte_sha256=engine_sha),
            setup_summary=binding(phase / "summary.json", summary), runtime_identity=binding(phase / "identity.json", identity), **inputs),
        preserved_setup_disposition={key: summary[key] for key in (
            "assigned_calls", "observed_calls", "assigned_rows", "completed_rows", "missing_rows", "complete", "passed", "integrity_passed", "terminal_error")},
        pair=dict(source_cohort=0, rows=32, base_request_ids_identical=left["call_spec"]["base_request_ids"] == right["call_spec"]["base_request_ids"],
            hooks_empty_before_after=all(r["hooks_before"] == r["hooks_after"] == {} for r in reports),
            raw_answer_tokens_and_text_identical=raw[0] == raw[1], raw_outputs_sha256=[digest(value) for value in raw],
            changed_record_fields_except_trial_id=changed, projection_differences=projections,
            all_observed_inputs_cache_lengths_and_sampling_identical=all(all(step["exact_equal"].values()) for step in steps),
            exact_repeat_gate_passed=False),
        steps=steps, runtime={key: identity[key] for key in ("gpu", "attention", "numerical_flags")},
        interpretation=dict(concrete_implementation_defect_found=False, numerical_cause_established=False,
            finding="Decode residuals diverge before the baseline steering hook; identical sampled answers do not repair full-state repeat failure.",
            source_review="Each call constructs fresh DynamicCache and row generators; zero coefficients bypass addition; observers clone without mutation; hook cleanup is empty. No cross-call state leak was identified.",
            original_gate_remains_failed=True, retry_or_benchmark_authorized=False),
        limitations=["No saved cache tensor contents or hashes; equal cache lengths do not establish identical cache values.",
            "No selected CUDA kernel trace or allocator/workspace trace; SDPA/cuBLAS/cache causes cannot be distinguished.",
            "Full hidden arrays are saved only for physical rows0 and31; all-row differences use original whole-row fingerprints.",
            "This engineering receipt does not interpret condition outcomes or amend any numerical/scientific threshold."])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    require(not args.out.exists(), "Diagnosis output is immutable")
    result = diagnose()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(dict(path=str(args.out), byte_sha256=sha(args.out), exact_repeat_gate_passed=False,
        full_logit_mismatch_counts=[s["mismatch_counts"]["full_vocabulary_logits"] for s in result["steps"]])))
