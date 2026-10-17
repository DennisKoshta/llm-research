"""Independent, bounded numerical recheck of saved sustained-engineering evidence.

This does not call the producer's validators or replace the complete source/record
provenance audit. It cannot recover full logits from their saved fingerprints.
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from curiosity_study.artifacts import digest, read, write_once

CONDITIONS = ("baseline", "zero_sham", "one_row_nonzero", "clean_repeat")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def active_predictions(condition, row):
    return [
        (f["forward_index"], f["prediction_ordinal"], f["full_fp32_logits_sha256"][row])
        for f in condition["forwards"]
        if f["active_prediction_rows"][row]
    ]


def compare(left, right, indices):
    token_matches, fingerprint_matches, expected = 0, 0, 0
    inputs_equal = True
    for row in indices:
        a = left["rows"][row]["completion"]
        b = right["rows"][row]["completion"]
        token_matches += a == b
        lp, rp = active_predictions(left, row), active_predictions(right, row)
        expected += max(len(lp), len(rp))
        fingerprint_matches += sum(x == y for x, y in zip(lp, rp))
        for lf, rf in zip(left["forwards"], right["forwards"]):
            if lf["active_prediction_rows"][row] or rf["active_prediction_rows"][row]:
                inputs_equal &= all(
                    lf[key][row] == rf[key][row]
                    for key in ("input_ids", "position_ids", "attention_mask_row_sha256")
                )
    return dict(
        passed=token_matches == len(indices) and fingerprint_matches == expected and inputs_equal,
        rows=len(indices),
        identical_completions=token_matches,
        compared_active_predictions=expected,
        identical_full_fp32_vocabulary_fingerprints=fingerprint_matches,
        active_input_position_and_mask_fingerprints_equal=bool(inputs_equal),
    )


def timeline(condition, requests, cap, padding, eos, pad):
    rows, frames = condition["rows"], condition["forwards"]
    lengths = [len(row["completion"]["generated_token_ids"]) for row in rows]
    width = ((max(len(r["token_prefix"]) for r in requests) + padding - 1) // padding) * padding
    valid = len(rows) == len(requests) == 32 and len(frames) == max(lengths)
    valid &= condition["attempted_forwards"] == len(frames) and condition["hooks_clean"] is True
    valid &= condition["error"] is None and all(1 <= n <= cap for n in lengths)
    for row in rows:
        tokens = row["completion"]["generated_token_ids"]
        valid &= not any(t in eos for t in tokens[:-1])
        valid &= row["completion"]["ended_with_eos"] is (tokens[-1] in eos)
        valid &= tokens[-1] in eos or len(tokens) == cap
    for step, frame in enumerate(frames):
        active = [step < n for n in lengths]
        valid &= frame["forward_index"] == step and frame["prediction_ordinal"] == step + 1
        valid &= frame["phase"] == ("prefill" if step == 0 else "decode")
        valid &= frame["active_prediction_rows"] == active
        valid &= frame["cache_length_before"] == (0 if step == 0 else width + step - 1)
        valid &= frame["cache_length_after"] == width + step
        valid &= frame["attention_mask_shape"] == [32, width + step]
        valid &= frame["logit_dtype"] == "float32" and not frame["cache_position_supplied"]
        for index, (row, request, live) in enumerate(zip(rows, requests, active)):
            prefix = request["token_prefix"]
            tokens = row["completion"]["generated_token_ids"]
            fingerprint = frame["full_fp32_logits_sha256"][index]
            valid &= isinstance(fingerprint, str) and len(fingerprint) == 64 if live else fingerprint is None
            expected_input = (
                [pad] * (width - len(prefix)) + prefix
                if step == 0
                else [tokens[step - 1] if step <= len(tokens) else pad]
            )
            valid &= frame["input_ids"][index] == expected_input
    return dict(
        passed=bool(valid),
        forwards=len(frames),
        row_lengths=lengths,
        maximum_tokens=max(lengths),
        minimum_tokens=min(lengths),
        ended_with_eos=sum(row["completion"]["ended_with_eos"] for row in rows),
        reached_cap=sum(n == cap for n in lengths),
        active_predictions=sum(lengths),
        decode_observed=len(frames) > 1,
        mixed_active_inactive_decode_observed=any(
            any(f["active_prediction_rows"]) and not all(f["active_prediction_rows"]) for f in frames[1:]
        ),
    )


def geometry(condition, arrays, direction, dose):
    before, after = arrays["before"], arrays["after"]
    require(before.dtype == after.dtype == np.float32, "Saved BF16 vectors must be represented as FP32")
    require(before.shape == after.shape and before.ndim == 2, "Before/after shape mismatch")
    unit = torch.as_tensor(direction, dtype=torch.float32)
    require(before.shape[1] == len(unit), "Direction width mismatch")
    require(np.isfinite(before).all() and np.isfinite(after).all(), "Nonfinite vectors")
    events = condition["rows"][0]["intervention_trace"]["events"]
    sites = condition["vector_sites"]
    require(len(events) == len(sites) == len(before), "Vector/event coverage mismatch")
    exact, bf16, index_ok = True, True, True
    metrics = {
        name: [] for name in ("delta_norm", "relative_delta_norm", "projection_change", "rounding_error")
    }
    scalar = torch.tensor(dose, dtype=torch.float32)
    for start in range(0, len(before), 128):
        a = torch.from_numpy(before[start : start + 128])
        b = torch.from_numpy(after[start : start + 128])
        requested = a + scalar * unit
        expected = requested.to(torch.bfloat16).to(torch.float32)
        exact &= torch.equal(expected, b)
        bf16 &= torch.equal(a.to(torch.bfloat16).float(), a) and torch.equal(b.to(torch.bfloat16).float(), b)
        delta = b - a
        magnitude = torch.linalg.vector_norm(delta, dim=1)
        metrics["delta_norm"].extend(magnitude.tolist())
        metrics["relative_delta_norm"].extend((magnitude / torch.linalg.vector_norm(a, dim=1)).tolist())
        metrics["projection_change"].extend((delta @ (unit / torch.linalg.vector_norm(unit))).tolist())
        metrics["rounding_error"].extend(torch.linalg.vector_norm(b - requested, dim=1).tolist())
    for step, (event, site) in enumerate(zip(events, sites)):
        phase = "prefill" if step == 0 else "decode"
        index_ok &= event["phase"] == phase and event["requested_scalar_delta"] == dose
        index_ok &= event["arithmetic_scalar_float32"] == float(scalar)
        index_ok &= event["processed_generated_token_index"] == (None if step == 0 else step)
        index_ok &= site["forward_index"] == step and site["prediction_ordinal"] == step + 1
        index_ok &= site["phase"] == phase and site["tensor_token_index"] == event["token_index"]
        index_ok &= condition["vector_activation_dtypes"][step] == "torch.bfloat16"
    ranges = {name: dict(min=min(values), max=max(values)) for name, values in metrics.items()}
    return dict(
        passed=bool(exact and bf16 and index_ok and len(events) > 1 and min(metrics["delta_norm"]) > 0),
        bitwise_reconstructed_after=bool(exact),
        saved_vectors_exact_bf16=bool(bf16),
        event_site_and_scalar_binding=bool(index_ok),
        vector_pairs=len(events),
        prefill_events=1,
        decode_events=len(events) - 1,
        requested_dose=dose,
        arithmetic_dose_float32=float(scalar),
        independently_recomputed_ranges=ranges,
    )


def check_evidence(manifest, wrapper, arrays):
    h = wrapper["harness"]
    conditions = h["conditions"]
    require(wrapper["status"] == "measured" and set(conditions) == set(CONDITIONS), "Incomplete measurements")
    require(wrapper["cap"] == h["cap"] == manifest["cap"], "Cap binding mismatch")
    require(h["request_sha256"] == digest(manifest["requests"]), "Request binding mismatch")
    direction = np.asarray(manifest["direction"]["unit"], dtype=np.float32)
    require(np.array_equal(direction, arrays["direction_float32"]), "Effective direction mismatch")
    require(h["dose"] == manifest["direction"]["dose"], "Dose binding mismatch")
    identities = dict(
        sham=compare(conditions["baseline"], conditions["zero_sham"], list(range(32))),
        unaffected31=compare(conditions["baseline"], conditions["one_row_nonzero"], list(range(1, 32))),
        cleanup_repeat=compare(conditions["baseline"], conditions["clean_repeat"], list(range(32))),
    )
    timelines = {
        name: timeline(
            c,
            manifest["requests"],
            h["cap"],
            h["prefill_padding_multiple"],
            set(h["eos_token_ids"]),
            h["pad_token_id"],
        )
        for name, c in conditions.items()
    }
    arithmetic = geometry(conditions["one_row_nonzero"], arrays, direction, h["dose"])
    return dict(
        passed=all(r["passed"] for r in [*identities.values(), *timelines.values(), arithmetic]),
        identities=identities,
        timelines=timelines,
        arithmetic=arithmetic,
        observed_maximum_tokens=max(row["maximum_tokens"] for row in timelines.values()),
    )


def check_run(run, audit_path):
    run, audit_path = Path(run).resolve(), Path(audit_path).resolve()
    manifest, audit = read(run / "manifest.json"), read(audit_path)
    require(audit["manifest_sha256"] == digest(manifest), "Audit manifest mismatch")
    require(
        all(audit.get(k) is True for k in ("passed", "complete", "integrity_passed")), "Main audit failed"
    )
    report_path = run / audit["report_artifact"]["path"]
    vector_path = run / audit["vector_artifact"]["path"]
    require(sha(report_path) == audit["report_artifact"]["sha256"], "Report bytes mismatch")
    require(sha(vector_path) == audit["vector_artifact"]["sha256"], "Vector bytes mismatch")
    wrapper = read(report_path)
    require(wrapper["manifest_sha256"] == digest(manifest), "Wrapper manifest mismatch")
    require(wrapper["vector_artifact"]["sha256"] == sha(vector_path), "Wrapper vector mismatch")
    with np.load(vector_path, allow_pickle=False) as loaded:
        result = check_evidence(manifest, wrapper, {key: loaded[key] for key in loaded.files})
    return dict(
        **result,
        run=str(run),
        cap=manifest["cap"],
        manifest_sha256=digest(manifest),
        audit_path=str(audit_path),
        audit_byte_sha256=sha(audit_path),
        report_path=str(report_path),
        report_byte_sha256=sha(report_path),
        vector_path=str(vector_path),
        vector_byte_sha256=sha(vector_path),
        performance=wrapper["performance"],
    )


def main(runs, audits, out):
    require(not Path(out).exists(), "Independent check receipt is immutable")
    require(len(runs) == len(audits) == 2, "Require short and full evidence together")
    results = [check_run(run, audit) for run, audit in zip(runs, audits)]
    require([r["cap"] for r in results] == [128, 4096], "Require ordered 128/4096 stages")
    result = dict(
        schema_version=1,
        assay="sustained_engineering_independent_arithmetic_check",
        passed=all(r["passed"] for r in results),
        scientific_admission=False,
        checker_byte_sha256=sha(__file__),
        stages=results,
        limitations=[
            "Independent CPU arithmetic and comparisons; no shared engineering validator is called.",
            "Full vocabulary identity compares recorded FP32 fingerprints, not retained vocabulary arrays.",
            "Relies on the main audit for source provenance, complete record coverage and full mask reconstruction.",
            "Observed lengths, active layouts and one-row curiosity perturbation do not certify all trajectories, "
            "all-row satisfaction steering, psychological specificity or subjective experience.",
        ],
    )
    write_once(Path(out), result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, action="append", required=True)
    p.add_argument("--audit", type=Path, action="append", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    print(dict(passed=main(args.run, args.audit, args.out)["passed"]))
