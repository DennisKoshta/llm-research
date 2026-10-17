"""Describe parity failures by physical row and sequence width without changing gates."""

import argparse
from collections import defaultdict
import hashlib
from pathlib import Path

import numpy as np

from curiosity_study.artifacts import RunStore, digest, read, write_once


def vector_difference(first, second):
    first, second = first.astype(float), second.astype(float)
    assert np.isfinite(first).all() and np.isfinite(second).all()
    return dict(
        identical=bool(np.array_equal(first, second)),
        fraction_different=float(np.mean(first != second)),
        maximum_absolute_difference=float(np.max(np.abs(first - second))),
        relative_l2=float(np.linalg.norm(second - first) / np.linalg.norm(first)),
        cosine=float(np.dot(first, second) / np.linalg.norm(first) / np.linalg.norm(second)),
    )


def diagnose(run, audit_path):
    manifest, audit = read(run / "manifest.json"), read(audit_path)
    assert audit["manifest_sha256"] == digest(manifest)
    records = RunStore(run, manifest).completed()
    assert set(records) == {a["trial_id"] for a in manifest["diagnostics"]}
    arrays = {}

    def activation(record):
        meta = record["activation_artifact"]
        path = run / meta["path"]
        if path not in arrays:
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path, allow_pickle=False) as archive:
                arrays[path] = sha, {key: archive[key] for key in archive.files}
        sha, array = arrays[path]
        assert sha == meta["sha256"]
        assert array["trial_ids"][meta["row"]] == record["trial_id"]
        assert list(array["layers"]) == meta["layers"] and list(array["positions"]) == meta["positions"]
        return array["hidden"][meta["row"]]

    cohorts, pairs = [], []
    size = manifest["execution"]["batch_size"]
    for offset in range(0, len(manifest["diagnostics"]), size):
        assignments = manifest["diagnostics"][offset : offset + size]
        local = [records[a["trial_id"]] for a in assignments]
        width = len(local[0]["measurement"]["input_ids"])
        rows, grouped = [], defaultdict(dict)
        for row, record in enumerate(local):
            measure = record["measurement"]
            assert len(measure["input_ids"]) == width
            assert record["activation_artifact"]["row"] == row
            grouped[record["assignment"]["prefix_id"]][record["assignment"]["arm"]] = (row, record)
            rows.append(
                dict(
                    row=row,
                    trial_id=record["trial_id"],
                    prefix_id=record["assignment"]["prefix_id"],
                    arm=record["assignment"]["arm"],
                    unpadded_length=len(measure["unpadded_input_ids"]),
                    padding=measure["attention_mask"].count(0),
                    flattened_capture_positions={
                        position: row * width + measure["attention_mask"].count(0) + token
                        for position, token in measure["unpadded_token_positions"].items()
                    },
                )
            )
        bad_pairs = []
        for key, arms in grouped.items():
            assert set(arms) == {"baseline", "sham"}
            br, baseline = arms["baseline"]
            sr, sham = arms["sham"]
            b, s = baseline["measurement"], sham["measurement"]
            bh, sh = activation(baseline), activation(sham)
            captures = []
            for li, layer in enumerate(baseline["activation_artifact"]["layers"]):
                for pi, position in enumerate(baseline["activation_artifact"]["positions"]):
                    captures.append(
                        dict(layer=layer, position=position, **vector_difference(bh[li, pi], sh[li, pi]))
                    )
            pair = dict(
                source=key,
                cohort_offset=offset,
                baseline_row=br,
                sham_row=sr,
                width=width,
                identical_inputs=all(
                    b[field] == s[field]
                    for field in ("input_ids", "attention_mask", "position_ids", "unpadded_token_positions")
                ),
                identical_label_logits=b["label_logits"] == s["label_logits"],
                identical_label_probabilities=b["label_probabilities_temperature_1"]
                == s["label_probabilities_temperature_1"],
                maximum_logit_difference=max(
                    abs(b["label_logits"][label] - s["label_logits"][label]) for label in b["label_logits"]
                ),
                maximum_probability_difference=max(
                    abs(
                        b["label_probabilities_temperature_1"][label]
                        - s["label_probabilities_temperature_1"][label]
                    )
                    for label in b["label_logits"]
                ),
                sham_delta_norm=s["intervention"]["delta_norm"],
                identical_hidden=all(c["identical"] for c in captures),
                captures=captures,
            )
            if not pair["identical_hidden"]:
                bad_pairs.append(dict(source=key, baseline_row=br, sham_row=sr))
            pairs.append(pair)
        cohorts.append(
            dict(
                offset=offset,
                width=width,
                width_mod8=width % 8,
                width_mod16=width % 16,
                flattened_rows_mod128=len(local) * width % 128,
                flattened_rows_mod256=len(local) * width % 256,
                rows=rows,
                different_hidden_pairs=bad_pairs,
            )
        )
    source_groups = defaultdict(list)
    for capture in audit["captures"]:
        source_groups[capture["layer"], capture["position"]].append(capture)
    summaries = [
        dict(
            layer=layer,
            position=position,
            pairs=len(captures),
            relative_l2_failures=sum(c["relative_l2"] > 0.03 for c in captures),
            maximum_relative_l2=max(c["relative_l2"] for c in captures),
            median_relative_l2=float(np.median([c["relative_l2"] for c in captures])),
            minimum_cosine=min(c["cosine"] for c in captures),
        )
        for (layer, position), captures in sorted(source_groups.items())
    ]
    return dict(
        manifest_sha256=digest(manifest),
        audit_sha256=digest(audit),
        diagnostic_implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        engine_sha256=manifest["engine_sha256"],
        source_capture_groups=summaries,
        all_input_pairs_identical=all(pair["identical_inputs"] for pair in pairs),
        identical_hidden_pairs=sum(pair["identical_hidden"] for pair in pairs),
        identical_label_logit_pairs=sum(pair["identical_label_logits"] for pair in pairs),
        identical_label_probability_pairs=sum(pair["identical_label_probabilities"] for pair in pairs),
        maximum_sham_logit_difference=max(pair["maximum_logit_difference"] for pair in pairs),
        maximum_sham_probability_difference=max(pair["maximum_probability_difference"] for pair in pairs),
        cohorts=cohorts,
        pairs=pairs,
        interpretation="Row localization is evidence for a numerical-layout hypothesis, not proof of a kernel cause. Differences already present before the intervention layer cannot be caused by that zero patch. Original-to-replay capture differences are a separate gate; no threshold is relaxed by this diagnostic.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args.run, args.audit)
    write_once(args.out, result)
    print(
        {
            key: value
            for key, value in result.items()
            if key not in ("cohorts", "pairs", "source_capture_groups")
        }
    )
