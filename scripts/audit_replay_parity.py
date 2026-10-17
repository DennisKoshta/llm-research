"""Audit exact-prefix replay against original incremental-decoding measurements."""

import argparse
from pathlib import Path
import hashlib
import numpy as np
from curiosity_study.artifacts import read, write_once, digest, RunStore


def audit(root):
    manifest = read(root / "manifest.json")
    rows = RunStore(root, manifest).completed()
    assert set(rows) == {a["trial_id"] for a in manifest["diagnostics"]}
    physical = sum(len(read(p)["records"]) for p in (root / "cohorts").glob("*/*.json"))
    assert physical == len(rows)
    by_source = {}
    arrays = {}

    def array(path, sha):
        if path not in arrays:
            assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
            with np.load(path, allow_pickle=False) as z:
                arrays[path] = {k: z[k] for k in z.files}
        return arrays[path]

    for a in manifest["diagnostics"]:
        r = rows[a["trial_id"]]
        p = manifest["prefixes"][a["prefix_id"]]
        assert r["assignment"] == a and r["source_assignment"] == p["assignment"]
        measurement = r["measurement"]
        assert measurement["unpadded_input_ids"] == p["token_prefix"]
        assert measurement["token_prefix_sha256"] == digest(p["token_prefix"]) == p["token_prefix_sha256"]
        assert measurement["unpadded_token_positions"] == p["prefix_positions"]
        assert measurement["exact_token_replay"] and measurement["hooks_remaining"] == 0
        assert r["reference_label_logits"] == p["reference_label_logits"]
        assert (
            r["reference_label_probabilities_temperature_1"]
            == p["reference_label_probabilities_temperature_1"]
        )
        by_source.setdefault(a["prefix_id"], {})[a["arm"]] = r
    differences, greedy_matches, shams, capture_checks, sham_captures = [], [], [], [], []
    for key, arms in by_source.items():
        assert set(arms) == {"baseline", "sham"}
        prefix = manifest["prefixes"][key]
        base, sham = arms["baseline"], arms["sham"]
        b, s = base["measurement"], sham["measurement"]
        shams.append(
            b["label_logits"] == s["label_logits"]
            and b["label_probabilities_temperature_1"] == s["label_probabilities_temperature_1"]
            and s["intervention"]["delta_norm"] == 0
        )
        reference = base["reference_label_probabilities_temperature_1"]
        differences.extend(
            abs(b["label_probabilities_temperature_1"][label] - reference[label]) for label in reference
        )
        greedy_matches.append(
            max(b["label_logits"], key=b["label_logits"].get)
            == max(base["reference_label_logits"], key=base["reference_label_logits"].get)
        )
        original_meta = prefix["source_activation"]
        original = array(root.parent / prefix["source_run"] / original_meta["path"], original_meta["sha256"])
        meta = base["activation_artifact"]
        got = array(root / meta["path"], meta["sha256"])
        sm = sham["activation_artifact"]
        sz = array(root / sm["path"], sm["sha256"])
        assert (
            got["trial_ids"][meta["row"]] == base["trial_id"]
            and sz["trial_ids"][sm["row"]] == sham["trial_id"]
        )
        baseline_hidden = got["hidden"][meta["row"]]
        sham_hidden = sz["hidden"][sm["row"]]
        assert baseline_hidden.shape == sham_hidden.shape
        assert np.isfinite(baseline_hidden).all() and np.isfinite(sham_hidden).all()
        sham_captures.append(
            dict(
                source=key,
                identical=bool(np.array_equal(baseline_hidden, sham_hidden)),
                maximum_absolute_difference=float(
                    np.max(np.abs(baseline_hidden.astype(float) - sham_hidden.astype(float)))
                ),
                fraction_different=float(np.mean(baseline_hidden != sham_hidden)),
            )
        )
        for li, layer in enumerate(meta["layers"]):
            oli = list(original["layers"]).index(layer)
            for pi, pos in enumerate(meta["positions"]):
                opi = list(original["positions"]).index("decision" if pos == "final" else pos)
                first = original["hidden"][original_meta["row"], oli, opi].astype(float)
                second = got["hidden"][meta["row"], li, pi].astype(float)
                assert np.isfinite(first).all() and np.isfinite(second).all()
                relative = np.linalg.norm(second - first) / np.linalg.norm(first)
                cosine = np.dot(first, second) / np.linalg.norm(first) / np.linalg.norm(second)
                capture_checks.append(
                    dict(
                        source=key,
                        layer=layer,
                        position=pos,
                        relative_l2=float(relative),
                        cosine=float(cosine),
                    )
                )
    differences = np.array(differences)
    criteria = manifest["parity_criteria"]
    checks = dict(
        all_assigned_measured=len(rows) == len(manifest["diagnostics"]),
        all_eligible_sources_measured=len(by_source) == len(manifest["prefixes"]),
        allowed_label_argmax_match=bool(all(greedy_matches)),
        maximum_probability_difference=float(differences.max())
        <= criteria["maximum_label_probability_difference"],
        fraction_small_differences=float(np.mean(differences <= 0.01)) >= criteria["fraction_within_0_01"],
        sham_identity=all(shams),
        sham_capture_identity=all(c["identical"] for c in sham_captures),
        capture_cosine=all(c["cosine"] >= 0.995 for c in capture_checks),
        capture_relative_l2=all(c["relative_l2"] <= 0.03 for c in capture_checks),
    )
    result = dict(
        manifest_sha256=digest(manifest),
        audit_implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model=manifest["model"],
        engine_sha256=manifest["engine_sha256"],
        assigned_sources=80,
        eligible_sources=len(by_source),
        excluded=manifest["excluded_sources"],
        measurements=len(rows),
        passed=all(checks.values()),
        checks=checks,
        maximum_probability_difference=float(differences.max()),
        fraction_probability_differences_within_0_01=float(np.mean(differences <= 0.01)),
        greedy_label_matches=sum(greedy_matches),
        minimum_capture_cosine=min(c["cosine"] for c in capture_checks),
        maximum_capture_relative_l2=max(c["relative_l2"] for c in capture_checks),
        sham_capture_pairs_identical=sum(c["identical"] for c in sham_captures),
        maximum_sham_capture_absolute_difference=max(c["maximum_absolute_difference"] for c in sham_captures),
        sham_captures=sham_captures,
        captures=capture_checks,
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = audit(args.run)
    write_once(args.out, result)
    print({k: v for k, v in result.items() if k not in ("captures", "sham_captures")})
