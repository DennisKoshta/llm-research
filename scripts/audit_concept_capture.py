"""Independently reconcile every plain-text prefix, capture and atomic cohort."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np

from curiosity_study import concept_capture as concept
from curiosity_study.artifacts import RunStore, digest, read, write_once
from curiosity_study.state_backend import PREFILL_PADDING_MULTIPLE
from collection_coverage import check_collection_coverage


def byte_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def archive_bindings(root, manifest):
    frozen = read(root / "freeze.json")
    concept.require(
        frozen["manifest_sha256"] == digest(manifest)
        and frozen["source_sha256"] == manifest["source_sha256"]
        and frozen["archive_sha256"] == byte_sha(root / "source.tar.gz"),
        "Concept collection source freeze mismatch",
    )
    with tarfile.open(root / "source.tar.gz", "r:gz") as archive:
        for binding in manifest["document_bindings"]:
            member = archive.getmember(binding["path"])
            concept.require(member.isfile(), "Frozen document must be a regular file")
            payload = archive.extractfile(member).read()
            concept.require(
                hashlib.sha256(payload).hexdigest()
                == binding["byte_sha256"]
                == frozen["documents"].get(binding["path"]),
                "Frozen document byte hash mismatch",
            )
        for binding in manifest["material_bindings"]:
            payload = archive.extractfile(binding["path"]).read()
            concept.require(
                hashlib.sha256(payload).hexdigest() == binding["byte_sha256"]
                and json.loads(payload) == manifest["materials"][binding["corpus_id"]],
                "Frozen corpus bytes disagree with the actual capture assignments",
            )
        review = manifest["material_review_binding"]
        payload = archive.extractfile(review["path"]).read()
        concept.require(
            hashlib.sha256(payload).hexdigest() == review["byte_sha256"]
            and json.loads(payload) == manifest["material_review"],
            "Frozen material-review binding mismatch",
        )
        implementation = archive.extractfile("src/curiosity_study/concept_capture.py").read()
        concept.require(
            hashlib.sha256(implementation).hexdigest()
            == manifest["task_module_sha256"]
            == byte_sha(concept.__file__),
            "Concept capture implementation differs from its freeze",
        )
    return dict(freeze_sha256=digest(frozen), archive_sha256=frozen["archive_sha256"])


def audit_integrity(integrity, manifest, prepared):
    hashes = [request["token_prefix_sha256"] for request, _ in prepared[:32]]
    expected_checks = {
        "full_initial_logits_equal",
        "every_row_fingerprint_equal",
        "all_captures_finite",
        "capture_shape",
        "no_intervention_or_generation",
        "hook_cleanup",
    }
    before = integrity.get("baseline_full_initial_logits_sha256", [])
    after = integrity.get("captured_full_initial_logits_sha256", [])
    concept.require(
        integrity.get("passed") is True
        and set(integrity.get("checks", {})) == expected_checks
        and all(value is True for value in integrity["checks"].values()),
        "Concept startup integrity failed",
    )
    concept.require(
        integrity.get("manifest_sha256") == digest(manifest)
        and integrity.get("cohort_trial_ids") == [a["trial_id"] for a in manifest["diagnostics"][:32]]
        and integrity.get("source_prefix_sha256") == hashes
        and integrity.get("batch_size") == 32
        and integrity.get("layers") == concept.LAYERS
        and integrity.get("positions") == concept.POSITIONS,
        "Startup integrity does not bind the actual first32 narrative prefixes and capture sites",
    )
    concept.require(
        len(before) == 32
        and before == after
        and all(isinstance(value, str) and len(value) == 64 for value in before),
        "Read-only captures changed full initial vocabulary logits",
    )
    longest = max(len(request["token_prefix"]) for request, _ in prepared[:32])
    width = ((longest + PREFILL_PADDING_MULTIPLE - 1) // PREFILL_PADDING_MULTIPLE) * PREFILL_PADDING_MULTIPLE
    concept.require(
        integrity.get("prefill_width") == width, "Startup layout width does not match frozen prefixes"
    )


def audit(root, tokenizer=None):
    root = Path(root)
    manifest = read(root / "manifest.json")
    assignments = concept.validate_manifest(manifest)
    frozen = archive_bindings(root, manifest)
    coverage = check_collection_coverage(root, expected_manifest_sha256=digest(manifest))
    records = RunStore(root, manifest).completed()
    concept.require(set(records) == {a["trial_id"] for a in assignments}, "Incomplete concept collection")
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            manifest["model"]["model_id"], revision=manifest["model"]["revision"], trust_remote_code=False
        )
    prepared = [concept.inference_request(a, tokenizer) for a in assignments]
    files, file_shas, integrity_bindings, record_bindings = {}, {}, {}, []
    for offset in range(0, len(assignments), 32):
        cohort = assignments[offset : offset + 32]
        ids = [a["trial_id"] for a in cohort]
        requests = prepared[offset : offset + 32]
        longest = max(len(request["token_prefix"]) for request, _ in requests)
        width = (
            (longest + PREFILL_PADDING_MULTIPLE - 1) // PREFILL_PADDING_MULTIPLE
        ) * PREFILL_PADDING_MULTIPLE
        artifact_paths, cohort_ids = set(), set()
        for row, (assignment, (request, binding)) in enumerate(zip(cohort, requests)):
            record = records[assignment["trial_id"]]
            measurement, artifact, layout = (
                record["measurement"],
                record["activation_artifact"],
                record["cohort"],
            )
            concept.require(
                record["assignment"] == assignment
                and record.get("terminal") is True
                and record.get("kind") == "concept_capture"
                and "failure" not in record
                and not {"correct", "valid", "answer", "expected"} & record.keys(),
                "Concept capture cannot filter or score narrative outcomes",
            )
            concept.require(
                record["tokenization"] == binding and record["cohort_sha256"] == digest(layout),
                "Narrative tokenization or cohort binding differs",
            )
            expected_layout = dict(
                id=layout["id"],
                offset=offset,
                row=row,
                size=32,
                trial_ids=ids,
                prefix_hashes=[r["token_prefix_sha256"] for r, _ in requests],
                prefill_width=width,
            )
            concept.require(
                layout == expected_layout, "Actual capture cohort differs from assigned membership/order"
            )
            cohort_ids.add(layout["id"])
            tokens = request["token_prefix"]
            pad = width - len(tokens)
            concept.require(
                measurement["unpadded_input_ids"] == tokens
                and measurement["token_prefix_sha256"] == digest(tokens)
                and measurement["exact_token_replay"] is True
                and measurement["input_ids"] == [tokenizer.pad_token_id] * pad + tokens
                and measurement["attention_mask"] == [0] * pad + [1] * len(tokens)
                and measurement["position_ids"] == [0] * pad + list(range(len(tokens)))
                and measurement["prefill_width"] == width
                and measurement["prefill_padding_multiple"] == PREFILL_PADDING_MULTIPLE,
                "Plain-text token IDs, padding, mask or positions differ from independently tokenized material",
            )
            concept.require(
                measurement["unpadded_token_positions"] == request["prefix_positions"]
                and measurement["record_token_index"] == pad + request["prefix_positions"]["record"]
                and measurement["rendered_prompt"] == tokenizer.decode(tokens, skip_special_tokens=False)
                and measurement["prompt_sha256"] == digest(measurement["rendered_prompt"]),
                "Narrative-end/final capture tokens or rendered prefix differ",
            )
            concept.require(
                measurement["hooks_remaining"] == 0
                and measurement["intervention"] is None
                and "completion" not in measurement
                and set(measurement["label_logits"])
                == set(measurement["label_probabilities_temperature_1"])
                == {"A", "B"}
                and all(np.isfinite(v) for v in measurement["label_logits"].values()),
                "Require read-only capture, hook cleanup, and engineering-only initial scores",
            )
            relative = Path(artifact["path"])
            concept.require(
                not relative.is_absolute()
                and ".." not in relative.parts
                and len(relative.parts) == 3
                and relative.parts[0] == "activations"
                and relative.name == f"{offset:06d}.npz",
                "Capture artifact path/cohort mismatch",
            )
            attempt = relative.parts[1]
            concept.require(layout["id"] == f"{attempt}-{offset:06d}", "Cohort attempt ID mismatch")
            artifact_paths.add(str(relative))
            if attempt not in integrity_bindings:
                integrity = read(root / "concept_integrity" / f"{attempt}.json")
                audit_integrity(integrity, manifest, prepared)
                environment = read(root / "environments" / f"{attempt}.json")
                concept.require(
                    all(environment.get(k) == v for k, v in manifest["model"].items()),
                    "Capture environment model identity differs",
                )
                integrity_bindings[attempt] = dict(
                    integrity_sha256=digest(integrity),
                    environment_sha256=digest(environment),
                    hidden_size=environment["concept_capture_hidden_size"],
                )
            if str(relative) not in files:
                path = root / relative
                concept.require(byte_sha(path) == artifact["sha256"], "Capture artifact byte hash mismatch")
                with np.load(path, allow_pickle=False) as z:
                    files[str(relative)] = {key: z[key] for key in z.files}
                file_shas[str(relative)] = artifact["sha256"]
            concept.require(
                artifact["sha256"] == file_shas[str(relative)], "Inconsistent hash for one capture file"
            )
            z = files[str(relative)]
            size = integrity_bindings[attempt]["hidden_size"]
            positions = [request["prefix_positions"][p] for p in concept.POSITIONS]
            concept.require(
                artifact["row"] == row
                and artifact["layers"] == concept.LAYERS
                and artifact["positions"] == concept.POSITIONS
                and artifact["token_positions"] == positions
                and artifact["available"] == [[True, True]] * 4,
                "Capture row/site metadata mismatch",
            )
            concept.require(
                z["hidden"].shape == (32, 4, 2, size)
                and z["trial_ids"].tolist() == ids
                and z["layers"].tolist() == concept.LAYERS
                and z["positions"].tolist() == concept.POSITIONS
                and z["token_positions"][row].tolist() == positions
                and z["token_prefix_sha256"][row] == request["token_prefix_sha256"]
                and z["available"].shape == (32, 4, 2)
                and np.isfinite(z["hidden"]).all()
                and z["available"].all(),
                "Every assigned narrative requires every finite, correctly located residual capture",
            )
            record_bindings.append(
                dict(
                    trial_id=assignment["trial_id"],
                    record_sha256=digest(record),
                    artifact_sha256=artifact["sha256"],
                    row=row,
                    token_prefix_sha256=request["token_prefix_sha256"],
                )
            )
        concept.require(
            len(artifact_paths) == len(cohort_ids) == 1, "Each cohort requires one atomic capture artifact"
        )
    return dict(
        schema_version=1,
        assay="concept_capture",
        complete=True,
        passed=True,
        manifest_sha256=digest(manifest),
        assigned=len(assignments),
        capture_plan=manifest["capture_plan"],
        model=manifest["model"],
        source_sha256=manifest["source_sha256"],
        max_new_tokens=0,
        record_bindings_sha256=digest(record_bindings),
        capture_files=len(files),
        atomic_cohorts=12,
        captures_per_text=8,
        all_assigned_captures_finite=True,
        coverage=coverage,
        integrity_bindings=integrity_bindings,
        source_freeze=frozen,
        material_review_sha256=digest(manifest["material_review"]),
        semantic_metrics_computed=False,
        directions_fitted=False,
        interpretation="Read-only capture integrity only. No completed answers, psychological validation, model state induction, subjective experience, or natural-generation engineering claim.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    concept.require(not args.out.exists(), "Concept audit receipts are immutable")
    result = audit(args.run)
    write_once(args.out, result)
    print({key: result[key] for key in ("complete", "passed", "assigned", "capture_files")})
