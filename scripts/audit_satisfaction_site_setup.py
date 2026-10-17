"""Recompute two-site setup evidence from frozen inputs and saved observations."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import tarfile
import tempfile

import numpy as np

from curiosity_study import causal_site_engineering as engine
from curiosity_study import satisfaction_causal_sites as study
from curiosity_study import satisfaction_site_setup as setup
from curiosity_study import voluntary_capture as receipts
from curiosity_study.artifacts import digest, read, write_once

from collection_coverage import check_collection_coverage
from satisfaction_site_sources import historical_baseline, setup_units, sha
from audit_satisfaction_screen import receipt_measurement

require = study.require


def checked_artifact(root, binding):
    name = Path(binding["path"])
    require(not name.is_absolute() and ".." not in name.parts, "Unsafe setup artifact path")
    path = Path(root) / name
    require(path.is_file() and path.stat().st_size == binding["byte_count"]
            and sha(path) == binding["sha256"], "Setup artifact bytes changed")
    return path


def frozen_documents(root, manifest):
    frozen = read(root / "freeze.json")
    require(frozen["manifest_sha256"] == digest(manifest)
            and frozen["source_sha256"] == manifest["source_sha256"]
            and frozen["archive_sha256"] == sha(root / "source.tar.gz"), "Setup freeze changed")
    documents, sources = {}, {}
    with tarfile.open(root / "source.tar.gz") as archive:
        for member in archive.getmembers():
            if member.name.startswith("src/curiosity_study/") and member.name.endswith(".py"):
                require(member.isfile(), "Source archive entry is not a regular file")
                sources[member.name.removeprefix("src/curiosity_study/")] = archive.extractfile(member).read().decode()
        require(digest(sources) == manifest["source_sha256"], "Frozen source inventory changed")
        for binding in manifest["document_bindings"]:
            name = binding["path"]
            require(not Path(name).is_absolute() and ".." not in Path(name).parts and name not in documents,
                    "Unsafe or duplicate source document")
            member = archive.getmember(name)
            require(member.isfile(), "Source document is not a regular file")
            payload = archive.extractfile(member).read()
            require(hashlib.sha256(payload).hexdigest() == binding["byte_sha256"] == frozen["documents"][name],
                    "Frozen document bytes changed")
            documents[name] = payload
        for name, expected in manifest["engine_sha256"].items():
            require(hashlib.sha256(sources[name].encode()).hexdigest() == expected == setup.engine_hashes()[name],
                    "Current or archived execution helper differs from fixed source")
        for name in ("satisfaction_site_sources.py", "prepare_satisfaction_site_setup.py",
                     "audit_satisfaction_screen.py", Path(__file__).name):
            payload = archive.extractfile("scripts/" + name).read()
            require(hashlib.sha256(payload).hexdigest() == sha(Path(__file__).with_name(name)),
                    "Current setup preparation/audit code differs from its freeze")
    return documents, dict(freeze_sha256=digest(frozen), archive_sha256=frozen["archive_sha256"])


def recompute_reference(report, arrays, manifest, *, tokenizer):
    requests, historical = manifest["requests"], setup.decode_historical(manifest["historical_hidden"])
    require(set(arrays) == {"hidden", "record_norms_float64"}, "Unexpected reference arrays")
    hidden, norms = arrays["hidden"], arrays["record_norms_float64"]
    require(hidden.dtype == np.float32 and hidden.shape == (288, 3, 2, historical.shape[-1])
            and norms.dtype == np.float64 and norms.shape == (288, 3)
            and np.array_equal(norms, np.linalg.norm(hidden[:, :, 0].astype(np.float64), axis=-1)),
            "Reference norms differ from the complete saved record vectors")
    bindings = {key: engine.array_binding(value) for key, value in arrays.items()}
    require(report["arrays"] == bindings and report["arrays_sha256"] == digest(bindings)
            and report["assigned"] == 288 and report["max_new_tokens"] == 0
            and report["layers"] == [15, 31, 47] and report["positions"] == ["record", "final"]
            and report["source_trial_ids"] == [r["source_trial_id"] for r in requests]
            and report["requests_sha256"] == digest(requests)
            and report["historical_binding"] == manifest["historical_binding"]
            and report["historical_binding_sha256"] == digest(manifest["historical_binding"])
            and report["original_dose"] == study.ORIGINAL_S_DOSE
            and len(report["cohorts"]) == 9, "Reference measurement contract changed")
    readonly, historical_equal, cleanup = [], [], []
    for index, cohort in enumerate(report["cohorts"]):
        offset = index * 32
        chunk, values, guard = requests[offset:offset + 32], cohort["measurements"], cohort["engineering"]
        width = ((max(len(r["token_prefix"]) for r in chunk) + engine.PREFILL_PADDING_MULTIPLE - 1)
                 // engine.PREFILL_PADDING_MULTIPLE) * engine.PREFILL_PADDING_MULTIPLE
        require(cohort["offset"] == offset and cohort["requests_sha256"] == digest(chunk)
                and len(values) == 32 and guard["source_trial_ids"] == [r["source_trial_id"] for r in chunk]
                and guard["token_prefix_sha256"] == [r["token_prefix_sha256"] for r in chunk]
                and guard["layers"] == [15, 31, 47] and guard["positions"] == ["record", "final"]
                and guard["active_batch_size"] == 32 and guard["max_new_tokens"] == 0
                and guard["prefill_width"] == width
                and set(guard["checks"]) == set(receipts.ENGINEERING_CHECKS)
                and all(type(value) is bool for value in guard["checks"].values())
                and guard["captured_array_sha256"] == receipts.array_sha(hidden[offset:offset + 32]),
                "Reference cohort identity or saved vector bytes changed")
        for row, request in zip(values, chunk, strict=True):
            require(receipt_measurement(row, request, tokenizer, width),
                    "Reference prompt, padding, positions, intervention or cleanup changed")
        left, right = guard["baseline_full_initial_logits_sha256"], guard["captured_full_initial_logits_sha256"]
        require(len(left) == len(right) == 32 and all(study.valid_sha(value) for value in left + right),
                "Missing full-vocabulary reference fingerprints")
        readable = left == right and all(row["hooks_remaining"] == 0 for row in values)
        readable = readable and all(guard["checks"].values()) and guard["passed"] is True
        readonly.append(readable)
        eq = np.array_equal(hidden[offset:offset + 32, :1], historical[offset:offset + 32])
        require(cohort["historical_layer15_equal"] is eq, "Historical identity flag disagrees with vectors")
        historical_equal.append(eq)
        cleanup.append(all(row["hooks_remaining"] == 0 for row in values))
    medians = np.median(norms, axis=0)
    finite = bool(np.isfinite(hidden).all() and np.isfinite(medians).all() and np.all(medians > 0))
    median_map = {str(site): float(value) for site, value in zip((15, 31, 47), medians, strict=True)}
    doses = {str(site): float(study.ORIGINAL_S_DOSE * medians[index] / medians[0])
             for index, site in enumerate((15, 31, 47)) if site != 15} if finite else None
    inventory = dict(requests_sha256=digest(requests), historical_binding_sha256=digest(manifest["historical_binding"]),
                     arrays=bindings, cohorts_sha256=digest(report["cohorts"]))
    reference = dict(formula=study.DOSE_FORMULA, original_dose=study.ORIGINAL_S_DOSE, base_branches=288,
                     source_screen_manifest_sha256=study.PRIOR_MANIFEST_SHA256, capture_sha256=digest(inventory),
                     median_record_norm=median_map, doses=doses)
    checks = dict(all288_ordered_requests_unchanged=True, every_cohort_read_only_identity=all(readonly),
                  all_layer15_vectors_match=all(historical_equal), captures_and_medians_finite_positive=finite,
                  hook_cleanup=all(cleanup))
    require(report["checks"] == checks and report["passed"] is all(checks.values())
            and report["medians"] == median_map and report["doses"] == doses
            and report["capture_inventory"] == inventory and report["capture_inventory_sha256"] == digest(inventory)
            and report["dose_reference"] == reference, "Saved dose or reference checks disagree with raw evidence")
    if all(checks.values()):
        study.validate_reference(reference)
    return checks, reference


def recompute_engineering(harness, arrays, requests, *, layer, direction, dose,
                          expected_model, eos_ids, pad_token_id, cap=128):
    import torch

    require(set(arrays) == {"before", "after", "direction_float32"}, "Unexpected engineering arrays")
    effective = np.asarray(direction, dtype=np.float32)
    bindings = {key: engine.array_binding(value) for key, value in arrays.items()}
    require(np.array_equal(arrays["direction_float32"], effective) and harness["arrays"] == bindings
            and harness["arrays_sha256"] == digest(bindings) and harness["layer"] == layer
            and harness["schedule"] == study.SCHEDULE and type(cap) is int and 1 <= cap <= 128
            and harness["cap"] == cap and harness["model"] == expected_model
            and harness["prefill_padding_multiple"] == engine.PREFILL_PADDING_MULTIPLE
            and harness["eos_token_ids"] == sorted(eos_ids) and harness["pad_token_id"] == pad_token_id
            and harness["scientific_admission"] is False and harness["full_cap_certified"] is False
            and harness["batch_size"] == 32 and harness["perturbed_row"] == 0
            and harness["request_sha256"] == digest(requests)
            and harness["source_trial_ids"] == [r["source_trial_id"] for r in requests]
            and harness["seeds"] == [r["seed"] for r in requests] and harness["dose"] == dose
            and harness["effective_direction"] == bindings["direction_float32"]
            and harness["supplied_direction"] == engine.array_binding(np.asarray(direction, dtype=np.float64)),
            "Engineering site, direction, dose, cap or request binding changed")
    conditions = harness["conditions"]
    require(set(conditions) == set(setup.CONDITIONS), "Missing engineering condition")
    require(all(frame["vocabulary_size"] == expected_model["vocabulary_size"]
                for condition in conditions.values() for frame in condition["forwards"]),
            "Engineering full-vocabulary width differs from the pinned model")
    require(all(type(token) is int and 0 <= token < expected_model["vocabulary_size"]
                for condition in conditions.values() for row in condition["rows"]
                for token in row["completion"]["generated_token_ids"]), "Out-of-vocabulary generation")
    zero = dict(layer=layer, position="record", mode="add", direction=effective, dose=0.0)
    patches = dict(baseline=[None] * 32, zero_sham=[dict(zero) for _ in range(32)],
                   one_row_nonzero=[{**zero, "dose": dose}] + [dict(zero) for _ in range(31)],
                   clean_repeat=[None] * 32)
    checks = {f"{name}_timeline": engine.validate_condition(
        value, requests, patches[name], layer=layer, cap=cap, pad_token_id=pad_token_id,
        eos_ids=set(eos_ids)) for name, value in conditions.items()}
    changed, malformed = conditions["one_row_nonzero"], harness["malformed_direction"]
    events = changed["rows"][0]["intervention_trace"]["events"]
    checks.update(
        every_perturbed_event_engaged=all(math.isfinite(e["delta_norm"]) and e["delta_norm"] > 0
            and math.isfinite(e["projection_after"]) and math.isfinite(e["projection_before"])
            and e["projection_after"] > e["projection_before"] for e in events),
        bfloat16_vector_observation=bool(changed["vector_activation_dtypes"])
            and all(dtype == "torch.bfloat16" for dtype in changed["vector_activation_dtypes"]),
        decode_observed_in_every_condition=all(any(f["phase"] == "decode" and any(f["active_prediction_rows"])
                                                  for f in c["forwards"]) for c in conditions.values()),
        nonzero_decode_engagement=any(e["phase"] == "decode" and e["delta_norm"] > 0 for e in events),
        full_sham_identity=engine.identical_rows(conditions["baseline"], conditions["zero_sham"], range(32)),
        unaffected31_identity=engine.identical_rows(conditions["baseline"], changed, range(1, 32)),
        cleanup_repeat_identity=engine.identical_rows(conditions["baseline"], conditions["clean_repeat"], range(32)),
        malformed_rejected_cleanly=malformed["error"] is not None
            and malformed["error"]["type"] == "ValueError"
            and malformed["error"]["message"] == "Invalid sustained direction or dose"
            and not malformed["forwards"] and malformed["attempted_forwards"] == 0 and malformed["hooks_clean"],
        independently_recomputed_geometry=engine.check_geometry(changed, arrays, effective, dose, torch),
        unchanged_requests=harness["request_sha256"] == digest(requests),
        hook_cleanup=all(c["hooks_clean"] and all(r["hooks_remaining"] == 0 for r in c["rows"])
                         for c in conditions.values()) and malformed["hooks_clean"],
    )
    require(harness["checks"] == checks and harness["passed"] is all(checks.values()),
            "Saved engineering checks disagree with the observations")
    require(harness["observed_baseline_max_tokens"]
            == max(len(r["completion"]["generated_token_ids"]) for r in conditions["baseline"]["rows"])
            and harness["observed_nonzero_row_tokens"] == len(changed["rows"][0]["completion"]["generated_token_ids"]),
            "Observed trajectory lengths changed")
    return checks


def source_provenance(root, manifest):
    """Replay the frozen input package before payment or observation auditing.

    No setup execution artifacts, model weights or numerical forwards are read.
    The temporary source tree contains only the explicitly bound archive files.
    """
    from prepare_satisfaction_site_setup import priority_gate, RESERVED
    from curiosity_study.reserved_object_materials import validate_reserved_inputs

    root = Path(root).resolve()
    documents, frozen = frozen_documents(root, manifest)
    with tempfile.TemporaryDirectory(prefix="satisfaction-site-source-audit-") as temporary:
        source_root = Path(temporary)
        for name, payload in documents.items():
            target = source_root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        _, requests, historical, binding, provenance = historical_baseline(source_root, reproduce=False)
        units, candidate = setup_units(source_root)
        priority = priority_gate(source_root)
        reserved = read(source_root / RESERVED)
        validate_reserved_inputs(reserved["selection"], reserved["base_assignments"])
        require(reserved["history_sha256"] == digest(reserved["history"])
                == reserved["selection"]["history_sha256"], "Reserved history receipt changed")
    certificate = manifest["source_certificate"]
    require(requests == manifest["requests"] and np.array_equal(historical, setup.decode_historical(manifest["historical_hidden"]))
            and binding == manifest["historical_binding"] and units == manifest["source_units"]
            and provenance == certificate["provenance"] and candidate == certificate["candidate_source"]
            and priority == certificate["priority"]
            and digest(reserved) == manifest["reserved_materials_sha256"] == certificate["reserved_materials_sha256"],
            "Setup inputs do not reproduce from frozen historical evidence")
    return dict(documents=documents, freeze=frozen, requests=requests, units=units,
                historical=historical, historical_binding=binding, provenance=provenance,
                candidate_source=candidate, priority=priority, reserved_materials=reserved)


def audit(root):
    from transformers import AutoConfig, AutoTokenizer, GenerationConfig

    root = Path(root).resolve()
    manifest = read(root / "manifest.json")
    assignments = setup.validate_manifest(manifest)
    inputs = source_provenance(root, manifest)
    frozen, requests, units = inputs["freeze"], inputs["requests"], inputs["units"]
    binding, priority, reserved = inputs["historical_binding"], inputs["priority"], inputs["reserved_materials"]
    options = dict(revision=setup.MODEL["revision"], local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(setup.MODEL["model_id"], trust_remote_code=False, **options)
    config = AutoConfig.from_pretrained(setup.MODEL["model_id"], **options)
    eos = GenerationConfig.from_pretrained(setup.MODEL["model_id"], **options).eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [] if eos is None else [eos])
    expected_model = dict(model_id=setup.MODEL["model_id"], revision=setup.MODEL["revision"],
                          model_type=config.model_type, hidden_size=config.hidden_size,
                          vocabulary_size=config.vocab_size, layers=config.num_hidden_layers,
                          device="cuda", parameter_dtype="torch.bfloat16")
    coverage = check_collection_coverage(root, expected_manifest_sha256=digest(manifest))
    records = {}
    for path in (root / "cohorts").glob("*/*.json"):
        for record in read(path)["records"]:
            require(record["trial_id"] not in records, "Duplicated setup disposition")
            records[record["trial_id"]] = record
    starts = read(root / "setup_started.json")
    require(starts["manifest_sha256"] == digest(manifest), "Started marker changed")
    attempt = starts["attempt"]
    require({path.parent.name for path in (root / "cohorts").glob("*/*.json")} == {attempt},
            "Setup cohorts do not belong to the sole started attempt")
    summary = read(root / "setup" / attempt / "summary.json")
    require(summary["manifest_sha256"] == digest(manifest) and summary["attempt"] == attempt
            and summary["complete"] is True
            and summary["assigned"] == summary["terminal"] == len(records) == len(assignments) == 544
            and summary["scientific_admission"] is False and summary["full_cap_certified"] is False
            and set(summary["reports"]) == {"reference", "layer31", "layer47"}, "Setup summary differs")
    results, reference = {}, None
    for name in ("reference", "layer31", "layer47"):
        wrapper = read(checked_artifact(root, summary["reports"][name]))
        require(wrapper["manifest_sha256"] == digest(manifest) and wrapper["attempt"] == attempt
                and wrapper["stage"] == name and wrapper["status"] == "measured" and wrapper["error"] is None
                and wrapper["scientific_admission"] is False and wrapper["full_cap_certified"] is False,
                "Setup stage is unavailable or bound to another attempt")
        environment = read(checked_artifact(root, wrapper["environment_artifact"]))
        require(wrapper["environment_artifact"]["path"] == f"environments/{attempt}.json"
                and digest(environment) == wrapper["environment_artifact"]["content_sha256"]
                and all(environment.get(k) == v for k, v in manifest["model"].items())
                and environment["dtype"] == "bfloat16" and environment["attention"] == "eager"
                and environment["device"] == "cuda" and "H200" in environment["gpu"], "Runtime model changed")
        with np.load(checked_artifact(root, wrapper["vector_artifact"]), allow_pickle=False) as saved:
            arrays = {key: saved[key].copy() for key in saved.files}
        require(wrapper["vector_artifact"]["array_bindings"]
                == {key: engine.array_binding(value) for key, value in arrays.items()}, "Saved array bindings changed")
        harness = wrapper["harness"]
        if name == "reference":
            checks, reference = recompute_reference(harness, arrays, manifest, tokenizer=tokenizer)
            rows = [a for a in assignments if a["stage"] == "reference"]
            performance = dict(read_only_capture_forwards=9, read_only_identity_forwards=9,
                               generated_tokens=0, attempted_model_forwards=18)
        else:
            layer = int(name.removeprefix("layer"))
            checks = recompute_engineering(harness, arrays, requests[:32], layer=layer,
                direction=units[str(layer)]["unit"], dose=reference["doses"][str(layer)],
                expected_model=expected_model, eos_ids=eos_ids, pad_token_id=tokenizer.pad_token_id)
            rows = [a for a in assignments if a["site"] == layer]
            counts = {key: value["attempted_forwards"] for key, value in harness["conditions"].items()}
            performance = dict(forwards_by_condition=counts,
                malformed_direction_attempts=harness["malformed_direction"]["attempted_forwards"],
                attempted_model_forwards=sum(counts.values()) + harness["malformed_direction"]["attempted_forwards"],
                generated_tokens=sum(len(row["completion"]["generated_token_ids"])
                                     for condition in harness["conditions"].values() for row in condition["rows"]))
        passed = all(checks.values())
        require(wrapper["passed"] is passed and wrapper["assigned"] == len(rows)
                and wrapper["performance"] == performance
                and type(wrapper["elapsed_seconds"]) in (int, float)
                and math.isfinite(wrapper["elapsed_seconds"]) and wrapper["elapsed_seconds"] >= 0,
                "Stage disposition or physical work accounting changed")
        for assignment in rows:
            record, index = records[assignment["trial_id"]], assignment["source_row"]
            measurement = (harness["cohorts"][index // 32]["measurements"][index % 32]
                           if name == "reference" else harness["conditions"][assignment["condition"]]["rows"][index])
            require(record["status"] == "measured" and record["engineering_passed"] is passed
                    and record["report_artifact"] == summary["reports"][name]
                    and record["activation_artifact"] == wrapper["vector_artifact"]
                    and record["measurement_sha256"] == digest(measurement)
                    and record["scientific_admission"] is False and record["full_cap_certified"] is False
                    and all(record[key] == assignment[key] for key in
                            ("stage", "site", "condition", "source_row", "source_trial_id", "request_sha256")),
                    "Terminal setup row differs from evidence")
        results[name] = dict(passed=passed, checks=checks, report_artifact=summary["reports"][name],
                             vector_artifact=wrapper["vector_artifact"])
    passed = all(value["passed"] for value in results.values())
    require(summary["passed"] is passed and summary["unavailable"] == 0 and summary["dose_error"] is None,
            "Final setup status differs from recomputed evidence")
    return dict(schema_version=1, assay="satisfaction_site_setup_audit", manifest_sha256=digest(manifest),
                source_sha256=manifest["source_sha256"], complete=True, assigned=544, passed=passed,
                scientific_admission=False, full_cap_certified=False, source_freeze=frozen,
                coverage=coverage, historical_binding_sha256=digest(binding), source_units_sha256=digest(units),
                priority_receipt_sha256=digest(priority), reserved_materials_sha256=digest(reserved),
                reserved_history_scope="Bound pre-measurement live-project exposure audit; this checker reconstructs the fixed materials, not the entire historical file/archive census.",
                stages=results, dose_reference=reference, dose_reference_sha256=digest(reference),
                auditor_byte_sha256=sha(__file__),
                scope="Frozen source replay, exact288 historical captures, full-vocabulary read-only identity fingerprints, reference norm arithmetic, short site timelines/sham/cleanup and independent BF16 row0 vector geometry. Full4096 scientific execution remains separately required.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "Setup audit receipt is immutable")
    result = audit(args.run)
    write_once(args.out, result)
    print(json.dumps({key: result[key] for key in ("passed", "assigned", "dose_reference")}))
