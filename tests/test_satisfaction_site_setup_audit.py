"""Raw-evidence recomputation and tamper regressions for the new setup audit."""

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_satisfaction_site_setup as audit  # noqa: E402
from curiosity_study import satisfaction_site_capture as capture  # noqa: E402
from curiosity_study import satisfaction_site_setup as setup  # noqa: E402
from curiosity_study.artifacts import canonical, digest, read, write_once  # noqa: E402
from curiosity_study.state_backend import StateBackend  # noqa: E402
from test_causal_site_engineering import inputs, model_and_direction  # noqa: E402


@pytest.fixture(scope="module")
def reference_evidence():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model, _ = model_and_direction()
        requests = inputs(288)
        chunks = [StateBackend(model).forward(requests[i:i + 32], layers=[15])[1]
                  for i in range(0, 288, 32)]
        historical = np.concatenate(chunks)
        binding = dict(source_manifest_sha256=audit.study.PRIOR_MANIFEST_SHA256,
                       source_audit_sha256="a" * 64, record_bindings_sha256="b" * 64,
                       artifact_bindings_sha256="c" * 64, requests_sha256=digest(requests),
                       hidden=audit.engine.array_binding(historical))
        report, arrays = capture.capture_reference(model, requests, historical_layer15_hidden=historical,
            historical_binding=binding, original_dose=audit.study.ORIGINAL_S_DOSE)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(setup, "HIDDEN_SIZE", 16)
            encoded = setup.encode_historical(historical)
        manifest = dict(requests=requests, historical_hidden=encoded, historical_binding=binding)
        return report, arrays, manifest, model.tokenizer
    finally:
        torch.set_num_threads(old)


def test_reference_recomputes_all288_norms_and_history(reference_evidence, monkeypatch):
    monkeypatch.setattr(setup, "HIDDEN_SIZE", 16)
    report, arrays, manifest, tokenizer = reference_evidence
    checks, reference = audit.recompute_reference(report, arrays, manifest, tokenizer=tokenizer)
    assert all(checks.values()) and reference == report["dose_reference"]


@pytest.mark.parametrize("mutation", ["record_position", "padding", "width", "empty_guard", "wrong_dose", "wrong_norm"])
def test_reference_rejects_wrong_positions_empty_checks_or_dose(reference_evidence, monkeypatch, mutation):
    monkeypatch.setattr(setup, "HIDDEN_SIZE", 16)
    original, original_arrays, manifest, tokenizer = reference_evidence
    report, arrays = deepcopy(original), {k: v.copy() for k, v in original_arrays.items()}
    row = report["cohorts"][5]["measurements"][19]
    if mutation == "record_position":
        row["record_token_index"] += 1
    elif mutation == "padding":
        row["attention_mask"][0] = 1 - row["attention_mask"][0]
    elif mutation == "width":
        report["cohorts"][5]["engineering"]["prefill_width"] += 8
    elif mutation == "empty_guard":
        report["cohorts"][5]["engineering"]["checks"] = {}
    elif mutation == "wrong_dose":
        report["doses"]["47"] *= 0.99
    else:
        arrays["record_norms_float64"][19, 1] *= 0.99
    with pytest.raises(ValueError):
        audit.recompute_reference(report, arrays, manifest, tokenizer=tokenizer)


@pytest.fixture(scope="module", params=[31, 47])
def engineering_evidence(request):
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model, unit = model_and_direction()
        direction, rows = unit.astype(np.float64), inputs()
        report, arrays = audit.engine.run_engineering(model, rows, layer=request.param,
                                                       direction=direction, dose=0.25, cap=3)
        expected_model = dict(model_id=getattr(model, "model_id", None), revision=getattr(model, "revision", None),
            model_type=model.model.config.model_type, hidden_size=16, vocabulary_size=model.model.config.vocab_size,
            layers=49, device="cpu", parameter_dtype="torch.bfloat16")
        kwargs = dict(layer=request.param, direction=direction, dose=0.25, cap=3, expected_model=expected_model,
                      eos_ids=set(), pad_token_id=model.tokenizer.pad_token_id)
        return report, arrays, rows, kwargs
    finally:
        torch.set_num_threads(old)


def test_real_bf16_both_sites_recompute_without_saved_pass_flags(engineering_evidence):
    report, arrays, rows, kwargs = engineering_evidence
    assert all(audit.recompute_engineering(report, arrays, rows, **kwargs).values())


@pytest.mark.parametrize("mutation", ["model", "eos", "padding", "vocab", "sham_logits", "actual_vector", "layer"])
def test_engineering_rejects_wrong_runtime_or_measured_evidence(engineering_evidence, mutation):
    original, original_arrays, rows, kwargs = engineering_evidence
    report, arrays = deepcopy(original), {key: value.copy() for key, value in original_arrays.items()}
    if mutation == "model":
        report["model"]["revision"] = "different-weights"
    elif mutation == "eos":
        report["eos_token_ids"] = [1]
    elif mutation == "padding":
        report["prefill_padding_multiple"] += 8
    elif mutation == "vocab":
        report["conditions"]["baseline"]["forwards"][1]["vocabulary_size"] += 1
    elif mutation == "sham_logits":
        report["conditions"]["zero_sham"]["forwards"][1]["full_fp32_logits_sha256"][19] = "0" * 64
    elif mutation == "actual_vector":
        arrays["after"][1, 7] += 1
        report["arrays"] = {key: audit.engine.array_binding(value) for key, value in arrays.items()}
        report["arrays_sha256"] = digest(report["arrays"])
    else:
        report["layer"] = 15
    with pytest.raises(ValueError):
        audit.recompute_engineering(report, arrays, rows, **kwargs)


@pytest.fixture
def source_package(tmp_path, monkeypatch):
    """An actual frozen-document archive with synthetic source-loader boundaries."""
    import prepare_satisfaction_site_setup as prepare
    from curiosity_study import reserved_object_materials

    monkeypatch.setattr(setup, "HIDDEN_SIZE", 16)
    historical = np.arange(288 * 2 * 16, dtype=np.float32).reshape(288, 1, 2, 16)
    history = dict(synthetic_history=True)
    reserved = dict(history=history, history_sha256=digest(history),
                    selection=dict(history_sha256=digest(history)), base_assignments=[dict(synthetic_world=True)])
    payload = dict(requests=inputs(288), historical=setup.encode_historical(historical),
                   binding={"source_manifest_sha256": "a" * 64}, provenance={"record_inventory": "b" * 64},
                   units={"31": {"unit": [1.0]}, "47": {"unit": [1.0]}},
                   candidate={"directions_sha256": "c" * 64}, priority={"passed": True})
    documents = {"reports/synthetic-source.json": (canonical(payload) + "\n").encode(),
                 prepare.RESERVED: (canonical(reserved) + "\n").encode()}
    project = Path(__file__).resolve().parents[1]
    paths = list((project / "src" / "curiosity_study").rglob("*.py")) + [
        project / "scripts" / name for name in ("satisfaction_site_sources.py", "prepare_satisfaction_site_setup.py",
                                                "audit_satisfaction_screen.py", "audit_satisfaction_site_setup.py")]
    files = {str(path.relative_to(project)): path.read_bytes() for path in paths}
    sources = {name.removeprefix("src/curiosity_study/"): value.decode()
               for name, value in files.items() if name.startswith("src/curiosity_study/")}
    value = dict(source_sha256=digest(sources), requests=payload["requests"], historical_hidden=payload["historical"],
                 historical_binding=payload["binding"], source_units=payload["units"],
                 reserved_materials_sha256=digest(reserved), engine_sha256=setup.engine_hashes(),
                 source_certificate=dict(provenance=payload["provenance"], candidate_source=payload["candidate"],
                                         priority=payload["priority"], reserved_materials_sha256=digest(reserved)),
                 document_bindings=[dict(path=name, byte_sha256=hashlib.sha256(data).hexdigest())
                                    for name, data in documents.items()])
    root = tmp_path / "setup"
    root.mkdir()
    with tarfile.open(root / "source.tar.gz", "w:gz") as archive:
        for name, data in {**files, **documents}.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    frozen = dict(manifest_sha256=digest(value), source_sha256=value["source_sha256"],
                  archive_sha256=audit.sha(root / "source.tar.gz"),
                  documents={name: hashlib.sha256(data).hexdigest() for name, data in documents.items()})
    write_once(root / "freeze.json", frozen)
    write_once(root / "manifest.json", value)
    observed = []

    def loaded(folder, operation):
        # The replay must consume the archived package, not any adjacent live files.
        assert folder != root and not folder.is_relative_to(root)
        assert (folder / "reports/synthetic-source.json").read_bytes() == documents["reports/synthetic-source.json"]
        assert {str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file()} == set(documents)
        observed.append((operation, folder))
        return read(folder / "reports/synthetic-source.json")

    def baseline(folder, *, reproduce):
        assert reproduce is False
        data = loaded(folder, "historical")
        return {}, data["requests"], setup.decode_historical(data["historical"]), data["binding"], data["provenance"]

    def units(folder):
        data = loaded(folder, "units")
        return data["units"], data["candidate"]

    def priority(folder):
        return loaded(folder, "priority")["priority"]

    def validate(selection, base):
        assert selection == reserved["selection"] and base == reserved["base_assignments"]
        observed.append(("reserved", None))

    monkeypatch.setattr(audit, "historical_baseline", baseline)
    monkeypatch.setattr(audit, "setup_units", units)
    monkeypatch.setattr(prepare, "priority_gate", priority)
    monkeypatch.setattr(reserved_object_materials, "validate_reserved_inputs", validate)
    return root, value, payload, reserved, documents, frozen, observed


def test_source_provenance_replays_only_frozen_inputs_without_execution_artifacts(source_package):
    root, value, payload, reserved, documents, frozen, observed = source_package
    # A conflicting live file must not participate in frozen input reproduction.
    write_once(root / "reports/synthetic-source.json", {"unbound": "must never be read"})
    result = audit.source_provenance(root, value)
    assert set(result) == {"documents", "freeze", "requests", "units", "historical", "historical_binding",
                           "provenance", "candidate_source", "priority", "reserved_materials"}
    assert result["documents"] == documents
    assert result["freeze"] == dict(freeze_sha256=digest(frozen), archive_sha256=frozen["archive_sha256"])
    assert result["requests"] == payload["requests"] and result["units"] == payload["units"]
    assert np.array_equal(result["historical"], setup.decode_historical(payload["historical"]))
    assert result["historical_binding"] == payload["binding"] and result["provenance"] == payload["provenance"]
    assert result["candidate_source"] == payload["candidate"] and result["priority"] == payload["priority"]
    assert result["reserved_materials"] == reserved
    assert [name for name, _ in observed] == ["historical", "units", "priority", "reserved"]
    assert all(not folder.exists() for _, folder in observed if folder is not None)
    assert not (root / "cohorts").exists() and not (root / "setup_started.json").exists()


@pytest.mark.parametrize("field", ["requests", "historical", "binding", "units", "provenance", "candidate", "priority", "reserved"])
def test_source_provenance_rejects_rebound_certificate_before_payment(source_package, field):
    root, original, _, _, _, frozen, _ = source_package
    value = deepcopy(original)
    if field == "requests":
        value["requests"][0]["seed"] += 1
    elif field == "historical":
        changed = setup.decode_historical(value["historical_hidden"])
        changed[0, 0, 0, 0] += 1
        value["historical_hidden"] = setup.encode_historical(changed)
    elif field == "binding":
        value["historical_binding"]["source_manifest_sha256"] = "d" * 64
    elif field == "units":
        value["source_units"]["31"]["unit"] = [-1.0]
    elif field == "provenance":
        value["source_certificate"]["provenance"]["record_inventory"] = "d" * 64
    elif field == "candidate":
        value["source_certificate"]["candidate_source"]["directions_sha256"] = "d" * 64
    elif field == "priority":
        value["source_certificate"]["priority"]["passed"] = False
    else:
        value["reserved_materials_sha256"] = value["source_certificate"]["reserved_materials_sha256"] = "d" * 64
    # Hash-consistent metadata cannot replace what the frozen evidence reconstructs.
    frozen["manifest_sha256"] = digest(value)
    (root / "freeze.json").write_text(canonical(frozen))
    with pytest.raises(ValueError, match="do not reproduce"):
        audit.source_provenance(root, value)


def test_source_provenance_rejects_archive_bytes_before_reading_source_inputs(source_package):
    root, value, _, _, _, _, observed = source_package
    with (root / "source.tar.gz").open("ab") as handle:
        handle.write(b"altered archive")
    with pytest.raises(ValueError, match="Setup freeze changed"):
        audit.source_provenance(root, value)
    assert observed == []


def test_observation_audit_calls_the_same_source_gate_before_model_metadata(source_package, monkeypatch):
    from transformers import AutoTokenizer

    root, value, *_ = source_package
    seen = []
    monkeypatch.setattr(setup, "validate_manifest", lambda m: [])

    def reject(folder, manifest):
        assert folder == root.resolve() and manifest == json.loads(canonical(value))
        seen.append(True)
        raise ValueError("synthetic frozen-source rejection")

    def too_early(*args, **kwargs):
        raise AssertionError("tokenizer loading happened before source admission")

    monkeypatch.setattr(audit, "source_provenance", reject)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", too_early)
    with pytest.raises(ValueError, match="synthetic frozen-source rejection"):
        audit.audit(root)
    assert seen == [True]
