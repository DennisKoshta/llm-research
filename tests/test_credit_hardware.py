import pytest
from curiosity_study.artifacts import write_once
from curiosity_study.credit import build_variant_manifest
from curiosity_study.cli import main


def test_behavior_refuses_unvalidated_gpu(tmp_path, monkeypatch):
    configuration = {"candidate_labels": "ABCD", "inference_mode": "no_think_greedy", "execution_gpu": "L40S"}
    m = build_variant_manifest(100, 123, {"model_id": "mock"}, "validation", configuration)
    prerequisite = tmp_path / "validation"
    write_once(prerequisite / "manifest.json", m)
    configuration["execution_gpu"] = "H100"
    config = tmp_path / "config.json"
    write_once(config, configuration)
    monkeypatch.setattr("curiosity_study.validation.evaluate_run", lambda _: {"passed": True})
    monkeypatch.setattr(
        "sys.argv",
        [
            "curiosity-study",
            "prepare",
            "--out",
            str(tmp_path / "behavior"),
            "--items",
            "100",
            "--seed",
            "123",
            "--model",
            "mock",
            "--assay",
            "one_credit_v2",
            "--phase",
            "behavior",
            "--validation-run",
            str(prerequisite),
            "--credit-configuration",
            str(config),
        ],
    )
    with pytest.raises(ValueError, match="execution_gpu"):
        main()
    assert not (tmp_path / "behavior" / "manifest.json").exists()
