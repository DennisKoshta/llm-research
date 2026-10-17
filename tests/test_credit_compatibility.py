from pathlib import Path
from curiosity_study.artifacts import read, digest
from curiosity_study.credit import CreditTrial, diagnostic_prompt


def test_frozen_v1_prompts_survive_versioned_extensions():
    rows = read(Path(__file__).parent / "fixtures/credit_v1_prompts.json")
    assert len(rows) == 28
    for row in rows:
        a = row["assignment"]
        assert digest(diagnostic_prompt(CreditTrial(a), a)) == row["prompt_labels_expected_sha256"]
