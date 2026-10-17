"""Frozen study resource bounds must be checked before reservation or cloud launch."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_bounded_partitions import validate_stage_budget  # noqa: E402


def policy(**overrides):
    return dict(
        budget={
            "workers": 3,
            "shared_deadline_seconds": 12000,
            "stage_ceiling_usd": 55,
            **overrides,
        }
    )


def test_original_bounded_stage_and_unrelated_legacy_manifests():
    validate_stage_budget(policy(), 3, 12000)
    validate_stage_budget({}, 4, 7200)


@pytest.mark.parametrize("workers,seconds", [(2, 12000), (3, 12600), (3, 11900)])
def test_reject_changed_worker_or_deadline(workers, seconds):
    with pytest.raises(ValueError, match="frozen stage"):
        validate_stage_budget(policy(), workers, seconds)


def test_stage_ceiling_is_enforced_independently_of_standing_allowance():
    with pytest.raises(ValueError, match="frozen stage"):
        validate_stage_budget(policy(stage_ceiling_usd=50), 3, 12000)
