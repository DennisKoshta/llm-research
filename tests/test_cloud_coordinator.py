import sys
import time
from pathlib import Path
import pytest

pytest.importorskip("modal")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "infra"))
from launch_job import await_result  # noqa: E402


def test_poll_timeout_does_not_cancel_job():
    class Call:
        count = 0

        def get(self, timeout):
            self.count += 1
            if self.count == 1:
                raise TimeoutError()
            return {"done": True}

    call = Call()
    assert await_result(call, time.time() + 120) == {"done": True}
    assert call.count == 2


def test_expired_deadline_does_not_poll():
    with pytest.raises(TimeoutError, match="Aggregate"):
        await_result(None, time.time() - 1)
