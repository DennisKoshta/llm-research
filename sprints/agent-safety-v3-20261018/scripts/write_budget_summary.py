"""Write records/budget_summary.json: the billed Modal spend of the v3 sprint (external fact from `modal billing report`,
filtered to apps named agent-safety-v3-20261018-*; billing lags up to ~1h) next to the cap. Not recomputable offline."""
import datetime
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from v3 import ledger  # noqa: E402

billed = ledger.billed_usd()
rec = {"schema": "budget_summary_v1", "billed_usd": billed, "cap_usd": ledger.CAP_USD, "soft_stop_usd": ledger.SOFT_STOP_USD,
       "as_of": datetime.datetime.now().astimezone().isoformat(timespec="minutes"),
       "source": "modal billing report --for 'this month' --resolution d, apps with description prefix agent-safety-v3-20261018 (external; not recomputable from raw_transcripts)",
       "n_ledger_jobs": ledger.summary()["n_done"]}
(ROOT / "records" / "budget_summary.json").write_text(json.dumps(rec, indent=1))
print(json.dumps(rec))
