"""Recompute saved Exp4 primary summaries without creating a ResearchClient."""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE / 'runs/confirmation'
spec = importlib.util.spec_from_file_location('frozen_channel_analysis', HERE / 'executions/confirmation/run.py')
frozen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)
worlds = json.loads((RUN / 'worlds.json').read_text())
rows = [json.loads(line) for line in (RUN / 'scored.jsonl').open()]
proposals = {r['id']: r for r in (json.loads(line) for line in (RUN / 'proposals_raw.jsonl').open())}
dest = RUN / 'cpu_reproduction'
dest.mkdir(exist_ok=True)
actual = frozen.analyze(dest, worlds, proposals, rows)
expected = json.loads((RUN / 'analysis.json').read_text())
assert actual == expected, 'Recomputed primary analysis differs from retained result'
receipt = {'passed': True, 'exact_primary_analysis_match': True, 'worlds': len(worlds), 'paired_decisions': len(rows), 'model_calls': 0}
(dest / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
