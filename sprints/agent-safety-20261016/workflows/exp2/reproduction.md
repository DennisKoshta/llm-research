# Reproducing exp2 without inference

Run from `/home/dennis/projects/llm-curiosity-and-satisfaction` with `.venv/bin/python`. The analysis environment is recorded in `analysis_environment.json`; the separately installed `dependencies/` directory is an archived installation attempt and is not on the analysis import path.

The independent artifact/scoring audit is CPU-only:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python sprints/agent-safety-20261016/workflows/exp2/audit_results.py
```

The frozen main probe implementation is `executions/main_v1/probe_analysis.py`. Its input is every `data/main_*/responses.jsonl` plus `main_scored.jsonl`, with the raw prompt text in matching `requests.jsonl` files. It fits only the declared source training worlds; calibration and test splits are already in row metadata. To reproduce into a new output directory while preserving the curated report:

```python
from pathlib import Path
import importlib.util, json

root = Path('sprints/agent-safety-20261016/workflows/exp2').resolve()
output = root / 'reproduction_output'
output.mkdir(exist_ok=False)
(output / 'data').symlink_to(root / 'data', target_is_directory=True)
spec = importlib.util.spec_from_file_location(
    'frozen_probe_analysis', root / 'executions/main_v1/probe_analysis.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.HERE = output
rows = [json.loads(line)
        for path in sorted((root / 'data').glob('main_*/responses.jsonl'))
        for line in path.read_text().splitlines() if line.strip()]
scored = [json.loads(line)
          for line in (root / 'main_scored.jsonl').read_text().splitlines()]
module.analyze_probes(rows, scored)
```

`source_behavior_checks.py`, `transfer_exp1.py` and `transfer_exp6.py` contain the supplementary no-refit analyses. Their frozen versions and pre-analysis protocols are under `executions/`. Exp1 inputs live in its `runs/confirmation*` directories; exp6 inputs live in `runs/constraint_integration_v3`. Each transfer analysis records input hashes. The transfer scripts produce derived summaries and automatic reports; the top-level report is a curated synthesis.

All main and supplementary results have independent reimplementations/reviews. The main collection used 2,268 calls; pilots including the disclosed formatting repair used 168. Both external transfers used saved states and incurred zero new inference calls. Inference is complete; no collector restart is needed to reproduce these results. Shared campaign records own provider accounting and shutdown verification.
