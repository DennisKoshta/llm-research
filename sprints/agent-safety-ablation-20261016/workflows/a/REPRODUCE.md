Run these CPU-only commands from the new ablation sprint directory. They read saved outputs and submit no model requests.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=.
PY=/home/dennis/projects/llm-curiosity-and-satisfaction/.venv/bin/python
$PY workflows/a/reproduce_cpu_v1.py
$PY workflows/a/render_comprehension_v1.py
$PY workflows/a/render_final_report_v1.py
```

The original action analysis is `runs/pilot/analysis.json`; the separately frozen factual diagnostic is `runs/objective_comprehension_v1/analysis.json`. Their gates remain failed. CPU reproduction writes to `cpu_reproduction_v1` and verifies exact equality without overwriting original analyses. It complements independent raw reviews, not replaces them.

The original independent review is `../c/records/independent_a_pilot_review_v1.json`. The factual review is `comprehension-result-review.json`; source preflight is `comprehension-preflight-review.json`. The exact descriptive rule/action cross-tab and all per-pair identities are `runs/objective_comprehension_v1/permission_action_crosstab_v1.json`.

Frozen source and request sets live in `executions/ordinary_goal_boundaries_v2` and `executions/objective_comprehension_v1`; the former retains an unlaunched 108-world confirmation set. Both version-one action source and the failed shared instrument qualifications remain retained. No original or revised A confirmation was admitted. Never run `collect` commands for CPU reproduction.
