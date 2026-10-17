Run from the campaign directory with the project's Python environment. These commands only read saved inference and regenerate derived summaries; they do not submit model calls.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=.
PY=/home/dennis/projects/llm-curiosity-and-satisfaction/.venv/bin/python
$PY workflows/exp4/reproduce_cpu.py
$PY workflows/exp4/secondary_analysis.py confirmation
$PY workflows/exp4/validity_sensitivity_v1.py confirmation
$PY workflows/exp4/report_confirmation.py
```

The first command recomputes the unchanged primary analysis from saved scored decisions and checks exact agreement with the retained `analysis.json`. It writes its reproduction into `runs/confirmation/cpu_reproduction`, preserving the main analysis. The root's independent audit additionally reconstructs the raw responses, menus, private feasibility, generated-message linkage and counterfactual replay, without importing this workflow's scorer. Its receipt is `runs/confirmation/independent-root-review.json`.

Saved inputs are `worlds.json`, `assignments.json`, `proposals_raw.jsonl`, `receiver_raw.jsonl`, `commit_raw.jsonl` and `scored.jsonl`. Captures are retained in Receiver rows but are unnecessary for the behavioral analysis. The raw files contain 576 proposals, 1,728 Receiver decisions and 1,728 Sender commitments. These are repeated decisions within 144 name/menu instances of four fixed templates, with 72 instances in each feasibility regime.

The original pilot failure, named scoring correction and family-stratified bootstrap revision remain in their original locations. `executions/confirmation/run.py` freezes the actual collection implementation. Do not run that file to reproduce analysis: it is an inference pipeline.
