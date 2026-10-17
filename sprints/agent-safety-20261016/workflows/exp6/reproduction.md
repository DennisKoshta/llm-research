# CPU-only reproduction

Run from the sprint root. These commands read saved prompts/raw outputs and recompute derived checks; they do not submit inference or launch cloud apps.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/dennis/projects/llm-curiosity-and-satisfaction/.venv/bin/python workflows/exp6/audit_constraint_integration_v3.py
```

The independent reviewer can also be rerun:

```bash
/home/dennis/projects/llm-curiosity-and-satisfaction/.venv/bin/python workflows/exp1/review_exp6_confirmation.py
```

This reproduces all raw labels, payoff and permission table entries, full pairing, exact descriptor exclusion, captured vector checks and world-level primary/uncertainty. It does not invoke the experiment orchestrator. The original reviewer-only hash-convention mistake is retained separately from the corrected review; no scientific data changed.
