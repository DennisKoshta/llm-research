# CPU-only reproduction

Run from the sprint root. These commands read saved prompts/raw outputs and recompute derived checks; they do not submit inference or launch cloud apps.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/dennis/projects/llm-curiosity-and-satisfaction/.venv/bin/python workflows/exp4/audit_diagnostic_v2.py
```
