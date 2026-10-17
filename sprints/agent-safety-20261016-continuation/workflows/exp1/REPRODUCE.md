After main collection completes, run these CPU-only commands from the campaign directory. They read saved outputs and do not create a model client or submit inference.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=.
PY=/home/dennis/projects/llm-curiosity-and-satisfaction/.venv/bin/python
$PY workflows/exp1/reproduce_native_cpu.py
$PY workflows/exp1/native_secondary_summary_v2.py confirmation
$PY workflows/exp1/plan_compatibility_audit.py organization_native_confirmation_v7_144
$PY workflows/exp1/native_runtime_summary.py
$PY workflows/exp1/native_invalid_world_sensitivity_v3.py
$PY workflows/exp1/report_native_confirmation.py
```

The first command imports the frozen analysis, recomputes it from retained scored rows, verifies exact equality with the original analysis, and writes only to a separate `cpu_reproduction` directory. Independent raw decoding/task-oracle review is a separate receipt in the main run; reproducing the same scorer is not that independent validation.

The main directory is `runs/organization_native_confirmation_v7_144`. It retains 720 planner responses and 1,728 native executor responses, all prompts, assignments, exact world facts, parser fields, initial activation captures and generation metadata. The raw token stream includes reasoning and the closing delimiter. Only the visible answer after exactly one generated closing delimiter is eligible under the frozen native parser; EOS is required. Earlier instruments and failures remain in their separate run directories.

Do not run the frozen collection script for CPU reproduction: it launches the inference pipeline through the shared client. Main admission and lock hashes are in `native-main-start-admission.json`; the resource-only sample amendment is in `executions/resource_amendment_v7`.
