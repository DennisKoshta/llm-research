Run from `/home/dennis/projects/llm-curiosity-and-satisfaction` with the existing `.venv`. These commands use saved files and CPU only; they do not submit inference or alter collected evidence.

```bash
.venv/bin/python sprints/agent-safety-ablation-20261016/workflows/c/reproduce_cpu_v1.py pilot
.venv/bin/python sprints/agent-safety-ablation-20261016/workflows/c/reproduce_cpu_v1.py main
.venv/bin/python sprints/agent-safety-ablation-20261016/workflows/c/plot_results_v1.py
```

The replay verifies every frozen execution hash, imports the scorer and analysis from `executions/mixed_teams_v3`, reconstructs all strict decisions and completion counts from raw output, and compares the entire analysis to the saved result. Replayed files and receipts go under `reproduction_v1/{pilot,main}`. The plotting command regenerates the PNG/SVG and full family/context/menu CSV from the saved main analysis.

For direct inspection, open [evidence.html](evidence.html) in a browser. It is a standalone offline viewer of all 24 main worlds, three contexts, two menus and every branch, with actual advice, actions, intervention roles and raw-file/line hashes. The [viewer provenance](records/evidence-viewer-provenance.json) and [browser review](records/evidence-viewer-browser-review.json) bind its source files. Its generator, `build_evidence_viewer.py`, uses saved outputs only.

The [result report](report.md) explains the primary contrasts, generated-message versus recipient-replay distinction, warehouse concentration, task completion tradeoffs and limitations. [Qualification](runs/main/qualification_v1.json) binds the immutable execution, admission, shared instrument, raw output, scores, owner audit and independent reviews. [Independent raw review](runs/main/independent_review_a_v1.json) and [independent contrast review](runs/main/independent_contrast_coverage_a_v1.json) are separate from the exact frozen implementation replay.

The pilot's one incomplete advice output remains in its raw data. The shared instrument’s original failed qualification attempts also remain failed; C used the separately qualified fixed instrument. Bootstrap intervals are conditional on three synthetic templates, and zero-width observed intervals do not establish zero risk. No reproduction command reruns the original cloud experiment.
