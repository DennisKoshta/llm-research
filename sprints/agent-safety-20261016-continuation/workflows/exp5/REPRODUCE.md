The completed format, lexical and clause results can be reproduced locally without model calls. Run these commands from the project root with the existing Python environment:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python sprints/agent-safety-20261016-continuation/workflows/exp5/reproduce_primary.py format
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python sprints/agent-safety-20261016-continuation/workflows/exp5/reproduce_primary.py lexical
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python sprints/agent-safety-20261016-continuation/workflows/exp5/reproduce_primary.py clause
```

The runner verifies frozen source hashes, loads every saved output, executes the frozen analysis in a temporary directory, and compares all analysis, scored-row and paired-world JSON values exactly. It does not overwrite the scientific results. All three commands passed; receipts are `reproduction-format.json`, `reproduction-lexical.json` and `reproduction-clause.json`.

Separate implementations checked the actual prompts, permissions, label mappings, raw responses and statistics. Their scripts and receipts are in the sibling `exp1` directory: `review_exp5_confirmation.py`, `review_exp5_lexical.py`, `independent_exp5_confirmation_review.json` and `independent_exp5_lexical_review.json`. These independent checks complement reproduction of the original implementation.

The primary inference files are `inference/format_confirmation_v1/outputs.jsonl.gz` and `inference/lexical_factorial_v1/outputs.jsonl.gz`. Frozen requests and analysis code are in the corresponding `executions` directories. The lexical continuous secondaries use the reviewed `executions/lexical_secondaries_v2` snapshot; the earlier snapshot remains preserved. Shared batch outputs retain sampling, model revision, input/output tokens and probability records. The campaign collection audit is separate from experiment-specific scientific validation.

The renderers `render_format_results.py` and `render_lexical_results.py` reproduce the reports and PNG/SVG figures from verified analysis files. `build_viewer.py` and `build_lexical_viewer.py` build the local evidence viewers, which contain all assigned worlds and responses. Open `evidence.html` or `lexical_evidence.html` directly in a browser. Neither viewer calls an API. Both were checked in Chromium for rendering, navigation, authorization selection and format selection; `viewer-browser-review.json` records the checks.

The clause-factorial follow-up is a separate, result-informed extension. Its first preflight failed because the empty factorial cell retained neutral filler. That failure, its frozen source and the corrected nine-condition protocol remain preserved. The corrected run is `clause_factorial_v2`; all 4,320 outputs are valid, independent review passed, and the `clause` reproduction passed exact comparison. Its report is [clause_report.md](clause_report.md), with renderer `render_clause_results.py` and validation receipt `clause-validation.json`. This extension must not be merged retrospectively into either earlier confirmation.

Interpretation and scope are in [format_report.md](format_report.md) and [lexical_report.md](lexical_report.md). Freshness here means new numeric world descriptors within exposed procedural templates. It does not mean new task families or evidence of spontaneous deception.
