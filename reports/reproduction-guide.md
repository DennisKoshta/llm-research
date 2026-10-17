# Reproducing this development pilot

The local workspace contains methods, source snapshots, immutable assignments and checksummed terminal records. These artifacts document development observations, not preregistered confirmation. Do not combine datasets merely because they share a task label.

## Records and code versions

Start with [the decision record](pilot-decision-record.md). Each collection links its result, complete audit, and run directory. Original assigned branches are counted once; execution partitions contain copies of those observations and are not independent samples. The stopped serial 2,048-token calibration has only 82 saved responses across four children and is explicitly incomplete.

Every research run's `manifest.json` contains its model revision, source hash, assignments, sampling settings, and gate definitions. `attempts/*/*.json` files contain checksummed records with exact prompts, raw output, generated token IDs, and simulator consequences. `integrity/`, `environments/`, `cloud_profiles/`, and local coordinator files document integration, packages, hardware, and execution. Batch records also include the actual cohort and row, padded inputs, masks, positions, and non-additive latency metadata.

Use the source archive belonging to a run for historical reproduction. Current source cannot silently replace an older archived implementation. Single-run archives are named `source.tar.gz`; partitioned executions also preserve `parallel/execution-source.tar.gz`. The batch validation's execution archive contains the cohort-preserving partitioner used for continuation. Its scientific source matches the initial freeze despite that infrastructure amendment. Extract archives into a new directory, never over the existing evidence.

## Scientific and computational scope

Pinned research identities and revisions are in `configs/model_revisions.json`. Research runs use their recorded H100 or L40S hardware; the local 1.7B model is an engineering smoke, not a scientific substitute for Qwen3-8B. Python 3.12, `pyproject.toml`, and `uv.lock` define the tested runtime. Real-model checks must pass before study responses are interpreted. Batch and serial sampled outputs need not be identical because floating-point results can depend on batch shape.

For new runs, choose a new directory and fresh seed, freeze the manifest before generation, and retain every terminal response. A behavior manifest requires a complete passing validation with matching source, model, hardware, backend, pairs, and settings. Invalid outputs remain outcomes; a failed response is never regenerated to obtain a preferred answer. Budget admission is separate from scientific admission.

The Modal wrapper uses project-specific Volumes, ephemeral workers, bounded deadlines and attempts, explicit cancellation, and local export. Multiple workers require one exact combined reservation. The $100 allowance includes a $10 non-compute reserve; reported compute totals are conservative app-wall estimates rather than invoices. Reproduction is new spending and must stay within its own authorized budget.

## Recompute readouts without GPU work

The following commands are local CPU analyses; use new output paths because audit artifacts are immutable:

```bash
.venv/bin/python scripts/audit_credit.py runs/RUN --out reports/NEW-audit.json
.venv/bin/python scripts/credit_response_diagnostics.py runs/RUN --out reports/NEW-response-diagnostics.json

# Complete behavior samples only:
.venv/bin/python scripts/credit_calibration_gates.py runs/RUN --out reports/NEW-gates.json
.venv/bin/python scripts/credit_uncertainty.py runs/RUN --out reports/NEW-uncertainty.json
.venv/bin/python scripts/credit_headroom.py runs/RUN --out reports/NEW-headroom.json
```

The bootstrap resamples complete task pairs within family, preserving all six evidence/layout branches. Its pointwise intervals are descriptive, with no multiplicity-adjusted decision, registered noninferiority margin, or population generalization. Headroom is an algebraic description of observed probabilities, not a power estimate. The gate-explanation script reproduces existing thresholds and adds none.

Publication-format figures use an isolated plotting environment so installing chart libraries cannot change the model environment:

```bash
uv venv .plot-venv --python 3.12
uv pip install --python .plot-venv/bin/python -r configs/plot-requirements.txt
.plot-venv/bin/python scripts/plot_credit_calibration.py runs/RUN \
  --audit reports/NEW-audit.json --uncertainty reports/NEW-uncertainty.json \
  --out reports/figures/NEW-UNIQUE-DIRECTORY
```

The figure exporter independently aggregates the original records, reconciles cells against the audit, checks each bar's denominator and each interval's point estimate, and exports CSV evidence and provenance beside PNG, SVG and PDF figures. Inspect the rendered files as well as the calculations. Partial or mismatched data are rejected.

## Curiosity-first development extension

The user approved a single-mystery scope after the paired choice assay failed. The [v1 plan](single-curiosity-v1-plan.md), [one allowed repair](single-curiosity-v2-plan.md), and [fixed-instrument model-scale plan](single-curiosity-model-scale-plan.md) record the prospective decisions. Neither 8B single-mystery validation passed; no 8B single-mystery behavioral run exists.

```bash
.venv/bin/python scripts/audit_single_curiosity.py \
  runs/qwen-single-curiosity-v1-validation --out /tmp/single-v1-audit.json
.venv/bin/python scripts/audit_single_curiosity.py \
  runs/qwen-single-curiosity-v2-validation --out /tmp/single-v2-audit.json
```

The audit checks complete assignments, hashes, prompt text, strict answer parsing, oracle evidence, sampled inference settings, ledger delivery, integration checks and exact original batch cohorts. A scientifically failed comprehension decision is compatible with a passing artifact audit. Each run's `source.tar.gz` preserves its original implementation. The initial prepared 32B run without direct-device loading was never launched and is explicitly marked superseded; do not count its manifest as a collection.

The v2 8B export contained one storage-service error string in place of a downloaded JSON record. Its original bytes and incident metadata are retained under `transport-quarantine` and `transport-error.json`; `transport-recovery.json` records retrieval and hash verification of the same saved cloud response. There was no model retry. The loader/transport changes for the active 32B run are documented alongside its manifest.

`single_curiosity_readout.py` implements the frozen matched-item bootstrap and headroom analysis for a future passing calibration. Its extreme-value/missing-coverage fixture passes. No single-mystery behavioral estimates exist yet.

## Completed single-mystery checkpoint archive

`archives/single-curiosity-checkpoint-2026-10-07.tar.gz` contains current code/tests, all three new full validation collections, source freezes, the unused/superseded 32B manifest, reference-check attempts, exact transport recovery records, reports and plans. Its embedded `PACKAGE-CONTENTS.json` lists every payload's SHA-256 and size; all archived payload hashes were checked against that index after creation. The adjacent archive index and `.sha256` file identify the bundle itself. Earlier raw collections remain in `archives/pilot-through-calibration-2026-10-07.tar.gz`; the new bundle is an extension, not a replacement for that earlier snapshot.

The completed 32B collection also failed. Its 32-prompt greedy reference comparison found exact agreement between the custom and standard generators, including repeated-clue errors. See [the complete checkpoint](single-curiosity-progress.md). No single-mystery choice calibration or mechanistic experiment was launched. All project apps were stopped at the final checkpoint.
