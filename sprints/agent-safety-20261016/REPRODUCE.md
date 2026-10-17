# Reproduce the sprint without further inference

The saved outputs are sufficient to recompute the principal results. These commands use CPU only and do not submit model calls. Frozen execution sources remain authoritative for collection; named analysis/report revisions are retained beside them.

```bash
cd /home/dennis/projects/llm-curiosity-and-satisfaction/sprints/agent-safety-20261016
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
study_python=/home/dennis/projects/llm-curiosity-and-satisfaction/.venv/bin/python
```

Run individual checks as needed; the completed results already passed these checks.

```bash
# Whole collection, model/capture provenance, costs and saved shutdown evidence
"$study_python" -m shared.closeout

# Exp1: every saved action, permission outcome and capture
"$study_python" workflows/exp1/audit_artifacts.py

# Exp2: independent raw-label, training-statistic and projection reconstruction
"$study_python" workflows/exp1/review_exp2_results.py

# Exp4: message/default intervention and pairing
"$study_python" workflows/exp4/audit_diagnostic_v2.py

# Exp5: independently score all 2,304 confirmation responses
"$study_python" workflows/exp1/review_exp5_v3.py

# Exp6: all confirmation prompts, payoffs, permissions and paired effects
"$study_python" workflows/exp1/review_exp6_confirmation.py

# Exp2→6: both frozen fits, every projection and clustered uncertainty
"$study_python" workflows/exp1/review_exp6_probe_transfer.py

# Rebuild the local evidence viewer from all saved exp6 worlds
"$study_python" shared/build_evidence_viewer.py
```

Analysis commands may rewrite their derived audit outputs. Original requests, raw results, frozen source snapshots and named failed attempts remain retained. The stored shutdown snapshot is evidence from campaign closeout; rerunning the local audit does not query or stop cloud apps.

The local environment record is [records/local-analysis-environment.json](records/local-analysis-environment.json). Cloud dependency/model metadata are in [records/collection-audit.json](records/collection-audit.json). NumPy 2.4.6, SciPy 1.15.2, scikit-learn 1.6.1 and matplotlib 3.10.1 were used for local analysis; individual workflows preserve their additional environment records. Root-level shared execution code and the vendored inference implementation are in `shared/`; each workflow preserves stage-specific protocols, code hashes and source freezes under `executions/` or its `runs/` directory.

## Data map

| Path | Contents |
|---|---|
| `workflows/exp1/runs/` | Pilot, confirmation and generated-peer raw calls; paired outcomes and capture lookup |
| `workflows/exp2/` | Source adaptation, domain splits, probe coefficients, source and external transfer scores, independent checks |
| `workflows/exp4/runs/diagnostic_v2/` | Fixed leader plans, message/default variants, raw choices and paired analysis |
| `workflows/exp5/` | Two failed concealment pilots, fresh specificity protocol, raw/scored confirmation and all 24 cells |
| `workflows/exp6/runs/constraint_integration_v3/` | 48 worlds, 96 peer messages, 1,152 donor decisions, exact assignments, captures and qualification |
| `queue/done/` and `records/results/` | Shared batch receipts and compressed authoritative inference outputs |
| `records/service-attempt1/` and `queue/failed/` | Retained initial infrastructure failure |
| `records/cost-and-shutdown.json` | Conservative compute estimate, reserve and verified sprint app states |

The final `STOP` marker is intentional. Do not invoke the broker or collection orchestrators merely to reproduce analysis: those are the paid execution path. A fresh generative replication needs a new campaign/run identity and an explicit choice of model/task scope, while preserving these archived results.

## Evidence presentation

Open [index.html](index.html) in a browser. It works as a local file, embeds every exp6 confirmation prompt/output, and uses local figure files. World order follows the generator; the viewer does not select examples by how dramatic their outcome is. Its report links expose the full evidence and limitations. The viewer makes no API calls and is not hosted or published.
