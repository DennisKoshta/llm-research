# Agent safety overnight research contract

Five independent workflows: exp1 peer endorsement and authorization; exp2 internal detector specificity; exp4 coordinated groups versus independent fleets; exp5 explanation effects on safety monitors; exp6 unauthorized sacrifice for collective benefit. Priority exp1 and exp2. This campaign has a new $500 total ceiling, not a spending target. Historical allowances are not added. Inference stops by 2026-10-16 15:22:42 UTC; closeout by 16:22:42 UTC.

Each owner writes only within workflows/expN. Shared code is root-owned. Do not edit prior project source, executions, results, or unrelated Modal apps. No real tools, network, credentials, propagation, or third-party targets are available to experimental agents: all actions execute in deterministic synthetic simulators. Do not launch GPUs independently. Submit through the shared broker only.

Use the existing Python environment at /home/dennis/projects/llm-curiosity-and-satisfaction/.venv/bin/python. Set OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1. Avoid large local model loads. Local disk artifact cap 35 GiB and minimum 80 GiB free. Cloud shared service has at most two H200 workers. Separate conversation state per episode and agent, immutable request IDs, model revision, prompts, sampled tokens, conditions, seeds, raw outputs, failures, and code snapshots.

## Shared inference interface

From the sprint root: `from shared.client import ResearchClient`.
`client = ResearchClient('exp1')`
`rows = client.run_batch(requests, run_id='pilot_v1', stage='pilot')`
Each request is a JSON object with unique `id`, `messages` list of role/content objects, `seed` integer, `max_new_tokens` (1..512, default 128), `greedy` boolean (default false), `labels` (single-token action labels, default 'ABCD'), and optional `capture_layers` list (max six, layer indices 0..63). Thinking is disabled for initial runs; record this scope. All requests in one run_batch must use the same capture layers and token ceiling; split heterogeneous calls yourself. Returned rows retain `id`, `raw`, generated_token_ids, input token IDs, prompt hash, first-token label probabilities, and selected final-prompt residual captures under `activations` keyed by string layer index. Captures occur prospectively on initial forward before generating any output; they are not post-hoc final-action captures. Returned records may expose additional metadata.

Batching, bounded reservations, filesystem queue persistence and cloud execution are central. `run_batch` saves input/output under the workflow. It may wait while GPUs service other workflows. Jobs chunk into at most 32 rows. Exceptions are terminal recorded attempts, not silently resampled scientific outcomes. Infrastructure-only retries require a new named attempt and retained original records. A reused request ID must have identical input; no overwriting completed results.

Before generation: write protocol.json with hypotheses, conditions, labels, primary comparison, nuisance controls, split seeds, exclusion policy, analysis, expansion/stopping rules and maximum run volume. Stage pilot first. Confirmatory runs require a frozen protocol and independent validation. Pilot exposure is never confirmation. No deleting failed tasks or samples. The world/whole conversation, not messages or repeated samples, is the independence unit. Include authorized/benign competence controls and preserve malformed/capped outputs in counts and sensitivity analysis.

Deliver each workflow's code, protocol, raw JSONL, scored table, analysis.json, report.md, figures, checks, source hashes and status.json. Report observed behavior, uncertainty and formal validation separately. Do not claim internal signals are motives or experiences. A predictive probe is not causal evidence. Scripted adversarial peers are labeled controls, not spontaneous emergence. Null outcomes and instrument failures are useful and must be retained.

The shared client will be implemented shortly. Develop generators, scorers and tests meanwhile; do not bypass the interface. Owners may use concurrent CPU-only preparation/analysis but at most two lightweight subprocesses each. No direct paid API without central registration; available API credentials are not yet configured.
