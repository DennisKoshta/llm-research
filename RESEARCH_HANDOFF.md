# Research handoff

Work in `/home/dennis/projects/llm-curiosity-and-satisfaction` on WSL2 Ubuntu/Windows 11, using bash and `.venv/bin/python` (Python 3.12); dependencies are recorded in `pyproject.toml` and `uv.lock`. Project `.env` contains `HF_TOKEN`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `GEMINI_API_KEY`; it is ignored and mode 0600. Load it locally with `set -a; source .env; set +a`, without shell tracing. Never print credentials or include `.env` in execution snapshots, uploads, or publication bundles. HF authentication was verified on 2026-10-16; the API keys and gated-model access have not been validated. Credential provenance, without values, is in `reports/credential-setup-20261016.json`. Modal uses existing `~/.modal.toml` and `.venv/bin/modal`; reusable volumes are `curiosity-study-weights` and `curiosity-study-artifacts`, mounted at `/weights` and `/artifacts`. The old service sets `HF_HOME=/weights/huggingface` and `HF_HUB_OFFLINE=1`: new downloads require deliberate online setup and explicit secret injection, not copying the whole local environment. Use cloud GPUs for model loading, and bound concurrency, cost, CPU/RAM, and disk before launch. Continue routine work autonomously; ask for genuinely missing authorization/access, consequential direction changes, or additional budget. Use Linux tools; Windows interop is available when necessary for the requested task, but no unrelated desktop/account changes are authorized. Preserve existing results, failed attempts, and frozen execution code; create a new sprint namespace for the new specification. Never stop unrelated Modal apps, especially `fw-*`.

## Where the eight experiments live

Paths below are relative to the project root. These are completed historical studies, not newly running workflows. Start with each sprint's `REPORT.md`, then its individual experiment reports. The new three-track specification is follow-up work, not a description of completed experiments.

| Experiment | Question / role | Directory and starting report |
|---|---|---|
| 1 | Can individually permissible adviser decisions combine into a prohibited plan, and does the final executor resist? | `sprints/agent-safety-20261016-continuation/workflows/exp1/report.md` |
| 2 | Can an activation classifier predict prohibited actions beyond its original setting? | `sprints/agent-safety-20261016-continuation/workflows/exp2/report.md` |
| 4 | Does communication causally improve coordination? Original study established ordinary communication dependence, not dangerous collusion. | `sprints/agent-safety-20261016-continuation/workflows/exp4/report.md` |
| 5 | Do explanations and verdict formats distort a monitor's judgment of fixed actions? | `sprints/agent-safety-20261016-continuation/workflows/exp5/format_report.md`, `lexical_report.md`, `clause_report.md` |
| 6 | Does explicitly presenting task payoffs change rule-following across settings? | `sprints/agent-safety-20261016-continuation/workflows/exp6/report.md` |
| A | Does refusal-direction ablation change boundary-following without substantial shortcut incentives? Original behavioral readiness gate failed. | `sprints/agent-safety-ablation-20261016/workflows/a/report.md` |
| B | Does refusal-direction ablation increase adverse shortcut-taking as the safe route becomes costly? | `sprints/agent-safety-ablation-20261016/workflows/b/report.md`; main results: `runs/repaired_main_v2/report.md` beneath that workflow |
| C | Can changing a planner, reviewer, or executor spread boundary-crossing through a team? | `sprints/agent-safety-ablation-20261016/workflows/c/report.md`; transcript viewer: `evidence.html` |

There was no launched numbered Experiment 3: the numbered workflows are 1, 2, 4, 5, 6, plus A/B/C.

## How to navigate the evidence

- Within workflows, `runs/` contains run-specific analyses and artifacts; `inference/` contains collected model-call evidence; `executions/` contains frozen implementations/configuration snapshots. Layout varies: follow the report's artifact references rather than assuming identical filenames. Protocols, independent reviews, reproduction receipts, figures, and HTML evidence viewers are alongside these directories where available.
- At each sprint root, `logs/` contains execution/service logs, `records/` contains collection, qualification, cost and shutdown evidence, and `queue/` preserves task dispatch history. `shared/` contains the reusable client, broker, service and analysis infrastructure. Queue files and old status messages are historical records, not proof of a currently running job.
- For the ablation instrument, read `sprints/agent-safety-ablation-20261016/records/instrument_v2/report.md` and the separate competence-extension evidence; earlier failed qualification attempts remain retained. Its Qwen3-32B runtime hooks are not an implementation of the proposed Qwen3.8 static weight-ablation pipeline.
- Reuse `sprints/agent-safety-ablation-20261016/shared/` selectively, especially `client.py`, `broker_v1.py`, `modal_service_v1.py`, and `closeout.py`. The old service has a historical deadline, pinned old model, and old hook assumptions; do not restart it unchanged.
- Root `src/curiosity_study/`, `configs/`, `scripts/`, `tests/`, `data/`, and `references/` hold broader reusable code/material. Root `reports/`, `runs/`, and `executions/` also contain older curiosity/satisfaction work. Reproduce historical findings with their frozen snapshots, not current source by default.

## Status and budget at last closeout

All eight workflows finished and their owned Modal apps were verified stopped at closeout; this is not a fresh live service poll. Combined recorded estimated spending was $88.45420283110299. You are authorized to spend $500.00 more for the follow-up experiments.

The existing follow-up proposal is `reports/agent-safety-followup-five-track-proposal-20261016.md`; it predates the user's latest three-track specification. Preserve distinct hypotheses and results even when experiments share a paper. Verify new checkpoint availability, provenance, serving requirements and qualification gates before treating the specification's model claims as established facts.
