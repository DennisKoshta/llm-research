# Equal-uncertainty no-thinking validation: completed failures

October 14, 2026. Both frozen collections completed all 4,000 assigned branches, with 100 generated worlds per family, four known/query combinations and five diagnostic questions. Neither model passed the criteria. No optional choices, passive-representation extraction or scientific steering was advanced under this no-thinking instrument.

| Diagnostic | 8B correct / 800 | 8B valid / 800 | 32B correct / 800 | 32B valid / 800 |
|---|---:|---:|---:|---:|
| Recall observed fact | 800 | 800 | 800 | 800 |
| Count remaining candidates | 319 | 792 | 792 | 792 |
| Does the report add a new fact? | 402 | 800 | 635 | 800 |
| Is its fact already observed? | 586 | 800 | 556 | 800 |
| Instrumental information action | 405 | 800 | 400 | 798 |

Aggregate accuracy is descriptive; advancement required every family × diagnostic × known/query cell to pass. Twenty-three of forty 8B cells and thirteen of forty 32B cells failed. All 200 worlds/model and their branches remain in the denominators. An exact allowed response with one optional terminal period was accepted under the prospectively frozen grammar.

The 32B model recalled the supplied facts perfectly and counted the remaining candidates correctly whenever it returned a valid count. It nevertheless gave inconsistent information-novelty judgments and almost always selected the report in the instrumental task, including when the report repeated an already supplied fact. The 8B model had additional counting failures. These observations motivate studying the computation that binds an offered report to the existing knowledge, rather than assuming that correct fact recall implies correct evaluation of the report's information.

The simulator held total task uncertainty at one bit and varied report information between zero and one bit. The audit independently enumerated compatible rows and report outcomes. This establishes the task manipulation; it does not by itself establish equal subjective uncertainty in the model.

The subsequent [reasoning-enabled amendment](reasoning-information-v1-plan.md) uses the regime that succeeded in the earlier matched diagnosis, with fresh data and explicit trace measurements. It does not reinterpret these failures as passes or resume behavioral steering through the failed gate.

## Evidence

- [8B full audit and error inventory](qwen8-equal-uncertainty-v1-audit.json) · [full cell report](../runs/qwen8-equal-uncertainty-v1-validation-final/report.md)
- [32B full audit and error inventory](qwen32-equal-uncertainty-v1-audit.json) · [full cell report](../runs/qwen32-equal-uncertainty-v1-validation-final/report.md)
- [Prospective plan](equal-uncertainty-v1-plan.md) · [stopped-app cost reconciliation](equal-uncertainty-v1-cost-and-shutdown.json)

Both audits verified all 250 original cohorts, prompt/scoring replay, exact assignments, independent oracle enumeration and physical record counts. The 8B run generated 8,049 tokens with eight cap hits; the 32B run generated 8,112 tokens with seven cap hits. At the stopped-app boundary, estimated fresh-phase compute was **$3.6831312**, leaving **$296.3168688** of the fresh $300 allowance. All apps were stopped with zero tasks at that boundary.
