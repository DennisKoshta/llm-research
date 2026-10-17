# Informational-novelty diagnosis: completed results

October 14, 2026. Both frozen 288-branch collections completed: 576 original assigned branches from 24 matched worlds per model. The two model sizes, question formats, evidence conditions, and reasoning modes reuse the same underlying worlds. This is a small development diagnosis, not a fresh validation gate or confirmatory study.

## Main observation

Qwen3-32B answered **96/96 constrained questions correctly with thinking enabled**, compared with **46/96 without thinking**. Both description IDs and yes/no judgments were perfect in the thinking condition, with no invalid answers or token-cap hits. This demonstrates the distinction on the tested cases in that inference regime; it does not imply that reasoning is necessary in every possible interface or that performance will generalize automatically.

Qwen3-8B did not show the same complete recovery. Its description accuracy improved with thinking, but failures remained. Many thinking-mode binary responses contained an otherwise unambiguous terminal period, which the frozen exact-response grammar rejects. Those original scores are retained. The new equal-uncertainty study prospectively allows a single terminal period before any outcomes from that new task are observed.

| Model | Format | Thinking | Correct / assigned | Valid / assigned | Cap hits |
|---|---|---|---:|---:|---:|
| Qwen3-8B | Description | No | 20 / 48 | 48 / 48 | 0 |
| Qwen3-8B | Description | Yes | 30 / 48 | 46 / 48 | 2 |
| Qwen3-8B | Yes/no | No | 22 / 48 | 48 / 48 | 0 |
| Qwen3-8B | Yes/no | Yes | 18 / 48 | 21 / 48 | 0 |
| Qwen3-32B | Description | No | 16 / 48 | 41 / 48 | 7 |
| Qwen3-32B | Description | Yes | 48 / 48 | 48 / 48 | 0 |
| Qwen3-32B | Yes/no | No | 30 / 48 | 46 / 48 | 0 |
| Qwen3-32B | Yes/no | Yes | 48 / 48 | 48 / 48 | 0 |

All 192 free-explanation branches produced nonempty final answers with closed thinking where applicable. They were not automatically scored for correctness. In the inspected first-two-index no-thinking examples, the 8B model often discussed the existing observation instead of the offered clue. The 32B model usually identified the offered clue but sometimes described a repeated observation as confirmation that adds certainty. These examples were selected during analysis by taking the first two item indices. They are qualitative illustrations, not prevalence estimates or an independent accuracy endpoint.

## Interpretation and next experiment

The 32B result provides a concrete route to studying the transformation from available evidence to a correct information judgment. The immediate next task also removes two unnecessary ambiguities: it explicitly names the available report and guarantees observation/report reliability. It holds global task uncertainty fixed while varying the report's information gain. These changes were frozen while the original diagnosis was still running, motivated by its completed no-thinking branches. The full 32B thinking result is not used to select a favorable cell from the new experiment.

Thinking mode changes the prompt template and allows up to 2,048 generated tokens; the no-thinking constrained branches allow eight. The comparison identifies this inference-regime bundle, not a pure effect of elapsed computation time. Multiple questions come from the same worlds and must not be treated as 96 independent items. There were no psychological interventions, optional-choice outcomes, or measures of experience in this diagnosis.

## Integrity and cost

Each collection retained all 36 original eight-row cohorts. The audits replayed prompts, sampling settings, scoring, assignments, and cohort membership, verified hashes and physical record counts, and found no scoring discrepancies. An exact-membership guard was tightened in the current scorer after launch; it did not change any saved score in either collection. The archived source preserves the exact executed code.

- [8B audit](qwen8-information-diagnosis-v1-audit.json) and [full cell report](../runs/qwen8-information-diagnosis-v1/report.md)
- [32B audit](qwen32-information-diagnosis-v1-audit.json) and [full cell report](../runs/qwen32-information-diagnosis-v1/report.md)
- [Prospective diagnostic plan](internal-mechanisms-phase-plan.md) and [new task plan](equal-uncertainty-v1-plan.md)
- [Stopped-app cost reconciliation](internal-mechanisms-diagnosis-cost-and-shutdown.json)

The phase-boundary estimate is **$2.1902 of the fresh $300 compute allowance**, leaving **$297.8098**. All diagnostic apps were stopped with zero tasks at that boundary; subsequent validation runs have a separate combined reservation. These are conservative app-wall resource estimates, not invoice totals.
