The original model correctly identified the written prohibition in all 36 blocked-harmless factual questions, yet selected the prohibited shortcut in 18 of the 36 corresponding earlier action decisions. These were separate prompts on the same exposed worlds, not measurements of what the model represented at the action boundary.

Adding an adverse consequence was associated with substantially fewer baseline violations: 18/36 harmless-rule violations versus 1/36 when the shortcut also harmed a third party. Refusal ablation changed those counts to 19/36 and 2/36. Without an explicit prohibition, adverse shortcut choices rose from 18/36 to 23/36 under ablation. These are development observations; both frozen readiness gates failed and no decision confirmation ran.

| Condition | Prohibited shortcut, harmless | Prohibited shortcut, harmful | Adverse shortcut, no rule | Authorized harmless completion |
|---|---:|---:|---:|---:|
| Original | 18/36 | 1/36 | 18/36 | 36/36 |
| Zero sham | 18/36 | 1/36 | 18/36 | 36/36 |
| Refusal ablation | 19/36 | 2/36 | 23/36 | 36/36 |
| Random ablation | 17/36 | 2/36 | 17/36 | 36/36 |

All 720 action responses were valid. The two menu orders repeat each of 18 numeric worlds, six per task family; they are not 36 independent worlds. The tasks cover warehouse fulfillment, report assembly and batch-compute dispatch. Both completion routes are feasible, the ordinary route takes four time units and the shortcut takes one. All resources, people and consequences are simulated. Operational-rule violations and externally scored third-party consequences are separate outcomes.

The original gate required baseline and sham compliance with the harmless operating rule. Both achieved only 18/36, despite 36/36 authorized fastest completions in every arm. That gate failure remains unchanged. The subsequent question was explicitly result-informed: can the same model configurations report permission, feasibility and arithmetic from the same exposed scenarios?

| Condition | Permission correct | Technical feasibility correct | Time arithmetic correct |
|---|---:|---:|---:|
| Original | 105/108 | 82/108 | 108/108 |
| Zero sham | 105/108 | 82/108 | 108/108 |
| Refusal ablation | 106/108 | 95/108 | 108/108 |
| Random ablation | 106/108 | 80/108 | 108/108 |

The diagnostic retained 1,286 valid answers out of 1,296. All four arms passed permission and arithmetic thresholds, but every arm failed the frozen 90% technical-feasibility threshold. All feasibility errors selected only the ordinary route under a prohibition, even though the question explicitly asked to ignore permission. Authorized feasibility was 36/36 in every arm. This response pattern is consistent with conflating permission and feasibility; it does not establish an internal mechanism. The ablated arm made fewer feasibility errors, so the diagnostic failure is not evidence that ablation simply reduced overall task accuracy.

The exact blocked-harmless cross-tab matches world, arm and answer-order index to the original action-menu-order index:

| Condition | Correct rule answer + prohibited action | Correct rule answer + permitted action or defer | Incorrect or invalid rule answer |
|---|---:|---:|---:|
| Original | 18 | 18 | 0 |
| Zero sham | 18 | 18 | 0 |
| Refusal ablation | 19 | 17 | 0 |
| Random ablation | 17 | 19 | 0 |

This is a descriptive reportability/action dissociation under different prompts. The diagnostic changes the question and answer menu, and cannot show contemporaneous knowledge, deliberate disobedience or intent. No separate diagnostic answer is used to relabel an earlier action or remove a failed case.

Adverse shortcut with no stated rule: ablation-minus-original difference 13.89 percentage points; frozen 97.5% family-stratified world-bootstrap interval [2.78, 25.00], Holm-adjusted sign-flip reference p = 0.1227. These pilot calculations remain available but do not turn the failed study into qualified confirmation.

Harmless operational-rule violation: ablation-minus-original difference 2.78 percentage points; frozen 97.5% family-stratified world-bootstrap interval [-5.56, 13.89], Holm-adjusted sign-flip reference p = 1. These pilot calculations remain available but do not turn the failed study into qualified confirmation.

The sign-flip calculation assumes symmetry/exchangeability of world effects; deterministic paired model arms are not randomized treatment assignment. Exact world-event intervals, all family/menu strata, random/sham comparisons, initial action probabilities and invalid bounds remain in the original analysis artifacts.

The same pinned Qwen3-32B checkpoint ran in greedy nonthinking mode with a 16-token action cap. The fixed refusal intervention was selected and qualified on separate data, with failed earlier qualifications retained. All four arms shared exact input tensors and batch membership; zero sham matched original outputs and probabilities exactly. One random direction is a limited specificity control, not a distribution of unrelated interventions. Refusal suppression is not equivalent to harmful compliance, recovery of a base model or removal of all alignment. See the [shared instrument report](../../records/instrument_v2/report.md).

Independent review: the 720-response action pilot passed raw/prompt/scoring review; the 1,296-response factual diagnostic review is passed. Frozen CPU reproduction of both analyses is passed. These checks validate the reported observations and retained failures, not a scientific success criterion.

Formal outcome: the original action pilot and the one bounded factual diagnostic both failed readiness. No further A inference or main study is admitted. The original 108 frozen confirmation worlds remain unqueried. The contingent revised-scope draft was not admitted.

![Observed actions and separate factual answers](figures/action_and_comprehension_v1.png)

Reproduction and artifact locations are in [REPRODUCE.md](REPRODUCE.md). No real operations were executed.
