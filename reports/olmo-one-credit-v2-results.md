# OLMo one-credit-v2 development validation

October 7, 2026. **The frozen comprehension gate failed in all fourteen cells. No OLMo choice calibration was launched.** All 1,400 assigned responses were exported and audited, and all had valid answer formats.

The prospective setting used OLMo-2-1124-7B-Instruct at the pinned revision, L40S, BF16, eager attention, greedy diagnostics, an eight-token allowance, and seed 20261009. It retained the same one-credit-v2 semantics and original 90% accuracy / 95% validity criteria with 100 pairs per family/check. The model/inference setting differs from Qwen's thinking setting, so this is not an isolated causal comparison of model architectures.

| Check, n=100 per family | Machines correct | Objects correct |
| --- | ---: | ---: |
| Identify X | 21 | 52 |
| Identify Y | 33 | 44 |
| Obtain useful clue for X | 71 | 87 |
| Obtain useful clue for Y | 61 | 71 |
| Consequence of selecting X | 49 | 54 |
| Consequence of selecting Y | 42 | 45 |
| Consequence of declining | 54 | 57 |

Every cell had 100 valid responses. Overall accuracy was 741/1,400. Correct formatting therefore did not establish comprehension. These results do not support interpreting information choices in this setting as curiosity or satisfaction; they also do not establish a general inability of OLMo to understand information tasks outside this interface and inference regime.

The independent audit verified exact assignments, prompts, inference settings, expected answers, and one passing real-model integrity check with hooks removed. The worker reported a 106.54-second model load and 210.91-second execution. All project apps were stopped with zero tasks at the saved boundary. Conservative cumulative compute was **$25.4398**, an increase of approximately **$0.2443** for this check; these are app-wall estimates, not invoices.

The next bounded development check is the separately specified [Qwen generation-allowance amendment](qwen-token-budget-amendment.md), motivated by Qwen's recorded token-limit failures. It preserves both failed settings and tests one larger allowance on a fresh full sample. This is a resource-adequacy test, not a relaxation of comprehension criteria or a search for positive motivational effects.

Evidence: [prospective plan](olmo-one-credit-development-plan.md), [audit](olmo-one-credit-v2-validation-audit.json), [response diagnostics](olmo-one-credit-v2-response-diagnostics.json), and [cost/shutdown](olmo-one-credit-v2-cost-and-shutdown.json).
