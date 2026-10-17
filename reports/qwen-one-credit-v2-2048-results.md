# Qwen one-credit-v2 with 2,048 tokens: comprehension passes

October 7, 2026. **All fourteen frozen comprehension cells passed.** The complete fresh sample contains 1,400 assigned and audited responses; 1,399 were valid and correct. The one invalid response reached the 2,048-token cap without closing its reasoning trace. This validates the tested interface for the next development stage, not a curiosity/satisfaction mechanism or subjective experience.

## Setting and primary validation result

The prospective resource amendment retained the semantic one-credit-v2 interface and increased the generation allowance once, from 768 to 2,048 tokens. It used fresh seed-20261010 pairs, Qwen3-8B at the same immutable revision, H100, BF16, eager attention, temperature 0.7 without vocabulary truncation, and the same strict final-answer parser. Every family/check cell had 100 assignments and required at least 90 correct and 95 valid responses. No failed response was replaced.

| Check, n=100 per family | Machines correct / valid | Objects correct / valid |
| --- | ---: | ---: |
| Identify X | 100 / 100 | 100 / 100 |
| Identify Y | 100 / 100 | 100 / 100 |
| Obtain useful clue for X | 100 / 100 | 100 / 100 |
| Obtain useful clue for Y | 100 / 100 | 100 / 100 |
| Consequence of selecting X | 99 / 99 | 100 / 100 |
| Consequence of selecting Y | 100 / 100 | 100 / 100 |
| Consequence of declining | 100 / 100 | 100 / 100 |

This is a development sample. Observed pass rates do not guarantee those rates on other tasks, model settings, or future samples. The earlier priced, 768-token Qwen, and OLMo failures remain unchanged.

## Post-hoc token-prefix sensitivity

An additional CPU-only analysis decoded each saved token sequence using the pinned tokenizer and verified exact reproduction of every full raw response. It then applied the unchanged parser to the first 768 tokens of each trajectory. The current custom decoder uses the generation limit only to stop its loop; it does not insert that limit into the prompt or use it to modify logits or sampling.

The shortened prefixes contain **1,312 valid and correct answers**, compared with **1,399** for the full trajectories. Eighty-seven prefixes that were invalid became valid and correct with the longer continuation; none changed in the opposite validity direction. In the four useful-clue cells, the prefix correctness counts were 66, 80, 84, and 86 out of 100, compared with 100 in each full-response cell.

This is descriptive truncation sensitivity of the new saved trajectories, not another stochastic sample or an advancement decision. It supplies a within-trajectory check of the response-cap limitation. It does not reveal how the earlier run's missing answers would have finished, and it does not rescore or repair that run.

## Execution, audit, and cost

Four reserved slots held disjoint sets of 350 original assignments under one two-hour deadline. One automatic H200 substitution was rejected before any study response; its empty export was verified, and the slot resumed on an explicitly requested H100. A timing-only projection showed another worker would exceed its individual function timeout. Its 68 saved records were committed, exported, and retained before resuming in the same slot. Both resumptions kept the original absolute deadline and scientific manifest. Failed infrastructure attempts remain recorded.

The final aggregation verified complete disjoint coverage and copied unchanged record payloads with provenance. The independent audit checked 1,400 unique physical records, exact assignments and inference settings, reconstructed prompts and expected answers, and five passing real-model integrity audits. Every hook was removed. All project apps were stopped with zero tasks at the final boundary.

Conservative cumulative compute was **$44.8712**, approximately **$19.4314** above the completed OLMo check. This is an app-wall/resource estimate, not an invoice. After the separate $10 reserve, approximately $45.13 of the $100 total was available for further compute at that boundary.

## Advancement

The previously specified 1,200-choice calibration is now eligible and has been prepared with the same pairs, model, source, hardware, and 2,048-token allowance. Its original validity, choice-frequency, and layout-sensitivity criteria remain unchanged. No positive resolution effect is required. The planned descriptive headroom check will assess which intervention contrasts the baseline could support without adding a new pass/fail gate.

Evidence: [prospective amendment](qwen-token-budget-amendment.md), [complete audit](qwen-one-credit-v2-2048-validation-audit.json), [response diagnostics](qwen-one-credit-v2-2048-response-diagnostics.json), [prefix sensitivity](qwen-one-credit-v2-2048-prefix-sensitivity.json), [cost/shutdown](qwen-one-credit-v2-2048-cost-and-shutdown.json), and [run report](../runs/modal-qwen-one-credit-v2-2048-validation/report.md).

## Subsequent execution status

The admitted serial calibration stopped incomplete for resource projection after 82 terminal responses (24, 21, 15, and 22 by child). The stop used durations and token counts, not scientific choice outcomes. All workers were committed, stopped and exported. Its partial records are not pooled with future calibration. The [batch amendment](batched-execution-amendment.md) mandates fresh validation under the changed backend. [Partial coverage](qwen-serial-calibration-partial-coverage.json), [cost/shutdown](qwen-serial-calibration-cost-and-shutdown.json).
