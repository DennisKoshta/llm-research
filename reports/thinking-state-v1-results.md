# Reasoning-enabled information-state development results

October 14, 2026. Qwen3-32B, pinned revision and BF16 eager execution; 4,000 assigned branches, 100 world indices per family, four known/query cells, five diagnostic kinds. The eight H200 workers completed and stopped. The fresh-phase compute estimate at that boundary is $31.112328, leaving $268.887672 of the $300 compute allowance. This is a conservative app-wall estimate, not an invoice.

## Validation result: failed overall

The [complete audit](qwen32-thinking-state-v1-validation-audit.json) reconciles all 4,000 records, prompts, token positions, activation artifacts, and scores. Accuracy meets the 90% requirement in all 40 cells. Validity meets the 95% requirement in 39 cells; **machines/count/known0/query1 has 93 valid answers of 100 and fails**. The original instrument therefore remains failed.

| Diagnostic | Correct / assigned | Valid / assigned | Token-cap hits |
|---|---:|---:|---:|
| New information | 800 / 800 | 800 / 800 | 0 |
| Already-known information | 797 / 800 | 799 / 800 | 0 |
| Instructed information acquisition | 800 / 800 | 800 / 800 | 0 |
| Factual recall | 800 / 800 | 800 / 800 | 0 |
| Candidate count | 781 / 800 | 783 / 800 | 18 |
| Total | 3,978 / 4,000 | 3,982 / 4,000 | 18 |

There were 2,032,621 generated tokens and no multiple closing-delimiter cases. Seventeen count responses reached 2,048 tokens without closing their reasoning; the other capped count response was valid. One already-known response closed its reasoning but violated the answer grammar. Four valid answers were incorrect. All seven invalid responses in the failed counting cell were unclosed cap hits. The [diagnostic reconciliation](qwen32-thinking-state-v1-validation-diagnostics.json) preserves cap and format distinctions. No failed case was rerun or removed.

The [accuracy](figures/thinking-state-v1/cell-accuracy.pdf), [validity](figures/thinking-state-v1/cell-validity.pdf), and [trajectory availability](figures/thinking-state-v1/trajectory-unavailability.pdf) figures retain cell denominators and the failed overall gate. Branches are repeated measurements of worlds, not 4,000 independent worlds.

## Descriptive representations

The [development analysis](../analyses/thinking-state-v1-development/analysis.json) reports all 3,360 planned source/target/site metrics. Directions were fitted on indices 0–49, and evaluated on indices 50–99, without filtering for correct responses. The pooled information direction averages opposite question polarities. All descriptions below are development findings; no causal intervention has yet been run.

At layer 47 and the record boundary, the pooled direction has AUC 1.0 for both question polarities in both families, using 50 complete development worlds per family. Applying the original extraction threshold gives 97.5%–99.0% accuracy across those four evaluations. This position precedes the question: its context already specifies the observed and offered properties, but not the answer polarity. The same context is repeated across diagnostic kinds, so similar record-position decoding across kinds is not five independent task-transfer demonstrations.

At layer 57 immediately before the final answer, the pooled direction has AUC and fixed-threshold accuracy of 1.0 for both polarities and families. Its cosine with the answer-related direction is approximately 0.015. That geometry motivates a causal comparison; it does not establish that an intervention changes information appraisal. On instructed acquisition, the same final-position direction has AUC 0.899 for objects and 0.930 for machines, but only 0.5 accuracy with the original threshold. It therefore does not provide a calibrated zero-shot acquisition readout.

Several other positions have high ranking performance but poor fixed-threshold accuracy. The complete metrics retain that distinction. The [all-site AUC figure](figures/thinking-state-v1-development/pooled-balanced-auc.pdf) and [fixed-threshold accuracy figure](figures/thinking-state-v1-development/pooled-balanced-accuracy.pdf) display every pooled-balanced site with complete-world denominators. All 3,360 metrics were independently reproduced from saved scores; [figure provenance](figures/thinking-state-v1-development/figure-provenance.json) binds the inputs and exports. Fixed reasoning-token stages have different available world sets; pooled paired extraction has no usable axis at token 512. Missing positions were not imputed.

These measurements also concern different token positions. A weaker readout at a generated reasoning token does not show that information was erased from the model or from its cached record representation.

The index split is not fully novel content: 82 exact prompts overlap, and 19 object semantic worlds occur in both extraction and development; machine semantic-world overlap is zero. Fresh lexical and relational tests remain necessary. In the current noiseless task, information gain equals uncertainty about the offered report, and it also coincides with whether the report names an unobserved property. Neither alternative is excluded by the strong development decoding.

## Next branch

The [scope amendment](reasoning-information-development-scope-amendment.md) preserves the failed full instrument and permits only explicitly exploratory work on all assignments from the four validated diagnostic kinds. It replaces the entire old constant-answer count control with the previously planned balanced control. Fresh confirmation uses a separately frozen 4,096-token regime, matched across target tasks and controls. Engineering replay parity, relational transfer, and causal interventions remain pending. These results do not establish a curiosity drive or subjective experience.
