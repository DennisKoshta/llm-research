# Sustained steering feasibility

October 14, 2026. Prospective and deferred until the voluntary pilot has been audited and a separate intervention plan is frozen. This note authorizes no implementation, experiment or compute. No behavioral outcomes were inspected for this review.

**Execution update, October 14 evening:** Both actual-model engineering stages are now complete and passed their main audit and an independent numerical check. The 128-token smoke exercised capped generation; the 4,096-token allowance produced trajectories up to 2,382 tokens, all ending normally. Zero-sham, untouched companion rows, cleanup and saved BF16 perturbation arithmetic passed exactly within those observed trajectories. These tests used curiosity steering in one row and do not establish satisfaction-driven behavior or certify every possible 4,096-step trajectory. See the [engineering results](sustained-engineering-v1-results.md) for evidence and scope. The separately specified [satisfaction screen](satisfaction-screen-v1-plan.md) remains the next scientific test. The feasibility discussion below records the original proposal.

A bounded test is technically feasible with unchanged model weights. Whether the locked description-derived candidates affect optional behavior remains an empirical question; fine-tuning is neither required for the proposed implementation nor assumed to be unnecessary for obtaining a functional effect. The distinction from the motivating paper is covered in the [alignment review](pain-axis-method-alignment-review.md). The recommendations here are methodological inferences, not replications of that paper's findings.

## Minimal explicit schedule

Preserve the existing `prefill_only` default. A new `record_prefill_plus_decode` schedule would patch the layer-15 block output at the original record token once during prefill, then patch position zero of each one-token cached decoding input for unfinished rows only. Preserve original prompt IDs, row seeds, attention masks, position IDs and a fresh cache for each condition. Never patch padded or terminal rows.

The first sampled token receives only the prefill intervention. Processing generated token j supplies the decoding intervention that can affect token j+1. A terminal EOS or cap-reaching token is not processed again. The expected per-row injection count is one prefill injection plus one for each nonterminal generated token actually processed. A final-prefill-plus-decode schedule or context-wide prefill steering would be a different intervention and must not be substituted silently.

Simply retaining the current hook is incorrect: its named token positions refer to the full prefill, and its capture/patch storage would be overwritten during decoding. The extension needs explicit prefill/decode phases, separate decoding geometry and hook-count records, and guaranteed hook cleanup on completion or exception. Prefill captures retain their existing meaning.

## Cache and engineering requirements

A post-block layer-15 patch changes downstream computations and higher-layer caches; it does not rewrite that block's already-computed keys and values. Removing the hook stops new injections but leaves previous cache effects. An on/off comparison cannot be described as an immediate state reset or unqualified relief test.

First use tiny-transformer tests for cached decoding, mixed EOS times, active-row masking, exceptions and hook cleanup. Then test one actual 32-prefix cohort in baseline, zero-dose sham and one-row nonzero conditions, retaining identical source rows, companions and seeds across conditions. Require:

- Exact baseline/sham generated-token identity and full-vocabulary logit identity at every active decoding step.
- The same identity for all 31 unperturbed rows under one-row steering, through each row's actual termination.
- Correct injection counts, token indices and active masks; no terminal-row patches or leaked hooks.
- Finite captures and recorded realized BF16 projection changes, perturbation norms and relative residual perturbations.

A bounded 128-token run can establish short-run engineering behavior. It cannot establish optional-choice effects when outputs remain unfinished. Before interpreting a 4096-token study, a separately bound full-cap baseline/sham check is required. Existing prefill-only evidence cannot certify sustained steering. Engineering receipts must bind model, source, direction, layer, schedule, cap, batch layout and sampling policy.

## Dose and interpretation limits

Use the locked corpus-A layer-15 candidate directions with their original signs and raw contrast norms. The previously proposed 0.5 and 1.0 norm multipliers are prospective calibration doses, not established effective doses. Curiosity and satisfaction remain separate interventions. A representation-selected layer is not necessarily an effective injection layer; a null remains site-, dose- and schedule-specific.

Report per-step and cumulative achieved perturbation and the number of tokens exposed. Completion length is treatment-affected: equal per-token doses do not imply equal total exposure, and exposure should not be treated as a pretreatment covariate. Random controls require matched per-token perturbation magnitudes under the same active-mask rule. Nuisance-direction controls require demonstrated engagement; weak or ineffective controls cannot establish specificity.

A later behavioral plan must preserve knowledge and mandatory competence across intervention arms within each evidence condition, include the locked nuisance controls and rotated action labels, and retain every invalid, capped or missing outcome. Informative, redundant and unrelated-novelty choices help distinguish selective acquisition from broad response bias. Steering through final-answer generation also permits direct answer-selection effects. Worlds remain the independent units.

Such a pilot could test functional influence on optional information seeking without fine-tuning. Neither a sustained effect nor a pulse-only null establishes a subjective or owned state. No intervention study is specified or authorized by this feasibility note.
