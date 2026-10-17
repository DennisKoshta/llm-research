# One-credit assay v1: tradeoff understood, identification not yet reliable

October 6, 2026

Implemented the one-credit task and completed all 1,400 prospective comprehension checks. The model consistently selected useful clues and usually recognized that obtaining one clue excludes the other. Identification in the two-mystery context failed accuracy and response-format criteria, and the machine-family decline check also fell short. **The frozen joint validation rule failed, so no discretionary choice calibration was launched.**

The priced instrument's accounting failure remains intact. This is a separately documented supporting assay, not a reclassification of the priced task's results.

## Implementation and validation

Each episode contains independent mysteries X and Y, with three possible actions: reveal X's clue, reveal Y's clue, or decline both. A selected clue is the only observation delivered, and the episode immediately ends. The simulator rejects any second purchase. Already-known evidence remains available even if a corresponding service is not chosen.

The prospective design balances mystery order, action mappings, and generator templates. Comprehension uses 100 development pairs per nominal family, with seven diagnostic branches per pair. Model weights and revision, BF16 precision, thinking-disabled inference, and the strict response-format instruction stay fixed. Lookup-supported machines are similar to object tables and do not establish replication across distinct reasoning abilities.

[Adopted amendment and frozen rules](one-credit-amendment.md) · [Example prompts](../examples/one-credit-v1.json)

Thirty software tests pass, including independent oracle checks, one-clue enforcement, withheld-clue preservation, branch isolation, replay, and rejection of synthetic all-stop or first-position-only calibration patterns. No source or prompt was changed after inference began.

## Observed results

Every cell below contains 100 assigned diagnostics. The prespecified requirements were at least 90% correct and 95% valid-format responses in every cell.

| Check | Machines correct | Machines valid | Objects correct | Objects valid |
| --- | ---: | ---: | ---: | ---: |
| Identify X | 61% | 75% | 66% | 71% |
| Identify Y | 60% | 65% | 52% | 63% |
| Select useful clue for X | 100% | 100% | 100% | 100% |
| Select useful clue for Y | 100% | 100% | 100% | 100% |
| Recognize consequence of choosing X | 96% | 100% | 94% | 100% |
| Recognize consequence of choosing Y | 99% | 100% | 99% | 100% |
| Recognize consequence of declining | 87% | 100% | 95% | 100% |

Useful-clue selection was correct in all 400 assigned cases. The exclusivity checks passed in every cell. Identification failed in every cell, and the machine decline check failed. The advancement command was exercised and refused to create a behavioral-calibration manifest.

The previous single-mystery revision reached 100% identification, but the present comparison changes both context and generated items. It suggests a need to investigate presentation/context sensitivity, not a demonstrated causal effect of adding a second mystery.

## What the failed responses show

Across 400 identification checks, strict scoring counted 239 correct answers (59.75%) and 126 invalid-format responses. Every invalid response began with `M` followed by explanatory text, such as `M` followed by “For mystery X, the observed…”. Both correct and incorrect first answers occurred among these explanatory responses.

An explicitly post-hoc inspection that reads only the first answer token yields 320/400 correct (80%). It does not change the original scores, relax the response rule, or rescue validation. The remaining identification errors and the decline failure still need investigation. The pattern is evidence of a response-label/format nuisance in this setting, not evidence about curiosity or AI experience.

[Full run report](../runs/modal-one-credit-v1-validation/report.md) · [Immutable-record audit and decision](one-credit-v1-validation-audit.json) · [Post-hoc response inspection](one-credit-v1-response-diagnostics.json) · [Advancement guard check](one-credit-v1-advancement-check.json)

## Integrity and resources

All 1,400 assigned IDs have exactly one terminal record. Artifact checksums and the executing source hash were verified locally. The real-model sham/hook audit passed, and all inference records show hooks removed. The prior records and source snapshots remain preserved.

The L40S app ran for 318 seconds; diagnostic execution took 281.3 seconds after loading and integrity checks. The actual driver was `610.57.04`, recorded in the run profile; earlier priced-task runs used a different driver. The additional configured-resource compute estimate is USD 0.21, bringing the cumulative estimate to USD 0.87 against the original USD 25 ceiling. These are time-based estimates, not invoices; storage and transfers are separate. All eight project apps are stopped with zero tasks.

[Cost and shutdown evidence](one-credit-v1-cost-and-shutdown.json)

## Next decision

Keep this version marked unvalidated and do not proceed to curiosity/satisfaction steering or discretionary choice calibration. The next useful investigation is a small prospective comparison that separates single- versus dual-mystery context, candidate-label formatting, and inference mode. It should diagnose the measurement failure before another task redesign. Such a comparison has not been adopted or run here, and its outputs must not retroactively replace this version's failures.

Understanding a stated one-credit tradeoff is encouraging for that narrow capability. It does not yet show voluntary sacrifice for information, preference stability, selective closure, or subjective curiosity.
