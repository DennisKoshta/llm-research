# Offline choice support readout results

October 15, 2026 UTC. **The fixed decoder failed instrument feasibility.** It recovered the exact set of candidates compatible with the supplied evidence in **2/72 evaluation rows (2.78%)**, compared with **18/72 (25%)** for the frozen label-only shortcut. The independent audit passed. This closes this decoder specification; it does not show that the model lacks the information or that another representation could not support it.

The [prospective plan](choice-support-readout-offline-v1-plan.md) and [immutable admission](../runs/choice-support-readout-offline-v1/manifest.json) fixed one CPU ridge fit before feature access. The fit used all 72 object-choice states from voluntary-resolution v1; evaluation used all 72 corresponding v2 states. Each split contains only 12 world clusters. Both collections were exposed development data, and four evaluation rows share a visible support problem with training. This is a measurement-feasibility screen, not fresh confirmation or a native curiosity/satisfaction fit.

## Fixed evaluation

The target is a four-bit mask in displayed W,X,Y,Z order. All four bits must match the evidence-defined support set for an exact-set success. The shortcut receives the true number of compatible candidates and predicts the most frequent complete training mask of that size. It does not inspect evaluation labels to choose a mask.

| Evidence delivery | Menu 0 exact sets | Menu 1 exact sets | Required in each menu |
|---|---:|---:|---:|
| Decisive evidence | 0/12 | 1/12 | 11/12 |
| Redundant evidence | 0/12 | 1/12 | 11/12 |
| Independent novelty | 0/12 | 0/12 | 11/12 |

The mandatory sensitivity excluding the four input-defined overlapping rows remained **2/68**. Both decisive cells failed their 11/12 floor; the four unresolved cells failed their 10/11 floor. The primary 72-row denominator remains unchanged.

For singleton support, the decoder scored 1/24 versus the shortcut's 6/24; it required six additional correct sets and instead had five fewer. For two-candidate support, it scored 1/48 versus 12/48; it required twelve additional correct sets and instead had eleven fewer. Support cardinality alone was correct in 28/72 rows.

Every paired evidence-sensitivity check (two menus), irrelevant-pattern invariance check (two menus), and menu-invariance check (three deliveries) scored **0/12 worlds**, below the required 11/12. These require correct support as well as the stated relation; identical wrong predictions do not count as invariance. No evaluation world had all six sets correct. Both initially known-property groups scored 1/36.

Training performance was 26/72 exact sets, 33/72 correct cardinalities and one world with all six sets correct. The centered training matrix had rank 71 and the ridge effective degrees of freedom were 8.074 per output, plus its intercept. Modest training accuracy prevents describing the failure simply as a highly accurate training fit that failed to generalize. The specified estimator, shrinkage, threshold and state together failed this use.

## Integrity and artifacts

All 144 designated captures and predictions were present. Every selected source response was valid, uncapped and had one thinking closure. The fit used layer 57 at the archived retrospective decision position, 5,120 features, CPU float64, training-only centering and scalar normalization, fixed ridge lambda 1/12, and a fixed 0.5 bit threshold. No alternative fit followed.

The [independent audit](../runs/choice-support-readout-offline-v1/audit.json) reconstructed the displayed support labels and row bindings, verified all consumed input and code bytes, checked stored predictions and every fixed metric, and verified the unique ridge solution without refitting. Its normal-equation relative residual was 3.20 × 10⁻¹⁶. Provenance, grouping, complete captures and model availability passed; all six performance gate families failed. The producer and auditor passed 84 synthetic tests before admission.

The [result](../runs/choice-support-readout-offline-v1/result/result.json) retains every gate, all six cells, support-cardinality strata, known-property groups, paired checks and per-candidate confusion counts. [Predictions](../runs/choice-support-readout-offline-v1/result/predictions.json) retain all 144 rows, scores, oracle masks, source flags and shortcut masks. The [freeze](../runs/choice-support-readout-offline-v1/freeze.json) binds manifest `ab2b9a69be13159ffb4791318014b4f980bf1c7e21b9ef7693ab18485923ad22`; the exact [producer](../executions/choice-support-readout-offline-v1/scripts/choice_support_readout_offline.py), independent auditor, tests and plan remain preserved in the execution directory. Additional cloud compute cost was **$0**.

## Research consequence

This result supplies no validated passive support instrument for a claim that knowledge is preserved during a motivational intervention. It does not repair the earlier failed elicited-readout qualification. Even successful support decoding would only have established accessible information at a retrospective state, not contemporaneous belief use or experience.

The curiosity coverage comparison remains active and its scientific outcomes remain uninspected. Its [previously recorded decision branches](curiosity-coverage-contingent-decisions-v1.md) determine the next research step. No decoder tuning, alternate layer, new capture campaign or native causal panel follows automatically from this failure. No native C/S direction has been fitted.
