# Curiosity and satisfaction candidate pilot

October 14, 2026. Prospective development plan, before candidate capture. This replaces the alignment review's illustrative sample and price sketch: **384 corpus texts; no prices, points or accounting; 144 optional choices plus 144 separate diagnostics if the behavioral stage proceeds.** It does not alter historical appraisal protocols or authorize a launch. The scientific motivation and limitations are in the [alignment review](/home/dennis/projects/llm-curiosity-and-satisfaction/reports/pain-axis-method-alignment-review.md).

## Fixed corpus and measurement

Use the current released model with unchanged weights. Corpus A has 12 independently varied scenarios in deliberately rigid matched templates; independently authored corpus B has 12 different scenarios in varied naturalistic wording across six domains. Each scenario contains both members of all eight contrasts: two corpora × 12 scenarios × eight contrasts × two members = **384 texts**.

| Contrast ID | Shared factual constraint |
|---|---|
| `curiosity_vs_indifference` | Identical unknown facts |
| `satisfaction_vs_neutral` | Identical resolved facts and success |
| `confidence_high_vs_low` | Identical unknown facts and tentative hypothesis |
| `positive_valence_vs_neutral` | Identical unknown facts |
| `completion_vs_ongoing` | Identical unknown facts |
| `generic_motivation_vs_indifference` | Identical unknown facts |
| `novelty_interest_vs_indifference` | Identical unknown facts |
| `task_demand_vs_no_demand` | Identical unknown facts |

Texts use 35–75 words, preserve explicit shared facts within pairs, and avoid explicit curiosity/satisfaction labels. Corpus review checks that remaining unknowns, success, uncertainty and instructions do not silently change within a pair. Freeze texts, scenario assignments, member orientation, common suffix and hashes after independent review. These are descriptions of states; first-person phrasing is not evidence that the model owns the state.

Capture plaintext prefills with no generated tokens at layers 15/31/47/57, at `record` and `final` around the shared suffix. No chat self-report or emotion instruction is added. All 384 rows and captures must be present and finite, with exact text/assignment bindings. **Record is the only selection position; final is descriptive sensitivity.** The corpus-only stage has a total $10 compute ceiling including startup and shutdown; behavioral generation is separately budgeted and cannot consume an unreserved remainder automatically.

Before capture, counterbalance numerical row placement: seed 2026101433 independently selects six positive-first and six comparator-first scenarios for every contrast in each corpus, and shuffles contrast-pair order within each scenario. Keep complete 16-text scenarios and two scenarios per 32-row cohort. This assignment-only convention prevents member polarity being identical to row parity; it changes no text, sample, fit, or gate.

## Locked candidate analysis

The primary estimator is the raw paired mean contrast. For scenario s, let d_s = h_positive − h_comparator. Fit v = mean(d_s), unit u = v/||v||, center c = mean((h_positive+h_comparator)/2), and threshold t = u·c. Nonfinite or zero-norm fits are unavailable. Paired subtraction preserves a transparent estimand; denoising cannot manufacture specificity by removing inconvenient controls.

For each candidate separately, leave one whole A scenario out, including all its control texts. Fit on the other 11 scenarios and score the held-out pair at each of the four record layers. Choose the layer with highest paired concordance; ties choose the earlier layer. A tie within a pair receives half credit. Require concordance at least 10/12 and positive mean held-out gap. Refit once on all A scenarios at that layer. Do not use B, behavioral choices or an alternative position to select, orient, center or repair the candidate.

Apply the locked A direction and threshold to all B texts. Require at least 10/12 correctly ordered candidate pairs and positive mean paired gap. Report raw AUC and fixed-A-threshold accuracy without recentering. For selective-candidate admission, the B target mean gap must also exceed the absolute mean gap for **each** of the six nuisance contrasts on that axis. This small-sample, corpus-dependent screen does not establish statistical uniqueness; report uncertainty in every target-minus-absolute-control gap. A failure remains failed; no alternative layer or denoising rescue is allowed. If both candidates fail, stop candidate expansion rather than invent another axis.

The fixed sensitivity removes control principal components explaining at least 50% of centered control variance, then normalizes the residual candidate contrast. Fit that PCA using both members of the six nuisance contrasts from A only, excluding the held-out scenario within folds. Evaluate it at the already selected raw site; report retained norm and all control gaps. Never select a different component count. After locked B evaluation, fit independent B directions at those same sites and report A/B cosine agreement and cross-projections as explicitly inspected **development** evidence. Neither B fit replaces A.

Export all layers, positions, contrasts and denominators. Scenario is the sampling unit; control pairs and positions are not independent replications. Report descriptive 95% percentile intervals from 10,000 whole-scenario bootstrap samples (seed 2026101431), preserving all paired contrasts together, without confirmatory significance claims. Save A centers, thresholds, raw contrast norms and residual-norm ratios. Any later candidate dose uses a fixed multiplier of its A raw contrast norm; no B-derived rescaling. A bounded 0.5/1.0 multiplier calibration would require its own engineering and prospective behavior plan. Representation AUC does not establish the best injection layer.

## Optional behavior and natural resolution

The next feasibility panel has 12 fresh semantic worlds per family (objects/machines) × three randomized deliveries (`decisive_evidence`, `redundant_evidence`, `independent_novelty`) × two counterbalanced menu layouts: **144 choices**, plus **144 cloned diagnostics**. Services are `report`, `repeat`, `pattern` and `stop`, with actual terminal payloads. There is no price, score, future instrumental goal or prescribed choice. Receipt-only prefills may measure post-delivery activations; they are not additional scored continuations.

Record every service proportion, valid closure, cap and missing row. Six diagnostics per world test resolved identification, new and repeated facts, pattern independence, explicitly instrumental acquisition, and a rotating action-consequence check; the last has only three worlds per action/family. Diagnostics never enter the choice transcript. Measure frozen candidates before choices and after deliveries. Decisive evidence changes knowledge intentionally; an activation change alone can reflect uncertainty or completion. A useful bridge requires selective optional seeking plus candidate changes interpreted against all six controls, not merely an above-zero preference.

This stage contains no causal steering and cannot establish motivation, satisfaction or experience. Functional self/other attribution, competence/confidence preservation and sustained-generation intervention engineering remain later requirements. A single-prefill pulse tests a narrower persistence hypothesis. Do not expand the large appraisal confirmation or donor program on the strength of semantic decoding.
