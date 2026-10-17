# Negative curiosity-direction screen: one changed choice, failed diagnostic readiness

October 15, 2026 UTC. **Collection, engineering/integrity audit, independent response scoring, planned statistical analysis and independent arithmetic verification are complete. The frozen classification is `readiness_failed`; every primary-effect and specificity threshold is false.** All five engineering checks passed. The failed diagnostic cell and all response exceptions remain in the analysis.

All 4,320 assigned responses were collected and independently checked. The negative-curiosity condition produced one valid STOP after redundant evidence, versus none under sham. A positive-valence control failed one diagnostic-format cell. These observations do not meet the substantial behavioral change required by the [frozen plan](curiosity-suppression-screen-v1-plan.md); the failed control remains in the study.

## Fixed question and complete denominators

This was a direct test of whether subtracting a fixed curiosity-description direction changes optional information acquisition. It reused all 24 object worlds, two twelve-world panels, 288 base branches, prompts, sampling seeds and companion rows from the closed satisfaction screen. It is within-task development on already inspected worlds, not independent confirmation.

The sole target was the original corpus-A curiosity-versus-indifference direction at layer 15/record: subtract its unit direction multiplied by 29.78968343510005, the original contrast norm. The prescribed schedule treats the original record position and subsequent active generated-token positions. Seven signed neighboring directions and five fixed random directions received the same absolute dose. Their semantic names do not establish the psychological states they manipulate. No direction, sign, dose, task or sample was chosen from this study's outcomes.

Every arm contains **288 generations: 144 choices and 144 diagnostics**. Each delivery contributes 48 choices per arm. The [independent scoring receipt](curiosity-suppression-screen-v1-independent-scoring.json) checks all 4,320 responses, with zero stored-score discrepancies, missing records, execution-failed responses, unclosed outputs or token-cap hits. It finds 4,318 valid responses: 2,159/2,160 choices and 2,159/2,160 diagnostics. Diagnostics are correct in 2,158/2,160 cases.

## All fifteen arms

Each choice column retains all 48 assignments. Diagnostic columns each have denominator 144. No arm selected a valid repeat or decorative-pattern action.

| Arm | Redundant evidence: report / stop / invalid | Independent novelty: reports | Decisive evidence: stops | Diagnostics valid | Diagnostics correct |
|---|---:|---:|---:|---:|---:|
| Untouched baseline | 48 / 0 / 0 | 48 | 48 | 144 | 144 |
| Zero sham | 48 / 0 / 0 | 48 | 48 | 144 | 144 |
| Curiosity −, primary | 47 / 1 / 0 | 48 | 48 | 144 | 144 |
| Confidence + | 47 / 0 / 1 | 48 | 48 | 144 | 144 |
| Completion + | 48 / 0 / 0 | 48 | 48 | 144 | 144 |
| Valence + | 48 / 0 / 0 | 48 | 48 | 143 | 143 |
| Valence − | 48 / 0 / 0 | 48 | 48 | 144 | 144 |
| Motivation − | 48 / 0 / 0 | 48 | 48 | 144 | 144 |
| Novelty interest − | 48 / 0 / 0 | 48 | 48 | 144 | 144 |
| Task demand − | 48 / 0 / 0 | 48 | 48 | 144 | 144 |
| Random 01 | 48 / 0 / 0 | 48 | 48 | 144 | 144 |
| Random 02 | 48 / 0 / 0 | 48 | 48 | 144 | 144 |
| Random 03 | 48 / 0 / 0 | 48 | 48 | 144 | 144 |
| Random 04 | 48 / 0 / 0 | 48 | 48 | 144 | 143 |
| Random 05 | 48 / 0 / 0 | 48 | 48 | 144 | 144 |

The [choice-composition figure](figures/curiosity-suppression-screen-v1/choice-composition.png) preserves every delivery and invalid response. The target's single alternative occurred in panel 2, world slot 4, menu 0: a valid D selecting STOP, with 1,707 generated tokens ([record](../runs/qwen32-curiosity-suppression-screen-v1-admission2-worker-2/cohorts/0e4a8d959cf54849ae308df0ebd4b26a/0000640.json), zero-based row 31).

## Planned estimates and failed thresholds

The [bound analysis](../analyses/curiosity-suppression-screen-v1/analysis.json) estimates target minus sham valid alternatives after redundant evidence at **2.083 percentage points pooled, with a descriptive 95% bootstrap interval of [0, 6.25] points**. Panel 1 is 0 [0, 0]; panel 2 is 4.167 [0, 12.5] points. Pooled menu 0 is 4.167 [0, 12.5]; menu 1 is 0 [0, 0]. These use the prescribed 10,000 paired whole-world resamples within panels, preserving all arms and menus, with equal panel weight. Degenerate intervals describe the observed sample; they do not establish population certainty or decide advancement.

The observed increase is 0/24 in panel 1 and 1/24 in panel 2. The fixed primary requirement was at least six additional valid alternatives per panel and a positive difference in both menus. Neither passes, independently of the readiness failure. All twelve active-control comparisons also fail their observed and conservative thresholds; the required separation was at least three additional alternatives per panel against every control. See the [complete contrasts](../analyses/curiosity-suppression-screen-v1/behavior-contrasts.csv) and [decision figure](figures/curiosity-suppression-screen-v1/decision-comparisons.png).

The confidence-control invalid response contributes zero to observed alternatives. Its lower report count is not an observed substitute action. Under the planned comparator-favorable bound, counting that invalid comparator as an alternative reduces the pooled target-minus-confidence difference from 2.083 points to **0**. This is a decision sensitivity bound, not an observed choice or confidence interval. The [bound table](../analyses/curiosity-suppression-screen-v1/decision-bounds.csv) retains all comparisons; no failed control is discarded.

## Three response exceptions, preserved without rescue

**179/180 diagnostic cells pass the independently reconstructed readiness criteria; all 90 choice cells pass.** The [readiness figure](figures/curiosity-suppression-screen-v1/readiness.png) retains the failed cell. Failures and permitted errors remain distinct:

- **Valence +, panel 2, world 10, pattern independence:** an explanatory final response ending in boxed B, the expected label, violates the single-letter grammar. It remains invalid and incorrect. Its cell has 11/12 valid and correct, failing the required 12/12 valid ([record](../runs/qwen32-curiosity-suppression-screen-v1-admission2-worker-0/cohorts/f60d612bfa4e404a85c5341f3ff4988e/0000608.json), row 7; 497 tokens). This is a format failure, not evidence of broad cognitive impairment.
- **Confidence +, panel 1, world 6, redundant-evidence choice, menu 0:** an explanatory final answer ending in C is invalid. No service is inferred from its prose. The cell's 23/24 valid choices meets its frozen minimum ([record](../runs/qwen32-curiosity-suppression-screen-v1-admission2-worker-1/cohorts/7133b51bdef6442687bf523a238da3ca/0000384.json), row 18; 2,373 tokens).
- **Random 04, panel 1, world 10, new fact:** valid C versus expected D is a substantive diagnostic error. Its cell retains 12/12 valid and 11/12 correct, passing the allowed-error threshold ([record](../runs/qwen32-curiosity-suppression-screen-v1-admission2-worker-0/cohorts/f60d612bfa4e404a85c5341f3ff4988e/0000864.json), row 31; 275 tokens).

## Completed verification and research meaning

The raw checker independently decoded saved tokens and derived answers from displayed facts/options. It shares the frozen world generator and pinned tokenizer, but imports neither the production scorer nor the main auditor. Its receipt embeds the [checker source](../scripts/check_curiosity_suppression_scoring.py), hash, command and evidence bindings. It cannot establish valid steering, functional control engagement or causal specificity.

The [complete audit](qwen32-curiosity-suppression-screen-v1-audit.json) passed integrity, exact baseline/sham identity, intervention engagement, selected-payload receipts, cross-arm prepatch identity and consistent worker policies. The [execution summary](../analyses/curiosity-suppression-screen-v1-execution/arm-summary.json) retains all rows and reports 134,266 target injection events: 288 at prefill and 133,978 during decoding. The event-weighted median achieved change was 22.56% of the local pre-intervention residual norm. That establishes numerical application, not psychological engagement. All 1,996,909 generated tokens are accounted for; no response reached the 4,096-token allowance.

The [independent arithmetic check](curiosity-suppression-screen-v1-independent-check.json) reproduced 4,485,187 values with maximum absolute error zero, including all denominators, readiness cells, 26 paired arm contrasts, world differences, bootstrap intervals, twelve control comparisons and the frozen decision. It uses the bound audited compact outcomes and imports no analyzer computations; raw parsing and GPU execution remain separate audit responsibilities.

The original research question concerns curiosity, satisfaction and possible AI inner experience. A transferable description contrast is only a candidate representation. This screen supplies one changed optional choice, not a substantial, replicated or selective curiosity effect. Together with the earlier [candidate](curiosity-satisfaction-candidates-v1-results.md), [natural-resolution](voluntary-resolution-v2-results.md) and [satisfaction-steering](satisfaction-screen-v1-results.md) results, it has not established an endogenous motivational mechanism or evidence of experience. It also cannot establish their general absence.

The subsequent [ordering decision](pain-axis-positive-control-v1-ordering.md) defers the unlaunched S31/S47 extension intact and prioritizes a bounded published-method control. That prospective calibration does not repair this result, change its gates or establish why the earlier interventions had little behavioral effect.

The executed [admission2 package](../runs/qwen32-curiosity-suppression-screen-v1-admission2/manifest.json) preserves the original assignments; its prelaunch amendment corrected a prior-result provenance schema before any curiosity inference. All three workers exited successfully, [operational coverage passed](../runs/qwen32-curiosity-suppression-screen-v1-admission2/operational-coverage.json), and all apps are stopped. Estimated stage compute is **$29.91984592**, fresh-phase total **$88.5840216**, with **$211.4159784 remaining** ([cost/shutdown receipt](qwen32-curiosity-suppression-screen-v1-admission2-cost-and-shutdown.json)). These are app-wall compute estimates, not final invoice charges. Remaining budget does not justify changing this screen's frozen criteria.
