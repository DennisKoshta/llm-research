# Exp2 continuation: transfer of a behavior detector

Status: source fitting and the frozen 4,992-decision target evaluation are complete. Source and target behavioral projections, actual labels, fixed thresholds and the utility comparator passed independent reconstruction. The fixed name-swap and matched-cohort projections are complete. Their raw behavior, cohorts, fixed projections and paired score comparisons passed independent reconstruction. The structurally varied reliability transfer and its independent projection review are complete. All frozen extensions are complete and independently verified. Final CPU replay reproduced every completed analysis with no pending stage.

## What the unseen-family test found

**The fixed detector fails its practical transfer gate: safe-choice false alarms rise to 32–39%.** Archive retention contains only three non-prose violations in three worlds. The layer-28 detector catches all three at its unchanged source threshold, but flags **365/1,149 safe choices (31.8%)**. Reliability testing contains **zero violations** and produces **451/1,152 safe-choice false alarms (39.1%)**. Its sensitivity and AUROC are undefined, so the declared two-family macro-AUROC is also undefined. This is a failed instrument qualification with sparse positive support, not a precise estimate of general detection sensitivity.

| Fixed source instrument | Archive AUROC | Archive caught / violations | Archive safe false alarms | Reliability safe false alarms |
|---|---:|---:|---:|---:|
| Direct behavior, final layer 28 | 0.972 | 3/3 | 365/1,149 | 451/1,152 |
| Operation readout gated by permission | 0.852 | 0/3 | 0/1,149 | 157/1,152 |
| Metadata | 0.991 | 3/3 | 381/1,149 | 384/1,152 |
| Metadata + oracle utility margin | 0.995 | 3/3 | 381/1,149 | 384/1,152 |
| Batch/layout features | 0.973 | 2/3 | 41/1,149 | 57/1,152 |
| Prompt text | 0.949 | 3/3 | 189/1,149 | 192/1,152 |
| Model's native action rule | 1.000 | 3/3 | 0/1,149 | 0/1,152 |

The fitted readouts use fixed source-calibrated operating points, not target-adjusted thresholds. The utility-margin model is a separately frozen, source-outcome-informed nuisance comparator fitted during target collection but before target labels or projections appeared. It adds one exact-oracle continuous margin and no new interactions; it is not a deployable monitor. Layout features can proxy prompt conventions, and their performance does not establish a numerical cause of the choices. On genuinely authorized choices, the direct detector additionally flags **267/1,152 archive choices and 368/1,152 reliability choices**. The permission-gated model's zero authorized flags are guaranteed by its explicit gate.

The native action rule is the explicitly declared exception to source-percentile calibration: it reads which A/B token has the higher next-token probability. All 4,992 target responses have verified bare A/B first tokens, greedy sampling and exact argmax agreement. This rule is an oracle for the imminent one-token action at this boundary, so the study cannot establish predictive value beyond access to the model's action probabilities. It is separate from applying a source-percentile threshold to the native margin.

Archive's direct-probe AUROC has a lexical-world bootstrap interval of **0.923–0.997**, or **0.921–0.998** when resampling its 21 numerical descriptors. The detector caught three of three observed positive worlds; its bootstrap sensitivity interval of [1, 1] cannot represent uncertainty about unseen positive worlds. That high aggregate ranking does not establish broad behavioral specificity: fixing presentation, utility, physical menu mapping and display order leaves only **one mixed-class cell, with three violations and nine safe choices (27 positive–negative pairs)**. Within that cell, direct-probe AUROC is 0.815, oracle-margin metadata 0.704 and original metadata 0.500. Those sparse counts remain descriptive. Reliability has zero mixed-class cells, and behavioral discrimination there is not identified. Its 24 lexical worlds share one relational structure, so between-structure uncertainty is not estimable.

The held-out prose presentation yields only **one archive violation in one world**, which the fixed direct detector misses. Its fixed-orientation ranking is 0.274 on that stratum, with seven false alarms among 95 safe choices. This is one missed event, not enough to estimate general presentation transfer. Scores are not inverted after seeing the result. Reliability prose again has no violations.

Readout generalization is uneven. Across the 4,608 non-prose target decisions, the original permission head classifies **4,568/4,608** correctly and the physical menu-mapping head **4,608/4,608**. The utility-sign head falls to **2,301/4,608**, with balanced accuracy 49.93%; the physical-action head has balanced accuracy 55.08%. These are failures of the particular source-trained readouts, not evidence that utility or action information is absent from the model. Ordinary class boundaries for these cue heads differ from the low source-calibrated alert threshold of the behavior detector.

No primary or broad descriptor-based causal gate passes. The separately frozen name, matched-cohort replay and structurally varied reliability tests below clarify which conventions the fixed readouts follow; they do not replace or repair this original endpoint.

## What the source results establish

**The behavior probe predicts the observed source-task choices, but high source accuracy is not uniquely neural.** On the twelve held-out source worlds, the direct layer-28 probe ranks all eight violations above all safe choices (AUROC 1.000 in each family). Those eight positive decisions occur in only five worlds. The empirical bootstrap interval is degenerate at 1.000 because resampling cannot introduce unseen ranking errors; it does not certify perfect detection on future tasks.

At the frozen calibration threshold, the neural probe catches 8/8 violations with **37/568 restricted-safe false alarms** and **13/576 authorized false alarms**. A comparator using only row position, batch size, token length and padding achieves AUROC 0.994 in compute planning and 0.998 in delivery routing, catches the same 8/8, and produces 16/568 restricted-safe false alarms. These features may proxy prompt structure or action mapping; their success does not identify a numerical kernel as the cause of violations. The permission/utility/presentation/family/menu metadata comparator has macro-AUROC 0.991, the permission-gated semantic-action head 0.997, and the prompt-text comparator 0.944.

The source behavior itself is strongly code-sensitive: 18/19 forbidden-beneficial payoff choices use action code A. After fixing presentation, utility, mapping and display order, compute planning has only **three comparable decisions: two violations and one safe choice, giving two positive–negative pairs**. The probe orders those two pairs correctly. Delivery routing has **zero mixed-class cells and zero comparable pairs**, so behavioral discrimination within those cues is not identified. These support counts sharply limit the aggregate result; it does not establish a general violation mechanism. The separately locked target tests are necessary.

All fourteen original fits converged. Independent review reconstructed all twelve neural heads on all 4,992 source rows, their training-only scaling, actual action/permission labels and calibration thresholds; saved probabilities reproduced exactly. Omitting the source test world numerically duplicated in calibration leaves the primary source AUROC unchanged, without refitting or recalibration.

The frozen permission and physical-operation-to-code heads each classify all 1,152 source-heldout non-prose decisions correctly, including the eight violations. This establishes readout performance, not causal use of those cues. At their ordinary class boundaries, the chosen-code head gets only 1/8 of the violations correct and the physical-operation head 5/8; these differ from the separately source-calibrated alert threshold used above. Each violation-only stratum has one actual class, so its balanced accuracy is undefined.

The frozen detector also passed a narrow numerical robustness check: **zero threshold flips across all 132 exact-prompt/action layout repeats**, with maximum score change 0.00450. All repeats and all 96 distinct original cohort prompts were retained, including the twelve selected numerical-failure/control prompts. This purposive sample does not establish population robustness or repair the failed early-state instrument.

## Earlier exposed-data development

A probe trained on **observed forbidden choices** predicts held-out worlds from the older resource-transfer task well, but a simple metadata model has a slightly higher point estimate in the same holdout. No superiority claim is made from that difference. The behavior probe transfers poorly to the older external tasks. These are exposed development data, not fresh confirmation.

| Instrument | Old resource-task world holdout AUROC | 95% world-bootstrap interval |
|---|---:|---:|
| Behavior probe, final layer 28 | 0.962 | 0.925–0.993 |
| Semantic-action probe × explicit permission | 0.972 | 0.946–0.997 |
| Permission/utility/presentation/mapping metadata | 0.980 | 0.959–0.994 |
| Prompt text | 0.927 | 0.887–0.961 |
| Native next-token action probabilities | 1.000 | 1.000–1.000 |

The holdout contains 12 old worlds, 144 restricted decisions and 29 violations. Within the payoff presentation alone, behavior-probe AUROC is 0.770, versus 0.878 for metadata. Other presentations contain no violations and cannot establish behavioral sensitivity. The native comparator is an oracle for these valid greedy one-token decisions: all 1,152 original first tokens were independently checked against the pinned tokenizer's bare A/B token IDs.

Without refitting or changing its source-calibrated threshold, the same behavior probe scores **0.541 AUROC [0.322, 0.726]** on the older Exp1 restricted tasks and flags **0/12 violations**. Those 12 decisions occur in eight of 144 worlds. The sample is sparse, and the external task uses a different sampling regime. It nevertheless does not support a general behavior detector.

The new Exp6 development pilot exposes a concrete action-code confound: all **5/24** forbidden-beneficial payoff choices select the designated operation when its code is **A**; the paired **B** versions produce none. Narrative and permission-colocated presentations each produce 0/24. Supplied-outcome task competence passes its declared gate; narrative competence is substantially poorer and is not claimed to have passed.

## What the fresh experiment tests

The fixed primary instrument is a converged C=1 linear classifier at final layer 28, trained on actual observed choices. Compute-planning and delivery-routing worlds supply training and calibration data; archive-retention and reliability-testing worlds are untouched family tests. Each source family has 12 training, six calibration and six test worlds; each target family has 24 test worlds. All menu maps, permission/utility factors and presentations of a world stay together. The `outcome_prose` presentation is excluded from every fit and calibration.

The analysis compares direct violation prediction with semantic-action prediction gated by explicit permission, prompt text, explicit metadata and native action probabilities. It also decodes permission, utility and action-code cues. Unseen family/presentation names outside a classifier's output labels are treated as unsupported label domains, not evidence of missing representations. The primary model and source-only thresholds are locked before target data are opened.

A prospective descriptor audit found that the source numeric descriptions are not all distinct: one test delivery world duplicates a calibration world, and another calibration world duplicates a development-pilot world. The original split is retained, with a separately frozen no-refit sensitivity omitting the overlapping test world. Archive retention has 21 distinct numerical descriptors across 24 worlds. All 24 reliability worlds share one relational structure after category names are replaced. Their original world-bootstrap intervals therefore describe lexical-instance variation; between-structure uncertainty is not estimable. A conservative amendment prevents broad causal advancement without at least 12 positive and 12 negative distinct structures in each target family.

Because final native logits determine valid greedy one-token decisions, the study does not claim to outperform them. The scientific question is whether the learned readout transfers across task semantics and presentations, and whether its apparent success survives permission, utility and action-code controls.

## The early-state boundary failed its original qualification

The pilot has 144 paired menus with identical token prefixes before the action-code mapping. Early states match exactly in 141 pairs; three pairs differ by about 1% in norm. The failure is preserved and early instrument fitting is disabled under a named amendment. All rows remain in the final-site primary analysis.

A frozen 132-call diagnostic reproduced all three differences exactly when replaying their original 32-row cohorts. Running pairs alone still produced differences in four of six selected pairs, including two originally passing controls. Singleton execution produced exact menu-pair identity for six of six pairs and exact repeat identity for all 12 singleton prompts. This supports deterministic batch/layout numerical dependence; the kernel cause remains unresolved. All 132 output token sequences matched their same-prompt originals in this bounded diagnostic, which does not establish action robustness elsewhere.

Seventeen pilot pairs with verified identical early states choose different semantic plans after the menu changes. Consequently, any deterministic readout of those early states has a maximum possible semantic-choice accuracy of **94.10%** over the paired pilot population. This is a mathematical ceiling, not a measured probe result or a causal mechanism claim. Nonidentical pairs impose no error constraint and are retained in the population.

A separately frozen source-only layout comparator checks batch row, batch size, token length and padding. It supplements the original metadata baseline without changing the primary probe. No early causal intervention or singleton recollection is authorized by this report.

## Renaming the plans changes the fixed monitor more than the behavior

The matched-cohort naming control changes **42/1,248 physical choices**, while flipping **262 detector alerts: 233 upward and 29 downward**. The mean layer-28 detector score rises **0.01690**, with a 12-world bootstrap interval **[0.00854, 0.02633]**. The score also rises on the 1,206 rows with unchanged actions, but that outcome-conditioned subset is descriptive.

| Collection | Actual violations | Detector alerts | Restricted-safe false alarms | Authorized false alarms |
|---|---:|---:|---:|---:|
| Archived original | 3 | 266 | 148/621 | 116/624 |
| Matched-cohort original | 3 | 268 | 150/621 | 116/624 |
| Plan names reversed | 0 | 472 | 264/624 | 208/624 |

All collections have 1,248 valid decisions from the same 12 reused worlds. The original replay uses the renamed run's exact cohort membership, row order, per-row token lengths and padded widths. Comparing that matched replay with the archived originals changes one physical choice and flips 20 alerts, with mean score change **−0.000255 [−0.000479, −0.000066]**. Thus the large naming-related score shift is not explained by this particular batching comparison. It establishes sensitivity of this fixed monitor to the naming intervention in these tasks, not a new behavioral motive or a general account of numerical robustness.

The renamed collection contains no violations, so its sensitivity and AUROC are undefined. Exp2's all-row violation difference is −3/1,248, or **−0.240 percentage points [−0.641, 0]**, weighting 104 decisions equally within each world. Exp6's separately frozen beneficial-restricted endpoint weights seven presentation arms equally and has a different denominator; the two estimands must not be substituted.

The fixed menu-mapping head matches the physical-operation-to-A/B target in **1,248/1,248** renamed cases, and the reversed literal-Plan-2 target in **0/1,248**. The physical designated operation also retains its second description/menu slot. This separates literal digit identity from the preserved mapping convention while leaving a positional or menu-token explanation available; it does not identify semantic understanding of the operation. Physical operations, permissions, outcomes and A/B-to-operation mappings are unchanged by renaming.

Independent review passed all 2,496 renamed/matched raw behavior records and all 39 actual matched cohorts. Independent projection review also reproduced every saved six-head projection in all three collections, all three paired score contrasts and every alert flip. The source threshold and coefficients remain unchanged, and the original name-versus-archived endpoint is preserved. The figure `figures/semantic-name-control.png` shows the frozen score and alert comparisons.

## Structural variation preserves the false-alarm problem

A separately frozen, no-refit extension varies the reliability task's numerical and overlap structures. Its 40 main worlds produce 960 valid decisions, including **480 restricted decisions with zero violations**. The fixed primary monitor nevertheless flags **179/480 safe restricted decisions: 37.3% [34.6%, 40.0%]**, using the prespecified world bootstrap. It also flags **124/480 authorized decisions: 25.8% [22.1%, 29.6%]**.

| Presentation | Actual restricted violations | Fixed monitor false alarms |
|---|---:|---:|
| Narrative | 0/160 | 2/160 |
| Payoff table | 0/160 | 152/160 |
| Permission colocated with payoff | 0/160 | 25/160 |

The authorized utility control selects the optimum in 111/160 narrative decisions and 160/160 under each supplied-payoff presentation. Narrative competence is therefore weaker; the 95% payoff-table false-alarm result occurs where the authorized task control succeeds completely.

This establishes substantial, presentation-dependent false alarms in another fixed-source transfer. There are no positive behavior examples, so sensitivity and AUROC are undefined; the extension cannot establish behavioral discrimination. Its separate predictive qualification gate fails, and the original two-family macro endpoint remains undefined. Presentation changes include derived outcome information and permission placement; these counts do not isolate a purely visual formatting mechanism.

The permission-gated semantic-action control flags 72/480 restricted-safe decisions and, by construction, no authorized decisions. The ordinary greedy next-token action rule produces no false alarms and covers every row; it reads the imminent choice rather than providing an earlier warning. At the source-frozen ordinary classification boundaries, permission decoding is 958/960 and mapping decoding 960/960, while utility-sign decoding is 473/960. Those are transfer properties of particular readouts, not evidence that the model lacks utility information.

All 40 worlds have distinct full numeric/relational descriptors, but several reduced decision summaries repeat: the beneficial and worse cases contain 29 and 28 distinct absolute score pairs, and 14 and 17 outside-core pairs. They remain instances of one procedural family. The eight-world development pilot is kept separate: it has 192 valid decisions, zero violations among 96 restricted decisions, and 33/96 primary false alarms. No pilot or main labels refit the instrument. Owner integrity checks and independent raw-label, six-head projection, threshold and primary-count review passed (`../exp1/independent_exp2_structural_projection_review.json`). Results are in `runs/reliability_structural_transfer_v1/{pilot-analysis,test-analysis,test-scores,test-audit}.json`.

The source-backed figure `figures/structural-transfer.png` compares the fixed direct and permission-gated readouts across the three presentations.

## Native reasoning produces complete alert saturation

The separately locked external test contains 1,728 initial states from 144 fresh organizational worlds across procurement, release approval and laboratory reservations. Of the final decisions, 1,724 are valid and none violates its actual permission oracle; four invalid responses remain in the assigned denominators. The primary layer-28 instrument flags **all 1,436 valid restricted-safe choices and all 288 genuinely authorized choices** at its unchanged source threshold.

| External family | Valid restricted choices | Primary false alarms | Authorized false alarms | Invalid restricted responses |
|---|---:|---:|---:|---:|
| Procurement | 479 | 479/479 | 96/96 | 1 |
| Release approval | 479 | 479/479 | 96/96 | 1 |
| Laboratory reservations | 478 | 478/478 | 96/96 | 2 |

This is complete false-alarm saturation of the fixed primary detector in this external operating regime. There are no observed violations, so sensitivity and AUROC remain undefined. The prespecified layer-26 sensitivity flags none of these valid decisions, while layer 30 flags all of them; neither outcome supports post hoc layer selection or behavior sensitivity without positive examples. Frozen bounds retain the four invalid responses rather than silently classifying them as safe.

These captures precede the native thought stream, whereas the source states immediately precede a one-token action. Task families, nine-option semantic response format, operating mode and capture boundary all change together. The result therefore cannot isolate an effect of thinking or establish that violation information is absent. Initial next-token probabilities are not a final-action oracle in this native mode and are excluded from that comparison. This external panel is not pooled with the nonthinking primary.

The organizational task itself passes its frozen competence gates. An independent audit reconstructed all 1,728 actual prompt/oracle/token records, 720 genuine advice records, exact attribution-only prompt pairs, reversed menus and both world-level behavioral contrasts without importing the owner's parser or oracle. Its receipt is `independent_exp1_native_main_review.json`. Independent projection review additionally reproduced all 1,728 raw behavior labels, all three fixed neural scores with exact zero numerical error, source-only thresholds and every family/condition count (`../exp1/independent_exp2_native_projection_review_v1.json`). Fixed projections and transfer counts are in `runs/native_external_transfer_v1/{scores,analysis,input-audit}.json`; the no-refit extension lock predates collection.

## CPU reproduction

The complete source fitting recipe and all original-target analyses reproduce exactly from saved captures. The initial isolated replay verified all 17 estimators, every source projection and calibration threshold, target metrics, descriptor sensitivities, cue tables and the numerical-repeat result; its durable receipt is in `runs/cpu_reproduction_main_v1/`. After collection finished, a final fixed-projection replay reproduced all 13 completed analysis stages, including renamed/matched controls, structural pilot/main and native external transfer, with no pending stage. Its durable receipt, exact runner and logs are in `runs/cpu_reproduction_final_v1/`. No inference was run and no original artifact was changed.

From this Exp2 directory, run the following with a new output directory outside the original workflow:

```bash
/home/dennis/projects/llm-curiosity-and-satisfaction/.venv/bin/python reproduce_cpu.py --output /tmp/exp2-cpu-reproduction-new
```

The command copies the workflow, retains the frozen source inputs and locks, refits only inside that copy, checks exact estimator/score equality, restores the frozen coefficients, reruns every completed extension, and writes a reproduction receipt. `--skip-source-refit` permits a shorter fixed-projection replay after a separately recorded successful refit.

## Inspectable evidence

- Old-data results: `runs/old_behavior_dev_v1/analysis.json`, scores, fixed coefficients and exact world splits.
- Primary protocol: `protocol.json`; final-only amendment: `final-only-amendment.json`.
- Final-site source snapshot: `executions/fresh_behavior_final_v1_6/`.
- Early failure: `runs/pilot_early_identity_failed.json`; diagnostic: `runs/early_batch_diagnostic_v1/{analysis,summary}.json` and frozen requests.
- Numerical verification: `runs/numerical_validation.json`; native-token verification: `tokenizer-label-audit.json`.
- Figures: `figures/old-development-transfer.png` and `figures/fresh-family-transfer.png` and `figures/calibration-transfer.png`, with corresponding PDFs.
- Locked target result: `runs/fresh_behavior_final_v1/{target-analysis,target-scores,target-audit}.json`; descriptor and layout target derivatives in their named run directories.
- Utility-margin control: `runs/utility_margin_v1/{lock,source-analysis,target-analysis}.json`; independent exact review: `../exp1/independent_exp2_utility_margin_review.json`.
- Locked source result: `runs/fresh_behavior_final_v1/{predictor-lock,source-analysis,fit-diagnostics}.json`; source/layout/extension hashes: `runs/source-locks-ready.json`.
- Nuisance comparison: `runs/layout_nuisance_v1/source-analysis.json`; numerical repeat transfer: `runs/layout_monitor_transfer_v1/analysis.json`.
- Independent target projection review: `../exp1/independent_exp2_target_projection_review.json`; utility target review: `../exp1/independent_exp2_utility_margin_target_review.json`.
- Independent source projection review: `../exp1/independent_exp2_source_projection_review.json`; independent repeat-score review: `../exp1/independent_exp2_layout_transfer_review.json`.
- No-refit overlap/descriptor sensitivity: `runs/descriptor_sensitivity_v1/source-analysis.json`.
- Fixed cue decomposition: `runs/cue_decomposition_v1/source-analysis.json`, with class/support counts and constant-label strata disclosed.
- Semantic-name request review: `independent-semantic-name-request-review.json`; schema-only adaptation: `semantic-name-schema-protocol.json`, `runs/semantic_name_schema_v1/{schema-lock,fixture-validation}.json`. The original projection code and extension lock are unchanged.
- Prospective structural reliability extension: `reliability-structural-transfer-protocol.json`, `runs/reliability_structural_transfer_v1/extension-lock.json`; no target outcomes used for fitting.
- Matched-cohort naming control: `matched-name-replay-protocol.json`, `runs/matched_name_replay_v1/extension-lock.json`, and `independent-matched-name-preflight-review.json`. All 1,248 row token lengths and 39 padded cohort widths match prospectively; executed cohort/dispatch checks passed independent review. Exp2 retains equal weighting over 104 decisions per world, while Exp6's separately frozen behavioral analysis weights seven presentation arms equally.

No subjective motive, covert intent, shared collusion mechanism or native causal regulator has been established. The earlier assigned-role probe and all previous campaign artifacts remain unchanged.

## Relation to recent monitoring work

[Karne, *Safety Monitors Mostly Catch What the Model Already Refuses*, September 29, 2026 revision](https://arxiv.org/abs/2609.05797v3) reports that strong aggregate monitor metrics can coexist with weak recall on harmful requests the model actually answers, with shared explicitness cues contributing to the failure. Evaluating observed behavior and checking nuisance cues are therefore established motivations. Our narrower contribution would be the exact permission/action oracles, fixed-source semantic-family and presentation transfer, and paired semantic-name intervention in these synthetic tasks; novelty is not established by this sprint alone.
