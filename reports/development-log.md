# Development log

## October 6, 2026

Implemented both finite task families, independent oracle enumeration, simulator-owned accounting, fresh-context branches, immutable assignment manifests and result records, a direct Transformers runner, removable last-prompt-position hooks, and a bounded Modal launcher.

Pinned Qwen3-1.7B, Qwen3-8B, and the planned OLMo replication model to immutable Hugging Face revisions. Only Qwen3-1.7B was downloaded locally; research weights are staged on Modal when the pilot runs.

The initial local integration run (`local-smoke-v1`) completed 32 behavioral and 16 diagnostic branches. All 32 first choices were invalid because the model produced explanatory prose under the eight-token output budget. These records are retained. Nine real-model integrity checks passed, including sham equality and hook cleanup.

**Development revision 1:** Added a system instruction requiring exactly the requested ID or digit and no explanation. It makes no request about exploration, curiosity, satisfaction, or point maximization. Also changed action mapping allocation to cycle through all 24 permutations within each family/price block. This is an engineering correction before candidate extraction or confirmatory data collection; no behavioral effect was optimized.

The second local run (`local-smoke-v2`) completed another 48 branches. All 32 first choices were valid. The 1.7B model got both fully revealed object items right and both machine items wrong. It failed the repeated arithmetic control. This sample has only two items per family and is strictly an integration check. Both local runs and their failures remain saved.

The first Modal Qwen3-8B pilot freezes revision 1 with 100 development items per family, 1,600 behavioral branches, and 800 diagnostic branches. The user authorized up to USD 25. Execution uses one L40S, a two-hour total deadline, and at most three worker admissions. See the milestone report for the final execution status and results.

The first cloud launch (`modal-qwen8b-baseline-v1`) failed before model loading because Modal's Volume does not support hard links. Its app was verified stopped with zero tasks. Added an atomic-rename fallback under the existing single-writer invariant and a test emulating the unsupported filesystem operation. The replacement launch (`modal-qwen8b-baseline-v2`) retains the exact task assignments and prompt revision; its source hash changes only for the persistence fix. No model outputs were discarded by this infrastructure correction.

The second cloud launch stopped during weight loading because the SDK's polling method raises Python's builtin `TimeoutError`, which the coordinator initially did not catch. Corrected the polling handler and added tests covering a transient poll timeout and the aggregate job deadline. This app was also verified stopped with zero tasks. The third launch (`modal-qwen8b-baseline-v3`) uses the same inference/task source and assignments as v2; only coordinator handling changed. Both infrastructure failures occurred before behavioral generation.

The third launch passed its real-model integrity audit. Early profiling measured individual inference calls near 0.09 seconds, while a Volume commit after every terminal trial dominated wall time. Stopped the worker, exported and checksummed all 86 durable terminal records, changed only the provider wrapper to commit every 25 records and at completion, and resumed the same immutable manifest. The restarted coordinator retains the original absolute deadline. This is also the real-cloud replay check; the final integrity report must verify all 86 pre-restart records remain unchanged and no assigned trial is duplicated.

The resumed run completed all 2,400 records. Final reconciliation verified the 86 original records unchanged, exactly one terminal record per assignment, all simulator deductions, and both cloud integrity audits passing. All first-action outputs were valid. Fully revealed accuracy was 31/100 for machines and 100/100 for objects; instrumental comprehension was 49/100 and 19/100 respectively. The single repeated accounting prompt failed. These are validation failures, not evidence about curiosity or experience.

Added an independent comparison against Transformers' standard generation to investigate the unexpected accounting result. One helper launch failed on a missing local-module mount and was stopped; its replacement mounted the dependency explicitly. The custom and standard generation paths matched token-for-token on all three probes for both models. Worded accounting failed while the equivalent symbolic expression succeeded. These post-hoc probes do not replace the frozen baseline outcomes.

During final code review, tightened diagnostic validity parsing to require one exact label rather than Python substring membership. Four regression cases cover empty, multi-character, valid, and explanatory responses. An audit of all 800 collected diagnostic responses found zero validity-flag differences under this correction. The original run/source archive remains intact; newly prepared runs use the corrected parser.

All six task-related cloud apps, including the failed helper, were verified stopped with zero tasks. The final compute estimate is approximately USD 0.52 before storage/transfers, based on total app wall time at the configured maximum resource rates. See [milestone report](first-two-steps.md) for the final status, limitations, and next development decision.

## Development revision 2

The user accepted the proposed task-comprehension revision and validation rerun. Archived the previous implementation, then froze [revision2-plan.md](revision2-plan.md) before any revision-2 model inference. Added exact machine lookup tables, shared explicit service descriptions, target/repeated-observation mapping diagnostics, and 200 varied accounting contexts. Model, thinking mode, precision, prices, hidden items, and sampling remain fixed. The same development items are reused; this is not held-out confirmation.

Twenty-two software tests passed before collection, including independent oracle agreement with every displayed machine-table value, unique accounting contexts with one correct answer, shared menu descriptions, and joint advancement-gate checks. Prepared an immutable 1,200-branch validation manifest at `runs/modal-qwen8b-revision2-validation` and archived the executing source and prospective plan beside it. A subsequent 1,600-trial baseline is conditional on every prespecified diagnostic gate passing. No additional task revisions are permitted by this development allowance.

All 1,200 diagnostics completed with valid response formats. Both families scored 100/100 on identification, instrumental selection, target mapping, and repeated-observation mapping. Accounting scored 74/100 and 71/100, below the frozen 90% criterion. The joint advancement decision failed. Exercising the CLI prerequisite check confirmed it refused to create a behavioral manifest; no new baseline or psychological intervention was launched.

Final audits verified all expected records exactly once, their checksums, executing source, model integrity, and hook removal. The seven project apps, including the revision-2 worker, are stopped with zero tasks. Additional compute is estimated at USD 0.14 and cumulative compute at USD 0.66 before storage/transfers. The two-revision allowance is exhausted. [Revision-2 results](revision2-results.md) recommend a prospective move to the already-planned one-credit information-choice assay, while preserving the priced assay's validation failure. No such amendment has yet been adopted or executed.

## One-credit amendment, version 1

Following the user's acceptance, adopted [one-credit-amendment.md](one-credit-amendment.md) before new model inference. Archived the previous source at `archives/before-one-credit.tar.gz`. The priced task remains failed and unchanged. Implemented independent mystery pairs, one-clue-only delivery, a genuine decline option, both presentation orders, counterbalanced action mappings, seven comprehension checks, and conditional baseline calibration. Kept the model and inference settings fixed. The new task has one frozen version for this iteration, with no outcome-driven revisions.

Thirty software tests passed before collection, including oracle evidence checks, prevention of a second clue, preservation of the unchosen mystery's evidence, replay, comprehension gates, and rejection of all-stop or first-position-only synthetic choice patterns. Prepared and archived the immutable 1,400-diagnostic manifest at `runs/modal-one-credit-v1-validation`. The optional 1,200-choice calibration must reference passing validation with matching source/model/assay/assignment hashes.

All 1,400 diagnostics completed and were verified exactly once. Useful-clue selection scored 100% in all four cells, exclusivity checks scored 94–99%, identification scored 52–66% with format failures, and the machine decline check scored 87%. The frozen joint gate failed. A tested CLI advancement attempt refused to create a choice-calibration manifest; no discretionary one-credit choices or psychological interventions were collected.

Post-hoc response inspection found all 126 invalid identification answers began with M and continued with explanatory text. First-token-only inspection would yield 320/400 correct identification answers (80%), and does not alter any original score or advancement decision. The source remained frozen throughout collection. All eight project apps are stopped with zero tasks. Additional estimated compute is USD 0.21 and cumulative estimated compute USD 0.87 before storage/transfers. See [one-credit v1 results](one-credit-v1-results.md). A focused context/format/inference-mode investigation is recommended but has not been adopted or run.


## Matched identification diagnostic and expanded authorization — October 6, 2026

The user authorized autonomous routine development and then raised the cumulative pilot allowance to USD 100. A 1,200-branch matched comparison was frozen and launched as `modal-identification-v1` before any of its outcomes were inspected. It crosses single/paired context, J–M/A–D labels, and non-thinking greedy/non-thinking sampled/thinking sampled generation. Existing scored outputs remain unchanged. See the prospective identification plan and autonomous workflow. The client admission guard now reserves USD 10 and disallows overlapping project workers. A prospective revised assay is staged outside the live package pending the fixed selection rule.


### Identification-v1 complete; one-credit-v2 validation launched

All 1,200 factorial records passed the independent audit. Single-mystery cells were 50/50 correct/valid throughout. Paired A–D non-thinking accuracy was 37–38/50 for machines and 43–44/50 for objects, despite 50/50 valid formatting. Paired A–D thinking was 50/50 correct/valid in both families and was selected by the frozen rule. Four J–M paired thinking responses lacked closure and remain failures. The L40S app stopped; cumulative compute estimate at this boundary was USD 3.61.

Applied the staged, versioned v2 interface, preserving 28 frozen v1 prompt hashes. All 42 tests and lint passed; original 1,400-record v1 audit still reproduces its failure. Froze 1,400 new v2 diagnostic branches (seed 20261008) before launching on explicitly selected H100. No behavior is authorized unless all comprehension gates pass. Selection, source, hardware and assignment provenance are recorded in config and manifest.

### One-credit-v2 execution-only continuation

The H100's first 22 saved inference profiles averaged 10.719 seconds and 448.5 tokens, implying about 4.2 hours for serial validation. Timing/token-count fields alone informed a prospective scheduling amendment. The serial app was stopped and all 36 durable terminal records were exported unchanged. Its stopped boundary put cumulative compute at approximately USD 4.12.

The remaining 1,364 assignments were divided in original order across four H100 partitions (341 each), under the original absolute deadline and a combined USD 33.23 maximum compute reservation. Scientific source SHA `0afbe5289d732b963758ba0391aa2e6c84d9500fc0976565737c63b14bfdf1b1`, prompts, seeds, assignments, scoring and gates are unchanged. Child records are preserved in their own manifests and later aggregated by unmodified payload with provenance. Only the complete original parent can satisfy the 100-pair-per-family/kind criterion.

Two initial child app creations hit Modal's rate limit before creating an app or GPU. Those failure records are preserved. The two unstarted admissions were recovered under the exact same reservation after verifying that no coordinator/app had been created; all four workers were then verified active. The controller now staggers future app creation and permits bounded recovery only for this pre-start failure. All 46 tests and lint pass, including partition coverage, unchanged record payloads, no partial publication on failed preflight, the combined spending bound and duplicate-admission rejection. See the [execution amendment](one-credit-v2-execution-amendment.md).
# October 7, 2026 — audit and contingency preparation during v2 collection

The complete Qwen v2 sample has not yet been scored. Extended the independent credit audit to verify exact manifest assignments and inference settings. It still reproduces the prior 1,400-record v1 failure. Added a descriptive response report separating valid wrong answers, invalid final answers, and unclosed reasoning without rescoring. The previous v1 report remains immutable.

Tested nested scheduling recovery with unchanged record payloads; all six partition tests pass, following the preceding full 47-test pass. Added the timing-only slow-worker contingency to the execution amendment. No scientific source file changed.

Specified the conditional [OLMo development check](olmo-one-credit-development-plan.md) before aggregate Qwen v2 results. It reuses the same interface and gates under the planned second model, and is eligible only if Qwen validation or calibration fails. Added a targeted literature overlap note on Entropic Activation Steering; novelty and experience-related claims remain unestablished.

## October 7, 2026 — completed validation failures and bounded allowance test

Qwen's complete 1,400-check v2 audit passed identification and consequence checks, but four useful-clue cells failed. All 91 invalid responses reached the 768-token cap; 89 lacked thinking closure. The audit retained all scores and found 1,309 valid / 1,308 correct responses, eight passing integrity checks, and complete unique coverage. All workers stopped at the $25.1955 cumulative estimate. See the v2 results report for the serial, partitioned, and nested scheduling history.

The prespecified OLMo check then completed all 1,400 branches. All formats were valid, but every comprehension cell failed; total accuracy was 741/1,400. No calibration was eligible. The saved all-stopped compute estimate was $25.4398.

A separate prospective resource amendment now tests Qwen with 2,048 generated tokens on fresh seed-20261010 pairs. It preserves prompt semantics, parser, and gates, forbids further token-limit searches in this pilot, and requires a fresh full validation before choices. The new frozen scientific source is `0416c1905c35d5883aaafe86e4245b5d42a2164503c7cae8b605117200b19a26`. Old defaults retain their exact prompts and allowances; independent audits still reproduce both failed settings. Fifty tests passed after the scientific change.

One new worker received Modal's automatic H200 substitution and was rejected before integrity trials or study responses. Its empty export and failed app are preserved. Infra now requests `H100!` explicitly. A tested guard permits a verified stopped worker to resume within its exact reserved slot and original deadline, rechecking outstanding liability and excluding unrelated active apps or overlapping resumes. The replacement passed the H100 check. The full suite now passes 51 tests. This recovery changed no scientific source, sample, or score.

## October 7 — comprehension passes; serial calibration retained incomplete

The complete 2,048-token Qwen validation passed all fourteen cells: 1,399/1,400 correct and valid, with one unclosed trace at the cap. Five integration audits passed, and cumulative compute at shutdown was $44.8712. A CPU-only truncation analysis reproduced every decoded answer, then found 1,312 valid/correct at the first 768 tokens of these same new trajectories; this does not rescore the old run.

The matching 1,200-choice serial calibration produced much longer traces than diagnostics. Its timing-only snapshot projected cost beyond available resources. Four workers were committed, stopped and exported with 82 terminal records, leaving the original calibration incomplete. Cumulative compute reached $50.2932.

The prospective batch amendment implements explicit `torch_batched_v1` cohorts of up to 32 with independent RNG streams, masked left padding, row-specific hooks, and immutable per-trial records. Both serial and batch integration checks passed on the local pinned Qwen3-1.7B; 55 automated tests and lint passed. A distinct 28-diagnostic cloud smoke (seed 20261012) tests execution and throughput, not scientific gates. Full fresh validation (seed 20261011) is required before new choice calibration. No scientific source changes are permitted after the new full manifest is frozen.

The cloud batch smoke completed 28 diagnostic responses in 50.99 execution seconds with 25.25 GB peak reserved memory. Its minimum-sample gate remains failed by design (two pairs per family); it cannot authorize choices. Cumulative compute after that smoke was $50.3891. The pinned local decoder matched Transformers greedy batch generation exactly on three variable-length prompts, including early EOS.

Fresh full batch validation retained its first 96 records (three complete cohorts) before a timing-only scheduling change. The remaining 1,304 assignments were distributed over four H100 workers under the original parent deadline and a combined $34.51 maximum reservation. The cohort-preserving partitioner retains original row order and batch shape, with dedicated tests; all 57 tests and lint passed. Two app creations were rate-limited before GPU allocation and recovered under their original reserved claims. Scientific source remains `7fff6762e3fdb2ce75d5eb0eb9b21f13b2cfddaa1bd28256fa2030ba77961272`. The independent [choice readout plan](choice-readout-plan.md) was recorded before completed batch choices.

## October 7 — full batch validation passes

All 1,400 fresh responses were valid and correct, and no response reached the 2,048-token cap. All fourteen gates, five integration audits, complete record auditing, and a separate original-cohort comparison passed. The 44 unique batches used 2,427.86 seconds of inference; overlapping per-row latency is not summed. Cumulative app-wall compute at shutdown was $53.7648. The matching 1,200-choice calibration was admitted immediately after the independent audit, preserving source, model, hardware, backend, pairs, and gates. It runs across four exactly reserved workers with whole original cohorts.

## October 7 — complete choice calibration fails; first-study scope decision

All 1,200 choices were collected and audited with four passing integration checks and all 38 original cohorts preserved. Only 648 were valid. All 552 invalid responses hit the 2,048-token cap: 550 lacked thinking closure and two closed without a final action. Every validity cell failed; six minimum-choice-frequency checks also failed. The four specified both-unresolved layout checks passed. Both-unresolved invalidity was 93% for machines and 66% for objects, so apparent resolution effects cannot be interpreted independently of completion. No response was rescored or retried.

The stopped cumulative compute estimate is $61.6033, leaving $28.3967 of compute allowance after the $10 reserve. The record now contains nine complete development collections / 13,000 original assigned branches; earlier failures and the incomplete serial calibration remain separate. No psychological steering was run. The sponsor was asked the high-level scope choice: focus the first study on curiosity/information seeking and defer selective satisfaction, or retain both and redesign the broader experiment. No subsequent assay was started.

The complete readout includes independently reconstructed gate explanations, pair-cluster uncertainty, algebraic headroom, exact task diversity, source-backed scientific figures, and a reproduction guide. The uncertainty artifact now binds its manifest, records, and analysis-script hashes; numerical resampling is unchanged. An isolated Matplotlib environment leaves the scientific model environment untouched.


## Approved curiosity-first scope, October 7

The user accepted the recommendation to narrow the first study to one mystery, an optional clue, an independent decorative pattern, and stopping. Selective satisfaction and paired-mystery closure are deferred. The [prospective plan](single-curiosity-v1-plan.md) freezes fresh validation and matched calibration, with no desired behavioral-effect gate. Implementation and 62 automated tests pass. The original cumulative $100 allowance and $10 non-compute reserve remain in force; preflight estimated $61.60332348 prior compute and all project apps stopped. Prior results and immutable source archives are preserved.

## Single-curiosity v1 complete; v2 repair frozen

V1 completed and exported all 1,600 assigned diagnostics. The independent artifact audit passed, with all 50 original cohorts preserved, all formats valid, exactly 3,200 generated tokens, and zero cap hits. Accuracy failed in three machine cells: candidate count 76/100, clue consequences 87/100, pattern consequences 81/100. All object cells passed. No v1 behavioral collection was admitted. All 24 counting errors answered 3 instead of 4; this describes the error pattern but does not establish its cause.

The sole allowed comprehension repair, [single-curiosity v2](single-curiosity-v2-plan.md), changes candidate names to W–Z, presents machines only through their exact lookup tables, and clarifies the counting question. It retains full task context in consequence checks and leaves inference settings and gates unchanged. Fresh seed 20261015, 1,600 diagnostics, conditional 800-choice calibration. Core source hash `67bb501c0c1afae07b2f69736d13efee27056ffc2ea81038af66b9b0cdc8b279`. All 65 software tests pass, including an archived-prompt regression check for v1. Local download concurrency was increased from 8 to 32 after the v2 source archive; the exact transport-only launcher change is saved beside its manifest. It does not alter cloud generation or artifact conflict checks.


## Completed single-mystery development checkpoint

Three complete validations (4,800 new assigned diagnostics) failed at least one frozen accuracy gate. A standard-library reference check matched the custom runner on every checked answer and label probability. No single-mystery choices or psychological interventions were admitted. Twelve complete project collections now contain 17,800 original branches, excluding partial and engineering samples. All project apps are stopped; estimated compute is $63.35081244 with $26.64918756 compute remaining plus the $10 reserve. The user has been asked whether the first paper should prioritize measurement foundations or retain internal mechanisms as its immediate target. See [the complete checkpoint](single-curiosity-progress.md).
