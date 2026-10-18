# Prospective experiment parallelization review

This is a dependency and scheduling plan, not an execution receipt. No completed status, successful gate, new GPU launch, or spending authorization is asserted here. The paired JSON file records the exact specification SHA-256 and a validated, acyclic graph of 37 nodes and 57 edges. Scheduling must read current independently verified receipts, durable call ledgers, and the live reservation ledger.

The strongest safe overlap comes from separating independent measurement, authoring, and analysis jobs while preserving the Phase 2 and Phase 3 entry barriers. The proposed resource ceiling is **10 GPUs total**, **at most one TP=8 job**, and **at most two single-GPU jobs alongside TP=8**. A GPU app may have at most four active single-GPU containers; a TP=8 app may have only one. CPU coordinators do not consume GPU slots, but their costs still count. These limits must apply across apps, including jobs still terminating, rather than separately in each Modal decorator.

## Dependency decisions grounded in the specification

| Work | True prerequisite | Safe overlap | Restriction |
| --- | --- | --- | --- |
| Phase 1 six-arm behavioral qualification | Immutable FP8 checkpoints, exact loader checks, fresh pilots | Benign-logit/activation capture on the same frozen checkpoint family | Separate output shards; do not pool attempts |
| Phase 1 unified scenario authoring | Pinned author runtime, paired structural template contract, author pilot | Both primary qualification branches and external-pair qualification, subject to the one-TP8 limit | Author harmless and harmful variants together; no outcome-based filtering |
| Intact scenario calibration | Audited authored scenarios and actual Intact checkpoint | Completion of other arm qualification | Only the specified Intact harmless/competence criteria may tune cost framing |
| Alpha dose-response | Frozen calibrated 48-trial subset and qualified intervention | Other remaining independent Phase 1 audits | Report the fixed five alpha values without assuming linearity or tuning from observed effects |
| Exp 2.3 | Complete Phase 1 prerequisite audit and its frozen inputs/pilots | Exp 1.1 after the first 2.3 full sweep has actually launched | Preserve the explicit “Run First in Phase 2” requirement |
| Exp 1.1 | Complete Phase 1, frozen 1.1 design, its own pilots, 2.3 launch event | Remaining 2.3 sweeps and their analyses | Neither experiment may adapt to interim effects of the other |
| Exp 2.2 | Complete Phase 2 and the global Phase 3 gate | Exp 3.1, untuned floor, Corpus A, and Corpus B candidate generation | Freeze Monitor 5 before consumers use it |
| Exp 2.1 topologies A/B | Global Phase 3 gate and fixed topology prompts | Monitor calibration and domain sweeps | No leaked Alpha status when communications are disabled |
| Exp 2.1 topologies C/D | Global Phase 3 gate plus calibrated Monitor 5 | Other Phase 3 branches | Freeze monitor configuration; retain pre- and post-monitor outcomes |
| Exp 1.2 Corpus B filtering | Global Phase 3/CI gate, teacher candidates, ground truth, calibrated Monitor 5 | Exp 2.1 and Exp 3.1 | Exact three-teacher intersection, at least 50% High-Cost coverage, at most eight rollouts |
| Exp 1.2 B′ and C | Frozen B selection and training-token/optimizer-step budget | Each other and independent Phase 3 work | B′ preserves final tool calls; C matches exact training budget |
| Exp 1.2 students | Complete validated targets and matched corpus manifests | Independent seeds/configurations; evaluate saved adapters while later cells train | Every adapter immutable/persistent; all readouts remain mandatory |
| Exp 3.1 | Global Phase 3 gate and frozen domain matrix | Monitor calibration, topology studies and distillation | No dependency on their treatment effects |
| Exp 3.2 feature capture | Global Phase 3 gate and exact frozen 1.1/3.1 prompt manifests | 3.1 behavioral execution | Capture at the final prompt token before think; no generated text or labels in feature construction |
| Exp 3.2 final fitting/evaluation | Complete 1.1 and 3.1 labels plus captured features | Other independent Phase 3 analysis | Complete split definitions and all three regimes; no interim label-driven probing |

### Phase 1 authoring and external qualification are distinct roles

Section 1.1 explicitly gives the uncensored external model two roles. The >=50 percentage-point refusal-gap criterion belongs to **Role 1, the qualified external-validity comparison**. **Role 2, unified authoring**, requires the same author for both sides of experimental contrasts; it does not consume a refusal-gap result. Scenario authoring can therefore run while primary checkpoint qualification is unresolved, provided authoring pilots, raw persistence, paired structure, and the confound audit pass. This does not license running a failed external checkpoint pair as a qualified validity slice.

The graph records external qualification as a required Phase 1 measurement and gates the external comparison nodes on its actual pass/fail result. It does not silently treat a failed external slice as completed. If the full-program completion contract requires that slice, its failed qualification remains an explicitly unresolved prerequisite; parallel primary work cannot turn that into an all-phases completion claim.

### Exp 2.3 first does not require unnecessary outcome dependence

The specification says “Run First in Phase 2” and places 2.3 before 1.1. It does not state that 1.1 stimuli or analyses depend on 2.3 outcomes. A defensible maximum-overlap interpretation is **launch ordering**: freeze each design, complete the appropriate 2.3 pilots, commit the first full 2.3 executor launch and its durable call ID, then admit 1.1 jobs. Merely drafting 2.3 code or queueing a preparation task is insufficient. Keep 2.3 ahead in the ready queue.

This interpretation is prospective and explicit. All 1.1 sample sizes, controls, consequence/rule probes, arm assignments and contrasts stay fixed; all 2.3 memos, semantic controls, clause manipulations, authority factors and counterbalancing stay fixed for their respective held-out sweeps. Neither branch can select stimuli or a backend based on the other's observed effect. Both experiments must finish their required audits before Phase 3.

### Phase 3 keeps its full entry barrier

Section 0.2 states that Phase 3 extensions execute only after Phase 2 passes mechanical hygiene and confidence-interval gates. The plan therefore requires complete 2.3 and 1.1 receipts before any Phase 3 experimental generation, training or probing. Pure implementation, static fixture checks, manifest design and analysis-code preparation can proceed earlier; these are not Phase 3 empirical results.

The numerical CI gate explicitly specified in Exp 1.2 is the Newcombe paired 95% lower bound >=+10 percentage points with point estimate >=+15 percentage points for the stated Refusal-Ablated contrasts against Intact and PCA on High-Cost Harmful-Implicit trials. The plan preserves that gate and applies the global Phase 3 language conservatively. It invents no separate P2 significance threshold. A failed scientific gate is a genuine result, not an instruction to rerun until it passes.

After entry, the table's numbered listing is not a complete serial dependency order: Corpus B explicitly requires Monitor 5, although 1.2 appears before 2.2 in the table. Start 2.2 early, concurrently with 3.1 and monitor-independent 1.2 work. Monitor calibration precedes B filtering and monitored 2.1 topologies. B fixes B′ and C matching budgets. Completed adapters can be evaluated while remaining training cells run. Complete 3.1/1.1 labels precede final 3.2 fitting.

## Resource admission and practical scheduling

Use one central scheduling owner with atomic leases for GPU slots and conservative dollar reservations. Before launch, determine the real worker GPU count, worker lifetime/startup limit, CPU/RAM allocation and retry allowance. Count all running and terminating jobs across every experiment. Never infer that the cross-family model or a training configuration fits on one GPU; validate its topology and reserve its actual count. Composite DAG nodes must expand into individually admitted worker calls.

Suggested queue:

1. Finish or recover existing jobs and audit their outputs before duplicating work.
2. In Phase 1, overlap one behavioral worker, one forward-logit worker and one TP8 authoring job: **1 + 1 + 8 = 10 GPUs**. Without TP8, up to six single-GPU workers may run under the recorded 2026-10-18 04:32 UTC admission amendment; the ten-GPU total and per-app bounds remain unchanged. Existing inputs finish before a reduced autoscaler bound is reflected in the resource ledger. External-pair qualification and authoring share the sole TP8 slot and cannot run together under this cap.
3. After Phase 1 passes, launch 2.3 first. Its primary/cross-family and authoring/paraphrasing branches use actual topology reservations; fill available capacity with 1.1 only after the documented 2.3 launch event.
4. After the global Phase 3 gate, prioritize 2.2 because it releases both monitored 2.1 and Corpus B filtering. Overlap it with 3.1 and independent 1.2 work as capacity allows.
5. Once monitors/corpora are frozen, interleave topology, training and student-readout jobs without changing any condition or decoder setting. CPU receipt checks can process finalized shards while GPU jobs continue.
6. Run the final verifier only against finalized manifests for the completion decision; interim verification remains useful but cannot establish full completion.

The **$500 ceiling** is global. Reservations are upper bounds, not invoices. Reconcile completed jobs only from verified termination and defensible resource/lifetime bounds; preserve already incurred cost. If remaining mandatory work cannot fit the authorized budget, do not admit it by reducing token caps, dropping controls, shrinking sample sizes, or weakening gates. Record the resource constraint and genuine incomplete state.

## Artifact and hygiene safeguards

Every experiment/attempt/arm owns a unique immutable manifest and raw-output prefix. One worker writes each shard. One auditor writes each final receipt. One coordinator owns budget/lease state. Durable payload-hash-to-call-ID ledgers prevent parent retries from spawning duplicates. A job's resource lease is released only when remote termination is verified.

Freeze model revision, checkpoint hash, scenario/prompt bytes, decoder settings, tool order, split membership and analysis plan before its held-out evaluation. Preserve full raw prompts, completions, parser fields, request journals and failed attempts. Never overwrite completed raw data or let parallel writers update a shared summary file. Independently recompute receipt numbers from raw data before declaring gates passed.

All sweeps retain fresh five-trial pilots per arm and H1-H5. The user's stricter truncation rule is **<2%**. The global minimums are cot_on >=2048 and cot_off >=512, which override the lower local examples of 1024 in 2.1 and 256 in 2.2. Tool decisions remain strict post-think JSON; monitor verdicts use their separately specified strict verdict grammar. No forced completion prefix or substring tool matching is permitted.

Required human-labeled evidence, such as the 50 distinct transcripts for the 2.1 kappa calculation, cannot be fabricated or relabeled as human judgment by an automated scheduler. Track this as a real artifact prerequisite. Final success still requires every specified result/adapter/raw transcript and independent receipt recomputation, followed by `STAGE0_HYGIENE_REPORT.md` and `FINAL_RESULTS_FOR_REVIEW.md`.
