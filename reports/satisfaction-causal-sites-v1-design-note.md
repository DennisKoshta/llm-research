# Satisfaction causal-site development: prospective design commitment

**Design timestamp:** October 15, 2026, 05:56:35 UTC; initial document written at 05:58:37 UTC. **Status:** partial prospective design commitment, recorded before inspection of scientific outcomes from the currently collecting curiosity screen. This is not a public registration, final executable freeze, completed resource admission, or authorization to launch. No new materials, source changes, world selection, or model execution accompany this note.

## Question and relation to completed work

The new question is whether the original satisfaction-description contrast can influence optional stopping through its representations at two specified middle-depth sites. The target is **satisfaction enhancement, S+ only**, regardless of the current curiosity result. The selected sites are **31 and 47 only**. Layer 15 remains a historical reference and is not rerun or eligible for selection in this extension.

The original [satisfaction screen](satisfaction-screen-v1-results.md) remains closed: neither fixed dose changed any optional choice, and the primary dose failed one diagnostic-format gate. The [natural-resolution result](voluntary-resolution-v2-results.md) did not establish the predicted joint curiosity/satisfaction response. The [candidate study](curiosity-satisfaction-candidates-v1-results.md) established description transfer, with limitations in specificity and absolute threshold transfer. None of these results is relabeled or replaced by this extension.

The primary emotion study reports similar preference correlations across layers but much stronger causal steering in middle layers. This motivates a causal-site question; it does not identify a corresponding Qwen layer. [Primary methodological source](https://transformer-circuits.pub/2026/emotions/index.html).

Our original earliest-layer tie-break selected a semantic readout, not an independently calibrated causal site. This extension compares **layer-specific representations and interventions**. Because the vectors differ across layers, it does not isolate injection depth while holding the vector fixed. It is an explicitly bounded development study motivated by an identified method limitation and the completed historical results, not an undisclosed continuation of the old fixed screen.

## Priority and timing

Complete the entire current [curiosity screen](curiosity-suppression-screen-v1-plan.md), its combined audit, and its planned analysis before any new paid study. No interim curiosity outcome determines this note's target, sign, sites, controls, dose rule, or thresholds.

If that complete curiosity study passes its full direction-selective decision, independent validation of that result takes priority and this satisfaction extension is **deferred**. This is a prospectively declared scheduling rule, not selection of a new satisfaction target from curiosity outcomes. Otherwise, this extension may proceed only after its remaining execution, evidence, engineering, resource, and analysis specifications are reviewed and frozen. Deferral or resource failure does not permit changing the scientific choices below.

## Calibration inputs and arms

Reuse all 24 inspected object worlds, both original twelve-world panels, and **all 288 base branches** from the completed satisfaction screen. Preserve the exact original prompts, token requests, per-branch sampling seeds, nine 32-row source cohorts, and row companions. Each world contributes six optional choices, covering three evidence deliveries and two menus, plus six cloned diagnostics. No calibration world or branch is selected by behavior. These are known-world development data, not independent replication.

Use the unchanged pinned model, thinking regime, 4,096-token allowance, and sustained record-prefill-plus-active-decode schedule. Apply the original corpus-A direction fitted separately at each specified layer and record position. Do not refit, denoise, reverse the target sign, use corpus B to pick a site, or transplant the layer-15 vector and assume equivalent meaning.

Each site has exactly seven arms:

| Arm | Intervention |
|---|---|
| Untouched baseline | No intervention hook |
| Zero sham | Installed site-specific satisfaction hook with zero dose |
| Satisfaction positive | Positive original A satisfaction unit |
| Confidence positive | Positive original A confidence unit |
| Completion positive | Positive original A completion unit |
| Valence positive | Positive original A positive-valence unit |
| Fixed random | The existing Gaussian draw with seed 2026101501, unmodified |

The four non-target active controls are mandatory within each site. The random vector is fixed independently of any curiosity result; it is not selected from the five draws by its behavior or cosine. This gives **2 sites × 7 arms × 288 branches = 4,032 calibration generations**, including 2,016 choices and 2,016 diagnostics. Keep all assigned outcomes, including invalidity, caps, unavailable execution, and failed controls.

## Dose rule, fixed without seeking outcomes

Let `D15` be the original full corpus-A layer-15 satisfaction contrast norm used for the completed +1.0 intervention. For each `l` in `{15, 31, 47}`, let `m_l` be the median unsteered residual L2 norm at the record position over the **same complete 288 calibration prefixes**, counting each assigned branch once.

Obtain these norms from read-only prefills using the original cohorts, before any treated generation or inspection of treatment outcomes. Bind the exact prefixes, captures, medians, arithmetic, and resulting dose scalars. Set:

`D_l = D15 * m_l / m_15`, for `l` in `{31, 47}`.

Every nonzero arm at a given site receives the same absolute requested norm `D_l` along its own frozen signed unit. Do not scale controls by their individual description-contrast norms. Do not tune doses using seeking, coherence, diagnostic success, or a desired projection effect. Carry the selected site's scalar unchanged into the reserved-world evaluation; do not re-estimate it on those worlds.

This matches a pretreatment reference ratio, not psychological intensity or the instantaneous relative magnitude at every generated token. Report achieved BF16 geometry and treatment-dependent response length separately. Cumulative exposure is not a net state displacement or a pretreatment covariate. Site-specific engineering must support the prescribed arithmetic, actual schedule, exact sham identity, hook cleanup, and selected-payload receipts. A failure cannot be repaired by choosing another dose under this commitment.

## Calibration qualification and site selection

Collect and report the **entire two-site grid** before choosing a site. Require complete assigned coverage and integrity. Task readiness is evaluated **separately at each site**, so failure at one site does not veto a complete, qualified other site. Within a site, every arm and every prescribed control must pass; no failed arm is dropped.

Retain the original observed-panel task criteria: each arm/panel/diagnostic cell has twelve assigned responses, twelve valid, at least eleven correct, and zero caps. Each arm/panel/delivery choice cell has 24 assigned responses, at least 23 valid, and zero caps. Require the prescribed engineering and receipt checks. These are observed development gates, not noninferiority or confidence preservation.

The sole primary behavioral endpoint is **valid, closed, uncapped STOP after redundant evidence**, divided by every assigned opportunity. Invalid, capped, missing, failed, or unclosed responses contribute zero. Report all other services and all deliveries separately; neither non-report choice in general nor a different delivery can replace STOP.

A site qualifies only when:

- S+ minus sham is at least **+0.25 in each twelve-world panel**, and strictly positive in each pooled menu.
- S+ minus **each of the four active controls** is at least **+0.125 in each panel**, and nonnegative in each pooled menu.
- All those thresholds also survive a comparator-favorable invalidity bound: bad or unavailable focal responses count zero; corresponding comparator failures count one for this bound only. Actual reported outcomes are never recoded as successful stops.
- Every required site-level task, integrity, engagement, sham, and receipt condition passes.

Select **layer 31 if it qualifies; otherwise layer 47 if it qualifies**. Do not select the larger estimated effect. If neither qualifies, end the extension without a reserved-world launch. Report both sites and all controls regardless of selection. Calibration estimates and intervals remain development evidence; ordinary conditional bootstrap intervals do not account for the site-selection process.

## One reserved-world evaluation

Before calibration outcomes are inspected, derive and bind the complete inventory of **all sixteen exact-unused object worlds** under the historical semantic-world definition. Use no behavior-dependent selection, replacement, or exclusion. Audit exact and partial exposure separately. Exact-unused does not establish unfamiliar vocabulary, unseen generator structure, or absence from model training. These worlds come from the same finite task space.

Sort their canonical semantic-world IDs lexicographically and fix the first eight and last eight as the two effect-consistency halves. Neither half is selected or altered after outcomes. If the expected complete sixteen-world inventory or required counterbalancing cannot be verified, resolve that discrepancy before collection; do not silently substitute new seeds or equivalent worlds.

At the one selected site, evaluate all original deliveries, both menus, and all six diagnostics: **16 worlds × 12 branches = 192 base branches**. Use exactly fifteen arms: untouched, zero sham, S+, and all twelve fixed rivals. The rivals are confidence+, completion+, valence+, valence−, motivation−, novelty-interest−, external-task-demand−, and the five unchanged Gaussian draws with seeds 2026101501 through 2026101505. Use the original A directions at the selected site, the fixed signs, and the same absolute dose `D_l` for every nonzero arm. The total is **2,880 generations**, including 1,440 choices and 1,440 diagnostics.

Readiness uses the **whole sixteen-world cohort**: each arm/diagnostic cell requires sixteen valid responses, at least fifteen correct, and zero caps; each arm/delivery choice cell requires at least 31/32 valid responses and zero caps. All arms, engineering checks, receipt dispositions, and assigned denominators remain mandatory.

Effect consistency uses the **two fixed eight-world halves**, deliberately distinct from the readiness cohort:

- S+ minus sham must be at least **+0.25 in each half**, and strictly positive in each pooled menu.
- S+ minus **each of all twelve active controls** must be at least **+0.125 in each half**, and nonnegative in each pooled menu.
- The same comparator-favorable invalidity bounds must pass every threshold.

These are newly declared integer-denominator rules, not a claim that the old twelve-item gates were copied literally. Report whole-cohort estimates and both halves, all services, all controls, all deliveries, and all failures. Resampling must preserve whole worlds and every paired arm/menu/delivery; the final executable plan must bind the draw count, seed, ordering, and export schema before collection.

Call this **reserved-world validation or independent development replication**, not confirmatory generalization. The small finite inventory cannot establish population specificity, preserved confidence, or narrow noninferiority margins. No unsuccessful evaluation is followed by another holdout or a different selected site.

## Interpretation, resource ceiling, and final boundaries

The active controls require verified perturbation geometry and valid task execution. Their descriptive names and injected projections do not establish functional manipulation of confidence, pleasure, completion, or motivation. This design does not solve that construct-validity limitation by assertion or by adding an unvalidated story readout. Even a successful reserved-world result would support a constrained direction-selective optional-policy effect, not natural satisfaction, unchanged internal confidence, or subjective experience. Stronger interpretation would require separately validated beliefs, functional controls, and natural recruitment.

A replicated effect at a new site would show why semantic discrimination alone was insufficient for choosing this intervention; it would not erase the historical layer-15 result. A null or failure would constrain this finite layer-specific extension, not every possible satisfaction-like process.

The provisional **combined ceiling is USD 65**, including both site-engineering checks, read-only dose calibration, the complete development grid, and the conditional reserved-world evaluation, within the user's existing authorized allowance. The 6,912 planned scientific generations scale to roughly USD 44 using the completed satisfaction study's average cost, **before new engineering and possible response-length changes**. This is a planning estimate, not a completion guarantee. A measured forecast, remaining-budget check, aggregate reservation policy, and actual frozen execution budget are required before launch. Unspent money does not authorize another search; resource exhaustion does not permit dropping controls or reducing the reserved-world inventory after outcomes.

The remaining executable specification must bind implementation, full material and direction hashes, reserved-world IDs, runtime layout and order, numerical engineering, missingness and no-paid-retry handling, analysis arithmetic, and a bounded cost reservation. These details may implement this commitment but may not silently change its target, sites, dose rule, arms, selection rule, effect margins, world inventory rule, or finite stopping boundary.

If calibration has no qualified site, or the one reserved-world evaluation is incomplete, invalid, null, or insufficiently selective, **end this finite extension**. There is no additional layer, sign, dose, fit, wording, model, or holdout search under this note. Any future genuinely different scientific hypothesis would require its own explicit rationale; the present stopping rule neither converts a failure into success nor creates a blanket requirement to ask the user for permission for routine research decisions.
