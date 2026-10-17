# Natural donor projection: prospective engineering specification

October 14, 2026. This is a code-based design audit and implementation specification. No donor executor is implemented by this document, no scientific schedule is frozen, and no target-relevance outcomes were inspected. It does not advance the layer-47 candidate, authorize a different site, or bypass the behavioral, representation, calibration, check-block, confirmation, or budget gates in [the prospective causal plan](/home/dennis/projects/llm-curiosity-and-satisfaction/reports/target-relevance-causal-prospective-plan.md).

**Deferred infrastructure.** The user has requested a return to the original Pain Axis-inspired curiosity/satisfaction and inner-experience motivation. This specification is retained for possible reuse, not a proposal to launch or expand the information-appraisal intervention program now. Its engineering details do not supply independent curiosity/satisfaction candidates or voluntary information-seeking evidence.

The preferred design is **row-matched donor capture followed by a fresh natural continuation**. An unpatched baseline prepass captures each recipient in its scheduled row. Three additional prepasses permute the same 32 source prompts so that each counterfactual donor is captured in its recipient's row. Scientific conditions then reuse the unchanged recipient layout. Every pass starts with a fresh cache. Only an observed scalar component is transplanted; no donor text, reasoning tokens, full hidden vector, or cache is supplied to the recipient.

## Current implementation and required boundaries

The inspected implementation already provides useful components:

- `StateBackend.forward` captures layer outputs before the hook applies a patch, supports `project` and `matched_add`, records requested and achieved geometry, and removes hooks before sampling the continuation. `max_new_tokens=1` and 4,096 both use a cache-enabled initial forward; zero would take a different `use_cache=False` path.
- `replay_information.intervention` resolves fitted and random directions and reference axes. Its existing projection modes can be reused only after natural-stage validation and provenance are extended.
- `natural_information` binds exact original prompt tokens, source assignments, paired seeds and generation allowance. The current natural manifest validator explicitly accepts **additive interventions only**. Its executor has no donor-prepass artifact or resolved-donor schedule.
- The natural preparer packs condition-major cohorts, preserves source rows across arms, randomizes condition order prospectively, and partitions whole source blocks. Count worlds use 96-source superblocks where necessary.
- The natural auditor checks exact prompt reconstruction, cohort membership, source bindings, generated-token integrity, baseline/zero-sham identity and prior additive engineering. It does not yet certify natural donor provenance or the equality of a projection arm's runtime scalar and its matched controls.

Therefore, lifting the additive-only assertion is insufficient. A donor implementation needs a new explicitly versioned capability, prepass records, runtime-resolution audit and separate production-model engineering proof. Existing additive and old 2,048-token artifacts must remain readable under their original rules. A passing additive engineering receipt must not satisfy a donor-capability requirement.

The current scientific source gate remains strict global validation. A one-token donor prepass is a measurement artifact, **not a behavioral validation source** and not a reason to declare a generation-regime exception. The separately proposed self-contained confirmation baseline route needs its own admission implementation; it is not introduced through donor support.

## Why donors must be captured in recipient rows

Condition-major scientific arms compare the same recipient prompt in the same physical row and with the same companions. This addresses a cross-row baseline/sham comparison. It does not establish that a counterfactual donor captured in another row has a numerically interchangeable state.

The current cell order ties `(u,q)` to a position within each four-row world. A native-row donor difference can therefore contain a row-layout numerical contribution. The stronger design is preferred prospectively even if an engineering sensitivity check finds that contribution small; do not choose between designs based on favorable behavioral effects.

For one source block and readout/template group, let its fixed recipient list be `R[0:32]`, containing eight complete four-cell target worlds. Define the following bijections within each world:

| Prepass role | Prompt placed at recipient row r | Meaning |
|---|---|---|
| `baseline` | `R[r] = (u,q)` | Unpatched recipient and self target |
| `opposite_relation` | `(1-u,q)` | Primary opposite-gain donor; offered query fixed |
| `opposite_query` | `(u,1-q)` | Secondary opposite-gain donor; table fixed |
| `same_gain` | `(1-u,1-q)` | Both factors changed; target gain preserved |

Each role is a permutation of exactly the same 32 source IDs. Thus batch size, source-length multiset and maximum padded width remain identical. Every donor retains the same family, base world, readout kind and rendering as its recipient. The donor has its **own** exact prompt tokens, relative record position and prompt length. Its physical record index is `padded_width − donor_prompt_length + donor_record_position`; it must not be forced to the recipient's record index. The designated target and all four oracle cells must be reconstructed independently.

The execution seed in every role prepass is the recipient row's seed. Preserve the donor's original assignment and source seed separately; an explicit execution-seed override is part of the prepass recipe, not a mutation of the donor source record. Initial hidden states precede sampling, but recording the seed rule prevents accidental RNG carryover or first-token comparisons with the wrong stream. Scientific conditions retain their original recipient seeds.

The layout identity includes `(parent source block, kind, template, source subcohort, ordered recipient IDs, role)`. A block number alone is not enough across readouts or 96-source count superblocks. Target donor permutations operate on complete four-cell worlds. Do not invent corresponding donor permutations for the six-cell balanced-count assay; its controls remain the separately specified ±D additions.

## Prepass execution and immutable resolution

For every required source block/readout:

1. Verify the frozen assignment inventory, complete factorial worlds, source-prefix hashes, donor bijections, direction bindings, model revision, dtype, numerical flags and execution environment. Construct the exact role layouts before reading any outcomes.
2. Run each required unpatched layout through `StateBackend.forward` with the same capture layers and positions as its scientific conditions, `positions=(record, final)`, `max_new_tokens=1`, sampled policy and fresh cache. No intervention is active. Capture the layer-47 record component and initial full-vocabulary logit fingerprint. The sampled token is an unscored engineering byproduct and never enters another call's prompt.
3. Persist each complete 32-row capture artifact and its role receipt atomically, with file and array hashes. Persist the baseline and all required donor-role receipts **before any scientific condition for that block/readout**. Require complete role coverage; do not run science from a partial donor set.
4. Materialize one immutable resolved-target receipt from the prepass artifacts and the frozen donor recipe. Bind every target value to its donor vector, effective direction and recipient row. No correctness, closure, cap, projection sign, shift size or predicted effect can change the mapping or remove a source.
5. Execute the full 4,096-token baseline, zero sham and intervention conditions in their frozen randomized order, each from fresh input/cache and fresh per-row generators. No cache or RNG state carries over from a prepass or preceding condition.
6. Audit the scientific baseline and zero sham against the baseline-role prepass, and audit each arm against its resolved target receipt. Check operational terminal coverage separately from scientific validity and competence.

Use frozen **recipes** for quantities that cannot be known until capture. The initial manifest fixes all logical trial IDs, donor mappings, role layouts, units, sampling, resolution algorithm and condition order. A later immutable receipt supplies measured target scalars and vector hashes. Scientific records bind both the unchanged logical assignment and that receipt. Do not overwrite the frozen manifest or silently substitute a measured scalar into an assignment whose hash originally bound another value. A separate prepass collection followed by a second fully materialized static manifest is also possible, but adds launch/loading overhead and must preserve the original selection rules.

On resume, reuse the exact bound prepass artifacts and resolved receipt. Never replace a donor artifact after any dependent scientific row exists. Reject a half-written cohort, duplicate or extra role, changed direction/layout, conflicting prepass attempt, incomplete world, stale manifest hash or partial scientific cohort. Recomputing a prepass for a diagnostic comparison is not permission to replace its committed values. All arms and role prepasses for a source block stay on one worker; resumption on new hardware must still satisfy the bound environment and exact baseline comparisons.

## Projection and matched-control semantics

Let `v` be the effective information unit vector actually used by the hook, `h_r` the recipient's unpatched layer-47 record output, and `h_d(r,role)` the actual donor output captured in recipient row r under the corresponding role layout. The target is `t = dot(h_d(r,role), v)`. The information intervention requests

`delta = t − dot(h_r, v)` and `h'_r = h_r + delta v`.

For the primary donor's matched control direction `w`, request

`delta_control = t_primary − dot(h_r, v)` and `h'_r = h_r + delta_control w`.

The control uses the **information reference axis**, not `dot(h_r,w)`. It also uses the primary donor's target, not a different donor chosen for that control. Fix the `u`, `q` and five random control directions before outcomes. Reference signs are identical to the primary projection's runtime scalar for that recipient. No absolute-value dose, positive-only random control, outcome-dependent sign, or normalization by a control's observed behavioral effect is permitted. Secondary and same-gain donors remain separately labeled; they do not share the primary control comparison merely because their absolute shift happens to match.

Resolve the effective vector once using the same normalization, dtype and arithmetic path for prepass projection and runtime intervention. Store its float32 bytes/hash and norm, as well as the source NPZ indices/hash. Prefer calculating each target with the same device-side float32 dot operation as the hook, using the exact BF16 layer output cast to float32. A CPU float64 dot followed by another normalization can manufacture a self-projection shift despite identical hidden vectors. A float64 recomputation is a useful independent check within its declared floating-point bound, not a silent replacement for the committed runtime target.

`self` uses the baseline prepass's recipient component and is a projection through the information direction. Keep it separate from the true zero-addition sham. In the proposed identical-arithmetic route, self requests should be exactly zero; test that expectation before scientific use. If the implementation instead permits a roundoff-limited residual, that policy and bounds must be declared and tested before the donor engineering run. It must not inherit the true zero-sham identity label. Do not loosen zero-sham criteria to accommodate self projection.

Requested scalar matching is stronger than using a source-derived additive dose: it measures the recipient's unpatched state during each actual condition. It is valid only if those prepatch states match across conditions. Require exact layer-47 prepatch vector identity to the recipient baseline-role prepass for **every arm**, not only baseline and sham. Compute the reference projection identically and require equal runtime scalars between the primary projection and every matched control. A per-arm formula check alone cannot detect two individually consistent but differently rounded recipient states.

BF16 rounding can still make achieved L2 changes differ between unit directions. Retain `requested_scalar_delta`, `requested_delta_norm`, `delta_norm`, `rounding_error_norm`, effective unit norm, before/after projections, reference projections and target error. Audit requested equality and the declared mode-specific numerical bounds; report achieved norms and their ratios, including zero/unresolved changes. Do not describe the experiment as matching achieved perturbation norm unless that stronger claim is verified. Large systematic differences in achieved magnitude weaken specificity; do not repair them by excluding low-engagement rows or silently retitrating controls after outcomes.

Semantic orientation stays tied to the assigned donor relation and source target gain. A donor projection whose ordering is inverted relative to gain remains assigned, as does a zero or numerically unresolved shift. Such cases can reduce the intervention's efficacy; they cannot be relabeled or discarded to improve counterfactual-response rates.

## Proposed artifact and record schema

Use a new donor-capability version, preserving legacy additive formats. The following fields are required concepts; exact names should be finalized with the preparer and auditor before freezing.

| Artifact | Required bindings and contents |
|---|---|
| Frozen donor recipe | Parent manifest/assignment digest; prospective plan and candidate-receipt hashes; engine/capability/version; source/global-audit and exclusion-registry hashes; layer 47/record; direction source, unit bytes/hash and normalization version; complete donor mapping; role-layout IDs; row/seed policy; arm order; all expected prepass and scientific IDs |
| Role-layout receipt | Source block/readout/template/subcohort; role; ordered recipient IDs and actual donor IDs; complete source-multiset hash; donor source assignment/prompt/token hashes; donor relative and padded positions; per-row source and execution seeds; input/mask/position-ID hashes; common padded width and padding multiple; dtype/device/model/environment/engine; fresh-cache and no-intervention assertions; prepass cap 1 |
| Capture NPZ | Exact row ordering and recipient/donor IDs; layer and position arrays; finite float32 captures preserving BF16 values; shape/dtype/byte order; per-vector raw-byte hashes; file SHA-256; initial full-logit fingerprints; any sampled first-token IDs kept separate from scientific outcomes |
| Resolved-target receipt | Recipe and all prepass receipt/artifact hashes; logical trial/recipient/condition ID; donor source ID, donor role and role-layout row; donor vector hash; effective reference-unit hash; committed target projection and baseline reference projection; anticipated scalar for metadata; parent resolved-receipt hash if partitioned |
| Scientific record | Unchanged frozen logical assignment; resolved-target receipt hash; original recipient provenance; runtime patch mode, layer/position/token index; requested and achieved geometry; baseline-prepass capture/logit match evidence; fresh completion/token hash and cap; full response-oracle metadata; execution failure status |
| Audit receipt | Exact expected versus observed prepass/scientific inventories; role-bijection and row-location checks; target recomputation; baseline/sham/prepatch parity; cross-arm scalar matching; mode-specific geometry; cleanup/isolation evidence; prior donor-capability engineering binding; complete coverage and separate scientific readiness |

For compatibility with existing causal grouping, use stable `condition`, `donor_role`, `comparison_set`, and `matched_interchange_condition` fields. Projection and matched-control arms can retain `dose_multiplier=1.0`; any stored `dose` is an explicitly labeled anticipated reference shift, not the authoritative runtime dose. Preserve donor fields in compact outcomes. A self projection should resolve through a fitted-direction projection arm, not the current `arm='sham'` resolver, which forces a coordinate unit and zero additive dose.

Prepass captures have `max_new_tokens=1` and a distinct capture purpose. Do not pass them to the current natural scientific capture validator as though they were 4,096-token scientific records. Conversely, changing a prepass cap must not change the scientific generation allowance or its source-regime binding.

## Engineering proof and tests

Local tests should exercise real tiny-transformer hooks and independent audit helpers, with BF16 fixtures where production bit-identity is claimed. A float32 tolerance test is not a substitute for the production precision check. Synthetic metadata tests supplement, rather than replace, actual forward-pass tests.

1. **Donor mapping and layout:** verify all four cells, all donor involutions/bijections, same source multiset, exact row identity, unchanged width, proper donor-specific record indices, and recipient seed overrides. Include unequal prompt lengths, family/readout separation, repeated source IDs across roles, malformed mappings and 96-source count subcohort identities. Reject cross-world/kind/template donors and incomplete worlds.
2. **Capture parity across generation limits:** same 32-row baseline layout at cap 1 and cap 4,096 must have identical initial captures and full-vocabulary logits; its first generated token must agree. The 4,096 baseline and zero-addition sham must have identical complete token sequences, logits, initial captures and zero achieved patch norm. No truncation to a shared prefix may hide a divergence.
3. **Actual project and matched-add behavior:** check positive, negative, exact-zero and roundoff-limited targets. Verify observed donor target values, reference-axis use, exact primary/control runtime scalar identity, requested norm agreement, achieved rounding bounds and self policy. Add tests where control reference axis, donor role, target, vector hash or row is wrong; the auditor must reject each.
4. **Row isolation and cleanup:** in the actual 32-row shape, perturb one row using project and matched-add modes; all unperturbed rows retain initial captures/logits and their short seeded continuations. Also test a single-row donor-input substitution during a prepass to check isolation before the intervention layer. A subsequent clean baseline must be unchanged. Include exception paths and verify no hooks or caches leak. Keep actual full-run zero-sham equality as a separate gate.
5. **Row-layout sensitivity:** from engineering prepasses, compare native-row and row-matched donor projections and vector differences, scalar sign changes, and differences relative to the intended shift. These diagnostics quantify the nuisance; they do not select the better-looking method or exclude rows. The planned method remains row-matched.
6. **Persistence and coverage:** corrupt file bytes, target receipt references, direction normalization, source row order, padding, seeds, role inventory and parent/child cohort membership; each must fail closed. Resume tests reject changed prepasses and partial cohorts and preserve the exact frozen role/condition inventory. No `execution_error` row or missing terminal ID can be converted into an engineering pass.
7. **Scientific outcome isolation:** score only newly generated final answers with the independently reconstructed target oracle. Tests ensure no prepass token, first-token logit, donor correctness, stale source completion, or actual shift sign changes semantic scoring or source inclusion. Zero planned reference dose must not bypass a project-mode geometry audit.

Production engineering must be a distinct pinned 32B/H200 collection with the actual batch size, packing policy, dtype, generation cap, effective-unit arithmetic and donor capability. A representative eight-world, 32-source, 13-arm donor panel has 416 full continuations, plus four 32-source prepasses. The currently running additive 416-row pilot does not become this donor proof merely because its dimensions match. Require at least one substantial baseline continuation under the existing ≥128-token criterion, both signs and nonzero engagement coverage for each implemented patch mode, exact zero-sham identity, and all declared mode/geometry/coverage checks. Synthetic nonzero geometry probes may supplement a fixed panel containing tiny observed donor shifts, but are labeled engineering probes and do not select scientifically favorable worlds.

The prior-engineering audit must name the donor mode, role layout and resolution capability. It must be distinct from a scientific collection, match the later engine and execution regime, and fail scientific admission if a relevant implementation hash changes. Existing startup guards inspect an additive perturbation via its scheduled `dose`; donor startup checks must use resolved runtime semantics rather than accidentally satisfying that additive branch.

## Estimator and scientific-admission dependencies

The existing natural analyzer needs mode-specific changes before donor results can be interpreted. It currently uses `elif row['dose']` to collect non-sham engagement: a projection with zero anticipated dose must still be audited. Keep true zero shams under strict identity checks, self projection under its own declared policy, and all projection/matched-add rows under their actual runtime geometry. Add an explicit within-prefix primary/control scalar-identity audit and bind every donor artifact. Do not rely solely on the existing fixed-replay norm-matching helper's per-row tolerances.

Retain complete four-cell source worlds and the full assigned denominator. The primary outcome remains a valid donor-predicted counterfactual response; invalids and distractors receive zero. A signed response-margin change can reward invalidation and remains descriptive. Donor grouping must use stable condition/role identities instead of grouping by varying numeric donor doses. The five-random comparator is formed within source/world before resampling. Semantic uniqueness, fixed family/complement/action-map strata, the 24 primary endpoints, 100,000 paired world resamples, and six heterogeneous-world preservation bounds remain as specified in the main plan. Engineering receipts do not certify any of those scientific claims.

For balanced-count competence, a global D derived from confirmation donor targets requires **all target prepasses needed for that rule to finish before count ±D conditions are resolved or run**. Per-worker or per-wave maxima are not the global frozen rule. Persist the aggregate D receipt, bind it into the count schedule, and charge the extra orchestration/idle time. If the implementation cannot enforce that dependency, do not substitute a convenient partial maximum.

A same-gain donor is not inherently a norm-matched null: its observed shift can differ from the opposite-gain donor's shift. Memory and balanced-count controls address selected forms of competence, not every off-manifold effect. Row matching removes one layout concern but does not establish a unique information-value representation, and a transplanted scalar can still alter downstream answer policy. Successful engineering provides trustworthy manipulation, not semantic specificity or evidence of inner experience.

## Compute overhead and prospective sequencing

Every role prepass performs a full prompt prefill and allocates a fresh cache, even though it generates only one token. Budget prompt tokens, padding, capture transfer/storage, startup probes, model loading, idle dependency barriers and shutdown using measured wall time. Do not price it as a single decoded token or assume the extra calls are free.

For the default confirmation target sample, each scheduled readout has 864 distinct prefixes in 27 batches of 32. If all four role layouts are captured for instrument/new/repeat/memory, that is **432 batch prepasses and 13,824 one-token row completions**. A smaller rule can be fixed before execution: instrument needs all four layouts (108 batches), whereas each of new/repeat/memory needs baseline and primary opposite-relation only (162 batches combined). That totals **270 batch prepasses and 8,640 one-token row completions**. Either inventory must be fixed and audited; omitted unused roles are not missing observations, and required roles cannot be dropped to meet a deadline. The global D rule still uses every assigned primary donor required by the main plan.

The donor engineering panel adds at least 128 one-token row completions to its 416 full continuations, plus declared isolation/startup probes. These are separate from the full source-qualification or self-contained-baseline counts in the main plan. No additional compute allowance is granted here, and the $35 development ceiling is unchanged. Measure a representative donor-engineering pass before committing the larger stage, reserve the full frozen workload, and stop for insufficient budget rather than silently omitting roles, worlds or controls.

Implementation can be prepared regardless of the target candidate outcome. Scientific execution remains conditional on that candidate's prescribed admission and later development checks. A failed target candidate is a failed candidate, not a reason to use this infrastructure at a newly selected site without a new prospective hypothesis.
