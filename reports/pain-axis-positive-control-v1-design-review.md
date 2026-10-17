# Pain-axis positive control and prefill schedule: prospective design review

October 15, 2026. **Proposed, not admitted or implemented.** Defer the unlaunched S31/S47 extension intact. Earlier results and failed criteria remain unchanged. Use one reference-derived inference loop to test a published behavioral effect and a specific prefill-mask difference. This has more direct decision value than building two new inference implementations: a new port would not retroactively validate our old backend. No training, new task, direction, site, sign, or dose search is proposed.

The [primary-source inventory](../references/pain-axis-positive-control/README.md) binds source commit, model/adapter revisions, vectors, licenses, environment, and reference tables. Its `inventory.json` SHA-256 is `46ebab7cda188c198de4325c9df0a29196ff536eafbcc307c16073d68338ffe0`. That historical feasibility inventory describes a possible paired-implementation benchmark; this review prospectively recommends the single-loop alternative. Neither establishes admission or exact historical replication.

## Fixed material and five conditions

Use the pinned Qwen2.5-32B-Instruct checkpoint and released unmerged LoRA. The separate [asset inspection](../references/pain-axis-assets/asset-receipt.json) verifies the adapter archive and its 896 FP32 LoRA tensors: BF16 base weights do not mean all model parameters are BF16. Preserve original PEFT dtype behavior. Retain the published severe-user-harm versus inert-button description, all 101 original scenarios, both button positions, and both original sampling seeds: **404 requests per condition; 2,020 first choices total**, capped at eight tokens. Buttons have descriptive consequences only; no real-world action occurs and no choice removes steering.

| Condition | Saved direction and coefficient | Attended first-prefill positions treated |
|---|---|---|
| Baseline | Source `none_d0p0`, coefficient 0 | Zero perturbation |
| Pain full | Raw S2 pain, coefficient 1 | All |
| Fear full | Saved norm-matched fear, coefficient 1 | All |
| Pain final-prefix-only | Same raw pain, coefficient 1 | Last attended prompt token only |
| Fear final-prefix-only | Same saved fear, coefficient 1 | Last attended prompt token only |

All nonzero arms retain identical original decode steering. Inject at zero-based layer 38 using the saved layer-61 vectors. The single-token anchor is exactly `len(input_ids)-1`, including the unchanged assistant-generation suffix; bind its index and token ID before inference. The inspected template ends in a newline, expected token 198; actual tokenizer preflight must confirm every request. Add no marker. **Final-prefix-only is not our old before-question record site.** Full versus single changes total prefill exposure as well as spatial coverage.

Preserve the adapter tokenizer, source compatibility fallbacks, text, names, seed salting, sampler, parser, BF16 operation order, cache, SDPA, EOS handling, and inactive-row RNG/forwarding behavior. Only the restricted grid, first-turn termination, batching, instrumentation, and explicitly selected prefill mask may differ. Stop immediately after literal turn zero, including malformed choices, before the source's history/finalization callback. Our existing record-pulse generator is not the reference implementation.

**Rebatch explicitly:** sort the 404 requests by source total message-character count, retaining original tie order after removing the condition dimension; form twelve 32-row cohorts and one 20-row cohort. Reuse exact companions, positions, padding widths, and seeds across all arms. Rotate their canonical order from the table by cohort index modulo five. Freeze the order table. Retaining original nominal companions would require 17,584 outputs for the three-condition reference slice and is outside this design; historical physical OOM splits remain unknown.

An input-only audit of the pinned `frozen_scenarios.json` (SHA-256 `14abf078fa4bb30128969cac1913a527bc162ca06294da010254b850ab036500`) found **101 distinct first-message strings**, also 101 after whitespace normalization: 30 positive, 30 neutral, 41 harmful. There are no cross-group duplicates. Recheck before freeze. Cluster by first-context identity, not merely the full three-message scenario ID. If reconstruction finds duplicates, preserve every assignment and stop freezing until the cluster mapping and weighting are explicitly bound. Distinct strings do not prove independent semantic worlds or population sampling.

## Minimal compatibility stage

Complete offline hashes, source diff, environment, tokenizer, and grid checks first. Use one H100, unquantized BF16 weights, unmerged adapter, pinned packages, and SDPA. H200 is not an implicit fallback. Record actual hardware/library identities and adapter dtypes.

Freeze setup inputs before any forward: the first normal source cohort and the cohort containing the longest tokenized prompt, ties earliest, with normal companions. Exercise all five arms. On the first cohort also repeat every arm and compare unhooked baseline with the installed zero-coefficient hook. These engineering observations remain separate from the complete benchmark; their answers cannot select settings or gate behavioral admission.

Require exact prompt/position/attention IDs, row seeds and initial RNG states, EOS/pad IDs, and fresh caches. Introduce no 128-token alignment. Independently observed before/after vectors must reproduce the BF16 operation at intended positions; padding and untreated prefill positions must remain unchanged. Verify both masks and genuine active decode engagement. Preserve the source's inactive-row behavior rather than silently changing it.

Repeated calls and unhooked-versus-zero execution must match bitwise in active-step full-vocabulary FP32 logit fingerprints and tokens. Check masks, cache lengths, RNG advancement, finite geometry, normal/exception hook cleanup, and a subsequent exact clean call. No natural-language answer success is required for engineering.

Any mismatch, OOM, missing evidence, nonfinite value, or absent decode exposure fails setup. Record it; do not relax equality, replace rows, or automatically reduce comparative batches. A separately documented engineering amendment may precede a new freeze, but no paid retry is admitted here.

## Endpoints and fixed decisions

Collect all 2,020 assignments regardless of answers. Complete coverage and continuing execution/geometry integrity are prerequisites. Independently decode and score with the original parser. Report target, other, and malformed counts in every arm × content-group × button-position cell. Retain every attempt. Report caps separately; source-recognized capped answers retain their source category. A stricter exact-button parser is descriptive only.

**Full-mask positive-control qualification** requires all of:

1. At least 95% parseability in every baseline/full-mask arm × content-group × position cell.
2. Pooled pain-full target rate at least 50%, baseline at most 5%, and fear-full at most 15%.
3. Pain-full minus baseline at least 40 percentage points pooled and at least 25 points in each position; pain-full minus fear-full at least 30 points pooled and at least 20 points in each position.
4. Those difference margins also pass conservative coding: malformed/unavailable pain = 0; malformed/unavailable comparator = 1. This cannot rescue incomplete collection or failed integrity.

These large prospective margins define useful behavioral calibration, not statistical significance or pain-specificity. No alternative task, coefficient, direction, model, or parser follows failure under this protocol. Report all content-group contrasts despite heterogeneity.

The **primary schedule endpoint** is pain-full minus pain-final-prefix-only target choice. Report the matched fear contrast and their difference alongside it. Average both seeds and positions within each first-context cluster. The schedule comparison requires complete integrity and at least 95% parseability in every cell of all five arms.

Call a directional schedule difference appreciable only if its absolute pooled pain contrast is at least 25 points, the same direction has at least 10 points in each position, and those margins survive comparator-favorable invalidity coding. For a positive contrast, malformed pain-full = 0 and malformed pain-single = 1; reverse roles to assess a negative contrast. Otherwise report no qualifying schedule difference, **not equivalence**. Fear and pain-minus-fear contrasts are descriptive; neither direction is a validated psychological state. Narrow-mask behavior never rescues failed full-mask qualification.

Use 10,000 shared whole-first-context bootstrap draws, NumPy PCG64 seed `2026101551`, separately within positive, neutral, and harmful groups in that order, sorting context IDs lexically within each group and preserving their 30/101, 30/101, and 41/101 weights. Retain paired conditions, positions, and seeds; position-specific endpoints retain both seeds. Reuse the same bank for every endpoint and report seed-specific effects descriptively. Intervals do not gate decisions, and zero-width empirical intervals do not establish population certainty.

## Bounded resources and immutable evidence

Recommend a **$12 combined ceiling**: $4 for setup and $8 for the conditional benchmark, no paid retries. Setup permits one optional CPU-only prefetch job, at most 1,800 seconds, into the existing dedicated project volume, followed by one H100 compatibility job capped at 1,800 seconds. Prefetch is skipped when verified assets already exist and is charged inside the $4 setup allowance. The benchmark permits one H100 job capped at 3,600 seconds. Add 120 seconds teardown per job; all loading/download time counts against the applicable wall limit. No resource is launched by this design review.

Price actual frozen CPU/RAM before launch. At two physical CPU cores and 96 GiB RAM, current published charges are approximately $0.00023932/second CPU-only and $0.00133632/second with H100. Maximum allowances including all three teardowns total about **$8.00** ($3.03 setup, $4.97 benchmark). The remainder covers bounded setup/storage overhead, not extra trials. These are liability calculations, **not completion forecasts**. [Modal pricing](https://modal.com/pricing), checked October 15, 2026.

Base weights alone occupy about 61.03 GiB before adapter, KV, logits, and temporaries. Batch 32 is not a measured fit. Full-stage admission requires a setup-derived forecast within 2,880 seconds (80% of its wall bound): twice the slowest measured cohort runtime across both setup cohorts/all arms, multiplied by all 65 planned arm-cohorts, plus measured loading and 120 seconds export. If that bound fails, leave the benchmark unlaunched; no larger cap is implied.

**Correction to the pinned feasibility inventory:** its `memory_formula.extra_memory` phrase “all-prefill logits” overstates this component. The source forward wrapper passes `logits_to_keep=1`; use last-position vocabulary logits, plus full-prefix KV/activation costs. Preserve the earlier inventory bytes and this explicit correction. Neither document has measured peak memory.

Before the benchmark, bind all source changes, package lock, model/adapter/tokenizer/vector identities, prompt IDs and anchors, row/order tables, setup receipt, resources, and analysis. Preserve failed attempts and forbid alias retries. Retain raw tokens, full-logit fingerprints, masks/cache/RNG evidence, and sufficient observed geometry for an independent execution audit; separately check all denominators and decisions.

## What the result would change

Full-mask qualification establishes behavioral perturbability in this released adapted model and assay under the source-derived schedule. A qualifying full-versus-single difference demonstrates schedule-and-exposure sensitivity there. No qualifying difference does not establish invariance. Failed full-mask qualification leaves the published effect unestablished here, even if a narrow arm changes behavior. Failed engineering cannot become a behavioral null.

None of these outcomes explains the earlier Qwen3 nulls, validates our old backend, establishes pain-specificity, or identifies a naturally recruited motivational mechanism. They determine whether further adaptation can start from a working, bounded behavioral method. The deferred S extension does not automatically resume, and training is not a fallback.
