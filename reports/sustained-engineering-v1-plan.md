# Sustained steering: bounded research-model engineering checks

October 14, 2026. Written after the complete [voluntary-resolution v2 result](voluntary-resolution-v2-results.md) and before any actual-model sustained-steering output. The source study's failed global readiness is preserved. This plan tests the intervention machinery only; source task readiness is recorded and does not determine engineering eligibility. No scientific outcome is scored here.

## Fixed source, model and perturbation

Use the first actual physical cohort of 32 original prompt prefixes from `runs/qwen32-voluntary-resolution-v2`, in its original order. Its manifest SHA256 is `b2ef2c264176003c9a0bcedf03a3e7bd80668cef38167d29f9ce3d9005861ab6`. Require complete source integrity and bind every original source record, unpadded prompt token, record/final position, assigned seed and companion row. Do not include generated source text, select on answers or lengths, or replace the first perturbed row.

Keep pinned Qwen3-32B revision `9216db5781bf21249d130ec9da846c4624c16137`, unchanged BF16 weights, eager attention, FP32 patch arithmetic, the existing 128-token padding multiple, temperature 0.7 and full-vocabulary sampling on one H200. The fixed engineering vector is corpus-A curiosity versus indifference, layer 15, record position, positive unit direction with dose exactly one original raw contrast norm (29.78968343510005). This choice is mechanical and does not assert a behavioral curiosity manipulation.

Use `record_prefill_plus_decode`: patch the original record-token block output once during prefill, then the current one-token input's block output during each cached decoding forward for an unfinished row. The first sampled token receives only the prefill intervention. EOS and cap-reaching tokens are not forwarded again. Preserve prior effects in the native cache; hook removal does not reset that cache.

## Two separately frozen stages

First freeze and execute a **128-token allowance**. Each stage contains four complete conditions on the same 32 prefixes, companions and seeds: baseline without an intervention hook; a genuine zero-dose sham on all rows; the same hook layout with only physical row zero at nonzero dose; and a clean baseline repeat after a deliberately malformed-direction exception. Each measured stage contains 128 terminal records. Normal engineering-check failures remain measured failures; runtime failures remain explicit unavailable execution dispositions; a hard interruption remains incomplete. No partial or started attempt is rerun.

Only an independently reproduced passing short-stage audit permits preparation of a separate **4,096-token allowance** stage. It must bind the same source, vector, dose, model, arithmetic, schedule and engine bytes. The cap and stage documents are the intended differences. The full stage re-audits the exact short-stage report and vector archive, rather than accepting its saved pass flag alone.

The allowance is not an observed trajectory length. Report every row's actual termination and the observed maximum. Neither early EOS nor an unexercised long context is concealed through row replacement. A short-stage pass cannot certify the full allowance; a full-stage pass covers the observed trajectories and specified configuration.

## Required evidence

Before terminal records are published, save the complete condition report and before/after residual arrays for the perturbed row, binding their actual file bytes to every row. Preserve full FP32-vocabulary fingerprints at every active prediction, including prefill and the EOS-predicting step. Reconstruct actual input IDs, mask hashes, position IDs, cache lengths and active rows from the bound prefixes and generated tokens.

Require exact baseline/sham token and full-vocabulary identity throughout all active predictions; the same identity for all 31 unmodified rows in the one-row condition; and exact identity in the clean baseline repeat. Require the correct prefill/decode injection counts and positions, finite independently observed BF16 before/after geometry, actual nonzero decoding engagement, malformed-direction rejection before model execution, and complete hook cleanup. A changed answer is not required. Immediate EOS on the perturbed row cannot demonstrate sustained engagement and does not trigger a replacement source.

The offline audit reconstructs evidence rather than trusting saved flags, but reuses the collector's timeline, identity and geometry validator functions. Independent cached-versus-full-prefix reference tests complement that shared-code check. The pre-launch repository suite passed 858 tests and Ruff; further unrelated development files may be included only after their local checks pass and the source freeze is stable.

Report execution time by condition, number of forwards, token counts, actual report/vector sizes, and worker-wide peak allocated/reserved CUDA memory. These are descriptive engineering measurements, not behavioral eligibility criteria. Use the short run to assess the likely full-run cost; do not promise that a longer run will complete within its allowance.

## Spending and stop conditions

Reserve at most **1,200 seconds** for the short stage on one H200, with a **$2 incremental ceiling** including the configured startup/shutdown allowance. All project GPU apps must be stopped before admission. Stop and export before auditing locally.

If the short stage passes and its performance makes a full run reasonable, reserve at most **5,400 seconds** for the separate full stage, with an **$8 incremental ceiling**. The combined ceiling is **$10**, within the remaining fresh $300 allowance. No paid automatic retries, deadline extensions, overlapping project jobs, source replacement, or relaxed identity thresholds. An incomplete or failed stage cannot authorize scientific steering. Correcting an identified implementation fault would require a new, separately documented stage preserving these failures.

Even two passes authorize no psychological claim. A later object-only satisfaction screen must have its own frozen assignments, hypotheses, stopping rules, concurrent controls, actual information delivery and interpretation limits. The natural satisfaction prediction remains unsupported in the source study.
