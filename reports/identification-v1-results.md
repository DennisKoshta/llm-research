# Matched identification diagnostic: results and next decision

Completed October 6, 2026. **All 1,200 assigned branches were exported and independently audited.** The frozen development selection rule selects **A–D candidate labels with sampled thinking mode** for fresh assay validation.

## What the comparison showed

Every single-mystery condition scored 50/50 correct and valid in each family. In paired context, changing J–M to A–D removed the non-thinking formatting failures, but did not restore adequate identification accuracy. The selected thinking configuration scored 50/50 correct and valid in each family.

| Family | Paired setting | Correct / 50 | Valid / 50 |
| --- | --- | ---: | ---: |
| Machines | J–M, no thinking, greedy | 21 | 27 |
| Machines | J–M, no thinking, sampled | 21 | 25 |
| Machines | A–D, no thinking, greedy | 37 | 50 |
| Machines | A–D, no thinking, sampled | 38 | 50 |
| Machines | J–M, thinking, sampled | 47 | 47 |
| Machines | A–D, thinking, sampled | **50** | **50** |
| Objects | J–M, no thinking, greedy | 32 | 34 |
| Objects | J–M, no thinking, sampled | 32 | 34 |
| Objects | A–D, no thinking, greedy | 43 | 50 |
| Objects | A–D, no thinking, sampled | 44 | 50 |
| Objects | J–M, thinking, sampled | 49 | 49 |
| Objects | A–D, thinking, sampled | **50** | **50** |

The four unsuccessful thinking responses lacked a closing reasoning delimiter: three machine and one object answer, all in paired J–M conditions. They remain invalid and incorrect under the frozen scoring rule. No first-token rescoring or answer retries were used.

These matched manipulations support an interface-and-computation explanation for the earlier measurement failure. They do not isolate a neural mechanism. In particular, thinking mode includes a larger generation allowance and final-answer parsing after the reasoning delimiter; its improvement cannot be attributed solely to a chat-template switch. The result does not demonstrate curiosity, satisfaction or inner experience.

## Design and audit

Twenty-five new development pairs per family were crossed with queried side, single/paired context, J–M/A–D labels and three inference settings. Each displayed cell contains 50 queried mysteries clustered in 25 pairs. Repeated conditions are not independent observations; these are descriptive development results, with no confirmatory p-values or population-generalization claim. Sampled cells contain one fixed draw per item/side.

The same target and distractor were reused across the matched comparisons. No-thinking generation allowed eight tokens; thinking allowed 768. Both sampled settings used temperature 0.7, full vocabulary and top-p 1. The original one-credit-v1 results remain unchanged.

The audit verifies exact assignment coverage, envelope checksums, reconstructed prompts and expected answers, strict scoring, inference settings, the model-integrity checks and hook cleanup. [Full cell report](../runs/modal-identification-v1/report.md), [audit and matched contrasts](identification-v1-audit.json), [prospective plan](identification-v1-plan.md), [frozen manifest](../runs/modal-identification-v1/manifest.json), [source archive](../runs/modal-identification-v1/source.tar.gz).

## Advancement

Only A–D with sampled thinking meets the paired-context requirements in both families. The selected setting is written to [the v2 configuration](../configs/one-credit-v2.json), with the selection audit and manifest hashes. This selection does not itself authorize discretionary choices.

The separately versioned one-credit-v2 assay now enters the [prospectively specified fresh validation](one-credit-v2-conditional-plan.md): 100 new pairs per family, seven checks per pair, 1,400 diagnostic branches. Consequence-answer IDs are 1–4 while actions remain A–C. An H100 is explicitly pinned for this thinking-enabled follow-on. Hardware, model, source and assignments must match between its validation and any subsequent calibration. Previous v1 prompts remain byte-for-byte reproducible under regression checks.

## Resources

The comparison used one L40S, driver 580.95.05, the pinned Qwen3-8B revision and BF16 eager attention. Model loading took 9.93 seconds; execution took 4,033.68 seconds. Peak allocated memory was approximately 16.61 GB. The app stopped with zero active tasks before the next job was admitted.

The run added approximately **$2.74** in conservative app-wall compute estimates, bringing cumulative project compute to **$3.61** at its completion. The user-authorized cumulative allowance is **$100**, with $10 reserved for storage/transfers. These estimates are not invoices and do not assume available account credits. [Cost and shutdown evidence](identification-v1-cost-and-shutdown.json).
