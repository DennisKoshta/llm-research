# Published-method control: setup failed, benchmark unlaunched

October 15, 2026 UTC. This closes the admitted setup attempt for `qwen25-pain-profile-control-v1`. The full 2,020-choice benchmark was never admitted. No behavioral qualification or schedule-effect result is available.

The source-derived Qwen2.5-32B model and released adapter loaded successfully on an H100. The setup then stopped at its first repeated baseline: all 32 answer token sequences matched, but internal activations and full-vocabulary output-logit fingerprints did not. The frozen exact-repeat requirement failed. Identical sampled answers do not satisfy an internal-state repeatability check.

## What was completed

The bounded CPU preparation verified the pinned runtime, base weights and seven adapter files. The GPU setup completed 11 of 21 assigned calls and 292 of 612 assigned row slots. The remaining 320 slots are missing because execution stopped at the failed check. The fault-injection, clean-follow-up and unaffected-row tests later in the schedule were not reached. All partial records remain retained.

Loading took 138.03 seconds. The slowest of the ten primary setup calls took 5.94 seconds, giving the frozen conservative main-run forecast of 1,029.95 seconds against a 2,880-second admission limit. That timing check passed; it cannot override the failed engineering check.

## Repeat diagnosis

The comparison is setup call `0000` against repeat `0010`, with the same 32 prompts, companions, seeds and baseline intervention. Prompt inputs, attention masks, positions, cache lengths, sampling evidence, RNG evidence and generated token sequences match. Prefill activation and logit fingerprints also match. Divergence begins during decoding, before the steering hook.

| Forward | Active rows | Different full-logit fingerprints, all rows | Different full-logit fingerprints, active rows |
|---|---:|---:|---:|
| Prefill | 32 | 0 | 0 |
| Decode 1 | 32 | 19 | 19 |
| Decode 2 | 26 | 25 | 21 |
| Decode 3 | 8 | 28 | 6 |

Before-hook residual fingerprints differ on 13, 21 and 26 rows at the three decode steps. Saved residual arrays confirm that this is forward-state divergence, not only a projection-reporting discrepancy. The [engineering diagnosis](pain-profile-baseline-repeat-diagnosis.json) binds the compared records and describes the observed differences without using condition effect rates.

Read-only source review found fresh caches and random generators per call, a zero-coefficient baseline hook, nonmutating observer copies, empty hook inventories before and after execution, and evaluation mode. It found no concrete implementation state leak. Cache tensor contents and the selected CUDA kernels were not captured, so the retained records cannot identify the cause. PyTorch documents that CUDA attention can select nondeterministic algorithms; this is a plausible diagnostic lead, not a demonstrated explanation of this run. [PyTorch 2.11 SDPA documentation](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html)

## Independent audit limitations

The original frozen auditor expected adapter inventory names with a leading `./`, while the frozen loader used canonical relative paths. A separately reviewed [reporting-only amendment](pain-profile-canonical-paths-amendment.json) binds the exact seven matching file hashes and corrects only that path comparison. The original source, numerical checks and scientific thresholds remain unchanged.

The corrected audit attempt encountered a second issue: a saved residual sample's CPU-computed norm fell outside the GPU-recorded norm range under the frozen absolute tolerance. It terminated with `Saved sample is outside observed norm range`; it did not produce a completed audit. The [failure receipt](pain-profile-setup-audit-failure.json) records that all 88 saved row events, containing 17,643,520 scalar components, match the source BF16 elementwise injection exactly. All 64 active saved row events nevertheless fail the CPU norm-containment test: the largest CPU/GPU endpoint difference is 0.0001068, or seven FP32 representable steps near a norm of 144. The frozen tolerance of 0.00001 is smaller than one such step. A float64 reference supports a reduction-rounding discrepancy. This is separate from the genuine baseline repeat failure.

This report therefore does **not** claim a passed independent setup audit. No tolerance was relaxed and no passing audit was substituted. The original campaign remains failed under both its recorded setup result and the incomplete independent audit.

## Scientific interpretation and disposition

This is an engineering failure, not a null result for pain steering and not a refutation of the original paper. The planned comparison between whole-prompt and final-prefix-only steering remains unmeasured by a qualified benchmark. The different adapted Qwen2.5 checkpoint also cannot explain the earlier Qwen3 curiosity/satisfaction results by itself.

The failed campaign remains closed to automatic paid retries. The main benchmark and deferred S31/S47 extension remain unlaunched. A separately declared [five-call deterministic diagnostic](pain-profile-deterministic-diagnostic-v1-plan.md) tests one fixed numerical preset, with a $2 ceiling, no behavioral scoring and no second preset. It cannot retroactively pass this attempt. The substantive objective remains naturally recruited curiosity/satisfaction mechanisms, with ordinary task following and utility appraisal as competing explanations.

## Cost and provenance

CPU preparation cost approximately **$0.19026** and H100 setup **$0.34878**, for **$0.53904** in this campaign. Fresh-phase compute is estimated at **$89.12306 of $300**, leaving **$210.87694** before the retained $10 storage/transfer reserve. Both campaign apps and all project GPU apps were confirmed stopped in the [shutdown receipt](qwen25-pain-profile-control-v1-setup-cost-and-shutdown.json). These are configured app-wall estimates, not vendor invoices.

- [Frozen campaign](../runs/qwen25-pain-profile-control-v1/freeze.json): manifest `e87c70267be0b7813750dbef9d752f0855cf5b2db90e81f2381127d544b59a63`; source `63bc3e330b25f5805724bfa5b8bf2f75d6fe37d048963524e1d28138e4bd4ff9`; archive `9c59998a8a162075a4adc201c4a374f821f76978b0c4ec518dce4630cd3f1fa7`.
- [CPU summary](../runs/qwen25-pain-profile-control-v1/phases/prefetch/summary.json): passed, no model inference.
- [Setup summary](../runs/qwen25-pain-profile-control-v1/phases/setup/summary.json): incomplete, integrity failed, terminal error at `setup-0010`.
- CPU app `ap-dIDLFBhYnMt6HlThw7gIH0`; setup app `ap-Kj0jGVW1leAqxx26pDVHqE`.
