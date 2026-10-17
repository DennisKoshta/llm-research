# Sustained steering engineering: both stages passed within observed scope

October 15, 2026 UTC. The frozen short and full-allowance stages are complete, independently audited, and stopped. They test intervention implementation, not curiosity, satisfaction, or experience. The [prospective plan](sustained-engineering-v1-plan.md), [short audit](qwen32-sustained-engineering-v1-128-audit.json), [full audit](qwen32-sustained-engineering-v1-4096-audit.json), and [independent numerical check](sustained-engineering-v1-independent-check.json) preserve the evidence and its limits.

Both stages use the same 32 original voluntary-v2 prefixes, seeds and companions. The four conditions are baseline, an installed all-row zero sham, a single nonzero curiosity-direction row with zero sham on its 31 companions, and a clean baseline after a malformed-intervention exception. The fixed corpus-A layer 15/record curiosity direction is added once at the original record boundary and during active generated-token processing. This is a machinery check; no task answer is scored or selected here.

| Measurement | Short stage | Full-allowance stage |
|---|---:|---:|
| Configured token allowance | 128 | 4,096 |
| Terminal condition-rows | 128 | 128 |
| Maximum observed trajectory | 128 | 2,382 |
| Perturbed-row observed trajectory | 128 | 744 |
| Model forwards across four conditions | 512 | 9,528 |
| Saved perturbed-row before/after pairs | 128 | 744 |
| Main execution harness time | 70.6 seconds | 2,145.6 seconds |
| Incremental app-wall compute estimate | $0.230 | $3.087 |

All short-stage rows reached the 128-token allowance. All 128 full-stage condition-rows ended with EOS before the 4,096-token limit; individual lengths ranged from 236 to 2,382. The full stage exercised mixed active and terminated rows. Thus it validates the observed trajectories and specified configuration, including native-cache decoding, but does not demonstrate 4,096 processed tokens or every terminal layout. No row was replaced to obtain a longer example.

The main audits reproduced all 13 required checks. A separate checker, importing none of the producer's engineering validators, independently reconstructed every saved BF16 after-vector bitwise from the observed before-vector and actual FP32 perturbation. It also compared the saved full-FP32-vocabulary fingerprints, completions and active inputs. In the full stage, 17,269 sham predictions, 16,629 untouched-companion predictions and 17,269 clean-repeat predictions matched their baseline counterparts exactly. The malformed direction was rejected before a model forward, and hooks were cleaned. Full vocabulary arrays are not retained; identity is evidenced by their recorded fingerprints. The complete source/archive/mask provenance chain remains the main auditor's responsibility.

The perturbed full-stage row had one prefill and 743 decode injections. Its measured relative per-injection perturbation magnitude ranged from 14.53% to 31.03% of the native residual norm. This is a substantial imposed perturbation. Cumulative norm sums describe exposure, not net displacement or psychological intensity. The check concerns one curiosity-direction row and does not establish all-row satisfaction behavior or specificity.

Each full-stage condition used 2,382 forwards and about 518–520 seconds, approximately 0.218 seconds per prediction step. Total generated tokens were 69,180. The saved harness JSON was 55,794,147 bytes and the residual archive 14,385,515 bytes. Worker peak allocated CUDA memory was 91,773,359,104 bytes (85.47 GiB), with 149,350,776,832 bytes reserved (139.09 GiB); these include model loading and startup, with no peak-counter reset. Reserved memory is not an OOM result, and longest-context memory behavior remains untested.

The planned [object-only satisfaction screen](satisfaction-screen-v1-plan.md) therefore retains concurrent exact sham comparisons, actual positive engagement, all-arm prepatch identity, runtime parity, task gates and selected-information receipts. Its 1,152 branches have not yet been selected, frozen or run at the time of this report. Resource sizing was revised before scientific outputs using the measured full-stage rate: three workers, a 12,000-second shared deadline and a $55 stage ceiling. The original natural satisfaction prediction remains unsupported; engineering success does not change that result.

The combined engineering-stage estimate is $3.317, within its $10 ceiling. Fresh-phase compute totals **$51.315 of $300**, leaving **$248.685**. All project apps are stopped with zero tasks. These are conservative app-wall estimates, not invoices. [Final cost and shutdown receipt](qwen32-sustained-engineering-v1-4096-cost-and-shutdown.json).
