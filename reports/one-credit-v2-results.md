# One-credit-v2 validation: token-limited useful-clue responses

October 7, 2026. **The frozen validation gate failed. No v2 discretionary-choice calibration was launched on Qwen.** All 1,400 original assignments are complete and independently audited. This is instrument development, not evidence about curiosity, satisfaction, or subjective experience.

## Frozen setting and results

Qwen3-8B used the A–D labels and sampled-thinking setting selected by the earlier matched diagnostic. Consequence answers used 1–4 while actions retained A–C. The fresh sample used seed 20261008, 100 pairs per family, seven checks per pair, H100, BF16, full-vocabulary temperature 0.7, and at most 768 generated tokens. Each family/check cell required at least 90 correct and 95 valid responses out of 100. Thinking responses required a closing delimiter and an exact final answer character.

| Check, n=100 per family | Machines correct / valid | Objects correct / valid | Both families pass? |
| --- | ---: | ---: | --- |
| Identify X | 98 / 98 | 99 / 100 | Yes |
| Identify Y | 99 / 99 | 100 / 100 | Yes |
| Obtain useful clue for X | 70 / 70 | 87 / 87 | No |
| Obtain useful clue for Y | 68 / 68 | 87 / 87 | No |
| Consequence of selecting X | 100 / 100 | 100 / 100 | Yes |
| Consequence of selecting Y | 100 / 100 | 100 / 100 | Yes |
| Consequence of declining | 100 / 100 | 100 / 100 | Yes |

There were 1,309 valid responses and 1,308 correct responses overall. Ten of fourteen cells passed. Identification generalized much better than in v1, and all consequence checks passed, but those successes cannot compensate for the four failed useful-clue cells.

## Failure description and limits

All 91 invalid responses reached the 768-token cap without EOS: 89 lacked the required thinking closure, and two lacked a valid final answer after closure. One additional response reached the cap with a valid final answer. There was also one valid but wrong object-identification answer. The descriptive report retains these categories separately.

Every valid response in the four useful-clue cells was correct, but incomplete responses remain failures in the assigned-trial denominator. A longer allowance might help; this run does not reveal how those responses would finish. Do not reinterpret missing answers as disinterest, claim general inability to understand clues, salvage a first token, or change the original gate. The validated operational setting remains unavailable.

## Execution and integrity

Serial timing projected beyond the original two-hour deadline. The prospective execution amendment retained 36 original records and partitioned the remaining 1,364 checks across four workers. Three finished all 341 assigned checks. The slower worker was explicitly committed and stopped at 244 saved records; those records were exported and verified before its remaining 97 checks were divided across three workers. All continuations retained the original absolute deadline and original prompts, seeds, scoring, and scientific source.

The nested and outer aggregations preserve original payloads and provenance. The complete audit verified 1,400 unique physical records, exact assignments and inference settings, reconstructed prompts and expected answers, and eight passing real-model integrity audits. All hooks were removed. Unstarted app-creation rate limits and cancellations remain in the logs; they were not scientific outcome retries. All project apps were stopped with zero tasks at the saved final boundary.

The conservative cumulative compute estimate was **$25.1955**, approximately **$21.5835** above the completed identification comparison. This includes interrupted and continuation apps at their configured resource rates. It is not a vendor invoice. The $100 cumulative limit and separate $10 reserve leave approximately $64.80 of compute allowance at this boundary.

## Next authorized step

The [conditional OLMo check](olmo-one-credit-development-plan.md), specified before these complete results, is now eligible. It tests the same semantic interface and gates with the planned second model's ordinary short-output mode. It does not repair this Qwen result or substitute for confirmatory replication. No concept extraction or motivational steering has yet been interpreted.

Evidence: [complete audit](one-credit-v2-validation-audit.json), [response diagnostics](one-credit-v2-response-diagnostics.json), [cost and shutdown](one-credit-v2-cost-and-shutdown.json), [execution amendment](one-credit-v2-execution-amendment.md), and [original run](../runs/modal-one-credit-v2-validation/report.md).
