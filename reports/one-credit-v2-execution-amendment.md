# One-credit-v2 execution amendment: partitioned continuation

October 6, 2026, during early validation execution. **Scientific prompts, assignments, inference settings, scoring and gates are unchanged.**

The first 22 persisted inference profiles averaged 10.719 seconds and 448.5 generated tokens. Only timing/token-count fields were inspected for this decision. Serial completion of 1,400 branches projects about 4.2 hours and cannot fit the two-hour aggregate deadline. The H100 did not deliver the expected serial speedup.

Stop the initial worker and export every durable terminal record. Preserve those records and the original 1,400-assignment manifest. Redistribute only unfinished assignments in deterministic round-robin order across at most four H100 workers, retaining the original absolute deadline. Every inference uses a fresh context and its original seed. Do not retry already-terminal answers or inspect correctness to choose a partition. Record all worker environments and integrity checks.

A controller reserves the combined worst-case compute before starting workers. At the configured H100/CPU/RAM rate of $0.00122976/s per worker, four workers for two hours would cost at most about $35.42 plus conservative startup headroom, within the remaining $100 cumulative allowance after the $10 storage/transfer reserve. Actual continuation time is shorter than two hours because the original deadline is retained. Other project launches remain blocked while the reserved workers are active. Reservations bind each exact child manifest and permit one client admission per child.

Each child contains an exact subset of the original assignments and the same model, source, task, configuration and hardware fields. Children do not independently qualify for advancement: the original minimum of 100 pairs per family/kind applies to the complete parent. Preserve child manifests and envelopes. Copy unmodified record payloads into derived parent envelopes for aggregation, retaining a provenance map of child manifest hashes and original record hashes. Check disjoint coverage, exact assignment identity and all integrity checks before evaluating the original full-sample gates. No completion based on partial results is allowed.

This replaces the earlier one-worker scheduling restriction for this validation only. Any future multiworker run requires its own bounded reservation; it does not increase the cumulative budget. Retain the interrupted run and all failed infrastructure attempts in the execution history. A killed in-flight call without a terminal record may be re-executed under the existing recovery rule; all durable completed answers remain fixed.

## Slow-worker continuation contingency

October 7, 2026, 07:14 UTC, before aggregate scoring. Progress logs show one of the four workers completing fewer branches per minute. If it remains unfinished when the other three have completed, stop that worker, export all durable terminal records, verify that every project app is stopped, and reserve up to three new workers for only its unfinished assignments. Retain the original absolute deadline, source, prompts, seeds, inference settings, and scoring. This decision uses completion timing only. New workers may have different host environments; record those rather than assuming bitwise reproducibility.

Aggregate the nested continuation into the original child first, then audit the full original parent. No completed answer is replaced, and neither a child nor a nested child can independently pass the original scientific gates. This contingency changes scheduling only and does not restart the two-hour clock.
