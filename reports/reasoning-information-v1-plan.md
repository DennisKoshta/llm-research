# Reasoning-enabled information-state mechanisms — prospective amendment

October 14, 2026. Both complete no-thinking equal-uncertainty validations failed. These failures remain unchanged. The 32B matched diagnostic nevertheless achieved 96/96 correct constrained judgments with thinking enabled. We therefore stop no-thinking interface refinement and study the reasoning-enabled computation. The active scientific target remains internal mechanisms relevant to information evaluation and acquisition; this amendment does not declare a curiosity regulator or experience-related result.

## Immediate feasibility collection

Qwen3-32B revision `9216db5781bf21249d130ec9da846c4624c16137`, unquantized BF16, eager attention, H200, fixed cohorts of 32, sampled thinking at temperature 0.7 and full vocabulary, maximum 2,048 generated tokens. Use the unchanged equal-uncertainty task and response grammar, one world per family crossed with four known/query combinations and five diagnostics: 40 branches, seed 2026101407. This is an engineering and development-feasibility screen, not full validation. Preserve all branches and failures with no outcome retries.

Before inference, require a real-model audit showing that read-only capture preserves generated tokens and initial label probabilities in separate engineering prompts. Verify actual GPU identity and retain memory/timing profiles. The feasibility screen permits full validation if all 40 branches complete, at least 38 are valid, at least 36 are correct, and each family × diagnostic has at least three of four correct. This small screen is explicitly not evidence of general competence. If it fails, inspect the failure before spending on a full collection.

## Trace measurements

At zero-based block indices 15, 31, 47 and 57, record residual activations at the factual-record boundary, the final prompt token, processed generated-token counts 32/128/512, the processed closing `</think>` token, and the activation immediately before sampling the first non-whitespace final-answer token (or EOS). The final-answer token itself is not included at that last position. The fixed-token measurements are not assumed to denote the same semantic operation in every trace. Missing positions remain explicitly missing, including early completion and token-cap failures. Do not restrict analyses to conveniently long traces or treat missing values as zeros.

The record-boundary measurement precedes the question in a causal transformer. Later positions can reflect the question, generated reasoning and anticipated answer; opposite question polarities and independent fact/count questions will help distinguish those possibilities. Reasoning text is an observable output, not assumed to be a faithful account of computation. Initial label probabilities before reasoning are not final-answer probabilities or a validated confidence measure.

All arrays, token positions, availability masks, prompts, raw reasoning, final responses, assignments and model/software metadata are retained and hashed. Whole cohorts are published atomically to avoid per-row storage overhead; every original response remains individually addressable. Capturing state is read-only. No psychological intervention is performed in this collection.

## Conditional full validation and mechanism development

If feasibility passes, freeze 100 fresh worlds per family (seed 2026101408), all four known/query combinations and all five diagnostics: 4,000 branches. Preserve the earlier full-validation criteria separately for every family × diagnostic × known/query cell: complete coverage, 100 distinct generated worlds, at least 90% correct and 95% valid. Partition only complete original cohorts, using the feasibility timing to bound simultaneous workers within the remaining allowance. Every child binds to the original parent assignment hash and source. No performance-based exclusions or retries are permitted.

Use the resulting trajectories for clearly labeled development analyses of when query-relative information becomes decodable and whether that representation generalizes across opposite answer polarities and task families. Freeze disjoint extraction/development/evaluation partitions before fitting. A causal follow-up must distinguish changing reasoning text, changing an answer representation, and changing information evaluation or acquisition; its precise intervention schedule and controls will be specified before outcomes are collected. The earlier no-thinking passive-capture and intervention code remains available but has not been used to bypass failed gates.

## Compute

The verified phase spend before this amendment is $3.6831312 of the fresh $300 allowance, leaving $296.3168688. [Modal pricing](https://modal.com/pricing), checked October 14, lists H200 GPU time at $0.001261/second. With the configured two CPU cores and 48 GiB RAM ceiling, the conservative guard uses $0.00139376/second (about $5.02/hour). [Modal's GPU documentation](https://modal.com/docs/guide/gpu) specifies 141 GB H200 memory. Larger batches are an intended efficiency benefit to be measured in the smoke, not an assumed achieved speedup. The initial collection has a 900-second aggregate deadline plus the guard's accounting margin; all jobs remain subject to the fresh $300 cap and explicit shutdown.

## Full-collection freeze after feasibility, before validation outcomes

The feasibility collection completed 40/40 valid and correct responses, with no caps or multiple closing delimiters. This passes the predefined screen. Its two underlying worlds are not a generalization test. The stopped-app phase estimate is now $4.20857872, leaving $295.79142128.

For the 4,000-branch validation, stably group assignments by diagnostic kind before observing outcomes, retaining randomized order within each kind. Divide the 125 original cohorts of 32 round-robin across eight children, then execute two waves of four H200 workers. Each wave has a 5,400-second common deadline plus the accounting margin; its maximum combined liability is approximately $30.77. Children bind to the exact parent manifest and retain original cohort membership. An isolated, hashed execution copy runs both waves; budget accounting continues to use the canonical project history. Development edits elsewhere cannot change the frozen execution.

Across all branches, world indices 0–49 are representation extraction and 50–99 are development. Later mechanism evaluation will use newly generated worlds and disjoint vocabulary, and its assignments will be frozen before intervention outcomes. There is no held-out mechanism test in this validation collection.

Record final decision-label logits and full-vocabulary probabilities at temperature 1 immediately before the first non-whitespace final token. These are descriptive output distributions, not validated confidence. If a completion contains multiple closing delimiters, retain its count and align the final captures with the last delimiter, matching the response scorer. Missing captures stay missing. A fixed-transcript causal follow-up will replay exact token prefixes, check unsteered distribution agreement against these recorded values, and separately test the closing-delimiter and final-decision positions. This conditional intervention does not by itself establish a mechanism in freely generated reasoning.
