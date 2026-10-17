# Conditional OLMo one-credit development check

October 7, 2026, specified while Qwen one-credit-v2 validation is incomplete and before its aggregate scores are inspected. This is instrument development, not confirmatory replication or a replacement for the protocol's primary model.

## Entry and scope

Enter only if the complete Qwen v2 validation or its conditional calibration fails. The protocol already includes OLMo as its second research model. Use it to check whether the same explicit information-choice interface works under a different model's ordinary response mode. Preserve every Qwen result. Do not search additional labels, rewrite consequence descriptions, relax gates, or choose the model based on a desired resolution contrast.

Use `allenai/OLMo-2-1124-7B-Instruct` at revision `470b1fba1ae01581f270116362ee4aa1b97f4c84`, unquantized BF16, eager attention, L40S, and the existing inference loop. The tokenizer/configuration preflight is saved in `olmo-metadata-preflight.json`; it does not replace the real-model integrity audit. No Qwen-specific thinking delimiters are requested. Diagnostics use greedy decoding with an eight-token allowance. Conditional behavior uses the existing full-vocabulary temperature-0.7 sampling rule, also with eight tokens. Invalid responses are retained.

Keep the one-credit-v2 semantic interface: A–D candidate labels, A–C action IDs, and 1–4 consequence-answer IDs. Exactly one clue can be selected; the other remains unavailable, and declining both is allowed. Both task families, all diagnostic kinds, layouts, choice mappings, and gates remain unchanged. Freeze the exact new manifest and source archive before inference.

## Sample and decisions

Use seed 20261009 with 100 new pairs per family, producing 1,400 diagnostic branches. Require at least 90% strict correctness and 95% valid answers separately in every family/kind cell, with all original assignments terminal. Audit exact assignments, prompts, inference settings, saved payloads, and model integrity before interpreting the result.

Only if all comprehension gates pass, collect the original 1,200-choice calibration on the same development pairs, with source/model/hardware/task matching. Require at least 95% valid actions per family/condition/layout cell, at least 10% absolute probability for each mystery in both-unresolved trials in each layout, and no more than 15 percentage points of layout sensitivity for either mystery. Do not require a positive resolution effect. Retain stop and invalid outcomes in every assigned-trial denominator. Use the existing pair-cluster descriptive intervals if calibration is collected.

If this instrument also fails, summarize the limitations across both models before a high-level decision about broader instrument redesign. A pass supports further development on OLMo; it does not repair Qwen's result, demonstrate cross-model replication, establish willingness to pay, or identify curiosity, satisfaction, or subjective experience.

## Resources

Launch only after all current project apps have stopped. Use the shared $100 cumulative allowance with the $10 storage/transfer reserve. Start with one L40S worker under the existing bounded launcher; retain its original deadline and all saved answers if execution recovery is necessary. This short-output setting does not justify importing the long-trace H100 schedule automatically. No budget increase is implied.
